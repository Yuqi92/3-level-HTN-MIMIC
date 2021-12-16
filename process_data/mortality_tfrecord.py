import pandas as pd
import os
import numpy as np
import tensorflow as tf
import sys
from itertools import islice
import re
import random
import math

NOTE_CSV = 'NOTEEVENTS.csv'
ADMISSIONS_CSV = 'ADMISSIONS.csv'
PATIENTS_CSV = 'PATIENTS.csv'


SPLIT_HR = 2
SPLIT_TIMESTAMP_SECOND = SPLIT_HR * 3600

SENTENCE_SEP = "<SENT_SEP>"
NOTE_SEP = "<NOTE_SEP>"


TFR_N_PATIENT = 3088

def get_age(row):
    """Calculate the age of patient by row
    Arg:
        row: the row of pandas dataframe.
        return the patient age of year
    """
    raw_age = row['ADMITTIME'].year - row['DOB'].year
    if (row['ADMITTIME'].month < row['DOB'].month) or ((row['ADMITTIME'].month == row['DOB'].month) and (row['ADMITTIME'].day < row['DOB'].day)):
        return raw_age - 1
    else:
        return raw_age

def filter_patient_admission(patient, admission):
    """
    1. keep adult
    2. keep one admission
    """
    merged = patient.merge(admission, how='inner', on='SUBJECT_ID')
    merged['ADMITTIME'] = pd.to_datetime(merged['ADMITTIME'])
    merged['DISCHTIME'] = pd.to_datetime(merged['DISCHTIME'])
    merged['DOB'] = pd.to_datetime(merged['DOB'])

    merged['age'] = merged.apply(get_age, axis=1)
    merged_cut = merged.groupby('SUBJECT_ID')[['HADM_ID','age','ADMITTIME','DISCHTIME']].agg(list).reset_index()
    merged_cut["min_age"] = merged_cut.age.apply(lambda x: min(x))
    merged_adult = merged_cut[merged_cut["min_age"] >= 18]

    merged_adult["admit_times"] = merged_adult["HADM_ID"].apply(lambda x: len(x))
    one_admission_adult = merged_adult[merged_adult['admit_times'] == 1]

    return merged, one_admission_adult


def filter_mortality(merged_onetimeadult):

    cut = merged_onetimeadult[["SUBJECT_ID","DISCHTIME","DOD","DEATHTIME","HOSPITAL_EXPIRE_FLAG"]]

    cut['DISCHTIME'] = pd.to_datetime(cut['DISCHTIME'])
    cut['DOD'] = pd.to_datetime(cut['DOD'])
    cut['DEATHTIME'] = pd.to_datetime(cut['DEATHTIME'])

    cut['dead_date'] = cut['DOD'] - cut['DISCHTIME']
    cut['dead_date'] = cut['dead_date'].dt.days

    def regenerate_dead_date(c):
        if c['HOSPITAL_EXPIRE_FLAG'] == 1:
            return -1.0
        else:
            return c['dead_date']
    cut['dead_after_disch_date'] = cut.apply(regenerate_dead_date,axis=1)

    def get_mortality_label(row):
        if row["dead_after_disch_date"] <= 1:
            return "in-hospital"
        if 1 < row["dead_after_disch_date"] <= 31:
            return "30-day"
        if 32 <= row["dead_after_disch_date"] <= 365:
            return "1-year"
        else:
            return "alive"

    cut['mortality_label'] = cut.apply(get_mortality_label, axis=1)

    patient_label = cut.groupby(['SUBJECT_ID'])['mortality_label'].apply(lambda x:list(set(x))).reset_index()

    patient_dic = dict()
    for index, row in patient_label.iterrows():
        patient_dic[row['SUBJECT_ID']] = ",".join(row['mortality_label'])

    return patient_dic


def filter_note(df_note, patient_id_list, merged_onetimeadult):
    # remove error note and discharge summary
    df_note = df_note[(df_note['ISERROR'] != 1) & (df_note['CATEGORY'] != "Discharge summary")]

    note_adult = df_note[df_note['SUBJECT_ID'].isin(patient_id_list)]

    note_adult['CHARTTIME'] = pd.to_datetime(note_adult['CHARTTIME'])
    note_adult['CHARTDATE'] = pd.to_datetime(note_adult['CHARTDATE'])
    note_adult.CHARTTIME = note_adult.CHARTTIME.combine_first(note_adult.CHARTDATE)
    note_adult_cut = note_adult[["SUBJECT_ID","HADM_ID","CHARTTIME","TEXT"]]

    # to remove notes that charttime after deathtime/DOD
    merged_onetimeadult['DEATHTIME'] = pd.to_datetime(merged_onetimeadult['DEATHTIME'])
    merged_onetimeadult['DOD'] = pd.to_datetime(merged_onetimeadult['DOD'])

    merged_onetimeadult_cut = merged_onetimeadult[["SUBJECT_ID","DISCHTIME","DOD","DEATHTIME"]]
    merged_onetimeadult_cut.DEATHTIME = merged_onetimeadult_cut.DEATHTIME.combine_first(merged_onetimeadult_cut.DOD)
    merged_onetimeadult_cut_tomerge = merged_onetimeadult_cut[["SUBJECT_ID","DISCHTIME","DEATHTIME"]]

    merge_note = pd.merge(note_adult_cut, merged_onetimeadult_cut_tomerge, on='SUBJECT_ID', how='left')
    merge_note_sort = merge_note.sort_values(by=['SUBJECT_ID','CHARTTIME','DEATHTIME'])

    # either the patient is not dead , keep all note. or the patient is dead, remove all notes with charttime after deathtime
    note_chartbeforedeath = merge_note_sort[(merge_note_sort["DEATHTIME"].isnull())|((merge_note_sort["DEATHTIME"].notnull()) & (merge_note_sort["CHARTTIME"] < merge_note_sort["DEATHTIME"]))]

    patient_note_save = note_chartbeforedeath[['SUBJECT_ID','CHARTTIME','TEXT']]
    patient_note_save_reset_index = patient_note_save.reset_index()

    final_note_patient = patient_note_save_reset_index[['SUBJECT_ID','CHARTTIME','TEXT']]
    final_note_patient['chartunix_timestamp'] = final_note_patient['CHARTTIME'].values.astype(np.int64) // (10 ** 9)

    return final_note_patient


def load_csv(input_csv_path):
    """
    pre-process dataframe of MIMIC-III
    :param input_csv_path: csv data path
    :return: diagnosis dictionary and note dataframe
    """
    # get csv file path
    mimic_note_events = os.path.join(input_csv_path, NOTE_CSV)
    mimic_admissions = os.path.join(input_csv_path, ADMISSIONS_CSV)
    mimic_patients = os.path.join(input_csv_path, PATIENTS_CSV)

    # load to pandas dataframe
    df_note = pd.read_csv(mimic_note_events)
    admission = pd.read_csv(mimic_admissions) # admission table
    patient = pd.read_csv(mimic_patients)

    merged, one_admission_adult = filter_patient_admission(patient, admission)
    patient_id_list = list(set(one_admission_adult['SUBJECT_ID']))

    merged_onetimeadult = merged[merged["SUBJECT_ID"].isin(patient_id_list)]

    mortality_dic = filter_mortality(merged_onetimeadult)
    df_note = filter_note(df_note, patient_id_list, merged_onetimeadult)

    return mortality_dic, df_note

def split_into_words_and_join(text):
    """
    regular expression to normalize the clinical text
    :param text: note input
    :return: a joined string. Sentences are separated by SENT_SEPARATOR. Word are separated by space.
    """
    text = text.replace("\n"," ")
    text = re.sub('(\[\*\*.*?\*\*\])',"PHI", text)
    text = re.sub('(\d{2}\:\d{2}\s?(?:|AM|PM|am|pm))',"time", text)
    text = re.sub('\d+\.\d+','float', text)
    text = re.sub("\w+\/\w+", 'unit', text)
    text = re.sub('_', '', text)
    text = re.sub(':', '', text)
    text = re.sub("\(", '', text)
    text = re.sub("\)", '', text)
    text = text.replace("\n","\n<stop>")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    sentences = text.split("<stop>")

    sentences = [" ".join(s.split()) for s in sentences]

    out_sentences = []
    for s in sentences:
        if s:
            out_sentences.append(s)

    return SENTENCE_SEP.join(out_sentences)

def adaptive_segmentation(list_of_timestamp, max_time_span):

    # For example
    # given list_of_timestamp = [1000, 2000, 2500, 3000, 3100, 5000]
    # if max_time_span = 10:
    # return [[1000], [2000], [2500], [3000], [3100], [5000]], [1, 1, 1, 1, 1, 1]

    # if max_time_span = 100:
    # return [[1000], [2000], [2500], [3000, 3100], [5000]], [1, 1, 1, 2, 1]

    # if max_time_span = 1000:
    # return [[1000], [2000, 2500], [3000, 3100], [5000]], [1, 2, 2, 1]

    # if max_time_span = 2000:
    # return [[1000], [2000, 2500, 3000, 3100], [5000]], [1, 4, 1]

    # if max_time_span = 4000:
    # return [[1000, 2000, 2500, 3000, 3100, 5000]], [6]
    # NOTE: This is a slow implementation O(L*T).
    # We should use binary-tree to achieve better performance O(L * log(T))

    """
    the items within the max time span will be grouped together
    :param list_of_timestamp: the list of timestamp
    :param max_time_span: the maximum time span to split the time
    :return: a list of the grouped list within max time stamp
    & a list of the number of items in each group
    """

    output_list = []
    for i in list_of_timestamp:
        output_list.append([i])
    while True:
        output_list, merged = _merge_output_list(output_list, max_time_span)
        if not merged:
            break

    output_list_length = [len(i) for i in output_list]
    return output_list, output_list_length

def _merge_output_list(current_output_list, max_time_span):
    """
    :param current_output_list:time stamp list
    :param max_time_span: user-defined max time span
    :return: new_output_list, whether_merged
    """
    if len(current_output_list) <= 1:
        return current_output_list, False
    time_span_list = []
    for i in range(len(current_output_list) - 1):
        this_output = current_output_list[i]
        next_output = current_output_list[i+1]
        this_start_int = this_output[0]
        next_end_int = next_output[-1]
        time_span_list.append(next_end_int - this_start_int)

    min_span = None
    min_span_index = None
    for (M, i) in enumerate(time_span_list):
        if (min_span is None) or (i < min_span):
            min_span_index = M
            min_span = i
    # cannot merge
    if min_span > max_time_span:
        return current_output_list, False
    # we can merge
    new_output_list = current_output_list[:min_span_index]
    new_output_list.append(current_output_list[min_span_index] + current_output_list[min_span_index+1])
    new_output_list += current_output_list[(min_span_index+2):]

    return new_output_list, True

## start to save tf records of raw data: three features: subject id, diagnosis label, clinical notes

### feature that are useful in storing tf record: byte for note and diagnosis, int for patient id.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(subject_id, notes, diagnosis):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible data type.
    feature = {
        'subject_id': _int64_feature(subject_id),
        'notes': _bytes_feature(notes.encode('ascii')),
        'diagnosis': _bytes_feature(diagnosis.encode('ascii')),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# official writing

def write_tfr(final_note_patient, diagnosis_dic, output_tfr_dir):
    """
    Write all the tf records
    :param final_note_patient: note data
    :param diagnosis_dic: diagnosis dictionary
    :param output_tfr_dir: output tf record directory
    :return: no return, just writing
    """

    # total note processing, group by patient
    list_chartunix_timestamp = final_note_patient.groupby('SUBJECT_ID')['chartunix_timestamp'].apply(list)
    list_note = final_note_patient.groupby('SUBJECT_ID')['TEXT'].apply(list)

    subject_id_list = list(set(final_note_patient['SUBJECT_ID']))

    # shuffle samples before saving
    random.shuffle(subject_id_list)

    subject_id_list_length = len(subject_id_list)

    print("subject_id_list_length: {}".format(subject_id_list_length))

    # how many records to be saved
    # common practice: into 10n files for easily splitting into train val and test

    n_tfr = int(math.ceil(subject_id_list_length / TFR_N_PATIENT))

    global_total_count = 0
    global_note_count = 0

    for i_tfr in range(n_tfr):
        current_subject_id_list = subject_id_list[
            i_tfr * TFR_N_PATIENT : min((i_tfr + 1) * TFR_N_PATIENT, subject_id_list_length)
        ]
        tfr_path = os.path.join(output_tfr_dir, "data-{}.tfrecords".format(i_tfr))

        patient_count, note_count = write_single_tfr(
            current_subject_id_list,
            diagnosis_dic,
            list_chartunix_timestamp,
            list_note,
            tfr_path
        )
        global_total_count += patient_count
        global_note_count += note_count
    print("Globally, get {} samples".format(global_total_count))
    print("Globally, get {} notes".format(global_note_count))


def write_single_tfr(subject_id_list, diagnosis_dic, list_chartunix_timestamp, list_note, tfr_path):
    """
    write single tf record
    :param subject_id_list: subject id from note data
    :param diagnosis_dic: diagnosis dictionary
    :param list_chartunix_timestamp: list of timestamp by patient
    :param list_note: list of note by patient
    :param tfr_path: tf record output path
    :return: the count of subject id into the record
    """
    print("Start to work on tfr {}".format(tfr_path))
    total_count = 0
    note_count = 0
    with tf.io.TFRecordWriter(tfr_path) as writer:
        for subject_id in subject_id_list:
            # each patient saving:

            if subject_id not in diagnosis_dic:
                # the subject id that only has note, no diagnosis
                print("subject_id {} not in diagnosis".format(subject_id))
                continue

            chartunix_timestamp_subjectid = list_chartunix_timestamp[subject_id] # the list of timestamp of this patient
            note_subjectid = list_note[subject_id] # the list of note of this patient

            # group the note according to timestamp, output the length of each group
            _, length = adaptive_segmentation(chartunix_timestamp_subjectid, SPLIT_TIMESTAMP_SECOND)

            it = iter(note_subjectid)
            sliced_note =[list(islice(it, 0, i)) for i in length]

            notes = []
            for sliced_note_list in sliced_note:
                note = []  # list of list of words (list of sentences)
                for note_i in sliced_note_list:
                    out_note = split_into_words_and_join(note_i)
                    if out_note: # have note
                        note.append(out_note)
                note_str = SENTENCE_SEP.join(note)
                if note_str: # have string note
                    notes.append(note_str)

            if not notes: # not empty
                continue

            # write tf record
            note_count += len(notes)

            notes_string = NOTE_SEP.join(notes)  # get string representation of notes for one patient
            diagnosis = diagnosis_dic.get(subject_id)  # get string representation of diagnosis for each patient

            # three features to be stored
            example_str = serialize_example(subject_id, notes_string, diagnosis)
            writer.write(example_str)
            total_count += 1

    print("Finish tfr. Get {} samples".format(total_count))
    print("Finish tfr. Get {} notes".format(note_count))

    return total_count, note_count


def main():
    """
    input csv: raw data
    output tf record directory: output tf record
    :return: write
    """
    input_csv_dir = sys.argv[1]
    output_tfr_dir = sys.argv[2]

    diagnosis_dic, df_note = load_csv(input_csv_dir)
    write_tfr(df_note, diagnosis_dic, output_tfr_dir)


if __name__ == '__main__':
    """
    python mortality_tfrecord.py <input_csv_dir> <output_tfr_dir>
    """
    main()



