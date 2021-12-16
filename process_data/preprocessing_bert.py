import tensorflow as tf
import os
from tokenization import FullTokenizer
import sys


FEATURE_DESCRIPTION = {
    'subject_id': tf.io.FixedLenFeature([], tf.int64),
    'notes': tf.io.FixedLenFeature([], tf.string),
    'diagnosis': tf.io.FixedLenFeature([], tf.string),
}


NOTE_SPLIT = "<NOTE_SEP>".lower()
SENT_SPLIT = "<SENT_SEP>".lower()

BERT_VOCAB_PATH = "./bert_vocab/uncased_L-12_H-768_A-12/vocab.txt"


BERT_TOKENIZER = FullTokenizer(
    BERT_VOCAB_PATH,
    do_lower_case=True)


BERT_MAX_SEQ_LENGTH = 64
MAX_DOC_LENGTH_PER_PATIENT = 50
MAX_SENT_LENGTH_PER_DOC = 80

print("BERT_MAX_SEQ_LENGTH: {}".format(BERT_MAX_SEQ_LENGTH))
print("MAX_DOC_LENGTH_PER_PATIENT: {}".format(MAX_DOC_LENGTH_PER_PATIENT))
print("MAX_SENT_LENGTH_PER_DOC: {}".format(MAX_SENT_LENGTH_PER_DOC))


def get_pos_subjectid_set_from_file(filename):
    subjectid_set = set()
    with open(filename,'r') as f:
        for line in f:
            line = line.strip()
            line_int = int(line)
            subjectid_set.add(line_int)
    return subjectid_set

### feature that are useful in storing tf record: byte for note and diagnosis, int for patient id.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(subject_id, notes, diagnosis, ids, masks, ids_shape):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible data type.
    feature = {
        'subject_id': _int64_feature([subject_id]),
        'notes': _bytes_feature(notes.encode('ascii')),
        'diagnosis': _bytes_feature(diagnosis.encode('ascii')),
        'ids': _int64_feature(ids),
        'masks': _int64_feature(masks),
        'ids_shape': _int64_feature(ids_shape),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def add_bert_to_one_tfr(input_tfr_path, output_tfr_path):

    def _parse_raw_data(example_proto):
        return tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)

    def _prepare_notes(sample):
        notes = sample["notes"]
        # lower
        notes = tf.strings.lower(notes)
        # split
        # split by doc. Get [n_doc]
        notes_split_by_doc = tf.strings.split(notes, sep=NOTE_SPLIT)
        # the first N documents
        # notes_split_by_doc = notes_split_by_doc[:MAX_DOC_LENGTH_PER_PATIENT]

        # the last N documents
        notes_split_by_doc = notes_split_by_doc[-MAX_DOC_LENGTH_PER_PATIENT:]

        # split by sent. get [n_doc, n_sent]
        notes_split_by_sent = tf.strings.split(notes_split_by_doc, sep=SENT_SPLIT)
        notes_split_by_sent = notes_split_by_sent[:, :MAX_SENT_LENGTH_PER_DOC]

        notes_split_by_sent = notes_split_by_sent.to_tensor(default_value="")
        ids, masks = tf.py_function(func=_tf_tokenize, inp=[notes_split_by_sent], Tout=[tf.int32, tf.int32])
        sample["ids"] = ids
        sample["masks"] = masks

        return sample

    def _tf_tokenize(sentences):

        doc_ids = []
        doc_mask = []

        for i in sentences:
            sent_ids = []
            sent_mask = []
            for sent in i:
                tokens = BERT_TOKENIZER.tokenize(sent.numpy())
                if len(tokens) > (BERT_MAX_SEQ_LENGTH - 2):
                    tokens = tokens[:(BERT_MAX_SEQ_LENGTH - 2)]

                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                ids = BERT_TOKENIZER.convert_tokens_to_ids(tokens)

                mask = [1] * len(ids)

                while len(ids) < BERT_MAX_SEQ_LENGTH:
                    ids.append(0)
                    mask.append(0)

                ids = ids[:BERT_MAX_SEQ_LENGTH]
                mask = mask[:BERT_MAX_SEQ_LENGTH]

                sent_ids.append(ids)
                sent_mask.append(mask)
            #
            doc_ids.append(sent_ids)
            doc_mask.append(sent_mask)

        return doc_ids, doc_mask

    dataset = tf.data.TFRecordDataset(
        input_tfr_path
    ).map(_parse_raw_data).map(_prepare_notes)

    with tf.io.TFRecordWriter(output_tfr_path) as writer:

        for sample in dataset:

            subject_id = sample["subject_id"].numpy()
            notes = sample["notes"].numpy().decode('ascii')
            diagnosis = sample["diagnosis"].numpy().decode('ascii')
            ids = sample["ids"].numpy()
            ids_shape = list(ids.shape)
            ids = ids.flatten().tolist()
            masks = sample["masks"].numpy().flatten().tolist()
            example_str = serialize_example(subject_id, notes, diagnosis, ids, masks, ids_shape)
            writer.write(example_str)


def add_bert_to_tfr(input_tfr_dir, output_tfr_dir):
    """
    TODO
    for ...
        call extend_phenotype_to_one_tfr
    :param input_tfr_dir:
    :param output_tfr_dir:
    :return:
    """
    tf_dir_sets = ['training','dev','test']
    for tf_dir in tf_dir_sets:
        tf_folder = input_tfr_dir + tf_dir
        for filename in os.listdir(tf_folder):
            print("Processing: "+filename)
            if filename.endswith(".tfrecords"):
                input_full_filename = tf_folder + "/" +filename
                output_full_filename = output_tfr_dir + tf_dir +"/"+filename
                add_bert_to_one_tfr(input_full_filename, output_full_filename)

def main():
    """
    input csv: raw data
    output tf record directory: output tf record
    :return: write
    """
    input_tfr_dir = sys.argv[1]
    output_tfr_dir = sys.argv[2]

    add_bert_to_tfr(input_tfr_dir, output_tfr_dir)




if __name__ == '__main__':
    """
    python preprocessing_bert.py <input_tfr_dir> <output_tfr_dir>
    """
    main()
