import logging
import os
from datetime import datetime


class PredictResultWriter:
    def __init__(self, result_path, label_list):
        """
        :param result_path: path to write the file
        :param label_list: task label as a list
        """
        result_path_without_ext, ext = os.path.splitext(result_path)
        self.result_file_dict = dict()
        for label in label_list:
            result_file_path = "{}_{}{}".format(result_path_without_ext, label, ext)
            self.result_file_dict[label] = open(result_file_path, "a")
            logging.info("Init result writer to path {} for label {}".format(
                result_file_path, label
            ))
        self.label_list = label_list


    def write(self, patient_ids, predict_p, true_label):
        """

        :param patient_ids: patient subject id in MIMIC-III
        :param predict_p:  predict probablity
        :param true_label:  gold standard label
        :return:
        """
        patient_ids = patient_ids.numpy()
        predict_p = predict_p.numpy()
        true_label = true_label.numpy()

        for (M, label) in enumerate(self.label_list):
            file_to_write = self.result_file_dict[label]
            label_predict_p = predict_p[:, M]
            true_single_label = true_label[:, M]

            for (N, pid) in enumerate(patient_ids):
                s = "{},{},{}\n".format(
                    pid, label_predict_p[N], true_single_label[N]
                )
                file_to_write.write(s)
                file_to_write.flush()

    def writer_sep(self):
        for value in self.result_file_dict.values():
            value.write("{}=================\n".format(datetime. now()))
            value.flush()

    def destroy(self):
        for value in self.result_file_dict.values():
            value.close()

