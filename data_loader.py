import json
import glob
import os
import logging
import tensorflow as tf
import numpy as np
from tokenization import FullTokenizer

#TODO this is new bert.

# Get parameters from data loader config
class DataLoaderConfig:
    def __init__(self, training_dir, dev_dir, test_dir, label_file, batch_size,
                 note_split, sent_split, word_embedding_file,
                 unk_word, pad_word,
                 bert_vocab_path, bert_max_seq_length,
                 max_doc_len_per_patient,max_sent_len_per_doc,max_word_len_per_sent,
                 use_preprocessed_bert):
        self.training_files = glob.glob(os.path.join(training_dir, "*.tfrecords"))
        self.dev_files = glob.glob(os.path.join(dev_dir, "*.tfrecords"))
        self.test_files = glob.glob(os.path.join(test_dir, "*.tfrecords"))
        self.label_file = label_file
        self.batch_size = batch_size
        self.note_split = note_split.lower()
        self.sent_split = sent_split.lower()
        self.unk_word = unk_word
        self.pad_word = pad_word
        self.bert_vocab_path = bert_vocab_path
        self.bert_max_seq_length = bert_max_seq_length
        self.max_doc_len_per_patient = max_doc_len_per_patient
        self.max_sent_len_per_doc = max_sent_len_per_doc
        self.max_word_len_per_sent = max_word_len_per_sent
        self.label_list = self.get_label_list(label_file)
        self.n_label = len(self.label_list)
        self.use_preprocessed_bert = use_preprocessed_bert

        # when using w2v embedding
        if word_embedding_file:
            self.word_list, self.embedding_matrix, self.unk_index, self.pad_index = self._load_embedding(
                word_embedding_file)
        else:
            self.word_list, self.embedding_matrix, self.unk_index, self.pad_index = None, None, None, None


        logging.info("Size: Training - {}, Dev - {}, Test - {}".format(
            len(self.training_files), len(self.dev_files), len(self.test_files)
        ))


    @staticmethod
    def load_from_json(json_path):
        with open(json_path) as f:
            d = json.load(f)
            logging.info("DataLoaderConfig: {}".format(d))

        return DataLoaderConfig(
            training_dir=d["training_dir"],
            dev_dir=d["dev_dir"],
            test_dir=d["test_dir"],
            label_file=d["label_file"],
            batch_size=d["batch_size"],
            note_split=d["note_split"],
            sent_split=d["sent_split"],
            # when using w2v embedding
            word_embedding_file=d.get("word_embedding_file"),
            unk_word=d.get("unk_word"),
            pad_word=d.get("pad_word"),

            # when using bert
            bert_vocab_path=d.get("bert_vocab_path"),
            bert_max_seq_length=d.get("bert_max_seq_length"),
            use_preprocessed_bert=d.get("use_preprocessed_bert"),

            max_doc_len_per_patient = d["max_doc_len_per_patient"],
            max_sent_len_per_doc = d["max_sent_len_per_doc"],
            max_word_len_per_sent =d.get("max_word_len_per_sent")

        )

    @staticmethod
    def get_label_list(label_file):
        label_list = []
        with open(label_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    label_list.append(line)
        return label_list


    def _load_embedding(self, word_embedding_file):
        """
        Load an embedding file into a word list and an embedding matrix

        We also check unknown word and pad_word
        1. If pad word does not exist -> use zero padding
        2. If unk word does not exist -> use zero in matrix
        :param word_embedding_file:
        :return:
        """
        unk_index = -1
        pad_index = -1

        with open(word_embedding_file) as embedding_file:
            word_list = []
            embedding_list = []
            for (M, line) in enumerate(embedding_file):
                values = line.split()
                word = values[0]
                coef = np.asarray(values[1:], dtype='float32')
                word_list.append(word)
                embedding_list.append(coef)

                if word == self.unk_word:
                    unk_index = M
                if word == self.pad_word:
                    pad_index = M

        if unk_index == -1:
            logging.info("Do not have UNK word in embedding: {}".format(self.unk_word))
            unk_index = len(word_list)
            word_list.append(self.unk_word)
            embedding_list.append(np.zeros_like(coef))

        if pad_index == -1:
            logging.info("Do not have pad word in embedding: {}".format(self.pad_word))
            pad_index = len(word_list)
            word_list.append(self.pad_word)
            embedding_list.append(np.zeros_like(coef))

        return word_list, np.stack(embedding_list), unk_index, pad_index


class DataLoader:
    def __init__(self, data_loader_config: DataLoaderConfig):
        self.config = data_loader_config
        if self.config.use_preprocessed_bert:
            self.feature_description = {
                    'subject_id': tf.io.FixedLenFeature([], tf.int64),
                    'diagnosis': tf.io.FixedLenFeature([], tf.string),
                    'ids': tf.io.VarLenFeature(tf.int64),
                    'masks': tf.io.VarLenFeature(tf.int64),
                    'ids_shape': tf.io.FixedLenFeature([3], tf.int64),
                }
        else:
            self.feature_description = {
                    'subject_id': tf.io.FixedLenFeature([], tf.int64),
                    'notes': tf.io.FixedLenFeature([], tf.string),
                    'diagnosis': tf.io.FixedLenFeature([], tf.string),
                }

        self.label_lookup_table = tf.lookup.StaticHashTable(
            tf.lookup.TextFileInitializer(
                self.config.label_file,
                tf.string,
                tf.lookup.TextFileIndex.WHOLE_LINE,
                tf.int64,
                tf.lookup.TextFileIndex.LINE_NUMBER,
                delimiter=" "
            ),
            -1
        )
        logging.info("N Label: {}".format(self.label_lookup_table.size()))

        if self.config.word_list:
            self.word_lookup_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(self.config.word_list, dtype=tf.string),
                    tf.range(len(self.config.word_list))
                ),
                self.config.unk_index
            )
        else:
            self.word_lookup_table = None

        if self.config.bert_vocab_path:

            self.bert_tokenizer = FullTokenizer(
                data_loader_config.bert_vocab_path,
                do_lower_case=True)
        else:
            self.bert_tokenizer = None

    def _parse_raw_data(self, example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, self.feature_description)


    def _prepare_diagnosis(self, sample):
        diagnosis = sample["diagnosis"]
        # split
        diagnosis = tf.strings.split(diagnosis, sep=",")
        # lookup
        diagnosis = self.label_lookup_table.lookup(diagnosis)
        # one hot
        diagnosis = tf.one_hot(
            diagnosis,
            tf.cast(self.label_lookup_table.size(), tf.int32),
            dtype=tf.int32
        )
        diagnosis = tf.reduce_max(diagnosis, axis=0)
        sample["diagnosis"] = diagnosis
        return sample

    def _prepare_bert_ids_masks(self, sample):
        ids = sample["ids"]
        masks = sample["masks"]
        ids = tf.sparse.to_dense(ids)
        masks = tf.sparse.to_dense(masks)
        ids_shape = sample["ids_shape"]
        ids = tf.reshape(ids, ids_shape)
        masks = tf.reshape(masks, ids_shape)
        # the first n items for normal setting
        # ids = ids[:self.config.max_doc_len_per_patient, :self.config.max_sent_len_per_doc, :]
        # masks = masks[:self.config.max_doc_len_per_patient, :self.config.max_sent_len_per_doc, :]

        # the last n items for larger bert
        ids = ids[-self.config.max_doc_len_per_patient:, :self.config.max_sent_len_per_doc, :]
        masks = masks[-self.config.max_doc_len_per_patient:, :self.config.max_sent_len_per_doc, :]

        ids = tf.cast(ids, tf.int32)
        masks = tf.cast(masks, tf.int32)
        sample["ids"] = ids
        sample["masks"] = masks
        del sample["ids_shape"]
        return sample


    def _prepare_notes(self, sample):
        notes = sample["notes"]
        # lower
        notes = tf.strings.lower(notes)
        # split
        # split by doc. Get [n_doc]
        notes_split_by_doc = tf.strings.split(notes, sep=self.config.note_split)

        #the first n items for normal settings
        #notes_split_by_doc = notes_split_by_doc[:self.config.max_doc_len_per_patient]

        # the last n items for larger bert
        notes_split_by_doc = notes_split_by_doc[-self.config.max_doc_len_per_patient:]

        # split by sent. get [n_doc, n_sent]
        notes_split_by_sent = tf.strings.split(notes_split_by_doc, sep=self.config.sent_split)
        notes_split_by_sent = notes_split_by_sent[:, :self.config.max_sent_len_per_doc]

        if self.word_lookup_table:
            # split by words. Get [n_doc, n_sent, n_word]
            notes_split_by_words = tf.strings.split(notes_split_by_sent, sep=" ")
            notes_split_by_words = notes_split_by_words[:, :, :self.config.max_word_len_per_sent]
            # RaggedTensor to tensor
            notes_split_by_words = notes_split_by_words.to_tensor(default_value='PAD')

            # embedding
            notes_lookup_result = self.word_lookup_table.lookup(notes_split_by_words)
            sample["notes"] = notes_lookup_result

        if self.bert_tokenizer:
            notes_split_by_sent = notes_split_by_sent.to_tensor(default_value="")
            ids, masks = tf.py_function(func=self._tf_tokenize, inp=[notes_split_by_sent], Tout=[tf.int32, tf.int32])
            sample["ids"] = ids
            sample["masks"] = masks
            del sample["notes"]

        return sample



    def _tf_tokenize(self, sentences):

        doc_ids = []
        doc_mask = []

        for i in sentences:
            sent_ids = []
            sent_mask = []
            for sent in i:
                tokens = self.bert_tokenizer.tokenize(sent.numpy())
                if len(tokens) > (self.config.bert_max_seq_length - 2):
                    tokens = tokens[:(self.config.bert_max_seq_length - 2)]
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)

                mask = [1] * len(ids)

                while len(ids) < self.config.bert_max_seq_length:
                    ids.append(0)
                    mask.append(0)

                ids = ids[:self.config.bert_max_seq_length]
                mask = mask[:self.config.bert_max_seq_length]
                # ids = [0] * self.config.bert_max_seq_length
                # mask = [0] * self.config.bert_max_seq_length
                sent_ids.append(ids)
                sent_mask.append(mask)
            #
            doc_ids.append(sent_ids)
            doc_mask.append(sent_mask)

        return doc_ids, doc_mask


    def _flat_dataset(self, sample):
        return sample["notes"], sample["diagnosis"], sample["subject_id"]

    def _bert_flat_dataset(self, sample):
        return (sample["ids"], sample["masks"]), sample["diagnosis"], sample["subject_id"]


    def get_dataset(self, dataset_type: str):
        """
        Get one kind of dataset
        :param dataset_type:
        :return:
        x -> [n_batch, n_doc, n_sent, n_word, embed_size]
        y -> [n_batch, n_label]
        """
        if dataset_type not in ["training", "dev", "test"]:
            raise NameError("Invalid dataset type: {}".format(dataset_type))

        # load raw dataset
        if dataset_type == "training":
            file_list = self.config.training_files

        elif dataset_type == "dev":
            file_list = self.config.dev_files
        else:
            file_list = self.config.test_files

        dataset = tf.data.TFRecordDataset(
                file_list,
                buffer_size=1024*1024*50,
                num_parallel_reads=min(5, len(file_list))
            )

        if dataset_type == "training":
            dataset = dataset.shuffle(
                buffer_size=self.config.batch_size * 100,
                reshuffle_each_iteration=True
            )

        dataset = dataset.map(self._parse_raw_data)
        dataset = dataset.map(self._prepare_diagnosis)
        if self.config.use_preprocessed_bert:
            dataset = dataset.map(self._prepare_bert_ids_masks)
        else:
            dataset = dataset.map(self._prepare_notes)

        # when using bert
        if self.bert_tokenizer or self.config.use_preprocessed_bert:
            dataset = dataset.padded_batch(
                self.config.batch_size,
                padded_shapes={
                    'subject_id': [],
                    'ids': [-1, -1, -1],
                    'masks': [-1, -1, -1],
                    'diagnosis': [self.config.n_label],
                },
                padding_values={
                    'subject_id': tf.constant(0, dtype=tf.int64),
                    'ids': 0,
                    'masks': 0,
                    'diagnosis': 0,
                },
                drop_remainder=True
            )
            dataset = dataset.map(self._bert_flat_dataset)

        else: # when using w2v embedding
            dataset = dataset.padded_batch(
                self.config.batch_size,
                padded_shapes={
                    'subject_id': [],
                    'notes': [-1, -1, -1],
                    'diagnosis': [self.config.n_label],
                },
                padding_values={
                    'subject_id': tf.constant(0, dtype=tf.int64),
                    'notes': self.config.pad_index,
                    'diagnosis': 0,
                },
                drop_remainder=True
            )
            dataset = dataset.map(self._flat_dataset)

        return dataset
