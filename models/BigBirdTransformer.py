import json
import logging
import tensorflow as tf
from bigbird.core.modeling import BertModel
from data_loader import DataLoaderConfig
from bigbird.core import utils
import numpy as np
# Config file of the deep learning model
class BigBirdTransformerModelConfig:
    def __init__(self, word_big_bird_config):

        self.word_big_bird_config = word_big_bird_config


    @staticmethod
    def load_from_json(json_path):
        """
        load the config parameters from model config file
        :param json_path:
        """
        with open(json_path) as f:
            d = json.load(f)
            logging.info("TransformerEncoderModelConfig: {}".format(d))
        return BigBirdTransformerModelConfig(
            word_big_bird_config=WordBigBirdConfig(d["word_big_bird_config"]))


class WordBigBirdConfig:
    def __init__(self, d):
        # TensorFlow hub url for loading BERT at word-level
        self.params_json = d["params"]


class BigBirdTransformerModel(tf.keras.models.Model):
    def __init__(self, config: BigBirdTransformerModelConfig, data_config: DataLoaderConfig):
        super(BigBirdTransformerModel, self).__init__()
        self.config = config
        self.n_label = data_config.n_label
        self.big_bird = BertModel(config.word_big_bird_config.params_json)

        self.final_fully_connect = tf.keras.layers.Dense(units=self.n_label, kernel_initializer=utils.create_initializer())
        self.dropout = tf.keras.layers.Dropout(0.4) # increase dropout


    def call(self, x, training=True):
        """

        :param x: shape [batch_size, n_doc, n_sent, n_word, n_embedding]
        :param training:
        :return:
        """

        ids = x
        x_shape = tf.shape(ids)
        #tf.Tensor([batch_size n_doc n_sent n_word], shape=(4,), dtype=int32)
        batch_size = x_shape[0]
        n_doc = x_shape[1]
        n_word = x_shape[2]


        # word-level bert
        ids = tf.reshape(ids, [batch_size*n_doc, n_word])

        _, x = self.big_bird(input_ids=ids, training=training) # dictionary, two keys: pooled_output, sequence_output


        # document-level encoder
        x = tf.reshape(x, [batch_size, -1, 768])  # shape:  [n_batch, n_doc, d_model]
        # x = self.document_transformer(x, training=training)  # [n_batch, d_model]   firstpool on doc level
        
        pooled = tf.reduce_mean(x, axis=1)  # mmax pooling on the second axis
        # x = self.pooled_output(x) # no transformer on document-level, only pooling on the second axis
        logits = self.final_fully_connect(pooled) # [n_batch, n_label]
        logits = self.dropout(logits, training=training)
        return logits

