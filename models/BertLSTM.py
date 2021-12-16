import json
import logging
import tensorflow as tf
import tensorflow_hub as hub
from data_loader import DataLoaderConfig
from models.HAN import BiLSTMAttentionConfig
from models.HAN import BiLSTMAttention

# Config file of the deep learning model
class BertLSTMModelConfig:
    def __init__(self, word_bert_config, sentence_attention_config,document_attention_config):

        self.word_bert_config = word_bert_config
        self.sentence_attention_config = sentence_attention_config
        self.document_attention_config = document_attention_config

    @staticmethod
    def load_from_json(json_path):
        """
        load the config parameters from HAN model config file
        :param json_path:
        :return: at each level, the hidden unit and output size of LSTM
        """
        with open(json_path) as f:
            d = json.load(f)
            logging.info("HBertModelConfig: {}".format(d))
        return BertLSTMModelConfig(
            word_bert_config=WordBertConfig(d["word_bert_config"]),
            sentence_attention_config=BiLSTMAttentionConfig(d["sentence_bi_lstm_attention"]),
            document_attention_config=BiLSTMAttentionConfig(d["document_bi_lstm_attention"])
        )

class WordBertConfig:
    def __init__(self, d):
        # TensorFlow hub url for loading BERT at word-level
        self.url = d["url"]
        self.trainable = d.get("trainable")
        if self.trainable is None:
            logging.info("!!!!!!!forget to set word-level bert trainable. Default: True")
            self.trainable = True

# Start to build Bert_LSTM

class BertLSTMModel(tf.keras.models.Model):
    def __init__(self, config: BertLSTMModelConfig, data_config: DataLoaderConfig):
        super(BertLSTMModel, self).__init__()
        self.config = config
        self.n_label = data_config.n_label

        self.bert = hub.KerasLayer(
            config.word_bert_config.url, trainable=config.word_bert_config.trainable)

        self.sentence_attention = BiLSTMAttention(
            self.config.sentence_attention_config
        )
        self.document_attention = BiLSTMAttention(
            self.config.document_attention_config
        )

        self.final_fully_connect = tf.keras.layers.Dense(units=data_config.n_label)

    def call(self, x, training=True):
        """

        :param x: shape [batch_size, n_doc, n_sent, n_word, n_embedding]
        :param training:
        :return:
        """

        ids, masks = x
        x_shape = tf.shape(ids)
        batch_size = x_shape[0]
        n_doc = x_shape[1]
        n_sent = x_shape[2]
        n_word = x_shape[3]

        # word-level bert
        ids = tf.reshape(ids, [batch_size*n_doc*n_sent, n_word])
        masks = tf.reshape(masks, [batch_size*n_doc*n_sent, n_word])
        input_dict = {
            "input_word_ids": ids,
            "input_mask": masks,
            "input_type_ids": tf.zeros([batch_size*n_doc*n_sent, n_word], dtype=tf.int32)
        }

        x = self.bert(inputs=input_dict,
                      training=training) # shape:  [batch_size* n_doc* n_sent, n_hidden]
        x = x["pooled_output"]
        # The encoder's outputs are the pooled_output to represents each input sequence as a whole

        # sentence-level LSTM-attention
        x = tf.reshape(x, [batch_size*n_doc, n_sent, -1])
        x = self.sentence_attention(x, training=training)
        # shape: [batch_size * n_doc, n_hidden]

        # document-level LSTM-attention
        x = tf.reshape(x, [batch_size, n_doc, -1])
        x = self.document_attention(x, training=training)
        # shape: [batch_size, n_hidden]

        logits = self.final_fully_connect(x) # [batch_size, n_label]
        return logits
