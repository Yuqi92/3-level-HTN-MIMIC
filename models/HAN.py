import json
import logging
import tensorflow as tf
from data_loader import DataLoaderConfig

# Config file of the deep learning model
class HANModelConfig:
    def __init__(self, word_attention_config, sentence_attention_config,
                 document_attention_config):
        """

        :param word_attention_config:
        :param sentence_attention_config:
        :param document_attention_config:
        """
        self.word_attention_config = word_attention_config
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
            logging.info("HANModelConfig: {}".format(d))
        return HANModelConfig(
            word_attention_config=BiLSTMAttentionConfig(d["word_bi_lstm_attention"]),
            sentence_attention_config=BiLSTMAttentionConfig(d["sentence_bi_lstm_attention"]),
            document_attention_config=BiLSTMAttentionConfig(d["document_bi_lstm_attention"])
        )

class BiLSTMAttentionConfig:
    def __init__(self, d):
        self.n_hidden = d["n_hidden"]
        self.attention_output_size = d["output_size"]
        self.use_attention = d.get("use_attention")
        if self.use_attention is None:
            logging.info("forget to set use_attention. Default: True")
            self.use_attention = True

# Start to build the HAN model
class HANModel(tf.keras.models.Model):
    def __init__(self, config: HANModelConfig, data_config: DataLoaderConfig):
        super(HANModel, self).__init__()
        self.config = config
        self.n_label = data_config.n_label

        self.word_embedding = tf.constant(
            data_config.embedding_matrix, dtype=tf.float32
        )

        self.word_attention = BiLSTMAttention(
            self.config.word_attention_config
        )
        self.sentence_attention = BiLSTMAttention(
            self.config.sentence_attention_config
        )
        self.document_attention = BiLSTMAttention(
            self.config.document_attention_config
        )

        self.final_fully_connect = tf.keras.layers.Dense(
            units=data_config.n_label)


    def call(self, x, training=True):
        """
        :param x: input shape [batch_size, n_doc, n_sent, n_word, n_embedding]
        :param training: default True
        :return:
        """
        x_shape = tf.shape(x)
        batch_size = x_shape[0]
        n_doc = x_shape[1]
        n_sent = x_shape[2]
        n_word = x_shape[3]

        # embedding layer
        x = tf.nn.embedding_lookup(
            self.word_embedding,
            x
        )  # shape: [batch_size, n_doc, n_sent, n_word, n_embedding]

        # word-level attention
        x = tf.reshape(x, [batch_size*n_doc*n_sent, n_word, -1])
        x = self.word_attention(x, training=training) # shape:  [batch_size, n_doc, n_sent, n_hidden]

        # sentence-level attention
        x = tf.reshape(x, [batch_size*n_doc, n_sent, -1])
        x = self.sentence_attention(x, training=training) # shape: [batch_size, n_doc, n_hidden]

        # document-level attention
        x = tf.reshape(x, [batch_size, n_doc, -1])
        x = self.document_attention(x, training=training)  # [batch_size, n_hidden]

        logits = self.final_fully_connect(x) # [batch_size, n_label]
        return logits

# build Bilstm-attention layer
class BiLSTMAttention(tf.keras.layers.Layer):
    """
    1. Bi-lstm.
    2. Attention
    """
    def __init__(self, config: BiLSTMAttentionConfig):
        super(BiLSTMAttention, self).__init__()
        self.config = config

        # tf 2.0 version of lstm layer
        self.rnn_layer = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                units=self.config.n_hidden,
                dropout=0.5,
                return_sequences=True
            )
        )
        if self.config.use_attention:
            # context attention layer
            self.context_attention = ContextAttention(
                output_size=self.config.attention_output_size)

    def call(self, x, training=True):
        """

        :param x: [n_batch, n_seq_length, hidden_size]
        :param training:
        :return: [n_batch, n_seq_length, hidden_size]
        """
        x = self.rnn_layer(x, training=training)
        if self.config.use_attention:
            x = self.context_attention(x, training=training)
        else:
            x = tf.reduce_max(x, axis=1)
        return x


# Context Attention layer from HAN paper
class ContextAttention(tf.keras.layers.Layer):
    def __init__(self, output_size):
        super(ContextAttention, self).__init__()
        self.output_size = output_size
        self.fully_connect = tf.keras.layers.Dense(
            output_size,
            activation=tf.tanh
        )

    def build(self, input_shape):
        self.attention_context_vector = self.add_weight(shape=(self.output_size, ),
                                                        trainable=True,
                                                        name="attention_context_vector"
                                                        )

    def call(self, inputs, training=True):
        # the code is from original HAN model attention component, this is the basic context attention
        input_projection = self.fully_connect(inputs)

        input_projection_with_context_vector = input_projection * self.attention_context_vector

        vector_attn = tf.reduce_sum(
            input_projection_with_context_vector,
            axis=2,
            keepdims=True
        )  # [batch_size * seq_length * 1]

        attention_weights = tf.nn.softmax(vector_attn, axis=1)  # [batch_size * seq_length * 1]

        weighted_projection = inputs * attention_weights

        outputs = tf.reduce_sum(weighted_projection, axis=1)

        return outputs

