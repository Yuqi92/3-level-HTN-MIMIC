import json
import logging
import tensorflow as tf
import tensorflow_hub as hub
from data_loader import DataLoaderConfig
import numpy as np
# Config file of the deep learning model
class BertTransformerModelConfig:
    def __init__(self, word_bert_config,
                 sentence_encoder_config,
                 document_encoder_config):

        self.word_bert_config = word_bert_config
        self.sentence_encoder_config = sentence_encoder_config
        self.document_encoder_config = document_encoder_config

    @staticmethod
    def load_from_json(json_path):
        """
        load the config parameters from model config file
        :param json_path:
        """
        with open(json_path) as f:
            d = json.load(f)
            logging.info("TransformerEncoderModelConfig: {}".format(d))
        return BertTransformerModelConfig(
            word_bert_config=WordBertConfig(d["word_bert_config"]),
            sentence_encoder_config=TransformerEncoderConfig(d["sentence_encoder"]),
            document_encoder_config=TransformerEncoderConfig(d["document_encoder"])
        )

class WordBertConfig:
    def __init__(self, d):
        # TensorFlow hub url for loading BERT at word-level
        self.url = d["url"]
        self.trainable = d.get("trainable")
        if self.trainable is None:
            logging.info("!!!!!!!!forget to set word-level bert trainable. Default: True")
            self.trainable = True


class TransformerEncoderConfig:
    def __init__(self, d):
        self.d_model = d["d_model"]
        self.num_heads = d["num_heads"]
        self.dff = d["dff"]
        self.dropout_rate = d.get("dropout_rate")
        if not self.dropout_rate:
            logging.info("=========!!!!!!!!forget to set dropout rate. Default: 0.1")
            self.dropout_rate = 0.1

        self.num_layers = d.get("num_layers")
        if not self.num_layers:
            logging.info("=========!!!!!!!!forget to set number of layers. Default: 2")
            self.num_layers = 2

        self.pooling_strategy = d.get("pooling_strategy")
        if not self.pooling_strategy:
            logging.info("=========!!!!!!!!forget to set pooling_strategy. Default: first")
            self.pooling_strategy = "first"

        self.apply_positional_encoding = d.get("apply_positional_encoding")
        if self.apply_positional_encoding == None:
            logging.info("!!!!!!!!forget to set apply_positional_encoding. Default: True")
            self.apply_positional_encoding = True



def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights



### multi-head-attention module
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


### Transformer Encoder module
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        #self.pooled_output = tf.keras.layers.Dense(d_model, activation=tf.tanh)


    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2

        # first_token_tensor = out2[:, 0, :]
        # here we employ the way that bert used:
        # "pool" the model by simply taking the hidden state corresponding to the first unit.
        # https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/modeling.py#L225
        # return self.pooled_output(first_token_tensor)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_layers, d_model, num_heads, dff, dropout_rate,
                 pooling_strategy, apply_positional_encoding, maximum_position_encoding=100):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.pooling_strategy = pooling_strategy
        self.apply_positional_encoding = apply_positional_encoding

        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)


        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.pooled_output = tf.keras.layers.Dense(d_model, activation=tf.tanh)

    def call(self, x, training):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # TODO positional encoding ablation study
        if self.apply_positional_encoding:
            x = x + self.pos_encoding[:, :seq_len, :]
        else:
            x = x

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask=None)   # shape : [n_batch, n_seq_len, d_model]

        if self.pooling_strategy == "first":
            pooling_tensor = x[:, 0, :]  # first pool on axis 1: [n_batch, d_model]  CLS
        elif self.pooling_strategy == "mean":
            pooling_tensor = tf.reduce_mean(x, axis=1)
        elif self.pooling_strategy == "max":
            pooling_tensor = tf.reduce_max(x, axis=1)
        elif self.pooling_strategy == "mean_max":
            pooling_tensor = tf.concat([tf.reduce_mean(x, axis=1),tf.reduce_max(x, axis=1)], axis=1)  # [n_batch, 2*d_model]

        else:
            pooling_tensor = x[:, -1, :]   # last SEP pooling

        return self.pooled_output(pooling_tensor)   # Dense: [n_batch, d_model]

class BertTransformerModel(tf.keras.models.Model):
    def __init__(self, config: BertTransformerModelConfig, data_config: DataLoaderConfig):
        super(BertTransformerModel, self).__init__()
        self.config = config
        self.n_label = data_config.n_label

        self.bert = hub.KerasLayer(
            config.word_bert_config.url, trainable=config.word_bert_config.trainable)

        self.sentence_transformer = Encoder(
            num_layers = self.config.sentence_encoder_config.num_layers,
            d_model = self.config.sentence_encoder_config.d_model,
            num_heads = self.config.sentence_encoder_config.num_heads,
            dff = self.config.sentence_encoder_config.dff,
            dropout_rate = self.config.sentence_encoder_config.dropout_rate,
            pooling_strategy = self.config.sentence_encoder_config.pooling_strategy,
            apply_positional_encoding = self.config.sentence_encoder_config.apply_positional_encoding,
            maximum_position_encoding = 100
        )
        self.document_transformer = Encoder(
            num_layers = self.config.document_encoder_config.num_layers,
            d_model = self.config.document_encoder_config.d_model,
            num_heads = self.config.document_encoder_config.num_heads,
            dff = self.config.document_encoder_config.dff,
            dropout_rate = self.config.document_encoder_config.dropout_rate,
            pooling_strategy = self.config.document_encoder_config.pooling_strategy,
            apply_positional_encoding = self.config.document_encoder_config.apply_positional_encoding,
            maximum_position_encoding = 100
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
        #tf.Tensor([batch_size n_doc n_sent n_word], shape=(4,), dtype=int32)
        batch_size = x_shape[0]
        n_doc = x_shape[1]
        n_sent = x_shape[2]
        n_word = x_shape[3]


        # word-level bert
        ids = tf.reshape(ids, [batch_size*n_doc*n_sent, n_word])
        masks = tf.reshape(masks, [batch_size*n_doc*n_sent, n_word])
        input_dict = {
            "input_word_ids":ids,
            "input_mask":masks,
            "input_type_ids":tf.zeros([batch_size*n_doc*n_sent, n_word], dtype=tf.int32)
        }
        # reference: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1  Basic Usage
        x = self.bert(inputs=input_dict,
                      training=training) # dictionary, two keys: pooled_output, sequence_output

        x = x["pooled_output"]    # shape:  [n_batch* n_doc* n_sent, d_model]

        # sentence-level encoder
        x = tf.reshape(x, [batch_size*n_doc, n_sent, -1])  # shape:  [n_batch*n_doc, n_sent, d_model]

        x = self.sentence_transformer(x, training=training)
        # shape: [n_batch*n_doc, d_model]  first-pool on sentence level


        # document-level encoder
        x = tf.reshape(x, [batch_size, n_doc, -1])  # shape:  [n_batch, n_doc, d_model]
        x = self.document_transformer(x, training=training)  # [n_batch, d_model]   firstpool on doc level

        logits = self.final_fully_connect(x) # [n_batch, n_label]
        return logits

