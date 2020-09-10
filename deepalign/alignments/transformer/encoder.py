import tensorflow as tf

from deepalign.alignments.transformer.positional_encoding import positional_encoding
from deepalign.alignments.transformer.multi_head_attention import MultiHeadAttention
from deepalign.alignments.transformer.point_wise_feed_forward_network import point_wise_feed_forward_network


class Encoder(tf.keras.layers.Layer):
    """
    Mostly taken from https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html
    """

    def __init__(self, num_layers, num_attributes, d_model, num_heads, dff, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(num_attributes, d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        attentions = []
        for i in range(self.num_layers):
            x, a = self.enc_layers[i](x, training, mask)
            attentions.append(a)

        return x, attentions  # (batch_size, input_seq_len, d_model)


class EncoderLayer(tf.keras.layers.Layer):
    """
    Mostly taken from https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html
    """

    def __init__(self, num_attributes, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.num_attributes = num_attributes

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # attn_output = []
        # attentions = []
        # for i, mha, d in zip(range(self.num_attributes), self.mha, self.dropout1): # (num_attributes, batch_size, input_seq_len, d_model)
        #     x_ = x[:, i::self.num_attributes]
        #     att_o, att = mha(x_, x_, x, mask[:, i::self.num_attributes])
        #
        #     attn_output.append(d(att_o))
        #     attentions.append(att)
        #
        # attn_output = tf.math.reduce_sum(attn_output, axis=0)

        attn_output, attentions = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attentions
