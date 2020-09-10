import numpy as np
import tensorflow as tf


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    """
    Mostly taken from https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html
    """

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def get_angles(pos, i, d_model):
    """
    Mostly taken from https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html
    """

    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))

    return pos * angle_rates
