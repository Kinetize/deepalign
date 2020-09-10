import tensorflow as tf
import numpy as np

from deepalign import Dataset
from deepalign.anomalydetection import AnomalyDetectionResult
from deepalign.anomalydetection.binet.core import binet_scores_fn
from deepalign.anomalydetection.binet.attention import Attention, BahdanauAttention
from deepalign.enums import Base, Heuristic, Mode, Strategy, AttributeType, FeatureType

import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# tf.config.experimental_run_functions_eagerly(True)
class BINetX(tf.keras.Model):
    supported_bases = [Base.LEGACY, Base.SCORES]
    supported_heuristics = [Heuristic.BEST, Heuristic.ELBOW_DOWN, Heuristic.ELBOW_UP,
                            Heuristic.LP_LEFT, Heuristic.LP_MEAN, Heuristic.LP_RIGHT,
                            Heuristic.MEAN, Heuristic.MEDIAN, Heuristic.RATIO, Heuristic.MANUAL]
    supported_strategies = [Strategy.SINGLE, Strategy.ATTRIBUTE, Strategy.POSITION, Strategy.POSITION_ATTRIBUTE]
    supported_modes = [Mode.BINARIZE, Mode.CLASSIFY]
    supports_attributes = True
    config = None
    version = None

    loaded = False

    def __init__(self,
                 dataset,
                 latent_dim=None,
                 use_case_attributes=None,
                 use_event_attributes=None,
                 use_present_activity=None,
                 use_present_attributes=None,
                 use_attention=None):
        super(BINetX, self).__init__()

        # Validate parameters
        if latent_dim is None:
            latent_dim = min(int(dataset.max_len * 10), 16)  # TODO: 64
        if use_event_attributes and dataset.num_attributes == 1:
            use_event_attributes = False
            use_case_attributes = False
        if use_present_activity and dataset.num_attributes == 1:
            use_present_activity = False
        if use_present_attributes and dataset.num_attributes == 1:
            use_present_attributes = False

        # Parameters
        self.latent_dim = latent_dim
        self.use_case_attributes = use_case_attributes
        self.use_event_attributes = use_event_attributes
        self.use_present_activity = use_present_activity
        self.use_present_attributes = use_present_attributes
        self.use_attention = use_attention

        # Single layers
        self.fc = None
        if self.use_case_attributes:
            self.fc = tf.keras.Sequential([
                tf.keras.layers.Dense(latent_dim // 8),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(latent_dim, activation='linear')
            ])

        # self.rnn = tf.keras.layers.GRU(latent_dim, return_sequences=True, return_state=True)
        rnn_word = tf.keras.layers.GRU(latent_dim, return_sequences=True)
        self.rnn_word = tf.keras.layers.TimeDistributed(tf.keras.layers.TimeDistributed(rnn_word))
        rnn_sentence = tf.keras.layers.GRU(latent_dim, return_sequences=True)
        self.rnn_sentence = tf.keras.layers.TimeDistributed(rnn_sentence)
        self.rnn_paragraph = tf.keras.layers.GRU(latent_dim * 2, return_sequences=True)

        # Layer lists
        self.fc_inputs = []
        self.rnn_inputs = []
        # self.attentions = []
        self.outs = []

        emb_dim = max([np.clip(int(dim + 1) // 10, 2, 10) for dim in dataset.attribute_dims])
        inputs = zip(dataset.attribute_dims, dataset.attribute_keys, dataset.attribute_types, dataset.feature_types)
        for dim, key, t, feature_type in inputs:
            if t == AttributeType.CATEGORICAL:
                voc_size = int(dim + 1)  # we start at 1, 0 is padding
                # emb_dim = np.clip(voc_size // 10, 2, 10)
                embed = tf.keras.layers.Embedding(input_dim=voc_size, output_dim=emb_dim, mask_zero=True)
                # , input_shape=(None, dataset.max_len, voc_size)
            else:
                embed = tf.keras.layers.Dense(1, activation='linear')
            #     embed = tf.keras.layers.TimeDistributed(
            #         tf.keras.layers.Embedding(input_dim=voc_size, output_dim=emb_dim, mask_zero=True))
            #     # , input_shape=(None, dataset.max_len, voc_size)
            # else:
            #     embed = tf.keras.layers.TimeDistributed(
            #         tf.keras.layers.Dense(1, activation='linear'))

            if feature_type == FeatureType.CASE:
                self.fc_inputs.append(embed)
            else:
                self.rnn_inputs.append(embed)

                # if self.use_attention:
                #     self.attentions.append(BahdanauAttention(16, name=f'attention_{key}'))  # TODO: TUNE #UNITS!
                #     self.attentions.append(Attention(return_sequences=True,
                #                                      return_coefficients=True,
                #                                      name=f'attention_{key}'))

                outs = [tf.keras.layers.Dense(dim + 1, activation='softmax') for _ in range(8)]
                self.outs.append(outs)

        if self.use_attention:
            attention_word = Attention(return_sequences=True,
                                       return_coefficients=True,
                                       name=f'attention_word')
            self.attention_word = tf.keras.layers.TimeDistributed(attention_word)

            attention_sentence = Attention(return_sequences=True,
                                           return_coefficients=True,
                                           name=f'attention_sentence')
            self.attention_sentence = tf.keras.layers.TimeDistributed(attention_sentence)

            self.attention_paragraph = Attention(return_sequences=True,
                                                 return_coefficients=True,
                                                 name=f'attention_paragraph')

        self.dataset = dataset

    def call(self, inputs, training=False, return_attention_weights=False, return_state=False, initial_state=None):
        if not isinstance(inputs, list):
            inputs = [inputs]

        split = len(self.rnn_inputs)

        rnn_x = inputs[:split]
        fc_x = inputs[split:]

        fc_embeddings = []
        for x, input_layer in zip(fc_x, self.fc_inputs):
            if isinstance(input_layer, tf.keras.layers.Dense):
                x = x[:, None]
            x = input_layer(x)
            fc_embeddings.append(x)

        if len(fc_embeddings) > 0:
            if len(fc_embeddings) > 1:
                fc_embeddings = tf.concat(fc_embeddings, axis=-1)
            else:
                fc_embeddings = fc_embeddings[0]

        fc_output = None
        if not isinstance(fc_embeddings, list):
            fc_output = self.fc(fc_embeddings)

        rnn_inputs = []
        for x, input_layer in zip(rnn_x, self.rnn_inputs):
            rnn_inputs.append(input_layer(x))
            # Attention(return_sequences=True,
            #           return_coefficients=True,
            #           name=f'attention_test')(attention_inputs[0])

        # Masks are the same for all inputs
        rnn_mask = rnn_inputs[0]._keras_mask

        rnn_inputs = tf.transpose(rnn_inputs, [1, 2, 3, 0, 4])
        attention_inputs = self.rnn_word(rnn_inputs)

        word_attentions = []
        word_attention_weights = []
        for i in range(attention_inputs.shape[1]):
            w_a, w_a_w = self.attention_word(attention_inputs[:, i])
            word_attentions.append(w_a)
            word_attention_weights.append(w_a_w)
        # word_attentions, word_attention_weights = self.attention_word(attention_inputs)

        word_attentions = tf.stack(word_attentions, axis=1)
        word_attention_weights = tf.stack(word_attention_weights, axis=1)

        sentence_input = tf.reshape(word_attentions,
                                     [*word_attentions.shape[:3], word_attentions.shape[-2] * word_attentions.shape[-1]])

        sentence_o = self.rnn_sentence(sentence_input)  # TODO: RE-ADD mask
        sentence_attentions, sentence_attention_weights = self.attention_sentence(sentence_o)

        paragraph_input = tf.reshape(sentence_attentions,
                                     [*sentence_attentions.shape[:2],
                                      sentence_attentions.shape[-2] * sentence_attentions.shape[-1]])

        paragraph_o = self.rnn_paragraph(paragraph_input) # TODO: Mask to identify cases which are completly padding!
        paragraph_attentions, paragraph_attention_weights = self.attention_paragraph(paragraph_o)

        # rnn_inputs = tf.transpose(rnn_inputs, [1, 2, 0, 3])
        # attention_inputs = self.rnn_word(rnn_inputs)
        # # attention_inputs = self.mask_word(attention_inputs)
        # word_attentions, word_attention_weights = self.attention_word(attention_inputs)
        #
        # word_attentions = tf.reshape(word_attentions,
        #                              [*word_attentions.shape[:2], word_attentions.shape[2] * word_attentions.shape[3]])
        #
        # sentence_o = self.rnn_sentence(word_attentions, mask=rnn_mask)
        # sentence_attentions, sentence_attention_weights = self.attention_sentence(sentence_o)

        # if initial_state is not None:
        #     rnn, h = self.rnn(word_attentions, initial_state=initial_state)
        # elif fc_output is not None:
        #     if len(fc_output.shape) == 3:
        #         fc_output = fc_output[:, 0]
        #     rnn, h = self.rnn(word_attentions, initial_state=fc_output)
        # else:
        #     rnn, h = self.rnn(word_attentions)

        # TODO USE
        # paragraph_attentions
        # here!

        # TODO: Maybe use decoding rnn to get sequence again? how do original HANs do that?

        outputs = []
        reshaped_mask = tf.reshape(rnn_mask, [rnn_mask.shape[0], rnn_mask.shape[1] * 8])
        for i, outs in enumerate(self.outs):
            x = paragraph_attentions

            # TODO: Reuse present attributes
            # if i > 0:
                # if self.use_present_attributes:
                #     # input_start = sum([self.rnn_inputs[j].output_shape[-1] for j in range(i)])
                #     # input_start = sum([self.rnn_inputs[j].output_dim for j in range(i)])
                #     # TODO: MAYBE USE "FUTURE" ATTRIBUTES AGAIN!
                #     present_attributes_before = tf.reshape(rnn_inputs[:, :, :, :i],
                #                                            (*rnn_inputs.shape[:3], i * rnn_inputs.shape[-1]))
                #     present_attributes_after = tf.reshape(rnn_inputs[:, :, :, i + 1:],
                #                                           (*rnn_inputs.shape[:3],
                #                                            (self.dataset.num_attributes - i - 1) * rnn_inputs.shape[
                #                                                -1]))
                #     x = tf.concat([x,
                #                    tf.pad(present_attributes_before[:, 1:x.shape[1]], [(0, 0), (0, 0), (0, 1), (0, 0)],
                #                           'constant', 0),
                #                    tf.pad(present_attributes_after[:, 1:x.shape[1]], [(0, 0), (0, 0), (0, 1), (0, 0)],
                #                           'constant', 0)], axis=-1)
                #     # tf.pad(
                #     #     rnn_inputs[:, 1:x.shape[1], input_start + self.rnn_inputs[i].output_dim:],
                #     #     [(0, 0), (0, 1), (0, 0)], 'constant', 0)],  # output_shape[-1]
                #
                # elif self.use_present_activity:
                #     present_activity = tf.reshape(rnn_inputs[:, :, :1, :],
                #                                   (*rnn_inputs.shape[:2], rnn_inputs.shape[-1]))
                #     x = tf.concat([x, tf.pad(present_activity[:, 1:x.shape[1]], [(0, 0), (0, 1), (0, 0)],
                #                              'constant', 0)], axis=-1)

            # Do we need to use time distributed here? No, we want to use the Attention of all timesteps!
            outputs.append([])
            for out in outs:
                x = out(x)
                outputs[i].append(x)

            outputs[i] = paragraph_to_sentence_level(outputs[i], reshaped_mask)

        word_attention_weights = paragraph_to_sentence_level(word_attention_weights, reshaped_mask)
        sentence_attention_weights = paragraph_to_sentence_level(sentence_attention_weights, reshaped_mask)

        # max_sentence_length = 8
        # outputs_permute = []
        # for i in range(max_sentence_length):
        #     outputs_permute.append(outputs[i])
        #     outputs_permute.append(outputs[i + max_sentence_length])
        #     outputs_permute.append(outputs[i + max_sentence_length * 2])
        #     outputs_permute.append(outputs[i + max_sentence_length * 3])
        #
        # outputs_conc = tf.concat(outputs_permute, axis=-1)
        # outputs_final = tf.reshape(outputs_conc, [outputs_conc.shape[0], outputs_conc.shape[1] * max_sentence_length,
        #                                           outputs_conc.shape[-1] // max_sentence_length])

        return outputs, word_attention_weights, sentence_attention_weights, paragraph_attention_weights
        # ret = [outputs]
        #
        # if return_state:
        #     ret.append(h)
        #
        # if return_attention_weights or self.use_attention and training == False:
        #     ret.append(attention_weights)
        #
        # return tuple(ret) if len(ret) > 1 else ret[0]

    def score(self, features, predictions):
        # Add perfect prediction for start symbol
        for i, prediction in enumerate(predictions):
            p = np.pad(prediction[:, :-1], ((0, 0), (1, 0), (0, 0)), mode='constant')
            p[:, 0, features[i][0, 0]] = 1
            predictions[i] = p
        return binet_scores_fn(np.dstack(features), predictions)

    def detect(self, dataset, batch_size=50):
        if isinstance(dataset, Dataset):
            features = dataset.hierarchic_features
            standard_features = dataset.features
        else:
            features = dataset
            standard_features = dataset

        # Get Attentions
        if self.use_attention:
            predictions = [[] for _ in range(dataset.num_attributes)]
            word_attentions = []
            sentence_attentions = []
            paragraph_attentions = []
            for step in range(dataset.num_cases // batch_size):
                prediction, word_attention, sentence_attention, paragraph_attention = self([f[:50] for f in features])

                for i in range(dataset.num_attributes):
                    predictions[i].append(prediction[i])

                word_attentions.append(word_attention)
                sentence_attentions.append(sentence_attention)
                paragraph_attentions.append(paragraph_attentions)

            # predictions = out[:len(self.rnn_inputs)]
            # attentions = self.split_attentions(out[len(self.rnn_inputs):])
        else:
            predictions = self.predict(features)
            word_attentions = None
            sentence_attentions = None
            paragraph_attentions = None

        for i in range(len(predictions)):
            predictions[i] = np.concatenate(predictions[i], axis=0)
            # predictions[i] = np.reshape(p, (dataset.num_cases, *p.shape[2:]))

        word_attentions = np.concatenate(word_attentions, axis=0)
        sentence_attentions = np.concatenate(sentence_attentions, axis=0)

        # Empty Attentions for start symbol
        word_attentions = np.pad(word_attentions[:, :-1], ((0, 0), (1, 0), (0, 0), (0, 0)), mode='constant')
        sentence_attentions = np.pad(sentence_attentions[:, :-1], ((0, 0), (1, 0), (0, 0)), mode='constant')

        if not isinstance(predictions, list):
            predictions = [predictions]

        return AnomalyDetectionResult(scores=self.score(standard_features, predictions),
                                      predictions=predictions, word_attentions=word_attentions,
                                      sentence_attentions=sentence_attentions,
                                      paragraph_attentions=paragraph_attentions)

    def load(self, model_file):
        if not self.loaded:
            self.compile(tf.keras.optimizers.Adam(), 'sparse_categorical_crossentropy')
            self([f[:50] for f in self.dataset.hierarchic_features])
            # self.fit([np.array([]), np.array([]), np.array([]), np.array([])], [np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])], epochs=0)
            # self.fit(self.dataset.features, [np.zeros(self.dataset.targets[0].shape) if i < self.dataset.num_attributes
            #                                  else np.zeros(
            #     (*self.dataset.targets[0].shape, self.dataset.targets[0].shape[-1]))
            #                                  for i in range(self.dataset.num_attributes + 1)], epochs=0)
            self.load_weights(str(model_file))
            self.loaded = True


def paragraph_to_sentence_level(x, mask):
    # TODO: Hardcoded Values
    max_paragraph_length = 8
    total_case_length = 119

    x = tf.concat(x, axis=-1)

    if x.shape[2] // max_paragraph_length > 1:
        x = tf.reshape(x, [x.shape[0], x.shape[1] * max_paragraph_length,
                                             x.shape[2] // max_paragraph_length, *x.shape[3:]])
    else:
        x = tf.reshape(x, [x.shape[0], x.shape[1] * max_paragraph_length, *x.shape[3:]])

    x = tf.ragged.boolean_mask(x, mask).to_tensor(0)
    return tf.pad(x, ((0, 0), (0, total_case_length - x.shape[1]), *((0, 0) for _ in range(len(x.shape) - 2))))
