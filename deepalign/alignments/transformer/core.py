import tensorflow as tf
import numpy as np
import math

from deepalign import Dataset
from deepalign.anomalydetection import AnomalyDetectionResult
from deepalign.alignments.transformer.encoder import Encoder
from deepalign.enums import Base, Heuristic, Mode, Strategy, AttributeType, FeatureType


class Transformer(tf.keras.Model):
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

    # TODO: Tune Params, e.g. num_encoder_layers, mha_heads
    def __init__(self,
                 dataset,
                 num_encoder_layers=2,
                 mha_heads=4,
                 dropout_rate=0.1,
                 ff_dim=32,
                 fixed_emb_dim=12,
                 use_present_attributes=True,
                 max_case_length_modificator=0):
        super(Transformer, self).__init__()

        # Parameters
        self.dataset = dataset

        # Layer lists
        self.fc_inputs = []
        self.cf_inputs = []
        self.pre_outs = []
        self.dropout_outs = []
        self.outs = []

        # This will also be the dim of the positional encoding; * 2 to guarantee an even number
        emb_dim = max([np.clip((dim + 1) // 20, 1, 5) * 2 for dim in dataset.attribute_dims]) \
            if fixed_emb_dim is None else fixed_emb_dim

        inputs = zip(dataset.attribute_dims, dataset.attribute_keys, dataset.attribute_types, dataset.feature_types)
        for dim, key, t, feature_type in inputs:
            if t == AttributeType.CATEGORICAL:
                voc_size = int(dim + 1)  # we start at 1, 0 is padding

                embed = tf.keras.layers.Embedding(input_dim=voc_size, output_dim=emb_dim, mask_zero=True,
                                                  input_shape=(None, dataset.max_len, voc_size))

                out = tf.keras.layers.Dense(voc_size + 1, activation='softmax')  # TODO: +1 is TEMP

            else:
                embed = tf.keras.layers.Dense(1, activation='linear')

                out = tf.keras.layers.Dense(1, activation='linear')

            if feature_type == FeatureType.CASE:
                self.fc_inputs.append(embed)
            else:
                self.cf_inputs.append(embed)
                self.pre_outs.append(tf.keras.layers.Dense(ff_dim, activation='relu'))
                self.dropout_outs.append(tf.keras.layers.Dropout(dropout_rate))
                self.outs.append(out)

        self.d_model = emb_dim
        self.use_present_attributes = use_present_attributes
        self.num_encoder_layers = num_encoder_layers
        self.num_total_attributes = dataset.num_attributes
        self.num_event_attributes = dataset.num_event_attributes
        self.max_case_length = (dataset.max_len + max_case_length_modificator) * self.num_event_attributes

        self.encoder = Encoder(num_encoder_layers, self.num_event_attributes, self.d_model, mha_heads, ff_dim,
                               self.max_case_length, rate=dropout_rate)

        self.look_ahead_mask = tf.linalg.band_part(tf.ones((self.max_case_length, self.max_case_length)), -1, 0)
        # self.look_ahead_mask = tf.pad(self.look_ahead_mask[:, 1:], ((0, 0), (0, 1)))  # Dont allow "self" attention

    def call(self, x_in_, training=False):
        if not isinstance(x_in_, list):
            # inputs = [inputs]
            # x_in = [x_in[:, :, i] for i in range(self.num_total_attributes)]
            x_in = []

            i = 0
            for j in range(self.num_total_attributes):
                width = self.max_case_length // self.num_event_attributes if self.dataset.feature_types[
                                                                                 j] != FeatureType.CASE else 1
                x_in.append(x_in_[:, i:i + width])
                i += width
        else:
            x_in = x_in_

        split = len(self.cf_inputs)

        cf_in = x_in[:split]
        fc_in = x_in[split:]

        # We dont have to check if we have event attributes, since we will always have at least 1 attribute (CF)
        # Otherwise, we wouldnt even have outputs
        x_emb = [input(x_, training=training) for x_, input in zip(cf_in, self.cf_inputs)]
        emb_mask = x_emb[0]._keras_mask  # Mask is the same for all attributes
        x_emb = tf.transpose(x_emb, [1, 2, 0, 3])  # Reorder attributes to be consecutive per event
        x_cf = tf.reshape(x_emb, [x_emb.shape[0], x_emb.shape[1] * x_emb.shape[2], x_emb.shape[3]])

        # Reapply Mask
        x_cf._keras_mask = tf.concat([emb_mask[:, i:i + 1] for i in range(emb_mask.shape[1])  # TODO: Inefficient?
                                      for _ in range(self.num_event_attributes)], axis=-1)

        # TODO: Positional Encoding should probably be the same for all attributes of the same event, right?
        x_cf, encoder_attentions = self.encoder(x_cf, training=training, mask=self.look_ahead_mask)
        # x = tf.reshape(x, [x.shape[0], x.shape[1] // self.num_attributes, x.shape[2] * self.num_attributes])

        # outputs = []
        # for i, p, d, o in zip(range(self.num_attributes), self.pre_outs, self.dropout_outs, self.outs):
        #     if self.use_present_attributes:
        #         present_attributes = tf.concat([x_emb[:, :, :i], x_emb[:, :, i + 1:]], axis=-2)
        #         present_attributes = tf.reshape(present_attributes,
        #                                         [*present_attributes.shape[:2],
        #                                          present_attributes.shape[2] * present_attributes.shape[3]])
        #
        #         x_local = tf.concat([x, present_attributes], axis=-1)
        #     else:
        #         x_local = x
        #
        #     pre_out = d(p(x_local), training=training)
        #     outputs.append(o(pre_out))

        if len(fc_in) > 0:
            x_fc = tf.concat([input(x_, training=training) for x_, input in zip(fc_in, self.fc_inputs)], axis=-1)
            x_fc = tf.repeat(x_fc, [self.max_case_length], axis=1)

            x = tf.concat([x_cf, x_fc], axis=-1)
        else:
            x = x_cf

        outputs = [o(p(x))[:, (i - 1) % self.num_event_attributes::self.num_event_attributes] for i, p, o in
                   zip(range(len(self.outs)), self.pre_outs, self.outs)]

        return outputs  # , tf.math.reduce_mean(encoder_attentions, axis=2)

    def score(self, features, predictions):
        # Add perfect prediction for start symbol

        for i, prediction in enumerate(predictions):
            p = np.pad(prediction[:, :-1], ((0, 0), (1, 0), (0, 0)), mode='constant')
            p[:, 0, features[i][0, 0]] = 1
            predictions[i] = p

        return transformer_scores_fn(np.dstack(features), predictions)

    def detect(self, dataset, batch_size=50):
        if isinstance(dataset, Dataset):
            # features = dataset.hierarchic_features
            features = dataset.features
            standard_features = dataset.features
        else:
            features = dataset
            standard_features = dataset

            # # Get attentions and predictions
            # predictions = [[] for _ in range(dataset.num_attributes)]
            # attentions = [[[[] for _ in range(dataset.num_attributes)] for _ in range(dataset.num_attributes)]
            #               for _ in range(self.num_encoder_layers)]
            # for step in range(dataset.num_cases // batch_size):
            #     prediction, attentions_raw = self([f[step * batch_size:(step + 1) * batch_size] for f in features],
            #                                       training=False)
            #
            #     for i in range(dataset.num_attributes):
            #         predictions[i].append(prediction[i].numpy())
            #
            #         for k in range(len(attentions_raw)):
            #             outer_mask = tf.equal(tf.range(attentions_raw[k].shape[-2]) % dataset.num_attributes, i)
            #             outer_masked = tf.boolean_mask(attentions_raw[k], outer_mask, axis=-2)
            #
            #             for j in range(dataset.num_attributes):
            #                 attentions[k][i][j].append(outer_masked[:, j].numpy().astype(np.float16))

            # Get attentions and predictions
        predictions = [[] for _ in range(dataset.num_attributes)]
        attentions = [[[[] for _ in range(dataset.num_attributes)] for _ in range(dataset.num_attributes)]
                      for _ in range(self.num_encoder_layers)]
        for step in range(dataset.num_cases // batch_size):
            prediction, attentions_raw = self([f[step * batch_size:(step + 1) * batch_size] for f in features],
                                              training=False)

            for i in range(dataset.num_attributes):
                if i != 0:
                    pred = tf.pad(prediction[i][:, 1:], ((0, 0), (0, 1), (0, 0)))
                else:
                    pred = prediction[i]

                predictions[i].append(pred.numpy())
                # TODO: ADD PERFECT PREDICTION FOR STOP SYMBOL! BUT AT THE RIGHT POSITION! (NOT NECCESSARY THE END)

                for k in range(len(attentions_raw)):
                    outer_mask = tf.equal(tf.range(attentions_raw[k].shape[-2]) % dataset.num_attributes, i)
                    outer_masked = tf.boolean_mask(attentions_raw[k], outer_mask, axis=1)

                    for j in range(dataset.num_attributes):
                        inner_mask = tf.equal(tf.range(attentions_raw[k].shape[-1]) % dataset.num_attributes, j)
                        inner_masked = tf.boolean_mask(outer_masked, inner_mask, axis=2)

                        attentions[k][i][j].append(inner_masked.numpy().astype(np.float16))

            if step % 10 == 0:
                print('prediction step %s' % step)

        for i in range(dataset.num_attributes):
            predictions[i] = np.concatenate(predictions[i], axis=0)

            for k in range(len(attentions)):
                for j in range(dataset.num_attributes):
                    attentions[k][i][j] = np.concatenate(attentions[k][i][j], axis=0)
                    # attentions_concat = np.concatenate(attentions[k][i][j], axis=0)
                    # attentions[k][i][j] = np.pad(attentions_concat[:, :-1], ((0, 0), (1, 0), (0, 0)), mode='constant')

        if not isinstance(predictions, list):
            predictions = [predictions]

        return AnomalyDetectionResult(scores=self.score(standard_features, predictions),
                                      predictions=predictions, attentions=attentions)

    def load(self, model_file):
        # TODO: Remove hardcoded values
        if not self.loaded:
            self.compile(tf.keras.optimizers.Adam(), 'categorical_crossentropy')
            self([f[:50] for f in self.dataset.features])
            self.load_weights(str(model_file))
            self.loaded = True


def transformer_scores_fn(features, predictions):
    maxes = [np.repeat(np.expand_dims(np.max(p, axis=-1), axis=-1), p.shape[-1], axis=-1) for p in predictions]
    indices = [features[:, :, i:i + 1] for i in range(len(predictions))]
    scores_all = [(m - p) / m for p, m in zip(predictions, maxes)]

    scores = np.zeros(features.shape)
    for (i, j, k), f in np.ndenumerate(features):
        if f != 0 and k < len(scores_all):
            scores[i, j, k] = scores_all[k][i, j][indices[k][i, j]]

    return scores
