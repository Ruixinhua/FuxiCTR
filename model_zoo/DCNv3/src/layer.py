import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Concatenate, Dense, Layer


class MultiHeadFeatureEmbedding(layers.Layer):
    """
    MultiHeadFeatureEmbedding for DCNv3 model
    args: num_heads: int, number of heads

    """

    def __init__(self, num_heads=2):
        super(MultiHeadFeatureEmbedding, self).__init__()
        self.num_heads = num_heads

    def call(self, inputs):
        feature_emb = inputs  # B × F × D
        multihead_feature_emb = tf.split(feature_emb, self.num_heads, axis=-1)  # B x F x D/H
        multihead_feature_emb = tf.stack(multihead_feature_emb, axis=1)  # B × H × F × D/H
        multihead_feature_emb1, multihead_feature_emb2 = tf.split(multihead_feature_emb, 2, axis=-1)  # B × H × F*D/2*H
        multihead_feature_emb1 = tf.reshape(multihead_feature_emb1, (-1, self.num_heads,
                                                                     multihead_feature_emb1.shape[2] * \
                                                                     multihead_feature_emb1.shape[
                                                                         3]))  # B x H x F*D/2*H
        multihead_feature_emb2 = tf.reshape(multihead_feature_emb2, (-1, self.num_heads,
                                                                     multihead_feature_emb2.shape[2] * \
                                                                     multihead_feature_emb2.shape[
                                                                         3]))  # B x H x F*D/2*H
        multihead_feature_emb = tf.concat([multihead_feature_emb1, multihead_feature_emb2], axis=-1)  # B × H × F*D/H
        return multihead_feature_emb


class DeepCrossNetv3(layers.Layer):
    """
    DeepCrossNet layer for DCNv3 model
    args:
        input_dim: int, Input dimension of the gated cross layer
        num_cross_layers: int, number of model layers
        layer_norm: bool, Layer normalziation flag
        batch_norm: bool, Batch normalization flag
        num_heads: int, number of heads
        net_dropout: int, rate of dropout
    """

    def __init__(self,
                 input_dim,
                 num_cross_layers=2,
                 layer_norm=False,
                 batch_norm=False,
                 net_dropout=0.0,
                 num_heads=1,
                 ):
        super(DeepCrossNetv3, self).__init__()
        self.num_cross_layers = num_cross_layers
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.net_dropout = net_dropout
        self.num_heads = num_heads
        # Initialize layers
        self.w = [Dense(input_dim // 2, use_bias=False) for _ in range(num_cross_layers)]
        self.b = [self.add_weight(shape=(input_dim // 2,), initializer='random_uniform',
                                  name=f'b_{i}') for i in range(num_cross_layers)]

        if layer_norm:
            self.layer_norm_layers = [tf.keras.layers.LayerNormalization() for _ in range(num_cross_layers)]
        if batch_norm:
            self.batch_norm_layers = [tf.keras.layers.BatchNormalization() for _ in range(num_cross_layers)]
        if net_dropout > 0:
            self.dropout_layers = [tf.keras.layers.Dropout(net_dropout) for _ in range(num_cross_layers)]

        self.fc = Dense(1)

    def call(self, x, training=False):

        for i in range(self.num_cross_layers):
            H = self.w[i](x)
            H = H + self.b[i]
            if self.batch_norm:
                H = self.batch_norm_layers[i](H, training=training)

            if self.layer_norm:
                norm_H = self.layer_norm_layers[i](H)
                mask = tf.nn.relu(norm_H)
            else:
                mask = tf.nn.relu(H)

            H = tf.concat([H, H * mask], axis=-1)
            x = x * (H) + x
            if self.net_dropout > 0:
                x = self.dropout_layers[i](x, training=training)
        return x


class ShallowCrossNetv3(layers.Layer):
    """
    ShallowCrossNet for DCNv3 model
    args:
        input_dim: int, Input dimension of the gated cross layer
        num_cross_layers: int, number of model layers
        layer_norm: bool, Layer normalziation flag
        batch_norm: bool, Batch normalization flag
        num_heads: int, number of heads
        net_dropout: int, rate of dropout
    """

    def __init__(self, input_dim, num_cross_layers=2, layer_norm=False, batch_norm=False, net_dropout=0.3, num_heads=2):
        super(ShallowCrossNetv3, self).__init__()

        self.num_cross_layers = num_cross_layers
        self.num_heads = num_heads
        self.layer_norms = [layers.LayerNormalization() if layer_norm else None for _ in range(self.num_cross_layers)]
        self.batch_norms = [layers.BatchNormalization() if batch_norm else None for _ in range(self.num_cross_layers)]
        self.dropouts = [layers.Dropout(net_dropout) if net_dropout > 0
                         else None for _ in range(self.num_cross_layers)]
        self.ws = [layers.Dense(input_dim // 2, use_bias=False) for _ in range(self.num_cross_layers)]
        self.bs = [self.add_weight(shape=(input_dim // 2,), initializer='zeros',
                                   trainable=True) for _ in range(self.num_cross_layers)]
        self.masker = layers.ReLU()
        self.sfc = Dense(1)

    def call(self, x):
        x0 = x
        for i in range(self.num_cross_layers):
            H = self.ws[i](x)
            H = H + self.bs[i]
            if self.batch_norms[i]:
                H = self.batch_norms[i](H)

            if self.layer_norms[i]:
                norm_H = self.layer_norms[i](H)
                mask = self.masker(norm_H)
            else:
                mask = self.masker(H)
            H = tf.concat([H, H * mask], axis=-1)
            x = x0 * (H) + x
            if self.dropouts[i]:
                x = self.dropouts[i](x)
        return x


class NoMaskLayer(Layer):
    def __init__(self, **kwargs):
        super(NoMaskLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(NoMaskLayer, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask):
        return None


def concat_func(inputs, axis=-1, mask=False):
    if not mask:
        inputs = list(map(NoMaskLayer(), inputs))
    if len(inputs) == 1:
        return inputs[0]
    else:
        return Concatenate(axis=axis)(inputs)
