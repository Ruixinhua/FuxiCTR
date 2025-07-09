import logging
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Concatenate, Dense, Layer, LayerNormalization


class Attention(Layer):
    def __init__(self, hidden_units=1, num_heads=2, key_dim=16, patch_nums=20, dropout_rate=0.1,
                    self_att=False, use_mlp=True, use_ln=False, use_fc=True, **kwargs):
        """
        Attention层
        输入shape：(batch_size, ..., input_dim)，2D输入时，(batch_size, input_dim)
        输出shape：(batch_size, ..., hidden_size[-1])，2D输入时(batch_size, hidden_size[-1])

        :param hidden_units: 每层神经元数量，一个整数列表
        :param num_heads: attention的head数量
        :param key_dim: attention的key_dim
        :param patch_nums: attention 的patch_num
        :param dropout_rate: [0, 1)之间的float参数，定义dropout率
        :param self_att: 是否使用self attention替代attention
        :param use_mlp: attention之后是否增加mlp层
        :param use_fc: 是否使用fc层适配
        :param kwargs: 其他参数
        """

        self.support_attention = True
        self.atten = None
        self.dropout_rate = dropout_rate
        try:
            from tensorflow.keras.layers import MultiHeadAttention
            self.atten = [MultiHeadAttention(num_heads, key_dim,
                          dropout=self.dropout_rate) for _ in range(hidden_units)]
        except ImportError as e:
            self.support_attention = False
            logging.info("not support MultiHeadAttention")

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.patch_nums = patch_nums
        self.dense_size = self.num_heads * self.key_dim
        self.self_att = self_att
        self.use_mlp = use_mlp
        self.use_ln = use_ln
        self.use_fc = use_fc

        self.first_fc = Dense(self.num_heads * self.key_dim * self.patch_nums, use_bias=True, activation=None)
        self.dense = [Dense(self.dense_size * 3,
                            use_bias=True,
                            activation=None) for _ in range(self.hidden_units)]

        self.mlp1 = [Dense(self.num_heads * self.key_dim * 2,
                           use_bias=True,
                           activation=tf.keras.activations.gelu) for _ in range(self.hidden_units)]
        self.mlp2 = [Dense(self.num_heads * self.key_dim,
                           use_bias=True,
                           activation=None) for _ in range(self.hidden_units)]
        self.ln = [LayerNormalization() for _ in range(self.hidden_units)]

        super(Attention, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        deep_input = inputs
        if self.use_fc:
            deep_input = self.first_fc(deep_input)
        deep_input = tf.reshape(deep_input, [-1, self.patch_nums, self.dense_size])

        if self.support_attention:
            for i in range(self.hidden_units):
                if self.self_att:
                    q = deep_input
                    k = deep_input
                    v = deep_input
                else:
                    qkv = self.dense[i](deep_input)
                    qkv = tf.reshape(qkv, [-1, self.patch_nums, 3, self.dense_size])
                    q = qkv[:, :, 0]
                    k = qkv[:, :, 1]
                    v = qkv[:, :, 2]
                deep_input = self.atten[i](q, k, v)

                if self.use_mlp:
                    if self.use_ln:
                        deep_input = self.ln[i](deep_input)
                    deep_input = self.mlp1[i](deep_input)
                    deep_input = self.mlp2[i](deep_input)

        return deep_input

    def get_config(self):
        config = {'hidden_units': self.hidden_units, 'num_heads': self.num_heads, 'key_dim': self.key_dim,
                  'patch_nums': self.patch_nums, 'dropout_rate': self.dropout_rate, 'self_att': self.self_att,
                  'use_mlp': self.use_mlp, 'use_ln': self.use_ln, 'use_fc': self.use_fc}
        base_config = super(Attention, self).get_config()
        base_config.update(config)
        return base_config


class MultiHeadFeatureEmbedding(layers.Layer):
    """
    MultiHeadFeatureEmbedding for DCNv3 model
    args: num_heads: int, number of heads

    """

    def __init__(self, num_heads=2):
        super(MultiHeadFeatureEmbedding, self).__init__()
        self.num_heads = num_heads

    def call(self, inputs, **kwargs):
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

    def call(self, x, training=False, **kwargs):

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

    def call(self, x, **kwargs):
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
