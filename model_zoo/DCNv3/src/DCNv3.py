import logging
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from .layer import MultiHeadFeatureEmbedding, DeepCrossNetv3, ShallowCrossNetv3
from fuxictr.tensorflow.models import BaseModel
from fuxictr.tensorflow.layers import FeatureEmbedding, Linear


class DCNv3(BaseModel):
    """
    DCN, EDCN and GDCN 实现
    """

    def __init__(self,
                 feature_map,
                 model_id="DCN",
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0,
                 num_cross_layers=3,
                 use_batch_norm=True,
                 use_layer_norm=True,
                 num_heads=2,
                 embedding_regularizer=None,
                 **kwargs):
        logging.info("Initializing DCNv3 Model")
        super(DCNv3, self).__init__(feature_map, model_id=model_id, **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim,
                                                embedding_regularizer=embedding_regularizer)
        self.fc = Linear(1, regularizer=0) # [cross_part, dnn_part] -> logit
        self.num_cross = num_cross_layers
        self.use_layer_norm = use_layer_norm
        self.use_batch_norm = use_batch_norm
        self.net_dropout = net_dropout
        self.logits_xld = None
        self.logits_xls = None
        self.num_heads = num_heads
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        if not (self.num_heads == 1 or self.num_heads % 2 == 0):
            raise Exception("num_heads should be 1 or multiple of 2")

    def call(self, inputs, **kwargs):
        """
        :param inputs:
        :return:
        """
        feature_emb = self.embedding_layer(self.get_inputs(inputs), flatten_emb=False)
        xl = MultiHeadFeatureEmbedding(num_heads=self.num_heads)(feature_emb)
        num = feature_emb.shape[2] / self.num_heads if self.num_heads != 0 else 0
        input_dim = int(feature_emb.shape[1] * num)
        xld = DeepCrossNetv3(num_heads=self.num_heads,
                             num_cross_layers=self.num_cross,
                             input_dim=input_dim,
                             layer_norm=self.use_layer_norm,
                             batch_norm=self.use_batch_norm,
                             net_dropout=self.net_dropout)(xl)
        xls = ShallowCrossNetv3(input_dim=input_dim,
                                num_heads=self.num_heads,
                                layer_norm=self.use_layer_norm,
                                batch_norm=self.use_batch_norm,
                                num_cross_layers=self.num_cross,
                                net_dropout=self.net_dropout)(xl)
        hidden_output = [xld, xls]
        self.logits_xld = Dense(1, activation=None)(Flatten()(hidden_output[0]))
        self.logits_xls = Dense(1, activation=None)(Flatten()(hidden_output[1]))
        logit_mean = (self.logits_xld + self.logits_xls) * (0.5)
        return_dict = {"y_pred": logit_mean}
        return return_dict

    def compute_loss(self, inputs, **kwargs):
        # return loss
        return_dict = self(inputs, **kwargs)
        y_true = self.get_labels(inputs)
        loss_xld = self.loss_fn(y_true, self.logits_xld)
        loss_xls = self.loss_fn(y_true, self.logits_xls)
        loss_bce = self.loss_fn(y_true, return_dict["y_pred"])
        wd = loss_xld - loss_bce
        ws = loss_xls - loss_bce
        wd = tf.where(wd > 0, wd, tf.zeros_like(wd))
        ws = tf.where(ws > 0, ws, tf.zeros_like(ws))
        total_loss = loss_bce + wd * loss_xld + ws * loss_xls
        # total_loss = self.add_loss(inputs) + sum(self.losses) # with regularization
        return total_loss

