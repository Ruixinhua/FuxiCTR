import logging
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten, Dense
from .layer import MultiHeadFeatureEmbedding, DeepCrossNetv3, ShallowCrossNetv3, concat_func
from fuxictr.tensorflow.models import BaseModel
from fuxictr.tensorflow.layers import FeatureEmbedding, Linear


class DCN(BaseModel):
    """
    DCN, EDCN and GDCN 实现
    """

    def __init__(self,
                 feature_map,
                 model_id="DCNv3",
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0,
                 num_cross_layers=3,
                 use_batch_norm=True,
                 use_layer_norm=True,
                 num_heads=2,
                 embedding_regularizer=None,
                 use_domain_aware_structure=False,
                 **kwargs):
        logging.info("Initializing DCNv3 Model")
        super(DCN, self).__init__(feature_map, model_id=model_id, **kwargs)
        self.num_cross = num_cross_layers
        self.use_layer_norm = use_layer_norm
        self.use_batch_norm = use_batch_norm
        self.net_dropout = net_dropout
        self.logits_xld = None
        self.logits_xls = None
        self.num_heads = num_heads
        self.use_domain_aware_structure = use_domain_aware_structure
        if self.use_domain_aware_structure:
            self.init_domain_aware_structure_params()
        # Model structure
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim,
                                                embedding_regularizer=embedding_regularizer)
        self.multi_head_feature_embed = MultiHeadFeatureEmbedding(num_heads=self.num_heads)
        num = embedding_dim / self.num_heads if self.num_heads != 0 else 1
        input_dim = int(len(feature_map.features) * num)
        self.ECN = DeepCrossNetv3(input_dim=input_dim,
                                  num_cross_layers=num_cross_layers,
                                  layer_norm=use_layer_norm,
                                  batch_norm=use_batch_norm,
                                  net_dropout=net_dropout,
                                  num_heads=num_heads)
        self.FCN = ShallowCrossNetv3(input_dim=input_dim,
                                     num_heads=num_heads,
                                     layer_norm=use_layer_norm,
                                     batch_norm=use_batch_norm,
                                     num_cross_layers=num_cross_layers,
                                     net_dropout=net_dropout)
        self.fc1 = Linear(1, regularizer=0) # deepart -> logit
        self.fc2 = Linear(1, regularizer=0) # shallowpart -> logit
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        if not (self.num_heads == 1 or self.num_heads % 2 == 0):
            raise Exception("num_heads should be 1 or multiple of 2")

    def init_domain_aware_structure_params(self):
        pass

    def call(self, inputs, **kwargs):
        """
        :param inputs: a dictionary of input tensors {feature: tensor(batch_size, feature_value)}
        :return: a return_dict with key "y_pred" (logits)
        """
        features = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(features, flatten_emb=False)
        xl = self.multi_head_feature_embed(feature_emb)  # xl: (batch_size, head_num, (emb_dim / head_num) * feat_num)
        xld = self.ECN(xl)  # xld: (batch_size, head_num, (emb_dim / head_num) * feat_num)
        xls = self.FCN(xl)  # xls: (batch_size, head_num, (emb_dim / head_num) * feat_num)
        if self.use_domain_aware_structure:
            # deep crossing 相关变量输出使用xld作为后缀
            output_xld = Flatten()(xld)
            # shallow crossing 相关变量输出使用xls作为后缀
            output_xls = Flatten()(xls)
            self.logits_xld = self.generate_domain_aware_logits(features, output_xld)
            self.logits_xls = self.generate_domain_aware_logits(features, output_xls)
        else:
            self.logits_xld = self.fc1(Flatten()(xld))
            self.logits_xls = self.fc2(Flatten()(xls))
        logit_mean = (self.logits_xld + self.logits_xls) * 0.5
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

    def generate_domain_aware_logits(self, features, net_output):
        """
        Function propagates the model's output into the multi-tower defined in itself.
        In this function based on the configuration of the multi-towers and features specified
        necessary calculations are completed. Based on the scene names and the network output is
        distributed to the towers defined. The function returns the batch of logits produced by the tower
        structure.
        :param self: BaseModel instance call by reference
        :type self: BaseModel class instance
        :param features: list of the features defined in the model to be used in training process
        :type features: list
        :param net_output: output tensor with shape [None, last_hidden_dimension]. The final output produced
        by the deep learning model
        :type net_output: tf.Tensor
        :return: returns the logits produced by the multi-tower structure
        :rtype: tf.Tensor
        """

        # 获取scene_id
        if self.use_scene_id_mapping:
            scene_id = self._scene_id_mapping(features)
            scene_id = tf.cast(tf.squeeze(scene_id, axis=-1), tf.int32)
        else:
            scene_id = tf.cast(tf.squeeze(features.get(self.scene_name, 'scene_id'), axis=-1), tf.int32)

        # 分场景/任务tower
        scene_tower_output = list()
        scene_tower_vector = list()
        negative_logits_list = list()
        for tower_index, hidden_units in enumerate(self.tower_hidden_units_list):
            dnn_kwargs = {}

            tower_output = DNN(hidden_units=hidden_units,
                               activation=self.tower_activation,
                               l2_reg=self.tower_l2_reg_list[tower_index],
                               dropout_rate=self.tower_dropout_list[tower_index],
                               use_bn=self.use_bn_tower,
                               seed=self.global_seed)(net_output, **dnn_kwargs)
            scene_tower_vector.append(tower_output)
            tower_logits = Dense(units=1, name='tower_' + str(tower_index + 1))(tower_output)
            scene_tower_output.append(tower_logits)

        from collections import namedtuple
        SceneConf = namedtuple('SceneConf', ['scene_id', 'scene_num', 'scene_num_shift'])
        scene_conf = SceneConf(scene_id, self.scene_num, self.scene_num_shift)
        logits = logits_routing(scene_tower_output, scene_conf)
        return logits

    def _scene_id_mapping(self, features):
        """
        通过给定的配置将特征值映射为scene_id
        :param features: dict, 训练数据特征和特征值的字典
        :return scene_ids: tensor, 映射后的scene_id
        """
        logging.info("using scene_id mapping")
        if self.mapping_feature_name in features:
            feature_values = features[self.mapping_feature_name]
        else:
            raise ValueError("Invalid feature name. Please re-check it.")

        scene_ids = tf.ones_like(feature_values, dtype=tf.int32) * self.default_value
        for feature_value, scene_id in self.feature2id_dict.items():
            if scene_id > self.scene_num or scene_id < 1:
                raise ValueError("Invalid scene_id. Please re-check it.")
            if self.mapping_feature_type == 'sparse':
                feature_value_mapped = self.feature_map_dict.get(feature_value)
                scene_ids = tf.where(tf.equal(feature_values, feature_value_mapped), scene_id, scene_ids)
            else:
                scene_ids = tf.where(tf.equal(feature_values, feature_value), scene_id, scene_ids)

        return scene_ids

def logits_routing(scene_tower_output, scene_conf):
    """
    给定一般logits list，根据scene_id合并各场景/任务tower的logits至最终logits

    参数：
        scene_tower_output: list(Tensor), 所有towers输出的logits
        scene_conf: tuple, (scene_id, scene_num, scene_num_shift)
    返回值:
        tensor: logits
    """

    scene_id, scene_num, scene_num_shift = scene_conf.scene_id, scene_conf.scene_num, scene_conf.scene_num_shift

    scene_logits = layers.Concatenate(name='scene_logits_concat', axis=1)(scene_tower_output)
    scene_select = tf.split(tf.one_hot(scene_id, scene_num_shift + scene_num),
                            [scene_num_shift, scene_num], axis=1)[1]
    scene_logits_final = tf.reduce_sum(input_tensor=scene_logits * scene_select, axis=-1)

    logits = layers.Reshape((1,), name='final')(scene_logits_final)
    return logits
