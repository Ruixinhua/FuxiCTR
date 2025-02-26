import logging
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, LayerNormalization
from .layer import MultiHeadFeatureEmbedding, DeepCrossNetv3, ShallowCrossNetv3
from fuxictr.tensorflow.models import BaseModel
from .layer import concat_func


class DCNv3(BaseModel):
    """
    DCN, EDCN and GDCN 实现
    """

    def __init__(self, params, config):
        logging.info("Initializing DCN/EDCN/GDCN Model")
        super(DCNv3, self).__init__(params, config)
        self.model_specific_params = self.params.get('model_specific_params')
        self.mtl_set = self.params.get('mtl_set', None)
        self.use_bridge = self.model_specific_params.get('use_bridge', False)
        self.use_gated_cross = self.model_specific_params.get('use_gated_cross', False)
        self.use_relu = self.model_specific_params.get('use_relu', False)
        self.use_approximation = self.model_specific_params.get('use_approximation', False)
        self.approximation_rank = self.model_specific_params.get('approximation_rank', 64)
        self.num_cross = self.model_specific_params.get('num_cross', 3)
        self.l2_reg = self.model_specific_params.get('l2_reg', 0)
        self.activation = self.model_specific_params.get('activation', 'relu')
        self.use_bn = self.model_specific_params.get('use_bn', False)
        self.dropout_rate = self.model_specific_params.get('dropout_rate', 0.0)
        self.use_price = self.model_specific_params.get('use_price', False)
        self.dcn_version = self.model_specific_params.get('dcn_version', 'v1')
        self.scene_num = self.model_specific_params.get('scene_num', 2)
        if self.dcn_version == "v3" or self.dcn_version == "v3s":
            self.logits_xld = None
            self.logits_xls = None
            self.num_heads = self.model_specific_params.get('num_heads', 1)
            if not (self.num_heads == 1 or self.num_heads % 2 == 0):
                raise Exception("num_heads should be 1 or multiple of 2")
        self.struct = self.model_specific_params.get('struct', 'Parallel')
        self.use_layer_norm = self.model_specific_params.get("use_layer_norm", False)
        self.use_glorot_norm_init = self.model_specific_params.get('use_glorot_norm_init', False)
        self.use_einsum_gated_cross = self.model_specific_params.get('use_einsum_gated_cross', False)

        if self.use_price:
            self.use_am2_print = self.model_specific_params.get('use_am2_print', False)
            self.am2_print_step = self.model_specific_params.get('am2_print_step', 1000)
            self.price_weight = self.model_specific_params.get('price_weight', 0.1)
            self.head_unit = self.model_specific_params.get('head_unit', [64, 64])
            self.head_unit_p = self.model_specific_params.get('head_unit_p', [64, 64])
            self.y_price = None
            self.y_ctr = None
            self.use_am2_float = self.model_specific_params.get('use_am2_float', False)
        if not self.use_bridge or self.struct == 'Stacked' or self.use_gated_cross:
            self.hidden_units = self.model_specific_params.get('hidden_units', [64, 64])
        self.use_residual_connection = self.model_specific_params.get("use_residual_connection", False)
        self.randomization_level = self.model_specific_params.get("randomization_level", 0)

    def _get_logits(self, features, train_flag, **kwargs):
        """
        :param features:
        :param train_flag:
        :param kwargs:
        :return:
        """
        if self.struct not in ['Parallel', 'Stacked', 'Combined']:
            raise ValueError('struct should be "Parallel", "Stacked" or "Combined"')
        if "personalization_feature_list" in self.model_specific_params:
            (_, sparse_embedding_dict), dense_value_output = self._get_keras_input_embedding_list(
                features, is_get_dict=True, **kwargs
            )
            personalization_feature_list = self.model_specific_params['personalization_feature_list']
            logging.info(f"Mask Personalized Features: {','.join(personalization_feature_list)}")
            scene_id = tf.cast(tf.squeeze(self._scene_id_mapping(features), axis=-1), tf.int32)  # TODO: add multi-scene
            scene_select = tf.split(tf.one_hot(scene_id, 3), [1, 2], axis=1)[1]  # TODO: change to variables
            for feature in personalization_feature_list:
                if feature not in sparse_embedding_dict:
                    continue
                non_personalized_users = tf.stop_gradient(tf.expand_dims(scene_select[:, 0:1], axis=-1) * sparse_embedding_dict[feature])
                personalized_users = tf.expand_dims(scene_select[:, 1:], axis=-1) * sparse_embedding_dict[feature]
                if self.model_specific_params.get("use_personalized_users_only", False):
                    logging.info("Only use personalized users for selected personalized features")
                    sparse_embedding_dict[feature] = personalized_users
                else:
                    sparse_embedding_dict[feature] = non_personalized_users + personalized_users
            sparse_embedding_list = [emb for emb in sparse_embedding_dict.values()]
        else:
            sparse_embedding_list, dense_value_output = self._get_keras_input_embedding_list(features, **kwargs)

        # update _get_keras_input_embedding_list return data from concat tensor to list
        embeddings = Flatten()(concat_func(sparse_embedding_list))
        if len(dense_value_output) > 0:
            dense_value_output = tf.concat(dense_value_output, -1)
            embeddings = concat_func([embeddings, dense_value_output])
        if self.use_layer_norm:
            embeddings = LayerNormalization(axis=-1)(embeddings)
        x0 = embeddings
        xl = x0
        embedding_dim = int(embeddings.shape[1])
        if "v3" in self.dcn_version:
            logging.info("Initializing DCNv3 Model")
            if tf.is_tensor(dense_value_output):
                dense_value_output_fix = tf.reshape(dense_value_output, [-1, self.dense_single_num,
                                                                         self.dense_embedding_size])
                sparse_embedding_list = (concat_func(sparse_embedding_list, axis=1))
                embdedings_for_dcn_v3 = concat_func([sparse_embedding_list, dense_value_output_fix], axis=1)
            else:
                sparse_embedding_list = (concat_func(sparse_embedding_list, axis=1))
                embdedings_for_dcn_v3 = sparse_embedding_list

            def weird_division(x, y):
                try:
                    return x / y
                except ZeroDivisionError:
                    return 0
            xl = MultiHeadFeatureEmbedding(num_heads=self.num_heads)(embdedings_for_dcn_v3)
            num = embdedings_for_dcn_v3.shape[2] / self.num_heads if self.num_heads != 0 else 0
            input_dim_for_dcnv3 = int(embdedings_for_dcn_v3.shape[1] * num)
            xld = DeepCrossNetv3(num_heads=self.num_heads,
                                 num_cross_layers=self.num_cross,
                                 input_dim=input_dim_for_dcnv3,
                                 layer_norm=self.use_layer_norm,
                                 batch_norm=self.use_bn,
                                 net_dropout=self.dropout_rate)(xl)
            hidden_output = [xld]

            if self.dcn_version == 'v3s':
                logging.info("Initializing SDCNv3 Model")
                xls = ShallowCrossNetv3(input_dim=input_dim_for_dcnv3,
                                        num_heads=self.num_heads,
                                        layer_norm=self.use_layer_norm,
                                        batch_norm=self.use_bn,
                                        num_cross_layers=self.num_cross,
                                        net_dropout=self.dropout_rate)(xl)
                hidden_output.append(xls)
        if self.dcn_version == 'v3s':
            self.logits_xld = Dense(1, activation=None)(Flatten()(hidden_output[0]))
            self.logits_xls = Dense(1, activation=None)(Flatten()(hidden_output[1]))
            logit_mean = (self.logits_xld + self.logits_xls) * (0.5)
            logits = [logit_mean]

        else:
            hidden_output = Flatten()(concat_func(hidden_output))
            logits = Dense(1, activation=None)(hidden_output)

        return logits

    def _get_loss(self, logits, labels, weights=1.0, **kwargs):
        if self.use_price:
            label_name_list = self.mtl_set.get('task_label_name_list', None)

            if label_name_list is None:
                raise NotImplementedError("pls check mtl_set")

            price_label, ctr_label = [tf.reshape(labels[label_name], [-1, 1]) for label_name in label_name_list]

            ctr_loss = super()._get_loss(logits, ctr_label, weights)

            price_label = tf.where(tf.math.is_nan(price_label), -tf.ones_like(price_label), price_label)
            price_instance_tensor = tf.convert_to_tensor(-1.0, dtype=tf.float32)
            price_instance_mask = tf.greater(price_label, price_instance_tensor)
            price_label = tf.boolean_mask(price_label, price_instance_mask)
            price_label = tf.compat.v1.to_float(price_label)
            if not self.use_am2_float:
                price_label = price_label / 1000  # 保留的精度值
            self.y_price = tf.boolean_mask(self.y_price, price_instance_mask)

            mae = tf.keras.losses.MeanAbsoluteError()
            price_loss = mae(y_true=price_label, y_pred=self.y_price)

            self.loss_item['loss_price'] = self.price_weight * price_loss
            self.loss_item['loss_ctr'] = ctr_loss

            loss = ctr_loss + self.price_weight * price_loss

        elif self.dcn_version == 'v3s':
            loss_xld = super()._get_loss(self.logits_xld, labels)
            loss_xls = super()._get_loss(self.logits_xls, labels)
            loss_mean = super()._get_loss(logits, labels)

            wd = loss_xld - loss_mean
            ws = loss_xls - loss_mean
            wd = tf.where(wd > 0, wd, tf.zeros_like(wd))
            ws = tf.where(ws > 0, ws, tf.zeros_like(ws))

            loss = loss_mean + wd * loss_xld + ws * loss_xls

        else:
            if isinstance(labels, dict):
                labels = labels['label']
                labels = tf.reshape(labels, [-1, 1])
            loss = super()._get_loss(logits, labels, weights)
        return loss

