import logging

import tensorflow as tf

from layer.loss import bce_loss
from model.dcn import DCN


class DCNv3CL(DCN):
    """
    DCNv3CL.
    Reference:
    DCNv3CL: A Contrastive Learning Framework for CTR Prediction
    The FI encoder is built upon DCN, its performance will be maximised when DCN version == "v3s"
    """

    def __init__(self, params, config):
        logging.info("Initializing DCNv3CL Model")
        super(DCNv3CL, self).__init__(params, config)

        self.params = params
        self.config = config
        self.model_specific_params = self.params.get('model_specific_params')
        self.mtl_set = self.params.get('mtl_set', None)

        # personalisation params
        self.use_personalisation = self.model_specific_params.get("use_personalisation", False)
        if self.use_personalisation:
            self.personalization_feature_list = self.model_specific_params['personalization_feature_list']
            self.non_personalization_feature_list = list()
            self.non_personalised_features = dict()
            self.non_personalised_dense_features = dict()
            for feature in self.feat_cls_list:
                if feature.name not in self.personalization_feature_list:
                    self.non_personalization_feature_list.append(feature.name)

        # DCNv3CL params
        self.mask_type = self.model_specific_params.get('mask_type', 'Personalisation')
        self.base_loss_weight = self.model_specific_params.get('base_loss_weight', 1)
        self.feature_alignment_loss_weight = self.model_specific_params.get('feature_alignment_loss_weight', 0)
        self.field_uniformity_loss_weight = self.model_specific_params.get('field_uniformity_loss_weight', 0)
        self.distance_loss_weight = self.model_specific_params.get('distance_loss_weight', 0.1)
        self.keep_prob = self.model_specific_params.get('keep_prob', 1.0)
        self.use_cl_mask = self.model_specific_params.get('use_cl_mask', False)
        self.feature_alignment_loss = None
        self.field_uniformity_loss = None
        self.distance_loss = None
        self.logits_dist = None

    def get_feature_embeddings(self, features, **kwargs):
        """
        obtains the embeddings for each feature
        :param features: dict, 模型输入特征
        :param **kwargs: 其它参数
        :return feature_embeddings: dict, 各个特征下的embedding
        """
        feature_embeddings = dict()
        for feature in features:
            if feature != self.params.get('unique_id'):
                feature_list = [feature]
                feature_features = dict()
                feature_features[feature] = features[feature]
                feature_kwargs = kwargs.copy()
                feature_kwargs['features_to_select'] = feature_list
                feature_sparse_embedding_list, feature_dense_value_output = self.get_keras_input_embedding_list(
                    feature_features, **feature_kwargs)
                if feature_dense_value_output:
                    feature_embeddings[feature] = feature_dense_value_output[0]
                else:
                    feature_embeddings[feature] = feature_sparse_embedding_list[0]

        return feature_embeddings

    def sum_unique_pairwise_distances(self, tensor):
        """
        Calculates the pairwise distance between each embedding under the same feature
        :param tensor: tensor, 单一特征下的embeddings
        :return sum_distances: tensor, 单一特征下embeddings间的pairwise距离之和
        :return n_pairs: tensor, 单一特征下embeddings间成对的数量
        """
        # Get the number of elements (batch size)
        m = tf.shape(tensor)[0]  # Number of embeddings

        # Number of unique pairs = m choose 2 = m*(m-1)/2
        n_pairs = tf.cast(m * (m - 1) / 2, dtype=tensor.dtype)

        # Early exit if there's only 1 element (no pairs)
        if m == 1:
            return tf.constant(0.0, dtype=tensor.dtype), tf.constant(0.0, dtype=tensor.dtype)

        # Create indices for upper triangular matrix (excluding diagonal)
        i, j = tf.meshgrid(tf.range(m), tf.range(m))
        mask = i < j  # Upper triangular mask

        # Gather unique pairs
        elements_i = tf.gather(tensor, tf.boolean_mask(i, mask))
        elements_j = tf.gather(tensor, tf.boolean_mask(j, mask))

        # Calculate L2 distances between unique pairs
        distances = tf.norm(elements_i - elements_j, axis=-1)

        # Sum all unique pairwise distances
        sum_distances = tf.reduce_sum(distances)

        return sum_distances, n_pairs

    def get_feature_alignment_loss(self, feature_embeddings, features):
        """
        calculates the feature alignment loss
        :param feature_embeddings: dict, 各个特征下的embedding
        :param features: dict, 模型输入特征
        :return feature_alignment_loss: tensor, feature alignment loss
        """
        total_distance = 0.0
        total_pairs = 0.0

        for feature in feature_embeddings:
            per_feature_embeddings = feature_embeddings[feature]
            sum_distances, n_pairs = self.sum_unique_pairwise_distances(per_feature_embeddings[0])
            total_distance += sum_distances
            total_pairs += n_pairs

        # Avoid division by zero if there are no pairs (batch_size = 1)
        feature_alignment_loss = tf.cond(
            total_pairs > 0,
            lambda: total_distance / total_pairs,
            lambda: tf.constant(0.0, dtype=total_distance.dtype))

        return feature_alignment_loss

    def get_field_uniformity_loss(self, feature_embeddings, features):
        """
        calculates the field uniformity loss
        :param feature_embeddings: dict, 各个特征下的embedding
        :param features: dict, 模型输入特征
        :return field_uniformity_loss: tensor, field uniformity loss
        """
        # The cosine similarity between embeddings of two features
        feature_cos_sim_list = []

        # Normalise feature embeddings
        feature_normalised_list = dict()
        for feature in feature_embeddings:
            # normalisation done to facilitate the calculation of field uniformity
            per_feature_embeddings = feature_embeddings[feature]
            normalised_embeddings = tf.math.l2_normalize([per_feature_embeddings[0]])
            feature_normalised_list[feature] = normalised_embeddings

        for feature_i in feature_normalised_list:
            feature_is_cos_sim_list = []
            for feature_j in feature_normalised_list:
                if feature_i != feature_j:
                    feature_i_j_cos_sim = tf.reduce_sum(
                        feature_normalised_list[feature_i] * feature_normalised_list[feature_j])
                    feature_is_cos_sim_list.append(feature_i_j_cos_sim)

            # Flatten and concatenate to get a list of cosine similarity
            flattened = [tf.reshape(t, [-1]) for t in feature_is_cos_sim_list]  # Convert all to 1D
            combined = tf.concat(flattened, axis=0)
            feature_cos_sim_list.append(tf.reduce_sum(combined) / len(feature_is_cos_sim_list))

        flattened = [tf.reshape(t, [-1]) for t in feature_cos_sim_list]  # Convert all to 1D
        combined = tf.concat(flattened, axis=0)
        field_uniformity_loss = tf.reduce_sum(combined) / len(features)

        return field_uniformity_loss

    def get_logits(self, features, train_flag, **kwargs):
        """
        :param features:
        :param train_flag:
        :param kwargs:
        :return:
        """
        if self.struct not in ['Parallel', 'Stacked', 'Combined']:
            raise ValueError('struct should be "Parallel", "Stacked" or "Combined"')

        # base CTR logits for CTR loss
        logits = super().get_logits(features, train_flag, **kwargs)

        # mask features based on feature "is_personalized"
        # other masking methods (random, dimension, feature) will be implemented in future versions
        if self.use_cl_mask and self.mask_type == 'Personalisation':
            # h1
            h1_logits = logits

            # extract all the non-personalised feature names
            for feature in features:
                if feature not in self.personalization_feature_list:
                    self.non_personalised_features[feature] = features[feature]
                    if feature in self.dense_name_list:
                        self.non_personalised_dense_features[feature] = features[feature]

            # extract non-personalised embeddings
            non_personalised_kwargs = kwargs.copy()
            if len(self.non_personalised_features) > 0:
                non_personalised_kwargs["features_to_select"] = self.non_personalization_feature_list

            # get the original config
            original_use_domain_aware_structure = self.use_domain_aware_structure
            original_use_gated_cross = self.use_gated_cross
            original_dense_single_num = self.dense_single_num

            # change config
            self.use_domain_aware_structure = False
            self.use_gated_cross = False
            self.dense_single_num = len(self.non_personalised_dense_features)
            
            # h2
            h2_logits = super().get_logits(self.non_personalised_features, train_flag, **non_personalised_kwargs)

            # set back to original config
            self.use_domain_aware_structure = original_use_domain_aware_structure
            self.use_gated_cross = original_use_gated_cross
            self.dense_single_num = original_dense_single_num

        # this is used to calculate the feature alignment and field uniformity
        if self.field_uniformity_loss_weight != 0 or self.feature_alignment_loss_weight != 0:
            feature_embeddings = self.get_feature_embeddings(features, **kwargs)
            # FEATURE ALIGNMENT
            self.feature_alignment_loss = self.get_feature_alignment_loss(feature_embeddings, features)
            # FIELD UNIFORMITY
            self.field_uniformity_loss = self.get_field_uniformity_loss(feature_embeddings, features)

        if self.distance_loss_weight != 0:
            # the L1 distance between h1 and h2 logits
            num_samples = tf.cast(tf.shape(h2_logits)[0], dtype=tf.float32)
            self.logits_dist = [tf.reduce_sum(tf.abs(h2_logits[0] - h1_logits[0]), axis=-1) / num_samples]

        return logits

    def _get_loss(self, logits, labels, weights=1.0, **kwargs):

        base_loss = super()._get_loss(logits, labels, weights=weights, **kwargs)

        wb = self.base_loss_weight  # weight for base CTR loss
        wfu = self.feature_alignment_loss_weight  # weight for field uniformity
        wfa = self.field_uniformity_loss_weight  # weight for feature alignment
        wdl = self.distance_loss_weight  # weight for CL distance loss

        loss = wb * base_loss
        if self.feature_alignment_loss is not None:
            logging.info("feature alignment loss used")
            loss += wfa * self.feature_alignment_loss
        if self.field_uniformity_loss is not None:
            loss += wfu * self.field_uniformity_loss
            logging.info("field uniformity loss used")
        if self.logits_dist is not None:
            # finds the relationship between the distance logits and the prediction results.
            distance_loss = bce_loss(self.logits_dist, labels)
            logging.info("distance loss used")
            loss += wdl * distance_loss

        return loss
