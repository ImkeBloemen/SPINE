"""
Module to generate diverse counterfactual explanations based on tensorflow 2.x
The ML models should be differentiable. Examples of these models are deep neural networks, linear/logistic regression, decision trees, etc.
Code is adapted and modified from the DiCE package (https://github.com/interpretml/DiCE/tree/main)
"""
import copy
import random
import timeit

import numpy as np
import pandas as pd
import tensorflow as tf
from math import e
from joblib import load

from VAE_DBS.models.DiCE.dice_ml import diverse_counterfactuals as exp
from VAE_DBS.models.DiCE.dice_ml.counterfactual_explanations import CounterfactualExplanations
from VAE_DBS.models.DiCE.dice_ml.explainer_interfaces.explainer_base import ExplainerBase

class DiceTensorFlow2(ExplainerBase):

    def __init__(self, data_interface, model_interface, autoencoder=None, encoder=None, multi_class=False):
        """Init method
        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.

        """
        # initiating data related parameters
        super().__init__(data_interface)
        # initializing model related variables
        self.model = model_interface
        self.model.load_model()  # loading trained model
        # self.model.transformer.feed_data_params(data_interface)
        # self.model.transformer.initialize_transform_func() #Currently using FunctionTransformer from sklearn.
        
        # temp data to create some attributes like encoded feature names
        # if hasattr(self.data_interface, "data_df"):
        #     temp_ohe_data = self.model.transformer.transform(self.data_interface.data_df.iloc[[0]])
        # else:
        #     temp_ohe_data = None
        # Encode the feature labels without using model.transformer
        if hasattr(self.data_interface, "data_df"):
            temp_ohe_data = self.data_interface.one_hot_encode_data(self.data_interface.data_df.iloc[[0]])
        else:
            temp_ohe_data = None
        self.data_interface.create_ohe_params(temp_ohe_data)
        self.minx, self.maxx, self.encoded_categorical_feature_indexes, self.encoded_continuous_feature_indexes, \
            self.cont_minx, self.cont_maxx, self.cont_precisions = self.data_interface.get_data_params_for_gradient_dice()

        # number of output nodes of ML model
        self.num_output_nodes = self.model.get_num_output_nodes(len(self.data_interface.ohe_encoded_feature_names)).shape[1]

        # variables required to generate CFs - see generate_counterfactuals() for more info
        self.cfs = []
        self.features_to_vary = []
        self.cf_init_weights = []  # total_CFs, algorithm, features_to_vary
        self.loss_weights = []  # yloss_type, diversity_loss_type, feature_weights
        self.feature_weights_input = ''
        self.hyperparameters = [1, 1, 1, 1]  # prediction_weight, proximity_weight, diversity_weight, categorical_penalty
        self.optimizer_weights = []  # optimizer, learning_rate
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.multi_class = multi_class

    def generate_counterfactuals(self, query_instance, total_CFs, desired_class="opposite", prediction_weight=1, proximity_weight=0.5,
                                 diversity_weight=1.0, categorical_penalty=0.1, ae_weight=0.7, proto_weight=0.7, class_prototypes=None, algorithm="DiverseCF",
                                 features_to_vary="all", permitted_range=None, yloss_type="hinge_loss",
                                 diversity_loss_type="dpp_style:inverse_dist", feature_weights="inverse_mad",
                                 optimizer="tensorflow:adam", learning_rate=0.05, min_iter=500, max_iter=5000,
                                 project_iter=0, loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False,
                                 init_near_query_instance=True, tie_random=False, stopping_threshold=0.5,
                                 posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear", limit_steps_ls=10000,
                                 pred_threshold=0.0001, pred_converge_steps=15, pred_converge_maxiter=3):
        """Generates diverse counterfactual explanations

        :param query_instance: Test point of interest. A dictionary of feature names and values or a single row dataframe
        :param total_CFs: Total number of counterfactuals required.
        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the
                              outcome class of query_instance for binary classification. --> changed to multi-class
        :param multi_class: Boolean to indicate if the model is multi-class. Default is False.
        :param prediction_weight: A positive float. Larger this weight, more close the counterfactuals are to the desired prediction.
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to
                                 the query_instance.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
        :param categorical_penalty: A positive float. A weight to ensure that all levels of a categorical
                                    variable sums to 1.
        :param ae_weight: A positive float. A weight to ensure that the counterfactuals are close to the original.
        :param proto_weight: A positive float. A weight to ensure that the counterfactuals are close to the prototype.
        :param algorithm: Counterfactual generation algorithm. Either "DiverseCF" or "RandomInitCF".
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param permitted_range: Dictionary with continuous feature names as keys and permitted min-max range in list as values.
                               Defaults to the range inferred from training data. If None, uses the parameters initialized
                               in data_interface.
        :param yloss_type: Metric for y-loss of the optimization function. Takes "l2_loss" or "log_loss" or "hinge_loss".
        :param diversity_loss_type: Metric for diversity loss of the optimization function.
                                    Takes "avg_dist" or "dpp_style:inverse_dist".
        :param feature_weights: Either "inverse_mad" or a dictionary with feature names as keys and corresponding
                                weights as values. Default option is "inverse_mad" where the weight for a continuous feature
                                is the inverse of the Median Absolute Devidation (MAD) of the feature's values in the training
                                set; the weight for a categorical feature is equal to 1 by default.
        :param optimizer: Tensorflow optimization algorithm. Currently tested only with "tensorflow:adam".

        :param learning_rate: Learning rate for optimizer.
        :param min_iter: Min iterations to run gradient descent for.
        :param max_iter: Max iterations to run gradient descent for.
        :param project_iter: Project the gradients at an interval of these many iterations.
        :param loss_diff_thres: Minimum difference between successive loss values to check convergence.
        :param loss_converge_maxiter: Maximum number of iterations for loss_diff_thres to hold to declare convergence.
                                      Defaults to 1, but we assigned a more conservative value of 2 in the paper.
        :param verbose: Print intermediate loss value.
        :param init_near_query_instance: Boolean to indicate if counterfactuals are to be initialized near query_instance.
        :param tie_random: Used in rounding off CFs and intermediate projection.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary".
                                           Prefer binary search when a feature range is large
                                           (for instance, income varying from 10k to 1000k) and only if the features
                                           share a monotonic relationship with predicted outcome in the model.
        :param limit_steps_ls: Defines an upper limit for the linear search step in the posthoc_sparsity_enhancement
        :param pred_threshold: Threshold for prediction change to stop optimization.
        :param pred_converge_steps: Number of steps to check for prediction convergence.
        :param pred_converge_maxiter: Maximum number of iterations for prediction convergence.

        :return: A CounterfactualExamples object to store and visualize the resulting counterfactual explanations
                (see diverse_counterfactuals.py).

        """
        # check feature MAD validity and throw warnings
        if feature_weights == "inverse_mad":
            self.data_interface.get_valid_mads(display_warnings=False, return_mads=False)

        # check permitted range for continuous features
        if permitted_range is not None:
            # if not self.data_interface.check_features_range(permitted_range):
            #     raise ValueError(
            #         "permitted range of features should be within their original range")
            # else:
            self.data_interface.permitted_range = permitted_range
            self.minx, self.maxx = self.data_interface.get_minx_maxx(normalized=True)
            self.cont_minx = []
            self.cont_maxx = []
            for feature in self.data_interface.continuous_feature_names:
                self.cont_minx.append(self.data_interface.permitted_range[feature][0])
                self.cont_maxx.append(self.data_interface.permitted_range[feature][1])

        # if([total_CFs, algorithm, features_to_vary] != self.cf_init_weights):
        self.do_cf_initializations(total_CFs, algorithm, features_to_vary)
        # if [yloss_type, diversity_loss_type, feature_weights] != self.loss_weights:
        self.do_loss_initializations(yloss_type, diversity_loss_type, feature_weights)
        if [prediction_weight, proximity_weight, diversity_weight, categorical_penalty] != self.hyperparameters:
            self.update_hyperparameters(prediction_weight, proximity_weight, diversity_weight, categorical_penalty, ae_weight, proto_weight)

        final_cfs_df, test_instance_df, final_cfs_df_sparse, intermediate_cfs, gradients, losses, intermediate_predictions = \
            self.find_counterfactuals(query_instance, desired_class, optimizer,
                                      learning_rate, min_iter, max_iter, project_iter,
                                      loss_diff_thres, loss_converge_maxiter, verbose,
                                      init_near_query_instance, tie_random, stopping_threshold,
                                      posthoc_sparsity_param, posthoc_sparsity_algorithm, limit_steps_ls, class_prototypes,
                                      pred_threshold, pred_converge_steps, pred_converge_maxiter)
    

        counterfactual_explanations = exp.CounterfactualExamples(
            data_interface=self.data_interface,
            intermediate_cfs=intermediate_cfs,
            gradients=gradients,
            losses=losses,
            intermediate_predictions=intermediate_predictions,
            final_cfs_df=final_cfs_df,
            test_instance_df=test_instance_df,
            final_cfs_df_sparse=final_cfs_df_sparse,
            posthoc_sparsity_param=posthoc_sparsity_param,
            desired_class=desired_class,
            iterations=self.max_iterations_run
            )

        return CounterfactualExplanations(cf_examples_list=[counterfactual_explanations], intermediate_cfs=intermediate_cfs, gradients=gradients, losses=losses, intermediate_predictions=intermediate_predictions, iterations=self.max_iterations_run)

    # def predict_fn(self, input_instance):
    #     """prediction function"""
    #     temp_preds = self.model.get_output(input_instance).numpy()
    #     if self.multi_class == False:
    #         return np.array([preds[(self.num_output_nodes-1):] for preds in temp_preds], dtype=np.float32)
    #     else:
    #         return temp_preds

    def predict_fn(self, input_instance):
        """Prediction function that returns probabilities."""
        logits = self.model.get_output(input_instance)
        if self.multi_class:
            probs = tf.nn.softmax(logits).numpy()
            return probs[0]
        else:
            probs = tf.nn.sigmoid(logits).numpy().flatten()
            return probs[0]


    # def predict_fn_for_sparsity(self, input_instance):
    #     """prediction function for sparsity correction"""
    #     input_instance = self.model.transformer.transform(input_instance).to_numpy()
    #     return self.predict_fn(tf.constant(input_instance, dtype=tf.float32))

    def predict_fn_for_sparsity(self, input_instance):
        """prediction function for sparsity correction"""
        # input_instance = self.model.transformer.transform(input_instance).to_numpy()
        # input_instance = self.transformer.transform(input_instance) #not needed bc data already scaled?
        predictions = self.predict_fn(tf.constant(input_instance, dtype=tf.float32))
        return predictions
        
        # Check if the output is binary or multiclass
        # if predictions.shape[-1] == 1:
        # if predictions.ndim == 0:
        #     # Scalar prediction
        #     # Binary classification
        #     return tf.sigmoid(predictions).numpy()
        # else:
        #     # Multiclass classification
        #     return tf.nn.softmax(predictions).numpy()

    def do_cf_initializations(self, total_CFs, algorithm, features_to_vary):
        """Intializes CFs and other related variables."""

        self.cf_init_weights = [total_CFs, algorithm, features_to_vary]

        if algorithm == "RandomInitCF":
            # no. of times to run the experiment with random inits for diversity
            self.total_random_inits = total_CFs
            self.total_CFs = 1          # size of counterfactual set
        else: #we want this
            self.total_random_inits = 0
            self.total_CFs = total_CFs  # size of counterfactual set

        # freeze those columns that need to be fixed
        if features_to_vary != self.features_to_vary:
            self.features_to_vary = features_to_vary
        self.feat_to_vary_idxs = self.data_interface.get_indexes_of_features_to_vary(features_to_vary=features_to_vary)
        self.freezer = tf.constant([1.0 if ix in self.feat_to_vary_idxs else 0.0 for ix in range(len(self.minx[0]))])

        # CF initialization
        # if len(self.cfs) != self.total_CFs:
        #     self.cfs = []
        #     for _ in range(self.total_CFs):
        #         one_init = [[]]
        #         for jx in range(self.minx.shape[1]):
        #             one_init[0].append(np.random.uniform(self.minx[0][jx], self.maxx[0][jx]))
        #         self.cfs.append(tf.Variable(one_init, dtype=tf.float32))

        self.cfs = []
        for _ in range(self.total_CFs):
            one_init = [[]]
            for jx in range(self.minx.shape[1]):
                one_init[0].append(np.random.uniform(self.minx[0][jx], self.maxx[0][jx]))
            self.cfs.append(tf.Variable(one_init, dtype=tf.float32))

    def do_loss_initializations(self, yloss_type, diversity_loss_type, feature_weights):
        """Intializes variables related to main loss function"""

        self.loss_weights = [yloss_type, diversity_loss_type, feature_weights]

        # define the loss parts
        self.yloss_type = yloss_type
        self.diversity_loss_type = diversity_loss_type

        # define feature weights
        if feature_weights != self.feature_weights_input:
            self.feature_weights_input = feature_weights
            if feature_weights == "inverse_mad":
                normalized_mads = self.data_interface.get_valid_mads(normalized=True)
                feature_weights = {}
                for feature in normalized_mads:
                    feature_weights[feature] = round(1/normalized_mads[feature], 2)

            feature_weights_list = []
            for feature in self.data_interface.ohe_encoded_feature_names:
                if feature in feature_weights:
                    feature_weights_list.append(feature_weights[feature])
                else:
                    feature_weights_list.append(1.0)
            self.feature_weights_list = tf.constant([feature_weights_list], dtype=tf.float32)

    def update_hyperparameters(self, prediction_weight, proximity_weight, diversity_weight, categorical_penalty, ae_weight, proto_weight):
        """Update hyperparameters of the loss function"""

        self.hyperparameters = [prediction_weight, proximity_weight, diversity_weight, categorical_penalty, ae_weight, proto_weight]
        self.prediction_weight = prediction_weight
        self.proximity_weight = proximity_weight
        self.diversity_weight = diversity_weight
        self.categorical_penalty = categorical_penalty
        self.ae_weight = ae_weight
        self.proto_weight = proto_weight

    def do_optimizer_initializations(self, optimizer, learning_rate):
        """Initializes gradient-based TensorFLow optimizers."""
        opt_method = optimizer.split(':')[1]

        # optimizater initialization
        if opt_method == "adam":
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        elif opt_method == "rmsprop":
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate)

    def compute_yloss(self):
        """Computes the first part (y-loss) of the loss function."""
        yloss = 0.0
        for i in range(self.total_CFs):
            if self.multi_class:
                logits = self.model.get_output(self.cfs[i])
                temp_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=[self.target_cf_class], logits=logits)
                yloss += temp_loss[0]
            else:
                if self.yloss_type == "l2_loss":
                    temp_loss = tf.pow((self.model.get_output(self.cfs[i]) - self.target_cf_class), 2)
                    temp_loss = temp_loss[:, (self.num_output_nodes-1):][0][0]
                elif self.yloss_type == "log_loss":
                    temp_logits = tf.compat.v1.log((tf.abs(
                        self.model.get_output(
                            self.cfs[i]) - 0.000001))/(1 - tf.abs(self.model.get_output(self.cfs[i]) - 0.000001)))
                    temp_logits = temp_logits[:, (self.num_output_nodes-1):]
                    temp_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=temp_logits, labels=self.target_cf_class)[0][0]
                elif self.yloss_type == "hinge_loss":
                    temp_logits = tf.compat.v1.log((tf.abs(
                        self.model.get_output(
                            self.cfs[i]) - 0.000001))/(1 - tf.abs(self.model.get_output(self.cfs[i]) - 0.000001)))
                    temp_logits = temp_logits[:, (self.num_output_nodes-1):]
                    temp_loss = tf.compat.v1.losses.hinge_loss(
                        logits=temp_logits[0][0], labels=self.target_cf_class)
                elif self.yloss_type == "boundary_loss":
                    temp_loss = tf.abs(self.model.get_output(self.cfs[i]) - self.stopping_threshold)
                    temp_loss = temp_loss[:, (self.num_output_nodes-1):][0][0]

                yloss += temp_loss

        return yloss/self.total_CFs

    def compute_dist(self, x_hat, x1):
        """Compute weighted distance between two vectors."""
        return tf.reduce_sum(tf.multiply((tf.abs(x_hat - x1)), self.feature_weights_list))

    # def compute_proximity_loss(self):
    #     """Compute the second part (distance from x1) of the loss function."""
    #     proximity_loss = 0.0
    #     for i in range(self.total_CFs):
    #         proximity_loss += self.compute_dist(self.cfs[i], self.x1)
    #     return proximity_loss/tf.cast((tf.multiply(len(self.minx[0]), self.total_CFs)), dtype=tf.float32)
    
    def compute_proximity_loss(self):
        proximity_loss = tf.reduce_sum([self.compute_dist(self.cfs[i], self.x1) for i in range(self.total_CFs)])
        return proximity_loss / (len(self.minx[0]) * self.total_CFs)

    def dpp_style(self, submethod):
        """Computes the DPP of a matrix."""
        det_entries = []
        if submethod == "inverse_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_temp_entry = tf.divide(1.0, tf.add(
                        1.0, self.compute_dist(self.cfs[i], self.cfs[j])))
                    if i == j:
                        det_temp_entry = tf.add(det_temp_entry, 0.0001)
                    det_entries.append(det_temp_entry)

        elif submethod == "exponential_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_temp_entry = tf.divide(1.0, tf.exp(
                        self.compute_dist(self.cfs[i], self.cfs[j])))
                    det_entries.append(det_temp_entry)

        det_entries = tf.reshape(det_entries, [self.total_CFs, self.total_CFs])
        diversity_loss = tf.compat.v1.matrix_determinant(det_entries)
        return diversity_loss

    def compute_diversity_loss(self):
        """Computes the third part (diversity) of the loss function."""
        if self.total_CFs == 1:
            return tf.constant(0.0)

        if "dpp" in self.diversity_loss_type:
            submethod = self.diversity_loss_type.split(':')[1]
            return tf.reduce_sum(self.dpp_style(submethod))
        elif self.diversity_loss_type == "avg_dist":
            diversity_loss = 0.0
            count = 0.0
            # computing pairwise distance and transforming it to normalized similarity
            for i in range(self.total_CFs):
                for j in range(i+1, self.total_CFs):
                    count += 1.0
                    diversity_loss += 1.0/(1.0 + self.compute_dist(self.cfs[i], self.cfs[j]))

            return 1.0 - (diversity_loss/count)

    # def compute_regularization_loss(self):
    #     """Adds a linear equality constraints to the loss functions - to ensure all levels
    #        of a categorical variable sums to one"""
    #     regularization_loss = 0.0
    #     for i in range(self.total_CFs):
    #         for v in self.encoded_categorical_feature_indexes:
    #             regularization_loss += tf.pow((tf.reduce_sum(self.cfs[i][0, v[0]:v[-1]+1]) - 1.0), 2)

        # return regularization_loss

    def compute_regularization_loss(self):
        # Stack all CFs into a single tensor
        cfs_tensor = tf.stack([cf[0] for cf in self.cfs], axis=0)
        regularization_loss = 0.0
        for v in self.encoded_categorical_feature_indexes:
            # Extract the categorical vectors for all CFs
            cat_vectors = cfs_tensor[:, v[0]:v[-1]+1]
            # Compute the sum along the categorical dimensions
            cat_sums = tf.reduce_sum(cat_vectors, axis=1)
            # Compute the regularization loss for all CFs
            regularization_loss += tf.reduce_sum(tf.square(cat_sums - 1.0))
        return regularization_loss
    
    def compute_ae_loss(self):
        loss_ae = 0.0
        for i in range(self.total_CFs):
            ae_output = self.autoencoder(self.cfs[i])
            loss_ae += tf.reduce_mean(tf.square(ae_output - self.cfs[i]))
        return loss_ae / self.total_CFs

    def compute_proto_loss(self):
        loss_proto = 0.0
        for i in range(self.total_CFs):
            cf_encoded = self.encoder(self.cfs[i])
            loss_proto += tf.reduce_sum(tf.square(cf_encoded - self.prototype_target_class))
        return loss_proto / self.total_CFs


    def compute_loss(self):
        """Computes the overall loss"""
        self.yloss = self.compute_yloss()
        self.proximity_loss = self.compute_proximity_loss() if self.proximity_weight > 0 else 0.0
        self.diversity_loss = self.compute_diversity_loss() if self.diversity_weight > 0 else 0.0
        self.regularization_loss = self.compute_regularization_loss()

        #New loss components:
        self.loss_ae = self.compute_ae_loss()
        self.loss_proto = self.compute_proto_loss()

        self.loss = (self.prediction_weight * self.yloss) + \
                (self.proximity_weight * self.proximity_loss) - \
                (self.diversity_weight * self.diversity_loss) + \
            (self.categorical_penalty * self.regularization_loss) + \
            (self.ae_weight * self.loss_ae) + \
            (self.proto_weight * self.loss_proto)
        return self.loss

    def initialize_CFs(self, query_instance, init_near_query_instance=True):
        """Initialize counterfactuals."""
        for n in range(self.total_CFs):
            one_init = []
            for i in range(len(self.minx[0])):
                if i in self.feat_to_vary_idxs: #all features
                    if init_near_query_instance: #True
                        init_value = query_instance[0][i] + (n * 1e-5)
                        init_value = np.clip(init_value, self.minx[0][i], self.maxx[0][i])
                        one_init.append(init_value)
                        # one_init.append(query_instance[0][i]+(n*1*e-5))
                    else:
                        one_init.append(np.random.uniform(self.minx[0][i], self.maxx[0][i]))
                else:
                    one_init.append(query_instance[0][i])
            one_init = np.array([one_init], dtype=np.float32)
            self.cfs[n].assign(one_init)

    def round_off_cfs(self, assign=False):
        """function for intermediate projection of CFs."""
        temp_cfs = []
        for index, tcf in enumerate(self.cfs):
            cf = tcf.numpy()
            for i, v in enumerate(self.encoded_continuous_feature_indexes):
                # continuous feature in orginal scale
                org_cont = (cf[0, v]*(self.cont_maxx[i] - self.cont_minx[i])) + self.cont_minx[i]
                org_cont = round(org_cont, self.cont_precisions[i])  # rounding off
                normalized_cont = (org_cont - self.cont_minx[i])/(self.cont_maxx[i] - self.cont_minx[i])
                cf[0, v] = normalized_cont  # assign the projected continuous value

            for v in self.encoded_categorical_feature_indexes:
                maxs = np.argwhere(
                    cf[0, v[0]:v[-1]+1] == np.amax(cf[0, v[0]:v[-1]+1])).flatten().tolist()
                if len(maxs) > 1:
                    if self.tie_random:
                        ix = random.choice(maxs)
                    else:
                        ix = maxs[0]
                else:
                    ix = maxs[0]
                for vi in range(len(v)):
                    if vi == ix:
                        cf[0, v[vi]] = 1.0
                    else:
                        cf[0, v[vi]] = 0.0

            temp_cfs.append(cf)
            if assign:
                self.cfs[index].assign(temp_cfs[index])

        if assign:
            return None
        else:
            return temp_cfs

    def stop_loop(self, itr, loss_diff):

        # do GD for min iterations
        if itr < self.min_iter:
            return False
        
        # intermediate projections
        # if self.project_iter > 0 and itr > 0:
        #     if itr % self.project_iter == 0:
        #         self.round_off_cfs(assign=True)

        # Check if CFs meet the desired prediction
        # temp_cfs = self.round_off_cfs(assign=False)   

        if self.multi_class:
            # test_preds = [self.predict_fn(tf.constant(cf, dtype=tf.float32)) for cf in temp_cfs]
            test_preds = [self.predict_fn(tf.constant(cf, dtype=tf.float32)) for cf in self.cfs]
            predicted_classes = [np.argmax(pred) for pred in test_preds]
            if int(predicted_classes[0]) == int(self.target_cf_class) and test_preds[0][int(self.target_cf_class)] > self.stopping_threshold:
                self.converged = True
                print("Converged because all CFs have reached the target class and probability.")
                return True
        else:
            # test_preds = [self.predict_fn(tf.constant(cf, dtype=tf.float32)) for cf in temp_cfs]
            test_preds = [self.predict_fn(tf.constant(cf, dtype=tf.float32)) for cf in self.cfs]
            if self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in test_preds):
                self.converged = True
                return True
            elif self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in test_preds):
                self.converged = True
                return True
            
        # stop GD if max iter is reached
        if itr >= self.max_iter:
            print("Stopping optimization because maximum iterations reached.")
            return True
        

        # Early stopping based on per-CF prediction stagnation
        if self.boundary_crossed:
            if all(counter >= self.pred_converge_maxiter for counter in self.pred_converge_iters):
                print(f"Stopping optimization because all CFs have stagnated for {self.pred_converge_maxiter} consecutive checks.")
                return True

        # Check for loss convergence
        if loss_diff <= self.loss_diff_thres:
            self.loss_converge_iter += 1
            if self.loss_converge_iter < self.loss_converge_maxiter:
                return False
            else:
                print("Stopping optimization because loss has converged.")
                return True  
        else:
            self.loss_converge_iter = 0
                # temp_cfs = self.round_off_cfs(assign=False)
                # test_preds = [self.predict_fn(tf.constant(cf, dtype=tf.float32))[0] for cf in temp_cfs]

                # if self.multi_class:
                #     predicted_classes = [np.argmax(pred) for pred in test_preds]

                #     if all(pred_class == self.target_cf_class for pred_class in predicted_classes):
                #         self.converged = True
                #         return True


                # elif self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in test_preds):
                #     self.converged = True
                #     return True
                # elif self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in test_preds):
                #     self.converged = True
                #     return True
                # else:
                #     return False

        return False


    def find_counterfactuals(self, query_instance, desired_class, optimizer, learning_rate, min_iter,
                             max_iter, project_iter, loss_diff_thres, loss_converge_maxiter, verbose,
                             init_near_query_instance, tie_random, stopping_threshold, posthoc_sparsity_param,
                             posthoc_sparsity_algorithm, limit_steps_ls, class_prototypes,
                             pred_threshold, pred_converge_steps, pred_converge_maxiter):
        """Finds counterfactuals by gradient-descent."""

        # query_instance = self.model.transformer.transform(query_instance).to_numpy()
        # query_instance = self.transformer.transform(query_instance) #not needed bc data already scaled

        self.x1 = tf.constant(query_instance, dtype=tf.float32)

        # find the predicted value of query_instance
        test_pred = self.predict_fn(tf.constant(query_instance, dtype=tf.float32)) #prediction is slightly different?
        print(f"Initial prediction for query instance: {test_pred}")
        if self.multi_class:
            if desired_class == "opposite":
                desired_class = (np.argmax(test_pred) + 1) % self.num_output_nodes
            else:
                self.target_cf_class = int(desired_class)
                print(f"Target class defined: {self.target_cf_class}")
        else:
            if desired_class == "opposite":
                desired_class = 1.0 - round(test_pred)
            # self.target_cf_class = np.array([[desired_class]], dtype=np.float32)
            self.target_cf_class = float(desired_class)

        # Ensure desired_class is the correct type (float or int)
        desired_class = int(desired_class) if self.multi_class else float(desired_class)
        # Set the target class prototype
        self.prototype_target_class = class_prototypes[desired_class]

        # Initialize a small learning rate and define the step increase factor
        current_learning_rate = learning_rate
        learning_rate_increase_factor = 1.2  # Increase rate after boundary is crossed
        self.boundary_crossed = False
        self.do_optimizer_initializations(optimizer, current_learning_rate)

        self.min_iter = min_iter
        self.max_iter = max_iter
        self.project_iter = project_iter
        self.loss_diff_thres = loss_diff_thres
        # no. of iterations to wait to confirm that loss has converged
        self.loss_converge_maxiter = loss_converge_maxiter
        self.loss_converge_iter = 0
        self.converged = False

        self.stopping_threshold = stopping_threshold
        self.boundary_threshold = 0.5
        if self.multi_class and self.num_output_nodes == 2:
            self.stopping_threshold = 0.6
        elif self.multi_class and self.num_output_nodes > 2:
            self.stopping_threshold = (1.0/self.num_output_nodes) + 0.1
        elif self.target_cf_class == 0 and self.stopping_threshold > 0.5:
            self.stopping_threshold = 0.4
        elif self.target_cf_class == 1 and self.stopping_threshold < 0.5:
            self.stopping_threshold = 0.6

        # to resolve tie - if multiple levels of an one-hot-encoded categorical variable take value 1
        self.tie_random = tie_random

        # running optimization steps
        start_time = timeit.default_timer()
        self.final_cfs = []

        # looping the find CFs depending on whether its random initialization or not
        loop_find_CFs = self.total_random_inits if self.total_random_inits > 0 else 1

        # variables to backup best known CFs so far in the optimization process -
        # if the CFs dont converge in max_iter iterations, then best_backup_cfs is returned.
        self.best_backup_cfs = [0]*max(self.total_CFs, loop_find_CFs)
        self.best_backup_cfs_preds = [0]*max(self.total_CFs, loop_find_CFs)
        self.min_dist_from_threshold = [100]*loop_find_CFs  # for backup CFs

        intermediate_cfs = []
        losses = []
        gradients = []
        intermediate_predictions = []

        # Initialize variables for early stopping based on prediction changes
        self.pred_threshold = pred_threshold
        self.pred_converge_steps = pred_converge_steps
        self.pred_converge_maxiter = pred_converge_maxiter
        self.pred_histories = [[] for _ in range(self.total_CFs)]
        self.pred_converge_iters = [0 for _ in range(self.total_CFs)]       

        for loop_ix in range(loop_find_CFs):
            # CF init
            if self.total_random_inits > 0:
                self.initialize_CFs(query_instance, False)
            else:
                self.initialize_CFs(query_instance, init_near_query_instance)

            # initialize optimizer
            self.do_optimizer_initializations(optimizer, learning_rate)

            iterations = 0
            loss_diff = 1.0
            #prev_loss = 0.0
            prev_loss = np.inf

            with tf.GradientTape() as tape:
                    loss_value = self.compute_loss()

            losses.append(loss_value.numpy())

            # get gradients
            with tf.device('/CPU:0'):
                grads = tape.gradient(loss_value, self.cfs)
            gradients.append(grads)

            # freeze features other than feat_to_vary_idxs
            # for ix in range(self.total_CFs):
            #     grads[ix] *= self.freezer

            # apply gradients and update the variables
            self.optimizer.apply_gradients(zip(grads, self.cfs))

            # Save the intermediate CFs at each iteration
            current_cfs = [cf.numpy() for cf in self.cfs]
            intermediate_cfs.append(copy.deepcopy(current_cfs))

            # projection step
            for j in range(0, self.total_CFs):
                temp_cf = self.cfs[j].numpy()
                clip_cf = np.clip(temp_cf, self.minx, self.maxx)  # clipping
                clip_cf = np.add(clip_cf, np.array([np.zeros([self.minx.shape[1]])]))
                self.cfs[j].assign(clip_cf)

            if verbose and (iterations % 50 == 0):
                print('step %d,  loss=%g' % (iterations+1, loss_value))

            loss_diff = abs(loss_value - prev_loss)
            prev_loss = loss_value
            
            iterations += 1

            # backing up CFs if they are valid
            # temp_cfs_stored = self.round_off_cfs(assign=False)
            if self.multi_class:
                # test_preds_stored = [self.predict_fn(tf.constant(cf, dtype=tf.float32)) for cf in temp_cfs_stored]
                test_preds_stored = [self.predict_fn(tf.constant(cf, dtype=tf.float32)) for cf in self.cfs]
            else:
                # test_preds_stored = [self.predict_fn(tf.constant(cf, dtype=tf.float32)) for cf in temp_cfs_stored]
                test_preds_stored = [self.predict_fn(tf.constant(cf, dtype=tf.float32)) for cf in self.cfs]
            intermediate_predictions.append(test_preds_stored)

            # Update pred_histories and pred_converge_iters BEFORE calling stop_loop
            for idx, pred in enumerate(test_preds_stored):
                if self.multi_class:
                    current_pred = pred[self.target_cf_class]  # Probability of the target class
                else:
                    current_pred = pred  # Scalar probability
                current_pred = round(current_pred, 6)
                self.pred_histories[idx].append(current_pred)
                # Keep only the last pred_converge_steps predictions
                if len(self.pred_histories[idx]) > self.pred_converge_steps:
                    self.pred_histories[idx].pop(0)
                # Compute the maximum absolute change if enough history is available
                if len(self.pred_histories[idx]) >= self.pred_converge_steps:
                    pred_diffs = [abs(self.pred_histories[idx][i] - self.pred_histories[idx][i - 1])
                                for i in range(1, len(self.pred_histories[idx]))]
                    max_pred_change = max(pred_diffs)
                    if max_pred_change <= self.pred_threshold:
                        self.pred_converge_iters[idx] += 1
                    else:
                        self.pred_converge_iters[idx] = 0

            # if self.multi_class:
            #         print(f"Iteration -1: CF probabilities = {[pred for pred in test_preds_stored]}")
            # else:                                                   
            #     print(f"Iteration -1: CF predictions = {test_preds_stored}")

            # Check for boundary crossing
            if not self.boundary_crossed:
                if self.multi_class:
                    predicted_classes = [np.argmax(pred[0]) for pred in test_preds_stored]
                    if any(pred_class == self.target_cf_class for pred_class in predicted_classes):
                        self.boundary_crossed = True
                        print("Desired class achieved.")
                else:
                    if ((self.target_cf_class == 0 and any(i <= self.boundary_threshold for i in test_preds_stored)) or
                        (self.target_cf_class == 1 and any(i >= self.boundary_threshold for i in test_preds_stored))):
                        self.boundary_crossed = True
                        print("Decision boundary crossed.")

            # Now, call stop_loop with the updated pred_converge_iters
            if self.stop_loop(iterations, loss_diff):
                break

            #Start the longer loop:
            while True:
                # compute loss and tape the variables history
                with tf.GradientTape() as tape:
                    loss_value = self.compute_loss()

                losses.append(loss_value.numpy())

                # get gradients
                with tf.device('/CPU:0'):
                    grads = tape.gradient(loss_value, self.cfs)
                gradients.append(grads)

                # freeze features other than feat_to_vary_idxs
                for ix in range(self.total_CFs):
                    grads[ix] *= self.freezer

                # apply gradients and update the variables
                self.optimizer.apply_gradients(zip(grads, self.cfs))

                # Save the intermediate CFs at each iteration
                current_cfs = [cf.numpy() for cf in self.cfs]
                # current_cfs_descaled = [self.transformer.inverse_transform(cf) for cf in current_cfs]
                intermediate_cfs.append(copy.deepcopy(current_cfs))

                # projection step
                for j in range(0, self.total_CFs):
                    temp_cf = self.cfs[j].numpy()
                    clip_cf = np.clip(temp_cf, self.minx, self.maxx)  # clipping
                    clip_cf = np.add(clip_cf, np.array([np.zeros([self.minx.shape[1]])]))
                    self.cfs[j].assign(clip_cf)

                if verbose and (iterations % 50 == 0):
                    print('step %d,  loss=%g' % (iterations+1, loss_value))

                loss_diff = abs(loss_value - prev_loss)
                prev_loss = loss_value
                # Note: Removed iterations += 1 here

                # backing up CFs if they are valid
                # temp_cfs_stored = self.round_off_cfs(assign=False)
                # print("cfs long loop", self.cfs)
                if self.multi_class:
                    # test_preds_stored = [self.predict_fn(tf.constant(cf, dtype=tf.float32)) for cf in temp_cfs_stored]
                    test_preds_stored = [self.predict_fn(tf.constant(cf, dtype=tf.float32)) for cf in self.cfs]
                else:
                    # test_preds_stored = [self.predict_fn(tf.constant(cf, dtype=tf.float32)) for cf in temp_cfs_stored]
                    test_preds_stored = [self.predict_fn(tf.constant(cf, dtype=tf.float32)) for cf in self.cfs]
                intermediate_predictions.append(test_preds_stored)

                # Update pred_histories and pred_converge_iters BEFORE calling stop_loop
                for idx, pred in enumerate(test_preds_stored):
                    if self.multi_class:
                        current_pred = pred[self.target_cf_class]  # Probability of the target class
                    else:
                        current_pred = pred  # Scalar probability
                    current_pred = round(current_pred, 6)
                    self.pred_histories[idx].append(current_pred)
                    # Keep only the last pred_converge_steps predictions
                    if len(self.pred_histories[idx]) > self.pred_converge_steps:
                        self.pred_histories[idx].pop(0)
                    # Compute the maximum absolute change if enough history is available
                    if len(self.pred_histories[idx]) >= self.pred_converge_steps:
                        pred_diffs = [abs(self.pred_histories[idx][i] - self.pred_histories[idx][i - 1])
                                    for i in range(1, len(self.pred_histories[idx]))]
                        max_pred_change = max(pred_diffs)
                        if max_pred_change <= self.pred_threshold:
                            self.pred_converge_iters[idx] += 1
                        else:
                            self.pred_converge_iters[idx] = 0

                # if self.multi_class:
                #     print(f"Iteration {iterations}: CF probabilities = {[pred for pred in test_preds_stored]}")
                # else:                                                   
                #     print(f"Iteration {iterations}: CF predictions = {test_preds_stored}")

                # Now, call stop_loop with the updated pred_converge_iters
                if self.stop_loop(iterations, loss_diff):
                    break

                # Now increment iterations
                iterations += 1

                # Check if counterfactuals have crossed the decision boundary
                if not self.boundary_crossed:
                    if self.multi_class:
                        predicted_classes = [np.argmax(pred[0]) for pred in test_preds_stored]
                        if any(pred_class == self.target_cf_class for pred_class in predicted_classes):
                            self.boundary_crossed = True
                            print("Desired class achieved. Gradually increasing the learning rate.")
                    else:
                        if ((self.target_cf_class == 0 and any(i <= self.boundary_threshold for i in test_preds_stored)) or
                            (self.target_cf_class == 1 and any(i >= self.boundary_threshold for i in test_preds_stored))):
                            self.boundary_crossed = True
                            print("Decision boundary crossed. Gradually increasing the learning rate.")

                # If boundary is crossed, gradually increase the learning rate
                if self.boundary_crossed and iterations % 50 == 0:
                    print(f"Boundary crossed. Gradually increasing the learning rate. #iterations: {iterations}")
                    if current_learning_rate > 0.1:  # Define a maximum cap
                        current_learning_rate = 0.1
                    else:
                        current_learning_rate *= learning_rate_increase_factor
                        self.do_optimizer_initializations(optimizer, current_learning_rate)

                # Backing up best CFs if they are valid
                # if self.multi_class:
                #     desired_class_probs = [pred[0][self.target_cf_class] for pred in test_preds_stored]
                #     if all(prob >= self.stopping_threshold for prob in desired_class_probs):
                #         avg_preds_dist = np.mean([abs(prob - self.stopping_threshold) for prob in desired_class_probs])
                #         if avg_preds_dist < self.min_dist_from_threshold[loop_ix]:
                #             self.min_dist_from_threshold[loop_ix] = avg_preds_dist
                #             for ix in range(self.total_CFs):
                #                 self.best_backup_cfs[loop_ix + ix] = copy.deepcopy(temp_cfs_stored[ix])
                #                 self.best_backup_cfs_preds[loop_ix + ix] = copy.deepcopy(test_preds_stored[ix])

                # if self.multi_class:
                #     desired_class_probs = [pred[self.target_cf_class] for pred in test_preds_stored]
                #     if all(prob >= self.stopping_threshold for prob in desired_class_probs):
                #         avg_preds_dist = np.mean([abs(prob - self.stopping_threshold) for prob in desired_class_probs])
                #         if avg_preds_dist < self.min_dist_from_threshold[loop_ix]:
                #             self.min_dist_from_threshold[loop_ix] = avg_preds_dist
                #             for ix in range(self.total_CFs):
                #                 # self.best_backup_cfs[loop_ix + ix] = copy.deepcopy(temp_cfs_stored[ix])
                #                 self.best_backup_cfs[loop_ix + ix] = copy.deepcopy(self.cfs[ix])
                #                 self.best_backup_cfs_preds[loop_ix + ix] = copy.deepcopy(test_preds_stored[ix])

                # else:
                #     if ((self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in test_preds_stored)) or
                #     (self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in test_preds_stored))):
                #         avg_preds_dist = np.mean([abs(pred - self.stopping_threshold) for pred in test_preds_stored])
                #         if avg_preds_dist < self.min_dist_from_threshold[loop_ix]:
                #             self.min_dist_from_threshold[loop_ix] = avg_preds_dist
                #             for ix in range(self.total_CFs):
                #                 # self.best_backup_cfs[loop_ix + ix] = copy.deepcopy(temp_cfs_stored[ix])
                #                 self.best_backup_cfs[loop_ix + ix] = copy.deepcopy(self.cfs[ix])
                #                 self.best_backup_cfs_preds[loop_ix + ix] = copy.deepcopy(test_preds_stored[ix])

                # Note: Removed iterations += 1 here (since it's now after stop_loop)

            # rounding off final cfs - not necessary when inter_project=True
            # self.round_off_cfs(assign=True)

            # storing final CFs
            for j in range(0, self.total_CFs):
                temp = self.cfs[j].numpy()
                self.final_cfs.append(temp)

            # max iterations at which GD stopped
            self.max_iterations_run = iterations

        self.elapsed = timeit.default_timer() - start_time
        self.cfs_preds = [self.predict_fn(tf.constant(cfs, dtype=tf.float32)) for cfs in self.final_cfs]

        # update final_cfs from backed up CFs if valid CFs are not found
        # if self.multi_class:
        #     self.cfs_preds = [self.predict_fn(tf.constant(cfs, dtype=tf.float32)) for cfs in self.final_cfs]
        #     print("self.cfs_preds: final check ", self.cfs_preds)
        #     # desired_class_probs = [pred[self.target_cf_class] for pred in self.cfs_preds]
        #     desired_class_probs = self.cfs_preds[0][self.target_cf_class]
        #     print("desired_class_probs final check: ", desired_class_probs)
        #     if desired_class_probs > self.stopping_threshold:
        #         # Use backup CFs if available
        #         for loop_ix in range(loop_find_CFs):
        #             if self.min_dist_from_threshold[loop_ix] != 100:
        #                 for ix in range(self.total_CFs):
        #                     self.final_cfs[loop_ix + ix] = copy.deepcopy(self.best_backup_cfs[loop_ix + ix])
        #                     self.cfs_preds[loop_ix + ix] = copy.deepcopy(self.best_backup_cfs_preds[loop_ix + ix])
        # else:
        #     self.cfs_preds = [self.predict_fn(tf.constant(cfs, dtype=tf.float32)) for cfs in self.final_cfs]
        #     if ((self.target_cf_class == 0 and any(i > self.stopping_threshold for i in self.cfs_preds)) or
        #     (self.target_cf_class == 1 and any(i < self.stopping_threshold for i in self.cfs_preds))):
        #         for loop_ix in range(loop_find_CFs):
        #             if self.min_dist_from_threshold[loop_ix] != 100:
        #                 for ix in range(self.total_CFs):
        #                     self.final_cfs[loop_ix+ix] = copy.deepcopy(self.best_backup_cfs[loop_ix+ix])
        #                     self.cfs_preds[loop_ix+ix] = copy.deepcopy(self.best_backup_cfs_preds[loop_ix+ix])

        # do inverse transform of CFs to original user-fed format
        cfs = np.array([self.final_cfs[i][0] for i in range(len(self.final_cfs))])
        # final_cfs_df = self.model.transformer.inverse_transform(
        #        self.data_interface.get_decoded_data(cfs))
        # inverse_transformed_cfs = self.transformer.inverse_transform(
                # self.data_interface.get_decoded_data(cfs))
        
        final_cfs_df = pd.DataFrame(cfs, columns=self.data_interface.feature_names)


        if self.multi_class:
            # For each prediction, extract the probability of the target class
            cfs_preds = [np.round(pred[self.target_cf_class], 3) for pred in self.cfs_preds]
            cfs_class = [np.argmax(pred) for pred in self.cfs_preds]
            # print("cfs_preds: ", cfs_preds)
            # print("cfs_class: ", cfs_class)
        else:
            cfs_preds = [np.round(pred, 3) for pred in self.cfs_preds]

        final_cfs_df[self.data_interface.outcome_name] = cfs_preds
        final_cfs_df['class'] = cfs_class
        # final_cfs_df[self.data_interface.outcome_name] = cfs_preds

        # test_instance_df = self.model.transformer.inverse_transform(
        #         self.data_interface.get_decoded_data(query_instance))
        # inverse_transformed_test_instance_df = self.transformer.inverse_transform(
        #          self.data_interface.get_decoded_data(query_instance))
        test_instance_df = pd.DataFrame(query_instance, columns=self.data_interface.feature_names)

        test_instance_df[self.data_interface.outcome_name] = np.array(np.round(test_pred[self.target_cf_class], 3))

        # post-hoc operation on continuous features to enhance sparsity - only for public data
        if posthoc_sparsity_param is not None and posthoc_sparsity_param > 0 and \
                'data_df' in self.data_interface.__dict__:
            final_cfs_df_sparse = final_cfs_df.copy()
            final_cfs_df_sparse = self.do_posthoc_sparsity_enhancement(final_cfs_df_sparse,
                                                                       test_instance_df,
                                                                       posthoc_sparsity_param,
                                                                       posthoc_sparsity_algorithm,
                                                                       limit_steps_ls)
        else:
            final_cfs_df_sparse = None
        # need to check the above code on posthoc sparsity

        # if posthoc_sparsity_param != None and posthoc_sparsity_param > 0 and 'data_df' in self.data_interface.__dict__:
        #     final_cfs_sparse = copy.deepcopy(self.final_cfs)
        #     cfs_preds_sparse = copy.deepcopy(self.cfs_preds)
        #     self.final_cfs_sparse, self.cfs_preds_sparse = self.do_posthoc_sparsity_enhancement(
        #           self.total_CFs, final_cfs_sparse, cfs_preds_sparse, query_instance, posthoc_sparsity_param,
        #           posthoc_sparsity_algorithm, total_random_inits=self.total_random_inits)
        # else:
        #     self.final_cfs_sparse = None
        #     self.cfs_preds_sparse = None

        m, s = divmod(self.elapsed, 60)
        # if self.multi_class:
        #     if all(i >= self.stopping_threshold for i in self.cfs_preds):
        #         self.total_CFs_found = max(loop_find_CFs, self.total_CFs)
        #         valid_ix = [ix for ix in range(max(loop_find_CFs, self.total_CFs))]
        #     print('Diverse Multi-class Counterfactuals found! total time taken: %02d' %
        #         m, 'min %02d' % s, 'sec')
        valid_ix = []

        if self.multi_class:
            predicted_classes = [np.argmax(pred) for pred in self.cfs_preds]
            # probability_values = [pred[self.target_cf_class] for pred in self.cfs_preds]
            if int(predicted_classes[0]) == int(self.target_cf_class): #and probability_values[0] > self.stopping_threshold:
                self.total_CFs_found = max(loop_find_CFs, self.total_CFs)
                valid_ix = [ix for ix in range(max(loop_find_CFs, self.total_CFs))]  # indexes of valid CFs
                print('Diverse Counterfactuals found! total time taken: %02d' %
                    m, 'min %02d' % s, 'sec')
            else:
                print('No Counterfactuals found for the given configuation')
        # elif ((self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in self.cfs_preds)) or
        # (self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in self.cfs_preds))): #binary
        elif ((self.target_cf_class == 0) or (self.target_cf_class == 1)):
            self.total_CFs_found = max(loop_find_CFs, self.total_CFs)
            valid_ix = [ix for ix in range(max(loop_find_CFs, self.total_CFs))]  # indexes of valid CFs
            print('Diverse Counterfactuals found! total time taken: %02d' %
                m, 'min %02d' % s, 'sec')
        else: #binary, but no cfs found
            self.total_CFs_found = 0
            valid_ix = []  # indexes of valid CFs
            for cf_ix, pred in enumerate(self.cfs_preds):
                if self.multi_class:
                    if all(i >= self.stopping_threshold for i in pred):
                        self.total_CFs_found += 1
                        valid_ix.append(cf_ix)
                elif ((self.target_cf_class == 0 and pred < self.stopping_threshold) or
                   (self.target_cf_class == 1 and pred > self.stopping_threshold)):
                    self.total_CFs_found += 1
                    valid_ix.append(cf_ix)

            if self.total_CFs_found == 0:
                print('No Counterfactuals found for the given configuation, perhaps try with different ',
                        'values of proximity (or diversity) weights or learning rate...',
                        '; total time taken: %02d' % m, 'min %02d' % s, 'sec')
            else:
                print('Only %d (required %d)' % (self.total_CFs_found, max(loop_find_CFs, self.total_CFs)),
                        ' Diverse Counterfactuals found for the given configuation, perhaps try with different',
                        'values of proximity (or diversity) weights or learning rate...',
                        '; total time taken: %02d' % m, 'min %02d' % s, 'sec')

        if final_cfs_df_sparse is not None:
            final_cfs_df_sparse = final_cfs_df_sparse.iloc[valid_ix].reset_index(drop=True)
        # returning only valid CFs
        # return final_cfs_df.iloc[valid_ix].reset_index(drop=True), test_instance_df, final_cfs_df_sparse, intermediate_cfs, gradients, losses, intermediate_predictions
        return final_cfs_df.iloc[valid_ix].reset_index(drop=True), test_instance_df, final_cfs_df_sparse, intermediate_cfs, gradients, losses, intermediate_predictions