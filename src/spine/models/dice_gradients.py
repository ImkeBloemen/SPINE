import os
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# import dice_ml
import VAE_DBS
from VAE_DBS.models.DiCE.dice_ml.utils import helpers  # helper functions
from sklearn.feature_selection import VarianceThreshold
from tensorflow.keras.models import load_model
from VAE_DBS.models.DiCE.dice_ml.utils.helpers import DataTransfomer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import VAE_DBS.models.DiCE.dice_ml as dice_ml
from VAE_DBS.utils.utils import *
from VAE_DBS.data.load_data import *
import tensorflow as tf
tf.random.set_seed(1)
import random
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.losses import binary_crossentropy

from VAE_DBS.models.vae import train_variational_autoencoder

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

class DiceCounterfactual:
    def __init__(self, dataset_name, data_columns, model_name, outcome_name, preprocessing, backend='TF2', 
                 model_format='h5', multi_class=False, comparison=False, comparison_method=None, comparison_data_name=None, input_path=None,
                 tt_number=None, classifier_name=None):
        self.dataset_name = dataset_name
        self.data_columns = data_columns
        self.model_name = model_name
        self.outcome_name = outcome_name
        self.preprocessing = preprocessing
        self.backend = backend
        self.model_format = model_format
        self.multi_class = multi_class
        self.comparison = comparison
        self.comparison_method = comparison_method

        self.tt_number = tt_number
        self.classifier_name = classifier_name
        self.cf_points_binary = []

        self.counterfactuals = None
        self.intermediate_points = None
        
        if self.comparison == True:
            if self.comparison_method == 'sdbm':
                self.X_train, self.y_train = load_data_sdbm(self.outcome_name, f'{comparison_data_name}', tt_number)
            elif self.comparison_method == 'deepview':
                self.X_train, self.y_train = load_data_deepview(self.outcome_name, f'{comparison_data_name}', tt_number)
            else:
                ValueError("Comparison method not recognized.")
            if len(self.y_train.shape) == 2 and self.y_train.shape[1] > 1:
                self.y_train = np.argmax(self.y_train, axis=1)
            self.dice_input_data = pd.DataFrame(self.X_train, columns=self.data_columns[:-1])
            self.dataset = self.dice_input_data
            self.target = self.y_train
            self.dice_input_data[self.outcome_name] = self.y_train
            self.continuous_features = list(self.data_columns[:-1])
            self.dice_data = self.prepare_dice_data(self.dice_input_data)
        elif self.comparison_method == 'train_test':
            self.X_train, self.y_train = load_data_train_test(self.outcome_name, f'{comparison_data_name}', tt_number)
            if len(self.y_train.shape) == 2 and self.y_train.shape[1] > 1:
                self.y_train = np.argmax(self.y_train, axis=1)
            selector = VarianceThreshold(threshold=0.0)
            X_train = pd.DataFrame(self.X_train, columns=self.data_columns[:-1])
            selector.fit(X_train)
            features_selected = X_train[X_train.columns[selector.get_support(indices=True)]]
            X_train = np.array(features_selected)
            self.dice_input_data = pd.DataFrame(X_train, columns=self.data_columns[:-1])
            self.dataset = self.dice_input_data
            self.target = pd.Series(self.y_train) if not isinstance(self.y_train, pd.Series) else self.y_train
            self.dice_input_data[self.outcome_name] = self.y_train
            self.continuous_features = list(self.data_columns[:-1])
            self.dice_data = self.prepare_dice_data(self.dice_input_data)
        else:
            self.dataset, self.target = load_dataset(self.outcome_name, self.dataset_name)
            features = self.dataset.drop(columns=[self.outcome_name])
            selector = VarianceThreshold(threshold=0.0)
            selector.fit(features)
            features_selected = features[features.columns[selector.get_support(indices=True)]]
            self.dataset = features_selected
            self.dataset[self.outcome_name] = self.target
            self.dataset.reset_index(drop=True, inplace=True)
            self.continuous_features = [col for col in features_selected.columns if col != self.outcome_name]
            self.dice_data = self.prepare_dice_data()
    

        self.model, self.model_for_predictions = load_model_dice(self.backend, self.model_name, self.tt_number, self.classifier_name, self.model_format, self.preprocessing, self.multi_class, self.comparison_method)
        
        self.train_variational_autoencoder()
        print("VAE trained")
        self.compute_class_prototypes()
        print("Class prototypes computed")
        
        self.dice_exp = self.initialize_dice()
        print("DiCE explainer initialized")
        self.predictions_dataset = self.get_predictions(self.dataset)
        print("Predictions obtained")
        self.predictions_dataset = list(self.predictions_dataset)
        if np.isnan(self.dataset.values).any():
            print("Warning: Data contains NaN values.")
        if np.isinf(self.dataset.values).any():
            print("Warning: Data contains infinite values.")

    def save_data(self, path, version):
        """
        Save the original data, counterfactuals, and intermediate gradients to separate files.
        """
        self.dataset[self.outcome_name] = self.target
        intermediate_y_pred_binary = [np.argmax(x) for x in self.intermediate_y_pred]
        self.intermediate_points[self.outcome_name] = intermediate_y_pred_binary
        self.counterfactuals[self.outcome_name] = self.cf_points_binary

        # Save original data
        if self.dataset is not None:
            original_path = os.path.join(path, f"_original_{version}.csv")
            self.dataset.to_csv(original_path, index=False)
            print(f"Original data saved to {original_path}")

        # Save counterfactuals
        if self.counterfactuals is not None:
            cf_path = os.path.join(path, f"_counterfactuals_{version}.csv")
            self.counterfactuals.to_csv(cf_path, index=False)
            print(f"Counterfactuals saved to {cf_path}")

        # Save intermediate gradients
        if self.intermediate_points is not None:
            gradients_path = os.path.join(path, f"_intermediate_gradients_{version}.csv")
            self.intermediate_points.to_csv(gradients_path, index=False)
            print(f"Intermediate gradient points saved to {gradients_path}")

        if self.all_gradients is not None:
            all_gradients_path = os.path.join(path, f"_all_gradients_{version}.csv")
            self.all_gradients.to_csv(all_gradients_path, index=False)
            print(f"All gradients saved to {all_gradients_path}")

    def train_variational_autoencoder(self, encoding_dim=10, epochs=50, batch_size=256):
        """Train a Variational Autoencoder and extract the encoder."""
        # Get the features (exclude the target)
        X = self.dataset.drop(columns=[self.outcome_name]) if self.outcome_name in self.dataset.columns else self.dataset
        X = X.values
        input_dim = X.shape[1]

        # Train the VAE using the function from vae.py
        self.autoencoder, self.encoder = train_variational_autoencoder(
            X,
            input_dim,
            encoding_dim=encoding_dim,
            epochs=epochs,
            batch_size=batch_size
        )

    def evaluate_autoencoder(self, save_path=None, version=None):
            """Evaluate the VAE's reconstruction performance."""
            X = self.dataset.drop(columns=[self.outcome_name]) if self.outcome_name in self.dataset.columns else self.dataset
            X = X.values
            X_reconstructed = self.autoencoder.predict(X)

            
            # Compute Reconstruction Loss
            mse = np.mean(np.power(X - X_reconstructed, 2), axis=1)
            mean_mse = np.mean(mse)
            print(f"Mean Reconstruction MSE: {mean_mse}")
            
            # Compute MAE
            mae = np.mean(np.abs(X - X_reconstructed), axis=1)
            mean_mae = np.mean(mae)
            print(f"Mean Reconstruction MAE: {mean_mae}")

            # Save MSE and MAE to a text file
            if save_path is not None and version is not None:
                metrics_path = os.path.join(save_path, f"autoencoder_metrics_{version}.txt")
                with open(metrics_path, 'w') as f:
                    f.write(f"Mean Reconstruction MSE: {mean_mse}\n")
                    f.write(f"Mean Reconstruction MAE: {mean_mae}\n")
                print(f"Reconstruction metrics saved to {metrics_path}")
            
            num_samples = 3
            indices = np.random.choice(len(X), num_samples, replace=False)
            for i in indices:
                plt.figure(figsize=(10, 4))
                plt.plot(X[i], 'b-', label='Original')
                plt.plot(X_reconstructed[i], 'r--', label='Reconstructed')
                plt.title(f'Sample {i}', fontsize=25)
                plt.legend(fontsize=16)
                plt.savefig(os.path.join(save_path, f'autoencoder_reconstruction_sample_{version}_{i}.png'))
            print("Reconstruction samples saved")
            
            # Latent Space Visualization
            X_encoded = self.encoder.predict(X)
            if X_encoded.shape[1] >= 2:
                tsne = TSNE(n_components=2, random_state=42)
                latent_tsne = tsne.fit_transform(X_encoded)
                
                plt.figure(figsize=(8, 6))
                plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=self.target, cmap='viridis')
                plt.colorbar()
                plt.title('t-SNE Projection of the Latent Space')
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                plt.savefig(os.path.join(save_path, f'autoencoder_latent_space_{version}_{i}.png'))
                print("Latent space visualization saved")
            
            # Reconstruction Error Distribution
            sns.histplot(mse, bins=50, kde=True)
            plt.title('Distribution of Reconstruction Error (MSE)', fontsize=25)
            plt.xlabel('Reconstruction Error', fontsize=18)
            plt.ylabel('Frequency', fontsize=18)
            plt.savefig(os.path.join(save_path, f'autoencoder_error_plot_{version}_{i}.png'))
            print("Reconstruction error plot saved")

    def compute_class_prototypes(self):
        """Compute class prototypes using the encoder."""
        X = self.dataset.drop(columns=[self.outcome_name]) if self.outcome_name in self.dataset.columns else self.dataset
        X = X.values
        X_encoded = self.encoder.predict(X) # This is the z_mean, which represents the mean of the latent distribution for each input
        y = self.target

        self.class_prototypes = {}
        for label in np.unique(y):
            class_indices = np.where(y == label)[0]
            class_encoded = X_encoded[class_indices]
            class_prototype = np.mean(class_encoded, axis=0)
            self.class_prototypes[label] = class_prototype

    def prepare_dice_data(self, data=None):
        """Prepare the dataset for DiCE."""
        dice_data = dice_ml.Data(
            dataframe=data if data is not None else self.dataset,
            continuous_features=self.continuous_features,
            outcome_name=self.outcome_name,
            encoding='none'
        )
        return dice_data

    def initialize_dice(self):
        """Initialize the DiCE explainer."""
        dice_exp = dice_ml.Dice(self.dice_data, self.model, method="gradient", autoencoder=self.autoencoder, encoder=self.encoder, multi_class=self.multi_class)
        return dice_exp
    
    def get_predictions(self, data):
        """Get model predictions for all instances in the test data."""
        data = data.drop(columns=[self.outcome_name]) if self.outcome_name in data.columns else data
        raw_predictions = self.model_for_predictions.predict(data)

        # Adjust predictions based on output shape
        if raw_predictions.ndim == 2:
            if raw_predictions.shape[1] == 1:
                # Binary classification with sigmoid output
                predictions = raw_predictions.flatten()
            elif raw_predictions.shape[1] >= 2:
                # Multiclass classification with softmax output
                predictions = raw_predictions 
            else:
                raise ValueError(f"Unexpected shape of predictions: {raw_predictions.shape}")
        else:
            raise ValueError(f"Unexpected shape of predictions: {raw_predictions.shape}")

        return predictions
    
    def stratified_random_sampling(self, num_samples=None):
        """
        Perform stratified random sampling based on the predicted class labels.
        """
        predictions = self.get_predictions(self.dataset)
        total_instances = len(predictions)

        if self.multi_class:
            # Get predicted class labels
            predicted_labels = np.argmax(predictions, axis=1)
            unique_classes, class_counts = np.unique(predicted_labels, return_counts=True)

            if num_samples is None or num_samples < len(unique_classes):
                print("Number of samples is too small for stratification. Using random sampling instead.")
                if num_samples:
                    sampled_indices = np.random.choice(total_instances, size=num_samples, replace=False)
                else:
                    sampled_indices = np.arange(total_instances)
                sampled_predictions = predictions[sampled_indices]
                return sampled_indices, sampled_predictions

            # Compute number of samples per class
            class_proportions = class_counts / total_instances
            class_sample_sizes = (class_proportions * num_samples).astype(int)

            # Ensure at least one sample per class
            class_sample_sizes[class_sample_sizes == 0] = 1

            # Adjust total samples if necessary
            total_samples_allocated = np.sum(class_sample_sizes)
            if total_samples_allocated > num_samples:
                # Reduce samples from classes with the highest counts
                surplus = total_samples_allocated - num_samples
                for _ in range(surplus):
                    max_class_idx = np.argmax(class_sample_sizes)
                    class_sample_sizes[max_class_idx] -= 1
            elif total_samples_allocated < num_samples:
                # Add samples to classes with the highest counts
                deficit = int(num_samples - total_samples_allocated)
                for _ in range(deficit):
                    max_class_idx = np.argmax(class_sample_sizes)
                    class_sample_sizes[max_class_idx] += 1

            sampled_indices = []
            for class_label, sample_size in zip(unique_classes, class_sample_sizes):
                class_indices = np.where(predicted_labels == class_label)[0]
                if len(class_indices) > 0:
                    sampled_class_indices = np.random.choice(class_indices, size=sample_size, replace=False)
                    sampled_indices.extend(sampled_class_indices)
                else:
                    print(f"No instances for class {class_label}.")

            sampled_indices = np.array(sampled_indices)
            sampled_predictions = predictions[sampled_indices]

        else:
            # Binary classification case
            predictions = predictions.flatten()
            predicted_labels = (predictions >= 0.5).astype(int)
            unique_classes, class_counts = np.unique(predicted_labels, return_counts=True)

            if num_samples is None or num_samples < len(unique_classes):
                print("Number of samples is too small for stratification. Using random sampling instead.")
                if num_samples:
                    sampled_indices = np.random.choice(total_instances, size=num_samples, replace=False)
                else:
                    sampled_indices = np.arange(total_instances)
                sampled_predictions = predictions[sampled_indices]
                return sampled_indices, sampled_predictions

            # Compute number of samples per class
            class_proportions = class_counts / total_instances
            class_sample_sizes = (class_proportions * num_samples).astype(int)

            # Ensure at least one sample per class
            class_sample_sizes[class_sample_sizes == 0] = 1

            # Adjust total samples if necessary
            total_samples_allocated = np.sum(class_sample_sizes)
            if total_samples_allocated > num_samples:
                # Reduce samples from classes with the highest counts
                surplus = total_samples_allocated - num_samples
                for _ in range(surplus):
                    max_class_idx = np.argmax(class_sample_sizes)
                    class_sample_sizes[max_class_idx] -= 1
            elif total_samples_allocated < num_samples:
                # Add samples to classes with the highest counts
                deficit = num_samples - total_samples_allocated
                for _ in range(deficit):
                    max_class_idx = np.argmax(class_sample_sizes)
                    class_sample_sizes[max_class_idx] += 1

            sampled_indices = []
            for class_label, sample_size in zip(unique_classes, class_sample_sizes):
                class_indices = np.where(predicted_labels == class_label)[0]
                if len(class_indices) > 0:
                    sampled_class_indices = np.random.choice(class_indices, size=sample_size, replace=False)
                    sampled_indices.extend(sampled_class_indices)
                else:
                    print(f"No instances for class {class_label}.")

            sampled_indices = np.array(sampled_indices)
            sampled_predictions = predictions[sampled_indices]

        return sampled_indices, sampled_predictions
    
    def transform_intermediate_points_gradient(self, intermediate_points_gradient):
        """
        Transforms the data to collect all iteration points for each counterfactual point
        and flatten them into a single DataFrame, preserving the correct order. This is only necessary when
        total_cfs > 1.

        Args:
        - intermediate_points_gradient: A list where each element corresponds to an instance of interest,
        and within each instance, a list of iterations, and within each iteration, a list of points (e.g., 10 points).

        Returns:
        - df: A DataFrame where all points are stacked underneath each other, with columns indicating
        the instance of interest, the iteration number, and the features (29 columns).
        """
        
        transformed_data = []

        if self.multi_class and len(intermediate_points_gradient[0]) > 1:
            num_counterfactuals = len(intermediate_points_gradient[0][0])

            # Loop over each counterfactual point (e.g., the first point, second point, ..., nth point)
            for counterfactual_idx in range(num_counterfactuals):
                # Now loop over the iterations for this counterfactual point
                for pred_class in range(len(intermediate_points_gradient)):
                    for iteration_idx, iteration in enumerate(intermediate_points_gradient):
                        # Extract the point corresponding to this counterfactual in this iteration
                        point = iteration[counterfactual_idx]
                        
                        # Append the point's data
                        transformed_data.append(point[0].flatten())
                    
        elif self.multi_class and len(intermediate_points_gradient[0]) == 1:
            num_counterfactuals = len(intermediate_points_gradient[0])

            for counterfactual_idx in range(num_counterfactuals):

                # Loop over each counterfactual point (e.g., the first point, second point, ..., nth point)
                for iteration_idx, iteration in enumerate(intermediate_points_gradient):
                    point = iteration[counterfactual_idx]
                        
                    if isinstance(point, tf.Tensor):
                        point = point.numpy()

                    transformed_data.append(point.flatten())

        else:
            # Loop over each instance of interest
            num_counterfactuals = len(intermediate_points_gradient[0])  # The number of points per iteration (e.g., 10)

            # Loop over each counterfactual point (e.g., the first point, second point, ..., nth point)
            for counterfactual_idx in range(num_counterfactuals):
                # Now loop over the iterations for this counterfactual point
                for iteration_idx, iteration in enumerate(intermediate_points_gradient):
                    # Extract the point corresponding to this counterfactual in this iteration
                    point = iteration[counterfactual_idx]
                    
                    # Append the point's data
                    transformed_data.append(point[0].flatten())
        
        # Convert the transformed data into a DataFrame
        columns = self.dataset.columns[:-1] if self.outcome_name in self.dataset.columns else self.dataset.columns
        df = pd.DataFrame(transformed_data, columns=columns)

        return df
    
    def transform_gradients(self, gradients):
        """
        Transforms the data to collect all iteration points for each counterfactual point
        and flatten them into a single DataFrame, preserving the correct order. This is only necessary when
        total_cfs > 1.

        Args:
        - intermediate_points_gradient: A list where each element corresponds to an instance of interest,
        and within each instance, a list of iterations, and within each iteration, a list of points (e.g., 10 points).

        Returns:
        - df: A DataFrame where all points are stacked underneath each other, with columns indicating
        the instance of interest, the iteration number, and the features (29 columns).
        """
        
        transformed_data = []

        if self.multi_class and len(gradients[0]) > 1:
            num_counterfactuals = len(gradients[0][0])  # The number of points per iteration (e.g., 10)

            # Loop over each counterfactual point (e.g., the first point, second point, ..., nth point)
            for counterfactual_idx in range(num_counterfactuals):
                for pred_class in range(len(gradients)):
                    # Now loop over the iterations for this counterfactual point
                    for iteration_idx, iteration in enumerate(gradients):
                        # Extract the point corresponding to this counterfactual in this iteration
                        point = iteration[counterfactual_idx]
                        
                        # Append the point's data
                        transformed_data.append(point[0].numpy().flatten())

        elif self.multi_class and len(gradients[0]) == 1:
            num_counterfactuals = len(gradients[0])

            for counterfactual_idx in range(num_counterfactuals):
                # Loop over each counterfactual point (e.g., the first point, second point, ..., nth point)
                for iteration_idx, iteration in enumerate(gradients):
                    point = iteration[counterfactual_idx]
                    
                    transformed_data.append(point[0].numpy().flatten())

        else:
            # Loop over each instance of interest
            num_counterfactuals = len(gradients[0])  # The number of points per iteration (e.g., 10)

            # Loop over each counterfactual point (e.g., the first point, second point, ..., nth point)
            for counterfactual_idx in range(num_counterfactuals):
                # Now loop over the iterations for this counterfactual point
                for iteration_idx, iteration in enumerate(gradients):
                    # Extract the point corresponding to this counterfactual in this iteration
                    point = iteration[counterfactual_idx]
                    
                    # Append the point's data
                    transformed_data.append(point[0].numpy().flatten())
        
        # Convert the transformed data into a DataFrame
        columns = self.dataset.columns[:-1] if self.outcome_name in self.dataset.columns else self.dataset.columns
        df = pd.DataFrame(transformed_data, columns=columns)

        return df
    
    def transform_predictions(self, intermediate_predictions):
        """
        Transforms the data to collect all iteration points for each counterfactual point
        and flatten them into a single DataFrame, preserving the correct order. This is only necessary when
        total_cfs > 1.

        Args:
        - intermediate_points_gradient: A list where each element corresponds to an instance of interest,
        and within each instance, a list of iterations, and within each iteration, a list of points (e.g., 10 points).

        Returns:
        - list of predictions
        """
        
        transformed_predictions = []

        if self.multi_class and len(intermediate_predictions[0]) > 1:
            num_counterfactuals = len(intermediate_predictions[0][0])

            # Loop over each counterfactual point (e.g., the first point, second point, ..., nth point)
            for counterfactual_idx in range(num_counterfactuals):
                # Now loop over the iterations for this counterfactual point
                for iteration_idx, iteration in enumerate(intermediate_predictions):
                    # Extract the point corresponding to this counterfactual in this iteration
                    pred = iteration[counterfactual_idx]
                    
                    # Append the point's data
                    transformed_predictions.append(pred[0][0])

        elif self.multi_class and len(intermediate_predictions[0]) == 1:
            num_counterfactuals = len(intermediate_predictions[0])
            
            for counterfactual_idx in range(num_counterfactuals):
                # Loop over each counterfactual point (e.g., the first point, second point, ..., nth point)
                for iteration_idx, iteration in enumerate(intermediate_predictions):
                    pred = iteration[counterfactual_idx]
                    
                    transformed_predictions.append(pred[0])
        
        else:
            # Loop over each instance of interest
            num_counterfactuals = len(intermediate_predictions[0])  # The number of points per iteration (e.g., 10)

            # Loop over each counterfactual point (e.g., the first point, second point, ..., nth point)
            for counterfactual_idx in range(num_counterfactuals):
                # Now loop over the iterations for this counterfactual point
                for iteration_idx, iteration in enumerate(intermediate_predictions):
                    # Extract the point corresponding to this counterfactual in this iteration
                    pred = iteration[counterfactual_idx]
                    
                    # Append the point's data
                    transformed_predictions.append(pred[0][0])
        
            
        # Convert the transformed data into a list
        return transformed_predictions



    def generate_counterfactuals(self, instance_index, total_cfs=8, desired_class="opposite", prediction_weight=1.5, proximity_weight=0, diversity_weight=0, categorical_penalty=0, ae_weight=0.5, proto_weight=0.7, learning_rate=0.005, min_iter=50):
        """Generate and visualize counterfactual explanations."""
        instance = self.dataset.iloc[instance_index:instance_index+1].drop(columns=[self.outcome_name]) if self.outcome_name in self.dataset.columns else self.dataset.iloc[instance_index:instance_index+1]
        instance = instance.values
        print("Generation of counterfactuals started...")
        dice_exp = self.dice_exp.generate_counterfactuals(instance, 
                    total_CFs=total_cfs, 
                    desired_class=desired_class, 
                    prediction_weight=prediction_weight, 
                    proximity_weight=proximity_weight,
                    diversity_weight=diversity_weight, 
                    categorical_penalty=categorical_penalty,
                    ae_weight=ae_weight,
                    proto_weight=proto_weight,
                    class_prototypes=self.class_prototypes,
                    algorithm="DiverseCF", 
                    learning_rate=learning_rate,
                    min_iter=min_iter,
                    max_iter=3000, 
                    project_iter=5,
                    loss_diff_thres=1e-4, 
                    posthoc_sparsity_param=None, 
                    pred_threshold=1e-4,    
                    pred_converge_steps=10, 
                    pred_converge_maxiter=5, 
                    verbose=True
                    )
        print("Generation of counterfactuals completed.")
        
        cf_points = dice_exp.cf_examples_list[0].final_cfs_df
        intermediate_points_gradient = dice_exp.cf_examples_list[0].intermediate_cfs
        gradients = dice_exp.cf_examples_list[0].gradients
        intermediate_predictions = dice_exp.cf_examples_list[0].intermediate_predictions
        num_iterations = dice_exp.cf_examples_list[0].iterations

        cf_points_binary = cf_points['class'].values

        return cf_points, cf_points_binary, intermediate_points_gradient, gradients, intermediate_predictions, num_iterations
        
    def extract_paths(self, intermediate_points_gradient):
        """
        Extracts paths from intermediate_points_gradient.
        
        Parameters
        ----------
        intermediate_points_gradient : list
            A list where each element corresponds to an iteration. Each iteration is a list of
            arrays representing each counterfactual’s state at that iteration.
            For example:
            intermediate_points_gradient[iteration][cf_index] = np.array of shape (1, n_features)
            
        Returns
        -------
        paths : list of np.ndarray
            A list where each element is an array of shape (num_iterations, n_features) representing
            one counterfactual path from the initial to the final iteration.
        """
        num_iterations = len(intermediate_points_gradient)
        if num_iterations == 0:
            return []

        # Number of counterfactuals is determined by the length of the first iteration’s list
        num_cfs = len(intermediate_points_gradient[0])
        paths = []

        for cf_idx in range(num_cfs):
            path_points = []
            for iter_idx in range(num_iterations):
                point = intermediate_points_gradient[iter_idx][cf_idx]
                if isinstance(point, tf.Tensor):
                    point = point.numpy()
                path_points.append(point[0])  
            
            path_array = np.vstack(path_points) 
            paths.append(path_array)

        return paths
    
    def fit_transform_all_gradient(self, num_samples, total_cfs, desired_class="opposite", prediction_weight=1, ae_weight=0, proto_weight=0.7, learning_rate=0.005, samples=None, min_iter=50):

        if samples is not None:
            sampled_indices = samples
            self.sampled_predictions = self.get_predictions(self.dataset)
        else:
            sampled_indices, self.sampled_predictions = self.stratified_random_sampling(num_samples=num_samples)
            print(f"Sampled indices: {sampled_indices}")
        
        all_intermediate_points_gradient = pd.DataFrame(columns=self.dataset.columns)
        all_gradients = pd.DataFrame(columns=self.dataset.columns)
    
        counterfactual_idxs = []
        intermediate_y_pred = []

        iterations_list = []
        all_cf_points = []
        all_cf_points_binary = []

        self.class_paths = {label: {'paths': [], 'predictions': []} for label in np.unique(self.target)}
        
        for i, idx in enumerate(sampled_indices):
            print(f"Processing #{i} instance {idx}... with prediction {self.sampled_predictions[i]}")

            if self.multi_class:
                current_prediction = self.sampled_predictions[i]
                current_class = np.argmax(current_prediction)
                if self.comparison:
                    all_classes = np.unique(self.target)
                else:
                    all_classes = np.unique(self.target.values)

                cf_points = pd.DataFrame()
                cf_points_binary = []
                intermediate_points_gradient = []
                gradients = []
                intermediate_predictions = []
                iterations_list = []
                
                
                for target_class in all_classes:
                    if target_class != current_class:
                        print(f"Generating counterfactuals for current class {current_class} and target class {target_class}...")
                        cf_points_class, cf_points_binary_class, intermediate_points_gradient_class, gradients_class, intermediate_predictions_class, iterations_list_class = self.generate_counterfactuals(idx,
                                                                                                                                                                                                          total_cfs, 
                                                                                                                                                                                                          desired_class=target_class,
                                                                                                                                                                                                          prediction_weight=prediction_weight, 
                                                                                                                                                                                                          ae_weight=ae_weight, 
                                                                                                                                                                                                          proto_weight=proto_weight,
                                                                                                                                                                                                          learning_rate=learning_rate,
                                                                                                                                                                                                          min_iter=min_iter)
                        cf_points = pd.concat([cf_points, cf_points_class], axis=0, ignore_index=True)
                        cf_points_binary.extend(cf_points_binary_class)
                        intermediate_points_gradient.extend(intermediate_points_gradient_class)
                        gradients.extend(gradients_class)
                        intermediate_predictions.extend(intermediate_predictions_class)
                        iterations_list.append(iterations_list_class)

                        intermediate_points_gradient_class_path = [current_class] + intermediate_points_gradient_class
                        self.class_paths[target_class]['paths'].append(intermediate_points_gradient_class_path)
                        self.class_paths[target_class]['predictions'].append(intermediate_predictions_class)

            else:
                cf_points, cf_points_binary, intermediate_points_gradient, gradients, intermediate_predictions, iterations_list = self.generate_counterfactuals(idx, total_cfs, desired_class, prediction_weight=prediction_weight, ae_weight=ae_weight, proto_weight=proto_weight)

            if cf_points.empty:
                print(f"No counterfactuals generated for instance {idx}. Skipping.")
                continue

            #Transform counterfactual points:
            cf_points = cf_points.drop(columns=[self.outcome_name]) if self.outcome_name in cf_points.columns else cf_points.copy()
            all_cf_points.append(cf_points)

            if total_cfs > 1:
                intermediate_points_converted = self.transform_intermediate_points_gradient(intermediate_points_gradient)
                gradients_converted_df = self.transform_gradients(gradients)
                predictions_converted = self.transform_predictions(intermediate_predictions)
            else:
                columns = self.dataset.columns[:-1] if self.outcome_name in self.dataset.columns else self.dataset.columns
                intermediate_points_combined = np.vstack([np.squeeze(arr, axis=0) for arr in intermediate_points_gradient])
                intermediate_points_converted = pd.DataFrame(intermediate_points_combined, columns=columns)
                gradient_points_combined = np.vstack([np.squeeze(arr, axis=0) for arr in gradients])
                gradients_converted_df = pd.DataFrame(gradient_points_combined, columns=columns)
                predictions_converted = intermediate_predictions
                    
    
            all_intermediate_points_gradient = pd.concat([all_intermediate_points_gradient, intermediate_points_converted], axis=0, ignore_index=True)

            all_gradients = pd.concat([all_gradients, gradients_converted_df], axis=0, ignore_index=True)

            all_cf_points_binary = all_cf_points_binary + list(cf_points_binary)

            intermediate_y_pred = intermediate_y_pred + predictions_converted

            iterations_list.append(iterations_list)

        self.counterfactuals = pd.concat(all_cf_points, axis=0, ignore_index=True)
        self.cf_points_binary = all_cf_points_binary
        self.intermediate_points = all_intermediate_points_gradient.drop(columns=[self.outcome_name]) if self.outcome_name in all_intermediate_points_gradient.columns else all_intermediate_points_gradient
        self.intermediate_y_pred = intermediate_y_pred
        self.all_gradients = all_gradients

        return self.counterfactuals, self.dataset, self.predictions_dataset, self.intermediate_y_pred, self.dice_data, self.cf_points_binary, sampled_indices, self.sampled_predictions, self.intermediate_points, counterfactual_idxs, all_gradients, self.class_paths