#DiCE code adapted from: https://github.com/interpretml/DiCE

import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import spine.models.DiCE.dice_ml as dice_ml
from spine.data.load_data import *
import tensorflow as tf
tf.random.set_seed(1)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from VAE_DBS.models.vae import train_variational_autoencoder

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

class DiceCounterfactual:
    """
    A class for generating counterfactual examples using DiCE (Diverse Counterfactual Explanations)
    with optional integration of a Variational Autoencoder (VAE) to guide the search in latent space.

    This class:
      1) Loads and preprocesses a dataset.
      2) Loads a pre-trained or newly trained classifier (TensorFlow-based).
      3) Optionally trains a Variational Autoencoder to help guide counterfactual generation.
      4) Computes and stores class prototypes in the VAE latent space.
      5) Uses DiCE to generate counterfactuals for instances of interest.
      6) Exports the generated counterfactuals, intermediate gradient steps, and model predictions.

    Attributes
    ----------
    dataset_name : str
        Identifier for the dataset being used.
    data_columns : list
        List of column names for the dataset features + target.
    model_name : str
        Name of the classification model to load or train.
    outcome_name : str
        Name of the outcome/target column in the dataset.
    preprocessing : bool
        Whether preprocessing steps (scaling, encoding) are required for the model.
    backend : str
        Backend for DiCE, typically 'TF2'.
    model_format : str
        Format of the model file, e.g., 'h5' or 'SavedModel'.
    multi_class : bool
        Indicates if the classification problem is multi-class.
    comparison : bool
        If True, the class is used in a comparison setting with other methods.
    comparison_method : str
        Name of the comparison method (e.g., 'sdbm', 'deepview', etc.).
    input_path : str, optional
        Path to load existing models or data if needed.
    tt_number : int, optional
        Train/test index or seed for splitting data.
    classifier_name : str, optional
        Additional specification for the classifier name.
    cf_points_binary : list
        Predicted class labels for the generated counterfactual points (binary or integer-coded).
    counterfactuals : pd.DataFrame or None
        DataFrame containing the final counterfactual points.
    intermediate_points : pd.DataFrame or None
        DataFrame containing intermediate points from the gradient-based generation process.
    dataset : pd.DataFrame
        The loaded and preprocessed dataset.
    target : pd.Series
        The target column corresponding to dataset.
    dice_input_data : pd.DataFrame
        A copy of the feature portion of `dataset` used for DiCE.
    continuous_features : list
        List of continuous feature names used by DiCE.
    dice_data : dice_ml.data
        DiCE-compatible data object.
    model : tf.keras.Model
        The loaded/trained TensorFlow model for DiCE.
    model_for_predictions : callable or tf.keras.Model
        The model (or a function) used to generate predictions from data.
    autoencoder : tf.keras.Model
        Trained autoencoder model (if VAE is used).
    encoder : tf.keras.Model
        The encoder portion of the trained VAE.
    class_prototypes : dict
        Dictionary mapping class labels to their mean latent-space embeddings.
    dice_exp : dice_ml.Dice
        The main DiCE explainer object used for generating counterfactuals.
    predictions_dataset : np.ndarray
        Predictions of the model on the entire dataset.
    intermediate_y_pred : list
        List of predictions for all intermediate points.
    all_gradients : pd.DataFrame or None
        DataFrame containing all gradient information if needed.
    class_paths : dict
        Dictionary that stores intermediate paths for each class.

    Methods
    -------
    load_and_prepare_data():
        Loads and optionally preprocesses the dataset using `load_data_*` functions.
    load_classification_model():
        Loads the specified classification model (TensorFlow).
    save_data(path, version):
        Persists original data, counterfactuals, intermediate points, and all gradients to CSV.
    train_variational_autoencoder(encoding_dim=10, epochs=50, batch_size=256):
        Trains a Variational Autoencoder (VAE) on the dataset features.
    evaluate_autoencoder(save_path=None, version=None):
        Evaluates the reconstruction performance of the trained VAE and saves plots.
    compute_class_prototypes():
        Computes the centroid (mean) of latent embeddings for each class.
    prepare_dice_data(data=None):
        Prepares data for DiCE by instantiating `dice_ml.Data`.
    initialize_dice():
        Initializes the DiCE explainer object with the chosen method ("gradient").
    get_predictions(data):
        Retrieves predictions from `model_for_predictions` for given data.
    stratified_random_sampling(num_samples=None):
        Samples instances from the dataset in a stratified manner based on predicted labels.
    transform_intermediate_points_gradient(intermediate_points_gradient):
        Flattens the nested structure of intermediate gradient points into a single DataFrame.
    transform_gradients(gradients):
        Flattens the nested structure of the gradients into a single DataFrame.
    transform_predictions(intermediate_predictions):
        Flattens the nested structure of predictions into a single list.
    generate_counterfactuals(instance_index, total_cfs=8, ...):
        Generates counterfactuals for a single instance in the dataset using DiCE.
    extract_paths(intermediate_points_gradient):
        Extracts gradient-based path arrays for each counterfactual from the nested structure.
    fit_transform_all_gradient(num_samples, total_cfs, ...):
        Master method to sample multiple instances, generate counterfactuals, and consolidate results.
    """

    def __init__(
        self,
        dataset_name,
        data_columns,
        model_name,
        outcome_name,
        preprocessing,
        backend='TF2',
        model_format='h5',
        multi_class=False,
        comparison=False,
        comparison_method=None,
        comparison_data_name=None,
        input_path=None,
        tt_number=None,
        classifier_name=None
    ):
        """
        Initialize the DiceCounterfactual object with dataset, model details, and parameters
        for generating counterfactuals.

        Parameters
        ----------
        dataset_name : str
            Identifier for the dataset.
        data_columns : list
            Column names for the dataset features and target.
        model_name : str
            Name of the classification model file to load.
        outcome_name : str
            Name of the outcome/target column in the dataset.
        preprocessing : bool
            Whether or not to apply preprocessing steps.
        backend : str, optional
            Backend for DiCE (default 'TF2').
        model_format : str, optional
            Format for loading the model (default 'h5').
        multi_class : bool, optional
            If True, indicates a multi-class classification setup (default False).
        comparison : bool, optional
            If True, class is used for method comparison (default False).
        comparison_method : str, optional
            Method name if used for comparison (e.g., 'sdbm', 'deepview').
        comparison_data_name : str, optional
            Additional data name for comparison (not used by default).
        input_path : str, optional
            Path to input data/models if needed.
        tt_number : int, optional
            A train/test number or seed.
        classifier_name : str, optional
            Additional classifier name for loading.

        Returns
        -------
        None
        """
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

        # Load and prepare data
        self.dataset, self.target = self.load_and_prepare_data()
        self.target = pd.Series(self.target) if not isinstance(self.target, pd.Series) else self.target
        self.dice_input_data = self.dataset.copy()
        self.continuous_features = list(self.data_columns[:-1])  # all except target
        self.dice_data = self.prepare_dice_data(self.dice_input_data)

        # Load or build the classification model
        self.model, self.model_for_predictions = self.load_classification_model()

        # Train a Variational Autoencoder and compute class prototypes
        self.train_variational_autoencoder()
        self.compute_class_prototypes()
        
        # Initialize DiCE explainer
        self.dice_exp = self.initialize_dice()
        
        # Store model predictions on the entire dataset
        self.predictions_dataset = self.get_predictions(self.dataset)
    

    def load_and_prepare_data(self):
        """
        Load the dataset based on the specified comparison method and outcome name.
        Also applies variance-threshold feature selection to remove constant features.

        Returns
        -------
        dataset : pd.DataFrame
            The loaded and preprocessed dataset.
        y : np.ndarray
            The target array.
        """
        if self.comparison:
            # For comparing methods
            if self.comparison_method == 'sdbm':
                X, y = load_data_sdbm(self.dataset_name, self.tt_number)
            elif self.comparison_method == 'deepview':
                X, y = load_data_deepview(self.dataset_name, self.tt_number)
            else:
                raise ValueError(f"Unrecognized comparison method: {self.comparison_method}")
            if y.ndim == 2 and y.shape[1] > 1:
                # Convert one-hot/multi-label to single labels
                y = np.argmax(y, axis=1)
            dataset = pd.DataFrame(X, columns=self.data_columns[:-1])
            dataset[self.outcome_name] = y
            return dataset, y

        elif self.comparison_method == 'train_test':
            # For a train/test-based approach
            X, y = load_data_train_test(self.dataset_name, self.tt_number)
            if y.ndim == 2 and y.shape[1] > 1:
                y = np.argmax(y, axis=1)

            X = pd.DataFrame(X, columns=self.data_columns[:-1])
            # Remove zero-variance features
            selector = VarianceThreshold(threshold=0.0)
            selector.fit(X)
            selected_features = X[X.columns[selector.get_support(indices=True)]]
            dataset = pd.DataFrame(selected_features)
            dataset[self.outcome_name] = y
            return dataset, y

        else:
            # Load a generic dataset
            dataset, target = load_dataset(self.outcome_name, self.dataset_name)
            features = dataset.drop(columns=[self.outcome_name])

            # Remove zero-variance features
            selector = VarianceThreshold(threshold=0.0)
            selector.fit(features)
            selected_features = features[features.columns[selector.get_support(indices=True)]]

            dataset = selected_features.copy()
            dataset[self.outcome_name] = target
            return dataset, target


    def load_classification_model(self):
        """
        Load the specified classification model using a helper function `load_model_dice`.

        Returns
        -------
        model : tf.keras.Model
            The fully loaded Keras model for gradient-based operations.
        model_for_predictions : callable
            Model or function used strictly for generating predictions.
        """
        model, model_for_predictions = load_model_dice(
            self.backend,
            self.model_name,
            self.tt_number,
            self.classifier_name,
            self.model_format,
            self.preprocessing,
            self.multi_class,
            self.comparison_method
        )
        return model, model_for_predictions


    def save_data(self, path, version):
        """
        Save the original data, counterfactuals, and intermediate gradients to CSV files.

        Parameters
        ----------
        path : str
            Directory path where the files will be saved.
        version : str or int
            Suffix appended to each saved file name.

        Returns
        -------
        None
        """
        # Reconstruct the outcome column for the original dataset
        self.dataset[self.outcome_name] = self.target
        
        # Convert intermediate/final predictions to class labels
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

        # Save intermediate points (gradient steps)
        if self.intermediate_points is not None:
            gradients_path = os.path.join(path, f"_intermediate_gradients_{version}.csv")
            self.intermediate_points.to_csv(gradients_path, index=False)
            print(f"Intermediate gradient points saved to {gradients_path}")

        # Save all gradients if available
        if self.all_gradients is not None:
            all_gradients_path = os.path.join(path, f"_all_gradients_{version}.csv")
            self.all_gradients.to_csv(all_gradients_path, index=False)
            print(f"All gradients saved to {all_gradients_path}")


    def train_variational_autoencoder(self, encoding_dim=10, epochs=50, batch_size=256):
        """
        Train a Variational Autoencoder (VAE) on the dataset's features.

        Parameters
        ----------
        encoding_dim : int, optional
            Dimension of the latent representation (default=10).
        epochs : int, optional
            Number of epochs to train (default=50).
        batch_size : int, optional
            Batch size (default=256).

        Returns
        -------
        None
        """
        # Exclude the target column
        X = self.dataset.drop(columns=[self.outcome_name]) if self.outcome_name in self.dataset.columns else self.dataset
        X = X.values
        input_dim = X.shape[1]

        # Train the VAE using the external function
        self.autoencoder, self.encoder = train_variational_autoencoder(
            X,
            input_dim,
            encoding_dim=encoding_dim,
            epochs=epochs,
            batch_size=batch_size
        )


    def evaluate_autoencoder(self, save_path=None, version=None):
        """
        Evaluate the VAE's reconstruction performance on the dataset and save diagnostic plots:
          1) Reconstruction MSE and MAE
          2) Example reconstructions for random samples
          3) t-SNE visualization of the latent space
          4) Distribution of reconstruction errors

        Parameters
        ----------
        save_path : str, optional
            Directory where plots and metrics will be saved (None means no saving).
        version : str or int, optional
            Suffix used in naming saved plots/files.

        Returns
        -------
        None
        """
        # Prepare data
        X = self.dataset.drop(columns=[self.outcome_name]) if self.outcome_name in self.dataset.columns else self.dataset
        X = X.values

        # Predict with autoencoder
        X_reconstructed = self.autoencoder.predict(X)

        # Compute Reconstruction Loss (MSE)
        mse = np.mean(np.power(X - X_reconstructed, 2), axis=1)
        mean_mse = np.mean(mse)
        print(f"Mean Reconstruction MSE: {mean_mse}")

        # Compute MAE
        mae = np.mean(np.abs(X - X_reconstructed), axis=1)
        mean_mae = np.mean(mae)
        print(f"Mean Reconstruction MAE: {mean_mae}")

        # Optionally save metrics
        if save_path is not None and version is not None:
            metrics_path = os.path.join(save_path, f"autoencoder_metrics_{version}.txt")
            with open(metrics_path, 'w') as f:
                f.write(f"Mean Reconstruction MSE: {mean_mse}\n")
                f.write(f"Mean Reconstruction MAE: {mean_mae}\n")
            print(f"Reconstruction metrics saved to {metrics_path}")

        # Plot a few sample reconstructions
        num_samples = 3
        indices = np.random.choice(len(X), num_samples, replace=False)
        for i in indices:
            plt.figure(figsize=(10, 4))
            plt.plot(X[i], 'b-', label='Original')
            plt.plot(X_reconstructed[i], 'r--', label='Reconstructed')
            plt.title(f'Sample {i}', fontsize=25)
            plt.legend(fontsize=16)
            if save_path and version:
                plt.savefig(os.path.join(save_path, f'autoencoder_reconstruction_sample_{version}_{i}.png'))
            plt.close()
        print("Reconstruction samples saved")

        # t-SNE of the latent space (only if we have at least 2D in latent)
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
            if save_path and version:
                plt.savefig(os.path.join(save_path, f'autoencoder_latent_space_{version}.png'))
            plt.close()
            print("Latent space visualization saved")

        # Distribution of reconstruction errors
        plt.figure(figsize=(8, 6))
        sns.histplot(mse, bins=50, kde=True)
        plt.title('Distribution of Reconstruction Error (MSE)', fontsize=25)
        plt.xlabel('Reconstruction Error', fontsize=18)
        plt.ylabel('Frequency', fontsize=18)
        if save_path and version:
            plt.savefig(os.path.join(save_path, f'autoencoder_error_plot_{version}.png'))
        plt.close()
        print("Reconstruction error plot saved")


    def compute_class_prototypes(self):
        """
        Compute the class prototypes in the latent space by taking the mean of the encoded
        vectors for each class label.
        
        Returns
        -------
        None
        """
        # Prepare data
        X = self.dataset.drop(columns=[self.outcome_name]) if self.outcome_name in self.dataset.columns else self.dataset
        X = X.values
        
        # Get latent representations (e.g., z_mean)
        X_encoded = self.encoder.predict(X)
        y = self.target

        self.class_prototypes = {}
        for label in np.unique(y):
            class_indices = np.where(y == label)[0]
            class_encoded = X_encoded[class_indices]
            class_prototype = np.mean(class_encoded, axis=0)
            self.class_prototypes[label] = class_prototype


    def prepare_dice_data(self, data=None):
        """
        Prepare the dataset for DiCE usage by specifying continuous features and outcome.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to be used for DiCE, defaults to self.dataset.

        Returns
        -------
        dice_data : dice_ml.data
            A DiCE-compatible data object.
        """
        dice_data = dice_ml.Data(
            dataframe=data if data is not None else self.dataset,
            continuous_features=self.continuous_features,
            outcome_name=self.outcome_name,
            encoding='none'
        )
        return dice_data


    def initialize_dice(self):
        """
        Initialize the DiCE explainer object using the 'gradient' method and the trained autoencoder.

        Returns
        -------
        dice_exp : dice_ml.Dice
            DiCE explainer object.
        """
        dice_exp = dice_ml.Dice(
            self.dice_data,
            self.model,
            method="gradient",
            autoencoder=self.autoencoder,
            encoder=self.encoder,
            multi_class=self.multi_class
        )
        return dice_exp


    def get_predictions(self, data):
        """
        Generate model predictions on the provided data.

        Parameters
        ----------
        data : pd.DataFrame
            Input data for prediction (should not include the target column).

        Returns
        -------
        predictions : np.ndarray
            Model predictions, shaped (n_samples,) for binary or (n_samples, n_classes) for multi-class.
        """
        data = data.drop(columns=[self.outcome_name]) if self.outcome_name in data.columns else data
        raw_predictions = self.model_for_predictions.predict(data)

        # Adjust shape depending on model output
        if raw_predictions.ndim == 2:
            if raw_predictions.shape[1] == 1:
                # Binary classification with a single output neuron
                predictions = raw_predictions.flatten()
            elif raw_predictions.shape[1] >= 2:
                # Multi-class classification with softmax
                predictions = raw_predictions
            else:
                raise ValueError(f"Unexpected shape of predictions: {raw_predictions.shape}")
        else:
            raise ValueError(f"Unexpected shape of predictions: {raw_predictions.shape}")

        return predictions


    def stratified_random_sampling(self, num_samples=None):
        """
        Perform stratified sampling based on predicted class labels.

        Parameters
        ----------
        num_samples : int, optional
            Total number of samples to draw. If None, returns all indices.

        Returns
        -------
        sampled_indices : np.ndarray
            Array of sampled row indices.
        sampled_predictions : np.ndarray
            Predictions corresponding to those sampled indices.
        """
        predictions = self.get_predictions(self.dataset)
        total_instances = len(predictions)

        if self.multi_class:
            # Multi-class scenario
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
            class_sample_sizes[class_sample_sizes == 0] = 1  # ensure at least one sample/class

            # Adjust if we overshoot/undershoot
            total_samples_allocated = np.sum(class_sample_sizes)
            if total_samples_allocated > num_samples:
                surplus = total_samples_allocated - num_samples
                for _ in range(surplus):
                    max_class_idx = np.argmax(class_sample_sizes)
                    class_sample_sizes[max_class_idx] -= 1
            elif total_samples_allocated < num_samples:
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
            # Binary classification
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

            # Compute samples per class
            class_proportions = class_counts / total_instances
            class_sample_sizes = (class_proportions * num_samples).astype(int)
            class_sample_sizes[class_sample_sizes == 0] = 1

            total_samples_allocated = np.sum(class_sample_sizes)
            if total_samples_allocated > num_samples:
                surplus = total_samples_allocated - num_samples
                for _ in range(surplus):
                    max_class_idx = np.argmax(class_sample_sizes)
                    class_sample_sizes[max_class_idx] -= 1
            elif total_samples_allocated < num_samples:
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
        Flattens nested iteration structures of intermediate points from the DiCE gradient approach.

        This function specifically handles cases when `total_cfs > 1` or multi-class.

        Parameters
        ----------
        intermediate_points_gradient : list
            A nested list structure where:
            intermediate_points_gradient[iteration][cf_index] = np.array of shape (1, n_features).
        
        Returns
        -------
        df : pd.DataFrame
            Flattened DataFrame of all points (rows) from all iterations.
        """
        transformed_data = []

        if self.multi_class and len(intermediate_points_gradient[0]) > 1:
            # multi-class with multiple CFs
            num_counterfactuals = len(intermediate_points_gradient[0][0])

            for counterfactual_idx in range(num_counterfactuals):
                for pred_class in range(len(intermediate_points_gradient)):
                    for iteration_idx, iteration in enumerate(intermediate_points_gradient):
                        point = iteration[counterfactual_idx]
                        transformed_data.append(point[0].flatten())

        elif self.multi_class and len(intermediate_points_gradient[0]) == 1:
            # multi-class with single CF
            num_counterfactuals = len(intermediate_points_gradient[0])

            for counterfactual_idx in range(num_counterfactuals):
                for iteration_idx, iteration in enumerate(intermediate_points_gradient):
                    point = iteration[counterfactual_idx]
                    if isinstance(point, tf.Tensor):
                        point = point.numpy()
                    transformed_data.append(point.flatten())

        else:
            # binary or single CF scenario
            num_counterfactuals = len(intermediate_points_gradient[0])
            for counterfactual_idx in range(num_counterfactuals):
                for iteration_idx, iteration in enumerate(intermediate_points_gradient):
                    point = iteration[counterfactual_idx]
                    transformed_data.append(point[0].flatten())

        # Use dataset columns except outcome
        columns = self.dataset.columns[:-1] if self.outcome_name in self.dataset.columns else self.dataset.columns
        df = pd.DataFrame(transformed_data, columns=columns)
        return df


    def transform_gradients(self, gradients):
        """
        Similar flattening method for the gradient vectors themselves.

        Parameters
        ----------
        gradients : list
            Nested list structure storing gradients from the DiCE approach.

        Returns
        -------
        df : pd.DataFrame
            Flattened DataFrame of all gradient vectors from all iterations.
        """
        transformed_data = []

        if self.multi_class and len(gradients[0]) > 1:
            num_counterfactuals = len(gradients[0][0])
            for counterfactual_idx in range(num_counterfactuals):
                for pred_class in range(len(gradients)):
                    for iteration_idx, iteration in enumerate(gradients):
                        point = iteration[counterfactual_idx]
                        transformed_data.append(point[0].numpy().flatten())

        elif self.multi_class and len(gradients[0]) == 1:
            num_counterfactuals = len(gradients[0])
            for counterfactual_idx in range(num_counterfactuals):
                for iteration_idx, iteration in enumerate(gradients):
                    point = iteration[counterfactual_idx]
                    transformed_data.append(point[0].numpy().flatten())

        else:
            num_counterfactuals = len(gradients[0])
            for counterfactual_idx in range(num_counterfactuals):
                for iteration_idx, iteration in enumerate(gradients):
                    point = iteration[counterfactual_idx]
                    transformed_data.append(point[0].numpy().flatten())

        columns = self.dataset.columns[:-1] if self.outcome_name in self.dataset.columns else self.dataset.columns
        df = pd.DataFrame(transformed_data, columns=columns)
        return df


    def transform_predictions(self, intermediate_predictions):
        """
        Flattens nested predictions from intermediate gradient steps.

        Parameters
        ----------
        intermediate_predictions : list
            Nested list of predictions at each iteration for each counterfactual.

        Returns
        -------
        transformed_predictions : list
            Flattened list of prediction values.
        """
        transformed_predictions = []

        if self.multi_class and len(intermediate_predictions[0]) > 1:
            num_counterfactuals = len(intermediate_predictions[0][0])
            for counterfactual_idx in range(num_counterfactuals):
                for iteration_idx, iteration in enumerate(intermediate_predictions):
                    pred = iteration[counterfactual_idx]
                    transformed_predictions.append(pred[0][0])

        elif self.multi_class and len(intermediate_predictions[0]) == 1:
            num_counterfactuals = len(intermediate_predictions[0])
            for counterfactual_idx in range(num_counterfactuals):
                for iteration_idx, iteration in enumerate(intermediate_predictions):
                    pred = iteration[counterfactual_idx]
                    transformed_predictions.append(pred[0])

        else:
            num_counterfactuals = len(intermediate_predictions[0])
            for counterfactual_idx in range(num_counterfactuals):
                for iteration_idx, iteration in enumerate(intermediate_predictions):
                    pred = iteration[counterfactual_idx]
                    transformed_predictions.append(pred[0][0])

        return transformed_predictions


    def generate_counterfactuals(
        self,
        instance_index,
        total_cfs=8,
        desired_class="opposite",
        prediction_weight=1.5,
        proximity_weight=0,
        diversity_weight=0,
        categorical_penalty=0,
        ae_weight=0.5,
        proto_weight=0.7,
        learning_rate=0.005,
        min_iter=50
    ):
        """
        Generate counterfactual explanations for a single instance in the dataset using DiCE.

        Parameters
        ----------
        instance_index : int
            Index of the instance in the dataset.
        total_cfs : int, optional
            Number of counterfactuals to generate (default=8).
        desired_class : str or int, optional
            Either 'opposite' for binary or a specific class label for multi-class.
        prediction_weight : float, optional
            Weight for the prediction loss term in the gradient objective.
        proximity_weight : float, optional
            Weight for the proximity term (distance in feature space).
        diversity_weight : float, optional
            Weight for the diversity term among multiple CFs.
        categorical_penalty : float, optional
            Penalty for changing categorical features (not used if all are continuous).
        ae_weight : float, optional
            Weight for the autoencoder reconstruction term.
        proto_weight : float, optional
            Weight for the prototype distance term (guiding CFs to be near class prototypes).
        learning_rate : float, optional
            Learning rate for the gradient-based CF generation.
        min_iter : int, optional
            Minimum number of iterations of gradient descent.

        Returns
        -------
        cf_points : pd.DataFrame
            Final counterfactual points (features + predicted class).
        cf_points_binary : np.ndarray
            Class labels (integer) for the CF points.
        intermediate_points_gradient : list
            Nested structure with intermediate CF points per iteration.
        gradients : list
            Gradient vectors at each iteration.
        intermediate_predictions : list
            Model predictions at each iteration for each CF.
        num_iterations : int
            Total number of iterations performed.
        """
        # Extract the instance
        instance = self.dataset.iloc[instance_index:instance_index+1].drop(columns=[self.outcome_name]) \
                   if self.outcome_name in self.dataset.columns else self.dataset.iloc[instance_index:instance_index+1]
        instance = instance.values
        print("Generation of counterfactuals started...")

        # Call DiCE's gradient-based CF generation
        dice_exp = self.dice_exp.generate_counterfactuals(
            instance,
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
        
        # Extract relevant components
        cf_points = dice_exp.cf_examples_list[0].final_cfs_df
        intermediate_points_gradient = dice_exp.cf_examples_list[0].intermediate_cfs
        gradients = dice_exp.cf_examples_list[0].gradients
        intermediate_predictions = dice_exp.cf_examples_list[0].intermediate_predictions
        num_iterations = dice_exp.cf_examples_list[0].iterations

        cf_points_binary = cf_points['class'].values

        return cf_points, cf_points_binary, intermediate_points_gradient, gradients, intermediate_predictions, num_iterations


    def extract_paths(self, intermediate_points_gradient):
        """
        Extract gradient-based path arrays for each counterfactual from the nested structure.

        Each iteration is a list of CF points. This function stacks them
        into a single array for each CF to represent its path from initial to final iteration.

        Parameters
        ----------
        intermediate_points_gradient : list
            For iteration i: intermediate_points_gradient[i] is a list of CF arrays (1, n_features).

        Returns
        -------
        paths : list of np.ndarray
            Each element is (num_iterations, n_features), representing one CF's entire path.
        """
        num_iterations = len(intermediate_points_gradient)
        if num_iterations == 0:
            return []

        # Number of CFs = length of the list for the first iteration
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


    def fit_transform_all_gradient(
        self,
        num_samples,
        total_cfs,
        desired_class="opposite",
        prediction_weight=1,
        ae_weight=0,
        proto_weight=0.7,
        learning_rate=0.005,
        samples=None,
        min_iter=50
    ):
        """
        Perform counterfactual generation on multiple instances. Optionally use stratified sampling
        or a provided list of samples.

        Parameters
        ----------
        num_samples : int
            Number of instances to sample for CF generation.
        total_cfs : int
            Number of CFs to generate per instance.
        desired_class : str or int, optional
            Desired class for the CF, or "opposite" in binary classification.
        prediction_weight : float, optional
            Weight on the prediction loss term.
        ae_weight : float, optional
            Weight for the autoencoder reconstruction term.
        proto_weight : float, optional
            Weight for the prototype distance term.
        learning_rate : float, optional
            Learning rate for gradient-based updates.
        samples : list, optional
            If provided, uses these sample indices instead of random sampling.
        min_iter : int, optional
            Minimum number of iterations for gradient updates.

        Returns
        -------
        self.counterfactuals : pd.DataFrame
            DataFrame of all generated counterfactual points.
        self.dataset : pd.DataFrame
            The original dataset.
        self.predictions_dataset : np.ndarray
            Predictions on the entire dataset.
        self.intermediate_y_pred : list
            Predictions for intermediate points.
        self.dice_data : dice_ml.data
            DiCE data object.
        self.cf_points_binary : list
            Integer-coded class labels for CF points.
        sampled_indices : np.ndarray
            Indices of sampled instances.
        self.sampled_predictions : np.ndarray
            Model predictions for the sampled instances.
        self.intermediate_points : pd.DataFrame
            DataFrame of intermediate gradient steps for all generated CFs.
        counterfactual_idxs : list
            (Currently unused, but set aside for storing instance indexes if needed).
        all_gradients : pd.DataFrame
            Flattened DataFrame of all gradient vectors.
        self.class_paths : dict
            Dictionary storing the path of intermediate steps for each class (multi-class scenario).
        """
        if samples is not None:
            # Use provided samples
            sampled_indices = samples
            self.sampled_predictions = self.get_predictions(self.dataset)
        else:
            # Perform stratified random sampling
            sampled_indices, self.sampled_predictions = self.stratified_random_sampling(num_samples=num_samples)
            print(f"Sampled indices: {sampled_indices}")
        
        all_intermediate_points_gradient = pd.DataFrame(columns=self.dataset.columns)
        all_gradients = pd.DataFrame(columns=self.dataset.columns)
    
        counterfactual_idxs = []
        intermediate_y_pred = []

        iterations_list = []
        all_cf_points = []
        all_cf_points_binary = []

        # Initialize class_paths dict to hold intermediate steps for each label
        self.class_paths = {label: {'paths': [], 'predictions': []} for label in np.unique(self.target)}
        
        for i, idx in enumerate(sampled_indices):
            print(f"Processing #{i} instance {idx}... with prediction {self.sampled_predictions[i]}")

            if self.multi_class:
                # Multi-class scenario
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
                
                # Generate CFs targeting each class other than the current
                for target_class in all_classes:
                    if target_class != current_class:
                        print(f"Generating counterfactuals for current class {current_class} and target class {target_class}...")
                        (cf_points_class,
                         cf_points_binary_class,
                         intermediate_points_gradient_class,
                         gradients_class,
                         intermediate_predictions_class,
                         iterations_list_class) = self.generate_counterfactuals(
                             idx,
                             total_cfs,
                             desired_class=target_class,
                             prediction_weight=prediction_weight,
                             ae_weight=ae_weight,
                             proto_weight=proto_weight,
                             learning_rate=learning_rate,
                             min_iter=min_iter
                         )
                        cf_points = pd.concat([cf_points, cf_points_class], axis=0, ignore_index=True)
                        cf_points_binary.extend(cf_points_binary_class)
                        intermediate_points_gradient.extend(intermediate_points_gradient_class)
                        gradients.extend(gradients_class)
                        intermediate_predictions.extend(intermediate_predictions_class)
                        iterations_list.append(iterations_list_class)

                        # Store paths for visualization
                        intermediate_points_gradient_class_path = [current_class] + intermediate_points_gradient_class
                        self.class_paths[target_class]['paths'].append(intermediate_points_gradient_class_path)
                        self.class_paths[target_class]['predictions'].append(intermediate_predictions_class)

            else:
                # Binary scenario
                (cf_points,
                 cf_points_binary,
                 intermediate_points_gradient,
                 gradients,
                 intermediate_predictions,
                 iterations_list) = self.generate_counterfactuals(
                     idx,
                     total_cfs,
                     desired_class,
                     prediction_weight=prediction_weight,
                     ae_weight=ae_weight,
                     proto_weight=proto_weight
                 )

            if cf_points.empty:
                print(f"No counterfactuals generated for instance {idx}. Skipping.")
                continue

            # Drop the outcome column if it exists in CF points
            cf_points = cf_points.drop(columns=[self.outcome_name]) if self.outcome_name in cf_points.columns else cf_points.copy()
            all_cf_points.append(cf_points)

            # Flatten the intermediate points if total_cfs > 1
            if total_cfs > 1:
                intermediate_points_converted = self.transform_intermediate_points_gradient(intermediate_points_gradient)
                gradients_converted_df = self.transform_gradients(gradients)
                predictions_converted = self.transform_predictions(intermediate_predictions)
            else:
                # For single CF scenario or simpler structure
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

        # Consolidate final results
        self.counterfactuals = pd.concat(all_cf_points, axis=0, ignore_index=True)
        self.cf_points_binary = all_cf_points_binary
        self.intermediate_points = all_intermediate_points_gradient.drop(columns=[self.outcome_name]) \
            if self.outcome_name in all_intermediate_points_gradient.columns else all_intermediate_points_gradient
        self.intermediate_y_pred = intermediate_y_pred
        self.all_gradients = all_gradients

        return self.counterfactuals, self.dataset, self.predictions_dataset, self.intermediate_y_pred, self.dice_data, self.cf_points_binary, sampled_indices, self.sampled_predictions, self.intermediate_points, counterfactual_idxs, all_gradients, self.class_paths
