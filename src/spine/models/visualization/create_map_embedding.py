import os
import random
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler
import umap.umap_ as umap
import matplotlib.cm as cm
from matplotlib.colors import to_rgba
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
from sklearn.manifold import TSNE
from matplotlib.colors import Normalize
from sklearn.utils.extmath import cartesian
from spine.models.transformers.ssnp import SSNP
from joblib import load
from matplotlib.colors import ListedColormap
from spine.models.visualization.plots import *
from sklearn.decomposition import PCA
from collections import defaultdict
import json
from spine.models.transformers.lamp_scaled import Lamp
from timeit import default_timer as timer
from datetime import timedelta
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

class PredictionMap:
    """
    The `PredictionMap` class projects high-dimensional data onto a 2D space,
    then creates a color-coded grid (a decision boundary) reflecting model predictions
    for the entire 2D projection. It can handle various dimensionality-reduction methods
    (UMAP, SSNP, LAMP, PCA), and provides functionalities such as:

    - Fitting and transforming data to 2D.
    - Generating a grid of points in 2D and inversely estimating their positions in the
      original high-dimensional space (knn weighted interpolation).
    - Visualizing original data, gradient paths, test points, and model decision boundaries.
    - Computing metrics (like a Jaccard-based filtering) to refine the visualization.
    
    Attributes
    ----------
    grid_size : int
        The resolution of the grid used in the 2D plot.
    original_data : np.ndarray
        Original high-dimensional data (features only).
    intermediate_gradient_points : np.ndarray
        Points along gradient paths in high-dimensional space before the final counterfactuals.
    counterfactuals : np.ndarray
        Final gradient points (counterfactuals).
    number_of_neighbors : int
        Number of neighbors used in the knn_weighted_interpolation process.
    model_for_predictions : tf.keras.Model or similar
        The predictive model to generate probability predictions.
    scaler_2D : MinMaxScaler
        Scaler used to transform the 2D projection into a [0, grid_size-1] range.
    n_classes : int
        Number of classes in the classification model.
    outcome_name : str
        Name of the target column in the original dataset.
    intermediate_predictions : np.ndarray
        Model predictions for intermediate points.
    original_predictions : np.ndarray
        Model predictions for original data points.
    version : int
        Version index, used as a suffix in saved filenames.
    projection_method : str
        The projection method to use: 'umap', 'ssnp', 'lamp', 'PCA', or direct 2D.
    all_gradients : np.ndarray
        All gradients associated with the entire path (optional usage).
    comparison : bool
        Whether this instance is used for a side-by-side comparison.
    dataset_name : str
        Name of the dataset for saving references.
    gradients : bool
        Whether to project and plot the gradient sub-steps separately.
    hidden_layer : bool
        If True, extracts outputs from the penultimate layer of the model.
    filter_jd : bool
        If True, applies Jaccard filtering to remove outlier points in 2D.
    class_colors : list
        A list of matplotlib-compatible RGBA color definitions for each class.
    custom_cmap : ListedColormap
        A custom colormap based on class_colors.
    pixel_colors : np.ndarray
        RGB or RGBA colors for each pixel in the final decision map.
    grid_points_2D : np.ndarray
        2D coordinates of the grid used for plotting.
    original_data_2D : np.ndarray
        2D projection of original_data.
    intermediate_points_2D : np.ndarray
        2D projection of intermediate_gradient_points.
    grid_predictions_prob : np.ndarray
        Probability predictions across the 2D grid.
    """

    def __init__(
        self,
        grid_size,
        original_data,
        intermediate_gradient_points,
        counterfactuals,
        number_of_neighbors,
        model_for_predictions,
        scaler_path_2D: str = None,
        projection_method='umap',
        projection_name=None,
        all_gradients=None,
        intermediate_predictions=None,
        original_predictions=None,
        counterfactual_predictions=None,
        outcome_name='target',
        n_classes=2,
        version=1,
        comparison=False,
        dataset_name=None,
        gradients=False,
        hidden_layer=False,
        filter_jd=False
    ):
        """
        Initialize the PredictionMap object with all data and parameters needed for 2D mapping
        and subsequent predictions.

        Parameters
        ----------
        grid_size : int
            Resolution for the 2D grid.
        original_data : pd.DataFrame or np.ndarray
            Original dataset containing features (and optionally the outcome).
        intermediate_gradient_points : pd.DataFrame or np.ndarray
            Interpolated points between the original data and counterfactuals in high-dimensional space.
        counterfactuals : pd.DataFrame or np.ndarray
            Counterfactual points in high-dimensional space.
        number_of_neighbors : int
            Number of neighbors for the knn_weighted_interpolation process.
        model_for_predictions : tf.keras.Model
            Trained model used for making predictions.
        scaler_path_2D : str, optional
            If provided, loads a trained scaler for transforming the 2D projection.
        projection_method : str, optional
            Method for projecting data to 2D (default is 'umap').
        projection_name : str, optional
            Identifier name for the projection (used in saving/loading).
        all_gradients : pd.DataFrame or np.ndarray, optional
            Full set of gradient points if needed.
        intermediate_predictions : np.ndarray, optional
            Model predictions for the intermediate gradient points.
        original_predictions : np.ndarray, optional
            Model predictions for the original data points.
        counterfactual_predictions : np.ndarray, optional
            Model predictions for the final counterfactuals.
        outcome_name : str, optional
            Name of the target column in the dataset.
        n_classes : int, optional
            Number of classes in the classification problem (default is 2).
        version : int, optional
            Version identifier for saving images (default is 1).
        comparison : bool, optional
            Whether or not to load an existing SSNP model for comparison.
        dataset_name : str, optional
            Name of the dataset, used for saving references.
        gradients : bool, optional
            If True, also project and plot intermediate gradient steps separately.
        hidden_layer : bool, optional
            If True, extracts outputs from the penultimate layer for the 2D projection.
        filter_jd : bool, optional
            If True, applies Jaccard filtering to remove outlier points in 2D.

        Returns
        -------
        None
        """
        self.grid_size = grid_size
        self.number_of_neighbors = number_of_neighbors
        self.model_for_predictions = model_for_predictions

        # Load or instantiate the scaler for 2D transformations
        if scaler_path_2D is not None:
            self.scaler_2D = load(scaler_path_2D)
        else:
            self.scaler_2D = MinMaxScaler()

        self.n_classes = n_classes
        self.outcome_name = outcome_name
        self.intermediate_predictions = intermediate_predictions
        self.original_predictions = original_predictions
        self.version = version
        self.projection_method = projection_method
        self.all_gradients = all_gradients
        self.comparison = comparison
        self.dataset_name = dataset_name
        self.gradients = gradients
        self.hidden_layer = hidden_layer
        self.filter_jd = filter_jd

        # Ensure original_data is a NumPy array for consistent indexing
        if isinstance(original_data, pd.DataFrame):
            self.target = original_data[outcome_name].values
            self.original_data_df = (
                original_data.drop(columns=[outcome_name])
                if outcome_name in original_data.columns
                else original_data
            )
            self.original_data = self.original_data_df.values
            if np.isnan(self.original_data).any():
                print("Input data X contains NaN values.")
        else:
            self.original_data = original_data

        # Ensure intermediate_gradient_points is a NumPy array
        if isinstance(intermediate_gradient_points, pd.DataFrame):
            self.intermediate_gradient_points = (
                intermediate_gradient_points.drop(columns=[outcome_name])
                if outcome_name in intermediate_gradient_points.columns
                else intermediate_gradient_points
            )
            self.intermediate_gradient_points = self.intermediate_gradient_points.values
        else:
            self.intermediate_gradient_points = intermediate_gradient_points

        # Ensure counterfactuals is a NumPy array
        if isinstance(counterfactuals, pd.DataFrame):
            self.counterfactuals = (
                counterfactuals.drop(columns=[outcome_name])
                if outcome_name in counterfactuals.columns
                else counterfactuals
            )
            self.counterfactuals = (
                self.counterfactuals.drop(columns='class')
                if 'class' in self.counterfactuals.columns
                else self.counterfactuals
            )
            self.counterfactuals = self.counterfactuals.values
        else:
            self.counterfactuals = counterfactuals

        # Ensure all_gradients is a NumPy array
        if isinstance(all_gradients, pd.DataFrame):
            self.all_gradients = (
                all_gradients.drop(columns=[outcome_name]) 
                if outcome_name in all_gradients.columns
                else all_gradients
            )
            self.all_gradients = self.all_gradients.values
        else:
            self.all_gradients = all_gradients

        reshaped_intermediate_predictions = self.intermediate_predictions.reshape(-1, 1)
        reshaped_counterfactual_predictions = counterfactual_predictions.reshape(-1, 1)

        # Combine intermediate points and counterfactuals
        self.intermediate_gradient_points_stacked = np.concatenate(
            (self.intermediate_gradient_points, self.counterfactuals)
        )
        self.intermediate_predictions_stacked = np.vstack(
            (reshaped_intermediate_predictions, reshaped_counterfactual_predictions)
        )

        # If hidden_layer is True, transform all data by extracting outputs from penultimate layer
        if self.hidden_layer:
            self.original_data_for_predictions = self.original_data.copy()
            self.intermediate_gradient_points_stacked_for_predictions = (
                self.intermediate_gradient_points_stacked.copy()
            )
            # Convert original data
            self.original_data = self.get_layer_output(
                self.model_for_predictions, self.original_data
            )
            # Convert gradient points
            self.intermediate_gradient_points_stacked = self.get_layer_output(
                self.model_for_predictions, self.intermediate_gradient_points_stacked
            )

        self.pixel_colors = None
        self.grid_points_2D = None 
        self.original_data_2D = None 
        self.intermediate_points_2D = None 
        self.grid_predictions_prob = None 
        self.projection_name = projection_name

        # Assign a distinct color for each class
        if self.n_classes == 2:
            self.class_colors = [cm.tab20(0), cm.tab20(19)]
        elif self.n_classes <= 10:
            self.class_colors = [cm.tab20(i) for i in range(self.n_classes)]
        else:
            self.class_colors = [
                to_rgba(cm.tab20(i % 20)) for i in range(n_classes)
            ]
        self.custom_cmap = ListedColormap(self.class_colors)

    def create_grid(self, xmin, xmax, ymin, ymax):
        """
        Generate 2D grid points based on the specified (x, y) bounds and the grid size.

        Parameters
        ----------
        xmin : float
            Minimum X-coordinate in the 2D projection.
        xmax : float
            Maximum X-coordinate in the 2D projection.
        ymin : float
            Minimum Y-coordinate in the 2D projection.
        ymax : float
            Maximum Y-coordinate in the 2D projection.

        Returns
        -------
        None
        """
        # Generate grid points in 2D space
        x_intrvls = np.linspace(xmin, xmax, num=self.grid_size)
        y_intrvls = np.linspace(ymin, ymax, num=self.grid_size)

        # Create a meshgrid for 2D points
        X_grid, Y_grid = np.meshgrid(x_intrvls, y_intrvls)
        self.X_grid = X_grid
        self.Y_grid = Y_grid
        self.grid_points_2D = np.c_[X_grid.ravel(), Y_grid.ravel()]

    def get_layer_output(self, model, X, layer_index=-2):
        """
        Extract the output of a specified layer in a Keras Sequential model.

        Parameters
        ----------
        model : tf.keras.Model
            Keras model from which to extract layer outputs.
        X : np.ndarray
            Input data for which the layer outputs are computed.
        layer_index : int, optional
            Index of the layer for extraction (default is -2, the penultimate layer).

        Returns
        -------
        np.ndarray
            Output of the specified layer for the given input data.
        """
        # If there's only one layer, just return that layer's output
        if len(model.layers) == 1:
            print("Warning: only one layer in the model (no hidden layers).")
            sub_model = tf.keras.Model(
                inputs=model.input,
                outputs=model.layers[0].output
            )
            return sub_model.predict(X)
        
        # Build a sub-model that ends at `layer_index`
        sub_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.layers[layer_index].output
        )
        return sub_model.predict(X)

    def to_grid(self, points_2D, grid_size=300):
        """
        Uniformly rescale and clip a 2D point set to fit into a (grid_size x grid_size) coordinate grid.

        Parameters
        ----------
        points_2D : np.ndarray
            Array of shape (N, 2) containing 2D coordinates.
        grid_size : int, optional
            Size of the grid along each dimension.

        Returns
        -------
        np.ndarray
            Integer grid coordinates in the range [0, grid_size-1].
        """
        # Shift so the minimum corner is at (0, 0)
        shifted = points_2D - self.min_xy

        # Compute uniform scale
        ranges = self.max_xy - self.min_xy
        largest_dim = np.max(ranges)
        if largest_dim == 0:
            # Degenerate case: all points are identical
            return np.full_like(points_2D, (grid_size - 1) // 2, dtype=int)

        scale = (grid_size - 1) / largest_dim
        scaled = shifted * scale

        # Clip and convert to int
        scaled = np.clip(scaled, 0, grid_size - 1).astype(int)
        return scaled

    def fit_points_2D(self, path=None, input_path=None):
        """
        Project the original and intermediate points to 2D using the specified projection method.
        Then create a grid, optionally filter using Jaccard, and plot some initial scatter plots.

        Parameters
        ----------
        path : str, optional
            Directory path where images will be saved.
        input_path : str, optional
            Path to the SSNP model if comparison is True.

        Returns
        -------
        None
        """
        if self.projection_method == 'umap':
            # Project data to 2D using UMAP
            self.umap_reducer = umap.UMAP(n_components=2, n_neighbors=15, n_epochs=200)
            self.original_data_2D = self.umap_reducer.fit_transform(self.original_data)

            self.min_xy = self.original_data_2D.min(axis=0)
            self.max_xy = self.original_data_2D.max(axis=0)

            # Scale and clip original points to match the grid size
            self.normalized_original = (
                self.scaler_2D.fit_transform(self.original_data_2D).astype('float32')
                * (self.grid_size - 1)
            )
            self.normalized_original = np.clip(
                self.normalized_original, 0, self.grid_size - 1
            ).astype(int)

        elif self.projection_method == 'ssnp':
            # Project data to 2D using SSNP
            self.ssnp_reducer = SSNP()
            if self.comparison and self.projection_name is not None:
                # Load a previously trained model if in comparison mode
                self.ssnp_reducer.load_model(
                    os.path.join(input_path, f'{self.dataset_name}_ssnp_{self.version}.h5')
                )
            else:
                # Train a new SSNP model
                patience = 5
                epochs = 200
                verbose = False
                self.ssnp_reducer = SSNP(
                    epochs=epochs,
                    verbose=verbose,
                    patience=patience,
                    opt='adam',
                    bottleneck_activation='linear'
                )
                self.ssnp_reducer.fit(self.original_data, self.target)
        
            self.original_data_2D = self.ssnp_reducer.transform(self.original_data)
            self.min_xy = self.original_data_2D.min(axis=0)
            self.max_xy = self.original_data_2D.max(axis=0)

            self.normalized_original = (
                self.scaler_2D.fit_transform(self.original_data_2D).astype('float32')
                * (self.grid_size - 1)
            )
            self.normalized_original = np.clip(
                self.normalized_original, 0, self.grid_size - 1
            ).astype(int)

        elif self.projection_method == 'lamp':
            # Use a sample of the data
            sample_size = len(self.original_data)
            X_sample = self.original_data[
                random.sample(range(len(self.original_data)), sample_size), :
            ]

            # 2D projection of the sample with TSNE
            start = timer()
            y_sample = TSNE(n_components=2, perplexity=12, random_state=0).fit_transform(X_sample)

            # LAMP
            self.lamp_reducer = Lamp(nr_neighbors=12).fit(
                X_sample=X_sample, y_sample=y_sample
            )

            self.original_data_2D = self.lamp_reducer.transform(self.original_data)

            end = timer()
            print('Lamp took {0} to execute'.format(timedelta(seconds=end - start)))

            self.min_xy = self.original_data_2D.min(axis=0)
            self.max_xy = self.original_data_2D.max(axis=0)
            self.normalized_original = self.to_grid(
                self.original_data_2D, grid_size=self.grid_size
            )

        elif self.projection_method == 'PCA':
            # Use PCA
            self.pca_reducer = PCA(n_components=2)
            self.original_data_2D = self.pca_reducer.fit_transform(self.original_data)

            self.min_xy = self.original_data_2D.min(axis=0)
            self.max_xy = self.original_data_2D.max(axis=0)

            self.normalized_original = self.original_data_2D
        else:
            # Directly use the provided 2D data if no projection method is specified
            self.original_data_2D = self.original_data
            self.min_xy = self.original_data_2D.min(axis=0)
            self.max_xy = self.original_data_2D.max(axis=0)
            self.normalized_original = self.original_data_2D

        print("Original data projected to 2D.")

        # Project intermediate points
        if self.projection_method == 'umap':
            self.intermediate_points_2D = self.umap_reducer.transform(
                self.intermediate_gradient_points_stacked
            )
            if self.gradients:
                # Optionally project intermediate gradients separately
                self.intermediate_points_2D_gradients = self.umap_reducer.transform(
                    self.intermediate_gradient_points
                )
            
            self.normalized_intermediate = (
                self.scaler_2D.transform(self.intermediate_points_2D)
                * (self.grid_size - 1)
            )
            self.normalized_intermediate = np.clip(
                self.normalized_intermediate, 0, self.grid_size - 1
            ).astype(int)

            if self.gradients:
                self.normalized_intermediate_gradients = (
                    self.scaler_2D.transform(self.intermediate_points_2D_gradients)
                ).astype('float32') * (self.grid_size - 1)
                self.normalized_intermediate_gradients = np.clip(
                    self.normalized_intermediate_gradients, 0, self.grid_size - 1
                ).astype(int)

        elif self.projection_method == 'ssnp':
            self.intermediate_points_2D = self.ssnp_reducer.transform(
                self.intermediate_gradient_points_stacked
            )
            if self.gradients:
                self.intermediate_points_2D_gradients = self.ssnp_reducer.transform(
                    self.intermediate_gradient_points
                )

            self.normalized_intermediate = (
                self.scaler_2D.transform(self.intermediate_points_2D).astype("float32")
                * (self.grid_size - 1)
            )
            self.normalized_intermediate = np.clip(
                self.normalized_intermediate, 0, self.grid_size - 1
            ).astype(int)

            if self.gradients:
                self.normalized_intermediate_gradients = (
                    self.scaler_2D.transform(self.intermediate_points_2D_gradients).astype('float32')
                    * (self.grid_size - 1)
                )
                self.normalized_intermediate_gradients = np.clip(
                    self.normalized_intermediate_gradients, 0, self.grid_size - 1
                ).astype(int)

        elif self.projection_method == 'lamp':
            self.intermediate_points_2D = self.lamp_reducer.transform(
                self.intermediate_gradient_points_stacked
            )
            if self.gradients:
                self.intermediate_points_2D_gradients = self.lamp_reducer.transform(
                    self.intermediate_gradient_points
                )
            
            self.normalized_intermediate = self.to_grid(
                self.intermediate_points_2D, grid_size=self.grid_size
            )
            if self.gradients:
                self.normalized_intermediate_gradients = self.to_grid(
                    self.intermediate_points_2D_gradients
                )

        elif self.projection_method == 'PCA':
            self.intermediate_points_2D = self.pca_reducer.transform(
                self.intermediate_gradient_points_stacked
            )
            if self.gradients:
                self.intermediate_points_2D_gradients = self.pca_reducer.transform(
                    self.intermediate_gradient_points
                )

            self.normalized_intermediate = self.intermediate_points_2D
            if self.gradients:
                self.normalized_intermediate_gradients = self.intermediate_points_2D_gradients
        else:
            self.intermediate_points_2D = self.intermediate_gradient_points_stacked
            if self.gradients:
                self.intermediate_points_2D_gradients = self.intermediate_gradient_points
            self.normalized_intermediate = self.intermediate_points_2D

        self.combined_normalized_2D = np.vstack(
            (self.normalized_original, self.normalized_intermediate)
        )

        if self.filter_jd:
            # Filter out points with low Jaccard values
            jaccard_values = self.compute_jaccard_and_filter_entire_set(
                k=15, remove_ratio=0.15
            )

        # Create an initial scatter plot of original points and intermediate gradient points
        plt.figure(figsize=(10, 10))
        plt.scatter(
            self.normalized_original[:, 0],
            self.normalized_original[:, 1],
            c=self.target,
            marker='o',
            alpha=0.5,
            label='Original Points'
        )
        plt.scatter(
            self.normalized_intermediate[:, 0],
            self.normalized_intermediate[:, 1],
            c='red',
            marker='x',
            label='Intermediate Gradient Points'
        )
        plt.xlabel(f'{self.projection_method.upper()} Component 1')
        plt.ylabel(f'{self.projection_method.upper()} Component 2')
        plt.title('Scatter Plot of Original and Intermediate Gradient Points in 2D', fontsize=25)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(path, f'{self.version}_lamp_scatter_plot.png'))

        self.combined_2D = np.vstack((self.original_data_2D, self.intermediate_points_2D))

        # Build a grid on the min-max range
        xmin = self.min_xy[0]
        xmax = self.max_xy[0]
        ymin = self.min_xy[1]
        ymax = self.max_xy[1]
        self.create_grid(xmin, xmax, ymin, ymax)
        print("Grid created.")

        # Plot data points colored by predictions
        self.plot_data_with_predictions(path=path)

        # Optionally, plot only the original data points
        self.plot_original_data_points(path=path)

    def compute_jaccard_and_filter_entire_set(self, k=15, remove_ratio=0.15):
        """
        Compute the Jaccard Distance measure for each point by comparing k-nearest neighbors
        in the original high-dimensional space vs. the 2D projection. Remove the bottom `remove_ratio`
        fraction of points with the lowest Jaccard values.

        Parameters
        ----------
        k : int
            Number of neighbors to consider.
        remove_ratio : float
            Fraction of points to remove based on the lowest Jaccard values.

        Returns
        -------
        np.ndarray
            Array of Jaccard values for each point in the combined dataset.
        """
        # 1) Combine original_data + intermediate_gradient_points_stacked
        nD_data = np.vstack((self.original_data, self.intermediate_gradient_points_stacked))
        twoD_data = self.combined_normalized_2D

        if len(nD_data) != len(twoD_data):
            raise ValueError("nD_data and combined_normalized_2D must have same length.")

        # 2) KNN in nD
        nbrs_nd = NearestNeighbors(n_neighbors=k, algorithm='auto')
        nbrs_nd.fit(nD_data)
        _, indices_nd = nbrs_nd.kneighbors(nD_data)

        # 3) KNN in 2D
        nbrs_2d = NearestNeighbors(n_neighbors=k, algorithm='auto')
        nbrs_2d.fit(twoD_data)
        _, indices_2d = nbrs_2d.kneighbors(twoD_data)

        # 4) Compute Jaccard
        jaccard_vals = np.zeros(len(nD_data), dtype=float)
        for i in range(len(nD_data)):
            nd_neighbors = set(indices_nd[i])
            two_d_neighbors = set(indices_2d[i])
            inter = nd_neighbors.intersection(two_d_neighbors)
            union = nd_neighbors.union(two_d_neighbors)
            jaccard_vals[i] = len(inter) / len(union) if len(union) > 0 else 0.0

        self.jaccard_indices_entire = jaccard_vals  # store for reference

        # 5) Remove the bottom 15%
        cutoff_count = int(remove_ratio * len(nD_data))
        sorted_idx = np.argsort(jaccard_vals)  # ascending
        to_remove = sorted_idx[:cutoff_count]
        to_keep = sorted_idx[cutoff_count:]

        # Filter out points
        self.filtered_nD_data = nD_data[to_keep]
        self.filtered_2D_data = twoD_data[to_keep]

        print(f"Removed {cutoff_count} points (lowest {remove_ratio*100:.0f}% JD) from entire set.")
        return jaccard_vals

    def plot_data_with_predictions(self, grid_lines=False, X_test=None, path=None):
        """
        Plot and save an image of the data points (original + intermediate) colored by 
        the model's predicted classes.

        Parameters
        ----------
        grid_lines : bool, optional
            Whether to overlay a grid on the plot.
        X_test : np.ndarray, optional
            If provided, test points are projected and overlaid.
        path : str, optional
            Directory path where the output plot is saved.

        Returns
        -------
        None
        """
        sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=1.2)
        data = np.ones((self.grid_size, self.grid_size, 3))

        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        extent = (0, self.grid_size - 1, 0, self.grid_size - 1)

        # Show the empty background
        ax.imshow(data, origin='lower', extent=extent, alpha=0.2)

        # Optionally show grid lines
        if grid_lines:
            ax.set_xticks(np.linspace(0, self.grid_size - 1, self.grid_size), minor=False)
            ax.set_yticks(np.linspace(0, self.grid_size - 1, self.grid_size), minor=False)
            ax.grid(which='both', color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        # Plot original data points
        scatter1 = ax.scatter(
            self.normalized_original[:, 0],
            self.normalized_original[:, 1],
            c=self.original_predictions,
            vmin=0,
            vmax=self.n_classes - 1,
            cmap=self.custom_cmap,
            s=20,
            edgecolors='black',
            label='Original Data'
        )

        # Plot intermediate gradient points
        scatter2 = ax.scatter(
            self.normalized_intermediate[:, 0],
            self.normalized_intermediate[:, 1],
            c=self.intermediate_predictions_stacked,
            vmin=0,
            vmax=self.n_classes - 1,
            cmap=self.custom_cmap,
            s=20,
            edgecolors='r',
            marker='x',
            label='Intermediate Points'
        )

        # Optionally overlay X_test
        if X_test is not None:
            # Predict on X_test in high-dimensional space
            y_pred_proba = self.model_for_predictions.predict(X_test)
            if self.n_classes == 2:
                # Binary classification threshold
                if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                    y_pred_proba = y_pred_proba[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                # Multi-class
                y_pred = np.argmax(y_pred_proba, axis=1)

            # Currently placeholders if you have a separate projection pipeline for X_test_2D
            X_test_2D = self.X_test_2D.copy()
            scaled_2D = self.scaled_2D.copy()

            scatter3 = ax.scatter(
                scaled_2D[:, 0],
                scaled_2D[:, 1],
                c=y_pred,
                cmap=self.custom_cmap,
                vmin=0,
                vmax=self.n_classes - 1,
                s=20,
                edgecolors='black',
                marker='v',
                label='Test Points',
                alpha=0.5,
                linewidths=0.5
            )

        ax.legend(title="Data Points", loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
        ax.set_xlabel(f'{self.projection_method.upper()} Component 1', fontsize=18)
        ax.set_ylabel(f'{self.projection_method.upper()} Component 2', fontsize=18)
        ax.set_title('Data Points Colored by Predictions', fontsize=25)

        save_path = os.path.join(path, f'{self.version}_predicted_points.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved as {save_path}.")

    def plot_original_data_points(self, grid_lines=False, path=None):
        """
        Plot and save an image showing only the original data points in 2D, color-coded by prediction.

        Parameters
        ----------
        grid_lines : bool, optional
            Whether to overlay a grid on the plot.
        path : str, optional
            Directory path where the output plot is saved.

        Returns
        -------
        None
        """
        sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=1.2)
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        extent = (0, self.grid_size - 1, 0, self.grid_size - 1)

        # Only plot original data points
        scatter = ax.scatter(
            self.normalized_original[:, 0],
            self.normalized_original[:, 1],
            c=self.original_predictions,
            cmap=self.custom_cmap,
            vmin=0,
            vmax=self.n_classes - 1,
            s=20,
            edgecolors='black',
            linewidths=0.5,
            label='Original Data'
        )

        if grid_lines:
            ax.set_xticks(np.linspace(0, self.grid_size - 1, self.grid_size), minor=False)
            ax.set_yticks(np.linspace(0, self.grid_size - 1, self.grid_size), minor=False)
            ax.grid(which='both', color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        ax.legend(title="Legend", loc="upper left", fontsize=16)
        ax.set_xlabel(f'{self.projection_method.upper()} Component 1', fontsize=18)
        ax.set_ylabel(f'{self.projection_method.upper()} Component 2', fontsize=18)
        ax.set_title('Original Data Points in 2D', fontsize=25)

        save_path = os.path.join(path, f'{self.version}_original_data_points.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved as {save_path}.")

    def fit_grid_knn_weighted_interpolation(self, path=None):
        """
        For each pixel in the 2D grid, find the k nearest points (in the 2D embedding)
        which map back to the high-dimensional space. Weight them (Gaussian-based weights)
        to estimate the pixel's high-dimensional coordinates, then use the trained model
        to predict the class for that pixel.

        Parameters
        ----------
        path : str, optional
            Directory path where the output images and results will be saved.

        Returns
        -------
        None
        """
        # Combine high-dimensional data
        if self.hidden_layer:
            combined_hd_data = np.vstack(
                (
                    self.original_data_for_predictions,
                    self.intermediate_gradient_points_stacked_for_predictions
                )
            )
        else:
            combined_hd_data = np.vstack(
                (self.original_data, self.intermediate_gradient_points_stacked)
            )

        self.combined_hd_data = combined_hd_data
        self.hd_ids_combined = np.arange(self.combined_hd_data.shape[0])

        # Combine 2D data
        self.combined_2D_data = np.vstack(
            (self.original_data_2D, self.intermediate_points_2D)
        )
        self.ld_ids_combined = np.arange(self.combined_2D_data.shape[0])

        # Build KDTree in embedding space
        print(f"Constructing KDTree in {self.projection_method.upper()} space with combined data...")
        tree_embedding = KDTree(self.combined_2D_data)
        print("KDTree constructed.")

        num_original = self.original_data.shape[0]
        prediction_prob = []
        predictions = []
        estimated_hd_points_all = []
        total_grid_points = len(self.grid_points_2D)
        batch_size = 100  # process in batches to handle large grids

        for start_idx in range(0, total_grid_points, batch_size):
            end_idx = min(start_idx + batch_size, total_grid_points)
            batch_grid_points = self.grid_points_2D[start_idx:end_idx]

            # Find K nearest neighbors in the 2D embedding
            distances_embedding, indices_embedding = tree_embedding.query(
                batch_grid_points, k=self.number_of_neighbors
            )

            # Retrieve high-dimensional neighbor points
            neighbors_hd = combined_hd_data[indices_embedding]

            # Gaussian weights
            sigma = np.mean(distances_embedding, axis=1, keepdims=True)
            weights = np.exp(- (distances_embedding ** 2) / (2 * sigma ** 2))
            weights = weights / np.sum(weights, axis=1, keepdims=True)

            # Weighted average in high-dimensional space
            estimated_hd_points = np.einsum('ijk,ij->ik', neighbors_hd, weights)
            estimated_hd_points_all.extend(estimated_hd_points)

            # Model prediction on estimated HD points
            batch_predictions_prob = self.model_for_predictions.predict(estimated_hd_points)

            if self.n_classes == 2:
                if batch_predictions_prob.ndim > 1 and batch_predictions_prob.shape[1] > 1:
                    batch_predictions_prob = batch_predictions_prob[:, 1]
                else:
                    batch_predictions_prob = batch_predictions_prob.flatten()
                batch_predictions = (batch_predictions_prob > 0.5).astype(int)
            else:
                batch_predictions = np.argmax(batch_predictions_prob, axis=1)

            predictions.extend(batch_predictions)
            prediction_prob.extend(batch_predictions_prob)

            if start_idx % (batch_size * 10) == 0:
                print(f"Processed {start_idx} / {total_grid_points} grid points.")

        predictions = np.vstack(predictions)
        if self.n_classes > 2:
            predictions = predictions.flatten()
        prediction_prob = np.vstack(prediction_prob)
        print("Predictions computed for all grid points.")

        self.original_predictions = self.original_predictions.reshape(-1, 1)

        # Distances used in alpha-blending
        distances = calculate_distances_to_same_class(
            self.grid_points_2D,
            predictions,
            self.combined_2D,
            (np.vstack((self.original_predictions, self.intermediate_predictions_stacked))).flatten()
        )
        
        distances_grid = distances.reshape((self.grid_size, self.grid_size))

        # Create distance plot
        plot_distance_grid(
            distances_grid,
            version=self.version,
            training_points=self.combined_2D,
            grid_points=self.grid_points_2D,
            grid_size=self.grid_size,
            path=path
        )

        # Map predictions to colors
        self.colors = np.array(
            [self.class_colors[int(pred) % len(self.class_colors)] for pred in predictions]
        )

        # Reshape for the grid
        self.pixel_colors = self.colors.reshape((self.grid_size, self.grid_size, -1))
        self.grid_predictions_prob = prediction_prob.reshape(
            (self.grid_size, self.grid_size, -1)
        )

        # Save the main decision boundary image
        self._save_image(path=path)

        # Save confidence map
        self.save_confidence_image(self.grid_predictions_prob.copy(), self.grid_size, path)

        # Save alpha-blended map
        self.save_alpha_image(distances=distances_grid, path=path)

        self.estimated_hd_points_all = np.vstack(estimated_hd_points_all)

        # Compute MSE between actual HD data and the estimated HD points (for each pixel)
        results = self.assign_points_to_pixels_and_compute_mse()
        with open(f"{path}/MSE_results_ssnp.json", "w") as json_file:
            json.dump(results, json_file)

        estimated_hd_points_df = pd.DataFrame(
            self.estimated_hd_points_all, columns=self.original_data_df.columns
        )
        save_path = os.path.join(path, f'{self.version}_estimated_high_dim_points.csv')
        estimated_hd_points_df.to_csv(save_path, index=False)
        print(f"Estimated high-dimensional points saved as {save_path}.")

    def assign_points_to_pixels_and_compute_mse(self):
        """
        Assign each original 2D data point to its nearest pixel, then compute the MSE
        between the real HD coordinate and the estimated HD point for that pixel.

        Returns
        -------
        dict
            A dictionary of pixel-level results including MSE values, pixel coordinates,
            and a list of assigned data point IDs.
        """
        if not hasattr(self, 'grid_points_2D') or not hasattr(self, 'estimated_hd_points_all'):
            raise ValueError(
                "Grid data or estimated HD points not found. "
                "Make sure you've run fit_grid_knn_weighted_interpolation first."
            )

        # 1. Build KDTree on the pixel coordinates
        pixel_tree = KDTree(self.grid_points_2D)
        dist, idx = pixel_tree.query(self.combined_2D_data, k=1)

        # 2. Group data by pixel index
        pixel_to_ids = defaultdict(list)
        N = self.combined_2D_data.shape[0]
        for i in range(N):
            pixel_index = idx[i, 0]  # the nearest pixel index
            ld_id = self.ld_ids_combined[i]
            hd_id = self.hd_ids_combined[i]
            pixel_to_ids[pixel_index].append((ld_id, hd_id))

        # 3. Compute MSE for each pixel
        pixel_results = {}
        for pixel_index, id_list in pixel_to_ids.items():
            est_hd = self.estimated_hd_points_all[pixel_index]
            mses = []

            for (ld_id, hd_id) in id_list:
                real_hd = self.combined_hd_data[hd_id]
                mse_val = np.mean((est_hd - real_hd)**2)
                mses.append(mse_val)

            pixel_results[int(pixel_index)] = {
                'pixel_coord': self.grid_points_2D[pixel_index].tolist(),
                'ids': [(int(ld), int(hd)) for ld, hd in id_list],
                'mses': mses,
                'mse_mean': float(np.mean(mses)),
                'count': len(mses)
            }

        self.pixel_results = pixel_results
        return pixel_results

    def evaluate_mapping_testset(self, X_test, X_test_trans=None):
        """
        Evaluate how each test sample maps onto the 2D decision boundary and compare
        the model’s original prediction vs. the “pixel label.”

        Steps:
        1) Model prediction for X_test in high-dimensional space => original_label
        2) Project X_test to 2D => X_test_2D
        3) Scale 2D => pixel coordinates
        4) Retrieve the label at that pixel => pixel_label
        5) Return a DataFrame comparing them.

        Parameters
        ----------
        X_test : np.ndarray
            Test data in the original high-dimensional space.

        Returns
        -------
        pd.DataFrame
            Columns: ['x2D', 'y2D', 'pixel_x', 'pixel_y', 'original_label', 'pixel_label'].
        """
        # These are placeholders to demonstrate the idea
        preds_hd = self.preds_hd.copy()   

        if self.n_classes == 2:
            # Binary classification => threshold
            if preds_hd.ndim > 1 and preds_hd.shape[1] > 1:
                preds_hd = preds_hd[:, 1]
            original_label = (preds_hd > 0.5).astype(int)
        else:
            # Multi-class => argmax
            original_label = np.argmax(preds_hd, axis=1)

        # Project X_test to 2D
        X_test_2D = self.X_test_2D.copy()
        scaled_2D = self.scaled_2D.copy()
        if X_test_trans is not None:

            preds_hd_trans = self.preds_hd_trans.copy()

            if self.n_classes == 2:
                # Binary classification => threshold
                if preds_hd_trans.ndim > 1 and preds_hd_trans.shape[1] > 1:
                    preds_hd_trans = preds_hd_trans[:, 1]
                original_label_trans = (preds_hd_trans > 0.5).astype(int)
            else:
                # Multi-class => argmax
                original_label_trans = np.argmax(preds_hd_trans, axis=1)

            X_test_2D_trans = self.X_test_2D_trans.copy()
            scaled_2D_trans = self.scaled_2D_trans.copy()

            pixel_coords_trans = scaled_2D_trans.astype(int)

            pixel_labels_trans = []
            for i in range(len(pixel_coords_trans)):
                px = pixel_coords_trans[i, 0]
                py = pixel_coords_trans[i, 1]
                if (px < 0 or px >= self.grid_size or py < 0 or py >= self.grid_size):
                    pixel_labels_trans.append(-1)
                    continue

                if self.n_classes == 2:
                    prob = self.grid_predictions_prob[py, px]
                    pixel_label_trans = 1 if prob > 0.5 else 0
                else:
                    pixel_label_trans = np.argmax(self.grid_predictions_prob[py, px])
                pixel_labels_trans.append(pixel_label_trans)

            pixel_labels_trans = np.array(pixel_labels_trans, dtype=int)

            results_df_trans = pd.DataFrame({
                'x2D': X_test_2D_trans[:, 0],
                'y2D': X_test_2D_trans[:, 1],
                'pixel_x': pixel_coords_trans[:, 0],
                'pixel_y': pixel_coords_trans[:, 1],
                'original_label': original_label_trans,
                'pixel_label': pixel_labels_trans
            })

            return results_df_trans

        pixel_coords = scaled_2D.astype(int)

        pixel_labels = []
        for i in range(len(X_test)):
            px = pixel_coords[i, 0]
            py = pixel_coords[i, 1]
            if (px < 0 or px >= self.grid_size or py < 0 or py >= self.grid_size):
                pixel_labels.append(-1)
                continue

            if self.n_classes == 2:
                prob = self.grid_predictions_prob[py, px]
                pixel_label = 1 if prob > 0.5 else 0
            else:
                pixel_label = np.argmax(self.grid_predictions_prob[py, px])
            pixel_labels.append(pixel_label)

        pixel_labels = np.array(pixel_labels, dtype=int)

        results_df = pd.DataFrame({
            'x2D': X_test_2D[:, 0],
            'y2D': X_test_2D[:, 1],
            'pixel_x': pixel_coords[:, 0],
            'pixel_y': pixel_coords[:, 1],
            'original_label': original_label,
            'pixel_label': pixel_labels
        })

        return results_df

    def _save_image(self, path=None):
        """
        Save the prediction grid as an image (PNG). Each pixel's color corresponds
        to a predicted class.

        Parameters
        ----------
        path : str, optional
            Path to save the resulting image.

        Returns
        -------
        None
        """
        # Convert colors to 0-255 for image saving
        self.image_data = (self.pixel_colors * 255).astype(np.uint8)
        self.image = Image.fromarray(
            self.image_data,
            mode='RGBA' if self.image_data.shape[2] == 4 else 'RGB'
        )
        # Rotate and flip to match typical Cartesian orientation
        self.image = self.image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)

        save_path = os.path.join(path, f'{self.version}_mapping_plot.png')
        self.image.save(save_path)
        print(f"Prediction grid image saved as {save_path}.")

    def save_alpha_image(self, distances, path):
        """
        Create an alpha-blended version of the main decision boundary image. 
        Pixels with higher distances to training points of the same class become more transparent.

        Parameters
        ----------
        distances : np.ndarray
            2D array of distances (same dimension as the grid).
        path : str
            Directory to save the image.

        Returns
        -------
        None
        """
        # Convert image_data to float for alpha manipulation
        rgb_data = self.image_data.astype(float) / 255.0
        h, w, c = rgb_data.shape

        # If we already have an RGBA image, ignore the existing alpha
        if c == 4:
            rgb_data = rgb_data[:, :, :3]

        # Normalize distances between 0 and 1
        max_dist = distances.max() if distances.max() != 0 else 1.0
        normalized_dist = distances / max_dist

        # Define alpha: higher distance -> more transparent
        alpha = 1.0 - normalized_dist

        # Combine alpha channel
        rgba_data = np.dstack((rgb_data, alpha))
        rgba_data_uint8 = (rgba_data * 255).astype(np.uint8)

        alpha_image = Image.fromarray(rgba_data_uint8, mode='RGBA')
        alpha_image = alpha_image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)

        save_path = os.path.join(path, f'{self.version}_alpha_plot.png')
        alpha_image.save(save_path)
        print(f"Alpha-blended prediction grid image saved as '{save_path}'.")

    def plot_decision_boundary_with_new_point(self, x_new, path):
        """
        Given a new data point, predict its class in HD, project it to 2D, and overlay
        it on the existing decision boundary map.

        Parameters
        ----------
        x_new : np.ndarray
            A single data point (or batch) in high-dimensional space.
        path : str
            Directory path where the output image is saved.

        Returns
        -------
        None
        """
        # Predict new datapoint's class
        y_pred_proba = self.model_for_predictions.predict(x_new)
        predicted_class = np.argmax(y_pred_proba, axis=1)[0]
        print("Predicted class for the new datapoint:", predicted_class)
        
        # Project the new datapoint into 2D
        if self.projection_method == 'umap':
            x_new_2D = self.umap_reducer.transform(x_new)
            self.normalized_x_new_2D = self.scaler_2D.transform(x_new_2D).astype('float32')
            self.normalized_x_new_2D = self.to_grid(self.normalized_x_new_2D, grid_size=self.grid_size)
        elif self.projection_method == 'ssnp':
            x_new_2D = self.ssnp_reducer.transform(x_new)
            self.normalized_x_new_2D = self.scaler_2D.transform(x_new_2D).astype('float32')
            self.normalized_x_new_2D = np.clip(self.normalized_x_new_2D, 0, self.grid_size - 1).astype(int)
        elif self.projection_method == 'lamp':
            x_new_2D = self.lamp_reducer.transform(x_new)
            self.normalized_x_new_2D = self.to_grid(x_new_2D, grid_size=self.grid_size)
        elif self.projection_method == 'PCA':
            x_new_2D = self.pca_reducer.transform(x_new)
            self.normalized_x_new_2D = self.to_grid(x_new_2D, grid_size=self.grid_size)

        self.image_data = (self.pixel_colors * 255).astype(np.uint8)
        self.image = Image.fromarray(
            self.image_data, 
            mode='RGBA' if self.image_data.shape[2] == 4 else 'RGB'
        )
        self.image = self.image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
        x_coord, y_coord = self.normalized_x_new_2D[0]
        
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image_data, origin='lower')

        # Plot the new point as a red cross
        plt.scatter([x_coord], [y_coord], marker='x', c='red', s=100, linewidths=2, label='New Data Point')

        plt.legend(loc='upper right', fontsize=12)
        plt.axis('off')

        output_path = os.path.join(path, f'{self.version}_new_point_plot.png')
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Prediction grid image with point saved as '{output_path}'.")

    def plot_test_points_on_mapping(
        self,
        X_test,
        path,
        y_test=None,
        X_test_trans=None,
        y_test_trans=None,
        version='',
        X_test_subset=None,
        y_test_subset=None,
        adversarial=False,
        generalize=False
    ):
        """
        Overlay test points on the existing decision boundary image. Optionally overlay
        transformed (adversarial) points, highlighting their shift in the 2D space.

        Parameters
        ----------
        X_test : np.ndarray
            Original test data in high-dimensional space.
        path : str
            Directory path where outputs are saved.
        y_test : np.ndarray, optional
            Ground truth labels of X_test (not always needed).
        X_test_trans : np.ndarray, optional
            Transformed version of X_test (e.g., adversarial or otherwise).
        y_test_trans : np.ndarray, optional
            Labels for transformed test data.
        version : str, optional
            Identifier for naming output images.
        X_test_subset : np.ndarray, optional
            A subset of X_test to plot separately (useful for comparisons).
        y_test_subset : np.ndarray, optional
            Labels of the subset.
        adversarial : bool, optional
            If True, plots additional lines to highlight adversarial transformations.
        generalize : bool, optional
            If True, similar approach for a different scenario.

        Returns
        -------
        None
        """
        # Predict X_test in high-dimensional space
        self.preds_hd = self.model_for_predictions.predict(X_test)
        y_pred_proba = self.preds_hd.copy()
        if X_test_trans is not None:
            self.preds_hd_trans = self.model_for_predictions.predict(X_test_trans)
            y_pred_proba_trans = self.preds_hd_trans.copy()

        # For binary classification
        if self.n_classes == 2:
            if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                y_pred_proba = y_pred_proba[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)

        # If hidden_layer is True, transform X_test
        if self.hidden_layer:
            X_test = self.get_layer_output(self.model_for_predictions, X_test)

        # Project X_test
        if self.projection_method == 'umap':
            self.X_test_2D = self.umap_reducer.transform(X_test)
            X_test_2D = self.X_test_2D.copy()
            scaled_2D = (
                self.scaler_2D.transform(X_test_2D).astype('float32') 
                * (self.grid_size - 1)
            )
            self.scaled_2D = np.clip(scaled_2D, 0, self.grid_size - 1).astype(int)
            scaled_2D = self.scaled_2D.copy()

        elif self.projection_method == 'ssnp':
            self.X_test_2D = self.ssnp_reducer.transform(X_test)
            X_test_2D = self.X_test_2D.copy()
            scaled_2D = (
                self.scaler_2D.transform(X_test_2D).astype('float32')
                * (self.grid_size - 1)
            )
            self.scaled_2D = np.clip(scaled_2D, 0, self.grid_size - 1).astype(int)
            scaled_2D = self.scaled_2D.copy()

        elif self.projection_method == 'lamp':
            self.X_test_2D = self.lamp_reducer.transform(X_test)
            X_test_2D = self.X_test_2D.copy()
            self.scaled_2D = self.to_grid(X_test_2D, grid_size=self.grid_size)
            scaled_2D = self.scaled_2D.copy()

        elif self.projection_method == 'PCA':
            self.X_test_2D = self.pca_reducer.transform(X_test)
            X_test_2D = self.X_test_2D.copy()
            self.scaled_2D = self.to_grid(X_test_2D, grid_size=self.grid_size)
            scaled_2D = self.scaled_2D.copy()

        else:
            # Direct 2D usage
            self.X_test_2D = X_test
            X_test_2D = self.X_test_2D.copy()
            self.scaled_2D = self.to_grid(X_test_2D, grid_size=self.grid_size)
            scaled_2D = self.scaled_2D.copy()
        
        # If there's transformed test data
        if X_test_trans is not None and y_test_trans is not None:
            y_trans_pred_proba = self.model_for_predictions.predict(X_test_trans)
            if self.n_classes == 2:
                if y_trans_pred_proba.ndim > 1 and y_trans_pred_proba.shape[1] > 1:
                    y_trans_pred_proba = y_trans_pred_proba[:, 1]
                y_trans_pred = (y_trans_pred_proba > 0.5).astype(int)
            else:
                y_trans_pred = np.argmax(y_trans_pred_proba, axis=1)

            if self.hidden_layer:
                X_test_trans = self.get_layer_output(self.model_for_predictions, X_test_trans)
            if self.projection_method == 'umap':
                X_test_2D_trans = self.umap_reducer.transform(X_test_trans)
                scaled_2D_trans = (
                    self.scaler_2D.transform(X_test_2D_trans).astype('float32') 
                    * (self.grid_size - 1)
                )
                scaled_2D_trans = np.clip(scaled_2D_trans, 0, self.grid_size - 1).astype(int)
            elif self.projection_method == 'ssnp':
                X_test_2D_trans = self.ssnp_reducer.transform(X_test_trans)
                scaled_2D_trans = (
                    self.scaler_2D.transform(X_test_2D_trans).astype('float32') 
                    * (self.grid_size - 1)
                )
                scaled_2D_trans = np.clip(scaled_2D_trans, 0, self.grid_size - 1).astype(int)
            elif self.projection_method == 'lamp':
                self.X_test_2D_trans = self.lamp_reducer.transform(X_test_trans)
                X_test_2D_trans = self.X_test_2D_trans.copy()
                self.scaled_2D_trans = self.to_grid(X_test_2D_trans, grid_size=self.grid_size)
                scaled_2D_trans = self.scaled_2D_trans.copy()
            elif self.projection_method == 'PCA':
                X_test_2D_trans = self.pca_reducer.transform(X_test_trans)
                scaled_2D_trans = self.to_grid(X_test_2D_trans, grid_size=self.grid_size)
            else:
                X_test_2D_trans = X_test_trans
                scaled_2D_trans = self.to_grid(X_test_2D_trans, grid_size=self.grid_size)

            # Flip Y-coords for display
            scaled_2D_trans[:, 1] = self.grid_size - 1 - scaled_2D_trans[:, 1]

        if X_test_subset is not None:
            y_subset_pred_proba = self.model_for_predictions.predict(X_test_subset)
            if self.n_classes == 2:
                if y_subset_pred_proba.ndim > 1 and y_subset_pred_proba.shape[1] > 1:
                    y_subset_pred_proba = y_subset_pred_proba[:, 1]
                y_subset_pred = (y_subset_pred_proba > 0.5).astype(int)
            else:
                y_subset_pred = np.argmax(y_subset_pred_proba, axis=1)

            if self.hidden_layer:
                X_test_subset = self.get_layer_output(self.model_for_predictions, X_test_subset)
            if self.projection_method == 'umap':
                X_test_2D_subset = self.umap_reducer.transform(X_test_subset)
                scaled_2D_subset = (
                    self.scaler_2D.transform(X_test_2D_subset).astype('float32') 
                    * (self.grid_size - 1)
                )
                scaled_2D_subset = np.clip(scaled_2D_subset, 0, self.grid_size - 1).astype(int)
            elif self.projection_method == 'ssnp':
                X_test_2D_subset = self.ssnp_reducer.transform(X_test_subset)
                scaled_2D_subset = (
                    self.scaler_2D.transform(X_test_2D_subset).astype('float32') 
                    * (self.grid_size - 1)
                )
                scaled_2D_subset = np.clip(scaled_2D_subset, 0, self.grid_size - 1).astype(int)
            elif self.projection_method == 'lamp':
                X_test_2D_subset = self.lamp_reducer.transform(X_test_subset)
                scaled_2D_subset = self.to_grid(X_test_2D_subset, grid_size=self.grid_size)
            elif self.projection_method == 'PCA':
                X_test_2D_subset = self.pca_reducer.transform(X_test_subset)
                scaled_2D_subset = self.to_grid(X_test_2D_subset, grid_size=self.grid_size)

            scaled_2D_subset[:, 1] = self.grid_size - 1 - scaled_2D_subset[:, 1]

        # Load existing decision boundary image
        grid_image_path = os.path.join(path, f'{self.version}_alpha_plot.png')
        grid_image = Image.open(grid_image_path)

        # Flip Y-coords for the main scaled_2D
        scaled_2D[:, 1] = self.grid_size - 1 - scaled_2D[:, 1]

        # Plot with test points
        sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=1.2)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_image, origin='upper')

        # Main test points
        plt.scatter(
            scaled_2D[:, 0],
            scaled_2D[:, 1],
            c=y_pred,
            cmap=self.custom_cmap,
            vmin=0,
            vmax=self.n_classes - 1,
            s=15,
            alpha=0.5,
            edgecolor='black',
            linewidth=0.8,
            label='Test Points'
        )

        # Transformed test points
        if X_test_trans is not None and y_test_trans is not None:
            plt.scatter(
                scaled_2D_trans[:, 0],
                scaled_2D_trans[:, 1],
                c=y_trans_pred,
                cmap=self.custom_cmap,
                vmin=0,
                vmax=self.n_classes - 1,
                s=35,
                alpha=0.5,
                edgecolor='black',
                linewidth=1,
                marker='^',
                label='Transformed Test Points'
            )

        # Subset of points
        if X_test_subset is not None:
            plt.scatter(
                scaled_2D_subset[:, 0],
                scaled_2D_subset[:, 1],
                c=y_subset_pred,
                cmap=self.custom_cmap,
                vmin=0,
                vmax=self.n_classes - 1,
                s=35,
                alpha=0.5,
                edgecolor='black',
                linewidth=1,
                marker='o',
                label='Original Points'
            )

        plt.legend(loc='upper right', fontsize=12)
        plt.title(f"Decision Boundary Mapping with Test Points", fontsize=25)
        plt.axis('off')

        save_path = os.path.join(path, f'{self.version}_overlay_plot{version}.png')
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Test points overlay image saved as '{save_path}'.")

        # If adversarial is True, highlight adversarial transformations
        if adversarial is True:
            if X_test_trans is not None and y_test_trans is not None:
                sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=1.2)
                plt.figure(figsize=(10, 10))
                plt.imshow(grid_image, origin='upper', alpha=0.6)

                if X_test_subset is not None:
                    plt.scatter(
                        scaled_2D_subset[:, 0],
                        scaled_2D_subset[:, 1],
                        c=y_subset_pred,
                        cmap=self.custom_cmap,
                        vmin=0,
                        vmax=self.n_classes - 1,
                        s=35,
                        alpha=1,
                        edgecolor='black',
                        linewidth=1,
                        marker='o',
                        label='Original Test Points'
                    )

                plt.scatter(
                    scaled_2D_trans[:, 0],
                    scaled_2D_trans[:, 1],
                    c=y_trans_pred,
                    cmap=self.custom_cmap,
                    vmin=0,
                    vmax=self.n_classes - 1,
                    s=30,
                    alpha=1,
                    edgecolor='black',
                    linewidth=1,
                    marker='^',
                    label='Transformed Test Points'
                )
                
                num_points = min(len(scaled_2D_subset), len(scaled_2D_trans))
                for i in range(num_points):
                    start_x, start_y = scaled_2D_subset[i]
                    end_x, end_y = scaled_2D_trans[i]
                    alpha = 1.0 if y_subset_pred[i] != y_trans_pred[i] else 0.2

                    plt.annotate(
                        "",
                        xy=(end_x, end_y),
                        xytext=(start_x, start_y),
                        arrowprops=dict(
                            arrowstyle="->",
                            color="black",
                            linestyle='dashed',
                            linewidth=0.5,
                            shrinkA=0,
                            shrinkB=0.5,
                            alpha=alpha
                        )
                    )
                
                overview_data = {
                    'y_subset_pred': y_subset_pred.tolist(),
                    'y_trans_pred': y_trans_pred.tolist()
                }
                overview_path = os.path.join(path, f'{self.version}_overview.json')
                with open(overview_path, 'w') as json_file:
                    json.dump(overview_data, json_file)

                print(f"Overview of y_subset_pred and y_trans_pred saved as '{overview_path}'.")
            
                plt.legend(loc='upper right', fontsize=12)
                plt.title(f"Decision Boundary Mapping with Adversarial Test Points", fontsize=25)
                plt.axis('off')

                save_path = os.path.join(path, f'{self.version}_overlay_plot_trans.png')
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Test trans points overlay image saved as '{save_path}'.")

        # If generalize is True, do a similar approach for other transformations
        if generalize is True:
            if X_test_trans is not None and y_test_trans is not None:
                sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=1.2)
                plt.figure(figsize=(10, 10))
                plt.imshow(grid_image, origin='upper', alpha=0.7)

                if X_test_subset is not None:
                    plt.scatter(
                        scaled_2D_subset[:, 0],
                        scaled_2D_subset[:, 1],
                        c=y_subset_pred,
                        cmap=self.custom_cmap,
                        vmin=0,
                        vmax=self.n_classes - 1,
                        s=35,
                        alpha=1,
                        edgecolor='black',
                        linewidth=1,
                        marker='o',
                        label='Original Test Points'
                    )

                plt.scatter(
                    scaled_2D_trans[:, 0],
                    scaled_2D_trans[:, 1],
                    c=y_trans_pred,
                    cmap=self.custom_cmap,
                    vmin=0,
                    vmax=self.n_classes - 1,
                    s=30,
                    alpha=1,
                    edgecolor='black',
                    linewidth=1,
                    marker='^',
                    label='Transformed Test Points'
                )
                
                num_points = min(len(scaled_2D_subset), len(scaled_2D_trans))
                for i in range(num_points):
                    start_x, start_y = scaled_2D_subset[i]
                    end_x, end_y = scaled_2D_trans[i]
                    alpha = 1.0 if y_subset_pred[i] != y_trans_pred[i] else 0.2

                    plt.annotate(
                        "",
                        xy=(end_x, end_y),
                        xytext=(start_x, start_y),
                        arrowprops=dict(
                            arrowstyle="->",
                            color="black",
                            linewidth=0.5,
                            linestyle="dashed",
                            shrinkA=0,
                            shrinkB=0.2,
                            alpha=alpha
                        )
                    )
                
                overview_data = {
                    'y_subset_pred': y_subset_pred.tolist(),
                    'y_trans_pred': y_trans_pred.tolist()
                }
                overview_path = os.path.join(path, f'{self.version}_overview.json')
                with open(overview_path, 'w') as json_file:
                    json.dump(overview_data, json_file)

                print(f"Overview of y_subset_pred and y_trans_pred saved as '{overview_path}'.")
            
                plt.legend(loc='upper right', fontsize=12)
                plt.title(f"Decision Boundary Mapping with Transformed Test Points", fontsize=25)
                plt.axis('off')

                save_path = os.path.join(path, f'{self.version}_overlay_plot_trans.png')
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Test trans points overlay image saved as '{save_path}'.")

    def plot_class_paths(self, target_class, paths_dict, path):
        """
        Plot a figure for a given target_class using paths stored in a dictionary structure.

        Each entry in `paths_dict[target_class]` is expected to contain:
         - 'paths': list of arrays representing the path for each iteration.
         - 'predictions': corresponding list of model predictions for each step.

        The color of the path is interpolated based on the predicted probability of `target_class`.

        Parameters
        ----------
        target_class : int
            The class we want to visualize the gradient paths toward.
        paths_dict : dict
            Dictionary that maps class -> {
                'paths': [...],
                'predictions': [...]
            }
        path : str
            Directory path where the output image is saved.

        Returns
        -------
        None
        """
        sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=1.2)

        if target_class not in paths_dict:
            print(f"No paths found for class {target_class}.")
            return

        class_data = paths_dict[target_class]
        paths_list = class_data['paths']
        predictions_list = class_data['predictions']
        
        # Convert self.image to grayscale for background
        img = np.array(self.image.convert('RGB'))
        img_gray = np.mean(img, axis=2, keepdims=True)
        img_gray = np.repeat(img_gray, 3, axis=2)
        img_gray = np.rot90(img_gray, 2)
        img_gray = np.fliplr(img_gray)

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.imshow(img_gray.astype(np.uint8), origin='lower')

        def blend(c1, c2, ratio):
            return tuple(c1[i]*(1-ratio) + c2[i]*ratio for i in range(3))

        white = (1.0, 1.0, 1.0)

        # Plot each path in the dictionary
        for p_idx, path_array in enumerate(paths_list):
            preds = predictions_list[p_idx]
            start_class = path_array[0]
            path_array = path_array[1:]  # actual path points after the first element

            if len(preds) != len(path_array):
                print(f"Path and predictions length mismatch at path {p_idx}.")
                continue

            start_class_color = self.class_colors[start_class]
            target_class_color = self.class_colors[target_class]

            # Convert each point in path to 2D
            path_coords = np.vstack([pt[0] for pt in path_array])
            # Predict for path coords
            path_predictions = self.model_for_predictions.predict(path_coords)

            if self.hidden_layer:
                path_coords = self.get_layer_output(self.model_for_predictions, path_coords)

            if self.projection_method == 'umap':
                path_coords_2D = self.umap_reducer.transform(path_coords)
                path_coords_2D_scaled = (
                    self.scaler_2D.transform(path_coords_2D).astype('float32')
                    * (self.grid_size - 1)
                )
                path_coords_2D_scaled = np.clip(path_coords_2D_scaled, 0, self.grid_size - 1).astype(int)

            elif self.projection_method == 'ssnp':
                path_coords_2D = self.ssnp_reducer.transform(path_coords)
                path_coords_2D_scaled = (
                    self.scaler_2D.transform(path_coords_2D).astype('float32')
                    * (self.grid_size - 1)
                )
                path_coords_2D_scaled = np.clip(path_coords_2D_scaled, 0, self.grid_size - 1).astype(int)

            elif self.projection_method == 'lamp':
                path_coords_2D = self.lamp_reducer.transform(path_coords)
                path_coords_2D_scaled = self.to_grid(path_coords_2D, grid_size=self.grid_size)

            elif self.projection_method == 'PCA':
                path_coords_2D = self.pca_reducer.transform(path_coords)
                path_coords_2D_scaled = self.to_grid(path_coords_2D, grid_size=self.grid_size)

            n = path_coords_2D_scaled.shape[0]
            threshold = (1/self.n_classes) + 0.05

            point_colors = []
            pred_targets = []
            for i in range(n):
                pred_target = preds[i][0][target_class]

                if pred_target < threshold:
                    # Blend start_class_color -> white
                    ratio = pred_target / threshold
                    color = blend(start_class_color, white, ratio)
                else:
                    # Blend white -> target_class_color
                    ratio = (pred_target - threshold) / threshold
                    color = blend(white, target_class_color, ratio)
                point_colors.append(color)
                pred_targets.append(pred_target)

            # Draw path lines
            for i in range(n-1):
                ax.plot(
                    [path_coords_2D_scaled[i, 0], path_coords_2D_scaled[i+1, 0]],
                    [path_coords_2D_scaled[i, 1], path_coords_2D_scaled[i+1, 1]],
                    color=point_colors[i],
                    linewidth=2
                )

            # Scatter the path points
            for i in range(n):
                ax.scatter(
                    path_coords_2D_scaled[i, 0],
                    path_coords_2D_scaled[i, 1],
                    color=point_colors[i],
                    s=20
                )

            # Label the starting point
            ax.text(
                path_coords_2D_scaled[0,0],
                path_coords_2D_scaled[0,1],
                f"Start: {start_class}",
                color='white',
                fontsize=8,
                ha='right',
                va='bottom',
                alpha=0.8
            )

        ax.set_title(f"Intermediate Gradient Paths Leading to Class {target_class}", fontsize=25)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")

        output_path = os.path.join(path, f"class_{target_class}_paths_overlay_{self.version}.png")
        plt.savefig(output_path, dpi=300)
        plt.show()
        plt.close()
        print(f"Class {target_class} paths overlay image saved as '{output_path}'.")

    def plot_gradient(self, path):
        """
        Plot the gradient grid overlay. Relies on a custom `plot_gradient_grid` function
        assumed to be imported from `spine.models.visualization.plots`.

        Parameters
        ----------
        path : str
            Directory path to save the gradient plot.

        Returns
        -------
        None
        """
        plot_gradient_grid(
            self.image,
            self.version,
            self.normalized_intermediate_gradients,
            self.grid_points_2D,
            self.grid_size,
            self.all_gradients,
            self.scaler_2D,
            path
        )

    def save_confidence_image(self, prob_matrix, grid_size, output_path):
        """
        Save an image that shows the confidence of the predictions (prob_matrix)
        using a colormap (viridis) and include a colorbar legend.

        Confidence is calculated differently for binary vs. multi-class:
            - Binary: confidence = |p - 0.5| * 2
            - Multi-class: confidence = max(probabilities) - mean(probabilities)

        Parameters
        ----------
        prob_matrix : np.ndarray
            3D array with shape (grid_size, grid_size, n_classes) containing predicted probabilities.
        grid_size : int
            Size of the grid along each dimension.
        output_path : str
            Path to save the resulting confidence image.

        Returns
        -------
        None
        """
        if self.n_classes == 2:
            # For binary classification
            confidence_matrix = np.abs(prob_matrix - 0.5) * 2
        else:
            # For multi-class
            confidence_matrix = np.max(prob_matrix, axis=-1) - np.mean(prob_matrix, axis=-1)

        norm = Normalize(vmin=0, vmax=1)
        cmap = cm.viridis

        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        im = ax.imshow(
            confidence_matrix, cmap=cmap, norm=norm, origin='lower', interpolation='none'
        )
        ax.grid(False)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Confidence', rotation=-90, va="bottom", fontsize=18)
        ax.set_title("Confidence Map", fontsize=25, pad=20)
        plt.tight_layout()
        os.makedirs(output_path, exist_ok=True)
        save_path = os.path.join(output_path, f'{self.version}_confidence_plot.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        print(f"Confidence map image saved to {save_path}")
