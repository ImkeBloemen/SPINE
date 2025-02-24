import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler
import umap.umap_ as umap
import pickle
import matplotlib.cm as cm
from matplotlib.colors import to_rgba
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from sklearn.utils.extmath import cartesian
from VAE_DBS.models.transformers.ssnp import SSNP
from joblib import load
from matplotlib.colors import ListedColormap
from VAE_DBS.visualization.plots import *
from sklearn.decomposition import PCA
from collections import defaultdict
import json
from VAE_DBS.models.transformers.lamp_scaled import Lamp
from sklearn.manifold import MDS,TSNE
from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing
import tensorflow as tf

def rotate_points(points, angle):
    """
    Rotate points by a given angle around the origin.
    angle is in radians.
    """
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    return points @ rotation_matrix.T

class PredictionMap:
    def __init__(self, grid_size, original_data,
                 number_of_neighbors, model_for_predictions, scaler_path_2D: str = None, projection_method='umap', projection_name=None,
                 original_predictions=None,
                 outcome_name='target', n_classes=2, version=1, comparison=False, dataset_name=None, hidden_layer=False):
        self.grid_size = grid_size
        self.number_of_neighbors = number_of_neighbors
        self.model_for_predictions = model_for_predictions
        if scaler_path_2D is not None:
            self.scaler_2D = load(scaler_path_2D)
        elif scaler_path_2D is None:
            self.scaler_2D = MinMaxScaler()
        self.n_classes = n_classes
        self.outcome_name = outcome_name
        self.original_predictions = original_predictions
        self.version = version
        self.projection_method = projection_method

        self.comparison = comparison
        self.dataset_name = dataset_name
        self.hidden_layer = hidden_layer
    

        # Ensure original_data is a NumPy array for consistent indexing
        if isinstance(original_data, pd.DataFrame):
            self.target = original_data[outcome_name].values
            self.original_data_df = original_data.drop(columns=[outcome_name]) if outcome_name in original_data.columns else original_data
            self.original_data = self.original_data_df.values
            if np.isnan(self.original_data).any():
                print("Input data X contains NaN values.")
        else:
            self.original_data = original_data

        self.pixel_colors = None
        self.grid_points_2D = None  # Placeholder for grid points in 2D space
        self.original_data_2D = None  # To store the 2D projection of original data

        self.grid_predictions_prob = None #Placeholder for grid predictions for confidence plot

        # Projection saving/loading parameters
        self.projection_name = projection_name

        # Define specific colors for class labels 0 and 1
        if self.n_classes == 2:
            self.class_colors = [cm.tab20(0), cm.tab20(19)]
        elif self.n_classes <= 10:
            self.class_colors = [cm.tab20(i) for i in range(self.n_classes)]
        elif self.n_classes >= 10:
            self.class_colors = [to_rgba(cm.tab20(i % 20)) for i in range(n_classes)]
        self.custom_cmap = ListedColormap(self.class_colors)

        if self.hidden_layer:
            self.original_data_for_predictions = self.original_data.copy()
            self.original_data = self.get_layer_output(self.model_for_predictions, self.original_data)


    def create_grid(self, xmin, xmax, ymin, ymax):
        """Generates 2D grid points based on specified grid size."""
        
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
        Extracts the output of 'layer_index' in a Keras Sequential model.
        By default, layer_index = -2 (the penultimate layer),
        but you can set it to any valid index (0, 1, -1, etc.).
        
        If the model has only one layer, we return the output of that layer.
        """
        # If there's only one layer, just return that layer's output
        if len(model.layers) == 1:
            print("Warning: only one layer in the model (no hidden layers).")
            # Return the final (and only) layer’s output
            sub_model = tf.keras.Model(inputs=model.input, outputs=model.layers[0].output)
            return sub_model.predict(X)
        
        # Otherwise, build a sub-model that ends at `layer_index`
        sub_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.layers[layer_index].output
        )
        return sub_model.predict(X)

    def to_grid(self, points_2D, grid_size=300):
        """
        Uniformly rescale and translate a 2D point set so it fits
        into a (grid_size x grid_size) coordinate grid.
        """

        # 2. Shift so min corner is at (0, 0)
        shifted = points_2D - self.min_xy

        # 3. Compute uniform scale
        ranges = self.max_xy - self.min_xy  # shape (2,)
        largest_dim = np.max(ranges)
        if largest_dim == 0:
            # Degenerate case: all points are identical
            return np.full_like(points_2D, (grid_size - 1) // 2, dtype=int)

        scale = (grid_size - 1) / largest_dim
        scaled = shifted * scale

        # 4. Clip and convert to int
        scaled = np.clip(scaled, 0, grid_size - 1).astype(int)
        return scaled



    def fit_points_2D(self, path=None, input_path=None):
        """Predicts and creates a grid color map based on model predictions."""
        if self.projection_method == 'umap':
            # Project original high-dimensional data to 2D using UMAP
            self.umap_reducer = umap.UMAP(n_components=2,
                                          n_neighbors=15,
                                          n_epochs=200)
            self.original_data_2D = self.umap_reducer.fit_transform(self.original_data)

            self.min_xy = self.original_data_2D.min(axis=0)
            self.max_xy = self.original_data_2D.max(axis=0)

            self.normalized_original = self.scaler_2D.fit_transform(self.original_data_2D).astype('float32') * (self.grid_size - 1)
            self.normalized_original = np.clip(self.normalized_original, 0, self.grid_size - 1).astype(int)
            # self.normalized_original = self.to_grid(self.normalized_original, grid_size=self.grid_size)

        elif self.projection_method == 'ssnp':
            self.ssnp_reducer = SSNP()
            if self.comparison == True and self.projection_name is not None:
                self.ssnp_reducer.load_model(os.path.join(input_path, f'{self.dataset_name}_ssnp_{self.version}.h5'))
            else:
                grid_size = 300
                patience = 5
                epochs = 200
                min_delta = 0.05
                verbose = False
                self.ssnp_reducer = SSNP(epochs=epochs, verbose=verbose, patience=patience, opt='adam', bottleneck_activation='linear')
                self.ssnp_reducer.fit(self.original_data, self.target)
            # Project the original data using the loaded SSNP model
            self.original_data_2D = self.ssnp_reducer.transform(self.original_data)

            self.min_xy = self.original_data_2D.min(axis=0)
            self.max_xy = self.original_data_2D.max(axis=0)

            self.normalized_original = self.scaler_2D.fit_transform(self.original_data_2D).astype('float32') * (self.grid_size - 1)
            self.normalized_original = np.clip(self.normalized_original, 0, self.grid_size - 1).astype(int)
            # self.normalized_original = self.to_grid(self.normalized_original, grid_size=self.grid_size)

        elif self.projection_method == 'lamp':
            
            sample_size = len(self.original_data)
            X_sample = self.original_data[random.sample(range(len(self.original_data)), sample_size), :]

            # project the sample
            start = timer()
            y_sample = TSNE(n_components=2,
                            perplexity=12,
                            random_state=0).fit_transform(X_sample)

            self.lamp_reducer = Lamp(nr_neighbors=12).fit(X_sample=X_sample,
                                                y_sample=y_sample)

            self.original_data_2D = self.lamp_reducer.transform(X=self.original_data)

            end = timer()

            print('Lamp took {0} to execute'.format(timedelta(seconds=end - start)))

            # self.normalized_original = self.original_data_2D * (self.grid_size - 1)
            # self.normalized_original = np.clip(self.normalized_original, 0, self.grid_size - 1).astype(int)
            self.min_xy = self.original_data_2D.min(axis=0)
            self.max_xy = self.original_data_2D.max(axis=0)
            self.normalized_original = self.to_grid(self.original_data_2D, grid_size=self.grid_size)
        
        elif self.projection_method == 'PCA':
            self.pca_reducer = PCA(n_components=2)
            self.original_data_2D = self.pca_reducer.fit_transform(self.original_data)

            self.min_xy = self.original_data_2D.min(axis=0)
            self.max_xy = self.original_data_2D.max(axis=0)

            self.normalized_original = self.original_data_2D

        print("Original data projected to 2D.")


        self.combined_normalized_2D = self.normalized_original

        # Create a scatter plot of the original points and intermediate gradient points in 2D
        plt.figure(figsize=(10, 10))
        plt.scatter(self.normalized_original[:, 0], self.normalized_original[:, 1], c=self.target, marker='o', alpha=0.5, label='Original Points')
        plt.xlabel(f'{self.projection_method.upper()} Component 1')
        plt.ylabel(f'{self.projection_method.upper()} Component 2')
        plt.title('Scatter Plot of Original and Intermediate Gradient Points in 2D', fontsize=25)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(path, f'{self.version}_lamp_scatter_plot.png'))

        self.combined_2D = self.original_data_2D

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


    def plot_data_with_predictions(self, grid_lines=False, X_test=None, path=None):
        """Plots data points colored by predictions and saves the image."""

        sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=1.2)
        data = np.ones((self.grid_size, self.grid_size, 3))

        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        extent = (0, self.grid_size - 1, 0, self.grid_size - 1)
        ax.imshow(data, origin='lower', extent=extent, alpha=0.2)

        # Overlay the grid lines
        # Overlay the grid lines
        if grid_lines:
            ax.set_xticks(np.linspace(0, self.grid_size - 1, self.grid_size), minor=False)
            ax.set_yticks(np.linspace(0, self.grid_size - 1, self.grid_size), minor=False)
            ax.grid(which='both', color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        # Overlay original data points
        scatter1 = ax.scatter(
            self.normalized_original[:, 0],
            self.normalized_original[:, 1],
            c=self.original_predictions,
            cmap=self.custom_cmap,
            s=20,
            edgecolors='k',
            label='Original Data'
        )

        if X_test is not None:
            # Step 0: Predict X_test in high-dimensional space
            y_pred_proba = self.model_for_predictions.predict(X_test)
            if self.n_classes == 2:
                # Binary classification => threshold
                if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                    # E.g. model gave [p0, p1], we consider p1 as the "positive" class
                    y_pred_proba = y_pred_proba[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                # Multi-class => take arg
                y_pred = np.argmax(y_pred_proba, axis=1)
            # Step 1: Project X_test to 2D
            if self.projection_method == 'umap':
                X_test_2D = self.umap_reducer.transform(X_test)
                scaled_2D = self.scaler_2D.transform(X_test_2D).astype('float32') * (self.grid_size - 1)
                scaled_2D = np.clip(scaled_2D, 0, self.grid_size - 1).astype(int)
            elif self.projection_method == 'ssnp':
                X_test_2D = self.ssnp_reducer.transform(X_test)
                scaled_2D = self.scaler_2D.transform(X_test_2D).astype('float32') * (self.grid_size - 1)
                scaled_2D = np.clip(scaled_2D, 0, self.grid_size - 1).astype(int)
            elif self.projection_method == 'lamp':
                # X_test_2D = self.lamp_reducer.transform(X_test)
                # scaled_2D = self.to_grid(X_test_2D, grid_size=self.grid_size)
                scaled_2D = self.scaled_2D.copy()
            elif self.projection_method == 'PCA':
                X_test_2D = self.pca_reducer.transform(X_test)
                scaled_2D = self.to_grid(X_test_2D, grid_size=self.grid_size)
            else:
                raise ValueError(f"Unsupported projection_method={self.projection_method}")
            

            scatter3 = ax.scatter(
                    scaled_2D[:, 0],
                    scaled_2D[:, 1],
                    c=y_pred,
                    cmap=self.custom_cmap,
                    s=20,
                    edgecolors='b',
                    marker='v',
                    label='Test Points',
                    alpha=0.5,
                    linewidths=0.5
                )

        # Customize legend
        ax.legend(title="Data Points", loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)

        # Set labels and title
        ax.set_xlabel(f'{self.projection_method.upper()} Component 1', fontsize=18)
        ax.set_ylabel(f'{self.projection_method.upper()} Component 2', fontsize=18)
        ax.set_title('Data Points Colored by Predictions', fontsize=25)

        # Save the plot
        save_path = os.path.join(path, f'{self.version}_predicted_points.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved as {save_path}.")

    def plot_original_data_points(self, grid_lines=False, path=None):
        """Plots and saves an image showing only the original data points in 2D."""
        sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=1.2)
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        extent = (0, self.grid_size - 1, 0, self.grid_size - 1)

        # Plot only the original data points
        scatter = ax.scatter(
            self.normalized_original[:, 0],
            self.normalized_original[:, 1],
            c=self.original_predictions,
            cmap=self.custom_cmap,
            s=20,
            edgecolors='k',
            linewidths=0.5,
            label='Original Data'
        )

        # Overlay the grid lines
        if grid_lines:
            ax.set_xticks(np.linspace(0, self.grid_size - 1, self.grid_size), minor=False)
            ax.set_yticks(np.linspace(0, self.grid_size - 1, self.grid_size), minor=False)
            ax.grid(which='both', color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        # Add legend and labels
        ax.legend(title="Legend", loc="upper left", fontsize=16)
        ax.set_xlabel(f'{self.projection_method.upper()} Component 1', fontsize=18)
        ax.set_ylabel(f'{self.projection_method.upper()} Component 2', fontsize=18)
        ax.set_title('Original Data Points in 2D', fontsize=25)

        # Save the figure
        save_path = os.path.join(path, f'{self.version}_original_data_points.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved as {save_path}.")


    def fit_grid_multilateration(self, path=None):

        # Combine high-dimensional data
        if self.hidden_layer:
            combined_hd_data = self.original_data_for_predictions
        else:
            # Combine high-dimensional data
            combined_hd_data = self.original_data
        # combined_hd_data = self.original_data
        self.combined_hd_data = combined_hd_data
        self.hd_ids_combined = np.arange(self.combined_hd_data.shape[0])

        # Combine corresponding 2D projections
        self.combined_2D_data = self.original_data_2D
        self.ld_ids_combined = np.arange(self.combined_2D_data.shape[0])

        # Build KDTree using the combined 2D data
        print(f"Constructing KDTree in {self.projection_method.upper()} space with combined data...")
        tree_embedding = KDTree(self.combined_2D_data)
        print("KDTree constructed.")

        # Total number of original data points for index mapping
        num_original = self.original_data.shape[0]

        prediction_prob = []
        predictions = []
        estimated_hd_points_all = []
        total_grid_points = len(self.grid_points_2D)
        batch_size = 100  # Adjust as needed
        for start_idx in range(0, total_grid_points, batch_size):
            end_idx = min(start_idx + batch_size, total_grid_points)
            batch_grid_points = self.grid_points_2D[start_idx:end_idx]

            # Find K nearest neighbors in dimensionality reduction space for each grid point
            distances_embedding, indices_embedding = tree_embedding.query(batch_grid_points, k=self.number_of_neighbors)

            # Retrieve the corresponding high-dimensional neighbor points
            neighbors_hd = combined_hd_data[indices_embedding]  # Shape: (batch_size, K, num_features)

            # Compute weights based on distances in UMAP space
            # epsilon = 1e-6  # To avoid division by zero
            # weights = 1 / (distances_embedding + epsilon)
            # weights = weights / np.sum(weights, axis=1, keepdims=True)  # Normalize weights

            ##Changed: Use Gaussian weights instead of inverse distances
            sigma = np.mean(distances_embedding, axis=1, keepdims=True)  # or use local adaptive sigma
            weights = np.exp(- (distances_embedding ** 2) / (2 * sigma ** 2))
            weights = weights / np.sum(weights, axis=1, keepdims=True)

            # Estimate high-dimensional positions
            estimated_hd_points = np.einsum('ijk,ij->ik', neighbors_hd, weights)
            estimated_hd_points_all.extend(estimated_hd_points)

            # Make predictions on the estimated high-dimensional points
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

        distances = calculate_distances_to_same_class(self.grid_points_2D, 
                                                           predictions, 
                                                           self.combined_2D, 
                                                           self.original_predictions.flatten())
        
        distances_grid = distances.reshape((self.grid_size, self.grid_size))

        plot_distance_grid(distances_grid, version=self.version, training_points=self.combined_2D, grid_points=self.grid_points_2D, grid_size=self.grid_size, path=path)

        # Map predictions to colors
        self.colors = np.array([self.class_colors[int(pred) % len(self.class_colors)] for pred in predictions])

        # Reshape colors to the grid shape
        self.pixel_colors = self.colors.reshape((self.grid_size, self.grid_size, -1))
        self.grid_predictions_prob = prediction_prob.reshape((self.grid_size, self.grid_size, -1))
        # Save the image
        self._save_image(path=path)

        self.save_confidence_image(self.grid_predictions_prob.copy(), self.grid_size, path)

        self.save_alpha_image(distances=distances_grid, path=path)

        self.estimated_hd_points_all = np.vstack(estimated_hd_points_all)

        results = self.assign_points_to_pixels_and_compute_mse()
        with open(f"{path}/MSE_results_ssnp.json", "w") as json_file:
            json.dump(results, json_file)

        estimated_hd_points_df = pd.DataFrame(self.estimated_hd_points_all, columns=self.original_data_df.columns)
        save_path = os.path.join(path, f'{self.version}_estimated_high_dim_points.csv')
        estimated_hd_points_df.to_csv(save_path, index=False)
        print(f"Estimated high-dimensional points saved as {save_path}.")

    def assign_points_to_pixels_and_compute_mse(self):
        # Safety check
        if not hasattr(self, 'grid_points_2D') or not hasattr(self, 'estimated_hd_points_all'):
            raise ValueError("Grid data or estimated HD points not found. Make sure you've run fit_grid_multilateration first.")

        # 1. Build KDTree on the pixels
        pixel_tree = KDTree(self.grid_points_2D)
        dist, idx = pixel_tree.query(self.combined_2D_data, k=1)

        # 2. Group data by pixel index
        pixel_to_ids = defaultdict(list)
        N = self.combined_2D_data.shape[0]
        for i in range(N):
            pixel_index = idx[i, 0]  # the index of the nearest pixel
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

            pixel_results[int(pixel_index)] = {  # Ensure pixel_index is serializable
                'pixel_coord': self.grid_points_2D[pixel_index].tolist(),  # Convert to list
                'ids': [(int(ld), int(hd)) for ld, hd in id_list],  # Convert to serializable types
                'mses': mses,
                'mse_mean': float(np.mean(mses)),  # Convert to float
                'count': len(mses)
            }

        self.pixel_results = pixel_results
        return pixel_results

    def evaluate_mapping_testset(self, X_test):
        """
        Evaluate how each test sample maps onto the 2D decision boundary
        and compare the model’s original prediction vs. the “pixel label.”

        Steps:
        1) Model prediction for X_test in high-dimensional space => original_label
        2) Project X_test to 2D => X_test_2D
        3) Scale 2D => pixel coordinates
        4) Retrieve the label at that pixel => pixel_label
        5) Return a DataFrame comparing them.

        Assumes:
        - self.grid_predictions_prob is populated (via fit_grid_multilateration).
        - self.grid_size, self.projection_method, etc. are set.
        - If self.comparison == True, uses self.scaler_2D to scale coords;
            otherwise uses the margin-based normalization from fit_points_2D().

        Returns
        -------
        pd.DataFrame with columns:
        [
            'x2D', 'y2D', 'pixel_x', 'pixel_y',
            'original_label', 'pixel_label'
        ]
        (plus more columns if desired)
        """

        # 1) Predict with the model in HD space
        #    shape => (N, num_classes) or (N,) depending on your model's output
        # self.preds_hd = self.model_for_predictions.predict(X_test)
        preds_hd = self.preds_hd.copy()

        # if self.hidden_layer:
        #     X_test = self.get_layer_output(self.model_for_predictions, X_test)
        if self.n_classes == 2:
            # Binary classification => threshold
            if preds_hd.ndim > 1 and preds_hd.shape[1] > 1:
                # E.g. model gave [p0, p1], we consider p1 as the "positive" class
                preds_hd = preds_hd[:, 1]
            original_label = (preds_hd > 0.5).astype(int)
        else:
            # Multi-class => take argmax
            original_label = np.argmax(preds_hd, axis=1)

        # 2) Project X_test to 2D using UMAP or SSNP
        pixel_coords = []
        if self.projection_method == 'umap':
            # X_test_2D = self.umap_reducer.transform(X_test)
            # scaled_2D = self.scaler_2D.transform(X_test_2D).astype('float32') * (self.grid_size - 1)
            # scaled_2D = np.clip(scaled_2D, 0, self.grid_size - 1).astype(int)
            X_test_2D = self.X_test_2D.copy()
            scaled_2D = self.scaled_2D.copy()
        elif self.projection_method == 'ssnp':
            # X_test_2D = self.ssnp_reducer.transform(X_test)
            # scaled_2D = self.scaler_2D.transform(X_test_2D).astype('float32') * (self.grid_size - 1)
            # scaled_2D = np.clip(scaled_2D, 0, self.grid_size - 1).astype(int)
            X_test_2D = self.X_test_2D.copy()
            scaled_2D = self.scaled_2D.copy()
        elif self.projection_method == 'lamp':
            # X_test_2D = self.lamp_reducer.transform(X_test)
            # scaled_2D = self.to_grid(X_test_2D, grid_size=self.grid_size)
            X_test_2D = self.X_test_2D.copy()
            scaled_2D = self.scaled_2D.copy()
        elif self.projection_method == 'PCA':
            # X_test_2D = self.pca_reducer.transform(X_test)
            # scaled_2D = self.to_grid(X_test_2D, grid_size=self.grid_size)
            X_test_2D = self.X_test_2D.copy()
            scaled_2D = self.scaled_2D.copy()
        else:
            raise ValueError(f"Unsupported projection_method={self.projection_method}")
        
        pixel_coords = scaled_2D.astype(int)

    
        # 4) For each pixel, retrieve the decision-boundary label
        pixel_labels = []
        for i in range(len(X_test)):
            px = pixel_coords[i, 0]
            py = pixel_coords[i, 1]
            if (px < 0 or px >= self.grid_size or py < 0 or py >= self.grid_size):
                # out of bounds => skip or default to -1
                pixel_labels.append(-1)
                continue

            if self.n_classes == 2:
                prob = self.grid_predictions_prob[py, px]
                pixel_label = 1 if prob > 0.5 else 0
            else:
                # shape => (num_classes,)
                pixel_label = np.argmax(self.grid_predictions_prob[py, px])
            pixel_labels.append(pixel_label)

        pixel_labels = np.array(pixel_labels, dtype=int)

        # 5) Construct a results DataFrame
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
        """Saves the prediction grid as an image."""
        # Convert colors to 0-255 range for image saving
        self.image_data = (self.pixel_colors * 255).astype(np.uint8)
        self.image = Image.fromarray(self.image_data, mode='RGBA' if self.image_data.shape[2] == 4 else 'RGB')
        self.image = self.image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
        save_path = os.path.join(path, f'{self.version}_mapping_plot.png')
        self.image.save(save_path)
        print(f"Prediction grid image saved as {save_path}.")

    def save_alpha_image(self, distances, path):
        """
        Create a plot where the image data from self.image is displayed, 
        but the transparency of these colors is determined by the distance values.
        Pixels with a higher distance should have a lower alpha (more transparent).
        """

        # Convert self.image_data to float [0,1] just to be safe
        rgb_data = self.image_data.astype(float) / 255.0  # shape: (grid_size, grid_size, 3 or 4)
        h, w, c = rgb_data.shape

        # Ensure we have an RGB image (not RGBA) to start with
        # If it's RGBA, ignore the existing alpha
        if c == 4:
            rgb_data = rgb_data[:, :, :3]

        # Normalize distances between 0 and 1
        max_dist = distances.max() if distances.max() != 0 else 1.0
        normalized_dist = distances / max_dist  # range [0,1]

        # Define alpha: higher distance -> more transparent
        # For example, alpha = 1 - normalized distance
        alpha = 1.0 - normalized_dist  # alpha in [0,1], where 0 is fully transparent, 1 is fully opaque

        # Stack alpha channel to create an RGBA image
        rgba_data = np.dstack((rgb_data, alpha))

        # Convert back to 0-255 for saving as an image
        rgba_data_uint8 = (rgba_data * 255).astype(np.uint8)

        # Create an Image from this RGBA data
        alpha_image = Image.fromarray(rgba_data_uint8, mode='RGBA')
        alpha_image = alpha_image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)

        # Optionally rotate or flip if needed (based on your existing transformations):
        # alpha_image = alpha_image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)

        # Save the image
        save_path = os.path.join(path, f'{self.version}_alpha_plot.png')
        alpha_image.save(save_path)
        print(f"Alpha-blended prediction grid image saved as '{save_path}'.")

    def plot_decision_boundary_with_new_point(self, x_new):
        """
        Given a decision boundary (distance_grid) and its corresponding grid_points_2D,
        this function takes a new datapoint, predicts its class using `clf`,
        projects it into the 2D space using `transform_to_2D`, and plots it
        as a cross marker on top of the decision boundary plot.
        """
        
        # Predict the class of the new datapoint
        y_pred_proba = self.model_for_predictions.predict(x_new)  # if clf is keras model returning probabilities
        predicted_class = np.argmax(y_pred_proba, axis=1)[0]
        print("Predicted class for the new datapoint:", predicted_class)
        
        # Transform the new datapoint into the 2D space
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

        # Plot the decision boundary (distance_grid)
        #self.image_data = (self.pixel_colors * 255).astype(np.uint8)
        self.image_data = (self.pixel_colors * 255).astype(np.uint8)
        self.image = Image.fromarray(self.image_data, mode='RGBA' if self.image_data.shape[2] == 4 else 'RGB')
        self.image = self.image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
        x_coord, y_coord = self.normalized_x_new_2D[0]
        
        plt.figure(figsize=(10, 10))
        # Display the image_data as an image
        # origin='lower' to keep consistent orientation if needed
        plt.imshow(self.image_data, origin='lower')

        # Plot the point as a red cross
        plt.scatter([x_coord], [y_coord], marker='x', c='red', s=100, linewidths=2, label='New Data Point')

        plt.legend(loc='upper right', fontsize=12)
        plt.axis('off')

        output_path = f"../../../results/images/mapping_images/prediction_grid_image_with_point_{self.version}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Prediction grid image with point saved as '{output_path}'.")

    def plot_test_points_on_mapping(self, X_test, path, y_test=None, X_test_trans=None, y_test_trans=None, version='', X_test_subset=None, adversarial=False, generalize=False):
        """
        Plot the test points on top of the existing mapping image.

        Parameters:
        ----------
        X_test : np.ndarray
            The test data in the original high-dimensional space.
        version : int
            The version of the grid image to save or overlay.

        Returns:
        -------
        None
        """
        # Step 0: Predict X_test in high-dimensional space
        self.preds_hd = self.model_for_predictions.predict(X_test)
        y_pred_proba = self.preds_hd.copy()
        if self.hidden_layer:
            X_test = self.get_layer_output(self.model_for_predictions, X_test)

        if self.n_classes == 2:
            # Binary classification => threshold
            if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                # E.g. model gave [p0, p1], we consider p1 as the "positive" class
                y_pred_proba = y_pred_proba[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            # Multi-class => take arg
            y_pred = np.argmax(y_pred_proba, axis=1)
        # Step 1: Project X_test to 2D
        if self.projection_method == 'umap':
            self.X_test_2D = self.umap_reducer.transform(X_test)
            X_test_2D = self.X_test_2D.copy()
            scaled_2D = self.scaler_2D.transform(X_test_2D).astype('float32') * (self.grid_size - 1)
            self.scaled_2D = np.clip(scaled_2D, 0, self.grid_size - 1).astype(int)
            scaled_2D = self.scaled_2D.copy()
        elif self.projection_method == 'ssnp':
            self.X_test_2D = self.ssnp_reducer.transform(X_test)
            X_test_2D = self.X_test_2D.copy()
            scaled_2D = self.scaler_2D.transform(X_test_2D).astype('float32') * (self.grid_size - 1)
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
            raise ValueError(f"Unsupported projection_method={self.projection_method}")
        
        if X_test_trans is not None and y_test_trans is not None:
            y_trans_pred_proba = self.model_for_predictions.predict(X_test_trans)

            if self.n_classes == 2:
                # Binary classification => threshold
                if y_trans_pred_proba.ndim > 1 and y_trans_pred_proba.shape[1] > 1:
                    # E.g. model gave [p0, p1], we consider p1 as the "positive" class
                    y_trans_pred_proba = y_trans_pred_proba[:, 1]
                y_trans_pred = (y_trans_pred_proba > 0.5).astype(int)
            else:
                # Multi-class => take arg
                y_trans_pred = np.argmax(y_trans_pred_proba, axis=1)

            if self.hidden_layer:
                X_test_trans = self.get_layer_output(self.model_for_predictions, X_test_trans)
            if self.projection_method == 'umap':
                X_test_2D_trans = self.umap_reducer.transform(X_test_trans)
                scaled_2D_trans = self.scaler_2D.transform(X_test_2D_trans).astype('float32') * (self.grid_size - 1)
                scaled_2D_trans = np.clip(scaled_2D_trans, 0, self.grid_size - 1).astype(int)
            elif self.projection_method == 'ssnp':
                X_test_2D_trans = self.ssnp_reducer.transform(X_test_trans)
                scaled_2D_trans = self.scaler_2D.transform(X_test_2D_trans).astype('float32') * (self.grid_size - 1)
                scaled_2D_trans = np.clip(scaled_2D_trans, 0, self.grid_size - 1).astype(int)
            elif self.projection_method == 'lamp':
                X_test_2D_trans = self.lamp_reducer.transform(X_test_trans)
                scaled_2D_trans = self.to_grid(X_test_2D_trans, grid_size=self.grid_size)
            elif self.projection_method == 'PCA':
                X_test_2D_trans = self.pca_reducer.transform(X_test_trans)
                scaled_2D_trans = self.to_grid(X_test_2D_trans, grid_size=self.grid_size)

        if X_test_subset is not None:
            y_subset_pred_proba = self.model_for_predictions.predict(X_test_subset)

            if self.n_classes == 2:
                # Binary classification => threshold
                if y_subset_pred_proba.ndim > 1 and y_subset_pred_proba.shape[1] > 1:
                    # E.g. model gave [p0, p1], we consider p1 as the "positive" class
                    y_subset_pred_proba = y_subset_pred_proba[:, 1]
                y_subset_pred = (y_subset_pred_proba > 0.5).astype(int)
            else:
                # Multi-class => take arg
                y_subset_pred = np.argmax(y_subset_pred_proba, axis=1)
            if self.hidden_layer:
                X_test_subset = self.get_layer_output(self.model_for_predictions, X_test_subset)
            if self.projection_method == 'umap':
                X_test_2D_subset = self.umap_reducer.transform(X_test_subset)
                scaled_2D_subset = self.scaler_2D.transform(X_test_2D_subset).astype('float32') * (self.grid_size - 1)
                scaled_2D_subset = np.clip(scaled_2D_subset, 0, self.grid_size - 1).astype(int)
            elif self.projection_method == 'ssnp':
                X_test_2D_subset = self.ssnp_reducer.transform(X_test_subset)
                scaled_2D_subset = self.scaler_2D.transform(X_test_2D_subset).astype('float32') * (self.grid_size - 1)
                scaled_2D_subset = np.clip(scaled_2D_trans, 0, self.grid_size - 1).astype(int)
            elif self.projection_method == 'lamp':
                X_test_2D_subset = self.lamp_reducer.transform(X_test_subset)
                scaled_2D_subset = self.to_grid(X_test_2D_subset, grid_size=self.grid_size)
            elif self.projection_method == 'PCA':
                X_test_2D_subset = self.pca_reducer.transform(X_test_subset)
                scaled_2D_subset = self.to_grid(X_test_2D_subset, grid_size=self.grid_size)

        # Step 3: Load the existing grid image
        grid_image_path = os.path.join(path, f'{self.version}_alpha_plot.png')
        grid_image = Image.open(grid_image_path)

        # Step 4: Adjust scaled test points for the rotated and flipped image
        # scaled_2D[:, 0] = self.grid_size - 1 - scaled_2D[:, 0]  # Flip X-coordinates
        scaled_2D[:, 1] = self.grid_size - 1 - scaled_2D[:, 1]  # Flip Y-coordinates

        sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=1.2)
        # Step 4: Plot the grid image with test points overlaid
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_image, origin='upper')

        # Overlay test points
        plt.scatter(
            scaled_2D[:, 0], 
            scaled_2D[:, 1], 
            c=y_pred, #if y_test is None else y_test,  # Use the same colors as in the grid_image
            cmap=self.custom_cmap,
            vmin=0,
            vmax=self.n_classes - 1,
            s=15,     # Size of the test points
            alpha=0.5,
            edgecolor='black',  # Add a black edge color
            linewidth=0.8,  # Set the width of the edge line
            label='Test Points'
        )

        if X_test_trans is not None and y_test_trans is not None:
            plt.scatter(
                scaled_2D_trans[:, 0], 
                scaled_2D_trans[:, 1], 
                c=y_trans_pred,  # Use the same colors as in the grid_image
                cmap=self.custom_cmap,
                vmin=0,
                vmax=self.n_classes - 1,
                s=35,     # Size of the test points
                alpha=0.5,
                edgecolor='black',  # Add a black edge color
                linewidth=1,  # Set the width of the edge line
                marker='o',  # Change shape to triangles
                label='Transformed Test Points'
                    )
            
        if X_test_subset is not None:
            plt.scatter(
                scaled_2D_subset[:, 0], 
                scaled_2D_subset[:, 1], 
                c=y_subset_pred,  # Use the same colors as in the grid_image
                cmap=self.custom_cmap,
                vmin=0,
                vmax=self.n_classes - 1,
                s=35,     # Size of the test points
                alpha=0.5,
                edgecolor='black',  # Add a black edge color
                linewidth=1,  # Set the width of the edge line
                marker='^',  # Change shape to triangles
                label='Original Points'
                    )

        # Add a legend and title
        plt.legend(loc='upper right', fontsize=12)
        plt.title(f"Decision Boundary Mapping with Test Points", fontsize=25)
        plt.axis('off')  # Hide axes

        # Save the overlay image
        save_path = os.path.join(path, f'{self.version}_overlay_plot{version}.png')
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Test points overlay image saved as '{save_path}'.")

        if adversarial is True:
            if X_test_trans is not None and y_test_trans is not None:
                sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=1.2)
                # Step 4: Plot the grid image with test points overlaid
                plt.figure(figsize=(10, 10))
                plt.imshow(grid_image, origin='upper', alpha=0.6)

                if X_test_subset is not None:
                    plt.scatter(
                        scaled_2D_subset[:, 0], 
                        scaled_2D_subset[:, 1], 
                        c=y_subset_pred,  # Use the same colors as in the grid_image
                        cmap=self.custom_cmap,
                        vmin=0,
                        vmax=self.n_classes - 1,
                        s=35,     # Size of the test points
                        alpha=1,
                        edgecolor='black',  # Add a black edge color
                        linewidth=1,  # Set the width of the edge line
                        marker='o',  # Change shape to triangles
                        label='Original Test Points'
                            )

                plt.scatter(
                scaled_2D_trans[:, 0], 
                scaled_2D_trans[:, 1], 
                c=y_trans_pred,  # Use the same colors as in the grid_image
                cmap=self.custom_cmap,
                vmin=0,
                vmax=self.n_classes - 1,
                s=30,     # Size of the test points
                alpha=1,
                edgecolor='black',  # Add a black edge color
                linewidth=1,  # Set the width of the edge line
                marker='^',  # Change shape to triangles
                label='Transformed Test Points'
                    )
                
                num_points = min(len(scaled_2D_subset), len(scaled_2D_trans))
                for i in range(num_points):
                    start_x, start_y = scaled_2D_subset[i]
                    end_x,   end_y   = scaled_2D_trans[i]

                    # Determine alpha based on whether the points are predicted in the same class
                    alpha = 1.0 if y_subset_pred[i] != y_trans_pred[i] else 0.2

                    # Using annotate with arrowprops to draw an arrow:
                    plt.annotate(
                        "",
                        xy=(end_x, end_y),
                        xytext=(start_x, start_y),
                        arrowprops=dict(
                            arrowstyle="->",  # arrow style
                            color="black",
                            linestyle='dashed',
                            linewidth=0.5,
                            shrinkA=0,         # tail shrinking
                            shrinkB=0.5,       # head shrinking
                            alpha=alpha
                        )
                    )
                
                # Create an overview of y_subset_pred and y_trans_pred
                overview_data = {
                    'y_subset_pred': y_subset_pred.tolist(),
                    'y_trans_pred': y_trans_pred.tolist()
                }

                # Save the overview as a JSON file
                overview_path = os.path.join(path, f'{self.version}_overview.json')
                with open(overview_path, 'w') as json_file:
                    json.dump(overview_data, json_file)

                print(f"Overview of y_subset_pred and y_trans_pred saved as '{overview_path}'.")
            
                # Add a legend and title
                plt.legend(loc='upper right', fontsize=12)
                plt.title(f"Decision Boundary Mapping with Adversarial Test Points", fontsize=25)
                plt.axis('off')  # Hide axes

                # Save the overlay image
                save_path = os.path.join(path, f'{self.version}_overlay_plot_trans.png')
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Test trans points overlay image saved as '{save_path}'.")

        if generalize is True:
            if X_test_trans is not None and y_test_trans is not None:
                sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=1.2)
                # Step 4: Plot the grid image with test points overlaid
                plt.figure(figsize=(10, 10))
                plt.imshow(grid_image, origin='upper', alpha=0.7)

                if X_test_subset is not None:
                    plt.scatter(
                        scaled_2D_subset[:, 0], 
                        scaled_2D_subset[:, 1], 
                        c=y_subset_pred,  # Use the same colors as in the grid_image
                        cmap=self.custom_cmap,
                        vmin=0,
                        vmax=self.n_classes - 1,
                        s=35,     # Size of the test points
                        alpha=1,
                        edgecolor='black',  # Add a black edge color
                        linewidth=1,  # Set the width of the edge line
                        marker='o',  # Change shape to triangles
                        label='Original Test Points'
                            )

                plt.scatter(
                scaled_2D_trans[:, 0], 
                scaled_2D_trans[:, 1], 
                c=y_trans_pred,  # Use the same colors as in the grid_image
                cmap=self.custom_cmap,
                vmin=0,
                vmax=self.n_classes - 1,
                s=30,     # Size of the test points
                alpha=1,
                edgecolor='black',  # Add a black edge color
                linewidth=1,  # Set the width of the edge line
                marker='^',  # Change shape to triangles
                label='Transformed Test Points'
                    )
                
                num_points = min(len(scaled_2D_subset), len(scaled_2D_trans))
                for i in range(num_points):
                    start_x, start_y = scaled_2D_subset[i]
                    end_x,   end_y   = scaled_2D_trans[i]

                    # Determine alpha based on whether the points are predicted in the same class
                    alpha = 1.0 if y_subset_pred[i] != y_trans_pred[i] else 0.2

                    # Using annotate with arrowprops to draw an arrow:
                    plt.annotate(
                        "",
                        xy=(end_x, end_y),
                        xytext=(start_x, start_y),
                        arrowprops=dict(
                            arrowstyle="->",  # arrow style
                            color="black",
                            linewidth=0.5,
                            linestyle="dashed",  # make the line dashed
                            shrinkA=0,         # tail shrinking
                            shrinkB=0.2,       # head shrinking
                            alpha=alpha
                        )
                    )
                
                # Create an overview of y_subset_pred and y_trans_pred
                overview_data = {
                    'y_subset_pred': y_subset_pred.tolist(),
                    'y_trans_pred': y_trans_pred.tolist()
                }

                # Save the overview as a JSON file
                overview_path = os.path.join(path, f'{self.version}_overview.json')
                with open(overview_path, 'w') as json_file:
                    json.dump(overview_data, json_file)

                print(f"Overview of y_subset_pred and y_trans_pred saved as '{overview_path}'.")
            
                # Add a legend and title
                plt.legend(loc='upper right', fontsize=12)
                plt.title(f"Decision Boundary Mapping with Transformed Test Points", fontsize=25)
                plt.axis('off')  # Hide axes

                # Save the overlay image
                save_path = os.path.join(path, f'{self.version}_overlay_plot_trans.png')
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Test trans points overlay image saved as '{save_path}'.")


    def plot_class_paths(self, target_class, paths_dict):
        """
        Plots a figure for a given target_class using a dictionary structure:
        - paths_dict is a dictionary where each key is a class, and each value is another dictionary:
        {
            'paths': [current_class, path_array_1, path_array_2, ...],
            'predictions': [pred_path_1, pred_path_2, ...]
        }

        Each path_array is of shape (iterations, D). The first column of each path presumably includes the start class (if so, we skip it),
        or as per the new structure, the first element in 'paths' list is the current_class and subsequent items are path arrays.
        
        'predictions' is a list of arrays with shape (iterations, num_classes), giving predicted probabilities for each point.

        Colors:
        - Start from the start_class's color for pred_target=0.
        - Gradually go to white at pred_target=0.5.
        - Then go to target_class color at pred_target=1.0.

        Parameters
        ----------
        version : int or str
            Version identifier for saving the figure.
        target_class : int or str
            The class for which we are plotting these paths.
        img : np.ndarray
            Original background image as a numpy array (H, W, C).
        paths_dict : dict
            Dictionary of paths and predictions indexed by class.
        grid_points : np.ndarray
            2D positions for the background image extent (optional).
        scaler_2D : object
            A scaler to map points_2D into image coordinates (if needed).
        class_colors : dict
            Arary with the RGB colors.
        """
        sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=1.2)

        if target_class not in paths_dict:
            print(f"No paths found for class {target_class}.")
            return

        class_data = paths_dict[target_class]
        # 'paths': [current_class, path_1, path_2, ...], [current_class, path_1, path_2, ...]
        # 'predictions': [pred_1, pred_2, ...], [pred_1, pred_2, ...]
        paths_list = class_data['paths']
        predictions_list = class_data['predictions']
        
        img = np.array(self.image.convert('RGB'))

        # Convert image to grayscale if needed
        # if img.shape[2] == 4:
        #     alpha_channel = img[:, :, 3:]
        #     img_gray = np.mean(img[:, :, :3], axis=2, keepdims=True)
        #     img_gray = np.concatenate([img_gray, alpha_channel], axis=2)
        # else:
        img_gray = np.mean(img, axis=2, keepdims=True)
        img_gray = np.repeat(img_gray, 3, axis=2)
        img_gray = np.rot90(img_gray, 2)
        img_gray = np.fliplr(img_gray)

        fig, ax = plt.subplots(figsize=(10, 10))

        # Display the grayscale background image
        ax.imshow(img_gray.astype(np.uint8), origin='lower')

        # Define helper functions
        def blend(c1, c2, ratio):
            # linear blend between c1 and c2 with ratio in [0,1]
            return tuple(c1[i]*(1-ratio) + c2[i]*ratio for i in range(3))

        white = (1.0, 1.0, 1.0)

        # Plot each path
        for p_idx, path_array in enumerate(paths_list):
            preds = predictions_list[p_idx]  # shape (n, num_classes)
            start_class = path_array[0]
            path_array = path_array[1:]

            if len(preds) != len(path_array):
                print(f"Path and predictions length mismatch at path {p_idx}.")
                continue
            
              # actual paths start after the first element

            start_class_color = self.class_colors[start_class]
            target_class_color = self.class_colors[target_class]

            # path_array might have shape (iterations, D), where columns 1 and 2 are x,y coords
            # If path_array includes the start class in column 0, and coords in col 1,2:
            path_coords = np.vstack([pt[0] for pt in path_array])
            path_predictions = self.model_for_predictions.predict(path_coords)
            if self.projection_method == 'umap':
                path_coords_2D = self.umap_reducer.transform(path_coords)
                path_coords_2D_scaled = self.scaler_2D.transform(path_coords_2D).astype('float32') * (self.grid_size - 1)
                path_coords_2D_scaled = np.clip(path_coords_2D_scaled, 0, self.grid_size - 1).astype(int)
            elif self.projection_method == 'ssnp':
                path_coords_2D = self.ssnp_reducer.transform(path_coords)
                path_coords_2D_scaled = self.scaler_2D.transform(path_coords_2D).astype('float32') * (self.grid_size - 1)
                path_coords_2D_scaled = np.clip(path_coords_2D_scaled, 0, self.grid_size - 1).astype(int)
            elif self.projection_method == 'lamp':
                path_coords_2D = self.lamp_reducer.transform(path_coords)#.astype('float32') * (self.grid_size - 1)
                # path_coords_2D_scaled = np.clip(path_coords_2D, 0, self.grid_size - 1).astype(int)
                path_coords_2D_scaled = self.to_grid(path_coords_2D, grid_size=self.grid_size)
            elif self.projection_method == 'PCA':
                path_coords_2D = self.pca_reducer.transform(path_coords)
                path_coords_2D_scaled = self.to_grid(path_coords_2D, grid_size=self.grid_size)

            # path_coords_2D_scaled = self.scaler_2D.transform(path_coords_2D) * (self.grid_size - 1)
            # path_coords_2D_scaled = np.clip(path_coords_2D_scaled, 0, self.grid_size - 1).astype(int)
            
            n = path_coords_2D_scaled.shape[0]

            # Vary transparency along the path: from 0.9 down to 0.3
            alphas = np.linspace(0.9, 0.3, n)

            threshold = (1/self.n_classes) + 0.05

            # Compute colors for each point based on pred_target = preds[i, target_class]
            point_colors = []
            pred_targets = []
            for i in range(n): #iteration i
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

            # Plot the path lines
            for i in range(n-1):
                ax.plot([path_coords_2D_scaled[i, 0], path_coords_2D_scaled[i+1, 0]],
                        [path_coords_2D_scaled[i, 1], path_coords_2D_scaled[i+1, 1]],
                        color=point_colors[i], linewidth=2) #alpha=alphas[i],

            # Plot the path points
            for i in range(n):
                ax.scatter(path_coords_2D_scaled[i, 0], path_coords_2D_scaled[i, 1],
                        color=point_colors[i], s=20) #alpha=alphas[i]

            # Label the starting point
            ax.text(path_coords_2D_scaled[0,0], path_coords_2D_scaled[0,1],
                    f"Start: {start_class}", color='white', fontsize=8,
                    ha='right', va='bottom', alpha=0.8)

        ax.set_title(f"Intermediate Gradient Paths Leading to Class {target_class}", fontsize=25)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")

        # Save figure
        output_path = f"../../../results/images/gradient_plots/class_{target_class}_paths_overlay_{self.version}.png"
        plt.savefig(output_path, dpi=300)
        plt.show()
        plt.close()
        print(f"Class {target_class} paths overlay image saved as '{output_path}'.")

    def save_confidence_image(self, prob_matrix, grid_size, output_path):
        """
        Save an image that shows the confidence of the predictions (prob_matrix)
        using a colormap (viridis) and include the colorbar legend.
        Confidence is high when probabilities are near 0 or 1, and low when near 0.5.
        """

        # Compute the confidence as the distance from 0.5
        if self.n_classes == 2:
            confidence_matrix = np.abs(prob_matrix - 0.5) * 2  # Values between 0 and 1
        else:
            confidence_matrix = np.max(prob_matrix, axis=-1) - np.mean(prob_matrix, axis=-1)
        
        # Normalize the confidence matrix for the color map (range 0 to 1)
        norm = Normalize(vmin=0, vmax=1)
        
        # Use the viridis colormap
        cmap = cm.viridis
        
        # Create a figure and axis to plot the image
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        
        # Display the confidence matrix as an image
        im = ax.imshow(confidence_matrix, cmap=cmap, norm=norm, origin='lower', interpolation='none')
        ax.grid(False)
        
        # Add a colorbar next to the image
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Confidence', rotation=-90, va="bottom", fontsize=18)
        
        # Add a title to the plot
        ax.set_title("Confidence Map", fontsize=25, pad=20)
        
        # Adjust margins and layout
        plt.tight_layout()
        
        # Ensure the output directory exists
        os.makedirs(output_path, exist_ok=True)
        save_path = os.path.join(output_path, f'{self.version}_confidence_plot.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        print(f"Confidence map image saved to {save_path}")
