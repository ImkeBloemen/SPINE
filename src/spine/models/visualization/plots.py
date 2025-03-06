import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
import random
from sklearn.neighbors import KDTree
import matplotlib.cm as cm
import os
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA

# Function to generate a scatterplot matrix

def scatterplot_matrix(data, features=None, hue=None, save_path=None, figsize=(12, 12), random_subset_size=None, seed=None):
    """
    Generates a scatterplot matrix to visualize pairwise relationships between features.
    
    Parameters:
        data (pd.DataFrame): The dataset containing the features.
        features (list): List of feature column names to include in the matrix. If None, use all features.
        hue (str): Column name to use for coloring points by category.
        save_path (str): Path to save the plot as an image. If None, the plot is not saved.
        figsize (tuple): Size of the figure (default is (12, 12)).
        random_subset_size (int): Number of features to randomly select for visualization. If None, use all features.
        seed (int): Random seed for reproducibility of feature selection.
        
    Returns:
        None
    """
    # Use all columns if features are not specified
    if features is None:
        features = data.columns.tolist()
    
    # Randomly select a subset of features if requested
    if random_subset_size is not None:
        if seed is not None:
            random.seed(seed)
        features = random.sample(features, min(random_subset_size, len(features)))
        print(f"Randomly selected features: {features}")

    # Check for valid hue column
    if hue and hue not in data.columns:
        raise ValueError(f"The specified hue column '{hue}' does not exist in the data.")
    if hue and data[hue].isna().any():
        raise ValueError(f"The hue column '{hue}' contains NaN values. Please clean or drop them.")
    
    sns.set(style='ticks', font='Times New Roman')
    pairplot = sns.pairplot(
        data[features],
        hue=hue if hue else None,
        diag_kind='kde',
        corner=True,
        palette='tab20',
        plot_kws={'alpha': 0.7},
    )
    pairplot.fig.set_size_inches(figsize)
    pairplot.fig.subplots_adjust(top=0.92, hspace=0.5, wspace=0.5)
    pairplot.fig.suptitle('Scatterplot Matrix', fontsize=25, fontweight='bold', ha='center')

    # Customize axis labels and title fonts
    for ax in pairplot.axes.flatten():
        if ax is not None:
            ax.title.set_fontsize(25)
            ax.title.set_fontweight('bold')
            ax.xaxis.label.set_fontsize(14)
            ax.yaxis.label.set_fontsize(14)
            ax.tick_params(labelsize=16)

    plt.tight_layout()
    if save_path:
        pairplot.fig.savefig(os.path.join(save_path, 'scatterplot'), dpi=300, bbox_inches='tight')
        print(f"Scatterplot matrix saved to {save_path}")
    plt.show()

def calculate_distances_to_same_class(grid_points: np.ndarray, predictions: np.ndarray, 
                                      training_points: np.ndarray, training_labels: np.ndarray) -> np.ndarray:
    """
    Calculate the distance of each pixel point (grid point) to the nearest training point of the same predicted class.

    Parameters:
        grid_points (np.ndarray): Grid points in high-dimensional space.
        predictions (np.ndarray): Predicted class labels for the grid points.
        training_points (np.ndarray): Training points in high-dimensional space.
        training_labels (np.ndarray): True class labels for the training points.

    Returns:
        np.ndarray: Distances of each grid point to the nearest training point in the same predicted class.
    """
    # Initialize distances with infinity
    distances = np.full(grid_points.shape[0], np.inf)

    # Iterate over each unique class
    unique_classes = np.unique(training_labels)
    for cls in unique_classes:
        # Get training points of the current class
        class_training_points = training_points[training_labels == cls]

        # Build a KDTree for the current class's training points
        kdtree = KDTree(class_training_points)

        # Find grid points predicted as the current class
        class_mask = predictions.flatten() == cls
        class_grid_points = grid_points[class_mask]

        # Compute distances to the nearest training point of the same class
        if class_grid_points.shape[0] > 0:  # Ensure there are grid points of this class
            class_distances, _ = kdtree.query(class_grid_points, k=1)
            distances[class_mask] = class_distances.flatten()  # Assign flattened distances

    return distances

def plot_distance_grid(distance_grid: np.ndarray, version: int, 
                       training_points: np.ndarray, grid_points: np.ndarray, 
                       grid_size: int, path: str = None) -> None:
    """
    Visualize the grid colored by distances to the nearest training point of the same class.

    Parameters:
        distance_grid (np.ndarray): Grid of distances.
        version (int): Version for saving the plot.
        training_points (np.ndarray): Original training points in high-dimensional space.
        grid_points (np.ndarray): Grid points in high-dimensional space.
        grid_size (int): Size of the grid for visualization.
        path (str, optional): Path to save the plot. Defaults to None.

    Returns:
        None
    """
    normalized_training_points = normalize_to_grid(training_points, grid_points, grid_size)

    plt.figure(figsize=(8, 8))
    cmap = cm.viridis
    im = plt.imshow(distance_grid, origin='lower', cmap=cmap)
    cbar = plt.colorbar(im, shrink=0.8, pad=0.05, aspect=10)
    cbar.set_label("Euclidean Distance", fontsize=18)
    cbar.ax.tick_params(labelsize=10)

    plt.title("Pixel Distance to Nearest Training Point of Same Class", fontsize=25, pad=20)

    # Overlay training points as white dots
    plt.scatter(
        normalized_training_points[:, 0],
        normalized_training_points[:, 1],
        c='white',
        s=10,  # Adjust the size of the points
        label='Training Points',
        alpha=0.5,
    )

    plt.axis('off')

    # Save the plot
    if path is not None:
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f'{version}_distance_plot.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        print(f"Distance plot saved as {save_path}")


def plot_gradient_grid(image, version, intermediate_points_2D, grid_points, grid_size, gradient_values, scaler_2D, path):
    """
    Visualize original points and intermediate gradient points.
    - Original points are plotted in black.
    - Intermediate points are colored by HSV, derived from gradient vectors (via PCA).

    Parameters:
        image (np.ndarray or str): Background image or path to it.
        version (int): Version for saving the plot.
        intermediate_points_2D (np.ndarray): Coordinates of intermediate points, shape (n_points, 2).
        grid_points (np.ndarray): The 2D positions corresponding to grid points, shape (m*n, 2).
        grid_size (tuple): The grid dimensions (m, n).
        gradient_values (np.ndarray): High-dimensional gradient values for each intermediate point, shape (n_points, n_features).
        scaler_2D: Scaler for 2D data normalization.
        path (str): Directory path to save the plot.

    Returns:
        None
    """

    def gradient_to_hsv(gX, gY):
        angle = np.arctan2(gY, gX)
        H = (angle + np.pi) / (2 * np.pi)
        magnitude = np.sqrt(gX**2 + gY**2)
        mag_max = magnitude.max() if magnitude.max() != 0 else 1
        S = magnitude / mag_max
        V = np.ones_like(H)
        hsv = np.stack((H, S, V), axis=-1)
        return hsv

    # 1. Perform PCA on gradient_values to reduce to 2D
    pca = PCA(n_components=2)
    T = pca.fit_transform(gradient_values)  # shape: (n_points, 2)
    gX = T[:, 0]
    gY = T[:, 1]

    # 2. Convert PCA results to HSV
    hsv = gradient_to_hsv(gX, gY)

    # 3. Convert HSV to RGB
    rgb = mcolors.hsv_to_rgb(hsv)  # shape: (n_points, 3)

    # Load background image if provided as a path
    if isinstance(image, str):
        img = plt.imread(image)
    else:
        img = image

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    # Plot intermediate gradient points
    intermediate_points_2D[:, 1] = grid_size - 1 - intermediate_points_2D[:, 1]
    scatter = ax.scatter(intermediate_points_2D[:, 0], intermediate_points_2D[:, 1], c=rgb, s=10, alpha=0.7, label='Gradient Points')
    magnitude = np.sqrt(gX**2 + gY**2)

    cbar = plt.colorbar(
        cm.ScalarMappable(norm=plt.Normalize(vmin=magnitude.min(), vmax=magnitude.max()), cmap='hsv'),
        ax=ax,
        orientation='vertical',
        pad=0.02,
        aspect=10,
        shrink=0.8
    )
    cbar.set_label("Gradient Magnitude (relative)", fontsize=18)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title("Gradient Points Overlay", fontsize=25, pad=20)
    ax.set_xlabel("Dimension 1", fontsize=18)
    ax.set_ylabel("Dimension 2", fontsize=18)

    ax.legend(loc='upper right', fontsize=12, title="Legend")

    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    save_path = os.path.join(path, f"gradient_points_overlay_{version}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close()


def normalize_to_grid(points, grid_points, grid_size):
    """
    Normalize high-dimensional points to the grid coordinates for visualization.
    
    Parameters:
        points (np.ndarray): High-dimensional points to normalize.
        grid_points (np.ndarray): Full set of grid points for scaling.
        grid_size (tuple): Grid size for scaling.
    
    Returns:
        np.ndarray: Normalized points for plotting on the grid.
    """
    mins = grid_points.min(axis=0)
    maxs = grid_points.max(axis=0)
    scaled_points = (points - mins) / (maxs - mins)
    normalized_points = (scaled_points * (np.array(grid_size) - 1)).astype(int)
    return normalized_points

