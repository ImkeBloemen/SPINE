# Description: This script evaluates the mapping of different projection techniques on pre-generated
# intermediate points using predictive models and visualization methods.

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))  # Check for available GPUs

import os
import numpy as np
import pandas as pd
from spine.data.load_data import *
import spine.models.visualization.create_map_embedding as create_map_embedding

from sklearn.decomposition import PCA
from scipy.stats import entropy

def measure_intrinsic_dimensionality_pca(X: np.ndarray, threshold: float = 0.95) -> int:
    """
    Measures intrinsic dimensionality using PCA's cumulative variance approach.
    
    Parameters:
    X (np.ndarray): Input dataset.
    threshold (float): Proportion of variance to retain (default is 0.95).
    
    Returns:
    int: Number of components needed to exceed `threshold` proportion of variance.
    """
    pca = PCA().fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    intrinsic_dim = np.argmax(cumulative_variance > threshold) + 1
    return intrinsic_dim

def compute_kl_divergence(X: np.ndarray, X_reference: np.ndarray, bins: int = 30) -> float:
    """
    Compute the KL divergence between two distributions based on their histograms.
    
    Parameters:
    X (np.ndarray): First dataset.
    X_reference (np.ndarray): Reference dataset.
    bins (int): Number of bins for histograms (default is 30).
    
    Returns:
    float: KL divergence value.
    """
    x_vals = X[:, 0]
    x_ref = X_reference[:, 0]

    # Define bin edges based on combined data range
    data_min = min(x_vals.min(), x_ref.min())
    data_max = max(x_vals.max(), x_ref.max())
    bins_edges = np.linspace(data_min, data_max, bins + 1)

    # Compute histograms and avoid zero values
    hist_X, _ = np.histogram(x_vals, bins=bins_edges, density=True)
    hist_ref, _ = np.histogram(x_ref, bins=bins_edges, density=True)
    hist_X = np.maximum(hist_X, 1e-9)
    hist_ref = np.maximum(hist_ref, 1e-9)

    kl_div = entropy(hist_X, hist_ref)
    return kl_div

def run_experiment(model_path, tt_number, projection_method, input_path, dataset_name, model_names, outcome_name, output_path_train_test, test_input_path, grid_size, n_classes):
    """
    Run an experiment to evaluate mapping accuracy of projection methods using intermediate points.

    Parameters:
    model_path (str): Path to the trained model.
    tt_number (str): Test-train split identifier.
    projection_method (str): Projection method (e.g., 'lamp').
    input_path (str): Path to input data.
    dataset_name (str): Name of the dataset.
    model_names (list): List of model names to evaluate.
    outcome_name (str): Name of the outcome variable.
    output_path_train_test (str): Output path for results.
    test_input_path (str): Path to test set data.
    grid_size (int): Size of the grid for mapping.
    n_classes (int): Number of classes in the dataset.
    """
    all_results = []

    for model_name in model_names:
        # Load input data
        intermediate_points = pd.read_csv(os.path.join(input_path, f'_intermediate_gradients_{model_name}.csv'))
        counterfactuals = pd.read_csv(os.path.join(input_path, f'_counterfactuals_{model_name}.csv'))
        dataset = pd.read_csv(os.path.join(input_path, f'_original_{model_name}.csv'))

        # Prepare directories and data
        os.makedirs(output_path_train_test, exist_ok=True)
        X_test = np.load(os.path.join(test_input_path, f'X_test.npy'))
        X_train = np.load(os.path.join(test_input_path, f'X_train.npy'))
        
        # Load predictions and model
        predictions_dataset = dataset[outcome_name]
        intermediate_y_pred = intermediate_points[outcome_name]
        cf_points_binary = counterfactuals[outcome_name]

        _, model_for_predictions = load_model_dice('TF2', model_path, tt_number, classifier_name=model_name)

        # Create and fit the prediction map
        pred_map = create_map_embedding.PredictionMap(
            grid_size=grid_size,
            original_data=dataset,
            intermediate_gradient_points=intermediate_points,
            counterfactuals=counterfactuals,
            number_of_neighbors=3,
            model_for_predictions=model_for_predictions,
            projection_method=projection_method,
            projection_name=model_name,
            intermediate_predictions=np.array(intermediate_y_pred),
            original_predictions=np.array(predictions_dataset),
            counterfactual_predictions=np.array(cf_points_binary),
            outcome_name=outcome_name,
            n_classes=n_classes,
            version=model_name,
            comparison=False,
            dataset_name=dataset_name,
            hidden_layer=False,
            filter_jd=False
        )

        # Generate and evaluate mappings
        pred_map.fit_points_2D(path=output_path_train_test)
        pred_map.fit_grid_knn_weighted_interpolation(path=output_path_train_test)
        pred_map.plot_test_points_on_mapping(X_test, path=output_path_train_test)
        pred_map.plot_data_with_predictions(X_test=X_test, path=output_path_train_test)

        df_results = pred_map.evaluate_mapping_testset(X_test)
        df_results.to_csv(os.path.join(output_path_train_test, 'pixel_results.csv'), index=False)
        accuracy = (df_results["pixel_label"] == df_results["original_label"]).mean()

        # Measure intrinsic dimensionality and KL divergence
        X_high_dim = pd.read_csv(os.path.join(output_path_train_test, f'{model_name}_estimated_high_dim_points.csv')).values
        dim_95_train = measure_intrinsic_dimensionality_pca(X_train)
        dim_95_high_dim = measure_intrinsic_dimensionality_pca(X_high_dim)

        kl_div = compute_kl_divergence(X_train, X_high_dim, bins=30)

        results = {
            "classifier": model_name,
            "accuracy": accuracy,
            "dim_95_train": dim_95_train,
            "dim_95_high_dim": dim_95_high_dim,
            "kl_div": kl_div
        }
        all_results.append(results)
    
    # Save experiment results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_path_train_test, 'results.csv'), index=False)

if __name__ == '__main__':
    tt_numbers = ['1', '2', '3', '4', '5']
    projection_method = 'lamp'
    for tt_number in tt_numbers:
        run_experiment(
            model_path='evaluation/reduced_mnist_data',
            tt_number=tt_number,
            projection_method=projection_method,
            input_path=f'../../../results/experiment_output/{projection_method}/reduced_mnist_data_new_eval/train_test_{tt_number}',
            dataset_name='reduced_mnist_data',
            model_names=['LogisticRegression', 'MLP'],
            outcome_name='label',
            output_path_train_test=f'../../../results/experiment_output/{projection_method}/reduced_mnist_data_filter/train_test_{tt_number}',
            test_input_path=f'../../../data/experiment_input/{projection_method}/reduced_mnist_data/train_test_{tt_number}',
            grid_size=300,
            n_classes=10
        )
