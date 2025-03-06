#Evaluate projection techniques

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

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
    ----------
    X : np.ndarray
        Input dataset.
    threshold : float
        Proportion of variance to retain.

    Returns:
    -------
    int
        Number of components needed to exceed `threshold` proportion of variance.
    """
    pca = PCA().fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    intrinsic_dim = np.argmax(cumulative_variance > threshold) + 1
    return intrinsic_dim


def compute_kl_divergence(X: np.ndarray, X_reference: np.ndarray, bins: int = 30) -> float:
    """
    Compute KL divergence between two distributions.

    Parameters:
    ----------
    X : np.ndarray
        First dataset.
    X_reference : np.ndarray
        Reference dataset.
    bins : int
        Number of bins for the histograms.

    Returns:
    -------
    float
        KL divergence value.
    """
    x_vals = X[:, 0]
    x_ref = X_reference[:, 0]

    data_min = min(x_vals.min(), x_ref.min())
    data_max = max(x_vals.max(), x_ref.max())
    bins_edges = np.linspace(data_min, data_max, bins + 1)

    hist_X, _ = np.histogram(x_vals, bins=bins_edges, density=True)
    hist_ref, _ = np.histogram(x_ref, bins=bins_edges, density=True)

    # Avoid zero values by adding a small constant
    hist_X = np.maximum(hist_X, 1e-9)
    hist_ref = np.maximum(hist_ref, 1e-9)

    kl_div = entropy(hist_X, hist_ref)
    return kl_div

def run_experiment(model_path, tt_number, projection_method, input_path, dataset_name, model_names, outcome_name, output_path_train_test, test_input_path, grid_size, n_classes):

    all_results = []
    for model_name in model_names:
        intermediate_points = pd.read_csv(os.path.join(input_path, f'_intermediate_gradients_{model_name}.csv'))
        counterfactuals = pd.read_csv(os.path.join(input_path, f'_counterfactuals_{model_name}.csv'))
        dataset = pd.read_csv(os.path.join(input_path, f'_original_{model_name}.csv'))

        os.makedirs(output_path_train_test, exist_ok=True)

        train_data_path = os.path.join(test_input_path, f"X_train.npy")
        test_data_path = os.path.join(test_input_path, f"X_test.npy")
        y_test_path = os.path.join(test_input_path, f"y_test.npy")
        X_test_trans_path = os.path.join(test_input_path, f"X_test_adv.npy")
        y_test_trans_path = os.path.join(test_input_path, f"y_test_adv.npy")
        X_test_subset_path = os.path.join(test_input_path, f"X_test_subset.npy")

        X_train = np.load(train_data_path)
        X_test = np.load(test_data_path)
        y_test = np.load(y_test_path)
        X_test_trans = np.load(X_test_trans_path)
        y_test_trans = np.load(y_test_trans_path)
        X_test_subset = np.load(X_test_subset_path)
        
        predictions_dataset = dataset[outcome_name]
        intermediate_y_pred = intermediate_points[outcome_name]
        cf_points_binary = counterfactuals[outcome_name]

        _, model_for_predictions = load_model_dice('TF2', f'{model_path}', tt_number, classifier_name=model_name, comparison_method_name=None)

        predictions_dataset_output = predictions_dataset
        intermediate_y_pred_output = intermediate_y_pred

        pred_map = create_map_embedding.PredictionMap(grid_size=grid_size, 
                                                    original_data=dataset, 
                                                    intermediate_gradient_points=intermediate_points, 
                                                    counterfactuals=counterfactuals,  
                                                    number_of_neighbors=3, 
                                                    model_for_predictions=model_for_predictions,
                                                    projection_method=projection_method,
                                                    projection_name=model_name,
                                                    intermediate_predictions=np.array(intermediate_y_pred_output), 
                                                    original_predictions=np.array(predictions_dataset_output), 
                                                    counterfactual_predictions=np.array(cf_points_binary), 
                                                    outcome_name=outcome_name, 
                                                    n_classes=n_classes, 
                                                    version=f'{model_name}', 
                                                    comparison=False,
                                                    dataset_name=dataset_name,
                                                    hidden_layer=True)
        
        
        pred_map.fit_points_2D(path=output_path_train_test)

        pred_map.fit_grid_knn_weighted_interpolation(path=output_path_train_test)

        pred_map.plot_test_points_on_mapping(X_test, path=output_path_train_test, y_test=y_test, X_test_trans=X_test_trans, y_test_trans=y_test_trans, version='adv', X_test_subset=X_test_subset, adversarial=True)

        pred_map.plot_test_points_on_mapping(X_test, path=output_path_train_test, version='pred_values')

        pred_map.plot_data_with_predictions(X_test=X_test, path=output_path_train_test)

        df_results = pred_map.evaluate_mapping_testset(X_test)

        pred_map.plot_test_points_on_mapping(X_test_trans, path=output_path_train_test)

        df_results_adv = pred_map.evaluate_mapping_testset(X_test_trans)

        df_results.to_csv(os.path.join(output_path_train_test, 'pixel_results.csv'), index=False)
        accuracy = (df_results["pixel_label"] == df_results["original_label"]).mean()

        df_results_adv.to_csv(os.path.join(output_path_train_test, 'pixel_results_adv.csv'), index=False)
        accuracy_adv = (df_results_adv["pixel_label"] == df_results_adv["original_label"]).mean()

        X_high_dim_path = os.path.join(output_path_train_test, f'{model_name}_estimated_high_dim_points.csv')
        X_high_dim = pd.read_csv(X_high_dim_path)
        if isinstance(X_train, pd.DataFrame):
            X_reference = X_train.values
        else:
            X_reference = X_train
        X_high_dim = X_high_dim.values

        # Intrinsic Dimensionality
        dim_95_train = measure_intrinsic_dimensionality_pca(X_reference)
        dim_95_high_dim = measure_intrinsic_dimensionality_pca(X_high_dim)
        print(f"Intrinsic dimensionality (PCA >=95% var): {dim_95_train} (train), {dim_95_high_dim} (high-dim)")

        # KL Divergence
        kl_div = compute_kl_divergence(X_reference, X_high_dim, bins=30)
        print(f"KL divergence (1D, feature[0]) between full data and train data: {kl_div:.4f}")

        # Store results
        results = {
            "classifier": model_name,
            "accuracy": accuracy,
            "accuracy_adv": accuracy_adv,
            "dim_95_train": dim_95_train,
            "dim_95_high_dim": dim_95_high_dim,
            "kl_div": kl_div
        }

        all_results.append(results)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_path_train_test, 'results.csv'), index=False)

if __name__ == '__main__':
    tt_numbers = ['1'] #, '2', '3', '4', '5']
    projection_method = 'lamp'
    for tt_number in tt_numbers:
        run_experiment(model_path='evaluation/winequality-white_adv',
                    tt_number=tt_number,
                    projection_method=projection_method,
                    input_path=f'../../../results/experiment_output/{projection_method}/winequality-white_adv/train_test_{tt_number}', 
                    dataset_name='winequality-white_adv', 
                    model_names= ['MLP'],
                    outcome_name='label', 
                    output_path_train_test=f'../../../results/experiment_output/{projection_method}/winequality-white_adv/train_test_{tt_number}', 
                    test_input_path=f'../../../data/experiment_input/[{projection_method}]/winequality-white_adv/train_test_{tt_number}',
                    grid_size=300,
                    n_classes=5)