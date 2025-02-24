# experimentation.py

import os
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import List, Tuple, Dict, Callable
import seaborn as sns
import requests
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from itertools import permutations, combinations
from joblib import load
import importlib

#from numba import jit, cuda

import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml

from sklearn.pipeline import Pipeline

import VAE_DBS
from VAE_DBS.models.dice_gradients import DiceCounterfactual
from VAE_DBS.utils.utils import *
from VAE_DBS.data.load_data import *
from VAE_DBS.models.DiCE.dice_ml.utils.helpers import DataTransfomer
import random
from VAE_DBS.visualization.create_map_embedding import PredictionMap
import VAE_DBS.visualization.create_map.create_map_UMAP_test_no_intermediate as create_map_UMAP_test_no_intermediate
import VAE_DBS.models.transformers.ssnp

from sklearn.metrics.pairwise import pairwise_kernels, pairwise_distances
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy
from skdim import id
from sklearn.decomposition import PCA

# Ensure output directories exist
os.makedirs('../../../results/experiments/hyperparameters/metrics', exist_ok=True)
os.makedirs('../../../results/experiments/hyperparameters/images', exist_ok=True)

# Helper Functions

def calculate_intrinsic_dimensionality(data_array, method_name='estimated_high_dim_points', save_path='../../../results/experiments/hyperparameters/metrics'):
    X = data_array

    # Method 1: PCA with cumulative variance
    pca = PCA().fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    intrinsic_dimensionality_pca = np.argmax(cumulative_variance > 0.95) + 1
    print(f"Intrinsic dimensionality (cumulative variance) for {method_name}: {intrinsic_dimensionality_pca}")

    # Save cumulative variance plot
    plt.figure()
    plt.plot(np.arange(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'Cumulative Variance Explained by PCA for {method_name}')
    plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(save_path, f'cumulative_variance_{method_name}.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"Cumulative variance plot saved to {plot_filename}")

    # Method 2: skdim's lPCA
    lpca = id.lPCA()
    intrinsic_dimensionality_lpca = lpca.fit(X).dimension_
    print(f"Intrinsic dimensionality (lPCA) for {method_name}: {intrinsic_dimensionality_lpca}")

    # Save results to a CSV file
    results = pd.DataFrame({
        'Method': ['PCA (95% variance)', 'skdim lPCA'],
        'Intrinsic Dimensionality': [intrinsic_dimensionality_pca, intrinsic_dimensionality_lpca]
    })
    results_filename = os.path.join(save_path, f'intrinsic_dimensionality_{method_name}.csv')
    results.to_csv(results_filename, index=False)
    print(f"Intrinsic dimensionality results saved to {results_filename}")

    return intrinsic_dimensionality_pca, intrinsic_dimensionality_lpca

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

def calculate_mmd(X1, X2, kernel='rbf', gamma=1.0):
    XX = pairwise_kernels(X1, X1, metric=kernel, gamma=gamma)
    YY = pairwise_kernels(X2, X2, metric=kernel, gamma=gamma)
    XY = pairwise_kernels(X1, X2, metric=kernel, gamma=gamma)

    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd

def calculate_wasserstein_distance(X1, X2):
    distance = np.mean(pairwise_distances(X1, X2))
    return distance

def count_boundary_predictions(prediction_probs, stopping_threshold=0.5, epsilon=0.05):
    lower_bound = stopping_threshold - epsilon
    upper_bound = stopping_threshold + epsilon
    within_range = (prediction_probs >= lower_bound) & (prediction_probs <= upper_bound)
    count = np.sum(within_range)
    percentage = (count / len(prediction_probs)) * 100
    return count, percentage

# Experimentation Loop

def run_experimentation():
    # Hyperparameter grids
    proto_weights = [0.5]
    ae_weights = [0.5, 0.7, 0.9]
    prediction_weights = [0.5, 1, 1.5]
    number_of_neighbors_values = [3, 5, 7, 10]

    # Load dataset and model
    # Assume you have functions or code to load your dataset and model
    dataset_name = 'reduced_mnist_data'
    data = pd.read_csv('../../../data/raw/reduced_mnist_data.csv')
    model_name = 'evaluation/mnist'
    classifier_name = 'mlp_model'
    outcome_name = 'label'
    num_samples = 15
    total_cfs = 1
    dice_cf = DiceCounterfactual(
        data_columns=data.columns,
        dataset_name=dataset_name,
        model_name=model_name,
        classifier_name=classifier_name,
        outcome_name=outcome_name,
        preprocessing=None,
        backend='TF2',
        model_format='h5',
        multi_class=True,
        comparison=True,
        comparison_method='sdbm',
        tt_number='1'
    )

    # Prepare a DataFrame to store results
    results_df = pd.DataFrame(columns=[
        'Proto Weight', 'AE Weight', 'Prediction Weight', 'Num Neighbors',
        # 'Intrinsic Dim PCA', 'Intrinsic Dim lPCA',
        'KL Divergence',
        # 'Boundary Count', 'Boundary Percentage'
    ])

    # Iterate over hyperparameters
    for proto_weight in proto_weights:
        for ae_weight in ae_weights:
            for prediction_weight in prediction_weights:
                for number_of_neighbors in number_of_neighbors_values:
                    print(f"Running experiment with proto_weight={proto_weight}, ae_weight={ae_weight}, prediction_weight={prediction_weight}, number_of_neighbors={number_of_neighbors}")
                    # Run your method
                    counterfactuals, dataset, predictions_dataset, intermediate_y_pred, _, cf_points_binary, _, _, intermediate_points, _, _, _ = dice_cf.fit_transform_all_gradient(
                        num_samples=num_samples,
                        total_cfs=total_cfs,
                        desired_class='opposite',
                        prediction_weight=prediction_weight,
                        ae_weight=ae_weight,
                        proto_weight=proto_weight,
                        samples=None,
                        learning_rate=0.01
                    )

                    predictions_dataset_output = [np.argmax(x) for x in predictions_dataset]
                    intermediate_y_pred_output = [np.argmax(x) for x in intermediate_y_pred]

                    # Initialize your PredictionMap
                    pred_map = PredictionMap(
                        grid_size=300,
                        original_data=dataset,
                        intermediate_gradient_points=intermediate_points,
                        counterfactuals=counterfactuals,
                        number_of_neighbors=number_of_neighbors,
                        model_for_predictions=dice_cf.model.model,
                        scaler_path_2D='../models/transformers/scalers/minmax_scaler_mnist.save',
                        projection_method='ssnp',
                        projection_name='mnist_ssnp',
                        intermediate_predictions=np.array(intermediate_y_pred_output),
                        original_predictions=np.array(predictions_dataset_output),
                        counterfactual_predictions=np.array(cf_points_binary),
                        outcome_name='label',
                        n_classes=10,
                        version=f'2.0_pw{proto_weight}_aw{ae_weight}_pw{prediction_weight}_nn{number_of_neighbors}'
                    )
                    # Run the mapping
                    pred_map.fit_points_2D()
                    pred_map.fit_grid_multilateration(path='../../../results/experiments/hyperparameters/points')

                    # Perform measurements
                    estimated_hd_points_all = np.vstack(pred_map.estimated_hd_points_all)
                    X_original = dataset.values
                    X_intermediate = intermediate_points
                    X_estimated = estimated_hd_points_all

                    # Calculate intrinsic dimensionality
                    # id_pca, id_lpca = calculate_intrinsic_dimensionality(
                    #     X_estimated,
                    #     method_name=f'estimated_hd_points_pw{proto_weight}_aw{ae_weight}_pw{prediction_weight}_nn{number_of_neighbors}'
                    # )

                    # Calculate distribution metrics
                    kl_div_original_estimated = compute_kl_divergence(X_original, X_estimated)
                    # mmd_original_estimated = calculate_mmd(X_original, X_estimated)
                    # wasserstein_original_estimated = calculate_wasserstein_distance(X_original, X_estimated)
                    # print(f"KL Divergence between Original and Estimated: {kl_div_original_estimated}")
                    # print(f"MMD between Original and Estimated: {mmd_original_estimated}")
                    # print(f"Wasserstein Distance between Original and Estimated: {wasserstein_original_estimated}")

                    # Count boundary predictions
                    # prediction_prob = pred_map.grid_predictions_prob.reshape(-1, pred_map.n_classes)
                    # print("prediction_prob", prediction_prob)
                    # if pred_map.n_classes == 2:
                    #     # For binary classification
                    #     count, percentage = count_boundary_predictions(prediction_prob.flatten())
                    # else:
                    #     # For multi-class classification
                    #     max_probs = np.max(prediction_prob, axis=1)
                    #     print("max_probs", max_probs)
                    #     count, percentage = count_boundary_predictions(max_probs)
                    # print(f"Number of points near the boundary: {count}")
                    # print(f"Percentage of points near the boundary: {percentage:.2f}%")

                    # Save metrics to DataFrame
                    new_row = {
                        'Proto Weight': proto_weight,
                        'AE Weight': ae_weight,
                        'Prediction Weight': prediction_weight,
                        'Num Neighbors': number_of_neighbors,
                        'KL Divergence': kl_div_original_estimated,
                    }

                    results_df.loc[len(results_df)] = new_row

                    # Save intermediate results
                    results_df.to_csv('../../../results/experiments/hyperparameters/metrics/KL_experimentation_results_test.csv', index=False)
                    print("Results saved to '../../../results/experiments/hyperparameters/metrics/KL_experimentation_results_test.csv'")

    # Final save
    results_df.to_csv('../../../results/experiments/hyperparameters/metrics/KL_experimentation_results_test.csv', index=False)
    print("Final results saved to '../../../results/experiments/hyperparameters/metrics/KL_experimentation_results_test.csv'")

if __name__ == '__main__':
    run_experimentation()
