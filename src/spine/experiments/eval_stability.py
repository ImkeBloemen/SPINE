import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

import os
import numpy as np
import pandas as pd
import logging
from copy import deepcopy
from typing import List, Tuple, Dict, Callable
import seaborn as sns
import requests
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.distance import euclidean
from itertools import permutations, combinations
from joblib import load
import importlib

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
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

import VAE_DBS

# from VAE_DBS.models import dice_gradients
from VAE_DBS.models.dice_gradients import DiceCounterfactual
from VAE_DBS.utils.utils import *
from VAE_DBS.data.load_data import *
from VAE_DBS.models.DiCE.dice_ml.utils.helpers import DataTransfomer
import random
import VAE_DBS.visualization.create_map_embedding as create_map_embedding
import VAE_DBS.visualization.create_map.create_map_UMAP_test_no_intermediate as create_map_UMAP_test_no_intermediate
# import ssnp_main.code.ssnp as ssnp
import VAE_DBS.models.transformers.ssnp

import skdim
from sklearn.decomposition import PCA
from scipy.stats import entropy

#Run this experiment for every train/test combination. This file runs the experiment for the same dataset,
#on one train/test split, and three different classifiers: Logistic Regression, SVM, and MLP. For one comparison method.

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def setup_experiment(data_path: str) -> pd.DataFrame:
    """
    Load and preprocess the dataset for the experiment.

    Parameters:
    ----------
    data_path : str
        Path to the dataset CSV file.

    Returns:
    -------
    pd.DataFrame
        Loaded dataset.
    """
    try:
        logging.info(f"Loading dataset from {data_path}...")
        return pd.read_csv(data_path)
    except FileNotFoundError:
        logging.error(f"Dataset not found at path: {data_path}")
        raise


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



def run_classifier_experiment(
    data: pd.DataFrame,
    classifier_name: str,
    model_path: str,
    dataset_name: str,
    input_path: str,
    output_path: str,
    tt_number: str,
    iteration: int
) -> Dict:
    """
    Run the experiment for a single classifier.

    Parameters:
    ----------
    data : pd.DataFrame
        Dataset.
    classifier_name : str
        Name of the classifier.
    model_path : str
        Path to the classifier model.
    dataset_name : str
        Name of the dataset.
    path : str
        Output directory.
    tt_number : str
        Train-test split identifier.

    Returns:
    -------
    Dict
        Results dictionary.
    """
    logging.info(f"Running experiment for classifier: {classifier_name}...")

    train_test_path = f'train_test_{tt_number}/iteration_{iteration}'
    os.makedirs(os.path.join(output_path, train_test_path), exist_ok=True)

    # Load train and test data
    train_data_path = os.path.join(input_path, f'train_test_{tt_number}', f"X_train.npy")
    test_data_path = os.path.join(input_path, f'train_test_{tt_number}', f"X_test.npy")

    try:
        X_train = np.load(train_data_path)
        X_test = np.load(test_data_path)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise

    input_path_dice = os.path.join(input_path, train_test_path)

    # Initialize DiceCounterfactual
    dice_cf = DiceCounterfactual(
        dataset_name=dataset_name,
        data_columns=data.columns,
        model_name=model_path,
        outcome_name="label",
        preprocessing=None,
        backend="TF2",
        model_format="h5",
        multi_class=True,
        comparison=False,
        comparison_method="train_test",
        input_path=input_path_dice,
        tt_number = tt_number,
        classifier_name = classifier_name,
        comparison_data_name=dataset_name
    )

    output_path_train_test = os.path.join(output_path, train_test_path)

    # dice_cf.evaluate_autoencoder(save_path=output_path_train_test, version=f'{classifier_name}')

    # Fit DiceCounterfactual
    counterfactuals, dataset, predictions_dataset, intermediate_y_pred, _, cf_points_binary, _, _, intermediate_points, _, _, _, _, _, _, _ = dice_cf.fit_transform_all_gradient(
        num_samples=11, total_cfs=1, desired_class="opposite", learning_rate=0.01, min_iter=10
    )                                                                
    #Save the results
    
    dice_cf.save_data(path=output_path_train_test, version=f'{classifier_name}')

    _, model_for_predictions = load_model_dice('TF2', f'{model_path}', tt_number, classifier_name=classifier_name, comparison_method_name=None)

    predictions_dataset_output = [np.argmax(x) for x in predictions_dataset]
    intermediate_y_pred_output = [np.argmax(x) for x in intermediate_y_pred]


    pred_map = create_map_embedding.PredictionMap(grid_size=300, 
                                                original_data=dataset, 
                                                intermediate_gradient_points=intermediate_points, 
                                                counterfactuals=counterfactuals,  
                                                number_of_neighbors=3, 
                                                model_for_predictions=model_for_predictions,
                                                # scaler_path_2D = os.path.join(input_path_dice, f'minmax_scaler_2D_{classifier_name}.save'),
                                                projection_method='ssnp',
                                                projection_name=classifier_name,
                                                intermediate_predictions=np.array(intermediate_y_pred_output), 
                                                original_predictions=np.array(predictions_dataset_output), 
                                                counterfactual_predictions=np.array(cf_points_binary), 
                                                outcome_name='label', 
                                                n_classes=10, 
                                                version=f'{classifier_name}', 
                                                comparison=False,
                                                dataset_name=dataset_name,
                                                hidden_layer=False)
    
    
    pred_map.fit_points_2D(path=output_path_train_test, input_path=input_path_dice)

    pred_map.fit_grid_multilateration(path=output_path_train_test)

    pred_map.plot_test_points_on_mapping(X_test, path=output_path_train_test)

    pred_map.plot_data_with_predictions(X_test=X_test, path=output_path_train_test)

    df_results = pred_map.evaluate_mapping_testset(X_test)
    df_results.to_csv(os.path.join(output_path_train_test, 'pixel_results.csv'), index=False)
    accuracy = (df_results["pixel_label"] == df_results["original_label"]).mean()

    X_high_dim_path = os.path.join(output_path_train_test, f'{classifier_name}_estimated_high_dim_points.csv')
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
    # For demonstration, let's do a naive approach where X_reference = X_train:
    # In a real scenario, you might have a "original dataset" vs "synthetic" or so.
    kl_div = compute_kl_divergence(X_reference, X_high_dim, bins=30)
    print(f"KL divergence (1D, feature[0]) between full data and train data: {kl_div:.4f}")

    # Store results
    results = {
        "classifier": classifier_name,
        "accuracy": accuracy,
        "dim_95_train": dim_95_train,
        "dim_95_high_dim": dim_95_high_dim,
        "kl_div": kl_div
    }

    return results
    # # Append results to a DataFrame
    # if 'results_df' not in locals():
    #     results_df = pd.DataFrame(results, index=[0])
    # else:
    #     results_df = results_df.append(results, ignore_index=True)

    # # Save the DataFrame to a CSV file
    # results_df.to_csv(f"{path}/experiment_results.csv", index=False)

def run_experiment(
    data_path: str,
    dataset_name: str,
    classifiers: List[str],
    classifier_paths: List[str],
    input_dir: str,
    output_dir: str,
    tt_number: str,
    iteration: int
):
    """
    Main experiment runner.

    Parameters:
    ----------
    data_path : str
        Path to the dataset.
    dataset_name : str
        Name of the dataset.
    classifiers : List[str]
        List of classifier names.
    classifier_paths : List[str]
        List of paths to classifier models.
    output_dir : str
        Directory for output files.
    tt_number : str
        Train-test split identifier.
    """
    data = setup_experiment(data_path)
    results_list = []

    for classifier, model_path in zip(classifiers, classifier_paths):
        result = run_classifier_experiment(data, classifier, model_path, dataset_name, input_dir, output_dir, tt_number, iteration)
        results_list.append(result)

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(os.path.join(output_dir, f'train_test_{tt_number}', "experiment_results.csv"), index=False)
    logging.info("Experiment completed. Results saved.")

if __name__ == "__main__":
    # Experiment parameters
    DATA_PATH = "C:/Users/imke.bloemen/OneDrive - Accenture/1. Graduation/Code/NewCode/VAE_DBS/data/raw/reduced_mnist_data.csv"
    DATASET_NAME = "reduced_mnist_data"
    INPUT_DIR = "C:/Users/imke.bloemen/OneDrive - Accenture/1. Graduation/Code/NewCode/VAE_DBS/experiment_input/lamp/reduced_mnist_data"
    OUTPUT_DIR = "C:/Users/imke.bloemen/OneDrive - Accenture/1. Graduation/Code/NewCode/VAE_DBS/experiment_output/ssnp/reduced_mnist_data_repeat"
    TT_NUMBERS = ["1"] #, "2", "3", "4", "5"]
    # num_samples_list = [50]

    #Do not forget to check load_data.py!

    CLASSIFIERS = ['MLP']      # #"underfit_model", , "overfit_model"
    CLASSIFIER_PATHS = [
        "evaluation/reduced_mnist_data",
        # "evaluation/reduced_mnist_data"
        # "evaluation/mnist_filtered_sep"
        # "evaluation/mnist_sdbm"
        # "evaluation/mnist_filtered"
    ]

    # for num_samples in num_samples_list:
    for TT_NUMBER in TT_NUMBERS:
        for iteration in range(5):
            run_experiment(DATA_PATH, DATASET_NAME, CLASSIFIERS, CLASSIFIER_PATHS, INPUT_DIR, OUTPUT_DIR, TT_NUMBER, iteration)