import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

import os
import numpy as np
import pandas as pd
import logging
from typing import List, Dict

import tensorflow as tf

from spine.models.dice_gradients import DiceCounterfactual
from spine.data.load_data import *
import spine.models.visualization.create_map_embedding as create_map_embedding

from sklearn.decomposition import PCA
from scipy.stats import entropy

#Run this experiment to execute the adversarial attack experiment.

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
    outcome_name: str,
    n_classes: int,
    projection_method: str,
    model_path: str,
    dataset_name: str,
    input_path: str,
    output_path: str,
    tt_number: str,
    iteration: int, 
    grid_size: int,
    n_samples: int
) -> Dict:
    """
    Run the experiment for a single classifier.

    Parameters:
    ----------
    data : pd.DataFrame
        Dataset.
    classifier_name : str
        Name of the classifier.
    outcome_name : str
        Name of the outcome variable.
    n_classes : int
        Number of output classes in the dataset.
    projection_method : str
        Name of the projection method.
    model_path : str
        Path to the classifier model.
    dataset_name : str
        Name of the dataset.
    path : str
        Output directory.
    tt_number : str
        Train-test split identifier.
    iteration : int
        Iteration number.
    grid_size : int
        Size of the grid for the mapping.
    n_samples : int
        Number of samples for VAE-driven boundary sampling.

    Returns:
    -------
    Dict
        Results dictionary.
    """
    logging.info(f"Running experiment for classifier: {classifier_name}...")

    train_test_path = f'train_test_{tt_number}'
    os.makedirs(os.path.join(output_path, train_test_path), exist_ok=True)

    # Load train and test data
    X_train_path = os.path.join(input_path, f'train_test_{tt_number}', f"X_train.npy")
    X_test_path = os.path.join(input_path, f'train_test_{tt_number}', f"X_test.npy")
    y_test_path = os.path.join(input_path, f'train_test_{tt_number}', f"y_test.npy")
    X_test_trans_path = os.path.join(input_path, f'train_test_{tt_number}', f"X_test_adv.npy")
    y_test_trans_path = os.path.join(input_path, f'train_test_{tt_number}', f"y_test_adv.npy")
    X_test_subset_path = os.path.join(input_path, f'train_test_{tt_number}', f"X_test_subset.npy")

    try:
        X_train = np.load(X_train_path)
        X_test_total = np.load(X_test_path)
        y_test_total = np.load(y_test_path)
        X_test_trans = np.load(X_test_trans_path)
        y_test_trans = np.load(y_test_trans_path)
        X_test_subset = np.load(X_test_subset_path)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise

    input_path_dice = os.path.join(input_path, train_test_path)

    # Initialize DiceCounterfactual
    dice_cf = DiceCounterfactual(
        dataset_name=dataset_name,
        data_columns=data.columns,
        model_name=model_path,
        outcome_name=outcome_name,
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

    dice_cf.evaluate_autoencoder(save_path=output_path_train_test, version=f'{classifier_name}')

    # Fit DiceCounterfactual
    counterfactuals, dataset, predictions_dataset, intermediate_y_pred, _, cf_points_binary, _, _, intermediate_points, _, _, _ = dice_cf.fit_transform_all_gradient(
        num_samples=n_samples, total_cfs=1, desired_class="opposite", learning_rate=0.01, min_iter=10
    )                                                                
    #Save the results
    
    dice_cf.save_data(path=output_path_train_test, version=f'{classifier_name}')

    _, model_for_predictions = load_model_dice('TF2', f'{model_path}', tt_number, classifier_name=classifier_name, comparison_method_name=None)

    predictions_dataset_output = [np.argmax(x) for x in predictions_dataset]
    intermediate_y_pred_output = [np.argmax(x) for x in intermediate_y_pred]

    pred_map = create_map_embedding.PredictionMap(grid_size=grid_size, 
                                                original_data=dataset, 
                                                intermediate_gradient_points=intermediate_points, 
                                                counterfactuals=counterfactuals,  
                                                number_of_neighbors=3, 
                                                model_for_predictions=model_for_predictions,
                                                projection_method=projection_method,
                                                projection_name=classifier_name,
                                                intermediate_predictions=np.array(intermediate_y_pred_output), 
                                                original_predictions=np.array(predictions_dataset_output), 
                                                counterfactual_predictions=np.array(cf_points_binary), 
                                                outcome_name=outcome_name, 
                                                n_classes=5, 
                                                version=f'{classifier_name}', 
                                                comparison=False,
                                                dataset_name=dataset_name,
                                                hidden_layer = True)
    
    
    pred_map.fit_points_2D(path=output_path_train_test, input_path=input_path_dice)

    pred_map.fit_grid_knn_weighted_interpolation(path=output_path_train_test)

    pred_map.plot_test_points_on_mapping(X_test_subset, path=output_path_train_test, y_test=y_test_total, X_test_trans=X_test_trans, version='adv', y_test_trans=y_test_trans)

    pred_map.plot_test_points_on_mapping(X_test_total, path=output_path_train_test, version='pred_values')

    pred_map.plot_data_with_predictions(X_test=X_test_total, path=output_path_train_test)

    df_results = pred_map.evaluate_mapping_testset(X_test_total)

    pred_map.plot_test_points_on_mapping(X_test_trans, path=output_path_train_test)

    df_results_adv = pred_map.evaluate_mapping_testset(X_test_trans)

    df_results.to_csv(os.path.join(output_path_train_test, 'pixel_results.csv'), index=False)
    accuracy = (df_results["pixel_label"] == df_results["original_label"]).mean()

    df_results_adv.to_csv(os.path.join(output_path_train_test, 'pixel_results_adv.csv'), index=False)
    accuracy_adv = (df_results_adv["pixel_label"] == df_results_adv["original_label"]).mean()

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
    kl_div = compute_kl_divergence(X_reference, X_high_dim, bins=30)
    print(f"KL divergence (1D, feature[0]) between full data and train data: {kl_div:.4f}")

    # Store results
    results = {
        "classifier": classifier_name,
        "accuracy": accuracy,
        "accuracy_adv": accuracy_adv,
        "dim_95_train": dim_95_train,
        "dim_95_high_dim": dim_95_high_dim,
        "kl_div": kl_div
    }

    return results

def run_experiment(
    data_path: str,
    dataset_name: str,
    outcome_name: str,
    projection_method: str,
    classifiers: List[str],
    classifier_paths: List[str],
    input_dir: str,
    output_dir: str,
    tt_number: str,
    n_classes: int,
    grid_size: int,
    n_samples: int
):
    """
    Main experiment runner.

    Parameters:
    ----------
    data_path : str
        Path to the dataset.
    dataset_name : str
        Name of the dataset.
    outcome_name : str
        Name of the outcome variable.
    projection_method : str
        Name of the projection method.
    classifiers : List[str]
        List of classifier names.
    classifier_paths : List[str]
        List of paths to classifier models.
    output_dir : str
        Directory for output files.
    tt_number : str
        Train-test split identifier.
    n_classes : int
        Number of output classes in the dataset.
    grid_size : int
        Size of the grid for the prediction map.
    n_samples : int
        Number of samples for VAE-driven boundary sampling.
    """
    data = setup_experiment(data_path)
    results_list = []

    for classifier, model_path in zip(classifiers, classifier_paths):
        result = run_classifier_experiment(data, classifier, outcome_name, n_classes, projection_method, model_path, dataset_name, input_dir, output_dir, tt_number, grid_size, n_samples)
        results_list.append(result)

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(os.path.join(output_dir, f'train_test_{tt_number}', "experiment_results.csv"), index=False)
    logging.info("Experiment completed. Results saved.")

if __name__ == "__main__":
    # Experiment parameters
    PROJECTION_METHOD = "lamp"
    DATA_PATH = "../../../data/raw/winequality-white.csv"
    DATASET_NAME = "winequality-white"
    INPUT_DIR = "../../../data/experiment_input/winequality-white_adv"
    OUTPUT_DIR = "../../../results/experiment_output/lamp/winequality-white_adv_extreme"
    TT_NUMBERS = ["1"]
    OUTCOME_NAME = "quality"
    N_CLASSES = 5
    GRID_SIZE = 300
    N_SAMPLES = 15

    CLASSIFIERS = ['MLP']
    CLASSIFIER_PATHS = [
        "evaluation/lamp/winequality-white_adv"
    ]

    for TT_NUMBER in TT_NUMBERS:
        print("TT_NUMBER", TT_NUMBER)
        run_experiment(DATA_PATH, DATASET_NAME, OUTCOME_NAME, PROJECTION_METHOD, CLASSIFIERS, CLASSIFIER_PATHS, INPUT_DIR, OUTPUT_DIR, TT_NUMBER, N_CLASSES, GRID_SIZE, N_SAMPLES)