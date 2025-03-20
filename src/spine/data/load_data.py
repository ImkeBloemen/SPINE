import spine.models.DiCE.dice_ml as dice_ml
from spine.models.DiCE.dice_ml.utils import helpers
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model as tf_load_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import joblib
import h5py
import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras.mixed_precision import Policy

def load_blobs(blob_name):
    """Load the blobs from the specified file."""
    blobs_X = np.load(f'data/{blob_name}/X.npy')
    blobs_y = np.load(f'data/{blob_name}/y.npy')
    blobs_df = pd.DataFrame(blobs_X, columns=[f'feature_{i}' for i in range(blobs_X.shape[1])])
    blobs_df['target'] = blobs_y
    return blobs_df, blobs_y

def load_dataset(outcome_name, dataset_name):
        """Load and return the dataset."""
        dataset = pd.read_csv(f'../../../data/raw/{dataset_name}.csv')
        target = dataset[outcome_name]

        return dataset, target

def split_dataset(dataset, target, test_size=0.2, random_state=0):
    """Split dataset into training and testing sets."""
    train_data, test_data, y_train, y_test = train_test_split(
        dataset,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target
    )
    
    return train_data, test_data, y_train, y_test

def load_data_sdbm(data_path, tt_number):
    X = np.load(f'../../../data/experiment_input/sdbm/{data_path}/train_test_{tt_number}/X_train.npy')
    y = np.load(f'../../../data/experiment_input/sdbm/{data_path}/train_test_{tt_number}/y_train.npy')

    return X, y

def load_data_deepview(data_path, tt_number):
    X = np.load(f'../../../data/experiment_input/deepview/{data_path}/train_test_{tt_number}/X_sub_{data_path}.npy')
    y = np.load(f'../../../data/experiment_input/deepview/{data_path}/train_test_{tt_number}/y_sub_{data_path}.npy')

    return X, y

def load_data_train_test(data_path, tt_number):
    X = np.load(f'../../../data/experiment_input/{data_path}/train_test_{tt_number}/X_train.npy')
    y = np.load(f'../../../data/experiment_input/{data_path}/train_test_{tt_number}/y_train.npy')

    return X, y

def load_model_dice(backend, model_name, tt_number='', classifier_name='', model_format='h5', func='ohe-min-max', multi_class=False, comparison_method_name=None):
    """
    Load the pre-trained model for use with DiceML, ensuring compatibility with TensorFlow optimizers.
    
    Parameters:
    - backend: The model backend, either 'TF' (TensorFlow) or 'PYT' (PyTorch).
    - model_name: The name of the model file to load (without extension).
    - model_format: The format of the model file, either 'h5' for TensorFlow/Keras models or 'joblib' for scikit-learn models.
    - func: Preprocessing function used in the model.
    
    Returns:
    - model: The DiceML model instance.
    - model_for_predictions: The loaded model instance for making predictions.
    """

    # Define the function to replace 'batch_shape' with 'batch_input_shape'
    def load_model_replace_batch_shape(model_path):
        with h5py.File(model_path, 'r') as f:
             # Load the model configuration
            model_config = f.attrs.get('model_config')
            if model_config is None:
                raise ValueError('No model found in config.')
            
            # Check if model_config is bytes and decode if necessary
            if isinstance(model_config, bytes):
                model_config = model_config.decode('utf-8')
            # Replace 'batch_shape' with 'batch_input_shape' in the configuration
            model_config = model_config.replace('"batch_shape"', '"batch_input_shape"')
            # Define custom_objects
            custom_objects = {'DTypePolicy': Policy}
            # Reconstruct the model from the updated configuration
            model = model_from_json(model_config, custom_objects=custom_objects)
            # Load the weights into the model
            model.load_weights(model_path)
            return model

    # Set the model path based on the model name and format
    if comparison_method_name == 'sdbm':
        #For SDBM
        model_path = f'../models/input_classifiers/{model_name}/{classifier_name}_{tt_number}.{model_format}'
    #For Deepview
    elif comparison_method_name == 'deepview':
        model_path = f'../models/input_classifiers/{model_name}_{tt_number}.{model_format}'
    else:
        model_path = f'../models/input_classifiers/{model_name}/{classifier_name}_{tt_number}.{model_format}'
    print(os.path.exists(model_path))

    if 'TF' in backend and model_format == 'h5':
        # Use the custom function to load the model and replace 'batch_shape'
        model_for_predictions = load_model_replace_batch_shape(model_path)
        # Recompile with a compatible optimizer
        if multi_class:
            model_for_predictions.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            model_for_predictions.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        # Initialize the DiceML model with the recompiled model
        model = dice_ml.Model(model=model_for_predictions, backend=backend, func=func)

    elif model_format == 'joblib':
        # Load the scikit-learn model from a .joblib file
        model_for_predictions = joblib.load(model_path)
        
        # Initialize the DiceML model with the loaded scikit-learn model
        model = dice_ml.Model(model=model_for_predictions, backend="sklearn", func="ohe-min-max")

    elif backend == 'PYT':
        # Implement PyTorch model loading if necessary
        pass
    else:
        raise ValueError("Unsupported backend or model format specified. Use 'TF' with 'h5' or 'joblib'.")
    
    return model, model_for_predictions