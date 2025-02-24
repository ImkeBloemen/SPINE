import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

def save_dataset(name, X, y, base_dir):
    """
    Save the train and test datasets after processing.
    """
    print(name, X.shape)

    lenc = LabelEncoder()
    y = lenc.fit_transform(y)

    for l in np.unique(y):
        print('Train -->', l, np.count_nonzero(y == l))

    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    # Save train and test datasets
    np.save(os.path.join(base_dir, 'X_enriched.npy'), X)
    np.save(os.path.join(base_dir, 'y_enriched.npy'), y)

def process_data(name, outcome_name, tt_number, rs):
    """
    Load and preprocess the dataset for all classes, creating train and test sets.
    """
    data = pd.read_csv(f'input_data/breast_cancer/enriched/_intermediate_gradients_0.csv')
    y = data[outcome_name]
    X = data.drop(columns=[outcome_name])

    base_dir = 'input_data/breast_cancer/train_test_1/enriched/'

    # # Save the scaler
    # scaler_filename = os.path.join(base_dir, 'scaler.pkl')
    # with open(scaler_filename, 'wb') as scaler_file:
    #     import pickle
    #     pickle.dump(scaler, scaler_file)

    # Save the train and test datasets
    save_dataset(name, X, y, base_dir)

if __name__ == '__main__':
    random_states = [42]  # Add your random states here  9514, 8331, 5816, 3807
    tt_numbers = ['1']  # Add your tt_number identifiers here , '2', '3', '4', '5'
    # Example usage with tt_number
    for tt_number, rs in zip(tt_numbers, random_states):  # Adjust the range for the desired number of train-test splits
        process_data(name='breast_cancer', outcome_name='label', tt_number=tt_number, rs=rs)
