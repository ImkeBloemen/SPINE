import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

def save_dataset(name, X_train, y_train, X_test, y_test, base_dir):
    """
    Save the train and test datasets after processing.
    """
    print(name, X_train.shape, X_test.shape)

    lenc = LabelEncoder()
    y_train = lenc.fit_transform(y_train)
    y_test = lenc.transform(y_test)

    for l in np.unique(y_train):
        print('Train -->', l, np.count_nonzero(y_train == l))
    for l in np.unique(y_test):
        print('Test -->', l, np.count_nonzero(y_test == l))

    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    # Save train and test datasets
    np.save(os.path.join(base_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(base_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(base_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(base_dir, 'y_test.npy'), y_test)

def process_data(name, outcome_name, tt_number, rs):
    """
    Load and preprocess the dataset for all classes, creating train and test sets.
    """
    org_data = pd.read_csv(f'input_data/new_data/enriched/_original_MLP_{tt_number}.csv')
    intermediate_data = pd.read_csv(f'input_data/new_data/enriched/_intermediate_gradients_MLP_{tt_number}.csv')

    data = pd.concat([org_data, intermediate_data], axis=0)
    y = data[outcome_name].values
    X = data.drop(columns=[outcome_name]).values

    base_dir = f'input_data/{name}/train_test_{tt_number}'
    os.makedirs(base_dir, exist_ok=True)

    # Remove low-variance features
    # selector = VarianceThreshold(threshold=0.0)
    # selector.fit(X)
    # features_selected = X[X.columns[selector.get_support(indices=True)]]
    # X = np.array(features_selected)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs, stratify=y)

    #For CIFAR-10, the data is already scaled.
    # scaler = MinMaxScaler()
    # # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train.astype('float32'))
    # X_test = scaler.transform(X_test.astype('float32'))

    # # Save the scaler
    # scaler_filename = os.path.join(base_dir, 'scaler.pkl')
    # with open(scaler_filename, 'wb') as scaler_file:
    #     import pickle
    #     pickle.dump(scaler, scaler_file)

    # Save the train and test datasets
    save_dataset(name, X_train, y_train, X_test, y_test, base_dir)

if __name__ == '__main__':
    random_states = [9514, 8331, 5816, 3807]  # Add your random states here 42, 9514, 8331, 5816, 3807
    tt_numbers = ['2', '3', '4', '5']  # Add your tt_number identifiers here '1', '2', '3', '4', '5'
    # Example usage with tt_number
    for tt_number, rs in zip(tt_numbers, random_states):  # Adjust the range for the desired number of train-test splits
        process_data(name='reduced_mnist_data_enriched', outcome_name='label', tt_number=tt_number, rs=rs)
