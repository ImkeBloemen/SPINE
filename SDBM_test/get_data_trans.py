import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

def save_dataset(name, X_train, y_train, X_test, y_test, X_test_trans, y_test_trans, X_test_control, y_test_control, base_dir):
    """
    Save the train and test datasets after processing.
    """
    print(name, X_train.shape, X_test.shape)

    lenc = LabelEncoder()
    y_train = lenc.fit_transform(y_train)
    y_test = lenc.transform(y_test)
    y_test_trans = lenc.transform(y_test_trans)
    y_test_control = lenc.transform(y_test_control)

    for l in np.unique(y_train):
        print('Train -->', l, np.count_nonzero(y_train == l))
    for l in np.unique(y_test):
        print('Test -->', l, np.count_nonzero(y_test == l))
    for l in np.unique(y_test_trans):
        print('Test -->', l, np.count_nonzero(y_test_trans == l))
    for l in np.unique(y_test_control):
        print('Test -->', l, np.count_nonzero(y_test_control == l))

    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    # Save train and test datasets
    np.save(os.path.join(base_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(base_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(base_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(base_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(base_dir, 'X_test_trans.npy'), X_test_trans)
    np.save(os.path.join(base_dir, 'y_test_trans.npy'), y_test_trans)
    np.save(os.path.join(base_dir, 'X_test_control.npy'), X_test_control)
    np.save(os.path.join(base_dir, 'y_test_control.npy'), y_test_control)

def process_data(name, outcome_name, tt_number, rs):
    """
    Load and preprocess the dataset for all classes, creating train and test sets.
    """
    X_train = pd.read_csv(f'../data/reduced_mnist_data_train.csv')
    X_test = pd.read_csv(f'../data/reduced_mnist_data_test.csv') #complete test set with transformation
    X_test_trans = pd.read_csv(f'../data/reduced_mnist_data_test_trans.csv') #transformed test subset
    X_test_control = pd.read_csv(f'../data/reduced_mnist_data_test_control.csv') #original subset before transformation
    y_train = X_train[outcome_name]
    y_test = X_test[outcome_name]
    y_test_trans = X_test_trans[outcome_name]
    X_train = X_train.drop(columns=[outcome_name])
    X_test = X_test.drop(columns=[outcome_name])
    X_test_trans = X_test_trans.drop(columns=[outcome_name])
    y_test_control = X_test_control[outcome_name]
    X_test_control = X_test_control.drop(columns=[outcome_name])

    base_dir = f'input_data/{name}/train_test_{tt_number}'
    os.makedirs(base_dir, exist_ok=True)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.astype('float32'))
    X_test = scaler.transform(X_test.astype('float32'))
    X_test_trans = scaler.transform(X_test_trans.astype('float32'))
    X_test_control = scaler.transform(X_test_control.astype('float32'))

    # Save the train and test datasets
    save_dataset(name, X_train, y_train, X_test, y_test, X_test_trans, y_test_trans, X_test_control, y_test_control, base_dir)

if __name__ == '__main__':
    random_states = [42] #, 9514, 8331, 5816, 3807] 
    tt_numbers = ['1']#, '2', '3', '4', '5'] 
    # Example usage with tt_number
    for tt_number, rs in zip(tt_numbers, random_states):  
        process_data(name='reduced_mnist_data_trans', outcome_name='label', tt_number=tt_number, rs=rs)
