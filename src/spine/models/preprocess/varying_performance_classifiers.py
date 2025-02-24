import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
import pickle
from joblib import dump, load

# Load dataset
# mnist = pd.read_csv('../../../data/raw/reduced_mnist_data.csv')
# mnist_filtered = mnist[mnist['label'].isin([0.0, 1.0])]
# mnist_filtered = mnist_filtered.sample(n=2000, random_state=42) if mnist_filtered.shape[0] > 2000 else mnist_filtered

# y = mnist_filtered['label'].values
# X = mnist_filtered.drop('label', axis=1).values

# # Convert labels to binary: 0 for '3', 1 for '5'
# y = (y == 1.0).astype(int)

# dataset = pd.DataFrame(X, columns=mnist_filtered.columns[:-1])
# dataset['label'] = y

# dataset.to_csv('../../../data/raw/mnist_filtered_sep.csv', index=False)
# print("Dataset saved.")
dataset = pd.read_csv('../../../data/raw/mnist_filtered.csv')

X = dataset.drop('label', axis=1).values
y = dataset['label'].values

# Create directory to save files
input_data_dir = '../../../experiment_input/ssnp/mnist_filtered'
input_model_dir = '../models/input_classifiers/evaluation/mnist_filtered'
os.makedirs(input_data_dir, exist_ok=True)
os.makedirs(input_model_dir, exist_ok=True)


def build_balanced_model(input_dim):
    """
    Moderately sized network, moderate regularization => balanced fit.
    """
    model = Sequential([
        Dense(64, activation='relu', 
              input_shape=(input_dim,),
              kernel_regularizer=regularizers.l2(1e-4)),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-2)),
        Dense(2, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs, description):
    """
    Trains the model, prints training and test accuracy.
    """
    print(f"\n--- {description} ---")
    print(f"Model summary:")
    model.summary()

    # Fit the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=128,
        validation_split=0.1,
        verbose=0
    )

    # Evaluate on training data
    y_pred_train = (model.predict(X_train) > 0.5).astype("int32")
    train_acc = accuracy_score(y_train, y_pred_train)

    # Evaluate on test data
    y_pred_test = (model.predict(X_test) > 0.5).astype("int32")
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy    : {test_acc:.4f}")

    # Save accuracies to a text file
    with open(os.path.join(input_model_dir, f"accuracy_{description.replace(' ', '_').lower()}_{i}.txt"), 'w') as f:
        f.write(f"Training Accuracy: {train_acc:.4f}\n")
        f.write(f"Test Accuracy    : {test_acc:.4f}\n")

    return history

# Loop to create train-test splits with different random states
for i, random_state in enumerate([42, 9514, 8331, 5816, 3807]):
    print(random_state, i) # Different random states
    i = i + 1
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    # Standardize features
    # scaler = StandardScaler()
    scaler = MinMaxScaler() #SSNP works better with MinMaxScaler, VAE as well
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # One-hot encode labels
    ohe = OneHotEncoder(sparse=False)
    y_train = ohe.fit_transform(y_train.reshape(-1, 1))
    y_test = ohe.transform(y_test.reshape(-1, 1))

    input_data_dir_tt = os.path.join(input_data_dir, f"train_test_{i}")
    os.makedirs(input_data_dir_tt, exist_ok=True)
    
    # Save the scaler
    scaler_filename = os.path.join(input_data_dir_tt, 'scaler.save')
    dump(scaler, scaler_filename)
    
    # Save train and test sets
    np.save(os.path.join(input_data_dir_tt, f"X_train.npy"), X_train)
    np.save(os.path.join(input_data_dir_tt, f"X_test.npy"), X_test)
    np.save(os.path.join(input_data_dir_tt, f"y_train.npy"), y_train)
    np.save(os.path.join(input_data_dir_tt, f"y_test.npy"), y_test)

    input_dim = X_train.shape[1]


    # 3. Build and train Balanced model
    balanced_model = build_balanced_model(input_dim)
    train_and_evaluate(
        balanced_model,
        X_train, y_train, X_test, y_test,
        epochs=20,  # moderate epochs
        description="Balanced MLP Model"
    )
    balanced_model.save(os.path.join(input_model_dir, f"MLP_model_{i}.h5"))
    

print("Train-test splits and models saved.")
