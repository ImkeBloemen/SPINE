import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Load the dataset
dataset_name = 'reduced_mnist_data'
outcome_name = 'label'
dataset = pd.read_csv('../../../data/reduced_mnist_data.csv')
# Define directories to save input data and trained models
input_data_dir = '../../../data/experiment_input/reduced_mnist_data'
input_model_dir = '../models/input_classifiers/evaluation/reduced_mnist_data'

# Filter out unwanted classes (quality 0 and 6)
if "wine" in dataset_name:
    dataset = dataset[dataset['quality'] != 0]
    dataset = dataset[dataset['quality'] != 6]

    # Split dataset into features (X) and target (y)
    y = dataset[outcome_name].values
    X = dataset.drop(outcome_name, axis=1).values
    y = y - 1  # Adjust target values to start from 0
else:
    y = dataset[outcome_name].values
    X = dataset.drop(outcome_name, axis=1).values

# Recreate the dataset for clarity and save processed quality
dataset = pd.DataFrame(X, columns=dataset.columns[:-1])
dataset[outcome_name] = y

os.makedirs(input_data_dir, exist_ok=True)
os.makedirs(input_model_dir, exist_ok=True)

# Define classifier names and file paths for saving models
classifier_names = ["LogisticRegression", "MLP"]
classifier_paths = ["lr_model.h5", "mlp_model.h5"]

# Loop through different random states for data splitting
for i, random_state in enumerate([42, 9514, 8331, 5816, 3807]):
    print(f"Random State: {random_state}, Split Index: {i+1}")
    i = i + 1
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Normalize the features using Min-Max Scaling
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create directories to save train-test splits
    input_data_dir_tt = os.path.join(input_data_dir, f"train_test_{i}")
    os.makedirs(input_data_dir_tt, exist_ok=True)
    
    # Save the processed train and test sets as numpy arrays
    np.save(os.path.join(input_data_dir_tt, "X_train.npy"), X_train)
    np.save(os.path.join(input_data_dir_tt, "X_test.npy"), X_test)
    np.save(os.path.join(input_data_dir_tt, "y_train.npy"), y_train)
    np.save(os.path.join(input_data_dir_tt, "y_test.npy"), y_test)

    # Ensure labels are integer type for categorical conversion
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Display class distribution in the training set
    for label in np.unique(y_train):
        print(f'--> Class {label}: {np.count_nonzero(y_train == label)} samples')

    # Convert labels to one-hot encoded format for training
    n_classes = len(np.unique(y_train))
    y_train_one_hot = to_categorical(y_train, num_classes=n_classes)
    y_test_one_hot = to_categorical(y_test, num_classes=n_classes)

    # Define the number of training epochs
    epochs = 25

    # Train and save models for each classifier
    for classifier_path, clf_name in zip(classifier_paths, classifier_names):
        model_filename = f"{clf_name}_{i}.h5"
        print(f"Training Model: {model_filename}")
        model_save_path = os.path.join(input_model_dir, model_filename)

        # Define model architecture based on classifier type
        if clf_name == "LogisticRegression":
            clf = Sequential([
                Dense(n_classes, input_dim=X_train.shape[1], activation='softmax')
            ])
        elif clf_name == "MLP":
            clf = Sequential([
                Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(0.1),
                Dense(256, activation='relu'),
                Dropout(0.1),
                Dense(256, activation='relu'),
                Dropout(0.1),
                Dense(n_classes, activation='softmax')
            ])
            
        # Compile and train the Logistic Regression model
        if clf_name == "LogisticRegression":
            clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            history = clf.fit(
                X_train, y_train_one_hot,
                batch_size=5,
                epochs=epochs,
                validation_split=0.2
            )
            
        # Compile and train the MLP model with early stopping
        elif clf_name == "MLP":
            clf.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
                loss='categorical_crossentropy', 
                metrics=['accuracy']
            )
            history = clf.fit(
                X_train, y_train_one_hot, 
                batch_size=64, 
                epochs=epochs, 
                validation_split=0.2, 
                callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], 
                verbose=1
            )

        # Evaluate model performance on test data
        y_pred = clf.predict(X_test)
        loss, accuracy = clf.evaluate(X_test, y_test_one_hot, verbose=0)

        # Save evaluation results to a text file
        result_filename = f"{clf_name}_{i}.txt"
        print(f"\tAccuracy on test data: {accuracy}")
        with open(os.path.join(input_model_dir, result_filename), "w") as f:
            f.write(f"Accuracy on test data: {[loss, accuracy]}\n")

        # Save the trained model to disk
        clf.save(model_save_path)
        print(f"Model saved at {model_save_path}")

print("Train-test splits and models saved successfully.")
