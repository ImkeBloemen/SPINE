import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


# Load MNIST dataset
dataset = pd.read_csv('../../../data/raw/winequality-white.csv')
# dataset = dataset.sample(n=3000, random_state=42)
# dataset.to_csv('../../../data/raw/reduced_mnist_data_subset.csv', index=False)
# Drop all rows with y value 0
dataset = dataset[dataset['quality'] != 0]
dataset = dataset[dataset['quality'] != 6]
y = dataset['quality'].values
X = dataset.drop('quality', axis=1).values
y = y - 1

dataset = pd.DataFrame(X, columns=dataset.columns[:-1])
dataset['quality'] = y

# dataset.to_csv('../../../data/raw/winequality-white.csv', index=False)
# print("Dataset saved")

# Create directory to save files
input_data_dir = '../../../experiment_input/lamp/winequality-white'
input_model_dir = '../models/input_classifiers/evaluation/lamp/winequality-white'
os.makedirs(input_data_dir, exist_ok=True)
os.makedirs(input_model_dir, exist_ok=True)

classifier_names = ["LogisticRegression", "MLP"]
classifier_paths = ["lr_model.h5", "mlp_model.h5"]

# Loop to create train-test splits with different random states
for i, random_state in enumerate([42, 9514, 8331, 5816, 3807]):
    print(random_state, i) # Different random states
    i = i + 1
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    input_data_dir_tt = os.path.join(input_data_dir, f"train_test_{i}")
    os.makedirs(input_data_dir_tt, exist_ok=True)
    
    # Save train and test sets
    np.save(os.path.join(input_data_dir_tt, f"X_train.npy"), X_train)
    np.save(os.path.join(input_data_dir_tt, f"X_test.npy"), X_test)
    np.save(os.path.join(input_data_dir_tt, f"y_train.npy"), y_train)
    np.save(os.path.join(input_data_dir_tt, f"y_test.npy"), y_test)

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    for l in np.unique(y_train):
        print('-->', l, np.count_nonzero(y_train == l))

    n_classes = len(np.unique(y_train))

    y_train_one_hot = to_categorical(y_train, num_classes=n_classes)
    y_test_one_hot = to_categorical(y_test, num_classes=n_classes)

    epochs = 25

    for classifier_path, clf_name in zip(classifier_paths, classifier_names):

        model_filename = f"{clf_name}_{i}.h5"
        print(model_filename)
        model_save_path = os.path.join(input_model_dir, model_filename)

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
            
        if clf_name == "LogisticRegression":
                clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                history = clf.fit(
                    X_train, y_train_one_hot,
                    batch_size=5,
                    epochs=epochs,
                    validation_split=0.2)
                
                y_pred = clf.predict(X_test)
                print(y_pred)
                
                loss, accuracy = clf.evaluate(X_test, y_test_one_hot, verbose=0)

                print("\tAccuracy on test data: ", [loss, accuracy])
                with open(os.path.join(input_model_dir, f"{clf_name}.txt"), "w") as f:
                    f.write(f"Accuracy on test data: {[loss, accuracy]}\n")
        elif clf_name == "MLP":
            clf.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
            history = clf.fit(
                X_train, y_train_one_hot, 
                batch_size=64, 
                epochs=epochs, 
                validation_split=0.2, 
                callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], 
                verbose=1)
            
            y_pred = clf.predict(X_test)
            print(y_pred)
            loss, accuracy = clf.evaluate(X_test, y_test_one_hot, verbose=0)

            print("\tAccuracy on test data: ", [loss, accuracy])
            with open(os.path.join(input_model_dir, f"{clf_name}_{i}.txt"), "w") as f:
                f.write(f"Accuracy on test data: {[loss, accuracy]}\n")
            print(f"Accuracy on test data: {accuracy}")

        clf.save(model_save_path)
        print(f"Model saved at {model_save_path}")


print("Train-test splits and models saved.")
