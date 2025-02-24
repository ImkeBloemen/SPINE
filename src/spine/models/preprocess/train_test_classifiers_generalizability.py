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

input_data_dir = '../../../experiment_input/lamp/reduced_mnist_data_gen'
input_model_dir = '../models/input_classifiers/evaluation/lamp/reduced_mnist_data_gen'
os.makedirs(input_data_dir, exist_ok=True)
os.makedirs(input_model_dir, exist_ok=True)

classifier_names = ["MLP"]
classifier_paths = ["mlp_model.h5"]

# Loop to create train-test splits with different random states
X_train = pd.read_csv('../../../data/raw/reduced_mnist_data_train.csv')
y_train = X_train['label'].values
X_train = X_train.drop('label', axis=1).values

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

X_test = pd.read_csv('../../../data/raw/reduced_mnist_data_test_control.csv')
y_test = X_test['label'].values
X_test = X_test.drop('label', axis=1).values

# X_test = scaler.transform(X_test)

X_test_trans = pd.read_csv('../../../data/raw/reduced_mnist_data_test_trans.csv')
y_test_trans = X_test_trans['label'].values
X_test_trans = X_test_trans.drop('label', axis=1).values

X_test_trans = scaler.transform(X_test_trans)

X_test_subset = pd.read_csv('../../../data/raw/reduced_mnist_data_test_subset.csv')
y_test_subset = X_test_subset['label'].values
X_test_subset = X_test_subset.drop('label', axis=1).values

X_test_total = np.concatenate([X_test, X_test_trans], axis=0)
y_test_total = np.concatenate([y_test, y_test_trans], axis=0)

X_test_total = scaler.transform(X_test_total)
X_test = scaler.transform(X_test)

y_train = y_train.astype(int)
y_test = y_test.astype(int)
y_test_trans = y_test_trans.astype(int)
y_test_total = y_test_total.astype(int)

for l in np.unique(y_train):
    print('-->', l, np.count_nonzero(y_train == l))

n_classes = len(np.unique(y_train))

y_train_one_hot = to_categorical(y_train, num_classes=n_classes)
y_test_one_hot = to_categorical(y_test, num_classes=n_classes)
y_test_trans_one_hot = to_categorical(y_test_trans, num_classes=n_classes)
y_test_total_one_hot = to_categorical(y_test_total, num_classes=n_classes)

epochs = 25
i = 1

input_data_dir_tt = os.path.join(input_data_dir, f"train_test_{i}")
os.makedirs(input_data_dir_tt, exist_ok=True)

# Save train and test sets
np.save(os.path.join(input_data_dir_tt, f"X_train.npy"), X_train)
np.save(os.path.join(input_data_dir_tt, f"X_test.npy"), X_test)
np.save(os.path.join(input_data_dir_tt, f"y_train.npy"), y_train)
np.save(os.path.join(input_data_dir_tt, f"y_test.npy"), y_test)
np.save(os.path.join(input_data_dir_tt, f"y_test_trans.npy"), y_test_trans)
np.save(os.path.join(input_data_dir_tt, "X_test_trans.npy"), X_test_trans)
np.save(os.path.join(input_data_dir_tt, "X_test_total.npy"), X_test_total)
np.save(os.path.join(input_data_dir_tt, "y_test_total.npy"), y_test_total)
np.save(os.path.join(input_data_dir_tt, "X_test_subset.npy"), X_test_subset)
np.save(os.path.join(input_data_dir_tt, "y_test_subset.npy"), y_test_subset)

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

            loss_trans, accuracy_trans = clf.evaluate(X_test_trans, y_test_trans_one_hot, verbose=0)

            loss_total, accuracy_total = clf.evaluate(X_test_total, y_test_total_one_hot, verbose=0)

            print("\tAccuracy on test data: ", [loss, accuracy])
            with open(os.path.join(input_model_dir, f"{clf_name}.txt"), "w") as f:
                f.write(f"Accuracy on test data: {[loss, accuracy]}\n")
                f.write(f"Accuracy on test trans data: {[loss_trans, accuracy_trans]}\n")
                f.write(f"Accuracy on test total data: {[loss_total, accuracy_total]}\n")

            print(f"Accuracy on test data: {accuracy}")
            print(f"Accuracy on test trans data: {accuracy_trans}")
            print(f"Accuracy on test total data: {accuracy_total}")


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
        
        loss, accuracy = clf.evaluate(X_test, y_test_one_hot, verbose=0)
        loss_trans, accuracy_trans = clf.evaluate(X_test_trans, y_test_trans_one_hot, verbose=0)
        loss_total, accuracy_total = clf.evaluate(X_test_total, y_test_total_one_hot, verbose=0)

        print("\tAccuracy on test data: ", [loss, accuracy])
        with open(os.path.join(input_model_dir, f"{clf_name}_{i}.txt"), "w") as f:
            f.write(f"Accuracy on test data: {[loss, accuracy]}\n")
            f.write(f"Accuracy on test trans data: {[loss_trans, accuracy_trans]}\n")
            f.write(f"Accuracy on test total data: {[loss_total, accuracy_total]}\n")
            
        print(f"Accuracy on test data: {accuracy}")
        print(f"Accuracy on test trans data: {accuracy_trans}")
        print(f"Accuracy on test total data: {accuracy_total}")

    clf.save(model_save_path)
    print(f"Model saved at {model_save_path}")


print("Train-test splits and models saved.")
