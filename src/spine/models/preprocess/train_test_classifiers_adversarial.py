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

print(y)

dataset = pd.DataFrame(X, columns=dataset.columns[:-1])
dataset['quality'] = y

# dataset.to_csv('../../../data/raw/winequality-white.csv', index=False)
# print("Dataset saved")

# Create directory to save files
input_data_dir = '../../../experiment_input/lamp/winequality-white_adv'
input_model_dir = '../models/input_classifiers/evaluation/lamp/winequality-white_adv'
os.makedirs(input_data_dir, exist_ok=True)
os.makedirs(input_model_dir, exist_ok=True)

classifier_names = ["LogisticRegression", "MLP"]
classifier_paths = ["lr_model.h5", "mlp_model.h5"]

# Loop to create train-test splits with different random states
for i, random_state in enumerate([42]): #, 9514, 8331, 5816, 3807]):
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

    epochs = 30

    # ------------------------------------------------------
    # Define a helper for FGSM
    # ------------------------------------------------------
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    def generate_fgsm(model, x_in, y_in, epsilon=0.01):
        """
        Generate FGSM adversarial samples for a batch of (x_in, y_in).
        x_in:  tf.float32 Tensor, shape (batch, features)
        y_in:  tf.float32 Tensor, shape (batch, n_classes) one-hot
        """
        x_in = tf.cast(x_in, tf.float32)
        y_in = tf.cast(y_in, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x_in)
            preds = model(x_in, training=False)
            loss_value = loss_fn(y_in, preds)
        # Gradient of loss w.r.t. x_in
        grads = tape.gradient(loss_value, x_in)
        # Sign of gradient
        signed_grads = tf.sign(grads)
        # FGSM step
        adv_x = x_in + epsilon * signed_grads
        # Clip to valid range if desired, e.g. [0,1] because we used MinMaxScaler
        adv_x = tf.clip_by_value(adv_x, 0, 1)
        return adv_x

    for classifier_path, clf_name in zip(classifier_paths, classifier_names):

        model_filename  = f"{clf_name}_{i}.h5"
        model_save_path = os.path.join(input_model_dir, model_filename)

        if clf_name == "LogisticRegression":
            # Just a single dense layer with softmax
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

        # Compile
        clf.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train
        history = clf.fit(
            X_train, y_train_one_hot,
            batch_size=64,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
            verbose=1
        )

        # Evaluate on clean test
        loss_clean, acc_clean = clf.evaluate(X_test, y_test_one_hot, verbose=0)
        print(f"{clf_name} [Clean Test] Loss: {loss_clean:.4f}, Accuracy: {acc_clean:.4f}")

        # ---- Adversarial Code Below ----
        # Convert X_test, y_test_one_hot to TF Tensors
        # Select a subset of the test data
        subset_size = 50  # Define the subset size
        random_indices = np.random.choice(len(X_test), subset_size, replace=False)
        X_test_subset = X_test[random_indices]
        y_test_subset_one_hot = y_test_one_hot[random_indices]
        y_test_subset = y_test[random_indices]

        x_test_tensor = tf.convert_to_tensor(X_test_subset, dtype=tf.float32)
        y_test_tensor = tf.convert_to_tensor(y_test_subset_one_hot, dtype=tf.float32)

        # Generate adversarial samples using FGSM in feature space
        epsilon = 0.04  # tweak as needed
        X_test_adv = generate_fgsm(clf, x_test_tensor, y_test_tensor, epsilon=epsilon)

        # Save adversarial test set
        np.save(os.path.join(input_data_dir_tt, f"X_test_adv.npy"), X_test_adv.numpy())
        np.save(os.path.join(input_data_dir_tt, f"y_test_adv.npy"), y_test_subset)
        np.save(os.path.join(input_data_dir_tt, f"X_test_subset.npy"), X_test_subset)

        # Replace the transformed rows in X_test by the adversarial samples
        X_test[random_indices] = X_test_adv.numpy()
        y_test[random_indices] = y_test[random_indices]
        
        # Save normal test set
        np.save(os.path.join(input_data_dir_tt, f"X_test_total.npy"), X_test)
        np.save(os.path.join(input_data_dir_tt, f"y_test_total.npy"), y_test)

        # Evaluate model on adversarial test set
        loss, acc = clf.evaluate(X_test, y_test_one_hot, verbose=0)
        loss_adv, acc_adv = clf.evaluate(X_test_adv, y_test_subset_one_hot, verbose=0)
        print(f"{clf_name} [Adversarial Test, eps={epsilon}] Loss: {loss_adv:.4f}, Accuracy: {acc_adv:.4f}")
        
        # Save evaluation results to a text file
        results_file = os.path.join(input_data_dir_tt, f"evaluation_results_{clf_name}.txt")
        with open(results_file, "w") as f:
            f.write(f"{clf_name} [Total Test] Loss: {loss:.4f}, Accuracy: {acc:.4f}\n")
            f.write(f"{clf_name} [Adversarial Test, eps={epsilon}] Loss: {loss_adv:.4f}, Accuracy: {acc_adv:.4f}\n")
        
        # Save model
        clf.save(model_save_path)
        print(f"Model saved at {model_save_path}")

print("Train-test splits and models (with adversarial evaluations) saved.")