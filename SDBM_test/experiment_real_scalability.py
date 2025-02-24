import numpy as np
import os
import pickle
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.utils.extmath import cartesian
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from skimage.color import rgb2hsv, hsv2rgb
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import matplotlib.cm as cm
from PIL import Image
import pandas as pd
from joblib import dump, load
from sklearn.decomposition import PCA
from scipy.stats import entropy
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score
from matplotlib.colors import to_rgba

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model

import ssnp

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 25

def run_experiment(name, classifier_names, classifier_paths, tt_number, epochs_dataset, classes_mult, grid_size):
    # -------------------------------------------------------------------------
    # 1) Start overall timer:
    total_experiment_start = time()
    accumulated_training_time = 0.0
    # -------------------------------------------------------------------------

    output_dir = f'../data/experiment_output/{name}_time_MLP/train_test_{tt_number}_{grid_size}'

    def save_confidence_image(prob_matrix, grid_size, output_path):
        """Save confidence image with consistent formatting."""
        norm = Normalize(vmin=0, vmax=1)
        cmap = cm.viridis
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        im = ax.imshow(prob_matrix, cmap=cmap, norm=norm, origin='lower')
        ax.axis('off')  # Remove axes
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Confidence', rotation=-90, va="bottom", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"Confidence map image saved to {output_path}")

    def results_to_png(np_matrix, prob_matrix, grid_size, n_classes, dataset_name, classifier_name, output_dir=output_dir,
                       real_points=None, real_labels=None, real_confidence=None, indexes_in_range=None, indexes_confidences=None,
                       max_value_hsv=None, suffix=None, normalized_new_point=None, test_set=None, test_labels=None):

        if suffix is not None:
            suffix = f"_{suffix}"
        else:
            suffix = ""

        if n_classes == 2:
            class_colors = [cm.tab20(0), cm.tab20(19)]
        elif n_classes <= 10:
            class_colors = [cm.tab20(i) for i in range(n_classes)]
        else:
            class_colors = [to_rgba(cm.tab20(i % 20)) for i in range(n_classes)]

        custom_cmap = ListedColormap(class_colors)

        data = np.zeros((grid_size, grid_size, 4), dtype=float)
        for row in range(grid_size):
            for col in range(grid_size):
                class_idx = int(np_matrix[row, col])
                class_idx = max(0, min(class_idx, n_classes - 1))
                data[row, col] = class_colors[class_idx % len(class_colors)]

        data_vanilla = data[..., :3]

        if max_value_hsv is not None:
            data_vanilla = rgb2hsv(data_vanilla)
            data_vanilla[:, :, 2] = max_value_hsv
            data_vanilla = hsv2rgb(data_vanilla)

        # Vanilla plot without points
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        ax.imshow(data_vanilla, origin='lower')
        if normalized_new_point is not None:
            x_coord, y_coord = normalized_new_point[0]
            plt.scatter([x_coord], [y_coord], marker='x', c='red', s=100, linewidths=2, label='New Data Point')
        plt.axis('off')
        ax.axis('off')
        plt.savefig(os.path.join(output_dir, f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}_vanilla_no_points{suffix}.png"),
                    bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        # Vanilla plot with test points
        if test_set is not None:
            fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
            ax.imshow(data_vanilla, origin='lower')
            scatter = ax.scatter(test_set[:, 1], test_set[:, 0], c=test_labels, cmap=custom_cmap, vmin=0,
                                 vmax=n_classes - 1, s=20, edgecolors='k', label='Test Points')
            plt.legend(loc='upper right', fontsize=16, title="Legend")
            ax.axis('off')
            plt.savefig(os.path.join(output_dir, f"{classifier_name}_vanilla_test_points{suffix}.png"),
                        bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

        # Vanilla plot with real output points
        if real_points is not None and real_labels is not None:
            fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
            ax.imshow(data_vanilla, origin='lower')
            scatter = ax.scatter(real_points[:, 1], real_points[:, 0], c=real_labels, cmap=custom_cmap, vmin=0,
                                 vmax=n_classes - 1, s=20, edgecolors='k')
            ax.axis('off')
            plt.savefig(os.path.join(output_dir, f"{classifier_name}_vanilla_real_output{suffix}.png"),
                        bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

        # Confidence plot with real confidence points
        if real_points is not None and real_confidence is not None:
            norm = Normalize(vmin=0, vmax=1)
            cmap = cm.viridis
            fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
            ax.imshow(data_vanilla, origin='lower')
            scatter = ax.scatter(real_points[:, 1], real_points[:, 0], c=real_confidence, cmap=cmap, norm=norm,
                                 s=20, edgecolors='k')
            plt.legend(loc='upper right', fontsize=16, title="Legend")
            ax.axis('off')
            plt.savefig(os.path.join(output_dir, f"{classifier_name}_confidence{suffix}.png"),
                        bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

        # Alpha plot
        data_alpha = data.copy()
        data_alpha[:, :, 3] = prob_matrix
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        ax.imshow(data_alpha, origin='lower')
        ax.axis('off')
        plt.savefig(os.path.join(output_dir, f"{classifier_name}_alpha{suffix}.png"),
                    bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        # HSV plot
        data_hsv = data[:, :, :3].copy()
        data_hsv = rgb2hsv(data_hsv)
        data_hsv[:, :, 2] = prob_matrix
        data_hsv = hsv2rgb(data_hsv)
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        ax.imshow(data_hsv, origin='lower')
        if real_points is not None and real_labels is not None:
            ax.scatter(real_points[:, 1], real_points[:, 0], c=real_labels, cmap=custom_cmap, vmin=0,
                       vmax=n_classes - 1, s=20, edgecolors='k')
        ax.axis('off')
        plt.savefig(os.path.join(output_dir, f"{classifier_name}_hsv{suffix}.png"),
                    bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    def evaluate_boundary_mapping_batch_sdbm(ssnp_model,
                                             clf,
                                             hd_samples,
                                             mesh_classes,
                                             x_intrvls,
                                             y_intrvls):
        """
        Evaluate the SDBM decision boundary mapping for each sample in `hd_samples`.
        """
        import numpy as np

        hd_samples = np.array(hd_samples, dtype=np.float32)
        N = len(hd_samples)

        # 1) Predict on the *original* samples
        orig_probs = clf.predict(hd_samples)

        # 2) Embed each sample into 2D
        points_2d = ssnp_model.transform(hd_samples)

        # 3) Inverse-map the 2D points => HD
        hd_recon = ssnp_model.inverse_transform(points_2d)

        # 4) Predict again on the reconstructed sample
        recon_probs = clf.predict(hd_recon)

        # 5) Find the pixel label from the background mesh
        pixel_labels = np.zeros(N, dtype=int)
        grid_size = mesh_classes.shape[0]

        for i in range(N):
            x_2d, y_2d = points_2d[i]
            x_idx = np.searchsorted(x_intrvls, x_2d)
            x_idx = np.clip(x_idx, 0, grid_size - 1)
            y_idx = np.searchsorted(y_intrvls, y_2d)
            y_idx = np.clip(y_idx, 0, grid_size - 1)
            pixel_labels[i] = mesh_classes[y_idx, x_idx]

        return points_2d, hd_recon, orig_probs, recon_probs, pixel_labels

    def measure_intrinsic_dimensionality_pca(X: np.ndarray, threshold: float = 0.95) -> int:
        """
        Measures intrinsic dimensionality using PCA's cumulative variance approach.
        """
        pca = PCA().fit(X)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        intrinsic_dim = np.argmax(cumulative_variance > threshold) + 1
        return intrinsic_dim

    def compute_kl_divergence(X: np.ndarray, X_reference: np.ndarray, bins: int = 30) -> float:
        """
        Compute KL divergence between two distributions.
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

    def round_trip_hd_points_with_mse(X_train, ssnpgt, output_dir, run_name="my_run"):
        """
        For each sample in X_train (high-dimensional):
        1) Project to 2D
        2) Reconstruct
        3) Compute MSE
        4) Save to CSV
        """
        os.makedirs(output_dir, exist_ok=True)
        X_2d = ssnpgt.transform(X_train)
        X_hd_recon = ssnpgt.inverse_transform(X_2d)
        diff = X_train - X_hd_recon
        mse_per_sample = np.mean(diff ** 2, axis=1)

        n_samples, n_features = X_train.shape
        orig_col_names = [f"orig_feature_{i}" for i in range(n_features)]
        recon_col_names = [f"recon_feature_{i}" for i in range(n_features)]

        df_orig = pd.DataFrame(X_train, columns=orig_col_names)
        df_recon = pd.DataFrame(X_hd_recon, columns=recon_col_names)

        df = pd.concat([df_orig, df_recon], axis=1)
        df.insert(0, "ID", np.arange(n_samples))
        df["MSE"] = mse_per_sample

        overall_mse = mse_per_sample.mean()
        csv_path = os.path.join(output_dir, f"round_trip_hd_points_{run_name}.csv")
        df.to_csv(csv_path, index=False)

        print(f"Saved {len(df)} row(s) of original & reconstructed HD points to {csv_path}")
        print(f"Overall MSE across all training samples: {overall_mse:.6f}")
        return df

    os.makedirs(output_dir, exist_ok=True)

    data_dir = f'input_data/{name}/train_test_{tt_number}'
    grid_size = grid_size
    patience = 5
    min_delta = 0.05
    verbose = False

    X_train = np.load(os.path.join(data_dir, f'X_train.npy'))
    print("X_train shape:", X_train.shape)
    y_train = np.load(os.path.join(data_dir, f'y_train.npy'))
    X_test = np.load(os.path.join(data_dir, f'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, f'y_test.npy'))

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    for l in np.unique(y_train):
        print('-->', l, np.count_nonzero(y_train == l))

    n_classes = len(np.unique(y_train))
    y_train_one_hot = to_categorical(y_train, num_classes=n_classes)
    y_test_one_hot = to_categorical(y_test, num_classes=n_classes)

    results_list = []

    for classifier_path, clf_name in zip(classifier_paths, classifier_names):
        # ---------------------------------------------------------------------
        # Everything before actual training is normal experiment time
        # ---------------------------------------------------------------------
        
        input_dim = X_train.shape[1]
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

        out_name = f"{clf_name}_{grid_size}x{grid_size}"
        out_file = os.path.join(output_dir, out_name + ".npy")

        print('------------------------------------------------------')

        epochs = epochs_dataset[name]
        epochs_ssnp = 200
        ssnpgt = ssnp.SSNP(epochs=epochs_ssnp, verbose=verbose, patience=patience,
                           opt='adam', bottleneck_activation='linear')
        name_projector = f"{name}_ssnp_{clf_name}.h5"

        ssnpgt.fit(X_train, y_train)
        ssnpgt.save_model(os.path.join(output_dir, name_projector))
        print("Projection finished")

        model_filename = f"{clf_name}.h5"
        model_save_path = os.path.join(output_dir, model_filename)

        # ---------------------------------------------------------------------
        # 2) Training starts
        training_start = time()
        # ---------------------------------------------------------------------
        
        if clf_name == "LogisticRegression":
            clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            history = clf.fit(
                X_train, y_train_one_hot,
                batch_size=5,
                epochs=epochs,
                validation_split=0.2
            )
            loss, accuracy = clf.evaluate(X_test, y_test_one_hot, verbose=0)
            endtime = time() - training_start  # time just for training
            with open(os.path.join(output_dir, f"{clf_name}.txt"), "w") as f:
                f.write(f"Accuracy on test data: {[loss, accuracy]}\n")
                f.write(f"Finished training classifier... {endtime}\n")
            
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
            loss, accuracy = clf.evaluate(X_test, y_test_one_hot, verbose=0)
            endtime = time() - training_start  # time just for training
            with open(os.path.join(output_dir, f"{clf_name}.txt"), "w") as f:
                f.write(f"Accuracy on test data: {[loss, accuracy]}\n")
                f.write(f"Finished training classifier... {endtime}\n")

        # ---------------------------------------------------------------------
        # Accumulate the training time and then resume the rest of experiment
        accumulated_training_time += endtime
        # ---------------------------------------------------------------------

        print("\tAccuracy on test data: ", [loss, accuracy])
        print(f"Training for {clf_name} took {endtime:.2f} seconds.")

        clf.save(model_save_path)
        print(f"Model saved to {model_save_path}")

        # ---------------------------------------------------------------------
        # Continue experiment tasks after training
        # ---------------------------------------------------------------------
        
        X_ssnpgt = ssnpgt.transform(X_train)
        np.save(os.path.join(output_dir, f'X_SSNP_{clf_name}.npy'), X_ssnpgt)

        high_dim_csv_path = os.path.join(output_dir, f"high_dim_data_{clf_name}.csv")
        all_high_dim_data = []

        scaler_2D = MinMaxScaler()
        scaler_2D.fit(X_ssnpgt)
        scaler_filename = os.path.join(output_dir, f"minmax_scaler_2D_{clf_name}.save")
        dump(scaler_2D, scaler_filename)
        xmin, xmax = np.min(X_ssnpgt[:, 0]), np.max(X_ssnpgt[:, 0])
        ymin, ymax = np.min(X_ssnpgt[:, 1]), np.max(X_ssnpgt[:, 1])
        print("x_min, x_max, y_min, y_max =", xmin, xmax, ymin, ymax)

        img_grid, prob_grid = np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size))
        x_intrvls, y_intrvls = np.linspace(xmin, xmax, num=grid_size), np.linspace(ymin, ymax, num=grid_size)
        x_grid, y_grid = np.linspace(0, grid_size - 1, num=grid_size), np.linspace(0, grid_size - 1, num=grid_size)
        pts, pts_grid = cartesian((x_intrvls, y_intrvls)), cartesian((x_grid, y_grid)).astype(int)

        batch_size = 100000
        pbar = tqdm(total=len(pts))
        position = 0
        while position < len(pts):
            pts_batch = pts[position:position + batch_size]
            image_batch = ssnpgt.inverse_transform(pts_batch)
            all_high_dim_data.append(image_batch)
            probs = clf.predict(image_batch)
            labels = np.argmax(probs, axis=1)
            alpha = np.max(probs, axis=1)

            pts_grid_batch = pts_grid[position:position + batch_size]
            img_grid[pts_grid_batch[:, 0], pts_grid_batch[:, 1]] = labels
            prob_grid[pts_grid_batch[:, 0], pts_grid_batch[:, 1]] = alpha
            position += batch_size
            pbar.update(batch_size)
        pbar.close()

        pd.DataFrame(np.vstack(all_high_dim_data)).to_csv(high_dim_csv_path, index=False)
        np.save(os.path.join(output_dir, f"{out_name}.npy"), img_grid)
        np.save(os.path.join(output_dir, f"{out_name}_prob.npy"), prob_grid)

        normalized = scaler_2D.transform(X_ssnpgt).astype('float32') * (grid_size - 1)
        normalized = np.clip(normalized, 0, grid_size - 1).astype(int)
        confidence_real = clf.predict(X_train)
        confidence_real = np.max(confidence_real, axis=1)

        X_test_ssnpgt = ssnpgt.transform(X_test)
        normalized_test = scaler_2D.transform(X_test_ssnpgt).astype('float32') * (grid_size - 1)
        normalized_test = np.clip(normalized_test, 0, grid_size - 1).astype(int)
        probs_test = clf.predict(X_test)
        labels_test = np.argmax(probs_test, axis=1)

        results_to_png(
            np_matrix=img_grid,
            prob_matrix=prob_grid,
            grid_size=grid_size,
            dataset_name=name,
            classifier_name=clf_name,
            output_dir=output_dir,
            n_classes=n_classes,
            real_points=normalized,
            real_labels=y_train,
            real_confidence=confidence_real,
            test_set=normalized_test,
            test_labels=labels_test
        )

        confidence_output_path = os.path.join(output_dir, f"{clf_name}_{grid_size}x{grid_size}_confidence.png")
        save_confidence_image(prob_grid, grid_size, confidence_output_path)

        points_2d, hd_recon, orig_probs, recon_probs, pixel_labels = evaluate_boundary_mapping_batch_sdbm(
            ssnp_model=ssnpgt,
            clf=clf,
            hd_samples=X_test,
            mesh_classes=img_grid,
            x_intrvls=x_intrvls,
            y_intrvls=y_intrvls
        )

        recon_accuracy = np.mean(np.argmax(orig_probs, axis=1) == np.argmax(recon_probs, axis=1))

        high_dim_data = pd.read_csv(os.path.join(output_dir, f"high_dim_data_{clf_name}.csv")).values
        X_reference = X_train

        dim_95_train = measure_intrinsic_dimensionality_pca(X_reference)
        dim_95_high_dim = measure_intrinsic_dimensionality_pca(high_dim_data)
        kl_div = compute_kl_divergence(X_reference, high_dim_data, bins=30)

        results_list.append({
            "classifier": clf_name,
            "accuracy": recon_accuracy,
            "dim_95_train": dim_95_train,
            "dim_95_high_dim": dim_95_high_dim,
            "kl_div": kl_div
        })

        mse_results = round_trip_hd_points_with_mse(X_train, ssnpgt, output_dir, run_name=f"MSE_{name}_{clf_name}")

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(os.path.join(output_dir, f"results.csv"), index=False)

    # -------------------------------------------------------------------------
    # 3) End overall timer and subtract the accumulated training time
    total_experiment_end = time()
    total_experiment_time = total_experiment_end - total_experiment_start
    total_experiment_excl_training = total_experiment_time - accumulated_training_time

    time_report_path = os.path.join(output_dir, "time_report.txt")
    with open(time_report_path, "w") as f:
        f.write("===========================================\n")
        f.write(f"Total experiment time (including training):  {total_experiment_time:.2f} sec\n")
        f.write(f"Total training time (all models combined):   {accumulated_training_time:.2f} sec\n")
        f.write(f"Experiment time (excluding training):        {total_experiment_excl_training:.2f} sec\n")
        f.write("===========================================\n")
    print(f"Time report saved to {time_report_path}")


name = 'reduced_mnist_data'
classifier_names = ["MLP"]
classifier_paths = ["MLP.h5"] #, "MLP.h5"]
tt_numbers = ["1"]
epochs_dataset = {name: 30}
classes_mult = {name: 2}

if __name__ == "__main__":
    for tt_number in tt_numbers:
        for grid_size in [50, 100, 150, 200, 250, 300]:
            run_experiment(name, classifier_names, classifier_paths, tt_number, epochs_dataset, classes_mult, grid_size)
