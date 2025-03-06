import numpy as np
import os
from time import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from sklearn.utils.extmath import cartesian
from skimage.color import rgb2hsv, hsv2rgb
from tqdm import tqdm
import matplotlib.pyplot as plt  # Correct import
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.cm as cm
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import entropy
import json
from matplotlib.colors import to_rgba

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import ssnp

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 25

def run_experiment(name, classifier_names, classifier_paths, tt_number, epochs_dataset, classes_mult):

    output_dir = f'experiment_output/{name}/train_test_{tt_number}'

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


    def results_to_png(np_matrix, prob_matrix, grid_size, n_classes, dataset_name, classifier_name, 
                       output_dir =output_dir, real_points=None, real_labels=None, real_confidence=None, 
                       indexes_in_range=None, indexes_confidences=None, max_value_hsv=None, suffix=None, 
                       normalized_new_point=None, test_set=None, test_labels=None, test_set_trans=None, 
                       test_labels_trans=None, test_set_subset=None, test_subset_pred=None):
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
                class_idx = int(np_matrix[row, col])  # or round if necessary
                # Make sure we clamp it between [0, n_classes-1].
                class_idx = max(0, min(class_idx, n_classes-1))
                # Use the same color indexing that you used in your class_colors.
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
            # Plot the point as a red cross
            x_coord, y_coord = normalized_new_point[0]
            plt.scatter([x_coord], [y_coord], marker='x', c='red', s=100, linewidths=2, label='New Data Point')

        plt.axis('off')
        ax.axis('off')
        plt.savefig(os.path.join(output_dir, f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}_vanilla_no_points{suffix}.png"), bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        # Vanilla plot with test points
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        ax.imshow(data_vanilla, origin='lower')

        # Vanilla plot with test points
        if test_set is not None:
            fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
            ax.imshow(data_vanilla, origin='lower')
            scatter = ax.scatter(test_set[:, 1], test_set[:, 0], c=test_labels,  cmap=custom_cmap, vmin=0,
                        vmax=n_classes - 1
                        , s=20, edgecolors='k', label='Test Points')
            plt.legend(loc='upper right', fontsize=16, title="Legend")
            ax.axis('off')  # Remove axes
            plt.title("Decision Boundary Mapping with Test Points", font='Times New Roman', fontsize=25)
            plt.savefig(os.path.join(output_dir, f"{classifier_name}_vanilla_test_points{suffix}.png"), bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

        if test_set_trans is not None:
            fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
            ax.imshow(data_vanilla, origin='lower')
            scatter = ax.scatter(test_set_trans[:, 1], test_set_trans[:, 0], c=test_labels_trans, cmap=custom_cmap, vmin=0,
                        vmax=n_classes - 1, s=20, edgecolors='k', label='Transformed Test Points')
            plt.legend(loc='upper right', fontsize=16, title="Legend")
            ax.axis('off')  # Remove axes
            plt.title("Decision Boundary Mapping with Transformed Test Points", font='Times New Roman', fontsize=25)
            plt.savefig(os.path.join(output_dir, f"{classifier_name}_vanilla_test_points_trans{suffix}.png"), bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

        if test_set_trans is not None and test_set_subset is not None:
            fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
            ax.imshow(data_vanilla, origin='lower', alpha = 0.7)
            ax.scatter(test_set_subset[:, 1], test_set_subset[:, 0], c=test_subset_pred, cmap=custom_cmap, vmin=0,
                        vmax=n_classes - 1, s=35,     # Size of the test points
                        alpha=1,
                        edgecolor='black',  # Add a black edge color
                        linewidth=1,  # Set the width of the edge line
                        marker='o',  # Change shape to triangles
                        label='Original Test Points')
            ax.scatter(test_set_trans[:, 1], test_set_trans[:, 0], c=test_labels_trans,s=30,     # Size of the test points
                alpha=1,
                edgecolor='black',  # Add a black edge color
                linewidth=1,  # Set the width of the edge line
                marker='^',  # Change shape to triangles
                label='Transformed Test Points'
                    )
            num_points = min(len(test_set_subset), len(test_set_trans))
            for i in range(num_points):
                start_y, start_x = test_set_subset[i]
                end_y,   end_x   = test_set_trans[i]

                # Determine alpha based on whether the points are predicted in the same class
                alpha = 1.0 if test_subset_pred[i] != test_labels_trans[i] else 0.2

                # Using annotate with arrowprops to draw an arrow:
                plt.annotate(
                    "",
                    xy=(end_x, end_y),
                    xytext=(start_x, start_y),
                    arrowprops=dict(
                        arrowstyle="->",  # arrow style
                        color="black",
                        linestyle='dashed',
                        linewidth=0.5,
                        shrinkA=0,         # tail shrinking
                        shrinkB=0.2,       # head shrinking
                        alpha=alpha
                    )
                )
            
            # Create an overview of y_subset_pred and y_trans_pred
            overview_data = {
                'y_subset_pred': test_subset_pred.tolist(),
                'y_trans_pred': test_labels_trans.tolist()
            }

            # Save the overview as a JSON file
            overview_path = os.path.join(output_dir, f'_overview.json')
            with open(overview_path, 'w') as json_file:
                json.dump(overview_data, json_file)

            print(f"Overview of y_subset_pred and y_trans_pred saved as '{overview_path}'.")
        
            # Add a legend and title
            plt.legend(loc='upper right', fontsize=12)
            plt.title(f"Decision Boundary Mapping with Transformed Test Points", fontsize=25)
            plt.axis('off')  # Hide axes

            # Save the overlay image
            save_path = os.path.join(output_dir, f'_overlay_plot_trans.png')
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Test trans points overlay image saved as '{save_path}'.")

            # Vanilla plot with real output points
        if real_points is not None and real_labels is not None:
            fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
            ax.imshow(data_vanilla, origin='lower')
            scatter = ax.scatter(real_points[:, 1], real_points[:, 0], c=real_labels,  cmap=custom_cmap, vmin=0,
                        vmax=n_classes - 1, s=20, edgecolors='k')
            ax.axis('off')  # Remove axes
            plt.savefig(os.path.join(output_dir, f"{classifier_name}_vanilla_real_output{suffix}.png"), bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

        # Confidence plot with real confidence points
        if real_points is not None and real_confidence is not None:
            norm = Normalize(vmin=0, vmax=1)
            cmap = cm.viridis
            fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
            ax.imshow(data_vanilla, origin='lower')
            scatter = ax.scatter(real_points[:, 1], real_points[:, 0], c=real_confidence, cmap=cmap, norm=norm, s=20, edgecolors='k')
            plt.legend(loc='upper right', fontsize=16, title="Legend")
            ax.axis('off')  # Remove axes
            plt.savefig(os.path.join(output_dir, f"{classifier_name}_confidence{suffix}.png"), bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

        # Alpha plot
        data_alpha = data.copy()
        data_alpha[:, :, 3] = prob_matrix

        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        ax.imshow(data_alpha, origin='lower')

        # if real_points is not None and real_labels is not None:
        #     ax.scatter(real_points[:, 1], real_points[:, 0], c=real_labels, cmap='tab20', s=20, edgecolors='k') #flatten()

        ax.axis('off')
        plt.savefig(os.path.join(output_dir, f"{classifier_name}_alpha{suffix}.png"), bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        # HSV plot
        data_hsv = data[:, :, :3].copy()
        data_hsv = rgb2hsv(data_hsv)
        data_hsv[:, :, 2] = prob_matrix
        data_hsv = hsv2rgb(data_hsv)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        ax.imshow(data_hsv, origin='lower')

        if real_points is not None and real_labels is not None:
            ax.scatter(real_points[:, 1], real_points[:, 0], c=real_labels,  cmap=custom_cmap, vmin=0,
                        vmax=n_classes - 1, s=20, edgecolors='k') #flatten()

        ax.axis('off')
        plt.savefig(os.path.join(output_dir, f"{classifier_name}_hsv{suffix}.png"), bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    def evaluate_boundary_mapping_batch_sdbm(ssnp_model,
                                        clf,
                                        hd_samples,
                                        mesh_classes,
                                        x_intrvls,
                                        y_intrvls):
            """
            Evaluate the SDBM decision boundary mapping for each sample in `hd_samples`.

            Steps for each sample x in hd_samples:
            1) Predict with `clf` on the *original* x => (orig_probs).
            2) Embed x in 2D using ssnp_model.transform(x) => (points_2d).
            3) Inverse-map that 2D point(s) back to HD => (hd_recon).
            4) Predict again on the reconstructed sample => (recon_probs).
            5) Find the 'pixel label' from the background mesh => (pixel_labels).

            Parameters
            ----------
            ssnp_model : ssnp.SSNP
                A trained SDBM/SSNP model with .transform() and .inverse_transform().
            clf : Any classifier with .predict()
                Typically a Keras model, or scikit-learn model, that can handle
                shape (N, feature_dim) => (N, n_classes).
            hd_samples : np.ndarray, shape (N, D)
                High-dimensional samples to evaluate.
            mesh_classes : np.ndarray, shape (grid_size, grid_size)
                The label assigned to each pixel in the 2D decision boundary mesh.
                Typically produced by looping over a 2D grid of points and calling
                clf.predict().
            x_intrvls : np.ndarray, shape (grid_size,)
                The x-coordinates used to build the SDBM mesh (e.g., np.linspace(xmin, xmax, grid_size)).
            y_intrvls : np.ndarray, shape (grid_size,)
                The y-coordinates used to build the SDBM mesh (e.g., np.linspace(ymin, ymax, grid_size)).

            Returns
            -------
            points_2d : np.ndarray, shape (N, 2)
                2D embedding of `hd_samples` in SDBM space.
            hd_recon : np.ndarray, shape (N, D)
                The reconstructed HD samples from the 2D embedding.
            orig_probs : np.ndarray, shape (N, n_classes)
                Predictions on the *original* samples.
            recon_probs : np.ndarray, shape (N, n_classes)
                Predictions on the reconstructed samples.
            pixel_labels : np.ndarray, shape (N,)
                The label from `mesh_classes` for each sample's 2D location.
            """
            import numpy as np

            hd_samples = np.array(hd_samples, dtype=np.float32)
            N = len(hd_samples)

            # 1) Predict with the model on the *original* samples
            orig_probs = clf.predict(hd_samples)  # shape => (N, n_classes)

            # 2) Embed each sample into 2D via SDBM
            #    .transform => (N, 2)
            points_2d = ssnp_model.transform(hd_samples)

            # 3) Inverse-map the 2D points => HD
            hd_recon = ssnp_model.inverse_transform(points_2d)  # shape => (N, D)

            # 4) Predict again on the reconstructed sample
            recon_probs = clf.predict(hd_recon)  # shape => (N, n_classes)

            # 5) Find the pixel label from the background mesh
            pixel_labels = np.zeros(N, dtype=int)

            grid_size = mesh_classes.shape[0]  # assume square mesh
            # optional check => mesh_classes.shape should be (grid_size, grid_size)

            for i in range(N):
                x_2d, y_2d = points_2d[i]

                # - find the nearest index in x_intrvls
                #   (lowest i where x_intrvls[i] >= x_2d),
                #    then clip to [0..grid_size-1]
                x_idx = np.searchsorted(x_intrvls, x_2d)
                x_idx = np.clip(x_idx, 0, grid_size - 1)

                # - find the nearest index in y_intrvls
                y_idx = np.searchsorted(y_intrvls, y_2d)
                y_idx = np.clip(y_idx, 0, grid_size - 1)

                # get label from mesh
                pixel_labels[i] = mesh_classes[y_idx, x_idx]

            return points_2d, hd_recon, orig_probs, recon_probs, pixel_labels
    
    def measure_intrinsic_dimensionality_pca(X: np.ndarray, threshold: float = 0.95) -> int:
        """
        Measures intrinsic dimensionality using PCA's cumulative variance approach.

        Parameters:
        ----------
        X : np.ndarray
            Input dataset.
        threshold : float
            Proportion of variance to retain.

        Returns:
        -------
        int
            Number of components needed to exceed `threshold` proportion of variance.
        """
        pca = PCA().fit(X)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        intrinsic_dim = np.argmax(cumulative_variance > threshold) + 1
        return intrinsic_dim


    def compute_kl_divergence(X: np.ndarray, X_reference: np.ndarray, bins: int = 30) -> float:
        """
        Compute KL divergence between two distributions.

        Parameters:
        ----------
        X : np.ndarray
            First dataset.
        X_reference : np.ndarray
            Reference dataset.
        bins : int
            Number of bins for the histograms.

        Returns:
        -------
        float
            KL divergence value.
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
        1) Project it to 2D using `ssnpgt.transform(...)`.
        2) Reconstruct it back to high-dim using `ssnpgt.inverse_transform(...)`.
        3) Compute the per-sample MSE between original and reconstructed points.
        Save both original and reconstructed HD points (plus MSE) in a single CSV.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_samples, n_features)
            Original high-dimensional training data.
        ssnpgt : object
            A fitted model/transformer that provides:
            - transform(X) -> 2D
            - inverse_transform(X_2D) -> back to HD
        output_dir : str
            Directory where the resulting CSV file should be saved.
        run_name : str, optional
            A name for the output file, defaults to "my_run".

        Returns
        -------
        df : pd.DataFrame
            A DataFrame containing:
            - columns for the original features (orig_feature_X),
            - columns for the reconstructed features (recon_feature_X),
            - per-sample MSE,
            - a sample ID (row index).
            The same data is also saved to a CSV file.
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1) Project training data to 2D
        X_2d = ssnpgt.transform(X_train)  # shape: (n_samples, 2)

        # 2) Inverse-transform the 2D points back to HD
        X_hd_recon = ssnpgt.inverse_transform(X_2d)  # shape: (n_samples, n_features)

        # 3) Compute the MSE for each sample
        #    MSE_i = mean( (X_train[i] - X_hd_recon[i])^2 )
        diff = X_train - X_hd_recon
        mse_per_sample = np.mean(diff**2, axis=1)

        # 4) Build a DataFrame that pairs original HD features with their reconstructed counterparts
        n_samples, n_features = X_train.shape
        orig_col_names = [f"orig_feature_{i}" for i in range(n_features)]
        recon_col_names = [f"recon_feature_{i}" for i in range(n_features)]
        
        df_orig = pd.DataFrame(X_train, columns=orig_col_names)
        df_recon = pd.DataFrame(X_hd_recon, columns=recon_col_names)

        # Combine into a single DataFrame
        df = pd.concat([df_orig, df_recon], axis=1)

        # Add ID and MSE as columns
        df.insert(0, "ID", np.arange(n_samples))
        df["MSE"] = mse_per_sample

        # Compute overall MSE (average across all samples)
        overall_mse = mse_per_sample.mean()

        # 5) Save to CSV
        csv_path = os.path.join(output_dir, f"round_trip_hd_points_{run_name}.csv")
        df.to_csv(csv_path, index=False)

        print(f"Saved {len(df)} row(s) of original & reconstructed HD points to {csv_path}")
        print(f"Overall MSE across all training samples: {overall_mse:.6f}")

        return df

    os.makedirs(output_dir, exist_ok=True)

    data_dir = f'input_data/{name}/train_test_{tt_number}'
    # data_dirs = [name]
    grid_size = 300
    patience = 10
    # epochs = 200
    min_delta = 0.05
    verbose = False
    results = []

    X_train = np.load(os.path.join(data_dir, f'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, f'y_train.npy'))
    X_test = np.load(os.path.join(data_dir, f'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, f'y_test.npy'))
    X_test_trans = np.load(os.path.join(data_dir, f'X_test_trans.npy'))
    y_test_trans = np.load(os.path.join(data_dir,f'y_test_trans.npy'))
    X_test_control = np.load(os.path.join(data_dir, f'X_test_control.npy'))

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    y_test_trans = y_test_trans.astype(int)

    for l in np.unique(y_train):
        print('-->', l, np.count_nonzero(y_train == l))

    n_classes = len(np.unique(y_train))

    y_train_one_hot = to_categorical(y_train, num_classes=n_classes)
    y_test_one_hot = to_categorical(y_test, num_classes=n_classes)
    y_test_trans_one_hot = to_categorical(y_test_trans, num_classes=n_classes)

    results_list = []

    for classifier_path, clf_name in zip(classifier_paths, classifier_names):
        
        # clf = load_model(os.path.join(data_dir, 'models', f'{classifier_path}'))
        input_dim=X_train.shape[1]
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
            # clf = Sequential([
            #     Dense(64, activation='relu', 
            #         input_shape=(input_dim,),
            #         kernel_regularizer=regularizers.l2(1e-4)),
            #     Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-2)),
            #     Dense(n_classes, activation='softmax')
            # ])
        
        out_name = f"{clf_name}_{grid_size}x{grid_size}"
        out_file = os.path.join(output_dir, out_name + ".npy")

        print('------------------------------------------------------')

        epochs = epochs_dataset[name]
        epochs_ssnp = 400
        ssnpgt = ssnp.SSNP(epochs=epochs_ssnp, verbose=verbose, patience=patience, opt='adam', bottleneck_activation='linear')
        name_projector = f"{name}_ssnp_{clf_name}.h5"
        
        ssnpgt.fit(X_train, y_train)
        ssnpgt.save_model(os.path.join(output_dir, name_projector))
        
        print("Projection finished")

        model_filename = f"{clf_name}.h5"
        model_save_path = os.path.join(output_dir, model_filename)

        start = time()
                
        
        if clf_name == "LogisticRegression":
            clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            history = clf.fit(
                X_train, y_train_one_hot,
                batch_size=5,
                epochs=epochs,
                validation_split=0.2)

            loss, accuracy = clf.evaluate(X_test, y_test_one_hot, verbose=0)
            loss_trans, accuracy_trans = clf.evaluate(X_test_trans, y_test_trans_one_hot, verbose=0)

            print("\tAccuracy on test data: ", [loss, accuracy])
            print("\tAccuracy on test trans data: ", [loss_trans, accuracy_trans])
            endtime = time() - start
            with open(os.path.join(output_dir, f"{clf_name}.txt"), "w") as f:
                f.write(f"Accuracy on test data: {[loss, accuracy]}\n")
                f.write(f"Accuracy on test trans data: {[loss_trans, accuracy_trans]}\n")
                f.write(f"Finished training classifier... {endtime}\n")
            print(f"Accuracy on test data: {accuracy}")

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
            
            loss, accuracy = clf.evaluate(X_test, y_test_one_hot, verbose=0)
            loss_trans, accuracy_trans = clf.evaluate(X_test_trans, y_test_trans_one_hot, verbose=0)

            print("\tAccuracy on test data: ", [loss, accuracy])
            print("\tAccuracy on test trans data: ", [loss_trans, accuracy_trans])
            endtime = time() - start
            with open(os.path.join(output_dir, f"{clf_name}.txt"), "w") as f:
                f.write(f"Accuracy on test data: {[loss, accuracy]}\n")
                f.write(f"Accuracy on test trans data: {[loss_trans, accuracy_trans]}\n")
                f.write(f"Finished training classifier... {endtime}\n")
            print(f"Accuracy on test data: {accuracy}")
    
        clf.save(model_save_path)
        print(f"Model saved to {model_save_path}")

        X_ssnpgt = ssnpgt.transform(X_train)
      
        np.save(os.path.join(output_dir, f'X_SSNP_{clf_name}.npy'), X_ssnpgt)

        high_dim_csv_path = os.path.join(output_dir, f"high_dim_data_{clf_name}.csv")
        all_high_dim_data = []

        # scaler = load('scalers/minmax_scaler_hapiness.save')
        # scaler.fit(X_ssnpgt)
        scaler_2D = MinMaxScaler()
        # scaler_2D = StandardScaler()
        scaler_2D.fit(X_ssnpgt)
        # scaler_filename = os.path.join(output_dir, f"minmax_scaler_2D_{clf_name}.save")
        # dump(scaler_2D, scaler_filename)
        xmin, xmax, ymin, ymax = np.min(X_ssnpgt[:, 0]), np.max(X_ssnpgt[:, 0]), np.min(X_ssnpgt[:, 1]), np.max(X_ssnpgt[:, 1])

        print("x_min, x_max, y_min, y_max =", xmin, xmax, ymin, ymax)

        img_grid, prob_grid = np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size))
        x_intrvls, y_intrvls = np.linspace(xmin, xmax, num=grid_size), np.linspace(ymin, ymax, num=grid_size)
        x_grid, y_grid = np.linspace(0, grid_size-1, num=grid_size), np.linspace(0, grid_size-1, num=grid_size)
        pts, pts_grid = cartesian((x_intrvls, y_intrvls)), cartesian((x_grid, y_grid)).astype(int)

        batch_size = 100000
        pbar = tqdm(total=len(pts))

        position = 0
        while position < len(pts):
            pts_batch = pts[position:position+batch_size]
            print("pts batch", pts_batch)
            image_batch = ssnpgt.inverse_transform(pts_batch)
            print("image batch", image_batch)
            all_high_dim_data.append(image_batch)
            probs = clf.predict(image_batch)
            labels = np.argmax(probs, axis=1)
            unique_labels, counts = np.unique(labels, return_counts=True)
            alpha = np.max(probs, axis=1)

            pts_grid_batch = pts_grid[position:position+batch_size]
            # x_idx = pts_grid_batch[:, 0]
            # y_idx = pts_grid_batch[:, 1]

            # img_grid[x_idx, y_idx] = labels
            # prob_grid[x_idx, y_idx] = alpha
            # img_grid[pts_grid_batch[:, 0], pts_grid_batch[:, 1]] = labels
            # prob_grid[pts_grid_batch[:, 0], pts_grid_batch[:, 1]] = alpha
            pts_grid_batch = pts_grid[position:position+batch_size]
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

        X_test_trans_ssnpgt = ssnpgt.transform(X_test_trans)
        normalized_test_trans = scaler_2D.transform(X_test_trans_ssnpgt).astype('float32') * (grid_size - 1)
        normalized_test_trans = np.clip(normalized_test_trans, 0, grid_size - 1).astype(int)
        probs_test_trans = clf.predict(X_test_trans)
        labels_test_trans = np.argmax(probs_test_trans, axis=1)

        X_test_control_ssnpgt = ssnpgt.transform(X_test_control)
        normalized_test_subset = scaler_2D.transform(X_test_control_ssnpgt).astype('float32') * (grid_size - 1)
        normalized_test_subset = np.clip(normalized_test_subset, 0, grid_size - 1).astype(int)
        probs_test_subset = clf.predict(X_test_control)
        labels_test_subset = np.argmax(probs_test_subset, axis=1)

        # new_point = pd.read_csv('input_data/hapiness/x_new.csv').values[0]
        # new_point_2D = ssnpgt.transform(new_point.reshape(1, -1))
        # normalized_new_point = scaler.transform(new_point_2D.reshape(1, -1)).astype('float32') * (grid_size - 1)
        # normalized_new_point = np.clip(normalized_new_point, 0, grid_size - 1).astype(int)

        # indexes_in_range = np.where((confidence_real >= 0.2) & (confidence_real <= 0.8))[0]
        # np.save(os.path.join(output_dir, f"{dataset_name}_indexes_confidence_0.2_0.8.npy"), indexes_in_range)

        results_to_png(np_matrix=img_grid, 
                    prob_matrix=prob_grid, 
                    grid_size=grid_size, 
                    dataset_name=name, 
                    classifier_name=clf_name, 
                    output_dir = output_dir,
                    n_classes=n_classes, 
                    real_points=normalized, 
                    real_labels=y_train, 
                    real_confidence=confidence_real, 
                    # indexes_in_range=indexes_in_range, 
                    # indexes_confidences=confidence_real[indexes_in_range],
                    # normalized_new_point=normalized_new_point,
                    test_set=normalized_test,
                    test_labels=labels_test,
                    test_set_trans=normalized_test_trans,
                    test_labels_trans=labels_test_trans,
                    test_set_subset=normalized_test_subset,
                    test_subset_pred=labels_test_subset)


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

        points_2d, hd_recon, orig_probs_trans, recon_probs_trans, pixel_labels = evaluate_boundary_mapping_batch_sdbm(
            ssnp_model=ssnpgt,
            clf=clf,
            hd_samples=X_test_trans,
            mesh_classes=img_grid,
            x_intrvls=x_intrvls,
            y_intrvls=y_intrvls
        )

        # img_vanilla_path = os.path.join(output_dir, f"{clf_name}_{grid_size}x{grid_size}_{name}_vanilla_no_points.png")
        # img_vanilla = plt.imread(img_vanilla_path)

        # plot_test_points_on_sdbm(
        #     ssnp_model=ssnpgt,
        #     X_test=X_test,         # your HD test data
        #     labels=y_test,            # optional if you want color-coded points
        #     img_grid=img_grid,  # shape (grid_size, grid_size, 3 or 4)
        #     x_intrvls=np.linspace(xmin, xmax, grid_size),
        #     y_intrvls=np.linspace(ymin, ymax, grid_size),
        #     output_dir=output_dir
        # )

        # Compute the accuracy of the classifier on the reconstructed samples
        recon_accuracy = np.mean(np.argmax(orig_probs, axis=1) == np.argmax(recon_probs, axis=1))
        recon_accuracy_trans = np.mean(np.argmax(orig_probs_trans, axis=1) == np.argmax(recon_probs_trans, axis=1))
        
        high_dim_data = pd.read_csv(os.path.join(output_dir, f"high_dim_data_{clf_name}.csv"))

        X_reference = X_train
        X_high_dim = high_dim_data.values

        dim_95_train = measure_intrinsic_dimensionality_pca(X_reference)
        dim_95_high_dim = measure_intrinsic_dimensionality_pca(X_high_dim)

        kl_div = compute_kl_divergence(X_reference, X_high_dim, bins=30)


        results = {
                "classifier": clf_name,
                "accuracy": recon_accuracy,
                "accuracy trans": recon_accuracy_trans,
                "dim_95_train": dim_95_train,
                "dim_95_high_dim": dim_95_high_dim,
                "kl_div": kl_div
            }
        
        results_list.append(results)

        mse_results = round_trip_hd_points_with_mse(X_train, ssnpgt, output_dir, run_name=f"MSE_{name}_{clf_name}")

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(os.path.join(output_dir, f"results.csv"), index=False)

name = 'reduced_mnist_data_trans'
classifier_names = ["MLP"]
classifier_paths = ["MLP.h5"]
tt_numbers = ["1"] #, "2", "3", "4", "5"]
epochs_dataset = {name: 30}
classes_mult = {name: 10}

if __name__ == "__main__":
    for tt_number in tt_numbers:
        run_experiment(name, classifier_names, classifier_paths, tt_number, epochs_dataset, classes_mult)