from sklearn.preprocessing import LabelEncoder
from copy import deepcopy
from scipy.spatial.distance import euclidean
import numpy as np

def label_encoders(data):
    label_encoders = {}
    data_new = deepcopy(data)
    for column in data_new.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        new_value = le.fit_transform(data.loc[:,column])
        data_new.loc[:,column] = new_value
        label_encoders[column] = le
    return data_new, label_encoders

def decode_labels(encoded_data, label_encoders):
    decoded_data = deepcopy(encoded_data)
    for column, le in label_encoders.items():
        if column in decoded_data.columns:
            decoded_data[column] = le.inverse_transform(decoded_data[column].astype(int))
    return decoded_data

def encode_labels(data):
    data_encoded, encoders = label_encoders(data)
    return data_encoded, encoders

def plot_lines_between_points(points, ax, color, linestyle='--', line_alpha=0.3):
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linestyle=linestyle, alpha=line_alpha)

def find_shortest_route(routes_with_distances):
    if not routes_with_distances:
        return None  # Handle empty input case

    # Find the tuple with the minimum distance
    shortest_route = min(routes_with_distances, key=lambda x: x[1])
    
    return shortest_route

def rotate_and_flip_points(intermediate_points_2D, grid_size):
    """
    Rotate points by 180 degrees and then flip horizontally within a grid of given size.
    (x, y) -> rotate 180° -> (grid_size-1 - x, grid_size-1 - y)
              flip horizontally -> (grid_size-1 - (grid_size-1 - x), grid_size-1 - y)
                               -> (x, grid_size-1 - y)
    This ends up being effectively a vertical flip if you apply both steps as described.
    If you need a specific final transformation, adjust the logic accordingly.
    """

    # Rotate by 180 degrees
    flipped = np.column_stack((grid_size - 1 - intermediate_points_2D[:, 0],
                               intermediate_points_2D[:, 1]))


    return flipped