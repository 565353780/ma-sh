import numpy as np
from scipy.spatial.transform import Rotation


def rotate_obb_z(angle):
    r = Rotation.from_euler('z', angle, degrees=True)
    return r.as_matrix()

def transform_to_obb(points, rotation_matrix, translation_vector):
    centered_points = points - translation_vector
    return np.dot(centered_points, rotation_matrix.T)

def transform_from_obb(points, rotation_matrix, translation_vector):
    transformed_points = np.dot(points, rotation_matrix)
    return transformed_points + translation_vector

def clip_with_obb(points, obb_size, rotation_angle, obb_center=(0, 0, 0)):
    rotation_matrix = rotate_obb_z(rotation_angle)
    translation_vector = np.array(obb_center)
    points_obb = transform_to_obb(points, rotation_matrix, translation_vector)

    half_size = np.array(obb_size) / 2
    x_min, y_min, z_min = -half_size
    x_max, y_max, z_max = half_size

    mask = ((points_obb[:, 0] >= x_min) & (points_obb[:, 0] <= x_max) &
            (points_obb[:, 1] >= y_min) & (points_obb[:, 1] <= y_max) &
            (points_obb[:, 2] >= z_min) & (points_obb[:, 2] <= z_max))
    clipped_points_obb = points_obb[mask]

    clipped_points = transform_from_obb(clipped_points_obb, rotation_matrix, translation_vector)
    return clipped_points
