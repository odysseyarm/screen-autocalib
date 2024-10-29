import numpy as np
import numpy.typing as npt
import quaternion
import pyrealsense2 as rs
from typing import Any, List, Literal, Optional, Tuple, cast
from skspatial.objects import Plane
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class PlaneModel(BaseEstimator, RegressorMixin):
    def fit(self, X, y): # type: ignore
        points = np.c_[X, y]
        plane = Plane.best_fit(points)
        self.normal_ = plane.normal
        self.point_ = plane.point
        return self

    def predict(self, X): # type: ignore
        d = -self.point_.dot(self.normal_)
        return (-self.normal_[0] * X[:, 0] - self.normal_[1] * X[:, 1] - d) / self.normal_[2] # type: ignore

def plane_from_points(points: npt.NDArray[np.float32], min_samples: int) -> Optional[Tuple[np.ndarray[Literal[3], np.dtype[np.float32]], np.ndarray[Literal[3], np.dtype[np.float32]]]]:
    # Prepare data for RANSAC
    X = points[:, :2]  # X and Y coordinates
    y = points[:, 2]   # Z coordinate

    if len(points) < min_samples:
        return None

    # Fit RANSAC regressor with PlaneModel
    ransac = RANSACRegressor(
        PlaneModel(), 
        min_samples=min_samples,  # Minimum number of samples to define a plane
        residual_threshold=0.01, 
        max_trials=10000
    )
    ransac.fit(X, y)

    # Extract inlier points
    inlier_mask = ransac.inlier_mask_ # type: ignore
    inliers = points[inlier_mask]

    # Use skspatial to find the best fitting plane for the inliers
    plane = Plane.best_fit(inliers)

    # Extract centroid and normal from the fitted plane
    centroid = plane.point
    normal = plane.normal

    return centroid, normal

def compute_xy_transformation_matrix(plane: Tuple[np.ndarray[Literal[3], np.float32], np.ndarray[Literal[3], np.float32]]) -> np.ndarray[Literal[4, 4], np.float64]:
    point, normal = plane

    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    if normal[2] > 0:
        z = normal
    else:
        z = -normal

    x = np.cross(np.array([0, 1, 0]), z)
    y = np.cross(z, x)

    # Construct the rotation matrix (column-major)
    rotation_matrix = np.eye(4, dtype=np.float64)
    rotation_matrix[:3, :3] = np.array([x, y, z], dtype=np.float64)

    # Construct the translation matrix (column-major)
    translation_matrix = np.eye(4, dtype=np.float64)
    translation_matrix[:3, 3] = -point

    # Combine translation and rotation to form the transformation matrix
    transformation_matrix = np.dot(rotation_matrix, translation_matrix)

    return transformation_matrix

def apply_transformation(array, transformation_matrix):
    """
    Apply a transformation to a NumPy array.

    Parameters:
    array (np.ndarray): The input array to be transformed. Each row represents a point/vector.
    transformation_matrix (np.ndarray): The transformation matrix to apply.

    Returns:
    np.ndarray: The transformed array.
    """
    if array.shape[1] + 1 != transformation_matrix.shape[0]:
        raise ValueError("The transformation matrix and array dimensions do not match.")
    
    # Add a column of ones to the input array for homogeneous coordinates
    ones = np.ones((array.shape[0], 1))
    homogeneous_array = np.hstack((array, ones))
    
    # Apply the transformation
    transformed_homogeneous_array = homogeneous_array.dot(transformation_matrix.T)
    
    # Convert back from homogeneous coordinates
    transformed_array = transformed_homogeneous_array[:, :-1] / transformed_homogeneous_array[:, -1][:, np.newaxis]
    
    return transformed_array

def evaluate_plane(plane: Tuple[np.ndarray[Literal[3], np.dtype[np.float32]], np.ndarray[Literal[3], np.dtype[np.float32]]], point: np.ndarray[Literal[3], np.dtype[np.float32]]) -> np.float32:
    centroid, normal = plane
    return np.dot(normal, point - centroid)

def approximate_intersection(plane: Tuple[np.ndarray[Literal[3], np.dtype[np.float32]], np.ndarray[Literal[3], np.dtype[np.float32]]], intrin, x, y, min_z, max_z, epsilon=1e-14):
    def deproject(x, y, z):
        return np.array(rs.rs2_deproject_pixel_to_point(intrin, [x, y], z))
    
    min_point = deproject(x, y, min_z)
    max_point = deproject(x, y, max_z)
    
    min_eval = evaluate_plane(plane, min_point)
    max_eval = evaluate_plane(plane, max_point)
    
    if min_eval * max_eval > 0:
        return np.array([0, 0, 0])
    
    while max_z - min_z > epsilon:
        mid_z = (min_z + max_z) / 2
        mid_point = deproject(x, y, mid_z)
        mid_eval = evaluate_plane(plane, mid_point)

        if mid_eval == 0:
            return mid_point

        if min_eval * mid_eval < 0:
            max_z = mid_z
            max_eval = mid_eval
        else:
            min_z = mid_z
            min_eval = mid_eval

    return deproject(x, y, (min_z + max_z) / 2)

def calculate_gravity_alignment_matrix(gravity_vector: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]]) -> np.ndarray[Tuple[Literal[3], Literal[3]], np.dtype[np.float64]]:
    """
    Create a rotation matrix to align Y-axis with the gravity vector.

    Parameters:
    gravity_vector (np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]]): Gravity vector

    Returns:
    np.ndarray[Tuple[Literal[3], Literal[3]], np.dtype[np.float64]]: Rotation matrix
    """
    y_axis = np.array([0, 1, 0], dtype=np.float64)
    gravity_vector = gravity_vector / np.linalg.norm(gravity_vector)
    rotation_axis = np.cross(y_axis, gravity_vector)
    rotation_axis /= np.linalg.norm(rotation_axis)
    cos_angle = np.dot(y_axis, gravity_vector)
    angle = np.arccos(cos_angle)
    
    q = np.quaternion(np.cos(angle / 2), *(rotation_axis * np.sin(angle / 2)))
    rotation_matrix = quaternion.as_rotation_matrix(q)

    return rotation_matrix

def marker_pattern():
    # Define the points using normalized coordinates with (0,0) as the top-left corner
    points = [
        np.array([0.2, 1.]),
        np.array([0.2, 0.]),
        np.array([0.8, 0.]),
        np.array([0.5, 0.]),
        np.array([0.5, 1.]),
        np.array([0.8, 1.]),
    ]

    return np.array(points)
