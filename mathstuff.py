import numpy as np
import numpy.typing as npt
import pyrealsense2 as rs
from typing import Any, List, Literal, Tuple, cast
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

def plane_from_points(points: npt.NDArray[np.float32]) -> Tuple[np.ndarray[Literal[3], np.dtype[np.float32]], np.ndarray[Literal[3], np.dtype[np.float32]]]:
    # Prepare data for RANSAC
    X = points[:, :2]  # X and Y coordinates
    y = points[:, 2]   # Z coordinate

    # Fit RANSAC regressor with PlaneModel
    ransac = RANSACRegressor(
        PlaneModel(), 
        min_samples=15,  # Minimum number of samples to define a plane
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

def compute_transformation_matrix(plane: Tuple[np.ndarray[Literal[3], np.dtype[np.float32]], np.ndarray[Literal[3], np.dtype[np.float32]]]) -> np.ndarray[Literal[4, 4], np.dtype[np.float64]]:
    normal = plane[1]
    d = plane[0]
    z_axis = normal / np.linalg.norm(normal)
    x_axis = np.cross(np.array([0, 1, 0]), z_axis)
    if np.linalg.norm(x_axis) == 0:
        x_axis = np.cross(np.array([1, 0, 0]), z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    rotation_matrix = np.vstack([x_axis, y_axis, z_axis])
    translation = -np.dot(rotation_matrix, normal * d / np.dot(normal, normal))
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation

    return transformation_matrix

def apply_transformation(points: np.ndarray[Any, np.dtype[np.float32]], transformation_matrix: np.ndarray[Literal[4, 4], np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float32]]:
    points = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float32)])
    points = np.dot(transformation_matrix, points.T).T
    return points[:, :3]

def evaluate_plane(plane: Tuple[np.ndarray[Literal[3], np.dtype[np.float32]], np.ndarray[Literal[3], np.dtype[np.float32]]], point: np.ndarray[Literal[3], np.dtype[np.float32]]) -> np.float32:
    centroid, normal = plane
    return np.dot(normal, point - centroid)

def approximate_intersection(plane: Tuple[np.ndarray[Literal[3], np.dtype[np.float32]], np.ndarray[Literal[3], np.dtype[np.float32]]], intrin, x, y, min_z, max_z, epsilon=1e-3):
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
