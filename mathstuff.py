import numpy as np
import pyrealsense2 as rs
from typing import List

def plane_from_points(points):
    # Ensure points is a numpy array
    points = np.array(points)
    
    # Calculate the centroid
    centroid = np.mean(points, axis=0)
    
    # Center the points around the centroid
    centered_points = points - centroid
    
    # Calculate the covariance matrix
    covariance_matrix = np.cov(centered_points, rowvar=False)
    
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # Find the smallest eigenvalue and its corresponding eigenvector
    smallest_eigenvalue_index = np.argmin(eigenvalues)
    plane_normal = eigenvectors[:, smallest_eigenvalue_index]
    
    return centroid, plane_normal

def compute_transformation_matrix(plane: np.ndarray) -> np.ndarray:
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

def apply_transformation(points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points = points_homogeneous.dot(transformation.T)
    return transformed_points[:, :3]

def evaluate_plane(plane, point):
    centroid, normal = plane
    return np.dot(normal, point - centroid)

def approximate_intersection(plane, intrin, x, y, min_z, max_z, epsilon=1e-3):
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
