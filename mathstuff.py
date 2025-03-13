import numpy as np
import numpy.typing as npt
import quaternion
import pyrealsense2 as rs
from typing import Any, List, Literal, Optional, Tuple, cast, TypeVar
from depth_sensor.interface.stream_profile import CameraIntrinsic, CameraDistortion

from skspatial.objects import Plane
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin

# Define a generic type variable for the floating point type.
T = TypeVar('T', np.float32, np.float64)

class PlaneModel(BaseEstimator, RegressorMixin):
    def fit(self, X: npt.NDArray[T], y: npt.NDArray[T]) -> "PlaneModel":
        # Combine predictors and response into a set of 3D points.
        # Ensure that we work in the same type as X.
        points: npt.NDArray[T] = np.c_[X, y].astype(X.dtype)
        # Compute best fitting plane with skspatial.
        plane = Plane.best_fit(points, full_matrices=False)
        # Cast plane attributes to the same type.
        self.normal_ = plane.normal.astype(X.dtype)
        self.point_ = plane.point.astype(X.dtype)
        return self

    def predict(self, X: npt.NDArray[T]) -> npt.NDArray[T]:
        # Compute d = - (point . normal)
        d: T = -self.point_.dot(self.normal_)
        # Given X (with X and Y coordinates), compute Z
        prediction = (-self.normal_[0] * X[:, 0] - self.normal_[1] * X[:, 1] - d) / self.normal_[2]
        return prediction.astype(X.dtype)

def plane_from_points(
    points: npt.NDArray[T],
    min_samples: int = 30,
    residual_threshold: float = 0.01,
    max_trials: int = 10000
) -> Tuple[
    Optional[Tuple[np.ndarray, np.ndarray]],
    float,
    float,
    npt.NDArray[T]
]:
    """
    Fits a plane to a set of 3D points using a RANSAC approach with skspatial.
    
    Parameters:
        points (npt.NDArray[T]): Input 3D points of shape (N, 3) in type T.
        min_samples (int): Minimum number of points to define a plane.
        residual_threshold (float): Maximum allowed residual for a point to be an inlier.
        max_trials (int): Maximum iterations for RANSAC.
    
    Returns:
        A tuple containing:
          - Optional tuple (centroid, normal) of the plane, where each is an array of shape (3,)
            and in type T. Returns None if not enough points or an error occurs.
          - RMSE (float) of the inlier distances from the plane.
          - Maximum error (float) among the inliers.
          - Inlier points (npt.NDArray[T]) as an array of shape (M, 3).
    """
    # Prepare data: use first two coordinates as predictors (X) and third coordinate as response (y).
    X: npt.NDArray[T] = points[:, :2]
    y: npt.NDArray[T] = points[:, 2]
    
    if len(points) < min_samples:
        print(f"Only got {len(points)} points. Need at least {min_samples}.")
        return None, -1.0, -1.0, np.array([], dtype=points.dtype)

    try:
        # Fit RANSAC with our custom PlaneModel.
        ransac = RANSACRegressor(
            PlaneModel(),
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=max_trials
        )
        ransac.fit(X, y)
        
        # Get inlier mask and select inlier points.
        inlier_mask = ransac.inlier_mask_  # type: ignore
        inliers: npt.NDArray[T] = points[inlier_mask]
        
        # Use skspatial to compute a refined plane using only the inliers.
        plane = Plane.best_fit(inliers.astype(points.dtype), full_matrices=False)
        centroid: np.ndarray = plane.point.astype(points.dtype)
        normal: np.ndarray = plane.normal.astype(points.dtype)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, -1.0, -1.0, np.array([], dtype=points.dtype)
    
    # Compute error metrics:
    # The absolute distance of a point p from the plane defined by (centroid, normal)
    errors: npt.NDArray[T] = np.abs(np.dot(inliers - centroid, normal))
    rmse: float = float(np.sqrt(np.mean(errors ** 2)))
    max_error: float = float(np.max(errors))
    
    return (centroid, normal), rmse, max_error, inliers

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
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)

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

def approximate_intersection(plane: Tuple[np.ndarray[Literal[3], np.dtype[np.float32]], np.ndarray[Literal[3], np.dtype[np.float32]]], intrin, x, y, min_z, max_z, epsilon=1e-12):
    def deproject(x, y, z):
        return np.array(rs.rs2_deproject_pixel_to_point(intrin, [x, y], z))
    
    min_point = deproject(x, y, min_z)
    max_point = deproject(x, y, max_z)
    
    min_eval = evaluate_plane(plane, min_point)
    max_eval = evaluate_plane(plane, max_point)
    
    if min_eval * max_eval > 0:
        print("Plane evaluation at min and max points have the same sign")
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

from typing import TypeVar, Optional, Tuple
import numpy as np
from depth_sensor.interface.stream_profile import CameraIntrinsic, CameraDistortion

F = TypeVar("F", bound=np.float32|np.float64)
def undistort_iterative_unproject(type: np.dtype[F], intrinsic: CameraIntrinsic, pixel: np.ndarray[Literal[2], np.dtype[np.int32]]) -> Optional[np.ndarray[Literal[2], np.dtype[F]]]:
    # Cast the pixel coordinates to the floating type F
    pixel_f = pixel.astype(np.float32)  # or np.float64, depending on F

    # Normalize pixel coordinates.
    xd: F = (pixel_f[0] - intrinsic.cx) / intrinsic.fx
    yd: F = (pixel_f[1] - intrinsic.cy) / intrinsic.fy

    # Initialize with normalized coordinates.
    x: F = xd
    y: F = yd
    best_err: F = 99999  # A large initial error.

    # Use the distortion coefficients from the intrinsic parameters.
    disto: CameraDistortion = intrinsic.dist_coeffs

    # Iterative refinement (up to 40 iterations).
    for _ in range(40):
        r2: F = x * x + y * y
        r4: F = r2 * r2
        r6: F = r4 * r2
        
        # Compute the inverse radial distortion factor.
        kr_inv: F = (1 + disto.k4 * r2 + disto.k5 * r4 + disto.k6 * r6) / (1 + disto.k1 * r2 + disto.k2 * r4 + disto.k3 * r6)
        
        # Compute tangential distortion corrections.
        dx: F = disto.p1 * 2 * x * y + disto.p2 * (r2 + 2 * x * x)
        dy: F = disto.p2 * 2 * x * y + disto.p1 * (r2 + 2 * y * y)
        
        # Update the distortion-free coordinates.
        x = (xd - dx) * kr_inv
        y = (yd - dy) * kr_inv

        # Re-apply the distortion model for error evaluation.
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        a1: F = 2 * x * y
        a2: F = r2 + 2 * x * x
        a3: F = r2 + 2 * y * y
        
        cdist: F = (1 + disto.k1 * r2 + disto.k2 * r4 + disto.k3 * r6) / (1 + disto.k4 * r2 + disto.k5 * r4 + disto.k6 * r6)
        xd0: F = x * cdist + disto.p1 * a1 + disto.p2 * a2
        yd0: F = y * cdist + disto.p1 * a3 + disto.p2 * a1
        
        # Re-project to pixel coordinates.
        x_proj: F = xd0 * intrinsic.fx + intrinsic.cx
        y_proj: F = yd0 * intrinsic.fy + intrinsic.cy
        
        error: F = np.sqrt((x_proj - pixel_f[0])**2 + (y_proj - pixel_f[1])**2)
        
        if error > best_err:
            break
        
        best_err = error
        
        if error < 0.01:
            break

    # If the best error is too high, consider the result as invalid.
    if best_err > 0.5:
        return None

    return np.array([x, y], dtype=pixel_f.dtype)

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
