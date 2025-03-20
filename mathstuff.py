import numpy as np
import numpy.typing as npt
import quaternion
import pyrealsense2 as rs
from typing import Any, List, Literal, Optional, Tuple, cast, TypeVar
from depth_sensor.interface.stream_profile import CameraIntrinsic, CameraDistortion, DistortionModel, Extrinsic
from depth_sensor.interface.frame import DepthFrame

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

def compute_xy_transformation_matrix(plane: Tuple[np.ndarray[Literal[3], np.float32], np.ndarray[Literal[3], np.float32]]) -> np.ndarray[Literal[4, 4], np.float32]:
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
    rotation_matrix = np.eye(4, dtype=np.float32)
    rotation_matrix[:3, :3] = np.array([x, y, z], dtype=np.float32)

    # Construct the translation matrix (column-major)
    translation_matrix = np.eye(4, dtype=np.float32)
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

def approximate_intersection(plane: Tuple[np.ndarray[Literal[3], np.dtype[np.float32]], np.ndarray[Literal[3], np.dtype[np.float32]]], intrin: CameraIntrinsic, x: np.float32, y: np.float32, min_z: np.float32, max_z: np.float32, epsilon: np.float32=np.float32(1e-10)):
    def deproject(x: np.float32, y: np.float32, d: np.float32) -> Optional[np.ndarray[Literal[3], np.dtype[np.float32]]]:
        pt = undistort_deproject(np.dtype(np.float32), intrin, np.array([x, y]))
        if pt is None:
            return None
        return np.array([pt[0], pt[1], 1], dtype=np.float32) * d

    min_point = deproject(x, y, min_z)
    max_point = deproject(x, y, max_z)

    if min_point is None:
        return np.array([0, 0, 0])

    if max_point is None:
        return np.array([0, 0, 0])

    min_eval = evaluate_plane(plane, min_point)
    max_eval = evaluate_plane(plane, max_point)

    if min_eval * max_eval > 0:
        print("Plane evaluation at min and max points have the same sign")
        return np.array([0, 0, 0])

    while max_z - min_z > epsilon:
        mid_z = (min_z + max_z) / 2
        mid_point = deproject(x, y, mid_z)
        if mid_point is None:
            return np.array([0, 0, 0])
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

def calculate_gravity_alignment_matrix(gravity_vector: np.ndarray[Tuple[Literal[3]], np.dtype[np.float32]]) -> np.ndarray[Tuple[Literal[3], Literal[3]], np.dtype[np.float32]]:
    """
    Create a rotation matrix to align Y-axis with the gravity vector.

    Parameters:
    gravity_vector (np.ndarray[Tuple[Literal[3]], np.dtype[np.float32]]): Gravity vector

    Returns:
    np.ndarray[Tuple[Literal[3], Literal[3]], np.dtype[np.float32]]: Rotation matrix
    """
    y_axis = np.array([0, 1, 0], dtype=np.float32)
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
def undistort_deproject(
    _: np.dtype[F], intrinsic: CameraIntrinsic, pixel: np.ndarray[Literal[2], np.dtype[F]]
) -> Optional[np.ndarray[Literal[2], np.dtype[F]]]:
    """
    Undistorts a pixel coordinate and deprojects it to normalized (x, y) camera space.
    
    - Uses **direct** computation for Inverse Brown-Conrady.
    - Uses **iterative** refinement for Brown-Conrady.
    
    Returns normalized (x, y) coordinates if successful, otherwise None.
    """

    # Normalize pixel coordinates
    xd: F = (pixel[0] - intrinsic.cx) / intrinsic.fx
    yd: F = (pixel[1] - intrinsic.cy) / intrinsic.fy

    disto: CameraDistortion = intrinsic.dist_coeffs

    if intrinsic.model == DistortionModel.INV_BROWN_CONRADY:
        # **Inverse Brown-Conrady (RealSense)**
        x: F = xd
        y: F = yd
        xo: F = xd
        yo: F = yd

        # Iterative refinement (up to 40 iterations)
        for _ in range(40):
            r2: F = x * x + y * y
            icdist: F = 1.0 / (1.0 + ((disto.k5 * r2 + disto.k2) * r2 + disto.k1) * r2)

            xq: F = x / icdist
            yq: F = y / icdist

            delta_x: F = 2 * disto.p1 * xq * yq + disto.p2 * (r2 + 2 * xq * xq)
            delta_y: F = 2 * disto.p2 * xq * yq + disto.p1 * (r2 + 2 * yq * yq)

            x = (xo - delta_x) * icdist
            y = (yo - delta_y) * icdist

        return np.array([x, y], dtype=pixel.dtype)

    elif intrinsic.model == DistortionModel.BROWN_CONRADY:
        # **Brown-Conrady (Exact Math from Provided C++ Code)**
        x: F = xd
        y: F = yd
        best_err: F = cast(F, np.float32(99999)) # Large initial error

        for _ in range(20):  # Fixed at 20 iterations
            r2: F = x * x + y * y
            r4: F = r2 * r2
            r6: F = r4 * r2

            kr_inv: F = (1 + disto.k4 * r2 + disto.k5 * r4 + disto.k6 * r6) / (1 + disto.k1 * r2 + disto.k2 * r4 + disto.k3 * r6)
            dx: F = disto.p1 * 2 * x * y + disto.p2 * (r2 + 2 * x * x)
            dy: F = disto.p2 * 2 * x * y + disto.p1 * (r2 + 2 * y * y)

            x = (xd - dx) * kr_inv
            y = (yd - dy) * kr_inv

            # Compute projection error to ensure convergence
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2

            a1 = 2 * x * y
            a2 = r2 + 2 * x * x
            a3 = r2 + 2 * y * y

            cdist = (1 + disto.k1 * r2 + disto.k2 * r4 + disto.k3 * r6) / (1 + disto.k4 * r2 + disto.k5 * r4 + disto.k6 * r6)
            xd0 = x * cdist + disto.p1 * a1 + disto.p2 * a2
            yd0 = y * cdist + disto.p1 * a3 + disto.p2 * a1

            x_proj = xd0 * intrinsic.fx + intrinsic.cx
            y_proj = yd0 * intrinsic.fy + intrinsic.cy
            error = np.sqrt((x_proj - pixel[0]) ** 2 + (y_proj - pixel[1]) ** 2)

            if error > best_err:
                break

            best_err = error

            if error < 0.01:
                break

        return np.array([x, y], dtype=pixel.dtype)

    else:
        raise ValueError("Unsupported distortion model")

def transform_point(
    point: np.ndarray[Literal[3], np.dtype[np.float32]],
    extrinsics: Extrinsic
) -> np.ndarray[Literal[3], np.dtype[np.float32]]:
    """
    Applies an extrinsic transformation (rotation and translation) to a 3D point.
    """
    return extrinsics.rot @ point + extrinsics.translation

def project_point_to_pixel(
    intrinsic: CameraIntrinsic,
    point: np.ndarray[Literal[3], np.dtype[np.float32]]
) -> np.ndarray[Literal[2], np.dtype[np.float32]]:
    """
    Projects a 3D point in camera space to a 2D image pixel using intrinsic parameters.
    """
    x, y, z = point
    if z == 0:
        return np.array([-1, -1], dtype=np.float32)  # Invalid projection

    px = intrinsic.fx * (x / z) + intrinsic.cx
    py = intrinsic.fy * (y / z) + intrinsic.cy

    return np.array([px, py], dtype=np.float32)

def project_point_to_pixel_with_distortion(
    intrinsic: CameraIntrinsic,
    point: np.ndarray[Literal[3], np.dtype[F]]
) -> np.ndarray[Literal[2], np.dtype[F]]:
    """
    Projects a 3D point in camera space to a 2D image pixel using intrinsic parameters 
    and manually applies distortion.
    """
    x, y, z = point
    if z == 0:
        return np.array([-1, -1], dtype=point.dtype)  # Invalid projection

    # Normalize 3D coordinates
    xn: F = x / z
    yn: F = y / z

    disto: CameraDistortion = intrinsic.dist_coeffs

    r2: F = xn * xn + yn * yn
    r4: F = r2 * r2
    r6: F = r4 * r2

    if intrinsic.model == DistortionModel.BROWN_CONRADY:
        # **Brown-Conrady distortion**
        radial: F = (1 + disto.k1 * r2 + disto.k2 * r4 + disto.k3 * r6) / (1 + disto.k4 * r2 + disto.k5 * r4 + disto.k6 * r6)
        tangential_x: F = 2 * disto.p1 * xn * yn + disto.p2 * (r2 + 2 * xn * xn)
        tangential_y: F = disto.p1 * (r2 + 2 * yn * yn) + 2 * disto.p2 * xn * yn

        xd: F = xn * radial + tangential_x
        yd: F = yn * radial + tangential_y

    elif intrinsic.model == DistortionModel.INV_BROWN_CONRADY:
        # **Inverse Brown-Conrady distortion**
        icdist: F = 1.0 / (1.0 + ((disto.k5 * r2 + disto.k2) * r2 + disto.k1) * r2)
        tangential_x: F = 2 * disto.p1 * xn * yn + disto.p2 * (r2 + 2 * xn * xn)
        tangential_y: F = 2 * disto.p2 * xn * yn + disto.p1 * (r2 + 2 * yn * yn)

        xd: F = (xn + tangential_x) * icdist
        yd: F = (yn + tangential_y) * icdist

    else:
        raise ValueError("Unsupported distortion model")

    # Convert back to pixel space
    px: F = intrinsic.fx * xd + intrinsic.cx
    py: F = intrinsic.fy * yd + intrinsic.cy

    return np.array([px, py], dtype=point.dtype)

# Warning: this truncates for depth so it is imprecise
def project_color_pixel_to_depth_pixel(
    color_pixel: np.ndarray[Literal[2], np.dtype[np.float32]],
    depth_frame: DepthFrame, 
    depth_intrinsic: CameraIntrinsic, 
    color_intrinsic: CameraIntrinsic, 
    color_to_depth: Extrinsic
) -> Optional[np.ndarray[Literal[2], np.dtype[np.int32]]]:
    """
    Projects a color pixel into depth space by searching along the epipolar line.
    """

    min_depth = 0.1  # Minimum valid depth in meters
    max_depth = 5.0  # Maximum reasonable depth in meters

    # Deproject color pixel at min/max depth
    min_point = undistort_deproject(np.dtype(np.float32), color_intrinsic, color_pixel)
    max_point = min_point

    if min_point is None or max_point is None:
        return None

    min_point = np.array([min_point[0], min_point[1], 1.0]) * min_depth
    max_point = np.array([max_point[0], max_point[1], 1.0]) * max_depth

    # Transform to depth space
    min_transformed = transform_point(min_point, color_to_depth)
    max_transformed = transform_point(max_point, color_to_depth)

    # Project to depth image
    min_depth_pixel = project_point_to_pixel(depth_intrinsic, min_transformed)
    max_depth_pixel = project_point_to_pixel(depth_intrinsic, max_transformed)

    if np.any(min_depth_pixel == -1) or np.any(max_depth_pixel == -1):
        return None

    # Search along the line for closest valid depth pixel
    best_pixel = None
    min_dist = float('inf')

    line_pixels = np.linspace(min_depth_pixel, max_depth_pixel, num=50, dtype=np.dtype(np.float32))

    for pixel in line_pixels:
        px, py = pixel
        if px < 0 or py < 0 or px >= depth_intrinsic.width or py >= depth_intrinsic.height:
            continue

        depth_value = depth_frame.get_distance(int(px), int(py)) # Get depth in meters
        if depth_value <= 0:
            continue

        # Reproject to color space
        point_3d = undistort_deproject(np.dtype(np.float32), depth_intrinsic, np.array([px, py]))
        if point_3d is None:
            return None
        point_3d = np.array([point_3d[0], point_3d[1], 1.0], np.float32)
        point_3d *= depth_value

        transformed_point = transform_point(point_3d, color_to_depth.inv())
        reprojected_pixel = project_point_to_pixel_with_distortion(color_intrinsic, transformed_point)

        dist = np.linalg.norm(reprojected_pixel - color_pixel)
        if dist < min_dist:
            min_dist = dist
            best_pixel = pixel

    if best_pixel is None:
        return None
    else:
        return np.array([int(best_pixel[0]), int(best_pixel[1])])

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
