import numpy as np
import numpy.typing as npt
import cv2
import pyrealsense2 as rs
from typing import Any, List, Dict, Literal, Tuple, Callable, Optional, cast
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QScreen
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QStackedLayout
from platformdirs import user_data_dir
import io
import json
import time
import quaternion
from ahrs.filters import Madgwick
from data_acquisition import DataAcquisitionThread
from mathstuff import plane_from_points, compute_xy_transformation_matrix, apply_transformation, evaluate_plane, approximate_intersection, calculate_gravity_alignment_matrix, marker_pattern
from calibration_data import CalibrationData

import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib import pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

def define_charuco_board_2d_points(board_size: Tuple[int, int], square_length: float) -> Dict[int, np.ndarray[Literal[2], np.dtype[np.float32]]]:
    points = dict[int, np.ndarray[Literal[2], np.dtype[np.float32]]]()
    id = 0
    for y in range(board_size[1]-1):
        for x in range(board_size[0]-1):
            points[id] = np.array([x * square_length, y * square_length])
            id += 1
    return points

# -1,-1 for unaligned corners
def align_corners_to_depth(charuco_corners: np.ndarray, depth_frame, raw_color_frame, depth_sensor):
    # Adapted from https://github.com/IntelRealSense/librealsense/issues/5603#issuecomment-574019008
    depth_scale = depth_sensor.get_depth_scale()
    # idk what values make sense
    depth_min = 0.11 #meter
    depth_max = 8.0 #meter

    depth_vsp = rs.video_stream_profile(depth_frame.profile)
    color_vsp = rs.video_stream_profile(raw_color_frame.profile)
    depth_intrin = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
    color_intrin = rs.video_stream_profile(raw_color_frame.profile).get_intrinsics()

    depth_to_color_extrin =  depth_vsp.get_extrinsics_to(raw_color_frame.profile)
    color_to_depth_extrin =  color_vsp.get_extrinsics_to(depth_frame.profile)

    aligned_corners = np.full_like(charuco_corners, np.nan)
    for i, color_point in enumerate(charuco_corners):
       aligned_corners[i] = rs.rs2_project_color_pixel_to_depth_pixel(
                    depth_frame.get_data(), depth_scale,
                    depth_min, depth_max,
                    depth_intrin, color_intrin, color_to_depth_extrin, depth_to_color_extrin, color_point.ravel())
    # depth_scale = depth_sensor.get_depth_scale()
    # depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    # color_intrin = raw_color_frame.profile.as_video_stream_profile().intrinsics
    # depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(raw_color_frame.profile)
    return aligned_corners

def extract_3d_point(charuco_corner: np.ndarray, depth_frame: rs.depth_frame) -> Optional[np.ndarray[Literal[3], np.dtype[np.float32]]]:
    depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
    u, v = charuco_corner.ravel()
    if u < 0 or v < 0:
        return None
    depth = depth_frame.get_distance(int(u), int(v))
    return np.array(rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth), dtype=np.float32)

def extract_3d_points(charuco_corners: np.ndarray[Any, np.dtype[Any]], depth_frame: rs.depth_frame) -> List[np.ndarray[Literal[3], np.dtype[np.float32]]]:
    points_3d = list[np.ndarray[Literal[3], np.dtype[np.float32]]]()

    corner: np.ndarray[Literal[2], np.dtype[np.float32]]
    for corner in charuco_corners:
        point_3d = extract_3d_point(corner, depth_frame)
        points_3d.append(point_3d)

    return points_3d

class Page3(QWidget):
    def __init__(self, parent: QWidget, next_page: Callable[[], None], exit_application: Callable[[], None], pipeline: Optional[rs.pipeline], auto_progress: bool, calibration_data: CalibrationData, screen: QScreen) -> None:
        super().__init__(parent)
        self.screen_size = screen.size()
        self.exit_application = exit_application
        self.motion_support = False
        self.pipeline = pipeline
        self.pipeline_profile = None
        self.hdr_merge: Optional[rs.hdr_merge] = None  # Placeholder for the HDR merge processing block
        self.temporal_filter: Optional[rs.temporal_filter] = None  # Placeholder for the temporal filter
        self.spatial_filter: Optional[rs.spatial_filter] = None  # Placeholder for the spatial filter
        self.align: Optional[rs.align] = None  # Placeholder for the align processing block
        self.auto_progress = auto_progress
        self.go_countdown_time = 3
        self.next_countdown_time = 5
        self.next_timer: Optional[QTimer] = None
        self.countdown_timer: Optional[QTimer] = None
        self.latest_raw_color_frame: Optional[rs.frame] = None
        self.latest_color_frame: Optional[rs.frame] = None
        self.latest_depth_frame: Optional[rs.frame] = None
        self.madgwick = Madgwick(gain=0.5)  # Initialize Madgwick filter
        self.Q = np.array([1.0, 0.0, 0.0, 0.0])  # Initial quaternion
        self.data_thread: Optional[DataAcquisitionThread] = None
        self.cols = 16
        self.rows = 9
        self.next_page = next_page
        self.calibration_data = calibration_data
        self.init_ui()

    def init_ui(self) -> None:
        self.stacked_layout = QStackedLayout(self)

        # Layout for the initial state with "Go" button
        self.initial_layout = QVBoxLayout()
        self.initial_widget = QWidget()
        self.initial_widget.setLayout(self.initial_layout)

        self.instructions_label = QLabel("Press 'Go' to display the ChArUco board. It will disappear after 3 seconds.")
        self.instructions_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.initial_layout.addWidget(self.instructions_label)

        self.go_button = QPushButton("Go")
        self.go_button.clicked.connect(self.on_go_clicked)
        self.initial_layout.addWidget(self.go_button)

        self.next_button = QPushButton("Next")
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self.go_next)
        self.initial_layout.addWidget(self.next_button)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.exit_application)
        self.initial_layout.addWidget(self.exit_button)

        self.stacked_layout.addWidget(self.initial_widget)

        # Layout for displaying the ChArUco board
        self.charuco_layout = QVBoxLayout()
        self.charuco_layout.setContentsMargins(0, 0, 0, 0)
        self.charuco_widget = QWidget()
        self.charuco_widget.setLayout(self.charuco_layout)

        self.label = QLabel()
        # self.charuco_layout.addWidget(self.label)

        self.stacked_layout.addWidget(self.charuco_widget)

        self.stacked_layout.setCurrentWidget(self.initial_widget)

    def start(self) -> None:
        if self.auto_progress:
            self.start_go_countdown()
        self.start_data_acquisition()

    def start_go_countdown(self) -> None:
        """Start the countdown for the initial 'Go'."""
        self.go_timer = QTimer(self)
        self.go_timer.timeout.connect(self.update_go_countdown)
        self.go_timer.start(1000)

    def update_go_countdown(self) -> None:
        """Update the 'Going in t' countdown."""
        self.go_countdown_time -= 1
        self.instructions_label.setText(f"Going in {self.go_countdown_time}")
        if self.go_countdown_time <= 0:
            self.go_timer.stop()
            self.on_go_clicked()

    def start_next_countdown(self, success: bool) -> None:
        """Start the countdown after 'Done' or 'Autocalibration unsuccessful'."""
        self.next_timer = QTimer(self)
        self.next_timer.timeout.connect(lambda: self.update_next_countdown(success))
        self.next_timer.start(1000)

    def update_next_countdown(self, success: bool) -> None:
        """Update the countdown based on whether the calibration was successful or not."""
        self.next_countdown_time -= 1
        if success:
            self.instructions_label.setText(f"Next in {self.next_countdown_time}")
        else:
            self.instructions_label.setText(f"Autocalibration unsuccessful. Exiting in {self.next_countdown_time}")

        if self.next_countdown_time <= 0:
            if success:
                self.go_next()
            else:
                if self.next_timer is not None:
                    self.next_timer.stop()
                self.exit_application()
    
    def go_next(self) -> None:
        if self.next_timer is not None:
            self.next_timer.stop()
        self.next_page()

    def resizeEvent(self, event: Any) -> None:
        self.create_charuco_board()

    def on_go_clicked(self) -> None:
        self.stacked_layout.setCurrentWidget(self.charuco_widget)
        self.label.showFullScreen()
        QTimer.singleShot(3000, self.detect_charuco_corners)  # type: ignore

    def create_charuco_board(self) -> None:
        # Create the ChArUco board
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        board = cv2.aruco.CharucoBoard((self.cols, self.rows), 0.04, 0.03, aruco_dict)

        # Generate the ChArUco board image
        board_image = board.generateImage(self.screen_size.toTuple(), marginSize=0, borderBits=1)

        height, width = board_image.shape[:2]
        bytes_per_line = width
        qt_image = QImage(board_image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qt_image)

        self.label.setPixmap(pixmap)

    def fail(self, msg: str) -> None:
        print(f"Detection failure: {msg}")
        self.instructions_label.setText(f"Autocalibration unsuccessful. Exiting in {self.next_countdown_time}")
        if self.auto_progress:
            self.start_next_countdown(success=False)
        self.label.hide()
        self.stacked_layout.setCurrentWidget(self.initial_widget)

    def detect_charuco_corners(self) -> None:
        if not self.latest_color_frame or not self.latest_depth_frame:
            self.fail("No frames available for ChArUco board detection.")
            return

        raw_color_frame = self.latest_raw_color_frame
        color_frame = self.latest_color_frame
        depth_frame = self.latest_depth_frame
        depth_sensor = self.pipeline_profile.get_device().first_depth_sensor()

        raw_color_image = np.asanyarray(raw_color_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # ------------------------------
        # Step 1. Detect the ChArUco board corners on the raw RGB image.
        # ------------------------------
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        board = cv2.aruco.CharucoBoard((self.cols, self.rows), 0.04, 0.03, aruco_dict)

        detector_parameters = cv2.aruco.DetectorParameters()
        charuco_parameters = cv2.aruco.CharucoParameters()
        charuco_detector = cv2.aruco.CharucoDetector(board, charuco_parameters, detector_parameters)

        gray = cv2.cvtColor(raw_color_image, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
        if charuco_ids is None:
            self.fail("No ChArUco board detected.")
            return

        # For visualization, draw the detected corners on the color image.
        cv2.aruco.drawDetectedCornersCharuco(color_image, charuco_corners, charuco_ids, cornerColor=(0, 0, 255))
        self.calibration_data.color_image = color_image

        # ------------------------------
        # Step 2. Compute the expected board points and normalize them based on screen dimensions.
        # ------------------------------
        # Assume this function returns a dict mapping marker IDs to 2D board coordinates.
        charuco_board_2d_points = define_charuco_board_2d_points((self.cols, self.rows), 1)

        # Calculate scaling factors based on screen dimensions.
        screen_width = self.screen_size.width()
        screen_height = self.screen_size.height()

        board_aspect_ratio = self.cols / self.rows
        screen_aspect_ratio = screen_width / screen_height

        board_width_in_pixels = screen_height * board_aspect_ratio
        board_height_in_pixels = screen_width / board_aspect_ratio

        if screen_aspect_ratio < board_aspect_ratio:
            scale_x = 0
            scale_y = (screen_height - board_height_in_pixels) / (board_height_in_pixels / self.rows)
        else:
            scale_x = (screen_width - board_width_in_pixels) / (board_width_in_pixels / self.cols)
            scale_y = 0

        # Normalize the expected board points to the unit square.
        normalized_expected_board_points = {
            id_: [
                (1 + point[0] + scale_x / 2) / (self.cols + scale_x),
                (1 + point[1] + scale_y / 2) / (self.rows + scale_y)
            ]
            for id_, point in charuco_board_2d_points.items()
        }

        # Build arrays for computing the initial homography from detected points to expected board coordinates.
        detected_points = []
        expected_points = []
        for i, id_ in enumerate(charuco_ids):
            if id_[0] in normalized_expected_board_points:
                if np.any(charuco_corners[i] == -1):
                    continue
                detected_points.append(charuco_corners[i][0])
                expected_points.append(normalized_expected_board_points[id_[0]])
        detected_points = np.array(detected_points, dtype=np.float32)
        expected_points = np.array(expected_points, dtype=np.float32)
        if len(detected_points) < 16:
            self.fail("Insufficient detected points.")
            return
        h_board, _ = cv2.findHomography(detected_points, expected_points)
        if h_board is None:
            self.fail("Homography computation failed.")
            return
        
        # ------------------------------
        # Step 3. Define the board region in the raw RGB image.
        # ------------------------------
        board_polygon_board = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ], dtype=np.float32)
        h_board_inv = np.linalg.inv(h_board)
        board_polygon_raw = cv2.perspectiveTransform(board_polygon_board.reshape(-1, 1, 2), h_board_inv)
        board_polygon_raw = board_polygon_raw.astype(np.int32)

        # Create a mask for the board region in the raw_color_image.
        board_mask = np.zeros(raw_color_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(board_mask, [board_polygon_raw.reshape(-1, 2)], 255)

        # self.fail("debugging")
        # # plot draw the color image and the mask
        # plt.figure()
        # plt.imshow(raw_color_image)
        # plt.plot(board_polygon_raw[:, 0, 0], board_polygon_raw[:, 0, 1], 'r-')
        # plt.show()

        # ------------------------------
        # Step 4. Generate a grid of points over the board region and align them to the depth frame.
        # ------------------------------
        ys, xs = np.where(board_mask > 0)
        grid_points = np.vstack((xs, ys)).T.astype(np.float32)

        # Use align_corners_to_depth on the grid points.
        aligned_grid_points = align_corners_to_depth(grid_points, depth_frame, raw_color_frame, depth_sensor)
        # Filter out points that did not map correctly (marked as (-1, -1)).
        valid_mask = (aligned_grid_points[:, 0] != -1) & (aligned_grid_points[:, 1] != -1)
        valid_aligned_points = aligned_grid_points[valid_mask].astype(np.int32)

        # ------------------------------
        # Step 5. Gather 3D points from all valid depth pixels within the board region.
        # ------------------------------
        points_3d = []
        for pt in valid_aligned_points:
            x_d, y_d = pt

            point_3d = extract_3d_point(pt, depth_frame)
            points_3d.append(np.array(point_3d, dtype=np.float32))
        points_3d = np.array(points_3d)
        if len(points_3d) == 0:
            self.fail("No valid depth points found in board region.")
            return

        # Fit a plane using the custom plane_from_points function.
        (plane, plane_rmse, plane_max_error, inlier_points) = plane_from_points(points_3d)
        if plane is None:
            self.fail("Plane fitting failed")
            return
        self.calibration_data.plane = plane
        self.calibration_data.plane_rmse = plane_rmse
        self.calibration_data.plane_max_error = plane_max_error

        print("RMSE of plane fit: ", plane_rmse)
        print("Max error of plane fit: ", plane_max_error)

        # ------------------------------
        # Step 6. Align the fitted plane with gravity.
        # ------------------------------
        Q = np.array(self.Q)
        if self.motion_support:
            swap_yz = np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0],
            ])
            self.calibration_data.align_transform_mtx = (
                np.linalg.inv(swap_yz)
                @ quaternion.as_rotation_matrix(np.quaternion(*Q))
                @ swap_yz
            )
            align_transform_inv_mtx = np.linalg.inv(self.calibration_data.align_transform_mtx)
        else:
            self.calibration_data.align_transform_mtx = np.eye(3, dtype=np.float64)
            align_transform_inv_mtx = np.eye(3, dtype=np.float64)

        # Align the 3D points with gravity.
        self.calibration_data.points_3d_aligned = [
            self.calibration_data.align_transform_mtx @ point for point in inlier_points
        ]
        self.calibration_data.intrin = rs.video_stream_profile(depth_frame.profile).get_intrinsics()

        # # for debugging
        # self.fail("debugging")
        # # display the point cloud of the board region
        # plt.figure()
        # ax = plt.axes(projection='3d')

        # # for i, point in enumerate(inlier_points):
        # for i, point in enumerate(points_3d):
        #     if i % 100 == 0:
        #         ax.scatter(point[0], point[1], point[2], c='b', marker='x', s=0.5)

        # centroid, normal = plane
        # d = -np.dot(centroid, normal)
        # xx, yy = np.meshgrid(np.linspace(-2, 2, 2), np.linspace(-2, 2, 2))
        # zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
        # ax.plot_surface(xx, yy, zz, alpha=0.5)
        # plt.show()
        # return

        # ------------------------------
        # Step 7. Recompute a homography using deprojected board corners.
        # ------------------------------
        # For each detected board corner (from aligned charuco detection), approximate its intersection with the plane.
        deprojected_points = list[np.ndarray[Literal[3], np.dtype[np.float32]]]()
        detected_points_aligned_to_depth = align_corners_to_depth(detected_points, depth_frame, raw_color_frame, depth_sensor)
        for pt in detected_points_aligned_to_depth:
            deproj_pt = approximate_intersection(plane, self.calibration_data.intrin, pt[0], pt[1], 0, 1000)
            if np.all(deproj_pt == 0):
                continue
            deprojected_points.append(deproj_pt)

        deprojected_points_aligned = [
            self.calibration_data.align_transform_mtx @ point for point in deprojected_points
        ]
        deprojected_points_aligned = np.array(deprojected_points_aligned, dtype=np.float32)

        # Rotate the plane to align with gravity for computing a 2D transformation.
        plane_aligned = (
            self.calibration_data.align_transform_mtx @ self.calibration_data.plane[0],
            self.calibration_data.align_transform_mtx @ self.calibration_data.plane[1]
        )
        if len(np.unique(deprojected_points_aligned, axis=0)) <= 1:
            self.fail("Deprojected points are not distinct.")
            return

        # Compute transformation matrix to flatten the plane to the XY plane.
        self.calibration_data.xy_transformation_matrix_aligned = compute_xy_transformation_matrix(plane_aligned)
        print("Transformation matrix (aligned): ", self.calibration_data.xy_transformation_matrix_aligned)

        # Apply the transformation to the deprojected points.
        transformed_points = apply_transformation(
            deprojected_points_aligned,
            self.calibration_data.xy_transformation_matrix_aligned
        )

        # todo tomorrow the transformed points wont always have the same shape because we remove invalids
        # print(expected_points)
        # print(transformed_points[:, :2])

        # Compute the homography mapping expected board points to transformed deprojected points.
        self.calibration_data.h_aligned, _ = cv2.findHomography(expected_points, transformed_points[:, :2])
        if self.calibration_data.h_aligned is None:
            print("Error: Homography could not be computed.")
            return

        print("Homography (aligned): ", self.calibration_data.h_aligned)

        # ------------------------------
        # Step 8. Map unit square corners and adjust offsets.
        # ------------------------------
        normalized_corners = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ], dtype=np.float32)
        transformed_unit_square_2d_aligned = cv2.perspectiveTransform(
            normalized_corners.reshape(-1, 1, 2),
            self.calibration_data.h_aligned
        )
        offset_mtx = np.array([
            [1, 0, -transformed_unit_square_2d_aligned[0][0][0]],
            [0, 1, -transformed_unit_square_2d_aligned[0][0][1]],
            [0, 0, 1]
        ], dtype=np.float64)
        offset_mtx_3d = np.array([
            [1, 0, 0, -transformed_unit_square_2d_aligned[0][0][0]],
            [0, 1, 0, -transformed_unit_square_2d_aligned[0][0][1]],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        self.calibration_data.h_aligned = offset_mtx @ self.calibration_data.h_aligned
        self.calibration_data.xy_transformation_matrix_aligned = offset_mtx_3d @ self.calibration_data.xy_transformation_matrix_aligned
        transformed_unit_square_2d_aligned = cv2.perspectiveTransform(
            normalized_corners.reshape(-1, 1, 2),
            self.calibration_data.h_aligned
        )

        # ------------------------------
        # Step 9. Compute the final 3D quadrilateral.
        # ------------------------------
        transformed_unit_square_3d_aligned = transformed_unit_square_2d_aligned.reshape(-1, 2)
        transformed_unit_square_3d_aligned = np.hstack([
            transformed_unit_square_3d_aligned,
            np.zeros((4, 1), dtype=np.float32)
        ])
        self.calibration_data.best_quad_aligned = apply_transformation(
            transformed_unit_square_3d_aligned,
            np.linalg.inv(self.calibration_data.xy_transformation_matrix_aligned)
        )
        self.calibration_data.best_quad = [
            align_transform_inv_mtx @ point for point in self.calibration_data.best_quad_aligned
        ]
        self.calibration_data.best_quad_2d = []
        for point in self.calibration_data.best_quad:
            point_2d = rs.rs2_project_point_to_pixel(self.calibration_data.intrin, point)
            self.calibration_data.best_quad_2d.append(point_2d)
        self.calibration_data.best_quad_2d = np.array(self.calibration_data.best_quad_2d, dtype=np.int32)

        # ------------------------------
        # Step 10. Finalize: update UI and proceed.
        # ------------------------------
        self.instructions_label.setText(f"Next in {self.next_countdown_time}")
        self.next_button.setEnabled(True)
        if self.auto_progress:
            self.start_next_countdown(success=True)
        self.label.hide()
        self.stacked_layout.setCurrentWidget(self.initial_widget)

    def process_frame(self, frames: rs.frame) -> None:
        frames = self.hdr_merge.process(frames)
        frames = self.temporal_filter.process(frames)
        frames = self.spatial_filter.process(frames).as_frameset()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if self.motion_support:
            accel_frame = frames.first_or_default(rs.stream.accel)
            gyro_frame = frames.first_or_default(rs.stream.gyro)

        if self.motion_support:
            if not color_frame or not accel_frame or not gyro_frame:
                return
        else:
            if not color_frame:
                return

        self.latest_raw_color_frame = frames.get_color_frame()
        self.latest_color_frame = color_frame
        self.latest_depth_frame = depth_frame

        if self.motion_support:
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()

            # Update Madgwick filter
            # Swap coordinates because the Madgwick expects z to be gravity
            self.Q = self.madgwick.updateIMU(self.Q, gyr=[gyro_data.x, gyro_data.z, -gyro_data.y], acc=[accel_data.x, accel_data.z, -accel_data.y])

    def start_data_acquisition(self) -> None:
        if self.pipeline:
            self.data_thread = DataAcquisitionThread(self.pipeline)
            self.data_thread.data_updated.connect(self.process_frame)
            self.data_thread.start()

    def stop_data_thread(self) -> None:
        if self.data_thread and self.data_thread.isRunning():
            self.data_thread.stop()
            self.data_thread.wait()

    def closeEvent(self, event: Any) -> None:
        self.stop_data_thread()
        super().closeEvent(event)

    def exit_application(self) -> None:
        self.stop_data_thread()
        if hasattr(self, 'pipeline') and self.pipeline:
            self.pipeline.stop()
        sys.exit()
