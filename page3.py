import numpy as np
import numpy.typing as npt
import cv2
import pyrealsense2 as rs
from typing import Any, List, Dict, Literal, Tuple, Callable, Optional, cast
import matplotlib.pyplot as plt
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QStackedLayout
from platformdirs import user_data_dir
import io
import json
import time
from ahrs.filters import Madgwick
from data_acquisition import DataAcquisitionThread
from mathstuff import plane_from_points, compute_xy_transformation_matrix, apply_transformation, evaluate_plane, approximate_intersection, calculate_gravity_alignment_matrix, marker_pattern
from calibration_data import CalibrationData

def define_charuco_board_2d_points(board_size: Tuple[int, int], square_length: float) -> Dict[int, np.ndarray[Literal[2], np.dtype[np.float32]]]:
    points = dict[int, np.ndarray[Literal[2], np.dtype[np.float32]]]()
    id = 0
    for y in range(board_size[1]-1):
        for x in range(board_size[0]-1):
            points[id] = np.array([x * square_length, y * square_length])
            id += 1
    return points

def extract_3d_points(charuco_corners: np.ndarray[Any, np.dtype[Any]], depth_frame: rs.depth_frame) -> List[np.ndarray[Literal[3], np.dtype[np.float32]]]:
    depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
    points_3d = list[np.ndarray[Literal[3], np.dtype[np.float32]]]()

    corner: np.ndarray[Literal[2], np.dtype[np.float32]]
    for corner in charuco_corners:
        u, v = corner.ravel()
        depth = depth_frame.get_distance(int(u), int(v))
        point_3d = np.array(rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth), dtype=np.float32)
        points_3d.append(point_3d)

    return points_3d

class Page3(QWidget):
    def __init__(self, parent: QWidget, next_page: Callable[[], None], exit_application: Callable[[], None], pipeline: Optional[rs.pipeline], auto_progress: bool, calibration_data: CalibrationData) -> None:
        super().__init__(parent)
        self.main_window = parent
        self.exit_application = exit_application
        self.motion_support = False
        self.pipeline = pipeline
        self.pipeline_profile = None
        self.spatial_filter: Optional[rs.spatial_filter] = None  # Placeholder for the spatial filter
        self.temporal_filter: Optional[rs.temporal_filter] = None  # Placeholder for the temporal filter
        self.hole_filter: Optional[rs.hole_filling_filter] = None  # Placeholder for the hole filling filter
        self.align: Optional[rs.align] = None  # Placeholder for the align processing block
        self.auto_progress = auto_progress
        self.go_countdown_time = 3
        self.next_countdown_time = 5
        self.latest_color_frame: Optional[rs.frame] = None
        self.latest_depth_frame: Optional[rs.frame] = None
        self.madgwick = Madgwick(gain=0.5)  # Initialize Madgwick filter
        self.Q = np.array([1.0, 0.0, 0.0, 0.0])  # Initial quaternion
        self.data_thread: Optional[DataAcquisitionThread] = None
        self.cols = 14
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
        self.next_button.clicked.connect(self.next_page)
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
        self.charuco_layout.addWidget(self.label)

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
            self.next_timer.stop()
            if success:
                self.next_page()
            else:
                self.exit_application()

    def resizeEvent(self, event: Any) -> None:
        self.create_charuco_board()

    def on_go_clicked(self) -> None:
        self.stacked_layout.setCurrentWidget(self.charuco_widget)
        QTimer.singleShot(3000, self.detect_charuco_corners)  # type: ignore

    def create_charuco_board(self) -> None:
        # Create the ChArUco board
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        board = cv2.aruco.CharucoBoard((self.cols, self.rows), 0.04, 0.02, aruco_dict)

        window_width = self.main_window.width()
        window_height = self.main_window.height()

        # Generate the ChArUco board image
        board_image = board.generateImage((window_width, window_height), marginSize=0, borderBits=1)

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
        self.stacked_layout.setCurrentWidget(self.initial_widget)

    def detect_charuco_corners(self) -> None:
        if not self.latest_color_frame or not self.latest_depth_frame:
            self.fail("No frames available for ChArUco board detection.")
            return

        color_frame = self.latest_color_frame
        depth_frame = self.latest_depth_frame
        self.latest_depth_frame = None

        QTimer.singleShot(500, lambda: self._detect_charuco_corners_continued(color_frame, depth_frame))

    def _detect_charuco_corners_continued(self, color_frame, depth_frame):
        Q = np.array(self.Q)

        color_image = np.asanyarray(color_frame.get_data())

        # Detect ChArUco board corners
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        board = cv2.aruco.CharucoBoard((self.cols, self.rows), 0.04, 0.02, aruco_dict)

        # Initialize detector parameters
        detector_parameters = cv2.aruco.DetectorParameters()
        charuco_parameters = cv2.aruco.CharucoParameters()
        charuco_detector = cv2.aruco.CharucoDetector(board, charuco_parameters, detector_parameters)

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
        charuco_ids = cast(Optional["cv2.typing.MatLike"], charuco_ids)  # OpenCV typings are missing "| None" in several places

        if charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(color_image, charuco_corners, charuco_ids, cornerColor=(0, 0, 255))

            self.calibration_data.color_image = color_image

            # Extract 3D points from the depth frame
            points_3d = extract_3d_points(charuco_corners, depth_frame)

            # Calculate the transformation matrix and its inverse to align with gravity
            if self.motion_support:
                swap_yz = np.array([
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0],
                ])
                self.calibration_data.align_transform_mtx = np.linalg.inv(swap_yz) @ quaternion.as_rotation_matrix(np.quaternion(*Q)) @ swap_yz
                align_transform_inv_mtx = np.linalg.inv(self.calibration_data.align_transform_mtx)
            else:
                self.calibration_data.align_transform_mtx = np.eye(3, dtype=np.float64)
                align_transform_inv_mtx = np.eye(3, dtype=np.float64)

            # Align the 3D points with gravity
            self.calibration_data.points_3d_aligned = [self.calibration_data.align_transform_mtx @ point for point in points_3d]

            self.calibration_data.intrin = rs.video_stream_profile(depth_frame.profile).get_intrinsics()

            # Define the expected positions of the ChArUco board corners
            charuco_board_2d_points = define_charuco_board_2d_points((self.cols, self.rows), 1)

            screen_width = self.main_window.width()
            screen_height = self.main_window.height()

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

            # Normalize the expected positions to the unit square
            charuco_board_2d_points = {id_: [(1 + point[0] + scale_x / 2) / (self.cols + scale_x), (1 + point[1] + scale_y / 2) / (self.rows + scale_y)] for id_, point in charuco_board_2d_points.items()}

            # Filter expected points and detected points based on IDs
            expected_points = list[np.ndarray[Literal[2], np.dtype[np.float32]]]()
            detected_points = list[np.ndarray[Literal[3], np.dtype[np.float32]]]()
            for i, id_ in enumerate(charuco_ids):
                if id_[0] in charuco_board_2d_points:
                    expected_points.append(charuco_board_2d_points[id_[0]])
                    detected_points.append(points_3d[i])

            expected_points = np.array(expected_points, dtype=np.float32)
            detected_points = np.array(detected_points, dtype=np.float32)

            # Fit a plane using the custom plane fitting function
            self.calibration_data.plane = plane_from_points(detected_points, 15)
            if self.calibration_data.plane is None:
                self.fail("Plane fitting failed")
                return

            # Deproject the detected points to the 3D plane using transformed intrinsics
            deprojected_points = list[np.ndarray[Literal[3], np.dtype[np.float32]]]()
            for detected_point in charuco_corners:
                deprojected_point = approximate_intersection(self.calibration_data.plane, self.calibration_data.intrin, detected_point[0][0], detected_point[0][1], 0, 1000)
                deprojected_points.append(deprojected_point)

            deprojected_points_aligned = [self.calibration_data.align_transform_mtx @ point for point in deprojected_points]
            deprojected_points_aligned = np.array(deprojected_points_aligned, dtype=np.float32)

            # rotate plane to align with gravity
            plane_aligned = (self.calibration_data.align_transform_mtx @ self.calibration_data.plane[0], self.calibration_data.align_transform_mtx @ self.calibration_data.plane[1])

            print(deprojected_points_aligned)
            # Ensure deprojected_points are distinct
            if len(np.unique(deprojected_points_aligned, axis=0)) <= 1:
                self.fail("Deprojected points are not distinct.")
                return

            # Find transformation matrix that aligns the plane with the XY plane
            self.calibration_data.xy_transformation_matrix_aligned = compute_xy_transformation_matrix(plane_aligned)

            print("Deprojected points (aligned): ", deprojected_points_aligned)

            print("Transformation matrix (aligned): ", self.calibration_data.xy_transformation_matrix_aligned)

            # Apply the transformation to the detected points
            transformed_points = apply_transformation(deprojected_points_aligned, self.calibration_data.xy_transformation_matrix_aligned)

            print("Transformed points: ", transformed_points)

            # Map the unit square corners to the plane's coordinate system using homography
            self.calibration_data.h_aligned, _ = cv2.findHomography(expected_points, transformed_points[:, :2])
            self.calibration_data.h_aligned = cast(Optional["cv2.typing.MatLike"], self.calibration_data.h_aligned)  # OpenCV typings are missing "| None" in several places

            if self.calibration_data.h_aligned is None:
                print("Error: Homography could not be computed.")
                return

            # Define the unit square corners
            normalized_corners = np.array([
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1]
            ], dtype=np.float32)

            print("Homography (aligned): ", self.calibration_data.h_aligned)

            # Map the unit square corners to the plane's coordinate system using homography
            transformed_unit_square_2d_aligned = cv2.perspectiveTransform(normalized_corners.reshape(-1, 1, 2), self.calibration_data.h_aligned)

            # Translate transformed_unit_square_2d_aligned, h_aligned, and xy_transformation_matrix_aligned by -transformed_unit_square_2d_aligned[0][0]
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
            transformed_unit_square_2d_aligned = cv2.perspectiveTransform(normalized_corners.reshape(-1, 1, 2), self.calibration_data.h_aligned)

            # Transform the resulting 2D points back to the 3D plane
            transformed_unit_square_3d_aligned = transformed_unit_square_2d_aligned.reshape(-1, 2)
            transformed_unit_square_3d_aligned = cast(np.ndarray[Any, np.dtype[np.float32]], transformed_unit_square_3d_aligned)
            transformed_unit_square_3d_aligned = np.hstack([transformed_unit_square_3d_aligned, np.zeros((4, 1), dtype=np.float32)])
            self.calibration_data.best_quad_aligned = apply_transformation(transformed_unit_square_3d_aligned, np.linalg.inv(self.calibration_data.xy_transformation_matrix_aligned))

            # Unalign the best quad points for 2D projection
            self.calibration_data.best_quad = [align_transform_inv_mtx @ point for point in self.calibration_data.best_quad_aligned]

            # Project best_quad edges to the image
            self.calibration_data.best_quad_2d = []
            for point in self.calibration_data.best_quad:
                point_2d = rs.rs2_project_point_to_pixel(self.calibration_data.intrin, point)
                self.calibration_data.best_quad_2d.append(point_2d)

            self.calibration_data.best_quad_2d = np.array(self.calibration_data.best_quad_2d, dtype=np.int32)

            # Enable the "Next" button after showing the results
            self.next_button.setEnabled(True)
        else:
            self.fail("No ChArUco board detected.")
            return

        self.instructions_label.setText(f"Next in {self.next_countdown_time}")
        self.next_button.setEnabled(True)
        if self.auto_progress:
            self.start_next_countdown(success=True)

        self.stacked_layout.setCurrentWidget(self.initial_widget)

    def process_frame(self, frames: rs.frame) -> None:
        frames = self.spatial_filter.process(frames)
        frames = self.temporal_filter.process(frames)
        frames = self.hole_filter.process(frames).as_frameset()
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
