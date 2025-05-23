import numpy as np
import numpy.typing as npt
import cv2
from typing import Any, List, Dict, Literal, Tuple, Callable, Optional, cast
from PySide6.QtCore import Qt, QObject, QRunnable, QTimer, QThreadPool, Signal
from PySide6.QtGui import QCursor, QImage, QKeySequence, QScreen, QPixmap
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QStackedLayout
from platformdirs import user_data_dir
import io
import json
import time
import quaternion
from ahrs.filters import Madgwick
from data_acquisition import DataAcquisitionThread
import depth_sensor.interface.stream_profile
import depth_sensor.orbbec
import depth_sensor.orbbec.pipeline
from mathstuff import plane_from_points, compute_xy_transformation_matrix, apply_transformation, evaluate_plane, approximate_intersection, calculate_gravity_alignment_matrix, marker_pattern, project_color_pixel_to_depth_pixel, project_point_to_pixel, project_point_to_pixel_with_distortion, undistort_deproject
from calibration_data import CalibrationData
import depth_sensor.interface.pipeline
import depth_sensor.interface.frame
from depth_sensor.interface import pipeline as ds_pipeline
from depth_sensor.realsense import frame as ds_rs_frame
import pyrealsense2 as rs
import pyorbbecsdk as ob

import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib import pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class MainWindow(QWidget):
    data_thread: DataAcquisitionThread
    pipeline: depth_sensor.interface.pipeline.Pipeline[Any]
    calibration_data: CalibrationData
    threadpool: QThreadPool

def define_charuco_board_2d_points(board_size: Tuple[int, int], square_length: float) -> Dict[int, np.ndarray[Literal[2], np.dtype[np.float32]]]:
    points = dict[int, np.ndarray[Literal[2], np.dtype[np.float32]]]()
    id = 0
    for y in range(board_size[1]-1):
        for x in range(board_size[0]-1):
            points[id] = np.array([x * square_length, y * square_length])
            id += 1
    return points

def extract_3d_point(pixel: np.ndarray[Literal[2], np.dtype[np.float32]], intrins: depth_sensor.interface.stream_profile.CameraIntrinsic, depth_frame: depth_sensor.interface.frame.DepthFrame) -> Optional[np.ndarray[Literal[3], np.dtype[np.float32]]]:
    u, v = pixel.ravel()
    if u < 0 or v < 0:
        return None
    depth = depth_frame.get_distance(int(u), int(v))
    if depth < 0.1 or depth > 10:
        return None
    ud_pixel = undistort_deproject(np.dtype(np.float32), intrins, pixel)
    if ud_pixel is None:
        return None
    else:
        x, y = ud_pixel.ravel()
        return np.array([depth*x, depth*y, depth])

class Page3(QWidget):
    pipeline: depth_sensor.interface.pipeline.Pipeline[Any]
    Q: np.ndarray[Literal[4], np.dtype[np.float32]]

    def __init__(self, parent: MainWindow, next_page: Callable[[], None], exit_application: Callable[[], None], auto_progress: bool, enable_hdr: bool, screen: QScreen) -> None:
        super().__init__(parent)
        self.main_window = parent
        self.screen_size = screen.size()
        self.main_window_exit_application = exit_application
        self.motion_support = False
        self.auto_progress = auto_progress
        self.go_countdown_time = 3
        self.next_countdown_time = 5
        self.next_timer: Optional[QTimer] = None
        self.countdown_timer: Optional[QTimer] = None
        self.latest_color_frame: Optional[depth_sensor.interface.frame.ColorFrame] = None
        self.latest_depth_frame: Optional[depth_sensor.interface.frame.DepthFrame] = None
        self.madgwick = Madgwick(gain=0.5)
        self.Q = np.array([1.0, 0.0, 0.0, 0.0])
        self.cols = 30
        self.rows = 16
        # self.cols = 12
        # self.rows = 7
        self.next_page = next_page
        self.init_ui()
        self.latest_ob_accel_frame: Optional[ob.AccelFrame] = None
        self.latest_ob_gyro_frame: Optional[ob.GyroFrame] = None
        self.enable_hdr = enable_hdr

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
        self.go_button.setShortcut(QKeySequence("Ctrl+G"))
        self.initial_layout.addWidget(self.go_button)

        self.next_button = QPushButton("Next")
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self.go_next)
        self.next_button.setShortcut(QKeySequence("Ctrl+N"))
        self.initial_layout.addWidget(self.next_button)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.exit_application)
        self.exit_button.setShortcut(QKeySequence("Ctrl+Q"))
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
        self.start_data_thread()

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
        self.main_window.data_thread.frame_processor.signals.data_updated.disconnect()
        self.next_page()

    def resizeEvent(self, event: Any) -> None:
        self.create_charuco_board()

    class BigHugeProcess(QRunnable):
        from PySide6.QtCore import QSize

        color_frame: depth_sensor.interface.frame.ColorFrame
        depth_frame = depth_sensor.interface.frame.DepthFrame

        color_to_depth: depth_sensor.interface.stream_profile.Extrinsic

        cols: int
        rows: int

        screen_size: QSize

        Q: np.ndarray[Literal[4], np.dtype[np.float32]]
        motion_support: bool

        class Signals(QObject):
            def __init__(self):
                super().__init__()
            myFinished = Signal(bool, str, CalibrationData)

        signals: Signals

        def __init__(self):
            super().__init__()
            self.signals = self.Signals()

        def run(self):
            calibration_data: CalibrationData = CalibrationData()

            color_image = np.asanyarray(self.color_frame.get_data())

            # ------------------------------
            # Step 1. Detect the ChArUco board corners on the raw RGB image.
            # ------------------------------
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            board = cv2.aruco.CharucoBoard((self.cols, self.rows), 0.04, 0.03, aruco_dict)

            detector_parameters = cv2.aruco.DetectorParameters()
            charuco_parameters = cv2.aruco.CharucoParameters()
            charuco_detector = cv2.aruco.CharucoDetector(board, charuco_parameters, detector_parameters)

            gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
            if charuco_ids is None:
                self.signals.myFinished.emit(False, "No ChArUco board detected.", calibration_data)
                return
            
            print("finished step 1/5")

            # For visualization, draw the detected corners on the color image.
            charuco_corners_depth_aligned = charuco_corners
            cv2.aruco.drawDetectedCornersCharuco(color_image, charuco_corners_depth_aligned, charuco_ids, cornerColor=(0, 0, 255))
            calibration_data.color_image = color_image

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
                self.signals.myFinished.emit(False, "Insufficient detected points.", calibration_data)
                return
            h_board, _ = cv2.findHomography(detected_points, expected_points)
            if h_board is None:
                self.signals.myFinished.emit(False, "Homography computation failed.", calibration_data)
                return

            print("finished step 2/5")

            # ------------------------------
            # Step 3. Define the board region in the raw depth image.
            # ------------------------------
            depth_intrinsics = self.depth_frame.get_profile().as_video_stream_profile().get_intrinsic()
            color_intrinsics = self.color_frame.get_profile().as_video_stream_profile().get_intrinsic()

            board_polygon_board = np.array([
                [0.05, 0.05],
                [0.95, 0.05],
                [0.95, 0.95],
                [0.05, 0.95],
            ], dtype=np.float32)

            h_board_inv = np.linalg.inv(h_board)
            board_polygon_raw = cv2.perspectiveTransform(board_polygon_board.reshape(-1, 1, 2), h_board_inv)
            board_polygon_raw = board_polygon_raw.reshape(-1, 2)

            pixels = []
            for pixel in board_polygon_raw:
                depth_pixel = project_color_pixel_to_depth_pixel(
                    color_pixel=pixel,
                    depth_frame=self.depth_frame,
                    depth_intrinsic=depth_intrinsics,
                    color_intrinsic=color_intrinsics,
                    color_to_depth=self.color_to_depth
                )

                if depth_pixel is None or np.isnan(depth_pixel[0]) or np.isnan(depth_pixel[1]):
                    self.signals.myFinished.emit(False, "Failed to convert one or more of the screen polygon color corners to a depth corner. Probably screen is not in full view of infrared camera", calibration_data)
                    return
                
                pixels.append(depth_pixel)

            board_polygon_depth = np.array(pixels)

            # Create a mask for the board region in the depth image.
            board_mask = np.zeros((self.depth_frame.get_height(), self.depth_frame.get_width()), dtype=np.uint8)
            cv2.fillPoly(board_mask, [board_polygon_depth.astype(np.int32)], 255)

            # ------------------------------
            # Step 4. Generate a grid of points over the board region and process them in depth space.
            # ------------------------------
            ys, xs = np.where(board_mask > 0)
            grid_points = np.vstack((xs, ys)).T.astype(np.float32)

            # Filter out invalid points
            valid_mask = (grid_points[:, 0] != -1) & (grid_points[:, 1] != -1)
            grid_points = grid_points[valid_mask]

            # ------------------------------
            # Step 5. Gather 3D points from all valid depth pixels within the board region.
            # ------------------------------
            points_3d = []
            for pt in grid_points:
                x_d, y_d = pt

                if x_d < 0 or x_d >= self.depth_frame.get_width() or y_d < 0 or y_d >= self.depth_frame.get_height():
                    continue

                point_3d = extract_3d_point(pt, depth_intrinsics, self.depth_frame)  # Now using depth intrinsics
                if point_3d is not None:
                    points_3d.append(np.array(point_3d, dtype=np.float32))

            points_3d = np.array(points_3d, dtype=np.float32)

            if len(points_3d) == 0:
                self.signals.myFinished.emit(False, "No valid depth points found in board region.", calibration_data)
                return

            print("finished step 3/5")

            # # for debugging
            # self.page3.fail("debugging")
            # # display the point cloud of the board region
            # plt.figure()
            # ax = plt.axes(projection='3d')

            # for i, point in enumerate(inlier_points):
            # for i, point in enumerate(points_3d):
            #      if i % 100 == 0:
            #     ax.scatter(point[0], point[1], point[2], c='b', marker='x', s=0.5)

            # plt.show()
            # return

            # Fit a plane using the custom plane_from_points function.
            (depth_plane, plane_rmse, plane_max_error, inlier_points) = plane_from_points(points_3d)
            if depth_plane is None:
                self.signals.myFinished.emit(False, "Plane fitting failed", calibration_data)
                return

            depth_to_color = self.color_to_depth.inv()

            temp = np.array([depth_plane[0][0], depth_plane[0][1], depth_plane[0][2], 1])
            temp = depth_to_color.transform @ temp
            new_translation = temp[:3]
            new_rot = depth_to_color.rot @ depth_plane[1]
            plane = (new_translation, new_rot)

            calibration_data.depth_plane = depth_plane
            calibration_data.depth_to_color = depth_to_color
            calibration_data.color_intrinsics = color_intrinsics
            calibration_data.plane = plane
            calibration_data.plane_rmse = plane_rmse
            calibration_data.plane_max_error = plane_max_error

            print("RMSE of plane fit: ", plane_rmse)
            print("Max error of plane fit: ", plane_max_error)

            print("finished step 4/5")

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
                calibration_data.align_transform_mtx = (
                    np.linalg.inv(swap_yz)
                    @ quaternion.as_rotation_matrix(np.quaternion(*Q))
                    @ swap_yz
                )
                align_transform_inv_mtx = np.linalg.inv(calibration_data.align_transform_mtx)
            else:
                calibration_data.align_transform_mtx = np.eye(3, dtype=np.float32)
                align_transform_inv_mtx = np.eye(3, dtype=np.float32)

            # Align the 3D points with gravity.
            calibration_data.points_3d_aligned = [
                calibration_data.align_transform_mtx @ point for point in inlier_points
            ]

            # # for debugging
            # self.page3.fail("debugging")
            # # display the point cloud of the board region
            # plt.figure()
            # ax = plt.axes(projection='3d')

            # # for i, point in enumerate(inlier_points):
            # for i, point in enumerate(points_3d):
            #     if i % 10 == 0:
            #         ax.scatter(point[0], point[1], point[2], c='b', marker='x', s=0.5)

            # centroid, normal = plane
            # d = -np.dot(centroid, normal)
            # xx, yy = np.meshgrid(np.linspace(-0.2, 0.2, 10), np.linspace(-0.2, 0.2, 10))
            # zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
            # ax.plot_surface(xx, yy, zz, alpha=0.5)
            # plt.show()
            # return

            # ------------------------------
            # Step 7. Recompute a homography using deprojected board corners.
            # ------------------------------
            # For each detected board corner (from aligned charuco detection), approximate its intersection with the plane.
            deprojected_points = list[np.ndarray[Literal[3], np.dtype[np.float32]]]()
            detected_points_aligned_to_depth = detected_points
            for pt in detected_points_aligned_to_depth:
                deproj_pt = approximate_intersection(plane, color_intrinsics, pt[0], pt[1], 0, 1000)
                if np.all(deproj_pt == 0):
                    deprojected_points.append(np.array([np.NaN, np.NaN, np.NaN], dtype=np.float32))
                else:
                    deprojected_points.append(deproj_pt)

            deprojected_points_aligned = [
                calibration_data.align_transform_mtx @ point for point in deprojected_points
            ]
            deprojected_points_aligned = np.array(deprojected_points_aligned, dtype=np.float32)

            # Rotate the plane to align with gravity for computing a 2D transformation.
            plane_aligned = (
                calibration_data.align_transform_mtx @ calibration_data.plane[0],
                calibration_data.align_transform_mtx @ calibration_data.plane[1]
            )
            if len(np.unique(deprojected_points_aligned, axis=0)) <= 1:
                self.page3.fail("Deprojected points are not distinct.")
                return

            # Compute transformation matrix to flatten the plane to the XY plane.
            calibration_data.xy_transformation_matrix_aligned = compute_xy_transformation_matrix(plane_aligned)
            print("Transformation matrix (aligned): ", calibration_data.xy_transformation_matrix_aligned)

            # Apply the transformation to the deprojected points.
            transformed_points = apply_transformation(
                deprojected_points_aligned,
                calibration_data.xy_transformation_matrix_aligned
            )

            # todo tomorrow the transformed points wont always have the same shape because we remove invalids
            # print(expected_points)
            # print(transformed_points[:, :2])

            expected_points_without_nan = expected_points[~np.isnan(transformed_points[:, 0])]
            transformed_points_without_nan = transformed_points[~np.isnan(transformed_points[:, 0])]

            # Compute the homography mapping expected board points to transformed deprojected points.
            calibration_data.h_aligned, _ = cv2.findHomography(expected_points_without_nan, transformed_points_without_nan[:, :2])
            if calibration_data.h_aligned is None:
                print("Error: Homography could not be computed.")
                return

            print("Homography (aligned): ", calibration_data.h_aligned)

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
                calibration_data.h_aligned
            )
            offset_mtx = np.array([
                [1, 0, -transformed_unit_square_2d_aligned[0][0][0]],
                [0, 1, -transformed_unit_square_2d_aligned[0][0][1]],
                [0, 0, 1]
            ], dtype=np.float32)
            offset_mtx_3d = np.array([
                [1, 0, 0, -transformed_unit_square_2d_aligned[0][0][0]],
                [0, 1, 0, -transformed_unit_square_2d_aligned[0][0][1]],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            calibration_data.h_aligned = offset_mtx @ calibration_data.h_aligned
            calibration_data.xy_transformation_matrix_aligned = offset_mtx_3d @ calibration_data.xy_transformation_matrix_aligned
            transformed_unit_square_2d_aligned = cv2.perspectiveTransform(
                normalized_corners.reshape(-1, 1, 2),
                calibration_data.h_aligned
            )

            # ------------------------------
            # Step 9. Compute the final 3D quadrilateral.
            # ------------------------------
            transformed_unit_square_3d_aligned = transformed_unit_square_2d_aligned.reshape(-1, 2)
            transformed_unit_square_3d_aligned = np.hstack([
                transformed_unit_square_3d_aligned,
                np.zeros((4, 1), dtype=np.float32)
            ])
            calibration_data.best_quad_aligned = apply_transformation(
                transformed_unit_square_3d_aligned,
                np.linalg.inv(calibration_data.xy_transformation_matrix_aligned)
            )
            calibration_data.best_quad = [
                align_transform_inv_mtx @ point for point in calibration_data.best_quad_aligned
            ]
            calibration_data.best_quad_2d = []
            for point in calibration_data.best_quad:
                point_2d = project_point_to_pixel_with_distortion(color_intrinsics, point)
                calibration_data.best_quad_2d.append(point_2d)
            calibration_data.best_quad_2d = np.array(calibration_data.best_quad_2d, dtype=np.int32)

            print("finished step 5/5")

            self.signals.myFinished.emit(True, "", calibration_data)

    def on_go_clicked(self) -> None:
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.BlankCursor));
        self.stacked_layout.setCurrentWidget(self.charuco_widget)
        self.label.showFullScreen()
        QTimer.singleShot(5000, self.run_big_huge_process)  # type: ignore

    def create_charuco_board(self) -> None:
        # Create the ChArUco board
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        board = cv2.aruco.CharucoBoard((self.cols, self.rows), 0.04, 0.03, aruco_dict)

        # Generate the ChArUco board image
        board_image = board.generateImage(self.screen_size.toTuple(), marginSize=0, borderBits=1) # type: ignore

        height, width = board_image.shape[:2] # type: ignore
        bytes_per_line = width # type: ignore
        qt_image = QImage(board_image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8) # type: ignore
        pixmap = QPixmap.fromImage(qt_image)

        self.label.setPixmap(pixmap)

    def big_huge_process_finished(self, success: bool, err_msg: str, calibration_data: CalibrationData):
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.ArrowCursor));
        if success:
            self.main_window.calibration_data = calibration_data

            self.instructions_label.setText(f"Next in {self.next_countdown_time}")
            self.next_button.setEnabled(True)
            if self.auto_progress:
                self.start_next_countdown(success=True)
            self.label.hide()
            self.stacked_layout.setCurrentWidget(self.initial_widget)
        else:
            self.fail(err_msg)
        self.start_data_thread()

    def fail(self, msg: Optional[str]) -> None:
        print(f"Detection failure: {msg}")
        self.instructions_label.setText(f"Autocalibration unsuccessful. Exiting in {self.next_countdown_time}")
        if self.auto_progress:
            self.start_next_countdown(success=False)
        self.label.hide()
        self.stacked_layout.setCurrentWidget(self.initial_widget)

    def run_big_huge_process(self) -> None:
        color_frame = self.latest_color_frame
        depth_frame = self.latest_depth_frame

        if color_frame is None or depth_frame is None:
            self.fail("No frames available for ChArUco board detection.")
            return

        big_huge_process = self.BigHugeProcess()

        big_huge_process.color_frame = color_frame
        big_huge_process.depth_frame = depth_frame
        color_to_depth = color_frame.get_profile().get_extrinsic_to(depth_frame.get_profile())
        big_huge_process.color_to_depth = color_to_depth
        big_huge_process.rows = self.rows
        big_huge_process.cols = self.cols
        big_huge_process.Q = self.Q
        big_huge_process.motion_support = self.motion_support
        big_huge_process.screen_size = self.screen_size

        self.pipeline.stop()

        big_huge_process.signals.myFinished.connect(self.big_huge_process_finished)
        self.main_window.threadpool.start(big_huge_process)

    def process_frame(self, frames: depth_sensor.interface.frame.CompositeFrame) -> None:
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        accel_frame = None
        gyro_frame = None

        if self.motion_support and isinstance(frames, ds_rs_frame.CompositeFrame):
            # I'm lazy
            accel_frame = frames._internal.first_or_default(rs.stream.accel) # type: ignore
            gyro_frame = frames._internal.first_or_default(rs.stream.gyro) # type: ignore

            if not color_frame or not accel_frame or not gyro_frame:
                return

            # waste cpu resources because pylance
            if not accel_frame or not gyro_frame:
                return
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()

            # Update Madgwick filter
            # Swap coordinates because the Madgwick expects z to be gravity
            self.Q = self.madgwick.updateIMU(self.Q, gyr=np.array([gyro_data.x, gyro_data.z, -gyro_data.y]), acc=np.array([accel_data.x, accel_data.z, -accel_data.y])) # type: ignore
        else:
            if not color_frame:
                return

        self.latest_color_frame = color_frame
        self.latest_depth_frame = depth_frame

    def maybe_mad_update(self):
        a = self.latest_ob_accel_frame
        g = self.latest_ob_gyro_frame
        if a is not None and g is not None:
            # Update Madgwick filter
            # Swap coordinates because the Madgwick expects z to be gravity
            self.Q = self.madgwick.updateIMU(self.Q, gyr=np.array([g.get_x(), g.get_z(), -g.get_y()]), acc=np.array([a.get_x(), a.get_z(), -a.get_y()]))
            a = None
            g = None

    def ob_accel_updated(self, frame: ob.Frame):
        self.latest_ob_accel_frame = frame.as_accel_frame()
        self.maybe_mad_update()

    def ob_gyro_updated(self, frame: ob.Frame):
        self.latest_ob_gyro_frame = frame.as_gyro_frame()
        self.maybe_mad_update()

    def start_data_thread(self) -> None:
        # if self.main_window.data_thread and self.main_window.data_thread.running:
        #     return
        # self.main_window.data_thread = DataAcquisitionThread(self.pipeline, self.main_window.threadpool, start_pipeline)
        if self.enable_hdr:
            self.main_window.data_thread.frame_processor.set_filters(ds_pipeline.Filter.NOISE_REMOVAL | ds_pipeline.Filter.TEMPORAL | ds_pipeline.Filter.SPATIAL | ds_pipeline.Filter.HDR_MERGE)
        else:
            if isinstance(self.pipeline, depth_sensor.orbbec.pipeline.Pipeline):
                # the rest of the filters make it pretty inaccurate lol
                self.main_window.data_thread.frame_processor.set_filters(ds_pipeline.Filter.NOISE_REMOVAL)
            else:
                self.main_window.data_thread.frame_processor.set_filters(ds_pipeline.Filter.NOISE_REMOVAL | ds_pipeline.Filter.TEMPORAL | ds_pipeline.Filter.SPATIAL)
        self.main_window.data_thread.frame_processor.signals.data_updated.connect(self.process_frame)
        self.main_window.data_thread.signals.ob_accel_updated.connect(self.ob_accel_updated)
        self.main_window.data_thread.signals.ob_gyro_updated.connect(self.ob_gyro_updated)
        self.pipeline.start()
        self.pipeline.set_hdr_enabled(self.enable_hdr)
        # self.main_window.threadpool.start(self.main_window.data_thread)

    def closeEvent(self, event: Any) -> None:
        return

    def exit_application(self) -> None:
        self.main_window_exit_application()
