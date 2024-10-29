import numpy as np
import numpy.typing as npt
import quaternion
import cv2
import pyrealsense2 as rs
from typing import Any, List, Dict, Literal, Tuple, Callable, Optional, cast
import matplotlib.pyplot as plt
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QStackedLayout
from platformdirs import user_data_dir
from pathlib import Path
import io
import json
from ahrs.filters import Madgwick
from data_acquisition import DataAcquisitionThread
from mathstuff import plane_from_points, compute_xy_transformation_matrix, apply_transformation, evaluate_plane, approximate_intersection, calculate_gravity_alignment_matrix, marker_pattern

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
    def __init__(self, parent: QWidget, exit_application: Callable[[], None], pipeline: Optional[rs.pipeline], screen_id: int, output_dir: str, auto_progress: bool, depth_from_markers: bool) -> None:
        super().__init__(parent)
        self.main_window = parent
        self.exit_application = exit_application
        self.pipeline = pipeline
        self.temporal_filter: Optional[rs.temporal_filter] = None  # Placeholder for the temporal filter
        self.hole_filter: Optional[rs.hole_filling_filter] = None  # Placeholder for the hole filling filter
        self.align: Optional[rs.align] = None  # Placeholder for the align processing block
        self.screen_id = screen_id
        self.output_dir = output_dir
        self.auto_progress = auto_progress
        self.go_countdown_time = 3
        self.done_countdown_time = 10
        self.latest_color_frame: Optional[rs.frame] = None
        self.latest_depth_frame: Optional[rs.frame] = None
        self.latest_ir_frame: Optional[rs.frame] = None
        self.madgwick = Madgwick(gain=0.5)  # Initialize Madgwick filter
        # self.Q = np.array([1.0, 0.0, 0.0, 0.0])  # Initial quaternion
        self.data_thread: Optional[DataAcquisitionThread] = None
        self.depth_from_markers = depth_from_markers
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

        self.results_layout = QVBoxLayout()

        self.rgb_label = QLabel()
        self.rgb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_layout.addWidget(self.rgb_label)

        self.matplotlib_label = QLabel()
        self.matplotlib_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_layout.addWidget(self.matplotlib_label)

        self.initial_layout.addLayout(self.results_layout)

        self.next_button = QPushButton("Done")
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self.save_and_exit)
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

    def start_done_countdown(self, success: bool) -> None:
        """Start the countdown after 'Done' or 'Autocalibration unsuccessful'."""
        self.done_timer = QTimer(self)
        self.done_timer.timeout.connect(lambda: self.update_done_countdown(success))
        self.done_timer.start(1000)

    def update_done_countdown(self, success: bool) -> None:
        """Update the countdown based on whether the calibration was successful or not."""
        self.done_countdown_time -= 1
        if success:
            self.instructions_label.setText(f"Done in {self.done_countdown_time}")
        else:
            self.instructions_label.setText(f"Autocalibration unsuccessful. Exiting in {self.done_countdown_time}")
        
        if self.done_countdown_time <= 0:
            self.done_timer.stop()
            if success:
                self.save_and_exit()
            else:
                self.exit_application()

    def resizeEvent(self, event: Any) -> None:
        self.create_charuco_board()

    def on_go_clicked(self) -> None:
        self.stacked_layout.setCurrentWidget(self.charuco_widget)
        QTimer.singleShot(3000, self.detect_charuco_corners)  # type: ignore

    def create_charuco_board(self) -> None:
        # Create the ChArUco board
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        board = cv2.aruco.CharucoBoard((11, 7), 0.04, 0.02, aruco_dict)

        window_width = self.main_window.width()
        window_height = self.main_window.height()

        # Generate the ChArUco board image
        board_image = board.generateImage((window_width, window_height), marginSize=0, borderBits=1)

        height, width = board_image.shape[:2]
        bytes_per_line = width
        qt_image = QImage(board_image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qt_image)

        self.label.setPixmap(pixmap)

    def detect_charuco_corners(self) -> None:
        def fail(msg: str) -> None:
            print(f"Detection failure: {msg}")
            self.instructions_label.setText(f"Autocalibration unsuccessful. Exiting in {self.done_countdown_time}")
            if self.auto_progress:
                self.start_done_countdown(success=False)
            self.stacked_layout.setCurrentWidget(self.initial_widget)

        if not self.latest_color_frame or not self.latest_depth_frame or not self.latest_ir_frame:
            fail("No frames available for ChArUco board detection.")
            return

        color_frame = self.latest_color_frame
        depth_frame = self.latest_depth_frame
        ir_frame = self.latest_ir_frame

        color_image = np.asanyarray(color_frame.get_data())
        ir_image = np.asanyarray(ir_frame.get_data())

        # Detect ChArUco board corners
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        board = cv2.aruco.CharucoBoard((11, 7), 0.04, 0.02, aruco_dict)

        # Initialize detector parameters
        detector_parameters = cv2.aruco.DetectorParameters()
        charuco_parameters = cv2.aruco.CharucoParameters()
        charuco_detector = cv2.aruco.CharucoDetector(board, charuco_parameters, detector_parameters)

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
        charuco_ids = cast(Optional["cv2.typing.MatLike"], charuco_ids)  # OpenCV typings are missing "| None" in several places

        if charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(color_image, charuco_corners, charuco_ids, cornerColor=(0, 0, 255))

            # Extract 3D points from the depth frame
            points_3d = extract_3d_points(charuco_corners, depth_frame)

            # Calculate the transformation matrix and its inverse to align with gravity
            # align_transform_mtx = quaternion.as_rotation_matrix(np.quaternion(*self.Q))
            # align_transform_inv_mtx = np.linalg.inv(align_transform_mtx)
            align_transform_mtx = np.eye(3, dtype=np.float64)
            align_transform_inv_mtx = np.eye(3, dtype=np.float64)

            # Align the 3D points with gravity
            points_3d_aligned = [align_transform_mtx @ point for point in points_3d]

            intrin = rs.video_stream_profile(depth_frame.profile).get_intrinsics()

            # Define the expected positions of the ChArUco board corners
            charuco_board_2d_points = define_charuco_board_2d_points((11, 7), 1)

            screen_width = self.main_window.width()
            screen_height = self.main_window.height()

            board_aspect_ratio = 11 / 7
            screen_aspect_ratio = screen_width / screen_height

            board_width_in_pixels = screen_height * board_aspect_ratio
            board_height_in_pixels = screen_width / board_aspect_ratio

            if screen_aspect_ratio < board_aspect_ratio:
                scale_x = 0
                scale_y = (screen_height - board_height_in_pixels) / (board_height_in_pixels / 7)
            else:
                scale_x = (screen_width - board_width_in_pixels) / (board_width_in_pixels / 11)
                scale_y = 0

            # Normalize the expected positions to the unit square
            charuco_board_2d_points = {id_: [(1 + point[0] + scale_x / 2) / (11 + scale_x), (1 + point[1] + scale_y / 2) / (7 + scale_y)] for id_, point in charuco_board_2d_points.items()}

            # Filter expected points and detected points based on IDs
            expected_points = list[np.ndarray[Literal[2], np.dtype[np.float32]]]()
            detected_points = list[np.ndarray[Literal[3], np.dtype[np.float32]]]()
            for i, id_ in enumerate(charuco_ids):
                if id_[0] in charuco_board_2d_points:
                    expected_points.append(charuco_board_2d_points[id_[0]])
                    detected_points.append(points_3d[i])

            expected_points = np.array(expected_points, dtype=np.float32)
            detected_points = np.array(detected_points, dtype=np.float32)

            # Detect IR blobs in the depth image
            ir_image = ir_image.astype(np.uint8)

            # Apply a threshold to binarize the image
            _, binary_ir_image = cv2.threshold(ir_image, 240, 255, cv2.THRESH_BINARY)

            # Find contours in the binary image
            contours, _ = cv2.findContours(binary_ir_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected_markers_2d = []
            for cnt in contours:
                if cv2.contourArea(cnt) > 20:  # Filter small contours
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        detected_markers_2d.append([cx, cy])

            # Fit a plane using the custom plane fitting function
            if self.depth_from_markers:
                detected_markers_3d = extract_3d_points(np.array(detected_markers_2d, dtype=np.float32), depth_frame)
                detected_markers_3d = np.array(detected_markers_3d, dtype=np.float32)
                plane = plane_from_points(detected_markers_3d, 6)
            else:
                plane = plane_from_points(detected_points, 15)
            if plane is None:
                fail("Plane fitting failed")
                return

            # Deproject the detected points to the 3D plane using transformed intrinsics
            deprojected_points = list[np.ndarray[Literal[3], np.dtype[np.float32]]]()
            for detected_point in charuco_corners:
                deprojected_point = approximate_intersection(plane, intrin, detected_point[0][0], detected_point[0][1], 0, 1000)
                deprojected_points.append(deprojected_point)

            deprojected_points_aligned = [align_transform_mtx @ point for point in deprojected_points]
            deprojected_points_aligned = np.array(deprojected_points_aligned, dtype=np.float32)

            # rotate plane to align with gravite
            plane_aligned = (align_transform_mtx @ plane[0], align_transform_mtx @ plane[1])

            print(deprojected_points_aligned)
            # Ensure deprojected_points are distinct
            if len(np.unique(deprojected_points_aligned, axis=0)) <= 1:
                fail("Deprojected points are not distinct.")
                return

            # Find transformation matrix that aligns the plane with the XY plane
            xy_transformation_matrix_aligned = compute_xy_transformation_matrix(plane_aligned)

            print("Deprojected points (aligned): ", deprojected_points_aligned)

            print("Transformation matrix (aligned): ", xy_transformation_matrix_aligned)

            # Apply the transformation to the detected points
            transformed_points = apply_transformation(deprojected_points_aligned, xy_transformation_matrix_aligned)

            print("Transformed points: ", transformed_points)

            # Map the unit square corners to the plane's coordinate system using homography
            h_aligned, _ = cv2.findHomography(expected_points, transformed_points[:, :2])
            h_aligned = cast(Optional["cv2.typing.MatLike"], h_aligned)  # OpenCV typings are missing "| None" in several places

            if h_aligned is None:
                print("Error: Homography could not be computed.")
                return

            # Define the unit square corners
            normalized_corners = np.array([
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1]
            ], dtype=np.float32)

            print("Homography (aligned): ", h_aligned)

            # Map the unit square corners to the plane's coordinate system using homography
            transformed_unit_square_2d_aligned = cv2.perspectiveTransform(normalized_corners.reshape(-1, 1, 2), h_aligned)

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
            h_aligned = offset_mtx @ h_aligned
            xy_transformation_matrix_aligned = offset_mtx_3d @ xy_transformation_matrix_aligned
            transformed_unit_square_2d_aligned = cv2.perspectiveTransform(normalized_corners.reshape(-1, 1, 2), h_aligned)

            # Transform the resulting 2D points back to the 3D plane
            transformed_unit_square_3d_aligned = transformed_unit_square_2d_aligned.reshape(-1, 2)
            transformed_unit_square_3d_aligned = cast(np.ndarray[Any, np.dtype[np.float32]], transformed_unit_square_3d_aligned)
            transformed_unit_square_3d_aligned = np.hstack([transformed_unit_square_3d_aligned, np.zeros((4, 1), dtype=np.float32)])
            best_quad_aligned = apply_transformation(transformed_unit_square_3d_aligned, np.linalg.inv(xy_transformation_matrix_aligned))

            # Unalign the best quad points for 2D projection
            best_quad = [align_transform_inv_mtx @ point for point in best_quad_aligned]

            # Project best_quad edges to the image
            best_quad_2d = []
            for point in best_quad:
                point_2d = rs.rs2_project_point_to_pixel(intrin, point)
                best_quad_2d.append(point_2d)

            best_quad_2d = np.array(best_quad_2d, dtype=np.int32)

            # cv2.imshow("Binary IR Image", binary_ir_image)

            # Find the expected ir marker locations
            normalized_marker_pattern = marker_pattern()
            transformed_marker_pattern_3d_aligned = cv2.perspectiveTransform(normalized_marker_pattern.reshape(-1, 1, 2), h_aligned).reshape(-1, 2)
            transformed_marker_pattern_3d_aligned = cast(np.ndarray[Any, np.dtype[np.float32]], transformed_marker_pattern_3d_aligned)
            transformed_marker_pattern_3d_aligned = np.hstack([transformed_marker_pattern_3d_aligned, np.zeros((6, 1), dtype=np.float32)])
            expected_marker_pattern_aligned = apply_transformation(transformed_marker_pattern_3d_aligned, np.linalg.inv(xy_transformation_matrix_aligned))

            if len(detected_markers_2d) < len(expected_marker_pattern_aligned):
                fail("Number of detected IR blobs is less than expected.")
                return

            detected_markers_3d = []
            for point in detected_markers_2d:
                point_3d = approximate_intersection(plane, intrin, point[0], point[1], 0, 1000)
                detected_markers_3d.append(point_3d)
            
            print("Detected markers 3D: ", detected_markers_3d)

            detected_markers_3d_aligned = [align_transform_mtx @ point for point in detected_markers_3d]

            # Find the closest detected blobs to the expected positions
            detected_marker_pattern_aligned = []
            detected_marker_pattern_2d = []
            for point in expected_marker_pattern_aligned:
                distances = [np.linalg.norm(point - detected_marker) for detected_marker in detected_markers_3d_aligned]
                closest_index = np.argmin(distances)
                detected_marker_pattern_aligned.append(detected_markers_3d_aligned[closest_index])
                detected_marker_pattern_2d.append(detected_markers_2d[closest_index])
                detected_markers_3d_aligned.pop(closest_index)
                detected_markers_2d.pop(closest_index)

            # Draw the best quad on the image
            cv2.polylines(color_image, [best_quad_2d], isClosed=True, color=(0, 255, 0), thickness=2)

            # Draw the detected marker pattern on the image as purple circles
            for i,point in enumerate(detected_marker_pattern_2d):
                cv2.circle(color_image, tuple(point), 5, (255, 0, 255), -1)
                cv2.putText(color_image, str(i), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Show RGB image with detected parts
            height, width = color_image.shape[:2]
            bytes_per_line = width * 3
            qt_image = QImage(color_image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.main_window.size() / 3, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.rgb_label.setPixmap(scaled_pixmap)

            # Show Matplotlib 3D plot of detected best quad corners
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for point in best_quad_aligned:
                ax.scatter(point[0], point[2], point[1], c='r', marker='o')  # Swap Y and Z

            # Draw best quad to the plot
            for i in range(4):
                ax.plot([best_quad_aligned[i][0], best_quad_aligned[(i + 1) % 4][0]],
                        [best_quad_aligned[i][2], best_quad_aligned[(i + 1) % 4][2]],
                        [best_quad_aligned[i][1], best_quad_aligned[(i + 1) % 4][1]], 'g')  # Swap Y and Z

            # Now also draw the detected chessboard corner 3d points in the plot
            for point in points_3d_aligned:
                ax.scatter(point[0], point[2], point[1], c='b', marker='x', s=0.5)  # Swap Y and Z

            for point in detected_marker_pattern_aligned:
                ax.scatter(point[0], point[2], point[1], c='violet', marker='o', s=10)  # Swap Y and Z

            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')

            ax.invert_zaxis() # Actually inverting y axis
            ax.set_aspect('equal')

            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            matplotlib_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
            qt_image = QImage(matplotlib_image.data, width, height, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.main_window.size() / 3, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.matplotlib_label.setPixmap(scaled_pixmap)

            plt.close(fig)

            detected_marker_pattern_aligned = np.array(detected_marker_pattern_aligned, dtype=np.float32)

            detected_marker_pattern_aligned_transformed = apply_transformation(detected_marker_pattern_aligned, xy_transformation_matrix_aligned)

            # set members for saving
            self.homography = np.linalg.inv(h_aligned)
            self.object_points = detected_marker_pattern_aligned_transformed

            # Enable the "Done" button after showing the results
            self.next_button.setEnabled(True)
        else:
            fail("No ChArUco board detected.")
            return

        self.instructions_label.setText(f"Done in {self.done_countdown_time}")
        self.next_button.setEnabled(True)
        if self.auto_progress:
            self.start_done_countdown(success=True)

        self.stacked_layout.setCurrentWidget(self.initial_widget)

    def save_and_exit(self) -> None:
        if self.output_dir:
            _user_data_dir = Path(self.output_dir)
        else:
            _user_data_dir = Path(user_data_dir("odyssey", "odysseyarm", roaming=True))

        screens_dir = _user_data_dir.joinpath("screens")
        if not screens_dir.exists():
            screens_dir.mkdir(parents=True)

        screen_path = screens_dir.joinpath(f"screen_{self.screen_id}.json")

        print(f"Saving calibration data to {screen_path}...")

        with io.open(screen_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({
                "homography": self.homography.flatten().tolist(),
                "object_points": self.object_points.tolist(),
            }))
        
        print("Saved. Exiting application.")

        self.exit_application()

    def process_frame(self, frames: rs.frame) -> None:
        # frames = self.temporal_filter.process(frames)
        frames = self.hole_filter.process(frames).as_frameset()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        ir_frame = aligned_frames.get_infrared_frame(0)
        # accel_frame = frames.first_or_default(rs.stream.accel)
        # gyro_frame = frames.first_or_default(rs.stream.gyro)

        # if not color_frame or not ir_frame or not accel_frame or not gyro_frame:
        if not color_frame or not ir_frame:
            return

        self.latest_color_frame = color_frame
        self.latest_depth_frame = depth_frame
        self.latest_ir_frame = ir_frame

        # accel_data = accel_frame.as_motion_frame().get_motion_data()
        # gyro_data = gyro_frame.as_motion_frame().get_motion_data()

        # Update Madgwick filter
        # self.Q = self.madgwick.updateIMU(self.Q, gyr=[gyro_data.x, gyro_data.y, gyro_data.z], acc=[accel_data.x, accel_data.z, -accel_data.y])

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
