import numpy as np
import cv2
import pyrealsense2 as rs
from typing import Any, List, Dict, Literal, Tuple, Callable, Optional, cast
import matplotlib.pyplot as plt
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QStackedLayout
from mathstuff import *
from platformdirs import *
from pathlib import Path
import io
import json

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
    def __init__(self, parent: QWidget, exit_application: Callable[[], None], pipeline: Optional[rs.pipeline], screen_id: int) -> None:
        super().__init__(parent)
        self.main_window = parent
        self.exit_application = exit_application
        self.pipeline = pipeline
        self.temporal_filter: Optional[rs.temporal_filter] = None  # Placeholder for the temporal filter
        self.align: Optional[rs.align] = None  # Placeholder for the align processing block
        self.screen_id = screen_id
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

    def resizeEvent(self, event: Any) -> None:
        self.create_charuco_board()

    def on_go_clicked(self) -> None:
        self.stacked_layout.setCurrentWidget(self.charuco_widget)
        QTimer.singleShot(3000, self.detect_charuco_corners) # type: ignore

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
        if not self.pipeline or not self.temporal_filter or not self.align:
            print("Error: Pipeline, temporal filter, or align processing block is not initialized.")
            return

        # Allow some frames to stabilize
        for _ in range(30):
            self.pipeline.wait_for_frames()

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("Error: Could not capture frames.")
            return

        filtered_depth_frame = self.temporal_filter.process(depth_frame).as_depth_frame()

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(filtered_depth_frame.get_data())

        # Detect ChArUco board corners
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        board = cv2.aruco.CharucoBoard((11, 7), 0.04, 0.02, aruco_dict)

        # Initialize detector parameters
        detector_parameters = cv2.aruco.DetectorParameters()
        charuco_parameters = cv2.aruco.CharucoParameters()
        charuco_detector = cv2.aruco.CharucoDetector(board, charuco_parameters, detector_parameters)

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
        charuco_ids = cast(Optional[cv2.typing.MatLike], charuco_ids) # OpenCV typings are missing "| None" in several places

        if charuco_ids is not None:
            # cv2.aruco.drawDetectedMarkers(color_image, marker_corners, marker_ids)
            cv2.aruco.drawDetectedCornersCharuco(color_image, charuco_corners, charuco_ids, cornerColor=(0, 0, 255))

            # Extract 3D points from the depth frame
            points_3d = extract_3d_points(charuco_corners, filtered_depth_frame)

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
            charuco_board_2d_points = {id_: [(1 + point[0] + scale_x/2) / (11 + scale_x), (1 + point[1] + scale_y/2) / (7 + scale_y)] for id_, point in charuco_board_2d_points.items()}

            # Filter expected points and detected points based on IDs
            expected_points = list[np.ndarray[Literal[2], np.dtype[np.float32]]]()
            detected_points = list[np.ndarray[Literal[3], np.dtype[np.float32]]]()
            for i, id_ in enumerate(charuco_ids):
                if id_[0] in charuco_board_2d_points:
                    expected_points.append(charuco_board_2d_points[id_[0]])
                    detected_points.append(points_3d[i])

            expected_points = np.array(expected_points, dtype=np.float32)
            detected_points = np.array(detected_points, dtype=np.float32)

            points_3d = np.array(points_3d)

            # Fit a plane using the custom plane fitting function
            plane = plane_from_points(points_3d)

            # Deproject the detected points to the 3D plane
            deprojected_points = list[np.ndarray[Literal[3], np.dtype[np.float32]]]()
            intrin = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
            for detected_point in charuco_corners:
                deprojected_point = approximate_intersection(plane, intrin, detected_point[0][0], detected_point[0][1], 0, 1000)
                deprojected_points.append(deprojected_point)

            deprojected_points = np.array(deprojected_points)

            # Ensure deprojected_points are distinct
            if len(np.unique(deprojected_points, axis=0)) <= 1:
                print("Error: Deprojected points are not distinct.")
                return

            # print("before transformation")
            # print(deprojected_points)

            # Find transformation matrix that aligns the plane with the XY plane
            transformation_matrix = compute_transformation_matrix(plane)

            # Apply the transformation to the detected points
            transformed_points = apply_transformation(deprojected_points, transformation_matrix)

            # print("after transformation")
            # print(transformed_points)

            # Map the unit square corners to the plane's coordinate system using homography
            h, _ = cv2.findHomography(expected_points, transformed_points[:, :2])
            h = cast(Optional[cv2.typing.MatLike], h) # OpenCV typings are missing "| None" in several places

            if h is None:
                print("Error: Homography could not be computed.")
                return

            # Define the unit square corners
            normalized_corners = np.array([
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1]
            ], dtype=np.float32)

            # Map the unit square corners to the plane's coordinate system using homography
            transformed_unit_square_2d = cv2.perspectiveTransform(normalized_corners.reshape(-1,1,2), h)

            # Transform the resulting 2D points back to the 3D plane
            transformed_unit_square_3d = transformed_unit_square_2d.reshape(-1, 2)
            transformed_unit_square_3d = cast(np.ndarray[Any, np.dtype[np.float32]], transformed_unit_square_3d)
            transformed_unit_square_3d = np.hstack([transformed_unit_square_3d, np.zeros((4, 1), dtype=np.float32)])
            best_quad = apply_transformation(transformed_unit_square_3d, np.linalg.inv(transformation_matrix))

            # Project best_quad edges to the image
            best_quad_2d = []
            for point in best_quad:
                point_2d = rs.rs2_project_point_to_pixel(intrin, point)
                best_quad_2d.append(point_2d)

            best_quad_2d = np.array(best_quad_2d, dtype=np.int32)

            # Find the expected ir marker locations
            normalized_marker_pattern = marker_pattern()
            transformed_marker_pattern_3d = cv2.perspectiveTransform(normalized_marker_pattern.reshape(-1,1,2), h).reshape(-1, 2)
            transformed_marker_pattern_3d = cast(np.ndarray[Any, np.dtype[np.float32]], transformed_marker_pattern_3d)
            transformed_marker_pattern_3d = np.hstack([transformed_marker_pattern_3d, np.zeros((6, 1), dtype=np.float32)])
            expected_marker_pattern = apply_transformation(transformed_marker_pattern_3d, np.linalg.inv(transformation_matrix))

            # Project expected_marker_pattern to the image
            expected_marker_pattern_2d = []
            for point in expected_marker_pattern:
                point_2d = rs.rs2_project_point_to_pixel(intrin, point)
                expected_marker_pattern_2d.append(point_2d)

            expected_marker_pattern_2d = np.array(expected_marker_pattern_2d, dtype=np.int32)

            # TODO - Detect the ir markers based on the expected locations
            detected_marker_pattern = expected_marker_pattern

            # set members for saving
            self.plane = plane
            self.homography = h
            self.object_points = expected_marker_pattern

            # Draw the best quad on the image
            cv2.polylines(color_image, [best_quad_2d], isClosed=True, color=(0, 255, 0), thickness=2)

            # Draw the expected marker pattern on the image as purple outline circles
            for point in expected_marker_pattern_2d:
                cv2.circle(color_image, tuple(point), 5, (255, 0, 255), -1)

            # Show RGB image with detected parts
            height, width, channel = color_image.shape
            bytes_per_line = 3 * width
            qt_image = QImage(color_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.main_window.size() / 3, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.rgb_label.setPixmap(scaled_pixmap)

            # Show Matplotlib 3D plot of detected best quad corners with swapped Z and Y axes
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for point in best_quad:
                ax.scatter(point[0], point[2], -point[1], c='r', marker='o')  # Swap Y and Z, reverse Y

            # Draw best quad to the plot
            for i in range(4):
                ax.plot([best_quad[i][0], best_quad[(i + 1) % 4][0]],
                        [best_quad[i][2], best_quad[(i + 1) % 4][2]],
                        [-best_quad[i][1], -best_quad[(i + 1) % 4][1]], 'g')

            # Now also draw the detected chessboard corner 3d points in the plot
            for point in points_3d:
                ax.scatter(point[0], point[2], -point[1], c='b', marker='x', s=0.5)
            
            for point in expected_marker_pattern:
                ax.scatter(point[0], point[2], -point[1], c='violet', marker='o', s=10)

            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')

            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            matplotlib_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
            qt_image = QImage(matplotlib_image.data, width, height, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.main_window.size() / 3, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.matplotlib_label.setPixmap(scaled_pixmap)

            plt.close(fig)

            # Enable the "Done" button after showing the results
            self.next_button.setEnabled(True)

        self.stacked_layout.setCurrentWidget(self.initial_widget)
    
    def save_and_exit(self) -> None:
        _user_data_dir = Path(user_data_dir("odyssey", "odysseyarm", roaming=True))
        screens_dir = _user_data_dir.joinpath("screens")
        if not screens_dir.exists():
            screens_dir.mkdir(parents=True)
        screen_path = screens_dir.joinpath(f"screen_{self.screen_id}.json")
        print(f"Saving calibration data to {screen_path} then exiting")
        with io.open(screen_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({
                "plane": {
                    "origin": self.plane[0].tolist(),
                    "normal": self.plane[1].tolist(),
                },
                "homography": self.homography.tolist(),
                "object_points": self.object_points.tolist(),
            }))
        self.exit_application()
