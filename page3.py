from typing import Callable, Optional, List
import cv2
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QStackedLayout, QHBoxLayout

class Page3(QWidget):
    def __init__(self, parent: QWidget, next_page: Callable[[], None], exit_application: Callable[[], None], pipeline: Optional[rs.pipeline]) -> None:
        super().__init__(parent)
        self.main_window = parent
        self.next_page = next_page
        self.exit_application = exit_application
        self.pipeline = pipeline
        self.temporal_filter: Optional[rs.temporal_filter] = None  # Placeholder for the temporal filter
        self.align: Optional[rs.align] = None  # Placeholder for the align processing block
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

        self.results_layout = QHBoxLayout()

        self.rgb_label = QLabel()
        self.rgb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_layout.addWidget(self.rgb_label)

        self.matplotlib_label = QLabel()
        self.matplotlib_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_layout.addWidget(self.matplotlib_label)

        self.initial_layout.addLayout(self.results_layout)

        self.next_button = QPushButton("Done")
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
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.charuco_layout.addWidget(self.label)

        self.stacked_layout.addWidget(self.charuco_widget)

        self.stacked_layout.setCurrentWidget(self.initial_widget)

    def on_go_clicked(self) -> None:
        self.show_charuco_board()
        self.stacked_layout.setCurrentWidget(self.charuco_widget)
        QTimer.singleShot(3000, self.detect_charuco_corners)

    def show_charuco_board(self) -> None:
        # Create the ChArUco board
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        board = cv2.aruco.CharucoBoard((11, 7), 0.04, 0.02, aruco_dict)

        window_width = self.main_window.width()
        window_height = self.main_window.height()

        # Generate the ChArUco board image
        board_image = board.generateImage((window_width, window_height), marginSize=0, borderBits=1)

        height, width = board_image.shape[:2]
        bytes_per_line = width
        qt_image = QImage(board_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
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

        if charuco_ids is not None:
            cv2.aruco.drawDetectedMarkers(color_image, marker_corners, marker_ids)
            cv2.aruco.drawDetectedCornersCharuco(color_image, charuco_corners, charuco_ids)

            # Show RGB image with detected parts
            height, width, channel = color_image.shape
            bytes_per_line = 3 * width
            qt_image = QImage(color_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.main_window.size() / 2, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.rgb_label.setPixmap(scaled_pixmap)

            # Show Matplotlib 3D plot of detected corners with swapped Z and Y axes
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            points_3d = self.project_to_3d(charuco_corners, filtered_depth_frame)
            for point in points_3d:
                ax.scatter(point[0], point[2], -point[1], c='r', marker='o')  # Swap Y and Z, reverse Y
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')

            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            matplotlib_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
            qt_image = QImage(matplotlib_image.data, width, height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.main_window.size() / 2, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.matplotlib_label.setPixmap(scaled_pixmap)

            plt.close(fig)

            # Enable the "Done" button after showing the results
            self.next_button.setEnabled(True)

        self.stacked_layout.setCurrentWidget(self.initial_widget)

    def project_to_3d(self, charuco_corners: np.ndarray, depth_frame: rs.depth_frame) -> List[np.ndarray]:
        depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        points_3d = []
        for corner in charuco_corners:
            u, v = corner.ravel()
            depth = depth_frame.get_distance(int(u), int(v))
            point_3d = np.array(rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth))
            points_3d.append(point_3d)
        return points_3d

    def hide_charuco_board(self) -> None:
        self.label.clear()
        self.stacked_layout.setCurrentWidget(self.initial_widget)
        self.next_button.setEnabled(True)
