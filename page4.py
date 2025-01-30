import cv2
import pyrealsense2 as rs
from typing import Any, List, Dict, Literal, Tuple, Callable, Optional, cast
from PySide6.QtCore import QTimer, Qt, QEvent
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QPushButton, QHBoxLayout, QLabel
from platformdirs import user_data_dir
from pathlib import Path
from data_acquisition import DataAcquisitionThread
import numpy as np
from calibration_data import CalibrationData
import mathstuff

class Page4(QWidget):
    def __init__(self, parent: QWidget, next_page: Callable[[], None], exit_application: Callable[[], None], pipeline: Optional[rs.pipeline], auto_progress: bool, ir_low_exposure: float, calibration_data: CalibrationData) -> None:
        super().__init__(parent)
        self.main_window = parent
        self.exit_application = exit_application
        self.motion_support = False
        self.pipeline = pipeline
        self.pipeline_profile = None
        self.auto_progress = auto_progress
        self.countdown_timer: Optional[QTimer] = None
        self.done_countdown_time = 10
        self.latest_ir_frame: Optional[rs.frame] = None
        self.data_thread: Optional[DataAcquisitionThread] = None
        self.ir_low_exposure = ir_low_exposure
        self.remaining_time = 30
        self.fail_countdown_time = 5
        self.next_page = next_page
        self.calibration_data = calibration_data
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.canvas = QGraphicsView()
        self.scene = QGraphicsScene(self)
        self.canvas.setScene(self.scene)
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.canvas.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self.canvas)

        self.camera_pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.camera_pixmap_item)

        self.countdown_label = QLabel("30", self)
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.countdown_label.setVisible(False)  # Initially hidden
        layout.addWidget(self.countdown_label)

        button_layout = QHBoxLayout()
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next)
        button_layout.addWidget(self.next_button)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.exit_application)
        button_layout.addWidget(self.exit_button)

        if self.auto_progress:
            self.next_button.setVisible(False)
            self.exit_button.setVisible(False)
            self.countdown_label.setVisible(True)  # Show countdown if auto_progress

        layout.addLayout(button_layout)

        self.instructions_label = QLabel("Turn markers on and get out of view. All six markers must be visible. Do not touch the RealSense camera. Restart the application if the markers are not visible instead.")
        self.instructions_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.instructions_label)

        self.setLayout(layout)

    def start_fail_countdown(self) -> None:
        """Start the countdown after 'Autocalibration unsuccessful'."""
        self.fail_timer = QTimer(self)
        self.fail_timer.timeout.connect(self.update_fail_countdown)
        self.fail_timer.start(1000)

    def update_fail_countdown(self) -> None:
        """Update the fail countdown."""
        self.fail_countdown_time -= 1
        self.instructions_label.setText(f"Autocalibration unsuccessful. Exiting in {self.fail_countdown_time}")

        if self.fail_countdown_time <= 0:
            self.fail_timer.stop()
            self.exit_application()

    def next(self) -> None:
        if self.countdown_timer is not None:
            self.countdown_timer.stop()
        if self.latest_ir_frame is None:
            self.fail("No frames for marker detection")
        elif not self.detect_markers(self.latest_ir_frame):
            self.fail("Marker detection failed")
        else:
            self.next_page()

    def detect_markers(self, ir_frame: rs.frame) -> bool:
        # Detect IR blobs in the depth image
        markers_ir_image = np.asanyarray(ir_frame.get_data())
        ir_image = markers_ir_image.astype(np.uint8)

        # Apply a threshold to binarize the image
        _, binary_ir_image = cv2.threshold(ir_image, 150, 255, cv2.THRESH_BINARY)

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

        # Find the expected ir marker locations
        normalized_marker_pattern = mathstuff.marker_pattern()
        transformed_marker_pattern_3d_aligned = cv2.perspectiveTransform(normalized_marker_pattern.reshape(-1, 1, 2), self.calibration_data.h_aligned).reshape(-1, 2)
        transformed_marker_pattern_3d_aligned = cast(np.ndarray[Any, np.dtype[np.float32]], transformed_marker_pattern_3d_aligned)
        transformed_marker_pattern_3d_aligned = np.hstack([transformed_marker_pattern_3d_aligned, np.zeros((len(normalized_marker_pattern), 1), dtype=np.float32)])
        expected_marker_pattern_aligned = mathstuff.apply_transformation(transformed_marker_pattern_3d_aligned, np.linalg.inv(self.calibration_data.xy_transformation_matrix_aligned))

        if len(detected_markers_2d) < len(expected_marker_pattern_aligned):
            print("Number of detected IR blobs is less than expected.")
            return False
        
        detected_markers_3d = []
        for point in detected_markers_2d:
            point_3d = mathstuff.approximate_intersection(self.calibration_data.plane, self.calibration_data.intrin, point[0], point[1], 0, 1000)
            detected_markers_3d.append(point_3d)
        
        detected_markers_3d_aligned = [self.calibration_data.align_transform_mtx @ point for point in detected_markers_3d]

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

        detected_marker_pattern_aligned_transformed = mathstuff.apply_transformation(np.array(detected_marker_pattern_aligned, dtype=np.float32), self.calibration_data.xy_transformation_matrix_aligned)

        self.calibration_data.detected_marker_pattern_2d = detected_marker_pattern_2d
        self.calibration_data.detected_marker_pattern_aligned = detected_marker_pattern_aligned
        self.calibration_data.detected_marker_pattern_aligned_transformed = detected_marker_pattern_aligned_transformed

        return True

    def fail(self, msg: str) -> None:
        print(f"Failure: {msg}")
        self.instructions_label.setText(f"Autocalibration unsuccessful. Exiting in {self.fail_countdown_time}")
        if self.auto_progress:
            self.start_fail_countdown()
        self.stacked_layout.setCurrentWidget(self.initial_widget)

    def start(self) -> None:
        depth_sensor = self.pipeline_profile.get_device().first_depth_sensor()
        if depth_sensor.is_option_read_only(rs.option.exposure):
            print("Warning: Depth sensor exposure is read-only.")
        else:
            depth_sensor.set_option(rs.option.exposure, self.ir_low_exposure)

        self.data_thread = DataAcquisitionThread(self.pipeline)
        self.data_thread.data_updated.connect(self.process_frame)
        self.data_thread.start()

        if self.auto_progress:
            self.start_countdown()

        self.next_button.setVisible(True)

    def start_countdown(self) -> None:
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.start(1000)  # Trigger every second

    def update_countdown(self) -> None:
        self.remaining_time -= 1
        self.countdown_label.setText(str(self.remaining_time))
        if self.remaining_time <= 0:
            self.next()

    def resizeEvent(self, event: QEvent) -> None:
        super().resizeEvent(event)
        if hasattr(self, 'camera_pixmap_item') and self.camera_pixmap_item:
            self.canvas.setSceneRect(0, 0, self.width(), self.height())

    def process_frame(self, frames: rs.frame) -> None:
        ir_frame = frames.get_infrared_frame(0)
        if not ir_frame:
            return

        self.latest_ir_frame = ir_frame

        ir_image = np.asanyarray(ir_frame.get_data())
        ir_image = cv2.cvtColor(ir_image, cv2.COLOR_BGR2RGB)

        h, w, ch = ir_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(ir_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.canvas.viewport().size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.camera_pixmap_item.setPixmap(scaled_pixmap)

        self.camera_pixmap_item.setOffset(-self.camera_pixmap_item.pixmap().width() / 2, -self.camera_pixmap_item.pixmap().height() / 2)
        self.camera_pixmap_item.setPos(self.canvas.viewport().width() / 2, self.canvas.viewport().height() / 2)

    def stop_data_thread(self) -> None:
        if self.data_thread and self.data_thread.isRunning():
            self.data_thread.stop()
            self.data_thread.wait()

    def closeEvent(self, event: QEvent) -> None:
        self.stop_data_thread()
        super().closeEvent(event)
