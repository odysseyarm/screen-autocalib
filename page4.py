import cv2
import pyrealsense2 as rs
from typing import Any, List, Dict, Literal, Tuple, Callable, Optional, cast
from PySide6.QtCore import QTimer, Qt, QEvent, QThreadPool
from PySide6.QtGui import QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QPushButton, QHBoxLayout, QLabel
from platformdirs import user_data_dir
from pathlib import Path
from data_acquisition import DataAcquisitionThread
import numpy as np
from calibration_data import CalibrationData
import mathstuff
from depth_sensor.interface import pipeline as ds_pipeline
from depth_sensor.interface import frame as ds_frame

class MainWindow(QWidget):
    pipeline: ds_pipeline.Pipeline[Any]
    threadpool: QThreadPool
    calibration_data: CalibrationData

class Page4(QWidget):
    pipeline: ds_pipeline.Pipeline[Any]

    def __init__(self, parent: MainWindow, next_page: Callable[[], None], exit_application: Callable[[], None], auto_progress: bool, ir_low_exposure: float, enable_hdr: bool) -> None:
        super().__init__(parent)
        self.enable_hdr = enable_hdr
        self.main_window = parent
        self.exit_application = exit_application
        self.motion_support = False
        self.pipeline_profile = None
        self.auto_progress = auto_progress
        self.countdown_timer: Optional[QTimer] = None
        self.latest_ir_frame: Optional[ds_frame.InfraredFrame] = None
        self.data_thread: Optional[DataAcquisitionThread] = None
        self.ir_low_exposure = ir_low_exposure
        self.remaining_time = 30
        self.next_page = next_page
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
        self.next_button.setShortcut(QKeySequence("Ctrl+N"))
        button_layout.addWidget(self.next_button)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.exit_application)
        self.exit_button.setShortcut(QKeySequence("Ctrl+Q"))
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

    def next(self) -> None:
        if self.countdown_timer is not None:
            self.countdown_timer.stop()
        if self.latest_ir_frame is None:
            self.fail("No frames for marker detection")
            self.remaining_time = 30
            if self.countdown_timer is not None:
                self.countdown_timer.start(1000)
        elif not self.detect_markers(self.latest_ir_frame):
            self.fail("Marker detection failed")
            self.remaining_time = 30
            if self.countdown_timer is not None:
                self.countdown_timer.start(1000)
        else:
            self.next_page()

    def detect_markers(self, ir_frame: ds_frame.InfraredFrame) -> bool:
        # Detect IR blobs in the depth image
        ir_image = cv2.cvtColor(np.asanyarray(ir_frame.get_data()), cv2.COLOR_RGB2GRAY)

        # Apply a threshold to binarize the image
        _, binary_ir_image = cv2.threshold(ir_image, 150, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_ir_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_markers_2d = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 10:  # Filter small contours
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = np.float32(M['m10']) / np.float32(M['m00'])
                    cy = np.float32(M['m01']) / np.float32(M['m00'])
                    detected_markers_2d.append([cx, cy])

        # Find the expected ir marker locations
        normalized_marker_pattern = mathstuff.marker_pattern()
        transformed_marker_pattern_3d_aligned = cv2.perspectiveTransform(normalized_marker_pattern.reshape(-1, 1, 2), self.main_window.calibration_data.h_aligned).reshape(-1, 2)
        transformed_marker_pattern_3d_aligned = cast(np.ndarray[Any, np.dtype[np.float32]], transformed_marker_pattern_3d_aligned)
        transformed_marker_pattern_3d_aligned = np.hstack([transformed_marker_pattern_3d_aligned, np.zeros((len(normalized_marker_pattern), 1), dtype=np.float32)])
        expected_marker_pattern_aligned = mathstuff.apply_transformation(transformed_marker_pattern_3d_aligned, np.linalg.inv(self.main_window.calibration_data.xy_transformation_matrix_aligned))

        if len(detected_markers_2d) < len(expected_marker_pattern_aligned):
            print("Number of detected IR blobs is less than expected.")
            return False

        self.pipeline.stop()

        print(f"Detected {len(detected_markers_2d)} IR blobs, now approximating 3D positions...")

        detected_markers_3d = []
        for point in detected_markers_2d:
            # print(f"Approximating 3D position for IR blob at {point}")
            point_3d = mathstuff.approximate_intersection(self.main_window.calibration_data.depth_plane, ir_frame.get_profile().as_video_stream_profile().get_intrinsic(), point[0], point[1], 0, 1000)
            temp = np.array([point_3d[0], point_3d[1], point_3d[2], 1])
            temp = self.main_window.calibration_data.depth_to_color.transform @ temp
            point_3d = temp[:3]
            detected_markers_3d.append(point_3d)

        detected_markers_3d_aligned = [self.main_window.calibration_data.align_transform_mtx @ point for point in detected_markers_3d]

        print("Finding closest detected blobs to expected positions...")
        # Find the closest detected blobs to the expected positions
        detected_marker_pattern_aligned = []
        detected_marker_pattern_2d = []
        for point in expected_marker_pattern_aligned:
            distances = [np.linalg.norm(point - detected_marker) for detected_marker in detected_markers_3d_aligned]
            closest_index = np.argmin(distances)
            detected_marker_pattern_aligned.append(detected_markers_3d_aligned[closest_index])
            projected_point = mathstuff.project_point_to_pixel_with_distortion(self.main_window.calibration_data.color_intrinsics, np.linalg.inv(self.main_window.calibration_data.align_transform_mtx) @ detected_markers_3d_aligned[closest_index])
            detected_marker_pattern_2d.append(projected_point)
            detected_markers_3d_aligned.pop(closest_index)

        detected_marker_pattern_aligned_transformed = mathstuff.apply_transformation(np.array(detected_marker_pattern_aligned, dtype=np.float32), self.main_window.calibration_data.xy_transformation_matrix_aligned)

        self.main_window.calibration_data.detected_marker_pattern_2d = detected_marker_pattern_2d
        self.main_window.calibration_data.detected_marker_pattern_aligned = detected_marker_pattern_aligned
        self.main_window.calibration_data.detected_marker_pattern_aligned_transformed = detected_marker_pattern_aligned_transformed

        return True

    def fail(self, msg: str) -> None:
        print(f"Failure: {msg}")
        self.instructions_label.setText(f"{msg}")

    def start(self) -> None:
        # todo
        # if depth_sensor.is_option_read_only(rs.option.hdr_enabled):
        #     print("Warning: Depth sensor HDR is read-only.")
        # else:
        #     depth_sensor.set_option(rs.option.hdr_enabled, 0)
        # if depth_sensor.is_option_read_only(rs.option.exposure):
        #     print("Warning: Depth sensor exposure is read-only.")
        # else:
        #     depth_sensor.set_option(rs.option.exposure, self.ir_low_exposure)

        self.pipeline.stop()
        self.data_thread.frame_processor.set_filters(None)
        self.pipeline.start()

        if self.enable_hdr:
            self.pipeline.set_hdr_enabled(False)

        self.pipeline.set_ir_exposure(int(self.ir_low_exposure))

        # self.data_thread = DataAcquisitionThread(self.pipeline, self.main_window.threadpool)
        self.data_thread.frame_processor.signals.data_updated.connect(self.process_frame)
        # self.main_window.threadpool.start(self.data_thread)

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

    def process_frame(self, frames: ds_frame.CompositeFrame) -> None:
        ir_frame = frames.get_infrared_frame()
        if not ir_frame:
            return

        ir_frame.set_format(ds_frame.StreamFormat.RGB)
        ir_image = cv2.Mat(ir_frame.get_data())

        self.latest_ir_frame = ir_frame

        h, w, ch = ir_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(ir_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.canvas.viewport().size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.camera_pixmap_item.setPixmap(scaled_pixmap)

        self.camera_pixmap_item.setOffset(-self.camera_pixmap_item.pixmap().width() / 2, -self.camera_pixmap_item.pixmap().height() / 2)
        self.camera_pixmap_item.setPos(self.canvas.viewport().width() / 2, self.canvas.viewport().height() / 2)

    def closeEvent(self, event: QEvent) -> None:
        return
