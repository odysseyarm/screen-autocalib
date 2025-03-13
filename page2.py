from typing import Callable, Optional
import numpy as np
import cv2
from PySide6.QtCore import QTimer, Qt, QEvent
from PySide6.QtGui import QImage, QPixmap, QPen, QPainter, QPaintEvent, QResizeEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QPushButton, QHBoxLayout, QLabel
from data_acquisition import DataAcquisitionThread
from depth_sensor.interface import frame, pipeline

class MainWindow(QWidget):
    pipeline: Optional[pipeline.Pipeline]
    pass

class BracketOverlay(QWidget):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.pen = QPen(Qt.GlobalColor.red, 5)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setPen(self.pen)

        window_width = self.width()
        window_height = self.height()

        # Top-left corner
        painter.drawLine(0, 0, 50, 0)
        painter.drawLine(0, 0, 0, 50)

        # Top-right corner
        painter.drawLine(window_width - 50, 0, window_width - 1, 0)
        painter.drawLine(window_width - 1, 0, window_width - 1, 50)

        # Bottom-left corner
        painter.drawLine(0, window_height - 1, 50, window_height - 1)
        painter.drawLine(0, window_height - 50, 0, window_height - 1)

        # Bottom-right corner
        painter.drawLine(window_width - 50, window_height - 1, window_width - 1, window_height - 1)
        painter.drawLine(window_width - 1, window_height - 50, window_width - 1, window_height - 1)

class Page2(QWidget):
    def __init__(self, parent: MainWindow, next_page: Callable[[], None], exit_application: Callable[[], None], auto_progress: bool) -> None:
        super().__init__(parent)
        self.main_window = parent
        self.next_page = next_page
        self.main_window_exit_application = exit_application
        self.auto_progress = auto_progress
        self.pipeline: Optional[pipeline.Pipeline] = None
        self.data_thread: Optional[DataAcquisitionThread] = None
        self.countdown_timer: Optional[QTimer] = None
        self.remaining_time = 30
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

        self.bracket_overlay = BracketOverlay(self)
        self.bracket_overlay.resize(self.size())

        self.countdown_label = QLabel("30", self)
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.countdown_label.setVisible(False)  # Initially hidden
        layout.addWidget(self.countdown_label)

        button_layout = QHBoxLayout()
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.go_next)
        button_layout.addWidget(self.next_button)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.exit_application)
        button_layout.addWidget(self.exit_button)

        if self.auto_progress:
            self.next_button.setVisible(False)
            self.exit_button.setVisible(False)
            self.countdown_label.setVisible(True)  # Show countdown if auto_progress

        layout.addLayout(button_layout)

        instructions_label = QLabel("Move the RealSense camera such that the right angle brackets are in the corners of the preview.")
        instructions_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instructions_label)

        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_realsense_device)
        self.timer.start(1000)

    def check_realsense_device(self) -> None:
        print(self.main_window.pipeline)
        if self.main_window.pipeline is not None:
            self.pipeline = self.main_window.pipeline
            self.start_steps()

    def start_steps(self) -> None:
        self.timer.stop()

        assert self.pipeline is not None

        self.data_thread = DataAcquisitionThread(self.pipeline)
        self.data_thread.frame_processor.data_updated.connect(self.process_frame)
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
            self.go_next()
    
    def go_next(self) -> None:
        if self.countdown_timer is not None:
            self.countdown_timer.stop()
        if self.data_thread is not None:
            self.data_thread.stop()
        self.next_page()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if hasattr(self, 'camera_pixmap_item') and self.camera_pixmap_item:
            self.canvas.setSceneRect(0, 0, self.width(), self.height())
            self.bracket_overlay.resize(self.size())

    def process_frame(self, composite_frame: frame.CompositeFrame) -> None:
        color_frame = composite_frame.get_color_frame()
        if not color_frame:
            return

        color_frame.set_format(frame.StreamFormat.RGB)
        color_image = cv2.Mat(color_frame.get_data())

        h, w, ch = color_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(color_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

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

    def exit_application(self) -> None:
        self.stop_data_thread()

        # if hasattr(self, 'pipeline') and self.pipeline and self.pipeline is not None:
        #     try:
        #         self.pipeline.stop()
        #     except Exception as e:
        #         print(f"Error stopping pipeline: {e}")
        #     self.pipeline = None

        self.main_window_exit_application()
