import pyrealsense2 as rs
import numpy as np
import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap, QPen, QPainter
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QPushButton, QHBoxLayout, QLabel

class BracketOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pen = QPen(Qt.red, 5)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

    def paintEvent(self, event):
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

class Page1(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.canvas = QGraphicsView()
        self.scene = QGraphicsScene(self)
        self.canvas.setScene(self.scene)
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.canvas.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.canvas)

        self.pipeline = None
        self.camera_pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.camera_pixmap_item)

        self.bracket_overlay = BracketOverlay(self)
        self.bracket_overlay.resize(self.size())

        button_layout = QHBoxLayout()
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.main_window.next_page)
        self.next_button.setVisible(False)
        button_layout.addWidget(self.next_button)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.main_window.exit_application)
        button_layout.addWidget(self.exit_button)

        layout.addLayout(button_layout)

        instructions_label = QLabel("Move the RealSense camera such that the right angle brackets are in the corners of the preview.")
        instructions_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions_label)

        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_realsense_device)
        self.timer.start(1000)

    def check_realsense_device(self):
        context = rs.context()
        if len(context.query_devices()) > 0:
            self.start_steps()

    def start_steps(self):
        self.timer.stop()
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_camera_preview)
        self.video_timer.start(30)

        self.next_button.setVisible(True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'camera_pixmap_item') and self.camera_pixmap_item:
            self.canvas.setSceneRect(0, 0, self.width(), self.height())
            self.bracket_overlay.resize(self.size())
            self.update_camera_preview()

    def update_camera_preview(self):
        if not self.pipeline:
            return

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        h, w, ch = color_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(color_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.canvas.viewport().size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_pixmap_item.setPixmap(scaled_pixmap)

        self.camera_pixmap_item.setOffset(-self.camera_pixmap_item.pixmap().width() / 2, -self.camera_pixmap_item.pixmap().height() / 2)
        self.camera_pixmap_item.setPos(self.canvas.viewport().width() / 2, self.canvas.viewport().height() / 2)
