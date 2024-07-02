from typing import Callable
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, QTimer

class Page1(QWidget):
    def __init__(self, parent: QWidget, next_page: Callable[[], None], exit_application: Callable[[], None]) -> None:
        super().__init__(parent)
        self.next_page = next_page
        self.exit_application = exit_application
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.status_label = QLabel("Waiting for RealSense camera to be connected...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        self.retry_button = QPushButton("Retry")
        self.retry_button.clicked.connect(self.check_camera_connection)
        layout.addWidget(self.retry_button)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.exit_application)
        layout.addWidget(self.exit_button)

        self.check_camera_connection()

    def check_camera_connection(self) -> None:
        try:
            import pyrealsense2 as rs
            context = rs.context()
            if len(context.query_devices()) > 0:
                self.status_label.setText("RealSense camera connected.")
                QTimer.singleShot(1000, self.next_page)  # Move to the next page after 1 second
            else:
                self.status_label.setText("No RealSense camera detected. Please connect the camera and press Retry.")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
