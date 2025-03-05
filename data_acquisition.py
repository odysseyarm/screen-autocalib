from PySide6.QtCore import QThread, Signal
import pyrealsense2 as rs
from typing import Optional

class DataAcquisitionThread(QThread):
    data_updated = Signal(rs.frame)  # Signal to emit updated frames

    def __init__(self, pipeline: rs.pipeline, parent: Optional[object] = None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames()
                self.data_updated.emit(frames)
            except Exception as e:
                print(f"Warning: {e}")

    def stop(self):
        self.running = False
        self.wait()
