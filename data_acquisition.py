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
            success, frames = self.pipeline.try_wait_for_frames()
            if success:
                self.data_updated.emit(frames)
            else:
                print("Failed to get frames (maybe doing blocking calculations)")

    def stop(self):
        self.running = False
