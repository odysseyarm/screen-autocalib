from PySide6.QtCore import QThread, Signal, QObject
from typing import Optional

import depth_sensor.interface.frames
import depth_sensor.interface.pipeline

class DataAcquisitionThread(QThread):
    data_updated = Signal(depth_sensor.interface.frames.CompositeFrame)  # Signal to emit updated frames

    def __init__(self, pipeline: depth_sensor.interface.pipeline.Pipeline, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            frames = self.pipeline.try_wait_for_frames()
            if frames is not None:
                self.data_updated.emit(frames)
            else:
                print("Failed to get frames (maybe doing blocking calculations)")

    def stop(self):
        self.running = False
