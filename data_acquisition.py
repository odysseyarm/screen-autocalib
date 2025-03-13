from PySide6.QtCore import QThread, Signal, QObject
from typing import Optional, TypeVar
import threading
import time

import pyorbbecsdk
import pyrealsense2

import depth_sensor.interface.frame
import depth_sensor.interface.pipeline
import depth_sensor.orbbec
import depth_sensor.orbbec.frame
import depth_sensor.realsense

class FrameProcessor(QThread):
    data_updated = Signal(depth_sensor.interface.frame.CompositeFrame)  # Signal to emit updated frames
    filters: Optional[depth_sensor.interface.pipeline.Filter] = None

    def __init__(self, pipeline: depth_sensor.interface.pipeline.Pipeline):
        super().__init__()
        self.daemon = True
        self.latest_frameset = None
        self.lock = threading.Lock()
        self.pipeline = pipeline

    def run(self):
        self.running = True
        while self.running:
            with self.lock:
                if self.latest_frameset is not None:
                    if self.filters is not None:
                        processed_frameset = self.pipeline.filters_process(self.latest_frameset, self.filters)
                    else:
                        processed_frameset = self.latest_frameset
                    if isinstance(processed_frameset, pyorbbecsdk.FrameSet):
                        self.data_updated.emit(depth_sensor.orbbec.frame.CompositeFrame(processed_frameset))
                    else:
                        self.data_updated.emit(depth_sensor.realsense.frame.CompositeFrame(processed_frameset))
                    self.latest_frameset = None
            time.sleep(0.001)

    def update_frame(self, frameset: pyorbbecsdk.FrameSet|pyrealsense2.composite_frame):
        with self.lock:
            self.latest_frameset = frameset

    def stop(self):
        with self.lock:
            self.running = False
            self.latest_frameset = None

class DataAcquisitionThread(QThread):
    frame_processor: FrameProcessor

    def __init__(self, pipeline: depth_sensor.interface.pipeline.Pipeline, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.daemon = True
        self.pipeline = pipeline
        self.frame_processor = FrameProcessor(pipeline)
        self.running = False

    def run(self):
        self.running = True
        self.frame_processor.start()
        while self.running:
            frames = self.pipeline.try_wait_for_frames()
            if frames is not None:
                self.frame_processor.latest_frameset = frames
            else:
                print("Failed to get frames (maybe doing blocking calculations)")

    def stop(self):
        self.running = False
        self.frame_processor.stop()
