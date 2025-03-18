from PySide6.QtCore import QRunnable, QThreadPool, Signal, QObject
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
import depth_sensor.realsense.frame

class FrameProcessor(QRunnable):

    class Signals(QObject):
        def __init__(self):
            super().__init__()
        data_updated = Signal(depth_sensor.interface.frame.CompositeFrame)  # Signal to emit updated frames

    signals = Signals()
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
                    if processed_frameset is not None:
                        if isinstance(processed_frameset, pyorbbecsdk.FrameSet):
                            self.signals.data_updated.emit(depth_sensor.orbbec.frame.CompositeFrame(processed_frameset))
                        else:
                            self.signals.data_updated.emit(depth_sensor.realsense.frame.CompositeFrame(processed_frameset))
                    self.latest_frameset = None
            time.sleep(0.001)

    def update_frame(self, frameset: pyorbbecsdk.FrameSet|pyrealsense2.composite_frame):
        with self.lock:
            self.latest_frameset = frameset

    def stop(self):
        with self.lock:
            self.running = False
            self.latest_frameset = None
    
    def set_filters(self, filters: Optional[depth_sensor.interface.pipeline.Filter]):
        with self.lock:
            self.filters = filters

class DataAcquisitionThread(QRunnable):
    threadpool: QThreadPool
    frame_processor: FrameProcessor
    start_pipeline: bool

    def __init__(self, pipeline: depth_sensor.interface.pipeline.Pipeline, threadpool: QThreadPool, start_pipeline: bool = False, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.daemon = True
        self.pipeline = pipeline
        self.threadpool = threadpool
        self.frame_processor = FrameProcessor(pipeline)
        self.start_pipeline = start_pipeline
        self.running = False

    def run(self):
        self.running = True
        self.threadpool.start(self.frame_processor)
        if self.start_pipeline:
            print("starting pipeline")
            self.pipeline.start()
        while self.running:
            frames = self.pipeline.try_wait_for_frames()
            if frames is not None:
                self.frame_processor.latest_frameset = frames
            else:
                # print("Failed to get frames (is pipeline running?)")
                pass

    def stop(self):
        return
    #     self.running = False
    #     self.frame_processor.stop()
