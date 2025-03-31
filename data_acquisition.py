from PySide6.QtCore import QRunnable, QThreadPool, Signal, QObject
from typing import Optional
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

    class Signals(QObject):
        def __init__(self):
            super().__init__()
        ob_accel_updated = Signal(pyorbbecsdk.AccelFrame)
        ob_gyro_updated = Signal(pyorbbecsdk.GyroFrame)

    signals = Signals()
    threadpool: QThreadPool
    frame_processor: FrameProcessor
    start_pipeline: bool

    _ob_accel: Optional[pyorbbecsdk.Sensor] = None
    _ob_accel_profile: Optional[pyorbbecsdk.StreamProfile] = None
    _ob_gyro: Optional[pyorbbecsdk.Sensor] = None
    _ob_gyro_profile: Optional[pyorbbecsdk.StreamProfile] = None

    def __init__(self, pipeline: depth_sensor.interface.pipeline.Pipeline, threadpool: QThreadPool, start_pipeline: bool = False, ob_accel: Optional[pyorbbecsdk.Sensor] = None, ob_gyro: Optional[pyorbbecsdk.Sensor] = None):
        super().__init__()
        self.daemon = True
        self.pipeline = pipeline
        self.threadpool = threadpool
        self.frame_processor = FrameProcessor(pipeline)
        self.start_pipeline = start_pipeline
        self.running = False
        if ob_accel is not None:
            accel_profile_list: pyorbbecsdk.StreamProfileList = ob_accel.get_stream_profile_list()
            accel_profile: pyorbbecsdk.StreamProfile = accel_profile_list.get_stream_profile_by_index(0)
            self._ob_accel_profile = accel_profile
            self._ob_accel = ob_accel
        if ob_gyro is not None:
            gyro_profile_list: pyorbbecsdk.StreamProfileList = ob_gyro.get_stream_profile_list()
            gyro_profile: pyorbbecsdk.StreamProfile = gyro_profile_list.get_stream_profile_by_index(0)
            self._ob_gyro_profile = gyro_profile
            self._ob_gyro = ob_gyro
    
    def _on_accel_frame_callback(self, frame: pyorbbecsdk.AccelFrame):
        self.signals.ob_accel_updated.emit(frame)
    def _on_gyro_frame_callback(self, frame: pyorbbecsdk.GyroFrame):
        self.signals.ob_gyro_updated.emit(frame)

    def run(self):
        self.running = True
        self.threadpool.start(self.frame_processor)
        if self.start_pipeline:
            print("starting pipeline")
            self.pipeline.start()
        if self._ob_accel_profile is not None:
            self._ob_accel.start(self._ob_accel_profile, self._on_accel_frame_callback) # type: ignore
        if self._ob_gyro_profile is not None:
            self._ob_gyro.start(self._ob_gyro_profile, self._on_gyro_frame_callback) # type: ignore
        while self.running:
            frames = self.pipeline.try_wait_for_frames()
            if frames is not None:
                self.frame_processor.latest_frameset = frames
            else:
                # print("Failed to get frames (is pipeline running?)")
                pass

    def stop(self):
        self.running = False
        self.frame_processor.stop()
        if self._ob_accel is not None:
            self._ob_accel.stop()
        if self._ob_gyro is not None:
            self._ob_gyro.stop()
