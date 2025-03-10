import depth_sensor.interface.frames
import pyrealsense2
import numpy.typing as npt

class ColorFrame:
    _internal: pyrealsense2.video_frame
    def __init__(self, frame: pyrealsense2.video_frame):
        self._internal = frame
    def get_data(self) -> npt.ArrayLike:
        return self._internal.get_data() # type: ignore
    def to_bgr(self) -> npt.ArrayLike:
        return self.get_data()

class CompositeFrame:
    _internal: pyrealsense2.composite_frame
    def __init__(self, frameset: pyrealsense2.composite_frame):
        self._internal = frameset
    def get_color_frame(self) -> depth_sensor.interface.frames.ColorFrame:
        return ColorFrame(self._internal.get_color_frame())
    def get_depth_frame(self) -> depth_sensor.interface.frames.DepthFrame:
        return self._internal.get_depth_frame() # type: ignore
    def get_infrared_frame(self, index: int = 0) -> depth_sensor.interface.frames.InfraredFrame:
        return self._internal.get_infrared_frame(index) # type: ignore
