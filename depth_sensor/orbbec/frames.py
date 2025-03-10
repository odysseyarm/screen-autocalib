import depth_sensor.interface.frames
import pyorbbecsdk
from . import utils
from typing import Optional
import numpy.typing as npt
import numpy as np

class ColorFrame:
    _internal: pyorbbecsdk.VideoFrame
    _converted: Optional[pyorbbecsdk.VideoFrame] = None

    def __init__(self, frame: pyorbbecsdk.VideoFrame):
        self._internal = frame

    def get_data(self) -> npt.ArrayLike:
        frame: pyorbbecsdk.VideoFrame
        if self._converted is None:
            frame = self._internal
        else:
            frame = self._converted
        width = frame.get_width()
        height = frame.get_height()
        image = np.reshape(frame.get_data(), (height, width, 3)) # type: ignore
        return np.array(image, dtype=np.uint8)

    def get_format(self) -> depth_sensor.interface.frames.StreamFormat:
        match self._internal.get_format():
            case pyorbbecsdk.OBFormat.RGB:
                return depth_sensor.interface.frames.StreamFormat.RGB
            case pyorbbecsdk.OBFormat.BGR:
                return depth_sensor.interface.frames.StreamFormat.BGR
            case pyorbbecsdk.OBFormat.Z16:
                return depth_sensor.interface.frames.StreamFormat.Z16
            case pyorbbecsdk.OBFormat.Y8:
                return depth_sensor.interface.frames.StreamFormat.Y8
            case _:
                return depth_sensor.interface.frames.StreamFormat.UNKNOWN

    def set_format(self, format: depth_sensor.interface.frames.StreamFormat) -> None:
        match format:
            case depth_sensor.interface.frames.StreamFormat.RGB:
                self._converted = utils.frame_to_rgb_frame(self._internal) # type: ignore
            case depth_sensor.interface.frames.StreamFormat.BGR:
                self._converted = utils.frame_to_bgr_image(self._internal) # type: ignore
            case depth_sensor.interface.frames.StreamFormat.Z16:
                raise ValueError("Unsupported")
            case depth_sensor.interface.frames.StreamFormat.Y8:
                raise ValueError("Unsupported")
            case _:
                raise ValueError("Unknown format")

class CompositeFrame:
    _internal: pyorbbecsdk.FrameSet
    def __init__(self, frameset: pyorbbecsdk.FrameSet):
        self._internal = frameset
    def get_color_frame(self) -> depth_sensor.interface.frames.ColorFrame:
        return ColorFrame(self._internal.get_color_frame())
    def get_depth_frame(self) -> depth_sensor.interface.frames.DepthFrame:
        return self._internal.get_depth_frame()
    def get_infrared_frame(self, index: int = 0) -> depth_sensor.interface.frames.InfraredFrame:
        if index != 0:
            raise ValueError("Only one IR frame is available")
        return self._internal.get_ir_frame()
