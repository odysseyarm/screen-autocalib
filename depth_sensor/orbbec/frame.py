import depth_sensor.interface.frame
import pyorbbecsdk
from . import utils
from . import stream_profile
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

    def get_format(self) -> depth_sensor.interface.frame.StreamFormat:
        match self._internal.get_format():
            case pyorbbecsdk.OBFormat.RGB:
                return depth_sensor.interface.frame.StreamFormat.RGB
            case pyorbbecsdk.OBFormat.BGR:
                return depth_sensor.interface.frame.StreamFormat.BGR
            case pyorbbecsdk.OBFormat.Z16:
                return depth_sensor.interface.frame.StreamFormat.Z16
            case pyorbbecsdk.OBFormat.Y8:
                return depth_sensor.interface.frame.StreamFormat.Y8
            case _:
                return depth_sensor.interface.frame.StreamFormat.UNKNOWN

    def set_format(self, format: depth_sensor.interface.frame.StreamFormat) -> None:
        match format:
            case depth_sensor.interface.frame.StreamFormat.RGB:
                self._converted = utils.frame_to_rgb_frame(self._internal) # type: ignore
            case depth_sensor.interface.frame.StreamFormat.BGR:
                self._converted = utils.frame_to_bgr_image(self._internal) # type: ignore
            case depth_sensor.interface.frame.StreamFormat.Z16:
                raise ValueError("Unsupported")
            case depth_sensor.interface.frame.StreamFormat.Y8:
                raise ValueError("Unsupported")
            case _:
                raise ValueError("Unknown format")

    def get_profile(self) -> stream_profile.StreamProfile:
        return stream_profile.StreamProfile(self._internal.get_stream_profile())

class DepthFrame:
    _internal: pyorbbecsdk.DepthFrame
    _reshaped: npt.NDArray[np.float32]

    def __init__(self, frame: pyorbbecsdk.DepthFrame):
        self._internal = frame
        self._reshaped = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape((frame.get_height(), frame.get_width())).astype(np.float32)

    def get_data(self) -> npt.NDArray[np.float32]:
        return self._reshaped

    def get_depth_scale(self) -> float:
        return self._internal.get_depth_scale()

    def get_distance(self, x: int, y: int) -> float:
        # meters
        ret = self._internal.get_depth_scale() * 0.001 * self.get_data()[y, x]
        return ret

    def get_profile(self) -> stream_profile.StreamProfile:
        return stream_profile.StreamProfile(self._internal.get_stream_profile())
    
    def get_width(self) -> int:
        # not a bug, this is intentional
        return self._internal.get_height()

    def get_height(self) -> int:
        # not a bug, this is intentional
        return self._internal.get_width()

class CompositeFrame:
    _internal: pyorbbecsdk.FrameSet
    def __init__(self, frameset: pyorbbecsdk.FrameSet):
        self._internal = frameset
    def get_color_frame(self) -> depth_sensor.interface.frame.ColorFrame:
        return ColorFrame(self._internal.get_color_frame())
    def get_depth_frame(self) -> depth_sensor.interface.frame.DepthFrame:
        return DepthFrame(self._internal.get_depth_frame())
    def get_infrared_frame(self, index: int = 0) -> depth_sensor.interface.frame.InfraredFrame:
        if index != 0:
            raise ValueError("Only one IR frame is available")
        return IRFrame(self._internal.get_ir_frame())
