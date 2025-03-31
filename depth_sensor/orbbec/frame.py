import depth_sensor.interface.frame
import pyorbbecsdk
from . import utils
from . import stream_profile
from typing import Optional
import numpy.typing as npt
import numpy as np
import cv2
import cv2.typing

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

    def get_width(self) -> int:
        return self._internal.get_width()

    def get_height(self) -> int:
        return self._internal.get_height()

class IRFrame:
    _internal: pyorbbecsdk.VideoFrame
    _reshaped: npt.NDArray[np.uint8|np.uint16]
    _converted: Optional[cv2.typing.MatLike] = None

    def __init__(self, frame: pyorbbecsdk.VideoFrame):
        self._internal = frame
        ir_data: npt.NDArray[np.uint8] = np.asanyarray(frame.get_data())
        data_type: type
        image_dtype: int
        match self.get_format():
            case depth_sensor.interface.frame.StreamFormat.Y8:
                data_type = np.uint8
                image_dtype = cv2.CV_8UC1
                ir_data = np.reshape(ir_data, (frame.get_height(), frame.get_width()))
                max_data = 255
            case depth_sensor.interface.frame.StreamFormat.MJPG:
                data_type = np.uint8
                image_dtype = cv2.CV_8UC1
                ir_data = cv2.imdecode(ir_data, cv2.IMREAD_UNCHANGED) # type: ignore
                ir_data = np.reshape(ir_data, (frame.get_height(), frame.get_width()))
                max_data = 255
            case _:
                data_type = np.uint16
                image_dtype = cv2.CV_16UC1
                ir_data = np.frombuffer(ir_data, dtype=np.uint16) # type: ignore
                max_data = 65535
        cv2.normalize(ir_data, ir_data, 0, max_data, cv2.NORM_MINMAX, dtype=image_dtype)
        self._reshaped = ir_data.astype(data_type)

    def get_data(self) -> npt.NDArray[np.uint8|np.uint16]|cv2.typing.MatLike:
        if self._converted is None:
            return self._reshaped
        else:
            return self._converted

    def get_format(self) -> depth_sensor.interface.frame.StreamFormat:
        match self._internal.get_format():
            case pyorbbecsdk.OBFormat.Y8:
                return depth_sensor.interface.frame.StreamFormat.Y8
            case pyorbbecsdk.OBFormat.MJPG:
                return depth_sensor.interface.frame.StreamFormat.MJPG
            case _:
                return depth_sensor.interface.frame.StreamFormat.UNKNOWN

    def set_format(self, format: depth_sensor.interface.frame.StreamFormat) -> None:
        match format:
            case depth_sensor.interface.frame.StreamFormat.RGB:
                self._converted = cv2.cvtColor(self._reshaped, cv2.COLOR_GRAY2RGB)
            case _:
                raise ValueError("Unsupported")

    def get_profile(self) -> stream_profile.StreamProfile:
        return stream_profile.StreamProfile(self._internal.get_stream_profile())

    def get_width(self) -> int:
        return self._internal.get_width()

    def get_height(self) -> int:
        return self._internal.get_height()

class DepthFrame:
    _internal: pyorbbecsdk.DepthFrame
    _reshaped: npt.NDArray[np.uint16]

    def __init__(self, frame: pyorbbecsdk.DepthFrame):
        self._internal = frame
        self._reshaped = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape((frame.get_height(), frame.get_width()))

    def get_data(self) -> npt.NDArray[np.uint16]:
        return self._reshaped

    def get_distance(self, x: int, y: int) -> np.float32:
        # meters
        ret = np.float32(self._internal.get_depth_scale()) * np.float32(0.001) * np.float32(self.get_data()[y, x])
        return ret

    def get_profile(self) -> stream_profile.StreamProfile:
        return stream_profile.StreamProfile(self._internal.get_stream_profile())

    def get_width(self) -> int:
        return self._internal.get_width()

    def get_height(self) -> int:
        return self._internal.get_height()
    
    def get_format(self) -> depth_sensor.interface.frame.StreamFormat:
        raise NotImplementedError
    
    def set_format(self, format: depth_sensor.interface.frame.StreamFormat) -> None:
        raise NotImplementedError

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
        return IRFrame(self._internal.get_frame_by_type(pyorbbecsdk.OBFrameType.LEFT_IR_FRAME).as_video_frame())
