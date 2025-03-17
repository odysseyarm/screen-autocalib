import depth_sensor.interface.frame
import pyrealsense2
from . import stream_profile
import numpy.typing as npt
import numpy as np
import cv2
from typing import Optional

class ColorFrame:
    _internal: pyrealsense2.video_frame
    _converted: Optional[cv2.typing.MatLike] = None

    def __init__(self, frame: pyrealsense2.video_frame):
        self._internal = frame

    def get_data(self) -> npt.ArrayLike:
        frame: npt.ArrayLike
        if self._converted is None:
            frame = np.asanyarray(self._internal)
        else:
            frame = self._converted
        return frame

    def get_format(self) -> depth_sensor.interface.frame.StreamFormat:
        match self._internal.get_profile().format:
            case pyrealsense2.format.rgb8:
                return depth_sensor.interface.frame.StreamFormat.RGB
            case pyrealsense2.format.bgr8:
                return depth_sensor.interface.frame.StreamFormat.BGR
            case pyrealsense2.format.z16:
                return depth_sensor.interface.frame.StreamFormat.Z16
            case pyrealsense2.format.y8:
                return depth_sensor.interface.frame.StreamFormat.Y8
            case _:
                return depth_sensor.interface.frame.StreamFormat.UNKNOWN

    def set_format(self, format: depth_sensor.interface.frame.StreamFormat) -> None:
        match format:
            case depth_sensor.interface.frame.StreamFormat.RGB:
                self._converted = np.asanyarray(self._internal)
            case depth_sensor.interface.frame.StreamFormat.BGR:
                self._converted = cv2.cvtColor(np.asanyarray(self._internal.get_data()), cv2.COLOR_RGB2BGR)
            case depth_sensor.interface.frame.StreamFormat.Z16:
                raise ValueError("Unsupported")
            case depth_sensor.interface.frame.StreamFormat.Y8:
                raise ValueError("Unsupported")
            case _:
                raise ValueError("Unknown format")

    def get_profile(self) -> stream_profile.StreamProfile:
        return stream_profile.StreamProfile(self._internal.get_profile())

    def get_width(self) -> int:
        return self._internal.get_width()

    def get_height(self) -> int:
        return self._internal.get_height()

class IRFrame:
    _internal: pyrealsense2.video_frame
    _reshaped: npt.NDArray[np.uint8|np.uint16]
    _converted: Optional[cv2.typing.MatLike] = None

    def __init__(self, frame: pyrealsense2.video_frame):
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
        match self._internal.get_profile().format:
            case pyrealsense2.format.y8:
                return depth_sensor.interface.frame.StreamFormat.Y8
            case pyrealsense2.format.mjpeg:
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
        return stream_profile.StreamProfile(self._internal.get_profile())

    def get_width(self) -> int:
        return self._internal.get_width()

    def get_height(self) -> int:
        return self._internal.get_height()

class DepthFrame:
    _internal: pyrealsense2.depth_frame
    _reshaped: npt.NDArray[np.uint16]

    def __init__(self, frame: pyrealsense2.depth_frame):
        self._internal = frame
        self._reshaped = np.frombuffer(np.asanyarray(frame.get_data()), dtype=np.uint16).reshape((frame.get_height(), frame.get_width()))

    def get_data(self) -> npt.NDArray[np.uint16]:
        return self._reshaped

    def get_distance(self, x: int, y: int) -> float:
        # meters
        return self._internal.get_distance(x, y) * 0.001

    def get_profile(self) -> stream_profile.StreamProfile:
        return stream_profile.StreamProfile(self._internal.get_profile())

    def get_width(self) -> int:
        return self._internal.get_width()

    def get_height(self) -> int:
        return self._internal.get_height()

    def get_format(self) -> depth_sensor.interface.frame.StreamFormat:
        raise NotImplementedError

    def set_format(self, format: depth_sensor.interface.frame.StreamFormat) -> None:
        raise NotImplementedError

class CompositeFrame:
    _internal: pyrealsense2.composite_frame
    def __init__(self, frameset: pyrealsense2.composite_frame):
        self._internal = frameset
    def get_color_frame(self) -> depth_sensor.interface.frame.ColorFrame:
        return ColorFrame(self._internal.get_color_frame())
    def get_depth_frame(self) -> depth_sensor.interface.frame.DepthFrame:
        return DepthFrame(self._internal.get_depth_frame())
    def get_infrared_frame(self, index: int = 0) -> depth_sensor.interface.frame.InfraredFrame:
        return IRFrame(self._internal.get_infrared_frame(index))
