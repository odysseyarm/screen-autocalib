import pyorbbecsdk

from . import frame
from . import stream_profile
import depth_sensor.interface.frame
import depth_sensor.interface.pipeline
import depth_sensor.interface.stream_profile

from typing import Optional

class Pipeline:
    _internal: pyorbbecsdk.Pipeline

    _color_profile: pyorbbecsdk.VideoStreamProfile
    _depth_profile: pyorbbecsdk.VideoStreamProfile
    _ir_profile: pyorbbecsdk.VideoStreamProfile

    # _spatial_filter: Optional[pyorbbecsdk.SpatialAdvancedFilter]

    def __init__(self):
        self._internal = pyorbbecsdk.Pipeline()
        # self._spatial_filter = pyorbbecsdk.SpatialAdvancedFilter()
    
    def try_wait_for_frames(self, timeout_ms: int = 5000) -> Optional[depth_sensor.interface.frame.CompositeFrame]:
        frameset = self._internal.wait_for_frames(timeout_ms)
        if frameset is None: # type: ignore
            return None

        # hack to get around receiving frames when not all streams are ready
        if frameset.get_color_frame() is None or frameset.get_ir_frame() is None: # type: ignore
            return None

        return frame.CompositeFrame(frameset)
    
    def start(self) -> None:
        config = pyorbbecsdk.Config()

        profile_list = self._internal.get_stream_profile_list(pyorbbecsdk.OBSensorType.COLOR_SENSOR)
        self._color_profile = profile_list.get_video_stream_profile(1920, 0, pyorbbecsdk.OBFormat.RGB, 30)

        profile_list = self._internal.get_stream_profile_list(pyorbbecsdk.OBSensorType.IR_SENSOR)
        self._ir_profile = profile_list.get_video_stream_profile(1600, 0, pyorbbecsdk.OBFormat.Y8, 30)

        profile_list = self._internal.get_d2c_depth_profile_list(self._color_profile, pyorbbecsdk.OBAlignMode.SW_MODE)
        self._depth_profile = profile_list.get_default_video_stream_profile()

        config.enable_stream(self._color_profile)
        config.enable_stream(self._ir_profile)
        config.enable_stream(self._depth_profile)

        config.set_align_mode(pyorbbecsdk.OBAlignMode.SW_MODE)

        self._internal.start(config) # type: ignore
    
    def stop(self) -> None:
        self._internal.stop()

    def enable_stream(self, stream: depth_sensor.interface.pipeline.Stream, format: depth_sensor.interface.pipeline.frame.StreamFormat) -> None:
        pass

    def get_video_stream_profile(self, stream: depth_sensor.interface.pipeline.Stream) -> depth_sensor.interface.pipeline.stream_profile.VideoStreamProfile:
        if stream == depth_sensor.interface.pipeline.Stream.COLOR:
            return stream_profile.VideoStreamProfile(self._color_profile)
        elif stream == depth_sensor.interface.pipeline.Stream.DEPTH:
            return stream_profile.VideoStreamProfile(self._depth_profile)
        elif stream == depth_sensor.interface.pipeline.Stream.INFRARED:
            return stream_profile.VideoStreamProfile(self._ir_profile)
        else:
            raise ValueError(f"Unsupported stream: {stream}")
    
    def get_extrinsic_between(self, stream1: depth_sensor.interface.pipeline.Stream, stream2: depth_sensor.interface.pipeline.Stream) -> depth_sensor.interface.stream_profile.Extrinsic:
        stream1_profile = self.__get_profile(stream1)
        stream2_profile = self.__get_profile(stream2)
        return stream_profile.Extrinsic(stream1_profile.get_extrinsic_to(stream2_profile))
    
    def __get_profile(self, stream: depth_sensor.interface.pipeline.Stream) -> pyorbbecsdk.VideoStreamProfile:
        if stream == depth_sensor.interface.pipeline.Stream.COLOR:
            return self._color_profile
        elif stream == depth_sensor.interface.pipeline.Stream.DEPTH:
            return self._depth_profile
        elif stream == depth_sensor.interface.pipeline.Stream.INFRARED:
            return self._ir_profile
        raise ValueError("Unknown stream")
