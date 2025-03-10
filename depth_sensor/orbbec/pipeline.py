import pyorbbecsdk

from . import frames
import depth_sensor.interface.frames

from typing import Optional

class Pipeline:
    _internal: pyorbbecsdk.Pipeline

    def __init__(self):
        self._internal = pyorbbecsdk.Pipeline()
    
    def try_wait_for_frames(self, timeout_ms: int = 5000) -> Optional[depth_sensor.interface.frames.CompositeFrame]:
        frameset = self._internal.wait_for_frames(timeout_ms)
        if not frameset:
            return None
        return frames.CompositeFrame(frameset)
    
    def start(self) -> None:
        config = pyorbbecsdk.Config()
        profile_list = self._internal.get_stream_profile_list(pyorbbecsdk.OBSensorType.COLOR_SENSOR)
        color_profile: pyorbbecsdk.VideoStreamProfile = profile_list.get_video_stream_profile(1920, 0, pyorbbecsdk.OBFormat.RGB, 30)
        config.enable_stream(color_profile)

        self._internal.start(config) # type: ignore
