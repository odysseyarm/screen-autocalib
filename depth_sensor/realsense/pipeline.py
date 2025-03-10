import pyrealsense2
import depth_sensor.interface.frames
from . import frames
import typing

from functools import singledispatch

class Pipeline:
    _internal: pyrealsense2.pipeline

    @singledispatch
    def __init__(self) -> None:
        self._internal = pyrealsense2.pipeline()

    @__init__.register
    def _(self, ctx: pyrealsense2.context) -> None:
        self._internal = pyrealsense2.pipeline(ctx)
    
    @__init__.register
    def _(self, _pipeline: pyrealsense2.pipeline) -> None:
        self._internal = _pipeline

    def try_wait_for_frames(self, timeout_ms: int = 5000) -> typing.Optional[depth_sensor.interface.frames.CompositeFrame]:
        success, frameset = self._internal.try_wait_for_frames(timeout_ms)
        if not success:
            return None
        return frames.CompositeFrame(frameset)
    
    def start(self) -> None:
        self._internal.start()
