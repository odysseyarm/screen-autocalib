import pyrealsense2 as rs
import depth_sensor.interface.frame
import depth_sensor.interface.pipeline
from . import frame
import typing

from functools import singledispatchmethod

class Pipeline:
    _internal: rs.pipeline

    _temporal_filter: typing.Optional[rs.temporal_filter] = None
    _spatial_filter: typing.Optional[rs.spatial_filter] = None

    _running: bool = False

    @singledispatchmethod
    def __init__(self):
        self._internal = rs.pipeline()

    @__init__.register
    def _(self, ctx: rs.context):
        self._internal = rs.pipeline(ctx)
    
    @__init__.register
    def _(self, _pipeline: rs.pipeline):
        self._internal = _pipeline

    def try_wait_for_frames(self, timeout_ms: int = 5000) -> typing.Optional[rs.composite_frame]:
        success, frameset = self._internal.try_wait_for_frames(timeout_ms)
        if not success:
            return None
        return frameset
    
    def start(self) -> None:
        if not self._running:
            self._internal.start()
            self._running = True

    def stop(self) -> None:
        # if it's stopped i can't grab extrinsics
        # if self._running:
        #     self._internal.stop()
        #     self._running = False
        return

    def filters_process(self, frameset: rs.composite_frame, filters: depth_sensor.interface.pipeline.Filter) -> typing.Optional[rs.composite_frame]:
        if depth_sensor.interface.pipeline.Filter.TEMPORAL in filters:
            if self._temporal_filter is None:
                self._temporal_filter = rs.temporal_filter()
                self._temporal_filter.set_option(rs.option.filter_smooth_alpha, .05)
                self._temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
                self._temporal_filter.set_option(rs.option.holes_fill, 7)
            frameset = self._temporal_filter.process(frameset.as_frameset())

        if depth_sensor.interface.pipeline.Filter.SPATIAL in filters:
            if self._spatial_filter is None:
                self._spatial_filter = rs.spatial_filter()
                self._spatial_filter.set_option(rs.option.filter_magnitude, 2)
                self._spatial_filter.set_option(rs.option.filter_smooth_alpha, .5)
                self._spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
            frameset = self._spatial_filter.process(frameset.as_frameset())

        if frameset is None: # type: ignore
            return None

        frameset = frameset.as_frameset()
        if frameset.get_color_frame() is None or frameset.get_infrared_frame() is None: # type: ignore
            return None

        return frameset
