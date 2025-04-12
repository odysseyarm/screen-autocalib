import pyrealsense2 as rs
import depth_sensor.interface.pipeline
import typing

class Pipeline:
    _internal: rs.pipeline

    _hdr_merge_filter: typing.Optional[rs.hdr_merge] = None
    _temporal_filter: typing.Optional[rs.temporal_filter] = None
    _spatial_filter: typing.Optional[rs.spatial_filter] = None

    _running: bool = False

    _config: rs.config

    def __init__(self, _pipeline: rs.pipeline, config: rs.config):
        self._internal = _pipeline
        self._config = config

    def try_wait_for_frames(self, timeout_ms: int = 5000) -> typing.Optional[rs.composite_frame]:
        # there's no way to see if the pipeline is running...
        try:
            success, frameset = self._internal.try_wait_for_frames(timeout_ms)
            if not success:
                return None
            # hack to get around receiving frames when not all streams are ready
            if frameset.get_color_frame() is None or frameset.get_infrared_frame() is None: # type: ignore
                return None
            return frameset
        except Exception: # I don't remember the right one
            return None
    
    def start(self) -> None:
        if not self._running:
            self._internal.start(self._config)
            self._running = True

    def stop(self) -> None:
        if self._running:
            self._internal.stop()
            self._running = False
        return

    def enable_stream(self, stream: depth_sensor.interface.pipeline.Stream, format: depth_sensor.interface.pipeline.frame.StreamFormat, framerate: int = 30) -> None:
        if stream == depth_sensor.interface.pipeline.Stream.COLOR:
            self._config.enable_stream(rs.stream.color, format_to_rs_format(format), framerate)
        elif stream == depth_sensor.interface.pipeline.Stream.DEPTH:
            self._config.enable_stream(rs.stream.depth, format_to_rs_format(format), framerate)
        elif stream == depth_sensor.interface.pipeline.Stream.INFRARED:
            self._config.enable_stream(rs.stream.infrared, format_to_rs_format(format), framerate)
        else:
            raise ValueError(f"Unsupported stream type: {stream}")
    
    def filter_supported(self, filter: depth_sensor.interface.pipeline.Filter) -> bool:
        if filter == depth_sensor.interface.pipeline.Filter.HDR_MERGE:
            return self.hdr_supported()
        elif filter == depth_sensor.interface.pipeline.Filter.TEMPORAL:
            return True
        elif filter == depth_sensor.interface.pipeline.Filter.SPATIAL:
            return True
        else:
            return False

    def filters_process(self, frameset: rs.composite_frame, filters: depth_sensor.interface.pipeline.Filter) -> typing.Optional[rs.composite_frame]:
        if depth_sensor.interface.pipeline.Filter.HDR_MERGE in filters:
            if self._hdr_merge_filter is None:
                self._hdr_merge_filter = rs.hdr_merge()
            frameset = self._hdr_merge_filter.process(frameset.as_frameset())

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
    
    def hdr_supported(self):
        return True

    def set_hdr_enabled(self, enabled: bool):
        depth_sensor = self._internal.get_active_profile().get_device().first_depth_sensor()
        if not depth_sensor.is_option_read_only(rs.option.hdr_enabled):
            depth_sensor.set_option(rs.option.hdr_enabled, enabled)
            depth_sensor.set_option(rs.option.enable_auto_exposure, True)

    def set_ir_exposure(self, exposure: int):
        depth_sensor = self._internal.get_active_profile().get_device().first_depth_sensor()
        if not depth_sensor.is_option_read_only(rs.option.exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, False)
            depth_sensor.set_option(rs.option.exposure, exposure)

def format_to_rs_format(format: depth_sensor.interface.pipeline.frame.StreamFormat) -> rs.format:
    if format == depth_sensor.interface.pipeline.frame.StreamFormat.RGB:
        return rs.format.rgb8
    elif format == depth_sensor.interface.pipeline.frame.StreamFormat.BGR:
        return rs.format.bgr8
    elif format == depth_sensor.interface.pipeline.frame.StreamFormat.Z16:
        return rs.format.z16
    elif format == depth_sensor.interface.pipeline.frame.StreamFormat.Y8:
        return rs.format.y8
    elif format == depth_sensor.interface.pipeline.frame.StreamFormat.MJPG:
        return rs.format.mjpeg
    else:
        raise ValueError(f"Unsupported stream format: {format}")
