import pyorbbecsdk

import depth_sensor.interface.frame
import depth_sensor.interface.pipeline
import depth_sensor.interface.stream_profile

from typing import Optional

class Pipeline:
    _internal: pyorbbecsdk.Pipeline

    _hdr_merge_filter: Optional[pyorbbecsdk.HDRMergeFilter] = None
    _noise_removal_filter: Optional[pyorbbecsdk.NoiseRemovalFilter] = None
    _temporal_filter: Optional[pyorbbecsdk.TemporalFilter] = None
    _spatial_filter: Optional[pyorbbecsdk.SpatialAdvancedFilter] = None
    _align_d2c_filter: Optional[pyorbbecsdk.AlignFilter] = None

    _running: bool = False

    _config: pyorbbecsdk.Config

    def __init__(self, ob_pipeline: pyorbbecsdk.Pipeline, config: pyorbbecsdk.Config):
        self._internal = ob_pipeline
        self._config = config
        self._internal.enable_frame_sync()

    def try_wait_for_frames(self, timeout_ms: int = 5000) -> Optional[pyorbbecsdk.FrameSet]:
        frameset = self._internal.wait_for_frames(timeout_ms)
        if frameset is None: # type: ignore
            return None

        # hack to get around receiving frames when not all streams are ready
        if frameset.get_color_frame() is None or frameset.get_frame_by_type(pyorbbecsdk.OBFrameType.LEFT_IR_FRAME) is None: # type: ignore
            return None

        return frameset

    def start(self) -> None:
        if not self._running:
            self._internal.start(self._config) # type: ignore
            self._running = True

    def stop(self) -> None:
        if self._running:
            self._internal.stop()
            self._running = False
        return

    def enable_stream(self, stream: depth_sensor.interface.pipeline.Stream, format: depth_sensor.interface.pipeline.frame.StreamFormat, framerate: int) -> None:
        profile_list = self._internal.get_stream_profile_list(ob_stream_to_ob_sensor_type(stream_to_ob_stream(stream)))
        profile = profile_list.get_stream_profile_by_format(format_to_ob_format(format), framerate)
        if profile is None:
            raise ValueError(f"Unsupported stream type: {stream}")
        self._config.enable_stream(profile)
        self.stop()
        self.start()
    
    def filter_supported(self, filter: depth_sensor.interface.pipeline.Filter) -> bool:
        if filter == depth_sensor.interface.pipeline.Filter.HDR_MERGE:
            return self.hdr_supported()
        elif filter == depth_sensor.interface.pipeline.Filter.TEMPORAL:
            return True
        elif filter == depth_sensor.interface.pipeline.Filter.SPATIAL:
            return True
        elif filter == depth_sensor.interface.pipeline.Filter.ALIGN_D2C:
            return True
        else:
            return False

    def hdr_supported(self) -> bool:
        return self._internal.get_device().is_property_supported(pyorbbecsdk.OBPropertyID.OB_STRUCT_DEPTH_HDR_CONFIG, pyorbbecsdk.OBPermissionType.PERMISSION_READ_WRITE)

    def filters_process(self, frameset: pyorbbecsdk.FrameSet, filters: depth_sensor.interface.pipeline.Filter) -> Optional[pyorbbecsdk.FrameSet]:
        try:
            if depth_sensor.interface.pipeline.Filter.HDR_MERGE in filters:
                if self._hdr_merge_filter is None:
                    self._hdr_merge_filter = pyorbbecsdk.HDRMergeFilter()
                # not in the docs nor the examples. and for some reason three pushes need to happen for process to return not None
                self._hdr_merge_filter.push_frame(frameset.get_depth_frame())
                maybe_frameset = self._hdr_merge_filter.process(frameset)
                if maybe_frameset is not None:
                    frameset = maybe_frameset
                else:
                    return None

            if depth_sensor.interface.pipeline.Filter.NOISE_REMOVAL in filters:
                if self._noise_removal_filter is None:
                    self._noise_removal_filter = pyorbbecsdk.NoiseRemovalFilter()
                    params = pyorbbecsdk.OBNoiseRemovalFilterParams()
                    params.disp_diff = 256
                    params.max_size = 80
                    self._noise_removal_filter.set_filter_params(params)

            if depth_sensor.interface.pipeline.Filter.TEMPORAL in filters:
                if self._temporal_filter is None:
                    self._temporal_filter = pyorbbecsdk.TemporalFilter()
                    self._temporal_filter.set_diff_scale(0.1)
                    self._temporal_filter.set_weight(0.1)
                frameset = self._temporal_filter.process(frameset)

            if depth_sensor.interface.pipeline.Filter.SPATIAL in filters:
                if self._spatial_filter is None:
                    self._spatial_filter = pyorbbecsdk.SpatialAdvancedFilter()
                    params = pyorbbecsdk.OBSpatialAdvancedFilterParams()
                    params.alpha = 0.1
                    params.disp_diff = 160
                    params.magnitude = 5
                    params.radius = 0
                    self._spatial_filter.set_filter_params(params)
                frameset = self._spatial_filter.process(frameset)

            if depth_sensor.interface.pipeline.Filter.ALIGN_D2C in filters:
                if self._align_d2c_filter is None:
                    self._align_d2c_filter = pyorbbecsdk.AlignFilter(align_to_stream=pyorbbecsdk.OBStreamType.COLOR_STREAM)
                frameset = self._align_d2c_filter.process(frameset)

            if frameset is None: # type: ignore
                return None
        except pyorbbecsdk.OBError as e:
            print(e)
            return None

        frameset = frameset.as_frame_set()
        if frameset.get_color_frame() is None or frameset.get_frame_by_type(pyorbbecsdk.OBFrameType.LEFT_IR_FRAME) is None: # type: ignore
            return None

        return frameset
    
    def set_hdr_enabled(self, enabled: bool):
        hdr_config = pyorbbecsdk.OBHdrConfig()
        hdr_config.enable = enabled
        if enabled:
            hdr_config.exposure_1 = 1000
            hdr_config.exposure_2 = 10000
            hdr_config.gain_1 = 16
            hdr_config.gain_2 = 16
        self._internal.get_device().set_hdr_config(hdr_config)
        return

    def set_ir_exposure(self, exposure: int):
        self._internal.get_device().set_bool_property(pyorbbecsdk.OBPropertyID.OB_PROP_IR_AUTO_EXPOSURE_BOOL, False)
        self._internal.get_device().set_int_property(pyorbbecsdk.OBPropertyID.OB_PROP_IR_EXPOSURE_INT, exposure)

def format_to_ob_format(format: depth_sensor.interface.pipeline.frame.StreamFormat) -> pyorbbecsdk.OBFormat:
    if format == depth_sensor.interface.pipeline.frame.StreamFormat.RGB:
        return pyorbbecsdk.OBFormat.RGB
    elif format == depth_sensor.interface.pipeline.frame.StreamFormat.BGR:
        return pyorbbecsdk.OBFormat.BGR
    elif format == depth_sensor.interface.pipeline.frame.StreamFormat.Z16:
        return pyorbbecsdk.OBFormat.Z16
    elif format == depth_sensor.interface.pipeline.frame.StreamFormat.Y8:
        return pyorbbecsdk.OBFormat.Y8
    elif format == depth_sensor.interface.pipeline.frame.StreamFormat.MJPG:
        return pyorbbecsdk.OBFormat.MJPG
    else:
        raise ValueError(f"Unsupported stream format: {format}")

def stream_to_ob_stream(stream: depth_sensor.interface.pipeline.Stream) -> pyorbbecsdk.OBStreamType:
    if stream == depth_sensor.interface.pipeline.Stream.COLOR:
        return pyorbbecsdk.OBStreamType.COLOR_STREAM
    elif stream == depth_sensor.interface.pipeline.Stream.DEPTH:
        return pyorbbecsdk.OBStreamType.DEPTH_STREAM
    elif stream == depth_sensor.interface.pipeline.Stream.INFRARED:
        return pyorbbecsdk.OBStreamType.LEFT_IR_STREAM
    else:
        raise ValueError(f"Unsupported stream type: {stream}")

def ob_stream_to_ob_sensor_type(stream: pyorbbecsdk.OBStreamType) -> pyorbbecsdk.OBSensorType:
    if stream == pyorbbecsdk.OBStreamType.COLOR_STREAM:
        return pyorbbecsdk.OBSensorType.COLOR_SENSOR
    elif stream == pyorbbecsdk.OBStreamType.DEPTH_STREAM:
        return pyorbbecsdk.OBSensorType.DEPTH_SENSOR
    elif stream == pyorbbecsdk.OBStreamType.LEFT_IR_STREAM:
        return pyorbbecsdk.OBSensorType.LEFT_IR_SENSOR
    else:
        raise ValueError(f"Unsupported stream type: {stream}")
