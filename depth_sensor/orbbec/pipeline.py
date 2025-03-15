import pyorbbecsdk

import depth_sensor.interface.frame
import depth_sensor.interface.pipeline
import depth_sensor.interface.stream_profile

from typing import Optional

class Pipeline:
    _internal: pyorbbecsdk.Pipeline

    _noise_removal_filter: Optional[pyorbbecsdk.NoiseRemovalFilter] = None
    _temporal_filter: Optional[pyorbbecsdk.TemporalFilter] = None
    _spatial_filter: Optional[pyorbbecsdk.SpatialAdvancedFilter] = None
    _align_d2c_filter: Optional[pyorbbecsdk.AlignFilter] = None

    _running: bool = False

    _config: pyorbbecsdk.Config

    def __init__(self, ob_pipeline: pyorbbecsdk.Pipeline, config: pyorbbecsdk.Config):
        self._internal = ob_pipeline
        self._config = config

    def try_wait_for_frames(self, timeout_ms: int = 5000) -> Optional[pyorbbecsdk.FrameSet]:
        frameset = self._internal.wait_for_frames(timeout_ms)
        if frameset is None: # type: ignore
            return None

        # hack to get around receiving frames when not all streams are ready
        if frameset.get_color_frame() is None or frameset.get_ir_frame() is None: # type: ignore
            return None

        return frameset

    def start(self) -> None:
        self._internal.enable_frame_sync()
        self._internal.start(self._config) # type: ignore
        self._running = True

    def stop(self) -> None:
        # if self._running:
        self._internal.stop()
        self._running = False

    def enable_stream(self, stream: depth_sensor.interface.pipeline.Stream, format: depth_sensor.interface.pipeline.frame.StreamFormat) -> None:
        pass

    def hdr_supported(self) -> bool:
        return self._internal.get_device().is_property_supported(pyorbbecsdk.OBPropertyID.OB_STRUCT_DEPTH_HDR_CONFIG, pyorbbecsdk.OBPermissionType.PERMISSION_READ_WRITE)

    def filters_process(self, frameset: pyorbbecsdk.FrameSet, filters: depth_sensor.interface.pipeline.Filter) -> Optional[pyorbbecsdk.FrameSet]:
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
                self._temporal_filter.set_weight(0.4)
            frameset = self._temporal_filter.process(frameset)

        if depth_sensor.interface.pipeline.Filter.SPATIAL in filters:
            if self._spatial_filter is None:
                self._spatial_filter = pyorbbecsdk.SpatialAdvancedFilter()
                params = pyorbbecsdk.OBSpatialAdvancedFilterParams()
                params.alpha = 0.5
                params.disp_diff = 160
                params.magnitude = 1
                params.radius = 1
                self._spatial_filter.set_filter_params(params)
            frameset = self._spatial_filter.process(frameset)

        if depth_sensor.interface.pipeline.Filter.ALIGN_D2C in filters:
            if self._align_d2c_filter is None:
                self._align_d2c_filter = pyorbbecsdk.AlignFilter(align_to_stream=pyorbbecsdk.OBStreamType.COLOR_STREAM)
            frameset = self._align_d2c_filter.process(frameset)

        if frameset is None: # type: ignore
            return None
        
        frameset = frameset.as_frame_set()
        if frameset.get_color_frame() is None or frameset.get_ir_frame() is None: # type: ignore
            return None

        return frameset
