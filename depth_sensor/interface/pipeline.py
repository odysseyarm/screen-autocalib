import enum
from typing import Generic, Optional, Protocol, TypeVar

from . import frame

import pyorbbecsdk
import pyrealsense2

class Stream(enum.Enum):
    COLOR = 0
    DEPTH = 1
    INFRARED = 2

class Filter(enum.Flag):
    HDR_MERGE = enum.auto()
    NOISE_REMOVAL = enum.auto()
    TEMPORAL = enum.auto()
    SPATIAL = enum.auto()
    ALIGN_D2C = enum.auto()

T=TypeVar("T", pyorbbecsdk.FrameSet, pyrealsense2.composite_frame)
class Pipeline(Protocol, Generic[T]):
    def try_wait_for_frames(self, timeout_ms: int = 5000) -> Optional[pyorbbecsdk.FrameSet|pyrealsense2.composite_frame]:
        ...
    
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...
    
    def enable_stream(self, stream: Stream, format: frame.StreamFormat, framerate: int) -> None:
        ...

    def filter_supported(self, filter: Filter) -> bool:
        ...

    def hdr_supported(self) -> bool:
        ...

    def filters_process(self, frameset: T, filters: Filter) -> Optional[T]:
        ...

    def set_hdr_enabled(self, enabled: bool):
        ...

    def set_ir_exposure(self, exposure: int):
        ...
