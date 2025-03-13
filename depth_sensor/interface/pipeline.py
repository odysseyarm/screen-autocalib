import enum
from typing import Optional, Protocol, TypeVar

from . import frame, stream_profile

import pyorbbecsdk
import pyrealsense2

class Stream(enum.Enum):
    COLOR = 0
    DEPTH = 1
    INFRARED = 2

class Filter(enum.Flag):
    NOISE_REMOVAL = enum.auto()
    TEMPORAL = enum.auto()
    SPATIAL = enum.auto()
    ALIGN_D2C = enum.auto()

class Pipeline(Protocol):
    def try_wait_for_frames(self, timeout_ms: int = 5000) -> Optional[pyorbbecsdk.FrameSet|pyrealsense2.composite_frame]:
        ...
    
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...
    
    def enable_stream(self, stream: Stream, format: frame.StreamFormat) -> None:
        ...

    def filter_supported(self, filter: Filter) -> bool:
        ...

    def hdr_supported(self) -> bool:
        ...

    T=TypeVar("T", covariant=True, bound=(pyorbbecsdk.FrameSet|pyrealsense2.composite_frame))
    def filters_process(self, frameset: T, filters: Filter) -> Optional[T]:
        ...
