import enum
from typing import Optional, Protocol

from . import frame, stream_profile

class Stream(enum.Enum):
    COLOR = 0
    DEPTH = 1
    INFRARED = 2

class Filter(enum.Enum):
    SPATIAL = 0
    TEMPORAL = 1
    HDR_MERGE = 2

class Pipeline(Protocol):
    def try_wait_for_frames(self, timeout_ms: int = 5000) -> Optional[frame.CompositeFrame]:
        ...
    
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...
    
    def enable_stream(self, stream: Stream, format: frame.StreamFormat) -> None:
        ...

    def filter_supported(self, filter: Filter) -> bool:
        ...
    
    def filter_process(self, filter: Filter, frame: frame.CompositeFrame) -> frame.CompositeFrame:
        ...
    
    def get_stream_profile(self, steam: Stream) -> stream_profile.StreamProfile:
        ...

    def get_extrinsic_between(self, stream1: Stream, stream2: Stream) -> stream_profile.Extrinsic:
        ...
