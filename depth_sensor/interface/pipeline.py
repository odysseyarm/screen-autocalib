import enum
from typing import Optional, Protocol

from . import frames

class Streams(enum.Enum):
    COLOR = 0
    DEPTH = 1
    INFRARED = 2

class Filters(enum.Enum):
    SPATIAL = 0
    TEMPORAL = 1
    HDR_MERGE = 2

class Pipeline(Protocol):
    def try_wait_for_frames(self, timeout_ms: int = 5000) -> Optional[frames.CompositeFrame]:
        ...
    
    def start(self) -> None:
        ...
    
    def enable_stream(self, stream: Streams, format: frames.StreamFormat) -> None:
        ...

    def filter_supported(self, filter: Filters) -> bool:
        ...
    
    def filter_process(self, filter: Filters, frame: frames.CompositeFrame) -> frames.CompositeFrame:
        ...
