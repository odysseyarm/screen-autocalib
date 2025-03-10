import enum
from typing import Protocol

import numpy.typing as npt

class StreamFormat(enum.Enum):
    RGB = 0
    BGR = 1
    Z16 = 2
    Y8 = 3
    UNKNOWN = 4

class Frame(Protocol):
    def get_data(self) -> npt.ArrayLike:
        ...

class ColorFrame(Frame, Protocol):
    def get_data(self) -> npt.ArrayLike:
        ...
    def get_format(self) -> StreamFormat:
        ...
    def set_format(self, format: StreamFormat) -> None:
        ...

class DepthFrame(Frame, Protocol):
    def get_data(self) -> npt.ArrayLike:
        ...

class InfraredFrame(Frame, Protocol):
    def get_data(self) -> npt.ArrayLike:
        ...

class CompositeFrame(Protocol):
    def get_color_frame(self) -> ColorFrame:
        ...
    def get_depth_frame(self) -> DepthFrame:
        ...
    def get_infrared_frame(self, index: int = 0) -> InfraredFrame:
        ...
