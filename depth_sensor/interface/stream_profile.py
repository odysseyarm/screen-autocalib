from __future__ import annotations
import enum

import numpy
from typing import Literal, Protocol, Self

class DistortionModel(enum.Enum):
    BROWN_CONRADY = 0,
    INV_BROWN_CONRADY = 1,

class CameraDistortion:
    k1: float
    k2: float
    k3: float
    k4: float
    k5: float
    k6: float
    p1: float
    p2: float

    def __str__(self):
        return (f"CameraDistortion(k1={self.k1}, k2={self.k2}, k3={self.k3}, "
                f"k4={self.k4}, k5={self.k5}, k6={self.k6}, p1={self.p1}, p2={self.p2})")

class CameraIntrinsic:
    dist_coeffs: CameraDistortion
    cx: float
    cy: float
    fx: float
    fy: float
    height: int
    width: int
    model: DistortionModel

    def __str__(self):
        return (f"CameraIntrinsic(cx={self.cx}, cy={self.cy}, fx={self.fx}, fy={self.fy}, "
                f"height={self.height}, width={self.width}, model={self.model}, "
                f"dist_coeffs={self.dist_coeffs})")

class Extrinsic:
    rot: numpy.ndarray[Literal[3,3], numpy.dtype[numpy.float32]]
    transform: numpy.ndarray[Literal[3], numpy.dtype[numpy.float32]]

class StreamProfile(Protocol):
    def get_extrinsic_to(self, to: Self) -> Extrinsic:
        ...
    def as_video_stream_profile(self) -> VideoStreamProfile:
        ...

class VideoStreamProfile(StreamProfile, Protocol):
    def get_intrinsic(self) -> CameraIntrinsic:
        ...

# Foo.foo_method.__annotations__ == {'to': 'StreamProfile'}
# typing.get_type_hints(Foo.foo_method) == {'to': StreamProfile}
