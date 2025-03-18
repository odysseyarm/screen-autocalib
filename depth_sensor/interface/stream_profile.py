from __future__ import annotations
import enum

import numpy
from typing import Literal, Protocol, Self

class DistortionModel(enum.Enum):
    BROWN_CONRADY = 0,
    INV_BROWN_CONRADY = 1,

class CameraDistortion:
    k1: numpy.float32
    k2: numpy.float32
    k3: numpy.float32
    k4: numpy.float32
    k5: numpy.float32
    k6: numpy.float32
    p1: numpy.float32
    p2: numpy.float32

    def __str__(self):
        return (f"CameraDistortion(k1={self.k1}, k2={self.k2}, k3={self.k3}, "
                f"k4={self.k4}, k5={self.k5}, k6={self.k6}, p1={self.p1}, p2={self.p2})")

class CameraIntrinsic:
    dist_coeffs: CameraDistortion
    cx: numpy.float32
    cy: numpy.float32
    fx: numpy.float32
    fy: numpy.float32
    height: int
    width: int
    model: DistortionModel

    def __str__(self):
        return (f"CameraIntrinsic(cx={self.cx}, cy={self.cy}, fx={self.fx}, fy={self.fy}, "
                f"height={self.height}, width={self.width}, model={self.model}, "
                f"dist_coeffs={self.dist_coeffs})")

class Extrinsic:
    rot: numpy.ndarray[Literal[3,3], numpy.dtype[numpy.float32]]
    translation: numpy.ndarray[Literal[3], numpy.dtype[numpy.float32]]
    transform: numpy.ndarray[Literal[4,4], numpy.dtype[numpy.float32]]

    def inv(self) -> Extrinsic:
        ...

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
