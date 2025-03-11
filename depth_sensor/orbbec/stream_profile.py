from __future__ import annotations

import numpy
import pyorbbecsdk
from typing import Literal, Self

import depth_sensor.interface.pipeline
import depth_sensor.interface.stream_profile

class CameraDistortion(depth_sensor.interface.stream_profile.CameraDistortion):
    def __init__(self, ob_d: pyorbbecsdk.OBCameraDistortion):
        self.k1, self.k2, self.k3, self.k4, self.k5, self.k6 = ob_d.k1, ob_d.k2, ob_d.k3, ob_d.k4, ob_d.k5, ob_d.k6
        self.p1, self.p2 = ob_d.p1, ob_d.p2

class CameraIntrinsic(depth_sensor.interface.stream_profile.CameraIntrinsic):
    def __init__(self, ob_intrinsic: pyorbbecsdk.OBCameraIntrinsic, distortion: pyorbbecsdk.OBCameraDistortion):
        self.dist_coeffs = CameraDistortion(distortion)
        self.cx, self.cy = ob_intrinsic.cx, ob_intrinsic.cy
        self.fx, self.fy = ob_intrinsic.fx, ob_intrinsic.fy
        self.height, self.width = ob_intrinsic.height, ob_intrinsic.width
        self.model = depth_sensor.interface.stream_profile.DistortionModel.BROWN_CONRADY

class Extrinsic(depth_sensor.interface.stream_profile.Extrinsic):
    rot: numpy.ndarray[Literal[3,3], numpy.dtype[numpy.float32]]
    transform: numpy.ndarray[Literal[3], numpy.dtype[numpy.float32]]

    def __init__(self, ob_extrinsic: pyorbbecsdk.OBExtrinsic):
        self.rot = ob_extrinsic.rot # type: ignore
        self.transform = ob_extrinsic.transform # type: ignore

class StreamProfile:
    _internal_profile: pyorbbecsdk.StreamProfile

    def __init__(self, orbbec_profile: pyorbbecsdk.StreamProfile):
        self._internal_profile = orbbec_profile

    def get_extrinsic_to(self, to: Self) -> depth_sensor.interface.stream_profile.Extrinsic:
        return Extrinsic(self._internal_profile.get_extrinsic_to(to._internal_profile))
    
    def as_video_stream_profile(self) -> depth_sensor.interface.stream_profile.VideoStreamProfile:
        return VideoStreamProfile(self._internal_profile.as_video_stream_profile())

class VideoStreamProfile(StreamProfile):
    _internal_video_profile: pyorbbecsdk.VideoStreamProfile

    def __init__(self, orbbec_profile: pyorbbecsdk.VideoStreamProfile):
        self._internal_video_profile = orbbec_profile
        self._internal_profile = orbbec_profile

    def get_intrinsic(self) -> CameraIntrinsic:
        return CameraIntrinsic(self._internal_video_profile.get_intrinsic(), self._internal_video_profile.get_distortion())
