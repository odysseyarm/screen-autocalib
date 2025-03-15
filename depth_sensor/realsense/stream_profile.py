from __future__ import annotations

import numpy as np
import pyrealsense2
from typing import Literal, Self

import depth_sensor.interface.pipeline
import depth_sensor.interface.stream_profile

class CameraIntrinsic(depth_sensor.interface.stream_profile.CameraIntrinsic):
    def __init__(self, rs_intrinsic: pyrealsense2.intrinsics):
        rs_d = rs_intrinsic.coeffs
        d = depth_sensor.interface.stream_profile.CameraDistortion()
        [d.k1, d.k2, d.p1, d.p2, d.k3] = np.array(rs_d)
        self.dist_coeffs = d
        self.cx, self.cy = np.float32(rs_intrinsic.ppx), np.float32(rs_intrinsic.ppy)
        self.fx, self.fy, self.height, self.width = np.float32(rs_intrinsic.fx), np.float32(rs_intrinsic.fy), rs_intrinsic.height, rs_intrinsic.width
        self.model = depth_sensor.interface.stream_profile.DistortionModel.INV_BROWN_CONRADY

class Extrinsic(depth_sensor.interface.stream_profile.Extrinsic):
    rot: np.ndarray[Literal[3,3], np.dtype[np.float32]]
    transform: np.ndarray[Literal[3], np.dtype[np.float32]]

    def __init__(self, rs_extrinsic: pyrealsense2.extrinsics):
        self.rot = np.array(rs_extrinsic.rotation)
        self.transform = np.array(rs_extrinsic.translation)

class StreamProfile:
    _internal_profile: pyrealsense2.stream_profile

    def __init__(self, rs_profile: pyrealsense2.stream_profile):
        self._internal_profile = rs_profile

    def get_extrinsic_to(self, to: Self) -> depth_sensor.interface.stream_profile.Extrinsic:
        return Extrinsic(self._internal_profile.get_extrinsics_to(to._internal_profile))

    def as_video_stream_profile(self) -> depth_sensor.interface.stream_profile.VideoStreamProfile:
        return VideoStreamProfile(self._internal_profile.as_video_stream_profile())

class VideoStreamProfile(StreamProfile):
    _internal_video_profile: pyrealsense2.video_stream_profile

    def __init__(self, rs_profile: pyrealsense2.video_stream_profile):
        self._internal_video_profile = rs_profile
        self._internal_profile = rs_profile

    def get_intrinsic(self) -> CameraIntrinsic:
        return CameraIntrinsic(self._internal_video_profile.get_intrinsics())
