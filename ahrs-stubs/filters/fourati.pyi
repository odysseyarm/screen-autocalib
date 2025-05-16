"""

Fourati's nonlinear attitude estimation
=======================================

Attitude estimation algorithm as proposed by [Fourati]_, whose approach
combines a quaternion-based nonlinear filter with the Levenberg Marquardt
Algorithm (LMA.)

The estimation algorithm has a complementary structure that exploits
measurements from an accelerometer, a magnetometer and a gyroscope, combined in
a strap-down system, based on the time integral of the angular velocity, using
the Earth's magnetic field and gravity vector to compensate the attitude
predicted by the gyroscope.

The **rigid body attitude** in space is determined when the body's orientation
frame :math:`(X_B, Y_B, Z_B)` is specified with respect to the navigation frame
:math:`(Y_N, Y_N, Z_N)`, where the navigation frame follows the NED convention
(North-East-Down.)

The unit quaternion, :math:`\\mathbf{q}`, is defined as a scalar-vector pair of
the form:

.. math::
    \\mathbf{q} = \\begin{pmatrix}s & \\mathbf{v}\\end{pmatrix}^T

where :math:`s` is the scalar part and :math:`\\mathbf{v}=\\begin{pmatrix}v_x & v_y & v_z\\end{pmatrix}^T`
is the vector part of the quaternion.

.. note::
    Most literature, and this package's documentation, use the notation
    :math:`\\mathbf{q}=\\begin{pmatrix}q_w & q_x & q_y & q_z\\end{pmatrix}` to
    define a quaternion, but this algorithm uses a different one, and it will
    preserved to keep the coherence with the original document.

The sensor configuration consists of a three-axis gyroscope, a three-axis
accelerometer, a three-axis magnetometer. Their outputs can be modelled,
respectively, as:

.. math::
    \\begin{array}{rcl}
    \\omega_G =& \\begin{bmatrix}\\omega_{GX} & \\omega_{GY} & \\omega_{GZ}\\end{bmatrix}^T &= \\omega + b + \\delta_G \\\\&&\\\\
    \\mathbf{f} =& \\begin{bmatrix}f_x & f_y & f_z\\end{bmatrix}^T &= M_N^B(\\mathbf{q}) (g+a) + \\delta_f \\\\&&\\\\
    \\mathbf{h} =& \\begin{bmatrix}h_x & h_y & h_z\\end{bmatrix}^T &= M_N^B(\\mathbf{q}) m + \\delta_h
    \\end{array}

where :math:`b\\in\\mathbb{R}^3` is the unknown gyro-bias vector and :math:`\\delta_G`,
:math:`\\delta_f` and :math:`\\delta_h\\in\\mathbb{R}^3` are assumed `white
Gaussian noises <https://en.wikipedia.org/wiki/Additive_white_Gaussian_noise>`_.
:math:`\\omega` is the *real* angular velocity, :math:`g` is the gravity vector,
:math:`a` denotes the Dynamic Body Acceleration (DBA), :math:`m` describes the
direction of the Earth's magnetic field on the local position, and
:math:`M_N^B(\\mathbf{q})` is the orthogonal matrix describing the attitude of
the body frame.

.. math::
    \\mathbf{M}_N^B(\\mathbf{q}) =
    \\begin{bmatrix}
    2(s^2 + v_x^2) - 1 & 2(v_xv_y + sv_z) & 2(v_xv_z - sv_y) \\\\
    2(v_xv_y - sv_z) & 2(s^2 + v_y^2) - 1 & 2(sv_x + v_yv_z) \\\\
    2(sv_y + v_xv_z) & 2(v_yv_z - sv_x) & 2(s^2 + v_z^2) - 1
    \\end{bmatrix}

The kinematic differential equation, in terms of the unit quaternion, that
describes the relationship between the rigid body attitude variation and the
angular velocity in the body frame is represented by:

.. math::
    \\begin{array}{rcl}
    \\dot{\\mathbf{q}} &=& \\frac{1}{2}\\mathbf{q}\\omega_\\mathbf{q} \\\\
    \\begin{bmatrix}\\dot{s}\\\\ \\dot{v}_x \\\\ \\dot{v}_y \\\\ \\dot{v}_z\\end{bmatrix}
    &=& \\frac{1}{2}\\begin{bmatrix}-\\mathbf{v}^T \\\\ \\mathbf{I}_3s+\\lfloor\\mathbf{v}\\rfloor_\\times\\end{bmatrix}
    \\begin{bmatrix}\\omega_x \\\\ \\omega_y \\\\ \\omega_z\\end{bmatrix}
    \\end{array}

where :math:`\\omega_\\mathbf{q}=\\begin{bmatrix}0 & \\omega^T\\end{bmatrix}^T`
is the equivalent to the angular velocity :math:`\\omega\\in\\mathbb{R}^3` of
the rigid body measured in :math:`B` and relative to :math:`N`, :math:`\\mathbf{I}_3`
is the :math:`3\\times 3` identity matrix, and :math:`\\lfloor\\mathbf{v}\\rfloor_\\times`
is the `Skew symmetric matrix <https://en.wikipedia.org/wiki/Skew-symmetric_matrix>`_
of the vector :math:`\\mathbf{v}`.

.. math::
    \\lfloor\\mathbf{v}\\rfloor_\\times =
    \\begin{bmatrix}0 & -v_z & v_y \\\\ v_z & 0 & -v_x \\\\ -v_y & v_x & 0\\end{bmatrix}

.. note::
    Any vector :math:`\\mathbf{x}=\\begin{bmatrix}x_1 & x_2 & x_3\\end{bmatrix}^T\\in\\mathbb{R}^3`
    that multiplies with a quaternion must be considered a `pure quaternion
    <https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Using_quaternion_as_rotations>`_,
    :math:`\\mathbf{x_q}=\\begin{bmatrix}0 & x_1 & x_2 & x_3\\end{bmatrix}^T\\in\\mathbb{R}^4`,
    so that they operate with a *Hamilton product*.

To achieve an optimal attitude estimation, a nonlinear system is developed,
whose **output** is the stack of the accelerometer and magnetometer measurements:

.. math::
    \\mathbf{y} = \\begin{bmatrix}f_x & f_y & f_z & h_x & h_y & h_z\\end{bmatrix}^T

The `World Magnetic Model <../wmm.html>`_ considers a magnetic vector
:math:`\\mathbf{m}=\\begin{bmatrix}m_x & m_y & m_z \\end{bmatrix}\\in\\mathbb{R}^3`
at any location on Earth to describe the geomagnetic field. For practical
purposes, the vector is simplified to
:math:`\\mathbf{m}=\\begin{bmatrix}m\\cos\\theta & 0 & m\\sin\\theta\\end{bmatrix}`,
with a dip angle :math:`\\theta` and a magnetic intensity :math:`m`, which
varies between 23000 and 67000 nT, depending on the region on Earth. This
simplified vector discards the Easterly magnetic field (:math:`m_y`), although
for an accurate reference, it is preferred to use it.

Similar to :math:`\\mathbf{y}`, the estimated values :math:`\\hat{\\mathbf{y}}`
are given by:

.. math::
    \\hat{\\mathbf{y}} = \\begin{bmatrix}\\hat{f}_x & \\hat{f}_y & \\hat{f}_z & \\hat{h}_x & \\hat{h}_y & \\hat{h}_z\\end{bmatrix}^T

whose components are calculated as:

.. math::
    \\begin{array}{rcl}
    \\hat{\\mathbf{f}} &=& \\begin{bmatrix}\\hat{f}_x & \\hat{f}_y & \\hat{f}_z \\end{bmatrix}^T = \\hat{\\mathbf{q}}^{-1}\\mathbf{g_q}\\hat{\\mathbf{q}} \\\\ && \\\\
    \\hat{\\mathbf{h}} &=& \\begin{bmatrix}\\hat{h}_x & \\hat{h}_y & \\hat{h}_z \\end{bmatrix}^T = \\hat{\\mathbf{q}}^{-1}\\mathbf{m_q}\\hat{\\mathbf{q}}
    \\end{array}

where :math:`\\mathbf{g_q}=\\begin{bmatrix}0 & 0 & 0 & 9.8\\end{bmatrix}^T` is
the **reference gravity vector** as a pure quaternion, and
:math:`\\mathbf{m_q}=\\begin{bmatrix}0 & m\\cos\\theta & 0 & m\\sin\\theta\\end{bmatrix}^T`
is the local **reference geomagnetic field** also represented as a pure
quaternion.

The modeling error, :math:`\\delta(\\hat{\\mathbf{q}})=\\mathbf{y}-\\hat{\\mathbf{y}}`,
represents the difference between the real measurements :math:`\\mathbf{y}` and
the estimated values :math:`\\hat{\\mathbf{y}}`.

The nonlinear filter of this model takes the form:

.. math::
    \\dot{\\mathbf{q}} =
    \\begin{bmatrix}\\dot{s}\\\\ \\dot{v}_x \\\\ \\dot{v}_y \\\\ \\dot{v}_z\\end{bmatrix} =
    \\frac{1}{2}\\hat{\\mathbf{q}}\\omega_\\mathbf{q}
    \\begin{bmatrix}1 \\\\ \\mathbf{K}\\end{bmatrix}

where :math:`\\hat{\\mathbf{q}}=\\begin{bmatrix}\\hat{s}& \\hat{v}_x & \\hat{v}_y & \\hat{v}_z\\end{bmatrix}^T\\in\\mathbb{R}^4`
is the **estimated state**, and :math:`\\mathbf{K}\\in\\mathbb{R}^{3\\times 6}`
is the **observer gain**.

This gain :math:`\\mathbf{K}` is used to correct the modeling error
:math:`\\delta(\\hat{\\mathbf{q}})`, which can be done if we locate the minimum
of the squared error function :math:`\\xi(\\hat{\\mathbf{q}})=\\delta(\\hat{\\mathbf{q}})^T\\delta(\\hat{\\mathbf{q}})`.

For this attitude estimator the `Levenberg-Marquardt Algorithm
<https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm>`_ (LMA)
is used to minimize the nonlinear function :math:`\\xi(\\hat{\\mathbf{q}})`.
So, the unique minimum can be computed with:

.. math::
    \\begin{array}{rcl}
    \\eta(\\hat{\\mathbf{q}}) &=& \\mathbf{K}\\delta(\\hat{\\mathbf{q}}) \\\\
    &=& k[\\mathbf{X}^T\\mathbf{X} + \\lambda\\mathbf{I}_3]^{-1}\\mathbf{X}^T\\delta(\\hat{\\mathbf{q}})
    \\end{array}

where the tiny value :math:`\\lambda` guarantees the inversion of the matrix,
the gain factor :math:`k` tunes the balance between measurement noise
supression and the filter's response time, and
:math:`\\mathbf{X}\\in\\mathbb{R}^{6\\times 3}` is the Jacobian matrix:

.. math::
    \\begin{array}{rcl}
    \\mathbf{X} &=& -2\\begin{bmatrix}\\lfloor\\hat{\\mathbf{f}}\\rfloor_\\times & \\lfloor\\hat{\\mathbf{h}}\\rfloor_\\times\\end{bmatrix} \\\\
    &=& -2\\begin{bmatrix}
    0 & -\\hat{f}_z & \\hat{f}_y & 0 & -\\hat{h}_z & \\hat{h}_y \\\\
    \\hat{f}_z & 0 & -\\hat{f}_x & \\hat{h}_z & 0 & -\\hat{h}_x \\\\
    -\\hat{f}_y & \\hat{f}_x & 0 & -\\hat{h}_y & \\hat{h}_x & 0
    \\end{bmatrix}
    \\end{array}

The resulting structure of the nonlinear filter is complementary: it blends the
low-frequency region (low bandwidth) of the accelerometer and magnetometer data,
where the attitude is typically more accurate, with the high-frequency region
(high bandwidth) of the gyroscope data, where the integration of the angular
velocity yields better attitude estimates.

By filtering the high-frequency components of the signals from the
accelerometer (DBA) and the low-frequency components of the gyroscope signal
(slow-moving drift), the nonlinear filter produces an accurate estimate of the
attitude.

The correction term, :math:`\\Delta\\in\\mathbb{R}^{4\\times 7}`, is computed
using the gain :math:`K` such as:

.. math::
    \\Delta =
    \\begin{bmatrix}1 & \\mathbf{0} \\\\ \\mathbf{0} & \\mathbf{K}\\end{bmatrix}
    \\begin{bmatrix}1 \\\\ \\delta(\\hat{\\mathbf{q}})\\end{bmatrix}

It is used to correct the estimated angular velocity, :math:`\\dot{\\hat{\\mathbf{q}}}`,
as:

.. math::
    \\dot{\\hat{\\mathbf{q}}} = \\big(\\frac{1}{2}\\hat{\\mathbf{q}}\\omega_\\mathbf{q}\\big)\\Delta

With the corrected angular velocity, we integrate it using the sampling step
:math:`\\Delta_t` and add it to the previous quaternion :math:`\\mathbf{q}_{t-1}`
to obtain the new attitude :math:`\\mathbf{q}_t`:

.. math::
    \\mathbf{q}_t = \\mathbf{q}_{t-1} + \\dot{\\hat{\\mathbf{q}}}\\Delta_t

.. warning::
    Do not confuse the correction term :math:`\\Delta` with the sampling step
    :math:`\\Delta_t`, which is actually the inverse of the sampling frequency
    :math:`f=\\frac{1}{\\Delta_t}`.

References
----------
.. [Fourati] Hassen Fourati, Noureddine Manamanni, Lissan Afilal, Yves
    Handrich. A Nonlinear Filtering Approach for the Attitude and Dynamic Body
    Acceleration Estimation Based on Inertial and Magnetic Sensors: Bio-Logging
    Application. IEEE Sensors Journal, Institute of Electrical and Electronics
    Engineers, 2011, 11 (1), pp. 233-244. 10.1109/JSEN.2010.2053353.
    (https://hal.archives-ouvertes.fr/hal-00624142/file/Papier_IEEE_Sensors_Journal.pdf)

"""
from __future__ import annotations
from ahrs.common.mathfuncs import cosd
from ahrs.common.mathfuncs import sind
from ahrs.common.mathfuncs import skew
from ahrs.common.orientation import am2q
from ahrs.common.orientation import q_conj
from ahrs.common.orientation import q_prod
from ahrs.utils.wgs84 import WGS
from ahrs.utils.wmm import WMM
import cmath as cmath
import numpy
import numpy as np
__all__ = ['DEG2RAD', 'DYNAMIC_ELLIPTICITY', 'EARTH_ATMOSPHERE_MASS', 'EARTH_AUTHALIC_RADIUS', 'EARTH_AXIS_RATIO', 'EARTH_C20_DYN', 'EARTH_C20_GEO', 'EARTH_C22_DYN', 'EARTH_EQUATOR_RADIUS', 'EARTH_EQUIVOLUMETRIC_RADIUS', 'EARTH_FIRST_ECCENTRICITY', 'EARTH_FIRST_ECCENTRICITY_2', 'EARTH_FLATTENING', 'EARTH_FLATTENING_INV', 'EARTH_GM', 'EARTH_GM_1', 'EARTH_GM_2', 'EARTH_GM_GPSNAV', 'EARTH_J2', 'EARTH_LINEAR_ECCENTRICITY', 'EARTH_MASS', 'EARTH_MEAN_AXIAL_RADIUS', 'EARTH_MEAN_RADIUS', 'EARTH_POLAR_CURVATURE_RADIUS', 'EARTH_POLAR_RADIUS', 'EARTH_ROTATION', 'EARTH_SECOND_ECCENTRICITY', 'EARTH_SECOND_ECCENTRICITY_2', 'EARTH_SIDEREAL_DAY', 'EQUATORIAL_NORMAL_GRAVITY', 'Fourati', 'GRAVITY', 'JUPITER_EQUATOR_RADIUS', 'JUPITER_GM', 'JUPITER_J2', 'JUPITER_MASS', 'JUPITER_POLAR_RADIUS', 'JUPITER_ROTATION', 'LIGHT_SPEED', 'MAG', 'MARS_EQUATOR_RADIUS', 'MARS_GM', 'MARS_J2', 'MARS_MASS', 'MARS_POLAR_RADIUS', 'MARS_ROTATION', 'MEAN_NORMAL_GRAVITY', 'MERCURY_EQUATOR_RADIUS', 'MERCURY_GM', 'MERCURY_J2', 'MERCURY_MASS', 'MERCURY_POLAR_RADIUS', 'MERCURY_ROTATION', 'MOON_EQUATOR_RADIUS', 'MOON_GM', 'MOON_J2', 'MOON_MASS', 'MOON_POLAR_RADIUS', 'MOON_ROTATION', 'MUNICH_HEIGHT', 'MUNICH_LATITUDE', 'MUNICH_LONGITUDE', 'M_PI', 'NEPTUNE_EQUATOR_RADIUS', 'NEPTUNE_GM', 'NEPTUNE_J2', 'NEPTUNE_MASS', 'NEPTUNE_POLAR_RADIUS', 'NEPTUNE_ROTATION', 'NORMAL_GRAVITY_FORMULA', 'NORMAL_GRAVITY_POTENTIAL', 'PLUTO_EQUATOR_RADIUS', 'PLUTO_GM', 'PLUTO_MASS', 'PLUTO_POLAR_RADIUS', 'PLUTO_ROTATION', 'POLAR_NORMAL_GRAVITY', 'RAD2DEG', 'SATURN_EQUATOR_RADIUS', 'SATURN_GM', 'SATURN_J2', 'SATURN_MASS', 'SATURN_POLAR_RADIUS', 'SATURN_ROTATION', 'SOMIGLIANA_GRAVITY', 'UNIVERSAL_GRAVITATION_CODATA2014', 'UNIVERSAL_GRAVITATION_CODATA2018', 'UNIVERSAL_GRAVITATION_WGS84', 'URANUS_EQUATOR_RADIUS', 'URANUS_GM', 'URANUS_J2', 'URANUS_MASS', 'URANUS_POLAR_RADIUS', 'URANUS_ROTATION', 'VENUS_EQUATOR_RADIUS', 'VENUS_GM', 'VENUS_J2', 'VENUS_MASS', 'VENUS_POLAR_RADIUS', 'VENUS_ROTATION', 'WGS', 'WMM', 'am2q', 'cmath', 'cosd', 'np', 'q_conj', 'q_prod', 'sind', 'skew']
class Fourati:
    """
    
        Fourati's attitude estimation
    
        Parameters
        ----------
        acc : numpy.ndarray, default: None
            N-by-3 array with measurements of acceleration in in m/s^2
        gyr : numpy.ndarray, default: None
            N-by-3 array with measurements of angular velocity in rad/s
        mag : numpy.ndarray, default: None
            N-by-3 array with measurements of magnetic field in mT
        frequency : float, default: 100.0
            Sampling frequency in Herz.
        Dt : float, default: 0.01
            Sampling step in seconds. Inverse of sampling frequency. Not required
            if `frequency` value is given.
        gain : float, default: 0.1
            Filter gain factor.
        q0 : numpy.ndarray, default: None
            Initial orientation, as a versor (normalized quaternion).
        magnetic_dip : float
            Magnetic Inclination angle, in degrees.
        gravity : float
            Normal gravity, in m/s^2.
    
        Attributes
        ----------
        gyr : numpy.ndarray
            N-by-3 array with N gyroscope samples.
        acc : numpy.ndarray
            N-by-3 array with N accelerometer samples.
        mag : numpy.ndarray
            N-by-3 array with N magnetometer samples.
        frequency : float
            Sampling frequency in Herz
        Dt : float
            Sampling step in seconds. Inverse of sampling frequency.
        gain : float
            Filter gain factor.
        q0 : numpy.ndarray
            Initial orientation, as a versor (normalized quaternion).
    
        Raises
        ------
        ValueError
            When dimension of input array(s) ``acc``, ``gyr``, or ``mag`` are not equal.
    
        
    """
    def __init__(self, gyr: numpy.ndarray = None, acc: numpy.ndarray = None, mag: numpy.ndarray = None, **kwargs):
        ...
    def _compute_all(self):
        """
        
                Estimate the quaternions given all data
        
                Attributes ``gyr``, ``acc`` and ``mag`` must contain data.
        
                Returns
                -------
                Q : array
                    M-by-4 Array with all estimated quaternions, where M is the number
                    of samples.
        
                
        """
    def update(self, q: numpy.ndarray, gyr: numpy.ndarray, acc: numpy.ndarray, mag: numpy.ndarray) -> numpy.ndarray:
        """
        
                Quaternion Estimation with a MARG architecture.
        
                Parameters
                ----------
                q : numpy.ndarray
                    A-priori quaternion.
                gyr : numpy.ndarray
                    Sample of tri-axial Gyroscope in rad/s
                acc : numpy.ndarray
                    Sample of tri-axial Accelerometer in m/s^2
                mag : numpy.ndarray
                    Sample of tri-axial Magnetometer in mT
        
                Returns
                -------
                q : numpy.ndarray
                    Estimated quaternion.
        
                
        """
DEG2RAD: float = 0.017453292519943295
DYNAMIC_ELLIPTICITY: float = 0.0032737949
EARTH_ATMOSPHERE_MASS: float = 5.148e+18
EARTH_AUTHALIC_RADIUS: float = 6371007.181
EARTH_AXIS_RATIO: float = 0.996647189335
EARTH_C20_DYN: float = -0.000484165143790815
EARTH_C20_GEO: float = -0.000484166774985
EARTH_C22_DYN: float = 2.43938357328313e-06
EARTH_EQUATOR_RADIUS: float = 6378137.0
EARTH_EQUIVOLUMETRIC_RADIUS: float = 6371000.79
EARTH_FIRST_ECCENTRICITY: float = 0.081819190842622
EARTH_FIRST_ECCENTRICITY_2: float = 0.0066943799901414
EARTH_FLATTENING: float = 0.0033528106647474805
EARTH_FLATTENING_INV: float = 298.257223563
EARTH_GM: float = 398600441800000.0
EARTH_GM_1: float = 398600098200000.0
EARTH_GM_2: float = 343590000.0
EARTH_GM_GPSNAV: float = 398600500000000.0
EARTH_J2: float = 0.00108263
EARTH_LINEAR_ECCENTRICITY: float = 521854.00842339
EARTH_MASS: float = 5.9721864e+24
EARTH_MEAN_AXIAL_RADIUS: float = 6371008.7714
EARTH_MEAN_RADIUS: float = 6371200.0
EARTH_POLAR_CURVATURE_RADIUS: float = 6399593.6258
EARTH_POLAR_RADIUS: float = 6356752.3142
EARTH_ROTATION: float = 7.292115e-05
EARTH_SECOND_ECCENTRICITY: float = 0.082094437949696
EARTH_SECOND_ECCENTRICITY_2: float = 0.006739496742276486
EARTH_SIDEREAL_DAY: float = 86164.09053083288
EQUATORIAL_NORMAL_GRAVITY: float = 9.7803253359
GRAVITY: numpy.float64  # value = 9.809030668031474
JUPITER_EQUATOR_RADIUS: float = 71492000.0
JUPITER_GM: float = 1.2669069494099998e+17
JUPITER_J2: float = 0.014736
JUPITER_MASS: float = 1.898187e+27
JUPITER_POLAR_RADIUS: float = 66854000.0
JUPITER_ROTATION: float = 0.0001758518138029551
LIGHT_SPEED: float = 299792458.0
MAG: dict  # value = {'X': 21018.348711415198, 'Y': 1591.7152699785447, 'Z': 43985.53321525235, 'H': 21078.532682692403, 'F': 48775.31826944238, 'I': 64.39555459733126, 'D': 4.3307314263515435, 'GV': 4.3307314263515435}
MARS_EQUATOR_RADIUS: float = 3396190.0
MARS_GM: float = 42829784016000.0
MARS_J2: float = 0.00196045
MARS_MASS: float = 6.41712e+23
MARS_POLAR_RADIUS: float = 3376200.0
MARS_ROTATION: float = 7.088235959185674e-05
MEAN_NORMAL_GRAVITY: float = 9.7976432223
MERCURY_EQUATOR_RADIUS: float = 2440530.0
MERCURY_GM: float = 22032798701999.996
MERCURY_J2: float = 5.03e-05
MERCURY_MASS: float = 3.30114e+23
MERCURY_POLAR_RADIUS: float = 2438260.0
MERCURY_ROTATION: float = 1.2399326882596827e-06
MOON_EQUATOR_RADIUS: float = 1738100.0
MOON_GM: float = 4902940780000.0
MOON_J2: float = 0.0002027
MOON_MASS: float = 7.346e+22
MOON_POLAR_RADIUS: float = 1736000.0
MOON_ROTATION: float = 1.109027709148159e-07
MUNICH_HEIGHT: float = 0.519
MUNICH_LATITUDE: float = 48.137154
MUNICH_LONGITUDE: float = 11.576124
M_PI: float = 3.141592653589793
NEPTUNE_EQUATOR_RADIUS: float = 24764000.0
NEPTUNE_GM: float = 6835324161799999.0
NEPTUNE_J2: float = 0.003411
NEPTUNE_MASS: float = 1.024126e+26
NEPTUNE_POLAR_RADIUS: float = 24341000.0
NEPTUNE_ROTATION: float = 0.0001083382527619075
NORMAL_GRAVITY_FORMULA: float = 0.003449786506841
NORMAL_GRAVITY_POTENTIAL: float = 6.26368517146
PLUTO_EQUATOR_RADIUS: float = 1188300.0
PLUTO_GM: float = 869661290000.0
PLUTO_MASS: float = 1.303e+22
PLUTO_POLAR_RADIUS: float = 1188300.0
PLUTO_ROTATION: float = -1.13855918346741e-05
POLAR_NORMAL_GRAVITY: float = 9.8321849379
RAD2DEG: float = 57.29577951308232
SATURN_EQUATOR_RADIUS: float = 60268000.0
SATURN_GM: float = 3.793120822819999e+16
SATURN_J2: float = 0.016298
SATURN_MASS: float = 5.683174e+26
SATURN_POLAR_RADIUS: float = 54364000.0
SATURN_ROTATION: float = 0.0001637884057802486
SOMIGLIANA_GRAVITY: float = 0.001931852652458
UNIVERSAL_GRAVITATION_CODATA2014: float = 6.67408e-11
UNIVERSAL_GRAVITATION_CODATA2018: float = 6.6743e-11
UNIVERSAL_GRAVITATION_WGS84: float = 6.67428e-11
URANUS_EQUATOR_RADIUS: float = 25559000.0
URANUS_GM: float = 5794140036099999.0
URANUS_J2: float = 0.00334343
URANUS_MASS: float = 8.68127e+25
URANUS_POLAR_RADIUS: float = 24973000.0
URANUS_ROTATION: float = -0.0001012376653716682
VENUS_EQUATOR_RADIUS: float = 6051800.0
VENUS_GM: float = 324869550209999.94
VENUS_J2: float = 4.458e-06
VENUS_MASS: float = 4.86747e+24
VENUS_POLAR_RADIUS: float = 6051800.0
VENUS_ROTATION: float = -2.9923691869737844e-07
