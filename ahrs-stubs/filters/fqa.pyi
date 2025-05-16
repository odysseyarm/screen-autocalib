"""

Factored Quaternion Algorithm
=============================

The factored quaternion algorithm (FQA) produces a quaternion output to
represent the orientation, restricting the use of magnetic data to the
determination of the rotation about the vertical axis.

The FQA and the `TRIAD <../triad.html>`_ algorithm produce an equivalent
solution to the same problem, with the difference that the former produces a
quaternion, and the latter a rotation matrix.

Magnetic variations cause only azimuth errors in FQA attitude estimation. A
singularity avoidance method is used, which allows the algorithm to track
through all orientations.

.. warning::
    This algorithm is not applicable to situations in which relatively large
    linear accelerations due to dynamic motion are present, unless it is used
    in a complementary or optimal filter together with angular rate information.

The *Earth-fixed coordinate system* :math:`(^ex,\\,^ey,\\,^ez)` is defined following
the `North-East-Down <https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates#Local_north,_east,_down_(NED)_coordinates>`_
(NED) convention.

A *body coordinate system* :math:`(^bx,\\,^by,\\,^bz)` is attached to the rigid
body whose orientation is being measured, and the *sensor frame* :math:`(^sx,\\,^sy,\\,^sz)`
corresponds to the sensor module conformed by the accelerometer/magnetometer
system.

The body coordinate system differs from the sensor frame by a constant offset,
if they are not coinciding. For this method *they are assumed to occupy the same
place*.

From Euler's theorem of rotations, we can use a unit quaternion :math:`\\mathbf{q}`
as an `axis-angle rotation <https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Unit_quaternions>`_:

.. math::
    \\mathbf{q} =
    \\begin{pmatrix}
    \\cos\\frac{\\beta}{2} \\\\
    u_x\\sin\\frac{\\beta}{2} \\\\
    u_y\\sin\\frac{\\beta}{2} \\\\
    u_z\\sin\\frac{\\beta}{2}
    \\end{pmatrix}

where :math:`\\mathbf{u}=\\begin{bmatrix}u_x & u_y & u_z\\end{bmatrix}^T` is
the rotation axis, and :math:`\\beta` is the rotation angle. See the
`Quaternion <../quaternion.html>`_ reference page for further details of it.

A rigid body can be placed in an arbitrary orientation by first rotating it
about its Z-axis by an angle :math:`\\psi` (azimuth or yaw rotation), then
about its Y-axis by angle :math:`\\theta` (elevation or pitch rotation), and
finally about its X-axis by angle :math:`\\phi` (bank or roll rotation).

Elevation Quaternion
--------------------

The X-axis accelerometer senses only the component of gravity along the X-axis,
and this component, in turn, depends only on the elevation angle.

Starting with the rigid body in its reference orientation, the X-axis
accelerometer is perpendicular to gravity and thus registers zero acceleration.
The Y-axis accelerometer also reads zero, while the Z-axis accelerometer reads
:math:`−g`. If the body is then rotated in azimuth about its Z-axis, the X-axis
accelerometer still reads zero, regardless of the azimuth angle.

If the rigid body is next rotated up through an angle :math:`\\theta`, the
X-axis accelerometer will instantaneously read

.. math::
    a_x = g\\sin\\theta

and the Z-axis will read

.. math::
    a_z = -g\\cos\\theta

where :math:`9.81 \\frac{m}{s^2}` is the gravitational acceleration and
:math:`\\mathbf{a}=\\begin{bmatrix}a_x & a_y & a_z\\end{bmatrix}^T` is the
**measured acceleration vector** in the body coordinate system.

For convenience, the accelerometer and magnetometer outputs are normalized to
unit vectors, so that:

.. math::
    \\begin{array}{rcl}
    \\mathbf{a} &=& \\frac{\\mathbf{a}}{\\|\\mathbf{a}\\|} \\\\
    \\mathbf{m} &=& \\frac{\\mathbf{m}}{\\|\\mathbf{m}\\|}
    \\end{array}

From the normalized accelerometer measurements, we can get:

.. math::
    \\sin\\theta = a_x

In order to obtain an elevation quaternion, a value is needed for
:math:`\\sin\\frac{\\theta}{2}` and :math:`\\cos\\frac{\\theta}{2}`. From
trigonometric half-angle formulas, **half-angle values** are given by

.. math::
    \\begin{array}{rcl}
    \\sin\\frac{\\theta}{2} &=& \\mathrm{sgn}(\\sin\\theta) \\sqrt{\\frac{1-\\cos\\theta}{2}} \\\\
    \\cos\\frac{\\theta}{2} &=& \\sqrt{\\frac{1+\\cos\\theta}{2}}
    \\end{array}

where :math:`\\mathrm{sgn}()` is the `sign <https://en.wikipedia.org/wiki/Sign_function>`_
function.

Because elevation is a rotation about the Y-axis, the unit quaternion
representing it will be expressed as:

.. math::
    \\mathbf{q}_e =
    \\begin{pmatrix}\\cos\\frac{\\theta}{2} \\\\ 0 \\\\ \\sin\\frac{\\theta}{2} \\\\ 0\\end{pmatrix}

Roll Quaternion
---------------

Changing the roll angle alters the measurements, so that the accelerometer
readings are:

.. math::
    \\begin{array}{rcl}
    a_y &=& -\\cos\\theta\\sin\\phi \\\\
    a_z &=& -\\cos\\theta\\cos\\phi
    \\end{array}

.. note::
    In reality the measurements are :math:`-g\\cos\\theta\\sin\\phi` and
    :math:`-g\\cos\\theta\\cos\\phi`, with a magnitude equal to :math:`g`, but
    when normalized their magnitude is equal to :math:`1`, and :math:`g` is
    overlooked.

If :math:`\\cos\\theta\\neq 0`, the values of :math:`\\sin\\phi` and :math:`\\cos\\phi`
are determined by:

.. math::
    \\begin{array}{rcl}
    \\sin\\phi &=& -\\frac{a_y}{\\cos\\theta} \\\\
    \\cos\\phi &=& -\\frac{a_z}{\\cos\\theta}
    \\end{array}

But if :math:`\\cos\\theta=0` the roll angle :math:`\\phi` is undefined and can
be assumed to be equal to zero. We obtain the half-angles similar to the
elevation quaternion, and roll quaternion is then computed as:

.. math::
    \\mathbf{q}_r =
    \\begin{pmatrix}\\cos\\frac{\\phi}{2} \\\\ \\sin\\frac{\\phi}{2} \\\\ 0 \\\\ 0\\end{pmatrix}

Azimuth Quaternion
------------------

Since the azimuth rotation has no effect on accelerometer data, the strategy is
to use the readings of the magnetometer, but first we have to rotate the
normalized magnetic readings :math:`^b\\mathbf{m}` into an intermediate
coordinate system through the elevation and roll quaternions:

.. math::
    ^e\\mathbf{m} = \\mathbf{q}_e\\mathbf{q}_r \\,^b\\mathbf{m}\\mathbf{q}_r^{-1}\\mathbf{q}_e^{-1}

where :math:`^b\\mathbf{m}=\\begin{pmatrix}0 & ^bm_x & ^bm_y & ^bm_z\\end{pmatrix}`
is the magnetic field measured in the body frame, and represented as a pure
quaternion.

The rotated magnetic measurements should correspond to the **normalized known
local geomagnetic field** [#]_ vector :math:`\\mathbf{n}=\\begin{bmatrix}n_x & n_y & n_z\\end{bmatrix}`,
except for the azimuth:

.. math::
    \\begin{bmatrix}n_x \\\\ n_y\\end{bmatrix}=
    \\begin{bmatrix}\\cos\\psi & -\\sin\\psi \\\\ \\sin\\psi & \\cos\\psi\\end{bmatrix}
    \\begin{bmatrix}^em_x \\\\ ^em_y\\end{bmatrix}

where :math:`\\psi` is the azimuth angle. We normalize both sides to enforce
equal length of its vectors:

.. math::
    \\begin{bmatrix}N_x \\\\ N_y\\end{bmatrix}=
    \\begin{bmatrix}\\cos\\psi & -\\sin\\psi \\\\ \\sin\\psi & \\cos\\psi\\end{bmatrix}
    \\begin{bmatrix}M_x \\\\ M_y\\end{bmatrix}

with:

.. math::
    \\begin{array}{rcl}
    \\begin{bmatrix}N_x \\\\ N_y\\end{bmatrix} &=& \\frac{1}{\\sqrt{n_x^2+n_y^2}} \\begin{bmatrix}n_x \\\\ n_y\\end{bmatrix} \\\\
    \\begin{bmatrix}M_x \\\\ M_y\\end{bmatrix} &=& \\frac{1}{\\sqrt{^em_x^2+^em_y^2}} \\begin{bmatrix}^em_x \\\\ ^em_y\\end{bmatrix}
    \\end{array}

And now we just solve for the azimuth angle with:

.. math::
    \\begin{bmatrix}\\cos\\psi \\\\ \\sin\\psi \\end{bmatrix} =
    \\begin{bmatrix}M_x & M_y \\\\ -My & M_x \\end{bmatrix}
    \\begin{bmatrix}N_x \\\\ N_y \\end{bmatrix}

In the same manner as with the elevation and roll, we estimate the half-angle
values and define the azimuth quaternion as:

.. math::
    \\mathbf{q}_a =
    \\begin{pmatrix}\\cos\\frac{\\psi}{2} \\\\ 0 \\\\ 0 \\\\ \\sin\\frac{\\psi}{2} \\end{pmatrix}

Final Quaternion
----------------

Having computed all three quaternions, the estimation representing the
orientation of the rigid body is given by their product:

.. math::
    \\mathbf{q} = \\mathbf{q}_a\\mathbf{q}_e\\mathbf{q}_r

It should be noted that this algorithm does not evaluate any trigonometric
function at any step, although a singularity occurs in the FQA if the elevation
angle is :math:`\\pm 90°`, making :math:`\\cos\\theta=0`, but that is dealt
with at the computation of the first quaternion.

Footnotes
---------
.. [#] The local geomagnetic field can be obtained with the World Magnetic
    Model. See the `WMM documentation <../WMM.html>`_ page for further details.

References
----------
.. [Yun] Xiaoping Yun et al. (2008) A Simplified Quaternion-Based Algorithm for
    Orientation Estimation From Earth Gravity and Magnetic Field Measurements.
    https://ieeexplore.ieee.org/document/4419916

"""
from __future__ import annotations
from ahrs.common.orientation import q_conj
from ahrs.common.orientation import q_prod
from ahrs.utils.wmm import WMM
import cmath as cmath
import numpy
import numpy as np
__all__ = ['DEG2RAD', 'DYNAMIC_ELLIPTICITY', 'EARTH_ATMOSPHERE_MASS', 'EARTH_AUTHALIC_RADIUS', 'EARTH_AXIS_RATIO', 'EARTH_C20_DYN', 'EARTH_C20_GEO', 'EARTH_C22_DYN', 'EARTH_EQUATOR_RADIUS', 'EARTH_EQUIVOLUMETRIC_RADIUS', 'EARTH_FIRST_ECCENTRICITY', 'EARTH_FIRST_ECCENTRICITY_2', 'EARTH_FLATTENING', 'EARTH_FLATTENING_INV', 'EARTH_GM', 'EARTH_GM_1', 'EARTH_GM_2', 'EARTH_GM_GPSNAV', 'EARTH_J2', 'EARTH_LINEAR_ECCENTRICITY', 'EARTH_MASS', 'EARTH_MEAN_AXIAL_RADIUS', 'EARTH_MEAN_RADIUS', 'EARTH_POLAR_CURVATURE_RADIUS', 'EARTH_POLAR_RADIUS', 'EARTH_ROTATION', 'EARTH_SECOND_ECCENTRICITY', 'EARTH_SECOND_ECCENTRICITY_2', 'EARTH_SIDEREAL_DAY', 'EQUATORIAL_NORMAL_GRAVITY', 'FQA', 'JUPITER_EQUATOR_RADIUS', 'JUPITER_GM', 'JUPITER_J2', 'JUPITER_MASS', 'JUPITER_POLAR_RADIUS', 'JUPITER_ROTATION', 'LIGHT_SPEED', 'MAG', 'MARS_EQUATOR_RADIUS', 'MARS_GM', 'MARS_J2', 'MARS_MASS', 'MARS_POLAR_RADIUS', 'MARS_ROTATION', 'MEAN_NORMAL_GRAVITY', 'MERCURY_EQUATOR_RADIUS', 'MERCURY_GM', 'MERCURY_J2', 'MERCURY_MASS', 'MERCURY_POLAR_RADIUS', 'MERCURY_ROTATION', 'MOON_EQUATOR_RADIUS', 'MOON_GM', 'MOON_J2', 'MOON_MASS', 'MOON_POLAR_RADIUS', 'MOON_ROTATION', 'MUNICH_HEIGHT', 'MUNICH_LATITUDE', 'MUNICH_LONGITUDE', 'M_PI', 'NEPTUNE_EQUATOR_RADIUS', 'NEPTUNE_GM', 'NEPTUNE_J2', 'NEPTUNE_MASS', 'NEPTUNE_POLAR_RADIUS', 'NEPTUNE_ROTATION', 'NORMAL_GRAVITY_FORMULA', 'NORMAL_GRAVITY_POTENTIAL', 'PLUTO_EQUATOR_RADIUS', 'PLUTO_GM', 'PLUTO_MASS', 'PLUTO_POLAR_RADIUS', 'PLUTO_ROTATION', 'POLAR_NORMAL_GRAVITY', 'RAD2DEG', 'SATURN_EQUATOR_RADIUS', 'SATURN_GM', 'SATURN_J2', 'SATURN_MASS', 'SATURN_POLAR_RADIUS', 'SATURN_ROTATION', 'SOMIGLIANA_GRAVITY', 'UNIVERSAL_GRAVITATION_CODATA2014', 'UNIVERSAL_GRAVITATION_CODATA2018', 'UNIVERSAL_GRAVITATION_WGS84', 'URANUS_EQUATOR_RADIUS', 'URANUS_GM', 'URANUS_J2', 'URANUS_MASS', 'URANUS_POLAR_RADIUS', 'URANUS_ROTATION', 'VENUS_EQUATOR_RADIUS', 'VENUS_GM', 'VENUS_J2', 'VENUS_MASS', 'VENUS_POLAR_RADIUS', 'VENUS_ROTATION', 'WMM', 'cmath', 'np', 'q_conj', 'q_prod']
class FQA:
    """
    
        Factored Quaternion Algorithm
    
        Parameters
        ----------
        acc : numpy.ndarray, default: None
            N-by-3 array with N measurements of the gravitational acceleration.
        mag : numpy.ndarray, default: None
            N-by-3 array with N measurements of the geomagnetic field.
        mag_ref : numpy.ndarray, default: None
            Reference geomagnetic field. If None is given, defaults to the
            geomagnetic field of Munich, Germany.
    
        Attributes
        ----------
        acc : numpy.ndarray
            N-by-3 array with N accelerometer samples.
        mag : numpy.ndarray
            N-by-3 array with N magnetometer samples.
        m_ref : numpy.ndarray
            Normalized reference geomagnetic field.
        Q : numpy.ndarray
            Estimated attitude as quaternion.
    
        Raises
        ------
        ValueError
            When dimension of input arrays ``acc`` and ``mag`` are not equal.
    
        
    """
    def __init__(self, acc: numpy.ndarray = None, mag: numpy.ndarray = None, mag_ref: numpy.ndarray = None):
        ...
    def _compute_all(self) -> numpy.ndarray:
        """
        
                Estimate the quaternions given all data.
        
                Attributes ``acc`` and ``mag`` must contain data.
        
                Returns
                -------
                Q : numpy.ndarray
                    M-by-4 Array with all estimated quaternions, where M is the number
                    of samples.
        
                
        """
    def estimate(self, acc: numpy.ndarray = None, mag: numpy.ndarray = None) -> numpy.ndarray:
        """
        Attitude Estimation.
        
                Parameters
                ----------
                acc : numpy.ndarray
                    Sample of tri-axial Accelerometer.
                mag : numpy.ndarray
                    Sample of tri-axial Magnetometer.
        
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
