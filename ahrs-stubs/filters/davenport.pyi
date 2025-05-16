"""

Davenport's q-Method
====================

In 1965 `Grace Wahba <https://en.wikipedia.org/wiki/Grace_Wahba>`_ came up with
a simple, yet very intuitive, way to describe the problem of finding a rotation
between two coordinate systems.

Given a set of :math:`N` vector measurements :math:`\\mathbf{u}` in the body
coordinate system, an optimal attitude matrix :math:`\\mathbf{A}` would
minimize the `loss function <https://en.wikipedia.org/wiki/Loss_function>`_:

.. math::
    L(\\mathbf{A}) = \\frac{1}{2}\\sum_{i=1}^Nw_i|u_i-\\mathbf{A}v_i|^2

where :math:`u_i` is the i-th vector measurement in the body frame, :math:`v_i`
is the i-th vector in the reference frame, and :math:`w_i` are a set of :math:`N`
nonnegative weights for each observation. This famous formulation is known as
`Wahba's problem <https://en.wikipedia.org/wiki/Wahba%27s_problem>`_.

A first elegant solution was proposed by [Davenport1968]_ that solves this in
terms of quaternions, yielding a unique optimal solution. The corresponding
**gain function** is defined as:

.. math::
    g(\\mathbf{A}) = 1 - L(\\mathbf{A}) = \\sum_{i=1}^Nw_i\\mathbf{U}^T\\mathbf{AV}

The gain function is at maximum when the loss function :math:`L(\\mathbf{A})`
is at minimum. The goal is, then, to find the optimal attitude matrix
:math:`\\mathbf{A}`, which *maximizes* :math:`g(\\mathbf{A})`. We first notice
that:

.. math::
    \\begin{array}{rl}
    g(\\mathbf{A}) =& \\sum_{i=1}^Nw_i\\mathrm{tr}\\big(\\mathbf{U}_i^T\\mathbf{AV}_i\\big) \\\\
    =& \\mathrm{tr}(\\mathbf{AB}^T)
    \\end{array}

where :math:`\\mathrm{tr}` denotes the `trace <https://en.wikipedia.org/wiki/Trace_(linear_algebra)>`_
of a matrix, and :math:`\\mathbf{B}` is the *attitude profile matrix*:

.. math::
    \\mathbf{B} = \\sum_{i=1}^Nw_i\\mathbf{UV}

Now, we must parametrize the attitude matrix in terms of a quaternion :math:`\\mathbf{q}`:

.. math::
    \\mathbf{A}(\\mathbf{q}) = (q_w^2-\\mathbf{q}_v\\cdot\\mathbf{q}_v)\\mathbf{I}_3+2\\mathbf{q}_v\\mathbf{q}_v^T-2q_w\\lfloor\\mathbf{q}\\rfloor_\\times

where :math:`\\mathbf{I}_3` is a :math:`3\\times 3` identity matrix, and the
expression :math:`\\lfloor \\mathbf{x}\\rfloor_\\times` is the `skew-symmetric
matrix <https://en.wikipedia.org/wiki/Skew-symmetric_matrix>`_ of a vector
:math:`\\mathbf{x}`. See the `quaternion page <../quaternion.html>`_ for further
details about this representation mapping.

The gain function, in terms of quaternion, becomes:

.. math::
    g(\\mathbf{q}) = (q_w^2-\\mathbf{q}_v\\cdot\\mathbf{q}_v)\\mathrm{tr}\\mathbf{B}^T + 2\\mathrm{tr}\\big(\\mathbf{q}_v\\mathbf{q}_v^T\\mathbf{B}^T\\big) + 2q_w\\mathrm{tr}(\\lfloor\\mathbf{q}\\rfloor_\\times\\mathbf{B}^T)

A simpler expression, using helper quantities, can be a bilinear relationship
of the form:

.. math::
    g(\\mathbf{q}) = \\mathbf{q}^T\\mathbf{Kq}

where the :math:`4\\times 4` matrix :math:`\\mathbf{K}` is built with:

.. math::
    \\mathbf{K} = \\begin{bmatrix}
    \\sigma & \\mathbf{z}^T \\\\
    \\mathbf{z} & \\mathbf{S}-\\sigma\\mathbf{I}_3
    \\end{bmatrix}

using the intermediate values:

.. math::
    \\begin{array}{rcl}
    \\sigma &=& \\mathrm{tr}\\mathbf{B} \\\\
    \\mathbf{S} &=& \\mathbf{B}+\\mathbf{B}^T \\\\
    \\mathbf{z} &=& \\begin{bmatrix}B_{23}-B_{32} \\\\ B_{31}-B_{13} \\\\ B_{12}-B_{21}\\end{bmatrix}
    \\end{array}

The optimal quaternion :math:`\\hat{\\mathbf{q}}`, which parametrizes the
optimal attitude matrix, is an eigenvector of :math:`\\mathbf{K}`. With the
help of `Lagrange multipliers <https://en.wikipedia.org/wiki/Lagrange_multiplier>`_,
:math:`g(\\mathbf{q})` is maximized if the eigenvector corresponding to the
largest eigenvalue :math:`\\lambda` is chosen.

.. math::
    \\mathbf{K}\\hat{\\mathbf{q}} = \\lambda\\hat{\\mathbf{q}}

The biggest disadvantage of this method is its computational load in the last
step of computing the eigenvalues and eigenvectors to find the optimal
quaternion.

References
----------
.. [Davenport1968] Paul B. Davenport. A Vector Approach to the Algebra of Rotations
    with Applications. NASA Technical Note D-4696. August 1968.
    (https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19680021122.pdf)
.. [Lerner2] Lerner, G. M. "Three-Axis Attitude Determination" in Spacecraft
    Attitude Determination and Control, edited by J.R. Wertz. 1978. p. 426-428.

"""
from __future__ import annotations
from ahrs.common.mathfuncs import cosd
from ahrs.common.mathfuncs import sind
from ahrs.common.mathfuncs import skew
from ahrs.utils.wgs84 import WGS
from ahrs.utils.wmm import WMM
import cmath as cmath
import numpy
import numpy as np
__all__ = ['DEG2RAD', 'DYNAMIC_ELLIPTICITY', 'Davenport', 'EARTH_ATMOSPHERE_MASS', 'EARTH_AUTHALIC_RADIUS', 'EARTH_AXIS_RATIO', 'EARTH_C20_DYN', 'EARTH_C20_GEO', 'EARTH_C22_DYN', 'EARTH_EQUATOR_RADIUS', 'EARTH_EQUIVOLUMETRIC_RADIUS', 'EARTH_FIRST_ECCENTRICITY', 'EARTH_FIRST_ECCENTRICITY_2', 'EARTH_FLATTENING', 'EARTH_FLATTENING_INV', 'EARTH_GM', 'EARTH_GM_1', 'EARTH_GM_2', 'EARTH_GM_GPSNAV', 'EARTH_J2', 'EARTH_LINEAR_ECCENTRICITY', 'EARTH_MASS', 'EARTH_MEAN_AXIAL_RADIUS', 'EARTH_MEAN_RADIUS', 'EARTH_POLAR_CURVATURE_RADIUS', 'EARTH_POLAR_RADIUS', 'EARTH_ROTATION', 'EARTH_SECOND_ECCENTRICITY', 'EARTH_SECOND_ECCENTRICITY_2', 'EARTH_SIDEREAL_DAY', 'EQUATORIAL_NORMAL_GRAVITY', 'GRAVITY', 'JUPITER_EQUATOR_RADIUS', 'JUPITER_GM', 'JUPITER_J2', 'JUPITER_MASS', 'JUPITER_POLAR_RADIUS', 'JUPITER_ROTATION', 'LIGHT_SPEED', 'MAG', 'MARS_EQUATOR_RADIUS', 'MARS_GM', 'MARS_J2', 'MARS_MASS', 'MARS_POLAR_RADIUS', 'MARS_ROTATION', 'MEAN_NORMAL_GRAVITY', 'MERCURY_EQUATOR_RADIUS', 'MERCURY_GM', 'MERCURY_J2', 'MERCURY_MASS', 'MERCURY_POLAR_RADIUS', 'MERCURY_ROTATION', 'MOON_EQUATOR_RADIUS', 'MOON_GM', 'MOON_J2', 'MOON_MASS', 'MOON_POLAR_RADIUS', 'MOON_ROTATION', 'MUNICH_HEIGHT', 'MUNICH_LATITUDE', 'MUNICH_LONGITUDE', 'M_PI', 'NEPTUNE_EQUATOR_RADIUS', 'NEPTUNE_GM', 'NEPTUNE_J2', 'NEPTUNE_MASS', 'NEPTUNE_POLAR_RADIUS', 'NEPTUNE_ROTATION', 'NORMAL_GRAVITY_FORMULA', 'NORMAL_GRAVITY_POTENTIAL', 'PLUTO_EQUATOR_RADIUS', 'PLUTO_GM', 'PLUTO_MASS', 'PLUTO_POLAR_RADIUS', 'PLUTO_ROTATION', 'POLAR_NORMAL_GRAVITY', 'RAD2DEG', 'SATURN_EQUATOR_RADIUS', 'SATURN_GM', 'SATURN_J2', 'SATURN_MASS', 'SATURN_POLAR_RADIUS', 'SATURN_ROTATION', 'SOMIGLIANA_GRAVITY', 'UNIVERSAL_GRAVITATION_CODATA2014', 'UNIVERSAL_GRAVITATION_CODATA2018', 'UNIVERSAL_GRAVITATION_WGS84', 'URANUS_EQUATOR_RADIUS', 'URANUS_GM', 'URANUS_J2', 'URANUS_MASS', 'URANUS_POLAR_RADIUS', 'URANUS_ROTATION', 'VENUS_EQUATOR_RADIUS', 'VENUS_GM', 'VENUS_J2', 'VENUS_MASS', 'VENUS_POLAR_RADIUS', 'VENUS_ROTATION', 'WGS', 'WMM', 'cmath', 'cosd', 'np', 'sind', 'skew']
class Davenport:
    """
    
        Davenport's q-Method for attitude estimation
    
        Parameters
        ----------
        acc : numpy.ndarray, default: None
            N-by-3 array with measurements of acceleration in in m/s^2
        mag : numpy.ndarray, default: None
            N-by-3 array with measurements of magnetic field in mT
        weights : array-like
            Array with two weights used in each observation.
        magnetic_dip : float
            Magnetic Inclination angle, in degrees. Defaults to magnetic dip of
            Munich, Germany.
        gravity : float
            Normal gravity, in m/s^2. Defaults to normal gravity of Munich,
            Germany.
    
        Attributes
        ----------
        acc : numpy.ndarray
            N-by-3 array with N accelerometer samples.
        mag : numpy.ndarray
            N-by-3 array with N magnetometer samples.
        w : numpy.ndarray
            Weights of each observation.
    
        Raises
        ------
        ValueError
            When dimension of input arrays ``acc`` and ``mag`` are not equal.
    
        
    """
    def __init__(self, acc: numpy.ndarray = None, mag: numpy.ndarray = None, **kw):
        ...
    def _compute_all(self) -> numpy.ndarray:
        """
        
                Estimate all quaternions given all data.
        
                Attributes ``acc`` and ``mag`` must contain data.
        
                Returns
                -------
                Q : array
                    M-by-4 Array with all estimated quaternions, where M is the number
                    of samples.
        
                
        """
    def estimate(self, acc: numpy.ndarray = None, mag: numpy.ndarray = None) -> numpy.ndarray:
        """
        
                Attitude Estimation
        
                Parameters
                ----------
                acc : numpy.ndarray
                    Sample of tri-axial Accelerometer in m/s^2
                mag : numpy.ndarray
                    Sample of tri-axial Magnetometer in T
        
                Returns
                -------
                q : numpy.ndarray
                    Estimated attitude as a quaternion.
        
                
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
