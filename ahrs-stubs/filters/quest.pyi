"""

QUEST
=====

QUaternion ESTimator as described by Shuster in [Shuster1981]_ and [Shuster1978]_.

We start to define the goal of finding an orthogonal matrix :math:`\\mathbf{A}`
that minimizes the loss function:

.. math::
    L(\\mathbf{A}) = \\frac{1}{2}\\sum_{i=1}^n |\\hat{\\mathbf{W}}_i - \\mathbf{A}\\hat{\\mathbf{V}}_i|^2

where :math:`a_i` are a set of non-negative weights such that :math:`\\sum_{i=1}^na_i=1`,
:math:`\\hat{\\mathbf{V}}_i` are nonparallel **reference vectors**, and
:math:`\\hat{\\mathbf{W}}_i` are the corresponding **observation vectors**.

The gain function :math:`g(\\mathbf{A})` is defined by

.. math::
    g(\\mathbf{A}) = 1 - L(\\mathbf{A}) = \\sum_{i=1}^na_i\\,\\hat{\\mathbf{W}}_i^T\\mathbf{A}\\hat{\\mathbf{V}}_i

The loss function :math:`L(\\mathbf{A})` is at its minimum when the gain
function :math:`g(\\mathbf{A})` is at its maximum. The gain function can be
reformulated as:

.. math::
    g(\\mathbf{A}) = \\sum_{i=1}^na_i\\mathrm{tr}\\big(\\hat{\\mathbf{W}}_i^T\\mathbf{A}\\hat{\\mathbf{V}}_i\\big) = \\mathrm{tr}(\\mathbf{AB}^T)

where :math:`\\mathrm{tr}` is the `trace <https://en.wikipedia.org/wiki/Trace_(linear_algebra)>`_
of a matrix, and :math:`\\mathbf{B}` is the **attitude profile matrix**:

.. math::
    \\mathbf{B} = \\sum_{i=1}^na_i\\hat{\\mathbf{W}}_i^T\\hat{\\mathbf{V}}_i

The quaternion :math:`\\bar{\\mathbf{q}}` representing a rotation is defined by
Shuster as:

.. math::
    \\bar{\\mathbf{q}} = \\begin{bmatrix}\\mathbf{Q} \\\\ q\\end{bmatrix}
    = \\begin{bmatrix}\\hat{\\mathbf{X}}\\sin\\frac{\\theta}{2} \\\\ \\cos\\frac{\\theta}{2}\\end{bmatrix}

where :math:`\\hat{\\mathbf{X}}` is the axis of rotation, and :math:`\\theta`
is the angle of rotation about :math:`\\hat{\\mathbf{X}}`.

.. warning::
    The definition of a quaternion used by Shuster sets the vector part
    :math:`\\mathbf{Q}` followed by the scalar part :math:`q`. This module,
    however, will return the estimated quaternion with the *scalar part first*
    and followed by the vector part: :math:`\\bar{\\mathbf{q}} = \\begin{bmatrix}q
    & \\mathbf{Q}\\end{bmatrix}`

Because the quaternion works as a versor, it must satisfy:

.. math::
    \\bar{\\mathbf{q}}^T\\bar{\\mathbf{q}} = |\\mathbf{Q}|^2 + q^2 = 1

The attitude matrix :math:`\\mathbf{A}` is related to the quaternion by:

.. math::
    \\mathbf{A}(\\bar{\\mathbf{q}}) = (q^2+\\mathbf{Q}\\cdot\\mathbf{Q})\\mathbf{I} + 2\\mathbf{QQ}^T + 2q\\lfloor\\mathbf{Q}\\rfloor_\\times

where :math:`\\mathbf{I}` is the identity matrix, and :math:`\\lfloor\\mathbf{Q}\\rfloor_\\times`
is the **antisymmetric matrix** of :math:`\\mathbf{Q}`, a.k.a. the
`skew-symmetric matrix <https://en.wikipedia.org/wiki/Skew-symmetric_matrix>`_:

.. math::
    \\lfloor\\mathbf{Q}\\rfloor_\\times =
    \\begin{bmatrix}0 & Q_3 & -Q_2 \\\\ -Q_3 & 0 & Q_1 \\\\ Q_2 & -Q_1 & 0\\end{bmatrix}

Now the gain function can be rewritten again, but in terms of quaternions:

.. math::
    g(\\bar{\\mathbf{q}}) = (q^2-\\mathbf{Q}\\cdot\\mathbf{Q})\\mathrm{tr}\\mathbf{B}^T + 2\\mathrm{tr}\\big(\\mathbf{QQ}^T\\mathbf{B}^T\\big) + 2q\\mathrm{tr}\\big(\\lfloor\\mathbf{Q}\\rfloor_\\times\\mathbf{B}^T\\big)

A further simplification gives:

.. math::
    g(\\bar{\\mathbf{q}}) = \\bar{\\mathbf{q}}^T\\mathbf{K}\\bar{\\mathbf{q}}

where the :math:`4\\times 4` matrix :math:`\\mathbf{K}` is given by:

.. math::
    \\mathbf{K} = \\begin{bmatrix} \\mathbf{S} - \\sigma\\mathbf{I} & \\mathbf{Z} \\\\ \\mathbf{Z}^T & \\sigma \\end{bmatrix}

using the helper values:

.. math::
    \\begin{array}{rcl}
    \\sigma &=& \\mathrm{tr}\\mathbf{B} \\\\ && \\\\
    \\mathbf{S} &=& \\mathbf{B} + \\mathbf{B}^T \\\\ && \\\\
    \\mathbf{Z} &=& \\sum_{i=1}^na_i\\big(\\hat{\\mathbf{W}}_i\\times\\hat{\\mathbf{V}}_i\\big)
    \\end{array}

.. note::
    :math:`\\mathbf{Z}` can be also defined from :math:`\\lfloor\\mathbf{Z}\\rfloor_\\times = \\mathbf{B} - \\mathbf{B}^T`

A new gain function :math:`g'(\\bar{\\mathbf{q}})` with `Lagrange multipliers
<https://en.wikipedia.org/wiki/Lagrange_multiplier>`_ is defined:

.. math::
    g'(\\bar{\\mathbf{q}}) = \\bar{\\mathbf{q}}^T\\mathbf{K}\\bar{\\mathbf{q}} - \\lambda\\bar{\\mathbf{q}}^T\\bar{\\mathbf{q}}

It is verified that :math:`\\mathbf{K}\\bar{\\mathbf{q}}=\\lambda\\bar{\\mathbf{q}}`.
Thus, :math:`g(\\bar{\\mathbf{q}})` will be maximized if :math:`\\bar{\\mathbf{q}}_\\mathrm{opt}`
is chosen to be the eigenvector of :math:`\\mathbf{K}` belonging to the largest
eigenvalue of :math:`\\mathbf{K}`:

.. math::
    \\mathbf{K}\\bar{\\mathbf{q}}_\\mathrm{opt} = \\lambda_\\mathrm{max}\\bar{\\mathbf{q}}_\\mathrm{opt}

which is the desired result. This equation can be rearranged to read, for any
eigenvalue :math:`\\lambda`:

.. math::
    \\lambda = \\sigma + \\mathbf{Z}\\cdot\\mathbf{Y}

where :math:`\\mathbf{Y}` is the `Gibbs vector
<https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Rodrigues_vector>`_,
a.k.a. the **Rodrigues vector**, defined as:

.. math::
    \\mathbf{Y} = \\frac{\\mathbf{Q}}{q} = \\hat{\\mathbf{X}}\\tan\\frac{\\theta}{2}

rewriting the quaternion as:

.. math::
    \\bar{\\mathbf{q}} = \\frac{1}{\\sqrt{1+|\\mathbf{Y}|^2}} = \\begin{bmatrix}\\mathbf{Y}\\\\ 1 \\end{bmatrix}

:math:`\\mathbf{Y}` and :math:`\\bar{\\mathbf{q}}` are representations of the
optimal attitude solution when :math:`\\lambda` is equal to :math:`\\lambda_\\mathrm{max}`,
leading to an equation for the eigenvalues:

.. math::
    \\lambda = \\sigma + \\mathbf{Z}^T \\frac{1}{(\\lambda+\\sigma)\\mathbf{I}-\\mathbf{S}}\\mathbf{Z}

which is equivalent to the characteristic equation of the eigenvalues of :math:`\\mathbf{K}`

With the aid of `Cayley-Hamilton theorem <https://en.wikipedia.org/wiki/Cayley%E2%80%93Hamilton_theorem>`_
we can get rid of the Gibbs vector to find a more convenient expression of the
characteristic equation:

.. math::
    \\lambda^4-(a+b)\\lambda^2-c\\lambda+(ab+c\\sigma-d)=0

where:

.. math::
    \\begin{array}{rcl}
    a &=& \\sigma^2-\\kappa \\\\ && \\\\
    b &=& \\sigma^2 + \\mathbf{Z}^T+\\mathbf{Z} \\\\ && \\\\
    c &=& \\Delta + \\mathbf{Z}^T\\mathbf{SZ} \\\\ && \\\\
    d &=& \\mathbf{Z}^T\\mathbf{S}^2\\mathbf{Z} \\\\ && \\\\
    \\sigma &=& \\frac{1}{2}\\mathrm{tr}\\mathbf{S} \\\\ && \\\\
    \\kappa &=& \\mathrm{tr}\\big(\\mathrm{adj}(\\mathbf{S})\\big) \\\\ && \\\\
    \\Delta &=& \\mathrm{det}(\\mathbf{S})
    \\end{array}

To find :math:`\\lambda` we implement the `Newton-Raphson method
<https://en.wikipedia.org/wiki/Newton%27s_method>`_ using the sum of the
weights :math:`a_i` (in the beginning is constrained to be equal to 1) as a
starting value.

.. math::
    \\lambda_{t+1} \\gets \\lambda_t - \\frac{f(\\lambda)}{f'(\\lambda)}
    = \\lambda_t - \\frac{\\lambda^4-(a+b)\\lambda^2-c\\lambda+(ab+c\\sigma-d)}{4\\lambda^3-2(a+b)\\lambda-c}

For sensor accuracies better than 1 arc-min (1 degree) the accuracy of a 64-bit
word is exhausted after only one iteration.

Finally, the **optimal quaternion** describing the attitude is found as:

.. math::
    \\bar{\\mathbf{q}}_\\mathrm{opt} = \\frac{1}{\\sqrt{\\gamma^2+|\\mathbf{Y}|^2}} = \\begin{bmatrix}\\mathbf{Y}\\\\ \\gamma \\end{bmatrix}

with:

.. math::
    \\begin{array}{rcl}
    \\mathbf{X} &=& (\\alpha\\mathbf{I} + \\beta\\mathbf{S} + \\mathbf{S}^2)\\mathbf{Z} \\\\ && \\\\
    \\gamma &=& (\\lambda + \\sigma)\\alpha - \\Delta \\\\ && \\\\
    \\alpha &=& \\lambda^2 - \\sigma^2 + \\kappa \\\\ && \\\\
    \\beta &=& \\lambda - \\sigma
    \\end{array}

This solution can still lead to an indeterminant result if both :math:`\\gamma`
and :math:`\\mathbf{X}` vanish simultaneously. :math:`\\gamma` vanishes if and
only if the angle of rotation is equal to :math:`\\pi`, even if
:math:`\\mathbf{X}` does not vanish along.

References
----------
.. [Shuster1981] Shuster, M.D. and Oh, S.D. "Three-Axis Attitude Determination
    from Vector Observations," Journal of Guidance and Control, Vol.4, No.1,
    Jan.-Feb. 1981, pp. 70-77.
.. [Shuster1978] Shuster, Malcom D. Approximate Algorithms for Fast Optimal
    Attitude Computation, AIAA Guidance and Control Conference. August 1978.
    (http://www.malcolmdshuster.com/Pub_1978b_C_PaloAlto_scan.pdf)

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
__all__ = ['DEG2RAD', 'DYNAMIC_ELLIPTICITY', 'EARTH_ATMOSPHERE_MASS', 'EARTH_AUTHALIC_RADIUS', 'EARTH_AXIS_RATIO', 'EARTH_C20_DYN', 'EARTH_C20_GEO', 'EARTH_C22_DYN', 'EARTH_EQUATOR_RADIUS', 'EARTH_EQUIVOLUMETRIC_RADIUS', 'EARTH_FIRST_ECCENTRICITY', 'EARTH_FIRST_ECCENTRICITY_2', 'EARTH_FLATTENING', 'EARTH_FLATTENING_INV', 'EARTH_GM', 'EARTH_GM_1', 'EARTH_GM_2', 'EARTH_GM_GPSNAV', 'EARTH_J2', 'EARTH_LINEAR_ECCENTRICITY', 'EARTH_MASS', 'EARTH_MEAN_AXIAL_RADIUS', 'EARTH_MEAN_RADIUS', 'EARTH_POLAR_CURVATURE_RADIUS', 'EARTH_POLAR_RADIUS', 'EARTH_ROTATION', 'EARTH_SECOND_ECCENTRICITY', 'EARTH_SECOND_ECCENTRICITY_2', 'EARTH_SIDEREAL_DAY', 'EQUATORIAL_NORMAL_GRAVITY', 'GRAVITY', 'JUPITER_EQUATOR_RADIUS', 'JUPITER_GM', 'JUPITER_J2', 'JUPITER_MASS', 'JUPITER_POLAR_RADIUS', 'JUPITER_ROTATION', 'LIGHT_SPEED', 'MAG', 'MARS_EQUATOR_RADIUS', 'MARS_GM', 'MARS_J2', 'MARS_MASS', 'MARS_POLAR_RADIUS', 'MARS_ROTATION', 'MEAN_NORMAL_GRAVITY', 'MERCURY_EQUATOR_RADIUS', 'MERCURY_GM', 'MERCURY_J2', 'MERCURY_MASS', 'MERCURY_POLAR_RADIUS', 'MERCURY_ROTATION', 'MOON_EQUATOR_RADIUS', 'MOON_GM', 'MOON_J2', 'MOON_MASS', 'MOON_POLAR_RADIUS', 'MOON_ROTATION', 'MUNICH_HEIGHT', 'MUNICH_LATITUDE', 'MUNICH_LONGITUDE', 'M_PI', 'NEPTUNE_EQUATOR_RADIUS', 'NEPTUNE_GM', 'NEPTUNE_J2', 'NEPTUNE_MASS', 'NEPTUNE_POLAR_RADIUS', 'NEPTUNE_ROTATION', 'NORMAL_GRAVITY_FORMULA', 'NORMAL_GRAVITY_POTENTIAL', 'PLUTO_EQUATOR_RADIUS', 'PLUTO_GM', 'PLUTO_MASS', 'PLUTO_POLAR_RADIUS', 'PLUTO_ROTATION', 'POLAR_NORMAL_GRAVITY', 'QUEST', 'RAD2DEG', 'SATURN_EQUATOR_RADIUS', 'SATURN_GM', 'SATURN_J2', 'SATURN_MASS', 'SATURN_POLAR_RADIUS', 'SATURN_ROTATION', 'SOMIGLIANA_GRAVITY', 'UNIVERSAL_GRAVITATION_CODATA2014', 'UNIVERSAL_GRAVITATION_CODATA2018', 'UNIVERSAL_GRAVITATION_WGS84', 'URANUS_EQUATOR_RADIUS', 'URANUS_GM', 'URANUS_J2', 'URANUS_MASS', 'URANUS_POLAR_RADIUS', 'URANUS_ROTATION', 'VENUS_EQUATOR_RADIUS', 'VENUS_GM', 'VENUS_J2', 'VENUS_MASS', 'VENUS_POLAR_RADIUS', 'VENUS_ROTATION', 'WGS', 'WMM', 'cmath', 'cosd', 'np', 'sind', 'skew']
class QUEST:
    """
    
        QUaternion ESTimator
    
        Parameters
        ----------
        acc : numpy.ndarray, default: None
            N-by-3 array with measurements of acceleration in in m/s^2
        mag : numpy.ndarray, default: None
            N-by-3 array with measurements of magnetic field in mT
        weights : array-like
            Array with two weights. One per sensor measurement.
        magnetic_dip : float
            Local magnetic inclination angle, in degrees.
        gravity : float
            Local normal gravity, in m/s^2.
    
        Attributes
        ----------
        acc : numpy.ndarray
            N-by-3 array with N accelerometer samples.
        mag : numpy.ndarray
            N-by-3 array with N magnetometer samples.
        w : numpy.ndarray
            Weights for each observation.
    
        Raises
        ------
        ValueError
            When dimension of input arrays ``acc`` and ``mag`` are not equal.
    
        
    """
    def __init__(self, acc: numpy.ndarray = None, mag: numpy.ndarray = None, **kw):
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
