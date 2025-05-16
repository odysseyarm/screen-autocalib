"""

Fast Linear Attitude Estimator
==============================

The Fast Linear Attitude Estimator (FLAE) obtains the attitude quaternion with
an eigenvalue-based solution as proposed by [Wu]_.

A symbolic solution to the corresponding characteristic polynomial is also
derived for a higher computation speed.

One-Dimensional Fusion
----------------------

We assume that we have a single observable (can be measured) frame. The sensor
outputs can be rotated with a :math:`3\\times 3` `Direction Cosine Matrix
<../dcm.html>`_ :math:`\\mathbf{C}` using:

.. math::
    \\mathbf{D}^b = \\mathbf{CD}^r

where :math:`\\mathbf{D}^b=\\begin{bmatrix}D_x^b & D_y^b & D_z^b\\end{bmatrix}^T`
is the observation vector in body frame and
:math:`\\mathbf{D}^r=\\begin{bmatrix}D_x^r & D_y^r & D_z^r\\end{bmatrix}^T` is
the observation vector in reference frame. To put it in terms of a quaternion,
we define the loss function :math:`\\mathbf{f}_D(\\mathbf{q})` as:

.. math::
    \\mathbf{f}_D(\\mathbf{q}) \\triangleq \\mathbf{CD}^r - \\mathbf{D}^b

where the quaternion :math:`\\mathbf{q}` is defined as:

.. math::
    \\begin{array}{rcl}
    \\mathbf{q}&=&\\begin{pmatrix}q_w & q_x & q_y & q_z\\end{pmatrix}^T \\\\
    &=& \\begin{pmatrix}\\cos\\frac{\\theta}{2} & n_x\\sin\\frac{\\theta}{2} & n_y\\sin\\frac{\\theta}{2} & n_z\\sin\\frac{\\theta}{2}\\end{pmatrix}^T
    \\end{array}

The purpose is to *minimize the loss function*. We start by expanding
:math:`\\mathbf{f}_D(\\mathbf{q})`:

.. math::
    \\begin{array}{rcl}
    \\mathbf{f}_D(\\mathbf{q}) &=& \\mathbf{CD}^r - \\mathbf{D}^b \\\\
    &=& \\mathbf{P}_D\\mathbf{q} - \\mathbf{D}^b \\\\
    &=& (D_x^r\\mathbf{P}_1 + D_y^r\\mathbf{P}_2 + D_z^r\\mathbf{P}_3)\\mathbf{q} - \\mathbf{D}^b \\\\
    &=& D_x^r\\mathbf{C}_1 + D_y^r\\mathbf{C}_2 + D_z^r\\mathbf{C}_3 - \\mathbf{D}^b
    \\end{array}

where :math:`\\mathbf{C}_1`, :math:`\\mathbf{C}_2` and :math:`\\mathbf{C}_3`
are the columns of :math:`\\mathbf{C}` represented as:

.. math::
    \\begin{array}{rcl}
    \\mathbf{C}_1 &=& \\mathbf{P}_1\\mathbf{q} = \\begin{bmatrix}q_w^2+q_x^2-q_y^2-q_z^2 \\\\ 2(q_xq_y + q_wq_z) \\\\ 2(q_xq_z - q_wq_y) \\end{bmatrix} \\\\ && \\\\
    \\mathbf{C}_2 &=& \\mathbf{P}_2\\mathbf{q} = \\begin{bmatrix}2(q_xq_y - q_wq_z) \\\\ q_w^2-q_x^2+q_y^2-q_z^2 \\\\2(q_wq_x + q_yq_z) \\end{bmatrix} \\\\ && \\\\
    \\mathbf{C}_3 &=& \\mathbf{P}_3\\mathbf{q} = \\begin{bmatrix}2(q_xq_z + q_wq_y) \\\\ 2(q_yq_z - q_wq_x) \\\\ q_w^2-q_x^2-q_y^2+q_z^2 \\end{bmatrix}
    \\end{array}

When :math:`\\mathbf{q}` is optimal, it satisfies:

.. math::
    \\mathbf{q} = \\mathbf{P}_D^\\dagger \\mathbf{D}^b

where :math:`\\mathbf{P}_D^\\dagger` is the `pseudo-inverse
<https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse>`_ of
:math:`\\mathbf{P}_D` *if and only if* it has full rank.

.. note::
    A matrix is said to have **full rank** if its `rank
    <https://en.wikipedia.org/wiki/Rank_(linear_algebra)>`_ is equal to the
    largest possible for a matrix of the same dimensions, which is the lesser
    of the number of rows and columns.

The analytical form of any pseudo-inverse is normally difficult to obtain, but
thanks to the orthogonality of :math:`\\mathbf{P}_D` we get:

.. math::
    \\mathbf{P}_D^\\dagger = \\mathbf{P}_D^T = D_x^r\\mathbf{P}_1^T + D_y^r\\mathbf{P}_2^T + D_z^r\\mathbf{P}_3^T

The orientation :math:`\\mathbf{q}` is obtained from:

.. math::
    \\mathbf{P}_D^\\dagger\\mathbf{D}^b - \\mathbf{q} = \\mathbf{Gq}

Solving :math:`\\mathbf{Gq}=0` (*if and only if* :math:`\\mathrm{det}(\\mathbf{G})=0`)
using elementary row transformations we obtain the wanted orthonormal quaternion.

N-Dimensional Fusion
--------------------

We assume having :math:`n` observation equations, such that the error residual
vector is given by augmenting :math:`\\mathbf{f}_D(\\mathbf{q})` as:

.. math::
    \\mathbf{f}_{\\Sigma D}(\\mathbf{q}) =
    \\begin{bmatrix}
    \\sqrt{a_1}(\\mathbf{P}_{D_1}\\mathbf{q}-D_1^b) \\\\
    \\sqrt{a_2}(\\mathbf{P}_{D_2}\\mathbf{q}-D_2^b) \\\\
    \\vdots \\\\
    \\sqrt{a_n}(\\mathbf{P}_{D_n}\\mathbf{q}-D_n^b)
    \\end{bmatrix}

When :math:`\\mathbf{f}_{\\Sigma D}(\\mathbf{q})=0`, the equation satisfies:

.. math::
    \\begin{array}{rcl}
    \\mathbf{P}_{\\Sigma D}\\mathbf{q} &=& \\mathbf{D}_\\Sigma^b \\\\
    \\begin{bmatrix}
    \\sqrt{a_1}\\mathbf{P}_{D_1} \\\\
    \\sqrt{a_2}\\mathbf{P}_{D_2} \\\\
    \\vdots \\\\
    \\sqrt{a_n}\\mathbf{P}_{D_n}
    \\end{bmatrix}\\mathbf{q} &=&
    \\begin{bmatrix}
    \\sqrt{a_1}\\mathbf{D}_1^b \\\\
    \\sqrt{a_2}\\mathbf{D}_2^b \\\\
    \\vdots \\\\
    \\sqrt{a_n}\\mathbf{D}_n^b
    \\end{bmatrix}
    \\end{array}

Intuitively, we would solve it with :math:`\\mathbf{q}=\\mathbf{P}_{\\Sigma D}^\\dagger\\mathbf{D}_\\Sigma^b`,
but the pseudo-inverse of :math:`\\mathbf{P}_{\\Sigma D}` is very difficult to
compute. However, it is possible to transform the equation by the pseudo-inverse
matrices of :math:`\\mathbf{q}` and :math:`\\mathbf{D}_\\Sigma^b`:

.. math::
    \\mathbf{q}^\\dagger = (\\mathbf{D}_\\Sigma^b)^\\dagger \\mathbf{P}_{\\Sigma D}

:math:`\\mathbf{P}_{\\Sigma D}` can be further expanded into:

.. math::
    \\mathbf{P}_{\\Sigma D} = \\mathbf{U}_{D_x}\\mathbf{P}_1 + \\mathbf{U}_{D_y}\\mathbf{P}_2 + \\mathbf{U}_{D_z}\\mathbf{P}_3

where :math:`\\mathbf{P}_1`, :math:`\\mathbf{P}_2` and :math:`\\mathbf{P}_3`
are :math:`3\\times 4` matrices, and :math:`\\mathbf{U}_{D_x}`,
:math:`\\mathbf{U}_{D_y}` and :math:`\\mathbf{U}_{D_z}` are :math:`3n\\times 3`
matrices. Hence,

.. math::
    (\\mathbf{D}_\\Sigma^b)^\\dagger\\mathbf{P}_{\\Sigma D} = (\\mathbf{D}_\\Sigma^b)^\\dagger\\mathbf{U}_{D_x}\\mathbf{P}_1 + (\\mathbf{D}_\\Sigma^b)^\\dagger\\mathbf{U}_{D_y}\\mathbf{P}_2 + (\\mathbf{D}_\\Sigma^b)^\\dagger\\mathbf{U}_{D_z}\\mathbf{P}_3

The fusion equation finally arrives to:

.. math::
    \\mathbf{H}_x\\mathbf{P}_1 + \\mathbf{H}_y\\mathbf{P}_2 + \\mathbf{H}_z\\mathbf{P}_3 - \\mathbf{q}^\\dagger = \\mathbf{0}_{1\\times 4}

where :math:`\\mathbf{H}_x`, :math:`\\mathbf{H}_y` and :math:`\\mathbf{H}_z`
are :math:`1\\times 3` matrices

.. math::
    \\begin{array}{rcl}
    \\mathbf{H}_x &= (\\mathbf{D}_\\Sigma^b)^\\dagger\\mathbf{U}_{D_x} =&
    \\begin{bmatrix} \\sum_{i=1}^n a_iD_{x,i}^rD_{x,i}^b & \\sum_{i=1}^n a_iD_{x,i}^rD_{y,i}^b & \\sum_{i=1}^n a_iD_{x,i}^rD_{z,i}^b & \\end{bmatrix} \\\\ && \\\\
    \\mathbf{H}_y &= (\\mathbf{D}_\\Sigma^b)^\\dagger\\mathbf{U}_{D_y} =&
    \\begin{bmatrix} \\sum_{i=1}^n a_iD_{y,i}^rD_{x,i}^b & \\sum_{i=1}^n a_iD_{y,i}^rD_{y,i}^b & \\sum_{i=1}^n a_iD_{y,i}^rD_{z,i}^b & \\end{bmatrix} \\\\ && \\\\
    \\mathbf{H}_z &= (\\mathbf{D}_\\Sigma^b)^\\dagger\\mathbf{U}_{D_z} =&
    \\begin{bmatrix} \\sum_{i=1}^n a_iD_{z,i}^rD_{x,i}^b & \\sum_{i=1}^n a_iD_{z,i}^rD_{y,i}^b & \\sum_{i=1}^n a_iD_{z,i}^rD_{z,i}^b & \\end{bmatrix}
    \\end{array}

Refactoring the equation with a transpose operation, we obtain:

.. math::
    \\begin{array}{rcl}
    \\mathbf{P}_1^T\\mathbf{H}_x^T + \\mathbf{P}_2^T\\mathbf{H}_y^T + \\mathbf{P}_3^T\\mathbf{H}_z^T - \\mathbf{q} &=& \\mathbf{0} \\\\
    (\\mathbf{W} - \\mathbf{I})\\mathbf{q} &=& \\mathbf{0}
    \\end{array}

where the elements of :math:`\\mathbf{W}` are given by:

.. math::
    \\mathbf{W} =
    \\begin{bmatrix}
    H_{x1} + H_{y2} + H_{z3} & -H_{y3} + H_{z2} & -H_{z1} + H_{x3} & -H_{x2} + H_{y1} \\\\
    -H_{y3} + H_{z2} & H_{x1} - H_{y2} - H_{z3} & H_{x2} + H_{y1} & H_{x3} + H_{z1} \\\\
    -H_{z1} + H_{x3} & H_{x2} + H_{y1} & H_{y2} - H_{x1} - H_{z3} & H_{y3} + H_{z2} \\\\
    -H_{x2} + H_{y1} & H_{x3} + H_{z1} & H_{y3} + H_{x2} & H_{z3} - H_{y2} - H_{x1}
    \\end{bmatrix}

Eigenvector solution
--------------------

The simplest solution is to find the eigenvector corresponding to the largest
eigenvalue of :math:`\\mathbf{W}`, as used by `Davenport's <davenport.html>`_
q-method.

This has the advantage of returning a normalized and valid quaternion, which is
used to represent the attitude.

The disadvantages include its computational load and, mainly,
:math:`(\\mathbf{W}-\\mathbf{I})` suffering from rank-deficient problems in the
sensor outputs.

Characteristic Polynomial
-------------------------

The fusion equation can be transformed by adding a small quaternion error
:math:`\\epsilon \\mathbf{q}`

.. math::
    \\mathbf{Wq} = (1+\\epsilon)\\mathbf{q}

recognizing that :math:`1+\\epsilon` is an eigenvalue of :math:`\\mathbf{W}`
the problem is now shifted to find the eigenvalue that is closest to 1.

Analytically the calculation of the eigenvalue of :math:`\\mathbf{W}` builds
first its characteristic polynomial as:

.. math::
    \\begin{array}{rcl}
    f(\\lambda) &=& \\mathrm{det}(\\mathbf{W}-\\lambda\\mathbf{I}_{4\\times 4}) \\\\
    &=& \\lambda^4 + \\tau_1\\lambda^2 + \\tau_2\\lambda + \\tau_3
    \\end{array}

where the coefficients are obtained from:

.. math::
    \\begin{array}{rcl}
    \\tau_1 &=& -2\\big(H_{x1}^2 + H_{x2}^2 + H_{x3}^2 + H_{y1}^2 + H_{y2}^2 + H_{y3}^2 + H_{z1}^2 + H_{z2}^2 + H_{z3}^2\\big) \\\\ && \\\\
    \\tau_2 &=& 8\\big(H_{x3}H_{y2}H_{z1} - H_{x2}H_{y3}H_{z1} - H_{x3}H_{y1}H_{z2} + H_{x1}H_{y3}H_{z2} + H_{x2}H_{y1}H_{z3} - H_{x1}H_{y2}H_{z3}\\big) \\\\ && \\\\
    \\tau_3 &=& \\mathrm{det}(\\mathbf{W})
    \\end{array}

Once :math:`\\lambda` is defined, the eigenvector can be obtained using
elementary row operations (Gaussian elimination).

There are two main methods to compute the optimal :math:`\\lambda`:

**1. Iterative Newton-Raphson method**

This 4th-order characteristic polynomial :math:`f(\\lambda)` can be solved with
the `Newton-Raphson's method <https://en.wikipedia.org/wiki/Newton%27s_method>`_
and the aid of its derivative, which is found to be:

.. math::
    f'(\\lambda) = 4\\lambda^3 + 2\\tau_1\\lambda + \\tau_2

The initial value for the root finding process can be set to 1, because
:math:`\\lambda` is very close to it. So, every iteration at :math:`n` updates
:math:`\\lambda` as:

.. math::
    \\lambda_{n+1} \\gets \\lambda_n - \\frac{f\\big(\\lambda_n\\big)}{f'\\big(\\lambda_n\\big)}
    = \\lambda_n - \\frac{\\lambda_n^4 + \\tau_1\\lambda_n^2 + \\tau_2\\lambda_n + \\tau_3}{4\\lambda_n^3 + 2\\tau_1\\lambda_n + \\tau_2}

The value of :math:`\\lambda` is commonly found after a couple iterations, but
the accuracy is not linear with the iteration steps and will not always achieve
good results.

**2. Symbolic method**

A more precise solution involves a symbolic approach, where four solutions to
the characteristic polynomial are obtained as follows:

.. math::
    \\begin{array}{rcl}
    \\lambda_1 &=& \\alpha \\Big(T_2 - \\sqrt{k_1 - k_2}\\Big) \\\\ && \\\\
    \\lambda_2 &=& \\alpha \\Big(T_2 + \\sqrt{k_1 - k_2}\\Big) \\\\ && \\\\
    \\lambda_3 &=& -\\alpha \\Big(T_2 + \\sqrt{k_1 + k_2}\\Big) \\\\ && \\\\
    \\lambda_4 &=& -\\alpha \\Big(T_2 - \\sqrt{k_1 + k_2}\\Big) \\\\
    \\end{array}

with the helper variables:

.. math::
    \\begin{array}{rcl}
    \\alpha &=& \\frac{1}{2\\sqrt{6}} \\\\ && \\\\
    k_1 &=& -T_2^2-12\\tau_1 \\\\ && \\\\
    k_2 &=& \\frac{12\\sqrt{6}\\tau_2}{T_2} \\\\ && \\\\
    T_0 &=& 2\\tau_1^3 + 27\\tau_2^2 - 72\\tau_1\\tau_3 \\\\ && \\\\
    T_1 &=& \\Big(T_0 + \\sqrt{-4(t_1^2+12\\tau_3)^3 + T_0^2}\\Big)^{\\frac{1}{3}} \\\\ && \\\\
    T_2 &=& \\sqrt{-4\\tau_1 + \\frac{2^{\\frac{4}{3}}(\\tau_1^2+12\\tau_3)}{T_1} + 2^{\\frac{2}{3}}T_1}
    \\end{array}

Then chose the :math:`\\lambda`, which is closest to 1. This way solving for
:math:`\\lambda` is truly shortened.

Optimal Quaternion
------------------

Having :math:`\\mathbf{N}=\\mathbf{W}-\\lambda\\mathbf{I}_{4\\times 4}`, the
matrix can be transformed via row operations to:

.. math::
    \\mathbf{N} \\to \\mathbf{N}' = \\begin{bmatrix}
    1 & 0 & 0 & \\chi \\\\ 0 & 1 & 0 & \\rho \\\\ 0 & 0 & 1 & \\upsilon \\\\ 0 & 0 & 0 & \\zeta
    \\end{bmatrix}

where :math:`\\zeta` is usually a very small number. To ensure that
:math:`(\\mathbf{W}-\\lambda\\mathbf{I}_{4\\times 4})=\\mathbf{q}=\\mathbf{0}`
has non-zero and unique solution, :math:`\\zeta` is chosen to be 0. Hence:

.. math::
    \\mathbf{N}' = \\begin{bmatrix}
    1 & 0 & 0 & \\chi \\\\ 0 & 1 & 0 & \\rho \\\\ 0 & 0 & 1 & \\upsilon \\\\ 0 & 0 & 0 & 0
    \\end{bmatrix}

Letting :math:`q_w=-1`, the solution to the optimal quaternion is obtained with:

.. math::
    \\mathbf{q} = \\begin{pmatrix}q_w\\\\q_x\\\\q_y\\\\q_z\\end{pmatrix} =
    \\begin{pmatrix}-1\\\\\\chi\\\\\\rho\\\\\\upsilon\\end{pmatrix}

Finally, the quaternion is normalized to be able to be used as a versor:

.. math::
    \\mathbf{q} = \\frac{1}{\\|\\mathbf{q}\\|} \\mathbf{q}

The decisive element of QUEST is its matrix :math:`\\mathbf{K}`, whereas for
FLAE :math:`\\mathbf{W}` plays the same essential role. Both algorithms spend
most of its computation obtaining said matrices.

FLAE has the same accuracy as other similar estimators (QUEST, SVD, etc.), but
with the advantage of being up to 47% faster than the fastest among them.

Another advantage is the symbolic formulation of the characteristic polynomial,
which does not contain any adjoint matrices, leading to a simpler (therefore
faster) calculation of the eigenvalues.

FLAE advocates for the symbolic method to calculate the eigenvalue. However,
the Newton iteration can be also used to achieve a similar performance to that
of QUEST.

References
----------
.. [Wu] Jin Wu, Zebo Zhou, Bin Gao, Rui Li, Yuhua Cheng, et al. Fast Linear
    Quaternion Attitude Estimator Using Vector Observations. IEEE Transactions
    on Automation Science and Engineering, Institute of Electrical and
    Electronics Engineers, 2018.
    (https://hal.inria.fr/hal-01513263)

"""
from __future__ import annotations
from ahrs.common.mathfuncs import cosd
from ahrs.common.mathfuncs import sind
from ahrs.common.mathfuncs import skew
from ahrs.utils.wmm import WMM
import cmath as cmath
import numpy
import numpy as np
__all__ = ['DEG2RAD', 'DYNAMIC_ELLIPTICITY', 'EARTH_ATMOSPHERE_MASS', 'EARTH_AUTHALIC_RADIUS', 'EARTH_AXIS_RATIO', 'EARTH_C20_DYN', 'EARTH_C20_GEO', 'EARTH_C22_DYN', 'EARTH_EQUATOR_RADIUS', 'EARTH_EQUIVOLUMETRIC_RADIUS', 'EARTH_FIRST_ECCENTRICITY', 'EARTH_FIRST_ECCENTRICITY_2', 'EARTH_FLATTENING', 'EARTH_FLATTENING_INV', 'EARTH_GM', 'EARTH_GM_1', 'EARTH_GM_2', 'EARTH_GM_GPSNAV', 'EARTH_J2', 'EARTH_LINEAR_ECCENTRICITY', 'EARTH_MASS', 'EARTH_MEAN_AXIAL_RADIUS', 'EARTH_MEAN_RADIUS', 'EARTH_POLAR_CURVATURE_RADIUS', 'EARTH_POLAR_RADIUS', 'EARTH_ROTATION', 'EARTH_SECOND_ECCENTRICITY', 'EARTH_SECOND_ECCENTRICITY_2', 'EARTH_SIDEREAL_DAY', 'EQUATORIAL_NORMAL_GRAVITY', 'FLAE', 'JUPITER_EQUATOR_RADIUS', 'JUPITER_GM', 'JUPITER_J2', 'JUPITER_MASS', 'JUPITER_POLAR_RADIUS', 'JUPITER_ROTATION', 'LIGHT_SPEED', 'MAG', 'MARS_EQUATOR_RADIUS', 'MARS_GM', 'MARS_J2', 'MARS_MASS', 'MARS_POLAR_RADIUS', 'MARS_ROTATION', 'MEAN_NORMAL_GRAVITY', 'MERCURY_EQUATOR_RADIUS', 'MERCURY_GM', 'MERCURY_J2', 'MERCURY_MASS', 'MERCURY_POLAR_RADIUS', 'MERCURY_ROTATION', 'MOON_EQUATOR_RADIUS', 'MOON_GM', 'MOON_J2', 'MOON_MASS', 'MOON_POLAR_RADIUS', 'MOON_ROTATION', 'MUNICH_HEIGHT', 'MUNICH_LATITUDE', 'MUNICH_LONGITUDE', 'M_PI', 'NEPTUNE_EQUATOR_RADIUS', 'NEPTUNE_GM', 'NEPTUNE_J2', 'NEPTUNE_MASS', 'NEPTUNE_POLAR_RADIUS', 'NEPTUNE_ROTATION', 'NORMAL_GRAVITY_FORMULA', 'NORMAL_GRAVITY_POTENTIAL', 'PLUTO_EQUATOR_RADIUS', 'PLUTO_GM', 'PLUTO_MASS', 'PLUTO_POLAR_RADIUS', 'PLUTO_ROTATION', 'POLAR_NORMAL_GRAVITY', 'RAD2DEG', 'SATURN_EQUATOR_RADIUS', 'SATURN_GM', 'SATURN_J2', 'SATURN_MASS', 'SATURN_POLAR_RADIUS', 'SATURN_ROTATION', 'SOMIGLIANA_GRAVITY', 'UNIVERSAL_GRAVITATION_CODATA2014', 'UNIVERSAL_GRAVITATION_CODATA2018', 'UNIVERSAL_GRAVITATION_WGS84', 'URANUS_EQUATOR_RADIUS', 'URANUS_GM', 'URANUS_J2', 'URANUS_MASS', 'URANUS_POLAR_RADIUS', 'URANUS_ROTATION', 'VENUS_EQUATOR_RADIUS', 'VENUS_GM', 'VENUS_J2', 'VENUS_MASS', 'VENUS_POLAR_RADIUS', 'VENUS_ROTATION', 'WMM', 'cmath', 'cosd', 'np', 'sind', 'skew']
class FLAE:
    """
    Fast Linear Attitude Estimator
    
        Parameters
        ----------
        acc : numpy.ndarray, default: None
            N-by-3 array with measurements of acceleration in in m/s^2
        mag : numpy.ndarray, default: None
            N-by-3 array with measurements of magnetic field in mT
        method : str, default: 'symbolic'
            Method used to estimate the attitude. Options are: 'symbolic', 'eig'
            and 'newton'.
        weights : np.ndarray, default: [0.5, 0.5]
            Weights used for each sensor. They must add up to 1.
        magnetic_dip : float
            Geomagnetic Inclination angle at local position, in degrees. Defaults
            to magnetic dip of Munich, Germany.
    
        Raises
        ------
        ValueError
            When estimation method is invalid.
    
        Examples
        --------
        >>> orientation = FLAE()
        >>> accelerometer = np.array([-0.2853546, 9.657394, 2.0018768])
        >>> magnetometer = np.array([12.32605, -28.825378, -26.586914])
        >>> orientation.estimate(acc=accelerometer, mag=magnetometer)
        array([-0.45447247, -0.69524546,  0.55014011, -0.08622285])
    
        You can set a different estimation method passing its name to parameter
        ``method``.
    
        >>> orientation.estimate(acc=accelerometer, mag=magnetometer, method='newton')
        array([ 0.42455176,  0.68971918, -0.58315259, -0.06305803])
    
        Or estimate all quaternions at once by giving the data to the constructor.
        All estimated quaternions are stored in attribute ``Q``.
    
        >>> orientation = FLAE(acc=acc_data, mag=mag_Data, method='eig')
        >>> orientation.Q.shape
        (1000, 4)
    
        
    """
    def P1Hx(self, Hx: numpy.ndarray) -> numpy.ndarray:
        ...
    def P2Hy(self, Hy: numpy.ndarray) -> numpy.ndarray:
        ...
    def P3Hz(self, Hz: numpy.ndarray) -> numpy.ndarray:
        ...
    def __init__(self, acc: numpy.ndarray = None, mag: numpy.ndarray = None, method: str = 'symbolic', **kw):
        ...
    def _compute_all(self) -> numpy.ndarray:
        """
        
                Estimate the quaternions given all data in class Data.
        
                Class Data must have, at least, `acc` and `mag` attributes.
        
                Returns
                -------
                Q : numpy.ndarray
                    M-by-4 Array with all estimated quaternions, where M is the number
                    of samples.
        
                
        """
    def _row_reduction(self, A: numpy.ndarray) -> numpy.ndarray:
        """
        Gaussian elimination
                
        """
    def estimate(self, acc: numpy.ndarray, mag: numpy.ndarray, method: str = 'symbolic') -> numpy.ndarray:
        """
        
                Estimate a quaternion with the given measurements and weights.
        
                Parameters
                ----------
                acc : numpy.ndarray
                    Sample of tri-axial Accelerometer.
                mag : numpy.ndarray
                    Sample of tri-axial Magnetometer.
                method : str, default: 'symbolic'
                    Method used to estimate the attitude. Options are: 'symbolic', 'eig'
                    and 'newton'.
        
                Returns
                -------
                q : numpy.ndarray
                    Estimated orienation as quaternion.
        
                Examples
                --------
                >>> accelerometer = np.array([-0.2853546, 9.657394, 2.0018768])
                >>> magnetometer = np.array([12.32605, -28.825378, -26.586914])
                >>> orientation = FLAE()
                >>> orientation.estimate(acc=accelerometer, mag=magnetometer)
                array([-0.45447247, -0.69524546,  0.55014011, -0.08622285])
                >>> orientation.estimate(acc=accelerometer, mag=magnetometer, method='eig')
                array([ 0.42455176,  0.68971918, -0.58315259, -0.06305803])
                >>> orientation.estimate(acc=accelerometer, mag=magnetometer, method='newton')
                array([ 0.42455176,  0.68971918, -0.58315259, -0.06305803])
        
                
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
