"""

TRIAD
=====

The Tri-Axial Attitude Determination (`TRIAD <https://en.wikipedia.org/wiki/Triad_method>`_)
was first described in [Black]_ to algebraically estimate an attitude
represented as a Direction Cosine Matrix from two orthogonal vector
observations.

Given two non-parallel reference *unit vectors* :math:`\\mathbf{v}_1` and
:math:`\\mathbf{v}_2` and their corresponding *unit vectors* :math:`\\mathbf{w}_1`
and :math:`\\mathbf{w}_2`, it is required to find an orthogonal matrix
:math:`\\mathbf{A}` satisfying:

.. math::
    \\mathbf{Av}_1

Two vectors :math:`\\mathbf{v}_1` and :math:`\\mathbf{v}_2` define an
orthogonal coordinate system with the **normalized** basis vectors
:math:`\\mathbf{q}`, :math:`\\mathbf{r}`, and :math:`\\mathbf{s}` as the
following triad:

.. math::
    \\begin{array}{rcl}
    \\mathbf{q}_r &=& \\mathbf{v}_1 \\\\
    \\mathbf{r}_r &=& \\frac{\\mathbf{v}_1\\times\\mathbf{v}_2}{|\\mathbf{v}_1\\times\\mathbf{v}_2|} \\\\
    \\mathbf{s}_r &=& \\mathbf{q}_r\\times\\mathbf{r}_r
    \\end{array}

The TRIAD method, initially developed to estimate the attitude of spacecrafts
[Shuster2007]_, uses the position of the sun (using a `star tracker
<https://en.wikipedia.org/wiki/Star_tracker>`_) and the magnetic field of Earth
as references [Hall]_ [Makley]_. These are represented as vectors to build an
appropriate *reference* frame :math:`\\mathbf{M}_r`:

.. math::
    \\mathbf{M}_r = \\begin{bmatrix} \\mathbf{q}_r & \\mathbf{r}_r & \\mathbf{s}_r \\end{bmatrix}

Similarly, at any given time, two measured vectors in the spacecraft's **body
frame** :math:`\\mathbf{w}_1` and :math:`\\mathbf{w}_2` determine the
:math:`3\\times 3` body matrix :math:`\\mathbf{M}_b`:

.. math::
    \\mathbf{M}_b = \\begin{bmatrix} \\mathbf{q}_b & \\mathbf{r}_b & \\mathbf{s}_b \\end{bmatrix}

where, like the first triad, the second triad is built as:

.. math::
    \\begin{array}{rcl}
    \\mathbf{q}_b &=& \\mathbf{w}_1 \\\\
    \\mathbf{r}_b &=& \\frac{\\mathbf{w}_1\\times\\mathbf{w}_2}{|\\mathbf{w}_1\\times\\mathbf{w}_2|} \\\\
    \\mathbf{s}_b &=& \\mathbf{q}_b\\times\\mathbf{r}_b
    \\end{array}

The attitude matrix :math:`\\mathbf{A}\\in\\mathbb{R}^{3\\times 3}` defines the
coordinate transformation,

.. math::
    \\mathbf{AM}_r = \\mathbf{M}_b

Solving for :math:`\\mathbf{A}` we obtain:

.. math::
    \\mathbf{A} = \\mathbf{M}_b\\mathbf{M}_r^{-1}

But we also know that :math:`\\mathbf{M}_r` is orthogonal. So, the solution is
simply:

.. math::
    \\mathbf{A} = \\mathbf{M}_b\\mathbf{M}_r^T

Inverse trigonometric functions are not required, a unique attitude is obtained,
and computational requirements are minimal.

It is only required that :math:`\\mathbf{M}_r` has an inverse, but that is
already ensured, since :math:`\\mathbf{q}_r`, :math:`\\mathbf{r}_r`,
and :math:`\\mathbf{s}_r` are linearly independent [Lerner1]_.

Strapdown INS
-------------

For estimations using a Strapdown INS on Earth, we identify two main reference
vectors: gravity :math:`\\mathbf{g}=\\begin{bmatrix}g_x & g_y & g_z\\end{bmatrix}`
and magnetic field :math:`\\mathbf{h}=\\begin{bmatrix}h_x & h_y & h_z\\end{bmatrix}`.

A common convention sets the *gravity vector* equal to :math:`0` along the X-
and Y-axis, and equal to :math:`\\sim 9.81` along the Z-axis. This assumes the
direction of the gravity is parallel to the vertical axis. Because TRIAD uses
normalized vectors, the Z-axis will turn out to be equal to :math:`1`:

.. math::
    \\mathbf{g} = \\begin{bmatrix}0 \\\\ 0 \\\\ 1 \\end{bmatrix}

The *magnetic field* is defined from the geographical position of the
measurement. Using the `World Magnetic Model <https://www.ngdc.noaa.gov/geomag/WMM/>`_,
we can estimate the magnetic field elements of our location on a given date.

The class :class:`ahrs.utils.WMM` can help us to retrieve it. Let's say we want
to know the geomagnetic field elements of Munich, Germany [#]_ on the 3rd of
October, 2020.

The city's location is 48.137154° N and 11.576124° E at 519 m above sea level.
We obtain its magnetic elements as:

.. code-block:: python

    >>> import datetime
    >>> from ahrs.utils import WMM
    >>> wmm = WMM(latitude=48.137154, longitude=11.576124, height=0.519, date=datetime.date(2020, 10, 3))
    >>> wmm.magnetic_elements
    {'X': 21009.66924050522, 'Y': 1333.4601319284525, 'Z': 43731.849938722924, 'H': 21051.943319296533, 'F': 48535.13177670226, 'I': 64.2944417667441, 'D': 3.631627635223863, 'GV': 3.631627635223863}

For further explanation of class :class:`WMM`, please check its `page <../WMM.html>`_.
Of our interest are only the values of ``X``, ``Y`` and ``Z`` representing the
magnetic field intensity, in nT, along the X-, Y- and Z-axis, respectively.

.. math::
    \\mathbf{h} = \\begin{bmatrix} 21009.66924 \\\\ 1333.46013 \\\\ 43731.84994 \\end{bmatrix}

But, again, TRIAD works with normalized vectors, so the reference magnetic
vector becomes:

.. math::
    \\mathbf{h} = \\begin{bmatrix} 0.43288 \\\\ 0.02747 \\\\ 0.90103 \\end{bmatrix}

.. code-block:: python

    >>> import numpy as np
    >>> h = np.array([wmm.magnetic_elements[x] for x in list('XYZ')])
    >>> h /= np.linalg.norm(h)      # Reference geomagnetic field (h)
    >>> h
    array([0.4328755 , 0.02747412, 0.90103495])

Both normalized vectors :math:`\\mathbf{g}` and :math:`\\mathbf{h}` build the
*reference triad* :math:`\\mathbf{M}_r`

Then, we have to measure their equivalent vectors, for which we use the
accelerometer to obtain :math:`\\mathbf{a} = \\begin{bmatrix}a_x & a_y & a_z \\end{bmatrix}`,
and the magnetometer for :math:`\\mathbf{m} = \\begin{bmatrix}m_x & m_y & m_z \\end{bmatrix}`.

Both measurement vectors are also normalized, meaning :math:`\\|\\mathbf{a}\\|=\\|\\mathbf{m}\\|=1`,
so that they can build the *body's measurement triad*  :math:`\\mathbf{M}_b`.

To get the Direction Cosine Matrix we simply call the method ``estimate`` with
the normalized measurement vectors:

.. code-block:: python

    >>> triad = ahrs.filters.TRIAD()
    >>> triad.v1 = np.array([0.0, 0.0, 1.0])                    # Reference gravity vector (g)
    >>> triad.v2 = h                                            # Reference geomagnetic field (h)
    >>> a = np.array([-2.499e-04, 4.739e-02, 0.9988763])        # Measured acceleration (normalized)
    >>> a /= np.linalg.norm(a)
    >>> m = np.array([-0.36663061, 0.17598138, -0.91357132])    # Measured magnetic field (normalized)
    >>> m /= np.linalg.norm(m)
    >>> triad.estimate(w1=a, w2=m)
    array([[-8.48320410e-01, -5.29483162e-01, -2.49900033e-04],
           [ 5.28878238e-01, -8.47373587e-01,  4.73900062e-02],
           [-2.53039690e-02,  4.00697428e-02,  9.98876431e-01]])

Optionally, it can return the estimation as a quaternion representation setting
``representation`` to ``'quaternion'``.

.. code-block:: python

    >>> triad.estimate(w1=a, w2=m, representation='quaternion')
    array([ 0.27531002, -0.00664729,  0.02275078,  0.96106327])

Giving the observation vector to the constructor, the attitude estimation
happens automatically, and is stored in the attribute ``A``.

.. code-block:: python

    >>> triad = ahrs.filters.TRIAD(w1=np.array([-2.499e-04, 4.739e-02, 0.9988763]), w2=np.array([-0.36663061, 0.17598138, -0.91357132]), v2=h)
    >>> triad.A
    array([[-8.48320410e-01, -5.29483162e-01, -2.49900033e-04],
           [ 5.28878238e-01, -8.47373587e-01,  4.73900062e-02],
           [-2.53039690e-02,  4.00697428e-02,  9.98876431e-01]])
    >>> triad = ahrs.filters.TRIAD(w1=np.array([-2.499e-04, 4.739e-02, 0.9988763]), w2=np.array([-0.36663061, 0.17598138, -0.91357132]), v2=h, representation='quaternion')
    >>> triad.A
    array([ 0.27531002, -0.00664729,  0.02275078,  0.96106327])

If the input data contains many observations, all will be estimated at once.

.. code-block:: python

    >>> a = np.array([[-0.000249905733, 0.0473926177, 0.998876307],
    ... [-0.00480145530, 0.0572267567, 0.998349660],
    ... [-0.00986626329, 0.0746539896, 0.997160688]])
    >>> m = np.array([[-0.36663061, 0.17598138, -0.91357132],
    ... [-0.37726367, 0.18069746, -0.90830642],
    ... [-0.3874741, 0.18536454, -0.9030525]])
    >>> triad = ahrs.filters.TRIAD(w1=a, w2=m, v2=h)
    >>> triad.A
    array([[[-8.48317898e-01, -5.29487187e-01, -2.49905733e-04],
            [ 5.28882192e-01, -8.47370974e-01,  4.73926177e-02],
            [-2.53055467e-02,  4.00718352e-02,  9.98876307e-01]],

           [[-8.43678607e-01, -5.36827117e-01, -4.80145530e-03],
            [ 5.35721702e-01, -8.42453178e-01,  5.72267567e-02],
            [-3.47658761e-02,  4.57087466e-02,  9.98349660e-01]],

           [[-8.32771974e-01, -5.53528225e-01, -9.86626329e-03],
            [ 5.51396878e-01, -8.30896061e-01,  7.46539896e-02],
            [-4.95209297e-02,  5.67295235e-02,  9.97160688e-01]]])
    >>> triad = ahrs.filters.TRIAD(w1=a, w2=m, representation='quaternion')
    >>> triad.A
    array([[ 0.27531229, -0.00664771,  0.02275202,  0.96106259],
           [ 0.2793823 , -0.01030667,  0.0268131 ,  0.95975016],
           [ 0.28874411, -0.01551933,  0.03433374,  0.95666461]])

The first disadvantage is that TRIAD can only use two observations per
estimation. If there are more observations, we must discard part of them
(losing accuracy), or mix them in such a way that we obtain only two
representative observations.

The second disadvantage is its loss of accuracy in a heavily dynamic state of
the measuring device. TRIAD assumes a quasi-static state of the body frame and,
therefore, its use is limited to motionless objects, preferably.

Footnotes
---------
.. [#] This package's author resides in Munich, and examples of geographical
    locations will take it as a reference.

References
----------
.. [Black] Black, Harold. "A Passive System for Determining the Attitude of a
    Satellite," AIAA Journal, Vol. 2, July 1964, pp. 1350–1351.
.. [Lerner1] Lerner, G. M. "Three-Axis Attitude Determination" in Spacecraft
    Attitude Determination and Control, edited by J.R. Wertz. 1978. p. 420-426.
.. [Hall] Chris Hall. Spacecraft Attitude Dynamics and Control. Chapter 4:
    Attitude Determination. 2003.
    (http://www.dept.aoe.vt.edu/~cdhall/courses/aoe4140/attde.pdf)
.. [Makley] F.L. Makley et al. Fundamentals of Spacecraft Attitude
    Determination and Control. 2014. Pages 184-186.
.. [Shuster2007] Shuster, Malcolm D. The optimization of TRIAD. The Journal of
    the Astronautical Sciences, Vol. 55, No 2, April – June 2007, pp. 245–257.
    (http://www.malcolmdshuster.com/Pub_2007f_J_OptTRIAD_AAS.pdf)

"""
from __future__ import annotations
from ahrs.common.mathfuncs import cosd
from ahrs.common.mathfuncs import sind
from ahrs.common.mathfuncs import skew
from ahrs.common.orientation import chiaverini
from ahrs.utils.wmm import WMM
import cmath as cmath
import numpy
import numpy as np
__all__ = ['DEG2RAD', 'DYNAMIC_ELLIPTICITY', 'EARTH_ATMOSPHERE_MASS', 'EARTH_AUTHALIC_RADIUS', 'EARTH_AXIS_RATIO', 'EARTH_C20_DYN', 'EARTH_C20_GEO', 'EARTH_C22_DYN', 'EARTH_EQUATOR_RADIUS', 'EARTH_EQUIVOLUMETRIC_RADIUS', 'EARTH_FIRST_ECCENTRICITY', 'EARTH_FIRST_ECCENTRICITY_2', 'EARTH_FLATTENING', 'EARTH_FLATTENING_INV', 'EARTH_GM', 'EARTH_GM_1', 'EARTH_GM_2', 'EARTH_GM_GPSNAV', 'EARTH_J2', 'EARTH_LINEAR_ECCENTRICITY', 'EARTH_MASS', 'EARTH_MEAN_AXIAL_RADIUS', 'EARTH_MEAN_RADIUS', 'EARTH_POLAR_CURVATURE_RADIUS', 'EARTH_POLAR_RADIUS', 'EARTH_ROTATION', 'EARTH_SECOND_ECCENTRICITY', 'EARTH_SECOND_ECCENTRICITY_2', 'EARTH_SIDEREAL_DAY', 'EQUATORIAL_NORMAL_GRAVITY', 'JUPITER_EQUATOR_RADIUS', 'JUPITER_GM', 'JUPITER_J2', 'JUPITER_MASS', 'JUPITER_POLAR_RADIUS', 'JUPITER_ROTATION', 'LIGHT_SPEED', 'MAG', 'MARS_EQUATOR_RADIUS', 'MARS_GM', 'MARS_J2', 'MARS_MASS', 'MARS_POLAR_RADIUS', 'MARS_ROTATION', 'MEAN_NORMAL_GRAVITY', 'MERCURY_EQUATOR_RADIUS', 'MERCURY_GM', 'MERCURY_J2', 'MERCURY_MASS', 'MERCURY_POLAR_RADIUS', 'MERCURY_ROTATION', 'MOON_EQUATOR_RADIUS', 'MOON_GM', 'MOON_J2', 'MOON_MASS', 'MOON_POLAR_RADIUS', 'MOON_ROTATION', 'MUNICH_HEIGHT', 'MUNICH_LATITUDE', 'MUNICH_LONGITUDE', 'M_PI', 'NEPTUNE_EQUATOR_RADIUS', 'NEPTUNE_GM', 'NEPTUNE_J2', 'NEPTUNE_MASS', 'NEPTUNE_POLAR_RADIUS', 'NEPTUNE_ROTATION', 'NORMAL_GRAVITY_FORMULA', 'NORMAL_GRAVITY_POTENTIAL', 'PLUTO_EQUATOR_RADIUS', 'PLUTO_GM', 'PLUTO_MASS', 'PLUTO_POLAR_RADIUS', 'PLUTO_ROTATION', 'POLAR_NORMAL_GRAVITY', 'RAD2DEG', 'SATURN_EQUATOR_RADIUS', 'SATURN_GM', 'SATURN_J2', 'SATURN_MASS', 'SATURN_POLAR_RADIUS', 'SATURN_ROTATION', 'SOMIGLIANA_GRAVITY', 'TRIAD', 'UNIVERSAL_GRAVITATION_CODATA2014', 'UNIVERSAL_GRAVITATION_CODATA2018', 'UNIVERSAL_GRAVITATION_WGS84', 'URANUS_EQUATOR_RADIUS', 'URANUS_GM', 'URANUS_J2', 'URANUS_MASS', 'URANUS_POLAR_RADIUS', 'URANUS_ROTATION', 'VENUS_EQUATOR_RADIUS', 'VENUS_GM', 'VENUS_J2', 'VENUS_MASS', 'VENUS_POLAR_RADIUS', 'VENUS_ROTATION', 'WMM', 'chiaverini', 'cmath', 'cosd', 'np', 'sind', 'skew']
class TRIAD:
    """
    
        Tri-Axial Attitude Determination
    
        TRIAD estimates the attitude as a Direction Cosine Matrix. To return it as
        a quaternion, set the parameter ``as_quaternion`` to ``True``.
    
        Parameters
        ----------
        w1 : numpy.ndarray
            First tri-axial observation vector in body frame. Usually a normalized
            acceleration vector :math:`\\mathbf{a} = \\begin{bmatrix}a_x & a_y & a_z \\end{bmatrix}`
        w2 : numpy.ndarray
            Second tri-axial observation vector in body frame. Usually a normalized
            magnetic field vector :math:`\\mathbf{m} = \\begin{bmatrix}m_x & m_y & m_z \\end{bmatrix}`
        v1 : numpy.ndarray, optional.
            First tri-axial reference vector. Defaults to normalized gravity vector
            :math:`\\mathbf{g} = \\begin{bmatrix}0 & 0 & 1 \\end{bmatrix}`
        v2 : numpy.ndarray, optional.
            Second tri-axial reference vector. Defaults to normalized geomagnetic
            field :math:`\\mathbf{h} = \\begin{bmatrix}h_x & h_y & h_z \\end{bmatrix}`
            in Munich, Germany.
        representation : str, default: ``'rotmat'``
            Attitude representation. Options are ``rotmat'`` or ``'quaternion'``.
        frame : str, default: 'NED'
            Local tangent plane coordinate frame. Valid options are right-handed
            ``'NED'`` for North-East-Down and ``'ENU'`` for East-North-Up.
    
        Attributes
        ----------
        w1 : numpy.ndarray
            First tri-axial observation vector in body frame.
        w2 : numpy.ndarray
            Second tri-axial observation vector in body frame.
        v1 : numpy.ndarray, optional.
            First tri-axial reference vector.
        v2 : numpy.ndarray, optional.
            Second tri-axial reference vector.
        A : numpy.ndarray
            Estimated attitude.
    
        Examples
        --------
        >>> from ahrs.filters import TRIAD
        >>> triad = TRIAD()
        >>> triad.v1 = np.array([0.0, 0.0, 1.0])                    # Reference gravity vector (g)
        >>> triad.v2 = np.array([21.0097, 1.3335, 43.732])          # Reference geomagnetic field (h)
        >>> a = np.array([-2.499e-04, 4.739e-02, 0.9988763])        # Measured acceleration (normalized)
        >>> a /= np.linalg.norm(a)
        >>> m = np.array([-0.36663061, 0.17598138, -0.91357132])    # Measured magnetic field (normalized)
        >>> m /= np.linalg.norm(m)
        >>> triad.estimate(w1=a, w2=m)
        array([[-8.48320410e-01, -5.29483162e-01, -2.49900033e-04],
               [ 5.28878238e-01, -8.47373587e-01,  4.73900062e-02],
               [-2.53039690e-02,  4.00697428e-02,  9.98876431e-01]])
    
        It also works by passing each array to its corresponding parameter. They
        will be normalized too.
    
        >>> triad = TRIAD(w1=a, w2=m, v1=[0.0, 0.0, 1.0], v2=[-0.36663061, 0.17598138, -0.91357132])
    
        
    """
    def __init__(self, w1: numpy.ndarray = None, w2: numpy.ndarray = None, v1: numpy.ndarray = None, v2: numpy.ndarray = None, representation: str = 'rotmat', frame: str = 'NED'):
        ...
    def _compute_all(self, representation) -> numpy.ndarray:
        """
        
                Estimate the attitude given all data.
        
                Attributes ``w1`` and ``w2`` must contain data.
        
                Parameters
                ----------
                representation : str
                    Attitude representation. Options are ``'rotmat'`` or ``'quaternion'``.
        
                Returns
                -------
                A : numpy.ndarray
                    M-by-3-by-3 with all estimated attitudes as direction cosine
                    matrices, where M is the number of samples. It is an N-by-4 array
                    if ``representation`` is set to ``'quaternion'``.
        
                
        """
    def _set_first_triad_reference(self, value, frame):
        ...
    def _set_second_triad_reference(self, value, frame):
        ...
    def estimate(self, w1: numpy.ndarray, w2: numpy.ndarray, representation: str = 'rotmat') -> numpy.ndarray:
        """
        
                Attitude Estimation.
        
                The equation numbers in the code refer to [Lerner1]_.
        
                Parameters
                ----------
                w1 : numpy.ndarray
                    Sample of first tri-axial sensor.
                w2 : numpy.ndarray
                    Sample of second tri-axial sensor.
                representation : str, default: ``'rotmat'``
                    Attitude representation. Options are ``rotmat'`` or ``'quaternion'``.
        
                Returns
                -------
                A : numpy.ndarray
                    Estimated attitude as 3-by-3 Direction Cosine Matrix. If
                    ``representation`` is set to ``'quaternion'``, it is returned as a
                    quaternion.
        
                Examples
                --------
                >>> triad = ahrs.filters.TRIAD()
                >>> triad.v1 = [0.0, 0.0, 1.0]                              # Normalized reference gravity vector (g)
                >>> triad.v2 = [0.4328755, 0.02747412, 0.90103495]          # Normalized reference geomagnetic field (h)
                >>> a = [4.098297, 8.663757, 2.1355896]                     # Measured acceleration
                >>> m = [-28715.50512, -25927.43566, 4756.83931]            # Measured magnetic field
                >>> triad.estimate(w1=a, w2=m)                              # Estimated attitude as DCM
                array([[-7.84261e-01  ,  4.5905718e-01,  4.1737417e-01],
                       [ 2.2883429e-01, -4.1126404e-01,  8.8232463e-01],
                       [ 5.7668844e-01,  7.8748232e-01,  2.1749032e-01]])
                >>> triad.estimate(w1=a, w2=m, representation='quaternion')          # Estimated attitude as quaternion
                array([ 0.07410345, -0.3199659, -0.53747247, -0.77669417])
        
                
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
