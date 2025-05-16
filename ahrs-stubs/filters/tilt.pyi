"""

Attitude from gravity (Tilt)
============================

Attitude estimation via gravity acceleration measurements.

The simplest way to estimate the attitude from the gravitational acceleration
is using 3D `geometric quadrants <https://en.wikipedia.org/wiki/Quadrant_(plane_geometry)>`_.

Although some methods use ``arctan`` to estimate the angles [ST-AN4509]_ [AD-AN1057]_,
it is preferred to use ``arctan2`` to explore all quadrants searching the tilt
angles.

First, we normalize the gravity vector, so that it has magnitude equal to 1.
Then, we get the angles to the main axes with `arctan2 <https://en.wikipedia.org/wiki/Atan2>`_
[FS-AN3461]_ [Trimpe]_:

.. math::
    \\begin{array}{ll}
    \\theta &= \\mathrm{arctan2}(a_y, a_z) \\\\
    \\phi &= \\mathrm{arctan2}\\big(-a_x, \\sqrt{a_y^2+a_z^2}\\big)
    \\end{array}

where :math:`\\theta` is the **roll** angle, :math:`\\phi` is the **pitch**
angle, and :math:`\\mathbf{a}=\\begin{bmatrix}a_x & a_y & a_z\\end{bmatrix}^T`
is the normalized vector of measured accelerations, which means
:math:`\\|\\mathbf{a}\\|=1`.

The attitude in terms of these two angles is called the **tilt**.

**Heading angle**

The heading angle, a.k.a. **yaw**, cannot be obtained from the measured
acceleration, and a different reference shall be used to obtain it. The most
common is the use of the geomagnetic information, in other words, `Earth's
magnetic field <https://en.wikipedia.org/wiki/Earth%27s_magnetic_field>`_.

With the pitch and roll angles estimated from the accelerometer, we can rotate
a magnetometer reading :math:`\\mathbf{m}=\\begin{bmatrix}m_x & m_y & m_z\\end{bmatrix}^T`,
and estimate the yaw angle :math:`\\psi` to update the orientation.

The vector :math:`\\mathbf{b}=\\begin{bmatrix}b_x & b_y & b_z\\end{bmatrix}^T`
represents the magnetometer readings after *rotating them back* to the plane,
where :math:`\\theta = \\phi = 0`.

.. math::
    \\begin{array}{cl}
    \\mathbf{b} &=
    R_y(-\\theta)R_x(-\\phi)\\mathbf{m} = R_y(\\theta)^TR_x(\\phi)^T\\mathbf{m} \\\\
    &=
    \\begin{bmatrix}
        \\cos\\theta & \\sin\\theta\\sin\\phi & \\sin\\theta\\cos\\phi \\\\
        0 & \\cos\\phi & -\\sin\\phi \\\\
        -\\sin\\theta & \\cos\\theta\\sin\\phi & \\cos\\theta\\cos\\phi
    \\end{bmatrix}
    \\begin{bmatrix}m_x \\\\ m_y \\\\ m_z\\end{bmatrix} \\\\
    \\begin{bmatrix}b_x \\\\ b_y \\\\ b_z\\end{bmatrix} &=
    \\begin{bmatrix}
        m_x\\cos\\theta + m_y\\sin\\theta\\sin\\phi + m_z\\sin\\theta\\cos\\phi \\\\
        m_y\\cos\\phi - m_z\\sin\\phi \\\\
        -m_x\\sin\\theta + m_y\\cos\\theta\\sin\\phi + m_z\\cos\\theta\\cos\\phi
    \\end{bmatrix}
    \\end{array}

Where :math:`\\mathbf{m}=\\begin{bmatrix}m_x & m_y & m_z\\end{bmatrix}^T` is
the *normalized* vector of the measured magnetic field, which means
:math:`\\|\\mathbf{m}\\|=1`.

The yaw angle :math:`\\psi` is the tilt-compensated heading angle relative to
magnetic North, computed as [FS-AN4248]_:

.. math::
    \\begin{array}{ll}
    \\psi &= \\mathrm{arctan2}(-b_y, b_x) \\\\
    &= \\mathrm{arctan2}\\big(m_z\\sin\\phi - m_y\\cos\\phi, \\; m_x\\cos\\theta + \\sin\\theta(m_y\\sin\\phi + m_z\\cos\\phi)\\big)
    \\end{array}

Finally, we transform the roll-pitch-yaw angles to a quaternion representation:

.. math::
    \\mathbf{q} =
    \\begin{pmatrix}q_w\\\\q_x\\\\q_y\\\\q_z\\end{pmatrix} =
    \\begin{pmatrix}
        \\cos\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big) + \\sin\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) \\\\
        \\sin\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big) - \\cos\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) \\\\
        \\cos\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big) + \\sin\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) \\\\
        \\cos\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) - \\sin\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big)
    \\end{pmatrix}

Setting the property ``as_angles`` to ``True`` will avoid this last conversion
returning the attitude as angles.

References
----------
.. [Trimpe] Sebastian Trimpe and Raffaello D'Andrea. The Balancing cube. A
    dynamic sculpture as test bed for distributed estimation and control. IEEE
    Control Systems Magazine. December 2012.
    (http://trimpe.is.tuebingen.mpg.de/publications/trimpe-CSM12.pdf)
.. [FS-AN3461] Mark Pedley. Tilt Sensing Using a Three-Axis Accelerometer.
    Freescale Semiconductor Application Note. Document Number: AN3461. 2013.
    (https://www.nxp.com/files-static/sensors/doc/app_note/AN3461.pdf)
.. [FS-AN4248] Talat Ozyagcilar. Implementing a Tilt-Compensated eCompass using
    Accelerometer and Magnetometer sensors. Freescale Semoconductor Application
    Note. Document Number: AN4248. 2015.
    (https://www.nxp.com/files-static/sensors/doc/app_note/AN4248.pdf)
.. [AD-AN1057] Christopher J. Fisher. Using an Accelerometer for Inclination
    Sensing. Analog Devices. Application Note. AN-1057.
    (https://www.analog.com/media/en/technical-documentation/application-notes/AN-1057.pdf)
.. [ST-AN4509] Tilt measurement using a low-g 3-axis accelerometer.
    STMicroelectronics. Application note AN4509. 2014.
    (https://www.st.com/resource/en/application_note/dm00119046.pdf)
.. [WikiConversions] Wikipedia: Conversion between quaternions and Euler angles.
    (https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles)

"""
from __future__ import annotations
import cmath as cmath
import numpy
import numpy as np
__all__ = ['DEG2RAD', 'DYNAMIC_ELLIPTICITY', 'EARTH_ATMOSPHERE_MASS', 'EARTH_AUTHALIC_RADIUS', 'EARTH_AXIS_RATIO', 'EARTH_C20_DYN', 'EARTH_C20_GEO', 'EARTH_C22_DYN', 'EARTH_EQUATOR_RADIUS', 'EARTH_EQUIVOLUMETRIC_RADIUS', 'EARTH_FIRST_ECCENTRICITY', 'EARTH_FIRST_ECCENTRICITY_2', 'EARTH_FLATTENING', 'EARTH_FLATTENING_INV', 'EARTH_GM', 'EARTH_GM_1', 'EARTH_GM_2', 'EARTH_GM_GPSNAV', 'EARTH_J2', 'EARTH_LINEAR_ECCENTRICITY', 'EARTH_MASS', 'EARTH_MEAN_AXIAL_RADIUS', 'EARTH_MEAN_RADIUS', 'EARTH_POLAR_CURVATURE_RADIUS', 'EARTH_POLAR_RADIUS', 'EARTH_ROTATION', 'EARTH_SECOND_ECCENTRICITY', 'EARTH_SECOND_ECCENTRICITY_2', 'EARTH_SIDEREAL_DAY', 'EQUATORIAL_NORMAL_GRAVITY', 'JUPITER_EQUATOR_RADIUS', 'JUPITER_GM', 'JUPITER_J2', 'JUPITER_MASS', 'JUPITER_POLAR_RADIUS', 'JUPITER_ROTATION', 'LIGHT_SPEED', 'MARS_EQUATOR_RADIUS', 'MARS_GM', 'MARS_J2', 'MARS_MASS', 'MARS_POLAR_RADIUS', 'MARS_ROTATION', 'MEAN_NORMAL_GRAVITY', 'MERCURY_EQUATOR_RADIUS', 'MERCURY_GM', 'MERCURY_J2', 'MERCURY_MASS', 'MERCURY_POLAR_RADIUS', 'MERCURY_ROTATION', 'MOON_EQUATOR_RADIUS', 'MOON_GM', 'MOON_J2', 'MOON_MASS', 'MOON_POLAR_RADIUS', 'MOON_ROTATION', 'MUNICH_HEIGHT', 'MUNICH_LATITUDE', 'MUNICH_LONGITUDE', 'M_PI', 'NEPTUNE_EQUATOR_RADIUS', 'NEPTUNE_GM', 'NEPTUNE_J2', 'NEPTUNE_MASS', 'NEPTUNE_POLAR_RADIUS', 'NEPTUNE_ROTATION', 'NORMAL_GRAVITY_FORMULA', 'NORMAL_GRAVITY_POTENTIAL', 'PLUTO_EQUATOR_RADIUS', 'PLUTO_GM', 'PLUTO_MASS', 'PLUTO_POLAR_RADIUS', 'PLUTO_ROTATION', 'POLAR_NORMAL_GRAVITY', 'RAD2DEG', 'SATURN_EQUATOR_RADIUS', 'SATURN_GM', 'SATURN_J2', 'SATURN_MASS', 'SATURN_POLAR_RADIUS', 'SATURN_ROTATION', 'SOMIGLIANA_GRAVITY', 'Tilt', 'UNIVERSAL_GRAVITATION_CODATA2014', 'UNIVERSAL_GRAVITATION_CODATA2018', 'UNIVERSAL_GRAVITATION_WGS84', 'URANUS_EQUATOR_RADIUS', 'URANUS_GM', 'URANUS_J2', 'URANUS_MASS', 'URANUS_POLAR_RADIUS', 'URANUS_ROTATION', 'VENUS_EQUATOR_RADIUS', 'VENUS_GM', 'VENUS_J2', 'VENUS_MASS', 'VENUS_POLAR_RADIUS', 'VENUS_ROTATION', 'cmath', 'np']
class Tilt:
    """
    
        Gravity-based estimation of attitude.
    
        Parameters
        ----------
        acc : numpy.ndarray, default: None
            N-by-3 array with measurements of acceleration in in m/s^2
        mag : numpy.ndarray, default: None
            N-by-3 array with measurements of magnetic field in mT
        as_angles : bool, default: False
            Whether to return the attitude as rpy angles.
    
        Attributes
        ----------
        acc : numpy.ndarray
            N-by-3 array with N tri-axial accelerometer samples.
        mag : numpy.ndarray
            N-by-3 array with N tri-axial magnetometer samples.
        Q : numpy.ndarray, default: None
            N-by-4 or N-by-3 array with
        as_angles : bool, default: False
            Whether to return the attitude as rpy angles.
    
        Raises
        ------
        ValueError
            When shape of input array ``acc`` is not (N, 3)
    
        Examples
        --------
        Assuming we have 3-axis accelerometer data in N-by-3 arrays, we can simply
        give these samples to the constructor. The tilt estimation works solely
        with accelerometer samples.
    
        >>> from ahrs.filters import Tilt
        >>> tilt = Tilt(acc_data)
    
        The estimated quaternions are saved in the attribute ``Q``.
    
        >>> tilt.Q
        array([[0.76901856, 0.60247641, -0.16815772, 0.13174072],
               [0.77310283, 0.59724644, -0.16900433, 0.1305612 ],
               [0.7735134,  0.59644005, -0.1697294,  0.1308748 ],
               ...,
               [0.7800751,  0.59908629, -0.14315079, 0.10993772],
               [0.77916118, 0.59945374, -0.14520157, 0.11171197],
               [0.77038613, 0.61061868, -0.14375869, 0.11394512]])
        >>> tilt.Q.shape
        (1000, 4)
    
        If we desire to estimate each sample independently, we call the
        corresponding method with each sample individually.
    
        >>> tilt = Tilt()
        >>> num_samples = len(acc_data)
        >>> Q = np.zeros((num_samples, 4))  # Allocate quaternions array
        >>> for t in range(num_samples):
        ...     Q[t] = tilt.estimate(acc_data[t])
        ...
        >>> tilt.Q[:5]
        array([[0.76901856, 0.60247641, -0.16815772, 0.13174072],
               [0.77310283, 0.59724644, -0.16900433, 0.1305612 ],
               [0.7735134,  0.59644005, -0.1697294,  0.1308748 ],
               [0.77294791, 0.59913005, -0.16502363, 0.12791369],
               [0.76936935, 0.60323746, -0.16540014, 0.12968487]])
    
        Originally, this estimation computes first the Roll-Pitch-Yaw angles and
        then converts them to Quaternions. If we desire the angles instead, we set
        it so in the parameters.
    
        >>> tilt = Tilt(acc_data, as_angles=True)
        >>> type(tilt.Q), tilt.Q.shape
        (<class 'numpy.ndarray'>, (1000, 3))
        >>> tilt.Q[:5]
        array([[8.27467200e-04,  4.36167791e-06, 0.00000000e+00],
               [9.99352822e-04,  8.38015258e-05, 0.00000000e+00],
               [1.30423484e-03,  1.72201573e-04, 0.00000000e+00],
               [1.60337482e-03,  8.53081042e-05, 0.00000000e+00],
               [1.98459171e-03, -8.34729603e-05, 0.00000000e+00]])
    
        .. note::
            It will return the angles, in degrees, following the standard order
            ``Roll->Pitch->Yaw``.
    
        The yaw angle is, expectedly, equal to zero, because the heading cannot be
        estimated with the gravity acceleration only.
    
        For this reason, magnetometer data can be used to estimate the yaw. This is
        also implemented and the magnetometer will be taken into account when given
        as parameter.
    
        >>> tilt = Tilt(acc=acc_data, mag=mag_data, as_angles=True)
        >>> tilt.Q[:5]
        array([[8.27467200e-04,  4.36167791e-06, -4.54352439e-02],
               [9.99352822e-04,  8.38015258e-05, -4.52836926e-02],
               [1.30423484e-03,  1.72201573e-04, -4.49355365e-02],
               [1.60337482e-03,  8.53081042e-05, -4.44276770e-02],
               [1.98459171e-03, -8.34729603e-05, -4.36931634e-02]])
    
        
    """
    def __init__(self, acc: numpy.ndarray = None, mag: numpy.ndarray = None, **kwargs):
        ...
    def _compute_all(self) -> numpy.ndarray:
        """
        
                Estimate the orientation given all data.
        
                Attributes ``acc`` and ``mag`` must contain data. It is assumed that
                these attributes have the same shape (M, 3), where M is the number of
                observations.
        
                The full estimation is vectorized, to avoid the use of a time-wasting
                loop.
        
                Returns
                -------
                Q : numpy.ndarray
                    M-by-4 array with all estimated quaternions, where M is the number
                    of samples. It returns an M-by-3 array, if the flag ``as_angles``
                    is set to ``True``.
        
                
        """
    def estimate(self, acc: numpy.ndarray, mag: numpy.ndarray = None) -> numpy.ndarray:
        """
        
                Estimate the quaternion from the tilting read by an orthogonal
                tri-axial array of accelerometers.
        
                The orientation of the roll and pitch angles is estimated using the
                measurements of the accelerometers, and finally converted to a
                quaternion representation according to [WikiConversions]_
        
                Parameters
                ----------
                acc : numpy.ndarray
                    Sample of tri-axial Accelerometer in m/s^2.
                mag : numpy.ndarray, default: None
                    N-by-3 array with measurements of magnetic field in mT.
        
                Returns
                -------
                q : numpy.ndarray
                    Estimated attitude.
        
                Examples
                --------
                >>> acc_data = np.array([4.098297, 8.663757, 2.1355896])
                >>> mag_data = np.array([-28.71550512, -25.92743566, 4.75683931])
                >>> from ahrs.filters import Tilt
                >>> tilt = Tilt()
                >>> tilt.estimate(acc=acc_data, mag=mag_data)   # Estimate attitude as quaternion
                array([0.09867706 0.33683592 0.52706394 0.77395607])
        
                Optionally, the attitude can be retrieved as roll-pitch-yaw angles, in
                degrees.
        
                >>> tilt = Tilt(as_angles=True)
                >>> tilt.estimate(acc=acc_data, mag=mag_data)
                array([ 76.15281566 -24.66891862 146.02634429])
        
                
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
