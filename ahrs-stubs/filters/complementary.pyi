"""

Complementary Filter
====================

Attitude quaternion obtained with gyroscope and accelerometer-magnetometer
measurements, via complementary filter.

First, the current orientation is estimated at time :math:`t`, from a previous
orientation at time :math:`t-1`, and a given angular velocity,
:math:`\\omega`, in rad/s.

This orientation is computed by numerically integrating the angular velocity
and adding it to the previous orientation, which is known as an **attitude
propagation**.

.. math::
    \\begin{array}{rcl}
    \\mathbf{q}_\\omega &=& \\Big(\\mathbf{I}_4 + \\frac{\\Delta t}{2}\\boldsymbol\\Omega_t\\Big)\\mathbf{q}_{t-1} \\\\
    &=&
    \\begin{bmatrix}
    1 & -\\frac{\\Delta t}{2}\\omega_x & -\\frac{\\Delta t}{2}\\omega_y & -\\frac{\\Delta t}{2}\\omega_z \\\\
    \\frac{\\Delta t}{2}\\omega_x & 1 & \\frac{\\Delta t}{2}\\omega_z & -\\frac{\\Delta t}{2}\\omega_y \\\\
    \\frac{\\Delta t}{2}\\omega_y & -\\frac{\\Delta t}{2}\\omega_z & 1 & \\frac{\\Delta t}{2}\\omega_x \\\\
    \\frac{\\Delta t}{2}\\omega_z & \\frac{\\Delta t}{2}\\omega_y & -\\frac{\\Delta t}{2}\\omega_x & 1
    \\end{bmatrix}
    \\begin{bmatrix}q_w \\\\ q_x \\\\ q_y \\\\ q_z \\end{bmatrix} \\\\
    &=&
    \\begin{bmatrix}
        q_w - \\frac{\\Delta t}{2} \\omega_x q_x - \\frac{\\Delta t}{2} \\omega_y q_y - \\frac{\\Delta t}{2} \\omega_z q_z\\\\
        q_x + \\frac{\\Delta t}{2} \\omega_x q_w - \\frac{\\Delta t}{2} \\omega_y q_z + \\frac{\\Delta t}{2} \\omega_z q_y\\\\
        q_y + \\frac{\\Delta t}{2} \\omega_x q_z + \\frac{\\Delta t}{2} \\omega_y q_w - \\frac{\\Delta t}{2} \\omega_z q_x\\\\
        q_z - \\frac{\\Delta t}{2} \\omega_x q_y + \\frac{\\Delta t}{2} \\omega_y q_x + \\frac{\\Delta t}{2} \\omega_z q_w
    \\end{bmatrix}
    \\end{array}

Secondly, the *tilt* is computed from the accelerometer measurements as:

.. math::
    \\begin{array}{rcl}
    \\theta &=& \\mathrm{arctan2}(a_y, a_z) \\\\
    \\phi &=& \\mathrm{arctan2}\\big(-a_x, \\sqrt{a_y^2+a_z^2}\\big)
    \\end{array}

Only the pitch, :math:`\\phi`, and roll, :math:`\\theta`, angles are computed,
leaving the yaw angle, :math:`\\psi` equal to zero.

If a magnetometer sample is available, the yaw angle can be computed. First
compensate the measurement using the *tilt*:

.. math::
    \\begin{array}{rcl}
    \\mathbf{b} &=&
    \\begin{bmatrix}
        \\cos\\theta & \\sin\\theta\\sin\\phi & \\sin\\theta\\cos\\phi \\\\
        0 & \\cos\\phi & -\\sin\\phi \\\\
        -\\sin\\theta & \\cos\\theta\\sin\\phi & \\cos\\theta\\cos\\phi
    \\end{bmatrix}
    \\begin{bmatrix}m_x \\\\ m_y \\\\ m_z\\end{bmatrix} \\\\
    \\begin{bmatrix}b_x \\\\ b_y \\\\ b_z\\end{bmatrix} &=&
    \\begin{bmatrix}
        m_x\\cos\\theta + m_y\\sin\\theta\\sin\\phi + m_z\\sin\\theta\\cos\\phi \\\\
        m_y\\cos\\phi - m_z\\sin\\phi \\\\
        -m_x\\sin\\theta + m_y\\cos\\theta\\sin\\phi + m_z\\cos\\theta\\cos\\phi
    \\end{bmatrix}
    \\end{array}

Then, the yaw angle, :math:`\\psi`, is obtained as:

.. math::
    \\begin{array}{rcl}
    \\psi &=& \\mathrm{arctan2}(-b_y, b_x) \\\\
    &=& \\mathrm{arctan2}\\big(m_z\\sin\\phi - m_y\\cos\\phi, \\; m_x\\cos\\theta + \\sin\\theta(m_y\\sin\\phi + m_z\\cos\\phi)\\big)
    \\end{array}

We transform the roll-pitch-yaw angles to a quaternion representation:

.. math::
    \\mathbf{q}_{am} =
    \\begin{pmatrix}q_w\\\\q_x\\\\q_y\\\\q_z\\end{pmatrix} =
    \\begin{pmatrix}
        \\cos\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big) + \\sin\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) \\\\
        \\sin\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big) - \\cos\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) \\\\
        \\cos\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big) + \\sin\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) \\\\
        \\cos\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) - \\sin\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big)
    \\end{pmatrix}

Finally, after each orientation is estimated independently, they are fused with
the complementary filter.

.. math::
    \\mathbf{q} = (1 - \\alpha) \\mathbf{q}_\\omega + \\alpha\\mathbf{q}_{am}

where :math:`\\mathbf{q}_\\omega` is the attitude estimated from the gyroscope,
:math:`\\mathbf{q}_{am}` is the attitude estimated from the accelerometer and
the magnetometer, and :math:`\\alpha` is the gain of the filter.

The filter gain must be a floating value within the range :math:`[0.0, 1.0]`.
It can be seen that when :math:`\\alpha=1`, the attitude is estimated entirely
with the accelerometer and the magnetometer. When :math:`\\alpha=0`, it is
estimated solely with the gyroscope. The values within the range decide how
much of each estimation is "blended" into the quaternion.

This is actually a simple implementation of `LERP
<https://en.wikipedia.org/wiki/Linear_interpolation>`_ commonly used to
linearly interpolate quaternions with small differences between them.

"""
from __future__ import annotations
import numpy as np
import numpy
__all__ = ['Complementary', 'np']
class Complementary:
    """
    
        Complementary filter for attitude estimation as quaternion.
    
        Parameters
        ----------
        gyr : numpy.ndarray, default: None
            N-by-3 array with measurements of angular velocity, in rad/s.
        acc : numpy.ndarray, default: None
            N-by-3 array with measurements of acceleration, in m/s^2.
        mag : numpy.ndarray, default: None
            N-by-3 array with measurements of magnetic field, in mT.
        frequency : float, default: 100.0
            Sampling frequency in Herz.
        Dt : float, default: 0.01
            Sampling step in seconds. Inverse of sampling frequency. Not required
            if ``frequency`` value is given.
        gain : float, default: 0.1
            Filter gain.
        q0 : numpy.ndarray, default: None
            Initial orientation, as a versor (normalized quaternion).
    
        Raises
        ------
        ValueError
            When dimension of input arrays ``acc``, ``gyr``, or ``mag`` are not equal.
    
        
    """
    def __init__(self, gyr: numpy.ndarray = None, acc: numpy.ndarray = None, mag: numpy.ndarray = None, frequency: float = 100.0, gain = 0.1, **kwargs):
        ...
    def _compute_all(self) -> numpy.ndarray:
        """
        Estimate the quaternions given all data
        
                Attributes ``gyr``, ``acc`` and, optionally, ``mag`` must contain data.
        
                Returns
                -------
                Q : numpy.ndarray
                    M-by-4 Array with all estimated quaternions, where M is the number
                    of samples.
        
                
        """
    def am_estimation(self, acc: numpy.ndarray, mag: numpy.ndarray = None) -> numpy.ndarray:
        """
        Attitude estimation from an Accelerometer-Magnetometer architecture.
        
                First estimate the tilt from a given accelerometer sample
                :math:`\\mathbf{a}=\\begin{bmatrix}a_x & a_y & a_z\\end{bmatrix}^T` as:
        
                .. math::
                    \\begin{array}{rcl}
                    \\theta &=& \\mathrm{arctan2}(a_y, a_z) \\\\
                    \\phi &=& \\mathrm{arctan2}\\big(-a_x, \\sqrt{a_y^2+a_z^2}\\big)
                    \\end{array}
        
                Then the yaw angle, :math:`\\psi`, is computed, if a magnetometer 
                sample :math:`\\mathbf{m}=\\begin{bmatrix}m_x & m_y & m_z\\end{bmatrix}^T`
                is available:
        
                .. math::
                    \\psi = \\mathrm{arctan2}(-b_y, b_x)
        
                where
        
                .. math::
                    \\begin{array}{rcl}
                    b_x &=& m_x\\cos\\theta + m_y\\sin\\theta\\sin\\phi + m_z\\sin\\theta\\cos\\phi \\\\
                    b_y &=& m_y\\cos\\phi - m_z\\sin\\phi
                    \\end{array}
        
                And the roll-pitch-yaw angles are transformed to a quaternion that is
                then returned:
        
                .. math::
                    \\mathbf{q}_{am} =
                    \\begin{pmatrix}q_w\\\\q_x\\\\q_y\\\\q_z\\end{pmatrix} =
                    \\begin{pmatrix}
                    \\cos\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big) + \\sin\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) \\\\
                    \\sin\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big) - \\cos\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) \\\\
                    \\cos\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big) + \\sin\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) \\\\
                    \\cos\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) - \\sin\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big)
                    \\end{pmatrix}
        
                Parameters
                ----------
                acc : numpy.ndarray
                    Tri-axial sample of the accelerometer.
                mag : numpy.ndarray, default: None
                    Tri-axial sample of the magnetometer.
        
                Returns
                -------
                q_am : numpy.ndarray
                    Estimated attitude.
                
        """
    def attitude_propagation(self, q: numpy.ndarray, omega: numpy.ndarray) -> numpy.ndarray:
        """
        
                Attitude propagation of the orientation.
        
                Estimate the current orientation at time :math:`t`, from a given
                orientation at time :math:`t-1` and a given angular velocity,
                :math:`\\omega`, in rad/s.
        
                It is computed by numerically integrating the angular velocity and
                adding it to the previous orientation.
        
                .. math::
                    \\mathbf{q}_\\omega =
                    \\begin{bmatrix}
                    q_w - \\frac{\\Delta t}{2} \\omega_x q_x - \\frac{\\Delta t}{2} \\omega_y q_y - \\frac{\\Delta t}{2} \\omega_z q_z\\\\
                    q_x + \\frac{\\Delta t}{2} \\omega_x q_w - \\frac{\\Delta t}{2} \\omega_y q_z + \\frac{\\Delta t}{2} \\omega_z q_y\\\\
                    q_y + \\frac{\\Delta t}{2} \\omega_x q_z + \\frac{\\Delta t}{2} \\omega_y q_w - \\frac{\\Delta t}{2} \\omega_z q_x\\\\
                    q_z - \\frac{\\Delta t}{2} \\omega_x q_y + \\frac{\\Delta t}{2} \\omega_y q_x + \\frac{\\Delta t}{2} \\omega_z q_w
                    \\end{bmatrix}
        
                Parameters
                ----------
                q : numpy.ndarray
                    A-priori quaternion.
                omega : numpy.ndarray
                    Tri-axial angular velocity, in rad/s.
        
                Returns
                -------
                q_omega : numpy.ndarray
                    Estimated orientation, as quaternion.
                
        """
    def update(self, q: numpy.ndarray, gyr: numpy.ndarray, acc: numpy.ndarray, mag: numpy.ndarray = None) -> numpy.ndarray:
        """
        
                Attitude Estimation from given measurements and previous orientation.
        
                The new orientation is first estimated with the angular velocity, then
                another orientation is computed using the accelerometers and
                magnetometers. The magnetometer is optional.
        
                Each orientation is estimated independently and fused with a
                complementary filter.
        
                .. math::
                    \\mathbf{q} = (1 - \\alpha) \\mathbf{q}_\\omega + \\alpha\\mathbf{q}_{am}
        
                Parameters
                ----------
                q : numpy.ndarray
                    A-priori quaternion.
                gyr : numpy.ndarray
                    Sample of tri-axial Gyroscope in rad/s.
                acc : numpy.ndarray
                    Sample of tri-axial Accelerometer in m/s^2.
                mag : numpy.ndarray, default: None
                    Sample of tri-axial Magnetometer in uT.
        
                Returns
                -------
                q : numpy.ndarray
                    Estimated quaternion.
        
                
        """
