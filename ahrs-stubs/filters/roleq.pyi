"""

Recursive Optimal Linear Estimator of Quaternion
================================================

This is a modified `OLEQ <./oleq.html>`_, where a recursive estimation of the
attitude is made with the measured angular velocity [Zhou2018]_. This
estimation is set as the initial value for the OLEQ estimation, simplyfing the
rotational operations.

First, the quaternion :math:`\\mathbf{q}_\\omega` is estimated from the angular
velocity, :math:`\\boldsymbol\\omega=\\begin{bmatrix}\\omega_x & \\omega_y &
\\omega_z \\end{bmatrix}^T`, measured by the gyroscopes, in rad/s, at a time
:math:`t` as:

.. math::
    \\mathbf{q}_\\omega = \\Big(\\mathbf{I}_4 + \\frac{\\Delta t}{2}\\boldsymbol\\Omega_t\\Big)\\mathbf{q}_{t-1} =
    \\begin{bmatrix}
    q_w - \\frac{\\Delta t}{2} \\omega_x q_x - \\frac{\\Delta t}{2} \\omega_y q_y - \\frac{\\Delta t}{2} \\omega_z q_z\\\\
    q_x + \\frac{\\Delta t}{2} \\omega_x q_w - \\frac{\\Delta t}{2} \\omega_y q_z + \\frac{\\Delta t}{2} \\omega_z q_y\\\\
    q_y + \\frac{\\Delta t}{2} \\omega_x q_z + \\frac{\\Delta t}{2} \\omega_y q_w - \\frac{\\Delta t}{2} \\omega_z q_x\\\\
    q_z - \\frac{\\Delta t}{2} \\omega_x q_y + \\frac{\\Delta t}{2} \\omega_y q_x + \\frac{\\Delta t}{2} \\omega_z q_w
    \\end{bmatrix}

Then, the attitude is "corrected" through OLEQ using a single multiplication of
its rotation operator:

.. math::
    \\mathbf{q}_\\mathbf{ROLEQ} = \\frac{1}{2}\\Big(\\mathbf{I}_4 + \\sum_{i=1}^na_i\\mathbf{W}_i\\Big)\\mathbf{q}_\\omega

where each :math:`\\mathbf{W}` (one for accelerations and one for magnetic
field) is built from their reference vectors, :math:`D^r`, and measurements,
:math:`D^b`, exactly as in OLEQ:

.. math::
    \\begin{array}{rcl}
    \\mathbf{W} &=& D_x^r\\mathbf{M}_1 + D_y^r\\mathbf{M}_2 + D_z^r\\mathbf{M}_3 \\\\ && \\\\
    \\mathbf{M}_1 &=&
    \\begin{bmatrix}
    D_x^b & 0 & D_z^b & -D_y^b \\\\
    0 & D_x^b & D_y^b & D_z^b \\\\
    D_z^b & D_y^b & -D_x^b & 0 \\\\
    -D_y^b & D_z^b & 0 & -D_x^b
    \\end{bmatrix} \\\\
    \\mathbf{M}_2 &=&
    \\begin{bmatrix}
    D_y^b & -D_z^b & 0 & D_x^b \\\\
    -D_z^b & -D_y^b & D_x^b & 0 \\\\
    0 & D_x^b & D_y^b & D_z^b \\\\
    D_x^b & 0 & D_z^b & -D_y^b
    \\end{bmatrix} \\\\
    \\mathbf{M}_3 &=&
    \\begin{bmatrix}
    D_z^b & D_y^b & -D_x^b & 0 \\\\
    D_y^b & -D_z^b & 0 & D_x^b \\\\
    -D_x^b & 0 & -D_z^b & D_y^b \\\\
    0 & D_x^b & D_y^b & D_z^b
    \\end{bmatrix}
    \\end{array}

It is noticeable that, for OLEQ, a random quaternion was used as a starting
value for an iterative procedure to find the optimal quaternion. Here, that
initial value is now :math:`\\mathbf{q}_\\omega` and a simple product (instead
of a large iterative product) is required.

In this way, the quaternions are recursively computed with much fewer
computations, and the accuracy is maintained.

For this case, however the three sensor data (gyroscopes, accelerometers and
magnetometers) have to be provided, along with the an initial quaternion,
:math:`\\mathbf{q}_0` from which the attitude will be built upon.

References
----------
.. [Zhou2018] Zhou, Z.; Wu, J.; Wang, J.; Fourati, H. Optimal, Recursive and
    Sub-Optimal Linear Solutions to Attitude Determination from Vector
    Observations for GNSS/Accelerometer/Magnetometer Orientation Measurement.
    Remote Sens. 2018, 10, 377.
    (https://www.mdpi.com/2072-4292/10/3/377)

"""
from __future__ import annotations
from ahrs.common.mathfuncs import cosd
from ahrs.common.mathfuncs import sind
from ahrs.common.orientation import ecompass
import numpy
import numpy as np
__all__ = ['ROLEQ', 'cosd', 'ecompass', 'np', 'sind']
class ROLEQ:
    """
    
        Recursive Optimal Linear Estimator of Quaternion
    
        Uses OLEQ to estimate the initial attitude.
    
        Parameters
        ----------
        gyr : numpy.ndarray, default: None
            N-by-3 array with measurements of angular velocity in rad/s.
        acc : numpy.ndarray, default: None
            N-by-3 array with measurements of acceleration in in m/s^2.
        mag : numpy.ndarray, default: None
            N-by-3 array with measurements of magnetic field in mT.
    
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
        Q : numpy.array, default: None
            M-by-4 Array with all estimated quaternions, where M is the number of
            samples. Equal to None when no estimation is performed.
    
        Raises
        ------
        ValueError
            When dimension of input arrays ``gyr``, ``acc`` or ``mag`` are not
            equal.
    
        Examples
        --------
        >>> gyr_data.shape, acc_data.shape, mag_data.shape      # NumPy arrays with sensor data
        ((1000, 3), (1000, 3), (1000, 3))
        >>> from ahrs.filters import ROLEQ
        >>> orientation = ROLEQ(gyr=gyr_data, acc=acc_data, mag=mag_data)
        >>> orientation.Q.shape                 # Estimated attitude
        (1000, 4)
    
        
    """
    def WW(self, Db, Dr):
        """
        W Matrix
        
                .. math::
                    \\mathbf{W} = D_x^r\\mathbf{M}_1 + D_y^r\\mathbf{M}_2 + D_z^r\\mathbf{M}_3
        
                Parameters
                ----------
                Db : numpy.ndarray
                    Normalized tri-axial observations vector.
                Dr : numpy.ndarray
                    Normalized tri-axial reference vector.
        
                Returns
                -------
                W_matrix : numpy.ndarray
                    W Matrix.
                
        """
    def __init__(self, gyr: numpy.ndarray = None, acc: numpy.ndarray = None, mag: numpy.ndarray = None, weights: numpy.ndarray = None, magnetic_ref: numpy.ndarray = None, frame: str = 'NED', **kwargs):
        ...
    def _compute_all(self) -> numpy.ndarray:
        """
        Estimate the quaternions given all data.
        
                Attributes ``gyr``, ``acc`` and ``mag`` must contain data.
        
                Returns
                -------
                Q : array
                    M-by-4 Array with all estimated quaternions, where M is the number
                    of samples.
        
                
        """
    def _set_reference_frames(self, mref: float, frame: str = 'NED'):
        ...
    def attitude_propagation(self, q: numpy.ndarray, omega: numpy.ndarray) -> numpy.ndarray:
        """
        Attitude estimation from previous quaternion and current angular velocity.
        
                .. math::
                    \\mathbf{q}_\\omega = \\Big(\\mathbf{I}_4 + \\frac{\\Delta t}{2}\\boldsymbol\\Omega_t\\Big)\\mathbf{q}_{t-1} =
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
                    Angular velocity, in rad/s.
        
                Returns
                -------
                q : numpy.ndarray
                    Attitude as a quaternion.
                
        """
    def oleq(self, acc: numpy.ndarray, mag: numpy.ndarray, q_omega: numpy.ndarray) -> numpy.ndarray:
        """
        OLEQ with a single rotation by R.
        
                Parameters
                ----------
                acc : numpy.ndarray
                    Sample of tri-axial Accelerometer.
                mag : numpy.ndarray
                    Sample of tri-axial Magnetometer.
                q_omega : numpy.ndarray
                    Preceding quaternion estimated with angular velocity.
        
                Returns
                -------
                q : np.ndarray
                    Final quaternion.
        
                
        """
    def update(self, q: numpy.ndarray, gyr: numpy.ndarray, acc: numpy.ndarray, mag: numpy.ndarray) -> numpy.ndarray:
        """
        Update Attitude with a Recursive OLEQ
        
                Parameters
                ----------
                q : numpy.ndarray
                    A-priori quaternion.
                gyr : numpy.ndarray
                    Sample of angular velocity in rad/s
                acc : numpy.ndarray
                    Sample of tri-axial Accelerometer in m/s^2
                mag : numpy.ndarray
                    Sample of tri-axial Magnetometer in mT
        
                Returns
                -------
                q : numpy.ndarray
                    Estimated quaternion.
        
                
        """
