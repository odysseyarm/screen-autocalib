"""

Optimal Linear Estimator of Quaternion
======================================

Considering an attitude determination model from a pair of vector observations:

.. math::
    \\mathbf{D}^b = \\mathbf{CD}^r

where :math:`\\mathbf{D}_i^b=\\begin{bmatrix}D_{x,i}^b & D_{y,i}^b & D_{z,i}^b\\end{bmatrix}^T`
and :math:`\\mathbf{D}_i^r=\\begin{bmatrix}D_{x,i}^r & D_{y,i}^r & D_{z,i}^r\\end{bmatrix}^T`
are the *i*-th pair of normalized vector observations from the body frame :math:`b`
and the reference frame :math:`r`.

The goal is to find the optimal attitude :math:`\\mathbf{C}\\in\\mathbb{R}^{3\\times 3}`
relating both vectors. The famous `Wahba's problem
<https://en.wikipedia.org/wiki/Wahba%27s_problem>`_ can help us to find
:math:`\\mathbf{C}` from a set of observations and a least-squares method of
the form:

.. math::
    L(\\mathbf{C}) = \\sum_{i=1}^n a_i \\|\\mathbf{D}_i^b - \\mathbf{CD}_i^r \\|^2
    
being :math:`a_i` the weight of the *i*-th sensor output. The goal of **OLEQ**
is to find this optimal attitude, but in the form of a quaternion [Zhou2018]_.

First, notice that the attitude matrix is related to quaternion
:math:`\\mathbf{q}=\\begin{bmatrix}q_w & q_x & q_y & q_z\\end{bmatrix}^T` via:

.. math::
    \\mathbf{C} = \\begin{bmatrix}\\mathbf{P}_1\\mathbf{q} & \\mathbf{P}_2\\mathbf{q} & \\mathbf{P}_3\\mathbf{q}\\end{bmatrix}

where the decomposition matrices are:

.. math::
    \\begin{array}{rcl}
    \\mathbf{P}_1 &=&
    \\begin{bmatrix}q_w & q_x & -q_y & -q_z \\\\ -q_z & q_y & q_x & -q_w \\\\ q_y & q_z & q_w & q_x \\end{bmatrix} \\\\
    \\mathbf{P}_2 &=&
    \\begin{bmatrix}q_z & q_y & q_x & q_w \\\\ q_w & -q_x & q_y & -q_z \\\\ -q_x & -q_w & q_z & q_y \\end{bmatrix} \\\\
    \\mathbf{P}_3 &=&
    \\begin{bmatrix}-q_y & q_z & -q_w & q_x \\\\ q_x & q_w & q_z & q_y \\\\ q_w & -q_x & -q_y & q_z \\end{bmatrix}
    \\end{array}

It is accepted that :math:`\\mathbf{P}_1^T=\\mathbf{P}_1^\\dagger`,
:math:`\\mathbf{P}_2^T=\\mathbf{P}_2^\\dagger`, and :math:`\\mathbf{P}_3^T=\\mathbf{P}_3^\\dagger`,
where the notation :math:`^\\dagger` stands for the `Moore-Penrose pseudo-
inverse <https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse>`_. So,
the reference and observation vectors can be related to the quaternion with a
:math:`4\\times 4` matrix of the form:

.. math::
    \\begin{array}{rcl}
    \\mathbf{D}^b &=& \\mathbf{K}(\\mathbf{q}) \\mathbf{q} \\\\
    \\mathbf{D}^b &=& \\big(D_x^r\\mathbf{P}_1 + D_y^r\\mathbf{P}_2 + D_z^r\\mathbf{P}_3\\big) \\mathbf{q}
    \\end{array}

Knowing that :math:`\\mathbf{K}^T(\\mathbf{q})=\\mathbf{K}^\\dagger(\\mathbf{q})`,
the expression can be expanded to:

.. math::
    \\begin{array}{rcl}
    \\mathbf{K}^T(\\mathbf{q})\\mathbf{D}^b &=&
    D_x^r\\mathbf{P}_1^T\\mathbf{D}^b + D_y^r\\mathbf{P}_2^T\\mathbf{D}^b + D_z^r\\mathbf{P}_3^T\\mathbf{D}^b \\\\
    \\mathbf{Wq} &=& D_x^r\\mathbf{M}_1\\mathbf{q} + D_y^r\\mathbf{M}_2\\mathbf{q} + D_z^r\\mathbf{M}_3\\mathbf{q}
    \\end{array}

where :math:`\\mathbf{W}` is built with:

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

Now the attitude estimation is shifted to :math:`\\mathbf{Wq}=\\mathbf{q}`. If
treated as an iterative dynamical system, the quatenion at the *n*-th itreation
is calculated as:

.. math::
    \\mathbf{q}(n) = \\mathbf{Wq}(n-1)

It is possible to list all rotation equations as:

.. math::
    \\begin{bmatrix}
    \\sqrt{a_1}\\mathbf{I}_4 \\\\ \\vdots \\\\ \\sqrt{a_n}\\mathbf{I}_4
    \\end{bmatrix} \\mathbf{q} =
    \\begin{bmatrix}
    \\sqrt{a_1}\\mathbf{W}_1 \\\\ \\vdots \\\\ \\sqrt{a_n}\\mathbf{W}_n
    \\end{bmatrix} \\mathbf{q}

Leading to a pre-multiplication of the form:

.. math::
    \\mathbf{q} = \\Big(\\sum_{i=1}^na_i\\mathbf{W}_i\\Big)\\mathbf{q}

A stable and continuous solution to each equation is done by pre-multiplying
:math:`\\frac{1}{2}(\\mathbf{W}_i+\\mathbf{I}_4)`.

.. math::
    \\begin{bmatrix}
    \\sqrt{a_1}\\mathbf{I}_4 \\\\ \\vdots \\\\ \\sqrt{a_n}\\mathbf{I}_4
    \\end{bmatrix} \\mathbf{q} =
    \\begin{bmatrix}
    \\frac{1}{2}\\sqrt{a_1}(\\mathbf{W}_1+\\mathbf{I}_4) \\\\ \\vdots \\\\ \\frac{1}{2}\\sqrt{a_n}(\\mathbf{W}_n+\\mathbf{I}_4)
    \\end{bmatrix} \\mathbf{q}

Based on `Brouwer's fixed-point theorem <https://en.wikipedia.org/wiki/Brouwer_fixed-point_theorem>`_,
it is possible to recursively obtain the normalized optimal quaternion by
rotating a randomly given initial quaternion, :math:`\\mathbf{q}_\\mathrm{rand}`,
over and over again indefinitely.

.. math::
    \\mathbf{q} = \\frac{\\mathbf{W} + \\mathbf{I}}{2} \\mathbf{q}_\\mathrm{rand}

This equals the least-square of the set of pre-computed single rotated
quaternions.

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
import numpy
import numpy as np
__all__ = ['OLEQ', 'cosd', 'np', 'sind']
class OLEQ:
    """
    
        Optimal Linear Estimator of Quaternion
    
        Parameters
        ----------
        acc : numpy.ndarray, default: None
            N-by-3 array with measurements of acceleration in in m/s^2
        mag : numpy.ndarray, default: None
            N-by-3 array with measurements of magnetic field in mT
        magnetic_ref : float or numpy.ndarray
            Local magnetic reference.
        frame : str, default: 'NED'
            Local tangent plane coordinate frame. Valid options are right-handed
            ``'NED'`` for North-East-Down and ``'ENU'`` for East-North-Up.
    
        Raises
        ------
        ValueError
            When dimension of input arrays ``acc`` and ``mag`` are not equal.
    
        Examples
        --------
        >>> acc_data.shape, mag_data.shape      # NumPy arrays with sensor data
        ((1000, 3), (1000, 3))
        >>> from ahrs.filters import OLEQ
        >>> orientation = OLEQ(acc=acc_data, mag=mag_data)
        >>> orientation.Q.shape                 # Estimated attitude
        (1000, 4)
    
        
    """
    def WW(self, Db: numpy.ndarray, Dr: numpy.ndarray) -> numpy.ndarray:
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
    def __init__(self, acc: numpy.ndarray = None, mag: numpy.ndarray = None, weights: numpy.ndarray = None, magnetic_ref: numpy.ndarray = None, frame: str = 'NED'):
        ...
    def _compute_all(self) -> numpy.ndarray:
        """
        Estimate the quaternions given all data.
        
                Attributes ``acc`` and ``mag`` must contain data.
        
                Returns
                -------
                Q : array
                    M-by-4 Array with all estimated quaternions, where M is the number
                    of samples.
        
                
        """
    def _set_reference_frames(self, mref: float, frame: str = 'NED') -> None:
        ...
    def estimate(self, acc: numpy.ndarray, mag: numpy.ndarray) -> numpy.ndarray:
        """
        Attitude Estimation
        
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
