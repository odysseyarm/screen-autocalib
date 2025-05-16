"""

Attitude Estimators
===================

These are the most common attitude filters.

"""
from __future__ import annotations
from ahrs.filters.angular import AngularRate
from ahrs.filters.aqua import AQUA
from ahrs.filters.complementary import Complementary
from ahrs.filters.davenport import Davenport
from ahrs.filters.ekf import EKF
from ahrs.filters.famc import FAMC
from ahrs.filters.flae import FLAE
from ahrs.filters.fourati import Fourati
from ahrs.filters.fqa import FQA
from ahrs.filters.madgwick import Madgwick
from ahrs.filters.mahony import Mahony
from ahrs.filters.oleq import OLEQ
from ahrs.filters.quest import QUEST
from ahrs.filters.roleq import ROLEQ
from ahrs.filters.saam import SAAM
from ahrs.filters.tilt import Tilt
from ahrs.filters.triad import TRIAD
from . import angular
from . import aqua
from . import complementary
from . import davenport
from . import ekf
from . import famc
from . import flae
from . import fourati
from . import fqa
from . import madgwick
from . import mahony
from . import oleq
from . import quest
from . import roleq
from . import saam
from . import tilt
from . import triad
__all__ = ['AQUA', 'AngularRate', 'Complementary', 'Davenport', 'EKF', 'FAMC', 'FLAE', 'FQA', 'Fourati', 'Madgwick', 'Mahony', 'OLEQ', 'QUEST', 'ROLEQ', 'SAAM', 'TRIAD', 'Tilt', 'angular', 'aqua', 'complementary', 'davenport', 'ekf', 'famc', 'flae', 'fourati', 'fqa', 'madgwick', 'mahony', 'oleq', 'quest', 'roleq', 'saam', 'tilt', 'triad']
