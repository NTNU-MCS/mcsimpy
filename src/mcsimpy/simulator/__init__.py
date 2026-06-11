"""
Vessel Simulation Models (:mod:`mcsimpy.simulator`)
=======================================================

The mcsimpy simulator package contains a set of
different simulation models.

Contents
---------
The following modules are found in the `mcsimpy.simulator`.

- vessel.py        --- Base class for vessel simulation models.

- csad.py          --- Simulation models for C/S Arctic Drillship.

- gunnerus.py      --- Simulation models for R/V Gunnerus.
"""

# Import all the models s.t we can access them from e.g.
from mcsimpy.simulator.csad import CSAD_DP_6DOF, CSADMan3DOF, CSAD_DP_Seakeeping
from mcsimpy.simulator.gunnerus import GunnerusManeuvering3DoF, RVG_DP_6DOF
