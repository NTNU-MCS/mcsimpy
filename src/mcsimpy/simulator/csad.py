# csad.py

# ----------------------------------------------------------------------------
# This code is part of the mcsimpy toolbox and repository.
# Created By: Jan-Erik Hygen
# Created Date: 2022-11-04
# Revised:
# Kristian Magnus Roen 06.11.25 Removed bug print and added dt fallback
# Copyright (C) 2023: NTNU, Trondheim
# Licensed under GPL-3.0-or-later
# ---------------------------------------------------------------------------

import numpy as np
import os
import json

from mcsimpy.simulator.vessel import Vessel3dof, Vessel6dof
from mcsimpy.utils import Smat, Rz

"""
Vessel models for C/S Arctic Drillship.

Data for CSAD Vessel is found in vessel_data/CSAD/

Models:
- 3DOF Maneuvering model
- 6DOF DP Model
"""

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "vessel_data", "CSAD")
)


class CSADMan3DOF(Vessel3dof):

    """
    CSAD 3DOF Maneuvering model.

    Attributes
    ----------

    self._Mrb : 3 x 3 array
        Rigid body mass matrix
    self._Ma : 3 x 3 array
        Added mass (infinte frequency)
    self._M : 3 x 3 array
        Total mass
    self._D : 3 x 3 array
        Linear potential damping
    self._Bv : 3 x 3 array
        Viscous damping
    """

    def __init__(self, dt, *args, config_file="vessel_json.json", **kwargs):
        config_file = os.path.join(DATA_DIR, config_file)
        super().__init__(dt, config_file=config_file)
        ind_3dof = np.ix_(self._dof_indices, self._dof_indices)
        with open(config_file, "r") as f:
            data = json.load(f)
        self._Mrb = np.asarray(data["MRB"])[ind_3dof]
        self._Ma = np.asarray(data["A"])[:, :, -1][ind_3dof]
        self._M = self._Mrb + self._Ma
        self._Minv = np.linalg.inv(self._M)
        self._D = np.asarray(data["B"])[:, :, -1][ind_3dof]
        self._Bv = np.asarray(data["Bv"])[ind_3dof]

    def x_dot(self, x, Uc, betac, tau):
        """Kinematic and kinetic equation for 6DOF simulation model.

        Parameters
        ----------
        x : array_like
            State vector with dimensions 2*DOF
        Uc : float
            Current velocity in earth-fixed frame
        betac : float
            Current direction in earth-fixed frame [rad]
        tau : array_like
            External loads (e.g wind, thrusters, ice, etc). Must correspond with numb. DOF.

        Returns
        -------
        x_dot : array_like
            The derivative of the state vector.
        """
        eta = x[: self._dof]
        nu = x[self._dof :]
        nu_cn = Uc * np.array([np.cos(betac), np.sin(betac), 0])
        nu_c = Rz(eta[2]).T @ nu_cn
        nu_r = nu - nu_c
        dnu_c = -Smat(nu) @ nu_c

        eta_dot = Rz(eta[2]) @ nu
        nu_dot = self._Minv @ (
            tau - self._Bv @ nu_r - self._D @ nu_r + self._Ma @ dnu_c
        )
        # self._x_dot = np.concatenate([eta_dot, nu_dot])
        return np.concatenate([eta_dot, nu_dot])


class CSAD_DP_6DOF(Vessel6dof):
    """6 DOF DP simulator model for CSAD.

    Simulator model for DP and low-speed applications.
    Zero-speed is assumed.

    No fluid memory effects are included yet.
    """

    def __init__(self, dt=0.01, *args, config_file="vessel_json.json", **kwargs):
        config_file = os.path.join(DATA_DIR, config_file)
        super().__init__(dt, config_file=config_file)
        with open(config_file, "r") as f:
            data = json.load(f)

        self._Mrb = np.asarray(data["MRB"])  # Rigid body mass matrix
        self._Ma = np.asarray(data["A"])[:, :, 41]  # Added mass matrix
        self._M = self._Mrb + self._Ma  # Total mass matrix
        self._Minv = np.linalg.inv(self._M)  # Inverse mass matrix

        self._Dp = np.asarray(data["B"])[:, :, 41]  # Potential damping
        self._Dv = np.asarray(data["Bv"])  # Viscous damping
        self._D = self._Dp + self._Dv  # Total damping
        self._D[3, 3] *= 2  # Increase roll damping

        self._G = np.asarray(data["C"])[:, :, 0]  # Restoring coefficients


class CSAD_DP_Seakeeping(Vessel6dof):
    def __init__(
        self,
        dt,
        config_file="vessel_json.json",
        vessel_abc="vesselABC_json.json",
    ):
        config_file = os.path.join(DATA_DIR, config_file)
        config_abc_file = os.path.join(DATA_DIR, vessel_abc)
        super().__init__(dt, config_file=config_file)
        with open(config_abc_file, "r") as f:
            vesselABC = json.load(f)
        with open(config_file, "r") as f:
            vessel_data = json.load(f)

        self._Mrb = np.asarray(vesselABC["MRB"])
        self._Ma = np.asarray(vesselABC["MA"])
        # Check if altering the different added mass term can better improve response
        # self._Ma[2, 2] = 235.0
        # self._Ma[2, 4] = 18.0
        # self._Ma[4, 4] = 115.0
        # self._Ma[3, 5] = self._Ma[3, 5]*0.1
        # self._Ma[3, 1] = self._Ma[3, 1]

        self._Minv = np.linalg.inv(self._Ma + self._Mrb)
        self._D = np.asarray(vessel_data["Bv"])
        # self._D[3,3] = 0.0
        self._G = np.asarray(vesselABC["G"])


# class CSAD_DP_Seakeeping(Vessel):

#     def __init__(self, dt, metod="RK4", config_file="vessel_json.json", vessel_abc="vesselABC_json.json", **kwargs):
#         with open(os.path.join(DATA_DIR, config_file)) as f:
#             vessel = json.load(f)
#         with open(os.path.join(DATA_DIR, vessel_abc)) as ff:
#             vesselABC = json.load(ff)
#         super().__init__(dt=dt, config_file=config_file, dof=6)
#         self._Mrb = np.asarray(vesselABC['MRB'])
#         self._Ma = np.asarray(vesselABC['MA'])
#         self._M = self._Mrb + self._Ma
#         self._Minv = np.linalg.inv(self._M)

#         self._Dp = np.asarray(vessel['B'])[:, :, -1]
#         self._Dv = np.asarray(vessel['Bv'])
#         self._D = self._Dp + self._Dv

#         self._G = np.asarray(vessel['C'])[:, :, 0]


# class CSAD6DOF(Vessel):
#     """
#     6 DOF stationkeeping model. Unified seakeeping and maneuvering model
#     with fluid memory effects.
#     """

#     def __init__(self, dt, *args, config_file="vessel_json.json", vessel_abc="vesselABC_json.json", **kwargs):
#         with open(os.path.join(DATA_DIR, config_file)) as f:
#             vessel = json.load(f)
#         with open(os.path.join(DATA_DIR, vessel_abc)) as ff:
#             vesselABC = json.load(ff)
#         super().__init__(dt=dt, config_file=config_file, dof=6)
#         self._dof = 6
#         self._Mrb = np.asarray(vessel['MRB'])
#         self._Ma = np.asarray(vessel['A'])[:, :, -1]
#         self._M = self._Mrb + self._Ma
#         self._Minv = np.linalg.inv(self._M)
#         self._D = np.asarray(vessel['Bv']) + np.asarray(vessel['B'])[:, :, -1]
#         # self._D = np.asarray(vessel['B'])[:, :, 0]
#         # self._D[3, 3] *= 10
#         # self._D[4, 4] *= 10
#         self._G = np.asarray(vessel['C'])[:, :, 0]
#         # self._G[3, 3] *= 5
#         self._x = np.zeros(2*self._dof)
#         self._x_dot = np.zeros_like(self._x)
#         self._eta = np.zeros(self._dof)
#         self._nu = np.zeros(self._dof)
#         self._dt = dt

#         # Set up state-space models for fluid memory effects.
#         # Fluid memory surge (no coupling for csad)
#         # self._Ar_surge = np.asarray(vesselABC['Ar'][0])
#         # self._Br_surge = np.asarray(vesselABC['Br'][0])
#         # self._Cr_surge = np.asarray(vesselABC['Cr'][0])
#         # self._xr_surge = np.zeros(self._Ar_surge.shape[0])
#         # self._dot_xr_sway = self._xr_surge

#         # # Fluid memory sway (sway-roll-yaw coupling)
#         # self._Ar_sway = [
#         #     np.asarray(vesselABC['Ar'][7]),
#         #     np.asarray(vesselABC['Ar'][9]),
#         #     np.asarray(vesselABC['Ar'][11])
#         # ]
#         # self._Br_sway = [np.asarray(vesselABC['Br'][i]) for i in [7, 9, 11]]
#         # self._Cr_sway = [np.asarray(vesselABC['Cr'][i]) for i in [7, 9, 11]]
#         # self._xr_sway = [np.zeros(self._Ar_sway[i].shape[0]) for i in range(3)]
#         # self._dot_xr_sway = self._xr_sway

#         # # Fluid memory heave (heave-pich coupling)
#         # self._Ar_heave = [
#         #     np.asarray(vesselABC['Ar'][14]),
#         #     np.asarray(vesselABC['Ar'][16]),
#         # ]
#         # self._Br_heave = [np.asarray(vesselABC['Br'][i]) for i in [14, 16]]
#         # self._Cr_heave = [np.asarray(vesselABC['Cr'][i]) for i in [14, 16]]
#         # self._xr_heave = [np.zeros(self._Ar_heave[i].shape[0]) for i in range(2)]
#         # self._dot_xr_heave = self._xr_heave

#         # # Fluid memory roll (sway-roll-yaw coupling)
#         # self._Ar_roll = [
#         #     np.asarray(vesselABC['Ar'][19]),
#         #     np.asarray(vesselABC['Ar'][21]),
#         #     np.asarray(vesselABC['Ar'][23])
#         # ]
#         # self._Br_roll = [np.asarray(vesselABC['Br'][i]) for i in [19, 21, 23]]
#         # self._Cr_roll = [np.asarray(vesselABC['Cr'][i]) for i in [19, 21, 23]]
#         # self._xr_roll = [np.zeros(self._Ar_roll[i].shape[0]) for i in range(3)]
#         # self._dot_xr_roll = self._xr_roll

#         # # Fluid memory pitch (heave-pitch coupling)
#         # self._Ar_pitch = [
#         #     np.asarray(vesselABC['Ar'][26]),
#         #     np.asarray(vesselABC['Ar'][28]),
#         # ]
#         # self._Br_pitch = [np.asarray(vesselABC['Br'][i]) for i in [26, 28]]
#         # self._Cr_pitch = [np.asarray(vesselABC['Cr'][i]) for i in [26, 28]]
#         # self._xr_pitch = [np.zeros(self._Ar_pitch[i].shape[0]) for i in range(2)]
#         # self._dot_xr_pitch = self._xr_pitch

#         # # Fluid memory yaw (sway-roll-yaw coupling)
#         # self._Ar_yaw = [
#         #     np.asarray(vesselABC['Ar'][31]),
#         #     np.asarray(vesselABC['Ar'][33]),
#         #     np.asarray(vesselABC['Ar'][35])
#         # ]
#         # self._Br_yaw = [np.asarray(vesselABC['Br'][i]) for i in [31, 33, 35]]
#         # self._Cr_yaw = [np.asarray(vesselABC['Cr'][i]) for i in [31, 33, 35]]
#         # self._xr_yaw = [np.zeros(self._Ar_yaw[i].shape[0]) for i in range(3)]
#         # self._dot_xr_yaw = self._xr_yaw


#     def x_dot(self, Uc, betac, tau):
#         nu_cn = Uc*np.array([np.cos(betac), np.sin(betac), 0, 0, 0, 0])
#         Jinv = np.linalg.inv(J(self._eta))
#         nu_c = Jinv@nu_cn

#         nu_r = self._nu - nu_c

#         self._eta_dot = J(self._eta)@self._nu
#         self._nu_dot = self._Minv@(tau - self._D@self._nu - self._G@Jinv@self._eta)
#         self._x_dot = np.concatenate([self._eta_dot, self._nu_dot])


#     @timeit
#     def memory_effects(self, nu):
#         # For surge: velocities = nu[0]
#         # For sway, roll and yaw: velocities = nu[1], nu[3], nu[5]
#         # For heave and pitch: velocities = nu[2], nu[4]
#         print(nu)
#         u1 = nu[0]
#         u2 = nu[[1, 3, 5]]
#         print(u2)
#         u3 = nu[[2, 4]]
#         u4 = u2
#         u5 = u3
#         u6 = u2

#         mew = np.zeros(self._dof)
#         # Surge
#         self._dot_xr_surge = self._Ar_surge@self._xr_surge + self._Br_surge*u1
#         self._xr_surge += self._dt * self._dot_xr_surge
#         mew[0] = self._Cr_surge@self._xr_surge

#         # Sway
#         for i in range(len(self._Ar_sway)):
#             self._dot_xr_sway[i] = self._Ar_sway[i]@self._xr_sway[i] + self._Br_sway[i]*u2[i]
#             self._xr_sway[i] += self._dt * self._dot_xr_sway[i]
#             mew[1] += self._Cr_sway[i]@self._xr_sway[i]

#         # Heave
#         for i in range(len(self._Ar_heave)):
#             self._dot_xr_heave[i] = self._Ar_heave[i]@self._xr_heave[i] + self._Br_heave[i]*u3[i]
#             self._xr_heave[i] += self._dt * self._dot_xr_heave[i]
#             mew[2] += self._Cr_heave[i]@self._xr_heave[i]

#         # Roll
#         for i in range(len(self._Ar_roll)):
#             self._dot_xr_roll[i] = self._Ar_roll[i]@self._xr_roll[i] + self._Br_roll[i]*u4[i]
#             self._xr_roll[i] += self._dt * self._dot_xr_roll[i]
#             mew[3] += self._Cr_roll[i]@self._xr_roll[i]

#         # Pitch
#         for i in range(len(self._Ar_pitch)):
#             self._dot_xr_pitch[i] = self._Ar_pitch[i]@self._xr_pitch[i] + self._Br_pitch[i]*u5[i]
#             self._xr_pitch[i] += self._dt * self._dot_xr_pitch[i]
#             mew[4] += self._Cr_pitch[i]@self._xr_pitch[i]

#         # Yaw
#         for i in range(len(self._Ar_yaw)):
#             self._dot_xr_yaw[i] = self._Ar_yaw[i]@self._xr_yaw[i] + self._Br_yaw[i]*u6[i]
#             self._xr_yaw[i] += self._dt * self._dot_xr_yaw[i]
#             mew[5] += self._Cr_yaw[i]@self._xr_yaw[i]

#         return mew
