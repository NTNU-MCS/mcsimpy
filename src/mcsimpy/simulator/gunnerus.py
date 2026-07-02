# gunnerus.py

# ----------------------------------------------------------------------------
# This code is part of the mcsimpy toolbox and repository.
# Created By: Jan-Erik Hygen
# Created Date: 2022-11-02
# Revised: 2023-02-13 Jan-Erik Hygen    Add 6DOF DP model for RVG
#
# Copyright (C) 2023: NTNU, Trondheim
# Licensed under GPL-3.0-or-later
# ---------------------------------------------------------------------------

from mcsimpy.simulator.vessel import Vessel3dof, Vessel6dof
from mcsimpy.utils import Rz, Smat

import numpy as np
import pickle
import json

import os

GUNNERUS_DATA = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "vessel_data", "gunnerus")
)


class GunnerusManeuvering3DoF(Vessel3dof):
    """3DOF Manuevering model for R/V Gunnerus. The model is based on maneuvering theory.
    Zero-Frequency model.

    References
    ----------
    Fossen 20--. Handbook of marine craft hydrodynamics and motion control
    """

    def __init__(self, dt, *args, config_file="parV_RVG3DOF.pkl", **kwargs):
        self._config = os.path.join(GUNNERUS_DATA, config_file)
        super().__init__(dt, config_file=config_file)
        with open(self._config, "rb") as f:
            self.data = pickle.load(f)
        self._dt = dt
        self._dof = 3
        self._Mrb = self.data["Mrb"]
        self._Ma = self.data["Ma"]
        self._Minv = np.linalg.inv(self._Mrb + self._Ma)
        self._D = np.zeros((3, 3))
        self._Dl = self.data["Dl"]
        self._Du = self.data["Du"]
        self._Dv = self.data["Dv"]
        self._Dr = self.data["Dr"]
        self._ref_vel = self.data["reference_velocity"]
        self._eta = np.zeros(3)
        self._nu = np.zeros(3)
        self._x = np.zeros(6)

    def x_dot(self, x, U_c, beta_c, tau):
        eta = x[: self._dof]
        nu = x[self._dof :]
        nu_c_n = U_c * np.array(
            [np.cos(beta_c), np.sin(beta_c), 0]
        )  # Current in NED-frame
        nu_c = Rz(eta[2]).T @ nu_c_n  # Current in body-frame
        S = Smat([0, 0, nu[2]])
        dnu_c = (S @ Rz(eta[2])).T @ nu_c_n
        nu_r = nu - nu_c

        self._D = (
            self._Dl
            + self._Du * np.abs(nu_r[0])
            + self._Dv * np.abs(nu_r[1])
            + self._Dr * np.abs(nu[2])
        )
        self._Ca = self.Cor3(nu_r, self._Ma)
        self._Crb = self.Cor3(nu, self._Mrb)

        eta_dot = Rz(eta[2]) @ self._nu
        nu_dot = self._Minv @ (
            tau - self._D @ nu_r - self._Crb @ nu - self._Ca @ nu_r + self._Ma @ dnu_c
        )
        return np.concatenate([eta_dot, nu_dot])
        # return eta_dot, nu_dot


class RVG_DP_6DOF(Vessel6dof):
    """6 Degree of Freedom simulation model of R/V Gunnerus.

    The model is created simply by using Veres data (ShipX).
    """

    _use_current_inertia = True

    def __init__(self, dt, config_file="vessel_2.json"):
        config_file = os.path.join(GUNNERUS_DATA, config_file)
        super().__init__(dt, config_file=config_file)
        with open(config_file, "r") as f:
            data = json.load(f)

        # Mass
        self._Mrb = np.asarray(data["MRB"])
        self._Ma = np.asarray(data["A"])[:, :, 30, 0]
        self._M = self._Mrb + self._Ma
        self._Minv = np.linalg.inv(self._M)

        # Damping
        self._Dp = np.asarray(data["B"])[:, :, 30, 0]
        self._Dv = np.asarray(data["Bv"])
        self._D = self._Dp + self._Dv

        # Restoring coefficients
        self._G = np.asarray(data["C"])[:, :, 0, 0]

    def _extract_hydrod_coeffs(self, param, freq_indx):
        all_dof = np.arange(6)
        Ma = np.asarray(param["A"])[:, all_dof, freq_indx, 0]
        Dp = np.asarray(param["B"])[:, all_dof, freq_indx, 0]
        return Ma, Dp
