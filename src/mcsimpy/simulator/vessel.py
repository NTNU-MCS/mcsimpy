# vessel.py

# ----------------------------------------------------------------------------
# This code is part of the mcsimpy toolbox and repository.
# Created By: Jan-Erik Hygen
# Created Date: 2022-10-31
# Revised: 2023-01-23 Author    added RK4 integration
#          2026-07-01 Author    introduced Vessel3dof/Vessel4dof/Vessel6dof
#
# Tested:  2023-01-23 Jan-Erik Hygen Test that RK4 integration works properly
#
# Copyright (C) 2022: NTNU, Trondheim
# Licensed under GPL-3.0-or-later
# ---------------------------------------------------------------------------

import json
from enum import IntEnum

import numpy as np
from abc import ABC, abstractmethod

from mcsimpy.utils import J, Rz, Smat, pipi


class DOF(IntEnum):
    """Degree of freedom indices, in the standard 6DOF ordering
    (surge, sway, heave, roll, pitch, yaw)."""

    SURGE = 0
    SWAY = 1
    HEAVE = 2
    ROLL = 3
    PITCH = 4
    YAW = 5


ANGULAR_DOFS = (DOF.ROLL, DOF.PITCH, DOF.YAW)


class Vessel(ABC):
    """Base class for simulator vessels."""

    def __init__(self, dt, config_file, *args, dof=6, **kwargs):
        self._config_file = config_file
        self._dof = dof
        self._dt = dt
        self._M = np.zeros((dof, dof))
        self._D = np.zeros_like(self._M)
        self._G = np.zeros_like(self._D)
        self._eta = np.zeros(dof)
        self._nu = np.zeros(dof)
        self._x = np.zeros(2 * dof)
        self._x_dot = np.zeros(2 * dof)

        dofs = getattr(self, "DOFS", tuple(DOF)[:dof])
        self._dof_indices = np.array([d.value for d in dofs])
        self._angular_local_idx = np.array(
            [i for i, d in enumerate(dofs) if d in ANGULAR_DOFS]
        )

    def __call__(self, *args, **kwargs):
        pass

    def set_eta(self, eta):
        """Set the pose of the vessel."""
        if eta.shape != self._eta.shape:
            raise ValueError(
                f"{eta.shape} does not correspond to the DOF. DOF = {self._dof}"
            )
        self._eta = eta
        self._x[: self._dof] = self._eta

    def set_nu(self, nu):
        if nu.shape != self._nu.shape:
            raise ValueError(
                f"{nu.shape} does not correspond to the DOF. DOF = {self._dof}"
            )
        self._nu = nu
        self._x[self._dof : 2 * self._dof] = self._nu

    def get_eta(self):
        """Get vessel pose eta.

        Returns
        -------
        self._eta : 6 x 1 array.
        """
        return self._eta

    def get_nu(self):
        """Get vessel velocity nu.

        Returns
        -------
        self._nu : 6 x 1 array.
        """
        return self._nu

    def get_x(self):
        """Get vessel state vector x.

        Returns
        -------
        self._x : DOF x 1 array.
        """
        return self._x

    def reset(self):
        """Reset state vector to zeroes."""
        self._x = np.zeros(2 * self._dof)
        self._x_dot = np.zeros_like(self._x)
        self._eta = np.zeros(self._dof)
        self._nu = np.zeros(self._dof)

    @abstractmethod
    def x_dot(self, Uc, beta_c, tau):
        """Kinematic and kinetic equation of vessel. The method must be overwritten
        by inherting vessel classes. The method should return the result of f(x, u, ..).
        It should not modify any of the object attributes.

        Parameters
        ----------
        Uc : float
            Current velocity
        beta_c : float
            Current direction in NED frame [rad]
        tau : array_like
            Sum of all loads corresponding to vessel DOF.

        Returns
        -------
        array_like
            Time derivative of the state vector.
        """
        raise NotImplementedError

    def integrate(self, Uc, beta_c, tau):
        """Integrate the state vector one forward, using RK4 integration.

        Parameters
        ----------
        Uc : float
            Current velocity
        beta_c : float
            Current direction in NED frame [rad]
        tau : array_like
            Sum of all loads corresponding to vessel DOF.
        """
        x = self.get_x()
        self._x = self.integrator(
            x, Uc, beta_c, tau
        )  # Compute new state vector through integration
        self._eta = self._x[: self._dof]  # Set eta
        self._eta[self._angular_local_idx] = pipi(
            self._eta[self._angular_local_idx]
        )  # Keep radians in (-pi, pi)
        self._nu = self._x[self._dof :]  # Set nu
        self._x = np.concatenate([self._eta, self._nu])

    def integrator(self, x, Uc, beta_c, tau):
        """Runge-Kutta 4 integration method."""
        k1 = self.x_dot(x, Uc, beta_c, tau)
        k2 = self.x_dot(x + k1 * self._dt / 2, Uc, beta_c, tau)
        k3 = self.x_dot(x + k2 * self._dt / 2, Uc, beta_c, tau)
        k4 = self.x_dot(x + k3 * self._dt, Uc, beta_c, tau)

        return self._x + (k1 + 2 * k2 + 2 * k3 + k4) * self._dt / 6


class Vessel3dof(Vessel):
    """Base class for 3DOF vessel simulation models (surge, sway, yaw)."""

    DOFS = (DOF.SURGE, DOF.SWAY, DOF.YAW)

    def __init__(self, dt, config_file, *args, **kwargs):
        super().__init__(dt, config_file, dof=len(self.DOFS), **kwargs)

    @staticmethod
    def Cor3(nu, M):
        """Generic 3DOF Coriolis-centripetal matrix for a mass matrix `M`
        evaluated at velocity `nu`."""
        return np.array(
            [
                [0, 0, -M[1, 1] * nu[1] - 0.5 * (M[1, 2] + M[2, 1]) * nu[2]],
                [0, 0, M[0, 0] * nu[0]],
                [
                    M[1, 1] * nu[1] + 0.5 * (M[1, 2] + M[2, 1]) * nu[2],
                    -M[0, 0] * nu[0],
                    0,
                ],
            ]
        )

    def _load_hydrod_parameters(self, config_file, freq_indx=-1):
        """Load MRB, added mass, and potential damping from a frequency-domain
        vessel json file (as generated by the MATLAB MSS toolbox's `vessel2ss`
        and converted to json), projected onto this vessel's DOF subset.

        Sets `self._Mrb`, `self._Ma`, `self._M`, `self._Minv`, `self._D`
        (potential damping only - viscous damping is not summed in, since
        maneuvering models typically apply it separately as a function of
        relative velocity) and `self._Dv`.
        """
        with open(config_file, "r") as f:
            param = json.load(f)
        idx = np.ix_(self._dof_indices, self._dof_indices)
        Bv = np.asarray(param["Bv"])
        if Bv.ndim == 3:
            Bv = Bv[:, :, freq_indx]
        self._Mrb = np.asarray(param["MRB"])[idx]
        self._Ma = np.asarray(param["A"])[:, :, freq_indx][idx]
        self._M = self._Mrb + self._Ma
        self._Minv = np.linalg.inv(self._M)
        self._D = np.asarray(param["B"])[:, :, freq_indx][idx]
        self._Dv = Bv[idx]


class Vessel4dof(Vessel):
    """Base class for 4DOF vessel simulation models (surge, sway, roll, yaw).

    No concrete vessel model uses this yet - it exists as scaffolding for
    future 4DOF models.
    """

    DOFS = (DOF.SURGE, DOF.SWAY, DOF.ROLL, DOF.YAW)

    def __init__(self, dt, config_file, *args, **kwargs):
        super().__init__(dt, config_file, dof=len(self.DOFS), **kwargs)


class Vessel6dof(Vessel):
    """Base class for 6DOF vessel simulation models.

    Implements the shared kinematics/kinetics used by DP/seakeeping-style
    6DOF models: `eta_dot = J(eta) @ nu` and
    `nu_dot = Minv @ (tau - D @ nu_r - G @ eta [+ Ma @ dnu_c])`. The
    current-induced added-mass term is only included when
    `_use_current_inertia` is set by the subclass.
    """

    DOFS = tuple(DOF)
    _use_current_inertia = False

    def __init__(self, dt, config_file, *args, **kwargs):
        super().__init__(dt, config_file, dof=len(self.DOFS), **kwargs)

    def x_dot(self, x, Uc, betac, tau):
        """Kinematic and kinetic equations.

        Parameters
        ----------
        x : array_like
            State vector with dimensions 12x1
        Uc : float
            Current velocity in earth-fixed frame
        betac : float
            Current direction in earth-fixed frame [rad]
        tau : array_like
            External loads (e.g wind, thrusters, ice, etc). Must be a 6x1 vector.

        Returns
        -------
        x_dot : array_like
            The derivative of the state vector.
        """
        eta = x[: self._dof]
        nu = x[self._dof :]

        nu_cn = Uc * np.array([np.cos(betac), np.sin(betac), 0])
        nu_c = Rz(eta[-1]).T @ nu_cn
        nu_c = np.insert(nu_c, [3, 3, 3], 0)
        nu_r = nu - nu_c

        eta_dot = J(eta) @ nu

        tau_current = 0.0
        if self._use_current_inertia:
            dnu_cb = -Smat([0.0, 0.0, nu[-1]]) @ Rz(eta[-1]).T @ nu_cn
            dnu_cb = np.insert(dnu_cb, [2, 2, 2], 0)
            tau_current = self._Ma @ dnu_cb

        nu_dot = self._Minv @ (
            tau - self._D @ nu_r - self._G @ eta + tau_current
        )
        return np.concatenate([eta_dot, nu_dot])

    def set_hydrod_parameters(self, freq):
        """Set the hydrodynamic added mass and damping for a given frequency.

        Parameters
        ----------
        freq : array_like
            Frequency in rad/s. Can either be a single frequency, or
            multiple frequencies with dimension n = DOF.

        Examples
        --------

        Set a hydrodynamic parameters for one frequency

        >>> dt = 0.01
        >>> model = CSAD_DP_6DOF(dt)
        >>> frequency = 2*np.pi
        >>> model.set_hydrod_parameters(frequency)

        Set frequency for individual components

        >>> freqs = [0., 0., 2*np.pi, 2*np.pi, 2*np.pi, 0.]
        >>> model.set_hydrod_parameters(freqs)
        """

        if type(freq) not in [list, np.ndarray]:
            freq = [freq]
        freq = np.asarray(freq)
        if (freq.shape[0] > 1) and (freq.shape[0] != self._dof):
            raise ValueError(
                f"Argument freq: {freq} must either be a float or have shape n = {self._dof}. \
                             freq.shape = {freq.shape} != {self._dof}."
            )
        with open(self._config_file, "r") as f:
            param = json.load(f)

        freqs = np.asarray(param["freqs"])
        if freq.shape[0] == 1:
            freq_indx = np.argmin(np.abs(freqs - freq))
        else:
            freq_indx = np.argmin(np.abs(freqs - freq[:, None]), axis=1)

        self._Ma, self._Dp = self._extract_hydrod_coeffs(param, freq_indx)
        self._Dv = self._extract_viscous_damping(param, freq_indx)
        self._M = self._Mrb + self._Ma
        self._Minv = np.linalg.inv(self._M)
        self._D = self._Dv + self._Dp

    def _extract_hydrod_coeffs(self, param, freq_indx):
        """Pull the added mass and potential damping matrices for the given
        frequency index out of the raw config data. Overridden by subclasses
        whose data files carry an extra trailing axis."""
        all_dof = np.arange(6)
        Ma = np.asarray(param["A"])[:, all_dof, freq_indx]
        Dp = np.asarray(param["B"])[:, all_dof, freq_indx]
        return Ma, Dp

    def _extract_viscous_damping(self, param, freq_indx):
        """Pull the viscous damping matrix out of the raw config data. Most
        data files store `Bv` as frequency-independent (n x n); some (e.g.
        WAMIT-derived data) store it per-frequency like `A`/`B`, in which case
        it is indexed the same way."""
        Bv = np.asarray(param["Bv"])
        if Bv.ndim == 2:
            return Bv
        all_dof = np.arange(6)
        return Bv[:, all_dof, freq_indx]

    def _extract_restoring_coeffs(self, param):
        """Pull the restoring coefficient matrix out of the raw config data.
        Overridden by subclasses whose data files carry an extra trailing
        axis."""
        return np.asarray(param["C"])[:, :, 0]

    def _load_hydrod_parameters(self, config_file, freq_indx):
        """Load MRB, added mass, damping, and restoring coefficients from a
        frequency-domain vessel json file (as generated by the MATLAB MSS
        toolbox's `vessel2ss` and converted to json). `freq_indx` selects the
        initial frequency column of A/B (and Bv, when it carries a frequency
        axis); it can be changed later via `set_hydrod_parameters`.
        """
        with open(config_file, "r") as f:
            param = json.load(f)
        self._Mrb = np.asarray(param["MRB"])
        self._Ma, self._Dp = self._extract_hydrod_coeffs(param, freq_indx)
        self._Dv = self._extract_viscous_damping(param, freq_indx)
        self._G = self._extract_restoring_coeffs(param)
        self._M = self._Mrb + self._Ma
        self._Minv = np.linalg.inv(self._M)
        self._D = self._Dp + self._Dv

    def _load_hydrod_abc(self, config_file):
        """Load MRB, infinite-frequency added mass, and restoring coefficients
        from a vessel ABC json file (the state-space realization companion to
        the frequency-domain json, generated by `vessel2ss`).
        """
        with open(config_file, "r") as f:
            param = json.load(f)
        self._Mrb = np.asarray(param["MRB"])
        self._Ma = np.asarray(param["MA"])
        self._M = self._Mrb + self._Ma
        self._Minv = np.linalg.inv(self._M)
        self._G = np.asarray(param["G"])
