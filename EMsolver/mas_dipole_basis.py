"""Continuity-consistent electric-dipole basis for MAS auxiliary sources.

Step 2-A intentionally keeps this basis independent of the production MAS
matrix solve.  Each auxiliary degree of freedom is interpreted as an electric
point-dipole moment p(t) [C m] with current moment m(t)=dp/dt [A m].  In the
sense of distributions,

    rho(r,t) = -p(t) . grad delta(r-r0),
    J(r,t)   =  m(t) delta(r-r0),

so d rho/dt + div J = 0 identically.  The class below provides the discrete
history bookkeeping used to verify that property before Step 2-B connects the
basis to the MAS impedance operator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np

_EPS = 1.0e-300


@dataclass
class DipoleTemporalAudit:
    step: int
    increment_residual_max: float
    increment_residual_rms: float
    dipole_moment_rms: float
    current_moment_rms: float


class ContinuityConsistentElectricDipoleBasis:
    """Three-vector electric-dipole basis attached to MAS source centers.

    Parameters
    ----------
    centers:
        Source centers with shape (N,3), in SI metres for the Step-2A tests.
    dt:
        Time step in seconds.
    max_time_steps:
        Circular-history capacity.
    initial_dipole_moment:
        Optional p(t=0), shape (N,3), in C m.

    Notes
    -----
    The stored *current moment* is m=dp/dt, not a point current density.  This
    distinction removes the ambiguous source-volume factor in the legacy
    point-current representation.  Three Cartesian components per MAS point
    are retained, so the basis still has 3N degrees of freedom.
    """

    def __init__(
        self,
        centers: np.ndarray,
        dt: float,
        max_time_steps: int,
        initial_dipole_moment: Optional[np.ndarray] = None,
    ):
        self.centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        self.dt = float(dt)
        self.max_time_steps = int(max_time_steps)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.max_time_steps < 2:
            raise ValueError("max_time_steps must be at least 2")
        n = len(self.centers)
        p0 = np.zeros((n, 3), dtype=np.float64) if initial_dipole_moment is None else np.asarray(
            initial_dipole_moment, dtype=np.float64
        ).reshape(n, 3)
        self.dipole_moment_history = np.zeros((self.max_time_steps, n, 3), dtype=np.float64)
        self.current_moment_history = np.zeros((self.max_time_steps, n, 3), dtype=np.float64)
        self.history_abs_steps = -np.ones(self.max_time_steps, dtype=np.int64)
        self.initial_dipole_moment = p0.copy()
        self.last_dipole_moment = p0.copy()
        self.last_current_moment = np.zeros((n, 3), dtype=np.float64)

    def _slot(self, step: int) -> int:
        return int(step) % self.max_time_steps

    def record_initial_current_moment(self, current_moment: np.ndarray) -> np.ndarray:
        """Record m^0 while keeping the supplied initial p^0."""
        m0 = np.asarray(current_moment, dtype=np.float64).reshape(len(self.centers), 3)
        slot = self._slot(0)
        self.dipole_moment_history[slot] = self.initial_dipole_moment
        self.current_moment_history[slot] = m0
        self.history_abs_steps[slot] = 0
        self.last_dipole_moment = self.initial_dipole_moment.copy()
        self.last_current_moment = m0.copy()
        return self.last_dipole_moment.copy()

    def advance_from_current_moment(self, step: int, current_moment: np.ndarray) -> np.ndarray:
        """Advance p with trapezoidal integration of m=dp/dt.

        For step n>=1,

            p^n = p^(n-1) + dt/2 * (m^(n-1) + m^n).

        Therefore the interval-averaged discrete continuity relation is exact:

            (p^n-p^(n-1))/dt = (m^(n-1)+m^n)/2.
        """
        step = int(step)
        if step <= 0:
            raise ValueError("advance_from_current_moment expects step >= 1")
        prev_slot = self._slot(step - 1)
        if self.history_abs_steps[prev_slot] != step - 1:
            raise RuntimeError("Previous dipole/current-moment history is unavailable")
        m = np.asarray(current_moment, dtype=np.float64).reshape(len(self.centers), 3)
        p_prev = self.dipole_moment_history[prev_slot]
        m_prev = self.current_moment_history[prev_slot]
        p = p_prev + 0.5 * self.dt * (m_prev + m)
        slot = self._slot(step)
        self.dipole_moment_history[slot] = p
        self.current_moment_history[slot] = m
        self.history_abs_steps[slot] = step
        self.last_dipole_moment = p.copy()
        self.last_current_moment = m.copy()
        return p.copy()

    def temporal_continuity_audit(self, step: int) -> DipoleTemporalAudit:
        """Audit the exact discrete identity used by the temporal basis."""
        step = int(step)
        if step <= 0:
            raise ValueError("step must be >= 1")
        s0 = self._slot(step - 1)
        s1 = self._slot(step)
        if self.history_abs_steps[s0] != step - 1 or self.history_abs_steps[s1] != step:
            raise RuntimeError("Requested history interval is unavailable")
        p0 = self.dipole_moment_history[s0]
        p1 = self.dipole_moment_history[s1]
        m0 = self.current_moment_history[s0]
        m1 = self.current_moment_history[s1]
        residual = (p1 - p0) / self.dt - 0.5 * (m0 + m1)
        norms = np.linalg.norm(residual, axis=1)
        return DipoleTemporalAudit(
            step=step,
            increment_residual_max=float(np.max(norms)) if len(norms) else 0.0,
            increment_residual_rms=float(np.sqrt(np.mean(norms * norms))) if len(norms) else 0.0,
            dipole_moment_rms=float(np.sqrt(np.mean(np.sum(p1 * p1, axis=1)))) if len(p1) else 0.0,
            current_moment_rms=float(np.sqrt(np.mean(np.sum(m1 * m1, axis=1)))) if len(m1) else 0.0,
        )

    @staticmethod
    def weak_continuity_residual(
        dipole_moment_rate: np.ndarray,
        test_function_gradients: np.ndarray,
    ) -> np.ndarray:
        """Distributional continuity residual for rho=-p.grad(delta), J=pdot*delta.

        For any smooth scalar test function phi,

            <d rho/dt, phi> =  pdot . grad(phi)(r0),
            <div J, phi>     = -pdot . grad(phi)(r0),

        hence the returned residual is identically zero up to roundoff.
        """
        pdot = np.asarray(dipole_moment_rate, dtype=np.float64).reshape(-1, 3)
        grad = np.asarray(test_function_gradients, dtype=np.float64).reshape(-1, 3)
        if len(pdot) != len(grad):
            raise ValueError("dipole_moment_rate and test_function_gradients must have the same length")
        drho_dt_pairing = np.einsum("ij,ij->i", pdot, grad)
        div_j_pairing = -np.einsum("ij,ij->i", pdot, grad)
        return drho_dt_pairing + div_j_pairing


def electric_point_dipole_fields(
    observation_points: np.ndarray,
    center: np.ndarray,
    t: float,
    p_of_t: Callable[[np.ndarray], np.ndarray],
    pdot_of_t: Callable[[np.ndarray], np.ndarray],
    pddot_of_t: Callable[[np.ndarray], np.ndarray],
    *,
    epsilon0: float = 8.8541878128e-12,
    c: float = 299792458.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact retarded fields of an ideal electric point dipole.

    The callback functions accept an array of retarded times and return arrays
    with shape (...,3), containing p, dp/dt and d2p/dt2 in SI units.
    """
    obs = np.asarray(observation_points, dtype=np.float64).reshape(-1, 3)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    rvec = obs - center[None, :]
    R = np.linalg.norm(rvec, axis=1)
    if np.any(R <= 0.0):
        raise ValueError("observation point coincides with dipole center")
    n = rvec / R[:, None]
    tr = float(t) - R / float(c)
    p = np.asarray(p_of_t(tr), dtype=np.float64).reshape(-1, 3)
    pd = np.asarray(pdot_of_t(tr), dtype=np.float64).reshape(-1, 3)
    pdd = np.asarray(pddot_of_t(tr), dtype=np.float64).reshape(-1, 3)

    ndotp = np.einsum("ij,ij->i", n, p)[:, None]
    ndotpd = np.einsum("ij,ij->i", n, pd)[:, None]
    near = (3.0 * n * ndotp - p) / R[:, None] ** 3
    induction = (3.0 * n * ndotpd - pd) / (float(c) * R[:, None] ** 2)
    radiation = np.cross(n, np.cross(n, pdd)) / (float(c) ** 2 * R[:, None])
    coeff_e = 1.0 / (4.0 * math.pi * float(epsilon0))
    E = coeff_e * (near + induction + radiation)

    mu0 = 1.0 / (float(epsilon0) * float(c) ** 2)
    B = (mu0 / (4.0 * math.pi)) * (
        np.cross(pd, n) / R[:, None] ** 2
        + np.cross(pdd, n) / (float(c) * R[:, None])
    )
    return E, B


def finite_segment_jefimenko_fields(
    observation_points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    length: float,
    t: float,
    p_scalar: Callable[[np.ndarray], np.ndarray],
    pdot_scalar: Callable[[np.ndarray], np.ndarray],
    pddot_scalar: Callable[[np.ndarray], np.ndarray],
    *,
    epsilon0: float = 8.8541878128e-12,
    c: float = 299792458.0,
    quadrature_order: int = 96,
) -> Tuple[np.ndarray, np.ndarray]:
    """Independent finite current-segment + endpoint-charge Jefimenko reference.

    A line segment of length ``length`` carries current I=dp/dt/length along
    ``axis``. Endpoint charges are Q_plus=+p/length and Q_minus=-p/length.
    This finite source satisfies charge continuity exactly.  As length/R -> 0,
    its field converges to the ideal point-electric-dipole field without fitted
    constants.
    """
    obs = np.asarray(observation_points, dtype=np.float64).reshape(-1, 3)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    anorm = float(np.linalg.norm(axis))
    if anorm <= 0.0:
        raise ValueError("axis must be non-zero")
    axis = axis / anorm
    length = float(length)
    if length <= 0.0:
        raise ValueError("length must be positive")
    c = float(c)
    eps0 = float(epsilon0)
    coeff_e = 1.0 / (4.0 * math.pi * eps0)
    mu0 = 1.0 / (eps0 * c * c)
    coeff_b = mu0 / (4.0 * math.pi)

    E = np.zeros((len(obs), 3), dtype=np.float64)
    B = np.zeros((len(obs), 3), dtype=np.float64)

    # Endpoint-charge contributions.
    for sign in (-1.0, 1.0):
        src = center + sign * 0.5 * length * axis
        rv = obs - src[None, :]
        R = np.linalg.norm(rv, axis=1)
        if np.any(R <= 0.0):
            raise ValueError("observation point coincides with segment endpoint")
        n = rv / R[:, None]
        tr = float(t) - R / c
        q = sign * np.asarray(p_scalar(tr), dtype=np.float64).reshape(-1) / length
        qdot = sign * np.asarray(pdot_scalar(tr), dtype=np.float64).reshape(-1) / length
        E += coeff_e * n * (q / (R * R) + qdot / (c * R))[:, None]

    # Line-current contributions, integrated with Gauss-Legendre quadrature.
    xg, wg = np.polynomial.legendre.leggauss(max(8, int(quadrature_order)))
    s = 0.5 * length * xg
    ds_weights = 0.5 * length * wg
    for ss, wds in zip(s, ds_weights):
        src = center + ss * axis
        rv = obs - src[None, :]
        R = np.linalg.norm(rv, axis=1)
        n = rv / R[:, None]
        tr = float(t) - R / c
        I = np.asarray(pdot_scalar(tr), dtype=np.float64).reshape(-1) / length
        Idot = np.asarray(pddot_scalar(tr), dtype=np.float64).reshape(-1) / length
        Jvec = I[:, None] * axis[None, :]
        dJvec = Idot[:, None] * axis[None, :]
        E += -coeff_e * dJvec * (wds / (c * c * R))[:, None]
        B += coeff_b * (
            np.cross(Jvec, n) / R[:, None] ** 2
            + np.cross(dJvec, n) / (c * R[:, None])
        ) * wds
    return E, B


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), _EPS))
