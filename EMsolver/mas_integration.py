"""
Self-consistent plasma / PEC-antenna coupling through MAS.

The class in this module is the intended replacement for the previous TDIE
surface-current coupling path.  It keeps the PIC side grid based while the metal
antenna is represented by analytic auxiliary point sources.
"""

from __future__ import annotations
from EMsolver.perf_diagnostics import guard_against_full_grid_cpu_mas

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence

import math
import numpy as np

from .mas_preprocessor import MASGeometry, assemble_impedance_matrix, preprocess_mas_antenna


_EPS = 1.0e-30
_TAPER_MODE_TO_INT = {None: 0, 'none': 0, 'hard': 0, 'hann': 1, 'hanning': 1, 'smoothstep': 2, 'exponential': 3, 'exp': 3}


def _taper_mode_to_int(mode) -> int:
    key = None if mode is None else str(mode).lower()
    if key not in _TAPER_MODE_TO_INT:
        raise ValueError(f"Unsupported taper mode: {mode}. Use one of: hard, none, hann, smoothstep, exponential.")
    return int(_TAPER_MODE_TO_INT[key])


def _spatial_taper_weights(R: np.ndarray, cutoff_radius: float, taper_width: float = 0.0, taper_type: int = 0) -> np.ndarray:
    """Vectorized equivalent of the CUDA spatial cutoff window."""
    R = np.asarray(R, dtype=np.float64)
    if cutoff_radius is None or float(cutoff_radius) <= 0.0:
        return np.ones_like(R, dtype=np.float64)
    cutoff_radius = float(cutoff_radius)
    w = np.where(R < cutoff_radius, 1.0, 0.0).astype(np.float64)
    taper_width = max(0.0, float(taper_width))
    taper_type = int(taper_type)
    if taper_width <= 0.0 or taper_type <= 0:
        return w
    inner = max(0.0, cutoff_radius - taper_width)
    denom = max(cutoff_radius - inner, _EPS)
    shell = (R > inner) & (R < cutoff_radius)
    u = np.clip((R - inner) / denom, 0.0, 1.0)
    if taper_type == 1:
        taper = 0.5 * (1.0 + np.cos(np.pi * u))
    elif taper_type == 2:
        taper = 1.0 - u * u * (3.0 - 2.0 * u)
    else:
        taper = np.exp(-u / np.maximum(1.0e-12, 1.0 - u))
    w = np.where(shell, taper, w)
    return w


@dataclass
class SinusoidalCurrentExcitation:
    """Preset active MAS surface-current density K(t) [A/m].

    The historical class name is retained for API compatibility.  Step 2-B
    interprets the returned vector as surface-current density K; multiplying by
    the MAS patch area gives the lumped electric current moment m [A m].
    """

    amplitude: float
    frequency: float
    direction: Sequence[float] = (0.0, 0.0, 1.0)
    phase: float = 0.0
    ramp_time: float = 0.0
    spatial_weights: Optional[np.ndarray] = None

    def evaluate(self, points: np.ndarray, t: float, step: int) -> np.ndarray:
        direction = np.asarray(self.direction, dtype=np.float64).reshape(3,)
        norm = float(np.linalg.norm(direction))
        if norm <= _EPS:
            raise ValueError("SinusoidalCurrentExcitation.direction must be non-zero")
        direction = direction / norm
        envelope = 1.0
        if self.ramp_time and self.ramp_time > 0.0:
            envelope = min(1.0, max(0.0, float(t) / float(self.ramp_time)))
        value = float(self.amplitude) * envelope * math.sin(2.0 * math.pi * float(self.frequency) * float(t) + float(self.phase))
        currents = value * np.ones((len(points), 1), dtype=np.float64) * direction[None, :]
        if self.spatial_weights is not None:
            weights = np.asarray(self.spatial_weights, dtype=np.float64).reshape(-1, 1)
            if len(weights) != len(points):
                raise ValueError("spatial_weights length must match number of MAS source points")
            currents = currents * weights
        return currents


def _as_numpy_field(field) -> np.ndarray:
    if hasattr(field, "get"):
        field = field.get()
    return np.asarray(field, dtype=np.float64)


class MASAntennaInteraction:
    """Main MAS feedback solver used inside the PIC time loop.

    The main loop is expected to do:

        1. push particles with the previous total field;
        2. deposit plasma rho/J onto the grid;
        3. call the grid Jefimenko solver to get the incident plasma field;
        4. call ``update_fields_with_device_interaction`` below;
        5. use the returned total field for the next particle push.
    """

    def __init__(self, geometry: MASGeometry, dt: float, epsilon0: float, c: float,
                 max_time_steps: int, grid_size, grid_spacing, grid_origin,
                 active_excitation: Optional[SinusoidalCurrentExcitation | Callable[[np.ndarray, float, int], np.ndarray]] = None,
                 source_softening: Optional[float] = None,
                 source_truncation_radius: Optional[float] = None,
                 source_taper: Optional[str] = None,
                 source_taper_width: Optional[float] = None,
                 surface_charge_ridge: float = 0.0):
        self.geometry = geometry
        self.dt = float(dt)
        self.epsilon0 = float(epsilon0)
        self.c = float(c)
        self.mu0 = 1.0 / (self.epsilon0 * self.c * self.c)
        self.max_time_steps = int(max_time_steps)
        if self.max_time_steps <= 0:
            raise ValueError("max_time_steps must be positive")

        self.grid_size = tuple(int(v) for v in grid_size)
        self.grid_spacing = tuple(float(v) for v in grid_spacing)
        self.grid_origin = tuple(float(v) for v in grid_origin)
        self.grid_points = self._get_grid_points()
        self.allow_full_grid_cpu_mas = False

        if (self.geometry.G is None or self.geometry.G_inv is None
                or getattr(self.geometry, "G_physical", None) is None
                or getattr(self.geometry, "G_p", None) is None
                or getattr(self.geometry, "G_kdot", None) is None):
            self.geometry = assemble_impedance_matrix(
                self.geometry, dt=self.dt, epsilon0=self.epsilon0, c=self.c,
                softening_distance=source_softening,
                spatial_solver=self.geometry.metadata.get("spatial_solver", "legacy_cartesian_square"),
                tikhonov_rel=self.geometry.metadata.get("tikhonov_rel", 1.0e-4),
                svd_rcond=self.geometry.metadata.get("svd_rcond", 1.0e-10),
            )

        self.active_excitation = active_excitation
        self.source_softening = self._resolve_source_softening(source_softening)
        self.surface_charge_ridge = max(0.0, float(surface_charge_ridge or 0.0))
        n = self.geometry.num_points
        # Step-2B continuity-consistent MAS temporal state.  K [A/m] is a
        # surface-current density coefficient; P [C/m] is its time integral.
        # Each source patch of area w therefore represents lumped moments
        # m=wK [A m] and p=wP [C m], exactly the Step-2A electric-dipole basis.
        self.antenna_J_history = np.zeros((self.max_time_steps, n, 3), dtype=np.float64)  # K
        self.active_J_history = np.zeros((self.max_time_steps, n, 3), dtype=np.float64)   # K_active
        self.passive_J_history = np.zeros((self.max_time_steps, n, 3), dtype=np.float64) # K_passive
        self.antenna_P_history = np.zeros((self.max_time_steps, n, 3), dtype=np.float64)
        self.active_P_history = np.zeros((self.max_time_steps, n, 3), dtype=np.float64)
        self.passive_P_history = np.zeros((self.max_time_steps, n, 3), dtype=np.float64)
        self.history_abs_steps = -np.ones(self.max_time_steps, dtype=np.int64)
        self.source_truncation_radius = 0.0 if source_truncation_radius is None else max(0.0, float(source_truncation_radius))
        self.source_taper_type = _taper_mode_to_int(source_taper)
        self.source_taper_width = 0.0 if source_taper_width is None else max(0.0, float(source_taper_width))
        self.time_step = 0
        self.time_basis_name = "piecewise_linear_nodal"
        self.last_field_data = None
        self.last_virtual_work = None
        self.last_surface_tangential_E = None
        self.last_active_current = np.zeros((n, 3), dtype=np.float64)
        self.last_passive_current = np.zeros((n, 3), dtype=np.float64)
        self.last_total_current = np.zeros((n, 3), dtype=np.float64)
        self.last_active_polarization = np.zeros((n, 3), dtype=np.float64)
        self.last_passive_polarization = np.zeros((n, 3), dtype=np.float64)
        self.last_total_polarization = np.zeros((n, 3), dtype=np.float64)
        self.last_source_closure_diagnostics = {
            'valid_history_derivative': False,
            'continuity_linked_rho_constructed': True,
            'source_model': 'rho=-p.grad(delta), J=pdot*delta with p=wP and pdot=wK',
            'local_operator_derivative_definition': 'Kdot^n=(K^n-K^(n-1))/dt',
            'pass2_history_derivative_definition': 'same piecewise-linear K derivative on the retarded interval',
        }

        # Surface-charge extension used by the floating-conductor sheath
        # benchmark.  Values are SI surface-charge densities [C/m^2].
        self.surface_charge_history = np.zeros((self.max_time_steps, n), dtype=np.float64)
        self.last_surface_charge = np.zeros(n, dtype=np.float64)
        self.last_surface_charge_info = {
            "net_charge_C": 0.0,
            "charge_constraint_error_C": 0.0,
            "floating_potential_V": 0.0,
            "potential_rms_residual_V": 0.0,
            "potential_max_residual_V": 0.0,
            "rms_tangential_E": 0.0,
            "max_tangential_E": 0.0,
            "plasma_potential_mean_V": 0.0,
            "self_potential_mean_V": 0.0,
            "self_potential_rms_variation_V": 0.0,
            "surface_charge_ridge": self.surface_charge_ridge,
            "surface_charge_regularization": "none" if self.surface_charge_ridge <= 0.0 else "scaled_patch_charge_ridge",
        }
        (self._surface_charge_E_matrix,
         self._surface_charge_E_tangent_matrix,
         self._surface_charge_potential_matrix) = self._assemble_surface_charge_operators()
        # Floating-conductor Case 2 now enforces the electrostatic conductor
        # condition directly: every surface collocation point is at one
        # unknown potential V_f, while the integrated conductor charge equals
        # the collected particle charge.  The fixed-geometry KKT inverse is
        # built once; time stepping updates only its right-hand side.
        self._prepare_floating_equipotential_solver()
        self._plasma_surface_potential_operator = None

    def _resolve_source_softening(self, source_softening: Optional[float]) -> float:
        if source_softening is not None:
            return max(float(source_softening), 0.0)
        offsets = np.linalg.norm(self.geometry.ant_obs_points - self.geometry.ant_src_points, axis=1)
        if len(offsets) == 0:
            return 0.0
        return 0.25 * float(np.median(offsets))

    def _get_grid_points(self) -> np.ndarray:
        nx, ny, nz = self.grid_size
        dx, dy, dz = self.grid_spacing
        x0, y0, z0 = self.grid_origin
        x = x0 + (np.arange(nx, dtype=np.float64) + 0.5) * dx
        y = y0 + (np.arange(ny, dtype=np.float64) + 0.5) * dy
        z = z0 + (np.arange(nz, dtype=np.float64) + 0.5) * dz
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    def _sample_vector_field_trilinear(self, field_3d: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Vectorized CPU trilinear sampler for diagnostics/fallback paths.

        Production GPU runs should normally use ``EMsolver.sample_current_E_at_points``
        so only the small surface-sample array is copied back to CPU.  This CPU
        version is kept for tests and non-GPU diagnostics, but avoids the old
        Python loop over observation points and the 8 neighboring cells.
        """
        original_shape = np.asarray(points).shape
        points_flat = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(points_flat) == 0:
            return np.zeros(tuple(original_shape[:-1]) + (3,), dtype=np.float64)

        nx, ny, nz = self.grid_size
        dx, dy, dz = self.grid_spacing
        x0, y0, z0 = self.grid_origin

        sx = (points_flat[:, 0] - x0) / dx - 0.5
        sy = (points_flat[:, 1] - y0) / dy - 0.5
        sz = (points_flat[:, 2] - z0) / dz - 0.5

        i0 = np.floor(sx).astype(np.int64)
        j0 = np.floor(sy).astype(np.int64)
        k0 = np.floor(sz).astype(np.int64)
        tx = sx - i0
        ty = sy - j0
        tz = sz - k0

        i1 = np.clip(i0 + 1, 0, nx - 1)
        j1 = np.clip(j0 + 1, 0, ny - 1)
        k1 = np.clip(k0 + 1, 0, nz - 1)
        i0 = np.clip(i0, 0, nx - 1)
        j0 = np.clip(j0, 0, ny - 1)
        k0 = np.clip(k0, 0, nz - 1)

        wx0 = (1.0 - tx)[:, None]
        wx1 = tx[:, None]
        wy0 = (1.0 - ty)[:, None]
        wy1 = ty[:, None]
        wz0 = (1.0 - tz)[:, None]
        wz1 = tz[:, None]

        c000 = field_3d[i0, j0, k0]
        c001 = field_3d[i0, j0, k1]
        c010 = field_3d[i0, j1, k0]
        c011 = field_3d[i0, j1, k1]
        c100 = field_3d[i1, j0, k0]
        c101 = field_3d[i1, j0, k1]
        c110 = field_3d[i1, j1, k0]
        c111 = field_3d[i1, j1, k1]

        c00 = c000 * wz0 + c001 * wz1
        c01 = c010 * wz0 + c011 * wz1
        c10 = c100 * wz0 + c101 * wz1
        c11 = c110 * wz0 + c111 * wz1
        c0 = c00 * wy0 + c01 * wy1
        c1 = c10 * wy0 + c11 * wy1
        samples = c0 * wx0 + c1 * wx1
        return samples.reshape(tuple(original_shape[:-1]) + (3,))

    def _uses_tangential_spatial_solver(self) -> bool:
        return str(self.geometry.metadata.get("spatial_solver", "legacy_cartesian_square")).lower() in (
            "tangential_oversampled", "tangential_weighted_scaled"
        )

    def _solve_points(self) -> np.ndarray:
        return self.geometry.solve_obs_points if self._uses_tangential_spatial_solver() else self.geometry.ant_obs_points

    def _solve_normals(self) -> np.ndarray:
        return self.geometry.solve_normals if self._uses_tangential_spatial_solver() else self.geometry.ant_normals

    def _solve_weights(self) -> np.ndarray:
        return self.geometry.solve_weights if self._uses_tangential_spatial_solver() else self.geometry.source_weights

    def _project_source_tangent(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float64).reshape(self.geometry.num_sources, 3)
        if not self._uses_tangential_spatial_solver():
            return vectors
        n = self.geometry.ant_normals
        return vectors - np.sum(vectors*n, axis=1)[:, None]*n

    def _field_to_tangential_coeffs(self, field_vectors: np.ndarray) -> np.ndarray:
        field_vectors = np.asarray(field_vectors, dtype=np.float64).reshape(self.geometry.num_collocation_points, 3)
        t1 = self.geometry.solve_tangent1
        t2 = self.geometry.solve_tangent2
        out = np.empty(2*self.geometry.num_collocation_points, dtype=np.float64)
        out[0::2] = np.sum(field_vectors*t1, axis=1)
        out[1::2] = np.sum(field_vectors*t2, axis=1)
        return out

    def _source_coeffs_to_vectors(self, coeff: np.ndarray) -> np.ndarray:
        coeff = np.asarray(coeff, dtype=np.float64).reshape(self.geometry.num_sources, 2)
        return coeff[:,0,None]*self.geometry.source_tangent1 + coeff[:,1,None]*self.geometry.source_tangent2

    def _evaluate_active_current(self, t: float) -> np.ndarray:
        n = self.geometry.num_points
        if self.active_excitation is None:
            return np.zeros((n, 3), dtype=np.float64)
        if hasattr(self.active_excitation, "evaluate"):
            current = self.active_excitation.evaluate(self.geometry.ant_src_points, t, self.time_step)
        else:
            current = self.active_excitation(self.geometry.ant_src_points, t, self.time_step)
        current = np.asarray(current, dtype=np.float64).reshape(n, 3)
        if self._uses_tangential_spatial_solver():
            current = self._project_source_tangent(current)
        if not np.all(np.isfinite(current)):
            raise ValueError("active excitation returned non-finite current values")
        return current

    def _apply_G(self, current_vectors: np.ndarray) -> np.ndarray:
        """Apply the legacy projected solve matrix (diagnostic compatibility only)."""
        if self._uses_tangential_spatial_solver():
            raise RuntimeError("Step-2G tangential solve uses G_physical plus tangent constraints, not _apply_G().")
        current_flat = np.asarray(current_vectors, dtype=np.float64).reshape(-1)
        return (self.geometry.G @ current_flat).reshape(-1, 3)

    def _apply_G_physical(self, current_vectors: np.ndarray) -> np.ndarray:
        """Physical current-node E coefficient before tangential projection."""
        current_flat = np.asarray(current_vectors, dtype=np.float64).reshape(-1)
        Gp = getattr(self.geometry, 'G_physical', None)
        if Gp is None:
            raise RuntimeError('Step-2B geometry is missing G_physical; rebuild MAS preprocessing.')
        return (Gp @ current_flat).reshape(-1, 3)

    def _project_tangential(self, field_vectors: np.ndarray) -> np.ndarray:
        field_vectors = np.asarray(field_vectors, dtype=np.float64).reshape(-1, 3)
        if len(field_vectors) == self.geometry.num_collocation_points:
            normals = self._solve_normals()
        elif len(field_vectors) == self.geometry.num_sources:
            normals = self.geometry.ant_normals
        else:
            raise ValueError("Field length does not match MAS source or solve-collocation cloud")
        normal_comp = np.sum(field_vectors * normals, axis=1)[:, None] * normals
        return field_vectors - normal_comp

    def _previous_temporal_state(self, history_K=None, history_P=None):
        """Return K^(n-1), P^(n-1), using zero virtual prehistory at n=0."""
        n = self.geometry.num_points
        if self.time_step <= 0:
            z = np.zeros((n, 3), dtype=np.float64)
            return z, z.copy()
        prev = self.time_step - 1
        if not self._history_value_available(prev):
            raise RuntimeError('Previous MAS temporal state is unavailable')
        slot = self._history_slot(prev)
        K_hist = self.antenna_J_history if history_K is None else history_K
        P_hist = self.antenna_P_history if history_P is None else history_P
        return K_hist[slot].copy(), P_hist[slot].copy()

    def _known_history_E(self, K_prev: np.ndarray, P_prev: np.ndarray) -> np.ndarray:
        """Known local E contribution after eliminating P^n and Kdot^n.

        E^n = G_physical K^n + E_history, with

          E_history = G_p (P^(n-1)+dt/2 K^(n-1))
                      -G_kdot K^(n-1)/dt.
        """
        G_p = getattr(self.geometry, 'G_p', None)
        G_kdot = getattr(self.geometry, 'G_kdot', None)
        if G_p is None or G_kdot is None:
            raise RuntimeError('Step-2B geometry is missing dipole operator blocks')
        kp = np.asarray(K_prev, dtype=np.float64).reshape(-1)
        pp = np.asarray(P_prev, dtype=np.float64).reshape(-1)
        field = G_p @ (pp + 0.5 * self.dt * kp) - (G_kdot @ kp) / self.dt
        return field.reshape(-1, 3)

    def _advance_polarization(self, K_now: np.ndarray, *, K_history=None, P_history=None) -> np.ndarray:
        """Return P^n from the Step-2A trapezoidal continuity identity."""
        K_now = np.asarray(K_now, dtype=np.float64).reshape(self.geometry.num_points, 3)
        K_prev, P_prev = self._previous_temporal_state(K_history, P_history)
        return P_prev + 0.5 * self.dt * (K_prev + K_now)

    def solve_passive_current(self, plasma_E_surface: np.ndarray, active_current: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve passive Step-2B/2G surface-current density K^n [A/m]."""
        if active_current is None:
            active_current = np.zeros((self.geometry.num_sources, 3), dtype=np.float64)
        active_current = np.asarray(active_current, dtype=np.float64).reshape(self.geometry.num_sources, 3)
        if self._uses_tangential_spatial_solver():
            active_current = self._project_source_tangent(active_current)
        K_prev, P_prev = self._previous_temporal_state()
        E_history = self._known_history_E(K_prev, P_prev)
        E_active_surface = self._apply_G_physical(active_current)
        plasma_E_surface = np.asarray(plasma_E_surface, dtype=np.float64).reshape(-1, 3)

        if self._uses_tangential_spatial_solver():
            if len(plasma_E_surface) != self.geometry.num_collocation_points:
                raise ValueError(
                    f"Step-2G plasma surface field has {len(plasma_E_surface)} points; "
                    f"expected {self.geometry.num_collocation_points} solve collocation points"
                )
            rhs = -self._field_to_tangential_coeffs(plasma_E_surface + E_history + E_active_surface)
            coeff = self.geometry.G_inv @ rhs
            passive = self._source_coeffs_to_vectors(coeff)
        else:
            if len(plasma_E_surface) != self.geometry.num_sources:
                raise ValueError("Legacy MAS surface field length mismatch")
            rhs = -self._project_tangential(plasma_E_surface + E_history + E_active_surface).reshape(-1)
            passive = (self.geometry.G_inv @ rhs).reshape(-1, 3)

        if not np.all(np.isfinite(passive)):
            raise FloatingPointError("MAS passive current solve produced non-finite values")
        return passive


    def _assemble_surface_charge_operators(self):
        """Build SI Coulomb operators for auxiliary surface-charge densities.

        The existing MAS point weights represent source-patch areas.  A basis
        value of one in the returned matrices therefore means 1 C/m^2 on one
        source patch; its integrated point charge is ``source_weights[j]`` C.
        """
        obs = np.asarray(self.geometry.ant_obs_points, dtype=np.float64).reshape(-1, 3)
        src = np.asarray(self.geometry.ant_src_points, dtype=np.float64).reshape(-1, 3)
        weights = np.asarray(self.geometry.source_weights, dtype=np.float64).reshape(-1)
        n = len(src)
        if len(obs) != n or n == 0:
            raise ValueError("MAS surface charge operators require matched non-empty observation/source points")
        if np.any(weights <= 0.0):
            raise ValueError("MAS source weights must be positive for surface-charge closure")
        dr = obs[:, None, :] - src[None, :, :]
        r2 = np.sum(dr * dr, axis=2) + max(self.source_softening, _EPS) ** 2
        inv_r = 1.0 / np.sqrt(np.maximum(r2, _EPS))
        coeff = 1.0 / (4.0 * math.pi * self.epsilon0)
        potential = coeff * inv_r * weights[None, :]
        field = coeff * dr * (inv_r ** 3)[:, :, None] * weights[None, :, None]
        normals = np.asarray(self.geometry.ant_normals, dtype=np.float64).reshape(n, 3)
        normal_part = np.sum(field * normals[:, None, :], axis=2)[:, :, None] * normals[:, None, :]
        field_tan = field - normal_part
        # Matrix rows follow field.reshape(-1): x,y,z entries for every obs point.
        full_matrix = np.transpose(field, (0, 2, 1)).reshape(3 * n, n)
        tan_matrix = np.transpose(field_tan, (0, 2, 1)).reshape(3 * n, n)
        return full_matrix, tan_matrix, potential

    def _prepare_floating_equipotential_solver(self):
        """Pre-factor the connected-conductor surface-charge closure.

        Unknowns are scaled patch charges ``x_j = q_j / qscale`` and the
        floating potential ``V_f``.  Without ridge the original square
        collocation system is used exactly:

            [ Pq*qscale  -1 ] [ x   ] = [ -phi_plasma ]
            [    1^T      0 ] [ V_f ]   [ Q/qscale   ]

        With ``surface_charge_ridge > 0`` the net charge equation is still
        enforced as a hard constraint, while the equipotential equations are
        solved as a Tikhonov-regularized constrained least-squares problem:

            minimize ||A*y - b||^2 + ridge*||x||^2
            subject to sum(x_j) = Q/qscale,

        where ``y = [x, V_f]`` and the potential ``V_f`` is not regularized.
        The ridge is dimensionless because it acts on the scaled patch-charge
        unknowns.  This is intentionally separate from geometry diagonal
        loading, which regularizes the electromagnetic MAS current operator.
        """
        weights = np.asarray(self.geometry.source_weights, dtype=np.float64).reshape(-1)
        n = self.geometry.num_points
        if len(weights) != n or np.any(weights <= 0.0):
            raise ValueError("positive MAS surface-patch areas are required")
        area_total = float(np.sum(weights))
        equivalent_radius = max(math.sqrt(area_total / (4.0 * math.pi)), self.source_softening, 1.0e-6)
        # Approximate isolated-body capacitance at 1 V; order-one scaling only.
        qscale = max(4.0 * math.pi * self.epsilon0 * equivalent_radius, 1.0e-15)
        potential_per_charge = self._surface_charge_potential_matrix / weights[None, :]

        # Potential equation matrix for y=[q/qscale, V_f].
        potential_block = potential_per_charge * qscale
        equation_matrix = np.zeros((n, n + 1), dtype=np.float64)
        equation_matrix[:, :n] = potential_block
        equation_matrix[:, n] = -1.0

        # Original square KKT/collocation matrix kept for ridge=0 and for
        # reporting the unregularized conditioning.
        kkt = np.zeros((n + 1, n + 1), dtype=np.float64)
        kkt[:n, :n] = potential_block
        kkt[:n, n] = -1.0
        kkt[n, :n] = 1.0

        self._floating_charge_scale_C = float(qscale)
        self._surface_potential_per_patch_charge = potential_per_charge
        self._floating_potential_equation_matrix = equation_matrix
        self._floating_kkt_matrix = kkt
        self._floating_unregularized_kkt_condition_number = float(np.linalg.cond(kkt))
        self._floating_kkt_inverse = np.linalg.pinv(kkt, rcond=1.0e-12)

        ridge = self.surface_charge_ridge
        self._floating_regularized_kkt_matrix = None
        self._floating_regularized_kkt_inverse = None
        self._floating_regularized_kkt_condition_number = None
        if ridge > 0.0:
            normal = equation_matrix.T @ equation_matrix
            normal[:n, :n] += ridge * np.eye(n, dtype=np.float64)
            reg_kkt = np.zeros((n + 2, n + 2), dtype=np.float64)
            reg_kkt[:n + 1, :n + 1] = normal
            reg_kkt[:n, n + 1] = 1.0
            reg_kkt[n + 1, :n] = 1.0
            self._floating_regularized_kkt_matrix = reg_kkt
            self._floating_regularized_kkt_condition_number = float(np.linalg.cond(reg_kkt))
            self._floating_regularized_kkt_inverse = np.linalg.pinv(reg_kkt, rcond=1.0e-12)
            self._floating_kkt_condition_number = self._floating_regularized_kkt_condition_number
        else:
            self._floating_kkt_condition_number = self._floating_unregularized_kkt_condition_number

    def _assemble_plasma_surface_potential_operator(self):
        """Build instantaneous Coulomb map from PIC volume rho to surface phi.

        Case 2 is quasi-electrostatic; for this benchmark the conductor boundary
        equation uses the instantaneous scalar potential generated by the PIC
        volume charge.  The operator is constant for a fixed grid and geometry.
        """
        obs = np.asarray(self.geometry.ant_obs_points, dtype=np.float64).reshape(-1, 3)
        cells = np.asarray(self.grid_points, dtype=np.float64).reshape(-1, 3)
        cell_volume = float(np.prod(np.asarray(self.grid_spacing, dtype=np.float64)))
        softening = max(0.5 * min(self.grid_spacing), _EPS)
        coeff = cell_volume / (4.0 * math.pi * self.epsilon0)
        dr = obs[:, None, :] - cells[None, :, :]
        r2 = np.sum(dr * dr, axis=2) + softening * softening
        return coeff / np.sqrt(np.maximum(r2, _EPS))

    def plasma_volume_charge_potential_on_surface(self, rho_volume_C_per_m3: np.ndarray) -> np.ndarray:
        """Evaluate PIC volume-charge scalar potential at MAS surface points."""
        rho = np.asarray(rho_volume_C_per_m3, dtype=np.float64).reshape(-1)
        if len(rho) != len(self.grid_points):
            raise ValueError("rho_volume_C_per_m3 length does not match MAS grid")
        if self._plasma_surface_potential_operator is None:
            self._plasma_surface_potential_operator = self._assemble_plasma_surface_potential_operator()
        return self._plasma_surface_potential_operator @ rho

    def surface_charge_field(self, surface_charge_density: Optional[np.ndarray] = None) -> np.ndarray:
        """Electric field at MAS observation points due to conductor surface charge."""
        sigma = self.last_surface_charge if surface_charge_density is None else np.asarray(surface_charge_density, dtype=np.float64).reshape(-1)
        return (self._surface_charge_E_matrix @ sigma).reshape(self.geometry.num_points, 3)

    def solve_floating_surface_charge(self, incident_potential_surface_V: np.ndarray, net_charge_C: float,
                                      incident_E_surface: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve an ideal connected floating conductor by equal-potential closure.

        The primary equations are now the electrostatic conductor conditions:

            phi_plasma(x_i) + phi_sigma(x_i) = V_f   for all surface points,
            integral_S sigma dS = net_charge_C.

        Surface tangential electric field is retained only as an independent
        diagnostic of the discretized field representation; it is not the
        quantity minimized to define the surface charge distribution.
        """
        n = self.geometry.num_points
        phi_plasma = np.asarray(incident_potential_surface_V, dtype=np.float64).reshape(n)
        charge_rhs = float(net_charge_C) / self._floating_charge_scale_C
        if self.surface_charge_ridge > 0.0:
            rhs = np.empty(n + 2, dtype=np.float64)
            rhs[:n + 1] = self._floating_potential_equation_matrix.T @ (-phi_plasma)
            rhs[n + 1] = charge_rhs
            solution_augmented = self._floating_regularized_kkt_inverse @ rhs
            solution = solution_augmented[:n + 1]
        else:
            rhs = np.empty(n + 1, dtype=np.float64)
            rhs[:n] = -phi_plasma
            rhs[n] = charge_rhs
            solution = self._floating_kkt_inverse @ rhs
        q_patch = solution[:n] * self._floating_charge_scale_C
        floating_potential = float(solution[n])
        weights = np.asarray(self.geometry.source_weights, dtype=np.float64).reshape(n)
        sigma = q_patch / weights
        if not np.all(np.isfinite(sigma)) or not math.isfinite(floating_potential):
            raise FloatingPointError("MAS floating equipotential surface-charge solve produced non-finite values")

        potential_self = self._surface_charge_potential_matrix @ sigma
        potential_total = phi_plasma + potential_self
        potential_residual = potential_total - floating_potential
        q_actual = float(np.sum(q_patch))

        if incident_E_surface is None:
            rms_tan = float("nan")
            max_tan = float("nan")
        else:
            incident_E = np.asarray(incident_E_surface, dtype=np.float64).reshape(n, 3)
            E_charge = self.surface_charge_field(sigma)
            residual_E = self._project_tangential(incident_E + E_charge)
            residual_norm = np.linalg.norm(residual_E, axis=1)
            rms_tan = float(np.sqrt(np.mean(residual_norm * residual_norm)))
            max_tan = float(np.max(residual_norm)) if len(residual_norm) else 0.0

        slot = self._history_slot(self.time_step)
        self.surface_charge_history[slot] = sigma
        self.last_surface_charge = sigma.copy()
        self.last_surface_charge_info = {
            "closure": "equipotential_plus_net_charge" if self.surface_charge_ridge <= 0.0 else "regularized_equipotential_plus_net_charge",
            "surface_charge_ridge": self.surface_charge_ridge,
            "surface_charge_regularization": "none" if self.surface_charge_ridge <= 0.0 else "scaled_patch_charge_ridge",
            "net_charge_C": q_actual,
            "charge_constraint_error_C": q_actual - float(net_charge_C),
            "floating_potential_V": floating_potential,
            "potential_rms_residual_V": float(np.sqrt(np.mean(potential_residual * potential_residual))),
            "potential_max_residual_V": float(np.max(np.abs(potential_residual))) if len(potential_residual) else 0.0,
            "rms_tangential_E": rms_tan,
            "max_tangential_E": max_tan,
            "plasma_potential_mean_V": float(np.mean(phi_plasma)),
            "self_potential_mean_V": float(np.mean(potential_self)),
            "self_potential_rms_variation_V": float(np.sqrt(np.mean((potential_self - np.mean(potential_self)) ** 2))),
            "kkt_condition_number": self._floating_kkt_condition_number,
            "unregularized_kkt_condition_number": self._floating_unregularized_kkt_condition_number,
            "regularized_kkt_condition_number": self._floating_regularized_kkt_condition_number,
            "charge_scale_C": self._floating_charge_scale_C,
        }
        return sigma

    def get_surface_charge_info(self) -> Dict[str, float]:
        return dict(self.last_surface_charge_info)

    def _history_slot(self, abs_step: int) -> int:
        return int(abs_step) % self.max_time_steps

    def _history_value_available(self, abs_step: int) -> bool:
        if abs_step < 0:
            return False
        return bool(self.history_abs_steps[self._history_slot(abs_step)] == int(abs_step))

    def _dipole_history_sample(self, source_id: int, tr: float, max_step: int,
                              K_history=None, P_history=None):
        """Sample the continuity-consistent temporal basis at retarded time tr.

        K is piecewise linear.  P is the exact time integral of that linear K,
        hence piecewise quadratic.  Kdot is constant on each interval.  At an
        exact node we use the backward interval derivative, matching the local
        operator used by the MAS solve.
        """
        if tr < 0.0:
            return None, None, None
        K_hist = self.antenna_J_history if K_history is None else K_history
        P_hist = self.antenna_P_history if P_history is None else P_history
        tau = tr / self.dt
        k = int(math.floor(tau))
        alpha = float(tau - k)
        tol = 1.0e-12
        if alpha < tol:
            if k < 0 or k > max_step or not self._history_value_available(k):
                return None, None, None
            slot = self._history_slot(k)
            P = P_hist[slot, source_id].copy()
            K = K_hist[slot, source_id].copy()
            if k <= 0 or not self._history_value_available(k - 1):
                Kdot = np.zeros(3, dtype=np.float64)
            else:
                prev = K_hist[self._history_slot(k - 1), source_id]
                Kdot = (K - prev) / self.dt
            return P, K, Kdot

        k1 = k + 1
        if k < 0 or k1 > max_step:
            return None, None, None
        if not (self._history_value_available(k) and self._history_value_available(k1)):
            return None, None, None
        s0 = self._history_slot(k)
        s1 = self._history_slot(k1)
        P0 = P_hist[s0, source_id]
        K0 = K_hist[s0, source_id]
        K1 = K_hist[s1, source_id]
        dK = K1 - K0
        K = K0 + alpha * dK
        Kdot = dK / self.dt
        P = P0 + self.dt * (alpha * K0 + 0.5 * alpha * alpha * dK)
        return P, K, Kdot

    def compute_point_source_fields(self, points: np.ndarray, history: Optional[np.ndarray] = None,
                                    max_step: Optional[int] = None, chunk_size: int = 4096,
                                    truncation_radius: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
        """CPU reference fields of Step-2B electric-dipole MAS sources.

        ``history`` is retained only for backward API compatibility and, when
        supplied, is interpreted as a K-history.  P is taken from the matching
        continuity-consistent ``antenna_P_history``.
        """
        if not getattr(self, "allow_full_grid_cpu_mas", False):
            guard_against_full_grid_cpu_mas(points, self.grid_points, name="MAS compute_point_source_fields")
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        K_hist = self.antenna_J_history if history is None else np.asarray(history, dtype=np.float64)
        P_hist = self.antenna_P_history
        if max_step is None:
            max_step = max(0, self.time_step)
        max_step = int(max_step)
        t = max_step * self.dt
        src = np.asarray(self.geometry.ant_src_points, dtype=np.float64)
        weights = np.asarray(self.geometry.source_weights, dtype=np.float64)
        E = np.zeros((len(points), 3), dtype=np.float64)
        B = np.zeros((len(points), 3), dtype=np.float64)
        if len(points) == 0 or len(src) == 0:
            return E, B
        if truncation_radius is None:
            truncation_radius = self.source_truncation_radius
        truncation_radius = max(0.0, float(truncation_radius or 0.0))
        coeff_e = 1.0 / (4.0 * np.pi * self.epsilon0)
        coeff_b = self.mu0 / (4.0 * np.pi)
        soft2 = self.source_softening * self.source_softening

        for io, ro in enumerate(points):
            e = np.zeros(3, dtype=np.float64)
            b = np.zeros(3, dtype=np.float64)
            for js, rs in enumerate(src):
                r = ro - rs
                R = float(np.sqrt(np.dot(r, r) + soft2))
                if R <= _EPS:
                    continue
                taper = float(_spatial_taper_weights(np.array([R]), truncation_radius,
                                                     self.source_taper_width,
                                                     self.source_taper_type)[0])
                if taper <= 0.0:
                    continue
                tr = t - R / self.c
                P, K, Kdot = self._dipole_history_sample(js, tr, max_step, K_hist, P_hist)
                if P is None:
                    continue
                w = weights[js] * taper
                rr = np.outer(r, r)
                near = (3.0 * rr / R**5 - np.eye(3) / R**3) @ P
                induction = ((3.0 * rr / R**4 - np.eye(3) / R**2) @ K) / self.c
                radiation = ((rr / R**3 - np.eye(3) / R) @ Kdot) / (self.c * self.c)
                e += coeff_e * w * (near + induction + radiation)
                b += coeff_b * w * (
                    np.cross(K, r) / R**3 + np.cross(Kdot, r) / (self.c * R**2)
                )
            E[io] = e
            B[io] = b
        return E, B


    def _compute_virtual_work_check(self, plasma_E_surface, active_current, passive_current) -> Dict[str, float]:
        K_prev, P_prev = self._previous_temporal_state()
        E_history = self._known_history_E(K_prev, P_prev)
        E_active = self._apply_G_physical(active_current)
        E_passive = self._apply_G_physical(passive_current)
        E_total_surface = plasma_E_surface + E_history + E_active + E_passive
        E_tan = self._project_tangential(E_total_surface)
        self.last_surface_tangential_E = E_tan.copy()
        weights = self._solve_weights()
        virtual_work = float(np.sum(weights * np.linalg.norm(E_tan, axis=1) ** 2))
        max_tan = float(np.max(np.linalg.norm(E_tan, axis=1))) if len(E_tan) else 0.0
        rms_tan = float(math.sqrt(np.mean(np.linalg.norm(E_tan, axis=1) ** 2))) if len(E_tan) else 0.0
        return {
            "virtual_work_tangential_E2": virtual_work,
            "max_tangential_E": max_tan,
            "rms_tangential_E": rms_tan,
        }

    def register_with_em_solver(self, em_solver):
        """Register MAS source and observation points with the GPU EM solver."""
        if em_solver is None:
            raise ValueError("em_solver must not be None")
        em_solver.set_antenna_sources(
            self.geometry.ant_src_points,
            source_weights=self.geometry.source_weights,
            softening_distance=self.source_softening,
        )
        if hasattr(em_solver, "set_spatial_truncation_radius") and (self.source_truncation_radius > 0.0 or self.source_taper_type > 0):
            em_solver.set_spatial_truncation_radius(
                point_radius=self.source_truncation_radius,
                point_taper=self._source_taper_name(),
                point_taper_width=self.source_taper_width,
            )
        if hasattr(em_solver, "register_sample_points"):
            em_solver.register_sample_points(self._solve_points())
        return em_solver


    def _source_taper_name(self) -> str:
        return {0: "hard", 1: "hann", 2: "smoothstep", 3: "exponential"}.get(int(self.source_taper_type), "hard")

    def infer_active_max_frequency(self) -> Optional[float]:
        """Best-effort frequency estimate from the configured active excitation."""
        if self.active_excitation is None:
            return None
        if hasattr(self.active_excitation, "frequency"):
            return abs(float(getattr(self.active_excitation, "frequency")))
        if hasattr(self.active_excitation, "frequencies"):
            arr = np.asarray(getattr(self.active_excitation, "frequencies"), dtype=np.float64)
            if arr.size:
                return float(np.max(np.abs(arr)))
        return None

    def validate_time_resolution(self, f_max: Optional[float] = None, min_period: Optional[float] = None,
                                 steps_per_period: int = 10, action: str = "warn") -> Dict[str, object]:
        """Check whether dt resolves the fastest imposed current variation.

        This does not prove absolute late-time stability, but it catches the most
        common setup error for explicit retarded-history interpolation: using a
        time step too large relative to the highest active or plasma frequency.
        """
        if f_max is None:
            if min_period is not None and min_period > 0.0:
                f_max = 1.0 / float(min_period)
            else:
                f_max = self.infer_active_max_frequency()
        if f_max is None or f_max <= 0.0:
            return {
                "ok": None,
                "dt": float(self.dt),
                "f_max": None,
                "recommended_dt_max": None,
                "steps_per_period": int(steps_per_period),
                "message": "No f_max was supplied and no active excitation frequency could be inferred.",
            }
        steps_per_period = max(1, int(steps_per_period))
        dt_max = 1.0 / (steps_per_period * float(f_max))
        ok = self.dt <= dt_max * (1.0 + 1.0e-12)
        info = {
            "ok": bool(ok),
            "dt": float(self.dt),
            "f_max": float(f_max),
            "recommended_dt_max": float(dt_max),
            "steps_per_period": int(steps_per_period),
            "actual_steps_per_period": float(1.0 / (float(f_max) * self.dt)),
        }
        if not ok:
            msg = (
                f"MAS/Jefimenko time step may under-resolve the fastest source: dt={self.dt:.6g}, "
                f"recommended <= {dt_max:.6g} for {steps_per_period} samples/period at f_max={float(f_max):.6g} Hz. "
                "Under-resolved retarded interpolation can seed late-time oscillations."
            )
            if action == "raise":
                raise ValueError(msg)
            if action == "warn":
                import warnings
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return info

    def validate_time_domain_mas_regime(self, f_max: Optional[float] = None,
                                        phase_delay_limit: float = 0.1,
                                        action: str = "warn") -> Dict[str, object]:
        """Estimate whether the time-local MAS matrix is adequate.

        The current MAS feedback uses a precomputed time-local matrix ``G``:
        it enforces the instantaneous surface boundary condition but does not
        solve a full convolutional mutual-retardation system among MAS source
        points.  This is usually acceptable when the antenna electrical size is
        small enough that the source-to-source propagation delay is a small
        fraction of the fastest period.  For wideband, resonant, or array
        antennas, a full time-domain MAS / MOT formulation may be required.
        """
        if f_max is None:
            f_max = self.infer_active_max_frequency()
        pts = np.vstack([self.geometry.ant_obs_points, self.geometry.ant_src_points])
        extent = np.ptp(pts, axis=0)
        max_dimension = float(np.linalg.norm(extent))
        max_delay = max_dimension / max(self.c, 1.0e-300)

        if f_max is None or f_max <= 0.0:
            return {
                "ok": None,
                "f_max": None,
                "max_dimension_m": max_dimension,
                "max_propagation_delay_s": max_delay,
                "delay_fraction_of_period": None,
                "phase_delay_cycles": None,
                "phase_delay_limit": float(phase_delay_limit),
                "message": "No f_max was supplied and no active excitation frequency could be inferred.",
            }

        period = 1.0 / float(f_max)
        delay_fraction = max_delay / period
        ok = delay_fraction <= float(phase_delay_limit)
        info = {
            "ok": bool(ok),
            "f_max": float(f_max),
            "max_dimension_m": max_dimension,
            "max_propagation_delay_s": max_delay,
            "period_s": float(period),
            "delay_fraction_of_period": float(delay_fraction),
            "phase_delay_cycles": float(delay_fraction),
            "phase_delay_limit": float(phase_delay_limit),
        }
        if not ok:
            msg = (
                f"Time-local MAS may be inaccurate for this electrical size: antenna delay/cycle="
                f"{delay_fraction:.6g}, limit={float(phase_delay_limit):.6g}. "
                "Consider a full time-domain MAS/MOT formulation with retarded mutual "
                "coupling between auxiliary sources, or reduce dt/frequency/antenna size."
            )
            if action == "raise":
                raise ValueError(msg)
            if action == "warn":
                import warnings
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return info

    def _weighted_source_rms(self, vectors: np.ndarray) -> float:
        vectors = np.asarray(vectors, dtype=np.float64).reshape(self.geometry.num_points, 3)
        weights = np.abs(np.asarray(self.geometry.source_weights, dtype=np.float64).reshape(-1))
        wsum = max(float(np.sum(weights)), _EPS)
        return float(np.sqrt(np.sum(weights * np.sum(vectors * vectors, axis=1)) / wsum))

    def _weighted_source_max(self, vectors: np.ndarray) -> float:
        vectors = np.asarray(vectors, dtype=np.float64).reshape(self.geometry.num_points, 3)
        return float(np.max(np.linalg.norm(vectors, axis=1))) if len(vectors) else 0.0

    def _temporal_operator_audit(self, K_history: np.ndarray, P_history: np.ndarray,
                                 K_now: np.ndarray, P_now: np.ndarray, abs_step: int) -> Dict[str, float]:
        """Audit the Step-2B temporal source identity and local/Pass-2 semantics."""
        K_now = np.asarray(K_now, dtype=np.float64).reshape(self.geometry.num_points, 3)
        P_now = np.asarray(P_now, dtype=np.float64).reshape(self.geometry.num_points, 3)
        if abs_step <= 0:
            K_prev = np.zeros_like(K_now)
            P_prev = np.zeros_like(P_now)
        else:
            prev_step = int(abs_step) - 1
            if not self._history_value_available(prev_step):
                return {'history_derivative_available': False}
            slot = self._history_slot(prev_step)
            K_prev = K_history[slot]
            P_prev = P_history[slot]
        Kdot = (K_now - K_prev) / self.dt
        temporal_residual = (P_now - P_prev) / self.dt - 0.5 * (K_prev + K_now)
        # The local node derivative and Pass-2 interval derivative are now the
        # same object by construction.
        return {
            'history_derivative_available': True,
            'surface_current_weighted_rms_A_per_m': self._weighted_source_rms(K_now),
            'surface_current_max_A_per_m': self._weighted_source_max(K_now),
            'Kdot_weighted_rms_A_per_m_s': self._weighted_source_rms(Kdot),
            'Kdot_max_A_per_m_s': self._weighted_source_max(Kdot),
            'temporal_continuity_residual_weighted_rms': self._weighted_source_rms(temporal_residual),
            'temporal_continuity_residual_max': self._weighted_source_max(temporal_residual),
            'local_vs_pass2_Kdot_relative_rms': 0.0,
            'local_to_pass2_Kdot_norm_ratio': 1.0,
        }

    def _update_source_closure_diagnostics(self, active_current, passive_current, total_current,
                                           active_P, passive_P, total_P):
        step = int(self.time_step)
        total = self._temporal_operator_audit(self.antenna_J_history, self.antenna_P_history,
                                              total_current, total_P, step)
        active = self._temporal_operator_audit(self.active_J_history, self.active_P_history,
                                               active_current, active_P, step)
        passive = self._temporal_operator_audit(self.passive_J_history, self.passive_P_history,
                                                passive_current, passive_P, step)
        self.last_source_closure_diagnostics = {
            'time_step': step,
            'time_s': float(step * self.dt),
            'valid_history_derivative': bool(total.get('history_derivative_available', False)),
            'continuity_linked_rho_constructed': True,
            'source_model': 'electric dipole: rho=-p.grad(delta), J=pdot delta; p=wP, pdot=wK',
            'coefficient_units': {'P': 'C/m', 'K': 'A/m', 'source_weight': 'm^2'},
            'local_operator_derivative_definition': 'Kdot^n=(K^n-K^(n-1))/dt',
            'pass2_history_derivative_definition': 'same derivative of piecewise-linear K on [t_(n-1),t_n]',
            'temporal_interpolation': 'K linear; P is its exact piecewise-quadratic time integral',
            'total': total,
            'active': active,
            'passive': passive,
        }

    def get_source_closure_diagnostics(self) -> Dict[str, object]:
        out = dict(self.last_source_closure_diagnostics)
        for key in ('total', 'active', 'passive'):
            if isinstance(out.get(key), dict):
                out[key] = dict(out[key])
        return out

    def solve_currents_from_surface_field(self, plasma_E_surface: np.ndarray) -> np.ndarray:
        """Solve/store active+passive surface-current density K^n [A/m]."""
        expected_surface_points = self.geometry.num_collocation_points if self._uses_tangential_spatial_solver() else self.geometry.num_sources
        plasma_E_surface = np.asarray(plasma_E_surface, dtype=np.float64).reshape(expected_surface_points, 3)
        t = self.time_step * self.dt
        active_current = self._evaluate_active_current(t)
        passive_current = self.solve_passive_current(plasma_E_surface, active_current)
        total_current = active_current + passive_current

        # Advance P consistently for total/active/passive histories.  All use a
        # virtual zero prehistory at n=0, matching the precomputed local G.
        def advance_component(K_now, K_hist, P_hist):
            if self.time_step <= 0:
                K_prev = np.zeros_like(K_now)
                P_prev = np.zeros_like(K_now)
            else:
                ps = self._history_slot(self.time_step - 1)
                K_prev = K_hist[ps]
                P_prev = P_hist[ps]
            return P_prev + 0.5 * self.dt * (K_prev + K_now)

        active_P = advance_component(active_current, self.active_J_history, self.active_P_history)
        passive_P = advance_component(passive_current, self.passive_J_history, self.passive_P_history)
        total_P = advance_component(total_current, self.antenna_J_history, self.antenna_P_history)

        slot = self._history_slot(self.time_step)
        self.active_J_history[slot] = active_current
        self.passive_J_history[slot] = passive_current
        self.antenna_J_history[slot] = total_current
        self.active_P_history[slot] = active_P
        self.passive_P_history[slot] = passive_P
        self.antenna_P_history[slot] = total_P
        self.history_abs_steps[slot] = self.time_step
        self.last_active_current = active_current.copy()
        self.last_passive_current = passive_current.copy()
        self.last_total_current = total_current.copy()
        self.last_active_polarization = active_P.copy()
        self.last_passive_polarization = passive_P.copy()
        self.last_total_polarization = total_P.copy()
        self._update_source_closure_diagnostics(
            active_current, passive_current, total_current,
            active_P, passive_P, total_P,
        )
        self.last_virtual_work = self._compute_virtual_work_check(plasma_E_surface, active_current, passive_current)
        return total_current


    def advance_time_step(self):
        self.time_step += 1

    def update_fields_with_device_interaction(self, plasma_E=None, plasma_B=None, rho_plasma=None, J_plasma=None,
                                              surface_hits=None, em_solver=None, plasma_E_surface=None,
                                              compute_grid_antenna_field: bool = False,
                                              return_gpu_fields: bool = True):
        """Solve MAS feedback and optionally produce the total grid field.

        Fast path for production:

            1. Run ``em_solver.Jefimenko_solver(rho, Jx, Jy, Jz, include_antenna=False)``.
            2. Call this method with ``em_solver=em_solver`` and without passing
               full ``plasma_E/plasma_B`` arrays.
            3. This method samples only the antenna observation points on GPU,
               solves the small MAS system on CPU, pushes the antenna currents
               into the same EM history slot, and calls
               ``em_solver.recompute_current_fields(...)`` to get the total field.

        The CPU full-grid point-source fallback is retained behind
        ``compute_grid_antenna_field=True`` for small debugging cases only.
        """
        if plasma_E_surface is None:
            if em_solver is not None and hasattr(em_solver, "sample_current_E_at_points"):
                plasma_E_surface = em_solver.sample_current_E_at_points(self.geometry.ant_obs_points)
            else:
                if plasma_E is None:
                    raise ValueError("Provide either plasma_E_surface, plasma_E, or em_solver with sample_current_E_at_points().")
                plasma_E_np = _as_numpy_field(plasma_E)
                nx, ny, nz = self.grid_size
                plasma_E_3d = plasma_E_np.reshape(nx, ny, nz, 3)
                plasma_E_surface = self._sample_vector_field_trilinear(plasma_E_3d, self.geometry.ant_obs_points)

        total_current = self.solve_currents_from_surface_field(plasma_E_surface)

        if em_solver is not None:
            # Push to the same EM time node and recompute the already-deposited
            # plasma sources plus MAS antenna sources.  This avoids CPU grid-wide
            # point-source evaluation and avoids advancing EM time twice.
            expected = getattr(em_solver, "point_source_count", 0)
            if expected != self.geometry.num_points:
                self.register_with_em_solver(em_solver)
            em_solver.recompute_current_fields(
                antenna_surface_currents=total_current,
                antenna_polarization=self.last_total_polarization,
                include_antenna=True,
            )
            Ex, Ey, Ez, Bx, By, Bz = em_solver.acquire_EB_field()
            self.last_field_data = {
                "incident": None,
                "antenna": None,
                "scattered": None,
                "total": {"E_components": (Ex, Ey, Ez), "B_components": (Bx, By, Bz)},
            }
            self.advance_time_step()
            if return_gpu_fields:
                return (Ex, Ey, Ez), (Bx, By, Bz)
            E = np.stack([_as_numpy_field(Ex), _as_numpy_field(Ey), _as_numpy_field(Ez)], axis=-1)
            B = np.stack([_as_numpy_field(Bx), _as_numpy_field(By), _as_numpy_field(Bz)], axis=-1)
            return E.reshape(-1), B.reshape(-1)

        # CPU-only fallback.  This branch is intentionally opt-in because the
        # complexity is O(N_grid * N_MAS); use it only for tiny validation cases.
        if not compute_grid_antenna_field:
            self.last_field_data = None
            self.advance_time_step()
            return None, None

        if plasma_E is None or plasma_B is None:
            raise ValueError("CPU fallback needs both plasma_E and plasma_B arrays.")
        plasma_E_np = _as_numpy_field(plasma_E)
        plasma_B_np = _as_numpy_field(plasma_B)
        nx, ny, nz = self.grid_size
        plasma_E_3d = plasma_E_np.reshape(nx, ny, nz, 3)
        plasma_B_3d = plasma_B_np.reshape(nx, ny, nz, 3)
        antenna_E, antenna_B = self.compute_point_source_fields(
            self.grid_points, self.antenna_J_history, self.time_step,
            truncation_radius=self.source_truncation_radius,
        )
        antenna_E_3d = antenna_E.reshape(nx, ny, nz, 3)
        antenna_B_3d = antenna_B.reshape(nx, ny, nz, 3)
        total_E_3d = plasma_E_3d + antenna_E_3d
        total_B_3d = plasma_B_3d + antenna_B_3d
        self.last_field_data = {
            "incident": {"E": plasma_E_3d.copy(), "B": plasma_B_3d.copy()},
            "antenna": {"E": antenna_E_3d.copy(), "B": antenna_B_3d.copy()},
            "scattered": {"E": antenna_E_3d.copy(), "B": antenna_B_3d.copy()},
            "total": {"E": total_E_3d.copy(), "B": total_B_3d.copy()},
        }
        self.advance_time_step()
        return total_E_3d.reshape(-1), total_B_3d.reshape(-1)

    def get_simulation_info(self) -> Dict[str, object]:
        return {
            "coupling_method": "MAS",
            "num_mas_points": self.geometry.num_points,
            "time_step": self.time_step,
            "grid_size": self.grid_size,
            "grid_spacing": self.grid_spacing,
            "grid_origin": self.grid_origin,
            "time_basis_name": self.time_basis_name,
            "source_softening": self.source_softening,
            "source_truncation_radius": self.source_truncation_radius,
            "source_taper": self._source_taper_name(),
            "source_taper_width": self.source_taper_width,
            "antenna_history_length": self.max_time_steps,
            "condition_number_G": self.geometry.metadata.get("condition_number_G", None),
            "source_closure_diagnostics": self.get_source_closure_diagnostics(),
            "surface_charge_closure": "equipotential + net-charge constrained floating conductor" if self.surface_charge_ridge <= 0.0 else "ridge-regularized equipotential + net-charge constrained floating conductor",
            "surface_charge_ridge": self.surface_charge_ridge,
            "surface_charge_regularization": "none" if self.surface_charge_ridge <= 0.0 else "scaled_patch_charge_ridge",
            "floating_surface_charge_kkt_condition_number": getattr(self, "_floating_kkt_condition_number", None),
            "floating_surface_charge_unregularized_kkt_condition_number": getattr(self, "_floating_unregularized_kkt_condition_number", None),
            "floating_surface_charge_regularized_kkt_condition_number": getattr(self, "_floating_regularized_kkt_condition_number", None),
            "geometry": dict(self.geometry.metadata),
        }

    def get_stability_info(self) -> Dict[str, object]:
        return {
            "method": "MAS continuity-consistent electric-dipole time-local feedback solve",
            "matrix_condition_number": self.geometry.metadata.get("condition_number_G", None),
            "history_length": self.max_time_steps,
            "source_softening": self.source_softening,
            "source_truncation_radius": self.source_truncation_radius,
            "source_taper": self._source_taper_name(),
            "source_taper_width": self.source_taper_width,
            "time_resolution": self.validate_time_resolution(action="ignore"),
            "time_domain_mas_regime": self.validate_time_domain_mas_regime(action="ignore"),
            "note": "History arrays are circular and tagged by absolute step to prevent modulo aliasing; use validate_time_resolution() to catch under-resolved active-current histories; increase source_offset/softening or reduce MAS point count if G is poorly conditioned.",
        }

    def get_virtual_work_check(self) -> Dict[str, float]:
        if self.last_virtual_work is None:
            return {
                "virtual_work_tangential_E2": 0.0,
                "max_tangential_E": 0.0,
                "rms_tangential_E": 0.0,
            }
        return dict(self.last_virtual_work)

    def get_surface_currents(self) -> np.ndarray:
        """Return Step-2B surface-current density K [A/m]."""
        return self.last_total_current.copy()

    def get_surface_polarization(self) -> np.ndarray:
        """Return Step-2B surface-polarization density P [C/m]."""
        return self.last_total_polarization.copy()

    def get_active_currents(self) -> np.ndarray:
        return self.last_active_current.copy()

    def get_passive_currents(self) -> np.ndarray:
        return self.last_passive_current.copy()

    def get_surface_charges(self) -> np.ndarray:
        """Return the latest solved conductor surface-charge density [C/m^2]."""
        return self.last_surface_charge.copy()

    def get_last_field_data(self):
        if self.last_field_data is None:
            raise RuntimeError("No field data cached yet; call update_fields_with_device_interaction first.")
        copied = {}
        for source, source_dict in self.last_field_data.items():
            if source_dict is None:
                copied[source] = None
                continue
            copied[source] = {}
            for comp, values in source_dict.items():
                if isinstance(values, tuple):
                    copied[source][comp] = tuple(v.copy() if hasattr(v, "copy") else v for v in values)
                else:
                    copied[source][comp] = values.copy() if hasattr(values, "copy") else values
        return copied

    def sample_field_at_points(self, probes: Iterable[dict]) -> np.ndarray:
        if self.last_field_data is None:
            raise RuntimeError("No field data cached yet; call update_fields_with_device_interaction first.")
        probes = list(probes or [])
        component_to_index = {"x": 0, "y": 1, "z": 2}
        values = np.zeros(len(probes), dtype=np.float64)
        for idx, probe in enumerate(probes):
            point = np.asarray(probe["point"], dtype=np.float64).reshape(3,)
            field_name = str(probe["field"]).upper()
            component = str(probe["component"]).lower()
            source = str(probe["source"]).lower()
            if source not in self.last_field_data:
                raise ValueError(f"Unknown source '{source}' in probe {probe}. Use incident, antenna/scattered, or total.")
            if self.last_field_data[source] is None or field_name not in self.last_field_data[source]:
                raise RuntimeError(
                    "Requested full-grid field data is not cached in the optimized GPU path. "
                    "Run update_fields_with_device_interaction(..., compute_grid_antenna_field=True) "
                    "for small CPU diagnostics, or sample directly from EMsolver on GPU."
                )
            if field_name not in ("E", "B"):
                raise ValueError(f"Unknown field '{field_name}' in probe {probe}.")
            if component not in component_to_index:
                raise ValueError(f"Unknown component '{component}' in probe {probe}.")
            sampled = self._sample_vector_field_trilinear(self.last_field_data[source][field_name], point)
            values[idx] = float(sampled[component_to_index[component]])
        return values


# Backwards-compatible name for code that previously imported PlasmaDeviceInteraction.
PlasmaDeviceInteraction = MASAntennaInteraction


def create_mas_antenna_simulation(geometry_params: dict, simulation_params: dict) -> MASAntennaInteraction:
    active = simulation_params.get("active_excitation", None)
    if active is None and "active_frequency" in simulation_params:
        active = SinusoidalCurrentExcitation(
            amplitude=simulation_params.get("active_amplitude", 0.0),
            frequency=simulation_params["active_frequency"],
            direction=simulation_params.get("active_direction", (0.0, 0.0, 1.0)),
            phase=simulation_params.get("active_phase", 0.0),
            ramp_time=simulation_params.get("active_ramp_time", 0.0),
        )
    geometry = preprocess_mas_antenna(
        kind=geometry_params.get("kind", "cylinder"),
        dt=simulation_params.get("dt", 1.0e-12),
        epsilon0=simulation_params.get("epsilon0", 8.854187817e-12),
        c=simulation_params.get("c", 299792458.0),
        **{k: v for k, v in geometry_params.items() if k != "kind"},
    )
    return MASAntennaInteraction(
        geometry=geometry,
        dt=simulation_params.get("dt", 1.0e-12),
        epsilon0=simulation_params.get("epsilon0", 8.854187817e-12),
        c=simulation_params.get("c", 299792458.0),
        max_time_steps=simulation_params.get("max_time_steps", 100),
        grid_size=simulation_params.get("grid_size", (32, 32, 32)),
        grid_spacing=simulation_params.get("grid_spacing", (1.0e-3, 1.0e-3, 1.0e-3)),
        grid_origin=simulation_params.get("grid_origin", (-0.016, -0.016, -0.016)),
        active_excitation=active,
        source_softening=simulation_params.get("source_softening", None),
        source_truncation_radius=simulation_params.get("source_truncation_radius", None),
        source_taper=simulation_params.get("source_taper", None),
        source_taper_width=simulation_params.get("source_taper_width", None),
        surface_charge_ridge=simulation_params.get("surface_charge_ridge", 0.0),
    )


def create_plasma_device_simulation(cube_center=None, cube_size=None, simulation_params=None, **kwargs):
    """Compatibility wrapper replacing the TDIE factory.

    Old TDIE examples passed ``cube_center`` and ``cube_size``.  MAS needs an
    antenna point cloud, so this wrapper maps those values to a short cylinder
    when no explicit ``geometry_params`` dictionary is supplied.
    """
    simulation_params = dict(simulation_params or {})
    geometry_params = dict(simulation_params.pop("geometry_params", {}))
    if not geometry_params:
        center = (0.0, 0.0, 0.0) if cube_center is None else cube_center
        size = 1.0e-3 if cube_size is None else float(cube_size)
        geometry_params = {
            "kind": "cylinder",
            "radius": 0.25 * size,
            "length": size,
            "center": center,
            "axis": (0.0, 0.0, 1.0),
            "n_length": simulation_params.get("n_length", 16),
            "n_phi": simulation_params.get("n_phi", 8),
        }
    geometry_params.update(kwargs.pop("geometry_params", {}))
    simulation_params.update(kwargs)
    return create_mas_antenna_simulation(geometry_params, simulation_params)
