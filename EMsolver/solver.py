
import numpy as np
import math
import warnings
import cupy
from numba import cuda
from .cuda_functions import *

BOUNDARY_MODE_TO_INT = {'open': BOUNDARY_OPEN, 'periodic': BOUNDARY_PERIODIC, 'reflective': BOUNDARY_REFLECTIVE}
TAPER_MODE_TO_INT = {None: 0, 'none': 0, 'hard': 0, 'hann': 1, 'hanning': 1, 'smoothstep': 2, 'exponential': 3, 'exp': 3}


def time_index_in_GPU(i_time, len_time_snapshots):
    return i_time % len_time_snapshots


class EMsolver:
    """Jefimenko solver supporting both the original legacy constructor and a size/grid keyword constructor."""

    def __init__(self, *args, **kwargs):
        if kwargs and ('source_size' in kwargs or 'source_grid' in kwargs or 'observation_size' in kwargs):
            self._init_from_region_keywords(**kwargs)
        else:
            self._init_from_legacy_signature(*args, **kwargs)

        self.total_grid_o = self.x_grid_size_o * self.y_grid_size_o * self.z_grid_size_o
        self.total_grid_s = self.x_grid_size_s * self.y_grid_size_s * self.z_grid_size_s
        self.all_grids = self.total_grid_o * self.total_grid_s

        self.threadsperblock = 32
        self.blockspergrid_o = (self.total_grid_o + (self.threadsperblock - 1)) // self.threadsperblock
        self.blockspergrid_s = (self.total_grid_s + (self.threadsperblock - 1)) // self.threadsperblock
        self.blockspergrid_all = (self.all_grids + (self.threadsperblock - 1)) // self.threadsperblock

        # Production grid-source histories.  Charge remains cell centered,
        # while the three current-density components remain on their native
        # finite-volume faces.  The legacy cell-centered grid-current histories
        # were removed after the Step-1C/1D/1F validation campaign showed that
        # face-to-cell reconstruction breaks the discrete continuity relation.
        self.rho_GPU = cupy.zeros(
            [self.len_time_snapshots, self.total_grid_s], dtype=np.float64
        )
        # Persistent zero-valued cell-centered current history used by the
        # charge-only grid kernel in the native staggered-current formulation.
        # This prevents the full-EM charge kernel from accidentally interpreting
        # rho history as J history when the atomic accumulation mode is selected.
        self.grid_zero_GPU = cupy.zeros(
            [self.len_time_snapshots, self.total_grid_s], dtype=np.float64
        )
        self.grid_current_layout = 'staggered_face'
        self.total_Jx_face_s = (self.x_grid_size_s + 1) * self.y_grid_size_s * self.z_grid_size_s
        self.total_Jy_face_s = self.x_grid_size_s * (self.y_grid_size_s + 1) * self.z_grid_size_s
        self.total_Jz_face_s = self.x_grid_size_s * self.y_grid_size_s * (self.z_grid_size_s + 1)
        self.Jx_face_GPU = cupy.zeros(
            [self.len_time_snapshots, self.total_Jx_face_s], dtype=np.float64
        )
        self.Jy_face_GPU = cupy.zeros(
            [self.len_time_snapshots, self.total_Jy_face_s], dtype=np.float64
        )
        self.Jz_face_GPU = cupy.zeros(
            [self.len_time_snapshots, self.total_Jz_face_s], dtype=np.float64
        )

        self.point_source_count = 0
        self.point_source_softening = 0.0
        self.point_src_x_GPU = cupy.zeros(1, dtype=np.float64)
        self.point_src_y_GPU = cupy.zeros(1, dtype=np.float64)
        self.point_src_z_GPU = cupy.zeros(1, dtype=np.float64)
        self.point_source_weights_GPU = cupy.ones(1, dtype=np.float64)
        # Optional monopole surface-charge history (floating-conductor branch).
        self.point_rho_GPU = cupy.zeros([self.len_time_snapshots, 1], dtype=np.float64)
        # Step-2B continuity-consistent electric-dipole histories.  P [Q/L]
        # and K=dP/dt [I/L] are surface densities; point_weights [L^2] convert
        # them to lumped p [Q L] and m [I L].
        self.point_Px_GPU = cupy.zeros([self.len_time_snapshots, 1], dtype=np.float64)
        self.point_Py_GPU = cupy.zeros([self.len_time_snapshots, 1], dtype=np.float64)
        self.point_Pz_GPU = cupy.zeros([self.len_time_snapshots, 1], dtype=np.float64)
        self.point_Kx_GPU = cupy.zeros([self.len_time_snapshots, 1], dtype=np.float64)
        self.point_Ky_GPU = cupy.zeros([self.len_time_snapshots, 1], dtype=np.float64)
        self.point_Kz_GPU = cupy.zeros([self.len_time_snapshots, 1], dtype=np.float64)
        # One zero history is shared by the legacy monopole point-source kernel
        # as its three dummy current arguments; no legacy point-current history
        # is allocated.
        self.point_zero_GPU = cupy.zeros([self.len_time_snapshots, 1], dtype=np.float64)
        self.point_dipole_history_abs_steps = -np.ones(self.len_time_snapshots, dtype=np.int64)
        self.point_rho_history_abs_steps = -np.ones(self.len_time_snapshots, dtype=np.int64)
        self._sample_point_x_GPU = cupy.zeros(1, dtype=np.float64)
        self._sample_point_y_GPU = cupy.zeros(1, dtype=np.float64)
        self._sample_point_z_GPU = cupy.zeros(1, dtype=np.float64)
        self._sample_point_count = 0

        self.coeff_E = 1.0 / (4.0 * math.pi * self.epsilon0) * self.dx_s * self.dy_s * self.dz_s
        self.coeff_B = -1.0 / (4.0 * math.pi * self.epsilon0 * self.c ** 2) * self.dx_s * self.dy_s * self.dz_s
        self.time_tick = 0
        self.time_basis_name = 'piecewise_linear_nodal'
        self.boundary_mode = getattr(self, 'boundary_mode', 'open')
        # Radius <= 0 disables spatial truncation.  History truncation is always
        # active in the CUDA kernels to avoid circular-buffer aliasing.
        self.grid_truncation_radius = 0.0
        self.point_truncation_radius = 0.0
        self.grid_taper_width = 0.0
        self.point_taper_width = 0.0
        self.grid_taper_type = 0
        self.point_taper_type = 0
        self._warned_history_capacity = False
        # Production grid-source backend.
        #
        # Atomic and observation-parallel are mathematically equivalent
        # accumulation strategies for the same Jefimenko discretization.  The
        # article-scale H100 benchmark with index-based self-cell exclusion
        # measured the Atomic path as the faster backend, so Atomic is the
        # production default.  Observation remains available for parity tests.
        self.kernel_mode = 'atomic'
        self._warn_if_history_capacity_short()

    def _init_from_legacy_signature(self,
                                    len_time_snapshots,
                                    x_grid_size_o, y_grid_size_o, z_grid_size_o,
                                    x_grid_size_s, y_grid_size_s, z_grid_size_s,
                                    dx_o, dy_o, dz_o, x_left_boundary_o, y_left_boundary_o, z_left_boundary_o,
                                    dx_s, dy_s, dz_s, x_left_boundary_s, y_left_boundary_s, z_left_boundary_s,
                                    dt, epsilon0, c, is_electrostatic, boundary_mode='open'):
        self.len_time_snapshots = int(len_time_snapshots)
        self.x_grid_size_o, self.y_grid_size_o, self.z_grid_size_o = int(x_grid_size_o), int(y_grid_size_o), int(z_grid_size_o)
        self.x_grid_size_s, self.y_grid_size_s, self.z_grid_size_s = int(x_grid_size_s), int(y_grid_size_s), int(z_grid_size_s)
        self.dx_o, self.dy_o, self.dz_o = float(dx_o), float(dy_o), float(dz_o)
        self.dx_s, self.dy_s, self.dz_s = float(dx_s), float(dy_s), float(dz_s)
        self.x_left_boundary_o, self.y_left_boundary_o, self.z_left_boundary_o = float(x_left_boundary_o), float(y_left_boundary_o), float(z_left_boundary_o)
        self.x_left_boundary_s, self.y_left_boundary_s, self.z_left_boundary_s = float(x_left_boundary_s), float(y_left_boundary_s), float(z_left_boundary_s)
        self.dt = float(dt)
        self.epsilon0 = float(epsilon0)
        self.c = float(c)
        self.is_electrostatic = bool(is_electrostatic)
        self.boundary_mode = str(boundary_mode).lower()
        if self.boundary_mode not in BOUNDARY_MODE_TO_INT:
            raise ValueError(f"Unsupported EM boundary mode: {self.boundary_mode}")

    def _init_from_region_keywords(self, **kwargs):
        source_size = np.asarray(kwargs.pop('source_size'), dtype=np.float64)
        source_grid = np.asarray(kwargs.pop('source_grid'), dtype=np.int32)
        observation_size = np.asarray(kwargs.pop('observation_size', source_size), dtype=np.float64)
        observation_grid = np.asarray(kwargs.pop('observation_grid', source_grid), dtype=np.int32)

        self.len_time_snapshots = int(kwargs.pop('len_time_snapshots'))
        self.c = float(kwargs.pop('c'))
        self.dt = float(kwargs.pop('dt'))
        self.epsilon0 = float(kwargs.pop('epsilon0', 8.854187817e-12))
        self.is_electrostatic = bool(kwargs.pop('is_electrostatic', False))
        self.boundary_mode = str(kwargs.pop('boundary_mode', 'open')).lower()
        if self.boundary_mode not in BOUNDARY_MODE_TO_INT:
            raise ValueError(f"Unsupported EM boundary mode: {self.boundary_mode}")

        x_left_generic = float(kwargs.pop('x_left_bound', kwargs.pop('x_left_boundary', 0.0)))
        y_left_generic = float(kwargs.pop('y_left_bound', kwargs.pop('y_left_boundary', 0.0)))
        z_left_generic = float(kwargs.pop('z_left_bound', kwargs.pop('z_left_boundary', 0.0)))

        self.x_left_boundary_o = float(kwargs.pop('x_left_bound_o', kwargs.pop('x_left_boundary_o', x_left_generic)))
        self.y_left_boundary_o = float(kwargs.pop('y_left_bound_o', kwargs.pop('y_left_boundary_o', y_left_generic)))
        self.z_left_boundary_o = float(kwargs.pop('z_left_bound_o', kwargs.pop('z_left_boundary_o', z_left_generic)))
        self.x_left_boundary_s = float(kwargs.pop('x_left_bound_s', kwargs.pop('x_left_boundary_s', x_left_generic)))
        self.y_left_boundary_s = float(kwargs.pop('y_left_bound_s', kwargs.pop('y_left_boundary_s', y_left_generic)))
        self.z_left_boundary_s = float(kwargs.pop('z_left_bound_s', kwargs.pop('z_left_boundary_s', z_left_generic)))

        self.x_grid_size_o, self.y_grid_size_o, self.z_grid_size_o = map(int, observation_grid.tolist())
        self.x_grid_size_s, self.y_grid_size_s, self.z_grid_size_s = map(int, source_grid.tolist())
        self.dx_o, self.dy_o, self.dz_o = (observation_size / observation_grid).tolist()
        self.dx_s, self.dy_s, self.dz_s = (source_size / source_grid).tolist()

        if kwargs:
            unknown = ', '.join(sorted(kwargs.keys()))
            raise TypeError(f'Unknown EMsolver keyword arguments: {unknown}')

    def set_grid_current_layout(self, layout):
        """Validate the production staggered-face grid-current layout.

        The legacy cell-centered PIC-current path has been retired.  This
        method is kept only as a compatibility/validation hook for case setup;
        only staggered-face aliases are accepted.
        """
        key = str(layout).strip().lower()
        aliases = {
            'face', 'faces', 'staggered', 'staggered_face', 'staggered-face'
        }
        if key not in aliases:
            raise ValueError(
                "The legacy cell-centered grid-current path has been removed; "
                "use grid_current_layout='staggered_face'."
            )
        if self.boundary_mode != 'open':
            raise NotImplementedError(
                "Staggered-face Jefimenko grid currents currently support open EM boundaries only."
            )
        self.grid_current_layout = 'staggered_face'
        # Current layout and GPU reduction strategy are independent choices.
        # Do not silently change kernel_mode here.
        return self

    def get_grid_current_layout(self):
        return self.grid_current_layout

    def get_grid_backend_metadata(self):
        """Return implementation metadata without changing the physical model."""
        return {
            'grid_current_layout': self.grid_current_layout,
            'kernel_mode': str(getattr(self, 'kernel_mode', 'atomic')).lower(),
            'quasi_electrostatic': bool(self.is_electrostatic),
            'full_em': bool(not self.is_electrostatic),
            'charge_current_deposition_owned_by_particle_transport': True,
            'self_cell_exclusion': 'index_identity_on_coincident_cell_center_grids',
        }

    def push_staggered_grid_currents(self, face_currents, i_time=None):
        """Store one native-face plasma-current snapshot.

        ``face_currents`` must be ``(Jx_face, Jy_face, Jz_face)`` with shapes
        ``(nx+1,ny,nz)``, ``(nx,ny+1,nz)``, and ``(nx,ny,nz+1)``.
        Values are current densities in the same code units as the legacy
        cell-centered J arrays.
        """
        if self.grid_current_layout != 'staggered_face':
            raise RuntimeError(
                "push_staggered_grid_currents requires "
                "set_grid_current_layout('staggered_face')"
            )
        if face_currents is None or len(face_currents) != 3:
            raise ValueError('face_currents must be a 3-tuple (Jx_face, Jy_face, Jz_face)')
        if i_time is None:
            if not hasattr(self, 'i_time'):
                raise RuntimeError('No current EM history slot is active')
            i_time = self.i_time
        expected = (
            (self.x_grid_size_s + 1, self.y_grid_size_s, self.z_grid_size_s),
            (self.x_grid_size_s, self.y_grid_size_s + 1, self.z_grid_size_s),
            (self.x_grid_size_s, self.y_grid_size_s, self.z_grid_size_s + 1),
        )
        arrays = []
        for arr, shape, name in zip(face_currents, expected, ('Jx_face', 'Jy_face', 'Jz_face')):
            a = cupy.asarray(arr, dtype=np.float64)
            if tuple(a.shape) != shape:
                raise ValueError(f'{name} has shape {tuple(a.shape)}, expected {shape}')
            arrays.append(a.reshape(-1))
        self.Jx_face_GPU[i_time, :] = arrays[0]
        self.Jy_face_GPU[i_time, :] = arrays[1]
        self.Jz_face_GPU[i_time, :] = arrays[2]
        return self

    def set_boundary_mode(self, mode):
        mode = str(mode).lower()
        if mode not in BOUNDARY_MODE_TO_INT:
            raise ValueError(f"Unsupported EM boundary mode: {mode}")
        if mode != 'open':
            raise NotImplementedError(
                "Step 1-D staggered-face Jefimenko current path currently supports open EM boundaries only."
            )
        self.boundary_mode = mode

    def set_kernel_mode(self, mode):
        """Select the Jefimenko grid-source accumulation strategy.

        ``atomic`` uses source/observation-pair CUDA atomic additions.
        ``observation`` assigns one CUDA thread to each observation point and
        accumulates source contributions locally.

        Both strategies use the same native staggered-face current histories and
        the same retarded Jefimenko formulation. The choice is therefore a GPU
        accumulation strategy, not a physical-model switch. Full-EM operation
        still requires ``is_electrostatic=False`` and open boundaries for the
        current staggered-face implementation.
        """
        key = str(mode).strip().lower()
        if key in ('atomic', 'atomic_add', 'source_parallel'):
            self.kernel_mode = 'atomic'
        elif key in ('obs', 'observation_parallel', 'observation'):
            self.kernel_mode = 'observation'
        else:
            raise ValueError("kernel_mode must be 'atomic' or 'observation'")
        return self


    def _taper_mode_to_int(self, mode):
        key = None if mode is None else str(mode).lower()
        if key not in TAPER_MODE_TO_INT:
            raise ValueError(f"Unsupported taper mode: {mode}. Use one of: hard, none, hann, smoothstep, exponential.")
        return int(TAPER_MODE_TO_INT[key])

    def set_spatial_truncation_radius(self, grid_radius=None, point_radius=None, common_radius=None,
                                      taper=None, taper_width=None,
                                      grid_taper=None, point_taper=None,
                                      grid_taper_width=None, point_taper_width=None):
        """Configure optional distance cutoffs for integral accumulation.

        A radius <= 0 or ``None`` disables the corresponding cutoff.  The grid
        cutoff applies to grid-source contributions; the point cutoff applies to
        MAS/antenna auxiliary point sources.  This is an approximation and is
        therefore disabled by default.

        ``taper`` may be ``'hard'``/``'none'`` (default), ``'hann'``,
        ``'smoothstep'``, or ``'exponential'``.  A positive taper width applies
        the window only in the outer shell ``[radius - taper_width, radius]``.
        If a smooth taper is requested and no width is supplied, the width
        defaults to 20% of the corresponding cutoff radius.
        """
        if common_radius is not None:
            grid_radius = common_radius
            point_radius = common_radius
        if grid_radius is not None:
            self.grid_truncation_radius = max(0.0, float(grid_radius))
        if point_radius is not None:
            self.point_truncation_radius = max(0.0, float(point_radius))

        if taper is not None:
            grid_taper = taper
            point_taper = taper
        if taper_width is not None:
            grid_taper_width = taper_width
            point_taper_width = taper_width
        if grid_taper is not None:
            self.grid_taper_type = self._taper_mode_to_int(grid_taper)
        if point_taper is not None:
            self.point_taper_type = self._taper_mode_to_int(point_taper)
        if grid_taper_width is not None:
            self.grid_taper_width = max(0.0, float(grid_taper_width))
        if point_taper_width is not None:
            self.point_taper_width = max(0.0, float(point_taper_width))

        # Smooth tapers need a finite shoulder.  Treat taper='hann' or similar
        # without an explicit width as a request for a conservative default.
        if grid_taper is not None and self.grid_taper_type > 0 and grid_taper_width is None and self.grid_truncation_radius > 0.0:
            self.grid_taper_width = 0.20 * self.grid_truncation_radius
        if point_taper is not None and self.point_taper_type > 0 and point_taper_width is None and self.point_truncation_radius > 0.0:
            self.point_taper_width = 0.20 * self.point_truncation_radius
        return self

    def history_horizon_distance(self):
        """Maximum retarded distance represented by the circular history buffer."""
        return max(0.0, (self.len_time_snapshots - 1) * self.dt * self.c)

    def _region_center_bounds(self, nx, ny, nz, dx, dy, dz, x0, y0, z0):
        mins = np.array([x0 + 0.5 * dx, y0 + 0.5 * dy, z0 + 0.5 * dz], dtype=np.float64)
        maxs = np.array([x0 + (nx - 0.5) * dx, y0 + (ny - 0.5) * dy, z0 + (nz - 0.5) * dz], dtype=np.float64)
        return mins, maxs

    def _max_distance_between_boxes(self, a_min, a_max, b_min, b_max):
        sep = np.maximum(np.abs(a_min - b_max), np.abs(a_max - b_min))
        return float(np.linalg.norm(sep))

    def _max_distance_points_to_box(self, points, b_min, b_max):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(points) == 0:
            return 0.0
        sep = np.maximum(np.abs(points - b_min[None, :]), np.abs(points - b_max[None, :]))
        return float(np.max(np.linalg.norm(sep, axis=1)))

    def estimate_required_history_distance(self, include_antenna=True):
        """Estimate the largest open-boundary retarded distance in the solve."""
        obs_min, obs_max = self._region_center_bounds(
            self.x_grid_size_o, self.y_grid_size_o, self.z_grid_size_o,
            self.dx_o, self.dy_o, self.dz_o,
            self.x_left_boundary_o, self.y_left_boundary_o, self.z_left_boundary_o,
        )
        src_min, src_max = self._region_center_bounds(
            self.x_grid_size_s, self.y_grid_size_s, self.z_grid_size_s,
            self.dx_s, self.dy_s, self.dz_s,
            self.x_left_boundary_s, self.y_left_boundary_s, self.z_left_boundary_s,
        )
        required = self._max_distance_between_boxes(obs_min, obs_max, src_min, src_max)
        if include_antenna and getattr(self, "point_source_count", 0) > 0:
            points = np.column_stack([
                self.point_src_x_GPU.get(),
                self.point_src_y_GPU.get(),
                self.point_src_z_GPU.get(),
            ])
            required = max(required, self._max_distance_points_to_box(points, obs_min, obs_max))
        return required

    def validate_history_capacity(self, include_antenna=True, action="warn"):
        """Check whether the ring buffer covers the largest retarded distance.

        The CUDA kernels now skip unavailable history instead of aliasing modulo
        slots.  This method tells the user when that truncation will occur and
        how many time snapshots would be needed for full open-boundary coverage.
        """
        horizon = self.history_horizon_distance()
        required = self.estimate_required_history_distance(include_antenna=include_antenna)
        # If an explicit spatial cutoff is active and smaller than the domain,
        # the effective required history is only the cutoff radius.
        active_cutoffs = []
        if self.grid_truncation_radius > 0.0:
            active_cutoffs.append(self.grid_truncation_radius)
        if include_antenna and self.point_truncation_radius > 0.0:
            active_cutoffs.append(self.point_truncation_radius)
        if active_cutoffs:
            required = min(required, max(active_cutoffs))
        needed = int(math.ceil(required / max(self.c * self.dt, 1.0e-300))) + 1
        ok = horizon + 1.0e-12 >= required
        info = {
            "ok": bool(ok),
            "history_horizon_distance": float(horizon),
            "estimated_required_distance": float(required),
            "len_time_snapshots": int(self.len_time_snapshots),
            "recommended_len_time_snapshots": int(max(self.len_time_snapshots, needed)),
            "grid_truncation_radius": float(self.grid_truncation_radius),
            "point_truncation_radius": float(self.point_truncation_radius),
            "grid_taper_width": float(self.grid_taper_width),
            "point_taper_width": float(self.point_taper_width),
            "grid_taper_type": int(self.grid_taper_type),
            "point_taper_type": int(self.point_taper_type),
        }
        if not ok:
            msg = (
                "Jefimenko history buffer is shorter than the estimated retarded "
                f"distance ({horizon:.6g} < {required:.6g}). Contributions older "
                "than the circular buffer are now truncated to avoid modulo aliasing. "
                f"Use len_time_snapshots >= {needed} for full open-boundary history, "
                "or set a physically justified spatial truncation radius."
            )
            if action == "raise":
                raise ValueError(msg)
            if action == "warn":
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return info

    def _warn_if_history_capacity_short(self):
        if not getattr(self, "_warned_history_capacity", False):
            self.validate_history_capacity(include_antenna=False, action="warn")
            self._warned_history_capacity = True

    def set_antenna_sources(self, src_points, source_weights=None, softening_distance=0.0):
        """Register analytic MAS/antenna point sources for the hybrid kernel.

        ``src_points`` has shape ``(N, 3)`` and ``source_weights`` contains patch
        areas. Step-2B stores continuity-linked P/K electric-dipole histories;
        the separate point-rho channel is retained for monopole conductor charge.
        """
        src_points = np.asarray(src_points, dtype=np.float64).reshape(-1, 3)
        if len(src_points) == 0:
            self.clear_antenna_sources()
            return
        if source_weights is None:
            source_weights = np.ones(len(src_points), dtype=np.float64)
        source_weights = np.asarray(source_weights, dtype=np.float64).reshape(-1,)
        if len(source_weights) != len(src_points):
            raise ValueError('source_weights must have the same length as src_points')
        self.point_source_count = int(len(src_points))
        self.point_source_softening = float(max(0.0, softening_distance))
        self.point_src_x_GPU = cupy.asarray(src_points[:, 0], dtype=np.float64)
        self.point_src_y_GPU = cupy.asarray(src_points[:, 1], dtype=np.float64)
        self.point_src_z_GPU = cupy.asarray(src_points[:, 2], dtype=np.float64)
        self.point_source_weights_GPU = cupy.asarray(source_weights, dtype=np.float64)
        shape = [self.len_time_snapshots, self.point_source_count]
        self.point_rho_GPU = cupy.zeros(shape, dtype=np.float64)
        self.point_Px_GPU = cupy.zeros(shape, dtype=np.float64)
        self.point_Py_GPU = cupy.zeros(shape, dtype=np.float64)
        self.point_Pz_GPU = cupy.zeros(shape, dtype=np.float64)
        self.point_Kx_GPU = cupy.zeros(shape, dtype=np.float64)
        self.point_Ky_GPU = cupy.zeros(shape, dtype=np.float64)
        self.point_Kz_GPU = cupy.zeros(shape, dtype=np.float64)
        self.point_zero_GPU = cupy.zeros(shape, dtype=np.float64)
        self.point_dipole_history_abs_steps = -np.ones(self.len_time_snapshots, dtype=np.int64)
        self.point_rho_history_abs_steps = -np.ones(self.len_time_snapshots, dtype=np.int64)
        self.validate_history_capacity(include_antenna=True, action="warn")

    def clear_antenna_sources(self):
        self.point_source_count = 0
        self.point_source_softening = 0.0
        self.point_src_x_GPU = cupy.zeros(1, dtype=np.float64)
        self.point_src_y_GPU = cupy.zeros(1, dtype=np.float64)
        self.point_src_z_GPU = cupy.zeros(1, dtype=np.float64)
        self.point_source_weights_GPU = cupy.ones(1, dtype=np.float64)
        shape = [self.len_time_snapshots, 1]
        self.point_rho_GPU = cupy.zeros(shape, dtype=np.float64)
        self.point_Px_GPU = cupy.zeros(shape, dtype=np.float64)
        self.point_Py_GPU = cupy.zeros(shape, dtype=np.float64)
        self.point_Pz_GPU = cupy.zeros(shape, dtype=np.float64)
        self.point_Kx_GPU = cupy.zeros(shape, dtype=np.float64)
        self.point_Ky_GPU = cupy.zeros(shape, dtype=np.float64)
        self.point_Kz_GPU = cupy.zeros(shape, dtype=np.float64)
        self.point_zero_GPU = cupy.zeros(shape, dtype=np.float64)
        self.point_dipole_history_abs_steps = -np.ones(self.len_time_snapshots, dtype=np.int64)
        self.point_rho_history_abs_steps = -np.ones(self.len_time_snapshots, dtype=np.int64)

    def _current_absolute_source_step(self):
        return max(int(getattr(self, 'time_tick', 0)) - 1, 0)

    def push_antenna_charges(self, surface_charge_density, i_time=None):
        """Push conductor/MAS surface charge density into the point-source history.

        The registered point weight has units of source-patch area, so
        ``surface_charge_density * source_weights`` is the charge represented
        by each auxiliary point.
        """
        if self.point_source_count <= 0:
            raise RuntimeError('No antenna point sources registered. Call set_antenna_sources first.')
        if i_time is None:
            i_time = getattr(self, 'i_time', time_index_in_GPU(max(self.time_tick - 1, 0), self.len_time_snapshots))
        charge = cupy.asarray(surface_charge_density, dtype=np.float64).reshape(self.point_source_count)
        self.point_rho_GPU[i_time, :] = charge
        self.point_rho_history_abs_steps[int(i_time)] = self._current_absolute_source_step()

    def push_antenna_dipole_state(self, polarization_density, surface_current_density, i_time=None):
        """Write Step-2B P [Q/L] and K=dP/dt [I/L] at one EM history node."""
        if self.point_source_count <= 0:
            raise RuntimeError('No antenna point sources registered. Call set_antenna_sources first.')
        P = cupy.asarray(polarization_density, dtype=np.float64).reshape(self.point_source_count, 3)
        K = cupy.asarray(surface_current_density, dtype=np.float64).reshape(self.point_source_count, 3)
        if i_time is None:
            i_time = time_index_in_GPU(self.time_tick, self.len_time_snapshots)
        self.point_Px_GPU[i_time, :] = P[:, 0]
        self.point_Py_GPU[i_time, :] = P[:, 1]
        self.point_Pz_GPU[i_time, :] = P[:, 2]
        self.point_Kx_GPU[i_time, :] = K[:, 0]
        self.point_Ky_GPU[i_time, :] = K[:, 1]
        self.point_Kz_GPU[i_time, :] = K[:, 2]
        self.point_dipole_history_abs_steps[int(i_time)] = self._current_absolute_source_step()

    def push_antenna_currents(self, *args, **kwargs):
        raise RuntimeError(
            'Legacy MAS point-current sources were removed in Step 2-B. '
            'Use push_antenna_dipole_state(P, K) with continuity-linked electric-dipole sources.'
        )


    @staticmethod
    def _point_weighted_rms(vectors, weights):
        vectors = np.asarray(vectors, dtype=np.float64)
        weights = np.abs(np.asarray(weights, dtype=np.float64).reshape(-1))
        if vectors.ndim == 1:
            values2 = vectors * vectors
        else:
            values2 = np.sum(vectors * vectors, axis=-1)
        wsum = max(float(np.sum(weights)), 1.0e-300)
        return float(np.sqrt(np.sum(weights * values2) / wsum))

    @staticmethod
    def _point_max_norm(vectors):
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim == 1:
            return float(np.max(np.abs(vectors))) if vectors.size else 0.0
        return float(np.max(np.linalg.norm(vectors, axis=-1))) if len(vectors) else 0.0

    def get_point_source_closure_diagnostics(self, absolute_step=None):
        """Audit P/K histories consumed by the Step-2B point-dipole kernel."""
        if self.point_source_count <= 0:
            return {'available': False, 'reason': 'no point sources registered'}
        if absolute_step is None:
            absolute_step = self._current_absolute_source_step()
        absolute_step = int(absolute_step)
        slot = absolute_step % self.len_time_snapshots
        weights = cupy.asnumpy(self.point_source_weights_GPU).reshape(-1)
        P = np.stack([
            cupy.asnumpy(self.point_Px_GPU[slot]), cupy.asnumpy(self.point_Py_GPU[slot]), cupy.asnumpy(self.point_Pz_GPU[slot])
        ], axis=1)
        K = np.stack([
            cupy.asnumpy(self.point_Kx_GPU[slot]), cupy.asnumpy(self.point_Ky_GPU[slot]), cupy.asnumpy(self.point_Kz_GPU[slot])
        ], axis=1)
        rho = cupy.asnumpy(self.point_rho_GPU[slot]).reshape(-1)
        dipole_written = bool(self.point_dipole_history_abs_steps[slot] == absolute_step)
        rho_written = bool(self.point_rho_history_abs_steps[slot] == absolute_step)
        out = {
            'available': True,
            'absolute_step': absolute_step,
            'history_slot': int(slot),
            'dipole_state_written_for_step': dipole_written,
            'rho_monopole_written_for_step': rho_written,
            'P_weighted_rms_code_Q_per_L': self._point_weighted_rms(P, weights),
            'K_weighted_rms_code_I_per_L': self._point_weighted_rms(K, weights),
            'rho_monopole_weighted_rms_code': self._point_weighted_rms(rho, weights),
            'temporal_continuity_available': False,
            'temporal_continuity_residual_weighted_rms': float('nan'),
            'Kdot_weighted_rms_code': float('nan'),
            'local_vs_pass2_Kdot_relative_rms': float('nan'),
            'point_dipole_E_terms_enabled': True,
            'source_model': 'electric dipole p=wP, m=wK with K=dP/dt',
        }
        prev_abs = absolute_step - 1
        if prev_abs >= 0:
            prev_slot = prev_abs % self.len_time_snapshots
            if self.point_dipole_history_abs_steps[prev_slot] == prev_abs and dipole_written:
                Pprev = np.stack([
                    cupy.asnumpy(self.point_Px_GPU[prev_slot]), cupy.asnumpy(self.point_Py_GPU[prev_slot]), cupy.asnumpy(self.point_Pz_GPU[prev_slot])
                ], axis=1)
                Kprev = np.stack([
                    cupy.asnumpy(self.point_Kx_GPU[prev_slot]), cupy.asnumpy(self.point_Ky_GPU[prev_slot]), cupy.asnumpy(self.point_Kz_GPU[prev_slot])
                ], axis=1)
                residual = (P - Pprev) / self.dt - 0.5 * (Kprev + K)
                Kdot = (K - Kprev) / self.dt
                out['temporal_continuity_available'] = True
                out['temporal_continuity_residual_weighted_rms'] = self._point_weighted_rms(residual, weights)
                out['temporal_continuity_residual_max'] = self._point_max_norm(residual)
                out['Kdot_weighted_rms_code'] = self._point_weighted_rms(Kdot, weights)
                out['local_vs_pass2_Kdot_relative_rms'] = 0.0
        return out


    def _boundary_image_extent(self):
        if self.boundary_mode == 'open':
            return 0, 0, 0
        Lx = max(self.dx_s * self.x_grid_size_s, 1.0e-30)
        Ly = max(self.dy_s * self.y_grid_size_s, 1.0e-30)
        Lz = max(self.dz_s * self.z_grid_size_s, 1.0e-30)
        memory_horizon = min(self.t, max(0.0, (self.len_time_snapshots - 1) * self.dt))
        reach = max(0.0, memory_horizon * self.c)
        return (int(math.ceil(reach / Lx)) + 1,
                int(math.ceil(reach / Ly)) + 1,
                int(math.ceil(reach / Lz)) + 1)

    def _run_jefimenko_kernel(self, include_antenna=True):
        """Run the field kernel for the already-loaded source histories.

        This method deliberately does not advance ``time_tick``.  It is used by
        both the normal time-marching solve and the MAS two-pass workflow:
        first grid-only for the incident plasma field, then grid+antenna after
        the MAS current has been solved for the same time node.
        """
        GEx, GEy, GEz, GBx, GBy, GBz = (cupy.zeros(self.total_grid_o, dtype=np.float64) for _ in range(6))
        max_img_x, max_img_y, max_img_z = self._boundary_image_extent()
        active_point_count = self.point_source_count if include_antenna else 0
        kernel_mode = str(getattr(self, 'kernel_mode', 'atomic')).lower()
        if kernel_mode == 'atomic':
            # Source/observation-pair kernel with atomic accumulation. Grid
            # charge and d(rho)/dt remain cell centered; native staggered-face
            # current terms are added by a separate atomic kernel below.
            Jefimenko_kernel[self.blockspergrid_all, self.threadsperblock](
                self.rho_GPU, self.grid_zero_GPU, self.grid_zero_GPU, self.grid_zero_GPU,
                self.len_time_snapshots, self.i_time, self.dt, self.t,
                self.total_grid_o, self.total_grid_s,
                self.x_grid_size_o, self.y_grid_size_o, self.z_grid_size_o,
                self.x_grid_size_s, self.y_grid_size_s, self.z_grid_size_s,
                self.dx_o, self.dy_o, self.dz_o,
                self.dx_s, self.dy_s, self.dz_s,
                self.x_left_boundary_o, self.y_left_boundary_o, self.z_left_boundary_o,
                self.x_left_boundary_s, self.y_left_boundary_s, self.z_left_boundary_s,
                GEx, GEy, GEz, GBx, GBy, GBz, self.coeff_E, self.coeff_B, self.c, self.all_grids, self.is_electrostatic,
                BOUNDARY_MODE_TO_INT[self.boundary_mode],
                self.dx_s * self.x_grid_size_s, self.dy_s * self.y_grid_size_s, self.dz_s * self.z_grid_size_s,
                max_img_x, max_img_y, max_img_z,
                self.point_src_x_GPU, self.point_src_y_GPU, self.point_src_z_GPU, self.point_source_weights_GPU,
                self.point_rho_GPU, self.point_zero_GPU, self.point_zero_GPU, self.point_zero_GPU,
                active_point_count, self.point_source_softening,
                self.grid_truncation_radius, self.point_truncation_radius,
                self.grid_taper_width, self.point_taper_width,
                self.grid_taper_type, self.point_taper_type
            )
        elif kernel_mode == 'observation':
            # Charge and optional monopole point-source terms.
            Jefimenko_kernel_observation_parallel[self.blockspergrid_o, self.threadsperblock](
                self.rho_GPU, self.rho_GPU, self.rho_GPU, self.rho_GPU,
                self.len_time_snapshots, self.i_time, self.dt, self.t,
                self.total_grid_o, self.total_grid_s,
                self.x_grid_size_o, self.y_grid_size_o, self.z_grid_size_o,
                self.x_grid_size_s, self.y_grid_size_s, self.z_grid_size_s,
                self.dx_o, self.dy_o, self.dz_o,
                self.dx_s, self.dy_s, self.dz_s,
                self.x_left_boundary_o, self.y_left_boundary_o, self.z_left_boundary_o,
                self.x_left_boundary_s, self.y_left_boundary_s, self.z_left_boundary_s,
                GEx, GEy, GEz, GBx, GBy, GBz, self.coeff_E, self.coeff_B, self.c, self.all_grids, self.is_electrostatic,
                0,
                BOUNDARY_MODE_TO_INT[self.boundary_mode],
                self.dx_s * self.x_grid_size_s, self.dy_s * self.y_grid_size_s, self.dz_s * self.z_grid_size_s,
                max_img_x, max_img_y, max_img_z,
                self.point_src_x_GPU, self.point_src_y_GPU, self.point_src_z_GPU, self.point_source_weights_GPU,
                self.point_rho_GPU, self.point_zero_GPU, self.point_zero_GPU, self.point_zero_GPU,
                active_point_count, self.point_source_softening,
                self.grid_truncation_radius, self.point_truncation_radius,
                self.grid_taper_width, self.point_taper_width,
                self.grid_taper_type, self.point_taper_type
            )
        else:
            raise RuntimeError(f"unknown kernel_mode={kernel_mode!r}")

        # Step-2B electric-dipole MAS source contribution.  Surface-charge
        # monopoles (Case 2) were handled by the main kernel above; this kernel
        # adds the continuity-linked P/K dipole source without any legacy point J.
        if active_point_count > 0:
            Jefimenko_point_dipole_kernel_observation_parallel[
                self.blockspergrid_o, self.threadsperblock
            ](
                self.point_src_x_GPU, self.point_src_y_GPU, self.point_src_z_GPU,
                self.point_source_weights_GPU,
                self.point_Px_GPU, self.point_Py_GPU, self.point_Pz_GPU,
                self.point_Kx_GPU, self.point_Ky_GPU, self.point_Kz_GPU,
                active_point_count, self.point_source_softening,
                self.point_truncation_radius, self.point_taper_width, self.point_taper_type,
                self.len_time_snapshots, self.dt, self.t,
                self.dx_s, self.dy_s, self.dz_s,
                self.total_grid_o,
                self.x_grid_size_o, self.y_grid_size_o, self.z_grid_size_o,
                self.dx_o, self.dy_o, self.dz_o,
                self.x_left_boundary_o, self.y_left_boundary_o, self.z_left_boundary_o,
                GEx, GEy, GEz, GBx, GBy, GBz,
                self.c, self.is_electrostatic,
            )

        # Full-EM current-dependent terms use native staggered faces. In the
        # quasi-electrostatic article path there are no J or dJ/dt field terms.
        if not self.is_electrostatic:
            if self.boundary_mode != 'open':
                raise NotImplementedError(
                    "staggered-face grid currents currently support open EM boundaries only"
                )

            if kernel_mode == 'observation':
                Jefimenko_staggered_current_kernel_observation_parallel[
                    self.blockspergrid_o, self.threadsperblock
                ](
                    self.Jx_face_GPU, self.Jy_face_GPU, self.Jz_face_GPU,
                    self.len_time_snapshots, self.dt, self.t,
                    self.x_grid_size_s, self.y_grid_size_s, self.z_grid_size_s,
                    self.total_grid_o,
                    self.x_grid_size_o, self.y_grid_size_o, self.z_grid_size_o,
                    self.dx_o, self.dy_o, self.dz_o,
                    self.dx_s, self.dy_s, self.dz_s,
                    self.x_left_boundary_o, self.y_left_boundary_o, self.z_left_boundary_o,
                    self.x_left_boundary_s, self.y_left_boundary_s, self.z_left_boundary_s,
                    GEx, GEy, GEz, GBx, GBy, GBz,
                    self.c, self.is_electrostatic,
                    self.grid_truncation_radius, self.grid_taper_width, self.grid_taper_type,
                )
            elif kernel_mode == 'atomic':
                total_faces = (
                    self.total_Jx_face_s
                    + self.total_Jy_face_s
                    + self.total_Jz_face_s
                )
                all_face_pairs = self.total_grid_o * total_faces
                blocks_face_pairs = (
                    all_face_pairs + (self.threadsperblock - 1)
                ) // self.threadsperblock
                Jefimenko_staggered_current_kernel_atomic[
                    blocks_face_pairs, self.threadsperblock
                ](
                    self.Jx_face_GPU, self.Jy_face_GPU, self.Jz_face_GPU,
                    self.len_time_snapshots, self.dt, self.t,
                    self.x_grid_size_s, self.y_grid_size_s, self.z_grid_size_s,
                    self.total_grid_o,
                    self.x_grid_size_o, self.y_grid_size_o, self.z_grid_size_o,
                    self.dx_o, self.dy_o, self.dz_o,
                    self.dx_s, self.dy_s, self.dz_s,
                    self.x_left_boundary_o, self.y_left_boundary_o, self.z_left_boundary_o,
                    self.x_left_boundary_s, self.y_left_boundary_s, self.z_left_boundary_s,
                    GEx, GEy, GEz, GBx, GBy, GBz,
                    self.c,
                    self.grid_truncation_radius, self.grid_taper_width, self.grid_taper_type,
                )
            else:
                raise RuntimeError(f"unknown kernel_mode={kernel_mode!r}")

        self.GEx, self.GEy, self.GEz = GEx * self.coeff_E, GEy * self.coeff_E, GEz * self.coeff_E
        self.GBx, self.GBy, self.GBz = GBx * self.coeff_B, GBy * self.coeff_B, GBz * self.coeff_B

    def Jefimenko_solver(self, rho, antenna_surface_currents=None, antenna_polarization=None, antenna_charges=None,
                         include_antenna=True, face_currents=None, antenna_currents=None):
        self.i_time = time_index_in_GPU(self.time_tick, self.len_time_snapshots)
        self.t = self.time_tick * self.dt
        self.time_tick += 1

        self.rho_GPU[self.i_time] = rho
        if face_currents is None:
            raise ValueError(
                "face_currents=(Jx_face,Jy_face,Jz_face) is required; "
                "the legacy cell-centered grid-current path has been removed"
            )
        self.push_staggered_grid_currents(face_currents, i_time=self.i_time)

        if antenna_charges is not None:
            self.push_antenna_charges(antenna_charges, i_time=self.i_time)
        if antenna_currents is not None:
            raise RuntimeError('antenna_currents legacy point-current API was removed in Step 2-B')
        if (antenna_surface_currents is None) != (antenna_polarization is None):
            raise ValueError('Provide both antenna_surface_currents=K and antenna_polarization=P, or neither')
        if antenna_surface_currents is not None:
            self.push_antenna_dipole_state(antenna_polarization, antenna_surface_currents, i_time=self.i_time)

        self._run_jefimenko_kernel(include_antenna=include_antenna)

    def recompute_current_fields(self, antenna_surface_currents=None, antenna_polarization=None, antenna_charges=None,
                                 include_antenna=True, antenna_currents=None):
        """Re-run the field kernel for the current time node without advancing time.

        MAS coupling needs a two-pass field solve at one PIC step:

        1. ``Jefimenko_solver(..., include_antenna=False)`` gives the plasma-only
           incident field used in the boundary condition.
        2. MAS solves the passive antenna current.
        3. ``recompute_current_fields(antenna_surface_currents=K,
           antenna_polarization=P, antenna_charges=..., include_antenna=True)``
           pushes conductor sources to the same history
           slot and computes the total grid+antenna field.
        """
        if not hasattr(self, 'i_time') or not hasattr(self, 't'):
            raise RuntimeError('No current EM time node exists. Call Jefimenko_solver first.')
        if antenna_charges is not None:
            self.push_antenna_charges(antenna_charges, i_time=self.i_time)
        if antenna_currents is not None:
            raise RuntimeError('antenna_currents legacy point-current API was removed in Step 2-B')
        if (antenna_surface_currents is None) != (antenna_polarization is None):
            raise ValueError('Provide both antenna_surface_currents=K and antenna_polarization=P, or neither')
        if antenna_surface_currents is not None:
            self.push_antenna_dipole_state(antenna_polarization, antenna_surface_currents, i_time=self.i_time)
        self._run_jefimenko_kernel(include_antenna=include_antenna)

    def register_sample_points(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self._sample_point_count = int(len(points))
        if self._sample_point_count == 0:
            self._sample_point_x_GPU = cupy.zeros(1, dtype=np.float64)
            self._sample_point_y_GPU = cupy.zeros(1, dtype=np.float64)
            self._sample_point_z_GPU = cupy.zeros(1, dtype=np.float64)
            return
        self._sample_point_x_GPU = cupy.asarray(points[:, 0], dtype=np.float64)
        self._sample_point_y_GPU = cupy.asarray(points[:, 1], dtype=np.float64)
        self._sample_point_z_GPU = cupy.asarray(points[:, 2], dtype=np.float64)

    def sample_current_E_at_points(self, points=None):
        """Sample the current GPU E field at arbitrary points with one tiny kernel.

        Only the small N_points x 3 result is copied back to CPU, so the MAS
        solve does not require transferring the full 3-D field arrays.
        """
        if points is not None:
            self.register_sample_points(points)
        n_points = int(self._sample_point_count)
        if n_points == 0:
            return np.zeros((0, 3), dtype=np.float64)
        samples = cupy.zeros((n_points, 3), dtype=np.float64)
        blocks = (n_points + (self.threadsperblock - 1)) // self.threadsperblock
        trilinear_sample_vector_kernel[blocks, self.threadsperblock](
            self.GEx, self.GEy, self.GEz,
            self._sample_point_x_GPU, self._sample_point_y_GPU, self._sample_point_z_GPU,
            samples, n_points,
            self.x_grid_size_o, self.y_grid_size_o, self.z_grid_size_o,
            self.dx_o, self.dy_o, self.dz_o,
            self.x_left_boundary_o, self.y_left_boundary_o, self.z_left_boundary_o,
        )
        return samples.get()

    def terminate(self):
        cuda.close()

    def acquire_EB_field(self):
        return self.GEx, self.GEy, self.GEz, self.GBx, self.GBy, self.GBz

    def get_configuration(self):
        return {
            'time_basis_name': self.time_basis_name,
            'len_time_snapshots': self.len_time_snapshots,
            'observation_grid': (self.x_grid_size_o, self.y_grid_size_o, self.z_grid_size_o),
            'source_grid': (self.x_grid_size_s, self.y_grid_size_s, self.z_grid_size_s),
            'observation_spacing': (self.dx_o, self.dy_o, self.dz_o),
            'source_spacing': (self.dx_s, self.dy_s, self.dz_s),
            'epsilon0': self.epsilon0,
            'c': self.c,
            'dt': self.dt,
            'is_electrostatic': self.is_electrostatic,
            'boundary_mode': self.boundary_mode,
            'kernel_mode': getattr(self, 'kernel_mode', 'observation'),
            'history_horizon_distance': self.history_horizon_distance(),
            'grid_truncation_radius': self.grid_truncation_radius,
            'point_truncation_radius': self.point_truncation_radius,
        }
