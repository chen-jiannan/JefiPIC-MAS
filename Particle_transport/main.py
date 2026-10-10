import cupy as cp
import numpy as np
import math
from .cuda_kernels import *

class pic_transport():
    def __init__(self, pos, velocity, u, dt, charges, number_charges, masses,
                 x_left_bound, y_left_bound, z_left_bound,
                 nx, ny, nz, dx, dy, dz,
                 initial_velocity, initial_u, c,
                 use_periodic, use_reflective,
                 projection_bc_mode=None):
        self.pos = {}
        self.velocity = {}
        self.u = {}
        self.prev_pos = {}
        self.active = {}

        for i_key in pos.keys():
            self.pos[i_key] = cp.array(pos[i_key], dtype=cp.float64)
            self.velocity[i_key] = cp.array(velocity[i_key], dtype=cp.float64)
            self.u[i_key] = cp.array(u[i_key], dtype=cp.float64)
            self.prev_pos[i_key] = cp.array(pos[i_key], dtype=cp.float64)
            self.active[i_key] = cp.ones(len(pos[i_key]), dtype=cp.int8)

        self.dt = dt
        self.masses = masses
        self.charges = charges
        self.number_charges = number_charges
        self.initial_velocity = initial_velocity
        self.initial_u = initial_u
        self.c = c

        self.left_bound_x = x_left_bound
        self.left_bound_y = y_left_bound
        self.left_bound_z = z_left_bound
        self.right_bound_x = x_left_bound + nx * dx
        self.right_bound_y = y_left_bound + ny * dy
        self.right_bound_z = z_left_bound + nz * dz

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz

        self.use_periodic = use_periodic
        self.use_reflective = use_reflective
        self.requested_boundary_mode = self._resolve_boundary_mode(projection_bc_mode)
        self.boundary_mode = self.requested_boundary_mode
        self.threadsperblock = 32

        self.cell_volume = dx * dy * dz
        self.inv_cell_volume = 1.0 / self.cell_volume

        self._rho_previous = None
        # Production source-closure diagnostics.  Charge is cell centered and
        # current density remains on its native staggered faces all the way into
        # the Jefimenko solver.  The legacy face-to-cell reconstruction is no
        # longer part of the production path.
        self.last_charge_continuity_error_before = 0.0
        self.last_charge_continuity_error_after = 0.0
        self.last_face_currents = None
        self.last_continuity_diagnostics = {
            'valid': False,
            'face_current': {},
        }

        self.device_enabled = False
        self.device_absorb_particles = True
        self.device_cube_center = np.zeros(3, dtype=np.float64)
        self.device_cube_size = 0.0
        self._last_surface_hits = {
            'points': np.zeros((0, 3), dtype=np.float64),
            'charges': np.zeros((0,), dtype=np.float64),
            'species': np.zeros((0,), dtype=np.int32),
        }
        self.particle_boundary_guard_cells = 0.0

        # Step 1-F: optional thermal open-reservoir particle boundary.  The
        # reservoir keeps the physical particle boundary at the actual box
        # face.  Escaping macro-particles are recycled as independent incoming
        # thermal-flux samples, but outgoing and incoming trajectories are
        # deposited as separate current events so no spurious whole-box current
        # is created by the recycle operation.
        self._thermal_open_reservoir = {
            'configured': False,
            'thermal_speed_by_species': {},
            'rng': np.random.default_rng(12345),
            'seed': 12345,
        }
        self._open_boundary_extra_segments = []
        self._open_boundary_flux_events = []
        self._last_open_reservoir_diagnostics = {
            'enabled': False,
            'outflow_total': 0,
            'inflow_total': 0,
            'outflow_by_species': {},
            'inflow_by_species': {},
            'outflow_by_face': [0, 0, 0, 0, 0, 0],
            'inflow_by_face': [0, 0, 0, 0, 0, 0],
        }
        self._sync_boundary_mode()

    def _resolve_boundary_mode(self, projection_bc_mode=None):
        """Resolve the physical transport boundary mode.

        The legacy ``projection_bc_mode`` argument is accepted only for API
        compatibility. No Poisson/projection correction is applied in this
        revision because such a correction injects boundary-condition-dependent,
        nonphysical broadband field content into the incident field.
        """
        if self.use_periodic:
            return 'periodic'
        if self.use_reflective:
            return 'reflective'
        return 'open'

    def _calculate_grid_config(self, num_particles):
        return (num_particles + (self.threadsperblock - 1)) // self.threadsperblock

    def _sync_boundary_mode(self):
        # Keep the legacy metal-device behavior for the historical modes.  A
        # configured thermal reservoir is itself an open boundary and therefore
        # does not need to be demoted to the legacy frozen-particle 'open' mode.
        if self.device_enabled and self.requested_boundary_mode != 'thermal_open':
            self.boundary_mode = 'open'
        else:
            self.boundary_mode = self.requested_boundary_mode
        self.use_periodic = (self.boundary_mode == 'periodic')
        self.use_reflective = (self.boundary_mode == 'reflective')


    def set_particle_boundary_guard_cells(self, guard_cells):
        """Inset reflective particle-center bounds by ``guard_cells`` cells.

        For cell-centered CIC, a half-cell guard keeps the complete particle
        shape inside the source mesh.  This avoids clipping the charge cloud at
        the outer edge and lets the Step-1C face-current identity remain valid
        with a zero-normal-flux reflective particle boundary.
        """
        guard = float(guard_cells)
        if guard < 0.0:
            raise ValueError('particle boundary guard must be non-negative')
        self.particle_boundary_guard_cells = guard

    def get_particle_transport_bounds(self):
        """Return the particle-center bounds used by the pusher."""
        if self.boundary_mode != 'reflective':
            return (
                self.left_bound_x, self.right_bound_x,
                self.left_bound_y, self.right_bound_y,
                self.left_bound_z, self.right_bound_z,
            )
        gx = self.particle_boundary_guard_cells * self.dx
        gy = self.particle_boundary_guard_cells * self.dy
        gz = self.particle_boundary_guard_cells * self.dz
        return (
            self.left_bound_x + gx, self.right_bound_x - gx,
            self.left_bound_y + gy, self.right_bound_y - gy,
            self.left_bound_z + gz, self.right_bound_z - gz,
        )

    def set_boundary_mode(self, mode):
        if mode not in ('open', 'periodic', 'reflective', 'thermal_open'):
            raise ValueError(f'Unsupported boundary mode: {mode}')
        self.requested_boundary_mode = mode
        self._sync_boundary_mode()

    def configure_thermal_open_reservoir(self, thermal_speed_by_species, seed=12345):
        """Configure a constant-weight thermal open-reservoir boundary.

        ``thermal_speed_by_species`` contains the one-dimensional Maxwellian
        velocity standard deviation for each species in the same code units as
        ``self.velocity``.  When a particle center leaves the physical box, its
        outgoing trajectory is terminated at the first crossed box face and the
        macro-particle slot is recycled as an independent incoming particle.

        Incoming normal speeds are sampled from the flux-weighted half
        Maxwellian ``p(v_n) = v_n/sigma^2 exp[-v_n^2/(2 sigma^2)]``; tangential
        components remain Gaussian N(0,sigma^2).  The boundary crossing time is
        uniform within the time step.  Outflow and inflow are deposited as
        separate current events, including the half-CIC boundary-face flux, so
        the discrete continuity identity is preserved without a whole-domain
        recycle current.
        """
        speeds = {int(k): float(v) for k, v in dict(thermal_speed_by_species).items()}
        if any(v < 0.0 or not np.isfinite(v) for v in speeds.values()):
            raise ValueError('thermal reservoir speeds must be finite and non-negative')
        self._thermal_open_reservoir = {
            'configured': True,
            'thermal_speed_by_species': speeds,
            'rng': np.random.default_rng(int(seed)),
            'seed': int(seed),
        }
        self.requested_boundary_mode = 'thermal_open'
        self._sync_boundary_mode()
        return self.get_open_reservoir_diagnostics()

    def disable_thermal_open_reservoir(self):
        self._thermal_open_reservoir['configured'] = False
        if self.requested_boundary_mode == 'thermal_open':
            self.requested_boundary_mode = 'open'
        self._sync_boundary_mode()

    @staticmethod
    def _inside_open_box(p, lx, rx, ly, ry, lz, rz):
        return (lx < p[0] < rx and ly < p[1] < ry and lz < p[2] < rz)

    def _first_box_crossing(self, p0, p1):
        """Return first physical-box crossing (point, axis, side, t).

        ``side`` is -1 for the lower face and +1 for the upper face.  The
        routine assumes p0 is inside and p1 is outside; ties at an edge/corner
        are resolved deterministically by the first axis, which is sufficient
        because such exact ties have measure zero for thermal particles.
        """
        d = np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
        left = (self.left_bound_x, self.left_bound_y, self.left_bound_z)
        right = (self.right_bound_x, self.right_bound_y, self.right_bound_z)
        candidates = []
        eps = 1.0e-14
        for ax in range(3):
            if d[ax] < 0.0 and p1[ax] <= left[ax]:
                t = (left[ax] - p0[ax]) / d[ax]
                if -eps <= t <= 1.0 + eps:
                    candidates.append((max(0.0, min(1.0, float(t))), ax, -1))
            elif d[ax] > 0.0 and p1[ax] >= right[ax]:
                t = (right[ax] - p0[ax]) / d[ax]
                if -eps <= t <= 1.0 + eps:
                    candidates.append((max(0.0, min(1.0, float(t))), ax, +1))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[0], x[1]))
        t, ax, side = candidates[0]
        cross = np.asarray(p0, dtype=np.float64) + t * d
        cross[ax] = left[ax] if side < 0 else right[ax]
        # Roundoff at a tangential coordinate must not push the crossing point
        # outside the closed physical face.
        for tangential in range(3):
            if tangential == ax:
                continue
            cross[tangential] = min(max(cross[tangential], left[tangential]), right[tangential])
        return cross, ax, side, t

    def _sample_thermal_open_injection(self, species_id):
        cfg = self._thermal_open_reservoir
        if not cfg.get('configured', False):
            raise RuntimeError('thermal open reservoir is not configured')
        sigma = float(cfg['thermal_speed_by_species'].get(int(species_id), 0.0))
        rng = cfg['rng']

        areas = np.asarray([
            self.dy * self.dz, self.dy * self.dz,
            self.dx * self.dz, self.dx * self.dz,
            self.dx * self.dy, self.dx * self.dy,
        ], dtype=np.float64)
        probs = areas / np.sum(areas)
        left = np.asarray([self.left_bound_x, self.left_bound_y, self.left_bound_z], dtype=np.float64)
        right = np.asarray([self.right_bound_x, self.right_bound_y, self.right_bound_z], dtype=np.float64)

        for _ in range(256):
            face_id = int(rng.choice(6, p=probs))
            ax = face_id // 2
            side = -1 if (face_id % 2) == 0 else +1
            inward_sign = -side

            cross = rng.uniform(left, right)
            cross[ax] = left[ax] if side < 0 else right[ax]

            v = rng.normal(0.0, sigma, size=3) if sigma > 0.0 else np.zeros(3, dtype=np.float64)
            if sigma > 0.0:
                u = max(float(rng.random()), 1.0e-300)
                normal_speed = sigma * math.sqrt(-2.0 * math.log(u))
            else:
                normal_speed = 0.0
            v[ax] = inward_sign * normal_speed

            # Conditional on one crossing occurring in this time step, a
            # stationary Poisson reservoir has a uniform crossing phase.
            tau = float(rng.random()) * self.dt
            p1 = cross + v * (self.dt - tau)
            if self._inside_open_box(
                p1,
                self.left_bound_x, self.right_bound_x,
                self.left_bound_y, self.right_bound_y,
                self.left_bound_z, self.right_bound_z,
            ):
                return cross, p1, v, face_id, tau

        # Extremely large thermal draws can cross a second face.  Rather than
        # admit an invalid event, fall back to a crossing late in the step.
        face_id = int(rng.choice(6, p=probs))
        ax = face_id // 2
        side = -1 if (face_id % 2) == 0 else +1
        inward_sign = -side
        cross = rng.uniform(left, right)
        cross[ax] = left[ax] if side < 0 else right[ax]
        v = rng.normal(0.0, sigma, size=3) if sigma > 0.0 else np.zeros(3, dtype=np.float64)
        if sigma > 0.0:
            u = max(float(rng.random()), 1.0e-300)
            v[ax] = inward_sign * sigma * math.sqrt(-2.0 * math.log(u))
        remain = min(1.0e-6 * self.dt, self.dt)
        p1 = cross + v * remain
        p1 = np.minimum(np.maximum(p1, left + 1.0e-13), right - 1.0e-13)
        return cross, p1, v, face_id, self.dt - remain

    def _queue_boundary_flux_event(self, axis, side, point, charge, sense):
        self._open_boundary_flux_events.append({
            'axis': int(axis),
            'side': int(side),
            'point': np.asarray(point, dtype=np.float64).copy(),
            'charge': float(charge),
            # +1 = outgoing/removal, -1 = incoming/creation.
            'sense': int(sense),
        })

    def _handle_thermal_open_reservoir_species(self, species_id, charge):
        pos_np = cp.asnumpy(self.pos[species_id])
        prev_np = cp.asnumpy(self.prev_pos[species_id])
        active_np = cp.asnumpy(self.active[species_id]).astype(bool)
        outside = np.zeros(len(pos_np), dtype=bool)
        for i, p in enumerate(pos_np):
            if active_np[i] and not self._inside_open_box(
                p,
                self.left_bound_x, self.right_bound_x,
                self.left_bound_y, self.right_bound_y,
                self.left_bound_z, self.right_bound_z,
            ):
                outside[i] = True

        indices = np.where(outside)[0]
        if len(indices) == 0:
            return 0

        diag = self._last_open_reservoir_diagnostics
        for idx in indices:
            crossing = self._first_box_crossing(prev_np[idx], pos_np[idx])
            if crossing is None:
                # This should not occur for a slot that started inside.  Keep a
                # fail-loud diagnostic rather than inventing a trajectory.
                raise RuntimeError(
                    f'thermal-open particle {idx} left the box without a resolvable first crossing: '
                    f'p0={prev_np[idx]}, p1={pos_np[idx]}'
                )
            cross_out, axis_out, side_out, _ = crossing
            face_out = 2 * axis_out + (1 if side_out > 0 else 0)

            # Deposit the physical outgoing trajectory only up to the real box
            # face.  The half-CIC cloud left on the boundary is removed by an
            # explicit normal boundary-flux event below.
            self._open_boundary_extra_segments.append((
                prev_np[idx].copy(), cross_out.copy(), float(charge)
            ))
            self._queue_boundary_flux_event(axis_out, side_out, cross_out, charge, +1)

            # Recycle the macro-particle as an *independent* reservoir particle.
            # Its prev_pos is the new crossing point, so the regular CIC current
            # deposition represents only the incoming in-domain segment.
            cross_in, p_in, v_in, face_in, _ = self._sample_thermal_open_injection(species_id)
            axis_in = face_in // 2
            side_in = -1 if (face_in % 2) == 0 else +1
            self._queue_boundary_flux_event(axis_in, side_in, cross_in, charge, -1)

            pos_np[idx] = p_in
            prev_np[idx] = cross_in
            gamma = 1.0 / math.sqrt(max(1.0 - min(float(np.dot(v_in, v_in)) / (self.c * self.c), 1.0 - 1.0e-15), 1.0e-15))
            u_in = gamma * v_in
            self.pos[species_id][idx] = cp.asarray(p_in)
            self.prev_pos[species_id][idx] = cp.asarray(cross_in)
            self.velocity[species_id][idx] = cp.asarray(v_in)
            self.u[species_id][idx] = cp.asarray(u_in)

            diag['outflow_total'] += 1
            diag['inflow_total'] += 1
            diag['outflow_by_species'][int(species_id)] = diag['outflow_by_species'].get(int(species_id), 0) + 1
            diag['inflow_by_species'][int(species_id)] = diag['inflow_by_species'].get(int(species_id), 0) + 1
            diag['outflow_by_face'][face_out] += 1
            diag['inflow_by_face'][face_in] += 1
        return int(len(indices))

    def _deposit_open_boundary_flux_event(self, Jx_face, Jy_face, Jz_face, event):
        """Deposit the half-CIC normal flux at an open physical box face."""
        axis = int(event['axis'])
        side = int(event['side'])
        p = np.asarray(event['point'], dtype=np.float64)
        charge = float(event['charge'])
        sense = int(event['sense'])
        spacings = (self.dx, self.dy, self.dz)
        lefts = (self.left_bound_x, self.left_bound_y, self.left_bound_z)
        ns = (self.nx, self.ny, self.nz)
        coeff = sense * side * charge * spacings[axis] / (2.0 * self.dt)

        if axis == 0:
            wy = self._cic_endpoint_axis_weights(p[1], self.left_bound_y, self.dy)
            wz = self._cic_endpoint_axis_weights(p[2], self.left_bound_z, self.dz)
            face = 0 if side < 0 else self.nx
            for j, wj in wy.items():
                if not (0 <= j < self.ny):
                    continue
                for k, wk in wz.items():
                    if 0 <= k < self.nz:
                        Jx_face[face, j, k] += coeff * wj * wk
        elif axis == 1:
            wx = self._cic_endpoint_axis_weights(p[0], self.left_bound_x, self.dx)
            wz = self._cic_endpoint_axis_weights(p[2], self.left_bound_z, self.dz)
            face = 0 if side < 0 else self.ny
            for i, wi in wx.items():
                if not (0 <= i < self.nx):
                    continue
                for k, wk in wz.items():
                    if 0 <= k < self.nz:
                        Jy_face[i, face, k] += coeff * wi * wk
        else:
            wx = self._cic_endpoint_axis_weights(p[0], self.left_bound_x, self.dx)
            wy = self._cic_endpoint_axis_weights(p[1], self.left_bound_y, self.dy)
            face = 0 if side < 0 else self.nz
            for i, wi in wx.items():
                if not (0 <= i < self.nx):
                    continue
                for j, wj in wy.items():
                    if 0 <= j < self.ny:
                        Jz_face[i, j, face] += coeff * wi * wj

    def get_open_reservoir_diagnostics(self):
        out = dict(self._last_open_reservoir_diagnostics)
        out['outflow_by_species'] = dict(out.get('outflow_by_species', {}))
        out['inflow_by_species'] = dict(out.get('inflow_by_species', {}))
        out['outflow_by_face'] = list(out.get('outflow_by_face', [0] * 6))
        out['inflow_by_face'] = list(out.get('inflow_by_face', [0] * 6))
        out['enabled'] = bool(
            self.boundary_mode == 'thermal_open'
            and self._thermal_open_reservoir.get('configured', False)
        )
        out['boundary_mode'] = self.boundary_mode
        out['seed'] = self._thermal_open_reservoir.get('seed')
        return out

    def set_metal_device(self, cube_center, cube_size, absorb_particles=True):
        self.device_enabled = True
        self._sync_boundary_mode()
        self.device_absorb_particles = bool(absorb_particles)
        self.device_cube_center = np.asarray(cube_center, dtype=np.float64)
        self.device_cube_size = float(cube_size)

    def clear_metal_device(self):
        self.device_enabled = False
        self._sync_boundary_mode()
        self.device_cube_center = np.zeros(3, dtype=np.float64)
        self.device_cube_size = 0.0
        self._last_surface_hits = {
            'points': np.zeros((0, 3), dtype=np.float64),
            'charges': np.zeros((0,), dtype=np.float64),
            'species': np.zeros((0,), dtype=np.int32),
        }

    def get_last_surface_hits(self):
        return self._last_surface_hits

    def consume_surface_hits(self):
        hits = self._last_surface_hits
        self._last_surface_hits = {
            'points': np.zeros((0, 3), dtype=np.float64),
            'charges': np.zeros((0,), dtype=np.float64),
            'species': np.zeros((0,), dtype=np.int32),
        }
        return hits

    def _build_face_current_x(self, Jx_center):
        nx, ny, nz = Jx_center.shape
        Jx_face = cp.zeros((nx + 1, ny, nz), dtype=Jx_center.dtype)
        if nx > 1:
            Jx_face[1:nx] = 0.5 * (Jx_center[:-1] + Jx_center[1:])
        if self.boundary_mode == 'periodic':
            wrap = 0.5 * (Jx_center[-1] + Jx_center[0])
            Jx_face[0] = wrap
            Jx_face[nx] = wrap
        elif self.boundary_mode == 'reflective':
            Jx_face[0] = 0.0
            Jx_face[nx] = 0.0
        else:
            Jx_face[0] = Jx_center[0]
            Jx_face[nx] = Jx_center[-1]
        return Jx_face

    def _build_face_current_y(self, Jy_center):
        nx, ny, nz = Jy_center.shape
        Jy_face = cp.zeros((nx, ny + 1, nz), dtype=Jy_center.dtype)
        if ny > 1:
            Jy_face[:, 1:ny] = 0.5 * (Jy_center[:, :-1] + Jy_center[:, 1:])
        if self.boundary_mode == 'periodic':
            wrap = 0.5 * (Jy_center[:, -1] + Jy_center[:, 0])
            Jy_face[:, 0] = wrap
            Jy_face[:, ny] = wrap
        elif self.boundary_mode == 'reflective':
            Jy_face[:, 0] = 0.0
            Jy_face[:, ny] = 0.0
        else:
            Jy_face[:, 0] = Jy_center[:, 0]
            Jy_face[:, ny] = Jy_center[:, -1]
        return Jy_face

    def _build_face_current_z(self, Jz_center):
        nx, ny, nz = Jz_center.shape
        Jz_face = cp.zeros((nx, ny, nz + 1), dtype=Jz_center.dtype)
        if nz > 1:
            Jz_face[:, :, 1:nz] = 0.5 * (Jz_center[:, :, :-1] + Jz_center[:, :, 1:])
        if self.boundary_mode == 'periodic':
            wrap = 0.5 * (Jz_center[:, :, -1] + Jz_center[:, :, 0])
            Jz_face[:, :, 0] = wrap
            Jz_face[:, :, nz] = wrap
        elif self.boundary_mode == 'reflective':
            Jz_face[:, :, 0] = 0.0
            Jz_face[:, :, nz] = 0.0
        else:
            Jz_face[:, :, 0] = Jz_center[:, :, 0]
            Jz_face[:, :, nz] = Jz_center[:, :, -1]
        return Jz_face

    def _apply_face_boundary_conditions(self, Jx_face, Jy_face, Jz_face):
        Jx_bc = Jx_face.copy()
        Jy_bc = Jy_face.copy()
        Jz_bc = Jz_face.copy()
        if self.boundary_mode == 'periodic':
            Jx_bc[-1] = Jx_bc[0]
            Jy_bc[:, -1] = Jy_bc[:, 0]
            Jz_bc[:, :, -1] = Jz_bc[:, :, 0]
        elif self.boundary_mode == 'reflective':
            Jx_bc[0] = 0.0
            Jx_bc[-1] = 0.0
            Jy_bc[:, 0] = 0.0
            Jy_bc[:, -1] = 0.0
            Jz_bc[:, :, 0] = 0.0
            Jz_bc[:, :, -1] = 0.0
        return Jx_bc, Jy_bc, Jz_bc

    def _divergence_from_faces(self, Jx_face, Jy_face, Jz_face, dx, dy, dz):
        div_x = (Jx_face[1:] - Jx_face[:-1]) / dx
        div_y = (Jy_face[:, 1:] - Jy_face[:, :-1]) / dy
        div_z = (Jz_face[:, :, 1:] - Jz_face[:, :, :-1]) / dz
        return div_x + div_y + div_z

    @staticmethod
    def _continuity_metric_dict(residual, drho_dt, div_j):
        """Return absolute and normalized scalar norms for one continuity residual.

        The normalized metrics use the sum of the charge-rate and divergence
        norms as the reference scale.  This keeps the diagnostic dimensionless
        without silently hiding cancellation between the two terms.
        """
        tiny = 1.0e-300
        abs_linf = float(cp.max(cp.abs(residual)).get())
        abs_rms = float(cp.sqrt(cp.mean(residual * residual)).get())
        drho_linf = float(cp.max(cp.abs(drho_dt)).get())
        div_linf = float(cp.max(cp.abs(div_j)).get())
        drho_rms = float(cp.sqrt(cp.mean(drho_dt * drho_dt)).get())
        div_rms = float(cp.sqrt(cp.mean(div_j * div_j)).get())
        return {
            'linf_abs': abs_linf,
            'rms_abs': abs_rms,
            'linf_rel': abs_linf / max(drho_linf + div_linf, tiny),
            'rms_rel': abs_rms / max(drho_rms + div_rms, tiny),
            'drho_dt_linf': drho_linf,
            'div_j_linf': div_linf,
            'drho_dt_rms': drho_rms,
            'div_j_rms': div_rms,
        }

    @staticmethod
    def _defect_metric_dict(defect, reference):
        tiny = 1.0e-300
        abs_linf = float(cp.max(cp.abs(defect)).get())
        abs_rms = float(cp.sqrt(cp.mean(defect * defect)).get())
        ref_linf = float(cp.max(cp.abs(reference)).get())
        ref_rms = float(cp.sqrt(cp.mean(reference * reference)).get())
        return {
            'linf_abs': abs_linf,
            'rms_abs': abs_rms,
            'linf_rel_to_face_divergence': abs_linf / max(ref_linf, tiny),
            'rms_rel_to_face_divergence': abs_rms / max(ref_rms, tiny),
            'reference_face_divergence_linf': ref_linf,
            'reference_face_divergence_rms': ref_rms,
        }

    def _enforce_charge_conservation(self, rho_current, Jx_face, Jy_face, Jz_face,
                                     dx, dy, dz, nx, ny, nz):
        """Audit the production staggered-source continuity relation.

        R_face = (rho^n-rho^{n-1})/dt + D_f J_face.

        No face-to-cell reconstruction is formed or audited here because that
        representation has been retired from the production Jefimenko path.
        """
        div_face = self._divergence_from_faces(Jx_face, Jy_face, Jz_face, dx, dy, dz)

        if self._rho_previous is None or self._rho_previous.shape != rho_current.shape:
            self.last_charge_continuity_error_before = 0.0
            self.last_charge_continuity_error_after = 0.0
            self.last_continuity_diagnostics = {
                'valid': False,
                'reason': 'previous rho is unavailable at the first deposited time node',
                'face_current': {},
            }
        else:
            drho_dt = (rho_current - self._rho_previous) / self.dt
            residual_face = drho_dt + div_face
            face_metrics = self._continuity_metric_dict(residual_face, drho_dt, div_face)
            self.last_charge_continuity_error_before = face_metrics['linf_abs']
            self.last_charge_continuity_error_after = face_metrics['linf_abs']
            self.last_continuity_diagnostics = {
                'valid': True,
                'face_current': face_metrics,
                'definition': {
                    'face': '(rho^n-rho^{n-1})/dt + finite-volume divergence of deposited face current',
                    'normalization': 'residual norm divided by sum of |drho/dt| and |div J| norms',
                },
            }

        self._rho_previous = rho_current.copy()
        self.last_face_currents = (Jx_face, Jy_face, Jz_face)
        return Jx_face, Jy_face, Jz_face

    def get_last_face_currents(self, copy=True):
        """Return the latest native staggered PIC current-density arrays.

        Shapes are ``(nx+1,ny,nz)``, ``(nx,ny+1,nz)``, and
        ``(nx,ny,nz+1)``.  These are the charge-conserving face currents
        produced by the deposition step before any face-to-cell reconstruction.
        """
        if self.last_face_currents is None:
            return None
        if copy:
            return tuple(arr.copy() for arr in self.last_face_currents)
        return self.last_face_currents

    def get_continuity_diagnostics(self):
        """Return the latest source-closure audit without copying grid arrays."""
        out = dict(self.last_continuity_diagnostics)
        for key in ('face_current', 'definition'):
            if isinstance(out.get(key), dict):
                out[key] = dict(out[key])
        return out

    def _segment_cube_hits(self, prev_pos, curr_pos, active_mask):
        n = prev_pos.shape[0]
        if n == 0 or not self.device_enabled or not self.device_absorb_particles:
            return cp.zeros(n, dtype=cp.bool_), cp.zeros_like(curr_pos)

        half = 0.5 * self.device_cube_size
        mins = cp.asarray(self.device_cube_center - half)
        maxs = cp.asarray(self.device_cube_center + half)

        p0 = prev_pos
        p1 = curr_pos
        d = p1 - p0
        eps = 1.0e-30

        tmin = cp.zeros(n, dtype=cp.float64)
        tmax = cp.ones(n, dtype=cp.float64)
        valid = active_mask.astype(cp.bool_)

        for ax in range(3):
            p0a = p0[:, ax]
            da = d[:, ax]
            stationary = cp.abs(da) < eps
            inside = (p0a >= mins[ax]) & (p0a <= maxs[ax])
            valid = valid & ((~stationary) | inside)

            inv_da = cp.where(stationary, 0.0, 1.0 / da)
            t1 = (mins[ax] - p0a) * inv_da
            t2 = (maxs[ax] - p0a) * inv_da
            ta = cp.minimum(t1, t2)
            tb = cp.maximum(t1, t2)
            tmin = cp.where(stationary, tmin, cp.maximum(tmin, ta))
            tmax = cp.where(stationary, tmax, cp.minimum(tmax, tb))

        valid = valid & (tmin <= tmax) & (tmax >= 0.0) & (tmin <= 1.0)
        t_hit = cp.clip(tmin, 0.0, 1.0)
        hit_points = p0 + d * t_hit[:, None]
        return valid, hit_points

    def _cic_indices_weights(self, coord_norm, n_cells):
        base = int(math.floor(coord_norm))
        frac = coord_norm - base
        indices = []
        weights = []
        for offset, weight in ((0, 1.0 - frac), (1, frac)):
            idx = base + offset
            if 0 <= idx < n_cells and weight != 0.0:
                indices.append(idx)
                weights.append(weight)
        return indices, weights

    def _shape1_weights(self, s, n_cells):
        base = int(math.floor(s))
        frac = s - base
        indices = []
        weights = []
        for offset, weight in ((0, 1.0 - frac), (1, frac)):
            idx = base + offset
            if 0 <= idx < n_cells and weight != 0.0:
                indices.append(idx)
                weights.append(weight)
        return indices, weights

    def _shape2_weights(self, s, n_cells):
        center = int(math.floor(s + 0.5))
        indices = []
        weights = []
        for offset in (-1, 0, 1):
            idx = center + offset
            if 0 <= idx < n_cells:
                d = abs(s - idx)
                if d < 0.5:
                    w = 0.75 - d * d
                elif d < 1.5:
                    w = 0.5 * (1.5 - d) ** 2
                else:
                    w = 0.0
                if w != 0.0:
                    indices.append(idx)
                    weights.append(w)
        total = sum(weights)
        if total > 0.0:
            weights = [w / total for w in weights]
        return indices, weights

    def _center_shape_indices_weights(self, s, n_cells, order):
        if order <= 1:
            return self._shape1_weights(s, n_cells)
        return self._shape2_weights(s, n_cells)

    def _face_shape_indices_weights(self, s, n_faces, order):
        if order <= 1:
            return self._shape1_weights(s, n_faces)
        return self._shape2_weights(s, n_faces)

    def _current_component_face_weights(self, pm, component, order):
        if component == 'x':
            sy = (pm[1] - self.left_bound_y) / self.dy - 0.5
            sz = (pm[2] - self.left_bound_z) / self.dz - 0.5
            iy, wy = self._center_shape_indices_weights(sy, self.ny, order)
            iz, wz = self._center_shape_indices_weights(sz, self.nz, order)
            fx = (pm[0] - self.left_bound_x) / self.dx
            ix, wx = self._face_shape_indices_weights(fx, self.nx + 1, order)
            return ix, wx, iy, wy, iz, wz
        if component == 'y':
            sx = (pm[0] - self.left_bound_x) / self.dx - 0.5
            sz = (pm[2] - self.left_bound_z) / self.dz - 0.5
            ix, wx = self._center_shape_indices_weights(sx, self.nx, order)
            iz, wz = self._center_shape_indices_weights(sz, self.nz, order)
            fy = (pm[1] - self.left_bound_y) / self.dy
            iy, wy = self._face_shape_indices_weights(fy, self.ny + 1, order)
            return ix, wx, iy, wy, iz, wz
        sx = (pm[0] - self.left_bound_x) / self.dx - 0.5
        sy = (pm[1] - self.left_bound_y) / self.dy - 0.5
        ix, wx = self._center_shape_indices_weights(sx, self.nx, order)
        iy, wy = self._center_shape_indices_weights(sy, self.ny, order)
        fz = (pm[2] - self.left_bound_z) / self.dz
        iz, wz = self._face_shape_indices_weights(fz, self.nz + 1, order)
        return ix, wx, iy, wy, iz, wz

    def _split_segment_for_flux_deposition(self, p0, p1):
        ts = [0.0, 1.0]
        d = p1 - p0
        eps = 1.0e-14
        bounds = ((self.left_bound_x, self.dx, self.nx),
                  (self.left_bound_y, self.dy, self.ny),
                  (self.left_bound_z, self.dz, self.nz))
        for ax, (left, spacing, n_cells) in enumerate(bounds):
            if abs(d[ax]) < eps:
                continue
            s0 = (p0[ax] - left) / spacing
            s1 = (p1[ax] - left) / spacing
            lo = min(s0, s1)
            hi = max(s0, s1)
            k_start = int(math.floor(lo + 0.5))
            k_end = int(math.floor(hi + 0.5))
            for k in range(k_start, k_end + 1):
                boundary = left + (k + 0.5) * spacing
                t = (boundary - p0[ax]) / d[ax]
                if eps < t < 1.0 - eps:
                    ts.append(float(t))
        ts = np.array(sorted(set(max(0.0, min(1.0, t)) for t in ts)), dtype=np.float64)
        return ts

    def _deposit_rho_numpy(self, rho, pos_np, charge, order):
        for p in pos_np:
            if not (self.left_bound_x < p[0] < self.right_bound_x and
                    self.left_bound_y < p[1] < self.right_bound_y and
                    self.left_bound_z < p[2] < self.right_bound_z):
                continue
            sx = (p[0] - self.left_bound_x) / self.dx - 0.5
            sy = (p[1] - self.left_bound_y) / self.dy - 0.5
            sz = (p[2] - self.left_bound_z) / self.dz - 0.5
            ix, wx = self._center_shape_indices_weights(sx, self.nx, order)
            iy, wy = self._center_shape_indices_weights(sy, self.ny, order)
            iz, wz = self._center_shape_indices_weights(sz, self.nz, order)
            for a, wa in zip(ix, wx):
                for b, wb in zip(iy, wy):
                    for c, wc in zip(iz, wz):
                        rho[a, b, c] += charge * wa * wb * wc

    @staticmethod
    def _cic_endpoint_axis_weights(position, left, spacing):
        """Return the un-clipped first-order cell-center shape weights.

        The charge deposition used by this code is cell centered, i.e. the
        normalized charge coordinate is ``s = (x-left)/dx - 1/2``.  Current
        deposition needs the *same* shape function at the two trajectory
        endpoints.  Keeping the un-clipped indices here is important: near a
        physical boundary an outside-grid shape weight represents charge flux
        through the boundary face rather than a reason to renormalize the
        remaining in-domain charge.
        """
        s = (float(position) - float(left)) / float(spacing) - 0.5
        base = int(math.floor(s))
        frac = s - base
        return {base: 1.0 - frac, base + 1: frac}

    @staticmethod
    def _cic_axis_endpoint_data(position_a, position_b, left, spacing):
        """Build endpoint, mean, difference and cumulative-flux data.

        If ``S^a_i`` and ``S^b_i`` are the CIC charge weights at the two
        endpoints, define ``dS_i = S^b_i-S^a_i``.  The longitudinal face-flux
        coefficients ``F`` are constructed so that

            F_{i+1} - F_i = -dS_i.

        This is the one-dimensional discrete continuity identity.
        """
        wa = pic_transport._cic_endpoint_axis_weights(position_a, left, spacing)
        wb = pic_transport._cic_endpoint_axis_weights(position_b, left, spacing)
        indices = list(range(min(min(wa), min(wb)), max(max(wa), max(wb)) + 1))
        mean = {}
        delta = {}
        for i in indices:
            a = wa.get(i, 0.0)
            b = wb.get(i, 0.0)
            mean[i] = 0.5 * (a + b)
            delta[i] = b - a

        face_flux = {indices[0]: 0.0}
        running = 0.0
        for i in indices:
            running -= delta[i]
            face_flux[i + 1] = running
        return indices, mean, delta, face_flux

    def _deposit_cic_segment_charge_conserving(self, Jx_face, Jy_face, Jz_face, pa, pb, charge):
        """Deposit one trajectory segment with exact CIC charge continuity.

        For first-order cell-centered charge shapes, the endpoint product
        difference is decomposed symmetrically as

          d(Sx Sy Sz) =
              dSx (Sy_bar Sz_bar + dSy dSz/12)
            + dSy (Sz_bar Sx_bar + dSz dSx/12)
            + dSz (Sx_bar Sy_bar + dSx dSy/12).

        Combining this identity with cumulative longitudinal face fluxes makes
        ``(rho^{n+1}-rho^n)/dt + div_f J_f`` an algebraic identity for every
        interior CIC trajectory segment (up to floating-point roundoff).
        """
        ix, mx, dxs, fx = self._cic_axis_endpoint_data(pa[0], pb[0], self.left_bound_x, self.dx)
        iy, my, dys, fy = self._cic_axis_endpoint_data(pa[1], pb[1], self.left_bound_y, self.dy)
        iz, mz, dzs, fz = self._cic_axis_endpoint_data(pa[2], pb[2], self.left_bound_z, self.dz)

        coeff_x = float(charge) * self.dx / self.dt
        coeff_y = float(charge) * self.dy / self.dt
        coeff_z = float(charge) * self.dz / self.dt

        for face, flux in fx.items():
            if flux == 0.0 or not (0 <= face <= self.nx):
                continue
            for j in iy:
                if not (0 <= j < self.ny):
                    continue
                for k in iz:
                    if not (0 <= k < self.nz):
                        continue
                    transverse = my[j] * mz[k] + (dys[j] * dzs[k]) / 12.0
                    if transverse != 0.0:
                        Jx_face[face, j, k] += coeff_x * flux * transverse

        for i in ix:
            if not (0 <= i < self.nx):
                continue
            for face, flux in fy.items():
                if flux == 0.0 or not (0 <= face <= self.ny):
                    continue
                for k in iz:
                    if not (0 <= k < self.nz):
                        continue
                    transverse = mz[k] * mx[i] + (dzs[k] * dxs[i]) / 12.0
                    if transverse != 0.0:
                        Jy_face[i, face, k] += coeff_y * flux * transverse

        for i in ix:
            if not (0 <= i < self.nx):
                continue
            for j in iy:
                if not (0 <= j < self.ny):
                    continue
                transverse = mx[i] * my[j] + (dxs[i] * dys[j]) / 12.0
                if transverse == 0.0:
                    continue
                for face, flux in fz.items():
                    if flux == 0.0 or not (0 <= face <= self.nz):
                        continue
                    Jz_face[i, j, face] += coeff_z * flux * transverse

    def _deposit_current_esirkepov_style_numpy(self, Jx_face, Jy_face, Jz_face, prev_np, curr_np, active_np, charge, order):
        dt = self.dt
        for idx in range(len(prev_np)):
            if not active_np[idx]:
                continue
            p0 = prev_np[idx]
            p1 = curr_np[idx]
            # Step 1-Ec: do not use np.allclose() to classify a trajectory as
            # stationary.  In LHQCD code units the absolute coordinates can be
            # O(1e3), so numpy's default relative tolerance can classify real
            # ion displacements O(1e-2) as zero.  The endpoint-difference CIC
            # deposition is well-defined even for an exactly stationary path;
            # zero displacement simply produces zero face flux.
            ts = self._split_segment_for_flux_deposition(p0, p1)
            for t0, t1 in zip(ts[:-1], ts[1:]):
                if t1 <= t0:
                    continue
                pa = p0 + (p1 - p0) * t0
                pb = p0 + (p1 - p0) * t1

                # Step 1-C: for CIC use a genuinely charge-conserving
                # endpoint-difference deposition.  The previous midpoint face
                # interpolation did not satisfy the discrete continuity
                # identity except at special geometric locations.
                if order <= 1:
                    self._deposit_cic_segment_charge_conserving(
                        Jx_face, Jy_face, Jz_face, pa, pb, charge
                    )
                    continue

                # TSC/order>1 remains on the legacy path in Step 1-C.  It is
                # deliberately not claimed as charge-conserving until a
                # separate higher-order derivation/benchmark is completed.
                pm = 0.5 * (pa + pb)
                dp = pb - pa

                ix_face, wx_face, iy, wy, iz, wz = self._current_component_face_weights(pm, 'x', order)
                if dp[0] != 0.0:
                    for i, wxi in zip(ix_face, wx_face):
                        for j, wyj in zip(iy, wy):
                            for k, wzk in zip(iz, wz):
                                Jx_face[i, j, k] += charge * dp[0] / dt * wxi * wyj * wzk

                ix, wx, iy_face, wy_face, iz, wz = self._current_component_face_weights(pm, 'y', order)
                if dp[1] != 0.0:
                    for i, wxi in zip(ix, wx):
                        for j, wyj in zip(iy_face, wy_face):
                            for k, wzk in zip(iz, wz):
                                Jy_face[i, j, k] += charge * dp[1] / dt * wxi * wyj * wzk

                ix, wx, iy, wy, iz_face, wz_face = self._current_component_face_weights(pm, 'z', order)
                if dp[2] != 0.0:
                    for i, wxi in zip(ix, wx):
                        for j, wyj in zip(iy, wy):
                            for k, wzk in zip(iz_face, wz_face):
                                Jz_face[i, j, k] += charge * dp[2] / dt * wxi * wyj * wzk

    def _apply_physical_face_bc_numpy(self, Jx_face, Jy_face, Jz_face):
        if self.boundary_mode == 'periodic':
            Jx_face[-1] = Jx_face[0]
            Jy_face[:, -1] = Jy_face[:, 0]
            Jz_face[:, :, -1] = Jz_face[:, :, 0]
        elif self.boundary_mode == 'reflective':
            Jx_face[0] = 0.0
            Jx_face[-1] = 0.0
            Jy_face[:, 0] = 0.0
            Jy_face[:, -1] = 0.0
            Jz_face[:, :, 0] = 0.0
            Jz_face[:, :, -1] = 0.0
        return Jx_face, Jy_face, Jz_face

    def _quadratic_face_to_center_axis(self, face_values, axis):
        if axis == 0:
            n = face_values.shape[0] - 1
            center = 0.5 * (face_values[:-1] + face_values[1:])
            if n >= 2:
                center[1:-1] = (-face_values[:-3] + 9.0 * face_values[1:-2] + 9.0 * face_values[2:-1] - face_values[3:]) / 16.0
            return center
        if axis == 1:
            n = face_values.shape[1] - 1
            center = 0.5 * (face_values[:, :-1] + face_values[:, 1:])
            if n >= 2:
                center[:, 1:-1] = (-face_values[:, :-3] + 9.0 * face_values[:, 1:-2] + 9.0 * face_values[:, 2:-1] - face_values[:, 3:]) / 16.0
            return center
        n = face_values.shape[2] - 1
        center = 0.5 * (face_values[:, :, :-1] + face_values[:, :, 1:])
        if n >= 2:
            center[:, :, 1:-1] = (-face_values[:, :, :-3] + 9.0 * face_values[:, :, 1:-2] + 9.0 * face_values[:, :, 2:-1] - face_values[:, :, 3:]) / 16.0
        return center

    def _face_to_center_currents(self, Jx_face, Jy_face, Jz_face, order):
        if order <= 1:
            Jx = 0.5 * (Jx_face[:-1] + Jx_face[1:])
            Jy = 0.5 * (Jy_face[:, :-1] + Jy_face[:, 1:])
            Jz = 0.5 * (Jz_face[:, :, :-1] + Jz_face[:, :, 1:])
        else:
            Jx = self._quadratic_face_to_center_axis(Jx_face, 0)
            Jy = self._quadratic_face_to_center_axis(Jy_face, 1)
            Jz = self._quadratic_face_to_center_axis(Jz_face, 2)
        return Jx, Jy, Jz

    def _evaluate_rho_J_esirkepov(self, particle_order=1):
        rho = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        Jx_face = np.zeros((self.nx + 1, self.ny, self.nz), dtype=np.float64)
        Jy_face = np.zeros((self.nx, self.ny + 1, self.nz), dtype=np.float64)
        Jz_face = np.zeros((self.nx, self.ny, self.nz + 1), dtype=np.float64)

        for i_species in self.pos.keys():
            if i_species >= len(self.charges):
                continue
            charge = float(self.charges[i_species])
            pos_np = cp.asnumpy(self.pos[i_species])
            prev_np = cp.asnumpy(self.prev_pos[i_species])
            active_np = cp.asnumpy(self.active[i_species]).astype(bool)
            if len(pos_np) == 0:
                continue
            self._deposit_rho_numpy(rho, pos_np[active_np], charge, particle_order)
            self._deposit_current_esirkepov_style_numpy(Jx_face, Jy_face, Jz_face, prev_np, pos_np, active_np, charge, particle_order)

        # Step 1-F open-reservoir bookkeeping.  A recycled slot represents an
        # incoming trajectory only; the old outgoing segment and the half-CIC
        # normal fluxes at both physical box faces are deposited separately.
        if self._open_boundary_extra_segments:
            for p0_evt, p1_evt, q_evt in self._open_boundary_extra_segments:
                self._deposit_current_esirkepov_style_numpy(
                    Jx_face, Jy_face, Jz_face,
                    np.asarray([p0_evt], dtype=np.float64),
                    np.asarray([p1_evt], dtype=np.float64),
                    np.asarray([True], dtype=bool),
                    float(q_evt), particle_order,
                )
        if self._open_boundary_flux_events:
            if particle_order != 1:
                raise RuntimeError('thermal_open boundary is continuity-validated only for CIC/order=1 in Step 1-F')
            for event in self._open_boundary_flux_events:
                self._deposit_open_boundary_flux_event(Jx_face, Jy_face, Jz_face, event)

        Jx_face, Jy_face, Jz_face = self._apply_physical_face_bc_numpy(Jx_face, Jy_face, Jz_face)

        # Events are one-step source bookkeeping; consume them only after the
        # deposited face currents have been assembled.
        self._open_boundary_extra_segments = []
        self._open_boundary_flux_events = []

        inv_dv = 1.0 / self.cell_volume
        rho_cp = cp.asarray(rho * inv_dv)
        Jx_face_cp = cp.asarray(Jx_face * inv_dv)
        Jy_face_cp = cp.asarray(Jy_face * inv_dv)
        Jz_face_cp = cp.asarray(Jz_face * inv_dv)
        self.last_face_currents = (Jx_face_cp, Jy_face_cp, Jz_face_cp)
        return rho_cp, Jx_face_cp, Jy_face_cp, Jz_face_cp

    def proceed_one_step(self, Ex, Ey, Ez, Bx, By, Bz,
                         dx, dy, dz, nx, ny, nz, field_order=1):
        Ex_reshaped = Ex.reshape([nx, ny, nz])
        Ey_reshaped = Ey.reshape([nx, ny, nz])
        Ez_reshaped = Ez.reshape([nx, ny, nz])
        Bx_reshaped = Bx.reshape([nx, ny, nz])
        By_reshaped = By.reshape([nx, ny, nz])
        Bz_reshaped = Bz.reshape([nx, ny, nz])

        particle_EM = None
        all_hit_points = []
        (
            particle_left_x, particle_right_x,
            particle_left_y, particle_right_y,
            particle_left_z, particle_right_z,
        ) = self.get_particle_transport_bounds()
        all_hit_charges = []
        all_hit_species = []

        self._open_boundary_extra_segments = []
        self._open_boundary_flux_events = []
        self._last_open_reservoir_diagnostics = {
            'enabled': bool(self.boundary_mode == 'thermal_open'),
            'outflow_total': 0,
            'inflow_total': 0,
            'outflow_by_species': {},
            'inflow_by_species': {},
            'outflow_by_face': [0, 0, 0, 0, 0, 0],
            'inflow_by_face': [0, 0, 0, 0, 0, 0],
        }

        for i_species in self.pos.keys():
            num_particles = len(self.pos[i_species])
            if num_particles == 0:
                continue

            blockspergrid = self._calculate_grid_config(num_particles)
            particle_EM = cp.zeros([num_particles, 6], dtype=cp.float64)

            mass = self.masses[i_species]
            charge = self.charges[i_species]
            self.prev_pos[i_species][:] = self.pos[i_species]

            find_EM[blockspergrid, self.threadsperblock](
                num_particles,
                self.pos[i_species],
                self.velocity[i_species],
                nx, ny, nz,
                Ex_reshaped, Ey_reshaped, Ez_reshaped,
                Bx_reshaped, By_reshaped, Bz_reshaped,
                dx, dy, dz,
                mass, charge, self.dt,
                self.left_bound_x, self.left_bound_y, self.left_bound_z,
                self.right_bound_x, self.right_bound_y, self.right_bound_z,
                particle_EM, field_order
            )

            update_particle_info[blockspergrid, self.threadsperblock](
                num_particles,
                self.pos[i_species],
                self.velocity[i_species],
                self.u[i_species],
                nx, ny, nz, dx, dy, dz,
                mass, charge, self.dt,
                particle_left_x, particle_left_y, particle_left_z,
                particle_right_x, particle_right_y, particle_right_z,
                particle_EM, field_order,
                self.initial_velocity, self.initial_u, self.c,
                self.use_reflective, self.use_periodic
            )

            if self.boundary_mode == 'thermal_open':
                if not self._thermal_open_reservoir.get('configured', False):
                    raise RuntimeError('boundary_mode=thermal_open but no thermal reservoir was configured')
                self._handle_thermal_open_reservoir_species(i_species, charge)

            if self.device_enabled and (not self.use_periodic):
                hit_mask, hit_points = self._segment_cube_hits(
                    self.prev_pos[i_species],
                    self.pos[i_species],
                    self.active[i_species]
                )

                n_hits = int(cp.count_nonzero(hit_mask).get())
                if n_hits > 0:
                    hit_points_np = hit_points[hit_mask].get()
                    all_hit_points.append(hit_points_np)
                    all_hit_charges.append(np.full(n_hits, float(charge), dtype=np.float64))
                    all_hit_species.append(np.full(n_hits, int(i_species), dtype=np.int32))

                    self.active[i_species][hit_mask] = 0
                    outside = cp.asarray([self.right_bound_x + 10.0 * self.dx,
                                          self.right_bound_y + 10.0 * self.dy,
                                          self.right_bound_z + 10.0 * self.dz], dtype=cp.float64)
                    self.pos[i_species][hit_mask] = outside
                    self.velocity[i_species][hit_mask] = 0.0
                    self.u[i_species][hit_mask] = 0.0

        if all_hit_points:
            self._last_surface_hits = {
                'points': np.concatenate(all_hit_points, axis=0),
                'charges': np.concatenate(all_hit_charges, axis=0),
                'species': np.concatenate(all_hit_species, axis=0),
            }
        else:
            self._last_surface_hits = {
                'points': np.zeros((0, 3), dtype=np.float64),
                'charges': np.zeros((0,), dtype=np.float64),
                'species': np.zeros((0,), dtype=np.int32),
            }

        return particle_EM

    def evaluate_rho_J(self, dx, dy, dz, nx, ny, nz, particle_order=1):
        if (dx, dy, dz, nx, ny, nz) != (self.dx, self.dy, self.dz, self.nx, self.ny, self.nz):
            raise ValueError('evaluate_rho_J must be called with the transport grid used at initialization.')

        particle_order = 1 if particle_order is None else int(particle_order)
        if particle_order not in (1, 2):
            raise ValueError(f'Unsupported particle_order={particle_order}; use 1 (CIC) or 2 (TSC).')

        rho_physical, Jx_face, Jy_face, Jz_face = self._evaluate_rho_J_esirkepov(
            particle_order=particle_order
        )
        self._enforce_charge_conservation(
            rho_physical, Jx_face, Jy_face, Jz_face,
            dx, dy, dz, nx, ny, nz
        )
        return rho_physical, Jx_face, Jy_face, Jz_face

    def _evaluate_species_nu(self, species_id, dx, dy, dz, nx, ny, nz):
        if species_id not in self.pos:
            return cp.zeros([nx, ny, nz], dtype=cp.float64)

        num_particles = len(self.pos[species_id])
        if num_particles == 0:
            return cp.zeros([nx, ny, nz], dtype=cp.float64)

        blockspergrid = self._calculate_grid_config(num_particles)
        rho_n = cp.zeros([nx, ny, nz], dtype=cp.float64)

        find_n_nu[blockspergrid, self.threadsperblock](
            num_particles,
            self.pos[species_id],
            dx, dy, dz,
            self.number_charges[species_id],
            rho_n,
            self.left_bound_x, self.left_bound_y, self.left_bound_z,
            self.right_bound_x, self.right_bound_y, self.right_bound_z
        )
        return rho_n / (dx * dy * dz)

    def evaluate_e_nu(self, dx, dy, dz, nx, ny, nz):
        return self._evaluate_species_nu(0, dx, dy, dz, nx, ny, nz)

    def evaluate_i_nu(self, dx, dy, dz, nx, ny, nz):
        return self._evaluate_species_nu(1, dx, dy, dz, nx, ny, nz)

    def evaluate_all_species_nu(self, dx, dy, dz, nx, ny, nz):
        result = {}
        for species_id in self.pos.keys():
            result[species_id] = self._evaluate_species_nu(species_id, dx, dy, dz, nx, ny, nz)
        return result

    def get_particle_counts(self):
        counts = {}
        for species_id in self.pos.keys():
            counts[species_id] = len(self.pos[species_id])
        return counts

    def get_active_particle_counts(self):
        counts = {}
        for species_id in self.pos.keys():
            counts[species_id] = int(cp.sum(self.active[species_id]).get())
        return counts

    def get_domain_bounds(self):
        left_bounds = (self.left_bound_x, self.left_bound_y, self.left_bound_z)
        right_bounds = (self.right_bound_x, self.right_bound_y, self.right_bound_z)
        return left_bounds, right_bounds

    def get_projection_info(self):
        return {
            'projection_enabled': False,
            'projection_bc_mode': None,
            'physical_boundary_mode': self.boundary_mode,
            'grid_size': (self.nx, self.ny, self.nz),
            'grid_spacing': (self.dx, self.dy, self.dz),
            'supported_particle_orders': (1, 2),
            'device_forces_open_boundary': True,
            'thermal_open_reservoir': self.get_open_reservoir_diagnostics(),
        }

# ----------------------------------------------------------------------
# Optional fast CIC deposition backend
# ----------------------------------------------------------------------
def _install_default_fast_deposition():
    """
    Install the Numba deposition backend for pic_transport.

    The Step-1C order-1 CIC current path is charge-conserving.  The order-2
    TSC path remains legacy and is not yet claimed as continuity-validated.
    Set environment variable JEFIPIC_FAST_DEPOSITION=0 to disable.
    """
    import os
    import warnings

    flag = os.environ.get("JEFIPIC_FAST_DEPOSITION", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return

    try:
        from .fast_deposition import install_fast_cic_deposition_for_class
        install_fast_cic_deposition_for_class(pic_transport)
        print("[Particle_transport] Fast Numba CIC/TSC deposition enabled.")
    except Exception as exc:
        warnings.warn(
            "[Particle_transport] Fast Numba CIC/TSC deposition was not enabled; "
            f"falling back to original Python/Numpy deposition. Reason: {exc!r}"
        )


_install_default_fast_deposition()

