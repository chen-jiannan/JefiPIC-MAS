import cupy
from numba import cuda
import math

BOUNDARY_OPEN = 0
BOUNDARY_PERIODIC = 1
BOUNDARY_REFLECTIVE = 2

# INDEX_SELF_EXCLUSION_PATCH_V1

# give the distance between two grids
@cuda.jit(device=True)
def r_distance(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


# give the lower index of the retarded time from time_snapshots
@cuda.jit(device=True)
def retarded_time_index(t, dr, dt, len_time_snapshots, c):
    tr = t - dr / c

    # if the signal has not been transmitted
    if tr < 0.0:
        return -1

    # 采用不超前的下侧时间索引，避免原来的 off-by-one 提前一拍问题
    ti_abs = int(math.floor(tr / dt))
    return ti_abs % len_time_snapshots


# copy newly calculated rho and J to GPU memory
# This is the cuda kernel function
@cuda.jit
def copy_new_rho_J_to_GPU(i_time, rho, Jx, Jy, Jz,
                          rho_GPU, Jx_GPU, Jy_GPU, Jz_GPU,
                          total_grid_s):

    i = cuda.grid(1)

    # loop through each spatial grid
    if i < total_grid_s:
        rho_GPU[i_time, i] = rho[i]
        Jx_GPU[i_time, i] = Jx[i]
        Jy_GPU[i_time, i] = Jy[i]
        Jz_GPU[i_time, i] = Jz[i]


# give the real position from index and left_boundary
@cuda.jit(device=True)
def real_position(i, d, left_boundary):
    return i * d + d / 2 + left_boundary


@cuda.jit
def trilinear_sample_vector_kernel(Fx, Fy, Fz,
                                   point_x, point_y, point_z,
                                   samples,
                                   n_points,
                                   nx, ny, nz,
                                   dx, dy, dz,
                                   x_left, y_left, z_left):
    """Sample a flattened vector field at arbitrary physical points.

    The field layout matches Jefimenko output: i = iz + iy*nz + ix*nz*ny.
    Coordinates are interpreted as cell-centered values located at
    x_left + (i + 0.5) * dx, etc. Outside points are clamped to the nearest
    boundary cell, matching the CPU MAS sampler.
    """
    p = cuda.grid(1)
    if p >= n_points:
        return

    sx = (point_x[p] - x_left) / dx - 0.5
    sy = (point_y[p] - y_left) / dy - 0.5
    sz = (point_z[p] - z_left) / dz - 0.5

    i0 = int(math.floor(sx))
    j0 = int(math.floor(sy))
    k0 = int(math.floor(sz))
    tx = sx - i0
    ty = sy - j0
    tz = sz - k0

    vx = 0.0
    vy = 0.0
    vz = 0.0

    for di in range(2):
        ii = i0 + di
        if ii < 0:
            ii = 0
        if ii >= nx:
            ii = nx - 1
        wx = (1.0 - tx) if di == 0 else tx
        for dj in range(2):
            jj = j0 + dj
            if jj < 0:
                jj = 0
            if jj >= ny:
                jj = ny - 1
            wy = (1.0 - ty) if dj == 0 else ty
            for dk in range(2):
                kk = k0 + dk
                if kk < 0:
                    kk = 0
                if kk >= nz:
                    kk = nz - 1
                wz = (1.0 - tz) if dk == 0 else tz
                w = wx * wy * wz
                idx = kk + jj * nz + ii * nz * ny
                vx += w * Fx[idx]
                vy += w * Fy[idx]
                vz += w * Fz[idx]

    samples[p, 0] = vx
    samples[p, 1] = vy
    samples[p, 2] = vz


@cuda.jit(device=True)
def _interpolated_sources_at(rho_GPU_s, Jx_GPU_s, Jy_GPU_s, Jz_GPU_s, idx_lower, idx_upper, alpha, i_s):
    rho_lower = rho_GPU_s[idx_lower, i_s]
    rho_upper = rho_GPU_s[idx_upper, i_s]
    rho_s = (1.0 - alpha) * rho_lower + alpha * rho_upper

    Jx_lower = Jx_GPU_s[idx_lower, i_s]
    Jx_upper = Jx_GPU_s[idx_upper, i_s]
    Jy_lower = Jy_GPU_s[idx_lower, i_s]
    Jy_upper = Jy_GPU_s[idx_upper, i_s]
    Jz_lower = Jz_GPU_s[idx_lower, i_s]
    Jz_upper = Jz_GPU_s[idx_upper, i_s]

    Jx_s = (1.0 - alpha) * Jx_lower + alpha * Jx_upper
    Jy_s = (1.0 - alpha) * Jy_lower + alpha * Jy_upper
    Jz_s = (1.0 - alpha) * Jz_lower + alpha * Jz_upper

    return rho_s, Jx_lower, Jx_upper, Jy_lower, Jy_upper, Jz_lower, Jz_upper, Jx_s, Jy_s, Jz_s


@cuda.jit(device=True)
def _truncation_weight(dr, cutoff_radius, taper_width, taper_type):
    """Return a smooth spatial cutoff weight.

    taper_type:
        0 = hard cutoff / no taper
        1 = Hann half-window in the outer taper_width shell
        2 = C1 smoothstep in the outer taper_width shell
        3 = exponential C-infinity-like shoulder, still forced to zero at cutoff
    """
    if cutoff_radius <= 0.0:
        return 1.0
    if dr >= cutoff_radius:
        return 0.0
    if taper_type <= 0 or taper_width <= 0.0:
        return 1.0
    inner = cutoff_radius - taper_width
    if inner < 0.0:
        inner = 0.0
    if dr <= inner:
        return 1.0
    denom = cutoff_radius - inner
    if denom <= 1.0e-300:
        return 1.0
    u = (dr - inner) / denom
    if u <= 0.0:
        return 1.0
    if u >= 1.0:
        return 0.0
    if taper_type == 1:
        return 0.5 * (1.0 + math.cos(math.pi * u))
    if taper_type == 2:
        # 1 - smoothstep(u)
        return 1.0 - u * u * (3.0 - 2.0 * u)
    # Exponential shoulder.  The denominator makes the value approach zero
    # smoothly as u -> 1, while the hard cutoff above enforces exact compact support.
    return math.exp(-u / max(1.0e-12, 1.0 - u))


@cuda.jit(device=True)
def _accumulate_open_contribution(x_o, y_o, z_o, x_s, y_s, z_s,
                                  rho_s, prho_pt_s,
                                  Jx_s, Jy_s, Jz_s, pJx_pt_s, pJy_pt_s, pJz_pt_s,
                                  i_o, GEx, GEy, GEz, GBx, GBy, GBz, c, is_electrostatic,
                                  truncation_radius, taper_width, taper_type):
    drx = x_o - x_s
    dry = y_o - y_s
    drz = z_o - z_s
    dr = math.sqrt(drx * drx + dry * dry + drz * drz)
    if dr <= 1.0e-19:
        return
    taper = _truncation_weight(dr, truncation_radius, taper_width, taper_type)
    if taper <= 0.0:
        return

    if is_electrostatic:
        cuda.atomic.add(GEx, i_o, taper * (drx / dr ** 3 * rho_s))
        cuda.atomic.add(GEy, i_o, taper * (dry / dr ** 3 * rho_s))
        cuda.atomic.add(GEz, i_o, taper * (drz / dr ** 3 * rho_s))
    else:
        cuda.atomic.add(GEx, i_o, taper * (drx / dr ** 3 * rho_s + drx / dr ** 2 / c * prho_pt_s - 1.0 / dr / c ** 2 * pJx_pt_s))
        cuda.atomic.add(GEy, i_o, taper * (dry / dr ** 3 * rho_s + dry / dr ** 2 / c * prho_pt_s - 1.0 / dr / c ** 2 * pJy_pt_s))
        cuda.atomic.add(GEz, i_o, taper * (drz / dr ** 3 * rho_s + drz / dr ** 2 / c * prho_pt_s - 1.0 / dr / c ** 2 * pJz_pt_s))
        cuda.atomic.add(GBx, i_o, taper * (((dry * Jz_s - drz * Jy_s) / dr ** 3 + (dry * pJz_pt_s - drz * pJy_pt_s) / dr ** 2 / c)))
        cuda.atomic.add(GBy, i_o, taper * (((drz * Jx_s - drx * Jz_s) / dr ** 3 + (drz * pJx_pt_s - drx * pJz_pt_s) / dr ** 2 / c)))
        cuda.atomic.add(GBz, i_o, taper * (((drx * Jy_s - dry * Jx_s) / dr ** 3 + (drx * pJy_pt_s - dry * pJx_pt_s) / dr ** 2 / c)))



@cuda.jit(device=True)
def _accumulate_point_current_sources(x_o, y_o, z_o,
                                      point_src_x, point_src_y, point_src_z, point_weights,
                                      point_rho_GPU, point_Jx_GPU, point_Jy_GPU, point_Jz_GPU,
                                      point_source_count, point_softening, point_truncation_radius,
                                      point_taper_width, point_taper_type,
                                      len_time_snapshots, dt, t,
                                      dx_s, dy_s, dz_s,
                                      i_o, GEx, GEy, GEz, GBx, GBy, GBz, c, is_electrostatic):
    """Add MAS auxiliary surface-charge and surface-current point sources.

    The output arrays are the unscaled Jefimenko accumulators.  Since the grid
    solver later multiplies E by coeff_E=(1/4πeps0)*dV and B by
    coeff_B=-(mu0/4π)*dV, point-source contributions are normalized by the
    source-grid cell volume so that a point source weight is interpreted as the
    equivalent integration measure of that source.
    """
    cell_volume = dx_s * dy_s * dz_s
    if cell_volume <= 1.0e-300:
        cell_volume = 1.0
    current_abs = int(math.floor(t / dt + 1.0e-9))
    oldest_abs = current_abs - len_time_snapshots + 1
    if oldest_abs < 0:
        oldest_abs = 0

    for js in range(point_source_count):
        drx = x_o - point_src_x[js]
        dry = y_o - point_src_y[js]
        drz = z_o - point_src_z[js]
        dr2 = drx * drx + dry * dry + drz * drz + point_softening * point_softening
        if dr2 <= 1.0e-38:
            continue
        dr = math.sqrt(dr2)
        point_taper = _truncation_weight(dr, point_truncation_radius, point_taper_width, point_taper_type)
        if point_taper <= 0.0:
            continue
        # Quasi-electrostatic closure is instantaneous: conductor and plasma
        # charge at the current PIC node define the Coulomb field used in the
        # same node.  Retarded-time gating is retained only for full EM runs.
        tr = t if is_electrostatic else t - dr / c
        if tr < 0.0:
            continue

        tau = tr / dt
        tau_floor = math.floor(tau)
        frac = tau - tau_floor
        tau_floor_int = int(tau_floor)
        if frac < 1.0e-12:
            idx_lower_abs = tau_floor_int
            idx_upper_abs = tau_floor_int
            alpha = 0.0
        else:
            idx_lower_abs = tau_floor_int
            idx_upper_abs = tau_floor_int + 1
            alpha = frac

        if idx_lower_abs < 0:
            continue
        if idx_upper_abs > current_abs:
            idx_lower_abs = current_abs
            idx_upper_abs = current_abs
            alpha = 0.0
        # Guard the circular history buffer. Without this check, a retarded
        # request older than the ring capacity would alias to a newer slot after
        # the modulo operation and create a non-physical history echo.
        if idx_lower_abs < oldest_abs or idx_upper_abs < oldest_abs:
            continue

        idx_lower = idx_lower_abs % len_time_snapshots
        idx_upper = idx_upper_abs % len_time_snapshots

        rho_lower = point_rho_GPU[idx_lower, js]
        rho_upper = point_rho_GPU[idx_upper, js]
        rho_s = (1.0 - alpha) * rho_lower + alpha * rho_upper
        Jx_lower = point_Jx_GPU[idx_lower, js]
        Jy_lower = point_Jy_GPU[idx_lower, js]
        Jz_lower = point_Jz_GPU[idx_lower, js]
        Jx_upper = point_Jx_GPU[idx_upper, js]
        Jy_upper = point_Jy_GPU[idx_upper, js]
        Jz_upper = point_Jz_GPU[idx_upper, js]

        Jx_s = (1.0 - alpha) * Jx_lower + alpha * Jx_upper
        Jy_s = (1.0 - alpha) * Jy_lower + alpha * Jy_upper
        Jz_s = (1.0 - alpha) * Jz_lower + alpha * Jz_upper

        if idx_upper_abs == idx_lower_abs:
            prho_pt_s = 0.0
            pJx_pt_s = 0.0
            pJy_pt_s = 0.0
            pJz_pt_s = 0.0
        else:
            prho_pt_s = (rho_upper - rho_lower) / dt
            pJx_pt_s = (Jx_upper - Jx_lower) / dt
            pJy_pt_s = (Jy_upper - Jy_lower) / dt
            pJz_pt_s = (Jz_upper - Jz_lower) / dt

        weight_over_cell = point_weights[js] / cell_volume
        inv_dr = 1.0 / dr
        inv_dr2 = inv_dr * inv_dr
        inv_dr3 = inv_dr2 * inv_dr

        weight_over_cell = weight_over_cell * point_taper

        if is_electrostatic:
            cuda.atomic.add(GEx, i_o, (drx * inv_dr3 * rho_s) * weight_over_cell)
            cuda.atomic.add(GEy, i_o, (dry * inv_dr3 * rho_s) * weight_over_cell)
            cuda.atomic.add(GEz, i_o, (drz * inv_dr3 * rho_s) * weight_over_cell)
        else:
            cuda.atomic.add(GEx, i_o, (drx * inv_dr3 * rho_s + drx * inv_dr2 / c * prho_pt_s - inv_dr / (c * c) * pJx_pt_s) * weight_over_cell)
            cuda.atomic.add(GEy, i_o, (dry * inv_dr3 * rho_s + dry * inv_dr2 / c * prho_pt_s - inv_dr / (c * c) * pJy_pt_s) * weight_over_cell)
            cuda.atomic.add(GEz, i_o, (drz * inv_dr3 * rho_s + drz * inv_dr2 / c * prho_pt_s - inv_dr / (c * c) * pJz_pt_s) * weight_over_cell)

        cuda.atomic.add(GBx, i_o, ((dry * Jz_s - drz * Jy_s) * inv_dr3 +
                                   (dry * pJz_pt_s - drz * pJy_pt_s) * inv_dr2 / c) * weight_over_cell)
        cuda.atomic.add(GBy, i_o, ((drz * Jx_s - drx * Jz_s) * inv_dr3 +
                                   (drz * pJx_pt_s - drx * pJz_pt_s) * inv_dr2 / c) * weight_over_cell)
        cuda.atomic.add(GBz, i_o, ((drx * Jy_s - dry * Jx_s) * inv_dr3 +
                                   (drx * pJy_pt_s - dry * pJx_pt_s) * inv_dr2 / c) * weight_over_cell)

@cuda.jit(device=True)
def _open_contribution_local(x_o, y_o, z_o, x_s, y_s, z_s,
                             rho_s, prho_pt_s,
                             Jx_s, Jy_s, Jz_s, pJx_pt_s, pJy_pt_s, pJz_pt_s,
                             c, is_electrostatic,
                             truncation_radius, taper_width, taper_type):
    """Return one open-boundary Jefimenko contribution without atomics."""
    drx = x_o - x_s
    dry = y_o - y_s
    drz = z_o - z_s
    dr2 = drx * drx + dry * dry + drz * drz
    if dr2 <= 1.0e-38:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dr = math.sqrt(dr2)
    taper = _truncation_weight(dr, truncation_radius, taper_width, taper_type)
    if taper <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    inv_dr = 1.0 / dr
    inv_dr2 = inv_dr * inv_dr
    inv_dr3 = inv_dr2 * inv_dr

    if is_electrostatic:
        ex = taper * (drx * inv_dr3 * rho_s)
        ey = taper * (dry * inv_dr3 * rho_s)
        ez = taper * (drz * inv_dr3 * rho_s)
        return ex, ey, ez, 0.0, 0.0, 0.0

    ex = taper * (drx * inv_dr3 * rho_s + drx * inv_dr2 / c * prho_pt_s - inv_dr / (c * c) * pJx_pt_s)
    ey = taper * (dry * inv_dr3 * rho_s + dry * inv_dr2 / c * prho_pt_s - inv_dr / (c * c) * pJy_pt_s)
    ez = taper * (drz * inv_dr3 * rho_s + drz * inv_dr2 / c * prho_pt_s - inv_dr / (c * c) * pJz_pt_s)
    bx = taper * ((dry * Jz_s - drz * Jy_s) * inv_dr3 + (dry * pJz_pt_s - drz * pJy_pt_s) * inv_dr2 / c)
    by = taper * ((drz * Jx_s - drx * Jz_s) * inv_dr3 + (drz * pJx_pt_s - drx * pJz_pt_s) * inv_dr2 / c)
    bz = taper * ((drx * Jy_s - dry * Jx_s) * inv_dr3 + (drx * pJy_pt_s - dry * pJx_pt_s) * inv_dr2 / c)
    return ex, ey, ez, bx, by, bz


@cuda.jit(device=True)
def _point_current_sources_local(x_o, y_o, z_o,
                                 point_src_x, point_src_y, point_src_z, point_weights,
                                 point_rho_GPU, point_Jx_GPU, point_Jy_GPU, point_Jz_GPU,
                                 point_source_count, point_softening, point_truncation_radius,
                                 point_taper_width, point_taper_type,
                                 len_time_snapshots, dt, t,
                                 dx_s, dy_s, dz_s, c, is_electrostatic):
    """Return MAS surface-charge/current point-source contribution without atomics."""
    ex_total = 0.0
    ey_total = 0.0
    ez_total = 0.0
    bx_total = 0.0
    by_total = 0.0
    bz_total = 0.0

    cell_volume = dx_s * dy_s * dz_s
    if cell_volume <= 1.0e-300:
        cell_volume = 1.0
    current_abs = int(math.floor(t / dt + 1.0e-9))
    oldest_abs = current_abs - len_time_snapshots + 1
    if oldest_abs < 0:
        oldest_abs = 0

    for js in range(point_source_count):
        drx = x_o - point_src_x[js]
        dry = y_o - point_src_y[js]
        drz = z_o - point_src_z[js]
        dr2 = drx * drx + dry * dry + drz * drz + point_softening * point_softening
        if dr2 <= 1.0e-38:
            continue
        dr = math.sqrt(dr2)
        point_taper = _truncation_weight(dr, point_truncation_radius, point_taper_width, point_taper_type)
        if point_taper <= 0.0:
            continue
        # Quasi-electrostatic closure is instantaneous: conductor and plasma
        # charge at the current PIC node define the Coulomb field used in the
        # same node.  Retarded-time gating is retained only for full EM runs.
        tr = t if is_electrostatic else t - dr / c
        if tr < 0.0:
            continue

        tau = tr / dt
        tau_floor = math.floor(tau)
        frac = tau - tau_floor
        tau_floor_int = int(tau_floor)
        if frac < 1.0e-12:
            idx_lower_abs = tau_floor_int
            idx_upper_abs = tau_floor_int
            alpha = 0.0
        else:
            idx_lower_abs = tau_floor_int
            idx_upper_abs = tau_floor_int + 1
            alpha = frac

        if idx_lower_abs < 0:
            continue
        if idx_upper_abs > current_abs:
            idx_lower_abs = current_abs
            idx_upper_abs = current_abs
            alpha = 0.0
        if idx_lower_abs < oldest_abs or idx_upper_abs < oldest_abs:
            continue

        idx_lower = idx_lower_abs % len_time_snapshots
        idx_upper = idx_upper_abs % len_time_snapshots

        rho_lower = point_rho_GPU[idx_lower, js]
        rho_upper = point_rho_GPU[idx_upper, js]
        rho_s = (1.0 - alpha) * rho_lower + alpha * rho_upper

        Jx_lower = point_Jx_GPU[idx_lower, js]
        Jy_lower = point_Jy_GPU[idx_lower, js]
        Jz_lower = point_Jz_GPU[idx_lower, js]
        Jx_upper = point_Jx_GPU[idx_upper, js]
        Jy_upper = point_Jy_GPU[idx_upper, js]
        Jz_upper = point_Jz_GPU[idx_upper, js]

        Jx_s = (1.0 - alpha) * Jx_lower + alpha * Jx_upper
        Jy_s = (1.0 - alpha) * Jy_lower + alpha * Jy_upper
        Jz_s = (1.0 - alpha) * Jz_lower + alpha * Jz_upper

        if idx_upper_abs == idx_lower_abs:
            prho_pt_s = 0.0
            pJx_pt_s = 0.0
            pJy_pt_s = 0.0
            pJz_pt_s = 0.0
        else:
            prho_pt_s = (rho_upper - rho_lower) / dt
            pJx_pt_s = (Jx_upper - Jx_lower) / dt
            pJy_pt_s = (Jy_upper - Jy_lower) / dt
            pJz_pt_s = (Jz_upper - Jz_lower) / dt

        inv_dr = 1.0 / dr
        inv_dr2 = inv_dr * inv_dr
        inv_dr3 = inv_dr2 * inv_dr
        weight_over_cell = point_weights[js] / cell_volume * point_taper

        if is_electrostatic:
            ex_total += (drx * inv_dr3 * rho_s) * weight_over_cell
            ey_total += (dry * inv_dr3 * rho_s) * weight_over_cell
            ez_total += (drz * inv_dr3 * rho_s) * weight_over_cell
        else:
            ex_total += (drx * inv_dr3 * rho_s + drx * inv_dr2 / c * prho_pt_s - inv_dr / (c * c) * pJx_pt_s) * weight_over_cell
            ey_total += (dry * inv_dr3 * rho_s + dry * inv_dr2 / c * prho_pt_s - inv_dr / (c * c) * pJy_pt_s) * weight_over_cell
            ez_total += (drz * inv_dr3 * rho_s + drz * inv_dr2 / c * prho_pt_s - inv_dr / (c * c) * pJz_pt_s) * weight_over_cell
        bx_total += ((dry * Jz_s - drz * Jy_s) * inv_dr3 + (dry * pJz_pt_s - drz * pJy_pt_s) * inv_dr2 / c) * weight_over_cell
        by_total += ((drz * Jx_s - drx * Jz_s) * inv_dr3 + (drz * pJx_pt_s - drx * pJz_pt_s) * inv_dr2 / c) * weight_over_cell
        bz_total += ((drx * Jy_s - dry * Jx_s) * inv_dr3 + (drx * pJy_pt_s - dry * pJx_pt_s) * inv_dr2 / c) * weight_over_cell

    return ex_total, ey_total, ez_total, bx_total, by_total, bz_total


@cuda.jit
def Jefimenko_kernel_observation_parallel(rho_GPU_s, Jx_GPU_s, Jy_GPU_s, Jz_GPU_s,
                                          len_time_snapshots, i_time, dt, t,
                                          total_grid_o, total_grid_s,
                                          x_grid_size_o, y_grid_size_o, z_grid_size_o,
                                          x_grid_size_s, y_grid_size_s, z_grid_size_s,
                                          dx_o, dy_o, dz_o,
                                          dx_s, dy_s, dz_s,
                                          x_left_boundary_o, y_left_boundary_o, z_left_boundary_o,
                                          x_left_boundary_s, y_left_boundary_s, z_left_boundary_s,
                                          GEx, GEy, GEz, GBx, GBy, GBz, coeff_E, coeff_B, c, all_grids, is_electrostatic,
                                          grid_current_enabled,
                                          boundary_mode, Lx_s, Ly_s, Lz_s, max_img_x, max_img_y, max_img_z,
                                          point_src_x, point_src_y, point_src_z, point_weights,
                                          point_rho_GPU, point_Jx_GPU, point_Jy_GPU, point_Jz_GPU,
                                          point_source_count, point_softening,
                                          grid_truncation_radius, point_truncation_radius,
                                          grid_taper_width, point_taper_width,
                                          grid_taper_type, point_taper_type):
    """Observation-parallel Jefimenko kernel without atomic add contention."""
    i_o = cuda.grid(1)
    if i_o >= total_grid_o:
        return

    iz_o = i_o % z_grid_size_o
    rest_o = i_o // z_grid_size_o
    iy_o = rest_o % y_grid_size_o
    ix_o = rest_o // y_grid_size_o

    x_o = real_position(ix_o, dx_o, x_left_boundary_o)
    y_o = real_position(iy_o, dy_o, y_left_boundary_o)
    z_o = real_position(iz_o, dz_o, z_left_boundary_o)

    # Robust direct self-cell exclusion for coincident source/observation grids.
    # Do not infer self identity from reconstructed floating-point coordinates:
    # equal mathematical coordinates can differ by ~1 ULP on GPU and turn the
    # singular 1/R^2 Coulomb self term into a catastrophic numerical spike.
    same_grid_indexing = (
        x_grid_size_o == x_grid_size_s and
        y_grid_size_o == y_grid_size_s and
        z_grid_size_o == z_grid_size_s and
        dx_o == dx_s and dy_o == dy_s and dz_o == dz_s and
        x_left_boundary_o == x_left_boundary_s and
        y_left_boundary_o == y_left_boundary_s and
        z_left_boundary_o == z_left_boundary_s
    )

    ex_total = 0.0
    ey_total = 0.0
    ez_total = 0.0
    bx_total = 0.0
    by_total = 0.0
    bz_total = 0.0

    if point_source_count > 0:
        pex, pey, pez, pbx, pby, pbz = _point_current_sources_local(
            x_o, y_o, z_o,
            point_src_x, point_src_y, point_src_z, point_weights,
            point_rho_GPU, point_Jx_GPU, point_Jy_GPU, point_Jz_GPU,
            point_source_count, point_softening, point_truncation_radius,
            point_taper_width, point_taper_type,
            len_time_snapshots, dt, t,
            dx_s, dy_s, dz_s, c, is_electrostatic
        )
        ex_total += pex
        ey_total += pey
        ez_total += pez
        bx_total += pbx
        by_total += pby
        bz_total += pbz

    current_abs = int(math.floor(t / dt + 1.0e-9))
    oldest_abs = current_abs - len_time_snapshots + 1
    if oldest_abs < 0:
        oldest_abs = 0

    for i_s in range(total_grid_s):
        # Direct same-cell source is singular and must be excluded by identity,
        # not by a floating-point distance threshold.
        if same_grid_indexing and i_s == i_o:
            continue

        iz_s = i_s % z_grid_size_s
        rest_s = i_s // z_grid_size_s
        iy_s = rest_s % y_grid_size_s
        ix_s = rest_s // y_grid_size_s

        x_s0 = real_position(ix_s, dx_s, x_left_boundary_s)
        y_s0 = real_position(iy_s, dy_s, y_left_boundary_s)
        z_s0 = real_position(iz_s, dz_s, z_left_boundary_s)

        drx0 = x_o - x_s0
        dry0 = y_o - y_s0
        drz0 = z_o - z_s0
        dr0 = math.sqrt(drx0 * drx0 + dry0 * dry0 + drz0 * drz0)
        if dr0 <= 1.0e-19:
            continue
        if boundary_mode == BOUNDARY_OPEN and _truncation_weight(dr0, grid_truncation_radius, grid_taper_width, grid_taper_type) <= 0.0:
            continue

        # In quasi-electrostatic mode, evaluate rho from the current time
        # node; otherwise an initially charged conductor incorrectly produces
        # zero field until a fictitious propagation delay has elapsed.
        tr0 = t if is_electrostatic else t - dr0 / c
        if tr0 < 0.0:
            continue

        tau = tr0 / dt
        tau_floor = math.floor(tau)
        frac = tau - tau_floor
        tau_floor_int = int(tau_floor)

        if frac < 1.0e-12:
            if tau_floor_int > 0:
                idx_lower_abs = tau_floor_int - 1
                idx_upper_abs = tau_floor_int
                alpha = 1.0
            else:
                idx_lower_abs = 0
                idx_upper_abs = 0
                alpha = 0.0
        else:
            idx_lower_abs = tau_floor_int
            idx_upper_abs = tau_floor_int + 1
            alpha = frac

        if idx_lower_abs < oldest_abs or idx_upper_abs < oldest_abs:
            continue
        if idx_upper_abs > current_abs:
            idx_lower_abs = current_abs
            idx_upper_abs = current_abs
            alpha = 0.0

        idx_lower = idx_lower_abs % len_time_snapshots
        idx_upper = idx_upper_abs % len_time_snapshots

        rho_lower = rho_GPU_s[idx_lower, i_s]
        rho_upper = rho_GPU_s[idx_upper, i_s]
        rho_s = rho_lower * (1.0 - alpha) + rho_upper * alpha

        # The production PIC path stores only native staggered face-current
        # histories.  ``grid_current_enabled == 0`` therefore evaluates only
        # the cell-centered charge contribution here; current terms are added
        # by Jefimenko_staggered_current_kernel_observation_parallel below.
        if grid_current_enabled != 0:
            Jx_lower = Jx_GPU_s[idx_lower, i_s]
            Jx_upper = Jx_GPU_s[idx_upper, i_s]
            Jy_lower = Jy_GPU_s[idx_lower, i_s]
            Jy_upper = Jy_GPU_s[idx_upper, i_s]
            Jz_lower = Jz_GPU_s[idx_lower, i_s]
            Jz_upper = Jz_GPU_s[idx_upper, i_s]
            Jx_s0 = Jx_lower * (1.0 - alpha) + Jx_upper * alpha
            Jy_s0 = Jy_lower * (1.0 - alpha) + Jy_upper * alpha
            Jz_s0 = Jz_lower * (1.0 - alpha) + Jz_upper * alpha
        else:
            Jx_lower = 0.0
            Jx_upper = 0.0
            Jy_lower = 0.0
            Jy_upper = 0.0
            Jz_lower = 0.0
            Jz_upper = 0.0
            Jx_s0 = 0.0
            Jy_s0 = 0.0
            Jz_s0 = 0.0

        if idx_upper_abs == idx_lower_abs:
            prho_pt_s = 0.0
            pJx_pt_s0 = 0.0
            pJy_pt_s0 = 0.0
            pJz_pt_s0 = 0.0
        else:
            prho_pt_s = (rho_upper - rho_lower) / dt
            if grid_current_enabled != 0:
                pJx_pt_s0 = (Jx_upper - Jx_lower) / dt
                pJy_pt_s0 = (Jy_upper - Jy_lower) / dt
                pJz_pt_s0 = (Jz_upper - Jz_lower) / dt
            else:
                pJx_pt_s0 = 0.0
                pJy_pt_s0 = 0.0
                pJz_pt_s0 = 0.0

        if boundary_mode == BOUNDARY_OPEN:
            cex, cey, cez, cbx, cby, cbz = _open_contribution_local(
                x_o, y_o, z_o, x_s0, y_s0, z_s0,
                rho_s, prho_pt_s,
                Jx_s0, Jy_s0, Jz_s0, pJx_pt_s0, pJy_pt_s0, pJz_pt_s0,
                c, is_electrostatic, grid_truncation_radius, grid_taper_width, grid_taper_type
            )
            ex_total += cex
            ey_total += cey
            ez_total += cez
            bx_total += cbx
            by_total += cby
            bz_total += cbz
        elif boundary_mode == BOUNDARY_PERIODIC:
            for mx in range(-max_img_x, max_img_x + 1):
                xs = x_s0 + mx * Lx_s
                for my in range(-max_img_y, max_img_y + 1):
                    ys = y_s0 + my * Ly_s
                    for mz in range(-max_img_z, max_img_z + 1):
                        zs = z_s0 + mz * Lz_s
                        cex, cey, cez, cbx, cby, cbz = _open_contribution_local(
                            x_o, y_o, z_o, xs, ys, zs,
                            rho_s, prho_pt_s,
                            Jx_s0, Jy_s0, Jz_s0, pJx_pt_s0, pJy_pt_s0, pJz_pt_s0,
                            c, is_electrostatic, grid_truncation_radius, grid_taper_width, grid_taper_type
                        )
                        ex_total += cex
                        ey_total += cey
                        ez_total += cez
                        bx_total += cbx
                        by_total += cby
                        bz_total += cbz
        else:
            rel_x = x_s0 - x_left_boundary_s
            rel_y = y_s0 - y_left_boundary_s
            rel_z = z_s0 - z_left_boundary_s
            period_x = 2.0 * Lx_s
            period_y = 2.0 * Ly_s
            period_z = 2.0 * Lz_s
            for mx in range(-max_img_x, max_img_x + 1):
                for rx in range(2):
                    sign_x = -1.0 if rx == 1 else 1.0
                    xs_rel = rel_x if rx == 0 else (2.0 * Lx_s - rel_x)
                    xs = x_left_boundary_s + xs_rel + mx * period_x
                    for my in range(-max_img_y, max_img_y + 1):
                        for ry in range(2):
                            sign_y = -1.0 if ry == 1 else 1.0
                            ys_rel = rel_y if ry == 0 else (2.0 * Ly_s - rel_y)
                            ys = y_left_boundary_s + ys_rel + my * period_y
                            for mz in range(-max_img_z, max_img_z + 1):
                                for rz in range(2):
                                    sign_z = -1.0 if rz == 1 else 1.0
                                    zs_rel = rel_z if rz == 0 else (2.0 * Lz_s - rel_z)
                                    zs = z_left_boundary_s + zs_rel + mz * period_z
                                    cex, cey, cez, cbx, cby, cbz = _open_contribution_local(
                                        x_o, y_o, z_o, xs, ys, zs,
                                        rho_s, prho_pt_s,
                                        sign_x * Jx_s0, sign_y * Jy_s0, sign_z * Jz_s0,
                                        sign_x * pJx_pt_s0, sign_y * pJy_pt_s0, sign_z * pJz_pt_s0,
                                        c, is_electrostatic, grid_truncation_radius, grid_taper_width, grid_taper_type
                                    )
                                    ex_total += cex
                                    ey_total += cey
                                    ez_total += cez
                                    bx_total += cbx
                                    by_total += cby
                                    bz_total += cbz

    GEx[i_o] = ex_total
    GEy[i_o] = ey_total
    GEz[i_o] = ez_total
    GBx[i_o] = bx_total
    GBy[i_o] = by_total
    GBz[i_o] = bz_total

# cuda kernel for obtaining E(t)  and B(t)  with given rho(t) /Jx(t) /Jy(t) /Jz(t)
# time_snapshots only saves the recent time info, a full time info is on cpu and denoted as time_sequence
# rho_GPU, Jx_GPU, Jy_GPU, Jz_GPU are of shape [len_time_snapshots, total_grid_s]
@cuda.jit
def Jefimenko_kernel(rho_GPU_s, Jx_GPU_s, Jy_GPU_s, Jz_GPU_s,
                     len_time_snapshots, i_time, dt, t,
                     total_grid_o, total_grid_s,
                     x_grid_size_o, y_grid_size_o, z_grid_size_o,
                     x_grid_size_s, y_grid_size_s, z_grid_size_s,
                     dx_o, dy_o, dz_o,
                     dx_s, dy_s, dz_s,
                     x_left_boundary_o, y_left_boundary_o, z_left_boundary_o,
                     x_left_boundary_s, y_left_boundary_s, z_left_boundary_s,
                     GEx, GEy, GEz, GBx, GBy, GBz, coeff_E, coeff_B, c, all_grids, is_electrostatic,
                     boundary_mode, Lx_s, Ly_s, Lz_s, max_img_x, max_img_y, max_img_z,
                     point_src_x, point_src_y, point_src_z, point_weights,
                     point_rho_GPU, point_Jx_GPU, point_Jy_GPU, point_Jz_GPU,
                     point_source_count, point_softening,
                     grid_truncation_radius, point_truncation_radius,
                     grid_taper_width, point_taper_width,
                     grid_taper_type, point_taper_type):

    i_grid = cuda.grid(1)

    if i_grid < all_grids:
        iz_o = i_grid % z_grid_size_o
        iz_rest_o = i_grid // z_grid_size_o
        iy_o = iz_rest_o % y_grid_size_o
        iy_rest_o = iz_rest_o // y_grid_size_o
        ix_o = iy_rest_o % x_grid_size_o
        ix_rest_o = iy_rest_o // x_grid_size_o
        i_o = iz_o + iy_o * z_grid_size_o + ix_o * z_grid_size_o * y_grid_size_o

        iz_s = ix_rest_o % z_grid_size_s
        iz_rest_s = ix_rest_o // z_grid_size_s
        iy_s = iz_rest_s % y_grid_size_s
        iy_rest_s = iz_rest_s // y_grid_size_s
        ix_s = iy_rest_s % x_grid_size_s
        i_s = iz_s + iy_s * z_grid_size_s + ix_s * z_grid_size_s * y_grid_size_s

        # Use index identity for the direct self-cell term whenever source and
        # observation grids are exactly the same discretization.  This makes
        # Atomic and Observation follow the same explicit self-exclusion rule.
        same_grid_indexing = (
            x_grid_size_o == x_grid_size_s and
            y_grid_size_o == y_grid_size_s and
            z_grid_size_o == z_grid_size_s and
            dx_o == dx_s and dy_o == dy_s and dz_o == dz_s and
            x_left_boundary_o == x_left_boundary_s and
            y_left_boundary_o == y_left_boundary_s and
            z_left_boundary_o == z_left_boundary_s
        )
        if same_grid_indexing and i_s == i_o:
            return

        x_o = real_position(ix_o, dx_o, x_left_boundary_o)
        y_o = real_position(iy_o, dy_o, y_left_boundary_o)
        z_o = real_position(iz_o, dz_o, z_left_boundary_o)

        x_s0 = real_position(ix_s, dx_s, x_left_boundary_s)
        y_s0 = real_position(iy_s, dy_s, y_left_boundary_s)
        z_s0 = real_position(iz_s, dz_s, z_left_boundary_s)

        if point_source_count > 0 and i_s == 0:
            _accumulate_point_current_sources(
                x_o, y_o, z_o,
                point_src_x, point_src_y, point_src_z, point_weights,
                point_rho_GPU, point_Jx_GPU, point_Jy_GPU, point_Jz_GPU,
                point_source_count, point_softening, point_truncation_radius,
                point_taper_width, point_taper_type,
                len_time_snapshots, dt, t,
                dx_s, dy_s, dz_s,
                i_o, GEx, GEy, GEz, GBx, GBy, GBz, c, is_electrostatic
            )

        dr0 = r_distance(x_o, y_o, z_o, x_s0, y_s0, z_s0)
        if boundary_mode == BOUNDARY_OPEN and _truncation_weight(dr0, grid_truncation_radius, grid_taper_width, grid_taper_type) <= 0.0:
            return
        # In quasi-electrostatic mode, evaluate rho from the current time
        # node; otherwise an initially charged conductor incorrectly produces
        # zero field until a fictitious propagation delay has elapsed.
        tr0 = t if is_electrostatic else t - dr0 / c
        if tr0 < 0.0:
            return

        current_abs = int(math.floor(t / dt + 1.0e-9))
        oldest_abs = current_abs - len_time_snapshots + 1
        if oldest_abs < 0:
            oldest_abs = 0

        tau = tr0 / dt
        tau_floor = math.floor(tau)
        frac = tau - tau_floor
        tau_floor_int = int(tau_floor)

        if frac < 1.0e-12:
            if tau_floor_int > 0:
                idx_lower_abs = tau_floor_int - 1
                idx_upper_abs = tau_floor_int
                alpha = 1.0
            else:
                idx_lower_abs = 0
                idx_upper_abs = 0
                alpha = 0.0
        else:
            idx_lower_abs = tau_floor_int
            idx_upper_abs = tau_floor_int + 1
            alpha = frac

        if idx_lower_abs < oldest_abs or idx_upper_abs < oldest_abs:
            return
        if idx_upper_abs > current_abs:
            idx_lower_abs = current_abs
            idx_upper_abs = current_abs
            alpha = 0.0

        idx_lower = idx_lower_abs % len_time_snapshots
        idx_upper = idx_upper_abs % len_time_snapshots

        rho_s, Jx_lower, Jx_upper, Jy_lower, Jy_upper, Jz_lower, Jz_upper, Jx_s0, Jy_s0, Jz_s0 = _interpolated_sources_at(
            rho_GPU_s, Jx_GPU_s, Jy_GPU_s, Jz_GPU_s, idx_lower, idx_upper, alpha, i_s
        )

        if idx_upper_abs == idx_lower_abs:
            prho_pt_s = 0.0
            pJx_pt_s0 = 0.0
            pJy_pt_s0 = 0.0
            pJz_pt_s0 = 0.0
        else:
            prho_pt_s = (rho_GPU_s[idx_upper, i_s] - rho_GPU_s[idx_lower, i_s]) / dt
            pJx_pt_s0 = (Jx_upper - Jx_lower) / dt
            pJy_pt_s0 = (Jy_upper - Jy_lower) / dt
            pJz_pt_s0 = (Jz_upper - Jz_lower) / dt

        if boundary_mode == BOUNDARY_OPEN:
            _accumulate_open_contribution(x_o, y_o, z_o, x_s0, y_s0, z_s0,
                                          rho_s, prho_pt_s,
                                          Jx_s0, Jy_s0, Jz_s0, pJx_pt_s0, pJy_pt_s0, pJz_pt_s0,
                                          i_o, GEx, GEy, GEz, GBx, GBy, GBz, c, is_electrostatic, grid_truncation_radius, grid_taper_width, grid_taper_type)
        elif boundary_mode == BOUNDARY_PERIODIC:
            for mx in range(-max_img_x, max_img_x + 1):
                xs = x_s0 + mx * Lx_s
                for my in range(-max_img_y, max_img_y + 1):
                    ys = y_s0 + my * Ly_s
                    for mz in range(-max_img_z, max_img_z + 1):
                        zs = z_s0 + mz * Lz_s
                        _accumulate_open_contribution(x_o, y_o, z_o, xs, ys, zs,
                                                      rho_s, prho_pt_s,
                                                      Jx_s0, Jy_s0, Jz_s0, pJx_pt_s0, pJy_pt_s0, pJz_pt_s0,
                                                      i_o, GEx, GEy, GEz, GBx, GBy, GBz, c, is_electrostatic, grid_truncation_radius, grid_taper_width, grid_taper_type)
        else:
            rel_x = x_s0 - x_left_boundary_s
            rel_y = y_s0 - y_left_boundary_s
            rel_z = z_s0 - z_left_boundary_s
            period_x = 2.0 * Lx_s
            period_y = 2.0 * Ly_s
            period_z = 2.0 * Lz_s
            for mx in range(-max_img_x, max_img_x + 1):
                for rx in range(2):
                    sign_x = -1.0 if rx == 1 else 1.0
                    xs_rel = rel_x if rx == 0 else (2.0 * Lx_s - rel_x)
                    xs = x_left_boundary_s + xs_rel + mx * period_x
                    for my in range(-max_img_y, max_img_y + 1):
                        for ry in range(2):
                            sign_y = -1.0 if ry == 1 else 1.0
                            ys_rel = rel_y if ry == 0 else (2.0 * Ly_s - rel_y)
                            ys = y_left_boundary_s + ys_rel + my * period_y
                            for mz in range(-max_img_z, max_img_z + 1):
                                for rz in range(2):
                                    sign_z = -1.0 if rz == 1 else 1.0
                                    zs_rel = rel_z if rz == 0 else (2.0 * Lz_s - rel_z)
                                    zs = z_left_boundary_s + zs_rel + mz * period_z
                                    _accumulate_open_contribution(
                                        x_o, y_o, z_o, xs, ys, zs,
                                        rho_s, prho_pt_s,
                                        sign_x * Jx_s0, sign_y * Jy_s0, sign_z * Jz_s0,
                                        sign_x * pJx_pt_s0, sign_y * pJy_pt_s0, sign_z * pJz_pt_s0,
                                        i_o, GEx, GEy, GEz, GBx, GBy, GBz, c, is_electrostatic, grid_truncation_radius, grid_taper_width, grid_taper_type
                                    )



@cuda.jit
def Jefimenko_kernel_quasi_neutral(rho_GPU_s, Jx_GPU_s, Jy_GPU_s, Jz_GPU_s,
                                   len_time_snapshots, i_time, dt, t,
                                   total_grid_o, total_grid_s,
                                   x_grid_size_o, y_grid_size_o, z_grid_size_o,
                                   x_grid_size_s, y_grid_size_s, z_grid_size_s,
                                   dx_o, dy_o, dz_o,
                                   dx_s, dy_s, dz_s,
                                   x_left_boundary_o, y_left_boundary_o, z_left_boundary_o,
                                   x_left_boundary_s, y_left_boundary_s, z_left_boundary_s,
                                   GEx, GEy, GEz, GBx, GBy, GBz, coeff_E, coeff_B, c, all_grids):

    # the i-th spatial grid
    i_grid = cuda.grid(1)

    # loop through each spatial grid in the observation region
    if i_grid < all_grids:
        # convert 1D grid index into 3D
        iz_o = i_grid % z_grid_size_o
        iz_rest_o = i_grid // z_grid_size_o
        iy_o = iz_rest_o % y_grid_size_o
        iy_rest_o = iz_rest_o // y_grid_size_o
        ix_o = iy_rest_o % x_grid_size_o
        ix_rest_o = iy_rest_o // x_grid_size_o
        i_o = iz_o + iy_o * z_grid_size_o + ix_o * z_grid_size_o * y_grid_size_o

        iz_s = ix_rest_o % z_grid_size_s
        iz_rest_s = ix_rest_o // z_grid_size_s
        iy_s = iz_rest_s % y_grid_size_s
        iy_rest_s = iz_rest_s // y_grid_size_s
        ix_s = iy_rest_s % x_grid_size_s
        i_s = iz_s + iy_s * z_grid_size_s + ix_s * z_grid_size_s * y_grid_size_s

        # express the position corresponding to the index
        x_o, y_o, z_o = real_position(ix_o, dx_o, x_left_boundary_o), \
                        real_position(iy_o, dy_o, y_left_boundary_o), \
                        real_position(iz_o, dz_o, z_left_boundary_o)

        # positions
        x_s, y_s, z_s = real_position(ix_s, dx_s, x_left_boundary_s), \
                        real_position(iy_s, dy_s, y_left_boundary_s), \
                        real_position(iz_s, dz_s, z_left_boundary_s)

        # dr = r - r' = r_o - r_s
        dr = r_distance(x_o, y_o, z_o, x_s, y_s, z_s)

        # exclude the divergence when dr=0 (this is the screening)
        if dr > 10 ** (-19):
            tr = t - dr / c

            # no updation since signal has not arrived yet
            if tr < 0.0:
                pass
            else:
                tau = tr / dt
                tau_floor = math.floor(tau)
                frac = tau - tau_floor
                tau_floor_int = int(tau_floor)

                if frac < 1.0e-12:
                    if tau_floor_int > 0:
                        idx_lower_abs = tau_floor_int - 1
                        idx_upper_abs = tau_floor_int
                        alpha = 1.0
                    else:
                        idx_lower_abs = 0
                        idx_upper_abs = 0
                        alpha = 0.0
                else:
                    idx_lower_abs = tau_floor_int
                    idx_upper_abs = tau_floor_int + 1
                    alpha = frac

                current_abs = int(math.floor(t / dt + 1.0e-9))
                oldest_abs = current_abs - len_time_snapshots + 1
                if oldest_abs < 0:
                    oldest_abs = 0

                if idx_lower_abs < oldest_abs or idx_upper_abs < oldest_abs:
                    return
                if idx_upper_abs > current_abs:
                    idx_lower_abs = current_abs
                    idx_upper_abs = current_abs
                    alpha = 0.0

                idx_lower = idx_lower_abs % len_time_snapshots
                idx_upper = idx_upper_abs % len_time_snapshots

                Jx_lower = Jx_GPU_s[idx_lower, i_s]
                Jx_upper = Jx_GPU_s[idx_upper, i_s]
                Jy_lower = Jy_GPU_s[idx_lower, i_s]
                Jy_upper = Jy_GPU_s[idx_upper, i_s]
                Jz_lower = Jz_GPU_s[idx_lower, i_s]
                Jz_upper = Jz_GPU_s[idx_upper, i_s]

                Jx_s = (1.0 - alpha) * Jx_lower + alpha * Jx_upper
                Jy_s = (1.0 - alpha) * Jy_lower + alpha * Jy_upper
                Jz_s = (1.0 - alpha) * Jz_lower + alpha * Jz_upper

                if idx_upper_abs == idx_lower_abs:
                    pJx_pt_s = 0.0
                    pJy_pt_s = 0.0
                    pJz_pt_s = 0.0
                else:
                    pJx_pt_s = (Jx_upper - Jx_lower) / dt
                    pJy_pt_s = (Jy_upper - Jy_lower) / dt
                    pJz_pt_s = (Jz_upper - Jz_lower) / dt

                cuda.atomic.add(GEx, i_o, (-1.0 / dr / c ** 2 * pJx_pt_s))
                cuda.atomic.add(GEy, i_o, (-1.0 / dr / c ** 2 * pJy_pt_s))
                cuda.atomic.add(GEz, i_o, (-1.0 / dr / c ** 2 * pJz_pt_s))
                cuda.atomic.add(GBx, i_o, ((((y_o - y_s) * Jz_s - (z_o - z_s) * Jy_s) / dr ** 3 + ((y_o - y_s) * pJz_pt_s - (z_o - z_s) * pJy_pt_s) / dr ** 2 / c)))
                cuda.atomic.add(GBy, i_o, ((((z_o - z_s) * Jx_s - (x_o - x_s) * Jz_s) / dr ** 3 + ((z_o - z_s) * pJx_pt_s - (x_o - x_s) * pJz_pt_s) / dr ** 2 / c)))
                cuda.atomic.add(GBz, i_o, ((((x_o - x_s) * Jy_s - (y_o - y_s) * Jx_s) / dr ** 3 + ((x_o - x_s) * pJy_pt_s - (y_o - y_s) * pJx_pt_s) / dr ** 2 / c)))

# ----------------------------------------------------------------------
# Step 1-D: staggered-face plasma-current Jefimenko path
# ----------------------------------------------------------------------
@cuda.jit(device=True)
def _staggered_history_value_and_derivative(history, source_index,
                                             len_time_snapshots, dt, t, dr, c):
    """Piecewise-linear retarded history sample for one staggered component.

    Returns ``(valid, value, dvalue_dt)`` using exactly the same retarded-time
    indexing convention as the cell-centered production Jefimenko kernel.
    """
    tr = t - dr / c
    if tr < 0.0:
        return False, 0.0, 0.0

    current_abs = int(math.floor(t / dt + 1.0e-9))
    oldest_abs = current_abs - len_time_snapshots + 1
    if oldest_abs < 0:
        oldest_abs = 0

    tau = tr / dt
    tau_floor = math.floor(tau)
    frac = tau - tau_floor
    tau_floor_int = int(tau_floor)

    if frac < 1.0e-12:
        if tau_floor_int > 0:
            idx_lower_abs = tau_floor_int - 1
            idx_upper_abs = tau_floor_int
            alpha = 1.0
        else:
            idx_lower_abs = 0
            idx_upper_abs = 0
            alpha = 0.0
    else:
        idx_lower_abs = tau_floor_int
        idx_upper_abs = tau_floor_int + 1
        alpha = frac

    if idx_lower_abs < oldest_abs or idx_upper_abs < oldest_abs:
        return False, 0.0, 0.0
    if idx_upper_abs > current_abs:
        idx_lower_abs = current_abs
        idx_upper_abs = current_abs
        alpha = 0.0

    idx_lower = idx_lower_abs % len_time_snapshots
    idx_upper = idx_upper_abs % len_time_snapshots
    lower = history[idx_lower, source_index]
    upper = history[idx_upper, source_index]
    value = (1.0 - alpha) * lower + alpha * upper
    if idx_upper_abs == idx_lower_abs:
        derivative = 0.0
    else:
        derivative = (upper - lower) / dt
    return True, value, derivative


@cuda.jit
def Jefimenko_staggered_current_kernel_observation_parallel(
        Jx_face_GPU, Jy_face_GPU, Jz_face_GPU,
        len_time_snapshots, dt, t,
        nx_s, ny_s, nz_s,
        total_grid_o, nx_o, ny_o, nz_o,
        dx_o, dy_o, dz_o,
        dx_s, dy_s, dz_s,
        x_left_o, y_left_o, z_left_o,
        x_left_s, y_left_s, z_left_s,
        GEx, GEy, GEz, GBx, GBy, GBz,
        c, is_electrostatic,
        truncation_radius, taper_width, taper_type):
    """Add current terms with each component kept at its true staggered face.

    Charge density and d(rho)/dt remain on the cell-centered grid and are
    evaluated by the ordinary Jefimenko kernel.  This kernel adds only the
    current-dependent electric and magnetic terms, using

      Jx: (x_left+i*dx,       y_left+(j+1/2)dy, z_left+(k+1/2)dz)
      Jy: (x_left+(i+1/2)dx, y_left+j*dy,       z_left+(k+1/2)dz)
      Jz: (x_left+(i+1/2)dx, y_left+(j+1/2)dy, z_left+k*dz)

    Step 1-D intentionally supports the open-boundary source geometry first;
    periodic/reflective image handling stays on the legacy path until the
    staggered formulation is validated.
    """
    io = cuda.grid(1)
    if io >= total_grid_o:
        return
    if is_electrostatic:
        # There are no J or dJ/dt terms in the quasi-electrostatic path.
        return

    iz_o = io % nz_o
    rest_o = io // nz_o
    iy_o = rest_o % ny_o
    ix_o = rest_o // ny_o
    x_o = real_position(ix_o, dx_o, x_left_o)
    y_o = real_position(iy_o, dy_o, y_left_o)
    z_o = real_position(iz_o, dz_o, z_left_o)

    ex = 0.0
    ey = 0.0
    ez = 0.0
    bx = 0.0
    by = 0.0
    bz = 0.0

    # Jx face grid: (nx+1, ny, nz)
    total_jx = (nx_s + 1) * ny_s * nz_s
    for s in range(total_jx):
        k = s % nz_s
        rest = s // nz_s
        j = rest % ny_s
        i = rest // ny_s
        x_s = x_left_s + i * dx_s
        y_s = y_left_s + (j + 0.5) * dy_s
        z_s = z_left_s + (k + 0.5) * dz_s
        drx = x_o - x_s
        dry = y_o - y_s
        drz = z_o - z_s
        dr2 = drx * drx + dry * dry + drz * drz
        if dr2 <= 1.0e-38:
            continue
        dr = math.sqrt(dr2)
        taper = _truncation_weight(dr, truncation_radius, taper_width, taper_type)
        if taper <= 0.0:
            continue
        valid, J, dJ = _staggered_history_value_and_derivative(
            Jx_face_GPU, s, len_time_snapshots, dt, t, dr, c
        )
        if not valid:
            continue
        ex += taper * (-dJ / (dr * c * c))
        by += taper * (drz * J / (dr ** 3) + drz * dJ / (dr * dr * c))
        bz += taper * (-dry * J / (dr ** 3) - dry * dJ / (dr * dr * c))

    # Jy face grid: (nx, ny+1, nz)
    total_jy = nx_s * (ny_s + 1) * nz_s
    for s in range(total_jy):
        k = s % nz_s
        rest = s // nz_s
        j = rest % (ny_s + 1)
        i = rest // (ny_s + 1)
        x_s = x_left_s + (i + 0.5) * dx_s
        y_s = y_left_s + j * dy_s
        z_s = z_left_s + (k + 0.5) * dz_s
        drx = x_o - x_s
        dry = y_o - y_s
        drz = z_o - z_s
        dr2 = drx * drx + dry * dry + drz * drz
        if dr2 <= 1.0e-38:
            continue
        dr = math.sqrt(dr2)
        taper = _truncation_weight(dr, truncation_radius, taper_width, taper_type)
        if taper <= 0.0:
            continue
        valid, J, dJ = _staggered_history_value_and_derivative(
            Jy_face_GPU, s, len_time_snapshots, dt, t, dr, c
        )
        if not valid:
            continue
        ey += taper * (-dJ / (dr * c * c))
        bx += taper * (-drz * J / (dr ** 3) - drz * dJ / (dr * dr * c))
        bz += taper * (drx * J / (dr ** 3) + drx * dJ / (dr * dr * c))

    # Jz face grid: (nx, ny, nz+1)
    total_jz = nx_s * ny_s * (nz_s + 1)
    for s in range(total_jz):
        k = s % (nz_s + 1)
        rest = s // (nz_s + 1)
        j = rest % ny_s
        i = rest // ny_s
        x_s = x_left_s + (i + 0.5) * dx_s
        y_s = y_left_s + (j + 0.5) * dy_s
        z_s = z_left_s + k * dz_s
        drx = x_o - x_s
        dry = y_o - y_s
        drz = z_o - z_s
        dr2 = drx * drx + dry * dry + drz * drz
        if dr2 <= 1.0e-38:
            continue
        dr = math.sqrt(dr2)
        taper = _truncation_weight(dr, truncation_radius, taper_width, taper_type)
        if taper <= 0.0:
            continue
        valid, J, dJ = _staggered_history_value_and_derivative(
            Jz_face_GPU, s, len_time_snapshots, dt, t, dr, c
        )
        if not valid:
            continue
        ez += taper * (-dJ / (dr * c * c))
        bx += taper * (dry * J / (dr ** 3) + dry * dJ / (dr * dr * c))
        by += taper * (-drx * J / (dr ** 3) - drx * dJ / (dr * dr * c))

    # One observation thread owns these six output entries, so no atomics are
    # required.  The ordinary rho kernel has already initialized the arrays.
    GEx[io] += ex
    GEy[io] += ey
    GEz[io] += ez
    GBx[io] += bx
    GBy[io] += by
    GBz[io] += bz



@cuda.jit
def Jefimenko_staggered_current_kernel_atomic(
        Jx_face_GPU, Jy_face_GPU, Jz_face_GPU,
        len_time_snapshots, dt, t,
        nx_s, ny_s, nz_s,
        total_grid_o, nx_o, ny_o, nz_o,
        dx_o, dy_o, dz_o,
        dx_s, dy_s, dz_s,
        x_left_o, y_left_o, z_left_o,
        x_left_s, y_left_s, z_left_s,
        GEx, GEy, GEz, GBx, GBy, GBz,
        c,
        truncation_radius, taper_width, taper_type):
    """Atomic-add counterpart of the native staggered-face full-EM current kernel.

    One CUDA thread owns one (observation, staggered-current-face) pair and
    atomically accumulates the current-dependent Jefimenko terms. Charge-density
    and d(rho)/dt terms remain handled by the ordinary grid kernel.

    This kernel is mathematically equivalent to
    Jefimenko_staggered_current_kernel_observation_parallel for open boundaries;
    the difference is only the GPU accumulation strategy.
    """
    pair = cuda.grid(1)

    total_jx = (nx_s + 1) * ny_s * nz_s
    total_jy = nx_s * (ny_s + 1) * nz_s
    total_jz = nx_s * ny_s * (nz_s + 1)
    total_faces = total_jx + total_jy + total_jz
    total_pairs = total_grid_o * total_faces

    if pair >= total_pairs:
        return

    io = pair // total_faces
    sf = pair - io * total_faces

    iz_o = io % nz_o
    rest_o = io // nz_o
    iy_o = rest_o % ny_o
    ix_o = rest_o // ny_o

    x_o = real_position(ix_o, dx_o, x_left_o)
    y_o = real_position(iy_o, dy_o, y_left_o)
    z_o = real_position(iz_o, dz_o, z_left_o)

    # Jx face: (nx+1, ny, nz)
    if sf < total_jx:
        s = sf
        k = s % nz_s
        rest = s // nz_s
        j = rest % ny_s
        i = rest // ny_s

        x_s = x_left_s + i * dx_s
        y_s = y_left_s + (j + 0.5) * dy_s
        z_s = z_left_s + (k + 0.5) * dz_s

        drx = x_o - x_s
        dry = y_o - y_s
        drz = z_o - z_s
        dr2 = drx * drx + dry * dry + drz * drz
        if dr2 <= 1.0e-38:
            return
        dr = math.sqrt(dr2)
        taper = _truncation_weight(dr, truncation_radius, taper_width, taper_type)
        if taper <= 0.0:
            return
        valid, J, dJ = _staggered_history_value_and_derivative(
            Jx_face_GPU, s, len_time_snapshots, dt, t, dr, c
        )
        if not valid:
            return

        cuda.atomic.add(GEx, io, taper * (-dJ / (dr * c * c)))
        cuda.atomic.add(GBy, io, taper * (drz * J / (dr ** 3) + drz * dJ / (dr * dr * c)))
        cuda.atomic.add(GBz, io, taper * (-dry * J / (dr ** 3) - dry * dJ / (dr * dr * c)))
        return

    # Jy face: (nx, ny+1, nz)
    sf -= total_jx
    if sf < total_jy:
        s = sf
        k = s % nz_s
        rest = s // nz_s
        j = rest % (ny_s + 1)
        i = rest // (ny_s + 1)

        x_s = x_left_s + (i + 0.5) * dx_s
        y_s = y_left_s + j * dy_s
        z_s = z_left_s + (k + 0.5) * dz_s

        drx = x_o - x_s
        dry = y_o - y_s
        drz = z_o - z_s
        dr2 = drx * drx + dry * dry + drz * drz
        if dr2 <= 1.0e-38:
            return
        dr = math.sqrt(dr2)
        taper = _truncation_weight(dr, truncation_radius, taper_width, taper_type)
        if taper <= 0.0:
            return
        valid, J, dJ = _staggered_history_value_and_derivative(
            Jy_face_GPU, s, len_time_snapshots, dt, t, dr, c
        )
        if not valid:
            return

        cuda.atomic.add(GEy, io, taper * (-dJ / (dr * c * c)))
        cuda.atomic.add(GBx, io, taper * (-drz * J / (dr ** 3) - drz * dJ / (dr * dr * c)))
        cuda.atomic.add(GBz, io, taper * (drx * J / (dr ** 3) + drx * dJ / (dr * dr * c)))
        return

    # Jz face: (nx, ny, nz+1)
    s = sf - total_jy
    if s >= total_jz:
        return

    k = s % (nz_s + 1)
    rest = s // (nz_s + 1)
    j = rest % ny_s
    i = rest // ny_s

    x_s = x_left_s + (i + 0.5) * dx_s
    y_s = y_left_s + (j + 0.5) * dy_s
    z_s = z_left_s + k * dz_s

    drx = x_o - x_s
    dry = y_o - y_s
    drz = z_o - z_s
    dr2 = drx * drx + dry * dry + drz * drz
    if dr2 <= 1.0e-38:
        return
    dr = math.sqrt(dr2)
    taper = _truncation_weight(dr, truncation_radius, taper_width, taper_type)
    if taper <= 0.0:
        return
    valid, J, dJ = _staggered_history_value_and_derivative(
        Jz_face_GPU, s, len_time_snapshots, dt, t, dr, c
    )
    if not valid:
        return

    cuda.atomic.add(GEz, io, taper * (-dJ / (dr * c * c)))
    cuda.atomic.add(GBx, io, taper * (dry * J / (dr ** 3) + dry * dJ / (dr * dr * c)))
    cuda.atomic.add(GBy, io, taper * (-drx * J / (dr ** 3) - drx * dJ / (dr * dr * c)))


# -----------------------------------------------------------------------------
# Step-2B continuity-consistent MAS electric-dipole point sources
# -----------------------------------------------------------------------------
@cuda.jit
def Jefimenko_point_dipole_kernel_observation_parallel(
        point_src_x, point_src_y, point_src_z, point_weights,
        point_Px_GPU, point_Py_GPU, point_Pz_GPU,
        point_Kx_GPU, point_Ky_GPU, point_Kz_GPU,
        point_source_count, point_softening,
        point_truncation_radius, point_taper_width, point_taper_type,
        len_time_snapshots, dt, t,
        dx_s, dy_s, dz_s,
        total_grid_o,
        x_grid_size_o, y_grid_size_o, z_grid_size_o,
        dx_o, dy_o, dz_o,
        x_left_boundary_o, y_left_boundary_o, z_left_boundary_o,
        GEx, GEy, GEz, GBx, GBy, GBz,
        c, is_electrostatic):
    """Add Step-2B electric-dipole MAS sources at native point locations.

    Source coefficients are surface-polarization density P [Q/L] and
    surface-current density K=dP/dt [I/L].  Multiplication by point_weights
    [L^2] yields the lumped dipole/current moments p [Q L] and m [I L].

    K is piecewise linear in time.  P is evaluated as the exact integral of
    that linear K on each retarded interval, and Kdot is the interval slope.
    The returned accumulators are normalized by the source-grid cell volume so
    the caller's usual coeff_E/coeff_B scaling can be reused.
    """
    i_o = cuda.grid(1)
    if i_o >= total_grid_o or point_source_count <= 0:
        return

    iz_o = i_o % z_grid_size_o
    iy_o = (i_o // z_grid_size_o) % y_grid_size_o
    ix_o = i_o // (y_grid_size_o * z_grid_size_o)
    x_o = x_left_boundary_o + (ix_o + 0.5) * dx_o
    y_o = y_left_boundary_o + (iy_o + 0.5) * dy_o
    z_o = z_left_boundary_o + (iz_o + 0.5) * dz_o

    ex = 0.0
    ey = 0.0
    ez = 0.0
    bx = 0.0
    by = 0.0
    bz = 0.0

    cell_volume = dx_s * dy_s * dz_s
    if cell_volume <= 1.0e-300:
        cell_volume = 1.0
    current_abs = int(math.floor(t / dt + 1.0e-9))
    oldest_abs = current_abs - len_time_snapshots + 1
    if oldest_abs < 0:
        oldest_abs = 0

    for js in range(point_source_count):
        rx = x_o - point_src_x[js]
        ry = y_o - point_src_y[js]
        rz = z_o - point_src_z[js]
        R2 = rx*rx + ry*ry + rz*rz + point_softening*point_softening
        if R2 <= 1.0e-38:
            continue
        R = math.sqrt(R2)
        taper = _truncation_weight(R, point_truncation_radius, point_taper_width, point_taper_type)
        if taper <= 0.0:
            continue

        Px = 0.0
        Py = 0.0
        Pz = 0.0
        Kx = 0.0
        Ky = 0.0
        Kz = 0.0
        Kdx = 0.0
        Kdy = 0.0
        Kdz = 0.0

        if is_electrostatic:
            idx = current_abs % len_time_snapshots
            Px = point_Px_GPU[idx, js]
            Py = point_Py_GPU[idx, js]
            Pz = point_Pz_GPU[idx, js]
            Kx = point_Kx_GPU[idx, js]
            Ky = point_Ky_GPU[idx, js]
            Kz = point_Kz_GPU[idx, js]
        else:
            tr = t - R / c
            if tr < 0.0:
                continue
            tau = tr / dt
            k = int(math.floor(tau))
            alpha = tau - math.floor(tau)
            if alpha < 1.0e-12:
                if k < oldest_abs or k > current_abs:
                    continue
                sk = k % len_time_snapshots
                Px = point_Px_GPU[sk, js]
                Py = point_Py_GPU[sk, js]
                Pz = point_Pz_GPU[sk, js]
                Kx = point_Kx_GPU[sk, js]
                Ky = point_Ky_GPU[sk, js]
                Kz = point_Kz_GPU[sk, js]
                if k - 1 >= oldest_abs:
                    sp = (k - 1) % len_time_snapshots
                    Kdx = (Kx - point_Kx_GPU[sp, js]) / dt
                    Kdy = (Ky - point_Ky_GPU[sp, js]) / dt
                    Kdz = (Kz - point_Kz_GPU[sp, js]) / dt
            else:
                k1 = k + 1
                if k < oldest_abs or k1 < oldest_abs or k1 > current_abs:
                    continue
                s0 = k % len_time_snapshots
                s1 = k1 % len_time_snapshots
                P0x = point_Px_GPU[s0, js]
                P0y = point_Py_GPU[s0, js]
                P0z = point_Pz_GPU[s0, js]
                K0x = point_Kx_GPU[s0, js]
                K0y = point_Ky_GPU[s0, js]
                K0z = point_Kz_GPU[s0, js]
                dKx = point_Kx_GPU[s1, js] - K0x
                dKy = point_Ky_GPU[s1, js] - K0y
                dKz = point_Kz_GPU[s1, js] - K0z
                Kx = K0x + alpha * dKx
                Ky = K0y + alpha * dKy
                Kz = K0z + alpha * dKz
                Kdx = dKx / dt
                Kdy = dKy / dt
                Kdz = dKz / dt
                aa = 0.5 * alpha * alpha
                Px = P0x + dt * (alpha * K0x + aa * dKx)
                Py = P0y + dt * (alpha * K0y + aa * dKy)
                Pz = P0z + dt * (alpha * K0z + aa * dKz)

        invR = 1.0 / R
        invR2 = invR * invR
        invR3 = invR2 * invR
        invR4 = invR3 * invR
        invR5 = invR4 * invR
        scale = point_weights[js] / cell_volume * taper

        rP = rx*Px + ry*Py + rz*Pz
        # Electrostatic dipole term: (3 rr/R^5 - I/R^3) P.
        ex += scale * (3.0 * rx * rP * invR5 - Px * invR3)
        ey += scale * (3.0 * ry * rP * invR5 - Py * invR3)
        ez += scale * (3.0 * rz * rP * invR5 - Pz * invR3)

        if not is_electrostatic:
            rK = rx*Kx + ry*Ky + rz*Kz
            rKd = rx*Kdx + ry*Kdy + rz*Kdz
            ex += scale * ((3.0*rx*rK*invR4 - Kx*invR2)/c +
                           (rx*rKd*invR3 - Kdx*invR)/(c*c))
            ey += scale * ((3.0*ry*rK*invR4 - Ky*invR2)/c +
                           (ry*rKd*invR3 - Kdy*invR)/(c*c))
            ez += scale * ((3.0*rz*rK*invR4 - Kz*invR2)/c +
                           (rz*rKd*invR3 - Kdz*invR)/(c*c))

        # Raw B uses r x K because the caller multiplies by a negative coeff_B.
        bx += scale * ((ry*Kz - rz*Ky) * invR3 +
                       (ry*Kdz - rz*Kdy) * invR2 / c)
        by += scale * ((rz*Kx - rx*Kz) * invR3 +
                       (rz*Kdx - rx*Kdz) * invR2 / c)
        bz += scale * ((rx*Ky - ry*Kx) * invR3 +
                       (rx*Kdy - ry*Kdx) * invR2 / c)

    GEx[i_o] += ex
    GEy[i_o] += ey
    GEz[i_o] += ez
    GBx[i_o] += bx
    GBy[i_o] += by
    GBz[i_o] += bz
