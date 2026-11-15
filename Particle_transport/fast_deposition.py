# -*- coding: utf-8 -*-
"""
Fast Numba deposition backend for Particle_transport.main.pic_transport.

This module installs a validated CPU Numba-JIT fast path for deposition.

Runtime behavior
----------------
- particle_order == 1: Step-1C charge-conserving CIC current deposition.
- particle_order == 2: legacy fast TSC path (not yet continuity-validated).
- other orders: fall back to original Python/Numpy methods.

It accelerates:
    pic_transport._deposit_rho_numpy(...)
    pic_transport._deposit_current_esirkepov_style_numpy(...)

Important
---------
This is not a CUDA deposition kernel. It is a CPU Numba-JIT backend that removes
the Python-level particle/stencil loops. It still follows the original code's
CPU-side deposition structure in _evaluate_rho_J_esirkepov(), including
cp.asnumpy(...) before deposition and cp.asarray(...) after deposition.

Disable for regression testing:
    JEFIPIC_FAST_DEPOSITION=0 python your_case.py ...

Verbose installation/debug output:
    JEFIPIC_FAST_DEPOSITION_VERBOSE=1 python your_case.py ...
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Any, Dict

try:
    import numpy as np
    from numba import njit
except Exception as exc:  # pragma: no cover
    np = None
    njit = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


FAST_DEPOSITION_STATS: Dict[str, Any] = {
    "installed": False,
    "rho_calls_fast": 0,
    "current_calls_fast": 0,
    "rho_calls_fallback": 0,
    "current_calls_fallback": 0,
    "rho_time_fast_s": 0.0,
    "current_time_fast_s": 0.0,
    "rho_time_fallback_s": 0.0,
    "current_time_fallback_s": 0.0,
    "rho_calls_order1": 0,
    "rho_calls_order2": 0,
    "current_calls_order1": 0,
    "current_calls_order2": 0,
}


def get_fast_deposition_stats() -> Dict[str, Any]:
    return dict(FAST_DEPOSITION_STATS)


def reset_fast_deposition_stats() -> None:
    for key in FAST_DEPOSITION_STATS:
        if key == "installed":
            continue
        FAST_DEPOSITION_STATS[key] = 0.0 if key.endswith("_s") else 0


def _verbose_enabled() -> bool:
    return os.environ.get("JEFIPIC_FAST_DEPOSITION_VERBOSE", "0").strip().lower() in (
        "1", "true", "yes", "on"
    )


if njit is not None:

    @njit(cache=True)
    def _shape_weights_numba(s: float, n: int, order: int, idx_out: np.ndarray, w_out: np.ndarray) -> int:
        """
        Numba equivalent of pic_transport._shape1_weights/_shape2_weights.

        order <= 1:
            CIC / linear, up to 2 entries, no boundary renormalization.

        order > 1:
            TSC / quadratic, up to 3 entries, with the same valid-cell
            renormalization used by the original _shape2_weights().
        """
        count = 0

        if order <= 1:
            base = int(np.floor(s))
            frac = s - base

            idx = base
            w = 1.0 - frac
            if 0 <= idx < n and w != 0.0:
                idx_out[count] = idx
                w_out[count] = w
                count += 1

            idx = base + 1
            w = frac
            if 0 <= idx < n and w != 0.0:
                idx_out[count] = idx
                w_out[count] = w
                count += 1

            return count

        center = int(np.floor(s + 0.5))
        total = 0.0

        for offset in range(-1, 2):
            idx = center + offset
            if 0 <= idx < n:
                d = abs(s - idx)
                if d < 0.5:
                    w = 0.75 - d * d
                elif d < 1.5:
                    tmp = 1.5 - d
                    w = 0.5 * tmp * tmp
                else:
                    w = 0.0

                if w != 0.0:
                    idx_out[count] = idx
                    w_out[count] = w
                    total += w
                    count += 1

        if total > 0.0:
            for i in range(count):
                w_out[i] = w_out[i] / total

        return count


    @njit(cache=True)
    def _insert_t_numba(ts: np.ndarray, n_ts: int, t: float, eps: float) -> int:
        if t <= eps or t >= 1.0 - eps:
            return n_ts

        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0

        for i in range(n_ts):
            if abs(ts[i] - t) <= eps:
                return n_ts

        pos = n_ts
        for i in range(n_ts):
            if t < ts[i]:
                pos = i
                break

        for j in range(n_ts, pos, -1):
            ts[j] = ts[j - 1]
        ts[pos] = t
        return n_ts + 1


    @njit(cache=True)
    def _splits_numba(
        p0: np.ndarray,
        p1: np.ndarray,
        left_x: float,
        left_y: float,
        left_z: float,
        dx: float,
        dy: float,
        dz: float,
        nx: int,
        ny: int,
        nz: int,
        ts: np.ndarray,
    ) -> int:
        """
        Numba version of _split_segment_for_flux_deposition().
        """
        eps = 1.0e-14
        ts[0] = 0.0
        ts[1] = 1.0
        n_ts = 2

        d0 = p1[0] - p0[0]
        d1 = p1[1] - p0[1]
        d2 = p1[2] - p0[2]

        if abs(d0) >= eps:
            s0 = (p0[0] - left_x) / dx
            s1 = (p1[0] - left_x) / dx
            lo = s0 if s0 < s1 else s1
            hi = s1 if s0 < s1 else s0
            k_start = int(np.floor(lo + 0.5))
            k_end = int(np.floor(hi + 0.5))
            for k in range(k_start, k_end + 1):
                boundary = left_x + (k + 0.5) * dx
                t = (boundary - p0[0]) / d0
                n_ts = _insert_t_numba(ts, n_ts, t, eps)

        if abs(d1) >= eps:
            s0 = (p0[1] - left_y) / dy
            s1 = (p1[1] - left_y) / dy
            lo = s0 if s0 < s1 else s1
            hi = s1 if s0 < s1 else s0
            k_start = int(np.floor(lo + 0.5))
            k_end = int(np.floor(hi + 0.5))
            for k in range(k_start, k_end + 1):
                boundary = left_y + (k + 0.5) * dy
                t = (boundary - p0[1]) / d1
                n_ts = _insert_t_numba(ts, n_ts, t, eps)

        if abs(d2) >= eps:
            s0 = (p0[2] - left_z) / dz
            s1 = (p1[2] - left_z) / dz
            lo = s0 if s0 < s1 else s1
            hi = s1 if s0 < s1 else s0
            k_start = int(np.floor(lo + 0.5))
            k_end = int(np.floor(hi + 0.5))
            for k in range(k_start, k_end + 1):
                boundary = left_z + (k + 0.5) * dz
                t = (boundary - p0[2]) / d2
                n_ts = _insert_t_numba(ts, n_ts, t, eps)

        return n_ts


    @njit(cache=True)
    def _allclose3_numba(p0: np.ndarray, p1: np.ndarray) -> bool:
        """
        Numba-compatible approximation to np.allclose(p0, p1) for 3-vectors.
        Uses numpy defaults rtol=1e-05, atol=1e-08.
        """
        for ax in range(3):
            if abs(p0[ax] - p1[ax]) > (1.0e-8 + 1.0e-5 * abs(p1[ax])):
                return False
        return True


    @njit(cache=True)
    def _deposit_rho_shape_numba(
        rho: np.ndarray,
        pos_np: np.ndarray,
        charge: float,
        order: int,
        left_x: float,
        left_y: float,
        left_z: float,
        right_x: float,
        right_y: float,
        right_z: float,
        dx: float,
        dy: float,
        dz: float,
        nx: int,
        ny: int,
        nz: int,
    ) -> None:
        ix = np.empty(3, dtype=np.int64)
        iy = np.empty(3, dtype=np.int64)
        iz = np.empty(3, dtype=np.int64)
        wx = np.empty(3, dtype=np.float64)
        wy = np.empty(3, dtype=np.float64)
        wz = np.empty(3, dtype=np.float64)

        for pidx in range(pos_np.shape[0]):
            px = pos_np[pidx, 0]
            py = pos_np[pidx, 1]
            pz = pos_np[pidx, 2]

            if not (left_x < px < right_x and left_y < py < right_y and left_z < pz < right_z):
                continue

            sx = (px - left_x) / dx - 0.5
            sy = (py - left_y) / dy - 0.5
            sz = (pz - left_z) / dz - 0.5

            nxw = _shape_weights_numba(sx, nx, order, ix, wx)
            nyw = _shape_weights_numba(sy, ny, order, iy, wy)
            nzw = _shape_weights_numba(sz, nz, order, iz, wz)

            for a in range(nxw):
                ia = ix[a]
                wa = wx[a]
                for b in range(nyw):
                    ib = iy[b]
                    wb = wy[b]
                    for c in range(nzw):
                        ic = iz[c]
                        wc = wz[c]
                        rho[ia, ib, ic] += charge * wa * wb * wc


    @njit(cache=True)
    def _cic_weight_at_index_numba(s: float, idx: int) -> float:
        base = int(np.floor(s))
        frac = s - base
        if idx == base:
            return 1.0 - frac
        if idx == base + 1:
            return frac
        return 0.0


    @njit(cache=True)
    def _cic_axis_endpoint_data_numba(
        position_a: float,
        position_b: float,
        left: float,
        spacing: float,
        idx_out: np.ndarray,
        mean_out: np.ndarray,
        delta_out: np.ndarray,
        face_idx_out: np.ndarray,
        face_flux_out: np.ndarray,
    ) -> tuple:
        """Numba helper for exact first-order continuity data on one axis."""
        sa = (position_a - left) / spacing - 0.5
        sb = (position_b - left) / spacing - 0.5
        ba = int(np.floor(sa))
        bb = int(np.floor(sb))
        lo = ba if ba < bb else bb
        hi_a = ba + 1
        hi_b = bb + 1
        hi = hi_a if hi_a > hi_b else hi_b

        n = hi - lo + 1
        for m in range(n):
            i = lo + m
            wa = _cic_weight_at_index_numba(sa, i)
            wb = _cic_weight_at_index_numba(sb, i)
            idx_out[m] = i
            mean_out[m] = 0.5 * (wa + wb)
            delta_out[m] = wb - wa

        face_idx_out[0] = lo
        face_flux_out[0] = 0.0
        running = 0.0
        for m in range(n):
            running -= delta_out[m]
            face_idx_out[m + 1] = lo + m + 1
            face_flux_out[m + 1] = running

        return n, n + 1


    @njit(cache=True)
    def _deposit_current_shape_numba(
        Jx_face: np.ndarray,
        Jy_face: np.ndarray,
        Jz_face: np.ndarray,
        prev_np: np.ndarray,
        curr_np: np.ndarray,
        active_np: np.ndarray,
        charge: float,
        order: int,
        dt: float,
        left_x: float,
        left_y: float,
        left_z: float,
        dx: float,
        dy: float,
        dz: float,
        nx: int,
        ny: int,
        nz: int,
    ) -> None:
        # Legacy shape-work arrays (used for order > 1).
        idx_a = np.empty(3, dtype=np.int64)
        idx_b = np.empty(3, dtype=np.int64)
        idx_c = np.empty(3, dtype=np.int64)
        w_a = np.empty(3, dtype=np.float64)
        w_b = np.empty(3, dtype=np.float64)
        w_c = np.empty(3, dtype=np.float64)

        # Exact CIC endpoint-difference work arrays.  A segment split at CIC
        # shape knots has at most three cell indices per axis; length four is
        # kept as a small roundoff guard.
        ix_cic = np.empty(4, dtype=np.int64)
        iy_cic = np.empty(4, dtype=np.int64)
        iz_cic = np.empty(4, dtype=np.int64)
        mx = np.empty(4, dtype=np.float64)
        my = np.empty(4, dtype=np.float64)
        mz = np.empty(4, dtype=np.float64)
        dsx = np.empty(4, dtype=np.float64)
        dsy = np.empty(4, dtype=np.float64)
        dsz = np.empty(4, dtype=np.float64)
        fx_idx = np.empty(5, dtype=np.int64)
        fy_idx = np.empty(5, dtype=np.int64)
        fz_idx = np.empty(5, dtype=np.int64)
        fx_val = np.empty(5, dtype=np.float64)
        fy_val = np.empty(5, dtype=np.float64)
        fz_val = np.empty(5, dtype=np.float64)

        ts = np.empty(nx + ny + nz + 8, dtype=np.float64)

        for pidx in range(prev_np.shape[0]):
            if not active_np[pidx]:
                continue

            p0 = prev_np[pidx]
            p1 = curr_np[pidx]

            # Step 1-Ec: never suppress a nonzero trajectory with an
            # allclose-style relative tolerance.  Large LHQCD coordinates make
            # that optimization non-conservative for slowly moving ions.
            n_ts = _splits_numba(
                p0,
                p1,
                left_x,
                left_y,
                left_z,
                dx,
                dy,
                dz,
                nx,
                ny,
                nz,
                ts,
            )

            d_all0 = p1[0] - p0[0]
            d_all1 = p1[1] - p0[1]
            d_all2 = p1[2] - p0[2]

            for m in range(n_ts - 1):
                t0 = ts[m]
                t1 = ts[m + 1]
                if t1 <= t0:
                    continue

                pa0 = p0[0] + d_all0 * t0
                pa1 = p0[1] + d_all1 * t0
                pa2 = p0[2] + d_all2 * t0

                pb0 = p0[0] + d_all0 * t1
                pb1 = p0[1] + d_all1 * t1
                pb2 = p0[2] + d_all2 * t1

                if order <= 1:
                    nix, nfx = _cic_axis_endpoint_data_numba(
                        pa0, pb0, left_x, dx, ix_cic, mx, dsx, fx_idx, fx_val
                    )
                    niy, nfy = _cic_axis_endpoint_data_numba(
                        pa1, pb1, left_y, dy, iy_cic, my, dsy, fy_idx, fy_val
                    )
                    niz, nfz = _cic_axis_endpoint_data_numba(
                        pa2, pb2, left_z, dz, iz_cic, mz, dsz, fz_idx, fz_val
                    )

                    coeff_x = charge * dx / dt
                    coeff_y = charge * dy / dt
                    coeff_z = charge * dz / dt

                    for af in range(nfx):
                        face = fx_idx[af]
                        flux = fx_val[af]
                        if flux == 0.0 or face < 0 or face > nx:
                            continue
                        for b in range(niy):
                            j = iy_cic[b]
                            if j < 0 or j >= ny:
                                continue
                            for c in range(niz):
                                k = iz_cic[c]
                                if k < 0 or k >= nz:
                                    continue
                                transverse = my[b] * mz[c] + dsy[b] * dsz[c] / 12.0
                                if transverse != 0.0:
                                    Jx_face[face, j, k] += coeff_x * flux * transverse

                    for a in range(nix):
                        i = ix_cic[a]
                        if i < 0 or i >= nx:
                            continue
                        for bf in range(nfy):
                            face = fy_idx[bf]
                            flux = fy_val[bf]
                            if flux == 0.0 or face < 0 or face > ny:
                                continue
                            for c in range(niz):
                                k = iz_cic[c]
                                if k < 0 or k >= nz:
                                    continue
                                transverse = mz[c] * mx[a] + dsz[c] * dsx[a] / 12.0
                                if transverse != 0.0:
                                    Jy_face[i, face, k] += coeff_y * flux * transverse

                    for a in range(nix):
                        i = ix_cic[a]
                        if i < 0 or i >= nx:
                            continue
                        for b in range(niy):
                            j = iy_cic[b]
                            if j < 0 or j >= ny:
                                continue
                            transverse = mx[a] * my[b] + dsx[a] * dsy[b] / 12.0
                            if transverse == 0.0:
                                continue
                            for cf in range(nfz):
                                face = fz_idx[cf]
                                flux = fz_val[cf]
                                if flux == 0.0 or face < 0 or face > nz:
                                    continue
                                Jz_face[i, j, face] += coeff_z * flux * transverse
                    continue

                # TSC/order>1 remains on the legacy midpoint path in Step 1-C.
                pm0 = 0.5 * (pa0 + pb0)
                pm1 = 0.5 * (pa1 + pb1)
                pm2 = 0.5 * (pa2 + pb2)

                dp0 = pb0 - pa0
                dp1 = pb1 - pa1
                dp2 = pb2 - pa2

                if dp0 != 0.0:
                    fx = (pm0 - left_x) / dx
                    sy = (pm1 - left_y) / dy - 0.5
                    sz = (pm2 - left_z) / dz - 0.5

                    na = _shape_weights_numba(fx, nx + 1, order, idx_a, w_a)
                    nb = _shape_weights_numba(sy, ny, order, idx_b, w_b)
                    nc = _shape_weights_numba(sz, nz, order, idx_c, w_c)

                    coeff = charge * dp0 / dt
                    for a in range(na):
                        ia = idx_a[a]
                        wa = w_a[a]
                        for b in range(nb):
                            ib = idx_b[b]
                            wb = w_b[b]
                            for c in range(nc):
                                ic = idx_c[c]
                                wc = w_c[c]
                                Jx_face[ia, ib, ic] += coeff * wa * wb * wc

                if dp1 != 0.0:
                    sx = (pm0 - left_x) / dx - 0.5
                    fy = (pm1 - left_y) / dy
                    sz = (pm2 - left_z) / dz - 0.5

                    na = _shape_weights_numba(sx, nx, order, idx_a, w_a)
                    nb = _shape_weights_numba(fy, ny + 1, order, idx_b, w_b)
                    nc = _shape_weights_numba(sz, nz, order, idx_c, w_c)

                    coeff = charge * dp1 / dt
                    for a in range(na):
                        ia = idx_a[a]
                        wa = w_a[a]
                        for b in range(nb):
                            ib = idx_b[b]
                            wb = w_b[b]
                            for c in range(nc):
                                ic = idx_c[c]
                                wc = w_c[c]
                                Jy_face[ia, ib, ic] += coeff * wa * wb * wc

                if dp2 != 0.0:
                    sx = (pm0 - left_x) / dx - 0.5
                    sy = (pm1 - left_y) / dy - 0.5
                    fz = (pm2 - left_z) / dz

                    na = _shape_weights_numba(sx, nx, order, idx_a, w_a)
                    nb = _shape_weights_numba(sy, ny, order, idx_b, w_b)
                    nc = _shape_weights_numba(fz, nz + 1, order, idx_c, w_c)

                    coeff = charge * dp2 / dt
                    for a in range(na):
                        ia = idx_a[a]
                        wa = w_a[a]
                        for b in range(nb):
                            ib = idx_b[b]
                            wb = w_b[b]
                            for c in range(nc):
                                ic = idx_c[c]
                                wc = w_c[c]
                                Jz_face[ia, ib, ic] += coeff * wa * wb * wc


def install_fast_deposition_for_class(pic_transport_cls: Any) -> bool:
    """
    Install the fast Numba deposition methods on a pic_transport class.

    The order-1 CIC current path is charge-conserving after Step 1-C.
    Order-2/TSC is retained as a legacy path pending a separate continuity derivation.
    Unsupported orders fall back to the original Python/Numpy methods.
    """
    if njit is None:
        warnings.warn(f"[fast_deposition] Numba unavailable; using original deposition. {_IMPORT_ERROR!r}")
        return False

    if getattr(pic_transport_cls, "_fast_deposition_installed", False):
        return True

    if not hasattr(pic_transport_cls, "_deposit_rho_numpy") or not hasattr(
        pic_transport_cls, "_deposit_current_esirkepov_style_numpy"
    ):
        warnings.warn("[fast_deposition] required pic_transport deposition methods not found; skipped.")
        return False

    original_rho = pic_transport_cls._deposit_rho_numpy
    original_current = pic_transport_cls._deposit_current_esirkepov_style_numpy

    def _deposit_rho_fast(self, rho, pos_np, charge, order):
        order_i = int(order)
        if order_i not in (1, 2):
            FAST_DEPOSITION_STATS["rho_calls_fallback"] += 1
            t0 = time.perf_counter()
            try:
                return original_rho(self, rho, pos_np, charge, order)
            finally:
                FAST_DEPOSITION_STATS["rho_time_fallback_s"] += time.perf_counter() - t0

        FAST_DEPOSITION_STATS["rho_calls_fast"] += 1
        if order_i == 1:
            FAST_DEPOSITION_STATS["rho_calls_order1"] += 1
        else:
            FAST_DEPOSITION_STATS["rho_calls_order2"] += 1

        t0 = time.perf_counter()
        try:
            return _deposit_rho_shape_numba(
                rho,
                pos_np,
                float(charge),
                order_i,
                float(self.left_bound_x),
                float(self.left_bound_y),
                float(self.left_bound_z),
                float(self.right_bound_x),
                float(self.right_bound_y),
                float(self.right_bound_z),
                float(self.dx),
                float(self.dy),
                float(self.dz),
                int(self.nx),
                int(self.ny),
                int(self.nz),
            )
        finally:
            FAST_DEPOSITION_STATS["rho_time_fast_s"] += time.perf_counter() - t0

    def _deposit_current_fast(self, Jx_face, Jy_face, Jz_face, prev_np, curr_np, active_np, charge, order):
        order_i = int(order)
        if order_i not in (1, 2):
            FAST_DEPOSITION_STATS["current_calls_fallback"] += 1
            if _verbose_enabled():
                print("[fast_deposition] unsupported particle_order; falling back to original current deposition.")
            t0 = time.perf_counter()
            try:
                return original_current(self, Jx_face, Jy_face, Jz_face, prev_np, curr_np, active_np, charge, order)
            finally:
                FAST_DEPOSITION_STATS["current_time_fallback_s"] += time.perf_counter() - t0

        FAST_DEPOSITION_STATS["current_calls_fast"] += 1
        if order_i == 1:
            FAST_DEPOSITION_STATS["current_calls_order1"] += 1
        else:
            FAST_DEPOSITION_STATS["current_calls_order2"] += 1

        t0 = time.perf_counter()
        try:
            return _deposit_current_shape_numba(
                Jx_face,
                Jy_face,
                Jz_face,
                prev_np,
                curr_np,
                active_np,
                float(charge),
                order_i,
                float(self.dt),
                float(self.left_bound_x),
                float(self.left_bound_y),
                float(self.left_bound_z),
                float(self.dx),
                float(self.dy),
                float(self.dz),
                int(self.nx),
                int(self.ny),
                int(self.nz),
            )
        finally:
            FAST_DEPOSITION_STATS["current_time_fast_s"] += time.perf_counter() - t0

    pic_transport_cls._deposit_rho_numpy_original = original_rho
    pic_transport_cls._deposit_current_esirkepov_style_numpy_original = original_current
    pic_transport_cls._deposit_rho_numpy = _deposit_rho_fast
    pic_transport_cls._deposit_current_esirkepov_style_numpy = _deposit_current_fast
    pic_transport_cls._fast_deposition_installed = True
    pic_transport_cls._fast_cic_deposition_installed = True  # backward compatibility
    pic_transport_cls.get_fast_deposition_stats = staticmethod(get_fast_deposition_stats)
    pic_transport_cls.reset_fast_deposition_stats = staticmethod(reset_fast_deposition_stats)
    FAST_DEPOSITION_STATS["installed"] = True

    if _verbose_enabled():
        print("[fast_deposition] Installed fast Numba CIC/TSC deposition backend.")
        print("[fast_deposition] particle_order 1 and 2 use fast path.")

    return True


# Backward-compatible function name used by your current Particle_transport/main.py.
def install_fast_cic_deposition_for_class(pic_transport_cls: Any) -> bool:
    return install_fast_deposition_for_class(pic_transport_cls)
