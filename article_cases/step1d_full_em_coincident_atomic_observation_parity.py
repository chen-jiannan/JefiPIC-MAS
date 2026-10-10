#!/usr/bin/env python3
"""Full-EM coincident-grid Atomic-vs-Observation parity regression.

Run only after installing the accompanying EMsolver patch. The test uses the
same rho and native staggered-face J histories in both accumulation modes and
requires the resulting E and B fields to agree to floating-point tolerance.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import cupy as cp
except Exception as exc:
    raise RuntimeError(f"GPU/CuPy is required: {exc}")

from EMsolver.solver import EMsolver


def rel_l2(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-300))


def build_histories(steps, nx, ny, nz):
    # Deterministic smooth histories. Continuity is not required for this
    # implementation-parity test; both kernels receive exactly the same data.
    rho = np.empty((steps, nx, ny, nz), dtype=np.float64)
    jx = np.empty((steps, nx + 1, ny, nz), dtype=np.float64)
    jy = np.empty((steps, nx, ny + 1, nz), dtype=np.float64)
    jz = np.empty((steps, nx, ny, nz + 1), dtype=np.float64)

    xc = (np.arange(nx) + 0.5) / nx
    yc = (np.arange(ny) + 0.5) / ny
    zc = (np.arange(nz) + 0.5) / nz
    Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing="ij")

    xf = np.arange(nx + 1) / nx
    Xx, Yx, Zx = np.meshgrid(xf, yc, zc, indexing="ij")
    yf = np.arange(ny + 1) / ny
    Xy, Yy, Zy = np.meshgrid(xc, yf, zc, indexing="ij")
    zf = np.arange(nz + 1) / nz
    Xz, Yz, Zz = np.meshgrid(xc, yc, zf, indexing="ij")

    for n in range(steps):
        ph = 0.37 * n
        rho[n] = 1.0e-9 * (
            np.sin(2 * np.pi * Xc + ph)
            + 0.7 * np.cos(2 * np.pi * Yc - 0.3 * ph)
            + 0.3 * np.sin(2 * np.pi * Zc + 0.2 * ph)
        )
        jx[n] = 2.0e-3 * np.cos(2 * np.pi * Xx + 0.8 * ph) * (1 + 0.1 * np.sin(2 * np.pi * Yx))
        jy[n] = 1.5e-3 * np.sin(2 * np.pi * Yy - 0.5 * ph) * (1 + 0.08 * np.cos(2 * np.pi * Zy))
        jz[n] = 1.2e-3 * np.cos(2 * np.pi * Zz + 0.6 * ph) * (1 + 0.06 * np.sin(2 * np.pi * Xz))

    return rho, (jx, jy, jz)


def make_solver(mode, steps, nx, ny, nz, dt, c, eps0):
    # Deliberately use coincident source/observation cell-center grids.  This
    # exercises the index-based self-cell exclusion together with the full-EM
    # native staggered-current path.
    dx, dy, dz = 6.0e-3, 7.0e-3, 8.0e-3
    s = EMsolver(
        steps,
        nx, ny, nz,
        nx, ny, nz,
        dx, dy, dz, 0.0, 0.0, 0.0,
        dx, dy, dz, 0.0, 0.0, 0.0,
        dt, eps0, c, False, "open",
    )
    s.set_grid_current_layout("staggered_face")
    s.set_kernel_mode(mode)
    return s


def load_and_run(s, rho, faces):
    steps = len(rho)
    s.rho_GPU[:, :] = cp.asarray(rho.reshape(steps, -1))
    s.Jx_face_GPU[:, :] = cp.asarray(faces[0].reshape(steps, -1))
    s.Jy_face_GPU[:, :] = cp.asarray(faces[1].reshape(steps, -1))
    s.Jz_face_GPU[:, :] = cp.asarray(faces[2].reshape(steps, -1))
    s.i_time = steps - 1
    s.t = (steps - 1) * s.dt
    s.time_tick = steps - 1
    s._run_jefimenko_kernel(include_antenna=False)
    cp.cuda.Stream.null.synchronize()
    E = np.stack(
        [cp.asnumpy(s.GEx), cp.asnumpy(s.GEy), cp.asnumpy(s.GEz)], axis=1
    )
    B = np.stack(
        [cp.asnumpy(s.GBx), cp.asnumpy(s.GBy), cp.asnumpy(s.GBz)], axis=1
    )
    return E, B


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tol", type=float, default=5.0e-10)
    p.add_argument("--out", default="step1d_full_em_coincident_atomic_observation_parity")
    a = p.parse_args()

    steps = 7
    nx, ny, nz = 5, 4, 3
    dt = 2.0e-10
    c = 299792458.0
    eps0 = 8.8541878128e-12

    rho, faces = build_histories(steps, nx, ny, nz)
    obs = make_solver("observation", steps, nx, ny, nz, dt, c, eps0)
    ato = make_solver("atomic", steps, nx, ny, nz, dt, c, eps0)

    Eo, Bo = load_and_run(obs, rho, faces)
    Ea, Ba = load_and_run(ato, rho, faces)

    e_rel = rel_l2(Ea, Eo)
    b_rel = rel_l2(Ba, Bo)
    e_max = float(np.max(np.abs(Ea - Eo)))
    b_max = float(np.max(np.abs(Ba - Bo)))
    finite = bool(
        np.all(np.isfinite(Eo)) and np.all(np.isfinite(Bo))
        and np.all(np.isfinite(Ea)) and np.all(np.isfinite(Ba))
    )
    passed = bool(finite and e_rel <= a.tol and b_rel <= a.tol)

    report = {
        "experiment": "FULL_EM_COINCIDENT_GRID_NATIVE_STAGGERED_ATOMIC_VS_OBSERVATION_PARITY",
        "full_em": True,
        "grid_current_layout": "staggered_face",
        "boundary_mode": "open",
        "steps": steps,
        "source_grid": [nx, ny, nz],
        "observation_grid": [nx, ny, nz],
        "coincident_source_observation_grid": True,
        "index_self_exclusion_required": True,
        "tolerance_rel_l2": a.tol,
        "finite": finite,
        "E_atomic_vs_observation_rel_l2": e_rel,
        "B_atomic_vs_observation_rel_l2": b_rel,
        "E_atomic_vs_observation_max_abs": e_max,
        "B_atomic_vs_observation_max_abs": b_max,
        "pass": passed,
        "interpretation": (
            "A pass establishes implementation parity of the two GPU accumulation "
            "strategies for the same full-EM native staggered-face Jefimenko source "
            "histories. It is not a separate physical-model validation."
        ),
    }

    out = Path("article_cases_output") / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "atomic_observation_parity_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("full-EM atomic/observation parity gate failed")


if __name__ == "__main__":
    main()
