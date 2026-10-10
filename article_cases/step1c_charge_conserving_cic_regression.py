#!/usr/bin/env python3
"""
Step 1-C: regression suite for the charge-conserving CIC current deposition.

This test validates the revised first-order (CIC) face-current deposition only.
It does NOT validate the current face-to-cell reconstruction used by the current
Jefimenko field kernel; that quantity is intentionally reported as a separate
non-gating diagnostic for Step 1-D.

Gating criteria
---------------
1. Discrete face continuity:
       (rho^{n+1}-rho^n)/dt + div_f(J_face) = 0
   with relative L_inf <= 1e-12.
2. Current moment:
       integral J dV = q (r^{n+1}-r^n)/dt
   with relative vector error <= 1e-12 for non-stationary paths.
3. Python-reference and Numba-fast deposition arrays agree to <= 1e-12.

The suite contains:
- deterministic A-H regression trajectories from Step 1-B;
- a 1-D fractional-position/displacement sweep;
- random 3-D interior trajectory stress tests;
- face-to-cell continuity residuals as an explicitly non-gating diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

# Make direct execution from article_cases/ robust without requiring PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

try:
    import cupy as cp  # noqa: F401  (required by Particle_transport.main)
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This benchmark imports Particle_transport.main and therefore needs the "
        "same CuPy environment used by JefiPIC. Run it in your normal project environment."
    ) from exc

from Particle_transport.main import pic_transport


PASS_REL = 1.0e-12
TINY = 1.0e-300


def _make_transport_stub(nx, ny, nz, dx, dy, dz, dt):
    obj = pic_transport.__new__(pic_transport)
    obj.dt = float(dt)
    obj.nx = int(nx)
    obj.ny = int(ny)
    obj.nz = int(nz)
    obj.dx = float(dx)
    obj.dy = float(dy)
    obj.dz = float(dz)
    obj.left_bound_x = 0.0
    obj.left_bound_y = 0.0
    obj.left_bound_z = 0.0
    obj.right_bound_x = nx * dx
    obj.right_bound_y = ny * dy
    obj.right_bound_z = nz * dz
    obj.boundary_mode = "open"
    obj.cell_volume = dx * dy * dz
    obj.inv_cell_volume = 1.0 / obj.cell_volume
    return obj


def _divergence_faces(Jx, Jy, Jz, dx, dy, dz):
    return (
        (Jx[1:, :, :] - Jx[:-1, :, :]) / dx
        + (Jy[:, 1:, :] - Jy[:, :-1, :]) / dy
        + (Jz[:, :, 1:] - Jz[:, :, :-1]) / dz
    )


def _divergence_reconstructed_cell(obj, Jx, Jy, Jz):
    """Current Step-1 reconstruction diagnostic; NOT a Step-1C pass criterion."""
    Jxc, Jyc, Jzc = obj._face_to_center_currents(Jx, Jy, Jz, 1)
    Jxf = np.zeros((obj.nx + 1, obj.ny, obj.nz), dtype=np.float64)
    Jyf = np.zeros((obj.nx, obj.ny + 1, obj.nz), dtype=np.float64)
    Jzf = np.zeros((obj.nx, obj.ny, obj.nz + 1), dtype=np.float64)

    if obj.nx > 1:
        Jxf[1:obj.nx] = 0.5 * (Jxc[:-1] + Jxc[1:])
    Jxf[0] = Jxc[0]
    Jxf[obj.nx] = Jxc[-1]

    if obj.ny > 1:
        Jyf[:, 1:obj.ny] = 0.5 * (Jyc[:, :-1] + Jyc[:, 1:])
    Jyf[:, 0] = Jyc[:, 0]
    Jyf[:, obj.ny] = Jyc[:, -1]

    if obj.nz > 1:
        Jzf[:, :, 1:obj.nz] = 0.5 * (Jzc[:, :, :-1] + Jzc[:, :, 1:])
    Jzf[:, :, 0] = Jzc[:, :, 0]
    Jzf[:, :, obj.nz] = Jzc[:, :, -1]

    return _divergence_faces(Jxf, Jyf, Jzf, obj.dx, obj.dy, obj.dz)


def _continuity_metrics(residual, drho_dt, div_j):
    linf_abs = float(np.max(np.abs(residual)))
    rms_abs = float(np.sqrt(np.mean(residual * residual)))
    drho_linf = float(np.max(np.abs(drho_dt)))
    div_linf = float(np.max(np.abs(div_j)))
    drho_rms = float(np.sqrt(np.mean(drho_dt * drho_dt)))
    div_rms = float(np.sqrt(np.mean(div_j * div_j)))
    return {
        "linf_abs": linf_abs,
        "rms_abs": rms_abs,
        "linf_rel": linf_abs / max(drho_linf + div_linf, TINY),
        "rms_rel": rms_abs / max(drho_rms + div_rms, TINY),
        "drho_dt_linf": drho_linf,
        "div_j_linf": div_linf,
    }


def _array_compare(a, b):
    diff = np.asarray(a) - np.asarray(b)
    peak = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), TINY)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "rms_abs": float(np.sqrt(np.mean(diff * diff))),
        "max_rel_to_peak": float(np.max(np.abs(diff))) / peak,
    }


def _get_backends(obj):
    cls = type(obj)
    rho_py = getattr(cls, "_deposit_rho_numpy_original", cls._deposit_rho_numpy)
    current_py = getattr(
        cls,
        "_deposit_current_esirkepov_style_numpy_original",
        cls._deposit_current_esirkepov_style_numpy,
    )
    fast_installed = bool(getattr(cls, "_fast_deposition_installed", False))
    return {
        "python": (rho_py, current_py),
        "fast": (cls._deposit_rho_numpy, cls._deposit_current_esirkepov_style_numpy)
        if fast_installed
        else None,
        "fast_installed": fast_installed,
    }


def _deposit_one(obj, backend_pair, p0, p1, charge=1.0):
    rho_fn, current_fn = backend_pair
    rho0 = np.zeros((obj.nx, obj.ny, obj.nz), dtype=np.float64)
    rho1 = np.zeros_like(rho0)
    Jx = np.zeros((obj.nx + 1, obj.ny, obj.nz), dtype=np.float64)
    Jy = np.zeros((obj.nx, obj.ny + 1, obj.nz), dtype=np.float64)
    Jz = np.zeros((obj.nx, obj.ny, obj.nz + 1), dtype=np.float64)

    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p0a = p0.reshape(1, 3)
    p1a = p1.reshape(1, 3)
    active = np.ones(1, dtype=np.bool_)

    rho_fn(obj, rho0, p0a, float(charge), 1)
    rho_fn(obj, rho1, p1a, float(charge), 1)
    current_fn(obj, Jx, Jy, Jz, p0a, p1a, active, float(charge), 1)

    # Match evaluate_rho_J(): convert the stored charge/current moments to densities.
    inv_dv = 1.0 / obj.cell_volume
    rho0 *= inv_dv
    rho1 *= inv_dv
    Jx *= inv_dv
    Jy *= inv_dv
    Jz *= inv_dv

    drho_dt = (rho1 - rho0) / obj.dt
    div_face = _divergence_faces(Jx, Jy, Jz, obj.dx, obj.dy, obj.dz)
    face = _continuity_metrics(drho_dt + div_face, drho_dt, div_face)

    div_cell = _divergence_reconstructed_cell(obj, Jx, Jy, Jz)
    cell = _continuity_metrics(drho_dt + div_cell, drho_dt, div_cell)

    # Integral J dV.  For the face samples used here, sum(J_face)*cell_volume
    # is the natural discrete current moment and should equal q*Delta r/dt.
    moment = np.array(
        [
            float(np.sum(Jx) * obj.cell_volume),
            float(np.sum(Jy) * obj.cell_volume),
            float(np.sum(Jz) * obj.cell_volume),
        ],
        dtype=np.float64,
    )
    target = float(charge) * (p1 - p0) / obj.dt
    diff = moment - target
    target_norm = float(np.linalg.norm(target))
    moment_abs = float(np.linalg.norm(diff))
    moment_rel = moment_abs / max(target_norm, TINY) if target_norm > 0.0 else moment_abs

    return {
        "rho0": rho0,
        "rho1": rho1,
        "Jx": Jx,
        "Jy": Jy,
        "Jz": Jz,
        "face": face,
        "cell": cell,
        "moment": moment,
        "moment_target": target,
        "moment_abs_error": moment_abs,
        "moment_rel_error": moment_rel,
        "charge0": float(np.sum(rho0) * obj.cell_volume),
        "charge1": float(np.sum(rho1) * obj.cell_volume),
    }


def _backend_parity(py, fast):
    parts = {
        "rho0": _array_compare(fast["rho0"], py["rho0"]),
        "rho1": _array_compare(fast["rho1"], py["rho1"]),
        "Jx": _array_compare(fast["Jx"], py["Jx"]),
        "Jy": _array_compare(fast["Jy"], py["Jy"]),
        "Jz": _array_compare(fast["Jz"], py["Jz"]),
    }
    overall = max(v["max_rel_to_peak"] for v in parts.values())
    return parts, overall


def _deterministic_paths():
    return [
        ("A_stationary", (3.20, 3.30, 3.40), (3.20, 3.30, 3.40)),
        ("B_x_subcell", (3.20, 3.30, 3.40), (3.35, 3.30, 3.40)),
        ("C_x_cross_grid_face", (3.90, 3.30, 3.40), (4.10, 3.30, 3.40)),
        ("D_x_cross_shape_knot", (3.45, 3.30, 3.40), (3.55, 3.30, 3.40)),
        ("E_xy_diagonal", (3.20, 3.20, 3.40), (3.35, 3.37, 3.40)),
        ("F_xyz_diagonal", (3.20, 3.20, 3.20), (3.35, 3.37, 3.39)),
        ("G_xyz_multicell", (2.20, 2.30, 2.40), (4.70, 3.80, 5.10)),
        ("H_reverse_xyz", (3.35, 3.37, 3.39), (3.20, 3.20, 3.20)),
    ]


def _record_case(obj, backends, name, p0, p1):
    py = _deposit_one(obj, backends["python"], p0, p1)
    rec = {
        "name": name,
        "p0": list(map(float, p0)),
        "p1": list(map(float, p1)),
        "python_face_linf_rel": py["face"]["linf_rel"],
        "python_face_rms_rel": py["face"]["rms_rel"],
        "python_cell_linf_rel": py["cell"]["linf_rel"],
        "python_moment_rel_error": py["moment_rel_error"],
        "python_moment_abs_error": py["moment_abs_error"],
        "charge0": py["charge0"],
        "charge1": py["charge1"],
    }
    if backends["fast"] is not None:
        fast = _deposit_one(obj, backends["fast"], p0, p1)
        _, parity = _backend_parity(py, fast)
        rec.update(
            {
                "fast_face_linf_rel": fast["face"]["linf_rel"],
                "fast_moment_rel_error": fast["moment_rel_error"],
                "fast_vs_python_max_rel": parity,
            }
        )
    else:
        rec.update(
            {
                "fast_face_linf_rel": None,
                "fast_moment_rel_error": None,
                "fast_vs_python_max_rel": None,
            }
        )
    return rec


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _finite_max(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return max(vals) if vals else None


def run(out_dir: Path, random_count: int, seed: int):
    # Larger grid than Step 1-B so the random/multi-cell trajectories stay far
    # from physical boundaries; boundary bookkeeping is intentionally not part
    # of this Step 1-C interior-deposition test.
    obj = _make_transport_stub(16, 16, 16, 1.0, 1.0, 1.0, 0.25)
    backends = _get_backends(obj)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warm up Numba before timed/stress loops and surface any JIT typing error early.
    if backends["fast"] is not None:
        _deposit_one(obj, backends["fast"], (6.2, 6.3, 6.4), (6.35, 6.37, 6.39))

    deterministic = []
    for name, p0, p1 in _deterministic_paths():
        deterministic.append(_record_case(obj, backends, name, p0, p1))

    # 1-D fractional-position sweep: deliberately includes non-special midpoint
    # positions and both motion directions.
    fractional = []
    displacements = (0.01, 0.10, 0.40, 0.90, 1.70, -0.01, -0.10, -0.40, -0.90, -1.70)
    for frac_i in range(1, 100):
        frac = frac_i / 100.0
        x0 = 7.0 + frac
        for disp in displacements:
            p0 = (x0, 7.23, 7.41)
            p1 = (x0 + disp, 7.23, 7.41)
            fractional.append(
                _record_case(obj, backends, f"frac_{frac:.2f}_dx_{disp:+.2f}", p0, p1)
            )

    # Random 3-D interior trajectories, including multi-cell and mixed-sign motion.
    rng = np.random.default_rng(seed)
    random_rows = []
    for i in range(int(random_count)):
        p0 = rng.uniform(3.0, 13.0, size=3)
        disp = rng.uniform(-2.4, 2.4, size=3)
        p1 = np.clip(p0 + disp, 2.0, 14.0)
        if np.linalg.norm(p1 - p0) < 1.0e-10:
            p1[0] += 0.05
        random_rows.append(_record_case(obj, backends, f"random_{i:04d}", p0, p1))

    all_rows = deterministic + fractional + random_rows
    nonstationary = [r for r in all_rows if r["name"] != "A_stationary"]

    py_face_max = _finite_max(r["python_face_linf_rel"] for r in nonstationary)
    py_moment_max = _finite_max(r["python_moment_rel_error"] for r in nonstationary)
    cell_max = _finite_max(r["python_cell_linf_rel"] for r in nonstationary)
    fast_face_max = _finite_max(r["fast_face_linf_rel"] for r in nonstationary)
    fast_moment_max = _finite_max(r["fast_moment_rel_error"] for r in nonstationary)
    parity_max = _finite_max(r["fast_vs_python_max_rel"] for r in all_rows)

    det_face_pass = all(
        r["python_face_linf_rel"] <= PASS_REL
        for r in deterministic
        if r["name"] != "A_stationary"
    )
    frac_face_pass = all(r["python_face_linf_rel"] <= PASS_REL for r in fractional)
    random_face_pass = all(r["python_face_linf_rel"] <= PASS_REL for r in random_rows)
    moment_pass = all(r["python_moment_rel_error"] <= PASS_REL for r in nonstationary)
    parity_pass = parity_max is None or parity_max <= PASS_REL
    fast_face_pass = fast_face_max is None or fast_face_max <= PASS_REL
    fast_moment_pass = fast_moment_max is None or fast_moment_max <= PASS_REL

    overall = (
        det_face_pass
        and frac_face_pass
        and random_face_pass
        and moment_pass
        and parity_pass
        and fast_face_pass
        and fast_moment_pass
    )

    report = {
        "benchmark": "Step 1-C charge-conserving CIC deposition regression",
        "particle_order": 1,
        "pass_threshold_rel": PASS_REL,
        "random_count": int(random_count),
        "seed": int(seed),
        "fast_installed": backends["fast_installed"],
        "JEFIPIC_FAST_DEPOSITION": os.environ.get("JEFIPIC_FAST_DEPOSITION", "<unset; default enabled>"),
        "summary": {
            "overall_step1c_pass": overall,
            "deterministic_face_pass": det_face_pass,
            "fractional_sweep_face_pass": frac_face_pass,
            "random_stress_face_pass": random_face_pass,
            "current_moment_pass": moment_pass,
            "fast_face_pass": fast_face_pass,
            "fast_current_moment_pass": fast_moment_pass,
            "fast_vs_python_pass": parity_pass,
            "python_worst_face_linf_rel": py_face_max,
            "python_worst_current_moment_rel": py_moment_max,
            "fast_worst_face_linf_rel": fast_face_max,
            "fast_worst_current_moment_rel": fast_moment_max,
            "fast_vs_python_worst_rel": parity_max,
            "face_to_cell_worst_linf_rel_non_gating": cell_max,
            "note": (
                "Step 1-C passes only the CIC face-source deposition. The face-to-cell residual is intentionally "
                "non-gating and is expected to remain large until Step 1-D changes the Jefimenko source representation."
            ),
        },
        "deterministic": deterministic,
    }

    (out_dir / "step1c_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(out_dir / "step1c_deterministic.csv", deterministic)
    _write_csv(out_dir / "step1c_fractional_sweep.csv", fractional)
    _write_csv(out_dir / "step1c_random_stress.csv", random_rows)

    lines = [
        "Step 1-C charge-conserving CIC deposition regression",
        f"PASS threshold = {PASS_REL:.1e}; random_count={random_count}; seed={seed}",
        f"fast Numba backend installed: {backends['fast_installed']}",
        "",
        "GATING RESULTS",
        f"  deterministic A-H face continuity : {'PASS' if det_face_pass else 'FAIL'}",
        f"  fractional-position sweep         : {'PASS' if frac_face_pass else 'FAIL'}",
        f"  random 3-D stress                 : {'PASS' if random_face_pass else 'FAIL'}",
        f"  physical current moment           : {'PASS' if moment_pass else 'FAIL'}",
        f"  fast face continuity              : {'PASS' if fast_face_pass else 'FAIL'}",
        f"  fast current moment               : {'PASS' if fast_moment_pass else 'FAIL'}",
        f"  fast/Python array parity          : {'PASS' if parity_pass else 'FAIL'}",
        "",
        f"  worst Python face L_inf relative residual : {py_face_max:.6e}",
        f"  worst Python current-moment relative error: {py_moment_max:.6e}",
    ]
    if fast_face_max is not None:
        lines.extend(
            [
                f"  worst fast face L_inf relative residual   : {fast_face_max:.6e}",
                f"  worst fast current-moment relative error  : {fast_moment_max:.6e}",
                f"  worst fast/Python relative array mismatch : {parity_max:.6e}",
            ]
        )
    lines.extend(
        [
            "",
            "NON-GATING STEP 1-D DIAGNOSTIC",
            f"  worst face->cell continuity L_inf relative residual: {cell_max:.6e}",
            "  (This is expected to remain large; do NOT use it to fail Step 1-C.)",
            "",
            f"OVERALL STEP 1-C: {'PASS' if overall else 'FAIL'}",
            "",
            "Deterministic A-H:",
            f"{'test':26s} {'face rel':>12s} {'moment rel':>12s} {'cell rel*':>12s} {'fast/Py':>12s}",
            "-" * 79,
        ]
    )
    for r in deterministic:
        fp = "n/a" if r["fast_vs_python_max_rel"] is None else f"{r['fast_vs_python_max_rel']:.3e}"
        lines.append(
            f"{r['name']:26s} {r['python_face_linf_rel']:12.3e} "
            f"{r['python_moment_rel_error']:12.3e} {r['python_cell_linf_rel']:12.3e} {fp:>12s}"
        )
    lines.append("\n* cell rel is diagnostic only; it belongs to Step 1-D.")

    text = "\n".join(lines) + "\n"
    (out_dir / "step1c_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote results to: {out_dir}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="article_cases_output/step1c_charge_conserving_cic",
        help="output directory",
    )
    parser.add_argument("--random-count", type=int, default=1000, help="number of random 3-D stress trajectories")
    parser.add_argument("--seed", type=int, default=20260921, help="random stress seed")
    args = parser.parse_args()
    run(Path(args.out), args.random_count, args.seed)
    # Diagnostic scripts always return zero so failed tests still leave complete output.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
