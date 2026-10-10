#!/usr/bin/env python3
"""Step 2-A: continuity-consistent MAS electric-dipole source benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from EMsolver.mas_dipole_basis import (
    ContinuityConsistentElectricDipoleBasis,
    electric_point_dipole_fields,
    finite_segment_jefimenko_fields,
    relative_l2,
)

EPS0 = 8.8541878128e-12
C0 = 299792458.0


def _out_dir(text: str) -> Path:
    p = Path(text)
    if not p.is_absolute() and (not p.parts or p.parts[0] != "article_cases_output"):
        p = Path("article_cases_output") / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def _harmonic_callbacks(axis, p_amp, omega):
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis /= np.linalg.norm(axis)

    def p_scalar(t):
        return p_amp * np.sin(omega * np.asarray(t, dtype=np.float64))

    def pd_scalar(t):
        return p_amp * omega * np.cos(omega * np.asarray(t, dtype=np.float64))

    def pdd_scalar(t):
        return -p_amp * omega * omega * np.sin(omega * np.asarray(t, dtype=np.float64))

    def p_vec(t):
        return p_scalar(t)[..., None] * axis

    def pd_vec(t):
        return pd_scalar(t)[..., None] * axis

    def pdd_vec(t):
        return pdd_scalar(t)[..., None] * axis

    return p_scalar, pd_scalar, pdd_scalar, p_vec, pd_vec, pdd_vec


def weak_continuity_test(rng, count=5000):
    pdot = rng.normal(size=(count, 3)) * 1.0e-6
    grad = rng.normal(size=(count, 3))
    res = ContinuityConsistentElectricDipoleBasis.weak_continuity_residual(pdot, grad)
    scale = np.maximum(np.abs(np.einsum("ij,ij->i", pdot, grad)), 1.0e-300)
    return {
        "samples": int(count),
        "abs_max": float(np.max(np.abs(res))),
        "relative_max": float(np.max(np.abs(res) / scale)),
    }


def temporal_refinement_test(axis, p_amp, frequency):
    omega = 2.0 * math.pi * frequency
    _, _, _, p_vec, pd_vec, _ = _harmonic_callbacks(axis, p_amp, omega)
    rows = []
    previous_error = None
    for spp in (20, 40, 80, 160):
        dt = 1.0 / (frequency * spp)
        steps = 2 * spp
        p0 = p_vec(np.array([0.0]))[0][None, :]
        basis = ContinuityConsistentElectricDipoleBasis(
            np.zeros((1, 3)), dt, steps + 4, initial_dipole_moment=p0
        )
        basis.record_initial_current_moment(pd_vec(np.array([0.0]))[0][None, :])
        max_identity = 0.0
        max_p_error = 0.0
        p_scale = p_amp
        for n in range(1, steps + 1):
            t = n * dt
            m = pd_vec(np.array([t]))[0][None, :]
            p_num = basis.advance_from_current_moment(n, m)
            audit = basis.temporal_continuity_audit(n)
            max_identity = max(max_identity, audit.increment_residual_max)
            p_exact = p_vec(np.array([t]))[0]
            max_p_error = max(max_p_error, float(np.linalg.norm(p_num[0] - p_exact) / max(p_scale, 1e-300)))
        order = None
        if previous_error is not None and max_p_error > 0.0:
            order = math.log(previous_error / max_p_error, 2.0)
        rows.append({
            "steps_per_period": spp,
            "dt_s": dt,
            "max_discrete_continuity_abs_A_m": max_identity,
            "max_dipole_moment_relative_error": max_p_error,
            "observed_order": order,
        })
        previous_error = max_p_error
    return rows


def field_convergence_test(axis, p_amp, frequency):
    omega = 2.0 * math.pi * frequency
    p_scalar, pd_scalar, pdd_scalar, p_vec, pd_vec, pdd_vec = _harmonic_callbacks(axis, p_amp, omega)
    center = np.array([0.013, -0.021, 0.008], dtype=np.float64)
    obs = np.array([
        [0.21, 0.07, 0.05],
        [-0.16, 0.24, 0.11],
        [0.08, -0.19, 0.27],
        [0.32, 0.18, -0.14],
        [-0.27, -0.12, 0.20],
    ], dtype=np.float64)
    t = 0.37 / frequency
    E_point, B_point = electric_point_dipole_fields(
        obs, center, t, p_vec, pd_vec, pdd_vec, epsilon0=EPS0, c=C0
    )
    min_R = float(np.min(np.linalg.norm(obs - center[None, :], axis=1)))
    rows = []
    for ratio in (0.20, 0.10, 0.05, 0.025):
        ell = ratio * min_R
        E_seg, B_seg = finite_segment_jefimenko_fields(
            obs, center, axis, ell, t, p_scalar, pd_scalar, pdd_scalar,
            epsilon0=EPS0, c=C0, quadrature_order=128,
        )
        rows.append({
            "length_over_min_R": ratio,
            "length_m": ell,
            "E_relative_L2": relative_l2(E_seg, E_point),
            "B_relative_L2": relative_l2(B_seg, B_point),
        })
    for key in ("E_relative_L2", "B_relative_L2"):
        for i in range(1, len(rows)):
            e0, e1 = rows[i-1][key], rows[i][key]
            rows[i][key.replace("relative_L2", "observed_order")] = (
                math.log(e0 / e1, 2.0) if e0 > 0.0 and e1 > 0.0 else None
            )
        rows[0][key.replace("relative_L2", "observed_order")] = None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="step2a_mas_dipole_source_closure")
    ap.add_argument("--frequency", type=float, default=30.0e6)
    ap.add_argument("--p-amp", type=float, default=2.0e-12, help="electric-dipole moment amplitude [C m]")
    ap.add_argument("--seed", type=int, default=24017)
    args = ap.parse_args()

    out = _out_dir(args.out)
    rng = np.random.default_rng(args.seed)
    axis = np.array([0.31, -0.47, 0.826], dtype=np.float64)
    axis /= np.linalg.norm(axis)

    weak = weak_continuity_test(rng)
    temporal = temporal_refinement_test(axis, args.p_amp, args.frequency)
    field = field_convergence_test(axis, args.p_amp, args.frequency)

    temporal_identity_max = max(r["max_discrete_continuity_abs_A_m"] for r in temporal)
    fine_p_error = temporal[-1]["max_dipole_moment_relative_error"]
    temporal_orders = [r["observed_order"] for r in temporal[2:] if r["observed_order"] is not None]
    temporal_order_min = min(temporal_orders) if temporal_orders else float("nan")
    fine_E = field[-1]["E_relative_L2"]
    fine_B = field[-1]["B_relative_L2"]
    field_E_monotone = all(field[i]["E_relative_L2"] < field[i-1]["E_relative_L2"] for i in range(1, len(field)))
    field_B_monotone = all(field[i]["B_relative_L2"] < field[i-1]["B_relative_L2"] for i in range(1, len(field)))

    gates = {
        "distributional_continuity": weak["abs_max"] <= 1.0e-18,
        "discrete_temporal_identity": temporal_identity_max <= 1.0e-16,
        "temporal_second_order": temporal_order_min >= 1.8,
        "temporal_fine_error": fine_p_error <= 5.0e-4,
        "no_fit_field_convergence_E": field_E_monotone and fine_E <= 2.0e-3,
        "no_fit_field_convergence_B": field_B_monotone and fine_B <= 2.0e-3,
    }
    overall = all(gates.values())

    with (out / "step2a_temporal_refinement.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(temporal[0].keys()))
        w.writeheader(); w.writerows(temporal)
    with (out / "step2a_field_convergence.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(field[0].keys()))
        w.writeheader(); w.writerows(field)

    report = {
        "basis_definition": {
            "rho": "-p(t) dot grad delta(r-r0)",
            "J": "dp/dt delta(r-r0)",
            "unknown_for_future_step2b": "electric current moment m=dp/dt [A m] with 3 Cartesian DOF per MAS point",
            "production_MAS_G_changed_in_step2a": False,
        },
        "frequency_Hz": args.frequency,
        "p_amp_C_m": args.p_amp,
        "axis": axis.tolist(),
        "weak_continuity": weak,
        "temporal_refinement": temporal,
        "finite_segment_to_point_dipole": field,
        "gates": gates,
        "overall_pass": overall,
    }
    (out / "step2a_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = []
    lines.append("Step 2-A continuity-consistent MAS electric-dipole source audit")
    lines.append("================================================================")
    lines.append("")
    lines.append("SOURCE DEFINITION")
    lines.append("  rho_MAS = -p(t) . grad delta(r-r0)")
    lines.append("  J_MAS   =  dp/dt delta(r-r0)")
    lines.append("  future Step-2B unknown: current moment m=dp/dt [A m], 3 Cartesian DOF/source")
    lines.append("  production G modified here: NO")
    lines.append("")
    lines.append("DISTRIBUTIONAL CONTINUITY")
    lines.append(f"  weak-form absolute max residual : {weak['abs_max']:.6e}")
    lines.append(f"  weak-form relative max residual : {weak['relative_max']:.6e}")
    lines.append("")
    lines.append("TEMPORAL REFINEMENT (trapezoidal p update from m=dp/dt)")
    for r in temporal:
        order = "-" if r["observed_order"] is None else f"{r['observed_order']:.3f}"
        lines.append(
            f"  spp={r['steps_per_period']:3d}  continuity={r['max_discrete_continuity_abs_A_m']:.3e}  "
            f"p-error={r['max_dipole_moment_relative_error']:.3e}  order={order}"
        )
    lines.append("")
    lines.append("NO-FIT FIELD CHECK: finite continuous segment -> ideal electric point dipole")
    for r in field:
        lines.append(
            f"  ell/Rmin={r['length_over_min_R']:.3f}  E-L2={r['E_relative_L2']:.3e}  "
            f"B-L2={r['B_relative_L2']:.3e}"
        )
    lines.append("")
    lines.append("GATING")
    for k, v in gates.items():
        lines.append(f"  {k:34s}: {'PASS' if v else 'FAIL'}")
    lines.append("")
    lines.append(f"OVERALL STEP 2-A: {'PASS' if overall else 'FAIL'}")
    (out / "step2a_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
