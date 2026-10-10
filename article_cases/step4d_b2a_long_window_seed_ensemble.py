#!/usr/bin/env python3
"""
Step 4D-B2A — long-window coherent-response seed ensemble.

Why this stage exists
---------------------
The atomic 12-cycle B1 run is numerically valid and boundary-valid, but the
predeclared final 1/2/3-cycle "steady phasor" readiness test fails badly.
Inspection shows that this is not solver runaway: the nonlinear PIC response
contains strong stochastic/broadband/slow content, so one-to-three-cycle fits
are too short for a stable coherent 3 MHz estimator.

This stage does NOT erase that failed B1 criterion. Instead it asks a new,
controlled question:

    Is the long-window coherent 3 MHz response reproducible across independent
    PIC seeds at the article particle count?

No MAS/PIC physics parameter changes are made.

Estimator frozen here
---------------------
- total duration: 12 RF cycles
- representative particle count: 24000/species
- seeds: 101, 202, 303
- actual plasma kernel: atomic
- primary coherent estimator: final 9 RF cycles
- burn-in sensitivity: final 8 and 10 RF cycles
- within-seed readiness: relative span across 8/9/10-cycle ratios <= 20%
- temporal characterization: non-overlapping 3-cycle blocks 1-3, 4-6, 7-9, 10-12
- across-seed response is REPORTED, not tuned to any target value.
- an order-of-magnitude consistency flag uses max/min <= 10 only because the
  independent warm-fluid/kinetic anchors were explicitly treated as scale
  anchors rather than exact predictions.

The primary reviewer-facing quantity from this stage is the median of the
final-9-cycle coherent ratios across seeds, together with the seed range and
within-seed long-window sensitivity.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))

import step4d_b_article_scale_sensitivity as b

PARTICLES = 24000
SEEDS = (101, 202, 303)
LONG_WINDOWS = (8.0, 9.0, 10.0)
PRIMARY_WINDOW = 9.0
WITHIN_SEED_SPAN_GATE = 0.20
ORDER_OF_MAGNITUDE_GATE = 10.0
BLOCKS = ((1, 3), (4, 6), (7, 9), (10, 12))
TINY = b.TINY


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fit_range(times, current_moment, polarization_moment, start_cycle, end_cycle):
    t = np.asarray(times, dtype=np.float64)
    M = np.asarray(current_moment, dtype=np.float64)
    D = np.asarray(polarization_moment, dtype=np.float64)
    t0 = (float(start_cycle) - 1.0) / b.F_DRIVE
    t1 = float(end_cycle) / b.F_DRIVE
    mask = (t >= t0 - 0.5*b.DT) & (t <= t1 + 0.5*b.DT)
    if int(np.count_nonzero(mask)) < 16:
        raise RuntimeError("too few samples in requested block")
    fm = b.d4a.v2a.fit_multitone(t[mask], M[mask])
    fd = b.d4a.v2a.fit_multitone(t[mask], D[mask])
    Mh = np.asarray(fm["phasors"][3.0e6], np.complex128).reshape(3)
    Dh = np.asarray(fd["phasors"][3.0e6], np.complex128).reshape(3)
    return {
        "start_cycle": int(start_cycle),
        "end_cycle": int(end_cycle),
        "num_samples": int(np.count_nonzero(mask)),
        "plasma_abs_Mz_A_m": float(abs(Mh[2])),
        "plasma_continuity_abs_iwD_over_M": float(
            abs(1j*2*np.pi*3.0e6*Dh[2]) / max(abs(Mh[2]), TINY)
        ),
        "fit_condition_number": float(max(fm["condition_number"], fd["condition_number"])),
    }


def load_case_history(root_out, seed):
    cdir = b.case_dir(root_out, PARTICLES, seed)
    p = cdir / "step4d_a_diagnostics.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    z = np.load(p)
    return {
        "time_s": np.asarray(z["time_s"], dtype=np.float64),
        "current_moment_A_m": np.asarray(z["current_moment_A_m"], dtype=np.float64),
        "polarization_moment_C_m": np.asarray(z["polarization_moment_C_m"], dtype=np.float64),
    }


def analyze_seed(root_out, seed, case_report):
    hist = load_case_history(root_out, seed)
    vacuum = b._vacuum_history()

    windows = []
    for fc in LONG_WINDOWS:
        pf = b.fit_final_window(
            hist["time_s"], hist["current_moment_A_m"],
            hist["polarization_moment_C_m"], fc
        )
        vf = b.fit_final_window(
            vacuum["time_s"], vacuum["current_moment_A_m"],
            vacuum["polarization_moment_C_m"], fc
        )
        p3 = float(pf["frequencies"]["3MHz"]["axial_current_moment_A_m"])
        v3 = float(vf["frequencies"]["3MHz"]["axial_current_moment_A_m"])
        ratio = p3 / max(v3, TINY)
        windows.append({
            "seed": int(seed),
            "fit_cycles": float(fc),
            "plasma_abs_Mz_A_m": p3,
            "vacuum_abs_Mz_A_m": v3,
            "plasma_to_vacuum_ratio": float(ratio),
            "plasma_continuity_abs_iwD_over_M": float(
                pf["frequencies"]["3MHz"]["continuity_harmonic_ratio_abs_iwD_over_M"]
            ),
            "vacuum_continuity_abs_iwD_over_M": float(
                vf["frequencies"]["3MHz"]["continuity_harmonic_ratio_abs_iwD_over_M"]
            ),
            "plasma_fit_condition_number": float(pf["current_fit_condition_number"]),
            "vacuum_fit_condition_number": float(vf["current_fit_condition_number"]),
        })

    ratios = np.asarray([r["plasma_to_vacuum_ratio"] for r in windows], dtype=float)
    median = float(np.median(ratios))
    rel_span = float((np.max(ratios)-np.min(ratios))/max(abs(median), TINY))
    primary = next(r for r in windows if abs(r["fit_cycles"]-PRIMARY_WINDOW) < 1e-12)

    blocks = []
    # Matched vacuum block fits use the same exact cycle ranges.
    for a, c in BLOCKS:
        pb = fit_range(
            hist["time_s"], hist["current_moment_A_m"],
            hist["polarization_moment_C_m"], a, c
        )
        vb = fit_range(
            vacuum["time_s"], vacuum["current_moment_A_m"],
            vacuum["polarization_moment_C_m"], a, c
        )
        blocks.append({
            "seed": int(seed),
            "start_cycle": a,
            "end_cycle": c,
            "plasma_abs_Mz_A_m": pb["plasma_abs_Mz_A_m"],
            "vacuum_abs_Mz_A_m": vb["plasma_abs_Mz_A_m"],
            "plasma_to_vacuum_ratio": float(
                pb["plasma_abs_Mz_A_m"] / max(vb["plasma_abs_Mz_A_m"], TINY)
            ),
            "plasma_continuity_abs_iwD_over_M": pb["plasma_continuity_abs_iwD_over_M"],
            "fit_condition_number": max(
                pb["fit_condition_number"], vb["fit_condition_number"]
            ),
        })

    return {
        "seed": int(seed),
        "two_way_execution_valid": bool(case_report["two_way_execution_valid"]),
        "boundary_within_predeclared_5pct": bool(case_report["boundary_within_predeclared_5pct"]),
        "worst_holdout": case_report["worst_holdout"],
        "numerical_runaway_detected": bool(case_report.get("numerical_runaway_detected", False)),
        "plasma_grid_kernel_mode_actual": case_report.get("plasma_grid_kernel_mode_actual"),
        "long_windows": windows,
        "primary_final_9cycle_ratio": float(primary["plasma_to_vacuum_ratio"]),
        "long_window_ratio_relative_span_8_9_10": rel_span,
        "long_window_span_gate": WITHIN_SEED_SPAN_GATE,
        "long_window_stable": bool(
            np.all(np.isfinite(ratios))
            and rel_span <= WITHIN_SEED_SPAN_GATE
            and max(
                max(r["plasma_fit_condition_number"], r["vacuum_fit_condition_number"])
                for r in windows
            ) < b.FIT_CONDITION_LIMIT
        ),
        "three_cycle_blocks": blocks,
    }


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def self_test():
    # Reuse B's synthetic fit test and check ensemble summary arithmetic.
    vals = np.asarray([8.0, 9.0, 10.0])
    median = float(np.median(vals))
    rel_span = float((np.max(vals)-np.min(vals))/median)
    pass_math = abs(median-9.0) < 1e-15 and abs(rel_span-2.0/9.0) < 1e-15
    report = {
        "self_test_pass": bool(pass_math),
        "self_test_requires_gpu": False,
        "primary_window_cycles": PRIMARY_WINDOW,
        "long_windows_cycles": list(LONG_WINDOWS),
        "seeds": list(SEEDS),
        "particles_per_species": PARTICLES,
    }
    print(json.dumps(report, indent=2))
    if not report["self_test_pass"]:
        raise RuntimeError("Step 4D-B2A self-test failed")


def run(args):
    d4a_report = load_json(args.step4d_a_report)
    pre = b.validate_step4d_a_precondition(
        d4a_report, bool(args.accept_known_c3_limitation)
    )
    if abs(b.FORMAL_CYCLES - 12.0) > 1e-12:
        raise RuntimeError("B2A requires the 12-cycle Step 4D-B script")
    if tuple(b.FORMAL_FIT_CYCLES) != (1.0, 2.0, 3.0):
        raise RuntimeError("unexpected base Step 4D-B fit-window configuration")

    root = b.root_dir(args.root_out)
    root.mkdir(parents=True, exist_ok=True)

    seed_reports = []
    all_window_rows = []
    all_block_rows = []

    for seed in SEEDS:
        case = b.run_case(
            args.root_out, PARTICLES, seed, pre,
            force_rerun=bool(args.force_rerun)
        )
        if case.get("plasma_grid_kernel_mode_actual") != "atomic":
            raise RuntimeError(f"seed {seed} did not use atomic plasma kernel")
        sr = analyze_seed(args.root_out, seed, case)
        seed_reports.append(sr)
        all_window_rows.extend(sr["long_windows"])
        all_block_rows.extend(sr["three_cycle_blocks"])

    primary = np.asarray(
        [r["primary_final_9cycle_ratio"] for r in seed_reports], dtype=float
    )
    seed_median = float(np.median(primary))
    seed_mean = float(np.mean(primary))
    seed_min = float(np.min(primary))
    seed_max = float(np.max(primary))
    seed_cv = float(np.std(primary, ddof=1)/max(abs(seed_mean), TINY)) if len(primary) > 1 else 0.0
    max_min = float(seed_max/max(seed_min, TINY))

    all_execution = all(r["two_way_execution_valid"] and not r["numerical_runaway_detected"] for r in seed_reports)
    all_boundary = all(r["boundary_within_predeclared_5pct"] for r in seed_reports)
    all_long_stable = all(r["long_window_stable"] for r in seed_reports)
    order_consistent = bool(max_min <= ORDER_OF_MAGNITUDE_GATE)

    if not all_execution:
        classification = "SEED_ENSEMBLE_EXECUTION_FAILURE"
        next_step = "Stop. Diagnose the failing seed before any particle-count sensitivity."
    elif not all_long_stable:
        classification = "SEED_ENSEMBLE_TEMPORALLY_NONSTATIONARY"
        next_step = (
            "Do not force a single steady-state ratio. Report the coherent-response "
            "distribution and temporal block variability; decide manuscript scope before "
            "spending the lower-particle matrix."
        )
    elif not order_consistent:
        classification = "SEED_ENSEMBLE_REALIZATION_SENSITIVE_OVER_ONE_DECADE"
        next_step = (
            "Do not run the lower-particle matrix yet. The corrected high-particle response "
            "is too seed-sensitive for a single representative magnitude claim."
        )
    else:
        classification = "SEED_ENSEMBLE_READY_FOR_PARTICLE_COUNT_SENSITIVITY"
        next_step = (
            "Proceed to the 6000/12000 particle-count sensitivity using the same 12-cycle "
            "atomic formulation and final-9-cycle primary coherent estimator. Keep 8/10-cycle "
            "fits as burn-in sensitivity diagnostics."
        )

    report = {
        "experiment": "STEP4D_B2A_LONG_WINDOW_COHERENT_SEED_ENSEMBLE",
        "status": "complete",
        "reason_for_revised_estimator": (
            "The predeclared final 1/2/3-cycle steady-phasor readiness assumption was "
            "falsified by the valid 12-cycle atomic B1 run. Short windows are dominated "
            "by stochastic/broadband/slow PIC content and one-cycle trend/fundamental "
            "coupling. No solver physics parameter is changed."
        ),
        "frozen": {
            "cycles": 12.0,
            "particles_per_species": PARTICLES,
            "seeds": list(SEEDS),
            "kernel": "atomic",
            "primary_window_cycles": PRIMARY_WINDOW,
            "burnin_sensitivity_windows_cycles": list(LONG_WINDOWS),
            "within_seed_relative_span_gate": WITHIN_SEED_SPAN_GATE,
            "order_of_magnitude_consistency_max_min_gate": ORDER_OF_MAGNITUDE_GATE,
        },
        "step4d_a_precondition": pre,
        "seed_results": seed_reports,
        "ensemble": {
            "primary_final_9cycle_ratios": primary.tolist(),
            "median_ratio": seed_median,
            "mean_ratio": seed_mean,
            "min_ratio": seed_min,
            "max_ratio": seed_max,
            "sample_std_ratio": float(np.std(primary, ddof=1)) if len(primary)>1 else 0.0,
            "coefficient_of_variation": seed_cv,
            "max_to_min_ratio": max_min,
            "all_two_way_execution_valid": all_execution,
            "all_boundary_within_predeclared_5pct": all_boundary,
            "all_within_seed_long_window_stable": all_long_stable,
            "order_of_magnitude_consistent": order_consistent,
        },
        "classification": classification,
        "next_step": next_step,
        "old_1_2_3_cycle_b1_failure_retained": True,
        "c3_qualification_relabelled_as_pass": False,
    }

    (root / "step4d_b2a_seed_ensemble_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary = [
        "Step 4D-B2A long-window coherent-response seed ensemble",
        f"classification: {classification}",
        f"primary window: final {PRIMARY_WINDOW:g} cycles",
        f"seed primary ratios: {primary.tolist()}",
        f"ensemble median: {seed_median:.9g}",
        f"ensemble range: {seed_min:.9g} .. {seed_max:.9g}",
        f"max/min: {max_min:.9g}",
        f"all execution valid: {all_execution}",
        f"all boundary <5%: {all_boundary}",
        f"all within-seed 8/9/10-cycle spans <=20%: {all_long_stable}",
        f"next: {next_step}",
    ]
    (root / "step4d_b2a_seed_ensemble_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    flat_seed = []
    for r in seed_reports:
        flat_seed.append({
            "seed": r["seed"],
            "two_way_execution_valid": r["two_way_execution_valid"],
            "boundary_within_predeclared_5pct": r["boundary_within_predeclared_5pct"],
            "worst_holdout_region": r["worst_holdout"]["region"],
            "worst_holdout_gate": r["worst_holdout"]["gate_value"],
            "primary_final_9cycle_ratio": r["primary_final_9cycle_ratio"],
            "long_window_ratio_relative_span_8_9_10": r["long_window_ratio_relative_span_8_9_10"],
            "long_window_stable": r["long_window_stable"],
        })
    write_csv(root / "step4d_b2a_seed_summary.csv", flat_seed)
    write_csv(root / "step4d_b2a_long_window_rows.csv", all_window_rows)
    write_csv(root / "step4d_b2a_three_cycle_blocks.csv", all_block_rows)

    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--step4d-a-report", default=None)
    p.add_argument("--root-out", default="step4d_b_atomic12_formal")
    p.add_argument("--accept-known-c3-limitation", action="store_true")
    p.add_argument("--force-rerun", action="store_true")
    a = p.parse_args()
    if a.self_test:
        self_test()
        return
    if a.step4d_a_report is None:
        raise RuntimeError("--step4d-a-report is required")
    run(a)


if __name__ == "__main__":
    main()
