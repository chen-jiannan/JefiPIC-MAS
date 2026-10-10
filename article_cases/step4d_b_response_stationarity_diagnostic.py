#!/usr/bin/env python3
"""
Post-process an existing full-EM Atomic Step4D-B article-scale run.

Purpose
-------
The Step4D-B numerical-method qualification (finite execution, causal closure,
homogeneous-tail stability, and independent boundary residual) is kept separate
from the auxiliary-current-moment response diagnostic.

This script does NOT rerun PIC and does NOT define a new physical acceptance
gate.  It quantifies how the 3 MHz auxiliary-current-moment ratio depends on
fit-window length and end time, so the manuscript can report window dependence
instead of forcing a single "converged gain" from a nonstationary finite-particle
record.

Expected input layout
---------------------
article_cases_output/<ROOT>/
    step4d_b_vacuum_history.npz
    runs/N24000_seed101/
        step4d_a_full_em_atomic_diagnostics.npz

Outputs
-------
response_stationarity_rolling.csv
response_stationarity_report.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

F_DRIVE = 3.0e6
FREQS = (3.0e6, 6.0e6, 9.0e6)
DT = 1.0e-9
TINY = 1.0e-300


def fit_multitone(times, values, freqs=FREQS):
    """DC + linear trend + joint 3/6/9 MHz least-squares fit."""
    t = np.asarray(times, dtype=np.float64)
    y = np.asarray(values)
    if y.shape[0] != len(t):
        raise ValueError("time/value length mismatch")
    if len(t) < 2 + 2 * len(freqs):
        raise RuntimeError("insufficient samples for multitone fit")

    tmid = float(np.mean(t))
    tscale = max(float(np.ptp(t)), 1.0e-30)
    tn = (t - tmid) / tscale

    cols = [np.ones(len(t)), tn]
    for f in freqs:
        w = 2.0 * np.pi * float(f)
        cols.extend([np.cos(w * t), np.sin(w * t)])
    X = np.column_stack(cols)

    shape = y.shape[1:]
    Y = y.reshape(len(t), -1)
    coef, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)

    phasors = {}
    for i, f in enumerate(freqs):
        a = coef[2 + 2 * i]
        b = coef[3 + 2 * i]
        phasors[float(f)] = (a - 1j * b).reshape(shape)

    return {
        "condition_number": float(np.linalg.cond(X)),
        "rank": int(rank),
        "phasors": phasors,
        "num_samples": int(len(t)),
    }


def fit_window(t, M, end_cycle, window_cycles):
    end_t = float(end_cycle) / F_DRIVE
    start_t = float(end_cycle - window_cycles) / F_DRIVE
    mask = (t >= start_t - 0.5 * DT) & (t <= end_t + 0.5 * DT)
    if np.count_nonzero(mask) < 8:
        raise RuntimeError("too few samples in requested fit window")
    f = fit_multitone(t[mask], M[mask])
    M3 = np.asarray(f["phasors"][3.0e6])
    return {
        "abs_Mz_A_m": float(abs(M3[2])),
        "condition_number": float(f["condition_number"]),
        "num_samples": int(f["num_samples"]),
    }


def qstats(x):
    a = np.asarray(x, dtype=float)
    return {
        "count": int(len(a)),
        "min": float(np.min(a)),
        "q25": float(np.quantile(a, 0.25)),
        "median": float(np.median(a)),
        "q75": float(np.quantile(a, 0.75)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="step4d_b_full_em_atomic_final")
    p.add_argument("--run", default="N24000_seed101")
    p.add_argument("--windows", nargs="+", type=float, default=[1, 2, 3, 4, 6])
    p.add_argument("--late-end-cycles", type=int, default=6,
                   help="Number of latest integer-cycle end points summarized.")
    args = p.parse_args()

    root = Path("article_cases_output") / args.root
    pdiag = root / "runs" / args.run / "step4d_a_full_em_atomic_diagnostics.npz"
    pvac = root / "step4d_b_vacuum_history.npz"
    if not pdiag.exists():
        raise FileNotFoundError(pdiag)
    if not pvac.exists():
        raise FileNotFoundError(pvac)

    z = np.load(pdiag)
    zv = np.load(pvac)
    t = np.asarray(z["time_s"], dtype=np.float64)
    M = np.asarray(z["current_moment_A_m"], dtype=np.float64)
    tv = np.asarray(zv["time_s"], dtype=np.float64)
    Mv = np.asarray(zv["current_moment_A_m"], dtype=np.float64)

    total_cycles = int(np.floor(float(t[-1]) * F_DRIVE + 1.0e-9))
    rows = []

    for wc in sorted(set(float(x) for x in args.windows)):
        first_end = int(np.ceil(wc))
        for ec in range(first_end, total_cycles + 1):
            fp = fit_window(t, M, ec, wc)
            fv = fit_window(tv, Mv, ec, wc)
            ratio = fp["abs_Mz_A_m"] / max(fv["abs_Mz_A_m"], TINY)
            rows.append({
                "end_cycle": int(ec),
                "window_cycles": float(wc),
                "plasma_abs_Mz_A_m": fp["abs_Mz_A_m"],
                "vacuum_abs_Mz_A_m": fv["abs_Mz_A_m"],
                "plasma_to_vacuum_ratio": float(ratio),
                "plasma_fit_condition_number": fp["condition_number"],
                "vacuum_fit_condition_number": fv["condition_number"],
                "num_samples": fp["num_samples"],
            })

    out_csv = root / "response_stationarity_rolling.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    late0 = max(1, total_cycles - int(args.late_end_cycles) + 1)
    by_window = {}
    for wc in sorted(set(float(x) for x in args.windows)):
        vals = [
            r["plasma_to_vacuum_ratio"] for r in rows
            if abs(r["window_cycles"] - wc) < 1.0e-12
            and r["end_cycle"] >= max(late0, int(np.ceil(wc)))
        ]
        by_window[f"{wc:g}cycle"] = qstats(vals) if vals else None

    final_windows = {
        f"{r['window_cycles']:g}cycle": r
        for r in rows if r["end_cycle"] == total_cycles
    }

    report = {
        "experiment": "STEP4D_B_RESPONSE_STATIONARITY_DIAGNOSTIC",
        "simulation_rerun": False,
        "root": str(root),
        "run": args.run,
        "total_cycles": total_cycles,
        "observable": "axial auxiliary electric-current moment Mz",
        "observable_is_not_port_current": True,
        "observable_is_not_impedance_or_gain": True,
        "purpose": (
            "Quantify finite-time/window dependence of the auxiliary response "
            "without using it as a numerical-method acceptance gate."
        ),
        "final_window_values": final_windows,
        "late_cycle_ratio_statistics_by_window": by_window,
        "interpretation_rule": (
            "If ratios vary strongly with fit window/end time while the underlying "
            "two-way execution and boundary metrics remain valid, report the response "
            "as window/realization dependent and do not promote one ratio to a "
            "converged physical scalar."
        ),
    }
    out_json = root / "response_stationarity_report.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
