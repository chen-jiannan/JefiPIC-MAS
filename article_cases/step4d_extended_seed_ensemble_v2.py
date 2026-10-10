#!/usr/bin/env python3
"""Extend the 24k-particle article-scale seed ensemble without changing physics.

Place in JefiPIC_MAS/article_cases/. This wrapper reuses the exact 12-cycle
Step-4D case runner and the final-8/9/10-cycle post-processing already used by
the audited manuscript. Existing seeds are reused from disk; only missing seeds
are simulated.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))

import numpy as np

import step4d_b_article_scale_sensitivity as b
import step4d_b2a_long_window_seed_ensemble as b2a


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--step4d-a-report", default=None,
                   help="Optional baseline report. If omitted or invalid, search existing successful runs under --root-out.")
    p.add_argument("--root-out", default="step4d_b_atomic12_formal")
    p.add_argument("--seeds", type=int, nargs="+", required=True,
                   help="Complete seed list to characterize, e.g. 101 202 303 404 505 606 707 808")
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--accept-known-c3-limitation", action="store_true")
    return p.parse_args()


def bootstrap_median_ci(x, nboot=20000, seed=24680):
    x = np.asarray(x, float)
    if len(x) < 5:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(nboot, len(x)))
    med = np.median(x[idx], axis=1)
    return [float(np.quantile(med, 0.025)), float(np.quantile(med, 0.975))]


def _select_valid_precondition(args):
    candidates = []
    if args.step4d_a_report:
        candidates.append(Path(args.step4d_a_report))

    root = Path("article_cases_output") / args.root_out
    # Prefer an already successful 24k run because it is the exact ensemble
    # configuration being extended; fall back to any completed Step4D-B run.
    candidates.extend(sorted(root.glob("runs/N24000_seed*/step4d_a_report.json")))
    candidates.extend(sorted(root.glob("runs/N*_seed*/step4d_a_report.json")))

    seen = set()
    failures = []
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if not p.exists():
            failures.append([str(p), "file not found"])
            continue
        try:
            report = json.loads(p.read_text(encoding="utf-8"))
            pre = b.validate_step4d_a_precondition(
                report, bool(args.accept_known_c3_limitation)
            )
            return report, pre, str(p), failures
        except Exception as exc:
            failures.append([str(p), str(exc)])

    detail = "\n".join(f"  - {p}: {msg}" for p, msg in failures)
    raise RuntimeError(
        "No valid quasi-electrostatic atomic Step4D-A precondition report was found.\n"
        "The added seeds have not been started. Checked:\n" + detail
    )


def run(args):
    report, pre, pre_path, pre_failures = _select_valid_precondition(args)

    rows = []
    for seed in args.seeds:
        case = b.run_case(
            args.root_out, b2a.PARTICLES, int(seed), pre,
            force_rerun=bool(args.force_rerun),
        )
        sr = b2a.analyze_seed(args.root_out, int(seed), case)
        rows.append({
            "seed": int(seed),
            "two_way_execution_valid": bool(sr["two_way_execution_valid"]),
            "boundary_within_predeclared_5pct": bool(sr["boundary_within_predeclared_5pct"]),
            "worst_holdout_region": sr["worst_holdout"]["region"],
            "worst_holdout_gate": float(sr["worst_holdout"]["gate_value"]),
            "ratio_final_9_cycles": float(sr["primary_final_9cycle_ratio"]),
            "relative_span_8_9_10_cycles": float(sr["long_window_ratio_relative_span_8_9_10"]),
            "long_window_stable_by_predeclared_rule": bool(sr["long_window_stable"]),
        })

    ratios = np.asarray([r["ratio_final_9_cycles"] for r in rows], float)
    summary = {
        "experiment": "STEP4D_EXTENDED_24000_PARTICLE_SEED_ENSEMBLE",
        "physics_parameters_changed": False,
        "precondition_report_used": pre_path,
        "rejected_precondition_candidates": pre_failures,
        "particles_per_species": int(b2a.PARTICLES),
        "cycles": 12.0,
        "primary_estimator": "final 9 RF cycles",
        "burn_in_sensitivity": "final 8 and 10 RF cycles",
        "seeds": [int(r["seed"]) for r in rows],
        "n_seeds": int(len(rows)),
        "ratios": ratios.tolist(),
        "median": float(np.median(ratios)),
        "mean": float(np.mean(ratios)),
        "min": float(np.min(ratios)),
        "max": float(np.max(ratios)),
        "sample_std": float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0,
        "coefficient_of_variation": float(np.std(ratios, ddof=1) / max(abs(np.mean(ratios)), 1e-300)) if len(ratios) > 1 else 0.0,
        "empirical_q25": float(np.quantile(ratios, 0.25)),
        "empirical_q75": float(np.quantile(ratios, 0.75)),
        "bootstrap_95pct_CI_for_median_descriptive_only": bootstrap_median_ci(ratios),
        "all_execution_valid": bool(all(r["two_way_execution_valid"] for r in rows)),
        "global_boundary_pass_count": int(sum(r["boundary_within_predeclared_5pct"] for r in rows)),
        "long_window_rule_pass_count": int(sum(r["long_window_stable_by_predeclared_rule"] for r in rows)),
        "interpretation_note": (
            "Use this ensemble to describe finite-particle realization variability. "
            "Do not reinterpret the auxiliary-current-moment ratio as a calibrated physical gain."
        ),
    }

    out = Path("article_cases_output") / args.root_out
    out.mkdir(parents=True, exist_ok=True)
    (out / "step4d_extended_seed_ensemble_report.json").write_text(
        json.dumps({"summary": summary, "cases": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (out / "step4d_extended_seed_ensemble.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run(parse_args())
