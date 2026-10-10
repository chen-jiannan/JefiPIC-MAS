#!/usr/bin/env python3
"""Step 4D-B2A V2 — corrected classification from existing B2A outputs.

No PIC rerun. Fixes the original decision-tree omission that computed
`all_boundary` but did not use it in the final classification.

No new post-hoc seed-CV pass/fail threshold is introduced.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
TINY=1e-300

def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def write_csv(path, rows):
    if not rows: return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def run(args):
    root=Path("article_cases_output")/args.root_out
    src=root/"step4d_b2a_seed_ensemble_report.json"
    if not src.exists(): raise FileNotFoundError(src)
    old=load_json(src)
    if old.get("experiment")!="STEP4D_B2A_LONG_WINDOW_COHERENT_SEED_ENSEMBLE":
        raise RuntimeError("unexpected B2A report")
    seeds=old["seed_results"]
    all_execution=all(
        bool(r.get("two_way_execution_valid"))
        and not bool(r.get("numerical_runaway_detected",False))
        and r.get("plasma_grid_kernel_mode_actual")=="atomic"
        for r in seeds
    )
    all_long=all(bool(r.get("long_window_stable")) for r in seeds)
    all_boundary=all(bool(r.get("boundary_within_predeclared_5pct")) for r in seeds)
    primary=np.asarray([float(r["primary_final_9cycle_ratio"]) for r in seeds],float)
    mean=float(np.mean(primary)); median=float(np.median(primary))
    std=float(np.std(primary,ddof=1)) if len(primary)>1 else 0.0
    cv=std/max(abs(mean),TINY)
    maxmin=float(np.max(primary)/max(np.min(primary),TINY))
    if not all_execution:
        classification="B2A_EXECUTION_FAILURE"
        next_step="Stop. Diagnose the failing seed before any particle-count study."
    elif not all_long:
        classification="B2A_LONG_WINDOW_ESTIMATOR_NOT_STABLE"
        next_step="Stop. Report temporal nonstationarity; do not run particle-count sensitivity."
    elif not all_boundary:
        classification="B2A_LONG_WINDOW_STABLE_BUT_BOUNDARY_NOT_UNIFORM_ACROSS_SEEDS"
        next_step=(
            "Do not claim uniform <5% boundary qualification. Proceed only to "
            "particle-count ROBUSTNESS CHARACTERIZATION to determine whether boundary "
            "pass fraction and seed spread improve with particle count."
        )
    else:
        classification="B2A_LONG_WINDOW_STABLE_AND_BOUNDARY_UNIFORM_ACROSS_SEEDS"
        next_step=(
            "Proceed to particle-count robustness characterization. Report the measured "
            "seed spread quantitatively; do not invent a new post-hoc seed-CV gate."
        )
    rows=[]
    for r in seeds:
        rows.append({
            "seed":r["seed"],
            "two_way_execution_valid":r["two_way_execution_valid"],
            "boundary_within_5pct":r["boundary_within_predeclared_5pct"],
            "worst_region":r["worst_holdout"]["region"],
            "worst_gate":r["worst_holdout"]["gate_value"],
            "primary_final_9cycle_ratio":r["primary_final_9cycle_ratio"],
            "long_window_span_8_9_10":r["long_window_ratio_relative_span_8_9_10"],
            "long_window_stable":r["long_window_stable"],
        })
    report={
        "experiment":"STEP4D_B2A_V2_CORRECTED_CLASSIFICATION",
        "status":"complete",
        "no_PIC_rerun":True,
        "source_report":str(src),
        "logic_fix":(
            "Original B2A computed all_boundary but omitted it from the decision tree. "
            "V2 corrects that omission. No new seed-CV pass/fail threshold is introduced."
        ),
        "execution_all_valid":all_execution,
        "long_window_estimator_all_stable":all_long,
        "boundary_all_seeds_within_predeclared_5pct":all_boundary,
        "boundary_pass_count":int(sum(bool(r["boundary_within_predeclared_5pct"]) for r in seeds)),
        "boundary_seed_count":len(seeds),
        "primary_final_9cycle_ratios":primary.tolist(),
        "ensemble_median_ratio":median,
        "ensemble_mean_ratio":mean,
        "ensemble_sample_std_ratio":std,
        "ensemble_coefficient_of_variation":cv,
        "ensemble_max_to_min_ratio":maxmin,
        "seed_variability_note":(
            "Seed variability is material and must be reported. The original B2A "
            "same-order-of-magnitude criterion (max/min <=10) is preserved historically; "
            "V2 does not replace it after seeing the data."
        ),
        "classification":classification,
        "next_step":next_step,
        "c3_qualification_relabelled_as_pass":False,
    }
    (root/"step4d_b2a_v2_corrected_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    (root/"step4d_b2a_v2_corrected_summary.txt").write_text(
        "\n".join([
            "Step 4D-B2A V2 corrected classification",
            f"classification: {classification}",
            f"execution all valid: {all_execution}",
            f"long-window stable all seeds: {all_long}",
            f"boundary pass fraction: {report['boundary_pass_count']}/{report['boundary_seed_count']}",
            f"9-cycle ratios: {primary.tolist()}",
            f"median: {median:.9g}",
            f"mean: {mean:.9g}",
            f"CV: {cv:.6g}",
            f"max/min: {maxmin:.6g}",
            f"next: {next_step}",
        ])+"\n", encoding="utf-8")
    write_csv(root/"step4d_b2a_v2_seed_summary.csv",rows)
    print(json.dumps(report,indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root-out",default="step4d_b_atomic12_formal")
    a=p.parse_args(); run(a)
if __name__=="__main__": main()
