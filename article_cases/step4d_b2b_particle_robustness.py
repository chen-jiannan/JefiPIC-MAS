#!/usr/bin/env python3
"""Step 4D-B2B — particle-count robustness characterization.

Uses the same 12-cycle atomic formulation and final-9-cycle coherent estimator.
Runs the missing 6000/12000 x seeds 101/202/303 cases and reuses all completed
24000 cases. Reports trends; does not create a new qualification gate.
"""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
_THIS=Path(__file__).resolve()
_ROOT=_THIS.parent.parent
if str(_ROOT) not in sys.path: sys.path.insert(0,str(_ROOT))
if str(_THIS.parent) not in sys.path: sys.path.insert(0,str(_THIS.parent))
import step4d_b_article_scale_sensitivity as b

PARTICLES=(6000,12000,24000)
SEEDS=(101,202,303)
WINDOWS=(8.0,9.0,10.0)
PRIMARY=9.0
WINDOW_SPAN_GATE=0.20
TINY=b.TINY

def write_csv(path,rows):
    if not rows:return
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)

def analyze_case(root_out,n,seed,case):
    p=b.case_dir(root_out,n,seed)/"step4d_a_diagnostics.npz"
    if not p.exists(): raise FileNotFoundError(p)
    z=np.load(p)
    t=np.asarray(z["time_s"],float)
    M=np.asarray(z["current_moment_A_m"],float)
    D=np.asarray(z["polarization_moment_C_m"],float)
    vac=b._vacuum_history()
    rows=[]
    for fc in WINDOWS:
        pf=b.fit_final_window(t,M,D,fc)
        vf=b.fit_final_window(vac["time_s"],vac["current_moment_A_m"],vac["polarization_moment_C_m"],fc)
        p3=float(pf["frequencies"]["3MHz"]["axial_current_moment_A_m"])
        v3=float(vf["frequencies"]["3MHz"]["axial_current_moment_A_m"])
        rows.append({
            "particles_per_species":int(n),"seed":int(seed),"fit_cycles":float(fc),
            "plasma_abs_Mz_A_m":p3,"vacuum_abs_Mz_A_m":v3,
            "plasma_to_vacuum_ratio":float(p3/max(v3,TINY)),
            "plasma_continuity_abs_iwD_over_M":float(
                pf["frequencies"]["3MHz"]["continuity_harmonic_ratio_abs_iwD_over_M"]),
            "fit_condition_number":float(max(pf["current_fit_condition_number"],vf["current_fit_condition_number"])),
        })
    vals=np.asarray([r["plasma_to_vacuum_ratio"] for r in rows],float)
    med=float(np.median(vals))
    span=float((np.max(vals)-np.min(vals))/max(abs(med),TINY))
    primary=next(r for r in rows if abs(r["fit_cycles"]-PRIMARY)<1e-12)
    return {
        "particles_per_species":int(n),"seed":int(seed),
        "two_way_execution_valid":bool(case["two_way_execution_valid"]),
        "numerical_runaway_detected":bool(case.get("numerical_runaway_detected",False)),
        "kernel":case.get("plasma_grid_kernel_mode_actual"),
        "boundary_within_5pct":bool(case["boundary_within_predeclared_5pct"]),
        "worst_region":case["worst_holdout"]["region"],
        "worst_gate":float(case["worst_holdout"]["gate_value"]),
        "primary_final_9cycle_ratio":float(primary["plasma_to_vacuum_ratio"]),
        "long_window_span_8_9_10":span,
        "long_window_stable":bool(
            span<=WINDOW_SPAN_GATE and
            max(r["fit_condition_number"] for r in rows)<b.FIT_CONDITION_LIMIT
        ),
        "windows":rows,
    }

def self_test():
    vals=np.asarray([7.,8.,9.])
    ok=abs(float(np.median(vals))-8.)<1e-15
    rep={"self_test_pass":bool(ok),"self_test_requires_gpu":False,
         "particles":list(PARTICLES),"seeds":list(SEEDS),"primary_window_cycles":PRIMARY}
    print(json.dumps(rep,indent=2))
    if not ok: raise RuntimeError("B2B self-test failed")

def run(args):
    d4a=json.loads(Path(args.step4d_a_report).read_text(encoding="utf-8"))
    pre=b.validate_step4d_a_precondition(d4a,bool(args.accept_known_c3_limitation))
    if abs(b.FORMAL_CYCLES-12.0)>1e-12: raise RuntimeError("B2B requires 12-cycle base B")
    root=b.root_dir(args.root_out);root.mkdir(parents=True,exist_ok=True)
    cases=[];window_rows=[]
    for n in PARTICLES:
        for seed in SEEDS:
            case=b.run_case(args.root_out,n,seed,pre,force_rerun=bool(args.force_rerun))
            if case.get("plasma_grid_kernel_mode_actual")!="atomic":
                raise RuntimeError(f"N={n} seed={seed} did not use atomic")
            a=analyze_case(args.root_out,n,seed,case)
            cases.append(a);window_rows.extend(a["windows"])
    summaries=[]
    for n in PARTICLES:
        rr=[x for x in cases if x["particles_per_species"]==n]
        vals=np.asarray([x["primary_final_9cycle_ratio"] for x in rr],float)
        mean=float(np.mean(vals)); med=float(np.median(vals)); std=float(np.std(vals,ddof=1))
        summaries.append({
            "particles_per_species":n,
            "primary_ratio_median":med,"primary_ratio_mean":mean,
            "primary_ratio_min":float(np.min(vals)),"primary_ratio_max":float(np.max(vals)),
            "primary_ratio_sample_std":std,
            "primary_ratio_cv":float(std/max(abs(mean),TINY)),
            "primary_ratio_max_to_min":float(np.max(vals)/max(np.min(vals),TINY)),
            "boundary_pass_count":int(sum(x["boundary_within_5pct"] for x in rr)),
            "boundary_seed_count":len(rr),
            "max_worst_boundary_gate":float(max(x["worst_gate"] for x in rr)),
            "long_window_stable_count":int(sum(x["long_window_stable"] for x in rr)),
            "execution_valid_count":int(sum(x["two_way_execution_valid"] and not x["numerical_runaway_detected"] for x in rr)),
        })
    meds={x["particles_per_species"]:x["primary_ratio_median"] for x in summaries}
    convergence={
        "median_relative_change_6000_to_12000":float(abs(meds[12000]-meds[6000])/max(abs(meds[12000]),TINY)),
        "median_relative_change_12000_to_24000":float(abs(meds[24000]-meds[12000])/max(abs(meds[24000]),TINY)),
    }
    report={
        "experiment":"STEP4D_B2B_PARTICLE_COUNT_ROBUSTNESS_CHARACTERIZATION",
        "status":"complete",
        "purpose":"Characterize particle-count convergence, seed spread, long-window stability, and boundary pass fraction; not a new qualification gate.",
        "frozen":{"cycles":12.0,"kernel":"atomic","particles":list(PARTICLES),"seeds":list(SEEDS),
                  "primary_window_cycles":PRIMARY,"window_sensitivity_cycles":list(WINDOWS)},
        "step4d_a_precondition":pre,
        "case_results":[{k:v for k,v in x.items() if k!="windows"} for x in cases],
        "particle_summaries":summaries,
        "convergence":convergence,
        "all_executions_valid":all(x["two_way_execution_valid"] and not x["numerical_runaway_detected"] for x in cases),
        "historical_uniform_5pct_claim_not_made":True,
        "interpretation_rule":(
            "Use the 24000-particle three-seed median/range as the highest-sampling response characterization. "
            "Use 6000/12000 only to show whether median and seed spread trend toward that distribution. "
            "Do not tune parameters to force monotonicity."
        ),
        "next_step":(
            "After inspecting convergence trends, freeze the numerical evidence and update the manuscript/reviewer response. "
            "If medians/spreads do not show a clear convergence trend, report sampling/realization dependence rather than adding more sweeps."
        ),
    }
    (root/"step4d_b2b_particle_robustness_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    (root/"step4d_b2b_particle_robustness_summary.txt").write_text(
        "\n".join([
            "Step 4D-B2B particle-count robustness characterization",
            *[f"N={s['particles_per_species']}: median={s['primary_ratio_median']:.6g}, range={s['primary_ratio_min']:.6g}..{s['primary_ratio_max']:.6g}, CV={s['primary_ratio_cv']:.4g}, boundary={s['boundary_pass_count']}/{s['boundary_seed_count']}, long-window={s['long_window_stable_count']}/{s['boundary_seed_count']}" for s in summaries],
            f"relative median change 6000->12000: {convergence['median_relative_change_6000_to_12000']:.6g}",
            f"relative median change 12000->24000: {convergence['median_relative_change_12000_to_24000']:.6g}",
            report["next_step"],
        ])+"\n",encoding="utf-8")
    write_csv(root/"step4d_b2b_cases.csv",[{k:v for k,v in x.items() if k!="windows"} for x in cases])
    write_csv(root/"step4d_b2b_particle_summaries.csv",summaries)
    write_csv(root/"step4d_b2b_window_rows.csv",window_rows)
    print(json.dumps(report,indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--step4d-a-report",default=None)
    p.add_argument("--root-out",default="step4d_b_atomic12_formal")
    p.add_argument("--accept-known-c3-limitation",action="store_true")
    p.add_argument("--force-rerun",action="store_true")
    a=p.parse_args()
    if a.self_test:self_test();return
    if a.step4d_a_report is None:raise RuntimeError("--step4d-a-report is required")
    run(a)
if __name__=="__main__":main()
