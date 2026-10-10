#!/usr/bin/env python3
"""Single-frequency causal Retarded-MOT characterization.

This script reuses the Step4C2-V4 geometry, continuous retarded harmonic
reference, independent holdouts, and comparison metrics at one user-selected
frequency. Frequencies outside the original 3/6/9 MHz spatial-qualification
set are characterization points; they do not retroactively modify that set.
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
for q in (_ROOT, _THIS.parent):
    if str(q) not in sys.path:
        sys.path.insert(0, str(q))

import step4c2_v4_retarded_frequency_regression as c2

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--frequency-mhz",type=float,required=True)
    p.add_argument("--quick",action="store_true")
    p.add_argument("--gmres-rtol",type=float,default=1e-9)
    p.add_argument("--gmres-maxiter",type=int,default=300)
    p.add_argument("--out",default=None)
    return p.parse_args()

def main():
    a=parse_args()
    if not np.isfinite(a.frequency_mhz) or a.frequency_mhz<=0:
        raise ValueError("--frequency-mhz must be positive")
    s=c2.settings(a.quick)
    outname=a.out or f"step4c2_{a.frequency_mhz:g}MHz_characterization"
    out=Path("article_cases_output")/outname
    out.mkdir(parents=True,exist_ok=True)
    f=float(a.frequency_mhz)*1e6
    g=c2.geometry(s)
    t0=time.perf_counter()
    Hp,Hk,An,delay=c2.assemble_mot(g,s["obs_chunk"])
    ass=time.perf_counter()-t0
    if delay["max_delay_over_dt"]>=1:
        raise RuntimeError("one-step Retarded-MOT invalid: max delay/dt >= 1")
    prep=c2.prepare_real(An,g.solve_weights); del An; gc.collect()
    holds=c2.make_holdouts(a.quick)
    result=c2.run_freq(f,g,Hp,Hk,prep,holds,a.quick,s["obs_chunk"],a.gmres_rtol,a.gmres_maxiter)
    report={
        "experiment":"STEP4C2_SINGLE_FREQUENCY_CHARACTERIZATION",
        "mode":s["mode"],
        "frequency_Hz":f,
        "frequency_MHz":float(a.frequency_mhz),
        "original_spatial_qualification_frequencies_MHz":[3.0,6.0,9.0],
        "this_frequency_is_in_original_spatial_qualification_set":
            bool(any(abs(f-x)<0.5 for x in c2.FORMAL_FREQS)),
        "interpretation":(
            "A frequency outside 3/6/9 MHz is a supplemental temporal/harmonic "
            "characterization. It does not change the original spatial qualification."
        ),
        "settings":{
            "total_length_m":s["total_length_m"],
            "sources_per_arm":s["sources_per_arm"],
            "depth_frac":c2.DEPTH_FRAC,
            "lambda_rel":c2.LAMBDA_REL,
            "dt_s":c2.DT,
        },
        "one_step_delay":delay,
        "operator_assembly_seconds":ass,
        "result":result,
    }
    (out/"single_frequency_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    lines=[
        "Single-frequency causal Retarded-MOT characterization",
        f"frequency: {a.frequency_mhz:g} MHz",
        f"candidate collocation residual: {result['mot_candidate']['continuous_physics_collocation_rel_l2']:.9g}",
        f"candidate worst holdout: {result['mot_candidate']['worst_holdout_gate']:.9g}",
        f"worst amplitude relative error: {result['worst_field_amplitude_relative_error']:.9g}",
        f"worst phase error deg: {result['worst_field_phase_error_deg']:.9g}",
        f"candidate-reference global difference: {result['worst_candidate_minus_reference_global_norm']:.9g}",
        f"passed comparison checks: {result['passed']}",
    ]
    (out/"single_frequency_summary.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))

if __name__=="__main__":
    main()
