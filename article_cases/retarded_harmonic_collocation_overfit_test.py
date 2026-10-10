#!/usr/bin/env python3
"""Controlled collocation-vs-holdout diagnostic using the final retarded harmonic MAS operator."""
from __future__ import annotations
import argparse, csv, gc, json, math, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

_THIS=Path(__file__).resolve()
_ROOT=_THIS.parent.parent
for q in (_ROOT,_THIS.parent):
    if str(q) not in sys.path:
        sys.path.insert(0,str(q))

import step4c2_v4_retarded_frequency_regression as c2
import step2mb1_incident_normalization_feed_localization as mb1

TINY=1e-300

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--formal-geometry",action="store_true")
    p.add_argument("--sources-per-arm",type=int,default=900)
    p.add_argument("--fit-fractions",nargs="+",type=float,default=[1.0,0.5,0.25])
    p.add_argument("--frequency-mhz",type=float,default=3.0)
    p.add_argument("--lambda-rel",type=float,default=1e-4)
    p.add_argument("--out",default="retarded_harmonic_collocation_overfit")
    return p.parse_args()

def subset_geometry(g, idx):
    idx=np.asarray(idx,dtype=np.int64)
    d=dict(vars(g))
    for name in ["solve_obs_points","solve_normals","solve_tangent1","solve_tangent2","solve_weights"]:
        d[name]=np.asarray(getattr(g,name))[idx]
    return SimpleNamespace(**d)

def choose_training_indices(n, frac):
    m=max(1,min(n,int(round(float(frac)*n))))
    if m==n: return np.arange(n,dtype=np.int64)
    # Deterministic interleaved subset; this changes only the fit set.
    return np.unique(np.linspace(0,n-1,m,dtype=np.int64))

def tangential_rel_rms(total, inc, normals):
    return c2.trms(total,normals)/max(c2.trms(inc,normals),TINY)

def independent_holdout_metrics(g,q,omega,holds):
    rows=[]
    for name,P,N,masks in holds:
        inc=mb1.inc_feed_localized(P,2e-3,1.0).astype(np.complex128)
        scat=c2.eval_retarded_field(P,g,q,omega)
        mm=c2.metrics(inc+scat,inc,N,masks)
        worst=c2.worst_gate(mm)
        rows.append({
            "pattern":name,
            "global_rel":float(mm["global"]["local_rel"]),
            "worst_region":worst["region"],
            "worst_metric":worst["metric"],
            "worst_gate_value":float(worst["value"]),
        })
    return rows

def run(a):
    if any(x<=0 or x>1 for x in a.fit_fractions): raise ValueError("fit fractions must be in (0,1]")
    s=c2.settings(not a.formal_geometry); s=dict(s)
    s["sources_per_arm"]=int(a.sources_per_arm)
    s["collocation_points_per_arm"]=2*int(a.sources_per_arm)
    s["obs_chunk"]=min(int(s.get("obs_chunk",8)),16)
    g=c2.geometry(s); holds=c2.make_holdouts(not a.formal_geometry)
    old_lambda=c2.LAMBDA_REL; c2.LAMBDA_REL=float(a.lambda_rel)
    try:
        f=float(a.frequency_mhz)*1e6; omega=2*np.pi*f
        nobs=len(g.solve_obs_points); all_idx=np.arange(nobs,dtype=np.int64)
        results=[]
        for frac in a.fit_fractions:
            train=choose_training_indices(nobs,frac)
            mask=np.ones(nobs,dtype=bool); mask[train]=False; unused=all_idx[mask]
            gt=subset_geometry(g,train)
            A,diag=c2.assemble_retarded_harmonic(gt,omega,int(s["obs_chunk"]))
            Etrain=mb1.inc_feed_localized(gt.solve_obs_points,2e-3,1.0).astype(np.complex128)
            etrain=c2.incident_coeff(Etrain,gt)
            prep=c2.prepare_complex(A,gt.solve_weights)
            q=c2.solve_complex(prep,-etrain)
            fit=float(np.linalg.norm(A@q+etrain)/max(np.linalg.norm(etrain),TINY))
            unused_rel=None
            if len(unused):
                P=np.asarray(g.solve_obs_points)[unused]
                N=np.asarray(g.solve_normals)[unused]
                I=mb1.inc_feed_localized(P,2e-3,1.0).astype(np.complex128)
                S=c2.eval_retarded_field(P,g,q,omega)
                unused_rel=tangential_rel_rms(I+S,I,N)
            hrows=independent_holdout_metrics(g,q,omega,holds)
            worst=max(r["worst_gate_value"] for r in hrows)
            worst_global=max(r["global_rel"] for r in hrows)
            results.append({
                "frequency_Hz":f,
                "sources_per_arm":int(a.sources_per_arm),
                "total_sources":int(len(g.ant_src_points)),
                "unknowns":int(2*len(g.ant_src_points)),
                "available_collocation_points":int(nobs),
                "fit_fraction":float(frac),
                "fit_points":int(len(train)),
                "fit_equations":int(2*len(train)),
                "lambda_rel":float(a.lambda_rel),
                "collocation_fit_rel_l2":fit,
                "unused_collocation_rel_l2":unused_rel,
                "independent_holdout_global_rel_max":worst_global,
                "independent_holdout_worst_gate":worst,
                "holdout_to_fit_ratio":float(worst/max(fit,TINY)),
                "max_pair_phase_delay_deg":float(diag["max_pair_phase_delay_deg"]),
                "holdout_patterns":hrows,
            })
            del A,prep,q; gc.collect()
    finally:
        c2.LAMBDA_REL=old_lambda
    out=Path("article_cases_output")/a.out; out.mkdir(parents=True,exist_ok=True)
    report={
        "experiment":"RETARDED_HARMONIC_COLLOCATION_OVERFIT_DIAGNOSTIC",
        "purpose":"quantify collocation-vs-independent-holdout generalization using the final retarded harmonic electric-dipole MAS model",
        "geometry":"60-mm article geometry" if a.formal_geometry else "compact diagnostic geometry",
        "interpretation":"Fit residual is not a boundary-generalization metric; independent holdout is required.",
        "results":results,
    }
    (out/"overfit_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    fields=[k for k in results[0] if k!="holdout_patterns"]
    with (out/"overfit_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in results: w.writerow({k:r[k] for k in fields})
    for r in results:
        print(f"fit_fraction={r['fit_fraction']:.3f} eq/unknown={r['fit_equations']/r['unknowns']:.3f} "
              f"fit={100*r['collocation_fit_rel_l2']:.4f}% "
              f"independent_worst={100*r['independent_holdout_worst_gate']:.4f}%")

if __name__=="__main__":
    run(parse_args())
