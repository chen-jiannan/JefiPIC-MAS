#!/usr/bin/env python3
"""
Step 2-MB1: incident-normalization and feed-localization confirmation.

NUMBERING
---------
Step 2-MB introduced the rounded two-arm dipole + feed-gap geometry using the
frozen Step 2-MA best construction (d/R=0.80, global-FPS, tangent2). MB showed
that the two-arm geometry itself remained accurate for the uniform and
off-axis external-dipole incidents, while two diagnostic issues required
separation:

1) weakly excited regions can show large local-relative percentages because
   the regional incident tangential-field RMS is very small;
2) the previous tanh(z/w) "feed-like" field approaches +/- z-hat away from the
   gap, so it is actually a global axial antisymmetric stress rather than a
   localized feed-gap field.

The present Step 2-MB1 changes only the diagnostic design. Geometry, source
placement, source count, basis, collocation, regularization, and the two
independent holdout patterns remain frozen.

PURPOSE
-------
Use three article-relevant acceptance incidents:
  - uniform,
  - off-axis external electrostatic dipole,
  - a bounded smooth localized axial feed field sech^2(z/w).

Also retain the previous tanh axial antisymmetric field as a NON-GATING stress
test that documents the current electric-dipole MAS limitation for rounded
ends under nearly pure axial loading.

For every holdout region MB1 reports:
  - local-relative tangential residual,
  - residual normalized by the GLOBAL incident tangential RMS,
  - regional incident-strength ratio eta_r.

The pre-declared acceptance rule is:
  - global region: local-relative error < holdout gate;
  - regional eta_r >= weak-region threshold: local-relative error < gate;
  - regional eta_r < weak-region threshold: global-normalized regional
    residual < gate.
The same rule must pass on both independent holdout patterns for all three
article-relevant incidents. The axial antisymmetric stress test is reported but
does not participate in acceptance.

This script is diagnostic only. It does NOT modify production modules or the
manuscript.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree

try:
    import step2kd1_coherent_closed_auxiliary_surface as kd1
    import step2ma_smooth_capsule_acceptance as ma
except ImportError as exc:
    raise SystemExit(
        "Step 2-MB requires article_cases/step2kd1_coherent_closed_auxiliary_surface.py "
        "and article_cases/step2ma_smooth_capsule_acceptance.py."
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
TINY = kd1.TINY
EPS0 = kd1.EPS0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="step2mb1_incident_normalization_feed_localization")
    p.add_argument("--radius", type=float, default=1.0e-3)
    p.add_argument("--total-length", type=float, default=60.0e-3)
    p.add_argument("--feed-gap", type=float, default=2.0e-3)
    p.add_argument("--depth-frac", type=float, default=0.80)
    p.add_argument("--sources-per-arm", type=int, default=900)
    p.add_argument("--collocation-factor", type=float, default=2.0)
    p.add_argument("--lambda-rel", type=float, default=1.0e-4)
    p.add_argument("--dt", type=float, default=1.0e-9)
    p.add_argument("--holdout-gate", type=float, default=0.05)
    p.add_argument("--collocation-gate", type=float, default=0.05)
    p.add_argument("--weak-region-threshold", type=float, default=0.20)
    p.add_argument("--feed-width-over-gap", type=float, default=1.0)
    return p.parse_args()


def shift_triangles(T, dz):
    U = np.asarray(T, float).copy()
    U[:, :, 2] += float(dz)
    return U


def shift_points(P, dz):
    Q = np.asarray(P, float).copy()
    Q[:, 2] += float(dz)
    return Q


def make_arm(radius, Rphysical, Larm, z0, ncyl, nphi, nhemi):
    Tloc = ma.generate_capsule(radius, Rphysical, Larm, ncyl, nphi, nhemi)
    T = shift_triangles(Tloc, z0)
    C0, _, A = kd1.tri_geom(Tloc)
    C, N = ma.project_to_capsule(C0, radius, Rphysical, Larm)
    return T, shift_points(C, z0), N, A, Tloc


def bary_arm(Tloc, weights, R, Larm, z0):
    P, N = ma.barycentric_points(Tloc, weights, R, Larm)
    return shift_points(P, z0), N


def classify(P, arm_id, R, Larm, z_bottom, z_top):
    P = np.asarray(P, float)
    arm_id = np.asarray(arm_id, int)
    zloc = np.where(arm_id == 0, P[:, 2] - z_bottom, P[:, 2] - z_top)
    caps = (zloc < R) | (zloc > Larm - R)
    band = 0.75 * R
    transition = (np.abs(zloc - R) <= band) | (np.abs(zloc - (Larm - R)) <= band)
    side_mid = (~caps) & (~transition)
    # arm 0 is lower arm: feed tip is its +z end. arm 1 upper arm: feed tip is its -z end.
    feed_tip = ((arm_id == 0) & (zloc >= Larm - 0.50*R)) | ((arm_id == 1) & (zloc <= 0.50*R))
    outer_tip = ((arm_id == 0) & (zloc <= 0.50*R)) | ((arm_id == 1) & (zloc >= Larm - 0.50*R))
    feed_rounded = ((arm_id == 0) & (zloc > Larm - R)) | ((arm_id == 1) & (zloc < R))
    outer_rounded = ((arm_id == 0) & (zloc < R)) | ((arm_id == 1) & (zloc > Larm - R))
    feed_transition = ((arm_id == 0) & (np.abs(zloc - (Larm - R)) <= band)) | ((arm_id == 1) & (np.abs(zloc - R) <= band))
    return {
        "global": np.ones(len(P), bool),
        "feed_rounded": feed_rounded,
        "feed_tips": feed_tip,
        "feed_transition": feed_transition,
        "outer_rounded": outer_rounded,
        "outer_tips": outer_tip,
        "all_transition": transition,
        "side_mid": side_mid,
        "lower_arm": arm_id == 0,
        "upper_arm": arm_id == 1,
    }


def static_dipole(P, r0, p):
    P = np.asarray(P, float)
    r = P - np.asarray(r0, float)
    rr = np.maximum(np.linalg.norm(r, axis=1), 1e-15)
    h = r / rr[:, None]
    p = np.asarray(p, float)
    return (1.0 / (4.0*np.pi*EPS0)) * (
        3.0*h*np.sum(h*p, axis=1)[:, None] - p
    ) / rr[:, None]**3


def inc_external(P, total_length, R):
    return static_dipole(P, [4.0*R, 3.0*R, 0.21*total_length], [1.0e-12, -0.45e-12, 0.75e-12])


def inc_feed_localized(P, gap, width_over_gap=1.0):
    # Smooth, bounded, localized axial field. It is the z-gradient of a tanh
    # potential and decays away from the feed gap instead of becoming a global
    # +/- axial field.
    P = np.asarray(P, float)
    w = max(float(width_over_gap)*float(gap), 1e-15)
    u = P[:, 2] / w
    # Stable sech^2 without overflow for large |u|.
    au = np.abs(u)
    e = np.exp(-2.0*au)
    sech2 = 4.0*e / (1.0 + e)**2
    E = np.zeros((len(P), 3), float)
    E[:, 2] = sech2
    return E


def inc_axial_stress(P, gap):
    # Retain the old MB field as a NON-GATING limitation/stress diagnostic.
    P = np.asarray(P, float)
    w = max(0.5*float(gap), 1e-15)
    E = np.zeros((len(P), 3), float)
    E[:, 2] = np.tanh(P[:, 2] / w)
    return E


def tangential_rms(E, N, mask):
    E = np.asarray(E, float)
    N = np.asarray(N, float)
    mask = np.asarray(mask, bool)
    if not np.any(mask):
        return None
    Et = E - N*np.sum(E*N, axis=1)[:, None]
    return float(np.sqrt(np.mean(np.sum(Et[mask]*Et[mask], axis=1))))


def region_metrics(total, inc, N, masks):
    global_mask = np.ones(len(N), dtype=bool)
    inc_global = tangential_rms(inc, N, global_mask)
    out = {}
    for name, mask in masks.items():
        if not np.any(mask):
            continue
        res_rms = tangential_rms(total, N, mask)
        inc_rms = tangential_rms(inc, N, mask)
        out[name] = {
            "residual_rms": res_rms,
            "incident_rms": inc_rms,
            "local_rel": float(res_rms / max(inc_rms, TINY)),
            "global_norm": float(res_rms / max(inc_global, TINY)),
            "incident_ratio": float(inc_rms / max(inc_global, TINY)),
        }
    return out


def gate_metric(region, metric, threshold):
    if region == "global" or metric["incident_ratio"] >= threshold:
        return "local_rel", metric["local_rel"]
    return "global_norm", metric["global_norm"]


def write_csv(path, rows):
    if not rows:
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def cross_min(A, B, scale):
    if len(A) == 0 or len(B) == 0:
        return None
    d, _ = cKDTree(B).query(A, k=1)
    return float(np.min(d) / max(scale, TINY))


def factor_ridge(As, lambda_rel):
    smax = kd1.estimate_smax(As)
    damp = float(lambda_rel) * float(smax)
    H = As.T @ As
    H.flat[::H.shape[0]+1] += damp*damp
    cf = cho_factor(H, lower=True, overwrite_a=True, check_finite=False)
    return cf, damp, float(smax)


def solve_factored(As, cs, rs, b, cf, damp):
    bw = b * rs
    g = As.T @ bw
    y = cho_solve(cf, g, check_finite=False)
    x = y / cs
    r = As @ y - bw
    grad = As.T @ r + damp*damp*y
    scale = np.linalg.norm(As, ord="fro") * np.linalg.norm(r) + damp*damp*np.linalg.norm(y)
    return x, {
        "weighted_residual_rel": float(np.linalg.norm(r)/max(np.linalg.norm(bw), TINY)),
        "damped_optimality_rel": float(np.linalg.norm(grad)/max(scale, TINY)),
    }


def safe_max(rows, keys):
    vals = [float(r[k]) for r in rows for k in keys if k in r and r[k] is not None]
    return max(vals) if vals else None


def main():
    a = parse_args()
    out = ROOT / "article_cases_output" / a.out
    out.mkdir(parents=True, exist_ok=True)

    R = float(a.radius)
    total = 16.0e-3 if a.quick else float(a.total_length)
    gap = float(a.feed_gap)
    Larm = 0.5*(total-gap)
    if R <= 0 or gap <= 0 or Larm <= 2.0*R:
        raise ValueError("Require R>0, gap>0, and each arm length > 2R")
    frac = float(a.depth_frac)
    if not (0.0 < frac < 1.0):
        raise ValueError("depth-frac must lie in (0,1)")
    d, Rs = frac*R, (1.0-frac)*R

    z_bottom = -0.5*total
    z_top = 0.5*gap

    if a.quick:
        nphi, ncyl, nhemi = 28, 28, 10
        nsrc_arm = min(180, int(a.sources_per_arm))
    else:
        nphi, ncyl, nhemi = 72, 96, 24
        nsrc_arm = int(a.sources_per_arm)
    ncoll_arm = max(nsrc_arm+1, int(math.ceil(float(a.collocation_factor)*nsrc_arm)))

    # Physical two-arm geometry.
    T0, C0, N0, A0, T0loc = make_arm(R, R, Larm, z_bottom, ncyl, nphi, nhemi)
    T1, C1, N1, A1, T1loc = make_arm(R, R, Larm, z_top, ncyl, nphi, nhemi)
    T = np.concatenate((T0, T1), axis=0)
    kd1.write_ascii_stl(out / "step2mb1_physical_rounded_dipole.stl", T)

    area_arm = float(np.sum(A0))

    # Frozen physical collocation: same density on both arms.
    i0 = ma.select_fps(C0, ncoll_arm, seed=3101)
    i1 = ma.select_fps(C1, ncoll_arm, seed=3102)
    OP = np.concatenate((C0[i0], C1[i1]), axis=0)
    ON = np.concatenate((N0[i0], N1[i1]), axis=0)
    W0, _ = kd1.voronoi_weights(C0, A0, C0[i0])
    W1, _ = kd1.voronoi_weights(C1, A1, C1[i1])
    OW = np.concatenate((W0, W1))

    # Two independent holdout patterns, every physical triangle.
    H00, HN00 = bary_arm(T0loc, [0.19,0.37,0.44], R, Larm, z_bottom)
    H01, HN01 = bary_arm(T1loc, [0.19,0.37,0.44], R, Larm, z_top)
    HP = np.concatenate((H00,H01)); HN = np.concatenate((HN00,HN01))
    hid = np.concatenate((np.zeros(len(H00),int), np.ones(len(H01),int)))
    masks = classify(HP, hid, R, Larm, z_bottom, z_top)

    H20, HN20 = bary_arm(T0loc, [0.53,0.29,0.18], R, Larm, z_bottom)
    H21, HN21 = bary_arm(T1loc, [0.53,0.29,0.18], R, Larm, z_top)
    HP2 = np.concatenate((H20,H21)); HN2 = np.concatenate((HN20,HN21))
    hid2 = np.concatenate((np.zeros(len(H20),int), np.ones(len(H21),int)))
    masks2 = classify(HP2, hid2, R, Larm, z_bottom, z_top)

    # Independent coherent inner capsule on each arm, fixed MA-best depth.
    Ts0, SC0, SN0, SA0, _ = make_arm(Rs, R, Larm, z_bottom, ncyl, nphi, nhemi)
    Ts1, SC1, SN1, SA1, _ = make_arm(Rs, R, Larm, z_top, ncyl, nphi, nhemi)
    j0 = ma.select_fps(SC0, nsrc_arm, seed=4101)
    j1 = ma.select_fps(SC1, nsrc_arm, seed=4102)
    SP0, SP1 = SC0[j0], SC1[j1]
    SNs0, SNs1 = SN0[j0], SN1[j1]
    SW0, _ = kd1.voronoi_weights(SC0, SA0, SP0)
    SW1, _ = kd1.voronoi_weights(SC1, SA1, SP1)
    SW0 *= area_arm/max(float(np.sum(SW0)),TINY)
    SW1 *= area_arm/max(float(np.sum(SW1)),TINY)
    SP = np.concatenate((SP0,SP1)); SN = np.concatenate((SNs0,SNs1)); SW=np.concatenate((SW0,SW1))
    src_arm_id = np.concatenate((np.zeros(len(SP0),int),np.ones(len(SP1),int)))
    modes=np.full(len(SP),2,dtype=np.int32); off=kd1.offsets(modes)

    kd1.write_ascii_stl(out / "step2mb1_inner_lower_arm.stl", Ts0)
    kd1.write_ascii_stl(out / "step2mb1_inner_upper_arm.stl", Ts1)

    nn0=ma.nn_stats(SP0,d); nn1=ma.nn_stats(SP1,d)
    cross_src = cross_min(SP0,SP1,d)
    cross_phys = cross_min(C0,C1,R)

    print("Step 2-MB1 incident normalization + feed localization confirmation")
    print("  Step 3 started                 : NO")
    print(f"  total length / gap / R         : {1e3*total:.3f} / {1e3*gap:.3f} / {1e3*R:.3f} mm")
    print(f"  arm length                     : {1e3*Larm:.3f} mm")
    print(f"  fixed depth d/R                : {frac:.3f}")
    print(f"  sources per arm / total        : {nsrc_arm} / {len(SP)}")
    print(f"  unknowns                       : {off[-1]}")
    print(f"  collocation total              : {len(OP)} ({len(OP)/len(SP):.2f}x sources)")
    print(f"  holdout points                 : {len(HP)} + {len(HP2)}")
    print(f"  inner-source cross-arm min/d   : {cross_src:.4f}")

    print(f"\nassembling {2*len(OP)} x {off[-1]} operator ...", flush=True)
    Mat=kd1.assemble(OP,ON,SP,SN,SW,modes,off,a.dt)
    rs,_,cs,As=kd1.prepare_scaled_operator(Mat,OW)
    coldiag=kd1.operator_column_diagnostics(cs,modes,off,SN)
    print("factoring ridge normal matrix once for all incidents ...", flush=True)
    cf,damp,smax=factor_ridge(As,a.lambda_rel)

    acceptance_incidents=("uniform","external_dipole","feed_localized_sech2")
    stress_incidents=("axial_antisymmetric_stress",)
    incidents={
        "uniform": lambda P: kd1.inc_uniform(P),
        "external_dipole": lambda P: inc_external(P,total,R),
        "feed_localized_sech2": lambda P: inc_feed_localized(P,gap,a.feed_width_over_gap),
        "axial_antisymmetric_stress": lambda P: inc_axial_stress(P,gap),
    }
    gate_regions=("global","feed_rounded","feed_tips","feed_transition",
                  "outer_rounded","outer_tips","all_transition","side_mid")

    rows=[]
    gate_rows=[]
    coeff_rows=[]
    for name,fn in incidents.items():
        b=kd1.tangent_rhs(fn(OP),ON)
        x,diag=solve_factored(As,cs,rs,b,cf,damp)
        coll=float(np.linalg.norm(Mat@x-b)/max(np.linalg.norm(b),TINY))

        E1,_=kd1.eval_field(HP,SP,SN,SW,x,modes,off,a.dt)
        I1=fn(HP)
        m1=region_metrics(I1+E1,I1,HN,masks)

        E2,_=kd1.eval_field(HP2,SP,SN,SW,x,modes,off,a.dt)
        I2=fn(HP2)
        m2=region_metrics(I2+E2,I2,HN2,masks2)

        row={
            "incident":name,
            "acceptance_incident":name in acceptance_incidents,
            "collocation_rel_l2":coll,
            "direct_weighted_residual_rel":diag["weighted_residual_rel"],
            "direct_damped_optimality_rel":diag["damped_optimality_rel"],
            "damp":damp,
        }
        for prefix,mm in (("holdout",m1),("spot",m2)):
            for region,vals in mm.items():
                # Compatibility alias: rel_rms remains the local-relative value.
                row[f"{prefix}_{region}_rel_rms"]=vals["local_rel"]
                row[f"{prefix}_{region}_local_rel_rms"]=vals["local_rel"]
                row[f"{prefix}_{region}_global_norm_rms"]=vals["global_norm"]
                row[f"{prefix}_{region}_incident_ratio"]=vals["incident_ratio"]
                row[f"{prefix}_{region}_incident_rms"]=vals["incident_rms"]
                row[f"{prefix}_{region}_residual_rms"]=vals["residual_rms"]
        rows.append(row)

        if name in acceptance_incidents:
            gate_rows.append({
                "incident":name,"pattern":"collocation","region":"global",
                "incident_ratio":1.0,"gate_metric":"collocation_rel_l2",
                "gate_value":coll,"gate":a.collocation_gate,
                "passed":bool(coll < a.collocation_gate),
            })
            for pattern,mm in (("holdout",m1),("spot",m2)):
                for region in gate_regions:
                    vals=mm[region]
                    metric_name,metric_value=gate_metric(
                        region,vals,a.weak_region_threshold
                    )
                    gate_rows.append({
                        "incident":name,"pattern":pattern,"region":region,
                        "incident_ratio":vals["incident_ratio"],
                        "gate_metric":metric_name,
                        "local_rel_rms":vals["local_rel"],
                        "global_norm_rms":vals["global_norm"],
                        "gate_value":metric_value,"gate":a.holdout_gate,
                        "passed":bool(metric_value < a.holdout_gate),
                    })

        for j in range(len(SP)):
            aa,bb=int(off[j]),int(off[j+1])
            coeff_rows.append({
                "incident":name,"source_index":j,
                "arm":"lower" if src_arm_id[j]==0 else "upper",
                "coeff_l2":float(np.linalg.norm(x[aa:bb]))
            })

        status="GATING" if name in acceptance_incidents else "STRESS"
        print(
            f"  {name:27s} [{status}] coll={100*coll:7.3f}% "
            f"global={100*m1['global']['local_rel']:7.3f}% "
            f"feed={100*m1['feed_rounded']['local_rel']:7.3f}% "
            f"eta_feed={m1['feed_rounded']['incident_ratio']:6.3f} "
            f"outer={100*m1['outer_rounded']['local_rel']:7.3f}% "
            f"eta_outer={m1['outer_rounded']['incident_ratio']:6.3f}"
        )

    accepted=all(bool(r["passed"]) for r in gate_rows)
    worst_gate=max(float(r["gate_value"]) for r in gate_rows) if gate_rows else None
    weak_gate_rows=[
        r for r in gate_rows
        if r["pattern"]!="collocation"
        and float(r.get("incident_ratio",1.0)) < a.weak_region_threshold
    ]
    worst_local_all=max(
        float(r[k]) for r in rows
        for k in r if k.endswith("_local_rel_rms") and r[k] is not None
    )
    stress_rows=[r for r in rows if not bool(r["acceptance_incident"])]
    worst_stress_local=max(
        [float(r[k]) for r in stress_rows
         for k in r if k.endswith("_local_rel_rms") and r[k] is not None],
        default=None,
    )

    source_rows=[]
    for j in range(len(SP)):
        source_rows.append({
            "source_index":j,"arm":"lower" if src_arm_id[j]==0 else "upper",
            "x":SP[j,0],"y":SP[j,1],"z":SP[j,2],
            "nx":SN[j,0],"ny":SN[j,1],"nz":SN[j,2],
            "weight":SW[j],"mode":"tangent2"
        })

    write_csv(out/"step2mb1_incident_results.csv",rows)
    write_csv(out/"step2mb1_gate_results.csv",gate_rows)
    write_csv(out/"step2mb1_source_cloud.csv",source_rows)
    write_csv(out/"step2mb1_coefficient_norms.csv",coeff_rows)

    summary={
        "experiment":"Step 2-MB1","step3_started":False,
        "purpose":"incident-normalization and feed-localization confirmation on the frozen rounded two-arm dipole",
        "geometry":"two smooth capsules / rounded dipole","sharp_edges":False,
        "radius":R,"total_length":total,"feed_gap":gap,"arm_length":Larm,
        "depth_over_R":frac,"sources_per_arm":nsrc_arm,"source_count":len(SP),
        "unknown_count":int(off[-1]),"collocation_points":len(OP),
        "holdout_points_primary":len(HP),"holdout_points_spot":len(HP2),
        "source_nn_lower_p50_over_d":nn0["p50"],
        "source_nn_upper_p50_over_d":nn1["p50"],
        "source_cross_arm_min_over_d":cross_src,
        "physical_cross_arm_min_over_R":cross_phys,
        "column_norm_spread":coldiag["all"]["max_over_min"],
        "smax_scaled_est":smax,"damp":damp,
        "acceptance_incidents":list(acceptance_incidents),
        "non_gating_stress_incidents":list(stress_incidents),
        "weak_region_threshold":float(a.weak_region_threshold),
        "regional_gate_rule":{
            "global":"local_relative",
            "energized_region":f"local_relative when incident_ratio >= {a.weak_region_threshold}",
            "weak_region":f"global_normalized residual when incident_ratio < {a.weak_region_threshold}",
        },
        "feed_localized_definition":{
            "Ez":"sech^2(z/w)",
            "width_over_gap":float(a.feed_width_over_gap),
            "width_m":float(a.feed_width_over_gap*gap),
        },
        "worst_acceptance_gate_value":worst_gate,
        "weak_region_gate_count":len(weak_gate_rows),
        "worst_local_relative_reported_all_incidents":worst_local_all,
        "worst_non_gating_stress_local_relative":worst_stress_local,
        "accepted":bool(accepted),
        "production_integration_authorized":False,
        "incidents":rows,
        "gate_results":gate_rows,
    }
    with open(out/"step2mb1_report.json","w") as f:
        json.dump(summary,f,indent=2)

    lines=[
        "Step 2-MB1 incident normalization + feed localization confirmation",
        "=================================================================",
        "",
        "NUMBERING",
        "  major step                         : Step 2",
        "  branch                             : M",
        "  diagnostic                         : MB1",
        "  Step 3 started                     : NO",
        "",
        "FROZEN MAS / GEOMETRY",
        f"  physical total length / gap / R    : {1e3*total:.3f} / {1e3*gap:.3f} / {1e3*R:.3f} mm",
        f"  arm length                         : {1e3*Larm:.3f} mm",
        f"  source depth d/R                   : {frac:.3f}",
        "  source sampling                    : global FPS independently on each inner arm",
        "  basis                              : tangent2 electric dipoles",
        f"  sources per arm / total            : {nsrc_arm} / {len(SP)}",
        f"  unknowns                           : {off[-1]}",
        f"  collocation points                 : {len(OP)}",
        f"  holdout points                     : {len(HP)} + {len(HP2)} spot-check",
        f"  feed-localized width w/gap         : {a.feed_width_over_gap:.3f}",
        "",
        "PRE-DECLARED REGIONAL ACCEPTANCE RULE",
        f"  energized threshold eta_r          : {a.weak_region_threshold:.3f}",
        f"  global / energized-region gate     : local-relative < {100*a.holdout_gate:.2f}%",
        f"  weak-region gate                   : global-normalized residual < {100*a.holdout_gate:.2f}%",
        f"  collocation gate                   : < {100*a.collocation_gate:.2f}%",
        "  axial antisymmetric stress         : REPORTED, NON-GATING",
        "",
        "GEOMETRY / SCALING",
        f"  lower-arm source NN p50/d          : {nn0['p50']:.4f}",
        f"  upper-arm source NN p50/d          : {nn1['p50']:.4f}",
        f"  source cross-arm min/d             : {cross_src:.4f}",
        f"  physical gap/R                     : {gap/R:.4f}",
        f"  column norm spread                 : {coldiag['all']['max_over_min']:.4e}",
        "",
    ]
    for r in rows:
        status="GATING" if r["acceptance_incident"] else "NON-GATING STRESS"
        lines += [
            f"  {r['incident']} [{status}]",
            f"    collocation                      : {100*r['collocation_rel_l2']:.3f}%",
            f"    global local-relative            : {100*r['holdout_global_local_rel_rms']:.3f}% / spot {100*r['spot_global_local_rel_rms']:.3f}%",
            f"    feed rounded local/global-norm   : {100*r['holdout_feed_rounded_local_rel_rms']:.3f}% / {100*r['holdout_feed_rounded_global_norm_rms']:.3f}%",
            f"    feed rounded eta                 : {r['holdout_feed_rounded_incident_ratio']:.4f}",
            f"    feed transition local/global-norm: {100*r['holdout_feed_transition_local_rel_rms']:.3f}% / {100*r['holdout_feed_transition_global_norm_rms']:.3f}%",
            f"    feed transition eta              : {r['holdout_feed_transition_incident_ratio']:.4f}",
            f"    outer rounded local/global-norm  : {100*r['holdout_outer_rounded_local_rel_rms']:.3f}% / {100*r['holdout_outer_rounded_global_norm_rms']:.3f}%",
            f"    outer rounded eta                : {r['holdout_outer_rounded_incident_ratio']:.4f}",
        ]

    failed=[r for r in gate_rows if not r["passed"]]
    lines += [
        "",
        f"worst formal acceptance gate value   : {100*worst_gate:.3f}%",
        f"weak-region gate evaluations         : {len(weak_gate_rows)}",
        f"formal gate failures                 : {len(failed)}",
        f"ARTICLE-RELEVANT MB1 ACCEPTANCE      : {'PASS' if accepted else 'FAIL'}",
        f"worst local-relative incl. stress    : {100*worst_local_all:.3f}%",
        f"worst NON-GATING axial-stress local  : {100*worst_stress_local:.3f}%" if worst_stress_local is not None else "worst NON-GATING axial-stress local  : n/a",
        "",
        "Step 2-MB1 PRODUCTION INTEGRATION    : NOT AUTHORIZED BY THIS DIAGNOSTIC",
        "",
        "Interpretation:",
        "  Acceptance concerns the intended electrically-small, time-local near-field",
        "  regime and the three article-relevant incidents only.",
        "  Large local-relative percentages in weakly excited regions remain reported,",
        "  but they do not dominate the gate when the local incident RMS is <20% of",
        "  the global incident RMS; their absolute/global-normalized residual is gated.",
        "  The axial antisymmetric case remains visible as a limitation/stress result",
        "  and cannot be used to claim universal PEC or arbitrary-polarization accuracy.",
    ]
    (out/"step2mb1_summary.txt").write_text("\n".join(lines)+"\n")
    print("\n"+"\n".join(lines))
    if failed:
        print("\nFailed formal gates:")
        for r in failed:
            print(
                f"  {r['incident']:24s} {r['pattern']:10s} {r['region']:16s} "
                f"{r['gate_metric']:12s}={100*float(r['gate_value']):7.3f}% "
                f"eta={float(r.get('incident_ratio',1.0)):.4f}"
            )
    print(f"\nWrote results to: {out}")


if __name__ == "__main__":
    main()
