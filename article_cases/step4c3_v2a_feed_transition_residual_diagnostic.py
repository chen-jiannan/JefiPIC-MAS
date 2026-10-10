#!/usr/bin/env python3
"""
Step 4C3 V2A — feed-transition residual localization diagnostic.

This is a DIAGNOSTIC, not a new physical formulation.

It reuses:
- the exact Step-4C3 broadband forcing NPZ;
- the exact Step-4C3-V2 Retarded-MOT backend;
- the exact Step-4C2-V4 causal retarded harmonic backend.

It does NOT change:
- dt;
- lambda_rel;
- source density;
- collocation density;
- zero softening;
- geometry;
- plasma forcing;
- the 5% gate.

Questions answered
------------------
1. Is the 5.212% feed-transition failure persistent or concentrated in a few
   diagnostic times?

2. Is the feed-transition residual dominated by 3/6/9 MHz content, or by the
   remaining broadband/transient component?

3. If the ACTUAL recorded plasma+feed spatial phasors at 3/6/9 MHz are solved
   with the continuous causal retarded harmonic MAS, does feed_transition still
   exceed 5%?

Interpretation
--------------
- If any actual 3/6/9 MHz harmonic spatial audit itself exceeds 5% at
  feed_transition, spatial representation/generalization is implicated and a
  controlled density/collocation refinement is justified.

- If all three actual-harmonic audits pass while the full broadband replay
  remains >5% locally, do NOT tune density yet. The remaining discrepancy is
  associated with broadband/transient/electrostatic state closure, and the
  next diagnostic should examine explicit surface-charge continuity.

No result from V2A is a manuscript-final amplification factor.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np

import step4c3_v2_retarded_mot_broadband_replay as c3
import step4c2_v4_retarded_frequency_regression as c2


FREQS = (3.0e6, 6.0e6, 9.0e6)
TARGET_REGION = "feed_transition"
REPORT_REPRO_TOL = 1.0e-8
FIT_CONDITION_LIMIT = 100.0
MIN_HARMONIC_STRENGTH_RATIO = 0.01


def tangential_vectors_real(E, N):
    E=np.asarray(E,dtype=np.float64)
    N=np.asarray(N,dtype=np.float64)
    return E-N*np.sum(E*N,axis=1)[:,None]


def fit_multitone(times, values, freqs=FREQS):
    """
    Joint least-squares fit:
      DC + linear trend + sum_f [a_f cos(wt)+b_f sin(wt)].

    values shape: [Nt, ...].

    exp(+i wt) phasor convention:
      x(t)=Re{xhat exp(+i wt)} = Re(xhat) cos - Im(xhat) sin
      => xhat = a - i b.
    """
    t=np.asarray(times,dtype=np.float64)
    y=np.asarray(values)
    if y.shape[0] != len(t):
        raise ValueError("time/value length mismatch")
    if len(t) < 2+2*len(freqs):
        raise RuntimeError(
            f"insufficient samples for joint multitone fit: {len(t)}"
        )

    tmid=float(np.mean(t))
    tscale=max(float(np.ptp(t)), 1.0e-30)
    tn=(t-tmid)/tscale

    cols=[np.ones(len(t)), tn]
    for f in freqs:
        w=2.0*np.pi*float(f)
        cols.extend([np.cos(w*t), np.sin(w*t)])
    X=np.column_stack(cols)

    shape=y.shape[1:]
    Y=y.reshape(len(t), -1)
    coef, _, rank, svals=np.linalg.lstsq(X,Y,rcond=None)
    fitted=(X@coef).reshape(y.shape)
    remainder=y-fitted

    phasors={}
    for i,f in enumerate(freqs):
        a=coef[2+2*i]
        b=coef[3+2*i]
        phasors[float(f)]=(a-1j*b).reshape(shape)

    return {
        "condition_number": float(np.linalg.cond(X)),
        "rank": int(rank),
        "num_samples": int(len(t)),
        "phasors": phasors,
        "fitted": fitted,
        "remainder": remainder,
        "dc": coef[0].reshape(shape),
        "trend_per_window": coef[1].reshape(shape),
    }


def vector_rms_time_space(v):
    v=np.asarray(v)
    if v.size==0:
        return 0.0
    return float(np.sqrt(np.mean(np.sum(np.abs(v)**2,axis=-1))))


def complex_region_metrics(total, incident, normals, mask_matrix, region_names):
    """
    Same C3 physical gate rule, but for complex harmonic phasors.
    The "global" normalization uses the recorded global mask, exactly as C3.
    """
    Et_total=c2.tangential_vectors(total,normals)
    Et_inc=c2.tangential_vectors(incident,normals)

    if "global" not in region_names:
        raise RuntimeError("holdout metadata has no global region")
    gi=region_names.index("global")
    gm=mask_matrix[gi]
    if not np.any(gm):
        raise RuntimeError("global holdout mask is empty")

    global_inc=float(np.sqrt(np.mean(np.sum(np.abs(Et_inc[gm])**2,axis=1))))
    rows={}
    for ir,region in enumerate(region_names):
        mask=mask_matrix[ir]
        if not np.any(mask):
            continue
        rr=float(np.sqrt(np.mean(np.sum(np.abs(Et_total[mask])**2,axis=1))))
        ii=float(np.sqrt(np.mean(np.sum(np.abs(Et_inc[mask])**2,axis=1))))
        incident_ratio=ii/max(global_inc,c3.TINY)
        local_rel=rr/max(ii,c3.TINY)
        global_norm=rr/max(global_inc,c3.TINY)
        if region=="global" or incident_ratio>=c3.WEAK_REGION_THRESHOLD:
            metric="local_rel"; value=local_rel
        else:
            metric="global_norm"; value=global_norm
        rows[region]={
            "region":region,
            "residual_rms":rr,
            "incident_rms":ii,
            "incident_ratio":incident_ratio,
            "local_rel":local_rel,
            "global_norm":global_norm,
            "gate_metric":metric,
            "gate_value":value,
            "passed":bool(value<c3.BOUNDARY_GATE),
        }
    return rows


def top_energy_rows(rows, key, n=12):
    rows=sorted(rows,key=lambda r:r[key],reverse=True)
    total=sum(float(r[key]) for r in rows)
    running=0.0
    out=[]
    for r in rows[:n]:
        running+=float(r[key])
        q=dict(r)
        q["fraction_of_total"]=float(r[key]/max(total,c3.TINY))
        q["cumulative_fraction"]=float(running/max(total,c3.TINY))
        out.append(q)
    return out


def concentration_fractions(energies):
    e=np.sort(np.asarray(energies,dtype=np.float64))[::-1]
    total=float(np.sum(e))
    if total<=0:
        return {"top_1_step":0.0,"top_3_steps":0.0,"top_5_steps":0.0,
                "top_20pct_steps":0.0}
    n=len(e)
    return {
        "top_1_step":float(np.sum(e[:1])/total),
        "top_3_steps":float(np.sum(e[:min(3,n)])/total),
        "top_5_steps":float(np.sum(e[:min(5,n)])/total),
        "top_20pct_steps":float(np.sum(e[:max(1,math.ceil(.2*n))])/total),
    }


def load_reference_report(path):
    if path is None:
        return None
    p=Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def run_diagnostic(args):
    forcing_path=Path(args.forcing)
    z,meta=c3.load_forcing(forcing_path)
    c3.validate_forcing_metadata(meta)

    out=Path("article_cases_output")/args.out
    out.mkdir(parents=True,exist_ok=True)

    keys=(
        "mode","total_length_m","sources_per_arm","n_phi","n_cyl",
        "n_hemi","cycles","analysis_start_cycles","diag_stride",
        "holdout_points_target",
    )
    settings={k:meta[k] for k in keys}
    g=c3.build_geometry(settings)
    fp=c3.geometry_fingerprint(g)
    if fp!=meta["geometry_fingerprint"]:
        raise RuntimeError("forcing/replay geometry fingerprint mismatch")

    plasma_coll=z["plasma_collocation_E_V_per_m"]
    times=np.asarray(z["time_s"],dtype=np.float64)
    diag_steps=np.asarray(z["diag_steps"],dtype=np.int64)
    plasma_holdout=z["plasma_holdout_E_V_per_m"]
    Hpoints=np.asarray(z["holdout_points_m"],dtype=np.float64)
    Hnormals=np.asarray(z["holdout_normals"],dtype=np.float64)
    Hmask=np.asarray(z["holdout_mask_matrix"],dtype=np.uint8).astype(bool)
    region_names=list(meta["holdout_region_names"])

    if TARGET_REGION not in region_names:
        raise RuntimeError(f"{TARGET_REGION} mask not found")
    target_ir=region_names.index(TARGET_REGION)
    target_mask=Hmask[target_ir]
    if not np.any(target_mask):
        raise RuntimeError(f"{TARGET_REGION} mask is empty")

    if len(plasma_coll)!=len(times):
        raise RuntimeError("forcing collocation time length mismatch")
    if len(plasma_holdout)!=len(diag_steps):
        raise RuntimeError("forcing holdout/diag-step length mismatch")

    analysis_start_s=float(settings["analysis_start_cycles"])/c3.F_DRIVE
    analysis_mask=times>=analysis_start_s
    analysis_idx=np.flatnonzero(analysis_mask)
    diag_lookup={int(s):i for i,s in enumerate(diag_steps)}
    diag_analysis=[int(s) for s in diag_steps if int(s)<len(times) and times[int(s)]>=analysis_start_s]
    if len(diag_analysis)<8:
        raise RuntimeError("too few post-ramp holdout diagnostic samples")

    print("[Step 4C3 V2A] assembling exact C1-V4 Retarded-MOT replay backend ...",flush=True)
    obs_chunk=8 if settings["mode"]=="formal" else 16
    Hp,Hk,An,delay=c3.assemble_retarded_mot_operators(g,obs_chunk)
    if delay["max_delay_over_dt"]>=1.0:
        raise RuntimeError("one-step MOT invalid at collocation")
    prep=c3.prepare_fixed_ridge_inplace(An,g.solve_weights)
    del An
    gc.collect()

    ndof=2*g.num_sources
    q_prev=np.zeros(ndof,dtype=np.float64)
    p_prev=np.zeros(ndof,dtype=np.float64)

    coll_res_sum=0.0
    coll_inc_sum=0.0
    region_res_sum={r:0.0 for r in region_names}
    region_inc_sum={r:0.0 for r in region_names}
    region_count={r:0 for r in region_names}

    region_time_rows=[]
    target_inc_t=[]
    target_res_t=[]
    target_times=[]
    target_point_res_energy=np.zeros(np.count_nonzero(target_mask),dtype=np.float64)
    target_point_inc_energy=np.zeros_like(target_point_res_energy)
    target_time_res_energy=[]

    completed=0
    print(
        f"[Step 4C3 V2A] replaying {len(times)} nodes; "
        f"target={TARGET_REGION}, diagnostic samples={len(diag_analysis)}",
        flush=True,
    )

    for step,t_s in enumerate(times):
        plasma=np.asarray(plasma_coll[step],dtype=np.float64)
        incident=plasma+c3.feed_at_time(g.solve_obs_points,float(t_s))
        e=c3.tangent_coeffs(incident,g.solve_tangent1,g.solve_tangent2)

        history=Hp@p_prev+Hk@q_prev
        q=c3.solve_fixed_ridge(prep,-(e+history))
        p=p_prev+0.5*c3.DT*(q_prev+q)

        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            raise RuntimeError(f"non-finite replay state at step {step}")

        residual_coeff=e+history+c3.operator_now_coeff(prep,q)
        if t_s>=analysis_start_s:
            coll_res_sum+=float(np.dot(residual_coeff,residual_coeff))
            coll_inc_sum+=float(np.dot(e,e))

        if step in diag_lookup and t_s>=analysis_start_s:
            hi=diag_lookup[step]
            incident_H=(
                np.asarray(plasma_holdout[hi],dtype=np.float64)
                + c3.feed_at_time(Hpoints,float(t_s))
            )
            self_H,_=c3.evaluate_retarded_mot_field(
                Hpoints,g,p_prev,q_prev,q
            )
            total_H=incident_H+self_H
            res_tan=tangential_vectors_real(total_H,Hnormals)
            inc_tan=tangential_vectors_real(incident_H,Hnormals)
            res_sq=np.sum(res_tan*res_tan,axis=1)
            inc_sq=np.sum(inc_tan*inc_tan,axis=1)

            # Instantaneous global incident RMS for descriptive per-time ratios.
            gi=region_names.index("global")
            gm=Hmask[gi]
            global_inc_inst=math.sqrt(float(np.mean(inc_sq[gm])))

            for ir,region in enumerate(region_names):
                mask=Hmask[ir]
                if not np.any(mask):
                    continue
                re=float(np.sum(res_sq[mask]))
                ie=float(np.sum(inc_sq[mask]))
                cnt=int(np.count_nonzero(mask))
                region_res_sum[region]+=re
                region_inc_sum[region]+=ie
                region_count[region]+=cnt
                rr=math.sqrt(re/cnt)
                ii=math.sqrt(ie/cnt)
                incident_ratio=ii/max(global_inc_inst,c3.TINY)
                local_rel=rr/max(ii,c3.TINY)
                global_norm=rr/max(global_inc_inst,c3.TINY)
                metric=(
                    "local_rel"
                    if region=="global" or incident_ratio>=c3.WEAK_REGION_THRESHOLD
                    else "global_norm"
                )
                value=local_rel if metric=="local_rel" else global_norm
                if region==TARGET_REGION:
                    region_time_rows.append({
                        "step":int(step),
                        "time_s":float(t_s),
                        "rf_cycles":float(t_s*c3.F_DRIVE),
                        "residual_energy":re,
                        "incident_energy":ie,
                        "residual_rms":rr,
                        "incident_rms":ii,
                        "incident_ratio_to_global":incident_ratio,
                        "local_rel":local_rel,
                        "global_norm":global_norm,
                        "instantaneous_gate_metric":metric,
                        "instantaneous_gate_value":value,
                    })

            rt=res_tan[target_mask]
            it=inc_tan[target_mask]
            target_res_t.append(rt.copy())
            target_inc_t.append(it.copy())
            target_times.append(float(t_s))
            target_point_res_energy+=np.sum(rt*rt,axis=1)
            target_point_inc_energy+=np.sum(it*it,axis=1)
            target_time_res_energy.append(float(np.sum(rt*rt)))

        q_prev,p_prev=q,p
        completed=step+1

        if step%max(1,int(settings["diag_stride"])*4)==0 or step==len(times)-1:
            print(f"[Step 4C3 V2A] step {step+1}/{len(times)}",flush=True)

    # Exact aggregate reproduction of C3 V2.
    coll_agg=math.sqrt(coll_res_sum/max(coll_inc_sum,c3.TINY))
    global_count=region_count["global"]
    global_inc_rms=math.sqrt(region_inc_sum["global"]/global_count)

    region_agg={}
    for region in region_names:
        cnt=region_count[region]
        if cnt<=0:
            continue
        rr=math.sqrt(region_res_sum[region]/cnt)
        ii=math.sqrt(region_inc_sum[region]/cnt)
        incident_ratio=ii/max(global_inc_rms,c3.TINY)
        local_rel=rr/max(ii,c3.TINY)
        global_norm=rr/max(global_inc_rms,c3.TINY)
        metric=(
            "local_rel"
            if region=="global" or incident_ratio>=c3.WEAK_REGION_THRESHOLD
            else "global_norm"
        )
        value=local_rel if metric=="local_rel" else global_norm
        region_agg[region]={
            "residual_rms":rr,
            "incident_rms":ii,
            "incident_ratio":incident_ratio,
            "local_rel":local_rel,
            "global_norm":global_norm,
            "gate_metric":metric,
            "gate_value":value,
            "passed":bool(value<c3.BOUNDARY_GATE),
            "sample_count_accumulated":int(cnt),
        }

    # Reproducibility check against the already-reviewed C3 V2 report.
    ref_report=load_reference_report(args.reference_report)
    repro=None
    if ref_report is not None:
        ref_coll=float(ref_report["time_aggregated_collocation_rel_l2"])
        ref_target=float(
            ref_report["independent_holdout"]["regions"][TARGET_REGION]["gate_value"]
        )
        dc=abs(coll_agg-ref_coll)
        dh=abs(region_agg[TARGET_REGION]["gate_value"]-ref_target)
        repro={
            "reference_collocation":ref_coll,
            "diagnostic_collocation":coll_agg,
            "absolute_difference_collocation":dc,
            "reference_target_gate":ref_target,
            "diagnostic_target_gate":region_agg[TARGET_REGION]["gate_value"],
            "absolute_difference_target_gate":dh,
            "tolerance":REPORT_REPRO_TOL,
            "passed":bool(dc<=REPORT_REPRO_TOL and dh<=REPORT_REPRO_TOL),
        }
        if not repro["passed"]:
            raise RuntimeError(
                "V2A did not reproduce the reviewed V2 aggregate result; "
                f"coll_diff={dc}, target_diff={dh}"
            )

    # Temporal localization: rank by residual ENERGY, not by instantaneous ratio.
    target_total_energy=float(np.sum(target_time_res_energy))
    running=0.0
    ranked=sorted(region_time_rows,key=lambda r:r["residual_energy"],reverse=True)
    top_time=[]
    for row in ranked[:12]:
        running+=row["residual_energy"]
        q=dict(row)
        q["fraction_of_target_residual_energy"]=float(
            row["residual_energy"]/max(target_total_energy,c3.TINY)
        )
        q["cumulative_fraction"]=float(
            running/max(target_total_energy,c3.TINY)
        )
        top_time.append(q)

    temporal_concentration=concentration_fractions(target_time_res_energy)

    # Spatial localization inside feed_transition.
    nt=max(len(target_times),1)
    point_res_rms=np.sqrt(target_point_res_energy/nt)
    point_inc_rms=np.sqrt(target_point_inc_energy/nt)
    point_rel=point_res_rms/np.maximum(point_inc_rms,c3.TINY)
    target_indices=np.flatnonzero(target_mask)
    target_points=Hpoints[target_mask]
    order=np.argsort(target_point_res_energy)[::-1]
    total_point_energy=float(np.sum(target_point_res_energy))
    point_rows=[]
    running=0.0
    for j in order[:20]:
        running+=float(target_point_res_energy[j])
        point_rows.append({
            "holdout_index":int(target_indices[j]),
            "point_m":[float(x) for x in target_points[j]],
            "residual_rms":float(point_res_rms[j]),
            "incident_rms":float(point_inc_rms[j]),
            "local_rel":float(point_rel[j]),
            "fraction_of_target_residual_energy":float(
                target_point_res_energy[j]/max(total_point_energy,c3.TINY)
            ),
            "cumulative_fraction":float(
                running/max(total_point_energy,c3.TINY)
            ),
        })

    # Joint 3/6/9 MHz fit on the actual recorded target-region time series.
    target_times_arr=np.asarray(target_times,dtype=np.float64)
    target_inc_arr=np.asarray(target_inc_t,dtype=np.float64)
    target_res_arr=np.asarray(target_res_t,dtype=np.float64)
    fit_i=fit_multitone(target_times_arr,target_inc_arr)
    fit_r=fit_multitone(target_times_arr,target_res_arr)

    spectral={
        "num_samples":int(len(target_times_arr)),
        "sample_dt_nominal_s":float(np.median(np.diff(target_times_arr))),
        "design_condition_number":float(max(
            fit_i["condition_number"],fit_r["condition_number"]
        )),
        "fit_model":"DC + linear trend + simultaneous 3/6/9 MHz cos/sin",
        "total_incident_rms":vector_rms_time_space(target_inc_arr),
        "total_residual_rms":vector_rms_time_space(target_res_arr),
        "joint_fit_residual_rms_after_DC_trend_3_6_9":vector_rms_time_space(
            fit_r["remainder"]
        ),
        "joint_fit_incident_rms_after_DC_trend_3_6_9":vector_rms_time_space(
            fit_i["remainder"]
        ),
        "harmonics":{},
    }
    for f in FREQS:
        Ih=fit_i["phasors"][f]
        Rh=fit_r["phasors"][f]
        ir=float(np.sqrt(np.mean(np.sum(np.abs(Ih)**2,axis=-1))))
        rr=float(np.sqrt(np.mean(np.sum(np.abs(Rh)**2,axis=-1))))
        spectral["harmonics"][f"{f/1e6:.0f}MHz"]={
            "incident_phasor_rms":ir,
            "residual_phasor_rms":rr,
            "residual_to_incident_phasor_ratio":rr/max(ir,c3.TINY),
        }

    # ------------------------------------------------------------------
    # Actual-plasma-spatial-harmonic audit.
    # Fit total incident field at full collocation sampling and at the recorded
    # holdout diagnostic sampling, then solve a CONTINUOUS causal harmonic MAS.
    # ------------------------------------------------------------------
    print("[Step 4C3 V2A] fitting actual recorded 3/6/9 MHz incident phasors ...",flush=True)
    ta=times[analysis_idx]
    # Build total incident time series in a memory-conscious array.
    coll_total=np.empty(
        (len(analysis_idx),g.num_collocation_points,3),dtype=np.float64
    )
    for jj,step in enumerate(analysis_idx):
        coll_total[jj]=(
            np.asarray(plasma_coll[step],dtype=np.float64)
            + c3.feed_at_time(g.solve_obs_points,float(times[step]))
        )
    fit_coll=fit_multitone(ta,coll_total)
    del coll_total
    gc.collect()

    hold_total=np.empty(
        (len(diag_analysis),len(Hpoints),3),dtype=np.float64
    )
    hold_times=np.empty(len(diag_analysis),dtype=np.float64)
    for jj,step in enumerate(diag_analysis):
        hi=diag_lookup[step]
        hold_times[jj]=times[step]
        hold_total[jj]=(
            np.asarray(plasma_holdout[hi],dtype=np.float64)
            + c3.feed_at_time(Hpoints,float(times[step]))
        )
    fit_hold=fit_multitone(hold_times,hold_total)
    del hold_total
    gc.collect()

    mask_dict={region:Hmask[ir] for ir,region in enumerate(region_names)}
    harmonic_audit={}

    # A relative PEC residual is not meaningful for an essentially unexcited
    # harmonic.  Determine which fitted harmonics have enough incident-field
    # strength to support a spatial pass/fail statement.
    coll_phasor_rms={}
    for f in FREQS:
        Eh=np.asarray(fit_coll["phasors"][f],dtype=np.complex128)
        coll_phasor_rms[f]=float(
            np.sqrt(np.mean(np.sum(np.abs(Eh)**2,axis=-1)))
        )
    max_coll_phasor=max(coll_phasor_rms.values())
    fit_condition=max(
        float(fit_coll["condition_number"]),
        float(fit_hold["condition_number"]),
    )
    fit_reliable=bool(fit_condition<FIT_CONDITION_LIMIT)

    # Hp/Hk/prep are no longer needed after time replay; release the solve state
    # before allocating the complex harmonic operator.
    del Hp,Hk,prep
    gc.collect()

    valid_results=[]
    for f in FREQS:
        strength_ratio=coll_phasor_rms[f]/max(max_coll_phasor,c3.TINY)
        assessable=bool(
            fit_reliable
            and strength_ratio>=MIN_HARMONIC_STRENGTH_RATIO
        )

        if not assessable:
            harmonic_audit[f"{f/1e6:.0f}MHz"]={
                "diagnostic_valid":False,
                "skip_reason":(
                    "joint harmonic fit ill-conditioned"
                    if not fit_reliable else
                    "incident harmonic too weak for a stable relative PEC diagnostic"
                ),
                "collocation_incident_phasor_rms":coll_phasor_rms[f],
                "incident_strength_ratio_to_max_harmonic":strength_ratio,
                "minimum_strength_ratio":MIN_HARMONIC_STRENGTH_RATIO,
                "fit_condition_number":fit_condition,
            }
            continue

        print(f"[Step 4C3 V2A] actual spatial harmonic audit {f/1e6:.0f} MHz ...",flush=True)
        omega=2.0*np.pi*f
        Ecoll=np.asarray(fit_coll["phasors"][f],dtype=np.complex128)
        Ehold=np.asarray(fit_hold["phasors"][f],dtype=np.complex128)
        e=c2.incident_coeff(Ecoll,g)

        Aref,ret_diag=c2.assemble_retarded_harmonic(g,omega,obs_chunk)
        refprep=c2.prepare_complex(Aref,g.solve_weights)
        qref=c2.solve_complex(refprep,-e)

        coll=float(
            np.linalg.norm(Aref@qref+e)
            / max(float(np.linalg.norm(e)),c3.TINY)
        )
        Esc=c2.eval_retarded_field(Hpoints,g,qref,omega)
        rows=complex_region_metrics(
            Ehold+Esc,Ehold,Hnormals,Hmask,region_names
        )
        target=rows[TARGET_REGION]
        worst=max(rows.values(),key=lambda r:r["gate_value"])
        passed=bool(
            coll<c3.BOUNDARY_GATE
            and target["gate_value"]<c3.BOUNDARY_GATE
        )

        harmonic_audit[f"{f/1e6:.0f}MHz"]={
            "diagnostic_valid":True,
            "collocation_incident_phasor_rms":coll_phasor_rms[f],
            "incident_strength_ratio_to_max_harmonic":strength_ratio,
            "fit_condition_number":fit_condition,
            "continuous_retarded_collocation_rel_l2":coll,
            "feed_transition":target,
            "worst_region":worst,
            "all_regions":rows,
            "retardation_diagnostic":ret_diag,
            "collocation_passed":bool(coll<c3.BOUNDARY_GATE),
            "feed_transition_passed":bool(
                target["gate_value"]<c3.BOUNDARY_GATE
            ),
            "passed":passed,
        }
        valid_results.append(passed)
        del Aref,refprep,qref,Esc
        gc.collect()

    assessable_count=len(valid_results)
    all_assessable_pass=bool(
        assessable_count>0 and all(valid_results)
    )

    # Diagnostic conclusion is rule-based and does not alter any gate.
    if not fit_reliable:
        diagnosis=(
            "HOLDOUT_MULTITONE_FIT_NOT_RELIABLE; "
            "do_not_infer_spatial_resolution_from_harmonic_audit"
        )
        next_test=(
            "Increase diagnostic sampling density only; do not change the "
            "physical solver or C3 gate."
        )
    elif assessable_count==0:
        diagnosis=(
            "NO_HARMONIC_HAS_SUFFICIENT_INCIDENT_STRENGTH_FOR_SPATIAL_AUDIT; "
            "do_not_infer_spatial_under_resolution"
        )
        next_test=(
            "Use time-localization/broadband closure diagnostics; do not "
            "density-retune from unexcited harmonics."
        )
    elif all_assessable_pass:
        diagnosis=(
            "ALL_ASSESSABLE_ACTUAL_HARMONIC_SPATIAL_PATTERNS_PASS; "
            "do_not_density_retune_yet; investigate broadband/transient/"
            "surface-charge-continuity closure next"
        )
        next_test="Step 4C3 V2B — surface-charge / continuity closure diagnostic"
    else:
        diagnosis=(
            "AT_LEAST_ONE_ASSESSABLE_ACTUAL_HARMONIC_FEED_TRANSITION_FAILS; "
            "spatial under-resolution/generalization is supported"
        )
        next_test=(
            "Step 4C3 V2B — controlled source/collocation-density refinement "
            "using the same recorded forcing/harmonic audit"
        )

    report={
        "step":"4C3-V2A",
        "purpose":"feed-transition residual localization diagnostic",
        "mode":settings["mode"],
        "forcing_file":str(forcing_path),
        "no_PIC_rerun":True,
        "no_physics_parameter_changed":True,
        "geometry_fingerprint_match":bool(fp==meta["geometry_fingerprint"]),
        "analysis_start_s":analysis_start_s,
        "aggregate_reproduction":{
            "collocation_rel_l2":coll_agg,
            "regions":region_agg,
            "reference_report_check":repro,
        },
        "target_region":TARGET_REGION,
        "target_aggregate":region_agg[TARGET_REGION],
        "temporal_localization":{
            "num_diagnostic_times":len(region_time_rows),
            "top_residual_energy_times":top_time,
            "concentration":temporal_concentration,
        },
        "spatial_localization":{
            "num_target_points":int(np.count_nonzero(target_mask)),
            "top_residual_energy_points":point_rows,
        },
        "target_region_multitone_fit":spectral,
        "actual_plasma_spatial_harmonic_audit":{
            "collocation_fit_condition_number":fit_coll["condition_number"],
            "holdout_fit_condition_number":fit_hold["condition_number"],
            "frequencies":harmonic_audit,
            "fit_condition_limit":FIT_CONDITION_LIMIT,
            "minimum_harmonic_strength_ratio":MIN_HARMONIC_STRENGTH_RATIO,
            "fit_reliable":fit_reliable,
            "assessable_harmonic_count":assessable_count,
            "all_assessable_harmonics_pass":all_assessable_pass,
        },
        "diagnosis":diagnosis,
        "recommended_next_test":next_test,
        "scope_note":(
            "V2A is diagnostic only. It does not relax the 5% C3 gate and "
            "does not make response amplitudes manuscript-final."
        ),
    }
    (out/"step4c3_v2a_report.json").write_text(
        json.dumps(report,indent=2),encoding="utf-8"
    )

    # Save compact machine-readable time/point diagnostics.
    np.savez_compressed(
        out/"step4c3_v2a_diagnostics.npz",
        target_times_s=target_times_arr,
        target_incident_tangent=target_inc_arr.astype(np.float32),
        target_residual_tangent=target_res_arr.astype(np.float32),
        target_holdout_indices=target_indices.astype(np.int64),
        target_holdout_points_m=target_points.astype(np.float64),
        target_point_residual_rms=point_res_rms,
        target_point_incident_rms=point_inc_rms,
        target_point_local_rel=point_rel,
    )

    lines=[
        "Step 4C3 V2A feed-transition diagnostic",
        f"mode: {settings['mode']}",
        f"reproduced collocation: {coll_agg:.12g}",
        f"reproduced feed_transition gate: "
        f"{region_agg[TARGET_REGION]['gate_value']:.12g}",
        f"diagnostic times: {len(region_time_rows)}",
        f"multitone fit condition: {spectral['design_condition_number']:.6g}",
    ]
    for f in FREQS:
        a=harmonic_audit[f"{f/1e6:.0f}MHz"]
        if a.get("diagnostic_valid"):
            lines.append(
                f"{f/1e6:.0f} MHz actual-spatial harmonic: "
                f"strength={a['incident_strength_ratio_to_max_harmonic']:.6g}, "
                f"coll={a['continuous_retarded_collocation_rel_l2']:.6g}, "
                f"feed_transition={a['feed_transition']['gate_value']:.6g}, "
                f"pass={a['feed_transition_passed']}"
            )
        else:
            lines.append(
                f"{f/1e6:.0f} MHz actual-spatial harmonic: "
                f"NOT ASSESSABLE; "
                f"strength={a['incident_strength_ratio_to_max_harmonic']:.6g}; "
                f"reason={a['skip_reason']}"
            )
    lines += [
        f"diagnosis: {diagnosis}",
        f"recommended next test: {next_test}",
    ]
    (out/"step4c3_v2a_summary.txt").write_text(
        "\n".join(lines)+"\n",encoding="utf-8"
    )
    print("\n".join(lines))
    return report


def self_test():
    # Backend algebra tests.
    print("[V2A self-test] C3 V2 Retarded-MOT backend")
    c3.self_test()
    print("[V2A self-test] C2 V4 harmonic backend")
    c2.self_test()

    # Multitone recovery test at the same sparse cadence class as formal
    # holdout diagnostics.
    t=np.arange(0,22,dtype=float)*20.0e-9+260.0e-9
    rng=np.random.default_rng(20261002)
    shape=(7,3)
    expected={}
    y=np.zeros((len(t),)+shape,dtype=np.float64)
    dc=rng.normal(scale=.2,size=shape)
    trend=rng.normal(scale=.05,size=shape)
    tn=(t-t.mean())/np.ptp(t)
    y+=dc[None,...]+tn[:,None,None]*trend[None,...]
    for f in FREQS:
        ph=rng.normal(size=shape)+1j*rng.normal(size=shape)
        expected[f]=ph
        y+=np.real(ph[None,...]*np.exp(1j*2*np.pi*f*t)[:,None,None])
    fit=fit_multitone(t,y)
    errs=[]
    for f in FREQS:
        errs.append(
            float(np.linalg.norm(fit["phasors"][f]-expected[f])
                  /max(float(np.linalg.norm(expected[f])),c3.TINY))
        )
    passed=bool(max(errs)<1e-11 and fit["condition_number"]<100)
    rep={
        "self_test_pass":passed,
        "multitone_relative_errors":errs,
        "multitone_design_condition_number":fit["condition_number"],
    }
    print(json.dumps(rep,indent=2))
    if not passed:
        raise RuntimeError("V2A multitone self-test failed")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--forcing",default=None)
    p.add_argument("--reference-report",default=None)
    p.add_argument("--out",default="step4c3_v2a_formal")
    a=p.parse_args()
    if a.self_test:
        self_test()
        return
    if a.forcing is None:
        raise RuntimeError("--forcing is required")
    run_diagnostic(a)


if __name__=="__main__":
    main()
