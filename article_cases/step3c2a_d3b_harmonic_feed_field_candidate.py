#!/usr/bin/env python3
"""
Step 3-C2A-D3B — harmonic feed-field MAS formulation candidate.

Purpose
-------
D1/D2 identified two structural defects in the current Case-3 time-domain
formulation:
1) the prescribed active surface-current mode is almost completely cancelled
   by the passive PEC solve;
2) the free electrostatic/polarization G_p history recurrence is unstable.

D3B tests a replacement formulation suitable for the paper's monochromatic
3 MHz RF problem.  It does NOT modify production code.

For an e^(+i omega t) convention,

    dP/dt = K  ->  P = K/(i omega),
    dK/dt      ->  i omega K.

Therefore the complete electric-dipole operator is

    G_omega = G_p/(i omega) + G_k + i omega G_kdot.

All 1/R^3, 1/R^2 and 1/R terms are retained.  The unstable free P-history
recurrence is not used.

The excitation is an incident/feed field, not a prescribed all-surface current
mode.  This is consistent with the already validated Step-2 MB1 incident-field
tests and avoids active/passive self-cancellation.

Acceptance
----------
Use the same frozen smooth rounded geometry, d/R=0.80, independent 2x
collocation, weighted/column-scaled ridge lambda_rel=1e-4, and the same 5%
independent holdout PEC gates used in A1/MB1.

Formal acceptance incidents:
- uniform
- off-axis external dipole
- localized feed field

The axial antisymmetric stress remains non-gating.

A D3B pass validates only the *harmonic boundary formulation candidate*.
It does not yet validate a full PIC-coupled manuscript Case-3 rerun.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

import step2kd1_coherent_closed_auxiliary_surface as kd1
import step2mb1_incident_normalization_feed_localization as mb1
from EMsolver.mas_preprocessor import (
    generate_rounded_two_arm_dipole_points,
    build_local_tangent_frames,
)

TINY = kd1.TINY
EPS0 = kd1.EPS0
C0 = kd1.C0


def complex_smax_power(A, niter=8):
    rng = np.random.default_rng(1234)
    x = rng.normal(size=A.shape[1]).astype(np.complex128)
    x /= max(float(np.linalg.norm(x)), TINY)
    for _ in range(int(niter)):
        y = A @ x
        z = A.conj().T @ y
        nz = float(np.linalg.norm(z))
        if nz < TINY:
            return 1.0
        x = z / nz
    return float(np.linalg.norm(A @ x))


def harmonic_field_block(obs, src, weight, omega):
    r = np.asarray(obs, float) - np.asarray(src, float)[None, :]
    R = np.maximum(np.linalg.norm(r, axis=1), 1e-15)
    rr = r[:, :, None] * r[:, None, :]
    I = np.eye(3)[None, :, :]
    ce = 1.0 / (4.0 * np.pi * EPS0)

    Gp = ce * float(weight) * (
        3.0 * rr / R[:, None, None]**5
        - I / R[:, None, None]**3
    )
    Gk = ce * float(weight) * (
        3.0 * rr / R[:, None, None]**4
        - I / R[:, None, None]**2
    ) / C0
    Gd = ce * float(weight) * (
        rr / R[:, None, None]**3
        - I / R[:, None, None]
    ) / (C0*C0)

    return Gp/(1j*omega) + Gk + 1j*omega*Gd


def assemble_harmonic_tangent_operator(g, omega):
    OP = np.asarray(g.solve_obs_points, float)
    ON = np.asarray(g.solve_normals, float)
    SP = np.asarray(g.ant_src_points, float)
    SW = np.asarray(g.source_weights, float)

    ot1, ot2 = build_local_tangent_frames(ON)
    st1 = np.asarray(g.source_tangent1, float)
    st2 = np.asarray(g.source_tangent2, float)

    A = np.empty((2*len(OP), 2*len(SP)), dtype=np.complex128)

    for j in range(len(SP)):
        B = harmonic_field_block(OP, SP[j], SW[j], omega)
        S = np.column_stack((st1[j], st2[j]))
        T = np.einsum("mab,bc->mac", B, S)
        A[0::2, 2*j:2*j+2] = np.einsum("ma,mac->mc", ot1, T)
        A[1::2, 2*j:2*j+2] = np.einsum("ma,mac->mc", ot2, T)

    return A


def tangent_rhs(E, N):
    E = np.asarray(E, dtype=np.complex128).reshape(-1, 3)
    t1, t2 = build_local_tangent_frames(N)
    b = np.empty(2*len(N), dtype=np.complex128)
    b[0::2] = -np.sum(E*t1, axis=1)
    b[1::2] = -np.sum(E*t2, axis=1)
    return b


def prepare_solver(A, point_weights, lambda_rel):
    rs = np.repeat(
        np.sqrt(np.asarray(point_weights, float) /
                max(float(np.mean(point_weights)), TINY)),
        2,
    )
    Aw = A * rs[:, None]
    cs = np.linalg.norm(Aw, axis=0)
    cs = np.where(cs > TINY, cs, 1.0)
    As = Aw / cs[None, :]

    smax = max(complex_smax_power(As, 8), TINY)
    alpha = float(lambda_rel) * smax

    H = As.conj().T @ As
    H.flat[::H.shape[0]+1] += alpha*alpha
    cf = cho_factor(H, lower=True, overwrite_a=True, check_finite=False)

    return {
        "rs": rs,
        "cs": cs,
        "As": As,
        "smax": smax,
        "alpha": alpha,
        "cf": cf,
    }


def solve_prepared(A, prepared, b):
    rs = prepared["rs"]
    cs = prepared["cs"]
    As = prepared["As"]
    bw = b * rs
    y = cho_solve(
        prepared["cf"],
        As.conj().T @ bw,
        check_finite=False,
    )
    x = y / cs

    raw = A @ x - b
    weighted = As @ y - bw
    return x, {
        "collocation_rel_l2":
            float(np.linalg.norm(raw)/max(np.linalg.norm(b), TINY)),
        "weighted_residual_rel":
            float(np.linalg.norm(weighted)/max(np.linalg.norm(bw), TINY)),
    }


def coeff_to_current(g, x):
    q = np.asarray(x, dtype=np.complex128).reshape(-1, 2)
    return (
        q[:, 0, None] * g.source_tangent1
        + q[:, 1, None] * g.source_tangent2
    )


def eval_harmonic_field(points, g, x, omega, obs_chunk=256, src_chunk=64):
    K = coeff_to_current(g, x)
    P = np.asarray(points, float)
    SP = np.asarray(g.ant_src_points, float)
    SW = np.asarray(g.source_weights, float)

    E = np.zeros((len(P), 3), dtype=np.complex128)
    ce = 1.0/(4.0*np.pi*EPS0)

    for oa in range(0, len(P), obs_chunk):
        ob = min(len(P), oa+obs_chunk)
        X = P[oa:ob]
        Eb = np.zeros((len(X), 3), dtype=np.complex128)

        for sa in range(0, len(SP), src_chunk):
            sb = min(len(SP), sa+src_chunk)
            r = X[:, None, :] - SP[None, sa:sb, :]
            R = np.maximum(np.linalg.norm(r, axis=2), 1e-15)
            Ks = K[sa:sb]
            rk = np.sum(r*Ks[None, :, :], axis=2)
            w = SW[None, sa:sb, None]

            gp = (
                3.0*r*rk[:, :, None]/R[:, :, None]**5
                - Ks[None, :, :]/R[:, :, None]**3
            )
            gk = (
                3.0*r*rk[:, :, None]/R[:, :, None]**4
                - Ks[None, :, :]/R[:, :, None]**2
            )
            gd = (
                r*rk[:, :, None]/R[:, :, None]**3
                - Ks[None, :, :]/R[:, :, None]
            )

            contrib = ce*w*(
                gp/(1j*omega)
                + gk/C0
                + 1j*omega*gd/(C0*C0)
            )
            Eb += np.sum(contrib, axis=1)

        E[oa:ob] = Eb

    return E


def tangential_rms(E, N, mask):
    E = np.asarray(E, dtype=np.complex128)
    N = np.asarray(N, float)
    Et = E - N*np.sum(E*N, axis=1)[:, None]
    vals = np.sum(np.abs(Et[mask])**2, axis=1)
    return float(np.sqrt(np.mean(vals))) if len(vals) else 0.0


def region_metrics(total, inc, N, masks):
    global_mask = np.ones(len(N), dtype=bool)
    inc_global = tangential_rms(inc, N, global_mask)
    out = {}

    for name, mask in masks.items():
        if not np.any(mask):
            continue
        rr = tangential_rms(total, N, mask)
        ii = tangential_rms(inc, N, mask)
        out[name] = {
            "residual_rms": rr,
            "incident_rms": ii,
            "local_rel": rr/max(ii, TINY),
            "global_norm": rr/max(inc_global, TINY),
            "incident_ratio": ii/max(inc_global, TINY),
        }
    return out


def gate_metric(region, m, weak_threshold):
    if region == "global" or m["incident_ratio"] >= weak_threshold:
        return "local_rel", m["local_rel"]
    return "global_norm", m["global_norm"]


def make_holdouts(R, total, gap, Larm, ncyl, nphi, nhemi):
    z_bottom = -0.5*total
    z_top = 0.5*gap

    T0, C0, N0, A0, T0loc = mb1.make_arm(
        R, R, Larm, z_bottom, ncyl, nphi, nhemi
    )
    T1, C1, N1, A1, T1loc = mb1.make_arm(
        R, R, Larm, z_top, ncyl, nphi, nhemi
    )

    H00, HN00 = mb1.bary_arm(
        T0loc, [0.19, 0.37, 0.44], R, Larm, z_bottom
    )
    H01, HN01 = mb1.bary_arm(
        T1loc, [0.19, 0.37, 0.44], R, Larm, z_top
    )
    HP = np.concatenate((H00, H01))
    HN = np.concatenate((HN00, HN01))
    aid = np.concatenate((
        np.zeros(len(H00), int),
        np.ones(len(H01), int),
    ))
    masks = mb1.classify(
        HP, aid, R, Larm, z_bottom, z_top
    )

    H20, HN20 = mb1.bary_arm(
        T0loc, [0.53, 0.29, 0.18], R, Larm, z_bottom
    )
    H21, HN21 = mb1.bary_arm(
        T1loc, [0.53, 0.29, 0.18], R, Larm, z_top
    )
    HP2 = np.concatenate((H20, H21))
    HN2 = np.concatenate((HN20, HN21))
    aid2 = np.concatenate((
        np.zeros(len(H20), int),
        np.ones(len(H21), int),
    ))
    masks2 = mb1.classify(
        HP2, aid2, R, Larm, z_bottom, z_top
    )

    return (
        (HP, HN, masks),
        (HP2, HN2, masks2),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="step3c2a_d3b_harmonic_feed_field")
    p.add_argument("--frequency-hz", type=float, default=3.0e6)
    p.add_argument("--radius", type=float, default=1.0e-3)
    p.add_argument("--total-length", type=float, default=60.0e-3)
    p.add_argument("--feed-gap", type=float, default=2.0e-3)
    p.add_argument("--depth-frac", type=float, default=0.80)
    p.add_argument("--sources-per-arm", type=int, default=900)
    p.add_argument("--collocation-factor", type=float, default=2.0)
    p.add_argument("--lambda-rel", type=float, default=1.0e-4)
    p.add_argument("--holdout-gate", type=float, default=0.05)
    p.add_argument("--collocation-gate", type=float, default=0.05)
    p.add_argument("--weak-region-threshold", type=float, default=0.20)
    p.add_argument("--feed-width-over-gap", type=float, default=1.0)
    a = p.parse_args()

    out = Path("article_cases_output")/a.out
    out.mkdir(parents=True, exist_ok=True)

    R = float(a.radius)
    total = 16.0e-3 if a.quick else float(a.total_length)
    gap = float(a.feed_gap)
    Larm = 0.5*(total-gap)

    if a.quick:
        nphi, ncyl, nhemi, nsrc = 28, 28, 10, min(180, int(a.sources_per_arm))
    else:
        nphi, ncyl, nhemi, nsrc = 72, 96, 24, int(a.sources_per_arm)

    omega = 2.0*np.pi*float(a.frequency_hz)

    g = generate_rounded_two_arm_dipole_points(
        radius=R,
        total_length=total,
        feed_gap=gap,
        depth_frac=a.depth_frac,
        sources_per_arm=nsrc,
        collocation_factor=a.collocation_factor,
        n_phi=nphi,
        n_cyl=ncyl,
        n_hemi=nhemi,
    )

    print("Step 3-C2A-D3B harmonic feed-field candidate")
    print(f"  mode                : {'QUICK' if a.quick else 'FORMAL'}")
    print(f"  f                   : {a.frequency_hz/1e6:.6f} MHz")
    print(f"  sources/arm         : {nsrc}")
    print(f"  collocation/arm     : {g.metadata['collocation_points_per_arm']}")
    print("  assembling complex harmonic operator ...", flush=True)

    A = assemble_harmonic_tangent_operator(g, omega)
    prepared = prepare_solver(A, g.solve_weights, a.lambda_rel)

    holdouts = make_holdouts(
        R, total, gap, Larm, ncyl, nphi, nhemi
    )

    incidents = {
        "uniform": lambda P: kd1.inc_uniform(P),
        "external_dipole": lambda P: mb1.inc_external(P, total, R),
        "feed_localized_sech2":
            lambda P: mb1.inc_feed_localized(
                P, gap, a.feed_width_over_gap
            ),
        "axial_antisymmetric_stress":
            lambda P: mb1.inc_axial_stress(P, gap),
    }
    acceptance_incidents = (
        "uniform",
        "external_dipole",
        "feed_localized_sech2",
    )

    gate_regions = (
        "global",
        "feed_rounded",
        "feed_tips",
        "feed_transition",
        "outer_rounded",
        "outer_tips",
        "all_transition",
        "side_mid",
    )

    incident_results = {}
    gate_rows = []

    for name, fn in incidents.items():
        inc_coll = fn(g.solve_obs_points).astype(np.complex128)
        b = tangent_rhs(inc_coll, g.solve_normals)
        x, diag = solve_prepared(A, prepared, b)

        item = {
            "acceptance_incident": name in acceptance_incidents,
            "collocation": diag,
            "holdouts": {},
            "coefficient_l2": float(np.linalg.norm(x)),
        }

        if name in acceptance_incidents:
            gate_rows.append({
                "incident": name,
                "pattern": "collocation",
                "region": "global",
                "gate_metric": "collocation_rel_l2",
                "gate_value": diag["collocation_rel_l2"],
                "gate": a.collocation_gate,
                "passed": bool(diag["collocation_rel_l2"] < a.collocation_gate),
            })

        for hname, (HP, HN, masks) in zip(
            ("holdout", "spot"), holdouts
        ):
            Escat = eval_harmonic_field(HP, g, x, omega)
            I = fn(HP).astype(np.complex128)
            mm = region_metrics(I+Escat, I, HN, masks)
            item["holdouts"][hname] = mm

            if name in acceptance_incidents:
                for region in gate_regions:
                    if region not in mm:
                        continue
                    metric_name, value = gate_metric(
                        region, mm[region], a.weak_region_threshold
                    )
                    gate_rows.append({
                        "incident": name,
                        "pattern": hname,
                        "region": region,
                        "incident_ratio": mm[region]["incident_ratio"],
                        "gate_metric": metric_name,
                        "gate_value": value,
                        "gate": a.holdout_gate,
                        "passed": bool(value < a.holdout_gate),
                    })

        incident_results[name] = item

    failed = [r for r in gate_rows if not r["passed"]]
    max_gate = max((float(r["gate_value"]) for r in gate_rows), default=None)

    worst = None
    if gate_rows:
        worst = max(gate_rows, key=lambda r: float(r["gate_value"]))

    accepted = len(failed) == 0

    report = {
        "step": "3-C2A-D3B",
        "purpose": "3 MHz harmonic feed-field formulation candidate after D1/D2 recurrence failure",
        "mode": "quick" if a.quick else "formal",
        "frequency_Hz": float(a.frequency_hz),
        "phasor_convention": "exp(+i omega t)",
        "harmonic_operator":
            "G_omega = G_p/(i omega) + G_k + i omega G_kdot",
        "retained_physics":
            "all electric-dipole 1/R^3 + 1/R^2 + 1/R terms",
        "free_P_history_recurrence_used": False,
        "prescribed_all_surface_active_current_used": False,
        "drive_interpretation":
            "incident/feed-field phasor; not calibrated port current",
        "production_geometry": {
            "radius_m": R,
            "total_length_m": total,
            "feed_gap_m": gap,
            "depth_frac": float(a.depth_frac),
            "sources_per_arm": nsrc,
            "collocation_points_per_arm":
                int(g.metadata["collocation_points_per_arm"]),
            "collocation_factor": float(a.collocation_factor),
            "lambda_rel": float(a.lambda_rel),
        },
        "solver": {
            "weighted_scaled_complex_ridge": True,
            "smax_scaled_estimate": prepared["smax"],
            "ridge_alpha": prepared["alpha"],
        },
        "incidents": incident_results,
        "gate_rows": gate_rows,
        "failed_gate_count": len(failed),
        "max_formal_gate_value": max_gate,
        "worst_formal_gate": worst,
        "accepted": accepted,
        "decision": (
            "D3B_PASS_harmonic_feed_field_candidate_proceed_to_narrowband_Case3_integration_design"
            if accepted else
            "D3B_FAIL_harmonic_candidate_not_boundary_accurate_enough"
        ),
        "scope_note": (
            "A D3B pass validates boundary accuracy of the harmonic formulation "
            "only. It does not by itself validate the old 154x/289x/17x Case-3 "
            "numbers or a full PIC feedback loop."
        ),
    }

    (out/"step3c2a_d3b_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    lines = [
        "Step 3-C2A-D3B harmonic feed-field candidate",
        f"accepted: {accepted}",
        f"mode: {report['mode']}",
        f"frequency: {a.frequency_hz/1e6:.6f} MHz",
        f"sources_per_arm: {nsrc}",
        f"failed_gate_count: {len(failed)}",
        f"max_formal_gate_value: {max_gate}",
        f"worst_formal_gate: {worst}",
        f"decision: {report['decision']}",
    ]
    (out/"step3c2a_d3b_summary.txt").write_text(
        "\n".join(lines)+"\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
