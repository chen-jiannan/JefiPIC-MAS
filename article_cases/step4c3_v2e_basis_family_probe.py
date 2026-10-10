#!/usr/bin/env python3
"""
Step 4C3 V2E — fixed-observation auxiliary electric-dipole basis-family probe.

Scientific question
-------------------
V2C-A showed that simply increasing the number of tangent electric-dipole/current
sources did not qualify the 9 MHz boundary response. V2D showed that two
predeclared geometry-only placement changes at fixed N=900 did not qualify the
band either. V2E therefore changes one remaining representation variable only:
the orientation span of each auxiliary electric-dipole/current-moment source.

Frozen control
--------------
* original formal Step-4C3 forcing and original production source locations;
* 900 sources/arm, d/R=0.8, zero softening;
* original collocation points/normals/weights/tangent frames;
* original independent holdout points/normals/region masks;
* synchronized post-ramp 22-time multitone phasor estimator;
* 3/6/9 MHz, 5% gate, eta=0.20 weak-region rule;
* same retarded electric-dipole Green tensor Gp/(i*w)+Gk+i*w*Gd;
* same per-column scaling and lambda_rel=1e-4 ridge solve;
* same source patch weights and no residual-targeted tuning.

Basis families
--------------
1) tangent_electric_dipole (control)
   Two local source orientations per source: t1, t2. This is the existing
   production/V2B/V2C/V2D harmonic basis.

2) full_vector_electric_dipole (single predeclared augmentation)
   Three local source orientations per source: t1, t2, n. The same retarded
   electric-dipole tensor is used; no independent scalar-charge unknown is added.
   This probes whether the two-orientation restriction, rather than source count
   or source placement, limits the fixed-observation representation.

The added normal orientation is an AUXILIARY-SOURCE degree of freedom. It is not
claimed to be a literal physical normal PEC surface current. If this diagnostic
qualifies, production promotion requires an explicit 3-DOF auxiliary-source
implementation and a subsequent real broadband/long-tail confirmation before
Step 4D.

Stop rule
---------
This script intentionally does not sweep additional arbitrary basis families,
regularization strengths, source depths, seeds, or source counts. If the single
physics-preserving orientation augmentation does not qualify the frozen gates,
V2E is a stop-loss experiment: report the upper-band limitation rather than
continue post-hoc basis hunting.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
for _p in (str(HERE), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import step4c3_v2d_source_placement_localization_probe as v2d
except Exception as exc:
    raise RuntimeError(
        "V2E requires the completed V2D script in article_cases/: "
        "step4c3_v2d_source_placement_localization_probe.py"
    ) from exc

c3 = v2d.c3
c2 = v2d.c2
v2a = v2d.v2a
v2b = v2d.v2b

FREQS = (3.0e6, 6.0e6, 9.0e6)
SOURCE_COUNT_PER_ARM = 900
DEPTH_FRAC = 0.80
LAMBDA_REL = 1.0e-4
SOFTENING_M = 0.0
BOUNDARY_GATE = 0.05
PHASOR_SAMPLE_COUNT = 22
BASELINE_REPRO_TOL = 2.0e-8
OPERATOR_EVALUATOR_TOL = 5.0e-11
TREND_TOL = 1.0e-8
PAIRWISE_GUARD_M = 1.0e-15
TINY = 1.0e-300

BASELINE_BASIS = "tangent_electric_dipole"
AUGMENTED_BASIS = "full_vector_electric_dipole"
BASIS_FAMILIES = (BASELINE_BASIS, AUGMENTED_BASIS)


def parse_args():
    p = argparse.ArgumentParser(
        description="Step 4C3 V2E fixed-observation auxiliary electric-dipole basis-family probe"
    )
    p.add_argument(
        "--forcing",
        default="article_cases_output/step4c3_formal/step4c3_forcing.npz",
        help="Original formal Step-4C3 forcing NPZ",
    )
    p.add_argument(
        "--v2d-report",
        default="article_cases_output/step4c3_v2d_formal/step4c3_v2d_report.json",
        help="Completed V2D formal report",
    )
    p.add_argument(
        "--out",
        default="step4c3_v2e_formal",
        help="Output directory name under article_cases_output, or an absolute path",
    )
    p.add_argument("--operator-obs-chunk", type=int, default=8)
    p.add_argument("--eval-obs-chunk", type=int, default=128)
    p.add_argument("--eval-src-chunk", type=int, default=64)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def resolve_input_path(value):
    p = Path(value)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def resolve_out_path(value):
    p = Path(value)
    if p.is_absolute():
        return p
    if len(p.parts) > 1 or p.parts[0] == "article_cases_output":
        return (PROJECT_ROOT / p).resolve()
    return (PROJECT_ROOT / "article_cases_output" / p).resolve()


def freq_key(f):
    return str(int(round(float(f))))


def source_frames(g, family):
    t1 = np.asarray(g.source_tangent1, dtype=np.float64)
    t2 = np.asarray(g.source_tangent2, dtype=np.float64)
    if family == BASELINE_BASIS:
        return np.stack([t1, t2], axis=1)
    if family == AUGMENTED_BASIS:
        n = np.asarray(g.ant_normals, dtype=np.float64)
        nn = np.linalg.norm(n, axis=1)
        if np.any(nn <= 0.0):
            raise RuntimeError("zero source normal in full-vector basis")
        n = n / nn[:, None]
        return np.stack([t1, t2, n], axis=1)
    raise ValueError(f"unknown basis family: {family}")


def project_general(obs_frames, tensor, src_frames):
    """Project 3x3 source-to-observer tensor onto 2 obs tangents x D source orientations."""
    ot = np.asarray(obs_frames, dtype=np.float64)
    st = np.asarray(src_frames, dtype=np.float64)
    x = np.einsum("mua,mjab,jvb->mujv", ot, tensor, st, optimize=True)
    return x.reshape(2 * len(ot), st.shape[1] * len(st))


def assemble_full_vector_harmonic(g, omega, chunk):
    """Same retarded electric-dipole kernel as c2, but t1/t2/n source orientations."""
    obs = np.asarray(g.solve_obs_points, dtype=np.float64)
    src = np.asarray(g.ant_src_points, dtype=np.float64)
    sw = np.asarray(g.source_weights, dtype=np.float64)
    ot = np.stack([g.solve_tangent1, g.solve_tangent2], axis=1)
    st = source_frames(g, AUGMENTED_BASIS)
    A = np.empty((2 * len(obs), 3 * len(src)), dtype=np.complex128)
    diff2 = 0.0
    local2 = 0.0
    max_phase = 0.0
    rmin = math.inf
    rmax = 0.0
    for a in range(0, len(obs), int(chunk)):
        b = min(len(obs), a + int(chunk))
        Gp, Gk, Gd, R = c2.pair_blocks(obs[a:b], src, sw)
        Tlocal = Gp / (1j * omega) + Gk + 1j * omega * Gd
        phase = np.exp(-1j * omega * R / c2.C0)[:, :, None, None]
        Tret = phase * Tlocal
        Pret = project_general(ot[a:b], Tret, st)
        Plocal = project_general(ot[a:b], Tlocal, st)
        A[2 * a:2 * b] = Pret
        diff2 += float(np.sum(np.abs(Pret - Plocal) ** 2))
        local2 += float(np.sum(np.abs(Plocal) ** 2))
        max_phase = max(max_phase, float(np.max(omega * R / c2.C0)))
        rmin = min(rmin, float(np.min(R)))
        rmax = max(rmax, float(np.max(R)))
        del Gp, Gk, Gd, R, Tlocal, Tret, Pret, Plocal, phase
        gc.collect()
    return A, {
        "operator_retarded_minus_local_rel_fro": math.sqrt(diff2 / max(local2, TINY)),
        "max_pair_phase_delay_deg": max_phase * 180.0 / math.pi,
        "min_distance_m": rmin,
        "max_distance_m": rmax,
        "source_orientations_per_point": 3,
    }


def assemble_basis_harmonic(g, omega, family, chunk):
    if family == BASELINE_BASIS:
        A, diag = c2.assemble_retarded_harmonic(g, omega, int(chunk))
        diag = dict(diag)
        diag["source_orientations_per_point"] = 2
        return A, diag
    if family == AUGMENTED_BASIS:
        return assemble_full_vector_harmonic(g, omega, int(chunk))
    raise ValueError(f"unknown basis family: {family}")


def coefficients_to_vector(g, q, family):
    frames = source_frames(g, family)
    d = frames.shape[1]
    qq = np.asarray(q, dtype=np.complex128).reshape(len(frames), d)
    return np.einsum("jd,jda->ja", qq, frames, optimize=True)


def eval_basis_field(points, g, q, omega, family, obs_chunk=128, src_chunk=64):
    """Independent pairwise evaluator for either 2-DOF or 3-DOF source orientation basis."""
    K = coefficients_to_vector(g, q, family)
    P = np.asarray(points, dtype=np.float64)
    SP = np.asarray(g.ant_src_points, dtype=np.float64)
    SW = np.asarray(g.source_weights, dtype=np.float64)
    E = np.zeros((len(P), 3), dtype=np.complex128)
    ce = 1.0 / (4.0 * np.pi * c2.EPS0)
    for oa in range(0, len(P), int(obs_chunk)):
        ob = min(len(P), oa + int(obs_chunk))
        X = P[oa:ob]
        Eb = np.zeros((len(X), 3), dtype=np.complex128)
        for sa in range(0, len(SP), int(src_chunk)):
            sb = min(len(SP), sa + int(src_chunk))
            r = X[:, None, :] - SP[None, sa:sb, :]
            Rraw = np.linalg.norm(r, axis=2)
            if np.any(Rraw <= PAIRWISE_GUARD_M):
                raise RuntimeError(
                    "pairwise evaluator singularity guard would become active; zero-softening contract violated"
                )
            R = Rraw
            Ks = K[sa:sb]
            rk = np.sum(r * Ks[None, :, :], axis=2)
            w = SW[None, sa:sb, None]
            ph = np.exp(-1j * omega * R / c2.C0)[:, :, None]
            gp = 3 * r * rk[:, :, None] / R[:, :, None] ** 5 - Ks[None, :, :] / R[:, :, None] ** 3
            gk = (3 * r * rk[:, :, None] / R[:, :, None] ** 4 - Ks[None, :, :] / R[:, :, None] ** 2) / c2.C0
            gd = (r * rk[:, :, None] / R[:, :, None] ** 3 - Ks[None, :, :] / R[:, :, None]) / (c2.C0 * c2.C0)
            Eb += np.sum(ce * w * ph * (gp / (1j * omega) + gk + 1j * omega * gd), axis=1)
        E[oa:ob] = Eb
    return E


def pairwise_operator_audit(A, g, q, omega, family, obs_count=12):
    n = g.num_collocation_points
    k = min(int(obs_count), n)
    idx = np.unique(np.linspace(0, n - 1, k, dtype=np.int64))
    E = eval_basis_field(
        np.asarray(g.solve_obs_points)[idx], g, q, omega, family,
        obs_chunk=max(1, len(idx)), src_chunk=64,
    )
    pair_coeff = v2b.field_to_coeff(
        E,
        np.asarray(g.solve_tangent1)[idx],
        np.asarray(g.solve_tangent2)[idx],
    )
    rows = np.empty(2 * len(idx), dtype=np.int64)
    rows[0::2] = 2 * idx
    rows[1::2] = 2 * idx + 1
    matrix_coeff = A[rows] @ np.asarray(q, dtype=np.complex128)
    rel = float(
        np.linalg.norm(pair_coeff - matrix_coeff)
        / max(float(np.linalg.norm(matrix_coeff)), TINY)
    )
    if rel > OPERATOR_EVALUATOR_TOL:
        raise RuntimeError(
            f"{family} assembled operator / pairwise evaluator mismatch: "
            f"rel={rel:.6g} > {OPERATOR_EVALUATOR_TOL:.6g}"
        )
    return {
        "sampled_collocation_point_count": int(len(idx)),
        "relative_l2_error": rel,
        "tolerance": OPERATOR_EVALUATOR_TOL,
        "passed": True,
    }


def solver_diagnostics(prep):
    As = np.asarray(prep["As"])
    fro2 = float(np.vdot(As, As).real)
    smax = float(prep["smax"])
    alpha = float(prep["ridge_alpha"])
    cs = np.asarray(prep["cs"], dtype=np.float64)
    stable_rank = float(fro2 / max(smax * smax, TINY))
    reg_cond_upper = float((smax * smax + alpha * alpha) / max(alpha * alpha, TINY))
    return {
        "scaled_operator_smax_estimate": smax,
        "ridge_alpha": alpha,
        "lambda_rel": LAMBDA_REL,
        "column_scale_min": float(np.min(cs)),
        "column_scale_median": float(np.median(cs)),
        "column_scale_max": float(np.max(cs)),
        "column_scale_max_over_min": float(np.max(cs) / max(float(np.min(cs)), TINY)),
        "stable_rank_proxy": stable_rank,
        "regularized_normal_condition_upper_bound": reg_cond_upper,
        "note": "diagnostic only; not a physics gate",
    }


def coefficient_energy_diagnostics(g, q, family):
    frames = source_frames(g, family)
    d = frames.shape[1]
    qq = np.asarray(q, dtype=np.complex128).reshape(len(frames), d)
    w = np.asarray(g.source_weights, dtype=np.float64)
    tangent_e = float(np.sum(w * np.sum(np.abs(qq[:, :2]) ** 2, axis=1)))
    normal_e = 0.0 if d == 2 else float(np.sum(w * np.abs(qq[:, 2]) ** 2))
    total_e = tangent_e + normal_e
    total_area = max(float(np.sum(w)), TINY)
    K = coefficients_to_vector(g, q, family)
    weighted_rms = float(np.sqrt(np.sum(w * np.sum(np.abs(K) ** 2, axis=1)) / total_area))
    return {
        "weighted_vector_coefficient_rms_A_per_m": weighted_rms,
        "weighted_tangent_energy_proxy": tangent_e,
        "weighted_normal_energy_proxy": normal_e,
        "weighted_total_energy_proxy": total_e,
        "normal_energy_fraction": float(normal_e / max(total_e, TINY)),
        "normal_to_tangent_energy_ratio": float(normal_e / max(tangent_e, TINY)),
        "source_orientations_per_point": int(d),
        "unknown_count": int(len(q)),
    }


def load_v2d_reference(path):
    if not path.exists():
        raise FileNotFoundError(f"V2D report not found: {path}")
    r = json.loads(path.read_text(encoding="utf-8"))
    if r.get("experiment") != "STEP4C3_V2D_FIXED_N_SOURCE_PLACEMENT_LOCALIZATION":
        raise RuntimeError("unexpected V2D report schema")
    if r.get("status") != "complete":
        raise RuntimeError("V2E requires a completed V2D formal run")
    branch = str(r.get("decision_tree", {}).get("branch", ""))
    if branch != "B_PLACEMENT_SENSITIVE_BUT_NOT_QUALIFIED":
        raise RuntimeError(
            "This V2E package is predeclared for the observed V2D Branch B; "
            f"found {branch!r}. Refusing to reinterpret a different upstream result."
        )
    if str(r.get("decision_tree", {}).get("predeclared_candidate_rule", "none")) != "none":
        raise RuntimeError("V2E requires V2D to have no qualified promoted placement")
    return r


def v2d_baseline_reproduction_check(v2d_report, f, coll_rel, regions, worst):
    ref = v2d_report["placements"]["baseline_inner_surface_fps"]["frequency_results"][freq_key(f)]
    diffs = {
        "collocation_global": abs(float(coll_rel) - float(ref["collocation_global"])),
        "holdout_global": abs(float(regions["global"]["gate_value"]) - float(ref["holdout_global"])),
        "worst_gate": abs(float(worst["gate_value"]) - float(ref["worst_region"]["gate_value"])),
    }
    for region, rr in regions.items():
        if region not in ref["regions"]:
            raise RuntimeError(f"V2D baseline missing region {region}")
        diffs[f"region:{region}"] = abs(
            float(rr["gate_value"]) - float(ref["regions"][region]["gate_value"])
        )
    maxdiff = max(diffs.values())
    name_match = str(worst["region"]) == str(ref["worst_region"]["region"])
    passed = bool(maxdiff <= BASELINE_REPRO_TOL and name_match)
    out = {
        "max_absolute_metric_difference": float(maxdiff),
        "tolerance": BASELINE_REPRO_TOL,
        "worst_region_name_match": bool(name_match),
        "passed": passed,
        "differences": diffs,
    }
    if not passed:
        raise RuntimeError(f"V2E tangent baseline does not reproduce V2D baseline at {f/1e6:.0f} MHz: {out}")
    return out


def decision_tree(report):
    base = report["basis_families"][BASELINE_BASIS]
    aug = report["basis_families"][AUGMENTED_BASIS]
    if base.get("status") != "complete" or aug.get("status") != "complete":
        return {
            "branch": "INCOMPLETE_RUN",
            "promoted_basis": "none",
            "explanation": "The full 2-family x 3-frequency matrix is incomplete.",
            "next_step_constraint": "Step 4D remains blocked.",
        }

    def fr(pr, f):
        return pr["frequency_results"][freq_key(f)]

    all_aug_pass = all(bool(fr(aug, f)["passed_collocation_and_all_holdout_regions"]) for f in FREQS)
    b3, b6, b9 = fr(base, 3e6), fr(base, 6e6), fr(base, 9e6)
    a3, a6, a9 = fr(aug, 3e6), fr(aug, 6e6), fr(aug, 9e6)

    facts = {
        "baseline_all_frequency_qualified": bool(all(fr(base, f)["passed_collocation_and_all_holdout_regions"] for f in FREQS)),
        "augmented_all_frequency_qualified": bool(all_aug_pass),
        "delta_augmented_minus_baseline": {},
    }
    for f in FREQS:
        bf, af = fr(base, f), fr(aug, f)
        facts["delta_augmented_minus_baseline"][freq_key(f)] = {
            "collocation_global": float(af["collocation_global"] - bf["collocation_global"]),
            "holdout_global": float(af["holdout_global"] - bf["holdout_global"]),
            "worst_region_gate": float(af["worst_region"]["gate_value"] - bf["worst_region"]["gate_value"]),
            "baseline_worst_region": str(bf["worst_region"]["region"]),
            "augmented_worst_region": str(af["worst_region"]["region"]),
        }

    if all_aug_pass:
        branch = "A_FULL_VECTOR_ELECTRIC_DIPOLE_BASIS_QUALIFIES"
        promoted = AUGMENTED_BASIS
        explanation = (
            "The single predeclared full-vector electric-dipole augmentation passes every frozen 3/6/9 MHz collocation and independent holdout gate. "
            "This identifies the two-orientation tangent restriction as a material representation limitation in the harmonic diagnostic."
        )
        next_step = (
            "Do not enter Step 4D yet. Implement the same 3-DOF auxiliary-source orientation basis in the production time-domain MAS operator, keep the original observation cloud fixed, and run production confirmation, the real broadband replay, and the frozen long-tail/finiteness/fingerprint gates."
        )
    else:
        six_regression = bool(not a6["passed_collocation_and_all_holdout_regions"])
        nine_worse = float(a9["worst_region"]["gate_value"]) > float(b9["worst_region"]["gate_value"]) + TREND_TOL
        train_holdout_tradeoff = (
            float(a9["collocation_global"]) < float(b9["collocation_global"]) - TREND_TOL
            and float(a9["worst_region"]["gate_value"]) > float(b9["worst_region"]["gate_value"]) + TREND_TOL
        )
        clean_improvement = (
            float(a9["holdout_global"]) < float(b9["holdout_global"]) - TREND_TOL
            and float(a9["worst_region"]["gate_value"]) < float(b9["worst_region"]["gate_value"]) - TREND_TOL
            and bool(a6["passed_collocation_and_all_holdout_regions"])
            and float(a3["worst_region"]["gate_value"]) <= float(b3["worst_region"]["gate_value"]) + TREND_TOL
        )
        promoted = "none"
        if train_holdout_tradeoff:
            branch = "C_BASIS_TRAINING_HOLDOUT_TRADEOFF"
            explanation = (
                "The augmented basis improves the 9 MHz collocation fit but worsens the independent 9 MHz holdout. "
                "This is a generalization tradeoff, not qualification."
            )
        elif six_regression or nine_worse:
            branch = "D_BASIS_REGRESSION_OR_MIXED_RESPONSE"
            explanation = (
                "The full-vector augmentation either breaks the 6 MHz non-regression control or worsens the primary 9 MHz independent holdout. "
                "Do not tune the basis after seeing this result."
            )
        elif clean_improvement:
            branch = "B_BASIS_SENSITIVE_BUT_NOT_QUALIFIED"
            explanation = (
                "The full-vector augmentation improves the 9 MHz independent holdout without 6 MHz regression or 3 MHz worst-region regression, but one or more frozen 5% gates remain open. "
                "The representation is basis-sensitive, but this predeclared augmentation does not close the band."
            )
        else:
            branch = "E_NO_QUALIFIED_BASIS_BENEFIT"
            explanation = (
                "The single predeclared orientation-basis augmentation does not qualify the band and does not provide a clean cross-frequency improvement pattern."
            )
        next_step = (
            "V2E is the predeclared stop-loss basis experiment. Do not continue arbitrary basis-family, lambda, d/R, seed, density, or residual-targeted sweeps merely to cross 5%. "
            "Treat the remaining upper-band error as a quantified limitation unless the manuscript's central claim truly requires uniform <5% accuracy over 3-9 MHz; in that case a separately designed method-level reformulation is required. Step 4D remains blocked under the frozen qualification tree."
        )

    return {
        "branch": branch,
        "promoted_basis": promoted,
        "trend_tolerance_absolute": TREND_TOL,
        "facts": facts,
        "explanation": explanation,
        "next_step_constraint": next_step,
    }


def write_outputs(out, report):
    report["decision_tree"] = decision_tree(report)
    v2d.write_json(out / "step4c3_v2e_report.json", report)

    summary_fields = [
        "basis_family", "frequency_Hz", "unknown_count", "collocation_global",
        "holdout_global", "worst_region", "worst_region_gate", "passed",
        "operator_evaluator_rel_l2", "stable_rank_proxy", "ridge_alpha",
        "column_scale_max_over_min", "weighted_vector_coefficient_rms_A_per_m",
        "normal_energy_fraction",
    ]
    with (out / "step4c3_v2e_basis_frequency_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for family in BASIS_FAMILIES:
            pr = report["basis_families"].get(family, {})
            for fk, rr in pr.get("frequency_results", {}).items():
                if rr.get("status") != "complete":
                    continue
                sd = rr["solver_diagnostics"]
                ed = rr["coefficient_energy_diagnostics"]
                w.writerow({
                    "basis_family": family,
                    "frequency_Hz": int(fk),
                    "unknown_count": ed["unknown_count"],
                    "collocation_global": rr["collocation_global"],
                    "holdout_global": rr["holdout_global"],
                    "worst_region": rr["worst_region"]["region"],
                    "worst_region_gate": rr["worst_region"]["gate_value"],
                    "passed": rr["passed_collocation_and_all_holdout_regions"],
                    "operator_evaluator_rel_l2": rr["operator_evaluator_consistency"]["relative_l2_error"],
                    "stable_rank_proxy": sd["stable_rank_proxy"],
                    "ridge_alpha": sd["ridge_alpha"],
                    "column_scale_max_over_min": sd["column_scale_max_over_min"],
                    "weighted_vector_coefficient_rms_A_per_m": ed["weighted_vector_coefficient_rms_A_per_m"],
                    "normal_energy_fraction": ed["normal_energy_fraction"],
                })

    region_fields = [
        "basis_family", "frequency_Hz", "region", "point_count", "residual_rms",
        "incident_rms", "incident_ratio", "local_rel", "global_norm", "gate_metric",
        "gate_value", "passed",
    ]
    with (out / "step4c3_v2e_region_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=region_fields)
        w.writeheader()
        for family in BASIS_FAMILIES:
            pr = report["basis_families"].get(family, {})
            for fk, rr in pr.get("frequency_results", {}).items():
                if rr.get("status") != "complete":
                    continue
                for region, m in rr["regions"].items():
                    row = {"basis_family": family, "frequency_Hz": int(fk), "region": region}
                    for k in region_fields:
                        if k in m:
                            row[k] = m[k]
                    w.writerow(row)

    cond_fields = [
        "basis_family", "frequency_Hz", "unknown_count", "scaled_operator_smax_estimate",
        "ridge_alpha", "lambda_rel", "column_scale_min", "column_scale_median",
        "column_scale_max", "column_scale_max_over_min", "stable_rank_proxy",
        "regularized_normal_condition_upper_bound",
    ]
    with (out / "step4c3_v2e_condition_effective_rank_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cond_fields)
        w.writeheader()
        for family in BASIS_FAMILIES:
            pr = report["basis_families"].get(family, {})
            for fk, rr in pr.get("frequency_results", {}).items():
                if rr.get("status") != "complete":
                    continue
                sd = rr["solver_diagnostics"]
                ed = rr["coefficient_energy_diagnostics"]
                w.writerow({"basis_family": family, "frequency_Hz": int(fk), "unknown_count": ed["unknown_count"], **{k: sd[k] for k in cond_fields if k in sd}})

    energy_fields = [
        "basis_family", "frequency_Hz", "unknown_count", "source_orientations_per_point",
        "weighted_vector_coefficient_rms_A_per_m", "weighted_tangent_energy_proxy",
        "weighted_normal_energy_proxy", "weighted_total_energy_proxy",
        "normal_energy_fraction", "normal_to_tangent_energy_ratio",
    ]
    with (out / "step4c3_v2e_coefficient_energy_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=energy_fields)
        w.writeheader()
        for family in BASIS_FAMILIES:
            pr = report["basis_families"].get(family, {})
            for fk, rr in pr.get("frequency_results", {}).items():
                if rr.get("status") != "complete":
                    continue
                ed = rr["coefficient_energy_diagnostics"]
                w.writerow({"basis_family": family, "frequency_Hz": int(fk), **{k: ed[k] for k in energy_fields if k in ed}})

    lines = [
        "Step 4C3 V2E fixed-observation basis-family probe",
        f"status: {report['status']}",
        f"decision: {report['decision_tree']['branch']}",
        f"promoted basis: {report['decision_tree']['promoted_basis']}",
        "",
    ]
    for family in BASIS_FAMILIES:
        pr = report["basis_families"].get(family, {})
        lines.append(f"[{family}] status={pr.get('status')}")
        for f in FREQS:
            rr = pr.get("frequency_results", {}).get(freq_key(f))
            if rr and rr.get("status") == "complete":
                lines.append(
                    f"  {f/1e6:.0f} MHz: coll={rr['collocation_global']:.8g}, "
                    f"hold_global={rr['holdout_global']:.8g}, "
                    f"worst={rr['worst_region']['gate_value']:.8g} ({rr['worst_region']['region']}), "
                    f"pass={rr['passed_collocation_and_all_holdout_regions']}, "
                    f"normal_energy_fraction={rr['coefficient_energy_diagnostics']['normal_energy_fraction']:.8g}"
                )
        lines.append("")
    lines.append(report["decision_tree"]["explanation"])
    lines.append(report["decision_tree"]["next_step_constraint"])
    (out / "step4c3_v2e_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def initial_report(forcing_path, v2d_path, out):
    return {
        "experiment": "STEP4C3_V2E_FIXED_OBSERVATION_BASIS_FAMILY",
        "status": "not_run_due_to_initialization",
        "purpose": "single predeclared source-orientation basis augmentation after V2C-A density and V2D placement probes",
        "output_directory": str(out),
        "forcing_file": str(forcing_path),
        "forcing_sha256": "not_computed_due_to_initialization",
        "v2d_reference_report": str(v2d_path),
        "v2d_reference_report_sha256": "not_computed_due_to_initialization",
        "frozen": {
            "frequencies_Hz": [int(x) for x in FREQS],
            "sources_per_arm": SOURCE_COUNT_PER_ARM,
            "source_depth_fraction": DEPTH_FRAC,
            "lambda_rel": LAMBDA_REL,
            "softening_m": SOFTENING_M,
            "phasor_time_sample_count": PHASOR_SAMPLE_COUNT,
            "boundary_gate": BOUNDARY_GATE,
            "weak_region_threshold": float(c3.WEAK_REGION_THRESHOLD),
            "source_placement": "original production baseline_inner_surface_fps only",
            "basis_families": list(BASIS_FAMILIES),
            "basis_selection_dependency": "predeclared_from_source_semantics_not_from_residual_targeting",
            "kernel_rule": "same retarded electric-dipole tensor; augmentation changes only local source orientation span",
            "collocation_rule": "original formal 900-case collocation fixed bitwise",
            "holdout_rule": "original independent holdout fixed bitwise",
        },
        "upstream_precondition": {"status": "not_run_due_to_initialization"},
        "fixed_observations": {"status": "not_run_due_to_initialization"},
        "phasor_fit": {"status": "not_run_due_to_initialization"},
        "basis_families": {family: {"status": "not_run_due_to_initialization"} for family in BASIS_FAMILIES},
        "decision_tree": {
            "branch": "not_evaluated_due_to_initialization",
            "promoted_basis": "none",
            "explanation": "not_evaluated_due_to_initialization",
            "next_step_constraint": "Step 4D remains blocked.",
        },
        "failure": {"present": False, "type": "none", "message": "none", "traceback": "none"},
    }


def run_formal(args):
    forcing_path = resolve_input_path(args.forcing)
    v2d_path = resolve_input_path(args.v2d_report)
    out = resolve_out_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = initial_report(forcing_path, v2d_path, out)
    current_family = "none"
    try:
        if not forcing_path.exists():
            raise FileNotFoundError(f"forcing file not found: {forcing_path}")
        if not v2d_path.exists():
            raise FileNotFoundError(f"V2D report not found: {v2d_path}")
        if min(int(args.operator_obs_chunk), int(args.eval_obs_chunk), int(args.eval_src_chunk)) <= 0:
            raise ValueError("chunk sizes must be positive")

        z, meta = c3.load_forcing(forcing_path)
        c3.validate_forcing_metadata(meta)
        if str(meta.get("mode")) != "formal" or int(meta.get("sources_per_arm", -1)) != SOURCE_COUNT_PER_ARM:
            raise RuntimeError("V2E requires the original formal 900-sources/arm forcing dataset")
        if abs(float(meta.get("runtime_source_softening_m", 0.0)) - SOFTENING_M) > 1e-18:
            raise RuntimeError("runtime source softening must remain zero")
        if abs(float(c2.LAMBDA_REL) - LAMBDA_REL) > 1e-18:
            raise RuntimeError("Step4C2-V4 lambda_rel differs from frozen V2E value")
        if abs(float(c3.DEPTH_FRAC) - DEPTH_FRAC) > 1e-18:
            raise RuntimeError("Step4C3 source depth differs from frozen V2E value")

        keys = (
            "mode", "total_length_m", "sources_per_arm", "n_phi", "n_cyl",
            "n_hemi", "cycles", "analysis_start_cycles", "diag_stride",
            "holdout_points_target",
        )
        settings = {k: meta[k] for k in keys}
        g = c3.build_geometry(settings)
        if c3.geometry_fingerprint(g) != meta["geometry_fingerprint"]:
            raise RuntimeError("forcing / reconstructed production geometry fingerprint mismatch")

        Hpoints = np.asarray(z["holdout_points_m"], dtype=np.float64)
        Hnormals = np.asarray(z["holdout_normals"], dtype=np.float64)
        Hmask = np.asarray(z["holdout_mask_matrix"], np.uint8).astype(bool)
        region_names = list(meta["holdout_region_names"])
        colloc_fp, holdout_fp = v2d.fixed_observation_fingerprints(g, Hpoints, Hnormals, Hmask, region_names)
        forcing_sha = v2d.sha256_file(forcing_path)
        report["forcing_sha256"] = forcing_sha
        report["fixed_observations"] = {
            "status": "complete",
            "collocation_fingerprint": colloc_fp,
            "holdout_fingerprint": holdout_fp,
            "collocation_point_count": int(g.num_collocation_points),
            "holdout_point_count": int(len(Hpoints)),
            "region_names": region_names,
        }

        fit_coll, fit_hold, fit_diag = v2d.fit_synchronized_phasors(z, meta, g, Hpoints)
        if int(fit_diag["sample_count"]) != PHASOR_SAMPLE_COUNT:
            raise RuntimeError("unexpected synchronized phasor sample count")
        report["phasor_fit"] = {"status": "complete", **fit_diag}

        v2dr = load_v2d_reference(v2d_path)
        report["v2d_reference_report_sha256"] = v2d.sha256_file(v2d_path)
        if str(v2dr.get("forcing_sha256")) != forcing_sha:
            raise RuntimeError("V2D report was generated from a different forcing file")
        vf = v2dr.get("fixed_observations", {})
        if str(vf.get("collocation_fingerprint")) != colloc_fp:
            raise RuntimeError("V2D collocation fingerprint differs from V2E")
        if str(vf.get("holdout_fingerprint")) != holdout_fp:
            raise RuntimeError("V2D holdout fingerprint differs from V2E")
        vrfit = v2dr.get("phasor_fit", {})
        if int(vrfit.get("sample_count", -1)) != int(fit_diag["sample_count"]):
            raise RuntimeError("V2D synchronized phasor sample count differs")
        if [int(x) for x in vrfit.get("diagnostic_steps", [])] != [int(x) for x in fit_diag["diagnostic_steps"]]:
            raise RuntimeError("V2D synchronized diagnostic steps differ")
        report["upstream_precondition"] = {
            "status": "complete",
            "v2d_branch": str(v2dr["decision_tree"]["branch"]),
            "v2d_promoted_placement": str(v2dr["decision_tree"]["predeclared_candidate_rule"]),
            "forcing_sha256_match": True,
            "collocation_fingerprint_match": True,
            "holdout_fingerprint_match": True,
            "diagnostic_steps_match": True,
            "source_placement_locked_to": "baseline_inner_surface_fps",
        }

        # The evaluator's 1e-15 singularity guard must remain inactive.
        dcoll = np.min(np.linalg.norm(g.solve_obs_points[:, None, :] - g.ant_src_points[None, :, :], axis=2))
        dhold = np.min(np.linalg.norm(Hpoints[:, None, :] - g.ant_src_points[None, :, :], axis=2))
        if min(float(dcoll), float(dhold)) <= PAIRWISE_GUARD_M:
            raise RuntimeError("pairwise singularity guard would be active; zero-softening contract violated")

        report["status"] = "running"
        for family in BASIS_FAMILIES:
            current_family = family
            d = 2 if family == BASELINE_BASIS else 3
            pr = {
                "status": "running",
                "basis_family": family,
                "source_orientations_per_point": d,
                "source_count_total": int(g.num_sources),
                "unknown_count": int(d * g.num_sources),
                "source_placement": "baseline_inner_surface_fps",
                "frequency_results": {},
            }
            report["basis_families"][family] = pr

            for f in FREQS:
                fk = freq_key(f)
                omega = 2.0 * np.pi * float(f)
                print(f"[Step 4C3 V2E] basis={family}, f={f/1e6:.0f} MHz: assemble/solve/evaluate ...", flush=True)
                Ecoll = np.asarray(fit_coll["phasors"][float(f)], dtype=np.complex128)
                Ehold = np.asarray(fit_hold["phasors"][float(f)], dtype=np.complex128)
                e = c2.incident_coeff(Ecoll, g)

                A, opdiag = assemble_basis_harmonic(g, omega, family, int(args.operator_obs_chunk))
                expected_cols = d * g.num_sources
                if A.shape != (2 * g.num_collocation_points, expected_cols):
                    raise RuntimeError(f"unexpected operator shape for {family}: {A.shape}")
                prep = c2.prepare_complex(A, g.solve_weights)
                q = c2.solve_complex(prep, -e)
                coll_res = A @ q + e
                coll_rel = float(np.linalg.norm(coll_res) / max(float(np.linalg.norm(e)), TINY))
                sdiag = solver_diagnostics(prep)
                audit = pairwise_operator_audit(A, g, q, omega, family)
                ediag = coefficient_energy_diagnostics(g, q, family)
                del prep, coll_res
                gc.collect()

                Escat = eval_basis_field(
                    Hpoints, g, q, omega, family,
                    obs_chunk=int(args.eval_obs_chunk),
                    src_chunk=int(args.eval_src_chunk),
                )
                total_hold = Ehold + Escat
                regions = v2a.complex_region_metrics(total_hold, Ehold, Hnormals, Hmask, region_names)
                for ir, region in enumerate(region_names):
                    if region in regions:
                        regions[region]["point_count"] = int(np.sum(Hmask[ir]))
                worst = v2d.worst_region(regions)
                all_regions_pass = bool(all(bool(x["passed"]) for x in regions.values()))
                passed = bool(coll_rel < BOUNDARY_GATE and all_regions_pass)

                fr = {
                    "status": "complete",
                    "frequency_Hz": int(f),
                    "collocation_global": coll_rel,
                    "holdout_global": float(regions["global"]["gate_value"]),
                    "worst_region": worst,
                    "regions": regions,
                    "passed_collocation_and_all_holdout_regions": passed,
                    "retarded_operator_diagnostics": opdiag,
                    "operator_evaluator_consistency": audit,
                    "solver_diagnostics": sdiag,
                    "coefficient_energy_diagnostics": ediag,
                    "softening_m": SOFTENING_M,
                }
                if family == BASELINE_BASIS:
                    fr["v2d_baseline_reproduction"] = v2d_baseline_reproduction_check(
                        v2dr, f, coll_rel, regions, worst
                    )
                else:
                    fr["v2d_baseline_reproduction"] = {
                        "applicable": False,
                        "reason": "full-vector family intentionally changes the source orientation basis at fixed source locations",
                    }
                pr["frequency_results"][fk] = fr
                del A, q, Escat, total_hold
                gc.collect()

            pr["status"] = "complete"
            report["basis_families"][family] = pr

        report["status"] = "complete"
        report["failure"] = {"present": False, "type": "none", "message": "none", "traceback": "none"}
        write_outputs(out, report)
        return report

    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {
            "present": True,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if current_family in report["basis_families"]:
            pr = report["basis_families"][current_family]
            if pr.get("status") != "complete":
                pr["status"] = "failed"
                pr["failure_reason"] = str(exc)
        write_outputs(out, report)
        raise


def self_test():
    settings = {
        "mode": "formal",
        "total_length_m": 16.0e-3,
        "sources_per_arm": 24,
        "n_phi": 12,
        "n_cyl": 10,
        "n_hemi": 4,
        "cycles": 2.0,
        "analysis_start_cycles": 0.5,
        "diag_stride": 2,
        "holdout_points_target": 100,
    }
    g = c3.build_geometry(settings)
    omega = 2.0 * np.pi * 3.0e6
    A2, _ = c2.assemble_retarded_harmonic(g, omega, chunk=8)
    A3, _ = assemble_full_vector_harmonic(g, omega, chunk=8)

    # The first two columns of every 3-column local block must exactly recover
    # the legacy tangent basis up to roundoff.
    idx = np.arange(3 * g.num_sources, dtype=np.int64).reshape(-1, 3)[:, :2].reshape(-1)
    tangent_slice_rel = float(
        np.linalg.norm(A3[:, idx] - A2)
        / max(float(np.linalg.norm(A2)), TINY)
    )
    if tangent_slice_rel > 5.0e-13:
        raise RuntimeError(f"full-vector tangent slice does not reproduce baseline operator: {tangent_slice_rel}")

    rng = np.random.default_rng(20261003)
    q3 = (rng.normal(size=A3.shape[1]) + 1j * rng.normal(size=A3.shape[1])) * 1.0e-6
    E3 = eval_basis_field(g.solve_obs_points, g, q3, omega, AUGMENTED_BASIS, obs_chunk=16, src_chunk=16)
    e3 = v2b.field_to_coeff(E3, g.solve_tangent1, g.solve_tangent2)
    op_pair_rel = float(
        np.linalg.norm(e3 - A3 @ q3) / max(float(np.linalg.norm(A3 @ q3)), TINY)
    )
    if op_pair_rel > OPERATOR_EVALUATOR_TOL:
        raise RuntimeError(f"full-vector operator/evaluator self-test mismatch: {op_pair_rel}")

    # Weighted/scaled solve chain on a realizable synthetic rhs.
    incident = -(A3 @ q3)
    prep = c2.prepare_complex(A3.copy(), g.solve_weights)
    qfit = c2.solve_complex(prep, -incident)
    solve_rel = float(
        np.linalg.norm(A3 @ qfit + incident) / max(float(np.linalg.norm(incident)), TINY)
    )
    if solve_rel >= 5.0e-4:
        raise RuntimeError(f"full-vector weighted/scaled solve residual too large: {solve_rel}")
    audit = pairwise_operator_audit(A3, g, qfit, omega, AUGMENTED_BASIS, obs_count=8)

    # Ensure normal columns are genuinely new and non-zero.
    nidx = np.arange(3 * g.num_sources, dtype=np.int64).reshape(-1, 3)[:, 2]
    normal_norm = float(np.linalg.norm(A3[:, nidx]))
    if normal_norm <= 0.0:
        raise RuntimeError("normal-orientation augmentation produced zero columns")

    report = {
        "self_test_pass": True,
        "tangent_slice_vs_legacy_operator_relative_l2": tangent_slice_rel,
        "full_vector_operator_vs_pairwise_relative_l2": op_pair_rel,
        "known_coefficient_solve_relative_l2": solve_rel,
        "solve_evaluate_chain_relative_l2": audit["relative_l2_error"],
        "normal_column_frobenius_norm": normal_norm,
        "baseline_unknown_count": int(A2.shape[1]),
        "augmented_unknown_count": int(A3.shape[1]),
        "self_test_requires_forcing": False,
    }
    print(json.dumps(report, indent=2))


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run_formal(args)


if __name__ == "__main__":
    main()
