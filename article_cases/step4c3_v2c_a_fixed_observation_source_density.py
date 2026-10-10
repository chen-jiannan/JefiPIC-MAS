#!/usr/bin/env python3
"""
Step 4C3 V2C-A — fixed-observation source-basis capacity probe.

Scientific control:
  * reuse the recorded Step-4C3 formal PIC/Jefimenko forcing;
  * reuse the V2B synchronized 22-time multitone phasor estimator;
  * keep the original 900-case collocation and independent holdout points fixed;
  * change only sources_per_arm = 900, 1200, 1500 on the same candidate surface;
  * keep d/R=0.8, lambda_rel=1e-4, zero softening, 3/6/9 MHz, and the 5% gate;
  * solve with the Step-4C2-V4 weighted/scaled complex ridge solve;
  * evaluate formal holdout fields with the independent pairwise harmonic evaluator.

This file intentionally replaces the deprecated step4c3_v2c_source_capacity_probe.py
scaffold.  It does not change production collocation points and therefore does not
require plasma forcing at any new observation locations.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
for _p in (str(HERE), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import step4c3_v2_retarded_mot_broadband_replay as c3
import step4c2_v4_retarded_frequency_regression as c2
import step4c3_v2a_feed_transition_residual_diagnostic as v2a
import step4c3_v2b_spatial_cross_validation as v2b
from EMsolver import mas_preprocessor as mp

FREQS = (3.0e6, 6.0e6, 9.0e6)
DENSITIES = (900, 1200, 1500)
BOUNDARY_GATE = 0.05
DEPTH_FRAC = 0.80
LAMBDA_REL = 1.0e-4
SOFTENING_M = 0.0
PHASOR_SAMPLE_COUNT = 22
FIT_CONDITION_LIMIT = 100.0
V2B_REPRO_TOL = 2.0e-8
OPERATOR_EVALUATOR_TOL = 5.0e-11
TREND_TOL = 1.0e-8
TINY = 1.0e-300


def parse_args():
    p = argparse.ArgumentParser(
        description="Step 4C3 V2C-A fixed-observation source-density probe"
    )
    p.add_argument(
        "--forcing",
        default="article_cases_output/step4c3_formal/step4c3_forcing.npz",
        help="Recorded formal Step-4C3 forcing NPZ (original 900-case observations)",
    )
    p.add_argument(
        "--v2b-report",
        default="article_cases_output/step4c3_v2b_formal/step4c3_v2b_report.json",
        help="Formal V2B report used as a 900-density regression reference",
    )
    p.add_argument(
        "--out",
        default="step4c3_v2c_a_formal",
        help="Output directory name under article_cases_output, or an absolute path",
    )
    p.add_argument("--operator-obs-chunk", type=int, default=8)
    p.add_argument("--eval-obs-chunk", type=int, default=128)
    p.add_argument("--eval-src-chunk", type=int, default=64)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def resolve_input_path(value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def resolve_out_path(value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    if len(p.parts) > 1 or p.parts[0] == "article_cases_output":
        return (PROJECT_ROOT / p).resolve()
    return (PROJECT_ROOT / "article_cases_output" / p).resolve()


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hash_arrays_exact(*arrays, extra_strings=()) -> str:
    h = hashlib.sha256()
    for a in arrays:
        x = np.ascontiguousarray(np.asarray(a))
        h.update(str(x.dtype).encode("ascii"))
        h.update(str(x.shape).encode("ascii"))
        h.update(x.tobytes(order="C"))
    for s in extra_strings:
        b = str(s).encode("utf-8")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


def hash_indices(per_arm_indices) -> str:
    h = hashlib.sha256()
    for arm, idx in enumerate(per_arm_indices):
        x = np.asarray(idx, dtype="<i8")
        h.update(int(arm).to_bytes(4, "little", signed=False))
        h.update(len(x).to_bytes(8, "little", signed=False))
        h.update(x.tobytes(order="C"))
    return h.hexdigest()


def assert_json_clean(obj, path="root"):
    if obj is None:
        raise RuntimeError(f"JSON contains null at {path}")
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise RuntimeError(f"JSON contains non-finite float at {path}: {obj!r}")
        return
    if isinstance(obj, (np.floating,)):
        if not np.isfinite(obj):
            raise RuntimeError(f"JSON contains non-finite numpy float at {path}: {obj!r}")
        return
    if isinstance(obj, str):
        if obj.strip().lower() == "pending":
            raise RuntimeError(f"JSON contains forbidden pending marker at {path}")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_json_clean(v, f"{path}.{k}")
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_json_clean(v, f"{path}[{i}]")
        return


def write_json(path: Path, report: dict):
    assert_json_clean(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def density_key(n: int) -> str:
    return str(int(n))


def freq_key(f: float) -> str:
    return str(int(round(float(f))))


def freq_label(f: float) -> str:
    return f"{float(f)/1e6:.0f}MHz"


def fixed_observation_fingerprints(g0, Hpoints, Hnormals, Hmask, region_names):
    colloc = hash_arrays_exact(
        g0.solve_obs_points,
        g0.solve_normals,
        g0.solve_weights,
        g0.solve_tangent1,
        g0.solve_tangent2,
    )
    holdout = hash_arrays_exact(
        Hpoints,
        Hnormals,
        np.asarray(Hmask, dtype=np.uint8),
        extra_strings=tuple(region_names),
    )
    return colloc, holdout




def bind_fixed_solve_cloud(g, points, normals, weights, tangent1, tangent2):
    """Use the production setter, then restore the frozen 900-case arrays bitwise.

    set_solve_collocation() correctly normalizes normals and rebuilds tangents, but a
    second normalization can change last-bit floating values. V2C-A freezes the
    recorded observation contract exactly, so the authoritative baseline arrays are
    restored after invoking the public API.
    """
    g.set_solve_collocation(points, normals, weights)
    g.solve_obs_points = np.asarray(points, dtype=np.float64).copy()
    g.solve_normals = np.asarray(normals, dtype=np.float64).copy()
    g.solve_weights = np.asarray(weights, dtype=np.float64).copy()
    g.solve_tangent1 = np.asarray(tangent1, dtype=np.float64).copy()
    g.solve_tangent2 = np.asarray(tangent2, dtype=np.float64).copy()
    return g

def source_candidate_reference(settings):
    """Reconstruct the exact current production inner candidate clouds and FPS indices."""
    R = 1.0e-3
    total = float(settings["total_length_m"])
    gap = 2.0e-3
    Larm = 0.5 * (total - gap)
    Rs = (1.0 - DEPTH_FRAC) * R
    inner_local = mp._generate_capsule_triangles(
        Rs,
        R,
        Larm,
        int(settings["n_cyl"]),
        int(settings["n_phi"]),
        int(settings["n_hemi"]),
    )
    C0, _, Ainner = mp._triangle_geometry(inner_local)
    Cinner, Ninner = mp._project_to_capsule(C0, Rs, R, Larm)
    zshifts = (-0.5 * total, 0.5 * gap)
    clouds = []
    normals = []
    per_arm = {}
    for arm, zshift in enumerate(zshifts):
        cg = mp._shift_z(Cinner, zshift)
        clouds.append(cg)
        normals.append(Ninner.copy())
        per_arm[arm] = {
            n: mp._fps_indices(cg, int(n), seed=6100 + arm)
            for n in DENSITIES
        }
    fp = hash_arrays_exact(
        np.concatenate(clouds),
        np.concatenate(normals),
        np.tile(np.asarray(Ainner, dtype=np.float64), 2),
    )
    return {
        "candidate_points_per_arm": int(len(Cinner)),
        "candidate_points_total": int(2 * len(Cinner)),
        "candidate_geometry_fingerprint": fp,
        "clouds": clouds,
        "per_arm_indices": per_arm,
    }


def source_slices(nsrc: int):
    return (slice(0, nsrc), slice(nsrc, 2 * nsrc))


def source_cloud_diagnostics(g, nsrc: int, candidate_ref, prev=None):
    if g.num_sources != 2 * nsrc:
        raise RuntimeError(f"source count mismatch for density {nsrc}: {g.num_sources}")
    candidate_count = int(candidate_ref["candidate_points_per_arm"])
    if nsrc > candidate_count:
        raise RuntimeError(
            f"design blocked: requested {nsrc} sources/arm but candidate cloud has only {candidate_count}"
        )

    per_arm_idx = [candidate_ref["per_arm_indices"][arm][nsrc] for arm in (0, 1)]
    for arm, sl in enumerate(source_slices(nsrc)):
        expected = candidate_ref["clouds"][arm][per_arm_idx[arm]]
        actual = np.asarray(g.ant_src_points[sl], float)
        if not np.array_equal(actual, expected):
            raise RuntimeError(f"production builder source selection mismatch on arm {arm}, density {nsrc}")

    nested_ok = True
    nested_details = []
    if prev is not None:
        prev_n, prev_g = prev
        for arm, (psl, csl) in enumerate(zip(source_slices(prev_n), source_slices(nsrc))):
            curr_prefix = np.asarray(g.ant_src_points[csl], float)[:prev_n]
            prev_points = np.asarray(prev_g.ant_src_points[psl], float)
            pos_ok = bool(np.array_equal(prev_points, curr_prefix))
            t1_ok = bool(np.array_equal(prev_g.source_tangent1[psl], g.source_tangent1[csl][:prev_n]))
            t2_ok = bool(np.array_equal(prev_g.source_tangent2[psl], g.source_tangent2[csl][:prev_n]))
            idx_ok = bool(np.array_equal(
                candidate_ref["per_arm_indices"][arm][prev_n],
                candidate_ref["per_arm_indices"][arm][nsrc][:prev_n],
            ))
            ok = pos_ok and t1_ok and t2_ok and idx_ok
            nested_ok = nested_ok and ok
            nested_details.append({
                "arm": arm,
                "previous_density": prev_n,
                "positions_prefix_exact": pos_ok,
                "tangent1_prefix_exact": t1_ok,
                "tangent2_prefix_exact": t2_ok,
                "selected_index_prefix_exact": idx_ok,
                "passed": ok,
            })
        if not nested_ok:
            raise RuntimeError(f"non-nested source placement detected at density {nsrc}")

    P = np.asarray(g.ant_src_points, float)
    nn = cKDTree(P).query(P, k=2)[0][:, 1]
    arm_spacing = []
    for arm, sl in enumerate(source_slices(nsrc)):
        Pa = P[sl]
        da = cKDTree(Pa).query(Pa, k=2)[0][:, 1]
        arm_spacing.append({
            "arm": arm,
            "min_nearest_spacing_m": float(np.min(da)),
            "median_nearest_spacing_m": float(np.median(da)),
        })

    source_fp = hash_arrays_exact(
        g.ant_src_points,
        g.ant_normals,
        g.source_weights,
        g.source_tangent1,
        g.source_tangent2,
    )
    return {
        "sources_per_arm": int(nsrc),
        "sources_total": int(g.num_sources),
        "candidate_points_per_arm": candidate_count,
        "candidate_source_ratio_per_arm": float(candidate_count / nsrc),
        "candidate_geometry_fingerprint": candidate_ref["candidate_geometry_fingerprint"],
        "selected_source_index_hash": hash_indices(per_arm_idx),
        "source_fingerprint": source_fp,
        "nested_prefix_ok": bool(nested_ok),
        "nested_prefix_details": nested_details if nested_details else [{
            "arm": -1,
            "previous_density": 0,
            "positions_prefix_exact": True,
            "tangent1_prefix_exact": True,
            "tangent2_prefix_exact": True,
            "selected_index_prefix_exact": True,
            "passed": True,
        }],
        "min_nearest_spacing_m": float(np.min(nn)),
        "median_nearest_spacing_m": float(np.median(nn)),
        "per_arm_spacing": arm_spacing,
        "source_weight_sum": float(np.sum(g.source_weights)),
        "source_weighting_rule": str(g.metadata.get("source_weighting", "unknown")),
        "source_sampling_rule": str(g.metadata.get("source_sampling", "unknown")),
    }


def fit_synchronized_phasors(z, meta, g0, Hpoints):
    times = np.asarray(z["time_s"], float)
    plasma_coll = z["plasma_collocation_E_V_per_m"]
    diag_steps = np.asarray(z["diag_steps"], np.int64)
    plasma_hold = z["plasma_holdout_E_V_per_m"]
    lookup = {int(s): i for i, s in enumerate(diag_steps)}
    analysis_start_s = float(meta["analysis_start_cycles"]) / c3.F_DRIVE
    diag_analysis = [
        int(s) for s in diag_steps
        if int(s) < len(times) and times[int(s)] >= analysis_start_s
    ]
    if len(diag_analysis) != PHASOR_SAMPLE_COUNT:
        raise RuntimeError(
            f"V2C-A requires the V2B synchronized {PHASOR_SAMPLE_COUNT}-time estimator; "
            f"found {len(diag_analysis)} post-ramp diagnostic samples"
        )

    coll = np.empty((len(diag_analysis), g0.num_collocation_points, 3), dtype=np.float64)
    hold = np.empty((len(diag_analysis), len(Hpoints), 3), dtype=np.float64)
    hold_times = np.empty(len(diag_analysis), dtype=np.float64)
    for j, step in enumerate(diag_analysis):
        hi = lookup[step]
        hold_times[j] = times[step]
        coll[j] = np.asarray(plasma_coll[step], float) + c3.feed_at_time(
            g0.solve_obs_points, float(times[step])
        )
        hold[j] = np.asarray(plasma_hold[hi], float) + c3.feed_at_time(
            Hpoints, float(times[step])
        )

    fit_coll = v2a.fit_multitone(hold_times, coll, freqs=FREQS)
    fit_hold = v2a.fit_multitone(hold_times, hold, freqs=FREQS)
    del coll, hold
    gc.collect()
    max_cond = max(float(fit_coll["condition_number"]), float(fit_hold["condition_number"]))
    if max_cond >= FIT_CONDITION_LIMIT:
        raise RuntimeError(f"synchronized multitone fit ill-conditioned: {max_cond}")

    rms = {}
    for f in FREQS:
        E = np.asarray(fit_coll["phasors"][float(f)], np.complex128)
        rms[f] = float(np.sqrt(np.mean(np.sum(np.abs(E) ** 2, axis=1))))
    max_rms = max(rms.values())
    strength = {f: float(rms[f] / max(max_rms, TINY)) for f in FREQS}
    weak = [f for f in FREQS if strength[f] < v2b.MIN_HARMONIC_STRENGTH_RATIO]
    if weak:
        raise RuntimeError(
            "V2C-A expected the V2B-assessable 3/6/9 MHz harmonics, but found weak harmonics: "
            + ", ".join(freq_label(f) for f in weak)
        )
    phasor_fp = hash_arrays_exact(
        *(np.asarray(fit_coll["phasors"][f], np.complex128) for f in FREQS),
        *(np.asarray(fit_hold["phasors"][f], np.complex128) for f in FREQS),
        hold_times,
    )
    return fit_coll, fit_hold, {
        "sample_count": int(len(diag_analysis)),
        "diagnostic_steps": [int(x) for x in diag_analysis],
        "times_s": [float(x) for x in hold_times],
        "collocation_fit_condition_number": float(fit_coll["condition_number"]),
        "holdout_fit_condition_number": float(fit_hold["condition_number"]),
        "max_fit_condition_number": max_cond,
        "minimum_harmonic_strength_ratio": float(v2b.MIN_HARMONIC_STRENGTH_RATIO),
        "incident_harmonic_strength_ratio_to_max": {
            freq_key(f): strength[f] for f in FREQS
        },
        "phasor_fingerprint": phasor_fp,
    }


def weighted_current_rms(g, q):
    q2 = np.asarray(q, np.complex128).reshape(-1, 2)
    sw = np.asarray(g.source_weights, float)
    return float(np.sqrt(
        np.sum(sw * np.sum(np.abs(q2) ** 2, axis=1))
        / max(float(np.sum(sw)), TINY)
    ))


def pairwise_operator_audit(A, g, q, omega, obs_count=12):
    n = g.num_collocation_points
    k = min(int(obs_count), n)
    idx = np.unique(np.linspace(0, n - 1, k, dtype=np.int64))
    E = c2.eval_retarded_field(
        np.asarray(g.solve_obs_points)[idx],
        g,
        q,
        omega,
        obs_chunk=max(1, len(idx)),
        src_chunk=64,
    )
    pair_coeff = v2b.field_to_coeff(
        E,
        np.asarray(g.solve_tangent1)[idx],
        np.asarray(g.solve_tangent2)[idx],
    )
    rows = np.empty(2 * len(idx), dtype=np.int64)
    rows[0::2] = 2 * idx
    rows[1::2] = 2 * idx + 1
    matrix_coeff = A[rows] @ q
    rel = float(
        np.linalg.norm(pair_coeff - matrix_coeff)
        / max(float(np.linalg.norm(matrix_coeff)), TINY)
    )
    if rel > OPERATOR_EVALUATOR_TOL:
        raise RuntimeError(
            f"assembled operator / pairwise evaluator mismatch: rel={rel:.6g} "
            f"> {OPERATOR_EVALUATOR_TOL:.6g}"
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
    cs = np.asarray(prep["cs"], float)
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
        "note": "stable_rank_proxy is ||A_scaled||_F^2/sigma_max^2; diagnostic only, not a physics gate",
    }


def worst_region(regions):
    if not regions:
        raise RuntimeError("empty region metrics")
    return dict(max(regions.values(), key=lambda x: float(x["gate_value"])))


def load_v2b_reference(path: Path):
    if not path.exists():
        raise RuntimeError(f"required V2B reference report not found: {path}")
    r = json.loads(path.read_text(encoding="utf-8"))
    if r.get("step") != "4C3-V2B":
        raise RuntimeError(f"unexpected V2B report schema in {path}")
    return r


def v2b_reproduction_check(v2b_report, f, coll_rel, regions, worst):
    key = freq_label(f)
    ref = v2b_report["frequencies"][key]["synchronized_sampling_baseline"]
    diffs = {
        "collocation_rel_l2": abs(float(coll_rel) - float(ref["collocation_rel_l2"])),
        "global_gate": abs(float(regions["global"]["gate_value"]) - float(ref["regions"]["global"]["gate_value"])),
        "feed_transition_gate": abs(
            float(regions["feed_transition"]["gate_value"])
            - float(ref["regions"]["feed_transition"]["gate_value"])
        ),
        "worst_gate": abs(float(worst["gate_value"]) - float(ref["worst_region"]["gate_value"])),
    }
    maxdiff = max(diffs.values())
    name_match = str(worst["region"]) == str(ref["worst_region"]["region"])
    passed = bool(maxdiff <= V2B_REPRO_TOL and name_match)
    if not passed:
        raise RuntimeError(
            f"900-density V2B synchronized baseline reproduction failed at {key}: "
            f"maxdiff={maxdiff:.6g}, name_match={name_match}"
        )
    return {
        "reference_report_step": str(v2b_report.get("step")),
        "reference_frequency_label": key,
        "absolute_differences": diffs,
        "max_absolute_difference": float(maxdiff),
        "worst_region_name_match": name_match,
        "tolerance": V2B_REPRO_TOL,
        "passed": True,
    }


def make_initial_report(forcing_path: Path, v2b_path: Path, out: Path):
    return {
        "experiment": "STEP4C3_V2C_A_FIXED_OBSERVATION_SOURCE_DENSITY",
        "status": "initialized",
        "purpose": "source-basis capacity probe with fixed actual-plasma observation data",
        "output_directory": str(out),
        "forcing_file": str(forcing_path),
        "forcing_sha256": "not_computed_due_to_initialization",
        "v2b_reference_report": str(v2b_path),
        "frozen": {
            "frequencies_Hz": [int(f) for f in FREQS],
            "sources_per_arm": [int(n) for n in DENSITIES],
            "source_depth_fraction": DEPTH_FRAC,
            "lambda_rel": LAMBDA_REL,
            "softening_m": SOFTENING_M,
            "phasor_time_sample_count": PHASOR_SAMPLE_COUNT,
            "boundary_gate": BOUNDARY_GATE,
            "weak_region_threshold": c3.WEAK_REGION_THRESHOLD,
            "collocation_rule": "original formal 900-case production collocation points, normals, weights, tangents fixed exactly",
            "holdout_rule": "original formal independent holdout points, normals, masks fixed exactly",
            "candidate_surface_resolution_rule": "n_phi/n_cyl/n_hemi frozen to forcing metadata for every density",
        },
        "fixed_observations": {
            "collocation_fingerprint": "not_computed_due_to_initialization",
            "holdout_fingerprint": "not_computed_due_to_initialization",
            "collocation_point_count": 0,
            "holdout_point_count": 0,
        },
        "phasor_fit": {"status": "not_run_due_to_initialization"},
        "source_candidate": {"status": "not_run_due_to_initialization"},
        "densities": {
            density_key(n): {"status": "not_run_due_to_initialization"}
            for n in DENSITIES
        },
        "decision_tree": {
            "branch": "not_evaluated_due_to_initialization",
            "minimum_all_frequency_qualified_density": "none",
            "facts": {},
        },
        "failure": {
            "present": False,
            "type": "none",
            "message": "none",
            "traceback": "none",
        },
    }


def decision_tree_assessment(report):
    series = {}
    all_complete = True
    for f in FREQS:
        fk = freq_key(f)
        coll = []
        hold = []
        passed = []
        for n in DENSITIES:
            dr = report["densities"][density_key(n)]
            if dr.get("status") != "complete" or fk not in dr.get("frequency_results", {}):
                all_complete = False
                break
            fr = dr["frequency_results"][fk]
            coll.append(float(fr["collocation_global"]))
            hold.append(float(fr["worst_region"]["gate_value"]))
            passed.append(bool(fr["passed_collocation_and_all_holdout_regions"]))
        if len(coll) == len(DENSITIES):
            series[fk] = {
                "collocation": coll,
                "worst_holdout": hold,
                "all_gates_pass": passed,
                "collocation_monotone_nonincreasing": bool(all(coll[i+1] <= coll[i] + TREND_TOL for i in range(2))),
                "holdout_monotone_nonincreasing": bool(all(hold[i+1] <= hold[i] + TREND_TOL for i in range(2))),
                "any_material_worsening": bool(any(
                    coll[i+1] > coll[i] + TREND_TOL or hold[i+1] > hold[i] + TREND_TOL
                    for i in range(2)
                )),
            }

    qualified = "none"
    if all_complete:
        for n in DENSITIES:
            if all(report["densities"][density_key(n)]["frequency_results"][freq_key(f)]["passed_collocation_and_all_holdout_regions"] for f in FREQS):
                qualified = str(n)
                break

    branch = "INCOMPLETE_RUN"
    explanation = "Formal density/frequency matrix is incomplete; no scientific branch assigned."
    if all_complete:
        s9 = series[freq_key(9.0e6)]
        s6 = series[freq_key(6.0e6)]
        s3 = series[freq_key(3.0e6)]
        if (
            qualified != "none"
            and s9["collocation_monotone_nonincreasing"]
            and s9["holdout_monotone_nonincreasing"]
            and s6["all_gates_pass"][0]
            and all(s6["all_gates_pass"])
            and not s3["any_material_worsening"]
        ):
            branch = "A_CLEAR_MONOTONE_CONVERGENCE_AND_GATE_CROSSING"
            explanation = (
                "A higher source density gives monotone 9 MHz improvement, an all-frequency qualified density exists, "
                "6 MHz remains a passing non-regression control, and 3 MHz does not materially worsen. Proceed only to V2C-B production confirmation."
            )
        elif any(v["any_material_worsening"] for v in series.values()):
            branch = "D_HIGHER_DENSITY_WORSENS_AT_LEAST_ONE_FROZEN_METRIC"
            explanation = (
                "At least one collocation or worst-holdout metric worsens beyond the fixed numerical trend tolerance. "
                "Do not select the best-looking density or retune lambda; inspect spacing/rank/regularization diagnostics."
            )
        else:
            c9 = s9["collocation"]
            h9 = s9["worst_holdout"]
            if c9[-1] < c9[0] - TREND_TOL and h9[-1] >= BOUNDARY_GATE:
                branch = "B_OR_C_PLATFORM_REVIEW_REQUIRED"
                explanation = (
                    "9 MHz collocation improves but the final independent holdout gate remains open, or the remaining trend is too mixed "
                    "for an unambiguous automatic B-vs-C label. Apply the predeclared B/C plateau distinction from the handoff to the reported raw series; do not add sources without limit."
                )
            else:
                branch = "C_OR_MIXED_NO_CLEAR_CAPACITY_CLOSURE"
                explanation = (
                    "No all-frequency qualified density is established and there is no material worsening. Review whether collocation and holdout are both plateauing; "
                    "if so the current basis family is insufficient under the predeclared decision tree."
                )
    return {
        "branch": branch,
        "minimum_all_frequency_qualified_density": qualified,
        "trend_tolerance_absolute": TREND_TOL,
        "facts": series,
        "explanation": explanation,
        "predeclared_next_step_constraint": (
            "V2C-A never closes C3 by itself. V2C-B production confirmation requires real PIC field on any changed production collocation points; "
            "Step 4D remains blocked until production confirmation, real broadband replay, all 5% gates, long-tail <=1.001, finiteness, and fingerprint checks pass."
        ),
    }


def write_csv_outputs(out: Path, report: dict):
    out.mkdir(parents=True, exist_ok=True)

    summary_fields = [
        "sources_per_arm", "frequency_Hz", "collocation_global", "holdout_global",
        "worst_region", "worst_region_gate", "passed", "weighted_current_rms_A_per_m",
        "operator_evaluator_rel_l2", "scaled_operator_smax_estimate", "ridge_alpha",
        "stable_rank_proxy", "column_scale_max_over_min",
    ]
    with (out / "step4c3_v2c_a_density_frequency_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for n in DENSITIES:
            dr = report["densities"].get(density_key(n), {})
            for fk, fr in dr.get("frequency_results", {}).items():
                if fr.get("status") != "complete":
                    continue
                sd = fr["solver_diagnostics"]
                w.writerow({
                    "sources_per_arm": n,
                    "frequency_Hz": int(fk),
                    "collocation_global": fr["collocation_global"],
                    "holdout_global": fr["holdout_global"],
                    "worst_region": fr["worst_region"]["region"],
                    "worst_region_gate": fr["worst_region"]["gate_value"],
                    "passed": fr["passed_collocation_and_all_holdout_regions"],
                    "weighted_current_rms_A_per_m": fr["weighted_current_rms_A_per_m"],
                    "operator_evaluator_rel_l2": fr["operator_evaluator_consistency"]["relative_l2_error"],
                    "scaled_operator_smax_estimate": sd["scaled_operator_smax_estimate"],
                    "ridge_alpha": sd["ridge_alpha"],
                    "stable_rank_proxy": sd["stable_rank_proxy"],
                    "column_scale_max_over_min": sd["column_scale_max_over_min"],
                })

    region_fields = [
        "sources_per_arm", "frequency_Hz", "region", "point_count", "residual_rms",
        "incident_rms", "incident_ratio", "local_rel", "global_norm", "gate_metric",
        "gate_value", "passed",
    ]
    with (out / "step4c3_v2c_a_region_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=region_fields)
        w.writeheader()
        for n in DENSITIES:
            dr = report["densities"].get(density_key(n), {})
            for fk, fr in dr.get("frequency_results", {}).items():
                if fr.get("status") != "complete":
                    continue
                for r, rr in fr["regions"].items():
                    w.writerow({"sources_per_arm": n, "frequency_Hz": int(fk), **{k: rr[k] for k in region_fields if k in rr}})

    geom_fields = [
        "sources_per_arm", "sources_total", "candidate_points_per_arm",
        "candidate_source_ratio_per_arm", "candidate_geometry_fingerprint",
        "selected_source_index_hash", "source_fingerprint", "nested_prefix_ok",
        "min_nearest_spacing_m", "median_nearest_spacing_m", "source_weight_sum",
        "source_weighting_rule", "source_sampling_rule",
    ]
    with (out / "step4c3_v2c_a_source_geometry_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=geom_fields)
        w.writeheader()
        for n in DENSITIES:
            gd = report["densities"].get(density_key(n), {}).get("source_geometry")
            if gd:
                w.writerow({k: gd[k] for k in geom_fields})

    cond_fields = [
        "sources_per_arm", "frequency_Hz", "scaled_operator_smax_estimate", "ridge_alpha",
        "lambda_rel", "column_scale_min", "column_scale_median", "column_scale_max",
        "column_scale_max_over_min", "stable_rank_proxy", "regularized_normal_condition_upper_bound",
    ]
    with (out / "step4c3_v2c_a_condition_effective_rank_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cond_fields)
        w.writeheader()
        for n in DENSITIES:
            dr = report["densities"].get(density_key(n), {})
            for fk, fr in dr.get("frequency_results", {}).items():
                if fr.get("status") != "complete":
                    continue
                sd = fr["solver_diagnostics"]
                w.writerow({"sources_per_arm": n, "frequency_Hz": int(fk), **{k: sd[k] for k in cond_fields if k in sd}})


def write_summary(out: Path, report: dict):
    lines = [
        "Step 4C3 V2C-A fixed-observation source-density probe",
        f"status: {report['status']}",
        f"forcing_sha256: {report['forcing_sha256']}",
        f"collocation_fingerprint: {report['fixed_observations']['collocation_fingerprint']}",
        f"holdout_fingerprint: {report['fixed_observations']['holdout_fingerprint']}",
        f"phasor_sample_count: {report.get('phasor_fit', {}).get('sample_count', 'not-run')}",
        f"candidate_points_per_arm: {report.get('source_candidate', {}).get('candidate_points_per_arm', 'not-run')}",
    ]
    for n in DENSITIES:
        dr = report["densities"].get(density_key(n), {})
        lines.append(f"density {n}: status={dr.get('status', 'missing')}")
        for fk, fr in dr.get("frequency_results", {}).items():
            if fr.get("status") == "complete":
                lines.append(
                    f"  {int(fk)/1e6:.0f}MHz colloc={fr['collocation_global']:.8g} "
                    f"holdout_global={fr['holdout_global']:.8g} "
                    f"worst={fr['worst_region']['gate_value']:.8g}({fr['worst_region']['region']}) "
                    f"pass={fr['passed_collocation_and_all_holdout_regions']}"
                )
    lines += [
        f"decision_tree_branch: {report['decision_tree']['branch']}",
        f"minimum_all_frequency_qualified_density: {report['decision_tree']['minimum_all_frequency_qualified_density']}",
        f"interpretation: {report['decision_tree'].get('explanation', 'none')}",
    ]
    if report["failure"]["present"]:
        lines.append(f"failure: {report['failure']['type']}: {report['failure']['message']}")
    (out / "step4c3_v2c_a_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_outputs(out: Path, report: dict):
    report["decision_tree"] = decision_tree_assessment(report)
    write_json(out / "step4c3_v2c_a_report.json", report)
    write_csv_outputs(out, report)
    write_summary(out, report)


def run_formal(args):
    forcing_path = resolve_input_path(args.forcing)
    v2b_path = resolve_input_path(args.v2b_report)
    out = resolve_out_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = make_initial_report(forcing_path, v2b_path, out)

    current_density = "none"
    try:
        if not forcing_path.exists():
            raise FileNotFoundError(f"forcing file not found: {forcing_path}")
        if int(args.operator_obs_chunk) <= 0 or int(args.eval_obs_chunk) <= 0 or int(args.eval_src_chunk) <= 0:
            raise ValueError("chunk sizes must be positive")

        z, meta = c3.load_forcing(forcing_path)
        c3.validate_forcing_metadata(meta)
        if str(meta.get("mode")) != "formal" or int(meta.get("sources_per_arm", -1)) != 900:
            raise RuntimeError("V2C-A requires the formal original 900-sources/arm forcing dataset")
        if abs(float(meta.get("runtime_source_softening_m", 0.0)) - SOFTENING_M) > 1e-18:
            raise RuntimeError("runtime_source_softening_m must be zero")
        if int(meta.get("num_collocation_points", -1)) != 3600:
            raise RuntimeError("unexpected formal collocation count; refusing to guess a different forcing contract")
        if abs(float(c2.LAMBDA_REL) - LAMBDA_REL) > 1e-18:
            raise RuntimeError("Step4C2-V4 lambda_rel differs from frozen V2C-A value")

        keys = (
            "mode", "total_length_m", "sources_per_arm", "n_phi", "n_cyl",
            "n_hemi", "cycles", "analysis_start_cycles", "diag_stride",
            "holdout_points_target",
        )
        settings = {k: meta[k] for k in keys}
        g0 = c3.build_geometry(settings)
        if c3.geometry_fingerprint(g0) != meta["geometry_fingerprint"]:
            raise RuntimeError("forcing / reconstructed 900-case geometry fingerprint mismatch")

        Hpoints = np.asarray(z["holdout_points_m"], float)
        Hnormals = np.asarray(z["holdout_normals"], float)
        Hmask = np.asarray(z["holdout_mask_matrix"], np.uint8).astype(bool)
        region_names = list(meta["holdout_region_names"])
        colloc_fp, holdout_fp = fixed_observation_fingerprints(g0, Hpoints, Hnormals, Hmask, region_names)

        report["forcing_sha256"] = sha256_file(forcing_path)
        report["forcing_metadata_geometry_fingerprint"] = str(meta["geometry_fingerprint"])
        report["fixed_observations"] = {
            "collocation_fingerprint": colloc_fp,
            "holdout_fingerprint": holdout_fp,
            "collocation_point_count": int(g0.num_collocation_points),
            "holdout_point_count": int(len(Hpoints)),
            "region_names": region_names,
        }

        fit_coll, fit_hold, fit_diag = fit_synchronized_phasors(z, meta, g0, Hpoints)
        report["phasor_fit"] = {"status": "complete", **fit_diag}

        candidate_ref = source_candidate_reference(settings)
        if max(DENSITIES) > candidate_ref["candidate_points_per_arm"]:
            raise RuntimeError("design blocked: highest requested density exceeds candidate point count")
        report["source_candidate"] = {
            "status": "complete",
            "candidate_points_per_arm": candidate_ref["candidate_points_per_arm"],
            "candidate_points_total": candidate_ref["candidate_points_total"],
            "candidate_geometry_fingerprint": candidate_ref["candidate_geometry_fingerprint"],
            "candidate_surface_resolution": {
                "n_phi": int(settings["n_phi"]),
                "n_cyl": int(settings["n_cyl"]),
                "n_hemi": int(settings["n_hemi"]),
            },
            "candidate_source_ratio_per_arm": {
                density_key(n): float(candidate_ref["candidate_points_per_arm"] / n)
                for n in DENSITIES
            },
        }

        v2b_report = load_v2b_reference(v2b_path)
        report["v2b_reference_report_sha256"] = sha256_file(v2b_path)
        report["status"] = "running"

        fixed_points = np.asarray(g0.solve_obs_points, float).copy()
        fixed_normals = np.asarray(g0.solve_normals, float).copy()
        fixed_weights = np.asarray(g0.solve_weights, float).copy()
        fixed_tangent1 = np.asarray(g0.solve_tangent1, float).copy()
        fixed_tangent2 = np.asarray(g0.solve_tangent2, float).copy()
        prev = None

        for nsrc in DENSITIES:
            current_density = density_key(nsrc)
            print(f"[Step 4C3 V2C-A] building {nsrc} sources/arm on frozen candidate surface ...", flush=True)
            s = dict(settings)
            s["sources_per_arm"] = int(nsrc)
            g = c3.build_geometry(s)
            generated_collocation_count = int(g.num_collocation_points)
            bind_fixed_solve_cloud(
                g, fixed_points, fixed_normals, fixed_weights, fixed_tangent1, fixed_tangent2
            )
            if fixed_observation_fingerprints(g, Hpoints, Hnormals, Hmask, region_names) != (colloc_fp, holdout_fp):
                raise RuntimeError(f"fixed observation fingerprint changed at density {nsrc}")

            geom_diag = source_cloud_diagnostics(g, nsrc, candidate_ref, prev=prev)
            min_collocation_distance = float(np.min(cKDTree(g.ant_src_points).query(g.solve_obs_points, k=1)[0]))
            min_holdout_distance = float(np.min(cKDTree(g.ant_src_points).query(Hpoints, k=1)[0]))
            evaluator_guard = 1.0e-15
            if min(min_collocation_distance, min_holdout_distance) <= evaluator_guard:
                raise RuntimeError("pairwise evaluator singularity guard would become active; zero-softening contract violated")
            geom_diag["min_source_to_fixed_collocation_distance_m"] = min_collocation_distance
            geom_diag["min_source_to_fixed_holdout_distance_m"] = min_holdout_distance
            geom_diag["pairwise_evaluator_zero_distance_guard_m"] = evaluator_guard
            geom_diag["pairwise_evaluator_guard_active"] = False
            prev = (nsrc, g)
            dr = {
                "status": "running",
                "generated_builder_collocation_count_discarded": generated_collocation_count,
                "fixed_collocation_point_count_used": int(g.num_collocation_points),
                "fixed_collocation_fingerprint": colloc_fp,
                "fixed_holdout_fingerprint": holdout_fp,
                "source_geometry": geom_diag,
                "geometry_fingerprint_with_fixed_observations": c3.geometry_fingerprint(g),
                "frequency_results": {},
            }
            report["densities"][current_density] = dr
            if nsrc == 900 and c3.geometry_fingerprint(g) != meta["geometry_fingerprint"]:
                raise RuntimeError("900-density fixed-observation geometry no longer matches forcing fingerprint")

            for f in FREQS:
                fk = freq_key(f)
                omega = 2.0 * np.pi * float(f)
                print(f"[Step 4C3 V2C-A] density={nsrc}, f={f/1e6:.0f} MHz: assemble/solve/evaluate ...", flush=True)
                Ecoll = np.asarray(fit_coll["phasors"][float(f)], np.complex128)
                Ehold = np.asarray(fit_hold["phasors"][float(f)], np.complex128)
                e = c2.incident_coeff(Ecoll, g)

                A, opdiag = c2.assemble_retarded_harmonic(g, omega, int(args.operator_obs_chunk))
                prep = c2.prepare_complex(A, g.solve_weights)
                q = c2.solve_complex(prep, -e)
                coll_res = A @ q + e
                coll_rel = float(np.linalg.norm(coll_res) / max(float(np.linalg.norm(e)), TINY))
                sdiag = solver_diagnostics(prep)
                audit = pairwise_operator_audit(A, g, q, omega)
                del prep, coll_res
                gc.collect()

                Escat = c2.eval_retarded_field(
                    Hpoints,
                    g,
                    q,
                    omega,
                    obs_chunk=int(args.eval_obs_chunk),
                    src_chunk=int(args.eval_src_chunk),
                )
                total_hold = Ehold + Escat
                regions = v2a.complex_region_metrics(
                    total_hold, Ehold, Hnormals, Hmask, region_names
                )
                for ir, region in enumerate(region_names):
                    if region in regions:
                        regions[region]["point_count"] = int(np.sum(Hmask[ir]))
                worst = worst_region(regions)
                all_regions_pass = bool(all(bool(r["passed"]) for r in regions.values()))
                passed = bool(coll_rel < BOUNDARY_GATE and all_regions_pass)

                fr = {
                    "status": "complete",
                    "frequency_Hz": int(f),
                    "collocation_global": coll_rel,
                    "holdout_global": float(regions["global"]["gate_value"]),
                    "worst_region": worst,
                    "regions": regions,
                    "passed_collocation_and_all_holdout_regions": passed,
                    "weighted_current_rms_A_per_m": weighted_current_rms(g, q),
                    "retarded_operator_diagnostics": opdiag,
                    "operator_evaluator_consistency": audit,
                    "solver_diagnostics": sdiag,
                    "softening_m": SOFTENING_M,
                }
                if nsrc == 900:
                    fr["v2b_synchronized_baseline_reproduction"] = v2b_reproduction_check(
                        v2b_report, f, coll_rel, regions, worst
                    )
                else:
                    fr["v2b_synchronized_baseline_reproduction"] = {
                        "applicable": False,
                        "reason": "V2B reference is the frozen 900-sources/arm baseline",
                    }
                dr["frequency_results"][fk] = fr

                del A, q, Escat, total_hold
                gc.collect()

            dr["status"] = "complete"
            report["densities"][current_density] = dr

        report["status"] = "complete"
        report["failure"] = {"present": False, "type": "none", "message": "none", "traceback": "none"}
        finalize_outputs(out, report)
        return report

    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {
            "present": True,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if current_density != "none" and current_density in report["densities"]:
            d = report["densities"][current_density]
            if isinstance(d, dict) and d.get("status") != "complete":
                d["status"] = "failed"
                d["failure_reason"] = str(exc)
        failed_seen = False
        for n in DENSITIES:
            k = density_key(n)
            d = report["densities"][k]
            if d.get("status") == "failed":
                failed_seen = True
            elif failed_seen and d.get("status", "").startswith("not_run"):
                d["status"] = "not_run_due_to_prior_failure"
        finalize_outputs(out, report)
        raise


def self_test():
    # 1) 900/1200/1500-style deterministic nested FPS logic, without forcing.
    u = np.linspace(0.0, 1.0, 1800, endpoint=False)
    candidate = np.column_stack([
        np.cos(2 * np.pi * 37 * u),
        np.sin(2 * np.pi * 37 * u),
        4.0 * u,
    ])
    i900 = mp._fps_indices(candidate, 900, seed=17)
    i1200 = mp._fps_indices(candidate, 1200, seed=17)
    i1500 = mp._fps_indices(candidate, 1500, seed=17)
    nested = bool(
        np.array_equal(i900, i1200[:900])
        and np.array_equal(i1200, i1500[:1200])
    )
    if not nested:
        raise RuntimeError("self-test nested FPS failed")

    # 2) Small real production geometry, then source-density change with fixed observations.
    base_kwargs = dict(
        radius=1.0e-3,
        total_length=16.0e-3,
        feed_gap=2.0e-3,
        depth_frac=DEPTH_FRAC,
        collocation_factor=2.0,
        n_phi=12,
        n_cyl=10,
        n_hemi=4,
    )
    g12 = mp.generate_rounded_two_arm_dipole_points(sources_per_arm=12, **base_kwargs)
    fixed = (
        g12.solve_obs_points.copy(),
        g12.solve_normals.copy(),
        g12.solve_weights.copy(),
        g12.solve_tangent1.copy(),
        g12.solve_tangent2.copy(),
    )
    fixed_fp = hash_arrays_exact(
        g12.solve_obs_points, g12.solve_normals, g12.solve_weights,
        g12.solve_tangent1, g12.solve_tangent2,
    )
    g16 = mp.generate_rounded_two_arm_dipole_points(sources_per_arm=16, **base_kwargs)
    bind_fixed_solve_cloud(g16, *fixed)
    fixed_fp_16 = hash_arrays_exact(
        g16.solve_obs_points, g16.solve_normals, g16.solve_weights,
        g16.solve_tangent1, g16.solve_tangent2,
    )
    if fixed_fp_16 != fixed_fp:
        raise RuntimeError("self-test fixed observations changed after source-density change")

    # 3) Real retarded harmonic assembly vs independent pairwise evaluator.
    omega = 2 * np.pi * 3.0e6
    A, _ = c2.assemble_retarded_harmonic(g12, omega, chunk=8)
    rng = np.random.default_rng(20261003)
    q_true = (rng.normal(size=A.shape[1]) + 1j * rng.normal(size=A.shape[1])) * 1.0e-6
    Epair = c2.eval_retarded_field(
        g12.solve_obs_points,
        g12,
        q_true,
        omega,
        obs_chunk=16,
        src_chunk=16,
    )
    pair_coeff = v2b.field_to_coeff(Epair, g12.solve_tangent1, g12.solve_tangent2)
    matrix_coeff = A @ q_true
    op_pair_rel = float(
        np.linalg.norm(pair_coeff - matrix_coeff)
        / max(float(np.linalg.norm(matrix_coeff)), TINY)
    )
    if op_pair_rel > OPERATOR_EVALUATOR_TOL:
        raise RuntimeError(f"self-test operator/evaluator mismatch: {op_pair_rel}")

    # 4) Known-coefficient weighted/scaled solve/evaluate chain.
    incident = -matrix_coeff
    prep = c2.prepare_complex(A.copy(), g12.solve_weights)
    q_fit = c2.solve_complex(prep, -incident)
    solve_rel = float(
        np.linalg.norm(A @ q_fit + incident)
        / max(float(np.linalg.norm(incident)), TINY)
    )
    if solve_rel >= 5.0e-4:
        raise RuntimeError(f"self-test weighted/scaled solve residual too large: {solve_rel}")
    eval_audit = pairwise_operator_audit(A, g12, q_fit, omega, obs_count=8)

    synthetic_report = {
        "self_test_pass": True,
        "nested_900_1200_1500_style": nested,
        "fixed_observations_unchanged": fixed_fp_16 == fixed_fp,
        "operator_pairwise_relative_l2": op_pair_rel,
        "known_coefficient_solve_relative_l2": solve_rel,
        "solve_evaluate_chain_relative_l2": eval_audit["relative_l2_error"],
        "softening_m": SOFTENING_M,
        "lambda_rel": LAMBDA_REL,
        "frequencies_Hz": [int(f) for f in FREQS],
        "boundary_gate": BOUNDARY_GATE,
        "json_contract": "no null, no pending, finite numeric fields",
    }
    assert_json_clean(synthetic_report)
    print(json.dumps(synthetic_report, indent=2, allow_nan=False))
    return synthetic_report


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    report = run_formal(args)
    print(json.dumps({
        "experiment": report["experiment"],
        "status": report["status"],
        "decision_tree_branch": report["decision_tree"]["branch"],
        "minimum_all_frequency_qualified_density": report["decision_tree"]["minimum_all_frequency_qualified_density"],
        "report": str(resolve_out_path(args.out) / "step4c3_v2c_a_report.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
