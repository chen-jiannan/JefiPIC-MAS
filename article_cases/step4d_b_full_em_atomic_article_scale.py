#!/usr/bin/env python3
"""
Step 4D-B — full-EM Atomic article-scale representative + particle/seed/window sensitivity
for the two-way PIC/Jefimenko <-> Retarded-MOT MAS formulation.

This script intentionally does NOT alter the Step-4C3 5% qualification history.
It runs only after Step 4D-A has established stable two-way execution with the
known C3 upper-band limitation carried forward.

Workflow
--------
B1 representative:
    grid 17^3, 24000 particles/species, seed 101, 12 RF cycles.
    The final 1/2/3-cycle windows are fitted against a matched vacuum reference.
    B1 is considered ready for the full sensitivity matrix when the underlying
    two-way execution is valid, all fits are well conditioned, and the 3 MHz
    plasma/vacuum ratio relative span across the three final windows is <=20%.
    This 20% criterion is a convergence/readiness stop rule, not a physical gate.

B2 sensitivity:
    particles/species = 6000, 12000, 24000
    seeds             = 101, 202, 303
    fit windows       = final 1, 2, 3 RF cycles
    grid              = 17^3
    duration          = 12 RF cycles

The B1 24000/101 run is reused inside B2; it is not recomputed.  Remaining cases
are resumable from per-case reports.

The reviewer-facing primary estimator is predeclared as the 24000-particle,
final-2-cycle response averaged across seeds.  The 1/3-cycle fits and lower
particle counts are sensitivity diagnostics, not alternative values to cherry-pick.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))

import step4d_a_two_way_retarded_mot_closure_full_em_atomic as d4a

F_DRIVE = d4a.F_DRIVE
DT = d4a.DT
TINY = d4a.TINY

FORMAL_GRID_N = 17
FORMAL_CYCLES = 12.0
FORMAL_PARTICLES = (6000, 12000, 24000)
FORMAL_SEEDS = (101, 202, 303)
FORMAL_FIT_CYCLES = (1.0, 2.0, 3.0)
DURATION_EXTENSION_REASON = (
    "The validated 6-cycle atomic B1 execution passed stability and the 5% "
    "boundary gate, but its final 1/2/3-cycle 3 MHz plasma/vacuum ratios "
    "were 66.1366/45.5647/30.8579 with relative span 0.7743 > 0.20; "
    "duration is extended only, with all physics and estimator definitions frozen."
)
REPRESENTATIVE_PARTICLES = 24000
REPRESENTATIVE_SEED = 101
PRIMARY_FIT_CYCLES = 2.0
FIT_CONDITION_LIMIT = 100.0
B1_WINDOW_RELATIVE_SPAN_GATE = 0.20
ACCEPTED_D4A_CLASSIFICATIONS = {
    "TWO_WAY_EXECUTION_VALID_WITH_CARRIED_FORWARD_C3_SCOPE_LIMITATION",
    "TWO_WAY_EXECUTION_VALID_AND_BOUNDARY_WITHIN_PREDECLARED_5PCT",
}

_VACUUM_CACHE = {}
_LAST_VACUUM_KEY = None


def _json_clone(x):
    return json.loads(json.dumps(x))


def load_json(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def validate_step4d_a_precondition(report, accept_known_limitation):
    """Validate Step 4D-A semantically, not by one brittle classification string.

    Step 4D-B is allowed to proceed whenever the atomic Step 4D-A run is a
    complete, finite, non-runaway, valid two-way execution with the frozen
    production formulation. The current nonlinear D-A boundary may be either
    above or below the predeclared 5% gate; that does not retroactively rewrite
    the historical Step-4C3 qualification result.
    """
    if report.get("experiment") != "STEP4D_A_FULL_EM_ATOMIC_TWO_WAY_RETARDED_MOT_CLOSURE":
        raise RuntimeError("--step4d-a-report is not a Step 4D-A report")
    if report.get("status") != "complete":
        raise RuntimeError("Step 4D-A report is not complete")
    if not bool(report.get("two_way_execution_valid", False)):
        raise RuntimeError("Step 4D-A did not establish valid two-way execution")

    execution = report.get("execution", {})
    if not bool(execution.get("all_states_finite", False)):
        raise RuntimeError("Step 4D-A contains non-finite states")
    if bool(execution.get("numerical_runaway_detected", True)):
        raise RuntimeError("Step 4D-A detected numerical runaway")

    classification = report.get("classification")
    if classification not in ACCEPTED_D4A_CLASSIFICATIONS:
        raise RuntimeError(
            "Step 4D-A has an unrecognized valid-state classification: "
            f"{classification}. Accepted states are {sorted(ACCEPTED_D4A_CLASSIFICATIONS)}"
        )

    if bool(report.get("c3_qualification_relabelled_as_pass", True)):
        raise RuntimeError("Step 4D-A incorrectly relabelled the historical C3 qualification as PASS")
    if not bool(report.get("known_upper_band_limitation_carried_forward", False)):
        raise RuntimeError("Step 4D-A did not carry forward the known C3 limitation")
    if not accept_known_limitation:
        raise RuntimeError(
            "Pass --accept-known-c3-limitation to confirm that Step 4D-B keeps the "
            "historical failed C3 qualification as an explicit manuscript scope limitation."
        )

    frozen = report.get("frozen", {})
    checks = {
        "sources_per_arm": int(frozen.get("sources_per_arm", -1)) == 900,
        "source_depth_fraction": abs(float(frozen.get("source_depth_fraction", -1)) - 0.8) < 1e-15,
        "lambda_rel": abs(float(frozen.get("lambda_rel", -1)) - 1e-4) < 1e-15,
        "softening_m": abs(float(frozen.get("softening_m", -1))) < 1e-30,
        "dt_s": abs(float(frozen.get("dt_s", -1)) - 1e-9) < 1e-18,
        "boundary_gate": abs(float(frozen.get("boundary_gate", -1)) - 0.05) < 1e-15,
        "basis": report.get("production_basis") == "2-tangential-DOF continuity-consistent electric dipoles",
        "plasma_grid_kernel_mode_actual": frozen.get("plasma_grid_kernel_mode_actual") == "atomic",
        "plasma_kernel_is_full_em": not bool(frozen.get("plasma_kernel_quasi_electrostatic", True)),
        "full_em_grid_source_enabled": bool(frozen.get("full_em_grid_source_enabled", False)),
        "full_em_atomic_staggered_current_patch": bool(frozen.get("full_em_atomic_staggered_current_patch", False)),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Step 4D-A frozen-state mismatch: {checks}")

    boundary = report.get("boundary", {})
    boundary_pass = bool(boundary.get("boundary_within_predeclared_5pct", False))

    return {
        "step4d_a_classification": classification,
        "step4d_a_boundary_within_predeclared_5pct": boundary_pass,
        "known_c3_limitation_explicitly_accepted": True,
        "c3_qualification_relabelled_as_pass": False,
        "historical_c3_state_preserved": True,
        "atomic_kernel_required_and_confirmed": True,
        "frozen_checks": checks,
    }


def formal_settings(particles, seed):
    s = d4a.c3record.mode_settings(False)
    s["mode"] = "formal"
    s["cycles"] = FORMAL_CYCLES
    s["particles_per_species"] = int(particles)
    s["grid_n"] = FORMAL_GRID_N
    s["seed"] = int(seed)
    # Keep the validated D-A diagnostic cadence and C3 analysis-start convention.
    return s


def case_key(particles, seed):
    return f"N{int(particles)}_seed{int(seed)}"


def case_relout(root_out, particles, seed):
    return f"{root_out}/runs/{case_key(particles, seed)}"


def case_dir(root_out, particles, seed):
    return Path("article_cases_output") / case_relout(root_out, particles, seed)


def root_dir(root_out):
    return Path("article_cases_output") / root_out


@contextmanager
def patched_seeded_config(seed):
    original = d4a.c3record.make_pic_config

    def seeded(out_name, settings):
        cfg = original(out_name, settings)
        cfg.seed = int(seed)
        cfg.quasi_electrostatic = False
        cfg.jefipic_kernel_mode = "atomic"
        return cfg

    d4a.c3record.make_pic_config = seeded
    try:
        yield
    finally:
        d4a.c3record.make_pic_config = original


def _vacuum_key(settings, g):
    return (
        d4a.c3.geometry_fingerprint(g),
        float(settings["cycles"]),
        float(settings["analysis_start_cycles"]),
        int(settings["diag_stride"]),
    )


def captured_vacuum_reference(settings, g, Hp, Hk, prep, holdout):
    """Exact D-A vacuum recurrence plus retained M/D histories for matched windows."""
    global _LAST_VACUUM_KEY
    key = _vacuum_key(settings, g)
    if key in _VACUUM_CACHE:
        _LAST_VACUUM_KEY = key
        return _json_clone(_VACUUM_CACHE[key]["report"])

    steps = d4a.c3record.steps_from_cycles(float(settings["cycles"]))
    times = np.arange(steps, dtype=np.float64) * DT
    analysis_start_s = float(settings["analysis_start_cycles"]) / F_DRIVE
    region_names = list(holdout["region_names"])
    Hpoints = np.asarray(holdout["points"], dtype=np.float64)
    Hnormals = np.asarray(holdout["normals"], dtype=np.float64)
    Hmask = np.asarray(holdout["mask_matrix"], dtype=bool)

    ndof = 2 * g.num_sources
    q_prev = np.zeros(ndof)
    p_prev = np.zeros(ndof)
    Mhist = np.zeros((steps, 3), dtype=np.float64)
    Dhist = np.zeros((steps, 3), dtype=np.float64)
    coll_res_sum = coll_inc_sum = 0.0
    hold_res_sum = {r: 0.0 for r in region_names}
    hold_inc_sum = {r: 0.0 for r in region_names}
    hold_count = {r: 0 for r in region_names}
    closure_max = 0.0

    for step, t_s in enumerate(times):
        incident = d4a.c3.feed_at_time(g.solve_obs_points, float(t_s))
        e = d4a.c3.tangent_coeffs(incident, g.solve_tangent1, g.solve_tangent2)
        history = Hp @ p_prev + Hk @ q_prev
        q = d4a.c3.solve_fixed_ridge(prep, -(e + history))
        p = p_prev + 0.5 * DT * (q_prev + q)
        closure_max = max(
            closure_max,
            d4a.coefficient_temporal_closure_rel(g, p_prev, q_prev, p, q),
        )
        residual = e + history + d4a.c3.operator_now_coeff(prep, q)
        if t_s >= analysis_start_s:
            coll_res_sum += float(np.dot(residual, residual))
            coll_inc_sum += float(np.dot(e, e))
        if step % int(settings["diag_stride"]) == 0 and t_s >= analysis_start_s:
            self_H, _ = d4a.c3.evaluate_retarded_mot_field(Hpoints, g, p_prev, q_prev, q)
            inc_H = d4a.c3.feed_at_time(Hpoints, float(t_s))
            total_H = inc_H + self_H
            rsq = d4a.c3.tangential_sq(total_H, Hnormals)
            isq = d4a.c3.tangential_sq(inc_H, Hnormals)
            for ir, region in enumerate(region_names):
                m = Hmask[ir]
                if np.any(m):
                    hold_res_sum[region] += float(np.sum(rsq[m]))
                    hold_inc_sum[region] += float(np.sum(isq[m]))
                    hold_count[region] += int(np.count_nonzero(m))
        M, D, _, _ = d4a.source_moments(g, q, p)
        Mhist[step] = M
        Dhist[step] = D
        q_prev, p_prev = q, p

    coll = math.sqrt(coll_res_sum / max(coll_inc_sum, TINY))
    regions, worst = d4a.aggregate_holdout_rows(
        hold_res_sum, hold_inc_sum, hold_count, region_names
    )
    response = d4a.fit_response(times, Mhist, Dhist, analysis_start_s)
    report = {
        "collocation_rel_l2": float(coll),
        "independent_holdout": {"regions": regions, "worst_gate": worst},
        "temporal_closure_max_relative": float(closure_max),
        "response": response,
    }
    _VACUUM_CACHE[key] = {
        "report": _json_clone(report),
        "time_s": times.copy(),
        "current_moment_A_m": Mhist.copy(),
        "polarization_moment_C_m": Dhist.copy(),
    }
    _LAST_VACUUM_KEY = key
    return _json_clone(report)


@contextmanager
def patched_vacuum_capture():
    original = d4a.run_vacuum_reference
    d4a.run_vacuum_reference = captured_vacuum_reference
    try:
        yield
    finally:
        d4a.run_vacuum_reference = original


def persist_vacuum_cache(root_out):
    if _LAST_VACUUM_KEY is None or _LAST_VACUUM_KEY not in _VACUUM_CACHE:
        raise RuntimeError("vacuum history cache is empty")
    root = root_dir(root_out)
    root.mkdir(parents=True, exist_ok=True)
    v = _VACUUM_CACHE[_LAST_VACUUM_KEY]
    np.savez_compressed(
        root / "step4d_b_vacuum_history.npz",
        time_s=v["time_s"],
        current_moment_A_m=v["current_moment_A_m"],
        polarization_moment_C_m=v["polarization_moment_C_m"],
    )
    (root / "step4d_b_vacuum_reference.json").write_text(
        json.dumps(v["report"], indent=2) + "\n", encoding="utf-8"
    )


def load_vacuum_cache(root_out, settings, g):
    global _LAST_VACUUM_KEY
    root = root_dir(root_out)
    npz = root / "step4d_b_vacuum_history.npz"
    js = root / "step4d_b_vacuum_reference.json"
    if not npz.exists() or not js.exists():
        return False
    z = np.load(npz)
    report = load_json(js)
    key = _vacuum_key(settings, g)
    _VACUUM_CACHE[key] = {
        "report": report,
        "time_s": np.asarray(z["time_s"], dtype=np.float64),
        "current_moment_A_m": np.asarray(z["current_moment_A_m"], dtype=np.float64),
        "polarization_moment_C_m": np.asarray(z["polarization_moment_C_m"], dtype=np.float64),
    }
    _LAST_VACUUM_KEY = key
    return True


def fit_final_window(times, current_moment, polarization_moment, fit_cycles):
    t = np.asarray(times, dtype=np.float64)
    M = np.asarray(current_moment, dtype=np.float64)
    D = np.asarray(polarization_moment, dtype=np.float64)
    if len(t) < 8:
        raise RuntimeError("too few samples for response-window fit")
    t_end = float(t[-1])
    start = t_end - float(fit_cycles) / F_DRIVE
    mask = t >= start - 0.5 * DT
    if int(np.count_nonzero(mask)) < 8:
        raise RuntimeError(f"too few samples in final {fit_cycles:g}-cycle window")
    fm = d4a.v2a.fit_multitone(t[mask], M[mask])
    fd = d4a.v2a.fit_multitone(t[mask], D[mask])
    out = {
        "fit_cycles": float(fit_cycles),
        "start_time_s": float(t[mask][0]),
        "end_time_s": t_end,
        "num_samples": int(np.count_nonzero(mask)),
        "current_fit_condition_number": float(fm["condition_number"]),
        "polarization_fit_condition_number": float(fd["condition_number"]),
        "frequencies": {},
    }
    for f in (3.0e6, 6.0e6, 9.0e6):
        Mh = np.asarray(fm["phasors"][f], np.complex128).reshape(3)
        Dh = np.asarray(fd["phasors"][f], np.complex128).reshape(3)
        out["frequencies"][f"{f/1e6:.0f}MHz"] = {
            "axial_current_moment_A_m": float(abs(Mh[2])),
            "current_moment_norm_A_m": float(np.linalg.norm(Mh)),
            "axial_polarization_moment_C_m": float(abs(Dh[2])),
            "polarization_moment_norm_C_m": float(np.linalg.norm(Dh)),
            "continuity_harmonic_ratio_abs_iwD_over_M": float(
                abs(1j * 2.0 * np.pi * f * Dh[2]) / max(abs(Mh[2]), TINY)
            ),
        }
    return out


def _vacuum_history():
    if _LAST_VACUUM_KEY is None or _LAST_VACUUM_KEY not in _VACUUM_CACHE:
        raise RuntimeError("matched vacuum history is unavailable")
    return _VACUUM_CACHE[_LAST_VACUUM_KEY]


def postprocess_case_windows(case_directory):
    diag_path = Path(case_directory) / "step4d_a_full_em_atomic_diagnostics.npz"
    raw_report_path = Path(case_directory) / "step4d_a_full_em_atomic_report.json"
    if not diag_path.exists() or not raw_report_path.exists():
        raise RuntimeError(f"incomplete raw Step 4D-A case directory: {case_directory}")
    z = np.load(diag_path)
    t = np.asarray(z["time_s"], dtype=np.float64)
    M = np.asarray(z["current_moment_A_m"], dtype=np.float64)
    D = np.asarray(z["polarization_moment_C_m"], dtype=np.float64)
    vacuum = _vacuum_history()
    rows = []
    windows = {}
    for fc in FORMAL_FIT_CYCLES:
        plasma_fit = fit_final_window(t, M, D, fc)
        vacuum_fit = fit_final_window(
            vacuum["time_s"], vacuum["current_moment_A_m"],
            vacuum["polarization_moment_C_m"], fc
        )
        p3 = plasma_fit["frequencies"]["3MHz"]["axial_current_moment_A_m"]
        v3 = vacuum_fit["frequencies"]["3MHz"]["axial_current_moment_A_m"]
        ratio = float(p3 / max(v3, TINY))
        windows[f"{fc:g}cycle"] = {
            "fit_cycles": fc,
            "plasma": plasma_fit,
            "vacuum": vacuum_fit,
            "3MHz_plasma_to_vacuum_axial_current_moment_ratio": ratio,
        }
        rows.append({
            "fit_cycles": fc,
            "3MHz_plasma_abs_A_m": p3,
            "3MHz_vacuum_abs_A_m": v3,
            "3MHz_plasma_to_vacuum_ratio": ratio,
            "3MHz_plasma_continuity_abs_iwD_over_M": plasma_fit["frequencies"]["3MHz"]["continuity_harmonic_ratio_abs_iwD_over_M"],
            "3MHz_vacuum_continuity_abs_iwD_over_M": vacuum_fit["frequencies"]["3MHz"]["continuity_harmonic_ratio_abs_iwD_over_M"],
            "plasma_fit_condition_number": plasma_fit["current_fit_condition_number"],
            "vacuum_fit_condition_number": vacuum_fit["current_fit_condition_number"],
        })
    return windows, rows, load_json(raw_report_path)


def b1_window_diagnostics(rows):
    ratios = np.asarray([float(r["3MHz_plasma_to_vacuum_ratio"]) for r in rows], dtype=float)
    median = float(np.median(ratios))
    rel_span = float((np.max(ratios) - np.min(ratios)) / max(abs(median), TINY))
    max_cond = float(max(
        max(float(r["plasma_fit_condition_number"]), float(r["vacuum_fit_condition_number"]))
        for r in rows
    ))
    return {
        "ratio_min": float(np.min(ratios)),
        "ratio_max": float(np.max(ratios)),
        "ratio_median": median,
        "ratio_relative_span": rel_span,
        "ratio_relative_span_gate": B1_WINDOW_RELATIVE_SPAN_GATE,
        "max_fit_condition_number": max_cond,
        "fit_condition_limit": FIT_CONDITION_LIMIT,
        "window_stability_ready": bool(
            np.all(np.isfinite(ratios))
            and rel_span <= B1_WINDOW_RELATIVE_SPAN_GATE
            and max_cond < FIT_CONDITION_LIMIT
        ),
    }


def free_gpu_memory():
    gc.collect()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def run_case(root_out, particles, seed, d4a_precondition, force_rerun=False):
    cdir = case_dir(root_out, particles, seed)
    cdir.mkdir(parents=True, exist_ok=True)
    case_report_path = cdir / "step4d_b_case_report.json"
    settings = formal_settings(particles, seed)
    g = d4a.c3.build_geometry(settings)
    load_vacuum_cache(root_out, settings, g)

    if case_report_path.exists() and not force_rerun:
        cached = load_json(case_report_path)
        meta = cached.get("case", {})
        if (
            int(meta.get("particles_per_species", -1)) == int(particles)
            and int(meta.get("seed", -1)) == int(seed)
            and int(meta.get("grid_n", -1)) == FORMAL_GRID_N
            and abs(float(meta.get("cycles", -1)) - FORMAL_CYCLES) < 1e-15
            and cached.get("status") == "complete"
        ):
            print(f"[Step 4D-B] reusing completed case {case_key(particles, seed)}", flush=True)
            return cached

    args = SimpleNamespace(out=case_relout(root_out, particles, seed))
    print(
        f"[Step 4D-B] running {case_key(particles, seed)}: "
        f"particles/species={particles}, seed={seed}, grid={FORMAL_GRID_N}^3, cycles={FORMAL_CYCLES:g}",
        flush=True,
    )
    with patched_seeded_config(seed), patched_vacuum_capture():
        raw = d4a.run_two_way(args, settings, g, d4a_precondition)
    persist_vacuum_cache(root_out)
    windows, window_rows, raw_disk = postprocess_case_windows(cdir)
    if raw_disk.get("classification") != raw.get("classification"):
        raise RuntimeError("raw report mismatch after Step 4D-A case write")
    result = {
        "experiment": "STEP4D_B_FULL_EM_ATOMIC_CASE",
        "status": "complete",
        "case": {
            "particles_per_species": int(particles),
            "seed": int(seed),
            "grid_n": FORMAL_GRID_N,
            "cycles": FORMAL_CYCLES,
            "fit_windows_cycles": list(FORMAL_FIT_CYCLES),
            "duration_extension_reason": DURATION_EXTENSION_REASON,
        },
        "two_way_execution_valid": bool(raw["two_way_execution_valid"]),
        "classification": raw["classification"],
        "numerical_runaway_detected": bool(raw.get("execution", {}).get("numerical_runaway_detected", False)),
        "plasma_grid_kernel_mode_actual": raw.get("frozen", {}).get("plasma_grid_kernel_mode_actual", "unknown"),
        "boundary_within_predeclared_5pct": bool(raw["boundary"]["boundary_within_predeclared_5pct"]),
        "worst_holdout": raw["boundary"]["independent_holdout"]["worst_gate"],
        "aggregate_collocation_rel_l2": float(raw["boundary"]["time_aggregated_collocation_rel_l2"]),
        "temporal_closure_max_relative": float(raw["execution"]["coefficient_temporal_closure_max_relative"]),
        "homogeneous_tail_growth": float(raw["homogeneous_mas_tail_from_coupled_final_state"]["median_growth_factor_last_window"]),
        "windows": windows,
        "raw_step4d_a_report": "step4d_a_full_em_atomic_report.json",
        "field_model": "full_em",
        "grid_source_backend": "atomic",
        "known_c3_limitation_carried_forward": True,
        "c3_qualification_relabelled_as_pass": False,
    }
    case_report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (cdir / "step4d_b_window_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(window_rows[0].keys()))
        w.writeheader(); w.writerows(window_rows)
    free_gpu_memory()
    return result


def run_representative(args, pre):
    root = root_dir(args.out)
    root.mkdir(parents=True, exist_ok=True)
    result = run_case(
        args.out, REPRESENTATIVE_PARTICLES, REPRESENTATIVE_SEED,
        pre, force_rerun=bool(args.force_rerun)
    )
    rows = []
    for key in ("1cycle", "2cycle", "3cycle"):
        w = result["windows"][key]
        pf = w["plasma"]
        vf = w["vacuum"]
        rows.append({
            "fit_cycles": float(w["fit_cycles"]),
            "3MHz_plasma_abs_A_m": pf["frequencies"]["3MHz"]["axial_current_moment_A_m"],
            "3MHz_vacuum_abs_A_m": vf["frequencies"]["3MHz"]["axial_current_moment_A_m"],
            "3MHz_plasma_to_vacuum_ratio": w["3MHz_plasma_to_vacuum_axial_current_moment_ratio"],
            "3MHz_plasma_continuity_abs_iwD_over_M": pf["frequencies"]["3MHz"]["continuity_harmonic_ratio_abs_iwD_over_M"],
            "3MHz_vacuum_continuity_abs_iwD_over_M": vf["frequencies"]["3MHz"]["continuity_harmonic_ratio_abs_iwD_over_M"],
            "plasma_fit_condition_number": pf["current_fit_condition_number"],
            "vacuum_fit_condition_number": vf["current_fit_condition_number"],
        })
    diag = b1_window_diagnostics(rows)
    ready = bool(result["two_way_execution_valid"] and diag["window_stability_ready"])
    report = {
        "experiment": "STEP4D_B1_ARTICLE_SCALE_REPRESENTATIVE",
        "status": "complete",
        "case": result["case"],
        "two_way_execution_valid": result["two_way_execution_valid"],
        "boundary_within_predeclared_5pct": result["boundary_within_predeclared_5pct"],
        "worst_holdout": result["worst_holdout"],
        "window_diagnostics": diag,
        "ready_for_b2_sensitivity": ready,
        "known_c3_limitation_carried_forward": True,
        "c3_qualification_relabelled_as_pass": False,
        "next_step": (
            "Run Step 4D-B2 particle/seed/window sensitivity."
            if ready else
            (
                "Do not extend duration. Diagnose the startup/kernel failure before any B2 run."
                if not result["two_way_execution_valid"] else
                "Do not start the 3x3 sensitivity matrix yet; extend only run duration until final-window response is converged. Do not retune MAS physics."
            )
        ),
    }
    (root / "step4d_b_b1_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (root / "step4d_b_b1_window_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = [
        "Step 4D-B1 article-scale representative",
        f"two_way_execution_valid: {result['two_way_execution_valid']}",
        f"boundary_within_predeclared_5pct: {result['boundary_within_predeclared_5pct']}",
        f"worst_holdout: {result['worst_holdout']['gate_value']:.9g} ({result['worst_holdout']['region']})",
        f"ratio_relative_span_1_2_3cycle: {diag['ratio_relative_span']:.9g}",
        f"readiness_span_gate: {B1_WINDOW_RELATIVE_SPAN_GATE:.9g}",
        f"ready_for_b2_sensitivity: {ready}",
        "C3 qualification relabelled as pass: False",
    ]
    (root / "step4d_b_b1_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def flatten_case_rows(case_report):
    c = case_report["case"]
    rows = []
    for key, win in case_report["windows"].items():
        pf = win["plasma"]
        vf = win["vacuum"]
        rows.append({
            "particles_per_species": int(c["particles_per_species"]),
            "seed": int(c["seed"]),
            "fit_cycles": float(win["fit_cycles"]),
            "grid_n": int(c["grid_n"]),
            "cycles_run": float(c["cycles"]),
            "two_way_execution_valid": bool(case_report["two_way_execution_valid"]),
            "boundary_within_predeclared_5pct": bool(case_report["boundary_within_predeclared_5pct"]),
            "worst_holdout_gate_value": float(case_report["worst_holdout"]["gate_value"]),
            "worst_holdout_region": case_report["worst_holdout"]["region"],
            "aggregate_collocation_rel_l2": float(case_report["aggregate_collocation_rel_l2"]),
            "3MHz_plasma_abs_A_m": float(pf["frequencies"]["3MHz"]["axial_current_moment_A_m"]),
            "3MHz_vacuum_abs_A_m": float(vf["frequencies"]["3MHz"]["axial_current_moment_A_m"]),
            "3MHz_plasma_to_vacuum_ratio": float(win["3MHz_plasma_to_vacuum_axial_current_moment_ratio"]),
            "3MHz_plasma_continuity_abs_iwD_over_M": float(pf["frequencies"]["3MHz"]["continuity_harmonic_ratio_abs_iwD_over_M"]),
            "plasma_fit_condition_number": float(pf["current_fit_condition_number"]),
        })
    return rows


def summarize_sensitivity(rows):
    groups = {}
    for r in rows:
        key = (int(r["particles_per_species"]), float(r["fit_cycles"]))
        groups.setdefault(key, []).append(r)
    out = []
    for (particles, fit_cycles), group in sorted(groups.items()):
        ratios = np.asarray([float(r["3MHz_plasma_to_vacuum_ratio"]) for r in group], dtype=float)
        pamps = np.asarray([float(r["3MHz_plasma_abs_A_m"]) for r in group], dtype=float)
        worst = np.asarray([float(r["worst_holdout_gate_value"]) for r in group], dtype=float)
        out.append({
            "particles_per_species": particles,
            "fit_cycles": fit_cycles,
            "n_seeds": len(group),
            "ratio_mean": float(np.mean(ratios)),
            "ratio_std": float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0,
            "ratio_cv_percent": float(100*np.std(ratios, ddof=1)/abs(np.mean(ratios))) if len(ratios) > 1 and abs(np.mean(ratios)) > 0 else 0.0,
            "ratio_min": float(np.min(ratios)),
            "ratio_max": float(np.max(ratios)),
            "plasma_abs_A_m_mean": float(np.mean(pamps)),
            "plasma_abs_A_m_std": float(np.std(pamps, ddof=1)) if len(pamps) > 1 else 0.0,
            "worst_holdout_gate_mean": float(np.mean(worst)),
            "worst_holdout_gate_max": float(np.max(worst)),
            "all_two_way_execution_valid": bool(all(bool(r["two_way_execution_valid"]) for r in group)),
        })
    return out


def run_sensitivity(args, pre):
    root = root_dir(args.out)
    b1_path = root / "step4d_b_b1_report.json"
    if not b1_path.exists():
        raise RuntimeError("Step 4D-B1 report is missing; run --stage representative first")
    b1 = load_json(b1_path)
    if not bool(b1.get("ready_for_b2_sensitivity", False)) and not bool(args.accept_b1_window_spread):
        raise RuntimeError(
            "B1 final-window response did not meet the predeclared 20% readiness span. "
            "Do not spend the 3x3 sensitivity matrix yet. Extend duration only, or pass "
            "--accept-b1-window-spread if you deliberately want diagnostic B2 data anyway."
        )

    all_cases = []
    for particles in FORMAL_PARTICLES:
        for seed in FORMAL_SEEDS:
            all_cases.append(run_case(
                args.out, particles, seed, pre,
                force_rerun=bool(args.force_rerun)
            ))

    rows = []
    for cr in all_cases:
        rows.extend(flatten_case_rows(cr))
    summaries = summarize_sensitivity(rows)

    with (root / "step4d_b_sensitivity_runs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with (root / "step4d_b_sensitivity_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys())); w.writeheader(); w.writerows(summaries)

    primary_group = [
        r for r in rows
        if int(r["particles_per_species"]) == REPRESENTATIVE_PARTICLES
        and abs(float(r["fit_cycles"]) - PRIMARY_FIT_CYCLES) < 1e-12
    ]
    primary_ratios = np.asarray([r["3MHz_plasma_to_vacuum_ratio"] for r in primary_group], dtype=float)
    if len(primary_ratios) != len(FORMAL_SEEDS):
        raise RuntimeError("primary 24000-particle/2-cycle group does not contain all three seeds")

    highN = [r for r in rows if int(r["particles_per_species"]) == REPRESENTATIVE_PARTICLES]
    highN_window_means = {}
    for fc in FORMAL_FIT_CYCLES:
        vals = np.asarray([
            r["3MHz_plasma_to_vacuum_ratio"] for r in highN
            if abs(float(r["fit_cycles"]) - fc) < 1e-12
        ], dtype=float)
        highN_window_means[f"{fc:g}cycle"] = float(np.mean(vals))
    wm = np.asarray(list(highN_window_means.values()), dtype=float)
    highN_window_span = float((np.max(wm)-np.min(wm))/max(abs(float(np.median(wm))), TINY))

    all_exec = bool(all(bool(c["two_way_execution_valid"]) for c in all_cases))
    final = {
        "experiment": "STEP4D_B_ARTICLE_SCALE_SENSITIVITY",
        "status": "complete",
        "design": {
            "grid_n": FORMAL_GRID_N,
            "cycles": FORMAL_CYCLES,
            "particles_per_species": list(FORMAL_PARTICLES),
            "seeds": list(FORMAL_SEEDS),
            "fit_windows_cycles": list(FORMAL_FIT_CYCLES),
            "primary_estimator": "24000 particles/species, final 2-cycle fit, mean across seeds 101/202/303",
            "representative_case_reused": case_key(REPRESENTATIVE_PARTICLES, REPRESENTATIVE_SEED),
        },
        "all_two_way_executions_valid": all_exec,
        "known_c3_limitation_carried_forward": True,
        "c3_qualification_relabelled_as_pass": False,
        "primary_reviewer_facing_response": {
            "observable": "3 MHz plasma/vacuum axial auxiliary electric-current moment ratio",
            "units": "dimensionless",
            "not_a_port_current": True,
            "not_impedance_or_admittance": True,
            "not_S_parameter": True,
            "particles_per_species": REPRESENTATIVE_PARTICLES,
            "fit_cycles": PRIMARY_FIT_CYCLES,
            "n_seeds": len(primary_ratios),
            "mean_ratio": float(np.mean(primary_ratios)),
            "std_ratio": float(np.std(primary_ratios, ddof=1)),
            "cv_percent": float(100*np.std(primary_ratios, ddof=1)/abs(np.mean(primary_ratios))) if abs(np.mean(primary_ratios)) > 0 else float("inf"),
            "min_ratio": float(np.min(primary_ratios)),
            "max_ratio": float(np.max(primary_ratios)),
        },
        "high_particle_window_sensitivity": {
            "mean_ratio_by_window": highN_window_means,
            "relative_span_across_window_means": highN_window_span,
        },
        "article_response_values_ready_for_manuscript": bool(all_exec),
        "interpretation_rule": (
            "Report the predeclared 24000-particle/2-cycle seed ensemble as the primary response, "
            "and report particle/seed/window variation rather than selecting the most favorable case. "
            "The known C3 upper-band boundary limitation remains explicit."
        ),
        "next_step": (
            "Freeze numerical results and revise manuscript/reviewer response (Major-2, Major-3, Major-4, terminology/units/scope)."
            if all_exec else
            "Diagnose failed sensitivity cases before manuscript-finalizing the RF response."
        ),
    }
    (root / "step4d_b_report.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    summary = [
        "Step 4D-B article-scale particle/seed/window sensitivity",
        f"all_two_way_executions_valid: {all_exec}",
        f"primary mean ratio (N=24000, final 2 cycles, 3 seeds): {final['primary_reviewer_facing_response']['mean_ratio']:.9g}",
        f"primary std ratio: {final['primary_reviewer_facing_response']['std_ratio']:.9g}",
        f"primary CV (%): {final['primary_reviewer_facing_response']['cv_percent']:.9g}",
        f"high-N window mean relative span: {highN_window_span:.9g}",
        "C3 qualification relabelled as pass: False",
        f"article response values ready for manuscript: {final['article_response_values_ready_for_manuscript']}",
    ]
    (root / "step4d_b_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2))
    return final


def self_test():
    # Synthetic exact multitone: validates final-window extraction and summary logic.
    steps = d4a.c3record.steps_from_cycles(FORMAL_CYCLES)
    t = np.arange(steps, dtype=np.float64) * DT
    f = F_DRIVE
    omega = 2*np.pi*f
    # Construct Mz with 3/6/9 MHz components. D is chosen to satisfy i*w*D=M per harmonic.
    M = np.zeros((steps, 3), dtype=np.float64)
    D = np.zeros((steps, 3), dtype=np.float64)
    components = [(3e6, 2.0), (6e6, 0.4), (9e6, 0.2)]
    for ff, amp in components:
        ww = 2*np.pi*ff
        M[:, 2] += amp*np.cos(ww*t + 0.21)
        # Re{Dhat exp(iwt)} with Dhat=Mhat/(iw): (amp/w) sin(wt+phase)
        D[:, 2] += (amp/ww)*np.sin(ww*t + 0.21)
    fits = [fit_final_window(t, M, D, fc) for fc in FORMAL_FIT_CYCLES]
    amps = np.asarray([x["frequencies"]["3MHz"]["axial_current_moment_A_m"] for x in fits])
    cont = np.asarray([x["frequencies"]["3MHz"]["continuity_harmonic_ratio_abs_iwD_over_M"] for x in fits])
    fit_ok = bool(np.max(np.abs(amps-2.0)) < 1e-10 and np.max(np.abs(cont-1.0)) < 1e-10)

    synthetic_rows = []
    for n in FORMAL_PARTICLES:
        for seed, ratio in zip(FORMAL_SEEDS, (50.0, 55.0, 60.0)):
            for fc in FORMAL_FIT_CYCLES:
                synthetic_rows.append({
                    "particles_per_species": n,
                    "seed": seed,
                    "fit_cycles": fc,
                    "3MHz_plasma_to_vacuum_ratio": ratio + 0.1*fc,
                    "3MHz_plasma_abs_A_m": 1e-9*(ratio+0.1*fc),
                    "worst_holdout_gate_value": 0.052,
                    "two_way_execution_valid": True,
                })
    sums = summarize_sensitivity(synthetic_rows)
    summary_ok = bool(len(sums) == 9 and all(int(x["n_seeds"]) == 3 for x in sums))

    fake_base = {
        "experiment": "STEP4D_A_FULL_EM_ATOMIC_TWO_WAY_RETARDED_MOT_CLOSURE",
        "status": "complete",
        "two_way_execution_valid": True,
        "c3_qualification_relabelled_as_pass": False,
        "known_upper_band_limitation_carried_forward": True,
        "production_basis": "2-tangential-DOF continuity-consistent electric dipoles",
        "execution": {
            "all_states_finite": True,
            "numerical_runaway_detected": False,
        },
        "frozen": {
            "sources_per_arm": 900,
            "source_depth_fraction": 0.8,
            "lambda_rel": 1e-4,
            "softening_m": 0.0,
            "dt_s": 1e-9,
            "boundary_gate": 0.05,
            "plasma_kernel_quasi_electrostatic": False,
            "full_em_grid_source_enabled": True,
            "full_em_atomic_staggered_current_patch": True,
            "plasma_grid_kernel_mode_actual": "atomic",
        },
    }
    fake_failgate = _json_clone(fake_base)
    fake_failgate["classification"] = "TWO_WAY_EXECUTION_VALID_WITH_CARRIED_FORWARD_C3_SCOPE_LIMITATION"
    fake_failgate["boundary"] = {"boundary_within_predeclared_5pct": False}
    fake_passgate = _json_clone(fake_base)
    fake_passgate["classification"] = "TWO_WAY_EXECUTION_VALID_AND_BOUNDARY_WITHIN_PREDECLARED_5PCT"
    fake_passgate["boundary"] = {"boundary_within_predeclared_5pct": True}

    pre0 = validate_step4d_a_precondition(fake_failgate, True)
    pre1 = validate_step4d_a_precondition(fake_passgate, True)
    pre_ok = bool(
        pre0["known_c3_limitation_explicitly_accepted"]
        and pre1["known_c3_limitation_explicitly_accepted"]
        and not pre0["c3_qualification_relabelled_as_pass"]
        and not pre1["c3_qualification_relabelled_as_pass"]
        and not pre0["step4d_a_boundary_within_predeclared_5pct"]
        and pre1["step4d_a_boundary_within_predeclared_5pct"]
        and pre0["atomic_kernel_required_and_confirmed"]
        and pre1["atomic_kernel_required_and_confirmed"]
    )
    settings = formal_settings(24000, 101)
    # Test the seed wrapper without importing the GPU-dependent case-common module.
    original_make = d4a.c3record.make_pic_config
    d4a.c3record.make_pic_config = lambda out_name, st: SimpleNamespace(
        seed=11, grid_n=int(st["grid_n"]), particles_per_species=int(st["particles_per_species"]), quasi_electrostatic=False, jefipic_kernel_mode="atomic"
    )
    try:
        with patched_seeded_config(101):
            cfg = d4a.c3record.make_pic_config("selftest", settings)
    finally:
        d4a.c3record.make_pic_config = original_make
    seed_ok = bool(int(cfg.seed) == 101 and int(cfg.grid_n) == 17 and int(cfg.particles_per_species) == 24000)

    passed = bool(fit_ok and summary_ok and pre_ok and seed_ok)
    report = {
        "self_test_pass": passed,
        "self_test_requires_gpu": False,
        "self_test_requires_forcing": False,
        "final_window_fit_exact_multitone": fit_ok,
        "max_3MHz_amplitude_abs_error": float(np.max(np.abs(amps-2.0))),
        "max_3MHz_continuity_abs_error": float(np.max(np.abs(cont-1.0))),
        "sensitivity_group_count": len(sums),
        "sensitivity_summary_grouping_ok": summary_ok,
        "step4d_a_semantic_precondition_accepts_both_valid_boundary_states": pre_ok,
        "seed_override_ok": seed_ok,
        "formal_design": {
            "grid_n": FORMAL_GRID_N,
            "cycles": FORMAL_CYCLES,
            "particles": list(FORMAL_PARTICLES),
            "seeds": list(FORMAL_SEEDS),
            "fit_cycles": list(FORMAL_FIT_CYCLES),
        },
    }
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("Step 4D-B self-test failed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--stage", choices=("representative", "sensitivity", "all"), default="representative")
    p.add_argument("--step4d-a-report", default=None)
    p.add_argument("--accept-known-c3-limitation", action="store_true")
    p.add_argument("--accept-b1-window-spread", action="store_true")
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--out", default="step4d_b_full_em_atomic_formal")
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    if a.step4d_a_report is None:
        raise RuntimeError("--step4d-a-report is required")
    d4a_report = load_json(a.step4d_a_report)
    pre = validate_step4d_a_precondition(d4a_report, bool(a.accept_known_c3_limitation))
    if a.stage in ("representative", "all"):
        b1 = run_representative(a, pre)
        if a.stage == "all" and not b1["ready_for_b2_sensitivity"] and not a.accept_b1_window_spread:
            raise RuntimeError("B1 is not ready for B2; stopping before sensitivity matrix")
    if a.stage in ("sensitivity", "all"):
        run_sensitivity(a, pre)


if __name__ == "__main__":
    main()
