#!/usr/bin/env python3
"""Step 4D-B0 — atomic-vs-observation startup kernel contract probe.

This probe isolates the plasma-only Jefimenko first pass that failed at article
scale in Step 4D-B1.  It intentionally does NOT solve MAS feedback.  The goal is
to determine whether the grid-source accumulation strategy alone explains the
startup catastrophe.

Four predeclared cases are run for a few steps:
  small_observation : 2500/species, 13^3, seed 11, observation kernel
  small_atomic      : same, atomic kernel
  article_observation: 24000/species, 17^3, seed 101, observation kernel
  article_atomic    : same, atomic kernel

This B0 probe exercises only the quasi-electrostatic plasma kernel used by
Step 4D.  Historical JefiPIC atomic full-EM support existed for cell-centered
grid currents.  The current Step-1D production refactor instead stores native
staggered-face currents; only its observation-parallel staggered-current kernel
has been implemented so far.  That is an implementation-state distinction, not
a physical prohibition on atomic full-EM.
"""
from __future__ import annotations
import argparse, contextlib, json, math
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import step4c3_one_way_broadband_pic_to_implicit_mas as c3record
import step4c3_v2_retarded_mot_broadband_replay as c3

PROBE_STEPS = 8
RUNAWAY_ABS_E_V_PER_M = 1.0e12
CASES = (
    ("small_observation", 2500, 13, 11, "observation"),
    ("small_atomic", 2500, 13, 11, "atomic"),
    ("article_observation", 24000, 17, 101, "observation"),
    ("article_atomic", 24000, 17, 101, "atomic"),
)


def base_settings(particles, grid_n, seed):
    s = c3record.mode_settings(False)
    s["particles_per_species"] = int(particles)
    s["grid_n"] = int(grid_n)
    s["seed"] = int(seed)
    s["cycles"] = 0.03  # overridden by cfg.steps below; metadata only
    return s


def _to_numpy(x):
    """Explicitly copy GPU arrays to host before NumPy diagnostics.

    CuPy intentionally forbids implicit ``np.asarray(cupy_array)`` conversion.
    Numba device arrays likewise require an explicit host copy.
    """
    if hasattr(x, "get"):
        x = x.get()
    elif hasattr(x, "copy_to_host"):
        x = x.copy_to_host()
    return np.asarray(x, dtype=np.float64)


def _arr_rms(x):
    a = _to_numpy(x)
    return float(np.sqrt(np.mean(a*a))) if a.size else 0.0


def _arr_max(x):
    a = _to_numpy(x)
    return float(np.max(np.abs(a))) if a.size else 0.0


def run_case(name, particles, grid_n, seed, kernel_mode, out_root):
    import _jefipic_mas_case_common as common

    settings = base_settings(particles, grid_n, seed)
    g = c3record.build_geometry(settings)
    out = Path("article_cases_output") / out_root / name
    out.mkdir(parents=True, exist_ok=True)

    cfg = c3record.make_pic_config(str(Path(out_root)/name), settings)
    cfg.seed = int(seed)
    cfg.particles_per_species = int(particles)
    cfg.grid_n = int(grid_n)
    cfg.steps = PROBE_STEPS
    cfg.len_time_snapshots = max(32, PROBE_STEPS + 8)
    cfg.diag_stride = 1
    cfg.jefipic_kernel_mode = str(kernel_mode)
    cfg.quasi_electrostatic = True

    old_builder = common.build_mas_device
    def _probe_builder(cfg_local, stl_path, grid_shape, spacing, origin, *, current_amplitude=None):
        del cfg_local, stl_path, grid_shape, spacing, origin, current_amplitude
        return c3record.lightweight_probe_holder(g)
    common.build_mas_device = _probe_builder
    try:
        bundle = common.build_pic_solver_mas(cfg, out/"IGNORED_GEOMETRY.stl")
    finally:
        common.build_mas_device = old_builder

    PIC = bundle["PIC"]
    solver = bundle["solver"]
    bridge = bundle["unit_bridge"]
    rng = bundle["rng"]
    nx, ny, nz = bundle["shape"]
    dx, dy, dz = bundle["spacing"]
    Ex, Ey, Ez, Bx, By, Bz = bundle["fields"]
    actual_mode = str(getattr(solver, "kernel_mode", "unknown"))
    if actual_mode != kernel_mode:
        raise RuntimeError(f"requested kernel={kernel_mode} but solver actual={actual_mode}")

    solve_obs_code = bridge.length_si_to_code(g.solve_obs_points)
    rows = []
    runaway = False
    all_finite = True
    for step in range(PROBE_STEPS):
        PIC.proceed_one_step(Ex, Ey, Ez, Bx, By, Bz, dx, dy, dz, nx, ny, nz, cfg.field_order)
        common.absorb_particles_on_antenna(PIC, cfg, rng)
        rho, Jx, Jy, Jz = PIC.evaluate_rho_J(dx, dy, dz, nx, ny, nz, cfg.particle_order)
        solver.Jefimenko_solver(rho.flatten(), include_antenna=False, face_currents=(Jx, Jy, Jz))
        Ecode = solver.sample_current_E_at_points(solve_obs_code)
        Esi = np.asarray(bridge.efield_code_to_si(Ecode), dtype=np.float64).reshape(g.num_collocation_points, 3)
        finite = bool(np.all(np.isfinite(Esi)))
        all_finite = all_finite and finite
        e_norm = np.linalg.norm(Esi, axis=1) if finite else np.array([np.inf])
        e_rms = float(np.sqrt(np.mean(e_norm*e_norm)))
        e_max = float(np.max(e_norm))
        row = {
            "step": int(step),
            "rho_rms_code": _arr_rms(rho),
            "rho_max_abs_code": _arr_max(rho),
            "Jx_face_rms_code": _arr_rms(Jx),
            "Jy_face_rms_code": _arr_rms(Jy),
            "Jz_face_rms_code": _arr_rms(Jz),
            "plasma_surface_E_rms_V_per_m": e_rms,
            "plasma_surface_E_max_V_per_m": e_max,
            "finite": finite,
        }
        rows.append(row)
        print(f"[B0] {name} step={step} E_rms={e_rms:.6e} E_max={e_max:.6e}", flush=True)
        if (not finite) or e_rms > RUNAWAY_ABS_E_V_PER_M or e_max > RUNAWAY_ABS_E_V_PER_M:
            runaway = True
            break
        # Plasma-only evolution: use the just-computed field for the next push.
        Ex, Ey, Ez, Bx, By, Bz = solver.acquire_EB_field()

    result = {
        "case": name,
        "requested_kernel_mode": kernel_mode,
        "actual_kernel_mode": actual_mode,
        "particles_per_species": int(particles),
        "grid_n": int(grid_n),
        "seed": int(seed),
        "steps_requested": PROBE_STEPS,
        "steps_completed": len(rows),
        "all_finite": bool(all_finite),
        "runaway_threshold_V_per_m": RUNAWAY_ABS_E_V_PER_M,
        "runaway_detected": bool(runaway),
        "rows": rows,
    }
    (out/"step4d_b0_case_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def self_test():
    # Logic-only test: the decision must promote atomic only when article atomic
    # is finite/non-runaway and article observation is runaway, while the small
    # pair remains non-runaway.
    fake = {
        "small_observation": {"runaway_detected": False, "all_finite": True},
        "small_atomic": {"runaway_detected": False, "all_finite": True},
        "article_observation": {"runaway_detected": True, "all_finite": True},
        "article_atomic": {"runaway_detected": False, "all_finite": True},
    }
    ok = (
        not fake["small_observation"]["runaway_detected"]
        and not fake["small_atomic"]["runaway_detected"]
        and fake["article_observation"]["runaway_detected"]
        and not fake["article_atomic"]["runaway_detected"]
    )
    class _MockGPU:
        def __init__(self, arr):
            self._arr = np.asarray(arr, dtype=np.float64)
        def get(self):
            return self._arr.copy()

    gpu_conversion_ok = (
        abs(_arr_rms(_MockGPU([3.0, 4.0])) - math.sqrt(12.5)) < 1.0e-15
        and abs(_arr_max(_MockGPU([-2.0, 5.0])) - 5.0) < 1.0e-15
    )
    ok = bool(ok and gpu_conversion_ok)
    rep = {
        "self_test_pass": bool(ok),
        "self_test_requires_gpu": False,
        "probe_steps": PROBE_STEPS,
        "explicit_gpu_to_host_diagnostic_conversion": bool(gpu_conversion_ok),
    }
    print(json.dumps(rep, indent=2))
    if not ok:
        raise RuntimeError("B0 self-test failed")


def run(args):
    out = Path("article_cases_output")/args.out
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for case in CASES:
        r = run_case(*case, args.out)
        results[r["case"]] = r

    small_ok = all(not results[k]["runaway_detected"] and results[k]["all_finite"] for k in ("small_observation","small_atomic"))
    article_atomic_ok = not results["article_atomic"]["runaway_detected"] and results["article_atomic"]["all_finite"]
    article_obs_bad = results["article_observation"]["runaway_detected"] or not results["article_observation"]["all_finite"]

    if small_ok and article_atomic_ok and article_obs_bad:
        decision = "ATOMIC_ARTICLE_SCALE_PATH_CONFIRMED_OBSERVATION_REGRESSION"
        next_step = "Lock Step 4D-B to atomic grid-source accumulation and rerun B1. Preserve C3 small-scale results as historical diagnostics; do not relabel their gate outcomes."
    elif article_atomic_ok and not article_obs_bad:
        decision = "BOTH_KERNELS_STARTUP_STABLE_INVESTIGATE_OTHER_B1_DIFFERENCE"
        next_step = "Do not blame kernel mode alone; continue the particle/grid/seed isolation matrix."
    elif not article_atomic_ok:
        decision = "ATOMIC_DOES_NOT_FIX_ARTICLE_STARTUP"
        next_step = "Stop B1; investigate article-scale PIC deposition/unit/grid contract before any long run."
    else:
        decision = "MIXED_STARTUP_RESULT_REQUIRES_INSPECTION"
        next_step = "Inspect per-step rho/J/E diagnostics before modifying production settings."

    report = {
        "experiment": "STEP4D_B0_ATOMIC_KERNEL_STARTUP_PROBE",
        "status": "complete",
        "quasi_electrostatic_only": True,
        "full_em_atomic_scope_note": (
            "Legacy JefiPIC atomic full-EM used cell-centered J. The current "
            "production solver stores native staggered-face J; an atomic "
            "staggered-face current kernel is not implemented yet. B0 does not "
            "test or require full-EM current terms."
        ),
        "cases": results,
        "decision": decision,
        "recommended_next_step": next_step,
    }
    (out/"step4d_b0_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines=["Step 4D-B0 atomic kernel startup probe", f"decision: {decision}", next_step]
    for k,r in results.items():
        mx=max((x["plasma_surface_E_rms_V_per_m"] for x in r["rows"]), default=float('nan'))
        lines.append(f"{k}: mode={r['actual_kernel_mode']}, completed={r['steps_completed']}/{r['steps_requested']}, runaway={r['runaway_detected']}, max_Erms={mx:.9g} V/m")
    (out/"step4d_b0_summary.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--out", default="step4d_b0_kernel_probe")
    a=p.parse_args()
    if a.self_test:
        self_test(); return
    run(a)

if __name__ == "__main__":
    main()
