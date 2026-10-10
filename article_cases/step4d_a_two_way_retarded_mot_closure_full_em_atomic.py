#!/usr/bin/env python3
"""
Step 4D-A full-EM validation variant — two-way PIC <-> Retarded-MOT MAS closure under an
explicitly carried-forward Step-4C3 upper-band limitation.

Why this script exists
----------------------
The predeclared Step-4C3 5% qualification tree was NOT passed.  V2C/V2D/V2E
showed that density, geometry-only placement, and a global 3-DOF basis
augmentation do not remove the 9 MHz independent-holdout error.  The decision
for the manuscript is therefore to retain the production 2-tangential-DOF MAS
basis and report the upper-band error as a quantified scope limitation rather
than continue post-hoc tuning.

This script does NOT relabel C3 as PASS.  Instead it answers the next reviewer-
facing question:

    Can the corrected causal Retarded-MOT conductor response be closed back
    into the nonlinear PIC/Jefimenko loop without reviving the retired
    active/passive cancellation or the unstable free polarization recurrence?

Drive convention
----------------
The retired prescribed all-surface active current is NOT used.  A localized
voltage-like/feed electric field is imposed consistently in two places:

1. it appears as the impressed incident field in the PEC boundary RHS; and
2. the same analytic E_feed(x,t) is added to the PIC grid field used for the
   next particle push.

The MAS unknown is therefore only the induced 2-tangential-DOF conductor
response.  No unconstrained passive solve can cancel a prescribed active source
coefficient because no active coefficient is prescribed.

The plasma field is obtained by a first Jefimenko pass with include_antenna=False.
The Retarded-MOT boundary solve then determines K^n/P^n.  Those conductor states
are pushed to the SAME EM history node, the grid field is recomputed with
include_antenna=True, and the impressed feed field is added for the next PIC
particle push.  This is the intended two-pass/two-way closure.

Important interpretation
------------------------
A successful Step 4D-A run is an execution/stability result, not a retroactive
3--9 MHz <5% qualification.  The report preserves both booleans:

- two_way_execution_valid
- boundary_within_predeclared_5pct

If the first is true and the second is false, the correct classification is:
TWO_WAY_EXECUTION_VALID_WITH_CARRIED_FORWARD_C3_SCOPE_LIMITATION.

The RF observable is an equivalent axial electric-current moment
    M_z(t) = sum_j w_j K_{j,z}(t)   [A m]
and its polarization analogue
    D_z(t) = sum_j w_j P_{j,z}(t)   [C m].
These are auxiliary-source moments, not port current, impedance, admittance, or
S-parameters.  The low-particle Step 4D-A result is not manuscript-final; an
article-scale representative/ensemble rerun is a later Step 4D-B task if this
closure is stable.
"""
from __future__ import annotations

import argparse
import csv
import sys
import gc
import json
import math
from pathlib import Path

import numpy as np

# Ensure project root and article_cases are importable when executed as a file.
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))

import step4c3_one_way_broadband_pic_to_implicit_mas as c3record
import step4c3_v2_retarded_mot_broadband_replay as c3
import step4c3_v2a_feed_transition_residual_diagnostic as v2a

DT = c3.DT
F_DRIVE = c3.F_DRIVE
BOUNDARY_GATE = c3.BOUNDARY_GATE
WEAK_REGION_THRESHOLD = c3.WEAK_REGION_THRESHOLD
TAIL_GROWTH_GATE = c3.TAIL_GROWTH_GATE
PLASMA_E_RUNAWAY_GATE_V_PER_M = 1.0e12
TAIL_STEPS = c3.TAIL_STEPS
TAIL_WINDOW = c3.TAIL_WINDOW
TINY = c3.TINY
CLOSURE_REL_GATE = 1.0e-10
EXPECTED_V2E_BRANCH = "C_BASIS_TRAINING_HOLDOUT_TRADEOFF"


def load_json(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def validate_v2e_precondition(report, accept_known_limitation):
    if report.get("experiment") != "STEP4C3_V2E_FIXED_OBSERVATION_BASIS_FAMILY":
        raise RuntimeError("--v2e-report is not a Step-4C3 V2E report")
    if report.get("status") != "complete":
        raise RuntimeError("V2E report is not complete")
    tree = report.get("decision_tree", {})
    branch = tree.get("branch")
    promoted = tree.get("promoted_basis")
    if branch != EXPECTED_V2E_BRANCH or promoted != "none":
        raise RuntimeError(
            "Step 4D-A is frozen for the observed V2E stop-loss state: "
            f"branch={EXPECTED_V2E_BRANCH}, promoted_basis=none; got "
            f"branch={branch}, promoted_basis={promoted}"
        )
    frozen = report.get("frozen", {})
    checks = {
        "sources_per_arm": int(frozen.get("sources_per_arm", -1)) == 900,
        "source_depth_fraction": abs(float(frozen.get("source_depth_fraction", -1)) - 0.8) < 1e-15,
        "lambda_rel": abs(float(frozen.get("lambda_rel", -1)) - 1e-4) < 1e-15,
        "softening_m": abs(float(frozen.get("softening_m", -1)) - 0.0) < 1e-30,
        "boundary_gate": abs(float(frozen.get("boundary_gate", -1)) - 0.05) < 1e-15,
        "source_placement": frozen.get("source_placement") == "original production baseline_inner_surface_fps only",
    }
    if not all(checks.values()):
        raise RuntimeError(f"V2E frozen-state mismatch: {checks}")
    if not accept_known_limitation:
        raise RuntimeError(
            "V2E explicitly kept Step 4D blocked under the original qualification tree. "
            "To run the reviewer-facing two-way demonstration under the revised manuscript "
            "scope, pass --accept-known-c3-limitation. This does NOT relabel C3 as PASS."
        )
    return {
        "v2e_branch": branch,
        "v2e_promoted_basis": promoted,
        "known_c3_limitation_explicitly_accepted": True,
        "frozen_checks": checks,
    }


def weighted_vector_rms(g, vectors):
    x = np.asarray(vectors, dtype=np.float64).reshape(g.num_sources, 3)
    w = np.asarray(g.source_weights, dtype=np.float64)
    return float(np.sqrt(np.sum(w*np.sum(x*x, axis=1))/max(float(np.sum(w)), TINY)))


def coefficient_temporal_closure_rel(g, p_prev, q_prev, p_now, q_now):
    residual = (np.asarray(p_now)-np.asarray(p_prev))/DT - 0.5*(np.asarray(q_prev)+np.asarray(q_now))
    num = c3.weighted_coeff_rms(g, residual)
    den = max(c3.weighted_coeff_rms(g, q_now), TINY)
    return float(num/den)


def source_moments(g, q_coeff, p_coeff):
    K = c3.coeff_to_vectors(g, q_coeff)
    P = c3.coeff_to_vectors(g, p_coeff)
    w = np.asarray(g.source_weights, dtype=np.float64)[:, None]
    M = np.sum(w*K, axis=0)  # A m
    D = np.sum(w*P, axis=0)  # C m
    return M, D, K, P


def aggregate_holdout_rows(res_sum, inc_sum, counts, region_names):
    global_count = int(counts.get("global", 0))
    global_inc_rms = (
        math.sqrt(float(inc_sum["global"])/global_count)
        if global_count > 0 else 0.0
    )
    rows = []
    by_region = {}
    for region in region_names:
        count = int(counts.get(region, 0))
        if count <= 0:
            continue
        rr = math.sqrt(float(res_sum[region])/count)
        ii = math.sqrt(float(inc_sum[region])/count)
        eta = ii/max(global_inc_rms, TINY)
        local_rel = rr/max(ii, TINY)
        global_norm = rr/max(global_inc_rms, TINY)
        if region == "global" or eta >= WEAK_REGION_THRESHOLD:
            metric = "local_rel"
            value = local_rel
        else:
            metric = "global_norm"
            value = global_norm
        row = {
            "region": region,
            "residual_rms_V_per_m": rr,
            "incident_rms_V_per_m": ii,
            "incident_ratio": eta,
            "local_rel": local_rel,
            "global_norm": global_norm,
            "gate_metric": metric,
            "gate_value": value,
            "passed_5pct": bool(value < BOUNDARY_GATE),
            "sample_count_accumulated": count,
        }
        rows.append(row)
        by_region[region] = row
    worst = max(rows, key=lambda r: r["gate_value"]) if rows else None
    return by_region, worst


def fit_response(times, current_moment, polarization_moment, analysis_start_s):
    t = np.asarray(times, dtype=np.float64)
    M = np.asarray(current_moment, dtype=np.float64)
    D = np.asarray(polarization_moment, dtype=np.float64)
    mask = t >= float(analysis_start_s)
    if int(np.count_nonzero(mask)) < 8:
        raise RuntimeError("too few final-window samples for multitone response fit")
    fm = v2a.fit_multitone(t[mask], M[mask])
    fd = v2a.fit_multitone(t[mask], D[mask])
    out = {
        "num_samples": int(np.count_nonzero(mask)),
        "current_fit_condition_number": float(fm["condition_number"]),
        "polarization_fit_condition_number": float(fd["condition_number"]),
        "frequencies": {},
    }
    for f in (3.0e6, 6.0e6, 9.0e6):
        Mh = np.asarray(fm["phasors"][f], np.complex128).reshape(3)
        Dh = np.asarray(fd["phasors"][f], np.complex128).reshape(3)
        out["frequencies"][f"{f/1e6:.0f}MHz"] = {
            "current_moment_vector_A_m": [[float(z.real), float(z.imag)] for z in Mh],
            "current_moment_norm_A_m": float(np.linalg.norm(Mh)),
            "axial_current_moment_A_m": float(abs(Mh[2])),
            "polarization_moment_vector_C_m": [[float(z.real), float(z.imag)] for z in Dh],
            "polarization_moment_norm_C_m": float(np.linalg.norm(Dh)),
            "axial_polarization_moment_C_m": float(abs(Dh[2])),
            "continuity_harmonic_ratio_abs_iwD_over_M": float(
                abs(1j*2.0*np.pi*f*Dh[2])/max(abs(Mh[2]), TINY)
            ),
        }
    return out


def run_vacuum_reference(settings, g, Hp, Hk, prep, holdout):
    steps = c3record.steps_from_cycles(float(settings["cycles"]))
    times = np.arange(steps, dtype=np.float64)*DT
    analysis_start_s = float(settings["analysis_start_cycles"])/F_DRIVE
    region_names = list(holdout["region_names"])
    Hpoints = np.asarray(holdout["points"], dtype=np.float64)
    Hnormals = np.asarray(holdout["normals"], dtype=np.float64)
    Hmask = np.asarray(holdout["mask_matrix"], dtype=bool)

    ndof = 2*g.num_sources
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
        incident = c3.feed_at_time(g.solve_obs_points, float(t_s))
        e = c3.tangent_coeffs(incident, g.solve_tangent1, g.solve_tangent2)
        history = Hp @ p_prev + Hk @ q_prev
        q = c3.solve_fixed_ridge(prep, -(e + history))
        p = p_prev + 0.5*DT*(q_prev + q)
        closure_max = max(closure_max, coefficient_temporal_closure_rel(g, p_prev, q_prev, p, q))
        residual = e + history + c3.operator_now_coeff(prep, q)
        if t_s >= analysis_start_s:
            coll_res_sum += float(np.dot(residual, residual))
            coll_inc_sum += float(np.dot(e, e))
        if step % int(settings["diag_stride"]) == 0 and t_s >= analysis_start_s:
            self_H, _ = c3.evaluate_retarded_mot_field(Hpoints, g, p_prev, q_prev, q)
            inc_H = c3.feed_at_time(Hpoints, float(t_s))
            total_H = inc_H + self_H
            rsq = c3.tangential_sq(total_H, Hnormals)
            isq = c3.tangential_sq(inc_H, Hnormals)
            for ir, region in enumerate(region_names):
                m = Hmask[ir]
                if np.any(m):
                    hold_res_sum[region] += float(np.sum(rsq[m]))
                    hold_inc_sum[region] += float(np.sum(isq[m]))
                    hold_count[region] += int(np.count_nonzero(m))
        M, D, _, _ = source_moments(g, q, p)
        Mhist[step] = M
        Dhist[step] = D
        q_prev, p_prev = q, p

    coll = math.sqrt(coll_res_sum/max(coll_inc_sum, TINY))
    regions, worst = aggregate_holdout_rows(hold_res_sum, hold_inc_sum, hold_count, region_names)
    response = fit_response(times, Mhist, Dhist, analysis_start_s)
    return {
        "collocation_rel_l2": float(coll),
        "independent_holdout": {"regions": regions, "worst_gate": worst},
        "temporal_closure_max_relative": float(closure_max),
        "response": response,
    }


def build_feed_grid_template_code(bundle):
    bridge = bundle["unit_bridge"]
    nx, ny, nz = bundle["shape"]
    dx, dy, dz = bundle["spacing"]
    x0, y0, z0 = bundle["origin"]
    xs = x0 + (np.arange(nx, dtype=np.float64)+0.5)*dx
    ys = y0 + (np.arange(ny, dtype=np.float64)+0.5)*dy
    zs = z0 + (np.arange(nz, dtype=np.float64)+0.5)*dz
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts_code = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    pts_si = np.asarray(bridge.length_code_to_si(pts_code), dtype=np.float64)
    feed_si = c3.feed_shape(pts_si)
    feed_code = np.asarray(bridge.efield_si_to_code(feed_si), dtype=np.float64)
    return feed_code


def run_two_way(args, settings, frozen_g, v2e_precondition):
    import _jefipic_mas_case_common as common
    try:
        import cupy as cp
    except Exception as exc:
        raise RuntimeError(f"Step 4D-A formal/quick run requires CuPy/GPU: {exc}")

    out = Path("article_cases_output")/args.out
    out.mkdir(parents=True, exist_ok=True)

    cfg = c3record.make_pic_config(args.out, settings)
    # Full-EM validation variant:
    # - quasi_electrostatic=False enables retarded rho/drho, native staggered J/dJ,
    #   and the full retarded MAS dipole P/K/Kdot field terms.
    # - this variant requires the accompanying native staggered-face atomic
    #   full-EM EMsolver patch; physics is unchanged relative to observation mode.
    cfg.quasi_electrostatic = False
    cfg.jefipic_kernel_mode = "atomic"
    fp = c3.geometry_fingerprint(frozen_g)

    old_builder = common.build_mas_device
    def _build_probe_device(cfg_local, stl_path, grid_shape, spacing, origin, *, current_amplitude=None):
        del cfg_local, stl_path, grid_shape, spacing, origin, current_amplitude
        return c3record.lightweight_probe_holder(frozen_g)
    common.build_mas_device = _build_probe_device
    try:
        bundle = common.build_pic_solver_mas(cfg, out/"IGNORED_FROZEN_ROUNDED_GEOMETRY.stl")
    finally:
        common.build_mas_device = old_builder

    PIC = bundle["PIC"]
    solver = bundle["solver"]
    actual_kernel_mode = str(getattr(solver, "kernel_mode", "unknown"))
    if actual_kernel_mode != "atomic":
        raise RuntimeError(
            f"Full-EM atomic Step 4D-A requires kernel_mode='atomic', "
            f"got {actual_kernel_mode!r}"
        )
    if bool(cfg.quasi_electrostatic):
        raise RuntimeError("Full-EM Step 4D-A unexpectedly remained quasi-electrostatic")
    bridge = bundle["unit_bridge"]
    rng = bundle["rng"]
    nx, ny, nz = bundle["shape"]
    dx, dy, dz = bundle["spacing"]
    Ex, Ey, Ez, Bx, By, Bz = bundle["fields"]
    g = bundle["mas"].geometry
    if c3.geometry_fingerprint(g) != fp:
        raise RuntimeError("Step 4D-A geometry fingerprint changed inside PIC builder")

    holdout = c3record.build_stratified_holdout(settings)
    Hpoints = np.asarray(holdout["points"], dtype=np.float64)
    Hnormals = np.asarray(holdout["normals"], dtype=np.float64)
    Hmask = np.asarray(holdout["mask_matrix"], dtype=bool)
    region_names = list(holdout["region_names"])
    solve_obs_code = bridge.length_si_to_code(g.solve_obs_points)
    holdout_code = bridge.length_si_to_code(Hpoints)

    obs_chunk = 8 if settings["mode"] == "formal" else 16
    Hp, Hk, An, delay = c3.assemble_retarded_mot_operators(g, obs_chunk)
    if delay["max_delay_over_dt"] >= 1.0:
        raise RuntimeError("Step 4D-A collocation one-step MOT invalid: max delay >= dt")
    prep = c3.prepare_fixed_ridge_inplace(An, g.solve_weights)
    del An
    gc.collect()

    max_hold_delay = 0.0
    SP = np.asarray(g.ant_src_points, dtype=np.float64)
    for a in range(0, len(Hpoints), 128):
        R = np.linalg.norm(Hpoints[a:a+128, None, :]-SP[None, :, :], axis=2)
        max_hold_delay = max(max_hold_delay, float(np.max((R/c3.C0)/DT)))
    if max_hold_delay >= 1.0:
        raise RuntimeError("Step 4D-A holdout one-step MOT invalid: max delay >= dt")

    print("[Step 4D-A] running matched vacuum Retarded-MOT reference ...", flush=True)
    vacuum = run_vacuum_reference(settings, g, Hp, Hk, prep, holdout)

    feed_template_code = build_feed_grid_template_code(bundle)
    feed_x = cp.asarray(feed_template_code[:, 0])
    feed_y = cp.asarray(feed_template_code[:, 1])
    feed_z = cp.asarray(feed_template_code[:, 2])
    del feed_template_code

    steps = int(cfg.steps)
    times = np.arange(steps, dtype=np.float64)*DT
    analysis_start_s = float(settings["analysis_start_cycles"])/F_DRIVE
    ndof = 2*g.num_sources
    q_prev = np.zeros(ndof, dtype=np.float64)
    p_prev = np.zeros(ndof, dtype=np.float64)

    Mhist = np.full((steps, 3), np.nan)
    Dhist = np.full((steps, 3), np.nan)
    plasma_surface_rms = np.full(steps, np.nan)
    coll_residual = np.full(steps, np.nan)
    state_norms = np.full(steps, np.nan)
    closure_rel = np.full(steps, np.nan)

    coll_res_sum = coll_inc_sum = 0.0
    hold_res_sum = {r: 0.0 for r in region_names}
    hold_inc_sum = {r: 0.0 for r in region_names}
    hold_count = {r: 0 for r in region_names}

    absorbed_e = absorbed_i = 0
    finite = True
    runaway_detected = False
    completed = 0
    max_solver_closure = 0.0

    print(
        f"[Step 4D-A] mode={settings['mode']} steps={steps}, "
        f"particles/species={cfg.particles_per_species}, grid={cfg.grid_n}^3, "
        f"sources/arm={settings['sources_per_arm']}", flush=True
    )

    for step in range(steps):
        t_s = float(times[step])
        PIC.proceed_one_step(Ex, Ey, Ez, Bx, By, Bz, dx, dy, dz, nx, ny, nz, cfg.field_order)
        hits = common.absorb_particles_on_antenna(PIC, cfg, rng)
        absorbed_e += int(hits.get(0, 0))
        absorbed_i += int(hits.get(1, 0))

        rho, Jx_face, Jy_face, Jz_face = PIC.evaluate_rho_J(dx, dy, dz, nx, ny, nz, cfg.particle_order)
        solver.Jefimenko_solver(
            rho.flatten(), include_antenna=False,
            face_currents=(Jx_face, Jy_face, Jz_face),
        )

        Ecode = solver.sample_current_E_at_points(solve_obs_code)
        plasma = np.asarray(bridge.efield_code_to_si(Ecode), dtype=np.float64).reshape(g.num_collocation_points, 3)
        if not np.all(np.isfinite(plasma)):
            finite = False
            break
        plasma_surface_rms[step] = float(np.sqrt(np.mean(np.sum(plasma*plasma, axis=1))))
        if plasma_surface_rms[step] > PLASMA_E_RUNAWAY_GATE_V_PER_M:
            runaway_detected = True
            finite = False
            completed = step + 1
            print(
                f"[Step 4D-A] numerical runaway guard tripped at step {step}: "
                f"plasma surface RMS={plasma_surface_rms[step]:.6e} V/m",
                flush=True,
            )
            break

        incident = plasma + c3.feed_at_time(g.solve_obs_points, t_s)
        e = c3.tangent_coeffs(incident, g.solve_tangent1, g.solve_tangent2)
        history = Hp @ p_prev + Hk @ q_prev
        q = c3.solve_fixed_ridge(prep, -(e + history))
        p = p_prev + 0.5*DT*(q_prev + q)
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            finite = False
            break

        closure_rel[step] = coefficient_temporal_closure_rel(g, p_prev, q_prev, p, q)
        residual = e + history + c3.operator_now_coeff(prep, q)
        inc2 = float(np.dot(e, e)); res2 = float(np.dot(residual, residual))
        coll_residual[step] = math.sqrt(res2/max(inc2, TINY))
        state_norms[step] = c3.state_norm(g, q, p)
        if t_s >= analysis_start_s:
            coll_res_sum += res2
            coll_inc_sum += inc2

        if step % int(settings["diag_stride"]) == 0 and t_s >= analysis_start_s:
            Hcode = solver.sample_current_E_at_points(holdout_code)
            plasma_H = np.asarray(bridge.efield_code_to_si(Hcode), dtype=np.float64).reshape(len(Hpoints), 3)
            if not np.all(np.isfinite(plasma_H)):
                finite = False
                break
            self_H, hdelay = c3.evaluate_retarded_mot_field(Hpoints, g, p_prev, q_prev, q)
            max_hold_delay = max(max_hold_delay, float(hdelay))
            incident_H = plasma_H + c3.feed_at_time(Hpoints, t_s)
            total_H = incident_H + self_H
            rsq = c3.tangential_sq(total_H, Hnormals)
            isq = c3.tangential_sq(incident_H, Hnormals)
            for ir, region in enumerate(region_names):
                m = Hmask[ir]
                if np.any(m):
                    hold_res_sum[region] += float(np.sum(rsq[m]))
                    hold_inc_sum[region] += float(np.sum(isq[m]))
                    hold_count[region] += int(np.count_nonzero(m))

        M, D, Ksi, Psi = source_moments(g, q, p)
        Mhist[step] = M; Dhist[step] = D
        Kcode = bridge.surface_current_density_si_to_code(Ksi)
        Pcode = bridge.surface_polarization_density_si_to_code(Psi)
        solver.recompute_current_fields(
            antenna_surface_currents=Kcode,
            antenna_polarization=Pcode,
            include_antenna=True,
        )
        Ex0, Ey0, Ez0, Bx, By, Bz = solver.acquire_EB_field()
        ramp, _ = c3.smoothstep_ramp_and_derivative(t_s)
        feed_scale = float(ramp*math.cos(2.0*math.pi*F_DRIVE*t_s))
        Ex = Ex0 + feed_scale*feed_x
        Ey = Ey0 + feed_scale*feed_y
        Ez = Ez0 + feed_scale*feed_z

        if hasattr(solver, "get_point_source_closure_diagnostics"):
            sd = solver.get_point_source_closure_diagnostics()
            val = sd.get("temporal_continuity_residual_weighted_rms", float("nan"))
            if np.isfinite(val):
                max_solver_closure = max(max_solver_closure, float(abs(val)))

        q_prev, p_prev = q, p
        completed = step + 1
        if step % int(settings["diag_stride"]) == 0 or step == steps-1:
            print(
                f"[Step 4D-A] step {step+1}/{steps}, colloc={coll_residual[step]:.6g}, "
                f"plasma_E={plasma_surface_rms[step]:.6e} V/m, "
                f"state={state_norms[step]:.6e}, closure={closure_rel[step]:.3e}",
                flush=True,
            )

    if completed <= 0:
        raise RuntimeError("Step 4D-A completed zero steps")

    agg_coll = math.sqrt(coll_res_sum/max(coll_inc_sum, TINY)) if coll_inc_sum > 0 else float("inf")
    regions, worst = aggregate_holdout_rows(hold_res_sum, hold_inc_sum, hold_count, region_names)
    worst_value = float(worst["gate_value"]) if worst else float("inf")
    response = fit_response(times[:completed], Mhist[:completed], Dhist[:completed], analysis_start_s)

    # Homogeneous MAS tail from the actual coupled final state.  This is a
    # component stability audit, not a full plasma-after-drive-off simulation.
    tail_norms = [c3.state_norm(g, q_prev, p_prev)]
    tail_ratios = []
    for _ in range(TAIL_STEPS):
        history = Hp @ p_prev + Hk @ q_prev
        q = c3.solve_fixed_ridge(prep, -history)
        p = p_prev + 0.5*DT*(q_prev + q)
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            finite = False
            break
        nn = c3.state_norm(g, q, p)
        tail_ratios.append(nn/max(tail_norms[-1], TINY))
        tail_norms.append(nn)
        q_prev, p_prev = q, p
    if len(tail_ratios) >= TAIL_WINDOW:
        tail_growth = float(np.median(tail_ratios[-TAIL_WINDOW:]))
    else:
        tail_growth = float("inf")

    closure_max_rel = float(np.nanmax(closure_rel[:completed]))
    two_way_execution_valid = bool(
        completed == steps
        and finite
        and not runaway_detected
        and delay["max_delay_over_dt"] < 1.0
        and max_hold_delay < 1.0
        and closure_max_rel <= CLOSURE_REL_GATE
        and tail_growth <= TAIL_GROWTH_GATE
    )
    boundary_within_5pct = bool(agg_coll < BOUNDARY_GATE and worst_value < BOUNDARY_GATE)

    pv = response["frequencies"]["3MHz"]["axial_current_moment_A_m"]
    vv = vacuum["response"]["frequencies"]["3MHz"]["axial_current_moment_A_m"]
    ratio = float(pv/max(vv, TINY))

    if runaway_detected:
        classification = "TWO_WAY_NUMERICAL_RUNAWAY_ABORTED"
    elif not two_way_execution_valid:
        classification = "TWO_WAY_EXECUTION_FAILED_DIAGNOSE_BEFORE_ARTICLE_RERUN"
    elif boundary_within_5pct:
        classification = "TWO_WAY_EXECUTION_VALID_AND_BOUNDARY_WITHIN_PREDECLARED_5PCT"
    else:
        classification = "TWO_WAY_EXECUTION_VALID_WITH_CARRIED_FORWARD_C3_SCOPE_LIMITATION"

    report = {
        "experiment": "STEP4D_A_FULL_EM_ATOMIC_TWO_WAY_RETARDED_MOT_CLOSURE",
        "status": "complete",
        "mode": settings["mode"],
        "purpose": "full-EM validation of two-way PIC/Jefimenko <-> causal Retarded-MOT MAS closure",
        "v2e_precondition": v2e_precondition,
        "c3_qualification_relabelled_as_pass": False,
        "known_upper_band_limitation_carried_forward": True,
        "production_basis": "2-tangential-DOF continuity-consistent electric dipoles",
        "drive": {
            "type": "localized impressed feed electric field / voltage-like drive",
            "prescribed_active_surface_current_used": False,
            "unconstrained_active_passive_same_mode_cancellation_possible": False,
            "feed_frequency_Hz": F_DRIVE,
            "feed_field_amplitude_V_per_m": c3.FEED_FIELD_AMPLITUDE_V_PER_M,
            "ramp_cycles": c3.RAMP_CYCLES,
            "grid_injection": "same analytic E_feed(x,t) added to PIC E field; no external B is prescribed, while retarded plasma/MAS currents generate self-consistent magnetic fields",
            "boundary_rhs": "same analytic E_feed at collocation/holdout points",
        },
        "frozen": {
            "sources_per_arm": int(settings["sources_per_arm"]),
            "source_depth_fraction": c3.DEPTH_FRAC,
            "lambda_rel": c3.LAMBDA_REL,
            "softening_m": 0.0,
            "dt_s": DT,
            "collocation_factor": c3.COLLOCATION_FACTOR,
            "boundary_gate": BOUNDARY_GATE,
            "weak_region_threshold": WEAK_REGION_THRESHOLD,
            "plasma_kernel_quasi_electrostatic": bool(cfg.quasi_electrostatic),
            "full_em_grid_source_enabled": bool(not cfg.quasi_electrostatic and actual_kernel_mode in ("atomic", "observation")),
            "full_em_atomic_staggered_current_patch": bool(actual_kernel_mode == "atomic"),
            "plasma_grid_kernel_mode_actual": actual_kernel_mode,
        },
        "geometry": {
            "fingerprint": fp,
            "num_sources_total": int(g.num_sources),
            "num_collocation_points": int(g.num_collocation_points),
            "num_holdout_points": int(len(Hpoints)),
            "one_step_delay_collocation": delay,
            "holdout_max_delay_over_dt": float(max_hold_delay),
        },
        "execution": {
            "steps_requested": int(steps),
            "steps_completed": int(completed),
            "all_states_finite": bool(finite),
            "numerical_runaway_detected": bool(runaway_detected),
            "plasma_surface_rms_runaway_gate_V_per_m": PLASMA_E_RUNAWAY_GATE_V_PER_M,
            "coefficient_temporal_closure_max_relative": closure_max_rel,
            "coefficient_temporal_closure_gate": CLOSURE_REL_GATE,
            "solver_point_history_continuity_residual_max_code": float(max_solver_closure),
            "absorbed_electrons": int(absorbed_e),
            "absorbed_ions": int(absorbed_i),
        },
        "boundary": {
            "time_aggregated_collocation_rel_l2": float(agg_coll),
            "independent_holdout": {
                "regions": regions,
                "worst_gate": worst,
            },
            "boundary_within_predeclared_5pct": boundary_within_5pct,
            "interpretation": "reported diagnostically; failure does not get relabelled because C3 upper-band limitation is explicitly carried forward",
        },
        "homogeneous_mas_tail_from_coupled_final_state": {
            "steps": TAIL_STEPS,
            "tail_window": TAIL_WINDOW,
            "median_growth_factor_last_window": tail_growth,
            "growth_gate": TAIL_GROWTH_GATE,
            "component_stability_only": True,
        },
        "vacuum_reference": vacuum,
        "two_way_plasma_response": response,
        "rf_response_proxy": {
            "observable": "axial auxiliary electric-current moment M_z=sum_j w_j K_jz",
            "units": "A m",
            "not_a_port_current": True,
            "not_impedance_or_admittance": True,
            "not_S_parameter": True,
            "3MHz_vacuum_abs_A_m": float(vv),
            "3MHz_two_way_plasma_abs_A_m": float(pv),
            "3MHz_plasma_to_vacuum_ratio": ratio,
            "response_values_are_manuscript_final": False,
            "reason_not_final": "Step 4D-A uses the closure-test particle/grid settings; reviewer-facing article values require Step 4D-B representative and seed/window sensitivity reruns after closure stability is established.",
        },
        "two_way_execution_valid": two_way_execution_valid,
        "classification": classification,
        "next_step": (
            "Run Step 4D-B article-scale representative plus seed/window sensitivity using the identical frozen two-way formulation, then revise Major-2/Major-4 manuscript claims."
            if two_way_execution_valid else
            "Diagnose the two-way instability/closure failure before any article-scale rerun."
        ),
    }

    (out/"step4d_a_full_em_atomic_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        out/"step4d_a_full_em_atomic_diagnostics.npz",
        time_s=times[:completed],
        collocation_rel_l2=coll_residual[:completed],
        plasma_surface_rms_V_per_m=plasma_surface_rms[:completed],
        state_norm=state_norms[:completed],
        temporal_closure_rel=closure_rel[:completed],
        current_moment_A_m=Mhist[:completed],
        polarization_moment_C_m=Dhist[:completed],
        tail_state_norm=np.asarray(tail_norms, dtype=np.float64),
        tail_growth_ratio=np.asarray(tail_ratios, dtype=np.float64),
    )
    with (out/"step4d_a_full_em_atomic_region_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "region", "residual_rms_V_per_m", "incident_rms_V_per_m", "incident_ratio",
            "local_rel", "global_norm", "gate_metric", "gate_value", "passed_5pct",
            "sample_count_accumulated",
        ])
        w.writeheader()
        for r in regions.values():
            w.writerow(r)
    with (out/"step4d_a_full_em_atomic_response_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "case", "frequency", "axial_current_moment_A_m", "current_moment_norm_A_m",
            "axial_polarization_moment_C_m", "polarization_moment_norm_C_m",
            "continuity_harmonic_ratio_abs_iwD_over_M",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for case, rr in (("vacuum", vacuum["response"]), ("two_way_plasma", response)):
            for freq, row in rr["frequencies"].items():
                w.writerow({"case": case, "frequency": freq, **{k: row[k] for k in fieldnames[2:]}})

    summary = [
        "Step 4D-A full-EM atomic-accumulation two-way Retarded-MOT closure",
        f"classification: {classification}",
        f"two_way_execution_valid: {two_way_execution_valid}",
        f"boundary_within_predeclared_5pct: {boundary_within_5pct}",
        f"steps: {completed}/{steps}",
        f"aggregate collocation: {agg_coll:.9g}",
        f"worst holdout: {worst_value:.9g} ({None if worst is None else worst['region']})",
        f"closure max relative: {closure_max_rel:.9g}",
        f"homogeneous tail growth: {tail_growth:.9g}",
        f"3MHz vacuum axial current moment: {vv:.9g} A m",
        f"3MHz two-way plasma axial current moment: {pv:.9g} A m",
        f"3MHz plasma/vacuum moment ratio: {ratio:.9g}",
        "C3 qualification relabelled as pass: False",
        "response values manuscript-final: False",
    ]
    (out/"step4d_a_full_em_atomic_summary.txt").write_text("\n".join(summary)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def self_test():
    # CPU-only: verify the exact Retarded-MOT recurrence/evaluator chain and the
    # logical distinction between execution validity and boundary qualification.
    settings = {
        "mode": "quick", "total_length_m": 16e-3, "sources_per_arm": 24,
        "n_phi": 28, "n_cyl": 28, "n_hemi": 10, "cycles": 0.15,
        "analysis_start_cycles": 0.02, "diag_stride": 5,
        "holdout_points_target": 128,
    }
    g = c3.build_geometry(settings)
    Hp, Hk, An, delay = c3.assemble_retarded_mot_operators(g, obs_chunk=16)
    prep = c3.prepare_fixed_ridge_inplace(An, g.solve_weights)
    rng = np.random.default_rng(20261003)
    ndof = 2*g.num_sources
    q_prev = 1e-6*rng.normal(size=ndof)
    p_prev = 1e-15*rng.normal(size=ndof)
    incident = 1e-3*rng.normal(size=(g.num_collocation_points, 3))
    e = c3.tangent_coeffs(incident, g.solve_tangent1, g.solve_tangent2)
    history = Hp@p_prev + Hk@q_prev
    q = c3.solve_fixed_ridge(prep, -(e+history))
    p = p_prev + 0.5*DT*(q_prev+q)
    closure = coefficient_temporal_closure_rel(g, p_prev, q_prev, p, q)
    assembled = c3.operator_now_coeff(prep, q)
    # Independent full field evaluated at the collocation points must produce
    # the same current-node self field when history is included.
    self_field, hdelay = c3.evaluate_retarded_mot_field(g.solve_obs_points, g, p_prev, q_prev, q, obs_chunk=64, src_chunk=64)
    independent = c3.tangent_coeffs(self_field, g.solve_tangent1, g.solve_tangent2)
    chain_err = float(np.linalg.norm(independent-(history+assembled))/max(float(np.linalg.norm(independent)), TINY))
    # Logic guard: execution success is not equivalent to 5% qualification.
    fake_exec = True; fake_boundary = False
    classification = (
        "TWO_WAY_EXECUTION_VALID_WITH_CARRIED_FORWARD_C3_SCOPE_LIMITATION"
        if fake_exec and not fake_boundary else "unexpected"
    )
    passed = bool(
        delay["max_delay_over_dt"] < 1.0
        and hdelay < 1.0
        and closure < 1e-12
        and chain_err < 5e-11
        and classification == "TWO_WAY_EXECUTION_VALID_WITH_CARRIED_FORWARD_C3_SCOPE_LIMITATION"
    )
    report = {
        "self_test_pass": passed,
        "self_test_requires_forcing": False,
        "self_test_requires_gpu": False,
        "max_delay_over_dt": float(delay["max_delay_over_dt"]),
        "independent_max_delay_over_dt": float(hdelay),
        "temporal_closure_relative": float(closure),
        "assembled_history_plus_now_vs_pairwise_relative_l2": chain_err,
        "scope_logic_preserves_failed_5pct_gate": classification,
    }
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("Step 4D-A self-test failed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--v2e-report", default=None)
    p.add_argument("--accept-known-c3-limitation", action="store_true")
    p.add_argument("--out", default="step4d_a_full_em_atomic_formal")
    p.add_argument("--cycles", type=float, default=None)
    p.add_argument("--particles", type=int, default=None)
    p.add_argument("--grid-n", type=int, default=None)
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    if a.v2e_report is None:
        raise RuntimeError("--v2e-report is required")
    v2e = load_json(a.v2e_report)
    pre = validate_v2e_precondition(v2e, bool(a.accept_known_c3_limitation))
    settings = c3record.mode_settings(bool(a.quick))
    if a.cycles is not None:
        settings["cycles"] = float(a.cycles)
    if a.particles is not None:
        settings["particles_per_species"] = int(a.particles)
    if a.grid_n is not None:
        settings["grid_n"] = int(a.grid_n)
    g = c3.build_geometry(settings)
    run_two_way(a, settings, g, pre)


if __name__ == "__main__":
    main()
