#!/usr/bin/env python3
"""
Step 4C3 — one-way broadband PIC -> implicit MAS forcing.

Scientific question
-------------------
Can the full time-domain plasma electric field produced by PIC/Jefimenko drive
the Step-4C1-V3 implicit MAS boundary solver without:
- any 3 MHz projection,
- the retired active/passive recurrence,
- numerical runaway,
- loss of independent PEC-boundary accuracy?

This is deliberately ONE-WAY.

Stage RECORD:
    fixed validated external 3 MHz antenna probe
      -> nonlinear PIC/Jefimenko
      -> record complete E_plasma(surface, t)

Stage REPLAY:
    E_feed(t) + recorded broadband E_plasma(surface, t)
      -> frozen zero-softening backward-Euler MAS
      -> K(t), P(t), boundary residuals

The replayed MAS response is NOT fed back into PIC in Step 4C3.
Two-way nonlinear feedback is reserved for Step 4D.

Important frozen choices
------------------------
- No single-frequency projection of E_plasma(t).
- Zero operator softening.
- d/R = 0.80.
- 2x independent collocation.
- weighted + column-scaled ridge, lambda_rel = 1e-4.
- dt = 1 ns.
- full electric-dipole 1/R^3 + 1/R^2 + 1/R terms.
- backward-Euler polarization closure:
      P^n = P^(n-1) + dt K^n
      Kdot^n = (K^n - K^(n-1))/dt
- the old MAS solve_currents_from_surface_field() is NOT used in replay.

Acceptance gates
----------------
RECORD:
- fixed external vacuum source collocation residual < 5%;
- every requested PIC step completes;
- every recorded plasma surface field is finite.

REPLAY (post-ramp analysis window):
- every time node finite;
- time-aggregated collocation tangential PEC residual < 5%;
- worst stratified independent-holdout physical gate < 5%;
- zero-input tail median state growth <= 1.001 / step;
- forcing/replay geometry fingerprints match exactly.

Response amplitudes and harmonic contents are diagnostic only.
They are NOT manuscript-final amplification factors.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.linalg import cho_factor, cho_solve

import step2mb1_incident_normalization_feed_localization as mb1

from EMsolver.mas_preprocessor import (
    generate_rounded_two_arm_dipole_points,
    assemble_impedance_matrix,
    _basis_matrix,
    _constraint_matrix,
)

EPS0 = 8.8541878128e-12
C0 = 299792458.0
TINY = 1.0e-300

F_DRIVE = 3.0e6
DT = 1.0e-9
DEPTH_FRAC = 0.80
COLLOCATION_FACTOR = 2.0
LAMBDA_REL = 1.0e-4
FEED_FIELD_AMPLITUDE_V_PER_M = 1.0
RAMP_CYCLES = 0.50

BOUNDARY_GATE = 0.05
TAIL_GROWTH_GATE = 1.001
WEAK_REGION_THRESHOLD = 0.20
TAIL_STEPS = 80

GATE_REGIONS = (
    "global",
    "feed_rounded",
    "feed_tips",
    "feed_transition",
    "outer_rounded",
    "outer_tips",
    "all_transition",
    "side_mid",
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

def mode_settings(quick: bool):
    if quick:
        return {
            "mode": "quick",
            "total_length_m": 16.0e-3,
            "sources_per_arm": 180,
            "n_phi": 28,
            "n_cyl": 28,
            "n_hemi": 10,
            "cycles": 1.00,
            "analysis_start_cycles": 0.50,
            "particles_per_species": 1000,
            "grid_n": 11,
            "diag_stride": 10,
            "holdout_points_target": 1024,
        }
    return {
        "mode": "formal",
        "total_length_m": 60.0e-3,
        "sources_per_arm": 900,
        "n_phi": 72,
        "n_cyl": 96,
        "n_hemi": 24,
        "cycles": 2.00,
        "analysis_start_cycles": 0.75,
        "particles_per_species": 2500,
        "grid_n": 13,
        "diag_stride": 20,
        "holdout_points_target": 2048,
    }


def steps_from_cycles(cycles: float):
    return int(np.ceil(float(cycles)/(F_DRIVE*DT))) + 1


def smoothstep_ramp_and_derivative(t_s: float, cycles: float = RAMP_CYCLES):
    duration = float(cycles)/F_DRIVE
    if duration <= 0.0:
        return 1.0, 0.0
    x = float(t_s)/duration
    if x <= 0.0:
        return 0.0, 0.0
    if x >= 1.0:
        return 1.0, 0.0
    ramp = x*x*(3.0 - 2.0*x)
    drdx = 6.0*x - 6.0*x*x
    return float(ramp), float(drdx/duration)


# -----------------------------------------------------------------------------
# Frozen geometry and fingerprints
# -----------------------------------------------------------------------------

def build_geometry(settings):
    return generate_rounded_two_arm_dipole_points(
        radius=1.0e-3,
        total_length=float(settings["total_length_m"]),
        feed_gap=2.0e-3,
        depth_frac=DEPTH_FRAC,
        sources_per_arm=int(settings["sources_per_arm"]),
        collocation_factor=COLLOCATION_FACTOR,
        n_phi=int(settings["n_phi"]),
        n_cyl=int(settings["n_cyl"]),
        n_hemi=int(settings["n_hemi"]),
    )


def array_sha256(arrays):
    h = hashlib.sha256()
    for a in arrays:
        x = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        h.update(str(x.shape).encode("ascii"))
        h.update(x.tobytes(order="C"))
    return h.hexdigest()


def geometry_fingerprint(g):
    return array_sha256([
        g.ant_src_points,
        g.source_weights,
        g.source_tangent1,
        g.source_tangent2,
        g.solve_obs_points,
        g.solve_normals,
        g.solve_weights,
        g.solve_tangent1,
        g.solve_tangent2,
    ])


# -----------------------------------------------------------------------------
# Direct tangent-space electric-dipole operators
# -----------------------------------------------------------------------------

def _dipole_blocks_for_chunk(obs, src, weights):
    r = obs[:, None, :] - src[None, :, :]
    R = np.maximum(np.linalg.norm(r, axis=2), 1.0e-15)
    rr = r[:, :, :, None] * r[:, :, None, :]
    I = np.eye(3, dtype=np.float64)[None, None, :, :]
    ce = 1.0/(4.0*np.pi*EPS0)
    w = weights[None, :, None, None]

    Gp = ce*w*(
        3.0*rr/R[:, :, None, None]**5
        - I/R[:, :, None, None]**3
    )
    Gk = ce*w*(
        3.0*rr/R[:, :, None, None]**4
        - I/R[:, :, None, None]**2
    )/C0
    Gd = ce*w*(
        rr/R[:, :, None, None]**3
        - I/R[:, :, None, None]
    )/(C0*C0)
    return Gp, Gk, Gd


def _project_block(obs_tangent, block, src_tangent):
    # block: [M,3,N,3]
    # obs_tangent: [M,2,3], src_tangent: [N,2,3]
    out = np.einsum(
        "mua,mjab,jvb->mujv",
        obs_tangent,
        block,
        src_tangent,
        optimize=True,
    )
    return out.reshape(2*len(obs_tangent), 2*len(src_tangent))


def assemble_projected_be_operators(g, *, obs_chunk=16):
    """
    Assemble tangent-space matrices directly, without first materializing the
    much larger Cartesian C @ G @ B route.

    Returns:
      Ap     : tangent G_p
      Ad     : tangent G_kdot
      A_node : tangent [dt G_p + G_k + G_kdot/dt]
    """
    obs = np.asarray(g.solve_obs_points, dtype=np.float64)
    src = np.asarray(g.ant_src_points, dtype=np.float64)
    weights = np.asarray(g.source_weights, dtype=np.float64)
    ot = np.stack(
        [g.solve_tangent1, g.solve_tangent2],
        axis=1,
    )
    st = np.stack(
        [g.source_tangent1, g.source_tangent2],
        axis=1,
    )

    m = len(obs)
    n = len(src)
    Ap = np.empty((2*m, 2*n), dtype=np.float64)
    Ad = np.empty_like(Ap)
    A_node = np.empty_like(Ap)

    for a in range(0, m, int(obs_chunk)):
        b = min(m, a + int(obs_chunk))
        Gp, Gk, Gd = _dipole_blocks_for_chunk(
            obs[a:b],
            src,
            weights,
        )
        Pp = _project_block(ot[a:b], Gp, st)
        Pd = _project_block(ot[a:b], Gd, st)
        Pk = _project_block(ot[a:b], Gk, st)

        sl = slice(2*a, 2*b)
        Ap[sl] = Pp
        Ad[sl] = Pd
        A_node[sl] = DT*Pp + Pk + Pd/DT

    return Ap, Ad, A_node


def complex_smax_power(A, niter=8):
    rng = np.random.default_rng(1234)
    x = (
        rng.normal(size=A.shape[1])
        + 1j*rng.normal(size=A.shape[1])
    )
    x /= max(float(np.linalg.norm(x)), TINY)
    for _ in range(int(niter)):
        y = A @ x
        z = A.conj().T @ y
        nz = float(np.linalg.norm(z))
        if nz < TINY:
            return 1.0
        x = z/nz
    return float(np.linalg.norm(A @ x))


def assemble_harmonic_tangent_operator(g, omega, *, obs_chunk=16):
    obs = np.asarray(g.solve_obs_points, dtype=np.float64)
    src = np.asarray(g.ant_src_points, dtype=np.float64)
    weights = np.asarray(g.source_weights, dtype=np.float64)
    ot = np.stack(
        [g.solve_tangent1, g.solve_tangent2],
        axis=1,
    )
    st = np.stack(
        [g.source_tangent1, g.source_tangent2],
        axis=1,
    )

    A = np.empty(
        (2*len(obs), 2*len(src)),
        dtype=np.complex128,
    )
    for a in range(0, len(obs), int(obs_chunk)):
        b = min(len(obs), a + int(obs_chunk))
        Gp, Gk, Gd = _dipole_blocks_for_chunk(
            obs[a:b], src, weights
        )
        B = Gp/(1j*omega) + Gk + 1j*omega*Gd
        A[2*a:2*b] = _project_block(
            ot[a:b], B, st
        )
    return A


def prepare_complex_ridge(A, point_weights):
    rs = np.repeat(
        np.sqrt(
            np.asarray(point_weights, dtype=float)
            / max(float(np.mean(point_weights)), TINY)
        ),
        2,
    )
    Aw = A*rs[:, None]
    cs = np.linalg.norm(Aw, axis=0)
    cs = np.where(cs > TINY, cs, 1.0)
    As = Aw/cs[None, :]

    smax = max(complex_smax_power(As, 8), TINY)
    alpha = LAMBDA_REL*smax
    H = As.conj().T @ As
    H.flat[::H.shape[0]+1] += alpha*alpha
    cf = cho_factor(
        H,
        lower=True,
        overwrite_a=True,
        check_finite=False,
    )
    return {
        "rs": rs,
        "cs": cs,
        "As": As,
        "cf": cf,
        "smax": float(smax),
        "alpha": float(alpha),
    }


def solve_complex_ridge(prepared, rhs):
    bw = np.asarray(rhs, dtype=np.complex128)*prepared["rs"]
    y = cho_solve(
        prepared["cf"],
        prepared["As"].conj().T @ bw,
        check_finite=False,
    )
    return y/prepared["cs"]


def prepare_real_be_solver_inplace(A_node, point_weights):
    """
    Reuse A_node storage as the weighted/column-scaled matrix As to keep formal
    memory bounded.
    """
    rs = np.repeat(
        np.sqrt(
            np.asarray(point_weights, dtype=float)
            / max(float(np.mean(point_weights)), TINY)
        ),
        2,
    )
    A_node *= rs[:, None]
    cs = np.linalg.norm(A_node, axis=0)
    cs = np.where(cs > TINY, cs, 1.0)
    A_node /= cs[None, :]
    As = A_node

    # Frozen Step-2/3 path: power-estimated smax + normal equations + Cholesky.
    # This local copy avoids constructing an explicit pseudoinverse.
    rng = np.random.default_rng(1234)
    x = rng.normal(size=As.shape[1])
    x /= max(float(np.linalg.norm(x)), TINY)
    for _ in range(8):
        y = As @ x
        z = As.T @ y
        nz = float(np.linalg.norm(z))
        if nz < TINY:
            break
        x = z/nz
    smax = max(float(np.linalg.norm(As @ x)), TINY)
    alpha = LAMBDA_REL*smax

    H = As.T @ As
    H.flat[::H.shape[0]+1] += alpha*alpha
    cf = cho_factor(
        H,
        lower=True,
        overwrite_a=True,
        check_finite=False,
    )
    return {
        "rs": rs,
        "cs": cs,
        "As": As,
        "cf": cf,
        "smax": float(smax),
        "alpha": float(alpha),
    }


def solve_real_be(prepared, rhs):
    bw = np.asarray(rhs, dtype=np.float64)*prepared["rs"]
    y = cho_solve(
        prepared["cf"],
        prepared["As"].T @ bw,
        check_finite=False,
    )
    return y/prepared["cs"]


def tangent_coeffs(field, t1, t2):
    field = np.asarray(field, dtype=np.float64)
    out = np.empty(2*len(field), dtype=np.float64)
    out[0::2] = np.sum(field*t1, axis=1)
    out[1::2] = np.sum(field*t2, axis=1)
    return out


def coeff_to_vectors(g, q):
    q = np.asarray(q, dtype=np.float64).reshape(-1, 2)
    return (
        q[:, 0, None]*g.source_tangent1
        + q[:, 1, None]*g.source_tangent2
    )


def weighted_coeff_rms(g, q):
    q = np.asarray(q).reshape(-1, 2)
    w = np.asarray(g.source_weights, dtype=float)
    mag2 = np.sum(np.abs(q)**2, axis=1)
    return float(np.sqrt(
        np.sum(w*mag2)
        / max(float(np.sum(w)), TINY)
    ))


def state_norm(g, q, p):
    return math.sqrt(
        weighted_coeff_rms(g, q)**2
        + weighted_coeff_rms(g, np.asarray(p)/DT)**2
    )


# -----------------------------------------------------------------------------
# Fixed external vacuum probe source for PIC RECORD
# -----------------------------------------------------------------------------

def feed_shape(points):
    return FEED_FIELD_AMPLITUDE_V_PER_M * mb1.inc_feed_localized(
        np.asarray(points, dtype=float),
        2.0e-3,
        1.0,
    )


def build_vacuum_probe_phasor(g):
    omega = 2.0*np.pi*F_DRIVE
    A = assemble_harmonic_tangent_operator(g, omega)
    prepared = prepare_complex_ridge(
        A,
        g.solve_weights,
    )
    Ehat = feed_shape(g.solve_obs_points).astype(np.complex128)
    e = np.empty(2*len(Ehat), dtype=np.complex128)
    e[0::2] = np.sum(
        Ehat*g.solve_tangent1,
        axis=1,
    )
    e[1::2] = np.sum(
        Ehat*g.solve_tangent2,
        axis=1,
    )

    q = solve_complex_ridge(prepared, -e)
    coll = float(
        np.linalg.norm(A @ q + e)
        / max(float(np.linalg.norm(e)), TINY)
    )

    q2 = q.reshape(-1, 2)
    Khat = (
        q2[:, 0, None]*g.source_tangent1
        + q2[:, 1, None]*g.source_tangent2
    )
    Phat = Khat/(1j*omega)

    # Release the large harmonic matrix before the PIC time loop.
    del A, prepared
    gc.collect()

    return Khat, Phat, coll


# -----------------------------------------------------------------------------
# Stratified independent holdout
# -----------------------------------------------------------------------------

def _choose_even(indices, count):
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= int(count):
        return indices
    pos = np.linspace(
        0,
        len(indices)-1,
        int(count),
        dtype=np.int64,
    )
    return indices[pos]


def build_stratified_holdout(settings):
    R = 1.0e-3
    total = float(settings["total_length_m"])
    gap = 2.0e-3
    Larm = 0.5*(total-gap)
    z_bottom = -0.5*total
    z_top = 0.5*gap

    ncyl = int(settings["n_cyl"])
    nphi = int(settings["n_phi"])
    nhemi = int(settings["n_hemi"])

    T0, _, _, _, T0loc = mb1.make_arm(
        R, R, Larm, z_bottom, ncyl, nphi, nhemi
    )
    T1, _, _, _, T1loc = mb1.make_arm(
        R, R, Larm, z_top, ncyl, nphi, nhemi
    )

    all_points = []
    all_normals = []
    all_masks_parts = {r: [] for r in GATE_REGIONS if r != "global"}

    for bary in (
        [0.19, 0.37, 0.44],
        [0.53, 0.29, 0.18],
    ):
        H0, HN0 = mb1.bary_arm(
            T0loc, bary, R, Larm, z_bottom
        )
        H1, HN1 = mb1.bary_arm(
            T1loc, bary, R, Larm, z_top
        )
        HP = np.concatenate((H0, H1))
        HN = np.concatenate((HN0, HN1))
        aid = np.concatenate((
            np.zeros(len(H0), dtype=int),
            np.ones(len(H1), dtype=int),
        ))
        masks = mb1.classify(
            HP, aid, R, Larm, z_bottom, z_top
        )

        all_points.append(HP)
        all_normals.append(HN)
        for region in all_masks_parts:
            all_masks_parts[region].append(
                np.asarray(masks.get(region, np.zeros(len(HP), bool)), dtype=bool)
            )

    P = np.concatenate(all_points)
    N = np.concatenate(all_normals)
    full_masks = {
        region: np.concatenate(parts)
        for region, parts in all_masks_parts.items()
    }

    target = int(settings["holdout_points_target"])
    nregions = len(GATE_REGIONS)
    per_region = max(16, target//nregions)

    global_pick = _choose_even(np.arange(len(P)), per_region)
    chosen = set(int(i) for i in global_pick)

    for region in GATE_REGIONS:
        if region == "global":
            continue
        idx = np.flatnonzero(full_masks.get(region, False))
        for i in _choose_even(idx, per_region):
            chosen.add(int(i))

    selected = np.asarray(sorted(chosen), dtype=np.int64)

    mask_matrix = np.zeros(
        (len(GATE_REGIONS), len(selected)),
        dtype=np.uint8,
    )
    # "global" is deliberately the independently uniform subset only; this
    # avoids allowing the region-balanced oversampling to bias the global gate.
    global_set = set(int(i) for i in global_pick)
    mask_matrix[0] = np.asarray(
        [int(i) in global_set for i in selected],
        dtype=np.uint8,
    )

    for ir, region in enumerate(GATE_REGIONS[1:], start=1):
        m = full_masks.get(region, np.zeros(len(P), dtype=bool))
        mask_matrix[ir] = m[selected].astype(np.uint8)

    return {
        "points": P[selected],
        "normals": N[selected],
        "mask_matrix": mask_matrix,
        "region_names": list(GATE_REGIONS),
        "full_candidate_count": int(len(P)),
        "selected_count": int(len(selected)),
    }


# -----------------------------------------------------------------------------
# PIC RECORD stage
# -----------------------------------------------------------------------------

def make_pic_config(out_name, settings):
    from _jefipic_mas_case_common import BaseConfig

    cfg = BaseConfig(output_dir=out_name)
    cfg.dt = DT
    cfg.f_drive = F_DRIVE
    cfg.current_amplitude = 0.0
    cfg.ramp_periods = 0.0

    cfg.total_length = float(settings["total_length_m"])
    cfg.radius = 1.0e-3
    cfg.feed_gap = 2.0e-3

    cfg.steps = steps_from_cycles(settings["cycles"])
    cfg.len_time_snapshots = int(cfg.steps + 16)
    cfg.diag_stride = int(settings["diag_stride"])

    cfg.particles_per_species = int(settings["particles_per_species"])
    cfg.grid_n = int(settings["grid_n"])

    cfg.jefipic_unit_mode = "LHQCD"
    cfg.jefipic_kernel_mode = "atomic"
    # Keep the same validated plasma kernel as Step 4B. C3 changes only the MAS side.
    cfg.quasi_electrostatic = True

    cfg.spatial_truncation_debye = 0.0
    cfg.taper_width_debye = 0.0
    cfg.taper = "none"
    cfg.point_radius = None

    cfg.quiet_start_pairs = True
    cfg.center_initial_velocities = True
    cfg.velocity_quiet_start_pairs = True
    cfg.particle_reinjection_mode = "boundary"
    cfg.initial_antenna_exclusion_debye = 0.5
    cfg.boundary_reinjection_antenna_exclusion_debye = 0.25
    cfg.plasma_to_mas_ramp_steps = 0

    cfg.particle_order = 1
    cfg.field_order = 1
    return cfg


def lightweight_probe_holder(g):
    return SimpleNamespace(
        geometry=g,
        source_softening=0.0,
        source_truncation_radius=0.0,
        source_taper_type=0,
        source_taper_width=0.0,
    )


def run_record(args):
    import _jefipic_mas_case_common as common

    settings = mode_settings(bool(args.quick))
    if args.cycles is not None:
        settings["cycles"] = float(args.cycles)
    if args.particles is not None:
        settings["particles_per_species"] = int(args.particles)
    if args.grid_n is not None:
        settings["grid_n"] = int(args.grid_n)

    out = Path("article_cases_output")/args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = make_pic_config(args.out, settings)

    frozen_g = build_geometry(settings)
    fp = geometry_fingerprint(frozen_g)

    # Override only the object used by build_pic_solver_mas for source
    # registration.  No old MAS time recurrence exists in RECORD.
    def _build_probe_device(
        cfg_local,
        stl_path,
        grid_shape,
        spacing,
        origin,
        *,
        current_amplitude=None,
    ):
        del cfg_local, stl_path, grid_shape, spacing, origin, current_amplitude
        return lightweight_probe_holder(frozen_g)

    common.build_mas_device = _build_probe_device

    print("[Step 4C3 RECORD] building PIC/Jefimenko bundle ...", flush=True)
    bundle = common.build_pic_solver_mas(
        cfg,
        out/"IGNORED_FROZEN_ROUNDED_GEOMETRY.stl",
    )

    PIC = bundle["PIC"]
    solver = bundle["solver"]
    mas_holder = bundle["mas"]
    rng = bundle["rng"]
    bridge = bundle["unit_bridge"]
    nx, ny, nz = bundle["shape"]
    dx, dy, dz = bundle["spacing"]
    Ex, Ey, Ez, Bx, By, Bz = bundle["fields"]

    g = mas_holder.geometry
    if geometry_fingerprint(g) != fp:
        raise RuntimeError("RECORD geometry fingerprint changed inside PIC builder")

    print("[Step 4C3 RECORD] solving validated vacuum probe phasor ...", flush=True)
    Khat, Phat, vacuum_collocation = build_vacuum_probe_phasor(g)
    if vacuum_collocation >= BOUNDARY_GATE:
        raise RuntimeError(
            "External fixed probe source failed the frozen 5% collocation gate: "
            f"{vacuum_collocation}"
        )

    holdout = build_stratified_holdout(settings)
    solve_obs_code = bridge.length_si_to_code(g.solve_obs_points)
    holdout_code = bridge.length_si_to_code(holdout["points"])

    steps = int(cfg.steps)
    ncoll = int(g.num_collocation_points)

    plasma_coll = np.empty(
        (steps, ncoll, 3),
        dtype=np.float32,
    )

    diag_steps = np.arange(
        0,
        steps,
        int(settings["diag_stride"]),
        dtype=np.int64,
    )
    if len(diag_steps) == 0 or int(diag_steps[-1]) != steps-1:
        diag_steps = np.append(diag_steps, steps-1)
    diag_lookup = {
        int(s): i
        for i, s in enumerate(diag_steps)
    }
    plasma_holdout = np.empty(
        (len(diag_steps), len(holdout["points"]), 3),
        dtype=np.float32,
    )

    plasma_rms = np.empty(steps, dtype=np.float64)
    omega = 2.0*np.pi*F_DRIVE
    finite = True
    completed = 0

    print(
        f"[Step 4C3 RECORD] mode={settings['mode']} "
        f"steps={steps}, particles/species={cfg.particles_per_species}, "
        f"grid={cfg.grid_n}^3, sources/arm={settings['sources_per_arm']}, "
        f"collocation={ncoll}, holdout={len(holdout['points'])}",
        flush=True,
    )

    for step in range(steps):
        PIC.proceed_one_step(
            Ex, Ey, Ez, Bx, By, Bz,
            dx, dy, dz,
            nx, ny, nz,
            cfg.field_order,
        )
        common.absorb_particles_on_antenna(
            PIC, cfg, rng
        )

        rho, Jx_face, Jy_face, Jz_face = PIC.evaluate_rho_J(
            dx, dy, dz,
            nx, ny, nz,
            cfg.particle_order,
        )
        solver.Jefimenko_solver(
            rho.flatten(),
            include_antenna=False,
            face_currents=(Jx_face, Jy_face, Jz_face),
        )

        Ecode = solver.sample_current_E_at_points(solve_obs_code)
        Esi = np.asarray(
            bridge.efield_code_to_si(Ecode),
            dtype=np.float64,
        ).reshape(ncoll, 3)

        if not np.all(np.isfinite(Esi)):
            finite = False
            break

        plasma_coll[step] = Esi.astype(np.float32)
        plasma_rms[step] = float(
            np.sqrt(np.mean(np.sum(Esi*Esi, axis=1)))
        )

        if step in diag_lookup:
            Hcode = solver.sample_current_E_at_points(holdout_code)
            Hsi = np.asarray(
                bridge.efield_code_to_si(Hcode),
                dtype=np.float64,
            ).reshape(len(holdout["points"]), 3)
            if not np.all(np.isfinite(Hsi)):
                finite = False
                break
            plasma_holdout[diag_lookup[step]] = Hsi.astype(np.float32)

        # Fixed external probe source for the NEXT PIC step.
        #
        # We ramp the polarization itself and obtain K=dP/dt analytically,
        # preserving source continuity during the ramp:
        #
        #   P(t)=r(t) Re[P_hat exp(iwt)]
        #   K(t)=r Re[K_hat exp(iwt)] + rdot Re[P_hat exp(iwt)].
        t_s = float(step*DT)
        ramp, ramp_dot = smoothstep_ramp_and_derivative(t_s)
        phase = np.exp(1j*omega*t_s)

        P_now = ramp*np.real(Phat*phase)
        K_now = (
            ramp*np.real(Khat*phase)
            + ramp_dot*np.real(Phat*phase)
        )

        solver.recompute_current_fields(
            antenna_surface_currents=
                bridge.surface_current_density_si_to_code(K_now),
            antenna_polarization=
                bridge.surface_polarization_density_si_to_code(P_now),
            include_antenna=True,
        )
        Ex, Ey, Ez, Bx, By, Bz = solver.acquire_EB_field()

        completed = step + 1
        if (
            step % max(1, int(settings["diag_stride"])) == 0
            or step == steps-1
        ):
            print(
                f"[Step 4C3 RECORD] step {step+1}/{steps}, "
                f"plasma E_surface rms={plasma_rms[step]:.6e} V/m",
                flush=True,
            )

    # Trim on failure so the saved file remains internally consistent.
    plasma_coll = plasma_coll[:completed]
    plasma_rms = plasma_rms[:completed]
    kept_diag = diag_steps < completed
    diag_steps = diag_steps[kept_diag]
    plasma_holdout = plasma_holdout[:len(diag_steps)]

    metadata = {
        **settings,
        "step": "4C3",
        "record_stage": True,
        "dt_s": DT,
        "frequency_Hz": F_DRIVE,
        "feed_field_amplitude_V_per_m": FEED_FIELD_AMPLITUDE_V_PER_M,
        "ramp_cycles": RAMP_CYCLES,
        "depth_frac": DEPTH_FRAC,
        "collocation_factor": COLLOCATION_FACTOR,
        "lambda_rel": LAMBDA_REL,
        "operator_softening_distance_m": 0.0,
        "runtime_source_softening_m": 0.0,
        "geometry_fingerprint": fp,
        "num_sources_total": int(g.num_sources),
        "num_collocation_points": int(g.num_collocation_points),
        "holdout_region_names": list(holdout["region_names"]),
        "holdout_full_candidate_count":
            int(holdout["full_candidate_count"]),
        "holdout_selected_count":
            int(holdout["selected_count"]),
        "plasma_field_storage_dtype": "float32",
        "plasma_field_dynamics_projection":
            "none_full_time_domain_samples_saved",
        "fixed_external_probe_source":
            "3MHz harmonic vacuum MAS source; plasma response not assumed harmonic",
        "quasi_electrostatic_plasma":
            bool(cfg.quasi_electrostatic),
    }

    forcing_path = out/"step4c3_forcing.npz"
    np.savez_compressed(
        forcing_path,
        metadata_json=np.asarray(json.dumps(metadata)),
        time_s=np.arange(completed, dtype=np.float64)*DT,
        plasma_collocation_E_V_per_m=plasma_coll,
        plasma_surface_rms_V_per_m=plasma_rms,
        diag_steps=diag_steps.astype(np.int64),
        plasma_holdout_E_V_per_m=plasma_holdout,
        holdout_points_m=np.asarray(
            holdout["points"], dtype=np.float64
        ),
        holdout_normals=np.asarray(
            holdout["normals"], dtype=np.float64
        ),
        holdout_mask_matrix=np.asarray(
            holdout["mask_matrix"], dtype=np.uint8
        ),
    )

    accepted = bool(
        completed == steps
        and finite
        and vacuum_collocation < BOUNDARY_GATE
    )

    report = {
        "step": "4C3",
        "stage": "record",
        "accepted": accepted,
        "mode": settings["mode"],
        "vacuum_external_probe_collocation_rel_l2":
            float(vacuum_collocation),
        "steps_requested": steps,
        "steps_completed": int(completed),
        "all_recorded_plasma_fields_finite": bool(finite),
        "geometry_fingerprint": fp,
        "num_collocation_points": int(ncoll),
        "num_holdout_points": int(len(holdout["points"])),
        "forcing_file": str(forcing_path),
        "forcing_file_contains_full_time_domain_plasma_field": True,
        "single_frequency_projection_used": False,
        "old_MAS_recurrence_used": False,
        "response_values_are_manuscript_final": False,
        "decision": (
            "STEP4C3_RECORD_PASS_run_offline_implicit_replay"
            if accepted else
            "STEP4C3_RECORD_FAIL_stop_before_replay"
        ),
    }
    (out/"step4c3_record_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    return forcing_path, accepted


# -----------------------------------------------------------------------------
# Independent time-local holdout field evaluator
# -----------------------------------------------------------------------------

def evaluate_time_local_dipole_field(
    points,
    g,
    P,
    K,
    Kdot,
    *,
    obs_chunk=128,
    src_chunk=128,
):
    points = np.asarray(points, dtype=np.float64)
    SP = np.asarray(g.ant_src_points, dtype=np.float64)
    SW = np.asarray(g.source_weights, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    Kdot = np.asarray(Kdot, dtype=np.float64)

    out = np.zeros((len(points), 3), dtype=np.float64)
    ce = 1.0/(4.0*np.pi*EPS0)

    for oa in range(0, len(points), int(obs_chunk)):
        ob = min(len(points), oa + int(obs_chunk))
        X = points[oa:ob]
        Eb = np.zeros((len(X), 3), dtype=np.float64)

        for sa in range(0, len(SP), int(src_chunk)):
            sb = min(len(SP), sa + int(src_chunk))
            r = X[:, None, :] - SP[None, sa:sb, :]
            R = np.maximum(
                np.linalg.norm(r, axis=2),
                1.0e-15,
            )
            w = SW[None, sa:sb, None]

            Ps = P[sa:sb]
            Ks = K[sa:sb]
            Ds = Kdot[sa:sb]

            rp = np.sum(r*Ps[None, :, :], axis=2)
            rk = np.sum(r*Ks[None, :, :], axis=2)
            rd = np.sum(r*Ds[None, :, :], axis=2)

            ep = (
                3.0*r*rp[:, :, None]/R[:, :, None]**5
                - Ps[None, :, :]/R[:, :, None]**3
            )
            ek = (
                3.0*r*rk[:, :, None]/R[:, :, None]**4
                - Ks[None, :, :]/R[:, :, None]**2
            )/C0
            ed = (
                r*rd[:, :, None]/R[:, :, None]**3
                - Ds[None, :, :]/R[:, :, None]
            )/(C0*C0)

            Eb += np.sum(
                ce*w*(ep + ek + ed),
                axis=1,
            )
        out[oa:ob] = Eb

    return out


def tangential_sq(field, normals):
    field = np.asarray(field, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    et = field - normals*np.sum(
        field*normals,
        axis=1,
    )[:, None]
    return np.sum(et*et, axis=1)


# -----------------------------------------------------------------------------
# OFFLINE implicit replay
# -----------------------------------------------------------------------------

def _load_forcing(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["metadata_json"].item()))
    return z, meta


def _feed_at_time(points, t_s):
    ramp, _ = smoothstep_ramp_and_derivative(t_s)
    return (
        ramp
        * math.cos(2.0*np.pi*F_DRIVE*t_s)
        * feed_shape(points)
    )


def _operator_self_coeff(prepared, q):
    # As = diag(rs) A_node diag(1/cs)
    # => A_node q = diag(1/rs) As (q*cs).
    return (
        prepared["As"] @ (np.asarray(q)*prepared["cs"])
    ) / prepared["rs"]


def run_replay(args, forcing_path):
    forcing_path = Path(forcing_path)
    z, meta = _load_forcing(forcing_path)

    out = Path("article_cases_output")/args.out
    out.mkdir(parents=True, exist_ok=True)

    settings = {
        k: meta[k]
        for k in (
            "mode",
            "total_length_m",
            "sources_per_arm",
            "n_phi",
            "n_cyl",
            "n_hemi",
            "cycles",
            "analysis_start_cycles",
            "diag_stride",
            "holdout_points_target",
        )
    }

    print("[Step 4C3 REPLAY] rebuilding frozen geometry ...", flush=True)
    g = build_geometry(settings)
    fp = geometry_fingerprint(g)
    if fp != meta["geometry_fingerprint"]:
        raise RuntimeError(
            "Forcing/replay geometry fingerprint mismatch. "
            f"record={meta['geometry_fingerprint']} replay={fp}"
        )

    if abs(float(meta["operator_softening_distance_m"])) > 1.0e-18:
        raise RuntimeError("Forcing metadata is not zero-softening frozen geometry")

    plasma_coll = z["plasma_collocation_E_V_per_m"]
    times = np.asarray(z["time_s"], dtype=np.float64)
    diag_steps = np.asarray(z["diag_steps"], dtype=np.int64)
    plasma_holdout = z["plasma_holdout_E_V_per_m"]
    Hpoints = np.asarray(z["holdout_points_m"], dtype=np.float64)
    Hnormals = np.asarray(z["holdout_normals"], dtype=np.float64)
    Hmask = np.asarray(z["holdout_mask_matrix"], dtype=np.uint8).astype(bool)
    region_names = list(meta["holdout_region_names"])

    if len(plasma_coll) != len(times):
        raise RuntimeError("Forcing time and collocation arrays have different lengths")
    if plasma_coll.shape[1] != g.num_collocation_points:
        raise RuntimeError(
            "Forcing collocation point count does not match replay geometry"
        )
    if Hmask.shape != (len(region_names), len(Hpoints)):
        raise RuntimeError("Holdout mask matrix shape mismatch")
    if len(plasma_holdout) != len(diag_steps):
        raise RuntimeError("Holdout forcing and diagnostic-step count mismatch")

    diag_lookup = {
        int(s): i
        for i, s in enumerate(diag_steps)
    }

    print(
        "[Step 4C3 REPLAY] assembling zero-softening tangent operators "
        "(Gp, Gkdot, G_BE) ...",
        flush=True,
    )
    Ap, Ad, A_node = assemble_projected_be_operators(
        g,
        obs_chunk=8 if settings["mode"] == "formal" else 16,
    )
    prepared = prepare_real_be_solver_inplace(
        A_node,
        g.solve_weights,
    )
    # A_node storage is now prepared["As"]; do not keep another alias.
    del A_node
    gc.collect()

    nsrc_dof = 2*g.num_sources
    q_prev = np.zeros(nsrc_dof, dtype=np.float64)
    p_prev = np.zeros_like(q_prev)

    feed_coll_shape = feed_shape(g.solve_obs_points)

    analysis_start_s = (
        float(settings["analysis_start_cycles"])/F_DRIVE
    )

    coll_residual = np.full(len(times), np.nan, dtype=np.float64)
    incident_rms = np.full(len(times), np.nan, dtype=np.float64)
    current_rms = np.full(len(times), np.nan, dtype=np.float64)
    state_norms = np.full(len(times), np.nan, dtype=np.float64)

    coll_res_sum = 0.0
    coll_inc_sum = 0.0
    coll_count = 0

    hold_res_sum = {
        r: 0.0 for r in region_names
    }
    hold_inc_sum = {
        r: 0.0 for r in region_names
    }
    hold_count = {
        r: 0 for r in region_names
    }

    harmonic_freqs = (3.0e6, 6.0e6, 9.0e6)
    qhat_sum = {
        f: np.zeros(nsrc_dof, dtype=np.complex128)
        for f in harmonic_freqs
    }
    harmonic_count = 0

    finite = True
    completed = 0

    print(
        f"[Step 4C3 REPLAY] replaying {len(times)} full broadband time nodes; "
        "no frequency projection",
        flush=True,
    )

    for step, t_s in enumerate(times):
        plasma = np.asarray(
            plasma_coll[step],
            dtype=np.float64,
        )
        feed = _feed_at_time(
            g.solve_obs_points,
            float(t_s),
        )
        incident = plasma + feed
        e = tangent_coeffs(
            incident,
            g.solve_tangent1,
            g.solve_tangent2,
        )

        history = (
            Ap @ p_prev
            - (Ad @ q_prev)/DT
        )
        rhs = -(e + history)
        q = solve_real_be(prepared, rhs)
        p = p_prev + DT*q

        if (
            not np.all(np.isfinite(q))
            or not np.all(np.isfinite(p))
        ):
            finite = False
            break

        node_self = _operator_self_coeff(
            prepared, q
        )
        residual_coeff = e + history + node_self

        inc_norm2 = float(np.dot(e, e))
        res_norm2 = float(
            np.dot(residual_coeff, residual_coeff)
        )
        coll_residual[step] = math.sqrt(
            res_norm2/max(inc_norm2, TINY)
        )
        incident_rms[step] = math.sqrt(
            inc_norm2/max(len(e)//2, 1)
        )
        current_rms[step] = weighted_coeff_rms(g, q)
        state_norms[step] = state_norm(g, q, p)

        if t_s >= analysis_start_s:
            coll_res_sum += res_norm2
            coll_inc_sum += inc_norm2
            coll_count += len(e)

            for f in harmonic_freqs:
                qhat_sum[f] += (
                    2.0*q*np.exp(-1j*2.0*np.pi*f*t_s)
                )
            harmonic_count += 1

        if step in diag_lookup and t_s >= analysis_start_s:
            hi = diag_lookup[step]
            plasma_H = np.asarray(
                plasma_holdout[hi],
                dtype=np.float64,
            )
            feed_H = _feed_at_time(Hpoints, float(t_s))
            incident_H = plasma_H + feed_H

            K = coeff_to_vectors(g, q)
            P = coeff_to_vectors(g, p)
            K_prev = coeff_to_vectors(g, q_prev)
            Kdot = (K-K_prev)/DT

            self_H = evaluate_time_local_dipole_field(
                Hpoints,
                g,
                P,
                K,
                Kdot,
            )
            total_H = incident_H + self_H
            res_sq = tangential_sq(total_H, Hnormals)
            inc_sq = tangential_sq(incident_H, Hnormals)

            for ir, region in enumerate(region_names):
                mask = Hmask[ir]
                if not np.any(mask):
                    continue
                hold_res_sum[region] += float(np.sum(res_sq[mask]))
                hold_inc_sum[region] += float(np.sum(inc_sq[mask]))
                hold_count[region] += int(np.count_nonzero(mask))

        q_prev = q
        p_prev = p
        completed = step + 1

        if (
            step % max(1, int(settings["diag_stride"])) == 0
            or step == len(times)-1
        ):
            print(
                f"[Step 4C3 REPLAY] step {step+1}/{len(times)}, "
                f"colloc={coll_residual[step]:.6g}, "
                f"K_rms={current_rms[step]:.6e}, "
                f"state={state_norms[step]:.6e}",
                flush=True,
            )

    if coll_inc_sum <= 0.0:
        aggregate_collocation = float("inf")
    else:
        aggregate_collocation = math.sqrt(
            coll_res_sum/coll_inc_sum
        )

    # Aggregate independent holdout metrics.
    if hold_count.get("global", 0) > 0:
        global_inc_rms = math.sqrt(
            hold_inc_sum["global"]/hold_count["global"]
        )
    else:
        global_inc_rms = 0.0

    holdout_metrics = {}
    holdout_rows = []
    for region in region_names:
        count = hold_count[region]
        if count <= 0:
            continue
        rr = math.sqrt(hold_res_sum[region]/count)
        ii = math.sqrt(hold_inc_sum[region]/count)
        incident_ratio = ii/max(global_inc_rms, TINY)
        local_rel = rr/max(ii, TINY)
        global_norm = rr/max(global_inc_rms, TINY)

        if (
            region == "global"
            or incident_ratio >= WEAK_REGION_THRESHOLD
        ):
            gate_metric = "local_rel"
            gate_value = local_rel
        else:
            gate_metric = "global_norm"
            gate_value = global_norm

        row = {
            "region": region,
            "residual_rms": rr,
            "incident_rms": ii,
            "incident_ratio": incident_ratio,
            "local_rel": local_rel,
            "global_norm": global_norm,
            "gate_metric": gate_metric,
            "gate_value": gate_value,
            "passed": bool(gate_value < BOUNDARY_GATE),
            "sample_count_accumulated": int(count),
        }
        holdout_metrics[region] = row
        holdout_rows.append(row)

    worst_holdout = (
        max(holdout_rows, key=lambda r: r["gate_value"])
        if holdout_rows else None
    )
    worst_holdout_value = (
        float(worst_holdout["gate_value"])
        if worst_holdout is not None else float("inf")
    )

    # Zero-input tail from the ACTUAL final forced state.
    tail_norms = [state_norm(g, q_prev, p_prev)]
    for _ in range(TAIL_STEPS):
        history = (
            Ap @ p_prev
            - (Ad @ q_prev)/DT
        )
        q = solve_real_be(prepared, -history)
        p = p_prev + DT*q
        if (
            not np.all(np.isfinite(q))
            or not np.all(np.isfinite(p))
        ):
            finite = False
            break
        tail_norms.append(state_norm(g, q, p))
        q_prev, p_prev = q, p

    ratios = []
    for a, b in zip(tail_norms[:-1], tail_norms[1:]):
        if np.isfinite(a) and np.isfinite(b) and a > TINY:
            ratios.append(b/a)
    tail_slice = ratios[-min(20, len(ratios)):]
    tail_growth = (
        float(np.median(tail_slice))
        if tail_slice else float("inf")
    )

    harmonic_response = {}
    if harmonic_count > 0:
        for f in harmonic_freqs:
            qhat = qhat_sum[f]/harmonic_count
            harmonic_response[f"{f/1e6:.0f}MHz"] = {
                "weighted_current_phasor_rms_A_per_m":
                    weighted_coeff_rms(g, qhat)
            }

    accepted = bool(
        completed == len(times)
        and finite
        and aggregate_collocation < BOUNDARY_GATE
        and worst_holdout_value < BOUNDARY_GATE
        and tail_growth <= TAIL_GROWTH_GATE
        and fp == meta["geometry_fingerprint"]
    )

    report = {
        "step": "4C3",
        "stage": "replay",
        "accepted": accepted,
        "mode": settings["mode"],
        "forcing_file": str(forcing_path),
        "forcing_geometry_fingerprint":
            meta["geometry_fingerprint"],
        "replay_geometry_fingerprint": fp,
        "geometry_fingerprint_match":
            bool(fp == meta["geometry_fingerprint"]),
        "old_MAS_recurrence_used": False,
        "single_frequency_projection_used": False,
        "implicit_update":
            "P^n=P^(n-1)+dt*K^n; Kdot^n=(K^n-K^(n-1))/dt",
        "operator_softening_distance_m": 0.0,
        "steps_requested": int(len(times)),
        "steps_completed": int(completed),
        "all_replay_states_finite": bool(finite),
        "analysis_start_s": float(analysis_start_s),
        "analysis_start_cycles":
            float(settings["analysis_start_cycles"]),
        "time_aggregated_collocation_rel_l2":
            float(aggregate_collocation),
        "collocation_gate": BOUNDARY_GATE,
        "independent_holdout": {
            "type": "stratified independent barycentric surface holdout",
            "num_unique_points": int(len(Hpoints)),
            "weak_region_threshold": WEAK_REGION_THRESHOLD,
            "regions": holdout_metrics,
            "worst_gate": worst_holdout,
            "gate": BOUNDARY_GATE,
        },
        "zero_input_tail": {
            "steps": TAIL_STEPS,
            "median_growth_factor_last_20": tail_growth,
            "growth_gate": TAIL_GROWTH_GATE,
            "initial_state_norm": float(tail_norms[0]),
            "final_state_norm": float(tail_norms[-1]),
        },
        "diagnostic_harmonic_response": harmonic_response,
        "response_values_are_manuscript_final": False,
        "decision": (
            "STEP4C3_PASS_proceed_to_STEP4D_two_way_broadband_nonlinear_coupling"
            if accepted else
            "STEP4C3_FAIL_do_not_close_the_loop"
        ),
        "scope_note": (
            "C3 validates one-way broadband plasma forcing only. "
            "The implicit MAS response is not fed back to PIC, so no "
            "self-consistent amplification factor is claimed."
        ),
    }
    (out/"step4c3_replay_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # Small diagnostic file; the large forcing NPZ can be excluded from the
    # result ZIP after the run.
    np.savez_compressed(
        out/"step4c3_diagnostics.npz",
        time_s=times[:completed],
        collocation_rel_l2=coll_residual[:completed],
        incident_tangent_rms=incident_rms[:completed],
        current_weighted_rms_A_per_m=current_rms[:completed],
        state_norm=state_norms[:completed],
        tail_state_norm=np.asarray(tail_norms, dtype=np.float64),
    )

    lines = [
        "Step 4C3 one-way broadband PIC -> implicit MAS",
        f"accepted: {accepted}",
        f"mode: {settings['mode']}",
        f"steps: {completed}/{len(times)}",
        f"aggregate collocation residual: {aggregate_collocation:.6g}",
        f"worst holdout gate: {worst_holdout_value:.6g}",
        f"tail growth: {tail_growth:.12g}",
        f"decision: {report['decision']}",
    ]
    (out/"step4c3_summary.txt").write_text(
        "\n".join(lines)+"\n",
        encoding="utf-8",
    )
    print("\n".join(lines))
    return accepted


# -----------------------------------------------------------------------------
# CPU-only self-test
# -----------------------------------------------------------------------------

def self_test():
    # ------------------------------------------------------------------
    # 1) Operator-equivalence test on a tiny geometry.
    # ------------------------------------------------------------------
    # Tiny density is appropriate here because this test compares two
    # implementations of the SAME matrix; it is not a physical PEC gate.
    settings_small = mode_settings(True)
    settings_small = dict(settings_small)
    settings_small.update({
        "sources_per_arm": 8,
        "n_phi": 12,
        "n_cyl": 12,
        "n_hemi": 6,
    })
    g0 = build_geometry(settings_small)

    Ap, Ad, Abe = assemble_projected_be_operators(
        g0, obs_chunk=8
    )

    g1 = build_geometry(settings_small)
    assemble_impedance_matrix(
        g1,
        dt=DT,
        epsilon0=EPS0,
        c=C0,
        softening_distance=0.0,
        spatial_solver="tangential_weighted_scaled",
        tikhonov_rel=LAMBDA_REL,
        svd_rcond=1.0e-10,
    )
    B = _basis_matrix(
        g1.source_tangent1,
        g1.source_tangent2,
    )
    C = _constraint_matrix(
        g1.solve_tangent1,
        g1.solve_tangent2,
    )
    Ap_ref = C @ g1.G_p @ B
    Ad_ref = C @ g1.G_kdot @ B
    Abe_ref = C @ (
        DT*g1.G_p + g1.G_k + g1.G_kdot/DT
    ) @ B

    rel_ap = float(
        np.linalg.norm(Ap-Ap_ref)
        / max(float(np.linalg.norm(Ap_ref)), TINY)
    )
    rel_ad = float(
        np.linalg.norm(Ad-Ad_ref)
        / max(float(np.linalg.norm(Ad_ref)), TINY)
    )
    rel_be = float(
        np.linalg.norm(Abe-Abe_ref)
        / max(float(np.linalg.norm(Abe_ref)), TINY)
    )

    # Free the tiny Cartesian reference before the physical replay smoke.
    del g0, g1, Ap, Ad, Abe, Ap_ref, Ad_ref, Abe_ref, B, C
    gc.collect()

    # ------------------------------------------------------------------
    # 2) Physical BE replay smoke at a density high enough to satisfy PEC.
    # ------------------------------------------------------------------
    # 96 sources/arm is still light enough for a fast CPU test, but unlike
    # the 8-source implementation check it is not grossly under-resolved.
    settings_feed = mode_settings(True)
    settings_feed = dict(settings_feed)
    settings_feed.update({
        "sources_per_arm": 96,
        "n_phi": 28,
        "n_cyl": 28,
        "n_hemi": 10,
    })
    gf = build_geometry(settings_feed)
    Apf, Adf, Abef = assemble_projected_be_operators(
        gf, obs_chunk=16
    )
    prepared = prepare_real_be_solver_inplace(
        Abef,
        gf.solve_weights,
    )
    q_prev = np.zeros(2*gf.num_sources)
    p_prev = np.zeros_like(q_prev)

    residual_sum = 0.0
    incident_sum = 0.0
    finite = True
    nsteps = 420

    for step in range(nsteps):
        t = step*DT
        inc = _feed_at_time(gf.solve_obs_points, t)
        e = tangent_coeffs(
            inc,
            gf.solve_tangent1,
            gf.solve_tangent2,
        )
        history = Apf @ p_prev - (Adf @ q_prev)/DT
        q = solve_real_be(
            prepared,
            -(e+history),
        )
        p = p_prev + DT*q
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            finite = False
            break
        residual = (
            e + history + _operator_self_coeff(prepared, q)
        )
        if t >= 0.5/F_DRIVE:
            residual_sum += float(np.dot(residual, residual))
            incident_sum += float(np.dot(e, e))
        q_prev, p_prev = q, p

    aggregate = math.sqrt(
        residual_sum/max(incident_sum, TINY)
    )

    # 3) Tail check from the actual final driven state.
    tail = [state_norm(gf, q_prev, p_prev)]
    for _ in range(40):
        history = Apf @ p_prev - (Adf @ q_prev)/DT
        q = solve_real_be(prepared, -history)
        p = p_prev + DT*q
        tail.append(state_norm(gf, q, p))
        q_prev, p_prev = q, p

    ratios = [
        b/a for a, b in zip(tail[:-1], tail[1:])
        if a > TINY
    ]
    tail_growth = float(np.median(ratios[-20:]))

    passed = bool(
        rel_ap < 1.0e-12
        and rel_ad < 1.0e-12
        and rel_be < 1.0e-12
        and finite
        and aggregate < BOUNDARY_GATE
        and tail_growth <= TAIL_GROWTH_GATE
    )

    report = {
        "self_test_pass": passed,
        "operator_equivalence_density_sources_per_arm": 8,
        "direct_vs_cartesian_projected_relative_error": {
            "Gp": rel_ap,
            "Gkdot": rel_ad,
            "G_BE": rel_be,
        },
        "feed_replay_density_sources_per_arm": 96,
        "feed_only_aggregate_collocation_rel_l2": aggregate,
        "tail_growth": tail_growth,
    }
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("Step 4C3 CPU self-test failed")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        choices=("record", "replay", "all"),
        default="all",
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument(
        "--out",
        default="step4c3_formal",
    )
    p.add_argument(
        "--forcing",
        default=None,
        help="For replay: path to step4c3_forcing.npz. "
             "Default: article_cases_output/<out>/step4c3_forcing.npz",
    )
    p.add_argument(
        "--cycles",
        type=float,
        default=None,
        help="Optional RECORD-only diagnostic override.",
    )
    p.add_argument(
        "--particles",
        type=int,
        default=None,
        help="Optional RECORD-only diagnostic override.",
    )
    p.add_argument(
        "--grid-n",
        type=int,
        default=None,
        help="Optional RECORD-only diagnostic override.",
    )
    a = p.parse_args()

    if a.self_test:
        self_test()
        return

    forcing_default = (
        Path("article_cases_output")/a.out/"step4c3_forcing.npz"
    )

    if a.stage == "record":
        _, ok = run_record(a)
        if not ok:
            raise RuntimeError("Step 4C3 RECORD failed; do not run replay.")
        return

    if a.stage == "replay":
        forcing = (
            Path(a.forcing)
            if a.forcing is not None
            else forcing_default
        )
        ok = run_replay(a, forcing)
        if not ok:
            raise RuntimeError("Step 4C3 REPLAY failed.")
        return

    # all
    forcing, ok_record = run_record(a)
    if not ok_record:
        raise RuntimeError("Step 4C3 RECORD failed; replay was not started.")

    # Encourage release of PIC/CUDA objects before the large CPU replay.
    gc.collect()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

    ok_replay = run_replay(a, forcing)
    if not ok_replay:
        raise RuntimeError("Step 4C3 REPLAY failed.")


if __name__ == "__main__":
    main()
