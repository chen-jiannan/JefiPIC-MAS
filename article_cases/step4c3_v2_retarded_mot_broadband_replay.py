#!/usr/bin/env python3
"""
Step 4C3 V2 — retarded-MOT broadband PIC-forcing replay.

Reuses the already recorded Step-4C3 broadband PIC/Jefimenko forcing file.
No PIC is rerun. The only formulation change relative to the failed C3 replay is:

    time-local BE  ->  causal one-step retarded MOT validated in C1/C2 V4.

Frozen gates:
- all replay states finite;
- time-aggregated collocation tangential PEC residual < 5%;
- worst stratified independent-holdout physical gate < 5%;
- 200-step zero-input tail, median growth over final 50 steps <= 1.001;
- forcing/replay geometry fingerprint exact match;
- max pairwise delay < dt for collocation and holdout points.

No harmonic projection is used in the dynamics.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

import step2mb1_incident_normalization_feed_localization as mb1
from EMsolver.mas_preprocessor import generate_rounded_two_arm_dipole_points

EPS0 = 8.8541878128e-12
C0 = 299792458.0
DT = 1.0e-9
TINY = 1.0e-300

F_DRIVE = 3.0e6
DEPTH_FRAC = 0.80
COLLOCATION_FACTOR = 2.0
LAMBDA_REL = 1.0e-4
FEED_FIELD_AMPLITUDE_V_PER_M = 1.0
RAMP_CYCLES = 0.50

BOUNDARY_GATE = 0.05
TAIL_GROWTH_GATE = 1.001
WEAK_REGION_THRESHOLD = 0.20
TAIL_STEPS = 200
TAIL_WINDOW = 50

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


def pair_blocks(obs, src, weights):
    r = obs[:, None, :] - src[None, :, :]
    R = np.linalg.norm(r, axis=2)
    if np.any(R <= 0.0):
        raise RuntimeError("zero source-observation distance")
    rr = r[:, :, :, None]*r[:, :, None, :]
    I = np.eye(3, dtype=np.float64)[None, None, :, :]
    W = np.asarray(weights, dtype=np.float64)[None, :, None, None]
    ce = 1.0/(4.0*np.pi*EPS0)
    Gp = ce*W*(3.0*rr/R[:, :, None, None]**5 - I/R[:, :, None, None]**3)
    Gk = ce*W*(3.0*rr/R[:, :, None, None]**4 - I/R[:, :, None, None]**2)/C0
    Gd = ce*W*(rr/R[:, :, None, None]**3 - I/R[:, :, None, None])/(C0*C0)
    return Gp, Gk, Gd, R


def project_pair_tensor(obs_tangent, pair_tensor, src_tangent):
    out = np.einsum(
        "mua,mjab,jvb->mujv",
        obs_tangent,
        pair_tensor,
        src_tangent,
        optimize=True,
    )
    return out.reshape(2*len(obs_tangent), 2*len(src_tangent))


def assemble_retarded_mot_operators(g, obs_chunk):
    """Return Hp, Hk, A_now for the exact C1-V4 one-step MOT recurrence."""
    obs = np.asarray(g.solve_obs_points, dtype=np.float64)
    src = np.asarray(g.ant_src_points, dtype=np.float64)
    weights = np.asarray(g.source_weights, dtype=np.float64)
    ot = np.stack([g.solve_tangent1, g.solve_tangent2], axis=1)
    st = np.stack([g.source_tangent1, g.source_tangent2], axis=1)

    Hp = np.empty((2*len(obs), 2*len(src)), dtype=np.float64)
    Hk = np.empty_like(Hp)
    An = np.empty_like(Hp)
    rmin = math.inf
    rmax = 0.0
    amin = math.inf
    amax = -math.inf

    for a in range(0, len(obs), int(obs_chunk)):
        b = min(len(obs), a + int(obs_chunk))
        Gp, Gk, Gd, R = pair_blocks(obs[a:b], src, weights)
        alpha = 1.0 - R/(C0*DT)
        if np.any(alpha < 0.0) or np.any(alpha > 1.0):
            raise RuntimeError(
                "Step 4C3 V2 requires one-step MOT but alpha is outside [0,1]: "
                f"{float(alpha.min())} .. {float(alpha.max())}"
            )
        rmin = min(rmin, float(R.min()))
        rmax = max(rmax, float(R.max()))
        amin = min(amin, float(alpha.min()))
        amax = max(amax, float(alpha.max()))

        A = alpha[:, :, None, None]
        A2 = (alpha*alpha)[:, :, None, None]
        hp = Gp
        hk = Gp*(DT*(A - 0.5*A2)) + Gk*(1.0 - A) - Gd/DT
        an = Gp*(0.5*DT*A2) + Gk*A + Gd/DT

        sl = slice(2*a, 2*b)
        Hp[sl] = project_pair_tensor(ot[a:b], hp, st)
        Hk[sl] = project_pair_tensor(ot[a:b], hk, st)
        An[sl] = project_pair_tensor(ot[a:b], an, st)
        del Gp, Gk, Gd, R, alpha, A, A2, hp, hk, an
        gc.collect()

    delay = {
        "min_distance_m": rmin,
        "max_distance_m": rmax,
        "min_delay_s": rmin/C0,
        "max_delay_s": rmax/C0,
        "max_delay_over_dt": (rmax/C0)/DT,
        "alpha_min": amin,
        "alpha_max": amax,
    }
    return Hp, Hk, An, delay


def estimate_smax_real(A, niter=8):
    rng = np.random.default_rng(1234)
    x = rng.normal(size=A.shape[1])
    x /= max(float(np.linalg.norm(x)), TINY)
    for _ in range(int(niter)):
        y = A @ x
        z = A.T @ y
        nz = float(np.linalg.norm(z))
        if nz <= TINY:
            return 1.0
        x = z/nz
    return float(np.linalg.norm(A @ x))


def prepare_fixed_ridge_inplace(A_now, point_weights):
    rs = np.repeat(
        np.sqrt(
            np.asarray(point_weights, dtype=np.float64)
            / max(float(np.mean(point_weights)), TINY)
        ),
        2,
    )
    A_now *= rs[:, None]
    cs = np.linalg.norm(A_now, axis=0)
    cs = np.where(cs > TINY, cs, 1.0)
    A_now /= cs[None, :]
    As = A_now
    smax = max(estimate_smax_real(As, 8), TINY)
    alpha = LAMBDA_REL*smax
    H = As.T @ As
    H.flat[::H.shape[0]+1] += alpha*alpha
    cf = cho_factor(H, lower=True, overwrite_a=True, check_finite=False)
    return {"rs": rs, "cs": cs, "As": As, "cf": cf,
            "smax": float(smax), "ridge_alpha": float(alpha)}


def solve_fixed_ridge(prepared, rhs):
    bw = np.asarray(rhs, dtype=np.float64)*prepared["rs"]
    y = cho_solve(
        prepared["cf"],
        prepared["As"].T @ bw,
        check_finite=False,
    )
    return y/prepared["cs"]


def operator_now_coeff(prepared, q):
    # As = diag(rs) A_now diag(1/cs)
    return (prepared["As"] @ (np.asarray(q)*prepared["cs"])) / prepared["rs"]


def tangent_coeffs(field, t1, t2):
    field = np.asarray(field, dtype=np.float64)
    out = np.empty(2*len(field), dtype=np.float64)
    out[0::2] = np.sum(field*t1, axis=1)
    out[1::2] = np.sum(field*t2, axis=1)
    return out


def coeff_to_vectors(g, q):
    q = np.asarray(q, dtype=np.float64).reshape(-1, 2)
    return q[:, 0, None]*g.source_tangent1 + q[:, 1, None]*g.source_tangent2


def weighted_coeff_rms(g, q):
    q = np.asarray(q).reshape(-1, 2)
    w = np.asarray(g.source_weights, dtype=np.float64)
    mag2 = np.sum(np.abs(q)**2, axis=1)
    return float(np.sqrt(np.sum(w*mag2)/max(float(np.sum(w)), TINY)))


def state_norm(g, q, p):
    return math.sqrt(
        weighted_coeff_rms(g, q)**2
        + weighted_coeff_rms(g, np.asarray(p)/DT)**2
    )


def feed_shape(points):
    return FEED_FIELD_AMPLITUDE_V_PER_M * mb1.inc_feed_localized(
        np.asarray(points, dtype=float), 2.0e-3, 1.0
    )


def feed_at_time(points, t_s):
    ramp, _ = smoothstep_ramp_and_derivative(t_s)
    return ramp*math.cos(2.0*np.pi*F_DRIVE*t_s)*feed_shape(points)


def evaluate_retarded_mot_field(
    points,
    g,
    p_prev_coeff,
    q_prev_coeff,
    q_now_coeff,
    *,
    obs_chunk=128,
    src_chunk=128,
):
    """Independent pairwise field evaluator for the C1-V4 temporal basis."""
    points = np.asarray(points, dtype=np.float64)
    SP = np.asarray(g.ant_src_points, dtype=np.float64)
    SW = np.asarray(g.source_weights, dtype=np.float64)
    Pprev = coeff_to_vectors(g, p_prev_coeff)
    Kprev = coeff_to_vectors(g, q_prev_coeff)
    Know = coeff_to_vectors(g, q_now_coeff)
    Kdot = (Know - Kprev)/DT

    out = np.zeros((len(points), 3), dtype=np.float64)
    ce = 1.0/(4.0*np.pi*EPS0)
    max_delay_over_dt = 0.0

    for oa in range(0, len(points), int(obs_chunk)):
        ob = min(len(points), oa + int(obs_chunk))
        X = points[oa:ob]
        Eb = np.zeros((len(X), 3), dtype=np.float64)
        for sa in range(0, len(SP), int(src_chunk)):
            sb = min(len(SP), sa + int(src_chunk))
            r = X[:, None, :] - SP[None, sa:sb, :]
            R = np.linalg.norm(r, axis=2)
            if np.any(R <= 0.0):
                raise RuntimeError("zero source-holdout distance")
            alpha = 1.0 - R/(C0*DT)
            max_delay_over_dt = max(
                max_delay_over_dt,
                float(np.max((R/C0)/DT)),
            )
            if np.any(alpha < 0.0) or np.any(alpha > 1.0):
                raise RuntimeError(
                    "Holdout one-step MOT invalid: delay exceeds dt; "
                    f"alpha={float(alpha.min())}..{float(alpha.max())}"
                )

            A = alpha[:, :, None]
            A2 = (alpha*alpha)[:, :, None]
            Kp = Kprev[sa:sb][None, :, :]
            Kn = Know[sa:sb][None, :, :]
            Pp = Pprev[sa:sb][None, :, :]
            D = Kdot[sa:sb][None, :, :]

            Kr = A*Kn + (1.0-A)*Kp
            Pr = Pp + DT*(A - 0.5*A2)*Kp + 0.5*DT*A2*Kn

            rp = np.sum(r*Pr, axis=2)
            rk = np.sum(r*Kr, axis=2)
            rd = np.sum(r*D, axis=2)
            w = SW[None, sa:sb, None]

            ep = 3.0*r*rp[:, :, None]/R[:, :, None]**5 - Pr/R[:, :, None]**3
            ek = (3.0*r*rk[:, :, None]/R[:, :, None]**4 - Kr/R[:, :, None]**2)/C0
            ed = (r*rd[:, :, None]/R[:, :, None]**3 - D/R[:, :, None])/(C0*C0)
            Eb += np.sum(ce*w*(ep + ek + ed), axis=1)
        out[oa:ob] = Eb

    return out, max_delay_over_dt


def tangential_sq(field, normals):
    field = np.asarray(field, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    et = field - normals*np.sum(field*normals, axis=1)[:, None]
    return np.sum(et*et, axis=1)


def load_forcing(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["metadata_json"].item()))
    return z, meta


def validate_forcing_metadata(meta):
    checks = {
        "dt_s": DT,
        "depth_frac": DEPTH_FRAC,
        "collocation_factor": COLLOCATION_FACTOR,
        "lambda_rel": LAMBDA_REL,
        "operator_softening_distance_m": 0.0,
    }
    for key, expected in checks.items():
        if key not in meta:
            raise RuntimeError(f"Forcing metadata missing required key: {key}")
        value = float(meta[key])
        if abs(value-expected) > max(1.0e-15, 1.0e-12*abs(expected)):
            raise RuntimeError(
                f"Forcing metadata mismatch: {key}={value!r}, expected {expected!r}"
            )
    if meta.get("plasma_field_dynamics_projection") != "none_full_time_domain_samples_saved":
        raise RuntimeError("Forcing file is not the full time-domain plasma dataset")


def load_old_baseline(path):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        old = json.loads(p.read_text(encoding="utf-8"))
        worst = old.get("independent_holdout", {}).get("worst_gate")
        return {
            "source": str(p),
            "accepted": old.get("accepted"),
            "time_aggregated_collocation_rel_l2": old.get("time_aggregated_collocation_rel_l2"),
            "worst_holdout_gate": None if worst is None else worst.get("value"),
            "zero_input_tail_growth": old.get("zero_input_tail", {}).get("median_growth_factor_last_20"),
        }
    except Exception as exc:
        return {"source": str(p), "load_error": repr(exc)}


def run_replay(args):
    forcing_path = Path(args.forcing)
    z, meta = load_forcing(forcing_path)
    validate_forcing_metadata(meta)

    out = Path("article_cases_output")/args.out
    out.mkdir(parents=True, exist_ok=True)

    keys = (
        "mode", "total_length_m", "sources_per_arm", "n_phi", "n_cyl",
        "n_hemi", "cycles", "analysis_start_cycles", "diag_stride",
        "holdout_points_target",
    )
    settings = {k: meta[k] for k in keys}

    print("[Step 4C3 V2] rebuilding frozen forcing geometry ...", flush=True)
    g = build_geometry(settings)
    fp = geometry_fingerprint(g)
    if fp != meta["geometry_fingerprint"]:
        raise RuntimeError(
            "Forcing/replay geometry fingerprint mismatch: "
            f"record={meta['geometry_fingerprint']} replay={fp}"
        )

    plasma_coll = z["plasma_collocation_E_V_per_m"]
    times = np.asarray(z["time_s"], dtype=np.float64)
    diag_steps = np.asarray(z["diag_steps"], dtype=np.int64)
    plasma_holdout = z["plasma_holdout_E_V_per_m"]
    Hpoints = np.asarray(z["holdout_points_m"], dtype=np.float64)
    Hnormals = np.asarray(z["holdout_normals"], dtype=np.float64)
    Hmask = np.asarray(z["holdout_mask_matrix"], dtype=np.uint8).astype(bool)
    region_names = list(meta["holdout_region_names"])

    if len(plasma_coll) != len(times):
        raise RuntimeError("forcing time and collocation arrays have different lengths")
    if plasma_coll.shape[1] != g.num_collocation_points:
        raise RuntimeError("forcing collocation count does not match geometry")
    if Hmask.shape != (len(region_names), len(Hpoints)):
        raise RuntimeError("holdout mask shape mismatch")
    if len(plasma_holdout) != len(diag_steps):
        raise RuntimeError("holdout forcing count does not match diag steps")

    diag_lookup = {int(s): i for i, s in enumerate(diag_steps)}

    print("[Step 4C3 V2] assembling C1-V4 retarded-MOT operators ...", flush=True)
    obs_chunk = 8 if settings["mode"] == "formal" else 16
    Hp, Hk, An, delay = assemble_retarded_mot_operators(g, obs_chunk)
    if delay["max_delay_over_dt"] >= 1.0:
        raise RuntimeError("collocation one-step MOT invalid: max delay >= dt")
    prep = prepare_fixed_ridge_inplace(An, g.solve_weights)
    del An
    gc.collect()

    # Explicit holdout delay precheck, independent of actual field values.
    max_holdout_delay_over_dt = 0.0
    SP = np.asarray(g.ant_src_points, dtype=np.float64)
    for a in range(0, len(Hpoints), 128):
        R = np.linalg.norm(Hpoints[a:a+128, None, :] - SP[None, :, :], axis=2)
        max_holdout_delay_over_dt = max(
            max_holdout_delay_over_dt,
            float(np.max((R/C0)/DT)),
        )
    if max_holdout_delay_over_dt >= 1.0:
        raise RuntimeError("holdout one-step MOT invalid: max delay >= dt")

    ndof = 2*g.num_sources
    q_prev = np.zeros(ndof, dtype=np.float64)
    p_prev = np.zeros(ndof, dtype=np.float64)
    analysis_start_s = float(settings["analysis_start_cycles"])/F_DRIVE

    coll_residual = np.full(len(times), np.nan, dtype=np.float64)
    incident_rms = np.full(len(times), np.nan, dtype=np.float64)
    current_rms = np.full(len(times), np.nan, dtype=np.float64)
    state_norms = np.full(len(times), np.nan, dtype=np.float64)

    coll_res_sum = 0.0
    coll_inc_sum = 0.0

    hold_res_sum = {r: 0.0 for r in region_names}
    hold_inc_sum = {r: 0.0 for r in region_names}
    hold_count = {r: 0 for r in region_names}

    harmonic_freqs = (3.0e6, 6.0e6, 9.0e6)
    qhat_sum = {f: np.zeros(ndof, dtype=np.complex128) for f in harmonic_freqs}
    harmonic_count = 0

    finite = True
    completed = 0

    print(
        f"[Step 4C3 V2] replaying {len(times)} broadband nodes; "
        f"mode={settings['mode']}, sources/arm={settings['sources_per_arm']}, "
        "no frequency projection",
        flush=True,
    )

    for step, t_s in enumerate(times):
        plasma = np.asarray(plasma_coll[step], dtype=np.float64)
        incident = plasma + feed_at_time(g.solve_obs_points, float(t_s))
        e = tangent_coeffs(incident, g.solve_tangent1, g.solve_tangent2)

        history = Hp @ p_prev + Hk @ q_prev
        q = solve_fixed_ridge(prep, -(e + history))
        p = p_prev + 0.5*DT*(q_prev + q)

        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            finite = False
            break

        residual_coeff = e + history + operator_now_coeff(prep, q)
        inc_norm2 = float(np.dot(e, e))
        res_norm2 = float(np.dot(residual_coeff, residual_coeff))
        coll_residual[step] = math.sqrt(res_norm2/max(inc_norm2, TINY))
        incident_rms[step] = math.sqrt(inc_norm2/max(len(e)//2, 1))
        current_rms[step] = weighted_coeff_rms(g, q)
        state_norms[step] = state_norm(g, q, p)

        if t_s >= analysis_start_s:
            coll_res_sum += res_norm2
            coll_inc_sum += inc_norm2
            for f in harmonic_freqs:
                qhat_sum[f] += 2.0*q*np.exp(-1j*2.0*np.pi*f*t_s)
            harmonic_count += 1

        if step in diag_lookup and t_s >= analysis_start_s:
            hi = diag_lookup[step]
            incident_H = (
                np.asarray(plasma_holdout[hi], dtype=np.float64)
                + feed_at_time(Hpoints, float(t_s))
            )
            self_H, hold_delay = evaluate_retarded_mot_field(
                Hpoints, g, p_prev, q_prev, q
            )
            max_holdout_delay_over_dt = max(max_holdout_delay_over_dt, hold_delay)
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

        q_prev, p_prev = q, p
        completed = step + 1

        if step % max(1, int(settings["diag_stride"])) == 0 or step == len(times)-1:
            print(
                f"[Step 4C3 V2] step {step+1}/{len(times)}, "
                f"colloc={coll_residual[step]:.6g}, "
                f"K_rms={current_rms[step]:.6e}, state={state_norms[step]:.6e}",
                flush=True,
            )

    aggregate_collocation = (
        math.sqrt(coll_res_sum/coll_inc_sum)
        if coll_inc_sum > 0.0 else float("inf")
    )

    global_count = hold_count.get("global", 0)
    global_inc_rms = (
        math.sqrt(hold_inc_sum["global"]/global_count)
        if global_count > 0 else 0.0
    )

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
        if region == "global" or incident_ratio >= WEAK_REGION_THRESHOLD:
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

    worst_holdout = max(holdout_rows, key=lambda r: r["gate_value"]) if holdout_rows else None
    worst_holdout_value = float(worst_holdout["gate_value"]) if worst_holdout else float("inf")

    # 200-step zero-input tail from the ACTUAL broadband-forced final state.
    tail_norms = [state_norm(g, q_prev, p_prev)]
    tail_ratios = []
    for _ in range(TAIL_STEPS):
        history = Hp @ p_prev + Hk @ q_prev
        q = solve_fixed_ridge(prep, -history)
        p = p_prev + 0.5*DT*(q_prev + q)
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            finite = False
            break
        nn = state_norm(g, q, p)
        prev = tail_norms[-1]
        tail_norms.append(nn)
        if prev > TINY:
            tail_ratios.append(nn/prev)
        q_prev, p_prev = q, p

    tail_growth = (
        float(np.median(tail_ratios[-min(TAIL_WINDOW, len(tail_ratios)):]))
        if tail_ratios else float("inf")
    )

    harmonic_response = {}
    if harmonic_count > 0:
        for f in harmonic_freqs:
            qhat = qhat_sum[f]/harmonic_count
            harmonic_response[f"{f/1e6:.0f}MHz"] = {
                "weighted_current_phasor_rms_A_per_m": weighted_coeff_rms(g, qhat)
            }

    accepted = bool(
        completed == len(times)
        and finite
        and aggregate_collocation < BOUNDARY_GATE
        and worst_holdout_value < BOUNDARY_GATE
        and tail_growth <= TAIL_GROWTH_GATE
        and fp == meta["geometry_fingerprint"]
        and delay["max_delay_over_dt"] < 1.0
        and max_holdout_delay_over_dt < 1.0
    )

    baseline = load_old_baseline(args.baseline_report)
    comparison = None
    if baseline and "load_error" not in baseline:
        comparison = {
            "old_time_local": baseline,
            "new_retarded_mot": {
                "time_aggregated_collocation_rel_l2": aggregate_collocation,
                "worst_holdout_gate": worst_holdout_value,
                "zero_input_tail_growth": tail_growth,
            },
        }
        oc = baseline.get("time_aggregated_collocation_rel_l2")
        oh = baseline.get("worst_holdout_gate")
        ot = baseline.get("zero_input_tail_growth")
        comparison["improvement_factors_old_over_new"] = {
            "collocation": None if oc is None else float(oc/max(aggregate_collocation, TINY)),
            "holdout": None if oh is None else float(oh/max(worst_holdout_value, TINY)),
            "tail_growth_excess_over_one": None if ot is None else float(
                max(float(ot)-1.0, 0.0)/max(tail_growth-1.0, TINY)
            ) if tail_growth > 1.0 else float("inf"),
        }

    report = {
        "step": "4C3-V2",
        "stage": "retarded_mot_replay",
        "accepted": accepted,
        "mode": settings["mode"],
        "forcing_file": str(forcing_path),
        "forcing_reused_without_PIC_rerun": True,
        "single_frequency_projection_used": False,
        "time_local_BE_used": False,
        "retarded_MOT_used": True,
        "implicit_temporal_basis": (
            "K_r=alpha*K^n+(1-alpha)*K^(n-1); "
            "Kdot_r=(K^n-K^(n-1))/dt; "
            "P_r=P^(n-1)+dt(alpha-alpha^2/2)K^(n-1)+dt(alpha^2/2)K^n; "
            "P^n=P^(n-1)+dt/2(K^(n-1)+K^n)"
        ),
        "geometry_fingerprint_match": bool(fp == meta["geometry_fingerprint"]),
        "forcing_geometry_fingerprint": meta["geometry_fingerprint"],
        "replay_geometry_fingerprint": fp,
        "operator_softening_distance_m": 0.0,
        "one_step_delay": {
            "collocation": delay,
            "holdout_max_delay_over_dt": float(max_holdout_delay_over_dt),
        },
        "steps_requested": int(len(times)),
        "steps_completed": int(completed),
        "all_replay_states_finite": bool(finite),
        "analysis_start_s": float(analysis_start_s),
        "analysis_start_cycles": float(settings["analysis_start_cycles"]),
        "time_aggregated_collocation_rel_l2": float(aggregate_collocation),
        "collocation_gate": BOUNDARY_GATE,
        "independent_holdout": {
            "type": "recorded stratified independent barycentric surface holdout",
            "num_unique_points": int(len(Hpoints)),
            "weak_region_threshold": WEAK_REGION_THRESHOLD,
            "regions": holdout_metrics,
            "worst_gate": worst_holdout,
            "gate": BOUNDARY_GATE,
        },
        "zero_input_tail": {
            "steps": TAIL_STEPS,
            "tail_window": TAIL_WINDOW,
            "median_growth_factor_last_50": float(tail_growth),
            "growth_gate": TAIL_GROWTH_GATE,
            "initial_state_norm": float(tail_norms[0]),
            "final_state_norm": float(tail_norms[-1]),
            "max_state_norm": float(np.max(tail_norms)),
        },
        "diagnostic_harmonic_response": harmonic_response,
        "old_vs_new": comparison,
        "response_values_are_manuscript_final": False,
        "decision": (
            "STEP4C3_V2_PASS_proceed_to_STEP4D_two_way_broadband_retarded_MOT"
            if accepted else
            "STEP4C3_V2_FAIL_do_not_close_two_way_loop"
        ),
        "scope_note": (
            "C3 V2 validates one-way broadband plasma forcing replay only. "
            "The MOT response is not fed back into PIC, so no self-consistent "
            "nonlinear amplification factor is claimed."
        ),
    }

    (out/"step4c3_v2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        out/"step4c3_v2_diagnostics.npz",
        time_s=times[:completed],
        collocation_rel_l2=coll_residual[:completed],
        incident_tangent_rms=incident_rms[:completed],
        current_weighted_rms_A_per_m=current_rms[:completed],
        state_norm=state_norms[:completed],
        tail_state_norm=np.asarray(tail_norms, dtype=np.float64),
        tail_growth_ratio=np.asarray(tail_ratios, dtype=np.float64),
    )

    lines = [
        "Step 4C3 V2 retarded-MOT broadband replay",
        f"accepted: {accepted}",
        f"mode: {settings['mode']}",
        f"steps: {completed}/{len(times)}",
        f"collocation aggregate: {aggregate_collocation:.12g}",
        f"worst holdout gate: {worst_holdout_value:.12g}",
        f"tail growth (last {TAIL_WINDOW}/{TAIL_STEPS}): {tail_growth:.12g}",
        f"collocation max delay/dt: {delay['max_delay_over_dt']:.12g}",
        f"holdout max delay/dt: {max_holdout_delay_over_dt:.12g}",
    ]
    if comparison:
        old = comparison["old_time_local"]
        lines += [
            "old time-local baseline:",
            f"  collocation: {old.get('time_aggregated_collocation_rel_l2')}",
            f"  holdout: {old.get('worst_holdout_gate')}",
            f"  tail: {old.get('zero_input_tail_growth')}",
        ]
    lines.append(f"decision: {report['decision']}")
    (out/"step4c3_v2_summary.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))
    return accepted


def self_test():
    # Tiny geometry: test algebraic equivalence between assembled MOT operator
    # and the independent pairwise retarded-field evaluator.
    settings = {
        "mode": "selftest",
        "total_length_m": 16.0e-3,
        "sources_per_arm": 12,
        "n_phi": 14,
        "n_cyl": 14,
        "n_hemi": 6,
    }
    g = build_geometry(settings)
    Hp, Hk, An, delay = assemble_retarded_mot_operators(g, obs_chunk=8)

    rng = np.random.default_rng(20261002)
    qprev = rng.normal(size=2*g.num_sources)
    qnow = rng.normal(size=2*g.num_sources)
    pprev = DT*rng.normal(size=2*g.num_sources)

    assembled = Hp@pprev + Hk@qprev + An@qnow
    E, hdelay = evaluate_retarded_mot_field(
        g.solve_obs_points, g, pprev, qprev, qnow,
        obs_chunk=16, src_chunk=32,
    )
    direct = tangent_coeffs(E, g.solve_tangent1, g.solve_tangent2)
    rel = float(np.linalg.norm(assembled-direct)/max(float(np.linalg.norm(direct)), TINY))

    # Also verify a short unforced recurrence remains finite with the exact
    # same solve/update path used by production replay.
    prep = prepare_fixed_ridge_inplace(An, g.solve_weights)
    q = qprev.copy()
    p = pprev.copy()
    norms = []
    finite = True
    for _ in range(40):
        qn = solve_fixed_ridge(prep, -(Hp@p + Hk@q))
        pn = p + 0.5*DT*(q + qn)
        nn = state_norm(g, qn, pn)
        if not np.isfinite(nn):
            finite = False
            break
        norms.append(nn)
        q, p = qn, pn

    passed = bool(
        rel < 1.0e-11
        and finite
        and delay["max_delay_over_dt"] < 1.0
        and hdelay < 1.0
    )
    report = {
        "self_test_pass": passed,
        "assembled_vs_independent_retarded_field_relative_error": rel,
        "collocation_max_delay_over_dt": delay["max_delay_over_dt"],
        "independent_evaluator_max_delay_over_dt": hdelay,
        "short_recurrence_finite": finite,
        "short_recurrence_final_state_norm": None if not norms else norms[-1],
    }
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("Step 4C3 V2 self-test failed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--forcing", default=None)
    p.add_argument("--baseline-report", default=None)
    p.add_argument("--out", default="step4c3_v2_formal")
    a = p.parse_args()
    if a.self_test:
        self_test()
        return
    if a.forcing is None:
        raise RuntimeError("--forcing is required for Step 4C3 V2 replay")
    ok = run_replay(a)
    if not ok:
        raise RuntimeError("Step 4C3 V2 replay failed frozen gates")


if __name__ == "__main__":
    main()
