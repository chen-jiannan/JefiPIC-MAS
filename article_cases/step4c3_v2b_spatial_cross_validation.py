#!/usr/bin/env python3
"""
Step 4C3 V2B — controlled spatial refinement / two-fold cross-validation.

Diagnostic purpose
------------------
Step 4C3 V2A showed that the ACTUAL recorded plasma+feed 3/6/9 MHz spatial
phasors are fitted well at the production collocation points but generalize
poorly to independent holdout points. The failure is not limited to a single
feed-transition point; several holdout regions are affected.

V2B asks one controlled question:

    With the source basis FIXED at 900 sources/arm, does increasing independent
    surface collocation COVERAGE improve unseen-surface generalization?

No PIC rerun and no production retuning are performed.

Method
------
1. Reproduce the V2A baseline harmonic audit exactly.
2. SAMPLING-CONSISTENCY CONTROL: refit collocation phasors using exactly the
   same 22 diagnostic times used by the holdout phasors. This prevents a
   dense-time vs sparse-time least-squares estimator mismatch from being
   misdiagnosed as spatial under-resolution.
3. Only if the synchronized baseline still fails, split ALL 2020 unique
   independent holdout points into deterministic, spatially interleaved folds
   A/B.
4. A->B: add fold A as supplemental boundary equations, keep fold B unseen.
5. B->A: swap roles.
6. Keep unchanged:
      - 900 sources/arm;
      - source geometry and source weights;
      - zero softening;
      - lambda_rel = 1e-4;
      - causal retarded harmonic operator;
      - 5% boundary gate.
7. Supplemental-point weights are inherited from the nearest production
   collocation Voronoi cell and divided by the number of supplemental points
   mapping to that cell. This prevents a deliberately stratified holdout cloud
   from being overweighted simply because a region contains more samples.

Interpretation
--------------
SAMPLING_ESTIMATOR_CONFOUND_IDENTIFIED:
  Synchronized collocation/holdout phasors pass without spatial refinement.
  Do NOT density-retune from the original V2A harmonic audit.

STRONG_COVERAGE_SUPPORT_AFTER_SAMPLING_CONTROL:
  Both cross-validation directions pass the 5% unseen validation gate at all
  assessable harmonics, while the original production collocation also remains
  <5%. This supports collocation-coverage refinement before source-density
  changes.

COVERAGE_HELPS_BUT_NOT_SUFFICIENT:
  At least one unseen validation error improves but one or more frozen gates
  remain >5%. Coverage matters, but source-basis density/placement may still be
  insufficient. Do not move to two-way coupling.

NO_CLEAR_COVERAGE_BENEFIT:
  Supplemental coverage does not improve unseen validation. The next controlled
  diagnostic should change the source basis while holding observation data
  fixed; do not retune lambda/dt/filtering.

This is a diagnostic. Cross-validation training points are NOT a production
solver and are never counted as independent validation in the same fold.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree

import step4c3_v2_retarded_mot_broadband_replay as c3
import step4c2_v4_retarded_frequency_regression as c2
import step4c3_v2a_feed_transition_residual_diagnostic as v2a

FREQS = (3.0e6, 6.0e6, 9.0e6)
BOUNDARY_GATE = 0.05
WEAK_REGION_THRESHOLD = 0.20
REPRO_TOL = 2.0e-8
FIT_CONDITION_LIMIT = 100.0
MIN_HARMONIC_STRENGTH_RATIO = 0.01
N_AXIAL_BINS = 12
N_PHI_BINS = 12
SUPPLEMENTAL_WEIGHT_SCALE = 1.0
GRAM_CHUNK_ROWS = 768
TINY = 1.0e-300


def tangent_basis_from_normals(normals):
    """Deterministic orthonormal tangent basis for arbitrary holdout normals."""
    n = np.asarray(normals, dtype=np.float64)
    nn = np.linalg.norm(n, axis=1)
    if np.any(nn <= 0.0):
        raise RuntimeError("zero holdout normal")
    n = n / nn[:, None]

    # Choose the Cartesian axis least aligned with each normal.
    axis = np.argmin(np.abs(n), axis=1)
    ref = np.zeros_like(n)
    ref[np.arange(len(n)), axis] = 1.0
    t1 = np.cross(n, ref)
    t1n = np.linalg.norm(t1, axis=1)
    if np.any(t1n <= 0.0):
        raise RuntimeError("failed to build holdout tangent basis")
    t1 /= t1n[:, None]
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2, axis=1)[:, None]
    return t1, t2


def field_to_coeff(E, t1, t2):
    E = np.asarray(E, dtype=np.complex128)
    out = np.empty(2 * len(E), dtype=np.complex128)
    out[0::2] = np.sum(E * t1, axis=1)
    out[1::2] = np.sum(E * t2, axis=1)
    return out


def coeff_rows_for_point_mask(mask):
    return np.flatnonzero(np.repeat(np.asarray(mask, dtype=bool), 2))


def assemble_retarded_harmonic_points(points, t1, t2, g, omega, obs_chunk=32):
    """Retarded continuous-harmonic operator at arbitrary observation points."""
    points = np.asarray(points, dtype=np.float64)
    ot = np.stack([np.asarray(t1, float), np.asarray(t2, float)], axis=1)
    src = np.asarray(g.ant_src_points, dtype=np.float64)
    sw = np.asarray(g.source_weights, dtype=np.float64)
    st = np.stack([g.source_tangent1, g.source_tangent2], axis=1)
    A = np.empty((2 * len(points), 2 * len(src)), dtype=np.complex128)
    max_phase = 0.0
    rmin = math.inf
    rmax = 0.0
    for a in range(0, len(points), int(obs_chunk)):
        b = min(len(points), a + int(obs_chunk))
        Gp, Gk, Gd, R = c2.pair_blocks(points[a:b], src, sw)
        phase = np.exp(-1j * omega * R / c2.C0)[:, :, None, None]
        T = phase * (Gp / (1j * omega) + Gk + 1j * omega * Gd)
        A[2*a:2*b] = c2.project(ot[a:b], T, st)
        max_phase = max(max_phase, float(np.max(omega * R / c2.C0)))
        rmin = min(rmin, float(np.min(R)))
        rmax = max(rmax, float(np.max(R)))
        del Gp, Gk, Gd, R, phase, T
        gc.collect()
    return A, {
        "min_distance_m": rmin,
        "max_distance_m": rmax,
        "max_pair_phase_delay_deg": max_phase * 180.0 / math.pi,
    }


def spatial_interleaved_folds(points):
    """
    Deterministic checkerboard split in axial rank x azimuth x arm.

    The split is intentionally interleaved rather than a left/right spatial
    partition: each training fold should cover the same geometric extent as its
    unseen validation fold.
    """
    P = np.asarray(points, dtype=np.float64)
    z = P[:, 2]
    arm = (z >= 0.0).astype(np.int64)
    phi = (np.arctan2(P[:, 1], P[:, 0]) + 2*np.pi) % (2*np.pi)
    phi_bin = np.minimum(
        (phi / (2*np.pi) * N_PHI_BINS).astype(np.int64),
        N_PHI_BINS - 1,
    )

    axial_bin = np.empty(len(P), dtype=np.int64)
    for a in (0, 1):
        idx = np.flatnonzero(arm == a)
        if len(idx) == 0:
            continue
        # Rank by distance from the feed plane within each arm; rank bins avoid
        # pathological duplicated-z quantile edges on cylindrical rings.
        order = idx[np.argsort(np.abs(z[idx]), kind="mergesort")]
        bins = np.minimum(
            np.arange(len(order), dtype=np.int64) * N_AXIAL_BINS // len(order),
            N_AXIAL_BINS - 1,
        )
        axial_bin[order] = bins

    fold = (arm + axial_bin + phi_bin) & 1
    return fold, {
        "method": "arm + axial-rank-bin + azimuth-bin checkerboard parity",
        "n_axial_bins": N_AXIAL_BINS,
        "n_phi_bins": N_PHI_BINS,
    }


def split_diagnostics(points, mask_matrix, region_names, fold, collocation_points):
    P = np.asarray(points, float)
    fold = np.asarray(fold, int)
    rows = {
        "fold_sizes": {
            "A": int(np.sum(fold == 0)),
            "B": int(np.sum(fold == 1)),
        },
        "region_counts": {},
    }
    for ir, r in enumerate(region_names):
        m = np.asarray(mask_matrix[ir], bool)
        rows["region_counts"][r] = {
            "A": int(np.sum(m & (fold == 0))),
            "B": int(np.sum(m & (fold == 1))),
            "total": int(np.sum(m)),
        }

    base_tree = cKDTree(np.asarray(collocation_points, float))
    dbase, _ = base_tree.query(P, k=1)
    rows["nearest_original_collocation_distance_m"] = {
        "median": float(np.median(dbase)),
        "p95": float(np.percentile(dbase, 95)),
        "max": float(np.max(dbase)),
    }
    rows["directions"] = {}
    for tr in (0, 1):
        va = 1 - tr
        train = fold == tr
        valid = fold == va
        aug_tree = cKDTree(np.vstack([collocation_points, P[train]]))
        daug, _ = aug_tree.query(P[valid], k=1)
        rows["directions"][f"{'A' if tr == 0 else 'B'}_to_{'B' if va == 1 else 'A'}"] = {
            "validation_point_count": int(np.sum(valid)),
            "baseline_nearest_distance_m": {
                "median": float(np.median(dbase[valid])),
                "p95": float(np.percentile(dbase[valid], 95)),
                "max": float(np.max(dbase[valid])),
            },
            "augmented_nearest_distance_m": {
                "median": float(np.median(daug)),
                "p95": float(np.percentile(daug, 95)),
                "max": float(np.max(daug)),
            },
        }
    return rows


def inherited_supplemental_weights(train_points, original_points, original_weights):
    """
    Assign each supplemental point the weight of its nearest production
    collocation Voronoi cell, divided among all supplemental points that map to
    that same cell.

    This avoids overweighting deliberately oversampled diagnostic regions.
    """
    tree = cKDTree(np.asarray(original_points, float))
    dist, nn = tree.query(np.asarray(train_points, float), k=1)
    counts = np.bincount(nn, minlength=len(original_weights))
    w = np.asarray(original_weights, float)[nn] / counts[nn]
    w *= SUPPLEMENTAL_WEIGHT_SCALE
    return w, nn, dist, {
        "unique_original_cells_covered": int(np.count_nonzero(counts)),
        "mean_nearest_original_collocation_distance_m": float(np.mean(dist)),
        "max_nearest_original_collocation_distance_m": float(np.max(dist)),
        "supplemental_total_weight": float(np.sum(w)),
        "original_total_weight": float(np.sum(original_weights)),
        "supplemental_over_original_weight": float(
            np.sum(w) / max(float(np.sum(original_weights)), TINY)
        ),
        "weight_rule": (
            "nearest production collocation weight divided by supplemental "
            "multiplicity in that Voronoi cell"
        ),
        "supplemental_weight_scale": SUPPLEMENTAL_WEIGHT_SCALE,
    }


def _column_norms_blocks(blocks, row_scales):
    n = blocks[0].shape[1]
    s2 = np.zeros(n, dtype=np.float64)
    for A, rs in zip(blocks, row_scales):
        for a in range(0, A.shape[0], GRAM_CHUNK_ROWS):
            b = min(A.shape[0], a + GRAM_CHUNK_ROWS)
            B = A[a:b]
            r = rs[a:b]
            s2 += np.sum(np.abs(B)**2 * (r*r)[:, None], axis=0)
    return np.sqrt(s2)


def _smax_blocks(blocks, row_scales, col_scales, niter=8):
    rng = np.random.default_rng(1234)
    n = blocks[0].shape[1]
    x = rng.normal(size=n) + 1j*rng.normal(size=n)
    x /= max(float(np.linalg.norm(x)), TINY)
    for _ in range(int(niter)):
        xc = x / col_scales
        ys = [rs * (A @ xc) for A, rs in zip(blocks, row_scales)]
        z = np.zeros(n, dtype=np.complex128)
        for A, rs, y in zip(blocks, row_scales, ys):
            z += A.conj().T @ (rs * y)
        z /= col_scales
        nz = float(np.linalg.norm(z))
        if nz <= TINY:
            return 1.0
        x = z / nz
    xc = x / col_scales
    n2 = 0.0
    for A, rs in zip(blocks, row_scales):
        y = rs * (A @ xc)
        n2 += float(np.vdot(y, y).real)
    return math.sqrt(max(n2, 0.0))


def prepare_complex_blocks(blocks, point_weights):
    """
    Memory-conscious exact analogue of c2.prepare_complex for multiple row
    blocks. No augmented dense vstack is materialized.
    """
    if len(blocks) != len(point_weights):
        raise ValueError("block/weight length mismatch")
    n = blocks[0].shape[1]
    if any(A.shape[1] != n for A in blocks):
        raise ValueError("all blocks must have same column count")

    all_w = np.concatenate([np.asarray(w, float) for w in point_weights])
    mean_w = max(float(np.mean(all_w)), TINY)
    row_scales = [
        np.repeat(np.sqrt(np.asarray(w, float) / mean_w), 2)
        for w in point_weights
    ]
    col_scales = _column_norms_blocks(blocks, row_scales)
    col_scales = np.where(col_scales > TINY, col_scales, 1.0)
    smax = max(_smax_blocks(blocks, row_scales, col_scales, niter=8), TINY)
    alpha = c2.LAMBDA_REL * smax

    H = np.zeros((n, n), dtype=np.complex128)
    for A, rs in zip(blocks, row_scales):
        for a in range(0, A.shape[0], GRAM_CHUNK_ROWS):
            b = min(A.shape[0], a + GRAM_CHUNK_ROWS)
            B = (A[a:b] * rs[a:b, None]) / col_scales[None, :]
            H += B.conj().T @ B
            del B
    H.flat[::n+1] += alpha * alpha
    cf = cho_factor(H, lower=True, overwrite_a=True, check_finite=False)
    return {
        "row_scales": row_scales,
        "col_scales": col_scales,
        "cf": cf,
        "smax": float(smax),
        "ridge_alpha": float(alpha),
    }


def solve_complex_blocks(prep, blocks, rhs_blocks):
    n = blocks[0].shape[1]
    z = np.zeros(n, dtype=np.complex128)
    for A, rs, b in zip(blocks, prep["row_scales"], rhs_blocks):
        z += A.conj().T @ ((rs*rs) * np.asarray(b, np.complex128))
    z /= prep["col_scales"]
    y = cho_solve(prep["cf"], z, check_finite=False)
    return y / prep["col_scales"]


def weighted_current_rms(g, q):
    q2 = np.asarray(q, np.complex128).reshape(-1, 2)
    sw = np.asarray(g.source_weights, float)
    return float(np.sqrt(
        np.sum(sw * np.sum(np.abs(q2)**2, axis=1))
        / max(float(np.sum(sw)), TINY)
    ))


def tangential_rel(total, incident, normals, mask):
    mask = np.asarray(mask, bool)
    if not np.any(mask):
        return float("nan")
    rt = v2a.tangential_vectors_real(np.real(total[mask]), normals[mask]) + 1j * v2a.tangential_vectors_real(np.imag(total[mask]), normals[mask])
    it = v2a.tangential_vectors_real(np.real(incident[mask]), normals[mask]) + 1j * v2a.tangential_vectors_real(np.imag(incident[mask]), normals[mask])
    return float(np.linalg.norm(rt.reshape(-1)) / max(float(np.linalg.norm(it.reshape(-1))), TINY))


def masked_region_metrics(total, incident, normals, mask_matrix, region_names, allowed_mask):
    """C3 region rule evaluated only on points that remain independent."""
    Et_total = c2.tangential_vectors(total, normals)
    Et_inc = c2.tangential_vectors(incident, normals)
    allowed = np.asarray(allowed_mask, bool)
    if "global" not in region_names:
        raise RuntimeError("missing global region")
    gi = region_names.index("global")
    gm = np.asarray(mask_matrix[gi], bool) & allowed
    if not np.any(gm):
        raise RuntimeError("validation global mask is empty")
    global_inc = float(np.sqrt(np.mean(np.sum(np.abs(Et_inc[gm])**2, axis=1))))
    rows = {}
    for ir, region in enumerate(region_names):
        m = np.asarray(mask_matrix[ir], bool) & allowed
        if not np.any(m):
            continue
        rr = float(np.sqrt(np.mean(np.sum(np.abs(Et_total[m])**2, axis=1))))
        ii = float(np.sqrt(np.mean(np.sum(np.abs(Et_inc[m])**2, axis=1))))
        ratio = ii / max(global_inc, TINY)
        local_rel = rr / max(ii, TINY)
        global_norm = rr / max(global_inc, TINY)
        if region == "global" or ratio >= WEAK_REGION_THRESHOLD:
            metric = "local_rel"
            value = local_rel
        else:
            metric = "global_norm"
            value = global_norm
        rows[region] = {
            "region": region,
            "point_count": int(np.sum(m)),
            "residual_rms": rr,
            "incident_rms": ii,
            "incident_ratio": ratio,
            "local_rel": local_rel,
            "global_norm": global_norm,
            "gate_metric": metric,
            "gate_value": value,
            "passed": bool(value < BOUNDARY_GATE),
        }
    return rows



def coeff_region_metrics(residual_coeff, incident_coeff, mask_matrix, region_names, allowed_mask=None):
    """C3 physical gate from any orthonormal two-component tangent basis."""
    r = np.asarray(residual_coeff, np.complex128).reshape(-1, 2)
    i = np.asarray(incident_coeff, np.complex128).reshape(-1, 2)
    if len(r) != len(i):
        raise ValueError("residual/incident point-count mismatch")
    if allowed_mask is None:
        allowed = np.ones(len(r), dtype=bool)
    else:
        allowed = np.asarray(allowed_mask, bool)
    gi = region_names.index("global")
    gm = np.asarray(mask_matrix[gi], bool) & allowed
    if not np.any(gm):
        raise RuntimeError("independent global mask is empty")
    inc2 = np.sum(np.abs(i)**2, axis=1)
    res2 = np.sum(np.abs(r)**2, axis=1)
    global_inc = float(np.sqrt(np.mean(inc2[gm])))
    rows = {}
    for ir, region in enumerate(region_names):
        m = np.asarray(mask_matrix[ir], bool) & allowed
        if not np.any(m):
            continue
        rr = float(np.sqrt(np.mean(res2[m])))
        ii = float(np.sqrt(np.mean(inc2[m])))
        ratio = ii / max(global_inc, TINY)
        local_rel = rr / max(ii, TINY)
        global_norm = rr / max(global_inc, TINY)
        if region == "global" or ratio >= WEAK_REGION_THRESHOLD:
            metric = "local_rel"
            value = local_rel
        else:
            metric = "global_norm"
            value = global_norm
        rows[region] = {
            "region": region,
            "point_count": int(np.sum(m)),
            "residual_rms": rr,
            "incident_rms": ii,
            "incident_ratio": ratio,
            "local_rel": local_rel,
            "global_norm": global_norm,
            "gate_metric": metric,
            "gate_value": value,
            "passed": bool(value < BOUNDARY_GATE),
        }
    return rows

def worst_region(rows):
    if not rows:
        return None
    return max(rows.values(), key=lambda x: x["gate_value"])


def load_v2a_report(path):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def run(args):
    forcing_path = Path(args.forcing)
    z, meta = c3.load_forcing(forcing_path)
    c3.validate_forcing_metadata(meta)
    ref_report = load_v2a_report(args.v2a_report)

    keys = (
        "mode", "total_length_m", "sources_per_arm", "n_phi", "n_cyl",
        "n_hemi", "cycles", "analysis_start_cycles", "diag_stride",
        "holdout_points_target",
    )
    settings = {k: meta[k] for k in keys}
    if settings["mode"] != "formal" or int(settings["sources_per_arm"]) != 900:
        raise RuntimeError(
            "Step 4C3 V2B is frozen for the formal 900-sources/arm dataset"
        )

    out = Path("article_cases_output") / args.out
    out.mkdir(parents=True, exist_ok=True)

    g = c3.build_geometry(settings)
    fp = c3.geometry_fingerprint(g)
    if fp != meta["geometry_fingerprint"]:
        raise RuntimeError("forcing/replay geometry fingerprint mismatch")

    times = np.asarray(z["time_s"], float)
    plasma_coll = z["plasma_collocation_E_V_per_m"]
    diag_steps = np.asarray(z["diag_steps"], np.int64)
    plasma_hold = z["plasma_holdout_E_V_per_m"]
    Hpoints = np.asarray(z["holdout_points_m"], float)
    Hnormals = np.asarray(z["holdout_normals"], float)
    Hmask = np.asarray(z["holdout_mask_matrix"], np.uint8).astype(bool)
    region_names = list(meta["holdout_region_names"])

    analysis_start_s = float(settings["analysis_start_cycles"]) / c3.F_DRIVE
    analysis_idx = np.flatnonzero(times >= analysis_start_s)
    diag_lookup = {int(s): i for i, s in enumerate(diag_steps)}
    diag_analysis = [
        int(s) for s in diag_steps
        if int(s) < len(times) and times[int(s)] >= analysis_start_s
    ]
    if len(diag_analysis) < 8:
        raise RuntimeError("too few post-ramp holdout diagnostic samples")

    # ------------------------------------------------------------------
    # Sampling-consistency control.
    # V2A used dense-in-time collocation phasors but sparse diagnostic-time
    # holdout phasors. In a nonlinear/broadband signal, those two LS estimators
    # need not be identical even with a well-conditioned fit. V2B explicitly
    # tests that possible confound before attributing anything to spatial
    # coverage.
    # ------------------------------------------------------------------
    print("[Step 4C3 V2B] fitting DENSE collocation phasors (V2A reproduction) ...", flush=True)
    coll_dense_total = np.empty(
        (len(analysis_idx), g.num_collocation_points, 3), dtype=np.float64
    )
    for jj, step in enumerate(analysis_idx):
        coll_dense_total[jj] = (
            np.asarray(plasma_coll[step], float)
            + c3.feed_at_time(g.solve_obs_points, float(times[step]))
        )
    fit_coll_dense = v2a.fit_multitone(times[analysis_idx], coll_dense_total)
    del coll_dense_total
    gc.collect()

    print("[Step 4C3 V2B] fitting SYNCHRONIZED diagnostic-time collocation/holdout phasors ...", flush=True)
    coll_sync_total = np.empty(
        (len(diag_analysis), g.num_collocation_points, 3), dtype=np.float64
    )
    hold_total = np.empty((len(diag_analysis), len(Hpoints), 3), dtype=np.float64)
    hold_times = np.empty(len(diag_analysis), dtype=np.float64)
    for jj, step in enumerate(diag_analysis):
        hi = diag_lookup[step]
        hold_times[jj] = times[step]
        coll_sync_total[jj] = (
            np.asarray(plasma_coll[step], float)
            + c3.feed_at_time(g.solve_obs_points, float(times[step]))
        )
        hold_total[jj] = (
            np.asarray(plasma_hold[hi], float)
            + c3.feed_at_time(Hpoints, float(times[step]))
        )
    fit_coll_sync = v2a.fit_multitone(hold_times, coll_sync_total)
    fit_hold = v2a.fit_multitone(hold_times, hold_total)
    del coll_sync_total, hold_total
    gc.collect()

    fit_condition = max(
        float(fit_coll_dense["condition_number"]),
        float(fit_coll_sync["condition_number"]),
        float(fit_hold["condition_number"]),
    )
    if fit_condition >= FIT_CONDITION_LIMIT:
        raise RuntimeError(
            f"harmonic fit is ill-conditioned: cond={fit_condition}"
        )

    hold_t1, hold_t2 = tangent_basis_from_normals(Hnormals)
    fold, split_method = spatial_interleaved_folds(Hpoints)
    split_diag = split_diagnostics(
        Hpoints, Hmask, region_names, fold, np.asarray(g.solve_obs_points, float)
    )

    # Frozen split-balance integrity: every named region must retain a
    # meaningful independent sample in both directions.
    for r, rr in split_diag["region_counts"].items():
        if rr["total"] > 0:
            frac = rr["A"] / rr["total"]
            if not (0.35 <= frac <= 0.65):
                raise RuntimeError(
                    f"cross-validation split is not balanced in region {r}: {rr}"
                )

    sync_rms = {}
    for f in FREQS:
        E = np.asarray(fit_coll_sync["phasors"][f], np.complex128)
        sync_rms[f] = float(np.sqrt(np.mean(np.sum(np.abs(E)**2, axis=1))))
    max_rms = max(sync_rms.values())

    ref_freqs = {}
    if ref_report is not None:
        ref_freqs = ref_report[
            "actual_plasma_spatial_harmonic_audit"
        ]["frequencies"]

    results = {}
    assessable = []
    baseline_repro_all = True
    sampling_control_passes = []
    comparisons = []

    obs_chunk = 8
    for f in FREQS:
        key = f"{f/1e6:.0f}MHz"
        strength = sync_rms[f] / max(max_rms, TINY)
        diagnostic_valid = bool(strength >= MIN_HARMONIC_STRENGTH_RATIO)
        if not diagnostic_valid:
            results[key] = {
                "diagnostic_valid": False,
                "incident_strength_ratio_to_max_harmonic": strength,
                "reason": "incident harmonic too weak",
            }
            continue
        assessable.append(key)

        print(f"[Step 4C3 V2B] {key}: assembling causal retarded operator ...", flush=True)
        omega = 2*np.pi*f
        Edense = np.asarray(fit_coll_dense["phasors"][f], np.complex128)
        Esync = np.asarray(fit_coll_sync["phasors"][f], np.complex128)
        Ehold = np.asarray(fit_hold["phasors"][f], np.complex128)
        e_dense = c2.incident_coeff(Edense, g)
        e_sync = c2.incident_coeff(Esync, g)
        eH = field_to_coeff(Ehold, hold_t1, hold_t2)

        sampling_rel = float(
            np.linalg.norm(e_sync - e_dense)
            / max(float(np.linalg.norm(e_dense)), TINY)
        )
        nd = float(np.linalg.norm(e_dense))
        ns = float(np.linalg.norm(e_sync))
        sampling_amp = abs(ns / max(nd, TINY) - 1.0)
        sampling_phase = (
            abs(float(np.angle(np.vdot(e_dense, e_sync), deg=True)))
            if nd > TINY and ns > TINY else float("inf")
        )

        A0, diag0 = c2.assemble_retarded_harmonic(g, omega, obs_chunk)
        AH, diagH = assemble_retarded_harmonic_points(
            Hpoints, hold_t1, hold_t2, g, omega, obs_chunk=16
        )

        # --------------------------------------------------------------
        # Dense baseline = exact V2A reproduction.
        # --------------------------------------------------------------
        denseprep = c2.prepare_complex(A0, g.solve_weights)
        q_dense = c2.solve_complex(denseprep, -e_dense)
        del denseprep
        gc.collect()
        dense_coll = float(
            np.linalg.norm(A0 @ q_dense + e_dense)
            / max(float(np.linalg.norm(e_dense)), TINY)
        )
        dense_residual_H = AH @ q_dense + eH
        dense_regions = coeff_region_metrics(
            dense_residual_H, eH, Hmask, region_names
        )
        dense_worst = worst_region(dense_regions)

        repro = None
        if key in ref_freqs and ref_freqs[key].get("diagnostic_valid"):
            rr = ref_freqs[key]
            ref_coll = float(rr["continuous_retarded_collocation_rel_l2"])
            ref_ft = float(rr["feed_transition"]["gate_value"])
            ref_worst = float(rr["worst_region"]["gate_value"])
            dcoll = abs(dense_coll - ref_coll)
            dft = abs(dense_regions["feed_transition"]["gate_value"] - ref_ft)
            dworst = abs(dense_worst["gate_value"] - ref_worst)
            repro = {
                "reference_collocation": ref_coll,
                "recomputed_collocation": dense_coll,
                "reference_feed_transition": ref_ft,
                "recomputed_feed_transition": dense_regions["feed_transition"]["gate_value"],
                "reference_worst_gate": ref_worst,
                "recomputed_worst_gate": dense_worst["gate_value"],
                "max_absolute_difference": max(dcoll, dft, dworst),
                "tolerance": REPRO_TOL,
                "passed": bool(max(dcoll, dft, dworst) <= REPRO_TOL),
            }
            if not repro["passed"]:
                baseline_repro_all = False
                raise RuntimeError(
                    f"V2B did not reproduce V2A {key}: {repro}"
                )

        # --------------------------------------------------------------
        # Synchronized baseline: same diagnostic-time temporal estimator on
        # collocation and holdout. This is the required control before any
        # spatial-refinement inference.
        # --------------------------------------------------------------
        syncprep = c2.prepare_complex(A0, g.solve_weights)
        q_sync = c2.solve_complex(syncprep, -e_sync)
        del syncprep
        gc.collect()
        sync_coll = float(
            np.linalg.norm(A0 @ q_sync + e_sync)
            / max(float(np.linalg.norm(e_sync)), TINY)
        )
        sync_residual_H = AH @ q_sync + eH
        sync_regions = coeff_region_metrics(
            sync_residual_H, eH, Hmask, region_names
        )
        sync_worst = worst_region(sync_regions)
        sync_pass = bool(
            sync_coll < BOUNDARY_GATE
            and sync_worst["gate_value"] < BOUNDARY_GATE
        )
        sampling_control_passes.append(sync_pass)

        directions = {}
        if not sync_pass:
            # Cross-validation is performed ONLY after the temporal sampling
            # estimator has been synchronized. Otherwise dense-vs-sparse fit
            # mismatch could masquerade as a spatial coverage problem.
            for train_fold in (0, 1):
                valid_fold = 1 - train_fold
                train_mask = fold == train_fold
                valid_mask = fold == valid_fold
                train_rows = coeff_rows_for_point_mask(train_mask)
                valid_rows = coeff_rows_for_point_mask(valid_mask)
                Atrain = AH[train_rows]
                Avalid = AH[valid_rows]
                etrain = eH[train_rows]
                evalid = eH[valid_rows]

                wtrain, nn, ndist, wdiag = inherited_supplemental_weights(
                    Hpoints[train_mask],
                    np.asarray(g.solve_obs_points, float),
                    np.asarray(g.solve_weights, float),
                )

                direction_name = (
                    "A_to_B" if train_fold == 0 else "B_to_A"
                )
                print(
                    f"[Step 4C3 V2B] {key} {direction_name}: "
                    f"{np.sum(train_mask)} supplemental points -> "
                    f"{np.sum(valid_mask)} unseen points ...",
                    flush=True,
                )

                prep = prepare_complex_blocks(
                    [A0, Atrain],
                    [np.asarray(g.solve_weights, float), wtrain],
                )
                qaug = solve_complex_blocks(
                    prep,
                    [A0, Atrain],
                    [-e_sync, -etrain],
                )

                orig_coll = float(
                    np.linalg.norm(A0 @ qaug + e_sync)
                    / max(float(np.linalg.norm(e_sync)), TINY)
                )
                train_rel = float(
                    np.linalg.norm(Atrain @ qaug + etrain)
                    / max(float(np.linalg.norm(etrain)), TINY)
                )
                valid_rel = float(
                    np.linalg.norm(Avalid @ qaug + evalid)
                    / max(float(np.linalg.norm(evalid)), TINY)
                )
                sync_base_valid_rel = float(
                    np.linalg.norm(Avalid @ q_sync + evalid)
                    / max(float(np.linalg.norm(evalid)), TINY)
                )

                aug_residual_H = AH @ qaug + eH
                aug_regions = coeff_region_metrics(
                    aug_residual_H, eH, Hmask, region_names, valid_mask
                )
                sync_val_regions = coeff_region_metrics(
                    sync_residual_H, eH, Hmask, region_names, valid_mask
                )
                aug_worst = worst_region(aug_regions)
                sync_val_worst = worst_region(sync_val_regions)

                qchange = float(
                    np.linalg.norm(qaug - q_sync)
                    / max(float(np.linalg.norm(q_sync)), TINY)
                )
                improvement = float(
                    sync_val_worst["gate_value"]
                    / max(aug_worst["gate_value"], TINY)
                )

                gates = {
                    "original_collocation_lt_5pct": bool(orig_coll < BOUNDARY_GATE),
                    "unseen_validation_worst_region_lt_5pct": bool(
                        aug_worst["gate_value"] < BOUNDARY_GATE
                    ),
                    "unseen_validation_all_unique_lt_5pct": bool(
                        valid_rel < BOUNDARY_GATE
                    ),
                }
                direction_pass = all(gates.values())

                directions[direction_name] = {
                    "train_fold": "A" if train_fold == 0 else "B",
                    "validation_fold": "B" if valid_fold == 1 else "A",
                    "train_point_count": int(np.sum(train_mask)),
                    "validation_point_count": int(np.sum(valid_mask)),
                    "supplemental_weighting": wdiag,
                    "ridge": {
                        "smax": prep["smax"],
                        "ridge_alpha": prep["ridge_alpha"],
                        "lambda_rel": c2.LAMBDA_REL,
                    },
                    "synchronized_baseline_validation_all_unique_rel_l2": sync_base_valid_rel,
                    "augmented_training_all_unique_rel_l2": train_rel,
                    "augmented_validation_all_unique_rel_l2": valid_rel,
                    "synchronized_baseline_validation_regions": sync_val_regions,
                    "synchronized_baseline_validation_worst_region": sync_val_worst,
                    "augmented_validation_regions": aug_regions,
                    "augmented_validation_worst_region": aug_worst,
                    "validation_worst_improvement_factor": improvement,
                    "original_production_collocation_after_augmentation": orig_coll,
                    "source_coefficient_relative_change_from_synchronized_baseline": qchange,
                    "synchronized_baseline_weighted_current_rms_A_per_m": weighted_current_rms(g, q_sync),
                    "augmented_weighted_current_rms_A_per_m": weighted_current_rms(g, qaug),
                    "gates": gates,
                    "passed": bool(direction_pass),
                }
                comparisons.append({
                    "frequency": key,
                    "direction": direction_name,
                    "baseline": sync_val_worst["gate_value"],
                    "augmented": aug_worst["gate_value"],
                    "improved": bool(
                        aug_worst["gate_value"] < sync_val_worst["gate_value"]
                    ),
                    "factor": improvement,
                })

                del prep, qaug, aug_residual_H, Atrain, Avalid
                gc.collect()

        results[key] = {
            "diagnostic_valid": True,
            "incident_strength_ratio_to_max_harmonic": strength,
            "sampling_consistency": {
                "dense_collocation_fit_condition_number": fit_coll_dense["condition_number"],
                "synchronized_collocation_fit_condition_number": fit_coll_sync["condition_number"],
                "holdout_fit_condition_number": fit_hold["condition_number"],
                "dense_vs_synchronized_tangential_phasor_rel_l2": sampling_rel,
                "dense_vs_synchronized_tangential_amplitude_relative_error": sampling_amp,
                "dense_vs_synchronized_tangential_phase_error_deg": sampling_phase,
            },
            "v2a_dense_baseline": {
                "collocation_rel_l2": dense_coll,
                "regions": dense_regions,
                "worst_region": dense_worst,
                "v2a_reproduction": repro,
                "weighted_current_rms_A_per_m": weighted_current_rms(g, q_dense),
            },
            "synchronized_sampling_baseline": {
                "collocation_rel_l2": sync_coll,
                "regions": sync_regions,
                "worst_region": sync_worst,
                "passed": sync_pass,
                "weighted_current_rms_A_per_m": weighted_current_rms(g, q_sync),
            },
            "retarded_operator": {
                "production_collocation": diag0,
                "holdout": diagH,
            },
            "coverage_cross_validation_was_needed": bool(not sync_pass),
            "directions": directions,
            "both_directions_pass": bool(
                len(directions) > 0 and all(d["passed"] for d in directions.values())
            ) if not sync_pass else None,
        }

        del A0, AH, q_dense, q_sync, dense_residual_H, sync_residual_H
        gc.collect()

    valid_freq_results = [results[k] for k in assessable]
    sampling_control_all_pass = bool(
        len(valid_freq_results) > 0
        and all(r["synchronized_sampling_baseline"]["passed"] for r in valid_freq_results)
    )
    failing_after_sync = [
        k for k in assessable
        if not results[k]["synchronized_sampling_baseline"]["passed"]
    ]
    strong_coverage_support = bool(
        len(failing_after_sync) > 0
        and all(results[k]["both_directions_pass"] for k in failing_after_sync)
    )

    num_improved = sum(int(x["improved"]) for x in comparisons)
    if sampling_control_all_pass:
        diagnosis = (
            "SAMPLING_ESTIMATOR_CONFOUND_IDENTIFIED: once collocation and holdout "
            "phasors use the same diagnostic-time samples, all assessable "
            "harmonic holdout gates pass without any spatial refinement. V2A's "
            "dense-vs-sparse harmonic comparison must not be used as evidence "
            "for source/collocation under-resolution."
        )
        next_test = (
            "Do not density-retune. Return to the time-domain 5.212% C3-V2 "
            "feed-transition residual and diagnose temporal/charge closure or "
            "increase holdout temporal recording density in a future controlled "
            "record if needed."
        )
    elif strong_coverage_support:
        diagnosis = (
            "STRONG_COVERAGE_SUPPORT_AFTER_SAMPLING_CONTROL: synchronized "
            "baseline still fails, but the fixed 900-source basis passes both "
            "unseen cross-validation directions after independent surface "
            "collocation coverage is added. Collocation coverage is the primary "
            "spatial limitation."
        )
        next_test = (
            "Freeze a production collocation-refinement design, regenerate only "
            "the surface forcing at the refined collocation set, then rerun C3 "
            "V2; do not increase source density yet."
        )
    elif num_improved > 0:
        diagnosis = (
            "COVERAGE_HELPS_BUT_NOT_SUFFICIENT_AFTER_SAMPLING_CONTROL: unseen "
            f"validation improves in {num_improved}/{len(comparisons)} "
            "frequency-direction comparisons, but at least one frozen 5% gate "
            "remains open."
        )
        next_test = (
            "Run a controlled source-basis-density/placement probe with the "
            "same synchronized observation phasors; keep lambda/dt/physics fixed."
        )
    else:
        diagnosis = (
            "NO_CLEAR_COVERAGE_BENEFIT_AFTER_SAMPLING_CONTROL: synchronized "
            "holdout still fails and supplemental independent collocation "
            "coverage does not improve unseen validation."
        )
        next_test = (
            "Run a controlled source-basis-density/placement diagnostic with "
            "the same synchronized incident phasors; do not retune lambda/dt."
        )

    report = {
        "step": "4C3-V2B",
        "purpose": (
            "sampling-controlled spatial collocation-coverage two-fold "
            "cross-validation"
        ),
        "mode": settings["mode"],
        "forcing_file": str(forcing_path),
        "no_PIC_rerun": True,
        "source_basis_fixed_sources_per_arm": int(settings["sources_per_arm"]),
        "lambda_rel_fixed": c2.LAMBDA_REL,
        "zero_softening": True,
        "boundary_gate": BOUNDARY_GATE,
        "geometry_fingerprint_match": bool(fp == meta["geometry_fingerprint"]),
        "harmonic_fit": {
            "dense_collocation_condition_number": fit_coll_dense["condition_number"],
            "synchronized_collocation_condition_number": fit_coll_sync["condition_number"],
            "holdout_condition_number": fit_hold["condition_number"],
            "max_condition_number": fit_condition,
            "minimum_harmonic_strength_ratio": MIN_HARMONIC_STRENGTH_RATIO,
        },
        "cross_validation_split": {
            **split_method,
            **split_diag,
        },
        "frequencies": results,
        "v2a_dense_baseline_reproduced": bool(baseline_repro_all),
        "sampling_control_all_assessable_harmonics_pass": sampling_control_all_pass,
        "frequencies_failing_after_sampling_control": failing_after_sync,
        "comparison_summary": comparisons,
        "num_validation_comparisons_improved": int(num_improved),
        "num_validation_comparisons_total": int(len(comparisons)),
        "strong_coverage_support_after_sampling_control": strong_coverage_support,
        "diagnosis": diagnosis,
        "recommended_next_test": next_test,
        "scope_note": (
            "V2B is diagnostic. Supplemental holdout points used for training "
            "in one fold are never counted as validation in that fold. The "
            "sampling-consistency control is evaluated before any spatial "
            "refinement inference. No cross-validation current amplitude is "
            "manuscript-final."
        ),
    }

    (out / "step4c3_v2b_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    lines = [
        "Step 4C3 V2B sampling-controlled spatial cross-validation",
        f"mode: {settings['mode']}",
        f"holdout folds: A={split_diag['fold_sizes']['A']}, "
        f"B={split_diag['fold_sizes']['B']}",
        f"max harmonic fit condition: {fit_condition:.6g}",
        f"V2A dense baseline reproduced: {baseline_repro_all}",
    ]
    for k in assessable:
        r = results[k]
        lines.append(
            f"{k}: dense/sync incident mismatch="
            f"{r['sampling_consistency']['dense_vs_synchronized_tangential_phasor_rel_l2']:.6g}; "
            f"V2A_dense_worst={r['v2a_dense_baseline']['worst_region']['gate_value']:.6g} "
            f"({r['v2a_dense_baseline']['worst_region']['region']}); "
            f"sync_worst={r['synchronized_sampling_baseline']['worst_region']['gate_value']:.6g} "
            f"({r['synchronized_sampling_baseline']['worst_region']['region']}); "
            f"sync_pass={r['synchronized_sampling_baseline']['passed']}"
        )
        for dname, d in r["directions"].items():
            lines.append(
                f"  {dname}: sync_base_val_worst="
                f"{d['synchronized_baseline_validation_worst_region']['gate_value']:.6g}, "
                f"aug_val_worst={d['augmented_validation_worst_region']['gate_value']:.6g}, "
                f"all_unique={d['augmented_validation_all_unique_rel_l2']:.6g}, "
                f"orig_coll={d['original_production_collocation_after_augmentation']:.6g}, "
                f"improve_x={d['validation_worst_improvement_factor']:.6g}, "
                f"pass={d['passed']}"
            )
    lines += [
        f"sampling control all harmonics pass: {sampling_control_all_pass}",
        f"frequencies failing after sampling control: {failing_after_sync}",
        f"validation comparisons improved: {num_improved}/{len(comparisons)}",
        f"strong coverage support after sampling control: {strong_coverage_support}",
        f"diagnosis: {diagnosis}",
        f"recommended next test: {next_test}",
    ]
    (out / "step4c3_v2b_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))

def self_test():
    # Tiny physical geometry for operator/evaluator comparison.
    s = c2.settings(True)
    s = dict(s)
    s.update(sources_per_arm=16, n_phi=16, n_cyl=16, n_hemi=6, obs_chunk=8)
    g = c2.geometry(s)
    omega = 2*np.pi*3.0e6

    P = np.asarray(g.solve_obs_points[:24], float)
    N = np.asarray(g.solve_normals[:24], float)
    t1, t2 = tangent_basis_from_normals(N)
    orth = max(
        float(np.max(np.abs(np.sum(t1*N, axis=1)))),
        float(np.max(np.abs(np.sum(t2*N, axis=1)))),
        float(np.max(np.abs(np.sum(t1*t2, axis=1)))),
        float(np.max(np.abs(np.linalg.norm(t1, axis=1)-1))),
        float(np.max(np.abs(np.linalg.norm(t2, axis=1)-1))),
    )

    A, _ = assemble_retarded_harmonic_points(P, t1, t2, g, omega, obs_chunk=8)
    rng = np.random.default_rng(20261002)
    q = rng.normal(size=A.shape[1]) + 1j*rng.normal(size=A.shape[1])
    E = c2.eval_retarded_field(P, g, q, omega, obs_chunk=32, src_chunk=32)
    ecoeff = field_to_coeff(E, t1, t2)
    op_err = float(
        np.linalg.norm(A @ q - ecoeff)
        / max(float(np.linalg.norm(ecoeff)), TINY)
    )

    # Tangent-coefficient region metrics must equal the direct 3-D evaluator.
    Einc = (rng.normal(size=(len(P), 3)) + 1j*rng.normal(size=(len(P), 3)))
    einc = field_to_coeff(Einc, t1, t2)
    residual_coeff = A @ q + einc
    mask_matrix = np.vstack([
        np.ones(len(P), dtype=bool),
        np.arange(len(P)) % 2 == 0,
    ])
    region_names = ["global", "side_mid"]
    direct_metrics = v2a.complex_region_metrics(
        Einc + E, Einc, N, mask_matrix, region_names
    )
    coeff_metrics = coeff_region_metrics(
        residual_coeff, einc, mask_matrix, region_names
    )
    metric_err = max(
        abs(direct_metrics[r]["gate_value"] - coeff_metrics[r]["gate_value"])
        for r in region_names
    )

    # Block ridge must be identical to explicit vstack on a small system.
    A0 = A[:28].copy()
    A1 = A[28:44].copy()
    w0 = np.linspace(0.7, 1.3, A0.shape[0]//2)
    w1 = np.linspace(0.9, 1.1, A1.shape[0]//2)
    b0 = rng.normal(size=A0.shape[0]) + 1j*rng.normal(size=A0.shape[0])
    b1 = rng.normal(size=A1.shape[0]) + 1j*rng.normal(size=A1.shape[0])

    explicit_A = np.vstack([A0, A1])
    explicit_w = np.concatenate([w0, w1])
    ep = c2.prepare_complex(explicit_A, explicit_w)
    qe = c2.solve_complex(ep, np.concatenate([b0, b1]))
    bp = prepare_complex_blocks([A0, A1], [w0, w1])
    qb = solve_complex_blocks(bp, [A0, A1], [b0, b1])
    ridge_err = float(
        np.linalg.norm(qb - qe) / max(float(np.linalg.norm(qe)), TINY)
    )

    # Split integrity on a deterministic surface sample.
    Psplit = np.asarray(g.solve_obs_points[:200], float)
    fold, _ = spatial_interleaved_folds(Psplit)
    split_ok = bool(
        np.all(np.isin(fold, [0, 1]))
        and np.sum(fold == 0) > 0
        and np.sum(fold == 1) > 0
    )

    passed = bool(
        orth < 1e-12
        and op_err < 1e-11
        and metric_err < 1e-12
        and ridge_err < 1e-8
        and split_ok
    )
    report = {
        "self_test_pass": passed,
        "tangent_orthonormality_max_error": orth,
        "arbitrary_point_operator_vs_direct_field_relative_error": op_err,
        "coefficient_metrics_vs_direct_3d_gate_max_abs_error": metric_err,
        "block_ridge_vs_explicit_vstack_relative_error": ridge_err,
        "split_nonempty_and_binary": split_ok,
        "split_counts": {
            "A": int(np.sum(fold == 0)),
            "B": int(np.sum(fold == 1)),
        },
    }
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("Step 4C3 V2B self-test failed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--forcing", default=None)
    p.add_argument("--v2a-report", default=None)
    p.add_argument("--out", default="step4c3_v2b_formal")
    a = p.parse_args()
    if a.self_test:
        self_test()
        return
    if a.forcing is None:
        raise RuntimeError("--forcing is required")
    if a.v2a_report is None:
        raise RuntimeError("--v2a-report is required for frozen baseline reproduction")
    run(a)


if __name__ == "__main__":
    main()
