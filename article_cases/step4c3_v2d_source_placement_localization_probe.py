#!/usr/bin/env python3
"""
Step 4C3 V2D — fixed-N source-placement/localization probe.

Scientific control:
  * prerequisite: completed V2C-A Branch D on the same recorded formal forcing;
  * reuse the V2B/V2C-A synchronized 22-time multitone phasor estimator;
  * freeze original 900-case collocation and independent holdout observations;
  * freeze sources_per_arm = 900, d/R=0.8, lambda_rel=1e-4, zero softening,
    3/6/9 MHz, 5% boundary gate, and eta=0.20 weak-region rule;
  * change only the deterministic, geometry-only source placement rule on the
    SAME inner candidate surface; no forcing, holdout labels, or residual values
    are used to choose source locations;
  * keep the same inner-candidate Voronoi source-weight rule;
  * reuse the Step-4C2-V4 Retarded harmonic operator, weighted/scaled solve,
    and independent pairwise holdout evaluator.

Predeclared placement rules (all 900 sources/arm):
  1) baseline_inner_surface_fps
       Frozen production global FPS on the coherent inner source surface.
  2) physical_anchor_fps
       FPS indices are chosen by Euclidean distance between the corresponding
       physical-surface anchors, while the actual sources remain on the same
       d/R=0.8 inner candidate surface.
  3) physical_area_stratified_anchor_fps
       The physical surface is split only by geometry into lower rounded end,
       cylindrical side, and upper rounded end. Source quotas are proportional
       to the frozen physical candidate areas; FPS within each stratum uses the
       physical-anchor coordinates. No diagnostic region or residual enters the
       selection.

This is a diagnostic controlled experiment. Passing V2D does NOT by itself
close Step 4C3 or permit Step 4D.
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
SOURCE_COUNT_PER_ARM = 900
BOUNDARY_GATE = 0.05
DEPTH_FRAC = 0.80
LAMBDA_REL = 1.0e-4
SOFTENING_M = 0.0
PHASOR_SAMPLE_COUNT = 22
FIT_CONDITION_LIMIT = 100.0
BASELINE_REPRO_TOL = 2.0e-8
OPERATOR_EVALUATOR_TOL = 5.0e-11
TREND_TOL = 1.0e-8
TINY = 1.0e-300

PLACEMENT_RULES = (
    "baseline_inner_surface_fps",
    "physical_anchor_fps",
    "physical_area_stratified_anchor_fps",
)
ALTERNATIVE_RULES = PLACEMENT_RULES[1:]
STRATA = ("lower_rounded", "cylindrical_side", "upper_rounded")
STRATUM_SEED_OFFSETS = {
    "lower_rounded": 0,
    "cylindrical_side": 1000,
    "upper_rounded": 2000,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Step 4C3 V2D fixed-N source-placement/localization probe"
    )
    p.add_argument(
        "--forcing",
        default="article_cases_output/step4c3_formal/step4c3_forcing.npz",
        help="Recorded formal Step-4C3 forcing NPZ at the original observation points",
    )
    p.add_argument(
        "--v2c-a-report",
        default="article_cases_output/step4c3_v2c_a_formal/step4c3_v2c_a_report.json",
        help="Completed V2C-A formal report; V2D requires the predeclared Branch D",
    )
    p.add_argument(
        "--out",
        default="step4c3_v2d_formal",
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
    if isinstance(obj, np.floating):
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
    """Invoke the production setter, then restore the frozen arrays bitwise."""
    g.set_solve_collocation(points, normals, weights)
    g.solve_obs_points = np.asarray(points, dtype=np.float64).copy()
    g.solve_normals = np.asarray(normals, dtype=np.float64).copy()
    g.solve_weights = np.asarray(weights, dtype=np.float64).copy()
    g.solve_tangent1 = np.asarray(tangent1, dtype=np.float64).copy()
    g.solve_tangent2 = np.asarray(tangent2, dtype=np.float64).copy()
    return g


def largest_remainder_quotas(total: int, weights, capacities, labels):
    total = int(total)
    w = np.asarray(weights, dtype=np.float64)
    cap = np.asarray(capacities, dtype=np.int64)
    if total <= 0 or len(w) == 0 or len(w) != len(cap) or len(w) != len(labels):
        raise ValueError("invalid quota inputs")
    if np.any(w <= 0.0) or np.any(cap <= 0):
        raise RuntimeError("all placement strata must have positive area and capacity")
    if total > int(np.sum(cap)):
        raise RuntimeError("stratified placement requests more sources than candidate capacity")

    raw = total * w / float(np.sum(w))
    q = np.floor(raw).astype(np.int64)
    # Every non-empty geometric stratum receives at least one source when possible.
    if total >= len(q):
        q = np.maximum(q, 1)

    while int(np.sum(q)) < total:
        candidates = [i for i in range(len(q)) if q[i] < cap[i]]
        if not candidates:
            raise RuntimeError("cannot satisfy stratified source quota within candidate capacities")
        # Largest positive remainder; stable tie break by declared stratum order.
        i = max(candidates, key=lambda j: (raw[j] - q[j], -j))
        q[i] += 1
    while int(np.sum(q)) > total:
        candidates = [i for i in range(len(q)) if q[i] > 1]
        if not candidates:
            raise RuntimeError("cannot reduce stratified source quota without emptying a stratum")
        # Remove first from the most over-allocated stratum relative to raw quota.
        i = max(candidates, key=lambda j: (q[j] - raw[j], -j))
        q[i] -= 1

    if np.any(q > cap):
        raise RuntimeError("stratified quota exceeds a geometric stratum candidate capacity")
    return {str(labels[i]): int(q[i]) for i in range(len(q))}


def source_candidate_reference(settings):
    """Reconstruct the exact frozen inner/physical candidate surfaces."""
    R = 1.0e-3
    total = float(settings["total_length_m"])
    gap = 2.0e-3
    Larm = 0.5 * (total - gap)
    Rs = (1.0 - DEPTH_FRAC) * R
    n_phi = int(settings["n_phi"])
    n_cyl = int(settings["n_cyl"])
    n_hemi = int(settings["n_hemi"])

    physical_local = mp._generate_capsule_triangles(R, R, Larm, n_cyl, n_phi, n_hemi)
    inner_local = mp._generate_capsule_triangles(Rs, R, Larm, n_cyl, n_phi, n_hemi)

    Cphys0, _, Aphys = mp._triangle_geometry(physical_local)
    Cphys, Nphys = mp._project_to_capsule(Cphys0, R, R, Larm)
    Cinner0, _, Ainner = mp._triangle_geometry(inner_local)
    Cinner, Ninner = mp._project_to_capsule(Cinner0, Rs, R, Larm)
    Aanchor, Nanchor = mp._project_to_capsule(Cinner, R, R, Larm)

    if not (len(Cphys) == len(Cinner) == len(Aphys) == len(Ainner) == len(Aanchor)):
        raise RuntimeError("physical/inner candidate topology mismatch")

    tol = 1.0e-14
    low = Aanchor[:, 2] < (R - tol)
    high = Aanchor[:, 2] > (Larm - R + tol)
    side = ~(low | high)
    zone_masks = {
        "lower_rounded": low,
        "cylindrical_side": side,
        "upper_rounded": high,
    }
    for name, mask in zone_masks.items():
        if not np.any(mask):
            raise RuntimeError(f"empty physical placement stratum: {name}")

    zone_areas = {name: float(np.sum(Aphys[mask])) for name, mask in zone_masks.items()}
    zone_capacities = {name: int(np.sum(mask)) for name, mask in zone_masks.items()}
    area_arm = float(np.sum(Aphys))
    zshifts = (-0.5 * total, 0.5 * gap)

    arms = {}
    for arm, zshift in enumerate(zshifts):
        inner_global = mp._shift_z(Cinner, zshift)
        anchor_global = mp._shift_z(Aanchor, zshift)
        physical_candidate_global = mp._shift_z(Cphys, zshift)
        physical_tri_global = mp._shift_z(physical_local, zshift)
        arms[arm] = {
            "inner_global": inner_global,
            "anchor_global": anchor_global,
            "physical_candidate_global": physical_candidate_global,
            "physical_triangles_global": physical_tri_global,
        }

    # Compatibility fingerprint exactly matches the V2C-A source-candidate contract.
    v2ca_fp = hash_arrays_exact(
        np.concatenate([arms[0]["inner_global"], arms[1]["inner_global"]]),
        np.concatenate([Ninner, Ninner]),
        np.tile(np.asarray(Ainner, dtype=np.float64), 2),
    )
    expanded_fp = hash_arrays_exact(
        np.concatenate([arms[0]["inner_global"], arms[1]["inner_global"]]),
        np.concatenate([arms[0]["anchor_global"], arms[1]["anchor_global"]]),
        np.concatenate([Ninner, Ninner]),
        np.tile(np.asarray(Ainner, dtype=np.float64), 2),
        np.tile(np.asarray(Aphys, dtype=np.float64), 2),
    )

    return {
        "R": R,
        "total": total,
        "gap": gap,
        "Larm": Larm,
        "Rs": Rs,
        "area_arm": area_arm,
        "inner_local": Cinner,
        "inner_normals": Ninner,
        "inner_areas": Ainner,
        "physical_candidate_local": Cphys,
        "physical_normals": Nphys,
        "physical_areas": Aphys,
        "anchor_local": Aanchor,
        "anchor_normals": Nanchor,
        "zone_masks": zone_masks,
        "zone_areas": zone_areas,
        "zone_capacities": zone_capacities,
        "arms": arms,
        "candidate_points_per_arm": int(len(Cinner)),
        "candidate_points_total": int(2 * len(Cinner)),
        "v2ca_candidate_geometry_fingerprint": v2ca_fp,
        "expanded_candidate_geometry_fingerprint": expanded_fp,
    }


def selection_indices(candidate_ref, arm: int, rule: str, nsrc: int):
    arm = int(arm)
    nsrc = int(nsrc)
    if rule not in PLACEMENT_RULES:
        raise ValueError(f"unknown placement rule: {rule}")
    if nsrc > int(candidate_ref["candidate_points_per_arm"]):
        raise RuntimeError("requested source count exceeds candidate surface capacity")

    inner = candidate_ref["arms"][arm]["inner_global"]
    anchor = candidate_ref["arms"][arm]["anchor_global"]
    base_seed = 6100 + arm

    if rule == "baseline_inner_surface_fps":
        return mp._fps_indices(inner, nsrc, seed=base_seed), {
            "quota_rule": "none_global_fps",
            "stratum_quotas": {name: 0 for name in STRATA},
        }

    if rule == "physical_anchor_fps":
        return mp._fps_indices(anchor, nsrc, seed=base_seed), {
            "quota_rule": "none_global_fps",
            "stratum_quotas": {name: 0 for name in STRATA},
        }

    weights = [candidate_ref["zone_areas"][name] for name in STRATA]
    capacities = [candidate_ref["zone_capacities"][name] for name in STRATA]
    quotas = largest_remainder_quotas(nsrc, weights, capacities, STRATA)
    selected = []
    for name in STRATA:
        candidates = np.flatnonzero(candidate_ref["zone_masks"][name])
        quota = int(quotas[name])
        seed = base_seed + int(STRATUM_SEED_OFFSETS[name])
        local = mp._fps_indices(anchor[candidates], quota, seed=seed)
        selected.append(candidates[local])
    idx = np.concatenate(selected).astype(np.int64, copy=False)
    if len(idx) != nsrc or len(np.unique(idx)) != nsrc:
        raise RuntimeError("stratified source selection did not produce exactly N unique indices")
    return idx, {
        "quota_rule": "largest_remainder_proportional_to_physical_candidate_area",
        "stratum_quotas": {name: int(quotas[name]) for name in STRATA},
    }


def build_geometry_for_rule(settings, candidate_ref, fixed_cloud, rule: str, nsrc: int):
    s = dict(settings)
    s["sources_per_arm"] = int(nsrc)
    if rule == "baseline_inner_surface_fps":
        g = c3.build_geometry(s)
        bind_fixed_solve_cloud(g, *fixed_cloud)
        return g

    anchors = []
    sources = []
    normals = []
    sweights = []
    all_physical_tri = []
    per_arm_selection_meta = []
    for arm in (0, 1):
        idx, smeta = selection_indices(candidate_ref, arm, rule, nsrc)
        inner_global = candidate_ref["arms"][arm]["inner_global"]
        anchor_global = candidate_ref["arms"][arm]["anchor_global"]
        SP = inner_global[idx].copy()
        SN = candidate_ref["inner_normals"][idx].copy()
        SW = mp._nearest_partition_weights(
            inner_global,
            candidate_ref["inner_areas"],
            SP,
        )
        SW *= candidate_ref["area_arm"] / max(float(np.sum(SW)), TINY)
        AP = anchor_global[idx].copy()
        anchors.append(AP)
        sources.append(SP)
        normals.append(SN)
        sweights.append(SW)
        all_physical_tri.append(candidate_ref["arms"][arm]["physical_triangles_global"])
        per_arm_selection_meta.append(smeta)

    g = mp.MASGeometry(
        ant_obs_points=np.concatenate(anchors),
        ant_src_points=np.concatenate(sources),
        ant_normals=np.concatenate(normals),
        source_weights=np.concatenate(sweights),
        solve_obs_points=np.asarray(fixed_cloud[0], dtype=np.float64).copy(),
        solve_normals=np.asarray(fixed_cloud[1], dtype=np.float64).copy(),
        solve_weights=np.asarray(fixed_cloud[2], dtype=np.float64).copy(),
        surface_triangles=np.concatenate(all_physical_tri),
        metadata={
            "kind": "rounded_two_arm_dipole",
            "validated_scope": "V2D_controlled_source_placement_probe",
            "radius": candidate_ref["R"],
            "total_length": candidate_ref["total"],
            "feed_gap": candidate_ref["gap"],
            "arm_length": candidate_ref["Larm"],
            "depth_frac": DEPTH_FRAC,
            "source_offset": DEPTH_FRAC * candidate_ref["R"],
            "inner_radius": candidate_ref["Rs"],
            "sources_per_arm": int(nsrc),
            "collocation_points_per_arm": int(len(fixed_cloud[0]) // 2),
            "collocation_factor": float(c3.COLLOCATION_FACTOR),
            "n_phi": int(settings["n_phi"]),
            "n_cyl": int(settings["n_cyl"]),
            "n_hemi": int(settings["n_hemi"]),
            "source_sampling": rule,
            "source_sampling_data_dependency": "geometry_only_no_forcing_no_holdout_labels_no_residuals",
            "source_weighting": "inner_voronoi_rescaled_to_physical_arm_area",
            "per_arm_selection_metadata": per_arm_selection_meta,
            "collocation_sampling": "original_formal_900_case_fixed_bitwise",
        },
    )
    # Match the production builder's source-frame convention exactly: build from
    # the coherent pre-constructor inner normals rather than from any second
    # normalization performed inside MASGeometry.__post_init__.
    g.source_tangent1, g.source_tangent2 = mp.build_local_tangent_frames(
        np.concatenate(normals)
    )
    bind_fixed_solve_cloud(g, *fixed_cloud)
    return g


def source_cloud_diagnostics(g, candidate_ref, rule: str, nsrc: int, baseline_indices):
    if int(g.num_sources) != 2 * int(nsrc):
        raise RuntimeError(f"source count mismatch for {rule}: {g.num_sources}")

    per_arm_idx = []
    per_arm_meta = []
    per_arm_rows = []
    all_inner_nn = []
    all_anchor_nn = []
    all_anchor_weighted_sq = []
    all_anchor_weight = []
    total_overlap = 0
    total_union = 0
    feed_count = 0
    outer_count = 0
    side_count = 0

    for arm in (0, 1):
        idx, smeta = selection_indices(candidate_ref, arm, rule, nsrc)
        per_arm_idx.append(idx)
        per_arm_meta.append(smeta)
        sl = slice(arm * nsrc, (arm + 1) * nsrc)
        expected = candidate_ref["arms"][arm]["inner_global"][idx]
        actual = np.asarray(g.ant_src_points[sl], dtype=np.float64)
        if not np.array_equal(actual, expected):
            raise RuntimeError(f"source selection mismatch for {rule}, arm {arm}")

        base = np.asarray(baseline_indices[arm], dtype=np.int64)
        sidx = set(int(x) for x in idx)
        bidx = set(int(x) for x in base)
        overlap = len(sidx & bidx)
        union = len(sidx | bidx)
        total_overlap += overlap
        total_union += union

        P = actual
        inner_nn = cKDTree(P).query(P, k=2)[0][:, 1]
        all_inner_nn.extend(inner_nn.tolist())

        anchors_selected = candidate_ref["arms"][arm]["anchor_global"][idx]
        anchor_nn = cKDTree(anchors_selected).query(anchors_selected, k=2)[0][:, 1]
        all_anchor_nn.extend(anchor_nn.tolist())

        inner_cover = cKDTree(P).query(candidate_ref["arms"][arm]["inner_global"], k=1)[0]
        anchor_cover = cKDTree(anchors_selected).query(candidate_ref["arms"][arm]["anchor_global"], k=1)[0]
        aw = np.asarray(candidate_ref["physical_areas"], dtype=np.float64)
        all_anchor_weighted_sq.extend((aw * anchor_cover * anchor_cover).tolist())
        all_anchor_weight.extend(aw.tolist())

        zcounts = {
            name: int(np.sum(candidate_ref["zone_masks"][name][idx]))
            for name in STRATA
        }
        if arm == 0:
            arm_feed = zcounts["upper_rounded"]
            arm_outer = zcounts["lower_rounded"]
        else:
            arm_feed = zcounts["lower_rounded"]
            arm_outer = zcounts["upper_rounded"]
        feed_count += arm_feed
        outer_count += arm_outer
        side_count += zcounts["cylindrical_side"]

        per_arm_rows.append({
            "arm": arm,
            "selected_source_count": int(len(idx)),
            "baseline_overlap_count": int(overlap),
            "baseline_overlap_fraction": float(overlap / nsrc),
            "baseline_jaccard": float(overlap / max(union, 1)),
            "inner_min_nearest_spacing_m": float(np.min(inner_nn)),
            "inner_median_nearest_spacing_m": float(np.median(inner_nn)),
            "physical_anchor_min_nearest_spacing_m": float(np.min(anchor_nn)),
            "physical_anchor_median_nearest_spacing_m": float(np.median(anchor_nn)),
            "inner_candidate_cover_max_m": float(np.max(inner_cover)),
            "inner_candidate_cover_p95_m": float(np.quantile(inner_cover, 0.95)),
            "physical_anchor_cover_max_m": float(np.max(anchor_cover)),
            "physical_anchor_cover_p95_m": float(np.quantile(anchor_cover, 0.95)),
            "zone_counts": zcounts,
            "feed_rounded_count": int(arm_feed),
            "outer_rounded_count": int(arm_outer),
            "selection_metadata": smeta,
        })

    source_fp = hash_arrays_exact(
        g.ant_src_points,
        g.ant_normals,
        g.source_weights,
        g.source_tangent1,
        g.source_tangent2,
    )
    weight_sum = float(np.sum(g.source_weights))
    anchor_weighted_rms = float(math.sqrt(
        np.sum(all_anchor_weighted_sq) / max(float(np.sum(all_anchor_weight)), TINY)
    ))
    return {
        "placement_rule": rule,
        "sources_per_arm": int(nsrc),
        "sources_total": int(g.num_sources),
        "candidate_points_per_arm": int(candidate_ref["candidate_points_per_arm"]),
        "candidate_source_ratio_per_arm": float(candidate_ref["candidate_points_per_arm"] / nsrc),
        "candidate_geometry_fingerprint": candidate_ref["expanded_candidate_geometry_fingerprint"],
        "v2ca_candidate_geometry_fingerprint": candidate_ref["v2ca_candidate_geometry_fingerprint"],
        "selected_source_index_hash": hash_indices(per_arm_idx),
        "source_fingerprint": source_fp,
        "selection_data_dependency": "geometry_only_no_forcing_no_holdout_labels_no_residuals",
        "source_weighting_rule": str(g.metadata.get("source_weighting", "unknown")),
        "source_weight_sum": weight_sum,
        "source_weight_min": float(np.min(g.source_weights)),
        "source_weight_median": float(np.median(g.source_weights)),
        "source_weight_max": float(np.max(g.source_weights)),
        "baseline_overlap_count_total": int(total_overlap),
        "baseline_overlap_fraction_total": float(total_overlap / (2 * nsrc)),
        "baseline_jaccard_total": float(total_overlap / max(total_union, 1)),
        "inner_min_nearest_spacing_m": float(np.min(all_inner_nn)),
        "inner_median_nearest_spacing_m": float(np.median(all_inner_nn)),
        "physical_anchor_min_nearest_spacing_m": float(np.min(all_anchor_nn)),
        "physical_anchor_median_nearest_spacing_m": float(np.median(all_anchor_nn)),
        "physical_anchor_area_weighted_cover_rms_m": anchor_weighted_rms,
        "feed_rounded_source_count_total": int(feed_count),
        "outer_rounded_source_count_total": int(outer_count),
        "cylindrical_side_source_count_total": int(side_count),
        "per_arm": per_arm_rows,
        "per_arm_selection_metadata": per_arm_meta,
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
            f"V2D requires the V2B/V2C-A synchronized {PHASOR_SAMPLE_COUNT}-time estimator; "
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
            "V2D expected the V2B-assessable 3/6/9 MHz harmonics, but found weak harmonics: "
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
        "note": "diagnostic only; not a physics gate",
    }


def worst_region(regions):
    if not regions:
        raise RuntimeError("empty region metrics")
    return dict(max(regions.values(), key=lambda x: float(x["gate_value"])))


def load_v2ca_reference(path: Path):
    if not path.exists():
        raise RuntimeError(f"required V2C-A formal report not found: {path}")
    r = json.loads(path.read_text(encoding="utf-8"))
    if r.get("experiment") != "STEP4C3_V2C_A_FIXED_OBSERVATION_SOURCE_DENSITY":
        raise RuntimeError(f"unexpected V2C-A report schema in {path}")
    if r.get("status") != "complete":
        raise RuntimeError("V2D requires a completed V2C-A formal run")
    branch = r.get("decision_tree", {}).get("branch")
    if branch != "D_HIGHER_DENSITY_WORSENS_AT_LEAST_ONE_FROZEN_METRIC":
        raise RuntimeError(
            "V2D was predeclared for V2C-A Branch D; refusing to reinterpret a different upstream result"
        )
    if str(r.get("decision_tree", {}).get("minimum_all_frequency_qualified_density")) != "none":
        raise RuntimeError("V2D requires V2C-A to have no all-frequency qualified density")
    return r


def v2ca_baseline_reproduction_check(v2ca_report, f, coll_rel, regions, worst):
    ref = v2ca_report["densities"][str(SOURCE_COUNT_PER_ARM)]["frequency_results"][freq_key(f)]
    diffs = {
        "collocation_global": abs(float(coll_rel) - float(ref["collocation_global"])),
        "holdout_global": abs(float(regions["global"]["gate_value"]) - float(ref["holdout_global"])),
        "worst_gate": abs(float(worst["gate_value"]) - float(ref["worst_region"]["gate_value"])),
    }
    for region, rr in regions.items():
        if region not in ref["regions"]:
            raise RuntimeError(f"V2C-A baseline missing region {region}")
        diffs[f"region:{region}"] = abs(
            float(rr["gate_value"]) - float(ref["regions"][region]["gate_value"])
        )
    maxdiff = max(diffs.values())
    name_match = str(worst["region"]) == str(ref["worst_region"]["region"])
    passed = bool(maxdiff <= BASELINE_REPRO_TOL and name_match)
    if not passed:
        raise RuntimeError(
            f"V2D baseline failed to reproduce V2C-A 900-source result at {freq_label(f)}: "
            f"maxdiff={maxdiff:.6g}, worst_name_match={name_match}"
        )
    return {
        "reference_experiment": v2ca_report["experiment"],
        "reference_density": SOURCE_COUNT_PER_ARM,
        "absolute_differences": diffs,
        "max_absolute_difference": float(maxdiff),
        "worst_region_name_match": name_match,
        "tolerance": BASELINE_REPRO_TOL,
        "passed": True,
    }


def make_initial_report(forcing_path: Path, v2ca_path: Path, out: Path):
    return {
        "experiment": "STEP4C3_V2D_FIXED_N_SOURCE_PLACEMENT_LOCALIZATION",
        "status": "initialized",
        "purpose": "geometry-only fixed-N source-placement/localization probe after V2C-A Branch D",
        "output_directory": str(out),
        "forcing_file": str(forcing_path),
        "forcing_sha256": "not_computed_due_to_initialization",
        "v2c_a_reference_report": str(v2ca_path),
        "v2c_a_reference_report_sha256": "not_computed_due_to_initialization",
        "frozen": {
            "frequencies_Hz": [int(f) for f in FREQS],
            "sources_per_arm": SOURCE_COUNT_PER_ARM,
            "source_depth_fraction": DEPTH_FRAC,
            "lambda_rel": LAMBDA_REL,
            "softening_m": SOFTENING_M,
            "phasor_time_sample_count": PHASOR_SAMPLE_COUNT,
            "boundary_gate": BOUNDARY_GATE,
            "weak_region_threshold": c3.WEAK_REGION_THRESHOLD,
            "placement_rules": list(PLACEMENT_RULES),
            "placement_rule_priority_if_multiple_qualify": list(ALTERNATIVE_RULES),
            "candidate_surface_rule": "same frozen n_phi/n_cyl/n_hemi inner candidate surface for every placement rule",
            "source_weight_rule": "same inner-candidate Voronoi weights rescaled to physical arm area; weights recomputed because placement induces patch partition",
            "collocation_rule": "original formal 900-case points/normals/weights/tangents fixed bitwise",
            "holdout_rule": "original independent physical holdout points/normals/masks fixed bitwise",
            "selection_data_dependency": "geometry_only_no_forcing_no_holdout_labels_no_residuals",
        },
        "upstream_precondition": {"status": "not_run_due_to_initialization"},
        "fixed_observations": {"status": "not_run_due_to_initialization"},
        "phasor_fit": {"status": "not_run_due_to_initialization"},
        "source_candidate": {"status": "not_run_due_to_initialization"},
        "placements": {
            rule: {"status": "not_run_due_to_initialization"}
            for rule in PLACEMENT_RULES
        },
        "decision_tree": {
            "branch": "not_evaluated_due_to_initialization",
            "predeclared_candidate_rule": "none",
            "facts": {},
            "explanation": "not_evaluated_due_to_initialization",
        },
        "failure": {
            "present": False,
            "type": "none",
            "message": "none",
            "traceback": "none",
        },
    }


def decision_tree_assessment(report):
    complete = all(report["placements"].get(rule, {}).get("status") == "complete" for rule in PLACEMENT_RULES)
    if not complete:
        return {
            "branch": "INCOMPLETE_RUN",
            "predeclared_candidate_rule": "none",
            "facts": {},
            "explanation": "Formal placement/frequency matrix is incomplete; no scientific branch assigned.",
            "next_step_constraint": "Step 4D remains blocked.",
        }

    base = report["placements"][PLACEMENT_RULES[0]]
    facts = {}
    qualified = []
    for rule in PLACEMENT_RULES:
        pr = report["placements"][rule]
        all_pass = all(
            bool(pr["frequency_results"][freq_key(f)]["passed_collocation_and_all_holdout_regions"])
            for f in FREQS
        )
        if rule in ALTERNATIVE_RULES and all_pass:
            qualified.append(rule)
        rfacts = {
            "all_frequency_qualified": bool(all_pass),
            "delta_vs_baseline": {},
        }
        for f in FREQS:
            fk = freq_key(f)
            fr = pr["frequency_results"][fk]
            br = base["frequency_results"][fk]
            rfacts["delta_vs_baseline"][fk] = {
                "collocation_global": float(fr["collocation_global"] - br["collocation_global"]),
                "holdout_global": float(fr["holdout_global"] - br["holdout_global"]),
                "worst_region_gate": float(fr["worst_region"]["gate_value"] - br["worst_region"]["gate_value"]),
                "worst_region": str(fr["worst_region"]["region"]),
                "all_gates_pass": bool(fr["passed_collocation_and_all_holdout_regions"]),
            }
        facts[rule] = rfacts

    candidate = "none"
    branch = "E_NO_QUALIFIED_PLACEMENT_FROM_PREDECLARED_RULES"
    explanation = (
        "Neither predeclared geometry-only alternative satisfies every 3/6/9 MHz collocation and independent holdout gate at fixed N=900. "
        "Do not invent additional residual-targeted placement heuristics after seeing these results."
    )

    if qualified:
        # Predeclared complexity order, not residual ranking: physical-anchor global FPS first.
        candidate = next(rule for rule in ALTERNATIVE_RULES if rule in qualified)
        branch = "A_GEOMETRY_ONLY_PLACEMENT_QUALIFIES_FIXED_N"
        explanation = (
            "At least one predeclared geometry-only placement rule satisfies all frozen harmonic gates at 900 sources/arm. "
            f"The candidate is {candidate} by the predeclared simplicity priority, not by post-hoc residual ranking. "
            "V2D still does not close C3; the placement must be promoted consistently and tested in the real broadband replay."
        )
    else:
        b9 = base["frequency_results"][freq_key(9.0e6)]
        any_clean_9_improvement = False
        any_train_holdout_tradeoff = False
        any_six_regression = False
        any_nine_worsening = False
        for rule in ALTERNATIVE_RULES:
            pr = report["placements"][rule]
            r3 = pr["frequency_results"][freq_key(3.0e6)]
            r6 = pr["frequency_results"][freq_key(6.0e6)]
            r9 = pr["frequency_results"][freq_key(9.0e6)]
            b3 = base["frequency_results"][freq_key(3.0e6)]
            if not bool(r6["passed_collocation_and_all_holdout_regions"]):
                any_six_regression = True
            if float(r9["worst_region"]["gate_value"]) > float(b9["worst_region"]["gate_value"]) + TREND_TOL:
                any_nine_worsening = True
            if (
                float(r9["collocation_global"]) < float(b9["collocation_global"]) - TREND_TOL
                and float(r9["worst_region"]["gate_value"]) > float(b9["worst_region"]["gate_value"]) + TREND_TOL
            ):
                any_train_holdout_tradeoff = True
            if (
                float(r9["collocation_global"]) <= float(b9["collocation_global"]) + TREND_TOL
                and float(r9["worst_region"]["gate_value"]) < float(b9["worst_region"]["gate_value"]) - TREND_TOL
                and bool(r6["passed_collocation_and_all_holdout_regions"])
                and float(r3["worst_region"]["gate_value"]) <= float(b3["worst_region"]["gate_value"]) + TREND_TOL
            ):
                any_clean_9_improvement = True

        if any_clean_9_improvement:
            branch = "B_PLACEMENT_SENSITIVE_BUT_NOT_QUALIFIED"
            explanation = (
                "A predeclared geometry-only placement improves the primary 9 MHz discriminator without 6 MHz regression and without worsening the 3 MHz worst holdout, "
                "but no placement passes all frozen gates. Placement matters, but it does not close the basis/coverage issue at fixed N=900."
            )
        elif any_train_holdout_tradeoff:
            branch = "C_PLACEMENT_TRAINING_HOLDOUT_TRADEOFF"
            explanation = (
                "At least one placement improves the 9 MHz collocation fit while worsening the independent 9 MHz holdout. "
                "This is a generalization tradeoff, not evidence for qualification."
            )
        elif any_six_regression or any_nine_worsening:
            branch = "D_PLACEMENT_REGRESSION_OR_MIXED_RESPONSE"
            explanation = (
                "The tested placement alternatives either break the 6 MHz non-regression control or worsen the primary 9 MHz independent holdout. "
                "Do not select the best-looking rule or tune geometry after the fact."
            )

    return {
        "branch": branch,
        "predeclared_candidate_rule": candidate,
        "trend_tolerance_absolute": TREND_TOL,
        "facts": facts,
        "explanation": explanation,
        "next_step_constraint": (
            "V2D never closes C3 by itself. If Branch A occurs, promote only the predeclared candidate source-placement rule while keeping the observation cloud fixed, then run the real broadband replay; no new PIC field is required solely for a source-placement change on unchanged observation points. "
            "If a later production change also changes collocation points, real PIC field on those new points is required. If no placement qualifies, do not continue ad-hoc placement tuning; move to a separately predeclared basis-family/local-basis experiment. Step 4D remains blocked until production confirmation, real broadband replay, all 5% gates, long-tail <=1.001, finiteness, and fingerprint checks pass."
        ),
    }


def write_csv_outputs(out: Path, report: dict):
    out.mkdir(parents=True, exist_ok=True)

    summary_fields = [
        "placement_rule", "frequency_Hz", "collocation_global", "holdout_global",
        "worst_region", "worst_region_gate", "passed", "weighted_current_rms_A_per_m",
        "operator_evaluator_rel_l2", "scaled_operator_smax_estimate", "ridge_alpha",
        "stable_rank_proxy", "column_scale_max_over_min",
    ]
    with (out / "step4c3_v2d_placement_frequency_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for rule in PLACEMENT_RULES:
            pr = report["placements"].get(rule, {})
            for fk, fr in pr.get("frequency_results", {}).items():
                if fr.get("status") != "complete":
                    continue
                sd = fr["solver_diagnostics"]
                w.writerow({
                    "placement_rule": rule,
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
        "placement_rule", "frequency_Hz", "region", "point_count", "residual_rms",
        "incident_rms", "incident_ratio", "local_rel", "global_norm", "gate_metric",
        "gate_value", "passed",
    ]
    with (out / "step4c3_v2d_region_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=region_fields)
        w.writeheader()
        for rule in PLACEMENT_RULES:
            pr = report["placements"].get(rule, {})
            for fk, fr in pr.get("frequency_results", {}).items():
                if fr.get("status") != "complete":
                    continue
                for region, rr in fr["regions"].items():
                    row = {"placement_rule": rule, "frequency_Hz": int(fk), "region": region}
                    for k in region_fields:
                        if k in rr:
                            row[k] = rr[k]
                    w.writerow(row)

    geom_fields = [
        "placement_rule", "sources_per_arm", "sources_total", "candidate_points_per_arm",
        "candidate_source_ratio_per_arm", "selected_source_index_hash", "source_fingerprint",
        "baseline_overlap_fraction_total", "baseline_jaccard_total",
        "inner_min_nearest_spacing_m", "inner_median_nearest_spacing_m",
        "physical_anchor_min_nearest_spacing_m", "physical_anchor_median_nearest_spacing_m",
        "physical_anchor_area_weighted_cover_rms_m", "feed_rounded_source_count_total",
        "outer_rounded_source_count_total", "cylindrical_side_source_count_total",
        "source_weight_sum", "source_weight_min", "source_weight_median", "source_weight_max",
        "min_source_to_fixed_collocation_distance_m", "min_source_to_fixed_holdout_distance_m",
    ]
    with (out / "step4c3_v2d_source_geometry_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=geom_fields)
        w.writeheader()
        for rule in PLACEMENT_RULES:
            gd = report["placements"].get(rule, {}).get("source_geometry")
            if gd:
                w.writerow({k: gd[k] for k in geom_fields})

    cond_fields = [
        "placement_rule", "frequency_Hz", "scaled_operator_smax_estimate", "ridge_alpha",
        "lambda_rel", "column_scale_min", "column_scale_median", "column_scale_max",
        "column_scale_max_over_min", "stable_rank_proxy", "regularized_normal_condition_upper_bound",
    ]
    with (out / "step4c3_v2d_condition_effective_rank_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cond_fields)
        w.writeheader()
        for rule in PLACEMENT_RULES:
            pr = report["placements"].get(rule, {})
            for fk, fr in pr.get("frequency_results", {}).items():
                if fr.get("status") != "complete":
                    continue
                sd = fr["solver_diagnostics"]
                row = {"placement_rule": rule, "frequency_Hz": int(fk)}
                for k in cond_fields:
                    if k in sd:
                        row[k] = sd[k]
                w.writerow(row)


def write_summary(out: Path, report: dict):
    lines = [
        "Step 4C3 V2D fixed-N source-placement/localization probe",
        f"status: {report['status']}",
        f"upstream_v2c_a_branch: {report.get('upstream_precondition', {}).get('v2c_a_branch', 'not-run')}",
        f"sources_per_arm: {SOURCE_COUNT_PER_ARM}",
        f"collocation_fingerprint: {report.get('fixed_observations', {}).get('collocation_fingerprint', 'not-run')}",
        f"holdout_fingerprint: {report.get('fixed_observations', {}).get('holdout_fingerprint', 'not-run')}",
        f"phasor_sample_count: {report.get('phasor_fit', {}).get('sample_count', 'not-run')}",
        f"selection_data_dependency: geometry_only_no_forcing_no_holdout_labels_no_residuals",
    ]
    for rule in PLACEMENT_RULES:
        pr = report["placements"].get(rule, {})
        lines.append(f"placement {rule}: status={pr.get('status', 'missing')}")
        for fk, fr in pr.get("frequency_results", {}).items():
            if fr.get("status") == "complete":
                lines.append(
                    f"  {int(fk)/1e6:.0f}MHz colloc={fr['collocation_global']:.8g} "
                    f"holdout_global={fr['holdout_global']:.8g} "
                    f"worst={fr['worst_region']['gate_value']:.8g}({fr['worst_region']['region']}) "
                    f"pass={fr['passed_collocation_and_all_holdout_regions']}"
                )
    lines += [
        f"decision_tree_branch: {report['decision_tree']['branch']}",
        f"predeclared_candidate_rule: {report['decision_tree']['predeclared_candidate_rule']}",
        f"interpretation: {report['decision_tree'].get('explanation', 'none')}",
    ]
    if report["failure"]["present"]:
        lines.append(f"failure: {report['failure']['type']}: {report['failure']['message']}")
    (out / "step4c3_v2d_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_outputs(out: Path, report: dict):
    report["decision_tree"] = decision_tree_assessment(report)
    write_json(out / "step4c3_v2d_report.json", report)
    write_csv_outputs(out, report)
    write_summary(out, report)


def run_formal(args):
    forcing_path = resolve_input_path(args.forcing)
    v2ca_path = resolve_input_path(args.v2c_a_report)
    out = resolve_out_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = make_initial_report(forcing_path, v2ca_path, out)
    current_rule = "none"

    try:
        if not forcing_path.exists():
            raise FileNotFoundError(f"forcing file not found: {forcing_path}")
        if not v2ca_path.exists():
            raise FileNotFoundError(f"V2C-A report not found: {v2ca_path}")
        if int(args.operator_obs_chunk) <= 0 or int(args.eval_obs_chunk) <= 0 or int(args.eval_src_chunk) <= 0:
            raise ValueError("chunk sizes must be positive")

        z, meta = c3.load_forcing(forcing_path)
        c3.validate_forcing_metadata(meta)
        if str(meta.get("mode")) != "formal" or int(meta.get("sources_per_arm", -1)) != SOURCE_COUNT_PER_ARM:
            raise RuntimeError("V2D requires the original formal 900-sources/arm forcing dataset")
        if abs(float(meta.get("runtime_source_softening_m", 0.0)) - SOFTENING_M) > 1e-18:
            raise RuntimeError("runtime_source_softening_m must be zero")
        if int(meta.get("num_collocation_points", -1)) != 3600:
            raise RuntimeError("unexpected formal collocation count; refusing to guess a different forcing contract")
        if abs(float(c2.LAMBDA_REL) - LAMBDA_REL) > 1e-18:
            raise RuntimeError("Step4C2-V4 lambda_rel differs from frozen V2D value")
        if abs(float(c3.DEPTH_FRAC) - DEPTH_FRAC) > 1e-18:
            raise RuntimeError("Step4C3 source depth differs from frozen V2D value")

        keys = (
            "mode", "total_length_m", "sources_per_arm", "n_phi", "n_cyl",
            "n_hemi", "cycles", "analysis_start_cycles", "diag_stride",
            "holdout_points_target",
        )
        settings = {k: meta[k] for k in keys}
        settings["sources_per_arm"] = SOURCE_COUNT_PER_ARM
        g0 = c3.build_geometry(settings)
        if c3.geometry_fingerprint(g0) != meta["geometry_fingerprint"]:
            raise RuntimeError("forcing / reconstructed 900-case geometry fingerprint mismatch")

        Hpoints = np.asarray(z["holdout_points_m"], float)
        Hnormals = np.asarray(z["holdout_normals"], float)
        Hmask = np.asarray(z["holdout_mask_matrix"], np.uint8).astype(bool)
        region_names = list(meta["holdout_region_names"])
        colloc_fp, holdout_fp = fixed_observation_fingerprints(g0, Hpoints, Hnormals, Hmask, region_names)

        forcing_sha = sha256_file(forcing_path)
        report["forcing_sha256"] = forcing_sha
        report["forcing_metadata_geometry_fingerprint"] = str(meta["geometry_fingerprint"])
        report["fixed_observations"] = {
            "status": "complete",
            "collocation_fingerprint": colloc_fp,
            "holdout_fingerprint": holdout_fp,
            "collocation_point_count": int(g0.num_collocation_points),
            "holdout_point_count": int(len(Hpoints)),
            "region_names": region_names,
        }

        fit_coll, fit_hold, fit_diag = fit_synchronized_phasors(z, meta, g0, Hpoints)
        report["phasor_fit"] = {"status": "complete", **fit_diag}

        candidate_ref = source_candidate_reference(settings)
        if SOURCE_COUNT_PER_ARM > candidate_ref["candidate_points_per_arm"]:
            raise RuntimeError("design blocked: 900 sources/arm exceeds candidate point count")
        report["source_candidate"] = {
            "status": "complete",
            "candidate_points_per_arm": candidate_ref["candidate_points_per_arm"],
            "candidate_points_total": candidate_ref["candidate_points_total"],
            "candidate_source_ratio_per_arm": float(candidate_ref["candidate_points_per_arm"] / SOURCE_COUNT_PER_ARM),
            "v2ca_candidate_geometry_fingerprint": candidate_ref["v2ca_candidate_geometry_fingerprint"],
            "expanded_candidate_geometry_fingerprint": candidate_ref["expanded_candidate_geometry_fingerprint"],
            "candidate_surface_resolution": {
                "n_phi": int(settings["n_phi"]),
                "n_cyl": int(settings["n_cyl"]),
                "n_hemi": int(settings["n_hemi"]),
            },
            "physical_strata_area_m2_per_arm": candidate_ref["zone_areas"],
            "physical_strata_candidate_counts_per_arm": candidate_ref["zone_capacities"],
        }

        v2ca = load_v2ca_reference(v2ca_path)
        report["v2c_a_reference_report_sha256"] = sha256_file(v2ca_path)
        if str(v2ca.get("forcing_sha256")) != forcing_sha:
            raise RuntimeError("V2C-A report was generated from a different forcing file")
        v2ca_fixed = v2ca.get("fixed_observations", {})
        if str(v2ca_fixed.get("collocation_fingerprint")) != colloc_fp:
            raise RuntimeError("V2C-A collocation fingerprint differs from V2D frozen observations")
        if str(v2ca_fixed.get("holdout_fingerprint")) != holdout_fp:
            raise RuntimeError("V2C-A holdout fingerprint differs from V2D frozen observations")
        ref_fit = v2ca.get("phasor_fit", {})
        if int(ref_fit.get("sample_count", -1)) != int(fit_diag["sample_count"]):
            raise RuntimeError("V2C-A synchronized phasor sample count differs from V2D")
        if [int(x) for x in ref_fit.get("diagnostic_steps", [])] != [int(x) for x in fit_diag["diagnostic_steps"]]:
            raise RuntimeError("V2C-A synchronized diagnostic steps differ from V2D")
        # Do not hard-gate on a bytewise phasor hash across independent BLAS/LAPACK
        # runs: least-squares output can differ in last bits while the estimator,
        # times, and physical metrics are unchanged. The formal baseline solve below
        # must reproduce every V2C-A 900-source regional gate within BASELINE_REPRO_TOL.
        phasor_fp_match = (
            str(ref_fit.get("phasor_fingerprint")) == str(fit_diag["phasor_fingerprint"])
        )
        if str(v2ca.get("source_candidate", {}).get("candidate_geometry_fingerprint")) != str(candidate_ref["v2ca_candidate_geometry_fingerprint"]):
            raise RuntimeError("V2C-A candidate source surface fingerprint differs from V2D")
        report["upstream_precondition"] = {
            "status": "complete",
            "v2c_a_branch": str(v2ca["decision_tree"]["branch"]),
            "v2c_a_minimum_all_frequency_qualified_density": str(v2ca["decision_tree"]["minimum_all_frequency_qualified_density"]),
            "forcing_sha256_match": True,
            "collocation_fingerprint_match": True,
            "holdout_fingerprint_match": True,
            "phasor_fingerprint_match_advisory": bool(phasor_fp_match),
            "phasor_equivalence_gate": "same 22 diagnostic steps plus formal 900-source baseline metric reproduction within tolerance",
            "candidate_surface_fingerprint_match": True,
        }

        fixed_cloud = (
            np.asarray(g0.solve_obs_points, float).copy(),
            np.asarray(g0.solve_normals, float).copy(),
            np.asarray(g0.solve_weights, float).copy(),
            np.asarray(g0.solve_tangent1, float).copy(),
            np.asarray(g0.solve_tangent2, float).copy(),
        )
        baseline_indices = {
            arm: selection_indices(candidate_ref, arm, PLACEMENT_RULES[0], SOURCE_COUNT_PER_ARM)[0]
            for arm in (0, 1)
        }

        report["status"] = "running"
        for rule in PLACEMENT_RULES:
            current_rule = rule
            print(f"[Step 4C3 V2D] building placement={rule}, {SOURCE_COUNT_PER_ARM} sources/arm ...", flush=True)
            g = build_geometry_for_rule(settings, candidate_ref, fixed_cloud, rule, SOURCE_COUNT_PER_ARM)
            if fixed_observation_fingerprints(g, Hpoints, Hnormals, Hmask, region_names) != (colloc_fp, holdout_fp):
                raise RuntimeError(f"fixed observation fingerprint changed for placement rule {rule}")
            if rule == PLACEMENT_RULES[0] and c3.geometry_fingerprint(g) != meta["geometry_fingerprint"]:
                raise RuntimeError("baseline placement no longer matches forcing geometry fingerprint")

            geom_diag = source_cloud_diagnostics(
                g, candidate_ref, rule, SOURCE_COUNT_PER_ARM, baseline_indices
            )
            min_collocation_distance = float(np.min(cKDTree(g.ant_src_points).query(g.solve_obs_points, k=1)[0]))
            min_holdout_distance = float(np.min(cKDTree(g.ant_src_points).query(Hpoints, k=1)[0]))
            evaluator_guard = 1.0e-15
            if min(min_collocation_distance, min_holdout_distance) <= evaluator_guard:
                raise RuntimeError(
                    "pairwise evaluator singularity guard would become active; zero-softening contract violated"
                )
            geom_diag["min_source_to_fixed_collocation_distance_m"] = min_collocation_distance
            geom_diag["min_source_to_fixed_holdout_distance_m"] = min_holdout_distance
            geom_diag["pairwise_evaluator_zero_distance_guard_m"] = evaluator_guard
            geom_diag["pairwise_evaluator_guard_active"] = False

            pr = {
                "status": "running",
                "placement_rule": rule,
                "fixed_collocation_point_count_used": int(g.num_collocation_points),
                "fixed_collocation_fingerprint": colloc_fp,
                "fixed_holdout_fingerprint": holdout_fp,
                "source_geometry": geom_diag,
                "geometry_fingerprint_with_fixed_observations": c3.geometry_fingerprint(g),
                "frequency_results": {},
            }
            report["placements"][rule] = pr

            for f in FREQS:
                fk = freq_key(f)
                omega = 2.0 * np.pi * float(f)
                print(
                    f"[Step 4C3 V2D] placement={rule}, f={f/1e6:.0f} MHz: assemble/solve/evaluate ...",
                    flush=True,
                )
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
                if rule == PLACEMENT_RULES[0]:
                    fr["v2c_a_900_baseline_reproduction"] = v2ca_baseline_reproduction_check(
                        v2ca, f, coll_rel, regions, worst
                    )
                else:
                    fr["v2c_a_900_baseline_reproduction"] = {
                        "applicable": False,
                        "reason": "alternative placement intentionally changes source indices at fixed N=900",
                    }
                pr["frequency_results"][fk] = fr

                del A, q, Escat, total_hold
                gc.collect()

            pr["status"] = "complete"
            report["placements"][rule] = pr

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
        if current_rule != "none" and current_rule in report["placements"]:
            pr = report["placements"][current_rule]
            if isinstance(pr, dict) and pr.get("status") != "complete":
                pr["status"] = "failed"
                pr["failure_reason"] = str(exc)
        failed_seen = False
        for rule in PLACEMENT_RULES:
            pr = report["placements"][rule]
            if pr.get("status") == "failed":
                failed_seen = True
            elif failed_seen and str(pr.get("status", "")).startswith("not_run"):
                pr["status"] = "not_run_due_to_prior_failure"
        finalize_outputs(out, report)
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
    g0 = c3.build_geometry(settings)
    fixed_cloud = (
        g0.solve_obs_points.copy(),
        g0.solve_normals.copy(),
        g0.solve_weights.copy(),
        g0.solve_tangent1.copy(),
        g0.solve_tangent2.copy(),
    )
    fixed_fp = hash_arrays_exact(*fixed_cloud)
    candidate_ref = source_candidate_reference(settings)
    baseline_indices = {
        arm: selection_indices(candidate_ref, arm, PLACEMENT_RULES[0], 24)[0]
        for arm in (0, 1)
    }

    determinism = {}
    geometry_fps_differs = False
    for rule in PLACEMENT_RULES:
        for arm in (0, 1):
            i1, m1 = selection_indices(candidate_ref, arm, rule, 24)
            i2, m2 = selection_indices(candidate_ref, arm, rule, 24)
            if not np.array_equal(i1, i2) or m1 != m2:
                raise RuntimeError(f"self-test nondeterministic placement selection: {rule}, arm {arm}")
            if len(i1) != 24 or len(np.unique(i1)) != 24:
                raise RuntimeError(f"self-test placement count/uniqueness failed: {rule}, arm {arm}")
        g = build_geometry_for_rule(settings, candidate_ref, fixed_cloud, rule, 24)
        gfp = hash_arrays_exact(
            g.solve_obs_points, g.solve_normals, g.solve_weights,
            g.solve_tangent1, g.solve_tangent2,
        )
        if gfp != fixed_fp:
            raise RuntimeError(f"self-test fixed observations changed for {rule}")
        diag = source_cloud_diagnostics(g, candidate_ref, rule, 24, baseline_indices)
        determinism[rule] = {
            "source_count": int(g.num_sources),
            "fixed_observations_unchanged": True,
            "selected_source_index_hash": diag["selected_source_index_hash"],
        }
        if rule != PLACEMENT_RULES[0]:
            geometry_fps_differs = geometry_fps_differs or (
                diag["baseline_overlap_fraction_total"] < 1.0
            )

    if not geometry_fps_differs:
        raise RuntimeError("self-test alternative placement rules did not change any selected source index")

    # Real retarded harmonic assembly vs independent pairwise evaluator on the
    # physical-anchor alternative, followed by the weighted/scaled solve chain.
    galt = build_geometry_for_rule(
        settings, candidate_ref, fixed_cloud, "physical_anchor_fps", 24
    )
    omega = 2.0 * np.pi * 3.0e6
    A, _ = c2.assemble_retarded_harmonic(galt, omega, chunk=8)
    rng = np.random.default_rng(20261003)
    q_true = (rng.normal(size=A.shape[1]) + 1j * rng.normal(size=A.shape[1])) * 1.0e-6
    Epair = c2.eval_retarded_field(
        galt.solve_obs_points,
        galt,
        q_true,
        omega,
        obs_chunk=16,
        src_chunk=16,
    )
    pair_coeff = v2b.field_to_coeff(Epair, galt.solve_tangent1, galt.solve_tangent2)
    matrix_coeff = A @ q_true
    op_pair_rel = float(
        np.linalg.norm(pair_coeff - matrix_coeff)
        / max(float(np.linalg.norm(matrix_coeff)), TINY)
    )
    if op_pair_rel > OPERATOR_EVALUATOR_TOL:
        raise RuntimeError(f"self-test operator/evaluator mismatch: {op_pair_rel}")

    incident = -matrix_coeff
    prep = c2.prepare_complex(A.copy(), galt.solve_weights)
    q_fit = c2.solve_complex(prep, -incident)
    solve_rel = float(
        np.linalg.norm(A @ q_fit + incident)
        / max(float(np.linalg.norm(incident)), TINY)
    )
    if solve_rel >= 5.0e-4:
        raise RuntimeError(f"self-test weighted/scaled solve residual too large: {solve_rel}")
    eval_audit = pairwise_operator_audit(A, galt, q_fit, omega, obs_count=8)

    synthetic_report = {
        "self_test_pass": True,
        "placement_rules": list(PLACEMENT_RULES),
        "deterministic_geometry_only_selection": True,
        "alternative_selection_changes_indices": geometry_fps_differs,
        "fixed_observations_unchanged": True,
        "placement_checks": determinism,
        "operator_pairwise_relative_l2": op_pair_rel,
        "known_coefficient_solve_relative_l2": solve_rel,
        "solve_evaluate_chain_relative_l2": eval_audit["relative_l2_error"],
        "softening_m": SOFTENING_M,
        "lambda_rel": LAMBDA_REL,
        "sources_per_arm_formal": SOURCE_COUNT_PER_ARM,
        "frequencies_Hz": [int(f) for f in FREQS],
        "boundary_gate": BOUNDARY_GATE,
        "selection_data_dependency": "geometry_only_no_forcing_no_holdout_labels_no_residuals",
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
        "predeclared_candidate_rule": report["decision_tree"]["predeclared_candidate_rule"],
        "report": str(resolve_out_path(args.out) / "step4c3_v2d_report.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
