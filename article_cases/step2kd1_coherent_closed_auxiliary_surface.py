#!/usr/bin/env python3
"""
Step 2-KD1: coherent closed auxiliary-surface isolation test.

Purpose
-------
Isolate the root cause found after Step 2-KD without modifying production code.
On the SAME closed physical cylinder, source anchors, weights, mixed-basis rule,
collocation clouds, incident fields, regularization, and independent hold-out,
compare:

A. independent_normal
   The Step-2KD construction: every selected physical anchor is shifted inward
   independently by d along its local physical normal.

B. coherent_inner_cylinder
   A single closed inner cylinder with
       R_s = R - d,
       z in [d, L-d],
   and cap disks of radius R_s.  The same source anchors are mapped one-to-one
   to this coherent inner surface so source count, weights, edge classification,
   and basis assignment stay fixed.

Diagnostics explicitly use ACTUAL auxiliary source positions, not physical
anchors.  They compare:
  * nearest-neighbor spacing / d,
  * side-cap cross-feature separation / d,
  * source-to-physical-PEC distance distribution / d,
  * weighted operator column-norm spread,
  * column-scaled spectral conditioning (dense SVD by default when feasible),
  * LSMR convergence and residual diagnostics,
  * independent global / edge / smooth / far PEC hold-out error.

This is a controlled diagnostic only.  It does NOT modify production modules.
Production integration is gated separately and requires the coherent geometry,
numerics, and global/edge/smooth hold-out all to pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.linalg import ArpackNoConvergence, lsmr, svds

EPS0 = 8.854187817e-12
C0 = 299792458.0
TINY = 1.0e-30
ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="step2kd1_coherent_closed_auxiliary_surface")
    p.add_argument("--radius", type=float, default=1.0e-3)
    p.add_argument("--arm-length", type=float, default=29.0e-3)
    p.add_argument("--source-offset-frac", type=float, default=0.8)
    p.add_argument("--edge-radius-over-d", type=float, default=1.5)
    p.add_argument("--smooth-h-over-d", type=float, default=0.65)
    p.add_argument("--edge-h-over-d", type=float, default=0.35)
    p.add_argument("--sharp-angle-deg", type=float, default=35.0)
    p.add_argument("--lambda-rel", type=float, default=1.0e-4)
    p.add_argument("--dt", type=float, default=1.0e-9)
    p.add_argument("--max-lsmr-iter", type=int, default=5000)
    p.add_argument("--rank-rtol", type=float, default=1.0e-10)
    p.add_argument(
        "--spectral-mode",
        choices=("auto", "dense", "svds", "none"),
        default="auto",
        help=(
            "Conditioning diagnostic for the weighted+column-scaled operator. "
            "auto uses dense SVD for matrices up to --dense-svd-max-unknowns "
            "and svds otherwise."
        ),
    )
    p.add_argument("--dense-svd-max-unknowns", type=int, default=2400)
    return p.parse_args()


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, TINY)


def tri_geom(T):
    e1 = T[:, 1] - T[:, 0]
    e2 = T[:, 2] - T[:, 0]
    cr = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    normal = unit(cr)
    cent = np.mean(T, axis=1)
    return cent, normal, area


def add_tri(tris, a, b, c):
    tris.append(np.asarray([a, b, c], dtype=np.float64))


def generate_closed_cylinder(radius, length, nz, nphi, nr):
    """Outward-oriented triangulated closed cylinder, z in [0,length]."""
    tris = []
    ph = 2 * np.pi * np.arange(nphi) / nphi

    z = np.linspace(0.0, length, nz + 1)
    for iz in range(nz):
        z0, z1 = z[iz], z[iz + 1]
        for j in range(nphi):
            k = (j + 1) % nphi
            p00 = np.array([radius * np.cos(ph[j]), radius * np.sin(ph[j]), z0])
            p10 = np.array([radius * np.cos(ph[k]), radius * np.sin(ph[k]), z0])
            p01 = np.array([radius * np.cos(ph[j]), radius * np.sin(ph[j]), z1])
            p11 = np.array([radius * np.cos(ph[k]), radius * np.sin(ph[k]), z1])
            add_tri(tris, p00, p10, p11)
            add_tri(tris, p00, p11, p01)

    rs = np.linspace(0.0, radius, nr + 1)
    for top in (False, True):
        zz = length if top else 0.0
        sign = 1.0 if top else -1.0
        c = np.array([0.0, 0.0, zz])
        r1 = rs[1]
        for j in range(nphi):
            k = (j + 1) % nphi
            aa = np.array([r1 * np.cos(ph[j]), r1 * np.sin(ph[j]), zz])
            bb = np.array([r1 * np.cos(ph[k]), r1 * np.sin(ph[k]), zz])
            if sign > 0:
                add_tri(tris, c, aa, bb)
            else:
                add_tri(tris, c, bb, aa)
        for ir in range(1, nr):
            ra, rb = rs[ir], rs[ir + 1]
            for j in range(nphi):
                k = (j + 1) % nphi
                a0 = np.array([ra * np.cos(ph[j]), ra * np.sin(ph[j]), zz])
                a1 = np.array([ra * np.cos(ph[k]), ra * np.sin(ph[k]), zz])
                b0 = np.array([rb * np.cos(ph[j]), rb * np.sin(ph[j]), zz])
                b1 = np.array([rb * np.cos(ph[k]), rb * np.sin(ph[k]), zz])
                if sign > 0:
                    add_tri(tris, a0, b0, b1)
                    add_tri(tris, a0, b1, a1)
                else:
                    add_tri(tris, a0, b1, b0)
                    add_tri(tris, a0, a1, b1)

    return np.asarray(tris, dtype=np.float64)


def write_ascii_stl(path, T):
    with open(path, "w", encoding="utf-8") as f:
        f.write("solid step2kd1_closed_arm\n")
        _, n, _ = tri_geom(T)
        for tri, nn in zip(T, n):
            f.write(f"  facet normal {nn[0]:.12e} {nn[1]:.12e} {nn[2]:.12e}\n")
            f.write("    outer loop\n")
            for v in tri:
                f.write(f"      vertex {v[0]:.12e} {v[1]:.12e} {v[2]:.12e}\n")
            f.write("    endloop\n  endfacet\n")
        f.write("endsolid step2kd1_closed_arm\n")


def vkey(v, scale=1e12):
    return tuple(np.round(np.asarray(v) * scale).astype(np.int64).tolist())


def detect_sharp_edges(T, angle_deg):
    _, nrm, _ = tri_geom(T)
    emap = defaultdict(list)
    ecoord = {}
    for it, tri in enumerate(T):
        for aa, bb in ((0, 1), (1, 2), (2, 0)):
            ka, kb = vkey(tri[aa]), vkey(tri[bb])
            key = (ka, kb) if ka < kb else (kb, ka)
            emap[key].append(it)
            ecoord[key] = (tri[aa].copy(), tri[bb].copy())

    sharp = []
    nonmanifold = 0
    boundary = 0
    cthr = math.cos(math.radians(angle_deg))
    for key, adj in emap.items():
        if len(adj) == 1:
            boundary += 1
            sharp.append(ecoord[key])
        elif len(adj) == 2:
            c = float(np.dot(nrm[adj[0]], nrm[adj[1]]))
            if c < cthr:
                sharp.append(ecoord[key])
        else:
            nonmanifold += 1
            sharp.append(ecoord[key])
    return np.asarray(sharp, dtype=np.float64), {
        "num_unique_edges": len(emap),
        "num_sharp_edges": len(sharp),
        "num_boundary_edges": boundary,
        "num_nonmanifold_edges": nonmanifold,
    }


def point_segment_distances(P, segments, chunk=2048):
    if len(segments) == 0:
        return np.full(len(P), np.inf)
    A = segments[:, 0]
    B = segments[:, 1]
    AB = B - A
    den = np.sum(AB * AB, axis=1)
    out = np.empty(len(P), float)
    for i in range(0, len(P), chunk):
        X = P[i : i + chunk]
        AP = X[:, None, :] - A[None, :, :]
        t = np.sum(AP * AB[None, :, :], axis=2) / np.maximum(den[None, :], TINY)
        t = np.clip(t, 0.0, 1.0)
        Q = A[None, :, :] + t[:, :, None] * AB[None, :, :]
        d2 = np.sum((X[:, None, :] - Q) ** 2, axis=2)
        out[i : i + chunk] = np.sqrt(np.min(d2, axis=1))
    return out


def fps_indices(P, n, seed=0):
    P = np.asarray(P, float)
    N = len(P)
    n = min(int(n), N)
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    start = int(seed % N)
    chosen = np.empty(n, dtype=np.int64)
    chosen[0] = start
    d2 = np.sum((P - P[start]) ** 2, axis=1)
    d2[start] = -1.0
    for k in range(1, n):
        j = int(np.argmax(d2))
        chosen[k] = j
        nd2 = np.sum((P - P[j]) ** 2, axis=1)
        d2 = np.minimum(d2, nd2)
        d2[chosen[: k + 1]] = -1.0
    return chosen


def select_regionwise(cent, area, edge_mask, h_smooth, h_edge, seed=0):
    sel = []
    budgets = {}
    for name, mask, h, ofs in (
        ("edge", edge_mask, h_edge, 17),
        ("smooth", ~edge_mask, h_smooth, 53),
    ):
        idx = np.where(mask)[0]
        A = float(np.sum(area[idx]))
        n = max(1, int(math.ceil(A / max(h * h, TINY))))
        loc = fps_indices(cent[idx], n, seed + ofs)
        picked = idx[loc]
        sel.extend(picked.tolist())
        budgets[name] = {"area": A, "target_h": h, "budget": len(picked)}
    return np.asarray(sel, dtype=np.int64), budgets


def voronoi_weights(candidate_cent, candidate_area, selected_cent):
    tree = cKDTree(selected_cent)
    _, lab = tree.query(candidate_cent, k=1)
    w = np.bincount(lab, weights=candidate_area, minlength=len(selected_cent)).astype(float)
    return w, lab


def cloud_quality(candidate_cent, edge_dist, selected_cent, d):
    tree = cKDTree(selected_cent)
    dd, _ = tree.query(candidate_cent, k=1)
    snn = cKDTree(selected_cent).query(selected_cent, k=2)[0][:, 1]
    q = {
        "anchor_nn_min_over_d": float(np.min(snn) / d),
        "anchor_nn_p50_over_d": float(np.median(snn) / d),
        "cover_p95_over_d": float(np.quantile(dd, 0.95) / d),
        "cover_max_over_d": float(np.max(dd) / d),
    }
    for name, mask in (("edge", edge_dist <= 1.5 * d), ("smooth", edge_dist > 1.5 * d)):
        if np.any(mask):
            q[f"{name}_cover_p95_over_d"] = float(np.quantile(dd[mask], 0.95) / d)
            q[f"{name}_cover_max_over_d"] = float(np.max(dd[mask]) / d)
    return q


def frames(n):
    n = np.asarray(n, float).reshape(-1, 3)
    t1 = np.zeros_like(n)
    t2 = np.zeros_like(n)
    for i, v in enumerate(n):
        axes = np.eye(3)
        h = axes[np.argmin(np.abs(axes @ v))]
        aa = h - v * np.dot(h, v)
        aa /= max(np.linalg.norm(aa), TINY)
        bb = np.cross(v, aa)
        bb /= max(np.linalg.norm(bb), TINY)
        t1[i] = aa
        t2[i] = bb
    return t1, t2


def field_block(obs, src, w, dt):
    r = np.asarray(obs) - np.asarray(src)[None, :]
    R = np.maximum(np.linalg.norm(r, axis=1), 1e-15)
    rr = r[:, :, None] * r[:, None, :]
    I = np.eye(3)[None, :, :]
    ce = 1 / (4 * np.pi * EPS0)
    Gp = ce * w * (3 * rr / R[:, None, None] ** 5 - I / R[:, None, None] ** 3)
    Gk = ce * w * (3 * rr / R[:, None, None] ** 4 - I / R[:, None, None] ** 2) / C0
    Gd = ce * w * (rr / R[:, None, None] ** 3 - I / R[:, None, None]) / (C0 * C0)
    return 0.5 * dt * Gp + Gk + Gd / dt


def build_modes(edge_dist, edge_radius):
    return np.where(edge_dist <= edge_radius, 3, 2).astype(np.int32)


def offsets(modes):
    off = np.zeros(len(modes) + 1, dtype=np.int64)
    off[1:] = np.cumsum(modes)
    return off


def assemble(OP, ON, SP, SN, SW, modes, off, dt):
    ot1, ot2 = frames(ON)
    st1, st2 = frames(SN)
    A = np.empty((2 * len(OP), int(off[-1])), float)
    for j in range(len(SP)):
        B = field_block(OP, SP[j], SW[j], dt)
        S = np.eye(3) if modes[j] == 3 else np.column_stack((st1[j], st2[j]))
        T = np.einsum("mab,bc->mac", B, S)
        aa, bb = int(off[j]), int(off[j + 1])
        A[0::2, aa:bb] = np.einsum("ma,mac->mc", ot1, T)
        A[1::2, aa:bb] = np.einsum("ma,mac->mc", ot2, T)
    return A


def tangent_rhs(E, N):
    t1, t2 = frames(N)
    b = np.empty(2 * len(N), float)
    b[0::2] = -np.sum(E * t1, axis=1)
    b[1::2] = -np.sum(E * t2, axis=1)
    return b


def coeff_to_K(x, SN, modes, off):
    st1, st2 = frames(SN)
    K = np.zeros((len(SN), 3), float)
    for j, m in enumerate(modes):
        aa, bb = int(off[j]), int(off[j + 1])
        q = x[aa:bb]
        K[j] = q if m == 3 else q[0] * st1[j] + q[1] * st2[j]
    return K


def eval_field(P, SP, SN, SW, x, modes, off, dt, obs_chunk=256, src_chunk=64):
    """Vectorized field evaluation, algebraically identical to field_block @ K."""
    K = coeff_to_K(x, SN, modes, off)
    P = np.asarray(P, float)
    SP = np.asarray(SP, float)
    SW = np.asarray(SW, float)
    E = np.zeros((len(P), 3), float)
    ce = 1.0 / (4.0 * np.pi * EPS0)

    for oa in range(0, len(P), obs_chunk):
        ob = min(len(P), oa + obs_chunk)
        X = P[oa:ob]
        Eb = np.zeros((len(X), 3), float)
        for sa in range(0, len(SP), src_chunk):
            sb = min(len(SP), sa + src_chunk)
            r = X[:, None, :] - SP[None, sa:sb, :]
            R = np.maximum(np.linalg.norm(r, axis=2), 1e-15)
            Ks = K[sa:sb]
            rk = np.sum(r * Ks[None, :, :], axis=2)
            w = SW[None, sa:sb, None]

            gp = 3.0 * r * rk[:, :, None] / R[:, :, None] ** 5 - Ks[None, :, :] / R[:, :, None] ** 3
            gk = 3.0 * r * rk[:, :, None] / R[:, :, None] ** 4 - Ks[None, :, :] / R[:, :, None] ** 2
            gd = r * rk[:, :, None] / R[:, :, None] ** 3 - Ks[None, :, :] / R[:, :, None]

            contrib = ce * w * (0.5 * dt * gp + gk / C0 + gd / (C0 * C0 * dt))
            Eb += np.sum(contrib, axis=1)
        E[oa:ob] = Eb
    return E, K


def estimate_smax(A, niter=8):
    rng = np.random.default_rng(1234)
    x = rng.normal(size=A.shape[1])
    x /= max(np.linalg.norm(x), TINY)
    for _ in range(niter):
        y = A @ x
        z = A.T @ y
        nz = np.linalg.norm(z)
        if nz < TINY:
            return 1.0
        x = z / nz
    return float(np.linalg.norm(A @ x))


def prepare_scaled_operator(A, row_point_weights):
    rs = np.repeat(np.sqrt(row_point_weights / np.mean(row_point_weights)), 2)
    Aw = A * rs[:, None]
    cs = np.linalg.norm(Aw, axis=0)
    cs = np.where(cs > TINY, cs, 1.0)
    As = Aw / cs[None, :]
    return rs, Aw, cs, As


def solve_scaled_lsmr_prepared(As, cs, rs, b, lam, maxiter):
    bw = b * rs
    smax = estimate_smax(As)
    damp = lam * smax
    sol = lsmr(As, bw, damp=damp, atol=1e-10, btol=1e-10, conlim=1e10, maxiter=maxiter)
    y = sol[0]
    x = y / cs
    r = As @ y - bw
    nr = np.linalg.norm(r)
    atr = As.T @ r
    grad = atr + (damp * damp) * y
    an = np.linalg.norm(As, ord="fro")
    undamped_ne = float(np.linalg.norm(atr) / max(an * nr, TINY))
    damped_scale = an * nr + (damp * damp) * np.linalg.norm(y)
    damped_opt = float(np.linalg.norm(grad) / max(damped_scale, TINY))
    diag = {
        "istop": int(sol[1]),
        "iterations": int(sol[2]),
        "acond_est": float(sol[6]),
        "weighted_residual_rel": float(nr / max(np.linalg.norm(bw), TINY)),
        "undamped_normal_eq_rel": undamped_ne,
        "damped_optimality_rel": damped_opt,
        # Backward-compatible name for comparison with Step 2-KD.
        "normal_eq_rel": undamped_ne,
        "damp": float(damp),
        "smax_scaled_est": float(smax),
    }
    return x, diag


def direct_scaled_tikhonov(U, s, Vt, As, cs, rs, b, lam):
    """Direct SVD solve of the same weighted, column-scaled damped objective."""
    bw = b * rs
    smax_est = estimate_smax(As)
    damp = lam * smax_est
    filt = s / (s * s + damp * damp)
    y = Vt.T @ (filt * (U.T @ bw))
    x = y / cs
    r = As @ y - bw
    grad = As.T @ r + (damp * damp) * y
    an = np.linalg.norm(As, ord="fro")
    nr = np.linalg.norm(r)
    scale = an * nr + (damp * damp) * np.linalg.norm(y)
    diag = {
        "weighted_residual_rel": float(nr / max(np.linalg.norm(bw), TINY)),
        "damped_optimality_rel": float(np.linalg.norm(grad) / max(scale, TINY)),
        "damp": float(damp),
        "smax_scaled_est": float(smax_est),
    }
    return x, diag


def inc_uniform(P):
    E = np.empty((len(P), 3), float)
    E[:] = [1.0, 0.35, 0.2]
    return E


def inc_dip(P, L, R):
    r0 = np.array([4.0 * R, 3.0 * R, 0.42 * L])
    p = np.array([1.0e-12, -0.45e-12, 0.75e-12])
    r = P - r0
    rr = np.maximum(np.linalg.norm(r, axis=1), 1e-15)
    h = r / rr[:, None]
    return (1 / (4 * np.pi * EPS0)) * (
        3 * h * np.sum(h * p, axis=1)[:, None] - p
    ) / rr[:, None] ** 3


def tangential(E, N):
    return E - np.sum(E * N, axis=1)[:, None] * N


def vrms(E):
    return float(np.sqrt(np.mean(np.sum(np.abs(E) ** 2, axis=1))))


def relerr(total, inc, N, mask):
    return vrms(tangential(total, N)[mask]) / max(vrms(tangential(inc, N)[mask]), TINY)


def make_holdout_points(T):
    w = np.array([0.21, 0.33, 0.46])
    P = np.einsum("k,nkj->nj", w, T)
    _, N, A = tri_geom(T)
    return P, N, A


def source_feature_masks(SN):
    cap = np.abs(SN[:, 2]) > 0.9
    side = ~cap
    return side, cap


def independent_normal_sources(anchor, normals, d):
    return anchor - d * normals


def coherent_inner_cylinder_sources(anchor, normals, R, L, d):
    """
    Map the SAME physical source anchors one-to-one to one coherent inner cylinder.

    Side anchors retain azimuth but their axial parameter is contracted from
    [0,L] to [d,L-d].  Cap anchors retain normalized radial coordinate r/R,
    contracted to radius R-d, and are placed on z=d or z=L-d.
    """
    Rs = R - d
    if Rs <= 0.0:
        raise ValueError("coherent inner cylinder requires source offset d < radius R")
    if L <= 2.0 * d:
        raise ValueError("coherent inner cylinder requires L > 2d")

    P = np.zeros_like(anchor, dtype=float)
    side, cap = source_feature_masks(normals)

    rho = np.linalg.norm(anchor[:, :2], axis=1)
    er = np.zeros((len(anchor), 2), float)
    nz = rho > TINY
    er[nz] = anchor[nz, :2] / rho[nz, None]
    # Side centroids always have rho>0 for this cylinder.  The fallback keeps
    # the mapping defined if a future geometry produces a degenerate point.
    er[~nz] = np.array([1.0, 0.0])

    zeta = np.clip(anchor[:, 2] / max(L, TINY), 0.0, 1.0)
    P[side, :2] = Rs * er[side]
    P[side, 2] = d + zeta[side] * (L - 2.0 * d)

    cap_scale = Rs / R
    P[cap, :2] = cap_scale * anchor[cap, :2]
    bottom = cap & (normals[:, 2] < 0.0)
    top = cap & (normals[:, 2] > 0.0)
    P[bottom, 2] = d
    P[top, 2] = L - d
    return P


def physical_cylinder_boundary_distance(P, R, L):
    """Exact nearest distance to the boundary for points inside the finite cylinder."""
    rho = np.linalg.norm(P[:, :2], axis=1)
    inside = (rho <= R + 1e-12 * max(R, 1.0)) & (P[:, 2] >= -1e-12) & (P[:, 2] <= L + 1e-12)
    if not np.all(inside):
        raise ValueError("source point outside physical cylinder in KD1 diagnostic")
    return np.minimum.reduce((R - rho, P[:, 2], L - P[:, 2]))


def qstats(x, scale=1.0):
    x = np.asarray(x, float) / scale
    return {
        "min": float(np.min(x)),
        "p01": float(np.quantile(x, 0.01)),
        "p05": float(np.quantile(x, 0.05)),
        "p50": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
    }


def source_position_diagnostics(SP, SN, R, L, d):
    side, cap = source_feature_masks(SN)
    dd, ii = cKDTree(SP).query(SP, k=2)
    nn = dd[:, 1]
    nn_idx = ii[:, 1]

    if np.any(side) and np.any(cap):
        d_sc, j_sc = cKDTree(SP[cap]).query(SP[side], k=1)
        k = int(np.argmin(d_sc))
        side_ids = np.where(side)[0]
        cap_ids = np.where(cap)[0]
        cross_min = float(d_sc[k])
        cross_pair = [int(side_ids[k]), int(cap_ids[int(j_sc[k])])]
    else:
        cross_min = float("nan")
        cross_pair = [-1, -1]

    pec = physical_cylinder_boundary_distance(SP, R, L)
    g = int(np.argmin(nn))
    global_pair = [g, int(nn_idx[g])]
    global_pair_features = [
        "side" if side[global_pair[0]] else "cap",
        "side" if side[global_pair[1]] else "cap",
    ]
    return {
        "actual_nn_over_d": qstats(nn, d),
        "actual_nn_pair": global_pair,
        "actual_nn_pair_features": global_pair_features,
        "cross_feature_min_over_d": float(cross_min / d),
        "cross_feature_pair": cross_pair,
        "source_to_pec_over_d": qstats(pec, d),
        "nearest_neighbor_index": nn_idx,
        "nearest_neighbor_distance": nn,
        "source_to_pec_distance": pec,
        "side_mask": side,
        "cap_mask": cap,
    }


def spread_stats(x):
    x = np.asarray(x, float)
    xp = x[x > TINY]
    if len(xp) == 0:
        return {"min": 0.0, "p01": 0.0, "p50": 0.0, "p99": 0.0, "max": 0.0, "max_over_min": float("inf"), "p99_over_p01": float("inf")}
    mn = float(np.min(xp))
    mx = float(np.max(xp))
    p01 = float(np.quantile(xp, 0.01))
    p50 = float(np.median(xp))
    p99 = float(np.quantile(xp, 0.99))
    return {
        "min": mn,
        "p01": p01,
        "p50": p50,
        "p99": p99,
        "max": mx,
        "max_over_min": float(mx / max(mn, TINY)),
        "p99_over_p01": float(p99 / max(p01, TINY)),
    }


def operator_column_diagnostics(cs, modes, off, SN):
    source_of_col = np.empty(int(off[-1]), dtype=np.int64)
    for j in range(len(modes)):
        source_of_col[int(off[j]) : int(off[j + 1])] = j
    side, cap = source_feature_masks(SN)
    groups = {
        "all": np.ones(len(cs), dtype=bool),
        "side": side[source_of_col],
        "cap": cap[source_of_col],
        "cartesian3": (modes == 3)[source_of_col],
        "tangent2": (modes == 2)[source_of_col],
        "smooth_tangent2": (modes == 2)[source_of_col],
    }
    return {name: spread_stats(cs[mask]) for name, mask in groups.items() if np.any(mask)}


def spectral_diagnostics(As, mode, rank_rtol, dense_max_unknowns):
    chosen = mode
    if chosen == "auto":
        chosen = "dense" if As.shape[1] <= dense_max_unknowns else "svds"
    if chosen == "none":
        return {
            "mode": "none",
            "condition_number": None,
            "sigma_max": None,
            "sigma_min": None,
            "effective_rank": None,
            "rank_fraction": None,
            "singular_values": None,
        }

    if chosen == "dense":
        U, s, Vt = np.linalg.svd(As, full_matrices=False)
        smax = float(s[0])
        smin = float(s[-1])
        rank = int(np.sum(s > rank_rtol * smax))
        return {
            "mode": "dense",
            "condition_number": float(smax / max(smin, TINY)),
            "sigma_max": smax,
            "sigma_min": smin,
            "effective_rank": rank,
            "rank_fraction": float(rank / len(s)),
            "singular_values": s,
            "_U": U,
            "_Vt": Vt,
        }

    try:
        smax = float(np.max(svds(As, k=1, which="LM", return_singular_vectors=False)))
        smin = float(np.min(svds(As, k=1, which="SM", return_singular_vectors=False)))
        return {
            "mode": "svds",
            "condition_number": float(smax / max(smin, TINY)),
            "sigma_max": smax,
            "sigma_min": smin,
            "effective_rank": None,
            "rank_fraction": None,
            "singular_values": None,
        }
    except (ArpackNoConvergence, ValueError) as exc:
        return {
            "mode": "svds_failed",
            "condition_number": None,
            "sigma_max": None,
            "sigma_min": None,
            "effective_rank": None,
            "rank_fraction": None,
            "singular_values": None,
            "error": repr(exc),
        }


def scalar_or_none(x):
    return None if x is None else float(x)


def clean_spectral_for_json(s):
    return {k: v for k, v in s.items() if k not in ("singular_values", "_U", "_Vt")}


def main():
    a = parse_args()
    out = ROOT / "article_cases_output" / a.out
    out.mkdir(parents=True, exist_ok=True)

    R = a.radius
    L = 8.0e-3 if a.quick else a.arm_length
    d = a.source_offset_frac * R
    edge_radius = a.edge_radius_over_d * d
    hs = a.smooth_h_over_d * d
    he = a.edge_h_over_d * d

    if not (0.0 < d < R):
        raise ValueError("KD1 coherent inner cylinder requires 0 < source offset d < R")
    if L <= 2.0 * d:
        raise ValueError("KD1 coherent inner cylinder requires L > 2d")

    if a.quick:
        nz, nphi, nr = 28, 40, 8
    else:
        nz, nphi, nr = 96, 72, 12

    T = generate_closed_cylinder(R, L, nz, nphi, nr)
    write_ascii_stl(out / "step2kd1_closed_arm.stl", T)

    C, N, Atri = tri_geom(T)
    total_area = float(np.sum(Atri))
    sharp, edge_info = detect_sharp_edges(T, a.sharp_angle_deg)
    edist = point_segment_distances(C, sharp)
    edge_mask = edist <= edge_radius

    sidx, budget = select_regionwise(C, Atri, edge_mask, hs, he, seed=11)
    Sanchor = C[sidx]
    Sn = N[sidx]
    Sedist = edist[sidx]
    SW, _ = voronoi_weights(C, Atri, Sanchor)
    modes = build_modes(Sedist, edge_radius)
    off = offsets(modes)

    coll_sets = []
    for seed in (101, 307):
        idx, _ = select_regionwise(C, Atri, edge_mask, hs, he, seed=seed)
        O = C[idx]
        ON = N[idx]
        OW, _ = voronoi_weights(C, Atri, O)
        coll_sets.append((O, ON, 0.5 * OW))
    OP = np.vstack([x[0] for x in coll_sets])
    ON = np.vstack([x[1] for x in coll_sets])
    OW = np.concatenate([x[2] for x in coll_sets])

    anchor_quality = cloud_quality(C, edist, Sanchor, d)

    HP, HN, HA = make_holdout_points(T)
    Hdist = point_segment_distances(HP, sharp)
    hedge = Hdist <= edge_radius
    hsmooth = ~hedge
    hfar = Hdist >= 2 * edge_radius

    variants = {
        "independent_normal": independent_normal_sources(Sanchor, Sn, d),
        "coherent_inner_cylinder": coherent_inner_cylinder_sources(Sanchor, Sn, R, L, d),
    }

    print("Step 2-KD1 coherent closed auxiliary-surface isolation test")
    print(f"  triangles              : {len(T)}")
    print(f"  sharp edges            : {edge_info['num_sharp_edges']}")
    print(f"  source anchors         : {len(Sanchor)}")
    print(f"  mixed unknowns         : {int(off[-1])}")
    print(f"  collocation points     : {len(OP)}")
    print(f"  R / d / R_s            : {R:.6e} / {d:.6e} / {R-d:.6e} m")

    variant_rows = []
    incident_rows = []
    direct_rows = []
    source_rows = []
    singular_rows = []
    variant_reports = {}

    for variant, SP in variants.items():
        print(f"\n[{variant}] assembling operator ...", flush=True)
        pos = source_position_diagnostics(SP, Sn, R, L, d)
        Mat = assemble(OP, ON, SP, Sn, SW, modes, off, a.dt)
        rs, Aw, cs, As = prepare_scaled_operator(Mat, OW)
        coldiag = operator_column_diagnostics(cs, modes, off, Sn)

        print(
            f"  actual nn min/d        : {pos['actual_nn_over_d']['min']:.6f}\n"
            f"  side-cap min/d         : {pos['cross_feature_min_over_d']:.6f}\n"
            f"  source->PEC min/d      : {pos['source_to_pec_over_d']['min']:.6f}\n"
            f"  column norm spread     : {coldiag['all']['max_over_min']:.3e}",
            flush=True,
        )

        spec = spectral_diagnostics(As, a.spectral_mode, a.rank_rtol, a.dense_svd_max_unknowns)
        if spec["condition_number"] is not None:
            print(
                f"  scaled condition       : {spec['condition_number']:.3e} "
                f"({spec['mode']})",
                flush=True,
            )

        if spec.get("singular_values") is not None:
            svals = spec["singular_values"]
            smax = max(float(svals[0]), TINY)
            for i, sv in enumerate(svals):
                singular_rows.append(
                    {
                        "variant": variant,
                        "index": i,
                        "sigma": float(sv),
                        "sigma_over_max": float(sv / smax),
                    }
                )

        max_iter = 0
        max_ne = 0.0
        all_stop = True
        worst_global = 0.0
        worst_edge = 0.0
        worst_smooth = 0.0
        worst_far = 0.0
        worst_coll = 0.0
        max_damped_opt = 0.0
        direct_worst_global = 0.0
        direct_worst_edge = 0.0
        direct_worst_smooth = 0.0
        direct_worst_far = 0.0
        direct_available = spec.get("_U") is not None and spec.get("_Vt") is not None

        for nm in ("uniform", "external_dipole"):
            fn = inc_uniform if nm == "uniform" else lambda P: inc_dip(P, L, R)
            Ei = fn(OP)
            b = tangent_rhs(Ei, ON)
            x, diag = solve_scaled_lsmr_prepared(As, cs, rs, b, a.lambda_rel, a.max_lsmr_iter)
            coll = float(np.linalg.norm(Mat @ x - b) / max(np.linalg.norm(b), TINY))

            Esc, _ = eval_field(HP, SP, Sn, SW, x, modes, off, a.dt)
            Eih = fn(HP)
            Et = Eih + Esc

            rg = relerr(Et, Eih, HN, np.ones(len(HP), bool))
            re = relerr(Et, Eih, HN, hedge)
            rsmooth = relerr(Et, Eih, HN, hsmooth)
            rf = relerr(Et, Eih, HN, hfar)

            worst_global = max(worst_global, rg)
            worst_edge = max(worst_edge, re)
            worst_smooth = max(worst_smooth, rsmooth)
            worst_far = max(worst_far, rf)
            worst_coll = max(worst_coll, coll)
            max_iter = max(max_iter, diag["iterations"])
            max_ne = max(max_ne, diag["normal_eq_rel"])
            max_damped_opt = max(max_damped_opt, diag["damped_optimality_rel"])
            all_stop = all_stop and diag["iterations"] < a.max_lsmr_iter

            incident_rows.append(
                {
                    "variant": variant,
                    "incident": nm,
                    "collocation_rel_l2": coll,
                    "global_rel_rms": rg,
                    "edge_rel_rms": re,
                    "smooth_rel_rms": rsmooth,
                    "far_rel_rms": rf,
                    **diag,
                }
            )

            if direct_available:
                xd, ddiag = direct_scaled_tikhonov(
                    spec["_U"], spec["singular_values"], spec["_Vt"],
                    As, cs, rs, b, a.lambda_rel
                )
                dcoll = float(np.linalg.norm(Mat @ xd - b) / max(np.linalg.norm(b), TINY))
                Ed, _ = eval_field(HP, SP, Sn, SW, xd, modes, off, a.dt)
                Etd = Eih + Ed
                drg = relerr(Etd, Eih, HN, np.ones(len(HP), bool))
                dre = relerr(Etd, Eih, HN, hedge)
                drs = relerr(Etd, Eih, HN, hsmooth)
                drf = relerr(Etd, Eih, HN, hfar)
                direct_worst_global = max(direct_worst_global, drg)
                direct_worst_edge = max(direct_worst_edge, dre)
                direct_worst_smooth = max(direct_worst_smooth, drs)
                direct_worst_far = max(direct_worst_far, drf)
                direct_rows.append({
                    "variant": variant,
                    "incident": nm,
                    "collocation_rel_l2": dcoll,
                    "global_rel_rms": drg,
                    "edge_rel_rms": dre,
                    "smooth_rel_rms": drs,
                    "far_rel_rms": drf,
                    **ddiag,
                })

        vr = {
            "variant": variant,
            "actual_nn_min_over_d": pos["actual_nn_over_d"]["min"],
            "actual_nn_p01_over_d": pos["actual_nn_over_d"]["p01"],
            "actual_nn_p50_over_d": pos["actual_nn_over_d"]["p50"],
            "cross_feature_min_over_d": pos["cross_feature_min_over_d"],
            "source_to_pec_min_over_d": pos["source_to_pec_over_d"]["min"],
            "source_to_pec_p01_over_d": pos["source_to_pec_over_d"]["p01"],
            "source_to_pec_p50_over_d": pos["source_to_pec_over_d"]["p50"],
            "column_norm_spread": coldiag["all"]["max_over_min"],
            "column_norm_p99_over_p01": coldiag["all"]["p99_over_p01"],
            "smooth_tangent_column_spread": coldiag.get("smooth_tangent2", {}).get("max_over_min", float("nan")),
            "scaled_condition_number": scalar_or_none(spec["condition_number"]),
            "scaled_rank_fraction": scalar_or_none(spec["rank_fraction"]),
            "max_lsmr_iterations": max_iter,
            "max_normal_eq_rel": max_ne,
            "max_damped_optimality_rel": max_damped_opt,
            "worst_collocation_rel_l2": worst_coll,
            "worst_global_rel_rms": worst_global,
            "worst_edge_rel_rms": worst_edge,
            "worst_smooth_rel_rms": worst_smooth,
            "worst_far_rel_rms": worst_far,
            "all_lsmr_stop_before_maxiter": bool(all_stop),
            "direct_svd_available": bool(direct_available),
            "direct_worst_global_rel_rms": float(direct_worst_global) if direct_available else None,
            "direct_worst_edge_rel_rms": float(direct_worst_edge) if direct_available else None,
            "direct_worst_smooth_rel_rms": float(direct_worst_smooth) if direct_available else None,
            "direct_worst_far_rel_rms": float(direct_worst_far) if direct_available else None,
        }
        variant_rows.append(vr)

        variant_reports[variant] = {
            "source_position": {
                "actual_nn_over_d": pos["actual_nn_over_d"],
                "actual_nn_pair": pos["actual_nn_pair"],
                "actual_nn_pair_features": pos["actual_nn_pair_features"],
                "cross_feature_min_over_d": pos["cross_feature_min_over_d"],
                "cross_feature_pair": pos["cross_feature_pair"],
                "source_to_pec_over_d": pos["source_to_pec_over_d"],
            },
            "operator_column_norms": coldiag,
            "spectral": clean_spectral_for_json(spec),
            "solve_worst": {
                "max_iterations": max_iter,
                "max_normal_eq_rel": max_ne,
                "max_damped_optimality_rel": max_damped_opt,
                "worst_collocation_rel_l2": worst_coll,
                "all_stop_before_maxiter": bool(all_stop),
            },
            "direct_svd_same_objective": {
                "available": bool(direct_available),
                "global_rel_rms": float(direct_worst_global) if direct_available else None,
                "edge_rel_rms": float(direct_worst_edge) if direct_available else None,
                "smooth_rel_rms": float(direct_worst_smooth) if direct_available else None,
                "far_rel_rms": float(direct_worst_far) if direct_available else None,
            },
            "holdout_worst": {
                "global_rel_rms": worst_global,
                "edge_rel_rms": worst_edge,
                "smooth_rel_rms": worst_smooth,
                "far_rel_rms": worst_far,
            },
        }

        side = pos["side_mask"]
        for j in range(len(SP)):
            source_rows.append(
                {
                    "variant": variant,
                    "source_index": j,
                    "feature": "side" if side[j] else "cap",
                    "basis_mode": "cartesian3" if modes[j] == 3 else "tangent2",
                    "edge_distance_over_d": float(Sedist[j] / d),
                    "weight": float(SW[j]),
                    "anchor_x": float(Sanchor[j, 0]),
                    "anchor_y": float(Sanchor[j, 1]),
                    "anchor_z": float(Sanchor[j, 2]),
                    "source_x": float(SP[j, 0]),
                    "source_y": float(SP[j, 1]),
                    "source_z": float(SP[j, 2]),
                    "source_to_pec_over_d": float(pos["source_to_pec_distance"][j] / d),
                    "nn_over_d": float(pos["nearest_neighbor_distance"][j] / d),
                    "nn_source_index": int(pos["nearest_neighbor_index"][j]),
                }
            )

    by_name = {r["variant"]: r for r in variant_rows}
    old = by_name["independent_normal"]
    coh = by_name["coherent_inner_cylinder"]

    def safe_improvement(old_value, new_value):
        return float(old_value / max(new_value, TINY))

    comparison = {
        "nn_min_improvement_factor": safe_improvement(coh["actual_nn_min_over_d"], old["actual_nn_min_over_d"]),
        "cross_feature_min_improvement_factor": safe_improvement(coh["cross_feature_min_over_d"], old["cross_feature_min_over_d"]),
        "source_to_pec_min_improvement_factor": safe_improvement(coh["source_to_pec_min_over_d"], old["source_to_pec_min_over_d"]),
        "column_norm_spread_reduction_factor": safe_improvement(old["column_norm_spread"], coh["column_norm_spread"]),
        "global_holdout_reduction_factor": safe_improvement(old["worst_global_rel_rms"], coh["worst_global_rel_rms"]),
        "edge_holdout_reduction_factor": safe_improvement(old["worst_edge_rel_rms"], coh["worst_edge_rel_rms"]),
        "smooth_holdout_reduction_factor": safe_improvement(old["worst_smooth_rel_rms"], coh["worst_smooth_rel_rms"]),
    }
    if old["scaled_condition_number"] is not None and coh["scaled_condition_number"] is not None:
        comparison["scaled_condition_reduction_factor"] = safe_improvement(
            old["scaled_condition_number"], coh["scaled_condition_number"]
        )
    else:
        comparison["scaled_condition_reduction_factor"] = None

    geometry_gates = {
        "coherent_source_to_pec_min_at_least_0p95d": bool(coh["source_to_pec_min_over_d"] >= 0.95),
        # The coherent cap disk is much smaller when d/R is large, so same-feature
        # cap spacing may remain tight even after invalid side-cap penetration is
        # removed. Gate cross-feature separation directly and retain a weak global
        # anti-collision floor.
        "coherent_actual_nn_min_over_d_above_0p03": bool(coh["actual_nn_min_over_d"] > 0.03),
        "coherent_cross_feature_min_over_d_above_0p10": bool(coh["cross_feature_min_over_d"] > 0.10),
    }
    scaling_gates = {
        "column_norm_spread_reduced_at_least_10x": bool(comparison["column_norm_spread_reduction_factor"] >= 10.0),
    }
    spectral_gates = {}
    if coh["scaled_condition_number"] is not None and old["scaled_condition_number"] is not None:
        spectral_gates["coherent_condition_not_worse_than_10x_independent"] = bool(
            coh["scaled_condition_number"] <= 10.0 * old["scaled_condition_number"]
        )
        if coh["scaled_rank_fraction"] is not None:
            spectral_gates["coherent_scaled_rank_fraction_above_95pct"] = bool(coh["scaled_rank_fraction"] >= 0.95)

    solve_gates = {
        "coherent_lsmr_stops_before_maxiter": bool(coh["all_lsmr_stop_before_maxiter"]),
        "coherent_damped_optimality_below_1e-6": bool(coh["max_damped_optimality_rel"] < 1.0e-6),
    }
    field_improvement_gates = {
        "global_holdout_reduced_at_least_5x": bool(comparison["global_holdout_reduction_factor"] >= 5.0),
        "edge_holdout_reduced_at_least_5x": bool(comparison["edge_holdout_reduction_factor"] >= 5.0),
        "smooth_holdout_not_worse": bool(coh["worst_smooth_rel_rms"] <= old["worst_smooth_rel_rms"]),
    }
    accuracy_gates = {
        "coherent_global_holdout_below_5pct": bool(coh["worst_global_rel_rms"] < 0.05),
        "coherent_edge_holdout_below_5pct": bool(coh["worst_edge_rel_rms"] < 0.05),
        "coherent_smooth_holdout_below_5pct": bool(coh["worst_smooth_rel_rms"] < 0.05),
    }

    # First-stage KD1 success is deliberately geometric/numerical: does a coherent
    # closed source surface remove the invalid cross-feature proximity and the
    # pathological raw column scaling?  Field error is judged separately.
    geometry_isolation_pass = all(geometry_gates.values()) and all(scaling_gates.values())
    spectral_ok = all(spectral_gates.values()) if spectral_gates else True
    production_ready = geometry_isolation_pass and spectral_ok and all(solve_gates.values()) and all(accuracy_gates.values())

    if production_ready:
        diagnosis = "coherent_closed_auxiliary_surface_confirmed_ready_for_production_integration"
    elif geometry_isolation_pass and not spectral_ok:
        diagnosis = "coherent_geometry_root_cause_confirmed_but_inner_surface_sampling_is_spectrally_degenerate"
    elif geometry_isolation_pass:
        diagnosis = "coherent_surface_geometry_root_cause_confirmed_but_accuracy_or_solver_needs_further_tuning"
    elif all(geometry_gates.values()) and not all(scaling_gates.values()):
        diagnosis = "coherent_geometry_fixed_but_operator_scaling_not_rescued"
    elif all(geometry_gates.values()):
        diagnosis = "coherent_geometry_fixed_but_field_error_not_materially_improved"
    else:
        diagnosis = "coherent_inner_cylinder_not_confirmed_by_kd1"

    area_rel = abs(np.sum(SW) - total_area) / max(total_area, TINY)
    allcart_unknowns = 3 * len(Sanchor)
    mixed_unknowns = int(off[-1])
    saving = 1.0 - mixed_unknowns / allcart_unknowns

    report = {
        "configuration": vars(a),
        "fixed_problem": {
            "used_arm_length": L,
            "radius": R,
            "source_offset": d,
            "inner_radius": R - d,
            "inner_z_min": d,
            "inner_z_max": L - d,
            "edge_radius": edge_radius,
            "smooth_target_h": hs,
            "edge_target_h": he,
            "triangle_count": len(T),
            "surface_area": total_area,
            "source_count": len(Sanchor),
            "mixed_unknowns": mixed_unknowns,
            "all_cartesian_unknowns_same_anchors": allcart_unknowns,
            "unknown_saving": saving,
            "num_cartesian_sources": int(np.sum(modes == 3)),
            "num_tangent_sources": int(np.sum(modes == 2)),
            "area_relative_error": area_rel,
        },
        "edge_detection": edge_info,
        "source_budget": budget,
        "anchor_quality": anchor_quality,
        "variants": variant_reports,
        "comparison": comparison,
        "gates": {
            "geometry": geometry_gates,
            "scaling": scaling_gates,
            "spectral": spectral_gates,
            "solve": solve_gates,
            "field_improvement": field_improvement_gates,
            "absolute_accuracy": accuracy_gates,
        },
        "geometry_isolation_pass": bool(geometry_isolation_pass),
        "production_ready": bool(production_ready),
        "overall_pass": bool(production_ready),
        "diagnosis": diagnosis,
    }
    (out / "step2kd1_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    with open(out / "step2kd1_variant_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(variant_rows[0].keys()))
        w.writeheader()
        w.writerows(variant_rows)

    with open(out / "step2kd1_incident_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(incident_rows[0].keys()))
        w.writeheader()
        w.writerows(incident_rows)

    if direct_rows:
        with open(out / "step2kd1_direct_svd_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(direct_rows[0].keys()))
            w.writeheader()
            w.writerows(direct_rows)

    with open(out / "step2kd1_source_geometry.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(source_rows[0].keys()))
        w.writeheader()
        w.writerows(source_rows)

    if singular_rows:
        with open(out / "step2kd1_singular_values.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(singular_rows[0].keys()))
            w.writeheader()
            w.writerows(singular_rows)

    lines = [
        "Step 2-KD1 coherent closed auxiliary-surface isolation test",
        "==========================================================",
        "",
        f"triangles                         : {len(T)}",
        f"sharp / boundary / nonmanifold    : {edge_info['num_sharp_edges']} / {edge_info['num_boundary_edges']} / {edge_info['num_nonmanifold_edges']}",
        f"sources                           : {len(Sanchor)}",
        f"Cartesian / tangent sources       : {int(np.sum(modes==3))} / {int(np.sum(modes==2))}",
        f"mixed / all-Cart unknowns         : {mixed_unknowns} / {allcart_unknowns}",
        f"unknown saving                    : {100*saving:.2f}%",
        f"anchor nn min / d                 : {anchor_quality['anchor_nn_min_over_d']:.6f}",
        "",
        "VARIANT COMPARISON",
    ]
    for row in variant_rows:
        lines += [
            f"  {row['variant']}",
            f"    actual nn min / p01 / p50 / d : {row['actual_nn_min_over_d']:.6f} / {row['actual_nn_p01_over_d']:.6f} / {row['actual_nn_p50_over_d']:.6f}",
            f"    actual nn closest pair type    : {'/'.join(variant_reports[row['variant']]['source_position']['actual_nn_pair_features'])}",
            f"    side-cap cross min / d         : {row['cross_feature_min_over_d']:.6f}",
            f"    source->PEC min / p01 / p50 /d : {row['source_to_pec_min_over_d']:.6f} / {row['source_to_pec_p01_over_d']:.6f} / {row['source_to_pec_p50_over_d']:.6f}",
            f"    weighted column spread         : {row['column_norm_spread']:.3e}",
            f"    p99/p01 column spread          : {row['column_norm_p99_over_p01']:.3e}",
            f"    smooth tangent column spread   : {row['smooth_tangent_column_spread']:.3e}",
            f"    scaled condition               : {row['scaled_condition_number'] if row['scaled_condition_number'] is not None else 'not computed'}",
            f"    max LSMR iterations            : {row['max_lsmr_iterations']}",
            f"    legacy undamped normal-eq rel  : {row['max_normal_eq_rel']:.3e}",
            f"    damped-objective optimality rel: {row['max_damped_optimality_rel']:.3e}",
            f"    worst collocation residual     : {100*row['worst_collocation_rel_l2']:.3f}%",
            f"    worst global holdout           : {100*row['worst_global_rel_rms']:.3f}%",
            f"    worst edge holdout             : {100*row['worst_edge_rel_rms']:.3f}%",
            f"    worst smooth holdout           : {100*row['worst_smooth_rel_rms']:.3f}%",
            f"    worst far holdout              : {100*row['worst_far_rel_rms']:.3f}%",
            f"    direct-SVD global/edge/smooth  : " + (
                f"{100*row['direct_worst_global_rel_rms']:.3f}% / {100*row['direct_worst_edge_rel_rms']:.3f}% / {100*row['direct_worst_smooth_rel_rms']:.3f}%"
                if row['direct_svd_available'] else "not available"
            ),
            "",
        ]

    lines += [
        "IMPROVEMENT FACTORS (coherent vs independent)",
        f"  actual nn min                    : {comparison['nn_min_improvement_factor']:.3f}x",
        f"  side-cap cross separation       : {comparison['cross_feature_min_improvement_factor']:.3f}x",
        f"  source->PEC minimum distance    : {comparison['source_to_pec_min_improvement_factor']:.3f}x",
        f"  column-spread reduction         : {comparison['column_norm_spread_reduction_factor']:.3f}x",
        f"  global holdout reduction        : {comparison['global_holdout_reduction_factor']:.3f}x",
        f"  edge holdout reduction          : {comparison['edge_holdout_reduction_factor']:.3f}x",
        f"  smooth holdout reduction        : {comparison['smooth_holdout_reduction_factor']:.3f}x",
    ]
    if comparison["scaled_condition_reduction_factor"] is not None:
        lines.append(f"  scaled-condition reduction      : {comparison['scaled_condition_reduction_factor']:.3f}x")

    lines += ["", "GATING"]
    for group_name, gates in (
        ("geometry", geometry_gates),
        ("scaling", scaling_gates),
        ("spectral", spectral_gates),
        ("solve", solve_gates),
        ("field improvement", field_improvement_gates),
        ("absolute accuracy", accuracy_gates),
    ):
        lines.append(f"  [{group_name}]")
        lines += [f"    {k:<54s}: {'PASS' if v else 'FAIL'}" for k, v in gates.items()]

    lines += [
        "",
        f"KD1 GEOMETRY-ISOLATION PASS: {'PASS' if geometry_isolation_pass else 'FAIL'}",
        f"KD1 PRODUCTION-READY PASS   : {'PASS' if production_ready else 'FAIL'}",
        f"DIAGNOSIS: {diagnosis}",
        "",
        "Interpretation:",
        "  The geometry-isolation gate asks whether the coherent closed inner",
        "  surface removes the invalid side-cap near-collision and pathological",
        "  raw operator column scaling.  Spectral conditioning is reported",
        "  separately because a geometrically valid surface can still be badly",
        "  sampled.  When dense SVD is available, the same damped objective is",
        "  also solved directly to separate geometry/basis error from LSMR error.",
        "  The production-ready gate is deliberately stricter: global, edge,",
        "  and smooth hold-out must each be <5%, and solver diagnostics must pass.",
        "  A tiny collocation residual alone is never treated as physical accuracy.",
    ]
    (out / "step2kd1_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0 if production_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
