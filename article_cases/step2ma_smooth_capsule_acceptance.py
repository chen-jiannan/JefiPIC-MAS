#!/usr/bin/env python3
"""
Step 2-MA: rounded/smooth closed antenna element independent-holdout validation.

NUMBERING
---------
Step 2-K diagnosed fixed-offset robustness on sharp closed STL geometry.
Step 2-L tested geometry/feature-aware source placement for the sharp-rim case.
The present Step 2-MA starts a NEW restricted-geometry acceptance branch.
Step 3 is NOT started.

PURPOSE
-------
The article is no longer required to support arbitrary sharp-edged STL PECs.
MA asks a narrower, publication-relevant question:

    Can the current time-local electric-dipole MAS closure accurately enforce
    the PEC condition on a smooth, electrically-small, closed antenna element?

The physical conductor is a smooth capsule/spherocylinder: a cylindrical body
with hemispherical rounded ends.  It has no sharp rim or corner.  Its exact
inward parallel surface is another capsule about the same centerline segment,
so source geometry remains coherent and non-self-intersecting.

For each depth delta/R, source points are sampled independently on the actual
inner capsule rather than obtained by one-to-one compression of physical PEC
anchors.  Physical matching points and auxiliary source points are therefore
fully decoupled.  The solve uses a 2x overdetermined matching cloud by default.

All auxiliary sources use the generalized tangent2 electric-dipole basis that
is closest to the current smooth-surface production rule.  Independent
hold-out points are different barycentric points on every physical STL
triangle and are never used in the solve.

PRIMARY ACCEPTANCE
------------------
A depth is accepted only if BOTH incident fields satisfy all independent PEC
hold-out gates (global, rounded-cap, central-side, and transition regions).
A small collocation residual alone is not sufficient.

This script is diagnostic only.  It does NOT modify production modules or the
manuscript.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree

try:
    import step2kd1_coherent_closed_auxiliary_surface as kd1
except ImportError as exc:
    raise SystemExit(
        "Step 2-MA requires article_cases/"
        "step2kd1_coherent_closed_auxiliary_surface.py from Step 2-KD1."
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
TINY = kd1.TINY


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="step2ma_smooth_capsule_acceptance")
    p.add_argument("--radius", type=float, default=1.0e-3)
    p.add_argument("--arm-length", type=float, default=29.0e-3,
                   help="Total capsule end-to-end length.")
    p.add_argument(
        "--depth-fracs", type=float, nargs="+",
        default=[0.50, 0.60, 0.70, 0.80],
        help="Inner-capsule offset depths delta/R."
    )
    p.add_argument("--source-count", type=int, default=900)
    p.add_argument("--collocation-factor", type=float, default=2.0)
    p.add_argument("--lambda-rel", type=float, default=1.0e-4)
    p.add_argument("--dt", type=float, default=1.0e-9)
    p.add_argument("--holdout-gate", type=float, default=0.05)
    p.add_argument("--collocation-gate", type=float, default=0.05)
    p.add_argument("--with-lsmr", action="store_true")
    p.add_argument("--max-lsmr-iter", type=int, default=12000)
    return p.parse_args()


def analytic_capsule_normal(P, R, L):
    P = np.asarray(P, float)
    z0, z1 = R, L - R
    qz = np.clip(P[:, 2], z0, z1)
    Q = np.column_stack((np.zeros(len(P)), np.zeros(len(P)), qz))
    V = P - Q
    return kd1.unit(V)


def project_to_capsule(P, radius, Rphysical, L):
    """Project points radially onto a capsule around centerline [R,L-R]."""
    P = np.asarray(P, float)
    z0, z1 = Rphysical, L - Rphysical
    qz = np.clip(P[:, 2], z0, z1)
    Q = np.column_stack((np.zeros(len(P)), np.zeros(len(P)), qz))
    V = P - Q
    N = kd1.unit(V)
    return Q + float(radius) * N, N


def capsule_clearance(P, R, L):
    """Positive inward distance from an interior point to the analytic PEC."""
    P = np.asarray(P, float)
    z0, z1 = R, L - R
    qz = np.clip(P[:, 2], z0, z1)
    Q = np.column_stack((np.zeros(len(P)), np.zeros(len(P)), qz))
    rho = np.linalg.norm(P - Q, axis=1)
    return R - rho


def orient_triangles_outward(T, R, L):
    C, N, _ = kd1.tri_geom(T)
    A = analytic_capsule_normal(C, R, L)
    flip = np.sum(N * A, axis=1) < 0.0
    if np.any(flip):
        T = T.copy()
        tmp = T[flip, 1].copy()
        T[flip, 1] = T[flip, 2]
        T[flip, 2] = tmp
    return T


def generate_capsule(radius, Rphysical, L, n_cyl, nphi, n_hemi):
    """Triangulated capsule around z-axis with centerline z in [R,L-R]."""
    if radius <= 0.0:
        raise ValueError("capsule radius must be positive")
    if L <= 2.0 * Rphysical:
        raise ValueError("arm length must exceed 2*physical radius")

    ph = 2.0 * np.pi * np.arange(nphi) / nphi
    rings = []

    # Bottom hemisphere rings, excluding pole but including equator.
    for th in np.linspace(-0.5*np.pi, 0.0, n_hemi + 1)[1:]:
        rr = radius * np.cos(th)
        zz = Rphysical + radius * np.sin(th)
        rings.append((float(zz), float(rr)))

    # Cylinder interior rings and top equator.
    z0, z1 = Rphysical, L - Rphysical
    for zz in np.linspace(z0, z1, n_cyl + 1)[1:]:
        rings.append((float(zz), float(radius)))

    # Top hemisphere rings, excluding equator and pole.
    for th in np.linspace(0.0, 0.5*np.pi, n_hemi + 1)[1:-1]:
        rr = radius * np.cos(th)
        zz = (L - Rphysical) + radius * np.sin(th)
        rings.append((float(zz), float(rr)))

    verts = []
    for zz, rr in rings:
        ring = np.column_stack((rr*np.cos(ph), rr*np.sin(ph), np.full(nphi, zz)))
        verts.append(ring)

    bottom = np.array([0.0, 0.0, Rphysical - radius])
    top = np.array([0.0, 0.0, L - Rphysical + radius])
    tris = []

    first = verts[0]
    for j in range(nphi):
        k = (j + 1) % nphi
        tris.append(np.asarray([bottom, first[j], first[k]], float))

    for a, b in zip(verts[:-1], verts[1:]):
        for j in range(nphi):
            k = (j + 1) % nphi
            tris.append(np.asarray([a[j], b[j], b[k]], float))
            tris.append(np.asarray([a[j], b[k], a[k]], float))

    last = verts[-1]
    for j in range(nphi):
        k = (j + 1) % nphi
        tris.append(np.asarray([last[j], top, last[k]], float))

    T = np.asarray(tris, float)
    return orient_triangles_outward(T, Rphysical, L)


def barycentric_points(T, weights, R, L):
    w = np.asarray(weights, float)
    P0 = np.einsum("k,nkj->nj", w, T)
    return project_to_capsule(P0, R, R, L)


def select_fps(P, n, seed):
    n = min(int(n), len(P))
    return kd1.fps_indices(P, n, seed=seed)


def nn_stats(P, scale):
    if len(P) < 2:
        return {"min": None, "p01": None, "p50": None, "p95": None}
    d, _ = cKDTree(P).query(P, k=2)
    x = d[:, 1] / max(scale, TINY)
    return {
        "min": float(np.min(x)),
        "p01": float(np.quantile(x, 0.01)),
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
    }


def direct_ridge(As, cs, rs, b, lambda_rel):
    bw = b * rs
    smax = kd1.estimate_smax(As)
    damp = float(lambda_rel) * float(smax)
    H = As.T @ As
    H.flat[::H.shape[0] + 1] += damp * damp
    g = As.T @ bw
    cf = cho_factor(H, lower=True, overwrite_a=True, check_finite=False)
    y = cho_solve(cf, g, check_finite=False)
    x = y / cs
    r = As @ y - bw
    grad = As.T @ r + damp*damp*y
    scale = np.linalg.norm(As, ord="fro") * np.linalg.norm(r) + damp*damp*np.linalg.norm(y)
    return x, {
        "damp": damp,
        "smax_scaled_est": float(smax),
        "weighted_residual_rel": float(np.linalg.norm(r) / max(np.linalg.norm(bw), TINY)),
        "damped_optimality_rel": float(np.linalg.norm(grad) / max(scale, TINY)),
    }


def feature_masks(P, R, L):
    z = P[:, 2]
    # Hemisphere regions.
    caps = (z < R) | (z > L - R)
    side = ~caps
    # Smooth C1 joins are near z=R and z=L-R; keep a dedicated band for diagnosis.
    band = 0.75 * R
    transition = (np.abs(z - R) <= band) | (np.abs(z - (L - R)) <= band)
    side_mid = side & ~transition
    cap_tip = (z <= 0.50*R) | (z >= L - 0.50*R)
    return {
        "global": np.ones(len(P), bool),
        "rounded_caps": caps,
        "cap_tips": cap_tip,
        "transition": transition,
        "side_mid": side_mid,
    }


def region_errors(total, inc, N, masks):
    out = {}
    for name, mask in masks.items():
        if np.any(mask):
            out[name] = kd1.relerr(total, inc, N, mask)
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k); keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def safe_max(rows, key):
    vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
    return max(vals) if vals else None


def main():
    a = parse_args()
    out = ROOT / "article_cases_output" / a.out
    out.mkdir(parents=True, exist_ok=True)

    R = float(a.radius)
    L = 8.0e-3 if a.quick else float(a.arm_length)
    if L <= 2.0*R:
        raise ValueError("arm length must exceed 2R")

    fracs = sorted({float(x) for x in a.depth_fracs if 0.0 < float(x) < 1.0})
    if not fracs:
        raise ValueError("need depth fractions in (0,1)")

    if a.quick:
        nphi, ncyl, nhemi = 28, 28, 10
        nsrc = min(180, int(a.source_count))
    else:
        nphi, ncyl, nhemi = 72, 96, 24
        nsrc = int(a.source_count)
    ncoll = max(nsrc + 1, int(math.ceil(float(a.collocation_factor) * nsrc)))

    # Physical capsule and exact projected candidate points.
    T = generate_capsule(R, R, L, ncyl, nphi, nhemi)
    kd1.write_ascii_stl(out / "step2ma_physical_capsule.stl", T)
    C0, _, Atri = kd1.tri_geom(T)
    C, CN = project_to_capsule(C0, R, R, L)
    total_area = float(np.sum(Atri))

    # Frozen physical matching cloud for all depths.
    cidx = select_fps(C, ncoll, seed=1201)
    OP, ON = C[cidx], CN[cidx]
    OW, _ = kd1.voronoi_weights(C, Atri, OP)

    # Independent hold-out: different barycentric point on EVERY triangle.
    HP, HN = barycentric_points(T, [0.19, 0.37, 0.44], R, L)
    masks = feature_masks(HP, R, L)

    # A second independent spot-check holdout guards against one barycentric pattern.
    HP2, HN2 = barycentric_points(T, [0.53, 0.29, 0.18], R, L)
    masks2 = feature_masks(HP2, R, L)

    print("Step 2-MA smooth closed antenna acceptance")
    print("  Step 3 started                 : NO")
    print("  physical geometry             : capsule / spherocylinder")
    print(f"  physical triangles            : {len(T)}")
    print(f"  source count                  : {nsrc}")
    print(f"  collocation points            : {len(OP)} ({len(OP)/nsrc:.2f}x sources)")
    print(f"  independent holdout points    : {len(HP)} + {len(HP2)} spot-check")
    print(f"  tangent2 unknown count        : {2*nsrc}")
    print(f"  depth sweep delta/R           : {', '.join(f'{x:.2f}' for x in fracs)}")

    variant_rows = []
    incident_rows = []
    source_rows = []
    report_variants = {}

    for frac in fracs:
        d = frac * R
        Rs = R - d
        if Rs <= 0:
            continue

        # Independent source sampling on the actual coherent inner capsule.
        Ts = generate_capsule(Rs, R, L, ncyl, nphi, nhemi)
        SC0, _, SAtri = kd1.tri_geom(Ts)
        SC, SCN = project_to_capsule(SC0, Rs, R, L)
        sidx = select_fps(SC, nsrc, seed=2201)
        SP, SN = SC[sidx], SCN[sidx]
        SW, _ = kd1.voronoi_weights(SC, SAtri, SP)
        # Keep total quadrature scale fixed across depth variants.
        SW *= total_area / max(float(np.sum(SW)), TINY)

        modes = np.full(nsrc, 2, dtype=np.int32)
        off = kd1.offsets(modes)

        nn = nn_stats(SP, d)
        clearance = capsule_clearance(SP, R, L) / d
        qclear = {
            "min": float(np.min(clearance)),
            "p01": float(np.quantile(clearance, 0.01)),
            "p50": float(np.quantile(clearance, 0.50)),
            "p99": float(np.quantile(clearance, 0.99)),
        }

        variant = f"capsule_depth_{frac:.2f}R".replace(".", "p")
        print(f"\n[{variant}] assembling {2*len(OP)} x {off[-1]} operator ...", flush=True)

        Mat = kd1.assemble(OP, ON, SP, SN, SW, modes, off, a.dt)
        rs, _, cs, As = kd1.prepare_scaled_operator(Mat, OW)
        coldiag = kd1.operator_column_diagnostics(cs, modes, off, SN)

        rows_here = []
        for incident in ("uniform", "external_dipole"):
            fn = kd1.inc_uniform if incident == "uniform" else lambda P: kd1.inc_dip(P, L, R)
            b = kd1.tangent_rhs(fn(OP), ON)
            x, ddiag = direct_ridge(As, cs, rs, b, a.lambda_rel)

            coll = float(np.linalg.norm(Mat @ x - b) / max(np.linalg.norm(b), TINY))

            E1, _ = kd1.eval_field(HP, SP, SN, SW, x, modes, off, a.dt)
            I1 = fn(HP)
            er1 = region_errors(I1 + E1, I1, HN, masks)

            E2, _ = kd1.eval_field(HP2, SP, SN, SW, x, modes, off, a.dt)
            I2 = fn(HP2)
            er2 = region_errors(I2 + E2, I2, HN2, masks2)

            row = {
                "variant": variant,
                "depth_over_R": frac,
                "incident": incident,
                "source_count": nsrc,
                "unknown_count": int(off[-1]),
                "collocation_points": len(OP),
                "collocation_rel_l2": coll,
                **{f"holdout_{k}_rel_rms": v for k, v in er1.items()},
                **{f"spot_{k}_rel_rms": v for k, v in er2.items()},
                "direct_weighted_residual_rel": ddiag["weighted_residual_rel"],
                "direct_damped_optimality_rel": ddiag["damped_optimality_rel"],
                "damp": ddiag["damp"],
            }

            if a.with_lsmr:
                xl, ld = kd1.solve_scaled_lsmr_prepared(
                    As, cs, rs, b, a.lambda_rel, a.max_lsmr_iter
                )
                row.update({
                    "lsmr_iterations": ld["iterations"],
                    "lsmr_istop": ld["istop"],
                    "lsmr_damped_optimality_rel": ld["damped_optimality_rel"],
                    "direct_vs_lsmr_coeff_rel": float(
                        np.linalg.norm(x-xl) / max(np.linalg.norm(x), TINY)
                    ),
                })

            incident_rows.append(row)
            rows_here.append(row)
            print(
                f"  {incident:16s} coll={100*coll:7.3f}% "
                f"global={100*er1['global']:7.3f}% "
                f"caps={100*er1['rounded_caps']:7.3f}% "
                f"side={100*er1['side_mid']:7.3f}% "
                f"transition={100*er1['transition']:7.3f}% "
                f"spot-global={100*er2['global']:7.3f}%"
            )

        gate_keys = (
            "holdout_global_rel_rms",
            "holdout_rounded_caps_rel_rms",
            "holdout_side_mid_rel_rms",
            "holdout_transition_rel_rms",
            "spot_global_rel_rms",
            "spot_rounded_caps_rel_rms",
            "spot_side_mid_rel_rms",
            "spot_transition_rel_rms",
        )
        max_hold = max(float(r[k]) for r in rows_here for k in gate_keys)
        max_coll = max(float(r["collocation_rel_l2"]) for r in rows_here)
        accepted = (max_hold < a.holdout_gate) and (max_coll < a.collocation_gate)

        vr = {
            "variant": variant,
            "depth_over_R": frac,
            "inner_radius_over_R": Rs/R,
            "source_count": nsrc,
            "unknown_count": int(off[-1]),
            "collocation_points": len(OP),
            "collocation_to_source_ratio": len(OP)/nsrc,
            "source_nn_min_over_d": nn["min"],
            "source_nn_p50_over_d": nn["p50"],
            "source_nn_p95_over_d": nn["p95"],
            "source_to_pec_min_over_d": qclear["min"],
            "source_to_pec_p50_over_d": qclear["p50"],
            "column_norm_spread": coldiag["all"]["max_over_min"],
            "worst_collocation_rel_l2": max_coll,
            "worst_independent_holdout_rel_rms": max_hold,
            "accepted": bool(accepted),
        }
        for key in (
            "holdout_global_rel_rms", "holdout_rounded_caps_rel_rms",
            "holdout_cap_tips_rel_rms", "holdout_transition_rel_rms",
            "holdout_side_mid_rel_rms", "spot_global_rel_rms"
        ):
            vr[f"worst_{key}"] = safe_max(rows_here, key)
        variant_rows.append(vr)
        report_variants[variant] = {"summary": vr, "incidents": rows_here}

        for j in range(len(SP)):
            source_rows.append({
                "variant": variant,
                "source_index": j,
                "x": SP[j,0], "y": SP[j,1], "z": SP[j,2],
                "nx": SN[j,0], "ny": SN[j,1], "nz": SN[j,2],
                "weight": SW[j],
                "mode": "tangent2",
                "clearance_over_d": clearance[j],
            })

        # Export each inner source surface for visual inspection.
        tag = f"{frac:.2f}".replace(".", "p")
        kd1.write_ascii_stl(out / f"step2ma_inner_capsule_{tag}R.stl", Ts)

    accepted_rows = [r for r in variant_rows if r["accepted"]]
    best = min(variant_rows, key=lambda r: r["worst_independent_holdout_rel_rms"])
    if accepted_rows:
        accepted_rows.sort(key=lambda r: r["worst_independent_holdout_rel_rms"])
        diagnosis = "smooth_closed_capsule_ACCEPTED"
        best_accepted = accepted_rows[0]["variant"]
    else:
        diagnosis = "smooth_closed_capsule_NOT_YET_ACCEPTED"
        best_accepted = None

    write_csv(out / "step2ma_variant_results.csv", variant_rows)
    write_csv(out / "step2ma_incident_results.csv", incident_rows)
    write_csv(out / "step2ma_source_cloud.csv", source_rows)

    report = {
        "experiment": "Step 2-MA",
        "step3_started": False,
        "purpose": "restricted smooth closed antenna independent-holdout acceptance",
        "geometry": "capsule/spherocylinder",
        "sharp_edges": False,
        "source_basis": "tangent2",
        "source_count": nsrc,
        "collocation_points": len(OP),
        "holdout_points_primary": len(HP),
        "holdout_points_spot": len(HP2),
        "depth_fracs": fracs,
        "holdout_gate": a.holdout_gate,
        "collocation_gate": a.collocation_gate,
        "diagnosis": diagnosis,
        "best_by_holdout": best["variant"],
        "best_accepted": best_accepted,
        "production_integration_authorized": False,
        "variants": report_variants,
    }
    with open(out / "step2ma_report.json", "w") as f:
        json.dump(report, f, indent=2)

    lines = [
        "Step 2-MA rounded/smooth closed antenna acceptance",
        "==================================================",
        "",
        "NUMBERING",
        "  major step                         : Step 2",
        "  branch                             : M",
        "  diagnostic                         : MA",
        "  Step 3 started                     : NO",
        "",
        "SCOPE",
        "  geometry                           : smooth capsule/spherocylinder",
        "  sharp edges/corners                : excluded by design",
        "  auxiliary basis                    : tangent2 electric dipoles",
        f"  sources / unknowns                 : {nsrc} / {2*nsrc}",
        f"  collocation points                 : {len(OP)}",
        f"  independent holdout points         : {len(HP)} + {len(HP2)} spot-check",
        f"  depth sweep delta/R                : {', '.join(f'{x:.2f}' for x in fracs)}",
        "",
        "ACCEPTANCE",
        f"  collocation gate                   : {100*a.collocation_gate:.2f}%",
        f"  every independent holdout gate     : {100*a.holdout_gate:.2f}%",
        "",
    ]
    for r in variant_rows:
        lines += [
            f"  {r['variant']}",
            f"    source NN p50/d                  : {r['source_nn_p50_over_d']:.4f}",
            f"    source->PEC min/d                : {r['source_to_pec_min_over_d']:.4f}",
            f"    column norm spread               : {r['column_norm_spread']:.4e}",
            f"    worst collocation                : {100*r['worst_collocation_rel_l2']:.3f}%",
            f"    worst independent holdout        : {100*r['worst_independent_holdout_rel_rms']:.3f}%",
            f"    accepted                         : {'PASS' if r['accepted'] else 'FAIL'}",
        ]
    lines += [
        "",
        f"DIAGNOSIS                            : {diagnosis}",
        f"best by independent holdout          : {best['variant']}",
        f"best accepted variant                : {best_accepted}",
        "",
        "Step 2-MA PRODUCTION INTEGRATION     : NOT AUTHORIZED BY THIS DIAGNOSTIC",
        "",
        "Interpretation:",
        "  PASS means the restricted smooth closed geometry family has at least",
        "  one tested working depth under two independent holdout point patterns.",
        "  FAIL means the remaining issue is not attributable to sharp PEC rims;",
        "  do not hide the failure by relying on collocation residual alone.",
    ]
    (out / "step2ma_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nWrote results to: {out}")


if __name__ == "__main__":
    main()
