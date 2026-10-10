
"""Case 2a diagnostic: conducting sphere in a uniform electrostatic field.

Purpose
-------
This is a particle-free, reservoir-free diagnostic for the Route-2 floating
surface-charge closure. It verifies whether an inward auxiliary-source /
equipotential surface solve can reproduce the analytic induced-charge solution
for a smooth conducting sphere in a uniform electrostatic field.

Analytic reference
------------------
For an uncharged conducting sphere of radius R in a uniform field E0 zhat,

    phi_ext = -E0 z,
    phi_total(r=R) = constant = 0,
    E_total(r=R) = 3 E0 cos(theta) rhat,
    E_t(r=R) = 0,
    p_induced = 4 pi eps0 E0 R^3 zhat,
    sigma_phys(theta) = 3 eps0 E0 cos(theta).

The MAS unknowns are auxiliary point charges located on an inward-offset source
surface at radius a = R - d. Therefore q_j / A_R should be compared with the
auxiliary-source coefficient

    sigma_aux(theta; d) = 3 eps0 E0 (R/a) cos(theta),

where A_R is the physical collocation patch area. The physical surface charge
sigma_phys is not expected to overlap with sigma_aux when d/R is nonzero.

Run from the JefiPIC repository root, for example:

    python article_cases/case2_sphere_uniform_field_induced_charge_test.py \
      --radius 0.004 \
      --E0 1.0 \
      --mas-points 96,192,320 \
      --source-offsets 0.10R,0.20R,0.35R \
      --eval-offset-fracs 0.05,0.10 \
      --surface-plot-offset 0.35R \
      --publication-formats png,pdf \
      --out article_cases_output/case2a_sphere_uniform_field
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS0 = 8.8541878128e-12
K_COUL = 1.0 / (4.0 * math.pi * EPS0)


def _parse_csv_ints(text: str) -> List[int]:
    vals: List[int] = []
    for token in str(text).split(','):
        token = token.strip()
        if token:
            vals.append(int(token))
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return vals


def _parse_csv_floats_or_radius_fracs(text: str, radius: float) -> List[float]:
    vals: List[float] = []
    for token in str(text).split(','):
        token = token.strip()
        if not token:
            continue
        low = token.lower()
        if low.endswith('r'):
            vals.append(float(low[:-1]) * float(radius))
        else:
            vals.append(float(token))
    if not vals:
        raise ValueError("expected at least one offset value")
    return vals


def _parse_one_float_or_radius_frac(text: str, radius: float) -> float:
    vals = _parse_csv_floats_or_radius_fracs(text, radius)
    if len(vals) != 1:
        raise ValueError(f"expected one value, got {text!r}")
    return float(vals[0])


def publication_formats(text: str) -> List[str]:
    allowed = {"png", "pdf", "svg"}
    out: List[str] = []
    for token in str(text).split(','):
        token = token.strip().lower()
        if token and token in allowed and token not in out:
            out.append(token)
    return out or ["png", "pdf"]


def save_figure(fig, base: Path, formats: Sequence[str], dpi: int) -> None:
    base = Path(base)
    for ext in formats:
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = int(dpi)
        fig.savefig(base.with_suffix("." + ext), **kwargs)
    plt.close(fig)


def fibonacci_sphere_points(n: int, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nearly uniform sphere points, outward normals, and equal physical patch areas."""
    if n < 4:
        raise ValueError("Need at least 4 points on the sphere")
    i = np.arange(n, dtype=np.float64)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / float(n)
    r_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = golden_angle * i
    x = r_xy * np.cos(phi)
    y = r_xy * np.sin(phi)
    normals = np.column_stack([x, y, z])
    points = float(radius) * normals
    areas = np.full(n, 4.0 * math.pi * float(radius) ** 2 / float(n), dtype=np.float64)
    return points, normals, areas


def potential_matrix(obs: np.ndarray, src: np.ndarray, softening: float = 0.0) -> np.ndarray:
    d = obs[:, None, :] - src[None, :, :]
    r2 = np.sum(d * d, axis=2)
    if softening and softening > 0.0:
        r2 = r2 + float(softening) ** 2
    r = np.sqrt(np.maximum(r2, 1e-300))
    return K_COUL / r


def point_charge_field(obs: np.ndarray, src: np.ndarray, q: np.ndarray, softening: float = 0.0) -> np.ndarray:
    d = obs[:, None, :] - src[None, :, :]
    r2 = np.sum(d * d, axis=2)
    if softening and softening > 0.0:
        r2 = r2 + float(softening) ** 2
    r2 = np.maximum(r2, 1e-300)
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))
    return K_COUL * np.einsum('j,ij,ijk->ik', q, inv_r3, d)


def analytic_conducting_sphere_field(points: np.ndarray, radius: float, E0: float) -> np.ndarray:
    """Total field outside an uncharged conducting sphere in uniform E0 zhat."""
    pts = np.asarray(points, dtype=np.float64)
    r = np.linalg.norm(pts, axis=1)
    n = pts / r[:, None]
    alpha = (float(radius) / r) ** 3
    zhat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cos_theta = n[:, 2]
    return float(E0) * (3.0 * alpha[:, None] * cos_theta[:, None] * n + (1.0 - alpha)[:, None] * zhat)


def analytic_conducting_sphere_potential(points: np.ndarray, radius: float, E0: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    r = np.linalg.norm(pts, axis=1)
    return -float(E0) * pts[:, 2] * (1.0 - (float(radius) / r) ** 3)


def rms(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def tangential_components(vec: np.ndarray, normals: np.ndarray) -> np.ndarray:
    dot = np.sum(vec * normals, axis=1)
    return vec - dot[:, None] * normals


def solve_equipotential_aux_sources(
    surface_points: np.ndarray,
    source_points: np.ndarray,
    phi_ext_surface: np.ndarray,
    net_charge: float,
    softening: float = 0.0,
) -> Tuple[np.ndarray, float, np.ndarray, float, float]:
    """Solve P q + phi_ext = Vf, sum(q)=net_charge."""
    P = potential_matrix(surface_points, source_points, softening=softening)
    n = P.shape[0]
    K = np.zeros((n + 1, n + 1), dtype=np.float64)
    K[:n, :n] = P
    K[:n, n] = -1.0
    K[n, :n] = 1.0
    rhs = np.zeros(n + 1, dtype=np.float64)
    rhs[:n] = -np.asarray(phi_ext_surface, dtype=np.float64)
    rhs[n] = float(net_charge)
    sol = np.linalg.solve(K, rhs)
    q = sol[:n]
    vf = float(sol[n])
    residual = P @ q + phi_ext_surface - vf

    # The raw KKT matrix mixes volts and coulombs. Scale it to report a useful
    # dimensionless conditioning diagnostic rather than a unit-dominated value.
    p_scale = max(float(np.max(np.abs(P))), 1.0)
    K_scaled = np.zeros_like(K)
    K_scaled[:n, :n] = P / p_scale
    K_scaled[:n, n] = -1.0
    K_scaled[n, :n] = 1.0 / max(float(n), 1.0)
    cond = float(np.linalg.cond(K_scaled))
    pcond = float(np.linalg.cond(P))
    return q, vf, residual, cond, pcond


def auxiliary_sigma_per_physical_area(normals: np.ndarray, radius: float, E0: float, source_offset: float, net_charge: float = 0.0) -> np.ndarray:
    """Analytic auxiliary-source coefficient q/A_R for an inward source sphere.

    The comparison uses the physical collocation patch area A_R = 4*pi*R^2/N
    because this is how the discrete q values are naturally reported in this
    diagnostic. For a source radius a=R-d, the dipolar coefficient is scaled by
    R/a. This is not the physical metal-surface charge density unless d=0.
    """
    a = float(radius) - float(source_offset)
    uniform = float(net_charge) / (4.0 * math.pi * float(radius) ** 2)
    return uniform + 3.0 * EPS0 * float(E0) * (float(radius) / a) * normals[:, 2]


def physical_sigma_reference(normals: np.ndarray, radius: float, E0: float, net_charge: float = 0.0) -> np.ndarray:
    uniform = float(net_charge) / (4.0 * math.pi * float(radius) ** 2)
    return uniform + 3.0 * EPS0 * float(E0) * normals[:, 2]


def solve_case_arrays(n_points: int, radius: float, E0: float, source_offset: float, net_charge: float, softening_frac: float):
    if source_offset <= 0.0 or source_offset >= radius:
        raise ValueError(f"source_offset must satisfy 0 < offset < radius; got {source_offset:g} for R={radius:g}")
    surf, normals, areas = fibonacci_sphere_points(n_points, radius)
    src = surf - float(source_offset) * normals
    phi_ext = -float(E0) * surf[:, 2]
    softening = float(softening_frac) * float(source_offset)
    q, vf, phi_residual, cond, pcond = solve_equipotential_aux_sources(
        surf, src, phi_ext, net_charge=net_charge, softening=softening
    )
    return surf, normals, areas, src, q, vf, phi_residual, cond, pcond, softening


def run_one_case(
    n_points: int,
    radius: float,
    E0: float,
    source_offset: float,
    eval_offset_fracs: Sequence[float],
    net_charge: float,
    softening_frac: float,
) -> dict:
    surf, normals, areas, src, q, vf, phi_residual, cond, pcond, softening = solve_case_arrays(
        n_points, radius, E0, source_offset, net_charge, softening_frac
    )

    E_ext = np.zeros_like(surf)
    E_ext[:, 2] = float(E0)
    E_aux_surface = point_charge_field(surf, src, q, softening=softening)
    E_total_surface = E_ext + E_aux_surface
    Et_total_surface = tangential_components(E_total_surface, normals)
    Et_inc_surface = tangential_components(E_ext, normals)
    En_total_surface = np.sum(E_total_surface * normals, axis=1)
    En_exact_surface = 3.0 * float(E0) * normals[:, 2]

    p_aux = np.sum(q[:, None] * src, axis=0)
    p_exact = np.array([0.0, 0.0, 4.0 * math.pi * EPS0 * float(E0) * float(radius) ** 3], dtype=np.float64)

    sigma_aux_model = q / areas
    sigma_aux_exact = auxiliary_sigma_per_physical_area(normals, radius, E0, source_offset, net_charge=net_charge)
    sigma_phys_exact = physical_sigma_reference(normals, radius, E0, net_charge=net_charge)

    row = {
        "mas_points": int(n_points),
        "radius_m": float(radius),
        "E0_V_per_m": float(E0),
        "source_offset_m": float(source_offset),
        "source_offset_over_R": float(source_offset / radius),
        "source_softening_m": float(softening),
        "net_charge_target_C": float(net_charge),
        "net_charge_solved_C": float(np.sum(q)),
        "floating_potential_V": float(vf),
        "potential_rms_residual_V": rms(phi_residual),
        "potential_max_abs_residual_V": float(np.max(np.abs(phi_residual))),
        "surface_total_tangential_E_rms_V_per_m": rms(np.linalg.norm(Et_total_surface, axis=1)),
        "surface_incident_tangential_E_rms_V_per_m": rms(np.linalg.norm(Et_inc_surface, axis=1)),
        "surface_total_tangential_rel_to_incident": rms(np.linalg.norm(Et_total_surface, axis=1)) / max(rms(np.linalg.norm(Et_inc_surface, axis=1)), 1e-300),
        "surface_normal_E_rms_rel_error": rms(En_total_surface - En_exact_surface) / max(rms(En_exact_surface), 1e-300),
        "auxiliary_sigma_per_physical_area_rms_C_per_m2": rms(sigma_aux_model),
        "auxiliary_sigma_per_physical_area_rel_error": rms(sigma_aux_model - sigma_aux_exact) / max(rms(sigma_aux_exact), 1e-300),
        "physical_surface_sigma_reference_rms_C_per_m2": rms(sigma_phys_exact),
        "auxiliary_to_physical_sigma_reference_rms_difference_rel": rms(sigma_aux_exact - sigma_phys_exact) / max(rms(sigma_phys_exact), 1e-300),
        "auxiliary_dipole_z_C_m": float(p_aux[2]),
        "analytic_dipole_z_C_m": float(p_exact[2]),
        "dipole_z_relative_error": float(abs(p_aux[2] - p_exact[2]) / max(abs(p_exact[2]), 1e-300)),
        "scaled_kkt_condition_number": cond,
        "potential_matrix_condition_number": pcond,
        "q_l2_C": rms(q),
        "q_max_abs_C": float(np.max(np.abs(q))),
    }

    for frac in eval_offset_fracs:
        rr = float(radius) * (1.0 + float(frac))
        probe = rr * normals
        E_model = np.zeros_like(probe)
        E_model[:, 2] = float(E0)
        E_model += point_charge_field(probe, src, q, softening=softening)
        E_exact = analytic_conducting_sphere_field(probe, radius, E0)
        phi_model = -float(E0) * probe[:, 2] + potential_matrix(probe, src, softening=softening) @ q
        phi_exact = analytic_conducting_sphere_potential(probe, radius, E0)
        Et_model = tangential_components(E_model, normals)
        Et_exact = tangential_components(E_exact, normals)
        key = f"offset_{frac:g}R"
        row[f"{key}_field_rms_rel_error"] = rms(np.linalg.norm(E_model - E_exact, axis=1)) / max(rms(np.linalg.norm(E_exact, axis=1)), 1e-300)
        row[f"{key}_tangential_rms_abs_error_V_per_m"] = rms(np.linalg.norm(Et_model - Et_exact, axis=1))
        row[f"{key}_tangential_rms_rel_error"] = rms(np.linalg.norm(Et_model - Et_exact, axis=1)) / max(rms(np.linalg.norm(Et_exact, axis=1)), 1e-300)
        row[f"{key}_potential_rms_abs_error_V"] = rms(phi_model - phi_exact)

    return row


def save_comparison_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _subset_for_offset(rows: Sequence[dict], off: float) -> List[dict]:
    return sorted([r for r in rows if abs(float(r["source_offset_over_R"]) - off) < 1e-10], key=lambda r: r["mas_points"])


def make_plots(
    out: Path,
    rows: Sequence[dict],
    eval_offset_fracs: Sequence[float],
    radius: float,
    E0: float,
    surface_plot_offset: float,
    net_charge: float,
    softening_frac: float,
    formats: Sequence[str],
    dpi: int = 300,
    show_physical_sigma_reference: bool = False,
) -> None:
    out = Path(out)
    if not rows:
        return

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "figure.dpi": int(dpi),
        "savefig.dpi": int(dpi),
    })
    offsets = sorted(set(round(float(r["source_offset_over_R"]), 12) for r in rows))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for off in offsets:
        subset = _subset_for_offset(rows, off)
        ax.loglog([r["mas_points"] for r in subset], [r["surface_total_tangential_rel_to_incident"] for r in subset], marker='o', label=f"d/R={off:.3g}")
    ax.set_xlabel("MAS collocation points")
    ax.set_ylabel("Surface tangential residual / incident tangential RMS")
    ax.set_title("Conducting sphere in a uniform electrostatic field")
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out / "case2a_sphere_surface_tangential_residual", formats, dpi)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for off in offsets:
        subset = _subset_for_offset(rows, off)
        ax.loglog([r["mas_points"] for r in subset], [r["auxiliary_sigma_per_physical_area_rel_error"] for r in subset], marker='o', label=f"d/R={off:.3g}")
    ax.set_xlabel("MAS collocation points")
    ax.set_ylabel("Auxiliary-source coefficient relative error")
    ax.set_title("Analytic auxiliary-source coefficient check")
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out / "case2a_sphere_auxiliary_sigma_error", formats, dpi)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for off in offsets:
        subset = _subset_for_offset(rows, off)
        ax.loglog([r["mas_points"] for r in subset], [r["dipole_z_relative_error"] for r in subset], marker='o', label=f"d/R={off:.3g}")
    ax.set_xlabel("MAS collocation points")
    ax.set_ylabel("Induced dipole relative error")
    ax.set_title("Analytic induced-dipole check")
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out / "case2a_sphere_dipole_error", formats, dpi)

    positive_eval = [float(x) for x in eval_offset_fracs if float(x) > 0.0]
    if positive_eval:
        frac = positive_eval[0]
        key = f"offset_{frac:g}R_field_rms_rel_error"
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for off in offsets:
            subset = _subset_for_offset(rows, off)
            ax.loglog([r["mas_points"] for r in subset], [r[key] for r in subset], marker='o', label=f"d/R={off:.3g}")
        ax.set_xlabel("MAS collocation points")
        ax.set_ylabel(f"Field RMS relative error at r/R={1.0 + frac:g}")
        ax.set_title("Off-surface analytic field comparison")
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        save_figure(fig, out / "case2a_sphere_offsurface_field_error", formats, dpi)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for off in offsets:
        subset = _subset_for_offset(rows, off)
        ax.loglog([r["mas_points"] for r in subset], [max(r["potential_rms_residual_V"], 1e-30) for r in subset], marker='o', label=f"d/R={off:.3g}")
    ax.set_xlabel("MAS collocation points")
    ax.set_ylabel("Surface potential RMS residual (V)")
    ax.set_title("Equipotential residual of the KKT solve")
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out / "case2a_sphere_equipotential_residual", formats, dpi)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for off in offsets:
        subset = _subset_for_offset(rows, off)
        ax.loglog([r["mas_points"] for r in subset], [r["scaled_kkt_condition_number"] for r in subset], marker='o', label=f"d/R={off:.3g}")
    ax.set_xlabel("MAS collocation points")
    ax.set_ylabel("Scaled KKT condition number")
    ax.set_title("Conditioning of the surface-charge solve")
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out / "case2a_sphere_scaled_kkt_condition", formats, dpi)

    # Distribution plot for one representative offset, using the largest N.
    available_offsets = np.array(offsets, dtype=float)
    target = float(surface_plot_offset / radius)
    chosen_off = float(available_offsets[np.argmin(np.abs(available_offsets - target))])
    subset = _subset_for_offset(rows, chosen_off)
    n_plot = int(max(r["mas_points"] for r in subset))
    surf, normals, areas, src, q, vf, phi_residual, cond, pcond, softening = solve_case_arrays(
        n_plot, radius, E0, chosen_off * radius, net_charge, softening_frac
    )
    sigma_model = q / areas
    sigma_aux = auxiliary_sigma_per_physical_area(normals, radius, E0, chosen_off * radius, net_charge=net_charge)
    sigma_phys = physical_sigma_reference(normals, radius, E0, net_charge=net_charge)
    order = np.argsort(normals[:, 2])
    scale = max(abs(EPS0 * float(E0)), 1e-300)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(normals[order, 2], sigma_model[order] / scale, '.', markersize=3.5, label="Solved auxiliary coefficient")
    ax.plot(normals[order, 2], sigma_aux[order] / scale, '-', linewidth=1.6, label="Analytic auxiliary coefficient")
    if show_physical_sigma_reference:
        ax.plot(normals[order, 2], sigma_phys[order] / scale, '--', linewidth=1.1, label="Physical surface reference")
    ax.set_xlabel(r"$\cos\theta$")
    ax.set_ylabel(r"Coefficient / $(\epsilon_0 E_0)$")
    ax.set_title(f"Auxiliary-source distribution, N={n_plot}, d/R={chosen_off:.3g}")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out / "case2a_sphere_auxiliary_sigma_distribution", formats, dpi)


def write_ascii_sphere_stl(path: Path, radius: float, n_theta: int = 24, n_phi: int = 48) -> None:
    path = Path(path)
    tris = []

    def sph(th: float, ph: float) -> np.ndarray:
        return float(radius) * np.array([
            math.sin(th) * math.cos(ph),
            math.sin(th) * math.sin(ph),
            math.cos(th),
        ], dtype=np.float64)

    for i in range(n_theta):
        th0 = math.pi * i / n_theta
        th1 = math.pi * (i + 1) / n_theta
        for j in range(n_phi):
            ph0 = 2.0 * math.pi * j / n_phi
            ph1 = 2.0 * math.pi * (j + 1) / n_phi
            p00 = sph(th0, ph0)
            p01 = sph(th0, ph1)
            p10 = sph(th1, ph0)
            p11 = sph(th1, ph1)
            if i == 0:
                tris.append((p00, p10, p11))
            elif i == n_theta - 1:
                tris.append((p00, p10, p01))
            else:
                tris.append((p00, p10, p11))
                tris.append((p00, p11, p01))

    def normal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm == 0.0:
            return np.array([0.0, 0.0, 1.0])
        return n / norm

    with path.open('w', encoding='utf-8') as f:
        f.write('solid case2_uniform_field_sphere\n')
        for a, b, c in tris:
            nn = normal(a, b, c)
            f.write(f'  facet normal {nn[0]:.16e} {nn[1]:.16e} {nn[2]:.16e}\n')
            f.write('    outer loop\n')
            for p in (a, b, c):
                f.write(f'      vertex {p[0]:.16e} {p[1]:.16e} {p[2]:.16e}\n')
            f.write('    endloop\n')
            f.write('  endfacet\n')
        f.write('endsolid case2_uniform_field_sphere\n')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--radius", type=float, default=0.004, help="Sphere radius in meters.")
    p.add_argument("--E0", type=float, default=1.0, help="Uniform incident electric field amplitude in V/m along +z.")
    p.add_argument("--mas-points", default="96,192,320", help="Comma-separated MAS/collocation point counts.")
    p.add_argument(
        "--source-offsets",
        default="0.10R,0.20R,0.35R",
        help="Comma-separated auxiliary source offsets. Values may be meters or fractions like 0.20R.",
    )
    p.add_argument(
        "--eval-offset-fracs",
        default="0.05,0.10",
        help="Comma-separated off-surface evaluation offsets as fractions of R. Use positive values for off-surface field checks.",
    )
    p.add_argument("--net-charge", type=float, default=0.0, help="Prescribed net conductor charge in C. Default 0 for induced-charge test.")
    p.add_argument(
        "--source-softening-frac",
        type=float,
        default=0.0,
        help="Optional source softening as a fraction of source_offset. Keep 0 for the strict electrostatic diagnostic.",
    )
    p.add_argument("--surface-plot-offset", default="0.35R", help="Representative d/R used for the auxiliary-source distribution figure.")
    p.add_argument("--show-physical-sigma-reference", action="store_true", help="Overlay the physical metal-surface sigma reference on the distribution plot.")
    p.add_argument("--out", default="article_cases_output/case2a_sphere_uniform_field")
    p.add_argument("--figure-dpi", type=int, default=300)
    p.add_argument("--publication-formats", default="png,pdf", help="Comma-separated figure formats, e.g. png,pdf or png,pdf,svg.")
    p.add_argument("--write-stl", action="store_true", help="Also write a visual sphere STL to the output directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    radius = float(args.radius)
    mas_points = _parse_csv_ints(args.mas_points)
    source_offsets = _parse_csv_floats_or_radius_fracs(args.source_offsets, radius)
    eval_offset_fracs = [float(x.strip()) for x in str(args.eval_offset_fracs).split(',') if x.strip()]
    surface_plot_offset = _parse_one_float_or_radius_frac(args.surface_plot_offset, radius)
    formats = publication_formats(args.publication_formats)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for n in mas_points:
        for offset in source_offsets:
            print(f"[case2a-sphere] N={n}; d/R={offset/radius:.4g}; d={offset:.6e} m")
            row = run_one_case(
                n_points=int(n),
                radius=radius,
                E0=float(args.E0),
                source_offset=float(offset),
                eval_offset_fracs=eval_offset_fracs,
                net_charge=float(args.net_charge),
                softening_frac=float(args.source_softening_frac),
            )
            rows.append(row)
            sub = out / f"N{int(n)}_d_{offset/radius:.4g}R".replace('.', 'p')
            sub.mkdir(exist_ok=True)
            (sub / "case2a_sphere_uniform_field_summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            print(
                "[case2a-sphere] "
                f"Et_rel={row['surface_total_tangential_rel_to_incident']:.3e}; "
                f"aux_sigma_rel={row['auxiliary_sigma_per_physical_area_rel_error']:.3e}; "
                f"dipole_rel={row['dipole_z_relative_error']:.3e}; "
                f"phi_rms={row['potential_rms_residual_V']:.3e} V; "
                f"scaled_cond={row['scaled_kkt_condition_number']:.3e}"
            )

    save_comparison_csv(out / "case2a_sphere_uniform_field_comparison.csv", rows)
    (out / "case2a_sphere_uniform_field_results.json").write_text(
        json.dumps({"model": "conducting sphere in uniform electrostatic field", "results": rows}, indent=2),
        encoding="utf-8",
    )
    (out / "case2a_run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    make_plots(
        out,
        rows,
        eval_offset_fracs=eval_offset_fracs,
        radius=radius,
        E0=float(args.E0),
        surface_plot_offset=surface_plot_offset,
        net_charge=float(args.net_charge),
        softening_frac=float(args.source_softening_frac),
        formats=formats,
        dpi=int(args.figure_dpi),
        show_physical_sigma_reference=bool(args.show_physical_sigma_reference),
    )
    if args.write_stl:
        write_ascii_sphere_stl(out / "case2a_sphere_uniform_field_sphere.stl", radius=radius)
    print(f"Finished: {out}")


if __name__ == "__main__":
    main()
