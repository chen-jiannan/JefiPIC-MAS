"""Case 2b (Route 2): charge-conserving floating-conductor boundary verification with MAS surface charge.

This replaces the earlier diagnostic-only absorption model for the physical
sheath validation step.  It keeps the validated PIC plasma volume sources
(rho_p, J_p) and adds an independent conductor surface-charge source sigma_s
through the MAS point-source channel:

    particle collection -> Q_conductor(t)
    Q_conductor constraint + equal-potential metal boundary -> sigma_s(t), V_f(t)
    sigma_s(t) -> EMsolver point_rho -> total field -> next PIC push

The tangential electric field is now an independent diagnostic only; it is no
longer minimized to define the conductor charge distribution.

The benchmark is intentionally quasi-electrostatic and uses one connected
conductor. It verifies the particle-collection, net-charge conservation, and
equipotential redistribution closure before the same boundary model is used in
the driven two-arm RF Case 3 demonstration. It should not be interpreted as a
fully converged quantitative sheath benchmark unless a separate long ion-time
scale and convergence study is performed.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import csv
import json
import math
import time

import numpy as np
import matplotlib.pyplot as plt

import _jefipic_mas_case_common as common
from _jefipic_mas_case_common import BaseConfig

EPS0 = common.EPS0
E_CHARGE = common.E_CHARGE
M_P = common.M_P

PRESETS = {
    "smoke": dict(steps=200, particles=4000, grid_n=17, mas_points=96, diag_stride=5),
    "draft": dict(steps=5000, particles=40000, grid_n=25, mas_points=128, diag_stride=25),
    "paper": dict(steps=60000, particles=120000, grid_n=49, mas_points=192, diag_stride=100),
}



_SURFACE_CHARGE_RIDGE_HOOK_INSTALLED = False


def _parse_positive_float(text: str, *, name: str) -> float:
    value = float(text)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite value; got {text!r}.")
    return value


def parse_mas_source_offset(value, radius_m: float) -> float:
    """Parse MAS source offset in metres or as a radius fraction such as '0.20R'."""
    if value is None:
        raise ValueError("parse_mas_source_offset requires a value.")
    if isinstance(value, (int, float)):
        out = float(value)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("--mas-source-offset cannot be empty.")
        if text.lower().endswith("r"):
            factor_text = text[:-1].strip()
            if not factor_text:
                raise ValueError("Use forms such as --mas-source-offset 0.20R, not just R.")
            out = float(factor_text) * float(radius_m)
        else:
            out = float(text)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"--mas-source-offset must be positive; got {value!r} -> {out!r}.")
    return out


def _cfg_extra_names():
    return (
        "case2_geometry",
        "sphere_radius",
        "sphere_n_lat",
        "sphere_n_lon",
        "surface_charge_ridge",
        "surface_charge_regularization",
        "surface_charge_ridge_hook",
    )


def cfg_to_serializable_dict(cfg: BaseConfig) -> dict:
    """Return dataclass config plus dynamic Case-2 extension fields."""
    data = asdict(cfg)
    for name in _cfg_extra_names():
        if hasattr(cfg, name):
            value = getattr(cfg, name)
            if isinstance(value, np.generic):
                value = value.item()
            data[name] = value
    return data


def save_case2_config(output_dir: Path, cfg: BaseConfig, name: str = "run_config.json") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / name).write_text(
        json.dumps(cfg_to_serializable_dict(cfg), indent=2, default=float),
        encoding="utf-8",
    )


def _state_float(state: dict, *keys: str, default=float("nan")) -> float:
    for key in keys:
        if key in state and state[key] is not None:
            try:
                return float(state[key])
            except (TypeError, ValueError):
                continue
    return float(default)





def volume_charge_density_code_to_si_compat(bridge, rho_code):
    """Convert JefiPIC volume charge density from code units to SI C/m^3.

    Some JefiPICUnitBridge versions used in older local trees do not provide
    volume_charge_density_code_to_si().  For a volume charge density,
        rho_code = rho_SI * Q / L^3,
    where Q and L are SI->code conversion factors.  Therefore,
        rho_SI = rho_code * L^3 / Q
               = rho_code * TO_Q / TO_L^3.
    """
    if hasattr(bridge, "volume_charge_density_code_to_si"):
        return bridge.volume_charge_density_code_to_si(rho_code)

    arr = np.asarray(rho_code, dtype=np.float64)

    if hasattr(bridge, "TO_Q") and hasattr(bridge, "TO_L"):
        return arr * (float(bridge.TO_Q) / (float(bridge.TO_L) ** 3))

    if hasattr(bridge, "Q") and hasattr(bridge, "L"):
        return arr * ((float(bridge.L) ** 3) / float(bridge.Q))

    # Last-resort SI-mode fallback.  If neither set of conversion constants is
    # available, there is no reliable way to convert non-SI code units.
    if getattr(bridge, "is_si", False):
        return arr

    raise AttributeError(
        "JefiPICUnitBridge lacks volume_charge_density_code_to_si and also "
        "does not expose TO_Q/TO_L or Q/L conversion constants."
    )

def _get_cp():
    cp = getattr(common, "cp", None)
    if cp is None:
        try:
            import cupy as cp  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "CuPy is required for local Case-2 particle collection/recycling "
                "because _jefipic_mas_case_common.py does not provide the newer helpers."
            ) from exc
    return cp


def _get_unit_bridge(PIC, cfg):
    bridge = getattr(PIC, "_jefipic_unit_bridge", None)
    if bridge is None:
        bridge = common.get_unit_bridge(cfg)
    return bridge


def _as_si_positions(pos, bridge):
    if bridge is not None and not getattr(bridge, "is_si", True):
        return pos * bridge.TO_L
    return pos


def _to_cpu_indices(mask):
    xp = _get_cp()
    idx = xp.where(mask)[0]
    return idx.get() if hasattr(idx, "get") else np.asarray(idx)


def _species_charge_si(species_id: int) -> float:
    # The article-case PIC builder uses species 0 = electron and species 1 = ion.
    return -E_CHARGE if int(species_id) == 0 else E_CHARGE


def _local_conductor_mask(pos, cfg: BaseConfig, bridge=None):
    """CuPy boolean mask for particles inside the Case-2 conductor."""
    cp = _get_cp()
    pos_si = _as_si_positions(pos, bridge)
    geometry = str(getattr(cfg, "case2_geometry", "connected_cylinder")).lower()

    if geometry == "sphere":
        radius = float(getattr(cfg, "sphere_radius", getattr(cfg, "radius", cfg.radius)))
        r2 = pos_si[:, 0] ** 2 + pos_si[:, 1] ** 2 + pos_si[:, 2] ** 2
        return r2 <= radius * radius

    r = cp.sqrt(pos_si[:, 0] ** 2 + pos_si[:, 1] ** 2)
    zabs = cp.abs(pos_si[:, 2])
    return (
        (r <= float(cfg.radius))
        & (zabs >= 0.5 * float(getattr(cfg, "feed_gap", 0.0)))
        & (zabs <= 0.5 * float(cfg.total_length))
    )


def _sample_replacement_particles(n: int, species_id: int, cfg: BaseConfig, rng: np.random.Generator):
    if int(species_id) == 0:
        T = float(cfg.Te_eV)
        mass = float(getattr(common, "M_E", 9.1093837015e-31))
    else:
        T = float(cfg.Ti_eV)
        mass = float(cfg.ion_mass)

    mode = str(getattr(cfg, "particle_reinjection_mode", "boundary")).lower()
    if mode == "boundary":
        sampler = getattr(common, "sample_boundary_reservoir_positions_velocities", None)
        if sampler is None:
            raise RuntimeError(
                "_jefipic_mas_case_common.py lacks sample_boundary_reservoir_positions_velocities(); "
                "cannot locally recycle collected/open-boundary particles."
            )
        return sampler(n, cfg, T, mass, rng)

    if mode == "uniform":
        pos_sampler = getattr(common, "sample_positions", None)
        vel_sampler = getattr(common, "sample_velocities", None)
        if pos_sampler is None or vel_sampler is None:
            raise RuntimeError(
                "_jefipic_mas_case_common.py lacks sample_positions()/sample_velocities(); "
                "cannot locally recycle collected/open-boundary particles."
            )
        pos = pos_sampler(
            n,
            cfg,
            rng,
            exclusion_debye=getattr(cfg, "boundary_reinjection_antenna_exclusion_debye", 0.0),
        )
        vel = vel_sampler(n, T, mass, rng)
        return pos, vel

    raise ValueError(f"Unknown particle_reinjection_mode={mode!r}; expected 'boundary' or 'uniform'.")


def _respawn_species_local(PIC, species_id: int, idx_cpu: np.ndarray, cfg: BaseConfig, rng: np.random.Generator) -> int:
    """Respawn particles and synchronize prev_pos to avoid artificial long-step currents."""
    if idx_cpu is None or len(idx_cpu) == 0:
        return 0

    cp = _get_cp()
    bridge = _get_unit_bridge(PIC, cfg)
    idx_cpu = np.asarray(idx_cpu, dtype=np.int64)
    new_pos_si, new_v_si = _sample_replacement_particles(len(idx_cpu), species_id, cfg, rng)
    new_pos_code = bridge.length_si_to_code(new_pos_si)
    new_v_code = bridge.velocity_si_to_code(new_v_si)

    idx_gpu = cp.asarray(idx_cpu, dtype=cp.int64)
    pos_gpu = cp.asarray(new_pos_code)
    vel_gpu = cp.asarray(new_v_code)

    PIC.pos[species_id][idx_gpu] = pos_gpu
    if hasattr(PIC, "velocity") and species_id in PIC.velocity:
        PIC.velocity[species_id][idx_gpu] = vel_gpu
    if hasattr(PIC, "u") and species_id in PIC.u:
        PIC.u[species_id][idx_gpu] = vel_gpu
    if hasattr(PIC, "prev_pos") and species_id in PIC.prev_pos:
        PIC.prev_pos[species_id][idx_gpu] = pos_gpu
    if hasattr(PIC, "position_old") and species_id in PIC.position_old:
        PIC.position_old[species_id][idx_gpu] = pos_gpu

    return int(len(idx_cpu))


def collect_particles_to_floating_conductor(PIC, cfg: BaseConfig, rng: np.random.Generator, macro_weight: float):
    """Compatibility wrapper for floating-conductor particle collection.

    Newer _jefipic_mas_case_common.py versions provide this helper.  Some local
    trees only have the older common.py, which caused:
        AttributeError: module '_jefipic_mas_case_common' has no attribute
        'collect_particles_to_floating_conductor'

    This local implementation keeps the benchmark runnable without editing
    common.py.  Collected particles are replaced from the configured reservoir,
    and the returned dQ is the signed charge added to the floating conductor.
    """
    helper = getattr(common, "collect_particles_to_floating_conductor", None)
    if helper is not None:
        return helper(PIC, cfg, rng, macro_weight)

    counts = {}
    dQ = 0.0
    bridge = _get_unit_bridge(PIC, cfg)

    for species_id in list(PIC.pos.keys()):
        mask = _local_conductor_mask(PIC.pos[species_id], cfg, bridge=bridge)
        idx_cpu = _to_cpu_indices(mask)
        n_hit = int(len(idx_cpu))
        counts[int(species_id)] = n_hit
        if n_hit > 0:
            _respawn_species_local(PIC, species_id, idx_cpu, cfg, rng)
            dQ += _species_charge_si(species_id) * float(macro_weight) * n_hit

    return counts, float(dQ)


def recycle_particles_at_open_reservoir(PIC, cfg: BaseConfig, rng: np.random.Generator, macro_weight: float):
    """Compatibility wrapper for open-reservoir boundary recycling.

    Returns (counts, dQ_out, dQ_in).  dQ_out is the signed charge carried out of
    the computational domain; dQ_in is the signed charge re-injected from the
    external reservoir.  For same-species replacement they are equal.
    """
    helper = getattr(common, "recycle_particles_at_open_reservoir", None)
    if helper is not None:
        return helper(PIC, cfg, rng, macro_weight)

    bridge = _get_unit_bridge(PIC, cfg)
    half_si = 0.5 * float(cfg.domain_size)

    counts = {}
    dQ_out = 0.0
    dQ_in = 0.0

    for species_id in list(PIC.pos.keys()):
        pos_si = _as_si_positions(PIC.pos[species_id], bridge)
        outside = (
            (pos_si[:, 0] < -half_si) | (pos_si[:, 0] > half_si)
            | (pos_si[:, 1] < -half_si) | (pos_si[:, 1] > half_si)
            | (pos_si[:, 2] < -half_si) | (pos_si[:, 2] > half_si)
        )
        idx_cpu = _to_cpu_indices(outside)
        n_out = int(len(idx_cpu))
        counts[int(species_id)] = n_out
        if n_out > 0:
            q = _species_charge_si(species_id) * float(macro_weight) * n_out
            dQ_out += q
            _respawn_species_local(PIC, species_id, idx_cpu, cfg, rng)
            dQ_in += q

    return counts, float(dQ_out), float(dQ_in)


def _install_surface_charge_ridge_hook(cfg: BaseConfig) -> None:
    """Inject surface_charge_ridge into MAS creation without editing common.py."""
    global _SURFACE_CHARGE_RIDGE_HOOK_INSTALLED

    ridge = float(getattr(cfg, "surface_charge_ridge", 0.0))
    setattr(common, "_case2_surface_charge_ridge_value", ridge)

    if _SURFACE_CHARGE_RIDGE_HOOK_INSTALLED:
        return

    original = getattr(common, "create_mas_antenna_simulation", None)
    if original is None:
        raise RuntimeError(
            "article_cases/_jefipic_mas_case_common.py does not expose "
            "create_mas_antenna_simulation; cannot inject surface_charge_ridge."
        )

    def _wrapped_create_mas_antenna_simulation(geometry_params, simulation_params):
        params = dict(simulation_params)
        ridge_value = float(getattr(common, "_case2_surface_charge_ridge_value", 0.0))
        params["surface_charge_ridge"] = ridge_value
        params["surface_charge_regularization"] = ridge_value
        return original(geometry_params, params)

    common.create_mas_antenna_simulation = _wrapped_create_mas_antenna_simulation
    _SURFACE_CHARGE_RIDGE_HOOK_INSTALLED = True


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="Alias for --preset smoke.")
    p.add_argument("--preset", choices=tuple(PRESETS), default="draft")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--particles", type=int, default=None)
    p.add_argument("--grid-n", type=int, default=None)
    p.add_argument("--mas-points", type=int, default=None)
    p.add_argument("--diag-stride", type=int, default=None)
    p.add_argument(
        "--diagonal-loading",
        type=float,
        default=None,
        help="Override cfg.diagonal_loading for the MAS/KKT surface-charge solve.",
    )
    p.add_argument(
        "--surface-charge-ridge",
        type=float,
        default=0.0,
        help=(
            "Ridge regularization strength for the MAS floating surface-charge solve. "
            "This script forwards the value into the MAS simulation parameters."
        ),
    )
    p.add_argument(
        "--mas-source-offset",
        default=None,
        help="Override MAS source offset. Use SI metres, e.g. 8e-4, or radius fraction, e.g. 0.20R.",
    )
    p.add_argument(
        "--case2-geometry",
        choices=("connected_cylinder", "sphere"),
        default=None,
        help="Optional geometry override for this base benchmark.",
    )
    p.add_argument("--sphere-radius", type=float, default=None, help="Sphere radius in metres; implies --case2-geometry sphere.")
    p.add_argument("--sphere-n-lat", type=int, default=24, help="Latitude divisions for generated sphere STL.")
    p.add_argument("--sphere-n-lon", type=int, default=None, help="Longitude divisions for generated sphere STL.")
    p.add_argument("--dt", type=float, default=2.0e-9)
    p.add_argument("--n0", type=float, default=1.0e11)
    p.add_argument("--Te-eV", type=float, default=0.2)
    p.add_argument("--Ti-eV", type=float, default=0.05)
    p.add_argument("--ion-mass-amu", type=float, default=16.0)
    p.add_argument("--domain-size", type=float, default=0.18)
    p.add_argument("--radius-debye", type=float, default=1.0)
    p.add_argument("--length-debye", type=float, default=4.0)
    p.add_argument("--particle-order", type=int, choices=(1, 2), default=1)
    p.add_argument("--field-order", type=int, choices=(1, 2), default=1)
    p.add_argument("--jefipic-unit-mode", choices=("LHQCD", "SI"), default="LHQCD")
    p.add_argument(
        "--jefipic-kernel-mode", choices=("atomic",), default="atomic",
        help="Case 2 surface-charge physics is fixed to the validated atomic kernel path."
    )
    p.add_argument("--initial-exclusion-debye", type=float, default=0.0)
    p.add_argument("--boundary-exclusion-debye", type=float, default=0.0)
    # Retained for command-line compatibility with the previous tangential-E
    # prototype; the equipotential KKT closure does not use this parameter.
    p.add_argument("--charge-regularization", type=float, default=1.0e-10, help=argparse.SUPPRESS)
    p.add_argument("--out", default=None)
    p.add_argument("--figure-dpi", type=int, default=300)
    p.add_argument("--publication-formats", default="png,pdf",
                   help="Comma-separated figure formats, e.g. png,svg,pdf.")
    p.add_argument(
        "--current-average-window",
        type=int,
        default=25,
        help=(
            "Number of diagnostic records used in the centered finite-window "
            "collection-current average. Use an odd value such as 11, 21, or 25."
        ),
    )
    return p.parse_args()


def ion_plasma_period(n0: float, ion_mass: float) -> float:
    omega_pi = math.sqrt(float(n0) * E_CHARGE * E_CHARGE / (EPS0 * float(ion_mass)))
    return 2.0 * math.pi / omega_pi


def configure(args):
    name = "smoke" if args.quick else args.preset
    selected = PRESETS[name]
    cfg = BaseConfig()
    cfg.seed = 17
    cfg.n0 = float(args.n0)
    cfg.Te_eV = float(args.Te_eV)
    cfg.Ti_eV = float(args.Ti_eV)
    cfg.ion_mass = float(args.ion_mass_amu) * M_P
    cfg.domain_size = float(args.domain_size)
    cfg.dt = float(args.dt)
    cfg.steps = int(selected["steps"] if args.steps is None else args.steps)
    cfg.particles_per_species = int(selected["particles"] if args.particles is None else args.particles)
    cfg.grid_n = int(selected["grid_n"] if args.grid_n is None else args.grid_n)
    cfg.max_mas_points = int(selected["mas_points"] if args.mas_points is None else args.mas_points)
    cfg.diag_stride = int(selected["diag_stride"] if args.diag_stride is None else args.diag_stride)
    if args.diagonal_loading is not None:
        cfg.diagonal_loading = float(args.diagonal_loading)
    if float(args.surface_charge_ridge) < 0.0:
        raise ValueError("--surface-charge-ridge must be non-negative.")
    cfg.surface_charge_ridge = float(args.surface_charge_ridge)
    cfg.surface_charge_regularization = float(args.surface_charge_ridge)
    cfg.surface_charge_ridge_hook = "case2_local_create_mas_wrapper"
    cfg.len_time_snapshots = max(64, min(512, cfg.steps + 1))
    cfg.patch_min_points = 12
    cfg.particle_order = int(args.particle_order)
    cfg.field_order = int(args.field_order)
    cfg.jefipic_unit_mode = str(args.jefipic_unit_mode)
    # Observation mode previously produced corrupt/zero-hit behavior in this
    # Route-2 point-rho path; retain only the validated atomic path.
    cfg.jefipic_kernel_mode = "atomic"

    lamD = common.debye_length(cfg.n0, cfg.Te_eV)
    cfg.radius = float(args.radius_debye) * lamD
    cfg.total_length = float(args.length_debye) * lamD
    cfg.feed_gap = 0.0  # a connected conductor
    cfg.n_phi = 36
    cfg.n_z_per_arm = 18
    cfg.n_r_cap = 5

    # Optional geometry override.  The default remains the connected finite
    # cylinder.  ``--sphere-radius`` is included so this base script can also be
    # driven directly by spherical Case-2 adapter/sweep commands.
    if args.case2_geometry is not None:
        cfg.case2_geometry = str(args.case2_geometry)
    if args.sphere_radius is not None:
        sphere_radius = _parse_positive_float(str(args.sphere_radius), name="--sphere-radius")
        cfg.case2_geometry = "sphere"
        cfg.sphere_radius = sphere_radius
        cfg.radius = sphere_radius
        cfg.total_length = 2.0 * sphere_radius
    if getattr(cfg, "case2_geometry", "connected_cylinder") == "sphere":
        cfg.sphere_radius = float(getattr(cfg, "sphere_radius", cfg.radius))
        cfg.sphere_n_lat = int(args.sphere_n_lat)
        cfg.sphere_n_lon = int(args.sphere_n_lon) if args.sphere_n_lon is not None else max(8, int(cfg.n_phi))

    cfg.mas_source_offset = min(0.25 * cfg.radius, 0.25 * lamD)
    if args.mas_source_offset is not None:
        cfg.mas_source_offset = parse_mas_source_offset(args.mas_source_offset, cfg.radius)

    cfg.current_amplitude = 0.0
    cfg.f_drive = 1.0
    cfg.ramp_periods = 0.0
    cfg.quasi_electrostatic = True
    cfg.spatial_truncation_debye = 0.0
    cfg.taper_width_debye = 0.0
    cfg.taper = "none"
    cfg.point_radius = None

    # Particle hits are handled by collect_particles_to_floating_conductor,
    # not by the diagnostic-only common absorption loop.
    cfg.absorb_particles_on_antenna = False
    cfg.particle_reinjection_mode = "boundary"
    # Physical Case 2 model: the far plasma is an external open reservoir, not
    # a reflecting box.  Collected particles and particles escaping the outer
    # domain are supplied/recycled through inward boundary samples.
    cfg.use_periodic = 0
    cfg.use_reflective = 0
    cfg.outer_boundary_reservoir_recycle = True
    cfg.initial_antenna_exclusion_debye = float(args.initial_exclusion_debye)
    cfg.boundary_reinjection_antenna_exclusion_debye = float(args.boundary_exclusion_debye)
    cfg.quiet_start_pairs = True
    cfg.velocity_quiet_start_pairs = True
    cfg.center_initial_velocities = True
    return cfg, name


def _write_case2_uv_sphere_stl(out_path: Path, radius: float, n_lat: int = 24, n_lon: int = 48) -> Path:
    """Write a UV-sphere STL centered at the origin.

    This helper is used only when cfg.case2_geometry == "sphere".  The filename
    is intentionally kept as requested by the caller, even if the filename still
    says "cylinder", because downstream Case2 code expects that path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    radius = float(radius)
    n_lat = int(n_lat)
    n_lon = int(n_lon)
    if radius <= 0.0:
        raise ValueError("Sphere radius must be positive.")
    if n_lat < 4 or n_lon < 8:
        raise ValueError("Use n_lat >= 4 and n_lon >= 8 for the sphere STL.")

    def vertex(i_lat: int, i_lon: int) -> np.ndarray:
        theta = np.pi * i_lat / n_lat
        phi = 2.0 * np.pi * (i_lon % n_lon) / n_lon
        st = np.sin(theta)
        return radius * np.array(
            [st * np.cos(phi), st * np.sin(phi), np.cos(theta)],
            dtype=np.float64,
        )

    def normal_of(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn <= 1.0e-300:
            n = a + b + c
            nn = np.linalg.norm(n)
        if nn <= 1.0e-300:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return n / nn

    tris = []
    for i in range(n_lat):
        for j in range(n_lon):
            if i == 0:
                tris.append((vertex(0, j), vertex(1, j), vertex(1, j + 1)))
            elif i == n_lat - 1:
                tris.append((vertex(i, j), vertex(n_lat, j), vertex(i, j + 1)))
            else:
                a = vertex(i, j)
                b = vertex(i + 1, j)
                c = vertex(i + 1, j + 1)
                d = vertex(i, j + 1)
                tris.append((a, b, c))
                tris.append((a, c, d))

    with out_path.open("w", encoding="utf-8") as f:
        f.write("solid case2_spherical_conductor\n")
        for a, b, c in tris:
            n = normal_of(a, b, c)
            centroid = (a + b + c) / 3.0
            if float(np.dot(n, centroid)) < 0.0:
                b, c = c, b
                n = normal_of(a, b, c)

            f.write(f"  facet normal {n[0]:.16e} {n[1]:.16e} {n[2]:.16e}\n")
            f.write("    outer loop\n")
            for p in (a, b, c):
                f.write(f"      vertex {p[0]:.16e} {p[1]:.16e} {p[2]:.16e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid case2_spherical_conductor\n")

    return out_path


def make_connected_cylinder_stl(cfg: BaseConfig, out_path: Path) -> Path:
    """Write the connected conductor STL for Case2.

    Default behavior is unchanged: finite connected cylinder.

    If cfg.case2_geometry == "sphere", this writes a spherical conductor instead.
    This preserves the old cylinder model for all existing Case2 runs while
    allowing the spherical validation adapter to use the same base benchmark.
    """
    geometry = str(getattr(cfg, "case2_geometry", "connected_cylinder")).lower()
    if geometry == "sphere":
        sphere_radius = float(getattr(cfg, "sphere_radius", getattr(cfg, "radius", cfg.radius)))
        sphere_n_lat = int(getattr(cfg, "sphere_n_lat", 24))
        sphere_n_lon = int(getattr(cfg, "sphere_n_lon", max(8, int(getattr(cfg, "n_phi", 48)))))

        print(
            "[case2-route2-floating] Writing SPHERE conductor STL: "
            f"radius={sphere_radius:.6e} m; mesh={sphere_n_lat}x{sphere_n_lon}; path={out_path}"
        )
        return _write_case2_uv_sphere_stl(
            out_path,
            radius=sphere_radius,
            n_lat=sphere_n_lat,
            n_lon=sphere_n_lon,
        )

    tris = []
    p0 = np.array([0.0, 0.0, -0.5 * cfg.total_length], dtype=float)
    p1 = np.array([0.0, 0.0, +0.5 * cfg.total_length], dtype=float)
    common.add_cylinder_between(
        tris, p0, p1, cfg.radius,
        n_phi=cfg.n_phi,
        n_len=max(8, 2 * cfg.n_z_per_arm),
        cap0=True,
        cap1=True,
        n_r_cap=cfg.n_r_cap,
    )
    return common.write_ascii_stl(tris, out_path, name="floating_connected_cylinder")


def publication_formats(text: str):
    allowed = {"png", "svg", "pdf"}
    values = []
    for value in str(text).split(","):
        value = value.strip().lower()
        if value and value in allowed and value not in values:
            values.append(value)
    return values or ["png", "pdf"]


def save_figure(fig, base: Path, formats, dpi: int):
    for ext in formats:
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = int(dpi)
        fig.savefig(base.with_suffix("." + ext), **kwargs)
    plt.close(fig)


def centered_window_slope(time_s, cumulative_quantity, window_records: int):
    """Return a centered finite-window slope d(cumulative_quantity)/dt.

    This is intended for event-driven particle collection. The instantaneous
    step current contains narrow macroparticle spikes, whereas the centered
    slope of the cumulative collected charge is a smoother publication-quality
    diagnostic. The input arrays are the diagnostic-history samples, not every
    particle step.
    """
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(cumulative_quantity, dtype=float)
    n = int(t.size)
    out = np.full(n, np.nan, dtype=float)
    if n < 2:
        return out
    w = max(3, int(window_records))
    if w % 2 == 0:
        w += 1
    half = max(1, w // 2)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n - 1, i + half)
        if hi <= lo:
            lo = max(0, i - 1)
            hi = min(n - 1, i + 1)
        dt = t[hi] - t[lo]
        if np.isfinite(dt) and dt > 0.0:
            out[i] = (y[hi] - y[lo]) / dt
    return out


def estimate_relative_potential_from_radial_field(solver, bridge, cfg, n_points=80):
    """Estimate V(surface+) - V(far) by integrating total radial E along +x."""
    lamD = common.debye_length(cfg.n0, cfg.Te_eV)
    dx_si = cfg.domain_size / cfg.grid_n
    r0 = cfg.radius + max(0.20 * lamD, 0.55 * dx_si)
    r1 = 0.5 * cfg.domain_size - max(dx_si, 0.25 * lamD)
    if r1 <= r0:
        return float("nan"), np.array([]), np.array([])
    radii = np.linspace(r0, r1, int(n_points))
    points_si = np.column_stack([radii, np.zeros_like(radii), np.zeros_like(radii)])
    E_code = solver.sample_current_E_at_points(bridge.length_si_to_code(points_si))
    E_si = bridge.efield_code_to_si(E_code)
    # V(r0)-V(r1) = integral_{r0}^{r1} E_r dr.
    v_rel = float(np.trapz(E_si[:, 0], radii))
    return v_rel, (radii - cfg.radius) / lamD, E_si[:, 0]


def run(cfg: BaseConfig, out: Path, args, preset_name: str):
    out.mkdir(parents=True, exist_ok=True)
    _install_surface_charge_ridge_hook(cfg)
    stl = make_connected_cylinder_stl(cfg, out / "floating_connected_cylinder.stl")
    bundle = common.build_pic_solver_mas(cfg, stl)
    PIC = bundle["PIC"]
    solver = bundle["solver"]
    mas = bundle["mas"]
    rng = bundle["rng"]
    bridge = bundle["unit_bridge"]
    W = float(bundle["macro_weight"])
    nx, ny, nz = bundle["shape"]
    dx, dy, dz = bundle["spacing"]
    Ex, Ey, Ez, Bx, By, Bz = bundle["fields"]
    obs_code = bridge.length_si_to_code(mas.geometry.ant_obs_points)

    lamD = common.debye_length(cfg.n0, cfg.Te_eV)
    Q = 0.0
    e_hits_total = 0
    i_hits_total = 0
    outer_recycled_e_total = 0
    outer_recycled_i_total = 0
    # For each conductor-collected particle the boundary reservoir injects a
    # same-species replacement.  Its signed charge is the net charge supplied
    # from the external plasma to the modeled (plasma + conductor) subsystem.
    Q_reservoir_in_due_to_collection = 0.0
    Q_outer_out = 0.0
    Q_outer_in = 0.0
    history = []
    final_profile = None
    ridge_warning_printed = False
    t0 = time.time()

    for step in range(cfg.steps):
        PIC.proceed_one_step(Ex, Ey, Ez, Bx, By, Bz, dx, dy, dz, nx, ny, nz, cfg.field_order)
        hits, dQ = collect_particles_to_floating_conductor(PIC, cfg, rng, W)
        Q += dQ
        Q_reservoir_in_due_to_collection += dQ
        e_hits_total += int(hits.get(0, 0))
        i_hits_total += int(hits.get(1, 0))

        outer_counts, dQ_outer_out, dQ_outer_in = recycle_particles_at_open_reservoir(
            PIC, cfg, rng, W
        )
        outer_recycled_e_total += int(outer_counts.get(0, 0))
        outer_recycled_i_total += int(outer_counts.get(1, 0))
        Q_outer_out += dQ_outer_out
        Q_outer_in += dQ_outer_in

        rho, Jx_face, Jy_face, Jz_face = PIC.evaluate_rho_J(dx, dy, dz, nx, ny, nz, cfg.particle_order)
        face_currents = (Jx_face, Jy_face, Jz_face)
        solver.Jefimenko_solver(rho.flatten(), include_antenna=False, face_currents=face_currents)
        incident_E_si = bridge.efield_code_to_si(solver.sample_current_E_at_points(obs_code))
        # Case 2 is quasi-electrostatic: use the instantaneous PIC volume charge
        # to evaluate the scalar potential entering the ideal-metal equal-
        # potential boundary condition.
        rho_np_code = rho.get() if hasattr(rho, "get") else np.asarray(rho)
        rho_si = volume_charge_density_code_to_si_compat(bridge, rho_np_code)
        incident_phi_si = mas.plasma_volume_charge_potential_on_surface(rho_si)

        if not hasattr(common, "unit_bridge_mas_update_fields_with_surface_charge"):
            raise RuntimeError(
                "_jefipic_mas_case_common.py lacks unit_bridge_mas_update_fields_with_surface_charge(). "
                "Please keep the integrated Route-2 surface-charge common.py/EMsolver patch installed. "
                "This script locally fixes the missing particle collection/recycling helpers, but the "
                "surface-charge field update still requires the core MAS/EMsolver route-2 implementation."
            )

        (Ex, Ey, Ez), (Bx, By, Bz), charge_state = common.unit_bridge_mas_update_fields_with_surface_charge(
            mas, solver, bridge, incident_E_si, incident_phi_si, Q,
            include_current_sources=False, return_gpu_fields=True
        )

        if (
            float(getattr(cfg, "surface_charge_ridge", 0.0)) > 0.0
            and not ridge_warning_printed
            and "surface_charge_regularization" not in charge_state
            and "regularized_kkt_condition_number" not in charge_state
            and "floating_surface_charge_regularized_kkt_condition_number" not in charge_state
        ):
            print(
                "[case2-route2-floating][warning] --surface-charge-ridge was requested, "
                "but charge_state did not report ridge diagnostics. Verify that "
                "EMsolver/mas_integration.py is the integrated ridge version."
            )
            ridge_warning_printed = True

        if step % cfg.diag_stride == 0 or step == cfg.steps - 1:
            v_relative, radial_x, radial_E = estimate_relative_potential_from_radial_field(solver, bridge, cfg)
            charge_from_particle_hits_C = float((int(i_hits_total) - int(e_hits_total)) * float(W) * E_CHARGE)
            charge_hit_consistency_error_C = float(Q - charge_from_particle_hits_C)
            charge_hit_consistency_error_normalized = float(
                abs(charge_hit_consistency_error_C)
                / max(abs(float(Q)), abs(charge_from_particle_hits_C), float(W) * E_CHARGE, 1.0e-30)
            )
            history.append({
                "step": int(step),
                "time_s": float((step + 1) * cfg.dt),
                "electron_hits_step": int(hits.get(0, 0)),
                "ion_hits_step": int(hits.get(1, 0)),
                "electron_hits_cumulative": int(e_hits_total),
                "ion_hits_cumulative": int(i_hits_total),
                "collected_charge_increment_C": float(dQ),
                "conductor_net_charge_C": float(Q),
                "charge_from_particle_hits_C": charge_from_particle_hits_C,
                "charge_hit_consistency_error_C": charge_hit_consistency_error_C,
                "charge_hit_consistency_error_normalized": charge_hit_consistency_error_normalized,
                "floating_potential_V": float(charge_state["floating_potential_V"]),
                "field_integrated_potential_surface_to_far_V": float(v_relative),
                "potential_rms_residual_V": float(charge_state["potential_rms_residual_V"]),
                "potential_max_residual_V": float(charge_state["potential_max_residual_V"]),
                "surface_charge_rms_tangential_E_V_per_m": float(charge_state["rms_tangential_E"]),
                "surface_charge_max_tangential_E_V_per_m": float(charge_state["max_tangential_E"]),
                "charge_constraint_error_C": float(charge_state["charge_constraint_error_C"]),
                "potential_rms_residual_normalized": float(charge_state["potential_rms_residual_V"]) / max(abs(float(charge_state["floating_potential_V"])), float(cfg.Te_eV), 1.0e-12),
                "charge_constraint_error_normalized": abs(float(charge_state["charge_constraint_error_C"])) / max(abs(float(Q)), float(W) * E_CHARGE, 1.0e-30),
                "independent_tangential_field_rms_V_per_m": float(charge_state["rms_tangential_E"]),
                "independent_tangential_field_max_V_per_m": float(charge_state["max_tangential_E"]),
                "electron_collected_charge_step_C": float(-E_CHARGE * W * int(hits.get(0, 0))),
                "ion_collected_charge_step_C": float(+E_CHARGE * W * int(hits.get(1, 0))),
                "electron_collection_current_A": float(-E_CHARGE * W * int(hits.get(0, 0)) / cfg.dt),
                "ion_collection_current_A": float(+E_CHARGE * W * int(hits.get(1, 0)) / cfg.dt),
                "net_collection_current_A": float(dQ / cfg.dt),
                "reservoir_injected_charge_due_to_collection_C": float(Q_reservoir_in_due_to_collection),
                "outer_boundary_outflow_charge_C": float(Q_outer_out),
                "outer_boundary_injected_charge_C": float(Q_outer_in),
                "open_system_charge_balance_error_C": float(Q - Q_reservoir_in_due_to_collection),
                "outer_boundary_recycled_electrons_cumulative": int(outer_recycled_e_total),
                "outer_boundary_recycled_ions_cumulative": int(outer_recycled_i_total),
                "plasma_surface_potential_mean_V": float(charge_state["plasma_potential_mean_V"]),
                "surface_charge_self_potential_mean_V": float(charge_state["self_potential_mean_V"]),
                "surface_charge_self_potential_rms_variation_V": float(charge_state["self_potential_rms_variation_V"]),
                "surface_charge_kkt_condition_number": float(charge_state["kkt_condition_number"]),
                "surface_charge_ridge_requested": float(getattr(cfg, "surface_charge_ridge", 0.0)),
                "surface_charge_regularization": _state_float(
                    charge_state,
                    "surface_charge_regularization",
                    default=float(getattr(cfg, "surface_charge_ridge", 0.0)),
                ),
                "surface_charge_unregularized_kkt_condition_number": _state_float(
                    charge_state,
                    "floating_surface_charge_unregularized_kkt_condition_number",
                    "unregularized_kkt_condition_number",
                    "surface_charge_unregularized_kkt_condition_number",
                    default=float(charge_state["kkt_condition_number"]),
                ),
                "surface_charge_regularized_kkt_condition_number": _state_float(
                    charge_state,
                    "floating_surface_charge_regularized_kkt_condition_number",
                    "regularized_kkt_condition_number",
                    "surface_charge_regularized_kkt_condition_number",
                    default=float(charge_state["kkt_condition_number"]),
                ),
            })
            final_profile = {
                "distance_over_lambda_D": radial_x.tolist(),
                "radial_total_E_V_per_m": radial_E.tolist(),
            }

        if step % max(1, cfg.steps // 10) == 0 or step == cfg.steps - 1:
            latest_v = history[-1]["floating_potential_V"] if history else float("nan")
            latest_vint = history[-1]["field_integrated_potential_surface_to_far_V"] if history else float("nan")
            print(
                f"[case2-route2-floating] step {step + 1}/{cfg.steps}; "
                f"Q={Q:.4e} C; Vf={latest_v:.4e} V; field-int={latest_vint:.4e} V; "
                f"hits(e,i)=({e_hits_total},{i_hits_total}); "
                f"outer-recycle(e,i)=({outer_recycled_e_total},{outer_recycled_i_total})"
            )

    elapsed = time.time() - t0
    fields = list(history[0].keys())
    with (out / "case2_route2_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    (out / "case2_route2_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if final_profile is not None:
        (out / "case2_route2_final_radial_field_profile.json").write_text(
            json.dumps(final_profile, indent=2), encoding="utf-8"
        )

    t = np.asarray([r["time_s"] for r in history])
    q = np.asarray([r["conductor_net_charge_C"] for r in history])
    v = np.asarray([r["floating_potential_V"] for r in history])
    v_field_integrated = np.asarray([r["field_integrated_potential_surface_to_far_V"] for r in history])
    potential_resid = np.asarray([r["potential_rms_residual_V"] for r in history])
    resid = np.asarray([r["surface_charge_rms_tangential_E_V_per_m"] for r in history])
    qerr = np.asarray([r["charge_constraint_error_C"] for r in history])
    potential_resid_norm = np.asarray([r.get("potential_rms_residual_normalized", np.nan) for r in history])
    qerr_norm = np.asarray([r.get("charge_constraint_error_normalized", np.nan) for r in history])
    e_hits_cum = np.asarray([r.get("electron_hits_cumulative", 0) for r in history])
    i_hits_cum = np.asarray([r.get("ion_hits_cumulative", 0) for r in history])
    e_current = np.asarray([r.get("electron_collection_current_A", np.nan) for r in history])
    i_current = np.asarray([r.get("ion_collection_current_A", np.nan) for r in history])
    net_current = np.asarray([r.get("net_collection_current_A", np.nan) for r in history])
    q_from_hits = np.asarray([r.get("charge_from_particle_hits_C", np.nan) for r in history])
    q_hit_err = np.asarray([r.get("charge_hit_consistency_error_C", np.nan) for r in history])
    q_hit_err_norm = np.asarray([r.get("charge_hit_consistency_error_normalized", np.nan) for r in history])

    electron_collected_charge_cum = -float(W) * E_CHARGE * e_hits_cum
    ion_collected_charge_cum = +float(W) * E_CHARGE * i_hits_cum
    net_collected_charge_cum = electron_collected_charge_cum + ion_collected_charge_cum
    e_current_avg = centered_window_slope(t, electron_collected_charge_cum, args.current_average_window)
    i_current_avg = centered_window_slope(t, ion_collected_charge_cum, args.current_average_window)
    net_current_avg = centered_window_slope(t, net_collected_charge_cum, args.current_average_window)

    tail = max(2, len(history) // 4)
    if len(history) >= tail and t[-1] > t[-tail]:
        late_current = float((q[-1] - q[-tail]) / (t[-1] - t[-tail]))
    else:
        late_current = float("nan")

    summary = {
        "model": "Route-2 MAS point-rho floating connected conductor; plasma grid rho/J retained; quasi-electrostatic validation",
        "preset": preset_name,
        "config": cfg_to_serializable_dict(cfg),
        "debye_length_m": float(lamD),
        "dx_over_lambda_D": float((cfg.domain_size / cfg.grid_n) / lamD),
        "ion_plasma_period_s": float(ion_plasma_period(cfg.n0, cfg.ion_mass)),
        "simulated_time_s": float(cfg.steps * cfg.dt),
        "simulated_ion_periods": float(cfg.steps * cfg.dt / ion_plasma_period(cfg.n0, cfg.ion_mass)),
        "macro_weight": float(W),
        "particle_collection_to_floating_conductor": True,
        "deposit_collected_particles_on_conductor": True,
        "surface_point_count": int(mas.geometry.num_points),
        "final_conductor_net_charge_C": float(Q),
        "final_charge_from_particle_hits_C": float(q_from_hits[-1]),
        "final_charge_hit_consistency_error_C": float(q_hit_err[-1]),
        "final_charge_hit_consistency_error_normalized": float(q_hit_err_norm[-1]),
        "max_abs_charge_hit_consistency_error_C": float(np.nanmax(np.abs(q_hit_err))),
        "max_charge_hit_consistency_error_normalized": float(np.nanmax(q_hit_err_norm)),
        "surface_charge_closure": "equipotential_plus_net_charge_with_optional_ridge",
        "surface_charge_ridge_requested": float(getattr(cfg, "surface_charge_ridge", 0.0)),
        "surface_charge_regularization_reported": float(history[-1].get("surface_charge_regularization", float("nan"))),
        "surface_charge_regularized_kkt_condition_number": float(history[-1].get("surface_charge_regularized_kkt_condition_number", history[-1]["surface_charge_kkt_condition_number"])),
        "surface_charge_unregularized_kkt_condition_number": float(history[-1].get("surface_charge_unregularized_kkt_condition_number", history[-1]["surface_charge_kkt_condition_number"])),
        "final_floating_potential_V": float(v[-1]),
        "final_field_integrated_potential_surface_to_far_V": float(v_field_integrated[-1]),
        "late_window_net_collection_current_A": late_current,
        "current_average_window_records": int(args.current_average_window),
        "late_window_mean_moving_average_net_collection_current_A": float(np.nanmean(net_current_avg[-tail:])),
        "late_window_mean_moving_average_electron_collection_current_A": float(np.nanmean(e_current_avg[-tail:])),
        "late_window_mean_moving_average_ion_collection_current_A": float(np.nanmean(i_current_avg[-tail:])),
        "late_window_mean_potential_rms_residual_V": float(np.mean(potential_resid[-tail:])),
        "late_window_mean_potential_rms_residual_normalized": float(np.nanmean(potential_resid_norm[-tail:])),
        "late_window_mean_charge_constraint_error_normalized": float(np.nanmean(qerr_norm[-tail:])),
        "late_window_mean_surface_tangential_residual_V_per_m_diagnostic": float(np.mean(resid[-tail:])),
        "max_charge_constraint_error_C": float(np.max(np.abs(qerr))),
        "surface_charge_kkt_condition_number": float(history[-1]["surface_charge_kkt_condition_number"]),
        "electron_hits_cumulative": int(e_hits_total),
        "ion_hits_cumulative": int(i_hits_total),
        "outer_boundary_recycled_electrons_cumulative": int(outer_recycled_e_total),
        "outer_boundary_recycled_ions_cumulative": int(outer_recycled_i_total),
        "reservoir_injected_charge_due_to_collection_C": float(Q_reservoir_in_due_to_collection),
        "outer_boundary_outflow_charge_C": float(Q_outer_out),
        "outer_boundary_injected_charge_C": float(Q_outer_in),
        "open_system_charge_balance_error_C": float(Q - Q_reservoir_in_due_to_collection),
        "elapsed_seconds": float(elapsed),
        "interpretation_limits": [
            "This Case 2 validates a connected floating-conductor surface-charge closure; it is not yet the RF two-electrode Case 3 model.",
            "The floating potential is the direct unknown of the equipotential-plus-net-charge surface solve; a radial field integral is retained only as a consistency diagnostic.",
            "Surface tangential electric field is a diagnostic, not the primary closure equation in this quasi-electrostatic floating-conductor model.",
            "The collected-particle charge ledger should match the conductor net charge; this is reported as charge_hit_consistency_error_C.",
            "The outer plasma is represented by an open reservoir: absorbed or outflow particles are recycled as inward boundary samples of the same species.",
            "Boundary-source events are recorded by charge ledgers; domain-size and reservoir-boundary sensitivity must be demonstrated before a paper-quality conclusion.",
            "A paper-quality result requires late-time current balance, Debye-scale spatial convergence, and adequate ion-response duration."
        ],
    }
    (out / "case2_route2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_case2_config(out, cfg)

    fmts = publication_formats(args.publication_formats)
    plt.rcParams.update({"font.size": 10, "figure.dpi": int(args.figure_dpi), "savefig.dpi": int(args.figure_dpi)})

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(t * 1e6, v, label="Equipotential solve: Vf")
    ax.plot(t * 1e6, v_field_integrated, linestyle="--", label="Field-integral check")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Potential (V)")
    ax.set_title("Floating-conductor potential evolution")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out / "case2_route2_floating_potential", fmts, args.figure_dpi)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(t * 1e6, q)
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Net collected charge (C)")
    ax.set_title("Charge retained on the floating conductor")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out / "case2_route2_conductor_charge", fmts, args.figure_dpi)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(t * 1e6, q, label="Conductor net charge")
    ax.plot(t * 1e6, q_from_hits, linestyle="--", label="Collected-particle ledger")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Charge (C)")
    ax.set_title("Conductor charge and particle-hit ledger")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out / "case2_route2_charge_hit_ledger", fmts, args.figure_dpi)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(t * 1e6, np.maximum(np.abs(q_hit_err), 1.0e-30))
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("|Qconductor - Qhits| (C)")
    ax.set_title("Collected-charge ledger consistency")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out / "case2_route2_charge_hit_consistency_error", fmts, args.figure_dpi)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(t * 1e6, np.maximum(potential_resid, 1.0e-30))
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Surface potential RMS residual (V)")
    ax.set_title("Equipotential closure residual")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out / "case2_route2_equipotential_residual", fmts, args.figure_dpi)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(t * 1e6, np.maximum(potential_resid_norm, 1.0e-30))
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Normalized equipotential residual")
    ax.set_title("Normalized equipotential closure residual")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out / "case2_route2_equipotential_residual_normalized", fmts, args.figure_dpi)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(t * 1e6, np.maximum(np.abs(qerr), 1.0e-30))
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Net-charge constraint error (C)")
    ax.set_title("Floating-conductor charge constraint")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out / "case2_route2_charge_constraint_error", fmts, args.figure_dpi)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(t * 1e6, np.maximum(qerr_norm, 1.0e-30))
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Normalized charge-constraint error")
    ax.set_title("Normalized charge-conservation closure error")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out / "case2_route2_charge_constraint_error_normalized", fmts, args.figure_dpi)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].plot(t * 1e6, q, label="Conductor net charge")
    axes[0].plot(t * 1e6, q_from_hits, linestyle="--", label="Particle-hit ledger")
    axes[0].set_xlabel("Time (µs)")
    axes[0].set_ylabel("Charge (C)")
    axes[0].set_title("(a) Charge collection")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, alpha=0.35)

    axes[1].semilogy(t * 1e6, np.maximum(potential_resid_norm, 1.0e-30))
    axes[1].set_xlabel("Time (µs)")
    axes[1].set_ylabel("Normalized residual")
    axes[1].set_title("(b) Equipotential closure")
    axes[1].grid(True, alpha=0.35)

    fig.tight_layout()
    save_figure(fig, out / "case2_route2_particle_collection_closure_summary", fmts, args.figure_dpi)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(t * 1e6, e_hits_cum, label="Electrons")
    ax.plot(t * 1e6, i_hits_cum, label="Ions")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Cumulative collected macroparticles")
    ax.set_title("Particle collection by the floating conductor")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out / "case2_route2_particle_collection_hits", fmts, args.figure_dpi)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(t * 1e6, e_current, label="Electron current")
    ax.plot(t * 1e6, i_current, label="Ion current")
    ax.plot(t * 1e6, net_current, linestyle="--", label="Net current")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Collection current (A)")
    ax.set_title("Stepwise collection-current diagnostic")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out / "case2_route2_collection_currents", fmts, args.figure_dpi)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(t * 1e6, e_current_avg, label="Electron current")
    ax.plot(t * 1e6, i_current_avg, label="Ion current")
    ax.plot(t * 1e6, net_current_avg, linestyle="--", label="Net current")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Moving-average collection current (A)")
    ax.set_title(f"Collection-current average ({int(args.current_average_window)} diagnostic records)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out / "case2_route2_collection_currents_moving_average", fmts, args.figure_dpi)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(t * 1e6, np.maximum(resid, 1.0e-30))
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Tangential-field RMS diagnostic (V/m)")
    ax.set_title("Tangential-field consistency diagnostic")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out / "case2_route2_tangential_field_diagnostic", fmts, args.figure_dpi)

    if final_profile is not None:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(final_profile["distance_over_lambda_D"], final_profile["radial_total_E_V_per_m"])
        ax.set_xlabel("Distance from conductor surface / Debye length")
        ax.set_ylabel("Radial electric field (V/m)")
        ax.set_title("Final radial electric-field profile")
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        save_figure(fig, out / "case2_route2_final_radial_field", fmts, args.figure_dpi)

    print("Finished:", out)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    args = parse_args()
    cfg, preset_name = configure(args)
    output_name = args.out or f"case2_floating_surface_charge_{preset_name}"
    out = Path("article_cases_output") / output_name
    run(cfg, out, args, preset_name)


if __name__ == "__main__":
    main()
