"""Full JefiPIC + MAS antenna sheath example.

This script follows the calling pattern of the root JefiPIC.ipynb, but inserts
a MAS antenna feedback step between the grid-only Jefimenko solve and the next
particle push.

Loop structure
--------------
For every time step:

1. PIC.proceed_one_step(...) advances particles using the previous total field.
2. PIC.evaluate_rho_J(...) deposits rho/J to the grid using the JefiPIC
   Esirkepov/CIC machinery.
3. EMsolver.Jefimenko_solver(..., include_antenna=False) computes the
   plasma-only incident field.
4. MASAntennaInteraction.update_fields_with_device_interaction(em_solver=...)
   samples that incident field on the antenna surface, solves the MAS passive
   current, pushes active+passive antenna currents to the EM solver, and
   recomputes the total plasma+antenna Jefimenko field.
5. The total field is used in the next particle step.

The code is intended as an executable research starting point.  The default
mode uses a quasi-electrostatic near-field time step (dt ~ 1 ns) appropriate
for MHz ionospheric sheath studies, not the fully retarded dx/c time step.

Run
---
From repo root:
    python MAS_Plasma_Antenna_Example.py --quick

From this directory:
    python Full_JefiPIC_MAS_Sheath.py --quick

For a longer run:
    python Full_JefiPIC_MAS_Sheath.py --steps 1200 --particles 8000 --grid-n 17
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional CUDA device selection can be supplied from the shell:
#   CUDA_VISIBLE_DEVICES=0 python ...
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
STL_DEMO_DIR = ROOT / "examples" / "stl_mas_demo"

for p in (ROOT, STL_DEMO_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cupy as cp

from EMsolver.solver import EMsolver
from EMsolver.mas_integration import SinusoidalCurrentExcitation, create_mas_antenna_simulation
from Particle_transport.main import pic_transport
from create_dipole_stl import generate_center_fed_dipole


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
M_E = 9.1093837015e-31
M_P = 1.67262192369e-27
C0 = 299792458.0


@dataclass
class Config:
    output_dir: str = "output"

    # Space plasma / ionosphere-like background
    n0: float = 1.0e11
    Te_eV: float = 0.2
    Ti_eV: float = 0.05
    ion_mass: float = 16.0 * M_P

    # Domain and grid
    domain_size: float = 0.16       # m, cubic box
    grid_n: int = 17                # 17^3 keeps Jefimenko cost manageable
    particle_order: int = 1
    field_order: int = 1
    use_periodic: int = 0
    use_reflective: int = 1

    # Time
    f_drive: float = 3.0e6          # Hz, near f_pe for n0=1e11 m^-3
    dt: float = 1.0e-9              # s, quasi-electrostatic near-field default
    steps: int = 900                # about 2.7 RF periods at 3 MHz
    len_time_snapshots: int = 512
    diag_stride: int = 10

    # Particles
    particles_per_species: int = 6000
    seed: int = 3

    # Antenna geometry
    total_length: float = 0.060
    radius: float = 0.0010
    feed_gap: float = 0.0020
    n_phi: int = 40
    n_z_per_arm: int = 10
    n_r_cap: int = 5

    # MAS
    mas_source_offset: float = 2.5e-4
    max_mas_points: int = 320
    patch_min_points: int = 24
    diagonal_loading: float = 1.0e-8
    use_pinv: bool = True

    # Active antenna source
    antenna_current_amplitude: float = 2.0e-3
    ramp_periods: float = 2.0

    # Jefimenko acceleration / near-field quasi-electrostatic setup
    quasi_electrostatic_near_field: bool = True
    enforce_retarded_time_resolution: bool = False
    spatial_truncation_debye: float = 8.0
    taper_width_debye: float = 2.0
    taper: str = "hann"
    point_source_truncation: str = "none"  # keep antenna source untruncated by default

    # Optional approximate particle absorber for the cylindrical antenna
    absorb_particles_on_antenna: bool = True


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="shorter test run")
    p.add_argument("--steps", type=int, default=Config.steps)
    p.add_argument("--particles", type=int, default=Config.particles_per_species)
    p.add_argument("--grid-n", type=int, default=Config.grid_n)
    p.add_argument("--dt", type=float, default=Config.dt)
    p.add_argument("--f-drive", type=float, default=Config.f_drive)
    p.add_argument("--mas-points", type=int, default=Config.max_mas_points)
    p.add_argument("--patch-min", type=int, default=Config.patch_min_points)
    p.add_argument("--current", type=float, default=Config.antenna_current_amplitude)
    p.add_argument("--out", default=Config.output_dir)
    p.add_argument("--fully-retarded", action="store_true",
                   help="use dx/c-like dt guidance externally; this flag only records intent and disables quasi-electrostatic notes")
    args = p.parse_args()

    cfg = Config()
    cfg.steps = args.steps
    cfg.particles_per_species = args.particles
    cfg.grid_n = args.grid_n
    cfg.dt = args.dt
    cfg.f_drive = args.f_drive
    cfg.max_mas_points = args.mas_points
    cfg.patch_min_points = args.patch_min
    cfg.antenna_current_amplitude = args.current
    cfg.output_dir = args.out

    if args.quick:
        cfg.steps = min(cfg.steps, 240)
        cfg.particles_per_species = min(cfg.particles_per_species, 1800)
        cfg.grid_n = min(cfg.grid_n, 13)
        cfg.max_mas_points = min(cfg.max_mas_points, 180)
        cfg.patch_min_points = min(cfg.patch_min_points, 16)
        cfg.len_time_snapshots = min(cfg.len_time_snapshots, 256)

    if args.fully_retarded:
        cfg.quasi_electrostatic_near_field = False
        cfg.enforce_retarded_time_resolution = True

    return cfg


# -----------------------------------------------------------------------------
# Physics helpers
# -----------------------------------------------------------------------------

def plasma_frequency(n0: float) -> float:
    omega = math.sqrt(n0 * E_CHARGE * E_CHARGE / (EPS0 * M_E))
    return omega / (2.0 * math.pi)


def debye_length(n0: float, Te_eV: float) -> float:
    return math.sqrt(EPS0 * Te_eV / (n0 * E_CHARGE))


def thermal_speed(T_eV: float, mass: float) -> float:
    return math.sqrt(max(T_eV, 0.0) * E_CHARGE / mass)


def antenna_mask_np(pos: np.ndarray, cfg: Config) -> np.ndarray:
    r = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    zabs = np.abs(pos[:, 2])
    return (r < cfg.radius) & (zabs >= 0.5 * cfg.feed_gap) & (zabs <= 0.5 * cfg.total_length)


def antenna_mask_cp(pos, cfg: Config):
    r = cp.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    zabs = cp.abs(pos[:, 2])
    return (r < cfg.radius) & (zabs >= 0.5 * cfg.feed_gap) & (zabs <= 0.5 * cfg.total_length)


def sample_positions(n: int, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    L = cfg.domain_size
    left = -0.5 * L
    pos = np.empty((n, 3), dtype=np.float64)
    filled = 0
    while filled < n:
        m = int(1.4 * (n - filled)) + 128
        cand = rng.uniform(left, -left, size=(m, 3))
        keep = ~antenna_mask_np(cand, cfg)
        cand = cand[keep]
        take = min(len(cand), n - filled)
        pos[filled:filled + take] = cand[:take]
        filled += take
    return pos


def sample_velocities(n: int, T_eV: float, mass: float, rng: np.random.Generator) -> np.ndarray:
    sigma = thermal_speed(T_eV, mass)
    return rng.normal(0.0, sigma, size=(n, 3))


def macro_weight(cfg: Config, n_macro: int) -> float:
    return cfg.n0 * cfg.domain_size ** 3 / float(n_macro)


def respawn_species(PIC, species_id: int, idx_cpu: np.ndarray, cfg: Config, rng: np.random.Generator):
    if len(idx_cpu) == 0:
        return 0
    if species_id == 0:
        T, mass = cfg.Te_eV, M_E
    else:
        T, mass = cfg.Ti_eV, cfg.ion_mass
    new_pos = sample_positions(len(idx_cpu), cfg, rng)
    new_v = sample_velocities(len(idx_cpu), T, mass, rng)
    idx_gpu = cp.asarray(idx_cpu, dtype=cp.int64)
    PIC.pos[species_id][idx_gpu] = cp.asarray(new_pos)
    PIC.velocity[species_id][idx_gpu] = cp.asarray(new_v)
    PIC.u[species_id][idx_gpu] = cp.asarray(new_v)  # nonrelativistic gamma ~= 1
    return len(idx_cpu)


def absorb_particles_on_cylindrical_antenna(PIC, cfg: Config, rng: np.random.Generator) -> dict:
    counts = {}
    if not cfg.absorb_particles_on_antenna:
        return {0: 0, 1: 0}
    for sid in list(PIC.pos.keys()):
        mask = antenna_mask_cp(PIC.pos[sid], cfg)
        idx = cp.where(mask)[0].get()
        counts[sid] = respawn_species(PIC, sid, idx, cfg, rng)
    return counts


# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

def build_particles(cfg: Config, rng: np.random.Generator):
    n = cfg.particles_per_species
    W = macro_weight(cfg, n)

    e_pos = sample_positions(n, cfg, rng)
    i_pos = sample_positions(n, cfg, rng)
    e_vel = sample_velocities(n, cfg.Te_eV, M_E, rng)
    i_vel = sample_velocities(n, cfg.Ti_eV, cfg.ion_mass, rng)

    pos = {0: e_pos, 1: i_pos}
    velocity = {0: e_vel, 1: i_vel}
    u = {0: e_vel.copy(), 1: i_vel.copy()}

    charges = [-E_CHARGE * W, E_CHARGE * W]
    masses = [M_E * W, cfg.ion_mass * W]
    number_charges = [W, W]

    return pos, velocity, u, charges, masses, number_charges, W


def build_active_spatial_weights(src_points: np.ndarray, cfg: Config) -> np.ndarray:
    # Current concentrated toward the feed gap, smooth along z.
    z_scale = max(0.30 * cfg.total_length, 1.0e-9)
    w = np.exp(-(src_points[:, 2] / z_scale) ** 2)
    if np.max(w) > 0.0:
        w = w / np.max(w)
    return w


def create_runtime_stl(cfg: Config, output_dir: Path) -> Path:
    stl_path = output_dir / "center_fed_dipole_runtime.stl"
    generate_center_fed_dipole(
        stl_path,
        total_length=cfg.total_length,
        radius=cfg.radius,
        feed_gap=cfg.feed_gap,
        n_phi=cfg.n_phi,
        n_z_per_arm=cfg.n_z_per_arm,
        n_r_cap=cfg.n_r_cap,
    )
    return stl_path


def build_pic_and_solver(cfg: Config, output_dir: Path):
    rng = np.random.default_rng(cfg.seed)

    L = cfg.domain_size
    nx = ny = nz = int(cfg.grid_n)
    dx = dy = dz = L / cfg.grid_n
    x0 = y0 = z0 = -0.5 * L

    pos, velocity, u, charges, masses, number_charges, W = build_particles(cfg, rng)

    initial_velocity = thermal_speed(cfg.Te_eV, M_E)
    initial_u = initial_velocity

    PIC = pic_transport(
        pos, velocity, u, cfg.dt, charges, number_charges, masses,
        x0, y0, z0,
        nx, ny, nz,
        dx, dy, dz,
        initial_velocity, initial_u, C0,
        cfg.use_periodic, cfg.use_reflective,
    )

    is_electrostatic = 0
    PIC_provider = EMsolver(
        cfg.len_time_snapshots,
        nx, ny, nz,
        nx, ny, nz,
        dx, dy, dz, x0, y0, z0,
        dx, dy, dz, x0, y0, z0,
        cfg.dt, EPS0, C0, is_electrostatic,
    )
    PIC_provider.set_kernel_mode("observation")

    lamD = debye_length(cfg.n0, cfg.Te_eV)
    grid_radius = cfg.spatial_truncation_debye * lamD
    taper_width = cfg.taper_width_debye * lamD

    point_radius = None if str(cfg.point_source_truncation).lower() in ("none", "off", "false") else grid_radius
    PIC_provider.set_spatial_truncation_radius(
        grid_radius=grid_radius,
        point_radius=point_radius,
        taper=cfg.taper,
        taper_width=taper_width,
    )

    stl_path = create_runtime_stl(cfg, output_dir)

    # First build geometry to get source points for spatial weights.
    geometry_params_base = {
        "kind": "stl",
        "stl_path": stl_path,
        "source_offset": cfg.mas_source_offset,
        "max_points": cfg.max_mas_points,
        "patch_min_points": cfg.patch_min_points,
        "patch_normal_axis": (0.0, 0.0, 1.0),
        "patch_normal_threshold": 0.75,
        "diagonal_loading": cfg.diagonal_loading,
        "use_pinv": cfg.use_pinv,
    }

    # create_mas_antenna_simulation builds geometry internally.  We pass a
    # SinusoidalCurrentExcitation.  Uniform weights are acceptable; for more
    # feed-localized current use a custom callable after geometry preloading.
    active = SinusoidalCurrentExcitation(
        amplitude=cfg.antenna_current_amplitude,
        frequency=cfg.f_drive,
        direction=(0.0, 0.0, 1.0),
        ramp_time=cfg.ramp_periods / cfg.f_drive,
    )

    simulation_params = {
        "dt": cfg.dt,
        "epsilon0": EPS0,
        "c": C0,
        "max_time_steps": cfg.len_time_snapshots,
        "grid_size": (nx, ny, nz),
        "grid_spacing": (dx, dy, dz),
        "grid_origin": (x0, y0, z0),
        "active_excitation": active,
        "source_softening": 0.25 * cfg.mas_source_offset,
        "source_truncation_radius": None,
    }

    mas_device = create_mas_antenna_simulation(geometry_params_base, simulation_params)
    mas_device.register_with_em_solver(PIC_provider)

    # Save MAS geometry for inspection.
    mas_device.geometry.save_npz(output_dir / "runtime_mas_geometry.npz")

    return PIC, PIC_provider, mas_device, rng, W, (nx, ny, nz), (dx, dy, dz), (x0, y0, z0)


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

def radial_density_from_particles(PIC, cfg: Config, W: float, bins=80):
    out = {}
    r_bins = np.linspace(0.0, 0.5 * cfg.domain_size, bins + 1)
    shell_vol = 4.0 / 3.0 * math.pi * (r_bins[1:] ** 3 - r_bins[:-1] ** 3)
    for sid, name in [(0, "electron"), (1, "ion")]:
        pos = PIC.pos[sid].get()
        rr = np.linalg.norm(pos, axis=1)
        counts, _ = np.histogram(rr, bins=r_bins)
        out[name] = counts * W / shell_vol
    out["r"] = 0.5 * (r_bins[:-1] + r_bins[1:])
    return out


def xz_charge_projection(PIC, cfg: Config, W: float, bins=96):
    xbins = np.linspace(-0.5 * cfg.domain_size, 0.5 * cfg.domain_size, bins + 1)
    zbins = np.linspace(-0.5 * cfg.domain_size, 0.5 * cfg.domain_size, bins + 1)
    epos = PIC.pos[0].get()
    ipos = PIC.pos[1].get()
    He, _, _ = np.histogram2d(epos[:, 0], epos[:, 2], bins=[xbins, zbins])
    Hi, _, _ = np.histogram2d(ipos[:, 0], ipos[:, 2], bins=[xbins, zbins])
    return xbins, zbins, (Hi - He) * W


def field_slice_xz(Ex, Ey, Ez, shape):
    nx, ny, nz = shape
    ex = Ex.get().reshape(nx, ny, nz)
    ey = Ey.get().reshape(nx, ny, nz)
    ez = Ez.get().reshape(nx, ny, nz)
    mid = ny // 2
    mag = np.sqrt(ex[:, mid, :] ** 2 + ey[:, mid, :] ** 2 + ez[:, mid, :] ** 2)
    return mag


def save_plots(output_dir: Path, cfg: Config, history: dict, PIC, W, last_fields, grid_shape):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Radial density
    rd = radial_density_from_particles(PIC, cfg, W)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(rd["r"], rd["electron"] / cfg.n0, label="electron / n0")
    ax.plot(rd["r"], rd["ion"] / cfg.n0, label="ion / n0")
    ax.axvline(cfg.radius, linestyle="--", linewidth=1.0, label="antenna radius")
    ax.set_xlabel("spherical radius from antenna center (m)")
    ax.set_ylabel("normalized density")
    ax.set_title("Final radial density diagnostic")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "final_radial_density.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # x-z charge projection
    xbins, zbins, Q = xz_charge_projection(PIC, cfg, W)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(Q.T, origin="lower", extent=[xbins[0], xbins[-1], zbins[0], zbins[-1]], aspect="equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("Final x-z ion-minus-electron macro density")
    fig.colorbar(im, ax=ax, label="macro-particle number density difference")
    fig.tight_layout()
    fig.savefig(output_dir / "final_xz_charge_projection.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Field magnitude slice
    if last_fields is not None:
        Ex, Ey, Ez, Bx, By, Bz = last_fields
        mag = field_slice_xz(Ex, Ey, Ez, grid_shape)
        L = cfg.domain_size
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        im = ax.imshow(mag.T, origin="lower", extent=[-L/2, L/2, -L/2, L/2], aspect="equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_title("Final |E| field slice at y=0")
        fig.colorbar(im, ax=ax, label="|E| (V/m)")
        fig.tight_layout()
        fig.savefig(output_dir / "final_E_magnitude_xz.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # MAS residual / absorbed particles
    t = np.asarray(history["time_s"])
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(t * 1e6, history["mas_rms_tangential_E"], label="RMS tangential E")
    ax.plot(t * 1e6, history["mas_max_tangential_E"], label="max tangential E")
    ax.set_xlabel("time (microseconds)")
    ax.set_ylabel("surface tangential E diagnostic (V/m)")
    ax.set_yscale("log")
    ax.set_title("MAS PEC boundary diagnostic")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mas_boundary_diagnostic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(t * 1e6, history["absorbed_electrons"], label="electron hits")
    ax.plot(t * 1e6, history["absorbed_ions"], label="ion hits")
    ax.set_xlabel("time (microseconds)")
    ax.set_ylabel("absorbed/reinjected macro-particles per diagnostic step")
    ax.set_title("Approximate cylindrical antenna particle-hit diagnostic")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "antenna_particle_hits.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------

def run(cfg: Config):
    output_dir = THIS_DIR / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    fpe = plasma_frequency(cfg.n0)
    lamD = debye_length(cfg.n0, cfg.Te_eV)
    period = 1.0 / cfg.f_drive
    dx = cfg.domain_size / cfg.grid_n
    perf_info = {
        "f_pe_Hz": fpe,
        "lambda_D_m": lamD,
        "drive_period_s": period,
        "steps_per_drive_period": period / cfg.dt,
        "dx_m": dx,
        "dx_over_c_s": dx / C0,
        "dt_over_dx_over_c": cfg.dt / (dx / C0),
        "quasi_electrostatic_near_field": cfg.quasi_electrostatic_near_field,
        "spatial_truncation_radius_m": cfg.spatial_truncation_debye * lamD,
        "taper_width_m": cfg.taper_width_debye * lamD,
    }
    (output_dir / "time_step_and_scale_summary.json").write_text(json.dumps(perf_info, indent=2), encoding="utf-8")

    print("Physical/time-step summary:")
    for k, v in perf_info.items():
        print(f"  {k}: {v}")

    if cfg.quasi_electrostatic_near_field:
        print("\nQuasi-electrostatic mode:")
        print("  Retarded dx/c resolution is not enforced.")
        print("  dt is chosen to resolve plasma frequency and RF drive, not light-cell crossing.")
    else:
        print("\nFully retarded intent:")
        print("  Make sure dt <= dx/c externally; this script does not automatically shrink dt.")

    PIC, PIC_provider, mas_device, rng, W, grid_shape, spacing, origin = build_pic_and_solver(cfg, output_dir)

    hist_info = PIC_provider.validate_history_capacity(include_antenna=True, action="warn")
    (output_dir / "history_capacity.json").write_text(json.dumps(hist_info, indent=2), encoding="utf-8")
    (output_dir / "mas_info.json").write_text(json.dumps(mas_device.get_simulation_info(), indent=2, default=str), encoding="utf-8")

    nx, ny, nz = grid_shape
    dx, dy, dz = spacing

    # Initial total field used by first particle push.
    Ex = cp.zeros(nx * ny * nz, dtype=cp.float64)
    Ey = cp.zeros(nx * ny * nz, dtype=cp.float64)
    Ez = cp.zeros(nx * ny * nz, dtype=cp.float64)
    Bx = cp.zeros(nx * ny * nz, dtype=cp.float64)
    By = cp.zeros(nx * ny * nz, dtype=cp.float64)
    Bz = cp.zeros(nx * ny * nz, dtype=cp.float64)

    history = {
        "step": [],
        "time_s": [],
        "mas_rms_tangential_E": [],
        "mas_max_tangential_E": [],
        "absorbed_electrons": [],
        "absorbed_ions": [],
    }

    start = time.time()
    last_fields = None

    for step in range(cfg.steps):
        # 1. Push particles with previous total field.
        PIC.proceed_one_step(Ex, Ey, Ez, Bx, By, Bz, dx, dy, dz, nx, ny, nz, cfg.field_order)

        hits = absorb_particles_on_cylindrical_antenna(PIC, cfg, rng)

        # 2. Deposit charge/current using the production staggered source layout.
        rho, Jx_face, Jy_face, Jz_face = PIC.evaluate_rho_J(
            dx, dy, dz, nx, ny, nz, cfg.particle_order
        )

        # 3. Grid-only incident plasma field.
        face_currents = (Jx_face, Jy_face, Jz_face)
        PIC_provider.Jefimenko_solver(
            rho.flatten(), include_antenna=False, face_currents=face_currents,
        )

        # 4. MAS feedback and total plasma+antenna field for the same time node.
        (Ex, Ey, Ez), (Bx, By, Bz) = mas_device.update_fields_with_device_interaction(
            em_solver=PIC_provider,
            return_gpu_fields=True,
        )
        last_fields = (Ex, Ey, Ez, Bx, By, Bz)

        if step % cfg.diag_stride == 0 or step == cfg.steps - 1:
            vw = mas_device.get_virtual_work_check()
            history["step"].append(int(step))
            history["time_s"].append(float(step * cfg.dt))
            history["mas_rms_tangential_E"].append(float(vw.get("rms_tangential_E", 0.0)))
            history["mas_max_tangential_E"].append(float(vw.get("max_tangential_E", 0.0)))
            history["absorbed_electrons"].append(int(hits.get(0, 0)))
            history["absorbed_ions"].append(int(hits.get(1, 0)))

        if step % max(1, cfg.steps // 10) == 0:
            print(f"step {step:6d}/{cfg.steps}: "
                  f"rmsEtan={history['mas_rms_tangential_E'][-1] if history['mas_rms_tangential_E'] else 0.0:.3e}, "
                  f"hits e/i={hits.get(0,0)}/{hits.get(1,0)}")

    elapsed = time.time() - start
    print(f"Finished in {elapsed:.2f} s")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Save final state arrays.
    np.savez(
        output_dir / "final_particle_state.npz",
        electron_pos=PIC.pos[0].get(),
        ion_pos=PIC.pos[1].get(),
        electron_velocity=PIC.velocity[0].get(),
        ion_velocity=PIC.velocity[1].get(),
    )

    save_plots(output_dir, cfg, history, PIC, W, last_fields, grid_shape)
    print("Outputs written to:", output_dir)


def main():
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
