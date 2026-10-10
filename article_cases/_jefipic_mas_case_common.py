"""Shared utilities for the five JefiPIC-MAS paper cases.

Place this directory under the JefiPIC repository root:

    JefiPIC/article_cases/

Then run, for example:

    cd JefiPIC
    python article_cases/case1_vacuum_mas_boundary.py --quick

The utilities intentionally use the real project modules:
- EMsolver.solver.EMsolver
- EMsolver.mas_integration.create_mas_antenna_simulation
- Particle_transport.main.pic_transport
- examples/stl_mas_demo/create_dipole_stl.py

The scripts assume the current JefiPIC-MAS package already contains the MAS
extensions discussed in the project conversation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import os
import sys
import time
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Prefer user-provided CUDA_VISIBLE_DEVICES, otherwise use device 0.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

try:
    import cupy as cp
except Exception as exc:  # pragma: no cover
    cp = None
    _CUPY_IMPORT_ERROR = exc
else:
    _CUPY_IMPORT_ERROR = None


# -----------------------------------------------------------------------------
# Repository path setup
# -----------------------------------------------------------------------------

def find_repo_root(start: Optional[Path] = None) -> Path:
    p = Path(start or __file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "EMsolver").exists() and (parent / "Particle_transport").exists():
            return parent
    # Expected layout when file is JefiPIC/article_cases/_common.py
    return Path(__file__).resolve().parents[1]


ROOT = find_repo_root()
STL_DEMO_DIR = ROOT / "examples" / "stl_mas_demo"
for _p in (ROOT, STL_DEMO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from EMsolver.solver import EMsolver
from EMsolver.mas_integration import SinusoidalCurrentExcitation, create_mas_antenna_simulation
from EMsolver.mas_preprocessor import preprocess_mas_antenna
from Particle_transport.main import pic_transport
from RBG_Maxwell.Unit_conversion.main import unit_conversion
from create_dipole_stl import generate_center_fed_dipole


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
M_E = 9.1093837015e-31
M_P = 1.67262192369e-27
C0 = 299792458.0

def add_common_particle_order_arg(parser):
    """
    Add common particle deposition order CLI argument.

    particle_order = 1: CIC
    particle_order = 2: TSC
    """
    parser.add_argument(
        "--particle-order",
        type=int,
        choices=(1, 2),
        default=None,
        help="Particle deposition order: 1=CIC, 2=TSC. Default uses BaseConfig.particle_order.",
    )


def apply_common_particle_order_arg(cfg, args):
    """
    Apply common particle deposition order CLI argument to BaseConfig.
    """
    if getattr(args, "particle_order", None) is not None:
        cfg.particle_order = int(args.particle_order)


# -----------------------------------------------------------------------------
# JefiPIC <-> MAS unit bridge
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class JefiPICUnitBridge:
    """Explicit unit bridge between SI MAS and JefiPIC internal code units.

    JefiPIC PIC/EMsolver is numerically intended to run in LHQCD/FU code units,
    while MAS geometry and the MAS boundary solve remain in SI.  Every
    JefiPIC--MAS interface therefore converts explicitly:

    - SI particle/geometry values -> JefiPIC code units for PIC/EMsolver;
    - JefiPIC E/B fields -> SI before entering the MAS boundary solve;
    - MAS surface-current density K [A/m] and surface-polarization density
      P [C/m] -> JefiPIC code units before writing the continuity-consistent
      electric-dipole state back to EMsolver.
    """

    mode: str = "LHQCD"
    hbar: float = 1.1822579826363647e-07
    c_code_value: float = 2.9979090484247153
    coef_J_to_E: float = 1.386275612896618e-15
    epsilon0_code_value: float = 90631227428685.16

    def __post_init__(self):
        mode = str(self.mode).upper()
        object.__setattr__(self, "mode", mode)

        if mode == "SI":
            ct = {
                "meter": 1.0,
                "second": 1.0,
                "kilogram": 1.0,
                "Coulomb": 1.0,
                "Ampere": 1.0,
                "Tesla": 1.0,
                "Volt/m": 1.0,
            }
            rv = {
                "TO_meter": 1.0,
                "TO_second": 1.0,
                "TO_kilogram": 1.0,
                "TO_Coulomb": 1.0,
                "TO_Ampere": 1.0,
                "TO_Tesla": 1.0,
                "TO_Volt/m": 1.0,
            }
            c_code = C0
            eps_code = EPS0
        elif mode == "LHQCD":
            ct = unit_conversion(
                "SI_to_LHQCD",
                coef_J_to_E=float(self.coef_J_to_E),
                hbar=float(self.hbar),
                c=float(self.c_code_value),
                k=1.0,
                epsilon0=float(self.epsilon0_code_value),
            )
            rv = unit_conversion(
                "LHQCD_to_SI",
                coef_J_to_E=float(self.coef_J_to_E),
                hbar=float(self.hbar),
                c=float(self.c_code_value),
                k=1.0,
                epsilon0=float(self.epsilon0_code_value),
            )
            c_code = float(self.c_code_value)
            eps_code = float(self.epsilon0_code_value)
        else:
            raise ValueError(f"Unsupported jefipic_unit_mode={self.mode!r}; use 'LHQCD' or 'SI'.")

        object.__setattr__(self, "ct", dict(ct))
        object.__setattr__(self, "rv", dict(rv))
        object.__setattr__(self, "c_code", float(c_code))
        object.__setattr__(self, "epsilon0_code", float(eps_code))

        # SI -> code coefficients.
        object.__setattr__(self, "L", float(ct["meter"]))
        object.__setattr__(self, "T", float(ct["second"]))
        object.__setattr__(self, "M", float(ct["kilogram"]))
        object.__setattr__(self, "Q", float(ct["Coulomb"]))
        object.__setattr__(self, "I", float(ct["Ampere"]))
        object.__setattr__(self, "E", float(ct["Volt/m"]))
        object.__setattr__(self, "B", float(ct["Tesla"]))

        # code -> SI coefficients.
        object.__setattr__(self, "TO_L", float(rv["TO_meter"]))
        object.__setattr__(self, "TO_T", float(rv["TO_second"]))
        object.__setattr__(self, "TO_M", float(rv["TO_kilogram"]))
        object.__setattr__(self, "TO_Q", float(rv["TO_Coulomb"]))
        object.__setattr__(self, "TO_I", float(rv["TO_Ampere"]))
        object.__setattr__(self, "TO_E", float(rv["TO_Volt/m"]))
        object.__setattr__(self, "TO_B", float(rv["TO_Tesla"]))

    @property
    def is_si(self) -> bool:
        return self.mode == "SI"

    def length_si_to_code(self, x):
        return np.asarray(x, dtype=np.float64) * self.L

    def length_code_to_si(self, x):
        return np.asarray(x, dtype=np.float64) * self.TO_L

    def velocity_si_to_code(self, v):
        return np.asarray(v, dtype=np.float64) * (self.L / self.T)

    def velocity_code_to_si(self, v):
        return np.asarray(v, dtype=np.float64) * (self.TO_L / self.TO_T)

    def time_si_to_code(self, t):
        return float(t) * self.T

    def charge_si_to_code(self, q):
        return np.asarray(q, dtype=np.float64) * self.Q

    def charge_code_to_si(self, q_code):
        return np.asarray(q_code, dtype=np.float64) * self.TO_Q

    def surface_charge_density_si_to_code(self, sigma_si):
        """Convert surface charge density from SI C/m^2 to code units."""
        return np.asarray(sigma_si, dtype=np.float64) * (self.Q / (self.L * self.L))

    def surface_charge_density_code_to_si(self, sigma_code):
        """Convert surface charge density from code units to SI C/m^2."""
        return np.asarray(sigma_code, dtype=np.float64) * (self.TO_Q / (self.TO_L * self.TO_L))

    def volume_charge_density_si_to_code(self, rho_si):
        """Convert volume charge density from SI C/m^3 to code units."""
        return np.asarray(rho_si, dtype=np.float64) * (self.Q / (self.L ** 3))

    def volume_charge_density_code_to_si(self, rho_code):
        """Convert volume charge density from code units to SI C/m^3."""
        return np.asarray(rho_code, dtype=np.float64) * (self.TO_Q / (self.TO_L ** 3))

    def mass_si_to_code(self, m):
        return np.asarray(m, dtype=np.float64) * self.M

    def efield_code_to_si(self, E_code):
        return np.asarray(E_code, dtype=np.float64) * self.TO_E

    def bfield_code_to_si(self, B_code):
        return np.asarray(B_code, dtype=np.float64) * self.TO_B

    def efield_si_to_code(self, E_si):
        return np.asarray(E_si, dtype=np.float64) * self.E

    def bfield_si_to_code(self, B_si):
        return np.asarray(B_si, dtype=np.float64) * self.B

    def surface_current_density_si_to_code(self, K_si):
        """Convert MAS surface-current density K from SI A/m to code units.

        Each MAS coefficient is multiplied by a source-patch area, so
        ``source_weight * K`` is a lumped electric current moment [A m].
        """
        return np.asarray(K_si, dtype=np.float64) * (self.I / self.L)

    def surface_current_density_code_to_si(self, K_code):
        return np.asarray(K_code, dtype=np.float64) * (self.TO_I / self.TO_L)

    def surface_polarization_density_si_to_code(self, P_si):
        """Convert Step-2B surface-polarization density P [C/m] to code units."""
        return np.asarray(P_si, dtype=np.float64) * (self.Q / self.L)

    def surface_polarization_density_code_to_si(self, P_code):
        return np.asarray(P_code, dtype=np.float64) * (self.TO_Q / self.TO_L)

    def area_si_to_code(self, a):
        return np.asarray(a, dtype=np.float64) * (self.L * self.L)

    def number_density_code_to_si(self, n_code):
        return np.asarray(n_code, dtype=np.float64) * (self.L ** 3)

    def number_density_si_to_code(self, n_si):
        return np.asarray(n_si, dtype=np.float64) / (self.L ** 3)

    def summary(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "meter_SI_to_code": self.L,
            "second_SI_to_code": self.T,
            "kilogram_SI_to_code": self.M,
            "Coulomb_SI_to_code": self.Q,
            "surface_charge_density_C_per_m2_SI_to_code": self.Q / (self.L * self.L),
            "volume_charge_density_C_per_m3_SI_to_code": self.Q / (self.L ** 3),
            "Ampere_SI_to_code": self.I,
            "surface_current_density_A_per_m_SI_to_code": self.I / self.L,
            "surface_polarization_density_C_per_m_SI_to_code": self.Q / self.L,
            "Volt_per_meter_SI_to_code": self.E,
            "Tesla_SI_to_code": self.B,
            "c_code": self.c_code,
            "epsilon0_code": self.epsilon0_code,
            "surface_current_density_A_per_m2_SI_to_code": self.I / (self.L * self.L),
        }


def get_unit_bridge(cfg: "BaseConfig") -> JefiPICUnitBridge:
    return JefiPICUnitBridge(
        mode=getattr(cfg, "jefipic_unit_mode", "LHQCD"),
        hbar=getattr(cfg, "lhqcd_hbar", 1.1822579826363647e-07),
        c_code_value=getattr(cfg, "lhqcd_c", 2.9979090484247153),
        coef_J_to_E=getattr(cfg, "lhqcd_coef_J_to_E", 1.386275612896618e-15),
        epsilon0_code_value=getattr(cfg, "lhqcd_epsilon0", 90631227428685.16),
    )


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class BaseConfig:
    output_dir: str = "output"
    seed: int = 11

    # Unit interface.  MAS remains SI; JefiPIC PIC/EMsolver runs in this mode.
    jefipic_unit_mode: str = "LHQCD"  # "LHQCD" or "SI"
    lhqcd_hbar: float = 1.1822579826363647e-07
    lhqcd_c: float = 2.9979090484247153
    lhqcd_coef_J_to_E: float = 1.386275612896618e-15
    lhqcd_epsilon0: float = 90631227428685.16
    jefipic_kernel_mode: str = "observation"  # production staggered-current kernel
    grid_current_layout: str = "staggered_face"  # production layout; legacy cell-current path removed

    # Space plasma
    n0: float = 1.0e11
    Te_eV: float = 0.2
    Ti_eV: float = 0.05
    ion_mass: float = 16.0 * M_P

    # Domain/grid
    domain_size: float = 0.18
    grid_n: int = 17
    len_time_snapshots: int = 384
    particle_order: int = 1
    field_order: int = 1
    use_periodic: int = 0
    use_reflective: int = 1
    # Optional explicit particle boundary selector.  'auto' preserves the
    # historical use_periodic/use_reflective flags.  Step 1-F adds
    # 'thermal_open': a constant-weight thermal-bath open reservoir.
    particle_boundary_mode: str = "auto"  # auto|reflective|periodic|open|thermal_open
    open_reservoir_seed_offset: int = 1701

    # Time
    dt: float = 1.0e-9
    steps: int = 900
    diag_stride: int = 10

    # Particles
    particles_per_species: int = 6000

    # Straight antenna geometry
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
    surface_charge_ridge: float = 0.0
    use_pinv: bool = True
    # Step 2-G production spatial MAS solve.  The tangent frames are built
    # from geometry normals, so this is not cylinder-coordinate specific.
    mas_spatial_solver: str = "tangential_oversampled"
    mas_collocation_oversample: int = 2
    mas_tikhonov_rel: float = 1.0e-4
    mas_svd_rcond: float = 1.0e-10
    # Optional STL facet subdivision target.  None leaves the imported mesh
    # unchanged; a positive value refines large/anisotropic facets before MAS
    # source sampling without changing the physical surface.
    mas_target_patch_size: Optional[float] = None
    mas_stl_max_subdivision_levels: int = 6

    # RF/drive
    f_drive: float = 3.0e6
    current_amplitude: float = 2.0e-3
    voltage_amplitude_proxy: float = 1.0
    ramp_periods: float = 2.0

    # Near-field acceleration
    quasi_electrostatic: bool = True
    spatial_truncation_debye: float = 0.0
    taper_width_debye: float = 0.0
    taper: str = "none"
    point_radius: Optional[float] = None

    # Diagnostics / boundary
    absorb_particles_on_antenna: bool = True
    particle_reinjection_mode: str = "uniform"  # "uniform" or "boundary"
    boundary_injection_offset_cells: float = 0.5
    # Step 1-Eb: keep reflective CIC particle centers at least half a cell
    # inside the source mesh so the full cell-centered CIC cloud remains on-grid.
    particle_center_guard_cells: float = 0.0
    # Optional startup controls. Defaults are OFF so other article cases keep
    # their original behavior unless a case explicitly enables them.
    initial_antenna_exclusion_debye: float = 0.0
    boundary_reinjection_antenna_exclusion_debye: float = 0.0
    plasma_to_mas_ramp_steps: int = 0
    quiet_start_pairs: bool = False
    center_initial_velocities: bool = False
    velocity_quiet_start_pairs: bool = False
    mas_axis_reduced_solve: bool = False
    mas_axis_reduced_rcond: float = 1.0e-8
    mas_axis_reduced_ridge: float = 1.0e-10
    diagnostic_skip_initial_steps: int = 0

    # Density diagnostics.  These are diagnostics only and do not affect dynamics.
    density_bins: int = 32
    density_sidewall_margin_debye: float = 0.25
    density_max_distance_debye: float = 8.0


def apply_quick_defaults(cfg: BaseConfig) -> BaseConfig:
    cfg.steps = min(cfg.steps, 160)
    cfg.particles_per_species = min(cfg.particles_per_species, 1500)
    cfg.grid_n = min(cfg.grid_n, 11)
    cfg.max_mas_points = min(cfg.max_mas_points, 160)
    cfg.patch_min_points = min(cfg.patch_min_points, 12)
    cfg.len_time_snapshots = min(cfg.len_time_snapshots, 192)
    cfg.diag_stride = max(5, min(cfg.diag_stride, 10))
    return cfg


def save_config(output_dir: Path, cfg: BaseConfig, name: str = "run_config.json") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / name).write_text(json.dumps(asdict(cfg), indent=2, default=float), encoding="utf-8")


def physical_summary(cfg: BaseConfig) -> Dict[str, float]:
    fpe = plasma_frequency(cfg.n0)
    lamD = debye_length(cfg.n0, cfg.Te_eV)
    dx = cfg.domain_size / cfg.grid_n
    unit_bridge = get_unit_bridge(cfg)
    summary = {
        "f_pe_Hz": fpe,
        "T_pe_s": 1.0 / fpe,
        "lambda_D_m": lamD,
        "drive_frequency_Hz": cfg.f_drive,
        "drive_period_s": 1.0 / cfg.f_drive,
        "dt_s": cfg.dt,
        "steps_per_drive_period": (1.0 / cfg.f_drive) / cfg.dt,
        "dx_m": dx,
        "dx_over_c_s": dx / C0,
        "dt_over_dx_over_c": cfg.dt / (dx / C0),
        "antenna_electrical_size_L_over_lambda": cfg.total_length / (C0 / cfg.f_drive),
        "spatial_truncation_radius_m": cfg.spatial_truncation_debye * lamD,
        "taper_width_m": cfg.taper_width_debye * lamD,
    }
    summary["jefipic_unit_mode"] = unit_bridge.mode
    summary["jefipic_unit_bridge"] = unit_bridge.summary()
    return summary


# -----------------------------------------------------------------------------
# Physics helpers
# -----------------------------------------------------------------------------

def plasma_frequency(n0: float) -> float:
    return math.sqrt(n0 * E_CHARGE * E_CHARGE / (EPS0 * M_E)) / (2.0 * math.pi)


def debye_length(n0: float, Te_eV: float) -> float:
    return math.sqrt(EPS0 * Te_eV / (n0 * E_CHARGE))


def thermal_speed(T_eV: float, mass: float) -> float:
    return math.sqrt(max(T_eV, 0.0) * E_CHARGE / mass)


def macro_weight(cfg: BaseConfig, n_macro: int) -> float:
    return cfg.n0 * cfg.domain_size ** 3 / float(n_macro)


def ramp(t: float, f: float, ramp_periods: float) -> float:
    rt = max(ramp_periods / f, 1.0e-30)
    if t >= rt:
        return 1.0
    s = t / rt
    return 0.5 * (1.0 - math.cos(math.pi * s))


def antenna_mask_np(pos: np.ndarray, cfg: BaseConfig) -> np.ndarray:
    r = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    zabs = np.abs(pos[:, 2])
    return (r < cfg.radius) & (zabs >= 0.5 * cfg.feed_gap) & (zabs <= 0.5 * cfg.total_length)


def antenna_mask_cp(pos, cfg: BaseConfig, bridge: Optional[JefiPICUnitBridge] = None):
    """CuPy mask for particle absorption on the straight-dipole metal.

    ``pos`` may be SI coordinates or JefiPIC code-unit coordinates.  With a
    bridge, code coordinates are converted to SI before comparison with cfg.
    """
    if bridge is not None and not bridge.is_si:
        pos_si = pos * bridge.TO_L
    else:
        pos_si = pos
    r = cp.sqrt(pos_si[:, 0] ** 2 + pos_si[:, 1] ** 2)
    zabs = cp.abs(pos_si[:, 2])
    return (r < cfg.radius) & (zabs >= 0.5 * cfg.feed_gap) & (zabs <= 0.5 * cfg.total_length)


def distance_to_straight_dipole_metal(pos: np.ndarray, cfg: BaseConfig) -> np.ndarray:
    """Approximate shortest distance from points to the straight-dipole metal.

    The straight dipole is represented as two finite cylindrical arms with
    radius ``cfg.radius`` and z intervals
        [feed_gap/2, total_length/2] and [-total_length/2, -feed_gap/2].

    The returned value is zero inside the metal and positive outside.  It is
    used only for particle initialization/reinjection exclusion layers; it does
    not alter the antenna geometry or MAS solve.
    """
    pos = np.asarray(pos, dtype=np.float64)
    r = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    zabs = np.abs(pos[:, 2])

    z0 = 0.5 * cfg.feed_gap
    z1 = 0.5 * cfg.total_length

    dr = np.maximum(r - cfg.radius, 0.0)
    dz = np.maximum.reduce([z0 - zabs, zabs - z1, np.zeros_like(zabs)])
    return np.sqrt(dr * dr + dz * dz)


def antenna_exclusion_mask_np(pos: np.ndarray, cfg: BaseConfig, clearance_m: float = 0.0) -> np.ndarray:
    """Mask points inside or within ``clearance_m`` of the straight dipole."""
    clearance_m = max(float(clearance_m), 0.0)
    if clearance_m <= 0.0:
        return antenna_mask_np(pos, cfg)
    return distance_to_straight_dipole_metal(pos, cfg) <= clearance_m


def sample_positions(
    n: int,
    cfg: BaseConfig,
    rng: np.random.Generator,
    exclusion_debye: Optional[float] = None,
) -> np.ndarray:
    """Sample particle positions, excluding the antenna and optional sheath layer.

    ``exclusion_debye`` controls an extra particle-free shell around the metal
    in units of the Debye length.  If omitted, this uses
    ``cfg.initial_antenna_exclusion_debye``.  The default in BaseConfig is zero,
    so existing cases are unchanged unless they explicitly enable it.
    """
    L = cfg.domain_size
    half = 0.5 * L
    guard = 0.0
    explicit_bc = str(getattr(cfg, "particle_boundary_mode", "auto")).lower()
    is_reflective = (
        explicit_bc == "reflective"
        or (
            explicit_bc == "auto"
            and bool(getattr(cfg, "use_reflective", 0))
            and not bool(getattr(cfg, "use_periodic", 0))
        )
    )
    if is_reflective:
        guard = float(getattr(cfg, "particle_center_guard_cells", 0.0)) * (L / float(cfg.grid_n))
    left, right = -half + guard, half - guard
    if not left < right:
        raise ValueError("particle-center guard leaves no interior sampling volume")
    out = np.empty((n, 3), dtype=np.float64)
    filled = 0

    if exclusion_debye is None:
        exclusion_debye = getattr(cfg, "initial_antenna_exclusion_debye", 0.0)
    clearance_m = float(exclusion_debye) * debye_length(cfg.n0, cfg.Te_eV)

    while filled < n:
        m = int(1.4 * (n - filled)) + 128
        cand = rng.uniform(left, right, size=(m, 3))
        keep = ~antenna_exclusion_mask_np(cand, cfg, clearance_m)
        cand = cand[keep]
        take = min(len(cand), n - filled)
        out[filled:filled + take] = cand[:take]
        filled += take
    return out


def sample_velocities(n: int, T_eV: float, mass: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, thermal_speed(T_eV, mass), size=(n, 3))


def sample_velocity_quiet_pairs(n: int, T_eV: float, mass: float, rng: np.random.Generator) -> np.ndarray:
    """Sample thermal velocities as +/- pairs to suppress initial current noise.

    This preserves a Maxwellian-like thermal speed distribution while making the
    initial bulk current much closer to zero locally when used with paired
    positions.  It is intended for quiet-start sheath diagnostics, not as a
    universal default for all article cases.
    """
    if n <= 0:
        return np.empty((0, 3), dtype=np.float64)
    n_pair = (n + 1) // 2
    v = sample_velocities(n_pair, T_eV, mass, rng)
    out = np.empty((2 * n_pair, 3), dtype=np.float64)
    out[0::2] = v
    out[1::2] = -v
    out = out[:n].copy()
    out -= np.mean(out, axis=0, keepdims=True)
    return out


def sample_quiet_pair_positions(n: int, cfg: BaseConfig, rng: np.random.Generator) -> np.ndarray:
    """Sample positions as duplicate pairs for local charge/current quiet start."""
    if n <= 0:
        return np.empty((0, 3), dtype=np.float64)
    n_pair = (n + 1) // 2
    base = sample_positions(
        n_pair,
        cfg,
        rng,
        exclusion_debye=getattr(cfg, "initial_antenna_exclusion_debye", 0.0),
    )
    out = np.repeat(base, 2, axis=0)[:n].copy()
    return out


def sample_boundary_reservoir_positions_velocities(
    n: int,
    cfg: BaseConfig,
    T_eV: float,
    mass: float,
    rng: np.random.Generator,
):
    """Sample replacement particles from an open-boundary reservoir.

    Particles are placed just inside a randomly selected domain face and given a
    half-Maxwellian inward normal velocity plus Maxwellian tangential velocity.
    If ``cfg.boundary_reinjection_antenna_exclusion_debye`` is positive, any
    proposed injected particle too close to the antenna is resampled.  This
    prevents reservoir particles from being recreated directly inside the
    near-surface sheath/startup exclusion layer.
    """
    L = float(cfg.domain_size)
    half = 0.5 * L
    dx = L / float(cfg.grid_n)
    guard_cells = float(getattr(cfg, "particle_center_guard_cells", 0.0))
    offset_cells = max(float(cfg.boundary_injection_offset_cells), guard_cells)
    # A tiny inward offset avoids equality with the strict pusher bound.
    offset = max(offset_cells * dx + 1.0e-12 * max(dx, 1.0), 1.0e-12)
    face_coord = half - offset

    clearance_m = (
        float(getattr(cfg, "boundary_reinjection_antenna_exclusion_debye", 0.0))
        * debye_length(cfg.n0, cfg.Te_eV)
    )

    pos = np.empty((n, 3), dtype=np.float64)
    vel = np.empty((n, 3), dtype=np.float64)
    filled = 0

    vth = max(thermal_speed(T_eV, mass), 0.0)
    min_inward = 0.02 * vth

    # Face convention: (face_id, axis, inward_sign)
    face_specs = [
        (0, 0, +1.0),
        (1, 0, -1.0),
        (2, 1, +1.0),
        (3, 1, -1.0),
        (4, 2, +1.0),
        (5, 2, -1.0),
    ]

    while filled < n:
        m = int(1.4 * (n - filled)) + 32
        faces = rng.integers(0, 6, size=m)
        cand_pos = rng.uniform(-half, half, size=(m, 3))
        cand_vel = sample_velocities(m, T_eV, mass, rng)

        for face, axis, sign in face_specs:
            mask = faces == face
            if not np.any(mask):
                continue
            cand_pos[mask, axis] = sign * face_coord
            cand_vel[mask, axis] = sign * (np.abs(cand_vel[mask, axis]) + min_inward)

        if clearance_m > 0.0:
            keep = ~antenna_exclusion_mask_np(cand_pos, cfg, clearance_m)
            cand_pos = cand_pos[keep]
            cand_vel = cand_vel[keep]

        take = min(len(cand_pos), n - filled)
        if take > 0:
            pos[filled:filled + take] = cand_pos[:take]
            vel[filled:filled + take] = cand_vel[:take]
            filled += take

    return pos, vel



def respawn_species(PIC, species_id: int, idx_cpu: np.ndarray, cfg: BaseConfig, rng: np.random.Generator) -> int:
    """Replace particles absorbed by the antenna.

    Two replacement modes are supported:

    ``uniform``:
        Historical behavior. Replacement particles are sampled uniformly in the
        full computational volume, excluding the antenna volume.

    ``boundary``:
        Open-reservoir behavior for Case 2. Replacement particles are injected
        just inside a randomly selected computational boundary face with inward
        half-Maxwellian normal velocity. This keeps the local domain supplied by
        an external plasma reservoir without recreating particles throughout the
        sheath region.
    """
    if len(idx_cpu) == 0:
        return 0

    if species_id == 0:
        T, mass = cfg.Te_eV, M_E
    else:
        T, mass = cfg.Ti_eV, cfg.ion_mass

    mode = getattr(cfg, "particle_reinjection_mode", "uniform").lower()
    if mode == "boundary":
        new_pos, new_v = sample_boundary_reservoir_positions_velocities(len(idx_cpu), cfg, T, mass, rng)
    elif mode == "uniform":
        new_pos = sample_positions(
            len(idx_cpu),
            cfg,
            rng,
            exclusion_debye=getattr(cfg, "boundary_reinjection_antenna_exclusion_debye", 0.0),
        )
        new_v = sample_velocities(len(idx_cpu), T, mass, rng)
    else:
        raise ValueError(f"Unknown particle_reinjection_mode={mode!r}; expected 'uniform' or 'boundary'.")

    bridge = getattr(PIC, "_jefipic_unit_bridge", get_unit_bridge(cfg))
    new_pos_code = bridge.length_si_to_code(new_pos)
    new_v_code = bridge.velocity_si_to_code(new_v)
    idx_gpu = cp.asarray(idx_cpu, dtype=cp.int64)
    PIC.pos[species_id][idx_gpu] = cp.asarray(new_pos_code)
    PIC.velocity[species_id][idx_gpu] = cp.asarray(new_v_code)
    PIC.u[species_id][idx_gpu] = cp.asarray(new_v_code)
    return len(idx_cpu)


def absorb_particles_on_antenna(PIC, cfg: BaseConfig, rng: np.random.Generator) -> Dict[int, int]:
    if not cfg.absorb_particles_on_antenna:
        return {0: 0, 1: 0}
    counts = {}
    bridge = getattr(PIC, "_jefipic_unit_bridge", get_unit_bridge(cfg))
    for sid in list(PIC.pos.keys()):
        mask = antenna_mask_cp(PIC.pos[sid], cfg, bridge=bridge)
        idx = cp.where(mask)[0].get()
        counts[sid] = respawn_species(PIC, sid, idx, cfg, rng)
    return counts


# -----------------------------------------------------------------------------
# Geometry / STL
# -----------------------------------------------------------------------------

def make_straight_dipole_stl(cfg: BaseConfig, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generate_center_fed_dipole(
        out_path,
        total_length=cfg.total_length,
        radius=cfg.radius,
        feed_gap=cfg.feed_gap,
        n_phi=cfg.n_phi,
        n_z_per_arm=cfg.n_z_per_arm,
        n_r_cap=cfg.n_r_cap,
    )
    return out_path


def _normal_of_triangle(tri: np.ndarray) -> np.ndarray:
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    norm = np.linalg.norm(n)
    return n / norm if norm > 1.0e-300 else np.zeros(3)


def _orthonormal_basis(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ez = np.asarray(direction, dtype=np.float64)
    ez = ez / np.linalg.norm(ez)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tmp, ez)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    ex = np.cross(tmp, ez)
    ex = ex / np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    return ex, ey, ez


def _circle_point(center, ex, ey, radius, phi):
    return center + radius * (math.cos(phi) * ex + math.sin(phi) * ey)


def add_cylinder_between(triangles, p0, p1, radius, n_phi=32, n_len=8, cap0=True, cap1=True, n_r_cap=4):
    """Append triangles for a cylinder segment between p0 and p1."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    axis = p1 - p0
    length = np.linalg.norm(axis)
    if length <= 0:
        return
    ex, ey, ez = _orthonormal_basis(axis)
    phis = np.linspace(0.0, 2.0 * math.pi, n_phi + 1)
    ts = np.linspace(0.0, 1.0, n_len + 1)

    def pt(it, ip):
        c = p0 + ts[it] * axis
        return _circle_point(c, ex, ey, radius, phis[ip])

    for iz in range(n_len):
        for ip in range(n_phi):
            p00 = pt(iz, ip)
            p01 = pt(iz, ip + 1)
            p10 = pt(iz + 1, ip)
            p11 = pt(iz + 1, ip + 1)
            triangles.append(np.array([p00, p01, p11]))
            triangles.append(np.array([p00, p11, p10]))

    def add_cap(center, normal_sign):
        n_r = max(1, int(n_r_cap))
        for ir in range(n_r):
            r0 = ir / n_r
            r1 = (ir + 1) / n_r
            for ip in range(n_phi):
                c0 = center
                p_i0 = _circle_point(center, ex, ey, radius * r0, phis[ip])
                p_i1 = _circle_point(center, ex, ey, radius * r0, phis[ip + 1])
                p_o0 = _circle_point(center, ex, ey, radius * r1, phis[ip])
                p_o1 = _circle_point(center, ex, ey, radius * r1, phis[ip + 1])
                if ir == 0:
                    tri = np.array([c0, p_o0, p_o1]) if normal_sign > 0 else np.array([c0, p_o1, p_o0])
                    triangles.append(tri)
                else:
                    if normal_sign > 0:
                        triangles.append(np.array([p_i0, p_o0, p_o1]))
                        triangles.append(np.array([p_i0, p_o1, p_i1]))
                    else:
                        triangles.append(np.array([p_i0, p_o1, p_o0]))
                        triangles.append(np.array([p_i0, p_i1, p_o1]))

    if cap0:
        add_cap(p0, normal_sign=-1)
    if cap1:
        add_cap(p1, normal_sign=+1)


def write_ascii_stl(triangles, path: Path, name="geometry") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"solid {name}\n")
        for tri in triangles:
            n = _normal_of_triangle(tri)
            f.write(f"  facet normal {n[0]:.12e} {n[1]:.12e} {n[2]:.12e}\n")
            f.write("    outer loop\n")
            for v in tri:
                f.write(f"      vertex {v[0]:.12e} {v[1]:.12e} {v[2]:.12e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {name}\n")
    return path


def make_bent_dipole_stl(cfg: BaseConfig, out_path: Path) -> Path:
    """Generate a center-fed two-arm dipole whose arms bend by 90 degrees.

    Geometry:
      - upper arm starts from +z feed and bends toward +x;
      - lower arm starts from -z feed and bends toward -x.
    """
    L_arm = 0.5 * (cfg.total_length - cfg.feed_gap)
    L1 = 0.45 * L_arm
    L2 = 0.55 * L_arm
    g = 0.5 * cfg.feed_gap
    r = cfg.radius
    tris = []

    # Upper arm: +z then +x
    p_feed_u = np.array([0.0, 0.0, g])
    p_elbow_u = np.array([0.0, 0.0, g + L1])
    p_tip_u = np.array([L2, 0.0, g + L1])

    # Lower arm: -z then -x
    p_feed_l = np.array([0.0, 0.0, -g])
    p_elbow_l = np.array([0.0, 0.0, -g - L1])
    p_tip_l = np.array([-L2, 0.0, -g - L1])

    add_cylinder_between(tris, p_feed_u, p_elbow_u, r, cfg.n_phi, max(3, cfg.n_z_per_arm // 2), cap0=True, cap1=False, n_r_cap=cfg.n_r_cap)
    add_cylinder_between(tris, p_elbow_u, p_tip_u, r, cfg.n_phi, max(3, cfg.n_z_per_arm // 2), cap0=False, cap1=True, n_r_cap=cfg.n_r_cap)
    add_cylinder_between(tris, p_feed_l, p_elbow_l, r, cfg.n_phi, max(3, cfg.n_z_per_arm // 2), cap0=True, cap1=False, n_r_cap=cfg.n_r_cap)
    add_cylinder_between(tris, p_elbow_l, p_tip_l, r, cfg.n_phi, max(3, cfg.n_z_per_arm // 2), cap0=False, cap1=True, n_r_cap=cfg.n_r_cap)
    return write_ascii_stl(tris, out_path, name="bent_center_fed_dipole")


# -----------------------------------------------------------------------------
# MAS and JefiPIC setup
# -----------------------------------------------------------------------------


def enable_axis_reduced_mas_solve(mas, axis=(0.0, 0.0, 1.0), rcond=1.0e-8, ridge=1.0e-10):
    """Constrain the straight-dipole MAS passive solve to an axial tangential basis.

    This is intended for the Case 2 straight-dipole quasi-static sheath test only.
    The general vector MAS solve has three current components per auxiliary source.
    For a thin straight dipole, the physically relevant passive current is mainly
    axial and tangential to the conductor.  Solving all three components with a
    pseudo-inverse can amplify weak particle shot noise into large normal/null-space
    currents.  This reduced solve uses one scalar unknown per useful surface point:

        J_j = a_j b_j,

    where b_j is the antenna axis projected onto the local tangent plane.
    """
    if str(mas.geometry.metadata.get("spatial_solver", "legacy_cartesian_square")).lower() == "tangential_oversampled":
        raise RuntimeError(
            "mas_axis_reduced_solve is a legacy Case-2 specialization and cannot be stacked on the Step-2G "
            "two-tangential-DOF solver. Disable mas_axis_reduced_solve; Step-2G already removes the normal DOF."
        )
    import types

    axis = np.asarray(axis, dtype=np.float64).reshape(3,)
    axis = axis / max(np.linalg.norm(axis), 1.0e-300)

    normals = np.asarray(mas.geometry.ant_normals, dtype=np.float64)
    basis = axis[None, :] - np.sum(normals * axis[None, :], axis=1)[:, None] * normals
    bn = np.linalg.norm(basis, axis=1)
    active_mask = bn > 1.0e-8

    basis[active_mask] /= bn[active_mask, None]
    basis[~active_mask] = 0.0

    n = mas.geometry.num_points
    cols = np.where(active_mask)[0]
    B = np.zeros((3 * n, len(cols)), dtype=np.float64)
    for col, j in enumerate(cols):
        B[3 * j:3 * j + 3, col] = basis[j]

    Gred = mas.geometry.G @ B

    if ridge and ridge > 0.0:
        scale = max(float(np.linalg.norm(Gred, ord=2)), 1.0)
        A = Gred.T @ Gred + (float(ridge) * scale) ** 2 * np.eye(Gred.shape[1])
        Gred_solver = np.linalg.solve(A, Gred.T)
    else:
        Gred_solver = np.linalg.pinv(Gred, rcond=float(rcond))

    # Avoid applying z-directed active current on end caps, where z is normal to
    # the local surface and does not represent a physical wire current.
    if hasattr(mas.active_excitation, "spatial_weights"):
        mas.active_excitation.spatial_weights = active_mask.astype(np.float64)

    original_solve_passive = mas.solve_passive_current

    def solve_passive_current_reduced(self, plasma_E_surface, active_current=None):
        if active_current is None:
            active_current = np.zeros((self.geometry.num_points, 3), dtype=np.float64)

        K_prev, P_prev = self._previous_temporal_state()
        E_history = self._known_history_E(K_prev, P_prev)
        E_active_surface = self._apply_G_physical(active_current)
        rhs = -self._project_tangential(plasma_E_surface + E_history + E_active_surface).reshape(-1)

        coeff = Gred_solver @ rhs
        passive_flat = B @ coeff
        passive = passive_flat.reshape(-1, 3)

        if not np.all(np.isfinite(passive)):
            raise FloatingPointError("Reduced-axis MAS passive current solve produced non-finite values")

        return passive

    mas.solve_passive_current_unreduced = original_solve_passive
    mas.solve_passive_current = types.MethodType(solve_passive_current_reduced, mas)

    mas.axis_reduced_basis = basis
    mas.axis_reduced_active_mask = active_mask
    mas.axis_reduced_metadata = {
        "enabled": True,
        "num_scalar_dofs": int(len(cols)),
        "num_original_vector_dofs": int(3 * n),
        "axis": axis.tolist(),
        "rcond": float(rcond),
        "ridge": float(ridge),
    }
    return mas


def build_mas_device(cfg: BaseConfig, stl_path: Path, grid_shape, spacing, origin, *,
                     current_amplitude: Optional[float] = None):
    amp = cfg.current_amplitude if current_amplitude is None else current_amplitude
    geometry_params = {
        "kind": "stl",
        "stl_path": stl_path,
        "source_offset": cfg.mas_source_offset,
        "max_points": cfg.max_mas_points,
        "patch_min_points": cfg.patch_min_points,
        "patch_normal_axis": (0.0, 0.0, 1.0),
        "patch_normal_threshold": 0.75,
        "diagonal_loading": cfg.diagonal_loading,
        "use_pinv": cfg.use_pinv,
        "spatial_solver": getattr(cfg, "mas_spatial_solver", "tangential_oversampled"),
        "collocation_oversample": int(getattr(cfg, "mas_collocation_oversample", 2)),
        "tikhonov_rel": float(getattr(cfg, "mas_tikhonov_rel", 1.0e-4)),
        "svd_rcond": float(getattr(cfg, "mas_svd_rcond", 1.0e-10)),
        "target_patch_size": getattr(cfg, "mas_target_patch_size", None),
        "max_subdivision_levels": int(getattr(cfg, "mas_stl_max_subdivision_levels", 6)),
    }
    active = SinusoidalCurrentExcitation(
        amplitude=amp,
        frequency=cfg.f_drive,
        direction=(0.0, 0.0, 1.0),
        ramp_time=cfg.ramp_periods / cfg.f_drive,
    )
    simulation_params = {
        "dt": cfg.dt,
        "epsilon0": EPS0,
        "c": C0,
        "max_time_steps": cfg.len_time_snapshots,
        "grid_size": grid_shape,
        "grid_spacing": spacing,
        "grid_origin": origin,
        "active_excitation": active,
        "source_softening": 0.25 * cfg.mas_source_offset,
        "source_truncation_radius": None,
        "surface_charge_ridge": float(getattr(cfg, "surface_charge_ridge", 0.0)),
    }
    mas = create_mas_antenna_simulation(geometry_params, simulation_params)
    if getattr(cfg, "mas_axis_reduced_solve", False):
        mas = enable_axis_reduced_mas_solve(
            mas,
            axis=(0.0, 0.0, 1.0),
            rcond=getattr(cfg, "mas_axis_reduced_rcond", 1.0e-8),
            ridge=getattr(cfg, "mas_axis_reduced_ridge", 1.0e-10),
        )
    return mas


def build_vacuum_mas_only(cfg: BaseConfig, stl_path: Path):
    grid_shape = (cfg.grid_n, cfg.grid_n, cfg.grid_n)
    dx = cfg.domain_size / cfg.grid_n
    spacing = (dx, dx, dx)
    origin = (-0.5 * cfg.domain_size, -0.5 * cfg.domain_size, -0.5 * cfg.domain_size)
    return build_mas_device(cfg, stl_path, grid_shape, spacing, origin)




def sync_emsolver_lhqcd_units(solver, bridge: JefiPICUnitBridge, dx_code: float, dy_code: float, dz_code: float):
    """Force EMsolver scalar constants to the JefiPIC code-unit system.

    This is intentionally defensive.  The legacy JefiPIC notebook passes the
    fixed LHQCD constants directly to EMsolver:
        epsilon0 = 90631227428685.16
        c        = 2.9979090484247153

    If any constructor overload, stale script, or accidental import leaves
    epsilon0/c/coeff_E in SI, the plasma field is amplified by ~1e25 and MAS
    sees a nonphysical startup impulse.  This helper recomputes the kernel
    coefficients from the code-unit grid and code-unit constants immediately
    after EMsolver construction.
    """
    solver.epsilon0 = float(bridge.epsilon0_code)
    solver.c = float(bridge.c_code)
    solver.coeff_E = 1.0 / (4.0 * math.pi * solver.epsilon0) * float(dx_code) * float(dy_code) * float(dz_code)
    solver.coeff_B = -1.0 / (4.0 * math.pi * solver.epsilon0 * solver.c ** 2) * float(dx_code) * float(dy_code) * float(dz_code)
    return solver


def assert_emsolver_lhqcd_units(solver, bridge: JefiPICUnitBridge, dx_code: float, dy_code: float, dz_code: float):
    expected_coeff_E = 1.0 / (4.0 * math.pi * float(bridge.epsilon0_code)) * float(dx_code) * float(dy_code) * float(dz_code)
    expected_coeff_B = -1.0 / (4.0 * math.pi * float(bridge.epsilon0_code) * float(bridge.c_code) ** 2) * float(dx_code) * float(dy_code) * float(dz_code)

    checks = [
        ("epsilon0", float(getattr(solver, "epsilon0")), float(bridge.epsilon0_code)),
        ("c", float(getattr(solver, "c")), float(bridge.c_code)),
        ("coeff_E", float(getattr(solver, "coeff_E")), expected_coeff_E),
        ("coeff_B", float(getattr(solver, "coeff_B")), expected_coeff_B),
    ]
    for name, actual, expected in checks:
        scale = max(1.0, abs(expected))
        if not np.isfinite(actual) or abs(actual - expected) / scale > 1.0e-12:
            raise RuntimeError(
                f"EMsolver unit mismatch for {name}: actual={actual:.16e}, expected={expected:.16e}. "
                "JefiPIC plasma solver must use LHQCD code constants while MAS remains SI."
            )


def register_mas_with_em_solver_unit_bridge(mas, solver, bridge: JefiPICUnitBridge):
    """Register SI MAS geometry with the JefiPIC EM solver in code units."""
    src_code = bridge.length_si_to_code(mas.geometry.ant_src_points)
    solve_obs_si = getattr(mas.geometry, "solve_obs_points", mas.geometry.ant_obs_points)
    obs_code = bridge.length_si_to_code(solve_obs_si)
    weights_code = bridge.area_si_to_code(mas.geometry.source_weights)
    softening_code = float(mas.source_softening) * bridge.L

    solver.set_antenna_sources(
        src_code,
        source_weights=weights_code,
        softening_distance=softening_code,
    )

    if hasattr(solver, "set_spatial_truncation_radius") and (
        getattr(mas, "source_truncation_radius", 0.0) > 0.0
        or getattr(mas, "source_taper_type", 0) > 0
    ):
        solver.set_spatial_truncation_radius(
            point_radius=float(getattr(mas, "source_truncation_radius", 0.0)) * bridge.L,
            point_taper=mas._source_taper_name() if hasattr(mas, "_source_taper_name") else None,
            point_taper_width=float(getattr(mas, "source_taper_width", 0.0)) * bridge.L,
        )

    if hasattr(solver, "register_sample_points"):
        solver.register_sample_points(obs_code)


def _as_numpy_field(field):
    if hasattr(field, "get"):
        field = field.get()
    return np.asarray(field, dtype=np.float64)


def unit_bridge_mas_update_fields(mas, solver, bridge: JefiPICUnitBridge,
                                  plasma_E_surface_si: np.ndarray,
                                  return_gpu_fields: bool = True):
    """Run one MAS feedback update with explicit SI<->JefiPIC conversion."""
    total_current_si = mas.solve_currents_from_surface_field(plasma_E_surface_si)
    total_polarization_si = mas.get_surface_polarization()

    total_current_code = bridge.surface_current_density_si_to_code(total_current_si)
    total_polarization_code = bridge.surface_polarization_density_si_to_code(total_polarization_si)

    expected = getattr(solver, "point_source_count", 0)
    if expected != mas.geometry.num_points:
        register_mas_with_em_solver_unit_bridge(mas, solver, bridge)

    solver.recompute_current_fields(
        antenna_surface_currents=total_current_code,
        antenna_polarization=total_polarization_code,
        include_antenna=True,
    )
    Ex, Ey, Ez, Bx, By, Bz = solver.acquire_EB_field()

    mas.last_field_data = {
        "incident": None,
        "antenna": None,
        "scattered": None,
        "total": {"E_components": (Ex, Ey, Ez), "B_components": (Bx, By, Bz)},
    }
    mas.advance_time_step()

    if return_gpu_fields:
        return (Ex, Ey, Ez), (Bx, By, Bz)

    E = np.stack([_as_numpy_field(Ex), _as_numpy_field(Ey), _as_numpy_field(Ez)], axis=-1)
    B = np.stack([_as_numpy_field(Bx), _as_numpy_field(By), _as_numpy_field(Bz)], axis=-1)
    return E.reshape(-1), B.reshape(-1)



def _surface_charge_density_si_to_code_compat(bridge: JefiPICUnitBridge, sigma_si):
    """Compatibility helper for older bridges without surface-charge converters."""
    if hasattr(bridge, "surface_charge_density_si_to_code"):
        return bridge.surface_charge_density_si_to_code(sigma_si)
    return np.asarray(sigma_si, dtype=np.float64) * (float(bridge.Q) / (float(bridge.L) * float(bridge.L)))


def unit_bridge_mas_update_fields_with_surface_charge(
    mas,
    solver,
    bridge: JefiPICUnitBridge,
    plasma_E_surface_si: np.ndarray,
    plasma_phi_surface_si: np.ndarray,
    conductor_net_charge_C: float,
    *,
    include_current_sources: bool = False,
    return_gpu_fields: bool = True,
):
    """Route-2 MAS update for a floating conductor represented by surface charge.

    The standard ``unit_bridge_mas_update_fields`` routine writes MAS surface
    currents into the EM solver.  Case 2 instead solves the connected-conductor
    electrostatic closure

        phi_plasma(surface) + phi_sigma(surface) = V_f,
        integral sigma dS = Q_conductor,

    then writes ``sigma`` to the EM solver's point-rho source channel.  The PIC
    grid rho/J history remains the plasma source only; conductor charge is kept
    in the separate MAS point-source history.
    """
    if not hasattr(mas, "solve_floating_surface_charge"):
        raise RuntimeError(
            "MAS object lacks solve_floating_surface_charge(); install the "
            "surface-charge version of EMsolver/mas_integration.py."
        )
    if not hasattr(solver, "recompute_current_fields") or not hasattr(solver, "push_antenna_charges"):
        raise RuntimeError(
            "EMsolver lacks point-rho support; install the surface-charge "
            "version of EMsolver/solver.py and EMsolver/cuda_functions.py."
        )

    plasma_E_surface_si = np.asarray(plasma_E_surface_si, dtype=np.float64).reshape(mas.geometry.num_points, 3)
    plasma_phi_surface_si = np.asarray(plasma_phi_surface_si, dtype=np.float64).reshape(mas.geometry.num_points)

    # Solve in SI units.  This also updates mas.last_surface_charge_info and the
    # diagnostic tangential-E residual computed with the same SI surface operators.
    sigma_si = mas.solve_floating_surface_charge(
        plasma_phi_surface_si,
        float(conductor_net_charge_C),
        incident_E_surface=plasma_E_surface_si,
    )
    sigma_code = _surface_charge_density_si_to_code_compat(bridge, sigma_si)

    expected = getattr(solver, "point_source_count", 0)
    if expected != mas.geometry.num_points:
        register_mas_with_em_solver_unit_bridge(mas, solver, bridge)

    if include_current_sources:
        current_si = mas.solve_currents_from_surface_field(plasma_E_surface_si)
        polarization_si = mas.get_surface_polarization()
        antenna_currents_code = bridge.surface_current_density_si_to_code(current_si)
        antenna_polarization_code = bridge.surface_polarization_density_si_to_code(polarization_si)
    else:
        # Explicit zero dipole state prevents stale current/polarization history
        # in recycled solver instances; the monopole sigma branch remains separate.
        antenna_currents_code = np.zeros((mas.geometry.num_points, 3), dtype=np.float64)
        antenna_polarization_code = np.zeros((mas.geometry.num_points, 3), dtype=np.float64)

    solver.recompute_current_fields(
        antenna_surface_currents=antenna_currents_code,
        antenna_polarization=antenna_polarization_code,
        antenna_charges=sigma_code,
        include_antenna=True,
    )
    Ex, Ey, Ez, Bx, By, Bz = solver.acquire_EB_field()

    mas.last_field_data = {
        "incident": None,
        "antenna": None,
        "scattered": None,
        "total": {"E_components": (Ex, Ey, Ez), "B_components": (Bx, By, Bz)},
    }
    mas.advance_time_step()

    charge_state = mas.get_surface_charge_info() if hasattr(mas, "get_surface_charge_info") else {}
    # Stable aliases used by older/newer article scripts and summaries.
    if "unregularized_kkt_condition_number" in charge_state:
        charge_state.setdefault(
            "floating_surface_charge_unregularized_kkt_condition_number",
            charge_state["unregularized_kkt_condition_number"],
        )
        charge_state.setdefault(
            "surface_charge_unregularized_kkt_condition_number",
            charge_state["unregularized_kkt_condition_number"],
        )
    if "regularized_kkt_condition_number" in charge_state:
        charge_state.setdefault(
            "floating_surface_charge_regularized_kkt_condition_number",
            charge_state["regularized_kkt_condition_number"],
        )
        charge_state.setdefault(
            "surface_charge_regularized_kkt_condition_number",
            charge_state["regularized_kkt_condition_number"],
        )
    if "kkt_condition_number" in charge_state:
        charge_state.setdefault("surface_charge_kkt_condition_number", charge_state["kkt_condition_number"])

    if return_gpu_fields:
        return (Ex, Ey, Ez), (Bx, By, Bz), charge_state

    E = np.stack([_as_numpy_field(Ex), _as_numpy_field(Ey), _as_numpy_field(Ez)], axis=-1)
    B = np.stack([_as_numpy_field(Bx), _as_numpy_field(By), _as_numpy_field(Bz)], axis=-1)
    return E.reshape(-1), B.reshape(-1), charge_state


def build_pic_solver_mas(cfg: BaseConfig, stl_path: Path):
    if cp is None:
        raise RuntimeError(f"CuPy is required for the full JefiPIC cases: {_CUPY_IMPORT_ERROR}")

    rng = np.random.default_rng(cfg.seed)
    bridge = get_unit_bridge(cfg)

    n = cfg.particles_per_species
    W = macro_weight(cfg, n)

    # SI grid/geometry for MAS and diagnostics.
    L_si = cfg.domain_size
    nx = ny = nz = int(cfg.grid_n)
    dx_si = dy_si = dz_si = L_si / cfg.grid_n
    x0_si = y0_si = z0_si = -0.5 * L_si

    # JefiPIC code-unit grid for PIC/EMsolver.
    L_code = float(bridge.length_si_to_code(L_si))
    dx_code = dy_code = dz_code = float(bridge.length_si_to_code(dx_si))
    x0_code = y0_code = z0_code = -0.5 * L_code
    dt_code = bridge.time_si_to_code(cfg.dt)

    if getattr(cfg, "quiet_start_pairs", False):
        if getattr(cfg, "velocity_quiet_start_pairs", False):
            base_pos_si = sample_quiet_pair_positions(n, cfg, rng)
        else:
            base_pos_si = sample_positions(n, cfg, rng)
        e_pos_si = base_pos_si.copy()
        i_pos_si = base_pos_si.copy()
    else:
        e_pos_si = sample_positions(n, cfg, rng)
        i_pos_si = sample_positions(n, cfg, rng)

    if getattr(cfg, "velocity_quiet_start_pairs", False):
        e_vel_si = sample_velocity_quiet_pairs(n, cfg.Te_eV, M_E, rng)
        i_vel_si = sample_velocity_quiet_pairs(n, cfg.Ti_eV, cfg.ion_mass, rng)
    else:
        e_vel_si = sample_velocities(n, cfg.Te_eV, M_E, rng)
        i_vel_si = sample_velocities(n, cfg.Ti_eV, cfg.ion_mass, rng)

    if getattr(cfg, "center_initial_velocities", False):
        e_vel_si -= np.mean(e_vel_si, axis=0, keepdims=True)
        i_vel_si -= np.mean(i_vel_si, axis=0, keepdims=True)

    pos = {
        0: bridge.length_si_to_code(e_pos_si),
        1: bridge.length_si_to_code(i_pos_si),
    }
    velocity = {
        0: bridge.velocity_si_to_code(e_vel_si),
        1: bridge.velocity_si_to_code(i_vel_si),
    }
    u = {0: velocity[0].copy(), 1: velocity[1].copy()}

    charges = [
        float(bridge.charge_si_to_code(-E_CHARGE * W)),
        float(bridge.charge_si_to_code(E_CHARGE * W)),
    ]
    masses = [
        float(bridge.mass_si_to_code(M_E * W)),
        float(bridge.mass_si_to_code(cfg.ion_mass * W)),
    ]
    number_charges = [W, W]

    PIC = pic_transport(
        pos, velocity, u, dt_code, charges, number_charges, masses,
        x0_code, y0_code, z0_code, nx, ny, nz, dx_code, dy_code, dz_code,
        float(bridge.velocity_si_to_code(thermal_speed(cfg.Te_eV, M_E))),
        float(bridge.velocity_si_to_code(thermal_speed(cfg.Te_eV, M_E))),
        bridge.c_code,
        cfg.use_periodic,
        cfg.use_reflective,
    )
    PIC._jefipic_unit_bridge = bridge

    requested_particle_boundary = str(getattr(cfg, "particle_boundary_mode", "auto")).lower()
    if requested_particle_boundary != "auto":
        if requested_particle_boundary not in ("reflective", "periodic", "open", "thermal_open"):
            raise ValueError(
                f"Unsupported particle_boundary_mode={requested_particle_boundary!r}; "
                "use auto, reflective, periodic, open, or thermal_open."
            )
        PIC.set_boundary_mode(requested_particle_boundary)

    if requested_particle_boundary == "thermal_open":
        if int(cfg.particle_order) != 1:
            raise ValueError("Step 1-F thermal_open boundary is continuity-validated only for CIC/particle_order=1")
        PIC.configure_thermal_open_reservoir(
            {
                0: float(bridge.velocity_si_to_code(thermal_speed(cfg.Te_eV, M_E))),
                1: float(bridge.velocity_si_to_code(thermal_speed(cfg.Ti_eV, cfg.ion_mass))),
            },
            seed=int(cfg.seed) + int(getattr(cfg, "open_reservoir_seed_offset", 1701)),
        )

    if hasattr(PIC, "set_particle_boundary_guard_cells"):
        PIC.set_particle_boundary_guard_cells(
            float(getattr(cfg, "particle_center_guard_cells", 0.0))
            if getattr(PIC, "boundary_mode", None) == "reflective"
            else 0.0
        )

    # Case 2 is intended to be a quasi-electrostatic sheath test.  Do not
    # hard-code the full electrodynamic Jefimenko kernel here: the full kernel
    # contains d(rho)/dt and dJ/dt terms, which create a large nonphysical
    # startup impulse when a finite-macro-particle plasma history is first
    # coupled into MAS.  Plasma cases that truly need full EM can set
    # cfg.quasi_electrostatic = False explicitly.
    is_electrostatic = 1 if getattr(cfg, "quasi_electrostatic", True) else 0
    PIC_provider = EMsolver(
        cfg.len_time_snapshots,
        nx, ny, nz,
        nx, ny, nz,
        dx_code, dy_code, dz_code, x0_code, y0_code, z0_code,
        dx_code, dy_code, dz_code, x0_code, y0_code, z0_code,
        dt_code, bridge.epsilon0_code, bridge.c_code, is_electrostatic,
    )
    sync_emsolver_lhqcd_units(PIC_provider, bridge, dx_code, dy_code, dz_code)
    assert_emsolver_lhqcd_units(PIC_provider, bridge, dx_code, dy_code, dz_code)
    if getattr(cfg, "debug_unit_bridge", False):
        print(
            "[DEBUG EMsolver units]",
            "bridge.mode=", bridge.mode,
            "bridge.epsilon0_code=", bridge.epsilon0_code,
            "bridge.c_code=", bridge.c_code,
            "solver.epsilon0=", PIC_provider.epsilon0,
            "solver.c=", PIC_provider.c,
            "solver.dt=", PIC_provider.dt,
            "solver.dx_s=", PIC_provider.dx_s,
            "solver.coeff_E=", PIC_provider.coeff_E,
            "solver.coeff_B=", PIC_provider.coeff_B,
        )
    grid_current_layout = "staggered_face"
    PIC_provider.set_grid_current_layout(grid_current_layout)
    requested_kernel_mode = str(getattr(cfg, "jefipic_kernel_mode", "atomic")).strip().lower()
    PIC_provider.set_kernel_mode(requested_kernel_mode)

    lamD_si = debye_length(cfg.n0, cfg.Te_eV)
    PIC_provider.set_spatial_truncation_radius(
        grid_radius=float(bridge.length_si_to_code(cfg.spatial_truncation_debye * lamD_si)),
        point_radius=None if cfg.point_radius is None else float(bridge.length_si_to_code(cfg.point_radius)),
        taper=cfg.taper,
        taper_width=float(bridge.length_si_to_code(cfg.taper_width_debye * lamD_si)),
    )

    mas = build_mas_device(
        cfg,
        stl_path,
        (nx, ny, nz),
        (dx_si, dy_si, dz_si),
        (x0_si, y0_si, z0_si),
    )
    register_mas_with_em_solver_unit_bridge(mas, PIC_provider, bridge)

    Ex = cp.zeros(nx * ny * nz, dtype=cp.float64)
    Ey = cp.zeros(nx * ny * nz, dtype=cp.float64)
    Ez = cp.zeros(nx * ny * nz, dtype=cp.float64)
    Bx = cp.zeros(nx * ny * nz, dtype=cp.float64)
    By = cp.zeros(nx * ny * nz, dtype=cp.float64)
    Bz = cp.zeros(nx * ny * nz, dtype=cp.float64)

    return {
        "PIC": PIC,
        "solver": PIC_provider,
        "mas": mas,
        "rng": rng,
        "macro_weight": W,
        "unit_bridge": bridge,
        "shape": (nx, ny, nz),
        "spacing": (dx_code, dy_code, dz_code),
        "origin": (x0_code, y0_code, z0_code),
        "spacing_si": (dx_si, dy_si, dz_si),
        "origin_si": (x0_si, y0_si, z0_si),
        "initial_positions_si": {0: e_pos_si.copy(), 1: i_pos_si.copy()},
        "fields": (Ex, Ey, Ez, Bx, By, Bz),
    }


def plasma_to_mas_ramp_scale(step: int, cfg: BaseConfig) -> float:
    """Smoothly turn on the particle/plasma incident field seen by MAS.

    This suppresses a nonphysical startup impulse caused by coupling a finite-
    macro-particle plasma field into an initially empty MAS current history.
    The default BaseConfig value is zero ramp steps, giving scale 1.0 and thus
    preserving existing behavior for all other cases.
    """
    n_ramp = int(getattr(cfg, "plasma_to_mas_ramp_steps", 0))
    if n_ramp <= 0:
        return 1.0
    x = min(max(float(step) / float(n_ramp), 0.0), 1.0)
    return float(x * x * (3.0 - 2.0 * x))


def _finite_stats_for_audit(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {'count': 0, 'min': None, 'median': None, 'mean': None, 'max': None, 'final': None}
    return {
        'count': int(arr.size),
        'min': float(np.min(arr)),
        'median': float(np.median(arr)),
        'mean': float(np.mean(arr)),
        'max': float(np.max(arr)),
        'final': float(arr[-1]),
    }


def write_source_closure_audit(history: Dict[str, list], output_dir: Path, label: str, cfg: BaseConfig, solver=None, mas=None):
    """Write a compact JSON/TXT report for the Step-1 source-closure audit."""
    keys = [
        'continuity_face_linf_rel',
        'continuity_face_rms_rel',
        'continuity_cell_linf_rel',
        'continuity_cell_rms_rel',
        'face_to_cell_divergence_defect_linf_rel',
        'face_to_cell_divergence_defect_rms_rel',
        'mas_local_J_over_dt_rms',
        'mas_history_dJdt_rms',
        'mas_local_vs_history_dJdt_rel',
        'mas_local_to_history_dJdt_norm_ratio',
        'pass2_point_rho_rms_code',
        'pass2_point_history_dJdt_rms_code',
        'mas_Kdot_rms',
        'mas_temporal_continuity_residual',
        'mas_local_vs_pass2_Kdot_rel',
        'pass2_point_P_rms_code',
        'pass2_point_K_rms_code',
        'pass2_point_temporal_continuity_residual',
        'actual_jefimenko_continuity_linf_rel',
        'actual_jefimenko_continuity_rms_rel',
    ]
    grid_current_layout = str(
        history.get(
            'jefimenko_grid_current_layout',
            getattr(solver, 'grid_current_layout', getattr(cfg, 'grid_current_layout', 'cell')),
        )
    )
    actual_definition = (
        'R_actual=R_face because native staggered J_face is passed directly to Jefimenko'
        if grid_current_layout == 'staggered_face'
        else 'R_actual=R_cell because reconstructed cell-centered J is passed to Jefimenko'
    )
    report = {
        'label': label,
        'purpose': 'Step-1 audit of PIC continuity, actual Jefimenko grid-source layout, MAS current/rho closure, and time-local versus Pass-2 temporal source semantics.',
        'physical_algorithm_changed': bool(grid_current_layout == 'staggered_face'),
        'definitions': {
            'PIC_face': 'R_face=(rho^n-rho^{n-1})/dt + D_f J_face',
            'PIC_cell': 'R_cell=(rho^n-rho^{n-1})/dt + D_f R_fc(J_cell), retained as the legacy reconstruction diagnostic',
            'PIC_actual_Jefimenko': actual_definition,
            'MAS_local': 'Step-2B electric-dipole local operator uses P^n=P^(n-1)+dt/2(K^(n-1)+K^n) and Kdot^n=(K^n-K^(n-1))/dt',
            'Pass2_temporal': 'retarded dipole kernel uses piecewise-linear K, its exact piecewise-quadratic integral P, and the same interval Kdot',
        },
        'statistics': {key: _finite_stats_for_audit(history.get(key, [])) for key in keys},
        'booleans': {
            'mas_continuity_linked_rho_constructed_any': bool(any(history.get('mas_continuity_linked_rho_constructed', []))),
            'pass2_point_J_written_all_sampled': bool(all(history.get('pass2_point_J_written', [False]))),
            'pass2_point_dipole_written_all_sampled': bool(all(history.get('pass2_point_J_written', [False]))),
            'pass2_point_rho_written_any_sampled': bool(any(history.get('pass2_point_rho_written', []))),
            'pass2_point_current_E_term_enabled_all_sampled': bool(all(history.get('pass2_point_current_E_term_enabled', [False]))),
        },
        'solver_mode': {
            'quasi_electrostatic': bool(getattr(cfg, 'quasi_electrostatic', False)),
            'grid_current_layout': grid_current_layout,
            'kernel_mode': str(getattr(solver, 'kernel_mode', getattr(cfg, 'jefipic_kernel_mode', 'unknown'))),
            'actual_grid_current_E_term_enabled': bool(
                grid_current_layout == 'staggered_face' and not bool(getattr(cfg, 'quasi_electrostatic', False))
            ) if grid_current_layout == 'staggered_face' else bool(not getattr(cfg, 'quasi_electrostatic', False)),
            'note': (
                'Step-2B MAS uses a continuity-linked electric-dipole source. Full EM includes 1/R^3 dipole, 1/R^2 induction, and 1/R radiation terms; quasi-electrostatic mode retains the instantaneous dipole field.'
            ),
        },
    }
    if mas is not None and hasattr(mas, 'get_source_closure_diagnostics'):
        report['latest_mas_diagnostic'] = mas.get_source_closure_diagnostics()
    if solver is not None and hasattr(solver, 'get_point_source_closure_diagnostics'):
        report['latest_pass2_point_diagnostic'] = solver.get_point_source_closure_diagnostics()

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f'{label}_source_closure_audit.json'
    json_path.write_text(json.dumps(report, indent=2, default=float), encoding='utf-8')

    lines = [
        f'Source-closure audit: {label}',
        '=' * (22 + len(label)),
        '',
        (
            'Step 1-E uses the native staggered PIC current in the actual Jefimenko grid-source path.'
            if grid_current_layout == 'staggered_face'
            else 'Legacy comparison run: Jefimenko uses reconstructed cell-centered PIC current.'
        ),
        f'Jefimenko grid-current layout: {grid_current_layout}',
        f"Jefimenko kernel mode: {report['solver_mode']['kernel_mode']}",
        '',
    ]
    for key in keys:
        st = report['statistics'][key]
        lines.append(f"{key}: median={st['median']} max={st['max']} final={st['final']} (n={st['count']})")
    lines += [
        '',
        f"MAS continuity-linked rho constructed: {report['booleans']['mas_continuity_linked_rho_constructed_any']}",
        f"Pass-2 point rho written in sampled RF-current steps: {report['booleans']['pass2_point_rho_written_any_sampled']}",
        f"Pass-2 point-current electric term enabled for all sampled steps: {report['booleans']['pass2_point_current_E_term_enabled_all_sampled']}",
        f"quasi_electrostatic: {report['solver_mode']['quasi_electrostatic']}",
        f"actual grid-current E term enabled: {report['solver_mode']['actual_grid_current_E_term_enabled']}",
        '',
        report['solver_mode']['note'],
    ]
    txt_path = output_dir / f'{label}_source_closure_audit.txt'
    txt_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


def run_full_jefipic_mas_loop(cfg: BaseConfig, stl_path: Path, output_dir: Path,
                              label: str = "run") -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_pic_solver_mas(cfg, stl_path)
    PIC = bundle["PIC"]
    solver = bundle["solver"]
    mas = bundle["mas"]
    rng = bundle["rng"]
    W = bundle["macro_weight"]
    bridge = bundle.get("unit_bridge", get_unit_bridge(cfg))
    nx, ny, nz = bundle["shape"]
    dx, dy, dz = bundle["spacing"]  # JefiPIC code-unit spacing
    Ex, Ey, Ez, Bx, By, Bz = bundle["fields"]

    history = {
        "step": [],
        "time_s": [],
        "mas_rms_tangential_E": [],
        "mas_max_tangential_E": [],
        "I_active_z": [],
        "I_passive_z": [],
        "I_total_z": [],
        "J_active_norm": [],
        "J_passive_norm": [],
        "J_total_norm": [],
        "J_active_max": [],
        "J_passive_max": [],
        "J_total_max": [],
        "incident_E_surface_rms": [],
        "incident_E_surface_max": [],
        "plasma_to_mas_ramp_scale": [],
        "absorbed_electrons": [],
        "absorbed_ions": [],
        "absorbed_electrons_cumulative": [],
        "absorbed_ions_cumulative": [],
        "isolated_antenna_charge_C_equiv": [],
        "active_electrons": [],
        "active_ions": [],
        # Step-1 source-closure audit: PIC deposition/reconstruction.
        "continuity_face_linf_abs": [],
        "continuity_face_linf_rel": [],
        "continuity_face_rms_rel": [],
        "continuity_cell_linf_abs": [],
        "continuity_cell_linf_rel": [],
        "continuity_cell_rms_rel": [],
        "face_to_cell_divergence_defect_linf_rel": [],
        "face_to_cell_divergence_defect_rms_rel": [],
        # Step-1 source-closure audit: time-local MAS vs Pass-2 history.
        "mas_local_J_over_dt_rms": [],
        "mas_history_dJdt_rms": [],
        "mas_local_vs_history_dJdt_rel": [],
        "mas_local_to_history_dJdt_norm_ratio": [],
        "mas_continuity_linked_rho_constructed": [],
        # Exact point histories used by the GPU Pass-2 kernel.
        "pass2_point_J_written": [],
        "pass2_point_rho_written": [],
        "pass2_point_rho_rms_code": [],
        "pass2_point_history_dJdt_rms_code": [],
        "pass2_point_current_E_term_enabled": [],
        "mas_Kdot_rms": [],
        "mas_temporal_continuity_residual": [],
        "mas_local_vs_pass2_Kdot_rel": [],
        "pass2_point_P_rms_code": [],
        "pass2_point_K_rms_code": [],
        "pass2_point_temporal_continuity_residual": [],
        # Step 1-E: continuity of the current layout that the Jefimenko grid
        # kernel actually consumes.
        "actual_jefimenko_continuity_linf_rel": [],
        "actual_jefimenko_continuity_rms_rel": [],
        # Step 1-F thermal open-reservoir transport diagnostics.
        "open_reservoir_outflow_total": [],
        "open_reservoir_inflow_total": [],
        "particle_boundary_mode": [],
    }
    grid_current_layout = (
        solver.get_grid_current_layout()
        if hasattr(solver, "get_grid_current_layout")
        else getattr(cfg, "grid_current_layout", "cell")
    )
    history["jefimenko_grid_current_layout"] = str(grid_current_layout)
    history["jefimenko_kernel_mode"] = str(getattr(solver, "kernel_mode", getattr(cfg, "jefipic_kernel_mode", "unknown")))
    history["jefimenko_full_em"] = not bool(getattr(cfg, "quasi_electrostatic", False))

    last_fields = None
    absorbed_electrons_cumulative = 0
    absorbed_ions_cumulative = 0
    t0 = time.time()
    for step in range(cfg.steps):
        PIC.proceed_one_step(Ex, Ey, Ez, Bx, By, Bz, dx, dy, dz, nx, ny, nz, cfg.field_order)
        hits = absorb_particles_on_antenna(PIC, cfg, rng)
        absorbed_electrons_cumulative += int(hits.get(0, 0))
        absorbed_ions_cumulative += int(hits.get(1, 0))

        rho, Jx_face, Jy_face, Jz_face = PIC.evaluate_rho_J(dx, dy, dz, nx, ny, nz, cfg.particle_order)
        face_currents = (Jx_face, Jy_face, Jz_face)
        solver.Jefimenko_solver(
            rho.flatten(),
            include_antenna=False,
            face_currents=face_currents,
        )

        plasma_E_surface = None
        plasma_ramp = plasma_to_mas_ramp_scale(step, cfg)
        if hasattr(solver, "sample_current_E_at_points"):
            solve_obs_si = getattr(mas.geometry, "solve_obs_points", mas.geometry.ant_obs_points)
            obs_code = bridge.length_si_to_code(solve_obs_si)
            plasma_E_surface_code = solver.sample_current_E_at_points(obs_code)
            plasma_E_surface = bridge.efield_code_to_si(plasma_E_surface_code)
            if plasma_E_surface is not None and plasma_ramp != 1.0:
                plasma_E_surface = plasma_E_surface * plasma_ramp

        (Ex, Ey, Ez), (Bx, By, Bz) = unit_bridge_mas_update_fields(
            mas,
            solver,
            bridge,
            plasma_E_surface,
            return_gpu_fields=True,
        )
        last_fields = (Ex, Ey, Ez, Bx, By, Bz)

        if step % cfg.diag_stride == 0 or step == cfg.steps - 1:
            vw = mas.get_virtual_work_check()
            Ja = mas.get_active_currents()
            Jp = mas.get_passive_currents()
            Jt = mas.get_surface_currents()
            weights = np.asarray(mas.geometry.source_weights).reshape(-1)
            history["step"].append(int(step))
            history["time_s"].append(float(step * cfg.dt))
            history["mas_rms_tangential_E"].append(float(vw.get("rms_tangential_E", 0.0)))
            history["mas_max_tangential_E"].append(float(vw.get("max_tangential_E", 0.0)))
            history["I_active_z"].append(float(np.sum(Ja[:, 2] * weights)))
            history["I_passive_z"].append(float(np.sum(Jp[:, 2] * weights)))
            history["I_total_z"].append(float(np.sum(Jt[:, 2] * weights)))

            def _weighted_norm(J):
                return float(np.sqrt(np.sum(weights * np.sum(J * J, axis=1))))
            def _max_norm(J):
                return float(np.max(np.linalg.norm(J, axis=1))) if len(J) else 0.0

            history["J_active_norm"].append(_weighted_norm(Ja))
            history["J_passive_norm"].append(_weighted_norm(Jp))
            history["J_total_norm"].append(_weighted_norm(Jt))
            history["J_active_max"].append(_max_norm(Ja))
            history["J_passive_max"].append(_max_norm(Jp))
            history["J_total_max"].append(_max_norm(Jt))

            if plasma_E_surface is not None:
                En = np.linalg.norm(np.asarray(plasma_E_surface), axis=1)
                history["incident_E_surface_rms"].append(float(np.sqrt(np.mean(En * En))))
                history["incident_E_surface_max"].append(float(np.max(En)))
            else:
                history["incident_E_surface_rms"].append(float("nan"))
                history["incident_E_surface_max"].append(float("nan"))
            history["plasma_to_mas_ramp_scale"].append(float(plasma_ramp))

            # --- Step-1 source-closure audit ---------------------------------
            pic_diag = PIC.get_continuity_diagnostics() if hasattr(PIC, "get_continuity_diagnostics") else {}
            face_diag = pic_diag.get("face_current", {}) if isinstance(pic_diag, dict) else {}
            cell_diag = pic_diag.get("jefimenko_cell_current", {}) if isinstance(pic_diag, dict) else {}
            recon_diag = pic_diag.get("face_to_cell_divergence_defect", {}) if isinstance(pic_diag, dict) else {}
            history["continuity_face_linf_abs"].append(float(face_diag.get("linf_abs", np.nan)))
            history["continuity_face_linf_rel"].append(float(face_diag.get("linf_rel", np.nan)))
            history["continuity_face_rms_rel"].append(float(face_diag.get("rms_rel", np.nan)))
            history["continuity_cell_linf_abs"].append(float(cell_diag.get("linf_abs", np.nan)))
            history["continuity_cell_linf_rel"].append(float(cell_diag.get("linf_rel", np.nan)))
            history["continuity_cell_rms_rel"].append(float(cell_diag.get("rms_rel", np.nan)))
            actual_diag = face_diag if grid_current_layout == "staggered_face" else cell_diag
            history["actual_jefimenko_continuity_linf_rel"].append(
                float(actual_diag.get("linf_rel", np.nan))
            )
            history["actual_jefimenko_continuity_rms_rel"].append(
                float(actual_diag.get("rms_rel", np.nan))
            )
            history["face_to_cell_divergence_defect_linf_rel"].append(
                float(recon_diag.get("linf_rel_to_face_divergence", np.nan))
            )
            history["face_to_cell_divergence_defect_rms_rel"].append(
                float(recon_diag.get("rms_rel_to_face_divergence", np.nan))
            )

            mas_diag = mas.get_source_closure_diagnostics() if hasattr(mas, "get_source_closure_diagnostics") else {}
            mas_total_diag = mas_diag.get("total", {}) if isinstance(mas_diag, dict) else {}
            # Legacy Step-1 temporal-mismatch keys are retained as NaN so old
            # analysis scripts do not silently reinterpret the new semantics.
            history["mas_local_J_over_dt_rms"].append(float("nan"))
            history["mas_history_dJdt_rms"].append(float("nan"))
            history["mas_local_vs_history_dJdt_rel"].append(float("nan"))
            history["mas_local_to_history_dJdt_norm_ratio"].append(float("nan"))
            history["mas_Kdot_rms"].append(float(mas_total_diag.get("Kdot_weighted_rms_A_per_m_s", np.nan)))
            history["mas_temporal_continuity_residual"].append(float(mas_total_diag.get("temporal_continuity_residual_weighted_rms", np.nan)))
            history["mas_local_vs_pass2_Kdot_rel"].append(float(mas_total_diag.get("local_vs_pass2_Kdot_relative_rms", np.nan)))
            history["mas_continuity_linked_rho_constructed"].append(
                bool(mas_diag.get("continuity_linked_rho_constructed", False)) if isinstance(mas_diag, dict) else False
            )

            point_diag = solver.get_point_source_closure_diagnostics() if hasattr(solver, "get_point_source_closure_diagnostics") else {}
            history["pass2_point_J_written"].append(bool(point_diag.get("dipole_state_written_for_step", False)))
            history["pass2_point_rho_written"].append(bool(point_diag.get("rho_monopole_written_for_step", False)))
            history["pass2_point_rho_rms_code"].append(float(point_diag.get("rho_monopole_weighted_rms_code", np.nan)))
            history["pass2_point_history_dJdt_rms_code"].append(float(point_diag.get("Kdot_weighted_rms_code", np.nan)))
            history["pass2_point_current_E_term_enabled"].append(bool(point_diag.get("point_dipole_E_terms_enabled", False)))
            history["pass2_point_P_rms_code"].append(float(point_diag.get("P_weighted_rms_code_Q_per_L", np.nan)))
            history["pass2_point_K_rms_code"].append(float(point_diag.get("K_weighted_rms_code_I_per_L", np.nan)))
            history["pass2_point_temporal_continuity_residual"].append(float(point_diag.get("temporal_continuity_residual_weighted_rms", np.nan)))

            history["absorbed_electrons"].append(int(hits.get(0, 0)))
            history["absorbed_ions"].append(int(hits.get(1, 0)))
            history["absorbed_electrons_cumulative"].append(int(absorbed_electrons_cumulative))
            history["absorbed_ions_cumulative"].append(int(absorbed_ions_cumulative))
            # Diagnostic only: the charge the antenna would have accumulated if
            # it were isolated and absorbed macro-particles were not replenished.
            # The actual boundary-reservoir model keeps the computational plasma
            # supplied by external particles and does not apply this charge back
            # to the antenna potential.
            q_iso = E_CHARGE * W * (absorbed_ions_cumulative - absorbed_electrons_cumulative)
            history["isolated_antenna_charge_C_equiv"].append(float(q_iso))
            active_counts = PIC.get_active_particle_counts() if hasattr(PIC, "get_active_particle_counts") else {}
            history["active_electrons"].append(int(active_counts.get(0, 0)))
            history["active_ions"].append(int(active_counts.get(1, 0)))
            reservoir_diag = (
                PIC.get_open_reservoir_diagnostics()
                if hasattr(PIC, "get_open_reservoir_diagnostics")
                else {}
            )
            history["open_reservoir_outflow_total"].append(int(reservoir_diag.get("outflow_total", 0)))
            history["open_reservoir_inflow_total"].append(int(reservoir_diag.get("inflow_total", 0)))
            history["particle_boundary_mode"].append(str(getattr(PIC, "boundary_mode", "unknown")))

        if step % max(1, cfg.steps // 5) == 0:
            print(f"[{label}] step {step}/{cfg.steps}")

    elapsed = time.time() - t0
    history["elapsed_seconds"] = elapsed
    (output_dir / f"{label}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    source_closure_audit = write_source_closure_audit(history, output_dir, label, cfg, solver=solver, mas=mas)
    return {
        "PIC": PIC,
        "solver": solver,
        "mas": mas,
        "macro_weight": W,
        "history": history,
        "last_fields": last_fields,
        "grid_shape": (nx, ny, nz),
        "unit_bridge": bridge,
        "grid_spacing_code": (dx, dy, dz),
        "initial_positions_si": bundle.get("initial_positions_si"),
        "source_closure_audit": source_closure_audit,
        "grid_current_layout": str(grid_current_layout),
    }


# -----------------------------------------------------------------------------
# Diagnostics / plots
# -----------------------------------------------------------------------------

def save_mas_geometry_plots(mas, output_dir: Path, prefix: str = "mas"):
    output_dir.mkdir(parents=True, exist_ok=True)
    geom = mas.geometry
    obs = geom.ant_obs_points
    src = geom.ant_src_points

    # 动态获取几何中心
    y_center = (obs[:, 1].min() + obs[:, 1].max()) / 2.0
    
    # ---------------------------------------------------------
    # 1. 动态 3D 剖视图 (切除面向镜头的一半，适用于所有几何体)
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    mask_y_obs = obs[:, 1] >= (y_center - 1e-5)
    mask_y_src = src[:, 1] >= (y_center - 1e-5)

    ax.scatter(obs[mask_y_obs, 0], obs[mask_y_obs, 1], obs[mask_y_obs, 2], 
               s=10, marker="o", alpha=0.25, label="obs (surface)", color='tab:blue')
    ax.scatter(src[mask_y_src, 0], src[mask_y_src, 1], src[mask_y_src, 2], 
               s=18, marker="^", alpha=0.9, label="src (inward)", color='tab:orange')
    
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("MAS Point Cloud (Adaptive 3D Cutaway)")
    ax.view_init(elev=20, azim=-55) 
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_point_cloud_3d.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # 2. 智能 2D 截面图：自动雷达锁定核心区域
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    slice_tolerance = 0.002
    mask_obs = np.abs(obs[:, 1] - y_center) < slice_tolerance
    mask_src = np.abs(src[:, 1] - y_center) < slice_tolerance
    
    ax.scatter(obs[mask_obs, 0], obs[mask_obs, 2], s=30, marker="o", alpha=0.6, label="obs (surface)", color='tab:blue')
    ax.scatter(src[mask_src, 0], src[mask_src, 2], s=30, marker="^", alpha=1.0, label="src (inward)", color='tab:orange')
    
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.axis("equal")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    # 【智能 Zoom 核心逻辑】：找到距离空间原点最近的观测点
    # 无论是单天线、弯天线还是双天线，镜头都会精准锁定几何最关键的局部
    distances = np.linalg.norm(obs, axis=1)
    closest_idx = np.argmin(distances)
    focus_x, focus_z = obs[closest_idx, 0], obs[closest_idx, 2]
    
    # 以该焦点为中心，框选 ±3mm 的视野
    window = 0.003 
    ax.set_xlim(focus_x - window, focus_x + window)
    ax.set_ylim(focus_z - window, focus_z + window)
    ax.set_title(f"MAS 2D Cross-Section (Zoomed at x={focus_x*1000:.1f}, z={focus_z*1000:.1f} mm)")
    
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_point_cloud_slice_2d.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # 3. G 矩阵热力图 (保持原样)
    # ---------------------------------------------------------
    Gabs = np.abs(geom.G)
    eps = max(np.max(Gabs) * 1e-12, 1e-30)
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    im = ax.imshow(np.log10(Gabs + eps), aspect="auto")
    ax.set_title(r"$\log_{10}(|G|)$")
    ax.set_xlabel("source index")
    ax.set_ylabel("equation index")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}(|G|)$")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_G_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _diagnostic_slice_from_history(history: Dict[str, list], skip_initial_steps: int = 0):
    t = np.asarray(history["time_s"], dtype=float)
    if skip_initial_steps <= 0 or "step" not in history:
        return np.ones_like(t, dtype=bool)
    steps = np.asarray(history["step"], dtype=int)
    return steps >= int(skip_initial_steps)


def plot_mas_time_history(history: Dict[str, list], output_dir: Path, prefix: str = "mas",
                          skip_initial_steps: int = 0):
    t = np.asarray(history["time_s"], dtype=float)
    mask = _diagnostic_slice_from_history(history, skip_initial_steps)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(t[mask] * 1e6, np.asarray(history["mas_rms_tangential_E"])[mask], label="RMS")
    ax.plot(t[mask] * 1e6, np.asarray(history["mas_max_tangential_E"])[mask], label="max")
    ax.set_yscale("log")
    ax.set_xlabel("time (microseconds)")
    ax.set_ylabel("surface tangential E (V/m)")
    ax.set_title("MAS PEC boundary diagnostic")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_boundary_residual.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if "I_total_z" in history:
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.plot(t[mask] * 1e6, np.asarray(history["I_active_z"])[mask], label="active z-projection")
        ax.plot(t[mask] * 1e6, np.asarray(history["I_passive_z"])[mask], label="passive z-projection")
        ax.plot(t[mask] * 1e6, np.asarray(history["I_total_z"])[mask], label="total z-projection", linewidth=1.5)
        ax.set_xlabel("time (microseconds)")
        ax.set_ylabel("projected current proxy")
        ax.set_title("Projected MAS current proxy")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_current_proxy.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if "J_total_norm" in history:
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.plot(t[mask] * 1e6, np.asarray(history["J_active_norm"])[mask], label="active norm")
        ax.plot(t[mask] * 1e6, np.asarray(history["J_passive_norm"])[mask], label="passive norm")
        ax.plot(t[mask] * 1e6, np.asarray(history["J_total_norm"])[mask], label="total norm", linewidth=1.5)
        ax.set_xlabel("time (microseconds)")
        ax.set_ylabel("weighted current-norm proxy")
        ax.set_title("MAS current norm diagnostic")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_current_norm.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if "absorbed_electrons_cumulative" in history:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.plot(t * 1e6, history["absorbed_electrons_cumulative"], label="electrons")
        ax.plot(t * 1e6, history["absorbed_ions_cumulative"], label="ions")
        ax.set_xlabel("time (microseconds)")
        ax.set_ylabel("cumulative absorbed macro-particles")
        ax.set_title("Particle absorption by the antenna")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        if "isolated_antenna_charge_C_equiv" in history:
            ax2 = ax.twinx()
            ax2.plot(t * 1e6, history["isolated_antenna_charge_C_equiv"], "k:", linewidth=1.0, label="isolated-charge equivalent")
            ax2.set_ylabel("isolated-charge equivalent (C)")
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_absorbed_particles.png", dpi=300, bbox_inches="tight")
        plt.close(fig)



def radial_density_axis(PIC, cfg: BaseConfig, W: float, bins=40,
                        bridge: Optional[JefiPICUnitBridge] = None) -> Dict[str, np.ndarray]:
    """Legacy full-cylinder radial density around the dipole axis.

    This is useful as a broad diagnostic, but it mixes sidewall, end-cap, and
    feed-gap regions.  It should not be used as the primary sheath-profile
    diagnostic for the finite-length antenna.
    """
    bridge = bridge or getattr(PIC, "_jefipic_unit_bridge", get_unit_bridge(cfg))
    rmax = 0.5 * cfg.domain_size
    r_bins = np.linspace(cfg.radius, rmax, bins + 1)
    shell_vol = math.pi * (r_bins[1:] ** 2 - r_bins[:-1] ** 2) * cfg.domain_size
    out = {"s": 0.5 * (r_bins[:-1] + r_bins[1:]) - cfg.radius}
    for sid, name in [(0, "electron"), (1, "ion")]:
        pos = bridge.length_code_to_si(PIC.pos[sid].get())
        rr = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        counts, _ = np.histogram(rr, bins=r_bins)
        out[name] = counts * W / np.maximum(shell_vol, 1.0e-300)
    return out


def _sidewall_density_from_positions(
    positions_si: Dict[int, np.ndarray],
    cfg: BaseConfig,
    W: float,
    bins: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Density profile using only the cylindrical sidewalls of the two arms.

    The distance coordinate is s = r - antenna_radius.  Particles near the feed
    gap and near both antenna ends are excluded so that the profile is not
    contaminated by end-cap or gap geometry.  This diagnostic is intended for
    sheath interpretation.
    """
    bins = int(bins or getattr(cfg, "density_bins", 48))
    lamD = debye_length(cfg.n0, cfg.Te_eV)
    margin = max(float(getattr(cfg, "density_sidewall_margin_debye", 0.25)) * lamD, 0.0)

    # Two arm sidewall intervals in |z|.  If the requested margin is too large,
    # fall back to zero margin rather than returning an empty diagnostic.
    z0 = 0.5 * cfg.feed_gap + margin
    z1 = 0.5 * cfg.total_length - margin
    if z1 <= z0:
        margin = 0.0
        z0 = 0.5 * cfg.feed_gap
        z1 = 0.5 * cfg.total_length

    sidewall_length_total = 2.0 * max(z1 - z0, 1.0e-300)

    max_s = float(getattr(cfg, "density_max_distance_debye", 8.0)) * lamD
    domain_limit = 0.5 * cfg.domain_size - cfg.radius
    max_s = min(max_s, domain_limit)
    max_s = max(max_s, 2.0 * lamD)
    s_edges = np.linspace(0.0, max_s, bins + 1)
    r_edges = cfg.radius + s_edges
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])

    shell_vol = math.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2) * sidewall_length_total
    out = {
        "s": s_centers,
        "s_edges": s_edges,
        "shell_volume": shell_vol,
        "lambda_D": lamD,
        "sidewall_z_abs_min": z0,
        "sidewall_z_abs_max": z1,
        "sidewall_margin_m": margin,
    }

    for sid, name in [(0, "electron"), (1, "ion")]:
        pos = np.asarray(positions_si[sid], dtype=np.float64)
        rr = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        zabs = np.abs(pos[:, 2])
        s = rr - cfg.radius
        mask = (
            (zabs >= z0)
            & (zabs <= z1)
            & (s >= s_edges[0])
            & (s < s_edges[-1])
        )
        counts, _ = np.histogram(s[mask], bins=s_edges)
        out[f"{name}_counts"] = counts
        out[name] = counts * W / np.maximum(shell_vol, 1.0e-300)

    return out


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(np.asarray(num, dtype=np.float64), np.nan, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    good = np.isfinite(den) & (np.abs(den) > 1.0e-300)
    out[good] = np.asarray(num, dtype=np.float64)[good] / den[good]
    return out


def _write_density_csv(path: Path, final_sw: Dict[str, np.ndarray], initial_sw: Dict[str, np.ndarray], cfg: BaseConfig):
    ratio_e = _safe_ratio(final_sw["electron"], initial_sw["electron"])
    ratio_i = _safe_ratio(final_sw["ion"], initial_sw["ion"])
    header = (
        "distance_from_sidewall_m,"
        "electron_final_over_n0,ion_final_over_n0,"
        "electron_initial_over_n0,ion_initial_over_n0,"
        "electron_final_over_initial,ion_final_over_initial,"
        "electron_counts_final,ion_counts_final,"
        "electron_counts_initial,ion_counts_initial"
    )
    data = np.column_stack([
        final_sw["s"],
        final_sw["electron"] / cfg.n0,
        final_sw["ion"] / cfg.n0,
        initial_sw["electron"] / cfg.n0,
        initial_sw["ion"] / cfg.n0,
        ratio_e,
        ratio_i,
        final_sw["electron_counts"],
        final_sw["ion_counts"],
        initial_sw["electron_counts"],
        initial_sw["ion_counts"],
    ])
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def plot_plasma_diagnostics(result: Dict[str, object], cfg: BaseConfig, output_dir: Path, prefix: str = "plasma"):
    output_dir.mkdir(parents=True, exist_ok=True)
    PIC = result["PIC"]
    W = result["macro_weight"]
    bridge = result.get("unit_bridge", getattr(PIC, "_jefipic_unit_bridge", get_unit_bridge(cfg)))
    bins = int(getattr(cfg, "density_bins", 48))

    # ------------------------------------------------------------------
    # Legacy full-cylinder diagnostic, kept for broad sanity checks only.
    # ------------------------------------------------------------------
    rd = radial_density_axis(PIC, cfg, W, bins=bins, bridge=bridge)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(rd["s"], rd["electron"] / cfg.n0, label="electron / n0")
    ax.plot(rd["s"], rd["ion"] / cfg.n0, label="ion / n0")
    lamD = debye_length(cfg.n0, cfg.Te_eV)
    ax.axvline(lamD, linestyle="--", linewidth=1.0, label=r"$\lambda_D$")
    ax.axvline(2.0 * lamD, linestyle=":", linewidth=1.0, label=r"$2\lambda_D$")
    ax.set_xlabel("distance from antenna axis minus radius (m)")
    ax.set_ylabel("normalized density")
    ax.set_title("Full-cylinder radial density diagnostic")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_radial_density_full_cylinder.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Recommended sheath diagnostic: sidewall-only density and
    # final/initial ratio.
    # ------------------------------------------------------------------
    final_positions_si = {
        0: bridge.length_code_to_si(PIC.pos[0].get()),
        1: bridge.length_code_to_si(PIC.pos[1].get()),
    }
    initial_positions_si = result.get("initial_positions_si")
    if initial_positions_si is None:
        # Fallback for older result dictionaries.  The ratio plot then becomes
        # equivalent to final/n0 and should not be used for final analysis.
        initial_positions_si = final_positions_si

    final_sw = _sidewall_density_from_positions(final_positions_si, cfg, W, bins=bins)
    initial_sw = _sidewall_density_from_positions(initial_positions_si, cfg, W, bins=bins)
    ratio_e = _safe_ratio(final_sw["electron"], initial_sw["electron"])
    ratio_i = _safe_ratio(final_sw["ion"], initial_sw["ion"])

    _write_density_csv(output_dir / f"{prefix}_radial_density_sidewall.csv", final_sw, initial_sw, cfg)

    # Figure A: final sidewall density normalized by the target bulk density.
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(final_sw["s"], final_sw["electron"] / cfg.n0, label="electron final / n0")
    ax.plot(final_sw["s"], final_sw["ion"] / cfg.n0, label="ion final / n0")
    ax.axvline(lamD, linestyle="--", linewidth=1.0, label=r"$\lambda_D$")
    ax.axvline(2.0 * lamD, linestyle=":", linewidth=1.0, label=r"$2\lambda_D$")
    ax.set_xlabel("distance from antenna sidewall (m)")
    ax.set_ylabel("normalized density")
    ax.set_title("Final sidewall radial density diagnostic")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_radial_density_sidewall.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure B: final density divided by the initial density in the same
    # sidewall-only bins.  This is the preferred plot for sheath interpretation
    # because it removes initialization and finite-bin sampling bias.
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(final_sw["s"], ratio_e, label="electron final / initial")
    ax.plot(final_sw["s"], ratio_i, label="ion final / initial")
    ax.axhline(1.0, linestyle="--", linewidth=1.0, label="no change")
    ax.axvline(lamD, linestyle="--", linewidth=1.0, label=r"$\lambda_D$")
    ax.axvline(2.0 * lamD, linestyle=":", linewidth=1.0, label=r"$2\lambda_D$")
    ax.set_xlabel("distance from antenna sidewall (m)")
    ax.set_ylabel("final / initial density")
    ax.set_title("Sidewall radial density evolution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    # Keep the historical filename pointing to the recommended diagnostic.
    fig.savefig(output_dir / f"{prefix}_radial_density.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{prefix}_radial_density_final_over_initial_sidewall.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Charge separation projection.
    # ------------------------------------------------------------------
    nx = ny = nz = int(cfg.grid_n)
    dx, dy, dz = result.get("grid_spacing_code", result.get("spacing", (cfg.domain_size / cfg.grid_n,) * 3))
    try:
        ne_code = PIC.evaluate_e_nu(dx, dy, dz, nx, ny, nz).get()
        ni_code = PIC.evaluate_i_nu(dx, dy, dz, nx, ny, nz).get()
        ne = bridge.number_density_code_to_si(ne_code)
        ni = bridge.number_density_code_to_si(ni_code)
        charge_norm = (ni - ne) / max(cfg.n0, 1.0e-300)
        j0 = ny // 2
        width = max(1, min(2, ny // 4))
        ys = slice(max(0, j0 - width), min(ny, j0 + width + 1))
        img = np.mean(charge_norm[:, ys, :], axis=1)
        xbins = np.linspace(-0.5 * cfg.domain_size, 0.5 * cfg.domain_size, nx + 1)
        zbins = np.linspace(-0.5 * cfg.domain_size, 0.5 * cfg.domain_size, nz + 1)
    except Exception:
        epos = final_positions_si[0]
        ipos = final_positions_si[1]
        bins2d = 48
        xbins = np.linspace(-0.5 * cfg.domain_size, 0.5 * cfg.domain_size, bins2d + 1)
        zbins = np.linspace(-0.5 * cfg.domain_size, 0.5 * cfg.domain_size, bins2d + 1)
        He, _, _ = np.histogram2d(epos[:, 0], epos[:, 2], bins=[xbins, zbins])
        Hi, _, _ = np.histogram2d(ipos[:, 0], ipos[:, 2], bins=[xbins, zbins])
        area = (xbins[1] - xbins[0]) * (zbins[1] - zbins[0]) * cfg.domain_size
        img = (Hi - He) * W / max(area * cfg.n0, 1.0e-300)

    vmax = float(np.nanpercentile(np.abs(img), 99.0))
    vmax = max(vmax, 1.0e-6)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(
        img.T,
        origin="lower",
        extent=[xbins[0], xbins[-1], zbins[0], zbins[-1]],
        aspect="equal",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    for zc in [0.25 * (cfg.total_length + cfg.feed_gap), -0.25 * (cfg.total_length + cfg.feed_gap)]:
        zhalf = 0.25 * (cfg.total_length - cfg.feed_gap)
        rect = plt.Rectangle(
            (-cfg.radius, zc - zhalf),
            2.0 * cfg.radius,
            2.0 * zhalf,
            fill=False,
            edgecolor="k",
            linewidth=1.0,
        )
        ax.add_patch(rect)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(r"Projected charge separation, $(n_i-n_e)/n_0$")
    fig.colorbar(im, ax=ax, label=r"$(n_i-n_e)/n_0$")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_xz_charge_projection.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def effective_admittance_from_history(history: Dict[str, list], cfg: BaseConfig) -> Dict[str, float]:
    """Estimate a single-frequency effective admittance proxy from total current."""
    t = np.asarray(history["time_s"], dtype=float)
    I = np.asarray(history["I_total_z"], dtype=float)
    if len(t) < 8:
        return {"Y_real": np.nan, "Y_imag": np.nan, "C_eff_F": np.nan, "R_eff_ohm": np.nan}
    # Use the second half of the signal to reduce ramp transient.
    start = len(t) // 2
    t2 = t[start:]
    I2 = I[start:]
    omega = 2.0 * math.pi * cfg.f_drive
    # drive voltage proxy: V = V0 sin(wt)
    V2 = cfg.voltage_amplitude_proxy * np.sin(omega * t2)
    basis = np.exp(-1j * omega * t2)
    Iph = 2.0 * np.mean(I2 * basis)
    Vph = 2.0 * np.mean(V2 * basis)
    Y = Iph / Vph if abs(Vph) > 0 else np.nan + 1j * np.nan
    R = 1.0 / np.real(Y) if np.isfinite(np.real(Y)) and abs(np.real(Y)) > 1e-30 else np.inf
    C = np.imag(Y) / omega if np.isfinite(np.imag(Y)) else np.nan
    return {
        "Y_real": float(np.real(Y)),
        "Y_imag": float(np.imag(Y)),
        "C_eff_F": float(C),
        "R_eff_ohm": float(R) if np.isfinite(R) else float("inf"),
        "I_phasor_abs": float(abs(Iph)),
        "V_phasor_abs": float(abs(Vph)),
    }


def plot_admittance_scan(scan_rows, output_dir: Path, xkey: str, prefix: str = "scan"):
    output_dir.mkdir(parents=True, exist_ok=True)
    xs = np.asarray([row[xkey] for row in scan_rows], dtype=float)
    C = np.asarray([row["C_eff_F"] for row in scan_rows], dtype=float)
    G = np.asarray([row["Y_real"] for row in scan_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(xs, C, marker="o")
    ax.set_xlabel(xkey)
    ax.set_ylabel("effective capacitance proxy (F)")
    ax.set_title("Sheath-modified effective capacitance proxy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_capacitance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(xs, G, marker="o")
    ax.set_xlabel(xkey)
    ax.set_ylabel("effective conductance proxy (S)")
    ax.set_title("Sheath-modified effective conductance proxy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_conductance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_physical_reasonability_report(output_dir: Path, cfg: BaseConfig, case_name: str) -> Dict[str, object]:
    """Write a reviewer-facing physical reasonability report for each case."""
    s = physical_summary(cfg)
    checks = {
        "case_name": case_name,
        "intended_regime": "space/ionospheric electrically-small near-field RF sheath",
        "space_plasma_baseline": True,
        "drive_near_plasma_frequency": bool(0.3 <= cfg.f_drive / s["f_pe_Hz"] <= 3.0),
        "antenna_electrically_small": bool(s["antenna_electrical_size_L_over_lambda"] < 1.0e-2),
        "dt_resolves_drive_period_more_than_100_steps": bool(s["steps_per_drive_period"] > 100.0),
        "dt_resolves_plasma_period_more_than_100_steps": bool((1.0 / s["f_pe_Hz"]) / cfg.dt > 100.0),
        "domain_contains_more_than_10_Debye_lengths": bool(cfg.domain_size / s["lambda_D_m"] > 10.0),
        "summary": s,
        "notes": [
            "These cases target near-field RF sheath and antenna-loading feedback, not far-field wavefront propagation.",
            "The quasi-electrostatic time step is physically motivated by the electrically small antenna: L/lambda_EM << 1.",
            "Grid-source spatial truncation is disabled in the article runs; the reported cases use full grid-source summation at the chosen mesh sizes. MAS antenna point sources are untruncated by default.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "physical_reasonability_report.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    return checks
