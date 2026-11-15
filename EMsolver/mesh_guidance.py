"""Practical mesh/time-step guidance for JefiPIC plasma-antenna runs.

Important distinction
---------------------
JefiPIC's Jefimenko electromagnetic field solver is an integral/retarded-time
solver, not a Yee-FDTD marching solver.  Therefore the usual electromagnetic
CFL condition,

    dt <= dx / (c sqrt(3)),

is not a fundamental field-solver stability condition for the integral solver.

However, a retarded-time integral still has an **accuracy** requirement: the
history time grid must resolve the light-travel-time variation between nearby
source/observer cells.  A useful default diagnostic/constraint is

    dt <= eta * dx_min / c,

where ``eta`` is typically 0.5-1.0.  This is not a blow-up stability condition;
it controls numerical smearing of retarded wavefronts and history
interpolation.

This helper therefore distinguishes:
- ``em_cfl``: FDTD/Yee stability diagnostic, usually not enforced in integral mode.
- ``retarded_time_resolution``: Jefimenko light-cone/history interpolation
  accuracy condition, enforced by default in integral mode.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np


E_CHARGE = 1.602176634e-19
EPSILON0 = 8.8541878128e-12
M_E = 9.1093837015e-31
C0 = 299792458.0
K_B = 1.380649e-23


def _thermal_energy_joule(T_e: float, temperature_unit: str) -> float:
    unit = str(temperature_unit).lower()
    if unit in ("ev", "electronvolt", "electronvolts"):
        return float(T_e) * E_CHARGE
    if unit in ("k", "kelvin"):
        return K_B * float(T_e)
    if unit in ("j", "joule", "joules"):
        return float(T_e)
    raise ValueError("temperature_unit must be 'eV', 'K', or 'J'")


def auto_calculate_mesh(
    f_max: float,
    n_e: float,
    T_e: float,
    *,
    temperature_unit: str = "eV",
    cells_per_wavelength: int = 20,
    cells_per_debye: float = 1.0,
    samples_per_period: int = 10,
    cfl_safety: float = 0.5,
    c: float = C0,
    domain_size: Optional[Sequence[float]] = None,
    truncation_radius: Optional[float] = None,
    solver_kind: str = "integral",
    enforce_em_cfl: Optional[bool] = None,
    enforce_retarded_time_resolution: Optional[bool] = None,
    retarded_time_safety: float = 1.0,
    particle_vmax: Optional[float] = None,
    particle_cell_fraction: float = 0.5,
) -> Dict[str, object]:
    """Return recommended spatial and temporal resolution ranges.

    Parameters
    ----------
    f_max:
        Highest antenna/source frequency that must be resolved, in Hz.
    n_e:
        Electron density in m^-3.
    T_e:
        Electron temperature, default unit eV.
    solver_kind:
        ``"integral"`` for Jefimenko/MAS runs or ``"fdtd"`` for FDTD-like runs.
    enforce_em_cfl:
        Whether the FDTD/Yee EM-CFL diagnostic participates in
        ``dt_recommended_max_s``.  Defaults to False in integral mode and True
        in FDTD mode.
    enforce_retarded_time_resolution:
        Whether ``dt <= retarded_time_safety * dx / c`` participates in
        ``dt_recommended_max_s``.  Defaults to True in integral mode.  This is
        an accuracy condition for retarded-time interpolation, not an FDTD
        stability condition.
    retarded_time_safety:
        Multiplier for ``dx/c``.  Use 1.0 as a minimal light-cell-crossing
        criterion; use 0.5 or smaller for sharper wavefront/history accuracy.
    particle_vmax:
        Optional maximum particle speed in m/s.  If supplied, a particle
        cell-crossing recommendation is added.
    particle_cell_fraction:
        Fraction of one cell a particle may cross in one time step, used only
        when ``particle_vmax`` is supplied.
    """
    f_max = float(f_max)
    n_e = float(n_e)
    T_e = float(T_e)
    if f_max <= 0.0:
        raise ValueError("f_max must be positive")
    if n_e <= 0.0:
        raise ValueError("n_e must be positive")
    if T_e <= 0.0:
        raise ValueError("T_e must be positive")

    solver_kind = str(solver_kind).lower()
    if solver_kind not in ("integral", "jefimenko", "mas", "fdtd", "yee"):
        raise ValueError("solver_kind must be 'integral'/'jefimenko'/'mas' or 'fdtd'/'yee'")
    integral_mode = solver_kind in ("integral", "jefimenko", "mas")

    if enforce_em_cfl is None:
        enforce_em_cfl = not integral_mode
    if enforce_retarded_time_resolution is None:
        enforce_retarded_time_resolution = integral_mode

    cells_per_wavelength = max(1, int(cells_per_wavelength))
    cells_per_debye = max(float(cells_per_debye), 1.0e-12)
    samples_per_period = max(1, int(samples_per_period))
    cfl_safety = max(1.0e-12, min(float(cfl_safety), 1.0))
    retarded_time_safety = max(1.0e-12, float(retarded_time_safety))
    particle_cell_fraction = max(1.0e-12, float(particle_cell_fraction))

    thermal_energy = _thermal_energy_joule(T_e, temperature_unit)
    omega_pe = math.sqrt(n_e * E_CHARGE * E_CHARGE / (EPSILON0 * M_E))
    f_pe = omega_pe / (2.0 * math.pi)
    lambda_D = math.sqrt(EPSILON0 * thermal_energy / (n_e * E_CHARGE * E_CHARGE))
    lambda_em = c / f_max

    dx_em_max = lambda_em / cells_per_wavelength
    dx_debye_max = lambda_D / cells_per_debye
    dx_recommended = min(dx_em_max, dx_debye_max)

    dt_source_max = 1.0 / (samples_per_period * f_max)
    dt_plasma_max = 1.0 / (samples_per_period * f_pe)

    # FDTD/Yee stability diagnostic.
    dt_em_cfl_diagnostic = cfl_safety * dx_recommended / (c * math.sqrt(3.0))

    # Retarded-time integral interpolation/light-cone resolution criterion.
    # For nonuniform grids, replace dx_recommended by min(dx,dy,dz) in the
    # caller or pass the limiting spacing through domain-specific logic.
    dt_retarded_max = retarded_time_safety * dx_recommended / c

    dt_candidates = {
        "source_frequency": dt_source_max,
        "electron_plasma_frequency": dt_plasma_max,
    }
    if enforce_retarded_time_resolution:
        dt_candidates["retarded_time_resolution"] = dt_retarded_max
    if enforce_em_cfl:
        dt_candidates["em_cfl"] = dt_em_cfl_diagnostic

    dt_particle_max = None
    if particle_vmax is not None and float(particle_vmax) > 0.0:
        dt_particle_max = particle_cell_fraction * dx_recommended / float(particle_vmax)
        dt_candidates["particle_cell_crossing"] = dt_particle_max

    limiting_name = min(dt_candidates, key=dt_candidates.get)
    dt_recommended = dt_candidates[limiting_name]

    history = None
    if truncation_radius is not None and float(truncation_radius) > 0.0:
        required_distance = float(truncation_radius)
    elif domain_size is not None:
        ds = np.asarray(domain_size, dtype=np.float64).reshape(3,)
        required_distance = float(np.linalg.norm(ds))
    else:
        required_distance = None
    if required_distance is not None:
        history = {
            "required_distance": required_distance,
            "recommended_len_time_snapshots": int(math.ceil(required_distance / max(c * dt_recommended, 1.0e-300))) + 1,
        }

    warnings = []
    if dx_recommended == dx_debye_max:
        warnings.append("Debye-length resolution is the limiting spatial constraint.")
    else:
        warnings.append("EM wavelength resolution is the limiting spatial constraint.")

    if integral_mode and not enforce_em_cfl:
        warnings.append("Integral Jefimenko/MAS mode: FDTD EM-CFL is reported as a diagnostic, not enforced as a stability limit.")
    elif enforce_em_cfl:
        warnings.append("FDTD/Yee EM-CFL is enforced in dt_recommended_max_s.")

    if enforce_retarded_time_resolution:
        warnings.append("Retarded-time light-cone resolution dt <= eta*dx/c is enforced as an accuracy condition.")
    else:
        warnings.append("Retarded-time light-cone resolution is reported but not enforced.")

    if limiting_name == "source_frequency":
        warnings.append("Highest imposed/source frequency is the limiting enforced time-step constraint.")
    elif limiting_name == "electron_plasma_frequency":
        warnings.append("Electron plasma-frequency resolution is the limiting enforced time-step constraint.")
    elif limiting_name == "retarded_time_resolution":
        warnings.append("Retarded-time interpolation/light-cone resolution is the limiting enforced time-step constraint.")
    elif limiting_name == "em_cfl":
        warnings.append("3-D FDTD/Yee electromagnetic CFL is the limiting enforced time-step constraint.")
    elif limiting_name == "particle_cell_crossing":
        warnings.append("Particle cell-crossing distance is the limiting enforced time-step constraint.")

    return {
        "inputs": {
            "f_max_Hz": f_max,
            "n_e_m^-3": n_e,
            "T_e": T_e,
            "temperature_unit": temperature_unit,
            "solver_kind": solver_kind,
            "enforce_em_cfl": bool(enforce_em_cfl),
            "enforce_retarded_time_resolution": bool(enforce_retarded_time_resolution),
            "retarded_time_safety": retarded_time_safety,
        },
        "derived": {
            "lambda_em_m": lambda_em,
            "omega_pe_rad_s": omega_pe,
            "f_pe_Hz": f_pe,
            "lambda_D_m": lambda_D,
        },
        "dx_constraints": {
            "dx_em_max_m": dx_em_max,
            "dx_debye_max_m": dx_debye_max,
            "dx_recommended_max_m": dx_recommended,
        },
        "dt_constraints": {
            "dt_source_max_s": dt_source_max,
            "dt_plasma_max_s": dt_plasma_max,
            "dt_retarded_max_s": dt_retarded_max,
            "dt_em_cfl_diagnostic_s": dt_em_cfl_diagnostic,
            "dt_cfl_max_s": dt_em_cfl_diagnostic,  # backward-compatible alias
            "dt_particle_max_s": dt_particle_max,
            "dt_recommended_max_s": dt_recommended,
            "limiting_constraint": limiting_name,
            "enforced_constraints": dt_candidates,
        },
        "history": history,
        "warnings": warnings,
    }


def print_mesh_recommendation(info: Dict[str, object]) -> None:
    """Pretty-print the dictionary returned by ``auto_calculate_mesh``."""
    d = info["derived"]
    dx = info["dx_constraints"]
    dt = info["dt_constraints"]
    inputs = info.get("inputs", {})
    print("Recommended JefiPIC mesh/time-step envelope")
    print(f"  EM solver mode:       {inputs.get('solver_kind', 'unknown')}")
    print(f"  FDTD CFL enforced:    {inputs.get('enforce_em_cfl', 'unknown')}")
    print(f"  Retarded dt enforced: {inputs.get('enforce_retarded_time_resolution', 'unknown')}")
    print(f"  EM wavelength:        {d['lambda_em_m']:.6e} m")
    print(f"  Debye length:         {d['lambda_D_m']:.6e} m")
    print(f"  Plasma frequency:     {d['f_pe_Hz']:.6e} Hz")
    print(f"  dx <= min({dx['dx_em_max_m']:.6e}, {dx['dx_debye_max_m']:.6e}) = {dx['dx_recommended_max_m']:.6e} m")

    enforced = dt.get("enforced_constraints", {})
    enforced_text = ", ".join(f"{name}={value:.6e}" for name, value in enforced.items())
    print(f"  enforced dt limits:   {enforced_text}")
    print(f"  dt recommended <=     {dt['dt_recommended_max_s']:.6e} s")
    print(f"  retarded dt limit:    {dt['dt_retarded_max_s']:.6e} s")
    print(f"  FDTD-CFL diagnostic:  {dt['dt_em_cfl_diagnostic_s']:.6e} s")

    if info.get("history"):
        h = info["history"]
        print(f"  len_time_snapshots >= {h['recommended_len_time_snapshots']} for {h['required_distance']:.6e} m retarded reach")
    for msg in info.get("warnings", []):
        print(f"  note: {msg}")
