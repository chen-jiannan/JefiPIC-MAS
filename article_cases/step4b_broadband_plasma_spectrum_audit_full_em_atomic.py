#!/usr/bin/env python3
"""
Step 4B — broadband plasma-spectrum and electrical-delay characterization.

Why this test exists
--------------------
Step 4A is not an acceptable final coupling model for the paper because the
plasma field is generally broadband and the PIC/MAS system is nonlinear.
Projecting E_plasma(t) to a single 3 MHz phasor discards harmonics, transients,
noise-driven content and nonlinear source feedback.

Step 4B therefore makes NO single-frequency approximation to the plasma field.

A validated 3 MHz vacuum harmonic antenna solution is used only as a fixed,
external probe source.  PIC/Jefimenko evolves in time and the plasma-only
electric field is sampled at representative antenna-surface probes every time
step.  The full time series is saved and Fourier analysed.

This is a one-way bandwidth audit, not the final two-way PIC<->MAS solver.

Questions
---------
1. How much plasma-field spectral energy lies outside the 3 MHz fundamental?
2. What are the 95% and 99% cumulative-energy bandwidths?
3. What electrical-delay scale, fD/c, corresponds to the measured bandwidth?
4. Are strong harmonics / broadband components present, requiring the causal
   MAS formulation to operate directly in the time domain?

No manuscript RF amplification claim is accepted or rejected by Step 4B.

Probe source
------------
The external probe is a 3 MHz localized feed-field vacuum solution on the
validated 60 mm rounded dipole:
- 900 sources/arm,
- d/R = 0.80,
- 2x independent collocation,
- lambda_rel = 1e-4.

The vacuum harmonic collocation residual must be <5% before PIC starts.
This source is a diagnostic external driver only; it is not a claim that the
plasma response is harmonic.

PIC spectrum-audit settings
---------------------------
Default:
- dt = 1 ns
- 4 RF cycles
- 3000 particles/species
- 13^3 grid
- 96 deterministic surface probes
- 1 V/m localized-feed proxy amplitude
- first 1 RF cycle discarded before spectrum estimation

For manuscript-facing use, run with ``--full-em-plasma`` and the Atomic
native-staggered-current backend.

Interpretation
--------------
If substantial energy exists above the fundamental, the causal MAS solver must
consume E_plasma(t) every time step with no 3 MHz projection.

The dimensionless quantities f95*D/c and f99*D/c are reported only as
electrical-delay descriptors of the measured bandwidth.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

EPS0 = 8.8541878128e-12
C0 = 299792458.0

F_DRIVE = 3.0e6
DT = 1.0e-9
FORMAL_SOURCES_PER_ARM = 900
DEPTH_FRAC = 0.80
COLLOCATION_FACTOR = 2.0
LAMBDA_REL = 1.0e-4
FEED_FIELD_AMPLITUDE = 1.0

DEFAULT_CYCLES = 4.0
DEFAULT_DISCARD_CYCLES = 1.0
DEFAULT_PARTICLES = 3000
DEFAULT_GRID_N = 13
DEFAULT_PROBES = 96

def smooth_ramp(t_s, frequency, cycles=1.0):
    duration = float(cycles) / float(frequency)
    if duration <= 0.0:
        return 1.0
    x = min(max(float(t_s) / duration, 0.0), 1.0)
    return float(x*x*(3.0 - 2.0*x))


def _build_validated_geometry_mas(cfg, stl_path, grid_shape, spacing, origin,
                                  *, current_amplitude=None):
    """Build validated geometry for source registration only.

    No old recursive MAS current solve is called in Step 4B.
    """
    from EMsolver.mas_integration import (
        SinusoidalCurrentExcitation,
        create_mas_antenna_simulation,
    )

    gp = {
        "kind": "rounded_two_arm_dipole",
        "radius": float(cfg.radius),
        "total_length": float(cfg.total_length),
        "feed_gap": float(cfg.feed_gap),
        "depth_frac": DEPTH_FRAC,
        "sources_per_arm": FORMAL_SOURCES_PER_ARM,
        "collocation_factor": COLLOCATION_FACTOR,
        "n_phi": 72,
        "n_cyl": 96,
        "n_hemi": 24,
        "spatial_solver": "tangential_weighted_scaled",
        "tikhonov_rel": LAMBDA_REL,
        "svd_rcond": 1.0e-10,
    }
    active = SinusoidalCurrentExcitation(
        amplitude=0.0,
        frequency=F_DRIVE,
        direction=(0.0, 0.0, 1.0),
        ramp_time=0.0,
    )
    sp = {
        "dt": float(cfg.dt),
        "epsilon0": EPS0,
        "c": C0,
        "max_time_steps": int(cfg.len_time_snapshots),
        "grid_size": tuple(grid_shape),
        "grid_spacing": tuple(spacing),
        "grid_origin": tuple(origin),
        "active_excitation": active,
        "source_softening": 0.0,
        "source_truncation_radius": None,
        "surface_charge_ridge": 0.0,
    }
    return create_mas_antenna_simulation(gp, sp)


def _make_config(out_name, *, cycles, particles, grid_n, full_em):
    from _jefipic_mas_case_common import BaseConfig

    cfg = BaseConfig(output_dir=out_name)
    cfg.dt = DT
    cfg.f_drive = F_DRIVE
    cfg.current_amplitude = 0.0
    cfg.ramp_periods = 0.0

    cfg.steps = int(np.ceil(float(cycles)/(F_DRIVE*DT))) + 1
    cfg.len_time_snapshots = max(256, min(4096, cfg.steps + 8))
    cfg.diag_stride = max(1, cfg.steps // 40)

    cfg.particles_per_species = int(particles)
    cfg.grid_n = int(grid_n)

    cfg.jefipic_unit_mode = "LHQCD"
    # GPU accumulation strategy and physical field model are separate choices.
    # Full-EM and quasi-electrostatic field content are physical-model choices;
    # Atomic is the reported grid-source reduction backend.
    cfg.jefipic_kernel_mode = "atomic"
    cfg.quasi_electrostatic = not bool(full_em)

    cfg.spatial_truncation_debye = 0.0
    cfg.taper_width_debye = 0.0
    cfg.taper = "none"
    cfg.point_radius = None

    cfg.quiet_start_pairs = True
    cfg.center_initial_velocities = True
    cfg.velocity_quiet_start_pairs = True
    cfg.particle_reinjection_mode = "boundary"
    cfg.initial_antenna_exclusion_debye = 0.5
    cfg.boundary_reinjection_antenna_exclusion_debye = 0.25
    cfg.plasma_to_mas_ramp_steps = 0

    cfg.particle_order = 1
    cfg.field_order = 1
    return cfg


def _build_vacuum_probe_source(g):
    """Return the validated 3 MHz vacuum feed-field source phasors."""
    import step2mb1_incident_normalization_feed_localization as mb1
    import step3c2a_d3b_harmonic_feed_field_candidate as harm

    omega = 2.0*np.pi*F_DRIVE
    A = harm.assemble_harmonic_tangent_operator(g, omega)
    prepared = harm.prepare_solver(A, g.solve_weights, LAMBDA_REL)

    feed_hat = (
        FEED_FIELD_AMPLITUDE
        * mb1.inc_feed_localized(g.solve_obs_points, 2.0e-3, 1.0)
    ).astype(np.complex128)

    b = harm.tangent_rhs(feed_hat, g.solve_normals)
    x, diag = harm.solve_prepared(A, prepared, b)
    Khat = harm.coeff_to_current(g, x)
    Phat = Khat/(1j*omega)

    # Release the large dense harmonic matrix before the long PIC run.
    del A, prepared
    return feed_hat, Khat, Phat, diag


def _select_probe_indices(npoints, nprobes):
    nprobes = min(max(8, int(nprobes)), int(npoints))
    idx = np.linspace(0, npoints-1, nprobes, dtype=np.int64)
    return np.unique(idx)


def _spectrum_from_timeseries(t, E, discard_before_s):
    t = np.asarray(t, dtype=float)
    E = np.asarray(E, dtype=float)
    use = np.isfinite(t) & (t >= float(discard_before_s))
    t = t[use]
    E = E[use]

    if len(t) < 64:
        raise RuntimeError("Too few post-ramp samples for broadband spectrum audit")

    # Shape: Nt x Nprobe x 3.  Remove each probe/component mean.
    X = E - np.mean(E, axis=0, keepdims=True)

    window = np.hanning(len(t))
    Xw = X * window[:, None, None]

    F = np.fft.rfft(Xw, axis=0)
    freq = np.fft.rfftfreq(len(t), d=float(np.median(np.diff(t))))

    # Aggregate spectral energy over all probes/components.
    power = np.sum(np.abs(F)**2, axis=(1, 2))
    if len(power):
        power[0] = 0.0

    total = float(np.sum(power))
    if total <= 0.0 or not np.isfinite(total):
        raise RuntimeError("Spectrum audit produced zero/non-finite AC spectral energy")

    frac = power / total
    cumulative = np.cumsum(frac)

    def percentile_frequency(p):
        i = int(np.searchsorted(cumulative, p, side="left"))
        i = min(max(i, 0), len(freq)-1)
        return float(freq[i])

    df = float(freq[1]-freq[0]) if len(freq) > 1 else float("nan")

    # Fundamental band = nearest bin to 3 MHz +/- one spectral bin.
    fund = np.abs(freq - F_DRIVE) <= max(df, 0.5e6)
    fundamental_fraction = float(np.sum(frac[fund]))

    def band_fraction(lo, hi=None):
        m = freq >= float(lo)
        if hi is not None:
            m &= freq < float(hi)
        return float(np.sum(frac[m]))

    # Harmonic-neighbour energy fractions for descriptive nonlinear content.
    harmonic = {}
    for h in range(1, 7):
        fh = h*F_DRIVE
        m = np.abs(freq-fh) <= max(df, 0.5e6)
        harmonic[str(h)] = {
            "frequency_Hz": float(fh),
            "energy_fraction": float(np.sum(frac[m])),
        }

    # Strongest non-DC bins.
    order = np.argsort(power)[::-1]
    top = []
    for i in order[:12]:
        top.append({
            "frequency_Hz": float(freq[i]),
            "energy_fraction": float(frac[i]),
        })

    return {
        "sample_count": int(len(t)),
        "duration_s": float(t[-1]-t[0]),
        "frequency_resolution_Hz": df,
        "nyquist_Hz": float(freq[-1]),
        "fundamental_3MHz_energy_fraction": fundamental_fraction,
        "outside_fundamental_energy_fraction": float(1.0-fundamental_fraction),
        "energy_fraction_below_10MHz": band_fraction(0.0, 10.0e6),
        "energy_fraction_10_to_30MHz": band_fraction(10.0e6, 30.0e6),
        "energy_fraction_30_to_100MHz": band_fraction(30.0e6, 100.0e6),
        "energy_fraction_above_100MHz": band_fraction(100.0e6, None),
        "f95_Hz": percentile_frequency(0.95),
        "f99_Hz": percentile_frequency(0.99),
        "harmonic_energy_fractions": harmonic,
        "top_spectral_bins": top,
        "frequency_Hz": freq,
        "power_fraction": frac,
    }


def _synthetic_spectrum_self_test():
    dt = 1.0e-9
    t = np.arange(0.0, 4.0e-6, dt)
    # Known multi-tone signal: 3, 9 and 30 MHz.
    x = (
        1.0*np.cos(2*np.pi*3e6*t)
        + 0.45*np.cos(2*np.pi*9e6*t + 0.2)
        + 0.20*np.cos(2*np.pi*30e6*t - 0.3)
    )
    E = np.zeros((len(t), 4, 3), dtype=float)
    E[:, :, 0] = x[:, None]

    s = _spectrum_from_timeseries(t, E, 0.0)
    peaks = [r["frequency_Hz"] for r in s["top_spectral_bins"][:12]]
    hf = s["harmonic_energy_fractions"]

    # Robust validation: check that the known 3 and 9 MHz tones carry
    # non-negligible energy, and verify a strong bin exists near 30 MHz.
    near_30 = any(abs(p-30.0e6) < 0.6e6 for p in peaks)
    ok = (
        hf["1"]["energy_fraction"] > 0.50
        and hf["3"]["energy_fraction"] > 0.05
        and near_30
    )
    report = {
        "self_test_pass": bool(ok),
        "top_peak_frequencies_Hz": peaks,
        "fundamental_energy_fraction": hf["1"]["energy_fraction"],
        "third_harmonic_energy_fraction": hf["3"]["energy_fraction"],
        "near_30MHz_peak_found": bool(near_30),
        "f95_Hz": s["f95_Hz"],
        "f99_Hz": s["f99_Hz"],
    }
    print(json.dumps(report, indent=2))
    if not ok:
        raise RuntimeError("Step 4B synthetic spectrum self-test failed")


def run_gpu(args):
    import _jefipic_mas_case_common as common

    out = Path("article_cases_output") / args.out
    out.mkdir(parents=True, exist_ok=True)

    cfg = _make_config(
        args.out,
        cycles=args.cycles,
        particles=args.particles,
        grid_n=args.grid_n,
        full_em=args.full_em_plasma,
    )

    common.build_mas_device = _build_validated_geometry_mas

    bundle = common.build_pic_solver_mas(
        cfg, out/"IGNORED_VALIDATED_PRODUCTION_GEOMETRY.stl"
    )
    PIC = bundle["PIC"]
    solver = bundle["solver"]
    mas = bundle["mas"]
    rng = bundle["rng"]
    bridge = bundle["unit_bridge"]
    nx, ny, nz = bundle["shape"]
    dx, dy, dz = bundle["spacing"]
    Ex, Ey, Ez, Bx, By, Bz = bundle["fields"]

    g = mas.geometry
    if int(g.metadata.get("sources_per_arm", -1)) != FORMAL_SOURCES_PER_ARM:
        raise RuntimeError("Step 4B must use the formal 900-source/arm geometry")

    print("[Step 4B] building validated 3 MHz vacuum probe source ...", flush=True)
    feed_hat, Khat, Phat, vac_diag = _build_vacuum_probe_source(g)
    if vac_diag["collocation_rel_l2"] >= 0.05:
        raise RuntimeError(
            "Validated external probe source failed the 5% collocation gate: "
            f"{vac_diag['collocation_rel_l2']}"
        )

    common.register_mas_with_em_solver_unit_bridge(mas, solver, bridge)

    probe_idx = _select_probe_indices(g.num_collocation_points, args.probes)
    probe_points_si = np.asarray(g.solve_obs_points, dtype=float)[probe_idx]
    probe_points_code = bridge.length_si_to_code(probe_points_si)

    omega = 2*np.pi*F_DRIVE

    times = []
    probe_E = []
    probe_rms = []
    finite = True
    steps_completed = 0

    print(
        f"[Step 4B] broadband audit: steps={cfg.steps}, "
        f"particles/species={cfg.particles_per_species}, grid={cfg.grid_n}^3, "
        f"probes={len(probe_idx)}, full_em_plasma={args.full_em_plasma}",
        flush=True,
    )

    for step in range(cfg.steps):
        PIC.proceed_one_step(
            Ex, Ey, Ez, Bx, By, Bz,
            dx, dy, dz, nx, ny, nz, cfg.field_order
        )
        common.absorb_particles_on_antenna(PIC, cfg, rng)

        rho, Jx_face, Jy_face, Jz_face = PIC.evaluate_rho_J(
            dx, dy, dz, nx, ny, nz, cfg.particle_order
        )
        solver.Jefimenko_solver(
            rho.flatten(),
            include_antenna=False,
            face_currents=(Jx_face, Jy_face, Jz_face),
        )

        plasma_code = solver.sample_current_E_at_points(probe_points_code)
        plasma_si = bridge.efield_code_to_si(plasma_code)
        plasma_si = np.asarray(plasma_si, dtype=float).reshape(len(probe_idx), 3)

        if not np.all(np.isfinite(plasma_si)):
            finite = False
            break

        t_s = float(step*DT)
        times.append(t_s)
        probe_E.append(plasma_si.copy())
        probe_rms.append(
            float(np.sqrt(np.mean(np.sum(plasma_si*plasma_si, axis=1))))
        )

        # Stable prescribed external antenna source for the next PIC step.
        ramp = smooth_ramp(t_s, F_DRIVE, cycles=1.0)
        phase = np.exp(1j*omega*t_s)
        K_now = ramp*np.real(Khat*phase)
        P_now = ramp*np.real(Phat*phase)

        solver.recompute_current_fields(
            antenna_surface_currents=
                bridge.surface_current_density_si_to_code(K_now),
            antenna_polarization=
                bridge.surface_polarization_density_si_to_code(P_now),
            include_antenna=True,
        )
        Ex, Ey, Ez, Bx, By, Bz = solver.acquire_EB_field()

        steps_completed = step+1
        if step % max(1, cfg.diag_stride) == 0 or step == cfg.steps-1:
            print(
                f"[Step 4B] step {step+1}/{cfg.steps}, "
                f"plasma probe E_rms={probe_rms[-1]:.6e} V/m",
                flush=True,
            )

    times = np.asarray(times, dtype=float)
    probe_E = np.asarray(probe_E, dtype=float)
    probe_rms = np.asarray(probe_rms, dtype=float)

    if steps_completed != cfg.steps or not finite:
        spectrum = None
        accepted_execution = False
    else:
        spectrum = _spectrum_from_timeseries(
            times,
            probe_E,
            discard_before_s=float(args.discard_cycles)/F_DRIVE,
        )
        accepted_execution = True

    if spectrum is not None:
        D = float(cfg.total_length)
        f95 = float(spectrum["f95_Hz"])
        f99 = float(spectrum["f99_Hz"])
        delay95 = f95*D/C0
        delay99 = f99*D/C0
        spectrum_serializable = {
            k: v for k, v in spectrum.items()
            if k not in ("frequency_Hz", "power_fraction")
        }
    else:
        D = float(cfg.total_length)
        delay95 = delay99 = None
        spectrum_serializable = None

    # Save the raw broadband signal and spectrum arrays.
    save_dict = {
        "time_s": times,
        "probe_indices": probe_idx,
        "probe_points_m": probe_points_si,
        "plasma_E_probe_V_per_m": probe_E,
        "plasma_probe_E_rms_V_per_m": probe_rms,
    }
    if spectrum is not None:
        save_dict["frequency_Hz"] = spectrum["frequency_Hz"]
        save_dict["power_fraction"] = spectrum["power_fraction"]
    np.savez_compressed(out/"step4b_broadband_timeseries_and_spectrum.npz", **save_dict)

    if spectrum is None:
        spectrum_classification = "execution_failed"
    elif spectrum["outside_fundamental_energy_fraction"] > 0.10:
        spectrum_classification = "broadband_time_domain_response"
    else:
        spectrum_classification = "fundamental_dominated_time_domain_response"

    report = {
        "step": "4B",
        "purpose": "broadband plasma-field spectrum and electrical-delay characterization",
        "accepted_execution": accepted_execution,
        "this_is_not_a_two_way_self_consistent_coupling_test": True,
        "plasma_field_was_not_projected_to_3MHz_for_dynamics": True,
        "configuration": {
            "frequency_Hz": F_DRIVE,
            "dt_s": DT,
            "cycles": float(args.cycles),
            "discard_cycles": float(args.discard_cycles),
            "particles_per_species": int(args.particles),
            "grid_n": int(args.grid_n),
            "num_surface_probes": int(len(probe_idx)),
            "formal_sources_per_arm": FORMAL_SOURCES_PER_ARM,
            "formal_collocation_points_per_arm": 2*FORMAL_SOURCES_PER_ARM,
            "feed_field_proxy_amplitude_V_per_m": FEED_FIELD_AMPLITUDE,
            "quasi_electrostatic_plasma": not bool(args.full_em_plasma),
            "jefipic_kernel_mode_requested": "atomic",
            "jefipic_kernel_mode_actual": str(getattr(solver, "kernel_mode", "unknown")),
            "full_em_native_staggered_current_enabled": bool(
                args.full_em_plasma and str(getattr(solver, "kernel_mode", "unknown")) == "atomic"
            ),
            "grid_source_backend_policy": "atomic-production; observation-reference-only",
        },
        "external_probe_source": {
            "vacuum_harmonic_collocation_rel_l2":
                float(vac_diag["collocation_rel_l2"]),
            "five_percent_gate_pass":
                bool(vac_diag["collocation_rel_l2"] < 0.05),
            "note":
                "The external source is harmonic; the measured plasma response is not forced to be harmonic.",
        },
        "execution": {
            "steps_requested": int(cfg.steps),
            "steps_completed": int(steps_completed),
            "all_probe_fields_finite": bool(finite),
        },
        "spectrum": spectrum_serializable,
        "electrical_delay_characterization": {
            "antenna_dimension_m": D,
            "one_way_dimension_delay_s": float(D / C0),
            "f95_times_D_over_c": delay95,
            "f99_times_D_over_c": delay99,
            "classification": spectrum_classification,
            "note": (
                "Descriptive electrical-size quantities only; they are not used "
                "to define or justify a time-local field approximation."
            ),
        },
        "decision": (
            "STEP4B_PASS_broadband_spectrum_characterized_for_causal_MAS"
            if accepted_execution else
            "STEP4B_FAIL_stop_and_diagnose_PIC_or_probe_source"
        ),
        "scope_note": (
            "Step 4B quantifies bandwidth only. It does not validate the old "
            "154x/289x/17x values and does not establish two-way nonlinear "
            "plasma-antenna self-consistency."
        ),
    }
    (out/"step4b_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    lines = [
        "Step 4B broadband plasma-spectrum audit",
        f"accepted_execution: {accepted_execution}",
        f"steps: {steps_completed}/{cfg.steps}",
        f"vacuum probe-source collocation residual: {vac_diag['collocation_rel_l2']:.6g}",
    ]
    if spectrum is not None:
        lines += [
            f"fundamental 3MHz energy fraction: {spectrum['fundamental_3MHz_energy_fraction']:.6g}",
            f"outside-fundamental energy fraction: {spectrum['outside_fundamental_energy_fraction']:.6g}",
            f"f95: {spectrum['f95_Hz']/1e6:.6g} MHz",
            f"f99: {spectrum['f99_Hz']/1e6:.6g} MHz",
            f"delay fraction at f95: {delay95:.6g}",
            f"delay fraction at f99: {delay99:.6g}",
            f"spectrum classification: {spectrum_classification}",
        ]
    lines.append(f"decision: {report['decision']}")
    (out/"step4b_summary.txt").write_text(
        "\n".join(lines)+"\n", encoding="utf-8"
    )
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--out", default="step4b_broadband_plasma_spectrum")
    p.add_argument("--cycles", type=float, default=DEFAULT_CYCLES)
    p.add_argument("--discard-cycles", type=float, default=DEFAULT_DISCARD_CYCLES)
    p.add_argument("--particles", type=int, default=DEFAULT_PARTICLES)
    p.add_argument("--grid-n", type=int, default=DEFAULT_GRID_N)
    p.add_argument("--probes", type=int, default=DEFAULT_PROBES)
    p.add_argument("--full-em-plasma", action="store_true")
    a = p.parse_args()

    if a.self_test:
        _synthetic_spectrum_self_test()
    else:
        run_gpu(a)


if __name__ == "__main__":
    main()
