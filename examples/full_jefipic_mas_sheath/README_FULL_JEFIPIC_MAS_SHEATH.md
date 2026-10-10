# Full JefiPIC + MAS antenna sheath example

This example is a real JefiPIC loop, not the earlier reduced CPU model.

It uses the same core interfaces as `JefiPIC.ipynb`:

```python
PIC.proceed_one_step(...)
PIC.evaluate_rho_J(...)
PIC_provider.Jefimenko_solver(...)
PIC_provider.acquire_EB_field()
```

and inserts the MAS antenna feedback in the field update:

```python
PIC_provider.Jefimenko_solver(..., include_antenna=False)
(Ex, Ey, Ez), (Bx, By, Bz) = mas_device.update_fields_with_device_interaction(
    em_solver=PIC_provider,
    return_gpu_fields=True,
)
```

## Quick run

From repository root:

```bash
python MAS_Plasma_Antenna_Example.py --quick
```

Or from this directory:

```bash
python Full_JefiPIC_MAS_Sheath.py --quick
```

## Larger run

```bash
python Full_JefiPIC_MAS_Sheath.py \
  --steps 1200 \
  --particles 8000 \
  --grid-n 17 \
  --dt 1e-9 \
  --f-drive 3e6
```

## Why the default `dt` is not `dx/c`

For this ionospheric sheath example, the default is a quasi-electrostatic
near-field run.  It deliberately does **not** enforce the retarded-time
resolution condition `dt <= dx/c`.

For example, with `dx ~ 1 cm`, `dx/c ~ 30 ps`.  A 3 MHz RF period is about
333 ns, so enforcing `dx/c` would require roughly 10,000 steps per RF cycle
for this grid spacing.  That is not practical for sheath evolution studies.

Instead, the default `dt = 1 ns` resolves:

- the 3 MHz RF period with about 333 steps per cycle;
- the electron plasma period for `n0 = 1e11 m^-3` with hundreds of steps;
- the particle pusher dynamics at this demonstration amplitude.

This is appropriate when the paper question is near-field sheath evolution,
RF capacitance/resistance trends, and plasma response around a small antenna,
not far-field wavefront propagation.

## Spatial truncation

The EM solver is configured as:

```python
lambda_D = sqrt(eps0 * T_eV / (n0 * e))
PIC_provider.set_spatial_truncation_radius(
    grid_radius = 8.0 * lambda_D,
    point_radius = None,
    taper = "hann",
    taper_width = 2.0 * lambda_D,
)
```

So grid-source contributions are tapered smoothly in the outer shell, while
MAS antenna point-source contributions remain untruncated by default.

## Outputs

The script writes to:

```text
examples/full_jefipic_mas_sheath/output/
```

Key files:

```text
final_radial_density.png
final_xz_charge_projection.png
final_E_magnitude_xz.png
mas_boundary_diagnostic.png
antenna_particle_hits.png
history.json
history_capacity.json
time_step_and_scale_summary.json
runtime_mas_geometry.npz
final_particle_state.npz
```

## Notes

A simple cylindrical antenna particle absorber/reinjection step is included
after the JefiPIC particle push.  JefiPIC's built-in metal-particle interaction
currently supports a cube device, while this example uses a cylindrical STL
dipole; the custom absorber keeps particles from accumulating inside the metal
without replacing the JefiPIC field solve, deposition, or particle pusher.
