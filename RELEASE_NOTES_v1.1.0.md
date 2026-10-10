# JefiPIC-MAS v1.1.0

This release updates the public reproducibility package to the final
full-electromagnetic JefiPIC-MAS validation chain associated with the manuscript
“A causal Jefimenko-PIC/MAS method for grid-decoupled metallic boundaries in
kinetic plasma simulations”.

## Main changes

- Native staggered-face full-EM current histories now have a pair-parallel
  atomic-add grid-source evaluator.
- Coincident source/observation self-cell identification uses integer grid
  indices instead of floating-coordinate equality.
- Full-EM Atomic and observation-parallel implementations agree to
  floating-point precision in the dedicated parity regression.
- First-order charge-conserving CIC deposition is unchanged.
- Added final full-EM broadband, 15 MHz temporal, controlled-overfit,
  full-EM closure, and 12-cycle article-scale reproduction scripts.
- Updated reference data to the final full-EM evidence chain.
- Retained quasi-electrostatic 3/6/9 MHz spatial and particle/seed datasets are
  explicitly labeled as auxiliary evidence rather than full-EM results.

## Manuscript-facing final values

- bin-resolved broadband: f95 ≈ 9.0 MHz, f99 ≈ 15.0 MHz;
- full-EM Step4D-A: 3.628% global / 4.393% worst-local holdout;
- 12-cycle full-EM Step4D-B: 4.431% global / 4.566% worst-local holdout;
- 15 MHz Retarded-MOT characterization: 0.590% worst holdout and
  0.0686% amplitude error relative to the continuous retarded harmonic reference.

The Mz auxiliary current moment remains a sensitivity diagnostic and is not
interpreted as a calibrated gain or terminal current.
