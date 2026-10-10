# Package validation

Release: **JefiPIC-MAS v1.1.0**

## Static validation

- Python source files were batch syntax-compiled after the final source merge.
- Syntax errors: **0**.
- A pre-existing documentation-string `SyntaxWarning` in
  `RBG_Maxwell/Unit_conversion/main.py` does not affect syntax or runtime.

## Unit / smoke tests

`pytest -q`

- **3 passed**
- **0 failed**

The tests cover the weak MAS continuity identity, a small conducting-sphere
case, and the Retarded-MOT self-test.

## Pure-CPU reproduction wrapper

`python scripts/run_cpu_repro.py`: **PASS**

Observed deterministic checks include:

- continuity-consistent electric-dipole audit: overall PASS;
- conducting sphere, 48 points, source offset d/R=0.35:
  - tangential residual / incident RMS ~= 2.756e-2
  - induced-dipole relative error ~= 6.927e-3
- conducting sphere, 96 points, source offset d/R=0.35:
  - tangential residual / incident RMS ~= 5.400e-3
  - induced-dipole relative error ~= 5.963e-4
- Retarded-MOT self-test:
  - self_test_pass = true
  - max_delay_over_dt ~= 5.0681e-2
  - temporal closure relative error ~= 1.146e-17
  - assembled/pairwise relative L2 difference ~= 1.873e-14

## CuPy/JefiPIC-dependent source-closure regression

The final Step1C charge-conserving CIC regression imports
`Particle_transport.main` and therefore requires the normal JefiPIC/CuPy
environment. It is not included in the pure-CPU wrapper.

The archived validated Step1C result reports:

- overall_step1c_pass = true;
- worst normalized face-continuity error ~= 9.79e-15;
- worst current-moment relative error ~= 2.78e-16;
- optimized-vs-reference deposition difference = 0 in the reported regression.

## GPU article-scale evidence

The article-scale GPU calculations were **not rerun during packaging**.
Their validated outputs are copied byte-for-byte from the author-supplied
result archives into the Mendeley-oriented data package and compactly mirrored
in `reference_data/`.

The packaging process does not alter the numerical arrays or raw reports.
