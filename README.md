# JefiPIC-MAS

JefiPIC-MAS couples a Cartesian kinetic Jefimenko particle-in-cell (PIC)
plasma representation to off-grid metallic boundaries represented by the
Method of Auxiliary Sources (MAS). The release associated with manuscript
version v10 uses charge-conserving first-order CIC deposition, native
staggered-face current histories, continuity-consistent electric-dipole MAS
sources, and a causal Retarded-MOT conductor update.

This repository is a **curated reproducibility release**, not a development-history dump.

## Current release: v1.1.0

The principal article-scale path is now full electromagnetic. The plasma
grid-source field uses the native staggered-current Jefimenko formulation with
pair-parallel atomic-add GPU accumulation. An observation-parallel
implementation is retained as an independent parity path. Both implementations
use the same source histories, retarded interpolation, source locations, and
index-based same-cell exclusion.

`Atomic` here means CUDA atomic accumulation; it is not an atomic-physics model.

## Quick start: deterministic CPU checks

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\Activate.ps1     # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements-cpu.txt
python scripts/run_cpu_repro.py
pytest -q
```

The CPU reproduction wrapper checks:

1. continuity-consistent electric-dipole source semantics;
2. conducting-sphere MAS accuracy;
3. Retarded-MOT recurrence/evaluator self-consistency.

The final charge-conserving CIC regression imports the JefiPIC particle module
and therefore requires the normal CuPy/JefiPIC environment even though the
regression itself targets deposition. Run it separately after the CUDA/CuPy
environment is installed.

## CUDA / GPU quick path

Install the CuPy wheel matching the local CUDA runtime, for example:

```bash
pip install -r requirements-cpu.txt
pip install cupy-cuda12x
python scripts/check_environment.py
python scripts/run_gpu_quick.py
```

The quick GPU path uses the full-electromagnetic Atomic grid-source backend.
After CuPy is available, the final CIC source-closure regression can also be run:

```bash
python article_cases/step1c_charge_conserving_cic_regression.py   --out article_cases_output/step1c_charge_conserving_cic_final_atomic
```

Article-scale runs are not launched in hosted CI.

## Final manuscript-facing evidence

Compact machine-readable summaries are in `reference_data/`.

Principal full-electromagnetic evidence includes:

- CIC face-current continuity at approximately machine precision;
- coincident-grid Atomic/observation full-EM parity at floating-point precision;
- broadband forcing with bin-resolved cumulative-energy frequencies
  \(f95 \approx 9 MHz\) and \(f99 \approx 15 MHz\);
- a supplemental 15 MHz Retarded-MOT harmonic characterization;
- a 668-step full-EM two-way closure run;
- a 12-cycle, 24000-particle-per-species full-EM two-way run.

The frozen 3/6/9 MHz spatial-diagnostic chain and the particle/seed sensitivity
ensemble retain their original quasi-electrostatic plasma-grid-source scope and
are labeled as auxiliary evidence rather than relabeled as full EM.

## GPU reduction strategies

For a discrete grid-source sum `F[i] = sum_j F[i,j]`:

- **pair-parallel atomic-add (Atomic):** one CUDA work item per
  source-observation pair; contributions to a shared observation are combined
  with atomic additions;
- **observation-parallel:** one CUDA work item per observation; that work item
  loops over all sources and writes once.

They are two reductions of the same physical discretization. This release does
not claim a universal performance ordering between them.

## Layout

```text
EMsolver/             Jefimenko + MAS field code
Particle_transport/   particle push and charge-conserving deposition
RBG_Maxwell/          unit-conversion and plasma utilities
article_cases/        manuscript validation / reproduction cases
reference_data/       compact manuscript-facing evidence
examples/             public launchers
scripts/              environment and reproduction helpers
tests/                deterministic CPU tests
docs/                 method scope, data map, GPU and reproducibility notes
provenance/           file manifest and hashes
```

## Scientific scope

This is a computational-method release. It does not claim a new plasma
mechanism or a calibrated antenna port observable. The article-scale
plasma-grid resolution is approximately one Debye length, the current
Retarded-MOT implementation is qualified in the one-step-history regime, and
the floating-charge and RF conductor branches are validated separately.

Applications outside the qualified geometry, frequency, delay, grid, or
boundary regimes should repeat the relevant holdout and source-closure tests.

## Data archive

A CPC/Mendeley-Data-oriented archive is distributed separately. It contains
the fuller numerical outputs, raw/processed broadband data, spatial-diagnostic
forcing, full-EM two-way diagnostics, and a file-by-file manuscript data map.

## License

Apache License 2.0. See `LICENSE`.

## Citation

See `CITATION.cff`. The associated manuscript is:

> Jian-Nan Chen and Yong-Dong Li, *A causal Jefimenko-PIC/MAS method for
> grid-decoupled metallic boundaries in kinetic plasma simulations* (2026 manuscript).
