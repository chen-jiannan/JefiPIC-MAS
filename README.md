# JefiPIC-MAS

JefiPIC-MAS couples a kinetic Jefimenko particle-in-cell model to metallic
boundaries represented independently of the Cartesian plasma-source grid.
This repository contains the current manuscript implementation: unsplit
charge-conserving particle sources, causal electric/magnetic auxiliary-source
histories, a normalized mixed boundary solve and native CUDA feedback.

## Source layout

```text
EMsolver/            Shared field kernels and MAS boundary operators
Particle_transport/  Particle motion and charge-conserving deposition
RBG_Maxwell/         Units and plasma utilities
support/             Geometry, drive and particle-setup helpers
examples/two_arm/    Driven 3-MHz two-arm case and electric/magnetic feedback
examples/sphere/     Self-consistent particle cloud near a PEC sphere
benchmarks/          SCUFF-EM and GetDP reference models
data/                Compact measured results and plot inputs
scripts/             Figure, reference-input and comparison reproduction
docs/                Method and usage notes
provenance/          Source relocation map and verification records
run.py               Public command entry point
```

Both physical examples use one shared source tree. No previous repository
snapshot or release directory is embedded here. Run from a source checkout;
install requirements rather than treating this as a PyPI wheel.

## CPU checks and figures

```bash
python -m pip install -r requirements-cpu.txt
python run.py verify
python run.py cpu
python run.py sphere-cpu
python run.py figures
```

The CPU checks test continuity, source/feedback arithmetic and the sphere
boundary using supplied fixtures. Figure regeneration uses archived data;
outputs are under outputs/. None of these commands is a new GPU particle run.

## CUDA calculations

Install a CuPy wheel matching the local CUDA environment, for example
`python -m pip install cupy-cuda12x` on a compatible CUDA 12 machine.

```bash
python run.py rf --gpu 0 --busy share --threads 4
python run.py sphere --gpu 0 --busy share
```

The RF driver performs CPU and native CUDA gates before advancing 4001 nodes.
`rf --mode replay` runs only the native gates. For instrumented timing,
use `rf --gpu 0 --busy wait --threads 4 --profile`; this includes diagnostic
synchronization, not repeated uninstrumented throughput.

The RF example has 1992 electric and 664 magnetic source points, 5976 unknowns,
theta=0.6 and dt=1 ns. The sphere example uses a separate reduced mixed
boundary with 192 electric sources. Both use unsplit particles and conserving
deposition. The physical cases and their reference configurations differ.

## Independent comparisons

```bash
python run.py warpx-inputs
python run.py scuff-mas
```

WarpX inputs use the shared sphere particle library and native Esirkepov
deposition. SCUFF-MAS computes the augmented periodic MAS response against
archived actual SCUFF fields; it does not rerun SCUFF or PIC. Its dense
5976-unknown algebra needs substantial host memory. See benchmarks/ for
external SCUFF and GetDP model execution with separately installed binaries.

The completed 12-cycle RF run has 0.604% global tangential-E residual,
1.368% worst accumulated regional E residual and 0.131% normal-B residual
normalized by incident vector B. Source continuity is below 5.42e-14.
These are boundary/source diagnostics. SCUFF offset E/B differences are
1.863%/1.119%. The sphere/WarpX comparison has 5.80% exterior primary E and
5.12% off-axis B differences; the near-cloud P0 discrepancy remains in data/.

The recurrence is qualified for smooth electrically small conductors with
source-to-boundary delays shorter than one time step. Native GPU results
refer to the executed records, not to this directory-reorganization step.
See provenance/layout_validation.json for path/import changes and CPU checks.

## Code and data

The associated manuscript is A causal Jefimenko-PIC/MAS method for
grid-decoupled metallic boundaries in kinetic plasma simulations, by
Jian-Nan Chen, Hong-Guang Wang and Yong-Dong Li. Cite the actual archived
dataset version used. GitHub hosts the current source; Mendeley keeps
published version history. No semantic release number is imposed by this
source layout. The internal algorithm identifiers still distinguish cases.

Source code is Apache-2.0. Reference solvers retain their own licenses and
are obtained separately. Numerical data licenses follow their archive record.
