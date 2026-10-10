# Method scope

The principal validated two-way configuration uses the full-electromagnetic
Jefimenko plasma grid-source formulation, charge-conserving native
staggered-face PIC current, pair-parallel atomic-add grid reduction, and the
causal Retarded-MOT MAS conductor response.

The following limits are explicit:

- the conductor boundary is grid decoupled, but the plasma remains on a Cartesian mesh;
- the article grid has dx/lambda_D approximately 1.01 and is not a
  grid-converged Debye-sheath study;
- the current Retarded-MOT implementation is qualified for R/c < dt and must
  be generalized to multi-step history for larger delays;
- the 9 MHz spatial-diagnostic residual is a property of the current MAS
  representation, not a physical frequency cutoff;
- the 5% boundary line is a study-specific relative tangential-field reference
  target, not a universal PIC/MAS standard;
- the floating-conductor charge branch and RF Retarded-MOT branch are
  independently validated but are not jointly advanced on the same
  particle-collecting RF conductor;
- Mz is an auxiliary current-moment sensitivity diagnostic, not a calibrated
  terminal current, gain, impedance, or converged plasma observable.

Quasi-electrostatic datasets are retained only where they provide auxiliary
spatial or finite-particle sensitivity evidence; they are not relabeled as
full-electromagnetic results.
