# Grid-source GPU reductions

For a discrete field sum

`F[i] = sum_j F[i,j]`

the code retains two GPU decompositions.

## Pair-parallel atomic-add (`Atomic`)

One CUDA work item evaluates one source-observation pair `(i,j)`. Many work
items may contribute to the same observation, so the field arrays are updated
with atomic additions.

## Observation-parallel

One CUDA work item owns one observation `i`, loops over all grid sources `j`,
accumulates locally, and writes the observation field once.

## Physical equivalence

Both reductions use the same:

- cell-centered charge history;
- native staggered-face current history in full EM;
- source/observation geometry;
- retarded-time interpolation;
- physical Jefimenko coefficients;
- integer-index same-cell exclusion on coincident cell-center grids.

The full-EM coincident-grid parity regression in `reference_data/full_em/`
shows floating-point-level agreement. The distinction is therefore an
implementation/reduction choice, not a different physical model.

The reported simulations use `Atomic` for plasma grid-source accumulation and
the observation-parallel point-source kernel for MAS dipoles. No universal
performance ranking between the two grid-source decompositions is claimed.
