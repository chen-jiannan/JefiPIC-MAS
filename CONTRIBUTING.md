# Contributing

Contributions improving reproducibility, portability, tests, documentation, and
numerical diagnostics are welcome.

Before submitting a change:

1. Run `python scripts/run_cpu_repro.py`.
2. Run `pytest -q`.
3. Keep SI units unless a module explicitly documents a conversion.
4. Do not replace independent holdout validation with collocation residuals.
5. For Retarded-MOT or PIC-source changes, state which invariant is preserved
   (continuity, causality, boundary accuracy, or another explicit criterion).
