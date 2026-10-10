# Reproducibility guide

## Tier 1: deterministic CPU checks

```bash
python scripts/run_cpu_repro.py
pytest -q
```

This exercises MAS dipole continuity, the conducting-sphere benchmark, and
the Retarded-MOT self-test without requiring CuPy.

## Tier 1b: PIC CIC source closure in the JefiPIC/CuPy environment

```bash
python article_cases/step1c_charge_conserving_cic_regression.py   --out article_cases_output/step1c_charge_conserving_cic_final_atomic
```

This script imports `Particle_transport.main`, so a working CuPy/JefiPIC
environment is required even though the validation target is the deposition
identity.

## Tier 2: GPU implementation parity

```bash
python article_cases/step1d_full_em_coincident_atomic_observation_parity.py \
  --out step1d_full_em_coincident_atomic_observation_parity
```

Expected: finite fields and Atomic/observation relative L2 differences near
floating-point precision.

## Tier 3: full-EM broadband audit

```bash
python article_cases/step4b_broadband_plasma_spectrum_audit_full_em_atomic.py \
  --full-em-plasma \
  --cycles 4 \
  --discard-cycles 1 \
  --particles 3000 \
  --grid-n 13 \
  --probes 96 \
  --out step4b_full_em_atomic_final
```

The archived final run has approximately 0.999 MHz spectral-bin spacing and
bin-resolved cumulative-energy frequencies near 9 MHz (95%) and 15 MHz (99%).

## Tier 4: Retarded-MOT temporal characterization

Original 3/6/9 MHz regression:

```bash
python article_cases/step4c2_v4_retarded_frequency_regression.py \
  --out step4c2_v4_formal
```

Supplemental 15 MHz point:

```bash
python article_cases/step4c2_single_frequency_characterization.py \
  --frequency-mhz 15 \
  --out step4c2_15MHz_characterization
```

## Tier 5: controlled collocation-overfit diagnostic

```bash
python article_cases/retarded_harmonic_collocation_overfit_test.py \
  --formal-geometry \
  --sources-per-arm 900 \
  --fit-fractions 1.0 0.5 0.25 \
  --frequency-mhz 3 \
  --out retarded_harmonic_collocation_overfit_3MHz_article_density
```

## Tier 6: full-EM two-way closure / article scale

The archived outputs are GPU-expensive. Use:

```bash
python article_cases/step4d_a_two_way_retarded_mot_closure_full_em_atomic.py \
  --v2e-report reference_data/quasi_auxiliary/step4c3_v2e_report.json \
  --accept-known-c3-limitation \
  --out step4d_a_full_em_atomic_final
```

and then the Step4D-B representative driver if article-scale resources are
available.

The full Mendeley-oriented archive contains the numerical outputs supporting
the manuscript; GitHub keeps compact summaries only.
