# Manuscript evidence map

| Manuscript claim | Public compact evidence |
|---|---|
| charge-conserving CIC continuity | `reference_data/full_em/step1c_charge_conserving_cic_report.json` |
| Atomic/observation full-EM parity | `reference_data/full_em/full_em_atomic_observation_parity_report.json` |
| full-EM broadband f95/f99 | `reference_data/full_em/step4b_full_em_broadband_report_raw.json` plus `FINAL_KEY_NUMBERS.csv` |
| Retarded-MOT long-horizon stability | `reference_data/temporal/step4c1_v4_formal_report.json` |
| 3/6/9 MHz harmonic regression | `reference_data/temporal/step4c2_v4_3_6_9MHz_report.json` |
| supplemental 15 MHz characterization | `reference_data/temporal/step4c2_15MHz_characterization_report.json` |
| controlled collocation overfit | `reference_data/spatial_auxiliary/` |
| 3/6/9 MHz MAS spatial envelope | `reference_data/quasi_auxiliary/step4c3_*` |
| 668-step full-EM closure | `reference_data/full_em/step4d_a_full_em_atomic_report.json` |
| 12-cycle full-EM article scale | `reference_data/full_em/step4d_b_full_em_article_report.json` |
| Mz window sensitivity | `reference_data/full_em/response_stationarity_*` |
| quasi particle/seed sensitivity | `reference_data/quasi_auxiliary/` |

The raw Step4B report retains an older internal field name associated with the
original audit script. Manuscript interpretation uses only the spectrum and
full-EM configuration fields; `FINAL_KEY_NUMBERS.csv` gives the final
manuscript-facing bandwidth summary.
