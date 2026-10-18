# Measured results

Primary seed: 101. Counts are per condition; scores are uncalibrated.

| Model | Condition | Correct / n | Accuracy | Macro precision | Macro recall | Macro F1 | Log loss |
|---|---|---:|---:|---:|---:|---:|---:|
| biased | train | 1200/1200 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0011 |
| biased | same_rule | 1200/1200 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0011 |
| biased | background_reversed | 0/1200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.8649 |
| biased | background_independent | 600/1200 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 3.4325 |
| biased | position_shifted | 1200/1200 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0045 |
| varied | train | 1200/1200 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2626 |
| varied | same_rule | 1200/1200 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2656 |
| varied | background_reversed | 1200/1200 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2552 |
| varied | background_independent | 1200/1200 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2603 |
| varied | position_shifted | 602/1200 | 0.5017 | 0.7504 | 0.5017 | 0.3370 | 0.4713 |

Full per-class metrics and confusion matrices: metrics.json. All seeds: metrics.csv and summary.csv.
Paired background conditions share source objects and noise. Do not treat them as independent replications.
