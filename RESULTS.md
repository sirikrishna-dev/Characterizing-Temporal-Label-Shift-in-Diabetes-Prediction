# Results (2011-2023 data)

This file answers the four review criticisms with the new modules. Every number below comes from a CSV in `results/` and every figure from `plots/*.png`. All estimates use the CDC weights, strata and PSUs; intervals are Rao-Wu bootstrap (500 replicates for model results, 200 for drift/decomposition, 1000 for audit simulations, seed 2026). Each module's confirmatory Benjamini-Hochberg family is defined in its docstring and was fixed before the results were looked at.

| Criticism | What answers it | Headline |
|---|---|---|
| Evaluation only: no method, no fix, no explanation | `pu_models.py`, `explain.py`, `decompose.py`, `screening_rules.py` | A PU-learning model narrows the hidden-case gap without blood tests. The gap is explained by age and healthcare access, and official screening rules are compared directly. |
| Mostly null results | `equivalence.py` plus the four modules above | AUROC is shown *stable within ±0.03*, not just "not significant". There are several positive findings below. |
| RQ3 cannot test monitors | `benchmark.py`, `audit_power.py`, `bbse.py` | 8,000 shifted replicates with known truth. Drift monitors are blind to every shift that harms hidden-case detection; only a lab audit tracks it, and it needs about 7,000 tested adults per period. |
| Short timeline | `rolling.py`, `extend_1999_2010.py` | Per-cycle expanding and frozen evaluation on 5 cycles; a loader for 1999-2010 ready to run where CDC is reachable. |

---

## 1. Fixing the gap: positive-unlabeled learning (`results/pu_eval.csv`, `plots/pu_gain.png`)

Every method is compared at the **same flag rate** (33.4% of adults, the y_diag logistic model's 2011-2016 rate), so a gain cannot come from flagging more people.

| 2017-2020, overall | Undiagnosed sens. | Diagnosed sens. | Gap | y_bio AUROC |
|---|---|---|---|---|
| y_diag LightGBM (current) | 63.0% | 78.4% | 15.4 pts | 0.789 |
| **SAR-EM LightGBM (PU, no lab labels)** | **69.8%** | 75.3% | **5.5 pts** | 0.789 |
| y_bio LightGBM (lab for everyone) | 68.3% | 78.2% | 9.9 pts | 0.792 |
| y_diag logistic (current) | 65.6% | 77.8% | 12.2 pts | 0.791 |
| SAR-EM logistic | 68.5% | 70.0% | 1.5 pts | 0.780 |

- **SAR-EM LightGBM** raises undiagnosed sensitivity by **+6.9 points (95% CI 2.0 to 11.7)** in 2017-2020. Overall y_bio sensitivity changes by -1.4 points (CI -3.5 to +0.7) and AUROC by 0.000, so the gain is close to free. In 2021-2023 the gain is +2.9 (CI -2.9 to +8.7).
- Gains are largest where the gap was. Among adults aged 20-59 in 2017-2020, undiagnosed sensitivity rises +12.5 points with SAR logistic (CI 4.4 to 20.7) and +10.7 with SAR LightGBM. Among the uninsured it rises +21.4, and among low-income adults +20.2.
- **Honest caveat:** none of the 96 tests in the pre-registered `pu_gain` family reaches q < 0.05 (smallest q = 0.23). Many CIs exclude zero, but the family is large and the undiagnosed cells are small (about 280 cases per period).
- **SAR logistic closes the gap mostly by flagging fewer diagnosed people.** It costs 6.0 points of overall y_bio sensitivity (CI -9.1 to -2.9). Report the LightGBM version as the fix and the logistic one as the cautionary case.
- **Training on y_bio is not an upper bound for hidden cases.** About 83% of y_bio cases are already diagnosed, so a y_bio model still learns mostly what diagnosed people look like (+0.7 points for logistic in 2017-2020). SAR-EM targets the undiagnosed directly.
- **Elkan-Noto (SCAR) fails, as the data predict.** It leaves the ranking unchanged (checked: identical AUROC), but its implied diabetes prevalence is 48% (truth 12.4%). 14% of adults get g/c > 1, which is impossible under SCAR. Diagnosis is not random among cases, and this is direct evidence of it.
- **Audit-label curve** (`results/pu_audit_curve.csv`, `plots/audit_curve.png`): giving SAR-EM true y_bio labels for 5-50% of training adults does **not** improve undiagnosed sensitivity beyond PU alone. The prior-pinned PU model already captures what the labels add. This is a negative result worth stating: the audit sample is worth more as a monitor (section 4) than as training data.

## 2. Why hidden cases are missed (`results/explain_decomposition.csv`, `plots/explain.png`)

The model's log-odds score splits exactly into feature contributions: TreeSHAP for LightGBM, coefficient times centred value for logistic. Additivity is checked in `test_analyses.py`. The gap in mean score between diagnosed and undiagnosed cases therefore splits exactly by feature.

Pooled 2017-2023 (LightGBM; logistic is almost identical):

- **Age: +0.16 log-odds (CI 0.06 to 0.25).** Undiagnosed cases are about 4 years younger (57.3 vs 61.2 in 2017-2020), and the model leans heavily on age.
- **Insurance + usual source of care: +0.086 (CI 0.047 to 0.124).** Undiagnosed cases are less often insured (82% vs 93%) and less often have a usual source of care (87% vs 95%). This holds **separately in both test periods** (2017-2020: 0.089, CI 0.032 to 0.146; 2021-2023: 0.081, CI 0.012 to 0.149). That is the "learns who gets diagnosed" mechanism, measured directly.
- **BMI + waist: -0.03 to -0.09 (CI includes 0).** Undiagnosed cases are slightly *heavier* (BMI 34.3 vs 33.0), which partly offsets the gap. The model does not miss them for looking healthy.
- Sentence for the report: *the model misses undiagnosed people because it relies on age and on having insurance and a usual place of care, which undiagnosed people have less of; their higher BMI only partly compensates.*
- In the pre-registered explanation family (20 tests), age and access have q = 0.075 in 2017-2020, so they are not significant after correction. The pooled estimates are clearly non-zero.
- `results/explain_coefficients.csv` supports this from two directions. (a) Switching the training label from y_diag to y_bio shrinks the standardised insurance coefficient from 0.064 to 0.010 and usual care from 0.34 to 0.25: part of what the y_diag model learns from access is diagnosis, not disease. (b) In the SAR-EM propensity model (being diagnosed, given diabetes), age is the strongest predictor (+1.90), men are less likely to be diagnosed (-0.74), and a usual source of care raises the chance (+0.61). Insurance adds little once usual care is in the model (+0.06).

## 3. Clinical screening rules (`results/screening_eval.csv`, `plots/screening.png`)

| 2017-2020 | Adults flagged | Undiagnosed sens. | Diagnosed minus undiagnosed gap |
|---|---|---|---|
| USPSTF 2021 (35-70, BMI >= 25 / 23 Asian) | 47% | 65.3% | **-3.8 pts** (no gap) |
| y_diag model at USPSTF's flag rate | 47% | 79.8% | 9.8 pts |
| ADA risk test, reduced (score >= 5) | 25-28% | 56.0% | 8.4 pts (men: 16.4) |
| ADA age >= 35 | 73% | 94.6% | 2.1 pts |
| ADA Standards (age >= 35 or overweight + risk factor) | 83% | 98.0% | 0.3 pts |

- **At USPSTF's own flag rate, the model catches 15-20 points more hidden cases** (LightGBM 2017-2020: +17.6, CI 8.3 to 26.9, q = 0.006; logistic 2021-2023: +19.9, q = 0.026). Most of the difference is adults over 70, whom USPSTF excludes (60+: +50 points, q < 0.001). 19 of 128 tests in the `rule_vs_model` family are significant.
- **The official rules do not share the model's blind spot, but the ADA risk score does.** USPSTF has no diagnosed-vs-undiagnosed gap (it is -9.7 points in men). The ADA risk test, which scores age and sex as the model does, shows the same pattern as the model: a gap of 16.4 points in men. The gap comes from weighting age, not from machine learning as such. Clinically, risk scores that lean on age will under-serve younger undiagnosed men.
- The SAR-EM model at each rule's flag rate has smaller gaps than the y_diag model, for example 1.3 vs 7.9 points at the reduced ADA test's flag rate and 4.7 vs 9.8 at USPSTF's (2017-2020).
- Limits: the ADA risk test is a **reduced** version (no family history, gestational diabetes or physical activity in the project file; these need the MCQ, RHQ and PAQ files). BP measurement switched from auscultatory to oscillometric in 2017.

## 4. Monitoring that can be tested (`results/benchmark_*.csv`, `plots/benchmark_power.png`)

The benchmark has 8 shift scenarios × 5 severities × 200 replicates, plus 1,000 no-shift replicates. Each replicate splits 2011-2016 into PSU half-samples and injects a known shift into one half. Every monitor alarms at the 95th percentile of its no-shift distribution; the null false-alarm rate is 4-6% for every monitor (checked in the tests). The truth is the population-level change in hidden-case sensitivity. 12 of the 40 cells are harmful (a drop of 5 points or more).

- **Drift monitors detect population change, not harm.** Feature PSI alarms in 53% of harmful cells but also in 33% of harmless ones. Domain classifier: 46% vs 36%. They alarm at 100% when the population gets older, which *improves* hidden-case sensitivity by 10.7 points.
- **Every diagnosis-mechanism shift is invisible to every drift monitor.** Screening falls among men 20-59, with or without loss of access, or hidden cases grow among men 20-59: every drift monitor, BBSE and y_diag prevalence stay at about 5%, the rate under no shift at all. That holds even at severity 5, where hidden-case sensitivity falls 13.7 points. This is the shift the project is about, and the standard tools cannot see it.
- **Only the lab audit tracks harm.** Its false-alarm rate on harmless shifts stays at about 5%, and its power rises with severity (n = 4,000: 46% at a 13.7-point drop, 69% at an 18-point drop). It never alarms for the harmless "older population" shift.
- **BBSE** detects pure prevalence shift best (100% at severity 4+). On the real data (`results/bbse.csv`) its estimate of 2017-2020 diabetes prevalence is 17.1% (soft version; CI 10.9 to 23.3) against a truth of 14.2%. That is unbiased within error but too imprecise to use as a monitor. The naive diagnosed prevalence underestimates the truth by 2.4 points (CI -2.8 to -2.0).

**Audit power analysis** (`results/audit_power.csv`, `results/audit_power_sim.csv`, `plots/audit_power.png`). Target: detect a 15-point drop in undiagnosed sensitivity with 80% power at one-sided alpha 0.05. The NHANES design effect is 3.3.

> **To detect a 15-point drop in hidden-case sensitivity you need about 7,000 lab-tested adults per period overall, about 6,000 within adults aged 20-59, and about 22,000 adults drawn at random to cover the uninsured.**

The simulation agrees for the overall cell: 82% power at 8,000 and 74% at 6,000. Subgroup simulations need more, because they also estimate the reference from a half-sample. For scale, the whole NHANES 2017-2020 lab sample is about 8,000 adults. A monitor of the hidden-case gap therefore needs an audit about the size of NHANES itself, which explains why the real-data 2×2 could never have caught a subgroup change.

## 5. Decomposition of the label change (`results/decomposition.csv`, `plots/decomposition.png`)

- **2011-2016 to 2017-2020:** diagnosed prevalence rose **+1.50 points (CI 0.46 to 2.55)**. That splits exactly into **+1.52 from more diabetes (CI 0.47 to 2.57)** and **-0.02 from more diagnosis (CI -0.41 to 0.38)**. The fraction diagnosed did not change; the rise is all more disease.
- **Population mix explains much of the rise.** Reweighting 2011-2016 to the later covariate mix (age, BMI, race and the rest) explains +0.76 of the +1.8-point rise in biomarker prevalence by 2017-2020 (CI 0.59 to 0.94), and +1.30 of +1.6 by 2021-2023 (CI 1.04 to 1.56). The 2021-2023 change is almost entirely an older, heavier population. The coarser Kitagawa check over 16 cells (no BMI) attributes less to composition, as expected.
- Consecutive cycles: no single cycle-to-cycle change is significant; all are in the `decomposition.csv` rows marked by cycle.

## 6. Null results made informative (`results/equivalence.csv`, `plots/equivalence.png`)

- **Overall AUROC is equivalent (stable within ±0.03) in all 8 overall cells**, both labels, both models and both periods; the 90% CIs lie within -0.028 to +0.017. Across all 64 AUROC cells, 25 are shown equivalent; the rest (mostly subgroup cells) are too imprecise to decide either way.
- **Hidden-case sensitivity is not shown stable in any cell.** The worst plausible change overall is -11 to -23 points. That is a precision statement, and it is exactly what the audit power analysis quantifies.

## 7. Timeline (`results/rolling_eval.csv`, `results/rolling_trend.csv`)

- **Frozen design** (train once on 2011-2012, never retrain): logistic regression keeps its AUROC against y_bio, 0.812 at a 2-year lag and 0.797 (CI 0.779 to 0.815) on 2021-2023, about ten years later. LightGBM loses more, from 0.795 to 0.768: the more flexible model ages worse. With four test points no slope is statistically detectable (LightGBM: -0.022 per decade, p = 0.28), so treat this as descriptive.
- **Expanding design** (retrain every cycle on all earlier cycles): the diagnosed-vs-undiagnosed gap is largest in 2017-2020 (12.1 points, CI 1.0 to 23.3), the only cycle where it excludes zero.
- **1999-2010:** `src/extend_1999_2010.py` has the per-cycle file map, weight rules (including 4-year weights for 1999-2002), RIDRETH1 race harmonization, HID010/HIQ011 insurance mapping and a validation table. **It has not been run yet:** CDC was unreachable from the environment it was written in. Items marked CONFIRM must be checked against the codebooks. Every module takes `--csv`, so the long file drops straight in.## Limitations to state

- Model intervals treat fitted models as fixed (as before). The decomposition's reweighting holds the density-ratio weights fixed.
- SAR-EM identifiability rests on two assumptions: risk features (not access) determine true diabetes, and the class prior is known from the lab sample.
- The benchmark is semi-synthetic. Its shifts are chosen by us, and real diagnosis-practice changes could look different.
- The ADA risk test is a reduced version, and y_bio is still a single HbA1c reading.

## Run order (after the original pipeline)

```
python -m src.pu_models                 # ~6 min
python -m src.pu_eval --reps 500        # ~15 min
python -m src.bbse --reps 200
python -m src.explain --reps 200
python -m src.screening_rules --reps 500  # ~6 min
python -m src.decompose --reps 200
python -m src.benchmark --reps 200 --null-reps 1000 --jobs 2   # ~10 min
python -m src.audit_power --reps 1000
python -m src.rolling --reps 200
python -m src.equivalence
python -m src.make_analysis_plots
python -m src.test_analyses
```
