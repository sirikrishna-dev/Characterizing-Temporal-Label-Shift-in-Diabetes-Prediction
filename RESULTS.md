# Results of the upgrade modules

Every number here comes from a CSV in `results/` (2011-2023) or `results_long/` (1999-2023), and every figure from `plots/upgrade_*.png`. All estimates use the CDC weights, strata and PSUs. Intervals are Rao-Wu bootstrap with seed 2026: 500 replicates for model results, 200 for drift, decomposition and the rolling runs, and 1,000 for the audit simulations.

Each module defines its Benjamini-Hochberg families in its docstring. The families `rule_vs_model`, `pu_gain`, `explain_groups` and the temporal family were fixed before the results were seen. Two families were added after the internal review and are secondary: `pu_beyond_ablation` and `uspstf_35_70`.

| Review point | Module | Short answer |
|---|---|---|
| No method or fix | `pu_models.py`, `pu_eval.py` | SAR-EM raises hidden-case sensitivity in 2017-2020, but simply dropping the access features gets about half of that, and the difference between the two is not significant. |
| No explanation | `explain.py`, `decompose.py` | The model misses undiagnosed people because they are younger and have less insurance and usual care. The rise in diagnosed diabetes is more diabetes, not more diagnosis. |
| No clinical comparison | `screening_rules.py`, `ada_items.py` | Over all adults the model beats USPSTF, but only because USPSTF stops at age 70. Inside ages 35-70 they catch the same share. |
| RQ3 had nothing to catch | `benchmark.py`, `audit_power.py`, `bbse.py` | Drift monitors react to population change, not to harm. A lab audit tracks harm but needs about 7,000 tested adults per period. |
| Mostly null results | `equivalence.py` | Overall AUROC is stable within 0.03 in all 8 cells. That conclusion depends on the margin: at 0.02 only 2 of 8 cells qualify. |
| Short timeline | `extend_1999_2010.py`, `rolling.py` | Eleven cycles, 1999-2023. A model trained on 1999-2000 alone still has AUROC 0.790 in 2021-2023. |

## 1. Fixing the gap with positive-unlabeled learning

Files: `results/pu_eval.csv`, `results/pu_summary.csv`, `plots/upgrade_pu_gain.png`.

Every method is compared at the same flag rate (33.4% of adults, the y_diag logistic model's 2011-2016 rate), so a gain cannot come from flagging more people.

| 2017-2020, overall | Undiagnosed sens. | Diagnosed sens. | Gap | y_bio sens. | y_bio AUROC |
|---|---|---|---|---|---|
| y_diag LightGBM (current model) | 63.0% | 78.4% | 15.4 pts | 75.8% | 0.789 |
| LightGBM without insurance and usual care | 66.1% | 76.8% | 10.7 pts | 75.0% | 0.788 |
| SAR-EM LightGBM (PU, no lab labels) | 69.8% | 75.3% | 5.5 pts | 74.4% | 0.789 |
| y_bio LightGBM (lab test for everyone) | 68.3% | 78.2% | 9.9 pts | 76.5% | 0.792 |
| y_diag logistic (current model) | 65.6% | 77.8% | 12.2 pts | 75.8% | 0.791 |
| Logistic without insurance and usual care | 69.3% | 76.6% | 7.2 pts | 75.3% | 0.790 |
| SAR-EM logistic | 68.5% | 70.0% | 1.5 pts | 69.8% | 0.780 |

What the table shows:

- SAR-EM LightGBM raises undiagnosed sensitivity by 6.9 points over the current model (95% CI 2.0 to 11.7) in 2017-2020. Its cost in overall y_bio sensitivity is -1.4 points (CI -3.5 to 0.7) and AUROC is unchanged. In 2021-2023 the gain is 2.9 points (CI -2.9 to 8.7).
- Half of that gain needs no PU learning. SAR-EM's classifier does not see insurance or usual care. Dropping those two features from an ordinary y_diag model already gives 3.1 points for LightGBM (CI -1.2 to 7.5) and 3.7 for logistic (CI 0.1 to 7.3). SAR-EM's advantage over that simpler model is 3.7 points for LightGBM (CI -0.7 to 8.2, q = 0.33) and -0.8 for logistic (CI -5.7 to 4.1). So the data do not show that the EM step adds anything beyond removing the access features. The one significant cell in that family goes the other way: for adults 60+ in 2021-2023, SAR-EM logistic catches 6.1 points fewer hidden cases than the simpler model (CI -8.3 to -3.8, q = 0.001).
- SAR-EM logistic narrows the gap mostly by flagging fewer diagnosed people. It costs 6.0 points of overall y_bio sensitivity (CI -9.1 to -2.9).
- None of the 96 tests in the pre-registered `pu_gain` family reaches q < 0.05 (smallest q = 0.23). Many intervals exclude zero, but the undiagnosed cells are small (about 280 cases per period).
- Training on y_bio does not fix hidden cases either. About 83% of y_bio cases are already diagnosed, so a y_bio model still learns mostly what diagnosed people look like (+0.7 points for logistic in 2017-2020).
- Elkan-Noto (SCAR) fails, as expected. It leaves the ranking unchanged (identical AUROC), but its implied diabetes prevalence is 48% against a true 12.4%, and 14% of adults get g/c above 1, which cannot happen if diagnosis were random among cases. This is direct evidence that diagnosis depends on who you are.
- Giving SAR-EM true y_bio labels for 5% to 50% of training adults does not raise undiagnosed sensitivity beyond PU alone (`results/pu_audit_curve.csv`, `plots/upgrade_audit_curve.png`). An audit sample is worth more as a monitor (section 4) than as training data.

The practical fix to report is the simplest one: leave insurance and usual care out of a model trained on diagnosis labels. SAR-EM is a principled version of the same idea, but this data cannot separate the two.

## 2. Why hidden cases are missed

Files: `results/explain_decomposition.csv`, `results/explain_coefficients.csv`, `plots/upgrade_explain.png`.

Each model's log-odds score splits exactly into feature contributions (TreeSHAP for LightGBM, coefficient times centred value for logistic; additivity is checked in `test_upgrades.py`). The gap in mean score between diagnosed and undiagnosed cases therefore splits exactly by feature. Pooled over 2017-2023, LightGBM (logistic is almost identical):

- Age contributes 0.16 log-odds (CI 0.06 to 0.25). Undiagnosed cases are about 4 years younger (57.3 vs 61.2 in 2017-2020), and the model leans heavily on age.
- Insurance plus usual source of care contributes 0.086 (CI 0.047 to 0.124). Undiagnosed cases are less often insured (82% vs 93%) and less often have a usual source of care (87% vs 95%). The interval excludes zero in each test period separately (2017-2020: 0.089, CI 0.032 to 0.146; 2021-2023: 0.081, CI 0.012 to 0.149).
- BMI plus waist contributes between -0.03 and -0.09 (CI includes 0). Undiagnosed cases are slightly heavier (BMI 34.3 vs 33.0), which partly offsets the gap.
- In the pre-registered `explain_groups` family (20 tests), age and access have q = 0.075 in 2017-2020, so they are not significant after correction. The pooled estimates are clearly non-zero.
- Switching the training label from y_diag to y_bio shrinks the standardised insurance coefficient from 0.064 to 0.010 and usual care from 0.34 to 0.25, so part of what the y_diag model learns from access is diagnosis, not disease. In the SAR-EM propensity model (being diagnosed, given diabetes) age is the strongest predictor (+1.90), men are less likely to be diagnosed (-0.74) and a usual source of care raises the chance (+0.61).

In one sentence: the model misses undiagnosed people because it relies on age and on having insurance and a usual place of care, which undiagnosed people have less of.

## 3. Clinical screening rules

Files: `results/screening_eval.csv`, `results/screening_eval_35_70.csv`, `plots/upgrade_screening.png`, `plots/upgrade_screening_35_70.png`.

Over all adults, 2017-2020:

| Rule | Adults flagged | Undiagnosed sens. | Diagnosed minus undiagnosed gap |
|---|---|---|---|
| USPSTF 2021 (ages 35-70, BMI 25+, 23+ for Asian adults) | 47% | 65.3% | -3.8 pts |
| y_diag model at USPSTF's flag rate | 47% | 79.8% | 9.8 pts |
| ADA risk test, reduced (score 5+) | 25-28% | 56.0% | 8.4 pts (men: 16.4) |
| ADA: everyone 35+ | 73% | 94.6% | 2.1 pts |
| ADA Standards (35+, or overweight with a risk factor) | 83% | 98.0% | 0.3 pts |

- Over all adults, the model at USPSTF's flag rate catches 15-20 more hidden cases per 100 (LightGBM 2017-2020: +17.6, CI 8.3 to 26.9, q = 0.006). 19 of the 128 tests in `rule_vs_model` are significant.
- That advantage comes from the age limit. USPSTF does not apply above 70, and most of the difference sits in adults 60+ (+50 points, q < 0.001). Restricted to adults aged 35-70, where the rule is meant to be used, USPSTF flags 75.6% of 2011-2016 adults. At that same flag rate the model and the rule catch the same share of hidden cases: 90.5% vs 90.6% for logistic in 2017-2020 (difference -0.1, CI -4.2 to 4.0) and 91.3% vs 90.6% for LightGBM (0.7, CI -3.7 to 5.2). None of the 32 tests in `uspstf_35_70` is significant (smallest q = 0.33).
- Inside ages 35-70 the model still has a diagnosed-vs-undiagnosed gap (5.4 to 5.9 points in 2017-2020) and USPSTF has none (-0.9). The rule treats diagnosed and undiagnosed people alike because it uses only age and BMI.
- The ADA risk score shares the model's blind spot: it scores age and sex as the model does and has a 16.4-point gap in men. The gap comes from weighting age, not from machine learning as such.
- The SAR-EM model at each rule's flag rate has smaller gaps than the y_diag model, for example 1.3 vs 7.9 points at the reduced ADA test's rate and 4.7 vs 9.8 at USPSTF's (2017-2020).

The ADA risk test above is the reduced version. `python -m src.ada_items` downloads the three missing items: family history (MCQ300C), gestational diabetes (RHQ162) and leisure-time activity. `screening_rules.py` then also scores the full seven-item test. CDC dropped MCQ300C and RHQ162 from the 2021-2023 questionnaires (MCQ_L and RHQ_L analytic notes), so the full test exists for 2011-2016 and 2017-2020 only. In 2017-2020 it replaces the reduced test in the `rule_vs_model` family. The public RHQ files carry complete pregnancy answers only for women 20-44, so gestational diabetes is mostly missing for older women and scores 0 there.

BP measurement switched from auscultatory to oscillometric in 2017; BP enters only the ADA rules.

## 4. Monitoring that can be tested

Files: `results/benchmark_*.csv`, `plots/upgrade_benchmark_power.png`.

The benchmark has 8 shift scenarios, 5 severities and 200 replicates each, plus 1,000 no-shift replicates. Each replicate splits 2011-2016 into PSU half-samples and injects a known shift into one half. Every monitor alarms at the 95th percentile of its own no-shift distribution, so the null false-alarm rate is 4-6% for every monitor (checked in the tests). The truth is the population-level change in hidden-case sensitivity. 12 of the 40 scenario cells are harmful (a drop of 5 points or more).

- Drift monitors detect population change, not harm. Weighted PSI alarms in 53% of harmful cells and 33% of harmless ones; the domain classifier in 46% and 36%. They alarm 100% of the time when the population gets older, a shift that improves hidden-case sensitivity by 10.7 points.
- A shift that only changes who gets diagnosed is invisible to feature monitors by construction. In `dx_label`, screening falls among men 20-59 and some of their diagnosed cases become undiagnosed, but no feature changes. The test of the other monitors is the informative part: at severity 5, y_diag prevalence alarms in 17% of replicates, BBSE in 4% and an audit of 4,000 adults in 10%.
- When the same men also lose insurance and usual care (`dx_access`), hidden-case sensitivity falls 13.7 points. Weighted PSI alarms in 8%, y_diag prevalence in 12%, and the audit of 4,000 adults in 46%.
- When hidden cases grow among men 20-59 (`undx_growth`, -6.7 points), the population itself changes a little, and unweighted PSI catches 24%. The audit stays near its null rate.
- Only the lab audit tracks harm without reacting to harmless change. Its false-alarm rate on harmless shifts stays at about 5% and it never alarms for the older-population shift. Its power is low unless the drop is large: 46% at a 13.7-point drop and 69% at an 18-point drop (mixed scenario) with 4,000 audited adults.
- BBSE detects pure prevalence shift best (100% at severity 4 and 5). On the real data (`results/bbse.csv`) its soft estimate of 2017-2020 diabetes prevalence is 17.1% (CI 10.9 to 23.3) against a truth of 14.2%: consistent with the truth but too imprecise to monitor with. Diagnosed prevalence underestimates the truth by 2.4 points (CI -2.8 to -2.0).

Audit sample size (`results/audit_power.csv`, `results/audit_power_sim.csv`, `plots/upgrade_audit_power.png`). Target: detect a 15-point drop in undiagnosed sensitivity with 80% power at one-sided alpha 0.05. The NHANES design effect is 3.3.

To detect a 15-point drop in hidden-case sensitivity, a monitor needs about 7,000 lab-tested adults per period overall, about 6,000 within adults aged 20-59, and about 22,000 adults drawn at random to cover the uninsured. The simulation agrees for the overall cell: 82% power at 8,000 and 74% at 6,000. For scale, the whole NHANES 2017-2020 lab sample is 8,000 adults. That is why the real-data 2x2 could never have caught a subgroup change.

## 5. Decomposing the label change

Files: `results/decomposition.csv`, `plots/upgrade_decomposition.png`.

- From 2011-2016 to 2017-2020, diagnosed prevalence rose 1.50 points (CI 0.46 to 2.55). That splits exactly into 1.52 points of more diabetes (CI 0.47 to 2.57) and -0.02 of more diagnosis (CI -0.41 to 0.38). The fraction diagnosed did not change; the rise is all more disease.
- Population mix explains much of the rise. Reweighting 2011-2016 to the later covariate mix (age, BMI, race and the rest) accounts for 0.76 of the 1.8-point rise in biomarker prevalence by 2017-2020 (CI 0.59 to 0.94), and 1.30 of 1.6 points by 2021-2023 (CI 1.04 to 1.56). The 2021-2023 change is almost entirely an older, heavier population. The coarser Kitagawa check over 16 cells (no BMI) attributes less to composition, as expected.
- No single cycle-to-cycle change is significant (rows marked by cycle in `decomposition.csv`).

## 6. Null results with a margin

Files: `results/equivalence.csv`, `results/equivalence_margins.csv`, `plots/upgrade_equivalence.png`.

The main margins are 0.03 for AUROC and 5 points for sensitivity. They are the team's judgment, not a published standard: 0.03 is about four times the bootstrap SE of the overall AUROC in one period (0.006 to 0.009), and 5 points is the benchmark's harm tolerance. Because the margin is a choice, the same tests are also reported at a stricter and a looser margin.

| AUROC change from 2011-2016, overall cells (both labels, models and test periods) | Margin 0.02 | Margin 0.03 | Margin 0.05 |
|---|---|---|---|
| Cells shown equivalent | 2 of 8 | 8 of 8 | 8 of 8 |

- All 8 overall AUROC changes have 90% intervals inside -0.028 to +0.017, so "stable within 0.03" holds. "Stable within 0.02" does not, so the report should state the margin with the claim.
- Among the 56 primary-subgroup AUROC cells, 17 are equivalent at 0.03 and 45 at 0.05.
- Hidden-case sensitivity is not shown stable in any overall cell at any margin up to 10 points. The worst plausible overall change is -11 to -23 points. That is a statement about precision, and section 4 says what sample size would fix it.

## 7. Timeline: 1999-2023

Files: `nhanes_out/nhanes_all_1999_2023.csv`, `nhanes_out/nhanes_all_1999_2023_validation.csv`, `results_long/rolling_eval.csv`, `results_long/rolling_trend.csv`, plus the 5-cycle runs in `results/rolling_*.csv`.

Building the file. `src/extend_1999_2010.py` adds six cycles (1999-2010), giving 119,555 people over eleven cycles. Every cycle break was checked against the CDC documentation:

- File names change for the early lab files (LAB10, L10_B, L10_C).
- When 1999-2000 and 2001-2002 are pooled, CDC's weighting tutorial requires the 4-year weights; `w_lab_4yr` carries them and `rolling.py` uses them whenever both cycles are in one training window.
- Race uses RIDRETH1 throughout, so there is no separate Asian group before 2011.
- Insurance is HID010 in 1999-2004 and HIQ011 from 2005, with the same question.
- The screening question DIQ180 starts in 2005-2006.
- HbA1c uses the original values, as CDC's A1c notice requires for trend analyses.
- Glucose stays as released, since no long-file analysis uses it.

Validation. Weighted diagnosed prevalence matches the published crude estimates of Selvin et al. 2017 (Ann Intern Med; non-pregnant adults 20+) in all four periods: 6.7% vs 6.35% (1999-2002), 7.9% vs 7.57% (2003-2006), 8.6% vs 8.52% (2007-2010) and 9.8% vs 9.60% (2011-2014). All intervals overlap. Undiagnosed prevalence has no published match: Selvin et al. define it by HbA1c and fasting glucose together, not HbA1c alone.

Findings over eleven cycles:

- A model trained once on 1999-2000 and never retrained keeps its AUROC against y_bio: logistic regression 0.795 at a 2-year lag and 0.790 in 2021-2023, 22 years later (slope -0.011 per decade, p = 0.09). LightGBM goes from 0.774 to 0.769. Retraining every cycle (expanding design) gives 0.794 in 2021-2023, no better than never retraining.
- The diagnosed-vs-undiagnosed gap is not a 2017-2020 event. For the expanding logistic model it is 18.3 points in 2003-2004, 15.6 in 2011-2012 and 14.0 in 2017-2020, and near zero or negative in most other cycles. The trend over the ten test cycles is flat (3.1 points per decade, p = 0.63). The gap comes and goes from cycle to cycle and does not grow.
- USPSTF has a negative gap in 8 of 10 test cycles: it catches undiagnosed cases at least as well as diagnosed ones.
- With 5 cycles (2011-2023), the frozen design trained on 2011-2012 gives logistic AUROC 0.812 at a 2-year lag and 0.797 (CI 0.779 to 0.815) on 2021-2023; LightGBM drops from 0.795 to 0.768. With only four test points no slope is detectable (LightGBM: -0.022 per decade, p = 0.28).

## Limitations

- Model intervals treat fitted models as fixed. The decomposition's reweighting holds the density-ratio weights fixed.
- SAR-EM rests on two assumptions: risk features, not access, determine true diabetes, and the class prior is known from the lab sample. The ablation shows its gain cannot be separated from simply dropping the access features.
- The benchmark is semi-synthetic. We chose its shifts, and real changes in diagnosis practice could look different.
- The ADA risk test is reduced unless `src.ada_items` has been run, and it cannot be complete for 2021-2023. y_bio is still a single HbA1c reading.
- Two BH families (`pu_beyond_ablation`, `uspstf_35_70`) were added after the internal review and are secondary.
- The long file's glucose values are not harmonised across instruments, so y_bio_fpg must not be used on it without applying CDC's trend equations.

## Run order (after the original pipeline)

```
python -m src.extend_1999_2010            # long file + validation, ~1 min with nhanes_raw/ present
python -m src.ada_items                   # needs internet: MCQ, RHQ, PAQ files for the full ADA test
python -m src.pu_models                   # ~6 min
python -m src.pu_eval --reps 500          # ~10 min
python -m src.bbse --reps 200
python -m src.explain --reps 200
python -m src.screening_rules --reps 500  # ~5 min, includes the ages 35-70 comparison
python -m src.decompose --reps 200
python -m src.benchmark --reps 200 --null-reps 1000 --jobs 2   # ~10 min
python -m src.audit_power --reps 1000
python -m src.rolling --reps 200
python -m src.rolling --csv nhanes_out/nhanes_all_1999_2023.csv --out results_long --reps 200
python -m src.equivalence
python -m src.make_upgrade_plots --out plots
python -m src.test_upgrades
```
