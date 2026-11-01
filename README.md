# Characterizing temporal label shift in diabetes prediction

Team Epoch, ADA Jackfruit project, PES University.

A diabetes model trained on "has a doctor told you that you have diabetes" learns who gets diagnosed, which is not the same as who has diabetes. We train on NHANES 2011-2016, test on 2017-March 2020 and August 2021-2023, and score every model twice: against the diagnosis label (y_diag) and against a biomarker label (y_bio = diagnosed or HbA1c >= 6.5%). Every estimate uses the CDC survey weights, strata and PSUs, so changes between cycles are not just changes in who NHANES chose to sample.

## Layout

```
nhanes_pipeline_v7.ipynb   data notebook: download, merge, weights, labels, NCHS checks
nhanes_out/                cleaned data (nhanes_all.csv) and the notebook's audit tables
src/
  crosswalk.py             raw NHANES names -> project names
  labels.py                labels and subgroups (fills any that are missing)
  pilot_table.py           weighted diagnosed prevalence, fraction diagnosed, screening
  survey_boot.py           Rao-Wu bootstrap and t intervals used by RQ2
  models.py                RQ2: fit logistic regression and LightGBM on 2011-2016
  model_eval.py            RQ2: weighted metrics, label gaps, change over time
  survey_drift.py          weighted PSI / KS / domain AUC and noise floors used by RQ3
  drift.py                 RQ3: feature, domain and label drift per period and subgroup
  monitor.py               RQ3: joins drift alarms with model degradation (2x2)
  make_review_plots.py     figures for the review deck
  test_*.py                checks, one file per module
results/                   every CSV the modules write
plots/                     review figures
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate          (Windows)   or   source .venv/bin/activate
pip install -r requirements.txt
```

## Run order

Everything runs from the project root. Run times are approximate.

```
python -m src.test_crosswalk
python -m src.test_labels
python -m src.test_survey_boot
python -m src.pilot_table                      # results/pilot_table.csv
python -m src.models                           # results/rq2_models.joblib, ~5 s
python -m src.test_models                      # ~20 s
python -m src.model_eval --reps 500            # model_eval.csv, model_temporal.csv, ~1.5 min
python -m src.test_model_eval
python -m src.drift                            # drift_*.csv, label_shift.csv, ~4 min
python -m src.test_drift
python -m src.monitor                          # monitoring_2x2.csv, monitoring_summary.csv
python -m src.make_review_plots --prevalence nhanes_out/step5_weighted_prevalence.csv --drift results/drift_features.csv --out plots
```

The notebook only needs re-running if the data has to be rebuilt from CDC. It downloads into nhanes_raw/, which git ignores.

## Design decisions

Weights. All HbA1c analyses use w_lab: the MEC exam weight for 2011-2020 and the phlebotomy weight WTPH2YR for 2021-2023. For the pooled 2011-2016 period the weight is divided by 3 (column w in nhanes_all.csv). The models train with that weight as sample_weight, and test_drift.py checks that drift.py pools to the same numbers. Replicates are drawn on every row of the period and the analysis population is applied afterwards by setting weights outside it to zero, so subgroup variances stay correct.

Features. Age, sex, education, income-to-poverty ratio (with a missing flag), BMI, waist, insurance and usual source of care. HbA1c, glucose, the diabetes medication questions and every label column are excluded, and models.py asserts this.

Thresholds and baseline. Each model's decision threshold is the weighted Youden point on out-of-fold predictions: each train cycle is predicted by models fitted on the other two. Picking it on in-sample predictions would favour LightGBM, whose in-sample AUROC (0.888) is far above its out-of-fold AUROC (0.794). The same out-of-fold predictions give the 2011-2016 baseline that the test periods are compared against.

Hidden cases. Besides sensitivity against y_diag and y_bio, model_eval.py reports sensitivity among undiagnosed adults with HbA1c >= 6.5% (undx). The y_diag minus y_bio sensitivity gap is that diagnosed-vs-undiagnosed gap multiplied by the undiagnosed share of y_bio cases (about 17%), which is why the y_bio gap looks small.

Multiple testing. Two families are fixed in advance and corrected with Benjamini-Hochberg. The label_gap family (model_eval.csv) is the AUROC gap between y_diag and y_bio plus the sensitivity gap between diagnosed and undiagnosed cases, overall and in the four primary subgroups (income_band2, insured, age_band2, sex), both models, both test periods. The temporal family (model_temporal.csv) is the change from the 2011-2016 baseline in AUROC on both labels and sensitivity on y_diag, y_bio and undx, over the same cells. Brier score, calibration-in-the-large, ECE and specificity are reported with intervals only, because their label gaps follow from the two labels having different prevalence. Exploratory subgroups (income_band, routine_care, age_band, race_eth4, race_eth) also get intervals only. A subgroup level that holds exactly the same people as an earlier one (income_band <1.3, age_band 60+, race_eth NH White and NH Black) is evaluated once.

Small cells. A cell with fewer than 30 undiagnosed cases keeps its point estimates but gets no interval or test. In 2021-2023 this removes the low-income and uninsured cells.

Joining RQ2 and RQ3. model_eval.csv and model_temporal.csv use the same keys as drift_alarms.csv: period_label (2017-2020, 2021-2023), subgroup, level, with the overall row as all / all.

## Results (500 replicates, seed 2026)

Discrimination did not change over time. Out-of-fold AUROC on 2011-2016 is 0.799 (logistic) and 0.794 (LightGBM); in both test periods it stays between 0.785 and 0.796 on either label. None of the 160 tests in the temporal family is significant after correction.

Undiagnosed cases are missed more often, mainly in 2017-2020. Overall in 2017-2020, logistic regression flags 77.8% of diagnosed cases and 65.6% of undiagnosed ones (gap 12.2 points, 95% CI 1.2 to 23.3, q = 0.21); for LightGBM it is 80.4% vs 63.8% (gap 16.6, CI 5.0 to 28.2, q = 0.11). Two cells survive correction, both LightGBM in 2017-2020: men (gap 28.3 points, CI 12.3 to 44.2, q = 0.047) and ages 20-59 (23.4 points, CI 9.9 to 36.9, q = 0.047). In 2021-2023 the overall gap is about 3 points with wide intervals, and the low-income and uninsured cells are too small to test, so a smaller gap there is not evidence that it closed.

The weighted drift monitors alarm, but not where the model fails. Weighting removes most of the apparent drift: in 2021-2023 the unweighted race/ethnicity PSI is 0.287 and the weighted one is 0.047, and only insurance (2021-2023) and education (2017-2020) remain real overall shifts. The domain classifier still separates the periods (weighted AUC 0.59 and 0.61). Across the 18 cells per model (overall and primary subgroups, both periods), the monitors alarm in 11. Performance degrades in none of the 16 cells that are large enough to test, so every alarm there is a false alarm with respect to model performance. The two cells with significant hidden-case gaps (men and ages 20-59 in 2017-2020) are cells where no monitor alarmed.

## Limitations

Intervals treat the fitted models and thresholds as fixed, so they leave out training variability. y_bio uses a single HbA1c reading, which is a research definition, not a clinical diagnosis. ECE depends on its ten fixed bins. Subgroups overlap, so tests within a family are correlated. The 2017-2020 cycle mixes 2017-2018 with part of a pandemic-shortened cycle and carries its own CDC weights.
