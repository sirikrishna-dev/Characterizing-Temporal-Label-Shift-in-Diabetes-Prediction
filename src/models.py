"""RQ2 diabetes prediction: survey-weighted models with train-only preprocessing.

Run from the project root:
    python -m src.models

Output: results/rq2_models.joblib, holding
  - the final logistic regression and LightGBM, fitted on all 2011-2016 rows
  - one decision threshold per model, chosen by weighted Youden J on
    out-of-fold predictions (leave one train cycle out), so the threshold is
    not tuned on predictions the model has already memorised
  - the out-of-fold predictions themselves (one row per training participant,
    keyed by SEQN). model_eval.py scores these as the 2011-2016 baseline.

No test-period row is used here for fitting, preprocessing or thresholds.
"""
from dataclasses import dataclass, field
from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

RAW_FEATURES = ("age", "education", "pir", "bmi", "waist", "insured", "routine_care")
FEATURE_NAMES = RAW_FEATURES + ("male", "pir_missing")
MODEL_NAMES = ("logistic", "lightgbm")
SEED = 2026

# Columns the model must never see. HbA1c and glucose define y_bio, and the
# medication questions (DIQ050 insulin, DIQ070 pills) are only asked of people
# who already know they have diabetes.
FORBIDDEN_FEATURES = ("hba1c", "fpg", "LBXGH", "LBXGLU", "DIQ010", "DIQ050", "DIQ070",
                      "y_diag", "y_bio", "y_bio_any", "y_bio_fpg", "undx", "undx_fpg",
                      "screened")
assert not set(FEATURE_NAMES) & set(FORBIDDEN_FEATURES)


def as_bool(series):
    """'adult' can come back from a CSV as bool, 0/1 or the text True/False."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def analysis_domain(df):
    """Shared analysis sample: adult, positive lab weight, both labels observed."""
    weight = pd.to_numeric(df["w"], errors="coerce")
    return (as_bool(df["adult"]) & weight.gt(0) & np.isfinite(weight)
            & df["y_diag"].notna() & df["y_bio"].notna())


class TrainOnlyPreprocessor:
    """Median-impute numeric fields and standardise them with statistics from
    the rows passed to fit() only.

    PIR missingness is kept as its own 0/1 feature. Sex must be male/female;
    anything else raises an error instead of being silently coded.
    """

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=False)
        self.scaler = StandardScaler()
        self.fitted = False

    def _raw(self, df):
        missing = set(RAW_FEATURES + ("sex",)) - set(df.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        numeric = df.loc[:, list(RAW_FEATURES)].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        sex = df["sex"].astype("string").str.strip().str.lower()
        if not sex.isin(["male", "female"]).all():
            raise ValueError("sex must be 'male' or 'female' for all analysis rows")
        male = sex.eq("male").astype(float).to_numpy()
        pir_missing = numeric["pir"].isna().astype(float).to_numpy()
        return numeric, male, pir_missing

    def fit(self, train_df):
        numeric, _, _ = self._raw(train_df)
        if numeric.isna().all().any():
            raise ValueError("A training numeric feature is entirely missing")
        self.scaler.fit(self.imputer.fit_transform(numeric))
        self.fitted = True
        return self

    def transform(self, df):
        if not self.fitted:
            raise RuntimeError("Fit the preprocessor on training rows first")
        numeric, male, pir_missing = self._raw(df)
        scaled = self.scaler.transform(self.imputer.transform(numeric))
        return np.column_stack((scaled, male, pir_missing))


def weighted_youden_threshold(y, probability, weight):
    """Threshold that maximises weighted Youden J = sensitivity + specificity - 1.

    Ties go to the highest threshold so the result is deterministic. The
    infinite 'predict nobody' threshold is excluded.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    w = np.asarray(weight, dtype=float)
    if not (len(y) == len(p) == len(w)) or not np.isfinite(p).all():
        raise ValueError("Invalid predictions or mismatched lengths")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Need both classes to choose a threshold")
    if not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Weights must be finite and positive")
    fpr, tpr, thresholds = roc_curve(y, p, sample_weight=w, drop_intermediate=False)
    finite = np.isfinite(thresholds)
    j = tpr - fpr
    best = np.flatnonzero(finite & np.isclose(j, j[finite].max(), atol=1e-12))
    return float(thresholds[best[0]])


def make_estimators():
    return {
        "logistic": LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED),
        "lightgbm": LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                   max_depth=-1, min_child_samples=40, random_state=SEED,
                                   n_jobs=1, verbosity=-1),
    }


def fit_on_rows(rows):
    """Fit the preprocessor and both models on the given training rows.

    Weights are divided by their mean. A constant rescale does not change the
    population the weights describe, and it keeps the logistic regression's
    default regularisation on a sensible scale.
    """
    y = pd.to_numeric(rows["y_diag"], errors="raise").to_numpy(dtype=int)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_diag must be binary")
    w = pd.to_numeric(rows["w"]).to_numpy(dtype=float)
    w = w / w.mean()
    prep = TrainOnlyPreprocessor().fit(rows)
    X = pd.DataFrame(prep.transform(rows), columns=FEATURE_NAMES)
    estimators = make_estimators()
    for estimator in estimators.values():
        estimator.fit(X, y, sample_weight=w)
    return prep, estimators


def predict_with(prep, estimator, df):
    X = pd.DataFrame(prep.transform(df), columns=FEATURE_NAMES)
    return estimator.predict_proba(X)[:, 1]


@dataclass
class ModelBundle:
    preprocessor: TrainOnlyPreprocessor
    estimators: dict
    thresholds: dict
    oof: pd.DataFrame = field(default=None)
    feature_names: tuple = FEATURE_NAMES

    def predict_probabilities(self, df, model_name):
        if model_name not in self.estimators:
            raise KeyError(f"Unknown model: {model_name}")
        return predict_with(self.preprocessor, self.estimators[model_name], df)

    def predict_labels(self, df, model_name):
        p = self.predict_probabilities(df, model_name)
        return (p >= self.thresholds[model_name]).astype(int)


def train_models(df):
    """Fit on 2011-2016 only, target y_diag only.

    y_bio is used here only to restrict to the rows where both labels exist,
    so that the later y_diag vs y_bio comparison is on identical people.

    Out-of-fold predictions: for each train cycle, refit everything
    (imputer, scaler, both models) on the other two cycles and predict the
    held-out one. Thresholds come from these predictions.
    """
    required = {"period", "cycle", "SEQN", "w", "adult", "y_diag", "y_bio", "sex",
                *RAW_FEATURES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    rows = df.loc[(df["period"] == "train") & analysis_domain(df)].reset_index(drop=True)
    if rows.empty:
        raise ValueError("No valid training rows")
    cycles = sorted(rows["cycle"].astype(str).unique())
    if len(cycles) < 2:
        raise ValueError("Need at least two train cycles for out-of-fold predictions")

    oof = pd.DataFrame({"SEQN": rows["SEQN"].to_numpy(), "cycle": rows["cycle"].astype(str)})
    for name in MODEL_NAMES:
        oof[name] = np.nan
    for held_out in cycles:
        test_mask = rows["cycle"].astype(str).eq(held_out).to_numpy()
        prep, estimators = fit_on_rows(rows.loc[~test_mask])
        for name, estimator in estimators.items():
            oof.loc[test_mask, name] = predict_with(prep, estimator, rows.loc[test_mask])
    if oof[list(MODEL_NAMES)].isna().any().any():
        raise RuntimeError("Some training rows got no out-of-fold prediction")

    y = rows["y_diag"].to_numpy(dtype=int)
    w = rows["w"].to_numpy(dtype=float)
    thresholds = {name: weighted_youden_threshold(y, oof[name].to_numpy(), w)
                  for name in MODEL_NAMES}

    prep, estimators = fit_on_rows(rows)
    return ModelBundle(prep, estimators, thresholds, oof)


def main():
    parser = argparse.ArgumentParser(description="Fit RQ2 models on 2011-2016 rows only")
    parser.add_argument("--csv", default="nhanes_out/nhanes_all.csv")
    parser.add_argument("--output", default="results/rq2_models.joblib")
    args = parser.parse_args()
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"Training rows: {int(((df['period'] == 'train') & analysis_domain(df)).sum())}")
    bundle = train_models(df)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    for name, threshold in bundle.thresholds.items():
        print(f"{name}: out-of-fold weighted Youden threshold = {threshold:.6f}")
    print(f"Saved models, thresholds and out-of-fold predictions: {output}")


if __name__ == "__main__":
    # Pickle the bundle under the importable module path src.models, not __main__.
    from src.models import main as canonical_main
    canonical_main()
