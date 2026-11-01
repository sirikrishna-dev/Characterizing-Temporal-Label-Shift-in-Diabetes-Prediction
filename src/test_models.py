"""Run from the project root: python -m src.test_models"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models import (FEATURE_NAMES, FORBIDDEN_FEATURES, TrainOnlyPreprocessor,
                        analysis_domain, train_models, weighted_youden_threshold)


def run_tests():
    file = Path("nhanes_out/nhanes_all.csv")
    if not file.exists():
        raise FileNotFoundError("Run from the project root with nhanes_out/nhanes_all.csv present")
    df = pd.read_csv(file, low_memory=False)
    mask = analysis_domain(df)
    counts = df.loc[mask].groupby("period").size().to_dict()
    assert counts == {"train": 15432, "test_P": 8000, "test_L": 5730}, counts
    print("PASS: shared adult / positive-weight / both-label sample sizes", counts)

    assert not set(FEATURE_NAMES) & set(FORBIDDEN_FEATURES)
    print("PASS: no HbA1c, glucose, medication or label column among the features")

    fitted = train_models(df)
    assert set(fitted.estimators) == {"logistic", "lightgbm"}
    train = df.loc[mask & df["period"].eq("train")].reset_index(drop=True)
    oof = fitted.oof.set_index("SEQN").loc[train["SEQN"]]
    assert len(oof) == len(train) and np.isfinite(oof[["logistic", "lightgbm"]]).all().all()
    for model in fitted.estimators:
        expected = weighted_youden_threshold(train["y_diag"].astype(int), oof[model], train["w"])
        assert fitted.thresholds[model] == expected
        sample = df.loc[mask & df["period"].eq("test_P")].head(100)
        p = fitted.predict_probabilities(sample, model)
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        assert np.array_equal(fitted.predict_labels(sample, model),
                              (p >= fitted.thresholds[model]).astype(int))
    print("PASS: thresholds come from out-of-fold predictions; outputs are valid")

    # Out-of-fold AUROC must not be inflated the way in-sample AUROC is.
    for model in fitted.estimators:
        y, w = train["y_diag"].astype(int), train["w"]
        in_sample = roc_auc_score(y, fitted.predict_probabilities(train, model), sample_weight=w)
        out_of_fold = roc_auc_score(y, oof[model], sample_weight=w)
        assert out_of_fold <= in_sample + 1e-9
        print(f"  {model}: in-sample AUROC {in_sample:.3f}, out-of-fold {out_of_fold:.3f}")

    # Out-of-fold predictions for a cycle never see that cycle's labels.
    held = sorted(train["cycle"].unique())[0]
    other = df.copy()
    flip = other["period"].eq("train") & other["cycle"].eq(held) & mask
    other.loc[flip, "y_diag"] = 1 - other.loc[flip, "y_diag"]
    refit = train_models(other)
    a = fitted.oof[fitted.oof["cycle"] == held]
    b = refit.oof[refit.oof["cycle"] == held]
    for model in fitted.estimators:
        assert np.allclose(a[model].to_numpy(), b[model].to_numpy(), atol=1e-12)
    print(f"PASS: flipping {held} labels leaves that cycle's out-of-fold predictions unchanged")

    small_train = train.head(250).copy()
    prep = TrainOnlyPreprocessor().fit(small_train)
    stats = prep.imputer.statistics_.copy()
    means = prep.scaler.mean_.copy()
    test = df.loc[mask & df["period"].eq("test_L")].head(20).copy()
    test.loc[:, "pir"] = 1e9
    Xt = prep.transform(test)
    assert Xt.shape == (20, 9)
    assert np.array_equal(prep.imputer.statistics_, stats)
    assert np.array_equal(prep.scaler.mean_, means)
    assert np.array_equal(Xt[:, -1], np.zeros(20))
    test.loc[test.index[:2], "pir"] = np.nan
    Xt2 = prep.transform(test)
    assert Xt2[0, -1] == Xt2[1, -1] == 1
    print("PASS: train-only imputation and scaling, PIR missing flag, 9 features")

    # Test-period data cannot change anything that was fitted.
    other = df.copy()
    test_rows = other["period"] != "train"
    other.loc[test_rows, ["y_diag", "y_bio"]] = 1
    other.loc[test_rows, "age"] = 999
    refit = train_models(other)
    probe = train.head(50)
    for model in fitted.estimators:
        assert np.allclose(fitted.predict_probabilities(probe, model),
                           refit.predict_probabilities(probe, model), atol=1e-12)
        assert fitted.thresholds[model] == refit.thresholds[model]
    print("PASS: changing test-period rows cannot change the fitted models or thresholds")
    print("ALL model tests passed")


if __name__ == "__main__":
    run_tests()
