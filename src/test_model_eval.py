"""Run from the project root: python -m src.test_model_eval"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.model_eval import (QUANTITIES, ModelArrays, bh_adjust, domain_list, domain_metrics,
                            evaluate, temporal_change, weighted_metric)
from src.models import analysis_domain


def check_fast_metrics():
    """The matrix version must equal the one-vector reference, ties included."""
    rng = np.random.default_rng(7)
    n, D = 400, 4
    p = np.round(rng.uniform(0, 1, n), 2)          # rounding creates ties
    y_bio = (rng.uniform(0, 1, n) < p).astype(float)
    y_diag = y_bio * (rng.uniform(0, 1, n) < 0.8)
    undx = ((y_bio == 1) & (y_diag == 0)).astype(float)
    labels = {"y_diag": y_diag, "y_bio": y_bio, "undx": undx}
    W = rng.uniform(0.5, 3, (n, D)) * (rng.uniform(0, 1, (n, D)) < 0.7)
    fast = domain_metrics(ModelArrays(p, 0.4, labels), W)
    for label, metric in QUANTITIES:
        for d in range(D):
            ref = weighted_metric(labels[label], p, W[:, d], metric, 0.4)
            assert np.isclose(fast[(label, metric)][d], ref, atol=1e-12), (label, metric, d)
    print("PASS: matrix metrics equal the reference metrics, with tied predictions")


def check_reference_metrics():
    y = np.array([0, 1, 0, 1])
    p = np.array([.1, .9, .2, .8])
    w = np.array([1., 2., 3., 4.])
    assert np.isclose(weighted_metric(y, p, w, "auroc", .5), 1.)
    assert np.isclose(weighted_metric(y, p, w, "sensitivity", .5), 1.)
    assert np.isclose(weighted_metric(y, p, w, "specificity", .5), 1.)
    assert np.isclose(weighted_metric(y, p, w, "brier", .5), np.dot(w, (p - y) ** 2) / w.sum())
    assert np.isclose(weighted_metric(y, p, w, "calibration_in_large", .5), np.dot(w, p - y) / w.sum())
    assert weighted_metric(y, p, w, "ece", .5) >= 0
    assert np.isnan(weighted_metric(y, p, np.zeros(4), "auroc", .5))
    assert np.isnan(weighted_metric(np.zeros(4), p, w, "auroc", .5))
    assert np.isnan(weighted_metric(np.zeros(4), p, w, "sensitivity", .5))
    np.testing.assert_allclose(bh_adjust([.01, .04, np.nan, .03]), [.03, .04, np.nan, .04])
    print("PASS: reference metrics, undefined denominators, BH adjustment")


def check_real_data():
    data = Path("nhanes_out/nhanes_all.csv")
    models = Path("results/rq2_models.joblib")
    if not (data.exists() and models.exists()):
        raise FileNotFoundError("Run from the project root after python -m src.models")
    df = pd.read_csv(data, low_memory=False)
    bundle = joblib.load(models)

    domains = domain_list(df)
    names = {(s, l) for s, l, _ in domains}
    assert ("income_band", "<1.3") not in names and ("income_band2", "<1.3") in names
    assert ("age_band", "60+") not in names and ("age_band2", "60+") in names
    assert ("race_eth", "NH White") not in names and ("race_eth4", "NH White") in names
    assert domains[0] == ("all", "all", "overall")
    print("PASS: duplicate subgroup levels are evaluated once, primary copy kept")

    out = evaluate(df, bundle, n_reps=5, seed=1)
    assert set(out["period_label"]) == {"2011-2016", "2017-2020", "2021-2023"}
    assert out.groupby("period")["design_df"].first().to_dict() == {
        "train_cv": 47, "test_P": 25, "test_L": 15}
    overall = out[(out["subgroup"] == "all") & (out["label_or_gap"] == "y_diag")
                  & (out["metric"] == "auroc") & (out["model"] == "logistic")]
    assert overall.set_index("period")["n_rows"].to_dict() == {
        "train_cv": 15432, "test_P": 8000, "test_L": 5730}

    fam = out[out["family"] == "label_gap"]
    assert set(zip(fam["label_or_gap"], fam["metric"])) == {
        ("gap_diag_minus_bio", "auroc"), ("gap_diag_minus_undx", "sensitivity")}
    assert set(fam["period"]) == {"test_P", "test_L"}
    assert set(fam["role"]) <= {"overall", "primary"}
    assert out.loc[out["family"] != "label_gap", "q_bh"].isna().all()
    assert out.loc[~out["label_or_gap"].str.startswith("gap_"), "p_value"].isna().all()
    print("PASS: label-gap BH family holds only the pre-specified tests")

    # sensitivity on y_bio is the diag/undx mix, weighted by the diagnosed share
    for period in ("test_P", "test_L"):
        o = out[(out["period"] == period) & (out["subgroup"] == "all")
                & (out["metric"] == "sensitivity") & (out["model"] == "logistic")]
        e = o.set_index("label_or_gap")["estimate"]
        full = df[(df["period"] == period)]
        a = full[analysis_domain(full)]
        share = (a["w"] * a["y_diag"]).sum() / (a["w"] * a["y_bio"]).sum()
        assert np.isclose(e["y_bio"], share * e["y_diag"] + (1 - share) * e["undx"], atol=1e-10)
    print("PASS: y_bio sensitivity = diagnosed share x diag + rest x undx")

    temporal = temporal_change(out)
    assert set(temporal["period_label"]) == {"2017-2020", "2021-2023"}
    tf = temporal[temporal["family"] == "temporal"]
    assert set(zip(tf["label_or_gap"], tf["metric"])) == {
        ("y_diag", "auroc"), ("y_bio", "auroc"), ("y_diag", "sensitivity"),
        ("y_bio", "sensitivity"), ("undx", "sensitivity")}
    row = temporal[(temporal["subgroup"] == "all") & (temporal["model"] == "logistic")
                   & (temporal["label_or_gap"] == "y_bio") & (temporal["metric"] == "auroc")
                   & (temporal["period"] == "test_P")].iloc[0]
    assert np.isclose(row["change"], row["test"] - row["baseline"])
    assert np.isclose(row["se"], np.hypot(row["baseline_se"], row["test_se"]))
    assert row["design_df"] == 25
    print("PASS: temporal change = test - baseline, independent-sample SE, smaller df")
    print("ALL evaluation tests passed")


if __name__ == "__main__":
    check_reference_metrics()
    check_fast_metrics()
    check_real_data()
