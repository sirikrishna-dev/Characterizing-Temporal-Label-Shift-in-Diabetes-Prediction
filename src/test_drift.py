"""Run from the project root: python -m src.test_drift"""
from pathlib import Path

import numpy as np
import pandas as pd

from src import survey_drift as sd


def synthetic():
    df = pd.DataFrame({"cycle": ["2011-2012", "2013-2014", "2015-2016"], "w": [3.0, 3.0, 3.0]})
    assert np.allclose(sd.pool_weights(df)["w"], 1.0)
    mixed = pd.DataFrame({"cycle": ["2015-2016", "2017-2020"], "w": [5.2, 5.2]})
    assert np.allclose(sd.pool_weights(mixed)["w"], [2.0, 3.2])
    print("PASS: pooling multiplies each cycle by its years / total years")

    rng = np.random.default_rng(3)
    x = rng.normal(size=2000)
    w = rng.uniform(1, 3, size=2000)
    assert sd.weighted_psi(x, w, x, w) < 1e-9
    assert sd.weighted_ks(x, w, x, w) < 1e-9
    shifted = x + 1.0
    assert sd.weighted_psi(x, w, shifted, w) > 0.25
    c = rng.integers(0, 4, size=2000)
    assert sd.weighted_categorical_psi(c, w, c, w) < 1e-9
    print("PASS: PSI and KS are 0 for identical samples and large for a clear shift")

    # Weighting matters: same rows, different weights describe different populations.
    group = (rng.uniform(size=2000) < 0.5).astype(int)
    w_ref = np.where(group == 1, 1.0, 1.0)
    w_new = np.where(group == 1, 4.0, 1.0)
    unweighted = sd.weighted_categorical_psi(group, np.ones(2000), group, np.ones(2000))
    weighted = sd.weighted_categorical_psi(group, w_ref, group, w_new)
    assert unweighted < 1e-9 < weighted
    print("PASS: a pure reweighting is invisible unweighted and visible weighted")


def real_data():
    file = Path("nhanes_out/nhanes_all.csv")
    if not file.exists():
        raise FileNotFoundError("Run from the project root with nhanes_out/nhanes_all.csv present")
    df = pd.read_csv(file, low_memory=False)
    # RQ2 trains with column w; RQ3 pools w_lab itself. They must agree on 2011-2016.
    train = df[df["period"] == "train"].copy()
    train["w"] = pd.to_numeric(train["w_lab"], errors="coerce")
    pooled = sd.pool_weights(train)["w"].fillna(0).to_numpy()
    assert np.allclose(pooled, pd.to_numeric(df.loc[df["period"] == "train", "w"],
                                             errors="coerce").fillna(0).to_numpy())
    print("PASS: drift.py's pooled 2011-2016 weight equals the weight the models train with")

    alarms = Path("results/drift_alarms.csv")
    evals = Path("results/model_temporal.csv")
    if alarms.exists() and evals.exists():
        a = pd.read_csv(alarms, dtype={"level": str})
        t = pd.read_csv(evals, dtype={"level": str})
        t = t[t["role"].isin(["overall", "primary"])]
        a_keys = set(zip(a["period"], a["subgroup"], a["level"]))
        t_keys = set(zip(t["period_label"], t["subgroup"], t["level"]))
        assert t_keys <= a_keys, sorted(t_keys - a_keys)
        print("PASS: every model cell has a matching drift cell (monitor.py can join them)")
    print("ALL drift tests passed")


if __name__ == "__main__":
    synthetic()
    real_data()
