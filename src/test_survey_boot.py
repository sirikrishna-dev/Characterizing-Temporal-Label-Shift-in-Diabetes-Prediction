"""Run from project root: python -m src.test_survey_boot"""
import numpy as np
import pandas as pd
from src.survey_boot import (rao_wu_replicate_weights, domain_weights,
                             design_degrees_of_freedom, bootstrap_t_ci)

# Two cycles: one stratum with 3 PSUs and one with 2.
df = pd.DataFrame({
    "cycle": ["A", "A", "A", "B", "B"],
    "stratum": ["s1", "s1", "s1", "s2", "s2"],
    "psu": ["p1", "p2", "p3", "p1", "p2"],
    "w": [2., 4., 6., 3., 5.],
})
assert design_degrees_of_freedom(df) == 3
reps = list(rao_wu_replicate_weights(df, n_reps=100, seed=42))
assert len(reps) == 100
for w in reps:
    assert len(w) == len(df)
    # Exactly preserve each stratum's total weight multipliers (not weighted total!).
    multiplier = w / df["w"].to_numpy()
    assert np.isclose(multiplier[:3].sum(), 3)
    assert np.isclose(multiplier[3:].sum(), 2)
assert np.array_equal(domain_weights(reps[0], [True, False, True, False, True])[[1, 3]], [0, 0])
assert np.array_equal(reps[0], next(rao_wu_replicate_weights(df, n_reps=100, seed=42)))
ci = bootstrap_t_ci(0.4, np.array([0.3, 0.5, 0.45, 0.35]), design_df=3)
assert ci["ci_low"] < 0.4 < ci["ci_high"]
assert 0 <= ci["p_value"] <= 1
print("PASS: unequal PSU counts, cycle separation, fixed seed, domain zeroing, CI")

# Actual dataset: construct design on full test_P cycle, then apply domain mask.
full = pd.read_csv("nhanes_out/nhanes_all.csv", low_memory=False)
period = full[full["period"].eq("test_P")].copy()
assert design_degrees_of_freedom(period) == 25
period["adult"] = period["adult"].astype(str).str.lower().eq("true")
domain = (period["adult"] & (period["w"] > 0)
          & period["y_diag"].notna() & period["y_bio"].notna())
assert int(domain.sum()) == 8000
rw = next(rao_wu_replicate_weights(period, n_reps=2, seed=42))
assert len(rw) == len(period)
assert np.count_nonzero(domain_weights(rw, domain)[~domain]) == 0
print("PASS: real NHANES 2017-2020, df=25, adult/label domain=8000")
