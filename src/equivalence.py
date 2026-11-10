"""Turn "no significant change" into "shown to be stable within a margin".

Two one-sided tests (TOST) on every change in the temporal family of
model_temporal.csv. A change is EQUIVALENT (negligible) when its 90% CI lies
entirely inside (-margin, +margin). Margins fixed in advance:
    AUROC        0.03   (smaller than the 0.8 vs 0.77 gap usually treated as
                         a meaningful difference between risk models)
    sensitivity  0.05   (5 percentage points; same tolerance as the benchmark)
For each cell the table also gives the largest change the data cannot rule
out (the 90% CI bound furthest from zero), so a reader sees the precision.

Output: results/equivalence.csv. Run: python -m src.equivalence
"""
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

MARGIN = {"auroc": 0.03, "sensitivity": 0.05}


def main():
    t = pd.read_csv("results/model_temporal.csv", dtype={"level": str})
    t = t[(t["family"] == "temporal") & t["status"].eq("ok")].copy()
    crit = student_t.ppf(0.95, t["design_df"])
    t["ci90_low"] = t["change"] - crit * t["se"]
    t["ci90_high"] = t["change"] + crit * t["se"]
    t["margin"] = t["metric"].map(MARGIN)
    t["equivalent"] = (t["ci90_low"] > -t["margin"]) & (t["ci90_high"] < t["margin"])
    t["worst_plausible_change"] = np.where(t["ci90_low"].abs() > t["ci90_high"].abs(), t["ci90_low"], t["ci90_high"])
    p_lo = student_t.sf((t["change"] + t["margin"]) / t["se"], t["design_df"])
    p_hi = student_t.cdf((t["change"] - t["margin"]) / t["se"], t["design_df"])
    t["p_tost"] = np.maximum(p_lo, p_hi)
    cols = ["period_label", "model", "subgroup", "level", "label_or_gap", "metric", "change", "se",
            "ci90_low", "ci90_high", "margin", "p_tost", "equivalent", "worst_plausible_change"]
    t[cols].to_csv("results/equivalence.csv", index=False)
    s = t.groupby(["metric", "label_or_gap"]).agg(cells=("equivalent", "size"), equivalent=("equivalent", "sum"))
    print(s)
    print(t[t["subgroup"] == "all"][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
