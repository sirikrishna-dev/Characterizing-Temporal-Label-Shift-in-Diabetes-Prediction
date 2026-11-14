"""Turn "no significant change" into "shown to be stable within a margin".

Two one-sided tests (TOST) on every change in the temporal family of
model_temporal.csv. A change is EQUIVALENT (negligible) when its 90% CI lies
entirely inside (-margin, +margin). Main margins:
    AUROC        0.03   a judgment call: about four times the bootstrap SE
                        of the overall AUROC in one period (0.006-0.009) and
                        small next to the 0.79-0.80 level, so a change inside
                        it would not change how the model is used
    sensitivity  0.05   5 percentage points, the same harm tolerance the
                        benchmark uses (HARM_TOL in benchmark.py)
The margins were chosen by the team, not taken from a guideline, so the
conclusion is also reported at a stricter and a looser margin
(results/equivalence_margins.csv): 0.02 / 0.05 for AUROC and 0.03 / 0.10 for
sensitivity. A conclusion that holds only at the looser margin is weak.
For each cell the table also gives the largest change the data cannot rule
out (the 90% CI bound furthest from zero), so a reader sees the precision.

Outputs: results/equivalence.csv, results/equivalence_margins.csv.
Run: python -m src.equivalence
"""
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

MARGIN = {"auroc": 0.03, "sensitivity": 0.05}
MARGIN_GRID = {"auroc": (0.02, 0.03, 0.05), "sensitivity": (0.03, 0.05, 0.10)}


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
    grid = []
    for metric, margins in MARGIN_GRID.items():
        for m in margins:
            g = t[t["metric"] == metric]
            eq = (g["ci90_low"] > -m) & (g["ci90_high"] < m)
            for (lab, overall), gg in g.assign(eq=eq, overall=g["subgroup"].eq("all")).groupby(["label_or_gap", "overall"]):
                grid.append(dict(metric=metric, label_or_gap=lab, margin=m,
                                 cells="overall" if overall else "primary subgroups",
                                 tested=len(gg), equivalent=int(gg["eq"].sum()),
                                 main_margin=bool(np.isclose(m, MARGIN[metric]))))
    grid = pd.DataFrame(grid)
    grid.to_csv("results/equivalence_margins.csv", index=False)
    print(grid.to_string(index=False))
    print(t[t["subgroup"] == "all"][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
