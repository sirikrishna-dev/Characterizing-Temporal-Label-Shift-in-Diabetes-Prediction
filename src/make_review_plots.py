"""
make_review_plots.py

Builds the two result plots for the Jackfruit review slides from the team's
real outputs. Nothing in here is hard-coded data: every number is read from
the files below.

  plot_prevalence_by_cycle.png
      from nhanes_out/step5_weighted_prevalence.csv (data notebook v7)
      diagnosed vs biomarker-defined diabetes per cycle, survey-weighted, 95% CI

  plot_screening_by_cycle.png
      from the same file: share of undiagnosed adults tested in the past 3 years

  plot_drift_weighted_vs_unweighted.png
      from results/drift_features.csv (src/drift.py)
      PSI per feature, 2011-16 vs 2021-23, unweighted next to weighted

Run from the repo root:
  python -m src.make_review_plots --prevalence nhanes_out/step5_weighted_prevalence.csv \
                                  --drift results/drift_features.csv --out plots
Either input can be left out; that plot is then skipped.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE = "#2e5bba"     # slide accent blue
ORANGE = "#d97a1e"   # second series, passes the colour-blind check against BLUE
INK = "#1f2a33"
MUTED = "#5b6670"
GRID = "#e3e6ea"

CYCLE_ORDER = ["2011-2012", "2013-2014", "2015-2016", "2017-2020", "2021-2023"]
CYCLE_LABEL = {"2011-2012": "2011-12", "2013-2014": "2013-14", "2015-2016": "2015-16",
               "2017-2020": "2017-Mar 2020", "2021-2023": "2021-23"}

FEATURE_LABEL = {"age": "Age", "pir": "Income-to-poverty ratio", "bmi": "BMI",
                 "waist": "Waist circumference", "sex": "Sex", "race_eth": "Race/ethnicity",
                 "education": "Education", "insured": "Insured", "routine_care": "Usual source of care"}

# Measure names written by the data notebook (v7, step5_weighted_prevalence.csv)
DIAGNOSED = "Diagnosed, lab sample"
BIOMARKER = "Total, HbA1c (y_bio, PRIMARY)"
SCREENED = "Screened in 3y, among undiagnosed (interview weight, MAIN)"

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 16,
    "xtick.labelsize": 15, "ytick.labelsize": 15,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


def plot_prevalence(path: str, out_dir: str) -> str:
    df = pd.read_csv(path)
    missing = {DIAGNOSED, BIOMARKER} - set(df["measure"])
    if missing:
        raise ValueError(f"{path} has no rows for {sorted(missing)}. Found: {sorted(df['measure'].unique())}")
    cycles = [c for c in CYCLE_ORDER if c in set(df["cycle"])]
    x = np.arange(len(cycles))

    fig, ax = plt.subplots(figsize=(12, 8.4), dpi=200)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)

    series = [(BIOMARKER, ORANGE, "Biomarker-defined (diagnosed or HbA1c ≥ 6.5%)"),
              (DIAGNOSED, BLUE, "Diagnosed by a doctor")]
    ends = {}
    for measure, colour, label in series:
        s = df[df["measure"] == measure].set_index("cycle").reindex(cycles)
        y = s["pct"].to_numpy()
        err = np.vstack([y - s["ci_low"].to_numpy(), s["ci_high"].to_numpy() - y])
        ax.errorbar(x, y, yerr=err, color=colour, linewidth=2.5, marker="o", markersize=9,
                    capsize=6, elinewidth=1.8, label=label, zorder=3)
        ends[measure] = y

    # shade the gap = undiagnosed share
    ax.fill_between(x, ends[DIAGNOSED], ends[BIOMARKER], color=ORANGE, alpha=0.10, zorder=1)
    gap_last = ends[BIOMARKER][-1] - ends[DIAGNOSED][-1]
    ax.annotate(f"gap = undiagnosed\n{gap_last:.1f} pts in {CYCLE_LABEL[cycles[-1]]}",
                xy=(x[-1], (ends[BIOMARKER][-1] + ends[DIAGNOSED][-1]) / 2),
                xytext=(14, 0), textcoords="offset points", va="center", fontsize=14, color=MUTED)

    ax.set_xticks(x, [CYCLE_LABEL[c] for c in cycles])
    ax.set_xlim(-0.3, len(cycles) - 0.3 + 0.9)
    ax.set_ylabel("Adults 20+ (%), survey-weighted")
    ax.set_title("Diagnosed vs biomarker-defined diabetes by NHANES cycle", loc="left", pad=16)
    ax.legend(frameon=False, loc="upper left", fontsize=14)
    ax.text(0, -0.13, "Error bars: 95% CI from the survey design (strata and PSUs).",
            transform=ax.transAxes, fontsize=12, color=MUTED)
    fig.tight_layout()
    out = os.path.join(out_dir, "plot_prevalence_by_cycle.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def plot_screening(path: str, out_dir: str) -> str:
    df = pd.read_csv(path)
    s = df[df["measure"] == SCREENED]
    if s.empty:
        raise ValueError(f"{path} has no rows for '{SCREENED}'.")
    cycles = [c for c in CYCLE_ORDER if c in set(s["cycle"])]
    s = s.set_index("cycle").reindex(cycles)
    x = np.arange(len(cycles))
    y = s["pct"].to_numpy()
    err = np.vstack([y - s["ci_low"].to_numpy(), s["ci_high"].to_numpy() - y])

    fig, ax = plt.subplots(figsize=(12, 8.4), dpi=200)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.errorbar(x, y, yerr=err, color=BLUE, linewidth=2.5, marker="o", markersize=9,
                capsize=6, elinewidth=1.8, zorder=3)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.1f}%", (xi, yi), xytext=(12, 0), textcoords="offset points",
                    va="center", fontsize=14, color=INK)
    ax.set_xticks(x, [CYCLE_LABEL[c] for c in cycles])
    ax.set_xlim(-0.3, len(cycles) - 0.4)
    ax.set_ylabel("Undiagnosed adults 20+ tested in past 3 years (%)")
    ax.set_title("Diabetes screening among undiagnosed adults (DIQ180)", loc="left", pad=16)
    ax.text(0, -0.13, "Survey-weighted (interview weight), 95% CI with design degrees of freedom.",
            transform=ax.transAxes, fontsize=12, color=MUTED)
    fig.tight_layout()
    out = os.path.join(out_dir, "plot_screening_by_cycle.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def plot_drift(path: str, out_dir: str, period: str = "2021-2023") -> str:
    df = pd.read_csv(path)
    d = df[(df["period"] == period) & (df["subgroup"] == "all") &
           (df["metric"] == "psi") & (df["status"] == "ok")].copy()
    if d.empty:
        raise ValueError(f"No overall PSI rows for period {period} in {path}.")
    d = d.sort_values("unweighted", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))
    h = 0.38

    fig, ax = plt.subplots(figsize=(12, 8.4), dpi=200)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.barh(y + h / 2 + 0.01, d["unweighted"], height=h, color=ORANGE, label="Unweighted (raw rows)")
    ax.barh(y - h / 2 - 0.01, d["weighted"], height=h, color=BLUE, label="Weighted (CDC survey weights)")

    xmax = max(d["unweighted"].max(), d["weighted"].max())
    for i, r in d.iterrows():
        if bool(r.get("sampling_artifact", False)):
            tag, colour = "sampling artifact", ORANGE
        elif bool(r.get("real_shift", False)):
            tag, colour = "real shift", BLUE
        else:
            continue
        ax.text(max(r["unweighted"], r["weighted"]) + xmax * 0.02, i, tag,
                va="center", fontsize=13, color=colour, fontweight="bold")

    ax.set_yticks(y, [FEATURE_LABEL.get(f, f) for f in d["feature"]])
    ax.set_xlim(0, xmax * 1.35)
    ax.set_xlabel("Population Stability Index (higher = more drift)")
    ax.set_title(f"Feature drift, 2011-16 vs {period}: raw vs survey-weighted", loc="left", pad=16)
    ax.legend(frameon=False, loc="lower right", fontsize=14)
    ax.text(0, -0.2, "A shift is called real only when the weighted PSI beats its bootstrap noise floor "
            "(Bonferroni across features).", transform=ax.transAxes, fontsize=12, color=MUTED)
    fig.tight_layout()
    out = os.path.join(out_dir, "plot_drift_weighted_vs_unweighted.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prevalence", help="nhanes_out/step5_weighted_prevalence.csv")
    ap.add_argument("--drift", help="results/drift_features.csv")
    ap.add_argument("--period", default="2021-2023", help="test period for the drift plot")
    ap.add_argument("--out", default="plots")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.prevalence:
        print("wrote", plot_prevalence(args.prevalence, args.out))
        print("wrote", plot_screening(args.prevalence, args.out))
    if args.drift:
        print("wrote", plot_drift(args.drift, args.out, args.period))
    if not (args.prevalence or args.drift):
        ap.error("give --prevalence and/or --drift")


if __name__ == "__main__":
    main()
