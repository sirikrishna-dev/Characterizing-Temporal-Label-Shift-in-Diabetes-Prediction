"""Figures for the upgrade chapters. Run after the upgrade modules:
    python -m src.make_analysis_plots --out plots
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # fixed categorical order
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "legend.frameon": False})


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=200)
    plt.close(fig)


def pu(res, out):
    e = pd.read_csv(res / "pu_eval.csv", dtype={"level": str})
    methods = [("noaccess", "Drop access features"), ("sar", "SAR-EM (PU)"), ("bio", "Train on y_bio (lab for all)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, kind, title in zip(axes, ("logistic", "lightgbm"), ("Logistic regression", "LightGBM")):
        for i, (stem, lab) in enumerate(methods):
            for j, (per, col) in enumerate([("2017-2020", S1), ("2021-2023", S2)]):
                r = e[(e["method"] == f"{stem}_{kind}|matched") & (e["subgroup"] == "all") & (e["period_label"] == per)
                      & (e["label_or_gap"] == f"diff_vs_diag_{kind}|matched:undx") & (e["metric"] == "sensitivity")].iloc[0]
                y = i + (j - 0.5) * 0.28
                ax.errorbar(r["estimate"] * 100, y, xerr=[[(r["estimate"] - r["ci_low"]) * 100], [(r["ci_high"] - r["estimate"]) * 100]],
                            fmt="o", color=col, ms=7, lw=2, capsize=0, label=per if i == 0 else None)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(range(len(methods)), [m[1] for m in methods])
        ax.set_title(title)
        ax.set_xlabel("Change in sensitivity among undiagnosed cases\nvs the y_diag model, same flag rate (points)")
        ax.invert_yaxis()
    axes[0].legend(loc="lower right")
    save(fig, out, "pu_gain.png")

    a = pd.read_csv(res / "pu_audit_curve.csv")
    a = a[(a["label_or_gap"] == "undx") & (a["metric"] == "sensitivity") & (a["kind"] == "logistic")]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for per, col in [("2011-2016", INK2), ("2017-2020", S1), ("2021-2023", S2)]:
        g = a[(a["period_label"] == per) & a["stem"].isin(["sar", "audit", "bio"])].sort_values("audit_rate")
        ax.plot(g["audit_rate"] * 100, g["estimate"] * 100, "-o", color=col, lw=2, ms=6, label=per)
        base = a[(a["period_label"] == per) & (a["stem"] == "diag")]["estimate"].iloc[0]
        ax.axhline(base * 100, color=col, lw=1, ls=":")
    ax.set_xscale("symlog", linthresh=5)
    ax.set_xticks([0, 5, 10, 20, 50, 100], ["0\n(PU only)", "5", "10", "20", "50", "100\n(y_bio)"])
    ax.set_xlabel("Share of training adults with a lab audit (%)")
    ax.set_ylabel("Undiagnosed sensitivity (%)")
    ax.set_title("Audit-label curve, logistic (dotted: y_diag model)")
    ax.legend()
    save(fig, out, "audit_curve.png")


def explain(res, out):
    d = pd.read_csv(res / "explain_decomposition.csv")
    d = d[(d["period_label"] == "2017-2023 pooled") & (d["subgroup"] == "all") & (d["metric"] == "logit_gap")]
    terms = [("age", "Age"), ("access", "Insurance + usual care"), ("male", "Sex"),
             ("socioeconomic", "Education + income"), ("body_size", "BMI + waist")]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for j, (m, col) in enumerate([("logistic", S1), ("lightgbm", S2)]):
        for i, (t, lab) in enumerate(terms):
            r = d[(d["method"] == m) & (d["label_or_gap"] == t)].iloc[0]
            y = i + (j - 0.5) * 0.3
            ax.errorbar(r["estimate"], y, xerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]],
                        fmt="o", color=col, ms=7, lw=2, label=m.replace("lightgbm", "LightGBM").replace("logistic", "Logistic") if i == 0 else None)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(terms)), [t[1] for t in terms])
    ax.invert_yaxis()
    ax.set_xlabel("Contribution to the score gap, diagnosed minus undiagnosed cases (log-odds)\n> 0: pushes undiagnosed cases below diagnosed ones")
    ax.set_title("Why hidden cases score lower (2017-2023 pooled, 95% CI)")
    ax.legend(loc="lower right")
    save(fig, out, "explain.png")


def benchmark(res, out):
    p = pd.read_csv(res / "benchmark_power.csv")
    mons = [("psi_w", "Feature PSI (weighted)", S1), ("domain_auc", "Domain classifier", S2),
            ("bbse", "BBSE prevalence", S3), ("audit_n4000", "Audit, n = 4000", S4)]
    sc = [("cov_older", "More adults 60+"), ("cov_uninsured", "More uninsured"), ("cov_lowincome", "More low-income"),
          ("prev_shift", "More diabetes (label shift)"), ("undx_growth", "More hidden cases, men 20-59"),
          ("dx_label", "Screening falls, men 20-59"), ("dx_access", "Screening + access fall, men 20-59"),
          ("mixed", "More uninsured + access fall")]
    null = p[p["scenario"] == "null"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), sharex=True, sharey=True)
    for ax, (s, title) in zip(axes.ravel(), sc):
        g = p[p["scenario"] == s]
        harm = g.drop_duplicates("severity")
        for _, r in harm.iterrows():
            if r["harmful"]:
                ax.axvspan(r["severity"] - 0.5, r["severity"] + 0.5, color="#f6e3dc", lw=0, zorder=0)
        for m, lab, col in mons:
            gm = pd.concat([null[null["monitor"] == m], g[g["monitor"] == m]]).sort_values("severity")
            ax.plot(gm["severity"], gm["detection_rate"] * 100, "-o", color=col, lw=2, ms=5, label=lab)
        ax.axhline(5, color=INK2, lw=0.8, ls=":")
        undx = harm.set_index("severity")["change_sens_undx"]
        ax.set_title(f"{title}\nhidden-case sensitivity at severity 5: {undx.get(5, np.nan) * 100:+.1f} pts",
                     fontsize=9.5)
        ax.set_xticks(range(6))
    for ax in axes[1]:
        ax.set_xlabel("Severity (0 = no shift)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Alarm rate (%)")
    axes[0, 0].legend(loc="center right", fontsize=8)
    fig.suptitle("Which monitor catches which shift (200 replicates per point; shaded = harmful: hidden-case sensitivity falls >= 5 pts)",
                 fontsize=11, color=INK)
    save(fig, out, "benchmark_power.png")


def audit(res, out):
    s = pd.read_csv(res / "audit_power_sim.csv")
    cells = [("all=all", "All adults", S1), ("age_band2=20-59", "Ages 20-59", S2), ("sex=male", "Men", S3),
             ("insured=0.0", "Uninsured", S4)]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for c, lab, col in cells:
        g = s[s["cell"] == c].sort_values("audit_n")
        ax.plot(g["audit_n"], g["power"] * 100, "-o", color=col, lw=2, ms=6, label=lab)
    ax.axhline(80, color=INK2, lw=1, ls=":")
    ax.text(500, 82, "80% power", color=INK2, fontsize=8)
    ax.set_xlabel("Lab-tested adults in the audit, per period")
    ax.set_ylabel("Power to detect a 15-pt drop (%)")
    ax.set_title("Audit sample size (simulation, 1000 reps)")
    ax.set_ylim(0, 100)
    ax.legend()
    save(fig, out, "audit_power.png")


def screening(res, out):
    e = pd.read_csv(res / "screening_eval.csv", dtype={"level": str})
    rules = [("uspstf2021", "USPSTF 2021"), ("ada_age35", "ADA: age 35+"), ("ada_soc", "ADA Standards"),
             ("ada_rtest_r", "ADA risk test (reduced)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, per in zip(axes, ("2017-2020", "2021-2023")):
        for i, (r, lab) in enumerate(rules):
            for j, (meth, col, name) in enumerate([(f"rule:{r}", S1, "Rule"), (f"diag_logistic@{r}_rate", S2, "y_diag model, same flag rate"),
                                                   (f"sar_logistic@{r}_rate", S3, "SAR-EM model, same flag rate")]):
                x = e[(e["method"] == meth) & (e["subgroup"] == "all") & (e["period_label"] == per)
                      & (e["label_or_gap"] == "undx") & (e["metric"] == "sensitivity")].iloc[0]
                y = i + (j - 1) * 0.25
                ax.errorbar(x["estimate"] * 100, y, xerr=[[(x["estimate"] - x["ci_low"]) * 100], [(x["ci_high"] - x["estimate"]) * 100]],
                            fmt="o", color=col, ms=7, lw=2, label=name if i == 0 else None)
        ax.set_yticks(range(len(rules)), [r[1] for r in rules])
        ax.invert_yaxis()
        ax.set_title(per)
        ax.set_xlabel("Sensitivity among undiagnosed cases (%)")
    axes[1].legend(loc="lower left", fontsize=8)
    save(fig, out, "screening.png")


def decomposition(res, out):
    d = pd.read_csv(res / "decomposition.csv")
    comps = ["2011-2016 -> 2017-2020", "2011-2016 -> 2021-2023", "2017-2020 -> 2021-2023"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    parts = [("part_more_diabetes", "More diabetes", S1), ("part_more_diagnosis", "More diagnosis", S2)]
    for i, c in enumerate(comps):
        for j, (k, lab, col) in enumerate(parts):
            r = d[(d["method"] == "BxF") & (d["period_label"] == c) & (d["subgroup"] == "all") & (d["label_or_gap"] == k)].iloc[0]
            y = i + (j - 0.5) * 0.3
            axes[0].errorbar(r["estimate"] * 100, y, xerr=[[(r["estimate"] - r["ci_low"]) * 100], [(r["ci_high"] - r["estimate"]) * 100]],
                             fmt="o", color=col, ms=7, lw=2, label=lab if i == 0 else None)
        for j, (k, lab, col) in enumerate([("B:population", "Population mix", S3), ("B:within", "Within population", S4)]):
            r = d[(d["method"] == "reweight") & (d["period_label"] == c) & (d["label_or_gap"] == k)].iloc[0]
            y = i + (j - 0.5) * 0.3
            axes[1].errorbar(r["estimate"] * 100, y, xerr=[[(r["estimate"] - r["ci_low"]) * 100], [(r["ci_high"] - r["estimate"]) * 100]],
                             fmt="o", color=col, ms=7, lw=2, label=lab if i == 0 else None)
    for ax in axes:
        ax.axvline(0, color=INK2, lw=1)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_yticks(range(3), comps)
    axes[0].invert_yaxis()
    axes[0].set_title("Change in diagnosed prevalence")
    axes[0].set_xlabel("Percentage points")
    axes[1].set_title("Change in biomarker prevalence")
    axes[1].set_xlabel("Percentage points")
    save(fig, out, "decomposition.png")


def equivalence(res, out):
    t = pd.read_csv(res / "equivalence.csv", dtype={"level": str})
    t = t[(t["metric"] == "auroc") & (t["label_or_gap"] == "y_bio") & (t["model"] == "logistic")]
    t = t.assign(cell=t["subgroup"].where(t["subgroup"] != "all", "overall") + " " + t["level"].where(t["level"] != "all", ""))
    fig, ax = plt.subplots(figsize=(7, 4.4))
    cells = list(dict.fromkeys(t["cell"]))
    for j, (per, col) in enumerate([("2017-2020", S1), ("2021-2023", S2)]):
        g = t[t["period_label"] == per].set_index("cell").reindex(cells)
        y = np.arange(len(cells)) + (j - 0.5) * 0.3
        ax.errorbar(g["change"], y, xerr=[g["change"] - g["ci90_low"], g["ci90_high"] - g["change"]],
                    fmt="o", color=col, ms=6, lw=2, label=per)
    ax.axvspan(-0.03, 0.03, color="#e8f1fb", zorder=0)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(cells)), cells)
    ax.invert_yaxis()
    ax.set_xlabel("Change in AUROC vs y_bio from 2011-2016 (90% CI; shaded = +/-0.03 equivalence margin)")
    ax.set_title("Stable, not just 'not significant' (logistic)")
    ax.legend(loc="lower right")
    save(fig, out, "equivalence.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="plots")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(exist_ok=True)
    for f in (pu, explain, benchmark, audit, screening, decomposition, equivalence):
        f(res, out)
        print("made", f.__name__)


if __name__ == "__main__":
    main()
