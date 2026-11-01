"""RQ3 monitoring 2x2: did the survey-weighted drift monitors alarm where the
model actually got worse?

Run from the project root, after src.drift and src.model_eval:
    python -m src.monitor

Inputs (all in --results)
    drift_alarms.csv     from drift.py: per period x cell, which monitors alarmed
    model_temporal.csv   from model_eval.py: change in performance from the
                         2011-2016 out-of-fold baseline
    model_eval.csv       from model_eval.py: paired label gaps per period

A cell is one period x subgroup level (overall plus the primary subgroups),
crossed with each model.

    degraded_bio   performance against the biomarker reference fell
                   significantly: a drop in y_bio AUROC, y_bio sensitivity or
                   sensitivity among undiagnosed cases with temporal-family
                   BH q < 0.05.
    degraded_diag  the same, judged only on y_diag (what a team that only has
                   diagnosis labels would see).
    alarm          any drift monitor in that cell alarmed (feature, domain or
                   y_diag label shift), from drift_alarms.csv.

    outcome        caught       alarm and degraded_bio
                   missed       no alarm, degraded_bio
                   false_alarm  alarm, no degradation
                   quiet        neither

The *_ci columns repeat the outcome with an unadjusted rule (95% CI of the
change entirely below zero) as a sensitivity analysis.

Outputs
    monitoring_2x2.csv      one row per period x cell x model
    monitoring_summary.csv  counts of each outcome per period x model
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["period_label", "subgroup", "level"]
BIO_CHECKS = [("y_bio", "auroc"), ("y_bio", "sensitivity"), ("undx", "sensitivity")]
DIAG_CHECKS = [("y_diag", "auroc"), ("y_diag", "sensitivity")]
OUTCOMES = ["caught", "missed", "false_alarm", "quiet"]


def outcome(alarm, degraded):
    if pd.isna(degraded):
        return "not_reportable"
    if alarm and degraded:
        return "caught"
    if degraded:
        return "missed"
    if alarm:
        return "false_alarm"
    return "quiet"


def read(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src.drift and src.model_eval first.")
    return pd.read_csv(path, dtype={"level": str, "subgroup": str})


def performance_table(temporal, gaps):
    """One row per period x cell x model with the changes and degradation flags."""
    t = temporal[temporal["family"] == "temporal"].copy()
    t["check"] = t["label_or_gap"] + "_" + t["metric"]
    t["drop_q"] = t["bh_significant_005"].astype(bool) & (t["change"] < 0)
    t["drop_ci"] = t["status"].eq("ok") & (t["ci_high"] < 0)
    idx = KEYS + ["model"]
    wide = t.pivot_table(index=idx, columns="check", values="change").add_prefix("change_")
    q = t.pivot_table(index=idx, columns="check", values="q_bh").add_prefix("q_")
    flags = t.pivot_table(index=idx, columns="check", values=["drop_q", "drop_ci"],
                          aggfunc="max")
    perf = wide.join(q)

    def any_drop(kind, checks):
        cols = [(kind, f"{label}_{metric}") for label, metric in checks]
        present = [c for c in cols if c in flags.columns]
        return flags[present].astype(bool).any(axis=1) if present else pd.Series(False, index=flags.index)

    perf["degraded_bio"] = any_drop("drop_q", BIO_CHECKS)
    perf["degraded_diag"] = any_drop("drop_q", DIAG_CHECKS)
    perf["degraded_bio_ci"] = any_drop("drop_ci", BIO_CHECKS)
    perf["degraded_diag_ci"] = any_drop("drop_ci", DIAG_CHECKS)
    perf = perf.reset_index()
    # Cells too small to test stay in the table with degraded_* left empty.
    cells = temporal.loc[temporal["role"].isin(["overall", "primary"]), idx].drop_duplicates()
    perf = cells.merge(perf, on=idx, how="left")

    g = gaps[(gaps["label_or_gap"] == "gap_diag_minus_undx") & (gaps["metric"] == "sensitivity")]
    g = g[KEYS + ["model", "estimate", "ci_low", "ci_high", "q_bh", "status"]].rename(columns={
        "estimate": "undx_gap", "ci_low": "undx_gap_ci_low", "ci_high": "undx_gap_ci_high",
        "q_bh": "undx_gap_q", "status": "undx_gap_status"})
    return perf.merge(g, on=KEYS + ["model"], how="left")


def main():
    parser = argparse.ArgumentParser(description="Join drift alarms with model degradation")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    res = Path(args.results)

    alarms = read(res / "drift_alarms.csv").rename(columns={"period": "period_label"})
    temporal = read(res / "model_temporal.csv")
    gaps = read(res / "model_eval.csv")

    perf = performance_table(temporal, gaps)
    alarm_cols = ["feature_alarms", "feature_alarm", "domain_alarm", "label_alarm",
                  "any_alarm", "overall_monitor_alarm", "y_bio_prevalence_change", "y_bio_shift"]
    table = perf.merge(alarms[KEYS + alarm_cols], on=KEYS, how="left")
    missing = table["any_alarm"].isna()
    if missing.any():
        cells = table.loc[missing, KEYS].drop_duplicates().to_string(index=False)
        raise ValueError(f"No drift result for these cells (re-run src.drift):\n{cells}")
    for c in ["any_alarm", "overall_monitor_alarm"]:
        table[c] = table[c].astype(str).str.lower().eq("true")

    table["outcome"] = [outcome(a, d) for a, d in zip(table["any_alarm"], table["degraded_bio"])]
    table["outcome_ci"] = [outcome(a, d) for a, d in zip(table["any_alarm"], table["degraded_bio_ci"])]
    table["outcome_overall_monitor"] = [outcome(a, d) for a, d in
                                        zip(table["overall_monitor_alarm"], table["degraded_bio"])]
    table["diag_view_disagrees"] = table["degraded_bio"] != table["degraded_diag"]
    table = table.sort_values(["period_label", "model", "subgroup", "level"]).reset_index(drop=True)

    summary = []
    for (period, model), g in table.groupby(["period_label", "model"]):
        for col in ["outcome", "outcome_ci", "outcome_overall_monitor"]:
            counts = g[col].value_counts()
            summary.append({"period_label": period, "model": model, "rule": col,
                            **{o: int(counts.get(o, 0)) for o in OUTCOMES},
                            "cells": len(g)})
    summary = pd.DataFrame(summary)

    table.to_csv(res / "monitoring_2x2.csv", index=False)
    summary.to_csv(res / "monitoring_summary.csv", index=False)
    print(f"Saved {len(table)} rows to {res / 'monitoring_2x2.csv'}")
    print(f"Saved {len(summary)} rows to {res / 'monitoring_summary.csv'}\n")
    print(summary[summary["rule"] == "outcome"].to_string(index=False))
    show = ["period_label", "model", "subgroup", "level", "any_alarm", "degraded_bio",
            "degraded_bio_ci", "outcome", "undx_gap"]
    print("\n" + table[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
