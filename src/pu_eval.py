"""Score every PU variant on the shared harness.

Run from the project root, after python -m src.pu_models:
    python -m src.pu_eval --reps 500

Outputs
    results/pu_eval.csv         shared schema; method = "<name>|matched" or "<name>|own"
    results/pu_audit_curve.csv  undiagnosed sensitivity vs audit rate (mean and
                                spread over the random audit draws)
    results/pu_summary.csv      one compact row per method x period (overall cell)

Confirmatory BH family "pu_gain" (fixed before looking at results): the change
in sensitivity among undiagnosed cases versus the y_diag model of the same
kind (logistic vs logistic, LightGBM vs LightGBM) at the matched flag rate,
for noaccess, sar and bio, overall and the four primary subgroups, both test
periods. Elkan-Noto is a sanity check (difference must be exactly 0) and the
audit variants are a dose-response curve; neither is in the family.

Second family "pu_beyond_ablation" (added after the internal review, so it is
secondary, not pre-registered): SAR-EM minus the noaccess model of the same
kind, sensitivity among undiagnosed cases at the matched flag rate, overall
and primary subgroups, both test periods. noaccess drops the insurance and
usual-care features but is still trained on y_diag. This family asks whether
SAR-EM does anything beyond leaving the access features out of the classifier.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.harness import (PERIODS, TEST_PERIODS, load, period_frame, score_methods,
                         reference_flag_share, threshold_for_share, add_family, SEED)
from src.models import analysis_domain

MAIN = ("diag", "noaccess", "en", "sar", "bio")
FAMILY_METHODS = ("noaccess", "sar", "bio")


def period_scores(full, bundle, period, methods):
    analysis = analysis_domain(full).to_numpy()
    out = {}
    if period == "train":
        o = bundle.oof.set_index("SEQN")
        for m in methods:
            p = np.full(len(full), np.nan)
            p[analysis] = full.loc[analysis, "SEQN"].map(o[m]).to_numpy(float)
            out[m] = p
    else:
        for m in methods:
            p = np.full(len(full), np.nan)
            p[analysis] = bundle.predict(full.loc[analysis], m)
            out[m] = p
    return out


def evaluate(df, bundle, base_bundle, n_reps, seed=SEED):
    methods = [c for c in bundle.oof.columns if c not in ("SEQN", "cycle")]
    share = reference_flag_share(base_bundle, df)
    train_rows = df.loc[(df["period"] == "train") & analysis_domain(df)]
    tw = train_rows["w"].to_numpy(float)
    o = bundle.oof.set_index("SEQN")
    matched = {m: threshold_for_share(train_rows["SEQN"].map(o[m]).to_numpy(float), tw, share)
               for m in methods}
    frames = []
    for i, (period, label) in enumerate(PERIODS.items()):
        full = period_frame(df, period)
        sc = period_scores(full, bundle, period, methods)
        scores, thr = {}, {}
        for m in methods:
            scores[f"{m}|matched"] = sc[m]
            thr[f"{m}|matched"] = matched[m]
            if m.split("_")[0] in MAIN:
                scores[f"{m}|own"] = sc[m]
                thr[f"{m}|own"] = bundle.own_threshold[m]
        compare = []
        for m in methods:
            kind = m.split("_")[-1]
            if not m.startswith("diag_"):
                compare.append((f"{m}|matched", f"diag_{kind}|matched"))
            if m.startswith("sar_") and f"noaccess_{kind}" in methods:
                compare.append((f"{m}|matched", f"noaccess_{kind}|matched"))
        res = score_methods(full, scores, thr, label, n_reps=n_reps, seed=seed + i,
                            compare=compare)
        frames.append(res)
        print(f"  scored {label}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    fam = (out["period_label"].isin([PERIODS[p] for p in TEST_PERIODS])
           & out["role"].isin(["overall", "primary"])
           & out["metric"].eq("sensitivity")
           & out["label_or_gap"].str.match(r"diff_vs_diag_(logistic|lightgbm)\|matched:undx$")
           & out["method"].str.split("_").str[0].isin(FAMILY_METHODS))
    out = add_family(out, "pu_gain", fam)
    abl = (out["period_label"].isin([PERIODS[p] for p in TEST_PERIODS])
           & out["role"].isin(["overall", "primary"])
           & out["metric"].eq("sensitivity")
           & out["label_or_gap"].str.match(r"diff_vs_noaccess_(logistic|lightgbm)\|matched:undx$")
           & out["method"].str.startswith("sar_"))
    out = add_family(out, "pu_beyond_ablation", abl)
    return out, share, matched


def audit_curve(res):
    r = res[(res["subgroup"] == "all") & res["method"].str.endswith("|matched")]
    r = r[r["label_or_gap"].isin(["undx", "y_diag", "y_bio", "gap_diag_minus_undx"])]
    r = r[r["metric"].isin(["sensitivity", "auroc"])]
    rows = []
    for (period, method, lab, met), g in r.groupby(["period_label", "method", "label_or_gap", "metric"]):
        name = method.split("|")[0]
        kind = name.split("_")[-1]
        stem = name.split("_")[0]
        if stem.startswith("audit"):
            rate = int(stem[5:7]) / 100
        elif stem == "sar":
            rate = 0.0
        elif stem == "bio":
            rate = 1.0
        elif stem == "diag":
            rate = np.nan
        else:
            continue
        rows.append(dict(period_label=period, kind=kind, stem=stem if not stem.startswith("audit") else "audit",
                         audit_rate=rate, label_or_gap=lab, metric=met,
                         estimate=g["estimate"].iloc[0], se=g["se"].iloc[0]))
    a = pd.DataFrame(rows)
    return (a.groupby(["period_label", "kind", "stem", "audit_rate", "label_or_gap", "metric"], dropna=False)
             .agg(estimate=("estimate", "mean"), draw_sd=("estimate", "std"),
                  mean_se=("se", "mean"), draws=("estimate", "size")).reset_index())


def summary(res):
    r = res[(res["subgroup"] == "all")]
    keep = r["label_or_gap"].isin(["undx", "y_diag", "y_bio", "gap_diag_minus_undx", "flagged"])
    r = r[keep & ~r["method"].str.startswith("audit")]
    r = r.assign(col=r["label_or_gap"] + ":" + r["metric"])
    return r.pivot_table(index=["period_label", "method"], columns="col", values="estimate").reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="nhanes_out/nhanes_all.csv")
    ap.add_argument("--pu", default="results/pu_models.joblib")
    ap.add_argument("--base", default="results/rq2_models.joblib")
    ap.add_argument("--out", default="results")
    ap.add_argument("--reps", type=int, default=500)
    args = ap.parse_args()
    df = load(args.csv)
    bundle = joblib.load(args.pu)
    base = joblib.load(args.base)
    res, share, matched = evaluate(df, bundle, base, args.reps)
    out = Path(args.out)
    res.to_csv(out / "pu_eval.csv", index=False)
    audit_curve(res).to_csv(out / "pu_audit_curve.csv", index=False)
    summary(res).to_csv(out / "pu_summary.csv", index=False)
    print(f"matched flag share = {share:.4f}")
    fam = res[res["family"] == "pu_gain"]
    print(f"pu_gain family: {len(fam)} tests, {int((fam['q_bh'] < 0.05).sum())} with q < 0.05")
    view = fam[fam["subgroup"] == "all"][["period_label", "method", "estimate", "ci_low", "ci_high", "q_bh"]]
    print(view.round(3).to_string(index=False))
    abl = res[res["family"] == "pu_beyond_ablation"]
    print(f"pu_beyond_ablation family: {len(abl)} tests, {int((abl['q_bh'] < 0.05).sum())} with q < 0.05")
    view = abl[abl["subgroup"] == "all"][["period_label", "method", "estimate", "ci_low", "ci_high", "q_bh"]]
    print(view.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
