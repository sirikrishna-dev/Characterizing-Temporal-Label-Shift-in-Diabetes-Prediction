"""Upgrade 3: do official screening rules miss the same people the model misses?

Rules (scored on the same adults, labels, weights and replicates as the model)
    uspstf2021      USPSTF 2021: age 35-70 and BMI >= 25 (>= 23 for NH Asian
                    adults, as the recommendation notes).
    ada_age35       ADA Standards of Care: test every adult aged 35 or older.
    ada_soc         ADA Standards of Care, fuller version: age >= 35, OR
                    BMI >= 25 (>= 23 NH Asian) with at least one risk factor
                    we can see: high-risk race/ethnicity (NH Black, Hispanic,
                    NH Asian) or measured BP >= 130/80. Family history,
                    inactivity, lipids, PCOS, GDM and CVD history are not in
                    the project file, so this rule undercounts.
    ada_rtest_r     ADA Diabetes Risk Test (Bang et al. 2009), REDUCED: the
                    items we have are age (0-3), male (1), high blood pressure
                    (measured >= 140/90, 1) and BMI (0-3). Family history,
                    gestational diabetes and physical activity are not in the
                    file (they need the MCQ, RHQ and PAQ files). Positive at the
                    official cut-off of 5 or more; because three items (up to 3
                    points) are missing, this flags fewer people than the full
                    test. ada_rtest_r4 uses a cut-off of 4 as a sensitivity check.
BP was auscultatory in 2011-2016 and oscillometric from 2017; it only enters
the two ADA rules that use measured BP.

Comparison: each model is thresholded so that it flags the same weighted share
of 2011-2016 adults as the rule does (out-of-fold predictions), then both are
scored in every period. Paired differences use the same replicates.

Confirmatory BH family "rule_vs_model" (fixed in advance): difference in
sensitivity among undiagnosed cases, model minus rule, at the rule's flag rate,
for the y_diag logistic and LightGBM models vs each of the four main rules,
overall and primary cells, both test periods.

Outputs
    results/screening_eval.csv     shared schema
    results/screening_overlap.csv  among undiagnosed cases, who is missed by
                                   the rule, the model, both or neither

Run: python -m src.screening_rules --reps 500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.harness import (PERIODS, TEST_PERIODS, period_frame, load, score_methods,
                         threshold_for_share, weighted_share, add_family, label_arrays, SEED)
from src.models import analysis_domain

HIGH_RISK_RACE = {"NH Black", "Mexican American", "Other Hispanic", "NH Asian"}
MAIN_RULES = ("uspstf2021", "ada_age35", "ada_soc", "ada_rtest_r")


def _num(df, c):
    return pd.to_numeric(df[c], errors="coerce")


def bmi_cut(df):
    return np.where(df["race_eth"].astype(str).eq("NH Asian"), 23.0, 25.0)


def ada_risk_score_reduced(df):
    age, bmi = _num(df, "age"), _num(df, "bmi")
    sbp, dbp = _num(df, "sbp"), _num(df, "dbp")
    s = np.select([age >= 60, age >= 50, age >= 40], [3, 2, 1], 0)
    s = s + df["sex"].astype(str).str.lower().eq("male").to_numpy().astype(int)
    s = s + ((sbp >= 140) | (dbp >= 90)).fillna(False).to_numpy().astype(int)
    s = s + np.select([bmi >= 40, bmi >= 30, bmi >= 25], [3, 2, 1], 0)
    return s.astype(float)


def rule_scores(df):
    """Dict rule -> 0/1 (or score) array. Missing BMI/BP count as not positive."""
    age, bmi = _num(df, "age"), _num(df, "bmi")
    sbp, dbp = _num(df, "sbp"), _num(df, "dbp")
    over = (bmi >= bmi_cut(df)).fillna(False).to_numpy()
    a35 = (age >= 35).to_numpy()
    usp = ((age >= 35) & (age <= 70)).to_numpy() & over
    risk = df["race_eth"].astype(str).isin(HIGH_RISK_RACE).to_numpy() | \
        ((sbp >= 130) | (dbp >= 80)).fillna(False).to_numpy()
    soc = a35 | (over & risk)
    score = ada_risk_score_reduced(df)
    return {"uspstf2021": usp.astype(float), "ada_age35": a35.astype(float),
            "ada_soc": soc.astype(float), "ada_rtest_r": score, "ada_rtest_r4": score}


RULE_THRESHOLD = {"uspstf2021": 0.5, "ada_age35": 0.5, "ada_soc": 0.5,
                  "ada_rtest_r": 5.0, "ada_rtest_r4": 4.0}


def model_scores(full, period, base, pu):
    an = analysis_domain(full).to_numpy()
    out = {}
    for name in ("logistic", "lightgbm"):
        p = np.full(len(full), np.nan)
        if period == "train":
            p[an] = full.loc[an, "SEQN"].map(base.oof.set_index("SEQN")[name]).to_numpy(float)
        else:
            p[an] = base.predict_probabilities(full.loc[an], name)
        out[f"diag_{name}"] = p
    if pu is not None:
        p = np.full(len(full), np.nan)
        if period == "train":
            p[an] = full.loc[an, "SEQN"].map(pu.oof.set_index("SEQN")["sar_logistic"]).to_numpy(float)
        else:
            p[an] = pu.predict(full.loc[an], "sar_logistic")
        out["sar_logistic"] = p
    return out


def evaluate(df, base, pu, n_reps, seed=SEED):
    train = period_frame(df, "train")
    an = analysis_domain(train).to_numpy()
    tw = train["w"].to_numpy(float)[an]
    rules_train = rule_scores(train)
    models_train = model_scores(train, "train", base, pu)
    rule_share = {r: weighted_share(rules_train[r][an] >= RULE_THRESHOLD[r], tw) for r in rules_train}
    model_thr = {(m, r): threshold_for_share(models_train[m][an], tw, rule_share[r])
                 for m in models_train for r in rules_train}
    frames, overlap = [], []
    for i, (period, label) in enumerate(PERIODS.items()):
        full = train if period == "train" else period_frame(df, period)
        rs = rule_scores(full)
        ms = model_scores(full, period, base, pu)
        scores, thr, compare = {}, {}, []
        for r in rs:
            scores[f"rule:{r}"] = rs[r]
            thr[f"rule:{r}"] = RULE_THRESHOLD[r]
            for m in ms:
                key = f"{m}@{r}_rate"
                scores[key] = ms[m]
                thr[key] = model_thr[(m, r)]
                compare.append((key, f"rule:{r}"))
        frames.append(score_methods(full, scores, thr, label, n_reps=n_reps, seed=seed + i,
                                    compare=compare))
        # overlap among undiagnosed cases (overall, point estimates)
        lab, a = label_arrays(full)
        u = lab["undx"] * full["w"].to_numpy(float)
        for r in rs:
            rf = rs[r] >= RULE_THRESHOLD[r]
            for m in ms:
                mf = np.nan_to_num(ms[m]) >= model_thr[(m, r)]
                tot = u.sum()
                overlap.append(dict(period_label=label, rule=r, model=m,
                                    missed_by_both=float(u[~rf & ~mf].sum() / tot),
                                    missed_by_rule_only=float(u[~rf & mf].sum() / tot),
                                    missed_by_model_only=float(u[rf & ~mf].sum() / tot),
                                    caught_by_both=float(u[rf & mf].sum() / tot),
                                    rule_share_2011_2016=rule_share[r]))
        print(f"  scored {label}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    fam = (out["period_label"].isin([PERIODS[p] for p in TEST_PERIODS])
           & out["role"].isin(["overall", "primary"]) & out["metric"].eq("sensitivity")
           & out["method"].str.match(r"diag_(logistic|lightgbm)@(" + "|".join(MAIN_RULES) + r")_rate$")
           & out["label_or_gap"].str.match(r"diff_vs_rule:(" + "|".join(MAIN_RULES) + r"):undx$"))
    out = add_family(out, "rule_vs_model", fam)
    return out, pd.DataFrame(overlap), rule_share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="nhanes_out/nhanes_all.csv")
    ap.add_argument("--models", default="results/rq2_models.joblib")
    ap.add_argument("--pu", default="results/pu_models.joblib")
    ap.add_argument("--out", default="results")
    ap.add_argument("--reps", type=int, default=500)
    args = ap.parse_args()
    df = load(args.csv)
    base = joblib.load(args.models)
    pu = joblib.load(args.pu) if Path(args.pu).exists() else None
    res, overlap, share = evaluate(df, base, pu, args.reps)
    res.to_csv(Path(args.out) / "screening_eval.csv", index=False)
    overlap.to_csv(Path(args.out) / "screening_overlap.csv", index=False)
    print("rule flag share 2011-2016:", {k: round(v, 3) for k, v in share.items()})
    v = res[(res["subgroup"] == "all") & res["method"].str.startswith("rule:")
            & res["label_or_gap"].isin(["undx", "y_diag", "gap_diag_minus_undx", "flagged"])]
    print(v.pivot_table(index=["method", "period_label"], columns="label_or_gap", values="estimate").round(3))
    fam = res[res["family"] == "rule_vs_model"]
    print(f"rule_vs_model: {len(fam)} tests, {int((fam['q_bh'] < 0.05).sum())} q<0.05")


if __name__ == "__main__":
    main()
