"""Upgrade 4 (shared infrastructure): evaluate each cycle as it arrives.

Instead of three pooled periods, every NHANES cycle is a time point. Two
designs, both written to results/rolling_eval.csv in the shared schema:

    expanding   for each cycle from the second on, train on ALL earlier cycles
                (pooled weights, years / total years) and test on that cycle:
                what a team that retrains every cycle would see.
    frozen      train once on the first cycle and test on every later cycle,
                so performance can be plotted against the lag in years:
                what a team that never retrains would see.

Every method is a (fit, score) pair, so workstreams plug in their own:
    METHODS["name"] = (fit(train_rows) -> fitted, score(fitted, rows) -> risk)
Built in: y_diag logistic / LightGBM, SAR-EM logistic, y_bio logistic, and
the USPSTF 2021 rule (no fitting).

Thresholds: each fitted model flags the matched share of its own training
population (the y_diag logistic 2011-2016 out-of-fold share, 33.4%), scored on
in-sample training predictions, so the operating point is comparable across
windows. Rules use their own cut-off.

Trends (results/rolling_trend.csv): inverse-variance weighted regression of
each metric on the test cycle's midpoint year (or on lag), with a t-test on
the slope. With five cycles there are at most four points, so this is
deliberately simple. When the 1999-2023 file exists, pass it with --csv and
the same code gives eleven points.

Run: python -m src.rolling --reps 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from src.harness import score_methods, threshold_for_share, label_arrays, SEED
from src.models import FEATURE_NAMES, analysis_domain
from src.pu_models import _fit_plain, fit_sar_em, RISK_FEATURES, weighted_prior
from src.screening_rules import rule_scores, RULE_THRESHOLD

MATCHED_SHARE = 0.3336   # y_diag logistic, 2011-2016 out-of-fold (harness.reference_flag_share)
CYCLE_YEARS = {"1999-2000": 2.0, "2001-2002": 2.0, "2003-2004": 2.0, "2005-2006": 2.0,
               "2007-2008": 2.0, "2009-2010": 2.0, "2011-2012": 2.0, "2013-2014": 2.0,
               "2015-2016": 2.0, "2017-2020": 3.2, "2021-2023": 2.0}
MIDPOINT = {"1999-2000": 2000.0, "2001-2002": 2002.0, "2003-2004": 2004.0, "2005-2006": 2006.0,
            "2007-2008": 2008.0, "2009-2010": 2010.0, "2011-2012": 2012.0, "2013-2014": 2014.0,
            "2015-2016": 2016.0, "2017-2020": 2018.6, "2021-2023": 2022.3}


def cycle_frame(df, cycles):
    """All rows of the given cycles, w = w_lab scaled by years / total years."""
    total = sum(CYCLE_YEARS[c] for c in cycles)
    f = df.loc[df["cycle"].isin(cycles)].copy().reset_index(drop=True)
    f["w"] = pd.to_numeric(f["w_lab"], errors="coerce").fillna(0.0) * \
        f["cycle"].map(lambda c: CYCLE_YEARS[c] / total)
    return f


def _plain(target, kind, feats=FEATURE_NAMES):
    def fit(rows):
        return _fit_plain(rows, rows[target].to_numpy(int), kind, feats)
    return fit


def _sar(rows):
    return fit_sar_em(rows, "logistic", weighted_prior(rows))


def _score_fitted(fitted, rows):
    return fitted.predict(rows)


def _rule(name):
    return (lambda rows: None, lambda _, rows: rule_scores(rows)[name], RULE_THRESHOLD[name])


METHODS = {
    "diag_logistic": (_plain("y_diag", "logistic"), _score_fitted, None),
    "diag_lightgbm": (_plain("y_diag", "lightgbm"), _score_fitted, None),
    "sar_logistic": (_sar, _score_fitted, None),
    "bio_logistic": (_plain("y_bio", "logistic"), _score_fitted, None),
    "rule:uspstf2021": _rule("uspstf2021"),
}


def evaluate_window(df, train_cycles, test_cycle, design, n_reps, seed, methods=METHODS):
    train = cycle_frame(df, train_cycles)
    tr = train.loc[analysis_domain(train)].reset_index(drop=True)
    test = cycle_frame(df, [test_cycle])
    an = analysis_domain(test).to_numpy()
    scores, thr = {}, {}
    for name, (fit, score, fixed_thr) in methods.items():
        fitted = fit(tr)
        s = np.full(len(test), np.nan)
        s[an] = score(fitted, test.loc[an])
        scores[name] = s
        if fixed_thr is None:
            thr[name] = threshold_for_share(score(fitted, tr), tr["w"].to_numpy(float), MATCHED_SHARE)
        else:
            thr[name] = fixed_thr
    label = f"{design}: {'+'.join(train_cycles) if len(train_cycles) < 3 else train_cycles[0] + '..' + train_cycles[-1]} -> {test_cycle}"
    out = score_methods(test, scores, thr, label, n_reps=n_reps, seed=seed,
                        compare=[(m, "diag_logistic") for m in scores if m != "diag_logistic"])
    out.insert(1, "design", design)
    out.insert(2, "test_cycle", test_cycle)
    out.insert(3, "train_cycles", "+".join(train_cycles))
    out.insert(4, "test_year", MIDPOINT[test_cycle])
    out.insert(5, "lag_years", MIDPOINT[test_cycle] - np.mean([MIDPOINT[c] for c in train_cycles]))
    return out


def trend(res, x="test_year"):
    """Inverse-variance weighted slope per design x method x cell x metric."""
    keep = res["status"].eq("ok") & res["se"].gt(0) & \
        res["label_or_gap"].isin(["undx", "y_diag", "y_bio", "gap_diag_minus_undx"]) & \
        res["metric"].isin(["auroc", "sensitivity"])
    rows = []
    for key, g in res[keep].groupby(["design", "method", "subgroup", "level", "label_or_gap", "metric"]):
        g = g.sort_values(x)
        if len(g) < 3:
            continue
        xv, yv, wv = g[x].to_numpy(float), g["estimate"].to_numpy(float), 1 / g["se"].to_numpy(float) ** 2
        X = np.column_stack([np.ones_like(xv), xv - xv.mean()])
        XtW = X.T * wv
        cov = np.linalg.inv(XtW @ X)
        beta = cov @ (XtW @ yv)
        dof = len(g) - 2
        resid = yv - X @ beta
        # scale by residual dispersion when it exceeds the sampling variance
        phi = max(1.0, float(np.sum(wv * resid ** 2) / dof)) if dof > 0 else 1.0
        se = float(np.sqrt(cov[1, 1] * phi))
        tval = beta[1] / se
        p = float(2 * student_t.sf(abs(tval), max(dof, 1)))
        rows.append(dict(zip(["design", "method", "subgroup", "level", "label_or_gap", "metric"], key),
                         x=x, points=len(g), slope_per_decade=float(beta[1] * 10),
                         se_per_decade=se * 10, p_value=p, first=float(yv[0]), last=float(yv[-1])))
    return pd.DataFrame(rows)


def run(df, n_reps=200, seed=SEED, methods=METHODS):
    cycles = [c for c in MIDPOINT if c in set(df["cycle"].unique())]
    frames = []
    for i, c in enumerate(cycles[1:], start=1):
        frames.append(evaluate_window(df, cycles[:i], c, "expanding", n_reps, seed + i, methods))
        print(f"  expanding -> {c}", flush=True)
        if i >= 2:
            frames.append(evaluate_window(df, cycles[:1], c, "frozen", n_reps, seed + 100 + i, methods))
            print(f"  frozen {cycles[0]} -> {c}", flush=True)
    res = pd.concat(frames, ignore_index=True)
    # lag 1 of the frozen design is the first expanding window; copy it so both designs are complete
    first = res[(res["design"] == "expanding") & (res["test_cycle"] == cycles[1])].copy()
    first["design"] = "frozen"
    first["period_label"] = first["period_label"].str.replace("expanding", "frozen")
    res = pd.concat([res, first], ignore_index=True)
    tr = pd.concat([trend(res[res["design"] == "expanding"], "test_year"),
                    trend(res[res["design"] == "frozen"], "lag_years")], ignore_index=True)
    return res, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="nhanes_out/nhanes_all.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    df = pd.read_csv(args.csv, low_memory=False)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    res, tr = run(df, args.reps)
    res.to_csv(Path(args.out) / "rolling_eval.csv", index=False)
    tr.to_csv(Path(args.out) / "rolling_trend.csv", index=False)
    v = res[(res["subgroup"] == "all") & res["label_or_gap"].isin(["undx", "y_bio", "gap_diag_minus_undx"])
            & ~res["method"].str.contains("diff")]
    print(v.pivot_table(index=["design", "test_cycle"], columns=["method", "label_or_gap", "metric"],
                        values="estimate").round(3).T.to_string())
    print(tr[(tr["subgroup"] == "all")].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
