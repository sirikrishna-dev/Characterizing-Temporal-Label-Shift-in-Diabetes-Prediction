"""
drift.py  (RQ3: does survey-weighted drift monitoring catch the failures?)

Reads the cleaned NHANES file from the data notebook and compares the train
period (2011-2016 pooled) with each test period (2017-March 2020 and
August 2021-2023), overall and inside every subgroup cell.

A drift monitor only sees what a deployed system would see: the features and
the diagnosis label y_diag. The biomarker labels (y_bio, y_bio_fpg) are
measured here only as a reference, so we can later check whether the
monitors alarmed where the label process actually changed.

Outputs written to --out:
  drift_features.csv  per feature: PSI (and KS for numeric), weighted and
                      unweighted, each against its own bootstrap noise floor
  drift_domain.csv    domain-classifier AUC, weighted and unweighted
  label_shift.csv     change in weighted prevalence of y_diag, y_bio,
                      y_bio_fpg, with design-based CI and BH-adjusted p-value
  drift_alarms.csv    one row per period x subgroup cell: which monitors
                      alarmed. monitor.py joins this with model_temporal.csv
                      (from model_eval.py) for the monitoring 2x2.

Run from the project root (survey_drift.py sits next to this file):
  python -m src.drift
which is the same as
  python -m src.drift --data nhanes_out/nhanes_all.csv --out results --reps 200

Weights: --weight must be the HbA1c-analysis weight, i.e. the MEC weight for
2011-2020 and WTPH2YR for 2021-2023 (column w_lab in the cleaned file).
--fast-weight is the fasting subsample weight, used only for y_bio_fpg.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import survey_drift as sd  # noqa: E402

# ---------------------------------------------------------------------------
# Study set-up (must match the definitions in the project description)
# ---------------------------------------------------------------------------

TRAIN_CYCLES = ["2011-2012", "2013-2014", "2015-2016"]
TEST_PERIODS = {
    "2017-2020": ["2017-2020"],
    "2021-2023": ["2021-2023"],
}

# Column names as written by the data notebook (v7, after the crosswalk)
NUMERIC_FEATURES = ["age", "pir", "bmi", "waist"]
CATEGORICAL_FEATURES = ["sex", "race_eth", "education", "insured", "routine_care"]

# Pre-specified primary subgroups from the data notebook's reporting plan
# (step 7). The exploratory ones run only with --exploratory.
PRIMARY_SUBGROUPS = ["income_band2", "insured", "age_band2", "sex"]
EXPLORATORY_SUBGROUPS = ["income_band", "routine_care", "age_band", "race_eth4"]
SUBGROUPS = list(PRIMARY_SUBGROUPS)

MONITOR_LABEL = "y_diag"          # visible to a deployed monitor
REFERENCE_LABEL = "y_bio"         # HbA1c-based, same weight as the features
FASTING_LABEL = "y_bio_fpg"       # fasting subsample only, fasting weight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def as_bool(s: pd.Series) -> pd.Series:
    """'adult' may arrive as bool, 0/1 or 'True'/'False' after a CSV round trip."""
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def check_columns(df: pd.DataFrame, cols) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns: {missing}.\nAvailable: {sorted(df.columns)}\n"
            "Fix the column names at the top of drift.py or in the data notebook."
        )


def check_cycles(df: pd.DataFrame) -> None:
    present = set(df["cycle"].astype(str).unique())
    needed = set(TRAIN_CYCLES) | {c for cs in TEST_PERIODS.values() for c in cs}
    absent = needed - present
    if absent:
        raise ValueError(f"Cycles {sorted(absent)} not in data. Found: {sorted(present)}")


def build_period(df, cycles, weight_col, population, n_rep, seed):
    """Rows of one period with positive weight, pooled weights, replicates.

    Replicates are built on everyone in the weighted subsample of these
    cycles. The analysis population (adults, non-pregnant, label present) is
    applied afterwards as a domain, so the variance stays correct.
    """
    d = df[df["cycle"].isin(cycles)].copy()
    d["w"] = pd.to_numeric(d[weight_col], errors="coerce")
    d = d[d["w"] > 0].reset_index(drop=True)
    if d.empty:
        raise ValueError(f"No rows with positive {weight_col} in cycles {cycles}.")
    d = sd.pool_weights(d)
    reps = sd.rao_wu_replicates(d, n_rep=n_rep, seed=seed)
    dof = sd.design_dof(d)
    keep = population(d).to_numpy()
    return d.loc[keep].reset_index(drop=True), reps[keep], dof


def subgroup_cells(ref, new):
    """('all', 'all') plus every level of every subgroup present in both periods."""
    cells = [("all", "all", None)]
    for var in SUBGROUPS:
        levels = sorted(set(ref[var].dropna().astype(str)) & set(new[var].dropna().astype(str)))
        for lev in levels:
            cells.append((var, lev, lev))
    return cells


def cell_mask(df, var, level):
    if var == "all":
        return np.ones(len(df), dtype=bool)
    return (df[var].astype(str) == level).to_numpy()


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    p = p.astype(float)
    ok = p.notna()
    out = pd.Series(np.nan, index=p.index)
    if ok.sum() == 0:
        return out
    vals = p[ok].to_numpy()
    order = np.argsort(vals)
    m = len(vals)
    adj = vals[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.clip(adj, 0, 1)
    out[ok] = res
    return out


# ---------------------------------------------------------------------------
# Monitors
# ---------------------------------------------------------------------------

def _psi_num(a, aw, b, bw):
    return sd.weighted_psi(a, aw, b, bw)


def _psi_cat(a, aw, b, bw):
    return sd.weighted_categorical_psi(a, aw, b, bw)


def _ks(a, aw, b, bw):
    return sd.weighted_ks(a, aw, b, bw)


def code_col(col):
    return f"_code_{col}"


def add_category_codes(df):
    """Integer codes for categorical features, shared across all cycles
    (missing = -1). Makes categorical PSI a fast bincount."""
    for col in CATEGORICAL_FEATURES:
        df[code_col(col)] = pd.factorize(df[col].astype("string"), use_na_sentinel=True)[0]
    return df


def feature_drift(ref, rr, new, nr, period, var, level, alpha, min_n):
    """Weighted and unweighted drift for every feature in one cell.

    The unweighted version uses the same bootstrap draws with the survey
    weights set to 1 (replicate / full weight = the resampling factor), so
    both versions get a fair null of their own.

    Each test gets a p-value against its bootstrap null. Within a cell the
    p-values are Bonferroni-corrected over all feature tests, so a cell does
    not alarm just because many features were checked.
    """
    rows = []
    rw_all, nw_all = ref["w"].to_numpy(), new["w"].to_numpy()
    rr_unw = rr / rw_all[:, None]
    nr_unw = nr / nw_all[:, None]
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        numeric = col in NUMERIC_FEATURES
        src = col if numeric else code_col(col)
        if numeric:
            rm, nm = ref[col].notna().to_numpy(), new[col].notna().to_numpy()
        else:
            rm, nm = (ref[src] >= 0).to_numpy(), (new[src] >= 0).to_numpy()
        base = {"period": period, "subgroup": var, "level": level, "feature": col,
                "n_ref": int(rm.sum()), "n_new": int(nm.sum())}
        if rm.sum() < min_n or nm.sum() < min_n:
            rows.append({**base, "metric": "psi", "status": "too_small"})
            continue
        rx, nx = ref[src].to_numpy()[rm], new[src].to_numpy()[nm]
        rx, nx = (rx.astype(float), nx.astype(float)) if numeric else (rx.astype(int), nx.astype(int))
        metrics = {"psi": _psi_num if numeric else _psi_cat}
        if numeric:
            metrics["ks"] = _ks
        for name, fn in metrics.items():
            wtd = sd.two_period_drift(fn, rx, rw_all[rm], rr[rm], nx, nw_all[nm], nr[nm], alpha)
            unw = sd.two_period_drift(fn, rx, np.ones(rm.sum()), rr_unw[rm],
                                      nx, np.ones(nm.sum()), nr_unw[nm], alpha)
            rows.append({**base, "metric": name, "status": "ok",
                         "unweighted": unw["observed"],
                         "unweighted_floor": unw["noise_floor"],
                         "unweighted_p": unw["p_value"],
                         "weighted": wtd["observed"],
                         "weighted_floor": wtd["noise_floor"],
                         "weighted_ci_low": wtd["ci_low"],
                         "weighted_ci_high": wtd["ci_high"],
                         "weighted_p": wtd["p_value"]})

    m = sum(1 for r in rows if r["status"] == "ok")
    for r in rows:
        if r["status"] != "ok":
            continue
        r["weighted_p_bonf"] = min(1.0, r["weighted_p"] * m)
        r["unweighted_p_bonf"] = min(1.0, r["unweighted_p"] * m)
        r["real_shift"] = r["weighted_p_bonf"] < alpha
        r["unweighted_shift"] = r["unweighted_p_bonf"] < alpha
        r["sampling_artifact"] = r["unweighted_shift"] and not r["real_shift"]
        r["hidden_by_unweighted"] = r["real_shift"] and not r["unweighted_shift"]
    return rows


def encode_features(ref, new):
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    both = pd.concat([ref[cols], new[cols]], ignore_index=True)
    for c in CATEGORICAL_FEATURES:
        both[c] = both[c].astype("string")
    X = pd.get_dummies(both, columns=CATEGORICAL_FEATURES, dummy_na=False).astype(float)
    return X.iloc[: len(ref)].to_numpy(), X.iloc[len(ref):].to_numpy()


def _domain(a, aw, b, bw):
    return sd.weighted_domain_auc(a, aw, b, bw, n_splits=3)


def domain_drift(ref, rr, new, nr, period, var, level, alpha, min_n, n_rep_domain):
    base = {"period": period, "subgroup": var, "level": level,
            "n_ref": len(ref), "n_new": len(new)}
    if len(ref) < min_n or len(new) < min_n:
        return {**base, "status": "too_small"}
    Xr, Xn = encode_features(ref, new)
    unweighted = _domain(Xr, np.ones(len(Xr)), Xn, np.ones(len(Xn)))
    observed = _domain(Xr, ref["w"].to_numpy(), Xn, new["w"].to_numpy())
    res = sd.replicate_auc_test(_domain, Xr, rr[:, :n_rep_domain], Xn, nr[:, :n_rep_domain],
                                observed, null_value=0.5, alpha=alpha)
    return {**base, "status": "ok", "unweighted_auc": unweighted,
            "weighted_auc": res["observed"], "weighted_se": res["se"],
            "weighted_ci_low": res["ci_low"], "weighted_ci_high": res["ci_high"],
            "p_value": res["p_value"], "real_shift": res["real_shift"]}


def label_shift(ref, rr, rdof, new, nr, ndof, label, period, var, level, alpha, min_n):
    rm = ref[label].notna().to_numpy()
    nm = new[label].notna().to_numpy()
    base = {"period": period, "subgroup": var, "level": level, "label": label,
            "visible_to_monitor": label == MONITOR_LABEL,
            "n_ref": int(rm.sum()), "n_new": int(nm.sum())}
    if rm.sum() < min_n or nm.sum() < min_n:
        return {**base, "status": "too_small"}
    ry = ref[label].to_numpy(float)[rm]
    ny = new[label].to_numpy(float)[nm]
    res = sd.difference_test(lambda a, w: sd.wmean(a, w),
                             ry, ref["w"].to_numpy()[rm], rr[rm], rdof,
                             ny, new["w"].to_numpy()[nm], nr[nm], ndof, alpha)
    return {**base, "status": "ok",
            "unweighted_ref": float(ry.mean()), "unweighted_new": float(ny.mean()),
            "unweighted_diff": float(ny.mean() - ry.mean()),
            "weighted_ref": res["ref"], "weighted_new": res["new"],
            "weighted_diff": res["diff"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
            "p_value": res["p_value"], "dof": res["dof"]}


# ---------------------------------------------------------------------------
# Alarm table
# ---------------------------------------------------------------------------

def build_alarms(feat, dom, lab, alpha):
    """One row per period x cell. A monitor alarms when its weighted test
    says the shift is real (labels: BH-adjusted p < alpha)."""
    ok_feat = feat[feat["status"] == "ok"]
    f = (ok_feat.groupby(["period", "subgroup", "level"])["real_shift"]
         .agg(feature_alarms="sum").reset_index())
    f["feature_alarm"] = f["feature_alarms"] > 0

    d = dom[dom["status"] == "ok"][["period", "subgroup", "level", "real_shift"]] \
        .rename(columns={"real_shift": "domain_alarm"})

    ld = lab[(lab["status"] == "ok") & (lab["label"] == MONITOR_LABEL)]
    ld = ld.assign(label_alarm=ld["p_adj"] < alpha)[["period", "subgroup", "level", "label_alarm"]]

    ref_lab = lab[(lab["status"] == "ok") & (lab["label"] == REFERENCE_LABEL)]
    ref_lab = ref_lab.assign(y_bio_shift=ref_lab["p_adj"] < alpha)[
        ["period", "subgroup", "level", "weighted_diff", "y_bio_shift"]
    ].rename(columns={"weighted_diff": "y_bio_prevalence_change"})

    out = f.merge(d, how="outer", on=["period", "subgroup", "level"]) \
           .merge(ld, how="outer", on=["period", "subgroup", "level"]) \
           .merge(ref_lab, how="left", on=["period", "subgroup", "level"])
    for c in ["feature_alarm", "domain_alarm", "label_alarm"]:
        out[c] = out[c].astype("boolean")
    out["any_alarm"] = out[["feature_alarm", "domain_alarm", "label_alarm"]].fillna(False).any(axis=1)

    # What a monitor that only watches the whole population would say
    overall = out[out["subgroup"] == "all"].set_index("period")["any_alarm"]
    out["overall_monitor_alarm"] = out["period"].map(overall)
    return out.sort_values(["period", "subgroup", "level"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="nhanes_out/nhanes_all.csv",
                    help="cleaned NHANES file (.csv or .parquet)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--weight", default="w_lab", help="HbA1c-analysis weight column")
    ap.add_argument("--fast-weight", default="w_fast", help="fasting subsample weight column")
    ap.add_argument("--reps", type=int, default=200, help="bootstrap replicates")
    ap.add_argument("--domain-reps", type=int, default=30, help="replicates for the domain classifier")
    ap.add_argument("--domain-subgroups", action="store_true",
                    help="also run the domain classifier inside every subgroup (slow)")
    ap.add_argument("--exploratory", action="store_true",
                    help="also run the exploratory subgroups from the reporting plan")
    ap.add_argument("--min-n", type=int, default=100, help="minimum rows per period in a cell")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.exploratory:
        SUBGROUPS.extend(EXPLORATORY_SUBGROUPS)

    t0 = time.time()
    df = pd.read_parquet(args.data) if args.data.endswith(".parquet") else pd.read_csv(args.data, low_memory=False)
    df["cycle"] = df["cycle"].astype(str)
    check_columns(df, ["cycle", "stratum", "psu", "adult", args.weight, MONITOR_LABEL,
                       REFERENCE_LABEL] + NUMERIC_FEATURES + CATEGORICAL_FEATURES + SUBGROUPS)
    check_cycles(df)
    df = add_category_codes(df)
    os.makedirs(args.out, exist_ok=True)

    def main_pop(d):
        return as_bool(d["adult"]) & d[REFERENCE_LABEL].notna()

    def fasting_pop(d):
        return as_bool(d["adult"]) & d[FASTING_LABEL].notna()

    print(f"Building train period {TRAIN_CYCLES} with {args.weight} ...")
    ref, rr, rdof = build_period(df, TRAIN_CYCLES, args.weight, main_pop, args.reps, args.seed)
    print(f"  {len(ref):,} rows, design dof {rdof}")

    do_fasting = FASTING_LABEL in df.columns and args.fast_weight in df.columns
    if do_fasting:
        ref_f, rr_f, rdof_f = build_period(df, TRAIN_CYCLES, args.fast_weight, fasting_pop,
                                           args.reps, args.seed + 100)
    else:
        print(f"  {FASTING_LABEL} or {args.fast_weight} not found: skipping the fasting label.")

    feat_rows, dom_rows, lab_rows = [], [], []
    for i, (period, cycles) in enumerate(TEST_PERIODS.items(), start=1):
        print(f"Test period {period} ...")
        new, nr, ndof = build_period(df, cycles, args.weight, main_pop, args.reps, args.seed + i)
        print(f"  {len(new):,} rows, design dof {ndof}")
        if do_fasting:
            new_f, nr_f, ndof_f = build_period(df, cycles, args.fast_weight, fasting_pop,
                                               args.reps, args.seed + 100 + i)

        for var, level, lev in subgroup_cells(ref, new):
            rm, nm = cell_mask(ref, var, lev), cell_mask(new, var, lev)
            r_c, n_c = ref.loc[rm].reset_index(drop=True), new.loc[nm].reset_index(drop=True)
            feat_rows += feature_drift(r_c, rr[rm], n_c, nr[nm], period, var, level,
                                       args.alpha, args.min_n)
            if var == "all" or args.domain_subgroups:
                dom_rows.append(domain_drift(r_c, rr[rm], n_c, nr[nm], period, var, level,
                                             args.alpha, args.min_n, args.domain_reps))
            for label in (MONITOR_LABEL, REFERENCE_LABEL):
                lab_rows.append(label_shift(r_c, rr[rm], rdof, n_c, nr[nm], ndof, label,
                                            period, var, level, args.alpha, args.min_n))
            if do_fasting:
                rmf, nmf = cell_mask(ref_f, var, lev), cell_mask(new_f, var, lev)
                lab_rows.append(label_shift(ref_f.loc[rmf].reset_index(drop=True), rr_f[rmf], rdof_f,
                                            new_f.loc[nmf].reset_index(drop=True), nr_f[nmf], ndof_f,
                                            FASTING_LABEL, period, var, level, args.alpha, args.min_n))
            print(f"    {var}={level} done ({time.time() - t0:.0f}s)")

    feat = pd.DataFrame(feat_rows)
    dom = pd.DataFrame(dom_rows)
    lab = pd.DataFrame(lab_rows)
    lab["p_adj"] = np.nan
    for label in lab["label"].unique():
        m = (lab["label"] == label) & (lab["status"] == "ok")
        lab.loc[m, "p_adj"] = benjamini_hochberg(lab.loc[m, "p_value"]).to_numpy()
    lab["real_shift"] = lab["p_adj"] < args.alpha

    alarms = build_alarms(feat, dom, lab, args.alpha)

    feat.to_csv(os.path.join(args.out, "drift_features.csv"), index=False)
    dom.to_csv(os.path.join(args.out, "drift_domain.csv"), index=False)
    lab.to_csv(os.path.join(args.out, "label_shift.csv"), index=False)
    alarms.to_csv(os.path.join(args.out, "drift_alarms.csv"), index=False)

    ok = feat[feat["status"] == "ok"]
    print(f"\nDone in {time.time() - t0:.0f}s. Files in {args.out}/")
    print(f"Feature tests: {len(ok)}; real shifts (weighted): {int(ok['real_shift'].sum())}; "
          f"sampling artifacts (unweighted only): {int(ok['sampling_artifact'].sum())}; "
          f"hidden by unweighted: {int(ok['hidden_by_unweighted'].sum())}")


if __name__ == "__main__":
    main()
