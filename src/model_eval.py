"""RQ2 evaluation: fixed models, survey-weighted metrics, Rao-Wu bootstrap CIs.

Run from the project root, after python -m src.models:
    python -m src.model_eval --reps 500

Periods
    train_cv  2011-2016, scored with the out-of-fold predictions saved by
              models.py (each cycle predicted by models fitted on the other
              two). This is the in-period baseline.
    test_P    2017-March 2020, scored with the final models.
    test_L    August 2021-2023, scored with the final models.

What is measured, for every model x period x domain
    auroc, sensitivity, specificity, brier, calibration_in_large, ece
        against y_diag (what a deployed system would see as the label) and
        against y_bio (diagnosed or HbA1c >= 6.5%).
    sensitivity among undx (undiagnosed adults with HbA1c >= 6.5%): the share
        of hidden cases the model would flag.
    paired gaps on the same people and the same bootstrap draws:
        gap_diag_minus_bio   metric(y_diag) - metric(y_bio)
        gap_diag_minus_undx  sensitivity among diagnosed - sensitivity among
                             undiagnosed. gap_diag_minus_bio for sensitivity
                             is this gap diluted by the share of y_bio cases
                             that are already diagnosed (about 83%).

Multiple testing (Benjamini-Hochberg), two pre-specified families
    label_gap  in model_eval.csv: AUROC gap_diag_minus_bio and sensitivity
               gap_diag_minus_undx, overall and primary subgroups, both models,
               both test periods.
    temporal   in model_temporal.csv: change from the 2011-2016 baseline in
               AUROC (y_diag, y_bio) and sensitivity (y_diag, y_bio, undx),
               overall and primary subgroups, both models, both test periods.
Brier, calibration-in-the-large, ECE and specificity are descriptive (CIs, no
q-values): their label gaps follow mechanically from the prevalence gap.
Exploratory subgroups get CIs only. Subgroup levels that are the same people
as an earlier level (income_band <1.3 = income_band2 <1.3, age_band 60+ =
age_band2 60+, race_eth NH White/NH Black = race_eth4) are evaluated once.

Join keys match drift.py: period_label is 2017-2020 / 2021-2023, the overall
row is subgroup='all', level='all', and levels are written as strings.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t as student_t
from sklearn.metrics import roc_auc_score

from src.models import analysis_domain
from src.survey_boot import (rao_wu_replicate_weights, design_degrees_of_freedom,
                             bootstrap_t_ci)

PRIMARY_SUBGROUPS = ("income_band2", "insured", "age_band2", "sex")
EXPLORATORY_SUBGROUPS = ("income_band", "routine_care", "age_band", "race_eth4", "race_eth")
PERIOD_LABELS = {"train_cv": "2011-2016", "test_P": "2017-2020", "test_L": "2021-2023"}
DATA_PERIOD = {"train_cv": "train", "test_P": "test_P", "test_L": "test_L"}  # value in the period column
TEST_PERIODS = ("test_P", "test_L")
ECE_BINS = 10

# (label, metric) pairs computed for every domain
QUANTITIES = [
    ("y_diag", "auroc"), ("y_bio", "auroc"),
    ("y_diag", "sensitivity"), ("y_bio", "sensitivity"), ("undx", "sensitivity"),
    ("y_diag", "specificity"), ("y_bio", "specificity"),
    ("y_diag", "brier"), ("y_bio", "brier"),
    ("y_diag", "calibration_in_large"), ("y_bio", "calibration_in_large"),
    ("y_diag", "ece"), ("y_bio", "ece"),
]
GAPS = [("gap_diag_minus_bio", m, "y_diag", "y_bio")
        for m in ("auroc", "sensitivity", "specificity", "brier", "calibration_in_large", "ece")]
GAPS.append(("gap_diag_minus_undx", "sensitivity", "y_diag", "undx"))

LABEL_GAP_FAMILY = {("gap_diag_minus_bio", "auroc"), ("gap_diag_minus_undx", "sensitivity")}
TEMPORAL_FAMILY = {("y_diag", "auroc"), ("y_bio", "auroc"), ("y_diag", "sensitivity"),
                   ("y_bio", "sensitivity"), ("undx", "sensitivity")}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def weighted_metric(y, probability, weight, metric, threshold, bins=ECE_BINS):
    """Reference implementation for one weight vector (used by the tests to
    check the fast matrix version below). Returns nan when undefined."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(probability, dtype=float)
    w = np.asarray(weight, dtype=float)
    if not (len(y) == len(p) == len(w)):
        raise ValueError("Metric arrays must have equal length")
    valid = (w > 0) & np.isfinite(w) & np.isfinite(p) & np.isfinite(y)
    if not valid.any():
        return float("nan")
    y, p, w = y[valid], p[valid], w[valid]
    if not np.isin(y, [0, 1]).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Labels must be binary and predictions in [0, 1]")
    flag = p >= threshold
    if metric == "auroc":
        return float(roc_auc_score(y, p, sample_weight=w)) if np.unique(y).size == 2 else float("nan")
    if metric == "sensitivity":
        pos = y == 1
        return float(w[pos & flag].sum() / w[pos].sum()) if w[pos].sum() > 0 else float("nan")
    if metric == "specificity":
        neg = y == 0
        return float(w[neg & ~flag].sum() / w[neg].sum()) if w[neg].sum() > 0 else float("nan")
    if metric == "brier":
        return float(np.dot(w, (p - y) ** 2) / w.sum())
    if metric == "calibration_in_large":
        return float(np.dot(w, p - y) / w.sum())
    if metric == "ece":
        b = np.minimum((p * bins).astype(int), bins - 1)
        predicted = np.bincount(b, weights=w * p, minlength=bins)
        observed = np.bincount(b, weights=w * y, minlength=bins)
        return float(np.abs(predicted - observed).sum() / w.sum())
    raise ValueError(f"Unknown metric: {metric}")


class ModelArrays:
    """Everything about one model's predictions that does not change between
    bootstrap draws, precomputed once so each draw is a few matrix products."""

    def __init__(self, probability, threshold, labels):
        p = np.where(np.isfinite(probability), probability, 0.0)
        self.p = p
        self.flag = (p >= threshold).astype(float)
        levels, inverse = np.unique(p, return_inverse=True)
        n = len(p)
        self.rank_onehot = sparse.csr_matrix(
            (np.ones(n), (inverse, np.arange(n))), shape=(len(levels), n))
        bin_index = np.minimum((p * ECE_BINS).astype(int), ECE_BINS - 1)
        self.bins = sparse.csr_matrix(
            (np.ones(n), (bin_index, np.arange(n))), shape=(ECE_BINS, n))
        self.labels = labels  # dict name -> 0/1 float array, 0 where undefined


def domain_metrics(arrays, W):
    """All QUANTITIES for every domain at once.

    W is an (n rows x D domains) weight matrix: the survey (or replicate)
    weight inside each domain and 0 outside. Returns {(label, metric): array(D)}.
    """
    out = {}
    total = W.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        for label in ("y_diag", "y_bio", "undx"):
            y = arrays.labels[label]
            pos_w = W.T @ y
            pos_flag = W.T @ (y * arrays.flag)
            out[(label, "sensitivity")] = np.where(pos_w > 0, pos_flag / pos_w, np.nan)
            if label == "undx":
                continue
            neg_w = total - pos_w
            neg_noflag = W.T @ ((1 - y) * (1 - arrays.flag))
            out[(label, "specificity")] = np.where(neg_w > 0, neg_noflag / neg_w, np.nan)

            # Weighted AUROC with ties counted as one half (same as sklearn).
            pos_by_level = arrays.rank_onehot @ (W * y[:, None])
            neg_by_level = arrays.rank_onehot @ (W * (1 - y)[:, None])
            neg_below = np.cumsum(neg_by_level, axis=0) - neg_by_level
            concordant = (pos_by_level * (neg_below + 0.5 * neg_by_level)).sum(axis=0)
            denom = pos_by_level.sum(axis=0) * neg_by_level.sum(axis=0)
            out[(label, "auroc")] = np.where(denom > 0, concordant / denom, np.nan)

            out[(label, "brier")] = np.where(total > 0, (W.T @ (arrays.p - y) ** 2) / total, np.nan)
            out[(label, "calibration_in_large")] = np.where(
                total > 0, (W.T @ (arrays.p - y)) / total, np.nan)
            predicted = arrays.bins @ (W * arrays.p[:, None])
            observed = arrays.bins @ (W * y[:, None])
            out[(label, "ece")] = np.where(
                total > 0, np.abs(predicted - observed).sum(axis=0) / total, np.nan)
    return out


def in_set(frame, label_col, pairs):
    """Boolean Series: is (frame[label_col], frame['metric']) one of pairs?"""
    return pd.Series([(a, m) in pairs for a, m in zip(frame[label_col], frame["metric"])],
                     index=frame.index, dtype=bool)


def bh_adjust(pvalues):
    """Benjamini-Hochberg q-values; nan stays nan (not in the family)."""
    p = np.asarray(pvalues, dtype=float)
    result = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return result
    order = valid[np.argsort(p[valid])]
    ranks = np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate((p[order] * len(order) / ranks)[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return result


# ---------------------------------------------------------------------------
# Domains
# ---------------------------------------------------------------------------

def domain_list(df):
    """(subgroup, level, role) for the overall row and every subgroup level.

    A level that selects exactly the same analysis rows as an earlier level
    (checked on the whole file) is dropped, so the same people are never
    tested twice. Primary subgroups come first and are always kept.
    """
    common = analysis_domain(df).to_numpy()
    domains = [("all", "all", "overall")]
    seen = {}
    for col in PRIMARY_SUBGROUPS + EXPLORATORY_SUBGROUPS:
        if col not in df:
            raise ValueError(f"Missing subgroup column {col}")
        values = df[col].astype("string")
        role = "primary" if col in PRIMARY_SUBGROUPS else "exploratory"
        for level in sorted(values[common].dropna().unique()):
            mask = common & values.eq(level).fillna(False).to_numpy(dtype=bool)
            key = np.flatnonzero(mask).tobytes()
            if key in seen:
                continue
            seen[key] = (col, level)
            domains.append((col, str(level), role))
    return domains


def domain_masks(full, domains):
    common = analysis_domain(full).to_numpy()
    masks = []
    for col, level, _ in domains:
        if col == "all":
            masks.append(common)
        else:
            masks.append(common & full[col].astype("string").eq(level).fillna(False).to_numpy(dtype=bool))
    return np.column_stack(masks)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def period_predictions(full, bundle, period):
    """Predicted risk for each model on the period's rows (nan outside the
    analysis domain). The baseline period uses out-of-fold predictions."""
    analysis = analysis_domain(full).to_numpy()
    preds = {}
    for name in bundle.estimators:
        p = np.full(len(full), np.nan)
        if period == "train_cv":
            if bundle.oof is None:
                raise ValueError("Model bundle has no out-of-fold predictions; rerun src.models")
            lookup = bundle.oof.set_index("SEQN")[name]
            p[analysis] = full.loc[analysis, "SEQN"].map(lookup).to_numpy(dtype=float)
            if not np.isfinite(p[analysis]).all():
                raise ValueError("Some baseline rows have no out-of-fold prediction")
        else:
            p[analysis] = bundle.predict_probabilities(full.loc[analysis], name)
        preds[name] = p
    return preds


def summarise(point, replicates, design_df, status):
    if status != "ok":
        return dict(estimate=point if np.isfinite(point) else np.nan,
                    se=np.nan, ci_low=np.nan, ci_high=np.nan, p_value=np.nan)
    ci = bootstrap_t_ci(point, replicates, design_df)
    return {k: ci[k] for k in ("estimate", "se", "ci_low", "ci_high", "p_value")}


def evaluate_period(df, bundle, period, domains, n_reps, seed, min_undx):
    full = df.loc[df["period"] == DATA_PERIOD[period]].copy().reset_index(drop=True)
    if full.empty:
        raise ValueError(f"Period missing: {period}")
    # Every PSU stays in the design, including rows with no lab weight.
    full["w"] = pd.to_numeric(full["w"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan).fillna(0.0)
    if (full["w"] < 0).any():
        raise ValueError("Negative survey weight")
    dfree = design_degrees_of_freedom(full)
    masks = domain_masks(full, domains)
    weights = full["w"].to_numpy(dtype=float)
    analysis = analysis_domain(full).to_numpy()
    labels = {}
    for name in ("y_diag", "y_bio", "undx"):
        y = pd.to_numeric(full[name], errors="coerce").to_numpy(dtype=float)
        if not np.isin(y[analysis], [0, 1]).all():
            raise ValueError(f"{name} must be 0/1 on every analysis row")
        labels[name] = np.where(analysis, y, 0.0)
    n_rows = masks.sum(axis=0)
    n_diag = (masks * labels["y_diag"][:, None]).sum(axis=0)
    n_undx = (masks * labels["undx"][:, None]).sum(axis=0)

    preds = period_predictions(full, bundle, period)
    arrays = {name: ModelArrays(preds[name], bundle.thresholds[name], labels)
              for name in preds}

    point = {name: domain_metrics(a, masks * weights[:, None]) for name, a in arrays.items()}
    reps = {name: {q: [] for q in QUANTITIES} for name in arrays}
    for rw in rao_wu_replicate_weights(full, n_reps=n_reps, seed=seed):
        W = masks * rw[:, None]
        for name, a in arrays.items():
            values = domain_metrics(a, W)
            for q in QUANTITIES:
                reps[name][q].append(values[q])
    reps = {name: {q: np.vstack(v) for q, v in r.items()} for name, r in reps.items()}

    records = []
    for name in arrays:
        series = [(label, metric, point[name][(label, metric)], reps[name][(label, metric)])
                  for label, metric in QUANTITIES]
        for gap, metric, a, b in GAPS:
            series.append((gap, metric, point[name][(a, metric)] - point[name][(b, metric)],
                           reps[name][(a, metric)] - reps[name][(b, metric)]))
        for label, metric, pt, rp in series:
            for d, (subgroup, level, role) in enumerate(domains):
                reportable = role == "overall" or n_undx[d] >= min_undx
                invalid = int((~np.isfinite(rp[:, d])).sum())
                status = ("not_reportable_small_undx" if not reportable
                          else "undefined_point" if not np.isfinite(pt[d])
                          else "invalid_bootstrap_replicates" if invalid else "ok")
                info = summarise(pt[d], rp[:, d], dfree, status)
                records.append(dict(
                    period=period, period_label=PERIOD_LABELS[period], model=name,
                    subgroup=subgroup, level=level, role=role, label_or_gap=label,
                    metric=metric, n_rows=int(n_rows[d]), n_diag=int(n_diag[d]),
                    n_undx=int(n_undx[d]), reportable=bool(reportable), status=status,
                    invalid_replicates=invalid, design_df=dfree, n_reps=n_reps,
                    threshold=bundle.thresholds[name], **info))
    return pd.DataFrame.from_records(records)


def evaluate(df, bundle, n_reps=500, seed=2026, min_undx=30, periods=None):
    """model_eval table for all periods, with the label_gap BH family."""
    if n_reps < 2:
        raise ValueError("Need at least two replicates")
    required = {"period", "cycle", "stratum", "psu", "w", "SEQN", "adult",
                "y_diag", "y_bio", "undx"}
    if required - set(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    periods = periods or tuple(PERIOD_LABELS)
    domains = domain_list(df)
    frames = [evaluate_period(df, bundle, period, domains, n_reps, seed + i, min_undx)
              for i, period in enumerate(periods)]
    result = pd.concat(frames, ignore_index=True)

    is_gap = result["label_or_gap"].str.startswith("gap_")
    result.loc[~is_gap, "p_value"] = np.nan  # single-label rows are not hypotheses
    in_family = (result["period"].isin(TEST_PERIODS)
                 & result["role"].isin(["overall", "primary"])
                 & result["status"].eq("ok")
                 & in_set(result, "label_or_gap", LABEL_GAP_FAMILY))
    result["family"] = np.where(in_family, "label_gap", "")
    result["q_bh"] = np.nan
    result.loc[in_family, "q_bh"] = bh_adjust(result.loc[in_family, "p_value"].to_numpy())
    result["ci_excludes_zero"] = (is_gap & result["status"].eq("ok")
                                  & ((result["ci_low"] > 0) | (result["ci_high"] < 0)))
    result["bh_significant_005"] = result["q_bh"].lt(0.05)
    return result


def temporal_change(result):
    """Change from the 2011-2016 out-of-fold baseline to each test period.

    The periods are separate samples (different cycles and PSUs), so
    SE = sqrt(SE_baseline^2 + SE_test^2), with the smaller design df.
    """
    single = result[~result["label_or_gap"].str.startswith("gap_")]
    keys = ["model", "subgroup", "level", "role", "label_or_gap", "metric"]
    base = single[single["period"] == "train_cv"].set_index(keys)
    rows = []
    for period in TEST_PERIODS:
        test = single[single["period"] == period].set_index(keys)
        for key, t in test.iterrows():
            if key not in base.index:
                continue
            b = base.loc[key]
            ok = b["status"] == "ok" and t["status"] == "ok"
            row = dict(zip(keys, key), period=period, period_label=PERIOD_LABELS[period],
                       baseline=b["estimate"], baseline_se=b["se"],
                       test=t["estimate"], test_se=t["se"],
                       change=t["estimate"] - b["estimate"],
                       n_undx_baseline=b["n_undx"], n_undx_test=t["n_undx"],
                       status="ok" if ok else "not_reportable")
            if ok:
                se = float(np.hypot(b["se"], t["se"]))
                dof = int(min(b["design_df"], t["design_df"]))
                crit = float(student_t.ppf(0.975, dof))
                row.update(se=se, design_df=dof, ci_low=row["change"] - crit * se,
                           ci_high=row["change"] + crit * se,
                           p_value=float(2 * student_t.sf(abs(row["change"]) / se, dof))
                           if se > 0 else np.nan)
            rows.append(row)
    out = pd.DataFrame(rows)
    in_family = (out["role"].isin(["overall", "primary"]) & out["status"].eq("ok")
                 & in_set(out, "label_or_gap", TEMPORAL_FAMILY))
    out["family"] = np.where(in_family, "temporal", "")
    out["q_bh"] = np.nan
    out.loc[in_family, "q_bh"] = bh_adjust(out.loc[in_family, "p_value"].to_numpy())
    out["bh_significant_005"] = out["q_bh"].lt(0.05)
    out["ci_excludes_zero"] = out["status"].eq("ok") & ((out["ci_low"] > 0) | (out["ci_high"] < 0))
    order = ["period", "period_label", "model", "subgroup", "level", "role", "label_or_gap",
             "metric", "baseline", "baseline_se", "test", "test_se", "change", "se", "design_df",
             "ci_low", "ci_high", "p_value", "family", "q_bh", "bh_significant_005",
             "ci_excludes_zero", "n_undx_baseline", "n_undx_test", "status"]
    return out[order]


def main():
    parser = argparse.ArgumentParser(description="Evaluate the frozen RQ2 models")
    parser.add_argument("--csv", default="nhanes_out/nhanes_all.csv")
    parser.add_argument("--models", default="results/rq2_models.joblib")
    parser.add_argument("--out", default="results")
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-undx", type=int, default=30)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    bundle = joblib.load(args.models)
    result = evaluate(df, bundle, n_reps=args.reps, seed=args.seed, min_undx=args.min_undx)
    temporal = temporal_change(result)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "model_eval.csv", index=False)
    temporal.to_csv(out / "model_temporal.csv", index=False)

    print(f"Saved {len(result)} rows to {out / 'model_eval.csv'}")
    print(f"Saved {len(temporal)} rows to {out / 'model_temporal.csv'}")
    print("Status counts:", result["status"].value_counts().to_dict())
    fam = result[result["family"] == "label_gap"]
    print(f"Label-gap family: {len(fam)} tests, {int(fam['bh_significant_005'].sum())} with q < 0.05")
    tfam = temporal[temporal["family"] == "temporal"]
    print(f"Temporal family: {len(tfam)} tests, {int(tfam['bh_significant_005'].sum())} with q < 0.05")

    view = result[(result["subgroup"] == "all")
                  & result["metric"].isin(["auroc", "sensitivity"])
                  & ~result["label_or_gap"].str.startswith("gap_")]
    print("\nOverall AUROC and sensitivity:")
    print(view.pivot_table(index=["model", "metric", "label_or_gap"], columns="period_label",
                           values="estimate").round(3).to_string())


if __name__ == "__main__":
    main()
