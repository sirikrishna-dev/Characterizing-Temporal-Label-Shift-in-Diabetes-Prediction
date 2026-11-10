"""Shared evaluation harness for every upgrade workstream.

Every new model, screening rule and monitor is scored here, with the metric
code in src/model_eval.py (ModelArrays, domain_metrics) and the Rao-Wu
replicates in src/survey_boot.py. Nobody writes a second AUROC or bootstrap.

Shared rules implemented here
    * matched flag rate: the weighted share of 2011-2016 adults that the
      y_diag logistic model flags on its out-of-fold predictions at its
      Youden threshold. Every method also gets a threshold that flags that
      same share on its own 2011-2016 out-of-fold predictions.
    * one output schema (SCHEMA below), the one monitor.py already joins on.
    * paired comparisons: every method in one call is scored on the same
      replicate weights, so differences between methods keep their covariance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from src.models import analysis_domain
from src.model_eval import ModelArrays, domain_metrics, bh_adjust, PRIMARY_SUBGROUPS
from src.survey_boot import rao_wu_replicate_weights, design_degrees_of_freedom, bootstrap_t_ci

SEED = 2026
PERIODS = {"train": "2011-2016", "test_P": "2017-2020", "test_L": "2021-2023"}
TEST_PERIODS = ("test_P", "test_L")
SCHEMA = ["period_label", "subgroup", "level", "role", "method", "label_or_gap", "metric",
          "estimate", "se", "ci_low", "ci_high", "p_value", "family", "q_bh", "status",
          "n_rows", "n_undx"]
LABELS = ("y_diag", "y_bio", "undx")

# What every method is scored on. (label, metric) as in model_eval.QUANTITIES.
CORE = [("y_bio", "auroc"), ("y_diag", "auroc"), ("y_diag", "sensitivity"),
        ("y_bio", "sensitivity"), ("undx", "sensitivity"), ("y_bio", "specificity"),
        ("y_bio", "calibration_in_large")]


def load(csv="nhanes_out/nhanes_all.csv"):
    return pd.read_csv(csv, low_memory=False)


def period_frame(df, period, weight_col="w"):
    """All rows of one period (every PSU stays in the design), weight 0 where
    missing, index reset so arrays line up with the replicate weights."""
    full = df.loc[df["period"] == period].copy().reset_index(drop=True)
    if full.empty:
        raise ValueError(f"No rows for period {period}")
    full["w"] = pd.to_numeric(full[weight_col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan).fillna(0.0)
    return full


CYCLE_YEARS = {"train": 6.0, "test_P": 3.2, "test_L": 2.0}


def pooled_frame(df, periods, weight_col="w"):
    """Several periods pooled into one sample, each period's weight scaled by
    its years over the total years (CDC pooling rule). Strata are already
    unique per cycle, so the design stays valid."""
    total = sum(CYCLE_YEARS[p] for p in periods)
    parts = []
    for p in periods:
        f = period_frame(df, p, weight_col)
        f["w"] = f["w"] * CYCLE_YEARS[p] / total
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def label_arrays(full):
    analysis = analysis_domain(full).to_numpy()
    out = {}
    for name in LABELS:
        y = pd.to_numeric(full[name], errors="coerce").to_numpy(dtype=float)
        out[name] = np.where(analysis, np.nan_to_num(y), 0.0)
    return out, analysis


def primary_domains(full, subgroups=PRIMARY_SUBGROUPS):
    """[(subgroup, level, role, mask)] for overall plus the primary subgroups."""
    analysis = analysis_domain(full).to_numpy()
    out = [("all", "all", "overall", analysis)]
    for col in subgroups:
        values = full[col].astype("string")
        for level in sorted(values[analysis].dropna().unique()):
            out.append((col, str(level), "primary",
                        analysis & values.eq(level).fillna(False).to_numpy(dtype=bool)))
    return out


def weighted_share(flag, w):
    w = np.asarray(w, float)
    return float(np.dot(w, np.asarray(flag, float)) / w.sum())


def threshold_for_share(p, w, share):
    """Lowest threshold whose weighted flagged share does not exceed `share`
    (scores >= threshold are flagged). Ties are kept together."""
    p = np.asarray(p, float)
    w = np.asarray(w, float)
    ok = np.isfinite(p) & (w > 0)
    p, w = p[ok], w[ok]
    order = np.argsort(-p, kind="mergesort")
    ps, cw = p[order], np.cumsum(w[order]) / w.sum()
    # last distinct score value at which cumulative share <= share
    distinct_end = np.r_[ps[1:] != ps[:-1], True]
    candidates = np.flatnonzero(distinct_end & (cw <= share + 1e-12))
    if len(candidates) == 0:
        return float(ps[0]) + 1e-12
    return float(ps[candidates[-1]])


def reference_flag_share(bundle, df):
    """Weighted share of 2011-2016 adults flagged by the y_diag logistic model
    (out-of-fold predictions, Youden threshold). This is the matched rate."""
    rows = df.loc[(df["period"] == "train") & analysis_domain(df)]
    p = rows["SEQN"].map(bundle.oof.set_index("SEQN")["logistic"]).to_numpy(float)
    return weighted_share(p >= bundle.thresholds["logistic"], rows["w"].to_numpy(float))


def _status(role, n_undx, point, reps, min_undx):
    if role != "overall" and n_undx < min_undx:
        return "not_reportable_small_undx"
    if not np.isfinite(point):
        return "undefined_point"
    if not np.isfinite(reps).all():
        return "invalid_bootstrap_replicates"
    return "ok"


def _summ(point, reps, dof, status):
    if status != "ok":
        return dict(estimate=point, se=np.nan, ci_low=np.nan, ci_high=np.nan, p_value=np.nan)
    ci = bootstrap_t_ci(point, reps, dof)
    return {k: ci[k] for k in ("estimate", "se", "ci_low", "ci_high", "p_value")}


def score_methods(full, scores, thresholds, period_label, n_reps=500, seed=SEED,
                  baseline=None, min_undx=30, domains=None, compare=None):
    """Score several methods on one period with shared replicate weights.

    full        rows of one period (from period_frame)
    scores      {method: risk score aligned with full (nan outside analysis)}
    thresholds  {method: score threshold, flagged = score >= threshold}
    baseline    method name; every other method also gets paired differences
                (method - baseline) labelled diff_vs_<baseline>
    compare     optional list of (method_a, method_b) for extra paired diffs

    Returns a DataFrame in SCHEMA (family/q_bh left empty for the caller).
    """
    labels, analysis = label_arrays(full)
    domains = domains or primary_domains(full)
    masks = np.column_stack([d[3] for d in domains])
    w = full["w"].to_numpy(float)
    dof = design_degrees_of_freedom(full)
    n_rows = masks.sum(0)
    n_undx = (masks * labels["undx"][:, None]).sum(0)

    arrays = {}
    for m, p in scores.items():
        p = np.where(analysis, p, np.nan)
        if not np.isfinite(p[analysis]).all():
            raise ValueError(f"{m}: missing scores on analysis rows")
        arrays[m] = ModelArrays(p, thresholds[m], labels)

    def compute(W):
        out = {}
        tot = W.sum(0)
        for m, a in arrays.items():
            vals = domain_metrics(a, W)
            d = {q: vals[q] for q in CORE}
            d[("gap_diag_minus_undx", "sensitivity")] = vals[("y_diag", "sensitivity")] - vals[("undx", "sensitivity")]
            with np.errstate(invalid="ignore", divide="ignore"):
                d[("flagged", "share")] = (W.T @ a.flag) / tot
            out[m] = d
        return out

    point = compute(masks * w[:, None])
    keys = list(next(iter(point.values())).keys())
    reps = {m: {k: [] for k in keys} for m in arrays}
    for rw in rao_wu_replicate_weights(full, n_reps=n_reps, seed=seed):
        vals = compute(masks * rw[:, None])
        for m in arrays:
            for k in keys:
                reps[m][k].append(vals[m][k])
    reps = {m: {k: np.vstack(v) for k, v in r.items()} for m, r in reps.items()}

    series = []  # (method, label_or_gap, metric, point array, reps matrix)
    for m in arrays:
        for k in keys:
            series.append((m, k[0], k[1], point[m][k], reps[m][k]))
    pairs = []
    if baseline is not None:
        pairs += [(m, baseline) for m in arrays if m != baseline]
    pairs += list(compare or [])
    for a, b in pairs:
        for k in keys:
            series.append((a, f"diff_vs_{b}:{k[0]}", k[1],
                           point[a][k] - point[b][k], reps[a][k] - reps[b][k]))

    records = []
    for m, lab, met, pt, rp in series:
        for j, (sg, lv, role, _) in enumerate(domains):
            st = _status(role, n_undx[j], pt[j], rp[:, j], min_undx)
            records.append(dict(period_label=period_label, subgroup=sg, level=lv, role=role,
                                method=m, label_or_gap=lab, metric=met,
                                **_summ(float(pt[j]), rp[:, j], dof, st),
                                family="", q_bh=np.nan, status=st,
                                n_rows=int(n_rows[j]), n_undx=int(n_undx[j])))
    out = pd.DataFrame.from_records(records)
    # p-values only mean something for differences
    out.loc[~out["label_or_gap"].str.startswith("diff_vs_"), "p_value"] = np.nan
    return out[SCHEMA]


def add_family(frame, name, mask):
    """Mark the pre-registered rows as one BH family and fill q_bh."""
    mask = mask & frame["status"].eq("ok") & frame["p_value"].notna()
    frame.loc[mask, "family"] = name
    frame.loc[mask, "q_bh"] = bh_adjust(frame.loc[mask, "p_value"].to_numpy())
    return frame


def t_summary(point, se, dof, null=0.0):
    crit = student_t.ppf(0.975, dof)
    p = 2 * student_t.sf(abs((point - null) / se), dof) if se > 0 else np.nan
    return dict(estimate=point, se=se, ci_low=point - crit * se, ci_high=point + crit * se, p_value=p)
