"""
survey_drift.py

Survey-weighted drift metrics for NHANES (team Epoch, ADA project).

Why this file exists
--------------------
NHANES is not a simple random sample. Each cycle oversamples some groups
on purpose, and the oversampling plan changed in August 2021-2023 (no more
oversampling by race/Hispanic origin and income, all ages 0-19 and 60+
selected). If you compare raw, unweighted rows across cycles, part of the
"drift" you see is just the change in who CDC chose to sample. This module
makes every drift and performance number design-aware:

  1. every estimate uses the correct CDC sample weight for that cycle
  2. every uncertainty uses the strata and PSUs (SDMVSTRA, SDMVPSU)
  3. a shift is only called real if it is bigger than the sampling noise
     you would see between two samples drawn from the SAME period

Rules the team must follow
--------------------------
* Use the weight of the smallest subsample in the analysis.
    - HbA1c (blood) models: MEC exam weight for G/H/I/P, phlebotomy weight
      WTPH2YR for L (2021-23 introduced it; it is 0 for people with no blood
      specimen).
    - Anything using fasting glucose: fasting subsample weight
      (WTSAF2YR for G/H/I/L, WTSAFPRP for P). Checked against the
      P_GLU and GLU_L codebooks.
    - WTMECPRP is the MEC weight name expected in P_DEMO. Open the P_DEMO
      codebook and confirm before running anything.
* Build the bootstrap replicates on the FULL cycle sample, then subset to
  subgroups. Never drop rows (e.g. "only adults", "only low income") before
  building replicates, or the variance will be wrong.
* Do not split 2017-March 2020 (P) into 2017-18 and 2019-20. CDC says there
  are no valid weights for 2019-March 2020 alone.
* When pooling cycles into one period (e.g. training = G+H+I), rescale the
  weights by cycle_years / total_years (pool_weights below).

Cycle keys used here
--------------------
G = 2011-12, H = 2013-14, I = 2015-16, P = 2017-March 2020, L = Aug 2021-2023
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# --------------------------------------------------------------------------
# 1. Weights and design variables
# --------------------------------------------------------------------------

# Weight for analyses that use exam or blood (HbA1c) variables.
EXAM_OR_BLOOD_WEIGHT = {
    "G": "WTMEC2YR",
    "H": "WTMEC2YR",
    "I": "WTMEC2YR",
    "P": "WTMECPRP",   # confirm in P_DEMO codebook
    "L": "WTPH2YR",    # 2021-23 phlebotomy weight, use for HbA1c
}

# Weight for analyses that use fasting glucose (fasting subsample).
FASTING_WEIGHT = {
    "G": "WTSAF2YR",
    "H": "WTSAF2YR",
    "I": "WTSAF2YR",
    "P": "WTSAFPRP",   # verified in P_GLU codebook
    "L": "WTSAF2YR",   # verified in GLU_L codebook
}

CYCLE_YEARS = {
    "G": 2.0, "H": 2.0, "I": 2.0, "P": 3.2, "L": 2.0,
    # the same cycles under the labels used in the team's cleaned file
    "2011-2012": 2.0, "2013-2014": 2.0, "2015-2016": 2.0,
    "2017-2020": 3.2, "2021-2023": 2.0,
}


def attach_design(df: pd.DataFrame, cycle: str, fasting: bool = False) -> pd.DataFrame:
    """Add columns w, stratum, psu, cycle to one cycle's merged dataframe.

    df must already contain SDMVSTRA, SDMVPSU and the needed weight column
    (merged on SEQN from DEMO / GHB / GLU files).
    Rows with weight 0 or missing are dropped: CDC sets the weight to 0 for
    people outside the subsample, and they must not enter weighted estimates.
    """
    table = FASTING_WEIGHT if fasting else EXAM_OR_BLOOD_WEIGHT
    wcol = table[cycle]
    for col in (wcol, "SDMVSTRA", "SDMVPSU"):
        if col not in df.columns:
            raise KeyError(
                f"Column {col} not found for cycle {cycle}. "
                f"Merge the right DEMO/lab file or fix the weight name."
            )
    out = df.copy()
    out["cycle"] = cycle
    out["w"] = out[wcol].astype(float)
    # Strata numbers are made unique across cycles by prefixing the cycle;
    # PSU ids (1, 2, ...) only mean something inside their stratum.
    out["stratum"] = cycle + "_" + out["SDMVSTRA"].astype(int).astype(str)
    out["psu"] = out["stratum"] + "_" + out["SDMVPSU"].astype(int).astype(str)
    keep = out["w"].notna() & (out["w"] > 0)
    return out.loc[keep].reset_index(drop=True)


def pool_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Rescale weights when several cycles are pooled into one period.

    Each cycle's weight is multiplied by cycle_years / total_years so the
    pooled weights describe the average population over the whole period.
    Example: G+H+I -> each weight * 2/6. I+P -> 2/5.2 and 3.2/5.2.
    """
    cycles = sorted(df["cycle"].unique())
    total = sum(CYCLE_YEARS[c] for c in cycles)
    out = df.copy()
    out["w"] = out["w"] * out["cycle"].map(lambda c: CYCLE_YEARS[c] / total)
    return out


def design_dof(df: pd.DataFrame) -> int:
    """Design degrees of freedom = number of PSUs - number of strata."""
    return int(df["psu"].nunique() - df["stratum"].nunique())


def rao_wu_replicates(df: pd.DataFrame, n_rep: int = 500, seed: int = 0) -> np.ndarray:
    """Rao-Wu rescaling bootstrap replicate weights.

    In each stratum with n_h PSUs, draw n_h - 1 PSUs with replacement.
    A PSU drawn r times gets weight multiplied by r * n_h / (n_h - 1);
    PSUs not drawn get 0. Returns an array of shape (n_rows, n_rep).

    Call this on one full period (all rows that carry a positive weight),
    then subset rows for subgroups.
    """
    rng = np.random.default_rng(seed)
    w = df["w"].to_numpy(float)
    reps = np.empty((len(df), n_rep))
    strata = df["stratum"].to_numpy()
    psus = df["psu"].to_numpy()

    groups = []
    for s in np.unique(strata):
        rows_s = np.flatnonzero(strata == s)
        psu_ids, psu_index = np.unique(psus[rows_s], return_inverse=True)
        groups.append((rows_s, len(psu_ids), psu_index))
        if len(psu_ids) < 2:
            warnings.warn(
                f"Stratum {s} has a single PSU; its weights are left unchanged "
                f"in every replicate. Collapse it with a neighbouring stratum."
            )

    for b in range(n_rep):
        factor = np.ones(len(df))
        for rows_s, n_h, psu_index in groups:
            if n_h < 2:
                continue
            draws = rng.integers(0, n_h, size=n_h - 1)
            r = np.bincount(draws, minlength=n_h)
            factor[rows_s] = (r * n_h / (n_h - 1))[psu_index]
        reps[:, b] = w * factor
    return reps


# --------------------------------------------------------------------------
# 2. Weighted building blocks
# --------------------------------------------------------------------------

def _clean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return x[ok], w[ok]


def wmean(x, w) -> float:
    x, w = _clean(x, w)
    return float(np.sum(w * x) / np.sum(w))


def wquantile(x, w, qs) -> np.ndarray:
    """Weighted quantiles (inverse of the weighted ECDF)."""
    x, w = _clean(x, w)
    order = np.argsort(x)
    x, w = x[order], w[order]
    cdf = np.cumsum(w) / np.sum(w)
    idx = np.searchsorted(cdf, np.asarray(qs, float), side="left")
    return x[np.clip(idx, 0, len(x) - 1)]


def _wecdf(x, w, grid) -> np.ndarray:
    x, w = _clean(x, w)
    order = np.argsort(x)
    x, w = x[order], w[order]
    cdf = np.concatenate([[0.0], np.cumsum(w) / np.sum(w)])
    return cdf[np.searchsorted(x, grid, side="right")]


# --------------------------------------------------------------------------
# 3. Weighted drift metrics (feature / label distribution)
# --------------------------------------------------------------------------

def weighted_psi(ref_x, ref_w, new_x, new_w, n_bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index with weighted bin shares.

    Bin edges are weighted quantiles of the reference period, so each bin
    holds about 1/n_bins of the reference POPULATION (not of the sample).
    """
    ref_x, ref_w = _clean(ref_x, ref_w)
    new_x, new_w = _clean(new_x, new_w)
    inner = np.unique(wquantile(ref_x, ref_w, np.linspace(0, 1, n_bins + 1)[1:-1]))
    rb = np.searchsorted(inner, ref_x, side="right")
    nb = np.searchsorted(inner, new_x, side="right")
    k = len(inner) + 1
    p = np.bincount(rb, weights=ref_w, minlength=k) / ref_w.sum()
    q = np.bincount(nb, weights=new_w, minlength=k) / new_w.sum()
    p, q = p + eps, q + eps
    return float(np.sum((q - p) * np.log(q / p)))


def weighted_categorical_psi(ref_c, ref_w, new_c, new_w, eps: float = 1e-6) -> float:
    """PSI for a categorical column (race/ethnicity, income band, insurance).

    Fast path: if both inputs are non-negative integer codes (e.g. from
    pd.factorize over both periods), shares are computed with bincount.
    """
    ref_c, new_c = np.asarray(ref_c), np.asarray(new_c)
    if np.issubdtype(ref_c.dtype, np.integer) and np.issubdtype(new_c.dtype, np.integer):
        k = int(max(ref_c.max(initial=0), new_c.max(initial=0))) + 1
        p = np.bincount(ref_c, weights=np.asarray(ref_w, float), minlength=k)
        q = np.bincount(new_c, weights=np.asarray(new_w, float), minlength=k)
        p, q = p / p.sum() + eps, q / q.sum() + eps
        return float(np.sum((q - p) * np.log(q / p)))
    ref_c = pd.Series(ref_c).astype(str).to_numpy()
    new_c = pd.Series(new_c).astype(str).to_numpy()
    ref_w = np.asarray(ref_w, float)
    new_w = np.asarray(new_w, float)
    cats = np.union1d(ref_c, new_c)
    p = np.array([ref_w[ref_c == c].sum() for c in cats]) / ref_w.sum()
    q = np.array([new_w[new_c == c].sum() for c in cats]) / new_w.sum()
    p, q = p + eps, q + eps
    return float(np.sum((q - p) * np.log(q / p)))


def weighted_ks(ref_x, ref_w, new_x, new_w) -> float:
    """Kolmogorov-Smirnov distance between the two weighted ECDFs."""
    ref_x, ref_w = _clean(ref_x, ref_w)
    new_x, new_w = _clean(new_x, new_w)
    grid = np.union1d(ref_x, new_x)
    return float(np.max(np.abs(_wecdf(ref_x, ref_w, grid) - _wecdf(new_x, new_w, grid))))


def weighted_wasserstein(ref_x, ref_w, new_x, new_w) -> float:
    ref_x, ref_w = _clean(ref_x, ref_w)
    new_x, new_w = _clean(new_x, new_w)
    return float(stats.wasserstein_distance(ref_x, new_x, u_weights=ref_w, v_weights=new_w))


def weighted_prevalence_diff(ref_y, ref_w, new_y, new_w) -> float:
    """new prevalence - reference prevalence (label shift on one label)."""
    return wmean(new_y, new_w) - wmean(ref_y, ref_w)


def weighted_domain_auc(ref_X, ref_w, new_X, new_w, n_splits: int = 5, seed: int = 0) -> float:
    """Domain classifier: can a model tell the two periods apart?

    Each period's weights are normalised to sum to 1, so the classifier sees
    two equally sized POPULATIONS. AUC near 0.5 means no detectable
    multivariate shift. Out-of-fold predictions, weighted AUC.
    """
    ref_X = np.asarray(ref_X, float)
    new_X = np.asarray(new_X, float)
    X = np.vstack([ref_X, new_X])
    d = np.r_[np.zeros(len(ref_X)), np.ones(len(new_X))]
    w = np.r_[np.asarray(ref_w, float) / np.sum(ref_w), np.asarray(new_w, float) / np.sum(new_w)]
    w = w / w.mean()
    oof = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, d):
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
        clf.fit(X[tr], d[tr], sample_weight=w[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(d, oof, sample_weight=w))


# --------------------------------------------------------------------------
# 4. Weighted model performance
# --------------------------------------------------------------------------

def weighted_auroc(y, p, w) -> float:
    return float(roc_auc_score(np.asarray(y), np.asarray(p), sample_weight=np.asarray(w)))


def weighted_brier(y, p, w) -> float:
    y, p, w = np.asarray(y, float), np.asarray(p, float), np.asarray(w, float)
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def calibration_in_the_large(y, p, w) -> float:
    """Weighted observed rate - weighted mean prediction. 0 is perfect."""
    return wmean(y, w) - wmean(p, w)


def weighted_ece(y, p, w, n_bins: int = 10) -> float:
    y, p, w = np.asarray(y, float), np.asarray(p, float), np.asarray(w, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    total = w.sum()
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if w[m].sum() == 0:
            continue
        ece += (w[m].sum() / total) * abs(np.sum(w[m] * y[m]) / w[m].sum()
                                          - np.sum(w[m] * p[m]) / w[m].sum())
    return float(ece)


# --------------------------------------------------------------------------
# 5. Uncertainty and the "is this shift real?" test
# --------------------------------------------------------------------------

def _rep_values(fn, args, reps):
    return np.array([fn(args, reps[:, b]) for b in range(reps.shape[1])])


def one_sample_ci(stat_fn, args, w_full, w_reps, dof: int, alpha: float = 0.05) -> dict:
    """Design-based estimate and CI for one period.

    stat_fn(args, w) -> float, e.g. lambda a, w: wmean(a, w)
    SE = standard deviation of the statistic across replicate weights,
    CI uses a t distribution with the design degrees of freedom.
    """
    est = stat_fn(args, w_full)
    se = float(np.std(_rep_values(stat_fn, args, w_reps), ddof=1))
    t = stats.t.ppf(1 - alpha / 2, dof)
    return {"estimate": est, "se": se, "ci_low": est - t * se, "ci_high": est + t * se, "dof": dof}


def difference_test(stat_fn, ref_args, ref_w, ref_reps, ref_dof,
                    new_args, new_w, new_reps, new_dof, alpha: float = 0.05) -> dict:
    """Test a signed difference (prevalence change, AUROC drop, mean HbA1c shift).

    Cycles are independent samples, so SE_diff = sqrt(SE_ref^2 + SE_new^2).
    Uses the smaller design dof (conservative). The shift is called real when
    the CI excludes 0.
    """
    r = one_sample_ci(stat_fn, ref_args, ref_w, ref_reps, ref_dof, alpha)
    n = one_sample_ci(stat_fn, new_args, new_w, new_reps, new_dof, alpha)
    diff = n["estimate"] - r["estimate"]
    se = float(np.hypot(r["se"], n["se"]))
    dof = min(ref_dof, new_dof)
    t = stats.t.ppf(1 - alpha / 2, dof)
    lo, hi = float(diff - t * se), float(diff + t * se)
    p = float(2 * stats.t.sf(abs(diff) / se, dof)) if se > 0 else float("nan")
    return {"ref": r["estimate"], "new": n["estimate"], "diff": diff, "se": se,
            "ci_low": lo, "ci_high": hi, "p_value": p, "dof": dof,
            "real_shift": bool(lo > 0 or hi < 0)}


def two_period_drift(metric, ref_args, ref_w, ref_reps,
                     new_args, new_w, new_reps, alpha: float = 0.05) -> dict:
    """For distance metrics (PSI, KS, Wasserstein, domain AUC) that are never
    negative and always show SOME drift because of sampling noise.

    noise floor: the (1 - alpha) quantile of the metric computed between two
    bootstrap replicates of the SAME period. Done for both periods and the
    larger floor is used, because the 2021-23 sample is smaller and noisier.
    The shift is called real only if the observed metric is above the floor.

    metric(ref_args, ref_w, new_args, new_w) -> float
    """
    observed = metric(ref_args, ref_w, new_args, new_w)
    B = min(ref_reps.shape[1], new_reps.shape[1])
    half = B // 2

    def null_for(args, reps):
        return np.array([metric(args, reps[:, b], args, reps[:, (b + half) % B]) for b in range(half)])

    null_ref = null_for(ref_args, ref_reps)
    null_new = null_for(new_args, new_reps)
    floor_ref = float(np.quantile(null_ref, 1 - alpha))
    floor_new = float(np.quantile(null_new, 1 - alpha))
    noise_floor = max(floor_ref, floor_new)
    # p-value from a gamma fitted (method of moments) to the noisier null.
    # The fit lets small p-values be estimated beyond what B/2 null draws can
    # resolve, which multiple-testing corrections need.
    null = null_ref if null_ref.mean() >= null_new.mean() else null_new
    p_value = gamma_tail_p(observed, null)
    boot = np.array([metric(ref_args, ref_reps[:, b], new_args, new_reps[:, b]) for b in range(B)])
    return {"observed": observed,
            "noise_floor": noise_floor,
            "noise_floor_ref": floor_ref,
            "noise_floor_new": floor_new,
            "p_value": p_value,
            "ci_low": float(np.quantile(boot, alpha / 2)),
            "ci_high": float(np.quantile(boot, 1 - alpha / 2)),
            "real_shift": bool(observed > noise_floor)}


def gamma_tail_p(observed: float, null) -> float:
    """P(null >= observed) under a gamma fitted to non-negative null distances."""
    null = np.asarray(null, float)
    m, v = null.mean(), null.var(ddof=1)
    if not np.isfinite(observed):
        return float("nan")
    if m <= 0 or v <= 0:
        return 1.0 if observed <= m else 0.0
    shape, scale = m * m / v, v / m
    return float(stats.gamma.sf(observed, shape, scale=scale))


def replicate_auc_test(metric, ref_args, ref_reps, new_args, new_reps, observed,
                       null_value: float = 0.5, alpha: float = 0.05) -> dict:
    """For a domain-classifier AUC: the null is 'cannot tell the periods
    apart', i.e. AUC = 0.5. SE from paired replicates, normal test."""
    B = min(ref_reps.shape[1], new_reps.shape[1])
    boot = np.array([metric(ref_args, ref_reps[:, b], new_args, new_reps[:, b]) for b in range(B)])
    se = float(np.std(boot, ddof=1))
    z = (observed - null_value) / se if se > 0 else float("inf")
    p = float(stats.norm.sf(z))
    zc = stats.norm.ppf(1 - alpha / 2)
    return {"observed": observed, "se": se,
            "ci_low": observed - zc * se, "ci_high": observed + zc * se,
            "p_value": p, "real_shift": bool(p < alpha)}


# --------------------------------------------------------------------------
# 6. The table your reviewer wants: weighted vs unweighted, side by side
# --------------------------------------------------------------------------

def drift_report(ref: pd.DataFrame, ref_reps: np.ndarray,
                 new: pd.DataFrame, new_reps: np.ndarray,
                 numeric_cols, categorical_cols=(), alpha: float = 0.05) -> pd.DataFrame:
    """One row per feature: unweighted PSI, weighted PSI, noise floor, verdict.

    Rows where unweighted says "drift" but weighted says "no real shift" are
    the sampling artifacts the reviewer warned about. Report them.

    ref, new: outputs of attach_design (and pool_weights if pooled).
    ref_reps / new_reps: rao_wu_replicates of the same dataframes.
    To run on a subgroup, pass ref[mask], ref_reps[mask.to_numpy()] etc.
    """
    rows = []
    specs = [(c, "numeric") for c in numeric_cols] + [(c, "categorical") for c in categorical_cols]
    for col, kind in specs:
        if kind == "numeric":
            rm = ref[col].notna().to_numpy()
            nm = new[col].notna().to_numpy()
            rx, nx = ref[col].to_numpy(float)[rm], new[col].to_numpy(float)[nm]
            metric = lambda a, aw, b, bw: weighted_psi(a, aw, b, bw)
        else:
            rm = ref[col].notna().to_numpy()
            nm = new[col].notna().to_numpy()
            rx, nx = ref[col].to_numpy()[rm], new[col].to_numpy()[nm]
            metric = lambda a, aw, b, bw: weighted_categorical_psi(a, aw, b, bw)
        unweighted = metric(rx, np.ones(len(rx)), nx, np.ones(len(nx)))
        res = two_period_drift(metric, rx, ref["w"].to_numpy()[rm], ref_reps[rm],
                               nx, new["w"].to_numpy()[nm], new_reps[nm], alpha)
        rows.append({"feature": col, "type": kind,
                     "psi_unweighted": unweighted,
                     "psi_weighted": res["observed"],
                     "noise_floor": res["noise_floor"],
                     "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                     "real_shift": res["real_shift"],
                     "artifact": bool(unweighted > 0.1 and not res["real_shift"])})
    return pd.DataFrame(rows)


def min_cell_check(df: pd.DataFrame, group_col: str, min_n: int = 30, min_psu: int = 12) -> pd.DataFrame:
    """Flag subgroups too small for stable design-based estimates.

    Thresholds are rough rules of thumb; also check the NCHS data
    presentation standards for proportions before reporting a cell.
    """
    g = df.groupby(group_col).agg(n=("w", "size"), psus=("psu", "nunique"),
                                  strata=("stratum", "nunique"))
    g["dof"] = g["psus"] - g["strata"]
    g["reportable"] = (g["n"] >= min_n) & (g["psus"] >= min_psu)
    return g.reset_index()
