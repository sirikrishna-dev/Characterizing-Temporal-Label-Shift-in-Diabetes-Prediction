"""Rao-Wu rescaled bootstrap for NHANES survey-weighted statistics.

IMPORTANT: pass all rows of the relevant full NHANES cycle(s) when
constructing replicates. Apply adult/lab/subgroup domain masks AFTERWARD
by setting weights outside the domain to zero.
"""

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def _survey_groups(df, cycle_col="cycle", stratum_col="stratum", psu_col="psu"):
    """Return arrays of row positions belonging to each PSU in each stratum."""
    required = [cycle_col, stratum_col, psu_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing survey design columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError("Cycle, stratum and PSU IDs must not be missing")

    # Group positions, NOT dataframe indexes (indexes may be nonconsecutive).
    design = df[required].reset_index(drop=True)
    groups = []
    for (_, _), stratum_rows in design.groupby([cycle_col, stratum_col], sort=False):
        psus = []
        for _, psu_rows in stratum_rows.groupby(psu_col, sort=False):
            psus.append(psu_rows.index.to_numpy(dtype=np.int64))
        if len(psus) < 2:
            raise ValueError("Found a singleton PSU stratum; cannot use this bootstrap")
        groups.append(psus)
    return groups


def design_degrees_of_freedom(df, cycle_col="cycle", stratum_col="stratum", psu_col="psu"):
    """Design df = number of PSU groups minus number of strata."""
    groups = _survey_groups(df, cycle_col, stratum_col, psu_col)
    return int(sum(len(psus) - 1 for psus in groups))


def rao_wu_replicate_weights(df, n_reps=500, seed=2026, weight_col="w",
                              cycle_col="cycle", stratum_col="stratum", psu_col="psu"):
    """Yield one full-length Rao-Wu rescaled bootstrap weight array per replicate.

    For a stratum with m PSUs, draw m-1 PSUs WITH replacement.
    A sampled PSU's multiplier is (m / (m-1)) * its draw count.
    Repeat this independently for every (cycle, stratum).

    Each yielded weight vector has the same row order as df. No domain
    rows are dropped. The generator avoids allocating a huge B x N matrix.
    """
    if not isinstance(n_reps, (int, np.integer)) or n_reps < 2:
        raise ValueError("n_reps must be an integer >= 2")
    if weight_col not in df:
        raise ValueError(f"Missing weight column: {weight_col}")
    base_w = pd.to_numeric(df[weight_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(base_w).all() or (base_w < 0).any():
        raise ValueError("Base weights must be finite and nonnegative")
    groups = _survey_groups(df, cycle_col, stratum_col, psu_col)
    rng = np.random.default_rng(seed)
    for _ in range(n_reps):
        multipliers = np.zeros(len(df), dtype=float)
        for psus in groups:
            m = len(psus)
            counts = rng.multinomial(m - 1, np.full(m, 1.0 / m))
            for positions, count in zip(psus, counts):
                if count:
                    multipliers[positions] = count * (m / (m - 1))
        yield base_w * multipliers


def domain_weights(weights, domain):
    """Keep the full sample, zeroing weights outside an analysis domain."""
    w = np.asarray(weights, dtype=float)
    mask = np.asarray(domain, dtype=bool)
    if w.ndim != 1 or mask.ndim != 1 or len(w) != len(mask):
        raise ValueError("weights and domain must be one-dimensional and aligned")
    return np.where(mask, w, 0.0)


def bootstrap_t_ci(point, replicate_estimates, design_df, confidence=0.95, null=0.0):
    """Return t-based CI and two-sided p-value for a point metric or paired gap.

    Pass a point estimate computed with ORIGINAL weights, then the same
    metric computed independently with each bootstrap weight vector.
    For a gap, pass (metric_diag - metric_bio) for each replicate, so the
    covariance between metrics is retained.

    Replicates must all be valid. Do not silently discard failed replicates.
    """
    rep = np.asarray(replicate_estimates, dtype=float)
    if rep.ndim != 1 or len(rep) < 2 or not np.isfinite(rep).all():
        raise ValueError("Need >=2 finite replicate estimates; investigate failures")
    if not np.isfinite(point) or design_df <= 0 or not 0 < confidence < 1:
        raise ValueError("Invalid point, design degrees of freedom or confidence")
    # Rao-Wu rescaled bootstrap variance; centered on the original estimate.
    se = float(np.sqrt(np.mean((rep - point) ** 2)))
    critical = float(student_t.ppf((1 + confidence) / 2, df=design_df))
    if se == 0:
        p = 1.0 if point == null else 0.0
    else:
        p = float(2 * student_t.sf(abs((point - null) / se), df=design_df))
    return {"estimate": float(point), "se": se,
            "ci_low": float(point - critical * se),
            "ci_high": float(point + critical * se),
            "p_value": p, "design_df": int(design_df),
            "n_reps": int(len(rep))}
