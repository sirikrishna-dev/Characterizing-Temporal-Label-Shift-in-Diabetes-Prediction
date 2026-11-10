"""Upgrade 2b: how many lab-tested adults per period does an audit need to
detect a 15-point drop in sensitivity among undiagnosed cases?

Target (fixed in advance): detect a drop of DELTA = 0.15 in undiagnosed-case
sensitivity of the frozen y_diag logistic model, 80% power, one-sided
alpha = 0.05 (an alarm only cares about drops; two-sided numbers are also
written).

Analytic
    p0 = the model's undiagnosed sensitivity in 2011-2016 (out-of-fold),
    p1 = p0 - DELTA. The reference p0 is treated as known (it comes from the
    full training lab sample), so the audit is a one-sample test:
        n_cases = deff * [z_a sqrt(p0 q0) + z_b sqrt(p1 q1)]^2 / DELTA^2
    deff (design effect) = survey bootstrap variance / simple-random-sampling
    variance of the undiagnosed sensitivity, from model_eval.csv (median over
    the three periods, per cell).
    lab-tested adults needed = n_cases / (unweighted share of undiagnosed
    cases among lab-tested adults in that cell). For a subgroup we also give
    the total audit size if adults are drawn at random from everyone.

Simulation cross-check
    Repeatedly split 2011-2016 into PSU half-samples. In the deployment half,
    un-flag a random share DELTA / p0 of the flagged undiagnosed cases (a
    15-point drop in expectation), draw a random audit sample of n adults, and
    alarm if (reference sensitivity - audit sensitivity) exceeds the 95th
    percentile of the same statistic with no drop. Power = alarm rate.

Outputs: results/audit_power.csv (analytic), results/audit_power_sim.csv
Run: python -m src.audit_power --reps 1000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.benchmark import prepare
from src.harness import SEED

DELTA = 0.15
POWER = 0.80
ALPHA = 0.05
N_GRID = (500, 1000, 2000, 3000, 4000, 6000, 8000)
CELLS = [("all", "all"), ("sex", "male"), ("sex", "female"), ("age_band2", "20-59"),
         ("age_band2", "60+"), ("insured", "0.0"), ("insured", "1.0"),
         ("income_band2", "<1.3"), ("income_band2", ">=1.3")]


def n_cases(p0, delta, deff, alpha=ALPHA, power=POWER, sided=1):
    p1 = p0 - delta
    za = norm.ppf(1 - alpha / sided)
    zb = norm.ppf(power)
    return deff * (za * np.sqrt(p0 * (1 - p0)) + zb * np.sqrt(p1 * (1 - p1))) ** 2 / delta ** 2


def analytic(model_eval, d, model="logistic"):
    me = model_eval[(model_eval["model"] == model) & (model_eval["label_or_gap"] == "undx")
                    & (model_eval["metric"] == "sensitivity") & model_eval["status"].eq("ok")].copy()
    me["level"] = me["level"].astype(str)
    me["srs_var"] = me["estimate"] * (1 - me["estimate"]) / me["n_undx"]
    me["deff"] = me["se"] ** 2 / me["srs_var"]
    rows = []
    n_adults = len(d)
    for sg, lv in CELLS:
        cell = me[(me["subgroup"] == sg) & (me["level"] == lv)]
        base = cell[cell["period"] == "train_cv"]
        if base.empty:
            continue
        p0 = float(base["estimate"].iloc[0])
        deff = float(cell["deff"].median())
        mask = np.ones(len(d), bool) if sg == "all" else d[sg].astype(str).eq(lv).to_numpy()
        share_undx = float(d.loc[mask, "undx"].mean())
        share_cell = float(mask.mean())
        for sided in (1, 2):
            nc = n_cases(p0, DELTA, deff, sided=sided)
            nc_srs = n_cases(p0, DELTA, 1.0, sided=sided)
            rows.append(dict(subgroup=sg, level=lv, sided=sided, p0=p0, p1=p0 - DELTA, deff=deff,
                             deff_periods=len(cell), undx_share_of_lab_adults=share_undx,
                             cell_share_of_adults=share_cell,
                             cases_needed_srs=int(np.ceil(nc_srs)), cases_needed=int(np.ceil(nc)),
                             lab_adults_needed_in_cell=int(np.ceil(nc / share_undx)),
                             total_audit_if_random=int(np.ceil(nc / share_undx / share_cell)),
                             nhanes_period_lab_adults=n_adults))
    return pd.DataFrame(rows)


def simulate(d, thr, reps=1000, seed=SEED):
    w = d["w"].to_numpy(float) * 2
    p = d["p"].to_numpy(float)
    ux = d["undx"].to_numpy(int)
    flag0 = p >= thr
    p0 = np.dot(w[ux == 1], flag0[ux == 1]) / w[ux == 1].sum()
    cell_masks = {f"{sg}={lv}": (np.ones(len(d), bool) if sg == "all" else d[sg].astype(str).eq(lv).to_numpy())
                  for sg, lv in CELLS}
    # each cell gets its own 15-point drop: un-flag DELTA / p0_cell of its flagged cases
    drop_share = {c: DELTA / (np.dot(w[(ux == 1) & m], flag0[(ux == 1) & m]) / w[(ux == 1) & m].sum())
                  for c, m in cell_masks.items()}
    psus = d[["str_key", "psu_key"]].drop_duplicates()
    groups = [g["psu_key"].to_numpy() for _, g in psus.groupby("str_key")]
    stats = {(drop, n, c): [] for drop in (0, 1) for n in N_GRID for c in cell_masks}
    rng = np.random.default_rng(seed)
    for r in range(reps):
        side = {}
        for keys in groups:
            perm = rng.permutation(len(keys))
            n_ref = len(keys) // 2 if len(keys) % 2 == 0 else len(keys) // 2 + rng.integers(0, 2)
            for i, k in enumerate(keys[perm]):
                side[k] = i < n_ref
        is_ref = d["psu_key"].map(side).to_numpy(bool)
        R, N = np.flatnonzero(is_ref), np.flatnonzero(~is_ref)
        u_draw = rng.random(len(d))
        for drop in (0, 1):
            for n in N_GRID:
                pick = rng.choice(N, size=min(n, len(N)), replace=False)
                for c, m in cell_masks.items():
                    flag = flag0
                    if drop:
                        flag = flag0 & ~(u_draw < drop_share[c])
                    rr = R[(ux[R] == 1) & m[R]]
                    aa = pick[(ux[pick] == 1) & m[pick]]
                    ref = np.dot(w[rr], flag0[rr]) / w[rr].sum()
                    aud = np.dot(w[aa], flag[aa]) / w[aa].sum() if len(aa) else ref
                    stats[(drop, n, c)].append(ref - aud)
    rows = []
    for n in N_GRID:
        for c in cell_masks:
            null = np.asarray(stats[(0, n, c)])
            alt = np.asarray(stats[(1, n, c)])
            cut = np.quantile(null, 1 - ALPHA)
            rows.append(dict(audit_n=n, cell=c, power=float(np.mean(alt > cut)),
                             false_alarm=float(np.mean(null > cut)), threshold=float(cut),
                             mean_drop_detected=float(alt.mean()), reps=reps))
    return pd.DataFrame(rows), p0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="nhanes_out/nhanes_all.csv")
    ap.add_argument("--models", default="results/rq2_models.joblib")
    ap.add_argument("--model-eval", default="results/model_eval.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--reps", type=int, default=1000)
    args = ap.parse_args()
    df = pd.read_csv(args.csv, low_memory=False)
    base = joblib.load(args.models)
    d = prepare(df, base)
    me = pd.read_csv(args.model_eval, dtype={"level": str})
    a = analytic(me, d)
    a.to_csv(Path(args.out) / "audit_power.csv", index=False)
    print(a[a["sided"] == 1][["subgroup", "level", "p0", "deff", "cases_needed", "lab_adults_needed_in_cell",
                              "total_audit_if_random"]].round(3).to_string(index=False))
    sim, p0 = simulate(d, base.thresholds["logistic"], args.reps)
    sim.to_csv(Path(args.out) / "audit_power_sim.csv", index=False)
    print(f"p0 (2011-2016 OOF undiagnosed sensitivity) = {p0:.3f}")
    print(sim.pivot_table(index="audit_n", columns="cell", values="power").round(2).to_string())


if __name__ == "__main__":
    main()
