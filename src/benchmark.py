"""Upgrade 2: a semi-synthetic monitoring benchmark where the truth is known.

The real 2017-2023 data had nothing for the monitors to catch. Here each
replicate builds its own "new period" from real NHANES rows and changes it by
a known amount, following Rabanser, Gunnemann and Lipton (2019).

One replicate
    1. Split 2011-2016 into two half-samples by PSU: in every stratum, PSUs are
       randomly divided between a reference half and a deployment half (the
       half-sample idea behind balanced repeated replication). Each half is a
       design-respecting sample of the same population.
    2. Apply a scenario at a severity to the deployment half. Population shifts
       are made by drawing each row's multiplicity from Poisson(lambda * factor),
       so they show up in raw counts (unweighted monitors) as well as in the
       weighted estimates. Label changes are made by flipping labels of
       randomly chosen people.
    3. Run every monitor (reference half vs deployment half).
The frozen model is the y_diag logistic regression. Rows keep their
leave-one-cycle-out predictions, so no row is scored by a model that saw it;
rows whose features change are re-scored with that same fold model.

Scenarios (severity 1-5; severity 0 = no change = the null)
    cov_older        adults 60+ up-weighted by k in {1.25, 1.5, 2, 3, 4}
    cov_uninsured    uninsured adults up-weighted by k
    cov_lowincome    income-to-poverty < 1.3 up-weighted by k
    prev_shift       all y_bio = 1 up-weighted by k (pure label shift)
    undx_growth      undiagnosed cases among men 20-59 up-weighted by k
                     (a higher hidden-case share where the model is weakest)
    dx_label         screening falls in men 20-59: a share f in
                     {0.1, ..., 0.5} of diagnosed people with HbA1c >= 6.5
                     become undiagnosed; features unchanged
    dx_access        same, and those people also lose insurance and their usual
                     source of care (features change, rows re-scored)
    mixed            cov_uninsured at k plus dx_access at f together

Ground truth for each scenario x severity is the population-level change in
the frozen model's performance on all of 2011-2016 with the shift applied in
expectation (not the noisy replicate value). A scenario is HARMFUL when
sensitivity among undiagnosed cases drops by at least HARM_TOL (5 points,
fixed before running).

Monitors (statistic, larger = more shift)
    psi_w, psi_u      weighted / unweighted feature PSI, 9 features
    ks_w, ks_u        weighted / unweighted feature KS, numeric features
    domain_auc        weighted domain-classifier AUC (cross-fitted logistic)
    diag_prev         |change in weighted y_diag prevalence|
    mean_risk         |change in mean predicted risk|
    bbse              |BBSE-estimated y_bio prevalence - reference y_bio prevalence|
    audit_n{500..4000}  drop in undiagnosed sensitivity measured on a
                      random lab-tested audit sample of n deployment adults
Multi-feature monitors use max_j (stat_j / null median_j), so the family-wise
false alarm rate is controlled. Every monitor alarms when its statistic exceeds
the 95th percentile of its null distribution (severity 0 replicates): the
same noise-floor rule drift.py uses, calibrated on null replicates.

Outputs
    results/benchmark_replicates.csv   one row per replicate, all statistics
    results/benchmark_truth.csv        population-level truth per scenario x severity
    results/benchmark_power.csv        detection rate per monitor x scenario x severity
    results/benchmark_2x2.csv          per monitor: caught / missed / false alarm / quiet
                                       over all harmful and harmless scenario cells

Run: python -m src.benchmark --reps 200 --null-reps 1000 --jobs 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.harness import period_frame, SEED
from src.models import analysis_domain, fit_on_rows, predict_with

HARM_TOL = 0.05
K_GRID = (1.25, 1.5, 2.0, 3.0, 4.0)
F_GRID = (0.1, 0.2, 0.3, 0.4, 0.5)
SCENARIOS = ("cov_older", "cov_uninsured", "cov_lowincome", "prev_shift", "undx_growth",
             "dx_label", "dx_access", "mixed")
NUMERIC = ("age", "pir", "bmi", "waist")
CATEG = ("sex", "race_eth", "education", "insured", "routine_care")
AUDIT_N = (500, 1000, 2000, 4000)
MONITORS = ("psi_w", "psi_u", "ks_w", "ks_u", "domain_auc", "diag_prev", "mean_risk", "bbse",
            *(f"audit_n{n}" for n in AUDIT_N))


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------

def prepare(df, base_bundle):
    full = period_frame(df, "train")
    an = analysis_domain(full).to_numpy()
    d = full.loc[an].reset_index(drop=True)
    d["p"] = d["SEQN"].map(base_bundle.oof.set_index("SEQN")["logistic"]).to_numpy(float)
    d["young_man"] = d["sex"].astype(str).eq("male") & (pd.to_numeric(d["age"]) < 60)
    d["hba1c_pos"] = pd.to_numeric(d["hba1c"], errors="coerce") >= 6.5
    # fold models (leave one cycle out) to re-score rows whose features change
    rows = df.loc[(df["period"] == "train") & analysis_domain(df)].reset_index(drop=True)
    folds = {}
    for c in sorted(rows["cycle"].unique()):
        prep, est = fit_on_rows(rows.loc[rows["cycle"] != c])
        folds[c] = (prep, est["logistic"])
    # check the fold models reproduce the stored out-of-fold predictions
    chk = np.concatenate([predict_with(*folds[c], d.loc[d["cycle"] == c]) for c in folds])
    ref = np.concatenate([d.loc[d["cycle"] == c, "p"].to_numpy() for c in folds])
    assert np.allclose(chk, ref, atol=1e-10), "fold models do not reproduce OOF predictions"
    d["p_noaccess"] = np.nan
    mod = d.copy()
    mod["insured"] = 0.0
    mod["routine_care"] = 0.0
    for c in folds:
        m = d["cycle"].eq(c).to_numpy()
        d.loc[m, "p_noaccess"] = predict_with(*folds[c], mod.loc[m])
    # numeric codes for categorical features
    for c in CATEG:
        d[c + "_code"] = pd.factorize(d[c], use_na_sentinel=False)[0]
    d["psu_key"] = d["cycle"].astype(str) + "|" + d["stratum"].astype(str) + "|" + d["psu"].astype(str)
    d["str_key"] = d["cycle"].astype(str) + "|" + d["stratum"].astype(str)
    return d


def scenario_factor(d, scenario, sev):
    """Row multipliers (population shifts) for severity index sev (0 = none)."""
    f = np.ones(len(d))
    if sev == 0:
        return f
    k = K_GRID[sev - 1]
    if scenario == "cov_older":
        f[pd.to_numeric(d["age"]).to_numpy() >= 60] = k
    elif scenario in ("cov_uninsured", "mixed"):
        f[pd.to_numeric(d["insured"]).to_numpy() == 0] = k
    elif scenario == "cov_lowincome":
        f[d["income_band2"].astype(str).eq("<1.3").to_numpy()] = k
    elif scenario == "prev_shift":
        f[d["y_bio"].to_numpy() == 1] = k
    elif scenario == "undx_growth":
        f[(d["undx"].to_numpy() == 1) & d["young_man"].to_numpy()] = k
    return f


def flip_eligible(d):
    return (d["young_man"] & d["hba1c_pos"] & (d["y_diag"] == 1)).to_numpy()


def flip_share(scenario, sev):
    if sev == 0 or scenario not in ("dx_label", "dx_access", "mixed"):
        return 0.0
    return F_GRID[sev - 1]


# ---------------------------------------------------------------------------
# weighted metrics on plain arrays
# ---------------------------------------------------------------------------

def wauc(y, p, w):
    if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
        return np.nan
    return roc_auc_score(y, p, sample_weight=w)


def sens(flag, pos, w):
    s = w[pos].sum()
    return np.dot(w[pos], flag[pos]) / s if s > 0 else np.nan


def performance(y_diag, y_bio, undx, p, w, thr):
    flag = p >= thr
    return {"auroc_bio": wauc(y_bio, p, w), "auroc_diag": wauc(y_diag, p, w),
            "sens_bio": sens(flag, y_bio == 1, w), "sens_undx": sens(flag, undx == 1, w),
            "sens_diag": sens(flag, y_diag == 1, w)}


def expected_shift(d, scenario, sev):
    """Population-level shifted data: rows eligible for flipping are split into
    a flipped copy (weight w f) and an unflipped copy (weight w (1 - f))."""
    fac = scenario_factor(d, scenario, sev)
    fs = flip_share(scenario, sev)
    w = d["w"].to_numpy(float) * fac
    yd, yb, ux = d["y_diag"].to_numpy(int), d["y_bio"].to_numpy(int), d["undx"].to_numpy(int)
    p = d["p"].to_numpy(float)
    if fs == 0:
        return yd, yb, ux, p, w
    el = flip_eligible(d)
    p_flip = d["p_noaccess"].to_numpy(float) if scenario in ("dx_access", "mixed") else p
    w_keep = np.where(el, w * (1 - fs), w)
    return (np.r_[yd, np.zeros(el.sum(), int)], np.r_[yb, yb[el]], np.r_[ux, np.ones(el.sum(), int)],
            np.r_[p, p_flip[el]], np.r_[w_keep, w[el] * fs])


def truth_table(d, thr):
    base = performance(d["y_diag"].to_numpy(int), d["y_bio"].to_numpy(int), d["undx"].to_numpy(int),
                       d["p"].to_numpy(float), d["w"].to_numpy(float), thr)
    rows = []
    for sc in SCENARIOS:
        for sev in range(0, 6):
            yd, yb, ux, p, w = expected_shift(d, sc, sev)
            perf = performance(yd, yb, ux, p, w, thr)
            row = dict(scenario=sc, severity=sev,
                       k=K_GRID[sev - 1] if sev and sc not in ("dx_label", "dx_access") else 1.0,
                       flip=flip_share(sc, sev),
                       y_bio_prev=np.dot(w, yb) / w.sum(), y_diag_prev=np.dot(w, yd) / w.sum(),
                       undx_share_of_cases=np.dot(w, ux) / np.dot(w, yb))
            for k, v in perf.items():
                row[k] = v
                row["change_" + k] = v - base[k]
            row["harmful"] = row["change_sens_undx"] <= -HARM_TOL
            row["harmful_bio_auroc"] = row["change_auroc_bio"] <= -0.02
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# monitors
# ---------------------------------------------------------------------------

def psi_num(xr, wr, xn, wn, bins=10):
    ok_r, ok_n = np.isfinite(xr), np.isfinite(xn)
    xr, wr, xn, wn = xr[ok_r], wr[ok_r], xn[ok_n], wn[ok_n]
    order = np.argsort(xr)
    cw = np.cumsum(wr[order]) / wr.sum()
    edges = np.unique(xr[order][np.searchsorted(cw, np.linspace(0, 1, bins + 1)[1:-1])])
    br = np.bincount(np.searchsorted(edges, xr, side="right"), weights=wr, minlength=len(edges) + 1)
    bn = np.bincount(np.searchsorted(edges, xn, side="right"), weights=wn, minlength=len(edges) + 1)
    pr, pn = br / br.sum() + 1e-6, bn / bn.sum() + 1e-6
    return float(np.sum((pn - pr) * np.log(pn / pr)))


def psi_cat(cr, wr, cn, wn, ncat):
    br = np.bincount(cr, weights=wr, minlength=ncat)
    bn = np.bincount(cn, weights=wn, minlength=ncat)
    pr, pn = br / br.sum() + 1e-6, bn / bn.sum() + 1e-6
    return float(np.sum((pn - pr) * np.log(pn / pr)))


def ks(xr, wr, xn, wn):
    ok_r, ok_n = np.isfinite(xr), np.isfinite(xn)
    xr, wr, xn, wn = xr[ok_r], wr[ok_r], xn[ok_n], wn[ok_n]
    grid = np.unique(np.r_[xr, xn])
    def ecdf(x, w):
        o = np.argsort(x)
        c = np.cumsum(w[o]) / w.sum()
        idx = np.searchsorted(x[o], grid, side="right") - 1
        return np.where(idx >= 0, c[np.clip(idx, 0, None)], 0.0)
    return float(np.max(np.abs(ecdf(xr, wr) - ecdf(xn, wn))))


def domain_matrix(d):
    x = pd.DataFrame({c: pd.to_numeric(d[c], errors="coerce") for c in NUMERIC + ("education",)})
    x = x.fillna(x.median())
    x["age2"] = (x["age"] - 50) ** 2
    x["bmi2"] = (x["bmi"] - 29) ** 2
    x["male"] = d["sex"].astype(str).eq("male").astype(float)
    for c in ("insured", "routine_care"):
        x[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    for lv in d["race_eth"].astype(str).unique()[1:]:
        x["r_" + lv] = d["race_eth"].astype(str).eq(lv).astype(float)
    return ((x - x.mean()) / x.std().replace(0, 1)).to_numpy()


def domain_auc(Xr, wr, Xn, wn, rng):
    X = np.r_[Xr, Xn]
    y = np.r_[np.zeros(len(Xr)), np.ones(len(Xn))]
    w = np.r_[wr / wr.sum(), wn / wn.sum()] * len(y) / 2
    fold = rng.integers(0, 2, len(y))
    p = np.zeros(len(y))
    for k in (0, 1):
        tr, te = fold != k, fold == k
        m = LogisticRegression(max_iter=500).fit(X[tr], y[tr], sample_weight=w[tr])
        p[te] = m.decision_function(X[te])
    return roc_auc_score(y, p, sample_weight=w)


def one_replicate(d, X, scenario, sev, rep, thr, seed):
    rng = np.random.default_rng([seed, SCENARIOS.index(scenario), sev, rep])
    # 1. half-sample split by PSU within strata
    psus = d[["str_key", "psu_key"]].drop_duplicates()
    side = {}
    for _, g in psus.groupby("str_key"):
        keys = g["psu_key"].to_numpy()
        perm = rng.permutation(len(keys))
        n_ref = len(keys) // 2 if len(keys) % 2 == 0 else (len(keys) // 2 + rng.integers(0, 2))
        for i, kk in enumerate(keys[perm]):
            side[kk] = i < n_ref
    is_ref = d["psu_key"].map(side).to_numpy(bool)
    R, N = np.flatnonzero(is_ref), np.flatnonzero(~is_ref)
    w = d["w"].to_numpy(float) * 2          # each half stands for the population
    # 2. scenario on the deployment half
    fac = scenario_factor(d, scenario, sev)[N]
    lam = len(N) / fac.sum()
    mult = rng.poisson(lam * fac).astype(float)
    yd = d["y_diag"].to_numpy(int).copy()
    ux = d["undx"].to_numpy(int).copy()
    yb = d["y_bio"].to_numpy(int)
    p = d["p"].to_numpy(float).copy()
    ins = pd.to_numeric(d["insured"]).to_numpy(float).copy()
    rc = pd.to_numeric(d["routine_care"]).to_numpy(float).copy()
    fs = flip_share(scenario, sev)
    if fs > 0:
        el = N[flip_eligible(d)[N]]
        flip = el[rng.random(len(el)) < fs]
        yd[flip], ux[flip] = 0, 1
        if scenario in ("dx_access", "mixed"):
            p[flip] = d["p_noaccess"].to_numpy(float)[flip]
            ins[flip], rc[flip] = 0.0, 0.0
    keep = mult > 0
    N, mult = N[keep], mult[keep]
    wr, wn = w[R], w[N] * mult
    ur, un = np.ones(len(R)), mult
    out = dict(scenario=scenario, severity=sev, rep=rep, n_ref=len(R), n_new=int(mult.sum()))
    # realised performance (noisy, for reference)
    perf_r = performance(yd[R], yb[R], ux[R], p[R], wr, thr)
    perf_n = performance(yd[N], yb[N], ux[N], p[N], wn, thr)
    for k in perf_r:
        out["realised_change_" + k] = perf_n[k] - perf_r[k]
    # 3. monitors
    feats = {}
    for c in NUMERIC:
        x = pd.to_numeric(d[c], errors="coerce").to_numpy(float)
        feats[c] = ("num", x)
    for c in CATEG:
        if c == "insured":
            feats[c] = ("cat", np.where(np.isnan(ins), 2, ins).astype(int))
        elif c == "routine_care":
            feats[c] = ("cat", np.where(np.isnan(rc), 2, rc).astype(int))
        else:
            feats[c] = ("cat", d[c + "_code"].to_numpy())
    for c, (kind, x) in feats.items():
        if kind == "num":
            out[f"psi_w:{c}"] = psi_num(x[R], wr, x[N], wn)
            out[f"psi_u:{c}"] = psi_num(x[R], ur, x[N], un)
            out[f"ks_w:{c}"] = ks(x[R], wr, x[N], wn)
            out[f"ks_u:{c}"] = ks(x[R], ur, x[N], un)
        else:
            nc = int(x.max()) + 1
            out[f"psi_w:{c}"] = psi_cat(x[R], wr, x[N], wn, nc)
            out[f"psi_u:{c}"] = psi_cat(x[R], ur, x[N], un, nc)
    Xn = X[N].copy()
    if fs > 0 and scenario in ("dx_access", "mixed"):
        Xn = domain_matrix(d.assign(insured=ins, routine_care=rc))[N]
    out["domain_auc"] = domain_auc(X[R], wr, Xn, wn, rng)
    out["diag_prev"] = abs(np.dot(wn, yd[N]) / wn.sum() - np.dot(wr, yd[R]) / wr.sum())
    out["mean_risk"] = abs(np.dot(wn, p[N]) / wn.sum() - np.dot(wr, p[R]) / wr.sum())
    fr, fn = p[R] >= thr, p[N] >= thr
    a = sens(fr, yb[R] == 1, wr)
    b = np.dot(wr[yb[R] == 0], fr[yb[R] == 0]) / wr[yb[R] == 0].sum()
    q = np.clip((np.dot(wn, fn) / wn.sum() - b) / (a - b), 0, 1)
    out["bbse"] = abs(q - np.dot(wr, yb[R]) / wr.sum())
    out["bbse_signed"] = q - np.dot(wr, yb[R]) / wr.sum()
    ref_undx = sens(fr, ux[R] == 1, wr)
    # audit: n deployment adults drawn with probability proportional to multiplicity
    for n in AUDIT_N:
        pick = rng.choice(len(N), size=min(n, len(N)), replace=False, p=mult / mult.sum())
        s = sens(fn[pick], ux[N][pick] == 1, w[N][pick])
        out[f"audit_n{n}"] = (ref_undx - s) if np.isfinite(s) else 0.0
        out[f"audit_n{n}_cases"] = int((ux[N][pick] == 1).sum())
    return out


def combine_features(rep, null):
    """Max over features of stat / null median, per multi-feature monitor."""
    for mon in ("psi_w", "psi_u", "ks_w", "ks_u"):
        cols = [c for c in rep.columns if c.startswith(mon + ":")]
        med = null[cols].median()
        rep[mon] = (rep[cols] / med).max(axis=1)
    return rep


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan
    z = 1.96
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def run(df, base, reps=200, null_reps=1000, jobs=2, seed=SEED):
    d = prepare(df, base)
    X = domain_matrix(d)
    thr = base.thresholds["logistic"]
    truth = truth_table(d, thr)
    tasks = [("cov_older", 0, r) for r in range(null_reps)]        # null: one shared pool
    tasks += [(sc, sev, r) for sc in SCENARIOS for sev in range(1, 6) for r in range(reps)]
    res = Parallel(n_jobs=jobs, batch_size=16)(
        delayed(one_replicate)(d, X, sc, sev, r, thr, seed) for sc, sev, r in tasks)
    rep = pd.DataFrame(res)
    rep.loc[rep["severity"] == 0, "scenario"] = "null"
    null = rep[rep["severity"] == 0]
    rep = combine_features(rep, null)
    null = rep[rep["severity"] == 0]
    cut = {m: float(np.quantile(null[m], 0.95)) for m in MONITORS}
    for m in MONITORS:
        rep["alarm:" + m] = rep[m] > cut[m]
    power = []
    for (sc, sev), g in rep.groupby(["scenario", "severity"]):
        t = truth[(truth["scenario"] == (sc if sc != "null" else "cov_older")) & (truth["severity"] == sev)].iloc[0]
        for m in MONITORS:
            k = int(g["alarm:" + m].sum())
            lo, hi = wilson(k, len(g))
            power.append(dict(scenario=sc, severity=sev, monitor=m, detection_rate=k / len(g),
                              ci_low=lo, ci_high=hi, reps=len(g), threshold=cut[m],
                              harmful=bool(t["harmful"]), change_sens_undx=t["change_sens_undx"],
                              change_auroc_bio=t["change_auroc_bio"], change_sens_bio=t["change_sens_bio"],
                              change_auroc_diag=t["change_auroc_diag"],
                              change_y_bio_prev=t["y_bio_prev"] - truth.loc[truth["severity"] == 0, "y_bio_prev"].iloc[0]))
    power = pd.DataFrame(power)
    cells = power[power["scenario"] != "null"]
    two = []
    for m, g in cells.groupby("monitor"):
        h, nh = g[g["harmful"]], g[~g["harmful"]]
        two.append(dict(monitor=m, harmful_cells=len(h), harmless_cells=len(nh),
                        caught=float(h["detection_rate"].sum()) * reps / reps,
                        mean_power_harmful=h["detection_rate"].mean(),
                        mean_alarm_rate_harmless=nh["detection_rate"].mean(),
                        null_false_alarm=float(power[(power["scenario"] == "null") & (power["monitor"] == m)]["detection_rate"].iloc[0]),
                        harmful_cells_detected_80=int((h["detection_rate"] >= 0.8).sum())))
    two = pd.DataFrame(two).drop(columns="caught")
    return rep, truth, power, two


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="nhanes_out/nhanes_all.csv")
    ap.add_argument("--models", default="results/rq2_models.joblib")
    ap.add_argument("--out", default="results")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--null-reps", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    df = pd.read_csv(args.csv, low_memory=False)
    base = joblib.load(args.models)
    rep, truth, power, two = run(df, base, args.reps, args.null_reps, args.jobs)
    out = Path(args.out)
    rep.to_csv(out / "benchmark_replicates.csv", index=False)
    truth.to_csv(out / "benchmark_truth.csv", index=False)
    power.to_csv(out / "benchmark_power.csv", index=False)
    two.to_csv(out / "benchmark_2x2.csv", index=False)
    print(truth[["scenario", "severity", "change_sens_undx", "change_auroc_bio", "harmful"]].round(3).to_string(index=False))
    print(two.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
