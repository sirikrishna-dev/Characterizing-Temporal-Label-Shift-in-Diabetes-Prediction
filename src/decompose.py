"""Upgrade 5b: how much of the change in diagnosed diabetes is more diabetes,
and how much is more diagnosis? And how much is just a different population?

1. Exact B x F split. Diagnosed prevalence D = B * F, with B the biomarker
   prevalence P(y_bio = 1) and F the fraction diagnosed P(y_diag = 1 | y_bio = 1)
   (every diagnosed adult is a y_bio case). Between periods A and B:
        dD = dB * mean(F)   +   mean(B) * dF
             more diabetes      more diagnosis
   Exact (symmetric, Shapley) split, overall and per primary subgroup.

2. Composition split. Reweight period A to period B's covariate mix with a
   cross-fitted, survey-weighted domain classifier (logistic regression on
   age, age^2, BMI, BMI^2, waist, income, education, sex, insurance, usual
   care, race/ethnicity). The reweighted A gives counterfactual B*, F*, D*:
        dX = (X*_A - X_A)  +  (X_B - X*_A)
             population       within-population (rates)
   The density-ratio weights are held fixed across replicates, so their
   uncertainty is left out (stated as a limitation).

3. Kitagawa decomposition over the 16 primary-subgroup cells (age band x sex
   x insured x income band), as a simpler cross-check of step 2.

Comparisons: 2011-2016 vs each test period, 2017-2020 vs 2021-2023, and each
pair of consecutive cycles (five cycles, w_lab unpooled).
Uncertainty: the two periods are independent samples; replicate r of a
change uses replicate r of each period (Rao-Wu, 200 replicates by default).

Output: results/decomposition.csv (shared schema; values are proportions,
multiply by 100 for percentage points).
Run: python -m src.decompose --reps 200
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.harness import period_frame, primary_domains, load, SEED, label_arrays, t_summary
from src.models import analysis_domain
from src.survey_boot import rao_wu_replicate_weights, design_degrees_of_freedom
from src.model_eval import bh_adjust

CELL_VARS = ("age_band2", "sex", "insured", "income_band2")


def cycle_frame(df, cycle):
    f = df.loc[df["cycle"] == cycle].copy().reset_index(drop=True)
    f["w"] = pd.to_numeric(f["w_lab"], errors="coerce").fillna(0.0)
    return f


def frames(df):
    out = {"2011-2016": period_frame(df, "train"), "2017-2020": period_frame(df, "test_P"),
           "2021-2023": period_frame(df, "test_L")}
    for c in sorted(df["cycle"].unique()):
        out[f"cycle {c}"] = cycle_frame(df, c)
    return out


COMPARISONS = [("2011-2016", "2017-2020"), ("2011-2016", "2021-2023"), ("2017-2020", "2021-2023"),
               ("cycle 2011-2012", "cycle 2013-2014"), ("cycle 2013-2014", "cycle 2015-2016"),
               ("cycle 2015-2016", "cycle 2017-2020")]


def bf(W, yb, yd):
    tot = W.sum(0)
    B = (W.T @ yb) / tot
    D = (W.T @ yd) / tot
    return B, D, D / B


def design_matrix(f):
    x = pd.DataFrame(index=f.index)
    for c in ("age", "bmi", "waist", "pir", "education"):
        v = pd.to_numeric(f[c], errors="coerce")
        x[c + "_miss"] = v.isna().astype(float)
        x[c] = v.fillna(v.median())
    x["age2"] = (x["age"] - 50) ** 2 / 100
    x["bmi2"] = (x["bmi"] - 29) ** 2 / 10
    x["male"] = f["sex"].astype(str).eq("male").astype(float)
    for c in ("insured", "routine_care"):
        x[c] = pd.to_numeric(f[c], errors="coerce").fillna(0.0)
    for lv in ("NH Black", "Mexican American", "Other Hispanic", "NH Asian", "Other/Multi"):
        x["race_" + lv] = f["race_eth"].astype(str).eq(lv).astype(float)
    return x


def density_ratio(fa, ma, fb, mb):
    """Weights r(x) on A's analysis rows so that A reweighted looks like B."""
    xa, xb = design_matrix(fa.loc[ma]), design_matrix(fb.loc[mb])
    X = pd.concat([xa, xb], ignore_index=True)
    mu, sd = X.mean(), X.std().replace(0, 1)
    X = ((X - mu) / sd).to_numpy()
    y = np.r_[np.zeros(len(xa)), np.ones(len(xb))]
    wa, wb = fa.loc[ma, "w"].to_numpy(float), fb.loc[mb, "w"].to_numpy(float)
    w = np.r_[wa / wa.sum(), wb / wb.sum()] * len(y) / 2
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        m = LogisticRegression(max_iter=3000, C=1.0).fit(X[tr], y[tr], sample_weight=w[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    pa = np.clip(p[: len(xa)], 1e-3, 1 - 1e-3)
    return pa / (1 - pa)


def kitagawa(fa, Wa, fb, Wb, base_a, base_b, out_a, out_b):
    """Composition and rate effects over the CELL_VARS cells, for every
    weight column (replicate). base_* = 0/1 who counts in the denominator."""
    def cells(f):
        key = f[CELL_VARS[0]].astype("string").fillna("NA")
        for c in CELL_VARS[1:]:
            key = key + "|" + f[c].astype("string").fillna("NA")
        return key.to_numpy(dtype=str)
    ka, kb = cells(fa), cells(fb)
    levels = sorted(set(ka) | set(kb))
    Ia = np.column_stack([ka == l for l in levels]).astype(float)
    Ib = np.column_stack([kb == l for l in levels]).astype(float)
    na = Ia.T @ (Wa * base_a[:, None])     # cells x reps
    nb = Ib.T @ (Wb * base_b[:, None])
    ya = Ia.T @ (Wa * (base_a * out_a)[:, None])
    yb = Ib.T @ (Wb * (base_b * out_b)[:, None])
    sa, sb = na / na.sum(0), nb / nb.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ra, rb = np.where(na > 0, ya / na, 0), np.where(nb > 0, yb / nb, 0)
    comp = ((sb - sa) * (ra + rb) / 2).sum(0)
    rate = ((sa + sb) / 2 * (rb - ra)).sum(0)
    return comp, rate


def run(df, n_reps=200, seed=SEED):
    fr = frames(df)
    cache = {}
    for i, (name, f) in enumerate(fr.items()):
        lab, an = label_arrays(f)
        dom = primary_domains(f)
        masks = np.column_stack([d[3] for d in dom])
        R = np.column_stack(list(rao_wu_replicate_weights(f, n_reps=n_reps, seed=seed + i)))
        cache[name] = dict(f=f, lab=lab, an=an, dom=dom, masks=masks, R=R,
                           w=f["w"].to_numpy(float), dof=design_degrees_of_freedom(f))
    rows = []

    def add(comp, method, part, point, reps, dof, sg="all", lv="all", role="overall", n=0):
        se = float(np.sqrt(np.mean((np.asarray(reps) - point) ** 2)))
        rows.append(dict(period_label=comp, subgroup=sg, level=lv, role=role, method=method,
                         label_or_gap=part, metric="prevalence_change",
                         **t_summary(float(point), se, dof), family="", q_bh=np.nan,
                         status="ok", n_rows=n, n_undx=0))

    for a, b in COMPARISONS:
        A, Bc = cache[a], cache[b]
        comp = f"{a.replace('cycle ', '')} -> {b.replace('cycle ', '')}"
        dof = min(A["dof"], Bc["dof"])
        assert [d[:2] for d in A["dom"]] == [d[:2] for d in Bc["dom"]]
        # 1. B x F per domain
        def parts(Wa, Wb):
            Ba, Da, Fa = bf(Wa, A["lab"]["y_bio"], A["lab"]["y_diag"])
            Bb, Db, Fb = bf(Wb, Bc["lab"]["y_bio"], Bc["lab"]["y_diag"])
            return {"delta_D": Db - Da, "delta_B": Bb - Ba, "delta_F": Fb - Fa,
                    "part_more_diabetes": (Bb - Ba) * (Fa + Fb) / 2,
                    "part_more_diagnosis": (Ba + Bb) / 2 * (Fb - Fa),
                    "B_from": Ba, "B_to": Bb, "F_from": Fa, "F_to": Fb, "D_from": Da, "D_to": Db}
        pt = parts(A["masks"] * A["w"][:, None], Bc["masks"] * Bc["w"][:, None])
        rp = [parts(A["masks"] * A["R"][:, [r]], Bc["masks"] * Bc["R"][:, [r]]) for r in range(n_reps)]
        for j, (sg, lv, role, _) in enumerate(A["dom"]):
            for k in pt:
                add(comp, "BxF", k, pt[k][j], [x[k][j] for x in rp], dof, sg, lv, role,
                    int(A["masks"][:, j].sum() + Bc["masks"][:, j].sum()))
        # 2. composition by reweighting (overall only)
        ma, mb = A["an"], Bc["an"]
        ratio = np.zeros(len(A["f"]))
        ratio[ma] = density_ratio(A["f"], ma, Bc["f"], mb)
        yb_a, yd_a = A["lab"]["y_bio"], A["lab"]["y_diag"]
        yb_b, yd_b = Bc["lab"]["y_bio"], Bc["lab"]["y_diag"]

        def comp_parts(wa, wb):
            wa = wa * ma
            wb = wb * mb
            wr = wa * ratio
            out = {}
            for nm, (num_a, den_a, num_b, den_b) in {
                    "B": (yb_a, 1, yb_b, 1), "D": (yd_a, 1, yd_b, 1),
                    "F": (yd_a * yb_a, yb_a, yd_b * yb_b, yb_b)}.items():
                xa = np.dot(wa, num_a) / np.dot(wa, den_a * np.ones_like(wa))
                xs = np.dot(wr, num_a) / np.dot(wr, den_a * np.ones_like(wr))
                xb = np.dot(wb, num_b) / np.dot(wb, den_b * np.ones_like(wb))
                out[f"{nm}:population"] = xs - xa
                out[f"{nm}:within"] = xb - xs
                out[f"{nm}:total"] = xb - xa
            return out
        pt2 = comp_parts(A["w"], Bc["w"])
        rp2 = [comp_parts(A["R"][:, r], Bc["R"][:, r]) for r in range(n_reps)]
        ess = float(np.dot(A["w"] * ratio, ma) ** 2 / np.dot((A["w"] * ratio) ** 2, ma))
        for k in pt2:
            add(comp, "reweight", k, pt2[k], [x[k] for x in rp2], dof, n=int(ess))
        # 3. Kitagawa cross-check
        Wa = np.column_stack([A["w"], A["R"]]) * ma[:, None]
        Wb = np.column_stack([Bc["w"], Bc["R"]]) * mb[:, None]
        one_a, one_b = np.ones(len(ma)), np.ones(len(mb))
        for nm, args in {"B": (one_a, one_b, yb_a, yb_b), "D": (one_a, one_b, yd_a, yd_b),
                         "F": (yb_a, yb_b, yd_a, yd_b)}.items():
            c, r = kitagawa(A["f"], Wa, Bc["f"], Wb, *args)
            add(comp, "kitagawa", f"{nm}:population", c[0], c[1:], dof)
            add(comp, "kitagawa", f"{nm}:within", r[0], r[1:], dof)
    out = pd.DataFrame(rows)
    out.loc[out["label_or_gap"].str.match(r"^(B|D|F)_(from|to)$"), "p_value"] = np.nan
    fam = (out["method"].eq("BxF") & out["label_or_gap"].isin(["part_more_diabetes", "part_more_diagnosis"])
           & out["role"].isin(["overall", "primary"])
           & out["period_label"].isin(["2011-2016 -> 2017-2020", "2011-2016 -> 2021-2023", "2017-2020 -> 2021-2023"]))
    out.loc[fam, "family"] = "decomposition"
    out.loc[fam, "q_bh"] = bh_adjust(out.loc[fam, "p_value"].to_numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="nhanes_out/nhanes_all.csv")
    ap.add_argument("--out", default="results/decomposition.csv")
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    out = run(load(args.csv), args.reps)
    out.to_csv(args.out, index=False)
    v = out[(out["subgroup"] == "all") & out["label_or_gap"].isin(
        ["delta_D", "part_more_diabetes", "part_more_diagnosis", "B:population", "B:within",
         "F:population", "F:within", "D:population", "D:within"])]
    print((v.assign(pp=v["estimate"] * 100, lo=v["ci_low"] * 100, hi=v["ci_high"] * 100)
           [["period_label", "method", "label_or_gap", "pp", "lo", "hi", "q_bh"]]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
