"""Upgrade 5a: why are undiagnosed cases missed?

For every true case (y_bio = 1) the frozen y_diag model's risk score on the
log-odds scale splits exactly into per-feature contributions:
    logistic  beta_j * (x_j - weighted training mean of x_j)
    LightGBM  TreeSHAP values (lightgbm pred_contrib=True), log-odds scale
Both are additive, so the gap in mean log-odds between diagnosed and
undiagnosed cases decomposes exactly:
    mean_logit(diagnosed) - mean_logit(undiagnosed)
        = sum_j [ mean contrib_j(diagnosed) - mean contrib_j(undiagnosed) ]
A feature with a large positive share is one that pushes diagnosed cases up
and undiagnosed cases down: the model relies on it, and undiagnosed people
do not show it.

Outputs (shared schema, method = "<model>")
    results/explain_decomposition.csv  per-feature share of the diagnosed vs
        undiagnosed score gap, per period x cell, with Rao-Wu CIs
    results/shap_summary.csv  mean signed and mean |contribution| per feature
        for four groups of cases: diagnosed flagged / missed, undiagnosed
        flagged / missed
    results/explain_feature_means.csv  weighted raw feature means for
        diagnosed vs undiagnosed cases, and their difference with CIs
    results/explain_coefficients.csv  standardised logistic coefficients when
        trained on y_diag vs y_bio vs SAR-EM (if results/pu_models.joblib exists)

Run: python -m src.explain --reps 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.harness import (PERIODS, period_frame, pooled_frame, primary_domains, load, SEED,
                         label_arrays, t_summary)
from src.models import FEATURE_NAMES, analysis_domain
from src.survey_boot import rao_wu_replicate_weights, design_degrees_of_freedom
from src.model_eval import bh_adjust

GROUPS_OF_FEATURES = {
    "access": ("insured", "routine_care"),
    "body_size": ("bmi", "waist"),
    "socioeconomic": ("education", "pir", "pir_missing"),
}
# the pre-registered explanation family: three feature groups plus age and sex
FAMILY_TERMS = ("age", "male", "access", "body_size", "socioeconomic")
RAW_MEANS = ("age", "bmi", "waist", "pir", "education", "insured", "routine_care")


def contributions(bundle, model, rows):
    """(n x 9) log-odds contributions and the base value."""
    X = pd.DataFrame(bundle.preprocessor.transform(rows), columns=FEATURE_NAMES)
    est = bundle.estimators[model]
    if model == "logistic":
        beta = est.coef_[0]
        # centre on the training-weighted mean of the transformed features
        mean = bundle.train_feature_mean
        C = (X.to_numpy() - mean) * beta
        base = est.intercept_[0] + mean @ beta
    else:
        raw = est.predict(X, pred_contrib=True)
        C, base = raw[:, :-1], raw[:, -1]
    return C, base


def _with_train_mean(bundle, df):
    rows = df.loc[(df["period"] == "train") & analysis_domain(df)]
    X = bundle.preprocessor.transform(rows)
    bundle.train_feature_mean = np.average(X, axis=0, weights=rows["w"].to_numpy(float))
    return bundle


def run(df, bundle, n_reps=200, seed=SEED):
    dec_rows, shap_rows, mean_rows = [], [], []
    frames = [("2017-2020", period_frame(df, "test_P")), ("2021-2023", period_frame(df, "test_L")),
              ("2017-2023 pooled", pooled_frame(df, ["test_P", "test_L"]))]
    for k, (plabel, full) in enumerate(frames):
        lab, an = label_arrays(full)
        dom = primary_domains(full)
        masks = np.column_stack([d[3] for d in dom])
        dof = design_degrees_of_freedom(full)
        w = full["w"].to_numpy(float)
        reps = list(rao_wu_replicate_weights(full, n_reps=n_reps, seed=seed + k))
        dx = lab["y_diag"]
        ux = lab["undx"]
        n_undx = (masks * ux[:, None]).sum(0)

        # raw feature means (diagnosed vs undiagnosed cases)
        R = full[list(RAW_MEANS)].apply(pd.to_numeric, errors="coerce")
        for f in RAW_MEANS:
            x = R[f].to_numpy(float)
            ok = np.isfinite(x)
            xv = np.where(ok, x, 0.0)

            def stat(wv):
                Wd = masks * (wv * dx * ok)[:, None]
                Wu = masks * (wv * ux * ok)[:, None]
                with np.errstate(invalid="ignore", divide="ignore"):
                    md, mu = (Wd.T @ xv) / Wd.sum(0), (Wu.T @ xv) / Wu.sum(0)
                return md, mu
            md, mu = stat(w)
            rp = [stat(rw) for rw in reps]
            for j, (sg, lv, role, _) in enumerate(dom):
                for what, pt, rr in [("mean_diagnosed", md[j], [r[0][j] for r in rp]),
                                     ("mean_undiagnosed", mu[j], [r[1][j] for r in rp]),
                                     ("diff_diag_minus_undx", md[j] - mu[j], [r[0][j] - r[1][j] for r in rp])]:
                    rr = np.asarray(rr)
                    se = float(np.sqrt(np.mean((rr - pt) ** 2)))
                    mean_rows.append(dict(period_label=plabel, subgroup=sg, level=lv, role=role,
                                          method="data", label_or_gap=f, metric=what,
                                          **t_summary(float(pt), se, dof), family="", q_bh=np.nan,
                                          status="ok" if (role == "overall" or n_undx[j] >= 30) else "not_reportable_small_undx",
                                          n_rows=int(masks[:, j].sum()), n_undx=int(n_undx[j])))

        for model in bundle.estimators:
            C = np.zeros((len(full), len(FEATURE_NAMES)))
            p = np.zeros(len(full))
            Ca, base = contributions(bundle, model, full.loc[an])
            C[an] = Ca
            p[an] = bundle.predict_probabilities(full.loc[an], model)
            flag = (p >= bundle.thresholds[model]).astype(float)
            # feature-group contributions
            G = {g: C[:, [FEATURE_NAMES.index(f) for f in fs]].sum(1) for g, fs in GROUPS_OF_FEATURES.items()}
            names = list(FEATURE_NAMES) + list(G)
            M = np.column_stack([C] + [G[g] for g in GROUPS_OF_FEATURES])

            def dec(wv):
                Wd = masks * (wv * dx)[:, None]
                Wu = masks * (wv * ux)[:, None]
                md = (M.T @ Wd) / Wd.sum(0)       # (features x domains)
                mu = (M.T @ Wu) / Wu.sum(0)
                diff = md - mu
                total = diff[: len(FEATURE_NAMES)].sum(0)
                return diff, total
            diff, total = dec(w)
            rp = [dec(rw) for rw in reps]
            for j, (sg, lv, role, _) in enumerate(dom):
                st = "ok" if (role == "overall" or n_undx[j] >= 30) else "not_reportable_small_undx"
                base_row = dict(period_label=plabel, subgroup=sg, level=lv, role=role,
                                method=model, family="", q_bh=np.nan, status=st,
                                n_rows=int(masks[:, j].sum()), n_undx=int(n_undx[j]))
                rt = np.array([r[1][j] for r in rp])
                se = float(np.sqrt(np.mean((rt - total[j]) ** 2)))
                dec_rows.append({**base_row, "label_or_gap": "total", "metric": "logit_gap",
                                 **t_summary(float(total[j]), se, dof)})
                for i, nm in enumerate(names):
                    rd = np.array([r[0][i, j] for r in rp])
                    se = float(np.sqrt(np.mean((rd - diff[i, j]) ** 2)))
                    dec_rows.append({**base_row, "label_or_gap": nm, "metric": "logit_gap",
                                     **t_summary(float(diff[i, j]), se, dof)})
                    sh = diff[i, j] / total[j]
                    rs = rd / rt
                    se = float(np.sqrt(np.mean((rs - sh) ** 2)))
                    dec_rows.append({**base_row, "label_or_gap": nm, "metric": "share_of_gap",
                                     **t_summary(float(sh), se, dof)})
            # four groups of cases, overall cell only (point estimates)
            cases = {"diagnosed_flagged": dx * flag, "diagnosed_missed": dx * (1 - flag),
                     "undiagnosed_flagged": ux * flag, "undiagnosed_missed": ux * (1 - flag)}
            for gname, ind in cases.items():
                wv = w * ind * an
                if wv.sum() == 0:
                    continue
                for i, nm in enumerate(names):
                    shap_rows.append(dict(period_label=plabel, subgroup="all", level="all",
                                          role="overall", method=model, label_or_gap=nm, metric=gname,
                                          mean_signed=float(np.dot(wv, M[:, i]) / wv.sum()),
                                          mean_abs=float(np.dot(wv, np.abs(M[:, i])) / wv.sum()),
                                          n_rows=int((ind * an > 0).sum())))
    dec = pd.DataFrame(dec_rows)
    dec.loc[dec["metric"] == "share_of_gap", "p_value"] = np.nan
    fam = ((dec["metric"] == "logit_gap") & dec["label_or_gap"].isin(FAMILY_TERMS) & dec["status"].eq("ok")
           & ~dec["period_label"].str.contains("pooled"))
    dec.loc[fam, "family"] = "explain_groups"
    dec.loc[fam, "q_bh"] = bh_adjust(dec.loc[fam, "p_value"].to_numpy())
    return dec, pd.DataFrame(shap_rows), pd.DataFrame(mean_rows)


def coefficient_table(pu_path):
    if not Path(pu_path).exists():
        return None
    pu = joblib.load(pu_path)
    rows = []
    for m in ("diag_logistic", "bio_logistic", "noaccess_logistic", "sar_logistic"):
        f = pu.final[m]
        for name, b in zip(f.features, f.model.coef_[0]):
            rows.append(dict(method=m, feature=name, coefficient=float(b)))
        if m == "sar_logistic":
            prop = f.extra["propensity"]
            for name, b in zip(f.extra["propensity_features"], prop.coef_[0]):
                rows.append(dict(method="sar_propensity", feature=name, coefficient=float(b)))
    return pd.DataFrame(rows).pivot_table(index="feature", columns="method", values="coefficient").reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="nhanes_out/nhanes_all.csv")
    ap.add_argument("--models", default="results/rq2_models.joblib")
    ap.add_argument("--pu", default="results/pu_models.joblib")
    ap.add_argument("--out", default="results")
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    df = load(args.csv)
    bundle = _with_train_mean(joblib.load(args.models), df)
    dec, shap, means = run(df, bundle, args.reps)
    out = Path(args.out)
    dec.to_csv(out / "explain_decomposition.csv", index=False)
    shap.to_csv(out / "shap_summary.csv", index=False)
    means.to_csv(out / "explain_feature_means.csv", index=False)
    coef = coefficient_table(args.pu)
    if coef is not None:
        coef.to_csv(out / "explain_coefficients.csv", index=False)
        print(coef.round(3).to_string(index=False))
    v = dec[(dec["subgroup"] == "all") & (dec["metric"] == "share_of_gap")
            & dec["label_or_gap"].isin(FAMILY_TERMS)]
    v = dec[(dec["subgroup"] == "all") & (dec["metric"] == "logit_gap") & dec["label_or_gap"].isin(FAMILY_TERMS)]
    print(v[["period_label", "method", "label_or_gap", "estimate", "ci_low", "ci_high", "q_bh"]].round(3).to_string(index=False))
    t = dec[(dec["subgroup"] == "all") & (dec["label_or_gap"] == "total")]
    print(t[["period_label", "method", "estimate", "ci_low", "ci_high"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
