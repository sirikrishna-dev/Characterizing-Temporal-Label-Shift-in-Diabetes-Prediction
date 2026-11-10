"""Black-box shift estimation (Lipton, Wang and Smola 2018) of true diabetes
prevalence in a new period from a model trained on diagnosis labels only.

The frozen y_diag model h is applied to the new period, where no HbA1c is
needed. Its confusion rates against y_bio come from the 2011-2016 lab sample
(out-of-fold predictions):
    a = P_src(h = 1 | y_bio = 1),   b = P_src(h = 1 | y_bio = 0)
Under label shift (P(x | y) fixed), the new prevalence is
    q_hat = (P_new(h = 1) - b) / (a - b)                 (hard BBSE)
    q_hat = (E_new[p] - E_src[p | y=0]) / (E_src[p | y=1] - E_src[p | y=0])   (soft)

Compared with what a team without lab tests would otherwise report:
    naive_diag   the weighted y_diag prevalence in the new period
    elkan_noto   mean(g) / c, with c fixed at its 2011-2016 value
and with the truth, the weighted y_bio prevalence of the new period.

Uncertainty: the source and new period are independent samples, so replicate
r of the estimate uses source replicate r and new-period replicate r.

Run: python -m src.bbse --reps 200    -> results/bbse.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.harness import (PERIODS, TEST_PERIODS, period_frame, primary_domains, load, SEED,
                         label_arrays, t_summary)
from src.models import analysis_domain
from src.survey_boot import rao_wu_replicate_weights, design_degrees_of_freedom
from src.model_eval import bh_adjust


def source_stats(W, p, flag, yb, yd):
    """Per-domain source quantities from weight matrix W (rows x domains)."""
    pos = W.T @ yb
    neg = W.sum(0) - pos
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(a=(W.T @ (flag * yb)) / pos, b=(W.T @ (flag * (1 - yb))) / neg,
                    pa=(W.T @ (p * yb)) / pos, pb=(W.T @ (p * (1 - yb))) / neg,
                    c=(W.T @ (p * yd)) / (W.T @ yd), prev=pos / W.sum(0),
                    diag=(W.T @ yd) / W.sum(0))


def target_stats(W, p, flag, yb, yd):
    tot = W.sum(0)
    return dict(mu=(W.T @ flag) / tot, mup=(W.T @ p) / tot, truth=(W.T @ yb) / tot,
                diag=(W.T @ yd) / tot)


def estimates(s, t):
    with np.errstate(invalid="ignore", divide="ignore"):
        hard = (t["mu"] - s["b"]) / (s["a"] - s["b"])
        soft = (t["mup"] - s["pb"]) / (s["pa"] - s["pb"])
        en = t["mup"] / s["c"]
    return {"bbse_hard": np.clip(hard, 0, 1), "bbse_soft": np.clip(soft, 0, 1),
            "elkan_noto": np.clip(en, 0, 1), "naive_diag": t["diag"], "truth": t["truth"],
            "source_truth": s["prev"]}


def run(df, bundle, model="logistic", n_reps=200, seed=SEED, domains_from=None):
    thr = bundle.thresholds[model]
    src = period_frame(df, "train")
    s_lab, s_an = label_arrays(src)
    s_p = np.zeros(len(src))
    s_p[s_an] = src.loc[s_an, "SEQN"].map(bundle.oof.set_index("SEQN")[model]).to_numpy(float)
    s_flag = (s_p >= thr).astype(float)
    s_dom = primary_domains(src)
    s_masks = np.column_stack([d[3] for d in s_dom])
    s_dof = design_degrees_of_freedom(src)
    s_point = source_stats(s_masks * src["w"].to_numpy()[:, None], s_p, s_flag, s_lab["y_bio"], s_lab["y_diag"])
    s_reps = [source_stats(s_masks * rw[:, None], s_p, s_flag, s_lab["y_bio"], s_lab["y_diag"])
              for rw in rao_wu_replicate_weights(src, n_reps=n_reps, seed=seed)]
    rows = []
    for k, period in enumerate(TEST_PERIODS):
        new = period_frame(df, period)
        n_lab, n_an = label_arrays(new)
        n_p = np.zeros(len(new))
        n_p[n_an] = bundle.predict_probabilities(new.loc[n_an], model)
        n_flag = (n_p >= thr).astype(float)
        n_dom = primary_domains(new)
        assert [d[:2] for d in n_dom] == [d[:2] for d in s_dom], "domain mismatch"
        n_masks = np.column_stack([d[3] for d in n_dom])
        dof = min(s_dof, design_degrees_of_freedom(new))
        n_point = target_stats(n_masks * new["w"].to_numpy()[:, None], n_p, n_flag, n_lab["y_bio"], n_lab["y_diag"])
        pt = estimates(s_point, n_point)
        reps = [estimates(s_reps[r], target_stats(n_masks * rw[:, None], n_p, n_flag,
                                                   n_lab["y_bio"], n_lab["y_diag"]))
                for r, rw in enumerate(rao_wu_replicate_weights(new, n_reps=n_reps, seed=seed + 10 + k))]
        for j, (sg, lv, role, _) in enumerate(n_dom):
            n_undx = int((n_masks[:, j] * n_lab["undx"]).sum())
            for est in ("bbse_hard", "bbse_soft", "elkan_noto", "naive_diag", "truth"):
                r = np.array([x[est][j] for x in reps])
                # level, error vs truth, and estimated change vs source prevalence
                for what, point, rep in [
                        ("prevalence", pt[est][j], r),
                        ("error_vs_truth", pt[est][j] - pt["truth"][j],
                         r - np.array([x["truth"][j] for x in reps])),
                        ("change_vs_source", pt[est][j] - pt["source_truth"][j],
                         r - np.array([x["source_truth"][j] for x in reps]))]:
                    if est == "truth" and what == "error_vs_truth":
                        continue
                    se = float(np.sqrt(np.mean((rep - point) ** 2)))
                    summ = t_summary(float(point), se, dof)
                    rows.append(dict(period_label=PERIODS[period], subgroup=sg, level=lv, role=role,
                                     method=f"{est}:{model}", label_or_gap="y_bio", metric=what,
                                     **summ, family="", q_bh=np.nan, status="ok",
                                     n_rows=int(n_masks[:, j].sum()), n_undx=n_undx))
    out = pd.DataFrame(rows)
    out.loc[out["metric"].eq("prevalence"), "p_value"] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="nhanes_out/nhanes_all.csv")
    ap.add_argument("--models", default="results/rq2_models.joblib")
    ap.add_argument("--out", default="results/bbse.csv")
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    df = load(args.csv)
    bundle = joblib.load(args.models)
    out = pd.concat([run(df, bundle, m, args.reps) for m in ("logistic", "lightgbm")], ignore_index=True)
    out.to_csv(args.out, index=False)
    v = out[(out["subgroup"] == "all") & out["metric"].isin(["prevalence", "error_vs_truth"])]
    print(v[["period_label", "method", "metric", "estimate", "ci_low", "ci_high"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
