"""Upgrade 1: can a model trained only on diagnosis labels recover hidden cases?

Positive-unlabeled (PU) framing. s = y_diag is the observed label: s = 1 means
diagnosed, and every diagnosed adult is a true case. s = 0 is unlabeled:
it mixes non-cases and hidden cases (y_bio = 1, y_diag = 0).

Methods, each fitted for logistic regression and LightGBM, each with
leave-one-cycle-out out-of-fold (OOF) predictions on 2011-2016 and a final
fit on all of 2011-2016:

    diag      trained on y_diag with all 9 features (the current model)
    noaccess  trained on y_diag without insurance and usual-care features.
              An ablation, so we can tell how much of any PU gain comes from
              just dropping the access features.
    en        Elkan and Noto (2008), SCAR: f(x) = g(x) / c, c = mean OOF score
              of the diag model among labelled positives. A monotone rescale,
              so ranking metrics must equal diag exactly (sanity check). Its
              useful output is the implied class prior mean(g) / c.
    sar       SAR-EM (Bekker, Robberechts and Davis 2019). Being diagnosed,
              given diabetes, depends on propensity attributes (insured,
              usual care, age, income, sex). EM alternates a classifier for
              P(y = 1 | risk features) and a propensity model
              e(x) = P(s = 1 | y = 1, x). The classifier sees the risk
              features only (age, sex, education, income, BMI, waist): the
              access features enter through the propensity model. To make the
              problem identifiable, the classifier's mean is pinned to the
              class prior pi (one number: the weighted y_bio prevalence of
              the 2011-2016 lab sample). No individual y_bio label is used.
    audit_r   SAR-EM where a random share r of training rows also has y_bio
              (a lab audit). Audited rows enter EM with their true label.
              r in {5, 10, 20, 50}%, several random draws each.
    bio       trained on y_bio: the upper bound, a blood test for everyone.

Run from the project root:
    python -m src.pu_models            # -> results/pu_models.joblib (~3-6 min)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from src.models import (TrainOnlyPreprocessor, FEATURE_NAMES, analysis_domain,
                        weighted_youden_threshold, SEED)

ACCESS = ("insured", "routine_care")
RISK_FEATURES = tuple(f for f in FEATURE_NAMES if f not in ACCESS)
PROPENSITY_FEATURES = ("insured", "routine_care", "age", "pir", "pir_missing", "male")
AUDIT_RATES = (0.05, 0.10, 0.20, 0.50)
AUDIT_DRAWS = 3
EM_ITER = 25
EM_TOL = 1e-4


def estimator(kind, n_estimators=200):
    if kind == "logistic":
        return LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED)
    return LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=15,
                          min_child_samples=40, random_state=SEED, n_jobs=1, verbosity=-1)


def raw_score(model, X):
    if isinstance(model, LogisticRegression):
        return model.decision_function(X)
    return model.predict(X, raw_score=True)


def cols(features):
    return [FEATURE_NAMES.index(f) for f in features]


@dataclass
class Fitted:
    """One fitted scorer: preprocessor + estimator + an additive logit offset
    (used by SAR-EM to pin the mean to the class prior) + optional divisor."""
    prep: TrainOnlyPreprocessor
    model: object
    features: tuple
    offset: float = 0.0
    divisor: float = 1.0      # Elkan-Noto c
    extra: dict = field(default_factory=dict)

    def predict(self, df):
        X = pd.DataFrame(self.prep.transform(df), columns=FEATURE_NAMES)[list(self.features)]
        p = expit(raw_score(self.model, X) + self.offset)
        # Elkan-Noto scores g / c are left unclipped: clipping at 1 would tie the
        # top of the ranking. Scores above 1 show that SCAR does not hold.
        return p / self.divisor


def _fit_plain(rows, y, kind, features):
    w = rows["w"].to_numpy(float)
    w = w / w.mean()
    prep = TrainOnlyPreprocessor().fit(rows)
    X = pd.DataFrame(prep.transform(rows), columns=FEATURE_NAMES)[list(features)]
    m = estimator(kind)
    m.fit(X, y, sample_weight=w)
    return Fitted(prep, m, tuple(features))


def _pin_offset(raw, w, prior):
    """Offset d so that the weighted mean of expit(raw + d) equals prior."""
    f = lambda d: np.dot(w, expit(raw + d)) / w.sum() - prior
    return brentq(f, -30, 30)


def fit_sar_em(rows, kind, prior, audit_mask=None, n_iter=EM_ITER, n_estimators=150):
    """SAR-EM with the classifier mean pinned to `prior`.

    audit_mask: rows whose y_bio is known (their latent label is fixed).
    Returns a Fitted classifier with the propensity model in .extra.
    """
    s = rows["y_diag"].to_numpy(int)
    ybio = rows["y_bio"].to_numpy(int)
    w = rows["w"].to_numpy(float)
    w = w / w.mean()
    audit = np.zeros(len(rows), bool) if audit_mask is None else np.asarray(audit_mask, bool)
    prep = TrainOnlyPreprocessor().fit(rows)
    Xall = pd.DataFrame(prep.transform(rows), columns=FEATURE_NAMES)
    Xc = Xall[list(RISK_FEATURES)]
    Xe = Xall[list(PROPENSITY_FEATURES)]
    n = len(rows)

    # start: SCAR solution (label frequency c = P(s=1)/prior)
    c0 = min(np.dot(w, s) / w.sum() / prior, 0.99)
    f = np.full(n, prior)
    e = np.full(n, c0)
    clf = prop = None
    offset = 0.0
    for it in range(n_iter):
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(s == 1, 1.0, f * (1 - e) / np.clip(1 - f * e, 1e-9, None))
        q = np.where(audit, ybio, q)
        # classifier on duplicated rows: (x, 1, w q) and (x, 0, w (1 - q))
        X2 = pd.concat([Xc, Xc], ignore_index=True)
        y2 = np.r_[np.ones(n), np.zeros(n)]
        w2 = np.r_[w * q, w * (1 - q)]
        keep = w2 > 1e-12
        clf = estimator(kind, n_estimators)
        clf.fit(X2[keep], y2[keep], sample_weight=w2[keep])
        raw = raw_score(clf, Xc)
        offset = _pin_offset(raw, w, prior) if not audit.any() else 0.0
        f_new = expit(raw + offset)
        # propensity on the (expected) positives: label s, weight w q
        pos_w = np.where(s == 1, w, w * q)
        keep_e = pos_w > 1e-12
        prop = LogisticRegression(max_iter=2000, random_state=SEED)
        prop.fit(Xe[keep_e], s[keep_e], sample_weight=pos_w[keep_e])
        e = np.clip(prop.predict_proba(Xe)[:, 1], 1e-3, 1.0)
        delta = np.max(np.abs(f_new - f))
        f = f_new
        if delta < EM_TOL:
            break
    out = Fitted(prep, clf, RISK_FEATURES, offset=offset)
    out.extra = {"propensity": prop, "propensity_features": PROPENSITY_FEATURES,
                 "iterations": it + 1, "prior": prior,
                 "mean_propensity_pos": float(np.dot(w * f, e) / np.dot(w, f))}
    return out


def method_specs(audit_draws=AUDIT_DRAWS, rates=AUDIT_RATES):
    specs = {}
    for kind in ("logistic", "lightgbm"):
        specs[f"diag_{kind}"] = ("plain", kind, "y_diag", FEATURE_NAMES)
        specs[f"noaccess_{kind}"] = ("plain", kind, "y_diag", RISK_FEATURES)
        specs[f"en_{kind}"] = ("en", kind, None, FEATURE_NAMES)
        specs[f"sar_{kind}"] = ("sar", kind, None, RISK_FEATURES)
        specs[f"bio_{kind}"] = ("plain", kind, "y_bio", FEATURE_NAMES)
        for r in rates:
            for d in range(audit_draws):
                specs[f"audit{int(r * 100):02d}d{d}_{kind}"] = ("audit", kind, (r, d), RISK_FEATURES)
    return specs


def audit_mask_for(rows, rate, draw):
    """Same audited people in every fold: decided once per SEQN."""
    rng = np.random.default_rng(SEED + 1000 * draw + int(rate * 100))
    u = pd.Series(rng.random(len(rows)), index=rows["SEQN"].to_numpy())
    return u, rate


def fit_method(spec, rows, prior, audit_u=None):
    how, kind, arg, feats = spec
    if how == "plain":
        return _fit_plain(rows, rows[arg].to_numpy(int), kind, feats)
    if how == "sar":
        return fit_sar_em(rows, kind, prior)
    if how == "audit":
        rate, draw = arg
        mask = rows["SEQN"].map(audit_u[(rate, draw)]).to_numpy() < rate
        return fit_sar_em(rows, kind, prior, audit_mask=mask)
    raise ValueError(how)


@dataclass
class PUBundle:
    final: dict            # method -> Fitted (trained on all of 2011-2016)
    oof: pd.DataFrame      # SEQN, cycle, one column per method
    own_threshold: dict    # method -> threshold at its own natural operating point
    prior: float
    en_c: dict             # kind -> Elkan-Noto c
    audit_seqn: dict       # (rate, draw) -> array of audited SEQN

    def predict(self, df, method):
        return self.final[method].predict(df)


def weighted_prior(rows):
    return float(np.average(rows["y_bio"].to_numpy(float), weights=rows["w"].to_numpy(float)))


def train_all(df, audit_draws=AUDIT_DRAWS, rates=AUDIT_RATES, only=None, verbose=True):
    rows = df.loc[(df["period"] == "train") & analysis_domain(df)].reset_index(drop=True)
    cycles = sorted(rows["cycle"].unique())
    specs = method_specs(audit_draws, rates)
    if only:
        specs = {k: v for k, v in specs.items() if any(k.startswith(o) for o in only)}
    rng_u = {}
    for r in rates:
        for d in range(audit_draws):
            rng = np.random.default_rng(SEED + 1000 * d + int(r * 100))
            rng_u[(r, d)] = pd.Series(rng.random(len(rows)), index=rows["SEQN"].to_numpy())
    prior_all = weighted_prior(rows)

    oof = pd.DataFrame({"SEQN": rows["SEQN"], "cycle": rows["cycle"]})
    final = {}
    # plain / sar / audit first; en is derived from diag afterwards
    for name, spec in specs.items():
        if spec[0] == "en":
            continue
        col = np.full(len(rows), np.nan)
        for held in cycles:
            te = rows["cycle"].eq(held).to_numpy()
            tr_rows = rows.loc[~te].reset_index(drop=True)
            fitted = fit_method(spec, tr_rows, weighted_prior(tr_rows), rng_u)
            col[te] = fitted.predict(rows.loc[te])
        oof[name] = col
        final[name] = fit_method(spec, rows, prior_all, rng_u)
        if verbose:
            print(f"  fitted {name}", flush=True)

    en_c = {}
    for kind in ("logistic", "lightgbm"):
        name = f"en_{kind}"
        if name not in specs:
            continue
        base = f"diag_{kind}"
        s = rows["y_diag"].to_numpy(int)
        w = rows["w"].to_numpy(float)
        c = float(np.average(oof[base].to_numpy()[s == 1], weights=w[s == 1]))
        en_c[kind] = c
        oof[name] = oof[base] / c
        f = final[base]
        final[name] = Fitted(f.prep, f.model, f.features, offset=f.offset, divisor=c)

    # own operating point: Youden against the label the method can see
    s = rows["y_diag"].to_numpy(int)
    yb = rows["y_bio"].to_numpy(int)
    w = rows["w"].to_numpy(float)
    own = {}
    for name, spec in specs.items():
        p = oof[name].to_numpy()
        if spec[0] == "plain" and spec[2] == "y_bio":
            own[name] = weighted_youden_threshold(yb, p, w)
        elif spec[0] == "audit":
            rate, draw = spec[2]
            m = rows["SEQN"].map(rng_u[(rate, draw)]).to_numpy() < rate
            own[name] = weighted_youden_threshold(yb[m], p[m], w[m])
        else:
            own[name] = weighted_youden_threshold(s, p, w)
    audit_seqn = {k: rows.loc[rows["SEQN"].map(u).to_numpy() < k[0], "SEQN"].to_numpy()
                  for k, u in rng_u.items()}
    return PUBundle(final, oof, own, prior_all, en_c, audit_seqn)


def main():
    ap = argparse.ArgumentParser(description="Fit PU-learning variants on 2011-2016")
    ap.add_argument("--csv", default="nhanes_out/nhanes_all.csv")
    ap.add_argument("--out", default="results/pu_models.joblib")
    ap.add_argument("--audit-draws", type=int, default=AUDIT_DRAWS)
    args = ap.parse_args()
    df = pd.read_csv(args.csv, low_memory=False)
    bundle = train_all(df, audit_draws=args.audit_draws)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"class prior (2011-2016 lab sample) = {bundle.prior:.4f}")
    for k, c in bundle.en_c.items():
        rows = df.loc[(df["period"] == "train") & analysis_domain(df)]
        g = rows["SEQN"].map(bundle.oof.set_index("SEQN")[f"diag_{k}"]).to_numpy()
        implied = np.average(g, weights=rows["w"]) / c
        above = np.average(g / c > 1, weights=rows["w"])
        print(f"Elkan-Noto {k}: c = {c:.3f}, implied prior = {implied:.4f}, "
              f"share of adults with g/c > 1 = {above:.3f} (SCAR violated if > 0)")
    print(f"saved {args.out}")


if __name__ == "__main__":
    from src.pu_models import main as canonical
    canonical()
