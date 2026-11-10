"""Checks for the upgrade modules. Run from the project root after the
modules have produced their outputs:
    python -m src.test_analyses

Each block checks one module on synthetic data where the answer is known,
then on the real results files.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

from src.harness import threshold_for_share, weighted_share, score_methods, period_frame, load
from src.model_eval import weighted_metric

df = load()
base = joblib.load("results/rq2_models.joblib")

# ---------------------------------------------------------------- harness
rng = np.random.default_rng(0)
p = rng.random(5000)
w = rng.random(5000) + 0.5
for share in (0.1, 0.3336, 0.7):
    t = threshold_for_share(p, w, share)
    assert abs(weighted_share(p >= t, w) - share) < 2e-3
full = period_frame(df, "test_P")
an = (full["adult"].astype(str).str.lower().eq("true") & (full["w"] > 0)
      & full["y_diag"].notna() & full["y_bio"].notna()).to_numpy()
s = np.full(len(full), np.nan)
s[an] = base.predict_probabilities(full.loc[an], "logistic")
res = score_methods(full, {"m": s, "m2": s}, {"m": base.thresholds["logistic"], "m2": base.thresholds["logistic"]},
                    "2017-2020", n_reps=20, compare=[("m2", "m")])
ov = res[(res["subgroup"] == "all") & (res["method"] == "m")].set_index(["label_or_gap", "metric"])["estimate"]
ref = weighted_metric(full.loc[an, "y_bio"], s[an], full.loc[an, "w"], "auroc", base.thresholds["logistic"])
assert np.isclose(ov[("y_bio", "auroc")], ref)
ref = weighted_metric(full.loc[an, "undx"], s[an], full.loc[an, "w"], "sensitivity", base.thresholds["logistic"])
assert np.isclose(ov[("undx", "sensitivity")], ref)
diff = res[res["label_or_gap"].str.startswith("diff_vs_m:")]
assert np.allclose(diff["estimate"].dropna(), 0) and np.allclose(diff["se"].dropna(), 0)
me = pd.read_csv("results/model_eval.csv")
old = me[(me["period"] == "test_P") & (me["model"] == "logistic") & (me["subgroup"] == "all")
         & (me["label_or_gap"] == "undx")]["estimate"].iloc[0]
assert np.isclose(ov[("undx", "sensitivity")], old)
print("PASS harness: matched-share threshold, metrics equal model_eval, paired diff of a method with itself is 0")

# ---------------------------------------------------------------- PU models
pu = joblib.load("results/pu_models.joblib")
o = pu.oof.set_index("SEQN")
old = base.oof.set_index("SEQN").loc[o.index]
assert np.allclose(o["diag_logistic"], old["logistic"]) and np.allclose(o["diag_lightgbm"], old["lightgbm"])
rows = df[(df["period"] == "train")].set_index("SEQN").loc[o.index]
for k in ("logistic", "lightgbm"):
    a1 = roc_auc_score(rows["y_bio"], o[f"diag_{k}"], sample_weight=rows["w"])
    a2 = roc_auc_score(rows["y_bio"], o[f"en_{k}"], sample_weight=rows["w"])
    assert abs(a1 - a2) < 1e-9, (k, a1, a2)      # Elkan-Noto: monotone rescale keeps AUROC
for m, f in pu.final.items():
    if m.startswith("sar_"):
        X = df[(df["period"] == "train")]
        Xa = X[X["adult"].astype(str).str.lower().eq("true") & X["y_bio"].notna() & (X["w"] > 0)]
        mean = np.average(f.predict(Xa), weights=Xa["w"])
        assert abs(mean - pu.prior) < 0.01, (m, mean, pu.prior)
assert set(pu.final) >= {"diag_logistic", "sar_logistic", "bio_logistic", "noaccess_logistic"}
# no test-period row is used: PU training rows are exactly the 2011-2016 analysis rows
assert set(o.index) == set(base.oof["SEQN"])
print("PASS PU: diag reproduces models.py, Elkan-Noto keeps ranking, SAR mean pinned to prior, train rows only")

ev = pd.read_csv("results/pu_eval.csv", dtype={"level": str})
en = ev[ev["label_or_gap"].str.match(r"diff_vs_diag_(logistic|lightgbm)\|matched:undx$")
        & ev["method"].str.startswith("en_")]
assert (en["estimate"].abs() < 1e-9).all()
fam = ev[ev["family"] == "pu_gain"]
assert len(fam) > 0 and fam["q_bh"].between(0, 1).all()
sh = ev[(ev["label_or_gap"] == "flagged") & ev["method"].str.endswith("|matched") & (ev["period_label"] == "2011-2016")
        & (ev["subgroup"] == "all")]
assert (sh["estimate"] - 0.3336).abs().max() < 0.005
print("PASS pu_eval: matched flag share 33.4% for every method in 2011-2016, BH family filled")

# ---------------------------------------------------------------- BBSE
from src.bbse import estimates
s_ = dict(a=np.array([0.8]), b=np.array([0.2]), pa=np.array([0.5]), pb=np.array([0.1]),
          c=np.array([0.5]), prev=np.array([0.1]), diag=np.array([0.08]))
q = 0.25
t_ = dict(mu=np.array([q * 0.8 + (1 - q) * 0.2]), mup=np.array([q * 0.5 + (1 - q) * 0.1]),
          truth=np.array([q]), diag=np.array([0.2]))
e = estimates(s_, t_)
assert np.isclose(e["bbse_hard"][0], q) and np.isclose(e["bbse_soft"][0], q)
print("PASS BBSE: recovers a known prior under exact label shift")

# ---------------------------------------------------------------- screening rules
from src.screening_rules import rule_scores, ada_risk_score_reduced
toy = pd.DataFrame({"age": [30, 36, 45, 71, 50, 65], "bmi": [30, 24, 23.5, 30, np.nan, 41],
                    "race_eth": ["NH White", "NH White", "NH Asian", "NH White", "NH Black", "NH White"],
                    "sex": ["male", "female", "female", "male", "male", "male"],
                    "sbp": [120, 120, 120, 150, 135, 120], "dbp": [70, 70, 70, 95, 70, 70]})
r = rule_scores(toy)
assert list(r["uspstf2021"]) == [0, 0, 1, 0, 0, 1]
assert list(r["ada_age35"]) == [0, 1, 1, 1, 1, 1]
assert list(r["ada_soc"]) == [0, 1, 1, 1, 1, 1]
assert list(ada_risk_score_reduced(toy)) == [3, 0, 1, 7, 3, 7]
se = pd.read_csv("results/screening_eval.csv")
assert (se[se["family"] == "rule_vs_model"]["q_bh"].between(0, 1)).all()
print("PASS screening rules: USPSTF Asian BMI cut-off, ADA age rule, reduced risk score")

# ---------------------------------------------------------------- decomposition
dec = pd.read_csv("results/decomposition.csv")
b = dec[(dec["method"] == "BxF")].pivot_table(index=["period_label", "subgroup", "level"],
                                               columns="label_or_gap", values="estimate")
assert np.allclose(b["part_more_diabetes"] + b["part_more_diagnosis"], b["delta_D"])
assert np.allclose(b["D_from"], b["B_from"] * b["F_from"])
rw = dec[dec["method"] == "reweight"].pivot_table(index="period_label", columns="label_or_gap", values="estimate")
assert np.allclose(rw["B:population"] + rw["B:within"], rw["B:total"])
assert np.allclose(rw["B:total"], b.xs(("all", "all"), level=("subgroup", "level"))["delta_B"].loc[rw.index])
print("PASS decomposition: B x F split is exact, reweighting parts add to the observed change")

# ---------------------------------------------------------------- explanation
ex = pd.read_csv("results/explain_decomposition.csv")
for (per, m, sg, lv), g in ex[ex["metric"] == "logit_gap"].groupby(["period_label", "method", "subgroup", "level"]):
    tot = g[g["label_or_gap"] == "total"]["estimate"].iloc[0]
    feats = g[g["label_or_gap"].isin(["age", "education", "pir", "bmi", "waist", "insured",
                                      "routine_care", "male", "pir_missing"])]["estimate"].sum()
    assert np.isclose(tot, feats), (per, m, sg, lv)
# additivity of TreeSHAP: contributions + base = raw score
from src.explain import contributions, _with_train_mean
bb = _with_train_mean(joblib.load("results/rq2_models.joblib"), df)
X = full.loc[an].iloc[:300]
for m in ("lightgbm", "logistic"):
    C, base_v = contributions(bb, m, X)
    raw = np.log(bb.predict_probabilities(X, m) / (1 - bb.predict_probabilities(X, m)))
    assert np.allclose(C.sum(1) + base_v, raw, atol=1e-6)
print("PASS explain: feature contributions add up to each score and to the diagnosed-undiagnosed gap")

# ---------------------------------------------------------------- benchmark
tr = pd.read_csv("results/benchmark_truth.csv")
assert (tr[tr["severity"] == 0][["change_sens_undx", "change_auroc_bio"]].abs() < 1e-12).all().all()
assert (tr[tr["scenario"] == "prev_shift"]["change_auroc_bio"].abs() < 1e-9).all()   # label shift keeps AUROC
pw = pd.read_csv("results/benchmark_power.csv")
null = pw[pw["scenario"] == "null"]
assert null["detection_rate"].between(0.03, 0.07).all(), null[["monitor", "detection_rate"]]
print("PASS benchmark: zero severity reproduces the data, label shift leaves AUROC unchanged, null alarm rate ~5%")

# ---------------------------------------------------------------- rolling
ro = pd.read_csv("results/rolling_eval.csv")
assert set(ro["design"]) == {"expanding", "frozen"}
assert ro.groupby("design")["test_cycle"].nunique().min() == 4
print("PASS rolling: both designs cover the four later cycles")
print("ALL upgrade tests passed")
