"""Upgrade 4: build nhanes_all_1999_2023.csv (six more cycles, 1999-2010).

Run from the project root:
    python -m src.extend_1999_2010
The CDC files are read from nhanes_raw/ when they are already there and are
downloaded otherwise. The output keeps every column of nhanes_out/nhanes_all.csv
plus w_lab_4yr, so every src module reruns on it with
--csv nhanes_out/nhanes_all_1999_2023.csv.

Cycle breaks, each checked against the CDC documentation
    file names   HbA1c and fasting glucose are LAB10 / LAB10AM (1999-2000),
                 L10_B / L10AM_B (2001-02), L10_C / L10AM_C (2003-04), then
                 GHB_D/E/F and GLU_D/E/F. Handled by FILE_MAP.
    weights      CDC's weighting tutorial: any analysis that combines
                 1999-2000 with 2001-2002 must start from the 4-year weights
                 (WTMEC4YR), because the 2-year weights of those two cycles
                 are not comparable. w_lab keeps the 2-year MEC weight, which
                 is right for one cycle on its own. w_lab_4yr holds
                 2 x WTMEC4YR for 1999-2000 and 2001-2002 (the same scale as
                 a 2-year weight), so the usual years / total years pooling
                 reproduces CDC's formula (for example 2/3 x WTMEC4YR for a
                 1999-2004 combination). src/rolling.py switches to it
                 whenever both cycles are pooled.
    race         RIDRETH3 (separate NH Asian) starts in 2011. The long file
                 uses RIDRETH1 for every cycle: race_eth_h has no Asian level
                 (NH Asian falls into "Other/Multi").
    insurance    HID010 in 1999-2004 and HIQ011 from 2005-2006. Same question
                 text and codes; CDC's HIQ_D notes call them comparable. Both
                 map to insured = 1 (yes) / 0 (no).
    screening    DIQ180 (blood test in the past 3 years) is asked from
                 2005-2006 on (DIQ_D, DIQ_E); 1999-2004 files do not have it.
                 Those cycles get screened = NaN, never imputed.
    HbA1c        No adjustment. CDC withdrew the 2003-2004 / 2005-2006
                 cross-over equation and says trend analyses of HbA1c should
                 use the original data (A1c web notice).
    glucose      CDC-released values, the same choice as the 2011-2023
                 notebook. GLU_D and GLU_E give regression equations for
                 trending fasting glucose across instruments; they are not
                 applied because only y_bio_fpg uses glucose, and no analysis
                 on the long file uses y_bio_fpg. Anyone who does must apply
                 those equations first.
    BP           auscultatory BPX file for every cycle before 2017.

Validation (nhanes_out/nhanes_all_1999_2023_validation.csv): weighted
diagnosed prevalence among non-pregnant adults 20+ for 1999-2002, 2003-2006,
2007-2010 and 2011-2014, with Taylor-linearised 95% CIs, next to the crude
estimates of Selvin et al. 2017 (Ann Intern Med, Table 1; same population,
fasting subsample). Biomarker and undiagnosed prevalence are listed per cycle
but have no published match: Selvin et al. define undiagnosed diabetes by
HbA1c and fasting glucose together, not by HbA1c alone.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

EARLY = {
    # cycle: (url year, suffix, legacy folder, period)
    "1999-2000": (1999, "", "1999-2000", "early"),
    "2001-2002": (2001, "_B", "2001-2002", "early"),
    "2003-2004": (2003, "_C", "2003-2004", "early"),
    "2005-2006": (2005, "_D", "2005-2006", "early"),
    "2007-2008": (2007, "_E", "2007-2008", "early"),
    "2009-2010": (2009, "_F", "2009-2010", "early"),
}
FILE_MAP = {
    "1999-2000": {"GHB": "LAB10", "GLU": "LAB10AM"},
    "2001-2002": {"GHB": "L10_B", "GLU": "L10AM_B"},
    "2003-2004": {"GHB": "L10_C", "GLU": "L10AM_C"},
}
COMPONENTS = {
    "DEMO": ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "DMDEDUC2", "INDFMPIR", "RIDEXPRG",
             "SDMVSTRA", "SDMVPSU"],
    "DIQ": ["SEQN", "DIQ010", "DIQ180", "DIQ050", "DIQ070"],
    "HIQ": ["SEQN", "HIQ011", "HID010"],
    "HUQ": ["SEQN", "HUQ030"],
    "BMX": ["SEQN", "BMXBMI", "BMXWAIST"],
    "BPX": ["SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"],
    "GHB": ["SEQN", "LBXGH"],
    "GLU": ["SEQN", "LBXGLU"],
}
# Selvin et al. 2017, Ann Intern Med 167(11):769-776, Table 1: diagnosed
# diabetes, % (SE), non-pregnant adults 20+, crude.
PUBLISHED_DIAGNOSED = {
    "1999-2002": (6.35, 0.53, ("1999-2000", "2001-2002")),
    "2003-2006": (7.57, 0.52, ("2003-2004", "2005-2006")),
    "2007-2010": (8.52, 0.53, ("2007-2008", "2009-2010")),
    "2011-2014": (9.60, 0.69, ("2011-2012", "2013-2014")),
}
FOUR_YEAR = ("1999-2000", "2001-2002")
RIDRETH1_LABELS = {1: "Mexican American", 2: "Other Hispanic", 3: "NH White", 4: "NH Black",
                   5: "Other/Multi"}
RACE4 = {"NH White": "NH White", "NH Black": "NH Black", "Mexican American": "Hispanic",
         "Other Hispanic": "Hispanic", "Other/Multi": "Other/Multi", "NH Asian": "Other/Multi"}


def file_name(component, cycle):
    if cycle in FILE_MAP and component in FILE_MAP[cycle]:
        return FILE_MAP[cycle][component]
    suffix = EARLY[cycle][1]
    return f"{component}{suffix}"


def urls(name, cycle):
    year, _, legacy, _ = EARLY[cycle]
    return [f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt",
            f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.XPT",
            f"https://wwwn.cdc.gov/Nchs/Nhanes/{legacy}/{name}.XPT"]


def download(name, cycle, raw_dir):
    import requests
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"{name}.xpt")
    if os.path.exists(path) and open(path, "rb").read(13) == b"HEADER RECORD":
        return path
    for u in urls(name, cycle):
        r = requests.get(u, timeout=120)
        if r.ok and r.content[:13] == b"HEADER RECORD":
            open(path, "wb").write(r.content)
            return path
    raise FileNotFoundError(f"{name} ({cycle}) not found at {urls(name, cycle)}")


def load_cycle(cycle, raw_dir):
    merged = None
    for comp, keep in COMPONENTS.items():
        d = pd.read_sas(download(file_name(comp, cycle), cycle, raw_dir), format="xport")
        wt = [c for c in d.columns if c.startswith("WT")]
        d = d[[c for c in keep if c in d.columns] + wt]
        if comp == "DEMO":
            merged = d
        else:
            dup = [c for c in d.columns if c in merged.columns and c != "SEQN"]
            merged = merged.merge(d.drop(columns=dup), on="SEQN", how="left")
    merged["cycle"] = cycle
    return merged


def harmonise(raw):
    from src.labels import add_labels, add_subgroups
    d = raw.copy()
    cyc = d["cycle"].iloc[0]
    # insurance: one raw column for labels.add_subgroups
    if "HIQ011" not in d or d["HIQ011"].isna().all():
        d["HIQ011"] = d.get("HID010")
    if "DIQ180" not in d:
        d["DIQ180"] = np.nan
    # weights: 2-year weights for single-cycle use; 4-year weights (on the
    # 2-year scale) for pooling 1999-2000 with 2001-2002
    d["w_mec"] = pd.to_numeric(d["WTMEC2YR"], errors="coerce")
    d["w_lab"] = d["w_mec"]
    d["w_lab_4yr"] = 2.0 * pd.to_numeric(d["WTMEC4YR"], errors="coerce") if cyc in FOUR_YEAR else np.nan
    d["w_int"] = pd.to_numeric(d["WTINT2YR"], errors="coerce")
    d["w_fast"] = pd.to_numeric(d.get("WTSAF2YR"), errors="coerce").fillna(0)
    # race (no NH Asian before 2011)
    d["race_eth"] = pd.to_numeric(d["RIDRETH1"], errors="coerce").map(RIDRETH1_LABELS)
    d["race_eth4"] = d["race_eth"].map(RACE4)
    d = add_labels(d)
    d = add_subgroups(d)
    out = pd.DataFrame({
        "SEQN": d["SEQN"].astype(int), "cycle": cyc, "period": "early",
        "stratum": cyc + "_" + d["SDMVSTRA"].astype(int).astype(str),
        "psu": cyc + "_" + d["SDMVSTRA"].astype(int).astype(str) + "_" + d["SDMVPSU"].astype(int).astype(str),
        "w_int": d["w_int"], "w_mec": d["w_mec"], "w_lab": d["w_lab"], "w_lab_4yr": d["w_lab_4yr"],
        "w_fast": d["w_fast"],
        "pool_factor": 1.0, "w": d["w_lab"], "w_int_pooled": d["w_int"], "w_fast_pooled": d["w_fast"],
        "age": d["RIDAGEYR"], "pir": d["INDFMPIR"], "education": d["DMDEDUC2"].where(d["DMDEDUC2"] <= 5),
        "bmi": d["BMXBMI"], "waist": d["BMXWAIST"],
        "sbp": d[[c for c in d if c.startswith("BPXSY")]].mean(axis=1),
        "dbp": d[[c for c in d if c.startswith("BPXDI")]].replace(0, np.nan).mean(axis=1),
        "bp_method": "auscultatory", "hba1c": d["LBXGH"], "fpg": d.get("LBXGLU"),
    })
    preg = pd.to_numeric(d["RIDEXPRG"], errors="coerce").eq(1)
    out["adult"] = (pd.to_numeric(d["RIDAGEYR"]) >= 20) & ~preg
    out["lab_sample"] = d["y_diag"].notna() & d["LBXGH"].notna()
    for c in ("y_diag", "y_bio", "y_bio_any", "undx", "y_bio_fpg", "undx_fpg", "screened",
              "insured", "routine_care", "income_band", "income_band2", "age_band", "age_band2",
              "sex", "race_eth", "race_eth4"):
        out[c] = d[c].values
    out["y_diag_fpg"] = np.where(out["y_bio_fpg"].notna(), out["y_diag"], np.nan)
    out["usual_source_care"] = out["routine_care"]
    out["race_eth_h"] = out["race_eth"].replace({"NH Asian": "Other/Multi"})
    return out


def svy_mean(d, y, w):
    """Weighted proportion and Taylor-linearised SE (strata, PSUs)."""
    from src.pilot_table import svy_prop
    p, se, n = svy_prop(d, y, w, domain=None)
    return p, se, n


def validation(long):
    """Diagnosed prevalence by 4-year period next to Selvin et al. 2017, plus
    per-cycle biomarker and undiagnosed prevalence."""
    adult = long["adult"].astype(str).str.lower().eq("true") & long["lab_sample"].astype(str).str.lower().eq("true")
    rows = []
    for period, (pub, pub_se, cycles) in PUBLISHED_DIAGNOSED.items():
        d = long[long["cycle"].isin(cycles)].copy()
        a = adult.loc[d.index]
        base = "w_lab_4yr" if set(cycles) == set(FOUR_YEAR) else "w_lab"
        d["wv"] = np.where(a, pd.to_numeric(d[base], errors="coerce").fillna(0) / len(cycles), 0.0)
        p, se, n = svy_mean(d, "y_diag", "wv")
        from scipy.stats import t as student_t
        dof = d["psu"].nunique() - d["stratum"].nunique()
        crit = student_t.ppf(0.975, dof)
        rows.append(dict(level="period", period=period, weight=base, n=n,
                         diagnosed_pct=100 * p, ci_low=100 * (p - crit * se), ci_high=100 * (p + crit * se),
                         published_pct=pub, published_ci_low=pub - 1.96 * pub_se,
                         published_ci_high=pub + 1.96 * pub_se,
                         intervals_overlap=bool(100 * (p - crit * se) <= pub + 1.96 * pub_se
                                                and pub - 1.96 * pub_se <= 100 * (p + crit * se))))
    for c, g in long.groupby("cycle"):
        a = g[adult.loc[g.index] & (pd.to_numeric(g["w_lab"], errors="coerce") > 0)]
        w = pd.to_numeric(a["w_lab"])
        rows.append(dict(level="cycle", period=c, weight="w_lab", n=len(a),
                         weight_sum_millions=w.sum() / 1e6,
                         diagnosed_pct=100 * np.average(a["y_diag"], weights=w),
                         biomarker_pct=100 * np.average(a["y_bio"], weights=w),
                         undiagnosed_pct=100 * np.average(a["undx"], weights=w)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="nhanes_raw")
    ap.add_argument("--current", default="nhanes_out/nhanes_all.csv")
    ap.add_argument("--out", default="nhanes_out/nhanes_all_1999_2023.csv")
    args = ap.parse_args()
    current = pd.read_csv(args.current, low_memory=False)
    early = [harmonise(load_cycle(c, args.raw)) for c in EARLY]
    current["race_eth_h"] = current["race_eth"].replace({"NH Asian": "Other/Multi"})
    current["w_lab_4yr"] = np.nan
    cols = list(current.columns)
    long = pd.concat([e.reindex(columns=cols) for e in early] + [current], ignore_index=True)
    long.to_csv(args.out, index=False)
    val = validation(long)
    val.to_csv(args.out.replace(".csv", "_validation.csv"), index=False)
    print(val.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
