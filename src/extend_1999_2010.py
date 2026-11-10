"""Upgrade 4: build nhanes_all_1999_2023.csv (six more cycles, 1999-2010).

STATUS: written without CDC access (the build environment could not reach
wwwn.cdc.gov), so the download and merge have NOT been run. Every file name,
variable name and code below must be checked against that cycle's codebook
before the output is used; items marked CONFIRM are the ones most likely to
need a fix. The output keeps exactly the columns of nhanes_out/nhanes_all.csv,
so every src module reruns on it with --csv nhanes_out/nhanes_all_1999_2023.csv.

Run (from the project root, on a machine that can reach CDC):
    python -m src.extend_1999_2010 --raw nhanes_raw --out nhanes_out/nhanes_all_1999_2023.csv

Known breaks and how they are handled
    file names   HbA1c and fasting glucose are LAB10 / LAB10AM (1999-2000),
                 L10_B / L10AM_B (2001-02), L10_C / L10AM_C (2003-04), then
                 GHB_D/E/F and GLU_D/E/F. Handled by FILE_MAP.
    weights      2-year MEC weights (WTMEC2YR) for every cycle by default. CDC
                 provides WTMEC4YR for 1999-2002 combined; with
                 --four-year-1999-2002 those two cycles use WTMEC4YR / 2 each
                 when pooled (CONFIRM against the analytic guidelines).
                 Pooling across cycles: w_lab * cycle_years / total_years, as in
                 src/rolling.py.
    race         RIDRETH3 (separate NH Asian) starts in 2011. The long file
                 uses RIDRETH1 for every cycle: race_eth_h has no Asian level
                 (NH Asian falls into "Other/Multi"). race_eth/race_eth4 keep
                 their meaning for 2011+ and are set from RIDRETH1 before 2011.
    insurance    HIQ011 from 2005-06 (CONFIRM); HID010 in 1999-2004. Both map
                 to insured = 1 (yes) / 0 (no).
    screening    DIQ180 (blood test in past 3 years) is only asked in some
                 cycles (CONFIRM which). Cycles without it get screened = NaN
                 and are left out of the screening trend, never imputed.
    lab methods  Check each GHB / LAB10 documentation for recommended
                 cross-cycle adjustments. Fasting glucose has documented
                 adjustment equations for some cycles (CONFIRM and apply in
                 GLUCOSE_ADJUST); HbA1c adjustments: CONFIRM none are needed.
    BP           auscultatory BPX file for every cycle before 2017.
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
GLUCOSE_ADJUST = {}   # cycle -> function(fpg) ; fill from CDC lab documentation (CONFIRM)
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


def harmonise(raw, four_year=False):
    from src.labels import add_labels, add_subgroups
    d = raw.copy()
    cyc = d["cycle"].iloc[0]
    # insurance: one raw column for labels.add_subgroups
    if "HIQ011" not in d or d["HIQ011"].isna().all():
        d["HIQ011"] = d.get("HID010")
    if cyc in GLUCOSE_ADJUST and "LBXGLU" in d:
        d["LBXGLU"] = GLUCOSE_ADJUST[cyc](d["LBXGLU"])
    if "DIQ180" not in d:
        d["DIQ180"] = np.nan
    # weights
    mec = "WTMEC4YR" if (four_year and cyc in ("1999-2000", "2001-2002")) else "WTMEC2YR"
    d["w_mec"] = d[mec] / (2.0 if mec == "WTMEC4YR" else 1.0)
    d["w_lab"] = d["w_mec"]
    d["w_int"] = d["WTINT4YR"] / 2.0 if (four_year and "WTINT4YR" in d and cyc in ("1999-2000", "2001-2002")) else d.get("WTINT2YR")
    d["w_fast"] = d.get("WTSAF4YR", d.get("WTSAF2YR")) if four_year else d.get("WTSAF2YR")
    d["w_fast"] = pd.to_numeric(d["w_fast"], errors="coerce").fillna(0)
    # race (no NH Asian before 2011)
    d["race_eth"] = pd.to_numeric(d["RIDRETH1"], errors="coerce").map(RIDRETH1_LABELS)
    d["race_eth4"] = d["race_eth"].map(RACE4)
    d = add_labels(d)
    d = add_subgroups(d)
    out = pd.DataFrame({
        "SEQN": d["SEQN"].astype(int), "cycle": cyc, "period": "early",
        "stratum": cyc + "_" + d["SDMVSTRA"].astype(int).astype(str),
        "psu": cyc + "_" + d["SDMVSTRA"].astype(int).astype(str) + "_" + d["SDMVPSU"].astype(int).astype(str),
        "w_int": d["w_int"], "w_mec": d["w_mec"], "w_lab": d["w_lab"], "w_fast": d["w_fast"],
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="nhanes_raw")
    ap.add_argument("--current", default="nhanes_out/nhanes_all.csv")
    ap.add_argument("--out", default="nhanes_out/nhanes_all_1999_2023.csv")
    ap.add_argument("--four-year-1999-2002", action="store_true")
    args = ap.parse_args()
    current = pd.read_csv(args.current, low_memory=False)
    early = [harmonise(load_cycle(c, args.raw), args.four_year_1999_2002) for c in EARLY]
    current["race_eth_h"] = current["race_eth"].replace({"NH Asian": "Other/Multi"})
    cols = list(current.columns)
    long = pd.concat([e.reindex(columns=cols) for e in early] + [current], ignore_index=True)
    long.to_csv(args.out, index=False)
    # validation table to compare with Menke et al. 2015 and NCHS reports
    rows = []
    for c, g in long.groupby("cycle"):
        a = g[g["adult"].astype(bool) & g["lab_sample"].astype(bool) & (g["w_lab"] > 0)]
        w = a["w_lab"]
        rows.append(dict(cycle=c, n=len(a), weight_sum_millions=w.sum() / 1e6,
                         diagnosed=np.average(a["y_diag"], weights=w),
                         biomarker=np.average(a["y_bio"], weights=w),
                         undiagnosed=np.average(a["undx"], weights=w)))
    pd.DataFrame(rows).to_csv(args.out.replace(".csv", "_validation.csv"), index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
