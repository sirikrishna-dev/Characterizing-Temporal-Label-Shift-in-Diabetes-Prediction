"""The three ADA Diabetes Risk Test items the main file does not carry.

Run from the project root, on a machine that can reach wwwn.cdc.gov:
    python -m src.ada_items

Writes nhanes_out/ada_items.csv (one row per SEQN, 2011-2023). When that file
exists, python -m src.screening_rules scores the full seven-item ADA risk test
next to the reduced one.

Items (checked against each cycle's CDC codebook)
    ada_family_history  MCQ300C "close relative had diabetes": 1 yes -> 1,
                        2 no -> 0, refused / don't know -> missing.
                        Asked of adults 20+.
    ada_gdm             RHQ162 "during pregnancy ever told ... diabetes or
                        gestational diabetes": 1 yes -> 1, 2 no or 3 borderline
                        -> 0, else missing. Women only. The public RHQ files
                        hold complete pregnancy records only for women 20-44,
                        so older women mostly have this item missing.
    ada_active          any moderate or vigorous leisure-time activity.
                        2011-2020: PAQ650 or PAQ665 = 1 -> 1, both = 2 -> 0.
                        2021-2023: PAD790Q > 0 or PAD810Q > 0 -> 1, both 0 -> 0
                        (7777 refused and 9999 don't know -> missing).
    ada_items_complete  True when the cycle has all three questions.

CDC dropped MCQ300C and RHQ162 from the 2021-2023 questionnaires (see the
Analytic Notes of MCQ_L and RHQ_L), so 2021-2023 rows get those two items
missing and ada_items_complete = False. The full test is therefore scored for
2011-2016 and 2017-2020 only.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# cycle -> (url start year, file names for MCQ, RHQ, PAQ)
FILES = {
    "2011-2012": (2011, {"MCQ": "MCQ_G", "RHQ": "RHQ_G", "PAQ": "PAQ_G"}),
    "2013-2014": (2013, {"MCQ": "MCQ_H", "RHQ": "RHQ_H", "PAQ": "PAQ_H"}),
    "2015-2016": (2015, {"MCQ": "MCQ_I", "RHQ": "RHQ_I", "PAQ": "PAQ_I"}),
    "2017-2020": (2017, {"MCQ": "P_MCQ", "RHQ": "P_RHQ", "PAQ": "P_PAQ"}),
    "2021-2023": (2021, {"MCQ": "MCQ_L", "RHQ": "RHQ_L", "PAQ": "PAQ_L"}),
}
NOT_ASKED_2021 = ("MCQ300C", "RHQ162")


def download(name, year, raw_dir):
    import requests
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"{name}.xpt")
    if os.path.exists(path) and open(path, "rb").read(13) == b"HEADER RECORD":
        return path
    for ext in ("xpt", "XPT"):
        url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.{ext}"
        r = requests.get(url, timeout=120)
        if r.ok and r.content[:13] == b"HEADER RECORD":
            with open(path, "wb") as fh:
                fh.write(r.content)
            return path
    raise FileNotFoundError(f"{name}.xpt not found for {year}")


def _code(series, yes, no):
    v = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.select([v.isin(yes), v.isin(no)], [1.0, 0.0], np.nan), index=series.index)


def items_from_frames(mcq, rhq, paq):
    """Pure function (no downloads) so the coding can be tested."""
    out = pd.DataFrame({"SEQN": pd.concat([mcq["SEQN"], rhq["SEQN"], paq["SEQN"]]).unique()})
    out = out.merge(mcq[["SEQN"] + [c for c in ("MCQ300C",) if c in mcq]], on="SEQN", how="left")
    out = out.merge(rhq[["SEQN"] + [c for c in ("RHQ162",) if c in rhq]], on="SEQN", how="left")
    pa_cols = [c for c in ("PAQ650", "PAQ665", "PAD790Q", "PAD810Q") if c in paq]
    out = out.merge(paq[["SEQN"] + pa_cols], on="SEQN", how="left")

    out["ada_family_history"] = _code(out["MCQ300C"], [1], [2]) if "MCQ300C" in out else np.nan
    out["ada_gdm"] = _code(out["RHQ162"], [1], [2, 3]) if "RHQ162" in out else np.nan
    if "PAQ650" in out:
        vig = _code(out["PAQ650"], [1], [2])
        mod = _code(out["PAQ665"], [1], [2])
    else:
        def freq(c):
            v = pd.to_numeric(out[c], errors="coerce")
            return pd.Series(np.select([v.between(1, 7776), v.eq(0)], [1.0, 0.0], np.nan), index=out.index)
        vig, mod = freq("PAD810Q"), freq("PAD790Q")
    active = np.where((vig == 1) | (mod == 1), 1.0, np.where((vig == 0) & (mod == 0), 0.0, np.nan))
    out["ada_active"] = active
    out["ada_items_complete"] = ("MCQ300C" in out) and ("RHQ162" in out) and bool(pa_cols)
    return out[["SEQN", "ada_family_history", "ada_gdm", "ada_active", "ada_items_complete"]]


def main():
    ap = argparse.ArgumentParser(description="Download and code the extra ADA risk test items")
    ap.add_argument("--raw", default="nhanes_raw")
    ap.add_argument("--out", default="nhanes_out/ada_items.csv")
    args = ap.parse_args()
    parts = []
    for cycle, (year, names) in FILES.items():
        frames = {k: pd.read_sas(download(n, year, args.raw), format="xport") for k, n in names.items()}
        for col in NOT_ASKED_2021:
            for f in frames.values():
                if cycle == "2021-2023" and col in f:
                    raise ValueError(f"{col} unexpectedly present in {cycle}: recheck the codebook")
        it = items_from_frames(frames["MCQ"], frames["RHQ"], frames["PAQ"])
        it.insert(1, "cycle", cycle)
        parts.append(it)
        print(f"{cycle}: {len(it)} people, complete items = {bool(it['ada_items_complete'].iloc[0])}")
    out = pd.concat(parts, ignore_index=True)
    out["SEQN"] = out["SEQN"].astype(int)
    out.to_csv(args.out, index=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
