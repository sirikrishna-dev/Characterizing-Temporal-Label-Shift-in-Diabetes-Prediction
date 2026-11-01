
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


# Same survey-weighted proportion function used in V7, Cell 12.
# Important: domain participants are zeroed, NOT removed.
def svy_prop(d, y, w, domain=None):
    """Calculate weighted proportion and Taylor standard error."""

    in_dom = (d[w].fillna(0) > 0) & d[y].notna()

    if domain is not None:
        in_dom &= domain

    wt = np.where(in_dom, d[w].fillna(0), 0.0)
    yv = np.where(in_dom, d[y].fillna(0), 0.0)

    W = wt.sum()

    if W == 0:
        return np.nan, np.nan, 0

    p = (wt * yv).sum() / W
    z = wt * (yv - p) / W

    tot = pd.DataFrame({
        "h": d["stratum"].values,
        "psu": d["psu"].values,
        "z": z
    }).groupby(["h", "psu"])["z"].sum().reset_index()

    var = 0.0

    for _, g in tot.groupby("h"):
        n = len(g)

        if n > 1:
            var += (
                n / (n - 1)
                * ((g["z"] - g["z"].mean()) ** 2).sum()
            )

    return p, np.sqrt(var), int(in_dom.sum())


# Primary subgroups specified in V7.
PRIMARY_SUBGROUPS = [
    "income_band2",
    "insured",
    "age_band2",
    "sex"
]

# All agreed subgroup variables from V7.
SUBGROUPS = [
    "income_band",
    "income_band2",
    "insured",
    "routine_care",
    "age_band",
    "age_band2",
    "sex",
    "race_eth",
    "race_eth4"
]

CYCLE_ORDER = [
    "2011-2012",
    "2013-2014",
    "2015-2016",
    "2017-2020",
    "2021-2023"
]


def create_pilot_table():

    # Load V7 analysis dataset.
    df = pd.read_csv("nhanes_out/nhanes_all.csv", low_memory=False)

    # A CSV round trip can turn the boolean adult flag into text.
    if df["adult"].dtype != bool:
        df["adult"] = df["adult"].astype(str).str.lower().eq("true")

    required = [
        "cycle", "period", "adult", "stratum", "psu",
        "w_int", "w_lab", "y_diag", "y_bio", "screened"
    ] + SUBGROUPS

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["adult"].isna().any():
        raise ValueError("The adult column contains missing values.")

    rows = []

    # Process each cycle separately.
    for cycle in CYCLE_ORDER:

        d = df[df["cycle"] == cycle].copy()

        if d.empty:
            raise ValueError(f"No participants found for cycle {cycle}")

        # Keep all cycle records for survey variance calculation.
        dof = d["psu"].nunique() - d["stratum"].nunique()

        if dof <= 0:
            raise ValueError(f"Invalid survey degrees of freedom in {cycle}")

        t_value = stats.t.ppf(0.975, dof)

        # Overall adult population plus subgroup levels.
        domains = [
            ("all_adults", "All adults (20+, non-pregnant)",
             d["adult"])
        ]

        for subgroup in SUBGROUPS:

            levels = sorted(
                d.loc[d["adult"], subgroup].dropna().unique(),
                key=str
            )

            for level in levels:

                domain = (
                    d["adult"]
                    & d[subgroup].eq(level)
                )

                domains.append((subgroup, str(level), domain))

        # Biomarker-positive adults (diagnosed or HbA1c >= 6.5%).
        # Fraction diagnosed is measured only inside this group.
        bio_positive = d["y_bio"].eq(1)

        # Three measures for every domain:
        #   diagnosed_prevalence: share of all adults told they have
        #       diabetes (interview weight)
        #   fraction_diagnosed: share of biomarker-positive adults who
        #       were diagnosed (lab weight, because y_bio needs HbA1c)
        #   screening_coverage_undiagnosed: share of undiagnosed adults
        #       tested in the past 3 years (DIQ180, interview weight)
        for subgroup, level, domain in domains:

            measures = [
                ("diagnosed_prevalence", "y_diag", "w_int", domain),
                ("fraction_diagnosed", "y_diag", "w_lab",
                 domain & bio_positive),
                ("screening_coverage_undiagnosed", "screened", "w_int",
                 domain),
            ]

            for measure, outcome, weight, measure_domain in measures:

                estimate, se, n = svy_prop(
                    d=d,
                    y=outcome,
                    w=weight,
                    domain=measure_domain
                )

                rows.append({
                    "cycle": cycle,
                    "period": d["period"].iloc[0],
                    "subgroup": subgroup,
                    "level": level,
                    "role": (
                        "overall" if subgroup == "all_adults"
                        else "primary" if subgroup in PRIMARY_SUBGROUPS
                        else "exploratory"
                    ),
                    "measure": measure,
                    "weight": weight,
                    "n_valid": n,
                    "dof": dof,
                    "estimate_pct": 100 * estimate,
                    "se_pct": 100 * se,
                    "ci_low_pct": 100 * (estimate - t_value * se),
                    "ci_high_pct": 100 * (estimate + t_value * se)
                })

    # Create and save the final pilot table.
    pilot_table = pd.DataFrame(rows)

    output_path = Path("results")
    output_path.mkdir(parents=True, exist_ok=True)

    pilot_table.to_csv(
        output_path / "pilot_table.csv",
        index=False
    )

    print("Survey-weighted pilot table created!")
    print("Total rows:", len(pilot_table))
    print("\nFirst 20 results:")
    print(pilot_table.head(20).to_string(index=False))

    print("\nSaved to results/pilot_table.csv")


if __name__ == "__main__":
    create_pilot_table()
