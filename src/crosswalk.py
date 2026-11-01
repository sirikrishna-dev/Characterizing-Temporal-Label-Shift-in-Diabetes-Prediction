
import pandas as pd


# Raw NHANES column names and their standardized names
CROSSWALK = {
    "SEQN": "SEQN",
    "RIDAGEYR": "age",
    "RIAGENDR": "sex",
    "RIDRETH3": "race_eth",
    "INDFMPIR": "pir",
    "BMXBMI": "bmi",
    "BMXWAIST": "waist",
    "LBXGH": "hba1c",
    "LBXGLU": "fpg",
    "SDMVSTRA": "stratum",
    "SDMVPSU": "psu",
}


# NHANES sex codes
SEX_LABELS = {
    1: "male",
    2: "female",
}


# NHANES race/ethnicity codes
RACE_LABELS = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "NH White",
    4: "NH Black",
    6: "NH Asian",
    7: "Other/Multi",
}


# Columns required in the standardized dataset
REQUIRED_COLUMNS = [
    "SEQN",
    "cycle",
    "period",
    "age",
    "sex",
    "race_eth",
    "pir",
    "bmi",
    "hba1c",
    "fpg",
    "psu",
    "stratum",
]


def apply_crosswalk(df):
    """
    Convert raw NHANES column names and values
    into standardized column names and values.

    Also supports data that is already standardized.
    """

    df = df.copy()

    # Convert sex codes before renaming
    if "RIAGENDR" in df.columns:
        df["sex"] = pd.to_numeric(
            df["RIAGENDR"], errors="coerce"
        ).map(SEX_LABELS)

    # Convert race/ethnicity codes before renaming
    if "RIDRETH3" in df.columns:
        df["race_eth"] = pd.to_numeric(
            df["RIDRETH3"], errors="coerce"
        ).map(RACE_LABELS)

    # Rename the remaining raw columns
    rename_map = {
        raw: standard
        for raw, standard in CROSSWALK.items()
        if raw in df.columns
        and standard not in df.columns
        and raw not in [
            "RIAGENDR",
            "RIDRETH3",
            "SDMVSTRA",
            "SDMVPSU",
        ]
    }

    df = df.rename(columns=rename_map)

    # Create unique survey strata and PSU identifiers
    if "SDMVSTRA" in df.columns and "SDMVPSU" in df.columns:
        df["stratum"] = (
            df["cycle"].astype(str)
            + "_"
            + df["SDMVSTRA"].astype("Int64").astype(str)
        )

        df["psu"] = (
            df["stratum"]
            + "_"
            + df["SDMVPSU"].astype("Int64").astype(str)
        )

    return df


def validate_crosswalk(df):
    """
    Check whether all required standardized columns exist.
    """

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("Crosswalk validation successful.")
    print("All required standardized columns are available.")

    return True
