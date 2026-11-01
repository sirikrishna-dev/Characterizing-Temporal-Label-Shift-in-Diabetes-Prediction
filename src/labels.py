
import numpy as np
import pandas as pd


# Diabetes outcome labels
LABEL_COLUMNS = [
    "y_diag",
    "y_bio",
    "y_bio_any",
    "y_bio_fpg",
    "undx",
    "undx_fpg",
    "screened",
]


# Subgroup variables
SUBGROUP_COLUMNS = [
    "income_band",
    "income_band2",
    "insured",
    "routine_care",
    "age_band",
    "age_band2",
    "sex",
    "race_eth",
    "race_eth4",
]


# Definitions from the V7 pipeline
LABEL_DEFINITIONS = {
    "y_diag": "Self-reported diabetes diagnosis.",
    "y_bio": "Diagnosis OR HbA1c >= 6.5%; requires valid diagnosis and HbA1c.",
    "y_bio_any": "Diagnosed participants remain positive even when HbA1c is missing.",
    "y_bio_fpg": "Diagnosis OR HbA1c >= 6.5% OR fasting glucose >= 126 mg/dL.",
    "undx": "Not diagnosed, but HbA1c >= 6.5%.",
    "undx_fpg": "Not diagnosed, but HbA1c >= 6.5% or fasting glucose >= 126 mg/dL.",
    "screened": "Blood sugar test in the past three years among undiagnosed participants.",
}


RACE_LABELS = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "NH White",
    4: "NH Black",
    6: "NH Asian",
    7: "Other/Multi",
}


RACE_LABELS_4 = {
    1: "Hispanic",
    2: "Hispanic",
    3: "NH White",
    4: "NH Black",
    6: "NH Asian/Other",
    7: "NH Asian/Other",
}


def _require_columns(df, columns, purpose):
    """Check that the raw columns needed for a calculation exist."""
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Cannot create {purpose}. Missing columns: {missing}"
        )


def add_labels(df):
    """
    Create diabetes labels using the V7 definitions.

    Existing V7-generated labels are preserved.
    Missing labels are created from raw NHANES variables.
    """

    df = df.copy()

    # Create diagnosis label if it is not already available
    if "y_diag" not in df.columns:
        _require_columns(df, ["DIQ010"], "y_diag")

        diagnosis = pd.to_numeric(df["DIQ010"], errors="coerce")
        df["y_diag"] = diagnosis.map({
            1: 1.0,
            2: 0.0,
            3: 0.0,
        })

    # Create HbA1c-based labels if they are missing
    hba1c_labels = ["y_bio", "undx", "y_bio_any"]

    if any(col not in df.columns for col in hba1c_labels):
        _require_columns(df, ["LBXGH"], "HbA1c labels")

        hba1c = pd.to_numeric(df["LBXGH"], errors="coerce")
        a1c_positive = hba1c >= 6.5
        has_a1c = hba1c.notna()
        answered = df["y_diag"].notna()

        lab_sample = answered & has_a1c

        if "y_bio" not in df.columns:
            df["y_bio"] = (
                ((df["y_diag"] == 1) | a1c_positive)
                .astype(float)
                .where(lab_sample)
            )

        if "undx" not in df.columns:
            df["undx"] = (
                ((df["y_diag"] == 0) & a1c_positive)
                .astype(float)
                .where(lab_sample)
            )

        if "y_bio_any" not in df.columns:
            df["y_bio_any"] = np.select(
                [
                    df["y_diag"].eq(1),
                    df["y_diag"].eq(0) & has_a1c,
                ],
                [
                    1.0,
                    a1c_positive.astype(float),
                ],
                default=np.nan,
            )

    # Create fasting-glucose sensitivity labels
    fasting_labels = ["y_bio_fpg", "undx_fpg"]

    if any(col not in df.columns for col in fasting_labels):
        _require_columns(
            df,
            ["LBXGH", "LBXGLU", "w_fast"],
            "fasting-glucose labels",
        )

        hba1c = pd.to_numeric(df["LBXGH"], errors="coerce")
        glucose = pd.to_numeric(df["LBXGLU"], errors="coerce")
        fasting_weight = pd.to_numeric(
            df["w_fast"], errors="coerce"
        ).fillna(0)

        a1c_positive = hba1c >= 6.5
        glucose_positive = glucose >= 126

        lab_sample = df["y_diag"].notna() & hba1c.notna()

        valid_fasting_sample = (
            lab_sample
            & glucose.notna()
            & (fasting_weight > 0)
        )

        biological_positive = a1c_positive | glucose_positive

        if "y_bio_fpg" not in df.columns:
            df["y_bio_fpg"] = (
                (
                    (df["y_diag"] == 1)
                    | biological_positive
                )
                .astype(float)
                .where(valid_fasting_sample)
            )

        if "undx_fpg" not in df.columns:
            df["undx_fpg"] = (
                (
                    (df["y_diag"] == 0)
                    & biological_positive
                )
                .astype(float)
                .where(valid_fasting_sample)
            )

    # Create screening label if it is missing
    if "screened" not in df.columns:
        _require_columns(df, ["DIQ180"], "screening label")

        screening = pd.to_numeric(
            df["DIQ180"], errors="coerce"
        )

        df["screened"] = np.where(
            df["y_diag"].eq(0),
            screening.map({1: 1.0, 2: 0.0}),
            np.nan,
        )

    missing = [
        col for col in LABEL_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing diabetes labels: {missing}")

    print("Diabetes labels are available.")
    return df


def add_subgroups(df):
    """
    Create subgroup variables using the V7 definitions.

    Existing V7 subgroup columns are preserved.
    """

    df = df.copy()

    # Income categories
    if "income_band" not in df.columns:
        _require_columns(df, ["INDFMPIR"], "income_band")

        income = pd.to_numeric(
            df["INDFMPIR"], errors="coerce"
        )

        df["income_band"] = pd.cut(
            income,
            [-np.inf, 1.3, 3.5, np.inf],
            labels=["<1.3", "1.3-3.5", ">3.5"],
            right=False,
        )

    if "income_band2" not in df.columns:
        _require_columns(df, ["INDFMPIR"], "income_band2")

        income = pd.to_numeric(
            df["INDFMPIR"], errors="coerce"
        )

        df["income_band2"] = pd.cut(
            income,
            [-np.inf, 1.3, np.inf],
            labels=["<1.3", ">=1.3"],
            right=False,
        )

    # Insurance status
    if "insured" not in df.columns:
        _require_columns(df, ["HIQ011"], "insured")

        df["insured"] = pd.to_numeric(
            df["HIQ011"], errors="coerce"
        ).map({
            1: 1.0,
            2: 0.0,
        })

    # Usual source of healthcare
    if "routine_care" not in df.columns:
        _require_columns(df, ["HUQ030"], "routine_care")

        df["routine_care"] = pd.to_numeric(
            df["HUQ030"], errors="coerce"
        ).map({
            1: 1.0,
            2: 0.0,
            3: 1.0,
        })

    # Age groups
    if "age_band" not in df.columns:
        _require_columns(df, ["RIDAGEYR"], "age_band")

        age = pd.to_numeric(
            df["RIDAGEYR"], errors="coerce"
        )

        df["age_band"] = pd.cut(
            age,
            [20, 40, 60, np.inf],
            labels=["20-39", "40-59", "60+"],
            right=False,
        )

    if "age_band2" not in df.columns:
        _require_columns(df, ["RIDAGEYR"], "age_band2")

        age = pd.to_numeric(
            df["RIDAGEYR"], errors="coerce"
        )

        df["age_band2"] = pd.cut(
            age,
            [20, 60, np.inf],
            labels=["20-59", "60+"],
            right=False,
        )

    # Sex
    if "sex" not in df.columns:
        _require_columns(df, ["RIAGENDR"], "sex")

        df["sex"] = pd.to_numeric(
            df["RIAGENDR"], errors="coerce"
        ).map({
            1: "male",
            2: "female",
        })

    # Race and ethnicity
    if "race_eth" not in df.columns:
        _require_columns(df, ["RIDRETH3"], "race_eth")

        race = pd.to_numeric(
            df["RIDRETH3"], errors="coerce"
        )

        df["race_eth"] = race.map(RACE_LABELS)

    if "race_eth4" not in df.columns:
        _require_columns(df, ["RIDRETH3"], "race_eth4")

        race = pd.to_numeric(
            df["RIDRETH3"], errors="coerce"
        )

        df["race_eth4"] = race.map(RACE_LABELS_4)

    missing = [
        col for col in SUBGROUP_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing subgroup columns: {missing}")

    print("All subgroup variables are available.")
    return df


def validate_labels(df):
    """
    Validate binary labels and subgroup categories.
    """

    # Check binary outcome labels
    for col in LABEL_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing label column: {col}")

        values = pd.to_numeric(df[col], errors="coerce")

        invalid = df[col].notna() & (
            values.isna() | ~values.isin([0, 1])
        )

        if invalid.any():
            raise ValueError(
                f"{col} contains values other than 0, 1 or missing."
            )

        print(
            f"{col}: {df[col].notna().sum()} non-missing values"
        )

    # Check subgroup columns
    for col in SUBGROUP_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing subgroup column: {col}")

    allowed_categories = {
        "income_band": {"<1.3", "1.3-3.5", ">3.5"},
        "income_band2": {"<1.3", ">=1.3"},
        "insured": {0.0, 1.0},
        "routine_care": {0.0, 1.0},
        "age_band": {"20-39", "40-59", "60+"},
        "age_band2": {"20-59", "60+"},
        "sex": {"male", "female"},
        "race_eth": set(RACE_LABELS.values()),
        "race_eth4": set(RACE_LABELS_4.values()),
    }

    for col, allowed in allowed_categories.items():
        observed = set(df[col].dropna().unique())
        unexpected = observed - allowed

        if unexpected:
            raise ValueError(
                f"{col} contains unexpected categories: {unexpected}"
            )

        print(f"{col}: subgroup categories validated")

    print("\nLabel and subgroup validation successful!")
    return True
