
import pandas as pd

from src.labels import (
    add_labels,
    add_subgroups,
    validate_labels
)

# Load the cleaned dataset
df = pd.read_csv("nhanes_out/nhanes_all.csv")

print("Original dataset:", df.shape)

# Check diabetes labels
df = add_labels(df)

# Check subgroup variables
df = add_subgroups(df)

# Validate label values
validate_labels(df)

print("\nDiabetes label counts:")

for col in [
    "y_diag",
    "y_bio",
    "y_bio_any",
    "y_bio_fpg",
    "undx",
    "undx_fpg",
    "screened"
]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))

print("\nAll label and subgroup tests completed!")
