
import pandas as pd
from src.crosswalk import apply_crosswalk, validate_crosswalk

# Load the combined dataset
df = pd.read_csv("nhanes_out/nhanes_all.csv")

# Check all five cycles
cycles = df["cycle"].unique()

print("CYCLE-WISE CROSSWALK VALIDATION")
print("--------------------------------")

for cycle in sorted(cycles):
    cycle_df = df[df["cycle"] == cycle].copy()

    cycle_df = apply_crosswalk(cycle_df)

    validate_crosswalk(cycle_df)

    print(f"{cycle}: {len(cycle_df)} participants - PASSED")

print("\nAll cycle checks completed.")
