import pandas as pd
import glob

# Load all evaluator CSV files
files = glob.glob("evaluator_assignment/*.csv")

all_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Count occurrences of each (item_id, model_id) pair
count_df = (
    all_df
    .groupby(["item_id", "model_id"])
    .size()
    .reset_index(name="count")
)

# Basic statistics
print("Total unique item_id:", all_df["item_id"].nunique())
print("Total unique model_id:", all_df["model_id"].nunique())
print("Total (item x model) combinations:", len(count_df))

print("\nDistribution of occurrence counts:")
print(count_df["count"].value_counts().sort_index())

EXPECTED = 3
missing = count_df[count_df["count"] < EXPECTED]
over = count_df[count_df["count"] > EXPECTED]

if len(missing) == 0 and len(over) == 0:
    print("\n* All (item_id, model_id) pairs appear exactly", EXPECTED, "times.")
else:
    print("\n* Issues detected")
    if len(missing) > 0:
        print("* Missing or under-assigned: ", len(missing))
        print(missing.head())
    if len(over) > 0:
        print("Over-assigned: ", len(over))
        print(over.head())
