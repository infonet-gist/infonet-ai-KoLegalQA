import pandas as pd
import numpy as np

SEED = 42
N_SAMPLE = 100

INPUT_PATH = "../data_final/held_out_split.csv"
OUTPUT_PATH = "../data_final/qa_sampled_100.csv"

def main():
    np.random.seed(SEED)

    df = pd.read_csv(INPUT_PATH)
    
    # Create sequential ID (for traceability)
    df = df.reset_index(drop=True)
    df["id"] = df.index

    if len(df) < N_SAMPLE:
        raise ValueError("Input item count is smaller than 100.")

    sampled_df = df.sample(n=N_SAMPLE, random_state=SEED).reset_index(drop=True)

    sampled_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(sampled_df)} sampled items to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
