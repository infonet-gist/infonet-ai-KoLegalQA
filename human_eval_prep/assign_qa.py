import pandas as pd
import numpy as np
import os

SEED = 42
np.random.seed(SEED)

SAMPLED_ITEMS_PATH = "../data_final/qa_sampled_100.csv"

MODEL_PATHS = {
    "a": "infer_outputs_numbered/beomi/base_output.csv",
    "b": "infer_outputs_numbered/beomi/trained_output.csv",
    "c": "infer_outputs_numbered/princeton/base_output.csv",
    "d": "infer_outputs_numbered/princeton/trained_output.csv",
    "e": "infer_outputs_numbered/seokdong/base_output.csv",
    "f": "infer_outputs_numbered/seokdong/trained_output.csv",
}

OUTPUT_DIR = "evaluator_assignment"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Group / batch / model assigning rules
GROUP_BATCH_MODELS = {
    "A": {  # Group A
        1: ["a", "b"],
        2: ["c", "d"],
        3: ["e", "f"],
    },
    "B": {  # Group B
        1: ["d", "e"],
        2: ["b", "f"],
        3: ["a", "c"],
    },
    "C": {  # Group C
        1: ["c", "f"],
        2: ["a", "e"],
        3: ["b", "d"],
    },
}

# Evaluator setup
EVALUATORS = {
    "A": ["A1", "A2", "A3", "A4"],
    "B": ["B1", "B2", "B3", "B4"],
    "C": ["C1", "C2", "C3", "C4"],
}

# Load the list of 100 samples
items_df = pd.read_csv(SAMPLED_ITEMS_PATH)
items_df = items_df.rename(columns={
    "id": "item_id",
    "answer": "reference_answer",
})


# merge inference outputs
model_frames = []

for model_id, path in MODEL_PATHS.items():
    df = pd.read_csv(path)

    df = df.rename(columns={
        "id": "item_id",
        "prediction": "model_answer"
    })

    df = df[["item_id", "model_answer"]].copy()
    df["model_id"] = model_id

    model_frames.append(df)

model_df = pd.concat(model_frames, ignore_index=True)

print("ITEMS DF:", items_df.columns)
print("MODEL DF:", model_df.columns)


# Create 100 x 6 master table
master_df = pd.merge(items_df, model_df, on="item_id", how="inner")

expected_rows = 100 * 6
if len(master_df) != expected_rows:
    raise ValueError(
        f"Master table size mismatch: {len(master_df)} rows, expected {expected_rows}"
    )


# Assign items to evaluators
def split_items_evenly(item_list, n_evaluators):
    np.random.shuffle(item_list)
    return np.array_split(item_list, n_evaluators)

for group_id, evaluator_list in EVALUATORS.items():

    n_eval = len(evaluator_list)

    for batch_id in [1, 2, 3]:

        models = GROUP_BATCH_MODELS[group_id][batch_id]
        item_ids = items_df["item_id"].tolist()

        split_item_lists = split_items_evenly(item_ids.copy(), n_eval)

        for evaluator_id, assigned_items in zip(evaluator_list, split_item_lists):

            batch_rows = []

            for model_id in models:
                subset = master_df[
                    (master_df["item_id"].isin(assigned_items)) &
                    (master_df["model_id"] == model_id)
                ]
                batch_rows.append(subset)

            evaluator_batch_df = pd.concat(batch_rows).reset_index(drop=True)

            evaluator_batch_df["evaluator_id"] = evaluator_id
            evaluator_batch_df["batch_id"] = batch_id

            # Shuffle order
            evaluator_batch_df = evaluator_batch_df.sample(
                frac=1,
                random_state=SEED
            ).reset_index(drop=True)

            evaluator_batch_df = evaluator_batch_df[
                [
                    "item_id",
                    "model_id",
                    "evaluator_id",
                    "batch_id",
                    "question",
                    "reference_answer",
                    "model_answer",
                ]
            ]

            output_path = os.path.join(
                OUTPUT_DIR,
                f"{evaluator_id}_batch{batch_id}.csv"
            )

            evaluator_batch_df.to_csv(output_path, index=False)

print("Evaluator assignment files generated successfully.")
