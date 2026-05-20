import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
from sklearn.utils import resample

# ===============================
# Config
# ===============================

INPUT_FILE = "analysis_ready.csv"
OUTPUT_FILE = "per_model_results.csv"
BOOTSTRAP_ITER = 3000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# ===============================
# Load dataset
# ===============================

df = pd.read_csv(INPUT_FILE)

metrics = [
    ("human_conclusion", "g_conclusion"),
    ("human_reasoning", "g_reasoning"),
    ("human_relevance", "g_relevance"),
    ("human_clarity", "g_clarity"),
    ("human_legal_core", "g_legal_core"),
    ("human_overall", "g_overall")
]

results = []

# ===============================
# Analysis Loop
# ===============================

for model_name in df["model"].unique():

    df_model = df[df["model"] == model_name]

    print("\n" + "#" * 70)
    print(f"Model: {model_name}")
    print("#" * 70)

    for h_col, g_col in metrics:

        sub = df_model[[h_col, g_col]].dropna().copy()
        sub.columns = ["human", "g"]

        if len(sub) < 10:
            continue

        # Spearman
        rho, pval = stats.spearmanr(sub["human"], sub["g"])

        # Bootstrap CI
        boot_vals = []
        for _ in range(BOOTSTRAP_ITER):
            sample = resample(sub)
            r, _ = stats.spearmanr(sample["human"], sample["g"])
            boot_vals.append(r)

        ci_low = np.percentile(boot_vals, 2.5)
        ci_high = np.percentile(boot_vals, 97.5)

        # ICC(2,k)
        long_df = pd.DataFrame({
            "target": np.repeat(np.arange(len(sub)), 2),
            "rater": np.tile(["human", "g"], len(sub)),
            "score": np.concatenate([sub["human"].values,
                                     sub["g"].values])
        })

        icc_table = pg.intraclass_corr(
            data=long_df,
            targets="target",
            raters="rater",
            ratings="score"
        )

        icc2k = icc_table[icc_table["Type"] == "ICC2k"].iloc[0]

        icc_val = icc2k["ICC"]
        icc_p = icc2k["pval"]
        icc_ci_low = icc2k["CI95%"][0]
        icc_ci_high = icc2k["CI95%"][1]

        metric_name = h_col.replace("human_", "")

        print("-" * 60)
        print(f"Metric: {metric_name}")
        print(f"Spearman rho = {rho:.4f}")
        print(f"p-value = {pval:.6f}")
        print(f"95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
        print()
        print(f"ICC(2,k) = {icc_val:.4f}")
        print(f"p-value = {icc_p:.6f}")
        print(f"95% CI = [{icc_ci_low:.4f}, {icc_ci_high:.4f}]")

        # results
        results.append({
            "Model": model_name,
            "Metric": metric_name,
            "Spearman_rho": rho,
            "Spearman_p": pval,
            "Spearman_CI95_low": ci_low,
            "Spearman_CI95_high": ci_high,
            "ICC2k": icc_val,
            "ICC2k_p": icc_p,
            "ICC2k_CI95_low": icc_ci_low,
            "ICC2k_CI95_high": icc_ci_high
        })

# Save

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved results to:", OUTPUT_FILE)
