import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
from sklearn.utils import resample

# ===============================
# Config
# ===============================

INPUT_FILE = "analysis_ready.csv"
OUTPUT_FILE = "correlation_results.csv"
BOOTSTRAP_ITER = 5000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# ===============================
# Load dataset
# ===============================

df = pd.read_csv(INPUT_FILE)

# define metric pairs for analysis
pairs = [
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

for h_col, g_col in pairs:

    sub = df[[h_col, g_col]].dropna().copy()
    sub.columns = ["human", "g"]

    # Spearman
    rho, pval = stats.spearmanr(sub["human"], sub["g"])

    # Bootstrap 95% CI
    boot_rhos = []
    for _ in range(BOOTSTRAP_ITER):
        sample = resample(sub)
        r, _ = stats.spearmanr(sample["human"], sample["g"])
        boot_rhos.append(r)

    ci_lower = np.percentile(boot_rhos, 2.5)
    ci_upper = np.percentile(boot_rhos, 97.5)

    # ICC(2,k)
    long_df = pd.DataFrame({
        "target": np.repeat(np.arange(len(sub)), 2),
        "rater": np.tile(["human", "g"], len(sub)),
        "score": np.concatenate([sub["human"].values, sub["g"].values])
    })

    icc_table = pg.intraclass_corr(
        data=long_df,
        targets="target",
        raters="rater",
        ratings="score"
    )

    icc2k = icc_table[icc_table["Type"] == "ICC2k"].iloc[0]

    icc_value = icc2k["ICC"]
    icc_ci_low = icc2k["CI95%"][0]
    icc_ci_high = icc2k["CI95%"][1]
    icc_pval = icc2k["pval"]

    # results
    results.append({
        "Metric": h_col.replace("human_", ""),
        "Spearman_rho": rho,
        "Spearman_p": pval,
        "Spearman_CI95_low": ci_lower,
        "Spearman_CI95_high": ci_upper,
        "ICC2k": icc_value,
        "ICC2k_p": icc_pval,
        "ICC2k_CI95_low": icc_ci_low,
        "ICC2k_CI95_high": icc_ci_high
    })

    print("="*60)
    print(f"Metric: {h_col.replace('human_', '')}")
    print(f"Spearman rho = {rho:.4f}")
    print(f"p-value = {pval:.6f}")
    print(f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
    print()
    print(f"ICC(2,k) = {icc_value:.4f}")
    print(f"p-value = {icc_pval:.6f}")
    print(f"95% CI = [{icc_ci_low:.4f}, {icc_ci_high:.4f}]")
    print("="*60)
    print()

# Save

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print("Saved results to:", OUTPUT_FILE)
