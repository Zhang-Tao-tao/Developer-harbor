import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def read_data(file_name):
    data = pd.read_csv(file_name)
    results = {}
    for i in range(10):
        df = data[data["label"] == i]
        results[i] = df[f"logits_{i}"].tolist()
    return results


results = read_data("eval_admm_result.csv")
results = {"Logit Group": [], "Value": [], "source": []}
for i, item in read_data("eval_admm_result.csv").items():
    results["Logit Group"].extend([f"label_{i}"] * len(item))
    results["Value"].extend(item)
    results["source"].extend(["optimizing_attention admm"] * len(item))

for i, item in read_data("eval_kaiwu_sa_result.csv").items():
    results["Logit Group"].extend([f"label_{i}"] * len(item))
    results["Value"].extend(item)
    results["source"].extend(["optimizing_attention kaiwu SA"] * len(item))

sns.set_theme(style="whitegrid", palette="Set2")

plt.figure(figsize=(16, 8))
box = sns.boxplot(
    x="Logit Group",
    y="Value",
    hue="source",
    data=results,
    showfliers=False,
    linewidth=1.2,
    width=0.8,
)

plt.title("Boxplot of sa vs admm (label0 - label9)", fontsize=16)
plt.xlabel("Label", fontsize=14)
plt.ylabel("Value", fontsize=14)

plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

plt.legend(title="Source", fontsize=12, title_fontsize=13)

plt.tight_layout()
plt.savefig("boxplot_sa_vs_admm.png", dpi=600, bbox_inches="tight")
plt.show()
