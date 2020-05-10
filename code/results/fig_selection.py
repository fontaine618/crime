import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

plt.style.use("seaborn")

importances = pd.read_csv("./data/results/rf_importances.csv", index_col=0)

estimates = pd.read_csv("./data/results/lr_regularized.csv", index_col=0)

estimates = estimates / estimates.sum()

imp = pd.concat([importances, estimates], 1)

imp.columns = [
    "Any (RF)", "Severe (RF)", "Mean (RF)",
    "Any (RLR)", "Severe (RLR)", "Mean (RLR)",
]

cols = [
    "Any (RF)", "Severe (RF)"
    "Any (RLR)", "Severe (RLR)"
]

colors = [
    "#0000AA",
    "#000066",
    "#AA0000",
    "#660000",
]

labs = imp.index
x = np.array(range(len(labs)))
width = 0.2

plt.figure(figsize=(10, 5))

plt.bar(x - width * 3 / 2, imp["Any (RF)"], width,
        label="Any (RF)", color=colors[0])
plt.bar(x - width * 1 / 2, imp["Severe (RF)"], width,
        label="Severe (RF)", color=colors[1])
plt.bar(x + width * 1 / 2, imp["Any (RLR)"], width,
        label="Any (RLR)", color=colors[2])
plt.bar(x + width * 3 / 2, imp["Severe (RLR)"], width,
        label="Severe (RLR)", color=colors[3])
plt.xticks(x, labs, rotation=90)
plt.title("Variable Importance", loc="left")
plt.legend(title="Response (Model)")
plt.tight_layout()
plt.savefig("./report/figs/importances.pdf")