import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv("results_summary_extended.csv")

# Compute percentage difference: (normal - abrupt) / normal
df["pct_mse"] = (df["mse"] - df["mse_abrupt"]) / df["mse"] * 100
df["pct_mae"] = (df["mae"] - df["mae_abrupt"]) / df["mae"] * 100

# Prediction windows
windows = sorted(df["pred_window"].unique())

color_mse = "#534CB0"   
color_mae = "#DDB852"  

# Loop for two features (M, S)
for feat in ["M", "S"]:

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    if feat == "M":
        fig.suptitle(f"Percentage Change (Normal - Abrupt) for Multivariable Prediction", fontsize=14)
    else:
        fig.suptitle(f"Percentage Change (Normal - Abrupt) for Single-Variable Prediction", fontsize=14)

    for idx, ext in enumerate([0, 1]):
        ax = axes[idx]

        sub = df[(df.feature_type == feat) & (df.extended == ext)]
        x = np.arange(len(windows))
        width = 0.35

        # Bars
        ax.bar(x - width/2, sub["pct_mse"], width=width, color=color_mse, label="MSE %")
        ax.bar(x + width/2, sub["pct_mae"], width=width, color=color_mae, label="MAE %")

        # Titles and labels
        if ext:
            ax.set_title(f"MLP Projector Extended with Signal derivatives")
        else:
            ax.set_title(f"Original MLP Projector")
        ax.set_xticks(x)
        ax.set_xticklabels(windows)
        ax.set_xlabel("Prediction Window")
        ax.grid(axis="y")

        if idx == 0:
            ax.set_ylabel("Percentage Change (%)")

        if idx == 1:
            ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"percentage_change_extended_feat{feat}.pdf")
    plt.show()
