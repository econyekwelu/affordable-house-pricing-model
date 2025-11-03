from global_imports import *
import matplotlib.pyplot as plt
import numpy as np

def plot_corr_heatmap_top_k_features(k=12, exclude = (), annotate = True ):

    number_of_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if "SalePrice" not in number_of_columns:
        raise KeyError("SalePrice numeric target not found in df")

    number_of_columns = [c for c in number_of_columns if c not in exclude]

    all_correlations = df[number_of_columns].corr(numeric_only=True)
    target_correlation = all_correlations["SalePrice"].drop(labels=["SalePrice"], errors="ignore").abs().sort_values(ascending=False)
    top_feats = target_correlation.head(k).index.tolist()

    cols = ["SalePrice"] + top_feats
    C = df[cols].corr(numeric_only=True).values

    fig, ax = plt.subplots(figsize=(max(8, 0.6*len(cols)), max(6, 0.6*len(cols))))
    im = ax.imshow(C, vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson r")

    if annotate and len(cols) <= 20:
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=7)

    ax.set_title(f"Correlation heatmap (top {len(top_feats)} features vs SalePrice)")
    fig.tight_layout()
    plt.show()

