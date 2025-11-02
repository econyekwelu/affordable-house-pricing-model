import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_path = "data/train.csv"
df = pd.read_csv(train_path)
def get_median_sales_price() -> float :
    median_saleprice = float(df["SalePrice"].median())
    return median_saleprice

def plot_target_distribution_classification():
    thr = get_median_sales_price()

    df["SalePriceClass"] = np.where(df["SalePrice"] > thr, "High", "Low")

    df["SalePriceClassNum"] = (df["SalePrice"] > thr).astype("int8")  # 0=Low, 1=High


    counts = (
        df["SalePriceClass"]
        .value_counts()
        .reindex(["Low", "High"])
        .fillna(0)
    )

    ax = counts.plot(kind="bar", rot=0)
    ax.set_title("Target Distribution: SalePrice (Low vs High)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")

    # Annotate counts on bars
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_corr_heatmap_topk(k=12, exclude=("Id", "MoSold", "YrSold"), annotate=True):

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "SalePrice" not in num_cols:
        raise KeyError("SalePrice numeric target not found in df")

    num_cols = [c for c in num_cols if c not in exclude]

    corr_all = df[num_cols].corr(numeric_only=True)
    target_corr = corr_all["SalePrice"].drop(labels=["SalePrice"], errors="ignore").abs().sort_values(ascending=False)
    top_feats = target_corr.head(k).index.tolist()

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

plot_corr_heatmap_topk()