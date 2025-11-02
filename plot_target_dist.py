
from global_imports import *
import matplotlib.pyplot as plt
import numpy as np

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

    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()