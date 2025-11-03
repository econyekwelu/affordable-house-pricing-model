
import numpy as np
import mlflow
from matplotlib import pyplot as plt

from global_imports import *

import numpy as np



def make_train_val_test_split_random(seed = 45):


    d = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


    n = len(d); n_tr = int(0.6 * n); n_va = int(0.2 * n)
    train = d.iloc[:n_tr].copy()
    val   = d.iloc[n_tr:n_tr+n_va].copy()
    test  = d.iloc[n_tr+n_va:].copy()


    thr = float(train["SalePrice"].median())
    for part in (train, val, test):
        part["SalePriceClassNum"] = (part["SalePrice"] > thr).astype("int8")
        part["SalePriceClass"]    = np.where(part["SalePrice"] > thr, "High", "Low")


    mlflow.set_experiment("Midpoint")
    with mlflow.start_run(run_name=f"data-split-random-seed-{seed}"):
        mlflow.log_param("split_strategy", f"random split with fixed seed={seed} (reproducible)")
        mlflow.log_param("seed", seed)
        mlflow.log_param("threshold_median_saleprice", thr)
        mlflow.log_param("rows_train", len(train))
        mlflow.log_param("rows_val", len(val))
        mlflow.log_param("rows_test", len(test))

    return {
        "train": train, "val": val, "test": test,
        "split_note": f"random split with fixed seed={seed} (reproducible)",
        "seed": seed, "threshold": thr,
    }
def plot_split(res, bins=40, log=False):

    tr = res["train"]["SalePrice"]
    va = res["val"]["SalePrice"]
    te = res["test"]["SalePrice"]

    fig, ax = plt.subplots()
    ax.hist(tr, bins=bins, alpha=0.5, label="Train")
    ax.hist(va, bins=bins, alpha=0.5, label="Validation")
    ax.hist(te, bins=bins, alpha=0.5, label="Test")

    if log:
        ax.set_xscale("log"); ax.set_xlabel("SalePrice (log scale)")
    else:
        ax.set_xlabel("SalePrice")

    ax.set_ylabel("Count")
    ax.set_title("SalePrice distribution by split")
    ax.legend()
    plt.tight_layout(); plt.show()