
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from baseline import train_required_baselines_mlflow

out = train_required_baselines_mlflow()

def plot_confusion_matrix_best_test():

    y_true = out["y_test_class_true"]   # 0=Low, 1=High
    y_pred = out["y_test_class_pred"]
    best   = out["best_clf_name"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots()
    im = ax.imshow(cm, vmin=0, vmax=max(cm.max(), 1))
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Low","High"]); ax.set_yticklabels(["Low","High"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Plot 3 — Confusion Matrix (Best: {best})")

    # annotate cell counts
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11)

    fig.colorbar(im, ax=ax).set_label("Count")
    plt.tight_layout()
    plt.show()


def plot_residuals_vs_pred_best_test():

    y_true = out["y_test_reg_true"]
    y_pred = out["y_test_reg_pred"]
    name   = out["best_reg_name"]

    resid = y_true - y_pred

    fig, ax = plt.subplots()
    ax.scatter(y_pred, resid, s=10, alpha=0.85)
    ax.axhline(0, ls="--", lw=1)
    ax.set_xlabel("Predicted SalePrice")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(f"Plot 4 — Residuals vs Predicted (Best: {name})")
    plt.tight_layout()
    plt.show()


