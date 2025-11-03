# ===== Imports =====
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor






import mlflow

from train_validation_test_split import make_train_val_test_split_random

try:
    _ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    _OHE_KW = {"sparse_output": False}
except TypeError:
    _OHE_KW = {"sparse": False}

DROP_COLS = ["SalePrice", "SalePriceClass", "SalePriceClassNum"]

def build_features_from_train(train_df: pd.DataFrame):

    feat = train_df.drop(columns=DROP_COLS, errors="ignore").copy()
    if "MSSubClass" in feat.columns:
        feat["MSSubClass"] = feat["MSSubClass"].astype(str)
    cat_cols = feat.select_dtypes(include=["object","string"]).columns.tolist()
    num_cols = feat.select_dtypes(include=[np.number]).columns.tolist()
    X_cols   = feat.columns.tolist()
    return X_cols, cat_cols, num_cols

def make_preprocessors(train_df: pd.DataFrame):
    X_cols, cat_cols, num_cols = build_features_from_train(train_df)
    pp_clf = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler(with_mean=True))]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh",  OneHotEncoder(handle_unknown="ignore", **_OHE_KW))]), cat_cols),
    ], remainder="drop")
    pp_reg = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh",  OneHotEncoder(handle_unknown="ignore", **_OHE_KW))]), cat_cols),
    ], remainder="drop")
    return X_cols, pp_clf, pp_reg

def train_required_baselines_mlflow(seed=45):

    mlflow.set_experiment("Midpoint")

    # The split
    S = make_train_val_test_split_random(seed)
    train, val, test = S["train"].copy(), S["val"].copy(), S["test"].copy()
    thr, seed = S["threshold"], S["seed"]

    # keep MSSubClass string everywhere to match TRAIN
    for part in (train, val, test):
        if "MSSubClass" in part.columns:
            part["MSSubClass"] = part["MSSubClass"].astype(str)

    # Preprocessors from TRAIN only
    X_cols, pp_clf, pp_reg = make_preprocessors(train)

    Xtr_c, ytr_c = train[X_cols], train["SalePriceClassNum"].values
    Xva_c, yva_c = val[X_cols],   val["SalePriceClassNum"].values
    Xte_c, yte_c = test[X_cols],  test["SalePriceClassNum"].values

    Xtr_r, ytr_r = train[X_cols], train["SalePrice"].values
    Xva_r, yva_r = val[X_cols],   val["SalePrice"].values
    Xte_r, yte_r = test[X_cols],  test["SalePrice"].values

    clf_models = {
        "Naive Bayes":         Pipeline([("pp", pp_clf), ("m", GaussianNB())]),
        "Logistic Regression": Pipeline([("pp", pp_clf), ("m", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed))]),
    }
    clf_rows = []; best_clf_name=None; best_val_f1=-1; best_test_pred_clf=None

    for name, pipe in clf_models.items():
        with mlflow.start_run(run_name=f"clf - {name} (seed={seed})"):
            pipe.fit(Xtr_c, ytr_c)
            yv = pipe.predict(Xva_c); yt = pipe.predict(Xte_c)

            val_acc = accuracy_score(yva_c, yv);  val_f1  = f1_score(yva_c, yv)
            test_acc= accuracy_score(yte_c, yt);  test_f1 = f1_score(yte_c, yt)

            mlflow.log_param("seed", seed)
            mlflow.log_param("threshold_median_saleprice", float(thr))
            mlflow.log_metrics({"val accuracy": val_acc, "val f1": val_f1,
                                "test accuracy": test_acc, "test f1": test_f1})

            clf_rows.append({"Model": name,
                             "Val Accuracy": val_acc, "Val F1": val_f1,
                             "Test Accuracy": test_acc, "Test F1": test_f1})

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_clf_name = name
                best_test_pred_clf = yt

    table1 = pd.DataFrame(clf_rows).round(4).sort_values(by="Test F1", ascending=False)

    reg_models = {
        "Linear Regression":      Pipeline([("pp", pp_reg), ("m", LinearRegression())]),
        "Decision Tree Regressor": Pipeline([("pp", pp_reg), ("m", DecisionTreeRegressor(random_state=seed))]),
    }
    reg_rows = []; best_reg_name=None; best_val_rmse=np.inf; best_test_pred_reg=None

    for name, pipe in reg_models.items():
        with mlflow.start_run(run_name=f"reg - {name} (seed={seed})"):
            pipe.fit(Xtr_r, ytr_r)
            yv = pipe.predict(Xva_r); yt = pipe.predict(Xte_r)

            val_mae  = mean_absolute_error(yva_r, yv)
            val_rmse = np.sqrt(mean_squared_error(yva_r, yv))
            test_mae = mean_absolute_error(yte_r, yt)
            test_rmse= np.sqrt(mean_squared_error(yte_r, yt))

            mlflow.log_param("seed", seed)
            mlflow.log_metrics({"val_mae": val_mae, "val_rmse": val_rmse,
                                "test_mae": test_mae, "test_rmse": test_rmse})

            reg_rows.append({"Model": name,
                             "Val MAE": val_mae, "Val RMSE": val_rmse,
                             "Test MAE": test_mae, "Test RMSE": test_rmse})

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_reg_name = name
                best_test_pred_reg = yt

    table2 = pd.DataFrame(reg_rows).round(2).sort_values(by="Test RMSE")

    return {
        "split_note": S["split_note"],
        "threshold": thr,
        "Table1": table1,
        "Table2": table2,
        "best_clf_name": best_clf_name,
        "y_test_class_true": yte_c,
        "y_test_class_pred": best_test_pred_clf,

        "best_reg_name": best_reg_name,
        "y_test_reg_true": yte_r,
        "y_test_reg_pred": best_test_pred_reg,
    }



def execute_baseline():
    out = train_required_baselines_mlflow()

    print("Split:", out["split_note"])
    print("\nTable 1 — Classification (Val/Test):")
    print(out["Table1"].to_string(index=False))

    print("\nTable 2 — Regression (Val/Test):")
    print(out["Table2"].to_string(index=False))