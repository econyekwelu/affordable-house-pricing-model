
import warnings

from global_imports import global_seed, train_path, df

warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error
)

import tensorflow as tf
from tensorflow import keras


def _make_ohe():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def run_normal_vs_neural_network(outputs_dir: str = "outputs"):
    # -----------------------------
    # 0) Reproducibility + folders
    # -----------------------------
    np.random.seed(global_seed)
    tf.random.set_seed(global_seed)
    os.makedirs(outputs_dir, exist_ok=True)

    # -----------------------------
    # 1) Load data
    # -----------------------------
    df_original = df
    df_cleaned = df_original.copy()

    # Drop obvious non-features if present
    columns_to_drop = [c for c in ["Id"] if c in df_cleaned.columns]
    df_features_only = df_cleaned.drop(columns=columns_to_drop, errors="ignore")

    # Targets
    if "SalePrice" not in df_features_only.columns:
        raise ValueError("Could not find 'SalePrice' column in the CSV.")

    target_regression = "SalePrice"
    saleprice_midpoint = float(df_features_only[target_regression].median())
    y_classification_full = (df_features_only[target_regression] >= saleprice_midpoint).astype(int)
    y_regression_full = df_features_only[target_regression]

    # Features
    feature_columns = [c for c in df_features_only.columns if c != target_regression]
    X_full = df_features_only[feature_columns]

    numeric_cols = X_full.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X_full.columns if c not in numeric_cols]

    # -----------------------------
    # 2) Train/Val/Test split
    # -----------------------------
    X_temp, X_test, y_class_temp, y_class_test, y_reg_temp, y_reg_test = train_test_split(
        X_full, y_classification_full, y_regression_full,
        test_size=0.20, random_state=global_seed, stratify=y_classification_full
    )

    X_train, X_val, y_class_train, y_class_val, y_reg_train, y_reg_val = train_test_split(
        X_temp, y_class_temp, y_reg_temp,
        test_size=0.25, random_state=global_seed, stratify=y_class_temp
    )  # -> 60/20/20 split

    # -----------------------------
    # 3) Preprocessing
    # -----------------------------
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    ohe = _make_ohe()

    full_preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", ohe, categorical_cols)
    ], remainder="drop")

    full_preprocessor.fit(X_train)
    X_train_proc = full_preprocessor.transform(X_train)
    X_val_proc   = full_preprocessor.transform(X_val)
    X_test_proc  = full_preprocessor.transform(X_test)

    # to numpy float32 for TF
    X_train_np = np.asarray(X_train_proc, dtype=np.float32)
    X_val_np   = np.asarray(X_val_proc,   dtype=np.float32)
    X_test_np  = np.asarray(X_test_proc,  dtype=np.float32)

    y_class_train_np = np.asarray(y_class_train, dtype=np.float32)
    y_class_val_np   = np.asarray(y_class_val,   dtype=np.float32)
    y_class_test_np  = np.asarray(y_class_test,  dtype=np.float32)

    y_reg_train_np = np.asarray(y_reg_train, dtype=np.float32)
    y_reg_val_np   = np.asarray(y_reg_val,   dtype=np.float32)
    y_reg_test_np  = np.asarray(y_reg_test,  dtype=np.float32)

    try:
        feature_names_out = full_preprocessor.get_feature_names_out()
    except Exception:
        feature_names_out = [f"f{i}" for i in range(X_train_np.shape[1])]

    input_dim = X_train_np.shape[1]

    # -----------------------------
    # 4) Classical models
    # -----------------------------
    # Logistic Regression (classification)
    logreg_model = LogisticRegression(max_iter=1000)
    logreg_model.fit(X_train_np, y_class_train_np)

    y_class_val_proba_lr = logreg_model.predict_proba(X_val_np)[:, 1]
    y_class_val_pred_lr  = (y_class_val_proba_lr >= 0.5).astype(int)
    y_class_test_proba_lr = logreg_model.predict_proba(X_test_np)[:, 1]
    y_class_test_pred_lr  = (y_class_test_proba_lr >= 0.5).astype(int)

    val_acc_lr  = accuracy_score(y_class_val_np,  y_class_val_pred_lr)
    val_auc_lr  = roc_auc_score(y_class_val_np,   y_class_val_proba_lr)
    test_acc_lr = accuracy_score(y_class_test_np, y_class_test_pred_lr)
    test_auc_lr = roc_auc_score(y_class_test_np,  y_class_test_proba_lr)

    # Linear Regression (regression)
    linreg_model = LinearRegression()
    linreg_model.fit(X_train_np, y_reg_train_np)

    y_reg_val_pred_lr  = linreg_model.predict(X_val_np)
    y_reg_test_pred_lr = linreg_model.predict(X_test_np)

    val_mae_lin  = mean_absolute_error(y_reg_val_np,  y_reg_val_pred_lr)
    val_rmse_lin = float(np.sqrt(mean_squared_error(y_reg_val_np,  y_reg_val_pred_lr)))
    test_mae_lin = mean_absolute_error(y_reg_test_np, y_reg_test_pred_lr)
    test_rmse_lin = float(np.sqrt(mean_squared_error(y_reg_test_np, y_reg_test_pred_lr)))

    # -----------------------------
    # 5) Neural Networks (two small MLPs)
    # -----------------------------
    # Classification MLP
    clf_nn = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    clf_nn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"), keras.metrics.AUC(name="auc")]
    )
    early_stop_clf = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history_clf = clf_nn.fit(
        X_train_np, y_class_train_np,
        validation_data=(X_val_np, y_class_val_np),
        epochs=200, batch_size=64, verbose=0, callbacks=[early_stop_clf]
    )

    y_class_val_proba_nn = clf_nn.predict(X_val_np,  verbose=0).ravel()
    y_class_val_pred_nn  = (y_class_val_proba_nn >= 0.5).astype(int)
    y_class_test_proba_nn = clf_nn.predict(X_test_np, verbose=0).ravel()
    y_class_test_pred_nn  = (y_class_test_proba_nn >= 0.5).astype(int)

    val_acc_nn_clf  = accuracy_score(y_class_val_np,  y_class_val_pred_nn)
    val_auc_nn_clf  = roc_auc_score(y_class_val_np,   y_class_val_proba_nn)
    test_acc_nn_clf = accuracy_score(y_class_test_np, y_class_test_pred_nn)
    test_auc_nn_clf = roc_auc_score(y_class_test_np,  y_class_test_proba_nn)

    # Regression MLP
    reg_nn = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1)
    ])
    reg_nn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
    )
    early_stop_reg = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history_reg = reg_nn.fit(
        X_train_np, y_reg_train_np,
        validation_data=(X_val_np, y_reg_val_np),
        epochs=300, batch_size=64, verbose=0, callbacks=[early_stop_reg]
    )

    y_reg_val_pred_nn  = reg_nn.predict(X_val_np,  verbose=0).ravel()
    y_reg_test_pred_nn = reg_nn.predict(X_test_np, verbose=0).ravel()

    val_mae_nn  = mean_absolute_error(y_reg_val_np,  y_reg_val_pred_nn)
    val_rmse_nn = float(np.sqrt(mean_squared_error(y_reg_val_np,  y_reg_val_pred_nn)))
    test_mae_nn = mean_absolute_error(y_reg_test_np, y_reg_test_pred_nn)
    test_rmse_nn = float(np.sqrt(mean_squared_error(y_reg_test_np, y_reg_test_pred_nn)))

    # -----------------------------
    # 6) Pick "best" by validation
    # -----------------------------
    if (val_acc_nn_clf > val_acc_lr) or (np.isclose(val_acc_nn_clf, val_acc_lr) and val_auc_nn_clf > val_auc_lr):
        best_clf_name = "Neural Network (MLP)"
        best_clf_pred_test = y_class_test_pred_nn
        best_clf_proba_test = y_class_test_proba_nn
    else:
        best_clf_name = "Logistic Regression"
        best_clf_pred_test = y_class_test_pred_lr
        best_clf_proba_test = y_class_test_proba_lr

    if val_mae_nn < val_mae_lin:
        best_reg_name = "Neural Network (MLP)"
        best_reg_pred_test = y_reg_test_pred_nn
    else:
        best_reg_name = "Linear Regression"
        best_reg_pred_test = y_reg_test_pred_lr

    # -----------------------------
    # 7) Tables to CSV (and print)
    # -----------------------------
    classification_table = pd.DataFrame([
        {"Model": "Logistic Regression", "Val_Accuracy": val_acc_lr, "Val_ROC_AUC": val_auc_lr,
         "Test_Accuracy": test_acc_lr, "Test_ROC_AUC": test_auc_lr},
        {"Model": "Neural Network (MLP)", "Val_Accuracy": val_acc_nn_clf, "Val_ROC_AUC": val_auc_nn_clf,
         "Test_Accuracy": test_acc_nn_clf, "Test_ROC_AUC": test_auc_nn_clf},
    ])
    regression_table = pd.DataFrame([
        {"Model": "Linear Regression", "Val_MAE": val_mae_lin, "Val_RMSE": val_rmse_lin,
         "Test_MAE": test_mae_lin, "Test_RMSE": test_rmse_lin},
        {"Model": "Neural Network (MLP)", "Val_MAE": val_mae_nn, "Val_RMSE": val_rmse_nn,
         "Test_MAE": test_mae_nn, "Test_RMSE": test_rmse_nn},
    ])

    classification_csv_path = os.path.join(outputs_dir, "table1_classification_comparison.csv")
    regression_csv_path     = os.path.join(outputs_dir, "table2_regression_comparison.csv")
    classification_table.to_csv(classification_csv_path, index=False)
    regression_table.to_csv(regression_csv_path, index=False)

    print("\n=== SALEPRICE midpoint used for classification ===")
    print(f"midpoint = {saleprice_midpoint:,.2f}")

    print("\n=== Classification comparison (Val/Test) ===")
    print(classification_table.to_string(index=False))

    print("\n=== Regression comparison (Val/Test) ===")
    print(regression_table.to_string(index=False))

    # -----------------------------
    # 8) Plots to PNG
    # -----------------------------
    # Plot 1: Learning curve (classification Neural Network)
    fig1 = plt.figure()
    plt.plot(history_clf.history["accuracy"], label="train_acc")
    plt.plot(history_clf.history["val_accuracy"], label="val_acc")
    plt.title("Learning Curve – Classification NN")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plot1_path = os.path.join(outputs_dir, "plot1_learning_curve_classification_nn.png")
    fig1.savefig(plot1_path, bbox_inches="tight"); plt.close(fig1)

    # Plot 2: Learning curve (regression Neural Network) – MAE
    fig2 = plt.figure()
    plt.plot(history_reg.history["mae"], label="train_mae")
    plt.plot(history_reg.history["val_mae"], label="val_mae")
    plt.title("Learning Curve – Regression NN")
    plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.legend()
    plot2_path = os.path.join(outputs_dir, "plot2_learning_curve_regression_nn.png")
    fig2.savefig(plot2_path, bbox_inches="tight"); plt.close(fig2)

    # Plot 3: Confusion matrix (best classification model) on TEST
    fig3, ax3 = plt.subplots()
    cm = confusion_matrix(y_class_test_np, best_clf_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format="d", ax=ax3, colorbar=False)
    ax3.set_title(f"Confusion Matrix – Best Classification ({best_clf_name})")
    plot3_path = os.path.join(outputs_dir, "plot3_confusion_matrix_best_classification.png")
    fig3.savefig(plot3_path, bbox_inches="tight"); plt.close(fig3)

    # Plot 4: Residuals vs predicted (best regression) on TEST
    fig4 = plt.figure()
    residuals = y_reg_test_np - best_reg_pred_test
    plt.scatter(best_reg_pred_test, residuals, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.title(f"Residuals vs Predicted – Best Regression ({best_reg_name})")
    plt.xlabel("Predicted"); plt.ylabel("Residuals (y - y_hat)")
    plot4_path = os.path.join(outputs_dir, "plot4_residuals_vs_predicted_best_regression.png")
    fig4.savefig(plot4_path, bbox_inches="tight"); plt.close(fig4)

    # Plot 5: Feature importance (|Linear Regression coefficients|) – Top 20
    coef_abs = pd.Series(linreg_model.coef_, index=feature_names_out).abs().sort_values(ascending=False)
    top_k = min(20, coef_abs.shape[0])
    fig5 = plt.figure(figsize=(8, max(4, top_k * 0.25)))
    coef_abs.iloc[:top_k][::-1].plot(kind="barh")
    plt.title("Feature Importance (|Linear Regression Coefficients|) – Top 20")
    plt.xlabel("Absolute Coefficient")
    plot5_path = os.path.join(outputs_dir, "plot5_feature_importance_regression_linear.png")
    fig5.savefig(plot5_path, bbox_inches="tight"); plt.close(fig5)

    # -----------------------------
    # 9) Save metadata
    # -----------------------------
    summary_payload = {
        "midpoint": saleprice_midpoint,
        "best_classification_model": best_clf_name,
        "best_regression_model": best_reg_name,
        "outputs": {
            "plots_png": [plot1_path, plot2_path, plot3_path, plot4_path, plot5_path],
            "tables_csv": [classification_csv_path, regression_csv_path]
        }
    }
    with open(os.path.join(outputs_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\nSaved files:")
    for p in summary_payload["outputs"]["plots_png"]:
        print("  PNG:", p)
    for p in summary_payload["outputs"]["tables_csv"]:
        print("  CSV:", p)

    return {
        "midpoint": saleprice_midpoint,
        "val": {
            "classification": {"acc_lr": val_acc_lr, "auc_lr": val_auc_lr, "acc_nn": val_acc_nn_clf, "auc_nn": val_auc_nn_clf},
            "regression": {"mae_lin": val_mae_lin, "rmse_lin": val_rmse_lin, "mae_nn": val_mae_nn, "rmse_nn": val_rmse_nn},
        },
        "test": {
            "classification": {"acc_lr": test_acc_lr, "auc_lr": test_auc_lr, "acc_nn": test_acc_nn_clf, "auc_nn": test_auc_nn_clf},
            "regression": {"mae_lin": test_mae_lin, "rmse_lin": test_rmse_lin, "mae_nn": test_mae_nn, "rmse_nn": test_rmse_nn},
        },
        "best_models": {"classification": best_clf_name, "regression": best_reg_name},
        "outputs_dir": outputs_dir
    }


