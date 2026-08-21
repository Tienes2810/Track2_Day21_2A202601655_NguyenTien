import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

EVAL_THRESHOLD = 0.70

RF_PARAM_KEYS = (
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "max_features",
)


def _rf_params(params: dict) -> dict:
    """Lay cac sieu tham so hop le cho RandomForestClassifier (bo None)."""
    cleaned = {}
    for key in RF_PARAM_KEYS:
        if key in params and params[key] is not None:
            cleaned[key] = params[key]
    return cleaned


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    rf_params = _rf_params(params)

    with mlflow.start_run():
        mlflow.log_params(rf_params)

        model = RandomForestClassifier(**rf_params, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Bonus 5: canh bao lech lac du lieu (ty le lop < 10%).
        label_dist = y_train.value_counts(normalize=True).sort_index()
        dist_payload = {
            f"class_ratio_{int(cls)}": float(ratio) for cls, ratio in label_dist.items()
        }
        for cls, ratio in dist_payload.items():
            mlflow.log_metric(cls, ratio)
            if ratio < 0.10:
                print(
                    f"WARNING: {cls} = {ratio:.2%} < 10% — du lieu lech lop, "
                    "can xem xet resampling hoac thu thap them mau."
                )

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": float(acc), "f1_score": float(f1), **dist_payload}, f, indent=2)

        # Bonus 3: bao cao confusion matrix + precision/recall theo lop.
        cm = confusion_matrix(y_eval, preds, labels=[0, 1, 2])
        prec = precision_score(
            y_eval, preds, labels=[0, 1, 2], average=None, zero_division=0
        )
        rec = recall_score(
            y_eval, preds, labels=[0, 1, 2], average=None, zero_division=0
        )
        report_lines = [
            "=== Performance Report ===",
            f"n_train={len(df_train)}  n_eval={len(df_eval)}",
            f"params={rf_params}",
            f"accuracy={acc:.4f}  f1_score={f1:.4f}",
            "",
            "Confusion matrix (rows=true, cols=pred) labels=[0, 1, 2]:",
            str(cm),
            "",
            "Per-class precision / recall:",
            f"  thap(0)       : precision={prec[0]:.4f}  recall={rec[0]:.4f}",
            f"  trung_binh(1) : precision={prec[1]:.4f}  recall={rec[1]:.4f}",
            f"  cao(2)        : precision={prec[2]:.4f}  recall={rec[2]:.4f}",
            "",
            classification_report(
                y_eval,
                preds,
                labels=[0, 1, 2],
                target_names=["thap", "trung_binh", "cao"],
                zero_division=0,
            ),
        ]
        report_text = "\n".join(report_lines)
        with open("outputs/report.txt", "w", encoding="utf-8") as f:
            f.write(report_text + "\n")
        print(report_text)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return float(acc)


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
