import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


def train_logistic_regression(X_train, X_test, y_train, y_test, config: dict):
    params = config["models"]["logistic_regression"]
    model_name = "LogisticRegression"

    with mlflow.start_run(run_name=model_name, nested=True):
        mlflow.log_params(params)
        mlflow.set_tag("model_type", model_name)

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
        mlflow.log_artifact(cm_path, artifact_path="plots")
        os.unlink(cm_path)

        mlflow.sklearn.log_model(
            sk_model=model,
            name=model_name,
            registered_model_name=None,
            input_example=X_train[:5],
        )

        print(f"{model_name} — F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        return model, metrics


def train_random_forest(X_train, X_test, y_train, y_test, config: dict):
    params = config["models"]["random_forest"]
    model_name = "RandomForest"

    with mlflow.start_run(run_name=model_name, nested=True):
        mlflow.log_params(params)
        mlflow.set_tag("model_type", model_name)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        # Feature importance grafigi
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        importances = pd.Series(model.feature_importances_, index=feature_names)
        fig, ax = plt.subplots(figsize=(6, 4))
        importances.sort_values().plot(kind="barh", ax=ax)
        ax.set_title("Feature Importances")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(tmp.name, artifact_path="plots")
        os.unlink(tmp.name)

        cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
        mlflow.log_artifact(cm_path, artifact_path="plots")
        os.unlink(cm_path)

        mlflow.sklearn.log_model(
            sk_model=model,
            name=model_name,
            registered_model_name=None,
            input_example=X_train[:5],
        )

        print(f"{model_name} — F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        return model, metrics


def train_gradient_boosting(X_train, X_test, y_train, y_test, config: dict):
    params = config["models"]["gradient_boosting"]
    model_name = "GradientBoosting"

    with mlflow.start_run(run_name=model_name, nested=True):
        mlflow.log_params(params)
        mlflow.set_tag("model_type", model_name)

        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
        mlflow.log_artifact(cm_path, artifact_path="plots")
        os.unlink(cm_path)

        mlflow.sklearn.log_model(
            sk_model=model,
            name=model_name,
            registered_model_name=None,
            input_example=X_train[:5],
        )

        print(f"{model_name} — F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        return model, metrics
