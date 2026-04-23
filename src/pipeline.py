"""
Asosiy MLOps pipeline: data → preprocessing → training → registry → serving
"""
import os
import sys
import mlflow
import yaml

# src papkasini path ga qo'shish
sys.path.insert(0, os.path.dirname(__file__))

from data_preprocessing import load_config, load_data, preprocess, split_and_scale, get_data_stats
from model_training import train_logistic_regression, train_random_forest, train_gradient_boosting
from model_registry import register_best_model, list_registered_models, load_production_model


def run_pipeline(config_path: str = "../configs/config.yaml"):
    config = load_config(config_path)

    # MLflow sozlash
    tracking_uri = os.path.join(os.path.dirname(__file__), "..", "..", "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(tracking_uri)}")
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    print("=" * 55)
    print("   MLOps Pipeline — Titanic Survival Prediction")
    print("=" * 55)

    # 1. Data yuklash
    data_path = os.path.join(os.path.dirname(__file__), "..", config["data"]["raw_path"])
    df = load_data(os.path.abspath(data_path))
    stats = get_data_stats(df)

    # 2. Preprocessing
    X, y = preprocess(df, config)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, config)

    # 3. Barcha modellarni parent run ostida trening
    with mlflow.start_run(run_name="experiment_comparison") as parent_run:
        mlflow.log_params({
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": X_train.shape[1],
        })
        mlflow.log_metrics({
            "data_survival_rate": stats["survival_rate"],
            "data_missing_values": stats["missing_values"],
        })
        mlflow.set_tag("pipeline_stage", "training")

        print("\n[1/3] Logistic Regression trening...")
        lr_model, lr_metrics = train_logistic_regression(
            X_train, X_test, y_train, y_test, config
        )

        print("[2/3] Random Forest trening...")
        rf_model, rf_metrics = train_random_forest(
            X_train, X_test, y_train, y_test, config
        )

        print("[3/3] Gradient Boosting trening...")
        gb_model, gb_metrics = train_gradient_boosting(
            X_train, X_test, y_train, y_test, config
        )

        # Natijalarni solishtirish
        results = {
            "LogisticRegression": lr_metrics,
            "RandomForest": rf_metrics,
            "GradientBoosting": gb_metrics,
        }

        print("\n" + "=" * 55)
        print(f"{'Model':<25} {'F1':>8} {'ROC-AUC':>10} {'Accuracy':>10}")
        print("-" * 55)
        for name, m in results.items():
            print(f"{name:<25} {m['f1']:>8.4f} {m['roc_auc']:>10.4f} {m['accuracy']:>10.4f}")
        print("=" * 55)

        best_name = max(results, key=lambda k: results[k]["roc_auc"])
        print(f"\nEng yaxshi model: {best_name} (ROC-AUC: {results[best_name]['roc_auc']:.4f})")

        parent_run_id = parent_run.info.run_id

    # 4. Eng yaxshi modelni registratsiya qilish
    print("\nModel registratsiya qilinmoqda...")
    try:
        mv = register_best_model(
            experiment_name=config["mlflow"]["experiment_name"],
            registered_model_name=config["mlflow"]["registered_model_name"],
            metric="roc_auc",
        )
        list_registered_models(config["mlflow"]["registered_model_name"])
    except Exception as e:
        print(f"Registry xatoligi (normal): {e}")

    # 5. Inference namunasi
    print("\nInference namunasi (birinchi 3 test namunasi):")
    best_models = {
        "LogisticRegression": lr_model,
        "RandomForest": rf_model,
        "GradientBoosting": gb_model,
    }
    best_model = best_models[best_name]
    sample_preds = best_model.predict(X_test[:3])
    sample_probs = best_model.predict_proba(X_test[:3])[:, 1]
    for i, (pred, prob) in enumerate(zip(sample_preds, sample_probs)):
        status = "Yashadi" if pred == 1 else "Halok bo'ldi"
        print(f"  Namuna {i+1}: {status} (ehtimollik: {prob:.2%})")

    print("\nPipeline muvaffaqiyatli yakunlandi!")
    print(f"MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    return results


if __name__ == "__main__":
    run_pipeline()
