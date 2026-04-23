import mlflow
from mlflow.tracking import MlflowClient


def register_best_model(
    experiment_name: str,
    registered_model_name: str,
    metric: str = "roc_auc",
):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' topilmadi")

    # Barcha runlarni olish
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=[f"metrics.{metric} DESC"],
    )

    if not runs:
        raise ValueError("Hech qanday run topilmadi")

    best_run = runs[0]
    best_metric = best_run.data.metrics.get(metric, 0)
    best_run_id = best_run.info.run_id
    model_name = best_run.data.tags.get("model_type", "unknown")

    print(f"\nEng yaxshi model: {model_name}")
    print(f"  Run ID: {best_run_id}")
    print(f"  {metric}: {best_metric:.4f}")

    # Model URI — nested run bo'lgani uchun artifact path ni aniqlash
    artifact_uri = best_run.info.artifact_uri
    model_uri = f"runs:/{best_run_id}/{model_name}"

    # Model registratsiya
    mv = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
    )

    # Production bosqichiga o'tkazish
    client.transition_model_version_stage(
        name=registered_model_name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )

    print(f"\nModel '{registered_model_name}' v{mv.version} → Production ga o'tkazildi")
    return mv


def list_registered_models(registered_model_name: str):
    client = MlflowClient()
    versions = client.get_latest_versions(registered_model_name)
    print(f"\n'{registered_model_name}' modeli versiyalari:")
    for v in versions:
        print(f"  v{v.version} | Stage: {v.current_stage} | Run: {v.run_id[:8]}...")
    return versions


def load_production_model(registered_model_name: str):
    model_uri = f"models:/{registered_model_name}/Production"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Production modeli yuklandi: {registered_model_name}")
    return model
