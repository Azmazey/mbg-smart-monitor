import os
import mlflow
from mlflow.tracking import MlflowClient

EXP = "Fruit Classification"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def log_version(name, version, model, acc, f1, approach, improvement, url, model_path):
    with mlflow.start_run(run_name=name):
        mlflow.log_params({
            "model": model,
            "version": version,
            "task": "Binary Image Classification",
            "dataset_size": 796,
            "classes": "Apple, Orange"
        })
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1})
        mlflow.set_tags({
            "application": "MBG Smart Monitor",
            "version": version,
            "model_type": "Computer Vision",
            "approach": approach,
            "improvement": improvement,
            "streamlit_url": url
        })

        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, "model")

        app = os.path.join(BASE, version, "app.py")
        if os.path.exists(app):
            mlflow.log_artifact(app, "application")


mlflow.set_experiment(EXP)
client = MlflowClient()
exp = client.get_experiment_by_name(EXP)

for r in client.search_runs([exp.experiment_id], max_results=1000):
    client.delete_run(r.info.run_id)

log_version(
    "CNN_V1", "V1", "CNN From Scratch", 0.9167, 0.9180,
    "CNN From Scratch", "Baseline",
    "https://mbg-monitor-v1.streamlit.app/",
    os.path.join(BASE, "V1", "models", "fruit_classifier",
                 "best_scratch_cnn_apple_orange.h5")
)

log_version(
    "MobileNetV2_V2", "V2", "MobileNetV2", 0.9500, 0.9474,
    "Transfer Learning", "Improved from CNN From Scratch",
    "https://mbg-monitor-v2.streamlit.app/",
    os.path.join(BASE, "V2", "models", "fruit_classifier",
                 "fixed_mobilenetv2_apple_orange.h5")
)

print("Fruit versioning selesai!")