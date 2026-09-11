import os
import mlflow
from mlflow.tracking import MlflowClient

EXP = "Sentiment Analysis"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def log_version(name, version, model, f1, approach, improvement, url, model_path=None):
    with mlflow.start_run(run_name=name):
        mlflow.log_params({
            "model": model,
            "version": version,
            "task": "Sentiment Classification",
            "dataset_size": 4453,
            "classes": "Negative, Neutral, Positive"
        })
        mlflow.log_metric("macro_f1", f1)
        mlflow.set_tags({
            "application": "MBG Smart Monitor",
            "version": version,
            "model_type": "Natural Language Processing",
            "approach": approach,
            "improvement": improvement,
            "streamlit_url": url
        })

        if model_path and os.path.exists(model_path):
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
    "Classic_V1", "V1", "TF-IDF + LinearSVC", 0.806458,
    "Classic NLP", "Baseline",
    "https://mbg-monitor-v1.streamlit.app/",
    os.path.join(BASE, "V1", "models", "sentiment_model",
                 "classic_sentiment_model.pkl")
)

log_version(
    "IndoBERT_V2", "V2", "IndoBERT", 0.904210,
    "Transformer", "Improved from TF-IDF + LinearSVC",
    "https://mbg-monitor-v2.streamlit.app/"
)

print("Sentiment versioning selesai!")