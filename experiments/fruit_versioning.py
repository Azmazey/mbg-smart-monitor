import mlflow

mlflow.set_experiment("Fruit Classification")

# CNN V1
with mlflow.start_run(run_name="CNN_V1"):
    mlflow.log_param("model", "CNN From Scratch")
    mlflow.log_param("version", "V1")

    mlflow.log_metric("accuracy", 0.9167)
    mlflow.log_metric("precision", 0.XXXX)
    mlflow.log_metric("recall", 0.XXXX)
    mlflow.log_metric("f1_score", 0.XXXX)

    mlflow.log_artifact("fruit_versioning.py")


# MobileNetV2 V2
with mlflow.start_run(run_name="MobileNetV2_V2"):
    mlflow.log_param("model", "MobileNetV2")
    mlflow.log_param("version", "V2")

    mlflow.log_metric("accuracy", 0.9500)
    mlflow.log_metric("precision", 0.XXXX)
    mlflow.log_metric("recall", 0.XXXX)
    mlflow.log_metric("f1_score", 0.9474)

    mlflow.log_artifact("fruit_versioning.py")