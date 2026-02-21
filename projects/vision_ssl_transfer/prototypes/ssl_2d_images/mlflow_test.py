import os
import tempfile
import mlflow
import torch
import torch.nn as nn
import os


# -------------------------------------------------
# 1. Connect to Docker MLflow server
# -------------------------------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")

print("Tracking URI:", mlflow.get_tracking_uri())

# -------------------------------------------------
# 2. Create or set experiment
# -------------------------------------------------
mlflow.set_experiment("docker_connection_test")

# -------------------------------------------------
# 3. Dummy PyTorch model
# -------------------------------------------------
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model = DummyModel()

# -------------------------------------------------
# 4. Start MLflow run
# -------------------------------------------------
with mlflow.start_run() as run:
    print("Run ID:", run.info.run_id)

    # Log parameters
    mlflow.log_param("test_param", 42)

    # Log metrics
    for step in range(5):
        mlflow.log_metric("test_metric", step * 0.1, step=step)

    mlflow.pytorch.log_model(model, name="dummy_model")

print("✅ Test completed successfully.")