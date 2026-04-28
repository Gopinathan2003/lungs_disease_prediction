import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.ingestion import ingest_data

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("lung_disease_prediction")

PROCESSED_DATA_DIR = Path("data/processed")
TRAIN_DATA_DIR = PROCESSED_DATA_DIR / "train"
TEST_DATA_DIR = PROCESSED_DATA_DIR / "test"
MODEL_EXPORT_DIR = Path("models/latest")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
BATCH_SIZE = 32
EPOCHS = 3


def ensure_processed_training_data():
    image_count = sum(
        1
        for path in TRAIN_DATA_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if image_count == 0:
        ingest_data()


def build_dataset(split: str):
    ensure_processed_training_data()
    split_dir = PROCESSED_DATA_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {split_dir}")

    dataset = ImageFolder(
        str(split_dir),
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    )

    if len(dataset) == 0:
        raise FileNotFoundError(
            f"No {split} images were found under {split_dir} even after ingestion. "
            "Check that data/raw/chest_xray contains the expected class folders and images."
        )

    return dataset


def build_model(num_classes: int):
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Standard transfer learning: keep the pretrained backbone fixed and train only the classifier head.
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    preds, true = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            predicted = torch.argmax(outputs, dim=1)
            preds.extend(predicted.cpu().numpy())
            true.extend(labels.cpu().numpy())

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": accuracy_score(true, preds),
        "f1_score": f1_score(true, preds, average="weighted", zero_division=0),
    }


def log_metric_plots(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metric_titles = [
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
        ("f1_score", "F1 Score"),
    ]

    epochs = range(1, len(history["train"]["loss"]) + 1)
    for axis, (metric_key, title) in zip(axes, metric_titles):
        axis.plot(epochs, history["train"][metric_key], marker="o", label="train")
        axis.plot(epochs, history["test"][metric_key], marker="o", label="test")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.tight_layout()
    mlflow.log_figure(fig, "plots/train_test_metrics.png")
    plt.close(fig)


def export_model(model):
    if MODEL_EXPORT_DIR.exists():
        shutil.rmtree(MODEL_EXPORT_DIR)
    mlflow.pytorch.save_model(model, str(MODEL_EXPORT_DIR))


def train():
    with mlflow.start_run(run_name="resnet18-transfer"):
        # Log Git commit for reproducibility

        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        mlflow.log_param("git_commit", commit)

        # Data
        train_dataset = build_dataset("train")
        test_dataset = build_dataset("test")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        model = build_model(len(train_dataset.classes))
        device = torch.device("cpu")  # local hardware
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
        history = {
            "train": {"loss": [], "accuracy": [], "f1_score": []},
            "test": {"loss": [], "accuracy": [], "f1_score": []},
        }

        # Training loop (simplified - 3 epochs for demo)
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            epoch_examples = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                batch_size = labels.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_examples += batch_size

            train_metrics = evaluate_model(model, train_loader, criterion, device)
            test_metrics = evaluate_model(model, test_loader, criterion, device)
            train_metrics["loss"] = epoch_loss / max(epoch_examples, 1)

            for split_name, split_metrics in (("train", train_metrics), ("test", test_metrics)):
                for metric_name, metric_value in split_metrics.items():
                    history[split_name][metric_name].append(metric_value)
                    mlflow.log_metric(f"{split_name}_{metric_name}", metric_value, step=epoch + 1)

        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("trainable_layers", "fc_only")
        mlflow.log_param("tracking_uri", MLFLOW_TRACKING_URI)
        mlflow.log_param("classes", ",".join(train_dataset.classes))
        log_metric_plots(history)

        # Log model (pyfunc for serving)
        mlflow.pytorch.log_model(model, "model", registered_model_name="lung_disease_model")
        export_model(model)

        final_test_f1 = history["test"]["f1_score"][-1]
        final_test_accuracy = history["test"]["accuracy"][-1]
        print(
            "✅ Training complete. "
            f"Run ID: {mlflow.active_run().info.run_id} | "
            f"Test Accuracy: {final_test_accuracy:.4f} | Test F1: {final_test_f1:.4f}"
        )

if __name__ == "__main__":
    train()
