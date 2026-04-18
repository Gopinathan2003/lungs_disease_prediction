import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import sys
from torchvision import models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from pathlib import Path
import subprocess
# from PIL import Image
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.ingestion import ingest_data

mlflow.set_experiment("lung_disease_prediction")

TRAIN_DATA_DIR = Path("data/processed/train")
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


def build_train_dataset():
    ensure_processed_training_data()
    if not TRAIN_DATA_DIR.exists():
        raise FileNotFoundError(f"Training directory not found: {TRAIN_DATA_DIR}")

    dataset = ImageFolder(
        str(TRAIN_DATA_DIR),
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    )

    if len(dataset) == 0:
        raise FileNotFoundError(
            "No training images were found under data/processed/train even after ingestion. "
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

def train():
    with mlflow.start_run(run_name="resnet18-transfer"):
        # Log Git commit for reproducibility

        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        mlflow.log_param("git_commit", commit)

        # Data
        train_dataset = build_train_dataset()
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        model = build_model(len(train_dataset.classes))
        device = torch.device("cpu")  # local hardware
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

        # Training loop (simplified - 3 epochs for demo)
        for epoch in range(EPOCHS):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for inputs, labels in train_loader:
                outputs = model(inputs.to(device))
                _, pred = torch.max(outputs, 1)
                preds.extend(pred.cpu().numpy())
                true.extend(labels.numpy())
        f1 = f1_score(true, preds, average="weighted")
        acc = accuracy_score(true, preds)

        mlflow.log_metrics({"f1_score": f1, "accuracy": acc})
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("trainable_layers", "fc_only")

        # Log model (pyfunc for serving)
        mlflow.pytorch.log_model(model, "model", registered_model_name="lung_disease_model")
        print(f"✅ Training complete. Run ID: {mlflow.active_run().info.run_id} | F1: {f1:.4f}")

if __name__ == "__main__":
    train()
