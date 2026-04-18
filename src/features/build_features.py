import json
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

LABELS = [
    "BACTERIAL_PNEUMONIA",
    "CORONA_VIRUS_DISEASE",
    "NORMAL",
    "TUBERCULOSIS",
    "VIRAL_PNEUMONIA",
]

LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def build_features():
    data_path = Path("data/processed")
    metadata = []
    pixel_means = []
    pixel_stds = []

    for split in ["train"]:
        for label in LABELS:
            label_dir = data_path / split / label
            if not label_dir.exists():
                continue

            for img_path in sorted(label_dir.iterdir()):
                if img_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img)
                metadata.append({
                    "path": str(img_path),
                    "label": LABEL_TO_ID[label],
                    "label_name": label,
                    "split": split,
                })
                pixel_means.append(tensor.mean(dim=[1, 2]).tolist())
                pixel_stds.append(tensor.std(dim=[1, 2]).tolist())

    if not metadata:
        raise ValueError(f"No training images found under {data_path / 'train'}")

    df = pd.DataFrame(metadata)
    df.to_csv(data_path / "train_metadata.csv", index=False)

    baseline = {
        "labels": LABELS,
        "label_to_id": LABEL_TO_ID,
        "mean": torch.tensor(pixel_means).mean(dim=0).tolist(),
        "std": torch.tensor(pixel_stds).mean(dim=0).tolist(),
    }
    with open(data_path / "baseline_stats.json", "w") as f:
        json.dump(baseline, f)

    print("✅ Feature engineering + baseline computed")


if __name__ == "__main__":
    build_features()
