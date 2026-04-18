import json
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd

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
        for label in ["NORMAL", "PNEUMONIA"]:
            for img_path in (data_path / split / label).glob("*.jpeg"):
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img)
                metadata.append({"path": str(img_path), "label": 0 if label == "NORMAL" else 1, "split": split})
                pixel_means.append(tensor.mean(dim=[1,2]).tolist())
                pixel_stds.append(tensor.std(dim=[1,2]).tolist())

    df = pd.DataFrame(metadata)
    df.to_csv(data_path / "train_metadata.csv", index=False)

    # Drift baseline (mean, variance per channel)
    baseline = {
        "mean": torch.tensor(pixel_means).mean(dim=0).tolist(),
        "std": torch.tensor(pixel_stds).mean(dim=0).tolist()
    }
    with open(data_path / "baseline_stats.json", "w") as f:
        json.dump(baseline, f)
    print("✅ Feature engineering + baseline computed")

if __name__ == "__main__":
    build_features()