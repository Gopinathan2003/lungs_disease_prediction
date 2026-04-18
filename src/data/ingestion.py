import os
import shutil
from pathlib import Path
import pandas as pd

RAW_TO_PROCESSED_LABELS = {
    "Bacterial Pneumonia": "BACTERIAL_PNEUMONIA",
    "Corona Virus Disease": "CORONA_VIRUS_DISEASE",
    "Normal": "NORMAL",
    "Tuberculosis": "TUBERCULOSIS",
    "Viral Pneumonia": "VIRAL_PNEUMONIA",
}


def ingest_data(raw_path: str = "data/raw/chest_xray", processed_path: str = "data/processed"):
    os.makedirs(processed_path, exist_ok=True)
    # Simple schema validation + copy
    for split in ["train", "test", "val"]:
        for raw_label, processed_label in RAW_TO_PROCESSED_LABELS.items():
            src = Path(raw_path) / split / raw_label
            dst = Path(processed_path) / split / processed_label
            dst.mkdir(parents=True, exist_ok=True)
            if not src.exists():
                continue
            for img in src.iterdir():
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                    continue
                shutil.copy(img, dst / img.name)
    # Log schema
    stats = {"splits": ["train", "test", "val"], "labels": list(RAW_TO_PROCESSED_LABELS.values())}
    pd.DataFrame([stats]).to_csv(f"{processed_path}/data_schema.csv", index=False)
    print("✅ Data ingestion & validation complete")

if __name__ == "__main__":
    ingest_data()
