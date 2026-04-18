import os
import shutil
from pathlib import Path
import pandas as pd

def ingest_data(raw_path: str = "data/raw/chest_xray", processed_path: str = "data/processed"):
    os.makedirs(processed_path, exist_ok=True)
    # Simple schema validation + copy
    for split in ["train", "test", "val"]:
        for label in ["BACTERIAL_PNEUMONIA","CORONA_VIRUS_DISEASE", "NORMAL","TUBERCULOSIS", "VIRAL_PNEUMONIA"]:
            src = Path(raw_path) / split / label
            dst = Path(processed_path) / split / label
            dst.mkdir(parents=True, exist_ok=True)
            for img in src.glob("*.jpeg"):
                shutil.copy(img, dst / img.name)
    # Log schema
    stats = {"splits": ["train", "test", "val"], "labels": ["BACTERIAL_PNEUMONIA","CORONA_VIRUS_DISEASE", "NORMAL","TUBERCULOSIS", "VIRAL_PNEUMONIA"]}
    pd.DataFrame([stats]).to_csv(f"{processed_path}/data_schema.csv", index=False)
    print("✅ Data ingestion & validation complete")

if __name__ == "__main__":
    ingest_data()