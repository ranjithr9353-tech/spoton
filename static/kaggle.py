"""
Downloads parking lot datasets directly from Kaggle.
Requires ~/.kaggle/kaggle.json with your API key.
"""
import os
import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "kaggle_frames"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_dataset():
    print("Checking Kaggle CLI...")
    try:
        # Downloads PKLot sample frames into data/kaggle_frames
        print(f"Downloading dataset to {DATA_DIR}...")
        subprocess.run([
            "kaggle", "datasets", "download", 
            "-d", "alceubr/pklot", 
            "-p", str(DATA_DIR), 
            "--unzip"
        ], check=True)
        print("Dataset downloaded and extracted successfully!")
    except FileNotFoundError:
        print("Error: 'kaggle' command not found. Install it with: pip install kaggle")
        print("Ensure your kaggle.json token is placed in ~/.kaggle/kaggle.json")
    except Exception as e:
        print(f"Failed to download from Kaggle: {e}")

if __name__ == "__main__":
    download_dataset()