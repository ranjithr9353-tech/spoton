"""
Kaggle Kernel & Dataset Fetcher
Automates pulling shivamb/objects-bounding-boxes-using-resnet50-imageai and parking datasets.
"""
import os, sys, subprocess

def check_kaggle_credentials():
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(kaggle_json):
        print("Note: ~/.kaggle/kaggle.json not found.")
        print("To download directly via Kaggle CLI:")
        print("1. Go to https://www.kaggle.com/settings -> 'Create New Token'")
        print(f"2. Place kaggle.json in {kaggle_dir} and chmod 600 {kaggle_json}")
        return False
    return True

def pull_resnet_kernel():
    if not check_kaggle_credentials():
        return
    print("Pulling Kaggle kernel: shivamb/objects-bounding-boxes-using-resnet50-imageai ...")
    try:
        subprocess.run(["kaggle", "kernels", "pull", "shivamb/objects-bounding-boxes-using-resnet50-imageai", "-p", "kaggle_kernel"], check=True)
        print("Kernel successfully pulled to ./kaggle_kernel/")
    except Exception as e:
        print(f"Failed to pull kernel: {e}")

def pull_pklot_dataset():
    if not check_kaggle_credentials():
        return
    print("Pulling PKLot parking dataset ...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", "alceubr/pklot", "-p", "data/", "--unzip"], check=True)
        print("PKLot dataset downloaded to ./data/")
    except Exception as e:
        print(f"Failed to pull dataset: {e}")

if __name__ == "__main__":
    pull_resnet_kernel()
    pull_pklot_dataset()
