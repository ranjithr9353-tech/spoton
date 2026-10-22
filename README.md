# SpotON – Smart Parking Locator (ResNet-50 + Maps)

This project integrates deep-learning ResNet-50 bounding box detection (adapted from Kaggle's `shivamb/objects-bounding-boxes-using-resnet50-imageai`) with an interactive map locator to detect and monitor parking vacancy in real-time.

## Features
- **ResNet-50 Object Detection Backbone**: Detects cars, buses, trucks, and motorcycles using neural bounding box predictions and IoU overlap over parking slots.
- **Fail-safe Computer Vision Fallback**: If running in an offline environment without PyTorch preloaded weights, automatically falls back to OpenCV edge density & variance heuristics without crashing.
- **Interactive Map Locator**: Displays real-time vacancy numbers directly on map markers with automatic camera inspection sidebar.
- **Kaggle Kernel & Dataset Utilities**: Includes `kaggle_setup.py` to pull the ResNet50 notebook and the PKLot dataset.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate default sample frame:
   ```bash
   python make_sample.py
   ```

3. Launch application:
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000` in your web browser.

## Pulling Kaggle Kernel & Dataset
Configure your `~/.kaggle/kaggle.json` token, then run:
```bash
python kaggle_setup.py
```
Or run directly:
```bash
kaggle kernels pull shivamb/objects-bounding-boxes-using-resnet50-imageai
kaggle datasets download -d alceubr/pklot -p data/ --unzip
```
