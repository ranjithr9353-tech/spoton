# SpotON: Smart Real-Time Multi-Zone Parking Management System

Self-contained vision-based parking vacancy detection and GIS locator dashboard.

## 🚀 Instant Run Instructions

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run with a single command**:
   ```bash
   python run.py
   ```
   *(Or run `python app.py`)*

3. **Open in your browser**:
   Navigate to:
   ```
   http://127.0.0.1:5000
   ```

---

## 🌟 Included Features
- **Zero API Key Requirement**: Uses high-performance dark Carto/OpenStreetMap tiles that load instantly anywhere without Google Cloud billing blocks.
- **Pre-packaged Sample Frame**: Automatically generates `data/Picture1.png` on launch.
- **Live Frame Upload**: Use the dashboard UI to upload any parking lot image (e.g. from Kaggle PKLot) to test real-time detection.
- **Multi-Zone Support**: 5 distinct zone categories (Street, Commercial, Residential, Campus, Temporary) with live slot meters.
