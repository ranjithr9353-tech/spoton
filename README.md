# SpotON – Smart Parking Dashboard (OpenCV + Flask)

## Run
```bash
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

`data/Picture1.png` is included. If you delete it, it is regenerated automatically on startup (`make_sample.py`).

## Using a Kaggle image
1. Download any parking-lot dataset from Kaggle (e.g. search "PKLot" or "parking lot images") and pick a frame.
2. In the dashboard, click **MG Road Street Parking → Upload new frame**.
   The image is resized to 480×240 and the 12 slot polygons are applied to it.

Note: the slot polygons and ground-truth labels in `detector.py` are fixed (slots 1–6 occupied, 7–12 vacant),
so the "accuracy" figure is only meaningful for the sample image. For a real dataset, redraw the `SLOTS`
polygons for that camera view and update `GROUND_TRUTH`.

## Files
- `detector.py` – slot polygons + edge/variance occupancy logic
- `app.py` – Flask API (`/api/zones`, `/api/annotated.jpg`, `/api/detect`, `/api/rerun`)
- `static/index.html` – map dashboard (MG Road is live; other zones are demo data)
- `make_sample.py` – generates the sample camera frame
- Tune `OCCUPANCY_THRESHOLD` in `detector.py` if accuracy is low.
