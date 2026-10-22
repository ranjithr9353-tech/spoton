# SpotON Pro – Smart Parking with real map + real image analysis

- **Map:** Google Maps (if `GOOGLE_MAPS_API_KEY` is set) or OpenStreetMap (no key needed). Drag pins to fix locations, click **+ Add zone** to add new ones.
- **Analysis:** upload a REAL photo of a parking area. YOLOv8 finds the vehicles, then each slot you defined is marked occupied/vacant.
  With no slots defined it simply counts vehicles and subtracts from the zone capacity.
- **Directions:** every zone has Google Maps directions / open links.

## Run
    pip install -r requirements.txt
    python app.py            # open http://127.0.0.1:5000

Optional Google Maps:
    export GOOGLE_MAPS_API_KEY=your_key      (Windows PowerShell: $env:GOOGLE_MAPS_API_KEY="your_key")

## Use
1. Pick a zone, **Upload image** (a Kaggle parking photo, e.g. from PKLot, or your own).
2. Optional but recommended: **Define slots** – choose "Row of bays", click the 4 corners of a row
   (top-left, top-right, bottom-right, bottom-left), enter how many bays it has. Repeat for each row, then Save.
3. Results, annotated image and map pin colour update automatically.

The first analysis downloads the small YOLO model (yolov8n.pt, ~6 MB).
Uploaded data is stored in `data/` (not committed to git).
