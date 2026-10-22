"""SpotON backend: runs the OpenCV detector and serves the dashboard.
Run:  python app.py   ->  http://127.0.0.1:5000
"""
import subprocess, sys
from pathlib import Path
import cv2
from flask import Flask, jsonify, request, Response, send_from_directory
from detector import classify_slots, draw_annotated_frame, accuracy

BASE = Path(__file__).parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)
IMAGE = DATA / "Picture1.png"

app = Flask(__name__, static_folder="static")
state = {"results": None, "jpg": None, "error": None, "accuracy": None}

# Zone 1 is live (camera image + OpenCV). Others are demo data.
DEMO_ZONES = [
    {"id": 2, "name": "City Mall Commercial Lot", "zone_type": "commercial", "total": 20, "occupied": 14, "x": 330, "y": 210, "area": "Brigade Road"},
    {"id": 3, "name": "Green Residency Apartments", "zone_type": "residential", "total": 15, "occupied": 6, "x": 160, "y": 370, "area": "Koramangala"},
    {"id": 4, "name": "Presidency University Campus Lot", "zone_type": "campus", "total": 25, "occupied": 15, "x": 70, "y": 90, "area": "Yelahanka"},
    {"id": 5, "name": "Weekend Market Temporary Lot", "zone_type": "temporary", "total": 10, "occupied": 7, "x": 320, "y": 440, "area": "HSR Layout"},
]


def run_detection():
    try:
        if not IMAGE.exists():  # first run: create the sample frame automatically
            subprocess.run([sys.executable, str(BASE / "make_sample.py")], cwd=BASE, check=True)
        img, results = classify_slots(str(IMAGE))
        ok, buf = cv2.imencode(".jpg", draw_annotated_frame(img, results))
        state.update(results=results, jpg=buf.tobytes(), error=None, accuracy=accuracy(results))
    except Exception as e:  # missing/corrupt image
        state.update(results=None, jpg=None, error=str(e), accuracy=None)


def build_zones():
    zones = []
    r = state["results"]
    if r:
        slots = [{"slot_label": f"MG-{i:02d}", "status": r[i]["label"].lower(),
                  "confidence": r[i]["confidence"]} for i in sorted(r)]
        occ = sum(s["status"] == "occupied" for s in slots)
        zones.append({"id": 1, "name": "MG Road Street Parking", "zone_type": "street",
                      "total_slots": len(slots), "occupied_slots": occ,
                      "vacant_slots": len(slots) - occ, "x": 200, "y": 150,
                      "area": "MG Road", "live": True, "accuracy": state["accuracy"],
                      "slots": slots})
    else:
        zones.append({"id": 1, "name": "MG Road Street Parking", "zone_type": "street",
                      "total_slots": 0, "occupied_slots": 0, "vacant_slots": 0,
                      "x": 200, "y": 150, "area": "MG Road", "live": True,
                      "error": state["error"], "slots": []})
    for z in DEMO_ZONES:
        prefix = z["zone_type"][:2].upper()
        slots = [{"slot_label": f"{prefix}-{i:02d}",
                  "status": "occupied" if i <= z["occupied"] else "vacant"}
                 for i in range(1, z["total"] + 1)]
        zones.append({**{k: z[k] for k in ("id", "name", "zone_type", "x", "y", "area")},
                      "total_slots": z["total"], "occupied_slots": z["occupied"],
                      "vacant_slots": z["total"] - z["occupied"], "live": False, "slots": slots})
    return zones


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/zones")
def api_zones():
    return jsonify(build_zones())


@app.get("/api/annotated.jpg")
def api_annotated():
    if not state["jpg"]:
        return Response(status=404)
    return Response(state["jpg"], mimetype="image/jpeg")


@app.post("/api/detect")
def api_detect():
    """Upload a new camera frame (form field 'image') and re-run detection."""
    f = request.files.get("image")
    if not f:
        return jsonify(error="No file uploaded (field name: image)"), 400
    f.save(IMAGE)
    run_detection()
    if state["error"]:
        return jsonify(error=state["error"]), 400
    return jsonify(ok=True, accuracy=state["accuracy"])


@app.post("/api/rerun")
def api_rerun():
    run_detection()
    return jsonify(ok=state["error"] is None, error=state["error"])


if __name__ == "__main__":
    run_detection()
    app.run(debug=True, use_reloader=False)
