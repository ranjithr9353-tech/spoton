"""SpotON backend: runs ResNet50 detector and serves dashboard with Google Maps."""
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

# Real Lat/Lng coordinates around Bengaluru
ZONES = [
    {"id": 1, "name": "MG Road Street Parking (Live ResNet50)", "zone_type": "street", "lat": 12.9756, "lng": 77.6066, "area": "MG Road", "live": True},
    {"id": 2, "name": "Brigade Road Commercial Lot", "zone_type": "commercial", "total": 20, "occupied": 14, "lat": 12.9719, "lng": 77.6070, "area": "Brigade Road", "live": False},
    {"id": 3, "name": "Koramangala 5th Block Complex", "zone_type": "residential", "total": 15, "occupied": 6, "lat": 12.9352, "lng": 77.6245, "area": "Koramangala", "live": False},
    {"id": 4, "name": "Presidency University Campus Lot", "zone_type": "campus", "total": 25, "occupied": 15, "lat": 13.1678, "lng": 77.5342, "area": "Yelahanka", "live": False},
    {"id": 5, "name": "HSR Sector 2 Market Lot", "zone_type": "temporary", "total": 10, "occupied": 7, "lat": 12.9116, "lng": 77.6389, "area": "HSR Layout", "live": False},
]

def run_detection():
    try:
        if not IMAGE.exists():
            subprocess.run([sys.executable, str(BASE / "make_sample.py")], cwd=BASE, check=True)
        img, results, detected_vehicles = classify_slots(str(IMAGE))
        annotated = draw_annotated_frame(img, results, detected_vehicles)
        ok, buf = cv2.imencode(".jpg", annotated)
        state.update(results=results, jpg=buf.tobytes(), error=None, accuracy=accuracy(results))
    except Exception as e:
        state.update(results=None, jpg=None, error=str(e), accuracy=None)

def build_zones():
    zones_output = []
    r = state["results"]
    
    # Zone 1 (Live AI-monitored camera)
    z1 = ZONES[0]
    if r:
        slots = [{"slot_label": f"MG-{i:02d}", "status": r[i]["label"].lower(),
                  "confidence": r[i]["confidence"]} for i in sorted(r)]
        occ = sum(s["status"] == "occupied" for s in slots)
        zones_output.append({
            **z1,
            "total_slots": len(slots),
            "occupied_slots": occ,
            "vacant_slots": len(slots) - occ,
            "accuracy": state["accuracy"],
            "slots": slots
        })
    else:
        zones_output.append({
            **z1,
            "total_slots": 0, "occupied_slots": 0, "vacant_slots": 0,
            "error": state["error"], "slots": []
        })

    # Demo Zones 2-5
    for z in ZONES[1:]:
        prefix = z["zone_type"][:2].upper()
        slots = [{"slot_label": f"{prefix}-{i:02d}",
                  "status": "occupied" if i <= z["occupied"] else "vacant"}
                 for i in range(1, z["total"] + 1)]
        zones_output.append({
            **z,
            "total_slots": z["total"],
            "occupied_slots": z["occupied"],
            "vacant_slots": z["total"] - z["occupied"],
            "slots": slots
        })
    return zones_output

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
    f = request.files.get("image")
    if not f:
        return jsonify(error="No file uploaded (field name: image)"), 400
    f.save(IMAGE)
    run_detection()
    if state["error"]:
        return jsonify(error=state["error"]), 400
    return jsonify(ok=True, accuracy=state["accuracy"])

if __name__ == "__main__":
    run_detection()
    app.run(debug=True, port=5000)
