import os, sys, subprocess, random
from pathlib import Path
from datetime import datetime
import cv2
from flask import Flask, jsonify, request, Response, send_from_directory
import detector

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)
STATIC_DIR = BASE / "static"
IMAGE = DATA / "Picture1.png"

app = Flask(__name__, static_folder=str(STATIC_DIR))
state = {"results": None, "jpg": None, "error": None, "accuracy": None, "boxes": None}

# Base definitions for all monitored zones
ZONES = [
    {"id": 1, "name": "MG Road Street Parking (Live Vision)", "zone_type": "street", "total": 12, "lat": 12.9756, "lng": 77.6066, "area": "MG Road", "live": True},
    {"id": 2, "name": "Brigade Road Commercial Lot", "zone_type": "commercial", "total": 20, "lat": 12.9719, "lng": 77.6070, "area": "Brigade Road", "live": False},
    {"id": 3, "name": "Koramangala 5th Block Complex", "zone_type": "residential", "total": 15, "lat": 12.9352, "lng": 77.6245, "area": "Koramangala", "live": False},
    {"id": 4, "name": "Presidency University Campus Lot", "zone_type": "campus", "total": 25, "lat": 13.1678, "lng": 77.5342, "area": "Yelahanka", "live": False},
    {"id": 5, "name": "HSR Sector 2 Temporary Lot", "zone_type": "temporary", "total": 10, "lat": 12.9116, "lng": 77.6389, "area": "HSR Layout", "live": False},
]

# Track live occupancy state across polling calls
live_occupancy_state = {
    2: 14,
    3: 6,
    4: 15,
    5: 7
}

def run_detection():
    try:
        if not IMAGE.exists():
            subprocess.run([sys.executable, str(BASE / "make_sample.py")], cwd=str(BASE), check=True)
        img, results, detected_boxes = detector.classify_slots(str(IMAGE))
        annotated = detector.draw_annotated_frame(img, results, detected_boxes)
        ok, buf = cv2.imencode(".jpg", annotated)
        state.update(
            results=results,
            jpg=buf.tobytes(),
            error=None,
            accuracy=detector.accuracy(results),
            boxes=detected_boxes
        )
    except Exception as e:
        state.update(results=None, jpg=None, error=str(e), accuracy=None, boxes=None)

def build_zones():
    zones_output = []
    r = state["results"]
    z1 = ZONES[0]
    
    # 1. Zone 1 (Camera-monitored live zone)
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
            "total_slots": 12, "occupied_slots": 6, "vacant_slots": 6,
            "error": state["error"], "slots": []
        })

    # 2. Zones 2-5: Dynamically drift occupancy randomly to simulate real-time vehicles arriving/departing
    for z in ZONES[1:]:
        zid = z["id"]
        total = z["total"]
        curr = live_occupancy_state[zid]
        
        # Random step: -1 (car left), 0 (no change), or +1 (car parked)
        change = random.choice([-1, 0, 1])
        new_occ = max(1, min(total, curr + change))
        live_occupancy_state[zid] = new_occ

        prefix = z["zone_type"][:2].upper()
        # Randomize slot distribution matching the new occupancy count
        slot_status = ["occupied"] * new_occ + ["vacant"] * (total - new_occ)
        random.seed(zid + new_occ) # stable per step
        random.shuffle(slot_status)

        slots = [{"slot_label": f"{prefix}-{i+1:02d}", "status": slot_status[i]} for i in range(total)]
        
        zones_output.append({
            **z,
            "total_slots": total,
            "occupied_slots": new_occ,
            "vacant_slots": total - new_occ,
            "slots": slots
        })
        
    return zones_output

@app.get("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")

@app.get("/api/zones")
def api_zones():
    return jsonify({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "zones": build_zones()
    })

@app.get("/api/annotated.jpg")
def api_annotated():
    if not state["jpg"]:
        return Response(status=404)
    return Response(state["jpg"], mimetype="image/jpeg")

@app.post("/api/detect")
def api_detect():
    f = request.files.get("image")
    if not f:
        return jsonify(error="No file uploaded (field: image)"), 400
    f.save(IMAGE)
    run_detection()
    if state["error"]:
        return jsonify(error=state["error"]), 400
    return jsonify(ok=True, accuracy=state["accuracy"])

if __name__ == "__main__":
    run_detection()
    print("Serving SpotON on http://127.0.0.1:5000 ...")
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
