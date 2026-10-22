"""SpotON Pro backend.
Run:  python app.py  ->  http://127.0.0.1:5000
Optional: set GOOGLE_MAPS_API_KEY to use Google Maps (otherwise OpenStreetMap is used).
"""
import json, os, threading, time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

import detector

BASE = Path(__file__).parent
DATA = BASE / "data"
IMG, ANN = DATA / "images", DATA / "annotated"
for p in (DATA, IMG, ANN):
    p.mkdir(parents=True, exist_ok=True)
STORE = DATA / "zones.json"
LOCK = threading.RLock()

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

# Approximate real Bengaluru locations. Drag a marker on the map to correct it.
SEED = [
    ("MG Road Street Parking", "street", "MG Road", 12.9756, 77.6066, 12),
    ("City Mall Commercial Lot", "commercial", "Brigade Road", 12.9719, 77.6072, 20),
    ("Green Residency Apartments", "residential", "Koramangala", 12.9352, 77.6245, 15),
    ("Presidency University Campus Lot", "campus", "Yelahanka", 13.1620, 77.5640, 25),
    ("Weekend Market Temporary Lot", "temporary", "HSR Layout", 12.9116, 77.6474, 10),
]


def new_zone(i, name, ztype, area, lat, lng, cap):
    return {"id": i, "name": name, "zone_type": ztype, "area": area, "lat": lat, "lng": lng,
            "capacity": cap, "slots": [], "method": "auto", "has_image": False, "result": None, "rev": 0}


def load():
    with LOCK:
        if not STORE.exists():
            save([new_zone(i + 1, *s) for i, s in enumerate(SEED)])
        return json.loads(STORE.read_text())


def save(zones):
    with LOCK:
        STORE.write_text(json.dumps(zones, indent=1))


def find(zones, zid):
    for z in zones:
        if z["id"] == zid:
            return z
    return None


def view(z):
    r = z["result"]
    out = {k: z[k] for k in ("id", "name", "zone_type", "area", "lat", "lng", "capacity", "slots", "method", "has_image", "rev")}
    out["result"] = r
    out["status"] = "live" if r else "no_data"
    out["total_slots"] = r["total"] if r else z["capacity"]
    out["occupied_slots"] = r["occupied"] if r else None
    out["vacant_slots"] = r["vacant"] if r else None
    return out


def run_analysis(z):
    path = IMG / f"zone_{z['id']}.jpg"
    out = detector.analyze(str(path), z["slots"], z["method"], z["capacity"])
    cv2.imwrite(str(ANN / f"zone_{z['id']}.jpg"), out.pop("annotated"))
    out["analyzed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    z["result"] = out
    z["rev"] += 1


def err(msg, code=400):
    return jsonify(error=str(msg)), code


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/config")
def api_config():
    return jsonify(google_key=os.environ.get("GOOGLE_MAPS_API_KEY", ""), yolo_installed=detector.yolo_installed())


@app.get("/api/zones")
def api_zones():
    return jsonify([view(z) for z in load()])


@app.post("/api/zones")
def api_add():
    d = request.get_json(silent=True) or {}
    try:
        lat, lng, cap = float(d["lat"]), float(d["lng"]), int(d.get("capacity", 10))
    except (KeyError, ValueError, TypeError):
        return err("lat, lng and capacity are required numbers")
    with LOCK:
        zones = load()
        z = new_zone(max([q["id"] for q in zones] + [0]) + 1, str(d.get("name") or "New zone")[:80],
                     str(d.get("zone_type") or "street"), str(d.get("area") or ""), lat, lng, max(cap, 1))
        zones.append(z)
        save(zones)
    return jsonify(view(z)), 201


@app.patch("/api/zones/<int:zid>")
def api_patch(zid):
    d = request.get_json(silent=True) or {}
    with LOCK:
        zones = load()
        z = find(zones, zid)
        if not z:
            return err("No such zone", 404)
        try:
            for k in ("name", "zone_type", "area"):
                if k in d:
                    z[k] = str(d[k])[:80]
            for k in ("lat", "lng"):
                if k in d:
                    z[k] = float(d[k])
            if "capacity" in d:
                z["capacity"] = max(int(d["capacity"]), 1)
            if d.get("method") in ("auto", "yolo", "classic"):
                z["method"] = d["method"]
        except (ValueError, TypeError):
            return err("Invalid value")
        if z["has_image"] and ("capacity" in d or "method" in d):
            try:
                run_analysis(z)
            except Exception as e:
                save(zones)
                return err(e)
        z["rev"] += 1
        save(zones)
        return jsonify(view(z))


@app.delete("/api/zones/<int:zid>")
def api_delete(zid):
    with LOCK:
        zones = [z for z in load() if z["id"] != zid]
        save(zones)
    for p in (IMG / f"zone_{zid}.jpg", ANN / f"zone_{zid}.jpg"):
        p.unlink(missing_ok=True)
    return jsonify(ok=True)


@app.post("/api/zones/<int:zid>/image")
def api_image(zid):
    f = request.files.get("image")
    if not f:
        return err("No file uploaded (field name: image)")
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return err("That file is not a valid image")
    h, w = img.shape[:2]
    s = 1600 / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    with LOCK:
        zones = load()
        z = find(zones, zid)
        if not z:
            return err("No such zone", 404)
        cv2.imwrite(str(IMG / f"zone_{zid}.jpg"), img)
        z["has_image"] = True
        try:
            run_analysis(z)
        except Exception as e:
            z["result"] = None
            z["rev"] += 1
            save(zones)
            return jsonify(error=str(e), zone=view(z)), 400
        save(zones)
        return jsonify(view(z))


@app.post("/api/zones/<int:zid>/analyze")
def api_analyze(zid):
    with LOCK:
        zones = load()
        z = find(zones, zid)
        if not z:
            return err("No such zone", 404)
        if not z["has_image"]:
            return err("Upload an image for this zone first")
        try:
            run_analysis(z)
        except Exception as e:
            return err(e)
        save(zones)
        return jsonify(view(z))


@app.put("/api/zones/<int:zid>/slots")
def api_slots(zid):
    d = request.get_json(silent=True) or {}
    try:
        slots = [[[min(max(float(x), 0), 1), min(max(float(y), 0), 1)] for x, y in poly] for poly in d["slots"]]
        assert all(len(p) >= 3 for p in slots) and len(slots) <= 300
    except Exception:
        return err("slots must be a list of polygons (>=3 points, max 300 slots)")
    with LOCK:
        zones = load()
        z = find(zones, zid)
        if not z:
            return err("No such zone", 404)
        z["slots"] = slots
        z["result"] = None if not z["has_image"] else z["result"]
        warning = None
        if z["has_image"]:
            try:
                run_analysis(z)
            except Exception as e:
                z["result"], warning = None, str(e)
        z["rev"] += 1
        save(zones)
        return jsonify(zone=view(z), warning=warning)


def _file(folder, zid):
    p = folder / f"zone_{zid}.jpg"
    if not p.exists():
        return Response(status=404)
    return Response(p.read_bytes(), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/zones/<int:zid>/image")
def api_raw(zid):
    return _file(IMG, zid)


@app.get("/api/zones/<int:zid>/annotated")
def api_annotated(zid):
    return _file(ANN, zid)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
