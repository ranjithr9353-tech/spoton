"""SpotON detector.
Real photo in -> vehicles found with YOLO (pretrained, COCO) -> each parking slot marked
occupied / vacant. Falls back to a basic OpenCV edge method if YOLO is not installed.
Slots are polygons in NORMALISED coordinates (0..1) so they survive image resizing.
"""
import importlib.util
import os

import cv2
import numpy as np

MAX_SIDE = 1280
VEHICLE_CLASSES = [2, 3, 5, 7]            # COCO: car, motorcycle, bus, truck
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.25"))
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8n.pt")   # auto-downloaded on first use
CLASSIC_THRESHOLD = 0.33

_yolo = {"model": None, "error": None, "tried": False}


def yolo_installed():
    return importlib.util.find_spec("ultralytics") is not None


def get_yolo():
    if not _yolo["tried"]:
        _yolo["tried"] = True
        try:
            from ultralytics import YOLO
            _yolo["model"] = YOLO(YOLO_MODEL)
        except Exception as e:  # not installed / no internet for the weights
            _yolo["error"] = f"{type(e).__name__}: {e}"
    return _yolo["model"]


def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img


def detect_vehicles(model, img):
    res = model.predict(img, conf=YOLO_CONF, imgsz=1024, classes=VEHICLE_CLASSES, verbose=False)[0]
    out = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append((x1, y1, x2, y2, float(b.conf[0])))
    return out


def _hull(poly):
    return cv2.convexHull(np.array(poly, np.float32)).reshape(-1, 2)


def assign_boxes(polys, boxes):
    """Give every vehicle box to at most one slot. Returns ({slot: conf}, [max_overlap per slot])."""
    hulls = [_hull(p) for p in polys]
    areas = [max(float(cv2.contourArea(h)), 1.0) for h in hulls]
    occupied, overlap = {}, [0.0] * len(polys)
    for x1, y1, x2, y2, conf in boxes:
        foot = (float((x1 + x2) / 2), float(y2 - 0.2 * (y2 - y1)))   # where the car meets the ground
        rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
        ratios = []
        for i, h in enumerate(hulls):
            inter, _ = cv2.intersectConvexConvex(h, rect)
            ratios.append(inter / areas[i])
            overlap[i] = max(overlap[i], ratios[-1])
        inside = [i for i, h in enumerate(hulls)
                  if cv2.pointPolygonTest(h.reshape(-1, 1, 2), foot, False) >= 0]
        best = max(inside, key=lambda i: ratios[i]) if inside else None
        if best is None and ratios:
            i = int(np.argmax(ratios))
            best = i if ratios[i] >= 0.35 else None
        if best is not None:
            occupied[best] = max(occupied.get(best, 0.0), conf)
    return occupied, overlap


def _classic_scores(img, polys):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    out = []
    for p in polys:
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillPoly(mask, [np.array(p, np.int32)], 255)
        n = cv2.countNonZero(mask)
        if n == 0:
            out.append(0.0)
            continue
        dens = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask)) / n
        out.append(dens + float(np.std(gray[mask == 255])) / 200.0)
    return out


def analyze(path, slots, method="auto", capacity=0):
    """slots: list of polygons in normalised coords. Returns dict incl. annotated BGR image."""
    img = read_image(path)
    h, w = img.shape[:2]
    polys = [[(x * w, y * h) for x, y in s] for s in slots]
    note, backend, boxes = None, "classic", []

    model = get_yolo() if method in ("auto", "yolo") else None
    if method in ("auto", "yolo") and model is None:
        if method == "yolo":
            raise RuntimeError("YOLO is not available: " + str(_yolo["error"]))
        note = "YOLO unavailable, used basic OpenCV method (" + str(_yolo["error"]) + ")"

    result_slots = []
    if model is not None:
        backend = "yolo"
        boxes = detect_vehicles(model, img)
        if polys:
            occ, overlap = assign_boxes(polys, boxes)
            for i in range(len(polys)):
                if i in occ:
                    result_slots.append({"label": f"S{i+1:02d}", "status": "occupied", "confidence": round(occ[i], 3)})
                else:
                    result_slots.append({"label": f"S{i+1:02d}", "status": "vacant", "confidence": round(1 - min(overlap[i], 1), 3)})
            occupied = len(occ)
            total = len(polys)
        else:
            occupied = len(boxes)
            total = max(int(capacity or 0), occupied)
    else:
        if not polys:
            raise ValueError("Define parking slots first (basic method needs slot polygons), or install ultralytics for YOLO.")
        for i, sc in enumerate(_classic_scores(img, polys)):
            conf = min(sc / (CLASSIC_THRESHOLD * 1.6), 1.0)
            st = "occupied" if sc >= CLASSIC_THRESHOLD else "vacant"
            result_slots.append({"label": f"S{i+1:02d}", "status": st, "confidence": round(conf if st == "occupied" else 1 - conf, 3)})
        occupied = sum(s["status"] == "occupied" for s in result_slots)
        total = len(polys)

    # annotated picture
    ann, t = img.copy(), max(1, round(w / 700))
    for i, p in enumerate(polys):
        col = (60, 60, 230) if result_slots[i]["status"] == "occupied" else (80, 200, 60)
        pts = np.array(p, np.int32)
        cv2.polylines(ann, [pts], True, col, t, cv2.LINE_AA)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(ann, str(i + 1), (cx - 6, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35 + 0.15 * t, col, t, cv2.LINE_AA)
    for x1, y1, x2, y2, c in boxes:
        cv2.rectangle(ann, (int(x1), int(y1)), (int(x2), int(y2)), (0, 215, 255), t)
        cv2.putText(ann, f"{c:.2f}", (int(x1), max(int(y1) - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 + 0.1 * t, (0, 215, 255), t, cv2.LINE_AA)
    return {"annotated": ann, "backend": backend, "vehicles": len(boxes), "slots": result_slots,
            "occupied": occupied, "vacant": total - occupied, "total": total, "note": note}
