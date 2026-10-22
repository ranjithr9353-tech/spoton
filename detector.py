"""
SpotON Detector: Full-frame vehicle detection using YOLO + OpenCV
Detects vehicles (car, truck, bus, motorcycle) and evaluates slot occupancy
via polygon intersection without requiring manual patch cropping.
"""
import os
import cv2
import numpy as np
from shapely.geometry import Polygon, box
from ultralytics import YOLO

# 12 Slot Polygons defined on a 480x240 reference frame
SLOTS = {
    1: [(65.0, 101.7), (93.3, 98.3), (88.3, 140.0), (58.3, 126.7)],
    2: [(93.3, 98.3), (123.3, 96.7), (120.0, 138.3), (88.3, 140.0)],
    3: [(123.3, 96.7), (153.3, 95.0), (151.7, 136.7), (120.0, 138.3)],
    4: [(153.3, 95.0), (183.3, 94.3), (181.7, 135.0), (151.7, 136.7)],
    5: [(183.3, 94.3), (213.3, 93.3), (211.7, 133.3), (181.7, 135.0)],
    6: [(213.3, 93.3), (243.3, 103.3), (240.0, 131.7), (211.7, 133.3)],
    7: [(243.3, 103.3), (273.3, 105.0), (271.7, 133.3), (240.0, 131.7)],
    8: [(273.3, 105.0), (303.3, 108.3), (301.7, 135.0), (271.7, 133.3)],
    9: [(303.3, 108.3), (333.3, 111.7), (331.7, 136.7), (301.7, 135.0)],
    10: [(333.3, 111.7), (363.3, 115.0), (361.7, 138.3), (331.7, 136.7)],
    11: [(363.3, 115.0), (393.3, 118.3), (391.7, 140.0), (361.7, 138.3)],
    12: [(393.3, 118.3), (423.3, 121.7), (421.7, 141.7), (391.7, 140.0)],
}

GROUND_TRUTH = {i: "Occupied" for i in range(1, 7)}
GROUND_TRUTH.update({i: "Vacant" for i in range(7, 13)})

REF_SIZE = (480, 240)
# COCO class IDs: 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# Initialize YOLO model (nano version for fast CPU inference)
model = YOLO("yolov8n.pt")

def compute_overlap(slot_poly_coords, bbox):
    """Calculates overlap ratio between YOLO bounding box and slot polygon."""
    s_poly = Polygon(slot_poly_coords)
    b_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
    if not s_poly.is_valid or not b_poly.is_valid:
        return 0.0
    intersection = s_poly.intersection(b_poly).area
    return intersection / max(s_poly.area, 1e-5)

def classify_slots(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_resized = cv2.resize(img, REF_SIZE)
    
    # Run YOLO object detection
    predictions = model(img_resized, classes=VEHICLE_CLASSES, conf=0.25, verbose=False)[0]

    detected_boxes = []
    for det in predictions.boxes:
        xyxy = det.xyxy[0].cpu().numpy()
        conf = float(det.conf[0].cpu().numpy())
        cls_id = int(det.cls[0].cpu().numpy())
        label_name = model.names[cls_id]

        detected_boxes.append({
            "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
            "label": label_name,
            "score": round(conf, 2)
        })

    results = {}
    for slot_id, poly in SLOTS.items():
        max_overlap = 0.0
        best_conf = 0.0

        for d in detected_boxes:
            ov = compute_overlap(poly, d["box"])
            if ov > max_overlap:
                max_overlap = ov
                best_conf = d["score"]

        # If a vehicle overlaps >= 18% of slot polygon area, mark Occupied
        is_occupied = max_overlap >= 0.18
        results[slot_id] = {
            "label": "Occupied" if is_occupied else "Vacant",
            "score": round(max_overlap, 3),
            "confidence": round(best_conf if is_occupied else (1.0 - max_overlap), 3),
            "method": "YOLOv8"
        }

    return img_resized, results, detected_boxes

def draw_annotated_frame(img, results, detected_boxes=None):
    annotated = img.copy()

    # 1. Draw YOLO vehicle bounding boxes using OpenCV
    if detected_boxes:
        for b in detected_boxes:
            x1, y1, x2, y2 = b["box"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(annotated, f"{b['label']} {b['score']}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 165, 0), 1, cv2.LINE_AA)

    # 2. Draw Slot Polygons with translucent overlay using OpenCV
    overlay = annotated.copy()
    for slot_id, polygon in SLOTS.items():
        pts = np.array(polygon, dtype=np.int32)
        is_occ = results[slot_id]["label"] == "Occupied"
        color = (0, 0, 220) if is_occ else (0, 200, 0)
        
        cv2.fillPoly(overlay, [pts], (0, 0, 160) if is_occ else (0, 120, 0))
        cv2.polylines(annotated, [pts], True, color, 2)

        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(annotated, f"S{slot_id}", (cx - 8, cy + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
    return annotated

def accuracy(results):
    correct = sum(1 for s, r in results.items() if r["label"] == GROUND_TRUTH[s])
    return round(100.0 * correct / len(results), 2)
