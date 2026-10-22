"""
SpotON Detector:
Integrates ResNet-50 object detection (Faster R-CNN / RetinaNet architecture as explored
in Kaggle's shivamb/objects-bounding-boxes-using-resnet50-imageai) with polygon intersection
to compute real-time parking slot occupancy. Includes robust fallback to edge/texture heuristics
if deep learning models cannot be loaded offline.
"""
import os
import cv2
import numpy as np
from shapely.geometry import Polygon, box

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
OCCUPANCY_THRESHOLD = 0.33
VEHICLE_CLASSES = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}

_model = None
_device = None
_torch_available = False

def init_model():
    global _model, _device, _torch_available
    try:
        import torch
        import torchvision
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        _model = fasterrcnn_resnet50_fpn(weights=weights)
        _model.to(_device)
        _model.eval()
        _torch_available = True
        print(f"ResNet-50 Faster-RCNN initialized on {_device}")
    except Exception as e:
        print(f"Deep learning init notice: {e}. Fallback to OpenCV edge & variance heuristic enabled.")
        _torch_available = False

def compute_overlap(slot_poly_coords, bbox):
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
    results = {}
    detected_vehicles = []

    if _torch_available and _model is not None:
        import torch
        from torchvision import transforms as T

        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor_img = T.ToTensor()(img_rgb).to(_device)

        with torch.no_grad():
            preds = _model([tensor_img])[0]

        boxes = preds["boxes"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()

        for b, s, l in zip(boxes, scores, labels):
            if l in VEHICLE_CLASSES and s >= 0.35:
                detected_vehicles.append({"box": b, "score": float(s), "label": VEHICLE_CLASSES[l]})

        for slot_id, poly in SLOTS.items():
            best_overlap = 0.0
            best_conf = 0.0
            for v in detected_vehicles:
                ov = compute_overlap(poly, v["box"])
                if ov > best_overlap:
                    best_overlap = ov
                    best_conf = v["score"]

            is_occupied = best_overlap >= 0.18
            results[slot_id] = {
                "label": "Occupied" if is_occupied else "Vacant",
                "score": round(best_overlap, 3),
                "confidence": round(best_conf if is_occupied else (1.0 - best_overlap), 3),
                "method": "ResNet50"
            }
    else:
        # Robust Computer Vision Fallback (Edge Density + Variance)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        for slot_id, polygon in SLOTS.items():
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
            area = max(cv2.countNonZero(mask), 1)
            edge_density = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask)) / area
            region_std = float(np.std(gray[mask == 255]))
            score = edge_density + (region_std / 200.0)
            confidence = min(score / (OCCUPANCY_THRESHOLD * 1.6), 1.0)
            results[slot_id] = {
                "label": "Occupied" if score >= OCCUPANCY_THRESHOLD else "Vacant",
                "score": round(score, 4),
                "confidence": round(confidence, 3),
                "method": "OpenCV-Heuristic"
            }

    return img_resized, results, detected_vehicles

def draw_annotated_frame(img, results, detected_vehicles=None):
    annotated = img.copy()
    if detected_vehicles:
        for v in detected_vehicles:
            x1, y1, x2, y2 = [int(c) for c in v["box"]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 1)
            cv2.putText(annotated, f"{v['label']} {v['score']:.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 180, 0), 1, cv2.LINE_AA)

    for slot_id, polygon in SLOTS.items():
        pts = np.array(polygon, dtype=np.int32)
        color = (0, 0, 255) if results[slot_id]["label"] == "Occupied" else (0, 255, 0)
        cv2.polylines(annotated, [pts], True, color, 2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(annotated, f"S{slot_id}", (cx - 8, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated

def accuracy(results):
    correct = sum(1 for s, r in results.items() if r["label"] == GROUND_TRUTH[s])
    return round(100.0 * correct / len(results), 2)
