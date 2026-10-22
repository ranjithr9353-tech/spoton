"""SpotON slot detector (your original Colab logic, refactored into a module)."""
import cv2
import numpy as np

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

OCCUPANCY_THRESHOLD = 0.33
REF_SIZE = (480, 240)  # (w, h) frame size the slot polygons were drawn for


def slot_mask(shape, polygon):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    return mask


def slot_feature_score(gray, edges, mask):
    area = max(cv2.countNonZero(mask), 1)
    edge_density = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask)) / area
    region_std = float(np.std(gray[mask == 255]))
    return edge_density + (region_std / 200.0)


def classify_slots(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.resize(img, REF_SIZE)  # any uploaded image is fitted to the polygon grid
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    results = {}
    for slot_id, polygon in SLOTS.items():
        mask = slot_mask(gray.shape, polygon)
        score = slot_feature_score(gray, edges, mask)
        confidence = min(score / (OCCUPANCY_THRESHOLD * 1.6), 1.0)
        results[slot_id] = {
            "label": "Occupied" if score >= OCCUPANCY_THRESHOLD else "Vacant",
            "score": round(score, 4),
            "confidence": round(confidence, 3),
        }
    return img, results


def draw_annotated_frame(img, results):
    annotated = img.copy()
    for slot_id, polygon in SLOTS.items():
        pts = np.array(polygon, dtype=np.int32)
        color = (0, 0, 255) if results[slot_id]["label"] == "Occupied" else (0, 255, 0)
        cv2.polylines(annotated, [pts], True, color, 1)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(annotated, str(slot_id), (cx - 4, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
    return annotated


def accuracy(results):
    correct = sum(1 for s, r in results.items() if r["label"] == GROUND_TRUTH[s])
    return round(100.0 * correct / len(results), 2)
