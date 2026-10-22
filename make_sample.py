"""Generates data/Picture1.png: synthetic multi-zone parking camera frame."""
import cv2
import numpy as np
from pathlib import Path
from detector import SLOTS, GROUND_TRUTH

W, H = 480, 240
rng = np.random.default_rng(7)
img = np.full((H, W, 3), (85, 85, 90), np.uint8)
img[:70] = (140, 165, 130)  # Sidewalk/curb
img[155:] = (75, 75, 80)    # Driving lane
img = cv2.add(img, rng.integers(0, 5, img.shape, dtype=np.uint8))
img = cv2.GaussianBlur(img, (3, 3), 0)

CAR_COLORS = [
    (35, 35, 190), (190, 110, 35), (200, 200, 200),
    (25, 25, 25), (35, 150, 190), (55, 170, 55)
]

def inset(poly, k=0.18):
    p = np.array(poly, np.float32)
    c = p.mean(axis=0)
    return (c + (p - c) * (1 - k)).astype(np.int32)

for sid, poly in SLOTS.items():
    cv2.polylines(img, [np.array(poly, np.int32)], True, (120, 120, 125), 1, cv2.LINE_AA)
    if GROUND_TRUTH[sid] == "Occupied":
        body = inset(poly, 0.12)
        col = CAR_COLORS[(sid - 1) % len(CAR_COLORS)]
        cv2.fillPoly(img, [body], col)
        cv2.polylines(img, [body], True, (20, 20, 20), 1, cv2.LINE_AA)
        roof = inset(poly, 0.40)
        cv2.fillPoly(img, [roof], tuple(int(c * 0.6) for c in col))
        cv2.polylines(img, [roof], True, (230, 230, 240), 1, cv2.LINE_AA)

Path("data").mkdir(exist_ok=True)
cv2.imwrite("data/Picture1.png", img)
print("Created initial sample image at data/Picture1.png")
