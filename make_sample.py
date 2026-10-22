"""Generates data/Picture1.png: a synthetic street-parking frame that matches the
slot polygons in detector.py (slots 1-6 have cars, 7-12 are empty).
Run:  python make_sample.py
"""
import cv2
import numpy as np
from pathlib import Path
from detector import SLOTS, GROUND_TRUTH

W, H = 480, 240
rng = np.random.default_rng(7)
img = np.full((H, W, 3), (92, 92, 96), np.uint8)
img[:70] = (150, 170, 140)                       # grass / sidewalk strip at the top
img[150:] = (80, 80, 84)                         # road
img = cv2.add(img, rng.integers(0, 6, img.shape, dtype=np.uint8))
img = cv2.GaussianBlur(img, (3, 3), 0)

CAR_COLORS = [(40, 40, 200), (200, 120, 40), (210, 210, 210), (30, 30, 30), (40, 160, 200), (60, 180, 60)]

def inset(poly, k=0.18):
    p = np.array(poly, np.float32); c = p.mean(axis=0)
    return (c + (p - c) * (1 - k)).astype(np.int32)

for sid, poly in SLOTS.items():
    cv2.polylines(img, [np.array(poly, np.int32)], True, (112, 112, 116), 1, cv2.LINE_AA)
    if GROUND_TRUTH[sid] == "Occupied":
        body = inset(poly, 0.12)
        col = CAR_COLORS[(sid - 1) % len(CAR_COLORS)]
        cv2.fillPoly(img, [body], col)
        cv2.polylines(img, [body], True, (15, 15, 15), 1, cv2.LINE_AA)
        roof = inset(poly, 0.40)
        cv2.fillPoly(img, [roof], tuple(int(c * 0.6) for c in col))
        cv2.polylines(img, [roof], True, (230, 230, 240), 1, cv2.LINE_AA)
        cx, cy = np.array(poly, np.float32).mean(axis=0)
        cv2.line(img, (int(cx - 8), int(cy)), (int(cx + 8), int(cy)), (240, 240, 240), 1)
        cv2.line(img, (int(cx), int(cy - 12)), (int(cx), int(cy + 12)), (20, 20, 20), 1)

Path("data").mkdir(exist_ok=True)
cv2.imwrite("data/Picture1.png", img)
print("wrote data/Picture1.png", img.shape)
