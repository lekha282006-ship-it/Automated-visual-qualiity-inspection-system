import cv2
import numpy as np
from src.inspector import PartInspector


def draw_washer(img, center=(200, 200), outer_r=120, inner_r=40, color=180, bg_color=30):
    cv2.circle(img, center, outer_r, color, -1)
    cv2.circle(img, center, inner_r, bg_color, -1)
    return img


def add_gaussian_noise(img, sigma=5):
    noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
    out = img.astype(np.int16) + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def test_noisy_chip_detection():
    w, h = 400, 400
    bg = 30
    ref = np.full((h, w), bg, dtype=np.uint8)
    ref = draw_washer(ref)

    test = ref.copy()
    cv2.circle(test, (200, 160), 15, bg, -1)
    test = add_gaussian_noise(test, sigma=6)

    insp = PartInspector(reference=ref, size_tol=5.0, defect_thresh=10)
    metrics, _ = insp.inspect(test)
    assert metrics.get("defect_count", 0) >= 1
    types = {d.get("type") for d in metrics.get("defects", [])}
    assert any(t == "Surface Chip" for t in types)


def test_partial_occlusion_chip_detection():
    w, h = 400, 400
    bg = 30
    ref = np.full((h, w), bg, dtype=np.uint8)
    ref = draw_washer(ref)

    test = ref.copy()
    # place chip near edge but still inside
    cv2.circle(test, (260, 200), 18, bg, -1)

    insp = PartInspector(reference=ref, size_tol=5.0, defect_thresh=10)
    metrics, _ = insp.inspect(test)
    assert metrics.get("defect_count", 0) >= 1
    types = {d.get("type") for d in metrics.get("defects", [])}
    assert any(t in ("Surface Chip", "Edge Burr") for t in types)


def test_low_contrast_scratch_detection():
    w, h = 400, 400
    bg = 30
    ref = np.full((h, w), bg, dtype=np.uint8)
    ref = draw_washer(ref)

    test = ref.copy()
    # low contrast scratch: slightly lighter than background
    cv2.line(test, (140, 200), (260, 200), 50, 3)

    insp = PartInspector(reference=ref, size_tol=5.0, defect_thresh=10)
    metrics, _ = insp.inspect(test)
    assert metrics.get("defect_count", 0) >= 1
    types = {d.get("type") for d in metrics.get("defects", [])}
    assert any(t in ("Surface Scratch", "Edge Burr") for t in types)
