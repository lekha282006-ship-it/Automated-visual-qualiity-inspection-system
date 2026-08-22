import cv2
import numpy as np
from src.inspector import PartInspector


def draw_washer(img, center=(200, 200), outer_r=150, inner_r=50, color=180, bg_color=30):
    cv2.circle(img, center, outer_r, color, -1)
    cv2.circle(img, center, inner_r, bg_color, -1)
    return img


def test_surface_scratch_detection():
    w, h = 400, 400
    bg = 30
    ref = np.full((h, w), bg, dtype=np.uint8)
    ref = draw_washer(ref, outer_r=120, inner_r=40, color=180, bg_color=bg)

    test = ref.copy()
    cv2.line(test, (140, 200), (260, 200), 80, 4)

    insp = PartInspector(reference=ref, size_tol=5.0, defect_thresh=10)
    metrics, ann = insp.inspect(test)
    assert isinstance(metrics, dict)
    assert metrics.get("defect_count", 0) >= 1
    types = {d.get("type") for d in metrics.get("defects", [])}
    assert any(t in ("Surface Scratch", "Edge Burr") for t in types)


def test_surface_chip_detection():
    w, h = 400, 400
    bg = 30
    ref = np.full((h, w), bg, dtype=np.uint8)
    ref = draw_washer(ref, outer_r=120, inner_r=40, color=180, bg_color=bg)

    test = ref.copy()
    cv2.circle(test, (110, 180), 15, bg, -1)

    insp = PartInspector(reference=ref, size_tol=5.0, defect_thresh=10)
    metrics, ann = insp.inspect(test)
    types = {d.get("type") for d in metrics.get("defects", [])}
    assert any(t == "Surface Chip" for t in types)


def test_undersized_and_oversized_detection():
    w, h = 400, 400
    bg = 30
    ref = np.full((h, w), bg, dtype=np.uint8)
    ref = draw_washer(ref, outer_r=120, inner_r=40, color=180, bg_color=bg)

    # undersized
    undersized = np.full((h, w), bg, dtype=np.uint8)
    undersized = draw_washer(undersized, outer_r=108, inner_r=40, color=180, bg_color=bg)
    insp = PartInspector(reference=ref, size_tol=5.0, defect_thresh=1000)
    metrics_u, _ = insp.inspect(undersized)
    assert "Undersized" in metrics_u.get("reason", "") or metrics_u.get("status") == "FAIL"

    # oversized
    oversized = np.full((h, w), bg, dtype=np.uint8)
    oversized = draw_washer(oversized, outer_r=132, inner_r=40, color=180, bg_color=bg)
    metrics_o, _ = insp.inspect(oversized)
    assert "Oversized" in metrics_o.get("reason", "") or metrics_o.get("status") == "FAIL"


def test_hole_misalignment_detection():
    w, h = 400, 400
    bg = 30
    ref = np.full((h, w), bg, dtype=np.uint8)
    ref = draw_washer(ref, outer_r=120, inner_r=40, color=180, bg_color=bg)

    mis = np.full((h, w), bg, dtype=np.uint8)
    mis = draw_washer(mis, outer_r=120, inner_r=40, color=180, bg_color=bg)
    # shift inner hole
    cv2.circle(mis, (250, 220), 40, bg, -1)

    insp = PartInspector(reference=ref, size_tol=5.0, defect_thresh=1000)
    metrics, _ = insp.inspect(mis)
    assert "Hole Misalignment" in metrics.get("reason", "") or metrics.get("status") == "FAIL"


def test_edge_burr_detection():
    w, h = 400, 400
    bg = 30
    ref = np.full((h, w), bg, dtype=np.uint8)
    ref = draw_washer(ref, outer_r=120, inner_r=40, color=180, bg_color=bg)

    burr = ref.copy()
    cv2.circle(burr, (360, 200), 12, 180, -1)

    insp = PartInspector(reference=ref, size_tol=5.0, defect_thresh=10)
    metrics, _ = insp.inspect(burr)
    types = {d.get("type") for d in metrics.get("defects", [])}
    assert any(t == "Edge Burr" or t == "Surface Chip" for t in types)
