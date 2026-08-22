import numpy as np
import cv2
from src.inspector import PartInspector


def make_reference():
    img = np.ones((400, 400, 3), dtype=np.uint8) * 220
    cv2.circle(img, (200, 200), 120, (50, 50, 50), -1)
    # add a mounting hole area to be used as ROI
    cv2.circle(img, (280, 200), 12, (220, 220, 220), -1)
    return img


def make_test_with_scratch():
    ref = make_reference()
    test = ref.copy()
    # draw a scratch (thin long)
    cv2.line(test, (150, 150), (230, 230), (220, 220, 220), 4)
    # rotate and translate
    M = cv2.getRotationMatrix2D((200, 200), 10, 1.0)
    test = cv2.warpAffine(test, M, (400, 400))
    return test


def test_inspector_detects_defect():
    ref = make_reference()
    test = make_test_with_scratch()

    insp = PartInspector(reference=ref, size_tol=5.0, defect_thresh=50)
    metrics, ann = insp.inspect(test)
    assert isinstance(metrics, dict)
    assert metrics["defect_count"] >= 1
    assert metrics["status"] in ("FAIL", "PASS", "WARN")
