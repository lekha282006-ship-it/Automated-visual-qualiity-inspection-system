import cv2
import numpy as np
from src.inspector import PartInspector

def run_quick_test():
    print("[1/3] Initializing PartInspector engine...")
    inspector = PartInspector(target_area=50000, area_tolerance=0.05)

    print("[2/3] Generating synthetic Golden Reference & Test Sample...")
    # Generate Golden Reference
    ref_img = np.ones((400, 400, 3), dtype=np.uint8) * 220
    cv2.circle(ref_img, (200, 200), 126, (50, 50, 50), -1)
    
    # Generate Defective Test Sample (with a simulated scratch)
    test_img = ref_img.copy()
    cv2.line(test_img, (150, 150), (220, 220), (220, 220, 220), 4)

    # Set reference image inside inspector
    inspector.reference_image = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    print("[3/3] Running inspection pipeline...")
    try:
        results = inspector.inspect(test_img)
        # `inspect` may return (metrics, annotated_img) or only metrics.
        if isinstance(results, tuple):
            results, _ = results
        
        print("\n" + "="*40)
        print("TEST PASSED: INSPECTION EXECUTED CLEANLY")
        print("="*40)
        print(f"Status      : {results.get('status', 'N/A')}")
        print(f"Reason      : {results.get('reason', 'N/A')}")
        print(f"Area Metric : {results.get('area', 'N/A')} px")
        print("="*40)
    except Exception as e:
        print("\n" + "="*40)
        print("TEST FAILED: RUNTIME ERROR DETECTED")
        print("="*40)
        print(f"Error: {e}")

if __name__ == "__main__":
    run_quick_test()