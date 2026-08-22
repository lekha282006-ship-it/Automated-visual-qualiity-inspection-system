# matlab_integration_test.py
import os
import cv2
import numpy as np
from src.matlab_bridge import start_matlab, align_images_matlab, subpixel_boundary


def draw_washer(img, center=(200,200), outer_r=120, inner_r=40, color=180, bg_color=30):
    cv2.circle(img, center, outer_r, color, -1)
    cv2.circle(img, center, inner_r, bg_color, -1)
    return img


def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def main():
    w,h = 400,400
    bg = 30
    ref = np.full((h, w), bg, dtype=np.uint8)
    ref = draw_washer(ref)

    # create test: rotate + translate + add a small chip
    test = rotate_image(ref, 12)  # rotate part in image
    # paste a chip (a small circular hole) at an offset location
    cv2.circle(test, (210, 170), 12, bg, -1)

    print("Starting MATLAB engine (this may open MATLAB)...")
    try:
        eng = start_matlab()
    except Exception as e:
        print("Failed to start MATLAB engine:", e)
        print("Ensure MATLAB and matlab.engine are installed and configured.")
        return

    # Add repo/matlab to MATLAB path so alignImages/subpixelBoundary are visible
    matlab_folder = os.path.join(os.getcwd(), "matlab")
    print("Adding MATLAB path:", matlab_folder)
    try:
        eng.addpath(matlab_folder, nargout=0)
    except Exception as e:
        print("Warning: could not add matlab path:", e)

    print("Calling MATLAB alignImages...")
    try:
        aligned, H = align_images_matlab(ref, test, engine=eng)
    except Exception as e:
        print("align_images_matlab failed:", e)
        return

    print("Homography H (shape):", None if H is None else np.array(H).shape)
    try:
        print(np.array(H))
    except Exception:
        print("Could not convert H to numpy")

    # show basic alignment stats
    diff = cv2.absdiff(ref, aligned)
    print("Alignment diff max/min:", int(diff.max()), int(diff.min()))
    nonzero = int(np.count_nonzero(diff))
    print("Alignment diff nonzero pixels:", nonzero)

    # build binary defect mask (where aligned differs from ref)
    _, th = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    print("Calling MATLAB subpixelBoundary on threshold mask...")
    try:
        contours = subpixel_boundary(th, scale=4)
    except Exception as e:
        print("subpixel_boundary failed:", e)
        return

    print("Contours (count):", len(contours))
    if len(contours) > 0:
        print("First contour sample points (first 5):", contours[0][:5])

    print("MATLAB integration test complete.")


if __name__ == "__main__":
    main()
