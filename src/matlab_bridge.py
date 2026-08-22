"""MATLAB engine bridge with safe fallbacks.

Provides:
- has_matlab(): whether matlab.engine can be used.
- start_matlab(): start and return engine (cached).
- align_images_matlab(ref_gray, test_gray, engine=None): if MATLAB available, call MATLAB `alignImages` helper; otherwise fallback to ORB homography-based alignment.
- subpixel_boundary(binary_img, scale=4): upscale+findContours then scale back to float coordinates for sub-pixel contours.

This module intentionally keeps MATLAB calls optional and returns numpy arrays when MATLAB is unavailable.
"""

from typing import Tuple, List
import numpy as np
import cv2

_matlab_engine = None


def has_matlab() -> bool:
    global _matlab_engine
    try:
        import matlab  # type: ignore
        import matlab.engine  # type: ignore
        return True
    except Exception:
        return False


def start_matlab():
    """Start MATLAB engine and cache it. Returns engine or raises if unavailable."""
    global _matlab_engine
    if _matlab_engine is not None:
        return _matlab_engine
    try:
        import matlab.engine as meng
        _matlab_engine = meng.start_matlab()
        return _matlab_engine
    except Exception as e:
        raise RuntimeError("MATLAB engine not available") from e


def align_images_matlab(ref_gray: np.ndarray, test_gray: np.ndarray, engine=None) -> Tuple[np.ndarray, np.ndarray]:
    """Try to align using MATLAB helper `alignImages` if engine provided, else fallback to ORB homography.

    Returns (aligned_test_gray, H) where H is 3x3 homography from test->ref (numpy float32).
    """
    # attempt to use MATLAB engine if available
    eng = engine
    if eng is None:
        try:
            eng = start_matlab()
        except Exception:
            eng = None

    if eng is not None:
        try:
            import matlab
            # convert numpy 2D arrays to matlab.uint8 matrices
            ref_mat = matlab.uint8(ref_gray.tolist())
            test_mat = matlab.uint8(test_gray.tolist())
            # call MATLAB function in matlab/ folder (ensure it's on MATLAB path)
            res = eng.alignImages(ref_mat, test_mat, nargout=2)
            aligned_mat = res[0]
            H_mat = res[1]
            # convert aligned_mat to numpy array
            aligned = np.array(aligned_mat, dtype=np.uint8)
            H = np.array(H_mat, dtype=np.float64)
            return aligned, H
        except Exception:
            # fallback to python implementation below
            pass

    # Fallback ORB alignment (re-implementation so this module is independent)
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(test_gray, None)
    if des1 is None or des2 is None:
        return test_gray, np.eye(3, dtype=np.float32)
    # FLANN LSH params
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    try:
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
    except Exception:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = [[m] for m in bf.match(des1, des2)]

    good = []
    for m in matches:
        if len(m) == 2:
            if m[0].distance < 0.75 * m[1].distance:
                good.append(m[0])
        elif len(m) == 1:
            good.append(m[0])
    if len(good) < 6:
        return test_gray, np.eye(3, dtype=np.float32)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if M is None:
        return test_gray, np.eye(3, dtype=np.float32)
    h, w = ref_gray.shape
    aligned = cv2.warpPerspective(test_gray, M, (w, h))
    return aligned, M


def subpixel_boundary(binary_img: np.ndarray, scale: int = 4) -> List[np.ndarray]:
    """Return contours with sub-pixel coordinates as numpy arrays of shape (N,1,2) with float coords.

    Upscales the binary image, finds contours, and scales them back to original coordinates with fractional values.
    """
    # Try MATLAB implementation if available
    try:
        eng = start_matlab()
        import matlab
        if binary_img.dtype != np.uint8:
            b = (binary_img > 0).astype(np.uint8) * 255
        else:
            b = binary_img
        mat_img = matlab.uint8(b.tolist())
        cell = eng.subpixelBoundary(mat_img, int(scale), nargout=1)
        # cell is a MATLAB cell array of Nx2 arrays; convert to list of numpy arrays
        contours = []
        try:
            for k in range(1, len(cell) + 1):
                arr = np.array(cell[k - 1])
                contours.append(arr.astype(np.float32))
            return contours
        except Exception:
            pass
    except Exception:
        pass

    # Python fallback
    if binary_img.dtype != np.uint8:
        b = (binary_img > 0).astype(np.uint8) * 255
    else:
        b = binary_img
    big = cv2.resize(b, (b.shape[1] * scale, b.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    contours_big, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = []
    for c in contours_big:
        c_float = c.astype(np.float32) / float(scale)
        # convert to Nx2 float array (x,y)
        pts = c_float.reshape(-1, 2)[:, ::-1].copy()  # cv contours are (x,y)?? ensure col,row->x,y
        contours.append(pts)
    return contours
