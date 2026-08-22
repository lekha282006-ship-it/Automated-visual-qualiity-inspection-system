import cv2
import numpy as np
import json
from typing import List, Tuple, Dict, Any, Optional


def _objp_for_checkerboard(checkerboard: Tuple[int, int], square_size_mm: float) -> np.ndarray:
    nx, ny = checkerboard
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * float(square_size_mm)
    return objp


def calibrate_from_images(
    image_paths: List[str],
    checkerboard: Tuple[int, int] = (9, 6),
    square_size_mm: float = 5.0,
    flags: int = 0,
) -> Dict[str, Any]:
    """Calibrate camera from multiple checkerboard images.

    Returns a dictionary containing camera_matrix, dist_coeffs, rvecs, tvecs,
    RMS reprojection error, and estimated mm_per_px scale.
    """
    objp = _objp_for_checkerboard(checkerboard, square_size_mm)

    objpoints = []  # 3d points in real world space (mm)
    imgpoints = []  # 2d points in image plane (px)

    pixel_scales = []

    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, checkerboard, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            # try alternative flags if not found
            found, corners = cv2.findChessboardCorners(gray, checkerboard, flags=0)
        if not found:
            continue
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
        imgpoints.append(corners2.reshape(-1, 2))
        objpoints.append(objp)

        # estimate pixel spacing between adjacent corners (mean)
        pts = corners2.reshape(-1, 2)
        # horizontal distances (where applicable)
        nx, ny = checkerboard
        dists = []
        for y in range(ny):
            for x in range(nx - 1):
                i = y * nx + x
                a = pts[i]
                b = pts[i + 1]
                dists.append(np.linalg.norm(a - b))
        for x in range(nx):
            for y in range(ny - 1):
                i = y * nx + x
                a = pts[i]
                b = pts[i + nx]
                dists.append(np.linalg.norm(a - b))
        if dists:
            mean_px = float(np.mean(dists))
            pixel_scales.append(square_size_mm / mean_px)

    if not objpoints or not imgpoints:
        raise ValueError("No checkerboard corners found in the provided images.")

    # image size from the last read image
    h, w = gray.shape[:2]
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

    mm_per_px = float(np.median(pixel_scales)) if pixel_scales else None

    return {
        "rms": float(rms),
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "checkerboard": checkerboard,
        "square_size_mm": float(square_size_mm),
        "mm_per_px": mm_per_px,
        "image_size": (w, h),
    }


def undistort_image(img: np.ndarray, calib: Dict[str, Any]) -> np.ndarray:
    if calib is None or "camera_matrix" not in calib:
        raise ValueError("Invalid calibration data")
    mtx = calib["camera_matrix"]
    dist = calib["dist_coeffs"]
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    und = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w2, h2 = roi
    if w2 > 0 and h2 > 0:
        und = und[y : y + h2, x : x + w2]
    return und


def save_calibration(path: str, calib: Dict[str, Any]):
    # store numeric arrays using np.savez
    np.savez(path, camera_matrix=calib["camera_matrix"], dist_coeffs=calib["dist_coeffs"], rms=calib.get("rms", 0.0), checkerboard=json.dumps(calib.get("checkerboard")), square_size_mm=calib.get("square_size_mm", 1.0), mm_per_px=calib.get("mm_per_px", None), image_size=json.dumps(calib.get("image_size")))


def load_calibration(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    out = {
        "camera_matrix": data["camera_matrix"],
        "dist_coeffs": data["dist_coeffs"],
        "rms": float(data.get("rms", 0.0)),
        "checkerboard": tuple(json.loads(data.get("checkerboard", b"[]"))),
        "square_size_mm": float(data.get("square_size_mm", 1.0)),
        "mm_per_px": float(data.get("mm_per_px", 0.0)) if data.get("mm_per_px", None) is not None else None,
        "image_size": tuple(json.loads(data.get("image_size", b"[]"))),
    }
    return out


def pixel_to_metric(x: float, y: float, calib: Dict[str, Any]) -> Tuple[float, float]:
    """Convert pixel coordinates (x,y) to millimeters using mm_per_px from calibration."""
    mmpp = calib.get("mm_per_px")
    if mmpp is None or mmpp == 0:
        raise ValueError("Calibration does not contain mm_per_px. Run calibrate with square_size_mm specified and sufficient images.")
    return float(x) * mmpp, float(y) * mmpp


def calibrate_with_matlab(image_paths: List[str], checkerboard: Tuple[int, int], square_size_mm: float) -> Dict[str, Any]:
    """Optional MATLAB engine integration. If matlab.engine is available, calls MATLAB camera calibration routines.

    If MATLAB engine is not installed, raises ImportError.
    """
    try:
        import matlab.engine as mengine
    except Exception as e:
        raise ImportError("MATLAB engine for Python is not available: " + str(e))

    eng = mengine.start_matlab()
    # For now, call a placeholder MATLAB function; a production integration would
    # transfer images or paths and call MATLAB's cameraCalibrator / estimateCameraParameters.
    raise NotImplementedError("MATLAB integration requires a MATLAB-side helper; implement as needed.")
