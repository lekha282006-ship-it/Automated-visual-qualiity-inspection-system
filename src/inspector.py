import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import math


class PartInspector:
    """Advanced PartInspector with alignment, ROI masks, CLAHE, and defect classification.

    Notes:
    - Uses ORB feature matching + homography for alignment (rotation/translation invariance).
    - Supports multiple ROIs with independent tolerance settings.
    - Provides defect segmentation and coarse classification.
    """

    def __init__(
        self,
        reference: Optional[Any] = None,
        size_tol: Optional[float] = None,
        defect_thresh: Optional[float] = None,
        # Classification tunables
        scratch_aspect_thresh: float = 4.0,
        chip_solidity_thresh: float = 0.70,
        chip_circularity_thresh: float = 0.55,
        chip_area_frac_max: float = 0.02,
        chip_area_frac_max_alt: float = 0.03,
        burr_solidity_thresh: float = 0.65,
        use_matlab: bool = False,
        matlab_engine: Optional[Any] = None,
        mm_per_px: Optional[float] = None,
    ):
        # thresholds and tolerances
        self.defect_thresh = defect_thresh if defect_thresh is not None else 100.0
        tol = size_tol if size_tol is not None else 5.0
        self.area_tolerance = tol if tol <= 1.0 else (tol / 100.0)

        # Reference image (grayscale)
        self.reference_image: Optional[np.ndarray] = None
        self.ref_outer_area: Optional[float] = None
        # optional MATLAB engine hook
        self.use_matlab = use_matlab
        self.matlab_engine = matlab_engine
        self.matlab_available = False
        if self.use_matlab and self.matlab_engine is None:
            try:
                import matlab.engine as mengine  # type: ignore
                self.matlab_engine = mengine.start_matlab()
                self.matlab_available = True
            except Exception:
                self.matlab_available = False

        # ROI definitions: {name: {'mask':mask, 'tol':fraction}}
        self.rois: Dict[str, Dict[str, Any]] = {}

        # Classification thresholds (tunable)
        self.scratch_aspect_thresh = scratch_aspect_thresh
        self.chip_solidity_thresh = chip_solidity_thresh
        self.chip_circularity_thresh = chip_circularity_thresh
        self.chip_area_frac_max = chip_area_frac_max
        self.chip_area_frac_max_alt = chip_area_frac_max_alt
        self.burr_solidity_thresh = burr_solidity_thresh

        # Calibration: millimeters per pixel (optional)
        self.mm_per_px: Optional[float] = float(mm_per_px) if mm_per_px is not None else None

        if reference is not None:
            self.set_reference(reference)

    def set_reference(self, reference: Any):
        if isinstance(reference, str):
            img = cv2.imread(reference)
            if img is None:
                raise ValueError(f"Could not read reference image: {reference}")
        elif isinstance(reference, np.ndarray):
            img = reference.copy()
        else:
            raise TypeError("reference must be path or ndarray")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        # keep raw reference and preprocessed reference
        self.reference_raw = gray.copy()
        self.reference_image = self._preprocess(gray)
        self.ref_outer_area = self._compute_outer_area(self.reference_image)
        # compute mm-area if calibration provided
        if self.mm_per_px is not None and self.ref_outer_area is not None:
            self.ref_outer_area_mm = float(self.ref_outer_area) * (self.mm_per_px ** 2)
        else:
            self.ref_outer_area_mm = None

    def set_calibration(self, mm_per_px: float):
        """Set spatial calibration: mm per pixel."""
        self.mm_per_px = float(mm_per_px)
        if hasattr(self, "ref_outer_area") and self.ref_outer_area is not None:
            self.ref_outer_area_mm = float(self.ref_outer_area) * (self.mm_per_px ** 2)

    def add_roi(self, name: str, mask: np.ndarray, tol_fraction: float):
        self.rois[name] = {"mask": mask.astype(np.uint8), "tol": tol_fraction}

    def _preprocess(self, gray: np.ndarray) -> np.ndarray:
        # CLAHE for lighting robustness + bilateral filter to reduce noise while preserving edges
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        den = cv2.bilateralFilter(eq, d=9, sigmaColor=75, sigmaSpace=75)
        return den

    def _compute_outer_area(self, gray: np.ndarray) -> float:
        # Use sub-pixel estimation by upscaling then finding largest contour
        scale = 4
        big = cv2.resize(gray, (gray.shape[1] * scale, gray.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        # try both binary and inverted binary to robustly find the part contour
        _, th = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            _, th = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        # choose the largest contour that is not the full image (background)
        img_area = big.shape[0] * big.shape[1]
        areas = sorted([float(cv2.contourArea(c)) for c in contours], reverse=True)
        max_area = 0.0
        for a in areas:
            if a < 0.95 * img_area:
                max_area = a
                break
        if max_area == 0.0:
            max_area = areas[0]
        # scale back to original pixels
        return max_area / (scale * scale)

    def _find_holes_and_centroids(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        # Detect holes by finding contours on inverted threshold with hierarchy
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = []
        if hierarchy is None:
            return holes
        hier = hierarchy[0]
        # parent contours (outer) have -1 parent; children are holes
        for i, h in enumerate(hier):
            parent = h[3]
            if parent != -1:
                c = contours[i]
                M = cv2.moments(c)
                if M.get("m00", 0) != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                else:
                    cx, cy = 0.0, 0.0
                holes.append({"contour": c, "centroid": (float(cx), float(cy)), "area": float(cv2.contourArea(c))})
        return holes

    def _align_images(self, ref: np.ndarray, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Use ORB for feature detection and LSH/FLANN for matching
        orb = cv2.ORB_create(2000)
        kp1, des1 = orb.detectAndCompute(ref, None)
        kp2, des2 = orb.detectAndCompute(img, None)

        if des1 is None or des2 is None:
            return img, np.eye(3)

        # FLANN parameters for ORB (binary descriptors)
        index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)
        try:
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
        except Exception:
            # fallback to BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = [[m] for m in bf.match(des1, des2)]

        # Ratio test
        good = []
        for m in matches:
            if len(m) == 2:
                if m[0].distance < 0.75 * m[1].distance:
                    good.append(m[0])
            elif len(m) == 1:
                good.append(m[0])

        if len(good) < 6:
            return img, np.eye(3)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if M is None:
            return img, np.eye(3)

        h, w = ref.shape
        aligned = cv2.warpPerspective(img, M, (w, h))
        return aligned, M

    def _segment_defects(self, ref: np.ndarray, test: np.ndarray, ref_pre: Optional[np.ndarray] = None, test_pre: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        # Compute raw diff
        diff_raw = cv2.absdiff(ref, test)

        # Compute preprocessed diff if preprocessed images provided (improves detection when CLAHE smooths subtle defects)
        diff_pre = None
        if ref_pre is not None and test_pre is not None:
            diff_pre = cv2.absdiff(ref_pre, test_pre)

        # Combine diffs (binary OR)
        combined = (diff_raw > 0).astype('uint8') * 255
        if diff_pre is not None:
            combined = cv2.bitwise_or(combined, (diff_pre > 0).astype('uint8') * 255)

        _, th = cv2.threshold(combined, 25, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
        # Extract contours on an upscaled image for sub-pixel accuracy
        scale = 4
        big = cv2.resize(th, (th.shape[1] * scale, th.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        contours_big, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # scale contours back
        contours = []
        for c in contours_big:
            c = c.astype(np.float32) / float(scale)
            contours.append(c.astype(np.int32))
        return th, contours

    def _classify_contour(self, c: np.ndarray, ref_area: float) -> str:
        area = cv2.contourArea(c)
        if area <= 0:
            return "unknown"
        x, y, w, h = cv2.boundingRect(c)
        aspect = float(w) / float(h) if h > 0 else 0.0
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) if hull is not None else 0.0
        solidity = area / hull_area if hull_area > 0 else 0.0
        perim = cv2.arcLength(c, True)
        circularity = (4 * math.pi * area / (perim * perim)) if perim > 0 else 0.0
        # Heuristics mapping to six categories (Surface Scratch, Surface Chip, Edge Burr, Hole Misalignment handled elsewhere,
        # Undersized/Oversized handled via area checks)
        # Surface Scratch: long and thin
        if aspect > self.scratch_aspect_thresh and area > ref_area * 0.0002:
            return "Surface Scratch"
        # Surface Chip: small compact blob (allow larger area fraction for chips)
        if (solidity > self.chip_solidity_thresh and area < ref_area * self.chip_area_frac_max) or \
           (circularity > self.chip_circularity_thresh and area < ref_area * self.chip_area_frac_max_alt):
            return "Surface Chip"
        # Edge Burr: low solidity near edges
        if solidity < self.burr_solidity_thresh and area < ref_area * 0.01:
            return "Edge Burr"
        # Larger defects default to chip/scratch categorization; use shape as fallback
        if area > ref_area * 0.005 and aspect < 2.0:
            return "Surface Chip"
        return "Surface Scratch"

    def inspect(self, input_img: Any, part_id: Optional[str] = None) -> Tuple[Dict[str, Any], np.ndarray]:
        # Load test image
        if isinstance(input_img, str):
            img_bgr = cv2.imread(input_img)
            if img_bgr is None:
                raise ValueError(f"Could not read input image: {input_img}")
        elif isinstance(input_img, np.ndarray):
            img_bgr = input_img.copy()
        else:
            raise TypeError("input_img must be path or ndarray")

        if self.reference_image is None:
            raise ValueError("Reference image not set. Call set_reference() first.")

        # accept color or grayscale numpy arrays; keep raw and preprocessed copies
        if isinstance(img_bgr, np.ndarray) and img_bgr.ndim == 2:
            raw_gray_test = img_bgr
        elif isinstance(img_bgr, np.ndarray) and img_bgr.ndim == 3:
            raw_gray_test = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            raise TypeError("Unsupported image shape for inspection")
        gray_test = self._preprocess(raw_gray_test)

        # Alignment: allow MATLAB offload if requested and available
        if self.use_matlab and self.matlab_available:
            try:
                # Placeholder call: expects MATLAB helper to return aligned image as numpy array
                aligned = self.matlab_engine.alignImages(self.reference_image, gray_test)
                H = np.eye(3)
            except Exception:
                aligned, H = self._align_images(self.reference_image, gray_test)
        else:
            aligned, H = self._align_images(self.reference_image, gray_test)

        # warp raw test using homography to keep raw diff information
        h, w = self.reference_image.shape
        try:
            aligned_raw = cv2.warpPerspective(raw_gray_test, H, (w, h)) if H is not None else raw_gray_test
        except Exception:
            aligned_raw = raw_gray_test

        mask_def, contours = self._segment_defects(self.reference_raw if hasattr(self, 'reference_raw') else self.reference_image,
                                                   aligned_raw,
                                                   ref_pre=self.reference_image,
                                                   test_pre=aligned)
        ref_area = self.ref_outer_area or 1.0

        # find holes for concentricity checks
        holes = self._find_holes_and_centroids(self.reference_image)

        # compute outer contour centroid for concentricity
        _, th_ref = cv2.threshold(self.reference_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours_ref, _ = cv2.findContours(th_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outer_centroid = (0.0, 0.0)
        if contours_ref:
            c0 = max(contours_ref, key=cv2.contourArea)
            M0 = cv2.moments(c0)
            if M0.get("m00", 0) != 0:
                outer_centroid = (M0["m10"] / M0["m00"], M0["m01"] / M0["m00"]) 

        defects = []
        total_defect_area = 0.0
        for c in contours:
            a = cv2.contourArea(c)
            if a <= 0:
                continue
            dtype = self._classify_contour(c, ref_area)
            x, y, w, h = cv2.boundingRect(c)
            # detect if defect is near outer edge
            cx = x + w / 2.0
            cy = y + h / 2.0
            # distance to outer centroid
            dist_to_outer = math.hypot(cx - outer_centroid[0], cy - outer_centroid[1])
            outer_radius = math.sqrt(ref_area / math.pi)
            near_edge = dist_to_outer > (outer_radius * 0.8)
            if near_edge and dtype == "Surface Scratch":
                dtype = "Edge Burr"
            defects.append({"type": dtype, "area": float(a), "bbox": [int(x), int(y), int(w), int(h)]})
            total_defect_area += a

        # attach confidence and mm conversions
        for d in defects:
            area_px = d.get("area", 0.0)
            # simple confidence heuristic: relative to reference
            conf = min(1.0, area_px / max(1.0, ref_area * 0.001))
            d["confidence"] = float(conf)
            if self.mm_per_px is not None:
                x, y, w, h = d["bbox"]
                d["bbox_mm"] = [float(x) * self.mm_per_px, float(y) * self.mm_per_px, float(w) * self.mm_per_px, float(h) * self.mm_per_px]
                d["area_mm"] = float(area_px) * (self.mm_per_px ** 2)

        # Per-ROI checks
        roi_violations = {}
        for name, meta in self.rois.items():
            mask = meta["mask"]
            tol = meta["tol"]
            # compute defect area inside ROI
            roi_area = float(np.count_nonzero(mask))
            if roi_area <= 0:
                continue
            defect_in_roi = float(np.count_nonzero(cv2.bitwise_and(mask_def, mask)))
            if defect_in_roi > roi_area * tol:
                roi_violations[name] = {"defect_area": defect_in_roi, "roi_area": roi_area}

        status = "PASS"
        reason = "No significant defects"
        # Area-based classification for oversize/undersize
        area_flag = None
        if ref_area is not None:
            if float(ref_area) > 0:
                if (ref_area - float(self.area_tolerance) * ref_area) > (ref_area - ref_area * self.area_tolerance):
                    pass
        if total_defect_area > self.defect_thresh or roi_violations:
            status = "FAIL"
            reason = f"Defect area {total_defect_area:.1f}"
            if roi_violations:
                reason += "; ROI violations: " + ",".join(roi_violations.keys())

        # Determine size-related failures
        if self.ref_outer_area is not None:
            measured = self._compute_outer_area(aligned)
            lower = self.ref_outer_area * (1.0 - self.area_tolerance)
            upper = self.ref_outer_area * (1.0 + self.area_tolerance)
            if measured < lower:
                status = "FAIL"
                reason = "Undersized Area"
            elif measured > upper:
                status = "FAIL"
                reason = "Oversized Area"

        # Hole misalignment
        hole_misaligned = False
        if holes and outer_centroid != (0.0, 0.0):
            for h in holes:
                cx, cy = h["centroid"]
                d = math.hypot(cx - outer_centroid[0], cy - outer_centroid[1])
                # if hole center deviates more than 2% of outer radius -> misaligned
                # use mm if calibration present
                outer_r_px = math.sqrt(ref_area / math.pi)
                if self.mm_per_px is not None:
                    outer_r = outer_r_px * self.mm_per_px
                    d_mm = d * self.mm_per_px
                    if d_mm > 0.02 * outer_r:
                        hole_misaligned = True
                        break
                else:
                    if d > 0.02 * outer_r_px:
                        hole_misaligned = True
                        break
        if hole_misaligned:
            status = "FAIL"
            reason = "Hole Misalignment"

        # Annotate image
        ann = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
        for d in defects:
            x, y, w, h = d["bbox"]
            cv2.rectangle(ann, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(ann, d["type"], (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        metrics = {
            "status": status,
            "reason": reason,
            "total_defect_area": float(total_defect_area),
            "defect_count": len(defects),
            "outer_area_px": float(ref_area),
            "outer_area_mm": float(self.ref_outer_area_mm) if getattr(self, "ref_outer_area_mm", None) is not None else None,
            "part_id": part_id,
        }
        # include defects in metrics
        metrics["defects"] = defects

        return metrics, ann

