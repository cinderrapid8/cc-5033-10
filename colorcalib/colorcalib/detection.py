"""Chart detection and per-patch color extraction.

Two entry modes:

* ``detect_chart`` — automatic OpenCV detection of the matte patch grid,
  perspective-unwarped to a canonical rectangle;
* ``extract_manual`` — caller supplies the four outer chart corners in the
  image (row-major, clockwise from top-left), used when automatic detection
  fails on cluttered product shots.

Every extracted patch carries diagnostics instead of a silent number:
channel clipping, near-clipping, highlight sheen (specular reflection), and
within-patch non-uniformity.  Fitting code consumes the diagnostics and
refuses to treat saturated or mis-detected patches as trustworthy samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

CLIP_LEVEL = 250          # any channel at/above this (8-bit) is clipped
NEAR_CLIP_LEVEL = 235     # headroom warning
DARK_LEVEL = 8             # below this, read noise dominates
SHEEN_RATIO = 1.25         # bright pixels 25% above patch median => sheen
UNIFORMITY_CV = 0.12       # coefficient-of-variation threshold per channel


@dataclass
class PatchDiagnosis:
    index: int
    name: str
    rgb_mean: np.ndarray      # 0..1, mean of the ROI
    rgb_median: np.ndarray    # 0..1, median of the ROI (adopted value)
    clipped: bool
    near_clipped: bool
    near_black: bool
    sheen: bool
    nonuniform: bool
    cv_per_channel: np.ndarray
    p98: np.ndarray           # 98th percentile per channel, 0..1
    roi_px: int
    rejected: bool = False
    reject_reasons: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.rejected


@dataclass
class DetectionResult:
    chart_name: str
    rows: int
    cols: int
    diagnoses: list[PatchDiagnosis]
    warp: np.ndarray | None          # 3x3 perspective matrix used
    corners: np.ndarray | None       # four outer corners in the source image
    unwarped: np.ndarray | None      # canonical view of the chart (uint8 BGR)
    detection_method: str
    confidence: float

    @property
    def adopted_rgb(self) -> np.ndarray:
        """Median sRGB-encoded values adopted per patch, (N,3) in 0..1."""
        return np.vstack([d.rgb_median for d in self.diagnoses])

    @property
    def usable_mask(self) -> np.ndarray:
        return np.array([d.usable for d in self.diagnoses], dtype=bool)

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array(
        [
            pts[np.argmin(s)],
            pts[np.argmin(diff)],
            pts[np.argmax(s)],
            pts[np.argmax(diff)],
        ],
        dtype=np.float32,
    )


def _quad_spans_extremes(
    quad: np.ndarray, ux: float, uy: float, ux2: float, uy2: float
) -> bool:
    tol_x, tol_y = 0.03 * (ux2 - ux), 0.03 * (uy2 - uy)
    return (
        quad[:, 0].min() <= ux + tol_x
        and quad[:, 0].max() >= ux2 - tol_x
        and quad[:, 1].min() <= uy + tol_y
        and quad[:, 1].max() >= uy2 - tol_y
    )


def _quad_is_rectangular(quad: np.ndarray, angle_tol: float = 12.0) -> bool:
    """True if four ordered corners form a plausible planar chart quad.

    Allows the moderate shear of a perspective photograph while rejecting the
    self-intersecting / truncated polygons that hull simplification produces
    when some cells are missing from the thresholded image.
    """
    tl, tr, br, bl = quad
    edges = [tr - tl, br - tr, bl - br, tl - bl]
    lengths = [float(np.linalg.norm(e)) for e in edges]
    if min(lengths) < 20:
        return False
    # Opposite edges must have comparable length. A genuine planar chart
    # quadrilateral — even under perspective — keeps each opposite pair within
    # ~1.6x; a hull polygon truncated by missing cells collapses one edge.
    if max(lengths[0], lengths[2]) / max(min(lengths[0], lengths[2]), 1e-6) > 1.6:
        return False
    if max(lengths[1], lengths[3]) / max(min(lengths[1], lengths[3]), 1e-6) > 1.6:
        return False
    angles = []
    crosses = []
    for i in range(4):
        a, b = edges[i], edges[(i + 1) % 4]
        cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        angles.append(np.degrees(np.arccos(np.clip(-cosang, -1, 1))))
        crosses.append(float(a[0] * b[1] - a[1] * b[0]))
    if not all(abs(ang - 90.0) <= angle_tol for ang in angles):
        return False
    # A convex quadrilateral has cross products of consistent sign; a
    # truncated hull polygon flips sign where it folds back on itself.
    if not (all(c >= 0 for c in crosses) or all(c <= 0 for c in crosses)):
        return False
    return True


def _quad_edges_slanted(quad: np.ndarray, min_deg: float) -> bool:
    """True when any boundary edge consistently leaves the image axes.

    Horizontal perspective keeps vertical edges nearly axis-aligned while
    slanting the top/bottom edges, so the trigger is the maximum edge angle
    rather than all four. Truncated hull polygons with one jagged edge are
    already rejected upstream by :func:`_quad_is_rectangular`.
    """
    tl, tr, br, bl = _order_corners(quad)
    horiz = [tr - tl, bl - br]
    vert = [br - tr, tl - bl]
    h_angles = [
        min(abs(np.degrees(np.arctan2(e[1], e[0]))),
            180 - abs(np.degrees(np.arctan2(e[1], e[0]))))
        for e in horiz
    ]
    v_angles = [
        min(abs(np.degrees(np.arctan2(e[0], e[1]))),
            180 - abs(np.degrees(np.arctan2(e[0], e[1]))))
        for e in vert
    ]
    return max(max(h_angles), max(v_angles)) >= min_deg


def _grid_rois(rows: int, cols: int, cell_w: int, cell_h: int, shrink: float):
    """Yield (index, x0, y0, x1, y1) ROIs in the unwarped chart image."""
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            cx0 = c * cell_w
            cy0 = r * cell_h
            pad_x = cell_w * (1 - shrink) / 2
            pad_y = cell_h * (1 - shrink) / 2
            x0 = int(cx0 + pad_x)
            y0 = int(cy0 + pad_y)
            x1 = int(cx0 + cell_w - pad_x)
            y1 = int(cy0 + cell_h - pad_y)
            yield idx, x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def _diagnose_patch(
    idx: int, name: str, roi: np.ndarray
) -> PatchDiagnosis:
    """roi: HxWx3 uint8 RGB.

    Uniformity uses the median absolute deviation relative to the *lightest*
    channel, so a dark but chromatic patch (large channel ratios, low photon
    counts) is not flagged merely because its dark channels are noisy.
    """
    flat = roi.reshape(-1, 3).astype(np.float64)
    median = np.median(flat, axis=0)
    mean = flat.mean(axis=0)
    p98 = np.percentile(flat, 98, axis=0)
    mad = np.median(np.abs(flat - median), axis=0)
    std = flat.std(axis=0)
    # Robust relative spread; denominator is the brightest channel level.
    level = max(float(median.max()), 32.0)
    cv = (1.4826 * mad) / level

    # True clipping: a non-trivial fraction of pixels pinned at the ceiling.
    frac_at_ceiling = float(np.mean(np.any(flat >= CLIP_LEVEL, axis=1)))
    clipped = frac_at_ceiling > 0.02 or bool(np.all(median >= CLIP_LEVEL - 2))
    near_clipped = bool(np.any(p98 >= NEAR_CLIP_LEVEL))
    near_black = bool(np.all(median <= DARK_LEVEL))
    # Sheen: a bright sub-population well above the patch's own level AND
    # approaching highlight territory (not just chromatic channel ratios).
    sheen = bool(
        np.any(p98 > median * SHEEN_RATIO)
        and np.max(p98) >= 180
        and float(np.max(p98) - np.max(median)) >= 45
    )
    nonuniform = bool(np.any(cv > UNIFORMITY_CV))

    reasons = []
    if clipped:
        reasons.append("channel_clipped")
    if sheen:
        reasons.append("specular_sheen")
    if nonuniform:
        reasons.append("nonuniform_roi")
    if near_black:
        reasons.append("near_black_unreliable")

    d = PatchDiagnosis(
        index=idx,
        name=name,
        rgb_mean=mean / 255.0,
        rgb_median=median / 255.0,
        clipped=clipped,
        near_clipped=near_clipped,
        near_black=near_black,
        sheen=sheen,
        nonuniform=nonuniform,
        cv_per_channel=cv,
        p98=p98 / 255.0,
        roi_px=flat.shape[0],
        rejected=clipped or sheen or nonuniform,
        reject_reasons=reasons,
    )
    return d


def _unwarp_and_sample(
    image_bgr: np.ndarray,
    corners: np.ndarray,
    rows: int,
    cols: int,
    names: list[str],
    cell_px: int = 160,
    shrink: float = 0.55,
) -> tuple[np.ndarray, np.ndarray, list[PatchDiagnosis]]:
    cell_w = cell_h = cell_px
    dst = np.array(
        [
            [0, 0],
            [cols * cell_w - 1, 0],
            [cols * cell_w - 1, rows * cell_h - 1],
            [0, rows * cell_h - 1],
        ],
        dtype=np.float32,
    )
    warp = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    unwarped = cv2.warpPerspective(
        image_bgr, warp, (cols * cell_w, rows * cell_h),
        flags=cv2.INTER_CUBIC,
    )
    rgb = cv2.cvtColor(unwarped, cv2.COLOR_BGR2RGB)
    diagnoses: list[PatchDiagnosis] = []
    for idx, x0, y0, x1, y1 in _grid_rois(rows, cols, cell_w, cell_h, shrink):
        roi = rgb[y0:y1, x0:x1]
        diagnoses.append(_diagnose_patch(idx, names[idx], roi))
    return warp, unwarped, diagnoses


def detect_chart(
    image_path: str,
    rows: int,
    cols: int,
    patch_names: list[str],
    chart_name: str = "chart",
    min_confidence: float = 0.35,
) -> DetectionResult:
    """Auto-detect a rectangular matte patch grid and sample its cells.

    Raises ``DetectionError`` when no plausible grid is found; callers should
    then fall back to manual corners rather than silently sampling wrong areas.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_area = h * w
    expected_aspect = cols / rows

    candidates: list[tuple[float, np.ndarray]] = []
    # Several thresholds make detection robust to exposure and background tone:
    # each chart cell is a bright component relative to its local surroundings.
    for thresh_frac in (0.35, 0.45, 0.55):
        thresh_val = max(40, int(np.percentile(gray, thresh_frac * 100)))
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        pad = int(min(h, w) * 0.05)
        binary = cv2.copyMakeBorder(
            binary, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
        )
        n, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        # Merge cell-level components: keep large bright blobs, take bounding
        # rectangles and group those whose union forms the chart rectangle.
        boxes = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < img_area * 0.002:
                continue
            boxes.append((x - pad, y - pad, bw, bh, area))
        if not boxes:
            continue
        xs = [b[0] for b in boxes]
        ys = [b[1] for b in boxes]
        x2 = [b[0] + b[2] for b in boxes]
        y2 = [b[1] + b[3] for b in boxes]
        ux, uy, ux2, uy2 = min(xs), min(ys), max(x2), max(y2)
        # A perspective chart is a trapezoid, which minAreaRect cannot express.
        # Take the convex hull of every detected cell corner and simplify it
        # to the tightest four-vertex polygon.
        all_corners = np.array(
            [
                corner
                for (x, y, bw, bh, _a) in boxes
                for corner in (
                    [x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]
                )
            ],
            dtype=np.float32,
        )
        hull = cv2.convexHull(all_corners)
        peri = cv2.arcLength(hull, True)
        hull_quad = None
        for eps_frac in (0.01, 0.02, 0.04, 0.07, 0.1):
            approx = cv2.approxPolyDP(hull, eps_frac * peri, True)
            if len(approx) != 4:
                continue
            quad = _order_corners(approx.reshape(4, 2).astype(np.float32))
            if not _quad_spans_extremes(quad, ux, uy, ux2, uy2):
                continue
            if _quad_is_rectangular(quad):
                hull_quad = quad
                break
        axis_box = np.array(
            [[ux, uy], [ux2, uy], [ux2, uy2], [ux, uy2]], dtype=np.float32
        )
        area = (ux2 - ux) * (uy2 - uy)
        (rw, rh) = cv2.minAreaRect(all_corners)[1]
        aspect = max(rw, rh) / max(min(rw, rh), 1.0)
        if area < img_area * 0.02 or area > img_area * 0.95:
            continue
        fill = sum(b[4] for b in boxes) / max(area, 1)
        if fill < 0.45:
            continue
        aspect_score = 1.0 - min(abs(aspect - expected_aspect) / expected_aspect, 1.0)
        base_score = (
            0.55 * aspect_score
            + 0.25 * min(fill, 1.0)
            + 0.2 * min(area / (img_area * 0.3), 1.0)
        )
        # Keep BOTH boundary hypotheses. The correct one is chosen later by
        # actually warping and checking that every grid cell lands on a
        # distinct patch — geometry heuristics alone cannot separate a
        # front-facing chart from a weakly perspective one.
        for cand in (axis_box, hull_quad):
            if cand is None:
                continue
            candidates.append((base_score, cand))

    if not candidates:
        raise DetectionError(
            "no chart grid found; use manual corner mode"
        )

    # De-duplicate overlapping hypotheses, then score each by warping it and
    # measuring real grid behavior: neighboring cells must be distinct colors
    # while each cell's interior stays uniform. This is the ground-truth test
    # that geometry-only heuristics approximate.
    unique_cands: list[np.ndarray] = []
    for _, quad in candidates:
        q = _order_corners(quad)
        if any(float(np.max(np.abs(q - u))) < 3.0 for u in unique_cands):
            continue
        unique_cands.append(q)

    best = None
    for q in unique_cands:
        _, _, diag = _unwarp_and_sample(image, q, rows, cols, patch_names)
        med = np.vstack([d.rgb_median for d in diag])
        diffs = [
            np.linalg.norm(med[r * cols + c] - med[r * cols + c + 1])
            for r in range(rows) for c in range(cols - 1)
        ]
        distinct_ratio = float(np.mean(np.array(diffs) > 0.03))
        mean_cv = float(np.mean([d.cv_per_channel.max() for d in diag]))
        n_clean = sum(not d.nonuniform for d in diag)
        if distinct_ratio < 0.5:
            continue
        quality = distinct_ratio * 2.0 + n_clean / len(diag) - mean_cv
        if best is None or quality > best[0]:
            best = (quality, q, diag)

    if best is None:
        raise DetectionError(
            "grid check failed for every candidate quadrilateral — use manual "
            "corner mode"
        )
    confidence = float(np.mean([s for s, _ in candidates]))
    _, corners, diagnoses = best
    warp, unwarped, _ = _unwarp_and_sample(
        image, corners, rows, cols, patch_names
    )
    corners = _order_corners(corners)

    return DetectionResult(
        chart_name=chart_name,
        rows=rows,
        cols=cols,
        diagnoses=diagnoses,
        warp=warp,
        corners=corners,
        unwarped=unwarped,
        detection_method="auto",
        confidence=float(confidence),
    )


def extract_manual(
    image_path: str,
    corners_xy: np.ndarray,
    rows: int,
    cols: int,
    patch_names: list[str],
    chart_name: str = "chart",
) -> DetectionResult:
    """Sample the chart using four caller-provided outer corners (TL..BL)."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    corners = _order_corners(np.asarray(corners_xy, dtype=np.float32))
    warp, unwarped, diagnoses = _unwarp_and_sample(
        image, corners, rows, cols, patch_names
    )
    return DetectionResult(
        chart_name=chart_name,
        rows=rows,
        cols=cols,
        diagnoses=diagnoses,
        warp=warp,
        corners=corners,
        unwarped=unwarped,
        detection_method="manual",
        confidence=1.0,
    )


class DetectionError(RuntimeError):
    pass
