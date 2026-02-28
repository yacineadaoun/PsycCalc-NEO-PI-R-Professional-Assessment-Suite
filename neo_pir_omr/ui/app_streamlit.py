from __future__ import annotations

import csv
import io
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

from .logging_conf import setup_logging, LoggingConfig


# ============================================================
# Data models
# ============================================================


@dataclass
class OMRConfig:
    # Geometry
    target_width: int = 1800
    rows: int = 30
    cols: int = 8

    # Detection
    use_micro_adjust: bool = True

    # Marking / human tolerance
    detect_blue: bool = True
    detect_black: bool = True
    black_dark_thresh: int = 110
    black_baseline_quantile: float = 15.0

    # Decision thresholds
    mark_threshold: float = 1.7
    ambiguity_gap: float = 0.9

    # Protocol rules (NEO PI-R typical)
    max_blank_invalid: int = 15
    max_N_invalid: int = 42
    impute_blank_if_leq: int = 10
    impute_option_index: int = 2  # N


@dataclass
class ScanResult:
    scan_id: str
    responses_raw: Dict[int, int]
    responses_final: Dict[int, int]
    meta: Dict[int, dict]
    protocol: dict
    facette_scores: Dict[str, int]
    domain_scores: Dict[str, int]
    overlay_bgr: np.ndarray
    debug_mask: np.ndarray
    warped_bgr: np.ndarray
    diagnostics: dict


# ============================================================
# Low-level helpers
# ============================================================

def cv_find_contours(binary: np.ndarray, mode: int, method: int):
    res = cv2.findContours(binary, mode, method)
    if len(res) == 2:
        contours, hierarchy = res
    else:
        _img, contours, hierarchy = res
    return contours, hierarchy


def rotate_k90(bgr: np.ndarray, k: int) -> np.ndarray:
    k = k % 4
    if k == 0:
        return bgr
    if k == 1:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if k == 2:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)


def resize_keep_aspect(bgr: np.ndarray, target_width: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w <= target_width:
        return bgr
    scale = target_width / float(w)
    nh = int(h * scale)
    return cv2.resize(bgr, (target_width, nh), interpolation=cv2.INTER_AREA)


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = max(1, int(max(widthA, widthB)))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = max(1, int(max(heightA, heightB)))

    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def binarize_inv(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )
    return thr


def build_grid_mask(thr_inv: np.ndarray) -> np.ndarray:
    h, w = thr_inv.shape[:2]
    hk = max(20, w // 18)
    vk = max(20, h // 18)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))

    hor = cv2.morphologyEx(thr_inv, cv2.MORPH_OPEN, h_kernel, iterations=1)
    ver = cv2.morphologyEx(thr_inv, cv2.MORPH_OPEN, v_kernel, iterations=1)

    grid = cv2.bitwise_or(hor, ver)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return grid


def _peaks_from_projection(proj: np.ndarray, thr: float, min_dist: int) -> List[int]:
    if proj.size == 0:
        return []
    p = proj.astype(np.float32)
    m = float(p.max())
    if m <= 0:
        return []
    p /= m

    idx = np.where(p >= thr)[0].tolist()
    if not idx:
        return []

    runs = []
    s = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            runs.append((s, prev))
            s = i
            prev = i
    runs.append((s, prev))

    peaks: List[int] = []
    last = -10**9
    for a, b in runs:
        seg = p[a : b + 1]
        k = int(a + np.argmax(seg))
        if k - last >= int(min_dist):
            peaks.append(k)
            last = k
    return peaks


def _best_regular_run(
    peaks: List[int], run_len: int, expected_min_step: int, expected_max_step: int
) -> Tuple[int, float]:
    if len(peaks) < run_len:
        return -1, float("inf")

    peaks = sorted(peaks)
    best_i = -1
    best_score = float("inf")

    for i in range(0, len(peaks) - run_len + 1):
        window = np.array(peaks[i : i + run_len], dtype=np.float32)
        diffs = np.diff(window)
        mean = float(np.mean(diffs)) if diffs.size else 0.0
        std = float(np.std(diffs)) if diffs.size else float("inf")

        penalty = 0.0
        if mean < float(expected_min_step) or mean > float(expected_max_step):
            penalty = 10.0

        score = std + penalty
        if score < best_score:
            best_score = score
            best_i = i

    return best_i, best_score


def find_table_bbox_soft(thr_inv: np.ndarray) -> Tuple[int, int, int, int]:
    grid = build_grid_mask(thr_inv)
    H, W = thr_inv.shape[:2]

    x_cut = int(W * 0.72)
    x_cut = max(1, min(W, x_cut))
    hor_roi = grid[:, :x_cut]

    proj_y = hor_roi.sum(axis=1).astype(np.float32)
    min_dist_y = max(8, H // 80)
    peaks_y = _peaks_from_projection(proj_y, thr=0.35, min_dist=min_dist_y)

    exp_min_step_y = max(10, H // 70)
    exp_max_step_y = max(14, H // 20)

    iy, _ = _best_regular_run(
        peaks_y, run_len=31, expected_min_step=exp_min_step_y, expected_max_step=exp_max_step_y
    )

    if iy < 0:
        fx = int(W * 0.05)
        fy = int(H * 0.10)
        fw = int(W * 0.90)
        fh = int(H * 0.74)
        return fx, fy, fw, fh

    y_top = int(peaks_y[iy])
    y_bot = int(peaks_y[iy + 30])

    y_cut = int(H * 0.86)
    y_cut = max(1, min(H, y_cut))
    ver_roi = grid[:y_cut, :]

    proj_x = ver_roi.sum(axis=0).astype(np.float32)
    min_dist_x = max(10, W // 120)
    peaks_x = _peaks_from_projection(proj_x, thr=0.35, min_dist=min_dist_x)

    exp_min_step_x = max(30, W // 30)
    exp_max_step_x = max(45, W // 10)

    ix, _ = _best_regular_run(
        peaks_x, run_len=9, expected_min_step=exp_min_step_x, expected_max_step=exp_max_step_x
    )

    if ix < 0:
        x_left = int(W * 0.05)
        x_right = int(W * 0.95)
    else:
        x_left = int(peaks_x[ix])
        x_right = int(peaks_x[ix + 8])

    pad = 2
    x1 = max(0, x_left - pad)
    x2 = min(W - 1, x_right + pad)
    y1 = max(0, y_top - pad)
    y2 = min(H - 1, y_bot + pad)

    w = max(1, int(x2 - x1))
    h = max(1, int(y2 - y1))
    return int(x1), int(y1), w, h


def split_grid_uniform(table_bbox: Tuple[int, int, int, int], rows: int, cols: int):
    x, y, w, h = table_bbox
    cw = w / cols
    ch = h / rows
    for r in range(rows):
        for c in range(cols):
            x1 = int(x + c * cw)
            y1 = int(y + r * ch)
            x2 = int(x + (c + 1) * cw)
            y2 = int(y + (r + 1) * ch)
            yield r, c, (x1, y1, x2, y2)


def split_grid_micro_adjust(thr_inv: np.ndarray, table_bbox: Tuple[int, int, int, int], rows: int, cols: int):
    x, y, w, h = table_bbox
    roi = thr_inv[y : y + h, x : x + w]
    grid = build_grid_mask(roi)

    proj_x = grid.sum(axis=0).astype(np.float32)
    proj_y = grid.sum(axis=1).astype(np.float32)

    if proj_x.max() > 0:
        proj_x /= proj_x.max()
    if proj_y.max() > 0:
        proj_y /= proj_y.max()

    def pick_peaks(proj: np.ndarray, n_needed: int, min_dist: int, thr: float) -> List[int]:
        idxs = np.where(proj >= thr)[0].tolist()
        if not idxs:
            return []
        peaks = []
        last = -10**9
        for i in idxs:
            if i - last >= min_dist:
                peaks.append(i)
                last = i
        if len(peaks) > n_needed:
            keep = np.linspace(0, len(peaks) - 1, n_needed).round().astype(int)
            peaks = [peaks[i] for i in keep]
        return peaks

    vx = pick_peaks(proj_x, n_needed=cols + 1, min_dist=max(6, w // 60), thr=0.35)
    hy = pick_peaks(proj_y, n_needed=rows + 1, min_dist=max(6, h // 90), thr=0.35)

    if len(vx) != cols + 1 or len(hy) != rows + 1:
        yield from split_grid_uniform(table_bbox, rows=rows, cols=cols)
        return

    vx = sorted(vx)
    hy = sorted(hy)

    vxg = [x + v for v in vx]
    hyg = [y + u for u in hy]

    for r in range(rows):
        for c in range(cols):
            x1, x2 = vxg[c], vxg[c + 1]
            y1, y2 = hyg[r], hyg[r + 1]
            yield r, c, (int(x1), int(y1), int(x2), int(y2))


def item_id_from_rc(r: int, c: int) -> int:
    return (r + 1) + 30 * c


def option_rois_in_cell(cell: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = cell
    w = x2 - x1
    h = y2 - y1

    left = x1 + int(0.22 * w)
    right = x2 - int(0.06 * w)
    top = y1 + int(0.20 * h)
    bottom = y2 - int(0.20 * h)

    inner_w = max(1, right - left)
    band_w = inner_w / 5.0

    rois = []
    for k in range(5):
        bx1 = left + k * band_w
        bx2 = left + (k + 1) * band_w
        rx1 = int(bx1 + 0.25 * band_w)
        rx2 = int(bx2 - 0.25 * band_w)
        ry1 = int(top + 0.12 * (bottom - top))
        ry2 = int(bottom - 0.12 * (bottom - top))
        rois.append((rx1, ry1, rx2, ry2))
    return rois


def ink_score(mask: np.ndarray, roi: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = roi
    patch = mask[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return (float(np.count_nonzero(patch)) / float(patch.size)) * 100.0


def build_blue_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    hsv = cv2.merge([h, s, v])

    lower = np.array([85, 40, 40], dtype=np.uint8)
    upper = np.array([145, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.medianBlur(mask, 3)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.dilate(mask, k, iterations=1)
    return mask


def build_black_raw_mask(bgr: np.ndarray, dark_thresh: int = 110) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    dark = cv2.inRange(gray, 0, int(dark_thresh))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k, iterations=1)
    return dark


def _confidence_from_meta(m: dict) -> float:
    # Simple, monotonic proxy: larger gap and larger best -> higher confidence.
    best = float(m.get("best", 0.0))
    gap = float(m.get("gap", 0.0))
    blank = bool(m.get("blank", False))
    if blank:
        return 0.0
    # normalize roughly
    conf = 1.0 / (1.0 + np.exp(-(0.9 * gap + 0.08 * (best - 2.0))))
    return float(np.clip(conf, 0.0, 1.0))


def read_responses_from_grid(
    warped_bgr: np.ndarray,
    thr_inv_print: np.ndarray,
    table_bbox: Tuple[int, int, int, int],
    rows: int,
    cols: int,
    use_micro_adjust: bool,
    mark_threshold: float,
    ambiguity_gap: float,
    detect_blue: bool,
    detect_black: bool,
    black_dark_thresh: int,
    black_baseline_quantile: float,
):
    overlay = warped_bgr.copy()

    blue_mask = build_blue_mask(warped_bgr) if detect_blue else None
    black_raw = build_black_raw_mask(warped_bgr, dark_thresh=black_dark_thresh) if detect_black else None

    iterator = (
        split_grid_micro_adjust(thr_inv_print, table_bbox, rows=rows, cols=cols)
        if use_micro_adjust
        else split_grid_uniform(table_bbox, rows=rows, cols=cols)
    )

    raw_blue: Dict[int, List[float]] = {}
    raw_black: Dict[int, List[float]] = {}

    for r, c, cell in iterator:
        item_id = item_id_from_rc(r, c)
        rois = option_rois_in_cell(cell)

        if detect_blue and blue_mask is not None:
            raw_blue[item_id] = [ink_score(blue_mask, roi) for roi in rois]

        if detect_black and black_raw is not None:
            raw_black[item_id] = [ink_score(black_raw, roi) for roi in rois]

    baseline_black = [0.0] * 5
    if detect_black and raw_black:
        arr = np.array(list(raw_black.values()), dtype=np.float32)
        q = float(np.clip(black_baseline_quantile, 0.0, 50.0))
        baseline_black = [float(np.percentile(arr[:, i], q)) for i in range(5)]

    responses: Dict[int, int] = {}
    meta: Dict[int, dict] = {}

    iterator2 = (
        split_grid_micro_adjust(thr_inv_print, table_bbox, rows=rows, cols=cols)
        if use_micro_adjust
        else split_grid_uniform(table_bbox, rows=rows, cols=cols)
    )

    for r, c, cell in iterator2:
        item_id = item_id_from_rc(r, c)
        rois = option_rois_in_cell(cell)

        fills_blue = raw_blue.get(item_id, [0.0] * 5)
        fills_black_raw = raw_black.get(item_id, [0.0] * 5)
        fills_black_adj = [max(0.0, fb - baseline_black[i]) for i, fb in enumerate(fills_black_raw)]

        fills = [
            (fills_blue[i] if detect_blue else 0.0) + (fills_black_adj[i] if detect_black else 0.0)
            for i in range(5)
        ]

        best_idx = int(np.argmax(fills))
        best = float(fills[best_idx])
        sorted_f = sorted(fills, reverse=True)
        second = float(sorted_f[1]) if len(sorted_f) > 1 else 0.0
        gap = best - second

        mean_f = float(np.mean(fills))
        rel = best - mean_f

        blank = (best < mark_threshold) or (rel < 0.12)
        ambiguous = (not blank) and (gap < ambiguity_gap)

        x1, y1, x2, y2 = cell
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 60, 60), 1)

        if blank:
            responses[item_id] = -1
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            responses[item_id] = best_idx
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if ambiguous:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 2)

        rx1, ry1, rx2, ry2 = rois[best_idx]
        if not blank:
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)

        meta[item_id] = {
            "fills_total": fills,
            "fills_blue": fills_blue,
            "fills_black_raw": fills_black_raw,
            "fills_black_adj": fills_black_adj,
            "baseline_black": baseline_black,
            "chosen_idx": best_idx,
            "best": best,
            "second": second,
            "gap": gap,
            "mean": mean_f,
            "rel": rel,
            "blank": blank,
            "ambiguous": ambiguous,
        }
        meta[item_id]["confidence"] = _confidence_from_meta(meta[item_id])

    debug_mask = np.zeros(warped_bgr.shape[:2], dtype=np.uint8)
    if detect_blue and blue_mask is not None:
        debug_mask = cv2.bitwise_or(debug_mask, blue_mask)
    if detect_black and black_raw is not None:
        debug_mask = cv2.bitwise_or(debug_mask, black_raw)

    return responses, meta, overlay, debug_mask


# ============================================================
# Orientation & document detection
# ============================================================

def _orientation_variance_score(warp_bgr: np.ndarray, rows: int, cols: int) -> float:
    thr = binarize_inv(warp_bgr)
    table_bbox = find_table_bbox_soft(thr)
    fills = []
    for _r, _c, cell in split_grid_uniform(table_bbox, rows=rows, cols=cols):
        rois = option_rois_in_cell(cell)
        fills.append([ink_score(thr, roi) for roi in rois])
    arr = np.array(fills, dtype=np.float32)
    return float(np.var(arr, axis=0).sum())


def _top_left_number_components(warp_bgr: np.ndarray, rows: int, cols: int) -> int:
    thr = binarize_inv(warp_bgr)
    table_bbox = find_table_bbox_soft(thr)
    _r0, _c0, cell0 = next(split_grid_uniform(table_bbox, rows=rows, cols=cols))
    x1, y1, x2, y2 = cell0
    w = x2 - x1
    h = y2 - y1

    nx1 = x1 + int(0.02 * w)
    nx2 = x1 + int(0.20 * w)
    ny1 = y1 + int(0.15 * h)
    ny2 = y2 - int(0.15 * h)
    patch = thr[ny1:ny2, nx1:nx2]
    if patch.size == 0:
        return 999

    patch = cv2.erode(patch, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    bin01 = (patch > 0).astype(np.uint8)
    n_labels, _labels, _stats, _centroids = cv2.connectedComponentsWithStats(bin01, connectivity=8)
    return int(max(0, n_labels - 1))


def choose_best_table_orientation(warp_bgr: np.ndarray, rows: int, cols: int) -> Tuple[np.ndarray, dict]:
    best_key = None
    best_img = None

    for r2 in [0, 1, 2, 3]:
        img = rotate_k90(warp_bgr, r2)
        try:
            comps = _top_left_number_components(img, rows=rows, cols=cols)
            var_sc = _orientation_variance_score(img, rows=rows, cols=cols)
        except Exception:
            continue
        key = (comps, var_sc)
        if best_key is None or key < best_key:
            best_key = key
            best_img = img

    if best_img is None:
        return warp_bgr, {"components": None, "variance": None}
    return best_img, {"components": best_key[0], "variance": float(best_key[1])}


def find_document_warp_auto_rotate(bgr: np.ndarray, target_width: int, rows: int, cols: int) -> Tuple[np.ndarray, dict]:
    best = None
    best_warp = None

    for k in [0, 1, 2, 3]:
        img = rotate_k90(bgr, k)
        resized = resize_keep_aspect(img, target_width)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 50, 150)

        cnts, _ = cv_find_contours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts[:12]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                warp = four_point_transform(resized, approx.reshape(4, 2))
                warp, orient_diag = choose_best_table_orientation(warp, rows=rows, cols=cols)

                comps = orient_diag.get("components", 999)
                var_sc = orient_diag.get("variance", float("inf"))
                key = (comps, var_sc, -float(cv2.contourArea(approx)))

                if best is None or key < best:
                    best = key
                    best_warp = warp
                break

    if best_warp is None:
        raise ValueError("Document not detected (4 corners). Try a closer, sharper photo with clear edges.")
    return best_warp, {"orientation_components": best[0], "orientation_variance": best[1]}


# ============================================================
# Scoring / protocol
# ============================================================

facet_bases = {
    "N1": [1],
    "N2": [6],
    "N3": [11],
    "N4": [16],
    "N5": [21],
    "N6": [26],
    "E1": [2],
    "E2": [7],
    "E3": [12],
    "E4": [17],
    "E5": [22],
    "E6": [27],
    "O1": [3],
    "O2": [8],
    "O3": [13],
    "O4": [18],
    "O5": [23],
    "O6": [28],
    "A1": [4],
    "A2": [9],
    "A3": [14],
    "A4": [19],
    "A5": [24],
    "A6": [29],
    "C1": [5],
    "C2": [10],
    "C3": [15],
    "C4": [20],
    "C5": [25],
    "C6": [30],
}

item_to_facette: Dict[int, str] = {}
for fac, bases in facet_bases.items():
    for b in bases:
        for kk in range(0, 240, 30):
            item_to_facette[b + kk] = fac

facettes_to_domain = {
    **{f"N{i}": "N" for i in range(1, 7)},
    **{f"E{i}": "E" for i in range(1, 7)},
    **{f"O{i}": "O" for i in range(1, 7)},
    **{f"A{i}": "A" for i in range(1, 7)},
    **{f"C{i}": "C" for i in range(1, 7)},
}


def _decode_csv_bytes(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def load_scoring_key_from_bytes(csv_bytes: bytes) -> Dict[int, List[int]]:
    text = _decode_csv_bytes(csv_bytes)
    f = io.StringIO(text)
    reader = csv.DictReader(f)

    required = {"item", "FD", "D", "N", "A", "FA"}
    if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
        raise ValueError(f"Missing columns in scoring key CSV. Required: {sorted(required)}")

    key: Dict[int, List[int]] = {}
    for row in reader:
        item = int(row["item"])
        key[item] = [int(row["FD"]), int(row["D"]), int(row["N"]), int(row["A"]), int(row["FA"]) ]

    missing = [i for i in range(1, 241) if i not in key]
    if missing:
        raise ValueError(f"scoring_key.csv incomplete. Missing items (ex): {missing[:12]}")

    bad = [i for i, v in key.items() if len(v) != 5 or any((x < 0 or x > 4) for x in v)]
    if bad:
        raise ValueError(f"scoring_key.csv invalid (values out of 0..4). Items (ex): {bad[:12]}")

    return key


def compute_scores(responses: Dict[int, int], scoring_key: Dict[int, List[int]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    facette_scores = {fac: 0 for fac in facettes_to_domain.keys()}
    for item_id, idx in responses.items():
        if idx == -1:
            continue
        fac = item_to_facette.get(item_id)
        if fac is None:
            continue
        facette_scores[fac] += scoring_key[item_id][idx]

    domain_scores = {d: 0 for d in {"N","E","O","A","C"}}
    for fac, sc in facette_scores.items():
        domain_scores[facettes_to_domain[fac]] += sc
    return facette_scores, domain_scores


def apply_protocol_rules(cfg: OMRConfig, responses: Dict[int, int]) -> Tuple[Dict[int, int], dict]:
    blanks = [i for i, v in responses.items() if v == -1]
    n_blank = len(blanks)
    n_count = sum(1 for v in responses.values() if v == cfg.impute_option_index)

    status = {
        "valid": True,
        "reasons": [],
        "blank_items": blanks,
        "n_blank": n_blank,
        "n_count": n_count,
        "imputed": 0,
    }

    if n_blank >= cfg.max_blank_invalid:
        status["valid"] = False
        status["reasons"].append(f"Too many blank items: {n_blank} (>= {cfg.max_blank_invalid})")

    if n_count >= cfg.max_N_invalid:
        status["valid"] = False
        status["reasons"].append(f"Too many 'N' answers: {n_count} (>= {cfg.max_N_invalid})")

    new_resp = dict(responses)
    if status["valid"] and 0 < n_blank <= cfg.impute_blank_if_leq:
        for item_id in blanks:
            new_resp[item_id] = cfg.impute_option_index
            status["imputed"] += 1

    return new_resp, status


# ============================================================
# Main Scanner
# ============================================================


class OMRScanner:
    def __init__(self, cfg: Optional[OMRConfig] = None, logging_cfg: Optional[LoggingConfig] = None):
        self.cfg = cfg or OMRConfig()
        self.logger = setup_logging(logging_cfg or LoggingConfig())

    def scan_pil(self, pil_img: Image.Image, scoring_key: Dict[int, List[int]]) -> ScanResult:
        scan_id = f"scan-{uuid.uuid4().hex[:10]}"
        log = self.logger

        bgr = pil_to_bgr(pil_img)
        log.info("Loaded image", extra={"scan_id": scan_id, "stage": "load"})

        warped, diag_doc = find_document_warp_auto_rotate(
            bgr,
            target_width=self.cfg.target_width,
            rows=self.cfg.rows,
            cols=self.cfg.cols,
        )
        log.info("Document warped", extra={"scan_id": scan_id, "stage": "warp"})

        thr_inv = binarize_inv(warped)
        table_bbox = find_table_bbox_soft(thr_inv)
        log.info("Table bbox detected", extra={"scan_id": scan_id, "stage": "bbox"})

        responses_raw, meta, overlay, debug_mask = read_responses_from_grid(
            warped_bgr=warped,
            thr_inv_print=thr_inv,
            table_bbox=table_bbox,
            rows=self.cfg.rows,
            cols=self.cfg.cols,
            use_micro_adjust=self.cfg.use_micro_adjust,
            mark_threshold=self.cfg.mark_threshold,
            ambiguity_gap=self.cfg.ambiguity_gap,
            detect_blue=self.cfg.detect_blue,
            detect_black=self.cfg.detect_black,
            black_dark_thresh=self.cfg.black_dark_thresh,
            black_baseline_quantile=self.cfg.black_baseline_quantile,
        )

        responses_final, protocol = apply_protocol_rules(self.cfg, responses_raw)
        facette_scores, domain_scores = compute_scores(responses_final, scoring_key)

        diagnostics = {
            "document": diag_doc,
            "table_bbox": {"x": table_bbox[0], "y": table_bbox[1], "w": table_bbox[2], "h": table_bbox[3]},
            "thresholds": {
                "mark_threshold": self.cfg.mark_threshold,
                "ambiguity_gap": self.cfg.ambiguity_gap,
                "black_dark_thresh": self.cfg.black_dark_thresh,
                "black_baseline_quantile": self.cfg.black_baseline_quantile,
            },
            "stats": {
                "ambiguous": int(sum(1 for m in meta.values() if m.get("ambiguous"))),
                "blank": int(sum(1 for m in meta.values() if m.get("blank"))),
                "low_conf": int(sum(1 for m in meta.values() if float(m.get("confidence", 0)) < 0.55)),
            },
        }

        log.info("Scan completed", extra={"scan_id": scan_id, "stage": "done"})

        return ScanResult(
            scan_id=scan_id,
            responses_raw=responses_raw,
            responses_final=responses_final,
            meta=meta,
            protocol=protocol,
            facette_scores=facette_scores,
            domain_scores=domain_scores,
            overlay_bgr=overlay,
            debug_mask=debug_mask,
            warped_bgr=warped,
            diagnostics=diagnostics,
        )
