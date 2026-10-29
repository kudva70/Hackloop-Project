"""
OpenCV damage analysis for the Repair vs Replace Assistant.

Classical computer vision (no training data needed). It measures:
  - burn / scorch marks     (dark-brown / carbonized colour regions)
  - rust / corrosion        (orange-brown colour regions)
  - crack / scratch lines   (long off-axis edges found with Canny + Hough)
  - photo quality           (blur, brightness, resolution)

These are heuristics. Thresholds are constants at the top: run
`python assistant.py --image ...` on your photos and tune them.
"""
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import cv2
import numpy as np

MAX_SIDE = 900

# Level cut-offs: (low, medium, high)
BURN_LEVELS = (0.003, 0.010, 0.030)   # fraction of the device area
RUST_LEVELS = (0.005, 0.015, 0.040)
# crack: (min off-axis lines, min total length / image diagonal) for low, medium, high
CRACK_LEVELS = ((5, 0.6), (8, 1.2), (14, 2.0))

COLORS = {"burn_marks": (0, 0, 255), "rust_corrosion": (0, 140, 255), "crack_lines": (0, 255, 255)}
LABELS = {"burn_marks": "Burn / scorch marks", "rust_corrosion": "Rust / corrosion",
          "crack_lines": "Crack / scratch lines"}


def load_image(path: str) -> np.ndarray:
    """Read an image; works with spaces and non-ASCII characters in Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def _resize_if_needed(img: np.ndarray) -> np.ndarray:
    scale = MAX_SIDE / max(img.shape[:2])
    if scale < 1:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def _level(value: float, cuts: Tuple[float, float, float]) -> str:
    low, med, high = cuts
    return "high" if value >= high else "medium" if value >= med else "low" if value >= low else "none"


def _blobs(mask: np.ndarray, min_area: int) -> List[np.ndarray]:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def _filter_wood_tone_burns(blobs: List[np.ndarray], v_channel: np.ndarray) -> List[np.ndarray]:
    """
    Filter out wood-tone false positives (e.g. brown tabletops, wood grain, cardboard).
    Genuine electrical scorch/charring has deep carbonization with very dark core pixels (min_v <= 45).
    Warm mid-tone surfaces like wooden desks have min_v > 45 and mean_v > 65.
    """
    genuine_blobs = []
    for c in blobs:
        mask = np.zeros(v_channel.shape, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        pixels = v_channel[mask == 255]
        if len(pixels) < 150:
            continue
        min_v = int(np.min(pixels))
        mean_v = float(np.mean(pixels))
        # Genuine scorch marks must have carbonized core; filter out warm wood tones
        if min_v <= 45 and mean_v <= 65:
            genuine_blobs.append(c)
    return genuine_blobs


def _crack_lines(gray: np.ndarray, diag: float):
    """
    Long straight edges whose direction does NOT match the device's dominant orientation.
    """
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)
    min_len = int(0.06 * diag)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=min_len, maxLineGap=6)
    candidates = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length > 0.45 * diag:
                continue
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 90
            candidates.append((x1, y1, x2, y2, length, angle))
    if not candidates:
        return [], 0.0

    hist = np.zeros(90)
    for *_, length, angle in candidates:
        hist[int(angle) % 90] += length
    smooth = sum(np.roll(hist, k) for k in range(-4, 5))
    dominant = float(np.argmax(smooth))

    off_axis = []
    for x1, y1, x2, y2, length, angle in candidates:
        diff = abs(angle - dominant)
        diff = min(diff, 90 - diff)
        if diff > 12:
            off_axis.append((x1, y1, x2, y2, length))
    total = sum(l[4] for l in off_axis) / diag
    return off_axis, total


def analyze_image(
    image_path: str,
    out_dir: str = "outputs/cv",
    crop_box: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Analyzes visual damage (burns, corrosion, cracks) and photo quality.
    If crop_box [x1, y1, x2, y2] is provided, crops to the detected laptop region
    with padding before analysis to eliminate background/desk false positives.
    """
    raw_img = load_image(image_path)
    h_orig, w_orig = raw_img.shape[:2]

    # Crop to detected laptop region if bounding box is provided
    if crop_box and len(crop_box) == 4:
        x1, y1, x2, y2 = [int(v) for v in crop_box]
        # Add 4% padding around the bounding box
        pad_x = int(0.04 * (x2 - x1))
        pad_y = int(0.04 * (y2 - y1))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w_orig, x2 + pad_x)
        y2 = min(h_orig, y2 + pad_y)
        if x2 - x1 > 40 and y2 - y1 > 40:
            raw_img = raw_img[y1:y2, x1:x2]

    img = _resize_if_needed(raw_img)
    h_img, w_img = img.shape[:2]
    diag = float(np.hypot(h_img, w_img))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ignore plain white studio background so percentages describe the device itself.
    foreground = ~((v > 235) & (s < 30))
    fg_area = max(int(foreground.sum()), 1)
    min_blob = max(250, int(0.003 * fg_area))

    # --- colour based findings ---------------------------------------------------
    # Initial burn candidate mask
    burn_mask = (((h >= 3) & (h <= 25) & (s >= 70) & (v >= 20) & (v <= 115)) & foreground)
    rust_mask = (((h >= 3) & (h <= 17) & (s >= 120) & (v >= 95) & (v <= 210)) & foreground & ~burn_mask)

    raw_burn_blobs = _blobs(burn_mask.astype(np.uint8) * 255, min_blob)
    rust_blobs = _blobs(rust_mask.astype(np.uint8) * 255, min_blob)

    # Filter out small or wood-tone false positives from burn blobs
    burn_blobs = _filter_wood_tone_burns(raw_burn_blobs, v)

    burn_frac = sum(cv2.contourArea(c) for c in burn_blobs) / fg_area
    rust_frac = sum(cv2.contourArea(c) for c in rust_blobs) / fg_area

    # --- crack / scratch lines -----------------------------------------------------
    off_lines, off_ratio = _crack_lines(gray, diag)
    n_off = len(off_lines)
    crack_level = "none"
    for name, (min_n, min_ratio) in zip(("low", "medium", "high"), CRACK_LEVELS):
        if n_off >= min_n and off_ratio >= min_ratio:
            crack_level = name

    # --- photo quality -------------------------------------------------------------
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(v.mean())
    warnings = []
    if blur < 50:
        warnings.append("Photo looks blurry; retake it in focus for reliable results.")
    if brightness < 60:
        warnings.append("Photo is very dark; add light.")
    if brightness > 215:
        warnings.append("Photo is overexposed; reduce glare.")
    if min(h_img, w_img) < 300:
        warnings.append("Photo resolution is low; get closer or use a better camera.")

    findings = []
    burn_level, rust_level = _level(burn_frac, BURN_LEVELS), _level(rust_frac, RUST_LEVELS)
    follow_up_question = None

    if burn_level != "none":
        findings.append({
            "id": "burn_marks",
            "label": LABELS["burn_marks"],
            "level": burn_level,
            "detail": f"{len(burn_blobs)} scorch/charred region(s) covering {burn_frac:.1%} of the device area"
        })
        # If burn finding is below "high" severity, do not force safety hold; add follow-up question
        if burn_level in ("low", "medium"):
            follow_up_question = {
                "question": f"OpenCV visual analysis detected minor dark discoloration ({burn_level} severity). Does the device have actual physical scorch marks, melted plastic, or burnt components?",
                "options": ["No", "Yes", "Not sure"]
            }

    if rust_level != "none":
        findings.append({
            "id": "rust_corrosion",
            "label": LABELS["rust_corrosion"],
            "level": rust_level,
            "detail": f"{len(rust_blobs)} orange-brown region(s) covering {rust_frac:.1%} of the device area"
        })
    if crack_level != "none":
        findings.append({
            "id": "crack_lines",
            "label": LABELS["crack_lines"],
            "level": crack_level,
            "detail": f"{n_off} off-axis line segments (total length {off_ratio:.1f}x image diagonal)"
        })

    # --- annotated image -----------------------------------------------------------
    overlay = img.copy()
    cv2.drawContours(overlay, burn_blobs, -1, COLORS["burn_marks"], -1)
    cv2.drawContours(overlay, rust_blobs, -1, COLORS["rust_corrosion"], -1)
    annotated = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    if burn_level != "none":
        cv2.drawContours(annotated, burn_blobs, -1, COLORS["burn_marks"], 2)
    if rust_level != "none":
        cv2.drawContours(annotated, rust_blobs, -1, COLORS["rust_corrosion"], 2)
    if crack_level != "none":
        for x1, y1, x2, y2, _ in off_lines:
            cv2.line(annotated, (x1, y1), (x2, y2), COLORS["crack_lines"], 2)
    legend = [f"{f['label']}: {f['level'].upper()}" for f in findings] or ["No visible damage detected"]
    cv2.rectangle(annotated, (0, 0), (16 + 9 * max(len(t) for t in legend), 12 + 24 * len(legend)), (0, 0, 0), -1)
    for i, text in enumerate(legend):
        color = (255, 255, 255)
        for f in findings:
            if text.startswith(f["label"]):
                color = COLORS[f["id"]]
        cv2.putText(annotated, text, (8, 22 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(image_path).stem}_cv.jpg"
    ok, buf = cv2.imencode(".jpg", annotated)
    buf.tofile(str(out_path))

    # Only "high" burn severity forces an immediate safety hold; low/medium asks follow-up
    safety_flag = (burn_level == "high")

    return {
        "annotated_image": str(out_path),
        "findings": findings,
        "safety_flag": safety_flag,
        "burn_level": burn_level,
        "follow_up_question": follow_up_question,
        "quality_warnings": warnings,
        "metrics": {
            "burn_pct": round(burn_frac * 100, 2),
            "rust_pct": round(rust_frac * 100, 2),
            "off_axis_lines": n_off,
            "off_axis_length_ratio": round(off_ratio, 2),
            "blur_score": round(blur, 1),
            "brightness": round(brightness, 1)
        },
    }
