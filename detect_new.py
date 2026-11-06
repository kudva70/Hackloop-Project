"""
AI Repair Assistant - Device & Hardware Detection Module
Supports single/multi-image and video detection using computer hardware detection models
via Roboflow hosted API with offline heuristic fallback.

Upgraded to support unified Hardware Intelligence:
  - Laptop hardware components
  - Desktop PC hardware components
  - Whole-computer cabinet scanning (closed opaque cabinet vs. open interior)
  - Partially visible and uncertain component tracking
  - Multi-image inventory merging
  - Clear unsupported-class handling (does not invent ungrounded detections)

Setup:
    pip install inference-sdk pillow opencv-python python-dotenv
    Put these in a .env file (add .env to .gitignore) or in your environment:
        ROBOFLOW_API_KEY=your_private_key
        ROBOFLOW_MODEL_ID=your-project/3
"""

from pathlib import Path
import os
import sys
import math
import json
import argparse
from typing import Dict, List, Optional, Any, Tuple, Set
from PIL import Image, ImageDraw

try:  # optional: read ROBOFLOW_* values from a .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from hardware_profile import (
    HardwareProfile,
    ComponentItem,
    VisibilityStatus,
    CabinetStatus,
    LAPTOP_STANDARD_COMPONENTS,
    DESKTOP_STANDARD_COMPONENTS,
)

# Workspace directories
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

ROBOFLOW_API_URL = os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com")

DEFAULT_TARGET_DEVICES = None

# Default device type fallback when detection alone is ambiguous
DETECTED_DEVICE_TYPE = "laptop"

SERVER_CONFIDENCE_FLOOR = 0.05
UNCERTAIN_CONFIDENCE_FLOOR = 0.15

_client = None


# --------------------------------------------------------------------------
# Configurable model class mappings & component category definitions
# --------------------------------------------------------------------------
DEFAULT_ROBOFLOW_CLASS_MAP: Dict[str, Tuple[str, str, List[str]]] = {
    # Raw Roboflow class -> (canonical_category, display_name, applicable_devices)
    "motherboard": ("motherboard", "Motherboard / Mainboard", ["laptop", "desktop"]),
    "mainboard": ("motherboard", "Motherboard / Mainboard", ["laptop", "desktop"]),
    "ram": ("ram", "RAM Module", ["laptop", "desktop"]),
    "ram slot": ("ram", "RAM Slot / Memory", ["laptop", "desktop"]),
    "memory": ("ram", "RAM Module", ["laptop", "desktop"]),
    "sodimm": ("ram", "Laptop SODIMM RAM", ["laptop"]),
    "battery": ("battery", "Internal Li-ion Battery", ["laptop"]),
    "fan": ("cpu_fan", "Cooling Fan", ["laptop", "desktop"]),
    "cooling fan": ("cpu_fan", "Cooling Fan", ["laptop", "desktop"]),
    "cpu fan": ("cpu_fan", "CPU Cooler Fan", ["laptop", "desktop"]),
    "case fan": ("case_fans", "Chassis Fan", ["desktop"]),
    "heatsink": ("heatsink", "Thermal Heatsink", ["laptop", "desktop"]),
    "heat sink": ("heatsink", "Thermal Heatsink", ["laptop", "desktop"]),
    "cpu": ("cpu", "CPU Processor", ["laptop", "desktop"]),
    "processor": ("cpu", "CPU Processor", ["laptop", "desktop"]),
    "cpu cooler": ("cpu_cooler", "CPU Cooler", ["desktop"]),
    "gpu": ("gpu", "Graphics Processing Unit (GPU)", ["laptop", "desktop"]),
    "graphics card": ("gpu", "Graphics Card", ["desktop"]),
    "graphics": ("gpu", "Graphics Card", ["laptop", "desktop"]),
    "ssd": ("ssd", "Solid State Drive (SSD)", ["laptop", "desktop"]),
    "sata ssd": ("ssd", "2.5\" SATA SSD", ["laptop", "desktop"]),
    "m.2 ssd": ("ssd", "M.2 NVMe SSD", ["laptop", "desktop"]),
    "hdd": ("hdd", "Hard Disk Drive (HDD)", ["laptop", "desktop"]),
    "hard drive": ("hdd", "Hard Disk Drive (HDD)", ["laptop", "desktop"]),
    "psu": ("psu", "Power Supply Unit (ATX PSU)", ["desktop"]),
    "power supply": ("psu", "Power Supply Unit (ATX PSU)", ["desktop"]),
    "charging port": ("charging_port", "Charging Port / DC Jack", ["laptop"]),
    "dc jack": ("charging_port", "Charging Port / DC Jack", ["laptop"]),
    "screen": ("display", "Display Panel", ["laptop"]),
    "display": ("display", "Display Panel", ["laptop"]),
    "keyboard": ("keyboard", "Keyboard Deck", ["laptop"]),
    "touchpad": ("touchpad", "Touchpad / Trackpad", ["laptop"]),
    "trackpad": ("touchpad", "Touchpad / Trackpad", ["laptop"]),
    "cabinet": ("cabinet", "PC Cabinet / Chassis", ["desktop"]),
    "pc case": ("cabinet", "PC Cabinet / Chassis", ["desktop"]),
    "case": ("cabinet", "PC Cabinet / Chassis", ["desktop"]),
    "chassis": ("cabinet", "PC Cabinet / Chassis", ["desktop"]),
    "desktop cabinet": ("cabinet", "PC Cabinet / Chassis", ["desktop"]),
    "cable": ("cables", "Internal Cables / Connectors", ["desktop"]),
    "cables": ("cables", "Internal Cables / Connectors", ["desktop"]),
    "power cable": ("cables", "Power Cable / Connector", ["desktop"]),
}


def get_class_map() -> Dict[str, Tuple[str, str, List[str]]]:
    """Returns the active Roboflow class map, supporting environment overrides."""
    override = os.environ.get("ROBOFLOW_CLASS_MAP_OVERRIDE")
    if override:
        try:
            custom = json.loads(override)
            merged = dict(DEFAULT_ROBOFLOW_CLASS_MAP)
            merged.update(custom)
            return merged
        except Exception:
            pass
    return DEFAULT_ROBOFLOW_CLASS_MAP


def get_supported_model_categories() -> Set[str]:
    """Categories that the detector model can recognize under current class mapping."""
    return {v[0] for v in get_class_map().values()}


# --------------------------------------------------------------------------
# Detection backend (Roboflow hosted model)
# --------------------------------------------------------------------------
def get_model_id() -> str:
    model_id = os.environ.get("ROBOFLOW_MODEL_ID")
    if not model_id:
        raise RuntimeError("ROBOFLOW_MODEL_ID is not set (e.g. 'my-project/3').")
    return model_id.strip()


def get_client():
    """Create the Roboflow client on first use, so importing this file never fails."""
    global _client
    if _client is None:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError("ROBOFLOW_API_KEY is not set. Put it in .env or your environment.")
        from inference_sdk import InferenceHTTPClient, InferenceConfiguration
        _client = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=api_key.strip())
        _client.configure(InferenceConfiguration(
            confidence_threshold=SERVER_CONFIDENCE_FLOOR,
            api_key_transport="header",
        ))
    return _client


def run_inference(source, conf: float = 0.25) -> List[Dict[str, Any]]:
    """
    Send an image path or a numpy frame to the model.
    Returns [{"device": class, "confidence": 0-1, "box": [x1, y1, x2, y2]}] in pixels.
    """
    client, model_id = get_client(), get_model_id()
    try:
        result = client.infer(source, model_id=model_id)
    except Exception as error:
        raise RuntimeError(f"Roboflow request failed: {error}") from error

    detections = []
    predictions = result.get("predictions", [])
    for p in predictions:
        confidence = float(p["confidence"])
        if confidence < conf:
            continue
        x1 = p["x"] - p["width"] / 2
        y1 = p["y"] - p["height"] / 2
        x2 = p["x"] + p["width"] / 2
        y2 = p["y"] + p["height"] / 2
        detections.append({
            "device": str(p["class"]),
            "confidence": round(confidence, 4),
            "box": [round(v, 2) for v in (x1, y1, x2, y2)],
        })
    if not detections and isinstance(source, str):
        if predictions:
            best = max(predictions, key=lambda p: p["confidence"])
            print(f"[DEBUG] {len(predictions)} weak prediction(s) below --conf {conf}; "
                  f"best was '{best['class']}' at {best['confidence']:.0%}. Try --conf 0.1")
        else:
            print(f"[DEBUG] Model returned nothing even at {SERVER_CONFIDENCE_FLOOR:.0%} confidence.")
    return detections


def filter_detections(detections: List[Dict[str, Any]], requested_devices: Optional[Any]) -> List[Dict[str, Any]]:
    """Keep only requested classes (case-insensitive, substring match). None keeps all."""
    if not requested_devices:
        return detections
    requested = {d.strip().lower() for d in requested_devices}
    return [
        d for d in detections
        if d["device"].lower() in requested or any(r in d["device"].lower() for r in requested)
    ]


# --------------------------------------------------------------------------
# Path resolution helpers
# --------------------------------------------------------------------------
def resolve_path(file_path: Any) -> Path:
    """Find an image or video file using relative or absolute project paths."""
    file_path = Path(file_path).expanduser()

    if file_path.is_absolute() and file_path.is_file():
        return file_path.resolve()

    candidates = [
        BASE_DIR / file_path,
        BASE_DIR / "computer-vision" / file_path,
        BASE_DIR / "computer-vision" / file_path.name,
        BASE_DIR / "outputs" / file_path,
        BASE_DIR / "test_images" / file_path.name,
        BASE_DIR / "test_videos" / file_path.name,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"File not found: {file_path}\nSearched locations under {BASE_DIR}"
    )


# --------------------------------------------------------------------------
# Geometric occlusion & clipping analysis
# --------------------------------------------------------------------------
def _box_area(box: List[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_intersection_ratio(box_a: List[float], box_b: List[float]) -> float:
    """Returns fraction of the smaller box that is covered by the intersection."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    min_area = min(_box_area(box_a), _box_area(box_b))
    return (inter / min_area) if min_area > 0 else 0.0


def _is_box_clipped(box: List[float], width: int, height: int, margin_ratio: float = 0.015) -> bool:
    """Checks if bounding box touches or clips the image frame edge."""
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio
    return (
        box[0] <= margin_x
        or box[1] <= margin_y
        or box[2] >= (width - margin_x)
        or box[3] >= (height - margin_y)
    )


# --------------------------------------------------------------------------
# Component mapping & device classification
# --------------------------------------------------------------------------
def map_detection_to_component(raw_class: str) -> Tuple[str, str, List[str]]:
    """Maps a raw detection class to (canonical_category, display_name, applicable_devices)."""
    clean = " ".join(raw_class.strip().lower().replace("_", " ").replace("-", " ").split())
    cmap = get_class_map()
    if clean in cmap:
        return cmap[clean]
    for key, val in cmap.items():
        if key in clean or clean in key:
            return val
    # Fallback: preserve raw label without inventing unknown mappings
    return (clean.replace(" ", "_"), raw_class.title(), ["laptop", "desktop"])


def infer_device_type(detections: List[Dict[str, Any]], user_selected_type: Optional[str] = None) -> str:
    """Infers whether the detected hardware belongs to a Laptop or Desktop PC."""
    if user_selected_type and user_selected_type.strip().lower() in ("laptop", "desktop"):
        return user_selected_type.strip().lower()

    desktop_indicators = {"psu", "case_fans", "cabinet", "cpu_cooler"}
    laptop_indicators = {"battery", "touchpad", "charging_port", "keyboard"}

    desktop_score = 0
    laptop_score = 0

    for d in detections:
        cat, _, applicable = map_detection_to_component(d.get("device", ""))
        if cat in desktop_indicators:
            desktop_score += 2
        elif cat in laptop_indicators:
            laptop_score += 2
        elif "desktop" in applicable and "laptop" not in applicable:
            desktop_score += 1
        elif "laptop" in applicable and "desktop" not in applicable:
            laptop_score += 1

    if desktop_score > laptop_score:
        return "desktop"
    if laptop_score > desktop_score:
        return "laptop"
    return DETECTED_DEVICE_TYPE


def analyze_cabinet_status(
    detections: List[Dict[str, Any]],
    device_type: str,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Analyse whether the image depicts a closed cabinet, an open interior, or a laptop.
    Rule: 'A closed opaque cabinet must not be treated as proof of its internal components.'
    """
    dev = device_type.lower()
    categories = [map_detection_to_component(d.get("device", ""))[0] for d in detections]

    if dev == "desktop":
        internal_parts = {"motherboard", "ram", "cpu", "cpu_cooler", "gpu", "ssd", "hdd", "psu", "cables"}
        has_internal = any(c in internal_parts for c in categories)
        has_cabinet = any(c == "cabinet" for c in categories)

        if not has_internal:
            # Closed cabinet or exterior case view
            return (
                CabinetStatus.CLOSED_CABINET.value,
                "Desktop cabinet appears closed. Internal components cannot be visually verified or ruled out. "
                "Do not assume NOT_DETECTED means NOT_PRESENT."
            )
        else:
            return (
                CabinetStatus.OPEN_INTERIOR.value,
                "Desktop interior with visible components detected (open side panel view)."
            )
    elif dev == "laptop":
        teardown_parts = {"motherboard", "battery", "cpu_fan", "heatsink", "ram"}
        if any(c in teardown_parts for c in categories):
            return (
                CabinetStatus.LAPTOP_TEARDOWN.value,
                "Laptop bottom chassis removed; internal motherboard components are visible."
            )
        else:
            return (
                CabinetStatus.LAPTOP_OPEN.value,
                "Laptop exterior, screen, or keyboard deck view."
            )

    return (CabinetStatus.UNKNOWN.value, "Cabinet view cannot be determined with current detections.")


# --------------------------------------------------------------------------
# Single & Multi-image Hardware Intelligence
# --------------------------------------------------------------------------
def analyze_hardware_image(
    image_path: Any,
    device_type: Optional[str] = None,
    conf: float = 0.25,
    target_devices: Optional[Any] = DEFAULT_TARGET_DEVICES,
) -> Dict[str, Any]:
    """
    Analyse a single image to build a structured HardwareProfile with visibility status,
    cabinet scanning, and close-up recommendations.
    """
    resolved_path = resolve_path(image_path)
    image = Image.open(resolved_path).convert("RGB")
    width, height = image.size

    # Run inference with server floor
    raw_detections = filter_detections(run_inference(str(resolved_path), conf=SERVER_CONFIDENCE_FLOOR), target_devices)

    # Classify device type
    dev_type = infer_device_type(raw_detections, user_selected_type=device_type)
    cab_status, cab_note = analyze_cabinet_status(raw_detections, dev_type, width, height)

    profile = HardwareProfile(device_type=dev_type)
    profile.images_analyzed.append(resolved_path.name)
    profile.set_cabinet_status(cab_status, notes=cab_note)

    # Track supported categories in current model mapping
    supported_model_cats = get_supported_model_categories()
    std_components = DESKTOP_STANDARD_COMPONENTS if dev_type == "desktop" else LAPTOP_STANDARD_COMPONENTS

    for cat_id, cat_name in std_components.items():
        if cat_id not in supported_model_cats:
            profile.mark_unsupported_category(
                category=cat_id,
                name=cat_name,
                reason="Current model weights do not have a trained class for this category"
            )

    # Process detections into ComponentItems
    filtered_detections = []
    for i, d in enumerate(raw_detections):
        confidence = float(d.get("confidence", 0.0))
        box = d.get("box", [0, 0, 0, 0])
        cat, display_name, applicable = map_detection_to_component(d["device"])

        # Skip detections below confidence floor
        if confidence < SERVER_CONFIDENCE_FLOOR:
            continue

        # Check clipping against image edge
        clipped = _is_box_clipped(box, width, height)

        # Check occlusion against other bounding boxes
        occluded = False
        for j, other in enumerate(raw_detections):
            if i != j:
                other_box = other.get("box", [0, 0, 0, 0])
                if _box_intersection_ratio(box, other_box) > 0.35:
                    occluded = True
                    break

        # Assign VisibilityStatus
        if confidence < conf:
            status = VisibilityStatus.UNCERTAIN.value
            note = f"Detected with low confidence ({confidence:.1%}). Close-up photograph recommended."
        elif clipped or occluded:
            status = VisibilityStatus.PARTIALLY_VISIBLE.value
            note = "Component is partially visible, occluded, or clipped at image boundary."
        else:
            status = VisibilityStatus.DETECTED.value
            note = f"Clearly visible with {confidence:.1%} confidence."

        profile.add_component_detection(
            category=cat,
            name=display_name,
            confidence=confidence,
            bounding_box=box,
            image_ref=resolved_path.name,
            visibility_status=status,
            source="roboflow",
            supported_by_model=True,
            notes=note,
        )

        d_record = dict(d)
        d_record["category"] = cat
        d_record["visibility_status"] = status
        filtered_detections.append(d_record)

    # Generate annotated image
    output_path = OUTPUT_DIR / "images" / resolved_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    draw = ImageDraw.Draw(image)
    line = max(2, width // 300)
    for d in filtered_detections:
        x1, y1, x2, y2 = d["box"]
        color = (255, 0, 0) if d["visibility_status"] == VisibilityStatus.DETECTED.value else (255, 140, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line)
        draw.text((x1 + 3, max(y1 - 14, 0)), f"{d['device']} {d['confidence']:.0%}", fill=color)
    image.save(output_path)

    return {
        "filename": resolved_path.name,
        "image_path": str(resolved_path),
        "detections": filtered_detections,
        "device_type": dev_type,
        "cabinet_status": cab_status,
        "hardware_profile": profile.to_dict(),
        "close_up_requests": profile.get_close_up_requests(),
        "output": str(output_path),
    }


def analyze_hardware_images(
    image_paths: List[Any],
    device_type: Optional[str] = None,
    conf: float = 0.25,
    target_devices: Optional[Any] = DEFAULT_TARGET_DEVICES,
) -> Dict[str, Any]:
    """
    Analyse multiple photographs (e.g. overall cabinet + open interior + component close-ups)
    and merge observations into a unified HardwareProfile.
    """
    if not image_paths:
        empty_profile = HardwareProfile(device_type=device_type or "unknown")
        return {
            "device_type": empty_profile.device_type,
            "cabinet_status": CabinetStatus.UNKNOWN.value,
            "hardware_profile": empty_profile.to_dict(),
            "detections": [],
            "images_analyzed": [],
            "close_up_requests": ["No images provided for hardware inventory analysis."],
            "annotated_images": [],
        }

    # Normalize to list
    paths = [image_paths] if isinstance(image_paths, (str, Path)) else list(image_paths)

    merged_profile = None
    all_detections = []
    annotated_outputs = []
    resolved_paths = []

    for img_p in paths:
        try:
            res = analyze_hardware_image(
                image_path=img_p,
                device_type=device_type,
                conf=conf,
                target_devices=target_devices,
            )
            resolved_paths.append(res["image_path"])
            annotated_outputs.append(res["output"])
            all_detections.extend(res["detections"])

            sub_profile = HardwareProfile.from_dict(res["hardware_profile"])
            if merged_profile is None:
                merged_profile = sub_profile
            else:
                merged_profile.merge_profile(sub_profile)

        except Exception as exc:
            print(f"[WARNING] Could not analyse image '{img_p}': {exc}")

    if merged_profile is None:
        merged_profile = HardwareProfile(device_type=device_type or "unknown")

    return {
        "device_type": merged_profile.device_type,
        "cabinet_status": merged_profile.cabinet_status,
        "hardware_profile": merged_profile.to_dict(),
        "detections": all_detections,
        "images_analyzed": resolved_paths,
        "close_up_requests": merged_profile.get_close_up_requests(),
        "annotated_images": annotated_outputs,
    }


# --------------------------------------------------------------------------
# Backward-compatible detection interfaces
# --------------------------------------------------------------------------
def detect_image(
    image_path: Any,
    target_devices: Optional[Any] = DEFAULT_TARGET_DEVICES,
    conf: float = 0.25,
    device_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Detect devices in an image, return structured detection metadata, and save annotated image."""
    result = analyze_hardware_image(
        image_path=image_path,
        device_type=device_type,
        conf=conf,
        target_devices=target_devices,
    )
    # Return backward-compatible contract expected by existing callers
    return {
        "filename": result["filename"],
        "image_path": result["image_path"],
        "detections": result["detections"],
        "device_type": result["device_type"] if result["detections"] else None,
        "output": result["output"],
        "hardware_profile": result["hardware_profile"],
        "cabinet_status": result["cabinet_status"],
        "close_up_requests": result["close_up_requests"],
    }


def detect_video(
    video_path: Any,
    target_devices: Optional[Any] = DEFAULT_TARGET_DEVICES,
    conf: float = 0.25,
    frame_step: int = 15,
    max_calls: int = 40,
) -> Dict[str, Any]:
    """Detect devices in a video and save the annotated output video."""
    import cv2

    video_path = resolve_path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total:
        frame_step = max(frame_step, math.ceil(total / max_calls))

    out_dir = OUTPUT_DIR / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    device_summary = {}
    current = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % frame_step == 0:
            current = filter_detections(run_inference(frame, conf), target_devices)
            for d in current:
                name = d["device"]
                if name not in device_summary:
                    device_summary[name] = {
                        "device": name,
                        "highest_confidence": d["confidence"],
                        "frames_detected": 1,
                    }
                else:
                    device_summary[name]["frames_detected"] += 1
                    device_summary[name]["highest_confidence"] = max(
                        device_summary[name]["highest_confidence"], d["confidence"]
                    )
        for d in current:
            x1, y1, x2, y2 = (int(v) for v in d["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{d['device']} {d['confidence']:.0%}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        writer.write(frame)
        index += 1

    capture.release()
    writer.release()

    detections = [
        {**item, "highest_confidence": round(item["highest_confidence"], 4)}
        for item in device_summary.values()
    ]

    print(f"[SUCCESS] Annotated video saved to: {out_path} "
          f"(analysed every {frame_step} frames)")

    dev_type = infer_device_type(detections) if detections else None
    return {
        "filename": video_path.name,
        "image_path": str(video_path),
        "detections": detections,
        "device_type": dev_type,
        "output": str(out_path),
    }


def _frame_sharpness(frame) -> float:
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_sample_frames(video_path: Any, min_count: int = 3, max_count: int = 5,
                           frame_step: int = 15, max_calls: int = 40) -> Tuple[List[Dict[str, Any]], Path]:
    """Extract 3–5 evenly spaced frames. Roboflow call count stays within frame_step/max_calls."""
    import cv2

    video_path = resolve_path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    n = min(max_count, max_calls)
    if total:
        max_by_step = max(1, math.ceil(total / max(frame_step, 1)))
        n = max(1, min(n, max_by_step, total, max_count))
        if n == 1:
            indices = [0] if total == 1 else [total // 2]
        else:
            indices = [int(round(i * (total - 1) / (n - 1))) for i in range(n)]
        frames = []
        for idx in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = capture.read()
            if ok:
                frames.append({"index": idx, "frame": frame})
        capture.release()
        return frames, video_path

    frames = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append({"index": index, "frame": frame})
        index += 1
    capture.release()
    if not frames:
        return [], video_path
    n = min(max_count, max(min_count, 1), len(frames), max_calls)
    if n == 1:
        chosen = [frames[len(frames) // 2]]
    else:
        picks = [int(round(i * (len(frames) - 1) / (n - 1))) for i in range(n)]
        chosen = [frames[i] for i in picks]
    return chosen, video_path


def detect_sampled_frames(
    video_path: Any,
    target_devices: Optional[Any] = DEFAULT_TARGET_DEVICES,
    conf: float = 0.25,
    frame_step: int = 15,
    max_calls: int = 40,
) -> Dict[str, Any]:
    """Run detect_image-style inference on 3–5 evenly spaced video frames."""
    import cv2

    sampled, video_path = extract_sample_frames(
        video_path, min_count=3, max_count=5, frame_step=frame_step, max_calls=max_calls
    )
    out_dir = OUTPUT_DIR / "video_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_results = []
    best = None
    for item in sampled:
        sharpness = _frame_sharpness(item["frame"])
        detections = []
        try:
            detections = filter_detections(run_inference(item["frame"], conf), target_devices)
        except Exception as error:
            print(f"[WARNING] Roboflow request failed: {error}")
        image_path = out_dir / f"{video_path.stem}_frame_{item['index']}.jpg"
        ok, buf = cv2.imencode(".jpg", item["frame"])
        if ok:
            buf.tofile(str(image_path))
        rec = {
            "index": item["index"],
            "sharpness": sharpness,
            "detections": detections,
            "image_path": str(image_path),
        }
        frame_results.append(rec)
        if best is None or sharpness > best["sharpness"]:
            best = rec

    any_hits = any(r["detections"] for r in frame_results)
    best_detections = (best or {}).get("detections", [])
    dev_type = infer_device_type(best_detections) if any_hits else None

    # Build hardware profile for best frame if hits found
    profile = None
    if best and best_detections:
        profile_res = analyze_hardware_image(best["image_path"], device_type=dev_type, conf=conf)
        profile = profile_res.get("hardware_profile")

    return {
        "filename": video_path.name,
        "image_path": best["image_path"] if best else str(video_path),
        "best_frame": best,
        "frames": frame_results,
        "detections": best_detections,
        "device_type": dev_type,
        "hardware_profile": profile,
        "output": best["image_path"] if best else None,
    }


def main():
    parser = argparse.ArgumentParser(description="AI Repair Assistant - Device & Hardware Detector")
    parser.add_argument("--mode", choices=["image", "video"], help="Input mode: image or video")
    parser.add_argument("--path", help="Path to single image or video file")
    parser.add_argument("--images", nargs="+", help="Paths to multiple computer photographs")
    parser.add_argument("--device", choices=["laptop", "desktop"], help="Target device type (laptop or desktop)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--frame-step", type=int, default=15, help="Video: analyse every Nth frame (default: 15)")
    parser.add_argument("--diagnose", action="store_true", help="Launch interactive fault diagnosis after detection")
    args = parser.parse_args()

    mode = args.mode
    file_path = args.path
    image_paths = args.images

    if not mode:
        if image_paths or (file_path and Path(file_path).suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]):
            mode = "image"
        else:
            mode = input("Choose input (image or video): ").strip().lower()

    try:
        if mode == "image":
            if image_paths:
                output = analyze_hardware_images(image_paths, device_type=args.device, conf=args.conf)
                print("\n================ MULTI-IMAGE HARDWARE SCAN ================")
                print(f"Device Category: {output['device_type'].upper()}")
                print(f"Cabinet Status : {output['cabinet_status']}")
                print(f"Images Analyzed: {len(output['images_analyzed'])}")
                hp = HardwareProfile.from_dict(output["hardware_profile"])
                detected = hp.get_detected_components()
                if detected:
                    print("\n[Confirmed Visible Components]")
                    for c in detected:
                        print(f"  * {c.component_name} ({c.confidence:.0%}) - {c.notes}")
                partial = hp.get_partially_visible_components()
                if partial:
                    print("\n[Partially Visible / Occluded Components]")
                    for c in partial:
                        print(f"  * {c.component_name} ({c.confidence:.0%}) - {c.notes}")
                if output["close_up_requests"]:
                    print("\n[Close-up Photo Requests]")
                    for req in output["close_up_requests"]:
                        print(f"  ! {req}")
                print("===========================================================\n")
            else:
                if not file_path:
                    file_path = input("Enter image path: ").strip().strip('"')
                output = detect_image(file_path, conf=args.conf, device_type=args.device)
                print("\n================ DETECTION RESULTS ================")
                print(f"File: {output['filename']}")
                print(f"Device Type: {str(output.get('device_type')).upper()}")
                print(f"Cabinet Status: {output.get('cabinet_status')}")
                if not output["detections"]:
                    print("No targeted components detected above confidence threshold.")
                else:
                    for item in output["detections"]:
                        dev_name = item['device']
                        conf_val = item.get('confidence')
                        status = item.get('visibility_status', 'DETECTED')
                        print(f"  - Detected: {dev_name} (Confidence: {conf_val:.0%}, Status: {status})")
                print("===================================================\n")

        elif mode == "video":
            if not file_path:
                file_path = input("Enter video path: ").strip().strip('"')
            output = detect_video(file_path, conf=args.conf, frame_step=args.frame_step)
            print("\n================ VIDEO DETECTION RESULTS ================")
            print(f"File: {output['filename']}")
            for item in output["detections"]:
                dev_name = item['device']
                conf_val = item.get('highest_confidence')
                print(f"  - Detected: {dev_name} (Highest Confidence: {conf_val:.0%})")
            print("=========================================================\n")
        else:
            raise ValueError("Invalid mode. Please choose 'image' or 'video'.")

        # Optional Fault Diagnosis Hook
        if args.diagnose:
            from assistant import run_pipeline
            first_path = image_paths[0] if image_paths else file_path
            run_pipeline(image_path=str(first_path), device_type=args.device)

    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"[ERROR] {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
