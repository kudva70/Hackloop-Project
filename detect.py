"""
AI Repair Assistant - Device Detection Module
Supports single image and video object detection with auto-detection of custom
fine-tuned models (fans, bulbs, switchboards, laptops) and standard YOLO models.
"""

from pathlib import Path
import sys
import argparse
from ultralytics import YOLO
from PIL import Image

# Workspace directories
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
WEIGHTS_DIR = BASE_DIR / "weights"

# Preferred model paths in order of priority:
# 1. Custom fine-tuned repair model (trained on fan, bulb, switchboard, laptop)
# 2. Pretrained YOLOv8 nano model
MODEL_CANDIDATES = [
    WEIGHTS_DIR / "best.pt",
    BASE_DIR / "runs" / "detect" / "repair_model" / "weights" / "best.pt",
    BASE_DIR / "yolov8n.pt",
    BASE_DIR / "yolo26n.pt",
]

def load_repair_model():
    """Find and load the best available YOLO model weight."""
    model_path = None
    for candidate in MODEL_CANDIDATES:
        if candidate.is_file():
            model_path = candidate
            break

    if model_path is None:
        # Fallback to downloading or using standard yolov8n.pt
        model_path = BASE_DIR / "yolov8n.pt"

    print(f"[INFO] Loading detection model from: {model_path}")
    model = YOLO(str(model_path))
    return model

model = load_repair_model()

# Target devices for the Repair Assistant
DEFAULT_TARGET_DEVICES = {"fan", "laptop", "bulb", "light bulb", "switchboard", "mouse", "cell phone", "tv"}

def get_class_filter(model_instance, requested_devices):
    """
    Resolve requested device names against model classes in a case-insensitive way.
    If no requested devices are present in model classes, returns None (no filtering)
    with a clear warning.
    """
    if not requested_devices:
        return None

    requested_lower = {dev.strip().lower() for dev in requested_devices}
    matched_ids = []
    matched_names = []

    for class_id, name in model_instance.names.items():
        name_lower = str(name).strip().lower()
        if name_lower in requested_lower or any(dev in name_lower for dev in requested_lower):
            matched_ids.append(class_id)
            matched_names.append(name)

    if not matched_ids:
        print(f"[WARNING] None of the requested devices {requested_devices} match the model's classes: {list(model_instance.names.values())}.")
        print("[WARNING] Detecting all recognizable objects instead.")
        return None

    return matched_ids


def resolve_path(file_path):
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


def detect_image(image_path, target_devices=DEFAULT_TARGET_DEVICES, conf=0.25):
    """Detect devices in an image, return structured detection metadata, and save annotated image."""
    image_path = resolve_path(image_path)
    classes_to_detect = get_class_filter(model, target_devices)

    results = model.predict(
        source=str(image_path),
        conf=conf,
        classes=classes_to_detect,
        save=False,
        verbose=False,
    )

    detections = []
    output_path = OUTPUT_DIR / "images" / image_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            device_name = result.names[class_id]
            confidence = round(float(box.conf[0].item()), 4)
            bbox = [round(float(v), 2) for v in box.xyxy[0].tolist()]

            detections.append({
                "device": device_name,
                "confidence": confidence,
                "box": bbox,
            })

        # Save annotated image using PIL
        annotated_image = result.plot()
        Image.fromarray(annotated_image[:, :, ::-1]).save(output_path)

    print(f"[SUCCESS] Annotated image saved to: {output_path}")

    return {
        "filename": image_path.name,
        "detections": detections,
        "output": str(output_path),
    }


def detect_video(video_path, target_devices=DEFAULT_TARGET_DEVICES, conf=0.25):
    """Detect devices in a video and save the annotated output video."""
    video_path = resolve_path(video_path)
    classes_to_detect = get_class_filter(model, target_devices)

    results = model.predict(
        source=str(video_path),
        conf=conf,
        classes=classes_to_detect,
        save=True,
        project=str(OUTPUT_DIR),
        name="videos",
        exist_ok=True,
        verbose=False,
        stream=True,
    )

    device_summary = {}

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            name = result.names[class_id]
            confidence = float(box.conf[0].item())

            if name not in device_summary:
                device_summary[name] = {
                    "device": name,
                    "highest_confidence": confidence,
                    "frames_detected": 1,
                }
            else:
                device_summary[name]["frames_detected"] += 1
                device_summary[name]["highest_confidence"] = max(
                    device_summary[name]["highest_confidence"],
                    confidence,
                )

    detections = [
        {
            **item,
            "highest_confidence": round(item["highest_confidence"], 4),
        }
        for item in device_summary.values()
    ]

    print(f"[SUCCESS] Annotated video saved under: {OUTPUT_DIR / 'videos'}")

    return {
        "filename": video_path.name,
        "detections": detections,
    }


def main():
    parser = argparse.ArgumentParser(description="AI Repair Assistant - Device Detector")
    parser.add_argument("--mode", choices=["image", "video"], help="Input mode: image or video")
    parser.add_argument("--path", help="Path to image or video file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--diagnose", action="store_true", help="Launch interactive fault diagnosis after detection")
    args = parser.parse_args()

    mode = args.mode
    file_path = args.path

    if not mode:
        mode = input("Choose input (image or video): ").strip().lower()
    if not file_path:
        file_path = input("Enter file path: ").strip().strip('"')

    try:
        if mode == "image":
            output = detect_image(file_path, conf=args.conf)
        elif mode == "video":
            output = detect_video(file_path, conf=args.conf)
        else:
            raise ValueError("Invalid mode. Please choose 'image' or 'video'.")

        print("\n================ DETECTION RESULTS ================")
        print(f"File: {output['filename']}")

        detected_devices = []
        if not output["detections"]:
            print("No targeted faulty devices detected.")
        else:
            for item in output["detections"]:
                dev_name = item['device']
                detected_devices.append(dev_name)
                conf_val = item.get('confidence') or item.get('highest_confidence')
                print(f"  - Detected: {dev_name} (Confidence: {conf_val})")

        print("===================================================\n")

        # Optional Fault Diagnosis Hook
        should_diagnose = args.diagnose
        if not should_diagnose and detected_devices and sys.stdin.isatty():
            prompt_diag = input("Would you like to diagnose this device for faults? (y/n) [default: y]: ").strip().lower()
            should_diagnose = prompt_diag in ["", "y", "yes"]

        if should_diagnose and detected_devices:
            from assistant import run_pipeline
            run_pipeline(image_path=str(file_path))

    except (FileNotFoundError, ValueError) as error:
        print(f"[ERROR] {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()

