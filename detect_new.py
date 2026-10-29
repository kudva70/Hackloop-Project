"""
AI Repair Assistant - Device Detection Module
Supports single image and video detection using the laptop-component model
trained on Roboflow (hosted API).

Only the detection backend changed from the previous version. Local YOLO
weights are no longer loaded; the model runs on Roboflow's servers.

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
import argparse
from PIL import Image, ImageDraw

try:  # optional: read ROBOFLOW_* values from a .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Workspace directories
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

ROBOFLOW_API_URL = os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com")

# None = keep every class the model returns (its classes are laptop parts).
# Set to e.g. {"ram", "battery"} to keep only some parts (case-insensitive).
DEFAULT_TARGET_DEVICES = None

# The model only knows laptop internals, so any detection implies a laptop.
# assistant.py can use this as the device type for the questionnaire/diagnosis.
DETECTED_DEVICE_TYPE = "laptop"

SERVER_CONFIDENCE_FLOOR = 0.05

_client = None


# --------------------------------------------------------------------------
# Detection backend (Roboflow hosted model)
# --------------------------------------------------------------------------
def get_model_id():
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
        # Roboflow's server hides predictions below ~40% by default, which would
        # silently override our --conf. Ask for everything above a low floor and
        # apply the real threshold ourselves in run_inference().
        _client.configure(InferenceConfiguration(
            confidence_threshold=SERVER_CONFIDENCE_FLOOR,
            api_key_transport="header",   # send the key in a header, not the URL/body
        ))
    return _client


def run_inference(source, conf=0.25):
    """
    Send an image path or a numpy frame to the model.
    Returns [{"device": class, "confidence": 0-1, "box": [x1, y1, x2, y2]}] in pixels.
    """
    client, model_id = get_client(), get_model_id()  # config errors surface as-is
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
        # Roboflow returns box CENTER + size; convert to corner coordinates.
        x1 = p["x"] - p["width"] / 2
        y1 = p["y"] - p["height"] / 2
        x2 = p["x"] + p["width"] / 2
        y2 = p["y"] + p["height"] / 2
        detections.append({
            "device": str(p["class"]),
            "confidence": round(confidence, 4),
            "box": [round(v, 2) for v in (x1, y1, x2, y2)],
        })
    if not detections and isinstance(source, str):  # diagnostics for single images
        if predictions:
            best = max(predictions, key=lambda p: p["confidence"])
            print(f"[DEBUG] {len(predictions)} weak prediction(s) below --conf {conf}; "
                  f"best was '{best['class']}' at {best['confidence']:.0%}. Try --conf 0.1")
        else:
            print(f"[DEBUG] Model returned nothing even at {SERVER_CONFIDENCE_FLOOR:.0%} confidence "
                  f"(image size {result.get('image', {}).get('width')}x{result.get('image', {}).get('height')}).")
    return detections


def filter_detections(detections, requested_devices):
    """Keep only requested classes (case-insensitive, substring match). None keeps all."""
    if not requested_devices:
        return detections
    requested = {d.strip().lower() for d in requested_devices}
    return [
        d for d in detections
        if d["device"].lower() in requested or any(r in d["device"].lower() for r in requested)
    ]


# --------------------------------------------------------------------------
# Unchanged helpers
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Image / video detection
# --------------------------------------------------------------------------
def detect_image(image_path, target_devices=DEFAULT_TARGET_DEVICES, conf=0.25):
    """Detect devices in an image, return structured detection metadata, and save annotated image."""
    image_path = resolve_path(image_path)
    detections = filter_detections(run_inference(str(image_path), conf), target_devices)

    output_path = OUTPUT_DIR / "images" / image_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save annotated image using PIL
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    line = max(2, image.width // 300)
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=line)
        draw.text((x1 + 3, max(y1 - 14, 0)), f"{d['device']} {d['confidence']:.0%}", fill=(255, 0, 0))
    image.save(output_path)

    print(f"[SUCCESS] Annotated image saved to: {output_path}")

    return {
        "filename": image_path.name,
        "image_path": str(image_path),
        "detections": detections,
        "device_type": DETECTED_DEVICE_TYPE if detections else None,
        "output": str(output_path),
    }


def detect_video(video_path, target_devices=DEFAULT_TARGET_DEVICES, conf=0.25,
                 frame_step=15, max_calls=40):
    """
    Detect devices in a video and save the annotated output video.
    Every API call costs time and quota, so only every `frame_step`-th frame is sent
    (raised automatically so a video never needs more than `max_calls` calls);
    in-between frames reuse the latest boxes.
    """
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

    return {
        "filename": video_path.name,
        "image_path": str(video_path),
        "detections": detections,
        "device_type": DETECTED_DEVICE_TYPE if detections else None,
        "output": str(out_path),
    }


def _frame_sharpness(frame):
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_sample_frames(video_path, min_count=3, max_count=5, frame_step=15, max_calls=40):
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


def detect_sampled_frames(video_path, target_devices=DEFAULT_TARGET_DEVICES, conf=0.25,
                          frame_step=15, max_calls=40):
    """Run detect_image-style inference on 3–5 evenly spaced video frames.

    Keeps the sharpest frame (Laplacian variance). Roboflow errors print one line
    and continue with empty detections for that frame.
    """
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
    return {
        "filename": video_path.name,
        "image_path": best["image_path"] if best else str(video_path),
        "best_frame": best,
        "frames": frame_results,
        "detections": (best or {}).get("detections", []),
        "device_type": DETECTED_DEVICE_TYPE if any_hits else None,
        "output": best["image_path"] if best else None,
    }


def main():
    parser = argparse.ArgumentParser(description="AI Repair Assistant - Device Detector")
    parser.add_argument("--mode", choices=["image", "video"], help="Input mode: image or video")
    parser.add_argument("--path", help="Path to image or video file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--frame-step", type=int, default=15, help="Video: analyse every Nth frame (default: 15)")
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
            output = detect_video(file_path, conf=args.conf, frame_step=args.frame_step)
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

    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"[ERROR] {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()