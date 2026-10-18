
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

# Project paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

# YOLO model
model = YOLO(str(BASE_DIR / "yolo26n.pt"))

# test detecting devices
ALLOWED_DEVICES = {"laptop", "cell phone", "mouse","tv"}

# Convert device names to YOLO class IDs
ALLOWED_CLASS_IDS = [
    class_id
    for class_id, name in model.names.items()
    if name in ALLOWED_DEVICES
]


def resolve_path(file_path):
    """Find a file using a path relative to this project."""
    file_path = Path(file_path).expanduser()

    if file_path.is_absolute() and file_path.is_file():
        return file_path.resolve()

    # Try the path relative to computer-vision/
    candidate = BASE_DIR / file_path
    if candidate.is_file():
        return candidate.resolve()

    # Also try the appropriate test folder using the filename
    for folder in ("test_images", "test_videos"):
        candidate = BASE_DIR / folder / file_path.name
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"File not found: {file_path}\n"
        f"Check that the file exists inside {BASE_DIR}"
    )


def get_device_name(result, box):
    class_id = int(box.cls[0].item())
    return result.names[class_id]


def detect_image(image_path):
    """Detect devices and save an annotated image."""
    image_path = resolve_path(image_path)

    results = model.predict(
        source=str(image_path),
        conf=0.15,
        classes=ALLOWED_CLASS_IDS,
        save=False,
        verbose=False,
    )

    detections = []
    output_path = OUTPUT_DIR / "images" / image_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for result in results:
        for box in result.boxes:
            detections.append({
                "device": get_device_name(result, box),
                "confidence": round(float(box.conf[0].item()), 4),
                "box": [
                    round(float(value), 2)
                    for value in box.xyxy[0].tolist()
                ],
            })

        # Draw bounding boxes, device names and confidence scores
        annotated_image = result.plot()
        Image.fromarray(annotated_image[:, :, ::-1]).save(output_path)

    print(f"Annotated image saved to: {output_path}")

    return {
        "filename": image_path.name,
        "detections": detections,
        "output": str(output_path),
    }


def detect_video(video_path):
    """Detect devices in a video and save the annotated video."""
    video_path = resolve_path(video_path)

    results = model.predict(
        source=str(video_path),
        conf=0.25,
        classes=ALLOWED_CLASS_IDS,
        save=True,
        project=str(OUTPUT_DIR),
        name="videos",
        exist_ok=True,
        verbose=False,
    )

    device_summary = {}

    for result in results:
        for box in result.boxes:
            name = get_device_name(result, box)
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
            "highest_confidence": round(
                item["highest_confidence"], 4
            ),
        }
        for item in device_summary.values()
    ]

    print(f"Annotated video saved under: {OUTPUT_DIR / 'videos'}")

    return {
        "filename": video_path.name,
        "detections": detections,
    }


if __name__ == "__main__":
    mode = input("Choose input: image or video: ").strip().lower()
    file_path = input("Enter file path: ").strip().strip('"')

    try:
        if mode == "image":
            output = detect_image(file_path)
        elif mode == "video":
            output = detect_video(file_path)
        else:
            raise ValueError("Please enter image or video.")

        print("\n--- DETECTION RESULTS ---")
        print(f"File: {output['filename']}")

        if not output["detections"]:
            print("No supported devices detected.")
        else:
            for item in output["detections"]:
                print(item)

        print("\nDetection complete.")

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
