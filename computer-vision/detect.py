"""
AI Repair Assistant - Detection module (Inside computer-vision directory)
"""

from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_CANDIDATES = [
    ROOT_DIR / "weights" / "best.pt",
    BASE_DIR / "weights" / "best.pt",
    ROOT_DIR / "yolov8n.pt",
    BASE_DIR / "yolov8n.pt",
]

def load_model():
    for p in MODEL_CANDIDATES:
        if p.is_file():
            return YOLO(str(p))
    return YOLO("yolov8n.pt")

model = load_model()


def detect_image(image_path):
    """Detect objects in an image and return structured results."""
    image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = BASE_DIR / image_path

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model.predict(
        source=str(image_path),
        conf=0.25,
        save=True,
        project=str(OUTPUT_DIR),
        name="images",
        exist_ok=True,
        verbose=False,
    )

    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            detections.append({
                "device": result.names[class_id],
                "confidence": round(confidence, 4),
                "box": [round(float(value), 2) for value in box.xyxy[0].tolist()],
            })

    return {
        "filename": image_path.name,
        "detections": detections,
    }


def detect_video(video_path):
    """Detect objects in a video and save an annotated video."""
    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = BASE_DIR / video_path

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    results = model.predict(
        source=str(video_path),
        conf=0.25,
        save=True,
        project=str(OUTPUT_DIR),
        name="videos",
        exist_ok=True,
        verbose=False,
        stream=True,
    )

    detections = []
    for frame_number, result in enumerate(results):
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            detections.append({
                "frame": frame_number,
                "device": result.names[class_id],
                "confidence": round(confidence, 4),
            })

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
            print("No objects detected.")
        else:
            for item in output["detections"]:
                print(item)

        print(f"\nAnnotated results saved to: {OUTPUT_DIR}")

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
