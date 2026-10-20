"""
AI Repair Assistant - Real-Time Live Camera / Stream Detection
Supports webcam (index 0) or IP Camera / DroidCam stream URLs.
"""

from pathlib import Path
import sys
import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

# Find best weights
MODEL_CANDIDATES = [
    ROOT_DIR / "weights" / "best.pt",
    BASE_DIR / "weights" / "best.pt",
    ROOT_DIR / "runs" / "detect" / "repair_model" / "weights" / "best.pt",
    ROOT_DIR / "yolov8n.pt",
    BASE_DIR / "yolov8n.pt",
]

def get_model():
    for p in MODEL_CANDIDATES:
        if p.is_file():
            print(f"[INFO] Loading model from: {p}")
            return YOLO(str(p))
    print("[INFO] Loading default YOLOv8n model...")
    return YOLO("yolov8n.pt")


def run_live(source=0, conf=0.35):
    model = get_model()
    print(f"[INFO] Connecting to video source: {source}...")

    # If numeric string passed, convert to int for camera index
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        print("Tip: If using DroidCam, ensure the DroidCam app is running and the IP/port match.")
        print("Tip: If using a laptop webcam, try device index 0.")
        return

    print("[SUCCESS] Connected! Live detection running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[INFO] Video stream ended or no frame received.")
            break

        results = model.predict(frame, conf=conf, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("AI Repair Assistant - Live Device Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    src_input = input("Enter video source (press Enter for default webcam 0, or enter DroidCam URL): ").strip()
    if not src_input:
        src = 0
    else:
        src = src_input
    run_live(source=src)