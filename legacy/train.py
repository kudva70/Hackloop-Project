"""
AI Repair Assistant - Model Training & Fine-Tuning Script
Trains the custom 4-class neural network (bulb, fan, laptop, switchboard)
using transfer learning from YOLOv8 nano backbone.
"""

from pathlib import Path
import shutil
import argparse
import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_CONFIG = BASE_DIR / "dataset" / "data.yaml"
ARCH_CONFIG = BASE_DIR / "models" / "repair_assistant_yolo.yaml"
WEIGHTS_DIR = BASE_DIR / "weights"

def train(epochs=10, batch_size=8, imgsz=640, device=None):
    if not DATA_CONFIG.exists():
        raise FileNotFoundError(f"Dataset config not found at {DATA_CONFIG}. Run prepare_dataset.py first.")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine hardware acceleration
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print("AI Repair Assistant - Neural Network Training")
    print(f"Device: {device} | Epochs: {epochs} | Batch: {batch_size} | ImgSz: {imgsz}")
    print("=" * 65)

    # 1. Initialize custom architecture with pretrained weights transfer
    print(f"[INFO] Initializing architecture from: {ARCH_CONFIG}")
    model = YOLO(str(ARCH_CONFIG))

    pretrained_backbone = BASE_DIR / "yolov8n.pt"
    if pretrained_backbone.exists():
        print(f"[INFO] Transferring pretrained weights from: {pretrained_backbone}")
        model = model.load(str(pretrained_backbone))
    else:
        print("[INFO] Downloading and transferring YOLOv8n weights...")
        model = model.load("yolov8n.pt")

    # 2. Run fine-tuning
    results = model.train(
        data=str(DATA_CONFIG),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        workers=0, # Recommended for Windows to avoid multiprocess locks
        project=str(BASE_DIR / "runs" / "detect"),
        name="repair_model",
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True,
        # Augmentation settings tailored for hardware inspection
        degrees=15.0,     # Rotation for ceiling/desk angles
        perspective=0.001,
        flipud=0.2,       # Vertical flip for fans/bulbs
        fliplr=0.5,
        mosaic=0.8,       # Multi-device context
    )

    # 3. Export trained weights to weights/best.pt
    trained_best = BASE_DIR / "runs" / "detect" / "repair_model" / "weights" / "best.pt"
    if trained_best.exists():
        dest_best = WEIGHTS_DIR / "best.pt"
        shutil.copy2(trained_best, dest_best)
        print("\n" + "=" * 65)
        print(f"[SUCCESS] Best trained model saved to: {dest_best}")
        print("=" * 65)
    else:
        print("[WARNING] Could not locate best.pt in runs/detect/repair_model/weights/")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Repair Assistant Detection Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or '0')")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch, imgsz=args.imgsz, device=args.device)

