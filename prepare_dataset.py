"""
AI Repair Assistant - Dataset Preparation & Generation Pipeline (5 Classes)
Prepares custom dataset for:
  Class 0: bulb
  Class 1: fan
  Class 2: laptop
  Class 3: mouse
  Class 4: switchboard

Extracts existing laptop and mouse images from the local dataset and generates synthetic/augmented
training samples for fans, light bulbs, and switchboards so training can be run immediately.
"""

from pathlib import Path
import shutil
import random
import numpy as np
import cv2
import yaml

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
CV_DIR = BASE_DIR / "computer-vision"

CLASSES = ["bulb", "fan", "laptop", "mouse", "switchboard"]

def create_directory_structure():
    # Clean previous dataset images/labels to avoid stale class IDs
    for split in ["train", "val", "test"]:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"
        if img_dir.exists():
            shutil.rmtree(img_dir)
        if lbl_dir.exists():
            shutil.rmtree(lbl_dir)
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)


def extract_existing_devices():
    """Extract laptop and mouse images, remapping class IDs:
       Original Class 4 ('Laptop') -> 2
       Original Class 5 ('MOUSE')  -> 3
    """
    print("[INFO] Extracting existing laptop & mouse samples from computer-vision dataset...")
    splits_map = {"train": "train", "valid": "val", "test": "test"}

    extracted_laptops = 0
    extracted_mice = 0
    total_images = 0

    for cv_split, new_split in splits_map.items():
        img_dir = CV_DIR / cv_split / "images"
        lbl_dir = CV_DIR / cv_split / "labels"

        if not (img_dir.exists() and lbl_dir.exists()):
            continue

        for lbl_path in lbl_dir.glob("*.txt"):
            remapped_boxes = []
            has_laptop = False
            has_mouse = False

            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    # In computer-vision data.yaml: 4 is 'Laptop', 5 is 'MOUSE'
                    if cls_id == 4:
                        remapped_boxes.append(f"2 {' '.join(parts[1:])}\n")
                        has_laptop = True
                    elif cls_id == 5:
                        remapped_boxes.append(f"3 {' '.join(parts[1:])}\n")
                        has_mouse = True

            if remapped_boxes:
                img_path = img_dir / f"{lbl_path.stem}.jpg"
                if not img_path.exists():
                    for ext in [".png", ".jpeg", ".JPG"]:
                        alt = img_dir / f"{lbl_path.stem}{ext}"
                        if alt.exists():
                            img_path = alt
                            break

                if img_path.exists():
                    target_img = DATASET_DIR / new_split / "images" / f"extracted_{img_path.name}"
                    target_lbl = DATASET_DIR / new_split / "labels" / f"extracted_{lbl_path.stem}.txt"
                    shutil.copy2(img_path, target_img)
                    with open(target_lbl, "w") as f:
                        f.writelines(remapped_boxes)
                    total_images += 1
                    if has_laptop:
                        extracted_laptops += 1
                    if has_mouse:
                        extracted_mice += 1

    print(f"[SUCCESS] Extracted {total_images} real images ({extracted_laptops} laptops, {extracted_mice} mice).")


def draw_synthetic_sample(device_type):
    """
    Renders realistic synthetic training samples for bulb, fan, or switchboard
    with varied orientations, lighting, backgrounds, and returns image + bbox.
    """
    w, h = 640, 640
    # Background: random indoor room colors (walls, ceilings, desks)
    bg_color = [random.randint(160, 240), random.randint(160, 240), random.randint(160, 240)]
    img = np.full((h, w, 3), bg_color, dtype=np.uint8)

    # Add background noise/texture
    noise = np.random.normal(0, 8, (h, w, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if device_type == "bulb":
        # Draw a distinctive light bulb with prominent metal screw base and socket
        cx = random.randint(220, 420)
        cy = random.randint(180, 320)
        radius = random.randint(55, 95)

        # Screw base (silver/bronze threaded cylinder)
        base_h = int(radius * 0.9)
        base_w = int(radius * 0.75)
        base_top = cy + radius - 10
        base_color = (random.randint(90, 130), random.randint(100, 150), random.randint(120, 170))
        cv2.rectangle(img, (cx - base_w // 2, base_top), (cx + base_w // 2, base_top + base_h), base_color, -1)

        # Thread ridges
        for t in range(base_top, base_top + base_h, 7):
            cv2.line(img, (cx - base_w // 2, t), (cx + base_w // 2, t), (40, 40, 40), 2)

        # Bottom contact point
        cv2.circle(img, (cx, base_top + base_h + 4), 6, (30, 30, 30), -1)

        # Glass globe (frosted white / warm yellow pear shape)
        globe_color = random.choice([
            (240, 250, 255),  # Daylight LED
            (200, 240, 255),  # Warm yellow
            (225, 225, 230),  # Frosted
        ])
        cv2.circle(img, (cx, cy), radius, globe_color, -1)
        # Inner LED stem or incandescent filament loops
        cv2.ellipse(img, (cx, cy), (radius // 3, radius // 2), 0, 0, 360, (160, 160, 160), 2)
        cv2.line(img, (cx - radius // 4, cy), (cx + radius // 4, cy), (80, 80, 80), 2)
        cv2.circle(img, (cx, cy), radius, (70, 70, 70), 2)

        x1 = max(0, cx - radius - 5)
        y1 = max(0, cy - radius - 5)
        x2 = min(w, cx + radius + 5)
        y2 = min(h, base_top + base_h + 10)

    elif device_type == "fan":
        cx = random.randint(240, 400)
        cy = random.randint(240, 400)
        hub_r = random.randint(30, 55)
        blade_len = random.randint(120, 180)
        blade_width = random.randint(25, 45)
        num_blades = random.choice([3, 4])
        angle_offset = random.uniform(0, 2 * np.pi)

        fan_color = random.choice([
            (40, 40, 40),
            (210, 210, 210),
            (35, 60, 110),
            (50, 100, 150)
        ])

        # Outer cage for table fan
        if random.random() > 0.4:
            cage_r = blade_len + 25
            cv2.circle(img, (cx, cy), cage_r, (120, 120, 120), 2)
            for deg in range(0, 360, 30):
                rad = np.deg2rad(deg)
                ex = int(cx + cage_r * np.cos(rad))
                ey = int(cy + cage_r * np.sin(rad))
                cv2.line(img, (cx, cy), (ex, ey), (160, 160, 160), 1)

        # Draw blades
        for i in range(num_blades):
            theta = angle_offset + i * (2 * np.pi / num_blades)
            tip_x = int(cx + blade_len * np.cos(theta))
            tip_y = int(cy + blade_len * np.sin(theta))
            perp_x = int(blade_width * -np.sin(theta))
            perp_y = int(blade_width * np.cos(theta))

            pts = np.array([
                [cx, cy],
                [tip_x - perp_x, tip_y - perp_y],
                [tip_x + perp_x, tip_y + perp_y]
            ], np.int32)
            cv2.fillPoly(img, [pts], fan_color)

        # Center motor hub
        cv2.circle(img, (cx, cy), hub_r, (fan_color[0] + 20, fan_color[1] + 20, fan_color[2] + 20), -1)
        cv2.circle(img, (cx, cy), hub_r, (20, 20, 20), 2)

        span = blade_len + 30
        x1 = max(0, cx - span)
        y1 = max(0, cy - span)
        x2 = min(w, cx + span)
        y2 = min(h, cy + span)

    elif device_type == "switchboard":
        bw = random.randint(240, 360)
        bh = random.randint(180, 260)
        bx1 = random.randint(120, w - bw - 120)
        by1 = random.randint(140, h - bh - 140)
        bx2 = bx1 + bw
        by2 = by1 + bh

        plate_color = (random.randint(230, 250), random.randint(230, 250), random.randint(235, 255))
        cv2.rectangle(img, (bx1, by1), (bx2, by2), plate_color, -1)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (180, 180, 180), 2)

        for sx, sy in [(bx1 + 12, by1 + 12), (bx2 - 12, by1 + 12), (bx1 + 12, by2 - 12), (bx2 - 12, by2 - 12)]:
            cv2.circle(img, (sx, sy), 4, (100, 100, 100), -1)

        sw_w = 32
        sw_h = 55
        start_x = bx1 + 30
        gap = (bw - 60) // 4
        for idx in range(3):
            sx = start_x + idx * gap
            sy = by1 + bh // 2 - sw_h // 2
            cv2.rectangle(img, (sx, sy), (sx + sw_w, sy + sw_h), (210, 210, 210), -1)
            cv2.rectangle(img, (sx, sy), (sx + sw_w, sy + sw_h), (90, 90, 90), 1)
            rocker_split = sy + sw_h // 2 + random.choice([-5, 5])
            cv2.line(img, (sx, rocker_split), (sx + sw_w, rocker_split), (60, 60, 60), 2)
            cv2.circle(img, (sx + sw_w // 2, sy + 8), 3, (0, 0, 220), -1)

        sock_x = start_x + 3 * gap
        sock_y = by1 + bh // 2
        cv2.circle(img, (sock_x + 15, sock_y - 12), 5, (40, 40, 40), -1)
        cv2.circle(img, (sock_x + 8, sock_y + 12), 4, (40, 40, 40), -1)
        cv2.circle(img, (sock_x + 22, sock_y + 12), 4, (40, 40, 40), -1)

        x1, y1, x2, y2 = bx1, by1, bx2, by2

    norm_cx = round(((x1 + x2) / 2) / w, 6)
    norm_cy = round(((y1 + y2) / 2) / h, 6)
    norm_w = round((x2 - x1) / w, 6)
    norm_h = round((y2 - y1) / h, 6)

    return img, (norm_cx, norm_cy, norm_w, norm_h)


def generate_missing_device_samples(samples_per_device=50):
    """Generate annotated samples for bulb, fan, switchboard:
       bulb: 0
       fan: 1
       switchboard: 4
    """
    print(f"[INFO] Generating samples for bulbs, fans, and switchboards ({samples_per_device} per device)...")

    device_classes = {
        "bulb": 0,
        "fan": 1,
        "switchboard": 4
    }

    for device_name, class_id in device_classes.items():
        for i in range(samples_per_device):
            r = random.random()
            if r < 0.75:
                split = "train"
            elif r < 0.90:
                split = "val"
            else:
                split = "test"

            img, bbox = draw_synthetic_sample(device_name)
            filename = f"syn_{device_name}_{i:03d}"

            img_path = DATASET_DIR / split / "images" / f"{filename}.jpg"
            lbl_path = DATASET_DIR / split / "labels" / f"{filename}.txt"

            cv2.imwrite(str(img_path), img)
            with open(lbl_path, "w") as f:
                f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

    print("[SUCCESS] Samples generated successfully.")


def write_data_yaml():
    data = {
        "path": str(DATASET_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"[SUCCESS] Wrote dataset config to: {yaml_path}")


def main():
    print("=" * 60)
    print("AI Repair Assistant - 5-Class Dataset Preparation Pipeline")
    print(f"Classes: {CLASSES}")
    print("=" * 60)
    create_directory_structure()
    extract_existing_devices()
    generate_missing_device_samples(samples_per_device=50)
    write_data_yaml()
    print("Dataset setup completed successfully!")

if __name__ == "__main__":
    main()
