"""
AI Repair Assistant - End-to-End Pipeline
Bridges Object Detection (YOLO) with the Rule-Based Fault Diagnosis Engine.

Workflow:
  1. Accepts an uploaded photo/video of a faulty device.
  2. Runs YOLO object detection to identify the device type (fan, bulb, switchboard, laptop).
  3. Prompts for or ingests user-described symptoms.
  4. Executes the Fault Diagnosis Engine.
  5. Displays a rich, structured diagnostic report and saves the JSON artifact.
"""

from pathlib import Path
import sys
import json
import argparse
from datetime import datetime

import detect
from fault_engine import engine, FaultDiagnosisEngine

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "diagnoses"


def print_banner():
    print("""
========================================================================
       REPAIR BEFORE REPLACE - AI HARDWARE FAULT DIAGNOSIS ASSISTANT
========================================================================
    """)


def print_diagnostic_card(report: dict):
    """Render a clean, human-readable terminal report."""
    primary = report["primary_diagnosis"]
    device = report["device"].upper()

    print("\n" + "=" * 70)
    print(f"  DIAGNOSTIC REPORT: {device}")
    print("=" * 70)

    # Detection metadata
    det = report.get("original_detection", {})
    if det.get("detection_confidence"):
        conf_pct = round(det["detection_confidence"] * 100, 1)
        print(f"  [+] Identified Device    : {det['detected_device']} ({conf_pct}% detection confidence)")
    else:
        print(f"  [+] Target Device        : {device}")

    print(f"  [+] Reported Symptoms    : \"{report['user_symptoms']}\"")
    print(f"  [+] Primary Fault        : {primary['fault_title']} (Confidence: {int(primary['confidence_score'] * 100)}%)")
    print(f"  [+] Severity Level       : {primary['severity']}")
    print(f"  [+] Repair Complexity    : {primary['repair_complexity']}")
    print(f"  [+] Recommendation       : >>> {primary['recommendation']} <<<")
    print(f"  [+] Rationale            : {primary['recommendation_rationale']}")
    print(f"  [+] Estimated Cost       : {primary['estimated_parts_cost']}")
    print(f"  [+] Estimated Time       : {primary['estimated_diy_time']}")

    print("\n" + "-" * 70)
    print("  CRITICAL SAFETY INSTRUCTIONS:")
    print("-" * 70)
    for warn in primary["safety_warnings"]:
        print(f"  [!] {warn}")

    print("\n" + "-" * 70)
    print("  STEP-BY-STEP SAFE TROUBLESHOOTING:")
    print("-" * 70)
    for step in primary["troubleshooting_steps"]:
        print(f"  {step}")

    print("\n" + "-" * 70)
    print("  REQUIRED TOOLS:")
    print("-" * 70)
    print("  " + ", ".join(primary["required_tools"]))

    if report.get("alternative_diagnoses"):
        print("\n" + "-" * 70)
        print("  OTHER POSSIBLE FAULTS TO CONSIDER:")
        print("-" * 70)
        for alt in report["alternative_diagnoses"]:
            alt_conf = int(alt["confidence_score"] * 100)
            print(f"  - {alt['fault_title']} ({alt_conf}% match) -> {alt['recommendation']}")

    print("=" * 70 + "\n")


def save_diagnostic_report(report: dict, source_name: str = "manual") -> Path:
    """Save structured diagnosis report to JSON file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_stem = Path(source_name).stem
    out_file = OUTPUT_DIR / f"{clean_stem}_{timestamp}_diagnosis.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return out_file


def run_pipeline(image_path: str = None, device_type: str = None, symptoms: str = None):
    print_banner()

    detected_device = None
    detection_conf = None
    detection_box = None

    # Step 1: Object detection if image is provided
    if image_path:
        print(f"[STEP 1] Running visual object detection on: {image_path}...")
        try:
            det_result = detect.detect_image(image_path)
            detections = det_result.get("detections", [])

            if detections:
                top_det = max(detections, key=lambda x: x["confidence"])
                detected_device = top_det["device"]
                detection_conf = top_det["confidence"]
                detection_box = top_det["box"]
                print(f"[SUCCESS] Detected Device: '{detected_device}' (Confidence: {round(detection_conf * 100, 1)}%)")
            else:
                print("[WARNING] No device detected with high confidence in the image.")
        except Exception as e:
            print(f"[ERROR] Object detection failed: {e}")

    # Step 2: Resolve device type
    if not detected_device:
        if device_type:
            detected_device = device_type
        else:
            print("\nPlease select or type the faulty device:")
            supported = engine.get_supported_devices()
            for idx, dev in enumerate(supported, 1):
                print(f"  {idx}. {dev.capitalize()}")
            choice = input("\nEnter number or device name: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(supported):
                detected_device = supported[int(choice) - 1]
            else:
                detected_device = choice

    # Step 3: Collect user symptoms
    if not symptoms:
        print(f"\n[STEP 2] Describe the issue with the {detected_device}:")
        print("  1. Type free-form description (e.g. 'Fan hums but won't spin unless pushed')")
        print("  2. Or choose from common symptom checklist")
        mode = input("\nSelect option (1 for text, 2 for checklist) [default: 1]: ").strip()

        if mode == "2":
            checklist = engine.get_device_symptom_checklist(detected_device)
            print("\nCommon Symptoms:")
            for idx, item in enumerate(checklist, 1):
                print(f"  [{idx}] {item}")
            selection = input("\nEnter symptom numbers separated by commas (e.g. 1, 3): ").strip()
            selected_items = []
            for num in selection.split(","):
                num = num.strip()
                if num.isdigit() and 1 <= int(num) <= len(checklist):
                    selected_items.append(checklist[int(num) - 1])
            symptoms = " ".join(selected_items) if selected_items else "Device is not operating properly."
        else:
            symptoms = input("\nEnter symptom description: ").strip()
            if not symptoms:
                symptoms = "Device not functioning."

    # Step 4: Run Fault Diagnosis Engine
    print("\n[STEP 3] Running deterministic fault diagnosis...")
    report = engine.diagnose(
        device=detected_device,
        user_symptoms=symptoms,
        detection_confidence=detection_conf,
        detection_bbox=detection_box
    )

    # Step 5: Render and save
    print_diagnostic_card(report)
    saved_path = save_diagnostic_report(report, source_name=image_path or detected_device)
    print(f"[INFO] Full structured report saved to: {saved_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Repair Before Replace - AI Hardware Fault Diagnosis Assistant")
    parser.add_argument("--image", "-i", help="Path to image of the faulty device")
    parser.add_argument("--device", "-d", help="Explicit device type (fan, bulb, switchboard, laptop)")
    parser.add_argument("--symptoms", "-s", help="Text description of the symptoms / failure")
    args = parser.parse_args()

    run_pipeline(image_path=args.image, device_type=args.device, symptoms=args.symptoms)


if __name__ == "__main__":
    main()

