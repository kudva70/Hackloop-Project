"""End-to-end Repair Before Replace pipeline.

Flow:
    optional image -> detect_new.py -> questionaire.py -> fault_engine.py

The questionnaire owns the diagnosis call, so this module deliberately does not
call the diagnosis engine a second time. Gemini is used when configured; the
rule-based engine remains the fallback.
"""

from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import sys

import detect_new
from questionaire import run_questionnaire, normalize_device
from fault_engine import engine

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "diagnoses"


def print_banner():
    print("\n" + "=" * 72)
    print("       REPAIR BEFORE REPLACE - AI HARDWARE FAULT DIAGNOSIS")
    print("=" * 72)


def choose_device():
    """Prompt for a supported device if image detection cannot identify one."""
    supported = sorted(engine.get_supported_devices())
    print("\nSelect the device you want to diagnose:")
    for index, device in enumerate(supported, start=1):
        print(f"  {index}. {device.title()}")

    choice = input("\nEnter a number or device name: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(supported):
        return supported[int(choice) - 1]
    return normalize_device(choice)


def make_questionnaire_detection(image_path=None, device_type=None, detection_result=None):
    """Adapt detect_new's detections to the input schema expected by questionaire.py."""
    detections = (detection_result or {}).get("detections", [])
    findings = [str(item.get("device", "")).strip() for item in detections]
    findings = [item for item in findings if item]

    confidence = None
    if detections:
        confidence = max(float(item.get("confidence", 0.0)) for item in detections)

    # detect_new currently uses a Roboflow laptop-component model. A detection
    # means the overall device is a laptop, while the classes become visual findings.
    detected_type = (detection_result or {}).get("device_type")
    resolved_device = device_type or detected_type

    return {
        "device_type": resolved_device,
        "findings": findings,
        "confidence": confidence,
        "image_path": str(image_path) if image_path else None,
    }


def print_report(report):
    """Print a compact report that matches the questionnaire diagnosis schema."""
    diagnosis = report.get("diagnosis") or {}
    recommendation = report.get("repair_recommendation") or {}

    print("\n" + "=" * 72)
    print(f"DIAGNOSTIC REPORT: {report.get('device_type', 'Unknown').upper()}")
    print("=" * 72)
    print(f"Visual findings       : {', '.join(report.get('detected_findings') or []) or 'None detected'}")
    print(f"Primary possible fault: {diagnosis.get('primary_fault', 'Not available')}")

    confidence = diagnosis.get("confidence")
    if isinstance(confidence, (int, float)):
        print(f"Diagnosis confidence : {confidence:.0%}")

    print(f"Severity              : {diagnosis.get('severity', 'Not specified')}")
    print(f"Recommended action    : {recommendation.get('action', 'Not available')}")
    print(f"Repair complexity     : {recommendation.get('complexity', 'Not specified')}")
    print(f"Estimated cost        : {recommendation.get('estimated_cost', 'Not available')}")
    print(f"Estimated time        : {recommendation.get('estimated_time', 'Not available')}")

    if report.get("safety_alert"):
        print("\n!!! SAFETY ALERT: Stop using the device and follow the safety instructions. !!!")

    safety_instructions = diagnosis.get("safety_instructions") or []
    if safety_instructions:
        print("\nSAFETY INSTRUCTIONS")
        for item in safety_instructions:
            print(f"  - {item}")

    steps = diagnosis.get("troubleshooting_steps") or []
    if steps:
        print("\nTROUBLESHOOTING STEPS")
        for index, step in enumerate(steps, start=1):
            print(f"  {index}. {step}")

    tools = recommendation.get("required_tools") or []
    if tools:
        print("\nREQUIRED TOOLS")
        print("  " + ", ".join(map(str, tools)))

    alternatives = diagnosis.get("secondary_faults") or []
    if alternatives:
        print("\nOTHER POSSIBLE FAULTS")
        for item in alternatives:
            print(f"  - {item.get('fault_title', item.get('fault_id', 'Possible fault'))}: "
                  f"{item.get('recommendation', 'Further inspection needed')}")

    print("=" * 72)


def save_report(report, source_name="manual"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stem = Path(source_name).stem or "diagnosis"
    output_path = OUTPUT_DIR / f"{safe_stem}_{timestamp}_diagnosis.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, default=str)
    return output_path


def run_pipeline(image_path=None, device_type=None, confidence=0.25):
    """Run detection (if possible), questionnaire, diagnosis, and JSON export."""
    print_banner()
    source_image = Path(image_path).expanduser().resolve() if image_path else None
    detection_result = None

    if source_image:
        if not source_image.is_file():
            raise FileNotFoundError(f"Image file not found: {source_image}")
        print(f"[STEP 1/3] Detecting device/components in {source_image.name}...")
        try:
            detection_result = detect_new.detect_image(str(source_image), conf=confidence)
        except Exception as error:
            print(f"[WARNING] Visual detection could not run: {error}")
            print("          You can continue by selecting the device manually.")

    detection_payload = make_questionnaire_detection(
        image_path=source_image,
        device_type=device_type,
        detection_result=detection_result,
    )

    if not detection_payload.get("device_type"):
        if detection_payload.get("findings"):
            detection_payload["device_type"] = "laptop"
        else:
            detection_payload["device_type"] = choose_device()

    print("\n[STEP 2/3] Answer the symptom and safety questions.")
    print("[STEP 3/3] Generating diagnosis (Gemini when configured; rule-based fallback otherwise)...")

    # run_questionnaire performs the diagnosis internally. Do not call the engine
    # again here, or the user will pay for/receive duplicate Gemini requests.
    report = run_questionnaire(
        detection_payload,
        image_path=str(source_image) if source_image else None,
        run_diagnosis=True,
    )

    print_report(report)
    saved_path = save_report(report, source_name=str(source_image or detection_payload["device_type"]))
    print(f"\n[INFO] Full JSON report saved to: {saved_path}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Repair Before Replace - AI hardware fault diagnosis assistant"
    )
    parser.add_argument("--image", "-i", help="Path to the faulty device image")
    parser.add_argument("--device", "-d", help="Device type (overrides automatic device type)")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Object detection confidence threshold (default: 0.25)")
    args = parser.parse_args()

    try:
        run_pipeline(image_path=args.image, device_type=args.device, confidence=args.confidence)
    except (FileNotFoundError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
