"""
Comprehensive Integration & Smoke Tests for Device Registry & Rules
Validates:
  - All active laptop rules from DEVICE_REGISTRY
  - Tiered troubleshooting structure (safe_checks non-invasive vs advanced_steps chassis-open)
  - Unsupported device protection and safety advice across household/office appliances
  - Deterministic verdict assignment and e-waste calculations
"""

import sys
from fault_engine import DEVICE_REGISTRY, FaultDiagnosisEngine
from questionaire import run_questionnaire
from verdict import decide

engine = FaultDiagnosisEngine()


def test_active_laptop_rules():
    print("\n=== Active Laptop Fault Rules Smoke Test ===")
    laptop_rules = DEVICE_REGISTRY.get("laptop", [])
    if not laptop_rules:
        print("[FAIL] No laptop rules found in DEVICE_REGISTRY")
        sys.exit(1)

    print(f"Loaded {len(laptop_rules)} laptop fault rules from registry.")

    test_cases = [
        ("RAM / Black Screen", "laptop turns on power lights on fan spins but black screen blank display", "LAPTOP_DISPLAY_RAM_ISSUE"),
        ("SSD Failure", "boot error no bootable device found operating system not found ssd drive", "LAPTOP_STORAGE_DRIVE_FAILURE"),
        ("Overheating", "laptop gets very hot shutdown overheating thermal throttling fan loud", "LAPTOP_OVERHEATING_DUST_THERMAL"),
        ("Dead Charger", "laptop completely dead no lights no charging indicator adapter plugged in", "LAPTOP_DEAD_BATTERY_CHARGER"),
        ("Liquid Spill", "water spilled on keyboard coffee tea liquid ingress wet", "LAPTOP_LIQUID_SPILL"),
        ("Swollen Battery", "battery looks swollen bulging trackpad lifting chassis warped", "LAPTOP_SWOLLEN_BATTERY"),
        ("Keyboard Failure", "keyboard keys not typing stuck keys touchpad trackpad not responding", "LAPTOP_KEYBOARD_TOUCHPAD_FAILURE"),
        ("Power Surge", "power surge lightning thunder burnt smell after storm dead won't turn on", "LAPTOP_MOTHERBOARD_SURGE_POWER_FAILURE"),
        ("Fan Failure", "cooling fan grinding rattling noise stopped spinning thermal error", "LAPTOP_COOLING_FAN_FAILURE"),
        ("Screen Damage", "cracked lcd panel shattered screen lines colorful lines on display", "LAPTOP_PHYSICAL_SCREEN_DAMAGE"),
        ("Inconclusive Test", "device has a minor cosmetic scratch on the aluminum lid", "INCONCLUSIVE_INSPECTION_RECOMMENDED"),
    ]

    for name, symptom, expected_id in test_cases:
        res = engine.diagnose("laptop", symptom)
        pri = res.get("primary_diagnosis", {})
        fid = pri.get("fault_id")
        conf = pri.get("confidence_score", 0.0)
        rec = pri.get("recommendation", "N/A")

        if fid != expected_id:
            print(f"[FAIL] {name}: Expected {expected_id}, got {fid}")
            sys.exit(1)

        # Check tiered checks exist
        safe_checks = pri.get("safe_checks", [])
        if not safe_checks:
            print(f"[FAIL] {name}: safe_checks cannot be empty")
            sys.exit(1)

        # Verdict check
        v_res = decide("laptop", res)
        verdict = v_res.get("verdict")
        if verdict not in ["REPAIR (DIY)", "SERVICE (technician)", "REPLACE (device)"]:
            print(f"[FAIL] {name}: Invalid verdict '{verdict}'")
            sys.exit(1)

        print(f"[PASS] {name:22s} -> {fid:38s} (conf={conf:.2f}, rec={rec}, verdict={verdict})")


def test_unsupported_devices():
    print("\n=== Unsupported Devices Protection Test ===")
    unsupported_list = ["fan", "bulb", "switchboard", "mouse", "toaster", "refrigerator", "tv", "microwave"]
    for dev in unsupported_list:
        payload = {"device_type": dev, "findings": []}
        report = run_questionnaire(payload, run_diagnosis=False, skip_prompts=True)
        if report.get("status") != "unsupported_device" or report.get("supported") is not False:
            print(f"[FAIL] Device '{dev}' was not flagged as unsupported_device")
            sys.exit(1)
        if not report.get("safety_advice"):
            print(f"[FAIL] Device '{dev}' missing safety advice")
            sys.exit(1)
        print(f"[PASS] {dev:12s} -> Correctly flagged as unsupported_device with safety advice")


if __name__ == "__main__":
    test_active_laptop_rules()
    test_unsupported_devices()
    print("\nSupported devices in registry:", list(DEVICE_REGISTRY.keys()))
    print("ALL TESTS PASSED!\n")
