"""
Smoke-test verifying active laptop fault rules and graceful unsupported-device handling.
Run: python test_all_devices.py
"""
from fault_engine import FaultDiagnosisEngine, DEVICE_REGISTRY

e = FaultDiagnosisEngine()

laptop_tests = [
    ("RAM / Black Screen", "laptop", "Lights/fan turn on but no display, black screen"),
    ("SSD Failure", "laptop", "No boot device / SSD or HDD not detected, drive error"),
    ("Overheating", "laptop", "Laptop runs unusually hot, fan noise, thermal throttling"),
    ("Dead Charger", "laptop", "Laptop dead, will not charge, no power or lights"),
    ("Liquid Spill", "laptop", "Spilled cup of water on keyboard"),
    ("Swollen Battery", "laptop", "Battery looks swollen and pushing up trackpad"),
    ("Keyboard Failure", "laptop", "Keyboard keys not working, touchpad unresponsive"),
    ("Power Surge", "laptop", "Power surge electrical event, dead after lightning spark"),
    ("Fan Failure", "laptop", "Cooling fan not spinning, fan error 90b"),
    ("Screen Damage", "laptop", "Visible screen crack with shattered glass ink bleed"),
    ("Inconclusive Test", "laptop", "The laptop is grey and was manufactured in 2022"),
]

unsupported_tests = ["fan", "bulb", "switchboard", "mouse", "toaster"]

print("\n=== Active Laptop Fault Rules Smoke Test ===\n")
all_ok = True

for test_name, device, symptoms in laptop_tests:
    res = e.diagnose(device, symptoms)
    pd = res.get("primary_diagnosis", {})
    fault_id = pd.get("fault_id", "N/A")
    conf = pd.get("confidence_score", 0.0)
    rec = pd.get("recommendation", "N/A")
    status = "PASS" if fault_id != "N/A" else "FAIL"
    print(f"[{status}] {test_name:22s} -> {fault_id:38s} (conf={conf:.2f}, rec={rec})")
    if status == "FAIL":
        all_ok = False

print("\n=== Unsupported Devices Protection Test ===\n")
for dev in unsupported_tests:
    res = e.diagnose(dev, "buzzing noise")
    if res.get("status") == "unsupported_device" and not res.get("supported"):
        print(f"[PASS] {dev:12s} -> Correctly flagged as unsupported_device with safety advice")
    else:
        print(f"[FAIL] {dev:12s} -> Did not return unsupported_device status")
        all_ok = False

print(f"\nSupported devices in registry: {list(DEVICE_REGISTRY.keys())}")
print("ALL TESTS PASSED!" if all_ok else "SOME TESTS FAILED.")
