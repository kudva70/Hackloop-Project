"""
Unit Tests for AI Repair Assistant - Fault Diagnosis Engine
Validates deterministic laptop diagnosis, DeviceRegistry, safe_checks / advanced_steps split,
inconclusive fallback, and acceptance tests (no false power+light charger match).
"""

import unittest
from fault_engine import (
    FaultDiagnosisEngine,
    evaluate_safety,
    DEVICE_REGISTRY,
    LAPTOP_RULES,
    process_questionnaire,
)
from verdict import decide


class TestFaultDiagnosisEngine(unittest.TestCase):

    def setUp(self):
        self.engine = FaultDiagnosisEngine()

    # ----------------------------------------------------------------------
    # 1. Acceptance Criteria: No False Charger Match & Correct Mappings
    # ----------------------------------------------------------------------

    def test_no_false_power_light_charger_match(self):
        """'Lights/fan turn on but no display' must map to RAM/Display, NOT Dead Charger."""
        answers = {
            "affected_component": "Motherboard / mainboard",
            "power_state": "Lights/fan turn on but no display",
            "startup_signals": "Blinking diagnostic LED",
            "memory_storage": "Works normally",
            "display": "Black screen",
            "temperature": "Normal temperature",
            "charging": "Works normally",
            "recent_event": "No known event",
            "hazards": "No",
        }
        payload = {
            "status": "success",
            "module": "questionnaire",
            "device_type": "laptop",
            "detected_findings": ["RAM"],
            "answers": answers,
        }
        res = process_questionnaire(payload, use_gemini=False)
        primary = res["diagnosis"]
        self.assertEqual(primary["fault_id"], "LAPTOP_DISPLAY_RAM_ISSUE")
        self.assertNotEqual(primary["fault_id"], "LAPTOP_DEAD_BATTERY_CHARGER")

    def test_inconclusive_diagnosis_when_no_rules_score(self):
        """If nothing scores > 0, return Inconclusive (never rules[0] / dead charger)."""
        symptom = "The device exterior is painted silver with rounded corners and has a small scratch."
        res = self.engine.diagnose("laptop", symptom)

        self.assertEqual(res["status"], "success")
        primary = res["primary_diagnosis"]
        self.assertEqual(primary["fault_id"], "INCONCLUSIVE_INSPECTION_RECOMMENDED")
        self.assertEqual(primary["fault_title"], "Inconclusive - professional inspection recommended")
        self.assertLessEqual(primary["confidence_score"], 0.20)
        self.assertEqual(primary["recommendation"], "SERVICE (Professional)")

    def test_ssd_storage_failure_mapping(self):
        """'No boot device / SSD or HDD not detected' maps to LAPTOP_STORAGE_DRIVE_FAILURE."""
        answers = {
            "affected_component": "SSD / HDD storage",
            "power_state": "Starts normally",
            "startup_signals": "Error message",
            "memory_storage": "No boot device / SSD or HDD not detected",
            "display": "Works normally",
            "temperature": "Normal temperature",
            "charging": "Works normally",
            "recent_event": "No known event",
            "hazards": "No",
        }
        res = process_questionnaire({"status": "success", "device_type": "laptop", "answers": answers}, use_gemini=False)
        self.assertEqual(res["diagnosis"]["fault_id"], "LAPTOP_STORAGE_DRIVE_FAILURE")

    def test_swollen_battery_mapping_and_empty_advanced_steps(self):
        """'Battery looks swollen' maps to LAPTOP_SWOLLEN_BATTERY (CRITICAL, SERVICE, empty advanced_steps)."""
        answers = {
            "affected_component": "Battery / charging",
            "charging": "Battery looks swollen",
            "hazards": "Yes",
        }
        res = process_questionnaire({"status": "success", "device_type": "laptop", "answers": answers}, use_gemini=False)
        self.assertEqual(res["diagnosis"]["fault_id"], "LAPTOP_SWOLLEN_BATTERY")
        self.assertEqual(res["diagnosis"]["severity"], "CRITICAL")
        self.assertEqual(res["repair_recommendation"]["action"], "SERVICE (Professional)")
        self.assertEqual(res["diagnosis"]["advanced_steps"], [])
        self.assertTrue(res["safety_alert"])

    def test_keyboard_touchpad_failure_mapping(self):
        """Keyboard/trackpad failure maps to LAPTOP_KEYBOARD_TOUCHPAD_FAILURE."""
        symptom = "Specific keyboard keys not working and touchpad unresponsive."
        res = self.engine.diagnose("laptop", symptom)
        self.assertEqual(res["primary_diagnosis"]["fault_id"], "LAPTOP_KEYBOARD_TOUCHPAD_FAILURE")

    def test_motherboard_surge_mapping_and_empty_advanced_steps(self):
        """Power surge / electrical event maps to LAPTOP_MOTHERBOARD_SURGE_POWER_FAILURE (CRITICAL, SERVICE, empty advanced_steps)."""
        answers = {
            "power_state": "No power or lights",
            "recent_event": "Power surge / electrical event",
            "hazards": "Yes",
        }
        res = process_questionnaire({"status": "success", "device_type": "laptop", "answers": answers}, use_gemini=False)
        self.assertEqual(res["diagnosis"]["fault_id"], "LAPTOP_MOTHERBOARD_SURGE_POWER_FAILURE")
        self.assertEqual(res["diagnosis"]["severity"], "CRITICAL")
        self.assertEqual(res["repair_recommendation"]["action"], "SERVICE (Professional)")
        self.assertEqual(res["diagnosis"]["advanced_steps"], [])

    def test_cooling_fan_failure_mapping(self):
        """'Fan not spinning' maps to LAPTOP_COOLING_FAN_FAILURE."""
        symptom = "Cooling fan not spinning and laptop displays fan error."
        res = self.engine.diagnose("laptop", symptom)
        self.assertEqual(res["primary_diagnosis"]["fault_id"], "LAPTOP_COOLING_FAN_FAILURE")

    # ----------------------------------------------------------------------
    # 2. Unsupported Device Handling (DEVICE_REGISTRY)
    # ----------------------------------------------------------------------

    def test_unsupported_device_returns_structured_result(self):
        """Unsupported devices return structured 'unsupported_device' status with safety advice."""
        for unsupported_dev in ["fan", "bulb", "switchboard", "mouse", "toaster", "microwave"]:
            res = self.engine.diagnose(unsupported_dev, "buzzing noise")
            self.assertEqual(res["status"], "unsupported_device")
            self.assertFalse(res["supported"])
            self.assertIn("not supported yet", res["message"].lower())
            self.assertGreater(len(res["safety_advice"]), 0)

    # ----------------------------------------------------------------------
    # 3. Rule Structure & Step Splitting
    # ----------------------------------------------------------------------

    def test_all_rules_have_safe_checks_and_critical_rules_have_empty_advanced(self):
        """All laptop rules have safe_checks; CRITICAL or SERVICE-only rules have empty advanced_steps."""
        rules = DEVICE_REGISTRY["laptop"]
        self.assertGreaterEqual(len(rules), 10)
        for r in rules:
            self.assertGreater(len(r.safe_checks), 0, f"Rule {r.fault_id} must have safe_checks")
            self.assertGreater(len(r.troubleshooting_steps), 0, f"Rule {r.fault_id} must have troubleshooting_steps")
            if r.severity == "CRITICAL" or "SERVICE" in r.recommendation:
                if r.fault_id in ("LAPTOP_LIQUID_SPILL", "LAPTOP_SWOLLEN_BATTERY", "LAPTOP_MOTHERBOARD_SURGE_POWER_FAILURE"):
                    self.assertEqual(
                        r.advanced_steps, [],
                        f"CRITICAL rule {r.fault_id} must have empty advanced_steps"
                    )

    # ----------------------------------------------------------------------
    # 4. Safety Guardrails & Verdict Acceptance Tests
    # ----------------------------------------------------------------------

    def test_not_sure_is_caution_not_hazard(self):
        """'Not sure' must be treated as caution, NOT as a confirmed hazard."""
        res = evaluate_safety(answers={"hazards": "Not sure"})
        self.assertFalse(res["hazard_detected"])
        self.assertFalse(res["safety_alert"])
        self.assertGreater(len(res["caution_notes"]), 0)

    def test_verdict_cheap_fix_repair_diy(self):
        """Cheap fix (e.g. RAM contact clean or heatsink paste) -> REPAIR (DIY)."""
        report = {
            "primary_diagnosis": {
                "fault_id": "LAPTOP_DISPLAY_RAM_ISSUE",
                "fault_title": "RAM Contact Issue",
                "estimated_parts_cost": "₹0 ($0)",
                "recommendation": "REPAIR (DIY)",
                "repair_complexity": "Low (Beginner)",
                "severity": "MEDIUM",
                "confidence_score": 0.85,
            }
        }
        decision = decide(device="laptop", report=report, age_years=1.0)
        self.assertEqual(decision["verdict"], "REPAIR (DIY)")
        self.assertFalse(decision["safety_hold"])
        self.assertFalse(decision["low_confidence"])
        self.assertGreater(len(decision["decision_factors"]), 0)

    def test_verdict_costly_and_old_replace_device(self):
        """Costly and old (repair >= 40% on device >= 70% of life) -> REPLACE (device)."""
        report = {
            "primary_diagnosis": {
                "fault_id": "LAPTOP_MOTHERBOARD_REPLACEMENT",
                "fault_title": "Motherboard Replacement",
                "estimated_parts_cost": "₹20,000 - ₹25,000",
                "recommendation": "REPLACE",
                "repair_complexity": "High (Advanced)",
                "severity": "HIGH",
                "confidence_score": 0.80,
            }
        }
        # 4.0 years / 5.0 years = 80% life used, 22.5k / 45k = 50% cost ratio
        decision = decide(device="laptop", report=report, age_years=4.0)
        self.assertEqual(decision["verdict"], "REPLACE (device)")
        self.assertTrue(any("REPLACE (device)" in r or "replacing the entire device" in r.lower() for r in decision["reasons"]))

    def test_verdict_liquid_spill_service_technician(self):
        """Liquid spill / swollen battery / critical hazard -> SERVICE (technician)."""
        report = {
            "primary_diagnosis": {
                "fault_id": "LAPTOP_LIQUID_SPILL",
                "fault_title": "Liquid Ingress Hazard",
                "estimated_parts_cost": "₹1,500 - ₹5,000",
                "recommendation": "SERVICE (Professional)",
                "repair_complexity": "Dangerous (Do Not DIY)",
                "severity": "CRITICAL",
                "confidence_score": 0.90,
            },
            "safety_alert": True,
            "requires_professional_attention": True,
        }
        decision = decide(device="laptop", report=report)
        self.assertEqual(decision["verdict"], "SERVICE (technician)")
        self.assertTrue(decision["safety_hold"])

    def test_verdict_inconclusive_low_confidence_flag(self):
        """Inconclusive diagnosis or unparsed cost -> low_confidence: True with reason."""
        report = {
            "primary_diagnosis": {
                "fault_id": "INCONCLUSIVE_INSPECTION_RECOMMENDED",
                "fault_title": "Inconclusive - professional inspection recommended",
                "estimated_parts_cost": "To be quoted after inspection",
                "recommendation": "SERVICE (Professional)",
                "repair_complexity": "Unknown",
                "severity": "LOW",
                "confidence_score": 0.15,
            }
        }
        decision = decide(device="laptop", report=report)
        self.assertTrue(decision["low_confidence"])
        self.assertTrue(any("low-confidence" in r.lower() or "inconclusive" in r.lower() for r in decision["reasons"]))
        self.assertIn(decision["verdict"], ["REPAIR (DIY)", "SERVICE (technician)", "REPLACE (device)"])


if __name__ == "__main__":
    unittest.main()
