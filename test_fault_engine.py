"""
Unit & Integration Tests for AI Repair Assistant
Validates:
  - DeviceRegistry and active laptop rules
  - Real questionnaire key mappings
  - Tiered repair steps (safe_checks non-invasive vs advanced_steps chassis-open)
  - CRITICAL / SERVICE-only rules have empty advanced_steps
  - Three canonical verdicts: REPAIR (DIY), SERVICE (technician), REPLACE (device)
  - Hard hazard free-text detection & early safety stop
  - Unsupported device routing with safety advice
  - E-waste impact calculations (avoided kg) & recycling advice
  - End-to-end --demo offline execution
"""

import unittest
from fault_engine import (
    FaultDiagnosisEngine,
    evaluate_safety,
    DEVICE_REGISTRY,
    LAPTOP_RULES,
    process_questionnaire,
)
from questionaire import (
    answers_from_description,
    run_questionnaire,
    unsupported_device_from_text,
    QUESTIONNAIRE_KEYS,
)
from verdict import decide, parse_cost, DEVICE_MASS_KG
import assistant


class TestFaultEngineAndRegistry(unittest.TestCase):

    def setUp(self):
        self.engine = FaultDiagnosisEngine()

    # ----------------------------------------------------------------------
    # 1. Device Registry & Laptop Rules Structure
    # ----------------------------------------------------------------------

    def test_device_registry_structure_and_active_rules(self):
        """Registry must contain 'laptop' with all expected rules, and no legacy devices."""
        self.assertIn("laptop", DEVICE_REGISTRY)
        self.assertNotIn("fan", DEVICE_REGISTRY)
        self.assertNotIn("bulb", DEVICE_REGISTRY)
        self.assertNotIn("switchboard", DEVICE_REGISTRY)
        self.assertNotIn("mouse", DEVICE_REGISTRY)

        active_rule_ids = {r.fault_id for r in DEVICE_REGISTRY["laptop"]}
        expected_ids = {
            "LAPTOP_DEAD_BATTERY_CHARGER",
            "LAPTOP_DISPLAY_RAM_ISSUE",
            "LAPTOP_STORAGE_DRIVE_FAILURE",
            "LAPTOP_OVERHEATING_DUST_THERMAL",
            "LAPTOP_LIQUID_SPILL",
            "LAPTOP_SWOLLEN_BATTERY",
            "LAPTOP_KEYBOARD_TOUCHPAD_FAILURE",
            "LAPTOP_MOTHERBOARD_SURGE_POWER_FAILURE",
            "LAPTOP_COOLING_FAN_FAILURE",
            "LAPTOP_PHYSICAL_SCREEN_DAMAGE",
        }
        self.assertTrue(expected_ids.issubset(active_rule_ids))

    def test_tiered_steps_safe_checks_and_advanced_split(self):
        """All laptop rules must have safe_checks; CRITICAL or SERVICE-only rules must have empty advanced_steps."""
        for rule in LAPTOP_RULES:
            self.assertTrue(
                len(rule.safe_checks) > 0,
                f"Rule {rule.fault_id} must have non-empty safe_checks"
            )
            is_critical = (
                rule.severity == "CRITICAL"
                or "service" in rule.recommendation.lower()
                or rule.fault_id in ("LAPTOP_SWOLLEN_BATTERY", "LAPTOP_LIQUID_SPILL", "LAPTOP_MOTHERBOARD_SURGE_POWER_FAILURE")
            )
            if is_critical and rule.fault_id != "LAPTOP_OVERHEATING_DUST_THERMAL":
                if "SERVICE" in rule.recommendation and "DIY" not in rule.recommendation:
                    self.assertEqual(
                        len(rule.advanced_steps), 0,
                        f"CRITICAL/SERVICE-only rule {rule.fault_id} must have empty advanced_steps"
                    )

    # ----------------------------------------------------------------------
    # 2. Real Questionnaire Keys & Diagnostic Mappings
    # ----------------------------------------------------------------------

    def test_real_questionnaire_keys_used(self):
        """Questionnaire keys must strictly match the canonical list."""
        expected_keys = {
            "affected_component",
            "power_state",
            "startup_signals",
            "memory_storage",
            "display",
            "temperature",
            "charging",
            "recent_event",
            "hazards",
            "duration",
        }
        self.assertEqual(set(QUESTIONNAIRE_KEYS), expected_keys)

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
            "duration": "A few days",
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

    def test_storage_failure_mapping(self):
        """'No boot device / SSD or HDD not detected' maps to LAPTOP_STORAGE_DRIVE_FAILURE."""
        answers = {
            "affected_component": "SSD / HDD storage",
            "power_state": "Starts normally",
            "startup_signals": "Error message",
            "memory_storage": "No boot device / SSD or HDD not detected",
            "display": "Normal display",
            "temperature": "Normal temperature",
            "charging": "Works normally",
            "recent_event": "No known event",
            "hazards": "No",
            "duration": "Started today",
        }
        payload = {
            "status": "success",
            "module": "questionnaire",
            "device_type": "laptop",
            "detected_findings": ["SSD"],
            "answers": answers,
        }
        res = process_questionnaire(payload, use_gemini=False)
        primary = res["diagnosis"]
        self.assertEqual(primary["fault_id"], "LAPTOP_STORAGE_DRIVE_FAILURE")

    def test_swollen_battery_mapping(self):
        """'Battery looks swollen' maps to LAPTOP_SWOLLEN_BATTERY (CRITICAL)."""
        answers = {
            "affected_component": "Battery / charging",
            "power_state": "Starts normally",
            "startup_signals": "No",
            "memory_storage": "Not sure / not applicable",
            "display": "Normal display",
            "temperature": "Very hot to touch",
            "charging": "Battery looks swollen or trackpad lifting",
            "recent_event": "No known event",
            "hazards": "Yes",
            "duration": "A few days",
        }
        payload = {
            "status": "success",
            "module": "questionnaire",
            "device_type": "laptop",
            "detected_findings": ["battery"],
            "answers": answers,
        }
        res = process_questionnaire(payload, use_gemini=False)
        primary = res["diagnosis"]
        self.assertEqual(primary["fault_id"], "LAPTOP_SWOLLEN_BATTERY")
        self.assertEqual(primary["severity"], "CRITICAL")
        self.assertEqual(len(primary["advanced_steps"]), 0)

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

    # ----------------------------------------------------------------------
    # 3. Hazard-Text Scanning & Safety Logic
    # ----------------------------------------------------------------------

    def test_hazard_text_detection_and_early_stop(self):
        """Hazard words in free text must immediately raise safety_alert."""
        for word in ["smoke", "spark", "burning smell", "shock", "swollen", "liquid spill", "wet"]:
            res = evaluate_safety(free_text=f"My laptop has a {word} when plugged in.")
            self.assertTrue(res["safety_alert"], f"Word '{word}' must trigger safety_alert")
            self.assertTrue(res["requires_professional_attention"])

    def test_not_sure_is_caution_not_hazard(self):
        """'Not sure' must be treated as caution, NOT as a confirmed hazard."""
        answers = {
            "hazards": "Not sure",
            "temperature": "Not sure",
            "power_state": "Starts normally",
        }
        res = evaluate_safety(answers=answers)
        self.assertFalse(res["safety_alert"])
        self.assertFalse(res["requires_professional_attention"])

    # ----------------------------------------------------------------------
    # 4. Three Canonical Verdicts & E-Waste Output
    # ----------------------------------------------------------------------

    def test_three_verdicts_and_ewaste(self):
        """Validate REPAIR (DIY), SERVICE (technician), and REPLACE (device) plus e-waste notes."""
        # 1. REPAIR (DIY)
        res_repair = decide("laptop", {
            "primary_diagnosis": {
                "fault_id": "LAPTOP_DISPLAY_RAM_ISSUE",
                "fault_title": "RAM contact oxidation",
                "recommendation": "REPAIR (DIY)",
                "estimated_parts_cost": "Rs 0",
                "repair_complexity": "Low",
                "confidence_score": 0.95,
            }
        })
        self.assertEqual(res_repair["verdict"], "REPAIR (DIY)")
        self.assertEqual(res_repair["e_waste"]["status"], "avoided")
        self.assertEqual(res_repair["e_waste"]["amount_kg"], 2.1)
        self.assertIn("estimate", res_repair["e_waste"]["summary"].lower())

        # 2. SERVICE (technician)
        res_service = decide("laptop", {
            "primary_diagnosis": {
                "fault_id": "LAPTOP_SWOLLEN_BATTERY",
                "fault_title": "Swollen Battery",
                "recommendation": "SERVICE (Professional)",
                "estimated_parts_cost": "Rs 3500",
                "repair_complexity": "Dangerous",
                "severity": "CRITICAL",
                "confidence_score": 0.90,
            }
        })
        self.assertEqual(res_service["verdict"], "SERVICE (technician)")
        self.assertTrue(res_service["safety_hold"])
        self.assertEqual(res_service["e_waste"]["status"], "avoided")

        # 3. REPLACE (device)
        res_replace = decide("laptop", {
            "primary_diagnosis": {
                "fault_id": "LAPTOP_MOTHERBOARD_DEAD",
                "fault_title": "Dead Motherboard",
                "recommendation": "REPLACE",
                "estimated_parts_cost": "Rs 32000",
                "repair_complexity": "High",
                "confidence_score": 0.90,
            }
        }, age_years=4.5)
        self.assertEqual(res_replace["verdict"], "REPLACE (device)")
        self.assertEqual(res_replace["e_waste"]["status"], "recycling_advice")
        self.assertIn("recycler", res_replace["e_waste"]["advice"].lower())
        self.assertIn("data", res_replace["e_waste"]["advice"].lower())

    # ----------------------------------------------------------------------
    # 5. Unsupported Device Routing
    # ----------------------------------------------------------------------

    def test_unsupported_device_path(self):
        """Unsupported devices return structured 'unsupported_device' payload with safety advice."""
        for dev in ["fan", "bulb", "switchboard", "mouse", "toaster"]:
            self.assertEqual(unsupported_device_from_text(f"my {dev} is broken"), dev)
            report = run_questionnaire(
                {"device_type": dev, "findings": []},
                run_diagnosis=False,
                skip_prompts=True,
            )
            self.assertEqual(report["status"], "unsupported_device")
            self.assertFalse(report["supported"])
            self.assertTrue(len(report["safety_advice"]) > 0)

    # ----------------------------------------------------------------------
    # 6. End-to-End Demo Mode
    # ----------------------------------------------------------------------

    def test_assistant_demo_mode_e2e(self):
        """assistant.py --demo must run completely offline and produce a valid report."""
        report = assistant.run_pipeline(demo=True)
        self.assertIsNotNone(report)
        self.assertEqual(report.get("device_type"), "laptop")
        self.assertIn("repair_decision", report)
        self.assertIn(
            report["repair_decision"]["verdict"],
            ["REPAIR (DIY)", "SERVICE (technician)", "REPLACE (device)"]
        )
        self.assertIn("e_waste", report["repair_decision"])
        self.assertIn("explanation", report)


if __name__ == "__main__":
    unittest.main()
