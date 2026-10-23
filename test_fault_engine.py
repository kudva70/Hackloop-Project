"""
Unit Tests for AI Repair Assistant - Fault Diagnosis Engine
Validates deterministic diagnosis accuracy, safety alerts, and recommendations.
"""

import unittest
from fault_engine import FaultDiagnosisEngine


class TestFaultDiagnosisEngine(unittest.TestCase):

    def setUp(self):
        self.engine = FaultDiagnosisEngine()

    def test_fan_weak_capacitor(self):
        symptom = "The ceiling fan hums loudly and won't spin unless I give it a push with a stick. It also spins slowly on high speed."
        res = self.engine.diagnose(device="fan", user_symptoms=symptom, detection_confidence=0.96)

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["device"], "fan")
        primary = res["primary_diagnosis"]
        self.assertEqual(primary["fault_id"], "FAN_WEAK_CAPACITOR")
        self.assertEqual(primary["recommendation"], "REPAIR (DIY)")
        self.assertIn("Capacitor", primary["fault_title"])
        self.assertTrue(any("discharge" in w.lower() for w in primary["safety_warnings"]))

    def test_fan_jammed_bearing(self):
        symptom = "Blades are stiff and hard to rotate by hand, makes a loud squeaking grinding noise."
        res = self.engine.diagnose(device="fan", user_symptoms=symptom)

        primary = res["primary_diagnosis"]
        self.assertEqual(primary["fault_id"], "FAN_JAMMED_BEARINGS")
        self.assertIn("Bearings", primary["fault_title"])

    def test_fan_burnt_stator(self):
        symptom = "There is black smoke and a strong burning copper smell coming from the motor. Fan is dead."
        res = self.engine.diagnose(device="fan", user_symptoms=symptom)

        primary = res["primary_diagnosis"]
        self.assertEqual(primary["fault_id"], "FAN_BURNT_STATOR_WINDING")
        self.assertEqual(primary["severity"], "HIGH")

    def test_laptop_ram_black_screen(self):
        symptom = "The laptop turns on and power lights / fan are spinning, but the screen stays completely black with no display."
        res = self.engine.diagnose(device="laptop", user_symptoms=symptom)

        primary = res["primary_diagnosis"]
        self.assertEqual(primary["fault_id"], "LAPTOP_DISPLAY_RAM_ISSUE")
        self.assertTrue(any("RAM" in step or "eraser" in step.lower() for step in primary["troubleshooting_steps"]))

    def test_laptop_water_spill(self):
        symptom = "I accidentally spilled a glass of water on the keyboard while working."
        res = self.engine.diagnose(device="laptop", user_symptoms=symptom)

        primary = res["primary_diagnosis"]
        self.assertEqual(primary["fault_id"], "LAPTOP_LIQUID_SPILL")
        self.assertEqual(primary["severity"], "CRITICAL")
        self.assertIn("Dangerous", primary["repair_complexity"])

    def test_bulb_flickering(self):
        symptom = "The LED bulb is blinking and flickering rapidly like a strobe light."
        res = self.engine.diagnose(device="bulb", user_symptoms=symptom)

        primary = res["primary_diagnosis"]
        self.assertEqual(primary["fault_id"], "BULB_FAILED_LED_DRIVER")
        self.assertIn("REPLACE", primary["recommendation"])

    def test_switchboard_sparking(self):
        symptom = "There is a blue spark every time I flip the switch, and there are black scorch marks on the plastic."
        res = self.engine.diagnose(device="switchboard", user_symptoms=symptom)

        primary = res["primary_diagnosis"]
        self.assertEqual(primary["fault_id"], "SWITCH_BURNT_CONTACT_ARCING")
        self.assertEqual(primary["severity"], "HIGH")
        self.assertTrue(any("MCB" in w or "breaker" in w.lower() for w in primary["safety_warnings"]))

    def test_switchboard_circuit_breaker_tripping(self):
        symptom = "Whenever I plug in my device, the MCB circuit breaker trips immediately with a loud click."
        res = self.engine.diagnose(device="switchboard", user_symptoms=symptom)

        primary = res["primary_diagnosis"]
        self.assertEqual(primary["fault_id"], "CIRCUIT_BREAKER_TRIP_SHORT")
        self.assertEqual(primary["severity"], "CRITICAL")

    def test_questionnaire_payload_processing(self):
        payload = {
            "status": "success",
            "module": "questionnaire",
            "device_type": "laptop",
            "detected_findings": ["screen crack"],
            "detection_confidence": 0.91,
            "answers": {
                "power": "Yes",
                "display": "Black screen",
                "performance": "Overheating",
                "charging": "Not tested",
                "duration": "A few days",
                "onset": "All the time",
                "possible_cause": "Liquid exposure",
                "previous_repair": "No",
                "hazards": "No"
            },
            "safety_alert": False,
            "requires_professional_attention": False,
            "diagnosis": None,
            "repair_recommendation": None,
            "created_at": "2026-10-21T05:15:54.051104+00:00"
        }

        from fault_engine import process_questionnaire
        result = process_questionnaire(payload)

        # Safety & Pro attention checks
        self.assertTrue(result["safety_alert"], "Safety alert must be triggered due to liquid exposure")
        self.assertTrue(result["requires_professional_attention"], "Liquid exposure requires professional attention")

        # Diagnosis block verification
        self.assertIsNotNone(result["diagnosis"])
        diag = result["diagnosis"]
        self.assertIn("fault_id", diag)
        self.assertIn("primary_fault", diag)
        self.assertGreater(len(diag["troubleshooting_steps"]), 0)

        # Recommendation block verification
        self.assertIsNotNone(result["repair_recommendation"])
        rec = result["repair_recommendation"]
        self.assertIn("action", rec)
        self.assertIn("estimated_cost", rec)
        self.assertIn("required_tools", rec)


if __name__ == "__main__":
    unittest.main()


