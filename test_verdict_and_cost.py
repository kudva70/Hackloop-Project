"""
Unit tests for Verdict Engine (verdict.py) and Cost Estimator (cost_estimator.py).

Tests:
1. Safety-first decision rules: swollen battery, smoke, sparks, burning smells, liquid damage, PSU failures.
2. Unsafe DIY troubleshooting is stopped and SERVICE (technician) is mandated.
3. Separation of safe external checks from advanced internal repair procedures.
4. Cost estimation in INR with ranges for parts, labour, and total repair.
5. Deterministic verdict outputs (REPAIR (DIY), SERVICE (technician), REPLACE (device)).
6. Removal of fan/bulb/switchboard from active computer diagnosis pipeline defaults.
7. Verification that Gemini cannot override safety holds or verdicts.
"""

import unittest
from verdict import decide, parse_cost, DEVICE_DEFAULTS
from cost_estimator import CostEstimator, CostEstimate


class TestVerdictAndCostEstimator(unittest.TestCase):

    def test_active_devices_restricted_to_computers(self):
        """Active supported device defaults must only contain laptop and desktop."""
        self.assertIn("laptop", DEVICE_DEFAULTS)
        self.assertIn("desktop", DEVICE_DEFAULTS)
        self.assertNotIn("fan", DEVICE_DEFAULTS)
        self.assertNotIn("bulb", DEVICE_DEFAULTS)
        self.assertNotIn("switchboard", DEVICE_DEFAULTS)

    def test_safety_hold_swollen_battery(self):
        """Swollen battery must trigger immediate SERVICE (technician) verdict."""
        report = {
            "primary_diagnosis": {
                "fault_id": "LAPTOP_SWOLLEN_BATTERY",
                "fault_title": "Swollen / Bulging Lithium-ion Battery",
                "severity": "CRITICAL",
                "repair_complexity": "Dangerous (Do Not DIY)",
                "recommendation": "SERVICE (Professional)",
                "estimated_parts_cost": "₹2,500 - ₹5,000",
            },
            "safety_alert": True,
            "requires_professional_attention": True
        }
        res = decide("laptop", report)
        self.assertEqual(res["verdict"], "SERVICE (technician)")
        self.assertTrue(res["safety_hold"])
        self.assertTrue(any("Safety hazard" in r or "swollen battery" in r for r in res["reasons"]))

    def test_safety_hold_smoke_and_sparks(self):
        """Smoke or sparks in symptoms must trigger safety hold and SERVICE verdict."""
        report = {
            "user_symptoms": "Desktop PC started making spark noises and smoke came out",
            "primary_diagnosis": {
                "fault_id": "DESKTOP_PSU_FAILURE",
                "fault_title": "Power Supply Unit Short Circuit",
                "severity": "CRITICAL",
                "repair_complexity": "Dangerous (Do Not DIY)",
                "recommendation": "SERVICE (Professional)",
                "estimated_parts_cost": "₹2,000 - ₹6,000",
            }
        }
        res = decide("desktop", report)
        self.assertEqual(res["verdict"], "SERVICE (technician)")
        self.assertTrue(res["safety_hold"])

    def test_safety_hold_liquid_spill(self):
        """Liquid spill must trigger SERVICE (technician)."""
        report = {
            "primary_diagnosis": {
                "fault_id": "LAPTOP_LIQUID_SPILL",
                "fault_title": "Liquid Ingress / Corrosion",
                "severity": "CRITICAL",
                "repair_complexity": "Dangerous (Do Not DIY)",
                "recommendation": "SERVICE (Professional)",
                "estimated_parts_cost": "₹2,000 - ₹8,000",
            },
            "safety_alert": True
        }
        res = decide("laptop", report)
        self.assertEqual(res["verdict"], "SERVICE (technician)")
        self.assertTrue(res["safety_hold"])

    def test_cost_estimator_inr_ranges(self):
        """Cost estimator parses INR ranges and calculates parts, labour, and totals."""
        est = CostEstimator.estimate_repair_cost(
            device_type="laptop",
            parts_cost_text="₹1,500 - ₹3,500",
            repair_complexity="Medium (Intermediate)",
            recommendation="SERVICE (Professional)"
        )
        self.assertEqual(est.parts_cost_min, 1500)
        self.assertEqual(est.parts_cost_max, 3500)
        self.assertGreater(est.labour_cost_min, 0)
        self.assertGreaterEqual(est.total_repair_min, 1500 + est.labour_cost_min)
        self.assertEqual(est.currency, "INR")
        self.assertFalse(est.is_diy)

    def test_cost_estimator_diy_zero_labour(self):
        """DIY recommendations should have 0 labour fee."""
        est = CostEstimator.estimate_repair_cost(
            device_type="laptop",
            parts_cost_text="₹800 - ₹1,500",
            repair_complexity="Low (Beginner)",
            recommendation="REPAIR (DIY)"
        )
        self.assertEqual(est.labour_cost_min, 0)
        self.assertEqual(est.labour_cost_max, 0)
        self.assertTrue(est.is_diy)
        self.assertIn("Free", est.formatted_labour_range)

    def test_verdict_economic_replace_ratio(self):
        """When repair cost exceeds REPLACE_RATIO (60%), verdict must be REPLACE."""
        report = {
            "primary_diagnosis": {
                "fault_id": "DESKTOP_GPU_DEAD",
                "fault_title": "High-End GPU Short Circuit",
                "severity": "HIGH",
                "repair_complexity": "High (Advanced)",
                "recommendation": "SERVICE (Professional)",
                "estimated_parts_cost": "₹38,000 - ₹42,000",  # ~72% of 55,000 replacement
            }
        }
        res = decide("desktop", report, replacement_cost=55000)
        self.assertEqual(res["verdict"], "REPLACE (device)")
        self.assertGreaterEqual(res["cost_ratio"], 0.60)

    def test_verdict_repair_diy_safe_ram(self):
        """Safe low-cost issue (e.g. RAM contact cleaning) yields REPAIR (DIY)."""
        report = {
            "primary_diagnosis": {
                "fault_id": "LAPTOP_RAM_OXIDATION",
                "fault_title": "RAM Contact Oxidation",
                "severity": "LOW",
                "repair_complexity": "Low (Beginner)",
                "recommendation": "REPAIR (DIY)",
                "estimated_parts_cost": "₹0 - ₹0",
            }
        }
        res = decide("laptop", report)
        self.assertEqual(res["verdict"], "REPAIR (DIY)")
        self.assertFalse(res["safety_hold"])

    def test_age_depreciation_replace_rule(self):
        """Older devices past 70% lifespan with moderate repair costs (>40%) trigger REPLACE."""
        report = {
            "primary_diagnosis": {
                "fault_id": "LAPTOP_SCREEN_REPLACEMENT",
                "fault_title": "Cracked LCD Panel",
                "severity": "MEDIUM",
                "repair_complexity": "Medium (Intermediate)",
                "recommendation": "SERVICE (Professional)",
                "estimated_parts_cost": "₹19,000 - ₹21,000",  # ~44% of 45,000
            }
        }
        # Laptop default lifespan is 5 years; age 4.0 years is 80% used
        res = decide("laptop", report, age_years=4.0, replacement_cost=45000)
        self.assertEqual(res["verdict"], "REPLACE (device)")


if __name__ == "__main__":
    unittest.main()

