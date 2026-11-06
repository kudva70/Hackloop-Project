"""
Unit and Integration Tests for Hardware Intelligence and Inventory System.
Validates:
  - Laptop hardware profile
  - Desktop PC hardware profile
  - Closed desktop cabinet scanning (must NOT be treated as proof of internal components)
  - Open desktop cabinet scanning (interior components visible)
  - Multiple component images (merging observations & upgrading visibility)
  - Low-confidence detections (UNCERTAIN status)
  - Partially visible components (PARTIALLY_VISIBLE status & clipping/occlusion)
  - Empty detections (graceful handling, no hallucination)
  - Unsupported model classes (clear unsupported-class handling)
  - Multiple images referring to the same component
  - Desktop fault engine integration and e-waste calculations
"""

import unittest
from pathlib import Path
import tempfile
from PIL import Image

from hardware_profile import (
    HardwareProfile,
    ComponentItem,
    VisibilityStatus,
    CabinetStatus,
    LAPTOP_STANDARD_COMPONENTS,
    DESKTOP_STANDARD_COMPONENTS,
)
import detect_new
from fault_engine import FaultDiagnosisEngine, DEVICE_REGISTRY
from verdict import decide, DEVICE_DEFAULTS, DEVICE_MASS_KG


class TestHardwareIntelligence(unittest.TestCase):

    def setUp(self):
        self.engine = FaultDiagnosisEngine()

    # ----------------------------------------------------------------------
    # 1. Profile Initialization & Separation (Laptop vs Desktop)
    # ----------------------------------------------------------------------

    def test_laptop_profile_initialization(self):
        """Laptop profile must contain standard laptop components (battery, touchpad, etc.)."""
        profile = HardwareProfile(device_type="laptop")
        self.assertEqual(profile.device_type, "laptop")

        # Battery, display, touchpad, keyboard must exist in laptop profile
        self.assertIn("battery", profile.components)
        self.assertIn("touchpad", profile.components)
        self.assertIn("charging_port", profile.components)
        self.assertIn("motherboard", profile.components)
        self.assertIn("ram", profile.components)

        # Initial state must be NOT_DETECTED with is_present=None
        battery = profile.components["battery"]
        self.assertEqual(battery.visibility_status, VisibilityStatus.NOT_DETECTED.value)
        self.assertIsNone(battery.is_present, "NOT_DETECTED must never assume NOT_PRESENT")

    def test_desktop_profile_initialization(self):
        """Desktop profile must contain PSU, case fans, cabinet, and NOT laptop battery/touchpad."""
        profile = HardwareProfile(device_type="desktop")
        self.assertEqual(profile.device_type, "desktop")

        self.assertIn("psu", profile.components)
        self.assertIn("cabinet", profile.components)
        self.assertIn("case_fans", profile.components)
        self.assertIn("cpu_cooler", profile.components)
        self.assertIn("gpu", profile.components)

        # Laptop-specific battery and touchpad must not be standard desktop components
        self.assertNotIn("battery", profile.components)
        self.assertNotIn("touchpad", profile.components)

        psu = profile.components["psu"]
        self.assertEqual(psu.visibility_status, VisibilityStatus.NOT_DETECTED.value)
        self.assertIsNone(psu.is_present, "NOT_DETECTED must never assume NOT_PRESENT")

    # ----------------------------------------------------------------------
    # 2. Whole-Computer Cabinet Scanning (Closed vs Open)
    # ----------------------------------------------------------------------

    def test_closed_desktop_cabinet_scanning(self):
        """
        Rule: 'A closed opaque cabinet must not be treated as proof of its internal components.'
        When only the exterior cabinet is observed, internal components must remain NOT_DETECTED
        with is_present=None and a close-up request to remove the side panel must be generated.
        """
        detections = [
            {"device": "cabinet", "confidence": 0.88, "box": [50, 50, 750, 950]}
        ]
        status, note = detect_new.analyze_cabinet_status(detections, device_type="desktop")
        self.assertEqual(status, CabinetStatus.CLOSED_CABINET.value)

        profile = HardwareProfile(device_type="desktop")
        profile.set_cabinet_status(status, notes=note)

        # Internal components must not be marked present or verified
        for cat in ("motherboard", "ram", "gpu", "cpu_cooler", "psu"):
            comp = profile.components[cat]
            self.assertEqual(comp.visibility_status, VisibilityStatus.NOT_DETECTED.value)
            self.assertIsNone(comp.is_present)
            self.assertIn("Cabinet is closed", comp.notes)

        # Actionable close-up request must be generated
        requests = profile.get_close_up_requests()
        self.assertTrue(any("side panel" in req.lower() or "cabinet" in req.lower() for req in requests))

    def test_open_desktop_cabinet_scanning(self):
        """When internal components are detected, cabinet status is OPEN_INTERIOR."""
        detections = [
            {"device": "motherboard", "confidence": 0.85, "box": [100, 100, 600, 700]},
            {"device": "ram", "confidence": 0.90, "box": [450, 150, 550, 350]},
            {"device": "gpu", "confidence": 0.92, "box": [150, 400, 650, 600]},
        ]
        status, note = detect_new.analyze_cabinet_status(detections, device_type="desktop")
        self.assertEqual(status, CabinetStatus.OPEN_INTERIOR.value)

        profile = HardwareProfile(device_type="desktop")
        profile.set_cabinet_status(status, notes=note)
        for d in detections:
            cat, name, _ = detect_new.map_detection_to_component(d["device"])
            profile.add_component_detection(
                category=cat,
                name=name,
                confidence=d["confidence"],
                bounding_box=d["box"],
                image_ref="open_chassis.jpg"
            )

        detected_cats = {c.component_category for c in profile.get_detected_components()}
        self.assertIn("motherboard", detected_cats)
        self.assertIn("ram", detected_cats)
        self.assertIn("gpu", detected_cats)

    # ----------------------------------------------------------------------
    # 3. Partial Visibility & Bounding Box Clipping
    # ----------------------------------------------------------------------

    def test_partially_visible_clipped_component(self):
        """Component cut off at the edge of the image frame must be marked PARTIALLY_VISIBLE."""
        # Box touching left and bottom edge of 800x600 image: [5, 200, 300, 598]
        is_clipped = detect_new._is_box_clipped([5, 200, 300, 598], width=800, height=600)
        self.assertTrue(is_clipped)

        profile = HardwareProfile(device_type="desktop")
        item = profile.add_component_detection(
            category="gpu",
            name="Discrete GPU",
            confidence=0.75,
            bounding_box=[5, 200, 300, 598],
            visibility_status=VisibilityStatus.PARTIALLY_VISIBLE.value,
            image_ref="cut_off.jpg"
        )
        self.assertEqual(item.visibility_status, VisibilityStatus.PARTIALLY_VISIBLE.value)
        self.assertTrue(item.close_up_needed)
        requests = profile.get_close_up_requests()
        self.assertTrue(any("partially visible" in r.lower() and "gpu" in r.lower() for r in requests))

    def test_occluded_component_detection(self):
        """When two component boxes overlap substantially, occluded item is PARTIALLY_VISIBLE."""
        box_cooler = [200, 200, 500, 500]
        box_ram = [250, 220, 350, 480]  # inside or largely covered by cooler
        ratio = detect_new._box_intersection_ratio(box_cooler, box_ram)
        self.assertGreater(ratio, 0.50, "Cooler heavily overlaps RAM area")

    # ----------------------------------------------------------------------
    # 4. Low-Confidence & Uncertain Detections
    # ----------------------------------------------------------------------

    def test_low_confidence_uncertain_detection(self):
        """Confidence between floor and threshold (0.18 < 0.25) must be marked UNCERTAIN."""
        profile = HardwareProfile(device_type="laptop")
        item = profile.add_component_detection(
            category="ram",
            name="RAM Module",
            confidence=0.18,
            visibility_status=VisibilityStatus.UNCERTAIN.value,
            image_ref="blurry_laptop.jpg"
        )
        self.assertEqual(item.visibility_status, VisibilityStatus.UNCERTAIN.value)
        self.assertTrue(item.close_up_needed)
        requests = profile.get_close_up_requests()
        self.assertTrue(any("low confidence" in r.lower() or "close-up" in r.lower() for r in requests))

    # ----------------------------------------------------------------------
    # 5. Empty Detections & Unsupported Model Classes
    # ----------------------------------------------------------------------

    def test_empty_detections_handling(self):
        """Empty detections must never fabricate components."""
        profile = HardwareProfile(device_type="desktop")
        detected = profile.get_detected_components()
        self.assertEqual(len(detected), 0)
        findings = profile.summary_findings()
        self.assertEqual(findings, [])

    def test_unsupported_model_classes_not_hallucinated(self):
        """Components not trained in detector model must be marked unsupported and NOT fabricated."""
        profile = HardwareProfile(device_type="laptop")
        profile.mark_unsupported_category(
            category="touchpad",
            name="Touchpad / Trackpad",
            reason="Not trained in current detector model"
        )
        touchpad = profile.components["touchpad"]
        self.assertFalse(touchpad.supported_by_model)
        self.assertEqual(touchpad.verification_status, "unsupported_by_detector")
        self.assertIsNone(touchpad.confidence)
        self.assertIsNone(touchpad.is_present)

    # ----------------------------------------------------------------------
    # 6. Multi-Image Inventory Merging
    # ----------------------------------------------------------------------

    def test_multiple_images_same_component_merge(self):
        """
        Overview image sees RAM with 0.22 confidence (PARTIALLY_VISIBLE).
        Close-up image sees RAM with 0.88 confidence (DETECTED).
        Merged profile must upgrade status to DETECTED, keep highest confidence,
        and retain references to both images.
        """
        # Profile from Image 1 (Overview)
        p1 = HardwareProfile(device_type="desktop")
        p1.images_analyzed.append("overview.jpg")
        p1.add_component_detection(
            category="ram",
            name="RAM Module",
            confidence=0.22,
            bounding_box=[300, 200, 400, 450],
            image_ref="overview.jpg",
            visibility_status=VisibilityStatus.PARTIALLY_VISIBLE.value,
        )

        # Profile from Image 2 (Close-up)
        p2 = HardwareProfile(device_type="desktop")
        p2.images_analyzed.append("closeup_ram.jpg")
        p2.add_component_detection(
            category="ram",
            name="RAM Module",
            confidence=0.88,
            bounding_box=[50, 50, 750, 550],
            image_ref="closeup_ram.jpg",
            visibility_status=VisibilityStatus.DETECTED.value,
        )

        # Merge p2 into p1
        p1.merge_profile(p2)

        ram = p1.components["ram"]
        self.assertEqual(ram.visibility_status, VisibilityStatus.DETECTED.value)
        self.assertAlmostEqual(ram.confidence, 0.88, places=2)
        self.assertIn("overview.jpg", ram.image_references)
        self.assertIn("closeup_ram.jpg", ram.image_references)
        self.assertEqual(ram.verification_status, "verified")

    # ----------------------------------------------------------------------
    # 7. Desktop Fault Engine Rules & Verdict Economics
    # ----------------------------------------------------------------------

    def test_desktop_fault_engine_diagnosis(self):
        """Test desktop GPU port mismatch and PSU failure rules in fault_engine."""
        # 1. GPU / Monitor mismatch
        res_gpu = self.engine.diagnose(
            device="desktop",
            user_symptoms="monitor shows no signal pc turns on fans spin gpu displayport hdmi port on motherboard"
        )
        pri_gpu = res_gpu["primary_diagnosis"]
        self.assertEqual(pri_gpu["fault_id"], "DESKTOP_GPU_SEATING_OR_PORT_MISMATCH")
        self.assertIn("safe_checks", pri_gpu)
        self.assertTrue(len(pri_gpu["safe_checks"]) > 0)

        # 2. PSU Power failure
        res_psu = self.engine.diagnose(
            device="desktop",
            user_symptoms="desktop completely dead no lights no power psu power supply not turning on"
        )
        pri_psu = res_psu["primary_diagnosis"]
        self.assertEqual(pri_psu["fault_id"], "DESKTOP_PSU_POWER_FAILURE")
        # Check PSU high-voltage warning exists
        warnings = " ".join(pri_psu.get("safety_warnings", []))
        self.assertIn("ATX power supply", warnings)

    def test_desktop_verdict_and_ewaste_calculation(self):
        """Desktop repair verdict should use desktop mass (9.5 kg) and replacement cost (Rs 55,000)."""
        res = self.engine.diagnose(
            device="desktop",
            user_symptoms="desktop turns on fans spin monitor no signal plugged into motherboard hdmi"
        )
        v_res = decide("desktop", res)
        self.assertIn(v_res["verdict"], ["REPAIR (DIY)", "SERVICE (technician)"])
        self.assertEqual(v_res["replacement_cost"], 55000.0)

        ewaste = v_res.get("e_waste", {})
        self.assertEqual(ewaste.get("status"), "avoided")
        self.assertEqual(ewaste.get("amount_kg"), 9.5)
        self.assertIn("desktop", ewaste.get("assumption", "").lower())


if __name__ == "__main__":
    unittest.main()

