"""
Unit tests for Evidence-Based Root Cause Analysis Engine (evidence_engine.py).

Tests:
1. Multi-source evidence representation (observed, user_reported, imported_specs, confirmed_test).
2. Distinction between component presence and component failure (presence != failure).
3. False-positive visual observation handling (dark area != motherboard burn).
4. Contradiction detection (e.g. CV spinning fan vs user reported stopped; working HDMI vs GPU dead).
5. Categorical confidence ratings (Strongly supported, Possible, Weakly supported, Inconclusive).
6. Missing evidence and recommended next safe check generation.
7. Integration with FaultDiagnosisEngine and backward compatibility.
8. Support for both Laptop and Desktop rule sets.
"""

import unittest
from evidence_engine import (
    EvidenceSource,
    EvidencePolarity,
    ConfidenceCategory,
    EvidenceItem,
    EvidenceContradiction,
    FaultHypothesis,
    EvidenceExtractor,
    EvidenceReasoner,
    EvidenceDiagnosisEngine,
)
from fault_engine import FaultDiagnosisEngine, engine


class TestEvidenceEngine(unittest.TestCase):

    def setUp(self):
        self.diag_engine = EvidenceDiagnosisEngine()
        self.laptop_rules = engine.registry.get("laptop", [])
        self.desktop_rules = engine.registry.get("desktop", [])

    def test_evidence_item_structure(self):
        """Verify EvidenceItem dataclass fields and serialization."""
        item = EvidenceItem(
            evidence_id="EV-TEST-1",
            description="RAM stick seated in slot A",
            source=EvidenceSource.OBSERVED.value,
            related_component="ram",
            evidence_type="visual",
            confidence_reliability=0.85,
            verification_status="verified",
            polarity=EvidencePolarity.NEUTRAL.value,
            reference_id="camera_frame_1.jpg"
        )
        d = item.to_dict()
        self.assertEqual(d["evidence_id"], "EV-TEST-1")
        self.assertEqual(d["source"], "observed")
        self.assertEqual(d["polarity"], "neutral")
        self.assertEqual(d["confidence_reliability"], 0.85)

    def test_presence_is_not_failure(self):
        """
        Detecting a component (e.g. RAM or GPU) in the hardware profile must NOT
        generate a fault or mark the component as defective.
        """
        mock_hw_profile = {
            "cabinet_status": "OPEN_INTERIOR",
            "components": {
                "ram": {
                    "component_name": "RAM Module",
                    "visibility_status": "DETECTED",
                    "confidence": 0.92,
                },
                "gpu": {
                    "component_name": "Dedicated Graphics Card (GPU)",
                    "visibility_status": "DETECTED",
                    "confidence": 0.88,
                }
            }
        }
        items = EvidenceExtractor.extract_from_hardware_profile(mock_hw_profile)
        self.assertEqual(len(items), 3)  # RAM, GPU, Cabinet
        for item in items:
            # Component presence must be NEUTRAL
            self.assertEqual(item.polarity, EvidencePolarity.NEUTRAL.value)

        # Evaluating hypotheses with ONLY presence evidence must result in INCONCLUSIVE
        analysis = self.diag_engine.analyze_evidence(
            device_type="desktop",
            rules=self.desktop_rules,
            hardware_profile=mock_hw_profile
        )
        self.assertTrue(analysis["is_inconclusive"])
        top_hyps = analysis["ranked_hypotheses"]
        if top_hyps:
            self.assertEqual(top_hyps[0]["confidence_category"], ConfidenceCategory.INCONCLUSIVE.value)

    def test_dark_patch_is_not_automatic_burn(self):
        """
        A low-level visual dark patch must not be treated as confirmed motherboard burn.
        """
        mock_cv = {
            "findings": [
                {
                    "id": "burn",
                    "level": "low",
                    "detail": "Localized dark region detected on motherboard surface"
                }
            ]
        }
        items = EvidenceExtractor.extract_from_cv_analysis(mock_cv)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].polarity, EvidencePolarity.NEUTRAL.value)
        self.assertIn("physical confirmation", items[0].description)

    def test_contradiction_detection_fan_spinning(self):
        """
        CV detects spinning fan while user reports fan not spinning.
        Contradiction must be highlighted with a clarifying action.
        """
        items = [
            EvidenceItem(
                evidence_id="EV-1",
                description="Fan motion observation: spinning (motion detected)",
                source=EvidenceSource.OBSERVED.value,
                related_component="cooling_fan",
                evidence_type="visual",
                confidence_reliability=0.85,
                verification_status="verified",
                polarity=EvidencePolarity.NEUTRAL.value
            ),
            EvidenceItem(
                evidence_id="EV-2",
                description="Questionnaire response [power_state]: fan not spinning",
                source=EvidenceSource.USER_REPORTED.value,
                related_component="cooling_fan",
                evidence_type="symptom",
                confidence_reliability=0.85,
                verification_status="unverified",
                polarity=EvidencePolarity.SUPPORTING.value
            )
        ]
        contradictions = EvidenceReasoner.detect_contradictions(items)
        self.assertEqual(len(contradictions), 1)
        self.assertEqual(contradictions[0].component, "cooling_fan")
        self.assertIn("Visual motion analysis", contradictions[0].explanation)
        self.assertIn("clarification_question", contradictions[0].to_dict())

    def test_contradiction_external_display_working(self):
        """
        If external HDMI monitor works normally, but user suspected dead motherboard/GPU,
        contradiction must be flagged and rule regarding total GPU failure contradicted.
        """
        items = [
            EvidenceItem(
                evidence_id="EV-1",
                description="Confirmed troubleshooting check [hdmi_test]: works normally in Windows",
                source=EvidenceSource.CONFIRMED_TEST.value,
                related_component="display",
                evidence_type="test_result",
                confidence_reliability=0.95,
                verification_status="verified",
                polarity=EvidencePolarity.SUPPORTING.value,
                reference_id="test.hdmi"
            ),
            EvidenceItem(
                evidence_id="EV-2",
                description="User stated problem: laptop motherboard dead or gpu dead",
                source=EvidenceSource.USER_REPORTED.value,
                related_component="motherboard",
                evidence_type="symptom",
                confidence_reliability=0.70,
                verification_status="unverified",
                polarity=EvidencePolarity.SUPPORTING.value
            )
        ]
        contradictions = EvidenceReasoner.detect_contradictions(items)
        self.assertEqual(len(contradictions), 1)
        self.assertEqual(contradictions[0].component, "display")

    def test_strong_evidence_hypothesis_ranking(self):
        """
        Multiple corroborating symptoms (e.g. black screen + fan spins + display issue)
        should result in Strongly supported or Possible confidence category.
        """
        analysis = self.diag_engine.analyze_evidence(
            device_type="laptop",
            rules=self.laptop_rules,
            free_text="laptop turns on fans spin but black screen",
            answers={
                "display": "Black screen",
                "power_state": "Lights/fan turn on but no display",
            }
        )
        self.assertFalse(analysis["is_inconclusive"])
        self.assertGreater(len(analysis["ranked_hypotheses"]), 0)
        primary = analysis["primary_hypothesis"]
        self.assertIn(primary["confidence_category"], ["Strongly supported", "Possible"])
        self.assertTrue("black" in primary["fault_name"].lower() or "screen" in primary["fault_name"].lower() or "display" in primary["fault_id"].lower())
        self.assertGreater(len(primary["supporting_evidence"]), 0)
        self.assertGreater(len(primary["missing_evidence"]), 0)
        self.assertTrue(len(primary["recommended_next_check"]) > 0)

    def test_specs_provenance_separation(self):
        """
        Specifications imported from dxdiag/systeminfo are cleanly isolated
        with source=imported_specs and do not invent visual claims.
        """
        specs = {
            "source_format": "txt",
            "ram_total_gb": 32.0,
            "ram_configuration": "2x 16GB DDR4",
            "gpu_model": "NVIDIA GeForce RTX 3070",
            "cpu_model": "Intel Core i7-12700K"
        }
        items = EvidenceExtractor.extract_from_specs(specs)
        self.assertEqual(len(items), 3)
        for item in items:
            self.assertEqual(item.source, EvidenceSource.IMPORTED_SPECS.value)
            self.assertEqual(item.verification_status, "verified")
            self.assertEqual(item.polarity, EvidencePolarity.NEUTRAL.value)

    def test_inconclusive_empty_symptoms(self):
        """
        When no symptoms or only neutral answers are provided, the engine must honestly
        return inconclusive rather than fabricating a diagnosis.
        """
        analysis = self.diag_engine.analyze_evidence(
            device_type="desktop",
            rules=self.desktop_rules,
            free_text="",
            answers={
                "power_state": "starts normally",
                "display": "works normally"
            }
        )
        self.assertTrue(analysis["is_inconclusive"])

    def test_fault_engine_integration(self):
        """
        Verify that fault_engine.diagnose_questionnaire populates
        confidence_category, ranked_hypotheses, and evidence_analysis.
        """
        payload = {
            "device_type": "desktop",
            "answers": {
                "power_state": "Lights/fan turn on but no display",
                "display": "Black screen",
            },
            "free_text": "PC boots fans spin monitor black"
        }
        result = engine.diagnose_questionnaire(payload, use_gemini=False)
        diag = result["diagnosis"]
        self.assertIn("confidence_category", diag)
        self.assertIn("ranked_hypotheses", diag)
        self.assertIn("evidence_analysis", diag)
        self.assertIn(diag["confidence_category"], ["Strongly supported", "Possible", "Weakly supported"])
        self.assertGreater(len(diag["ranked_hypotheses"]), 0)


if __name__ == "__main__":
    unittest.main()
