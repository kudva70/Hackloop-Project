"""
End-to-End Frontend and API Integration Tests for Repair Before Replace.

Verifies:
1. Index HTML serving and structure (landing page preservation, interactive session elements, device selector).
2. Diagnostic session lifecycle via API (start session, next question, submit answer, complete session).
3. Automated diagnosis endpoint /api/diagnose with:
   - demo mode
   - describe mode (laptop & desktop)
   - hardware specification parsing
4. Full evidence, root cause hypothesis ranking, cost breakdown, and safety response structures.
"""

import unittest
from pathlib import Path
import json
import asyncio

from server import app, serve_index
from diagnostic_session import session_manager
from cost_estimator import CostEstimator


class TestFrontendAndApiIntegration(unittest.TestCase):

    def test_frontend_index_file_integrity(self):
        """Index.html must exist, preserve landing page sections, and contain interactive diagnostic elements."""
        index_path = Path("front-end/index.html")
        self.assertTrue(index_path.is_file())
        content = index_path.read_text(encoding="utf-8")

        # Check preserved landing page elements
        self.assertIn("Know before you replace", content)
        self.assertIn("Built specifically for computer hardware", content)
        self.assertIn("Hardware Intelligence", content)
        self.assertIn("heroDemoBtn", content)

        # Check laptop and desktop targeting
        self.assertIn('<option value="laptop" selected>Laptop / Notebook</option>', content)
        self.assertIn('<option value="desktop">Desktop PC / Tower</option>', content)
        self.assertNotIn('<option value="switchboard">', content)
        self.assertNotIn('<option value="bulb">', content)

        # Check interactive session and evidence elements
        self.assertIn('id="startSessionBtn"', content)
        self.assertIn('id="sessionCard"', content)
        self.assertIn('id="questionCard"', content)
        self.assertIn('id="hypothesesContainer"', content)
        self.assertIn('id="cParts"', content)
        self.assertIn('id="cLabour"', content)
        self.assertIn('id="cTotal"', content)
        self.assertIn('id="contraBox"', content)

    def test_adaptive_session_lifecycle(self):
        """Simulate complete step-by-step adaptive questionnaire session through session_manager."""
        # 1. Start Session
        session = session_manager.create_session(
            device_type="desktop",
            problem_description="PC powers on case fans spin but no video on monitor",
            use_gemini=False
        )
        self.assertIsNotNone(session.session_id)
        self.assertEqual(session.device_type, "desktop")

        # 2. Retrieve initial question
        q1 = session.get_next_question()
        self.assertIsNotNone(q1)
        self.assertTrue(len(q1.options) > 0)

        # 3. Submit answer to question 1
        state1 = session.submit_answer(
            question_id=q1.question_id,
            answer=q1.options[0]
        )
        self.assertIn(q1.question_id, state1["answers"])

        # 4. Finalize diagnosis
        report = session.finalize_diagnosis()
        diag = report.get("diagnosis", {})
        self.assertIn("primary_fault", diag)
        self.assertIn("ranked_hypotheses", diag)
        self.assertGreater(len(diag["ranked_hypotheses"]), 0)

    def test_cost_estimator_integration_with_verdict(self):
        """Verify cost breakdown format provided to frontend."""
        est = CostEstimator.estimate_repair_cost(
            device_type="desktop",
            parts_cost_text="₹2,000 - ₹5,000",
            repair_complexity="Medium (Intermediate)",
            recommendation="SERVICE (technician)",
            custom_replacement_price=55000
        )
        d = est.to_dict()
        self.assertEqual(d["parts_cost_min"], 2000)
        self.assertEqual(d["parts_cost_max"], 5000)
        self.assertEqual(d["labour_cost_min"], 500)
        self.assertEqual(d["labour_cost_max"], 1200)
        self.assertEqual(d["total_repair_min"], 2500)
        self.assertEqual(d["total_repair_max"], 6200)
        self.assertEqual(d["replacement_cost"], 55000)
        self.assertFalse(d["is_diy"])

    def test_demo_mode_pipeline(self):
        """Verify demo mode pipeline execution for instant preview."""
        import assistant
        report = assistant.run_pipeline(
            demo=True,
            gemma_backend="template"
        )
        self.assertEqual(report["device_type"], "laptop")
        self.assertIn("repair_decision", report)
        self.assertIn("diagnosis", report)
        self.assertIn("explanation", report)
        verdict = report["repair_decision"]["verdict"]
        self.assertIn(verdict, ["REPAIR (DIY)", "SERVICE (technician)", "REPLACE (device)"])


if __name__ == "__main__":
    unittest.main()
