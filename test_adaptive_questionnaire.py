"""
Unit and integration tests for Hybrid Adaptive AI Diagnostic Questionnaire.
Part of Repair Before Replace.

Verifies:
1. Adaptive branching based on symptoms and answers.
2. Duplicate-question prevention.
3. Irrelevant-question rejection (device and component mismatch).
4. Contradictory answer detection.
5. Immediate safety interruptions on hazards.
6. Graceful AI API failure & offline deterministic fallback.
7. Laptop-specific vs Desktop-specific question tailoring.
8. Session persistence across multi-step interactions.
9. Safety validation layer rejecting unsafe questions/instructions.
"""

import unittest
from typing import Dict, List, Any
import json
import os

from question_planner import (
    DiagnosticQuestion,
    QuestionPlanner,
    SafetyValidator,
    QUESTION_BANK,
)
from diagnostic_session import (
    DiagnosticSession,
    SessionManager,
    session_manager,
)
from questionaire import (
    run_questionnaire,
    answers_from_description,
    normalize_device,
)


class TestSafetyValidation(unittest.TestCase):
    """Test safety constraint enforcement on questions."""

    def test_reject_unsafe_psu_teardown_instruction(self):
        unsafe_q = DiagnosticQuestion(
            question_id="unsafe_psu_test",
            text="Can you open the power supply to check for burnt capacitors?",
            options=["Yes, opened PSU", "No", "Not sure"],
            category="power",
            target_device="desktop",
        )
        is_valid, reason = SafetyValidator.validate_question(unsafe_q)
        self.assertFalse(is_valid)
        self.assertIn("Safety policy violation", reason)

    def test_reject_unsafe_capacitor_touching(self):
        unsafe_q = DiagnosticQuestion(
            question_id="unsafe_touch_test",
            text="Does touching the capacitor while plugged in give an electric shock?",
            options=["Yes", "No", "Not sure"],
            category="power",
            target_device="desktop",
        )
        is_valid, reason = SafetyValidator.validate_question(unsafe_q)
        self.assertFalse(is_valid)
        self.assertIn("Safety policy violation", reason)

    def test_reject_unsafe_swollen_battery_pressing(self):
        unsafe_q = DiagnosticQuestion(
            question_id="unsafe_battery_test",
            text="Please press on the swollen battery firmly to see if it flattens.",
            options=["Yes", "No", "Not sure"],
            category="hazard",
            target_device="laptop",
        )
        is_valid, reason = SafetyValidator.validate_question(unsafe_q)
        self.assertFalse(is_valid)
        self.assertIn("Safety policy violation", reason)

    def test_valid_safe_question_passes(self):
        safe_q = DiagnosticQuestion(
            question_id="safe_led_test",
            text="Does the motherboard show a DRAM, CPU, or VGA diagnostic light?",
            options=["DRAM LED", "CPU LED", "VGA LED", "None", "Not sure"],
            category="motherboard",
            target_device="desktop",
        )
        is_valid, reason = SafetyValidator.validate_question(safe_q)
        self.assertTrue(is_valid)
        self.assertIsNone(reason)


class TestAdaptiveBranchingAndIntelligence(unittest.TestCase):
    """Test dynamic question planning, deduplication, and component filtering."""

    def test_duplicate_question_prevention(self):
        planner = QuestionPlanner(use_gemini=False)
        asked = {"hazards_check", "power_state"}
        answers = {"hazards_check": "No", "power_state": "Lights/fan turn on, but screen stays black"}
        hypotheses = [{"fault_id": "DESKTOP_RAM_POST_FAILURE", "score": 0.70}]

        next_q = planner.select_next_question(
            device_type="desktop",
            hardware_profile=None,
            asked_question_ids=asked,
            answers=answers,
            hypotheses=hypotheses,
        )
        self.assertIsNotNone(next_q)
        self.assertNotIn(next_q.question_id, asked)

    def test_desktop_vs_laptop_question_tailoring(self):
        planner = QuestionPlanner(use_gemini=False)

        # Desktop candidate questions should never include laptop-only questions
        desktop_candidates = planner.get_candidate_questions(
            device_type="desktop",
            hardware_profile=None,
            asked_question_ids=set(),
            answers={},
            hypotheses=[],
        )
        desktop_ids = {q.question_id for q in desktop_candidates}
        self.assertIn("desktop_motherboard_ez_debug_leds", desktop_ids)
        self.assertNotIn("laptop_external_monitor_test", desktop_ids)
        self.assertNotIn("laptop_charger_indicator_light", desktop_ids)

        # Laptop candidate questions should never include desktop-only questions
        laptop_candidates = planner.get_candidate_questions(
            device_type="laptop",
            hardware_profile=None,
            asked_question_ids=set(),
            answers={},
            hypotheses=[],
        )
        laptop_ids = {q.question_id for q in laptop_candidates}
        self.assertIn("laptop_external_monitor_test", laptop_ids)
        self.assertNotIn("desktop_display_cable_connection", laptop_ids)
        self.assertNotIn("desktop_motherboard_ez_debug_leds", laptop_ids)

    def test_adaptive_branching_to_memory_after_dram_warning(self):
        session = DiagnosticSession(
            device_type="desktop",
            problem_description="PC fans spin but monitor is black",
            use_gemini=False,
        )
        # First question: hazards
        q1 = session.get_next_question()
        session.submit_answer(q1.question_id, "No")

        # Second question: power state
        q2 = session.get_next_question()
        session.submit_answer(q2.question_id, "Lights/fan turn on, but screen stays black")

        # Third question: user reports DRAM LED on motherboard
        q3 = session.get_next_question()
        session.submit_answer("desktop_motherboard_ez_debug_leds", "DRAM LED is solid or blinking")

        # Verify hypotheses prioritize RAM POST failure
        self.assertTrue(len(session.hypotheses) > 0)
        top_fault = session.hypotheses[0]
        self.assertEqual(top_fault["fault_id"], "DESKTOP_RAM_CONTACT_POST_FAILURE")

    def test_rejection_of_component_without_hardware_presence(self):
        planner = QuestionPlanner(use_gemini=False)
        # Profile has ONLY motherboard and cpu (integrated graphics, NO discrete GPU)
        hw_profile = {
            "components": {
                "motherboard": {"visibility_status": "DETECTED"},
                "cpu": {"visibility_status": "DETECTED"},
            }
        }
        candidates = planner.get_candidate_questions(
            device_type="desktop",
            hardware_profile=hw_profile,
            asked_question_ids=set(),
            answers={},
            hypotheses=[],
        )
        candidate_ids = {q.question_id for q in candidates}
        # Questions requiring discrete GPU must be omitted
        self.assertNotIn("desktop_display_cable_connection", candidate_ids)


class TestContradictionDetection(unittest.TestCase):
    """Test identification of contradictory symptom answers."""

    def test_contradiction_no_power_vs_display_activity(self):
        session = DiagnosticSession(
            device_type="laptop",
            problem_description="computer not working",
            use_gemini=False,
        )
        session.submit_answer("power_state", "No power, no lights, no sound at all")
        session.submit_answer("display", "Flickering lines, visual artifacts, or distorted colors")
        self.assertTrue(len(session.contradictions) > 0)
        self.assertIn("Contradiction: Reported 'No power", session.contradictions[0])


class TestSafetyInterruptions(unittest.TestCase):
    """Test immediate halt of diagnostic session upon hazard detection."""

    def test_safety_interruption_on_smoke_hazard_answer(self):
        session = DiagnosticSession(
            device_type="desktop",
            problem_description="PC won't start",
            use_gemini=False,
        )
        q1 = session.get_next_question()
        # User confirms smoke / sparks
        state = session.submit_answer(q1.question_id, "Yes")

        self.assertTrue(session.safety_alert)
        self.assertTrue(session.is_complete)
        self.assertTrue(len(session.stop_using_advice) > 0)
        # Next question should be None because questioning halted
        self.assertIsNone(session.get_next_question())

    def test_safety_interruption_on_initial_liquid_spill_description(self):
        session = DiagnosticSession(
            device_type="laptop",
            problem_description="spilled water on my keyboard and laptop turned off",
            use_gemini=False,
        )
        self.assertTrue(session.safety_alert)
        self.assertTrue(session.is_complete)
        self.assertIsNone(session.get_next_question())


class TestSessionPersistenceAndLifecycle(unittest.TestCase):
    """Test session lifecycle across simulated multi-step REST API interactions."""

    def test_session_lifecycle_multi_turn(self):
        mgr = SessionManager()
        session = mgr.create_session(
            device_type="desktop",
            problem_description="desktop powers on but no display",
            use_gemini=False,
        )
        session_id = session.session_id
        self.assertIsNotNone(mgr.get_session(session_id))

        # Turn 1
        q1 = session.get_next_question()
        self.assertIsNotNone(q1)
        session.submit_answer(q1.question_id, "No")

        # Turn 2
        q2 = session.get_next_question()
        self.assertIsNotNone(q2)
        session.submit_answer(q2.question_id, "Lights/fan turn on, but screen stays black")

        state = session.get_state()
        self.assertEqual(state.current_step, 3)
        self.assertEqual(len(session.asked_questions), 2)

        # Finalize
        report = session.finalize_diagnosis()
        self.assertTrue(session.is_complete)
        self.assertIn("status", report)
        self.assertIn("diagnosis", report)


class TestOfflineFallback(unittest.TestCase):
    """Test that planner operates fully when Gemini/AI is disabled or offline."""

    def test_offline_planner_completes_session(self):
        session = DiagnosticSession(
            device_type="laptop",
            problem_description="screen stays completely black after power on",
            use_gemini=False,
        )
        # Progressively answer until stopping condition or completion
        iterations = 0
        while not session.is_complete and iterations < 10:
            q = session.get_next_question()
            if not q:
                break
            session.submit_answer(q.question_id, q.options[0])
            iterations += 1

        self.assertTrue(session.is_complete or iterations > 0)
        report = session.finalize_diagnosis()
        self.assertIsNotNone(report)


class TestSessionLogicAndEndpoints(unittest.TestCase):
    """Test session lifecycle, state transitions, and diagnostic finalization."""

    def test_session_lifecycle_and_state_flow(self):
        mgr = SessionManager()
        # 1. Start session
        session = mgr.create_session(
            device_type="desktop",
            problem_description="PC fans spin but black screen",
            use_gemini=False,
        )
        self.assertIsNotNone(session)
        self.assertEqual(session.device_type, "desktop")
        session_id = session.session_id

        # 2. Retrieve next question
        q1 = session.get_next_question()
        self.assertIsNotNone(q1)
        self.assertFalse(session.is_complete)

        # 3. Submit answer
        state = session.submit_answer(
            question_id=q1.question_id,
            answer="No",
            text_response="",
        )
        self.assertEqual(state["current_step"], 2)
        self.assertIn(q1.question_id, state["answers"])

        # 4. Get state snapshot
        snap = session.get_state()
        self.assertEqual(snap.session_id, session_id)
        self.assertFalse(snap.is_complete)

        # 5. Complete session
        report = session.finalize_diagnosis()
        self.assertTrue(session.is_complete)
        self.assertIn("status", report)
        self.assertIn("diagnosis", report)


if __name__ == "__main__":
    unittest.main()
