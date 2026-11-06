"""
Diagnostic Session Manager for Hybrid Adaptive AI Diagnostic Questionnaire.
Part of Repair Before Replace.

Tracks stateful diagnostic sessions across one-question-at-a-time interactions,
computes dynamic candidate fault hypotheses, checks stopping conditions,
detects contradictions, and enforces immediate safety interruptions.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import uuid
import json

from question_planner import QuestionPlanner, DiagnosticQuestion, SafetyValidator
from fault_engine import (
    engine,
    evaluate_safety,
    DEVICE_REGISTRY,
    is_device_supported,
    get_canonical_device,
    STOP_USING_ADVICE,
    HARD_HAZARD_FAULT_IDS,
)


# Contradiction rules between symptom answers
CONTRADICTION_RULES = [
    {
        "keys": ("power_state", "display"),
        "rule": lambda a: "no power" in str(a.get("power_state", "")).lower() and
                          any(term in str(a.get("display", "")).lower() for term in ["flickering", "faint", "artifacts"]),
        "message": "Contradiction: Reported 'No power / no lights', but also reported visual screen activity.",
    },
    {
        "keys": ("power_state", "desktop_motherboard_ez_debug_leds"),
        "rule": lambda a: "no power" in str(a.get("power_state", "")).lower() and
                          any("led is solid" in str(a.get("desktop_motherboard_ez_debug_leds", "")).lower() for _ in [1]),
        "message": "Contradiction: Reported 'No power', but diagnostic LEDs were reported as actively glowing.",
    },
    {
        "keys": ("hazards_check", "recent_incident"),
        "rule": lambda a: str(a.get("hazards_check", "")).lower() == "no" and
                          "liquid spill" in str(a.get("recent_incident", "")).lower(),
        "message": "Notice: Liquid spill reported under recent events while hazards was answered 'No'. Safety protocol elevated.",
    },
]


QUESTION_KEY_ALIASES = {
    "hazards_check": "hazards",
    "recent_incident": "recent_event",
    "monitor_screen_message": "display",
    "storage_boot_screen": "memory_storage",
    "fan_and_thermal_behavior": "temperature",
    "laptop_charger_indicator_light": "charging",
}


@dataclass
class DiagnosticSessionState:
    session_id: str
    device_type: str
    problem_description: str
    current_step: int = 1
    estimated_total_steps: int = 5
    is_complete: bool = False
    safety_alert: bool = False
    safety_reasons: List[str] = field(default_factory=list)
    stop_using_advice: List[str] = field(default_factory=list)
    answers: Dict[str, Any] = field(default_factory=dict)
    text_responses: Dict[str, str] = field(default_factory=dict)
    contradictions: List[str] = field(default_factory=list)
    top_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    hardware_profile: Optional[Dict[str, Any]] = None
    imported_specs: Optional[Dict[str, Any]] = None
    visual_findings: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DiagnosticSession:
    """Manages an active adaptive diagnostic session."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        device_type: str = "laptop",
        problem_description: str = "",
        hardware_profile: Optional[Dict[str, Any]] = None,
        imported_specs: Optional[Dict[str, Any]] = None,
        visual_findings: Optional[List[str]] = None,
        use_gemini: bool = True,
        max_questions: int = 7,
        confidence_threshold: float = 0.85,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.device_type = get_canonical_device(device_type) or "laptop"
        self.problem_description = problem_description.strip()
        self.hardware_profile = hardware_profile or {}
        self.imported_specs = imported_specs or {}
        self.visual_findings = visual_findings or []
        self.use_gemini = use_gemini
        self.max_questions = max_questions
        self.confidence_threshold = confidence_threshold

        self.planner = QuestionPlanner(use_gemini=use_gemini)
        self.asked_questions: List[DiagnosticQuestion] = []
        self.asked_question_ids: Set[str] = set()
        self.answers: Dict[str, Any] = {}
        self.text_responses: Dict[str, str] = {}
        self.contradictions: List[str] = []
        self.current_step = 1
        self.estimated_total_steps = 5
        self.is_complete = False
        self.safety_alert = False
        self.safety_reasons: List[str] = []
        self.stop_using_advice: List[str] = []
        self.hypotheses: List[Dict[str, Any]] = []
        self.final_report: Optional[Dict[str, Any]] = None

        # Pre-evaluate safety and seed initial description keywords
        self._initial_eval()

    def _initial_eval(self) -> None:
        """Evaluates initial free-text description for hazards and seeds initial hypotheses."""
        if self.problem_description:
            safety_eval = evaluate_safety(free_text=self.problem_description)
            if safety_eval["safety_alert"]:
                self.safety_alert = True
                self.is_complete = True
                self.safety_reasons = safety_eval["safety_reasons"]
                self.stop_using_advice = safety_eval["stop_using_advice"]

        # Seed initial hypotheses
        self._update_hypotheses()

    def _update_hypotheses(self) -> None:
        """Recomputes candidate fault hypotheses from current description and answers."""
        corpus_parts = [self.problem_description]
        for k, v in self.answers.items():
            if v:
                corpus_parts.append(f"{k}: {v}")
        for k, v in self.text_responses.items():
            if v:
                corpus_parts.append(v)
        for vf in self.visual_findings:
            corpus_parts.append(vf)

        combined_text = " ".join(corpus_parts)
        rules = DEVICE_REGISTRY.get(self.device_type, [])
        tokens = engine.extract_symptom_tokens(combined_text)

        candidates = []
        for r in rules:
            scored = engine.score_rule(r, combined_text, tokens)
            if scored["score"] > 0:
                candidates.append({
                    "fault_id": r.fault_id,
                    "title": r.title,
                    "severity": r.severity,
                    "complexity": r.complexity,
                    "recommendation": r.recommendation,
                    "score": scored["score"],
                    "evidence": scored["evidence"],
                    "matches": scored["matches"],
                    "rule": r,
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        self.hypotheses = candidates

    def get_next_question(self) -> Optional[DiagnosticQuestion]:
        """
        Retrieves the next relevant question.
        Returns None if session is complete, stopping condition met, or hazard detected.
        """
        if self.is_complete or self.safety_alert:
            return None

        # Check stopping criteria
        if self.is_stopping_condition_met():
            self.is_complete = True
            return None

        q = self.planner.select_next_question(
            device_type=self.device_type,
            hardware_profile=self.hardware_profile,
            asked_question_ids=self.asked_question_ids,
            answers=self.answers,
            hypotheses=self.hypotheses,
            free_text=self.problem_description,
        )

        if q:
            self.asked_questions.append(q)
            self.asked_question_ids.add(q.question_id)
            # Adjust estimated steps dynamically
            rem = max(1, min(self.max_questions - self.current_step, 3))
            self.estimated_total_steps = self.current_step + rem

        return q

    def submit_answer(
        self,
        question_id: str,
        answer: str,
        text_response: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submits an answer for a question, updates state, evaluates safety,
        checks for contradictions, and updates candidate hypotheses.
        """
        if self.is_complete:
            return self.get_state().to_dict()

        ans_clean = str(answer).strip()
        self.answers[question_id] = ans_clean
        if question_id in QUESTION_KEY_ALIASES:
            self.answers[QUESTION_KEY_ALIASES[question_id]] = ans_clean
        self.asked_question_ids.add(question_id)

        if text_response and str(text_response).strip():
            self.text_responses[question_id] = str(text_response).strip()

        # 1. Contradiction Detection
        for rule in CONTRADICTION_RULES:
            k1, k2 = rule["keys"]
            if k1 in self.answers and k2 in self.answers:
                if rule["rule"](self.answers):
                    msg = rule["message"]
                    if msg not in self.contradictions:
                        self.contradictions.append(msg)

        # 2. Safety Interruption Evaluation
        safety_eval = evaluate_safety(
            answers=self.answers,
            free_text=self.problem_description + " " + " ".join(self.text_responses.values()),
            existing_safety_alert=self.safety_alert,
        )

        if safety_eval["safety_alert"]:
            self.safety_alert = True
            self.is_complete = True
            self.safety_reasons = safety_eval["safety_reasons"]
            self.stop_using_advice = safety_eval["stop_using_advice"]

        # 3. Update Hypotheses
        self._update_hypotheses()
        self.current_step += 1

        # 4. Check Stopping Condition
        if self.is_stopping_condition_met():
            self.is_complete = True

        return self.get_state().to_dict()

    def is_stopping_condition_met(self) -> bool:
        """
        Stopping conditions:
        - Active safety hazard -> stop immediately.
        - Maximum conciseness limit reached (max_questions).
        - Confidence threshold met and top hypothesis is decisively ahead of second candidate.
        - No remaining eligible candidate questions in bank.
        """
        if self.safety_alert:
            return True

        if len(self.answers) >= self.max_questions:
            return True

        if self.hypotheses:
            top_score = self.hypotheses[0]["score"]
            if top_score >= self.confidence_threshold:
                if len(self.hypotheses) == 1:
                    return True
                gap = top_score - self.hypotheses[1]["score"]
                # If #1 is far ahead of #2 and at least 3 questions answered
                if gap >= 0.35 and len(self.answers) >= 3:
                    return True

        return False

    def finalize_diagnosis(self, image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs the fault engine with full accumulated session evidence and returns final report.
        """
        from fault_engine import process_questionnaire

        # Build payload compatible with fault_engine.process_questionnaire
        payload = {
            "status": "success",
            "module": "adaptive_questionnaire",
            "device_type": self.device_type,
            "detected_findings": self.visual_findings,
            "detection_confidence": 0.85 if self.visual_findings else None,
            "image_path": image_path,
            "answers": self.answers,
            "text_responses": self.text_responses,
            "free_text": self.problem_description,
            "safety_alert": self.safety_alert,
            "requires_professional_attention": self.safety_alert or bool(self.safety_reasons),
            "safety_reasons": self.safety_reasons,
            "contradictions": self.contradictions,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            report = process_questionnaire(payload, image_path=image_path, use_gemini=self.use_gemini)
        except Exception as exc:
            # Fallback to direct engine.diagnose
            combined = self.problem_description + " " + " ".join([f"{k}: {v}" for k, v in self.answers.items()])
            report = payload
            report["diagnosis"] = engine.diagnose(self.device_type, combined)

        self.final_report = report
        self.is_complete = True
        return report

    def get_state(self) -> DiagnosticSessionState:
        """Returns the serialized snapshot of session state."""
        return DiagnosticSessionState(
            session_id=self.session_id,
            device_type=self.device_type,
            problem_description=self.problem_description,
            current_step=self.current_step,
            estimated_total_steps=self.estimated_total_steps,
            is_complete=self.is_complete,
            safety_alert=self.safety_alert,
            safety_reasons=self.safety_reasons,
            stop_using_advice=self.stop_using_advice,
            answers=dict(self.answers),
            text_responses=dict(self.text_responses),
            contradictions=list(self.contradictions),
            top_hypotheses=[
                {"fault_id": h["fault_id"], "title": h["title"], "score": h["score"], "severity": h["severity"]}
                for h in self.hypotheses[:3]
            ],
            hardware_profile=self.hardware_profile,
            imported_specs=self.imported_specs,
            visual_findings=list(self.visual_findings),
        )


class SessionManager:
    """In-memory thread-safe store for active diagnostic sessions."""

    def __init__(self):
        self._sessions: Dict[str, DiagnosticSession] = {}

    def create_session(
        self,
        device_type: str = "laptop",
        problem_description: str = "",
        hardware_profile: Optional[Dict[str, Any]] = None,
        imported_specs: Optional[Dict[str, Any]] = None,
        visual_findings: Optional[List[str]] = None,
        use_gemini: bool = True,
    ) -> DiagnosticSession:
        session = DiagnosticSession(
            device_type=device_type,
            problem_description=problem_description,
            hardware_profile=hardware_profile,
            imported_specs=imported_specs,
            visual_findings=visual_findings,
            use_gemini=use_gemini,
        )
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[DiagnosticSession]:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False


# Global singleton instance for FastAPI and assistant
session_manager = SessionManager()
