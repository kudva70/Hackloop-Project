"""Questionnaire for PC/laptop hardware diagnosis.
Upgraded to a Hybrid Adaptive AI Diagnostic Questionnaire.

Maintains existing questionnaire entry points while providing:
- Dynamic question selection based on hardware inventory and hypotheses.
- One-question-at-a-time interactive CLI.
- Immediate safety interruption on hazards.
- Seamless compatibility with DiagnosticSession and QuestionPlanner.
"""

from typing import Dict, List, Optional, Any
import re
from datetime import datetime, timezone

from fault_engine import (
    DEVICE_REGISTRY,
    is_device_supported,
    get_canonical_device,
    evaluate_safety,
    GENERAL_UNSUPPORTED_SAFETY_ADVICE,
    STOP_USING_ADVICE,
)
from question_planner import DiagnosticQuestion, QuestionPlanner, SafetyValidator
from diagnostic_session import DiagnosticSession, DiagnosticSessionState, session_manager

# Pre-fill questionnaire keys from a free-text description and skip asking those keys.
DESCRIPTION_ANSWER_MAP = {
    "black screen": {"display": "Black screen"},
    "blank screen": {"display": "Black screen"},
    "won't charge": {"charging": "Does not charge / no power"},
    "wont charge": {"charging": "Does not charge / no power"},
    "will not charge": {"charging": "Does not charge / no power"},
    "not charging": {"charging": "Does not charge / no power"},
}

QUESTIONNAIRE_KEYS = [
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
]

LAPTOP_MENTION_TERMS = ("laptop", "notebook", "macbook", "computer", "pc", "desktop", "tower", "cabinet", "rig")
UNSUPPORTED_DEVICE_TERMS = (
    "washing machine",
    "microwave",
    "switchboard",
    "television",
    "refrigerator",
    "fridge",
    "phone",
    "bulb",
    "toaster",
    "mouse",
    "fan",
    "tv",
)


def answers_from_description(free_text):
    """Map description phrases onto questionnaire answer keys."""
    answers = {}
    text = " ".join(str(free_text or "").lower().split())
    if not text:
        return answers
    for phrase, mapping in DESCRIPTION_ANSWER_MAP.items():
        if phrase in text:
            answers.update(mapping)
    if ("fan spins" in text or "fan spinning" in text) and (
        "black screen" in text or "blank screen" in text or "no display" in text
    ):
        answers["power_state"] = "Lights/fan turn on but no display"
    return answers


def unsupported_device_from_text(free_text):
    """If text names a household device and does not clearly say laptop/PC, return that name."""
    text = " ".join(str(free_text or "").lower().split())
    if not text:
        return None
    if any(re.search(r"\b" + re.escape(term) + r"\b", text) for term in LAPTOP_MENTION_TERMS):
        return None
    for term in UNSUPPORTED_DEVICE_TERMS:
        if re.search(r"\b" + re.escape(term) + r"\b", text):
            return term
    return None


def _unsupported_payload(raw_device):
    print(f"\n[NOTICE] Device '{raw_device}' is not supported yet.")
    print("Repair Before Replace currently specializes in Laptops & PCs only.\n")
    print("GENERAL SAFETY ADVICE:")
    for tip in GENERAL_UNSUPPORTED_SAFETY_ADVICE:
        print(f"  {tip}")
    return {
        "status": "unsupported_device",
        "module": "questionnaire",
        "device_type": str(raw_device),
        "supported": False,
        "message": (
            f"Device category '{raw_device}' is not supported yet. "
            "'Repair Before Replace' currently provides in-depth diagnosis for Laptops and PCs only."
        ),
        "safety_advice": GENERAL_UNSUPPORTED_SAFETY_ADVICE,
        "safety_alert": False,
        "requires_professional_attention": False,
        "diagnosis": None,
        "repair_recommendation": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _default_skipped_answer(options):
    for fallback in ("Not sure", "Not sure / not applicable", "No", "No known event"):
        if fallback in options:
            return fallback
    return options[-1] if options else None


def normalize_device(device_type):
    """Validate that the diagnostic target is supported in the DeviceRegistry."""
    if not isinstance(device_type, str) or not device_type.strip():
        raise ValueError("No computer device/component was identified. Provide a laptop/PC image or select --device laptop.")
    canonical = get_canonical_device(device_type)
    if canonical:
        return canonical
    return device_type.strip().lower()


def ask_question(question, options):
    print(f"\n{question}")
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option}")
    while True:
        choice = input("Enter your choice: ").strip()
        try:
            index = int(choice)
            if 1 <= index <= len(options):
                return options[index - 1]
        except ValueError:
            pass
        print(f"Please enter a number between 1 and {len(options)}.")


def run_questionnaire(
    detection_result,
    image_path=None,
    run_diagnosis=True,
    scripted_answers=None,
    free_text=None,
    skip_prompts=False,
    use_gemini=True,
):
    """Ask computer-hardware questions, then optionally run the diagnosis engine."""
    if not isinstance(detection_result, dict):
        raise ValueError("Detection result must be a dictionary.")

    named_unsupported = unsupported_device_from_text(free_text)
    if named_unsupported:
        return _unsupported_payload(named_unsupported)

    raw_device = detection_result.get("device_type")
    if not is_device_supported(raw_device):
        return _unsupported_payload(raw_device)
    device = normalize_device(raw_device)
    findings = detection_result.get("findings", detection_result.get("detected_findings", []))
    confidence = detection_result.get("confidence", detection_result.get("detection_confidence"))
    image_path = image_path or detection_result.get("image_path")
    if isinstance(findings, str):
        findings = [findings]
    if not isinstance(findings, list) or not all(isinstance(item, str) for item in findings):
        raise ValueError("Detected component findings must be a list of strings.")
    if confidence is not None:
        confidence = float(confidence)
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")

    print("\n" + "=" * 52)
    print("       REPAIR BEFORE REPLACE — PC HARDWARE")
    print("=" * 52)
    target_label = "Desktop PC / Tower" if device == "desktop" else "Laptop / Notebook"
    print(f"Diagnostic target: {target_label}")
    print("Detected component(s): " + (", ".join(findings) if findings else "No component label returned"))
    if confidence is not None:
        print(f"Detection confidence: {confidence:.1%}")

    # Initialize DiagnosticSession
    session = DiagnosticSession(
        device_type=device,
        problem_description=free_text or "",
        hardware_profile=detection_result.get("hardware_profile"),
        visual_findings=findings,
        use_gemini=use_gemini,
    )

    # If already in safety hazard state from free text
    if session.safety_alert:
        print("\n" + "!" * 60)
        print("CRITICAL SAFETY ALERT — HARD HAZARD DETECTED")
        print("!" * 60)
        for advice_line in session.stop_using_advice:
            print(f"  {advice_line}")
        print("!" * 60)
        return session.finalize_diagnosis(image_path=image_path)

    # Handle scripted answers or automated test inputs
    scripted_answers = scripted_answers or {}
    pre_filled = answers_from_description(free_text)

    # 1. Interactive Adaptive CLI Mode (when prompts are enabled and no scripted answers)
    if not skip_prompts and not scripted_answers:
        print("\nAnswer based on the affected computer/component. If unsure, choose Not sure.")
        while not session.is_complete:
            q = session.get_next_question()
            if not q:
                break

            print(f"\n[Adaptive Question {session.current_step}/{session.estimated_total_steps}]")
            choice = ask_question(q.text, q.options)
            session.submit_answer(q.question_id, choice)

            if session.safety_alert:
                print("\n" + "!" * 60)
                print("CRITICAL SAFETY ALERT — HARD HAZARD DETECTED")
                print("!" * 60)
                for advice_line in session.stop_using_advice:
                    print(f"  {advice_line}")
                print("!" * 60)
                break

    # 2. Automated / Scripted / Skip-prompts Mode
    else:
        # Backward-compatible questions bank mapping
        if device == "desktop":
            legacy_questions = {
                "affected_component": (
                    "Which detected component seems affected?",
                    (findings + ["Desktop Motherboard", "RAM / Memory DIMM", "Dedicated GPU / Graphics Card",
                                 "Power Supply Unit (PSU)", "CPU Cooler / Heatsink", "SSD / HDD storage",
                                 "PC Cabinet / Power Button", "Cables / Connectors", "Not sure"]),
                ),
                "power_state": (
                    "What happens when you press the power button?",
                    ["Starts normally", "No power or lights", "Lights/fan turn on but no display",
                     "Starts then shuts down", "Not sure"],
                ),
                "startup_signals": (
                    "Are there beep codes, diagnostic LEDs, or a boot error?",
                    ["No", "Beep code", "Blinking diagnostic LED", "Error message", "Not sure"],
                ),
                "memory_storage": (
                    "What memory or storage symptom do you notice?",
                    ["Works normally", "RAM not detected / memory error",
                     "No boot device / SSD or HDD not detected", "Files or system load very slowly",
                     "Frequent crashes/freezes", "Not sure / not applicable"],
                ),
                "display": (
                    "What is the monitor / GPU display condition?",
                    ["Works normally", "Black screen", "Flickering / lines",
                     "Cracked or physically damaged", "Monitor shows 'No Signal' / Cable plugged into motherboard",
                     "Not sure / not applicable"],
                ),
                "temperature": (
                    "Does the computer show overheating or cooling symptoms?",
                    ["Normal temperature", "Runs unusually hot", "Fan unusually loud",
                     "Fan not spinning", "Thermal shutdown", "Not sure"],
                ),
                "charging": (
                    "What is the power supply (PSU) or power cord behaviour?",
                    ["Works normally", "Does not charge / no power", "Charges intermittently",
                     "PSU fan not spinning", "Sparks or burning smell from PSU", "Not sure / not applicable"],
                ),
                "recent_event": (
                    "Did anything happen shortly before the problem began?",
                    ["No known event", "Drop / physical impact", "Liquid spill",
                     "Power surge / electrical event", "Recent upgrade or repair", "Unknown"],
                ),
                "hazards": (
                    "Is there smoke, a burning smell, sparks, electric shock, or a swollen battery?",
                    ["Yes", "No", "Not sure"],
                ),
                "duration": (
                    "How long has the problem been occurring?",
                    ["Started today", "A few days", "A few weeks", "Several months", "Not sure"],
                ),
            }
        else:
            legacy_questions = {
                "affected_component": (
                    "Which detected component seems affected?",
                    (findings + ["Motherboard / mainboard", "RAM / memory", "SSD / HDD storage",
                                 "Battery / charging", "Cooling fan / heatsink", "CPU / GPU",
                                 "Screen / display", "Keyboard / touchpad", "Port / Wi-Fi / other",
                                 "Not sure"]),
                ),
                "power_state": (
                    "What happens when you press the power button?",
                    ["Starts normally", "No power or lights", "Lights/fan turn on but no display",
                     "Starts then shuts down", "Not sure"],
                ),
                "startup_signals": (
                    "Are there beep codes, diagnostic LEDs, or a boot error?",
                    ["No", "Beep code", "Blinking diagnostic LED", "Error message", "Not sure"],
                ),
                "memory_storage": (
                    "What memory or storage symptom do you notice?",
                    ["Works normally", "RAM not detected / memory error",
                     "No boot device / SSD or HDD not detected", "Files or system load very slowly",
                     "Frequent crashes/freezes", "Not sure / not applicable"],
                ),
                "display": (
                    "What is the display condition?",
                    ["Works normally", "Black screen", "Flickering / lines",
                     "Cracked or physically damaged", "External monitor works but built-in screen does not",
                     "Not sure / not applicable"],
                ),
                "temperature": (
                    "Does the computer show overheating or cooling symptoms?",
                    ["Normal temperature", "Runs unusually hot", "Fan unusually loud",
                     "Fan not spinning", "Thermal shutdown", "Not sure"],
                ),
                "charging": (
                    "What is the battery or power-input behaviour?",
                    ["Works normally", "Does not charge / no power", "Charges intermittently",
                     "Battery drains quickly", "Battery looks swollen", "Not sure / not applicable"],
                ),
                "recent_event": (
                    "Did anything happen shortly before the problem began?",
                    ["No known event", "Drop / physical impact", "Liquid spill",
                     "Power surge / electrical event", "Recent upgrade or repair", "Unknown"],
                ),
                "hazards": (
                    "Is there smoke, a burning smell, sparks, electric shock, or a swollen battery?",
                    ["Yes", "No", "Not sure"],
                ),
                "duration": (
                    "How long has the problem been occurring?",
                    ["Started today", "A few days", "A few weeks", "Several months", "Not sure"],
                ),
            }

        answers = dict(pre_filled)
        for question_id, (q_text, options) in legacy_questions.items():
            options = list(dict.fromkeys(options))
            if question_id in answers and answers[question_id] in options:
                print(f"\n{q_text}\n  -> {answers[question_id]} (from description)")
                session.submit_answer(question_id, answers[question_id])
                continue
            if question_id in scripted_answers:
                answers[question_id] = scripted_answers[question_id]
                session.submit_answer(question_id, scripted_answers[question_id])
                continue
            if skip_prompts:
                def_ans = _default_skipped_answer(options)
                answers[question_id] = def_ans
                session.submit_answer(question_id, def_ans)
                continue

        # OpenCV follow-up
        if detection_result.get("cv_follow_up"):
            fu = detection_result["cv_follow_up"]
            fu_ans = scripted_answers.get("cv_discoloration_verified", "Not sure" if skip_prompts else ask_question(fu["question"], fu["options"]))
            session.submit_answer("cv_discoloration_verified", fu_ans)
            if fu_ans == "Yes":
                session.submit_answer("hazards", "Yes")

    if run_diagnosis:
        return session.finalize_diagnosis(image_path=image_path)

    return session.get_state().to_dict()
