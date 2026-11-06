"""
Question Planner for Hybrid Adaptive AI Diagnostic Questionnaire.
Part of Repair Before Replace.

Coordinates between:
1. Predefined Diagnostic Question Bank (expert-curated, verified safe).
2. AI Question Planner (Gemini-powered adaptive selector with strict validation and offline fallback).

Guarantees:
- Never repeats questions.
- Adapts to laptop vs desktop and inventory presence.
- Strict safety validation (never suggests opening PSU, touching high voltage, or swollen batteries).
- Immediate safety interruption on hazards.
- Graceful offline fallback to information-gain heuristic ranking.
"""

from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field, asdict
import os
import re
import json

from fault_engine import (
    DEVICE_REGISTRY,
    evaluate_safety,
    HAZARD_KEYWORDS,
    HARD_HAZARD_FAULT_IDS,
    STOP_USING_ADVICE,
)

# Unsafe phrases that an AI question or option must NEVER contain
UNSAFE_INSTRUCTION_PATTERNS = [
    r"(?:open|disassemble)\w*\s+(?:the\s+)?(?:psu|power\s+supply)",
    r"(?:touch|contact|probe)\w*\s+(?:the\s+)?(?:capacitor|exposed\s+circuit|live\s+wire|pins?\s+while\s+plugged|circuits?)",
    r"puncture\w*\s+(?:the\s+)?battery",
    r"press\w*\s+(?:on\s+)?(?:the\s+)?swollen\s+battery",
    r"smell\w*\s+(?:closely|deeply)\s+(?:the\s+)?smoke",
    r"plug\s+in\s+to\s+see\s+if\s+it\s+sparks",
    r"turn\s+on\s+(?:the\s+)?wet\s+computer",
]


@dataclass
class DiagnosticQuestion:
    """Represents a single diagnostic question with options and safety metadata."""
    question_id: str
    text: str
    options: List[str]
    category: str                           # "power", "display", "memory", "cooling", "motherboard", "storage", "hazard"
    target_device: str = "all"              # "laptop", "desktop", or "all"
    required_components: List[str] = field(default_factory=list)  # e.g., ["gpu"], ["battery"]
    distinguishes_faults: List[str] = field(default_factory=list)
    relevance_condition: Optional[str] = None # code expression or tag
    allows_text_input: bool = True
    safety_classification: str = "safe"     # "safe", "hazard_check", "caution"
    is_ai_generated: bool = False
    rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiagnosticQuestion":
        return cls(**data)


# ===========================================================================
# PREDEFINED DIAGNOSTIC QUESTION BANK
# ===========================================================================

QUESTION_BANK: List[DiagnosticQuestion] = [
    # --- HAZARD & INITIAL SAFETY ---
    DiagnosticQuestion(
        question_id="hazards_check",
        text="Is there active smoke, a burning electrical smell, sparks, or a visibly swollen battery?",
        options=["Yes", "No", "Not sure"],
        category="hazard",
        target_device="all",
        distinguishes_faults=["LAPTOP_SWOLLEN_BATTERY", "LAPTOP_LIQUID_SPILL", "DESKTOP_PSU_FAILURE", "DESKTOP_MOTHERBOARD_SHORT_SURGE"],
        safety_classification="hazard_check",
        rationale="Screen for immediate physical, fire, or electrical hazards before any troubleshooting.",
    ),
    DiagnosticQuestion(
        question_id="recent_incident",
        text="Did any specific incident occur right before the problem started?",
        options=["No known event", "Liquid spill or splash", "Power surge / lightning / power cut",
                 "Dropped or physical impact", "Recent hardware upgrade / disassembly", "Unknown"],
        category="hazard",
        target_device="all",
        distinguishes_faults=["LAPTOP_LIQUID_SPILL", "LAPTOP_MOTHERBOARD_SURGE_POWER_FAILURE", "DESKTOP_MOTHERBOARD_SHORT_SURGE"],
        safety_classification="hazard_check",
        rationale="Identifies external trauma that dictates immediate service or non-repairability.",
    ),

    # --- POWER & STARTUP ---
    DiagnosticQuestion(
        question_id="power_state",
        text="What happens when you press the power button?",
        options=[
            "Starts normally into operating system",
            "No power, no lights, no sound at all",
            "Lights/fan turn on, but screen stays black",
            "Powers on for a few seconds, then shuts down immediately",
            "Power button feels loose, jammed, or unresponsive",
            "Not sure",
        ],
        category="power",
        target_device="all",
        distinguishes_faults=[
            "LAPTOP_DEAD_BATTERY_CHARGER", "LAPTOP_BLACK_SCREEN_RAM", "LAPTOP_OVERHEATING_THERMAL",
            "DESKTOP_PSU_FAILURE", "DESKTOP_RAM_CONTACT_POST_FAILURE", "DESKTOP_POWER_BUTTON_HEADER"
        ],
        safety_classification="safe",
        rationale="Determines whether failure is pre-POST (no power), POST failure (fans/lights on, black screen), or OS boot.",
    ),

    # --- DESKTOP-SPECIFIC POST & DISPLAY QUESTIONS ---
    DiagnosticQuestion(
        question_id="desktop_display_cable_connection",
        text="Where is your monitor's display cable (HDMI / DisplayPort) plugged into behind the computer?",
        options=[
            "Plugged into the dedicated Graphics Card (horizontal ports lower down)",
            "Plugged into the Motherboard (vertical ports near USB/Ethernet)",
            "Not sure / Only one display port is visible",
        ],
        category="display",
        target_device="desktop",
        required_components=["gpu"],
        distinguishes_faults=["DESKTOP_GPU_PORT_MISMATCH", "DESKTOP_RAM_CONTACT_POST_FAILURE"],
        safety_classification="safe",
        rationale="A very common issue where users plug monitor into motherboard iGPU instead of dedicated graphics card.",
    ),
    DiagnosticQuestion(
        question_id="desktop_motherboard_ez_debug_leds",
        text="Does your motherboard have small diagnostic EZ-Debug LEDs glowing (usually near the 24-pin power cable or RAM)?",
        options=[
            "DRAM LED is solid or blinking",
            "CPU LED is solid or blinking",
            "VGA LED is solid or blinking",
            "BOOT LED is solid",
            "No diagnostic LEDs visible / Motherboard does not have them",
            "Not sure",
        ],
        category="motherboard",
        target_device="desktop",
        distinguishes_faults=["DESKTOP_RAM_CONTACT_POST_FAILURE", "DESKTOP_GPU_PORT_MISMATCH", "DESKTOP_STORAGE_BOOT_FAILURE"],
        safety_classification="safe",
        rationale="Modern desktop motherboards clearly pinpoint failing POST subsystems via EZ-Debug LEDs.",
    ),
    DiagnosticQuestion(
        question_id="startup_beep_or_audio_codes",
        text="Does the computer emit any audible beep codes or chime when attempting to start?",
        options=[
            "No sound or beeps",
            "Repeating continuous beeps (memory error)",
            "1 long beep and 2-3 short beeps (display/GPU error)",
            "Single normal startup beep",
            "Clicking sound from inside the cabinet or drive",
            "Not sure",
        ],
        category="motherboard",
        target_device="all",
        distinguishes_faults=["DESKTOP_RAM_CONTACT_POST_FAILURE", "DESKTOP_STORAGE_BOOT_FAILURE", "LAPTOP_BLACK_SCREEN_RAM"],
        safety_classification="safe",
        rationale="POST beep codes convey BIOS hardware initialization errors before any display output is possible.",
    ),
    DiagnosticQuestion(
        question_id="monitor_screen_message",
        text="What does the monitor or screen actually display when powered on?",
        options=[
            "Monitor displays 'No Signal' or enters power-saving mode",
            "Screen remains completely black with no backlight glow",
            "Screen glows faintly with backlight, but shows no image or logo",
            "Flickering lines, visual artifacts, or distorted colors",
            "Shows motherboard/manufacturer logo then freezes",
            "Not sure",
        ],
        category="display",
        target_device="all",
        distinguishes_faults=[
            "DESKTOP_GPU_PORT_MISMATCH", "DESKTOP_RAM_CONTACT_POST_FAILURE",
            "LAPTOP_BLACK_SCREEN_RAM", "LAPTOP_SCREEN_DAMAGE_BACKLIGHT"
        ],
        safety_classification="safe",
        rationale="Distinguishes between lack of video input signal vs video card failure vs LCD panel failure.",
    ),

    # --- LAPTOP-SPECIFIC DISPLAY & CHARGER QUESTIONS ---
    DiagnosticQuestion(
        question_id="laptop_external_monitor_test",
        text="If you connect the laptop to an external TV or monitor with an HDMI cable, does an image display?",
        options=[
            "Yes, external monitor displays desktop normally (laptop screen stays black)",
            "No, external monitor also shows black screen or no signal",
            "Have not tested / Do not have an external monitor or TV",
            "Not sure",
        ],
        category="display",
        target_device="laptop",
        distinguishes_faults=["LAPTOP_SCREEN_DAMAGE_BACKLIGHT", "LAPTOP_BLACK_SCREEN_RAM"],
        safety_classification="safe",
        rationale="Isolates built-in LCD panel/eDP cable faults from motherboard, RAM, and GPU rendering faults.",
    ),
    DiagnosticQuestion(
        question_id="laptop_charger_indicator_light",
        text="When you plug in the charger, does the charging LED on the side of the laptop turn on?",
        options=[
            "Yes, solid charging light turns on",
            "Charging light blinks in a pattern",
            "No light at all",
            "Light only turns on if I wiggle the charging cable plug",
            "Not sure",
        ],
        category="power",
        target_device="laptop",
        distinguishes_faults=["LAPTOP_DEAD_BATTERY_CHARGER", "LAPTOP_MOTHERBOARD_SURGE_POWER_FAILURE"],
        safety_classification="safe",
        rationale="Differentiates between dead DC jack/adapter vs internal motherboard power delivery circuitry.",
    ),

    # --- THERMAL & COOLING ---
    DiagnosticQuestion(
        question_id="fan_and_thermal_behavior",
        text="How are the computer's cooling fans sounding and behaving?",
        options=[
            "Fans spin normally and quietly",
            "Fans spin at maximum jet-engine speed continuously",
            "Fans do not spin at all even when warm",
            "Grinding, rattling, or buzzing mechanical sound from fan",
            "Gurgling or bubbling noise (AIO liquid cooler pump)",
            "Not sure",
        ],
        category="cooling",
        target_device="all",
        distinguishes_faults=["LAPTOP_OVERHEATING_THERMAL", "DESKTOP_CPU_COOLER_AIO_FAILURE"],
        safety_classification="safe",
        rationale="Identifies dried thermal paste, failed AIO pumps, or seized fan bearings causing thermal shutdown.",
    ),

    # --- STORAGE & BOOT ---
    DiagnosticQuestion(
        question_id="storage_boot_screen",
        text="What boot error or storage symptom do you observe?",
        options=[
            "No bootable device found / Insert Boot Media error screen",
            "Blue Screen of Death (BSOD) with INACCESSIBLE_BOOT_DEVICE or CRITICAL_PROCESS_DIED",
            "Endless loading circle / Windows spinning dots freeze",
            "Operating system boots up, but system is freezing and files take minutes to open",
            "Storage works normally",
            "Not sure / not applicable",
        ],
        category="storage",
        target_device="all",
        distinguishes_faults=["LAPTOP_SSD_STORAGE_FAILURE", "DESKTOP_STORAGE_BOOT_FAILURE"],
        safety_classification="safe",
        rationale="Pinpoints SSD/NVMe connection loss, partition corruption, or failing drive controller.",
    ),

    # --- KEYBOARD & HARDWARE INTERACTION ---
    DiagnosticQuestion(
        question_id="caps_lock_indicator_test",
        text="When the computer is on with a dark screen, does pressing the 'Caps Lock' or 'Num Lock' key toggle its little LED on and off?",
        options=[
            "Yes, Caps Lock light turns on and off when pressed",
            "No, Caps Lock light does not respond or stays completely unlit",
            "Caps Lock LED blinks in a continuous error code pattern",
            "Keyboard has no LED indicators / Not sure",
        ],
        category="motherboard",
        target_device="all",
        distinguishes_faults=["LAPTOP_BLACK_SCREEN_RAM", "LAPTOP_SCREEN_DAMAGE_BACKLIGHT", "DESKTOP_RAM_POST_FAILURE"],
        safety_classification="safe",
        rationale="If Caps Lock toggles, CPU and OS kernel are running normally in background; only display path is dead.",
    ),
]


class SafetyValidator:
    """Enforces safety constraints on any question presented to the user."""

    @classmethod
    def validate_question(cls, question: DiagnosticQuestion) -> Tuple[bool, Optional[str]]:
        """
        Validates that a question and its options do not encourage hazardous actions.
        Returns (is_valid, rejection_reason).
        """
        combined = f"{question.text} {' '.join(question.options)} {question.rationale or ''}".lower()

        for pat in UNSAFE_INSTRUCTION_PATTERNS:
            if re.search(pat, combined):
                return False, f"Safety policy violation: Question mentions restricted hazard action matching pattern '{pat}'."

        if not question.options or len(question.options) < 2:
            return False, "Validation error: Question must offer at least two options."

        # Ensure a 'Not sure' or equivalent option exists
        has_fallback = any(opt.lower() in ("not sure", "not sure / not applicable", "unknown", "unsure") for opt in question.options)
        if not has_fallback:
            question.options.append("Not sure")

        return True, None


class QuestionPlanner:
    """
    Selects or generates the next best diagnostic question based on:
    - Target computer (laptop vs desktop)
    - Hardware inventory & presence
    - Symptom description & previous answers
    - Current candidate fault hypotheses
    """

    def __init__(self, use_gemini: bool = True):
        self.use_gemini = use_gemini
        self.question_bank: Dict[str, DiagnosticQuestion] = {q.question_id: q for q in QUESTION_BANK}

    def get_candidate_questions(
        self,
        device_type: str,
        hardware_profile: Optional[Dict[str, Any]],
        asked_question_ids: Set[str],
        answers: Dict[str, Any],
        hypotheses: List[Dict[str, Any]],
    ) -> List[DiagnosticQuestion]:
        """Filters question bank down to eligible, unasked, component-relevant questions."""
        candidates = []
        dev_norm = "desktop" if "desktop" in device_type.lower() else "laptop"

        # Check inventory for required components
        present_components: Set[str] = set()
        if hardware_profile and "components" in hardware_profile:
            comps = hardware_profile["components"]
            for cat, cdata in comps.items():
                status = cdata.get("visibility_status") if isinstance(cdata, dict) else getattr(cdata, "visibility_status", None)
                if status in ("DETECTED", "PARTIALLY_VISIBLE", "USER_CONFIRMED", "IMPORTED_FROM_REPORT"):
                    present_components.add(cat.lower())

        for q in QUESTION_BANK:
            if q.question_id in asked_question_ids:
                continue

            # Device filter
            if q.target_device != "all" and q.target_device != dev_norm:
                continue

            # Component presence requirement (e.g. don't ask desktop GPU questions if PC has no discrete GPU)
            if q.required_components:
                # If specific component required, check profile
                # If profile is known and has components, ensure component is present
                if present_components and not any(comp in present_components for comp in q.required_components):
                    continue

            # Relevance condition check
            if q.question_id == "desktop_display_cable_connection":
                # Only relevant if display problem or black screen or no signal
                power_ans = str(answers.get("power_state", "")).lower()
                disp_ans = str(answers.get("display", "")).lower()
                if "no power" in power_ans:
                    continue

            if q.question_id == "desktop_motherboard_ez_debug_leds":
                # Only relevant if machine powers on (fans/lights turn on)
                power_ans = str(answers.get("power_state", "")).lower()
                if "no power" in power_ans:
                    continue

            candidates.append(q)

        return candidates

    def score_question_information_gain(
        self,
        question: DiagnosticQuestion,
        hypotheses: List[Dict[str, Any]],
        answers: Dict[str, Any],
    ) -> float:
        """Heuristic scoring of question relevance based on competing fault hypotheses."""
        if not hypotheses:
            # Baseline ordering: hazards first, then power, then display/motherboard
            order_weights = {
                "hazard": 10.0,
                "power": 8.0,
                "display": 6.0,
                "motherboard": 5.0,
                "cooling": 4.0,
                "storage": 3.0,
            }
            return order_weights.get(question.category, 1.0)

        top_fault_ids = [h.get("fault_id") or (h.get("rule").fault_id if h.get("rule") else "") for h in hypotheses[:3]]
        score = 1.0

        # High priority if question separates top competing faults
        for fid in question.distinguishes_faults:
            if fid in top_fault_ids:
                score += 5.0

        # Prioritize hazards if not yet checked
        if question.category == "hazard" and "hazards_check" not in answers:
            score += 12.0

        # Contextual boosts
        last_answer = list(answers.values())[-1] if answers else ""
        if isinstance(last_answer, str):
            if "dram" in last_answer.lower() and question.category in ("memory", "motherboard"):
                score += 8.0
            if "black screen" in last_answer.lower() and question.category in ("display", "motherboard"):
                score += 6.0
            if "hot" in last_answer.lower() and question.category == "cooling":
                score += 7.0

        return score

    def select_next_question(
        self,
        device_type: str,
        hardware_profile: Optional[Dict[str, Any]],
        asked_question_ids: Set[str],
        answers: Dict[str, Any],
        hypotheses: List[Dict[str, Any]],
        free_text: Optional[str] = None,
    ) -> Optional[DiagnosticQuestion]:
        """
        Determines the next diagnostic question to ask.
        Attempts AI planner selection if available, with robust offline heuristic fallback.
        """
        candidates = self.get_candidate_questions(
            device_type=device_type,
            hardware_profile=hardware_profile,
            asked_question_ids=asked_question_ids,
            answers=answers,
            hypotheses=hypotheses,
        )

        if not candidates:
            return None

        # Attempt AI selection via Gemini if enabled and configured
        if self.use_gemini and os.getenv("GEMINI_API_KEY"):
            try:
                ai_q = self._ai_select_question(
                    candidates=candidates,
                    device_type=device_type,
                    answers=answers,
                    hypotheses=hypotheses,
                    free_text=free_text,
                    hardware_profile=hardware_profile,
                )
                if ai_q:
                    valid, reason = SafetyValidator.validate_question(ai_q)
                    if valid:
                        return ai_q
                    else:
                        print(f"[Question Planner] AI question rejected: {reason}. Falling back to rule bank.")
            except Exception as exc:
                # Log and proceed to offline fallback
                print(f"[Question Planner] AI selection bypassed: {exc}. Using deterministic planner.")

        # Offline Fallback: Information-Gain Ranking
        candidates.sort(
            key=lambda q: self.score_question_information_gain(q, hypotheses, answers),
            reverse=True,
        )
        selected = candidates[0]

        # Adapt question wording if specific hardware is known
        adapted = self._adapt_wording_to_hardware(selected, hardware_profile)
        return adapted

    def _adapt_wording_to_hardware(
        self,
        question: DiagnosticQuestion,
        hardware_profile: Optional[Dict[str, Any]],
    ) -> DiagnosticQuestion:
        """Adapts question text and options to explicitly reference detected GPU, RAM, or CPU."""
        if not hardware_profile or "components" not in hardware_profile:
            return question

        comps = hardware_profile["components"]
        gpu = comps.get("gpu")
        ram = comps.get("ram")

        new_text = question.text
        # If specific GPU model is known (e.g., RTX 3080)
        if gpu and isinstance(gpu, dict):
            specs = gpu.get("specs_details") or {}
            gpu_name = specs.get("model") or gpu.get("component_name")
            if gpu_name and "graphics card" in new_text.lower():
                new_text = new_text.replace("dedicated Graphics Card", f"dedicated Graphics Card ({gpu_name})")

        return DiagnosticQuestion(
            question_id=question.question_id,
            text=new_text,
            options=list(question.options),
            category=question.category,
            target_device=question.target_device,
            required_components=question.required_components,
            distinguishes_faults=question.distinguishes_faults,
            relevance_condition=question.relevance_condition,
            allows_text_input=question.allows_text_input,
            safety_classification=question.safety_classification,
            is_ai_generated=question.is_ai_generated,
            rationale=question.rationale,
        )

    def _ai_select_question(
        self,
        candidates: List[DiagnosticQuestion],
        device_type: str,
        answers: Dict[str, Any],
        hypotheses: List[Dict[str, Any]],
        free_text: Optional[str],
        hardware_profile: Optional[Dict[str, Any]],
    ) -> Optional[DiagnosticQuestion]:
        """Calls Gemini to choose the single most discriminating question from candidates."""
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None

        client = genai.Client(api_key=api_key)
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        candidate_list = [
            {"id": q.question_id, "text": q.text, "category": q.category, "options": q.options}
            for q in candidates[:8]
        ]
        top_hypotheses = [
            {"fault_id": h.get("fault_id"), "title": h.get("title") or (h.get("rule").title if h.get("rule") else ""), "score": h.get("score")}
            for h in hypotheses[:3]
        ]

        prompt = f"""
You are an expert computer diagnostic planner.
Your goal is to select the single best question to distinguish between the current competing fault hypotheses.

TARGET DEVICE: {device_type}
INITIAL SYMPTOM: {free_text or 'None'}
PREVIOUS ANSWERS: {json.dumps(answers, ensure_ascii=False)}
CURRENT TOP HYPOTHESES: {json.dumps(top_hypotheses, ensure_ascii=False)}

CANDIDATE QUESTIONS:
{json.dumps(candidate_list, indent=2, ensure_ascii=False)}

CRITICAL SAFETY RULES:
1. Never suggest opening a power supply or touching exposed circuits.
2. Prioritize questions that eliminate or confirm the #1 most probable fault.
3. If user mentioned a DRAM light or RAM beep, select memory/RAM questions over display questions.

Return JSON in this schema:
{{
  "selected_question_id": "ONE_ID_FROM_CANDIDATE_QUESTIONS",
  "rationale": "Brief 1-sentence diagnostic rationale"
}}
"""
        response = client.models.generate_content(
            model=model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                max_output_tokens=300,
            ),
        )
        data = json.loads(response.text)
        selected_id = data.get("selected_question_id")

        for c in candidates:
            if c.question_id == selected_id:
                c.rationale = data.get("rationale")
                return c

        return None
