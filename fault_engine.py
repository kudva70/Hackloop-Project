"""
AI Repair Assistant - Rule-Based Fault Diagnosis Engine (Laptops & PCs)
Deterministic, expert-curated decision rules for computer hardware diagnosis.

MVP scope: full depth for laptops/PCs only, built around an extensible DEVICE_REGISTRY.
Unsupported devices return a structured "unsupported_device" result with general safety advice.
"""

from typing import Dict, List, Any, Optional
import os
import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class FaultRule:
    fault_id: str
    title: str
    severity: str          # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    complexity: str        # "Low (Beginner)", "Medium (Intermediate)", "High (Advanced)", "Dangerous (Do Not DIY)"
    recommendation: str    # "REPAIR (DIY)", "SERVICE (Professional)", "REPLACE"
    rationale: str
    required_tools: List[str]
    estimated_cost: str
    estimated_time: str
    safety_warnings: List[str]
    safe_checks: List[str]       # Non-invasive, no tools, no opening the chassis
    advanced_steps: List[str]    # Opening the chassis, repasting, reseating; EMPTY for CRITICAL/SERVICE-only rules
    positive_keywords: List[str]
    negative_keywords: List[str]
    common_symptoms: List[str]
    co_occurrence_patterns: List[List[str]]

    @property
    def troubleshooting_steps(self) -> List[str]:
        """Backward-compatible combination of safe_checks + advanced_steps."""
        return self.safe_checks + self.advanced_steps


# ===========================================================================
# EXPERT KNOWLEDGE BASE: LAPTOPS & PCS (10 Comprehensive Rules)
# ===========================================================================

LAPTOP_RULES: List[FaultRule] = [
    # 1. Dead Battery / Charger / DC Jack
    FaultRule(
        fault_id="LAPTOP_DEAD_BATTERY_CHARGER",
        title="Faulty Power Adapter / Dead Battery / DC Jack Failure",
        severity="MEDIUM",
        complexity="Low (Beginner)",
        recommendation="REPAIR (DIY / Replace Adapter)",
        rationale="Replacing an external AC adapter or testing wall power requires no internal laptop teardown.",
        required_tools=["Compatible OEM Charger", "Voltmeter / Multimeter", "Alternative confirmed wall outlet"],
        estimated_cost="₹800 - ₹2,500 ($10 - $30)",
        estimated_time="10 - 15 minutes",
        safety_warnings=[
            "Never use generic unbranded cheap chargers; they can cause voltage spikes and thermal runaway.",
            "If the charger brick or cable is excessively hot or smells of burnt plastic, unplug immediately."
        ],
        safe_checks=[
            "1. Perform an EC (Embedded Controller) reset: Unplug charger, hold power button down for 30 full seconds, then release.",
            "2. Inspect the charger cable for kinks, fraying, or loose pins in the charging connector.",
            "3. Try plugging into a different confirmed working wall socket without an extension strip.",
            "4. Check if the LED charging indicator illuminates when the charger is connected."
        ],
        advanced_steps=[
            "1. Disconnect AC charger and internal battery.",
            "2. Inspect the internal DC-in power jack harness for cracked solder joints or broken center pin.",
            "3. Measure DC adapter output voltage with a multimeter to ensure rated output voltage is present."
        ],
        positive_keywords=[
            "won't turn on", "wont turn on", "not turning on", "dead", "no power",
            "not charging", "charger", "plugged in not charging", "battery dies",
            "dc jack", "charging port", "no power or lights", "does not charge / no power"
        ],
        negative_keywords=["black screen", "blue screen", "spill", "water", "lights on", "fan spinning", "turn on but no display"],
        common_symptoms=[
            "Laptop completely dead, no lights or fan activity when pressing power button",
            "Battery does not charge when plugged in, or drops to 0% immediately when unplugged",
            "Charging only works when wiggling the charger cable at a specific angle"
        ],
        co_occurrence_patterns=[
            ["not charging", "battery"],
            ["wont turn on", "charger"],
            ["dead", "plugged in"]
        ]
    ),

    # 2. Overheating / Clogged Fan / Dried Thermal Paste
    FaultRule(
        fault_id="LAPTOP_OVERHEATING_DUST_THERMAL",
        title="Thermal Throttling / Clogged Heatsink / Dried Thermal Paste",
        severity="MEDIUM",
        complexity="Medium (Intermediate)",
        recommendation="REPAIR (DIY) / SERVICE (Professional)",
        rationale="Cleaning heatsink fins with compressed air and repasting thermal compound restores normal operating temperatures and stops thermal shutdowns.",
        required_tools=["Precision Screwdriver Set (PH00/T5)", "Compressed Air Can", "High-performance Thermal Paste (Arctic MX-4/Noctua)", "Isopropyl Alcohol (99%)"],
        estimated_cost="₹300 - ₹600 ($4 - $8)",
        estimated_time="30 - 45 minutes",
        safety_warnings=[
            "Always unplug the battery cable from the motherboard before touching heatsinks or fans.",
            "Use an anti-static wrist strap or touch a grounded metal object to prevent ESD damage."
        ],
        safe_checks=[
            "1. Elevate the rear of the laptop by 1 inch on a hard, flat surface to ensure air intakes are unobstructed.",
            "2. Feel the exhaust vent during use: if the fan is screaming but almost no warm air exits, the heatsink grill is clogged with lint.",
            "3. Run hardware temperature monitoring software (e.g. HWMonitor, CoreTemp) to verify if CPU/GPU reaches 95°C+."
        ],
        advanced_steps=[
            "1. Power off, remove bottom cover with precision screwdrivers, and disconnect the battery connector.",
            "2. Hold the fan blades gently (do not let them over-spin) and blow dust out with compressed air.",
            "3. Unscrew the copper heatsink assembly in reverse diagonal sequence (4-3-2-1).",
            "4. Clean old hardened paste from CPU/GPU dies using 99% isopropyl alcohol and a lint-free wipe.",
            "5. Apply a pea-sized dot of new high-grade thermal paste and reassemble heatsink firmly."
        ],
        positive_keywords=[
            "overheat", "overheating", "hot", "very hot", "fan loud", "fan noise",
            "shuts down randomly", "thermal throttling", "lagging", "high temp",
            "thermal paste", "runs unusually hot", "thermal shutdown"
        ],
        negative_keywords=["liquid", "spill", "water", "cracked", "fan stopped", "fan not spinning"],
        common_symptoms=[
            "Laptop gets extremely hot underneath, especially near the exhaust vent",
            "Cooling fan runs at maximum 100% speed continuously",
            "Laptop suddenly shuts down or restarts under heavy load (thermal protection)"
        ],
        co_occurrence_patterns=[
            ["hot", "fan"],
            ["overheating", "shuts down"],
            ["loud", "fan"],
            ["thermal", "throttling"]
        ]
    ),

    # 3. Display / RAM Contact Issue (Black screen with power)
    FaultRule(
        fault_id="LAPTOP_DISPLAY_RAM_ISSUE",
        title="Black Screen / RAM Contact Oxidation / Loose Display Cable",
        severity="MEDIUM",
        complexity="Medium (Intermediate)",
        recommendation="REPAIR (DIY)",
        rationale="Reseating RAM sticks with an eraser or reconnecting the eDP display cable fixes 80% of 'power on with black screen' problems.",
        required_tools=["Precision Phillips/Torx screwdriver", "Clean Pencil Eraser", "Microfiber cloth"],
        estimated_cost="₹0 ($0)",
        estimated_time="20 - 30 minutes",
        safety_warnings=[
            "Disconnect the battery before touching RAM slots or the display eDP cable. Shorting display pins can burn the motherboard backlight fuse!"
        ],
        safe_checks=[
            "1. Connect the laptop to an external TV or monitor via HDMI.",
            "2. If external monitor displays the desktop normally, the motherboard and GPU are fine; the issue is internal panel, cable, or backlight.",
            "3. Check Caps Lock / Num Lock: if the light toggles on and off when pressed, the OS is booting properly behind the dark screen.",
            "4. Shine a bright flashlight at an angle at the dark screen: if faint desktop icons are visible, the panel backlight or fuse has failed."
        ],
        advanced_steps=[
            "1. Power off, open bottom chassis, and disconnect the battery cable from the motherboard.",
            "2. Unclip the RAM modules and remove them.",
            "3. Rub the golden contact pins on both sides of each RAM stick with a clean pencil eraser to remove microscopic oxidation.",
            "4. Reinsert RAM firmly into the slots at a 30-degree angle and click down. Reconnect battery and test boot."
        ],
        positive_keywords=[
            "black screen", "blank screen", "no display", "screen dark",
            "keyboard lights up no display", "caps lock blinking", "ram", "display cable",
            "external monitor", "lights/fan turn on but no display", "lights turn on but no display",
            "beep code", "blinking diagnostic led"
        ],
        negative_keywords=["spill", "water", "cracked", "screen crack", "broken glass"],
        common_symptoms=[
            "Power LED and fan turn on, but the screen stays pitch black with no logo",
            "Caps Lock light blinks in a rhythmic diagnostic code",
            "External monitor works but built-in screen remains completely black"
        ],
        co_occurrence_patterns=[
            ["lights", "no display"],
            ["fan", "no display"],
            ["black screen", "power"],
            ["caps lock", "blink"],
            ["blank", "lights"]
        ]
    ),

    # 4. Liquid Ingress / Corrosion Hazard (CRITICAL, SERVICE ONLY)
    FaultRule(
        fault_id="LAPTOP_LIQUID_SPILL",
        title="Liquid Ingress / Motherboard Corrosion Hazard",
        severity="CRITICAL",
        complexity="Dangerous (Do Not DIY)",
        recommendation="SERVICE (Professional)",
        rationale="Liquid causes galvanic corrosion and micro-shorts under surface-mount BGAs. Requires immediate professional ultrasonic cleaning and board drying.",
        required_tools=["Professional Ultrasonic Cleaner", "99% Isopropyl Alcohol", "Stereo Inspection Microscope"],
        estimated_cost="₹1,500 - ₹5,000 ($20 - $60)",
        estimated_time="1 - 3 business days",
        safety_warnings=[
            "CRITICAL FIRE & SHORT HAZARD: DO NOT TURN ON THE LAPTOP. DO NOT PLUG IN THE CHARGER.",
            "Leaving power on with liquid present causes instant electrolysis that permanently eats through copper motherboard traces."
        ],
        safe_checks=[
            "1. IMMEDIATELY disconnect the AC charger from the wall and the laptop.",
            "2. Force power OFF by holding the power button for 10 full seconds if device was running.",
            "3. Place the laptop in an inverted 'tent' (V-shape) on a towel so liquid drains away from internal circuits.",
            "4. DO NOT use a hairdryer (it blows liquid deeper under chips and warps plastic keyboard membranes).",
            "5. DO NOT use uncooked rice (starch dust enters ports and accelerates corrosion)."
        ],
        advanced_steps=[],  # EMPTY: CRITICAL / SERVICE-only rule
        positive_keywords=[
            "spill", "liquid", "water", "coffee", "tea", "juice", "wet",
            "poured", "dropped in water", "liquid spill", "liquid exposure"
        ],
        negative_keywords=[],
        common_symptoms=[
            "Liquid was spilled onto keyboard, trackpad, or vents",
            "Laptop turned off immediately after liquid contact",
            "Sticky keys or crackling sound inside the laptop"
        ],
        co_occurrence_patterns=[
            ["water", "spill"],
            ["coffee", "keyboard"],
            ["liquid", "wet"],
            ["spill", "laptop"]
        ]
    ),

    # 5. Physical Screen Damage / Cracked LCD
    FaultRule(
        fault_id="LAPTOP_PHYSICAL_SCREEN_DAMAGE",
        title="Cracked LCD / OLED Display Panel (Physical Impact)",
        severity="MEDIUM",
        complexity="Medium (Intermediate)",
        recommendation="REPLACE (Display Panel Assembly)",
        rationale="Cracked glass or severed liquid crystal matrix cannot be repaired with software or soldering; it requires replacing the internal display panel.",
        required_tools=["Plastic Spudger / Prying Tool", "Precision Screwdriver PH00", "Replacement OEM Screen Panel"],
        estimated_cost="₹2,500 - ₹6,000 ($30 - $75)",
        estimated_time="30 - 45 minutes",
        safety_warnings=[
            "Disconnect the battery and AC charger before touching the eDP display cable to avoid blowing the motherboard backlight fuse."
        ],
        safe_checks=[
            "1. Connect laptop to an external monitor via HDMI to verify that GPU and motherboard boot successfully.",
            "2. If external monitor displays the OS, the core system is healthy and only the screen panel requires replacement."
        ],
        advanced_steps=[
            "1. Disconnect AC charger and internal battery.",
            "2. Remove the screen bezel frame using a plastic pry tool.",
            "3. Unscrew LCD panel mounting brackets and note the panel part number sticker on the back (e.g. NV156FHM-N48).",
            "4. Unclip the 30-pin or 40-pin eDP video connector gently.",
            "5. Install the replacement panel, reconnect eDP cable firmly, test before snapping bezel back."
        ],
        positive_keywords=[
            "screen crack", "crack", "cracked", "broken screen", "shattered",
            "ink bleed", "black spots", "lines on screen", "cracked or physically damaged"
        ],
        negative_keywords=[],
        common_symptoms=[
            "Visible crack, hairline fracture, or spiderweb pattern on the display glass",
            "Ink-bleed blotches or black circular voids on the screen",
            "Vertical or horizontal colored lines with fractured glass"
        ],
        co_occurrence_patterns=[
            ["screen", "crack"],
            ["broken", "display"],
            ["cracked", "glass"]
        ]
    ),

    # 6. SSD / HDD Storage Drive Failure
    FaultRule(
        fault_id="LAPTOP_STORAGE_DRIVE_FAILURE",
        title="SSD / HDD Storage Drive Failure or Missing Boot Device",
        severity="MEDIUM",
        complexity="Medium (Intermediate)",
        recommendation="REPAIR (DIY)",
        rationale="Solid-state drives degrade or corrupt over time. Swapping an M.2 NVMe SSD takes 20 minutes with a single screwdriver.",
        required_tools=["Precision Screwdriver PH00", "USB Flash Drive (OS Installer)", "Replacement M.2 NVMe SSD / 2.5-inch SATA SSD"],
        estimated_cost="₹2,000 - ₹5,500 ($25 - $65)",
        estimated_time="20 - 35 minutes",
        safety_warnings=[
            "Disconnect the battery before unseating or installing storage drives.",
            "Back up any accessible personal data before running disk formatting tools."
        ],
        safe_checks=[
            "1. Reboot laptop and enter BIOS setup (F2/F12/Del) to check if the internal SSD/HDD is detected in the storage menu.",
            "2. Check if a USB drive or SD card is inserted that may be interrupting boot order.",
            "3. Run manufacturer hardware diagnostics from the boot menu (e.g. Dell ePSA, HP Diagnostics)."
        ],
        advanced_steps=[
            "1. Power off, disconnect charger, open bottom panel, and unplug battery connector.",
            "2. Locate the M.2 NVMe drive slot.",
            "3. Unscrew the mounting screw, gently pull out the SSD at a 30-degree angle, and reinsert it firmly to ensure solid contact.",
            "4. If drive is dead, install a new M.2 NVMe SSD, fasten the screw, reassemble, and reinstall OS using USB installer media."
        ],
        positive_keywords=[
            "no boot device", "no bootable device", "ssd not detected", "hdd not detected",
            "hard drive failure", "drive not found", "boot device not found", "smart error",
            "reboot and select proper boot device", "storage", "no boot device / ssd or hdd not detected"
        ],
        negative_keywords=["black screen", "water", "spill"],
        common_symptoms=[
            "Error screen on boot: 'No bootable device found' or 'Hard Drive - Not Installed'",
            "SSD or HDD missing from BIOS hardware list",
            "System freezes on Windows logo with S.M.A.R.T. disk failure alert"
        ],
        co_occurrence_patterns=[
            ["no boot", "device"],
            ["ssd", "not detected"],
            ["drive", "not found"],
            ["storage", "error"]
        ]
    ),

    # 7. Swollen / Bulging Li-ion Battery (CRITICAL, SERVICE ONLY)
    FaultRule(
        fault_id="LAPTOP_SWOLLEN_BATTERY",
        title="Swollen / Bulging Li-ion Battery Pouch (Severe Fire Hazard)",
        severity="CRITICAL",
        complexity="Dangerous (Do Not DIY)",
        recommendation="SERVICE (Professional)",
        rationale="Lithium-ion polymer pouch cells produce hazardous gas when internal layers fail. Puncturing or charging a swollen battery presents severe fire and toxic smoke risks.",
        required_tools=["Fireproof Battery Container", "Authorized Service Center Disposal Unit"],
        estimated_cost="₹2,500 - ₹5,000 ($30 - $60)",
        estimated_time="1 - 2 business days",
        safety_warnings=[
            "CRITICAL FIRE HAZARD: DO NOT CHARGE OR PRESS ON A SWOLLEN BATTERY.",
            "Never puncture, bend, or force open a bulging battery. Escaping gas can self-ignite upon contact with oxygen."
        ],
        safe_checks=[
            "1. Stop using and disconnect charger from the wall immediately.",
            "2. Do not press down on the bulging trackpad, keyboard, or warped bottom chassis.",
            "3. Place laptop in a well-ventilated, non-flammable area (e.g. tile/concrete surface) away from combustible fabrics.",
            "4. Transport to an authorized repair technician or battery recycling center for safe disposal."
        ],
        advanced_steps=[],  # EMPTY: CRITICAL / SERVICE-only rule
        positive_keywords=[
            "swollen battery", "battery swollen", "bulging", "bulging battery",
            "trackpad lifting", "chassis expanding", "battery pushing up",
            "battery looks swollen", "swollen"
        ],
        negative_keywords=[],
        common_symptoms=[
            "Trackpad is physically pushed up above the palmrest surface",
            "Bottom chassis is warped or screws are pulling out from battery bulge",
            "Battery reported as swollen or bulging in questionnaire"
        ],
        co_occurrence_patterns=[
            ["battery", "swollen"],
            ["trackpad", "lifting"],
            ["battery", "bulging"],
            ["chassis", "expanding"]
        ]
    ),

    # 8. Keyboard / Touchpad Hardware Failure
    FaultRule(
        fault_id="LAPTOP_KEYBOARD_TOUCHPAD_FAILURE",
        title="Keyboard Membrane / Touchpad Controller Hardware Failure",
        severity="LOW",
        complexity="Medium (Intermediate)",
        recommendation="REPAIR (DIY) / SERVICE (Professional)",
        rationale="Individual non-working keys or frozen trackpad can be resolved by reseating the ribbon cable or replacing the keyboard deck.",
        required_tools=["Precision Screwdriver Set", "Plastic Spudger", "Replacement Keyboard/Trackpad Assembly"],
        estimated_cost="₹1,200 - ₹3,500 ($15 - $45)",
        estimated_time="30 - 60 minutes",
        safety_warnings=[
            "Disconnect power and internal battery before handling keyboard ribbon ZIF connectors."
        ],
        safe_checks=[
            "1. Plug in an external USB keyboard/mouse to confirm the OS and USB controllers respond normally.",
            "2. Open Device Manager, uninstall the keyboard/touchpad driver, and restart laptop to reload drivers.",
            "3. Inspect keys for trapped dust or debris and blow clean with compressed air."
        ],
        advanced_steps=[
            "1. Disconnect battery and charger.",
            "2. Inspect the keyboard and touchpad ribbon cables connecting to the motherboard.",
            "3. Ensure the locking latch on the ZIF connector is fully seated and not loose.",
            "4. If key traces are broken, unfasten keyboard bracket and install a replacement OEM keyboard."
        ],
        positive_keywords=[
            "keyboard not working", "keys not working", "keys typing wrong",
            "touchpad unresponsive", "trackpad not clicking", "cursor not moving",
            "stuck keys", "keyboard typing random", "touchpad not working"
        ],
        negative_keywords=["black screen", "spill", "water"],
        common_symptoms=[
            "Specific rows of keys or individual keys do not register keystrokes",
            "Touchpad does not register tap, gesture, or physical click",
            "Keyboard types repeated ghost characters without input"
        ],
        co_occurrence_patterns=[
            ["keyboard", "not working"],
            ["keys", "not working"],
            ["touchpad", "unresponsive"],
            ["trackpad", "clicking"]
        ]
    ),

    # 9. Motherboard Power Delivery / Surge Failure (CRITICAL, SERVICE ONLY)
    FaultRule(
        fault_id="LAPTOP_MOTHERBOARD_SURGE_POWER_FAILURE",
        title="Motherboard Power Circuit / VRM Damage from Electrical Surge",
        severity="CRITICAL",
        complexity="Dangerous (Do Not DIY)",
        recommendation="SERVICE (Professional)",
        rationale="Power surges destroy high-side MOSFETs, power management ICs, and decoupling capacitors. Requires diagnostic micro-soldering.",
        required_tools=["Bench Power Supply", "Hot Air Rework Station", "Multimeter with Micro-probes"],
        estimated_cost="₹3,500 - ₹9,000 ($45 - $110)",
        estimated_time="2 - 5 business days",
        safety_warnings=[
            "ELECTRICAL & FIRE HAZARD: Do not repeatedly press the power button or leave plugged in after a surge.",
            "Short-circuited input capacitors can heat up rapidly and char the motherboard PCB."
        ],
        safe_checks=[
            "1. Immediately disconnect charger and do not plug back into the wall.",
            "2. Smell around the charging port and exhaust vents for acrid burnt silicone/resin odor.",
            "3. Inspect the power adapter brick: if it is blackened or made a loud pop during the event, it is destroyed."
        ],
        advanced_steps=[],  # EMPTY: CRITICAL / SERVICE-only rule
        positive_keywords=[
            "power surge", "electrical event", "surge", "short circuit",
            "spark when plugged", "burnt odor", "dead after surge", "tripped breaker",
            "power surge / electrical event", "no power after lightning"
        ],
        negative_keywords=["cracked", "display cable"],
        common_symptoms=[
            "Laptop completely dead after an electrical storm or wall socket spark",
            "Burnt electrical smell coming from charging port or motherboard",
            "AC adapter LED indicator turns off immediately when connected to laptop (dead short)"
        ],
        co_occurrence_patterns=[
            ["power", "surge"],
            ["electrical", "event"],
            ["spark", "plugged"],
            ["burnt", "power"]
        ]
    ),

    # 10. Cooling Fan Failure / Seized Fan
    FaultRule(
        fault_id="LAPTOP_COOLING_FAN_FAILURE",
        title="Seized, Burnt, or Broken Cooling Fan Motor",
        severity="MEDIUM",
        complexity="Medium (Intermediate)",
        recommendation="REPAIR (DIY Fan Replacement) / SERVICE (Professional)",
        rationale="Cooling fan bearings dry out or the small 5V brushless motor burns out. Swapping the fan unit restores airflow.",
        required_tools=["Precision Screwdriver Set", "Replacement OEM Cooling Fan", "Thermal Paste", "Tweezers"],
        estimated_cost="₹500 - ₹1,500 ($6 - $18)",
        estimated_time="25 - 40 minutes",
        safety_warnings=[
            "Do not use the laptop under heavy load when the fan is stopped to avoid permanent GPU/CPU silicon degradation."
        ],
        safe_checks=[
            "1. Check for startup error codes: 'CPU Fan Error' or 'System Fan 90B error'.",
            "2. Shine a flashlight into the vent grill: observe if the fan blades spin at all during power-on.",
            "3. Listen for rattling, grinding, or buzzing mechanical friction from the fan casing."
        ],
        advanced_steps=[
            "1. Power off, disconnect battery, and remove bottom cover.",
            "2. Unscrew fan bracket screws and unplug 4-pin PWM fan connector from motherboard.",
            "3. Check if blades spin freely by hand; if stiff, bearings are seized.",
            "4. Install replacement OEM fan unit, reconnect 4-pin cable firmly, and test before closing chassis."
        ],
        positive_keywords=[
            "fan not spinning", "fan stopped", "fan error", "fan 90b",
            "fan rattling", "fan grinding", "cooling fan dead", "cpu fan error", "fan failure"
        ],
        negative_keywords=["screen crack", "spill", "water"],
        common_symptoms=[
            "Cooling fan does not spin at all, even when laptop is hot",
            "Fan makes harsh rattling, screeching, or buzzing noise",
            "System displays 'Fan (90b) error' on startup screen"
        ],
        co_occurrence_patterns=[
            ["fan", "not spinning"],
            ["fan", "stopped"],
            ["fan", "error"],
            ["fan", "rattling"]
        ]
    ),
]


# ===========================================================================
# DEVICE REGISTRY (Extensible dict: device_name -> rule_list)
# ===========================================================================

DEVICE_REGISTRY: Dict[str, List[FaultRule]] = {
    "laptop": LAPTOP_RULES
}
registry = DEVICE_REGISTRY


def is_device_supported(raw_device: str) -> bool:
    return engine.normalize_device_name(raw_device) in DEVICE_REGISTRY


def get_canonical_device(raw_device: str) -> Optional[str]:
    canonical = engine.normalize_device_name(raw_device)
    return canonical if canonical in DEVICE_REGISTRY else None

# Aliases mapping to registry keys
DEVICE_ALIASES: Dict[str, str] = {
    "laptop": "laptop",
    "pc": "laptop",
    "notebook": "laptop",
    "computer": "laptop",
    "laptop computer": "laptop"
}

# Component names that the Roboflow model may return (all map to laptop)
LAPTOP_COMPONENT_TERMS = {
    "motherboard", "mainboard", "ram", "memory", "ssd", "hdd", "hard drive",
    "storage", "battery", "cooling fan", "cpu fan", "heatsink",
    "heat sink", "cpu", "processor", "gpu", "graphics card", "graphics",
    "screen", "display", "lcd", "keyboard", "touchpad", "trackpad",
    "charging port", "dc jack", "charger", "power adapter", "wifi card",
    "wireless card", "speaker", "webcam", "usb port", "ram slot", "sodimm",
    "vram", "cmos battery", "bios chip", "hinge", "laptop", "notebook",
    "computer", "pc",
}

# Backward compatibility alias
KNOWLEDGE_BASE: Dict[str, List[FaultRule]] = {
    "laptop": LAPTOP_RULES
}


# ===========================================================================
# SHARED SAFETY & HAZARD EVALUATION
# ===========================================================================

HARD_HAZARD_FAULT_IDS = frozenset({
    "LAPTOP_LIQUID_SPILL",
    "LAPTOP_SWOLLEN_BATTERY",
    "LAPTOP_MOTHERBOARD_SURGE_POWER_FAILURE",
})


def _displayed_confidence(score: float, fault_id: str) -> float:
    """Cap displayed confidence at 0.95; floor hard-hazard faults at 0.60.

    INCONCLUSIVE_INSPECTION_RECOMMENDED stays at or below 0.20 (typically 0.15).
    Does not change safety flags.
    """
    try:
        value = float(score)
    except (TypeError, ValueError):
        value = 0.0
    if fault_id == "INCONCLUSIVE_INSPECTION_RECOMMENDED":
        return round(min(max(value, 0.0), 0.20), 2)
    value = min(max(value, 0.0), 0.95)
    if fault_id in HARD_HAZARD_FAULT_IDS:
        value = max(value, 0.60)
    return round(value, 2)


HAZARD_KEYWORDS = [
    "smoke",
    "spark",
    "burning smell",
    "burnt smell",
    "shock",
    "swollen",
    "bulging",
    "liquid",
    "spill",
    "water",
    "wet",
    "hot to touch",
    "hot-to-touch",
    "smells like burning",
]

STOP_USING_ADVICE = [
    "CRITICAL SAFETY WARNING: Stop using the device immediately.",
    "Do NOT power on, plug in, or charge the device.",
    "Do NOT attempt to open the casing or tamper with internal components (high risk of fire, electric shock, or chemical hazard).",
    "Disconnect from the wall outlet only if the plug and cord are completely cool and safe to touch.",
    "Take the device to a certified professional technician or authorized service center."
]

STOP_USING_TROUBLESHOOTING_STEPS = [
    "1. Immediately stop using the device and do not press the power button.",
    "2. If safely accessible without touching hot, wet, or sparking areas, unplug the charger from the wall socket.",
    "3. Do NOT open the laptop casing, touch internal circuit boards, or attempt DIY teardown.",
    "4. Place the device in a well-ventilated, non-flammable area away from liquids and combustible items.",
    "5. Contact a qualified hardware repair professional or authorized manufacturer service center."
]

GENERAL_UNSUPPORTED_SAFETY_ADVICE = [
    "1. Disconnect the device from mains electrical power and unplug all chargers or batteries before handling.",
    "2. Never open sealed power supplies, high-voltage sections, or swollen battery packs.",
    "3. If you notice smoke, sparks, burning smells, or liquid exposure, cease using the device immediately and consult a certified repair professional."
]


def evaluate_safety(
    answers: Optional[Dict[str, Any]] = None,
    free_text: Optional[str] = None,
    existing_safety_alert: bool = False,
    existing_requires_pro: bool = False,
) -> Dict[str, Any]:
    """
    Shared safety evaluation for rule-based, questionnaire, and LLM diagnostic paths.
    - Scans real questionnaire keys.
    - Scans free-text description for confirmed hazard words using word stems.
    - 'Not sure' is treated as caution/uncertainty, never a confirmed hazard.
    - Never lowers an existing flag (monotonic safety).
    - Returns structured safety assessment with instructions and reasons.
    """
    answers = answers or {}
    safety_reasons = []
    hazard_detected = False
    caution_notes = []

    # 1. Real questionnaire keys evaluation
    hazards_val = str(answers.get("hazards", "")).strip().lower()
    if hazards_val == "yes":
        hazard_detected = True
        safety_reasons.append("User confirmed active hazard (smoke, sparks, burning smell, shock, or swollen battery).")
    elif hazards_val == "not sure":
        caution_notes.append("User was unsure about physical hazards; exercising elevated caution.")

    event_val = str(answers.get("recent_event", "")).strip().lower()
    if "liquid spill" in event_val or "liquid" in event_val or "spill" in event_val:
        hazard_detected = True
        safety_reasons.append("Recent event reported: Liquid spill (causes electrolytic short circuits and board corrosion).")
    elif "power surge" in event_val:
        hazard_detected = True
        safety_reasons.append("Recent event reported: Power surge / electrical event (potential capacitor/VRM destruction).")
    elif event_val == "unknown":
        caution_notes.append("Preceding event is unknown.")

    charging_val = str(answers.get("charging", "")).strip().lower()
    if "swollen" in charging_val or "bulging" in charging_val:
        hazard_detected = True
        safety_reasons.append("Battery reported as swollen/bulging (severe thermal runaway and fire hazard).")

    cause_val = str(answers.get("possible_cause", "")).strip().lower()
    if any(k in cause_val for k in ["liquid", "spill", "water", "coffee", "tea"]):
        hazard_detected = True
        safety_reasons.append(f"Possible cause noted as liquid exposure: '{answers.get('possible_cause')}'.")

    # 2. Scan all answers values + free_text for hazard keywords
    text_corpus_parts = []
    if free_text:
        text_corpus_parts.append(str(free_text))
    for k, v in answers.items():
        if isinstance(v, (str, list, dict)):
            text_corpus_parts.append(str(v))

    combined_text = " ".join(text_corpus_parts).lower()

    for kw in HAZARD_KEYWORDS:
        pattern = r"\b" + re.escape(kw) if " " not in kw else re.escape(kw)
        if re.search(pattern, combined_text):
            hazard_detected = True
            safety_reasons.append(f"Free-text symptom description contains confirmed hazard term: '{kw}'.")

    # Monotonic safety rule: NEVER lower a flag
    safety_alert = existing_safety_alert or hazard_detected
    requires_pro = existing_requires_pro or hazard_detected

    return {
        "hazard_detected": hazard_detected,
        "safety_alert": safety_alert,
        "requires_professional_attention": requires_pro,
        "safety_reasons": list(dict.fromkeys(safety_reasons)),
        "caution_notes": list(dict.fromkeys(caution_notes)),
        "stop_using_advice": STOP_USING_ADVICE if safety_alert else [],
        "troubleshooting_override": STOP_USING_TROUBLESHOOTING_STEPS if safety_alert else None,
        "forced_action": "SERVICE (Professional)" if safety_alert else None,
        "forced_complexity": "Dangerous (Do Not DIY)" if safety_alert else None,
    }


# ===========================================================================
# DIAGNOSIS ENGINE LOGIC
# ===========================================================================

class FaultDiagnosisEngine:
    """
    Evaluates detected device and user-described symptoms against deterministic,
    expert-curated rule bases.
    """

    def __init__(self, registry_dict: Optional[Dict[str, List[FaultRule]]] = None):
        self.registry = registry_dict or DEVICE_REGISTRY
        self.kb = self.registry

    def normalize_device_name(self, raw_device: str) -> str:
        """Map YOLO detections or user labels to canonical device categories."""
        raw = " ".join(str(raw_device).strip().lower().replace("_", " ").split())
        if raw in DEVICE_ALIASES:
            return DEVICE_ALIASES[raw]
        if raw in LAPTOP_COMPONENT_TERMS or any(term in raw for term in LAPTOP_COMPONENT_TERMS):
            return "laptop"
        return raw

    def extract_symptom_tokens(self, text: str) -> List[str]:
        """Tokenize and clean symptom description."""
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return [word for word in cleaned.split() if len(word) > 2]

    def score_rule(self, rule: FaultRule, symptom_text: str, symptom_tokens: List[str]) -> Dict[str, Any]:
        """
        Compute match confidence, evidence, and co-occurrence boosts for a rule.
        Enforces strict whole-word / phrase matching.
        """
        text_lower = symptom_text.lower()
        matched_positive = []
        matched_negative = []
        co_occurrences_found = []

        # 1. Match positive keywords and phrases with whole-word/phrase boundaries
        for kw in rule.positive_keywords:
            kw_clean = kw.strip().lower()
            pattern = r"\b" + re.escape(kw_clean) + r"\b"
            if re.search(pattern, text_lower):
                matched_positive.append(kw_clean)

        # 2. Check negative (counter-indicator) keywords
        for neg in rule.negative_keywords:
            neg_clean = neg.strip().lower()
            pattern = r"\b" + re.escape(neg_clean) + r"\b"
            if re.search(pattern, text_lower):
                matched_negative.append(neg_clean)

        # 3. High-confidence co-occurrence patterns (whole-word/phrase matching)
        for pattern in rule.co_occurrence_patterns:
            all_matched = True
            for term in pattern:
                t_pattern = r"\b" + re.escape(term.strip().lower()) + r"\b"
                if not re.search(t_pattern, text_lower):
                    all_matched = False
                    break
            if all_matched:
                co_occurrences_found.append(" + ".join(pattern))

        # Base score calculation
        if not matched_positive and not co_occurrences_found:
            return {"score": 0.0, "evidence": [], "matches": []}

        score = min(0.40, len(matched_positive) * 0.15)

        if co_occurrences_found:
            score += len(co_occurrences_found) * 0.35

        if matched_negative:
            score -= len(matched_negative) * 0.25

        score = max(0.0, min(0.99, score))

        evidence = []
        if matched_positive:
            evidence.append(f"Reported symptoms directly match indicator terms: {', '.join(set(matched_positive))}")
        if co_occurrences_found:
            evidence.append(f"Strong symptom co-occurrence pattern identified: [{', '.join(co_occurrences_found)}]")
        if matched_negative:
            evidence.append(f"Contains counter-indicator terms: {', '.join(set(matched_negative))}")

        return {
            "score": round(score, 2),
            "evidence": evidence,
            "matches": list(set(matched_positive))
        }

    def diagnose(
        self,
        device: str,
        user_symptoms: str,
        detection_confidence: Optional[float] = None,
        detection_bbox: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Produce a complete structured diagnostic report.
        Unsupported devices return a structured 'unsupported_device' result.
        If nothing scores > 0, returns 'Inconclusive - professional inspection recommended' (never rules[0]).
        """
        canonical_device = self.normalize_device_name(device)
        if canonical_device not in self.registry:
            return {
                "status": "unsupported_device",
                "supported": False,
                "device": device,
                "message": f"Device category '{device}' is not supported yet. 'Repair Before Replace' currently provides in-depth diagnosis for Laptops and PCs only.",
                "safety_advice": GENERAL_UNSUPPORTED_SAFETY_ADVICE,
                "diagnosis": None,
                "repair_recommendation": None,
            }

        rules = self.registry.get(canonical_device, [])
        symptom_tokens = self.extract_symptom_tokens(user_symptoms)

        # Evaluate safety on user_symptoms text
        safety_eval = evaluate_safety(free_text=user_symptoms)

        scored_candidates = []
        for rule in rules:
            analysis = self.score_rule(rule, user_symptoms, symptom_tokens)
            if analysis["score"] > 0:
                scored_candidates.append({
                    "rule": rule,
                    "score": analysis["score"],
                    "evidence": analysis["evidence"],
                    "matches": analysis["matches"]
                })

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)

        if not scored_candidates:
            # (b) Inconclusive primary diagnosis if nothing scores above 0 (never fall back to rules[0])
            primary_diag = {
                "fault_id": "INCONCLUSIVE_INSPECTION_RECOMMENDED",
                "fault_title": "Inconclusive - professional inspection recommended",
                "confidence_score": _displayed_confidence(0.15, "INCONCLUSIVE_INSPECTION_RECOMMENDED"),
                "severity": "LOW",
                "repair_complexity": "Unknown",
                "recommendation": "SERVICE (Professional)",
                "recommendation_rationale": "The symptoms provided did not match any single known failure mode with sufficient confidence. A hands-on hardware inspection by a technician is recommended.",
                "estimated_diy_time": "Not applicable",
                "estimated_parts_cost": "To be quoted after inspection",
                "required_tools": [],
                "safety_warnings": ["Do not force power-on if unusual smells, sounds, or excessive heat develop."],
                "safe_checks": [
                    "1. Verify that the power adapter is properly plugged into a known working wall outlet.",
                    "2. Disconnect all external USB accessories, monitors, and docks, then test.",
                    "3. Clean exterior ventilation ports with a dry cloth."
                ],
                "advanced_steps": [],
                "troubleshooting_steps": [
                    "1. Verify that the power adapter is properly plugged into a known working wall outlet.",
                    "2. Disconnect all external USB accessories, monitors, and docks, then test.",
                    "3. Clean exterior ventilation ports with a dry cloth.",
                    "4. Bring the device to a certified repair technician for physical diagnostic testing."
                ],
                "supporting_evidence": ["Reported symptoms did not match specific component failure patterns."]
            }
            alternatives = []
            diagnosis_note = "Inconclusive check: symptoms did not score above zero for any single known fault."
            critical_alert = "Always disconnect power before inspection."
        else:
            top = scored_candidates[0]
            primary_rule = top["rule"]
            confidence_level = _displayed_confidence(top["score"], primary_rule.fault_id)
            diagnosis_note = f"High confidence rule match for '{primary_rule.title}' based on provided symptoms."
            evidence = top["evidence"]

            primary_diag = {
                "fault_id": primary_rule.fault_id,
                "fault_title": primary_rule.title,
                "confidence_score": confidence_level,
                "severity": primary_rule.severity,
                "repair_complexity": primary_rule.complexity,
                "recommendation": primary_rule.recommendation,
                "recommendation_rationale": primary_rule.rationale,
                "estimated_diy_time": primary_rule.estimated_time,
                "estimated_parts_cost": primary_rule.estimated_cost,
                "required_tools": list(primary_rule.required_tools),
                "safety_warnings": list(primary_rule.safety_warnings),
                "safe_checks": list(primary_rule.safe_checks),
                "advanced_steps": list(primary_rule.advanced_steps),
                "troubleshooting_steps": list(primary_rule.troubleshooting_steps),
                "supporting_evidence": evidence + safety_eval["safety_reasons"]
            }

            alternatives = []
            for item in scored_candidates[1:3]:
                alt_rule = item["rule"]
                alternatives.append({
                    "fault_id": alt_rule.fault_id,
                    "fault_title": alt_rule.title,
                    "confidence_score": _displayed_confidence(item["score"], alt_rule.fault_id),
                    "recommendation": alt_rule.recommendation,
                    "supporting_evidence": item["evidence"]
                })

            critical_alert = primary_rule.safety_warnings[0] if primary_rule.safety_warnings else "Always disconnect power before inspection."

        # Hard hazard override
        if safety_eval["safety_alert"]:
            primary_diag["recommendation"] = "SERVICE (Professional)"
            primary_diag["repair_complexity"] = "Dangerous (Do Not DIY)"
            primary_diag["safe_checks"] = STOP_USING_ADVICE
            primary_diag["advanced_steps"] = []  # CRITICAL / SERVICE-only rules have empty advanced_steps
            primary_diag["troubleshooting_steps"] = STOP_USING_TROUBLESHOOTING_STEPS
            primary_diag["safety_warnings"] = STOP_USING_ADVICE
            critical_alert = STOP_USING_ADVICE[0]

        return {
            "status": "success",
            "supported": True,
            "device": canonical_device,
            "original_detection": {
                "detected_device": device,
                "detection_confidence": detection_confidence,
                "bounding_box": detection_bbox
            },
            "user_symptoms": user_symptoms,
            "diagnosis_note": diagnosis_note,
            "safety_alert": safety_eval["safety_alert"],
            "requires_professional_attention": safety_eval["requires_professional_attention"],
            "primary_diagnosis": primary_diag,
            "alternative_diagnoses": alternatives,
            "critical_safety_alert": critical_alert
        }

    def _diagnose_with_gemini(
        self,
        questionnaire_payload: Dict[str, Any],
        image_path: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Gemini path: include the allowed fault_id list in the prompt, require it to return
        one fault_id plus confidence and reasoning, and fill everything else from the matching FaultRule.
        Unknown id -> fall back cleanly to the rule engine.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None

        device = questionnaire_payload.get("device_type", "laptop")
        canonical_device = self.normalize_device_name(device)
        rules = self.registry.get(canonical_device, [])
        if not rules:
            return None

        try:
            from google import genai
            from google.genai import types

            findings = questionnaire_payload.get("detected_findings", [])
            answers = questionnaire_payload.get("answers", {})
            model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

            # Build allowed fault_id list for prompt
            allowed_fault_ids = [r.fault_id for r in rules]
            catalog_lines = [f"- {r.fault_id}: {r.title} (Severity: {r.severity})" for r in rules]

            prompt = f"""
You are the diagnostic selector for a PC/laptop hardware repair application.
Analyze the user answers and visual findings.
You MUST select exactly ONE fault_id from the ALLOWED FAULT IDS below.
Do not invent any new fault_id, cost, time, or repair step. All steps and costs come strictly from the verified database.

ALLOWED FAULT IDS:
{chr(10).join(catalog_lines)}

Device: {canonical_device}
Visual findings: {json.dumps(findings, ensure_ascii=False)}
Questionnaire answers: {json.dumps(answers, ensure_ascii=False)}

Return ONLY valid JSON with this exact schema:
{{
  "selected_fault_id": "EXACT_FAULT_ID_FROM_ALLOWED_LIST",
  "confidence": 0.85,
  "reasoning": "brief explanation grounded strictly in the symptoms"
}}
"""
            contents = [prompt]
            if image_path:
                path = Path(image_path).expanduser()
                if path.is_file():
                    suffix = path.suffix.lower()
                    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                            ".png": "image/png", ".webp": "image/webp"}.get(suffix)
                    if mime:
                        contents.append(types.Part.from_bytes(
                            data=path.read_bytes(), mime_type=mime
                        ))

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    max_output_tokens=400,
                ),
            )
            data = json.loads(response.text)
            selected_id = str(data.get("selected_fault_id", "")).strip()

            # Unknown id -> fall back to the rule engine
            if selected_id not in allowed_fault_ids:
                return None

            matched_rule = next(r for r in rules if r.fault_id == selected_id)

            safety_eval = evaluate_safety(
                answers=answers,
                existing_safety_alert=bool(questionnaire_payload.get("safety_alert")),
                existing_requires_pro=bool(questionnaire_payload.get("requires_professional_attention"))
            )

            rec_action = matched_rule.recommendation
            rec_complexity = matched_rule.complexity
            safe_checks = list(matched_rule.safe_checks)
            advanced_steps = list(matched_rule.advanced_steps)
            troubleshooting_steps = list(matched_rule.troubleshooting_steps)
            safety_instructions = list(matched_rule.safety_warnings)

            if safety_eval["safety_alert"]:
                rec_action = "SERVICE (Professional)"
                rec_complexity = "Dangerous (Do Not DIY)"
                safe_checks = STOP_USING_ADVICE
                advanced_steps = []
                troubleshooting_steps = STOP_USING_TROUBLESHOOTING_STEPS
                safety_instructions = STOP_USING_ADVICE

            result = dict(questionnaire_payload)
            result["safety_alert"] = safety_eval["safety_alert"]
            result["requires_professional_attention"] = safety_eval["requires_professional_attention"]
            result["supported"] = True
            result["diagnosis"] = {
                "primary_fault": matched_rule.title,
                "fault_id": matched_rule.fault_id,
                "confidence": _displayed_confidence(
                    data.get("confidence", 0.8), matched_rule.fault_id
                ),
                "severity": matched_rule.severity,
                "reasoning": data.get("reasoning", matched_rule.rationale),
                "supporting_evidence": safety_eval["safety_reasons"] or [data.get("reasoning", "")],
                "secondary_faults": [],
                "safe_checks": safe_checks,
                "advanced_steps": advanced_steps,
                "safety_instructions": safety_instructions,
                "troubleshooting_steps": troubleshooting_steps,
            }
            result["repair_recommendation"] = {
                "action": rec_action,
                "complexity": rec_complexity,
                "rationale": matched_rule.rationale,
                "estimated_cost": matched_rule.estimated_cost,
                "estimated_time": matched_rule.estimated_time,
                "required_tools": list(matched_rule.required_tools),
            }
            result["diagnosis_provider"] = "gemini_selected_kb_backed"
            return result

        except Exception:
            return None

    def diagnose_questionnaire(
        self,
        questionnaire_payload: Dict[str, Any],
        image_path: Optional[str] = None,
        use_gemini: bool = True
    ) -> Dict[str, Any]:
        """
        Takes a questionnaire output payload and populates structured diagnosis.
        (a) Symptom text built from answer VALUES only, skipping neutral answers.
        """
        result = dict(questionnaire_payload)
        device = result.get("device_type", "laptop")
        canonical_device = self.normalize_device_name(device)

        if canonical_device not in self.registry:
            return {
                "status": "unsupported_device",
                "supported": False,
                "module": "questionnaire",
                "device_type": device,
                "message": f"Device category '{device}' is not supported yet. 'Repair Before Replace' currently provides in-depth diagnosis for Laptops and PCs only.",
                "safety_advice": GENERAL_UNSUPPORTED_SAFETY_ADVICE,
                "safety_alert": False,
                "requires_professional_attention": False,
                "diagnosis": None,
                "repair_recommendation": None,
            }

        answers = result.get("answers", {})
        findings = result.get("detected_findings", [])
        free_text = result.get("free_text")
        safety_text_parts = [part for part in (free_text, " ".join(findings) if findings else None) if part]

        # Shared safety evaluation
        safety_eval = evaluate_safety(
            answers=answers,
            free_text=" ".join(safety_text_parts) if safety_text_parts else None,
            existing_safety_alert=bool(result.get("safety_alert")),
            existing_requires_pro=bool(result.get("requires_professional_attention"))
        )

        if use_gemini:
            gemini_result = self._diagnose_with_gemini(questionnaire_payload, image_path=image_path)
            if gemini_result is not None:
                return gemini_result

        # (a) Build symptom text from answer VALUES only, skip neutral answers
        neutral_answers = {
            "", "no", "none", "not tested", "normal", "works normally",
            "normal temperature", "no known event", "unknown", "n/a",
            "none of the above", "not sure", "not sure / not applicable",
            "no component label returned", "starts normally"
        }
        symptom_values = []
        if findings:
            symptom_values.extend(findings)

        for k, ans in answers.items():
            cleaned = str(ans).strip()
            if cleaned.lower() not in neutral_answers:
                symptom_values.append(cleaned)
        if free_text:
            symptom_values.append(str(free_text).strip())

        combined_symptoms = ". ".join(symptom_values) or "Device is not operating as expected."

        base_diag = self.diagnose(
            device=canonical_device,
            user_symptoms=combined_symptoms,
            detection_confidence=result.get("detection_confidence", 0.90)
        )

        primary = base_diag["primary_diagnosis"]
        alternatives = base_diag.get("alternative_diagnoses", [])

        safety_alert = safety_eval["safety_alert"] or base_diag.get("safety_alert", False)
        requires_pro = safety_eval["requires_professional_attention"] or base_diag.get("requires_professional_attention", False)

        rec_action = primary["recommendation"]
        rec_complexity = primary["repair_complexity"]
        safe_checks = list(primary.get("safe_checks", []))
        advanced_steps = list(primary.get("advanced_steps", []))
        troubleshooting_steps = list(primary.get("troubleshooting_steps", []))
        safety_instructions = list(primary.get("safety_warnings", []))

        if safety_alert:
            rec_action = "SERVICE (Professional)"
            rec_complexity = "Dangerous (Do Not DIY)"
            safe_checks = STOP_USING_ADVICE
            advanced_steps = []
            troubleshooting_steps = STOP_USING_TROUBLESHOOTING_STEPS
            safety_instructions = STOP_USING_ADVICE

        result["safety_alert"] = safety_alert
        result["requires_professional_attention"] = requires_pro
        result["supported"] = True

        result["diagnosis"] = {
            "primary_fault": primary["fault_title"],
            "fault_id": primary["fault_id"],
            "confidence": _displayed_confidence(primary["confidence_score"], primary["fault_id"]),
            "severity": primary["severity"],
            "supporting_evidence": primary["supporting_evidence"] + safety_eval["safety_reasons"],
            "safe_checks": safe_checks,
            "advanced_steps": advanced_steps,
            "safety_instructions": safety_instructions,
            "troubleshooting_steps": troubleshooting_steps,
            "secondary_faults": alternatives,
        }

        result["repair_recommendation"] = {
            "action": rec_action,
            "complexity": rec_complexity,
            "rationale": primary.get("recommendation_rationale", ""),
            "estimated_cost": primary.get("estimated_parts_cost", "Unknown"),
            "estimated_time": primary.get("estimated_diy_time", "Unknown"),
            "required_tools": primary.get("required_tools", [])
        }

        return result

    def get_supported_devices(self) -> List[str]:
        return list(self.registry.keys())

    def get_device_symptom_checklist(self, device: str) -> List[str]:
        canonical = self.normalize_device_name(device)
        rules = self.registry.get(canonical, [])
        checklist = []
        for r in rules:
            checklist.extend(r.common_symptoms)
        return checklist


# Global engine singleton
engine = FaultDiagnosisEngine()


def run_diagnosis(device: str, symptoms: str) -> Dict[str, Any]:
    """Simple API helper function."""
    return engine.diagnose(device=device, user_symptoms=symptoms)


def process_questionnaire(
    questionnaire_data: Dict[str, Any],
    image_path: Optional[str] = None,
    use_gemini: bool = True,
) -> Dict[str, Any]:
    """Diagnose questionnaire answers with shared safety, falling back to local rules."""
    return engine.diagnose_questionnaire(
        questionnaire_data, image_path=image_path, use_gemini=use_gemini
    )
