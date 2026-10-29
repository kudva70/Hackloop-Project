"""
Repair-vs-Replace decision engine.
Deterministic, traceable, and explainable: no AI model decides this verdict.
Gemma / LLM only explains the result afterwards.

Outputs exactly three canonical verdicts:
  1. "REPAIR (DIY)"
  2. "SERVICE (technician)"
  3. "REPLACE (device)"

Distinguishes part replacement (e.g. replacing an LCD panel or battery) from whole-device replacement.
All cost/lifespan defaults are transparent assumptions (INR) that the user can override.
"""

from typing import Dict, Any, Optional, Tuple, List
import re

DEVICE_DEFAULTS = {            # replacement_cost in INR, life in years (assumptions)
    "laptop":      {"replacement_cost": 45000, "life_years": 5},
    "fan":         {"replacement_cost": 2500,  "life_years": 10},
    "bulb":        {"replacement_cost": 150,   "life_years": 3},
    "switchboard": {"replacement_cost": 900,   "life_years": 15},
}
GENERIC = {"replacement_cost": 5000, "life_years": 6}

# Typical device weight in kilograms (assumption, editable)
DEVICE_MASS_KG = {
    "laptop": 2.1,      # Average notebook/laptop mass
    "fan": 3.5,         # Ceiling / table fan
    "bulb": 0.1,        # LED bulb
    "switchboard": 0.3, # Switchboard assembly
}
DEFAULT_DEVICE_MASS_KG = 2.0

# Decision thresholds
REPLACE_RATIO = 0.60           # repair costs >= 60% of a new device -> REPLACE (device)
REPLACE_RATIO_OLD = 0.40       # ... or >= 40% when the device is already past 70% of typical life
OLD_LIFE_USED = 0.70           # 70% of lifespan


def parse_cost(text: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Parse only integers that follow ₹ or Rs (case-insensitive). Ignore $ amounts.

    '₹500 ($6)' -> (500, 500); '₹1,500 - ₹5,000 ($20 - $60)' -> (1500, 5000).
    Returns (None, None) if no INR figures are found.
    """
    matches = re.findall(r"(?:₹|rs)\s*(\d[\d,]*)", str(text or ""), flags=re.IGNORECASE)
    nums = [int(n.replace(",", "")) for n in matches]
    if not nums:
        return None, None
    return min(nums[:2]), max(nums[:2])


def decide(
    device: str,
    report: Dict[str, Any],
    cv_result: Optional[Dict[str, Any]] = None,
    age_years: Optional[float] = None,
    replacement_cost: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes deterministic repair-vs-replace decision yielding exactly one of:
      - "REPAIR (DIY)"
      - "SERVICE (technician)"
      - "REPLACE (device)"
    """
    primary = report.get("primary_diagnosis") or {}
    defaults = DEVICE_DEFAULTS.get(str(device).lower(), GENERIC)
    replacement = float(replacement_cost or defaults["replacement_cost"])
    life_years = float(defaults["life_years"])

    fault_id = str(primary.get("fault_id", ""))
    rec = str(primary.get("recommendation", "")).lower()
    severity = str(primary.get("severity", "")).upper()
    complexity = str(primary.get("repair_complexity", "")).lower()
    confidence = float(primary.get("confidence_score", 0.5))

    low, high = parse_cost(primary.get("estimated_parts_cost"))
    mid = (low + high) / 2.0 if low is not None and high is not None else None
    ratio = mid / replacement if mid is not None else None
    life_used = (age_years / life_years) if age_years is not None else None

    # Hard safety overrides: CRITICAL severity, swollen battery, or liquid spill always gives SERVICE
    is_critical = (
        severity == "CRITICAL"
        or "critical" in severity.lower()
        or fault_id in ("LAPTOP_SWOLLEN_BATTERY", "LAPTOP_LIQUID_SPILL", "LAPTOP_MOTHERBOARD_SURGE_POWER_FAILURE")
    )
    safety_hold = (
        is_critical
        or bool((cv_result or {}).get("safety_flag"))
        or bool(report.get("safety_alert"))
        or bool(report.get("requires_professional_attention"))
        or "dangerous" in complexity
    )

    reasons = []
    low_confidence = False
    confidence_notes = []

    # Flag unparsed cost or inconclusive diagnosis
    if low is None or high is None:
        low_confidence = True
        confidence_notes.append("Repair cost could not be parsed from diagnosis (requires physical quote).")

    if fault_id == "INCONCLUSIVE_INSPECTION_RECOMMENDED" or confidence < 0.30:
        low_confidence = True
        confidence_notes.append("Diagnosis is inconclusive and requires professional bench testing.")

    # 1. Determine Verdict
    if safety_hold:
        verdict = "SERVICE (technician)"
        reasons.append("Safety hazard / critical hardware risk detected (e.g. swollen battery, liquid exposure, or power surge). Immediate professional servicing is required; do not attempt DIY repair.")
    elif ratio is not None:
        reasons.append(f"Estimated repair parts cost {_inr(low)}-{_inr(high)} is {ratio:.0%} of a new device ({_inr(replacement)}).")
        if ratio >= REPLACE_RATIO:
            verdict = "REPLACE (device)"
            reasons.append(f"Repair cost exceeds {REPLACE_RATIO:.0%} of the replacement price. Replacing the entire device is more economical than repairing.")
        elif ratio >= REPLACE_RATIO_OLD and life_used is not None and life_used >= OLD_LIFE_USED:
            verdict = "REPLACE (device)"
            reasons.append(f"The device is {age_years:g} years old ({life_used:.0%} through typical {life_years:g}-year life) and repair cost ({ratio:.0%} of new) is substantial. Replacing the entire device is recommended.")
        else:
            # Repair is economical
            needs_pro = ("service" in rec or "professional" in rec or "technician" in rec
                         or any(w in complexity for w in ("advanced", "hard", "expert", "high")))
            verdict = "SERVICE (technician)" if needs_pro else "REPAIR (DIY)"
            reasons.append(f"Repair is clearly cheaper than replacing the entire device (estimated saving: {_inr(replacement - mid)}).")
            # Distinguish part replacement from whole device replacement
            if "replace" in rec:
                reasons.append(f"Note: Resolving this issue involves replacing a specific component ({primary.get('fault_title', 'faulty part')}), which preserves the laptop at a fraction of replacement cost.")
    else:
        # Cost not parsed / inconclusive
        needs_pro = ("service" in rec or "professional" in rec or "technician" in rec
                     or any(w in complexity for w in ("advanced", "hard", "expert", "high")))
        verdict = "SERVICE (technician)" if needs_pro else "REPAIR (DIY)"
        reasons.append("Exact parts cost could not be parsed; verdict assigned based on repair complexity and safety profile.")

    # Photo analysis findings
    findings = (cv_result or {}).get("findings", [])
    for f in findings:
        reasons.append(f"Photo analysis: {f['label'].lower()} ({f['level']}). {f['detail']}.")

    if confidence_notes:
        reasons.extend([f"Low-confidence note: {n}" for n in confidence_notes])

    # Estimated savings
    saving = None
    if verdict != "REPLACE (device)" and mid is not None:
        saving = max(int(replacement - mid), 0)

    # 2. Traceable decision factors list
    decision_factors = [
        {
            "factor": "cost_ratio",
            "repair_cost_mid_inr": mid,
            "replacement_cost_inr": replacement,
            "ratio": round(ratio, 3) if ratio is not None else None,
            "threshold_replace": REPLACE_RATIO,
            "threshold_replace_old": REPLACE_RATIO_OLD,
            "detail": f"Repair is {ratio:.1%} of replacement cost" if ratio is not None else "Cost not parsed"
        },
        {
            "factor": "device_age",
            "age_years": age_years,
            "typical_life_years": life_years,
            "life_used_ratio": round(life_used, 3) if life_used is not None else None,
            "detail": f"{age_years:g} years old ({life_used:.0%} of typical {life_years:g}-yr life)" if life_used is not None else f"Age not entered (typical lifespan: {life_years:g} years)"
        },
        {
            "factor": "safety_hold",
            "active": safety_hold,
            "severity": severity,
            "detail": "Safety hazard / critical severity override active" if safety_hold else "No safety hold"
        },
        {
            "factor": "complexity",
            "level": primary.get("repair_complexity", "Unknown"),
            "detail": f"Complexity: {primary.get('repair_complexity', 'Unknown')}"
        },
        {
            "factor": "confidence",
            "diagnostic_confidence": confidence,
            "is_low_confidence": low_confidence,
            "detail": ("Low-confidence verdict: " + "; ".join(confidence_notes)) if low_confidence else f"High confidence diagnosis ({confidence:.0%})"
        }
    ]

    # 3. E-waste impact assessment (estimates)
    dev_key = str(device).lower()
    dev_mass = DEVICE_MASS_KG.get(dev_key, DEFAULT_DEVICE_MASS_KG)

    if verdict in ("REPAIR (DIY)", "SERVICE (technician)"):
        e_waste = {
            "status": "avoided",
            "amount_kg": round(dev_mass, 2),
            "unit": "kg",
            "assumption": f"Typical {dev_key} mass is estimated at {dev_mass:g} kg (editable assumption).",
            "summary": f"Estimated e-waste avoided: ~{dev_mass:g} kg (estimate based on typical {dev_key} mass).",
        }
    else:
        e_waste = {
            "status": "recycling_advice",
            "amount_kg": None,
            "unit": "kg",
            "assumption": f"Typical {dev_key} mass is estimated at {dev_mass:g} kg (editable assumption).",
            "advice": (
                "Do not dispose of this device in household trash. Hand it over to an authorized e-waste recycler "
                "or manufacturer take-back channel. Ensure personal data is removed (securely wipe or extract storage drives) "
                "before handing over for recycling."
            ),
            "summary": "Recycling advice: Hand over to an authorized e-waste recycler (do not bin); wipe personal data from storage (estimate).",
        }

    return {
        "verdict": verdict,
        "safety_hold": safety_hold,
        "low_confidence": low_confidence,
        "repair_cost_range": [low, high],
        "replacement_cost": replacement,
        "cost_ratio": round(ratio, 3) if ratio is not None else None,
        "estimated_saving": saving,
        "e_waste": e_waste,
        "reasons": reasons,
        "decision_factors": decision_factors,
        "assumptions": [
            f"Replacement price: {_inr(replacement)}" + (" (default assumption; override with --replacement-cost)" if not replacement_cost else " (entered by user)"),
            f"Typical lifespan: {life_years:g} years" + (f"; device age: {age_years:g} years (user entered)" if age_years is not None else " (default assumption; override with --age)"),
            f"Typical {dev_key} mass for e-waste calculations: {dev_mass:g} kg (estimate assumption).",
            "Repair cost is an estimated parts range from knowledge base, not a formal service center quote.",
            "Part replacement (e.g. replacing an LCD panel or battery) is categorized as repair of the device, not whole-device replacement."
        ],
    }


def _inr(n: Optional[float]) -> str:
    return f"Rs {int(n):,}" if n is not None else "n/a"
