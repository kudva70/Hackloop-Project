"""Phase 4: Safe Troubleshooting Guidance."""

DANGEROUS_KEYWORDS = (
    "smoke",
    "sparks",
    "burning smell",
    "burnt smell",
    "electric shock",
    "swollen battery",
    "bulging battery",
    "exposed wire",
    "fire",
    "liquid spill",
    "water spill",
)


def assess_safety(fault, existing_warnings=None):
    """Return safety status and warnings before DIY instructions."""
    text = str(fault).lower()
    warnings = list(existing_warnings or [])

    danger_found = any(
        keyword in text for keyword in DANGEROUS_KEYWORDS
    )

    if danger_found:
        warnings.extend([
            "Stop ordinary troubleshooting immediately.",
            "Do not open or power on the device.",
            "Avoid touching damaged electrical components.",
            "Seek assistance from a qualified repair professional.",
        ])

        return {
            "safe_for_diy": False,
            "warnings": list(dict.fromkeys(warnings)),
            "recommendation": "SERVICE (Professional)",
        }

    return {
        "safe_for_diy": True,
        "warnings": warnings + [
            "Inspect only accessible external parts.",
            "Disconnect power before appropriate physical inspections.",
            "Stop if damage, unusual heat, or an electrical hazard appears.",
        ],
        "recommendation": "Proceed with appropriate low-risk checks",
    }


def adapt_steps(steps, skill_level="beginner"):
    """Filter instructions according to the user's skill level."""
    level = str(skill_level).strip().lower()

    if level not in {"beginner", "intermediate", "professional"}:
        level = "beginner"

    steps = list(steps or [])

    if level == "beginner":
        return [
            step for step in steps
            if not any(term in str(step).lower() for term in (
                "open the device",
                "multimeter",
                "motherboard",
                "rewinding",
                "discharge capacitor",
                "replace internal",
                "disconnect the battery cable",
                "remove the heatsink",
            ))
        ]

    if level == "intermediate":
        return [
            step for step in steps
            if not any(term in str(step).lower() for term in (
                "rewinding",
                "discharge capacitor",
                "high-voltage",
            ))
        ]

    # Professional users may receive the existing full list.
    # Hazardous work still requires appropriate training and equipment.
    return steps


def prepare_guidance(fault, steps, skill_level="beginner",
                     existing_warnings=None):
    """Prepare safety-aware troubleshooting output."""
    safety = assess_safety(fault, existing_warnings)

    if not safety["safe_for_diy"]:
        return {
            "safety": safety,
            "steps": [],
            "verdict": "professional_repair_required",
        }

    return {
        "safety": safety,
        "steps": adapt_steps(steps, skill_level),
        "verdict": "safe_basic_checks",
    }