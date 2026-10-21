import json
from datetime import datetime, timezone


# Supported device labels from the object-detection branch.
DEVICE_ALIASES = {
    "laptop computer": "laptop",
    "mobile phone": "phone",
    "smartphone": "phone",
    "cell phone": "phone",
    "headphones": "bluetooth headphones",
    "earphones": "bluetooth headphones",
    "watch": "electronic watch",
    "usb hub": "hub",
    "monitor": "screen",
    "display": "screen",
    "computer mouse": "mouse",
}

DEVICE_QUESTIONS = {
    "laptop": {
        "power": (
            "Does the laptop turn on?",
            ["Yes", "No", "Sometimes"],
        ),
        "display": (
            "What happens to the display?",
            [
                "Works normally",
                "Black screen",
                "Flickering",
                "Cracked or physically damaged",
                "Lines or unusual colours",
                "Not sure",
            ],
        ),
        "performance": (
            "How is its performance?",
            [
                "Normal",
                "Very slow",
                "Overheating",
                "Freezes frequently",
                "Shuts down unexpectedly",
                "Not sure",
            ],
        ),
        "charging": (
            "What happens when you charge it?",
            [
                "Charges normally",
                "Does not charge",
                "Charges slowly",
                "Charges intermittently",
                "Not tested",
            ],
        ),
    },
    "phone": {
        "power": (
            "Does the phone turn on?",
            ["Yes", "No", "Sometimes"],
        ),
        "screen": (
            "What is the screen condition?",
            [
                "Works normally",
                "Cracked",
                "Black screen",
                "Touch is not working",
                "Flickering or lines",
                "Not sure",
            ],
        ),
        "charging": (
            "What happens when you charge it?",
            [
                "Charges normally",
                "Does not charge",
                "Charges slowly",
                "Charges intermittently",
                "Not tested",
            ],
        ),
    },
    "keyboard": {
        "keys": (
            "What is wrong with the keys?",
            [
                "All keys work",
                "Some keys do not work",
                "Keys are stuck",
                "Keys are physically damaged",
                "Not sure",
            ],
        ),
        "connection": (
            "What about the connection?",
            [
                "Connects normally",
                "Does not connect",
                "Disconnects frequently",
                "Not sure",
            ],
        ),
    },
    "mouse": {
        "movement": (
            "Does the pointer move correctly?",
            [
                "Yes",
                "No movement",
                "Erratic movement",
                "Intermittent movement",
            ],
        ),
        "buttons": (
            "Are the buttons working?",
            [
                "All work",
                "Left button problem",
                "Right button problem",
                "Scroll wheel problem",
                "Not sure",
            ],
        ),
    },
    "bluetooth headphones": {
        "audio": (
            "What audio problem occurs?",
            [
                "Audio works normally",
                "No sound",
                "Sound in one ear only",
                "Distorted audio",
                "Audio cuts out",
            ],
        ),
        "connection": (
            "Can the headphones connect?",
            [
                "Connect normally",
                "Cannot pair",
                "Disconnect frequently",
                "Not tested",
            ],
        ),
        "battery": (
            "How is the battery?",
            [
                "Normal",
                "Drains quickly",
                "Does not charge",
                "Not sure",
            ],
        ),
    },
    "electronic watch": {
        "display": (
            "What is the display condition?",
            [
                "Works normally",
                "Blank screen",
                "Cracked screen",
                "Flickering",
                "Not sure",
            ],
        ),
        "power": (
            "Does the watch turn on?",
            ["Yes", "No", "Sometimes"],
        ),
    },
    "power bank": {
        "charging": (
            "Does the power bank charge?",
            ["Yes", "No", "Intermittently", "Not tested"],
        ),
        "output": (
            "Does it charge other devices?",
            ["Yes", "No", "Charges slowly", "Not tested"],
        ),
    },
    "hub": {
        "connection": (
            "Are connected devices detected?",
            [
                "All devices work",
                "Some devices are not detected",
                "No devices are detected",
                "Connection is intermittent",
            ],
        ),
        "ports": (
            "What is the port condition?",
            [
                "Looks normal",
                "Physically damaged",
                "Loose connection",
                "Not sure",
            ],
        ),
    },
    "screen": {
        "display": (
            "What display problem occurs?",
            [
                "Works normally",
                "Cracked screen",
                "Black screen",
                "Flickering",
                "Lines or unusual colours",
                "Not sure",
            ],
        ),
    },
}

COMMON_QUESTIONS = {
    "duration": (
        "How long has the problem been occurring?",
        [
            "Started today",
            "A few days",
            "A few weeks",
            "Several months",
            "Not sure",
        ],
    ),
    "onset": (
        "When does the problem occur?",
        [
            "All the time",
            "Occasionally",
            "Only during use",
            "After charging",
            "After a fall or impact",
            "Not sure",
        ],
    ),
    "possible_cause": (
        "Do you know what may have caused the problem?",
        [
            "Physical damage or impact",
            "Liquid exposure",
            "Power or charging issue",
            "Software or update issue",
            "Normal wear and tear",
            "Unknown",
        ],
    ),
    "previous_repair": (
        "Has the device been repaired before?",
        ["Yes", "No", "Not sure"],
    ),
    "hazards": (
        "Is there smoke, a burning smell, unusual heat, "
        "or a swollen battery?",
        ["Yes", "No", "Not sure"],
    ),
}


def normalize_device(device_type):
    """Convert a detected device label into a supported device name."""
    if not isinstance(device_type, str):
        raise ValueError("Detection result must include a device_type.")

    device = " ".join(
        device_type.strip().lower().replace("_", " ").split()
    )
    device = DEVICE_ALIASES.get(device, device)

    if device not in DEVICE_QUESTIONS:
        raise ValueError(
            f"Unsupported device detected: {device_type!r}"
        )

    return device


def ask_question(question, options):
    """Ask a question in the terminal and validate the selected option."""
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


def run_questionnaire(detection_result):
    """
    Receive detection results from the object-detection branch.

    Expected input:
    {
        "device_type": "laptop",
        "findings": ["screen crack"],
        "confidence": 0.91
    }
    """
    if not isinstance(detection_result, dict):
        raise ValueError("Detection result must be a dictionary.")

    device = normalize_device(detection_result.get("device_type"))

    findings = detection_result.get("findings", [])
    confidence = detection_result.get("confidence")

    if isinstance(findings, str):
        findings = [findings]

    if (
        not isinstance(findings, list)
        or not all(isinstance(item, str) for item in findings)
    ):
        raise ValueError("'findings' must be a list of strings.")

    if confidence is not None:
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            raise ValueError("Confidence must be between 0 and 1.")

        confidence = float(confidence)

    print("\n" + "=" * 45)
    print("       AI REPAIR ASSISTANT")
    print("=" * 45)
    print(f"Detected device: {device.title()}")
    print(f"Detected findings: {', '.join(findings) or 'None provided'}")

    if confidence is not None:
        print(f"Detection confidence: {confidence:.1%}")

    print("\nPlease answer the following questions.")

    answers = {}

    questions = {
        **DEVICE_QUESTIONS[device],
        **COMMON_QUESTIONS,
    }

    for question_id, (question, options) in questions.items():
        answers[question_id] = ask_question(question, options)

    safety_alert = answers["hazards"] == "Yes"

    result = {
        "status": "success",
        "module": "questionnaire",
        "device_type": device,
        "detected_findings": findings,
        "detection_confidence": confidence,
        "answers": answers,
        "safety_alert": safety_alert,
        "requires_professional_attention": safety_alert,
        "diagnosis": None,
        "repair_recommendation": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if safety_alert:
        print("\nSAFETY WARNING:")
        print(
            "Stop using or charging the device. "
            "Do not open a swollen battery. "
            "If smoke or fire is present, move away "
            "and seek emergency assistance."
        )

    print("\n" + "=" * 45)
    print("QUESTIONNAIRE RESULTS")
    print("=" * 45)
    print(json.dumps(result, indent=4))

    return result

