"""Integrated Repair Before Replace pipeline.

Run: python assistant.py --image "images/laptop.jpg"
Combines object detection, OpenCV visual heuristics, questionnaire diagnosis,
rule-based repair-vs-replace decision, and a plain-language explanation.
"""
from pathlib import Path
from datetime import datetime
import argparse, json, sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import detect_new
from questionaire import run_questionnaire, normalize_device
from fault_engine import engine, evaluate_safety, STOP_USING_ADVICE
import cv as cv_analysis
import verdict as verdict_engine
import gamma as explanation_engine

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "diagnoses"

CV_LEVEL_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
DEMO_DESCRIPTION = "laptop turns on but black screen, fan spins"
DEMO_SCRIPTED_ANSWERS = {
    "affected_component": "Screen / display",
    "power_state": "Lights/fan turn on but no display",
    "startup_signals": "No",
    "memory_storage": "Not sure / not applicable",
    "display": "Black screen",
    "temperature": "Not sure",
    "charging": "Works normally",
    "recent_event": "No known event",
    "hazards": "No",
    "duration": "Not sure",
}


def choose_device():
    supported = ["laptop"]
    print("\nSelect the computer device to diagnose:")
    for i, device in enumerate(supported, 1):
        print(f"  {i}. {device.title()}")
    choice = input("\nEnter a number or device name: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(supported):
        return supported[int(choice) - 1]
    return normalize_device(choice)


def merge_cv_results(cv_results):
    """Merge CV findings by id, keeping the highest level (none < low < medium < high)."""
    if not cv_results:
        return None
    merged_findings = {}
    merged = {
        "findings": [],
        "safety_flag": False,
        "quality_warnings": [],
        "follow_up_question": None,
        "annotated_image": None,
    }
    for cv in cv_results:
        if not cv:
            continue
        merged["safety_flag"] = merged["safety_flag"] or bool(cv.get("safety_flag"))
        for warning in cv.get("quality_warnings") or []:
            if warning not in merged["quality_warnings"]:
                merged["quality_warnings"].append(warning)
        if cv.get("follow_up_question"):
            merged["follow_up_question"] = cv["follow_up_question"]
        if cv.get("annotated_image"):
            merged["annotated_image"] = cv["annotated_image"]
        for finding in cv.get("findings") or []:
            fid = finding.get("id")
            prev = merged_findings.get(fid)
            if prev is None or CV_LEVEL_RANK.get(finding.get("level"), 0) > CV_LEVEL_RANK.get(prev.get("level"), 0):
                merged_findings[fid] = finding
    merged["findings"] = list(merged_findings.values())
    return merged


def _crop_box_from_detections(detections):
    if not detections:
        return None
    boxes = [d.get("box") for d in detections if d.get("box")]
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _prompt_optional(message, demo=False):
    if demo:
        return None
    try:
        value = input(message).strip()
    except EOFError:
        return None
    return value or None


def _parse_optional_float(raw):
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def make_detection_payload(image_path=None, device_type=None, detection_result=None):
    detections = (detection_result or {}).get("detections", [])
    components = [str(d.get("device", "")).strip() for d in detections]
    components = [item for item in components if item]
    confidence = max((float(d.get("confidence", 0)) for d in detections), default=None)
    return {
        "device_type": device_type or (detection_result or {}).get("device_type"),
        "findings": components,
        "confidence": confidence,
        "image_path": str(image_path) if image_path else None,
    }


def adapt_report_for_verdict(report):
    """Bridge current fault_engine schema to the older verdict.py input contract."""
    diagnosis = report.get("diagnosis") or {}
    recommendation = report.get("repair_recommendation") or {}
    primary = {
        "fault_title": diagnosis.get("primary_fault", "Unconfirmed possible fault"),
        "fault_id": diagnosis.get("fault_id", "UNKNOWN"),
        "confidence_score": diagnosis.get("confidence", 0.0),
        "severity": diagnosis.get("severity", "UNKNOWN"),
        "repair_complexity": recommendation.get("complexity", "Unknown"),
        "recommendation": recommendation.get("action", "Professional inspection recommended"),
        "recommendation_rationale": recommendation.get("rationale", ""),
        # This is currently a parts-cost estimate, not a verified full repair quote.
        "estimated_parts_cost": recommendation.get("estimated_cost"),
        "estimated_diy_time": recommendation.get("estimated_time", "Not estimated"),
        "safety_warnings": diagnosis.get("safety_instructions", []),
        "troubleshooting_steps": diagnosis.get("troubleshooting_steps", []),
        "required_tools": recommendation.get("required_tools", []),
        "supporting_evidence": diagnosis.get("supporting_evidence", []),
    }
    return {
        "primary_diagnosis": primary,
        "alternative_diagnoses": diagnosis.get("secondary_faults", []),
        "safety_alert": report.get("safety_alert", False),
        "requires_professional_attention": report.get("requires_professional_attention", False),
    }


def print_report(report):
    if report.get("status") in ("unsupported", "unsupported_device"):
        print("\n" + "=" * 72)
        print(f"DEVICE NOTICE: {str(report.get('device_type', 'unknown')).upper()} — NOT CURRENTLY SUPPORTED")
        print("=" * 72)
        print(f"Message: {report.get('message')}")
        print("\nGENERAL ELECTRICAL & HARDWARE SAFETY ADVICE:")
        for tip in report.get("safety_advice", []):
            print(f"  {tip}")
        print("=" * 72)
        return

    diagnosis = report.get("diagnosis") or {}
    recommendation = report.get("repair_recommendation") or {}
    decision = report.get("repair_decision") or {}
    visual = report.get("visual_analysis") or {}
    explanation = report.get("explanation") or {}
    print("\n" + "=" * 72)
    print(f"REPAIR ASSESSMENT: {report.get('device_type', 'unknown').upper()}")
    print("=" * 72)
    print(f"Possible fault       : {diagnosis.get('primary_fault', 'Not available')}")
    conf = diagnosis.get("confidence")
    if isinstance(conf, (int, float)):
        print(f"Diagnosis confidence : {conf:.0%}")
    print(f"Severity             : {diagnosis.get('severity', 'Not specified')}")
    print(f"Repair complexity    : {recommendation.get('complexity', 'Not specified')}")
    print(f"Fault-engine action  : {recommendation.get('action', 'Not available')}")
    print(f"Estimated parts cost : {str(recommendation.get('estimated_cost', 'Not available')).replace('₹', 'Rs ')}")
    print(f"Estimated time       : {recommendation.get('estimated_time', 'Not available')}")
    print("\nREPAIR VS REPLACE DECISION")
    print(f"  Verdict            : >>> {decision.get('verdict', 'Not available')} <<<")
    if decision.get("low_confidence"):
        print("  [!] LOW-CONFIDENCE NOTICE: Repair cost could not be parsed or diagnosis is inconclusive.")
    replacement = decision.get('replacement_cost')
    print(f"  Replacement price  : Rs {replacement:,.0f}" if replacement is not None else "  Replacement price  : Not available")
    if decision.get("estimated_saving") is not None:
        print(f"  Approx. potential saving: Rs {decision['estimated_saving']:,}")
    
    print("\nDECISION REASONS & ANALYSIS")
    for reason in decision.get("reasons", []):
        print(f"  - {reason}")
        
    factors = decision.get("decision_factors") or []
    if factors:
        print("\nDECISION FACTORS (Traceable Analysis)")
        for factor in factors:
            fname = factor.get('factor', '').replace('_', ' ').title()
            print(f"  * {fname:16s}: {factor.get('detail')}")

    assumptions = decision.get("assumptions") or []
    if assumptions:
        print("\nDECISION ASSUMPTIONS (User Overridable via --replacement-cost / --age)")
        for a in assumptions:
            print(f"  * {a}")

    ewaste = decision.get("e_waste") or report.get("e_waste")
    if ewaste:
        print("\nE-WASTE IMPACT (ESTIMATE)")
        if ewaste.get("status") == "avoided":
            print(f"  Status             : E-waste avoided")
            print(f"  Estimated mass saved: ~{ewaste.get('amount_kg')} kg (estimate)")
            print(f"  Basis / Assumption : {ewaste.get('assumption')}")
        elif ewaste.get("status") == "recycling_advice":
            print(f"  Status             : Whole-device replacement / recycling needed")
            print(f"  Basis / Assumption : {ewaste.get('assumption')}")
            print(f"  Take-back guidance : {ewaste.get('advice')}")

    if report.get("safety_alert") or decision.get("safety_hold"):
        print("\n" + "!" * 72)
        print("!!! SAFETY ALERT: Stop using the device and follow safety instructions. !!!")
        print("!" * 72)
    for warning in diagnosis.get("safety_instructions") or []:
        print(f"  SAFETY: {warning}")

    safe_checks = diagnosis.get("safe_checks") or []
    advanced_steps = diagnosis.get("advanced_steps") or []
    if safe_checks:
        print("\nSAFE CHECKS (Non-invasive, no opening the device)")
        for i, step in enumerate(safe_checks, 1):
            print(f"  {i}. {step}")
    if advanced_steps:
        print("\nADVANCED STEPS (Internal hardware / opening chassis required)")
        for i, step in enumerate(advanced_steps, 1):
            print(f"  {i}. {step}")
    elif diagnosis.get("troubleshooting_steps") and not safe_checks:
        print("\nSAFE TROUBLESHOOTING")
        for i, step in enumerate(diagnosis.get("troubleshooting_steps"), 1):
            print(f"  {i}. {step}")
    if visual:
        print("\nPHOTO ANALYSIS (heuristics, not a confirmed fault)")
        for item in visual.get("findings", []):
            print(f"  - {item.get('label')}: {item.get('level')} — {item.get('detail')}")
        for warning in visual.get("quality_warnings", []):
            print(f"  - Photo quality: {warning}")
        if visual.get("annotated_image"):
            print(f"  Annotated image: {visual['annotated_image']}")
    if explanation.get("text"):
        print("\nPLAIN-LANGUAGE SUMMARY")
        print(f"  {explanation['text']}")
        print(f"  Explanation source: {explanation.get('source', 'unknown')}")
    print("=" * 72)


def save_report(report, source_name="manual"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(source_name).stem or "diagnosis"
    output_path = OUTPUT_DIR / f"{stem}_{stamp}_diagnosis.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    return output_path


def run_pipeline(image_path=None, device_type=None, confidence=0.25, age_years=None,
                 replacement_cost=None, gemma_backend="auto", enable_second_opinion=False,
                 video_path=None, describe=None, demo=False):
    print("\n" + "=" * 72 + "\n REPAIR BEFORE REPLACE - INTEGRATED PIPELINE\n" + "=" * 72)
    if demo:
        describe = DEMO_DESCRIPTION
        gemma_backend = "template"
        image_path = None
        video_path = None
        enable_second_opinion = False

    source_image = Path(image_path).expanduser().resolve() if image_path else None
    if source_image and not source_image.is_file():
        raise FileNotFoundError(f"Image file not found: {source_image}")
    source_video = Path(video_path).expanduser().resolve() if video_path else None
    if source_video and not source_video.is_file():
        raise FileNotFoundError(f"Video file not found: {source_video}")

    detection_result = None
    cv_result = None
    if demo:
        print("\n[1/5] Demo mode: skipping image/video/network detection.")
        print("[2/5] Demo mode: skipping OpenCV photo analysis.")
    elif source_video:
        print(f"\n[1/5] Sampling frames from {source_video.name}...")
        try:
            detection_result = detect_new.detect_sampled_frames(str(source_video), conf=confidence)
        except Exception as exc:
            print(f"[WARNING] Object detection failed: {exc}")
            detection_result = {"detections": [], "frames": []}
        best_path = (detection_result or {}).get("image_path")
        if best_path:
            source_image = Path(best_path)
        print("[2/5] Analyzing visible damage on sampled frames with OpenCV...")
        cv_parts = []
        for frame in (detection_result or {}).get("frames") or []:
            frame_path = frame.get("image_path")
            if not frame_path:
                continue
            try:
                crop_box = _crop_box_from_detections(frame.get("detections") or [])
                cv_parts.append(cv_analysis.analyze_image(frame_path, crop_box=crop_box))
            except Exception as exc:
                print(f"[WARNING] OpenCV analysis failed: {exc}")
        cv_result = merge_cv_results(cv_parts)
    elif source_image:
        print(f"\n[1/5] Detecting device/components in {source_image.name}...")
        try:
            detection_result = detect_new.detect_image(str(source_image), conf=confidence)
        except Exception as exc:
            print(f"[WARNING] Object detection failed: {exc}\n          Will continue with manual device selection if needed.")
        print("[2/5] Analyzing visible damage and photo quality with OpenCV...")
        try:
            crop_box = _crop_box_from_detections((detection_result or {}).get("detections", []))
            cv_result = cv_analysis.analyze_image(str(source_image), crop_box=crop_box)
        except Exception as exc:
            print(f"[WARNING] OpenCV analysis failed: {exc}")
    else:
        print("\n[1/5] No image supplied; image-based checks are skipped.")

    payload = make_detection_payload(source_image, device_type, detection_result)
    if cv_result and cv_result.get("follow_up_question"):
        payload["cv_follow_up"] = cv_result["follow_up_question"]
    if cv_result and cv_result.get("safety_flag"):
        payload["safety_alert"] = True
    if not payload.get("device_type"):
        if demo or describe:
            payload["device_type"] = device_type or "laptop"
        elif payload["findings"]:
            payload["device_type"] = "laptop"
        else:
            payload["device_type"] = device_type or choose_device()
    if cv_result:
        for item in cv_result.get("findings", []):
            payload["findings"].append(
                f"OpenCV visual observation: {item['label']} ({item['level']}): {item['detail']}"
            )

    early_safety = evaluate_safety(free_text=describe)
    force_service = bool(early_safety["safety_alert"])
    if force_service:
        print("\n" + "!" * 60)
        print("CRITICAL SAFETY ALERT — HARD HAZARD DETECTED")
        print("!" * 60)
        for advice_line in STOP_USING_ADVICE:
            print(f"  {advice_line}")
        print("!" * 60)
        payload["safety_alert"] = True

    print("\n[3/5] Collecting symptoms and running diagnosis...")
    # When a hard hazard is found, skip prompts to produce report early and safely
    report = run_questionnaire(
        payload,
        image_path=str(source_image) if source_image else None,
        run_diagnosis=True,
        scripted_answers=DEMO_SCRIPTED_ANSWERS if demo else None,
        free_text=describe,
        skip_prompts=demo or force_service,
        use_gemini=False if (demo or force_service) else True,
    )

    if report.get("status") in ("unsupported", "unsupported_device"):
        print_report(report)
        saved = save_report(report, str(source_image or report.get("device_type", "unsupported")))
        print(f"\n[INFO] Full JSON report saved to: {saved}")
        return report

    print("[4/5] Calculating repair-vs-replace decision...")
    adapted = adapt_report_for_verdict(report)
    if force_service:
        adapted["safety_alert"] = True
        adapted["requires_professional_attention"] = True
    decision = verdict_engine.decide(
        device=report.get("device_type", payload["device_type"]),
        report=adapted, cv_result=cv_result,
        age_years=age_years, replacement_cost=replacement_cost,
    )
    if force_service:
        decision["verdict"] = "SERVICE (technician)"
        decision["safety_hold"] = True
    report["repair_decision"] = decision
    report["visual_analysis"] = cv_result
    report["input_image"] = str(source_image) if source_image else None
    report["decision_inputs"] = {"device_age_years": age_years,
                                  "replacement_cost_override_inr": replacement_cost}

    print("[5/5] Creating plain-language explanation...")
    diagnosis = report.get("diagnosis") or {}
    context = {
        "device": report.get("device_type"),
        "symptoms": report.get("answers", {}),
        "visual_findings": (cv_result or {}).get("findings", []),
        "diagnosis": {"fault_title": diagnosis.get("primary_fault", "Unconfirmed possible fault"),
                      "severity": diagnosis.get("severity", "Unknown"),
                      "supporting_evidence": diagnosis.get("supporting_evidence", [])},
        "verdict": decision,
        "e_waste": decision.get("e_waste"),
    }
    report["explanation"] = explanation_engine.explain(context, backend=gemma_backend)
    if enable_second_opinion and source_image:
        report["visual_second_opinion"] = explanation_engine.second_opinion(str(source_image), backend=gemma_backend)

    print_report(report)
    saved = save_report(report, str(source_image or report.get("device_type", "manual")))
    print(f"\n[INFO] Full JSON report saved to: {saved}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Integrated Repair Before Replace assistant")
    parser.add_argument("--image", "-i", help="Path to a device image (optional)")
    parser.add_argument("--video", help="Path to a device video (optional)")
    parser.add_argument("--describe", help="Free-text symptom description (optional)")
    parser.add_argument("--demo", action="store_true", help="Offline non-interactive sample run")
    parser.add_argument("--device", "-d", help="Computer device type (currently laptop/PC only)")
    parser.add_argument("--confidence", type=float, default=0.25, help="Object-detection confidence threshold")
    parser.add_argument("--age", type=float, help="Device age in years")
    parser.add_argument("--replacement-cost", type=float, help="Replacement price in INR")
    parser.add_argument("--gemma-backend", choices=["auto", "ollama", "api", "template"], default="auto")
    parser.add_argument("--second-opinion", action="store_true", help="Add a separate Gemma visual description")
    args = parser.parse_args()

    describe = args.describe
    age_years = args.age
    replacement_cost = args.replacement_cost
    if not args.demo:
        if not args.image and not args.video and describe is None:
            describe = _prompt_optional("Optional symptom description (Enter to skip): ", demo=False)
        if age_years is None:
            age_years = _parse_optional_float(
                _prompt_optional("Device age in years (Enter to skip): ", demo=False)
            )
        if replacement_cost is None:
            replacement_cost = _parse_optional_float(
                _prompt_optional("Replacement price in INR (Enter to skip): ", demo=False)
            )

    try:
        run_pipeline(
            args.image, args.device, args.confidence, age_years, replacement_cost,
            args.gemma_backend, args.second_opinion,
            video_path=args.video, describe=describe, demo=args.demo,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
