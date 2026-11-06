"""
Evidence-Based Root Cause Analysis Engine for Repair Before Replace.

Identifies and ranks possible hardware faults using multiple independent evidence sources:
- Hardware inventory & detection results (presence != failure)
- Visual inspection observations (OpenCV / Roboflow)
- Imported hardware specifications (data-source provenance)
- Initial problem description & symptoms
- Adaptive questionnaire answers
- Safe troubleshooting test results

Maintains strict separation between observed evidence, user-reported symptoms,
imported specifications, confirmed tests, and AI interpretations.
Detects contradictions between visual findings and user reports.
Produces categorical confidence ratings without fabricating percentages.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional, Set, Tuple
import re


class EvidenceSource(str, Enum):
    OBSERVED = "observed"                      # Camera, OpenCV heuristic, visual bounding box
    USER_REPORTED = "user_reported"            # Text prompt, questionnaire choice, manual response
    IMPORTED_SPECS = "imported_specs"          # Hardware diagnostic report (dxdiag, systeminfo, etc.)
    CONFIRMED_TEST = "confirmed_test"          # Safe check output (HDMI external monitor, LED code)
    AI_INTERPRETATION = "ai_interpretation"    # Gemini / LLM explanation/categorization


class EvidencePolarity(str, Enum):
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    NEUTRAL = "neutral"


class ConfidenceCategory(str, Enum):
    STRONGLY_SUPPORTED = "Strongly supported"
    POSSIBLE = "Possible"
    WEAKLY_SUPPORTED = "Weakly supported"
    INCONCLUSIVE = "Inconclusive"


@dataclass
class EvidenceItem:
    evidence_id: str
    description: str
    source: str                           # EvidenceSource value
    related_component: str                # e.g. "ram", "gpu", "display", "fan", "motherboard", "power"
    evidence_type: str                    # "visual", "symptom", "spec", "test_result", "heuristic"
    confidence_reliability: float         # 0.0 to 1.0 (source reliability weight)
    verification_status: str              # "verified", "unverified", "contradicted"
    polarity: str                         # EvidencePolarity value
    reference_id: Optional[str] = None    # e.g. question ID, image file, frame number, spec field

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceContradiction:
    contradiction_id: str
    component: str
    evidence_a: Dict[str, Any]
    evidence_b: Dict[str, Any]
    explanation: str
    clarification_question: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FaultHypothesis:
    fault_id: str
    fault_name: str
    confidence_category: str              # ConfidenceCategory value
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    contradicting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    recommended_next_check: str = ""
    severity: str = "MEDIUM"
    repair_complexity: str = "Moderate"
    recommendation: str = "SERVICE"
    estimated_cost: str = "Unknown"
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceExtractor:
    """
    Extracts structured EvidenceItem instances from heterogeneous inputs:
    - User symptoms / free text
    - Questionnaire answers
    - Hardware inventory (visible components != failed components)
    - Visual inspection findings (cv.py observations)
    - Imported specifications
    - Safe test results
    """

    @staticmethod
    def extract_from_user_input(
        free_text: Optional[str],
        answers: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceItem]:
        items: List[EvidenceItem] = []
        counter = 1

        if free_text and str(free_text).strip():
            text = str(free_text).strip()
            text_lower = text.lower()
            
            # Map obvious symptoms in text to component evidence
            component_maps = [
                (["screen", "display", "monitor", "blank", "black screen", "flicker", "lines"], "display", "Display / screen visual symptom reported in problem description"),
                (["fan", "spinning", "noise", "whine", "rattling", "airflow"], "cooling_fan", "Fan behavior symptom reported in problem description"),
                (["ram", "memory", "beep", "dram", "bsod", "blue screen"], "ram", "Memory/RAM symptom reported in problem description"),
                (["battery", "charging", "charger", "plug", "adapter", "drain"], "battery", "Battery / power charging symptom reported in problem description"),
                (["gpu", "graphics", "card", "artifact", "vga", "hdmi"], "gpu", "GPU / graphics display symptom reported in problem description"),
                (["power", "won't turn on", "dead", "no lights", "shuts down", "power button"], "power_delivery", "Power startup symptom reported in problem description"),
                (["smoke", "burn", "spark", "smell", "pop"], "motherboard", "Critical thermal/electrical hazard symptom reported in problem description"),
                (["storage", "ssd", "hdd", "boot device", "no boot", "drive"], "storage", "Storage / boot device symptom reported in problem description"),
            ]

            found_any = False
            for keywords, comp, desc in component_maps:
                if any(kw in text_lower for kw in keywords):
                    found_any = True
                    items.append(EvidenceItem(
                        evidence_id=f"EV-USR-TXT-{counter}",
                        description=f"{desc}: \"{text[:80]}{'...' if len(text) > 80 else ''}\"",
                        source=EvidenceSource.USER_REPORTED.value,
                        related_component=comp,
                        evidence_type="symptom",
                        confidence_reliability=0.75,
                        verification_status="unverified",
                        polarity=EvidencePolarity.SUPPORTING.value,
                        reference_id="problem_description"
                    ))
                    counter += 1

            if not found_any:
                items.append(EvidenceItem(
                    evidence_id=f"EV-USR-TXT-{counter}",
                    description=f"User stated problem: \"{text[:100]}\"",
                    source=EvidenceSource.USER_REPORTED.value,
                    related_component="system",
                    evidence_type="symptom",
                    confidence_reliability=0.70,
                    verification_status="unverified",
                    polarity=EvidencePolarity.NEUTRAL.value,
                    reference_id="problem_description"
                ))

        if answers:
            neutral_tokens = {
                "", "no", "none", "not tested", "normal", "works normally",
                "normal temperature", "no known event", "unknown", "n/a",
                "none of the above", "not sure", "not sure / not applicable",
                "starts normally"
            }
            q_num = 1
            for q_id, ans in answers.items():
                val = str(ans).strip()
                if not val or val.lower() in neutral_tokens:
                    continue

                val_lower = val.lower()
                related = "system"
                if any(k in val_lower for k in ["screen", "display", "black", "lines", "faint"]):
                    related = "display"
                elif any(k in val_lower for k in ["fan", "spin", "noisy", "whine"]):
                    related = "cooling_fan"
                elif any(k in val_lower for k in ["battery", "charge", "adapter", "plug"]):
                    related = "battery"
                elif any(k in val_lower for k in ["ram", "beep", "dram", "freeze"]):
                    related = "ram"
                elif any(k in val_lower for k in ["gpu", "graphic", "artifact", "vga", "pcie"]):
                    related = "gpu"
                elif any(k in val_lower for k in ["heat", "hot", "thermal", "overheat"]):
                    related = "cooling"
                elif any(k in val_lower for k in ["smoke", "smell", "spark", "liquid", "spill", "pop"]):
                    related = "motherboard"
                elif any(k in val_lower for k in ["power", "dead", "unresponsive", "shut"]):
                    related = "power_delivery"
                elif any(k in val_lower for k in ["drive", "disk", "ssd", "hdd", "bootable"]):
                    related = "storage"

                items.append(EvidenceItem(
                    evidence_id=f"EV-USR-ANS-{q_num}",
                    description=f"Questionnaire response [{q_id}]: {val}",
                    source=EvidenceSource.USER_REPORTED.value,
                    related_component=related,
                    evidence_type="symptom",
                    confidence_reliability=0.85,
                    verification_status="unverified",
                    polarity=EvidencePolarity.SUPPORTING.value,
                    reference_id=str(q_id)
                ))
                q_num += 1

        return items

    @staticmethod
    def extract_from_hardware_profile(
        hardware_profile: Optional[Dict[str, Any]]
    ) -> List[EvidenceItem]:
        """
        IMPORTANT: Component presence != component failure!
        Detecting RAM does not mean RAM is defective.
        Detecting a GPU means GPU-related tests are applicable, but does NOT make GPU faulty.
        """
        items: List[EvidenceItem] = []
        if not hardware_profile:
            return items

        comps = hardware_profile.get("components", {})
        count = 1
        for comp_key, comp_info in comps.items():
            vis = comp_info.get("visibility_status")
            conf = comp_info.get("confidence")
            cname = comp_info.get("component_name", comp_key)
            
            if vis in ("DETECTED", "PARTIALLY_VISIBLE"):
                # Clearly mark as NEUTRAL presence evidence
                items.append(EvidenceItem(
                    evidence_id=f"EV-HW-PRES-{count}",
                    description=f"Component present & visible in image: {cname} (Visibility: {vis}, Conf: {conf if conf is not None else 'N/A'})",
                    source=EvidenceSource.OBSERVED.value,
                    related_component=comp_key,
                    evidence_type="visual",
                    confidence_reliability=float(conf) if conf is not None else 0.80,
                    verification_status="verified",
                    polarity=EvidencePolarity.NEUTRAL.value,
                    reference_id=f"hardware_profile.{comp_key}"
                ))
                count += 1
            elif vis == "UNCERTAIN":
                items.append(EvidenceItem(
                    evidence_id=f"EV-HW-UNCERT-{count}",
                    description=f"Component identity uncertain: {cname} (Confidence: {conf if conf is not None else 'Low'})",
                    source=EvidenceSource.OBSERVED.value,
                    related_component=comp_key,
                    evidence_type="visual",
                    confidence_reliability=float(conf) if conf is not None else 0.40,
                    verification_status="unverified",
                    polarity=EvidencePolarity.NEUTRAL.value,
                    reference_id=f"hardware_profile.{comp_key}"
                ))
                count += 1

        cabinet = hardware_profile.get("cabinet_status")
        if cabinet:
            items.append(EvidenceItem(
                evidence_id=f"EV-HW-CAB-{count}",
                description=f"Computer chassis/cabinet orientation: {cabinet}",
                source=EvidenceSource.OBSERVED.value,
                related_component="chassis",
                evidence_type="visual",
                confidence_reliability=0.85,
                verification_status="verified",
                polarity=EvidencePolarity.NEUTRAL.value,
                reference_id="hardware_profile.cabinet_status"
            ))

        return items

    @staticmethod
    def extract_from_cv_analysis(
        cv_result: Optional[Dict[str, Any]]
    ) -> List[EvidenceItem]:
        """
        Extracts observations from OpenCV/visual heuristics.
        Applies safety calibration: a dark image region does not prove motherboard burn!
        """
        items: List[EvidenceItem] = []
        if not cv_result:
            return items

        findings = cv_result.get("findings", [])
        raw_cv = cv_result.get("raw_cv", {})

        count = 1
        for f in findings:
            fid = f.get("id", "")
            flevel = f.get("level", "none")
            fdetail = f.get("detail", "")
            if flevel == "none":
                continue

            # Check false-positive guard: dark patch != burn
            if fid == "burn":
                # Only treat as burn evidence if level is medium/high and not a mere dark chassis area
                polarity = EvidencePolarity.SUPPORTING.value if flevel in ("medium", "high") else EvidencePolarity.NEUTRAL.value
                reliability = 0.50 if flevel == "low" else 0.70  # Visual burns require physical verification
                desc = f"Visual heuristic flagged possible discoloration / dark patch ({flevel} intensity: {fdetail}). Requires physical confirmation before claiming electrical burn."
                items.append(EvidenceItem(
                    evidence_id=f"EV-CV-{count}",
                    description=desc,
                    source=EvidenceSource.OBSERVED.value,
                    related_component="motherboard",
                    evidence_type="heuristic",
                    confidence_reliability=reliability,
                    verification_status="unverified",
                    polarity=polarity,
                    reference_id=f"cv.{fid}"
                ))
            elif fid == "dust":
                items.append(EvidenceItem(
                    evidence_id=f"EV-CV-{count}",
                    description=f"Visible dust accumulation ({flevel} level: {fdetail})",
                    source=EvidenceSource.OBSERVED.value,
                    related_component="cooling_fan",
                    evidence_type="heuristic",
                    confidence_reliability=0.75,
                    verification_status="verified" if flevel in ("medium", "high") else "unverified",
                    polarity=EvidencePolarity.SUPPORTING.value,
                    reference_id=f"cv.{fid}"
                ))
            elif fid == "bulge":
                items.append(EvidenceItem(
                    evidence_id=f"EV-CV-{count}",
                    description=f"Chassis / component bulge detected ({flevel} level: {fdetail})",
                    source=EvidenceSource.OBSERVED.value,
                    related_component="battery",
                    evidence_type="heuristic",
                    confidence_reliability=0.80,
                    verification_status="unverified",
                    polarity=EvidencePolarity.SUPPORTING.value,
                    reference_id=f"cv.{fid}"
                ))
            elif fid == "fan_motion":
                # Video or motion observation of fan
                state = "spinning" if flevel in ("low", "medium", "high") else "stopped"
                items.append(EvidenceItem(
                    evidence_id=f"EV-CV-{count}",
                    description=f"Fan motion observation: {state} ({fdetail})",
                    source=EvidenceSource.OBSERVED.value,
                    related_component="cooling_fan",
                    evidence_type="visual",
                    confidence_reliability=0.85,
                    verification_status="verified",
                    polarity=EvidencePolarity.NEUTRAL.value,
                    reference_id=f"cv.{fid}"
                ))
            else:
                items.append(EvidenceItem(
                    evidence_id=f"EV-CV-{count}",
                    description=f"Visual heuristic observation [{fid}] ({flevel}): {fdetail}",
                    source=EvidenceSource.OBSERVED.value,
                    related_component="system",
                    evidence_type="heuristic",
                    confidence_reliability=0.65,
                    verification_status="unverified",
                    polarity=EvidencePolarity.SUPPORTING.value,
                    reference_id=f"cv.{fid}"
                ))
            count += 1

        return items

    @staticmethod
    def extract_from_specs(
        imported_specs: Optional[Dict[str, Any]]
    ) -> List[EvidenceItem]:
        """Extracts provenance-tracked hardware specification facts."""
        items: List[EvidenceItem] = []
        if not imported_specs:
            return items

        fmt = imported_specs.get("source_format", "unknown")
        count = 1

        if imported_specs.get("ram_total_gb") is not None:
            cfg = f" ({imported_specs.get('ram_configuration')})" if imported_specs.get("ram_configuration") else ""
            items.append(EvidenceItem(
                evidence_id=f"EV-SPEC-{count}",
                description=f"Imported specification confirmed: {imported_specs['ram_total_gb']} GB RAM installed{cfg}",
                source=EvidenceSource.IMPORTED_SPECS.value,
                related_component="ram",
                evidence_type="spec",
                confidence_reliability=0.95,
                verification_status="verified",
                polarity=EvidencePolarity.NEUTRAL.value,
                reference_id=f"specs.ram ({fmt})"
            ))
            count += 1

        if imported_specs.get("gpu_model"):
            items.append(EvidenceItem(
                evidence_id=f"EV-SPEC-{count}",
                description=f"Imported specification confirmed GPU: {imported_specs['gpu_model']}",
                source=EvidenceSource.IMPORTED_SPECS.value,
                related_component="gpu",
                evidence_type="spec",
                confidence_reliability=0.95,
                verification_status="verified",
                polarity=EvidencePolarity.NEUTRAL.value,
                reference_id=f"specs.gpu ({fmt})"
            ))
            count += 1

        if imported_specs.get("cpu_model"):
            items.append(EvidenceItem(
                evidence_id=f"EV-SPEC-{count}",
                description=f"Imported specification confirmed CPU: {imported_specs['cpu_model']}",
                source=EvidenceSource.IMPORTED_SPECS.value,
                related_component="cpu",
                evidence_type="spec",
                confidence_reliability=0.95,
                verification_status="verified",
                polarity=EvidencePolarity.NEUTRAL.value,
                reference_id=f"specs.cpu ({fmt})"
            ))
            count += 1

        if imported_specs.get("storage_summary") or imported_specs.get("storage_devices"):
            s_sum = imported_specs.get("storage_summary") or str(imported_specs.get("storage_devices"))
            items.append(EvidenceItem(
                evidence_id=f"EV-SPEC-{count}",
                description=f"Imported specification confirmed Storage: {s_sum}",
                source=EvidenceSource.IMPORTED_SPECS.value,
                related_component="storage",
                evidence_type="spec",
                confidence_reliability=0.95,
                verification_status="verified",
                polarity=EvidencePolarity.NEUTRAL.value,
                reference_id=f"specs.storage ({fmt})"
            ))
            count += 1

        return items

    @staticmethod
    def extract_from_test_results(
        test_results: Optional[Dict[str, Any]]
    ) -> List[EvidenceItem]:
        """Extracts results from confirmed safe troubleshooting checks (e.g. HDMI test, beep code)."""
        items: List[EvidenceItem] = []
        if not test_results:
            return items

        count = 1
        for test_key, test_val in test_results.items():
            val_str = str(test_val).strip()
            if not val_str:
                continue

            comp = "system"
            polarity = EvidencePolarity.SUPPORTING.value
            if "hdmi" in test_key.lower() or "external" in test_key.lower():
                comp = "display"
                if "works" in val_str.lower() or "normal" in val_str.lower():
                    # If external screen works, display panel/cable is faulty, GPU is NOT faulty!
                    polarity = EvidencePolarity.SUPPORTING.value
                elif "no display" in val_str.lower() or "black" in val_str.lower():
                    # External screen also black -> points to GPU/motherboard, contradicts sole panel fault
                    polarity = EvidencePolarity.CONTRADICTING.value

            items.append(EvidenceItem(
                evidence_id=f"EV-TEST-{count}",
                description=f"Confirmed troubleshooting check [{test_key}]: {val_str}",
                source=EvidenceSource.CONFIRMED_TEST.value,
                related_component=comp,
                evidence_type="test_result",
                confidence_reliability=0.90,
                verification_status="verified",
                polarity=polarity,
                reference_id=f"test.{test_key}"
            ))
            count += 1

        return items


class EvidenceReasoner:
    """
    Analyzes evidence bundles to:
    1. Detect contradictions between independent sources (e.g., visual vs user-reported).
    2. Rank fault hypotheses based on supporting, contradicting, and missing evidence.
    3. Categorize confidence into strongly supported, possible, weakly supported, or inconclusive.
    4. Guard against false positives (presence != fault, dark area != burn).
    """

    @staticmethod
    def detect_contradictions(evidence_items: List[EvidenceItem]) -> List[EvidenceContradiction]:
        """
        Detects contradictory evidence between different sources:
        e.g., Visual analysis observes fan spinning, but user reported fan not spinning.
        e.g., User reported no display, but visual analysis detected active glow / display on.
        """
        contradictions: List[EvidenceContradiction] = []
        c_count = 1

        # Separate items by component and source
        by_comp: Dict[str, List[EvidenceItem]] = {}
        for item in evidence_items:
            comp = item.related_component
            by_comp.setdefault(comp, []).append(item)

        # Check 1: Fan spinning vs Fan stopped
        fan_items = by_comp.get("cooling_fan", [])
        obs_spinning = [i for i in fan_items if i.source == EvidenceSource.OBSERVED.value and "spinning" in i.description.lower()]
        usr_not_spinning = [i for i in fan_items if i.source == EvidenceSource.USER_REPORTED.value and any(k in i.description.lower() for k in ["not spin", "fan stop", "no fan", "dead fan", "doesn't spin"])]

        if obs_spinning and usr_not_spinning:
            ev_a = obs_spinning[0]
            ev_b = usr_not_spinning[0]
            contradictions.append(EvidenceContradiction(
                contradiction_id=f"CONTRA-{c_count}",
                component="cooling_fan",
                evidence_a=ev_a.to_dict(),
                evidence_b=ev_b.to_dict(),
                explanation="Visual motion analysis observes the fan spinning, whereas user symptoms report the fan is stopped or not spinning.",
                clarification_question="Can you physically hear or see the internal cooling fan spinning when the power button is pressed?"
            ))
            c_count += 1

        # Check 2: Visual burn detected vs user report of no burns/no heat
        mobo_items = by_comp.get("motherboard", [])
        obs_burn = [i for i in mobo_items if i.source == EvidenceSource.OBSERVED.value and "discoloration" in i.description.lower()]
        usr_no_burn = [i for i in mobo_items if i.source == EvidenceSource.USER_REPORTED.value and any(k in i.description.lower() for k in ["no smoke", "no smell", "no heat", "no burn"])]

        if obs_burn and usr_no_burn:
            ev_a = obs_burn[0]
            ev_b = usr_no_burn[0]
            contradictions.append(EvidenceContradiction(
                contradiction_id=f"CONTRA-{c_count}",
                component="motherboard",
                evidence_a=ev_a.to_dict(),
                evidence_b=ev_b.to_dict(),
                explanation="Visual heuristic flagged possible component discoloration, but no smoke, smell, or burn symptoms were noted.",
                clarification_question="Inspect the highlighted area closely with good lighting: does it look like genuine electrical scorch marks, or is it normal dark PCB shading/dust?"
            ))
            c_count += 1

        # Check 3: External monitor works vs user thinks GPU/motherboard is dead
        disp_items = by_comp.get("display", [])
        test_hdmi_good = [i for i in disp_items if i.source == EvidenceSource.CONFIRMED_TEST.value and ("works" in i.description.lower() or "normal" in i.description.lower())]
        usr_dead = [i for i in evidence_items if i.source == EvidenceSource.USER_REPORTED.value and any(k in i.description.lower() for k in ["motherboard dead", "gpu dead", "completely broken"])]

        if test_hdmi_good and usr_dead:
            ev_a = test_hdmi_good[0]
            ev_b = usr_dead[0]
            contradictions.append(EvidenceContradiction(
                contradiction_id=f"CONTRA-{c_count}",
                component="display",
                evidence_a=ev_a.to_dict(),
                evidence_b=ev_b.to_dict(),
                explanation="Confirmed external monitor test outputs normal video, which contradicts suspicion of complete motherboard or GPU death.",
                clarification_question="Since the external screen displays Windows/BIOS normally, the core CPU, GPU, and motherboard are functioning. The issue is isolated to the laptop screen, eDP cable, or backlight."
            ))
            c_count += 1

        return contradictions

    @staticmethod
    def evaluate_hypotheses(
        rules: List[Any],
        evidence_items: List[EvidenceItem],
        contradictions: List[EvidenceContradiction],
        device_type: str = "laptop"
    ) -> List[FaultHypothesis]:
        """
        Ranks possible fault hypotheses across the verified FaultRule registry.
        For each hypothesis, determines:
        - Supporting evidence
        - Contradicting evidence
        - Missing evidence
        - Confidence category (Strongly supported, Possible, Weakly supported, Inconclusive)
        - Recommended next safe check
        """
        hypotheses: List[FaultHypothesis] = []
        contradicted_components = {c.component for c in contradictions}

        for rule in rules:
            supporting: List[Dict[str, Any]] = []
            contradicting: List[Dict[str, Any]] = []
            missing: List[str] = []

            # Determine relevant components for this rule
            category_name = getattr(rule, "category", "")
            rule_tokens = set(re.findall(r"\w+", f"{rule.fault_id} {rule.title} {category_name}".lower()))
            
            # 1. Match evidence items
            for item in evidence_items:
                # Presence evidence is NEUTRAL - presence != failure
                if item.polarity == EvidencePolarity.NEUTRAL.value:
                    continue

                item_text = f"{item.description} {item.related_component}".lower()
                
                # Check for match against rule's primary or secondary symptoms
                matched_symptom = False
                for sym in getattr(rule, "common_symptoms", []):
                    # Check token overlap
                    sym_tokens = [t for t in re.findall(r"\w+", sym.lower()) if len(t) > 3]
                    if any(t in item_text for t in sym_tokens):
                        matched_symptom = True
                        break

                # Also match against trigger keywords
                matched_trigger = False
                for trig in getattr(rule, "trigger_keywords", []):
                    if trig.lower() in item_text:
                        matched_trigger = True
                        break

                # Contradiction check: if HDMI works normally, rule about GPU or motherboard death is contradicted
                if "test.hdmi" in (item.reference_id or "") and "works" in item.description.lower():
                    if rule.fault_id in ("laptop_gpu_artifact", "desktop_gpu_vga_led", "laptop_dead_no_power", "desktop_dead_no_power"):
                        contradicting.append(item.to_dict())
                        continue

                # Component contradiction
                if item.related_component in contradicted_components and rule.fault_id.startswith(item.related_component):
                    contradicting.append(item.to_dict())
                    continue

                if matched_symptom or matched_trigger:
                    if item.polarity == EvidencePolarity.CONTRADICTING.value:
                        contradicting.append(item.to_dict())
                    else:
                        supporting.append(item.to_dict())

            # 2. Check missing evidence: What key verification is missing for this fault?
            safe_checks = getattr(rule, "safe_checks", [])
            fid_lower = f"{rule.fault_id} {rule.title}".lower()
            if "ram" in fid_lower:
                if not any("ram" in i.related_component and i.source == EvidenceSource.CONFIRMED_TEST.value for i in evidence_items):
                    missing.append("Physical single-stick RAM test or contact eraser cleaning not yet performed")
            elif "display" in rule.fault_id or "screen" in rule.fault_id:
                if not any("display" in i.related_component and i.source == EvidenceSource.CONFIRMED_TEST.value for i in evidence_items):
                    missing.append("External monitor HDMI check not confirmed")
            elif "fan" in rule.fault_id or "thermal" in rule.fault_id or "overheat" in rule.fault_id:
                if not any("cooling" in i.related_component and i.source == EvidenceSource.CONFIRMED_TEST.value for i in evidence_items):
                    missing.append("Internal temperature reading under load or airflow check not confirmed")
            elif "gpu" in rule.fault_id:
                if not any("gpu" in i.related_component and i.source == EvidenceSource.CONFIRMED_TEST.value for i in evidence_items):
                    missing.append("PCIe power connector check / diagnostic VGA LED status not verified")

            # 3. Compute score & confidence category
            sup_count = len(supporting)
            contra_count = len(contradicting)
            
            # Weighted support score
            sup_score = sum(i.get("confidence_reliability", 0.7) for i in supporting)
            score = max(0.0, sup_score - (contra_count * 1.5))

            if contra_count > 0 and sup_count <= contra_count:
                category = ConfidenceCategory.INCONCLUSIVE.value
            elif sup_count >= 2 and contra_count == 0 and sup_score >= 1.5:
                category = ConfidenceCategory.STRONGLY_SUPPORTED.value
            elif sup_count >= 1 and sup_score >= 0.7 and contra_count == 0:
                category = ConfidenceCategory.POSSIBLE.value
            elif sup_count >= 1:
                category = ConfidenceCategory.WEAKLY_SUPPORTED.value
            else:
                category = ConfidenceCategory.INCONCLUSIVE.value

            # Top recommended next safe check
            rec_check = safe_checks[0] if safe_checks else "Perform safe non-invasive inspection of exterior connections and power status."

            hypotheses.append(FaultHypothesis(
                fault_id=rule.fault_id,
                fault_name=rule.title,
                confidence_category=category,
                supporting_evidence=supporting,
                contradicting_evidence=contradicting,
                missing_evidence=missing,
                recommended_next_check=rec_check,
                severity=getattr(rule, "severity", "MEDIUM"),
                repair_complexity=getattr(rule, "complexity", "Moderate"),
                recommendation=getattr(rule, "recommendation", "SERVICE"),
                estimated_cost=getattr(rule, "estimated_cost", "Unknown"),
                score=score
            ))

        # Sort hypotheses by score descending
        hypotheses.sort(key=lambda h: h.score, reverse=True)
        return hypotheses


class EvidenceDiagnosisEngine:
    """
    Unified facade combining evidence extraction, contradiction detection,
    and root-cause hypothesis ranking.
    """

    def __init__(self, fault_rules: Optional[Dict[str, List[Any]]] = None):
        self.rules = fault_rules or {}

    def analyze_evidence(
        self,
        device_type: str,
        rules: List[Any],
        free_text: Optional[str] = None,
        answers: Optional[Dict[str, Any]] = None,
        hardware_profile: Optional[Dict[str, Any]] = None,
        cv_result: Optional[Dict[str, Any]] = None,
        imported_specs: Optional[Dict[str, Any]] = None,
        test_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Runs complete evidence-based root-cause analysis pipeline.
        Returns all gathered evidence items, detected contradictions,
        and ranked fault hypotheses.
        """
        evidence_items: List[EvidenceItem] = []

        # 1. Extract from all sources
        evidence_items.extend(EvidenceExtractor.extract_from_user_input(free_text, answers))
        evidence_items.extend(EvidenceExtractor.extract_from_hardware_profile(hardware_profile))
        evidence_items.extend(EvidenceExtractor.extract_from_cv_analysis(cv_result))
        evidence_items.extend(EvidenceExtractor.extract_from_specs(imported_specs))
        evidence_items.extend(EvidenceExtractor.extract_from_test_results(test_results))

        # 2. Contradiction Detection
        contradictions = EvidenceReasoner.detect_contradictions(evidence_items)

        # 3. Root Cause Hypothesis Ranking
        hypotheses = EvidenceReasoner.evaluate_hypotheses(
            rules=rules,
            evidence_items=evidence_items,
            contradictions=contradictions,
            device_type=device_type
        )

        # 4. Filter top hypotheses
        valid_hypotheses = [h for h in hypotheses if h.score > 0.0]
        top_hypotheses = valid_hypotheses[:4] if valid_hypotheses else hypotheses[:1]

        # Overall diagnosis determination
        primary_hypothesis = top_hypotheses[0] if top_hypotheses else None
        
        is_inconclusive = (
            primary_hypothesis is None or 
            primary_hypothesis.score <= 0.0 or 
            primary_hypothesis.confidence_category == ConfidenceCategory.INCONCLUSIVE.value
        )

        return {
            "evidence_items": [e.to_dict() for e in evidence_items],
            "contradictions": [c.to_dict() for c in contradictions],
            "has_contradictions": len(contradictions) > 0,
            "ranked_hypotheses": [h.to_dict() for h in top_hypotheses],
            "primary_hypothesis": primary_hypothesis.to_dict() if primary_hypothesis else None,
            "is_inconclusive": is_inconclusive,
            "recommended_next_check": primary_hypothesis.recommended_next_check if primary_hypothesis else "Physical technician bench inspection recommended.",
            "evidence_counts": {
                "total": len(evidence_items),
                "observed": len([e for e in evidence_items if e.source == EvidenceSource.OBSERVED.value]),
                "user_reported": len([e for e in evidence_items if e.source == EvidenceSource.USER_REPORTED.value]),
                "imported_specs": len([e for e in evidence_items if e.source == EvidenceSource.IMPORTED_SPECS.value]),
                "confirmed_tests": len([e for e in evidence_items if e.source == EvidenceSource.CONFIRMED_TEST.value]),
            }
        }
