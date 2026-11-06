"""
Unified Hardware Inventory and Intelligence System for Repair Before Replace.
Supports distinct profiles for both Laptops and Desktop PCs.

Defines component models, visibility statuses, profile aggregation logic,
and hardware specification report merging with strict data-source separation.

Rule: "Do not assume NOT_DETECTED means NOT_PRESENT."
Rule: "A closed opaque cabinet must not be treated as proof of its internal components."
Rule: "The system must not claim that the photograph itself established specs that came from reports."
"""

from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import json


class VisibilityStatus(str, Enum):
    DETECTED = "DETECTED"
    PARTIALLY_VISIBLE = "PARTIALLY_VISIBLE"
    UNCERTAIN = "UNCERTAIN"
    NOT_DETECTED = "NOT_DETECTED"
    USER_CONFIRMED = "USER_CONFIRMED"
    IMPORTED_FROM_REPORT = "IMPORTED_FROM_REPORT"


class CabinetStatus(str, Enum):
    OPEN_INTERIOR = "OPEN_INTERIOR"
    CLOSED_CABINET = "CLOSED_CABINET"
    PARTIAL_VIEW = "PARTIAL_VIEW"
    LAPTOP_OPEN = "LAPTOP_OPEN"
    LAPTOP_TEARDOWN = "LAPTOP_TEARDOWN"
    UNKNOWN = "UNKNOWN"


# Standard component definitions for Laptop and Desktop profiles
LAPTOP_STANDARD_COMPONENTS: Dict[str, str] = {
    "motherboard": "Motherboard / Mainboard",
    "ram": "RAM (SODIMM Memory)",
    "storage": "Storage Drive (SSD/HDD)",
    "battery": "Internal Li-ion Battery",
    "cpu_fan": "CPU Cooling Fan",
    "heatsink": "Copper Thermal Heatsink",
    "cpu": "Central Processing Unit (CPU)",
    "gpu": "Graphics Processing Unit (GPU / VRAM)",
    "display": "Display Panel / Screen Assembly",
    "charging_port": "Charging Connector / DC Jack",
    "keyboard": "Keyboard Deck",
    "touchpad": "Touchpad / Trackpad",
}

DESKTOP_STANDARD_COMPONENTS: Dict[str, str] = {
    "motherboard": "Desktop Motherboard (ATX/mATX/ITX)",
    "cpu": "Central Processing Unit (CPU)",
    "ram": "RAM Modules (DIMM)",
    "cpu_cooler": "CPU Cooler (Heatsink / Fan / AIO)",
    "gpu": "Discrete GPU (PCIe Graphics Card)",
    "ssd": "Solid State Drive (M.2 NVMe / SATA SSD)",
    "hdd": "Hard Disk Drive (3.5\" / 2.5\" HDD)",
    "psu": "Power Supply Unit (ATX PSU)",
    "case_fans": "Chassis / Case Cooling Fans",
    "cables": "Internal Power / SATA / Front Panel Cables",
    "cabinet": "PC Cabinet / Chassis",
}


@dataclass
class ComponentItem:
    """Represents a single hardware component in the inventory."""
    component_name: str
    component_category: str
    device_type: str                         # "laptop", "desktop", or "unknown"
    detection_source: str                    # "roboflow", "cv_heuristic", "specs_report", "user_confirmed", "rule_inference"
    confidence: Optional[float] = None
    visibility_status: str = VisibilityStatus.NOT_DETECTED.value
    bounding_box: Optional[List[float]] = None  # [x1, y1, x2, y2]
    image_references: List[str] = field(default_factory=list)
    verification_status: str = "unverified"     # "unverified", "verified", "requires_review", "user_confirmed", "verified_from_report", "verified_with_report"
    supported_by_model: bool = True
    close_up_needed: bool = False
    notes: Optional[str] = None
    # Crucial property: NOT_DETECTED never means NOT_PRESENT
    is_present: Optional[bool] = None          # True (confirmed), False (confirmed missing), None (unknown)
    # Hardware report specifications and conflict management
    specs_details: Optional[Dict[str, Any]] = None
    has_conflict: bool = False
    conflict_details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComponentItem":
        data = dict(data)
        if "image_reference" in data and "image_references" not in data:
            ref = data.pop("image_reference")
            data["image_references"] = [ref] if ref else []
        return cls(**data)


class HardwareProfile:
    """
    Unified hardware inventory tracking visible, partially visible, uncertain,
    unobserved components across computer photographs, and imported hardware specs reports.
    """

    def __init__(self, device_type: str = "unknown"):
        self.device_type = self._normalize_device_type(device_type)
        self.cabinet_status: str = CabinetStatus.UNKNOWN.value
        self.components: Dict[str, ComponentItem] = {}
        self.images_analyzed: List[str] = []
        self.close_up_requests: List[str] = []
        self.notes: List[str] = []
        self.imported_specs: Optional[Dict[str, Any]] = None
        self.specs_conflicts: List[Dict[str, Any]] = []
        self._init_standard_profile()

    @staticmethod
    def _normalize_device_type(raw_type: Optional[str]) -> str:
        if not raw_type:
            return "unknown"
        val = str(raw_type).strip().lower()
        if "desktop" in val or "tower" in val or val == "pc":
            return "desktop"
        if "laptop" in val or "notebook" in val:
            return "laptop"
        return val

    def _init_standard_profile(self) -> None:
        """Initialize empty standard component inventory for the specified device type."""
        std_map = {}
        if self.device_type == "desktop":
            std_map = DESKTOP_STANDARD_COMPONENTS
        elif self.device_type == "laptop":
            std_map = LAPTOP_STANDARD_COMPONENTS
        else:
            std_map = {**LAPTOP_STANDARD_COMPONENTS, **DESKTOP_STANDARD_COMPONENTS}

        for cat, name in std_map.items():
            self.components[cat] = ComponentItem(
                component_name=name,
                component_category=cat,
                device_type=self.device_type,
                detection_source="not_detected",
                confidence=None,
                visibility_status=VisibilityStatus.NOT_DETECTED.value,
                bounding_box=None,
                image_references=[],
                verification_status="unverified",
                supported_by_model=True,
                close_up_needed=False,
                notes="Not observed in provided image(s). Note: NOT_DETECTED does not mean NOT_PRESENT.",
                is_present=None,
            )

    def set_device_type(self, new_type: str) -> None:
        """Switch or specialize device type between laptop and desktop."""
        norm = self._normalize_device_type(new_type)
        if norm != self.device_type:
            self.device_type = norm
            existing = dict(self.components)
            self.components = {}
            self._init_standard_profile()
            for cat, item in existing.items():
                if item.visibility_status != VisibilityStatus.NOT_DETECTED.value or item.specs_details is not None:
                    item.device_type = self.device_type
                    self.components[cat] = item

    def add_component_detection(
        self,
        category: str,
        name: str,
        confidence: float,
        bounding_box: Optional[List[float]] = None,
        image_ref: Optional[str] = None,
        visibility_status: Optional[str] = None,
        source: str = "roboflow",
        supported_by_model: bool = True,
        notes: Optional[str] = None,
    ) -> ComponentItem:
        """Add or update a component observation from computer vision or model inference."""
        cat_key = category.strip().lower()

        if visibility_status is None:
            if confidence >= 0.40:
                visibility_status = VisibilityStatus.DETECTED.value
            elif confidence >= 0.20:
                visibility_status = VisibilityStatus.PARTIALLY_VISIBLE.value
            else:
                visibility_status = VisibilityStatus.UNCERTAIN.value

        existing = self.components.get(cat_key)
        images = [image_ref] if image_ref else []

        if existing and existing.visibility_status != VisibilityStatus.NOT_DETECTED.value:
            if image_ref and image_ref not in existing.image_references:
                images = existing.image_references + [image_ref]
            else:
                images = existing.image_references or images

            rank = {
                VisibilityStatus.USER_CONFIRMED.value: 5,
                VisibilityStatus.DETECTED.value: 4,
                VisibilityStatus.PARTIALLY_VISIBLE.value: 3,
                VisibilityStatus.UNCERTAIN.value: 2,
                VisibilityStatus.IMPORTED_FROM_REPORT.value: 1,
                VisibilityStatus.NOT_DETECTED.value: 0,
            }

            best_status = existing.visibility_status
            if rank.get(visibility_status, 0) > rank.get(existing.visibility_status, 0):
                best_status = visibility_status

            best_conf = max(
                c for c in [existing.confidence, confidence] if c is not None
            ) if (existing.confidence is not None or confidence is not None) else None

            best_box = bounding_box if bounding_box is not None else existing.bounding_box

            item = ComponentItem(
                component_name=name or existing.component_name,
                component_category=cat_key,
                device_type=self.device_type,
                detection_source=source or existing.detection_source,
                confidence=best_conf,
                visibility_status=best_status,
                bounding_box=best_box,
                image_references=images,
                verification_status="verified" if best_status == VisibilityStatus.DETECTED.value else existing.verification_status,
                supported_by_model=supported_by_model,
                close_up_needed=(best_status in (VisibilityStatus.PARTIALLY_VISIBLE.value, VisibilityStatus.UNCERTAIN.value)),
                notes=notes or existing.notes,
                is_present=True,
                specs_details=existing.specs_details,
                has_conflict=existing.has_conflict,
                conflict_details=existing.conflict_details,
            )
        else:
            # Preserve existing specs details if present
            saved_specs = existing.specs_details if existing else None
            saved_conflict = existing.has_conflict if existing else False
            saved_conflict_details = existing.conflict_details if existing else None

            item = ComponentItem(
                component_name=name,
                component_category=cat_key,
                device_type=self.device_type,
                detection_source=source,
                confidence=confidence,
                visibility_status=visibility_status,
                bounding_box=bounding_box,
                image_references=images,
                verification_status="verified" if visibility_status == VisibilityStatus.DETECTED.value else "unverified",
                supported_by_model=supported_by_model,
                close_up_needed=(visibility_status in (VisibilityStatus.PARTIALLY_VISIBLE.value, VisibilityStatus.UNCERTAIN.value)),
                notes=notes,
                is_present=True,
                specs_details=saved_specs,
                has_conflict=saved_conflict,
                conflict_details=saved_conflict_details,
            )

        self.components[cat_key] = item
        return item

    def mark_unsupported_category(self, category: str, name: str, reason: str = "Class not recognized by current model weights") -> ComponentItem:
        """Explicitly record a component that the vision model does NOT support detecting."""
        cat_key = category.strip().lower()
        item = ComponentItem(
            component_name=name,
            component_category=cat_key,
            device_type=self.device_type,
            detection_source="model_unsupported",
            confidence=None,
            visibility_status=VisibilityStatus.NOT_DETECTED.value,
            bounding_box=None,
            image_references=[],
            verification_status="unsupported_by_detector",
            supported_by_model=False,
            close_up_needed=False,
            notes=f"Unsupported class: {reason}. Not fabricated.",
            is_present=None,
        )
        self.components[cat_key] = item
        return item

    def set_cabinet_status(self, status: str, notes: Optional[str] = None) -> None:
        """Record whether a desktop cabinet is open or closed, or laptop state."""
        self.cabinet_status = status
        if notes:
            self.notes.append(notes)

        # Rule: A closed opaque cabinet must not be treated as proof of its internal components.
        if status == CabinetStatus.CLOSED_CABINET.value:
            for cat, comp in self.components.items():
                if cat not in ("cabinet",):
                    if comp.visibility_status == VisibilityStatus.NOT_DETECTED.value:
                        comp.notes = (
                            "Cabinet is closed. Internal components cannot be visually verified or ruled out. "
                            "Do not assume NOT_DETECTED means NOT_PRESENT."
                        )
                        comp.is_present = None
            req = "Desktop cabinet is closed. Remove the side panel and photograph the interior for motherboard, GPU, RAM, and cooler diagnosis."
            if req not in self.close_up_requests:
                self.close_up_requests.append(req)

    def merge_profile(self, other: "HardwareProfile") -> None:
        """Merge observations from another image or diagnostic run into this profile."""
        if not other:
            return

        if self.device_type == "unknown" and other.device_type != "unknown":
            self.set_device_type(other.device_type)

        for img in other.images_analyzed:
            if img not in self.images_analyzed:
                self.images_analyzed.append(img)

        for req in other.close_up_requests:
            if req not in self.close_up_requests:
                self.close_up_requests.append(req)

        if other.cabinet_status == CabinetStatus.OPEN_INTERIOR.value:
            self.cabinet_status = CabinetStatus.OPEN_INTERIOR.value
        elif self.cabinet_status == CabinetStatus.UNKNOWN.value and other.cabinet_status != CabinetStatus.UNKNOWN.value:
            self.cabinet_status = other.cabinet_status

        for cat, other_item in other.components.items():
            if other_item.visibility_status != VisibilityStatus.NOT_DETECTED.value:
                self.add_component_detection(
                    category=cat,
                    name=other_item.component_name,
                    confidence=other_item.confidence or 0.0,
                    bounding_box=other_item.bounding_box,
                    image_ref=other_item.image_references[0] if other_item.image_references else None,
                    visibility_status=other_item.visibility_status,
                    source=other_item.detection_source,
                    supported_by_model=other_item.supported_by_model,
                    notes=other_item.notes,
                )

        if other.imported_specs and not self.imported_specs:
            self.import_specs(other.imported_specs)

    # -----------------------------------------------------------------------
    # Hardware Specifications Report Integration
    # -----------------------------------------------------------------------

    def import_specs(self, specs: Any) -> Dict[str, Any]:
        """
        Merge imported hardware specifications with the visual profile.
        Maintains strict source separation and marks conflicts for verification.
        """
        if hasattr(specs, "to_dict"):
            specs_dict = specs.to_dict()
        elif isinstance(specs, dict):
            specs_dict = dict(specs)
        else:
            return {}

        self.imported_specs = specs_dict

        # 1. Device Type Conflict / Specialization Check
        spec_sys_type = specs_dict.get("system_type")
        if spec_sys_type:
            norm_spec_type = self._normalize_device_type(spec_sys_type)
            if self.device_type == "unknown":
                self.set_device_type(norm_spec_type)
            elif self.device_type != norm_spec_type:
                conflict_msg = (
                    f"Device type conflict: Visual/user input identified device as '{self.device_type}', "
                    f"but imported report specifies '{norm_spec_type}'. Marked for manual verification."
                )
                self.specs_conflicts.append({
                    "field": "device_type",
                    "visual_value": self.device_type,
                    "specs_value": norm_spec_type,
                    "conflict_message": conflict_msg,
                })

        # 2. CPU
        cpu_model = specs_dict.get("cpu_model")
        if cpu_model:
            cpu_details = {
                "manufacturer": specs_dict.get("cpu_manufacturer"),
                "model": cpu_model,
                "cores": specs_dict.get("cpu_cores"),
                "threads": specs_dict.get("cpu_threads"),
            }
            spec_lbl = f"{specs_dict.get('cpu_manufacturer') or ''} {cpu_model}".strip()
            self._apply_spec_to_component(
                category="cpu",
                display_name="Central Processing Unit (CPU)",
                specs_details=cpu_details,
                spec_label=spec_lbl,
            )

        # 3. GPU
        gpu_model = specs_dict.get("gpu_model")
        if gpu_model:
            gpu_details = {
                "manufacturer": specs_dict.get("gpu_manufacturer"),
                "model": gpu_model,
            }
            spec_lbl = f"{specs_dict.get('gpu_manufacturer') or ''} {gpu_model}".strip()
            self._apply_spec_to_component(
                category="gpu",
                display_name="Discrete GPU (PCIe Graphics Card)" if self.device_type == "desktop" else "Graphics Processing Unit (GPU / VRAM)",
                specs_details=gpu_details,
                spec_label=spec_lbl,
            )

        # 4. RAM
        ram_gb = specs_dict.get("ram_total_gb")
        if ram_gb is not None:
            ram_details = {
                "capacity_gb": ram_gb,
                "configuration": specs_dict.get("ram_configuration"),
                "slots_used": specs_dict.get("ram_slots_used"),
                "total_slots": specs_dict.get("ram_total_slots"),
            }
            ram_lbl = f"{ram_gb} GB RAM"
            if specs_dict.get("ram_configuration"):
                ram_lbl += f" ({specs_dict.get('ram_configuration')})"

            comp = self._apply_spec_to_component(
                category="ram",
                display_name="RAM Modules (DIMM)" if self.device_type == "desktop" else "RAM (SODIMM Memory)",
                specs_details=ram_details,
                spec_label=ram_lbl,
            )

            # Check for conflict if visual detection observed a discrepancy
            slots_used = specs_dict.get("ram_slots_used")
            if comp and comp.detection_source != "specs_report" and slots_used and slots_used > 1:
                if comp.notes and "1 module" in comp.notes.lower():
                    self.flag_conflict(
                        category="ram",
                        conflict_message=f"Visual check noted 1 RAM stick, but report specifies {slots_used} slots used ({ram_gb} GB).",
                    )

        # 5. Motherboard
        mobo_model = specs_dict.get("motherboard_model")
        if mobo_model:
            mobo_details = {
                "manufacturer": specs_dict.get("motherboard_manufacturer"),
                "model": mobo_model,
            }
            mobo_lbl = f"{specs_dict.get('motherboard_manufacturer') or ''} {mobo_model}".strip()
            self._apply_spec_to_component(
                category="motherboard",
                display_name="Desktop Motherboard (ATX/mATX/ITX)" if self.device_type == "desktop" else "Motherboard / Mainboard",
                specs_details=mobo_details,
                spec_label=mobo_lbl,
            )

        # 6. Storage
        storage_devs = specs_dict.get("storage_devices") or []
        if storage_devs:
            for sdev in storage_devs:
                stype = (sdev.get("type") or "").lower()
                target_cat = "storage"
                if self.device_type == "desktop":
                    target_cat = "hdd" if "hdd" in stype else "ssd"

                s_lbl = sdev.get("model") or "Storage Drive"
                if sdev.get("capacity_gb"):
                    s_lbl += f" ({sdev['capacity_gb']} GB)"

                self._apply_spec_to_component(
                    category=target_cat,
                    display_name="Solid State Drive (SSD)" if target_cat == "ssd" else "Hard Disk Drive (HDD)" if target_cat == "hdd" else "Storage Drive (SSD/HDD)",
                    specs_details=sdev,
                    spec_label=s_lbl,
                )

        return {
            "imported_specs": self.imported_specs,
            "conflicts": self.specs_conflicts,
        }

    def _apply_spec_to_component(
        self,
        category: str,
        display_name: str,
        specs_details: Dict[str, Any],
        spec_label: str,
    ) -> ComponentItem:
        """Applies imported specification details to a component while strictly preserving visual data separation."""
        cat_key = category.strip().lower()
        existing = self.components.get(cat_key)

        if existing and existing.visibility_status in (
            VisibilityStatus.DETECTED.value,
            VisibilityStatus.PARTIALLY_VISIBLE.value,
            VisibilityStatus.UNCERTAIN.value,
        ):
            # Preserve visual observation, bounding box, confidence, and detection source
            # Attach specs details and update verification
            existing.specs_details = specs_details
            existing.verification_status = "verified_with_report"
            existing.notes = (
                f"{existing.notes or ''} [Specs Report]: {spec_label}. "
                f"(Source separation: Specification details verified via imported report; visual presence verified via photo)."
            ).strip()
            return existing
        else:
            # Component was NOT observed in photos, but confirmed in the hardware specification report
            item = ComponentItem(
                component_name=f"{display_name} [{spec_label}]",
                component_category=cat_key,
                device_type=self.device_type,
                detection_source="specs_report",
                confidence=None,
                visibility_status=VisibilityStatus.IMPORTED_FROM_REPORT.value,
                bounding_box=None,
                image_references=[],
                verification_status="verified_from_report",
                supported_by_model=True,
                close_up_needed=False,
                notes=f"Imported from specification report ({spec_label}). Not visually observed in photos.",
                is_present=True,
                specs_details=specs_details,
            )
            self.components[cat_key] = item
            return item

    def flag_conflict(self, category: str, conflict_message: str) -> None:
        """Flags a data-source conflict on a component for human or technician review."""
        cat_key = category.strip().lower()
        if cat_key in self.components:
            comp = self.components[cat_key]
            comp.has_conflict = True
            comp.conflict_details = conflict_message
            comp.verification_status = "requires_review"
            self.specs_conflicts.append({
                "category": cat_key,
                "component_name": comp.component_name,
                "conflict": conflict_message,
            })

    def get_detected_components(self) -> List[ComponentItem]:
        """Return all components that are confirmed visible in photographs (DETECTED)."""
        return [c for c in self.components.values() if c.visibility_status == VisibilityStatus.DETECTED.value]

    def get_partially_visible_components(self) -> List[ComponentItem]:
        """Return components that are partially occluded or clipped in photographs."""
        return [c for c in self.components.values() if c.visibility_status == VisibilityStatus.PARTIALLY_VISIBLE.value]

    def get_uncertain_components(self) -> List[ComponentItem]:
        """Return components with marginal confidence or visual ambiguity."""
        return [c for c in self.components.values() if c.visibility_status == VisibilityStatus.UNCERTAIN.value]

    def get_imported_components(self) -> List[ComponentItem]:
        """Return components that were established or enriched from an imported hardware report."""
        return [
            c for c in self.components.values()
            if c.visibility_status == VisibilityStatus.IMPORTED_FROM_REPORT.value or c.specs_details is not None
        ]

    def get_close_up_requests(self) -> List[str]:
        """Generate actionable requests for close-up photos when needed."""
        requests = list(self.close_up_requests)
        for c in self.get_partially_visible_components():
            msg = f"Component '{c.component_name}' is partially visible or occluded. Provide a direct, well-lit close-up photo."
            if msg not in requests:
                requests.append(msg)
        for c in self.get_uncertain_components():
            msg = f"Component '{c.component_name}' was detected with low confidence. Provide a steady, non-glare close-up photo."
            if msg not in requests:
                requests.append(msg)
        return requests

    def summary_findings(self) -> List[str]:
        """Produce backward-compatible list of strings for questionaire.py and fault_engine.py."""
        findings = []
        for c in self.get_detected_components():
            conf_str = f" ({c.confidence:.0%})" if c.confidence is not None else ""
            findings.append(f"{c.component_name}{conf_str}")
        for c in self.get_partially_visible_components():
            findings.append(f"{c.component_name} (Partially Visible)")
        for c in self.get_uncertain_components():
            findings.append(f"{c.component_name} (Uncertain / Low Confidence)")
        for c in self.components.values():
            if c.visibility_status == VisibilityStatus.IMPORTED_FROM_REPORT.value:
                findings.append(f"{c.component_name} (Imported from Report)")
        return findings

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile to JSON-compatible dictionary."""
        return {
            "device_type": self.device_type,
            "cabinet_status": self.cabinet_status,
            "images_analyzed": self.images_analyzed,
            "close_up_requests": self.get_close_up_requests(),
            "notes": self.notes,
            "imported_specs": self.imported_specs,
            "specs_conflicts": self.specs_conflicts,
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "summary": {
                "detected_count": len(self.get_detected_components()),
                "partially_visible_count": len(self.get_partially_visible_components()),
                "uncertain_count": len(self.get_uncertain_components()),
                "imported_from_report_count": len([c for c in self.components.values() if c.visibility_status == VisibilityStatus.IMPORTED_FROM_REPORT.value]),
                "not_detected_count": len([c for c in self.components.values() if c.visibility_status == VisibilityStatus.NOT_DETECTED.value]),
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HardwareProfile":
        profile = cls(device_type=data.get("device_type", "unknown"))
        profile.cabinet_status = data.get("cabinet_status", CabinetStatus.UNKNOWN.value)
        profile.images_analyzed = data.get("images_analyzed", [])
        profile.close_up_requests = data.get("close_up_requests", [])
        profile.notes = data.get("notes", [])
        profile.imported_specs = data.get("imported_specs")
        profile.specs_conflicts = data.get("specs_conflicts", [])
        comps = data.get("components", {})
        for cat, cdata in comps.items():
            profile.components[cat] = ComponentItem.from_dict(cdata)
        return profile
