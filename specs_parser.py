"""
Hardware Specification Import & Sanitization System for Repair Before Replace.
Supports Laptops and Desktop PCs.

Parses hardware specification reports from multiple formats:
- TXT: systeminfo, dxdiag, lshw, inxi, and generic key-value exports
- JSON: structured hardware dumps
- CSV: tabular hardware specifications
- HTML: Speccy, HWiNFO, AIDA64 exports (safe stdlib html.parser, zero script execution)

Enforces data provenance separation:
  - Information imported from reports is tagged as 'specs_report' / 'IMPORTED_FROM_REPORT'
  - Never claims visual detection established specs that came from reports
  - Sanitizes sensitive data (product keys, MAC addresses, private user paths)
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from html.parser import HTMLParser
import csv
import io
import json
import re


MAX_REPORT_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB safety limit
ALLOWED_EXTENSIONS = {".txt", ".json", ".csv", ".html", ".htm", ".log"}


class SpecsSanitizer:
    """Sanitizes sensitive information from hardware reports before retention or display."""

    PRODUCT_KEY_REGEX = re.compile(r"\b[A-Z0-9]{5}-[A-Z0-9]{5}-[A-Z0-9]{5}-[A-Z0-9]{5}-[A-Z0-9]{5}\b", re.IGNORECASE)
    MAC_REGEX = re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")
    IPV4_REGEX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    USER_PATH_WIN = re.compile(r"([A-Za-z]:\\Users\\)([^\s\\/]+)", re.IGNORECASE)
    USER_PATH_NIX = re.compile(r"(/home/)([^\s/]+)", re.IGNORECASE)

    @classmethod
    def sanitize_text(cls, text: str) -> Tuple[str, List[str]]:
        """Redacts product keys, MAC addresses, private user folders, and IP addresses."""
        redacted_types = []
        out = text

        if cls.PRODUCT_KEY_REGEX.search(out):
            out = cls.PRODUCT_KEY_REGEX.sub("[REDACTED_PRODUCT_KEY]", out)
            redacted_types.append("Windows Product Key")

        if cls.MAC_REGEX.search(out):
            out = cls.MAC_REGEX.sub("[REDACTED_MAC_ADDRESS]", out)
            redacted_types.append("MAC Address")

        if cls.IPV4_REGEX.search(out):
            # Mask private/public IPs (leave common loopbacks if needed, or mask all)
            out = cls.IPV4_REGEX.sub("[REDACTED_IP]", out)
            redacted_types.append("IP Address")

        if cls.USER_PATH_WIN.search(out):
            out = cls.USER_PATH_WIN.sub(r"\1[REDACTED_USER]", out)
            redacted_types.append("User Directory Path")

        if cls.USER_PATH_NIX.search(out):
            out = cls.USER_PATH_NIX.sub(r"\1[REDACTED_USER]", out)
            redacted_types.append("User Directory Path")

        return out, list(set(redacted_types))


@dataclass
class ParsedHardwareSpecs:
    """Structured representation of hardware specifications extracted from an imported report."""
    system_type: Optional[str] = None          # "laptop", "desktop", or None
    os_name: Optional[str] = None
    os_version: Optional[str] = None
    cpu_manufacturer: Optional[str] = None     # "Intel", "AMD", "Apple", etc.
    cpu_model: Optional[str] = None
    cpu_cores: Optional[int] = None
    cpu_threads: Optional[int] = None
    gpu_manufacturer: Optional[str] = None     # "NVIDIA", "AMD", "Intel", etc.
    gpu_model: Optional[str] = None
    ram_total_gb: Optional[float] = None
    ram_configuration: Optional[str] = None    # e.g., "Dual Channel", "2x 16GB DDR4", "Slots: 2/4"
    ram_slots_used: Optional[int] = None
    ram_total_slots: Optional[int] = None
    storage_devices: List[Dict[str, Any]] = field(default_factory=list)
    storage_summary: Optional[str] = None
    motherboard_manufacturer: Optional[str] = None
    motherboard_model: Optional[str] = None
    source_format: str = "unknown"
    source_filename: Optional[str] = None
    sanitized_fields: List[str] = field(default_factory=list)
    raw_metadata: Dict[str, Any] = field(default_factory=dict)
    parsing_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParsedHardwareSpecs":
        return cls(**data)

    def is_empty(self) -> bool:
        """Returns True if no meaningful hardware specification was extracted."""
        return not any([
            self.cpu_model,
            self.gpu_model,
            self.ram_total_gb,
            self.motherboard_model,
            self.storage_devices,
            self.os_name,
        ])


def _normalize_system_type(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    val = text.strip().lower()
    if any(k in val for k in [
        "laptop", "notebook", "portable", "convertible", "thinkpad",
        "macbook", "zenbook", "latitude", "xps", "inspiron", "envy",
        "pavilion", "yoga", "ideapad", "surface book", "surface laptop"
    ]):
        return "laptop"
    if any(k in val for k in ["desktop", "tower", "pc", "all-in-one", "aio", "workstation", "rig"]):
        return "desktop"
    return None


def _clean_str(val: Optional[Any]) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def _parse_memory_gb(text: Any) -> Optional[float]:
    if text is None:
        return None
    if isinstance(text, (int, float)):
        # If value is very large, assuming bytes or MB
        val = float(text)
        if val > 1_000_000_000:
            return round(val / (1024**3), 1)
        if val > 1024:
            return round(val / 1024, 1)
        return round(val, 1)

    s = str(text).replace(",", "").strip()
    # Match e.g. "16384 MB" or "16.0 GB" or "32 GB" or "16GB"
    m_gb = re.search(r"([\d.]+)\s*(?:GB|GIB)", s, re.IGNORECASE)
    if m_gb:
        try:
            return round(float(m_gb.group(1)), 1)
        except ValueError:
            pass

    m_mb = re.search(r"([\d.]+)\s*(?:MB|MIB)", s, re.IGNORECASE)
    if m_mb:
        try:
            return round(float(m_mb.group(1)) / 1024.0, 1)
        except ValueError:
            pass

    m_bytes = re.search(r"(\d{9,})\s*(?:bytes|b)?", s, re.IGNORECASE)
    if m_bytes:
        try:
            return round(float(m_bytes.group(1)) / (1024**3), 1)
        except ValueError:
            pass

    # Bare number check
    m_num = re.search(r"^([\d.]+)$", s)
    if m_num:
        try:
            v = float(m_num.group(1))
            if v > 1024:
                return round(v / 1024.0, 1)
            return round(v, 1)
        except ValueError:
            pass

    return None


def _extract_cpu_vendor(cpu_str: Optional[str]) -> Optional[str]:
    if not cpu_str:
        return None
    s = cpu_str.upper()
    if "INTEL" in s:
        return "Intel"
    if "AMD" in s or "RYZEN" in s or "RADEON" in s:
        return "AMD"
    if "APPLE" in s or "M1" in s or "M2" in s or "M3" in s or "M4" in s:
        return "Apple"
    if "QUALCOMM" in s or "SNAPDRAGON" in s:
        return "Qualcomm"
    return None


def _extract_gpu_vendor(gpu_str: Optional[str]) -> Optional[str]:
    if not gpu_str:
        return None
    s = gpu_str.upper()
    if "NVIDIA" in s or "GEFORCE" in s or "RTX" in s or "GTX" in s or "QUADRO" in s:
        return "NVIDIA"
    if "AMD" in s or "RADEON" in s:
        return "AMD"
    if "INTEL" in s or "ARC" in s or "IRIS" in s or "UHD" in s or "HD GRAPHICS" in s:
        return "Intel"
    if "APPLE" in s:
        return "Apple"
    return None


# ---------------------------------------------------------------------------
# Base & Format Parsers
# ---------------------------------------------------------------------------

class BaseReportParser:
    """Abstract interface for format-specific hardware report parsers."""

    def can_parse(self, content: str, filename: Optional[str] = None) -> bool:
        raise NotImplementedError

    def parse(self, content: str, filename: Optional[str] = None) -> ParsedHardwareSpecs:
        raise NotImplementedError


class JsonReportParser(BaseReportParser):
    """Parses JSON hardware specification dumps (HWiNFO JSON, custom, lshw -json)."""

    def can_parse(self, content: str, filename: Optional[str] = None) -> bool:
        if filename and filename.lower().endswith(".json"):
            return True
        stripped = content.strip()
        return (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]"))

    def parse(self, content: str, filename: Optional[str] = None) -> ParsedHardwareSpecs:
        sanitized_content, redacted = SpecsSanitizer.sanitize_text(content)
        specs = ParsedHardwareSpecs(source_format="json", source_filename=filename, sanitized_fields=redacted)

        try:
            data = json.loads(sanitized_content)
        except Exception as exc:
            specs.parsing_warnings.append(f"JSON parsing error: {exc}")
            return specs

        if isinstance(data, list):
            # Convert list of component dicts to key-value or flatten
            flattened = {}
            for item in data:
                if isinstance(item, dict):
                    id_key = str(item.get("id") or item.get("class") or item.get("category") or item.get("name") or "")
                    flattened[id_key.lower()] = item
            data = flattened

        # Search recursively or top-level for standard hardware keys
        self._extract_from_dict(data, specs)
        return specs

    def _extract_from_dict(self, data: Dict[str, Any], specs: ParsedHardwareSpecs) -> None:
        # Case-insensitive lookup helper
        lookup = {str(k).lower(): v for k, v in data.items()}

        # OS
        os_info = lookup.get("os") or lookup.get("operating_system") or lookup.get("operating system") or lookup.get("system_os")
        if isinstance(os_info, dict):
            specs.os_name = _clean_str(os_info.get("name") or os_info.get("caption") or os_info.get("os_name"))
            specs.os_version = _clean_str(os_info.get("version") or os_info.get("build") or os_info.get("os_version"))
        elif isinstance(os_info, str):
            specs.os_name = os_info

        # System Type / Model
        sys_type = lookup.get("system_type") or lookup.get("device_type") or lookup.get("chassis") or lookup.get("form_factor")
        if not sys_type and "system" in lookup and isinstance(lookup["system"], dict):
            sys_type = lookup["system"].get("family") or lookup["system"].get("model") or lookup["system"].get("type")
        specs.system_type = _normalize_system_type(str(sys_type)) if sys_type else None

        # CPU
        cpu = lookup.get("cpu") or lookup.get("processor") or lookup.get("processors")
        if isinstance(cpu, list) and len(cpu) > 0:
            cpu = cpu[0]
        if isinstance(cpu, dict):
            specs.cpu_model = _clean_str(cpu.get("name") or cpu.get("model") or cpu.get("product") or cpu.get("caption"))
            specs.cpu_manufacturer = _clean_str(cpu.get("manufacturer") or cpu.get("vendor")) or _extract_cpu_vendor(specs.cpu_model)
            cores = cpu.get("cores") or cpu.get("core_count") or cpu.get("cpu_cores")
            threads = cpu.get("threads") or cpu.get("thread_count") or cpu.get("logical_processors")
            if cores is not None:
                try:
                    specs.cpu_cores = int(cores)
                except (ValueError, TypeError):
                    pass
            if threads is not None:
                try:
                    specs.cpu_threads = int(threads)
                except (ValueError, TypeError):
                    pass
        elif isinstance(cpu, str):
            specs.cpu_model = cpu
            specs.cpu_manufacturer = _extract_cpu_vendor(cpu)

        # GPU
        gpu = lookup.get("gpu") or lookup.get("graphics") or lookup.get("display") or lookup.get("video_card")
        if isinstance(gpu, list) and len(gpu) > 0:
            gpu = gpu[0]
        if isinstance(gpu, dict):
            specs.gpu_model = _clean_str(gpu.get("name") or gpu.get("model") or gpu.get("product") or gpu.get("card"))
            specs.gpu_manufacturer = _clean_str(gpu.get("manufacturer") or gpu.get("vendor")) or _extract_gpu_vendor(specs.gpu_model)
        elif isinstance(gpu, str):
            specs.gpu_model = gpu
            specs.gpu_manufacturer = _extract_gpu_vendor(gpu)

        # RAM
        ram = lookup.get("ram") or lookup.get("memory") or lookup.get("physical_memory") or lookup.get("total_memory")
        if isinstance(ram, dict):
            specs.ram_total_gb = _parse_memory_gb(ram.get("total") or ram.get("size") or ram.get("capacity") or ram.get("installed"))
            specs.ram_configuration = _clean_str(ram.get("configuration") or ram.get("channels") or ram.get("type"))
            slots_used = ram.get("slots_used") or ram.get("used_slots")
            total_slots = ram.get("total_slots") or ram.get("slots")
            if slots_used is not None:
                try:
                    specs.ram_slots_used = int(slots_used)
                except (ValueError, TypeError):
                    pass
            if total_slots is not None:
                try:
                    specs.ram_total_slots = int(total_slots)
                except (ValueError, TypeError):
                    pass
        elif isinstance(ram, (int, float, str)):
            specs.ram_total_gb = _parse_memory_gb(ram)

        # Motherboard
        mobo = lookup.get("motherboard") or lookup.get("mainboard") or lookup.get("board") or lookup.get("baseboard")
        if isinstance(mobo, dict):
            specs.motherboard_model = _clean_str(mobo.get("model") or mobo.get("product") or mobo.get("name"))
            specs.motherboard_manufacturer = _clean_str(mobo.get("manufacturer") or mobo.get("vendor"))
        elif isinstance(mobo, str):
            specs.motherboard_model = mobo

        # Storage
        disks = lookup.get("storage") or lookup.get("disks") or lookup.get("drives") or lookup.get("disk")
        if isinstance(disks, list):
            for d in disks:
                if isinstance(d, dict):
                    specs.storage_devices.append({
                        "model": _clean_str(d.get("model") or d.get("name") or d.get("product")),
                        "type": _clean_str(d.get("type") or d.get("interface") or "SSD/HDD"),
                        "capacity_gb": _parse_memory_gb(d.get("size") or d.get("capacity")),
                    })
                elif isinstance(d, str):
                    specs.storage_devices.append({"model": d, "capacity_gb": _parse_memory_gb(d)})
        elif isinstance(disks, dict):
            specs.storage_devices.append({
                "model": _clean_str(disks.get("model") or disks.get("name")),
                "type": _clean_str(disks.get("type") or "SSD/HDD"),
                "capacity_gb": _parse_memory_gb(disks.get("size") or disks.get("capacity")),
            })
        elif isinstance(disks, str):
            specs.storage_devices.append({"model": disks, "capacity_gb": _parse_memory_gb(disks)})

        if specs.storage_devices:
            summary_parts = []
            for d in specs.storage_devices:
                cap = f"{d['capacity_gb']}GB" if d.get("capacity_gb") else ""
                model = d.get("model") or "Storage Drive"
                summary_parts.append(f"{model} {cap}".strip())
            specs.storage_summary = ", ".join(summary_parts)


class CsvReportParser(BaseReportParser):
    """Parses CSV reports structured as key-value pairs or component tables."""

    def can_parse(self, content: str, filename: Optional[str] = None) -> bool:
        if filename and filename.lower().endswith(".csv"):
            return True
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if len(lines) >= 2 and "," in lines[0] and "," in lines[1]:
            return True
        return False

    def parse(self, content: str, filename: Optional[str] = None) -> ParsedHardwareSpecs:
        sanitized_content, redacted = SpecsSanitizer.sanitize_text(content)
        specs = ParsedHardwareSpecs(source_format="csv", source_filename=filename, sanitized_fields=redacted)

        reader = csv.reader(io.StringIO(sanitized_content))
        rows = [row for row in reader if row]
        if not rows:
            return specs

        # Two main CSV layouts:
        # Layout 1: 2 columns [Property/Key, Value]
        # Layout 2: Multi-column table with headers [Category, Component, Details, ...]
        is_key_value = all(len(r) == 2 for r in rows[:min(5, len(rows))])

        if is_key_value:
            for r in rows:
                k = r[0].strip().lower()
                v = r[1].strip()
                _assign_key_value(k, v, specs)
        else:
            headers = [h.strip().lower() for h in rows[0]]
            cat_idx = _find_index(headers, ["category", "component", "type", "device"])
            name_idx = _find_index(headers, ["model", "name", "description", "product", "value"])
            cap_idx = _find_index(headers, ["capacity", "size", "memory", "details"])

            for r in rows[1:]:
                if not r:
                    continue
                cat = r[cat_idx].lower() if cat_idx is not None and cat_idx < len(r) else ""
                name = r[name_idx] if name_idx is not None and name_idx < len(r) else ""
                cap = r[cap_idx] if cap_idx is not None and cap_idx < len(r) else ""

                combined = f"{name} {cap}".strip()
                _assign_key_value(cat, combined, specs)

        return specs


class SafeHtmlTableParser(HTMLParser):
    """Zero-script HTML parser using Python standard library to extract tables and key-value tags."""

    def __init__(self):
        super().__init__()
        self.in_script = False
        self.current_tag = None
        self.rows: List[List[str]] = []
        self.current_row: List[str] = []
        self.current_cell: List[str] = []
        self.text_blocks: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self.current_tag = tag.lower()
        if self.current_tag in ("script", "style"):
            self.in_script = True
        elif self.current_tag == "tr":
            self.current_row = []
        elif self.current_tag in ("td", "th"):
            self.current_cell = []

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in ("script", "style"):
            self.in_script = False
        elif tag_lower in ("td", "th"):
            text = " ".join("".join(self.current_cell).split())
            self.current_row.append(text)
            self.current_cell = []
        elif tag_lower == "tr":
            if self.current_row:
                self.rows.append(self.current_row)
                self.current_row = []
        self.current_tag = None

    def handle_data(self, data: str) -> None:
        if self.in_script:
            return
        text = data.strip()
        if text:
            if self.current_tag in ("td", "th"):
                self.current_cell.append(text)
            else:
                self.text_blocks.append(text)


class HtmlReportParser(BaseReportParser):
    """Parses HTML hardware reports (Speccy, HWiNFO, AIDA64) safely using standard library html.parser."""

    def can_parse(self, content: str, filename: Optional[str] = None) -> bool:
        if filename and (filename.lower().endswith(".html") or filename.lower().endswith(".htm")):
            return True
        s = content[:1024].lower()
        return "<html" in s or "<!doctype html" in s or "<table" in s

    def parse(self, content: str, filename: Optional[str] = None) -> ParsedHardwareSpecs:
        sanitized_content, redacted = SpecsSanitizer.sanitize_text(content)
        specs = ParsedHardwareSpecs(source_format="html", source_filename=filename, sanitized_fields=redacted)

        parser = SafeHtmlTableParser()
        try:
            parser.feed(sanitized_content)
        except Exception as exc:
            specs.parsing_warnings.append(f"HTML parsing error: {exc}")
            return specs

        # First evaluate structured table rows
        for row in parser.rows:
            if len(row) >= 2:
                k = row[0].strip().lower()
                v = " - ".join(x for x in row[1:] if x).strip()
                _assign_key_value(k, v, specs)

        # Fallback to text blocks if table didn't fill all fields
        if specs.is_empty() and parser.text_blocks:
            full_text = "\n".join(parser.text_blocks)
            text_parser = TextReportParser()
            fallback_specs = text_parser.parse(full_text, filename)
            fallback_specs.source_format = "html"
            fallback_specs.sanitized_fields = list(set(redacted + fallback_specs.sanitized_fields))
            return fallback_specs

        return specs


class TextReportParser(BaseReportParser):
    """Parses text reports: Windows systeminfo, dxdiag.txt, lshw, inxi, and key-value formats."""

    def can_parse(self, content: str, filename: Optional[str] = None) -> bool:
        return True  # Catch-all text parser

    def parse(self, content: str, filename: Optional[str] = None) -> ParsedHardwareSpecs:
        sanitized_content, redacted = SpecsSanitizer.sanitize_text(content)
        specs = ParsedHardwareSpecs(source_format="txt", source_filename=filename, sanitized_fields=redacted)

        # Identify known format signatures
        lower_content = sanitized_content.lower()
        if (
            (filename and "dxdiag" in filename.lower())
            or "directx" in lower_content
            or "system information report written at:" in lower_content
            or ("card name:" in lower_content and "operating system:" in lower_content)
        ):
            self._parse_dxdiag(sanitized_content, specs)
        elif (
            "os name:" in lower_content
            or "host name:" in lower_content
            or "os configuration:" in lower_content
            or "total physical memory:" in lower_content
        ):
            self._parse_systeminfo(sanitized_content, specs)
        elif "*-cpu" in lower_content or "*-memory" in lower_content or "lshw" in lower_content:
            self._parse_lshw(sanitized_content, specs)
        else:
            self._parse_generic_lines(sanitized_content, specs)

        return specs

    def _parse_dxdiag(self, text: str, specs: ParsedHardwareSpecs) -> None:
        """Parses dxdiag text report."""
        for line in text.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, _, v = line.partition(":")
            k = k.strip().lower()
            v = v.strip()

            if k == "operating system":
                specs.os_name = v
            elif k == "system manufacturer":
                if not specs.motherboard_manufacturer:
                    specs.motherboard_manufacturer = v
            elif k == "system model":
                if not specs.system_type:
                    specs.system_type = _normalize_system_type(v)
                if not specs.motherboard_model:
                    specs.motherboard_model = v
            elif k == "processor":
                specs.cpu_model = v
                specs.cpu_manufacturer = _extract_cpu_vendor(v)
                m_cores = re.search(r"\((\d+)\s*CPUs?\)", v, re.IGNORECASE)
                if m_cores:
                    specs.cpu_threads = int(m_cores.group(1))
            elif k == "memory":
                specs.ram_total_gb = _parse_memory_gb(v)
            elif k in ("card name", "chip type") and not specs.gpu_model:
                specs.gpu_model = v
                specs.gpu_manufacturer = _extract_gpu_vendor(v)

    def _parse_systeminfo(self, text: str, specs: ParsedHardwareSpecs) -> None:
        """Parses Windows systeminfo CLI report."""
        for line in text.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, _, v = line.partition(":")
            k = k.strip().lower()
            v = v.strip()

            if k == "os name":
                specs.os_name = v
            elif k == "os version":
                specs.os_version = v
            elif k == "system manufacturer":
                if not specs.motherboard_manufacturer:
                    specs.motherboard_manufacturer = v
            elif k == "system model":
                specs.motherboard_model = v
                if not specs.system_type:
                    specs.system_type = _normalize_system_type(v)
            elif k == "system type":
                if not specs.system_type:
                    specs.system_type = _normalize_system_type(v)
            elif k.startswith("processor(s)") or "[01]" in k and "intel" in v.lower() or "amd" in v.lower():
                # systeminfo format: "[01]: Intel64 Family 6 Model... ~2808 Mhz" or "Processor(s): 1 Processor(s) Installed."
                if "intel" in v.lower() or "amd" in v.lower() or "ghz" in v.lower():
                    specs.cpu_model = v
                    specs.cpu_manufacturer = _extract_cpu_vendor(v)
            elif k == "total physical memory":
                specs.ram_total_gb = _parse_memory_gb(v)

    def _parse_lshw(self, text: str, specs: ParsedHardwareSpecs) -> None:
        """Parses Linux lshw summary or output."""
        current_section = None
        for line in text.splitlines():
            sline = line.strip()
            if sline.startswith("*-"):
                current_section = sline[2:].lower().strip()
                continue
            if ":" not in sline:
                continue
            k, _, v = sline.partition(":")
            k = k.strip().lower()
            v = v.strip()

            if current_section == "core":
                if k == "product":
                    specs.motherboard_model = v
                elif k == "vendor":
                    specs.motherboard_manufacturer = v
            elif current_section == "cpu":
                if k == "product":
                    specs.cpu_model = v
                    specs.cpu_manufacturer = _extract_cpu_vendor(v)
            elif current_section in ("memory", "memory:0"):
                if k == "size":
                    specs.ram_total_gb = _parse_memory_gb(v)
            elif current_section == "display":
                if k == "product" and not specs.gpu_model:
                    specs.gpu_model = v
                    specs.gpu_manufacturer = _extract_gpu_vendor(v)
            elif current_section in ("disk", "storage"):
                if k == "product":
                    specs.storage_devices.append({"model": v, "capacity_gb": None})

    def _parse_generic_lines(self, text: str, specs: ParsedHardwareSpecs) -> None:
        """Parses lines of key-value pairs or markdown-style lists."""
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip markdown bullets
            line = re.sub(r"^[\*\-•]\s*", "", line)
            if ":" in line:
                k, _, v = line.partition(":")
                _assign_key_value(k.strip().lower(), v.strip(), specs)
            elif "=" in line:
                k, _, v = line.partition("=")
                _assign_key_value(k.strip().lower(), v.strip(), specs)


# ---------------------------------------------------------------------------
# Key-Value Assignment Helper
# ---------------------------------------------------------------------------

def _assign_key_value(key: str, val: str, specs: ParsedHardwareSpecs) -> None:
    """Intelligently maps various hardware naming conventions to ParsedHardwareSpecs fields."""
    if not val:
        return
    k = key.lower().replace("-", " ").replace("_", " ").strip()

    # System Type
    if k in ("system type", "device type", "chassis", "form factor", "computer type"):
        if not specs.system_type:
            specs.system_type = _normalize_system_type(val)
    elif "laptop" in val.lower() or "notebook" in val.lower():
        if not specs.system_type:
            specs.system_type = "laptop"

    # OS
    elif k in ("os", "operating system", "os name", "platform"):
        specs.os_name = val
    elif k in ("os version", "build", "kernel"):
        specs.os_version = val

    # CPU
    elif any(term in k for term in ["cpu", "processor", "central processor"]):
        if "cores" in k or "core count" in k:
            m = re.search(r"\d+", val)
            if m:
                specs.cpu_cores = int(m.group(0))
        elif "threads" in k:
            m = re.search(r"\d+", val)
            if m:
                specs.cpu_threads = int(m.group(0))
        else:
            if not specs.cpu_model:
                specs.cpu_model = val
                specs.cpu_manufacturer = _extract_cpu_vendor(val)
                # Check for inline core mention e.g. "Intel Core i7-12700K 12-Core"
                m_cores = re.search(r"(\d+)[\s-]*cores?", val, re.IGNORECASE)
                if m_cores:
                    specs.cpu_cores = int(m_cores.group(1))

    # GPU
    elif any(term in k for term in ["gpu", "graphics", "video card", "display adapter", "discrete gpu"]):
        if not specs.gpu_model:
            specs.gpu_model = val
            specs.gpu_manufacturer = _extract_gpu_vendor(val)

    # RAM
    elif any(term in k for term in ["ram", "memory", "installed memory", "physical memory", "total memory"]):
        if "type" in k or "speed" in k or "channels" in k or "configuration" in k:
            specs.ram_configuration = (f"{specs.ram_configuration or ''} {val}").strip()
        else:
            gb = _parse_memory_gb(val)
            if gb:
                specs.ram_total_gb = gb
            # Extract configuration if mentioned e.g. "16 GB (2x 8GB DDR4)"
            if "(" in val and ")" in val:
                inside = val[val.find("(") + 1 : val.find(")")].strip()
                specs.ram_configuration = inside
                if "2x" in inside.lower():
                    specs.ram_slots_used = 2
                elif "4x" in inside.lower():
                    specs.ram_slots_used = 4
                elif "1x" in inside.lower():
                    specs.ram_slots_used = 1

    # Motherboard
    elif any(term in k for term in ["motherboard", "mainboard", "baseboard", "mobo"]):
        if "manufacturer" in k or "vendor" in k:
            specs.motherboard_manufacturer = val
        else:
            specs.motherboard_model = val
            if not specs.motherboard_manufacturer:
                # Check common vendors
                for vendor in ["ASUS", "MSI", "Gigabyte", "ASRock", "Dell", "HP", "Lenovo", "Acer"]:
                    if vendor.lower() in val.lower():
                        specs.motherboard_manufacturer = vendor
                        break

    # Storage
    elif any(term in k for term in ["storage", "disk", "ssd", "hdd", "nvme", "drive", "hard drive"]):
        cap = _parse_memory_gb(val)
        drive_type = "SSD" if ("ssd" in val.lower() or "nvme" in val.lower()) else "HDD" if "hdd" in val.lower() else "Storage"
        specs.storage_devices.append({
            "model": val,
            "type": drive_type,
            "capacity_gb": cap,
        })
        specs.storage_summary = ", ".join(
            f"{d['model']} ({d['capacity_gb']}GB)" if d.get("capacity_gb") else d["model"]
            for d in specs.storage_devices
        )


def _find_index(headers: List[str], candidates: List[str]) -> Optional[int]:
    for i, h in enumerate(headers):
        for c in candidates:
            if c in h:
                return i
    return None


# ---------------------------------------------------------------------------
# Validation & Public API
# ---------------------------------------------------------------------------

def validate_specs_file(file_content: Union[str, bytes], filename: Optional[str] = None) -> None:
    """
    Validates file extension, maximum file size, and ensures no binary/executable code.
    Raises ValueError or TypeError if invalid.
    """
    size_bytes = len(file_content) if isinstance(file_content, bytes) else len(file_content.encode("utf-8", errors="ignore"))
    if size_bytes > MAX_REPORT_SIZE_BYTES:
        raise ValueError(
            f"Hardware report file exceeds maximum allowable size of {MAX_REPORT_SIZE_BYTES // (1024*1024)} MB "
            f"(received {size_bytes / (1024*1024):.2f} MB)."
        )

    if filename:
        ext = Path(filename).suffix.lower()
        if ext and ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported report file extension '{ext}'. "
                f"Supported extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

    # Check for executable or raw binary headers (e.g. Windows PE 'MZ', ELF, Mach-O)
    prefix = file_content[:4] if isinstance(file_content, bytes) else file_content[:4].encode("latin-1", errors="ignore")
    if prefix.startswith(b"MZ") or prefix.startswith(b"\x7fELF") or prefix.startswith(b"\xfe\xed\xfa"):
        raise ValueError("Security violation: Executable binary files cannot be processed as hardware reports.")

    # Check for null bytes indicative of compiled binaries
    if isinstance(file_content, bytes) and b"\x00" in file_content:
        raise ValueError("Binary file format detected. Please upload text, JSON, CSV, or HTML hardware reports.")


def parse_hardware_report(
    content: Union[str, bytes],
    filename: Optional[str] = None,
    format_hint: Optional[str] = None,
) -> ParsedHardwareSpecs:
    """
    Entry point to parse hardware specification content into a structured ParsedHardwareSpecs object.
    Automatically detects format (JSON, CSV, HTML, TXT) and safely redacts sensitive information.
    """
    validate_specs_file(content, filename=filename)

    if isinstance(content, bytes):
        try:
            text_content = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text_content = content.decode("latin-1")
            except Exception as exc:
                raise ValueError(f"Unable to decode hardware report file as text: {exc}")
    else:
        text_content = content

    parsers: List[BaseReportParser] = [
        JsonReportParser(),
        HtmlReportParser(),
        CsvReportParser(),
        TextReportParser(),
    ]

    # Explicit format hint override if specified
    if format_hint:
        hint = format_hint.lower().strip()
        if hint == "json":
            return JsonReportParser().parse(text_content, filename)
        if hint in ("html", "htm"):
            return HtmlReportParser().parse(text_content, filename)
        if hint == "csv":
            return CsvReportParser().parse(text_content, filename)
        if hint in ("txt", "text"):
            return TextReportParser().parse(text_content, filename)

    for p in parsers:
        if p.can_parse(text_content, filename):
            parsed = p.parse(text_content, filename)
            # If the parser successfully extracted specs or produced warnings, return it
            if not parsed.is_empty() or parsed.parsing_warnings or p is parsers[-1]:
                return parsed

    return TextReportParser().parse(text_content, filename)


def parse_hardware_report_file(file_path: Union[str, Path]) -> ParsedHardwareSpecs:
    """Reads a file path from disk with size and security checks, and parses it."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Hardware specification report not found: {path}")

    size = path.stat().st_size
    if size > MAX_REPORT_SIZE_BYTES:
        raise ValueError(
            f"Hardware report file exceeds maximum allowable size of {MAX_REPORT_SIZE_BYTES // (1024*1024)} MB "
            f"(file is {size / (1024*1024):.2f} MB)."
        )

    with open(path, "rb") as f:
        content = f.read()

    return parse_hardware_report(content, filename=path.name)
