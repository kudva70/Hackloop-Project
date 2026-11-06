"""
Unit tests for Hardware Specification Import System in Repair Before Replace.

Verifies:
1. Parsing of TXT (systeminfo, dxdiag, generic key-value), JSON, CSV, and HTML reports.
2. Sanitization of sensitive data (product keys, MAC addresses, private user paths).
3. Handling of missing specifications (diagnosis functions normally).
4. Handling of incomplete reports (no errors on partial specs).
5. Conflict resolution & detection (visual vs. report discrepancies flagged for review).
6. File validation & security (size limit >5MB rejected, invalid extensions rejected, binary execution blocked).
7. Strict data-source separation (visual detection does not claim spec metrics).
8. Both Laptop and Desktop PC profiles supported.
"""

import unittest
from pathlib import Path
import tempfile
import json
import os

from specs_parser import (
    ParsedHardwareSpecs,
    SpecsSanitizer,
    parse_hardware_report,
    parse_hardware_report_file,
    validate_specs_file,
    JsonReportParser,
    CsvReportParser,
    HtmlReportParser,
    TextReportParser,
    MAX_REPORT_SIZE_BYTES,
)
from hardware_profile import HardwareProfile, ComponentItem, VisibilityStatus
import assistant


class TestSpecsSanitizer(unittest.TestCase):
    """Test sanitization of sensitive personal & license information."""

    def test_sanitize_product_key(self):
        sample = "Windows 11 Pro Key: VK7JG-NPHTM-C97JM-9MPGT-3V66T activated"
        clean, redacted = SpecsSanitizer.sanitize_text(sample)
        self.assertNotIn("VK7JG-NPHTM-C97JM-9MPGT-3V66T", clean)
        self.assertIn("[REDACTED_PRODUCT_KEY]", clean)
        self.assertIn("Windows Product Key", redacted)

    def test_sanitize_mac_address(self):
        sample = "Network Adapter Physical Address: 00-1A-2B-3C-4D-5E connected"
        clean, redacted = SpecsSanitizer.sanitize_text(sample)
        self.assertNotIn("00-1A-2B-3C-4D-5E", clean)
        self.assertIn("[REDACTED_MAC_ADDRESS]", clean)
        self.assertIn("MAC Address", redacted)

    def test_sanitize_user_path(self):
        sample = "Report generated at C:\\Users\\johndoe\\Documents\\hwinfo.txt"
        clean, redacted = SpecsSanitizer.sanitize_text(sample)
        self.assertNotIn("johndoe", clean)
        self.assertIn("C:\\Users\\[REDACTED_USER]", clean)
        self.assertIn("User Directory Path", redacted)


class TestSpecsFileValidation(unittest.TestCase):
    """Test file security, extension filtering, and size boundary constraints."""

    def test_valid_file_types(self):
        # Should not raise for valid extensions
        for ext in [".txt", ".json", ".csv", ".html", ".htm"]:
            validate_specs_file("CPU: Intel Core i7", f"report{ext}")

    def test_unsupported_extension_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_specs_file("CPU: Intel Core i7", "report.exe")
        self.assertIn("Unsupported report file extension", str(ctx.exception))

    def test_oversized_file_rejected(self):
        large_content = "A" * (MAX_REPORT_SIZE_BYTES + 100)
        with self.assertRaises(ValueError) as ctx:
            validate_specs_file(large_content, "specs.txt")
        self.assertIn("exceeds maximum allowable size", str(ctx.exception))

    def test_binary_executable_rejected(self):
        # Fake Windows MZ PE header
        binary_payload = b"MZ\x90\x00\x03\x00\x00\x00"
        with self.assertRaises(ValueError) as ctx:
            validate_specs_file(binary_payload, "malicious.txt")
        self.assertIn("Security violation: Executable binary", str(ctx.exception))


class TestFormatParsers(unittest.TestCase):
    """Test format-specific extraction for TXT, JSON, CSV, and HTML."""

    def test_parse_json_desktop_specs(self):
        data = {
            "device_type": "desktop",
            "os": {"name": "Windows 11 Pro", "version": "23H2"},
            "cpu": {"name": "AMD Ryzen 7 7800X3D", "cores": 8, "threads": 16},
            "gpu": {"name": "NVIDIA GeForce RTX 4080", "vendor": "NVIDIA"},
            "ram": {"total": "32 GB", "configuration": "2x 16GB DDR5-6000", "slots_used": 2},
            "motherboard": {"manufacturer": "ASUS", "model": "ROG STRIX B650E-F"},
            "storage": [
                {"model": "Samsung 990 PRO", "size": "2048 GB", "type": "NVMe SSD"},
            ]
        }
        specs = parse_hardware_report(json.dumps(data), filename="hw.json")
        self.assertEqual(specs.system_type, "desktop")
        self.assertEqual(specs.cpu_manufacturer, "AMD")
        self.assertIn("7800X3D", specs.cpu_model)
        self.assertEqual(specs.cpu_cores, 8)
        self.assertEqual(specs.cpu_threads, 16)
        self.assertEqual(specs.gpu_manufacturer, "NVIDIA")
        self.assertIn("RTX 4080", specs.gpu_model)
        self.assertEqual(specs.ram_total_gb, 32.0)
        self.assertEqual(specs.motherboard_manufacturer, "ASUS")
        self.assertEqual(specs.motherboard_model, "ROG STRIX B650E-F")
        self.assertEqual(len(specs.storage_devices), 1)

    def test_parse_txt_dxdiag_laptop_specs(self):
        sample = """
------------------
System Information
------------------
Operating System: Windows 10 Home 64-bit
System Manufacturer: Dell Inc.
System Model: XPS 15 9500
Processor: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz (12 CPUs), ~2.6GHz
Memory: 16384MB RAM
Card name: NVIDIA GeForce GTX 1650 Ti
        """
        specs = parse_hardware_report(sample, filename="dxdiag.txt")
        self.assertEqual(specs.system_type, "laptop")
        self.assertIn("Windows 10", specs.os_name)
        self.assertEqual(specs.cpu_manufacturer, "Intel")
        self.assertIn("i7-10750H", specs.cpu_model)
        self.assertEqual(specs.cpu_threads, 12)
        self.assertEqual(specs.ram_total_gb, 16.0)
        self.assertEqual(specs.gpu_manufacturer, "NVIDIA")
        self.assertIn("GTX 1650 Ti", specs.gpu_model)

    def test_parse_csv_specs(self):
        sample = """Component,Specification
Device Type,Desktop PC
Processor,Intel Core i5-12600K
Memory,32 GB DDR4
Graphics,AMD Radeon RX 6700 XT
Motherboard,MSI PRO Z690-A
Storage,1000 GB Crucial P5 Plus SSD
"""
        specs = parse_hardware_report(sample, filename="specs.csv")
        self.assertEqual(specs.system_type, "desktop")
        self.assertIn("12600K", specs.cpu_model)
        self.assertEqual(specs.ram_total_gb, 32.0)
        self.assertEqual(specs.gpu_manufacturer, "AMD")
        self.assertIn("RX 6700 XT", specs.gpu_model)
        self.assertEqual(specs.motherboard_manufacturer, "MSI")

    def test_parse_html_report(self):
        sample = """
<!DOCTYPE html>
<html>
<head><title>Speccy Hardware Report</title></head>
<body>
<h1>Hardware Summary</h1>
<table>
  <tr><td>Operating System</td><td>Windows 11 Home 64-bit</td></tr>
  <tr><td>CPU</td><td>AMD Ryzen 5 5600 6-Core</td></tr>
  <tr><td>RAM</td><td>16.0GB Dual-Channel DDR4</td></tr>
  <tr><td>Motherboard</td><td>Gigabyte B550M DS3H</td></tr>
  <tr><td>Graphics</td><td>NVIDIA GeForce RTX 3060 12GB</td></tr>
  <tr><td>Storage</td><td>500GB Western Digital WDC SSD</td></tr>
</table>
</body>
</html>
        """
        specs = parse_hardware_report(sample, filename="speccy.html")
        self.assertEqual(specs.source_format, "html")
        self.assertIn("Windows 11", specs.os_name)
        self.assertEqual(specs.cpu_manufacturer, "AMD")
        self.assertIn("Ryzen 5 5600", specs.cpu_model)
        self.assertEqual(specs.cpu_cores, 6)
        self.assertEqual(specs.ram_total_gb, 16.0)
        self.assertEqual(specs.motherboard_manufacturer, "Gigabyte")
        self.assertIn("RTX 3060", specs.gpu_model)


class TestIncompleteAndMissingSpecs(unittest.TestCase):
    """Test handling of incomplete reports and missing specifications."""

    def test_partial_specs_cpu_only(self):
        sample = "CPU: Intel Core i9-13900K"
        specs = parse_hardware_report(sample)
        self.assertEqual(specs.cpu_manufacturer, "Intel")
        self.assertIn("13900K", specs.cpu_model)
        self.assertIsNone(specs.ram_total_gb)
        self.assertIsNone(specs.gpu_model)
        self.assertIsNone(specs.motherboard_model)
        self.assertEqual(specs.storage_devices, [])

    def test_empty_report_no_failure(self):
        sample = "# Notes only with no recognized hardware keys\nUser note: PC makes noise"
        specs = parse_hardware_report(sample)
        self.assertTrue(specs.is_empty())


class TestHardwareProfileDataProvenanceAndConflicts(unittest.TestCase):
    """Verify data-source separation and conflict tracking."""

    def test_data_source_separation_ram(self):
        profile = HardwareProfile(device_type="desktop")
        # Visual detection finds RAM module with bounding box
        profile.add_component_detection(
            category="ram",
            name="RAM Modules (DIMM)",
            confidence=0.85,
            bounding_box=[100, 150, 140, 220],
            source="roboflow",
        )
        visual_ram = profile.components["ram"]
        self.assertEqual(visual_ram.visibility_status, VisibilityStatus.DETECTED.value)
        self.assertEqual(visual_ram.detection_source, "roboflow")
        self.assertIsNone(visual_ram.specs_details)

        # Now import specs report with 32 GB RAM
        parsed = ParsedHardwareSpecs(
            ram_total_gb=32.0,
            ram_configuration="2x 16GB DDR4",
            ram_slots_used=2,
            source_format="txt",
        )
        profile.import_specs(parsed)

        updated_ram = profile.components["ram"]
        # Physical presence is still DETECTED from photo, source is roboflow
        self.assertEqual(updated_ram.visibility_status, VisibilityStatus.DETECTED.value)
        self.assertEqual(updated_ram.detection_source, "roboflow")
        self.assertEqual(updated_ram.verification_status, "verified_with_report")
        # Spec metric is strictly preserved in specs_details
        self.assertIsNotNone(updated_ram.specs_details)
        self.assertEqual(updated_ram.specs_details["capacity_gb"], 32.0)
        self.assertIn("Source separation: Specification details verified via imported report", updated_ram.notes)

    def test_device_type_conflict_flagged(self):
        profile = HardwareProfile(device_type="desktop")
        parsed = ParsedHardwareSpecs(
            system_type="laptop",
            cpu_model="Intel Core i7-1165G7",
        )
        profile.import_specs(parsed)
        self.assertEqual(len(profile.specs_conflicts), 1)
        conflict = profile.specs_conflicts[0]
        self.assertEqual(conflict["field"], "device_type")
        self.assertEqual(conflict["visual_value"], "desktop")
        self.assertEqual(conflict["specs_value"], "laptop")

    def test_unobserved_component_imported_cleanly(self):
        profile = HardwareProfile(device_type="desktop")
        # CPU is not visually observed under heatsink
        self.assertEqual(profile.components["cpu"].visibility_status, VisibilityStatus.NOT_DETECTED.value)

        parsed = ParsedHardwareSpecs(
            cpu_manufacturer="Intel",
            cpu_model="Core i7-12700K",
            cpu_cores=12,
        )
        profile.import_specs(parsed)

        cpu_comp = profile.components["cpu"]
        self.assertEqual(cpu_comp.visibility_status, VisibilityStatus.IMPORTED_FROM_REPORT.value)
        self.assertEqual(cpu_comp.detection_source, "specs_report")
        self.assertEqual(cpu_comp.verification_status, "verified_from_report")
        self.assertTrue(cpu_comp.is_present)


class TestAssistantEndToEndIntegration(unittest.TestCase):
    """Verify assistant runs with optional specs and reports properly without regressions."""

    def test_run_pipeline_with_specs_text(self):
        specs_input = """
OS: Windows 11 Pro
CPU: AMD Ryzen 9 5900X (12 cores)
RAM: 64 GB (2x 32GB DDR4)
GPU: NVIDIA GeForce RTX 3080
Motherboard: ASUS ROG CROSSHAIR VIII
"""
        report = assistant.run_pipeline(
            demo=True,
            specs_text=specs_input,
            device_type="desktop",
        )
        self.assertIsNotNone(report.get("imported_specs"))
        specs = report["imported_specs"]
        self.assertEqual(specs["cpu_manufacturer"], "AMD")
        self.assertEqual(specs["ram_total_gb"], 64.0)
        self.assertIn("3080", specs["gpu_model"])
        self.assertIn("repair_decision", report)

    def test_run_pipeline_without_specs_unaffected(self):
        report = assistant.run_pipeline(
            demo=True,
            specs_file=None,
            specs_text=None,
        )
        self.assertIsNone(report.get("imported_specs"))
        self.assertIn("repair_decision", report)
        self.assertEqual(report["repair_decision"]["verdict"], "REPAIR (DIY)")


if __name__ == "__main__":
    unittest.main()

