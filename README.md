# Repair Before Replace — AI Repair Assistant

> **Empowering people to diagnose electronic faults safely, weigh repair economics transparently, and reduce global e-waste.**

Many electronic devices are replaced prematurely not because they are unfixable, but because owners cannot identify the fault or determine whether a repair is worth the cost. **Repair Before Replace** is an AI-assisted hardware diagnostic system that evaluates faulty devices through photos, videos, or text descriptions. It identifies likely faults, provides tiered safety-first troubleshooting instructions, estimates repair complexity and cost, and delivers an objective verdict: **REPAIR (DIY)**, **SERVICE (technician)**, or **REPLACE (device)**.

---

## Architecture & Pipeline

The system is architected as an explainable, deterministic pipeline where **safety always beats diagnosis, and LLMs never decide financial or repair verdicts**.

```
[ Photo / Video / Text Input ]
              │
              ▼
    1. detect_new.py (Roboflow Inference)
       - Detects laptop components & extracts bounding boxes
       - Multi-frame sampling for video (sharpness-ranked)
              │
              ▼
    2. cv.py (Classical OpenCV Analysis)
       - Crops to detected device bounding box (4% padding)
       - Scorch/burn filtering (eliminates wood-tone desk false positives)
       - Detects cracks, rust, and image quality warnings
              │
              ▼
    3. questionaire.py (Symptom Collection)
       - Pre-fills answers from free-text descriptions
       - Prompts follow-up confirmation for minor visual discoloration
       - Early safety halt if confirmed hard hazards are detected
              │
              ▼
    4. fault_engine.py (Deterministic Fault Engine)
       - DEVICE_REGISTRY based (laptop in Phase 1)
       - Whole-word keyword & co-occurrence scoring
       - Strict fallback to "Inconclusive" (never false matches)
       - Optional Gemini API selection constrained to known rule IDs
              │
              ▼
    5. verdict.py (Deterministic Decision Engine)
       - Three verdicts: REPAIR (DIY), SERVICE (technician), REPLACE (device)
       - Cost-ratio (<40% / >60%) and lifespan depreciation thresholds
       - Traceable decision factors and user-overridable assumptions
       - E-waste impact assessment (avoided kg or take-back advice)
              │
              ▼
    6. gamma.py (Plain-Language Explanation)
       - Explains existing diagnosis facts (Ollama local Gemma, Gemini API, or built-in template)
       - Never invents faults, parts, or prices
              │
              ▼
    7. assistant.py (Integrated CLI & Report Generator)
       - Orchestrates pipeline, renders terminal report, and saves JSON artifact
```

---

## Core Principles & Safety Protocol

1. **Safety Beats Everything**:
   - Free-text descriptions and questionnaire answers are scanned for hard hazards (`smoke`, `spark`, `burning smell`, `shock`, `swollen`, `bulging`, `liquid`, `spill`, `wet`, `hot to touch`).
   - When a hard hazard is confirmed, the assistant **halts DIY troubleshooting immediately**, issues critical stop-using advice (do not charge, do not open casing), forces the verdict to **`SERVICE (technician)`**, and generates a complete report.
   - "Not sure" is treated as a caution, not a hazard floor.

2. **Tiered Guidance**:
   - Every diagnosis splits actions into two distinct tiers:
     - **Safe Checks (Tier 1)**: Non-invasive, external checks (HDMI to TV, flashlight test, Caps Lock light, charger testing).
     - **Advanced Steps (Tier 2)**: Opening the chassis, RAM oxidation cleaning with pencil eraser, thermal repasting.
   - For `CRITICAL` severity or safety-hold faults (swollen battery, liquid spill, motherboard surge), **Tier 2 is completely empty** to protect users.

3. **Deterministic Economic Verdicts**:
   - **`REPAIR (DIY)`**: Economical repair with safe, beginner/intermediate steps.
   - **`SERVICE (technician)`**: Requires specialized tools, microsoldering, or involves physical/electrical hazards.
   - **`REPLACE (device)`**: Parts repair cost exceeds 60% of replacement price (or exceeds 40% when the device is past 70% of its typical lifespan).
   - Component replacement (e.g. LCD screen or battery) is explicitly distinguished from whole-device replacement.

4. **E-Waste Metrics**:
   - For `REPAIR` or `SERVICE`, calculates **estimated e-waste avoided** (e.g. `~2.1 kg` based on standard laptop mass assumption).
   - For `REPLACE`, provides **responsible take-back advice** (authorized e-waste recycler, never bin, remove/wipe personal storage data).
   - All figures are transparently labelled as **estimates**.

5. **Graceful Degradation (Zero API Keys Needed)**:
   - If Roboflow, Gemini, or Ollama are unavailable, each emits a single warning line and falls back to deterministic rule matching and offline template explanations.

---

## Getting Started

### Installation

Clone the repository and install the minimal dependencies:

```bash
git clone https://github.com/adityakudva/Repair-Before-Replace.git
cd Repair-Before-Replace
pip install -r requirements.txt
```

### Environment Configuration (Optional)

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Key | Description | Optional? |
|---|---|---|
| `GEMINI_API_KEY` | Google AI Studio API key for Gemini fault selection & explanation | Yes (falls back to rules/template) |
| `GEMINI_MODEL` | Gemini model name (default: `gemini-2.5-flash`) | Yes |
| `ROBOFLOW_API_KEY` | Roboflow API key for component detection | Yes (falls back to manual selection) |
| `ROBOFLOW_MODEL_ID`| Roboflow model ID | Yes |
| `OLLAMA_URL` | Local Ollama endpoint (default: `http://localhost:11434`) | Yes |
| `GEMMA_MODEL` | Local Gemma model (e.g. `gemma3:4b`) | Yes |

---

## Usage

### 1. Offline Demo Mode (Zero Network Calls)

Runs an end-to-end diagnosis using offline scripted inputs and built-in templates:

```bash
python assistant.py --demo
```

### 2. Image Input

Diagnose a laptop from a photo:

```bash
python assistant.py --image path/to/laptop.jpg
```

### 3. Video Input

Extracts 3–5 sharpest frames, performs multi-frame damage analysis, and merges visual findings:

```bash
python assistant.py --video path/to/laptop_video.mp4
```

### 4. Free-Text Description Only

Run diagnosis directly from symptoms:

```bash
python assistant.py --describe "laptop turns on but screen stays black, fan is spinning"
```

### 5. Interactive Combination with Age and Cost Overrides

```bash
python assistant.py --image laptop.jpg --age 3.5 --replacement-cost 55000
```

*(If `--age` or `--replacement-cost` are omitted, the assistant prompts interactively: press `Enter` to use defaults).*

---

## Testing

Run the unit test suite:

```bash
python -m unittest test_fault_engine.py -v
```

Run the device registry & unsupported device smoke tests:

```bash
python test_all_devices.py
```

---

## Roadmap: Phase 1 & Registry-Based Expansion

- [x] **Phase 1 (Current Scope - Laptops & PCs)**:
  - Full depth for laptop hardware components (RAM, SSD, screen panel, battery, charging port, thermal fan, motherboard surge, keyboard/touchpad).
  - OpenCV crop-box damage heuristics (scorch marks with wood-tone rejection, corrosion, crack lines).
  - Deterministic 3-way verdicts, e-waste metrics, and safety-stop guarantees.
  - Device Registry architecture (`DEVICE_REGISTRY = {"laptop": [...]}`).
  - Unsupported device protection: entering household appliances (fans, bulbs, TVs, microwaves, phones) returns structured polite notices with electrical safety advice, preventing misdiagnoses or crashes.

- [ ] **Phase 2 (Registry Expansion)**:
  - Add smartphones and tablets (`DEVICE_REGISTRY["smartphone"]`).
  - Add PC desktop peripherals and monitors.
  - Multi-language support for regional repair communities.
  - Integration with local open-source repair manual APIs (e.g., iFixit guide links).

---

## License

MIT License. Designed for sustainable electronics and open-source repairability.
