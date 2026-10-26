# Repair Before Replace — AI Repair Assistant

An end-to-end, terminal-based prototype that helps users investigate a faulty device. It can analyze a device image with a Roboflow-hosted object-detection model, ask the user structured questions about the symptoms and safety risks, and produce a diagnosis with troubleshooting guidance. When configured, Gemini generates the diagnosis using the image and questionnaire answers; if Gemini is unavailable, the rule-based diagnosis path is used as a fallback.

> **Safety notice:** This tool provides advisory troubleshooting, not a certified repair assessment. Never open or handle powered electrical equipment, swollen batteries, or components that may be dangerous. Disconnect power only when it is safe to do so, and contact a qualified technician when a hazard is suspected.

## Workflow

```text
User starts assistant.py
        |
        v
Optional image supplied with --image
        |
        v
 detect_new.py
 Roboflow-hosted object/component detection
        |
        v
 Resolve device type and visual findings
 (manual device selection if detection is unavailable)
        |
        v
 questionaire.py
 Ask device-specific and general symptom/safety questions
        |
        v
 fault_engine.py
 Gemini diagnosis when configured; rule-based fallback otherwise
        |
        v
 Print diagnostic report in terminal
        |
        v
 Save JSON report to outputs/diagnoses/
```

### Module responsibilities

| File | Responsibility |
|---|---|
| `assistant.py` | Main entry point; coordinates detection, questionnaire, diagnosis, terminal output, and JSON export. **Run this file to start the complete workflow.** |
| `detect_new.py` | Sends the supplied image to the configured Roboflow model and returns component detections. |
| `questionaire.py` | Collects symptoms and safety information through terminal questions. The filename intentionally uses the spelling `questionaire.py` to match the import. |
| `fault_engine.py` | Produces a structured diagnosis using Gemini when configured and falls back to the existing rule-based engine when needed. |
| `.env.example` | Template for required API/model configuration. Copy it to `.env` and fill in your own values. |
| `outputs/diagnoses/` | Created automatically when a diagnosis is saved; contains JSON reports. |

## Requirements

- Python 3.10 or newer recommended.
- A Roboflow API key and a valid hosted model ID for image detection.
- A Gemini API key for Gemini-based diagnosis. The rule-based fallback can still be used if Gemini is not configured or a Gemini request fails.
- Internet access for calls to the hosted Roboflow model and Gemini API.

## Setup

Run the commands from the directory containing `assistant.py`.

### 1. Create and activate a virtual environment (recommended)

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install inference-sdk pillow opencv-python google-genai python-dotenv
```

If your existing project has a `requirements.txt`, install it as well:

```bash
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to a new file named `.env` in the same directory as `assistant.py`.

**Windows PowerShell:**

```powershell
Copy-Item .env.example .env
```

**macOS / Linux:**

```bash
cp .env.example .env
```

Open `.env` and replace the example values:

```env
GEMINI_API_KEY=your_actual_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
ROBOFLOW_API_KEY=your_actual_roboflow_api_key
ROBOFLOW_MODEL_ID=your-project/3
```

Use the model ID shown in your Roboflow project, not the example `your-project/3`. Keep `.env` private and add it to `.gitignore` so API keys are never committed to GitHub. Do not put API keys in frontend JavaScript or share them in screenshots.

### 4. Check the project files

Keep these files together in the same project directory:

```text
project-folder/
├── assistant.py
├── detect_new.py
├── questionaire.py
├── fault_engine.py
├── .env
├── .env.example
├── README.md
└── outputs/                 # created automatically when reports are saved
    └── diagnoses/
```

## Run the project

### Option A: Start without an image

```bash
python assistant.py
```

The program asks you to select a device. It then runs the questionnaire and diagnosis workflow. Detection confidence and visual findings will be unavailable when no image is supplied.

### Option B: Diagnose an image

```bash
python assistant.py --image "path/to/device.jpg"
```

Example on Windows:

```powershell
python assistant.py --image "images/laptop.jpg"
```

Example on macOS/Linux:

```bash
python assistant.py --image "images/laptop.jpg"
```

Use the actual path to your image. Supported image formats depend on the detection/image-processing libraries; JPEG and PNG are recommended.

### Option C: Provide the device type explicitly

```bash
python assistant.py --image "images/laptop.jpg" --device laptop
```

The explicit device type overrides automatic device-type resolution in the pipeline.

### Option D: Adjust the detection confidence threshold

```bash
python assistant.py --image "images/laptop.jpg" --confidence 0.40
```

The default confidence threshold is `0.25`. Higher values generally filter out lower-confidence detections; they can also result in fewer detections.

### See available command-line options

```bash
python assistant.py --help
```

## Output

After the questionnaire and diagnosis finish, the program prints a diagnostic report in the terminal. The report can include:

- Device type and detected visual findings
- Primary possible fault and confidence, when supplied by the diagnosis engine
- Severity and recommended action
- Repair complexity, estimated cost, and estimated time when available
- Safety instructions and troubleshooting steps
- Required tools and alternative possible faults, when available

A JSON report is saved automatically under:

```text
outputs/diagnoses/<source>_<timestamp>_diagnosis.json
```

The `outputs/diagnoses/` directory is created automatically. Each run produces a timestamped report so prior reports are not intentionally overwritten.

## Troubleshooting

| Problem | What to check |
|---|---|
| `ROBOFLOW_API_KEY is not set` | Confirm that `.env` exists beside the project files and contains your Roboflow API key. Check that `python-dotenv` is installed. |
| `ROBOFLOW_MODEL_ID is not set` | Set `ROBOFLOW_MODEL_ID` to the correct model ID from your Roboflow project. |
| Roboflow request fails | Check internet access, API key validity, model ID, and whether the hosted model is available. |
| Gemini diagnosis does not run | Verify `GEMINI_API_KEY`, model availability, internet access, and the installed `google-genai` package. The rule-based fallback may be used if the request fails. |
| Image file not found | Check the path and spelling. Use quotes around paths that contain spaces. |
| No visual detections | Try a clearer, well-lit image, verify the correct model is configured, or provide `--device` to continue with manual device selection. |
| `ModuleNotFoundError` | Activate the intended virtual environment and install the dependencies listed above. Ensure all four Python files are in the same directory. |

## Current limitations

- The configured Roboflow model in `detect_new.py` is a **laptop-component model**. It should not be assumed to identify fans, bulbs, or switchboards unless you configure a model trained for those devices and adapt the device-resolution logic.
- The questionnaire is currently terminal-based, not a web form.
- Diagnosis and cost/time estimates are advisory and may be uncertain; they should not be treated as confirmed measurements or a substitute for professional inspection.
- The project has not been validated against every device, fault type, or hardware configuration. Test with safe sample images and verify API setup before relying on the output.

## Git safety

Add these entries to `.gitignore` if they are not already present:

```gitignore
.env
.venv/
__pycache__/
outputs/
```

Review generated reports before sharing them, as they may contain user-provided symptom descriptions or other information from a diagnostic session.
