"""
Gemma layer. Gemma only EXPLAINS results that OpenCV + the fault engine + verdict.py
already produced; it never decides the verdict or invents faults, costs or parts.

Backends (auto order):
  1. Local Ollama  ->  `ollama pull gemma3:4b`   (offline, supports image input)
  2. Gemini API    ->  set GEMINI_API_KEY (free key from Google AI Studio), model gemma-3-27b-it
  3. Built-in template (always works, so the app never breaks)
"""
import base64
import json
import os

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("GEMMA_MODEL", "gemma3:4b")
API_MODEL = os.environ.get("GEMMA_API_MODEL", "gemma-3-27b-it")


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _ollama_up():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
        return r.ok and any(OLLAMA_MODEL.split(":")[0] in m.get("name", "") for m in r.json().get("models", []))
    except Exception:
        return False


def _ask_ollama(prompt, image_path=None):
    body = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.2, "num_predict": 300}}
    if image_path:
        body["images"] = [_b64(image_path)]
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=180)
    r.raise_for_status()
    return r.json()["response"].strip()


def _ask_api(prompt, image_path=None):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    parts = [{"text": prompt}]
    if image_path:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": _b64(image_path)}})
    # Gemma on the Gemini API has no separate system prompt: everything goes in the user turn.
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{API_MODEL}:generateContent",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json={"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.2, "maxOutputTokens": 400}},
        timeout=60)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def _ask(prompt, image_path=None, backend="auto"):
    errors = []
    if backend in ("auto", "ollama") and _ollama_up():
        try:
            return _ask_ollama(prompt, image_path), f"Gemma ({OLLAMA_MODEL}, local)"
        except Exception as e:
            errors.append(f"ollama: {e}")
    if backend in ("auto", "api") and os.environ.get("GEMINI_API_KEY"):
        try:
            return _ask_api(prompt, image_path), f"Gemma ({API_MODEL}, API)"
        except Exception as e:
            errors.append(f"api: {e}")
    raise RuntimeError("; ".join(errors) or "no Gemma backend available")


def template_explanation(ctx):
    v, d = ctx["verdict"], ctx["diagnosis"]
    lines = [f"Recommendation: {v['verdict']}."]
    lines.append(f"Most likely issue: {d['fault_title']}. " + " ".join(v["reasons"][:2]))
    if v.get("estimated_saving"):
        lines.append(f"Repairing instead of replacing could save about Rs {v['estimated_saving']:,}.")
    if v["safety_hold"]:
        lines.append("Safety first: unplug the device and do not power it on until a technician has checked it.")
    return " ".join(lines)


def explain(ctx, backend="auto"):
    """ctx: dict with device, symptoms, visual_findings, diagnosis (primary), verdict."""
    prompt = (
        "You are explaining a device repair assessment to a non-technical person.\n"
        "Use ONLY the facts in the JSON below. Do not add faults, parts, prices or advice that are not in it. "
        "Write 3 to 5 short plain sentences. Start with the recommendation. "
        "If safety_hold is true, end with a clear instruction to unplug the device and see a technician.\n\n"
        f"FACTS:\n{json.dumps(ctx, indent=2, default=str)}"
    )
    try:
        text, source = _ask(prompt, backend=backend)
        return {"text": text, "source": source}
    except Exception as e:
        return {"text": template_explanation(ctx), "source": "built-in template", "note": f"Gemma unavailable ({e})"}


def second_opinion(image_path, backend="auto"):
    """Optional: Gemma looks at the photo and lists visible damage. Shown separately, never used for the verdict."""
    prompt = ("Look at this photo of an electronic or electrical device. In at most 3 short bullet points, "
              "list only damage you can clearly SEE (burn marks, corrosion, cracks, bulging parts, dust build-up). "
              "If you see none, say 'No clear visible damage.' Do not guess about things you cannot see.")
    try:
        text, source = _ask(prompt, image_path=image_path, backend=backend)
        return {"text": text, "source": source}
    except Exception as e:
        return {"text": None, "source": None, "note": f"Gemma vision unavailable ({e})"}