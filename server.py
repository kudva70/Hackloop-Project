"""
FastAPI Server for Repair Before Replace (RepairEx)
Connects the web frontend to the integrated diagnostic pipeline.
Supports all execution modes:
  - image: Single or multiple image analysis with Roboflow + OpenCV + expert rules
  - video: Multi-frame video extraction + OpenCV analysis
  - describe: Direct free-text symptom diagnosis
  - demo: Fast offline non-interactive sample diagnosis
"""

from pathlib import Path
import tempfile
import shutil
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

import assistant
import verdict as verdict_engine
import gamma as explanation_engine
from diagnostic_session import session_manager

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "front-end"
UPLOAD_DIR = BASE_DIR / "outputs" / "web_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Repair Before Replace API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    return FileResponse(str(index_file))


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "app": "Repair Before Replace",
        "supported_devices": ["laptop", "desktop"],
    }


@app.post("/api/diagnose")
async def diagnose_device(
    mode: str = Form("describe"),  # "describe", "image", "video", "demo"
    device: Optional[str] = Form("laptop"),
    describe: Optional[str] = Form(None),
    age: Optional[float] = Form(None),
    replacement_cost: Optional[float] = Form(None),
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    specs_file: Optional[UploadFile] = File(None),
    specs_text: Optional[str] = Form(None),
):
    """
    Executes the repair diagnostic pipeline based on the selected mode:
      - demo: Offline scripted sample run
      - describe: Pure symptom text processing
      - image: Uploaded image file(s) + Hardware Intelligence + CV + rule engine
      - video: Uploaded video file + frame sampling + CV + rule engine
    """
    created_temp_paths = []
    try:
        is_demo = (mode == "demo")
        image_paths = []
        video_path = None

        if is_demo:
            report = assistant.run_pipeline(
                demo=True,
                age_years=age,
                replacement_cost=replacement_cost,
                gemma_backend="template",
            )
            return JSONResponse(content=report)

        # Handle optional hardware specifications file
        specs_path = None
        if specs_file and specs_file.filename:
            import specs_parser
            suffix = Path(specs_file.filename).suffix.lower()
            temp_specs = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_DIR)
            specs_path = Path(temp_specs.name)
            created_temp_paths.append(specs_path)
            shutil.copyfileobj(specs_file.file, temp_specs)
            temp_specs.close()

            # Security and format validation
            specs_parser.validate_specs_file(specs_path.read_bytes(), filename=specs_file.filename)

        # Collect all uploaded files (supports both single file and multi-file forms)
        incoming_files = []
        if files:
            incoming_files.extend([f for f in files if f and f.filename])
        if file and file.filename and file not in incoming_files:
            incoming_files.append(file)

        for up_file in incoming_files:
            suffix = Path(up_file.filename).suffix.lower()
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_DIR)
            t_path = Path(temp_file.name)
            created_temp_paths.append(t_path)
            shutil.copyfileobj(up_file.file, temp_file)
            temp_file.close()

            if mode == "video" or suffix in [".mp4", ".mov", ".avi", ".mkv"]:
                video_path = str(t_path)
            else:
                image_paths.append(str(t_path))

        # Run pipeline with inputs (skip_prompts=True for web API)
        report = assistant.run_pipeline(
            image_path=image_paths[0] if image_paths else None,
            image_paths=image_paths if len(image_paths) > 1 else None,
            video_path=video_path,
            device_type=device,
            describe=describe,
            age_years=age,
            replacement_cost=replacement_cost,
            gemma_backend="auto",
            demo=False,
            skip_prompts=True,
            specs_file=str(specs_path) if specs_path else None,
            specs_text=specs_text,
        )

        return JSONResponse(content=report)

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
        )
    finally:
        # Cleanup uploaded temporary files
        for t_path in created_temp_paths:
            if t_path and t_path.exists():
                try:
                    t_path.unlink()
                except Exception:
                    pass


# ===========================================================================
# ADAPTIVE DIAGNOSTIC SESSION ENDPOINTS
# ===========================================================================

@app.post("/api/session/start")
async def start_session(request: Request):
    """
    Initializes a new stateful diagnostic session.
    Accepts device type, initial symptoms, optional specs, and optional hardware photos.
    Supports both JSON and Multipart Form bodies.
    """
    content_type = request.headers.get("content-type", "")
    dev = "laptop"
    desc = ""
    specs_txt = None
    specs_file_obj = None

    if "application/json" in content_type:
        body = await request.json()
        dev = body.get("device") or "laptop"
        desc = body.get("describe") or ""
        specs_txt = body.get("specs_text")
    else:
        form = await request.form()
        dev = form.get("device") or "laptop"
        desc = form.get("describe") or ""
        specs_txt = form.get("specs_text")
        if "specs_file" in form:
            specs_file_obj = form["specs_file"]

    parsed_specs = None
    if specs_file_obj and hasattr(specs_file_obj, "filename") and specs_file_obj.filename:
        import specs_parser
        content = await specs_file_obj.read()
        specs_parser.validate_specs_file(content, filename=specs_file_obj.filename)
        parsed_specs = specs_parser.parse_hardware_report(content, filename=specs_file_obj.filename)
    elif specs_txt and str(specs_txt).strip():
        import specs_parser
        parsed_specs = specs_parser.parse_hardware_report(specs_txt)

    specs_dict = parsed_specs.to_dict() if parsed_specs else None
    if parsed_specs and parsed_specs.system_type and dev in ("laptop", "desktop", "unknown", None):
        dev = parsed_specs.system_type

    use_gem = False  # Keep API fast and deterministic by default for adaptive questionnaire

    session = session_manager.create_session(
        device_type=dev,
        problem_description=desc,
        imported_specs=specs_dict,
        use_gemini=use_gem,
    )

    first_q = session.get_next_question() if not session.is_complete else None

    return {
        "status": "ok",
        "session_id": session.session_id,
        "state": session.get_state().to_dict(),
        "has_next": first_q is not None,
        "question": first_q.to_dict() if first_q else None,
    }


@app.get("/api/session/{session_id}/next")
async def get_next_question(session_id: str):
    """Retrieves the next adaptive diagnostic question for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Diagnostic session not found")

    q = session.get_next_question()
    return {
        "status": "ok",
        "has_next": q is not None,
        "question": q.to_dict() if q else None,
        "state": session.get_state().to_dict(),
    }


@app.post("/api/session/{session_id}/answer")
async def submit_session_answer(session_id: str, request: Request):
    """Submits an answer to a diagnostic question and updates hypotheses."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Diagnostic session not found")

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        question_id = body.get("question_id")
        answer = body.get("answer")
        text_response = body.get("text_response")
    else:
        form = await request.form()
        question_id = form.get("question_id")
        answer = form.get("answer")
        text_response = form.get("text_response")

    if not question_id or answer is None:
        raise HTTPException(status_code=400, detail="Missing question_id or answer")

    updated_state = session.submit_answer(
        question_id=str(question_id),
        answer=str(answer),
        text_response=str(text_response) if text_response else None,
    )

    next_q = session.get_next_question() if not session.is_complete else None
    return {
        "status": "ok",
        "state": updated_state,
        "has_next": next_q is not None,
        "question": next_q.to_dict() if next_q else None,
    }


@app.get("/api/session/{session_id}/state")
async def get_session_state(session_id: str):
    """Retrieves current session state, hypotheses, and answers."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Diagnostic session not found")

    return {
        "status": "ok",
        "state": session.get_state().to_dict(),
    }


@app.post("/api/session/{session_id}/complete")
async def complete_session(session_id: str, request: Request = None):
    """Finalizes diagnosis, applies repair-vs-replace decision and Gemma explanation."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Diagnostic session not found")

    age = None
    cost = None
    if request:
        try:
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                body = await request.json()
                age = body.get("age")
                cost = body.get("replacement_cost")
            elif "form" in content_type:
                form = await request.form()
                age = float(form.get("age")) if form.get("age") else None
                cost = float(form.get("replacement_cost")) if form.get("replacement_cost") else None
        except Exception:
            pass

    raw_report = session.finalize_diagnosis()
    adapted = assistant.adapt_report_for_verdict(raw_report)
    if session.safety_alert:
        adapted["safety_alert"] = True
        adapted["requires_professional_attention"] = True

    dec = verdict_engine.decide(
        device=session.device_type,
        report=adapted,
        cv_result=None,
        age_years=age,
        replacement_cost=cost,
    )
    if session.safety_alert:
        dec["verdict"] = "SERVICE (technician)"
        dec["safety_hold"] = True

    raw_report["repair_decision"] = dec
    raw_report["hardware_profile"] = session.hardware_profile
    raw_report["imported_specs"] = session.imported_specs
    raw_report["decision_inputs"] = {"device_age_years": age, "replacement_cost_override_inr": cost}

    exp_ctx = {
        "device": session.device_type,
        "symptoms": session.answers,
        "visual_findings": session.visual_findings,
        "diagnosis": raw_report.get("diagnosis", {}),
        "verdict": dec,
        "e_waste": dec.get("e_waste"),
    }
    backend = "auto" if getattr(session, "use_gemini", True) else "template"
    raw_report["explanation"] = explanation_engine.explain(exp_ctx, backend=backend)

    return JSONResponse(content=raw_report)


# Serve frontend static assets if any exist
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
