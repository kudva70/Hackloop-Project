"""
FastAPI Server for Repair Before Replace (RepairEx)
Connects the web frontend to the integrated diagnostic pipeline.
Supports all execution modes:
  - image: Single image analysis with Roboflow + OpenCV + expert rules
  - video: Multi-frame video extraction + OpenCV analysis
  - describe: Direct free-text symptom diagnosis
  - demo: Fast offline non-interactive sample diagnosis
"""

from pathlib import Path
import tempfile
import shutil
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

import assistant

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "front-end"
UPLOAD_DIR = BASE_DIR / "outputs" / "web_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Repair Before Replace API", version="2.0.0")

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
    return {"status": "ok", "app": "Repair Before Replace", "supported_devices": ["laptop"]}


@app.post("/api/diagnose")
async def diagnose_device(
    mode: str = Form("describe"),  # "describe", "image", "video", "demo"
    device: Optional[str] = Form("laptop"),
    describe: Optional[str] = Form(None),
    age: Optional[float] = Form(None),
    replacement_cost: Optional[float] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """
    Executes the repair diagnostic pipeline based on the selected mode:
      - demo: Offline scripted sample run
      - describe: Pure symptom text processing
      - image: Uploaded image file + CV + rule engine
      - video: Uploaded video file + frame sampling + CV + rule engine
    """
    temp_path = None
    try:
        is_demo = (mode == "demo")
        image_path = None
        video_path = None

        if is_demo:
            report = assistant.run_pipeline(
                demo=True,
                age_years=age,
                replacement_cost=replacement_cost,
                gemma_backend="template",
            )
            return JSONResponse(content=report)

        # Handle file uploads if present
        if file and file.filename:
            suffix = Path(file.filename).suffix.lower()
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_DIR)
            temp_path = Path(temp_file.name)
            shutil.copyfileobj(file.file, temp_file)
            temp_file.close()

            if mode == "video" or suffix in [".mp4", ".mov", ".avi", ".mkv"]:
                video_path = str(temp_path)
            else:
                image_path = str(temp_path)

        # Run pipeline with inputs (skip_prompts=True for web API)
        report = assistant.run_pipeline(
            image_path=image_path,
            video_path=video_path,
            device_type=device,
            describe=describe,
            age_years=age,
            replacement_cost=replacement_cost,
            gemma_backend="auto",
            demo=False,
            skip_prompts=True,
        )

        return JSONResponse(content=report)

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
        )
    finally:
        # Cleanup uploaded temporary files if needed (or leave for review)
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


# Serve frontend static assets if any exist
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)

