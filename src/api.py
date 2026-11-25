"""Step 6 - API (FastAPI) + web UI host.

    uvicorn src.api:app --host 0.0.0.0 --port 8000

Swagger docs are at /docs. The UI is served from / .
"""
import os
import re
import socket
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config as C
from . import db, retrain
from . import model as M
from .preprocessing import AudioError, fit_length, list_clips, load_waveform
from .prediction import get_predictor

STARTED = time.time()
HOST = os.getenv("HOSTNAME") or socket.gethostname()
LABEL_RE = re.compile(r"[^a-z0-9_]+")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    get_predictor()          # load the model once at start-up, not on the first request
    yield


app = FastAPI(title="ForestEar API", version="1.0.0", lifespan=lifespan,
              description="Classify 5-second sound clips, upload new labelled clips and retrain the model.")


def _clean_label(raw: str) -> str:
    label = LABEL_RE.sub("_", raw.lower().strip()).strip("_")
    if not 2 <= len(label) <= 30:
        raise HTTPException(422, "Label must be 2-30 characters (letters, digits, spaces or underscores).")
    return label


async def _read(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > C.MAX_UPLOAD_MB * 1024 * 1024:
        raise AudioError(f"File is larger than {C.MAX_UPLOAD_MB} MB.")
    if not data:
        raise AudioError("File is empty.")
    return data


# ----------------------------------------------------------------- health
@app.get("/health", tags=["ops"])
def health():
    p = get_predictor()
    return {"status": "ok", "model_loaded": p.ready, "model_version": p.entry["version"] if p.ready else None,
            "uptime_s": round(time.time() - STARTED, 1), "host": HOST}


# ------------------------------------------------------------- prediction
@app.post("/api/predict", tags=["model"])
async def predict(file: UploadFile = File(..., description="WAV / FLAC / OGG clip (any length, any sample rate)")):
    """Predict the sound class of ONE clip."""
    try:
        data = await _read(file)
        result = get_predictor().predict(data)
    except AudioError as exc:
        raise HTTPException(422, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    db.log_prediction(result["label"], result["confidence"], result["latency_ms"], result["model_version"], HOST)
    return result


# ------------------------------------------------------------- data upload
@app.post("/api/upload", tags=["retraining"])
async def upload(label: str = Form(..., description="Class of ALL files in this upload"),
                 files: List[UploadFile] = File(..., description="Many WAV/FLAC/OGG clips")):
    """Bulk-upload labelled clips. They are validated, normalised to 16 kHz mono WAV, saved
    under data/uploads/<label>/ and recorded in SQLite. May start retraining automatically."""
    label = _clean_label(label)
    out_dir = C.UPLOAD_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted, rejected = 0, []
    for f in files:
        try:
            y = fit_length(load_waveform(await _read(f)))
            dest = out_dir / f"{uuid.uuid4().hex[:12]}.wav"
            sf.write(dest, y, C.SAMPLE_RATE, subtype="PCM_16")
            db.add_upload(label, f.filename, dest)
            accepted += 1
        except AudioError as exc:
            rejected.append({"file": f.filename, "reason": str(exc)})
    run_id = retrain.maybe_auto_retrain() if accepted else None
    return {"label": label, "accepted": accepted, "rejected": rejected, "pending_uploads": db.pending_uploads(),
            "auto_retrain_started": run_id is not None, "run_id": run_id}


@app.get("/api/uploads", tags=["retraining"])
def uploads():
    return {"classes": db.upload_counts(), "pending": db.pending_uploads()}


# --------------------------------------------------------------- retraining
@app.post("/api/retrain", tags=["retraining"], status_code=202)
def trigger_retrain(mode: str = "finetune"):
    """Start training in the background. mode=finetune (default) starts from the current model;
    mode=scratch trains a fresh network on train + uploads."""
    try:
        run_id = retrain.start("manual", mode)
    except retrain.RetrainBusy as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return {"run_id": run_id, "message": "Training started."}


@app.get("/api/retrain/status", tags=["retraining"])
def retrain_status():
    runs = db.get_runs(8)
    return {"running": retrain.is_running(), "latest": runs[0] if runs else None, "history": runs,
            "recommendation": retrain.retrain_recommendation(), "auto_retrain": C.AUTO_RETRAIN,
            "auto_retrain_min_clips": C.AUTO_RETRAIN_MIN_CLIPS}


# --------------------------------------------------------------- monitoring
@app.get("/api/status", tags=["ops"])
def status():
    p = get_predictor()
    p.reload()
    e = p.entry or {}
    m = e.get("metrics", {})
    return {
        "service": {"status": "up", "host": HOST, "pid": os.getpid(), "uptime_s": round(time.time() - STARTED, 1),
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(STARTED))},
        "model": {"loaded": p.ready, "version": e.get("version"), "file": e.get("file"), "classes": e.get("classes", []),
                  "created": e.get("created"), "trigger": e.get("trigger"), "params": e.get("params"),
                  "metrics": {k: m.get(k) for k in ("accuracy", "loss", "precision_macro", "recall_macro", "f1_macro",
                                                     "f1_weighted", "top2_accuracy", "roc_auc_macro", "n_samples")}},
        "traffic": db.prediction_stats(),
        "data": {"pending_uploads": db.pending_uploads(), "uploads": db.upload_counts()},
        "retraining": {"running": retrain.is_running(), "recommendation": retrain.retrain_recommendation(),
                       "auto_retrain": C.AUTO_RETRAIN, "min_clips": C.AUTO_RETRAIN_MIN_CLIPS},
        "thresholds": {"low_confidence": C.LOW_CONFIDENCE, "drift_window": C.DRIFT_WINDOW},
    }


@app.get("/api/model", tags=["model"])
def model_info():
    reg = M.load_registry()
    cur = M.current_entry(reg)
    return {"current": reg["current"], "versions": [{k: v for k, v in x.items() if k not in ("per_class",)}
                                                   for x in reversed(reg["versions"])],
            "per_class": cur.get("per_class") if cur else None}


@app.get("/api/insights", tags=["data"])
def insights():
    if not C.INSIGHTS_PATH.exists():
        raise HTTPException(404, "Run `python -m src.insights` to generate dataset insights.")
    return JSONResponse(content=__import__("json").loads(C.INSIGHTS_PATH.read_text()))


@app.get("/api/samples", tags=["data"])
def samples():
    """One held-out test clip per class so the UI can offer 'try a sample'."""
    seen, out = set(), []
    for p, c in list_clips(C.TEST_DIR):
        if c not in seen:
            seen.add(c)
            out.append({"label": c, "url": f"/samples/{c}/{p.name}"})
    return out


# --------------------------------------------------------------------- UI
if C.TEST_DIR.exists():
    app.mount("/samples", StaticFiles(directory=C.TEST_DIR), name="samples")
app.mount("/static", StaticFiles(directory=C.UI_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(C.UI_DIR / "index.html")
