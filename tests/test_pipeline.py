"""Fast tests (~20 s). Run: pytest -q     Slow end-to-end retraining: RUN_SLOW=1 pytest -q -k retrain"""
import io
import os
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from src import config as C
from src import preprocessing as P
from src.api import app

TEST_CLIPS = P.list_clips(C.TEST_DIR)


def sample(label):
    return next(p for p, c in TEST_CLIPS if c == label)


def wav_bytes(y, sr=C.SAMPLE_RATE):
    b = io.BytesIO(); sf.write(b, y, sr, format="WAV"); return b.getvalue()


# ------------------------------------------------------------ preprocessing
def test_features_have_fixed_shape():
    f = P.features_from_source(str(sample("rain")))
    assert f.shape == (C.N_MELS, C.N_FRAMES) and np.isfinite(f).all()


def test_fit_length_pads_and_keeps_loudest_window():
    assert P.fit_length(np.ones(1000, np.float32)).shape == (C.N_SAMPLES,)
    y = np.zeros(C.SAMPLE_RATE * 12, np.float32); y[8 * C.SAMPLE_RATE:9 * C.SAMPLE_RATE] = 0.9
    out = P.fit_length(y)
    assert out.shape == (C.N_SAMPLES,) and np.abs(out).max() > 0.5


def test_resampling_and_stereo_are_handled():
    t = np.linspace(0, 2, 2 * 44100, endpoint=False)
    stereo = np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1).astype(np.float32)
    b = io.BytesIO(); sf.write(b, stereo, 44100, format="WAV")
    assert P.features_from_source(b.getvalue()).shape == (C.N_MELS, C.N_FRAMES)


@pytest.mark.parametrize("bad", [b"not audio at all", wav_bytes(np.zeros(16000, np.float32)), wav_bytes(np.ones(100, np.float32))])
def test_bad_audio_is_rejected(bad):
    with pytest.raises(P.AudioError):
        P.load_waveform(bad)


def test_augment_keeps_shape_and_range():
    f = P.features_from_source(str(sample("dog")))
    a = P.augment(f, np.random.default_rng(0))
    assert a.shape == f.shape and a.min() >= C.DB_FLOOR


def test_validation_split_is_by_fold_and_stable():
    from pathlib import Path
    assert P.is_validation(Path("4-1234-A-0.wav")) and not P.is_validation(Path("2-1234-A-0.wav"))
    assert P.is_validation(Path("abc.wav")) == P.is_validation(Path("abc.wav"))


# --------------------------------------------------------------------- API
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["model_loaded"] is True


def test_predict_returns_valid_structure_and_is_mostly_right(client):
    hits = 0
    labels = sorted({c for _, c in TEST_CLIPS})
    for lab in labels:
        p = sample(lab)
        r = client.post("/api/predict", files={"file": (p.name, p.read_bytes(), "audio/wav")})
        assert r.status_code == 200
        j = r.json()
        assert j["label"] in j["top"][0]["label"] and 0 <= j["confidence"] <= 1 and len(j["spectrogram"]) == 32
        assert j["alert"] == (j["label"] in C.ALERT_CLASSES)
        hits += j["label"] == lab
    assert hits >= 6, f"only {hits}/{len(labels)} sanity clips right"


def test_predict_rejects_garbage(client):
    r = client.post("/api/predict", files={"file": ("x.wav", b"hello", "audio/wav")})
    assert r.status_code == 422


def test_upload_validates_label_and_audio(client):
    p = sample("dog")
    bad_label = client.post("/api/upload", data={"label": "!"}, files=[("files", (p.name, p.read_bytes(), "audio/wav"))])
    assert bad_label.status_code == 422
    r = client.post("/api/upload", data={"label": "Test Class"},
                    files=[("files", ("a.wav", p.read_bytes(), "audio/wav")), ("files", ("b.wav", b"junk", "audio/wav"))])
    j = r.json()
    assert r.status_code == 200 and j["label"] == "test_class" and j["accepted"] == 1 and len(j["rejected"]) == 1
    assert client.get("/api/uploads").json()["pending"] >= 1


def test_status_has_uptime_and_traffic(client):
    s = client.get("/api/status").json()
    assert s["service"]["uptime_s"] >= 0 and s["traffic"]["total"] >= 1 and s["model"]["loaded"]


def test_insights_and_samples(client):
    assert len(client.get("/api/insights").json()["stories"]) == 3
    assert len(client.get("/api/samples").json()) == 10


def test_ui_is_served(client):
    assert "ForestEar" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


# -------------------------------------------------------- slow: retraining
@pytest.mark.skipif(not os.getenv("RUN_SLOW"), reason="set RUN_SLOW=1 (takes a few minutes)")
def test_retrain_with_new_class_promotes_new_version(client):
    ext = os.getenv("NEW_CLASS_DIR")
    assert ext, "set NEW_CLASS_DIR to a folder of .wav files for a new class"
    files = sorted(__import__("pathlib").Path(ext).glob("*.wav"))[:12]
    r = client.post("/api/upload", data={"label": "newsound"}, files=[("files", (f.name, f.read_bytes(), "audio/wav")) for f in files])
    assert r.json()["accepted"] == len(files)
    assert client.post("/api/retrain?mode=finetune").status_code == 202
    for _ in range(600):
        st = client.get("/api/retrain/status").json()
        if not st["running"]:
            break
        time.sleep(2)
    run = st["latest"]
    assert run["status"] == "done", run["message"]
    assert "newsound" in client.get("/api/status").json()["model"]["classes"]
