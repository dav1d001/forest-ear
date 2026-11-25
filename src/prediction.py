"""Step 4 - Model testing / prediction.

`Predictor` owns the promoted model, hot-reloads it when another container (or
the retraining job) promotes a new version, and turns raw audio bytes into a
prediction in a few milliseconds.
"""
import os
import threading
import time

import numpy as np

from . import config as C
from . import model as M
from .preprocessing import features_from_source

TOP_K = 5


def _downsample_for_display(feat: np.ndarray) -> list:
    """64x126 dB spectrogram -> 32x63 uint8 grid the browser can paint directly."""
    f = feat.reshape(32, 2, 63, 2).mean(axis=(1, 3))
    lo, hi = float(np.percentile(f, 2)), float(np.percentile(f, 99.5))
    return np.clip((f - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8).tolist()


class Predictor:
    def __init__(self):
        self._lock = threading.Lock()
        self.model = None
        self.entry = None
        self._stamp = None
        self.reload(force=True)

    @staticmethod
    def _registry_stamp():
        p = C.MODELS_DIR / "registry.json"
        return p.stat().st_mtime_ns if p.exists() else None

    def reload(self, force=False):
        stamp = self._registry_stamp()
        if not force and stamp == self._stamp:
            return False
        with self._lock:
            self.model, self.entry = M.load_current()
            self._stamp = stamp
        return True

    @property
    def ready(self):
        return self.model is not None

    @property
    def classes(self):
        return self.entry["classes"] if self.entry else []

    def predict(self, source) -> dict:
        """source: path, bytes or file-like. Returns label, confidence, top-k, alert flag, spectrogram."""
        self.reload()
        if not self.ready:
            raise RuntimeError("No trained model yet - train one first (POST /api/retrain?mode=scratch).")
        t0 = time.perf_counter()
        feat = features_from_source(source)
        with self._lock:
            proba = self.model.predict(feat[None, ..., None], verbose=0)[0]
        order = np.argsort(proba)[::-1]
        top = [{"label": self.classes[i], "probability": float(proba[i])} for i in order[:TOP_K]]
        best = top[0]
        return {
            "label": best["label"], "confidence": best["probability"], "top": top,
            "alert": best["label"] in C.ALERT_CLASSES,
            "low_confidence": best["probability"] < C.LOW_CONFIDENCE,
            "model_version": self.entry["version"],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "spectrogram": _downsample_for_display(feat),
            "host": os.getenv("HOSTNAME", "local"),
        }


_predictor = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


def predict_file(path) -> dict:
    """Convenience for notebooks / scripts: predict_file('clip.wav')"""
    return get_predictor().predict(str(path))
