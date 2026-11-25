"""Step 2 - Data processing.

Raw audio -> fixed-length waveform -> log-mel spectrogram (64 x 126).
Also holds the augmentations used for training and the train/val split rule.
"""
import hashlib
import io
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from . import config as C


class AudioError(ValueError):
    """Raised for files we cannot decode or that contain no usable signal."""


# ------------------------------------------------------------------ loading
def load_waveform(src) -> np.ndarray:
    """Decode a path / bytes / file object into a mono float32 waveform at 16 kHz."""
    try:
        if isinstance(src, (bytes, bytearray)):
            src = io.BytesIO(src)
        y, sr = sf.read(src, dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile raises RuntimeError / LibsndfileError
        raise AudioError(f"Could not decode audio ({exc}). Use WAV, FLAC or OGG.") from exc
    y = y.mean(axis=1)
    if sr != C.SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=C.SAMPLE_RATE)
    if y.size < C.SAMPLE_RATE // 4:
        raise AudioError("Clip is shorter than 0.25 s.")
    if not np.isfinite(y).all() or float(np.abs(y).max()) < 1e-4:
        raise AudioError("Clip is silent or corrupt.")
    return y.astype(np.float32)


def fit_length(y: np.ndarray) -> np.ndarray:
    """Pad with silence (like ESC-50 does) or keep the loudest 5 s window."""
    n = C.N_SAMPLES
    if y.size <= n:
        return np.pad(y, (0, n - y.size))
    step = C.SAMPLE_RATE // 4
    csum = np.concatenate([[0.0], np.cumsum(y.astype(np.float64) ** 2)])
    starts = np.arange(0, y.size - n + 1, step)
    energy = csum[starts + n] - csum[starts]
    s = int(starts[int(np.argmax(energy))])
    return y[s:s + n]


# ----------------------------------------------------------------- features
def logmel(y: np.ndarray) -> np.ndarray:
    """(N_SAMPLES,) waveform -> (N_MELS, N_FRAMES) log-power mel spectrogram in dB."""
    S = librosa.feature.melspectrogram(
        y=y, sr=C.SAMPLE_RATE, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH,
        n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX, power=2.0)
    db = librosa.power_to_db(S, ref=1.0, amin=1e-10, top_db=None)
    return np.maximum(db, C.DB_FLOOR).astype(np.float32)


def features_from_source(src) -> np.ndarray:
    return logmel(fit_length(load_waveform(src)))


def to_model_input(feats: np.ndarray) -> np.ndarray:
    """(N, 64, 126) -> (N, 64, 126, 1)"""
    return feats[..., None].astype(np.float32)


# ------------------------------------------------------------ augmentation
def augment(feat: np.ndarray, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """Feature-level augmentation: time shift, gain change, SpecAugment masks.
    strength 0 = off, 1 = full, 0.5 = gentle (small data prefers gentle)."""
    if strength <= 0:
        return feat
    max_shift = int(feat.shape[1] * 0.5 * strength)
    x = np.roll(feat, int(rng.integers(-max_shift, max_shift + 1)), axis=1)
    x = np.maximum(x + rng.uniform(-12 * strength, 6 * strength), C.DB_FLOOR)   # loudness shift in dB
    floor = float(x.min())
    for _ in range(int(rng.integers(0, 1 + int(round(2 * strength))))):          # frequency masks
        w = int(rng.integers(2, 3 + int(6 * strength))); f0 = int(rng.integers(0, C.N_MELS - w))
        x[f0:f0 + w, :] = floor
    for _ in range(int(rng.integers(0, 1 + int(round(2 * strength))))):          # time masks
        w = int(rng.integers(3, 4 + int(12 * strength))); t0 = int(rng.integers(0, C.N_FRAMES - w))
        x[:, t0:t0 + w] = floor
    return x.astype(np.float32)


# ------------------------------------------------------------ dataset utils
_FOLD_RE = re.compile(r"^(\d)-")


def is_validation(path: Path) -> bool:
    """ESC clips are split by their official fold (same-source clips never leak);
    uploaded clips are split by a stable hash so the split survives retraining."""
    m = _FOLD_RE.match(path.name)
    if m:
        return int(m.group(1)) == C.VAL_FOLD
    return int(hashlib.md5(path.name.encode()).hexdigest(), 16) % 100 < 15


def list_clips(root: Path):
    """[(path, class_name)] for root/<class>/*.{wav,flac,ogg}"""
    out = []
    if not root.exists():
        return out
    for cdir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for f in sorted(cdir.iterdir()):
            if f.suffix.lower() in C.ALLOWED_EXT:
                out.append((f, cdir.name))
    return out


def build_features(clips, use_cache: bool = True):
    """Return (X (N,64,126), paths, labels) with a content-addressed on-disk cache."""
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sig = hashlib.md5("|".join(f"{p.name}:{p.stat().st_size}" for p, _ in clips).encode()).hexdigest()
    cache = C.CACHE_DIR / f"feat_{sig}.npy"
    if use_cache and cache.exists():
        X = np.load(cache)
    else:
        X = np.stack([features_from_source(str(p)) for p, _ in clips]).astype(np.float32)
        if use_cache:
            np.save(cache, X)
    return X, [p for p, _ in clips], [c for _, c in clips]


def encode_labels(labels, classes):
    idx = {c: i for i, c in enumerate(classes)}
    return np.array([idx[l] for l in labels], dtype=np.int64)
