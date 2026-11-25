"""Step 1 - Data acquisition.

Downloads ESC-50 (Piczak, 2015) from GitHub, keeps the 10-class ESC-10 subset,
converts every clip to 16 kHz / mono / 16-bit WAV and lays it out as

    data/train/<class>/<clip>.wav   (folds 1-4)
    data/test/<class>/<clip>.wav    (fold 5, never touched during training)

Run:  python -m src.data_acquisition
"""
import argparse
import csv
import io
import shutil
import urllib.request
import zipfile

import librosa
import numpy as np
import soundfile as sf

from . import config as C


def acquire(force: bool = False, zip_path: str | None = None) -> dict:
    if C.TRAIN_DIR.exists() and any(C.TRAIN_DIR.rglob("*.wav")) and not force:
        print("Dataset already present - use --force to rebuild.")
        return summarize()
    if zip_path:
        blob = open(zip_path, "rb")
    else:
        print("Downloading ESC-50 (~650 MB, one-off)...")
        with urllib.request.urlopen(C.ESC_URL) as r:
            blob = io.BytesIO(r.read())
    with zipfile.ZipFile(blob) as z:
        root = z.namelist()[0].split("/")[0]
        rows = list(csv.DictReader(io.TextIOWrapper(z.open(f"{root}/meta/esc50.csv"))))
        for d in (C.TRAIN_DIR, C.TEST_DIR):
            shutil.rmtree(d, ignore_errors=True)
        kept = 0
        for row in rows:
            if row["esc10"] != "True":
                continue
            fold = int(row["fold"])
            out = (C.TEST_DIR if fold == C.TEST_FOLD else C.TRAIN_DIR) / row["category"]
            out.mkdir(parents=True, exist_ok=True)
            with z.open(f"{root}/audio/{row['filename']}") as f:
                y, sr = sf.read(io.BytesIO(f.read()), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            y = librosa.resample(y, orig_sr=sr, target_sr=C.SAMPLE_RATE)
            sf.write(out / row["filename"], np.clip(y, -1, 1), C.SAMPLE_RATE, subtype="PCM_16")
            kept += 1
    print(f"Wrote {kept} clips.")
    return summarize()


def summarize() -> dict:
    out = {}
    for name, d in (("train", C.TRAIN_DIR), ("test", C.TEST_DIR)):
        out[name] = {p.name: len(list(p.glob("*.wav"))) for p in sorted(d.glob("*")) if p.is_dir()}
    print(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--zip", help="use an already-downloaded ESC-50 zip instead of downloading")
    a = ap.parse_args()
    acquire(a.force, a.zip)
