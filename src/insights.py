"""Dataset insights: acoustic features per class, used by the notebook and the UI."""
import json

import librosa
import numpy as np
import pandas as pd

from . import config as C
from .preprocessing import list_clips, load_waveform, logmel, fit_length


def clip_features(path) -> dict:
    y = fit_length(load_waveform(str(path)))
    stft_kw = dict(n_fft=C.N_FFT, hop_length=C.HOP_LENGTH)
    rms = librosa.feature.rms(y=y, frame_length=C.N_FFT, hop_length=C.HOP_LENGTH)[0]
    rms_db = 20 * np.log10(np.maximum(rms, 1e-6))
    active = rms_db > rms_db.max() - 25            # frames within 25 dB of the loudest one
    cent = librosa.feature.spectral_centroid(y=y, sr=C.SAMPLE_RATE, **stft_kw)[0]
    flat = librosa.feature.spectral_flatness(y=y, **stft_kw)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=C.N_FFT, hop_length=C.HOP_LENGTH)[0]
    w = active if active.any() else np.ones_like(active)
    return {
        "centroid_hz": float(cent[w].mean()),
        "loudness_db": float(20 * np.log10(max(np.sqrt(np.mean(y ** 2)), 1e-6))),
        "active_fraction": float(active.mean()),
        "zero_crossing_rate": float(zcr[w].mean()),
        "flatness": float(flat[w].mean()),
        "dynamic_range_db": float(rms_db.max() - np.percentile(rms_db, 10)),
        "mel_profile": logmel(y)[:, active].mean(axis=1) if active.any() else logmel(y).mean(axis=1),
    }


def feature_table(split_dirs=None) -> pd.DataFrame:
    rows = []
    for split, root in (split_dirs or {"train": C.TRAIN_DIR, "test": C.TEST_DIR}).items():
        for p, c in list_clips(root):
            f = clip_features(p)
            f.update(label=c, split=split, file=p.name)
            rows.append(f)
    return pd.DataFrame(rows)


SCALARS = ["centroid_hz", "loudness_db", "active_fraction", "zero_crossing_rate", "flatness", "dynamic_range_db"]


def build_insights(df: pd.DataFrame, stories=None) -> dict:
    per_class = {}
    for c, g in df.groupby("label"):
        per_class[c] = {k: {"mean": float(g[k].mean()), "std": float(g[k].std()),
                            "min": float(g[k].min()), "max": float(g[k].max())} for k in SCALARS}
        per_class[c]["mel_profile"] = np.mean(np.stack(g["mel_profile"].to_numpy()), axis=0).round(2).tolist()
        per_class[c]["n"] = int(len(g))
    mel_freqs = librosa.mel_frequencies(n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX).round(0).tolist()
    return {"classes": sorted(per_class), "per_class": per_class, "mel_freqs_hz": mel_freqs,
            "alert_classes": sorted(C.ALERT_CLASSES), "stories": stories or [],
            "n_clips": int(len(df)), "source": "ESC-10 (Piczak, 2015), 16 kHz mono, 5 s clips"}


def make_stories(df: pd.DataFrame) -> list:
    """Three data-driven interpretations; every number is computed from the data."""
    m = df.groupby("label")[SCALARS].mean()
    k = lambda col, n=1, asc=False: list(m[col].sort_values(ascending=asc).index[:n])
    hz = lambda c: f"{m.loc[c, 'centroid_hz'] / 1000:.1f} kHz"
    pc = lambda c: f"{m.loc[c, 'active_fraction']:.0%}"
    return [
        {"feature": "centroid_hz", "title": "Brightness: where the energy sits",
         "text": (f"Spectral centroid is the 'centre of mass' of a sound's spectrum. {k('centroid_hz')[0].capitalize()} is the brightest "
                  f"({hz(k('centroid_hz')[0])}, broadband hiss) while {k('centroid_hz', 1, True)[0]} is the darkest "
                  f"({hz(k('centroid_hz', 1, True)[0])}, low rotor thump). Chainsaw ({hz('chainsaw')}), crying baby "
                  f"({hz('crying_baby')}) and sneezing ({hz('sneezing')}) all land at almost the same value, so brightness alone "
                  "cannot separate them. The model needs the time-frequency pattern, which is why we feed it spectrograms.")},
        {"feature": "active_fraction", "title": "Duration: how long the sound is 'on'",
         "text": (f"Share of the 5 s clip within 25 dB of its loudest moment. Rain ({pc('rain')}), sea waves ({pc('sea_waves')}), "
                  f"chainsaw ({pc('chainsaw')}) and helicopter ({pc('helicopter')}) are continuous; sneezing ({pc('sneezing')}), "
                  f"dog ({pc('dog')}) and rooster ({pc('rooster')}) are short bursts. Two of the three forest-threat classes "
                  "(chainsaw, helicopter) are sustained, which is good news for a field sensor: a few seconds of audio is enough to catch them.")},
        {"feature": "flatness", "title": "Texture: tonal versus noise-like",
         "text": (f"Spectral flatness is near 0 for pure tones and near 1 for white noise. Rooster ({m.loc['rooster', 'flatness']:.3f}) "
                  f"and dog ({m.loc['dog', 'flatness']:.3f}) are tonal and harmonic; rain ({m.loc['rain', 'flatness']:.3f}) is the "
                  f"most noise-like. Chainsaw ({m.loc['chainsaw', 'flatness']:.3f}) and sea waves ({m.loc['sea_waves', 'flatness']:.3f}) "
                  "are also noise-like and, like rain, are 'always on', so the continuous noisy sounds are the group "
                  "we expect to be hardest to tell apart. The confusion matrix after training tests that.")},
    ]


def write_insights(df, stories=None):
    C.INSIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    C.INSIGHTS_PATH.write_text(json.dumps(build_insights(df, stories), indent=1))


if __name__ == "__main__":
    df = feature_table()
    df.drop(columns=["mel_profile"]).to_csv("/tmp/features.csv", index=False)
    write_insights(df, make_stories(df))
    print(df.groupby("label")[SCALARS].mean().round(3).to_string())
