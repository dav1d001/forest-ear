"""Central configuration. Every path and audio constant lives here so the
notebook, the API and the retraining job can never drift apart."""
import os
from pathlib import Path

ROOT = Path(os.getenv("FOREST_EAR_HOME", Path(__file__).resolve().parents[1]))
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", DATA_DIR / "uploads"))
CACHE_DIR = DATA_DIR / ".cache"
INSIGHTS_PATH = DATA_DIR / "insights.json"
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "forest_ear.db"))
UI_DIR = ROOT / "ui"

# ---- audio / feature constants (a clip is 5 s of 16 kHz mono audio) ----
SAMPLE_RATE = 16_000
CLIP_SECONDS = 5.0
N_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
N_FFT = 1024
HOP_LENGTH = 640
N_MELS = 64
FMIN, FMAX = 20, 8000
N_FRAMES = 1 + N_SAMPLES // HOP_LENGTH          # 126
INPUT_SHAPE = (N_MELS, N_FRAMES, 1)
DB_FLOOR = -100.0

# ---- dataset ----
ESC_URL = "https://codeload.github.com/karolpiczak/ESC-50/zip/refs/heads/master"
VAL_FOLD = 4          # ESC folds 1-3 train, 4 validation, 5 held-out test
TEST_FOLD = 5
ALLOWED_EXT = {".wav", ".flac", ".ogg"}
MAX_UPLOAD_MB = 15

# ---- product behaviour ----
ALERT_CLASSES = {"chainsaw", "crackling_fire", "helicopter"}   # forest-threat sounds
LOW_CONFIDENCE = 0.60
DRIFT_WINDOW = 50                  # last N predictions used for drift check
AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "1") == "1"
AUTO_RETRAIN_MIN_CLIPS = int(os.getenv("AUTO_RETRAIN_MIN_CLIPS", "20"))
PROMOTION_TOLERANCE = 0.03         # same classes: challenger may lose at most 3 pts macro-F1 on the fixed test set
EXPANSION_TOLERANCE = 0.08         # new class added: old classes may lose up to 8 pts ...
NEW_CLASS_MIN_F1 = 0.60            # ... but every new class must reach this F1 on its validation clips
SEED = 42
