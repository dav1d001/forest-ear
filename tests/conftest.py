import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Isolate everything the app writes (models, uploads, DB) in a temp dir, before src.config is imported.
_TMP = Path(tempfile.mkdtemp(prefix="forest_ear_test_"))
shutil.copytree(ROOT / "models", _TMP / "models", ignore=shutil.ignore_patterns(".retrain.lock"))
os.environ.update(MODELS_DIR=str(_TMP / "models"), UPLOAD_DIR=str(_TMP / "uploads"),
                  DB_PATH=str(_TMP / "test.db"), AUTO_RETRAIN="0")
