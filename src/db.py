"""Tiny SQLite layer: uploaded clips, served predictions, retraining runs.
WAL mode lets several API containers share one volume safely for this workload."""
import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from . import config as C

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads(
  id INTEGER PRIMARY KEY, label TEXT NOT NULL, filename TEXT, path TEXT NOT NULL UNIQUE,
  created TEXT NOT NULL, used_in_version INTEGER);
CREATE TABLE IF NOT EXISTS predictions(
  id INTEGER PRIMARY KEY, ts REAL NOT NULL, label TEXT, confidence REAL,
  latency_ms REAL, model_version INTEGER, host TEXT);
CREATE TABLE IF NOT EXISTS retrain_runs(
  id INTEGER PRIMARY KEY, started TEXT, finished TEXT, status TEXT, stage TEXT,
  trigger TEXT, mode TEXT, from_version INTEGER, to_version INTEGER, n_new_clips INTEGER,
  promoted INTEGER, metrics TEXT, message TEXT);
CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(ts);
"""


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def conn():
    C.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(C.DB_PATH, timeout=15)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=15000")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init():
    with conn() as c:
        c.executescript(SCHEMA)


# ---- uploads
def add_upload(label, filename, path):
    with conn() as c:
        c.execute("INSERT OR IGNORE INTO uploads(label,filename,path,created) VALUES(?,?,?,?)",
                  (label, filename, str(path), now_iso()))


def upload_counts():
    with conn() as c:
        rows = c.execute("SELECT label, COUNT(*) n, SUM(used_in_version IS NULL) pending "
                         "FROM uploads GROUP BY label ORDER BY label").fetchall()
    return [dict(r) for r in rows]


def pending_uploads():
    with conn() as c:
        return c.execute("SELECT COUNT(*) FROM uploads WHERE used_in_version IS NULL").fetchone()[0]


def mark_uploads_used(version):
    with conn() as c:
        c.execute("UPDATE uploads SET used_in_version=? WHERE used_in_version IS NULL", (version,))


# ---- predictions
def log_prediction(label, confidence, latency_ms, version, host):
    with conn() as c:
        c.execute("INSERT INTO predictions(ts,label,confidence,latency_ms,model_version,host) VALUES(?,?,?,?,?,?)",
                  (time.time(), label, confidence, latency_ms, version, host))


def prediction_stats():
    with conn() as c:
        total = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        recent = c.execute("SELECT confidence, latency_ms FROM predictions ORDER BY id DESC LIMIT ?",
                           (C.DRIFT_WINDOW,)).fetchall()
        last_min = c.execute("SELECT COUNT(*) FROM predictions WHERE ts > ?", (time.time() - 60,)).fetchone()[0]
        mix = c.execute("SELECT label, COUNT(*) n FROM predictions GROUP BY label ORDER BY n DESC").fetchall()
    conf = [r["confidence"] for r in recent if r["confidence"] is not None]
    lat = sorted(r["latency_ms"] for r in recent if r["latency_ms"] is not None)
    pct = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))] if lat else None
    return {"total": total, "last_minute": last_min,
            "recent_mean_confidence": sum(conf) / len(conf) if conf else None,
            "recent_n": len(conf), "latency_p50_ms": pct(0.5), "latency_p95_ms": pct(0.95),
            "label_mix": {r["label"]: r["n"] for r in mix}}


# ---- retraining runs
def start_run(trigger, mode, from_version, n_new):
    with conn() as c:
        cur = c.execute("INSERT INTO retrain_runs(started,status,stage,trigger,mode,from_version,n_new_clips) "
                        "VALUES(?,?,?,?,?,?,?)", (now_iso(), "running", "queued", trigger, mode, from_version, n_new))
        return cur.lastrowid


def update_run(run_id, **fields):
    if "metrics" in fields and not isinstance(fields["metrics"], str):
        fields["metrics"] = json.dumps(fields["metrics"])
    keys = ", ".join(f"{k}=?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE retrain_runs SET {keys} WHERE id=?", (*fields.values(), run_id))


def get_runs(limit=10):
    with conn() as c:
        rows = c.execute("SELECT * FROM retrain_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["metrics"] = json.loads(d["metrics"]) if d.get("metrics") else None
        out.append(d)
    return out
