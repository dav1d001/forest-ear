"""Step 5 - Retraining.

Pipeline:  uploaded clips (saved to disk + SQLite)  ->  preprocessing  ->
fine-tune the current model (our own pre-trained model)  ->  evaluate on the
fixed held-out test set  ->  promote only if it did not get worse.

Triggers:  1) the "Retrain" button / POST /api/retrain
           2) automatic - enough new clips have been uploaded
           3) automatic - live confidence has drifted down and new clips exist
CLI:       python -m src.retrain --mode scratch      (first training)
           python -m src.retrain --mode finetune     (retrain with uploads)
"""
import argparse
import fcntl
import threading
import traceback

import numpy as np

from . import config as C
from . import db
from . import model as M
from .preprocessing import build_features, encode_labels, is_validation, list_clips

MIN_CLIPS_PER_CLASS = 5


class RetrainBusy(RuntimeError):
    pass


def _acquire_lock():
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fh = open(C.MODELS_DIR / ".retrain.lock", "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        raise RetrainBusy("A training job is already running.")
    return fh


def is_running() -> bool:
    try:
        fh = _acquire_lock()
    except RetrainBusy:
        return True
    fcntl.flock(fh, fcntl.LOCK_UN); fh.close()
    return False


def retrain_recommendation() -> dict:
    """Should the operator (or the auto-trigger) retrain right now?"""
    pending = db.pending_uploads()
    stats = db.prediction_stats()
    drift = (stats["recent_n"] >= 30 and stats["recent_mean_confidence"] is not None
             and stats["recent_mean_confidence"] < C.LOW_CONFIDENCE)
    reason = None
    if pending >= C.AUTO_RETRAIN_MIN_CLIPS:
        reason = f"{pending} new clips are waiting (threshold {C.AUTO_RETRAIN_MIN_CLIPS})."
        trig = "auto-upload"
    elif drift and pending >= MIN_CLIPS_PER_CLASS:
        reason = (f"Mean confidence over the last {stats['recent_n']} predictions fell to "
                  f"{stats['recent_mean_confidence']:.0%} and {pending} new clips are available.")
        trig = "auto-drift"
    else:
        trig = None
    return {"recommended": reason is not None, "reason": reason, "trigger": trig,
            "pending_uploads": pending, "drift_detected": bool(drift)}


def maybe_auto_retrain():
    if not C.AUTO_RETRAIN:
        return None
    rec = retrain_recommendation()
    if rec["recommended"] and not is_running():
        try:
            return start(rec["trigger"], "finetune")
        except RetrainBusy:
            return None
    return None


def start(trigger="manual", mode="finetune", background=True) -> int:
    """Create a run and execute it (in a background thread by default)."""
    if mode not in {"finetune", "scratch"}:
        raise ValueError("mode must be 'finetune' or 'scratch'")
    lock = _acquire_lock()
    try:
        db.init()
        entry = M.current_entry()
        if entry is None:
            mode = "scratch"
        run_id = db.start_run(trigger, mode, entry["version"] if entry else None, db.pending_uploads())
    except Exception:
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
        raise

    def job():
        try:
            _execute(run_id, trigger, mode)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN); lock.close()

    if background:
        threading.Thread(target=job, daemon=True, name=f"retrain-{run_id}").start()
    else:
        job()
    return run_id


def _execute(run_id: int, trigger: str, mode: str):
    try:
        parent = M.current_entry()
        # ---- 1. collect + preprocess ------------------------------------------------
        db.update_run(run_id, stage="collecting data")
        clips = list_clips(C.TRAIN_DIR) + list_clips(C.UPLOAD_DIR)
        counts = {}
        for _, c in clips:
            counts[c] = counts.get(c, 0) + 1
        usable = {c for c, n in counts.items() if n >= MIN_CLIPS_PER_CLASS}
        skipped = sorted(set(counts) - usable)
        clips = [(p, c) for p, c in clips if c in usable]
        if not clips:
            raise RuntimeError("No training data found. Run `python -m src.data_acquisition` first.")

        db.update_run(run_id, stage="preprocessing audio -> log-mel spectrograms")
        X, paths, labels = build_features(clips)
        if mode == "scratch" or parent is None:
            classes = sorted(usable)
        else:
            classes = list(parent["classes"]) + sorted(usable - set(parent["classes"]))
        keep = np.array([l in classes for l in labels])
        X, paths, labels = X[keep], [p for p, k in zip(paths, keep) if k], [l for l, k in zip(labels, keep) if k]
        y = encode_labels(labels, classes)
        val_mask = np.array([is_validation(p) for p in paths])
        Xtr, ytr, Xva, yva = X[~val_mask], y[~val_mask], X[val_mask], y[val_mask]

        # ---- 2. train -----------------------------------------------------------------
        if mode == "scratch" or parent is None:
            db.update_run(run_id, stage=f"training from scratch ({len(Xtr)} clips)")
            params = {**M.DEFAULT_PARAMS, **(parent["params"] if parent else {})}
            model = M.compile_model(M.build_cnn(len(classes), **params), **params)
            _, secs = M.fit(model, Xtr, ytr, Xva, yva, len(classes), epochs=80, params=params)
        else:
            db.update_run(run_id, stage=f"fine-tuning v{parent['version']} ({len(Xtr)} clips)")
            parent_model, _ = M.load_current()
            params = parent["params"]
            model, _, secs = M.fine_tune(parent_model, parent["classes"], classes, params, Xtr, ytr, Xva, yva)

        # ---- 3. evaluate ----------------------------------------------------------------
        db.update_run(run_id, stage="evaluating on held-out test set")
        test_clips = [(p, c) for p, c in list_clips(C.TEST_DIR) if c in classes]
        if test_clips:
            Xte, _, lte = build_features(test_clips)
            test_m = M.evaluate(model, Xte, encode_labels(lte, classes), classes)
        else:
            test_m = M.evaluate(model, Xva, yva, classes)
        val_m = M.evaluate(model, Xva, yva, classes)

        # ---- 4. promote? -----------------------------------------------------------------
        champion_f1 = parent["metrics"].get("f1_macro") if parent else None
        new_classes = [c for c in classes if parent and c not in parent["classes"]]
        tol = C.EXPANSION_TOLERANCE if new_classes else C.PROMOTION_TOLERANCE
        keeps_old = champion_f1 is None or test_m["f1_macro"] >= champion_f1 - tol
        new_f1 = {c: val_m["per_class"][c]["f1"] for c in new_classes}
        learns_new = all(f >= C.NEW_CLASS_MIN_F1 for f in new_f1.values())
        promote = keeps_old and learns_new
        msg = (f"Trained in {secs:.0f}s. Original-sound test accuracy {test_m['accuracy']:.1%}, macro-F1 {test_m['f1_macro']:.3f}"
               + (f" (champion {champion_f1:.3f}, allowed drop {tol:.2f})." if champion_f1 is not None else "."))
        if new_f1:
            msg += " New sounds: " + ", ".join(f"{c} F1 {f:.2f}" for c, f in new_f1.items()) + "."
        if promote:
            msg += " Promoted to production."
        else:
            why = []
            if not keeps_old: why.append("it forgot too much of what the current model knows")
            if not learns_new: why.append(f"a new sound stayed below F1 {C.NEW_CLASS_MIN_F1:.2f}")
            msg += " Not promoted: " + " and ".join(why) + "."
        if skipped:
            msg += f" Skipped classes with <{MIN_CLIPS_PER_CLASS} clips: {', '.join(skipped)}."
        db.update_run(run_id, stage="saving model")
        entry = M.save_version(model, classes, test_m, params, trigger,
                               parent=parent["version"] if parent else None, promote=promote,
                               extra={"mode": mode, "train_clips": int(len(Xtr)), "val_clips": int(len(Xva)),
                                      "train_seconds": round(secs, 1)})
        db.mark_uploads_used(entry["version"])
        db.update_run(run_id, status="done", stage="finished", finished=db.now_iso(), to_version=entry["version"],
                      promoted=int(promote), metrics={"test": {k: v for k, v in test_m.items() if k != "per_class"},
                                                       "val_accuracy": val_m["accuracy"], "classes": classes},
                      message=msg)
        if promote:
            from .prediction import get_predictor
            get_predictor().reload(force=True)
    except Exception as exc:  # keep the API alive; surface the reason in the UI
        traceback.print_exc()
        db.update_run(run_id, status="failed", stage="failed", finished=db.now_iso(), message=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["scratch", "finetune"], default="finetune")
    ap.add_argument("--trigger", default="cli")
    a = ap.parse_args()
    db.init()
    rid = start(a.trigger, a.mode, background=False)
    print(db.get_runs(1)[0]["message"])
