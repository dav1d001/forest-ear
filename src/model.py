"""Step 3 - Model creation (+ evaluation, versioning and fine-tuning).

The network is a compact 4-block CNN over log-mel spectrograms. Optimisation
techniques used: L2 regularisation, spatial + dense dropout, batch-norm, label
smoothing, Adam with ReduceLROnPlateau, early stopping, SpecAugment-style
augmentation, and a small hyper-parameter search (see the notebook).
"""
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone

import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras
from keras import layers, regularizers
from sklearn import metrics as skm

from . import config as C
from .preprocessing import augment

DEFAULT_PARAMS = dict(base_filters=16, dropout=0.3, l2=1e-4, lr=2e-3, batch_size=16, label_smoothing=0.05, aug=0.5)
BLOCK_MULT = (1, 2, 4, 8)


# ------------------------------------------------------------------ network
def build_cnn(n_classes: int, base_filters=16, dropout=0.3, l2=1e-4, **_) -> keras.Model:
    reg = regularizers.l2(l2)
    inp = layers.Input(C.INPUT_SHAPE, name="mel_input")
    x = layers.Rescaling(1 / 30.0, offset=50 / 30.0, name="scale")(inp)
    x = layers.BatchNormalization(name="input_bn")(x)
    for i, m in enumerate(BLOCK_MULT, 1):
        f = base_filters * m
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, kernel_regularizer=reg, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.ReLU(name=f"relu{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
        x = layers.SpatialDropout2D(dropout / 2, name=f"sdrop{i}")(x)
    x = layers.Concatenate(name="embedding")([layers.GlobalAveragePooling2D()(x), layers.GlobalMaxPooling2D()(x)])
    x = layers.Dropout(dropout, name="head_dropout")(x)
    out = layers.Dense(n_classes, activation="softmax", kernel_regularizer=reg, name="head")(x)
    return keras.Model(inp, out, name="forest_ear_cnn")


def compile_model(model, lr=2e-3, label_smoothing=0.05, **_):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
                  metrics=["accuracy"])
    return model


class AugmentedBatches(keras.utils.PyDataset):
    """Shuffled mini-batches with on-the-fly augmentation (training only)."""

    def __init__(self, X, y_idx, n_classes, batch_size=16, augment_on=True, seed=C.SEED, aug=1.0, **kw):
        super().__init__(**kw)
        self.X, self.y = X, keras.utils.to_categorical(y_idx, n_classes)
        self.bs, self.aug, self.rng = batch_size, augment_on, np.random.default_rng(seed)
        self.strength = aug
        self.order = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.bs))

    def __getitem__(self, i):
        idx = self.order[i * self.bs:(i + 1) * self.bs]
        xb = self.X[idx]
        if self.aug:
            xb = np.stack([augment(x, self.rng, self.strength) for x in xb])
        return xb[..., None], self.y[idx]

    def on_epoch_end(self):
        self.rng.shuffle(self.order)


def fit(model, Xtr, ytr, Xva, yva, n_classes, epochs=80, params=None, patience=15, verbose=0):
    p = {**DEFAULT_PARAMS, **(params or {})}
    train_seq = AugmentedBatches(Xtr, ytr, n_classes, p["batch_size"], aug=p["aug"], augment_on=p["aug"] > 0)
    val_seq = AugmentedBatches(Xva, yva, n_classes, 32, augment_on=False)
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, start_from_epoch=5),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, patience // 3), min_lr=1e-5)]
    t0 = time.time()
    hist = model.fit(train_seq, validation_data=val_seq, epochs=epochs, callbacks=cbs, verbose=verbose)
    return hist.history, time.time() - t0


# --------------------------------------------------------------- evaluation
def predict_proba(model, X, batch_size=32):
    return model.predict(X[..., None], batch_size=batch_size, verbose=0)


def evaluate(model, X, y_idx, classes) -> dict:
    """Accuracy, loss, precision/recall/F1 (macro+weighted), top-2, ROC-AUC, confusion matrix."""
    n = len(classes)
    proba = predict_proba(model, X)
    pred = proba.argmax(1)
    labels = list(range(n))
    p, r, f, s = skm.precision_recall_fscore_support(y_idx, pred, labels=labels, zero_division=0)
    present = np.isin(labels, np.unique(y_idx))
    try:
        auc = float(skm.roc_auc_score(y_idx, proba, multi_class="ovr", average="macro", labels=labels)) \
            if present.all() else None
    except ValueError:
        auc = None
    onehot = keras.utils.to_categorical(y_idx, n)
    loss = float(keras.losses.CategoricalCrossentropy()(onehot, proba))
    top2 = float(np.mean([y in np.argsort(row)[-2:] for y, row in zip(y_idx, proba)]))
    return {
        "n_samples": int(len(y_idx)),
        "accuracy": float(skm.accuracy_score(y_idx, pred)),
        "loss": loss,
        "top2_accuracy": top2,
        "precision_macro": float(p[present].mean()), "recall_macro": float(r[present].mean()),
        "f1_macro": float(f[present].mean()),
        "f1_weighted": float(skm.f1_score(y_idx, pred, average="weighted", zero_division=0)),
        "roc_auc_macro": auc,
        "per_class": {classes[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                   "f1": float(f[i]), "support": int(s[i])} for i in labels},
        "confusion_matrix": skm.confusion_matrix(y_idx, pred, labels=labels).tolist(),
        "classes": list(classes),
    }


# ---------------------------------------------------------- model registry
def _registry_path():
    return C.MODELS_DIR / "registry.json"


def load_registry() -> dict:
    p = _registry_path()
    if p.exists():
        return json.loads(p.read_text())
    return {"current": None, "versions": []}


def _atomic_write(path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def current_entry(reg=None):
    reg = reg or load_registry()
    return next((v for v in reg["versions"] if v["version"] == reg["current"]), None)


def save_version(model, classes, metrics, params, trigger, parent=None, promote=True, extra=None) -> dict:
    """Write models/forestear_v<N>.h5 and record it. Promoted versions also become
    models/forestear_model.h5 (the file named in the assignment checklist)."""
    reg = load_registry()
    version = max([v["version"] for v in reg["versions"]], default=0) + 1
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"forestear_v{version}.h5"
    model.save(C.MODELS_DIR / fname)
    entry = {"version": version, "file": fname, "classes": list(classes), "params": params,
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "trigger": trigger, "parent": parent, "promoted": bool(promote),
             "metrics": {k: v for k, v in metrics.items() if k != "per_class"},
             "per_class": metrics.get("per_class"), **(extra or {})}
    reg["versions"].append(entry)
    if promote:
        reg["current"] = version
        shutil.copyfile(C.MODELS_DIR / fname, C.MODELS_DIR / "forestear_model.h5")
    _atomic_write(_registry_path(), json.dumps(reg, indent=2))
    return entry


def load_current():
    """-> (keras model, entry) for the promoted version, or (None, None)."""
    entry = current_entry()
    if not entry:
        return None, None
    return keras.models.load_model(C.MODELS_DIR / entry["file"], compile=False), entry


# -------------------------------------------------------------- fine-tuning
def transfer_weights(src_model, dst_model, src_classes, dst_classes):
    """Copy every shared layer by name; copy head rows for classes both models know."""
    for layer in dst_model.layers:
        if layer.name == "head":
            continue
        try:
            layer.set_weights(src_model.get_layer(layer.name).get_weights())
        except (ValueError, KeyError):
            pass
    ks, bs = src_model.get_layer("head").get_weights()
    kd, bd = dst_model.get_layer("head").get_weights()
    for j, c in enumerate(dst_classes):
        if c in src_classes:
            i = src_classes.index(c)
            kd[:, j], bd[j] = ks[:, i], bs[i]
    dst_model.get_layer("head").set_weights([kd, bd])


def fine_tune(parent_model, parent_classes, classes, params, Xtr, ytr, Xva, yva, epochs=40, verbose=0):
    """Use the existing custom model as the pre-trained starting point.
    New classes get a fresh head row; stage 1 trains only the head, stage 2 unfreezes all."""
    p = {**DEFAULT_PARAMS, **params}
    model = build_cnn(len(classes), **p)
    transfer_weights(parent_model, model, list(parent_classes), list(classes))
    log = {}
    if set(classes) != set(parent_classes):
        for l in model.layers:
            l.trainable = (l.name == "head")
        compile_model(model, lr=1e-3, label_smoothing=p["label_smoothing"])
        log["stage1"], _ = fit(model, Xtr, ytr, Xva, yva, len(classes), epochs=8, params=p, patience=8, verbose=verbose)
        for l in model.layers:
            l.trainable = True
    compile_model(model, lr=p["lr"] * 0.2, label_smoothing=p["label_smoothing"])
    log["stage2"], secs = fit(model, Xtr, ytr, Xva, yva, len(classes), epochs=epochs, params=p, patience=8, verbose=verbose)
    return model, log, secs
