"""Voiceprint verification model (Formative 2, Member C).

    from src.voice import verify_voice
    verify_voice("data/audio/ketia_approve_1.wav")                 # -> "ketia", or None if not recognised
    verify_voice("clip.wav", expected_member="ketia")              # -> "ketia" only if the voice is Ketia's

A voice is accepted only if BOTH checks pass:
  1. the classifier is confident (highest class probability >= threshold), and
  2. the clip is close to at least one of that member's own recordings: the nearest-neighbour distance
     on the MFCC features is within the radius learned from the members' genuine recordings.
Check 2 is what rejects strangers: a classifier trained on three people always picks one of the three,
often with high confidence, even for a voice it has never heard.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from . import audio_processing as ap
except ImportError:                      # run as a plain script
    import audio_processing as ap

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "voice_model.pkl"
PROB_THRESHOLD = 0.55        # minimum classifier confidence
DIST_PREFIXES = ("mfcc",)    # distance check uses the MFCC mean/std features (not delta, energy, ...)
RADIUS_MARGIN = 1.10         # radius = largest genuine nearest-neighbour distance x this margin
SEED = 42

CANDIDATES = {
    "Logistic Regression (C=0.1)": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
    "Logistic Regression (C=1)":   make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
    "Logistic Regression (C=10)":  make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=5000)),
    "Random Forest (300 trees)":   make_pipeline(StandardScaler(), RandomForestClassifier(
                                       n_estimators=300, min_samples_leaf=1, random_state=SEED)),
    "Random Forest (300 trees, leaf=3)": make_pipeline(StandardScaler(), RandomForestClassifier(
                                       n_estimators=300, min_samples_leaf=3, random_state=SEED)),
}


# ------------------------------------------------------------------ distance check
def distance_columns(cols):
    return [c for c in cols if c.startswith(DIST_PREFIXES)]


def nn_distance(ref: dict, x: np.ndarray, member) -> float:
    """Distance from one clip (raw features in ref['columns'] order) to the nearest recording of `member`."""
    z = ref["scaler"].transform(x.reshape(1, -1))
    return float(np.linalg.norm(ref["points"][member] - z, axis=1).min())


def fit_distance_check(df: pd.DataFrame, cols) -> dict:
    """Reference points per member, plus the acceptance radius. The radius is calibrated by leaving each
    original recording out in turn and measuring how far it is from the member's remaining recordings
    (and their augmentations); the largest such genuine distance, times a margin, becomes the radius."""
    dcols = distance_columns(cols)
    scaler = StandardScaler().fit(df[dcols].values)          # numpy, so later numpy inputs raise no warning
    points = {m: scaler.transform(g[dcols].values) for m, g in df.groupby("member")}
    genuine = []
    for clip, g in df[df.augmentation == "original"].groupby("clip_id"):
        rest = df[(df.clip_id != clip) & (df.member == g.member.iloc[0])]
        if len(rest):
            genuine.append(np.linalg.norm(scaler.transform(rest[dcols].values) - scaler.transform(g[dcols].values), axis=1).min())
    radius = float(max(genuine) * RADIUS_MARGIN) if genuine else np.inf
    return {"columns": dcols, "scaler": scaler, "points": points, "radius": radius,
            "genuine_distances": [float(d) for d in genuine]}


# ------------------------------------------------------------------ evaluation
def leave_one_clip_out(df: pd.DataFrame, model, cols: list[str]) -> pd.DataFrame:
    """Hold out one ORIGINAL recording at a time (with all its augmentations removed from training),
    train on everything else, and predict the held-out clip and its augmented versions.
    Keeping a clip's augmentations out of training stops the model from being tested on a near-copy."""
    out = []
    classes = sorted(df["member"].unique())
    for clip in df["clip_id"].unique():
        train, test = df[df.clip_id != clip], df[df.clip_id == clip]
        if train["member"].nunique() < len(classes):
            continue                                  # a member would vanish from training
        m = clone(model).fit(train[cols], train["member"])
        proba = m.predict_proba(test[cols])
        ref = fit_distance_check(train, cols)          # radius calibrated without the held-out clip
        for (_, row), p in zip(test.iterrows(), proba):
            pred = m.classes_[p.argmax()]
            dist = nn_distance(ref, row[ref["columns"]].values.astype(float), pred)
            out.append({"clip_id": clip, "augmentation": row.augmentation, "true": row.member, "pred": pred,
                        "confidence": float(p.max()), "distance": dist, "radius": ref["radius"],
                        **{f"p_{c}": v for c, v in zip(m.classes_, p)}})
    return pd.DataFrame(out)


def scores(cv: pd.DataFrame) -> dict:
    classes = sorted(cv["true"].unique())
    P = cv[[f"p_{c}" for c in classes]].values
    accepted = (cv.confidence >= PROB_THRESHOLD) & (cv.distance <= cv.radius) & (cv.pred == cv.true)
    return {"accuracy": accuracy_score(cv.true, cv.pred),
            "f1_macro": f1_score(cv.true, cv.pred, average="macro"),
            "log_loss": log_loss(cv.true, P, labels=classes),
            "genuine_accept_rate": float(accepted.mean())}


def compare_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Leave-one-clip-out scores for every candidate, on the held-out ORIGINAL recordings."""
    cols = ap.feature_columns(df)
    rows, cvs = [], {}
    for name, model in CANDIDATES.items():
        cv = leave_one_clip_out(df, model, cols)
        cvs[name] = cv
        rows.append({"model": name, **scores(cv[cv.augmentation == "original"]),
                     "accuracy_on_augmented": accuracy_score(*cv[cv.augmentation != "original"][["true", "pred"]].T.values)})
    table = pd.DataFrame(rows).sort_values(["log_loss", "f1_macro"], ascending=[True, False]).reset_index(drop=True)
    return table, cvs


# ------------------------------------------------------------------ final model
def train_final(df: pd.DataFrame, model_name: str, path: Path = DEFAULT_MODEL) -> dict:
    cols = ap.feature_columns(df)
    model = clone(CANDIDATES[model_name]).fit(df[cols], df["member"])
    bundle = {"model": model, "model_name": model_name, "feature_columns": cols, "classes": list(model.classes_),
              "threshold": PROB_THRESHOLD, "distance_check": fit_distance_check(df, cols),
              "members": sorted(df["member"].unique()), "n_training_rows": len(df)}
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle


_cache: dict = {}


def _load(path: Path = DEFAULT_MODEL) -> dict:
    path = Path(path)
    if path not in _cache:
        _cache[path] = joblib.load(path)
    return _cache[path]


def check_voice(audio_path, model_path: Path = DEFAULT_MODEL) -> dict:
    """Full result for one clip: best-matching member, confidence, distance, and whether it is accepted."""
    b = _load(model_path)
    x = pd.DataFrame([ap.extract_features(ap.load_clip(audio_path))])[b["feature_columns"]]
    p = b["model"].predict_proba(x)[0]
    best = b["classes"][int(p.argmax())]
    ref = b["distance_check"]
    dist = nn_distance(ref, x[ref["columns"]].values[0].astype(float), best)
    confident, close = p.max() >= b["threshold"], dist <= ref["radius"]
    reason = ("accepted" if confident and close else
              "low confidence" if not confident else "voice too far from this member's recordings")
    return {"best_match": best, "confidence": float(p.max()), "distance": dist, "radius": ref["radius"],
            "accepted": bool(confident and close), "reason": reason,
            "probabilities": dict(zip(b["classes"], map(float, p)))}


def verify_voice(audio_path, expected_member: str | None = None, model_path: Path = DEFAULT_MODEL):
    """Member name if the voice is recognised (and matches expected_member when given), else None."""
    r = check_voice(audio_path, model_path)
    if not r["accepted"]:
        return None
    if expected_member is not None and r["best_match"] != expected_member.lower():
        return None
    return r["best_match"]


if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser(description="Check one voice clip")
    a.add_argument("audio")
    a.add_argument("--expected")
    a.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = a.parse_args()
    print(json.dumps(check_voice(args.audio, args.model), indent=2))
    print("verify_voice ->", verify_voice(args.audio, args.expected, args.model))
