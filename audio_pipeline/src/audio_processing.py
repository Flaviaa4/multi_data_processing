"""Audio pipeline for the voiceprint verification model (Formative 2, Member C).

Steps: load and clean a recording -> augment it -> extract features -> audio_features.csv

File naming (data/audio/):   <member>_<phrase>_<take>.wav     e.g. ketia_approve_1.wav, ketia_confirm_2.wav
Unauthorized clips:          data/audio/unauthorized/<anything>.wav
Phone recordings (.m4a, .mp3, .ogg, .aac, .opus) are converted to 16 kHz mono WAV automatically.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

SR = 16_000                     # sample rate for every clip (Hz)
TOP_DB = 30                     # silence trimming threshold (dB below peak)
N_MFCC = 13
AUDIO_EXTS = {".wav", ".m4a", ".mp3", ".ogg", ".aac", ".opus", ".flac", ".webm"}
NAME_RE = re.compile(r"^(?P<member>[a-z0-9]+)_(?P<phrase>approve|confirm)_(?P<take>\d+)$", re.I)

# Each augmentation is (name, function(y, rng) -> y). At least two are applied to every clip.
# Pitch shifts are kept small (+-1 semitone): pitch is part of what identifies a speaker, so a
# large shift would turn one member's voice into something closer to another's.
AUGMENTATIONS = [
    ("pitch_up",     lambda y, rng: librosa.effects.pitch_shift(y, sr=SR, n_steps=1.0)),
    ("pitch_down",   lambda y, rng: librosa.effects.pitch_shift(y, sr=SR, n_steps=-1.0)),
    ("stretch_fast", lambda y, rng: librosa.effects.time_stretch(y, rate=1.15)),
    ("stretch_slow", lambda y, rng: librosa.effects.time_stretch(y, rate=0.87)),
    ("noise",        lambda y, rng: add_noise(y, snr_db=20, rng=rng)),
    ("gain_shift",   lambda y, rng: time_shift(y * 0.6, rng)),
]


# ------------------------------------------------------------------ loading
def to_wav(path: Path) -> Path:
    """Convert a non-WAV recording to 16 kHz mono WAV next to it (needs ffmpeg). Returns the WAV path."""
    if path.suffix.lower() == ".wav":
        return path
    out = path.with_suffix(".wav")
    if not out.exists():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-ac", "1", "-ar", str(SR), str(out)],
                       check=True)
    return out


def load_clip(path: str | Path) -> np.ndarray:
    """Load at 16 kHz mono, trim leading/trailing silence and peak-normalise."""
    y, _ = librosa.load(to_wav(Path(path)), sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)
    peak = np.max(np.abs(y)) if y.size else 0
    return y / peak * 0.95 if peak > 0 else y


def list_recordings(audio_dir: str | Path) -> pd.DataFrame:
    """Table of the members' recordings found in audio_dir, parsed from the file names."""
    audio_dir = Path(audio_dir)
    rows, skipped = [], []
    for p in sorted(audio_dir.iterdir()):
        if p.suffix.lower() not in AUDIO_EXTS or not p.is_file():
            continue
        if p.suffix.lower() != ".wav" and p.with_suffix(".wav").exists():
            continue                                    # already converted; use the WAV
        m = NAME_RE.match(p.stem)
        if not m:
            skipped.append(p.name)
            continue
        rows.append({"file": str(to_wav(p)), "member": m["member"].lower(),
                     "phrase": m["phrase"].lower(), "take": int(m["take"])})
    if skipped:
        print(f"Skipped (name not <member>_<approve|confirm>_<take>): {skipped}")
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No recordings named <member>_<approve|confirm>_<take>.wav in {audio_dir}")
    df["clip_id"] = df["member"] + "_" + df["phrase"] + "_" + df["take"].astype(str)
    return df


# ------------------------------------------------------------------ augmentation
def add_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white background noise at the given signal-to-noise ratio."""
    signal_power = np.mean(y ** 2)
    noise = rng.normal(0, np.sqrt(signal_power / 10 ** (snr_db / 10)), y.shape)
    return (y + noise).astype(np.float32)


def time_shift(y: np.ndarray, rng: np.random.Generator, max_s: float = 0.2) -> np.ndarray:
    """Pad a short random silence at the start (as if the speaker started a little later)."""
    return np.concatenate([np.zeros(int(rng.uniform(0.05, max_s) * SR), dtype=y.dtype), y])


def augment(y: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {name: fn(y, rng).astype(np.float32) for name, fn in AUGMENTATIONS}


# ------------------------------------------------------------------ features
def extract_features(y: np.ndarray) -> dict[str, float]:
    """MFCCs (mean, std), delta-MFCC means, spectral roll-off, centroid, bandwidth, zero-crossing rate,
    RMS energy and duration. Averaging over time gives the same number of features for any clip length."""
    feats: dict[str, float] = {}
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    for i in range(N_MFCC):
        feats[f"mfcc{i + 1}_mean"] = float(mfcc[i].mean())
        feats[f"mfcc{i + 1}_std"] = float(mfcc[i].std())
        feats[f"delta_mfcc{i + 1}_mean"] = float(delta[i].mean())
    for name, values in {
        "rolloff": librosa.feature.spectral_rolloff(y=y, sr=SR, roll_percent=0.85)[0],
        "centroid": librosa.feature.spectral_centroid(y=y, sr=SR)[0],
        "bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=SR)[0],
        "zcr": librosa.feature.zero_crossing_rate(y)[0],
        "rms_energy": librosa.feature.rms(y=y)[0],
    }.items():
        feats[f"{name}_mean"] = float(values.mean())
        feats[f"{name}_std"] = float(values.std())
    feats["duration_s"] = len(y) / SR
    return feats


def build_feature_table(audio_dir: str | Path, out_csv: str | Path | None = None,
                        aug_dir: str | Path | None = None) -> pd.DataFrame:
    """Features for every recording and every augmentation of it. Saves augmented WAVs to aug_dir and the
    table to out_csv. `clip_id` ties each augmentation to its original so they stay in the same split."""
    recs = list_recordings(audio_dir)
    aug_dir = Path(aug_dir) if aug_dir else Path(audio_dir) / "augmented"
    aug_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in enumerate(recs.itertuples()):
        y = load_clip(r.file)
        versions = {"original": y} | augment(y, seed=i)
        for aug_name, ya in versions.items():
            if aug_name != "original":
                sf.write(aug_dir / f"{r.clip_id}_{aug_name}.wav", ya, SR)
            rows.append({"clip_id": r.clip_id, "member": r.member, "phrase": r.phrase, "take": r.take,
                         "augmentation": aug_name, "source_file": Path(r.file).name, **extract_features(ya)})
    df = pd.DataFrame(rows)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


FEATURE_PREFIXES = ("mfcc", "delta_mfcc", "rolloff", "centroid", "bandwidth", "zcr", "rms_energy", "duration")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
