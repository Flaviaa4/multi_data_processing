"""Image feature extraction and face-recognition model utilities.

Expected image layout:
    data/images/<member>_neutral.jpg
    data/images/<member>_smiling.jpg
    data/images/<member>_surprised.jpg

The public ``verify_face`` function is intentionally small so the CLI can
load the saved model without knowing how image features are produced.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (64, 64)
MODEL_PATH = Path("models/face_model.pkl")
FEATURES_PATH = Path("data/image_features.csv")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
register_heif_opener()
SUPPORTED_EXTENSIONS.add(".heic")


def _read_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {path.suffix}")
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _feature_vector(image: Image.Image) -> np.ndarray:
    """Return a fixed-length vector combining pixels and RGB histograms."""
    resized = image.resize(IMAGE_SIZE)
    grayscale = np.asarray(ImageOps.grayscale(resized), dtype=np.float32) / 255.0
    pixels = grayscale.reshape(-1)
    histogram_parts = [
        np.histogram(np.asarray(resized)[:, :, channel], bins=16, range=(0, 256), density=True)[0]
        for channel in range(3)
    ]
    return np.concatenate([pixels, *histogram_parts]).astype(np.float32)


def extract_image_features(image_path: str | Path) -> np.ndarray:
    """Extract the same feature vector used during training."""
    return _feature_vector(_read_image(Path(image_path)))


def augment_image(image: Image.Image, name: str) -> Image.Image:
    """Create one deterministic augmentation for a training sample."""
    if name == "rotate":
        return image.rotate(12, expand=False)
    if name == "flip":
        return ImageOps.mirror(image)
    if name == "grayscale":
        return ImageOps.grayscale(image).convert("RGB")
    raise ValueError(f"Unknown augmentation: {name}")


def _samples(images_dir: Path) -> list[tuple[str, Path]]:
    samples: list[tuple[str, Path]] = []
    labels_by_key: dict[str, str] = {}
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            label = re.sub(r"\d+$", "", path.stem.rsplit("_", 1)[0]).strip()
            if not label:
                raise ValueError(f"Cannot infer member label from filename: {path.name}")
            label_key = label.casefold()
            canonical_label = labels_by_key.setdefault(label_key, label)
            samples.append((canonical_label, path))
    if not samples:
        raise FileNotFoundError(f"No supported images found in {images_dir}")
    return samples


def build_image_features_csv(
    images_dir: str | Path = "data/images",
    output_path: str | Path = FEATURES_PATH,
) -> Path:
    """Extract original and rotate/flip/grayscale features into a CSV."""
    images_path = Path(images_dir)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str, np.ndarray]] = []
    for label, path in _samples(images_path):
        image = _read_image(path)
        rows.append((label, path.name, _feature_vector(image)))
        for augmentation in ("rotate", "flip", "grayscale"):
            rows.append(
                (label, f"{path.name}:{augmentation}", _feature_vector(augment_image(image, augmentation)))
            )

    feature_names = [f"pixel_{index}" for index in range(len(rows[0][2]))]
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "source"] + feature_names)
        for label, source, vector in rows:
            writer.writerow([label, source, *vector.tolist()])
    return destination


def train_face_model(
    features_path: str | Path = FEATURES_PATH,
    model_path: str | Path = MODEL_PATH,
    metrics_path: str | Path = "models/face_metrics.json",
) -> dict[str, float]:
    """Train and save the face recognizer, returning accuracy/F1/log loss."""
    feature_file = Path(features_path)
    if not feature_file.is_file():
        raise FileNotFoundError(f"Feature CSV does not exist: {feature_file}")
    labels: list[str] = []
    vectors: list[list[float]] = []
    with feature_file.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            labels.append(row.pop("label"))
            row.pop("source")
            vectors.append([float(value) for value in row.values()])

    if len(set(labels)) < 2:
        raise ValueError("At least two member labels are required to train face recognition.")
    x = np.asarray(vectors, dtype=np.float32)
    y = np.asarray(labels)
    class_counts = np.unique(y, return_counts=True)[1]
    stratify = y if class_counts.min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=stratify
    )
    classifier = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    classifier.fit(x_train, y_train)
    probabilities = classifier.predict_proba(x_test)
    predictions = classifier.classes_[np.argmax(probabilities, axis=1)]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_weighted": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_test, probabilities, labels=classifier.classes_)),
    }
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, destination)
    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def verify_face(
    image_path: str | Path,
    model_path: str | Path = MODEL_PATH,
    threshold: float = 0.60,
) -> dict[str, Any]:
    """Identify a face image or return an explicit unauthorized result."""
    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Face model does not exist: {model_file}")
    classifier = joblib.load(model_file)
    probabilities = classifier.predict_proba([extract_image_features(image_path)])[0]
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    recognized = confidence >= threshold
    return {
        "recognized": recognized,
        "userId": str(classifier.classes_[best_index]) if recognized else None,
        "confidence": confidence,
    }
