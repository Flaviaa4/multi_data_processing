"""Face-and-voice-authorized transaction and product recommendation demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any
 
from audio_pipeline.voice_app.src.voice import check_voice

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from image_pipeline import verify_face
from tabular.product_model import MERGED_PATH, predict_product

DEMO_PROFILES = {
    "flavia": {"customer_id": 151, "display_name": "Flavia", "voice_member": "flav"},
    "shakira": {"customer_id": 192, "display_name": "Shakira", "voice_member": "shakira"},
    "placide": {"customer_id": 114, "display_name": "Placide", "voice_member": "plac"},
}


def _profile_for(label: str) -> dict[str, Any] | None:
    return DEMO_PROFILES.get(label.casefold())


def _latest_transaction(customer_id: int) -> dict[str, Any]:
    dataset = pd.read_csv(MERGED_PATH)
    history = dataset[dataset["customer_id"].eq(customer_id)].copy()
    if history.empty:
        raise ValueError(f"No transaction history found for customer {customer_id}.")
    latest = history.sort_values(["purchase_date", "transaction_id"]).iloc[-1]
    return {
        "transaction_id": int(latest["transaction_id"]),
        "purchase_date": str(latest["purchase_date"]),
        "purchase_amount": float(latest["purchase_amount"]),
    }


def run_transaction(
    image_path: str | Path,
    audio_path: str | Path | None = None,
    threshold: float = 0.60,
    voice_model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run face and voice authorization before returning transaction details."""
    face_result = verify_face(image_path, model_path=ROOT / "models" / "face_model.pkl", threshold=threshold)
    if not face_result["recognized"]:
        return {
            "authorized": False,
            "status": "UNAUTHORIZED",
            "message": "Access denied: face not recognized.",
            "face": face_result,
        }

    profile = _profile_for(str(face_result["userId"]))
    if profile is None:
        return {
            "authorized": False,
            "status": "UNAUTHORIZED",
            "message": "Access denied: recognized identity has no demo profile.",
            "face": face_result,
        }

    if audio_path is None:
        return {
            "authorized": False,
            "status": "UNAUTHORIZED",
            "message": "Access denied: a voice sample is required.",
            "face": face_result,
        }

    customer_id = int(profile["customer_id"])
    model_path = (
        Path(voice_model_path)
        if voice_model_path
        else ROOT / "audio_pipeline" / "models" / "voice_model.pkl"
    )
    voice_result = check_voice(audio_path, model_path=model_path)
    voice_result["expected_member"] = profile["voice_member"]
    if not voice_result["accepted"] or voice_result["best_match"] != profile["voice_member"]:
        return {
            "authorized": False,
            "status": "UNAUTHORIZED",
            "message": "Access denied: voice not recognized for the face identity.",
            "face": face_result,
            "voice": voice_result,
        }

    return {
        "authorized": True,
        "status": "AUTHORIZED",
        "message": "Face and voice verified. Transaction approved.",
        "face": face_result,
        "voice": voice_result,
        "customer": {
            "user_id": str(face_result["userId"]),
            "display_name": profile["display_name"],
            "customer_id": customer_id,
        },
        "transaction": _latest_transaction(customer_id),
        "recommendation": predict_product(customer_id),
    }
