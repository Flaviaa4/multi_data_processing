"""Face-and-voice-authorized transaction and recommendation CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from demo_app import run_transaction  # noqa: E402


def _print_result(result: dict) -> None:
    print("\n" + "=" * 58)
    print(f"  {result['status']}")
    print("=" * 58)
    print(result["message"])
    face = result["face"]
    print(f"Face confidence: {face['confidence']:.1%}")
    voice = result.get("voice")
    if voice is not None:
        print(f"Voice match: {voice['best_match']} ({voice['confidence']:.1%})")
    if not result["authorized"]:
        print("No transaction or recommendation was revealed.")
        return

    customer = result["customer"]
    transaction = result["transaction"]
    print(f"Customer: {customer['display_name']} (ID {customer['customer_id']})")
    print(f"Transaction: #{transaction['transaction_id']}")
    print(f"Date: {transaction['purchase_date']}")
    print(f"Amount: ${transaction['purchase_amount']:.2f}")
    print(f"Recommended category: {result['recommendation']}")


def _unauthorized_demo() -> dict:
    with NamedTemporaryFile(suffix=".png", delete=False) as file:
        image_path = Path(file.name)
    try:
        Image.new("RGB", (64, 64), color=(12, 24, 48)).save(image_path)
        return run_transaction(image_path, threshold=1.01)
    finally:
        image_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Face-and-voice-authorized transaction and product recommendation."
    )
    parser.add_argument("--image", type=Path, help="Image to authenticate.")
    parser.add_argument("--voice", type=Path, help="Voice sample to authenticate.")
    parser.add_argument(
        "--voice-model",
        type=Path,
        help="Voice model path (defaults to audio_pipeline\\models\\voice_model.pkl).",
    )
    parser.add_argument(
        "--unauthorized-demo",
        action="store_true",
        help="Run a guaranteed denied-access demonstration.",
    )
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()

    if args.unauthorized_demo:
        _print_result(_unauthorized_demo())
        return 0

    image_path = args.image
    if image_path is None:
        print("SECURE CHECKOUT")
        print("1. Authenticate an image")
        print("2. Run unauthorized demo")
        choice = input("Choose an option [1/2]: ").strip()
        if choice == "2":
            _print_result(_unauthorized_demo())
            return 0
        image_path = Path(input("Path to image: ").strip().strip('"'))
        voice_path = Path(input("Path to voice sample: ").strip().strip('"'))
    else:
        voice_path = args.voice

    try:
        if voice_path is None:
            parser.error("--voice is required when --image is provided")
        _print_result(
            run_transaction(
                image_path,
                audio_path=voice_path,
                threshold=args.threshold,
                voice_model_path=args.voice_model,
            )
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
