import argparse
import json
from pathlib import Path

import joblib
import numpy as np


MORAL_LABELS = [
    "Love",
    "Hate",
    "Joy",
    "Despair",
    "Peace",
    "Conflict",
    "Patience",
    "Impatience",
    "Kindness",
    "Cruelty",
    "Goodness",
    "Malice",
    "Faithfulness",
    "Betrayal",
    "Gentleness",
    "Harshness",
    "Self_control",
    "Recklessness",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on lyrics using a saved model bundle."
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to a joblib model bundle created by training/train.py",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        type=str,
        help="Raw lyrics text for inference",
    )
    input_group.add_argument(
        "--text-file",
        type=Path,
        help="Path to a UTF-8 text file containing lyrics",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold when model supports predict_proba (default: 0.5)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    return parser.parse_args()


def read_input_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text.strip()
    return args.text_file.read_text(encoding="utf-8").strip()


def load_bundle(model_path: Path) -> tuple:
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "vectorizer" not in bundle or "classifier" not in bundle:
        raise ValueError(
            "Expected a dict model bundle with keys 'vectorizer' and 'classifier'."
        )
    return bundle["vectorizer"], bundle["classifier"]


def predict_labels(classifier, vectorizer, text: str, threshold: float) -> dict:
    x_vec = vectorizer.transform([text])

    probs = None
    if hasattr(classifier, "predict_proba"):
        try:
            raw_probs = classifier.predict_proba(x_vec)
            probs = np.array([p[:, 1][0] if p.shape[1] > 1 else p[:, 0][0] for p in raw_probs])
        except Exception:
            probs = None

    if probs is not None:
        binary = (probs >= threshold).astype(int)
    else:
        binary = classifier.predict(x_vec)[0].astype(int)

    label_pairs = []
    for idx, label in enumerate(MORAL_LABELS):
        item = {
            "label": label,
            "predicted": bool(binary[idx]),
        }
        if probs is not None:
            item["probability"] = float(probs[idx])
        label_pairs.append(item)

    positive_labels = [entry["label"] for entry in label_pairs if entry["predicted"]]
    return {
        "positive_labels": positive_labels,
        "all_labels": label_pairs,
    }


def main() -> None:
    args = parse_args()
    text = read_input_text(args)
    if not text:
        raise ValueError("Input text is empty. Provide non-empty lyrics via --text or --text-file.")

    vectorizer, classifier = load_bundle(args.model_path)
    result = predict_labels(classifier, vectorizer, text, args.threshold)

    if args.pretty:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()