"""Recognition-only wrapper for the fine-tuned channel digit PaddleOCR model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence, Tuple

from PIL import Image


Prediction = Tuple[str, float]


def digits(text: str) -> str:
    return "".join(ch for ch in str(text) if "0" <= ch <= "9")


def _parse_prediction_item(item: Any) -> Prediction:
    if isinstance(item, dict):
        text = item.get("rec_text", item.get("text", ""))
        score = item.get("rec_score", item.get("score", item.get("confidence", 0.0)))
        return digits(str(text)), float(score)

    try:
        data = dict(item)
    except Exception:
        return digits(str(item)), 0.0
    text = data.get("rec_text", data.get("text", ""))
    score = data.get("rec_score", data.get("score", data.get("confidence", 0.0)))
    return digits(str(text)), float(score)


class ChannelDigitRecognizer:
    """PaddleOCR TextRecognition wrapper that preserves digit strings."""

    def __init__(
        self,
        model_dir: Path,
        *,
        model_name: str = "PP-OCRv5_mobile_rec",
        device: str = "cpu",
        input_shape: str = "3,48,320",
    ) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"channel digit model directory does not exist: {self.model_dir}")

        shape = tuple(int(part.strip()) for part in input_shape.split(","))
        try:
            # On Windows, loading Paddle before Torch can trigger a DLL load-order
            # issue through ModelScope. Import Torch first when it is present.
            try:
                import torch  # noqa: F401
            except ImportError:
                pass
            from paddleocr import TextRecognition

            self._recognizer = TextRecognition(
                model_name=model_name,
                model_dir=str(self.model_dir),
                device=device,
                input_shape=shape,
            )
        except Exception as exc:
            message = str(exc)
            hint = ""
            if "PreconditionNotMet" in message and "kernel output args" in message:
                hint = (
                    " This usually indicates a Paddle inference runtime/export-format mismatch. "
                    "The H200 training log says the model was trained/exported with Paddle 3.3.1; "
                    "the local runtime must support that inference.json format, or the model should "
                    "be re-exported to a compatible legacy inference format."
                )
            raise RuntimeError(f"failed to load channel digit recognizer from {self.model_dir}.{hint}") from exc

    def predict(self, crop: Image.Image) -> List[Prediction]:
        return self.predict_many([crop])[0]

    def predict_many(self, crops: Sequence[Image.Image]) -> List[List[Prediction]]:
        import numpy as np

        if not crops:
            return []
        images = [np.array(crop.convert("RGB")) for crop in crops]
        raw = self._recognizer.predict(images)
        results: List[List[Prediction]] = []
        for item in raw:
            item_predictions: List[Prediction] = []
            if isinstance(item, list):
                iterable = item
            else:
                iterable = [item]
            for sub_item in iterable:
                value, confidence = _parse_prediction_item(sub_item)
                if value:
                    item_predictions.append((value, confidence))
            results.append(item_predictions)
        while len(results) < len(images):
            results.append([])
        return results[: len(images)]
