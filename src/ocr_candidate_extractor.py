"""Extract OCR candidates with PaddleOCR for heuristic channel ranking.

This implements option A:

image(s) -> PaddleOCR text boxes -> heuristic_eval.json candidates

The output JSON can be passed directly to evaluate_heuristic.py:

python src/evaluate_heuristic.py --input dataset/ocr_candidates.json \
  --metrics-out dataset/ocr_metrics.json --failures-out dataset/ocr_failures.csv
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from PIL import Image


BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


@dataclass(frozen=True)
class OcrCandidate:
    id: str
    text: str
    bbox_xyxy: BBox
    ocr_conf: float
    detection_conf: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "bbox_xyxy": [round(value, 3) for value in self.bbox_xyxy],
            "ocr_conf": round(self.ocr_conf, 6),
            "detection_conf": round(self.detection_conf, 6),
        }


def extract_candidates_from_image(
    image_path: Path,
    ocr: Any,
    min_conf: float = 0.0,
) -> List[OcrCandidate]:
    """Run PaddleOCR on one image and return candidate boxes."""

    raw_result = run_paddle_ocr(ocr, image_path)
    candidates: List[OcrCandidate] = []
    for index, item in enumerate(iter_ocr_items(raw_result), 1):
        bbox_xyxy = polygon_to_xyxy(item["points"])
        text = item["text"].strip()
        conf = float(item["confidence"])
        if not text or conf < min_conf:
            continue
        candidates.append(
            OcrCandidate(
                id=f"ocr_{index:04d}",
                text=text,
                bbox_xyxy=bbox_xyxy,
                ocr_conf=conf,
                detection_conf=conf,
            )
        )
    return candidates


def build_heuristic_eval_json(
    image_paths: Sequence[Path],
    output_json: Path,
    lang: str = "en",
    min_conf: float = 0.0,
    ground_truth: Optional[Mapping[str, str]] = None,
    use_gpu: bool = False,
    ocr_version: str = "PP-OCRv4",
    text_detection_model_name: Optional[str] = None,
    text_recognition_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract OCR candidates for images and write heuristic_eval.json."""

    ocr = create_paddle_ocr(
        lang=lang,
        use_gpu=use_gpu,
        ocr_version=ocr_version,
        text_detection_model_name=text_detection_model_name,
        text_recognition_model_name=text_recognition_model_name,
    )
    images: List[Dict[str, Any]] = []
    for image_path in image_paths:
        width, height = read_image_size(image_path)
        image_id = image_path.stem
        candidates = extract_candidates_from_image(image_path, ocr=ocr, min_conf=min_conf)

        item: Dict[str, Any] = {
            "image_id": image_id,
            "image_path": str(image_path),
            "image_width": width,
            "image_height": height,
            "candidates": [candidate.to_dict() for candidate in candidates],
        }
        if ground_truth is not None and image_id in ground_truth:
            item["ground_truth_channel_number"] = ground_truth[image_id]
        images.append(item)

    doc = {"images": images}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return doc


def create_paddle_ocr(
    lang: str,
    use_gpu: bool,
    ocr_version: str = "PP-OCRv4",
    text_detection_model_name: Optional[str] = None,
    text_recognition_model_name: Optional[str] = None,
    text_recognition_model_dir: Optional[str] = None,
) -> Any:
    """Create a PaddleOCR instance while tolerating API differences."""

    # On Windows, PaddleOCR can import ModelScope, which imports torch after
    # Paddle DLLs are loaded. Importing torch first avoids a DLL load-order
    # failure in mixed Paddle/Torch environments.
    try:
        import torch  # noqa: F401
    except ImportError:
        pass

    from paddleocr import PaddleOCR

    model_kwargs: Dict[str, Any] = {"ocr_version": ocr_version}
    if text_detection_model_name:
        model_kwargs["text_detection_model_name"] = text_detection_model_name
    if text_recognition_model_name:
        model_kwargs["text_recognition_model_name"] = text_recognition_model_name
    # 파인튜닝된 rec 가중치를 쓰려면 로컬 inference 디렉토리를 지정 (det은 그대로).
    if text_recognition_model_dir:
        model_kwargs["text_recognition_model_dir"] = text_recognition_model_dir

    attempts = [
        {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "lang": lang,
            "text_det_limit_side_len": 960,
            "device": "gpu" if use_gpu else "cpu",
            "enable_mkldnn": False,
            **model_kwargs,
        },
        {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "lang": lang,
            "text_det_limit_side_len": 960,
            "enable_mkldnn": False,
            **model_kwargs,
        },
        {"use_textline_orientation": True, "lang": lang, "device": "gpu" if use_gpu else "cpu", "enable_mkldnn": False, **model_kwargs},
        {"use_textline_orientation": True, "lang": lang, "enable_mkldnn": False, **model_kwargs},
        {"use_angle_cls": True, "lang": lang, "use_gpu": use_gpu, "show_log": False},
        {"use_angle_cls": True, "lang": lang, "show_log": False},
        {"lang": lang},
    ]
    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            return PaddleOCR(**kwargs)
        except (TypeError, ValueError) as exc:
            last_error = exc
            continue
    raise RuntimeError(f"failed to create PaddleOCR: {last_error}")


def run_paddle_ocr(ocr: Any, image_path: Path) -> Any:
    """Run OCR with either PaddleOCR v2 or v3 style APIs."""

    if hasattr(ocr, "ocr"):
        try:
            return ocr.ocr(str(image_path), cls=True)
        except TypeError:
            return ocr.ocr(str(image_path))
    if hasattr(ocr, "predict"):
        return ocr.predict(str(image_path))
    raise RuntimeError("Unsupported PaddleOCR object: no ocr() or predict() method")


def iter_ocr_items(raw_result: Any) -> Iterable[Dict[str, Any]]:
    """Yield normalized OCR items from common PaddleOCR output shapes."""

    if raw_result is None:
        return

    # PaddleOCR v2 commonly returns: [[ [points], (text, confidence) ], ...]
    if isinstance(raw_result, list):
        pages = raw_result
        if len(pages) == 1 and isinstance(pages[0], list):
            pages = pages[0]
        for entry in pages:
            parsed = parse_v2_entry(entry)
            if parsed is not None:
                yield parsed
            elif isinstance(entry, Mapping):
                for parsed_mapping in parse_mapping_entry(entry):
                    yield parsed_mapping
        return

    # PaddleOCR v3 predict results may be custom dict-like objects.
    if isinstance(raw_result, Mapping):
        for parsed_mapping in parse_mapping_entry(raw_result):
            yield parsed_mapping


def parse_v2_entry(entry: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
        return None
    points = entry[0]
    text_score = entry[1]
    if not isinstance(text_score, (list, tuple)) or len(text_score) < 2:
        return None
    return {
        "points": parse_points(points),
        "text": str(text_score[0]),
        "confidence": float(text_score[1]),
    }


def parse_mapping_entry(entry: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    """Parse dict-style OCR output, including PaddleOCR v3 result dicts."""

    # Common batch fields.
    texts = entry.get("rec_texts") or entry.get("texts")
    scores = entry.get("rec_scores") or entry.get("scores")
    boxes = (
        entry.get("rec_polys")
        or entry.get("dt_polys")
        or entry.get("boxes")
        or entry.get("polys")
    )
    if texts is not None and boxes is not None:
        for index, text in enumerate(texts):
            score = 1.0 if scores is None else float(scores[index])
            yield {
                "points": parse_points(boxes[index]),
                "text": str(text),
                "confidence": score,
            }
        return

    # Single-item fields.
    text = entry.get("text") or entry.get("rec_text")
    score = entry.get("confidence") or entry.get("score") or entry.get("rec_score")
    points = entry.get("points") or entry.get("bbox") or entry.get("box")
    if text is not None and points is not None:
        yield {
            "points": parse_points(points),
            "text": str(text),
            "confidence": 1.0 if score is None else float(score),
        }


def parse_points(raw_points: Any) -> List[Point]:
    points: List[Point] = []
    for point in raw_points:
        try:
            if len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
        except TypeError:
            continue
        except ValueError:
            continue
    if not points:
        # Some APIs return flattened [x1, y1, x2, y2].
        try:
            values = [float(value) for value in raw_points]
        except TypeError:
            values = []
        if len(values) == 4:
            x1, y1, x2, y2 = values
            points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if len(points) < 2:
        raise ValueError(f"invalid OCR points: {raw_points}")
    return points


def polygon_to_xyxy(points: Sequence[Point]) -> BBox:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def iter_image_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(file_path for file_path in path.rglob("*") if file_path.suffix.lower() in exts)


def read_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def load_ground_truth(path: Optional[Path]) -> Optional[Dict[str, str]]:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(key): str(value) for key, value in data.items()}
    out: Dict[str, str] = {}
    for item in data:
        image_id = str(item.get("image_id") or item.get("id"))
        value = item.get("channel_number") or item.get("ground_truth_channel_number")
        if value is not None:
            out[image_id] = str(value)
    return out


def guess_ground_truth_from_filename(path: Path) -> Optional[str]:
    """Optional helper for quick ad-hoc tests; not used unless requested."""

    match = re.search(r"channel[_-]?(\d+)", path.stem, flags=re.I)
    return None if match is None else match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True, help="Image file or directory.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dataset/ocr_candidates.json"),
        help="Output heuristic_eval.json path.",
    )
    parser.add_argument("--lang", default="en")
    parser.add_argument("--min-conf", type=float, default=0.0)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--ocr-version", default="PP-OCRv4")
    parser.add_argument("--text-detection-model-name", default=None)
    parser.add_argument("--text-recognition-model-name", default=None)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Optional JSON mapping image_id -> channel number.",
    )
    args = parser.parse_args()

    image_paths = iter_image_files(args.images)
    ground_truth = load_ground_truth(args.ground_truth)
    doc = build_heuristic_eval_json(
        image_paths=image_paths,
        output_json=args.out,
        lang=args.lang,
        min_conf=args.min_conf,
        ground_truth=ground_truth,
        use_gpu=args.use_gpu,
        ocr_version=args.ocr_version,
        text_detection_model_name=args.text_detection_model_name,
        text_recognition_model_name=args.text_recognition_model_name,
    )
    total_candidates = sum(len(item["candidates"]) for item in doc["images"])
    print(f"wrote {args.out}")
    print(f"processed {len(doc['images'])} images with {total_candidates} OCR candidates")


if __name__ == "__main__":
    main()
