"""Run PaddleOCR candidate extraction with progress logging."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from ocr_candidate_extractor import (
    create_paddle_ocr,
    extract_candidates_from_image,
    iter_image_files,
    read_image_size,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--ocr-version", default="PP-OCRv4")
    parser.add_argument("--text-detection-model-name", default=None)
    parser.add_argument("--text-recognition-model-name", default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    image_paths = iter_image_files(args.images)
    ocr = create_paddle_ocr(
        lang=args.lang,
        use_gpu=args.use_gpu,
        ocr_version=args.ocr_version,
        text_detection_model_name=args.text_detection_model_name,
        text_recognition_model_name=args.text_recognition_model_name,
    )
    images: List[Dict[str, Any]] = []
    started = time.time()

    for index, image_path in enumerate(image_paths, 1):
        width, height = read_image_size(image_path)
        candidates = extract_candidates_from_image(image_path, ocr=ocr)
        images.append(
            {
                "image_id": image_path.stem,
                "image_path": str(image_path).replace("\\", "/"),
                "image_width": width,
                "image_height": height,
                "candidates": [candidate.to_dict() for candidate in candidates],
            }
        )
        if index == 1 or index % args.progress_every == 0 or index == len(image_paths):
            elapsed = time.time() - started
            rate = index / elapsed if elapsed > 0 else 0.0
            print(
                f"progress {index}/{len(image_paths)} "
                f"candidates={sum(len(item['candidates']) for item in images)} "
                f"rate={rate:.2f} img/s",
                flush=True,
            )

    doc = {"images": images}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    total_candidates = sum(len(item["candidates"]) for item in images)
    print(f"wrote {args.out}", flush=True)
    print(f"processed {len(images)} images with {total_candidates} OCR candidates", flush=True)


if __name__ == "__main__":
    main()
