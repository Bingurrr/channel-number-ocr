"""Run the fine-tuned PaddleOCR channel-digit recognizer on candidate regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

import sys as _sys
_sys.path.insert(0,'/home/irteam/teacher_model_v3_for_test/src')
from channel_digit_recognizer import ChannelDigitRecognizer

def _pad3(im, aspect=3.0):
    w,h=im.size; tw=int(round(h*aspect))
    if tw<=w: return im
    out=Image.new("RGB",(tw,h),(255,255,255)); out.paste(im,((tw-w)//2,0)); return out
from easyocr_numeric_recheck import likely_regions, resolve_image_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--yolo-label-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("runs/ocr/inference"))
    parser.add_argument("--model-name", default="PP-OCRv5_mobile_rec")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--input-shape", default="3,48,320")
    parser.add_argument("--min-conf", type=float, default=0.0)
    parser.add_argument(
        "--yolo-only",
        action="store_true",
        help="Only recheck YOLO class 0/3 regions; skip heuristic fallback regions.",
    )
    parser.add_argument("--progress-every", type=int, default=0)
    args = parser.parse_args()

    doc = json.loads(args.ocr_json.read_text(encoding="utf-8"))
    recognizer = ChannelDigitRecognizer(
        args.model_dir,
        model_name=args.model_name,
        device=args.device,
        input_shape=args.input_shape,
    )

    added = 0
    images = doc.get("images", [])
    for image_index, image in enumerate(images, 1):
        image_path = resolve_image_path(Path(str(image.get("image_path", ""))))
        if not image_path.exists():
            continue

        with Image.open(image_path) as src:
            rgb = src.convert("RGB")
            width, height = rgb.size
            image["image_width"] = width
            image["image_height"] = height
            for region_index, region in enumerate(
                likely_regions(
                    image,
                    args.yolo_label_dir,
                    include_fallbacks=not args.yolo_only,
                ),
                1,
            ):
                x1, y1, x2, y2 = [int(round(v)) for v in region]
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = _pad3(rgb.crop((x1, y1, x2, y2)))
                for hit_index, (value, conf) in enumerate(recognizer.predict(crop), 1):
                    if conf < args.min_conf:
                        continue
                    image.setdefault("candidates", []).append(
                        {
                            "id": f"channel_digit_{region_index:02d}_{hit_index:02d}",
                            "text": value,
                            "bbox_xyxy": [round(float(x1), 3), round(float(y1), 3), round(float(x2), 3), round(float(y2), 3)],
                            "ocr_conf": round(float(conf), 6),
                            "detection_conf": round(float(conf), 6),
                            "source": "paddleocr_channel_recheck",
                            "recognizer_model_dir": str(args.model_dir),
                        }
                    )
                    added += 1

        if args.progress_every and (
            image_index == 1 or image_index % args.progress_every == 0 or image_index == len(images)
        ):
            print(f"progress {image_index}/{len(images)} added={added}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} with {added} fine-tuned PaddleOCR channel candidates")


if __name__ == "__main__":
    main()

