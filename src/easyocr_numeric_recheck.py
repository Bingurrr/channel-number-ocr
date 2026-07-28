"""Run EasyOCR digit-only recheck on likely channel-number regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


BBox = Tuple[float, float, float, float]


def yolo_boxes(label_dir: Path, image_id: str, width: int, height: int) -> List[BBox]:
    path = label_dir / f"{image_id}.txt"
    boxes: List[BBox] = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5 or int(float(parts[0])) not in (0, 3):
            continue
        cx, cy, bw, bh = [float(v) for v in parts[1:5]]
        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height
        boxes.append((x1, y1, x2, y2))
    return boxes


def likely_regions(
    image: Dict[str, Any],
    label_dir: Path,
    include_fallbacks: bool = True,
) -> List[BBox]:
    width = int(image["image_width"])
    height = int(image["image_height"])
    regions = []
    for box in yolo_boxes(label_dir, image["image_id"], width, height):
        regions.append(expand(box, width, height, 1.35, 1.6))
    if not include_fallbacks:
        return dedupe_boxes(regions)
    # Lower-left TV channel strip proposals. These recover cases where detector misses.
    regions.extend(
        [
            (0, height * 0.72, width * 0.16, height * 0.84),
            (0, height * 0.74, width * 0.24, height * 0.90),
            (0, height * 0.78, width * 0.12, height * 0.86),
        ]
    )
    for c in image.get("candidates", []):
        digits = "".join(ch for ch in str(c.get("text", "")) if "0" <= ch <= "9")
        if digits and is_left_bottom(c.get("bbox_xyxy", []), width, height):
            regions.append(expand(tuple(float(v) for v in c["bbox_xyxy"]), width, height, 2.2, 2.0))
    return dedupe_boxes(regions)


def is_left_bottom(bbox: Sequence[float], width: int, height: int) -> bool:
    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return ((x1 + x2) / 2) < width * 0.22 and ((y1 + y2) / 2) > height * 0.62


def expand(box: BBox, width: int, height: int, x_scale: float, y_scale: float) -> BBox:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = max(6, x2 - x1) * x_scale, max(6, y2 - y1) * y_scale
    return (max(0, cx - bw / 2), max(0, cy - bh / 2), min(width, cx + bw / 2), min(height, cy + bh / 2))


def dedupe_boxes(boxes: Iterable[BBox]) -> List[BBox]:
    out: List[BBox] = []
    for box in boxes:
        if not any(iou(box, prev) > 0.75 for prev in out):
            out.append(box)
    return out


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - inter
    return 0.0 if area <= 0 else inter / area


def resolve_image_path(path: Path) -> Path:
    text = str(path).replace("\\", "/")
    path = Path(text)
    if path.exists():
        return path
    candidate = Path.cwd() / path
    return candidate if candidate.exists() else path


def run_easyocr(reader: Any, crop: Image.Image) -> List[Tuple[str, float, BBox]]:
    import numpy as np

    arr = np.array(crop.convert("RGB"))
    raw = reader.readtext(arr, allowlist="0123456789", detail=1, paragraph=False, decoder="greedy")
    results: List[Tuple[str, float, BBox]] = []
    for points, text, conf in raw:
        digits = "".join(ch for ch in str(text) if "0" <= ch <= "9")
        if not digits:
            continue
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        results.append((digits, float(conf), (min(xs), min(ys), max(xs), max(ys))))
    return results


def run_easyocr_recognizer_only(reader: Any, crop: Image.Image) -> List[Tuple[str, float, BBox]]:
    import numpy as np

    arr = np.array(crop.convert("RGB"))
    raw = reader.recognize(arr, allowlist="0123456789", detail=1, paragraph=False, decoder="greedy")
    results: List[Tuple[str, float, BBox]] = []
    for item in raw:
        if len(item) < 3:
            continue
        _, text, conf = item[:3]
        digits = "".join(ch for ch in str(text) if "0" <= ch <= "9")
        if not digits:
            continue
        results.append((digits, float(conf), (0.0, 0.0, float(crop.width), float(crop.height))))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--yolo-label-dir", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--yolo-only",
        action="store_true",
        help="Only recheck YOLO class 0/3 regions; skip heuristic fallback regions.",
    )
    parser.add_argument(
        "--recognizer-only",
        action="store_true",
        help="Skip EasyOCR text detector and recognize the whole recheck crop as one text region.",
    )
    parser.add_argument("--progress-every", type=int, default=0)
    args = parser.parse_args()

    try:
        import easyocr
    except ImportError as exc:
        raise SystemExit("easyocr is not installed; run `python -m pip install easyocr`") from exc

    doc = json.loads(args.ocr_json.read_text(encoding="utf-8"))
    reader = easyocr.Reader(["en"], gpu=args.gpu, detector=not args.recognizer_only)
    recheck_fn = run_easyocr_recognizer_only if args.recognizer_only else run_easyocr
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
                crop = rgb.crop((x1, y1, x2, y2))
                for hit_index, (digits, conf, local_box) in enumerate(recheck_fn(reader, crop), 1):
                    lx1, ly1, lx2, ly2 = local_box
                    image.setdefault("candidates", []).append(
                        {
                            "id": f"easyocr_{region_index:02d}_{hit_index:02d}",
                            "text": digits,
                            "bbox_xyxy": [round(x1 + lx1, 3), round(y1 + ly1, 3), round(x1 + lx2, 3), round(y1 + ly2, 3)],
                            "ocr_conf": round(conf, 6),
                            "detection_conf": round(conf, 6),
                            "source": "easyocr_numeric_recheck",
                        }
                    )
                    added += 1
        if args.progress_every and (
            image_index == 1 or image_index % args.progress_every == 0 or image_index == len(images)
        ):
            print(f"progress {image_index}/{len(images)} added={added}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} with {added} EasyOCR numeric candidates")


if __name__ == "__main__":
    main()
