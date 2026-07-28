"""Export recursive YOLO predictions and optionally evaluate channel GT boxes.

Ground truth is loaded only after model inference has completed. The exported
candidate labels therefore never depend on annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageFile

# Some captures on disk are truncated/corrupt jpgs. Tolerate them (load the
# partial image and move on) instead of crashing the whole run on one bad frame.
ImageFile.LOAD_TRUNCATED_IMAGES = True


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BBox = Tuple[float, float, float, float]


def iter_images(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def bbox_iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def to_yolo_line(cls: int, bbox: BBox, conf: float, width: float, height: float) -> str:
    x1, y1, x2, y2 = bbox
    return (
        f"{cls} {((x1 + x2) / 2.0) / width:.8f} {((y1 + y2) / 2.0) / height:.8f} "
        f"{(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f} {conf:.8f}"
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_gt_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = doc.get("images", doc if isinstance(doc, list) else [])
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        image_id = str(row.get("image_id") or Path(str(row.get("image_name") or "")).stem)
        if not image_id:
            continue
        bboxes = []
        for value in row.get("bboxes") or [row.get("bbox")]:
            if isinstance(value, (list, tuple)) and len(value) >= 4:
                bboxes.append(tuple(float(item) for item in value[:4]))
        result[image_id] = {**row, "bboxes": bboxes}
    return result


def load_exported_channel_predictions(raw_dir: Path, images: Sequence[Path]) -> Dict[str, List[Dict[str, Any]]]:
    predictions: Dict[str, List[Dict[str, Any]]] = {}
    for image_path in images:
        with Image.open(image_path) as image:
            width, height = image.size
        rows: List[Dict[str, Any]] = []
        label_path = raw_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) < 6 or int(float(parts[0])) != 0:
                    continue
                cx, cy, bw, bh, conf = (float(value) for value in parts[1:6])
                rows.append(
                    {
                        "bbox": (
                            (cx - bw / 2.0) * width,
                            (cy - bh / 2.0) * height,
                            (cx + bw / 2.0) * width,
                            (cy + bh / 2.0) * height,
                        ),
                        "conf": conf,
                    }
                )
        predictions[image_path.stem] = rows
    return predictions


def match_at_iou(predictions: Sequence[Mapping[str, Any]], gt_boxes: Sequence[BBox], threshold: float) -> Tuple[int, int, int]:
    matched = set()
    tp = 0
    for pred in sorted(predictions, key=lambda item: float(item["conf"]), reverse=True):
        best_index: Optional[int] = None
        best_iou = 0.0
        for index, gt_box in enumerate(gt_boxes):
            if index in matched:
                continue
            overlap = bbox_iou(tuple(pred["bbox"]), gt_box)
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index is not None and best_iou >= threshold:
            matched.add(best_index)
            tp += 1
    return tp, len(predictions) - tp, len(gt_boxes) - tp


def interpolated_ap(recall: Sequence[float], precision: Sequence[float]) -> float:
    if not recall:
        return 0.0
    total = 0.0
    for index in range(101):
        level = index / 100.0
        candidates = [p for r, p in zip(recall, precision) if r >= level]
        total += max(candidates) if candidates else 0.0
    return total / 101.0


def calculate_ap(predictions: Mapping[str, Sequence[Mapping[str, Any]]], gt: Mapping[str, Sequence[BBox]], iou_threshold: float) -> float:
    ranked = sorted(
        ((image_id, item) for image_id, items in predictions.items() for item in items),
        key=lambda pair: float(pair[1]["conf"]),
        reverse=True,
    )
    total_gt = sum(len(items) for items in gt.values())
    matched: Dict[str, set[int]] = defaultdict(set)
    tp_sum = fp_sum = 0
    recalls: List[float] = []
    precisions: List[float] = []
    for image_id, pred in ranked:
        best_index: Optional[int] = None
        best_iou = 0.0
        for index, gt_box in enumerate(gt.get(image_id, [])):
            if index in matched[image_id]:
                continue
            overlap = bbox_iou(tuple(pred["bbox"]), gt_box)
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index is not None and best_iou >= iou_threshold:
            matched[image_id].add(best_index)
            tp_sum += 1
        else:
            fp_sum += 1
        recalls.append(tp_sum / total_gt if total_gt else 0.0)
        precisions.append(tp_sum / (tp_sum + fp_sum))
    return interpolated_ap(recalls, precisions)


def evaluate_channel_predictions(
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    gt_rows: Mapping[str, Mapping[str, Any]],
    confidence: float,
    iou_threshold: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    gt = {image_id: list(row.get("bboxes") or []) for image_id, row in gt_rows.items()}
    image_rows: List[Dict[str, Any]] = []
    tp = fp = fn = hit_images = all_matched_images = 0
    for image_id, gt_boxes in gt.items():
        selected = [item for item in predictions.get(image_id, []) if float(item["conf"]) >= confidence]
        image_tp, image_fp, image_fn = match_at_iou(selected, gt_boxes, iou_threshold)
        tp += image_tp
        fp += image_fp
        fn += image_fn
        hit_images += int(image_tp > 0)
        all_matched_images += int(bool(gt_boxes) and image_fn == 0)
        image_rows.append(
            {
                "image_id": image_id,
                "provider_or_ui": gt_rows[image_id].get("provider_or_ui", ""),
                "gt_box_count": len(gt_boxes),
                "prediction_count": len(selected),
                "tp": image_tp,
                "fp": image_fp,
                "fn": image_fn,
                "any_gt_hit": int(image_tp > 0),
                "all_gt_boxes_matched": int(bool(gt_boxes) and image_fn == 0),
            }
        )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    ap50 = calculate_ap(predictions, gt, 0.5)
    ap_values = [calculate_ap(predictions, gt, 0.5 + step * 0.05) for step in range(10)]
    summary = {
        "image_count": len(gt),
        "gt_box_count": sum(len(items) for items in gt.values()),
        "confidence_threshold": confidence,
        "iou_threshold": iou_threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "image_hit_rate": hit_images / len(gt) if gt else 0.0,
        "all_gt_boxes_matched_image_rate": all_matched_images / len(gt) if gt else 0.0,
        "ap50": ap50,
        "map50_95": sum(ap_values) / len(ap_values),
        "ap_by_iou": {f"{0.5 + step * 0.05:.2f}": value for step, value in enumerate(ap_values)},
    }
    return summary, image_rows


def self_test() -> None:
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    predictions = {"a": [{"bbox": (0, 0, 10, 10), "conf": 0.9}]}
    rows = {"a": {"bboxes": [(0, 0, 10, 10)], "provider_or_ui": "UI_01"}}
    summary, _ = evaluate_channel_predictions(predictions, rows, 0.25, 0.5)
    assert summary["tp"] == 1 and summary["fp"] == 0 and summary["fn"] == 0
    assert summary["ap50"] == 1.0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.json"
        write_json(path, {"ok": True})
        assert json.loads(path.read_text(encoding="utf-8"))["ok"] is True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gt-rows-json", type=Path)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--raw-conf", type=float, default=0.001)
    parser.add_argument("--candidate-conf", type=float, default=0.05)
    parser.add_argument("--eval-conf", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("export_recursive_detector_predictions self-test passed")
        return
    missing = [name for name in ("model", "images_dir", "output_dir") if getattr(args, name) is None]
    if missing:
        parser.error("the following arguments are required: " + ", ".join("--" + name.replace("_", "-") for name in missing))

    from ultralytics import YOLO

    images = iter_images(args.images_dir)
    stems = [path.stem for path in images]
    if len(stems) != len(set(stems)):
        raise SystemExit("image stems are not globally unique; flat YOLO labels would collide")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw_labels_conf001"
    candidate_dir = args.output_dir / "labels"
    raw_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    pending = [
        path
        for path in images
        if not (raw_dir / f"{path.stem}.txt").exists() or not (candidate_dir / f"{path.stem}.txt").exists()
    ]
    completed = len(images) - len(pending)
    if completed:
        print(f"resume: {completed}/{len(images)} labels already exist", flush=True)
    for start in range(0, len(pending), args.batch):
        chunk = pending[start : start + args.batch]
        results = model.predict(
            source=[str(path) for path in chunk],
            imgsz=args.imgsz,
            conf=args.raw_conf,
            device=args.device,
            batch=len(chunk),
            stream=False,
            verbose=False,
        )
        if len(results) != len(chunk):
            raise RuntimeError(f"detector returned {len(results)} results for a {len(chunk)}-image chunk")
        for source_path, result in zip(chunk, results):
            image_id = source_path.stem
            height, width = result.orig_shape
            raw_lines: List[str] = []
            candidate_lines: List[str] = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu().tolist()
                classes = result.boxes.cls.detach().cpu().tolist()
                confidences = result.boxes.conf.detach().cpu().tolist()
                for box, cls_value, conf_value in zip(boxes, classes, confidences):
                    bbox = tuple(float(value) for value in box[:4])
                    cls = int(cls_value)
                    conf = float(conf_value)
                    line = to_yolo_line(cls, bbox, conf, float(width), float(height))
                    raw_lines.append(line)
                    if conf >= args.candidate_conf:
                        candidate_lines.append(line)
            (raw_dir / f"{image_id}.txt").write_text(
                "\n".join(raw_lines) + ("\n" if raw_lines else ""), encoding="utf-8"
            )
            (candidate_dir / f"{image_id}.txt").write_text(
                "\n".join(candidate_lines) + ("\n" if candidate_lines else ""), encoding="utf-8"
            )
        completed += len(chunk)
        if args.progress_every and (completed % args.progress_every < len(chunk) or completed == len(images)):
            print(f"processed {completed}/{len(images)}", flush=True)

    channel_predictions = load_exported_channel_predictions(raw_dir, images)

    summary: Dict[str, Any] = {
        "model": str(args.model),
        "images_dir": str(args.images_dir),
        "image_count": len(images),
        "gt_used_for_inference": False,
        "raw_conf": args.raw_conf,
        "candidate_conf": args.candidate_conf,
        "raw_label_dir": str(raw_dir),
        "candidate_label_dir": str(candidate_dir),
    }
    if args.gt_rows_json:
        gt_rows = load_gt_rows(args.gt_rows_json)
        detection, image_rows = evaluate_channel_predictions(
            channel_predictions, gt_rows, args.eval_conf, args.iou_threshold
        )
        summary["channel_number_detection"] = detection
        write_csv(args.output_dir / "per_image_detection.csv", image_rows)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
