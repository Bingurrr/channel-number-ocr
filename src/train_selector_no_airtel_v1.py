"""Train selector_no_airtel_v1 from no-Airtel candidate pools.

This wrapper keeps Airtel out of training, validation, threshold tuning, hard
negative mining, and feature selection. GT is used only to create supervised
labels and evaluation metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from candidate_ranker import (
    SOURCE_DETAIL_OPTIONS,
    annotate_candidate_pool_features,
    candidate_features,
    normalize_source_alias,
)
from channel_number_fusion import digits, numeric_equivalent_digits, resolve_image_path, yolo_boxes
from no_airtel_policy import assert_input_tree_allowed, assert_no_airtel_reference
from train_candidate_ranker import (
    build_matrix,
    build_pair_diffs,
    image_balanced_weights,
    standardize_fit,
    train_pairwise_linear,
    train_pointwise_logistic,
)


DEFAULT_TRAIN_CANDIDATES = Path("teacher_model_v2/dataset/exports/no_airtel_train_candidates_v1.json")
DEFAULT_VAL_CANDIDATES = Path("teacher_model_v2/dataset/exports/no_airtel_val_candidates_v1.json")
DEFAULT_TRAIN_GT = Path("teacher_model_v2/dataset/exports/no_airtel_candidate_pool_v1/gt/no_airtel_train_gt_v1.json")
DEFAULT_VAL_GT = Path("teacher_model_v2/dataset/exports/no_airtel_candidate_pool_v1/gt/no_airtel_val_gt_v1.json")
DEFAULT_YOLO_ROOT = Path("teacher_model_v2/dataset/yolo/no_airtel_13providers_v1")
DEFAULT_FEATURES_CSV = Path("teacher_model_v2/dataset/selector_training/no_airtel_selector_v1_features.csv")
DEFAULT_OUT_DIR = Path("teacher_model_v2/experiments/generalization/no_airtel_train_airtel_test_v1/selector")
DEFAULT_NO_UI_CANDIDATES = Path("teacher_model_v2/dataset/exports/fusion_dishtv_no_ui_slot_proposals_raw.json")
DEFAULT_NO_UI_IMAGES = Path("teacher_model_v2/dataset/sequence_test/dishtv_images")

METADATA_COLUMNS = [
    "split",
    "provider",
    "image_id",
    "image_path",
    "candidate_id",
    "candidate_index",
    "candidate_text",
    "candidate_digits",
    "numeric_candidate",
    "source",
    "raw_source",
    "bbox",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "gt_channel_number",
    "numeric_gt",
    "label_exact",
    "label_numeric_equiv",
    "label_train",
]


def rel(path: Path) -> str:
    return str(path).replace("\\", "/")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names: List[str] = list(fieldnames or [])
    if not names:
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(out) or math.isinf(out):
        return 0.0
    return out


def provider_from_image_id(image_id: str) -> str:
    for provider in [
        "Virtual_1",
        "Virtual_2",
        "Virtual_3",
        "Virtual_4",
        "Virtual_5",
        "DirecTV",
        "Spectrum",
        "Xfinity",
        "LG_U+",
        "Dlive",
        "SKT",
        "KT",
        "An",
    ]:
        if image_id.startswith(provider + "_"):
            return provider
    if image_id.lower().startswith("dishtv"):
        return "DishTV_no_ui"
    return image_id.split("_", 1)[0] if "_" in image_id else "unknown"


def load_gt(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    data = read_json(path)
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if str(value).strip():
                out[str(key)] = {"channel_number": str(value).strip(), "provider": provider_from_image_id(str(key))}
        return out
    for item in data:
        if not isinstance(item, Mapping):
            continue
        image_id = str(item.get("image_id") or item.get("id") or "")
        value = str(item.get("channel_number") or item.get("ground_truth_channel_number") or "").strip()
        if image_id and value:
            out[image_id] = {
                "channel_number": value,
                "provider": str(item.get("provider") or provider_from_image_id(image_id)),
            }
    return out


def candidate_bbox(candidate: Mapping[str, Any]) -> Optional[List[float]]:
    value = candidate.get("bbox_xyxy") or candidate.get("bbox") or candidate.get("box") or candidate.get("xyxy")
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    except (TypeError, ValueError):
        return None


def ensure_candidate_bbox(candidate: Dict[str, Any]) -> bool:
    box = candidate_bbox(candidate)
    if box is None:
        return False
    candidate["bbox_xyxy"] = box
    candidate.setdefault("bbox", box)
    return True


def image_size(image: Mapping[str, Any], images_dir: Path, image_id: str) -> Tuple[Path, int, int]:
    image_path = resolve_image_path(Path(str(image.get("image_path") or "")), images_dir, image_id)
    if image_path.exists():
        with Image.open(image_path) as src:
            return image_path, int(src.width), int(src.height)
    return image_path, int(image.get("image_width") or 1), int(image.get("image_height") or 1)


def source_name(candidate: Mapping[str, Any]) -> str:
    return normalize_source_alias(candidate.get("source") or candidate.get("raw_source") or "unknown")


def raw_source_name(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("raw_source") or candidate.get("source") or "unknown")


def build_rows_for_split(
    *,
    split: str,
    candidates_json: Path,
    images_dir: Path,
    yolo_label_dir: Path,
    gt_json: Optional[Path],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    doc = read_json(candidates_json)
    gt = load_gt(gt_json)
    rows: List[Dict[str, Any]] = []
    image_count = 0
    skipped_no_gt = 0
    skipped_no_bbox = 0
    oracle_hits = 0
    source_counts: Counter[str] = Counter()

    for image in doc.get("images", []):
        if not isinstance(image, dict):
            continue
        image_id = str(image.get("image_id") or image.get("image_name") or "")
        if not image_id:
            continue
        truth = str((gt.get(image_id) or {}).get("channel_number") or "")
        if split != "no_ui_val" and not truth:
            skipped_no_gt += 1
            continue
        image_count += 1
        image_path, width, height = image_size(image, images_dir, image_id)
        yolo = yolo_boxes(yolo_label_dir, image_id, width, height)
        candidates = [candidate for candidate in image.get("candidates", []) if isinstance(candidate, dict)]
        for candidate in candidates:
            ensure_candidate_bbox(candidate)
        annotate_candidate_pool_features(candidates, yolo, width, height)

        numeric_gt_exact = digits(truth)
        numeric_gt_equiv = numeric_equivalent_digits(truth) if truth else ""
        image_has_positive = False
        for index, candidate in enumerate(candidates, 1):
            if not ensure_candidate_bbox(candidate):
                skipped_no_bbox += 1
                continue
            candidate_digits = digits(str(candidate.get("text", "")))
            if not candidate_digits:
                continue
            features = candidate_features(candidate, yolo, width, height)
            numeric_candidate_equiv = numeric_equivalent_digits(candidate_digits)
            label_exact = int(bool(numeric_gt_exact) and candidate_digits == numeric_gt_exact)
            label_numeric_equiv = int(bool(numeric_gt_equiv) and numeric_candidate_equiv == numeric_gt_equiv)
            image_has_positive = image_has_positive or bool(label_exact)
            box = candidate["bbox_xyxy"]
            source = source_name(candidate)
            source_counts[source] += 1
            provider = str((gt.get(image_id) or {}).get("provider") or provider_from_image_id(image_id))
            row: Dict[str, Any] = {
                "split": split,
                "provider": provider,
                "image_id": image_id,
                "image_path": rel(image_path),
                "candidate_id": str(candidate.get("candidate_id") or candidate.get("id") or f"candidate_{index:04d}"),
                "candidate_index": index,
                "candidate_text": str(candidate.get("text", "")),
                "candidate_digits": candidate_digits,
                "numeric_candidate": numeric_candidate_equiv,
                "source": source,
                "raw_source": raw_source_name(candidate),
                "bbox": json.dumps([round(float(v), 3) for v in box], separators=(",", ":")),
                "bbox_x1": round(float(box[0]), 3),
                "bbox_y1": round(float(box[1]), 3),
                "bbox_x2": round(float(box[2]), 3),
                "bbox_y2": round(float(box[3]), 3),
                "gt_channel_number": truth,
                "numeric_gt": numeric_gt_equiv,
                "label_exact": label_exact,
                "label_numeric_equiv": label_numeric_equiv,
                "label_train": label_exact,
            }
            row.update(features)
            rows.append(row)
        oracle_hits += int(image_has_positive)

    summary = {
        "split": split,
        "candidate_json": rel(candidates_json),
        "images_dir": rel(images_dir),
        "yolo_label_dir": rel(yolo_label_dir),
        "gt_json": rel(gt_json) if gt_json else None,
        "image_count": image_count,
        "row_count": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
        "oracle_hit_count": oracle_hits if split != "no_ui_val" else 0,
        "candidate_oracle_exact_accuracy": None if split == "no_ui_val" or image_count == 0 else round(oracle_hits / image_count, 6),
        "skipped_no_gt": skipped_no_gt,
        "skipped_no_bbox": skipped_no_bbox,
    }
    return rows, summary


def infer_feature_names(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    names: List[str] = []
    for key in rows[0].keys():
        if key in METADATA_COLUMNS:
            continue
        if all(isinstance(row.get(key), (int, float, np.integer, np.floating)) for row in rows[:200]):
            names.append(key)
    return names


def score_rows(rows: Sequence[Dict[str, Any]], model: Mapping[str, Any]) -> np.ndarray:
    feature_names = [str(name) for name in model["feature_names"]]
    x = build_matrix(rows, feature_names)
    mean = np.array(model["mean"], dtype=np.float64)
    std = np.array(model["std"], dtype=np.float64)
    std[np.abs(std) < 1e-8] = 1.0
    weights = np.array(model["weights"], dtype=np.float64)
    return ((x - mean) / std) @ weights + float(model.get("bias", 0.0))


def group_indexes_by_image(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        out[str(row["image_id"])].append(index)
    return out


def select_best(rows: Sequence[Dict[str, Any]], scores: np.ndarray) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for image_id, indexes in group_indexes_by_image(rows).items():
        best_idx = max(indexes, key=lambda idx: float(scores[idx]))
        row = rows[best_idx]
        exact_exists = any(int(rows[idx].get("label_exact", 0)) for idx in indexes)
        numeric_exists = any(int(rows[idx].get("label_numeric_equiv", 0)) for idx in indexes)
        selected.append(
            {
                "split": row["split"],
                "provider": row["provider"],
                "image_id": image_id,
                "score": float(scores[best_idx]),
                "prediction": row["candidate_digits"],
                "prediction_numeric": row["numeric_candidate"],
                "ground_truth": row["gt_channel_number"],
                "ground_truth_numeric": row["numeric_gt"],
                "selected_candidate_id": row["candidate_id"],
                "selected_text": row["candidate_text"],
                "selected_source": row["source"],
                "selected_raw_source": row["raw_source"],
                "selected_exact": int(row.get("label_exact", 0)),
                "selected_numeric_equiv": int(row.get("label_numeric_equiv", 0)),
                "correct_candidate_exists": int(exact_exists),
                "numeric_equiv_candidate_exists": int(numeric_exists),
            }
        )
    return selected


def threshold_grid(scores: Sequence[float], steps: int) -> List[float]:
    if not scores:
        return [0.0]
    values = sorted(float(score) for score in scores)
    if values[0] == values[-1]:
        return [float("-inf"), values[0], float("inf")]
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, steps)).tolist()
    return sorted(set([float("-inf"), float("inf")] + [float(v) for v in quantiles]))


def evaluate_selected(selected: Sequence[Mapping[str, Any]], threshold: float) -> Dict[str, Any]:
    valid = [item for item in selected if str(item.get("split")) != "no_ui_val"]
    no_ui = [item for item in selected if str(item.get("split")) == "no_ui_val"]
    output_items = [item for item in valid if safe_float(item.get("score")) >= threshold]
    no_ui_output_items = [item for item in no_ui if safe_float(item.get("score")) >= threshold]
    exact_correct = sum(int(item.get("selected_exact", 0)) for item in output_items)
    numeric_correct = sum(int(item.get("selected_numeric_equiv", 0)) for item in output_items)
    valid_count = len(valid)
    output_count = len(output_items)
    oracle_hit_count = sum(int(item.get("correct_candidate_exists", 0)) for item in valid)
    no_correct = valid_count - oracle_hit_count
    wrong = output_count - exact_correct
    no_output = valid_count - output_count
    not_selected = sum(
        1
        for item in output_items
        if int(item.get("correct_candidate_exists", 0)) and not int(item.get("selected_exact", 0))
    )
    thresholded = sum(
        1
        for item in valid
        if int(item.get("correct_candidate_exists", 0)) and safe_float(item.get("score")) < threshold
    )
    return {
        "threshold": threshold,
        "valid_gt_count": valid_count,
        "output_count": output_count,
        "coverage": output_count / max(1, valid_count),
        "final_exact_correct_count": exact_correct,
        "final_numeric_equiv_correct_count": numeric_correct,
        "final_exact_accuracy": exact_correct / max(1, valid_count),
        "final_numeric_equiv_accuracy": numeric_correct / max(1, valid_count),
        "output_precision": exact_correct / max(1, output_count),
        "wrong_output_count": wrong,
        "no_output_count": no_output,
        "candidate_oracle_exact_accuracy": oracle_hit_count / max(1, valid_count),
        "oracle_hit_count": oracle_hit_count,
        "no_correct_candidate_count": no_correct,
        "selector_accuracy_given_correct_candidate_exists": exact_correct / max(1, oracle_hit_count),
        "correct_candidate_exists_but_not_selected_count": not_selected,
        "correct_candidate_exists_but_thresholded_or_rejected_count": thresholded,
        "no_ui_image_count": len(no_ui),
        "no_ui_fp": len(no_ui_output_items),
        "no_ui_fp_rate": len(no_ui_output_items) / max(1, len(no_ui)),
    }


def provider_metrics(selected: Sequence[Mapping[str, Any]], threshold: float, model_name: str) -> List[Dict[str, Any]]:
    by_provider: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in selected:
        if str(item.get("split")) == "no_ui_val":
            continue
        by_provider[str(item.get("provider") or "unknown")].append(item)
    rows: List[Dict[str, Any]] = []
    for provider, items in sorted(by_provider.items()):
        metrics = evaluate_selected(items, threshold)
        rows.append(
            {
                "model": model_name,
                "provider": provider,
                "images": metrics["valid_gt_count"],
                "exact_accuracy": round(metrics["final_exact_accuracy"], 6),
                "numeric_equiv_accuracy": round(metrics["final_numeric_equiv_accuracy"], 6),
                "coverage": round(metrics["coverage"], 6),
                "output_precision": round(metrics["output_precision"], 6),
                "wrong": metrics["wrong_output_count"],
                "no_output": metrics["no_output_count"],
                "candidate_oracle": round(metrics["candidate_oracle_exact_accuracy"], 6),
                "selector_accuracy": round(metrics["selector_accuracy_given_correct_candidate_exists"], 6),
            }
        )
    return rows


def write_predictions(path: Path, selected: Sequence[Mapping[str, Any]], threshold: float) -> None:
    rows: List[Dict[str, Any]] = []
    for item in selected:
        output = safe_float(item.get("score")) >= threshold
        rows.append(
            {
                **item,
                "threshold": threshold,
                "output": int(output),
                "final_prediction": item.get("prediction", "") if output else "",
                "exact_match": int(output and int(item.get("selected_exact", 0))),
                "numeric_equiv_match": int(output and int(item.get("selected_numeric_equiv", 0))),
            }
        )
    write_csv(path, rows)


def train_models(rows: Sequence[Dict[str, Any]], feature_names: Sequence[str], out_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    train_rows = [row for row in rows if row["split"] == "train"]
    x_train_raw = build_matrix(train_rows, feature_names)
    mean, std = standardize_fit(x_train_raw)
    x_train = (x_train_raw - mean) / std
    y_train = np.array([int(row["label_train"]) for row in train_rows], dtype=np.float64)
    weights = image_balanced_weights(train_rows)
    point_w, point_b, point_history = train_pointwise_logistic(
        x_train,
        y_train,
        weights,
        epochs=2500,
        lr=0.08,
        l2=0.0008,
    )
    pair_diffs = build_pair_diffs(train_rows, x_train, max_negatives_per_positive=24)
    pair_w, pair_history = train_pairwise_linear(
        pair_diffs,
        epochs=1800,
        lr=0.05,
        l2=0.0008,
    )
    models = {
        "pointwise_logistic": {
            "model_type": "linear_pointwise_logistic_ranker",
            "feature_names": list(feature_names),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "weights": point_w.tolist(),
            "bias": point_b,
            "label": "label_train",
            "policy": "no_airtel_train_airtel_final_test_only",
        },
        "pairwise_linear": {
            "model_type": "linear_pairwise_ranker",
            "feature_names": list(feature_names),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "weights": pair_w.tolist(),
            "bias": 0.0,
            "label": "label_train",
            "policy": "no_airtel_train_airtel_final_test_only",
        },
    }
    paths = {
        "pointwise_logistic": out_dir / "selector_no_airtel_v1_pointwise_logistic.json",
        "pairwise_linear": out_dir / "selector_no_airtel_v1_pairwise_linear.json",
    }
    for name, model in models.items():
        write_json(paths[name], model)
    training_summary = {
        "train_rows": len(train_rows),
        "train_images": len(group_indexes_by_image(train_rows)),
        "train_positive_rows": int(sum(int(row["label_train"]) for row in train_rows)),
        "feature_count": len(feature_names),
        "pair_count": int(len(pair_diffs)),
        "pointwise_loss_history": point_history,
        "pairwise_loss_history": pair_history,
        "model_paths": {name: rel(path) for name, path in paths.items()},
    }
    return models, training_summary


def tune_and_report(
    *,
    rows: Sequence[Dict[str, Any]],
    models: Mapping[str, Mapping[str, Any]],
    out_dir: Path,
    steps: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    sweep_rows: List[Dict[str, Any]] = []
    provider_rows: List[Dict[str, Any]] = []
    model_results: Dict[str, Any] = {}
    val_and_no_ui = [row for row in rows if row["split"] in {"val", "no_ui_val"}]
    for model_name, model in models.items():
        scores = score_rows(val_and_no_ui, model)
        selected = select_best(val_and_no_ui, scores)
        grid = threshold_grid([item["score"] for item in selected], steps)
        best_record: Optional[Dict[str, Any]] = None
        for threshold in grid:
            metrics = evaluate_selected(selected, threshold)
            record = {"model": model_name, **metrics}
            sweep_rows.append(record)
            key = (
                metrics["final_exact_accuracy"],
                -metrics["no_ui_fp_rate"],
                metrics["output_precision"],
                metrics["coverage"],
            )
            if best_record is None or key > best_record["_selection_key"]:
                best_record = {**record, "_selection_key": key}
        assert best_record is not None
        threshold = float(best_record["threshold"])
        threshold_doc = {
            "model": model_name,
            "selected_split": "val",
            "recommended_threshold": threshold,
            "recommended_threshold_metrics": {k: v for k, v in best_record.items() if k != "_selection_key"},
            "threshold_tuning_uses_airtel": False,
            "note": "Threshold selected on No-Airtel val only; no-UI FP is used only as a tie-breaker when available.",
        }
        threshold_path = out_dir / f"selector_no_airtel_v1_{model_name}_threshold.json"
        write_json(threshold_path, threshold_doc)
        write_predictions(out_dir / f"selector_no_airtel_v1_{model_name}_val_predictions.csv", selected, threshold)
        provider_model_rows = provider_metrics(selected, threshold, model_name)
        provider_rows.extend(provider_model_rows)
        worst = min((row["exact_accuracy"] for row in provider_model_rows), default=0.0)
        best = max((row["exact_accuracy"] for row in provider_model_rows), default=0.0)
        model_results[model_name] = {
            "threshold_path": rel(threshold_path),
            "threshold": threshold,
            "metrics": {k: v for k, v in best_record.items() if k != "_selection_key"},
            "provider_worst_accuracy": worst,
            "provider_best_accuracy": best,
        }
    return model_results, sweep_rows, provider_rows


def leakage_guard_report(
    *,
    out_dir: Path,
    train_candidates: Path,
    val_candidates: Path,
    train_gt: Path,
    val_gt: Path,
    no_ui_candidates: Optional[Path],
    no_ui_images: Optional[Path],
) -> Dict[str, Any]:
    checked: List[str] = []
    for path, role in [
        (train_candidates, "train"),
        (train_gt, "train"),
        (val_candidates, "validation"),
        (val_gt, "validation"),
    ]:
        assert_input_tree_allowed(path, role=role)
        checked.append(f"{role}: {rel(path)}")
    if no_ui_candidates and no_ui_candidates.exists():
        assert_input_tree_allowed(no_ui_candidates, role="validation")
        checked.append(f"validation no-ui candidates: {rel(no_ui_candidates)}")
    if no_ui_images and no_ui_images.exists():
        assert_input_tree_allowed(no_ui_images, role="validation", scan_content=False)
        checked.append(f"validation no-ui images: {rel(no_ui_images)}")
    doc = {
        "airtel_used_for_training": False,
        "airtel_used_for_validation": False,
        "airtel_used_for_threshold_tuning": False,
        "airtel_used_for_hard_negative_mining": False,
        "gt_used_for_candidate_generation": False,
        "checked_inputs": checked,
        "leakage_found": False,
    }
    lines = [
        "# No-Airtel Selector v1 Leakage Guard",
        "",
        "- airtel_used_for_training: false",
        "- airtel_used_for_validation: false",
        "- airtel_used_for_threshold_tuning: false",
        "- airtel_used_for_hard_negative_mining: false",
        "- gt_used_for_candidate_generation: false",
        "- leakage_found: false",
        "",
        "## Checked Inputs",
        "",
    ]
    lines.extend(f"- {item}" for item in checked)
    (out_dir / "leakage_guard_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return doc


def source_mapping() -> Dict[str, str]:
    return {name: f"source_detail_{name}" for name in SOURCE_DETAIL_OPTIONS}


def write_feature_schema(path: Path, feature_names: Sequence[str]) -> Dict[str, Any]:
    doc = {
        "feature_names": list(feature_names),
        "feature_count": len(feature_names),
        "excluded_metadata_columns": METADATA_COLUMNS,
        "source_feature_mapping": source_mapping(),
        "missing_feature_default_policy": "Missing numeric feature values are filled with 0.0 at training and inference.",
        "forbidden_as_model_features": [
            "image_id",
            "filename",
            "folder_name",
            "provider",
            "sequence_name",
            "gt_channel_number",
            "numeric_gt",
            "label_exact",
            "label_numeric_equiv",
            "label_train",
        ],
        "label_policy": {
            "positive_label": "digit-only exact match with leading zeros preserved",
            "numeric_equivalent": "recorded separately in label_numeric_equiv; not used as default positive label",
            "warning": "Image-level GT labels do not prove spatial correctness when the same number appears in multiple locations.",
        },
    }
    write_json(path, doc)
    return doc


def row_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_split = Counter(str(row["split"]) for row in rows)
    by_provider = Counter(str(row["provider"]) for row in rows if str(row["split"]) in {"train", "val"})
    by_source = Counter(str(row["source"]) for row in rows)
    by_image: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["split"]) in {"train", "val"}:
            by_image[str(row["image_id"])].append(row)
    return {
        "row_count": len(rows),
        "split_row_counts": dict(by_split),
        "provider_row_counts": dict(sorted(by_provider.items())),
        "source_row_counts": dict(sorted(by_source.items())),
        "images_with_exact_positive": sum(any(int(row["label_exact"]) for row in items) for items in by_image.values()),
    }


def choose_best_model(model_results: Mapping[str, Any]) -> str:
    return max(
        model_results,
        key=lambda name: (
            model_results[name]["metrics"]["final_exact_accuracy"],
            model_results[name]["metrics"]["output_precision"],
            model_results[name]["metrics"]["coverage"],
            model_results[name]["provider_worst_accuracy"],
            -model_results[name]["metrics"]["no_ui_fp_rate"],
        ),
    )


def build_training_report(
    *,
    feature_summary: Mapping[str, Any],
    split_summaries: Sequence[Mapping[str, Any]],
    training_summary: Mapping[str, Any],
    model_results: Mapping[str, Any],
    selected_model: str,
    no_ui_negative_absent: bool,
) -> str:
    lines = [
        "# selector_no_airtel_v1 Training Report",
        "",
        "## Policy",
        "",
        "- Airtel used for training: false",
        "- Airtel used for validation/tuning: false",
        "- Airtel used for hard negative mining: false",
        "- Previous Airtel-specialized selector_v2_oldsplit_rc1 is not used.",
        "- GT is used only for No-Airtel selector labels and No-Airtel val evaluation.",
        "",
        "## Label Policy",
        "",
        "- Positive label: digit-only exact match, leading zeros preserved.",
        "- Numeric-equivalent match is recorded separately and is not the default positive label.",
        "- Warning: image-level GT labels can be spatially ambiguous when the same number appears in multiple places.",
        "",
        "## Feature Dataset",
        "",
        f"- rows: {feature_summary['row_count']}",
        f"- split rows: {feature_summary['split_row_counts']}",
        f"- source rows: {feature_summary['source_row_counts']}",
        f"- feature_count: {training_summary['feature_count']}",
        f"- no_ui_negative_absent: {str(no_ui_negative_absent).lower()}",
        "",
        "## Candidate Oracle",
        "",
    ]
    for summary in split_summaries:
        if summary["split"] == "no_ui_val":
            continue
        lines.append(
            f"- {summary['split']}: images={summary['image_count']} oracle={summary['candidate_oracle_exact_accuracy']} "
            f"hits={summary['oracle_hit_count']}"
        )
    lines.extend(
        [
            "",
            "## Model Comparison",
            "",
            "| model | threshold | val exact accuracy | val numeric-equivalent accuracy | coverage | output precision | wrong | no_output | candidate_oracle | selector_accuracy | provider worst accuracy | provider best accuracy | no_ui_fp |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name, result in model_results.items():
        metrics = result["metrics"]
        lines.append(
            f"| {model_name} | {result['threshold']:.6f} | {metrics['final_exact_accuracy']:.6f} | "
            f"{metrics['final_numeric_equiv_accuracy']:.6f} | {metrics['coverage']:.6f} | "
            f"{metrics['output_precision']:.6f} | {metrics['wrong_output_count']} | {metrics['no_output_count']} | "
            f"{metrics['candidate_oracle_exact_accuracy']:.6f} | {metrics['selector_accuracy_given_correct_candidate_exists']:.6f} | "
            f"{result['provider_worst_accuracy']:.6f} | {result['provider_best_accuracy']:.6f} | {metrics['no_ui_fp']} |"
        )
    lines.extend(
        [
            "",
            "## Selected Model",
            "",
            f"- selected_model: `{selected_model}`",
            f"- selected_threshold: {model_results[selected_model]['threshold']:.6f}",
            "",
            "## Notes",
            "",
            "- Threshold tuning used No-Airtel val only. DishTV no-UI, when present, is used only for FP reporting/tie-breaking.",
            "- Prompt 5 Airtel final holdout evaluation can proceed after this selector is wired into the frozen No-Airtel inference config.",
        ]
    )
    return "\n".join(lines) + "\n"


def self_test() -> None:
    assert provider_from_image_id("KT_KT_test_1") == "KT"
    assert provider_from_image_id("Virtual_5_Virtual_5_test_1") == "Virtual_5"
    assert digits("041") == "041"
    assert numeric_equivalent_digits("041") == "41"
    for forbidden in ("image_id", "provider", "gt_channel_number", "label_train"):
        assert forbidden in METADATA_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-candidates", type=Path, default=DEFAULT_TRAIN_CANDIDATES)
    parser.add_argument("--val-candidates", type=Path, default=DEFAULT_VAL_CANDIDATES)
    parser.add_argument("--train-gt", type=Path, default=DEFAULT_TRAIN_GT)
    parser.add_argument("--val-gt", type=Path, default=DEFAULT_VAL_GT)
    parser.add_argument("--yolo-root", type=Path, default=DEFAULT_YOLO_ROOT)
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-ui-candidates", type=Path, default=DEFAULT_NO_UI_CANDIDATES)
    parser.add_argument("--no-ui-images", type=Path, default=DEFAULT_NO_UI_IMAGES)
    parser.add_argument("--threshold-steps", type=int, default=300)
    parser.add_argument("--disable-no-ui", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("train_selector_no_airtel_v1 self-test passed")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in [args.train_candidates, args.val_candidates, args.train_gt, args.val_gt, args.yolo_root]:
        if not path.exists():
            raise SystemExit(f"required path not found: {path}")
        assert_no_airtel_reference(path, role="train" if path in {args.train_candidates, args.train_gt} else "validation")

    no_ui_candidates = None if args.disable_no_ui or not args.no_ui_candidates.exists() else args.no_ui_candidates
    no_ui_images = None if args.disable_no_ui or not args.no_ui_images.exists() else args.no_ui_images
    leakage_doc = leakage_guard_report(
        out_dir=args.out_dir,
        train_candidates=args.train_candidates,
        val_candidates=args.val_candidates,
        train_gt=args.train_gt,
        val_gt=args.val_gt,
        no_ui_candidates=no_ui_candidates,
        no_ui_images=no_ui_images,
    )

    split_rows: List[Dict[str, Any]] = []
    split_summaries: List[Dict[str, Any]] = []
    for split, candidates_json, gt_json in [
        ("train", args.train_candidates, args.train_gt),
        ("val", args.val_candidates, args.val_gt),
    ]:
        rows, summary = build_rows_for_split(
            split=split,
            candidates_json=candidates_json,
            images_dir=args.yolo_root / "images" / split,
            yolo_label_dir=args.yolo_root / "labels" / split,
            gt_json=gt_json,
        )
        split_rows.extend(rows)
        split_summaries.append(summary)

    no_ui_negative_absent = True
    empty_yolo_dir = args.out_dir / "_empty_yolo_labels"
    empty_yolo_dir.mkdir(parents=True, exist_ok=True)
    if no_ui_candidates and no_ui_images:
        no_ui_rows, no_ui_summary = build_rows_for_split(
            split="no_ui_val",
            candidates_json=no_ui_candidates,
            images_dir=no_ui_images,
            yolo_label_dir=empty_yolo_dir,
            gt_json=None,
        )
        split_rows.extend(no_ui_rows)
        split_summaries.append(no_ui_summary)
        no_ui_negative_absent = False

    if not split_rows:
        raise SystemExit("no feature rows built")
    feature_names = infer_feature_names(split_rows)
    feature_schema = write_feature_schema(args.out_dir / "feature_schema.json", feature_names)
    write_csv(args.features_csv, split_rows, fieldnames=METADATA_COLUMNS + list(feature_names))
    models, training_summary = train_models(split_rows, feature_names, args.out_dir)
    model_results, sweep_rows, provider_rows = tune_and_report(
        rows=split_rows,
        models=models,
        out_dir=args.out_dir,
        steps=args.threshold_steps,
    )
    write_csv(args.out_dir / "threshold_sweep.csv", sweep_rows)
    write_csv(args.out_dir / "provider_breakdown.csv", provider_rows)

    selected_model = choose_best_model(model_results)
    selected_artifacts = {
        "pointwise_logistic": args.out_dir / "selector_no_airtel_v1_pointwise_logistic.json",
        "pairwise_linear": args.out_dir / "selector_no_airtel_v1_pairwise_linear.json",
    }
    manifest = {
        "name": "selector_no_airtel_v1",
        "selected_model": selected_model,
        "selected_selector_artifact": rel(selected_artifacts[selected_model]),
        "selected_threshold": model_results[selected_model]["threshold"],
        "selected_threshold_artifact": model_results[selected_model]["threshold_path"],
        "feature_schema_path": rel(args.out_dir / "feature_schema.json"),
        "features_csv": rel(args.features_csv),
        "train_candidate_path": rel(args.train_candidates),
        "val_candidate_path": rel(args.val_candidates),
        "detector_artifact_used_for_candidate_generation": "detector_no_airtel_13providers_v1",
        "no_ui_setting": {
            "no_ui_negative_absent": no_ui_negative_absent,
            "no_ui_candidates": rel(no_ui_candidates) if no_ui_candidates else None,
        },
        "airtel_used_for_training": False,
        "airtel_used_for_validation": False,
        "airtel_used_for_threshold_tuning": False,
        "airtel_used_for_hard_negative_mining": False,
        "gt_used_for_candidate_generation": False,
        "model_results": model_results,
        "leakage_guard": leakage_doc,
    }
    write_json(args.out_dir / "selector_no_airtel_v1_manifest.json", manifest)
    summary_doc = {
        "feature_summary": row_summary(split_rows),
        "split_summaries": split_summaries,
        "feature_schema": feature_schema,
        "training_summary": training_summary,
        "model_results": model_results,
        "selected_model": selected_model,
        "no_ui_negative_absent": no_ui_negative_absent,
    }
    write_json(args.out_dir / "training_summary.json", summary_doc)
    (args.out_dir / "training_report.md").write_text(
        build_training_report(
            feature_summary=summary_doc["feature_summary"],
            split_summaries=split_summaries,
            training_summary=training_summary,
            model_results=model_results,
            selected_model=selected_model,
            no_ui_negative_absent=no_ui_negative_absent,
        ),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
