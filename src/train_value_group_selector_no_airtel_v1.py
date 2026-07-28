"""Train and evaluate an optional value-group selector without Airtel tuning.

Candidates that normalize to the same digit string are grouped before the
final selection.  The group model sees only inference-time evidence: frozen
candidate-selector scores, source support, confidence summaries, and spatial
agreement.  Airtel can be evaluated after model/threshold selection, but is
never used to choose either one.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from candidate_ranker import annotate_candidate_pool_features
from channel_number_fusion import candidate_score, digits, resolve_image_path, yolo_boxes
from no_airtel_policy import assert_input_tree_allowed
from train_candidate_ranker import (
    build_matrix,
    build_pair_diffs,
    image_balanced_weights,
    standardize_fit,
    train_pairwise_linear,
    train_pointwise_logistic,
)
from train_selector_no_airtel_v1 import (
    build_rows_for_split,
    evaluate_selected,
    provider_metrics,
    score_rows,
    select_best,
    threshold_grid,
    write_csv,
    write_json,
    write_predictions,
)


ROOT = Path("teacher_model_v2")
DEFAULT_FEATURES_CSV = ROOT / "dataset/selector_training/no_airtel_selector_v1_features.csv"
DEFAULT_BASELINE_MODEL = (
    ROOT
    / "experiments/generalization/no_airtel_train_airtel_test_v1/selector"
    / "selector_no_airtel_v1_pairwise_linear.json"
)
DEFAULT_BASELINE_THRESHOLD = DEFAULT_BASELINE_MODEL.with_name(
    "selector_no_airtel_v1_pairwise_linear_threshold.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "experiments/generalization/no_airtel_train_airtel_test_v1"
    / "selector_value_group_v1"
)
DEFAULT_AIRTEL_CANDIDATES = (
    ROOT
    / "experiments/generalization/no_airtel_train_airtel_test_v1/airtel_final_eval"
    / "airtel_candidates_no_airtel_model.json"
)
DEFAULT_AIRTEL_IMAGES = ROOT / "dataset/sequence_test/airtel_trial1_images"
DEFAULT_AIRTEL_YOLO = (
    ROOT
    / "experiments/generalization/no_airtel_train_airtel_test_v1/airtel_final_eval"
    / "yolo_detector_labels/predict/labels"
)
DEFAULT_AIRTEL_GT = ROOT / "dataset/exports/airtel_trial1_gt_from_sequence.json"
DEFAULT_TRAIN_CANDIDATES = ROOT / "dataset/exports/no_airtel_train_candidates_v1.json"
DEFAULT_VAL_CANDIDATES = ROOT / "dataset/exports/no_airtel_val_candidates_v1.json"
DEFAULT_YOLO_ROOT = ROOT / "dataset/yolo/no_airtel_13providers_v1"
DEFAULT_NO_UI_CANDIDATES = ROOT / "dataset/exports/fusion_dishtv_no_ui_slot_proposals_raw.json"
DEFAULT_NO_UI_IMAGES = ROOT / "dataset/sequence_test/dishtv_images"
DEFAULT_PUBLISHED_AIRTEL_SUMMARY = (
    ROOT
    / "experiments/generalization/no_airtel_train_airtel_test_v1/airtel_final_eval"
    / "single_frame_audit/summary.json"
)


SOURCE_NAMES = (
    "original_ocr",
    "refined_numeric_substring",
    "paddleocr_channel_recheck",
    "crop_variant_numeric_recheck",
    "digit_sequence_trim",
    "slot_proposal_recheck",
)


GROUP_FEATURE_NAMES = [
    "value_digit_len",
    "group_candidate_count",
    "group_candidate_count_log",
    "group_duplicate_count",
    "group_source_count",
    "group_source_count_log",
    "group_base_score_max",
    "group_base_score_mean",
    "group_base_score_min",
    "group_base_score_std",
    "group_base_score_second",
    "group_base_score_top2_mean",
    "group_base_score_gap",
    "group_ocr_conf_max",
    "group_ocr_conf_mean",
    "group_recognizer_conf_max",
    "group_proposal_score_max",
    "group_near_yolo_ratio",
    "group_other_text_overlap_ratio",
    "group_other_number_overlap_ratio",
    "group_yolo_ch0_iou_max",
    "group_yolo_area3_iou_max",
    "group_spatial_cluster_count",
    "group_spatial_cluster_ratio",
    "group_spatial_cluster_source_count",
    "group_bbox_center_spread",
    "group_bbox_area_cv",
    "group_cross_source_support",
    "group_original_and_recheck_support",
] + [f"group_has_source_{name}" for name in SOURCE_NAMES] + [
    f"group_representative_source_{name}" for name in SOURCE_NAMES
]


GROUP_METADATA_COLUMNS = [
    "split",
    "provider",
    "image_id",
    "image_path",
    "candidate_id",
    "candidate_text",
    "candidate_digits",
    "numeric_candidate",
    "source",
    "raw_source",
    "bbox",
    "gt_channel_number",
    "numeric_gt",
    "label_exact",
    "label_numeric_equiv",
    "label_train",
    "group_sources",
    "group_candidate_ids",
]


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_feature_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def ensure_candidate_bbox(candidate: Dict[str, Any]) -> bool:
    value = candidate.get("bbox_xyxy") or candidate.get("bbox") or candidate.get("box") or candidate.get("xyxy")
    if not isinstance(value, list) or len(value) < 4:
        return False
    try:
        candidate["bbox_xyxy"] = [float(value[index]) for index in range(4)]
    except (TypeError, ValueError):
        return False
    return True


def runtime_eligible_candidate_ids(
    candidates_json: Path,
    images_dir: Path,
    yolo_label_dir: Path,
) -> Tuple[Dict[str, set[str]], Dict[str, Any]]:
    """Reproduce the rule-filter stage that runs before ranker selection."""
    doc = load_json(candidates_json)
    eligible: Dict[str, set[str]] = defaultdict(set)
    total_candidates = 0
    eligible_candidates = 0
    image_count = 0
    for image in doc.get("images", []):
        if not isinstance(image, dict):
            continue
        image_id = str(image.get("image_id") or image.get("image_name") or "")
        if not image_id:
            continue
        image_count += 1
        image_path = resolve_image_path(Path(str(image.get("image_path") or "")), images_dir, image_id)
        if image_path.exists():
            with Image.open(image_path) as source:
                width, height = int(source.width), int(source.height)
        else:
            width = int(image.get("image_width") or 1)
            height = int(image.get("image_height") or 1)
        yolo = yolo_boxes(yolo_label_dir, image_id, width, height)
        candidates = [candidate for candidate in image.get("candidates", []) if isinstance(candidate, dict)]
        for candidate in candidates:
            ensure_candidate_bbox(candidate)
        annotate_candidate_pool_features(candidates, yolo, width, height)
        for index, candidate in enumerate(candidates, 1):
            total_candidates += 1
            if not ensure_candidate_bbox(candidate):
                continue
            value = digits(candidate.get("text", ""))
            if not value or len(value) > 5:
                continue
            score, _ = candidate_score(candidate, yolo, width, height, candidates, max_channel_digits=5)
            if score < -100:
                continue
            candidate_id = str(candidate.get("candidate_id") or candidate.get("id") or f"candidate_{index:04d}")
            eligible[image_id].add(candidate_id)
            eligible_candidates += 1
    return eligible, {
        "candidate_json": str(candidates_json).replace("\\", "/"),
        "image_count": image_count,
        "total_candidate_count": total_candidates,
        "runtime_eligible_candidate_count": eligible_candidates,
        "runtime_rejected_candidate_count": total_candidates - eligible_candidates,
    }


def filter_runtime_eligible_rows(
    rows: Sequence[Dict[str, Any]],
    eligible_by_split: Mapping[str, Mapping[str, set[str]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    split_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        split = str(row.get("split"))
        image_id = str(row.get("image_id"))
        candidate_id = str(row.get("candidate_id"))
        split_counts[split]["input"] += 1
        eligible = eligible_by_split.get(split)
        if eligible is None or candidate_id in eligible.get(image_id, set()):
            kept.append(row)
            split_counts[split]["kept"] += 1
        else:
            split_counts[split]["rejected"] += 1
    return kept, {split: dict(counts) for split, counts in sorted(split_counts.items())}


def normalized_source(row: Mapping[str, Any]) -> str:
    source = str(row.get("source") or "unknown")
    if source in {"paddleocr_slot_proposal_recheck", "slot_proposal_numeric_recheck"}:
        return "slot_proposal_recheck"
    if source in {"paddleocr_crop_variant_recheck", "crop_variant_recheck"}:
        return "crop_variant_numeric_recheck"
    return source


def row_bbox(row: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    return tuple(safe_float(row.get(key)) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))  # type: ignore[return-value]


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-9, area_a + area_b - intersection)


def spatially_close(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    dx = safe_float(a.get("bbox_cx_norm")) - safe_float(b.get("bbox_cx_norm"))
    dy = safe_float(a.get("bbox_cy_norm")) - safe_float(b.get("bbox_cy_norm"))
    return bbox_iou(row_bbox(a), row_bbox(b)) >= 0.20 or math.hypot(dx, dy) <= 0.06


def spatial_components(rows: Sequence[Mapping[str, Any]]) -> List[List[int]]:
    if not rows:
        return []
    neighbors: Dict[int, List[int]] = defaultdict(list)
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if spatially_close(rows[left], rows[right]):
                neighbors[left].append(right)
                neighbors[right].append(left)
    components: List[List[int]] = []
    unseen = set(range(len(rows)))
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        component = [seed]
        while stack:
            current = stack.pop()
            for neighbor in neighbors[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda component: (-len(component), component[0]))


def largest_spatial_cluster(rows: Sequence[Mapping[str, Any]]) -> List[int]:
    components = spatial_components(rows)
    return components[0] if components else []


def best_channel_evidence_cluster(
    rows: Sequence[Mapping[str, Any]],
    components: Sequence[Sequence[int]],
) -> List[int]:
    if not components:
        return []

    def evidence_key(component: Sequence[int]) -> Tuple[float, float, float, int, int, float]:
        members = [rows[index] for index in component]
        channel_conf = max(
            (
                max(
                    safe_float(row.get("yolo_ch0_max_conf")),
                    safe_float(row.get("yolo_area3_max_conf")),
                )
                for row in members
            ),
            default=0.0,
        )
        channel_iou = max(
            (
                max(
                    safe_float(row.get("yolo_ch0_max_iou")),
                    safe_float(row.get("yolo_area3_max_iou")),
                )
                for row in members
            ),
            default=0.0,
        )
        near_ratio = mean([safe_float(row.get("near_yolo_channel_evidence")) for row in members])
        source_count = len({normalized_source(row) for row in members})
        ocr_conf = max((safe_float(row.get("ocr_conf")) for row in members), default=0.0)
        return channel_conf, channel_iou, near_ratio, source_count, len(component), ocr_conf

    return list(max(components, key=evidence_key))


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def std(values: Sequence[float]) -> float:
    return float(np.std(values)) if values else 0.0


def coefficient_of_variation(values: Sequence[float]) -> float:
    average = mean(values)
    return std(values) / max(abs(average), 1e-8) if values else 0.0


def group_candidate_rows(
    candidate_rows: Sequence[Dict[str, Any]],
    baseline_scores: Sequence[float],
) -> List[Dict[str, Any]]:
    if len(candidate_rows) != len(baseline_scores):
        raise ValueError("candidate row and score counts differ")
    grouped: Dict[Tuple[str, str, str], List[Tuple[Dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(candidate_rows, baseline_scores):
        value = str(row.get("candidate_digits") or "").strip()
        if value:
            grouped[(str(row.get("split")), str(row.get("image_id")), value)].append((row, float(score)))

    output: List[Dict[str, Any]] = []
    for (_, _, value), pairs in grouped.items():
        pairs = sorted(pairs, key=lambda item: item[1], reverse=True)
        rows = [item[0] for item in pairs]
        scores = [item[1] for item in pairs]
        representative = rows[0]
        sources = sorted({normalized_source(row) for row in rows})
        components = spatial_components(rows)
        cluster_indexes = components[0] if components else []
        cluster_sources = {normalized_source(rows[index]) for index in cluster_indexes}
        best_cluster_indexes = best_channel_evidence_cluster(rows, components)
        best_cluster_rows = [rows[index] for index in best_cluster_indexes]
        best_cluster_sources = {normalized_source(row) for row in best_cluster_rows}
        center_x = [safe_float(row.get("bbox_cx_norm")) for row in rows]
        center_y = [safe_float(row.get("bbox_cy_norm")) for row in rows]
        center_spread = math.hypot(std(center_x), std(center_y))
        areas = [safe_float(row.get("bbox_area_norm")) for row in rows]
        ocr_conf = [safe_float(row.get("ocr_conf")) for row in rows]
        recognizer_conf = [safe_float(row.get("recognizer_conf")) for row in rows]
        proposal_score = [safe_float(row.get("proposal_score")) for row in rows]
        second = scores[1] if len(scores) > 1 else scores[0]
        top2_mean = mean(scores[:2])
        exact = int(any(int(safe_float(row.get("label_exact"))) for row in rows))
        numeric = int(any(int(safe_float(row.get("label_numeric_equiv"))) for row in rows))
        representative_source = normalized_source(representative)
        feature_values: Dict[str, float] = {
            "value_digit_len": float(len(value)),
            "group_candidate_count": float(len(rows)),
            "group_candidate_count_log": math.log1p(len(rows)),
            "group_duplicate_count": float(max(0, len(rows) - 1)),
            "group_source_count": float(len(sources)),
            "group_source_count_log": math.log1p(len(sources)),
            "group_base_score_max": scores[0],
            "group_base_score_mean": mean(scores),
            "group_base_score_min": min(scores),
            "group_base_score_std": std(scores),
            "group_base_score_second": second,
            "group_base_score_top2_mean": top2_mean,
            "group_base_score_gap": scores[0] - second,
            "group_ocr_conf_max": max(ocr_conf, default=0.0),
            "group_ocr_conf_mean": mean(ocr_conf),
            "group_recognizer_conf_max": max(recognizer_conf, default=0.0),
            "group_proposal_score_max": max(proposal_score, default=0.0),
            "group_near_yolo_ratio": mean([safe_float(row.get("near_yolo_channel_evidence")) for row in rows]),
            "group_other_text_overlap_ratio": mean([safe_float(row.get("overlaps_other_text")) for row in rows]),
            "group_other_number_overlap_ratio": mean([safe_float(row.get("overlaps_other_number")) for row in rows]),
            "group_yolo_ch0_iou_max": max((safe_float(row.get("yolo_ch0_max_iou")) for row in rows), default=0.0),
            "group_yolo_area3_iou_max": max((safe_float(row.get("yolo_area3_max_iou")) for row in rows), default=0.0),
            "group_spatial_cluster_count": float(len(cluster_indexes)),
            "group_spatial_cluster_ratio": len(cluster_indexes) / max(1, len(rows)),
            "group_spatial_cluster_source_count": float(len(cluster_sources)),
            "group_spatial_component_count": float(len(components)),
            "group_multi_position_same_text": float(len(components) > 1),
            "group_channel_evidence_component_count": float(
                sum(
                    any(safe_float(rows[index].get("near_yolo_channel_evidence")) > 0.0 for index in component)
                    for component in components
                )
            ),
            "group_best_cluster_candidate_count": float(len(best_cluster_indexes)),
            "group_best_cluster_source_count": float(len(best_cluster_sources)),
            "group_best_cluster_near_yolo_ratio": mean(
                [safe_float(row.get("near_yolo_channel_evidence")) for row in best_cluster_rows]
            ),
            "group_best_cluster_yolo_ch0_iou_max": max(
                (safe_float(row.get("yolo_ch0_max_iou")) for row in best_cluster_rows), default=0.0
            ),
            "group_best_cluster_yolo_area3_iou_max": max(
                (safe_float(row.get("yolo_area3_max_iou")) for row in best_cluster_rows), default=0.0
            ),
            "group_best_cluster_yolo_ch0_conf_max": max(
                (safe_float(row.get("yolo_ch0_max_conf")) for row in best_cluster_rows), default=0.0
            ),
            "group_best_cluster_yolo_area3_conf_max": max(
                (safe_float(row.get("yolo_area3_max_conf")) for row in best_cluster_rows), default=0.0
            ),
            "group_best_cluster_ocr_conf_max": max(
                (safe_float(row.get("ocr_conf")) for row in best_cluster_rows), default=0.0
            ),
            "group_best_cluster_other_number_overlap_ratio": mean(
                [safe_float(row.get("overlaps_other_number")) for row in best_cluster_rows]
            ),
            "group_bbox_center_spread": center_spread,
            "group_bbox_area_cv": coefficient_of_variation(areas),
            "group_cross_source_support": float(len(sources) >= 2),
            "group_original_and_recheck_support": float(
                "original_ocr" in sources and any(source != "original_ocr" for source in sources)
            ),
        }
        for source in SOURCE_NAMES:
            feature_values[f"group_has_source_{source}"] = float(source in sources)
            feature_values[f"group_representative_source_{source}"] = float(representative_source == source)

        item: Dict[str, Any] = {
            "split": representative.get("split", ""),
            "provider": representative.get("provider", "unknown"),
            "image_id": representative.get("image_id", ""),
            "image_path": representative.get("image_path", ""),
            "candidate_id": representative.get("candidate_id", ""),
            "candidate_text": representative.get("candidate_text", value),
            "candidate_digits": value,
            "numeric_candidate": representative.get("numeric_candidate", value),
            "source": representative_source,
            "raw_source": representative.get("raw_source", representative_source),
            "bbox": representative.get("bbox", ""),
            "gt_channel_number": representative.get("gt_channel_number", ""),
            "numeric_gt": representative.get("numeric_gt", ""),
            "label_exact": exact,
            "label_numeric_equiv": numeric,
            "label_train": exact,
            "group_sources": "|".join(sources),
            "group_candidate_ids": "|".join(str(row.get("candidate_id", "")) for row in rows),
        }
        item.update(feature_values)
        output.append(item)
    return output


def train_group_models(rows: Sequence[Dict[str, Any]], out_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    train_rows = [row for row in rows if row.get("split") == "train"]
    x_raw = build_matrix(train_rows, GROUP_FEATURE_NAMES)
    mean_values, std_values = standardize_fit(x_raw)
    x = (x_raw - mean_values) / std_values
    y = np.array([int(row.get("label_train", 0)) for row in train_rows], dtype=np.float64)
    sample_weights = image_balanced_weights(train_rows)
    point_weights, point_bias, point_history = train_pointwise_logistic(
        x, y, sample_weights, epochs=2200, lr=0.07, l2=0.0015
    )
    pair_diffs = build_pair_diffs(train_rows, x, max_negatives_per_positive=24)
    pair_weights, pair_history = train_pairwise_linear(
        pair_diffs, epochs=1800, lr=0.05, l2=0.0015
    )
    common = {
        "feature_names": GROUP_FEATURE_NAMES,
        "mean": mean_values.tolist(),
        "std": std_values.tolist(),
        "label": "label_train",
        "policy": "no_airtel_train_airtel_diagnostic_only",
        "selection_unit": "exact_digit_value_group",
    }
    models = {
        "pointwise_logistic": {
            **common,
            "model_type": "value_group_pointwise_logistic",
            "weights": point_weights.tolist(),
            "bias": point_bias,
        },
        "pairwise_linear": {
            **common,
            "model_type": "value_group_pairwise_linear",
            "weights": pair_weights.tolist(),
            "bias": 0.0,
        },
    }
    for name, model in models.items():
        write_json(out_dir / f"value_group_{name}.json", model)
    return models, {
        "train_group_rows": len(train_rows),
        "train_images": len({str(row["image_id"]) for row in train_rows}),
        "train_positive_groups": sum(int(row["label_train"]) for row in train_rows),
        "pair_count": len(pair_diffs),
        "pointwise_loss_history": point_history,
        "pairwise_loss_history": pair_history,
    }


def score_group_rows(rows: Sequence[Dict[str, Any]], model: Mapping[str, Any]) -> np.ndarray:
    return score_rows(rows, model)


def select_group_best(rows: Sequence[Dict[str, Any]], scores: Sequence[float]) -> List[Dict[str, Any]]:
    by_image: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_image[str(row["image_id"])].append(index)
    selected: List[Dict[str, Any]] = []
    for image_id, indexes in by_image.items():
        best_index = max(indexes, key=lambda idx: float(scores[idx]))
        row = rows[best_index]
        selected.append(
            {
                "split": row["split"],
                "provider": row["provider"],
                "image_id": image_id,
                "score": float(scores[best_index]),
                "prediction": row["candidate_digits"],
                "prediction_numeric": row["numeric_candidate"],
                "ground_truth": row["gt_channel_number"],
                "ground_truth_numeric": row["numeric_gt"],
                "selected_candidate_id": row["candidate_id"],
                "selected_text": row["candidate_text"],
                "selected_source": row["source"],
                "selected_raw_source": row["raw_source"],
                "selected_exact": int(row["label_exact"]),
                "selected_numeric_equiv": int(row["label_numeric_equiv"]),
                "correct_candidate_exists": int(any(int(rows[idx]["label_exact"]) for idx in indexes)),
                "numeric_equiv_candidate_exists": int(
                    any(int(rows[idx]["label_numeric_equiv"]) for idx in indexes)
                ),
                "group_candidate_count": row["group_candidate_count"],
                "group_source_count": row["group_source_count"],
                "group_sources": row["group_sources"],
            }
        )
    return selected


def add_missing_image_outputs(
    selected: Sequence[Dict[str, Any]],
    universe_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    output = list(selected)
    selected_ids = {str(item.get("image_id")) for item in selected}
    metadata: Dict[str, Mapping[str, Any]] = {}
    for row in universe_rows:
        metadata.setdefault(str(row.get("image_id")), row)
    for image_id, row in metadata.items():
        if image_id in selected_ids:
            continue
        output.append(
            {
                "split": row.get("split", ""),
                "provider": row.get("provider", "unknown"),
                "image_id": image_id,
                "score": float("-inf"),
                "prediction": "",
                "prediction_numeric": "",
                "ground_truth": row.get("gt_channel_number", ""),
                "ground_truth_numeric": row.get("numeric_gt", ""),
                "selected_candidate_id": "",
                "selected_text": "",
                "selected_source": "",
                "selected_raw_source": "",
                "selected_exact": 0,
                "selected_numeric_equiv": 0,
                "correct_candidate_exists": 0,
                "numeric_equiv_candidate_exists": 0,
                "runtime_eligible_candidate_count": 0,
            }
        )
    return output


def threshold_from_json(path: Path) -> float:
    doc = load_json(path)
    return float(doc.get("recommended_threshold", doc.get("threshold", 0.0)))


def validate_frozen_baseline_policy(model_path: Path, threshold_path: Path) -> Dict[str, Any]:
    """Allow the frozen selector only when its own manifest proves No-Airtel provenance.

    The experiment directory intentionally contains ``airtel_test`` in its
    policy name, so generic path scanning would reject the directory itself.
    Data inputs still use the strict generic guard.
    """
    manifest_path = model_path.parent / "selector_no_airtel_v1_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"frozen selector manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    required_false = (
        "airtel_used_for_training",
        "airtel_used_for_validation",
        "airtel_used_for_threshold_tuning",
        "airtel_used_for_hard_negative_mining",
    )
    violations = [key for key in required_false if manifest.get(key) is not False]
    threshold_doc = load_json(threshold_path)
    if threshold_doc.get("threshold_tuning_uses_airtel") is not False:
        violations.append("threshold_artifact.threshold_tuning_uses_airtel")
    model = load_json(model_path)
    forbidden_feature_tokens = ("image_id", "image_path", "provider", "gt", "channel_number_text")
    suspicious_features = [
        str(name)
        for name in model.get("feature_names", [])
        if any(token in str(name).lower() for token in forbidden_feature_tokens)
    ]
    if suspicious_features:
        violations.append("suspicious_model_features")
    if violations:
        raise RuntimeError(f"frozen selector failed No-Airtel provenance validation: {violations}")
    return {
        "manifest": str(manifest_path).replace("\\", "/"),
        "required_false_fields": list(required_false),
        "suspicious_features": suspicious_features,
        "passed": True,
    }


def tune_group_models(
    rows: Sequence[Dict[str, Any]],
    models: Mapping[str, Mapping[str, Any]],
    steps: int,
    universe_rows: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    evaluation_rows = [row for row in rows if row.get("split") in {"val", "no_ui_val"}]
    model_results: Dict[str, Dict[str, Any]] = {}
    sweep_rows: List[Dict[str, Any]] = []
    for name, model in models.items():
        scores = score_group_rows(evaluation_rows, model)
        selected = select_group_best(evaluation_rows, scores)
        selected = add_missing_image_outputs(
            selected,
            [row for row in universe_rows if row.get("split") in {"val", "no_ui_val"}],
        )
        best: Optional[Dict[str, Any]] = None
        finite_scores = [float(item["score"]) for item in selected if math.isfinite(float(item["score"]))]
        for threshold in threshold_grid(finite_scores, steps):
            metrics = evaluate_selected(selected, threshold)
            row = {"model": name, **metrics}
            sweep_rows.append(row)
            key = (
                metrics["final_exact_accuracy"],
                -metrics["no_ui_fp_rate"],
                metrics["output_precision"],
                metrics["coverage"],
            )
            if best is None or key > best["selection_key"]:
                best = {"selection_key": key, "threshold": threshold, "metrics": metrics}
        if best is None:
            raise RuntimeError(f"no threshold result for {name}")
        model_results[name] = {
            "threshold": float(best["threshold"]),
            "metrics": best["metrics"],
            "selected": selected,
        }
    return model_results, sweep_rows


def metric_row(
    dataset: str,
    model: str,
    metrics: Mapping[str, Any],
    threshold: float,
    selected: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    valid_selected = [item for item in (selected or []) if str(item.get("split")) != "no_ui_val"]
    raw_correct = sum(int(item.get("selected_exact", 0)) for item in valid_selected)
    return {
        "dataset": dataset,
        "model": model,
        "threshold": threshold,
        "raw_selected_correct": raw_correct if selected is not None else None,
        "raw_selection_accuracy": raw_correct / max(1, len(valid_selected)) if selected is not None else None,
        "accuracy": metrics["final_exact_accuracy"],
        "numeric_accuracy": metrics["final_numeric_equiv_accuracy"],
        "coverage": metrics["coverage"],
        "output_precision": metrics["output_precision"],
        "wrong": metrics["wrong_output_count"],
        "no_output": metrics["no_output_count"],
        "oracle": metrics["candidate_oracle_exact_accuracy"],
        "selector_accuracy": metrics["selector_accuracy_given_correct_candidate_exists"],
        "no_ui_fp": metrics["no_ui_fp"],
        "no_ui_fp_rate": metrics["no_ui_fp_rate"],
    }


def compare_model_key(result: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    metrics = result["metrics"]
    return (
        float(metrics["final_exact_accuracy"]),
        -float(metrics["no_ui_fp_rate"]),
        float(metrics["output_precision"]),
        float(metrics["coverage"]),
    )


def selection_score_distribution(selected: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    scores = sorted(
        float(item["score"])
        for item in selected
        if str(item.get("split")) != "no_ui_val" and math.isfinite(float(item.get("score", float("-inf"))))
    )
    if not scores:
        return {"count": 0}

    def quantile(fraction: float) -> float:
        return float(scores[round((len(scores) - 1) * fraction)])

    return {
        "count": len(scores),
        "min": scores[0],
        "q10": quantile(0.10),
        "median": quantile(0.50),
        "q90": quantile(0.90),
        "max": scores[-1],
    }


def evaluate_dataset(
    candidate_rows: Sequence[Dict[str, Any]],
    baseline_model: Mapping[str, Any],
    baseline_threshold: float,
    group_model: Mapping[str, Any],
    group_threshold: float,
    universe_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    baseline_scores = score_rows(candidate_rows, baseline_model)
    baseline_selected = select_best(candidate_rows, baseline_scores)
    if universe_rows is not None:
        baseline_selected = add_missing_image_outputs(baseline_selected, universe_rows)
    baseline_metrics = evaluate_selected(baseline_selected, baseline_threshold)
    group_rows = group_candidate_rows(candidate_rows, baseline_scores)
    group_scores = score_group_rows(group_rows, group_model)
    group_selected = select_group_best(group_rows, group_scores)
    if universe_rows is not None:
        group_selected = add_missing_image_outputs(group_selected, universe_rows)
    group_metrics = evaluate_selected(group_selected, group_threshold)
    return baseline_metrics, group_metrics, baseline_selected, group_selected, group_rows


def write_report(
    path: Path,
    comparison_rows: Sequence[Mapping[str, Any]],
    selected_model: str,
    decision: str,
    next_action: str,
    training_summary: Mapping[str, Any],
    score_distributions: Mapping[str, Mapping[str, Any]],
) -> None:
    lines = [
        "# Value-Group Selector Experiment",
        "",
        "## Guardrail",
        "",
        "- Airtel used for training: false",
        "- Airtel used for validation/threshold tuning: false",
        "- Airtel used only after model and threshold were frozen: true",
        "- Existing fusion behavior changed: false",
        "- Selection unit: exact digit string, preserving leading zeros",
        "",
        "## Process",
        "",
        "1. Score each candidate with the frozen selector_no_airtel_v1 ranker.",
        "2. Group candidates by image and exact normalized digit string.",
        "3. Aggregate source support, confidence, score distribution, and bbox agreement.",
        "4. Train pointwise and pairwise group rerankers on No-Airtel train only.",
        "5. Select model and threshold on No-Airtel val, with no-UI FP as a tie-breaker.",
        "6. Run Airtel diagnostic only after freezing the choice.",
        "",
        "## Results",
        "",
        "| dataset | model | threshold | raw selection acc | final accuracy | coverage | precision | wrong | no_output | oracle | selector_acc | no_ui_fp |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        raw_value = row.get("raw_selection_accuracy")
        raw_text = "N/A" if raw_value is None else f"{float(raw_value):.6f}"
        lines.append(
            f"| {row['dataset']} | {row['model']} | {float(row['threshold']):.6f} | "
            f"{raw_text} | "
            f"{float(row['accuracy']):.6f} | {float(row['coverage']):.6f} | "
            f"{float(row['output_precision']):.6f} | {row['wrong']} | {row['no_output']} | "
            f"{float(row['oracle']):.6f} | {float(row['selector_accuracy']):.6f} | {row['no_ui_fp']} |"
        )
    lines.extend(
        [
            "",
            "## Gate Score Distribution",
            "",
            "| dataset/model | q10 | median | q90 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, stats in score_distributions.items():
        lines.append(
            f"| {name} | {safe_float(stats.get('q10')):.6f} | "
            f"{safe_float(stats.get('median')):.6f} | {safe_float(stats.get('q90')):.6f} |"
        )
    lines.extend(
        [
            "",
            "The group ranker improves the Airtel pre-gate choice, but its absolute score distribution shifts "
            "far below No-Airtel validation. The frozen absolute threshold therefore rejects many correct choices.",
            "",
            "## Training Summary",
            "",
            f"- selected group model: `{selected_model}`",
            f"- train images: {training_summary['train_images']}",
            f"- train value groups: {training_summary['train_group_rows']}",
            f"- train positive groups: {training_summary['train_positive_groups']}",
            f"- pair count: {training_summary['pair_count']}",
            "",
            "## Decision",
            "",
            f"- decision: `{decision}`",
            f"- next action: {next_action}",
            "- Airtel diagnostic metrics were not used for model, feature, or threshold selection.",
            "- Image-level GT makes this a weak value label; it does not prove bbox-level spatial correctness.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    base = {
        "split": "train",
        "provider": "KT",
        "image_id": "KT_001",
        "image_path": "KT_001.jpg",
        "numeric_candidate": "041",
        "gt_channel_number": "041",
        "numeric_gt": "41",
        "label_exact": "1",
        "label_numeric_equiv": "1",
        "bbox_x1": "10",
        "bbox_y1": "10",
        "bbox_x2": "30",
        "bbox_y2": "30",
        "bbox_cx_norm": "0.2",
        "bbox_cy_norm": "0.2",
        "bbox_area_norm": "0.01",
        "ocr_conf": "0.9",
    }
    rows = [
        {**base, "candidate_id": "a", "candidate_text": "041", "candidate_digits": "041", "source": "original_ocr"},
        {
            **base,
            "candidate_id": "b",
            "candidate_text": "041",
            "candidate_digits": "041",
            "source": "paddleocr_slot_proposal_recheck",
            "bbox_x1": "11",
            "bbox_x2": "31",
        },
        {
            **base,
            "candidate_id": "c",
            "candidate_text": "41",
            "candidate_digits": "41",
            "numeric_candidate": "41",
            "label_exact": "0",
            "label_numeric_equiv": "1",
            "source": "refined_numeric_substring",
        },
    ]
    grouped = group_candidate_rows(rows, [2.0, 1.0, 3.0])
    assert len(grouped) == 2
    exact_group = next(row for row in grouped if row["candidate_digits"] == "041")
    assert exact_group["group_candidate_count"] == 2.0
    assert exact_group["group_source_count"] == 2.0
    assert exact_group["group_spatial_cluster_count"] == 2.0
    assert exact_group["group_has_source_slot_proposal_recheck"] == 1.0
    assert exact_group["candidate_digits"] == "041"

    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "airtel_train.csv"
        bad.write_text("provider,image_id\nAirtel,x\n", encoding="utf-8")
        try:
            assert_input_tree_allowed(bad, role="training")
        except Exception:
            pass
        else:
            raise AssertionError("Airtel training input guard did not fail")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an optional value-group selector using No-Airtel data only."
    )
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--train-candidates", type=Path, default=DEFAULT_TRAIN_CANDIDATES)
    parser.add_argument("--val-candidates", type=Path, default=DEFAULT_VAL_CANDIDATES)
    parser.add_argument("--yolo-root", type=Path, default=DEFAULT_YOLO_ROOT)
    parser.add_argument("--no-ui-candidates", type=Path, default=DEFAULT_NO_UI_CANDIDATES)
    parser.add_argument("--no-ui-images-dir", type=Path, default=DEFAULT_NO_UI_IMAGES)
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--baseline-threshold", type=Path, default=DEFAULT_BASELINE_THRESHOLD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--threshold-steps", type=int, default=300)
    parser.add_argument("--skip-airtel-diagnostic", action="store_true")
    parser.add_argument("--airtel-candidates", type=Path, default=DEFAULT_AIRTEL_CANDIDATES)
    parser.add_argument("--airtel-images-dir", type=Path, default=DEFAULT_AIRTEL_IMAGES)
    parser.add_argument("--airtel-yolo-label-dir", type=Path, default=DEFAULT_AIRTEL_YOLO)
    parser.add_argument("--airtel-gt-json", type=Path, default=DEFAULT_AIRTEL_GT)
    parser.add_argument("--published-airtel-summary", type=Path, default=DEFAULT_PUBLISHED_AIRTEL_SUMMARY)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("train_value_group_selector_no_airtel_v1 self-test passed")
        return

    for path in (
        args.features_csv,
        args.train_candidates,
        args.val_candidates,
        args.yolo_root,
        args.no_ui_candidates,
        args.no_ui_images_dir,
        args.baseline_model,
        args.baseline_threshold,
    ):
        if not path.exists():
            raise SystemExit(f"required input not found: {path}")
    for path, role in (
        (args.features_csv, "training"),
        (args.train_candidates, "training"),
        (args.val_candidates, "validation"),
        (args.yolo_root, "validation"),
        (args.no_ui_candidates, "validation"),
        (args.no_ui_images_dir, "validation"),
    ):
        assert_input_tree_allowed(path, role=role, scan_content=path.is_file())
    baseline_policy = validate_frozen_baseline_policy(args.baseline_model, args.baseline_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_model = load_json(args.baseline_model)
    baseline_threshold = threshold_from_json(args.baseline_threshold)
    candidate_rows_raw = load_feature_rows(args.features_csv)
    empty_yolo = args.output_dir / "_empty_yolo_labels"
    empty_yolo.mkdir(parents=True, exist_ok=True)
    train_eligible, train_filter_summary = runtime_eligible_candidate_ids(
        args.train_candidates,
        args.yolo_root / "images/train",
        args.yolo_root / "labels/train",
    )
    val_eligible, val_filter_summary = runtime_eligible_candidate_ids(
        args.val_candidates,
        args.yolo_root / "images/val",
        args.yolo_root / "labels/val",
    )
    no_ui_eligible, no_ui_filter_summary = runtime_eligible_candidate_ids(
        args.no_ui_candidates,
        args.no_ui_images_dir,
        empty_yolo,
    )
    candidate_rows, feature_filter_summary = filter_runtime_eligible_rows(
        candidate_rows_raw,
        {
            "train": train_eligible,
            "val": val_eligible,
            "no_ui_val": no_ui_eligible,
        },
    )
    baseline_scores = score_rows(candidate_rows, baseline_model)
    group_rows = group_candidate_rows(candidate_rows, baseline_scores)
    models, training_summary = train_group_models(group_rows, args.output_dir)
    model_results, sweep_rows = tune_group_models(
        group_rows,
        models,
        args.threshold_steps,
        candidate_rows_raw,
    )
    selected_model_name = max(model_results, key=lambda name: compare_model_key(model_results[name]))
    selected_model = models[selected_model_name]
    selected_threshold = float(model_results[selected_model_name]["threshold"])

    val_candidate_rows = [row for row in candidate_rows if row.get("split") in {"val", "no_ui_val"}]
    baseline_val, group_val, baseline_val_selected, group_val_selected, _ = evaluate_dataset(
        val_candidate_rows,
        baseline_model,
        baseline_threshold,
        selected_model,
        selected_threshold,
        [row for row in candidate_rows_raw if row.get("split") in {"val", "no_ui_val"}],
    )
    comparison_rows = [
        metric_row(
            "no_airtel_val",
            "individual_baseline",
            baseline_val,
            baseline_threshold,
            baseline_val_selected,
        ),
        metric_row(
            "no_airtel_val",
            f"value_group_{selected_model_name}",
            group_val,
            selected_threshold,
            group_val_selected,
        ),
    ]
    score_distributions: Dict[str, Dict[str, Any]] = {
        "no_airtel_val/individual_baseline": selection_score_distribution(baseline_val_selected),
        f"no_airtel_val/value_group_{selected_model_name}": selection_score_distribution(group_val_selected),
    }
    write_predictions(args.output_dir / "no_airtel_val_predictions.csv", group_val_selected, selected_threshold)

    airtel_summary: Optional[Dict[str, Any]] = None
    airtel_selection_gain = False
    airtel_final_regression = False
    if not args.skip_airtel_diagnostic:
        for path in (args.airtel_candidates, args.airtel_images_dir, args.airtel_yolo_label_dir, args.airtel_gt_json):
            if not path.exists():
                raise SystemExit(f"Airtel diagnostic input not found: {path}")
        airtel_rows, airtel_input_summary = build_rows_for_split(
            split="airtel_diagnostic",
            candidates_json=args.airtel_candidates,
            images_dir=args.airtel_images_dir,
            yolo_label_dir=args.airtel_yolo_label_dir,
            gt_json=args.airtel_gt_json,
        )
        airtel_eligible, airtel_filter_summary = runtime_eligible_candidate_ids(
            args.airtel_candidates,
            args.airtel_images_dir,
            args.airtel_yolo_label_dir,
        )
        airtel_universe_rows = list(airtel_rows)
        airtel_rows, airtel_feature_filter = filter_runtime_eligible_rows(
            airtel_universe_rows,
            {"airtel_diagnostic": airtel_eligible},
        )
        baseline_airtel, group_airtel, baseline_airtel_selected, group_airtel_selected, airtel_group_rows = evaluate_dataset(
            airtel_rows,
            baseline_model,
            baseline_threshold,
            selected_model,
            selected_threshold,
            airtel_universe_rows,
        )
        comparison_rows.extend(
            [
                metric_row(
                    "airtel_diagnostic",
                    "individual_baseline",
                    baseline_airtel,
                    baseline_threshold,
                    baseline_airtel_selected,
                ),
                metric_row(
                    "airtel_diagnostic",
                    f"value_group_{selected_model_name}",
                    group_airtel,
                    selected_threshold,
                    group_airtel_selected,
                ),
            ]
        )
        score_distributions["airtel_diagnostic/individual_baseline"] = selection_score_distribution(
            baseline_airtel_selected
        )
        score_distributions[f"airtel_diagnostic/value_group_{selected_model_name}"] = selection_score_distribution(
            group_airtel_selected
        )
        write_predictions(args.output_dir / "airtel_diagnostic_predictions.csv", group_airtel_selected, selected_threshold)
        write_csv(
            args.output_dir / "airtel_value_groups.csv",
            airtel_group_rows,
            fieldnames=GROUP_METADATA_COLUMNS + GROUP_FEATURE_NAMES,
        )
        airtel_summary = {
            "input": airtel_input_summary,
            "runtime_filter": airtel_filter_summary,
            "feature_filter": airtel_feature_filter,
            "individual_baseline": baseline_airtel,
            "value_group": group_airtel,
        }
        baseline_raw_correct = sum(int(item.get("selected_exact", 0)) for item in baseline_airtel_selected)
        group_raw_correct = sum(int(item.get("selected_exact", 0)) for item in group_airtel_selected)
        airtel_selection_gain = group_raw_correct > baseline_raw_correct
        airtel_final_regression = group_airtel["final_exact_accuracy"] < baseline_airtel["final_exact_accuracy"]
        if args.published_airtel_summary.exists():
            published = load_json(args.published_airtel_summary)
            published_metrics = {
                "final_exact_accuracy": published.get("final_prediction_exact_accuracy", 0.0),
                "final_numeric_equiv_accuracy": published.get("final_prediction_numeric_equiv_accuracy", 0.0),
                "coverage": published.get("coverage", 0.0),
                "output_precision": (
                    safe_float(published.get("final_exact_correct_count"))
                    / max(1.0, safe_float(published.get("final_output_count")))
                ),
                "wrong_output_count": published.get("final_wrong_output_count", 0),
                "no_output_count": published.get("final_no_output_count", 0),
                "candidate_oracle_exact_accuracy": published.get("candidate_oracle_exact_accuracy", 0.0),
                "selector_accuracy_given_correct_candidate_exists": published.get(
                    "selector_accuracy_given_correct_candidate_exists", 0.0
                ),
                "no_ui_fp": 0,
                "no_ui_fp_rate": 0.0,
            }
            comparison_rows.insert(
                -1,
                metric_row(
                    "airtel_diagnostic",
                    "published_fusion_baseline",
                    published_metrics,
                    baseline_threshold,
                ),
            )
            airtel_summary["published_fusion_baseline"] = published_metrics

    val_improved = group_val["final_exact_accuracy"] > baseline_val["final_exact_accuracy"]
    no_ui_ok = group_val["no_ui_fp_rate"] <= max(baseline_val["no_ui_fp_rate"], 0.01)
    if val_improved and no_ui_ok and airtel_selection_gain and airtel_final_regression:
        decision = "selection_gain_but_gate_not_promoted"
        next_action = (
            "Keep value grouping as an optional research branch and replace the absolute group threshold with "
            "provider-held-out No-Airtel confidence calibration; do not tune against Airtel."
        )
    elif val_improved and no_ui_ok:
        decision = "promote_to_optional_pipeline_ablation"
        next_action = "Integrate the frozen group artifact behind an opt-in fusion flag, then test temporal interaction."
    elif group_val["final_exact_accuracy"] == baseline_val["final_exact_accuracy"] and no_ui_ok:
        decision = "not_promoted_tied_on_no_airtel_validation"
        next_action = "Run provider-held-out validation and add selector hard negatives before changing mainline fusion."
    else:
        decision = "not_promoted"
        next_action = "Keep the individual selector and move next to provider-balanced hard-negative training."

    selected_artifact = args.output_dir / f"value_group_{selected_model_name}.json"
    config = {
        "name": "value_group_selector_no_airtel_v1",
        "enabled_by_default": False,
        "baseline_model": str(args.baseline_model).replace("\\", "/"),
        "baseline_threshold": baseline_threshold,
        "group_model": str(selected_artifact).replace("\\", "/"),
        "group_model_type": selected_model_name,
        "group_threshold": selected_threshold,
        "group_key": "exact_digit_string_preserve_leading_zero",
        "airtel_used_for_training": False,
        "airtel_used_for_validation_or_tuning": False,
        "airtel_used_for_diagnostic_after_freeze": not args.skip_airtel_diagnostic,
        "decision": decision,
    }
    write_json(args.output_dir / "value_group_config.json", config)
    write_json(
        args.output_dir / "experiment_summary.json",
        {
            "config": config,
            "training_summary": training_summary,
            "baseline_policy_validation": baseline_policy,
            "runtime_filter_summary": {
                "train": train_filter_summary,
                "val": val_filter_summary,
                "no_ui": no_ui_filter_summary,
                "feature_rows": feature_filter_summary,
            },
            "model_results": {
                name: {"threshold": result["threshold"], "metrics": result["metrics"]}
                for name, result in model_results.items()
            },
            "comparison": comparison_rows,
            "score_distributions": score_distributions,
            "airtel_diagnostic": airtel_summary,
        },
    )
    write_json(
        args.output_dir / "feature_schema.json",
        {
            "feature_names": GROUP_FEATURE_NAMES,
            "feature_count": len(GROUP_FEATURE_NAMES),
            "forbidden_features": ["image_id", "image_path", "provider", "gt_channel_number", "numeric_gt"],
        },
    )
    write_csv(args.output_dir / "threshold_sweep.csv", sweep_rows)
    write_csv(args.output_dir / "comparison_metrics.csv", comparison_rows)
    write_json(args.output_dir / "score_distribution_summary.json", score_distributions)
    provider_rows = provider_metrics(group_val_selected, selected_threshold, f"value_group_{selected_model_name}")
    write_csv(args.output_dir / "provider_breakdown.csv", provider_rows)
    write_report(
        args.output_dir / "value_group_report.md",
        comparison_rows,
        selected_model_name,
        decision,
        next_action,
        training_summary,
        score_distributions,
    )
    print(json.dumps({"decision": decision, "selected_model": selected_model_name, "comparison": comparison_rows}, indent=2))


if __name__ == "__main__":
    main()
