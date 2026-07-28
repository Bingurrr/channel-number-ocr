"""Calibrate a domain-robust gate for the No-Airtel value-group selector.

The value-group ranker is kept fixed.  This script trains a small confidence
gate from provider-held-out (OOF) No-Airtel selections and relative, per-image
features such as top1/top2 margin, score z-score, entropy, and evidence support.
Airtel is evaluated only after the gate model and threshold are frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from no_airtel_policy import assert_input_tree_allowed
from train_candidate_ranker import build_matrix, image_balanced_weights, standardize_fit, train_pointwise_logistic
from train_selector_no_airtel_v1 import (
    build_rows_for_split,
    evaluate_selected,
    score_rows,
    threshold_grid,
    write_csv,
    write_json,
    write_predictions,
)
from train_value_group_selector_no_airtel_v1 import (
    DEFAULT_AIRTEL_CANDIDATES,
    DEFAULT_AIRTEL_GT,
    DEFAULT_AIRTEL_IMAGES,
    DEFAULT_AIRTEL_YOLO,
    DEFAULT_BASELINE_MODEL,
    DEFAULT_FEATURES_CSV,
    DEFAULT_NO_UI_CANDIDATES,
    DEFAULT_NO_UI_IMAGES,
    DEFAULT_TRAIN_CANDIDATES,
    DEFAULT_VAL_CANDIDATES,
    DEFAULT_YOLO_ROOT,
    GROUP_FEATURE_NAMES,
    filter_runtime_eligible_rows,
    group_candidate_rows,
    load_feature_rows,
    load_json,
    runtime_eligible_candidate_ids,
    safe_float,
    score_group_rows,
    selection_score_distribution,
    validate_frozen_baseline_policy,
)


ROOT = Path("teacher_model_v2")
DEFAULT_GROUP_DIR = (
    ROOT
    / "experiments/generalization/no_airtel_train_airtel_test_v1"
    / "selector_value_group_v1"
)
DEFAULT_GROUP_MODEL = DEFAULT_GROUP_DIR / "value_group_pointwise_logistic.json"
DEFAULT_GROUP_CONFIG = DEFAULT_GROUP_DIR / "value_group_config.json"
DEFAULT_OUT_DIR = (
    ROOT
    / "experiments/generalization/no_airtel_train_airtel_test_v1"
    / "selector_value_group_relative_gate_v1"
)
DEFAULT_PUBLISHED_AIRTEL_SUMMARY = (
    ROOT
    / "experiments/generalization/no_airtel_train_airtel_test_v1/airtel_final_eval"
    / "single_frame_audit/summary.json"
)


GATE_FEATURE_NAMES = [
    "has_candidate",
    "pool_value_group_count",
    "pool_value_group_count_log",
    "pool_candidate_count",
    "pool_score_std",
    "top1_top2_margin",
    "top1_top2_margin_per_pool_std",
    "top1_score_z",
    "top1_softmax_probability",
    "pool_softmax_entropy_normalized",
    "single_value_group",
    "top1_candidate_support",
    "top1_source_support",
    "top1_spatial_cluster_ratio",
    "top1_spatial_source_support",
    "top1_cross_source_support",
    "top1_original_and_recheck_support",
    "top1_near_yolo_ratio",
    "top1_other_text_overlap_ratio",
    "top1_other_number_overlap_ratio",
    "top1_ocr_conf_max",
    "top1_value_digit_len",
    "top1_minus_top2_candidate_support",
    "top1_minus_top2_source_support",
    "top1_minus_top2_spatial_support",
]


MULTI_POSITION_GATE_FEATURE_NAMES = GATE_FEATURE_NAMES + [
    "top1_spatial_component_count",
    "top1_multi_position_same_text",
    "top1_channel_evidence_component_count",
    "top1_best_cluster_candidate_support",
    "top1_best_cluster_source_support",
    "top1_best_cluster_near_yolo_ratio",
    "top1_best_cluster_yolo_ch0_iou_max",
    "top1_best_cluster_yolo_area3_iou_max",
    "top1_best_cluster_yolo_ch0_conf_max",
    "top1_best_cluster_yolo_area3_conf_max",
    "top1_best_cluster_ocr_conf_max",
    "top1_best_cluster_other_number_overlap_ratio",
]


def stable_no_ui_partition(image_id: str, train_percent: int = 60) -> str:
    bucket = int(hashlib.sha1(image_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "no_ui_gate_train" if bucket < train_percent else "no_ui_val"


def fit_pointwise_model(
    rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    *,
    model_type: str,
    epochs: int,
    lr: float,
    l2: float,
) -> Dict[str, Any]:
    x_raw = build_matrix(rows, feature_names)
    mean, std = standardize_fit(x_raw)
    x = (x_raw - mean) / std
    labels = np.array([int(row.get("label_train", 0)) for row in rows], dtype=np.float64)
    weights = image_balanced_weights(rows)
    model_weights, bias, history = train_pointwise_logistic(
        x,
        labels,
        weights,
        epochs=epochs,
        lr=lr,
        l2=l2,
    )
    return {
        "model_type": model_type,
        "feature_names": list(feature_names),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": model_weights.tolist(),
        "bias": bias,
        "loss_history": history,
        "policy": "no_airtel_train_val_only_airtel_post_freeze_diagnostic",
    }


def softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    shifted = np.array(values, dtype=np.float64) - max(values)
    exponent = np.exp(np.clip(shifted, -60.0, 0.0))
    denominator = float(exponent.sum())
    return (exponent / max(denominator, 1e-12)).tolist()


def build_gate_rows(
    group_rows: Sequence[Dict[str, Any]],
    group_scores: Sequence[float],
    universe_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    by_image: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(group_rows):
        by_image[str(row["image_id"])].append(index)
    metadata: Dict[str, Mapping[str, Any]] = {}
    for row in universe_rows:
        metadata.setdefault(str(row.get("image_id")), row)

    output: List[Dict[str, Any]] = []
    for image_id, meta in metadata.items():
        indexes = sorted(by_image.get(image_id, []), key=lambda idx: float(group_scores[idx]), reverse=True)
        if not indexes:
            item: Dict[str, Any] = {
                "split": meta.get("split", ""),
                "provider": meta.get("provider", "unknown"),
                "image_id": image_id,
                "prediction": "",
                "prediction_numeric": "",
                "ground_truth": meta.get("gt_channel_number", ""),
                "ground_truth_numeric": meta.get("numeric_gt", ""),
                "selected_candidate_id": "",
                "selected_text": "",
                "selected_source": "",
                "selected_raw_source": "",
                "selected_exact": 0,
                "selected_numeric_equiv": 0,
                "correct_candidate_exists": 0,
                "numeric_equiv_candidate_exists": 0,
                "label_train": 0,
            }
            item.update({name: 0.0 for name in MULTI_POSITION_GATE_FEATURE_NAMES})
            output.append(item)
            continue

        top = group_rows[indexes[0]]
        runner = group_rows[indexes[1]] if len(indexes) > 1 else None
        scores = [float(group_scores[index]) for index in indexes]
        probabilities = softmax(scores)
        score_std = float(np.std(scores)) if len(scores) > 1 else 0.0
        score_mean = float(np.mean(scores))
        margin = scores[0] - scores[1] if len(scores) > 1 else 0.0
        entropy = -sum(probability * math.log(max(probability, 1e-12)) for probability in probabilities)
        entropy_norm = entropy / max(math.log(len(probabilities)), 1e-12) if len(probabilities) > 1 else 0.0

        def top_value(name: str) -> float:
            return safe_float(top.get(name))

        def runner_value(name: str) -> float:
            return safe_float(runner.get(name)) if runner is not None else 0.0

        exact_exists = int(any(int(group_rows[index].get("label_exact", 0)) for index in indexes))
        numeric_exists = int(any(int(group_rows[index].get("label_numeric_equiv", 0)) for index in indexes))
        item = {
            "split": top.get("split", ""),
            "provider": top.get("provider", "unknown"),
            "image_id": image_id,
            "prediction": top.get("candidate_digits", ""),
            "prediction_numeric": top.get("numeric_candidate", ""),
            "ground_truth": top.get("gt_channel_number", ""),
            "ground_truth_numeric": top.get("numeric_gt", ""),
            "selected_candidate_id": top.get("candidate_id", ""),
            "selected_text": top.get("candidate_text", ""),
            "selected_source": top.get("source", ""),
            "selected_raw_source": top.get("raw_source", ""),
            "selected_exact": int(top.get("label_exact", 0)),
            "selected_numeric_equiv": int(top.get("label_numeric_equiv", 0)),
            "correct_candidate_exists": exact_exists,
            "numeric_equiv_candidate_exists": numeric_exists,
            "label_train": int(top.get("label_exact", 0)),
            "has_candidate": 1.0,
            "pool_value_group_count": float(len(indexes)),
            "pool_value_group_count_log": math.log1p(len(indexes)),
            "pool_candidate_count": 0.0,
            "pool_score_std": score_std,
            "top1_top2_margin": margin,
            "top1_top2_margin_per_pool_std": margin / max(score_std, 1e-6) if len(scores) > 1 else 0.0,
            "top1_score_z": (scores[0] - score_mean) / max(score_std, 1e-6) if len(scores) > 1 else 0.0,
            "top1_softmax_probability": probabilities[0],
            "pool_softmax_entropy_normalized": entropy_norm,
            "single_value_group": float(len(indexes) == 1),
            "top1_candidate_support": top_value("group_candidate_count"),
            "top1_source_support": top_value("group_source_count"),
            "top1_spatial_cluster_ratio": top_value("group_spatial_cluster_ratio"),
            "top1_spatial_source_support": top_value("group_spatial_cluster_source_count"),
            "top1_cross_source_support": top_value("group_cross_source_support"),
            "top1_original_and_recheck_support": top_value("group_original_and_recheck_support"),
            "top1_near_yolo_ratio": top_value("group_near_yolo_ratio"),
            "top1_other_text_overlap_ratio": top_value("group_other_text_overlap_ratio"),
            "top1_other_number_overlap_ratio": top_value("group_other_number_overlap_ratio"),
            "top1_ocr_conf_max": top_value("group_ocr_conf_max"),
            "top1_value_digit_len": top_value("value_digit_len"),
            "top1_minus_top2_candidate_support": top_value("group_candidate_count") - runner_value("group_candidate_count"),
            "top1_minus_top2_source_support": top_value("group_source_count") - runner_value("group_source_count"),
            "top1_minus_top2_spatial_support": top_value("group_spatial_cluster_source_count")
            - runner_value("group_spatial_cluster_source_count"),
            "top1_spatial_component_count": top_value("group_spatial_component_count"),
            "top1_multi_position_same_text": top_value("group_multi_position_same_text"),
            "top1_channel_evidence_component_count": top_value("group_channel_evidence_component_count"),
            "top1_best_cluster_candidate_support": top_value("group_best_cluster_candidate_count"),
            "top1_best_cluster_source_support": top_value("group_best_cluster_source_count"),
            "top1_best_cluster_near_yolo_ratio": top_value("group_best_cluster_near_yolo_ratio"),
            "top1_best_cluster_yolo_ch0_iou_max": top_value("group_best_cluster_yolo_ch0_iou_max"),
            "top1_best_cluster_yolo_area3_iou_max": top_value("group_best_cluster_yolo_area3_iou_max"),
            "top1_best_cluster_yolo_ch0_conf_max": top_value("group_best_cluster_yolo_ch0_conf_max"),
            "top1_best_cluster_yolo_area3_conf_max": top_value("group_best_cluster_yolo_area3_conf_max"),
            "top1_best_cluster_ocr_conf_max": top_value("group_best_cluster_ocr_conf_max"),
            "top1_best_cluster_other_number_overlap_ratio": top_value(
                "group_best_cluster_other_number_overlap_ratio"
            ),
        }
        item["pool_candidate_count"] = sum(
            safe_float(group_rows[index].get("group_candidate_count")) for index in indexes
        )
        output.append(item)
    return output


def gate_rows_to_selected(rows: Sequence[Dict[str, Any]], scores: Sequence[float]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for row, score in zip(rows, scores):
        item = dict(row)
        item["score"] = float(score) if safe_float(row.get("has_candidate")) else float("-inf")
        selected.append(item)
    return selected


def train_provider_oof_gate_rows(
    train_group_rows: Sequence[Dict[str, Any]],
    train_universe_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    providers = sorted({str(row.get("provider")) for row in train_universe_rows})
    oof_gate_rows: List[Dict[str, Any]] = []
    fold_summary: List[Dict[str, Any]] = []
    for provider in providers:
        fit_rows = [row for row in train_group_rows if str(row.get("provider")) != provider]
        holdout_rows = [row for row in train_group_rows if str(row.get("provider")) == provider]
        holdout_universe = [row for row in train_universe_rows if str(row.get("provider")) == provider]
        model = fit_pointwise_model(
            fit_rows,
            GROUP_FEATURE_NAMES,
            model_type="provider_oof_value_group_pointwise",
            epochs=900,
            lr=0.07,
            l2=0.0015,
        )
        scores = score_group_rows(holdout_rows, model)
        gate_rows = build_gate_rows(holdout_rows, scores, holdout_universe)
        for row in gate_rows:
            row["split"] = "train_oof"
        oof_gate_rows.extend(gate_rows)
        fold_summary.append(
            {
                "provider": provider,
                "fit_images": len({str(row["image_id"]) for row in fit_rows}),
                "holdout_images": len({str(row["image_id"]) for row in holdout_universe}),
                "raw_selected_correct": sum(int(row.get("selected_exact", 0)) for row in gate_rows),
            }
        )
    return oof_gate_rows, fold_summary


def tune_gate_threshold(
    selected: Sequence[Dict[str, Any]],
    *,
    steps: int,
    max_no_ui_fp_rate: float,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    finite_scores = [float(item["score"]) for item in selected if math.isfinite(float(item["score"]))]
    best: Optional[Dict[str, Any]] = None
    sweep: List[Dict[str, Any]] = []
    for threshold in threshold_grid(finite_scores, steps):
        metrics = evaluate_selected(selected, threshold)
        feasible = metrics["no_ui_fp_rate"] <= max_no_ui_fp_rate
        row = {"threshold": threshold, "no_ui_constraint_passed": int(feasible), **metrics}
        sweep.append(row)
        key = (
            int(feasible),
            metrics["final_exact_accuracy"],
            -metrics["wrong_output_count"],
            metrics["output_precision"],
            metrics["coverage"],
            -metrics["no_ui_fp_rate"],
        )
        if best is None or key > best["key"]:
            best = {"key": key, "threshold": threshold, "metrics": metrics}
    if best is None:
        raise RuntimeError("no gate threshold candidates")
    return float(best["threshold"]), dict(best["metrics"]), sweep


def metric_row(dataset: str, model: str, threshold: float, metrics: Mapping[str, Any], raw_correct: int) -> Dict[str, Any]:
    valid_count = int(metrics["valid_gt_count"])
    return {
        "dataset": dataset,
        "model": model,
        "threshold": threshold,
        "raw_selected_correct": raw_correct,
        "raw_selection_accuracy": raw_correct / max(1, valid_count),
        "final_accuracy": metrics["final_exact_accuracy"],
        "coverage": metrics["coverage"],
        "output_precision": metrics["output_precision"],
        "wrong": metrics["wrong_output_count"],
        "no_output": metrics["no_output_count"],
        "oracle": metrics["candidate_oracle_exact_accuracy"],
        "selector_accuracy": metrics["selector_accuracy_given_correct_candidate_exists"],
        "no_ui_count": metrics["no_ui_image_count"],
        "no_ui_fp": metrics["no_ui_fp"],
        "no_ui_fp_rate": metrics["no_ui_fp_rate"],
    }


def write_report(
    path: Path,
    comparison: Sequence[Mapping[str, Any]],
    decision: str,
    next_action: str,
    fold_summary: Sequence[Mapping[str, Any]],
    no_ui_full_metrics: Mapping[str, Any],
) -> None:
    lines = [
        "# Value-Group Relative Confidence Gate",
        "",
        "## Policy",
        "",
        "- Airtel used for group or gate training: false",
        "- Airtel used for threshold tuning: false",
        "- Airtel evaluated only after gate model and threshold freeze: true",
        "- Mainline fusion behavior changed: false",
        "",
        "## Method",
        "",
        "- Build value-group ranker selections with one provider held out at a time.",
        "- Train the gate on OOF selection correctness plus a deterministic 60% no-UI training partition.",
        "- Exclude absolute top score from gate features.",
        "- Use top1/top2 margin, score z-score, softmax probability/entropy, source support, and bbox agreement.",
        "- Tune the gate threshold on No-Airtel val and the remaining 40% no-UI partition.",
        "",
        "## Results",
        "",
        "| dataset | model | threshold | raw selection | final accuracy | coverage | precision | wrong | no_output | no-ui FP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {safe_float(row['threshold']):.6f} | "
            f"{safe_float(row['raw_selection_accuracy']):.6f} | {safe_float(row['final_accuracy']):.6f} | "
            f"{safe_float(row['coverage']):.6f} | {safe_float(row['output_precision']):.6f} | "
            f"{row['wrong']} | {row['no_output']} | {row['no_ui_fp']}/{row['no_ui_count']} |"
        )
    lines.extend(
        [
            "",
            "## OOF Summary",
            "",
            f"- provider folds: {len(fold_summary)}",
            f"- OOF images: {sum(int(row['holdout_images']) for row in fold_summary)}",
            f"- OOF raw correct: {sum(int(row['raw_selected_correct']) for row in fold_summary)}",
            f"- no-UI threshold holdout FP: {comparison[0]['no_ui_fp']}/{comparison[0]['no_ui_count']}",
            f"- no-UI full-set diagnostic FP: {no_ui_full_metrics['no_ui_fp']}/{no_ui_full_metrics['no_ui_image_count']}",
            "- The full no-UI number is diagnostic because its training partition was used by the gate; the holdout number is the gate check.",
            "",
            "## Decision",
            "",
            f"- decision: `{decision}`",
            f"- next action: {next_action}",
            "- Airtel metrics did not influence feature weights or threshold selection.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    base = {
        "split": "val",
        "provider": "KT",
        "image_id": "KT_001",
        "candidate_id": "a",
        "candidate_text": "041",
        "candidate_digits": "041",
        "numeric_candidate": "41",
        "source": "original_ocr",
        "raw_source": "original_ocr",
        "gt_channel_number": "041",
        "numeric_gt": "41",
        "label_exact": 1,
        "label_numeric_equiv": 1,
        "group_candidate_count": 2.0,
        "group_source_count": 2.0,
        "group_spatial_cluster_ratio": 1.0,
        "group_spatial_cluster_source_count": 2.0,
        "group_cross_source_support": 1.0,
        "group_original_and_recheck_support": 1.0,
        "group_near_yolo_ratio": 1.0,
        "group_ocr_conf_max": 0.9,
        "value_digit_len": 3.0,
    }
    runner = {
        **base,
        "candidate_id": "b",
        "candidate_text": "41",
        "candidate_digits": "41",
        "label_exact": 0,
        "group_candidate_count": 1.0,
        "group_source_count": 1.0,
    }
    gate_rows = build_gate_rows([base, runner], [2.0, 1.0], [base])
    assert len(gate_rows) == 1
    assert gate_rows[0]["top1_top2_margin"] == 1.0
    assert gate_rows[0]["top1_softmax_probability"] > 0.5
    assert gate_rows[0]["prediction"] == "041"
    assert "top1_best_cluster_near_yolo_ratio" in gate_rows[0]
    assert "top1_group_score" not in GATE_FEATURE_NAMES
    assert stable_no_ui_partition("same_id") == stable_no_ui_partition("same_id")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a provider-OOF relative confidence gate for the No-Airtel value-group selector."
    )
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--train-candidates", type=Path, default=DEFAULT_TRAIN_CANDIDATES)
    parser.add_argument("--val-candidates", type=Path, default=DEFAULT_VAL_CANDIDATES)
    parser.add_argument("--yolo-root", type=Path, default=DEFAULT_YOLO_ROOT)
    parser.add_argument("--no-ui-candidates", type=Path, default=DEFAULT_NO_UI_CANDIDATES)
    parser.add_argument("--no-ui-images-dir", type=Path, default=DEFAULT_NO_UI_IMAGES)
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--group-model", type=Path, default=DEFAULT_GROUP_MODEL)
    parser.add_argument("--group-config", type=Path, default=DEFAULT_GROUP_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--threshold-steps", type=int, default=300)
    parser.add_argument("--max-no-ui-fp-rate", type=float, default=0.01)
    parser.add_argument(
        "--gate-feature-schema",
        choices=("v1", "multi_position_v2"),
        default="v1",
        help="Keep v1 behavior by default; v2 adds best spatial-cluster evidence features.",
    )
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
        print("train_value_group_relative_gate_no_airtel_v1 self-test passed")
        return

    required = [
        args.features_csv,
        args.train_candidates,
        args.val_candidates,
        args.yolo_root,
        args.no_ui_candidates,
        args.no_ui_images_dir,
        args.baseline_model,
        args.group_model,
        args.group_config,
    ]
    for path in required:
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
    baseline_policy = validate_frozen_baseline_policy(
        args.baseline_model,
        args.baseline_model.with_name("selector_no_airtel_v1_pairwise_linear_threshold.json"),
    )
    group_config = load_json(args.group_config)
    if group_config.get("airtel_used_for_training") is not False or group_config.get(
        "airtel_used_for_validation_or_tuning"
    ) is not False:
        raise RuntimeError("group artifact provenance does not satisfy No-Airtel policy")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gate_feature_names = (
        MULTI_POSITION_GATE_FEATURE_NAMES
        if args.gate_feature_schema == "multi_position_v2"
        else GATE_FEATURE_NAMES
    )
    empty_yolo = args.output_dir / "_empty_yolo_labels"
    empty_yolo.mkdir(parents=True, exist_ok=True)
    raw_rows = load_feature_rows(args.features_csv)
    train_eligible, train_filter = runtime_eligible_candidate_ids(
        args.train_candidates, args.yolo_root / "images/train", args.yolo_root / "labels/train"
    )
    val_eligible, val_filter = runtime_eligible_candidate_ids(
        args.val_candidates, args.yolo_root / "images/val", args.yolo_root / "labels/val"
    )
    no_ui_eligible, no_ui_filter = runtime_eligible_candidate_ids(
        args.no_ui_candidates, args.no_ui_images_dir, empty_yolo
    )
    filtered_rows, feature_filter = filter_runtime_eligible_rows(
        raw_rows,
        {"train": train_eligible, "val": val_eligible, "no_ui_val": no_ui_eligible},
    )
    baseline_model = load_json(args.baseline_model)
    baseline_scores = score_rows(filtered_rows, baseline_model)
    group_rows = group_candidate_rows(filtered_rows, baseline_scores)
    train_groups = [row for row in group_rows if row.get("split") == "train"]
    train_universe = [row for row in raw_rows if row.get("split") == "train"]
    oof_rows, fold_summary = train_provider_oof_gate_rows(train_groups, train_universe)

    group_model = load_json(args.group_model)
    no_ui_groups = [row for row in group_rows if row.get("split") == "no_ui_val"]
    no_ui_universe = [row for row in raw_rows if row.get("split") == "no_ui_val"]
    no_ui_scores = score_group_rows(no_ui_groups, group_model)
    no_ui_gate_rows = build_gate_rows(no_ui_groups, no_ui_scores, no_ui_universe)
    for row in no_ui_gate_rows:
        row["split"] = stable_no_ui_partition(str(row["image_id"]))
        row["label_train"] = 0
        row["selected_exact"] = 0
        row["selected_numeric_equiv"] = 0

    gate_train_rows = oof_rows + [row for row in no_ui_gate_rows if row["split"] == "no_ui_gate_train"]
    gate_model = fit_pointwise_model(
        gate_train_rows,
        gate_feature_names,
        model_type=(
            "provider_oof_relative_confidence_gate_multi_position_v2"
            if args.gate_feature_schema == "multi_position_v2"
            else "provider_oof_relative_confidence_gate"
        ),
        epochs=2400,
        lr=0.06,
        l2=0.002,
    )
    write_json(args.output_dir / "relative_confidence_gate.json", gate_model)

    val_groups = [row for row in group_rows if row.get("split") == "val"]
    val_universe = [row for row in raw_rows if row.get("split") == "val"]
    val_group_scores = score_group_rows(val_groups, group_model)
    val_gate_rows = build_gate_rows(val_groups, val_group_scores, val_universe)
    tune_gate_rows = val_gate_rows + [row for row in no_ui_gate_rows if row["split"] == "no_ui_val"]
    tune_gate_scores = score_rows(tune_gate_rows, gate_model)
    tune_selected = gate_rows_to_selected(tune_gate_rows, tune_gate_scores)
    threshold, val_metrics, sweep = tune_gate_threshold(
        tune_selected,
        steps=args.threshold_steps,
        max_no_ui_fp_rate=args.max_no_ui_fp_rate,
    )
    write_json(
        args.output_dir / "relative_confidence_gate_threshold.json",
        {
            "recommended_threshold": threshold,
            "recommended_threshold_metrics": val_metrics,
            "threshold_tuning_uses_airtel": False,
            "threshold_tuning_split": "no_airtel_val_plus_deterministic_no_ui_holdout",
            "max_no_ui_fp_rate": args.max_no_ui_fp_rate,
        },
    )
    write_csv(args.output_dir / "threshold_sweep.csv", sweep)
    write_csv(args.output_dir / "oof_gate_training_rows.csv", gate_train_rows)
    write_csv(args.output_dir / "provider_oof_summary.csv", fold_summary)
    write_predictions(args.output_dir / "no_airtel_val_predictions.csv", tune_selected, threshold)

    no_ui_full_rows = [dict(row, split="no_ui_val") for row in no_ui_gate_rows]
    no_ui_full_scores = score_rows(no_ui_full_rows, gate_model)
    no_ui_full_selected = gate_rows_to_selected(no_ui_full_rows, no_ui_full_scores)
    no_ui_full_metrics = evaluate_selected(no_ui_full_selected, threshold)
    write_predictions(args.output_dir / "no_ui_full_predictions.csv", no_ui_full_selected, threshold)

    val_raw_correct = sum(int(row.get("selected_exact", 0)) for row in val_gate_rows)
    comparison = [metric_row("no_airtel_val", "value_group_relative_gate", threshold, val_metrics, val_raw_correct)]
    airtel_doc: Optional[Dict[str, Any]] = None
    decision = "no_airtel_validation_only"
    next_action = "Run the frozen gate on a new external provider before any mainline integration."

    if not args.skip_airtel_diagnostic:
        for path in (args.airtel_candidates, args.airtel_images_dir, args.airtel_yolo_label_dir, args.airtel_gt_json):
            if not path.exists():
                raise SystemExit(f"Airtel diagnostic input not found: {path}")
        airtel_rows_raw, airtel_input = build_rows_for_split(
            split="airtel_diagnostic",
            candidates_json=args.airtel_candidates,
            images_dir=args.airtel_images_dir,
            yolo_label_dir=args.airtel_yolo_label_dir,
            gt_json=args.airtel_gt_json,
        )
        airtel_eligible, airtel_filter = runtime_eligible_candidate_ids(
            args.airtel_candidates, args.airtel_images_dir, args.airtel_yolo_label_dir
        )
        airtel_rows, airtel_feature_filter = filter_runtime_eligible_rows(
            airtel_rows_raw, {"airtel_diagnostic": airtel_eligible}
        )
        airtel_baseline_scores = score_rows(airtel_rows, baseline_model)
        airtel_groups = group_candidate_rows(airtel_rows, airtel_baseline_scores)
        airtel_group_scores = score_group_rows(airtel_groups, group_model)
        airtel_gate_rows = build_gate_rows(airtel_groups, airtel_group_scores, airtel_rows_raw)
        airtel_gate_scores = score_rows(airtel_gate_rows, gate_model)
        airtel_selected = gate_rows_to_selected(airtel_gate_rows, airtel_gate_scores)
        airtel_metrics = evaluate_selected(airtel_selected, threshold)
        airtel_raw_correct = sum(int(row.get("selected_exact", 0)) for row in airtel_gate_rows)
        comparison.append(
            metric_row("airtel_diagnostic", "value_group_relative_gate", threshold, airtel_metrics, airtel_raw_correct)
        )
        write_predictions(args.output_dir / "airtel_diagnostic_predictions.csv", airtel_selected, threshold)
        published = load_json(args.published_airtel_summary) if args.published_airtel_summary.exists() else {}
        published_accuracy = safe_float(published.get("final_prediction_exact_accuracy"))
        published_wrong = int(safe_float(published.get("final_wrong_output_count")))
        no_ui_ok = val_metrics["no_ui_fp_rate"] <= args.max_no_ui_fp_rate
        if (
            airtel_metrics["final_exact_accuracy"] > published_accuracy
            and airtel_metrics["wrong_output_count"] <= published_wrong
            and no_ui_ok
        ):
            decision = "candidate_for_opt_in_pipeline_ablation"
            next_action = "Integrate ranking plus relative gate behind an opt-in fusion flag and evaluate temporal interaction."
        elif airtel_metrics["final_exact_accuracy"] > published_accuracy and no_ui_ok:
            decision = "accuracy_gain_wrong_output_tradeoff"
            next_action = "Keep research-only and improve the gate precision on provider-held-out No-Airtel data."
        else:
            decision = "not_promoted"
            next_action = "Keep the current baseline and move to provider-balanced hard-negative selector training."
        airtel_doc = {
            "input": airtel_input,
            "runtime_filter": airtel_filter,
            "feature_filter": airtel_feature_filter,
            "metrics": airtel_metrics,
            "raw_selection_score_distribution": selection_score_distribution(airtel_selected),
            "published_baseline_accuracy": published_accuracy,
            "published_baseline_wrong": published_wrong,
        }

    config = {
        "name": (
            "value_group_relative_gate_no_airtel_multi_position_v2"
            if args.gate_feature_schema == "multi_position_v2"
            else "value_group_relative_gate_no_airtel_v1"
        ),
        "gate_feature_schema": args.gate_feature_schema,
        "enabled_by_default": False,
        "group_model": str(args.group_model).replace("\\", "/"),
        "gate_model": str((args.output_dir / "relative_confidence_gate.json")).replace("\\", "/"),
        "gate_threshold": threshold,
        "absolute_group_score_used_as_gate_feature": False,
        "airtel_used_for_training": False,
        "airtel_used_for_validation_or_threshold_tuning": False,
        "decision": decision,
    }
    write_json(args.output_dir / "relative_gate_config.json", config)
    write_json(
        args.output_dir / "experiment_summary.json",
        {
            "config": config,
            "baseline_policy": baseline_policy,
            "comparison": comparison,
            "validation_metrics": val_metrics,
            "no_ui_full_diagnostic_metrics": no_ui_full_metrics,
            "provider_oof_summary": fold_summary,
            "filter_summary": {
                "train": train_filter,
                "val": val_filter,
                "no_ui": no_ui_filter,
                "feature_rows": feature_filter,
            },
            "airtel_diagnostic": airtel_doc,
        },
    )
    write_csv(args.output_dir / "comparison_metrics.csv", comparison)
    write_json(
        args.output_dir / "feature_schema.json",
        {
            "feature_names": gate_feature_names,
            "feature_count": len(gate_feature_names),
            "absolute_score_excluded": True,
            "forbidden_features": ["image_id", "image_path", "provider", "gt_channel_number", "channel_number_text"],
        },
    )
    write_report(
        args.output_dir / "relative_gate_report.md",
        comparison,
        decision,
        next_action,
        fold_summary,
        no_ui_full_metrics,
    )
    print(json.dumps({"decision": decision, "threshold": threshold, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
