"""Tiny linear candidate ranker inference.

This module uses only model-generated candidates and YOLO boxes. It does not
read annotation or ground-truth files.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from channel_number_fusion import (
    center_distance_norm,
    digits,
    has_alpha,
    iou,
    is_plain_numeric,
    overlap_min_fraction,
)


BBox = Tuple[float, float, float, float]


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(out) or math.isinf(out):
        return 0.0
    return out


def center(box: BBox) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def contains(box: BBox, point: Tuple[float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def source_name(candidate: Dict[str, Any]) -> str:
    source = str(candidate.get("source", "")).strip()
    return source or "ocr"


def nested_value(candidate: Dict[str, Any], keys: Iterable[str]) -> Any:
    """Read a field from candidate or shallow/nested metadata blocks."""
    wanted = {str(key) for key in keys}
    for key in wanted:
        if key in candidate:
            return candidate.get(key)
    stack: List[Any] = [candidate.get("metadata")]
    depth = 0
    while stack and depth < 8:
        depth += 1
        item = stack.pop()
        if not isinstance(item, dict):
            continue
        for key in wanted:
            if key in item:
                return item.get(key)
        for value in item.values():
            if isinstance(value, dict):
                stack.append(value)
    return None


def normalize_source_alias(source: Any) -> str:
    text = str(source or "").strip().lower()
    if not text:
        return "unknown"
    compact = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if "slot_proposal_v2" in compact or "v2_cc_textline" in compact:
        return "slot_proposal_v2_cc_textline"
    if "legacy_slot_repair" in compact or ("slot" in compact and "repair" in compact):
        return "legacy_slot_repair"
    if "history_roi_prior" in compact:
        return "slot_proposal_recheck"
    if "history_guided" in compact and "slot" in compact:
        return "slot_proposal_recheck"
    if "slot" in compact and "proposal" in compact:
        return "slot_proposal_recheck"
    if "crop" in compact and "variant" in compact:
        return "crop_variant_numeric_recheck"
    if "paddleocr" in compact and "channel" in compact:
        return "paddleocr_channel_recheck"
    if "refined" in compact and "numeric" in compact:
        return "refined_numeric_substring"
    if "digit_sequence_trim" in compact:
        return "digit_sequence_trim"
    if compact in {"ocr", "original_ocr", "easyocr", "paddleocr"}:
        return "original_ocr"
    if "yolo_channel_number_area" in compact:
        return "yolo_channel_number_area"
    if "yolo_channel_number" in compact:
        return "yolo_channel_number"
    return compact if compact in {
        "original_ocr",
        "refined_numeric_substring",
        "paddleocr_channel_recheck",
        "digit_sequence_trim",
        "crop_variant_numeric_recheck",
        "slot_proposal_recheck",
        "yolo_channel_number",
        "yolo_channel_number_area",
    } else "unknown"


def alias_flags(prefix: str, value: str, options: Sequence[str]) -> Dict[str, float]:
    return {f"{prefix}_{option}": float(value == option) for option in options}


def source_flags(source: str) -> Dict[str, float]:
    return {
        "source_ocr": float(source == "ocr"),
        "source_refined_numeric_substring": float(source == "refined_numeric_substring"),
        "source_paddleocr_channel_recheck": float(source == "paddleocr_channel_recheck"),
        "source_temporal_slot_recheck": float(source == "temporal_slot_recheck"),
        "source_easyocr_numeric_recheck": float(source == "easyocr_numeric_recheck"),
        "source_paddleocr_numeric_recheck": float(source == "paddleocr_numeric_recheck"),
        "source_digit_sequence_trim": float(source == "digit_sequence_trim"),
        "source_paddleocr_crop_variant_recheck": float(source == "paddleocr_crop_variant_recheck"),
        "source_paddleocr_slot_proposal_recheck": float(source == "paddleocr_slot_proposal_recheck"),
        "source_crop_variant_numeric_recheck": float(source == "crop_variant_numeric_recheck"),
        "source_slot_proposal_recheck": float(source in {"slot_proposal_recheck", "paddleocr_slot_proposal_recheck"}),
        "source_legacy_slot_repair": float(source == "legacy_slot_repair"),
        "source_slot_proposal_v2_cc_textline": float(source == "slot_proposal_v2_cc_textline"),
    }


def text_features(text: str, candidate_digits: str) -> Dict[str, float]:
    stripped = text.strip()
    alpha_count = sum(ch.isalpha() for ch in stripped)
    digit_count = sum(ch.isdigit() for ch in stripped)
    text_len = len(stripped)
    return {
        "text_len": float(text_len),
        "digit_count": float(digit_count),
        "alpha_count": float(alpha_count),
        "digit_ratio": digit_count / max(1, text_len),
        "alpha_ratio": alpha_count / max(1, text_len),
        "candidate_digit_len": float(len(candidate_digits)),
        "is_plain_numeric": float(is_plain_numeric(stripped)),
        "has_alpha": float(has_alpha(stripped)),
        "has_colon": float(":" in stripped),
        "has_pm_am": float(bool(re.search(r"\b[ap]m\b", stripped, flags=re.I))),
        "has_sign_or_slash": float(any(ch in stripped for ch in "-/.,[]()")),
        "starts_with_digit": float(bool(stripped) and stripped[0].isdigit()),
        "ends_with_digit": float(bool(stripped) and stripped[-1].isdigit()),
    }


def yolo_feature_block(
    prefix: str,
    candidate_box: BBox,
    boxes: Sequence[Dict[str, Any]],
    width: int,
    height: int,
) -> Dict[str, float]:
    if not boxes:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_max_iou": 0.0,
            f"{prefix}_max_overlap_min": 0.0,
            f"{prefix}_min_center_dist": 9.0,
            f"{prefix}_max_conf": 0.0,
            f"{prefix}_contains_candidate_center": 0.0,
        }
    candidate_center = center(candidate_box)
    return {
        f"{prefix}_count": float(len(boxes)),
        f"{prefix}_max_iou": max(iou(candidate_box, tuple(float(v) for v in box["bbox_xyxy"])) for box in boxes),
        f"{prefix}_max_overlap_min": max(
            overlap_min_fraction(candidate_box, tuple(float(v) for v in box["bbox_xyxy"])) for box in boxes
        ),
        f"{prefix}_min_center_dist": min(
            center_distance_norm(candidate_box, tuple(float(v) for v in box["bbox_xyxy"]), width, height) for box in boxes
        ),
        f"{prefix}_max_conf": max(safe_float(box.get("conf", 0.0)) for box in boxes),
        f"{prefix}_contains_candidate_center": float(
            any(contains(tuple(float(v) for v in box["bbox_xyxy"]), candidate_center) for box in boxes)
        ),
    }


def candidate_pre_score(candidate: Dict[str, Any]) -> float:
    for key in (
        "ranker_score",
        "fusion_score",
        "selector_score",
        "final_score",
        "score",
        "rule_score",
        "ocr_conf",
        "confidence",
        "recognizer_conf",
        "proposal_score",
    ):
        value = nested_value(candidate, (key,))
        if value is not None:
            return safe_float(value)
    return 0.0


def annotate_candidate_pool_features(
    candidates: Sequence[Dict[str, Any]],
    yolo: Sequence[Dict[str, Any]] | None = None,
    width: int = 0,
    height: int = 0,
) -> None:
    """Mutate candidates with pool-level features used by selector v2.

    These features are derived only from model-generated candidates and YOLO
    boxes. GT is intentionally not used here, so the same annotation can be
    applied at training-feature build time and at inference time.
    """
    usable: List[Dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    text_counts: Counter[str] = Counter()
    text_sources: Dict[str, set[str]] = defaultdict(set)
    text_best_scores: Dict[str, float] = defaultdict(float)
    source_seen: Counter[str] = Counter()

    for candidate in candidates:
        if not isinstance(candidate, dict) or "bbox_xyxy" not in candidate:
            continue
        value = digits(candidate.get("text", ""))
        if not value:
            continue
        source = normalize_source_alias(candidate.get("source") or nested_value(candidate, ("raw_source",)))
        usable.append(candidate)
        source_counts[source] += 1
        text_counts[value] += 1
        text_sources[value].add(source)
        text_best_scores[value] = max(text_best_scores[value], candidate_pre_score(candidate))

    for candidate in usable:
        value = digits(candidate.get("text", ""))
        source = normalize_source_alias(candidate.get("source") or nested_value(candidate, ("raw_source",)))
        source_seen[source] += 1
        candidate["candidate_rank_within_source"] = source_seen[source]
        candidate["source_candidate_count"] = source_counts[source]
        candidate["same_text_support_count"] = text_counts[value]
        candidate["duplicate_text_count"] = max(0, text_counts[value] - 1)
        candidate["same_text_source_count"] = len(text_sources[value])
        candidate["same_text_best_score"] = text_best_scores[value]


SOURCE_DETAIL_OPTIONS = (
    "original_ocr",
    "refined_numeric_substring",
    "paddleocr_channel_recheck",
    "digit_sequence_trim",
    "crop_variant_numeric_recheck",
    "slot_proposal_recheck",
    "legacy_slot_repair",
    "slot_proposal_v2_cc_textline",
    "yolo_channel_number",
    "yolo_channel_number_area",
    "unknown",
)


BROAD_REGION_OPTIONS = (
    "yolo_channel",
    "yolo_channel_area",
    "ocr_neighbor",
    "fallback",
    "fallback_lower_left_mid",
    "full_image",
    "unknown",
)


def normalize_broad_region(value: Any) -> str:
    text = str(value or "").strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not compact:
        return "unknown"
    if "fallback_lower_left_mid" in compact:
        return "fallback_lower_left_mid"
    if "fallback" in compact:
        return "fallback"
    if "channel_number_area" in compact or "yolo_area" in compact:
        return "yolo_channel_area"
    if "yolo" in compact and "channel" in compact:
        return "yolo_channel"
    if "ocr" in compact:
        return "ocr_neighbor"
    if "full" in compact:
        return "full_image"
    return compact if compact in BROAD_REGION_OPTIONS else "unknown"


def candidate_features(
    candidate: Dict[str, Any],
    yolo: Sequence[Dict[str, Any]],
    width: int,
    height: int,
) -> Dict[str, float]:
    text = str(candidate.get("text", ""))
    candidate_digits = digits(text)
    candidate_box = tuple(float(v) for v in candidate["bbox_xyxy"])  # type: ignore[assignment]
    x1, y1, x2, y2 = candidate_box
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    box_area = box_width * box_height
    cx, cy = center(candidate_box)
    source = source_name(candidate)
    normalized_source = normalize_source_alias(source)
    raw_source = normalize_source_alias(nested_value(candidate, ("raw_source",)) or source)
    broad_region = normalize_broad_region(nested_value(candidate, ("broad_region_source", "region_source", "proposal_region_source")))
    yolo_by_class = {cls: [box for box in yolo if int(box.get("class_id", -1)) == cls] for cls in (0, 1, 2, 3)}

    features: Dict[str, float] = {
        "ocr_conf": safe_float(candidate.get("ocr_conf", candidate.get("confidence", 0.0))),
        "detection_conf": safe_float(candidate.get("detection_conf", 0.0)),
        "candidate_input_score": candidate_pre_score(candidate),
        "proposal_score": safe_float(nested_value(candidate, ("proposal_score",))),
        "recognizer_conf": safe_float(nested_value(candidate, ("recognizer_conf", "ocr_conf", "confidence"))),
        "component_count": safe_float(nested_value(candidate, ("component_count",))),
        "crop_quality_score": safe_float(nested_value(candidate, ("crop_quality_score",))),
        "edge_density": safe_float(nested_value(candidate, ("edge_density",))),
        "foreground_ratio": safe_float(nested_value(candidate, ("foreground_ratio",))),
        "temporal_history_score": safe_float(candidate.get("history_score", 0.0)),
        "temporal_slot_iou": safe_float(nested_value(candidate, ("temporal_slot_iou", "history_iou"))),
        "temporal_slot_distance": safe_float(nested_value(candidate, ("temporal_slot_distance", "history_distance"))),
        "temporal_slot_boost": safe_float(nested_value(candidate, ("temporal_slot_boost",))),
        "history_overlap_min_fraction": safe_float(nested_value(candidate, ("history_overlap_min_fraction",))),
        "history_track_score": safe_float(nested_value(candidate, ("history_track_score",))),
        "history_track_seen_count": safe_float(nested_value(candidate, ("history_track_seen_count",))),
        "history_track_missed_count": safe_float(nested_value(candidate, ("history_track_missed_count",))),
        "history_track_stability": safe_float(nested_value(candidate, ("history_track_stability",))),
        "history_is_lockable": safe_float(nested_value(candidate, ("history_is_lockable",))),
        "from_temporal_locked_slot": safe_float(
            nested_value(candidate, ("from_temporal_locked_slot", "from_history_guided_slot", "from_history_roi_prior"))
        ),
        "from_history_roi_prior": safe_float(nested_value(candidate, ("from_history_roi_prior",))),
        "history_prior_score": safe_float(nested_value(candidate, ("history_prior_score", "history_score"))),
        "history_prior_iou_with_candidate": safe_float(nested_value(candidate, ("history_prior_iou_with_candidate", "history_iou"))),
        "history_prior_distance_to_candidate": safe_float(
            nested_value(candidate, ("history_prior_distance_to_candidate", "history_distance"))
        ),
        "history_prior_age": safe_float(nested_value(candidate, ("history_prior_age", "locked_slot_age"))),
        "yolo_history_agreement": safe_float(nested_value(candidate, ("yolo_history_agreement",))),
        "yolo_history_conflict": safe_float(nested_value(candidate, ("yolo_history_conflict",))),
        "roi_evidence_source_count": safe_float(nested_value(candidate, ("roi_evidence_source_count",))),
        "locked_slot_age": safe_float(nested_value(candidate, ("locked_slot_age",))),
        "candidate_rank_within_source": safe_float(candidate.get("candidate_rank_within_source", 0.0)),
        "same_text_support_count": safe_float(candidate.get("same_text_support_count", 1.0)),
        "duplicate_text_count": safe_float(candidate.get("duplicate_text_count", 0.0)),
        "same_text_best_score": safe_float(candidate.get("same_text_best_score", 0.0)),
        "same_text_source_count": safe_float(candidate.get("same_text_source_count", 1.0)),
        "source_candidate_count": safe_float(candidate.get("source_candidate_count", 1.0)),
        "bbox_x1_norm": x1 / max(1, width),
        "bbox_y1_norm": y1 / max(1, height),
        "bbox_x2_norm": x2 / max(1, width),
        "bbox_y2_norm": y2 / max(1, height),
        "bbox_cx_norm": cx / max(1, width),
        "bbox_cy_norm": cy / max(1, height),
        "bbox_w_norm": box_width / max(1, width),
        "bbox_h_norm": box_height / max(1, height),
        "bbox_area_norm": box_area / max(1, width * height),
        "bbox_area_ratio": box_area / max(1, width * height),
        "bbox_aspect": box_width / max(1.0, box_height),
    }
    features.update(text_features(text, candidate_digits))
    features.update(source_flags(source))
    features.update(alias_flags("source_detail", normalized_source, SOURCE_DETAIL_OPTIONS))
    features.update(alias_flags("raw_source_detail", raw_source, SOURCE_DETAIL_OPTIONS))
    features.update(alias_flags("broad_region", broad_region, BROAD_REGION_OPTIONS))
    for cls, prefix in ((0, "yolo_ch0"), (3, "yolo_area3"), (1, "yolo_other_num1"), (2, "yolo_other_text2")):
        features.update(yolo_feature_block(prefix, candidate_box, yolo_by_class[cls], width, height))
    features["near_yolo_channel_evidence"] = float(
        features["yolo_ch0_max_iou"] > 0.03
        or features["yolo_ch0_max_overlap_min"] > 0.20
        or features["yolo_ch0_min_center_dist"] < 0.08
        or features["yolo_area3_max_iou"] > 0.03
        or features["yolo_area3_max_overlap_min"] > 0.20
        or features["yolo_area3_contains_candidate_center"] > 0.0
    )
    features["overlaps_other_number"] = float(
        features["yolo_other_num1_max_iou"] > 0.03 or features["yolo_other_num1_max_overlap_min"] > 0.20
    )
    features["overlaps_other_text"] = float(
        features["yolo_other_text2_max_iou"] > 0.03 or features["yolo_other_text2_max_overlap_min"] > 0.20
    )
    return features


class CandidateRanker:
    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        model = json.loads(self.model_path.read_text(encoding="utf-8"))
        self.model_type = str(model.get("model_type", "linear_ranker"))
        self.feature_names = [str(name) for name in model["feature_names"]]
        self.mean = [float(v) for v in model["mean"]]
        self.std = [float(v) if abs(float(v)) > 1e-8 else 1.0 for v in model["std"]]
        self.weights = [float(v) for v in model["weights"]]
        self.bias = float(model.get("bias", 0.0))
        if not (len(self.feature_names) == len(self.mean) == len(self.std) == len(self.weights)):
            raise ValueError(f"invalid ranker model dimensions: {self.model_path}")

    def score_features(self, features: Dict[str, float]) -> float:
        total = self.bias
        for name, mean, std, weight in zip(self.feature_names, self.mean, self.std, self.weights):
            total += ((safe_float(features.get(name, 0.0)) - mean) / std) * weight
        return float(total)

    def score_candidate(
        self,
        candidate: Dict[str, Any],
        yolo: Sequence[Dict[str, Any]],
        width: int,
        height: int,
    ) -> float:
        return self.score_features(candidate_features(candidate, yolo, width, height))
