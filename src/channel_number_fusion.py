"""Fuse OCR candidates with YOLO channel-number boxes."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


BBox = Tuple[float, float, float, float]
DISTRACTOR_WORD_RE = re.compile(
    r"\b(news|sports?|season|episode|crime|india|bangla|telugu|tamil|sony|star|zee|cnn|cnbc|tv)\b",
    flags=re.I,
)
NUMERIC_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[0-9]{1,5}(?![A-Za-z0-9])")


def digits(text: str) -> str:
    return "".join(ch for ch in str(text) if "0" <= ch <= "9")


def normalized_digits(text: str) -> str:
    raw = digits(text)
    if not raw:
        return ""
    return raw


def numeric_equivalent_digits(text: str) -> str:
    raw = digits(text)
    if not raw:
        return ""
    return str(int(raw))


def has_alpha(text: str) -> bool:
    return any(ch.isalpha() for ch in str(text))


def is_alpha_only_text(text: str) -> bool:
    stripped = str(text).strip()
    return bool(stripped) and has_alpha(stripped) and not digits(stripped)


def is_numeric_recheck_source(source: Any) -> bool:
    return source in (
        "easyocr_numeric_recheck",
        "paddleocr_numeric_recheck",
        "paddleocr_channel_recheck",
        "paddleocr_crop_variant_recheck",
        "paddleocr_slot_proposal_recheck",
        "slot_proposal_recheck",
        "history_guided_slot_recheck",
        "history_roi_prior_recheck",
        "legacy_slot_repair",
        "slot_proposal_v2_cc_textline",
        "temporal_slot_recheck",
    )


def is_plain_numeric(text: str) -> bool:
    return bool(re.fullmatch(r"\s*[0-9]{1,5}\s*", str(text)))


def has_independent_numeric_token(text: str, value: str) -> bool:
    return any(digits(match.group(0)) == value for match in NUMERIC_TOKEN_RE.finditer(str(text)))


def refined_numeric_noise_context(candidate: Dict[str, Any], raw_digits: str, text: str) -> bool:
    """True when a refined numeric token looks like OCR noise around a number.

    Example: parent OCR reads "005poR", while the refined candidate is "005".
    This is different from extracting an arbitrary number from a long title.
    """
    if candidate.get("source") != "refined_numeric_substring":
        return False
    if not is_plain_numeric(text) or len(raw_digits) not in (3, 4, 5):
        return False
    parent_text = str(candidate.get("parent_text", "")).strip()
    if not parent_text or digits(parent_text) != raw_digits:
        return False
    compact_parent = re.sub(r"\s+", "", parent_text)
    compact_digits = re.sub(r"\s+", "", raw_digits)
    if not (compact_parent.startswith(compact_digits) or compact_parent.endswith(compact_digits)):
        return False
    noise = re.sub(r"[0-9\s:.\-/]", "", parent_text)
    return 0 < len(noise) <= 4


def trusted_numeric_candidate(candidate: Dict[str, Any], raw_digits: str, text: str) -> bool:
    if len(raw_digits) not in (2, 3, 4, 5):
        return False
    if is_plain_numeric(text):
        return True
    return refined_numeric_noise_context(candidate, raw_digits, text)


def mixed_alpha_without_numeric_token(candidate: Dict[str, Any], raw_digits: str, text: str) -> bool:
    if not has_alpha(text):
        return False
    if is_plain_numeric(text):
        return False
    return not has_independent_numeric_token(text, raw_digits)


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - inter
    return 0.0 if area <= 0 else inter / area


def bbox_area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def overlap_min_fraction(a: BBox, b: BBox) -> float:
    smaller = min(bbox_area(a), bbox_area(b))
    if smaller <= 0:
        return 0.0
    return intersection_area(a, b) / smaller


def center_distance_norm(a: BBox, b: BBox, width: int, height: int) -> float:
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return (((acx - bcx) / width) ** 2 + ((acy - bcy) / height) ** 2) ** 0.5


def yolo_boxes(label_dir: Path, image_id: str, width: int, height: int) -> List[Dict[str, Any]]:
    path = label_dir / f"{image_id}.txt"
    boxes = []
    if not path.exists():
        return boxes
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls not in (0, 1, 2, 3):
            continue
        cx, cy, bw, bh = [float(v) for v in parts[1:5]]
        conf = float(parts[5]) if len(parts) > 5 else 1.0
        bbox = ((cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height)
        class_names = {
            0: "channel_number",
            1: "other_number",
            2: "other_text",
            3: "channel_number_area",
        }
        boxes.append(
            {
                "id": f"yolo_{idx:04d}",
                "class_id": cls,
                "class_name": class_names[cls],
                "bbox_xyxy": [round(v, 3) for v in bbox],
                "conf": conf,
            }
        )
    return boxes


def resolve_image_path(path: Path, images_dir: Path, image_id: str) -> Path:
    text = str(path).replace("\\", "/")
    path = Path(text)
    for candidate in (path, Path.cwd() / path, images_dir / f"{image_id}.jpg", images_dir / f"{image_id}.png"):
        if candidate.exists():
            return candidate
    return path


def position_score(bbox: BBox, width: int, height: int) -> float:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    score = 0.0
    if cx < width * 0.18:
        score += 0.35
    if cy > height * 0.62:
        score += 0.30
    if x1 < width * 0.08 and cy > height * 0.70:
        score += 0.25
    return min(score, 0.8)


def text_quality_adjustment(candidate: Dict[str, Any], raw_digits: str, text: str) -> float:
    source = candidate.get("source")
    parent_text = str(candidate.get("parent_text", ""))
    context_text = parent_text if source == "refined_numeric_substring" and parent_text else text
    adjustment = 0.0

    if is_plain_numeric(text):
        adjustment += 0.12
    if DISTRACTOR_WORD_RE.search(context_text):
        adjustment -= 0.55

    alpha_context = has_alpha(text) or (source == "refined_numeric_substring" and has_alpha(parent_text))
    if alpha_context:
        if refined_numeric_noise_context(candidate, raw_digits, text):
            adjustment -= 0.20
        else:
            adjustment -= 0.65
            if not has_independent_numeric_token(context_text, raw_digits):
                adjustment -= 0.25
            elif source == "refined_numeric_substring":
                adjustment -= 0.10

    return adjustment


def original_ocr_candidates(candidates: Sequence[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for candidate in candidates:
        if "bbox_xyxy" not in candidate:
            continue
        if candidate.get("source") in (
            "easyocr_numeric_recheck",
            "paddleocr_numeric_recheck",
            "paddleocr_channel_recheck",
            "paddleocr_crop_variant_recheck",
            "paddleocr_slot_proposal_recheck",
            "legacy_slot_repair",
            "slot_proposal_v2_cc_textline",
            "refined_numeric_substring",
            "digit_sequence_trim",
            "temporal_slot_recheck",
        ):
            continue
        yield candidate


def same_location_ocr_adjustment(
    candidate: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    width: int,
    height: int,
) -> float:
    if "bbox_xyxy" not in candidate:
        return 0.0
    bbox = tuple(float(v) for v in candidate["bbox_xyxy"])
    value = normalized_digits(candidate.get("text", ""))
    if not value:
        return 0.0

    adjustment = 0.0
    has_numeric_support = False
    alpha_conflict = False
    for other in original_ocr_candidates(candidates):
        if other is candidate:
            continue
        if (
            candidate.get("source") == "refined_numeric_substring"
            and candidate.get("parent_id") is not None
            and str(other.get("id")) == str(candidate.get("parent_id"))
            and refined_numeric_noise_context(candidate, digits(str(candidate.get("text", ""))), str(candidate.get("text", "")))
        ):
            continue
        obox = tuple(float(v) for v in other["bbox_xyxy"])
        close = overlap_min_fraction(bbox, obox) > 0.45 or center_distance_norm(bbox, obox, width, height) < 0.045
        if not close:
            continue
        other_text = str(other.get("text", ""))
        other_value = normalized_digits(other_text)
        if is_plain_numeric(other_text) and other_value == value:
            has_numeric_support = True
        elif has_alpha(other_text):
            alpha_conflict = True

    if has_numeric_support:
        adjustment += 0.22

    if alpha_conflict:
        if candidate.get("source") == "temporal_slot_recheck":
            adjustment -= 0.18
        else:
            adjustment -= 0.95
        if is_numeric_recheck_source(candidate.get("source")) and not has_numeric_support:
            adjustment -= 0.25

    if is_numeric_recheck_source(candidate.get("source")) and not has_numeric_support:
        adjustment -= 0.12

    return adjustment


def same_location_alpha_only_context(
    candidate: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    width: int,
    height: int,
) -> bool:
    if not is_numeric_recheck_source(candidate.get("source")) or "bbox_xyxy" not in candidate:
        return False
    bbox = tuple(float(v) for v in candidate["bbox_xyxy"])
    for other in original_ocr_candidates(candidates):
        obox = tuple(float(v) for v in other["bbox_xyxy"])
        close = overlap_min_fraction(bbox, obox) > 0.45 or center_distance_norm(bbox, obox, width, height) < 0.045
        if close and is_alpha_only_text(other.get("text", "")):
            return True
    return False


def channel_recheck_text_support(
    candidate: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    yolo: Sequence[Dict[str, Any]],
    width: int,
    height: int,
    raw_digits: str,
    allow_single_digit: bool = False,
) -> bool:
    if candidate.get("source") != "paddleocr_channel_recheck" or "bbox_xyxy" not in candidate:
        return True
    bbox = tuple(float(v) for v in candidate["bbox_xyxy"])
    if len(raw_digits) < 2 and not allow_single_digit:
        return False
    for other in original_ocr_candidates(candidates):
        obox = tuple(float(v) for v in other["bbox_xyxy"])
        close = overlap_min_fraction(bbox, obox) > 0.25 or center_distance_norm(bbox, obox, width, height) < 0.08
        if not close:
            continue
        other_digits = digits(other.get("text", ""))
        if other_digits == raw_digits:
            return True
        extra = len(other_digits) - len(raw_digits)
        if len(raw_digits) >= 2 and 0 < extra <= 2 and (
            other_digits.startswith(raw_digits) or other_digits.endswith(raw_digits)
        ):
            return True
    bbox = tuple(float(v) for v in candidate["bbox_xyxy"])
    for box in yolo:
        if box.get("class_id") not in (0, 3):
            continue
        ybox = tuple(float(v) for v in box["bbox_xyxy"])
        close = (
            iou(bbox, ybox) > 0.12
            or overlap_min_fraction(bbox, ybox) > 0.35
            or center_distance_norm(bbox, ybox, width, height) < 0.08
        )
        if close and float(candidate.get("ocr_conf", candidate.get("confidence", 0.0))) >= 0.55:
            return True
    return False


def elongated_numeric_textline_penalty(candidate: Dict[str, Any], raw_digits: str, text: str, bbox: BBox) -> float:
    if candidate.get("source") or not is_plain_numeric(text) or len(raw_digits) not in (1, 2, 3, 4, 5):
        return 0.0
    x1, y1, x2, y2 = bbox
    height = max(1.0, y2 - y1)
    width = max(1.0, x2 - x1)
    per_digit_aspect = width / max(1, len(raw_digits)) / height
    if per_digit_aspect > 1.5:
        return 1.0
    return 0.0


def strong_class_zero_support(
    bbox: BBox,
    yolo: Sequence[Dict[str, Any]],
    width: int,
    height: int,
) -> bool:
    for box in yolo:
        if box.get("class_id") != 0 or float(box.get("conf", 1.0)) < 0.25:
            continue
        ybox = tuple(float(v) for v in box["bbox_xyxy"])
        if (
            iou(bbox, ybox) > 0.12
            or overlap_min_fraction(bbox, ybox) > 0.45
            or center_distance_norm(bbox, ybox, width, height) < 0.045
        ):
            return True
    return False


def likely_text_hallucination(
    candidate: Dict[str, Any],
    raw_digits: str,
    text: str,
    bbox: BBox,
    all_candidates: Sequence[Dict[str, Any]],
    yolo: Sequence[Dict[str, Any]],
    width: int,
    height: int,
) -> bool:
    """Conservatively reject a low-confidence number read from a text line.

    The check is opt-in and deliberately requires several image-only signals:
    weak OCR confidence, text-line geometry, and either overlapping alphabetic
    OCR or detector class-2 evidence. A current class-0 channel box overrides
    the rejection.
    """
    if not raw_digits or strong_class_zero_support(bbox, yolo, width, height):
        return False
    conf = float(candidate.get("ocr_conf", candidate.get("confidence", 0.5)) or 0.0)
    if conf >= 0.80:
        return False
    box_width = max(1.0, bbox[2] - bbox[0])
    box_height = max(1.0, bbox[3] - bbox[1])
    per_digit_aspect = box_width / max(1, len(raw_digits)) / box_height
    if per_digit_aspect < 1.45:
        return False

    alpha_context = same_location_alpha_only_context(candidate, all_candidates, width, height)
    if not alpha_context:
        for other in original_ocr_candidates(all_candidates):
            if other is candidate or "bbox_xyxy" not in other or not is_alpha_only_text(other.get("text", "")):
                continue
            other_bbox = tuple(float(v) for v in other["bbox_xyxy"])
            if overlap_min_fraction(bbox, other_bbox) > 0.45 or center_distance_norm(
                bbox, other_bbox, width, height
            ) < 0.045:
                alpha_context = True
                break
    parent_text = str(candidate.get("parent_text", ""))
    mixed_parent = bool(parent_text) and has_alpha(parent_text) and not has_independent_numeric_token(parent_text, raw_digits)
    class_two_overlap = 0.0
    for box in yolo:
        if box.get("class_id") != 2:
            continue
        ybox = tuple(float(v) for v in box["bbox_xyxy"])
        class_two_overlap = max(class_two_overlap, overlap_min_fraction(bbox, ybox))
    return bool(alpha_context or mixed_parent or class_two_overlap >= 0.45)


def candidate_score(
    candidate: Dict[str, Any],
    yolo: List[Dict[str, Any]],
    width: int,
    height: int,
    all_candidates: Sequence[Dict[str, Any]],
    max_channel_digits: int = 5,
    allow_single_digit_channel_recheck: bool = False,
    suppress_text_hallucinations: bool = False,
) -> Tuple[float, Optional[Dict[str, Any]]]:
    text = str(candidate.get("text", ""))
    d = digits(text)
    if not d or len(d) > max_channel_digits:
        return -999.0, None
    bbox = tuple(float(v) for v in candidate["bbox_xyxy"])
    conf = float(candidate.get("ocr_conf", candidate.get("confidence", 0.5)))
    if suppress_text_hallucinations and likely_text_hallucination(
        candidate,
        d,
        text,
        bbox,
        all_candidates,
        yolo,
        width,
        height,
    ):
        return -999.0, None
    if conf < 0.35 and same_location_alpha_only_context(candidate, all_candidates, width, height):
        return -999.0, None
    if not channel_recheck_text_support(
        candidate,
        all_candidates,
        yolo,
        width,
        height,
        d,
        allow_single_digit=allow_single_digit_channel_recheck,
    ):
        return -999.0, None
    score = 0.35 + 0.45 * max(0.0, min(conf, 1.0)) + position_score(bbox, width, height)
    score += 0.45 * max(0.0, min(float(candidate.get("history_score", 0.0)), 1.0))
    if len(d) in (2, 3):
        score += 0.20
    if ":" in text or "pm" in text.lower() or "am" in text.lower():
        score -= 0.55
    if candidate.get("source") == "refined_numeric_substring":
        score -= 0.10
    if is_numeric_recheck_source(candidate.get("source")):
        score += 0.08
    if candidate.get("source") == "temporal_slot_recheck":
        slot_score = max(0.0, min(float(candidate.get("slot_profile_score", 0.0)), 1.0))
        score += 0.35 + 0.25 * slot_score
    if (
        candidate.get("from_history_guided_slot")
        or candidate.get("from_history_roi_prior")
        or candidate.get("raw_source") in {"history_guided_slot_recheck", "history_roi_prior_recheck"}
    ):
        slot_boost = max(0.0, min(float(candidate.get("temporal_slot_boost", 0.0)), 1.0))
        score += 0.12 + 0.20 * slot_boost
    score += text_quality_adjustment(candidate, d, text)
    score += same_location_ocr_adjustment(candidate, all_candidates, width, height)
    score -= elongated_numeric_textline_penalty(candidate, d, text, bbox)
    best_box = None
    best_bonus = 0.0
    negative_penalty = 0.0
    trusted_numeric = trusted_numeric_candidate(candidate, d, text)
    weak_channel_area_evidence = mixed_alpha_without_numeric_token(candidate, d, text)
    for box in yolo:
        ybox = tuple(float(v) for v in box["bbox_xyxy"])
        ov = iou(bbox, ybox)
        dist = center_distance_norm(bbox, ybox, width, height)
        if box.get("class_id") in (1, 2) and ov > 0.10:
            penalty = 0.20 + 0.45 * ov
            if trusted_numeric:
                penalty *= 0.35 if box.get("class_id") == 1 else 0.50
            negative_penalty = max(negative_penalty, penalty)
            continue
        if box.get("class_id") not in (0, 3):
            continue
        class_bonus = 0.55 if box.get("class_id") == 0 else 0.35
        distance_window = 0.25 if box.get("class_id") == 0 else 0.32
        bonus = max(0.0, ov * class_bonus) + max(0.0, distance_window - dist)
        if box.get("class_id") == 3 and weak_channel_area_evidence:
            bonus *= 0.25
        if bonus > best_bonus:
            best_bonus = bonus
            best_box = box
    return score + best_bonus - negative_penalty, best_box


def same_location(a: Dict[str, Any], b: Dict[str, Any], width: int, height: int) -> bool:
    if "bbox_xyxy" not in a or "bbox_xyxy" not in b:
        return False
    abox = tuple(float(v) for v in a["bbox_xyxy"])
    bbox = tuple(float(v) for v in b["bbox_xyxy"])
    return overlap_min_fraction(abox, bbox) > 0.55 or center_distance_norm(abox, bbox, width, height) < 0.045


def parent_starts_with_digits(candidate: Dict[str, Any], value: str) -> bool:
    parent_digits = digits(str(candidate.get("parent_text", "")))
    return bool(parent_digits) and parent_digits.startswith(value)


def duplicate_digit_support(candidate: Dict[str, Any], ranked: Sequence[Dict[str, Any]], width: int, height: int) -> int:
    value = digits(candidate.get("text", ""))
    if not value:
        return 0
    return sum(
        1
        for other in ranked
        if other is not candidate
        and digits(other.get("text", "")) == value
        and same_location(candidate, other, width, height)
    )


def promote_supported_three_digit(ranked: List[Dict[str, Any]], width: int, height: int) -> List[Dict[str, Any]]:
    if not ranked:
        return ranked
    best = ranked[0]
    best_value = digits(best.get("text", ""))
    if len(best_value) >= 3:
        return ranked
    best_score = float(best.get("fusion_score", -999.0))
    for index, candidate in enumerate(ranked[1:], 1):
        value = digits(candidate.get("text", ""))
        if len(value) != 3:
            continue
        score = float(candidate.get("fusion_score", -999.0))
        if score < best_score:
            continue
        margin = best_score - score
        source = str(candidate.get("source", "ocr"))
        conf = float(candidate.get("ocr_conf", candidate.get("confidence", 0.0)))
        rule_score = float(candidate.get("rule_fusion_score", 0.0))
        has_yolo_alignment = bool(candidate.get("aligned_yolo_id"))
        comparable_rule = rule_score >= 0.25 and margin <= 1.0
        repeated_crop = source == "paddleocr_crop_variant_recheck" and conf >= 0.80 and duplicate_digit_support(candidate, ranked, width, height) >= 1
        aligned_recheck = source in ("paddleocr_channel_recheck", "paddleocr_crop_variant_recheck") and has_yolo_alignment and conf >= 0.55 and margin <= 2.0
        inherited_prefix = source == "digit_sequence_trim" and parent_starts_with_digits(candidate, value) and margin <= 1.0
        if comparable_rule and (same_location(best, candidate, width, height) or inherited_prefix or has_yolo_alignment):
            ranked.insert(0, ranked.pop(index))
            ranked[0]["promotion_reason"] = "supported_three_digit_rule"
            return ranked
        if repeated_crop or aligned_recheck or inherited_prefix:
            ranked.insert(0, ranked.pop(index))
            ranked[0]["promotion_reason"] = "supported_three_digit_rule"
            return ranked
    return ranked


def select_candidate(
    image: Dict[str, Any],
    yolo: List[Dict[str, Any]],
    ranker: Any = None,
    value_group_selector: Any = None,
    max_channel_digits: int = 5,
    allow_single_digit_channel_recheck: bool = False,
    suppress_text_hallucinations: bool = False,
) -> Dict[str, Any]:
    width = int(image["image_width"])
    height = int(image["image_height"])
    if ranker is not None:
        try:
            from candidate_ranker import annotate_candidate_pool_features

            annotate_candidate_pool_features(image.get("candidates", []), yolo, width, height)
        except Exception:
            pass
    ranked = []
    for candidate in image.get("candidates", []):
        if "bbox_xyxy" not in candidate:
            continue
        candidate_digits = digits(candidate.get("text", ""))
        if not candidate_digits or len(candidate_digits) > max_channel_digits:
            continue
        score, box = candidate_score(
            candidate,
            yolo,
            width,
            height,
            image.get("candidates", []),
            max_channel_digits=max_channel_digits,
            allow_single_digit_channel_recheck=allow_single_digit_channel_recheck,
            suppress_text_hallucinations=suppress_text_hallucinations,
        )
        if score < -100:
            continue
        item = dict(candidate)
        item["digits"] = candidate_digits
        item["normalized_digits"] = normalized_digits(item.get("text", ""))
        item["rule_fusion_score"] = round(score, 6)
        if ranker is None:
            item["fusion_score"] = round(score, 6)
        else:
            ranker_score = ranker.score_candidate(candidate, yolo, width, height)
            item["ranker_score"] = round(ranker_score, 6)
            item["fusion_score"] = round(ranker_score, 6)
            item["selection_model"] = getattr(ranker, "model_type", "candidate_ranker")
        item["aligned_yolo_id"] = None if box is None else box["id"]
        if box is not None:
            cbox = tuple(float(v) for v in item["bbox_xyxy"])
            ybox = tuple(float(v) for v in box["bbox_xyxy"])
            if iou(cbox, ybox) > 0.15 or center_distance_norm(cbox, ybox, width, height) < 0.06:
                item["original_bbox_xyxy"] = item["bbox_xyxy"]
                item["bbox_xyxy"] = box["bbox_xyxy"]
                item["bbox_source"] = "yolo_aligned" if box.get("class_id") == 0 else "channel_number_area_aligned"
        ranked.append(item)
    ranked.sort(key=lambda c: float(c["fusion_score"]), reverse=True)
    if value_group_selector is not None:
        return value_group_selector.select(ranked, yolo, width, height)
    ranked = promote_supported_three_digit(ranked, width, height)
    return {"best_candidate": ranked[0] if ranked else None, "ranked_candidates": ranked[:20]}


def has_channel_yolo_evidence(yolo: Sequence[Dict[str, Any]], min_conf: float) -> bool:
    return any(box.get("class_id") in (0, 3) and float(box.get("conf", 1.0)) >= min_conf for box in yolo)


def draw_result(image_path: Path, out_path: Path, image: Dict[str, Any], selected: Optional[Dict[str, Any]], yolo: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    with Image.open(image_path) as src:
        canvas = src.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for box in yolo:
        color = {
            0: (180, 80, 255),
            1: (255, 150, 80),
            2: (80, 210, 120),
            3: (255, 210, 50),
        }.get(box.get("class_id"), (255, 210, 50))
        draw.rectangle(tuple(box["bbox_xyxy"]), outline=color, width=3)
    for candidate in image.get("candidates", []):
        if "bbox_xyxy" not in candidate or not digits(candidate.get("text", "")):
            continue
        draw.rectangle(tuple(candidate["bbox_xyxy"]), outline=(80, 180, 255), width=1)
    if selected:
        draw.rectangle(tuple(selected["bbox_xyxy"]), outline=(255, 30, 30), width=5)
        draw.text((selected["bbox_xyxy"][0], max(0, selected["bbox_xyxy"][1] - 14)), f"SELECT {selected.get('digits')}", fill=(255, 30, 30), font=font)
    canvas.save(out_path)


def save_crop(image_path: Path, out_path: Path, bbox: Sequence[float]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as src:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        src.crop((max(0, x1), max(0, y1), min(src.width, x2), min(src.height, y2))).save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--ocr-json", type=Path, required=True)
    parser.add_argument("--yolo-label-dir", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--visualize-dir", type=Path, required=True)
    parser.add_argument("--crops-dir", type=Path, required=True)
    parser.add_argument(
        "--require-yolo-channel",
        action="store_true",
        help="Return no prediction when YOLO has no class 0/3 channel ROI evidence.",
    )
    parser.add_argument("--min-yolo-channel-conf", type=float, default=0.25)
    parser.add_argument(
        "--ranker-model",
        type=Path,
        default=None,
        help="Optional candidate ranker JSON. When supplied, candidates are selected by ranker score.",
    )
    parser.add_argument(
        "--ranker-threshold",
        type=float,
        default=None,
        help="When a ranker is used, reject the image if the best ranker score is below this threshold.",
    )
    parser.add_argument(
        "--value-group-ranker-model",
        type=Path,
        default=None,
        help="Opt-in model that reranks candidates grouped by exact digit value.",
    )
    parser.add_argument(
        "--relative-gate-model",
        type=Path,
        default=None,
        help="Opt-in relative confidence gate applied after value-group ranking.",
    )
    parser.add_argument(
        "--relative-gate-threshold",
        type=float,
        default=None,
        help="Threshold for --relative-gate-model. Requires both opt-in model paths.",
    )
    parser.add_argument(
        "--relative-gate-policy",
        choices=("auto", "known_present", "positive_first"),
        default="auto",
        help=(
            "auto applies the learned no-output gate. known_present bypasses rejection when the caller guarantees "
            "that channel UI is visible. positive_first emits Top-1 whenever a rankable candidate exists while "
            "retaining the gate result as a confidence status; use it for UI-dominant operating conditions."
        ),
    )
    parser.add_argument(
        "--relative-gate-multi-position-rescue",
        action="store_true",
        help=(
            "Opt-in rescue for a rejected top value repeated at multiple positions when one spatial cluster "
            "has strong class 0/3 YOLO and OCR evidence."
        ),
    )
    parser.add_argument("--relative-gate-rescue-min-yolo-conf", type=float, default=0.8)
    parser.add_argument("--relative-gate-rescue-min-ocr-conf", type=float, default=0.9)
    parser.add_argument("--relative-gate-rescue-min-margin", type=float, default=1.0)
    parser.add_argument(
        "--max-channel-digits",
        type=int,
        default=5,
        help="Maximum allowed digits in the final channel-number prediction.",
    )
    parser.add_argument(
        "--allow-single-digit-channel-recheck",
        action="store_true",
        help=(
            "Opt-in: admit one-digit paddleocr_channel_recheck candidates when the existing same-location OCR "
            "or class 0/3 YOLO evidence checks pass. The legacy default still rejects them."
        ),
    )
    parser.add_argument(
        "--suppress-text-hallucinations",
        action="store_true",
        help=(
            "Opt-in conservative rejection of low-confidence, text-line-shaped numeric candidates when "
            "alphabetic/class-2 evidence overlaps and no current class-0 channel box supports the crop."
        ),
    )
    args = parser.parse_args()

    doc = json.loads(args.ocr_json.read_text(encoding="utf-8"))
    gt = json.loads(args.ground_truth.read_text(encoding="utf-8")) if args.ground_truth else {}
    ranker = None
    if args.ranker_model is not None:
        from candidate_ranker import CandidateRanker

        ranker = CandidateRanker(args.ranker_model)
    opt_in_values = (
        args.value_group_ranker_model,
        args.relative_gate_model,
        args.relative_gate_threshold,
    )
    if any(value is not None for value in opt_in_values) and not all(value is not None for value in opt_in_values):
        raise SystemExit(
            "--value-group-ranker-model, --relative-gate-model, and --relative-gate-threshold must be supplied together"
        )
    if args.relative_gate_policy != "auto" and not all(value is not None for value in opt_in_values):
        raise SystemExit(
            "--relative-gate-policy known_present/positive_first requires the value-group relative-gate model options"
        )
    if args.relative_gate_multi_position_rescue and not all(value is not None for value in opt_in_values):
        raise SystemExit("--relative-gate-multi-position-rescue requires the value-group relative-gate model options")
    value_group_selector = None
    if all(value is not None for value in opt_in_values):
        if ranker is None:
            raise SystemExit("--ranker-model is required for opt-in value-group selection")
        if args.ranker_threshold is not None:
            raise SystemExit("do not combine --ranker-threshold with --relative-gate-threshold")
        from value_group_runtime import ValueGroupRelativeSelector

        value_group_selector = ValueGroupRelativeSelector(
            args.value_group_ranker_model,
            args.relative_gate_model,
            args.relative_gate_threshold,
            gate_policy=args.relative_gate_policy,
            multi_position_rescue=args.relative_gate_multi_position_rescue,
            rescue_min_yolo_conf=args.relative_gate_rescue_min_yolo_conf,
            rescue_min_ocr_conf=args.relative_gate_rescue_min_ocr_conf,
            rescue_min_margin=args.relative_gate_rescue_min_margin,
        )
    rows = []
    failures = []
    out_images = []
    for image in doc.get("images", []):
        image_id = image["image_id"]
        image_path = resolve_image_path(Path(str(image.get("image_path", ""))), args.images, image_id)
        if image_path.exists():
            with Image.open(image_path) as src:
                image["image_width"], image["image_height"] = src.size
        width, height = int(image["image_width"]), int(image["image_height"])
        yolo = yolo_boxes(args.yolo_label_dir, image_id, width, height)
        gate_passed = (not args.require_yolo_channel) or has_channel_yolo_evidence(yolo, args.min_yolo_channel_conf)
        selected_doc = (
            select_candidate(
                image,
                yolo,
                ranker=ranker,
                value_group_selector=value_group_selector,
                max_channel_digits=args.max_channel_digits,
                allow_single_digit_channel_recheck=args.allow_single_digit_channel_recheck,
                suppress_text_hallucinations=args.suppress_text_hallucinations,
            )
            if gate_passed
            else {"best_candidate": None, "ranked_candidates": [], "reject_reason": "no_yolo_channel_roi"}
        )
        selected = selected_doc["best_candidate"]
        if selected is not None and args.ranker_threshold is not None and value_group_selector is None:
            selected_score = float(selected.get("ranker_score", selected.get("fusion_score", -999.0)))
            if selected_score < args.ranker_threshold:
                selected_doc["reject_reason"] = "ranker_threshold"
                selected_doc["rejected_best_candidate"] = selected
                selected = None
                selected_doc["best_candidate"] = None
        raw_pred = "" if selected is None else selected.get("digits", "")
        pred = "" if selected is None else selected.get("normalized_digits", "")
        truth = str(gt.get(image_id, ""))
        normalized_truth = normalized_digits(truth)
        numeric_pred = numeric_equivalent_digits(raw_pred)
        numeric_truth = numeric_equivalent_digits(truth)
        string_exact = bool(truth) and raw_pred == normalized_truth
        exact = bool(truth) and numeric_pred == numeric_truth
        row = {
            "image_id": image_id,
            "ground_truth": truth,
            "normalized_ground_truth": normalized_truth,
            "prediction": pred,
            "raw_prediction": raw_pred,
            "numeric_prediction": numeric_pred,
            "numeric_ground_truth": numeric_truth,
            "exact_match": int(exact),
            "string_exact_match": int(string_exact),
            "score": "" if selected is None else selected.get("fusion_score", ""),
            "rule_fusion_score": "" if selected is None else selected.get("rule_fusion_score", ""),
            "ranker_score": "" if selected is None else selected.get("ranker_score", ""),
            "value_group_score": "" if selected is None else selected.get("value_group_score", ""),
            "relative_gate_score": selected_doc.get("relative_gate_score", ""),
            "relative_gate_threshold": selected_doc.get("relative_gate_threshold", ""),
            "relative_gate_policy": selected_doc.get("relative_gate_policy", ""),
            "relative_gate_score_passed": selected_doc.get("relative_gate_score_passed", ""),
            "relative_gate_bypassed": selected_doc.get("relative_gate_bypassed", ""),
            "positive_first_output": selected_doc.get("positive_first_output", ""),
            "output_confidence_status": selected_doc.get("output_confidence_status", ""),
            "relative_gate_multi_position_rescue_enabled": selected_doc.get(
                "relative_gate_multi_position_rescue_enabled", ""
            ),
            "relative_gate_multi_position_rescue_passed": selected_doc.get(
                "relative_gate_multi_position_rescue_passed", ""
            ),
            "selection_model": selected_doc.get(
                "selection_model",
                "" if selected is None else selected.get("selection_model", "rule_fusion"),
            ),
            "selected_id": "" if selected is None else selected.get("id", ""),
            "selected_text": "" if selected is None else selected.get("text", ""),
            "selected_source": "" if selected is None else selected.get("source", "ocr"),
            "selected_ocr_conf": "" if selected is None else selected.get("ocr_conf", selected.get("confidence", "")),
            "selected_detection_conf": "" if selected is None else selected.get("detection_conf", ""),
            "bbox_source": "" if selected is None else selected.get("bbox_source", "ocr"),
            "reject_reason": selected_doc.get("reject_reason", "") if gate_passed else "no_yolo_channel_roi",
            "yolo_channel_boxes": sum(1 for box in yolo if box.get("class_id") == 0),
            "yolo_channel_area_boxes": sum(1 for box in yolo if box.get("class_id") == 3),
        }
        rows.append(row)
        if truth and not exact:
            failures.append(row)
        if image_path.exists():
            draw_result(image_path, args.visualize_dir / f"{image_id}_fusion.jpg", image, selected, yolo)
            if selected:
                save_crop(image_path, args.crops_dir / f"{image_id}_{pred or 'none'}.jpg", selected["bbox_xyxy"])
        out_item = dict(image)
        out_item["yolo_channel_boxes"] = yolo
        out_item["fusion"] = selected_doc
        out_item["ground_truth_channel_number"] = truth
        out_item["normalized_ground_truth_channel_number"] = normalized_truth
        out_item["raw_predicted_channel_number"] = raw_pred
        out_item["predicted_channel_number"] = pred
        out_item["numeric_predicted_channel_number"] = numeric_pred
        out_item["numeric_ground_truth_channel_number"] = numeric_truth
        out_item["exact_match"] = exact
        out_item["string_exact_match"] = string_exact
        out_images.append(out_item)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"images": out_images}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    fail_path = args.out.with_name(args.out.stem + "_failures.csv")
    with fail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(failures)
    acc = sum(int(r["exact_match"]) for r in rows) / len(rows) if rows else 0.0
    print(f"wrote {args.out}")
    print(f"wrote {csv_path}")
    print(f"wrote {fail_path}")
    print(f"exact_match_accuracy={acc:.4f} ({sum(int(r['exact_match']) for r in rows)}/{len(rows)})")


if __name__ == "__main__":
    main()
