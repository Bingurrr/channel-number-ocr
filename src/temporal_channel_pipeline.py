"""History-aware channel-number candidate selection.

This is an offline/sequence runner for model-generated candidates. It does not
read annotation JSON for ROI discovery. If a ground-truth JSON is supplied, it
is used only after prediction for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from channel_number_fusion import (
    digits,
    draw_result,
    numeric_equivalent_digits,
    normalized_digits,
    resolve_image_path,
    save_crop,
    select_candidate,
    yolo_boxes,
)
from roi_history_tracker import (
    best_track_for_bbox,
    center_distance_norm,
    iou as history_bbox_iou,
    observations_from_candidates,
    observations_from_yolo,
    overlap_min_fraction,
    RoiHistoryTracker,
    TemporalSlotState,
)


HISTORY_PRIOR_MODES = {"roi_prior", "roi_prior_recheck"}
HISTORY_RECHECK_MODES = {"proposal_guided", "roi_prior_recheck"}


def order_images(images: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(images, key=lambda item: str(item.get("image_id", "")))


def load_sequence_manifest(path: Optional[Path], images: Sequence[Dict[str, Any]]) -> List[Tuple[str, str, List[Dict[str, Any]]]]:
    image_by_id = {str(image.get("image_id", "")): image for image in images}
    if path is None:
        return [("default", "", order_images(images))]
    doc = json.loads(path.read_text(encoding="utf-8"))
    sequences: List[Tuple[str, str, List[Dict[str, Any]]]] = []
    for index, seq in enumerate(doc.get("sequences", []), 1):
        sequence_id = str(seq.get("sequence_id") or f"sequence_{index:04d}")
        group_key = str(seq.get("group_key", ""))
        seq_images = []
        for image_id in seq.get("images", []):
            image = image_by_id.get(str(image_id))
            if image is not None:
                seq_images.append(image)
        if seq_images:
            sequences.append((sequence_id, group_key, seq_images))
    return sequences


def load_gt(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    out: Dict[str, str] = {}
    for item in data:
        image_id = str(item.get("image_id") or item.get("id"))
        value = item.get("channel_number") or item.get("ground_truth_channel_number")
        if value is not None:
            out[image_id] = str(value)
    return out


def compact_artifact_stem(global_frame_index: int, sequence_frame_index: int, image_id: str) -> str:
    digest = hashlib.sha1(image_id.encode("utf-8")).hexdigest()[:10]
    return f"f{global_frame_index:05d}_s{sequence_frame_index:03d}_{digest}"


def annotate_candidates_with_history(
    image: Dict[str, Any],
    stable_tracks: Sequence[Any],
    *,
    frame_width: int,
    frame_height: int,
    history_bonus_scale: float,
    lock_track_ids: Optional[Sequence[str]] = None,
) -> None:
    lock_ids = {str(track_id) for track_id in (lock_track_ids or [])}
    for candidate in image.get("candidates", []):
        if "bbox_xyxy" not in candidate:
            continue
        bbox = tuple(float(v) for v in candidate["bbox_xyxy"])
        candidate["history_iou"] = 0.0
        candidate["history_distance"] = 1.0
        candidate["history_overlap_min_fraction"] = 0.0
        candidate["history_track_score"] = 0.0
        candidate["history_track_seen_count"] = 0
        candidate["history_track_missed_count"] = 0
        candidate["history_track_stability"] = 0.0
        candidate["history_is_lockable"] = False
        track, proximity = best_track_for_bbox(
            bbox,
            stable_tracks,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if track is None:
            continue
        track_score = track.score(int(image.get("_frame_index", 0)), 8)
        history_score = max(0.0, min(1.0, track_score * proximity))
        candidate["history_track_id"] = track.track_id
        candidate["history_score"] = round(history_score * history_bonus_scale, 6)
        candidate["history_roi_bbox_xyxy"] = [round(v, 3) for v in track.bbox]
        candidate["history_iou"] = round(history_bbox_iou(bbox, track.bbox), 6)
        candidate["history_distance"] = round(
            center_distance_norm(bbox, track.bbox, frame_width, frame_height),
            6,
        )
        candidate["history_overlap_min_fraction"] = round(overlap_min_fraction(bbox, track.bbox), 6)
        candidate["history_track_score"] = round(track_score, 6)
        candidate["history_track_seen_count"] = int(track.seen_count)
        candidate["history_track_missed_count"] = int(track.missed_count)
        candidate["history_track_stability"] = round(float(track.stability), 6)
        candidate["history_is_lockable"] = str(track.track_id) in lock_ids
        candidate["temporal_slot_iou"] = candidate["history_iou"]
        candidate["temporal_slot_distance"] = candidate["history_distance"]


def keep_candidates_near_stable_roi(
    image: Dict[str, Any],
    min_history_score: float,
    tracks: Sequence[Any],
) -> Dict[str, Any]:
    allowed_track_ids = {str(getattr(track, "track_id", "")) for track in tracks}
    kept = []
    for candidate in image.get("candidates", []):
        if str(candidate.get("history_track_id", "")) not in allowed_track_ids:
            continue
        if float(candidate.get("history_score", 0.0)) >= min_history_score:
            kept.append(candidate)
    out = dict(image)
    out["candidates"] = kept
    out["history_locked_candidate_count"] = len(kept)
    return out


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def history_guided_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "min_conf": args.history_guided_min_conf,
        "max_digits": args.max_channel_digits,
        "max_candidates": args.history_guided_max_candidates,
        "min_crop_std": args.history_guided_min_crop_std,
        "max_crop_area_ratio": args.history_guided_max_crop_area_ratio,
        "raw_only": args.history_guided_raw_only,
        "one_digit_recovery": args.history_guided_one_digit_recovery,
        "one_digit_only": False,
    }


def candidate_bbox(candidate: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    raw = candidate.get("bbox_xyxy", candidate.get("bbox"))
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    return tuple(float(value) for value in raw)


def current_class_support(
    bbox: Tuple[float, float, float, float],
    yolo: Sequence[Dict[str, Any]],
    class_id: int,
    frame_width: int,
    frame_height: int,
    min_conf: float = 0.25,
) -> bool:
    for box in yolo:
        if box.get("class_id") != class_id or float(box.get("conf", 1.0)) < min_conf:
            continue
        other = candidate_bbox(box)
        if other is None:
            continue
        if (
            history_bbox_iou(bbox, other) > 0.12
            or overlap_min_fraction(bbox, other) > 0.42
            or center_distance_norm(bbox, other, frame_width, frame_height) < 0.045
        ):
            return True
    return False


def history_candidate_evidence_decision(
    candidate: Dict[str, Any],
    current_candidates: Sequence[Dict[str, Any]],
    yolo: Sequence[Dict[str, Any]],
    frame_width: int,
    frame_height: int,
    *,
    min_evidence_score: float,
    one_digit_min_evidence_score: float,
) -> Tuple[bool, str]:
    """Validate a newly OCR-read history crop using current-frame pixels/evidence."""
    bbox = candidate_bbox(candidate)
    value = digits(candidate.get("text", ""))
    if bbox is None or not value:
        return False, "missing_bbox_or_digits"
    evidence_score = float(candidate.get("visible_text_evidence_score", 0.0) or 0.0)
    threshold = one_digit_min_evidence_score if len(value) == 1 else min_evidence_score
    weak_visible_evidence = evidence_score < threshold
    low_contrast = float(candidate.get("crop_std", 0.0) or 0.0) < 6.0
    edge_density = float(candidate.get("edge_density", 0.0) or 0.0)
    implausible_edges = edge_density < 0.003 or edge_density > 0.48
    implausible_one_digit_geometry = False
    if len(value) == 1:
        digit_components = int(candidate.get("digit_like_component_count", 0) or 0)
        aspect = float(candidate.get("crop_aspect_per_digit", 0.0) or 0.0)
        implausible_one_digit_geometry = digit_components not in (1, 2) or not (0.18 <= aspect <= 2.20)

    class_zero_support = current_class_support(bbox, yolo, 0, frame_width, frame_height)
    exact_numeric_support = False
    alpha_conflict = False
    for other in current_candidates:
        other_bbox = candidate_bbox(other)
        if other_bbox is None:
            continue
        close = (
            overlap_min_fraction(bbox, other_bbox) > 0.45
            or center_distance_norm(bbox, other_bbox, frame_width, frame_height) < 0.045
        )
        if not close:
            continue
        other_text = str(other.get("text", ""))
        other_digits = digits(other_text)
        if other_digits == value and not any(ch.isalpha() for ch in other_text):
            exact_numeric_support = True
        elif any(ch.isalpha() for ch in other_text) and not other_digits:
            alpha_conflict = True

    class_two_conflict = current_class_support(bbox, yolo, 2, frame_width, frame_height)
    supported = class_zero_support or exact_numeric_support
    if supported and not (low_contrast and implausible_edges):
        return True, "accepted_supported"
    if (alpha_conflict or class_two_conflict) and not supported and (
        weak_visible_evidence or low_contrast or implausible_edges or implausible_one_digit_geometry
    ):
        return False, "text_or_class2_conflict_without_channel_support"
    if implausible_one_digit_geometry and not supported:
        return False, "implausible_one_digit_geometry"
    if low_contrast and implausible_edges:
        return False, "blank_or_low_content_crop"
    return True, "accepted"


def filter_history_candidates_by_evidence(
    candidates: Sequence[Dict[str, Any]],
    current_candidates: Sequence[Dict[str, Any]],
    yolo: Sequence[Dict[str, Any]],
    frame_width: int,
    frame_height: int,
    *,
    min_evidence_score: float,
    one_digit_min_evidence_score: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    kept: List[Dict[str, Any]] = []
    reasons: Dict[str, int] = {}
    for candidate in candidates:
        accepted, reason = history_candidate_evidence_decision(
            candidate,
            current_candidates,
            yolo,
            frame_width,
            frame_height,
            min_evidence_score=min_evidence_score,
            one_digit_min_evidence_score=one_digit_min_evidence_score,
        )
        candidate["history_visual_evidence_accepted"] = bool(accepted)
        candidate["history_visual_evidence_reason"] = reason
        reasons[reason] = reasons.get(reason, 0) + 1
        if accepted:
            kept.append(candidate)
    return kept, reasons


def eligible_one_digit_yolo_boxes(
    yolo: Sequence[Dict[str, Any]],
    current_candidates: Sequence[Dict[str, Any]],
    frame_width: int,
    frame_height: int,
    *,
    min_conf: float,
    max_aspect: float,
    max_area_ratio: float,
) -> List[Dict[str, Any]]:
    eligible: List[Dict[str, Any]] = []
    for box in yolo:
        if box.get("class_id") != 0 or float(box.get("conf", 1.0)) < min_conf:
            continue
        bbox = candidate_bbox(box)
        if bbox is None:
            continue
        box_width = max(1.0, bbox[2] - bbox[0])
        box_height = max(1.0, bbox[3] - bbox[1])
        if box_width / box_height > max_aspect:
            continue
        if box_width * box_height / max(1.0, float(frame_width * frame_height)) > max_area_ratio:
            continue
        has_multi_digit_support = False
        for candidate in current_candidates:
            other_bbox = candidate_bbox(candidate)
            if other_bbox is None or len(digits(candidate.get("text", ""))) < 2:
                continue
            if overlap_min_fraction(bbox, other_bbox) > 0.45:
                has_multi_digit_support = True
                break
        if not has_multi_digit_support:
            eligible.append(box)
    return eligible


def history_roi_prior_from_state(state: Dict[str, Any], frame_id: str = "") -> List[Dict[str, Any]]:
    """Return bbox-only ROI priors from previous-frame temporal state.

    A history ROI prior is deliberately not a detector output and does not
    contain channel-number text. It is only a weak ROI evidence box for the
    current frame.
    """
    if state.get("state") != "LOCKED" or not state.get("locked_slot_bbox"):
        return []
    bbox = [round(float(v), 3) for v in state.get("locked_slot_bbox") or []]
    if len(bbox) != 4:
        return []
    history_score = float(state.get("history_score", 0.0) or 0.0)
    return [
        {
            "id": f"history_roi_prior_{frame_id}" if frame_id else "history_roi_prior",
            "source": "history_roi_prior",
            "raw_source": "history_roi_prior",
            "class_id": 3,
            "class_name": "channel_number_area",
            "bbox_xyxy": bbox,
            "conf": 0.0,
            "history_prior_confidence": round(history_score, 6),
            "history_score": round(history_score, 6),
            "history_track_id": str(state.get("history_track_id", "")),
            "locked_slot_age": int(state.get("locked_slot_age", 0) or 0),
            "seen_count": int(state.get("locked_slot_age", 0) or 0),
            "missed_count": int(state.get("frames_since_seen", 0) or 0),
            "from_history_roi_prior": True,
            "history_state": str(state.get("state", "")),
            "history_reason": "stable_lockable_roi",
        }
    ]


def mark_history_roi_recheck_candidates(
    candidates: Sequence[Dict[str, Any]],
    priors: Sequence[Dict[str, Any]],
) -> None:
    if not priors:
        return
    prior = priors[0]
    for candidate in candidates:
        candidate["raw_source"] = "history_roi_prior_recheck"
        candidate["from_history_roi_prior"] = True
        candidate["history_roi_bbox"] = prior.get("bbox_xyxy", "")
        candidate["history_prior_score"] = prior.get("history_score", 0.0)
        candidate["history_prior_confidence"] = prior.get("history_prior_confidence", 0.0)
        candidate["history_track_id"] = prior.get("history_track_id", candidate.get("history_track_id", ""))
        candidate["locked_slot_age"] = prior.get("locked_slot_age", candidate.get("locked_slot_age", 0))
        candidate["history_recheck_variant"] = candidate.get("history_recheck_variant", "history_roi_prior_recheck")


def draw_history_guided_result(
    image_path: Path,
    out_path: Path,
    image: Dict[str, Any],
    selected: Optional[Dict[str, Any]],
    yolo: List[Dict[str, Any]],
    *,
    locked_slot_bbox: Optional[Sequence[float]],
    history_candidates: Sequence[Dict[str, Any]],
    final_output: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    with Image.open(image_path) as src:
        canvas = src.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for box in yolo:
        if box.get("from_history_roi_prior"):
            color = (255, 190, 0)
            width = 4
        else:
            color = (180, 80, 255) if box.get("class_id") == 0 else (255, 210, 50) if box.get("class_id") == 3 else (90, 160, 255)
            width = 2
        draw.rectangle(tuple(box["bbox_xyxy"]), outline=color, width=width)
        if box.get("from_history_roi_prior"):
            draw.text((float(box["bbox_xyxy"][0]), max(0.0, float(box["bbox_xyxy"][1]) - 14)), "HISTORY ROI PRIOR", fill=color, font=font)
    for candidate in image.get("candidates", []):
        if "bbox_xyxy" not in candidate or not digits(candidate.get("text", "")):
            continue
        is_history = candidate.get("from_history_guided_slot") or candidate.get("from_history_roi_prior")
        color = (40, 220, 170) if is_history else (80, 180, 255)
        width = 3 if is_history else 1
        draw.rectangle(tuple(candidate["bbox_xyxy"]), outline=color, width=width)
    if locked_slot_bbox is not None:
        draw.rectangle(tuple(float(v) for v in locked_slot_bbox), outline=(255, 190, 0), width=4)
        draw.text((float(locked_slot_bbox[0]), max(0.0, float(locked_slot_bbox[1]) - 14)), "LOCKED SLOT", fill=(255, 190, 0), font=font)
    if selected:
        draw.rectangle(tuple(selected["bbox_xyxy"]), outline=(255, 30, 30), width=5)
        draw.text((selected["bbox_xyxy"][0], max(0, selected["bbox_xyxy"][1] - 14)), f"SELECT {selected.get('digits')}", fill=(255, 30, 30), font=font)
    draw.text((8, 8), f"final={final_output or 'no_output'} history_candidates={len(history_candidates)}", fill=(255, 255, 255), font=font)
    canvas.save(out_path)


def lockable_tracks(stable_tracks: Sequence[Any], min_yolo_support: int) -> List[Any]:
    if min_yolo_support <= 0:
        return list(stable_tracks)
    return [
        track
        for track in stable_tracks
        if int(getattr(track, "channel_detector_count", 0)) >= min_yolo_support
    ]


def consensus_candidates(values: Sequence[str]) -> List[str]:
    out = set()
    for value in values:
        value = digits(value)
        if not value:
            continue
        out.add(value)
        if len(value) > 3:
            out.add(value[:3])
            out.add(value[-3:])
        if len(value) > 2:
            out.add(value[:2])
            out.add(value[-2:])
    return sorted(out, key=lambda item: (-len(item), item))


def supports_consensus(observed: str, candidate: str) -> bool:
    if observed == candidate:
        return True
    if len(candidate) < 2:
        return False
    return observed.startswith(candidate) or observed.endswith(candidate)


def choose_sequence_consensus(items: Sequence[Dict[str, Any]], max_channel_digits: int = 5) -> str:
    values: List[str] = []
    weights: List[float] = []
    for item in items:
        selected = ((item.get("temporal_selection") or {}).get("best_candidate") or {})
        value = digits(item.get("raw_predicted_channel_number", ""))
        if not value:
            continue
        conf = float(selected.get("ocr_conf", selected.get("confidence", 0.5)) or 0.5)
        score = float(selected.get("fusion_score", 0.0) or 0.0)
        source = str(selected.get("source", ""))
        source_bonus = 0.35 if source == "temporal_slot_recheck" else 0.15 if source == "paddleocr_channel_recheck" else 0.0
        length_bonus = 0.25 if len(value) in (2, 3) else -0.20 if len(value) == 1 else -0.05
        values.append(value)
        weights.append(max(0.05, 1.0 + conf + 0.25 * score + source_bonus + length_bonus))

    if len(values) < 2:
        return ""

    total_weight = sum(weights)
    best_value = ""
    best_support_weight = 0.0
    best_exact_count = 0
    for candidate in consensus_candidates(values):
        if len(candidate) > max_channel_digits:
            continue
        support_weight = 0.0
        exact_count = 0
        support_count = 0
        for observed, weight in zip(values, weights):
            if supports_consensus(observed, candidate):
                support_weight += weight
                support_count += 1
                if observed == candidate:
                    exact_count += 1
        if support_count < 2:
            continue
        if len(candidate) == 1 and exact_count < 2:
            continue
        rank = (support_weight, exact_count, 1 if len(candidate) in (2, 3) else 0, -len(candidate))
        best_rank = (
            best_support_weight,
            best_exact_count,
            1 if len(best_value) in (2, 3) else 0,
            -len(best_value),
        )
        if rank > best_rank:
            best_value = candidate
            best_support_weight = support_weight
            best_exact_count = exact_count

    if not best_value:
        return ""
    if best_exact_count >= 2 or best_support_weight >= total_weight * 0.55:
        return best_value
    return ""


def apply_sequence_consensus(
    out_images: Sequence[Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
    max_channel_digits: int = 5,
    guard_mode: str = "off",
    min_consensus_agreement: int = 2,
    protect_high_confidence_single_frame: float = 1.25,
    require_current_frame_support_for_consensus: bool = False,
) -> None:
    by_sequence: Dict[str, List[Dict[str, Any]]] = {}
    for item in out_images:
        by_sequence.setdefault(str(item.get("sequence_id", "")), []).append(item)
    row_by_id = {str(row.get("image_id", "")): row for row in rows}
    for sequence_items in by_sequence.values():
        consensus = choose_sequence_consensus(sequence_items, max_channel_digits=max_channel_digits)
        for item in sequence_items:
            item["sequence_consensus_channel_number"] = consensus
            item["sequence_consensus_applied"] = int(bool(consensus))
            row = row_by_id.get(str(item.get("image_id", "")))
            if row is not None:
                row["sequence_consensus_prediction"] = consensus
                row["sequence_consensus_applied"] = int(bool(consensus))
            if not consensus:
                continue
            current_raw = digits(item.get("raw_predicted_channel_number", ""))
            selection = item.get("temporal_selection") or {}
            selected = selection.get("best_candidate") or {}
            ranked = selection.get("ranked_candidates") or []
            current_support = any(
                digits(candidate.get("text") or candidate.get("digits") or "") == consensus
                for candidate in ranked[:10]
                if isinstance(candidate, dict)
            )
            agreement_count = sum(
                1
                for other in sequence_items
                if supports_consensus(digits(other.get("raw_predicted_channel_number", "")), consensus)
            )
            selected_score = 0.0
            try:
                selected_score = float(
                    selected.get(
                        "relative_gate_score",
                        selected.get("ranker_score", selected.get("fusion_score", 0.0)),
                    )
                    or 0.0
                )
            except (TypeError, ValueError):
                selected_score = 0.0
            guard_reason = ""
            if guard_mode != "off" and current_raw and current_raw != consensus:
                if agreement_count < min_consensus_agreement:
                    guard_reason = "low_consensus_agreement"
                elif require_current_frame_support_for_consensus and not current_support:
                    guard_reason = "no_current_frame_support_for_consensus"
                elif guard_mode == "conservative" and selected_score >= protect_high_confidence_single_frame:
                    guard_reason = "protected_high_confidence_single_frame"
                elif guard_mode == "balanced" and selected_score >= protect_high_confidence_single_frame and not current_support:
                    guard_reason = "protected_high_confidence_without_current_support"
            if guard_reason:
                item["sequence_consensus_guarded"] = True
                item["sequence_consensus_guard_reason"] = guard_reason
                if row is not None:
                    row["sequence_consensus_guarded"] = 1
                    row["sequence_consensus_guard_reason"] = guard_reason
                continue
            item["raw_predicted_channel_number"] = consensus
            item["predicted_channel_number"] = normalized_digits(consensus)
            item["numeric_predicted_channel_number"] = numeric_equivalent_digits(consensus)
            selection["sequence_consensus_prediction"] = consensus
            selection["sequence_consensus_applied"] = True
            selection["sequence_consensus_agreement_count"] = agreement_count
            selection["sequence_consensus_current_frame_support"] = bool(current_support)
            item["temporal_selection"] = selection
            if row is not None:
                row["raw_prediction"] = consensus
                row["prediction"] = normalized_digits(consensus)
                row["numeric_prediction"] = numeric_equivalent_digits(consensus)
                row["sequence_consensus_agreement_count"] = agreement_count
                row["sequence_consensus_current_frame_support"] = int(bool(current_support))


def run_self_test() -> None:
    class FakeRecognizer:
        def predict_many(self, crops: Sequence[Image.Image]) -> List[List[Tuple[str, float]]]:
            return [[("041", 0.99)] for _ in crops]

    tracker = RoiHistoryTracker(min_seen=3, stable_score=0.68)
    state = TemporalSlotState()
    width, height = 200, 100
    bbox = [20.0, 40.0, 64.0, 62.0]
    for frame_index in range(1, 4):
        yolo = [{"id": "yolo_0001", "class_id": 0, "bbox_xyxy": bbox, "conf": 0.95}]
        tracker.update(
            observations_from_yolo(yolo),
            frame_index=frame_index,
            frame_width=width,
            frame_height=height,
        )
        state.update_from_tracks(
            lockable_tracks(tracker.stable_tracks(frame_index), 1),
            frame_index=frame_index,
            frame_width=width,
            frame_height=height,
            max_missed_before_unlock=2,
        )
    assert state.state == "LOCKED", f"expected LOCKED after repeated bbox, got {state.state}"

    changed_text_candidate = [{"id": "cand_1", "source": "ocr", "text": "999", "bbox_xyxy": bbox, "ocr_conf": 0.95}]
    tracker.update(
        observations_from_candidates(changed_text_candidate),
        frame_index=4,
        frame_width=width,
        frame_height=height,
    )
    state.update_from_tracks(
        lockable_tracks(tracker.stable_tracks(4), 0),
        frame_index=4,
        frame_width=width,
        frame_height=height,
        max_missed_before_unlock=2,
    )
    assert state.state == "LOCKED", "bbox lock should survive text changes"
    assert "locked_text" not in state.to_dict(), "state must not lock channel number text"
    feature_candidates = [{"id": "feature_1", "source": "ocr", "text": "123", "bbox_xyxy": bbox, "ocr_conf": 0.9}]
    feature_image = {"_frame_index": 4, "candidates": feature_candidates}
    feature_tracks = tracker.stable_tracks(4)
    feature_lock_tracks = lockable_tracks(feature_tracks, 0)
    annotate_candidates_with_history(
        feature_image,
        feature_tracks,
        frame_width=width,
        frame_height=height,
        history_bonus_scale=1.0,
        lock_track_ids=[track.track_id for track in feature_lock_tracks],
    )
    assert len(feature_image["candidates"]) == 1, "feature-only annotation must preserve candidates"
    assert feature_candidates[0]["history_iou"] > 0.5
    assert feature_candidates[0]["history_distance"] < 0.05
    assert feature_candidates[0]["history_track_seen_count"] >= 3
    locked_priors = history_roi_prior_from_state(state.to_dict(), frame_id="selftest")
    assert locked_priors, "locked bbox should create a history ROI prior"
    assert locked_priors[0]["source"] == "history_roi_prior"
    assert locked_priors[0]["class_id"] == 3
    assert "text" not in locked_priors[0] and "value" not in locked_priors[0], "history ROI prior must not lock text/value"

    for frame_index in range(5, 9):
        tracker.update([], frame_index=frame_index, frame_width=width, frame_height=height)
        state.update_from_tracks(
            lockable_tracks(tracker.stable_tracks(frame_index), 1),
            frame_index=frame_index,
            frame_width=width,
            frame_height=height,
            max_missed_before_unlock=2,
        )
    assert state.state in {"UNLOCK_COOLDOWN", "SEARCH"}, f"expected unlock after missing evidence, got {state.state}"

    one_frame_tracker = RoiHistoryTracker(min_seen=3, stable_score=0.68)
    one_frame_state = TemporalSlotState()
    one_frame_tracker.update(
        observations_from_yolo([{"id": "wrong", "class_id": 0, "bbox_xyxy": [120, 10, 170, 30], "conf": 0.99}]),
        frame_index=1,
        frame_width=width,
        frame_height=height,
    )
    one_frame_state.update_from_tracks(
        lockable_tracks(one_frame_tracker.stable_tracks(1), 1),
        frame_index=1,
        frame_width=width,
        frame_height=height,
    )
    assert one_frame_state.state != "LOCKED", "single-frame bbox must not lock"

    from slot_proposal_numeric_recheck import run_history_guided_slot_recheck

    with tempfile.TemporaryDirectory() as tmp:
        image_path = Path(tmp) / "frame.jpg"
        image = Image.new("RGB", (200, 100), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.text((24, 42), "041", fill=(255, 255, 255))
        image.save(image_path)
        candidates = run_history_guided_slot_recheck(
            image_path,
            bbox,
            FakeRecognizer(),
            {"min_conf": 0.5, "max_candidates": 4, "min_crop_std": 0.1},
            frame_id="selftest",
            history_metadata={
                "state": "LOCKED",
                "history_track_id": "roi_selftest",
                "history_score": 0.91,
                "locked_slot_age": 3,
            },
        )
        mark_history_roi_recheck_candidates(candidates, locked_priors)
    assert candidates, "locked bbox should generate history-guided candidates"
    assert candidates[0]["raw_source"] == "history_roi_prior_recheck"
    assert candidates[0]["from_history_guided_slot"] is True
    assert candidates[0]["from_history_roi_prior"] is True
    assert candidates[0]["text"] == "041"

    good_candidate = {
        "text": "7",
        "bbox_xyxy": [20, 35, 42, 72],
        "crop_std": 52.0,
        "edge_density": 0.08,
        "digit_like_component_count": 1,
        "crop_aspect_per_digit": 0.59,
        "visible_text_evidence_score": 0.72,
    }
    accepted, reason = history_candidate_evidence_decision(
        good_candidate,
        [],
        [{"class_id": 0, "conf": 0.9, "bbox_xyxy": [19, 34, 43, 73]}],
        200,
        100,
        min_evidence_score=0.38,
        one_digit_min_evidence_score=0.48,
    )
    assert accepted and reason == "accepted_supported"
    bad_candidate = dict(good_candidate)
    bad_candidate.update(
        {
            "bbox_xyxy": [20, 35, 150, 72],
            "visible_text_evidence_score": 0.22,
            "crop_aspect_per_digit": 3.51,
        }
    )
    accepted, reason = history_candidate_evidence_decision(
        bad_candidate,
        [{"text": "NEWS", "bbox_xyxy": [20, 35, 150, 72]}],
        [{"class_id": 2, "conf": 0.9, "bbox_xyxy": [20, 35, 150, 72]}],
        200,
        100,
        min_evidence_score=0.38,
        one_digit_min_evidence_score=0.48,
    )
    assert not accepted and reason == "text_or_class2_conflict_without_channel_support"

    from channel_number_fusion import likely_text_hallucination

    text_like_number = {
        "text": "20",
        "source": "paddleocr_channel_recheck",
        "ocr_conf": 0.61,
        "bbox_xyxy": [20, 35, 160, 70],
    }
    alpha_ocr = {"text": "NEWS", "source": "ocr", "ocr_conf": 0.99, "bbox_xyxy": [20, 35, 160, 70]}
    class_two = {"class_id": 2, "conf": 0.9, "bbox_xyxy": [20, 35, 160, 70]}
    assert likely_text_hallucination(
        text_like_number,
        "20",
        "20",
        (20.0, 35.0, 160.0, 70.0),
        [text_like_number, alpha_ocr],
        [class_two],
        200,
        100,
    )
    assert not likely_text_hallucination(
        text_like_number,
        "20",
        "20",
        (20.0, 35.0, 160.0, 70.0),
        [text_like_number, alpha_ocr],
        [class_two, {"class_id": 0, "conf": 0.9, "bbox_xyxy": [20, 35, 160, 70]}],
        200,
        100,
    )
    print("temporal history self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="Run lightweight temporal history self-tests and exit.")
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument(
        "--candidate-json",
        type=Path,
        default=None,
        help="Model-generated OCR/recheck candidate JSON. Do not pass annotation JSON.",
    )
    parser.add_argument(
        "--sequence-manifest",
        type=Path,
        default=None,
        help="Optional image-id sequence groups. Used only to order frames and reset history.",
    )
    parser.add_argument("--yolo-label-dir", type=Path, default=None)
    parser.add_argument("--ground-truth", type=Path, default=None, help="Evaluation only.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--visualize-dir", type=Path, default=None)
    parser.add_argument("--crops-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["search", "auto", "locked"], default="auto")
    parser.add_argument(
        "--temporal-history-mode",
        choices=["off", "feature_only", "score_only", "proposal_guided", "roi_prior", "roi_prior_recheck"],
        default="score_only",
        help=(
            "feature_only annotates every candidate with history geometry without filtering; "
            "score_only preserves the legacy history scoring/filtering; "
            "roi_prior injects locked bbox as weak ROI evidence; "
            "roi_prior_recheck also OCR-rechecks that ROI. "
            "proposal_guided is the legacy alias for bbox recheck."
        ),
    )
    parser.add_argument("--min-seen", type=int, default=3)
    parser.add_argument("--stable-score", type=float, default=0.68)
    parser.add_argument("--history-bonus-scale", type=float, default=1.0)
    parser.add_argument("--lock-min-history-score", type=float, default=0.20)
    parser.add_argument("--history-min-lock-observations", type=int, default=3)
    parser.add_argument("--history-min-track-score", type=float, default=0.68)
    parser.add_argument("--history-min-lock-iou", type=float, default=0.40)
    parser.add_argument("--history-max-missed-before-unlock", type=int, default=5)
    parser.add_argument("--history-max-bbox-area-ratio", type=float, default=0.16)
    parser.add_argument("--history-max-bbox-aspect", type=float, default=7.0)
    parser.add_argument("--temporal-history-debug-dir", type=Path, default=None)
    parser.add_argument("--history-guided-model-dir", type=Path, default=Path("runs/ocr/inference"))
    parser.add_argument("--history-guided-model-name", default="PP-OCRv5_mobile_rec")
    parser.add_argument("--history-guided-device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--history-guided-input-shape", default="3,48,320")
    parser.add_argument("--history-guided-min-conf", type=float, default=0.90)
    parser.add_argument("--history-guided-max-candidates", type=int, default=8)
    parser.add_argument("--history-guided-min-crop-std", type=float, default=3.0)
    parser.add_argument("--history-guided-max-crop-area-ratio", type=float, default=0.18)
    parser.add_argument("--history-guided-raw-only", action="store_true")
    parser.add_argument(
        "--history-guided-one-digit-recovery",
        action="store_true",
        help="Opt in to tight connected-component proposals for a visibly single-glyph locked slot.",
    )
    parser.add_argument(
        "--history-guided-evidence-gate",
        action="store_true",
        help="Opt in to pixel/component evidence checks for newly generated history OCR candidates.",
    )
    parser.add_argument("--history-guided-min-evidence-score", type=float, default=0.38)
    parser.add_argument("--history-guided-one-digit-min-evidence-score", type=float, default=0.48)
    parser.add_argument(
        "--one-digit-yolo-roi-recheck",
        action="store_true",
        help="Opt in to one-digit OCR retry on compact current-frame YOLO class-0 boxes.",
    )
    parser.add_argument("--one-digit-yolo-min-conf", type=float, default=0.50)
    parser.add_argument("--one-digit-yolo-max-aspect", type=float, default=1.60)
    parser.add_argument("--one-digit-yolo-max-area-ratio", type=float, default=0.015)
    parser.add_argument(
        "--lock-min-yolo-support",
        type=int,
        default=1,
        help="Stable ROI must have this many YOLO class 0/3 observations before auto/locked mode can lock to it.",
    )
    parser.add_argument(
        "--sequence-consensus",
        action="store_true",
        help="Use repeated predictions within each sequence to stabilize the final channel number.",
    )
    parser.add_argument("--temporal-guard-mode", choices=["off", "conservative", "balanced"], default="off")
    parser.add_argument("--min-consensus-agreement", type=int, default=2)
    parser.add_argument("--protect-high-confidence-single-frame", type=float, default=1.25)
    parser.add_argument("--require-current-frame-support-for-consensus", action="store_true")
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
        help="When a ranker is used, reject the frame if the best ranker score is below this threshold.",
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
            "auto applies the learned no-output gate. known_present keeps value-group ranking but bypasses "
            "no-output rejection when the caller guarantees that channel UI is visible. positive_first emits "
            "Top-1 whenever a rankable candidate exists while preserving the learned confidence status."
        ),
    )
    parser.add_argument(
        "--allow-single-digit-channel-recheck",
        action="store_true",
        help=(
            "Opt in to one-digit paddleocr_channel_recheck candidates when existing same-location OCR or "
            "YOLO channel evidence supports them. The legacy default remains off."
        ),
    )
    parser.add_argument(
        "--suppress-text-hallucinations",
        action="store_true",
        help=(
            "Opt-in conservative removal of low-confidence numeric candidates that look like text lines, "
            "overlap alphabetic/class-2 evidence, and lack current class-0 support."
        ),
    )
    parser.add_argument(
        "--max-channel-digits",
        type=int,
        default=5,
        help="Maximum allowed digits in the final channel-number prediction.",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    missing_args = [
        name
        for name, value in (
            ("--images", args.images),
            ("--candidate-json", args.candidate_json),
            ("--yolo-label-dir", args.yolo_label_dir),
            ("--out", args.out),
        )
        if value is None
    ]
    if missing_args:
        raise SystemExit(f"missing required arguments: {', '.join(missing_args)}")

    doc = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    sequences = load_sequence_manifest(args.sequence_manifest, doc.get("images", []))
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
        )

    history_recognizer = None
    history_recheck_fn = None
    if args.temporal_history_mode in HISTORY_RECHECK_MODES or args.one_digit_yolo_roi_recheck:
        from channel_digit_recognizer import ChannelDigitRecognizer
        from slot_proposal_numeric_recheck import run_history_guided_slot_recheck

        history_recognizer = ChannelDigitRecognizer(
            args.history_guided_model_dir,
            model_name=args.history_guided_model_name,
            device=args.history_guided_device,
            input_shape=args.history_guided_input_shape,
        )
        history_recheck_fn = run_history_guided_slot_recheck

    rows: List[Dict[str, Any]] = []
    out_images: List[Dict[str, Any]] = []

    global_frame_index = 0
    for sequence_id, group_key, sequence_images in sequences:
        tracker = RoiHistoryTracker(min_seen=args.min_seen, stable_score=args.stable_score)
        slot_state = TemporalSlotState()
        for sequence_frame_index, raw_image in enumerate(sequence_images, 1):
            global_frame_index += 1
            image = dict(raw_image)
            image["candidates"] = list(image.get("candidates") or [])
            image["_frame_index"] = sequence_frame_index
            image["sequence_id"] = sequence_id
            image["sequence_group_key"] = group_key
            image_id = str(image["image_id"])
            image_path = resolve_image_path(Path(str(image.get("image_path", ""))), args.images, image_id)
            if image_path.exists():
                with Image.open(image_path) as src:
                    image["image_width"], image["image_height"] = src.size
            width, height = int(image["image_width"]), int(image["image_height"])

            yolo = yolo_boxes(args.yolo_label_dir, image_id, width, height)
            current_frame_candidates = list(image.get("candidates", []))
            state_before = slot_state.to_dict()
            history_roi_priors = history_roi_prior_from_state(state_before, frame_id=image_id) if args.temporal_history_mode in HISTORY_PRIOR_MODES else []
            roi_evidence = yolo + history_roi_priors
            generated_history_candidates: List[Dict[str, Any]] = []
            generated_one_digit_candidates: List[Dict[str, Any]] = []
            history_evidence_reasons: Dict[str, int] = {}
            one_digit_evidence_reasons: Dict[str, int] = {}
            if (
                args.one_digit_yolo_roi_recheck
                and image_path.exists()
                and history_recognizer is not None
                and history_recheck_fn is not None
            ):
                for yolo_box in eligible_one_digit_yolo_boxes(
                    yolo,
                    current_frame_candidates,
                    width,
                    height,
                    min_conf=args.one_digit_yolo_min_conf,
                    max_aspect=args.one_digit_yolo_max_aspect,
                    max_area_ratio=args.one_digit_yolo_max_area_ratio,
                ):
                    one_digit_config = history_guided_config_from_args(args)
                    one_digit_config.update(
                        {
                            "one_digit_recovery": True,
                            "one_digit_only": True,
                            "max_candidates": 3,
                        }
                    )
                    hits = history_recheck_fn(
                        image_path,
                        yolo_box.get("bbox_xyxy", []),
                        history_recognizer,
                        one_digit_config,
                        frame_id=f"{image_id}_yolo1",
                        history_metadata={"state": "CURRENT_FRAME", "history_score": 0.0},
                    )
                    for hit in hits:
                        hit.update(
                            {
                                "source": "paddleocr_channel_recheck",
                                "raw_source": "yolo_one_digit_recheck",
                                "normalized_source": "paddleocr_channel_recheck",
                                "from_history_guided_slot": False,
                                "from_history_roi_prior": False,
                                "from_current_yolo_one_digit_recheck": True,
                                "aligned_yolo_id": yolo_box.get("id", ""),
                            }
                        )
                    generated_one_digit_candidates.extend(hits)
                if args.history_guided_evidence_gate:
                    generated_one_digit_candidates, one_digit_evidence_reasons = filter_history_candidates_by_evidence(
                        generated_one_digit_candidates,
                        current_frame_candidates,
                        yolo,
                        width,
                        height,
                        min_evidence_score=args.history_guided_min_evidence_score,
                        one_digit_min_evidence_score=args.history_guided_one_digit_min_evidence_score,
                    )
                image["candidates"].extend(generated_one_digit_candidates)
            if (
                args.temporal_history_mode in HISTORY_RECHECK_MODES
                and slot_state.is_locked()
                and image_path.exists()
                and history_recognizer is not None
                and history_recheck_fn is not None
            ):
                if args.temporal_history_mode == "proposal_guided" and not history_roi_priors:
                    history_roi_priors = history_roi_prior_from_state(state_before, frame_id=image_id)
                generated_history_candidates = history_recheck_fn(
                    image_path,
                    slot_state.locked_slot_bbox or [],
                    history_recognizer,
                    history_guided_config_from_args(args),
                    frame_id=image_id,
                    history_metadata=state_before,
                )
                if args.history_guided_evidence_gate:
                    generated_history_candidates, history_evidence_reasons = filter_history_candidates_by_evidence(
                        generated_history_candidates,
                        current_frame_candidates,
                        yolo,
                        width,
                        height,
                        min_evidence_score=args.history_guided_min_evidence_score,
                        one_digit_min_evidence_score=args.history_guided_one_digit_min_evidence_score,
                    )
                if args.temporal_history_mode == "roi_prior_recheck":
                    mark_history_roi_recheck_candidates(generated_history_candidates, history_roi_priors)
                image["candidates"].extend(generated_history_candidates)

            observations = observations_from_yolo(yolo) + observations_from_candidates(image.get("candidates", []))
            tracker.update(
                observations,
                frame_index=sequence_frame_index,
                frame_width=width,
                frame_height=height,
            )
            stable_tracks = tracker.stable_tracks(sequence_frame_index)
            lock_tracks = lockable_tracks(stable_tracks, args.lock_min_yolo_support)

            if args.temporal_history_mode != "off":
                annotate_candidates_with_history(
                    image,
                    stable_tracks,
                    frame_width=width,
                    frame_height=height,
                    history_bonus_scale=args.history_bonus_scale,
                    lock_track_ids=[track.track_id for track in lock_tracks],
                )

            locked = bool(lock_tracks) and args.mode in ("auto", "locked") and args.temporal_history_mode != "off"
            history_filter_applied = locked and args.temporal_history_mode != "feature_only"
            selection_image = (
                keep_candidates_near_stable_roi(image, args.lock_min_history_score, lock_tracks)
                if history_filter_applied
                else image
            )
            if args.mode == "locked" and not lock_tracks:
                selected_doc = {"best_candidate": None, "ranked_candidates": [], "reject_reason": "no_stable_history_roi"}
            else:
                selected_doc = select_candidate(
                    selection_image,
                    roi_evidence,
                    ranker=ranker,
                    value_group_selector=value_group_selector,
                    max_channel_digits=args.max_channel_digits,
                    allow_single_digit_channel_recheck=args.allow_single_digit_channel_recheck,
                    suppress_text_hallucinations=args.suppress_text_hallucinations,
                )

            selected = selected_doc["best_candidate"]
            if selected is not None and args.ranker_threshold is not None and value_group_selector is None:
                selected_score = float(selected.get("ranker_score", selected.get("fusion_score", -999.0)))
                if selected_score < args.ranker_threshold:
                    selected_doc["reject_reason"] = "ranker_threshold"
                    selected_doc["rejected_best_candidate"] = selected
                    selected = None
                    selected_doc["best_candidate"] = None
            raw_pred = "" if selected is None else selected.get("digits", digits(selected.get("text", "")))
            pred = "" if selected is None else selected.get("normalized_digits", normalized_digits(raw_pred))
            numeric_pred = numeric_equivalent_digits(raw_pred)
            slot_state.update_from_tracks(
                lock_tracks,
                frame_index=sequence_frame_index,
                frame_width=width,
                frame_height=height,
                min_lock_observations=args.history_min_lock_observations,
                min_track_score=args.history_min_track_score,
                min_lock_iou=args.history_min_lock_iou,
                max_missed_before_unlock=args.history_max_missed_before_unlock,
                max_bbox_area_ratio=args.history_max_bbox_area_ratio,
                max_bbox_aspect=args.history_max_bbox_aspect,
                warmup_frames=tracker.warmup_frames,
            )
            state_after = slot_state.to_dict()

            row = {
                "frame_index": global_frame_index,
                "sequence_id": sequence_id,
                "sequence_group_key": group_key,
                "sequence_frame_index": sequence_frame_index,
                "image_id": image_id,
                "ground_truth": "",
                "prediction": pred,
                "raw_prediction": raw_pred,
                "numeric_prediction": numeric_pred,
                "numeric_ground_truth": "",
                "exact_match": "",
                "selected_source": "" if selected is None else selected.get("source", "ocr"),
                "selected_raw_source": "" if selected is None else selected.get("raw_source", selected.get("source", "ocr")),
                "selected_score": "" if selected is None else selected.get("fusion_score", ""),
                "value_group_score": "" if selected is None else selected.get("value_group_score", ""),
                "relative_gate_score": selected_doc.get("relative_gate_score", ""),
                "relative_gate_threshold": selected_doc.get("relative_gate_threshold", ""),
                "relative_gate_policy": selected_doc.get("relative_gate_policy", ""),
                "relative_gate_score_passed": selected_doc.get("relative_gate_score_passed", ""),
                "relative_gate_bypassed": selected_doc.get("relative_gate_bypassed", ""),
                "positive_first_output": selected_doc.get("positive_first_output", ""),
                "selection_model": selected_doc.get(
                    "selection_model",
                    "" if selected is None else selected.get("selection_model", "rule_fusion"),
                ),
                "selected_ocr_conf": "" if selected is None else selected.get("ocr_conf", selected.get("confidence", "")),
                "history_track_id": "" if selected is None else selected.get("history_track_id", ""),
                "history_score": "" if selected is None else selected.get("history_score", ""),
                "stable_track_count": len(stable_tracks),
                "lock_track_count": len(lock_tracks),
                "history_locked": int(locked),
                "history_candidate_filter_applied": int(history_filter_applied),
                "candidate_count_before_history_filter": len(image.get("candidates", [])),
                "candidate_count_after_history_filter": len(selection_image.get("candidates", [])),
                "temporal_history_mode": args.temporal_history_mode,
                "history_roi_prior_count": len(history_roi_priors),
                "history_roi_prior_used_as_roi_count": len(history_roi_priors),
                "history_state_before": state_before.get("state", ""),
                "history_state_after": state_after.get("state", ""),
                "history_transition_reason": state_after.get("last_transition_reason", ""),
                "history_generated_candidate_count": len(generated_history_candidates),
                "one_digit_yolo_recheck_candidate_count": len(generated_one_digit_candidates),
                "history_evidence_rejected_count": sum(
                    count for reason, count in history_evidence_reasons.items() if not reason.startswith("accepted")
                ),
                "one_digit_evidence_rejected_count": sum(
                    count for reason, count in one_digit_evidence_reasons.items() if not reason.startswith("accepted")
                ),
                "history_roi_recheck_candidate_count": len(generated_history_candidates) if args.temporal_history_mode == "roi_prior_recheck" else 0,
                "history_selected": int(bool(selected and selected.get("from_history_guided_slot"))),
                "history_selected_from_roi_prior": int(bool(selected and selected.get("from_history_roi_prior"))),
                "reject_reason": selected_doc.get("reject_reason", ""),
                "sequence_consensus_prediction": "",
                "sequence_consensus_applied": 0,
                "sequence_consensus_guarded": 0,
                "sequence_consensus_guard_reason": "",
                "sequence_consensus_agreement_count": "",
                "sequence_consensus_current_frame_support": "",
            }
            rows.append(row)

            artifact_stem = compact_artifact_stem(global_frame_index, sequence_frame_index, image_id)
            if image_path.exists() and args.visualize_dir:
                if args.temporal_history_mode in ("proposal_guided", "roi_prior", "roi_prior_recheck"):
                    draw_history_guided_result(
                        image_path,
                        args.visualize_dir / f"{artifact_stem}_temporal.jpg",
                        image,
                        selected,
                        roi_evidence,
                        locked_slot_bbox=state_before.get("locked_slot_bbox"),
                        history_candidates=generated_history_candidates,
                        final_output=raw_pred,
                    )
                else:
                    draw_result(image_path, args.visualize_dir / f"{artifact_stem}_temporal.jpg", image, selected, yolo)
            if image_path.exists() and args.crops_dir and selected:
                save_crop(image_path, args.crops_dir / f"{artifact_stem}_{pred or 'none'}.jpg", selected["bbox_xyxy"])

            out_item = dict(image)
            out_item["global_frame_index"] = global_frame_index
            out_item["sequence_frame_index"] = sequence_frame_index
            out_item["yolo_channel_boxes"] = yolo
            out_item["history_roi_priors"] = history_roi_priors
            out_item["roi_evidence_boxes"] = roi_evidence
            out_item["roi_history"] = tracker.snapshot(sequence_frame_index)
            out_item["stable_history_rois"] = [
                track.to_dict(sequence_frame_index, tracker.warmup_frames) for track in stable_tracks
            ]
            out_item["lockable_history_rois"] = [
                track.to_dict(sequence_frame_index, tracker.warmup_frames) for track in lock_tracks
            ]
            out_item["temporal_history_mode"] = args.temporal_history_mode
            out_item["history_candidate_filter_applied"] = bool(history_filter_applied)
            out_item["candidate_count_before_history_filter"] = len(image.get("candidates", []))
            out_item["candidate_count_after_history_filter"] = len(selection_image.get("candidates", []))
            out_item["temporal_slot_state_before"] = state_before
            out_item["temporal_slot_state_after"] = state_after
            out_item["history_guided_candidates"] = generated_history_candidates
            out_item["history_generated_candidate_count"] = len(generated_history_candidates)
            out_item["one_digit_yolo_recheck_candidates"] = generated_one_digit_candidates
            out_item["one_digit_yolo_recheck_candidate_count"] = len(generated_one_digit_candidates)
            out_item["history_evidence_reasons"] = history_evidence_reasons
            out_item["one_digit_evidence_reasons"] = one_digit_evidence_reasons
            out_item["temporal_selection"] = selected_doc
            out_item["raw_predicted_channel_number"] = raw_pred
            out_item["predicted_channel_number"] = pred
            out_item["numeric_predicted_channel_number"] = numeric_pred
            out_item["ground_truth_channel_number"] = ""
            out_item["normalized_ground_truth_channel_number"] = ""
            out_item["exact_match"] = ""
            out_item["sequence_consensus_channel_number"] = ""
            out_item["sequence_consensus_applied"] = 0
            out_images.append(out_item)

            if args.temporal_history_debug_dir:
                append_jsonl(
                    args.temporal_history_debug_dir / f"{sequence_id}.jsonl",
                    {
                        "sequence_id": sequence_id,
                        "frame_id": image_id,
                        "frame_index": global_frame_index,
                        "sequence_frame_index": sequence_frame_index,
                        "state_before": state_before,
                        "current_yolo_boxes": yolo,
                        "history_roi_priors": history_roi_priors,
                        "merged_roi_evidence": roi_evidence,
                        "lockable_tracks": [
                            track.to_dict(sequence_frame_index, tracker.warmup_frames) for track in lock_tracks
                        ],
                        "history_candidate_filter_applied": bool(history_filter_applied),
                        "candidate_count_before_history_filter": len(image.get("candidates", [])),
                        "candidate_count_after_history_filter": len(selection_image.get("candidates", [])),
                        "locked_slot_bbox": state_before.get("locked_slot_bbox"),
                        "generated_history_candidate_count": len(generated_history_candidates),
                        "generated_history_candidates": generated_history_candidates,
                        "generated_one_digit_candidates": generated_one_digit_candidates,
                        "history_evidence_reasons": history_evidence_reasons,
                        "one_digit_evidence_reasons": one_digit_evidence_reasons,
                        "selected_candidate": selected,
                        "final_output": raw_pred,
                        "state_after": state_after,
                        "transition_reason": state_after.get("last_transition_reason", ""),
                    },
                )

    if args.sequence_consensus:
        apply_sequence_consensus(
            out_images,
            rows,
            max_channel_digits=args.max_channel_digits,
            guard_mode=args.temporal_guard_mode,
            min_consensus_agreement=args.min_consensus_agreement,
            protect_high_confidence_single_frame=args.protect_high_confidence_single_frame,
            require_current_frame_support_for_consensus=args.require_current_frame_support_for_consensus,
        )

    gt = load_gt(args.ground_truth)
    failures: List[Dict[str, Any]] = []
    if gt:
        out_by_id = {str(item["image_id"]): item for item in out_images}
        for row in rows:
            image_id = str(row["image_id"])
            truth = str(gt.get(image_id, ""))
            normalized_truth = normalized_digits(truth)
            numeric_truth = numeric_equivalent_digits(truth)
            exact = bool(truth) and str(row["numeric_prediction"]) == numeric_truth
            row["ground_truth"] = truth
            row["numeric_ground_truth"] = numeric_truth
            row["exact_match"] = int(exact)
            out_item = out_by_id.get(image_id)
            if out_item is not None:
                out_item["ground_truth_channel_number"] = truth
                out_item["normalized_ground_truth_channel_number"] = normalized_truth
                out_item["exact_match"] = exact
            if truth and not exact:
                failures.append(row)

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

    evaluable_rows = [row for row in rows if row["ground_truth"]]
    correct = sum(int(row["exact_match"]) for row in evaluable_rows)
    total = len(evaluable_rows)
    acc = correct / total if total else 0.0
    print(f"wrote {args.out}")
    print(f"wrote {csv_path}")
    print(f"wrote {fail_path}")
    if total:
        print(f"temporal_exact_match_accuracy={acc:.4f} ({correct}/{total})")
    else:
        print("temporal_exact_match_accuracy=not_available (no ground truth)")


if __name__ == "__main__":
    main()
