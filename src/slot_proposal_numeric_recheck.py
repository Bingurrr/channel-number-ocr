"""Create small numeric slot proposals before channel digit recognition.

This runner keeps inference image-only: it does not read annotation JSON for
ROI discovery. It starts from the same YOLO/fallback regions used by the
existing numeric recheck, splits broad regions into smaller text-like boxes,
and runs the fine-tuned PaddleOCR channel digit recognizer on those crops.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from channel_digit_recognizer import ChannelDigitRecognizer
from easyocr_numeric_recheck import likely_regions, resolve_image_path


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Proposal:
    bbox: BBox
    kind: str
    score: float


@dataclass(frozen=True)
class HistoryGuidedSlotConfig:
    min_conf: float = 0.90
    max_digits: int = 5
    max_candidates: int = 8
    min_crop_std: float = 3.0
    max_crop_area_ratio: float = 0.18
    raw_only: bool = False
    one_digit_recovery: bool = False
    one_digit_only: bool = False


def clamp_box(box: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def box_area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    denom = box_area(a) + box_area(b) - inter
    return 0.0 if denom <= 0.0 else inter / denom


def expand_box(box: BBox, width: int, height: int, pad_x: float, pad_y: float) -> BBox:
    x1, y1, x2, y2 = box
    return clamp_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def dedupe_proposals(proposals: Iterable[Proposal], iou_threshold: float = 0.68) -> List[Proposal]:
    out: List[Proposal] = []
    for proposal in sorted(proposals, key=lambda item: (item.score, box_area(item.bbox)), reverse=True):
        if box_area(proposal.bbox) <= 0:
            continue
        if any(iou(proposal.bbox, prev.bbox) > iou_threshold for prev in out):
            continue
        out.append(proposal)
    return out


def geometric_subwindows(region: BBox, image_width: int, image_height: int) -> List[Proposal]:
    x1, y1, x2, y2 = region
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    proposals: List[Proposal] = []
    proposals.append(Proposal(clamp_box(region, image_width, image_height), "region_full", 0.38))

    # These are generic slot scans inside a broad UI strip, not provider
    # coordinates. They cover short labels that sit at one edge of a larger row.
    x_spans = [(0.0, 0.35), (0.0, 0.50), (0.0, 0.68), (0.12, 0.52), (0.28, 0.78), (0.48, 1.0)]
    y_spans = [(0.0, 0.45), (0.18, 0.68), (0.42, 1.0), (0.0, 1.0)]
    for xi, (xf1, xf2) in enumerate(x_spans):
        for yi, (yf1, yf2) in enumerate(y_spans):
            box = clamp_box(
                (
                    x1 + w * xf1,
                    y1 + h * yf1,
                    x1 + w * xf2,
                    y1 + h * yf2,
                ),
                image_width,
                image_height,
            )
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bw < 14 or bh < 10:
                continue
            score = 0.35 - 0.01 * (xi + yi)
            proposals.append(Proposal(box, "geometric_window", score))
    return proposals


def slot_regions(image_doc: Dict[str, Any], label_dir: Path, include_fallbacks: bool = True) -> List[BBox]:
    width = int(image_doc["image_width"])
    height = int(image_doc["image_height"])
    regions = list(likely_regions(image_doc, label_dir, include_fallbacks=include_fallbacks))
    if include_fallbacks:
        # General lower-left UI panel scans. The older fallback starts around
        # 72% height; some TV UIs place the channel row at the top of a lower
        # panel, so these bands start higher while remaining layout-agnostic.
        regions.extend(
            [
                (0.0, height * 0.60, width * 0.18, height * 0.78),
                (0.0, height * 0.62, width * 0.24, height * 0.84),
                (0.0, height * 0.66, width * 0.30, height * 0.88),
            ]
        )
    out: List[BBox] = []
    for region in regions:
        region = clamp_box(region, width, height)
        if box_area(region) <= 0:
            continue
        if any(iou(region, prev) > 0.82 for prev in out):
            continue
        out.append(region)
    return out


def foreground_mask(crop: Image.Image) -> np.ndarray:
    arr = np.array(crop.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Bright text, colored text, and strong local contrast are common channel
    # number styles. Combining them is more general than a provider-specific
    # color or position rule.
    bright = val > 135
    colored = (sat > 45) & (val > 55)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -6) > 0
    edges = cv2.Canny(gray, 60, 150) > 0
    mask = (bright | colored | adaptive | edges).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
    return mask


def component_boxes(crop: Image.Image) -> List[BBox]:
    mask = foreground_mask(crop)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    width, height = crop.size
    boxes: List[BBox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if area < 18:
            continue
        if h < max(5, height * 0.08) or h > height * 0.95:
            continue
        if w < 2 or w > width * 0.92:
            continue
        if w / max(1, h) > 8.0:
            continue
        boxes.append((float(x), float(y), float(x + w), float(y + h)))
    return sorted(boxes, key=lambda box: (box[1], box[0]))


def crop_visual_evidence(crop: Image.Image, expected_digits: int) -> Dict[str, Any]:
    """Measure visible glyph evidence without using OCR text or annotations."""
    gray = np.array(crop.convert("L"))
    if gray.size == 0:
        return {
            "crop_std": 0.0,
            "edge_density": 0.0,
            "foreground_ratio": 0.0,
            "connected_component_count": 0,
            "digit_like_component_count": 0,
            "component_height_consistency": 0.0,
            "border_touch_component_count": 0,
            "crop_aspect_per_digit": 0.0,
            "visible_text_evidence_score": 0.0,
        }

    height, width = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    expected = max(1, int(expected_digits))
    profiles: List[Dict[str, Any]] = []
    for threshold_mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, binary = cv2.threshold(blurred, 0, 255, threshold_mode | cv2.THRESH_OTSU)
        foreground_ratio = float(np.count_nonzero(binary)) / max(1.0, float(binary.size))
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        significant: List[Tuple[int, int, int, int]] = []
        digit_like: List[Tuple[int, int, int, int]] = []
        border_touch = 0
        for index in range(1, count):
            x, y, box_w, box_h, area = [int(value) for value in stats[index]]
            area_ratio = area / max(1.0, float(width * height))
            height_ratio = box_h / max(1.0, float(height))
            box_aspect = box_w / max(1.0, float(box_h))
            if area_ratio < 0.001 or height_ratio < 0.10 or box_w < 1:
                continue
            if box_w > width * 0.96 or box_h > height * 0.98 or box_aspect > 4.0:
                continue
            significant.append((x, y, box_w, box_h))
            if x <= 1 or y <= 1 or x + box_w >= width - 1 or y + box_h >= height - 1:
                border_touch += 1
            if 0.22 <= height_ratio <= 0.96 and 0.06 <= box_aspect <= 1.65 and area_ratio <= 0.65:
                digit_like.append((x, y, box_w, box_h))

        heights = [box[3] for box in digit_like]
        if len(heights) <= 1:
            consistency = 1.0 if heights else 0.0
        else:
            consistency = max(0.0, 1.0 - float(np.std(heights)) / max(1.0, float(np.mean(heights))))
        count_score = max(0.0, 1.0 - abs(len(digit_like) - expected) / max(2.0, float(expected + 1)))
        foreground_score = max(0.0, 1.0 - abs(foreground_ratio - 0.24) / 0.32)
        profile_score = 0.65 * count_score + 0.35 * foreground_score
        profiles.append(
            {
                "foreground_ratio": foreground_ratio,
                "connected_component_count": len(significant),
                "digit_like_component_count": len(digit_like),
                "component_height_consistency": consistency,
                "border_touch_component_count": border_touch,
                "digit_like_boxes": digit_like,
                "profile_score": profile_score,
            }
        )

    best = max(profiles, key=lambda item: float(item["profile_score"]))
    crop_std_value = float(np.std(gray))
    edge_density = float(np.count_nonzero(cv2.Canny(gray, 60, 150))) / max(1.0, float(gray.size))
    contrast_score = min(1.0, crop_std_value / 48.0)
    edge_score = min(1.0, edge_density / 0.10) * max(0.0, 1.0 - max(0.0, edge_density - 0.38) / 0.30)
    component_score = max(
        0.0,
        1.0 - abs(int(best["digit_like_component_count"]) - expected) / max(2.0, float(expected + 1)),
    )
    foreground_score = max(0.0, 1.0 - abs(float(best["foreground_ratio"]) - 0.24) / 0.32)
    border_penalty = min(0.25, 0.08 * int(best["border_touch_component_count"]))
    visible_score = (
        0.24 * contrast_score
        + 0.18 * edge_score
        + 0.30 * component_score
        + 0.14 * foreground_score
        + 0.14 * float(best["component_height_consistency"])
        - border_penalty
    )
    return {
        "crop_std": round(crop_std_value, 6),
        "edge_density": round(edge_density, 6),
        "foreground_ratio": round(float(best["foreground_ratio"]), 6),
        "connected_component_count": int(best["connected_component_count"]),
        "digit_like_component_count": int(best["digit_like_component_count"]),
        "component_height_consistency": round(float(best["component_height_consistency"]), 6),
        "border_touch_component_count": int(best["border_touch_component_count"]),
        "crop_aspect_per_digit": round((width / max(1.0, float(height))) / expected, 6),
        "visible_text_evidence_score": round(max(0.0, min(1.0, visible_score)), 6),
        "digit_like_boxes": list(best["digit_like_boxes"]),
    }


def history_one_digit_component_proposals(
    image: Image.Image,
    locked_slot_bbox: BBox,
    image_width: int,
    image_height: int,
) -> List[Proposal]:
    region = scale_box_around_center(locked_slot_bbox, image_width, image_height, 1.28, 1.24)
    x1, y1, x2, y2 = [int(round(value)) for value in region]
    if x2 <= x1 or y2 <= y1:
        return []
    crop = image.crop((x1, y1, x2, y2))
    evidence = crop_visual_evidence(crop, 1)
    digit_like_boxes = list(evidence.get("digit_like_boxes", []))
    # A multi-glyph line must not be split into several attractive one-digit
    # candidates. This path is only a conservative rescue for a visibly
    # single-glyph slot.
    if len(digit_like_boxes) != 1:
        return []
    proposals: List[Proposal] = []
    for box in digit_like_boxes:
        local_x, local_y, box_w, box_h = [float(value) for value in box]
        if box_h < 5 or box_w / max(1.0, box_h) > 1.65:
            continue
        pad_x = max(2.0, 0.18 * box_h)
        pad_y = max(2.0, 0.12 * box_h)
        global_box = clamp_box(
            (
                x1 + local_x - pad_x,
                y1 + local_y - pad_y,
                x1 + local_x + box_w + pad_x,
                y1 + local_y + box_h + pad_y,
            ),
            image_width,
            image_height,
        )
        proposals.append(Proposal(global_box, "history_one_digit_component", 0.84))
    return dedupe_proposals(proposals, iou_threshold=0.86)


def group_components_into_lines(boxes: Sequence[BBox]) -> List[List[BBox]]:
    lines: List[List[BBox]] = []
    for box in sorted(boxes, key=lambda item: ((item[1] + item[3]) / 2, item[0])):
        cy = (box[1] + box[3]) / 2
        bh = box[3] - box[1]
        placed = False
        for line in lines:
            line_cy = sum((b[1] + b[3]) / 2 for b in line) / len(line)
            line_h = sum(b[3] - b[1] for b in line) / len(line)
            if abs(cy - line_cy) <= max(8.0, 0.55 * max(bh, line_h)):
                line.append(box)
                placed = True
                break
        if not placed:
            lines.append([box])
    for line in lines:
        line.sort(key=lambda item: item[0])
    return lines


def span_box(parts: Sequence[BBox]) -> BBox:
    return (
        min(part[0] for part in parts),
        min(part[1] for part in parts),
        max(part[2] for part in parts),
        max(part[3] for part in parts),
    )


def component_span_proposals(region_crop: Image.Image, global_region: BBox, image_width: int, image_height: int) -> List[Proposal]:
    x0, y0, _, _ = global_region
    local_boxes = component_boxes(region_crop)
    proposals: List[Proposal] = []
    for line in group_components_into_lines(local_boxes):
        if not line:
            continue
        heights = [box[3] - box[1] for box in line]
        median_h = float(np.median(heights)) if heights else 12.0
        max_gap = max(8.0, median_h * 1.15)
        for start in range(len(line)):
            group = [line[start]]
            for end in range(start, min(len(line), start + 7)):
                if end > start:
                    prev = line[end - 1]
                    current = line[end]
                    gap = current[0] - prev[2]
                    if gap > max_gap:
                        break
                    group.append(current)
                local = span_box(group)
                lw, lh = local[2] - local[0], local[3] - local[1]
                if lw < 8 or lh < 8:
                    continue
                if lw / max(1.0, lh) > 7.0:
                    continue
                global_box = expand_box(
                    (local[0] + x0, local[1] + y0, local[2] + x0, local[3] + y0),
                    image_width,
                    image_height,
                    pad_x=max(3.0, 0.12 * lh),
                    pad_y=max(2.0, 0.10 * lh),
                )
                score = 0.62 + min(0.18, 0.02 * len(group))
                proposals.append(Proposal(global_box, "component_span", score))

        if len(line) > 1:
            local = span_box(line)
            lw, lh = local[2] - local[0], local[3] - local[1]
            if lw >= 10 and lh >= 8 and lw / max(1.0, lh) <= 9.0:
                proposals.append(
                    Proposal(
                        expand_box(
                            (local[0] + x0, local[1] + y0, local[2] + x0, local[3] + y0),
                            image_width,
                            image_height,
                            pad_x=max(4.0, 0.10 * lh),
                            pad_y=max(2.0, 0.10 * lh),
                        ),
                        "component_line",
                        0.50,
                    )
                )
    return proposals


def make_crop_variants(crop: Image.Image, raw_only: bool = False) -> List[Image.Image]:
    rgb = crop.convert("RGB")
    if raw_only:
        return [rgb]
    gray = ImageOps.grayscale(rgb).convert("RGB")
    contrast = ImageEnhance.Contrast(gray).enhance(1.8)
    sharp = ImageEnhance.Sharpness(rgb).enhance(1.7)
    variants = [rgb, contrast, sharp]
    if crop.width < 120 or crop.height < 36:
        variants.append(rgb.resize((crop.width * 2, crop.height * 2), Image.Resampling.BICUBIC))
    return variants


def recognition_hits(
    recognizer: ChannelDigitRecognizer,
    image: Image.Image,
    proposals: Sequence[Proposal],
    *,
    min_conf: float,
    max_digits: int,
    raw_only: bool,
) -> List[Dict[str, Any]]:
    crops: List[Image.Image] = []
    crop_meta: List[Tuple[Proposal, int, Image.Image]] = []
    for proposal in proposals:
        x1, y1, x2, y2 = [int(round(v)) for v in proposal.bbox]
        if x2 <= x1 or y2 <= y1:
            continue
        raw_crop = image.crop((x1, y1, x2, y2))
        for variant_index, variant in enumerate(make_crop_variants(raw_crop, raw_only=raw_only)):
            crops.append(variant)
            crop_meta.append((proposal, variant_index, raw_crop))

    hits: Dict[Tuple[str, Tuple[int, int, int, int]], Dict[str, Any]] = {}
    for (proposal, variant_index, raw_crop), predictions in zip(crop_meta, recognizer.predict_many(crops)):
        for value, conf in predictions:
            if conf < min_conf or not (1 <= len(value) <= max_digits):
                continue
            key_box = tuple(int(round(v)) for v in proposal.bbox)
            key = (value, key_box)
            prev = hits.get(key)
            if prev is not None and float(prev["ocr_conf"]) >= conf:
                continue
            is_repair = str(proposal.kind).startswith("repair_")
            visual_evidence = crop_visual_evidence(raw_crop, len(value))
            visual_evidence.pop("digit_like_boxes", None)
            hits[key] = {
                "text": value,
                "bbox_xyxy": [round(float(v), 3) for v in proposal.bbox],
                "ocr_conf": round(float(conf), 6),
                "detection_conf": round(float(proposal.score), 6),
                "source": "legacy_slot_repair" if is_repair else "paddleocr_slot_proposal_recheck",
                "raw_source": "legacy_slot_repair" if is_repair else "paddleocr_slot_proposal_recheck",
                "normalized_source": "legacy_slot_repair" if is_repair else "slot_proposal_recheck",
                "proposal_kind": proposal.kind,
                "proposal_score": round(float(proposal.score), 6),
                "variant_index": variant_index,
                **visual_evidence,
            }
            if is_repair:
                hits[key]["repair_source"] = "legacy_slot_repair"
                hits[key]["repair_type"] = proposal.kind.replace("repair_", "", 1)
                hits[key]["repair_score"] = round(float(proposal.score), 6)
    return sorted(hits.values(), key=lambda item: float(item["ocr_conf"]), reverse=True)


def repair_proposals_for_region(
    proposals: Sequence[Proposal],
    image_width: int,
    image_height: int,
    *,
    max_variants: int,
    padding_ratios: Sequence[float],
) -> List[Proposal]:
    repairs: List[Proposal] = []
    for proposal in proposals:
        x1, y1, x2, y2 = proposal.bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        aspect = w / h

        # Tight crops often miss the leading zero or clip vertically. Generate
        # conservative expansions without changing the legacy default path.
        if proposal.kind in {"component_span", "component_line"} or aspect < 4.8:
            for ratio in padding_ratios:
                repairs.append(
                    Proposal(
                        expand_box(proposal.bbox, image_width, image_height, max(3.0, w * ratio), max(2.0, h * ratio * 0.45)),
                        "repair_tight_expand",
                        min(0.86, proposal.score + 0.10 + ratio),
                    )
                )

        # Wide UI strips tend to include title text/background. Split them into
        # overlapping numeric-slot-like subwindows before OCR retry.
        if aspect >= 3.6 or proposal.kind in {"region_full", "geometric_window"}:
            spans = [(0.0, 0.38), (0.0, 0.50), (0.0, 0.62), (0.10, 0.55), (0.22, 0.72)]
            for left, right in spans:
                box = clamp_box((x1 + w * left, y1, x1 + w * right, y2), image_width, image_height)
                if box_area(box) <= 0:
                    continue
                repairs.append(Proposal(box, "repair_wide_split", min(0.84, proposal.score + 0.14)))

        # A light retry with slightly taller crops helps vertical clipping
        # without growing into neighboring text too aggressively.
        if h < image_height * 0.10:
            repairs.append(
                Proposal(
                    expand_box(proposal.bbox, image_width, image_height, max(2.0, w * 0.04), max(4.0, h * 0.30)),
                    "repair_vertical_expand",
                    min(0.82, proposal.score + 0.08),
                )
            )

    return dedupe_proposals(repairs)[:max_variants]


def proposals_for_region(
    region: BBox,
    image: Image.Image,
    *,
    max_proposals: int,
    enable_slot_repair: bool = False,
    slot_repair_mode: str = "off",
    slot_repair_max_variants: int = 10,
    slot_repair_padding_ratios: Sequence[float] = (0.08, 0.14, 0.22),
) -> List[Proposal]:
    width, height = image.size
    region = clamp_box(region, width, height)
    x1, y1, x2, y2 = [int(round(v)) for v in region]
    if x2 <= x1 or y2 <= y1:
        return []
    crop = image.crop((x1, y1, x2, y2))
    proposals = []
    proposals.extend(geometric_subwindows(region, width, height))
    proposals.extend(component_span_proposals(crop, region, width, height))
    proposals = dedupe_proposals(proposals)
    if enable_slot_repair and slot_repair_mode != "off":
        base_for_repair = proposals
        repairs = repair_proposals_for_region(
            base_for_repair,
            width,
            height,
            max_variants=slot_repair_max_variants,
            padding_ratios=slot_repair_padding_ratios,
        )
        if slot_repair_mode == "wide":
            repairs = [proposal for proposal in repairs if proposal.kind == "repair_wide_split"]
        elif slot_repair_mode == "tight":
            repairs = [proposal for proposal in repairs if proposal.kind != "repair_wide_split"]
        proposals = dedupe_proposals([*proposals, *repairs])
    return proposals[:max_proposals]


def image_identifier(image_doc: Dict[str, Any], fallback_index: int = 0) -> str:
    for key in ("image_id", "image_name", "filename"):
        value = image_doc.get(key)
        if value:
            return Path(str(value)).stem
    for key in ("image_path", "path"):
        value = image_doc.get(key)
        if value:
            return Path(str(value)).stem
    return f"image_{fallback_index:06d}"


def resolve_slot_image_path(image_doc: Dict[str, Any], images_dir: Optional[Path] = None) -> Path:
    raw_path = Path(str(image_doc.get("image_path", "") or image_doc.get("path", "")))
    resolved = resolve_image_path(raw_path)
    if resolved.exists():
        return resolved

    if images_dir is not None:
        image_id = image_identifier(image_doc)
        suffixes = [raw_path.suffix] if raw_path.suffix else []
        suffixes.extend([".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"])
        seen_suffixes: Set[str] = set()
        for suffix in suffixes:
            if not suffix or suffix in seen_suffixes:
                continue
            seen_suffixes.add(suffix)
            candidate = images_dir / f"{image_id}{suffix}"
            if candidate.exists():
                return candidate

    return resolved


def parse_padding_ratios(raw: str) -> Sequence[float]:
    ratios = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        ratios.append(float(part))
    return ratios or (0.08, 0.14, 0.22)


def center_distance_norm(a: BBox, b: BBox, width: int, height: int) -> float:
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return float((((acx - bcx) / max(1, width)) ** 2 + ((acy - bcy) / max(1, height)) ** 2) ** 0.5)


def scale_box_around_center(box: BBox, width: int, height: int, scale_x: float, scale_y: float) -> BBox:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w = max(1.0, (x2 - x1) * scale_x / 2.0)
    half_h = max(1.0, (y2 - y1) * scale_y / 2.0)
    return clamp_box((cx - half_w, cy - half_h, cx + half_w, cy + half_h), width, height)


def history_guided_proposals(
    locked_slot_bbox: BBox,
    image_width: int,
    image_height: int,
    *,
    max_crop_area_ratio: float = 0.18,
) -> List[Proposal]:
    base = clamp_box(locked_slot_bbox, image_width, image_height)
    x1, y1, x2, y2 = base
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    variants = [
        ("history_original", base, 0.88),
        ("history_pad_1p10", scale_box_around_center(base, image_width, image_height, 1.10, 1.10), 0.86),
        ("history_pad_1p25", scale_box_around_center(base, image_width, image_height, 1.25, 1.18), 0.84),
        (
            "history_left_leading_zero_pad",
            clamp_box((x1 - 0.35 * w, y1 - 0.10 * h, x2 + 0.12 * w, y2 + 0.10 * h), image_width, image_height),
            0.87,
        ),
        (
            "history_right_pad",
            clamp_box((x1 - 0.15 * w, y1 - 0.10 * h, x2 + 0.28 * w, y2 + 0.10 * h), image_width, image_height),
            0.82,
        ),
        (
            "history_vertical_pad",
            clamp_box((x1 - 0.10 * w, y1 - 0.30 * h, x2 + 0.10 * w, y2 + 0.30 * h), image_width, image_height),
            0.80,
        ),
    ]
    out: List[Proposal] = []
    for kind, box, score in variants:
        if box_area(box) <= 0:
            continue
        if box_area(box) / max(1.0, float(image_width * image_height)) > max_crop_area_ratio:
            continue
        out.append(Proposal(box, kind, score))
    return dedupe_proposals(out, iou_threshold=0.92)


def crop_std(image: Image.Image, bbox: BBox) -> float:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return 0.0
    gray = np.array(image.crop((x1, y1, x2, y2)).convert("L"))
    if gray.size == 0:
        return 0.0
    return float(np.std(gray))


def coerce_history_config(config: Any) -> HistoryGuidedSlotConfig:
    if isinstance(config, HistoryGuidedSlotConfig):
        return config
    if isinstance(config, dict):
        values = {
            field: config[field]
            for field in (
                "min_conf",
                "max_digits",
                "max_candidates",
                "min_crop_std",
                "max_crop_area_ratio",
                "raw_only",
                "one_digit_recovery",
                "one_digit_only",
            )
            if field in config
        }
        return HistoryGuidedSlotConfig(**values)
    return HistoryGuidedSlotConfig()


def run_history_guided_slot_recheck(
    image_path: Path | str,
    locked_slot_bbox: Sequence[float],
    recognizer: ChannelDigitRecognizer,
    config: Any = None,
    frame_id: str = "",
    history_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Generate OCR candidates from a previously locked slot bbox.

    This API is inference-only: it reads only the image and bbox state produced
    by model observations. It never reads GT/annotation or locks a previous
    channel-number text.
    """
    cfg = coerce_history_config(config)
    path = Path(image_path)
    if not path.exists():
        return []
    metadata = dict(history_metadata or {})
    bbox = tuple(float(v) for v in locked_slot_bbox)
    with Image.open(path) as src:
        image = src.convert("RGB")
        width, height = image.size
        proposals = [
            proposal
            for proposal in history_guided_proposals(
                bbox,
                width,
                height,
                max_crop_area_ratio=cfg.max_crop_area_ratio,
            )
            if crop_std(image, proposal.bbox) >= cfg.min_crop_std
        ]
        if cfg.one_digit_only:
            proposals = [
                proposal
                for proposal in proposals
                if proposal.kind in {"history_original", "history_pad_1p10"}
            ]
        if cfg.one_digit_recovery or cfg.one_digit_only:
            proposals.extend(history_one_digit_component_proposals(image, bbox, width, height))
            proposals = dedupe_proposals(proposals, iou_threshold=0.92)
        hits = recognition_hits(
            recognizer,
            image,
            proposals,
            min_conf=cfg.min_conf,
            max_digits=cfg.max_digits,
            raw_only=cfg.raw_only,
        )
        if cfg.one_digit_only:
            hits = [hit for hit in hits if len(str(hit.get("text", ""))) == 1]

    out: List[Dict[str, Any]] = []
    for index, hit in enumerate(hits[: cfg.max_candidates], 1):
        candidate = dict(hit)
        candidate_bbox = tuple(float(v) for v in candidate["bbox_xyxy"])
        history_iou = iou(candidate_bbox, bbox)
        history_distance = center_distance_norm(candidate_bbox, bbox, width, height)
        variant = str(candidate.get("proposal_kind", "history_guided"))
        score = float(candidate.get("proposal_score", candidate.get("detection_conf", 0.0)) or 0.0)
        history_score = float(metadata.get("history_score", 0.0) or 0.0)
        temporal_slot_boost = max(0.0, min(1.0, 0.55 * score + 0.45 * history_score))
        candidate.update(
            {
                "id": f"history_slot_{frame_id}_{index:03d}" if frame_id else f"history_slot_{index:03d}",
                "source": "slot_proposal_recheck",
                "raw_source": "history_guided_slot_recheck",
                "normalized_source": "slot_proposal_recheck",
                "normalized_text": "".join(ch for ch in str(candidate.get("text", "")) if ch.isdigit()),
                "score": round(max(float(candidate.get("ocr_conf", 0.0)), temporal_slot_boost), 6),
                "history_locked_bbox": [round(float(v), 3) for v in bbox],
                "history_state": str(metadata.get("state", "LOCKED")),
                "history_track_id": str(metadata.get("history_track_id", "")),
                "history_score": round(history_score, 6),
                "history_iou": round(history_iou, 6),
                "history_distance": round(history_distance, 6),
                "locked_slot_age": int(metadata.get("locked_slot_age", 0) or 0),
                "from_history_guided_slot": True,
                "from_temporal_locked_slot": 1.0,
                "history_recheck_variant": variant,
                "temporal_history_score": round(history_score, 6),
                "temporal_slot_iou": round(history_iou, 6),
                "temporal_slot_distance": round(history_distance, 6),
                "temporal_slot_boost": round(temporal_slot_boost, 6),
            }
        )
        candidate.setdefault("bbox", candidate.get("bbox_xyxy"))
        out.append(candidate)
    return out


def process_image_doc(
    image_doc: Dict[str, Any],
    *,
    image_index: int,
    recognizer: ChannelDigitRecognizer,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], int, str, str]:
    out_doc = copy.deepcopy(image_doc)
    original_candidates = list(out_doc.get("candidates") or [])
    image_path = resolve_slot_image_path(out_doc, args.images_dir)
    if not image_path.exists():
        if args.slot_only_output:
            out_doc["candidates"] = []
        return out_doc, 0, "missing_image", f"image not found: {image_path}"

    padding_ratios = parse_padding_ratios(args.slot_repair_padding_ratios)
    with Image.open(image_path) as src:
        rgb = src.convert("RGB")
        width, height = rgb.size
        out_doc["image_width"] = width
        out_doc["image_height"] = height
        out_doc["image_path"] = str(image_path)
        out_doc.setdefault("image_id", image_identifier(out_doc, image_index))

        image_hits: List[Dict[str, Any]] = []
        for region_index, region in enumerate(
            slot_regions(out_doc, args.yolo_label_dir, include_fallbacks=not args.yolo_only),
            1,
        ):
            proposals = proposals_for_region(
                region,
                rgb,
                max_proposals=args.max_proposals_per_region,
                enable_slot_repair=args.enable_slot_repair,
                slot_repair_mode=args.slot_repair_mode,
                slot_repair_max_variants=args.slot_repair_max_variants_per_region,
                slot_repair_padding_ratios=padding_ratios,
            )
            for hit_index, hit in enumerate(
                recognition_hits(
                    recognizer,
                    rgb,
                    proposals,
                    min_conf=args.min_conf,
                    max_digits=args.max_digits,
                    raw_only=args.raw_only,
                ),
                1,
            ):
                hit["id"] = f"slot_prop_{region_index:02d}_{hit_index:03d}"
                hit["parent_region_index"] = region_index
                image_hits.append(hit)

    image_hits = sorted(image_hits, key=lambda item: float(item["ocr_conf"]), reverse=True)
    slot_hits = image_hits[: args.max_candidates_per_image]
    if args.slot_only_output:
        out_doc["candidates"] = slot_hits
    else:
        out_doc["candidates"] = original_candidates
        for hit in slot_hits:
            out_doc.setdefault("candidates", []).append(hit)
    return out_doc, len(slot_hits), "ok", ""


def chunk_bounds(total: int, start: int, end: Optional[int], max_images: Optional[int], chunk_size: int) -> List[Tuple[int, int]]:
    start = max(0, start)
    stop = total if end is None else min(total, max(start, end))
    if max_images is not None and max_images > 0:
        stop = min(stop, start + max_images)
    if chunk_size <= 0:
        return [(start, stop)]
    return [(idx, min(stop, idx + chunk_size)) for idx in range(start, stop, chunk_size)]


def read_completed_indices(jsonl_path: Path) -> Set[int]:
    completed: Set[int] = set()
    if not jsonl_path.exists():
        return completed
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            index = record.get("image_index")
            if isinstance(index, int):
                completed.add(index)
    return completed


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_progress_state(
    path: Path,
    *,
    total_images: int,
    processed_count: int,
    added_count: int,
    current_chunk: str,
    completed_chunks: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "total_images": total_images,
                "processed_count": processed_count,
                "added_count": added_count,
                "current_chunk": current_chunk,
                "completed_chunks": list(completed_chunks),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def merge_partial_jsonl(doc: Dict[str, Any], partial_output_dir: Path, out_path: Path) -> Dict[str, Any]:
    images = doc.get("images", [])
    by_index: Dict[int, Dict[str, Any]] = {}
    records = 0
    added = 0
    status_counts: Dict[str, int] = {}
    for path in sorted(partial_output_dir.glob("chunk_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                index = record.get("image_index")
                if not isinstance(index, int):
                    continue
                status = str(record.get("status") or "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
                records += 1
                image = record.get("image")
                if isinstance(image, dict):
                    by_index[index] = image
                    added += len(image.get("candidates") or [])

    merged_images: List[Dict[str, Any]] = []
    for index, image_doc in enumerate(images):
        if index in by_index:
            merged_images.append(by_index[index])
        else:
            empty = copy.deepcopy(image_doc)
            empty["candidates"] = []
            merged_images.append(empty)

    merged = copy.deepcopy(doc)
    merged["images"] = merged_images
    merged.setdefault("metadata", {})
    merged["metadata"].update(
        {
            "slot_proposal_partial_output_dir": str(partial_output_dir),
            "slot_proposal_partial_record_count": records,
            "slot_proposal_added_candidate_count": added,
            "slot_proposal_status_counts": status_counts,
            "source": "slot_proposal_numeric_recheck_chunk_merge",
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "image_count": len(merged_images),
        "partial_record_count": records,
        "slot_candidate_count": added,
        "status_counts": status_counts,
        "out": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--yolo-label-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=Path("runs/ocr/inference"))
    parser.add_argument("--model-name", default="PP-OCRv5_mobile_rec")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--input-shape", default="3,48,320")
    parser.add_argument("--min-conf", type=float, default=0.90)
    parser.add_argument("--max-digits", type=int, default=5)
    parser.add_argument("--max-proposals-per-region", type=int, default=40)
    parser.add_argument("--max-candidates-per-image", type=int, default=35)
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Run recognition only on raw proposal crops, without contrast/scale variants.",
    )
    parser.add_argument(
        "--yolo-only",
        action="store_true",
        help="Only split YOLO class 0/3 regions; skip heuristic fallback regions.",
    )
    parser.add_argument(
        "--slot-proposal-mode",
        choices=["legacy", "cc_textline", "hybrid"],
        default="legacy",
        help="Use legacy behavior by default; cc_textline/hybrid delegates to slot_proposal_numeric_recheck_v2.",
    )
    parser.add_argument("--slot-v2-debug-dir", type=Path, default=None)
    parser.add_argument("--slot-v2-max-proposals-per-region", type=int, default=20)
    parser.add_argument("--slot-v2-min-component-height", type=float, default=5.0)
    parser.add_argument("--slot-v2-max-component-height-ratio", type=float, default=0.72)
    parser.add_argument("--slot-v2-save-debug-images", action="store_true")
    parser.add_argument("--enable-slot-repair", action="store_true")
    parser.add_argument("--slot-repair-mode", choices=["off", "wide", "tight", "both"], default="off")
    parser.add_argument("--slot-repair-debug-dir", type=Path, default=None)
    parser.add_argument("--slot-repair-max-variants-per-region", type=int, default=10)
    parser.add_argument("--slot-repair-min-component-height", type=float, default=5.0)
    parser.add_argument("--slot-repair-padding-ratios", default="0.08,0.14,0.22")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0, help="0-based inclusive start image index.")
    parser.add_argument("--end-index", type=int, default=None, help="0-based exclusive end image index.")
    parser.add_argument("--chunk-size", type=int, default=0, help="Process images in chunk files of this size.")
    parser.add_argument("--resume", action="store_true", help="Skip image records already present in partial JSONL files.")
    parser.add_argument("--checkpoint-jsonl", type=Path, default=None, help="Append every processed image record to this JSONL checkpoint.")
    parser.add_argument("--partial-output-dir", type=Path, default=None, help="Directory for chunk_XXXX_YYYY.jsonl partial outputs.")
    parser.add_argument("--merge-partials", action="store_true", help="Merge partial JSONL outputs into --out and exit.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional maximum image count after --start-index.")
    parser.add_argument(
        "--slot-only-output",
        action="store_true",
        help="Write only slot proposal candidates, omitting candidates already present in --ocr-json.",
    )
    args = parser.parse_args()

    if args.slot_proposal_mode != "legacy":
        from slot_proposal_numeric_recheck_v2 import run_cli_from_namespace

        run_cli_from_namespace(args)
        return

    doc = json.loads(args.ocr_json.read_text(encoding="utf-8"))
    images = doc.get("images", [])

    if args.merge_partials:
        if args.partial_output_dir is None:
            raise SystemExit("--partial-output-dir is required with --merge-partials.")
        summary = merge_partial_jsonl(doc, args.partial_output_dir, args.out)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.yolo_label_dir is None:
        raise SystemExit("--yolo-label-dir is required unless --merge-partials is used.")

    recognizer = ChannelDigitRecognizer(
        args.model_dir,
        model_name=args.model_name,
        device=args.device,
        input_shape=args.input_shape,
    )

    partial_mode = args.chunk_size > 0 or args.partial_output_dir is not None or args.checkpoint_jsonl is not None
    if partial_mode:
        partial_output_dir = args.partial_output_dir or args.out.with_suffix("").parent / f"{args.out.stem}_chunks"
        checkpoint_jsonl = args.checkpoint_jsonl or partial_output_dir / "checkpoint.jsonl"
        progress_state = partial_output_dir / "progress_state.json"
        completed_from_checkpoint = read_completed_indices(checkpoint_jsonl) if args.resume else set()
        total_added = 0
        total_processed = 0
        completed_chunks: List[str] = []
        bounds = chunk_bounds(len(images), args.start_index, args.end_index, args.max_images, args.chunk_size)

        for chunk_start, chunk_end in bounds:
            if chunk_end <= chunk_start:
                continue
            chunk_name = f"chunk_{chunk_start:04d}_{chunk_end - 1:04d}.jsonl"
            chunk_path = partial_output_dir / chunk_name
            expected_indices = set(range(chunk_start, chunk_end))
            completed_in_chunk = read_completed_indices(chunk_path) if args.resume else set()
            completed = completed_from_checkpoint | completed_in_chunk
            if args.resume and expected_indices.issubset(completed):
                completed_chunks.append(chunk_name)
                print(f"skip completed {chunk_name}", flush=True)
                continue

            for image_index in range(chunk_start, chunk_end):
                if args.resume and image_index in completed:
                    continue
                image_doc = images[image_index]
                image_id = image_identifier(image_doc, image_index)
                try:
                    processed_doc, added, status, error = process_image_doc(
                        image_doc,
                        image_index=image_index,
                        recognizer=recognizer,
                        args=args,
                    )
                except Exception as exc:  # Keep long external runs resumable.
                    processed_doc = copy.deepcopy(image_doc)
                    if args.slot_only_output:
                        processed_doc["candidates"] = []
                    added = 0
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"

                record = {
                    "image_index": image_index,
                    "image_id": image_id,
                    "status": status,
                    "added_candidate_count": added,
                    "error": error,
                    "image": processed_doc,
                }
                append_jsonl(chunk_path, record)
                append_jsonl(checkpoint_jsonl, record)
                total_added += added
                total_processed += 1

                if args.progress_every and (
                    total_processed == 1
                    or total_processed % args.progress_every == 0
                    or image_index == chunk_end - 1
                ):
                    print(
                        f"progress processed={total_processed} image_index={image_index} chunk={chunk_name} added={total_added}",
                        flush=True,
                    )
                    write_progress_state(
                        progress_state,
                        total_images=len(images),
                        processed_count=total_processed,
                        added_count=total_added,
                        current_chunk=chunk_name,
                        completed_chunks=completed_chunks,
                    )

            completed_chunks.append(chunk_name)
            write_progress_state(
                progress_state,
                total_images=len(images),
                processed_count=total_processed,
                added_count=total_added,
                current_chunk=chunk_name,
                completed_chunks=completed_chunks,
            )

        print(
            f"wrote partial slot proposal chunks to {partial_output_dir} "
            f"processed={total_processed} added={total_added} checkpoint={checkpoint_jsonl}",
            flush=True,
        )
        return

    added = 0
    bounds = chunk_bounds(len(images), args.start_index, args.end_index, args.max_images, 0)
    start, stop = bounds[0] if bounds else (0, 0)
    for display_index, image_index in enumerate(range(start, stop), 1):
        image_doc = images[image_index]
        processed_doc, image_added, _status, _error = process_image_doc(
            image_doc,
            image_index=image_index,
            recognizer=recognizer,
            args=args,
        )
        images[image_index] = processed_doc
        added += image_added

        if args.progress_every and (
            display_index == 1 or display_index % args.progress_every == 0 or image_index == stop - 1
        ):
            print(f"progress {display_index}/{stop - start} added={added}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} with {added} slot proposal channel candidates")


if __name__ == "__main__":
    main()
