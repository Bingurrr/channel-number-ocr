"""Connected-component/textline slot proposal recheck for channel digits.

This runner is intentionally independent from the legacy
``slot_proposal_numeric_recheck.py`` path.  It keeps inference image-only:
YOLO boxes, existing OCR candidate boxes, and layout fallbacks are used for
broad regions, while GT/annotation JSON is never read.

The heavy image/OCR dependencies are imported lazily so ``--help`` and the
synthetic ``--self-test`` can run in a lightweight Python environment.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


BBox = Tuple[float, float, float, float]
SOURCE_CC_TEXTLINE = "slot_proposal_v2_cc_textline"
SOURCE_LEGACY = "paddleocr_slot_proposal_recheck"
DIGIT_RE = re.compile(r"[0-9]+")


def digits(text: Any) -> str:
    return "".join(ch for ch in str(text) if "0" <= ch <= "9")


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def clamp_box(box: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = box
    return (
        clamp(float(x1), 0.0, float(width)),
        clamp(float(y1), 0.0, float(height)),
        clamp(float(x2), 0.0, float(width)),
        clamp(float(y2), 0.0, float(height)),
    )


def box_area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    denom = box_area(a) + box_area(b) - inter
    return 0.0 if denom <= 0 else inter / denom


def expand_box(box: BBox, width: int, height: int, pad_x: float, pad_y: float) -> BBox:
    return clamp_box((box[0] - pad_x, box[1] - pad_y, box[2] + pad_x, box[3] + pad_y), width, height)


def span_box(parts: Sequence[BBox]) -> BBox:
    return (
        min(part[0] for part in parts),
        min(part[1] for part in parts),
        max(part[2] for part in parts),
        max(part[3] for part in parts),
    )


def median(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return default
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def coeff_var(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


@dataclass
class GrayImage:
    width: int
    height: int
    pixels: List[int]

    @classmethod
    def blank(cls, width: int, height: int, value: int = 0) -> "GrayImage":
        return cls(width, height, [int(value)] * (width * height))

    @classmethod
    def from_pil(cls, image: Any) -> "GrayImage":
        gray = image.convert("L")
        return cls(gray.width, gray.height, list(gray.getdata()))

    def get(self, x: int, y: int) -> int:
        return self.pixels[y * self.width + x]

    def set(self, x: int, y: int, value: int) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y * self.width + x] = int(clamp(value, 0, 255))

    def fill_rect(self, box: BBox, value: int) -> None:
        x1, y1, x2, y2 = [int(round(v)) for v in clamp_box(box, self.width, self.height)]
        for y in range(y1, y2):
            row = y * self.width
            for x in range(x1, x2):
                self.pixels[row + x] = int(clamp(value, 0, 255))

    def crop(self, box: BBox) -> "GrayImage":
        x1, y1, x2, y2 = [int(round(v)) for v in clamp_box(box, self.width, self.height)]
        out_w = max(0, x2 - x1)
        out_h = max(0, y2 - y1)
        out: List[int] = []
        for y in range(y1, y2):
            start = y * self.width + x1
            out.extend(self.pixels[start : start + out_w])
        return GrayImage(out_w, out_h, out)

    def save_pgm(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            handle.write(f"P5\n{self.width} {self.height}\n255\n".encode("ascii"))
            handle.write(bytes(max(0, min(255, int(value))) for value in self.pixels))


@dataclass(frozen=True)
class Component:
    bbox: BBox
    pixel_count: int
    polarity: str

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def density(self) -> float:
        area = box_area(self.bbox)
        return 0.0 if area <= 0 else self.pixel_count / area


@dataclass(frozen=True)
class BroadRegion:
    bbox: BBox
    source: str


@dataclass(frozen=True)
class Proposal:
    bbox: BBox
    kind: str
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def bbox_from_mapping(mapping: Mapping[str, Any]) -> Optional[BBox]:
    value = first_present(mapping, ("bbox_xyxy", "bbox", "box", "xyxy", "full_bbox"))
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            parts = [part for part in re.split(r"[,;\s]+", value.strip()) if part]
            value = parts
    if isinstance(value, Mapping):
        if all(key in value for key in ("x1", "y1", "x2", "y2")):
            value = [value["x1"], value["y1"], value["x2"], value["y2"]]
        elif all(key in value for key in ("left", "top", "right", "bottom")):
            value = [value["left"], value["top"], value["right"], value["bottom"]]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return None
    if len(value) < 4:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    except (TypeError, ValueError):
        return None


def resolve_image_path(image_doc: Mapping[str, Any], images_dir: Optional[Path]) -> Path:
    image_id = str(first_present(image_doc, ("image_id", "image_name", "filename", "file_name")) or "")
    raw_path = str(first_present(image_doc, ("image_path", "path", "file_path", "filename", "file_name")) or "")
    candidates: List[Path] = []
    if raw_path:
        raw = Path(raw_path.replace("\\", "/"))
        candidates.extend([raw, Path.cwd() / raw])
        if images_dir is not None:
            candidates.append(images_dir / raw.name)
    if image_id and images_dir is not None:
        base = Path(image_id).stem
        candidates.extend(
            [
                images_dir / image_id,
                images_dir / f"{base}.jpg",
                images_dir / f"{base}.jpeg",
                images_dir / f"{base}.png",
                images_dir / f"{base}.bmp",
                images_dir / f"{base}.pgm",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else Path(image_id)


def load_gray_and_pil(path: Path) -> Tuple[GrayImage, Any]:
    if path.suffix.lower() == ".pgm":
        return load_pgm(path), None
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to read non-PGM images. Install/use the project Python environment "
            "for Airtel runs, or use --self-test for the built-in synthetic checks."
        ) from exc
    with Image.open(path) as src:
        rgb = src.convert("RGB")
    return GrayImage.from_pil(rgb), rgb


def load_pgm(path: Path) -> GrayImage:
    data = path.read_bytes()
    if not data.startswith(b"P5"):
        raise ValueError(f"unsupported PGM format: {path}")
    index = 2
    tokens: List[bytes] = []
    while len(tokens) < 3:
        while index < len(data) and data[index] in b" \t\r\n":
            index += 1
        if index < len(data) and data[index] == ord("#"):
            while index < len(data) and data[index] not in b"\r\n":
                index += 1
            continue
        start = index
        while index < len(data) and data[index] not in b" \t\r\n":
            index += 1
        tokens.append(data[start:index])
    width, height, max_value = [int(token) for token in tokens]
    while index < len(data) and data[index] in b" \t\r\n":
        index += 1
    pixels = list(data[index : index + width * height])
    if max_value != 255:
        pixels = [round(value * 255 / max_value) for value in pixels]
    return GrayImage(width, height, pixels)


def otsu_threshold(pixels: Sequence[int]) -> int:
    if not pixels:
        return 127
    hist = [0] * 256
    for value in pixels:
        hist[max(0, min(255, int(value)))] += 1
    total = len(pixels)
    sum_total = sum(index * count for index, count in enumerate(hist))
    sum_background = 0.0
    weight_background = 0
    best_threshold = 127
    best_var = -1.0
    for threshold, count in enumerate(hist):
        weight_background += count
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += threshold * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if between > best_var:
            best_var = between
            best_threshold = threshold
    return best_threshold


def contrast_stretch(image: GrayImage) -> GrayImage:
    if not image.pixels:
        return image
    ordered = sorted(image.pixels)
    lo = ordered[int(len(ordered) * 0.05)]
    hi = ordered[int(len(ordered) * 0.95)]
    if hi <= lo + 4:
        return image
    scale = 255.0 / (hi - lo)
    return GrayImage(image.width, image.height, [int(clamp((value - lo) * scale, 0, 255)) for value in image.pixels])


def binary_mask(image: GrayImage, polarity: str) -> Tuple[List[int], GrayImage, int]:
    stretched = contrast_stretch(image)
    threshold = otsu_threshold(stretched.pixels)
    if stretched.pixels:
        lo = min(stretched.pixels)
        hi = max(stretched.pixels)
        if hi > lo and (threshold <= lo + 2 or threshold >= hi - 2):
            threshold = int((lo + hi) / 2)
    if polarity == "light":
        mask = [1 if value >= threshold - 4 else 0 for value in stretched.pixels]
    else:
        mask = [1 if value <= threshold + 4 else 0 for value in stretched.pixels]
    debug = GrayImage(image.width, image.height, [255 if value else 0 for value in mask])
    return mask, debug, threshold


def connected_components(
    mask: Sequence[int],
    width: int,
    height: int,
    *,
    polarity: str,
    min_component_height: float,
    max_component_height_ratio: float,
) -> List[Component]:
    visited = bytearray(width * height)
    components: List[Component] = []
    for start in range(width * height):
        if visited[start] or not mask[start]:
            continue
        stack = [start]
        visited[start] = 1
        min_x = max_x = start % width
        min_y = max_y = start // width
        count = 0
        while stack:
            index = stack.pop()
            count += 1
            x = index % width
            y = index // width
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                next_index = ny * width + nx
                if not visited[next_index] and mask[next_index]:
                    visited[next_index] = 1
                    stack.append(next_index)

        box = (float(min_x), float(min_y), float(max_x + 1), float(max_y + 1))
        comp = Component(box, count, polarity)
        if keep_component(comp, width, height, min_component_height, max_component_height_ratio):
            components.append(comp)
    return sorted(components, key=lambda item: (item.center_y, item.bbox[0]))


def keep_component(comp: Component, width: int, height: int, min_height: float, max_height_ratio: float) -> bool:
    bw, bh = comp.width, comp.height
    if bw < 2 or bh < max(3.0, min_height):
        return False
    if bh > max(8.0, height * max_height_ratio):
        return False
    if bw > width * 0.92:
        return False
    if box_area(comp.bbox) < 10:
        return False
    aspect = bw / max(1.0, bh)
    if aspect > 8.0:
        return False
    if aspect > 5.0 and bh < max(5.0, height * 0.16):
        return False
    if comp.density < 0.08:
        return False
    if box_area(comp.bbox) > width * height * 0.42:
        return False
    return True


def group_components_into_lines(components: Sequence[Component]) -> List[List[Component]]:
    lines: List[List[Component]] = []
    for comp in sorted(components, key=lambda item: (item.center_y, item.bbox[0])):
        placed = False
        for line in lines:
            line_cy = sum(item.center_y for item in line) / len(line)
            line_h = sum(item.height for item in line) / len(line)
            if abs(comp.center_y - line_cy) <= max(5.0, 0.58 * max(comp.height, line_h)):
                line.append(comp)
                placed = True
                break
        if not placed:
            lines.append([comp])
    for line in lines:
        line.sort(key=lambda item: item.bbox[0])
    return lines


def split_line_runs(line: Sequence[Component]) -> List[List[Component]]:
    if not line:
        return []
    heights = [item.height for item in line]
    med_h = median(heights, 12.0)
    max_gap = max(6.0, med_h * 1.10)
    runs: List[List[Component]] = [[line[0]]]
    for prev, current in zip(line, line[1:]):
        gap = current.bbox[0] - prev.bbox[2]
        if gap > max_gap:
            runs.append([current])
        else:
            runs[-1].append(current)
    return runs


def textline_proposal_score(group: Sequence[Component], local_box: BBox, crop_width: int, crop_height: int) -> float:
    heights = [item.height for item in group]
    widths = [item.width for item in group]
    h_cv = coeff_var(heights)
    w_cv = coeff_var(widths)
    bw = local_box[2] - local_box[0]
    bh = local_box[3] - local_box[1]
    aspect = bw / max(1.0, bh)
    expected_aspect = 0.78 * len(group) + 0.45
    wide_excess = max(0.0, aspect - expected_aspect)
    density = sum(item.pixel_count for item in group) / max(1.0, box_area(local_box))

    score = 0.72
    score += min(0.13, 0.022 * len(group))
    score -= min(0.20, 0.28 * max(0.0, h_cv - 0.18))
    score -= min(0.18, 0.22 * max(0.0, w_cv - 0.35))
    score -= min(0.22, 0.08 * wide_excess)
    if aspect > 6.2:
        score -= 0.16
    if bh > crop_height * 0.72 or bw > crop_width * 0.86:
        score -= 0.13
    if density < 0.12:
        score -= 0.10
    if density > 0.84:
        score -= 0.04
    return clamp(score, 0.05, 0.96)


def dedupe_proposals(proposals: Iterable[Proposal], iou_threshold: float = 0.78) -> List[Proposal]:
    out: List[Proposal] = []
    for proposal in sorted(proposals, key=lambda item: (item.score, -box_area(item.bbox)), reverse=True):
        if box_area(proposal.bbox) <= 0:
            continue
        if any(iou(proposal.bbox, kept.bbox) > iou_threshold and proposal.source == kept.source for kept in out):
            continue
        out.append(proposal)
    return out


def cc_textline_proposals_for_region(
    image: GrayImage,
    region: BroadRegion,
    *,
    max_proposals: int,
    min_component_height: float,
    max_component_height_ratio: float,
    debug_dir: Optional[Path] = None,
    debug_prefix: str = "",
) -> List[Proposal]:
    region_box = clamp_box(region.bbox, image.width, image.height)
    x0, y0, x2, y2 = [int(round(v)) for v in region_box]
    if x2 <= x0 or y2 <= y0:
        return []
    crop = image.crop(region_box)
    proposals: List[Proposal] = []
    debug_payload: Dict[str, Any] = {
        "broad_region_source": region.source,
        "broad_region_bbox": list(region_box),
        "polarity": {},
    }

    if debug_dir is not None:
        crop.save_pgm(debug_dir / f"{debug_prefix}_broad.pgm")

    for polarity in ("light", "dark"):
        mask, mask_debug, threshold = binary_mask(crop, polarity)
        components = connected_components(
            mask,
            crop.width,
            crop.height,
            polarity=polarity,
            min_component_height=min_component_height,
            max_component_height_ratio=max_component_height_ratio,
        )
        debug_payload["polarity"][polarity] = {
            "threshold": threshold,
            "component_count": len(components),
            "components": [list(comp.bbox) for comp in components],
        }
        if debug_dir is not None:
            mask_debug.save_pgm(debug_dir / f"{debug_prefix}_{polarity}_threshold.pgm")
            write_boxes_svg(
                debug_dir / f"{debug_prefix}_{polarity}_components.svg",
                crop.width,
                crop.height,
                [(comp.bbox, "#e4572e", f"{idx}") for idx, comp in enumerate(components, 1)],
            )

        for line_index, line in enumerate(group_components_into_lines(components), 1):
            for run_index, run in enumerate(split_line_runs(line), 1):
                if not run:
                    continue
                max_group_len = min(6, len(run))
                groups: List[Sequence[Component]] = []
                if 1 <= len(run) <= 6:
                    groups.append(run)
                for start in range(len(run)):
                    for end in range(start + 1, min(len(run), start + max_group_len) + 1):
                        group = run[start:end]
                        if group not in groups:
                            groups.append(group)

                for group_index, group in enumerate(groups, 1):
                    local = span_box([comp.bbox for comp in group])
                    lw, lh = local[2] - local[0], local[3] - local[1]
                    if lw < 5 or lh < 5:
                        continue
                    if len(group) > 6:
                        continue
                    score = textline_proposal_score(group, local, crop.width, crop.height)
                    if score < 0.40:
                        continue
                    pad_x = max(3.0, 0.20 * lh)
                    pad_y = max(2.0, 0.14 * lh)
                    # Asymmetric left padding protects leading zero strokes.
                    global_box = expand_box(
                        (local[0] + x0, local[1] + y0, local[2] + x0, local[3] + y0),
                        image.width,
                        image.height,
                        pad_x=pad_x,
                        pad_y=pad_y,
                    )
                    if global_box[0] > local[0] + x0 - pad_x + 0.5:
                        global_box = (max(0.0, global_box[0] - 2.0), global_box[1], global_box[2], global_box[3])
                    proposal_id = f"{polarity}_l{line_index:02d}_r{run_index:02d}_g{group_index:02d}"
                    proposals.append(
                        Proposal(
                            bbox=global_box,
                            kind="cc_textline",
                            score=score,
                            source=SOURCE_CC_TEXTLINE,
                            metadata={
                                "raw_source": SOURCE_CC_TEXTLINE,
                                "broad_region_source": region.source,
                                "broad_region_bbox": [round(float(v), 3) for v in region_box],
                                "local_bbox": [round(float(v), 3) for v in local],
                                "full_bbox": [round(float(v), 3) for v in global_box],
                                "component_count": len(group),
                                "textline_group_id": proposal_id,
                                "polarity": polarity,
                                "threshold": threshold,
                                "proposal_score": round(float(score), 6),
                            },
                        )
                    )

    proposals = dedupe_proposals(proposals)[:max_proposals]
    if debug_dir is not None:
        write_boxes_svg(
            debug_dir / f"{debug_prefix}_proposals.svg",
            image.width,
            image.height,
            [(proposal.bbox, "#0b72b9", f"{idx}:{proposal.score:.2f}") for idx, proposal in enumerate(proposals, 1)],
        )
        debug_payload["proposals"] = [
            {"bbox": list(proposal.bbox), "score": proposal.score, "metadata": proposal.metadata}
            for proposal in proposals
        ]
        (debug_dir / f"{debug_prefix}_metadata.json").write_text(
            json.dumps(debug_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return proposals


def geometric_subwindows(region: BroadRegion, image_width: int, image_height: int) -> List[Proposal]:
    x1, y1, x2, y2 = region.bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    proposals = [
        Proposal(
            clamp_box(region.bbox, image_width, image_height),
            "legacy_region_full",
            0.38,
            SOURCE_LEGACY,
            {"broad_region_source": region.source, "broad_region_bbox": list(region.bbox)},
        )
    ]
    x_spans = [(0.0, 0.35), (0.0, 0.50), (0.0, 0.68), (0.12, 0.52), (0.28, 0.78), (0.48, 1.0)]
    y_spans = [(0.0, 0.45), (0.18, 0.68), (0.42, 1.0), (0.0, 1.0)]
    for xi, (xf1, xf2) in enumerate(x_spans):
        for yi, (yf1, yf2) in enumerate(y_spans):
            box = clamp_box((x1 + w * xf1, y1 + h * yf1, x1 + w * xf2, y1 + h * yf2), image_width, image_height)
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bw < 14 or bh < 10:
                continue
            proposals.append(
                Proposal(
                    box,
                    "legacy_geometric_window",
                    0.35 - 0.01 * (xi + yi),
                    SOURCE_LEGACY,
                    {"broad_region_source": region.source, "broad_region_bbox": list(region.bbox)},
                )
            )
    return proposals


def proposals_for_region(
    image: GrayImage,
    region: BroadRegion,
    *,
    mode: str,
    max_proposals: int,
    min_component_height: float,
    max_component_height_ratio: float,
    debug_dir: Optional[Path] = None,
    debug_prefix: str = "",
) -> List[Proposal]:
    proposals: List[Proposal] = []
    if mode in ("legacy", "hybrid"):
        proposals.extend(geometric_subwindows(region, image.width, image.height))
    if mode in ("cc_textline", "hybrid"):
        proposals.extend(
            cc_textline_proposals_for_region(
                image,
                region,
                max_proposals=max_proposals,
                min_component_height=min_component_height,
                max_component_height_ratio=max_component_height_ratio,
                debug_dir=debug_dir,
                debug_prefix=debug_prefix,
            )
        )
    return dedupe_proposals(proposals)[:max_proposals]


def yolo_regions(image_doc: Mapping[str, Any], label_dir: Optional[Path], width: int, height: int) -> List[BroadRegion]:
    if label_dir is None:
        return []
    image_id = str(first_present(image_doc, ("image_id", "image_name", "filename", "file_name")) or "")
    stem = Path(image_id).stem
    path = label_dir / f"{stem}.txt"
    if not path.exists():
        return []
    regions: List[BroadRegion] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, bw, bh = [float(value) for value in parts[1:5]]
        except ValueError:
            continue
        if cls not in (0, 3):
            continue
        box = ((cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height)
        if cls == 0:
            pad_x = max(10.0, (box[2] - box[0]) * 0.90)
            pad_y = max(8.0, (box[3] - box[1]) * 0.95)
            source = "yolo_channel_number"
        else:
            pad_x = max(6.0, (box[2] - box[0]) * 0.08)
            pad_y = max(5.0, (box[3] - box[1]) * 0.10)
            source = "yolo_channel_number_area"
        regions.append(BroadRegion(expand_box(box, width, height, pad_x, pad_y), source))
    return regions


def ocr_expansion_regions(image_doc: Mapping[str, Any], width: int, height: int) -> List[BroadRegion]:
    regions: List[BroadRegion] = []
    for candidate in image_doc.get("candidates", []) or []:
        if not isinstance(candidate, Mapping):
            continue
        box = bbox_from_mapping(candidate)
        if box is None or box_area(box) <= 0:
            continue
        text = str(first_present(candidate, ("text", "value", "pred_text", "number")) or "")
        source = str(first_present(candidate, ("source", "candidate_source")) or "ocr")
        bw, bh = box[2] - box[0], box[3] - box[1]
        if digits(text) or source in {"original_ocr", "easyocr", "paddleocr"}:
            regions.append(
                BroadRegion(
                    expand_box(box, width, height, pad_x=max(12.0, bw * 1.20), pad_y=max(8.0, bh * 0.85)),
                    "ocr_expanded",
                )
            )
    return regions


def fallback_regions(width: int, height: int) -> List[BroadRegion]:
    return [
        BroadRegion((0.0, height * 0.58, width * 0.20, height * 0.80), "fallback_lower_left_tight"),
        BroadRegion((0.0, height * 0.60, width * 0.28, height * 0.86), "fallback_lower_left_mid"),
        BroadRegion((0.0, height * 0.64, width * 0.36, height * 0.91), "fallback_lower_left_wide"),
        BroadRegion((0.0, height * 0.48, width * 0.30, height * 0.72), "fallback_mid_left"),
    ]


def broad_regions(
    image_doc: Mapping[str, Any],
    *,
    label_dir: Optional[Path],
    width: int,
    height: int,
    include_fallbacks: bool,
) -> List[BroadRegion]:
    regions: List[BroadRegion] = []
    regions.extend(yolo_regions(image_doc, label_dir, width, height))
    regions.extend(ocr_expansion_regions(image_doc, width, height))
    if include_fallbacks:
        regions.extend(fallback_regions(width, height))

    out: List[BroadRegion] = []
    for region in regions:
        box = clamp_box(region.bbox, width, height)
        if box_area(box) < 20:
            continue
        if any(iou(box, prev.bbox) > 0.86 for prev in out):
            continue
        out.append(BroadRegion(box, region.source))
    return out


def make_crop_variants(pil_image: Any, box: BBox, raw_only: bool) -> List[Any]:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    if x2 <= x1 or y2 <= y1:
        return []
    crop = pil_image.crop((x1, y1, x2, y2)).convert("RGB")
    if raw_only:
        return [crop]
    try:
        from PIL import ImageEnhance, ImageOps
    except ImportError:
        return [crop]
    gray = ImageOps.grayscale(crop).convert("RGB")
    contrast = ImageEnhance.Contrast(gray).enhance(1.8)
    sharp = ImageEnhance.Sharpness(crop).enhance(1.6)
    variants = [crop, contrast, sharp]
    if crop.width < 120 or crop.height < 38:
        variants.append(crop.resize((crop.width * 2, crop.height * 2)))
    return variants


def recognition_hits(
    recognizer: Any,
    pil_image: Any,
    proposals: Sequence[Proposal],
    *,
    min_conf: float,
    max_digits: int,
    raw_only: bool,
    debug_dir: Optional[Path],
    image_id: str,
) -> List[Dict[str, Any]]:
    crops: List[Any] = []
    crop_meta: List[Tuple[Proposal, int, Optional[Path]]] = []
    for proposal_index, proposal in enumerate(proposals, 1):
        variants = make_crop_variants(pil_image, proposal.bbox, raw_only=raw_only)
        for variant_index, variant in enumerate(variants):
            crop_path: Optional[Path] = None
            if debug_dir is not None:
                crop_dir = debug_dir / "crops"
                crop_dir.mkdir(parents=True, exist_ok=True)
                crop_path = crop_dir / f"{Path(image_id).stem}_{proposal_index:03d}_{variant_index}.png"
                try:
                    variant.save(crop_path)
                except Exception:
                    crop_path = None
            crops.append(variant)
            crop_meta.append((proposal, variant_index, crop_path))

    hits: Dict[Tuple[str, Tuple[int, int, int, int], str], Dict[str, Any]] = {}
    for (proposal, variant_index, crop_path), predictions in zip(crop_meta, recognizer.predict_many(crops)):
        for value, conf in predictions:
            value = digits(value)
            if conf < min_conf or not (1 <= len(value) <= max_digits):
                continue
            key_box = tuple(int(round(v)) for v in proposal.bbox)
            key = (value, key_box, proposal.source)
            prev = hits.get(key)
            if prev is not None and float(prev.get("ocr_conf", 0.0)) >= float(conf):
                continue
            metadata = dict(proposal.metadata)
            metadata.update(
                {
                    "recognizer_text": value,
                    "recognizer_conf": round(float(conf), 6),
                    "variant_index": variant_index,
                }
            )
            hit: Dict[str, Any] = {
                "text": value,
                "normalized_text": value,
                "bbox_xyxy": [round(float(v), 3) for v in proposal.bbox],
                "bbox": [round(float(v), 3) for v in proposal.bbox],
                "ocr_conf": round(float(conf), 6),
                "recognizer_conf": round(float(conf), 6),
                "detection_conf": round(float(proposal.score), 6),
                "score": round(float(conf) * 0.72 + float(proposal.score) * 0.28, 6),
                "source": proposal.source,
                "raw_source": proposal.metadata.get("raw_source", proposal.source),
                "proposal_kind": proposal.kind,
                "proposal_score": round(float(proposal.score), 6),
                "recognizer_text": value,
                "metadata": metadata,
            }
            for key_name in (
                "broad_region_source",
                "broad_region_bbox",
                "local_bbox",
                "full_bbox",
                "component_count",
                "textline_group_id",
                "proposal_score",
            ):
                if key_name in metadata:
                    hit[key_name] = metadata[key_name]
            if crop_path is not None:
                hit["crop_debug_path"] = str(crop_path)
            hits[key] = hit
    return sorted(hits.values(), key=lambda item: (float(item.get("score", 0.0)), float(item.get("ocr_conf", 0.0))), reverse=True)


def load_recognizer(args: argparse.Namespace) -> Any:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    try:
        from channel_digit_recognizer import ChannelDigitRecognizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ChannelDigitRecognizer dependencies are unavailable in this Python environment. "
            "Run with the project OCR environment that has Pillow, numpy, PaddleOCR/Paddle installed."
        ) from exc

    return ChannelDigitRecognizer(
        args.model_dir,
        model_name=args.model_name,
        device=args.device,
        input_shape=args.input_shape,
    )


def write_boxes_svg(path: Path, width: int, height: int, boxes: Sequence[Tuple[BBox, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#111"/>',
    ]
    for box, color, label in boxes:
        x1, y1, x2, y2 = box
        lines.append(
            f'<rect x="{x1:.2f}" y="{y1:.2f}" width="{max(0.0, x2-x1):.2f}" height="{max(0.0, y2-y1):.2f}" '
            f'fill="none" stroke="{color}" stroke-width="1.5"/>'
        )
        if label:
            lines.append(
                f'<text x="{x1 + 2:.2f}" y="{max(10.0, y1 + 10):.2f}" fill="{color}" '
                f'font-size="10" font-family="monospace">{escape_xml(label)}</text>'
            )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape_xml(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def process_document(args: argparse.Namespace) -> Dict[str, Any]:
    doc = json.loads(args.ocr_json.read_text(encoding="utf-8"))
    images = doc.get("images", [])
    if not isinstance(images, list):
        raise SystemExit("input JSON must contain an images[] list")

    recognizer = load_recognizer(args)
    added = 0
    debug_root: Optional[Path] = args.slot_v2_debug_dir if args.slot_v2_save_debug_images else None
    if debug_root is not None:
        debug_root.mkdir(parents=True, exist_ok=True)

    for image_index, image_doc in enumerate(images, 1):
        if not isinstance(image_doc, dict):
            continue
        image_id = str(first_present(image_doc, ("image_id", "image_name", "filename", "file_name")) or f"image_{image_index:06d}")
        image_path = resolve_image_path(image_doc, args.images_dir)
        if not image_path.exists():
            image_doc.setdefault("slot_proposal_v2_warnings", []).append(f"image_not_found:{image_path}")
            continue
        try:
            gray, pil_image = load_gray_and_pil(image_path)
        except Exception as exc:
            image_doc.setdefault("slot_proposal_v2_warnings", []).append(str(exc))
            continue
        if pil_image is None:
            image_doc.setdefault("slot_proposal_v2_warnings", []).append("recognition_requires_pillow_rgb_image")
            continue
        image_doc["image_width"] = gray.width
        image_doc["image_height"] = gray.height

        image_regions = broad_regions(
            image_doc,
            label_dir=args.yolo_label_dir,
            width=gray.width,
            height=gray.height,
            include_fallbacks=not args.yolo_only,
        )
        image_proposals: List[Proposal] = []
        for region_index, region in enumerate(image_regions, 1):
            region_debug_dir = None
            if debug_root is not None:
                region_debug_dir = debug_root / "regions"
                region_debug_dir.mkdir(parents=True, exist_ok=True)
            image_proposals.extend(
                proposals_for_region(
                    gray,
                    region,
                    mode=args.slot_proposal_mode,
                    max_proposals=args.slot_v2_max_proposals_per_region,
                    min_component_height=args.slot_v2_min_component_height,
                    max_component_height_ratio=args.slot_v2_max_component_height_ratio,
                    debug_dir=region_debug_dir,
                    debug_prefix=f"{Path(image_id).stem}_r{region_index:02d}",
                )
            )
        image_proposals = dedupe_proposals(image_proposals)[: args.max_proposals_per_image]
        hits = recognition_hits(
            recognizer,
            pil_image,
            image_proposals,
            min_conf=args.min_conf,
            max_digits=args.max_digits,
            raw_only=args.raw_only,
            debug_dir=debug_root,
            image_id=image_id,
        )
        hits = hits[: args.max_candidates_per_image]
        for hit_index, hit in enumerate(hits, 1):
            hit["id"] = f"slot_v2_{image_index:05d}_{hit_index:03d}"
            hit["candidate_id"] = hit["id"]
            image_doc.setdefault("candidates", []).append(hit)
            added += 1

        if args.progress_every and (
            image_index == 1 or image_index % args.progress_every == 0 or image_index == len(images)
        ):
            print(
                f"progress {image_index}/{len(images)} regions={len(image_regions)} proposals={len(image_proposals)} added={added}",
                flush=True,
            )

    doc.setdefault("slot_proposal_v2_summary", {})
    doc["slot_proposal_v2_summary"].update(
        {
            "mode": args.slot_proposal_mode,
            "source": SOURCE_CC_TEXTLINE,
            "added_candidate_count": added,
            "image_count": len(images),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} with {added} slot proposal v2 candidates")
    return doc


def draw_digit_text(image: GrayImage, text: str, x: int, y: int, *, scale: int, value: int) -> BBox:
    patterns = {
        "0": ["111", "101", "101", "101", "111"],
        "1": ["010", "110", "010", "010", "111"],
        "2": ["111", "001", "111", "100", "111"],
        "3": ["111", "001", "111", "001", "111"],
        "4": ["101", "101", "111", "001", "001"],
        "5": ["111", "100", "111", "001", "111"],
        "6": ["111", "100", "111", "101", "111"],
        "7": ["111", "001", "010", "010", "010"],
        "8": ["111", "101", "111", "101", "111"],
        "9": ["111", "101", "111", "001", "111"],
    }
    cursor = x
    boxes: List[BBox] = []
    for char in text:
        pattern = patterns[char]
        for py, row in enumerate(pattern):
            for px, cell in enumerate(row):
                if cell == "1":
                    image.fill_rect((cursor + px * scale, y + py * scale, cursor + (px + 1) * scale, y + (py + 1) * scale), value)
        boxes.append((cursor, y, cursor + 3 * scale, y + 5 * scale))
        cursor += 4 * scale
    return span_box(boxes)


def draw_letter_blob_text(image: GrayImage, text: str, x: int, y: int, *, scale: int, value: int) -> BBox:
    patterns = {
        "M": ["10001", "11011", "10101", "10001", "10001"],
        "a": ["0000", "0110", "0001", "0111", "0111"],
        "s": ["0111", "0100", "0110", "0001", "1110"],
        "Q": ["1110", "1001", "1001", "1110", "0011"],
        "c": ["011", "100", "100", "100", "011"],
        "m": ["00000", "11010", "10101", "10101", "10101"],
        "o": ["000", "111", "101", "101", "111"],
        "P": ["110", "101", "110", "100", "100"],
        "D": ["110", "101", "101", "101", "110"],
    }
    cursor = x
    boxes: List[BBox] = []
    for char in text:
        if char == " ":
            cursor += 3 * scale
            continue
        pattern = patterns.get(char, patterns["c"])
        local_scale = scale
        if char in {"a", "s", "c", "o"}:
            local_scale = max(1, scale - 1)
        char_y = y + (scale if char.islower() else 0)
        for py, row in enumerate(pattern):
            for px, cell in enumerate(row):
                if cell == "1":
                    image.fill_rect(
                        (
                            cursor + px * local_scale,
                            char_y + py * local_scale,
                            cursor + (px + 1) * local_scale,
                            char_y + (py + 1) * local_scale,
                        ),
                        value,
                    )
        boxes.append((cursor, char_y, cursor + len(pattern[0]) * local_scale, char_y + len(pattern) * local_scale))
        cursor += (len(pattern[0]) + 1) * local_scale
    return span_box(boxes) if boxes else (x, y, x, y)


def assert_has_tight_digit_proposal(proposals: Sequence[Proposal], target: BBox, message: str) -> None:
    matches = [
        proposal
        for proposal in proposals
        if iou(proposal.bbox, target) >= 0.45
        or (
            proposal.bbox[0] <= target[0] + 3
            and proposal.bbox[2] >= target[2] - 3
            and proposal.bbox[1] <= target[1] + 4
            and proposal.bbox[3] >= target[3] - 4
            and (proposal.bbox[2] - proposal.bbox[0]) <= (target[2] - target[0]) * 1.55
        )
    ]
    assert matches, message


def run_self_test() -> None:
    def proposals_for_synthetic(image: GrayImage) -> List[Proposal]:
        return cc_textline_proposals_for_region(
            image,
            BroadRegion((0, 0, image.width, image.height), "synthetic"),
            max_proposals=40,
            min_component_height=4,
            max_component_height_ratio=0.80,
        )

    dark = GrayImage.blank(180, 80, 25)
    target_041 = draw_digit_text(dark, "041", 18, 22, scale=7, value=240)
    dark_props = proposals_for_synthetic(dark)
    assert_has_tight_digit_proposal(dark_props, target_041, "dark panel white 041 was not proposed")
    assert min(proposal.bbox[0] for proposal in dark_props if proposal.score > 0.65) <= target_041[0] + 1, "leading zero left edge was clipped"

    light = GrayImage.blank(220, 90, 235)
    target_1234 = draw_digit_text(light, "1234", 24, 24, scale=7, value=20)
    light_props = proposals_for_synthetic(light)
    assert_has_tight_digit_proposal(light_props, target_1234, "light panel black 1234 was not proposed")

    mixed = GrayImage.blank(260, 90, 30)
    target_mixed = draw_digit_text(mixed, "041", 18, 24, scale=7, value=245)
    letters = draw_letter_blob_text(mixed, "QcmoPD", 112, 24, scale=6, value=245)
    mixed_props = proposals_for_synthetic(mixed)
    digit_like = [
        proposal
        for proposal in mixed_props
        if proposal.bbox[0] <= target_mixed[0] + 4
        and proposal.bbox[2] >= target_mixed[2] - 4
        and proposal.bbox[2] < letters[0] - 4
    ]
    assert digit_like, "numeric textline was not split from adjacent letters"

    letters_only = GrayImage.blank(160, 70, 235)
    draw_letter_blob_text(letters_only, "Mas", 24, 18, scale=7, value=20)
    letter_props = proposals_for_synthetic(letters_only)
    high_score_letters = [proposal for proposal in letter_props if proposal.score >= 0.80]
    assert not high_score_letters, "letter-only crop produced high-confidence numeric proposals"
    assert len(letter_props) <= 20, "letter-only crop produced an excessive number of numeric proposals"

    print("slot_proposal_numeric_recheck_v2 self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Connected-component/textline numeric slot proposal recheck.")
    parser.add_argument("--ocr-json", type=Path, required=False, help="Input OCR/candidate JSON with images[].")
    parser.add_argument("--out", type=Path, required=False, help="Output JSON with appended slot proposal candidates.")
    parser.add_argument("--images-dir", type=Path, default=None, help="Optional directory used to resolve image paths.")
    parser.add_argument("--yolo-label-dir", type=Path, default=None, help="YOLO label directory; class 0/3 boxes seed broad regions.")
    parser.add_argument("--model-dir", type=Path, default=Path("runs/ocr/inference"))
    parser.add_argument("--model-name", default="PP-OCRv5_mobile_rec")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--input-shape", default="3,48,320")
    parser.add_argument("--min-conf", type=float, default=0.90)
    parser.add_argument("--max-digits", type=int, default=5)
    parser.add_argument("--max-proposals-per-image", type=int, default=160)
    parser.add_argument("--max-candidates-per-image", type=int, default=45)
    parser.add_argument("--raw-only", action="store_true", help="Run recognition only on raw proposal crops.")
    parser.add_argument("--yolo-only", action="store_true", help="Only split YOLO class 0/3 regions; skip OCR/fallback broad regions.")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--slot-proposal-mode", choices=["legacy", "cc_textline", "hybrid"], default="cc_textline")
    parser.add_argument("--slot-v2-debug-dir", type=Path, default=None)
    parser.add_argument("--slot-v2-max-proposals-per-region", type=int, default=20)
    parser.add_argument("--slot-v2-min-component-height", type=float, default=5.0)
    parser.add_argument("--slot-v2-max-component-height-ratio", type=float, default=0.72)
    parser.add_argument("--slot-v2-save-debug-images", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic connected-component proposal tests.")
    return parser


def run_cli_from_namespace(args: argparse.Namespace) -> Dict[str, Any]:
    return process_document(args)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        run_self_test()
        return
    if args.ocr_json is None or args.out is None:
        parser.error("--ocr-json and --out are required unless --self-test is used")
    try:
        process_document(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
