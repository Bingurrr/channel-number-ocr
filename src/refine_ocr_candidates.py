"""Refine OCR candidates by adding numeric sub-candidates from mixed text."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


BBox = Tuple[float, float, float, float]
NUMERIC_RE = re.compile(r"[0-9][0-9:.\-/]*[0-9]|[0-9]")


def clean_digits(text: str) -> str:
    return "".join(ch for ch in text if "0" <= ch <= "9")


def is_whole_numeric(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and bool(re.fullmatch(r"[0-9\s:.\-/]+", stripped))


def split_numeric_candidates(candidate: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    text = str(candidate.get("text", ""))
    if is_whole_numeric(text):
        return
    bbox = tuple(float(v) for v in candidate["bbox_xyxy"])
    x1, y1, x2, y2 = bbox
    width = max(1.0, x2 - x1)
    n = max(1, len(text))
    index = 1
    for match in NUMERIC_RE.finditer(text):
        raw = match.group(0)
        digits = clean_digits(raw)
        if not digits:
            continue
        sx = x1 + width * (match.start() / n)
        ex = x1 + width * (match.end() / n)
        pad = max(2.0, (ex - sx) * 0.12)
        refined = dict(candidate)
        refined["id"] = f"{candidate.get('id', 'ocr')}_num_{index:02d}"
        refined["text"] = digits
        refined["bbox_xyxy"] = [round(max(0.0, sx - pad), 3), y1, round(ex + pad, 3), y2]
        refined["source"] = "refined_numeric_substring"
        refined["parent_id"] = candidate.get("id")
        refined["parent_text"] = text
        refined["ocr_conf"] = round(float(candidate.get("ocr_conf", candidate.get("confidence", 1.0))) * 0.94, 6)
        refined["detection_conf"] = refined["ocr_conf"]
        yield refined
        index += 1


def refine_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = {"images": []}
    for image in doc.get("images", []):
        candidates = [dict(c) for c in image.get("candidates", [])]
        existing_ids = {str(c.get("id")) for c in candidates}
        additions: List[Dict[str, Any]] = []
        for candidate in candidates:
            for refined in split_numeric_candidates(candidate):
                if refined["id"] in existing_ids:
                    continue
                additions.append(refined)
                existing_ids.add(refined["id"])
        new_image = dict(image)
        new_image["candidates"] = candidates + additions
        new_image["refined_candidate_count"] = len(additions)
        out["images"].append(new_image)
    return out


def draw_visualizations(doc: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    for image in doc.get("images", []):
        image_path = resolve_image_path(Path(str(image.get("image_path", ""))))
        if not image_path.exists():
            continue
        with Image.open(image_path) as src:
            canvas = src.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for candidate in image.get("candidates", []):
            bbox = candidate.get("bbox_xyxy")
            if not bbox:
                continue
            color = (255, 80, 40) if candidate.get("source") == "refined_numeric_substring" else (80, 180, 255)
            x1, y1, x2, y2 = [float(v) for v in bbox]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
            label = str(candidate.get("text", ""))[:20]
            draw.text((x1, max(0, y1 - 12)), label, fill=color, font=font)
        canvas.save(out_dir / f"{image['image_id']}_refined.jpg")


def resolve_image_path(path: Path) -> Path:
    text = str(path).replace("\\", "/")
    path = Path(text)
    if path.exists():
        return path
    base = Path.cwd()
    candidate = base / path
    if candidate.exists():
        return candidate
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--visualize-dir", type=Path, default=None)
    args = parser.parse_args()

    doc = json.loads(args.ocr_json.read_text(encoding="utf-8"))
    refined = refine_doc(doc)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(refined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.visualize_dir:
        draw_visualizations(refined, args.visualize_dir)
    added = sum(int(img.get("refined_candidate_count", 0)) for img in refined.get("images", []))
    print(f"wrote {args.out} with {added} refined candidates")


if __name__ == "__main__":
    main()
