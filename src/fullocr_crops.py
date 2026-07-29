#!/usr/bin/env python3
"""Run FULL OCR (PP-OCRv4, full charset) on each YOLO channel-box crop, then keep
only pure-digit tokens.

Fixes the merged 'DirecTV 123' case: the digit-only recognizer forces the letters
to fake digits, so you cannot tell the real number afterwards. Full OCR has the
whole charset, so it reads 'DirecTV' as letters and '123' as digits; we then keep
the digit run and drop the text. No position assumption (works whether the label
sits left/right/above the number) — any pure-digit token of 1-4 chars that is not
part of a time is a candidate; the selector picks among them.

Output candidates.json matches the recheck step's schema (feeds temporal pipeline).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from ocr_candidate_extractor import create_paddle_ocr, extract_candidates_from_image
from easyocr_numeric_recheck import yolo_boxes

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TIME_RE = re.compile(r"\d{1,2}\s*:\s*\d{2}")


def digit_tokens(text: str):
    """Pure-digit runs of length 1-4 that are not part of a clock time."""
    if TIME_RE.search(str(text)):
        return []
    return [t for t in re.findall(r"\d+", str(text)) if 1 <= len(t) <= 5]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--yolo-label-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pad", type=float, default=0.15,
                    help="채널박스 여백 비율 (숫자 가장자리 안 잘리게)")
    ap.add_argument("--min-height", type=int, default=40,
                    help="crop 높이가 이보다 작으면 확대 (작은 숫자 인식 보강)")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args()

    ocr = create_paddle_ocr(
        lang=args.lang, use_gpu=True, ocr_version="PP-OCRv4",
        text_detection_model_name="PP-OCRv4_mobile_det",
        text_recognition_model_name="en_PP-OCRv4_mobile_rec",
    )
    imgs = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMG_EXTS)
    tmp = Path(tempfile.mkdtemp(prefix="ocrcrop_"))
    out_images = []
    for idx, ip in enumerate(imgs, 1):
        try:
            with Image.open(ip) as im0:
                im = im0.convert("RGB")
        except Exception:
            out_images.append({"image_id": ip.stem, "image_path": str(ip),
                               "image_width": 0, "image_height": 0, "candidates": []})
            continue
        W, H = im.size
        cands = []
        for bi, (x1, y1, x2, y2) in enumerate(yolo_boxes(args.yolo_label_dir, ip.stem, W, H)):
            pw = (x2 - x1) * args.pad; ph = (y2 - y1) * args.pad
            cx1, cy1 = max(0.0, x1 - pw), max(0.0, y1 - ph)
            cx2, cy2 = min(W, x2 + pw), min(H, y2 + ph)
            if cx2 - cx1 < 4 or cy2 - cy1 < 4:
                continue
            crop = im.crop((int(cx1), int(cy1), int(cx2), int(cy2)))
            u = max(1.0, args.min_height / max(1, crop.height))       # 작으면 확대
            if u > 1.0:
                crop = crop.resize((int(crop.width * u), int(crop.height * u)))
            cp = tmp / "crop.jpg"
            crop.save(cp)
            for c in extract_candidates_from_image(cp, ocr=ocr):
                d = c.to_dict()
                for tok in digit_tokens(d.get("text", "")):
                    cands.append({
                        "id": f"crop_{bi:02d}_{tok}_{len(cands):03d}",
                        "text": tok,
                        "bbox_xyxy": [round(cx1, 1), round(cy1, 1), round(cx2, 1), round(cy2, 1)],
                        "ocr_conf": round(float(d.get("ocr_conf", 0.5) or 0.5), 6),
                        "detection_conf": round(float(d.get("ocr_conf", 0.5) or 0.5), 6),
                        "source": "yolocrop_fullocr",
                    })
        out_images.append({"image_id": ip.stem, "image_path": str(ip),
                           "image_width": W, "image_height": H, "candidates": cands})
        if idx % args.progress_every == 0 or idx == len(imgs):
            tot = sum(len(im_["candidates"]) for im_ in out_images)
            print(f"progress {idx}/{len(imgs)} candidates={tot}", flush=True)

    shutil.rmtree(tmp, ignore_errors=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"images": out_images}, ensure_ascii=False))
    print(f"wrote {args.out} ({sum(len(i['candidates']) for i in out_images)} digit candidates)",
          flush=True)


if __name__ == "__main__":
    main()
