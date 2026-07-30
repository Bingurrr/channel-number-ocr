#!/usr/bin/env python3
"""Read a channel number from its known ROI, using COLOR MASKING to strip an
overlapping program/broadcaster text before recognition.

Flow per image (ROI already known -> no full-image detection needed):
    crop ROI(+pad) -> color-mask to the channel font color -> upscale -> OCR
                   -> keep the pure-digit token (1..5)

The channel font color comes from --colors (uid -> [r,g,b], learned by the caller
from clean frames). If a uid has no color, we learn it from the crop itself; if the
masked read yields no digit, we fall back to the plain (unmasked) crop so masking
can only help, never hurt.

Note: this runs FULL OCR on the (tiny) masked crop to measure the masking benefit.
On-device this stage becomes rec-only (det skipped) since the ROI is known.
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

import color_mask as CM
from ocr_candidate_extractor import create_paddle_ocr, extract_candidates_from_image
from easyocr_numeric_recheck import yolo_boxes

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TIME_RE = re.compile(r"\d{1,2}\s*:\s*\d{2}")


def digit_tokens(text):
    if TIME_RE.search(str(text)):
        return []
    return [t for t in re.findall(r"\d+", str(text)) if 1 <= len(t) <= 5]


def _read(ocr, im, tmp):
    """Save `im` and return the first pure-digit token OCR finds, else ''."""
    cp = tmp / "c.png"
    im.convert("RGB").save(cp)
    for c in extract_candidates_from_image(cp, ocr=ocr):
        for tok in digit_tokens(c.to_dict().get("text", "")):
            return tok
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--yolo-label-dir", type=Path, required=True, help="ROI 라벨(채널박스) 디렉토리")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--colors", type=Path, help="uid -> [r,g,b] json (깨끗한 프레임에서 학습한 채널색)")
    ap.add_argument("--pad", type=float, default=0.2)
    ap.add_argument("--min-height", type=int, default=120)
    ap.add_argument("--tol", type=int, default=70, help="색 거리 허용치(작을수록 배경 더 제거)")
    ap.add_argument("--viz-dir", type=Path, help="마스킹 스텝 시각화 저장 디렉토리")
    ap.add_argument("--viz-per", type=int, default=8, help="시각화 최대 장수")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args()

    colors = {}
    if args.colors and args.colors.exists():
        colors = json.loads(args.colors.read_text())

    ocr = create_paddle_ocr(
        lang=args.lang, use_gpu=True, ocr_version="PP-OCRv4",
        text_detection_model_name="PP-OCRv4_mobile_det",
        text_recognition_model_name="en_PP-OCRv4_mobile_rec",
    )
    imgs = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMG_EXTS)
    tmp = Path(tempfile.mkdtemp(prefix="maskread_"))
    out_images = []
    n_masked, n_fallback, n_viz = 0, 0, 0
    for idx, ip in enumerate(imgs, 1):
        try:
            im = Image.open(ip).convert("RGB")
        except Exception:
            out_images.append({"image_id": ip.stem, "image_path": str(ip),
                               "image_width": 0, "image_height": 0, "candidates": []})
            continue
        W, H = im.size
        cands = []
        for bi, (x1, y1, x2, y2) in enumerate(yolo_boxes(args.yolo_label_dir, ip.stem, W, H)):
            crop = CM.crop_roi(im, [x1, y1, x2, y2], args.pad)
            if crop is None:
                continue
            col = colors.get(ip.stem) or CM.learn_text_color(crop)
            tok, via = "", "plain"
            if col is not None:                                # 1) 색 마스킹 후 읽기
                masked = CM.apply_color_mask(crop, tuple(col), args.tol)
                up, _ = CM.upscale(masked, args.min_height)
                tok = _read(ocr, up, tmp)
                if tok:
                    via = "mask"; n_masked += 1
            if not tok:                                        # 2) 폴백: 원본 crop 확대 읽기
                up, _ = CM.upscale(crop, args.min_height)
                tok = _read(ocr, up, tmp)
                if tok:
                    via = "plain"; n_fallback += 1
            if tok:
                cands.append({"id": f"mask_{bi:02d}_{tok}", "text": tok,
                              "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                              "ocr_conf": 0.9, "detection_conf": 0.9, "source": f"mask_read:{via}"})
            if args.viz_dir and n_viz < args.viz_per:
                CM.mask_steps(im, [x1, y1, x2, y2], args.viz_dir / ip.stem,
                              tuple(col) if col else None, tok, args.pad, args.min_height, args.tol)
                n_viz += 1
        out_images.append({"image_id": ip.stem, "image_path": str(ip),
                           "image_width": W, "image_height": H, "candidates": cands})
        if idx % args.progress_every == 0 or idx == len(imgs):
            print(f"progress {idx}/{len(imgs)}  mask={n_masked} fallback={n_fallback}", flush=True)

    shutil.rmtree(tmp, ignore_errors=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"images": out_images}, ensure_ascii=False))
    print(f"wrote {args.out}  (mask읽기 {n_masked}, 폴백 {n_fallback})", flush=True)


if __name__ == "__main__":
    main()
