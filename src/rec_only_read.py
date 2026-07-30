#!/usr/bin/env python3
"""Force a channel-number read on a KNOWN ROI using rec-only (no detection).

The full OCR is det+rec: if det finds no text box at the channel position (faint /
white-on-white / small), the frame comes back as none even though we KNOW from past
inference where the channel is. Since frames are captured at channel change, the
banner is present in every frame, so none = a read failure, not a real absence.

This module skips detection entirely: it crops the known ROI (+pad, upscale) and runs
the recognizer directly, always returning its best digit guess. Cheaper than det+rec
and never returns none. Optionally keeps rec_score so the caller can gate if desired.
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

# Import torch first to avoid the Paddle/Torch load-order crash paddlex can trigger.
try:
    import torch  # noqa: F401
except Exception:
    pass

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from easyocr_numeric_recheck import yolo_boxes

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def best_digit(text):
    """Longest pure-digit run of length 1-5; else digits-only fallback (forced)."""
    runs = [r for r in re.findall(r"\d+", str(text)) if 1 <= len(r) <= 5]
    if runs:
        return max(runs, key=len)
    only = re.sub(r"\D", "", str(text))
    return only[:5]


def load_recognizer(model_dir):
    from paddleocr import TextRecognition
    if model_dir and Path(model_dir).exists():
        return TextRecognition(model_dir=str(model_dir))
    return TextRecognition(model_name="en_PP-OCRv4_mobile_rec")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--yolo-label-dir", type=Path, required=True, help="채널 ROI 라벨 디렉토리")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rec-model-dir", default=None, help="파인튜닝 rec inference 디렉토리")
    ap.add_argument("--pad", type=float, default=0.2)
    ap.add_argument("--min-height", type=int, default=120)
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="이 점수 미만이면 버림(기본 0=무조건 답)")
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args()

    rec = load_recognizer(args.rec_model_dir)
    imgs = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMG_EXTS)
    tmp = Path(tempfile.mkdtemp(prefix="reconly_"))
    out_images = []
    forced = 0
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
            pw = (x2 - x1) * args.pad; ph = (y2 - y1) * args.pad
            cx1, cy1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
            cx2, cy2 = min(W, int(x2 + pw)), min(H, int(y2 + ph))
            if cx2 - cx1 < 3 or cy2 - cy1 < 3:
                continue
            crop = im.crop((cx1, cy1, cx2, cy2))
            u = max(1.0, args.min_height / max(1, crop.height))       # 작으면 확대
            if u > 1.0:
                crop = crop.resize((max(1, int(crop.width * u)), max(1, int(crop.height * u))))
            cp = tmp / "c.png"; crop.save(cp)
            r = rec.predict(str(cp))                                  # det 생략, rec만
            txt = r[0].get("rec_text", "") if r else ""
            score = float(r[0].get("rec_score", 0.0) or 0.0) if r else 0.0
            if score < args.min_score:
                continue
            d = best_digit(txt)
            if d:
                cands.append({"id": f"reconly_{bi:02d}_{d}", "text": d,
                              "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                              "ocr_conf": round(score, 6), "detection_conf": round(score, 6),
                              "source": "rec_only_forced"})
                forced += 1
        out_images.append({"image_id": ip.stem, "image_path": str(ip),
                           "image_width": W, "image_height": H, "candidates": cands})
        if idx % args.progress_every == 0 or idx == len(imgs):
            print(f"progress {idx}/{len(imgs)}  forced={forced}", flush=True)

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"images": out_images}, ensure_ascii=False))
    print(f"wrote {args.out}  (rec-only 강제읽기 {forced})", flush=True)


if __name__ == "__main__":
    main()
