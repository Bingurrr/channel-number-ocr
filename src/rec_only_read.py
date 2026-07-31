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
import os
import re
import sys
import tempfile
from pathlib import Path

# preprocess_digits.py는 리포 루트(src의 부모)에 있음 → import 경로 보장.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def _model_name_of(model_dir):
    """inference.yml에서 model_name 자동 감지 → v4/v6 등 버전 무관."""
    y = Path(model_dir) / "inference.yml"
    if y.exists():
        try:
            import yaml
            n = yaml.safe_load(open(y)).get("Global", {}).get("model_name")
            if n:
                return n
        except Exception:
            pass
    return "en_PP-OCRv4_mobile_rec"


def load_recognizer(model_dir):
    from paddleocr import TextRecognition
    # model_dir의 실제 model_name을 감지해서 전달 (v6 등 불일치 방지).
    if model_dir and Path(model_dir).exists():
        return TextRecognition(model_name=_model_name_of(model_dir), model_dir=str(model_dir))
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
    ap.add_argument("--color-isolate", action="store_true",
                    help="여러 프레임에서 폰트색을 학습해 숫자만 남기고 배경 제거 후 rec "
                         "(유사배경/저대비 프레임 읽기 개선)")
    ap.add_argument("--color-isolate-viz", type=Path, default=None,
                    help="색분리 과정 시각화 저장 디렉토리(옵션)")
    ap.add_argument("--viz-n", type=int, default=40, help="시각화 장수")
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args()

    rec = load_recognizer(args.rec_model_dir)
    imgs = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMG_EXTS)
    tmp = Path(tempfile.mkdtemp(prefix="reconly_"))

    # ── PASS 1: 모든 ROI crop 수집 (색분리면 폰트색 학습용) ──
    #    each item: (image_id, image_path, W, H, bi, [x1,y1,x2,y2], crop_PIL)
    items, per_img = [], {}
    for ip in imgs:
        try:
            im = Image.open(ip).convert("RGB")
        except Exception:
            per_img[ip.stem] = (str(ip), 0, 0)
            continue
        W, H = im.size
        per_img[ip.stem] = (str(ip), W, H)
        for bi, (x1, y1, x2, y2) in enumerate(yolo_boxes(args.yolo_label_dir, ip.stem, W, H)):
            pw = (x2 - x1) * args.pad; ph = (y2 - y1) * args.pad
            cx1, cy1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
            cx2, cy2 = min(W, int(x2 + pw)), min(H, int(y2 + ph))
            if cx2 - cx1 < 3 or cy2 - cy1 < 3:
                continue
            items.append([ip.stem, bi, [x1, y1, x2, y2], im.crop((cx1, cy1, cx2, cy2))])

    font = bg = None
    if args.color_isolate:
        import preprocess_digits as PD
        font, bg = PD.learn_font_color([it[3] for it in items])
        fs = "None" if font is None else tuple(int(v) for v in font)
        print(f"[color-isolate] {len(items)}개 crop에서 폰트색 학습 = {fs}", flush=True)
        if args.color_isolate_viz:
            args.color_isolate_viz.mkdir(parents=True, exist_ok=True)

    # ── PASS 2: (색분리 →) 확대 → rec ──
    cand_of = {}
    forced = viz = 0
    for idx, (stem, bi, box, crop) in enumerate(items, 1):
        clean, mask = (crop, None)
        if args.color_isolate:
            import preprocess_digits as PD
            clean, mask = PD.isolate(crop, font, bg)
        u = max(1.0, args.min_height / max(1, clean.height))          # 작으면 확대
        rc = clean.resize((max(1, int(clean.width * u)), max(1, int(clean.height * u)))) if u > 1.0 else clean
        cp = tmp / "c.png"; rc.convert("RGB").save(cp)
        r = rec.predict(str(cp))                                      # det 생략, rec만
        txt = r[0].get("rec_text", "") if r else ""
        score = float(r[0].get("rec_score", 0.0) or 0.0) if r else 0.0
        d = best_digit(txt)
        if args.color_isolate and args.color_isolate_viz and mask is not None and viz < args.viz_n:
            PD.make_viz(crop, mask, clean, font, bg, text=f"{d}({score:.2f})").save(
                args.color_isolate_viz / f"{stem}_{bi:02d}.jpg")
            viz += 1
        if score < args.min_score or not d:
            continue
        x1, y1, x2, y2 = box
        cand_of.setdefault(stem, []).append(
            {"id": f"reconly_{bi:02d}_{d}", "text": d,
             "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
             "ocr_conf": round(score, 6), "detection_conf": round(score, 6),
             "source": "rec_only_isolate" if args.color_isolate else "rec_only_forced"})
        forced += 1
        if idx % args.progress_every == 0 or idx == len(items):
            print(f"progress {idx}/{len(items)}  forced={forced}", flush=True)

    out_images = []
    for stem, (path, W, H) in per_img.items():
        out_images.append({"image_id": stem, "image_path": path,
                           "image_width": W, "image_height": H, "candidates": cand_of.get(stem, [])})

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"images": out_images}, ensure_ascii=False))
    print(f"wrote {args.out}  (rec-only 강제읽기 {forced})", flush=True)


if __name__ == "__main__":
    main()
