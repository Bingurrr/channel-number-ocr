#!/usr/bin/env python3
"""Run rec (recognition only) on a folder of channel crops and report accuracy.

Each crop's filename encodes the ground truth as the leading digits, e.g.
    5_ab12ef.jpg  -> gt "5"      705_9c1d.jpg -> gt "705"
(the naming produced by extract_real_crops.py / make_*_data.py). This runs the
recognizer directly (no detection) on each crop, extracts the channel digit, and
compares to the filename gt. Measurement only — fine on real crops (not training).

Usage:
  python eval_crops.py --crops ./crop_skylife/images \
      --rec-model-dir models/full_image_ocr/en_PP-OCRv4_mobile_rec_ft
  (omit --rec-model-dir, or pass 'none', to test the stock rec)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# torch first to dodge the Paddle/Torch load-order crash paddlex can trigger.
try:
    import torch  # noqa: F401
except Exception:
    pass

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def gt_of(name):
    m = re.match(r"(\d+)", Path(name).stem)
    return str(int(m.group(1))) if m else ""


def best_digit(text):
    runs = [r for r in re.findall(r"\d+", str(text)) if 1 <= len(r) <= 5]
    return str(int(max(runs, key=len))) if runs else ""


def load_rec(model_dir):
    from paddleocr import TextRecognition
    want = model_dir and str(model_dir).lower() not in ("none", "")
    if want and Path(model_dir).exists() and (Path(model_dir) / "inference.pdiparams").exists():
        print(f"[eval] 파인튜닝 rec 사용: {model_dir}", flush=True)
        return TextRecognition(model_name="en_PP-OCRv4_mobile_rec", model_dir=str(model_dir))
    if want:                                    # 경로 줬는데 없음 → 조용히 순정으로 새지 않게 경고
        print(f"[eval] ⚠ 지정한 rec 경로 없음 → 순정으로 폴백: {model_dir}", flush=True)
    else:
        print("[eval] 순정 rec 사용", flush=True)
    return TextRecognition(model_name="en_PP-OCRv4_mobile_rec")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True, help="crop 이미지 폴더(파일명 앞 숫자=정답)")
    ap.add_argument("--rec-model-dir", default=None, help="파인튜닝 rec 디렉토리('none'=순정)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--show", type=int, default=20, help="틀린 예시 N개 출력")
    args = ap.parse_args()

    imgs = [p for p in sorted(Path(args.crops).rglob("*")) if p.suffix.lower() in IMG_EXTS]
    imgs = [p for p in imgs if gt_of(p.name)]
    if not imgs:
        raise SystemExit(f"라벨 붙은 crop 없음: {args.crops}")
    rec = load_rec(args.rec_model_dir)

    total = correct = 0
    mism = []
    for i in range(0, len(imgs), args.batch):
        chunk = imgs[i:i + args.batch]
        results = rec.predict([str(p) for p in chunk])
        for p, r in zip(chunk, results):
            gt = gt_of(p.name)
            txt = r.get("rec_text", "") if isinstance(r, dict) else ""
            score = float(r.get("rec_score", 0.0) or 0.0) if isinstance(r, dict) else 0.0
            pred = best_digit(txt)
            total += 1
            if pred == gt:
                correct += 1
            elif len(mism) < args.show:
                mism.append((p.name, gt, txt, pred, round(score, 3)))
        if (i + args.batch) % 512 < args.batch:
            print(f"  {min(i+args.batch,len(imgs))}/{len(imgs)}  acc={correct/max(1,total)*100:.1f}%", flush=True)

    print(f"\n=== rec 정확도 ({'ft:'+str(args.rec_model_dir) if args.rec_model_dir and str(args.rec_model_dir).lower()!='none' else '순정'}) ===")
    print(f"맞음 {correct} / 전체 {total} = {correct/max(1,total)*100:.2f}%")
    if mism:
        print(f"\n틀린 예시 (파일 / gt / rec원문 / 추출 / conf):")
        for name, gt, txt, pred, sc in mism:
            print(f"  {name:<34} gt={gt:<6} rec={txt!r:<16} pred={pred or '-':<6} conf={sc}")


if __name__ == "__main__":
    main()
