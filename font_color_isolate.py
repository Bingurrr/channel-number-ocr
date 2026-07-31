#!/usr/bin/env python3
"""[실험] 폴더의 채널 ROI에서 '성공 detection 프레임의 폰트색'을 학습 → 그 색만 남기고
나머지 제거 → 다시 읽기. 흰-on-흰(플레이트 없는 흰 숫자) 실패 프레임 구제용.

사용자 아이디어:
  1. 채널영역(ROI)에서 숫자가 잘 읽힌(고conf) 프레임들의 '숫자 픽셀 색' = 폰트색 학습
  2. 실패 프레임의 ROI에서 폰트색 아닌 픽셀 다 제거 → 숫자만 남김
  3. 다시 OCR → none 해소

폴더 경로만 주면 그 안 이미지들에 대해 결과(정제 crop + 시각화 + 요약 CSV) 저장.

ROI 지정:
  --roi x1,y1,x2,y2         정규화 채널 영역 (예: 0.02,0.66,0.32,0.82)
  --field-box-from FILE     predict_folder_slot의 profile_report.json에서 channel_field_box(픽셀) 읽기
  (둘 다 없으면 전체 이미지)

예:
  python font_color_isolate.py --images /path/to/frames --out ./fc_out \
      --roi 0.02,0.66,0.32,0.82 --rec-model-dir models/full_image_ocr/en_PP-OCRv4_mobile_rec_ft
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
from pathlib import Path

try:
    import torch  # noqa: F401  (paddle/torch 로드 순서 크래시 회피)
except Exception:
    pass

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import preprocess_digits as PD

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def best_digit(text):
    runs = [r for r in re.findall(r"\d+", str(text)) if 1 <= len(r) <= 5]
    if runs:
        return max(runs, key=len)
    return re.sub(r"\D", "", str(text))[:5]


def _model_name_of(model_dir):
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
    if model_dir and Path(model_dir).exists():
        return TextRecognition(model_name=_model_name_of(model_dir), model_dir=str(model_dir))
    return TextRecognition(model_name="en_PP-OCRv4_mobile_rec")


def rec_read(rec, img, tmp, min_h=120):
    """rec 한 장 → (digit, conf). 작으면 확대."""
    u = max(1.0, min_h / max(1, img.height))
    im = img.resize((max(1, int(img.width * u)), max(1, int(img.height * u)))) if u > 1.0 else img
    cp = tmp / "c.png"; im.convert("RGB").save(cp)
    r = rec.predict(str(cp))
    txt = r[0].get("rec_text", "") if r else ""
    conf = float(r[0].get("rec_score", 0.0) or 0.0) if r else 0.0
    return best_digit(txt), conf


def roi_box(W, H, roi_norm, field_box, pad=0.15):
    if field_box:
        x1, y1, x2, y2 = field_box
        pw = (x2 - x1) * pad; ph = (y2 - y1) * pad
        return (max(0, int(x1 - pw)), max(0, int(y1 - ph)), min(W, int(x2 + pw)), min(H, int(y2 + ph)))
    if roi_norm:
        x1, y1, x2, y2 = roi_norm
        return (int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H))
    return (0, 0, W, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="이미지 폴더 경로")
    ap.add_argument("--out", required=True, help="결과 저장 폴더")
    ap.add_argument("--roi", default=None, help="정규화 채널 ROI x1,y1,x2,y2")
    ap.add_argument("--field-box-from", default=None, help="profile_report.json (channel_field_box 픽셀)")
    ap.add_argument("--rec-model-dir", default=None, help="파인튜닝 rec 디렉토리")
    ap.add_argument("--learn-conf", type=float, default=0.7, help="이 conf 이상 프레임에서만 폰트색 학습")
    ap.add_argument("--min-frac", type=float, default=0.10, help="폰트색 판정: 박스에서 이 비율 이상 나온 색")
    ap.add_argument("--tol", type=float, default=60, help="색분리 허용 거리(클수록 폰트색 근처 더 포함)")
    ap.add_argument("--viz-n", type=int, default=60)
    args = ap.parse_args()

    roi_norm = tuple(float(x) for x in args.roi.split(",")) if args.roi else None
    field_box = None
    if args.field_box_from:
        rep = json.loads(Path(args.field_box_from).read_text())
        for r in (rep if isinstance(rep, list) else [rep]):
            if r.get("channel_field_box"):
                field_box = r["channel_field_box"]; break

    outd = Path(args.out); clean_d = outd / "cleaned"; viz_d = outd / "viz"
    for d in (clean_d, viz_d):
        d.mkdir(parents=True, exist_ok=True)

    rec = load_recognizer(args.rec_model_dir)
    tmp = Path(tempfile.mkdtemp(prefix="fci_"))
    imgs = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"이미지 {len(imgs)}장, ROI={'field_box' if field_box else roi_norm or '전체'}", flush=True)

    # ── PASS 1: ROI crop + 원본 읽기 → 성공(고conf) crop 수집 ──
    rois, reads = [], []
    for p in imgs:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            rois.append(None); reads.append(("", 0.0)); continue
        b = roi_box(im.width, im.height, roi_norm, field_box)
        crop = im.crop(b)
        rois.append(crop)
        reads.append(rec_read(rec, crop, tmp))

    # ── 폰트색 학습: 고conf(성공) 프레임의 ROI에서, 히스토그램(10%↑+프레임간 일관) ──
    learn_crops = [rois[i] for i in range(len(imgs)) if reads[i][1] >= args.learn_conf and rois[i] is not None]
    if not learn_crops:            # 성공 프레임 없으면 전체로 학습 (차선)
        print(f"경고: conf>={args.learn_conf} 프레임 없음 → 전체 {len(imgs)}장으로 학습", flush=True)
        learn_crops = [r for r in rois if r is not None]
    font = PD.learn_font_color_hist(learn_crops, min_frac=args.min_frac)
    fs = "None" if font is None else tuple(int(v) for v in font)
    print(f"학습에 쓴 프레임 {len(learn_crops)}장 → 폰트색(히스토그램)={fs}", flush=True)
    if font is None:
        print("경고: 폰트색 학습 실패. --min-frac 낮추거나 ROI 확인.", flush=True)

    # ── PASS 2: 색분리 → 재읽기 → 저장 ──
    rows = []
    viz = improved = 0
    for i, p in enumerate(imgs):
        if rois[i] is None:
            continue
        old_d, old_c = reads[i]
        clean, mask = PD.isolate_contrast(rois[i], font, tol=args.tol)   # 대비색 배경
        new_d, new_c = rec_read(rec, clean, tmp)
        clean.save(clean_d / f"{p.stem}.png")
        imp = bool(new_d) and (not old_d or new_c > old_c + 0.05)
        improved += int(imp and old_c < args.learn_conf)
        rows.append({"frame": p.name, "old_read": old_d, "old_conf": round(old_c, 3),
                     "new_read": new_d, "new_conf": round(new_c, 3), "improved": imp})
        if viz < args.viz_n:
            PD.make_viz(rois[i], mask, clean, font, None,
                        text=f"{old_d}({old_c:.2f})->{new_d}({new_c:.2f})").save(viz_d / f"{p.stem}.jpg")
            viz += 1

    with (outd / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "old_read", "old_conf", "new_read", "new_conf", "improved"])
        w.writeheader(); w.writerows(rows)

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n완료: 정제 crop → {clean_d}\n      시각화 {viz}장 → {viz_d}\n      요약 → {outd}/summary.csv", flush=True)
    print(f"저conf였다가 색분리로 개선된 프레임: 약 {improved}개", flush=True)


if __name__ == "__main__":
    main()
