#!/usr/bin/env python3
"""v5 온디바이스형 — full_ocr.json 없이 프레임 스트리밍, 지연요소 제거.

제거한 지연요소:
  · full_ocr.json 저장/로드  → in-memory 스트리밍
  · 매 프레임 full OCR(det)   → 앞 window 프레임만 det+rec, 이후 rec-only
  · slot_v4의 프레임별 이미지 픽셀 샘플링(bg/대비/채도) → slot_v3 선택(이미지 로드 0)
동작:
  ① 앞 window 프레임: full OCR(det+rec) → slot_v3로 채널 위치(ROI) 확정
  ② 이후 프레임: det 생략, 그 ROI만 crop→rec-only 로 직접 읽음
모델은 프로세스당 1번만 로드. 디스크 I/O 없음.
"""
from __future__ import annotations
import argparse, os, re, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
try:
    import torch  # noqa
except Exception:
    pass
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import slot_v3 as V3
from ocr_candidate_extractor import create_paddle_ocr, extract_candidates_from_image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def best_digit(t):
    r = [x for x in re.findall(r"\d+", str(t)) if 1 <= len(x) <= 5]
    return max(r, key=len) if r else re.sub(r"\D", "", str(t))[:5]


def gt_of(stem):
    tail = stem.split("__")[-1]
    m = re.match(r"(?i)ch?0*(\d+)", tail) or re.match(r"0*(\d+)", tail)
    return m.group(1) if m else None


def load_rec(model_dir):
    from paddleocr import TextRecognition
    if model_dir and Path(model_dir).exists():
        name = "en_PP-OCRv4_mobile_rec"
        y = Path(model_dir) / "inference.yml"
        if y.exists():
            try:
                import yaml
                name = yaml.safe_load(open(y)).get("Global", {}).get("model_name") or name
            except Exception:
                pass
        return TextRecognition(model_name=name, model_dir=str(model_dir))
    return TextRecognition(model_name="en_PP-OCRv4_mobile_rec")


def rec_roi(rec, im, box, tmp, pad=0.2, min_h=120):
    W, H = im.size
    x1, y1, x2, y2 = box
    pw = (x2 - x1) * pad; ph = (y2 - y1) * pad
    cx1, cy1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
    cx2, cy2 = min(W, int(x2 + pw)), min(H, int(y2 + ph))
    if cx2 - cx1 < 3 or cy2 - cy1 < 3:
        return ""
    crop = im.crop((cx1, cy1, cx2, cy2))
    u = max(1.0, min_h / max(1, crop.height))
    if u > 1.0:
        crop = crop.resize((max(1, int(crop.width * u)), max(1, int(crop.height * u))))
    cp = tmp / "c.png"; crop.convert("RGB").save(cp)
    r = rec.predict(str(cp))
    return best_digit(r[0].get("rec_text", "") if r else "")


def stream_folder(paths, ocr, rec, tmp, window=5):
    """한 시퀀스를 온디바이스처럼 순차 처리. 반환: {stem: (채널, mode, 지연ms)}."""
    frames, ids, out = [], [], {}
    box = None
    for k, p in enumerate(paths):
        im = Image.open(p).convert("RGB"); W, H = im.size
        stem = Path(p).stem
        t0 = time.perf_counter()
        if box is None:                                   # ── phase A: full OCR ──
            cands = extract_candidates_from_image(Path(p), ocr=ocr)
            frames.append({"image_id": stem, "image_width": W, "image_height": H,
                           "candidates": [c.to_dict() for c in cands]})
            ids.append(stem)
            if len(frames) >= window:                     # window 차면 ROI 확정
                r = V3.rolling_analyze(frames, ids, window=window)
                box = r["box"] if r else None
                pf = r["per_frame"] if r else {}
                for j, sid in enumerate(ids):             # phase A 프레임 예측 채움
                    out[sid] = (re.sub(r"\D", "", str(pf.get(sid, ""))),
                                "full", (time.perf_counter() - t0) * 1000 if j == len(ids) - 1 else None)
            else:
                out[stem] = ("", "full(buffer)", (time.perf_counter() - t0) * 1000)
        else:                                             # ── phase B: rec-only ──
            d = rec_roi(rec, im, box, tmp)
            out[stem] = (d, "rec-only", (time.perf_counter() - t0) * 1000)
    return out, box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="한 시퀀스(채널 배너) 이미지 폴더")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--rec-model-dir", default="models/full_image_ocr/en_PP-OCRv4_mobile_rec_ft")
    ap.add_argument("--gt", action="store_true", help="파일명에서 정답 뽑아 정확도 출력")
    args = ap.parse_args()

    paths = sorted(p for p in Path(args.root).iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"프레임 {len(paths)} · window {args.window} · (json 없음, 스트리밍)", flush=True)
    rec_dir = args.rec_model_dir if Path(args.rec_model_dir).exists() else None
    ocr = create_paddle_ocr(lang="en", use_gpu=True, ocr_version="PP-OCRv4",
                            text_detection_model_name="PP-OCRv4_mobile_det",
                            text_recognition_model_name="en_PP-OCRv4_mobile_rec",
                            text_recognition_model_dir=rec_dir)
    rec = load_rec(rec_dir)
    tmp = Path(tempfile.mkdtemp(prefix="v5dev_"))

    out, box = stream_folder(paths, ocr, rec, tmp, args.window)

    full_ms = [v[2] for v in out.values() if v[1].startswith("full") and v[2]]
    rec_ms = [v[2] for v in out.values() if v[1] == "rec-only" and v[2]]
    print(f"\nROI 락: {box and [round(x,1) for x in box]}")
    if full_ms:
        print(f"phase A (full OCR) 평균 지연: {sum(full_ms)/len(full_ms):.0f} ms/frame  ({len(full_ms)}프레임)")
    if rec_ms:
        print(f"phase B (rec-only) 평균 지연: {sum(rec_ms)/len(rec_ms):.0f} ms/frame  ({len(rec_ms)}프레임)")
        print(f"→ 프레임당 {sum(full_ms)/len(full_ms)/(sum(rec_ms)/len(rec_ms)):.1f}x 빨라짐 (det 생략)")
    if args.gt:
        c = t = 0
        for stem, (v, mode, _ms) in out.items():
            g = gt_of(stem)
            if not g:
                continue
            t += 1
            pv = re.sub(r"\D", "", str(v))
            if pv and str(int(pv)) == str(int(g)):
                c += 1
        print(f"\n정확도: {c}/{t} = {c/max(1,t)*100:.1f}%")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
