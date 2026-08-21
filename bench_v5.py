#!/usr/bin/env python3
"""v4 vs v5 벤치마크 — 정확도 + 속도 비교 (Test_overlay_folder).

v4: 매 프레임 full OCR(det+rec) → slot_v4.rolling_analyze 로 채널 선택.
v5: 폴더(시퀀스)마다
      · 앞 window(기본 5)프레임까지는 계속 full OCR → 그동안 클러스터(채널 위치) 확정
      · window가 가득 차면, 이후 프레임은 det 생략하고 '이전 cluster ROI'만 crop해서
        recognition 모델로 직접 읽음(rec-only). det를 안 돌리므로 훨씬 빠름.
    (Test_overlay는 폴더 내 채널 '값'은 프레임마다 다르지만 UI 디자인=채널 '위치'는 고정 →
     위치를 락하고 값만 매 프레임 rec 로 읽는 구조가 성립.)

속도는 동일 머신·동일 모델 인스턴스로 각 호출을 실측해 합산(공정 비교).
--scale-h 로 입력 해상도를 낮춰(예: 270) 성능 유지도 측정.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

try:
    import torch  # noqa: F401  (paddle/torch 로드 순서)
except Exception:
    pass

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import slot_v4 as V4
from ocr_candidate_extractor import create_paddle_ocr, extract_candidates_from_image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def best_digit(text):
    runs = [r for r in re.findall(r"\d+", str(text)) if 1 <= len(r) <= 5]
    if runs:
        return max(runs, key=len)
    return re.sub(r"\D", "", str(text))[:5]


def norm(s):
    s = str(s)
    return str(int(s)) if s.isdigit() and s != "" else s


def gt_of(stem):
    m = re.match(r"0*(\d+)", stem)
    return m.group(1) if m else None


def load_rec(model_dir):
    from paddleocr import TextRecognition
    name = "en_PP-OCRv4_mobile_rec"
    if model_dir and Path(model_dir).exists():
        y = Path(model_dir) / "inference.yml"
        if y.exists():
            try:
                import yaml
                n = yaml.safe_load(open(y)).get("Global", {}).get("model_name")
                if n:
                    name = n
            except Exception:
                pass
        return TextRecognition(model_name=name, model_dir=str(model_dir))
    return TextRecognition(model_name=name)


def img_dict(path, ocr, W, H):
    """full OCR 1프레임 → slot 입력용 dict + 소요시간."""
    t0 = time.perf_counter()
    cands = extract_candidates_from_image(Path(path), ocr=ocr)
    dt = time.perf_counter() - t0
    return {"image_id": Path(path).stem, "image_path": str(path),
            "image_width": W, "image_height": H,
            "candidates": [c.to_dict() for c in cands]}, dt


def rec_roi(rec, im, box, tmp, pad=0.2, min_h=120):
    """ROI crop(+pad, 확대) → rec-only 읽기. (digit, score, 소요시간)."""
    W, H = im.size
    x1, y1, x2, y2 = box
    pw = (x2 - x1) * pad; ph = (y2 - y1) * pad
    cx1, cy1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
    cx2, cy2 = min(W, int(x2 + pw)), min(H, int(y2 + ph))
    if cx2 - cx1 < 3 or cy2 - cy1 < 3:
        return "", 0.0, 0.0
    crop = im.crop((cx1, cy1, cx2, cy2))
    u = max(1.0, min_h / max(1, crop.height))
    if u > 1.0:
        crop = crop.resize((max(1, int(crop.width * u)), max(1, int(crop.height * u))))
    cp = tmp / "c.png"; crop.convert("RGB").save(cp)
    t0 = time.perf_counter()
    r = rec.predict(str(cp))
    dt = time.perf_counter() - t0
    txt = r[0].get("rec_text", "") if r else ""
    score = float(r[0].get("rec_score", 0.0) or 0.0) if r else 0.0
    return best_digit(txt), score, dt


def prep_images(paths, scale_h, tmp):
    """scale_h>0 이면 높이=scale_h 로 리사이즈한 임시본 경로 반환. (경로, W, H, PIL)."""
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if scale_h and im.height != scale_h:
            w = max(1, round(im.width * scale_h / im.height))
            im = im.resize((w, scale_h))
            sp = tmp / f"s_{Path(p).stem}.png"; im.save(sp)
            out.append((str(sp), im.width, im.height, im))
        else:
            out.append((str(p), im.width, im.height, im))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/irteam/teacher_model/dataset/Test_overlay_folder")
    ap.add_argument("--out", default="bench_v5_out")
    ap.add_argument("--window", type=int, default=5, help="v5 phase A(full OCR) 프레임 수")
    ap.add_argument("--v4-window", type=int, default=24, help="v4 롤링 통계 윈도우")
    ap.add_argument("--rec-model-dir", default="models/full_image_ocr/en_PP-OCRv4_mobile_rec_ft")
    ap.add_argument("--scale-h", type=int, default=0, help="입력 높이 리사이즈(0=원본). 예: 270")
    ap.add_argument("--folders", type=int, default=0, help="폴더 수 제한(0=전체)")
    ap.add_argument("--per-folder", type=int, default=0, help="폴더당 프레임 제한(0=전체)")
    args = ap.parse_args()

    root = Path(args.root)
    folders = sorted([d for d in root.iterdir() if d.is_dir()])
    if args.folders:
        folders = folders[:args.folders]

    print("모델 로딩...", flush=True)
    rec_dir = args.rec_model_dir if Path(args.rec_model_dir).exists() else None
    ocr = create_paddle_ocr(lang="en", use_gpu=True, ocr_version="PP-OCRv4",
                            text_detection_model_name="PP-OCRv4_mobile_det",
                            text_recognition_model_name="en_PP-OCRv4_mobile_rec",
                            text_recognition_model_dir=rec_dir)
    rec = load_rec(rec_dir)
    tmp = Path(tempfile.mkdtemp(prefix="benchv5_"))
    print(f"scale_h={args.scale_h or '원본'}  window={args.window}  폴더={len(folders)}", flush=True)

    # 집계
    agg = {"v4": [0, 0, 0], "v5": [0, 0, 0]}   # [total, read, correct]
    time_v4 = time_v5 = 0.0
    n_full = n_rec = 0
    per_ui = []

    for fi, fd in enumerate(folders):
        paths = sorted(p for p in fd.iterdir() if p.suffix.lower() in IMG_EXTS)
        if args.per_folder:
            paths = paths[:args.per_folder]
        if len(paths) < args.window + 1:
            continue
        prepped = prep_images(paths, args.scale_h, tmp)
        ids = [p.stem for p in paths]                 # id/gt는 항상 '원본' 파일명 기준
        gts = [gt_of(i) for i in ids]

        # ---- full OCR 전 프레임(=v4 입력, phase A 재사용) ----
        frames, ftimes = [], []
        for (p, W, H, _im) in prepped:
            d, dt = img_dict(p, ocr, W, H)
            frames.append(d); ftimes.append(dt); n_full += 1

        # ---- v4: 전 프레임 full OCR + rolling ----
        t0 = time.perf_counter()
        r4 = V4.rolling_analyze(frames, ids, window=args.v4_window)
        sel4 = time.perf_counter() - t0
        pf4 = r4["per_frame"] if r4 else {}
        t_v4 = sum(ftimes) + sel4
        time_v4 += t_v4

        # ---- v5: 앞 window 프레임 full OCR → 클러스터 락 → 이후 rec-only ----
        W_ = args.window
        t0 = time.perf_counter()
        rA = V4.rolling_analyze(frames[:W_], ids[:W_], window=W_)
        selA = time.perf_counter() - t0
        box = rA["box"] if rA else None
        pf5 = dict(rA["per_frame"]) if rA else {}
        t_v5 = sum(ftimes[:W_]) + selA
        rectimes = []
        if box is not None:
            for k in range(W_, len(prepped)):
                _p, _W, _H, im = prepped[k]
                d, sc, dt = rec_roi(rec, im, box, tmp)
                pf5[ids[k]] = d; rectimes.append(dt); n_rec += 1
        else:                                   # 클러스터 실패 → 폴백: 이후도 full OCR(속도이득 없음)
            for k in range(W_, len(prepped)):
                pf5[ids[k]] = re.sub(r"\D", "", str(pf4.get(ids[k], "")))
                rectimes.append(ftimes[k])
        t_v5 += sum(rectimes)
        time_v5 += t_v5

        # ---- 정확도 집계 ----
        u4 = [0, 0, 0]; u5 = [0, 0, 0]
        for i, g in zip(ids, gts):
            if not g:
                continue
            for tag, pf, u in (("v4", pf4, u4), ("v5", pf5, u5)):
                p = re.sub(r"\D", "", str(pf.get(i, "")))
                a = agg[tag]; a[0] += 1; u[0] += 1
                if p:
                    a[1] += 1; u[1] += 1
                    if norm(p) == norm(g):
                        a[2] += 1; u[2] += 1
        per_ui.append({"ui": fd.name, "n": u4[0],
                       "v4_acc": round(u4[2] / max(1, u4[0]) * 100, 1),
                       "v5_acc": round(u5[2] / max(1, u5[0]) * 100, 1),
                       "box": [round(v, 1) for v in box] if box else None,
                       "v4_t": round(t_v4, 3), "v5_t": round(t_v5, 3)})
        print(f"  {fd.name:<8} v4={per_ui[-1]['v4_acc']:5.1f}%  v5={per_ui[-1]['v5_acc']:5.1f}%  "
              f"v4_t={t_v4:.2f}s v5_t={t_v5:.2f}s  box={'O' if box else 'X'}", flush=True)

    def acc(a):
        return round(a[2] / max(1, a[0]) * 100, 1)

    outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    summary = {
        "scale_h": args.scale_h or "orig", "window": args.window, "folders": len(per_ui),
        "v4": {"e2e_acc": acc(agg["v4"]), "read_rate": round(agg["v4"][1] / max(1, agg["v4"][0]) * 100, 1),
               "total_time_s": round(time_v4, 2), "frames": agg["v4"][0]},
        "v5": {"e2e_acc": acc(agg["v5"]), "read_rate": round(agg["v5"][1] / max(1, agg["v5"][0]) * 100, 1),
               "total_time_s": round(time_v5, 2), "frames": agg["v5"][0]},
        "full_ocr_calls": n_full, "rec_only_calls": n_rec,
        "speedup": round(time_v4 / max(1e-9, time_v5), 2),
        "per_ui": per_ui,
    }
    (outd / f"summary_h{args.scale_h or 'orig'}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n" + "=" * 60)
    print(f"[scale_h={args.scale_h or '원본'}]  폴더 {len(per_ui)}개")
    print(f"  v4  정확도 {summary['v4']['e2e_acc']:5.1f}%  시간 {time_v4:7.1f}s  (full OCR {n_full}회)")
    print(f"  v5  정확도 {summary['v5']['e2e_acc']:5.1f}%  시간 {time_v5:7.1f}s  (full OCR {W_ if per_ui else 0}/폴더 + rec-only {n_rec}회)")
    print(f"  → 속도 {summary['speedup']}x 빠름,  정확도차 {summary['v5']['e2e_acc']-summary['v4']['e2e_acc']:+.1f}%p")
    print("=" * 60)

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
