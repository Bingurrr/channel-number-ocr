#!/usr/bin/env python3
"""v4 vs v5 vs v6 벤치마크 — 정확도 + 속도 (Test_overlay_folder).

v4: 매 프레임 full OCR(det+rec) → slot_v4.rolling_analyze.
v5: 앞 window 프레임만 full OCR로 ROI 락 → 이후 rec-only. (붕괴 위험: 초반 오락 시 폴더 전체 실패)
v6: v5 + ① 락 신뢰도 게이트  ② 주기적 재검증(자가교정).
     · 앞 window 프레임 full OCR → 누적 버퍼로 클러스터, coverage(신뢰도) 산출.
     · 신뢰도 >= conf_min 이면 rec-only 모드, 미만이면 full OCR 유지(=v4 폴백, 회귀 방지).
     · rec-only 모드에서도 recheck_every 프레임마다 full OCR 1장을 섞어 버퍼에 추가 →
       누적 버퍼(값 다양성↑)로 재클러스터 → ROI가 이동하면 다시 락(초반 오락 자가교정).
     · rec 점수가 아주 낮으면(min_score) 그 프레임은 즉시 full OCR로 재검증.
비용: full OCR = window + (100/recheck_every) ≈ 폴더당 ~10장(v4는 100장) → 여전히 크게 빠름.

속도는 동일 머신·동일 모델 인스턴스로 각 호출 실측 합산(공정). --scale-h 로 저해상 유지율 측정.
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
    import torch  # noqa: F401
except Exception:
    pass

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import slot_v4 as V4
from ocr_candidate_extractor import create_paddle_ocr, extract_candidates_from_image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def best_digit(text):
    runs = [r for r in re.findall(r"\d+", str(text)) if 1 <= len(r) <= 5]
    return max(runs, key=len) if runs else re.sub(r"\D", "", str(text))[:5]


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
    t0 = time.perf_counter()
    cands = extract_candidates_from_image(Path(path), ocr=ocr)
    dt = time.perf_counter() - t0
    return {"image_id": Path(path).stem, "image_path": str(path),
            "image_width": W, "image_height": H,
            "candidates": [c.to_dict() for c in cands]}, dt


def rec_roi(rec, im, box, tmp, pad=0.2, min_h=120):
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


def _box_far(a, b):
    """두 박스 중심 거리 / 평균 높이 > 0.6 이면 '다른 위치'로 판단(자가교정 트리거)."""
    if not a or not b:
        return True
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    h = max(1.0, (a[3] - a[1] + b[3] - b[1]) / 2)
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 / h > 0.6


def v6_stream(frames, ftimes, ids, prepped, rec, tmp, window=5, recheck_every=20,
              clust_window=40, conf_min=0.5, min_score=0.30):
    """온라인 컨트롤러: 프레임을 순서대로 처리하며 full OCR/rec-only를 동적으로 선택.
       반환: preds(dict), used_time(sec), stats(dict)."""
    preds = {}
    buf_idx = []                 # full OCR을 실제로 돌린 프레임 인덱스(누적 버퍼)
    locked = None; conf = 0.0
    full_used = []               # v6가 full OCR 시간으로 계상할 인덱스
    rectimes = []; clust_t = 0.0
    n_full = n_rec = n_recheck = n_selfheal = 0

    for k in range(len(frames)):
        # rec-only 모드 진입 조건: 락 있고 신뢰도 충분. 아니면 full OCR.
        recly = (locked is not None) and (conf >= conf_min) and (k >= window)
        periodic = recly and (recheck_every > 0) and (k % recheck_every == 0)
        do_full = (not recly) or periodic

        if do_full:
            buf_idx.append(k); full_used.append(k); n_full += 1
            if periodic:
                n_recheck += 1
            sub_idx = buf_idx[-clust_window:]
            t0 = time.perf_counter()
            r = V4.rolling_analyze([frames[j] for j in sub_idx], [ids[j] for j in sub_idx],
                                   window=max(2, min(len(sub_idx), 24)))
            clust_t += time.perf_counter() - t0
            new_box = r["box"] if r else None
            new_conf = r["score"] if r else 0.0            # coverage = 신뢰도
            if new_box is not None:
                if locked is not None and _box_far(locked, new_box) and new_conf >= conf:
                    n_selfheal += 1                        # ROI가 이동 → 자가교정 재락
                # 더 신뢰도 높은 클러스터로 갱신(초반 오락 교정)
                if new_conf >= conf or locked is None:
                    locked = new_box; conf = new_conf
            # 이 프레임 예측: 클러스터가 뽑은 값(정식 선택) 우선, 없으면 락 ROI rec-only
            pv = r["per_frame"].get(ids[k]) if r else None
            if pv:
                preds[ids[k]] = re.sub(r"\D", "", str(pv))
            elif locked is not None:
                d, sc, dt = rec_roi(rec, prepped[k][3], locked, tmp)
                preds[ids[k]] = d; rectimes.append(dt); n_rec += 1
            else:
                preds[ids[k]] = ""
        else:
            d, sc, dt = rec_roi(rec, prepped[k][3], locked, tmp)
            rectimes.append(dt); n_rec += 1
            if sc < min_score:                              # 저신뢰 읽기 → 즉시 full 재검증
                buf_idx.append(k); full_used.append(k); n_full += 1
                sub_idx = buf_idx[-clust_window:]
                t0 = time.perf_counter()
                r = V4.rolling_analyze([frames[j] for j in sub_idx], [ids[j] for j in sub_idx],
                                       window=max(2, min(len(sub_idx), 24)))
                clust_t += time.perf_counter() - t0
                if r and r["box"] is not None and r["score"] >= conf:
                    locked = r["box"]; conf = r["score"]
                pv = r["per_frame"].get(ids[k]) if r else None
                preds[ids[k]] = re.sub(r"\D", "", str(pv)) if pv else d
            else:
                preds[ids[k]] = d

    used_time = sum(ftimes[j] for j in full_used) + sum(rectimes) + clust_t
    stats = {"n_full": n_full, "n_rec": n_rec, "n_recheck": n_recheck,
             "n_selfheal": n_selfheal, "conf": round(conf, 3),
             "locked": [round(v, 1) for v in locked] if locked else None}
    return preds, used_time, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/irteam/teacher_model/dataset/Test_overlay_folder")
    ap.add_argument("--out", default="bench_v6_out")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--v4-window", type=int, default=24)
    ap.add_argument("--recheck-every", type=int, default=20)
    ap.add_argument("--conf-min", type=float, default=0.5)
    ap.add_argument("--min-score", type=float, default=0.30)
    ap.add_argument("--rec-model-dir", default="models/full_image_ocr/en_PP-OCRv4_mobile_rec_ft")
    ap.add_argument("--scale-h", type=int, default=0)
    ap.add_argument("--folders", type=int, default=0)
    ap.add_argument("--per-folder", type=int, default=0)
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
    tmp = Path(tempfile.mkdtemp(prefix="benchv6_"))
    print(f"scale_h={args.scale_h or '원본'}  window={args.window}  recheck={args.recheck_every}  "
          f"conf_min={args.conf_min}  폴더={len(folders)}", flush=True)

    agg = {"v4": [0, 0, 0], "v5": [0, 0, 0], "v6": [0, 0, 0]}
    tt = {"v4": 0.0, "v5": 0.0, "v6": 0.0}
    n_full_v4 = n_rec_v5 = n_full_v6 = n_rec_v6 = n_selfheal = 0
    per_ui = []

    for fd in folders:
        paths = sorted(p for p in fd.iterdir() if p.suffix.lower() in IMG_EXTS)
        if args.per_folder:
            paths = paths[:args.per_folder]
        if len(paths) < args.window + 1:
            continue
        prepped = prep_images(paths, args.scale_h, tmp)
        ids = [p.stem for p in paths]
        gts = [gt_of(i) for i in ids]

        frames, ftimes = [], []
        for (p, W, H, _im) in prepped:
            d, dt = img_dict(p, ocr, W, H)
            frames.append(d); ftimes.append(dt); n_full_v4 += 1

        # v4
        t0 = time.perf_counter()
        r4 = V4.rolling_analyze(frames, ids, window=args.v4_window)
        sel4 = time.perf_counter() - t0
        pf4 = r4["per_frame"] if r4 else {}
        t_v4 = sum(ftimes) + sel4; tt["v4"] += t_v4

        # v5
        W_ = args.window
        t0 = time.perf_counter()
        rA = V4.rolling_analyze(frames[:W_], ids[:W_], window=W_)
        selA = time.perf_counter() - t0
        box = rA["box"] if rA else None
        pf5 = dict(rA["per_frame"]) if rA else {}
        t_v5 = sum(ftimes[:W_]) + selA; rt5 = []
        if box is not None:
            for k in range(W_, len(prepped)):
                d, sc, dt = rec_roi(rec, prepped[k][3], box, tmp)
                pf5[ids[k]] = d; rt5.append(dt); n_rec_v5 += 1
        else:
            for k in range(W_, len(prepped)):
                pf5[ids[k]] = re.sub(r"\D", "", str(pf4.get(ids[k], ""))); rt5.append(ftimes[k])
        t_v5 += sum(rt5); tt["v5"] += t_v5

        # v6
        pf6, t_v6, st6 = v6_stream(frames, ftimes, ids, prepped, rec, tmp,
                                   window=W_, recheck_every=args.recheck_every,
                                   conf_min=args.conf_min, min_score=args.min_score)
        tt["v6"] += t_v6
        n_full_v6 += st6["n_full"]; n_rec_v6 += st6["n_rec"]; n_selfheal += st6["n_selfheal"]

        u = {"v4": [0, 0, 0], "v5": [0, 0, 0], "v6": [0, 0, 0]}
        for i, g in zip(ids, gts):
            if not g:
                continue
            for tag, pf in (("v4", pf4), ("v5", pf5), ("v6", pf6)):
                p = re.sub(r"\D", "", str(pf.get(i, "")))
                agg[tag][0] += 1; u[tag][0] += 1
                if p:
                    agg[tag][1] += 1; u[tag][1] += 1
                    if norm(p) == norm(g):
                        agg[tag][2] += 1; u[tag][2] += 1
        a = lambda x: round(x[2] / max(1, x[0]) * 100, 1)
        per_ui.append({"ui": fd.name, "n": u["v4"][0], "v4": a(u["v4"]), "v5": a(u["v5"]),
                       "v6": a(u["v6"]), "v4_t": round(t_v4, 3), "v5_t": round(t_v5, 3),
                       "v6_t": round(t_v6, 3), "v6_full": st6["n_full"], "v6_heal": st6["n_selfheal"]})
        print(f"  {fd.name:<8} v4={a(u['v4']):5.1f} v5={a(u['v5']):5.1f} v6={a(u['v6']):5.1f}  "
              f"t v4={t_v4:.1f} v5={t_v5:.2f} v6={t_v6:.2f}  v6full={st6['n_full']} heal={st6['n_selfheal']}", flush=True)

    a = lambda x: round(x[2] / max(1, x[0]) * 100, 1)
    rr = lambda x: round(x[1] / max(1, x[0]) * 100, 1)
    outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    summary = {"scale_h": args.scale_h or "orig", "folders": len(per_ui),
               "window": args.window, "recheck_every": args.recheck_every, "conf_min": args.conf_min}
    for tag in ("v4", "v5", "v6"):
        summary[tag] = {"e2e_acc": a(agg[tag]), "read_rate": rr(agg[tag]),
                        "total_time_s": round(tt[tag], 2), "frames": agg[tag][0]}
    summary["v6_full_ocr_calls"] = n_full_v6
    summary["v6_rec_calls"] = n_rec_v6
    summary["v6_selfheal"] = n_selfheal
    summary["speedup_v5"] = round(tt["v4"] / max(1e-9, tt["v5"]), 2)
    summary["speedup_v6"] = round(tt["v4"] / max(1e-9, tt["v6"]), 2)
    summary["per_ui"] = per_ui
    (outd / f"summary_h{args.scale_h or 'orig'}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n" + "=" * 64)
    print(f"[scale_h={args.scale_h or '원본'}]  폴더 {len(per_ui)}개")
    for tag in ("v4", "v5", "v6"):
        s = summary[tag]
        print(f"  {tag}: 정확도 {s['e2e_acc']:5.1f}%  읽음 {s['read_rate']:5.1f}%  시간 {s['total_time_s']:7.1f}s")
    print(f"  full OCR: v4 {n_full_v4} · v5 {len(per_ui)*W_} · v6 {n_full_v6}  (rec-only: v5 {n_rec_v5} · v6 {n_rec_v6})")
    print(f"  속도: v5 {summary['speedup_v5']}x · v6 {summary['speedup_v6']}x   자가교정 {n_selfheal}회")
    print("=" * 64)

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
