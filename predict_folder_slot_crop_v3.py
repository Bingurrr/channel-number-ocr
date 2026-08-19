#!/usr/bin/env python3
"""predict_folder_slot_crop_v3 — 온보드용 'ROI 밖 화질 저하' 시뮬레이터 + 평가.

목적: 온보드에서 매 프레임 전체화면을 고화질로 det에 넣을 여유가 없다. 그런데
채널번호는 화면의 특정 영역에만 뜬다. 그래서:

  1) 워밍업(--warmup N프레임): 원본 화질로 돌려서 slot_v3가 채널 위치(ROI)를 학습
  2) 이후 프레임: ROI 근처만 원본 화질 유지, **나머지는 세로 --degrade-height(기본 270)
     로 다운스케일 후 원래 크기로 복원**(=화질만 떨어뜨리고 기하는 보존)
  3) 그 상태로 OCR → 전체 프레임을 합쳐 롤링 선택 → 정확도

기하(좌표계)를 원본 그대로 두기 때문에 GT/시각화/박스가 전부 그대로 맞는다.
'ROI 밖을 얼마나 버려도 되는가'를 정확도 숫자로 확인하는 것이 이 스크립트의 목적.

예:
  python predict_folder_slot_crop_v3.py \
      --root /path/to/frames --out results/crop_run \
      --det-model-dir models/full_image_ocr/det_overlay_frozen_v1 \
      --rec-model-dir models/full_image_ocr/rec_overlay_frozen_v1 \
      --warmup 24 --degrade-height 270 --roi-margin 0.15 \
      --gt-from-filename --viz-degrade 5 --viz-steps 10
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

import predict_folder as P
import slot_v3 as V3
from predict_folder_slot_v3 import gt_of, _safe


# ───────────────────────── 화질 저하 ─────────────────────────

def _expand(box, margin, W, H):
    """ROI를 margin 비율만큼 넓힘 (채널값이 박스 경계에 걸치는 걸 방지)."""
    x1, y1, x2, y2 = [float(v) for v in box]
    mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
    return (max(0, int(x1 - mw)), max(0, int(y1 - mh)),
            min(W, int(x2 + mw)), min(H, int(y2 + mh)))


def degrade_outside_roi(src, dst, boxes, deg_h, margin, quality):
    """ROI(들) 밖을 deg_h 세로해상도로 낮춘 뒤 원래 크기로 되돌려 저장.

    boxes 가 비어 있으면(=ROI 미학습) 전체를 저하시킨다. 반환: 원본 대비 보존 면적 비율.
    """
    from PIL import Image
    im = Image.open(src).convert("RGB")
    W, H = im.size
    if deg_h >= H:                                   # 이미 저해상도면 손대지 않음
        im.save(dst, quality=quality)
        return 1.0
    sw = max(1, int(round(W * deg_h / H)))
    deg = im.resize((sw, deg_h), Image.BILINEAR).resize((W, H), Image.BILINEAR)
    kept = 0
    for b in boxes or []:
        x1, y1, x2, y2 = _expand(b, margin, W, H)
        if x2 <= x1 or y2 <= y1:
            continue
        deg.paste(im.crop((x1, y1, x2, y2)), (x1, y1))
        kept += (x2 - x1) * (y2 - y1)
    deg.save(dst, quality=quality)
    return round(min(1.0, kept / float(W * H)), 4)


# ───────────────────────── OCR 실행 ─────────────────────────

def _resolve_model_dir(v):
    v = str(v or "").strip()
    if v.lower() in ("", "none"):
        return None
    p = Path(v)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return p if (p / "inference.pdiparams").exists() else None


def run_ocr(PY, SRC, images_dir, out_json, args, env, tag):
    """staged 이미지 폴더에 full OCR 실행. 소요시간(초) 반환."""
    cmd = [PY, f"{SRC}/run_paddleocr_export.py", "--images", images_dir, "--out", out_json,
           "--use-gpu", "--ocr-version", "PP-OCRv4",
           "--text-detection-model-name", "PP-OCRv4_mobile_det",
           "--text-recognition-model-name", "en_PP-OCRv4_mobile_rec",
           "--progress-every", 200]
    rdp = _resolve_model_dir(args.rec_model_dir)
    if rdp:
        cmd += ["--text-recognition-model-dir", str(rdp)]
    ddp = _resolve_model_dir(args.det_model_dir)
    if ddp:
        cmd += ["--text-detection-model-dir", str(ddp)]
    print(f"[{tag}] OCR 시작 ({len(list(Path(images_dir).glob('*')))}장)", flush=True)
    t0 = time.time()
    if P.sh(cmd, env) != 0:
        raise SystemExit(f"[{tag}] full OCR 실패")
    dt = time.time() - t0
    print(f"[{tag}] OCR 완료 {dt:.1f}s", flush=True)
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symlink", action="store_true")
    ap.add_argument("--gt-from-filename", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--by-height", action="store_true")
    ap.add_argument("--band", type=float, default=0.05)
    ap.add_argument("--rec-model-dir", default="models/full_image_ocr/en_PP-OCRv4_mobile_rec_ft")
    ap.add_argument("--det-model-dir", default="")
    ap.add_argument("--keep-staged", action="store_true")
    ap.add_argument("--no-qualitative", action="store_true")
    ap.add_argument("--samples-per-folder", type=int, default=20)
    ap.add_argument("--viz-steps", type=int, default=0, metavar="N")
    # ── crop_v3 전용 ──
    ap.add_argument("--warmup", type=int, default=24, metavar="N",
                    help="폴더당 앞 N프레임은 원본 화질로 돌려 ROI를 학습 (기본 24 = --window와 동일)")
    ap.add_argument("--degrade-height", type=int, default=270, metavar="H",
                    help="ROI 밖 영역을 낮출 세로 해상도 (기본 270). 원래 크기로 되돌려 기하는 보존")
    ap.add_argument("--roi-margin", type=float, default=0.15,
                    help="학습된 ROI를 넓힐 비율 (기본 0.15 = 상하좌우 15%%)")
    ap.add_argument("--degrade-quality", type=int, default=88,
                    help="저하 프레임 JPEG 품질 (기본 88)")
    ap.add_argument("--viz-degrade", type=int, default=0, metavar="N",
                    help="폴더당 N장: 실제 모델에 들어간 저하 프레임을 ROI 박스와 함께 저장")
    ap.add_argument("--keep-degraded", action="store_true",
                    help="저하 프레임 staging 폴더 유지 (기본은 삭제)")
    args = ap.parse_args()

    cfg = P.load_config()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/opt/conda/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    PY, SRC = cfg["python"], cfg["pipeline_src"]

    # ── 원본 staging (v3와 동일) ──
    flat = out / "images"
    index, meta, seqs = P.collect(root, flat, args.symlink)
    if not index:
        raise SystemExit(f"이미지 없음: {root}")
    print(f"images: {len(index)}  folders: {len(seqs)}", flush=True)
    staged = {p.stem: p for p in flat.iterdir() if p.is_file()}

    # ── 폴더별로 워밍업/본선 분할 ──
    warm_uids, rest_uids = [], {}
    for g, uids in sorted(seqs.items()):
        su = sorted(uids)
        warm_uids += su[:args.warmup]
        rest_uids[g] = su[args.warmup:]
    print(f"[split] 워밍업 {len(warm_uids)}장(폴더당 앞 {args.warmup}) / "
          f"저하대상 {sum(len(v) for v in rest_uids.values())}장", flush=True)

    # ═══ PASS A: 원본 화질로 ROI 학습 ═══
    warm_dir = out / "stage_warm"
    shutil.rmtree(warm_dir, ignore_errors=True)
    warm_dir.mkdir(parents=True, exist_ok=True)
    for uid in warm_uids:
        src = staged.get(uid)
        if src:
            os.symlink(src.resolve(), warm_dir / src.name)
    t_warm = run_ocr(PY, SRC, warm_dir, out / "full_ocr_warm.json", args, env, "warmup")
    warm_by_id = {im["image_id"]: im
                  for im in json.loads((out / "full_ocr_warm.json").read_text()).get("images", [])}

    roi_by_folder = {}
    for g, uids in sorted(seqs.items()):
        ok = [u for u in sorted(uids)[:args.warmup] if u in warm_by_id]
        if not ok:
            continue
        pr = V3.rolling_analyze([warm_by_id[u] for u in ok], ok, window=args.window,
                                by_height=args.by_height, band=args.band, conf_thr=args.min_conf)
        if not pr:
            print(f"  [warmup] {g:<38} ROI 학습 실패 → 이 폴더는 저하 안 함", flush=True)
            continue
        boxes = list(pr.get("group_boxes") or [])
        if pr.get("box") and pr["box"] not in boxes:
            boxes.append(pr["box"])
        roi_by_folder[g] = boxes
        print(f"  [warmup] {g:<38} ROI {len(boxes)}곳 box={pr['box']}", flush=True)
    print(f"[warmup] ROI 학습 성공 {len(roi_by_folder)}/{len(seqs)} 폴더", flush=True)

    # ═══ PASS B: ROI 밖 화질 저하 후 OCR ═══
    deg_dir = out / "stage_degraded"
    shutil.rmtree(deg_dir, ignore_errors=True)
    deg_dir.mkdir(parents=True, exist_ok=True)
    keep_ratios, n_deg = [], 0
    for g, uids in sorted(rest_uids.items()):
        boxes = roi_by_folder.get(g, [])
        for uid in uids:
            src = staged.get(uid)
            if not src:
                continue
            try:
                r = degrade_outside_roi(src, deg_dir / f"{uid}.jpg", boxes,
                                        args.degrade_height, args.roi_margin, args.degrade_quality)
            except Exception:
                continue
            keep_ratios.append(r); n_deg += 1
        if n_deg and n_deg % 1000 < len(uids):
            print(f"  [degrade] {n_deg}장 처리", flush=True)
    avg_keep = round(sum(keep_ratios) / len(keep_ratios), 4) if keep_ratios else 0.0
    print(f"[degrade] {n_deg}장 생성 (세로 {args.degrade_height}px 저하, "
          f"원본화질 보존 면적 평균 {avg_keep*100:.2f}%)", flush=True)

    t_deg = run_ocr(PY, SRC, deg_dir, out / "full_ocr_degraded.json", args, env, "degraded") if n_deg else 0.0
    deg_by_id = ({im["image_id"]: im
                  for im in json.loads((out / "full_ocr_degraded.json").read_text()).get("images", [])}
                 if n_deg else {})

    # ── 합치기: 워밍업(원본) + 본선(저하) ──
    by_id = dict(warm_by_id)
    by_id.update(deg_by_id)
    (out / "full_ocr.json").write_text(json.dumps({"images": list(by_id.values())}, ensure_ascii=False))

    # ═══ 롤링 선택 (v3와 동일, 단 입력이 혼합 화질) ═══
    rows, report, pred_by_uid, per_folder = [], [], {}, {}
    for g, uids in sorted(seqs.items()):
        ok = [u for u in sorted(uids) if u in by_id]
        if not ok:
            continue
        primary = V3.rolling_analyze([by_id[u] for u in ok], ok, window=args.window,
                                     by_height=args.by_height, band=args.band, conf_thr=args.min_conf)
        pf = primary["per_frame"] if primary else {}
        box = primary["box"] if primary else None
        report.append({"folder": g, "channel_field_box": box,
                       "warmup_roi": roi_by_folder.get(g, []),
                       "distinct_values": primary["distinct"] if primary else 0,
                       "coverage": round(len(pf) / max(1, len(ok)), 3)})
        per_folder[g] = (box, pf, primary.get("group_boxes", []) if primary else [])
        for uid, v in sorted(pf.items()):
            rows.append({"folder": g, "frame": meta.get(uid, uid), "channel_number": v,
                         "degraded": int(uid in deg_by_id)})
            pred_by_uid[uid] = v
        print(f"  {g:<40} box={box} cov={report[-1]['coverage']}", flush=True)

    with (out / "per_frame.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "frame", "channel_number", "degraded"])
        w.writeheader(); w.writerows(rows)
    (out / "profile_report.json").write_text(json.dumps(
        {"degrade_height": args.degrade_height, "roi_margin": args.roi_margin,
         "warmup": args.warmup, "avg_kept_area_ratio": avg_keep,
         "ocr_seconds": {"warmup": round(t_warm, 1), "degraded": round(t_deg, 1)},
         "folders": report}, ensure_ascii=False, indent=2))

    # ═══ 저하 프레임 미리보기 (실제 모델이 본 그림) ═══
    if args.viz_degrade > 0 and n_deg:
        try:
            from PIL import Image, ImageDraw
        except Exception:
            Image = None
        if Image is not None:
            vd = 0
            for g, uids in sorted(rest_uids.items()):
                pick = uids[::max(1, len(uids) // max(1, args.viz_degrade))][:args.viz_degrade]
                dd = out / "degrade_preview" / _safe(g, root)
                dd.mkdir(parents=True, exist_ok=True)
                for uid in pick:
                    p = deg_dir / f"{uid}.jpg"
                    if not p.exists():
                        continue
                    img = Image.open(p).convert("RGB")
                    W, H = img.size
                    d = ImageDraw.Draw(img)
                    for b in roi_by_folder.get(g, []):
                        d.rectangle(_expand(b, args.roi_margin, W, H), outline=(0, 255, 255), width=2)
                    d.text((10, 10), f"outside ROI -> {args.degrade_height}p", fill=(0, 255, 255))
                    img.save(dd / f"{meta.get(uid, uid)}.jpg", quality=90)
                    vd += 1
            print(f"[viz-degrade] {vd}장 → {out}/degrade_preview/ (청록=원본화질 보존 영역)", flush=True)

    # ═══ 정성 이미지 (v3와 동일) ═══
    if not args.no_qualitative:
        try:
            from PIL import Image, ImageDraw
        except Exception:
            Image = None
        if Image is not None:
            norm = lambda s: str(int(s)) if str(s).isdigit() else str(s)
            saved = 0
            for g, (box, pf, _gb) in per_folder.items():
                if not box:
                    continue
                ib = [int(v) for v in box]
                allu = [u for u in sorted(seqs.get(g, [])) if u in by_id]
                oks, fails = [], []
                for uid in allu:
                    pred = P._dg(pf.get(uid, ""))
                    if args.gt_from_filename:
                        gt = gt_of(meta.get(uid, uid))
                        isf = (not pred) or (gt and norm(pred) != norm(gt))
                    else:
                        isf = not pred
                    (fails if isf else oks).append(uid)
                step = max(1, len(oks) // max(1, args.samples_per_folder))
                draw = ([(u, False) for u in oks[::step][:args.samples_per_folder]]
                        + [(u, True) for u in fails])
                if not draw:
                    continue
                qd = out / "qualitative" / _safe(g, root); fdir = qd / "_failures"
                qd.mkdir(parents=True, exist_ok=True)
                if fails:
                    fdir.mkdir(parents=True, exist_ok=True)
                for uid, isf in draw:
                    try:
                        img = Image.open(by_id[uid].get("image_path")).convert("RGB")
                    except Exception:
                        continue
                    d = ImageDraw.Draw(img)
                    val = pf.get(uid, "") or "(none)"
                    col = (255, 0, 0) if isf else (0, 200, 0)
                    d.rectangle(ib, outline=col, width=3)
                    lab = f"ch:{val}" + ("[deg]" if uid in deg_by_id else "[orig]")
                    if isf and args.gt_from_filename:
                        lab += f" (gt:{gt_of(meta.get(uid, uid))})"
                    d.text((ib[0], max(0, ib[1] - 14)), lab, fill=col)
                    img.save((fdir if isf else qd) / f"{meta.get(uid, uid)}.jpg", quality=88)
                    saved += 1
            print(f"정성 이미지 {saved}장 ([deg]=저하 프레임, [orig]=워밍업 원본)", flush=True)

    # ═══ step-viz (v3와 동일) ═══
    if args.viz_steps > 0:
        try:
            from PIL import Image, ImageDraw
        except Exception:
            Image = None
        if Image is not None:
            norm = lambda s: str(int(s)) if str(s).isdigit() else str(s)
            sv = 0
            for g, (box, pf, gboxes) in per_folder.items():
                allu = [u for u in sorted(seqs.get(g, [])) if u in by_id]
                oks, fails = [], []
                for uid in allu:
                    pred = P._dg(pf.get(uid, ""))
                    gt = gt_of(meta.get(uid, uid)) if args.gt_from_filename else None
                    isf = (not pred) or (gt and norm(pred) != norm(gt))
                    (fails if isf else oks).append(uid)
                step = max(1, len(oks) // max(1, args.viz_steps))
                pick = ([(u, True) for u in fails[:args.viz_steps * 2]]
                        + [(u, False) for u in oks[::step][:args.viz_steps]])
                sd = out / "step_viz" / _safe(g, root); fd = sd / "_failures"
                sd.mkdir(parents=True, exist_ok=True)
                if fails:
                    fd.mkdir(parents=True, exist_ok=True)
                for uid, isf in pick:
                    im = by_id[uid]
                    try:
                        img = Image.open(im.get("image_path")).convert("RGB")
                    except Exception:
                        continue
                    d = ImageDraw.Draw(img)
                    for c in V3.preprocess_frame(im, args.min_conf):
                        b = [int(x) for x in c["box"]]
                        d.rectangle(b, outline=(60, 120, 255), width=1)
                        d.text((b[0], max(0, b[1] - 11)), c["value"], fill=(90, 160, 255))
                    for gb in gboxes:
                        d.rectangle([int(x) for x in gb], outline=(255, 210, 0), width=2)
                    val = pf.get(uid, "") or "(none)"
                    gt = gt_of(meta.get(uid, uid)) if args.gt_from_filename else None
                    col = (255, 40, 40) if isf else (0, 230, 0)
                    lab = f"CH={val}" + (f" gt={gt}" if isf and gt else "")
                    lab += " [deg]" if uid in deg_by_id else " [orig]"
                    d.text((10, 10), lab, fill=col)
                    if box:
                        d.rectangle([int(x) for x in box], outline=col, width=2)
                    img.save((fd if isf else sd) / f"{meta.get(uid, uid)}.jpg", quality=85)
                    sv += 1
            print(f"[step-viz] {sv}장 → {out}/step_viz/", flush=True)

    # ═══ 정확도: 전체 / 워밍업(원본) / 본선(저하) 분리 ═══
    if args.gt_from_filename:
        norm = lambda s: str(int(s)) if str(s).isdigit() else str(s)
        per, split = {}, {"orig": [0, 0, 0], "deg": [0, 0, 0]}
        for g, uids in seqs.items():
            for uid in uids:
                if uid not in by_id:
                    continue
                gt = gt_of(meta.get(uid, uid))
                if not gt:
                    continue
                t = per.setdefault(g, [0, 0, 0]); t[0] += 1
                s = split["deg" if uid in deg_by_id else "orig"]; s[0] += 1
                pr = P._dg(pred_by_uid.get(uid, ""))
                if pr:
                    t[1] += 1; s[1] += 1
                    if norm(pr) == norm(gt):
                        t[2] += 1; s[2] += 1
        tot = sum(v[0] for v in per.values())
        rdn = sum(v[1] for v in per.values())
        co = sum(v[2] for v in per.values())
        pc = lambda a, b: round(a / b * 100, 1) if b else 0
        print("\n=== 폴더별 정확도 (crop_v3) ===")
        print(f"  {'folder':<40}{'e2e%':>7}{'cov%':>7}{'읽었을때%':>10}{'correct/total':>16}")
        for g in sorted(per):
            a, b, c = per[g]
            print(f"  {g:<40}{pc(c,a):>7}{pc(b,a):>7}{pc(c,b):>10}{f'{c}/{a}':>16}")
        print(f"  {'-'*40}\n  {'전체':<40}{pc(co,tot):>7}{pc(rdn,tot):>7}{pc(co,rdn):>10}{f'{co}/{tot}':>16}")
        print("\n=== 화질별 비교 (이게 핵심) ===")
        for k, lab in (("orig", f"원본(워밍업 앞{args.warmup})"), ("deg", f"저하({args.degrade_height}p)")):
            a, b, c = split[k]
            print(f"  {lab:<28}{pc(c,a):>7}{pc(b,a):>7}{pc(c,b):>10}{f'{c}/{a}':>16}")
        print(f"\n  ROI 밖 저하로 보존한 면적: 평균 {avg_keep*100:.2f}% "
              f"(나머지는 {args.degrade_height}p로 저하)")
        print(f"  OCR 소요: 워밍업 {t_warm:.1f}s / 저하 {t_deg:.1f}s")

    print(f"\n결과: {out}/per_frame.csv, profile_report.json ({len(rows)}개)", flush=True)
    if not args.keep_staged:
        shutil.rmtree(flat, ignore_errors=True)
        shutil.rmtree(warm_dir, ignore_errors=True)
    if not args.keep_degraded and not args.keep_staged:
        shutil.rmtree(deg_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
