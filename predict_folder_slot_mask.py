#!/usr/bin/env python3
"""Channel inference (enhanced slot analysis) + COLOR-MASK re-read for overlap.

Same pipeline as predict_folder_slot.py, but the ROI re-read step (for frames the
first full OCR could not read) uses color masking instead of a plain crop:

  1. slot_analysis finds the channel ROI and reads most frames.
  2. From the frames it read cleanly, learn the channel FONT COLOR per folder
     (the value changes each zap but the color is constant).
  3. For each unread frame, crop the known ROI, keep only pixels near that color
     (removes an overlapping program/broadcaster text of a different color), upscale,
     and OCR -> recover the digit that was buried under the overlap.

--force-read turns the re-read on. --viz-mask N saves the per-step mask images
(ROI -> crop -> learned color -> masked -> read) so you can see the overlap removed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics as st
from pathlib import Path

import predict_folder as P
import slot_analysis as SA
from predict_folder_temporal_full import recover_field_values
import step_viz


def _safe(name, root):
    return name.replace("/", "__").replace(" ", "_") or root.name


def learn_folder_colors(by_id, seqs, per_folder):
    """uid -> [r,g,b] for UNREAD frames, using the channel color learned from the
    same folder's cleanly-read frames. Returns {} if PIL/color_mask unavailable."""
    try:
        from PIL import Image
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
        import color_mask as CM
    except Exception as e:
        print(f"[mask] 색 학습 불가({e}) → 마스킹 없이 폴백", flush=True)
        return {}
    colors = {}
    for g, (entry, field) in per_folder.items():
        if not field:
            continue
        box = field["median_box"]
        cols = []
        for uid in list(field["per_frame"]):          # 읽힌 프레임에서 색 학습
            im = by_id.get(uid)
            if not im or not im.get("image_path"):
                continue
            try:
                img = Image.open(im["image_path"]).convert("RGB")
            except Exception:
                continue
            crop = CM.crop_roi(img, box, 0.15)
            if crop is None:
                continue
            c = CM.learn_text_color(crop)
            if c:
                cols.append(c)
            if len(cols) >= 15:
                break
        if not cols:
            continue
        med = [int(st.median([c[i] for c in cols])) for i in range(3)]
        for uid in seqs.get(g, []):                    # 못 읽은 프레임에 폴더 색 부여
            if uid in by_id and uid not in field["per_frame"]:
                colors[uid] = med
    return colors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symlink", action="store_true")
    ap.add_argument("--gt-from-filename", action="store_true")
    ap.add_argument("--split-output", action="store_true")
    ap.add_argument("--samples-per-folder", type=int, default=20)
    ap.add_argument("--no-qualitative", action="store_true")
    ap.add_argument("--force-read", action="store_true",
                    help="못읽은 프레임의 확정 ROI를 색마스킹 후 재읽기 (겹침 복구)")
    ap.add_argument("--mask-tol", type=int, default=70, help="색 거리 허용치")
    ap.add_argument("--viz-mask", type=int, default=0, metavar="N",
                    help="마스킹 스텝 시각화(폴더당 N장): ROI→crop→색→마스킹→읽기")
    ap.add_argument("--viz-steps", type=int, default=0, metavar="N")
    ap.add_argument("--keep-staged", action="store_true")
    args = ap.parse_args()

    cfg = P.load_config()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/opt/conda/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    PY, SRC = cfg["python"], cfg["pipeline_src"]

    flat = out / "images"
    index, meta, seqs = P.collect(root, flat, args.symlink)
    if not index:
        raise SystemExit(f"이미지 없음: {root}")
    print(f"images: {len(index)}  folders(sequences): {len(seqs)}", flush=True)

    # === 유일한 모델: FULL OCR ===
    rc = P.sh([PY, f"{SRC}/run_paddleocr_export.py", "--images", flat, "--out", out / "full_ocr.json",
               "--use-gpu", "--ocr-version", "PP-OCRv4", "--text-detection-model-name",
               "PP-OCRv4_mobile_det", "--text-recognition-model-name", "en_PP-OCRv4_mobile_rec",
               "--progress-every", 200], env)
    if rc != 0:
        raise SystemExit(f"[slot-mask] full OCR 실패 (rc={rc})")
    cand = json.loads((out / "full_ocr.json").read_text())
    by_id = {im["image_id"]: im for im in cand.get("images", [])}

    # === 강화된 슬롯 분석 -> 채널 필드(+듀얼) ===
    report, per_folder = [], {}
    for g, uids in sorted(seqs.items()):
        ok_uids = [u for u in sorted(uids) if u in by_id]
        frames = [by_id[u] for u in ok_uids]
        if not frames:
            continue
        primary, duals, allm = SA.analyze(frames, ok_uids)
        if primary:
            merged = dict(primary["per_frame"])
            for d in duals:
                for uid, v in d["per_frame"].items():
                    merged.setdefault(uid, v)
            field = {"median_box": primary["box"], "per_frame": merged,
                     "score": primary["score"], "distinct": primary["distinct"],
                     "duals": [d["box"] for d in duals]}
            recover_field_values(frames, ok_uids, field)
        else:
            field = None
        report.append({"folder": g, "channel_field_box": field["median_box"] if field else None,
                       "field_score": field["score"] if field else 0.0,
                       "distinct_values": field["distinct"] if field else 0,
                       "dual_boxes": field.get("duals", []) if field else [],
                       "slots": allm[:6]})
        per_folder[g] = (report[-1], field)

    # === (선택) 색마스킹 ROI 재읽기 ===
    if args.force_read:
        field_lbl = out / "field_labels"; sub = out / "unread_imgs"
        for dd in (field_lbl, sub):
            shutil.rmtree(dd, ignore_errors=True); dd.mkdir(parents=True, exist_ok=True)
        unread = []
        for g, (entry, field) in per_folder.items():
            if not field:
                continue
            bx = field["median_box"]
            for uid in sorted(seqs.get(g, [])):
                if uid in by_id and uid not in field["per_frame"]:
                    im = by_id[uid]; W = float(im.get("image_width") or 1280) or 1280
                    H = float(im.get("image_height") or 720) or 720
                    cx = ((bx[0] + bx[2]) / 2) / W; cy = ((bx[1] + bx[3]) / 2) / H
                    bw = (bx[2] - bx[0]) / W; bh = (bx[3] - bx[1]) / H
                    (field_lbl / f"{uid}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    ip = im.get("image_path")
                    if ip:
                        link = sub / Path(ip).name
                        if not (link.exists() or link.is_symlink()):
                            try:
                                os.symlink(Path(ip).resolve(), link)
                            except Exception:
                                pass
                    unread.append(uid)
        if unread:
            colors = learn_folder_colors(by_id, seqs, per_folder)
            (out / "channel_colors.json").write_text(json.dumps(colors))
            cmd = [PY, f"{SRC}/mask_read.py", "--images", sub, "--yolo-label-dir", field_lbl,
                   "--out", out / "field_read.json", "--colors", out / "channel_colors.json",
                   "--pad", "0.2", "--min-height", "120", "--tol", str(args.mask_tol),
                   "--progress-every", "200"]
            if args.viz_mask > 0:
                cmd += ["--viz-dir", out / "mask_viz", "--viz-per", str(args.viz_mask * max(1, len(per_folder)))]
            rc = P.sh(cmd, env)
            if rc == 0 and (out / "field_read.json").exists():
                fr = json.loads((out / "field_read.json").read_text())
                readval = {}
                for im in fr.get("images", []):
                    for c in im.get("candidates", []):
                        d = P._dg(c.get("text", ""))
                        if d and 1 <= len(d) <= 5:
                            readval[im["image_id"]] = d; break
                filled = 0
                for g, (entry, field) in per_folder.items():
                    if not field:
                        continue
                    for uid in seqs.get(g, []):
                        if uid in readval and uid not in field["per_frame"]:
                            field["per_frame"][uid] = readval[uid]; filled += 1
                print(f"[mask-read] ROI 색마스킹 재읽기 {filled}프레임 (unread {len(unread)}중)", flush=True)
                if args.viz_mask > 0:
                    print(f"[viz-mask] 마스킹 스텝 저장 → {out}/mask_viz/", flush=True)
        shutil.rmtree(sub, ignore_errors=True)

    # === 출력 ===
    rows, per_folder3 = [], {}
    for g, (entry, field) in per_folder.items():
        frows = []
        if field:
            for uid, val in sorted(field["per_frame"].items()):
                r = {"folder": g, "frame": meta.get(uid, uid), "channel_number": val}
                rows.append(r); frows.append(r)
        per_folder3[g] = (entry, frows, field)
    with (out / "per_frame.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "frame", "channel_number"]); w.writeheader(); w.writerows(rows)
    (out / "profile_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    if args.split_output:
        used = {}
        for g, (entry, frows, _f) in per_folder3.items():
            nm = _safe(Path(g).name if g not in ("", "(root)") else root.name, root)
            while nm in used and used[nm] != g:
                nm += "_x"
            used[nm] = g
            gd = out / nm; gd.mkdir(parents=True, exist_ok=True)
            (gd / "profile_report.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2))
            with (gd / "per_frame.csv").open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=["frame", "channel_number"]); w.writeheader()
                for r in frows:
                    w.writerow({"frame": r["frame"], "channel_number": r["channel_number"]})

    # === 정성 이미지 (성공 N + 실패 전부, UI별) ===
    if not args.no_qualitative:
        try:
            from PIL import Image, ImageDraw
        except Exception:
            Image = None
        if Image is not None:
            norm = lambda s: str(int(s)) if s else ""
            saved, usedq = 0, {}
            for g, (entry, frows, field) in per_folder3.items():
                if not field:
                    continue
                box = [int(v) for v in field["median_box"]]; pf = field["per_frame"]
                allu = [u for u in sorted(seqs.get(g, [])) if u in by_id]
                oks, fails = [], []
                for uid in allu:
                    pred = P._dg(pf.get(uid, ""))
                    if args.gt_from_filename:
                        gt = P.gt_from_name(meta.get(uid, uid))
                        isf = (not pred) or (gt and norm(pred) != norm(gt))
                    else:
                        isf = not pred
                    (fails if isf else oks).append(uid)
                step = max(1, len(oks) // max(1, args.samples_per_folder))
                draw = [(u, False) for u in oks[::step][:args.samples_per_folder]] + [(u, True) for u in fails]
                if not draw:
                    continue
                nm = _safe(Path(g).name if g not in ("", "(root)") else root.name, root)
                while nm in usedq and usedq[nm] != g:
                    nm += "_x"
                usedq[nm] = g
                qd = (out / nm / "qualitative") if args.split_output else (out / "qualitative" / _safe(g, root))
                fdir = qd / "_failures"; qd.mkdir(parents=True, exist_ok=True)
                if fails:
                    fdir.mkdir(parents=True, exist_ok=True)
                for uid, isf in draw:
                    try:
                        img = Image.open(by_id[uid].get("image_path")).convert("RGB")
                    except Exception:
                        continue
                    d = ImageDraw.Draw(img); val = pf.get(uid, "") or "(none)"
                    col = (255, 0, 0) if isf else (0, 200, 0)
                    d.rectangle(box, outline=col, width=3)
                    lab = f"ch:{val}"
                    if isf and args.gt_from_filename:
                        lab += f" (gt:{P.gt_from_name(meta.get(uid, uid))})"
                    d.text((box[0], max(0, box[1] - 14)), lab, fill=col)
                    img.save((fdir if isf else qd) / f"{meta.get(uid, uid)}.jpg", quality=88)
                    saved += 1
            print(f"정성 이미지 {saved}장 (성공 샘플 + _failures/ 실패 전부)", flush=True)

    print("\n채널 필드(폴더별):")
    for r in report:
        dm = f"  +듀얼{len(r['dual_boxes'])}" if r["dual_boxes"] else ""
        print(f"  {r['folder']:<40} field={r['channel_field_box']}  score={r['field_score']}{dm}")

    if args.gt_from_filename:
        norm = lambda s: str(int(s)) if s else ""
        pbu = {}
        for g, (entry, frows, field) in per_folder3.items():
            if field:
                for uid, val in field["per_frame"].items():
                    pbu[uid] = P._dg(val)
        per = {}
        for g, uids in seqs.items():
            for uid in uids:
                if uid not in by_id:
                    continue
                gt = P.gt_from_name(meta.get(uid, uid))
                if not gt:
                    continue
                t = per.setdefault(g, [0, 0, 0]); t[0] += 1
                pr = pbu.get(uid, "")
                if pr:
                    t[1] += 1
                    if norm(pr) == norm(gt):
                        t[2] += 1
        tot = sum(v[0] for v in per.values()); rd = sum(v[1] for v in per.values()); co = sum(v[2] for v in per.values())
        pc = lambda a, b: round(a / b * 100, 1) if b else 0
        print("\n=== 폴더별 정확도 ===")
        print(f"  {'folder':<40}{'e2e%':>7}{'cov%':>7}{'읽었을때%':>10}{'correct/total':>16}")
        for g in sorted(per):
            a, b, c = per[g]
            print(f"  {g:<40}{pc(c,a):>7}{pc(b,a):>7}{pc(c,b):>10}{f'{c}/{a}':>16}")
        print(f"  {'-'*40}\n  {'전체':<40}{pc(co,tot):>7}{pc(rd,tot):>7}{pc(co,rd):>10}{f'{co}/{tot}':>16}")

    if args.viz_steps > 0:
        step_viz.render(by_id, seqs, per_folder3, meta, out, args.viz_steps,
                        args.gt_from_filename, lambda s: _safe(s, root))

    print(f"\n결과: {out}/per_frame.csv , profile_report.json  ({len(rows)}개)")
    if not args.keep_staged:
        shutil.rmtree(flat, ignore_errors=True)


if __name__ == "__main__":
    main()
