#!/usr/bin/env python3
"""Channel inference — FULL OCR ONLY (on-device lean build).

ONE model only: full-image OCR (PaddleOCR PP-OCRv4 **mobile**). No YOLO detector,
no numeric OCR. Everything else is pure rules (no learned UI layout -> no overfit).

Pipeline (per folder = one zap session, each frame a different channel):
    full OCR (mobile)        -> every text/number candidate + position, per frame
    temporal profiling       -> pick the channel-number FIELD from the frames:
                                slot clustering + VALUE DIVERSITY (zap changes value)
                                + shape/size/position + time/date/text exclusion
    recover                  -> unread frames: take a number candidate near the field
    read per frame           -> each frame's channel = the field's value

Deployable on device: mobile OCR + trivial rules, ~a few seconds per 5-frame zap.

Flags: --gt-from-filename --split-output --viz-steps N --debug-unread N --symlink ...
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import predict_folder as P
from temporal_profile_select import profile_sequence
from predict_folder_temporal_full import recover_field_values
import step_viz


def _safe(name, root):
    return name.replace("/", "__").replace(" ", "_") or root.name


def _present_ratio(p):
    try:
        a, b = str(p["present"]).split("/"); return int(a) / max(1, int(b))
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symlink", action="store_true")
    ap.add_argument("--gt-from-filename", action="store_true")
    ap.add_argument("--split-output", action="store_true")
    ap.add_argument("--samples-per-folder", type=int, default=20)
    ap.add_argument("--no-qualitative", action="store_true")
    ap.add_argument("--viz-steps", type=int, default=0, metavar="N")
    ap.add_argument("--debug-unread", type=int, default=0, metavar="N")
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

    # === 유일한 모델: FULL OCR (PP-OCRv4 mobile) ===
    rc = P.sh([PY, f"{SRC}/run_paddleocr_export.py", "--images", flat, "--out", out / "full_ocr.json",
               "--use-gpu", "--ocr-version", "PP-OCRv4", "--text-detection-model-name",
               "PP-OCRv4_mobile_det", "--text-recognition-model-name", "en_PP-OCRv4_mobile_rec",
               "--progress-every", 200], env)
    if rc != 0:
        raise SystemExit(f"[ocr_only] full OCR 실패 (rc={rc})")

    cand = json.loads((out / "full_ocr.json").read_text())
    by_id = {im["image_id"]: im for im in cand.get("images", [])}

    # === 프로파일링(규칙) -> 채널 필드 + 값 복구 ===
    report, per_folder = [], {}
    for g, uids in sorted(seqs.items()):
        ok_uids = [u for u in sorted(uids) if u in by_id]
        frames = [by_id[u] for u in ok_uids]
        if not frames:
            continue
        profs = profile_sequence(frames, ok_uids)
        field = profs[0] if profs and profs[0]["score"] > 0 else None
        if field is None and profs:                       # 하드 UI fallback (값 다양성 필수)
            c = [p for p in profs if 0.25 <= p["aspect"] <= 6.0 and p["area%"] < 6.0
                 and _present_ratio(p) >= 0.4 and p.get("distinct_values", 0) >= 2]
            if c:
                field = max(c, key=_present_ratio); field["_fallback"] = True
        if field:
            recover_field_values(frames, ok_uids, field)
        report.append({"folder": g, "channel_field_box": field["median_box"] if field else None,
                       "field_score": round(field["score"], 3) if field else 0.0,
                       "fallback": bool(field.get("_fallback")) if field else False,
                       "profiles": profs[:6]})
        per_folder[g] = (report[-1], field)

    # === rows + 출력 ===
    rows, per_folder3 = [], {}
    for g, (entry, field) in per_folder.items():
        frows = []
        if field:
            for uid, val in sorted(field["per_frame"].items()):
                r = {"folder": g, "frame": meta.get(uid, uid), "channel_number": val}
                rows.append(r); frows.append(r)
        per_folder3[g] = (entry, frows, field)
    with (out / "per_frame.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "frame", "channel_number"])
        w.writeheader(); w.writerows(rows)
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

    # === 정성 이미지 (성공 N + 실패 전부) ===
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
            print(f"정성 이미지 {saved}장 (성공 샘플 + _failures/)", flush=True)

    print("\n채널 필드(폴더별):")
    for r in report:
        print(f"  {r['folder']:<40} field={r['channel_field_box']}  score={r['field_score']}")

    # === 정확도 (폴더별) ===
    if args.gt_from_filename:
        norm = lambda s: str(int(s)) if s else ""
        pred_by_uid = {}
        for g, (entry, frows, field) in per_folder3.items():
            if field:
                for uid, val in field["per_frame"].items():
                    pred_by_uid[uid] = P._dg(val)
        per = {}
        for g, uids in seqs.items():
            for uid in uids:
                if uid not in by_id:
                    continue
                gt = P.gt_from_name(meta.get(uid, uid))
                if not gt:
                    continue
                t = per.setdefault(g, [0, 0, 0]); t[0] += 1
                pr = pred_by_uid.get(uid, "")
                if pr:
                    t[1] += 1
                    if norm(pr) == norm(gt):
                        t[2] += 1
        tot = sum(v[0] for v in per.values()); rd = sum(v[1] for v in per.values()); co = sum(v[2] for v in per.values())
        pc = lambda a, b: round(a / b * 100, 1) if b else 0
        print(f"\n=== 폴더별 정확도 (파일명 정답) ===")
        print(f"  {'folder':<40}{'e2e%':>7}{'cov%':>7}{'읽었을때%':>10}{'correct/total':>16}")
        for g in sorted(per):
            a, b, c = per[g]
            print(f"  {g:<40}{pc(c,a):>7}{pc(b,a):>7}{pc(c,b):>10}{f'{c}/{a}':>16}")
        print(f"  {'-'*40}\n  {'전체':<40}{pc(co,tot):>7}{pc(rd,tot):>7}{pc(co,rd):>10}{f'{co}/{tot}':>16}")

    if args.viz_steps > 0:
        step_viz.render(by_id, seqs, per_folder3, meta, out, args.viz_steps,
                        args.gt_from_filename, lambda s: _safe(s, root))

    print(f"\n결과: {out}/per_frame.csv , profile_report.json  (프레임별 채널 {len(rows)}개)")
    if not args.keep_staged:
        shutil.rmtree(flat, ignore_errors=True)


if __name__ == "__main__":
    main()
