#!/usr/bin/env python3
"""End-to-end channel inference by UI-INVARIANT temporal profiling (overfit-resistant).

Pipeline:
    detector + FULL OCR  -> every number/text candidate per frame (general models)
    temporal profiling   -> pick the channel-number FIELD from consecutive frames
                            using type-consistency + shape (aspect/size) + position
                            stability + mutual-exclusion (text/time/date slots)
    read per frame        -> each frame's channel number = the field's value

Nothing here learns your 40 UI layouts, so it does not overfit — the same rules
apply to unseen commercial UIs. Each frame may be a different channel (captured on
channel change); output is per-frame.

Same flags as predict_folder.py (--detector, --gt-from-filename, --symlink, ...).
Run on a folder of consecutive frames (one capture session = one UI).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path

import predict_folder as P
from temporal_profile_select import profile_sequence, classify, digits as _digits, TIME as _TIME, DATE as _DATE


def recover_field_values(frames, ids, field, dist_thr=0.06):
    """못 읽은 프레임 복구: 알아낸 채널 필드 위치 근처의 숫자 후보를 회수.
    (재OCR 없이 기존 full OCR 후보에서 필드 근처 1-4자리 숫자를 가져옴)"""
    box = field["median_box"]
    W0 = float(frames[0].get("image_width") or 1280) or 1280
    H0 = float(frames[0].get("image_height") or 720) or 720
    fcx = ((box[0] + box[2]) / 2) / W0
    fcy = ((box[1] + box[3]) / 2) / H0
    have = set(field["per_frame"].keys())
    recovered = 0
    for fi, im in enumerate(frames):
        uid = ids[fi]
        if uid in have:
            continue
        W = float(im.get("image_width") or W0) or W0
        H = float(im.get("image_height") or H0) or H0
        best_t, bd = None, dist_thr
        for c in im.get("candidates", []):
            b = c.get("bbox_xyxy"); t = c.get("text", "")
            if not b or len(b) != 4 or not _digits(t):
                continue
            cx = ((b[0] + b[2]) / 2) / W; cy = ((b[1] + b[3]) / 2) / H
            d = ((cx - fcx) ** 2 + (cy - fcy) ** 2) ** 0.5
            if d < bd:
                bd, best_t = d, t
        if best_t and not (_TIME.search(best_t) or _DATE.search(best_t)):
            toks = [x for x in re.findall(r"\d+", best_t) if 1 <= len(x) <= 4]
            if toks:
                field["per_frame"][uid] = toks[0]   # 회수됨
                recovered += 1
    return recovered


TYPE_COLOR = {"channelnum": (0, 200, 0), "time": (60, 130, 255), "date": (170, 120, 255),
              "text": (150, 150, 150), "othernum": (255, 150, 0)}


def make_step_viz(by_id, seqs, per_folder, meta, out, n_each, gt_mode):
    """Save N success + N fail montages: [1) input | 2) full OCR all text | 3) channel field]."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        print("[viz-steps] PIL 없음 → 건너뜀"); return
    norm = lambda s: str(int(s)) if s else ""
    succ, fail = [], []
    for g, (entry, frows, field) in per_folder.items():
        if not field:
            continue
        for uid in sorted(seqs.get(g, [])):
            if uid not in by_id:
                continue
            pred = P._dg(field["per_frame"].get(uid, ""))
            gt = P.gt_from_name(meta.get(uid, uid)) if gt_mode else ""
            ok = (pred and norm(pred) == norm(gt)) if (gt_mode and gt) else bool(pred)
            tgt = succ if ok else fail
            if len(tgt) < n_each:
                tgt.append((uid, field["median_box"], field["per_frame"].get(uid, ""), gt))
        if len(succ) >= n_each and len(fail) >= n_each:
            break

    sd = out / "step_viz"
    PW, PH = 560, 315
    for label, items in [("success", succ), ("fail", fail)]:
        d = sd / label; d.mkdir(parents=True, exist_ok=True)
        for i, (uid, box, val, gt) in enumerate(items):
            try:
                im = Image.open(by_id[uid].get("image_path")).convert("RGB")
            except Exception:
                continue
            # panel 2: full OCR — 모든 텍스트 후보 (타입별 색)
            B = im.copy(); db = ImageDraw.Draw(B)
            for c in by_id[uid].get("candidates", []):
                b = c.get("bbox_xyxy"); t = c.get("text", "")
                if not b or len(b) != 4:
                    continue
                col = TYPE_COLOR.get(classify(t), (200, 200, 200))
                db.rectangle([int(v) for v in b], outline=col, width=2)
                db.text((b[0], max(0, b[1] - 11)), str(t)[:12], fill=col)
            # panel 3: 채널 필드 선택 + 값
            C = im.copy(); dc = ImageDraw.Draw(C)
            bx = [int(v) for v in box]
            ok = (P._dg(val) and gt and norm(P._dg(val)) == norm(gt)) if gt else bool(P._dg(val))
            col = (0, 200, 0) if ok else (255, 0, 0)
            dc.rectangle(bx, outline=col, width=4)
            dc.text((bx[0], max(0, bx[1] - 16)),
                    f"pred:{P._dg(val) or '-'}" + (f"  gt:{gt}" if gt else ""), fill=col)

            def tile(x, title):
                x = x.resize((PW, PH)); dd = ImageDraw.Draw(x)
                dd.rectangle([0, 0, PW, 18], fill=(0, 0, 0)); dd.text((4, 2), title, fill=(255, 255, 0))
                return x
            A = tile(im.copy(), "1) input")
            B = tile(B, "2) full OCR (all text, green=pure num)")
            C = tile(C, "3) channel field selected")
            canvas = Image.new("RGB", (PW * 3 + 16, PH), (25, 25, 25))
            canvas.paste(A, (0, 0)); canvas.paste(B, (PW + 8, 0)); canvas.paste(C, (2 * PW + 16, 0))
            canvas.save(d / f"{i:02d}_{meta.get(uid, uid)}.jpg", quality=88)
    print(f"[viz-steps] 저장: {sd}  (success {len(succ)}, fail {len(fail)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--symlink", action="store_true")
    ap.add_argument("--detector", default=None, help="검출기 pt 오버라이드")
    ap.add_argument("--gt-from-filename", action="store_true", help="파일명=정답으로 정확도 채점")
    ap.add_argument("--split-output", action="store_true",
                    help="--out 아래에 UI별(폴더 이름) 하위폴더로 결과 분리 저장")
    ap.add_argument("--samples-per-folder", type=int, default=20,
                    help="정성 이미지: 폴더당 샘플 수 (채널 필드 박스 + 값)")
    ap.add_argument("--no-qualitative", action="store_true", help="정성 이미지 저장 안 함")
    ap.add_argument("--viz-steps", type=int, default=0, metavar="N",
                    help="알고리즘 단계별 몽타주 저장: 성공 N + 실패 N 예시 "
                         "[입력 | full OCR 전체 | 채널 필드 선택]")
    ap.add_argument("--keep-staged", action="store_true")
    args = ap.parse_args()

    cfg = P.load_config()
    if args.detector:
        p = Path(args.detector)
        cfg["detector"] = str(p if p.is_absolute() else (P.HERE / p).resolve())
        print(f"[detector] {cfg['detector']}", flush=True)
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
    manifest = out / "manifest.json"
    manifest.write_text(json.dumps({"sequences": [
        {"sequence_id": g.replace('/', '__'), "group_key": g, "images": sorted(v)}
        for g, v in sorted(seqs.items())]}, ensure_ascii=False))

    # 1) detector
    rc = P.sh([PY, f"{SRC}/export_recursive_detector_predictions.py", "--model", cfg["detector"],
               "--images-dir", flat, "--output-dir", out / "detector", "--imgsz", cfg["imgsz"],
               "--device", args.device, "--batch", args.batch, "--candidate-conf", 0.05], env)
    if rc != 0:
        raise SystemExit(f"[temporal_full] detector 실패 (rc={rc})")

    # 2) FULL OCR (general model -> all text/number candidates)
    rc = P.sh([PY, f"{SRC}/run_paddleocr_export.py", "--images", flat, "--out", out / "full_ocr.json",
               "--use-gpu", "--ocr-version", "PP-OCRv4", "--text-detection-model-name",
               "PP-OCRv4_mobile_det", "--text-recognition-model-name", "en_PP-OCRv4_mobile_rec",
               "--progress-every", 200], env)
    if rc != 0:
        raise SystemExit(f"[temporal_full] full OCR 실패 (rc={rc})")

    # 3) temporal profiling -> channel field + per-frame value
    cand = json.loads((out / "full_ocr.json").read_text())
    by_id = {im["image_id"]: im for im in cand.get("images", [])}

    def safe(name):
        return (name.replace("/", "__").replace(" ", "_")) or root.name

    rows, report, per_folder = [], [], {}
    used_names = {}
    for g, uids in sorted(seqs.items()):
        ok_uids = [u for u in sorted(uids) if u in by_id]
        frames = [by_id[u] for u in ok_uids]
        if not frames:
            continue
        profs = profile_sequence(frames, ok_uids)
        field = profs[0] if profs and profs[0]["score"] > 0 else None
        if field:
            rec = recover_field_values(frames, ok_uids, field)   # 못 읽은 프레임 필드위치서 복구
            if rec:
                print(f"  [{g}] 필드위치 복구 {rec}프레임", flush=True)
        entry = {"folder": g, "channel_field_box": field["median_box"] if field else None,
                 "field_score": field["score"] if field else 0.0, "profiles": profs[:6]}
        report.append(entry)
        frows = []
        if field:
            for uid, val in field["per_frame"].items():
                r = {"folder": g, "frame": meta.get(uid, uid), "channel_number": val}
                rows.append(r); frows.append(r)
        per_folder[g] = (entry, frows, field)

    with (out / "per_frame.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "frame", "channel_number"])
        w.writeheader(); w.writerows(rows)
    (out / "profile_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    # UI별 분리 저장
    if args.split_output:
        for g, (entry, frows, _field) in per_folder.items():
            base = Path(g).name if g not in ("", "(root)") else root.name
            name = safe(base)
            while name in used_names and used_names[name] != g:
                name += "_x"
            used_names[name] = g
            gdir = out / name
            gdir.mkdir(parents=True, exist_ok=True)
            (gdir / "profile_report.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2))
            with (gdir / "per_frame.csv").open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=["frame", "channel_number"])
                w.writeheader()
                for r in frows:
                    w.writerow({"frame": r["frame"], "channel_number": r["channel_number"]})

    # 정성 이미지: 성공 최대 N개 + 실패 전부(_failures/). 채널 필드 박스 + 값.
    #   GT(파일명) 있으면 성공=값 일치 / 실패=불일치·미검출.  없으면 실패=미검출.
    if not args.no_qualitative:
        try:
            from PIL import Image, ImageDraw
        except Exception:
            Image = None
        if Image is not None:
            norm = lambda s: str(int(s)) if s else ""
            saved = 0
            used_q = {}
            for g, (entry, frows, field) in per_folder.items():
                if not field:
                    continue
                box = [int(v) for v in field["median_box"]]
                pf = field["per_frame"]
                all_uids = [u for u in sorted(seqs.get(g, [])) if u in by_id]
                oks, fails = [], []
                for uid in all_uids:
                    pred = P._dg(pf.get(uid, ""))
                    if args.gt_from_filename:
                        gt = P.gt_from_name(meta.get(uid, uid))
                        is_fail = (not pred) or (gt and norm(pred) != norm(gt))
                    else:
                        is_fail = not pred            # GT 없으면: 채널 미검출 = 실패
                    (fails if is_fail else oks).append(uid)
                step = max(1, len(oks) // max(1, args.samples_per_folder))
                sample_ok = oks[::step][:args.samples_per_folder]
                to_draw = [(u, False) for u in sample_ok] + [(u, True) for u in fails]
                if not to_draw:
                    continue
                if args.split_output:
                    base = Path(g).name if g not in ("", "(root)") else root.name
                    nm = safe(base)
                    while nm in used_q and used_q[nm] != g:
                        nm += "_x"
                    used_q[nm] = g
                    qd = out / nm / "qualitative"
                else:
                    qd = out / "qualitative" / safe(g)
                faildir = qd / "_failures"
                qd.mkdir(parents=True, exist_ok=True)
                if fails:
                    faildir.mkdir(parents=True, exist_ok=True)
                for uid, is_fail in to_draw:
                    try:
                        img = Image.open(by_id[uid].get("image_path")).convert("RGB")
                    except Exception:
                        continue
                    d = ImageDraw.Draw(img)
                    val = pf.get(uid, "") or "(none)"
                    color = (255, 0, 0) if is_fail else (0, 200, 0)   # 빨강 실패 / 초록 성공
                    d.rectangle(box, outline=color, width=3)
                    label = f"ch:{val}"
                    if is_fail and args.gt_from_filename:
                        label += f" (gt:{P.gt_from_name(meta.get(uid, uid))})"
                    d.text((box[0], max(0, box[1] - 14)), label, fill=color)
                    dst = (faildir if is_fail else qd) / f"{meta.get(uid, uid)}.jpg"
                    img.save(dst, quality=88)
                    saved += 1
            print(f"정성 이미지 {saved}장 (성공 샘플 + _failures/ 실패 전부)", flush=True)

    print(f"\n채널 필드(폴더별):")
    for r in report:
        print(f"  {r['folder']:<40} field={r['channel_field_box']}  score={r['field_score']}")

    if args.gt_from_filename:
        norm = lambda s: str(int(s)) if s else ""
        pred_by_uid = {}
        for g, (entry, frows, field) in per_folder.items():
            if field:
                for uid, val in field["per_frame"].items():
                    pred_by_uid[uid] = P._dg(val)
        total = read = correct = 0                # total = 정답있는 전체 프레임
        for g, uids in seqs.items():
            for uid in uids:
                if uid not in by_id:
                    continue
                gt = P.gt_from_name(meta.get(uid, uid))
                if not gt:
                    continue
                total += 1
                pred = pred_by_uid.get(uid, "")
                if pred:
                    read += 1
                    if norm(pred) == norm(gt):
                        correct += 1
        e2e = round(correct / total * 100, 1) if total else 0
        cov = round(read / total * 100, 1) if total else 0
        acc = round(correct / read * 100, 1) if read else 0
        print(f"\n=== 정확도 (파일명 정답 기준) ===")
        print(f"  end-to-end (전체 대비):  {e2e}%   ({correct}/{total})  ← 진짜 성능")
        print(f"  커버리지 (값 읽은 비율):  {cov}%   ({read}/{total})")
        print(f"  읽었을 때 정확도:        {acc}%   ({correct}/{read})")
        print(f"  ※ end-to-end = 커버리지 × 읽었을때정확도. 못 읽은 {total-read}장이 진짜 실패.")

    if args.viz_steps > 0:
        make_step_viz(by_id, seqs, per_folder, meta, out, args.viz_steps, args.gt_from_filename)

    print(f"\n결과: {out}/per_frame.csv , profile_report.json  (프레임별 채널 {len(rows)}개)")
    if not args.keep_staged:
        shutil.rmtree(flat, ignore_errors=True)


if __name__ == "__main__":
    main()
