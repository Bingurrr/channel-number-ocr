#!/usr/bin/env python3
"""Channel inference with ENHANCED slot analysis (full OCR only, on-device).

Same as predict_folder_ocr_only.py but the slot step uses slot_analysis.analyze():
  * clustering by POSITION **+ SIZE** (separates crowded boxes near the channel)
  * VALUE DIVERSITY qualifies the channel (zap changes value)
  * DUAL-DISPLAY: a 2nd slot with matching values fills gaps of the primary
  * vertical channel-list penalty (pre/next-channel entries)

The dual slot and the known ROI are the "past experience" that corrects single-frame
errors: if the primary is unread in a frame, the dual (same channel) fills it; and
--force-read re-OCRs the primary ROI crop for anything still missing.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import predict_folder as P
import slot_analysis as SA
from predict_folder_temporal_full import recover_field_values
import step_viz


def _safe(name, root):
    return name.replace("/", "__").replace(" ", "_") or root.name


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
                    help="(선택) 못읽은 프레임의 확정 ROI를 crop+확대해서 full OCR로 재읽기")
    ap.add_argument("--viz-force-read", type=int, default=0, metavar="N",
                    help="force-read 변환 시각화(폴더당 N장): crop→padding→확대→읽기 스텝 저장")
    ap.add_argument("--viz-steps", type=int, default=0, metavar="N")
    ap.add_argument("--keep-staged", action="store_true")
    ap.add_argument("--rec-model-dir", default="models/full_image_ocr/en_PP-OCRv4_mobile_rec_ft",
                    help="파인튜닝된 full-OCR rec 디렉토리(있으면 사용). 순정으로 돌리려면 'none'")
    ap.add_argument("--min-conf", type=float, default=0.3,
                    help="슬롯 클러스터링 전 OCR 신뢰도 임계값(낮추면 none↓, 오답↑ 가능)")
    ap.add_argument("--force-answer", action="store_true",
                    help="못읽은 프레임을 rec-only(det 생략)로 강제 읽기 → none 없애기")
    ap.add_argument("--both-channel", action="store_true",
                    help="채널번호가 2곳(A/B)에 뜨는 UI: 둘 다 저장(ch1/ch2)하고, 한쪽이 "
                         "none이거나 conf 낮으면 다른쪽 값 사용")
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
    ocr_cmd = [PY, f"{SRC}/run_paddleocr_export.py", "--images", flat, "--out", out / "full_ocr.json",
               "--use-gpu", "--ocr-version", "PP-OCRv4", "--text-detection-model-name",
               "PP-OCRv4_mobile_det", "--text-recognition-model-name", "en_PP-OCRv4_mobile_rec",
               "--progress-every", 200]
    # 파인튜닝된 rec가 있으면 그 가중치로 읽기 (det은 그대로). 'none'이면 순정.
    rec_dir = str(args.rec_model_dir).strip()
    if rec_dir.lower() not in ("", "none"):
        rd = Path(rec_dir)
        if not rd.is_absolute():
            rd = Path(__file__).resolve().parent / rd
        if (rd / "inference.pdiparams").exists():
            ocr_cmd += ["--text-recognition-model-dir", str(rd)]
            print(f"[slot] 파인튜닝 rec 사용: {rd}", flush=True)
        else:
            print(f"[slot] rec-model-dir 없음 → 순정 rec 사용 ({rd})", flush=True)
    rc = P.sh(ocr_cmd, env)
    if rc != 0:
        raise SystemExit(f"[slot] full OCR 실패 (rc={rc})")
    cand = json.loads((out / "full_ocr.json").read_text())
    by_id = {im["image_id"]: im for im in cand.get("images", [])}

    # === 강화된 슬롯 분석 -> 채널 필드(+듀얼) ===
    report, per_folder = [], {}
    for g, uids in sorted(seqs.items()):
        ok_uids = [u for u in sorted(uids) if u in by_id]
        frames = [by_id[u] for u in ok_uids]
        if not frames:
            continue
        primary, duals, allm = SA.analyze(frames, ok_uids, conf_thr=args.min_conf)
        if primary:
            p_pf = primary["per_frame"]; p_conf = primary.get("per_frame_conf", {})
            # 듀얼(2번째 채널박스) 값/신뢰도 병합 (여러 듀얼이면 프레임별 최고 conf)
            d_pf, d_conf = {}, {}
            for d in duals:
                dc = d.get("per_frame_conf", {})
                for uid, v in d["per_frame"].items():
                    c = dc.get(uid, 0.5)
                    if uid not in d_conf or c > d_conf[uid]:
                        d_pf[uid] = v; d_conf[uid] = c
            if args.both_channel and d_pf:
                # A/B 교차: none이면 다른쪽, 둘 다 있으면 conf 높은쪽
                final = {}
                for uid in set(p_pf) | set(d_pf):
                    v1, v2 = p_pf.get(uid), d_pf.get(uid)
                    if v1 and v2:
                        final[uid] = v1 if p_conf.get(uid, 0) >= d_conf.get(uid, 0) else v2
                    else:
                        final[uid] = v1 or v2
                field = {"median_box": primary["box"], "per_frame": final,
                         "per_frame_1": dict(p_pf), "per_frame_2": dict(d_pf),
                         "box2": duals[0]["box"] if duals else None,
                         "score": primary["score"], "distinct": primary["distinct"],
                         "duals": [d["box"] for d in duals], "both": True}
            else:
                merged = dict(p_pf)
                for uid, v in d_pf.items():                     # 기존: primary 빈칸만 채움
                    merged.setdefault(uid, v)
                field = {"median_box": primary["box"], "per_frame": merged,
                         "score": primary["score"], "distinct": primary["distinct"],
                         "duals": [d["box"] for d in duals], "both": False}
            recover_field_values(frames, ok_uids, field)
        else:
            field = None
        report.append({"folder": g, "channel_field_box": field["median_box"] if field else None,
                       "field_score": field["score"] if field else 0.0,
                       "distinct_values": field["distinct"] if field else 0,
                       "dual_boxes": field.get("duals", []) if field else [],
                       "slots": allm[:6]})
        per_folder[g] = (report[-1], field)

    # === (선택) ROI crop full-OCR 재읽기 ===
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
            if args.force_answer:
                # rec-only(det 생략) 강제 읽기 → none 제거. 파인튜닝 rec 사용.
                rd = Path(rec_dir) if rec_dir.lower() not in ("", "none") else None
                if rd is not None and not rd.is_absolute():
                    rd = Path(__file__).resolve().parent / rd
                rr = [PY, f"{SRC}/rec_only_read.py", "--images", sub, "--yolo-label-dir", field_lbl,
                      "--out", out / "field_read.json", "--pad", "0.2", "--min-height", "120",
                      "--progress-every", "200"]
                if rd is not None and (rd / "inference.pdiparams").exists():
                    rr += ["--rec-model-dir", str(rd)]
                rc = P.sh(rr, env)
            else:
                rc = P.sh([PY, f"{SRC}/fullocr_crops.py", "--images", sub, "--yolo-label-dir", field_lbl,
                           "--out", out / "field_read.json", "--pad", "0.2", "--min-height", "48",
                           "--progress-every", "200"], env)
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
                print(f"[force-read/full-OCR] ROI 재읽기 {filled}프레임 (unread {len(unread)}중)", flush=True)
                # crop→pad→확대→읽기 스텝 시각화
                if args.viz_force_read > 0:
                    from PIL import Image as _I
                    cnt = {}
                    for g, (entry, field) in per_folder.items():
                        if not field:
                            continue
                        for uid in sorted(seqs.get(g, [])):
                            if uid in readval and cnt.get(g, 0) < args.viz_force_read:
                                try:
                                    img = _I.open(by_id[uid]["image_path"]).convert("RGB")
                                except Exception:
                                    continue
                                ui = _safe(Path(g).name if g not in ("", "(root)") else root.name, root)
                                step_viz.force_read_steps(
                                    img, field["median_box"],
                                    out / "force_read_viz" / ui / meta.get(uid, uid), readval[uid])
                                cnt[g] = cnt.get(g, 0) + 1
                    print(f"[viz-force-read] crop→pad→확대 저장 → {out}/force_read_viz/", flush=True)
        shutil.rmtree(sub, ignore_errors=True)

    # === 출력 ===
    cols = ["folder", "frame", "channel_number"] + (["channel_number_1", "channel_number_2"] if args.both_channel else [])
    rows, per_folder3 = [], {}
    for g, (entry, field) in per_folder.items():
        frows = []
        if field:
            for uid, val in sorted(field["per_frame"].items()):
                r = {"folder": g, "frame": meta.get(uid, uid), "channel_number": val}
                if args.both_channel:
                    r["channel_number_1"] = field.get("per_frame_1", {}).get(uid, "")
                    r["channel_number_2"] = field.get("per_frame_2", {}).get(uid, "")
                rows.append(r); frows.append(r)
        per_folder3[g] = (entry, frows, field)
    with (out / "per_frame.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    (out / "profile_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    if args.split_output:
        gcols = [c for c in cols if c != "folder"]
        used = {}
        for g, (entry, frows, _f) in per_folder3.items():
            nm = _safe(Path(g).name if g not in ("", "(root)") else root.name, root)
            while nm in used and used[nm] != g:
                nm += "_x"
            used[nm] = g
            gd = out / nm; gd.mkdir(parents=True, exist_ok=True)
            (gd / "profile_report.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2))
            with (gd / "per_frame.csv").open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=gcols); w.writeheader()
                for r in frows:
                    w.writerow({c: r.get(c, "") for c in gcols})

    # === 정성 이미지 (성공 N + 실패 전부, UI별 폴더) ===
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
