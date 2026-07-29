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
import shutil
from pathlib import Path

import predict_folder as P
from temporal_profile_select import profile_sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--symlink", action="store_true")
    ap.add_argument("--detector", default=None, help="검출기 pt 오버라이드")
    ap.add_argument("--gt-from-filename", action="store_true", help="파일명=정답으로 정확도 채점")
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
    rows, report = [], []
    for g, uids in sorted(seqs.items()):
        ok_uids = [u for u in sorted(uids) if u in by_id]
        frames = [by_id[u] for u in ok_uids]
        if not frames:
            continue
        profs = profile_sequence(frames, ok_uids)
        field = profs[0] if profs and profs[0]["score"] > 0 else None
        report.append({"folder": g, "channel_field_box": field["median_box"] if field else None,
                       "field_score": field["score"] if field else 0.0, "profiles": profs[:6]})
        if field:
            for uid, val in field["per_frame"].items():
                rows.append({"folder": g, "frame": meta.get(uid, uid), "channel_number": val})

    with (out / "per_frame.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "frame", "channel_number"])
        w.writeheader(); w.writerows(rows)
    (out / "profile_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"\n채널 필드(폴더별):")
    for r in report:
        print(f"  {r['folder']:<40} field={r['channel_field_box']}  score={r['field_score']}")

    if args.gt_from_filename:
        norm = lambda s: str(int(s)) if s else ""
        ok = tot = 0
        for r in rows:
            gt = P.gt_from_name(r["frame"])
            if gt:
                tot += 1; ok += (norm(P._dg(r["channel_number"])) == norm(gt))
        print(f"\n=== 정확도 (파일명 정답 기준): "
              f"{round(ok / tot * 100, 1) if tot else 0}%  ({ok}/{tot}) ===")

    print(f"\n결과: {out}/per_frame.csv , profile_report.json  (프레임별 채널 {len(rows)}개)")
    if not args.keep_staged:
        shutil.rmtree(flat, ignore_errors=True)


if __name__ == "__main__":
    main()
