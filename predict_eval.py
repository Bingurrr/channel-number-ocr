#!/usr/bin/env python3
"""Predict channel numbers for a folder and score against FILENAME labels.

Label source = the filename: `100.jpg` -> ground truth 100  (leading digits of
the name are the GT channel number; `041_x.jpg` -> 041).

Layout (subfolders optional; each subfolder is just a group for the report):
    ROOT/
      UI_A/  100.jpg  20.jpg  ...
      UI_B/  7.jpg    891.jpg ...
  (or a flat folder of images directly under ROOT)

Each image is scored INDEPENDENTLY (single frame) — use this when every image is
its own labelled example. It reports final channel-number accuracy overall and
per subfolder. Detection metrics are NOT produced (no bbox ground truth here).

Usage:
    python predict_eval.py --root /path/to/ROOT --out /path/to/output
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config():
    c = json.loads((HERE / "config.json").read_text())
    for k in ("pipeline_src", "detector", "numeric_ocr", "selector_dir",
              "recheck_padded", "full_image_ocr"):
        v = c.get(k)
        if v and not str(v).startswith("/"):
            c[k] = str((HERE / v).resolve())
    return c


def dg(x):
    return "".join(ch for ch in str(x) if ch.isdigit())


def gt_from_name(stem: str) -> str:
    """Leading digits of the filename stem = ground-truth channel number."""
    m = re.match(r"\s*(\d+)", stem)
    return m.group(1) if m else ""


def sh(cmd, env):
    print("RUN:", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd], env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--numeric-equiv", action="store_true",
                    help="ignore leading zeros when scoring (041 == 41)")
    args = ap.parse_args()
    cfg = load_config()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    PY, SRC = cfg["python"], cfg["pipeline_src"]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/opt/conda/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    # 1) collect images recursively + filename GT; unique uid per image (folder path
    #    embedded so duplicate 001.jpg across folders don't collide); single-frame seqs
    flat = out / "images"
    flat.mkdir(parents=True, exist_ok=True)
    index, gt, seqs, used = {}, {}, [], {}
    for img in sorted(root.rglob("*")):
        if not (img.is_file() and img.suffix.lower() in IMG_EXTS):
            continue
        group = str(img.parent.relative_to(root)) or "(root)"
        uid = f"{group}__{img.stem}".replace("/", "__").replace(" ", "_")
        while uid in used:
            uid += "_x"
        used[uid] = True
        link = flat / f"{uid}{img.suffix.lower()}"
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(img.resolve(), link)
        index[uid] = group
        gt[uid] = gt_from_name(img.stem)          # GT from ORIGINAL filename
        seqs.append({"sequence_id": uid, "group_key": group, "images": [uid]})
    manifest = out / "manifest.json"
    manifest.write_text(json.dumps({"sequence_count": len(seqs), "sequences": seqs},
                                   ensure_ascii=False))
    print(f"images: {len(gt)}  (filename = GT)", flush=True)

    # 2) pipeline: detector -> full OCR -> refine -> padded recheck -> selector (single frame)
    sh([PY, f"{SRC}/export_recursive_detector_predictions.py", "--model", cfg["detector"],
        "--images-dir", flat, "--output-dir", out / "detector", "--imgsz", cfg["imgsz"],
        "--device", args.device, "--batch", 64, "--candidate-conf", 0.05], env)
    sh([PY, f"{SRC}/run_paddleocr_export.py", "--images", flat, "--out", out / "full_ocr.json",
        "--use-gpu", "--ocr-version", "PP-OCRv4", "--text-detection-model-name",
        "PP-OCRv4_mobile_det", "--text-recognition-model-name", "en_PP-OCRv4_mobile_rec",
        "--progress-every", 200], env)
    sh([PY, f"{SRC}/refine_ocr_candidates.py", "--ocr-json", out / "full_ocr.json",
        "--out", out / "refined_ocr.json"], env)
    sh([PY, cfg["recheck_padded"], "--ocr-json", out / "refined_ocr.json",
        "--out", out / "candidates.json", "--yolo-label-dir", out / "detector/labels",
        "--model-dir", cfg["numeric_ocr"], "--model-name", "PP-OCRv5_mobile_rec",
        "--device", "gpu", "--input-shape", "3,48,320", "--min-conf", 0.0,
        "--progress-every", 200], env)
    sel = Path(cfg["selector_dir"])
    sh([PY, f"{SRC}/temporal_channel_pipeline.py", "--images", flat,
        "--candidate-json", out / "candidates.json", "--sequence-manifest", manifest,
        "--yolo-label-dir", out / "detector/labels", "--out", out / "predictions.json",
        "--mode", "search", "--temporal-history-mode", "roi_prior_recheck",
        "--history-guided-model-dir", cfg["numeric_ocr"], "--history-guided-device", "gpu",
        "--history-guided-input-shape", "3,48,320", "--history-guided-raw-only",
        "--allow-single-digit-channel-recheck", "--suppress-text-hallucinations",
        "--ranker-model", sel / "selector_no_airtel_v1_pairwise_linear.json",
        "--value-group-ranker-model", sel / "value_group_pointwise_logistic.json",
        "--relative-gate-model", sel / "relative_confidence_gate.json",
        "--relative-gate-threshold", 0.7690914017301402,
        "--relative-gate-policy", "positive_first"], env)

    # 3) score against filename GT
    doc = json.loads((out / "predictions.json").read_text())
    norm = (lambda s: str(int(s)) if s else "") if args.numeric_equiv else (lambda s: s)
    rows = []
    per_group = defaultdict(lambda: [0, 0])
    ok_all = tot_all = 0
    for im in doc["images"]:
        iid = im["image_id"]
        pred = dg(im.get("predicted_channel_number") or "")
        g = gt.get(iid, "")
        correct = int(g != "" and norm(pred) == norm(g))
        rows.append({"folder": index.get(iid, ""), "image": iid,
                     "gt": g, "prediction": pred, "correct": correct})
        if g:
            per_group[index.get(iid, "")][0] += correct
            per_group[index.get(iid, "")][1] += 1
            ok_all += correct
            tot_all += 1

    with (out / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "image", "gt", "prediction", "correct"])
        w.writeheader()
        w.writerows(rows)
    summary = {
        "overall_accuracy": round(ok_all / tot_all * 100, 2) if tot_all else None,
        "n_images": tot_all,
        "match_mode": "numeric_equiv" if args.numeric_equiv else "exact",
        "by_folder": {g: round(c / t * 100, 2) for g, (c, t) in sorted(per_group.items()) if t},
        "note": "detection metrics not available (filename gives value only, no bbox GT)",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\n=== 최종 채널번호 정확도 (파일명 정답 기준) ===")
    print(f"전체: {summary['overall_accuracy']}%  ({tot_all}장, {summary['match_mode']})")
    for g, acc in summary["by_folder"].items():
        print(f"  {g}: {acc}%")
    print(f"\n결과: {out}/predictions.csv , summary.json")
    print("※ detection 성능은 bbox 정답이 없어 계산 불가 (파일명은 값만 제공)")


if __name__ == "__main__":
    main()
