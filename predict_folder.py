#!/usr/bin/env python3
"""Folder-based channel-number inference.

Input layout (each subfolder = one UI; images inside are frames of that UI):
    ROOT/
      UI_A/  123.jpg  20.jpg  45.jpg ...
      UI_B/  1.jpg    99.jpg  ...

For each subfolder the frames are treated as one temporal sequence (the channel
position is fixed per UI), so accumulation across frames stabilises detection.
Runs: detector -> full-image OCR -> numeric recheck (3:1 aspect padding) ->
selector -> temporal. Outputs per-frame and per-folder predictions.

This is a thin orchestrator over the frozen pipeline scripts + the trained models.
Model paths are read from config.json so the shipped package is self-contained.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config():
    cfg = HERE / "config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        # resolve package-relative paths against the repo root (HERE)
        for k in ("pipeline_src","detector","numeric_ocr","selector_dir","recheck_padded","full_image_ocr"):
            v = c.get(k)
            if v and not str(v).startswith("/"):
                c[k] = str((HERE / v).resolve())
        return c
    # fallback: current experiment defaults
    return {
        "python": "/home/irteam/teacher_model_v3_for_test/.venv/bin/python",
        "pipeline_src": "/home/irteam/teacher_model_v3_for_test/src",
        "detector": str(HERE / "models/detector/best.pt"),
        "numeric_ocr": str(HERE / "models/numeric_ocr/inference"),
        "selector_dir": "/home/irteam/teacher_model_v3_for_test/models/selector",
        "imgsz": 960,
        "recheck_padded": str(HERE / "paddleocr_channel_recheck_padded.py"),
    }


def sh(cmd, env):
    print("RUN:", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd], env=env)


def build_manifest(root: Path, out: Path):
    """One sequence per subfolder (= one UI)."""
    seqs, index = [], {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        frames = sorted(p.stem for p in sub.iterdir()
                        if p.suffix.lower() in IMG_EXTS)
        if not frames:
            continue
        seqs.append({"sequence_id": sub.name, "group_key": sub.name, "images": frames})
        for f in frames:
            index[f] = sub.name
    out.write_text(json.dumps({"sequence_count": len(seqs), "sequences": seqs},
                              ensure_ascii=False, indent=2))
    return index


def flatten_images(root: Path, flat: Path):
    """Symlink all frames into one dir with unique stems (stem must be global-unique)."""
    flat.mkdir(parents=True, exist_ok=True)
    seen = set()
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in sorted(sub.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            if img.stem in seen:
                raise SystemExit(f"Duplicate image name '{img.stem}' across folders; "
                                 "frame filenames must be globally unique.")
            seen.add(img.stem)
            link = flat / img.name
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(img.resolve(), link)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder of per-UI subfolders")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    cfg = load_config()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    PY, SRC = cfg["python"], cfg["pipeline_src"]

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/opt/conda/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    flat = out / "images"
    flatten_images(root, flat)
    manifest = out / "manifest.json"
    index = build_manifest(root, manifest)

    # 1) detector
    sh([PY, f"{SRC}/export_recursive_detector_predictions.py", "--model", cfg["detector"],
        "--images-dir", flat, "--output-dir", out / "detector", "--imgsz", cfg["imgsz"],
        "--device", args.device, "--batch", 64, "--candidate-conf", 0.05], env) or None
    # 2) full-image OCR
    sh([PY, f"{SRC}/run_paddleocr_export.py", "--images", flat, "--out", out / "full_ocr.json",
        "--use-gpu", "--ocr-version", "PP-OCRv4", "--text-detection-model-name",
        "PP-OCRv4_mobile_det", "--text-recognition-model-name", "en_PP-OCRv4_mobile_rec",
        "--progress-every", 200], env)
    # 3) refine
    sh([PY, f"{SRC}/refine_ocr_candidates.py", "--ocr-json", out / "full_ocr.json",
        "--out", out / "refined_ocr.json"], env)
    # 4) numeric recheck WITH 3:1 padding
    sh([PY, cfg["recheck_padded"], "--ocr-json", out / "refined_ocr.json",
        "--out", out / "candidates.json", "--yolo-label-dir", out / "detector/labels",
        "--model-dir", cfg["numeric_ocr"], "--model-name", "PP-OCRv5_mobile_rec",
        "--device", "gpu", "--input-shape", "3,48,320", "--min-conf", 0.0,
        "--progress-every", 200], env)
    # 5) temporal (frames of the same subfolder accumulate)
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

    # aggregate: per-frame + per-folder consensus
    doc = json.loads((out / "predictions.json").read_text())
    per_frame, per_folder = [], defaultdict(list)
    for im in doc["images"]:
        iid = im["image_id"]
        pred = "".join(c for c in str(im.get("predicted_channel_number") or "") if c.isdigit())
        folder = index.get(iid, "")
        per_frame.append({"folder": folder, "frame": iid, "prediction": pred})
        if pred:
            per_folder[folder].append(pred)

    with (out / "per_frame.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "frame", "prediction"])
        w.writeheader(); w.writerows(per_frame)

    folder_rows = []
    for folder, preds in sorted(per_folder.items()):
        cnt = Counter(preds)
        best, n = cnt.most_common(1)[0]
        folder_rows.append({"folder": folder, "channel_number": best,
                            "confidence": round(n / len(preds), 3),
                            "n_frames": len(preds), "votes": dict(cnt)})
    (out / "per_folder.json").write_text(json.dumps(folder_rows, ensure_ascii=False, indent=2))
    with (out / "per_folder.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "channel_number", "confidence", "n_frames"])
        w.writeheader()
        for r in folder_rows:
            w.writerow({k: r[k] for k in ("folder", "channel_number", "confidence", "n_frames")})

    print(f"\n=== 폴더별 채널번호 (프레임 다수결) ===")
    for r in folder_rows:
        print(f"  {r['folder']}: {r['channel_number']}  (conf {r['confidence']}, {r['n_frames']}프레임)")
    print(f"\n결과: {out}/per_folder.csv , per_frame.csv")


if __name__ == "__main__":
    main()
