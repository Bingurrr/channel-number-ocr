#!/usr/bin/env python3
"""Folder-based channel-number inference (recursive; handles nested folders and
duplicate frame names like 001.jpg across folders).

Every image folder that contains images is treated as one group (= one UI /
capture). Frames in a group are run as a temporal sequence (channel position is
fixed per UI), so accumulation across frames stabilises detection.

    ROOT/.../<any depth>/<folder>/  001.jpg  002.jpg ...

Outputs:
  per_folder.csv  — one channel number per folder (majority vote over frames)
  per_frame.csv   — per-frame prediction

Model paths come from config.json (package-relative by default).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from collections import Counter, defaultdict
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


def collect(root: Path, flat: Path):
    """Recursively find images. Return (index: uid->group, meta: uid->orig_stem),
    symlinking each image into `flat` under a globally-unique uid name."""
    flat.mkdir(parents=True, exist_ok=True)
    index, meta, seqs = {}, {}, defaultdict(list)
    used = {}
    for img in sorted(root.rglob("*")):
        if not (img.is_file() and img.suffix.lower() in IMG_EXTS):
            continue
        group = str(img.parent.relative_to(root)) or "(root)"
        uid = f"{group}__{img.stem}".replace("/", "__").replace(" ", "_")
        while uid in used:                      # guarantee uniqueness
            uid += "_x"
        used[uid] = True
        link = flat / f"{uid}{img.suffix.lower()}"
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(img.resolve(), link)
        index[uid] = group
        meta[uid] = img.stem
        seqs[group].append(uid)
    return index, meta, seqs


def sh(cmd, env):
    print("RUN:", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd], env=env)


def run_pipeline(cfg, flat, out, manifest, device, env):
    PY, SRC = cfg["python"], cfg["pipeline_src"]
    sel = Path(cfg["selector_dir"])
    steps = [
        [PY, f"{SRC}/export_recursive_detector_predictions.py", "--model", cfg["detector"],
         "--images-dir", flat, "--output-dir", out / "detector", "--imgsz", cfg["imgsz"],
         "--device", device, "--batch", 64, "--candidate-conf", 0.05],
        [PY, f"{SRC}/run_paddleocr_export.py", "--images", flat, "--out", out / "full_ocr.json",
         "--use-gpu", "--ocr-version", "PP-OCRv4", "--text-detection-model-name",
         "PP-OCRv4_mobile_det", "--text-recognition-model-name", "en_PP-OCRv4_mobile_rec",
         "--progress-every", 200],
        [PY, f"{SRC}/refine_ocr_candidates.py", "--ocr-json", out / "full_ocr.json",
         "--out", out / "refined_ocr.json"],
        [PY, cfg["recheck_padded"], "--ocr-json", out / "refined_ocr.json",
         "--out", out / "candidates.json", "--yolo-label-dir", out / "detector/labels",
         "--model-dir", cfg["numeric_ocr"], "--model-name", "PP-OCRv5_mobile_rec",
         "--device", "gpu", "--input-shape", "3,48,320", "--min-conf", 0.0,
         "--progress-every", 200],
        [PY, f"{SRC}/temporal_channel_pipeline.py", "--images", flat,
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
         "--relative-gate-policy", "positive_first"],
    ]
    for cmd in steps:
        rc = sh(cmd, env)
        if rc != 0:
            raise SystemExit(f"\n[predict_folder] step failed (rc={rc}): {cmd[1]}\n"
                             "이 단계의 위 에러 메시지를 확인하세요 (보통 paddle 버전/GPU 문제).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    cfg = load_config()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/opt/conda/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    flat = out / "images"
    index, meta, seqs = collect(root, flat)
    if not index:
        raise SystemExit(f"이미지를 찾지 못했습니다: {root} (하위 폴더 포함 재귀 탐색함). "
                         "경로/확장자를 확인하세요.")
    print(f"images: {len(index)}  folders: {len(seqs)}", flush=True)
    manifest = out / "manifest.json"
    manifest.write_text(json.dumps(
        {"sequence_count": len(seqs),
         "sequences": [{"sequence_id": g.replace('/', '__'), "group_key": g,
                        "images": sorted(v)} for g, v in sorted(seqs.items())]},
        ensure_ascii=False))

    run_pipeline(cfg, flat, out, manifest, args.device, env)

    doc = json.loads((out / "predictions.json").read_text())
    per_frame, per_folder = [], defaultdict(list)
    for im in doc["images"]:
        uid = im["image_id"]
        pred = "".join(c for c in str(im.get("predicted_channel_number") or "") if c.isdigit())
        folder = index.get(uid, "")
        per_frame.append({"folder": folder, "frame": meta.get(uid, uid), "prediction": pred})
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
                            "confidence": round(n / len(preds), 3), "n_frames": len(preds)})
    with (out / "per_folder.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "channel_number", "confidence", "n_frames"])
        w.writeheader(); w.writerows(folder_rows)

    print("\n=== 폴더별 채널번호 (프레임 다수결) ===")
    for r in folder_rows:
        print(f"  {r['folder']}: {r['channel_number']}  (conf {r['confidence']}, {r['n_frames']}프레임)")
    print(f"\n결과: {out}/per_folder.csv , per_frame.csv")


if __name__ == "__main__":
    main()
