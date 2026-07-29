#!/usr/bin/env python3
"""Same as predict_folder.py but WITHOUT full-image OCR (A/B experiment).

Normal pipeline builds the candidate pool from full-image PaddleOCR (every text on
screen) and then refines. This variant SKIPS that: candidates come ONLY from the
detector's channel_number / channel_number_area boxes (class 0/3), cropped and read
by the numeric digit recognizer. Use it to test whether full-image OCR actually
helps vs. hurts on your data.

All flags/outputs are identical to predict_folder.py (--gt-from-filename,
--split-output, --no-label, --accumulate, --accum-frames, --viz-history, ...).
Only the internal pipeline differs (no run_paddleocr_export / refine step).

Usage (same as predict_folder.py):
    python predict_folder_nofullocr.py --root ... --out ... --gt-from-filename --split-output
"""
from __future__ import annotations

import json
from pathlib import Path

import predict_folder as P


def run_pipeline_nofullocr(cfg, flat, out, manifest, device, batch, env, accum_frames=None):
    PY, SRC = cfg["python"], cfg["pipeline_src"]
    sel = Path(cfg["selector_dir"])
    accum_args = []
    if accum_frames and accum_frames > 0:
        alpha = round(2.0 / (accum_frames + 1), 4)
        accum_args = ["--smooth-alpha", str(alpha),
                      "--warmup-frames", str(accum_frames),
                      "--min-seen", str(max(2, min(3, accum_frames)))]

    # 1) detector — YOLO boxes (channel_number=0, channel_number_area=3, ...)
    rc = P.sh([PY, f"{SRC}/export_recursive_detector_predictions.py", "--model", cfg["detector"],
               "--images-dir", flat, "--output-dir", out / "detector", "--imgsz", cfg["imgsz"],
               "--device", device, "--batch", batch, "--candidate-conf", 0.05], env)
    if rc != 0:
        raise SystemExit(f"\n[nofullocr] detector 단계 실패 (rc={rc})")

    # 2) minimal image list INSTEAD of full-image OCR (no run_paddleocr_export / refine)
    imgs = sorted(p for p in flat.iterdir() if p.suffix.lower() in P.IMG_EXTS)
    minimal = {"images": [{"image_id": p.stem, "image_path": str(p)} for p in imgs]}
    (out / "minimal_ocr.json").write_text(json.dumps(minimal, ensure_ascii=False))
    print(f"[nofullocr] full-image OCR 생략 — YOLO 채널박스만 사용 ({len(imgs)} images)", flush=True)

    # 3) numeric recheck — crop ONLY YOLO class 0/3 regions and read digits (--yolo-only)
    rc = P.sh([PY, cfg["recheck_padded"], "--ocr-json", out / "minimal_ocr.json",
               "--out", out / "candidates.json", "--yolo-label-dir", out / "detector/labels",
               "--model-dir", cfg["numeric_ocr"], "--model-name", "PP-OCRv5_mobile_rec",
               "--device", "gpu", "--input-shape", "3,48,320", "--min-conf", 0.0,
               "--yolo-only", "--progress-every", 200], env)
    if rc != 0:
        raise SystemExit(f"\n[nofullocr] recheck 단계 실패 (rc={rc})")

    # 4) temporal + selector (identical to predict_folder.py)
    rc = P.sh([PY, f"{SRC}/temporal_channel_pipeline.py", "--images", flat,
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
               "--relative-gate-policy", "positive_first", *accum_args], env)
    if rc != 0:
        raise SystemExit(f"\n[nofullocr] temporal 단계 실패 (rc={rc})")


# swap the pipeline; everything else (collect/qualitative/summary/flags) is reused
P.run_pipeline = run_pipeline_nofullocr

if __name__ == "__main__":
    P.main()
