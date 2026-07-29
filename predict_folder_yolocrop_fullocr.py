#!/usr/bin/env python3
"""Variant: YOLO ROI -> FULL OCR on the crop -> keep digit tokens (A/B experiment).

Targets the "merged label+number" problem (e.g. 'DirecTV 123' captured as ONE region):
  * detector finds WHERE the channel number is (its box may include the channel name)
  * FULL OCR (full charset) reads the crop, so letters stay letters and are not turned
    into fake digits
  * we keep only the pure-digit token(s) -> the channel number, no position assumption
  * selector + temporal decide the final value

This tends to beat both:
  - full-image OCR (slower; may merge label+number into one line),
  - numeric-only recheck (mangles letters into digits).

All flags/outputs identical to predict_folder.py.
"""
from __future__ import annotations

from pathlib import Path

import predict_folder as P


def run_pipeline_yolocrop(cfg, flat, out, manifest, device, batch, env, accum_frames=None):
    PY, SRC = cfg["python"], cfg["pipeline_src"]
    sel = Path(cfg["selector_dir"])
    accum_args = []
    if accum_frames and accum_frames > 0:
        alpha = round(2.0 / (accum_frames + 1), 4)
        accum_args = ["--smooth-alpha", str(alpha),
                      "--warmup-frames", str(accum_frames),
                      "--min-seen", str(max(2, min(3, accum_frames)))]

    # 1) detector
    rc = P.sh([PY, f"{SRC}/export_recursive_detector_predictions.py", "--model", cfg["detector"],
               "--images-dir", flat, "--output-dir", out / "detector", "--imgsz", cfg["imgsz"],
               "--device", device, "--batch", batch, "--candidate-conf", 0.05], env)
    if rc != 0:
        raise SystemExit(f"\n[yolocrop] detector 단계 실패 (rc={rc})")

    # 2) FULL OCR on each channel-box crop -> pure-digit candidates (fixes 'DirecTV 123')
    rc = P.sh([PY, f"{SRC}/fullocr_crops.py", "--images", flat,
               "--yolo-label-dir", out / "detector/labels", "--out", out / "candidates.json",
               "--pad", "0.15", "--min-height", "40", "--progress-every", "200"], env)
    if rc != 0:
        raise SystemExit(f"\n[yolocrop] full-OCR-on-crop 단계 실패 (rc={rc})")

    # 3) temporal + selector (identical to predict_folder.py)
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
        raise SystemExit(f"\n[yolocrop] temporal 단계 실패 (rc={rc})")


P.run_pipeline = run_pipeline_yolocrop

if __name__ == "__main__":
    P.main()
