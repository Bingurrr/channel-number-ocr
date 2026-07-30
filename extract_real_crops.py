#!/usr/bin/env python3
"""Extract REAL channel-number crops from a finished predict_folder_slot run.

Closes the synthetic->real gap: instead of only training on synthetic overlays,
crop the ACTUAL channel ROI from real labelled frames (filename = ground truth) and
train the recognizer on those. These crops carry the real look the model struggles
with (white digits, weak shadow, broadcaster name underneath, right-shifted).

Reuses the run's outputs (no OCR re-run needed):
  * profile_report.json  -> per-folder channel_field_box (the ROI)
  * per_frame.csv        -> (folder, frame) rows; frame stem digits = label

Output (PaddleOCR rec format):
  <out>/images/<label>_<hash>.jpg
  <out>/rec_gt.txt        "images/705_ab12.jpg\t705"
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def gt_of(stem):
    # 파이프라인 gt_from_name과 동일: '맨 앞' 숫자만 (101_2 -> 101, 모든 숫자 이어붙이면 안됨)
    m = re.match(r"\s*(\d+)", str(stem))
    return str(int(m.group(1))) if m else ""


def crop_box(img, box, pad, pad_right):
    """median 박스는 평균 너비라 긴(4자리) 채널은 오른쪽이 잘림 → 오른쪽을 특히 넉넉히."""
    W, H = img.size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = x2 - x1; ph = (y2 - y1) * pad
    cx1 = max(0, int(x1 - bw * pad))
    cx2 = min(W, int(x2 + bw * pad_right))        # 오른쪽 대폭 확장(잘림 방지)
    cy1, cy2 = max(0, int(y1 - ph)), min(H, int(y2 + ph))
    if cx2 - cx1 < 4 or cy2 - cy1 < 4:
        return None
    return img.crop((cx1, cy1, cx2, cy2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="predict_folder_slot --out 결과 디렉토리")
    ap.add_argument("--root", required=True, help="원본 이미지 루트(재귀 탐색)")
    ap.add_argument("--out", required=True, help="crop 저장 디렉토리")
    ap.add_argument("--pad", type=float, default=0.25, help="상/하/좌 여백")
    ap.add_argument("--pad-right", type=float, default=0.7, help="오른쪽 여백(4자리 잘림 방지, 크게)")
    ap.add_argument("--max-per-folder", type=int, default=0, help="폴더당 최대(0=전부)")
    args = ap.parse_args()

    res = Path(args.result)
    report = json.loads((res / "profile_report.json").read_text())
    boxes = {r["folder"]: r["channel_field_box"] for r in report if r.get("channel_field_box")}
    if not boxes:
        raise SystemExit("channel_field_box 없음 (먼저 predict_folder_slot 실행 필요)")

    # 원본 이미지: stem -> 경로
    stem2path = {}
    for p in Path(args.root).rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            stem2path.setdefault(p.stem, p)

    # per_frame.csv (folder, frame)
    rows = []
    cf = res / "per_frame.csv"
    with cf.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append((r.get("folder", ""), r.get("frame", "")))

    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True)
    gt_lines, made, per_folder_cnt, skip = [], 0, {}, 0
    for folder, frame in rows:
        box = boxes.get(folder) or (list(boxes.values())[0] if len(boxes) == 1 else None)
        gt = gt_of(frame)
        ip = stem2path.get(frame)
        if not box or not gt or ip is None:
            skip += 1; continue
        if args.max_per_folder and per_folder_cnt.get(folder, 0) >= args.max_per_folder:
            continue
        try:
            img = Image.open(ip).convert("RGB")
        except Exception:
            skip += 1; continue
        crop = crop_box(img, box, args.pad, args.pad_right)
        if crop is None:
            skip += 1; continue
        h = hashlib.md5(f"{frame}-{made}".encode()).hexdigest()[:10]
        fn = f"images/{gt}_{h}.jpg"
        crop.save(out / fn, quality=94)
        gt_lines.append(f"{fn}\t{gt}")
        made += 1; per_folder_cnt[folder] = per_folder_cnt.get(folder, 0) + 1

    (out / "rec_gt.txt").write_text("\n".join(gt_lines) + "\n")
    print(f"실제 crop {made}장 저장 (skip {skip}) → {out}", flush=True)
    print(f"폴더별: " + ", ".join(f"{Path(k).name or '.'}:{v}" for k, v in per_folder_cnt.items()), flush=True)
    for ln in gt_lines[:6]:
        print("  ", ln)


if __name__ == "__main__":
    main()
