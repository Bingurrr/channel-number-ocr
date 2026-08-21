#!/usr/bin/env python3
"""dump_ocr_cache — Test_overlay_folder 전체를 full OCR 한 번만 돌려 후보를 캐시한다.

score 항별 ablation을 OCR 재실행 없이 몇 초 만에 반복하기 위한 전처리.
bench_v5.py와 동일한 모델 설정(PP-OCRv4_mobile_det + en_PP-OCRv4_mobile_rec_ft)을 쓴다.
출력: {folder: {"ids": [...], "gts": [...], "frames": [slot 입력 dict, ...]}}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

try:
    import torch  # noqa: F401  (paddle/torch 로드 순서)
except Exception:
    pass

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from ocr_candidate_extractor import create_paddle_ocr, extract_candidates_from_image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def gt_of(stem):
    m = re.match(r"0*(\d+)", stem)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/irteam/teacher_model/dataset/Test_overlay_folder")
    ap.add_argument("--out", default="ocr_cache.json")
    ap.add_argument("--rec-model-dir", default="models/full_image_ocr/en_PP-OCRv4_mobile_rec_ft")
    ap.add_argument("--folders", type=int, default=0)
    ap.add_argument("--per-folder", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    folders = sorted(d for d in root.iterdir() if d.is_dir())
    if args.folders:
        folders = folders[:args.folders]

    rec_dir = args.rec_model_dir if Path(args.rec_model_dir).exists() else None
    ocr = create_paddle_ocr(lang="en", use_gpu=True, ocr_version="PP-OCRv4",
                            text_detection_model_name="PP-OCRv4_mobile_det",
                            text_recognition_model_name="en_PP-OCRv4_mobile_rec",
                            text_recognition_model_dir=rec_dir)

    out = {}
    for fi, fd in enumerate(folders):
        paths = sorted(p for p in fd.iterdir() if p.suffix.lower() in IMG_EXTS)
        if args.per_folder:
            paths = paths[:args.per_folder]
        if not paths:
            continue
        frames = []
        for p in paths:
            with Image.open(p) as im:
                W, H = im.size
            cands = extract_candidates_from_image(p, ocr=ocr)
            frames.append({"image_id": p.stem, "image_path": str(p),
                           "image_width": W, "image_height": H,
                           "candidates": [c.to_dict() for c in cands]})
        ids = [p.stem for p in paths]
        out[fd.name] = {"ids": ids, "gts": [gt_of(i) for i in ids], "frames": frames}
        print(f"[{fi+1}/{len(folders)}] {fd.name}  frames={len(frames)}", flush=True)

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False))
    print(f"saved {args.out}  folders={len(out)}")


if __name__ == "__main__":
    main()
