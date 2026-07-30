#!/usr/bin/env python3
"""Synthesize OVERLAP training crops for the channel-number recognizer.

Takes existing text/digit crops (e.g. the OpenImages channel-digit crops, which
already contain a number on a busy background) and OVERLAYS a fresh 1-5 digit
number on top -> exactly the real failure case: the channel number is rendered on
top of program/broadcaster text. The label is the TOP number we drew.

The overlay mimics an on-screen-display digit: bold font, high-contrast fill, a
thin dark outline for readability, prominent size, random position. Font / color /
size / position are randomized so the recognizer learns the glyph, not a UI.

Output:
    <out>/images/000000_<label>.jpg
    <out>/rec_gt.txt            "images/000000_705.jpg\t705"  (PaddleOCR rec format)
"""
from __future__ import annotations

import argparse
import glob
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_DIRS = [
    "/home/irteam/teacher_model/assets/google_fonts/ofl/gothica1",
    "/home/irteam/teacher_model/assets/google_fonts",
]
EXTRA_FONTS = ["/home1/irteam/.config/Ultralytics/Arial.ttf"]

# OSD 채널숫자에 흔한 밝은 고대비 색 + 몇 가지 유채색(어려운 케이스용)
FILL_COLORS = [(255, 255, 255), (245, 245, 245), (255, 235, 130), (255, 215, 0),
               (230, 230, 230), (200, 230, 255), (255, 255, 200), (180, 220, 255)]


def load_fonts():
    fs = list(EXTRA_FONTS)
    for d in FONT_DIRS:
        fs += glob.glob(f"{d}/**/*.ttf", recursive=True)
    seen, out = set(), []
    for f in fs:
        if f not in seen and Path(f).exists():
            seen.add(f); out.append(f)
    return out


def rand_number(rng):
    n = rng.randint(1, 5)
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


def draw_overlay(bg, label, font_path, rng):
    """Draw `label` on top of bg (a PIL RGB crop). Returns a new image."""
    im = bg.convert("RGB").copy()
    W, H = im.size
    d = ImageDraw.Draw(im)
    # 글자 높이를 crop 높이의 55~92%로 (채널숫자는 크게 표시됨)
    target_h = int(H * rng.uniform(0.55, 0.92))
    size = max(10, target_h)
    for _ in range(6):                       # 폭이 넘치면 크기 줄여 맞춤
        font = ImageFont.truetype(font_path, size)
        l, t, r, b = d.textbbox((0, 0), label, font=font)
        tw, th = r - l, b - t
        if tw <= W * 0.96 or size <= 12:
            break
        size = int(size * 0.85)
    x = rng.randint(0, max(0, W - tw)) - l
    y = rng.randint(0, max(0, H - th)) - t
    fill = rng.choice(FILL_COLORS)
    # 가독성용 얇은 외곽선(어두운색) — OSD 글자 특성
    outline = (20, 20, 20)
    ow = max(1, size // 22)
    d.text((x, y), label, font=font, fill=fill,
           stroke_width=ow, stroke_fill=outline)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/irteam/teacher_model/dataset/ocr_finetune/"
                    "open_images_channel_digits_41620_v3_regular/ocr_crops",
                    help="배경으로 쓸 crop 이미지 루트(재귀 탐색)")
    ap.add_argument("--out", default="/home/irteam/teacher_model/dataset/overlap_number_ocr")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-bg-h", type=int, default=28, help="너무 작은 배경 crop 제외")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fonts = load_fonts()
    if not fonts:
        raise SystemExit("폰트 없음")
    bgs = glob.glob(f"{args.src}/**/*.jpg", recursive=True) + \
        glob.glob(f"{args.src}/**/*.png", recursive=True)
    if not bgs:
        raise SystemExit(f"배경 crop 없음: {args.src}")
    rng.shuffle(bgs)
    print(f"배경 후보 {len(bgs)}장, 폰트 {len(fonts)}개 → {args.n}장 생성", flush=True)

    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True)
    gt_lines, made, bi = [], 0, 0
    while made < args.n and bi < len(bgs):
        bp = bgs[bi]; bi += 1
        try:
            bg = Image.open(bp).convert("RGB")
        except Exception:
            continue
        if bg.height < args.min_bg_h:
            continue
        label = rand_number(rng)
        im = draw_overlay(bg, label, rng.choice(fonts), rng)
        fn = f"images/{made:06d}_{label}.jpg"
        im.save(out / fn, quality=92)
        gt_lines.append(f"{fn}\t{label}")
        made += 1
    (out / "rec_gt.txt").write_text("\n".join(gt_lines) + "\n")
    (out / "digits_dict.txt").write_text("\n".join(str(i) for i in range(10)) + "\n")
    print(f"완료: {made}장 → {out}/images , 라벨 → {out}/rec_gt.txt", flush=True)
    for ln in gt_lines:
        print("  ", ln)


if __name__ == "__main__":
    main()
