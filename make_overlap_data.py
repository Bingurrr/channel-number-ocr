#!/usr/bin/env python3
"""Synthesize OVERLAP training crops for the channel-number recognizer.

Draws a fresh 1-5 digit number ON TOP of a busy background -> the real failure
case: the channel number rendered over program/broadcaster text. Label = the TOP
number we drew.

Two background modes:
  digit_crop : reuse existing digit crops (OpenImages) as background (already busy).
  text       : render random Korean/English words as the background, then overlay.

Output (filename encodes the label, like the source dataset):
    <out>/images/<label>_<hash>.jpg
    <out>/rec_gt.txt      "images/705_ab12ef34.jpg\t705"   (PaddleOCR rec format)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import random
import string
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_DIRS = ["/home/irteam/teacher_model/assets/google_fonts/ofl/gothica1",
             "/home/irteam/teacher_model/assets/google_fonts"]
EXTRA_FONTS = ["/home1/irteam/.config/Ultralytics/Arial.ttf"]
KOREAN_FONT_DIR = "/home/irteam/teacher_model/assets/google_fonts/ofl/gothica1"

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


def korean_fonts():
    return glob.glob(f"{KOREAN_FONT_DIR}/*.ttf") or load_fonts()


def rand_number(rng):
    return "".join(str(rng.randint(0, 9)) for _ in range(rng.randint(1, 5)))


def rand_bg_text(rng):
    """program-title-like mix of Korean syllables and English words."""
    def kw():
        return "".join(chr(rng.randint(0xAC00, 0xD7A3)) for _ in range(rng.randint(2, 4)))

    def ew():
        return "".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(3, 7)))
    parts = [kw() if rng.random() < 0.6 else ew() for _ in range(rng.randint(1, 3))]
    return " ".join(parts)


def make_text_bg(rng, font_path):
    """Render random Korean/English text on a colored background -> RGB image."""
    W, H = rng.randint(120, 260), rng.randint(44, 96)
    base = (rng.randint(10, 90), rng.randint(10, 90), rng.randint(10, 90))
    im = Image.new("RGB", (W, H), base)
    d = ImageDraw.Draw(im)
    txt = rand_bg_text(rng)
    fs = max(14, int(H * rng.uniform(0.4, 0.7)))
    try:
        font = ImageFont.truetype(font_path, fs)
    except Exception:
        font = ImageFont.load_default()
    tcol = (rng.randint(120, 220), rng.randint(120, 220), rng.randint(120, 220))
    d.text((rng.randint(2, 12), rng.randint(2, max(2, H - fs - 2))), txt, font=font, fill=tcol)
    return im


def draw_overlay(bg, label, font_path, rng):
    """Draw `label` on top of bg; return (image, overlay_bbox)."""
    im = bg.convert("RGB").copy()
    W, H = im.size
    d = ImageDraw.Draw(im)
    size = max(10, int(H * rng.uniform(0.6, 0.95)))
    for _ in range(6):                       # 폭 넘치면 축소
        font = ImageFont.truetype(font_path, size)
        l, t, r, b = d.textbbox((0, 0), label, font=font)
        tw, th = r - l, b - t
        if tw <= W * 0.98 or size <= 12:
            break
        size = int(size * 0.85)
    x = rng.randint(0, max(0, W - tw)) - l
    y = rng.randint(0, max(0, H - th)) - t
    ow = max(1, size // 22)
    d.text((x, y), label, font=font, fill=rng.choice(FILL_COLORS),
           stroke_width=ow, stroke_fill=(20, 20, 20))
    return im, (x + l, y + t, x + l + tw, y + t + th)


def tight_crop(im, bbox, rng):
    """Crop around the overlay with a small margin (여백 최소화, 겹침은 남김)."""
    W, H = im.size
    x1, y1, x2, y2 = bbox
    px = (x2 - x1) * rng.uniform(0.15, 0.30)
    py = (y2 - y1) * rng.uniform(0.22, 0.42)
    return im.crop((max(0, int(x1 - px)), max(0, int(y1 - py)),
                    min(W, int(x2 + px)), min(H, int(y2 + py))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/irteam/teacher_model/dataset/ocr_finetune/"
                    "open_images_channel_digits_41620_v3_regular/ocr_crops")
    ap.add_argument("--out", default="/home/irteam/teacher_model/dataset/overlap_number_ocr")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bg-mode", choices=["digit_crop", "text"], default="digit_crop")
    ap.add_argument("--min-bg-h", type=int, default=28)
    ap.add_argument("--append", action="store_true", help="rec_gt.txt에 이어붙임(모드 섞을 때)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fonts = load_fonts()
    kfonts = korean_fonts()
    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True)

    if args.bg_mode == "digit_crop":
        bgs = glob.glob(f"{args.src}/**/*.jpg", recursive=True) + \
            glob.glob(f"{args.src}/**/*.png", recursive=True)
        if not bgs:
            raise SystemExit(f"배경 crop 없음: {args.src}")
        rng.shuffle(bgs)
        print(f"[digit_crop] 배경 {len(bgs)}장 → {args.n}장", flush=True)
    else:
        print(f"[text] 한글/영어 배경 렌더 → {args.n}장", flush=True)

    gt_lines, made, bi = [], 0, 0
    while made < args.n:
        if args.bg_mode == "digit_crop":
            if bi >= len(bgs):
                break
            bp = bgs[bi]; bi += 1
            try:
                bg = Image.open(bp).convert("RGB")
            except Exception:
                continue
            if bg.height < args.min_bg_h:
                continue
        else:
            bg = make_text_bg(rng, rng.choice(kfonts))
        label = rand_number(rng)
        im, box = draw_overlay(bg, label, rng.choice(fonts), rng)
        im = tight_crop(im, box, rng)
        h = hashlib.md5(f"{label}-{made}-{rng.random()}".encode()).hexdigest()[:12]
        fn = f"images/{label}_{h}.jpg"
        im.save(out / fn, quality=92)
        gt_lines.append(f"{fn}\t{label}")
        made += 1

    gt = out / "rec_gt.txt"
    mode = "a" if args.append and gt.exists() else "w"
    with gt.open(mode) as f:
        f.write("\n".join(gt_lines) + "\n")
    (out / "digits_dict.txt").write_text("\n".join(str(i) for i in range(10)) + "\n")
    print(f"완료: {made}장 ({args.bg_mode}) → {out}/images , 라벨 {mode} → {gt}", flush=True)
    for ln in gt_lines:
        print("  ", ln)


if __name__ == "__main__":
    main()
