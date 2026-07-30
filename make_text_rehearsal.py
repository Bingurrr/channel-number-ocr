#!/usr/bin/env python3
"""Generate general Latin-text crops as REHEARSAL data for full-OCR rec fine-tuning.

When fine-tuning the main full-OCR recognizer to read channel digits better, we must
NOT let it forget letters/time/date (slot analysis needs them). This renders general
tokens the full OCR must keep reading -> mixed into the fine-tune set as rehearsal.

Token types (en_dict charset only: letters, digits, : - . / and space):
  word     News, Sports, Movie     (random letters)
  time     14:30, 9:05
  date     7/30, 12/25
  chan     CH12, Ch5, KBS2, MBC
  alnum    A12, 12B, HD1
  two      "News 9" (with space)

Rendered on solid/gradient or natural crop backgrounds (some clutter), tight-cropped.
Output: <out>/images/<idx>_<hash>.jpg  +  rec_gt.txt  ("images/..\tNews")
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
FILL_COLORS = [(255, 255, 255), (240, 240, 240), (255, 235, 130), (255, 215, 0),
               (210, 230, 255), (230, 230, 230), (255, 255, 200)]
CALLSIGNS = ["KBS", "MBC", "SBS", "CNN", "BBC", "HBO", "TVN", "JTBC", "EBS", "OCN",
             "News", "Sports", "Movie", "Drama", "Kids", "Live", "HD", "UHD", "TV"]


def load_fonts():
    fs = list(EXTRA_FONTS)
    for d in FONT_DIRS:
        fs += glob.glob(f"{d}/**/*.ttf", recursive=True)
    return [f for f in dict.fromkeys(fs) if Path(f).exists()]


def rand_token(rng):
    k = rng.random()
    if k < 0.28:                                   # word
        return "".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(3, 8)))
    if k < 0.46:                                   # time
        return f"{rng.randint(0, 23)}:{rng.randint(0, 59):02d}"
    if k < 0.60:                                   # date
        return f"{rng.randint(1, 12)}/{rng.randint(1, 28)}"
    if k < 0.78:                                   # channel-ish / callsign
        w = rng.choice(CALLSIGNS)
        return w + (str(rng.randint(1, 99)) if rng.random() < 0.6 else "")
    if k < 0.90:                                   # alnum
        a = "".join(rng.choice(string.ascii_uppercase) for _ in range(rng.randint(1, 2)))
        return (a + str(rng.randint(1, 99))) if rng.random() < 0.5 else (str(rng.randint(1, 99)) + a)
    return f"{rng.choice(CALLSIGNS)} {rng.randint(1, 99)}"   # two tokens (space)


def make_bg(rng, natural_bgs):
    """Clean background (solid / vertical gradient) so the token label is unambiguous.

    Rehearsal must keep labels clean (its job is to preserve letters/time reading);
    the overlap-digit set already provides the on-clutter signal. Natural digit crops
    are avoided here because their digits bleed into time/date tokens (label noise).
    """
    W, H = rng.randint(120, 260), rng.randint(40, 90)
    c1 = (rng.randint(8, 110), rng.randint(8, 110), rng.randint(8, 110))
    if rng.random() < 0.5:
        return Image.new("RGB", (W, H), c1)
    c2 = tuple(min(255, max(0, v + rng.randint(-60, 60))) for v in c1)   # 세로 그라데이션
    col = Image.new("RGB", (1, H))                                       # 1픽셀 열 → resize (빠름)
    cp = col.load()
    for y in range(H):
        f = y / max(1, H - 1)
        cp[0, y] = tuple(int(c1[i] * (1 - f) + c2[i] * f) for i in range(3))
    return col.resize((W, H))


def draw_token(bg, token, font_path, rng):
    im = bg.convert("RGB").copy()
    W, H = im.size
    d = ImageDraw.Draw(im)
    size = max(10, int(H * rng.uniform(0.55, 0.9)))
    for _ in range(6):
        font = ImageFont.truetype(font_path, size)
        l, t, r, b = d.textbbox((0, 0), token, font=font)
        tw, th = r - l, b - t
        if tw <= W * 0.98 or size <= 12:
            break
        size = int(size * 0.85)
    x = rng.randint(0, max(0, W - tw)) - l
    y = rng.randint(0, max(0, H - th)) - t
    d.text((x, y), token, font=font, fill=rng.choice(FILL_COLORS),
           stroke_width=max(1, size // 24), stroke_fill=(20, 20, 20))
    box = (x + l, y + t, x + l + tw, y + t + th)
    px = (box[2] - box[0]) * rng.uniform(0.15, 0.30)
    py = (box[3] - box[1]) * rng.uniform(0.22, 0.42)
    return im.crop((max(0, int(box[0] - px)), max(0, int(box[1] - py)),
                    min(W, int(box[2] + px)), min(H, int(box[3] + py))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/irteam/teacher_model/dataset/text_rehearsal")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--natural-src", default="/home/irteam/teacher_model/dataset/ocr_finetune/"
                    "open_images_channel_digits_41620_v3_regular/ocr_crops",
                    help="자연스러운 클러터 배경 소스(선택)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fonts = load_fonts()
    nat = glob.glob(f"{args.natural_src}/**/*.jpg", recursive=True)
    rng.shuffle(nat); nat = nat[:20000]
    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True)
    print(f"폰트 {len(fonts)}, 자연배경 {len(nat)} → {args.n}장", flush=True)

    lines = []
    for i in range(args.n):
        tok = rand_token(rng)
        im = draw_token(make_bg(rng, nat), tok, rng.choice(fonts), rng)
        h = hashlib.md5(f"{tok}-{i}-{rng.random()}".encode()).hexdigest()[:12]
        fn = f"images/{i:06d}_{h}.jpg"
        im.save(out / fn, quality=92)
        lines.append(f"{fn}\t{tok}")
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{args.n}", flush=True)
    (out / "rec_gt.txt").write_text("\n".join(lines) + "\n")
    print(f"완료 {args.n}장 → {out}", flush=True)
    for ln in lines[:8]:
        print("  ", ln)


if __name__ == "__main__":
    main()
