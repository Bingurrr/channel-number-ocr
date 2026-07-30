#!/usr/bin/env python3
"""Synthesize channel crops where the number sits next to broadcaster/program text,
in the real Korean-STB style (from a verbal description, NOT copyrighted frames).

Covers: [eng+num] [num+eng] [kor+num] [num+kor], font same/diff, color same/diff,
white with a weak drop shadow, right-shifted, 720p-ish degradation.

Charset note: the deployed rec is en_PP-OCRv4_mobile_rec (en_dict, NO Korean). So the
LABEL is the Latin-representable content only — English text is kept, Korean text is
dropped (the rec learns to skip Korean pixels and still read the number). Examples:
  "5 KBS"  -> label "5 KBS"      "뉴스 7" -> label "7"      "7 드라마" -> label "7"
The pipeline's best_digit then extracts the channel number from the read.

Output (PaddleOCR rec format): <out>/images/<label>_<hash>.jpg + rec_gt.txt
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIRS = ["/home/irteam/teacher_model/assets/google_fonts/ofl/gothica1",
             "/home/irteam/teacher_model/assets/google_fonts"]
EXTRA = ["/home1/irteam/.config/Ultralytics/Arial.ttf"]
KOREAN_FONT_DIR = "/home/irteam/teacher_model/assets/google_fonts/ofl/gothica1"

ENG = ["KBS", "MBC", "SBS", "EBS", "JTBC", "TVN", "OCN", "YTN", "MBN", "CH", "HD",
       "UHD", "SKY", "BTV", "News", "Sports", "Movie", "Drama", "Kids", "Live",
       "TV", "Plus", "World", "Home", "Life", "Gold", "Prime", "Star", "One"]
KOR = ["뉴스", "드라마", "영화", "스포츠", "예능", "다큐", "음악", "교육", "어린이", "만화",
       "홈쇼핑", "쇼핑", "특집", "생방송", "시사", "경제", "여행", "요리", "건강", "연예",
       "방송", "재방송", "게임", "바둑", "낚시", "키즈", "가요", "코미디", "골프", "낚시"]
_CONS = "bcdfghjklmnpqrstvwxyz"
_VOW = "aeiou"


def rand_eng(rng):
    """간단하고 읽기 쉬운(발음 가능한) 랜덤 영어 단어 — 고정 목록 반복 대신 다양성."""
    w = "".join(rng.choice(_CONS) + rng.choice(_VOW) + (rng.choice(_CONS) if rng.random() < 0.35 else "")
                for _ in range(rng.randint(2, 4)))
    r = rng.random()
    return w.upper() if r < 0.4 else (w.capitalize() if r < 0.85 else w)
WHITES = [(255, 255, 255), (248, 248, 248), (240, 240, 240), (250, 248, 245),
          (245, 248, 255), (236, 238, 236)]
COLORS = WHITES + [(255, 235, 130), (255, 215, 0), (200, 230, 255), (180, 220, 255)]


def load_fonts():
    fs = list(EXTRA)
    for d in FONT_DIRS:
        fs += glob.glob(f"{d}/**/*.ttf", recursive=True)
    return [f for f in dict.fromkeys(fs) if Path(f).exists()]


def korean_fonts():
    return glob.glob(f"{KOREAN_FONT_DIR}/*.ttf") or load_fonts()


def rand_number(rng):
    return "".join(str(rng.randint(0, 9)) for _ in range(rng.randint(1, 4)))


def build_tokens(rng):
    """Return [(text, type), ...] in draw order and the Latin-only label."""
    num = rand_number(rng)
    r = rng.random()
    if r < 0.25:                       # 숫자 단독
        toks = [(num, "num")]
    elif r < 0.43:                     # 한국어 동반 (18%로 줄임)
        comp = (rng.choice(KOR), "kor")   # 65% 텍스트-왼쪽 → 숫자 오른쪽 치우침
        toks = [comp, (num, "num")] if rng.random() < 0.65 else [(num, "num"), comp]
    else:                              # 영어 동반 (57%로 늘림, 랜덤단어 위주)
        word = rng.choice(ENG) if rng.random() < 0.35 else rand_eng(rng)   # 65% 랜덤
        comp = (word, "eng")
        toks = [comp, (num, "num")] if rng.random() < 0.65 else [(num, "num"), comp]
    label = " ".join(t for t, ty in toks if ty in ("num", "eng"))
    return toks, label


def shadowed(d, xy, text, font, fill, rng):
    off = rng.choice([1, 1, 2])
    d.text((xy[0] + off, xy[1] + off), text, font=font, fill=(rng.randint(10, 45),) * 3)
    d.text(xy, text, font=font, fill=fill)


def degrade(im, rng):
    W, H = im.size
    if rng.random() < 0.7:
        s = rng.uniform(0.45, 0.82)
        im = im.resize((max(8, int(W * s)), max(6, int(H * s)))).resize((W, H))
    if rng.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    return im


def make(rng, fonts, kfonts):
    W, H = rng.randint(150, 320), rng.randint(40, 84)
    base = (rng.randint(6, 55), rng.randint(6, 55), rng.randint(6, 60))
    im = Image.new("RGB", (W, H), base)
    d = ImageDraw.Draw(im)
    toks, label = build_tokens(rng)

    fs = max(14, int(H * rng.uniform(0.6, 0.92)))
    num_font_path = rng.choice(fonts)
    same_font = rng.random() < 0.5                 # 폰트 같음/다름
    same_color = rng.random() < 0.5                # 색상 같음/다름
    num_color = rng.choice(COLORS)
    # 토큰별 폰트/색
    def font_for(ty, size):
        p = num_font_path if (same_font or ty == "num") else \
            (rng.choice(kfonts) if ty == "kor" else rng.choice(fonts))
        if ty == "kor":                            # 한글은 한글폰트 필수
            p = num_font_path if (same_font and num_font_path in kfonts) else rng.choice(kfonts)
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            return ImageFont.truetype(rng.choice(kfonts), size)
    # 레이아웃: 가로로 이어 그림, 숫자는 오른쪽으로 치우치는 경향
    gap = rng.randint(3, 14)
    sizes = {"num": fs, "eng": int(fs * rng.uniform(0.5, 0.9)), "kor": int(fs * rng.uniform(0.5, 0.85))}
    widths = []
    for t, ty in toks:
        f = font_for(ty, sizes[ty]); l, tp, r, b = d.textbbox((0, 0), t, font=f)
        widths.append((t, ty, f, r - l, b - tp, l, tp))
    total = sum(w[3] for w in widths) + gap * (len(widths) - 1)
    # 숫자 우측 치우침 강조: 내용을 오른쪽에 배치(왼쪽 빈 공간). 단독숫자도 자주 우측.
    if len(toks) > 1:
        x = int((W - total) * rng.uniform(0.35, 0.9))
    else:
        x = int((W - total) * rng.uniform(0.45, 0.92))
    x = max(2, min(x, W - total - 2))
    for t, ty, f, tw, th, l, tp in widths:
        y = (H - th) // 2 - tp
        col = num_color if (same_color or ty == "num") else rng.choice(COLORS)
        shadowed(d, (x - l, y), t, f, col, rng)
        x += tw + gap
    im = degrade(im, rng)
    return im, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/irteam/teacher_model/dataset/broadcaster_ocr")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    fonts, kfonts = load_fonts(), korean_fonts()
    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(args.n):
        im, label = make(rng, fonts, kfonts)
        h = hashlib.md5(f"{label}-{i}-{rng.random()}".encode()).hexdigest()[:10]
        safe = label.replace(" ", "_") or "x"
        fn = f"images/{safe}_{h}.jpg"
        im.save(out / fn, quality=88)
        lines.append(f"{fn}\t{label}")
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{args.n}", flush=True)
    (out / "rec_gt.txt").write_text("\n".join(lines) + "\n")
    print(f"완료 {args.n}장 → {out}", flush=True)
    for ln in lines[:10]:
        print("  ", ln)


if __name__ == "__main__":
    main()
