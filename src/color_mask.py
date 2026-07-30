#!/usr/bin/env python3
"""Color masking to read a channel number that is overlaid on top of program /
broadcaster text (true pixel overlap inside the channel ROI).

Idea (uses "past inference" to help OCR):
  * The channel digits change every zap, but their FONT COLOR is constant across
    frames. So we learn the channel stroke color from frames where it was read
    cleanly (learn_text_color), then on a cluttered frame we keep only pixels near
    that color (apply_color_mask) -> the underlying program text (a different color)
    disappears -> a clean high-contrast digit remains for the recognizer.

Pure-PIL (no numpy/cv2) so it runs anywhere PaddleOCR's env has Pillow. ROIs are
tiny, so per-pixel Python loops are fine.
"""
from __future__ import annotations

from pathlib import Path


def learn_text_color(crop):
    """Estimate the (r,g,b) of the text strokes in a clean channel-ROI crop.

    Text strokes are the high-contrast MINORITY of pixels (bright-on-dark or
    dark-on-bright). Returns None when the crop has no clear high-contrast text.
    """
    g = crop.convert("L")
    gp = list(g.getdata())
    n = len(gp)
    if n < 20:
        return None
    s = sorted(gp)
    med = s[n // 2]
    lo, hi = s[int(n * 0.05)], s[int(n * 0.95)]
    if hi - lo < 25:                     # 대비가 거의 없음 → 텍스트 색 못 정함
        return None
    bright_thr = med + max(20, (hi - med) * 0.5)
    dark_thr = med - max(20, (med - lo) * 0.5)
    rgb = list(crop.convert("RGB").getdata())
    bright = [rgb[i] for i, v in enumerate(gp) if v >= bright_thr]
    dark = [rgb[i] for i, v in enumerate(gp) if v <= dark_thr]
    floor = max(8, int(n * 0.01))
    cands = [c for c in (bright, dark) if len(c) >= floor]
    if not cands:
        return None
    text = min(cands, key=len)           # 글자 획 = 소수의 고대비 픽셀
    k = len(text)
    return (sum(p[0] for p in text) // k, sum(p[1] for p in text) // k,
            sum(p[2] for p in text) // k)


def apply_color_mask(crop, color, tol=70):
    """Keep only pixels near `color`; return an "L" image = black text on white.

    tol is Euclidean color distance in RGB. Larger tol keeps more (safer when the
    color estimate is rough); smaller tol removes more background.
    """
    rgb = crop.convert("RGB")
    px = rgb.getdata()
    tr, tg, tb = color
    t2 = tol * tol
    out = bytearray(len(rgb.getdata()))
    for i, (r, g, b) in enumerate(px):
        d = (r - tr) * (r - tr) + (g - tg) * (g - tg) + (b - tb) * (b - tb)
        out[i] = 0 if d <= t2 else 255    # 채널색=검정(글자), 나머지=흰(배경)
    from PIL import Image
    m = Image.new("L", rgb.size)
    m.putdata(list(out))
    return m


def crop_roi(img, box, pad=0.2):
    """Crop box with padding (fraction of box size), clamped to the image."""
    W, H = img.size
    b = [float(v) for v in box]
    pw = (b[2] - b[0]) * pad
    ph = (b[3] - b[1]) * pad
    x1, y1 = max(0, int(b[0] - pw)), max(0, int(b[1] - ph))
    x2, y2 = min(W, int(b[2] + pw)), min(H, int(b[3] + ph))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return img.crop((x1, y1, x2, y2))


def upscale(im, min_height=120):
    u = max(1.0, min_height / max(1, im.height))
    if u <= 1.0:
        return im, 1.0
    return im.resize((max(1, int(im.width * u)), max(1, int(im.height * u)))), u


def mask_steps(img, box, sdir, color=None, read_val="", pad=0.2, min_height=120, tol=70):
    """Save per-step images: ROI -> crop -> (learned color) -> masked -> upscaled(read)."""
    from PIL import ImageDraw
    sdir = Path(sdir); sdir.mkdir(parents=True, exist_ok=True)
    b = [int(v) for v in box]

    im1 = img.copy(); d = ImageDraw.Draw(im1)
    d.rectangle(b, outline=(0, 220, 0), width=3)
    d.text((8, 8), "1) channel ROI (overlap case)", fill=(255, 255, 0))
    im1.save(sdir / "1_roi.jpg", quality=90)

    crop = crop_roi(img, box, pad)
    if crop is None:
        return
    crop.save(sdir / "2_crop.jpg")

    col = color or learn_text_color(crop)
    if col is not None:
        from PIL import Image, ImageDraw as _D
        swatch = Image.new("RGB", (200, 60), tuple(col))
        _D.Draw(swatch).text((6, 22), f"channel color {tuple(col)}", fill=(255, 255, 255))
        swatch.save(sdir / "3_learned_color.jpg")
        masked = apply_color_mask(crop, col, tol)
        masked.save(sdir / "4_masked.jpg")
        up, u = upscale(masked, min_height)
    else:                                 # 색 못 배우면 원본 crop 확대(폴백)
        up, u = upscale(crop, min_height)
    from PIL import ImageDraw as _D2
    up = up.convert("RGB"); d = _D2.Draw(up)
    d.text((4, 4), f"5) masked+upscaled x{u:.1f}" + (f" -> read:{read_val}" if read_val else ""),
           fill=(0, 180, 0))
    up.save(sdir / "5_masked_read.jpg", quality=90)
