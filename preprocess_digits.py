#!/usr/bin/env python3
"""Digit isolation by LEARNED font color (N-frame preprocessing).

zap 캡처에서 채널영역은 프레임마다 숫자는 바뀌지만 '폰트 색·플레이트'는 동일하다.
그래서 여러 프레임(같은 채널영역 crop)을 모아 폰트색을 학습하면, 각 프레임에서
숫자 픽셀만 남기고 배경(영상/플레이트)을 지울 수 있다 → rec에 깨끗한 입력.

핵심:
  1) 배경색 = crop 테두리 픽셀의 median (플레이트/영상은 테두리를 채움)
  2) 전경(숫자) 픽셀 = 배경색에서 충분히 먼 픽셀
  3) 폰트색 = N장의 전경 픽셀 median (영상 노이즈는 평균화되어 사라짐)
  4) isolate: 폰트색에 가까운 픽셀만 원본 유지, 나머지는 흰색 → tight crop

배경색이 숫자색과 거의 같은 극단은 색만으론 한계(그땐 위치정보 병행)지만, 대부분의
TV OSD(어두운 플레이트 + 밝은 숫자 / 뚜렷한 색 숫자)에서 대비를 확보해준다.

단독 사용:
    python preprocess_digits.py --crops ./crops --out ./crops_clean --viz ./viz
    (--crops: 채널영역 crop들; 같은 UI끼리 폴더로 두면 폴더별 폰트색 학습)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _bg_color(arr, frac=0.18):
    """테두리 픽셀 median = 배경(플레이트/영상)색."""
    h, w = arr.shape[:2]
    b = max(1, int(min(h, w) * frac))
    border = np.concatenate([arr[:b].reshape(-1, 3), arr[-b:].reshape(-1, 3),
                             arr[:, :b].reshape(-1, 3), arr[:, -b:].reshape(-1, 3)])
    return np.median(border, axis=0)


def _fg_mask(arr, bg, rel=0.35):
    """배경색에서 먼 픽셀 = 전경(숫자) 후보. rel: 최대거리 대비 비율 임계."""
    d = np.linalg.norm(arr - bg, axis=2)
    thr = max(30.0, d.max() * rel)
    return d >= thr


def learn_font_color(crops, rel=0.35):
    """여러 crop에서 폰트색·배경색 학습. 반환 (font_rgb, bg_rgb)."""
    fgs, bgs = [], []
    for c in crops:
        arr = np.asarray(c.convert("RGB"), dtype=np.float32)
        bg = _bg_color(arr)
        m = _fg_mask(arr, bg, rel)
        if m.sum() < 8:
            continue
        fgs.append(np.median(arr[m], axis=0)); bgs.append(bg)
    if not fgs:
        return None, None
    return np.median(np.stack(fgs), axis=0), np.median(np.stack(bgs), axis=0)


def isolate(crop, font, bg, pad=5, out_bg=(255, 255, 255), min_pixels=6):
    """폰트색에 가까운 픽셀만 남기고 나머지는 흰색 → tight crop.

    반환 (clean_img, mask_img). 학습 실패(font None)면 원본을 그대로 돌려준다.
    """
    arr = np.asarray(crop.convert("RGB"), dtype=np.float32)
    if font is None or bg is None:
        return crop.convert("RGB"), Image.new("L", crop.size, 0)
    df = np.linalg.norm(arr - font, axis=2)
    db = np.linalg.norm(arr - bg, axis=2)
    mask = (df < db) & (db > 25)                         # 폰트에 더 가깝고 배경과 충분히 다름
    # 잡티 제거 (median filter)
    mimg = Image.fromarray((mask * 255).astype("uint8")).filter(ImageFilter.MedianFilter(3))
    mask = np.asarray(mimg) > 127
    out = np.empty(arr.shape, dtype=np.uint8)            # 배경 = 흰색으로 채움
    out[:] = out_bg
    out[mask] = arr[mask].astype("uint8")                # 숫자 픽셀만 원본 유지
    ys, xs = np.where(mask)
    if len(xs) >= min_pixels:
        x0, x1 = max(0, xs.min() - pad), min(arr.shape[1], xs.max() + pad + 1)
        y0, y1 = max(0, ys.min() - pad), min(arr.shape[0], ys.max() + pad + 1)
        out = out[y0:y1, x0:x1]
    return Image.fromarray(out), Image.fromarray((mask * 255).astype("uint8"))


def make_viz(orig, mask, clean, font, bg, text=""):
    """[원본 | 마스크 | 분리결과] 나란히 + 학습된 색/인식결과 라벨."""
    h = 120
    def rs(im):
        im = im.convert("RGB"); w = max(1, int(im.width * h / max(1, im.height)))
        return im.resize((w, h))
    a, m, c = rs(orig), rs(mask), rs(clean)
    gap, sw = 10, 46
    W = a.width + m.width + c.width + gap * 3 + sw
    canvas = Image.new("RGB", (W, h + 22), (30, 30, 30))
    x = 0
    for im in (a, m, c):
        canvas.paste(im, (x, 0)); x += im.width + gap
    d = ImageDraw.Draw(canvas)
    if font is not None:
        d.rectangle([x, 4, x + sw, 4 + 20], fill=tuple(int(v) for v in font))       # 폰트색 견본
    if bg is not None:
        d.rectangle([x, 30, x + sw, 30 + 20], fill=tuple(int(v) for v in bg))       # 배경색 견본
    d.text((4, h + 4), f"orig | mask | isolated   read={text}", fill=(230, 230, 230))
    return canvas


def _load(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True, help="채널영역 crop 폴더(하위폴더=UI별)")
    ap.add_argument("--out", required=True, help="분리된 깨끗한 crop 저장")
    ap.add_argument("--viz", default=None, help="분리 과정 시각화 저장(옵션)")
    ap.add_argument("--viz-n", type=int, default=30, help="폴더당 시각화 장수")
    ap.add_argument("--rel", type=float, default=0.35)
    args = ap.parse_args()

    root = Path(args.crops)
    # 하위폴더별(=UI별)로 그룹핑. 파일이 root 바로 아래면 root 자체가 한 그룹.
    groups = {}
    for p in _load(root):
        groups.setdefault(p.parent, []).append(p)

    outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    vizd = Path(args.viz) if args.viz else None
    if vizd:
        vizd.mkdir(parents=True, exist_ok=True)

    for grp, paths in groups.items():
        crops = []
        for p in paths:
            try:
                crops.append(Image.open(p))
            except Exception:
                crops.append(None)
        valid = [c for c in crops if c is not None]
        font, bg = learn_font_color(valid, rel=args.rel)
        rel_name = grp.relative_to(root) if grp != root else Path(".")
        fs = "None" if font is None else tuple(int(v) for v in font)
        print(f"[{rel_name}] {len(valid)}장 학습  폰트색={fs}", flush=True)
        (outd / rel_name).mkdir(parents=True, exist_ok=True)
        if vizd:
            (vizd / rel_name).mkdir(parents=True, exist_ok=True)
        vn = 0
        for p, c in zip(paths, crops):
            if c is None:
                continue
            clean, mask = isolate(c, font, bg)
            clean.save(outd / rel_name / p.name)
            if vizd and vn < args.viz_n:
                make_viz(c, mask, clean, font, bg).save(vizd / rel_name / (p.stem + ".jpg"))
                vn += 1
    print(f"완료: 깨끗한 crop → {outd}" + (f" , 시각화 → {vizd}" if vizd else ""), flush=True)


if __name__ == "__main__":
    main()
