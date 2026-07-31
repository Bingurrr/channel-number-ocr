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


def learn_font_color_hist(crops, quant=24, min_frac=0.10):
    """[사용자 방식] 채널박스에서 '일정비율(min_frac)↑ + 프레임간 일관되게' 나오는 색 = 폰트색.

    배경(영상)은 프레임마다 바뀌어 여러 프레임에 걸쳐 일관되게 안 나옴 → 걸러짐.
    폰트색은 매 프레임 같은 위치·같은 색 → 일관성 높음 → 선택됨.
    반환: font_rgb(np.float32) 또는 None.
    """
    from collections import Counter
    presence, weight = Counter(), Counter()             # 색→(등장 프레임수, 누적비율)
    for c in crops:
        arr = np.asarray(c.convert("RGB"))
        q = (arr.astype(np.int32) // quant) * quant + quant // 2     # 색 양자화
        flat = q.reshape(-1, 3)
        colors, counts = np.unique(flat, axis=0, return_counts=True)
        n = len(flat)
        for col, cnt in zip(colors, counts):
            if cnt / n >= min_frac:
                key = (int(col[0]), int(col[1]), int(col[2]))
                presence[key] += 1; weight[key] += cnt / n
    if not presence:
        return None
    best = max(presence, key=lambda k: (presence[k], weight[k]))     # 일관성 우선, 비율 차선
    return np.array(best, dtype=np.float32)


def learn_font_color_temporal(crops, range_thr=28, quant=20, min_static=8):
    """[개선] 같은 채널 프레임들에서 '색+위치가 고정된 픽셀'(=글자획)의 색 = 폰트색.

    배경(영상)은 위치가 움직여 프레임간 변함 → 제외. 글자는 색·위치 고정 → 남음.
    '10% 색'만으론 자주 나오는 배경을 오인하므로, 위치 일관성(시간축 저변화)을 먼저 건다.
    crops는 '같은 채널' 프레임들의 동일 ROI여야 함(정렬 가정). 반환 font_rgb 또는 None.
    """
    from collections import Counter
    base = crops[0].size
    arrs = [np.asarray((c if c.size == base else c.resize(base)).convert("RGB"), dtype=np.float32) for c in crops]
    stack = np.stack(arrs, axis=0)                       # (N,H,W,3)
    rng = stack.max(0) - stack.min(0)                    # 프레임간 변화
    static = rng.max(axis=2) < range_thr                 # 색+위치 고정 픽셀
    if static.sum() < min_static:
        return None
    mean = stack.mean(0)                                 # 고정영역 평균색
    q = (mean.astype(np.int32) // quant) * quant + quant // 2
    cols = q[static].reshape(-1, 3)
    cnt = Counter(map(tuple, cols.tolist()))
    # 가장 흔한 '고정색'이 글자. 단, 너무 어두운(플레이트/검정) 것은 차선으로.
    ranked = cnt.most_common()
    best = ranked[0][0]
    return np.array(best, dtype=np.float32)


def _otsu(gray):
    h, _ = np.histogram(gray, 256, (0, 256)); tot = gray.size
    s = float((np.arange(256) * h).sum()); sB = wB = 0.0; best = (-1.0, 128)
    for t in range(256):
        wB += h[t]
        if wB == 0:
            continue
        wF = tot - wB
        if wF == 0:
            break
        sB += t * h[t]; mB = sB / wB; mF = (s - sB) / wF
        v = wB * wF * (mB - mF) ** 2
        if v > best[0]:
            best = (v, t)
    return best[1]


def learn_font_color_otsu(crops, min_fg=6):
    """[정답] 타이트 숫자박스에서 Otsu 전경분리 → (폰트색, 배경색). 극성은 테두리로 판정.

    글자획은 항상 소수 픽셀이라 '최빈색'은 배경을 잡는다(틀림). 대신 밝기 이분(Otsu) 후
    '테두리를 채운 쪽=배경', 반대쪽=글자로 보고 각 픽셀 색의 median을 반환.
    폰트색+배경색 둘 다 줘야 isolate가 '최근접'으로 비슷한 색도 잘 가른다.
    crops = 성공 detection의 타이트 숫자박스들. 반환 (font_rgb, bg_rgb) 또는 (None, None).
    """
    fgs, bgs = [], []
    for c in crops:
        arr = np.asarray(c.convert("RGB"), dtype=np.float32)
        if arr.shape[0] < 4 or arr.shape[1] < 4:
            continue
        g = arr.mean(axis=2)
        thr = _otsu(g)
        bw = max(1, int(min(g.shape) * 0.18))
        border = np.concatenate([g[:bw].ravel(), g[-bw:].ravel(), g[:, :bw].ravel(), g[:, -bw:].ravel()])
        bg_is_bright = border.mean() > thr                  # 배경이 밝은 쪽인가
        fg = (g <= thr) if bg_is_bright else (g > thr)      # 글자 = 배경 반대쪽
        if fg.sum() < min_fg or (~fg).sum() < min_fg:
            continue
        fgs.append(np.median(arr[fg], axis=0)); bgs.append(np.median(arr[~fg], axis=0))
    if not fgs:
        return None, None
    return (np.median(np.stack(fgs), axis=0).astype(np.float32),
            np.median(np.stack(bgs), axis=0).astype(np.float32))


def isolate_contrast(crop, font, bg=None, tol=60, margin=1.0, pad=5, min_pixels=6):
    """[사용자 방식] 폰트색 픽셀만 원색 유지, 나머지는 '폰트색 대비색'으로 채움.

    배경색(bg)을 주면 '폰트 vs 배경 최근접'으로 판정 → 폰트에 다소 비슷해도 배경에 더
    가까우면 지워짐(고정 tol이 큰 문제 해결). bg 없으면 절대거리 tol 사용.
    margin: font에 이만큼 '더' 가까워야 글자로 인정(>1이면 보수적으로 더 지움).
    흰 폰트 → 검정 배경, 어두운 폰트 → 흰 배경 → 고대비. 반환 (clean, mask).
    """
    arr = np.asarray(crop.convert("RGB"), dtype=np.float32)
    if font is None:
        return crop.convert("RGB"), Image.new("L", crop.size, 0)
    df = np.linalg.norm(arr - font, axis=2)
    if bg is not None:
        db = np.linalg.norm(arr - bg, axis=2)
        mask = df * margin < db                             # 폰트에 더 가까운 픽셀만
    else:
        mask = df < tol
    lum = 0.299 * font[0] + 0.587 * font[1] + 0.114 * font[2]
    bgc = (0, 0, 0) if lum > 110 else (255, 255, 255)       # 폰트 밝으면 배경 검정
    out = np.empty(arr.shape, dtype=np.uint8); out[:] = bgc
    out[mask] = arr[mask].astype(np.uint8)                  # 폰트픽셀은 원색 유지
    ys, xs = np.where(mask)
    if len(xs) >= min_pixels:
        x0, x1 = max(0, xs.min() - pad), min(arr.shape[1], xs.max() + pad + 1)
        y0, y1 = max(0, ys.min() - pad), min(arr.shape[0], ys.max() + pad + 1)
        out = out[y0:y1, x0:x1]
    return Image.fromarray(out), Image.fromarray((mask * 255).astype("uint8"))


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
