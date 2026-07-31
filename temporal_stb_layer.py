#!/usr/bin/env python3
"""Separate the set-top-box (STB) UI layer from the TV video using delay-triplets.

Airtel 캡처: 한 채널을 delay 2000/3000/4000(ms) 3장으로 찍는다. 3장은 STB 오버레이가
동일하고 TV 영상만 다르다(영상이 점점 로딩됨; delay=2000은 대개 검정). 따라서 한 묶음:
    STB 오버레이(채널번호·배너·라벨) = 고정 → 프레임간 변화 없음
    TV 영상                          = 변함 → 프레임간 변화 큼
영상은 로딩되며 점점 밝아지고(delay=2000은 대개 검정) STB 오버레이는 밝은 편이라,
픽셀 시간축 MIN(darker)이 밝은 영상을 검정으로 눌러 제거하고 STB 레이어만 남긴다.
반투명 배너(pixel = a·UI + (1-a)·video)에서도 MIN은 영상항이 가장 작은(검정) 프레임을
골라 α·UI(순수 UI)를 복원한다. 실측: median은 다수결로 영상을 남기지만 MIN은 완전 제거.

numpy 없이 PIL(ImageChops)만으로 계산 → on-device 친화적.
묶음 키 = (GT채널 Ch###, captured_Ch#, findPosition#) — delay/프레임번호만 다른 3장.

출력(--out): composites/{GT}__{key}.jpg (median = OCR 입력용),
             viz/... [3장 | median | 정적마스크 | STB-only] 나란히.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def triplet_key(name):
    """delay/프레임번호만 다른 3장을 하나로 묶는 키 + GT채널 추출."""
    gt = re.match(r"(Ch\d+)", name)
    slot = re.search(r"captured_(Ch\d+)_", name)
    pos = re.search(r"(findPosition\d+)", name)
    gt_s = gt.group(1) if gt else "Ch?"
    return gt_s, (gt_s, slot.group(1) if slot else "?", pos.group(1) if pos else "?")


def _median(ims):
    """PIL만으로 픽셀 median. N=3은 정확, 그외는 평균으로 근사."""
    if len(ims) == 1:
        return ims[0]
    if len(ims) == 3:
        a, b, c = ims                                        # median(a,b,c)
        return ImageChops.lighter(ImageChops.darker(a, b),
                                  ImageChops.darker(ImageChops.lighter(a, b), c))
    if len(ims) == 2:
        return ImageChops.blend(ims[0], ims[1], 0.5)
    acc = ims[0]                                             # N>3: 순차 평균
    for i, im in enumerate(ims[1:], start=2):
        acc = ImageChops.blend(acc, im, 1.0 / i)
    return acc


def separate(paths, mode="min"):
    """프레임들 → (composite, min, median). 크기 다르면 첫장 기준 리사이즈.

    mode='min'(기본): MIN darker = STB 레이어만 남기고 밝은 영상 제거 (실측 최고).
    mode='median'  : 시간축 median (다수결; 영상 로딩되면 못 지움 → 비교용).
    """
    ims, base = [], None
    for p in paths:
        im = Image.open(p).convert("RGB")
        if base is None:
            base = im.size
        elif im.size != base:
            im = im.resize(base)
        ims.append(im)
    mn = reduce(ImageChops.darker, ims)                     # 픽셀 최소 = 밝은영상 제거
    med = _median(ims)
    comp = mn if mode == "min" else med
    return comp, mn, med


def make_viz(paths, mn, med):
    """[원본들 | MIN(STB-only) | median] 나란히."""
    h = 130
    def rs(im):
        im = im.convert("RGB"); w = max(1, int(im.width * h / max(1, im.height)))
        return im.resize((w, h))
    panels = [rs(Image.open(p)) for p in paths] + [rs(mn), rs(med)]
    labels = [f"delay{i+1}" for i in range(len(paths))] + ["MIN=STB-only", "median(비교)"]
    gap = 8
    W = sum(p.width for p in panels) + gap * (len(panels) + 1)
    canvas = Image.new("RGB", (W, h + 20), (25, 25, 25))
    d = ImageDraw.Draw(canvas); x = gap
    for p, lab in zip(panels, labels):
        canvas.paste(p, (x, 0)); d.text((x, h + 4), lab, fill=(220, 220, 220)); x += p.width + gap
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="delay-triplet 이미지 폴더 (예: Airtel_Labeled)")
    ap.add_argument("--out", required=True, help="composites/ 와 viz/ 저장 위치")
    ap.add_argument("--mode", choices=["min", "median"], default="min",
                    help="min=MIN darker(STB만 남김, 기본) / median=비교용")
    ap.add_argument("--viz-n", type=int, default=40, help="시각화 묶음 수")
    ap.add_argument("--min-frames", type=int, default=2, help="묶음 최소 프레임 수")
    args = ap.parse_args()

    root = Path(args.root)
    groups, gt_of = defaultdict(list), {}
    for p in sorted(root.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        gt, key = triplet_key(p.name)
        groups[key].append(p); gt_of[key] = gt

    outd = Path(args.out); comp = outd / "composites"; viz = outd / "viz"
    comp.mkdir(parents=True, exist_ok=True); viz.mkdir(parents=True, exist_ok=True)
    print(f"묶음 {len(groups)}개 (이미지 {sum(len(v) for v in groups.values())}장)", flush=True)

    vn = made = 0
    for key, paths in sorted(groups.items()):
        if len(paths) < args.min_frames:
            continue
        composite, mn, med = separate(paths, args.mode)
        tag = f"{gt_of[key]}__{key[1]}_{key[2]}"
        gt_digits = re.sub(r"\D", "", gt_of[key])            # GT를 파일명 앞에 → --gt-from-filename
        composite.save(comp / f"{gt_digits}__{tag}.jpg", quality=92)
        made += 1
        if vn < args.viz_n:
            make_viz(paths, mn, med).save(viz / f"{tag}.jpg", quality=90); vn += 1
        if made % 50 == 0:
            print(f"  진행 {made} 묶음", flush=True)

    print(f"완료: median 합성 {made}장 → {comp}\n      시각화 {vn}장 → {viz}", flush=True)


if __name__ == "__main__":
    main()
