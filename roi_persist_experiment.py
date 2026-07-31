#!/usr/bin/env python3
"""[실험] 채널 ROI에서 '프레임간 새로 나타난 픽셀(영상)'을 지우고 '지속된 픽셀(UI)'만 남긴다.

동기: none이 많다. 채널 위치는 잘 찾는다. 그래서 채널 ROI만 잘라, 이전 프레임들과
비교해서 '색이 새로 등장한' 픽셀(=로딩된 영상)을 제거하면 UI 채널숫자가 돋보여 OCR이
none을 안 낼 것이다.

whole-image가 아니라 ROI 한정이라 '평평한 영상 배경' 함정을 피한다.

각 delay-triplet에 대해 ROI를 잘라 여러 방식 비교 viz 저장:
    [delay1 | delay2 | delay3 | MIN(darker) | persist(지속만) | persist_on_white]
persist = 프레임간 변화 큰 픽셀(새 영상) 제거, 변화 작은 픽셀(지속 UI)만 남김.

ROI는 정규화 좌표(--roi x1,y1,x2,y2). 기본 = 좌하단 채널번호 근처.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def triplet_key(name):
    gt = re.match(r"(Ch\d+)", name)
    slot = re.search(r"captured_(Ch\d+)_", name)
    pos = re.search(r"(findPosition\d+)", name)
    return (gt.group(1) if gt else "Ch?",
            (gt.group(1) if gt else "?", slot.group(1) if slot else "?", pos.group(1) if pos else "?"))


def crop_roi(im, roi):
    W, H = im.size
    x1, y1, x2, y2 = roi
    return im.crop((int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)))


def persist_clean(rois, chg_thr=30, on_white=False):
    """프레임간 변화 큰 픽셀(새 영상) 제거, 지속 픽셀만 남김.

    변화량 = max-min. 변화 > 임계 → 새로 나타난 영상 → 배경(검정/흰)으로 지움.
    남은 값은 median(지속 UI). on_white=True면 지운 자리를 흰색으로.
    """
    mx = reduce(ImageChops.lighter, rois)
    mn = reduce(ImageChops.darker, rois)
    chg = ImageChops.subtract(mx, mn).convert("L")           # 프레임간 변화
    persist = chg.point(lambda v: 255 if v < chg_thr else 0).filter(ImageFilter.MedianFilter(3))
    if len(rois) == 3:
        a, b, c = rois
        med = ImageChops.lighter(ImageChops.darker(a, b), ImageChops.darker(ImageChops.lighter(a, b), c))
    else:
        med = mn
    bg = Image.new("RGB", med.size, (255, 255, 255) if on_white else (0, 0, 0))
    return Image.composite(med, bg, persist)


def make_row(rois, mn, persist_blk, persist_wht, up=3):
    def rs(im):
        return im.resize((im.width * up, im.height * up))
    panels = [rs(r) for r in rois] + [rs(mn), rs(persist_blk), rs(persist_wht)]
    labels = [f"delay{i+1}" for i in range(len(rois))] + ["MIN", "persist(검정)", "persist(흰)"]
    h = max(p.height for p in panels); gap = 8
    W = sum(p.width for p in panels) + gap * (len(panels) + 1)
    cv = Image.new("RGB", (W, h + 18), (20, 20, 20)); d = ImageDraw.Draw(cv); x = gap
    for p, l in zip(panels, labels):
        cv.paste(p, (x, 0)); d.text((x, h + 3), l, fill=(230, 230, 230)); x += p.width + gap
    return cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--roi", default="0.02,0.66,0.32,0.82",
                    help="정규화 채널 ROI x1,y1,x2,y2 (기본=좌하단 채널번호 근처)")
    ap.add_argument("--chg-thr", type=int, default=30, help="이 값보다 변화 크면 '새 영상'으로 제거")
    ap.add_argument("--viz-n", type=int, default=20)
    args = ap.parse_args()

    roi = tuple(float(x) for x in args.roi.split(","))
    root = Path(args.root)
    groups = defaultdict(list)
    for p in sorted(root.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            groups[triplet_key(p.name)[1]].append(p)

    outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    vn = 0
    for key, paths in sorted(groups.items()):
        if len(paths) < 2 or vn >= args.viz_n:
            continue
        rois = [crop_roi(Image.open(p).convert("RGB"), roi) for p in paths]
        base = rois[0].size
        rois = [r if r.size == base else r.resize(base) for r in rois]
        mn = reduce(ImageChops.darker, rois)
        pb = persist_clean(rois, args.chg_thr, on_white=False)
        pw = persist_clean(rois, args.chg_thr, on_white=True)
        make_row(rois, mn, pb, pw).save(outd / f"{key[0]}__{key[1]}_{key[2]}.jpg", quality=92)
        vn += 1
    print(f"완료: ROI 실험 {vn}장 → {outd}  (roi={roi}, chg_thr={args.chg_thr})", flush=True)


if __name__ == "__main__":
    main()
