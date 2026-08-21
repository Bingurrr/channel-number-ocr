#!/usr/bin/env python3
"""Test_overlay 실제 이미지에서 채널번호 ROI vs 다른 텍스트의 '시각적' 차이 측정.
   대비 / 채도 / 글자굵기(ink) / 주변 배경 밝기·채도(하이라이트) → 어떤 신호가 구별력 있나."""
import glob, json, re
from collections import defaultdict
import numpy as np
from PIL import Image

ROOT = "/home/irteam/teacher_model/dataset/Test_overlay_folder"
CLS = {"channel_box": "채널번호", "broadcast_box": "방송사명",
       "program_box": "프로그램명", "timeline_box": "시간대"}
SAMPLE = 8   # 폴더당 프레임 수


def feats_of(im, box):
    """ROI crop의 시각 특징."""
    W, H = im.size
    x1, y1, x2, y2 = [int(v) for v in box]
    if x2 <= x1 or y2 <= y1:
        return None
    a = np.asarray(im.crop((max(0, x1), max(0, y1), min(W, x2), min(H, y2))), float)
    if a.size < 30:
        return None
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    lum = 0.299*R + 0.587*G + 0.114*B
    mx = a.max(-1); mn = a.min(-1)
    sat = (mx - mn) / (mx + 1e-6)                          # 채도
    lf = lum.ravel()
    contrast = np.percentile(lf, 90) - np.percentile(lf, 10)   # 텍스트-배경 대비
    bg_lum = np.median(lf)                                     # 배경(다수 픽셀) 밝기
    ink = float(np.mean(np.abs(lf - bg_lum) > 45))            # 글자 픽셀 비율(굵기/밀도)
    # 주변(박스 좌우 바깥) 배경 = 행 하이라이트 감지용
    pw = max(3, (x2-x1)//6)
    sur = []
    for sx1, sx2 in [(x1-pw, x1), (x2, x2+pw)]:
        sx1 = max(0, sx1); sx2 = min(W, sx2)
        if sx2 > sx1:
            sur.append(np.asarray(im.crop((sx1, max(0, y1), sx2, min(H, y2))), float))
    if sur:
        s = np.concatenate([p.reshape(-1, 3) for p in sur], 0)
        sur_lum = float(np.median(0.299*s[:, 0]+0.587*s[:, 1]+0.114*s[:, 2]))
        smx = s.max(1); smn = s.min(1); sur_sat = float(np.mean((smx-smn)/(smx+1e-6)))
    else:
        sur_lum, sur_sat = bg_lum, 0.0
    return {"contrast": float(contrast), "sat_txt": float(np.mean(sat)),
            "ink": ink, "sur_lum": sur_lum, "sur_sat": sur_sat}


def txt_cls(a):
    return CLS.get(a.get("class"))


agg = defaultdict(lambda: defaultdict(list))
for fd in sorted(glob.glob(ROOT + "/UI_*")):
    jfs = sorted(glob.glob(fd + "/*.json"))[:SAMPLE]
    for jf in jfs:
        d = json.load(open(jf))
        img_p = jf[:-5] + ".jpg"
        try:
            im = Image.open(img_p).convert("RGB")
        except Exception:
            continue
        for a in d.get("annotations", []):
            c = txt_cls(a); b = a.get("bbox")
            if not c or not b or len(b) != 4:
                continue
            f = feats_of(im, b)
            if f:
                for k, v in f.items():
                    agg[c][k].append(v)

order = ["채널번호", "방송사명", "프로그램명", "시간대"]
names = [("텍스트-배경 대비", "contrast", "%.0f"), ("글자 채도", "sat_txt", "%.2f"),
         ("글자 굵기(ink비율)", "ink", "%.2f"), ("주변배경 밝기", "sur_lum", "%.0f"),
         ("주변배경 채도(하이라이트)", "sur_sat", "%.2f")]
print("=" * 92)
print(f"채널번호 ROI vs 다른 텍스트 — 시각 특징 (실제 이미지, 40 UI × {SAMPLE}프레임)")
print("=" * 92)
print(f"{'시각 특징':<24}" + "".join(f"{c:>15}" for c in order))
print("-" * 92)
for nm, key, fmt in names:
    line = f"{nm:<24}"
    for c in order:
        v = agg[c].get(key, [])
        line += f"{(fmt % (np.mean(v) if v else 0)):>15}"
    print(line)
print("=" * 92)
# 채널 vs '다른 텍스트 평균' 대비 배율(구별력)
oth = ["방송사명", "프로그램명", "시간대"]
print("채널 / 다른텍스트평균 배율 (1보다 크면 채널이 높음):")
for nm, key, _ in names:
    ch = np.mean(agg["채널번호"].get(key, [0]))
    om = np.mean([np.mean(agg[c].get(key, [0])) for c in oth])
    print(f"  {nm:<24} {ch/ (om+1e-9):.2f}x")
print("=" * 92)
