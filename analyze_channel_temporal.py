#!/usr/bin/env python3
"""Test_overlay: 같은 UI를 여러 프레임 '누적'했을 때 채널번호 텍스트의 시간축 특징.
   (위치/크기는 얼마나 고정? 값은 얼마나 바뀜? 얼마나 자주 등장?)"""
import glob, json, re
from collections import defaultdict
import numpy as np

ROOT = "/home/irteam/teacher_model/dataset/Test_overlay_folder"
CLS = {"channel_box": "채널번호", "broadcast_box": "방송사명",
       "program_box": "프로그램명", "timeline_box": "시간대"}


def txt_of(a):
    at = a.get("attributes", {}) or {}
    for k in ("channel_number", "broadcast_name", "program_name", "timeline", "text"):
        if at.get(k):
            return str(at[k])
    return ""


# folder -> cls -> lists across frames
acc = defaultdict(lambda: defaultdict(lambda: {"cx": [], "cy": [], "h": [], "val": [], "n": 0}))
nframe = defaultdict(int)
for fd in sorted(glob.glob(ROOT + "/UI_*")):
    for jf in glob.glob(fd + "/*.json"):
        nframe[fd] += 1
        d = json.load(open(jf)); res = d.get("resolution", {})
        W = res.get("width", 1280); H = res.get("height", 720)
        for a in d.get("annotations", []):
            b = a.get("bbox"); cls = a.get("class")
            if not b or len(b) != 4 or cls not in CLS:
                continue
            x1, y1, x2, y2 = b
            if x2 <= x1 or y2 <= y1:
                continue
            s = acc[fd][CLS[cls]]
            s["cx"].append((x1+x2)/2/W); s["cy"].append((y1+y2)/2/H)
            s["h"].append((y2-y1)/H); s["val"].append(txt_of(a)); s["n"] += 1

# 클래스별 누적통계 집계
agg = defaultdict(lambda: defaultdict(list))
for fd, cm in acc.items():
    nf = max(1, nframe[fd])
    for c, s in cm.items():
        if s["n"] < 3:
            continue
        pos_std = float(np.std(s["cx"]) + np.std(s["cy"]))                 # 위치 흔들림(정규화)
        h_cv = float(np.std(s["h"]) / (np.mean(s["h"]) + 1e-9))            # 크기 변동계수
        distinct_ratio = len(set(s["val"])) / len(s["val"])               # 값 변화율
        coverage = s["n"] / nf                                            # 등장 프레임 비율
        agg[c]["pos_std"].append(pos_std)
        agg[c]["h_cv"].append(h_cv)
        agg[c]["distinct"].append(distinct_ratio)
        agg[c]["cov"].append(coverage)

order = ["채널번호", "방송사명", "프로그램명", "시간대"]
print("=" * 88)
print("여러 프레임 '누적' 시 특징 (같은 UI 폴더 내, 40 UI 평균)")
print("=" * 88)
rows = [
    ("위치 흔들림 (std, ↓고정)", "pos_std", "%.4f"),
    ("크기 변동계수 (↓일정)",     "h_cv",    "%.3f"),
    ("값 변화율 (distinct/총, ↑바뀜)", "distinct", "%.2f"),
    ("등장 프레임 비율 (↑항상)",   "cov",     "%.2f"),
]
print(f"{'누적 특징':<26}" + "".join(f"{c:>15}" for c in order))
print("-" * 88)
for name, key, fmt in rows:
    line = f"{name:<26}"
    for c in order:
        v = agg[c].get(key, [])
        line += f"{(fmt % (np.mean(v) if v else 0)):>15}"
    print(line)
print("=" * 88)
