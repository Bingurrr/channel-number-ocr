#!/usr/bin/env python3
"""점수 각 항 on/off ablation — Test_overlay(기존 full_ocr.json 재사용).
   v3(시각신호 전부 OFF) → v4(현재) → 각 항 빼기 → +채도, UI별/전체 정확도 비교."""
import json, re, sys
from collections import defaultdict
sys.path.insert(0, ".")
import slot_v4 as V4

OCR = "/tmp/claude-500/-home1-irteam/53ded4c1-f246-489d-aea8-0f9713428fd0/scratchpad/testfolder_v4/full_ocr.json"


def gt_of(stem):                       # UI_01__0204_xxxx → 204
    tail = stem.split("__", 1)[1] if "__" in stem else stem
    m = re.match(r"0*(\d+)", tail)
    return m.group(1) if m else None


def norm(s):
    s = re.sub(r"\D", "", str(s))
    return str(int(s)) if s else ""


# 그룹핑
d = json.load(open(OCR))
groups = defaultdict(list)
for im in d["images"]:
    groups[im["image_id"].split("__", 1)[0]].append(im)

# 설정들 (각 항 on/off). base = v4 현재.
V4W = dict(size_weight=0.4, bg_weight=0.3, contrast_weight=0.2, global_weight=0.5, sat_weight=0.0)
CONFIGS = [
    ("v3  (시각신호 전부 OFF)", dict(size_weight=0, bg_weight=0, contrast_weight=0, global_weight=0, sat_weight=0)),
    ("v4  (현재: 크기+배경+대비+전역)", dict(V4W)),
    ("v4 − 크기",        {**V4W, "size_weight": 0}),
    ("v4 − 배경하이라이트", {**V4W, "bg_weight": 0}),
    ("v4 − 대비",        {**V4W, "contrast_weight": 0}),
    ("v4 − 전역현저성",    {**V4W, "global_weight": 0}),
    ("v4 + 채도(0.35)",  {**V4W, "sat_weight": 0.35}),
    ("채도만 (v3+채도)",   dict(size_weight=0, bg_weight=0, contrast_weight=0, global_weight=0, sat_weight=0.35)),
]


def run(weights):
    per_ui = {}; tot = [0, 0]
    for ui in sorted(groups):
        ims = sorted(groups[ui], key=lambda x: x["image_id"]); ids = [x["image_id"] for x in ims]
        r = V4.rolling_analyze(ims, ids, window=24, **weights)
        pf = r["per_frame"] if r else {}
        c = t = 0
        for i in ids:
            g = gt_of(i)
            if not g:
                continue
            t += 1
            if norm(pf.get(i, "")) == norm(g):
                c += 1
        per_ui[ui] = (c, t); tot[0] += c; tot[1] += t
    return per_ui, tot


print("설정별 실행 중...", flush=True)
results = {}
for name, w in CONFIGS:
    pu, tot = run(w)
    results[name] = (pu, tot)
    print(f"  {name:<28} 전체 {tot[0]/max(1,tot[1])*100:5.1f}%  ({tot[0]}/{tot[1]})", flush=True)

# UI별 비교표 저장
base = results["v4  (현재: 크기+배경+대비+전역)"][0]
uis = sorted(base)
import csv
out = "/tmp/claude-500/-home1-irteam/53ded4c1-f246-489d-aea8-0f9713428fd0/scratchpad/ablation_terms.csv"
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["UI"] + [n for n, _ in CONFIGS])
    for ui in uis:
        row = [ui]
        for n, _ in CONFIGS:
            pu = results[n][0][ui]
            row.append(round(pu[0] / max(1, pu[1]) * 100, 1))
        w.writerow(row)
print("\nUI별 표 →", out)

# 각 항의 '기여도' = v4 − (그 항 뺀 것)
print("\n=== 각 항의 기여 (v4 전체 − 항 제거 시) ===")
v4o = results["v4  (현재: 크기+배경+대비+전역)"][1]
v4a = v4o[0] / max(1, v4o[1]) * 100
for label, key in [("크기", "v4 − 크기"), ("배경하이라이트", "v4 − 배경하이라이트"),
                   ("대비", "v4 − 대비"), ("전역현저성", "v4 − 전역현저성")]:
    t = results[key][1]; a = t[0] / max(1, t[1]) * 100
    print(f"  {label:<12}: {v4a-a:+.1f}%p  (빼면 {a:.1f}%)")
