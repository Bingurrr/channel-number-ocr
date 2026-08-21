#!/usr/bin/env python3
"""점수 각 항 on/off ablation — Airtel_Labeled (실제 상업용 STB, held-out)."""
import json, re, sys
from collections import defaultdict
sys.path.insert(0, ".")
import slot_v4 as V4

OCR = "/tmp/claude-500/-home1-irteam/53ded4c1-f246-489d-aea8-0f9713428fd0/scratchpad/airtel_ocr/full_ocr.json"


def gt_of(stem):                       # Ch012__... → 12
    m = re.match(r"(?i)ch0*(\d+)", stem)
    return m.group(1) if m else None


def norm(s):
    s = re.sub(r"\D", "", str(s))
    return str(int(s)) if s else ""


d = json.load(open(OCR))
ims = sorted(d["images"], key=lambda x: x["image_id"])
ids = [x["image_id"] for x in ims]
print(f"airtel: {len(ims)}장 (실제 상업용)", flush=True)

V4W = dict(size_weight=0.4, bg_weight=0.3, contrast_weight=0.2, global_weight=0.5, sat_weight=0.0)
CONFIGS = [
    ("v3  (시각신호 전부 OFF)", dict(size_weight=0, bg_weight=0, contrast_weight=0, global_weight=0, sat_weight=0)),
    ("v4  (현재)",            dict(V4W)),
    ("v4 − 크기",             {**V4W, "size_weight": 0}),
    ("v4 − 배경하이라이트",      {**V4W, "bg_weight": 0}),
    ("v4 − 대비",             {**V4W, "contrast_weight": 0}),
    ("v4 − 전역현저성",         {**V4W, "global_weight": 0}),
    ("v4 + 채도(0.35)",       {**V4W, "sat_weight": 0.35}),
    ("채도만 (v3+채도)",        dict(size_weight=0, bg_weight=0, contrast_weight=0, global_weight=0, sat_weight=0.35)),
]


def run(weights):
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
    return c, t


print("=" * 60)
res = {}
for name, w in CONFIGS:
    c, t = run(w)
    res[name] = (c, t)
    print(f"  {name:<24} {c/max(1,t)*100:5.1f}%  ({c}/{t})", flush=True)
print("=" * 60)
v4 = res["v4  (현재)"]; v4a = v4[0]/max(1, v4[1])*100
print("각 항 기여 (v4 − 항제거):")
for label, key in [("크기", "v4 − 크기"), ("배경하이라이트", "v4 − 배경하이라이트"),
                   ("대비", "v4 − 대비"), ("전역현저성", "v4 − 전역현저성")]:
    a = res[key][0]/max(1, res[key][1])*100
    print(f"  {label:<12}: {v4a-a:+.1f}%p")
