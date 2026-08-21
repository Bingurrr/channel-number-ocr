#!/usr/bin/env python3
"""Test_overlay_folder: 채널번호 ROI가 다른 텍스트와 뭐가 다른지 라벨 기반 정량 분석."""
import glob, json, re, statistics as st
from collections import defaultdict
import numpy as np

ROOT = "/home/irteam/teacher_model/dataset/Test_overlay_folder"
CLS = {"channel_box": "채널번호", "broadcast_box": "방송사명", "program_box": "프로그램명",
       "timeline_box": "시간대"}


def lum(hexc):
    h = (hexc or "").lstrip("#")
    if len(h) != 6:
        return None
    try:
        r, g, b = [int(h[i:i+2], 16) for i in (0, 2, 4)]
    except Exception:
        return None
    return 0.299*r + 0.587*g + 0.114*b


def txt_of(a):
    at = a.get("attributes", {}) or {}
    for k in ("channel_number", "broadcast_name", "program_name", "timeline", "text"):
        if at.get(k):
            return str(at[k])
    return ""


feat = defaultdict(lambda: defaultdict(list))     # cls -> feature -> [values]
per_folder_vals = defaultdict(lambda: defaultdict(list))  # folder -> cls -> [values]
rel_largest = {"채널번호": [0, 0]}                # [프레임에서 최대높이인 횟수, 전체]
rel_ratio = []                                    # 채널높이 / 그프레임 다른텍스트 중앙 높이

folders = sorted(glob.glob(ROOT + "/UI_*"))
for fd in folders:
    for jf in glob.glob(fd + "/*.json"):
        d = json.load(open(jf))
        res = d.get("resolution", {}); W = res.get("width", 1280); H = res.get("height", 720)
        rows = []
        for a in d.get("annotations", []):
            b = a.get("bbox")
            if not b or len(b) != 4:
                continue
            cls = a.get("class")
            if cls not in CLS:
                continue
            x1, y1, x2, y2 = b; w = x2-x1; h = y2-y1
            if w <= 0 or h <= 0:
                continue
            t = txt_of(a); digits = re.sub(r"\D", "", t); alnum = re.sub(r"[^0-9A-Za-z]", "", t)
            r = {
                "cls": CLS[cls],
                "h_norm": h/H, "w_norm": w/W, "fs": a.get("font size") or 0,
                "aspect": w/max(1, h), "cx": (x1+x2)/2/W, "cy": (y1+y2)/2/H,
                "lum": lum(a.get("font color")), "nchar": len(t.replace(" ", "")),
                "digit_ratio": len(digits)/max(1, len(alnum)) if alnum else (1.0 if digits else 0),
                "pure_digit": 1.0 if (t and re.fullmatch(r"\d{1,5}", t.strip())) else 0.0,
                "text": t,
            }
            rows.append(r)
            c = r["cls"]
            for k in ("h_norm", "w_norm", "fs", "aspect", "cx", "cy", "nchar", "digit_ratio", "pure_digit"):
                feat[c][k].append(r[k])
            if r["lum"] is not None:
                feat[c]["lum"].append(r["lum"])
            per_folder_vals[fd][c].append(r["text"])
        # 프레임 내 상대크기: 채널이 최대 높이인가
        ch = [r for r in rows if r["cls"] == "채널번호"]
        others = [r for r in rows if r["cls"] != "채널번호"]
        if ch and others:
            chh = ch[0]["h_norm"]; oh = [r["h_norm"] for r in others]
            rel_largest["채널번호"][1] += 1
            if chh >= max(oh):
                rel_largest["채널번호"][0] += 1
            rel_ratio.append(chh / max(1e-9, st.median(oh)))

# ── 값 다양성: 폴더 내 distinct 비율 ──
div = defaultdict(list)
for fd, cm in per_folder_vals.items():
    for c, vals in cm.items():
        if vals:
            div[c].append(len(set(vals))/len(vals))

def ms(v):
    v = [x for x in v if x is not None]
    return (np.mean(v), np.std(v)) if v else (0, 0)

print("="*92)
print("Test_overlay_folder — 채널번호 vs 다른 텍스트 특징 (40 UI × 100장, 라벨 기준)")
print("="*92)
order = ["채널번호", "방송사명", "프로그램명", "시간대"]
feats = [("글자높이(h/H)", "h_norm", "%.3f"), ("폰트크기(px)", "fs", "%.0f"),
         ("가로세로비 w/h", "aspect", "%.2f"), ("가로폭(w/W)", "w_norm", "%.3f"),
         ("중심 X(좌0-우1)", "cx", "%.2f"), ("중심 Y(상0-하1)", "cy", "%.2f"),
         ("글자수", "nchar", "%.1f"), ("숫자비율", "digit_ratio", "%.2f"),
         ("순수숫자 비율", "pure_digit", "%.2f"), ("색 밝기(0-255)", "lum", "%.0f")]
hdr = f"{'특징':<16}" + "".join(f"{c:>14}" for c in order)
print(hdr); print("-"*92)
for name, key, fmt in feats:
    line = f"{name:<16}"
    for c in order:
        m, s = ms(feat[c].get(key, []))
        line += f"{(fmt%m):>14}"
    print(line)
print("-"*92)
line = f"{'값다양성(폴더내)':<16}"
for c in order:
    m, s = ms(div.get(c, []))
    line += f"{('%.2f'%m):>14}"
print(line)
print("="*92)
tot = rel_largest["채널번호"]
print(f"· 채널번호가 그 프레임에서 '가장 큰 텍스트'인 비율: {tot[0]}/{tot[1]} = {tot[0]/max(1,tot[1])*100:.1f}%")
print(f"· 채널높이 / 다른텍스트 중앙높이 배율(중앙값): {st.median(rel_ratio):.2f}x  (평균 {np.mean(rel_ratio):.2f}x)")
print("="*92)

# ── 시각화 ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
FP="/home1/irteam/teacher_model/assets/google_fonts/ofl/nanumgothic/NanumGothic-Bold.ttf"
fm.fontManager.addfont(FP); plt.rcParams["font.family"]=fm.FontProperties(fname=FP).get_name()
plt.rcParams["axes.unicode_minus"]=False
cols={"채널번호":"#D6392E","방송사명":"#2E5FA3","프로그램명":"#E1812C","시간대":"#2E9E5B"}
fig,axs=plt.subplots(2,2,figsize=(13,8)); fig.suptitle("채널번호 ROI vs 다른 텍스트 — 구별 특징 분포 (Test_overlay 40 UI)",fontsize=15,fontweight="bold")
panels=[("폰트 크기 (px)","fs",axs[0,0]),("가로세로비 (w/h)","aspect",axs[0,1]),
        ("순수숫자 비율","pure_digit",axs[1,0]),("폴더내 값다양성","__div",axs[1,1])]
for title,key,ax in panels:
    data=[]; labs=[]; cs=[]
    for c in order:
        if key=="__div": v=div.get(c,[])
        else: v=feat[c].get(key,[])
        v=[x for x in v if x is not None]
        if v: data.append(v); labs.append(c); cs.append(cols[c])
    bp=ax.boxplot(data,tick_labels=labs,patch_artist=True,showfliers=False,widths=.6)
    for p,c in zip(bp["boxes"],cs): p.set_facecolor(c); p.set_alpha(.65)
    for med in bp["medians"]: med.set_color("k")
    ax.set_title(title,fontsize=13,fontweight="bold"); ax.grid(axis="y",ls=":",alpha=.4)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig("/tmp/claude-500/-home1-irteam/53ded4c1-f246-489d-aea8-0f9713428fd0/scratchpad/channel_features.png",dpi=140,facecolor="white")
print("saved channel_features.png")
