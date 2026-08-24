#!/usr/bin/env python3
"""slot_v8 ablation — 각 항 use_* 를 꺼서 중요도 측정."""
import argparse, csv, json, re
from collections import defaultdict
import slot_v8 as V8
TERMS=[("값다양성","use_div"),("빈도수","use_cov"),("절대크기","use_prom"),
       ("텍스트대비","use_con"),("크기불안정penalty","use_hcv")]
def gt_leading(s):
    tail=s.split("__")[-1]; m=re.match(r"0*(\d+)",tail); return m.group(1) if m else None
def gt_ch(s):
    m=re.match(r"(?i)ch0*(\d+)",s); return m.group(1) if m else None
def norm(s):
    s=re.sub(r"\D","",str(s)); return str(int(s)) if s else ""
ap=argparse.ArgumentParser()
ap.add_argument("--ocr",required=True); ap.add_argument("--group-sep",default="__")
ap.add_argument("--gt",default="leading"); ap.add_argument("--window",type=int,default=24)
ap.add_argument("--out",default="ablation_v8.csv")
a=ap.parse_args()
gt=gt_ch if a.gt=="ch" else gt_leading
d=json.load(open(a.ocr)); groups=defaultdict(list)
for im in d["images"]:
    g=im["image_id"].split(a.group_sep)[0] if a.group_sep else "ALL"; groups[g].append(im)
print(f"이미지 {len(d['images'])} · 그룹 {len(groups)}",flush=True)
CFG=[("v8  (전부)",{})]+[(f"v8 − {lab}",{key:False}) for lab,key in TERMS]
def run(flags):
    tot=[0,0]; per={}
    for g in sorted(groups):
        ims=sorted(groups[g],key=lambda x:x["image_id"]); ids=[x["image_id"] for x in ims]
        r=V8.rolling_analyze(ims,ids,window=a.window,**flags)
        pf=r["per_frame"] if r else {}; c=t=0
        for i in ids:
            gg=gt(i)
            if not gg: continue
            t+=1
            if norm(pf.get(i,""))==norm(gg): c+=1
        per[g]=(c,t); tot[0]+=c; tot[1]+=t
    return per,tot
print("="*56,flush=True); res={}
for name,fl in CFG:
    per,tot=run(fl); res[name]=(per,tot)
    print(f"  {name:<22} {tot[0]/max(1,tot[1])*100:5.1f}%  ({tot[0]}/{tot[1]})",flush=True)
print("="*56)
full=res["v8  (전부)"][1]; fa=full[0]/max(1,full[1])*100
print(f"v8 전부 = {fa:.1f}%\n중요도(빼면 하락, 클수록 중요):")
imp=[(lab, fa-res[f"v8 − {lab}"][1][0]/max(1,res[f"v8 − {lab}"][1][1])*100) for lab,_ in TERMS]
for rk,(lab,dr) in enumerate(sorted(imp,key=lambda x:-x[1]),1):
    print(f"  {rk}. {lab:<16} 빼면 {dr:+.1f}%p")
uis=sorted(res["v8  (전부)"][0])
with open(a.out,"w",newline="",encoding="utf-8-sig") as f:
    w=csv.writer(f); w.writerow(["group","n"]+[n for n,_ in CFG])
    for g in uis:
        n=res["v8  (전부)"][0][g][1]
        w.writerow([g,n]+[round(res[nm][0][g][0]/max(1,res[nm][0][g][1])*100,1) for nm,_ in CFG])
print(f"\n그룹별 표 → {a.out}")
