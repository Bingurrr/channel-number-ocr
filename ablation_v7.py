#!/usr/bin/env python3
"""slot_v7(곱셈식) ablation — 각 factor를 하나씩 빼서(지수=0) 중요도 측정.

Score = (1+빈도수)^a·(1/(ε+크기변동))^b·(1+값다양성)^c·(1/(ε+위치변동_높이))^d·(1+대비)^e

사용: python ablation_v7.py --ocr <full_ocr.json> --group-sep __ --gt leading [--out v7.csv]
"""
from __future__ import annotations
import argparse, csv, json, re
from collections import defaultdict
import slot_v7 as V7

TERMS = [("빈도수", "freq"), ("크기변동성", "size"), ("값다양성", "div"),
         ("위치변동성(높이)", "pos"), ("텍스트대비", "con")]


def make_gt(mode):
    def f(stem):
        tail = stem.split("__")[-1] if mode == "leading" else stem
        m = (re.match(r"(?i)ch0*(\d+)", tail) if mode == "ch" else None) or re.match(r"0*(\d+)", tail)
        return m.group(1) if m else None
    return f


def norm(s):
    s = re.sub(r"\D", "", str(s))
    return str(int(s)) if s else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", required=True)
    ap.add_argument("--group-sep", default="__")
    ap.add_argument("--gt", default="leading", choices=["leading", "ch"])
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--out", default="ablation_v7.csv")
    args = ap.parse_args()

    gt_of = make_gt(args.gt)
    d = json.load(open(args.ocr))
    groups = defaultdict(list)
    for im in d["images"]:
        g = im["image_id"].split(args.group_sep)[0] if args.group_sep else "ALL"
        groups[g].append(im)
    print(f"이미지 {len(d['images'])} · 그룹 {len(groups)}", flush=True)

    # 설정: full + 각 항 빼기(지수 0)
    base = dict(V7.DEFAULT_EXP)
    CFG = [("v7  (전부)", dict(base))]
    for label, key in TERMS:
        CFG.append((f"v7 − {label}", {**base, key: 0.0}))

    def run(exps):
        per, tot = {}, [0, 0]
        for g in sorted(groups):
            ims = sorted(groups[g], key=lambda x: x["image_id"]); ids = [x["image_id"] for x in ims]
            r = V7.rolling_analyze(ims, ids, window=args.window, exps=exps)
            pf = r["per_frame"] if r else {}
            c = t = 0
            for i in ids:
                gt = gt_of(i)
                if not gt:
                    continue
                t += 1
                if norm(pf.get(i, "")) == norm(gt):
                    c += 1
            per[g] = (c, t); tot[0] += c; tot[1] += t
        return per, tot

    print("=" * 60, flush=True)
    res = {}
    for name, e in CFG:
        per, tot = run(e); res[name] = (per, tot)
        print(f"  {name:<22} {tot[0]/max(1,tot[1])*100:5.1f}%  ({tot[0]}/{tot[1]})", flush=True)
    print("=" * 60)
    acc = lambda t: t[0]/max(1, t[1])*100
    full = acc(res["v7  (전부)"][1])
    print(f"v7 전부 = {full:.1f}%\n중요도 = 그 항을 빼면 성능이 얼마나 떨어지나(클수록 중요):")
    imp = []
    for label, key in TERMS:
        drop = full - acc(res[f"v7 − {label}"][1])
        imp.append((label, drop))
    for rank, (label, drop) in enumerate(sorted(imp, key=lambda x: -x[1]), 1):
        print(f"  {rank}. {label:<16} 빼면 {drop:+.1f}%p")

    uis = sorted(res["v7  (전부)"][0])
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["group", "n"] + [n for n, _ in CFG])
        for g in uis:
            n = res["v7  (전부)"][0][g][1]
            w.writerow([g, n] + [round(res[nm][0][g][0]/max(1, res[nm][0][g][1])*100, 1) for nm, _ in CFG])
    print(f"\n그룹별 표 → {args.out}")


if __name__ == "__main__":
    main()
