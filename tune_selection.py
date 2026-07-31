#!/usr/bin/env python3
"""Offline tuning: try several channel-SELECTION strategies on a finished run's
full_ocr.json and report accuracy for each vs filename GT. No OCR re-run, no deploy.

Goal: find the rule that best picks the channel among many candidates (channel vs
program numbers) BEFORE touching the pipeline. The winner becomes v3's logic.

Strategies (per frame, pick one channel value):
  current            : the pipeline's own prediction (per_frame.csv) — baseline
  top_conf           : highest-conf channelnum candidate
  shortest           : among high-conf channelnum, prefer fewer digits (1-3)
  region_lock        : find the channel REGION (position cluster with the most
                       distinct values across frames = value-diverse) and read the
                       candidate nearest that region each frame
  region_lock_short  : region_lock but prefer shorter value in the region
  within_agree       : value appearing at 2+ distinct positions in the frame
  agree_or_region    : within_agree, else region_lock

Run: python tune_selection.py --result ./result_ui_slot_debug
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from predict_folder import gt_from_name
from temporal_profile_select import classify, best_digit


def _cn(v):
    s = str(v)
    return str(int(s)) if s.isdigit() else s


def chan_cands(im):
    """channelnum 후보: (value, cx, cy, conf, ndigits)."""
    W = float(im.get("image_width") or 1280) or 1280
    H = float(im.get("image_height") or 720) or 720
    out = []
    for c in im.get("candidates", []):
        b = c.get("bbox_xyxy"); t = c.get("text", "")
        if not b or len(b) != 4 or classify(t) != "channelnum":
            continue
        v = _cn(best_digit(t)); cf = float(c.get("ocr_conf", 0.5) or 0.5)
        if not v:
            continue
        out.append((v, (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H, cf, len(v)))
    return out


def find_region(frames_cands, min_sep=0.06):
    """값 다양성이 가장 큰 위치 군집 = 채널 영역 (프로그램은 값 고정이라 다양성 낮음)."""
    clusters = []                                       # [cx,cy,[pts], set(values)]
    for cands in frames_cands:
        for v, cx, cy, cf, nd in cands:
            r = next((r for r in clusters if (cx - r[0]) ** 2 + (cy - r[1]) ** 2 < min_sep ** 2), None)
            if r:
                r[2].append((cx, cy)); r[0] = st.fmean(p[0] for p in r[2]); r[1] = st.fmean(p[1] for p in r[2])
                r[3].add(v)
            else:
                clusters.append([cx, cy, [(cx, cy)], {v}])
    if not clusters:
        return None
    best = max(clusters, key=lambda r: (len(r[3]), len(r[2])))   # 값 다양성 우선
    return (best[0], best[1])


def near_region(cands, region, short=False, near=0.06):
    if region is None:
        return None
    pool = [c for c in cands if ((c[1] - region[0]) ** 2 + (c[2] - region[1]) ** 2) ** 0.5 <= near]
    if not pool:
        return None
    if short:
        pool.sort(key=lambda c: (c[4], -c[3]))          # 자릿수 적고 conf 높은 것
    else:
        pool.sort(key=lambda c: -c[3])                  # conf 높은 것
    return pool[0][0]


def within_agree(cands, min_sep=0.08):
    byval = defaultdict(list)
    for v, cx, cy, cf, nd in cands:
        byval[v].append((cx, cy, cf))
    best = None
    for v, pts in byval.items():
        distinct = []
        for p in pts:
            if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_sep ** 2 for q in distinct):
                distinct.append(p)
        if len(distinct) >= 2:
            sc = sum(p[2] for p in distinct)
            if best is None or sc > best[0]:
                best = (sc, v)
    return best[1] if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    args = ap.parse_args()
    res = Path(args.result)
    imgs = json.loads((res / "full_ocr.json").read_text()).get("images", [])

    # 현재 파이프라인 예측
    cur = {}
    pf = res / "per_frame.csv"
    if pf.exists():
        for r in csv.DictReader(pf.open(encoding="utf-8-sig")):
            cur[r.get("frame", "")] = r.get("channel_number", "")

    rows = []                                            # (frame, gt, cands)
    for im in imgs:
        frame = im.get("image_id", "").split("__")[-1]
        gt = gt_from_name(frame)
        if not gt:
            continue
        rows.append((frame, _cn(gt), chan_cands(im)))
    region = find_region([c for _f, _g, c in rows])
    print(f"프레임 {len(rows)}  채널영역(값다양성 최대) = "
          f"({region[0]:.2f},{region[1]:.2f})" if region else "영역 없음", flush=True)

    strategies = {
        "current": lambda fr, g, c: _cn("".join(ch for ch in str(cur.get(fr, "")) if ch.isdigit())) if cur.get(fr) else "",
        "top_conf": lambda fr, g, c: (max(c, key=lambda x: x[3])[0] if c else ""),
        "shortest": lambda fr, g, c: (sorted(c, key=lambda x: (x[4], -x[3]))[0][0] if c else ""),
        "region_lock": lambda fr, g, c: near_region(c, region) or "",
        "region_lock_short": lambda fr, g, c: near_region(c, region, short=True) or "",
        "within_agree": lambda fr, g, c: within_agree(c) or "",
        "agree_or_region": lambda fr, g, c: within_agree(c) or near_region(c, region, short=True) or "",
    }
    print(f"\n{'strategy':<20}{'정확도%':>9}{'맞음/전체':>14}  (읽은것중)")
    for name, fn in strategies.items():
        ok = read = 0
        for fr, g, c in rows:
            p = fn(fr, g, c)
            if p:
                read += 1
                if p == g:
                    ok += 1
        n = len(rows)
        wr = ok / read * 100 if read else 0
        print(f"{name:<20}{ok/n*100:>8.1f}%{f'{ok}/{n}':>14}  ({wr:.1f}%)")


if __name__ == "__main__":
    main()
