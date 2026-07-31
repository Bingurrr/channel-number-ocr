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


def all_num_cands(im):
    """모든 후보에서 숫자만 남김: (value, cx, cy, conf). 'YTN'같은 글자 제거."""
    W = float(im.get("image_width") or 1280) or 1280
    H = float(im.get("image_height") or 720) or 720
    out = []
    for c in im.get("candidates", []):
        b = c.get("bbox_xyxy"); t = c.get("text", "")
        if not b or len(b) != 4:
            continue
        d = re.sub(r"\D", "", str(t))
        if not d or len(d) > 5:
            continue
        out.append((_cn(d), (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H, float(c.get("ocr_conf", 0.5) or 0.5)))
    return out


def learn_second_region(rows_num, primary_of, fbox, min_sep=0.10, min_votes=5):
    """v1이 확신한 프레임들에서 'primary값과 같은 숫자가 뜨는 다른 위치'를 누적 학습.

    반환: (region_cx, region_cy, votes) 또는 None. 프로그램 숫자는 우연히만 일치→
    투표 적음→탈락. 진짜 2번째 채널영역은 매 프레임 primary와 같이 바뀜→투표 많음.
    """
    clusters = []                                        # [cx, cy, [pts], votes]
    for fr, nums in rows_num.items():
        pv = primary_of.get(fr)
        if not pv:
            continue
        for v, cx, cy, cf in nums:
            if v != pv:
                continue
            if ((cx - fbox[0]) ** 2 + (cy - fbox[1]) ** 2) ** 0.5 <= min_sep:
                continue                                 # primary 위치 자신은 제외
            r = next((r for r in clusters if (cx - r[0]) ** 2 + (cy - r[1]) ** 2 < min_sep ** 2), None)
            if r:
                r[2].append((cx, cy)); r[0] = st.fmean(p[0] for p in r[2]); r[1] = st.fmean(p[1] for p in r[2]); r[3] += 1
            else:
                clusters.append([cx, cy, [(cx, cy)], 1])
    if not clusters:
        return None
    best = max(clusters, key=lambda r: r[3])
    return (best[0], best[1], best[3]) if best[3] >= min_votes else None


def read_at(nums, region, near=0.08):
    """region 근처 최고conf 숫자후보 (value, conf)."""
    if region is None:
        return (None, 0.0)
    pool = [(v, cf) for v, cx, cy, cf in nums if ((cx - region[0]) ** 2 + (cy - region[1]) ** 2) ** 0.5 <= near]
    if not pool:
        return (None, 0.0)
    pool.sort(key=lambda x: -x[1])
    return pool[0]


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

    # v1이 찾은 채널 field box (profile_report.json) → 정규화 중심 (좌하단)
    fbox = None
    pr = res / "profile_report.json"
    if pr.exists():
        for r in json.loads(pr.read_text()):
            b = r.get("channel_field_box")
            if b:
                W = 1280.0; H = 720.0
                fbox = ((b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H); break

    rows = []                                            # (frame, gt, cands)
    rows_num = {}                                         # frame -> all_num_cands
    for im in imgs:
        frame = im.get("image_id", "").split("__")[-1]
        gt = gt_from_name(frame)
        if not gt:
            continue
        rows.append((frame, _cn(gt), chan_cands(im)))
        rows_num[frame] = all_num_cands(im)
    region = find_region([c for _f, _g, c in rows])
    print(f"프레임 {len(rows)}  값다양성영역={region and (round(region[0],2),round(region[1],2))}  "
          f"v1_field={fbox and (round(fbox[0],2),round(fbox[1],2))}", flush=True)

    def curval(fr):
        s = "".join(ch for ch in str(cur.get(fr, "")) if ch.isdigit())
        return _cn(s) if s else ""

    # ★ 2번째 채널영역 학습 (사용자 아이디어): v1확신 프레임에서 primary값과 같은
    # 숫자가 뜨는 다른 위치를 누적 → 확정 → primary 실패 시 거기서 읽어 보완.
    primary_of = {fr: curval(fr) for fr, _g, _c in rows}
    second = learn_second_region(rows_num, primary_of, fbox) if fbox else None
    print(f"2번째채널영역 = {second and (round(second[0],2),round(second[1],2),f'votes={second[2]}')}", flush=True)

    def _primary_conf(fr, c):
        pv = curval(fr)
        if not pv:
            return 0.0
        pool = [x for x in c if x[0] == pv]              # x=(val,cx,cy,conf,nd)
        if not pool:
            return 0.4
        if fbox:
            pool.sort(key=lambda x: (x[1] - fbox[0]) ** 2 + (x[2] - fbox[1]) ** 2)
        return pool[0][3]

    def _pick_conf(fr, c):
        pv, pc = curval(fr), _primary_conf(fr, c)
        sv, sc = read_at(rows_num.get(fr, []), second)
        opts = ([(pc, pv)] if pv else []) + ([(sc, sv)] if sv else [])
        return (max(opts)[1] if opts else "")

    def _pick_agree(fr, c):
        pv = curval(fr); sv, sc = read_at(rows_num.get(fr, []), second)
        if pv and sv and pv == sv:
            return pv                                    # 두 위치 일치 → 확정
        if not pv:
            return sv or ""                              # primary 없음 → 2번째
        if not sv:
            return pv
        return pv if _primary_conf(fr, c) >= sc else sv  # 불일치 → conf 높은쪽

    strategies = {
        "current": lambda fr, g, c: curval(fr),
        "top_conf": lambda fr, g, c: (max(c, key=lambda x: x[3])[0] if c else ""),
        "shortest": lambda fr, g, c: (sorted(c, key=lambda x: (x[4], -x[3]))[0][0] if c else ""),
        "region_lock": lambda fr, g, c: near_region(c, region) or "",
        "region_lock_short": lambda fr, g, c: near_region(c, region, short=True) or "",
        "within_agree": lambda fr, g, c: within_agree(c) or "",
        # v1 field box(좌하단) 기반 — 값다양성영역보다 신뢰
        "field_lock": lambda fr, g, c: near_region(c, fbox, near=0.08) or "",
        "field_short": lambda fr, g, c: near_region(c, fbox, short=True, near=0.08) or "",
        "field_wide": lambda fr, g, c: near_region(c, fbox, near=0.14) or "",
        "field_short_wide": lambda fr, g, c: near_region(c, fbox, short=True, near=0.14) or "",
        "agree_or_field": lambda fr, g, c: within_agree(c) or near_region(c, fbox, short=True, near=0.12) or "",
        "field_or_top": lambda fr, g, c: near_region(c, fbox, short=True, near=0.10) or (max(c, key=lambda x: x[3])[0] if c else ""),
        # ★ 결합: 정확한 field_lock 우선 + 없으면 current(커버리지)
        "field_or_current": lambda fr, g, c: near_region(c, fbox, near=0.08) or curval(fr),
        "field10_or_current": lambda fr, g, c: near_region(c, fbox, near=0.10) or curval(fr),
        "field_or_agree_or_current": lambda fr, g, c: near_region(c, fbox, near=0.08) or within_agree(c) or curval(fr),
        # ★ 학습된 2번째영역 활용 (사용자 아이디어)
        # fill: primary 있으면 그것, 없으면(none) 2번째영역에서 읽어 보완
        "second_fill": lambda fr, g, c: curval(fr) or (read_at(rows_num.get(fr, []), second)[0] or ""),
        # conf: primary와 2번째영역 중 conf 높은쪽 (오독 교정 시도)
        "second_conf": lambda fr, g, c: _pick_conf(fr, c),
        # agree: 둘이 같으면 확정, 다르면 conf, primary 없으면 2번째
        "second_agree": lambda fr, g, c: _pick_agree(fr, c),
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
