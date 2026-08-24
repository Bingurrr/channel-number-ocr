#!/usr/bin/env python3
"""slot_v7 — 사용자 설계 곱셈식 점수.

Score = (1+빈도수)^a · (1/(ε+크기변동성))^b · (1+값다양성)^c · (1/(ε+위치변동성_높이))^d · (1+텍스트대비)^e

  · 빈도수(cov)          = 등장프레임/전체              (클수록 채널)
  · 크기변동성(h_cv)     = 최근 높이 변동계수            (작을수록 채널 → 1/x)
  · 값다양성(div)        = distinct값/누적수             (클수록 채널)
  · 위치변동성_높이(cy_std)= 최근 cy(위아래) 표준편차     (작을수록 채널 → 1/x)
        └ 사용자 정의: 정렬·자릿수에 따라 위아래로 흔들림 → cy(세로) 변동으로 계산
  · 텍스트대비(con)      = 박스 내부 밝기 p90-p10 정규화  (클수록 선명)

지수 a~e = '중요도 knob'. 값이 클수록 그 항이 점수를 더 크게 좌우함(=더 중요).
기본값은 사용자 중요도 순(빈도수>크기변동>값다양성>위치변동>대비), 대비만 눈에 띄게 낮게.
ablation: 어떤 항의 지수를 0으로 두면 그 factor=1 → '그 항을 뺀' 것.
"""
from __future__ import annotations
import statistics as st
from collections import defaultdict

import slot_v3 as V3
from slot_v4 import _load, _sample_contrast

# 기본 지수 = 중요도 순서. 대비(e)만 나머지보다 확실히 작게(조금 덜 중요).
DEFAULT_EXP = {"freq": 1.0, "size": 0.9, "div": 0.8, "pos": 0.7, "con": 0.4}


def _factors(s, n, eps):
    """슬롯의 5개 factor 값 (지수 적용 전)."""
    present = len(s["present"])
    if present == 0 or s["count"] == 0:
        return None
    cov = present / max(1, n)
    distinct = len(s["vals"]); div = distinct / max(1, s["count"])
    rec = s["recent"]; hs = [e[3] for e in rec]; cys = [e[2] for e in rec]
    h_cv = (st.pstdev(hs) / (sum(hs) / len(hs))) if len(hs) > 1 and sum(hs) > 0 else 0.0
    cy_std = st.pstdev(cys) if len(cys) > 1 else 0.0
    cr = s.get("conrecent", [])
    con = min(1.0, (st.median([e[1] for e in cr]) / 128.0)) if cr else 0.0
    return {"freq": 1.0 + cov, "size": 1.0 / (eps + h_cv), "div": 1.0 + div,
            "pos": 1.0 / (eps + cy_std), "con": 1.0 + con}


def _score(s, n, exps, eps):
    f = _factors(s, n, eps)
    if f is None:
        return 0.0
    aspect = s["asum"] / s["count"]
    score = 0.2 if not (0.2 <= aspect <= 8.0) else 1.0        # 채널다움 게이트(v3와 동일)
    for k, fv in f.items():
        e = exps.get(k, 0.0)
        if e:
            score *= fv ** e
    return score


def rolling_analyze(frames, ids, window=24, by_height=False, band=0.05,
                    size_lo=0.6, size_hi=1.7, min_present=2, conf_thr=0.3,
                    exps=None, eps=0.08, con_window=5):
    exps = {**DEFAULT_EXP, **(exps or {})}
    need_con = bool(exps.get("con", 0))
    mem = max(2, window)
    slots = []; pre_all = []
    for i in range(len(frames)):
        img = _load(frames[i].get("image_path")) if need_con else None
        cands = V3.preprocess_frame(frames[i], conf_thr)
        for c in cands:
            c["con"] = _sample_contrast(img, c["box"]) if need_con else None
        pre_all.append(cands)
        agreed = V3.within_frame_agreed(cands)
        cur = {}
        for c in cands:
            s = V3._assign(slots, c, by_height, band, size_lo, size_hi)
            V3._update(s, c, i, agreed, mem)
            if c.get("con") is not None:
                cr = s.setdefault("conrecent", []); cr.append((i, c["con"]))
                while cr and cr[0][0] < i - con_window + 1:
                    cr.pop(0)
            k = id(s)
            if k not in cur or c["conf"] > cur[k][1]:
                cur[k] = (c["value"], c["conf"], c["box"], s)
        for _k, (v, cf, bx, s) in cur.items():
            s.setdefault("pf", {})[i] = (v, cf, bx)

    elig = [s for s in slots if len(s["present"]) >= min_present]
    if not elig:
        return None
    n = len(frames)
    score_of = {id(s): _score(s, n, exps, eps) for s in elig}
    ranked = sorted(elig, key=lambda s: -score_of[id(s)])
    top = ranked[0]
    group = [top] + [s for s in ranked[1:] if V3._agree(top, s)]
    locs = [(s["cx"], s["cy"], s["mh"], score_of[id(s)]) for s in group]
    group_boxes = []
    for s in group:
        bx = [p[2] for p in s["pf"].values()]
        if bx:
            group_boxes.append([st.median([b[k] for b in bx]) for k in range(4)])

    # PASS 2 (v3와 동일: 프레임별 위치매칭 → 값 가중투표)
    per_frame, per_conf, boxes, per_box = {}, {}, [], {}
    for i, cands in enumerate(pre_all):
        reads = []
        for c in cands:
            for li, (lx, ly, lh, lsc) in enumerate(locs):
                if lh > 0 and not (size_lo <= c["h"] / lh <= size_hi):
                    continue
                if by_height:
                    if min(abs(c["cy"] - ly), abs(c["cx"] - lx)) > band:
                        continue
                elif ((c["cx"] - lx) ** 2 + (c["cy"] - ly) ** 2) ** 0.5 > band:
                    continue
                reads.append((c["value"], c["conf"], c["box"], li, lsc)); break
        if not reads:
            continue
        weight = defaultdict(float); best = {}
        for v, cf, bx, li, lsc in reads:
            weight[v] += (lsc + 0.01) * cf
            if v not in best or cf > best[v][0]:
                best[v] = (cf, bx)
        v = max(weight, key=weight.get); cf, bx = best[v]
        per_frame[ids[i]] = v; per_conf[ids[i]] = cf; boxes.append(bx); per_box[ids[i]] = bx
    if not boxes:
        return None
    box = [st.median([b[k] for b in boxes]) for k in range(4)]
    return {"box": box, "per_frame": per_frame, "per_frame_conf": per_conf,
            "per_frame_box": per_box, "locs": locs, "group_boxes": group_boxes,
            "score": round(len(per_frame) / max(1, len(ids)), 3),
            "distinct": len({V3._cnorm(v) for v in per_frame.values()})}
