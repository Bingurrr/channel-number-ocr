#!/usr/bin/env python3
"""slot_v8 — 실측 ablation을 반영한 '잘 되는' 곱셈식 점수.

v7 교훈:
  · 1/변동성(안정성)은 독 — 고정 라벨이 채널보다 안정적이라 라벨을 고름.
  · 진짜 판별 = 값다양성(강하게) + 절대 크기(채널은 큼, 라벨은 작음).

Score = 값다양성_강함 · (0.9+0.1·순도) · (1+w_cov·빈도수) · (1+w_prom·절대크기) · (1+w_con·대비) · 게이트

  · 값다양성_강함  : distinct<2 면 div_kill(0.15)로 죽임, 아니면 (1+0.5·distinct)   ← 실측 1등
  · 빈도수(cov)    : 등장프레임/전체
  · 절대크기(prom) : 화면 전체 멀티자리 中 char 높이 순위 [0,1] (전역현저성)          ← +5%p 검증
  · 대비(con)      : 박스 밝기 p90-p10 정규화 (조금 덜 중요, w_con=0.4)
  · 게이트         : aspect 이상 ×0.2, 높이 심하게 흔들리면(h_cv>0.30) ×0.5 (페널티만, 보상 X)
지수(로그) 대신 '가중치 w_*'로 중요도 조절: 값다양성 > 절대크기 > 빈도수 > 대비.
ablation: use_* 플래그로 각 항 on/off.
"""
from __future__ import annotations
import statistics as st
from collections import defaultdict

import slot_v3 as V3
from slot_v4 import _load, _sample_contrast


def _rep_digits(s):
    if not s["vals"]:
        return 0
    v = max(s["vals"], key=lambda x: s["vals"][x])
    return len("".join(ch for ch in str(v) if ch.isdigit()))


def rolling_analyze(frames, ids, window=24, by_height=False, band=0.05,
                    size_lo=0.6, size_hi=1.7, min_present=2, conf_thr=0.3,
                    w_cov=1.0, w_prom=0.8, w_con=0.4, div_kill=0.15, con_window=5,
                    use_div=True, use_cov=True, use_prom=True, use_con=True, use_hcv=True):
    need_con = use_con and bool(w_con)
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
    mhs = [s["mh"] for s in elig if s["mh"] > 0 and _rep_digits(s) >= 2]
    gmax = max(mhs) if mhs else 0.0; gmin = min(mhs) if mhs else 0.0

    def prom(s):
        if gmax <= gmin or _rep_digits(s) < 2:
            return 0.0
        return (s["mh"] - gmin) / (gmax - gmin)

    def score(s):
        present = len(s["present"])
        if present == 0 or s["count"] == 0:
            return 0.0
        distinct = len(s["vals"])
        divf = (1.0 + 0.5 * distinct) if distinct >= 2 else div_kill      # 값다양성(강함)
        if not use_div:
            divf = 1.0
        purity = s["psum"] / s["count"]
        cov = present / max(1, n)
        hs = [e[3] for e in s["recent"]]
        h_cv = (st.pstdev(hs) / (sum(hs) / len(hs))) if len(hs) > 1 and sum(hs) > 0 else 0.0
        cr = s.get("conrecent", [])
        con = min(1.0, st.median([e[1] for e in cr]) / 128.0) if cr else 0.0
        sc = divf * (0.9 + 0.1 * purity)
        if use_cov:
            sc *= (1.0 + w_cov * cov)
        if use_prom:
            sc *= (1.0 + w_prom * prom(s))
        if use_con:
            sc *= (1.0 + w_con * con)
        aspect = s["asum"] / s["count"]
        if not (0.2 <= aspect <= 8.0):
            sc *= 0.2
        if use_hcv and h_cv > 0.30:                                       # 심한 불안정만 페널티
            sc *= 0.5
        return sc

    sco = {id(s): score(s) for s in elig}
    ranked = sorted(elig, key=lambda s: -sco[id(s)])
    top = ranked[0]
    group = [top] + [s for s in ranked[1:] if V3._agree(top, s)]
    locs = [(s["cx"], s["cy"], s["mh"], sco[id(s)]) for s in group]
    group_boxes = []
    for s in group:
        bx = [p[2] for p in s["pf"].values()]
        if bx:
            group_boxes.append([st.median([b[k] for b in bx]) for k in range(4)])

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
