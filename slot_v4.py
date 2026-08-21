#!/usr/bin/env python3
"""slot_v4 — slot_v3 + '약한 배경색 힌트'.

v3(step1~3)는 그대로 두고, step3 점수에 **가벼운 배경(하이라이트) 보너스**를 얹는다.
아이디어(사용자): 채널 리스트에서 현재 채널 행은 배경이 하이라이트로 '튄다'.
  - 각 후보 박스 옆 배경색을 최근 bg_window(기본 5)프레임 누적(중앙값).
  - 형제 슬롯들의 배경과 비교해, 한 슬롯만 '명확히/일관되게' 다르면 그 슬롯에 약한 ×보너스.
  - 튀는 슬롯이 없으면(=배경 없는/단색 숫자 UI) 보너스 0 → 기존 v3와 동일(무해, self-gating).

배경색만 이미지에서 샘플하므로 비용 작음. 이미지 없으면 자동으로 v3와 동일.
"""
from __future__ import annotations

import statistics as st
from collections import defaultdict

import slot_v3 as V3

try:
    from PIL import Image
except Exception:
    Image = None

_IMG_CACHE = {}


def _load(path):
    if Image is None or not path:
        return None
    if path in _IMG_CACHE:
        return _IMG_CACHE[path]
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        im = None
    if len(_IMG_CACHE) > 8:
        _IMG_CACHE.clear()
    _IMG_CACHE[path] = im
    return im


def _sample_bg(img, box):
    """행 하이라이트 색 추정: 박스 '왼쪽' 하이라이트 바(텍스트 없는 여백)를 샘플.
    왼쪽이 화면밖이면 박스 내부(텍스트 제외 중앙값)로 폴백."""
    if img is None:
        return None
    W, H = img.size
    x1, y1, x2, y2 = [int(v) for v in box]
    h = max(1, y2 - y1)
    pw = max(3, h // 2)
    yy1, yy2 = max(0, y1 + h // 5), min(H, y2 - h // 5)
    lx1, lx2 = max(0, x1 - 2 - pw), max(1, x1 - 2)
    if lx2 - lx1 >= 3 and yy2 > yy1:                 # 왼쪽 하이라이트 바
        patch = img.crop((lx1, yy1, lx2, yy2))
    else:                                            # 폴백: 박스 내부(중앙값=배경 우세, 텍스트는 소수)
        patch = img.crop((x1, yy1, min(W, x2), yy2))
    px = list(patch.getdata())
    if not px:
        return None
    return (st.median([p[0] for p in px]), st.median([p[1] for p in px]), st.median([p[2] for p in px]))


def _cdist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _lum(p):
    return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]


def _sample_contrast(img, box):
    """박스 내부 텍스트-배경 대비(선명도) = 밝기 상위10% - 하위10%. 흰-on-흰이면 낮음."""
    if img is None:
        return None
    W, H = img.size
    x1, y1, x2, y2 = [int(v) for v in box]
    if x2 <= x1 or y2 <= y1:
        return None
    patch = img.crop((max(0, x1), max(0, y1), min(W, x2), min(H, y2)))
    lums = sorted(_lum(p) for p in patch.getdata())
    if len(lums) < 10:
        return None
    lo = lums[int(len(lums) * 0.1)]; hi = lums[int(len(lums) * 0.9)]
    return hi - lo


def _sample_sat(img, box):
    """글자(ink) 픽셀의 채도 = 컬러 강조도. 흰/회색 글자는 0에 가깝고, 컬러 강조 숫자는 높음.
    ink = 박스 내부에서 배경(중앙값)과 밝기차 큰 픽셀 → 그 픽셀들의 (max-min)/max 평균."""
    if img is None:
        return None
    W, H = img.size
    x1, y1, x2, y2 = [int(v) for v in box]
    if x2 <= x1 or y2 <= y1:
        return None
    px = list(img.crop((max(0, x1), max(0, y1), min(W, x2), min(H, y2))).getdata())
    if len(px) < 10:
        return None
    lums = [_lum(p) for p in px]
    med = sorted(lums)[len(lums) // 2]
    def sat(p):
        mx = max(p); mn = min(p)
        return (mx - mn) / (mx + 1e-6)
    ink = [sat(p) for p, l in zip(px, lums) if abs(l - med) > 40]   # 글자 픽셀
    if len(ink) < 5:
        ink = [sat(p) for p in px]                                  # 폴백: 전체
    return sum(ink) / len(ink)


def rolling_analyze(frames, ids, window=24, by_height=False, band=0.05,
                    size_lo=0.6, size_hi=1.7, min_present=2, conf_thr=0.3,
                    hysteresis=0.2, bg_window=5, bg_weight=0.3, bg_thresh=12.0,
                    contrast_weight=0.2, contrast_thresh=25.0,
                    size_weight=0.4, size_ratio=1.3, global_weight=0.5,
                    sat_weight=0.0, sat_thresh=0.18, sat_ratio=1.4):
    """v3와 동일하되, top1/top2가 근소할 때만(tie-break) 약한 시각 힌트 2종을 더한다:
       ① bg 하이라이트(현재 행이 배경색이 튀나)  ② 텍스트-배경 대비도(선명한가)."""
    mem = max(2, window)
    slots = []
    pre_all = []
    for i in range(len(frames)):
        img = _load(frames[i].get("image_path"))
        cands = V3.preprocess_frame(frames[i], conf_thr)
        for c in cands:
            c["bg"] = _sample_bg(img, c["box"])
            c["con"] = _sample_contrast(img, c["box"])
            c["sat"] = _sample_sat(img, c["box"]) if sat_weight else None
        pre_all.append(cands)
        agreed = V3.within_frame_agreed(cands)
        cur = {}
        for c in cands:
            s = V3._assign(slots, c, by_height, band, size_lo, size_hi)
            V3._update(s, c, i, agreed, mem)
            if c.get("bg") is not None:                       # 배경색 최근 bg_window 누적
                bg = s.setdefault("bgrecent", [])
                bg.append((i, c["bg"]))
                while bg and bg[0][0] < i - bg_window + 1:
                    bg.pop(0)
            if c.get("con") is not None:                      # 대비도 최근 bg_window 누적
                cr = s.setdefault("conrecent", [])
                cr.append((i, c["con"]))
                while cr and cr[0][0] < i - bg_window + 1:
                    cr.pop(0)
            if c.get("sat") is not None:                      # 채도(컬러 강조) 최근 누적
                sr = s.setdefault("satrecent", [])
                sr.append((i, c["sat"]))
                while sr and sr[0][0] < i - bg_window + 1:
                    sr.pop(0)
            k = id(s)
            if k not in cur or c["conf"] > cur[k][1]:
                cur[k] = (c["value"], c["conf"], c["box"], s)
        for _k, (v, cf, bx, s) in cur.items():
            s.setdefault("pf", {})[i] = (v, cf, bx)

    elig = [s for s in slots if len(s["present"]) >= min_present]
    if not elig:
        return None
    n = len(frames)

    # ── 약한 시각 tie-break: 강한 신호로 '근소한' 멀티자리 후보들 사이에서만 발동 ──
    def slot_bg(s):
        bg = s.get("bgrecent", [])
        if not bg:
            return None
        return (st.median([e[1][0] for e in bg]), st.median([e[1][1] for e in bg]),
                st.median([e[1][2] for e in bg]))

    def slot_con(s):
        cr = s.get("conrecent", [])
        return st.median([e[1] for e in cr]) if cr else None

    def slot_sat(s):
        sr = s.get("satrecent", [])
        return st.median([e[1] for e in sr]) if sr else None

    def rep_digits(s):
        if not s["vals"]:
            return 0
        v = max(s["vals"], key=lambda x: s["vals"][x])
        return len("".join(ch for ch in str(v) if ch.isdigit()))

    primary = {id(s): V3._score_persistent(s, n - 1) for s in elig}
    top_score = max(primary.values())
    # tie-break 대상 = 점수가 top과 근소(15%내)한 '멀티자리' 후보들 = 진짜 애매한 채널행들
    contenders = [s for s in elig if rep_digits(s) >= 2 and primary[id(s)] >= 0.85 * (top_score + 1e-9)]
    bonus = defaultdict(float)
    if len(contenders) >= 2:
        # ① 절대 크기: 한 후보만 명확히 크면(현재채널 강조) — 가장 깨끗한 신호
        hts = {id(s): s["mh"] for s in contenders if s["mh"] > 0}
        if len(hts) >= 2:
            oh = sorted(hts.values(), reverse=True)
            if oh[0] >= size_ratio * (oh[1] + 1e-9):
                bonus[max(hts, key=hts.get)] += size_weight
        # ② bg 하이라이트: 한 후보만 배경색이 명확히 튀면
        bgs = {id(s): slot_bg(s) for s in contenders if slot_bg(s) is not None}
        if len(bgs) >= 2:
            bl = list(bgs.values())
            ref = (st.median([b[0] for b in bl]), st.median([b[1] for b in bl]), st.median([b[2] for b in bl]))
            dd = {sid: _cdist(b, ref) for sid, b in bgs.items()}
            od = sorted(dd.values(), reverse=True)
            if od[0] >= bg_thresh and od[0] >= 1.5 * (od[1] + 1e-9):
                bonus[max(dd, key=dd.get)] += bg_weight
        # ③ 텍스트-배경 대비: 한 후보만 대비가 명확히 높으면(선명). 흰-on-흰은 낮아 자동 배제.
        cons = {id(s): slot_con(s) for s in contenders if slot_con(s) is not None}
        if len(cons) >= 2:
            oc = sorted(cons.values(), reverse=True)
            if oc[0] >= contrast_thresh and oc[0] >= 1.4 * (oc[1] + 1e-9):
                bonus[max(cons, key=cons.get)] += contrast_weight
        # ④ 채도(컬러 강조): 한 후보만 채도가 명확히 높으면(=컬러 강조 숫자). 흰/회색은 낮아 자동 배제.
        if sat_weight:
            sats = {id(s): slot_sat(s) for s in contenders if slot_sat(s) is not None}
            if len(sats) >= 2:
                os_ = sorted(sats.values(), reverse=True)
                if os_[0] >= sat_thresh and os_[0] >= sat_ratio * (os_[1] + 1e-9):
                    bonus[max(sats, key=sats.get)] += sat_weight

    # 전역 현저성(세 유형 공통, 형제 없어도 동작): 화면 전체에서 char 높이가 큰 멀티자리 슬롯 우대.
    # 앙상블의 한 축(약~중 가중치) — 채널번호가 작은 UI도 있으므로 최고 가중치 X, primary를 못 덮게.
    mhs = [s["mh"] for s in elig if s["mh"] > 0 and rep_digits(s) >= 2]
    gmax = max(mhs) if mhs else 0.0
    gmin = min(mhs) if mhs else 0.0

    def prom(s):
        if gmax <= gmin or rep_digits(s) < 2:
            return 0.0
        return (s["mh"] - gmin) / (gmax - gmin)

    def rank_score(s):
        return primary[id(s)] * (1.0 + bonus[id(s)] + global_weight * prom(s))

    ranked = sorted(elig, key=lambda s: -rank_score(s))
    top = ranked[0]
    group = [top] + [s for s in ranked[1:] if V3._agree(top, s)]
    locs = [(s["cx"], s["cy"], s["mh"], rank_score(s)) for s in group]
    group_boxes = []
    for s in group:
        bx = [p[2] for p in s["pf"].values()]
        if bx:
            group_boxes.append([st.median([b[k] for b in bx]) for k in range(4)])

    # ── PASS 2: v3와 동일 ──
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
                reads.append((c["value"], c["conf"], c["purity"], c["box"], li, lsc))
                break
        if not reads:
            continue
        weight = defaultdict(float); best_read = {}
        for v, cf, pu, bx, li, lsc in reads:
            weight[v] += (lsc + 0.01) * cf
            if v not in best_read or cf > best_read[v][0]:
                best_read[v] = (cf, bx)
        v = max(weight, key=weight.get)
        cf, bx = best_read[v]
        per_frame[ids[i]] = v; per_conf[ids[i]] = cf; boxes.append(bx); per_box[ids[i]] = bx
    if not boxes:
        return None
    box = [st.median([b[k] for b in boxes]) for k in range(4)]
    return {"box": box, "per_frame": per_frame, "per_frame_conf": per_conf,
            "per_frame_box": per_box,
            "locs": locs, "group_boxes": group_boxes,
            "score": round(len(per_frame) / max(1, len(ids)), 3),
            "distinct": len({V3._cnorm(v) for v in per_frame.values()}),
            "duals": [], "bg_bonus_used": bool(bonus)}
