#!/usr/bin/env python3
"""slot_v3 — 채널번호 선택 (사용자 설계 A/B/C 반영).

A. 숫자+텍스트 혼합 박스 분할: "000 MBC" → "000" (문자수 비율로 bbox를 잘라 숫자부만).
B. 강한 선택 기준(점수):
     - 높이(폰트 크기) 일관성 = 핵심 (채널번호 높이는 안 변함, 위치보다 신뢰)
     - 값 다양성 (프레임마다 값이 바뀜)
     - 2곳 일치 보너스 (같은 값이 한 프레임 두 위치에 → 채널 강한 신호)
     - 순수숫자 우대 / 한글오독·기호·시간·날짜 배제
     - 위치는 soft (광고/방송 때 UI가 이동하므로 하드락 안 함)
C. 롤링 FIFO 윈도우: 각 프레임을 '직전 N개' 누적으로만 판단(전체 배치 아님, TV CPU).
     안 묶이는(1회성) 후보는 버림. 이력 히스테리시스로 흔들림 억제.

핵심 통찰(사용자): 위치 고정이 아니라 '누적 클러스터 + 강한 판별'로 채널을 고른다.
"""
from __future__ import annotations

import re
import statistics as st
from collections import defaultdict

HANGUL = re.compile(r"[가-힣㄰-㆏ᄀ-ᇿ]")
DIGIT_RUN = re.compile(r"\d+")
TIME_RE = re.compile(r"\d{1,2}\s*[:：]\s*\d{2}")
DATE_RE = re.compile(r"\d{1,4}\s*[/.\-]\s*\d{1,2}(\s*[/.\-]\s*\d{1,4})?")
PREFIX_RE = re.compile(r"(?i)\b(ch|channel)\b")


def _cnorm(v):
    s = str(v)
    return str(int(s)) if s.isdigit() else s


def digit_runs(t):
    return [(m.group(), m.start(), m.end()) for m in DIGIT_RUN.finditer(str(t))]


def is_time_or_date(t):
    return bool(TIME_RE.search(str(t)) or DATE_RE.search(str(t)))


def split_candidate(c):
    """(A) 숫자+텍스트 혼합 박스를 숫자부만 잘라낸다. 시간/날짜면 버림. 반환 sub-candidate 리스트.

    en rec는 한글을 '영어/숫자 쓰레기'로 뱉으므로 한글 판별은 불가. 대신 '숫자 순도(purity)'
    = 숫자가 박스 영숫자 중 몇 %인가로 판별한다(채널=거의 숫자, 한글오독=긴 영문 속 숫자조각).
    """
    b = c.get("bbox_xyxy"); t = str(c.get("text", ""))
    if not b or len(b) != 4 or is_time_or_date(t):
        return []
    runs = digit_runs(t)
    if not runs:
        return []
    L = max(1, len(t))
    non_digit = re.sub(r"[\d\s]", "", t)               # 숫자·공백 제외 = 텍스트부 존재?
    alnum = re.sub(r"[^0-9A-Za-z]", "", t)             # 영숫자만 (순도 계산용)
    had_prefix = bool(PREFIX_RE.search(t))             # CH/Channel 접두사 (정당한 접두사)
    x1, y1, x2, y2 = b; W = x2 - x1
    out = []
    for d, s, e in runs:
        if not (1 <= len(d) <= 5):
            continue
        purity = 1.0 if had_prefix else len(d) / max(1, len(alnum))   # 숫자 순도
        if non_digit and not had_prefix:               # 혼합("000 MBC") → 문자비율로 숫자부만
            dx1 = x1 + W * (s / L); dx2 = x1 + W * (e / L)
        else:                                          # 순수숫자 / "CH123" → 원본 폭
            dx1, dx2 = x1, x2
        out.append({"bbox_xyxy": [dx1, y1, dx2, y2], "text": d,
                    "ocr_conf": float(c.get("ocr_conf", 0.5) or 0.5),
                    "had_text": bool(non_digit) and not had_prefix,
                    "had_prefix": had_prefix, "purity": purity})
    return out


def preprocess_frame(im, conf_thr=0.3):
    """한 프레임의 후보를 분할·정제해 채널 후보 리스트로. (cx,cy,h 정규화)"""
    W = float(im.get("image_width") or 1280) or 1280
    H = float(im.get("image_height") or 720) or 720
    out = []
    for c in im.get("candidates", []):
        for sc in split_candidate(c):
            if sc["ocr_conf"] < conf_thr:
                continue
            b = sc["bbox_xyxy"]
            out.append({"cx": (b[0] + b[2]) / 2 / W, "cy": (b[1] + b[3]) / 2 / H,
                        "h": (b[3] - b[1]) / H, "value": _cnorm(sc["text"]), "raw": sc["text"],
                        "conf": sc["ocr_conf"], "had_text": sc["had_text"],
                        "had_prefix": sc["had_prefix"], "purity": sc["purity"],
                        "box": [round(x, 1) for x in b]})
    return out


def within_frame_agreed(cands, min_sep=0.06):
    """한 프레임에서 같은 값이 멀리 떨어진 2곳↑에 → 그 값 집합(채널 강한 신호=2곳 일치)."""
    byv = defaultdict(list)
    for c in cands:
        byv[c["value"]].append((c["cx"], c["cy"]))
    agreed = set()
    for v, pts in byv.items():
        distinct = []
        for p in pts:
            if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_sep ** 2 for q in distinct):
                distinct.append(p)
        if len(distinct) >= 2:
            agreed.add(v)
    return agreed


def _assign(slots, c, by_height, band, size_lo, size_hi):
    """후보 c를 슬롯에 배정(높이 게이트 + 위치/평행이동축). 위치는 '최근' 기준이라 서서히 드리프트
    따라감. 없으면 새 슬롯 생성."""
    best, bd = None, 1e9
    for s in slots:
        r = c["h"] / s["mh"] if s["mh"] > 0 else 1.0
        if not (size_lo <= r <= size_hi):                       # 높이 게이트 (묶는 키)
            continue
        if by_height:
            dd = min(abs(c["cy"] - s["cy"]), abs(c["cx"] - s["cx"]))   # 같은 행/열(평행이동)
        else:
            dd = ((c["cx"] - s["cx"]) ** 2 + (c["cy"] - s["cy"]) ** 2) ** 0.5
        if dd > band:
            continue
        m = abs(r - 1.0) + dd
        if m < bd:
            bd, best = m, s
    if best is None:
        best = {"cx": c["cx"], "cy": c["cy"], "mh": c["h"], "last": -1, "recent": [],
                "present": set(), "vals": {}, "count": 0, "psum": 0.0, "asum": 0.0,
                "twoloc": 0, "conflict": 0}
        slots.append(best)
    return best


def _update(s, c, i, agreed, mem):
    """위치·높이는 '최근 mem'으로 드리프트 추적. 값다양성·2곳일치·순수도·커버리지는 'lifetime'
    누적(가끔 뜨는 위치도 놓치지 않게) — 선택 신호는 전체, 위치는 최근."""
    s["present"].add(i)                                        # lifetime 등장 프레임
    s["vals"][c["value"]] = s["vals"].get(c["value"], 0) + 1   # lifetime 값 다양성
    s["count"] += 1
    s["psum"] += c["purity"]
    s["asum"] += (c["box"][2] - c["box"][0]) / max(1e-6, (c["box"][3] - c["box"][1]))
    if c["value"] in agreed:
        s["twoloc"] += 1                                       # 다른 곳과 값 일치(진짜 채널 신호)
    elif agreed:
        s["conflict"] += 1                                    # 일치값이 있는데 여기 값은 다름(광고자리 등)
    s["last"] = i
    r = s["recent"]                                            # 위치/높이 드리프트용(최근만)
    r.append((i, c["cx"], c["cy"], c["h"]))
    while r and r[0][0] < i - mem + 1:
        r.pop(0)
    s["cx"] = sum(e[1] for e in r) / len(r)
    s["cy"] = sum(e[2] for e in r) / len(r)
    s["mh"] = st.median([e[3] for e in r])


def _score_persistent(s, i, use_div=True, use_cov=True, use_hcv=True, use_2loc=True):
    """(B) lifetime 통계로 선택 점수 — 가끔 뜨는 채널 위치도 살아남게.
    use_* = 각 핵심 항 on/off (ablation study용, 기본 전부 ON = 기존 동작)."""
    present = len(s["present"])
    if present == 0 or s["count"] == 0:
        return 0.0
    distinct = len(s["vals"])
    if use_div:
        base = (1.0 + 0.2 * distinct) if distinct >= 2 else 0.1  # 값 다양성 자격
    else:
        base = 1.0                                              # 값다양성 항 제거
    purity = s["psum"] / s["count"]
    hs = [e[3] for e in s["recent"]]
    h_cv = (st.pstdev(hs) / (sum(hs) / len(hs))) if len(hs) > 1 and sum(hs) > 0 else 0.0
    cov = (present / max(1, i + 1)) if use_cov else 1.0        # lifetime 커버리지(빈도수)
    score = cov * base * (0.9 + 0.1 * purity)                  # purity 거의 중립(방송사명 붙은 채널 안 깎음)
    if use_hcv and h_cv > 0.30:                                # 크기 변동 penalty
        score *= 0.4
    aspect = s["asum"] / s["count"]
    if not (0.2 <= aspect <= 8.0):
        score *= 0.2
    if use_2loc and s["twoloc"] > 0:                           # 2곳 일치 = 채널 강한 신호
        score *= (1.0 + 0.6 * min(s["twoloc"], 6))
    return score


def _agree(a, b, min_common=2, ratio=0.6):
    """두 슬롯이 같은 채널을 다른 위치에서 표시하는가(값이 공통 프레임에서 일치).

    배너가 가끔만 뜨는 위치(좌하단 등)도 놓치지 않도록 최소 공통 프레임은 2로 관대하게.
    대신 일치 '비율'은 높게(오탐 방지) — 공통 프레임에서 값이 거의 다 같아야 채널 짝으로 인정.
    """
    common = set(a.get("pf", {})) & set(b.get("pf", {}))
    if len(common) < min_common:
        return False
    match = sum(1 for f in common if a["pf"][f][0] == b["pf"][f][0])
    return match >= ratio * len(common)


def rolling_analyze(frames, ids, window=24, by_height=False, band=0.05,
                    size_lo=0.6, size_hi=1.7, min_present=2, conf_thr=0.3,
                    hysteresis=0.2):
    """(C) on-device: 슬롯별 '최근 window프레임 통계'만 유지(이미지 저장 X). 선택은 그 최근
    통계로(강함) + 위치 변하면 옛 슬롯 사라지고 적응. **채널이 2곳(좌하단+우상단 등)에 뜨면
    두 곳을 그룹으로 묶어 프레임마다 conf 높은 쪽으로 읽어 커버리지·정확도 회복(v1 second-channel).

    위치는 '묶는 키', 선택은 '내용'(값다양성·순수도·높이·2곳일치).
    """
    mem = max(2, window)
    # ── PASS 1: 스트리밍으로 채널 '위치(그룹)'를 확정 (최근 통계, 적응) ──
    slots = []
    pre_all = []
    for i in range(len(frames)):
        cands = preprocess_frame(frames[i], conf_thr)
        pre_all.append(cands)
        agreed = within_frame_agreed(cands)
        cur = {}
        for c in cands:
            s = _assign(slots, c, by_height, band, size_lo, size_hi)
            _update(s, c, i, agreed, mem)
            key = id(s)
            if key not in cur or c["conf"] > cur[key][1]:
                cur[key] = (c["value"], c["conf"], c["box"], s)
        for _k, (v, cf, bx, s) in cur.items():
            s.setdefault("pf", {})[i] = (v, cf, bx)             # pf는 lifetime(일치 판정용)
    elig = [s for s in slots if len(s["present"]) >= min_present]
    if not elig:
        return None
    n = len(frames)
    ranked = sorted(elig, key=lambda s: -_score_persistent(s, n - 1))
    top = ranked[0]
    group = [top] + [s for s in ranked[1:] if _agree(top, s)]    # 채널 위치 그룹(예: 좌하단+우상단)
    # 위치 가중치 = lifetime 채널점수(값다양성·커버리지·2곳일치) — 진짜 채널 위치(좌하단 배너)가 큼
    locs = [(s["cx"], s["cy"], s["mh"], _score_persistent(s, n - 1)) for s in group]
    group_boxes = []
    for s in group:
        bx = [p[2] for p in s["pf"].values()]
        if bx:
            group_boxes.append([st.median([b[k] for b in bx]) for k in range(4)])

    # ── PASS 2: 매 프레임 그룹 위치들에서 읽되, '같은 값이 2곳에 나오면(002=2) 그게 채널'
    #    (프레임내 일치 우선) → 광고 속 딴 숫자(881 등 한 곳뿐)는 배제. 일치 없으면 깨끗+conf. ──
    per_frame, per_conf, boxes = {}, {}, []
    for i, cands in enumerate(pre_all):
        reads = []                                             # (value, conf, purity, box, loc_idx, loc_score)
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
        # 값별 가중 투표: 각 값 점수 = Σ(위치점수 × conf). 같은 값이 여러 위치면 합산(일치 보너스),
        # 위치점수가 커서 '진짜 채널 위치(좌하단)'의 값이 distractor(우상단 단일숫자)를 이김.
        weight = defaultdict(float); best_read = {}
        for v, cf, pu, bx, li, lsc in reads:
            weight[v] += (lsc + 0.01) * cf
            if v not in best_read or cf > best_read[v][0]:
                best_read[v] = (cf, bx)
        v = max(weight, key=weight.get)
        cf, bx = best_read[v]
        per_frame[ids[i]] = v; per_conf[ids[i]] = cf; boxes.append(bx)
    if not boxes:
        return None
    box = [st.median([b[k] for b in boxes]) for k in range(4)]
    return {"box": box, "per_frame": per_frame, "per_frame_conf": per_conf,
            "locs": locs, "group_boxes": group_boxes,
            "score": round(len(per_frame) / max(1, len(ids)), 3),
            "distinct": len({_cnorm(v) for v in per_frame.values()}), "duals": []}
