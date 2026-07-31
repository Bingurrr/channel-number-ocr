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
        best = {"cx": c["cx"], "cy": c["cy"], "mh": c["h"], "last": -1, "recent": []}
        slots.append(best)
    return best


def _update(s, c, i, agreed, mem):
    """슬롯의 '최근 mem프레임' 관측만 유지(오래된 건 버림) → 위치 변화에 적응 + 신호 유지."""
    aspect = (c["box"][2] - c["box"][0]) / max(1e-6, (c["box"][3] - c["box"][1]))
    s["recent"].append((i, c["value"], c["purity"], c["h"], c["cx"], c["cy"], aspect,
                        1 if c["value"] in agreed else 0))
    lo = i - mem + 1
    while s["recent"] and s["recent"][0][0] < lo:               # mem프레임보다 오래된 관측 폐기
        s["recent"].pop(0)
    r = s["recent"]; s["last"] = i
    s["cx"] = sum(e[4] for e in r) / len(r)                     # 위치=최근 평균(드리프트 추적)
    s["cy"] = sum(e[5] for e in r) / len(r)
    s["mh"] = st.median([e[3] for e in r])


def _score_persistent(s, i, mem):
    """(B) '최근 mem프레임' 통계로 선택 점수. 전체 history 아님 → 위치 바뀌면 옛 슬롯 점수 급락."""
    r = s["recent"]
    if not r:
        return 0.0
    span = min(mem, i + 1)
    present = len({e[0] for e in r})                            # 최근 등장 프레임 수
    distinct = len({e[1] for e in r})
    base = (1.0 + 0.2 * distinct) if distinct >= 2 else 0.1     # 값 다양성 자격
    purity = sum(e[2] for e in r) / len(r)                      # 순수 숫자일수록 ↑(한글오독 ↓)
    hs = [e[3] for e in r]
    h_cv = (st.pstdev(hs) / (sum(hs) / len(hs))) if len(hs) > 1 and sum(hs) > 0 else 0.0
    cov = present / max(1, span)                                # 최근 커버리지(끊기면 급락→적응)
    score = cov * base * (0.55 + 0.45 * purity)                 # 순수도 감점 완화("000 YTN"도 채널)
    if h_cv > 0.30:
        score *= 0.4                                            # 높이 일관성 필수
    aspect = sum(e[6] for e in r) / len(r)
    if not (0.2 <= aspect <= 8.0):
        score *= 0.2
    twoloc = sum(e[7] for e in r)
    if twoloc > 0:                                             # 2곳 일치 = 채널 강한 신호
        score *= (1.0 + 0.6 * min(twoloc, 6))
    return score


def _agree(a, b, min_common=3, ratio=0.6):
    """두 슬롯이 같은 채널을 다른 위치에서 표시하는가(값이 공통 프레임에서 일치)."""
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
    slots = []
    per_frame, per_conf, boxes = {}, {}, []
    prev = None
    for i in range(len(frames)):
        cands = preprocess_frame(frames[i], conf_thr)
        agreed = within_frame_agreed(cands)
        cur = {}                                                # id(slot) -> (value, conf, box)
        for c in cands:
            s = _assign(slots, c, by_height, band, size_lo, size_hi)
            _update(s, c, i, agreed, mem)
            key = id(s)
            if key not in cur or c["conf"] > cur[key][1]:
                cur[key] = (c["value"], c["conf"], c["box"])
        for key, (v, cf, bx) in cur.items():                   # 슬롯별 프레임값 기록(교차채움/일치용)
            for s in slots:
                if id(s) == key:
                    s.setdefault("pf", {})[i] = (v, cf, bx)
                    for f in [f for f in s["pf"] if f < i - mem + 1]:
                        del s["pf"][f]
                    break
        slots = [s for s in slots if i - s["last"] < mem]       # 오래 안 보인 슬롯 제거(적응)
        elig = [s for s in slots
                if not (min(mem, i + 1) >= min_present and len({e[0] for e in s["recent"]}) < min_present)]
        if not elig:
            continue
        ranked = sorted(elig, key=lambda s: -(_score_persistent(s, i, mem)
                        * (1.0 + hysteresis if prev is not None and abs(s["cy"] - prev[1]) < band
                           and abs(s["cx"] - prev[0]) < max(band, 0.15) else 1.0)))
        top = ranked[0]
        prev = (top["cx"], top["cy"], top["mh"])
        group = [top] + [s for s in ranked[1:] if _agree(top, s)]   # 채널 2곳 그룹
        picks = [cur[id(s)] for s in group if id(s) in cur]         # 이 프레임에 잡힌 그룹 값들
        if picks:
            v, conf, box = max(picks, key=lambda x: x[1])           # conf 높은 위치로 읽음
            per_frame[ids[i]] = v; per_conf[ids[i]] = conf; boxes.append(box)
    if not boxes:
        return None
    box = [st.median([b[k] for b in boxes]) for k in range(4)]
    return {"box": box, "per_frame": per_frame, "per_frame_conf": per_conf,
            "score": round(len(per_frame) / max(1, len(ids)), 3),
            "distinct": len({_cnorm(v) for v in per_frame.values()}), "duals": []}
