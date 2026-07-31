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
    """(A) 숫자+텍스트 혼합 박스를 숫자부만 잘라낸다. 시간/날짜면 버림. 반환 sub-candidate 리스트."""
    b = c.get("bbox_xyxy"); t = str(c.get("text", ""))
    if not b or len(b) != 4 or is_time_or_date(t):
        return []
    runs = digit_runs(t)
    if not runs:
        return []
    L = max(1, len(t))
    non_digit = re.sub(r"[\d\s]", "", t)               # 숫자·공백 제외 = 텍스트부 존재?
    had_prefix = bool(PREFIX_RE.search(t))             # CH/Channel 접두사
    had_hangul = bool(HANGUL.search(t))
    x1, y1, x2, y2 = b; W = x2 - x1
    out = []
    for d, s, e in runs:
        if not (1 <= len(d) <= 5):
            continue
        if non_digit and not had_prefix:               # 혼합("000 MBC") → 문자비율로 숫자부만
            dx1 = x1 + W * (s / L); dx2 = x1 + W * (e / L)
        else:                                          # 순수숫자 / "CH123" → 원본 폭
            dx1, dx2 = x1, x2
        out.append({"bbox_xyxy": [dx1, y1, dx2, y2], "text": d,
                    "ocr_conf": float(c.get("ocr_conf", 0.5) or 0.5),
                    "had_text": bool(non_digit) and not had_prefix,
                    "had_prefix": had_prefix, "had_hangul": had_hangul})
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
                        "had_prefix": sc["had_prefix"], "had_hangul": sc["had_hangul"],
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


def _cluster(win, by_height, band, size_lo, size_hi):
    """윈도우(프레임별 후보 리스트)를 높이+정렬로 묶는다. by_height면 위치 대신 높이+평행이동축."""
    slots = []
    for li, cands in enumerate(win):
        for c in cands:
            best, bd = None, 1e9
            for s in slots:
                r = c["h"] / s["mh"] if s["mh"] > 0 else 1.0
                if not (size_lo <= r <= size_hi):        # 높이 게이트 (항상)
                    continue
                if by_height:
                    dd = min(abs(c["cy"] - s["cy"]), abs(c["cx"] - s["cx"]))  # 같은 행/열(평행이동)
                else:
                    dd = ((c["cx"] - s["cx"]) ** 2 + (c["cy"] - s["cy"]) ** 2) ** 0.5
                if dd > band:
                    continue
                m = abs(r - 1.0) + dd
                if m < bd:
                    bd, best = m, s
            if best is None:
                best = {"cx": c["cx"], "cy": c["cy"], "mh": c["h"], "items": [], "frames": set()}
                slots.append(best)
            best["items"].append((li, c)); best["frames"].add(li)
            k = len(best["items"])
            best["cx"] = (best["cx"] * (k - 1) + c["cx"]) / k
            best["cy"] = (best["cy"] * (k - 1) + c["cy"]) / k
            best["mh"] = st.median([it[1]["h"] for it in best["items"]])
    return slots


def _score(s, nwin, agreed_by_li):
    """(B) 강한 선택 점수. 높이일관성×값다양성×깨끗한숫자×(시간/날짜아님)×2곳일치보너스."""
    items = [c for _, c in s["items"]]
    present = len(s["frames"])
    distinct = len({c["value"] for c in items})
    base = (1.0 + 0.2 * distinct) if distinct >= 2 else 0.1      # 값 다양성 자격
    clean = sum(1.0 if not c["had_text"] else 0.5 for c in items) / len(items)
    hangul = sum(0.25 if c["had_hangul"] else 1.0 for c in items) / len(items)  # 한글오독 감점
    prefix = 1.0 + 0.3 * (sum(c["had_prefix"] for c in items) / len(items))     # CH/Channel 가점
    hs = [c["h"] for c in items]
    h_cv = (st.pstdev(hs) / (sum(hs) / len(hs))) if len(hs) > 1 and sum(hs) > 0 else 0.0
    score = (present / max(1, nwin)) * base * clean * hangul * prefix
    if h_cv > 0.30:
        score *= 0.4                                            # 높이 일관성 필수
    # 종횡비/면적 게이트
    ws = [c["box"][2] - c["box"][0] for c in items]; hpx = [c["box"][3] - c["box"][1] for c in items]
    aspect = (sum(ws) / len(ws)) / max(1.0, sum(hpx) / len(hpx))
    if not (0.2 <= aspect <= 8.0):
        score *= 0.2
    # 2곳 일치 보너스 (강력): 프레임 내 같은 값이 두 곳
    twol = sum(1 for li, c in s["items"] if c["value"] in agreed_by_li.get(li, ()))
    if twol > 0:
        score *= (1.0 + 0.6 * twol)
    return score, present, distinct


def rolling_analyze(frames, ids, window=5, by_height=False, band=0.05,
                    size_lo=0.6, size_hi=1.7, min_present=2, conf_thr=0.3, hysteresis=0.25):
    """(C) 롤링 FIFO. 각 프레임을 직전 window개 누적으로 판단 → 채널값. primary-유사 dict 반환."""
    pre = [preprocess_frame(im, conf_thr) for im in frames]
    agreed = [within_frame_agreed(cands) for cands in pre]
    per_frame, per_conf, boxes = {}, {}, []
    prev = None                                                 # (cx, cy, mh) 이력 → 히스테리시스
    for i in range(len(frames)):
        lo = max(0, i - window + 1)
        win = pre[lo:i + 1]; nwin = len(win)
        agreed_local = {li: agreed[lo + li] for li in range(nwin)}
        slots = _cluster(win, by_height, band, size_lo, size_hi)
        scored = []
        for s in slots:
            if len(s["frames"]) < min(min_present, nwin):       # 안 묶이는(1회성) 후보 버림
                continue
            sc, present, distinct = _score(s, nwin, agreed_local)
            if prev is not None:                                # 이력 슬롯이면 sticky(흔들림 억제)
                if abs(s["cy"] - prev[1]) < band and abs(s["cx"] - prev[0]) < max(band, 0.15) \
                   and (prev[2] <= 0 or size_lo <= s["mh"] / prev[2] <= size_hi):
                    sc *= (1.0 + hysteresis)
            scored.append((sc, s))
        if not scored:
            continue
        scored.sort(key=lambda x: -x[0])
        best = scored[0][1]
        prev = (best["cx"], best["cy"], best["mh"])
        cur = [c for li, c in best["items"] if li == nwin - 1]  # 현재(마지막) 프레임의 값
        if cur:
            c = max(cur, key=lambda x: x["conf"])
            per_frame[ids[i]] = c["value"]; per_conf[ids[i]] = c["conf"]; boxes.append(c["box"])
    if not boxes:
        return None
    box = [st.median([b[k] for b in boxes]) for k in range(4)]
    return {"box": box, "per_frame": per_frame, "per_frame_conf": per_conf,
            "score": round(len(per_frame) / max(1, len(ids)), 3),
            "distinct": len({_cnorm(v) for v in per_frame.values()}), "duals": []}
