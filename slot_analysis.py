#!/usr/bin/env python3
"""Enhanced slot analysis for channel-number detection across zap frames.

Improves over center-distance-only clustering:
  * clustering uses POSITION **+ SIZE (font height)** as a gate — a candidate joins
    a slot only if it's both near in position AND similar in glyph size. This keeps
    a small nearby number (e.g. a subscript) from merging into the channel slot when
    many boxes crowd around the channel.
  * VALUE DIVERSITY still qualifies the channel (zap changes the value).
  * DUAL-DISPLAY: a second slot whose per-frame values MATCH the primary in every
    common frame is accepted as a second channel readout (some STBs show it twice).
  * vertical CHANNEL-LIST penalty: a slot with a same-x, same-size vertical neighbour
    is likely a pre/next-channel list entry -> down-weighted.

analyze() returns (primary_slot, dual_slots, all_slots_sorted). Each slot exposes
box / per_frame values / geometry so a downstream re-OCR can target the exact ROI.
"""
from __future__ import annotations

import statistics as st

from temporal_profile_select import classify, best_digit, digits


def cluster_slots(frames, ids, pos_thr=0.04, size_lo=0.55, size_hi=1.8, conf_thr=0.3):
    """Greedy clustering by (position within pos_thr) AND (glyph height within size gate).

    conf_thr drops low-confidence OCR false positives (hallucinated digits on dark
    regions, ocr_conf ~0.1) so they never seed spurious slots. Kept conservative
    (0.3) so faint-but-real channel numbers survive; force-read recovers the rest.
    """
    slots = []
    for fi, im in enumerate(frames):
        W = float(im.get("image_width") or 1280) or 1280
        H = float(im.get("image_height") or 720) or 720
        for c in im.get("candidates", []):
            b = c.get("bbox_xyxy"); t = c.get("text", "")
            if not b or len(b) != 4 or not digits(t):
                continue
            if float(c.get("ocr_conf", 1.0) or 0.0) < conf_thr:   # 저신뢰 헛읽음 제외
                continue
            cx, cy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
            hgt = (b[3] - b[1]) / H
            best, bd = None, pos_thr
            for s in slots:
                pos_d = ((s["cx"] - cx) ** 2 + (s["cy"] - cy) ** 2) ** 0.5
                if pos_d >= bd:
                    continue
                if s["mh"] > 0:                                  # 크기(폰트) 게이트
                    r = hgt / s["mh"]
                    if not (size_lo <= r <= size_hi):
                        continue
                bd, best = pos_d, s
            if best is None:
                best = {"cx": cx, "cy": cy, "mh": hgt, "items": [], "boxes": [],
                        "cxs": [], "cys": [], "hs": []}
                slots.append(best)
            best["items"].append({"frame": fi, "uid": ids[fi], "text": t, "box": b,
                                  "type": classify(t), "value": best_digit(t),
                                  "conf": float(c.get("ocr_conf", 0.5) or 0.5)})
            best["boxes"].append(b); best["cxs"].append(cx); best["cys"].append(cy)
            best["hs"].append(hgt)
            k = len(best["items"])
            best["cx"] = (best["cx"] * (k - 1) + cx) / k
            best["cy"] = (best["cy"] * (k - 1) + cy) / k
            best["mh"] = st.median(best["hs"])
    return slots


def _metrics(s, n, W0, H0):
    items = s["items"]; types = [m["type"] for m in items]
    present = len(set(m["frame"] for m in items))
    chan_ratio = sum(t == "channelnum" for t in types) / max(1, len(types))
    text_ratio = sum(t == "text" for t in types) / max(1, len(types))
    time_ratio = sum(t in ("time", "date") for t in types) / max(1, len(types))
    vals = [m["value"] for m in items if m["type"] == "channelnum" and m["value"]]
    distinct = len(set(vals))
    box = [st.median([b[i] for b in s["boxes"]]) for i in range(4)]
    w = max(1.0, box[2] - box[0]); h = max(1.0, box[3] - box[1])
    aspect = w / h; area = (w * h) / (W0 * H0)
    pos_std = (st.pstdev(s["cxs"]) + st.pstdev(s["cys"])) if len(s["cxs"]) > 1 else 0.0
    h_cv = (st.pstdev(s["hs"]) / (sum(s["hs"]) / len(s["hs"]))) if len(s["hs"]) > 1 and sum(s["hs"]) > 0 else 0.0
    score = chan_ratio * (present / max(1, n))
    score *= (1.0 + 0.15 * distinct) if distinct >= 2 else 0.1     # 값 다양성 자격
    if time_ratio > 0.4 or text_ratio > 0.5:
        score = 0.0
    if not (0.25 <= aspect <= 6.0):
        score *= 0.15
    if area > 0.06:
        score *= 0.15
    if pos_std > 0.03:
        score *= 0.5
    if h_cv > 0.35:
        score *= 0.6
    # 프레임별 채널값 + 신뢰도(같은 프레임에 여러 후보면 최고 conf 채택)
    pf, pfc = {}, {}
    for m in items:
        if m["type"] == "channelnum" and m["value"]:
            c = m.get("conf", 0.5)
            if m["uid"] not in pfc or c > pfc[m["uid"]]:
                pf[m["uid"]] = m["value"]; pfc[m["uid"]] = c
    return {"box": [round(v, 1) for v in box], "score": round(score, 3), "distinct": distinct,
            "present": present, "n": n, "aspect": round(aspect, 2), "area": round(area, 4),
            "h": h, "cx": (box[0] + box[2]) / 2 / W0, "cy": (box[1] + box[3]) / 2 / H0,
            "chan_ratio": round(chan_ratio, 2), "per_frame": pf, "per_frame_conf": pfc,
            "sample": [m["text"] for m in items[:5]], "vertical_neighbor": False}


def analyze(frames, ids, pos_thr=0.04, conf_thr=0.3):
    """Return (primary, duals, all_sorted). primary = channel field slot (or None).

    conf_thr는 저신뢰 OCR 후보를 클러스터링 전에 걸러냄. 낮추면 none(미탐지)이 줄지만
    노이즈 숫자가 통과해 오답이 늘 수 있으니 값을 바꿔가며 확인.
    """
    n = len(frames)
    W0 = float(frames[0].get("image_width") or 1280) or 1280
    H0 = float(frames[0].get("image_height") or 720) or 720
    slots = cluster_slots(frames, ids, pos_thr, conf_thr=conf_thr)
    ms = [_metrics(s, n, W0, H0) for s in slots]
    # 세로 채널리스트 페널티: 같은 x + 같은 높이 + 세로 이웃 = 이전/다음 채널 리스트
    for a in ms:
        for b in ms:
            if a is b:
                continue
            if abs(a["cx"] - b["cx"]) < 0.03 and abs(a["h"] - b["h"]) / max(1.0, a["h"]) < 0.3 \
               and 0.02 < abs(a["cy"] - b["cy"]) < 0.18:
                a["score"] = round(a["score"] * 0.6, 3); a["vertical_neighbor"] = True
                break
    ms.sort(key=lambda m: -m["score"])
    primary = ms[0] if ms and ms[0]["score"] > 0 else None
    # 듀얼 표시: primary와 프레임별 값이 (공통 프레임에서) 모두 일치하는 다른 슬롯
    duals = []
    if primary:
        pv = primary["per_frame"]
        for m in ms[1:]:
            if m["score"] <= 0:
                continue
            common = set(pv) & set(m["per_frame"])
            if len(common) >= 2 and all(digits(pv[u]) == digits(m["per_frame"][u]) for u in common):
                duals.append(m)
    return primary, duals, ms
