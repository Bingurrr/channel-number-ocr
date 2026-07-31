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


def _cnorm(v):
    """채널값 정규화: '002' 와 '2' 를 같은 채널로 (앞의 0 제거)."""
    s = str(v)
    return str(int(s)) if s.isdigit() else s


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
    vals = [_cnorm(m["value"]) for m in items if m["type"] == "channelnum" and m["value"]]
    distinct = len(set(vals))         # 002/2 같은 채널로 (앞 0 무시)
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
        score *= 0.5           # 원본: 위치 흔들리면 감점 (85% 기준)
    if h_cv > 0.35:
        score *= 0.6
    # 프레임별 채널값 + 신뢰도(같은 프레임에 여러 후보면 최고 conf 채택)
    pf, pfc = {}, {}
    for m in items:
        if m["type"] == "channelnum" and m["value"]:
            c = m.get("conf", 0.5)
            if m["uid"] not in pfc or c > pfc[m["uid"]]:
                pf[m["uid"]] = _cnorm(m["value"]); pfc[m["uid"]] = c
    # (avg_conf는 참고용으로만 노출 — 점수엔 곱하지 않음. 85% 원본 유지)
    avg_conf = (sum(pfc.values()) / len(pfc)) if pfc else 0.5
    return {"box": [round(v, 1) for v in box], "score": round(score, 3),
            "avg_conf": round(avg_conf, 3), "distinct": distinct,
            "present": present, "n": n, "aspect": round(aspect, 2), "area": round(area, 4),
            "h": h, "cx": (box[0] + box[2]) / 2 / W0, "cy": (box[1] + box[3]) / 2 / H0,
            "chan_ratio": round(chan_ratio, 2), "pos_std": round(pos_std, 4),
            "per_frame": pf, "per_frame_conf": pfc,
            "sample": [m["text"] for m in items[:5]], "vertical_neighbor": False}


def within_frame_dupes(frames, ids, conf_thr=0.3, min_sep=0.05):
    """프레임 '내부'에서 같은 숫자가 2곳 이상(멀리 떨어진 위치)에 뜨면 그 값이 채널일 확률이
    높다 — 위치가 고정이든 '움직이든' 무관한 신호라 skylife의 이동 채널박스에 강함.

    반환: {uid: value}. 오탐 방지로 순수 1~5자리 + 두 위치가 min_sep 이상 떨어진 경우만.
    """
    out = {}
    for fi, im in enumerate(frames):
        W = float(im.get("image_width") or 1280) or 1280
        H = float(im.get("image_height") or 720) or 720
        by_val = {}
        for c in im.get("candidates", []):
            b = c.get("bbox_xyxy"); t = c.get("text", "")
            if not b or len(b) != 4 or classify(t) != "channelnum":
                continue
            v = best_digit(t)
            if not v or float(c.get("ocr_conf", 0.5) or 0.5) < conf_thr:
                continue
            cx, cy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
            by_val.setdefault(_cnorm(v), []).append((cx, cy, float(c.get("ocr_conf", 0.5) or 0.5)))
        best = None
        for v, pts in by_val.items():
            distinct = []                                  # 서로 멀리 떨어진 위치만
            for p in pts:
                if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_sep ** 2 for q in distinct):
                    distinct.append(p)
            if len(distinct) >= 2:
                sc = len(distinct) + max(p[2] for p in distinct)
                if best is None or sc > best[1]:
                    best = (v, sc)
        if best:
            out[ids[fi]] = best[0]
    return out


def read_at_field_box(frames, ids, field_box, conf_thr=0.3, near=0.08):
    """field box(v1이 찾은 채널 위치) 근처 '최고 conf 채널숫자'를 읽는다.

    튜너 검증: 이게 읽은것중 91.5% 정확 (v1의 87.7%보다 높음). 프로그램 숫자는
    다른 위치라 near 밖 → 배제. 반환: {uid: value} (근처 후보 없으면 그 프레임은 없음).
    """
    W0 = float(frames[0].get("image_width") or 1280) or 1280
    H0 = float(frames[0].get("image_height") or 720) or 720
    fcx = (field_box[0] + field_box[2]) / 2 / W0
    fcy = (field_box[1] + field_box[3]) / 2 / H0
    out = {}
    for fi, im in enumerate(frames):
        W = float(im.get("image_width") or 1280) or 1280
        H = float(im.get("image_height") or 720) or 720
        best = None
        for c in im.get("candidates", []):
            b = c.get("bbox_xyxy"); t = c.get("text", "")
            if not b or len(b) != 4 or classify(t) != "channelnum":
                continue
            v = best_digit(t); cf = float(c.get("ocr_conf", 0.5) or 0.5)
            if not v or cf < conf_thr:
                continue
            cx, cy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
            if ((cx - fcx) ** 2 + (cy - fcy) ** 2) ** 0.5 <= near and (best is None or cf > best[0]):
                best = (cf, v)
        if best:
            out[ids[fi]] = best[1]
    return out


def resolve_by_agreement(frames, ids, conf_thr=0.3, min_sep=0.08):
    """데이터 기반 채널 선택 (선택 문제 82% 해결용).

    핵심: '같은 숫자가 한 프레임에 2곳(멀리)에 나오면 채널' (우상단+좌하단 동시).
      프로그램 코드는 1곳만 → 배제. 채널은 2곳 일치 → 확정.
    1) 프레임내 2곳 일치값 = 그 프레임 채널 (강함). 일치가 일어난 '위치'들을 모음.
    2) 모인 위치 = 확정된 채널 영역(우상단/좌하단 2군데).
    3) 일치 없는 프레임(한쪽만 읽힘) = 확정 영역 근처 최고conf 후보 채택.
       (움직이는 좌하단도 확정 영역에 속하면 잡힘, 프로그램코드는 영역 밖이라 배제)

    반환: {uid: value}
    """
    fcands, agree_pf, region_pts = [], {}, []
    for fi, im in enumerate(frames):
        W = float(im.get("image_width") or 1280) or 1280
        H = float(im.get("image_height") or 720) or 720
        cands = []
        for c in im.get("candidates", []):
            b = c.get("bbox_xyxy"); t = c.get("text", "")
            if not b or len(b) != 4 or classify(t) != "channelnum":
                continue
            v = _cnorm(best_digit(t)); cf = float(c.get("ocr_conf", 0.5) or 0.5)
            if not v or cf < conf_thr:
                continue
            cands.append((v, (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H, cf))
        fcands.append(cands)
        byval = {}
        for v, cx, cy, cf in cands:
            byval.setdefault(v, []).append((cx, cy, cf))
        best = None
        for v, pts in byval.items():
            distinct = []
            for p in pts:
                if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_sep ** 2 for q in distinct):
                    distinct.append(p)
            if len(distinct) >= 2:                       # 2곳 일치 = 채널
                sc = sum(p[2] for p in distinct)
                if best is None or sc > best[0]:
                    best = (sc, v, distinct)
        if best:
            agree_pf[ids[fi]] = best[1]
            region_pts.extend((p[0], p[1]) for p in best[2])
    # 일치 위치 군집 = 확정된 채널 영역들
    regions = []
    for p in region_pts:
        r = next((r for r in regions if (p[0] - r[0]) ** 2 + (p[1] - r[1]) ** 2 < min_sep ** 2), None)
        if r:
            r[2].append(p); r[0] = sum(x[0] for x in r[2]) / len(r[2]); r[1] = sum(x[1] for x in r[2]) / len(r[2])
        else:
            regions.append([p[0], p[1], [p]])
    cregions = [(r[0], r[1]) for r in regions]
    out = {}
    for fi in range(len(frames)):
        uid = ids[fi]
        if uid in agree_pf:                              # 1) 프레임내 일치
            out[uid] = agree_pf[uid]
        elif cregions:                                   # 2) 확정 영역 근처 최고conf
            best = None
            for v, cx, cy, cf in fcands[fi]:
                d = min(((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5 for rx, ry in cregions)
                if d <= min_sep and (best is None or cf > best[0]):
                    best = (cf, v)
            if best:
                out[uid] = best[1]
    return out, cregions


def confirmed_second_regions(frames, ids, ref_pf, ref_box, conf_thr=0.3,
                             min_agree=2, sep=0.06):
    """'2번째 채널 위치' 확정 (사용자 아이디어): 기준값(ref_pf, 보통 우측상단 값)과
    '다른 위치'에서 같은 값이 나온 게 여러 프레임 반복되면, 그 위치를 채널로 확정.
    프로그램명 같은 우연 숫자는 값이 계속 일치하진 않으니 걸러진다.

    반환: 확정된 2번째 위치 중심 리스트 [(cx,cy),...] (정규화)
    """
    W0 = float(frames[0].get("image_width") or 1280) or 1280
    H0 = float(frames[0].get("image_height") or 720) or 720
    pcx = (ref_box[0] + ref_box[2]) / 2 / W0; pcy = (ref_box[1] + ref_box[3]) / 2 / H0
    agree = []
    for fi, im in enumerate(frames):
        rv = ref_pf.get(ids[fi])
        if not rv:
            continue
        for c in im.get("candidates", []):
            b = c.get("bbox_xyxy"); t = c.get("text", "")
            if not b or len(b) != 4 or classify(t) != "channelnum":
                continue
            if _cnorm(best_digit(t)) != rv or float(c.get("ocr_conf", 0.5) or 0.5) < conf_thr:
                continue
            cx, cy = (b[0] + b[2]) / 2 / W0, (b[1] + b[3]) / 2 / H0
            if (cx - pcx) ** 2 + (cy - pcy) ** 2 < sep ** 2:      # 기준 위치 자신은 제외
                continue
            agree.append((cx, cy))
    regions = []                                                 # 일치 위치 군집화(광고/방송 2위치)
    for p in agree:
        r = next((r for r in regions if (p[0] - r[0]) ** 2 + (p[1] - r[1]) ** 2 < sep ** 2), None)
        if r:
            r[2].append(p); r[0] = sum(x[0] for x in r[2]) / len(r[2]); r[1] = sum(x[1] for x in r[2]) / len(r[2])
        else:
            regions.append([p[0], p[1], [p]])
    return [(r[0], r[1]) for r in regions if len(r[2]) >= min_agree]


def top_channel_candidate(frames, ids, conf_thr=0.3, chan_boxes=None, near=0.08):
    """none 폴백: '확정된 채널 위치(chan_boxes) 근처' 후보만 뱉는다 (하드 게이트).

    이렇게 제한하지 않으면 하단 프로그램 숫자("2","8" 등)를 채널로 오인한다.
    chan_boxes = primary + '값 일치로 확정된 2번째 위치'. 그 근처가 아니면 안 뱉음(none 유지).
    """
    if not chan_boxes:
        return {}
    out = {}
    for fi, im in enumerate(frames):
        W = float(im.get("image_width") or 1280) or 1280
        H = float(im.get("image_height") or 720) or 720
        best = None
        for c in im.get("candidates", []):
            t = c.get("text", ""); b = c.get("bbox_xyxy")
            if not b or len(b) != 4 or classify(t) != "channelnum":
                continue
            v = _cnorm(best_digit(t)); cf = float(c.get("ocr_conf", 0.5) or 0.5)
            if not v or cf < conf_thr:
                continue
            cx, cy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
            d = min(((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5 for bx, by in chan_boxes)
            if d > near:                          # 채널 위치 근처 아니면 제외(프로그램숫자 배제)
                continue
            if best is None or cf > best[0]:
                best = (cf, v)
        if best:
            out[ids[fi]] = best[1]
    return out


def resolve_channel_per_frame(frames, ids, chan_boxes, chan_values, conf_thr=0.3,
                              near=0.06, min_sep=0.05, high_conf=0.6):
    """프레임마다 '읽히는 곳' 우선으로 채널값을 고른다 (사용자 설계).

    핵심: 위치가 고정이냐 아니냐가 아니라 '잘 읽혔냐(conf)'로 고른다.
      우측상단(고정, 못읽음) 대신 좌측하단(이동, 잘읽힘 예: '000 YTN'→0)을 채택.
    후보 채택 조건 (엉뚱한 숫자 배제):
      · 같은 값이 2곳(멀리) → 합의(강함)  또는
      · 값이 폴더 채널값 집합에 있음        또는
      · 단일이라도 high_conf 이상 (잘 읽힌 채널로 신뢰)
    점수 = conf + 합의보너스 + 채널영역근처보너스.
    반환: {uid: (value, conf, agree)}
    """
    cv = set(_cnorm(v) for v in chan_values)
    out = {}
    for fi, im in enumerate(frames):
        W = float(im.get("image_width") or 1280) or 1280
        H = float(im.get("image_height") or 720) or 720
        byval = {}
        for c in im.get("candidates", []):
            b = c.get("bbox_xyxy"); t = c.get("text", "")
            if not b or len(b) != 4 or classify(t) != "channelnum":
                continue
            v = _cnorm(best_digit(t))
            cf = float(c.get("ocr_conf", 0.5) or 0.5)
            if not v or cf < conf_thr:
                continue
            cx, cy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
            byval.setdefault(v, []).append((cx, cy, cf))
        best = None
        for v, pts in byval.items():
            distinct = []
            for p in pts:
                if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_sep ** 2 for q in distinct):
                    distinct.append(p)
            agree = len(distinct) >= 2                       # 2곳 합의
            maxconf = max(p[2] for p in pts)
            if not (agree or v in cv or maxconf >= high_conf):   # 잘읽힌 단일도 허용
                continue
            near_chan = any(min(((p[0] - bx) ** 2 + (p[1] - by) ** 2) ** 0.5
                                for bx, by in chan_boxes) < near for p in pts) if chan_boxes else False
            score = maxconf + (0.5 if agree else 0.0) + (0.2 if near_chan else 0.0)
            if best is None or score > best[0]:
                best = (score, v, maxconf, agree)
        if best:
            out[ids[fi]] = (best[1], best[2], best[3])
    return out


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
