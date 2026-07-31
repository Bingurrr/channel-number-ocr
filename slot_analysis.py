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


def cluster_slots(frames, ids, pos_thr=0.04, size_lo=0.55, size_hi=1.8, conf_thr=0.3,
                  by_size=False, band_thr=0.04):
    """Greedy clustering by (position within pos_thr) AND (glyph height within size gate).

    by_size=True: '글자 높이(폰트 크기)' + '평행이동 정렬'로 묶는다. 채널번호는 채널마다
    위치가 달라도 높이가 일정하고 그 이동이 순수 평행이동(같은 행=좌우 이동, 또는 같은
    열=상하 이동)이라, 높이가 같고 같은 행 '또는' 같은 열(band_thr 이내)에 있으면 한 슬롯으로
    모은다. 무관한 같은-높이 숫자(대각선 위치)는 제외된다.

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
            if by_size:                                           # 높이 + 평행이동 정렬로 묶기
                best, bd = None, 999.0
                for s in slots:
                    if s["mh"] <= 0:
                        continue
                    r = hgt / s["mh"]
                    if not (size_lo <= r <= size_hi):             # 높이 게이트
                        continue
                    dy = abs(cy - s["cy"]); dx = abs(cx - s["cx"])
                    if min(dy, dx) > band_thr:                    # 같은 행 또는 같은 열이어야(평행이동)
                        continue                                  # 대각선(무관 위치)은 제외
                    d = abs(r - 1.0) + min(dy, dx)                # 높이 유사 + 축 정렬 정도
                    if d < bd:
                        bd, best = d, s
            else:                                                 # 위치 + 높이 (기존)
                best, bd = None, pos_thr
                for s in slots:
                    pos_d = ((s["cx"] - cx) ** 2 + (s["cy"] - cy) ** 2) ** 0.5
                    if pos_d >= bd:
                        continue
                    if s["mh"] > 0:                               # 크기(폰트) 게이트
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


def _metrics(s, n, W0, H0, by_size=False):
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
    if not by_size and pos_std > 0.03:
        score *= 0.5           # 원본: 위치 흔들리면 감점 (85% 기준). by_size면 위치 무시
    if h_cv > 0.35:
        score *= 0.6           # by_size에서도 높이 일관성은 요구 (핵심 신호)
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


def analyze_streaming(frames, ids, warmup=15, pos_thr=0.04, conf_thr=0.3, by_size=False,
                      size_lo=0.55, size_hi=1.8, band_thr=0.04, relock_every=0):
    """[on-device] 워밍업-락: 처음 warmup개로 채널 영역을 '한 번' 확정 → 이후 각 프레임은 그
    고정 영역에서 O(1)로 읽기만. 무상태 재클러스터링(윈도우마다 흔들림)의 불안정을 없앤다.

    영역 자체는 O(1) 통계(좌표/높이)로 유지 → TV CPU 메모리 문제없음(원시 프레임은 안 쌓음).
    relock_every>0 이면 그만큼 프레임마다 최근 warmup개로 영역을 다시 확정(느린 적응).
    반환: primary-유사 dict 또는 None.
    """
    W0 = float(frames[0].get("image_width") or 1280) or 1280
    H0 = float(frames[0].get("image_height") or 720) or 720

    def lock_region(lo, hi):
        pr, du, _ = analyze(frames[lo:hi], ids[lo:hi], pos_thr, conf_thr, by_size)
        if not pr:
            return None
        b = pr["box"]
        return {"cx": (b[0] + b[2]) / 2 / W0, "cy": (b[1] + b[3]) / 2 / H0,
                "mh": (b[3] - b[1]) / H0, "box": b,
                "pf": dict(pr["per_frame"]), "pfc": dict(pr.get("per_frame_conf", {}))}

    reg = lock_region(0, min(warmup, len(frames)))
    if reg is None:                                          # 워밍업 실패 → 전체로 한 번
        reg = lock_region(0, len(frames))
    if reg is None:
        return None

    per_frame, per_conf, boxes = {}, {}, []
    for fi, im in enumerate(frames):
        uid = ids[fi]
        if relock_every and fi > 0 and fi % relock_every == 0:   # 느린 재확정(적응)
            r2 = lock_region(max(0, fi - warmup + 1), fi + 1)
            if r2:
                reg = r2
        # 워밍업 창이 이미 채운 값 우선
        if uid in reg["pf"]:
            per_frame[uid] = reg["pf"][uid]; per_conf[uid] = reg["pfc"].get(uid, 0.5); boxes.append(reg["box"]); continue
        # 고정 영역 근처 최고conf 채널숫자 읽기 (O(1))
        W = float(im.get("image_width") or 1280) or 1280
        H = float(im.get("image_height") or 720) or 720
        best = None
        for c in im.get("candidates", []):
            b = c.get("bbox_xyxy"); t = c.get("text", "")
            if not b or len(b) != 4 or classify(t) != "channelnum":
                continue
            v = _cnorm(best_digit(t)); cf = float(c.get("ocr_conf", 0.5) or 0.5)
            if not v or cf < conf_thr:
                continue
            cx, cy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
            hgt = (b[3] - b[1]) / H
            if by_size:
                if reg["mh"] > 0 and not (size_lo <= hgt / reg["mh"] <= size_hi):
                    continue
                if min(abs(cy - reg["cy"]), abs(cx - reg["cx"])) > band_thr:   # 같은 행/열(평행이동)
                    continue
            else:
                if ((cx - reg["cx"]) ** 2 + (cy - reg["cy"]) ** 2) ** 0.5 > pos_thr:
                    continue
            if best is None or cf > best[0]:
                best = (cf, v)
        if best:
            per_frame[uid] = best[1]; per_conf[uid] = best[0]; boxes.append(reg["box"])
    if not boxes:
        return None
    box = [st.median([b[k] for b in boxes]) for k in range(4)]
    return {"box": box, "per_frame": per_frame, "per_frame_conf": per_conf,
            "score": round(len(per_frame) / max(1, len(ids)), 3),
            "distinct": len({_cnorm(v) for v in per_frame.values()}), "duals": []}


def analyze_windowed(frames, ids, window=5, pos_thr=0.04, conf_thr=0.3, by_size=False):
    """각 프레임을 '직전 window개' 슬라이딩 윈도우로만 클러스터링해 그 프레임의 채널값을 정한다.

    on-device 실시간(최근 N프레임만 유지) 시뮬레이션. 폴더 전체(1000+)를 한 번에 묶지 않고,
    프레임 i는 [i-window+1 .. i] 만 보고 판단 → 실제 배포 환경과 동일. primary-유사 dict 반환.
    """
    per_frame, confs, boxes = {}, {}, []
    for i, uid in enumerate(ids):
        lo = max(0, i - window + 1)
        wf, wu = frames[lo:i + 1], ids[lo:i + 1]
        primary, duals, _ = analyze(wf, wu, pos_thr=pos_thr, conf_thr=conf_thr, by_size=by_size)
        if not primary:
            continue
        v = primary["per_frame"].get(uid); c = primary.get("per_frame_conf", {}).get(uid, 0.5)
        if v is None:                                        # 윈도우 primary가 이 프레임을 못 채우면 듀얼로
            for d in duals:
                if uid in d["per_frame"]:
                    v = d["per_frame"][uid]; c = d.get("per_frame_conf", {}).get(uid, 0.5); break
        if v is not None:
            per_frame[uid] = v; confs[uid] = c; boxes.append(primary["box"])
    if not boxes:
        return None
    box = [st.median([b[k] for b in boxes]) for k in range(4)]
    return {"box": box, "per_frame": per_frame, "per_frame_conf": confs,
            "score": round(len(per_frame) / max(1, len(ids)), 3),
            "distinct": len({_cnorm(v) for v in per_frame.values()}), "duals": []}


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


def _numcands_norm(im):
    """모든 후보에서 숫자만 남긴 정규화 후보: (value, cx, cy, conf). 'YTN' 등 글자 제거."""
    import re
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
        out.append((d, (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H, float(c.get("ocr_conf", 0.5) or 0.5)))
    return out


def learn_second_region(frames, ids, primary_of, field_box, min_sep=0.10, min_votes=5):
    """[v3/사용자 아이디어] primary가 확신한 프레임들에서 'primary값과 같은 숫자가 뜨는
    다른 위치'를 프레임마다 누적 투표 → 확정된 2번째 채널 표시 위치를 학습.

    진짜 2번째 채널영역은 매 프레임 primary와 같이 값이 바뀜 → 일치 투표 많음.
    프로그램 숫자는 우연히만 일치 → 투표 적음 → min_votes로 걸러짐.
    반환: (region_cx, region_cy, votes) 정규화좌표, 또는 None.
    """
    W0 = float(frames[0].get("image_width") or 1280) or 1280
    H0 = float(frames[0].get("image_height") or 720) or 720
    fcx = (field_box[0] + field_box[2]) / 2 / W0
    fcy = (field_box[1] + field_box[3]) / 2 / H0
    clusters = []                                        # [cx, cy, npts, votes]
    for fi, im in enumerate(frames):
        pv = primary_of.get(ids[fi])
        if not pv:
            continue
        pvn = _cnorm(pv)
        for v, cx, cy, cf in _numcands_norm(im):
            if _cnorm(v) != pvn:
                continue
            if ((cx - fcx) ** 2 + (cy - fcy) ** 2) ** 0.5 <= min_sep:
                continue                                 # primary 위치 자신은 제외
            r = next((r for r in clusters if (cx - r[0]) ** 2 + (cy - r[1]) ** 2 < min_sep ** 2), None)
            if r:
                n = r[2]; r[0] = (r[0] * n + cx) / (n + 1); r[1] = (r[1] * n + cy) / (n + 1); r[2] = n + 1; r[3] += 1
            else:
                clusters.append([cx, cy, 1, 1])
    if not clusters:
        return None
    best = max(clusters, key=lambda r: r[3])
    return (best[0], best[1], best[3]) if best[3] >= min_votes else None


def read_at_region(im, region, near=0.08):
    """한 프레임에서 region(정규화 cx,cy) 근처 최고conf 숫자후보 → (value, conf)."""
    if region is None:
        return (None, 0.0)
    pool = [(v, cf) for v, cx, cy, cf in _numcands_norm(im)
            if ((cx - region[0]) ** 2 + (cy - region[1]) ** 2) ** 0.5 <= near]
    if not pool:
        return (None, 0.0)
    pool.sort(key=lambda x: -x[1])
    return pool[0]


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


def analyze(frames, ids, pos_thr=0.04, conf_thr=0.3, by_size=False):
    """Return (primary, duals, all_sorted). primary = channel field slot (or None).

    by_size=True: 위치가 아니라 '글자 높이(폰트 크기)'로 클러스터링/선택 → 채널마다 채널박스
    위치가 달라지는 UI에 강함(값 다양성 + 채널숫자 + 높이 일관성으로 선택).

    conf_thr는 저신뢰 OCR 후보를 클러스터링 전에 걸러냄. 낮추면 none(미탐지)이 줄지만
    노이즈 숫자가 통과해 오답이 늘 수 있으니 값을 바꿔가며 확인.
    """
    n = len(frames)
    W0 = float(frames[0].get("image_width") or 1280) or 1280
    H0 = float(frames[0].get("image_height") or 720) or 720
    slots = cluster_slots(frames, ids, pos_thr, conf_thr=conf_thr, by_size=by_size)
    ms = [_metrics(s, n, W0, H0, by_size=by_size) for s in slots]
    # 세로 채널리스트 페널티: 같은 x + 같은 높이 + 세로 이웃 = 이전/다음 채널 리스트
    # (by_size에선 위치가 흩어져 있어 이 페널티가 잘못 걸릴 일이 적음)
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
