#!/usr/bin/env python3
"""Temporal profiling: find the CHANNEL-NUMBER field from consecutive frames using
UI-invariant signals only (no learned UI layout -> no overfitting to your designs).

Every frame may be a DIFFERENT channel (captured on channel change), so the channel
VALUE changes each frame and value-constancy is useless. What IS stable is the UI:
a fixed screen slot that consistently behaves a certain way. We cluster the number
candidates by screen position across frames and score each position by how much it
looks like the channel-number slot, using these UI-invariant signals:

  type consistency : slot consistently holds a PURE 1-4 digit number
                     (time = HH:MM, date = d/d, name/program = text -> excluded)
  mutual exclusion : a slot that is consistently text (program/name) is NOT channel
  shape            : channel box is not extremely wide/tall and not huge
                     (aspect ~0.25-5, small area) -> banners / bars excluded
  position stable  : the slot stays at (almost) the same place across frames
  presence         : appears in most frames

The best-scoring slot = the channel field. The channel number for each frame = that
slot's value in that frame. Box = per-slot median (fixes YOLO jitter / over-large).

Input: an OCR candidates JSON (images[].candidates[] with text + bbox_xyxy) + a
sequence manifest (group -> [image_ids]). No model needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path

TIME = re.compile(r"\d{1,2}\s*:\s*\d{2}")
DATE = re.compile(r"\d{1,4}\s*[/\-.]\s*\d{1,2}")


def digits(s):
    return "".join(c for c in str(s) if c.isdigit())


def best_digit(text):
    """채널 값 = 가장 긴 1~5자리 숫자 덩어리 (접두사/공백 무시)."""
    toks = [t for t in re.findall(r"\d+", str(text)) if 1 <= len(t) <= 5]
    return max(toks, key=len) if toks else ""


def classify(text):
    """time / date / channelnum(순수 or 접두사붙은 숫자) / othernum / text."""
    t = str(text)
    if TIME.search(t):
        return "time"
    if DATE.search(t):
        return "date"
    dg = digits(t)
    if not dg:
        return "text"
    nonspace = re.sub(r"\s", "", t)
    if len(dg) == len(nonspace) and 1 <= len(dg) <= 5:
        return "channelnum"                     # 순수 1~5자리
    # 접두사 붙은 채널: 숫자 하나 + 나머지가 짧은 라틴 글자 (CH123, Channel 123)
    toks = [x for x in re.findall(r"\d+", t) if 1 <= len(x) <= 5]
    if len(toks) == 1:
        rest = re.sub(r"\d", "", t).strip()
        if rest == "" or (rest.isascii() and rest.replace(".", "").replace("#", "").isalpha()
                          and len(rest) <= 8):
            return "channelnum"                 # 접두사형 (한글/기호 섞인 프로그램명은 제외)
    return "othernum"                           # 이름/프로그램/코드 (다른 숫자 섞임)


def load_sequences(manifest):
    d = json.loads(Path(manifest).read_text())
    return {s.get("group_key") or s.get("sequence_id"): s.get("images", [])
            for s in d.get("sequences", [])}


def profile_sequence(frames, ids, dist_thr=0.05):
    n = len(frames)
    W0 = float(frames[0].get("image_width") or 1280) or 1280
    H0 = float(frames[0].get("image_height") or 720) or 720
    clusters = []
    for fi, im in enumerate(frames):
        W = float(im.get("image_width") or W0) or W0
        H = float(im.get("image_height") or H0) or H0
        for c in im.get("candidates", []):
            b = c.get("bbox_xyxy")
            text = c.get("text", "")
            if not b or len(b) != 4 or not digits(text):
                continue
            cx, cy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
            best, bd = None, dist_thr
            for cl in clusters:
                d2 = ((cl["cx"] - cx) ** 2 + (cl["cy"] - cy) ** 2) ** 0.5
                if d2 < bd:
                    bd, best = d2, cl
            if best is None:
                best = {"cx": cx, "cy": cy, "items": [], "boxes": [], "cxs": [], "cys": []}
                clusters.append(best)
            best["items"].append({"frame": fi, "text": text, "type": classify(text),
                                  "value": best_digit(text)})
            best["boxes"].append(b); best["cxs"].append(cx); best["cys"].append(cy)
            k = len(best["items"])
            best["cx"] = (best["cx"] * (k - 1) + cx) / k
            best["cy"] = (best["cy"] * (k - 1) + cy) / k

    profs = []
    for cl in clusters:
        items = cl["items"]; types = [it["type"] for it in items]
        present = len(set(it["frame"] for it in items))
        chan_ratio = sum(t == "channelnum" for t in types) / max(1, len(types))
        text_ratio = sum(t == "text" for t in types) / max(1, len(types))
        time_ratio = sum(t in ("time", "date") for t in types) / max(1, len(types))
        # --- 값 다양성 (핵심): zap하면 값이 바뀌는 자리만 채널. 시계/고정숫자는 값이 안 변함 ---
        chan_vals = [it["value"] for it in items if it["type"] == "channelnum" and it["value"]]
        distinct = len(set(chan_vals))
        mbox = [st.median([b[k] for b in cl["boxes"]]) for k in range(4)]
        w = max(1.0, mbox[2] - mbox[0]); h = max(1.0, mbox[3] - mbox[1])
        aspect = w / h
        area_frac = (w * h) / (W0 * H0)
        pos_std = (st.pstdev(cl["cxs"]) + st.pstdev(cl["cys"])) if len(cl["cxs"]) > 1 else 0.0
        hs = [b[3] - b[1] for b in cl["boxes"]]
        h_mean = sum(hs) / max(1, len(hs))
        h_cv = (st.pstdev(hs) / h_mean) if (len(hs) > 1 and h_mean > 0) else 0.0

        score = chan_ratio * (present / max(1, n))
        # === 값 다양성 자격 조건 (v3 slot analysis) ===
        #   값이 2가지 이상 = 채널(zap마다 변함).  1가지뿐 = 시계/로고/고정숫자 -> 구조적 탈락
        score *= (1.0 + 0.15 * distinct) if distinct >= 2 else 0.1
        # --- UI-invariant gates ---
        if time_ratio > 0.4:                     # 시간/날짜 슬롯 = 채널 아님
            score = 0.0
        if text_ratio > 0.5:                     # 상호배제: 텍스트(이름/프로그램) 슬롯
            score = 0.0
        if not (0.25 <= aspect <= 6.0):          # 극단적 가로/세로 박스 = 배너/바 (5자리 위해 6까지)
            score *= 0.15
        if area_frac > 0.06:                     # 너무 큰 박스(배너)
            score *= 0.15
        if pos_std > 0.03:                       # 위치 불안정
            score *= 0.5
        if h_cv > 0.35:                          # 글자 높이 들쭉날쭉 = 채널(고정폰트) 아님
            score *= 0.6
        profs.append({
            "chan_ratio": round(chan_ratio, 2), "present": f"{present}/{n}",
            "distinct_values": distinct, "text_ratio": round(text_ratio, 2),
            "time_ratio": round(time_ratio, 2),
            "aspect": round(aspect, 2), "area%": round(area_frac * 100, 2),
            "pos_std": round(pos_std, 3), "h_cv": round(h_cv, 2), "score": round(score, 3),
            "median_box": [round(v, 1) for v in mbox],
            "per_frame": {ids[it["frame"]]: it["value"] for it in items if it["type"] == "channelnum"},
            "sample": [it["text"] for it in items[:4]],
        })
    profs.sort(key=lambda p: -p["score"])
    return profs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dist-thr", type=float, default=0.05)
    args = ap.parse_args()
    cand = json.loads(Path(args.candidates).read_text())
    by_id = {im["image_id"]: im for im in cand.get("images", [])}
    seqs = load_sequences(args.manifest)

    results, rows = [], []
    for group, ids in seqs.items():
        frames = [by_id[i] for i in ids if i in by_id]
        ok = [i for i in ids if i in by_id]
        if not frames:
            continue
        profs = profile_sequence(frames, ok, args.dist_thr)
        field = profs[0] if profs and profs[0]["score"] > 0 else None
        results.append({"group": group, "n_frames": len(frames),
                        "channel_field_box": field["median_box"] if field else None,
                        "field_score": field["score"] if field else 0.0, "profiles": profs[:8]})
        if field:
            for fid, val in field["per_frame"].items():
                rows.append({"group": group, "frame": fid, "channel_number": val})

    out = Path(args.out)
    out.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["group", "frame", "channel_number"])
        w.writeheader(); w.writerows(rows)

    print("=== 채널 필드 탐지 (타입일관성+비율+크기+위치+상호배제) ===")
    for r in results:
        print(f"\n[{r['group']}] 채널필드={r['channel_field_box']} 점수={r['field_score']}")
        print(f"  {'순수%':>6}{'출현':>7}{'텍스트%':>8}{'시간%':>7}{'비율':>6}{'면적%':>7}"
              f"{'위치std':>8}{'점수':>7}  샘플")
        for p in r["profiles"][:6]:
            print(f"  {p['chan_ratio']:>6}{p['present']:>7}{p['text_ratio']:>8}{p['time_ratio']:>7}"
                  f"{p['aspect']:>6}{p['area%']:>7}{p['pos_std']:>8}{p['score']:>7}  {p['sample']}")
    print(f"\n프레임별 채널번호: {out.with_suffix('.csv')}  ({len(rows)}개)")


if __name__ == "__main__":
    main()
