#!/usr/bin/env python3
"""Temporal profiling: locate the CHANNEL-NUMBER field using the fixed UI, when
every frame is a DIFFERENT channel (captured on each channel change).

Because each frame is a different channel, the channel-number VALUE changes every
frame, so value-constancy is useless. What IS stable is the UI layout: each fixed
screen position consistently shows the same TYPE of content:
    channel-number slot -> always a pure 1-4 digit number (value differs per frame)
    clock slot          -> always HH:MM
    name/program slot   -> always text
    date slot           -> always d/d

So we cluster number candidates by screen position across frames and score each
position by how consistently it holds a PURE 1-4 digit number (not a time, date,
symbol, or text-embedded number). The best position = the channel-number field;
the channel number for each frame = that field's value in that frame. The field
box is the per-position MEDIAN (fixes YOLO jitter / over-large boxes).

Input = an OCR candidates JSON (images[].candidates[] with text + bbox_xyxy) + a
sequence manifest (group -> [image_ids]). No model needed.

Usage:
    python temporal_profile_select.py --candidates OUT/full_ocr.json \
        --manifest OUT/manifest.json --out OUT/profile_result.json
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


def classify(text):
    """What kind of content is this OCR token?"""
    t = str(text)
    if TIME.search(t):
        return "time"
    if DATE.search(t):
        return "date"
    dg = digits(t)
    if not dg:
        return "text"
    nonspace = re.sub(r"\s", "", t)
    if len(dg) == len(nonspace) and 1 <= len(dg) <= 4:
        return "channelnum"                     # pure 1-4 digit (channel-number-like)
    return "othernum"                           # long number / digits mixed with letters


def load_sequences(manifest):
    d = json.loads(Path(manifest).read_text())
    out = {}
    for s in d.get("sequences", []):
        out[s.get("group_key") or s.get("sequence_id")] = s.get("images", [])
    return out


def profile_sequence(frames, ids, dist_thr=0.05):
    n = len(frames)
    clusters = []
    for fi, im in enumerate(frames):
        W = float(im.get("image_width") or 1) or 1
        H = float(im.get("image_height") or 1) or 1
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
                best = {"cx": cx, "cy": cy, "items": [], "boxes": []}
                clusters.append(best)
            best["items"].append({"frame": fi, "text": text, "type": classify(text),
                                  "value": digits(text)})
            best["boxes"].append(b)
            k = len(best["items"])
            best["cx"] = (best["cx"] * (k - 1) + cx) / k
            best["cy"] = (best["cy"] * (k - 1) + cy) / k

    profiles = []
    for cl in clusters:
        items = cl["items"]
        present = len(set(it["frame"] for it in items))
        types = [it["type"] for it in items]
        chan = sum(t == "channelnum" for t in types)
        chan_ratio = chan / max(1, len(types))
        time_ratio = sum(t == "time" for t in types) / max(1, len(types))
        text_ratio = sum(t == "text" for t in types) / max(1, len(types))
        mbox = [round(st.median([b[k] for b in cl["boxes"]]), 1) for k in range(4)]
        # channel field = consistently a pure short number, present in most frames
        score = round(chan_ratio * (present / max(1, n)), 3)
        profiles.append({
            "chan_ratio": round(chan_ratio, 2), "present": f"{present}/{n}",
            "time_ratio": round(time_ratio, 2), "text_ratio": round(text_ratio, 2),
            "score": score, "median_box": mbox,
            "per_frame": {ids[it["frame"]]: it["value"] for it in items if it["type"] == "channelnum"},
            "sample": [it["text"] for it in items[:4]],
        })
    profiles.sort(key=lambda p: -p["score"])
    return profiles


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

    results, per_frame_rows = [], []
    for group, ids in seqs.items():
        frames = [by_id[i] for i in ids if i in by_id]
        ok_ids = [i for i in ids if i in by_id]
        if not frames:
            continue
        profs = profile_sequence(frames, ok_ids, args.dist_thr)
        field = profs[0] if profs and profs[0]["score"] > 0 else None
        results.append({"group": group, "n_frames": len(frames),
                        "channel_field_box": field["median_box"] if field else None,
                        "field_score": field["score"] if field else 0.0,
                        "profiles": profs[:8]})
        if field:
            for fid, val in field["per_frame"].items():
                per_frame_rows.append({"group": group, "frame": fid, "channel_number": val})

    out = Path(args.out)
    out.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["group", "frame", "channel_number"])
        w.writeheader(); w.writerows(per_frame_rows)

    print("=== 채널번호 '필드' 탐지 (항상 순수 숫자인 위치) + 프레임별 값 ===")
    for r in results:
        print(f"\n[{r['group']}]  채널필드 박스={r['channel_field_box']}  점수={r['field_score']}")
        print(f"  {'순수숫자비율':>10}{'출현':>8}{'시간비율':>8}{'텍스트비율':>10}{'점수':>7}  샘플")
        for p in r["profiles"][:5]:
            print(f"  {p['chan_ratio']:>10}{p['present']:>8}{p['time_ratio']:>8}"
                  f"{p['text_ratio']:>10}{p['score']:>7}  {p['sample']}")
    print(f"\n프레임별 채널번호: {out.with_suffix('.csv')}  ({len(per_frame_rows)}개)")


if __name__ == "__main__":
    main()
