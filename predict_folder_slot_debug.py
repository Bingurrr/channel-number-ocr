#!/usr/bin/env python3
"""Diagnose WHY a predict_folder_slot run failed on specific frames.

Reads a finished run's outputs (no OCR re-run):
    <result>/full_ocr.json      -> per-frame OCR candidates
    <result>/per_frame.csv      -> the pipeline's prediction per frame
    <result>/profile_report.json-> the chosen channel field box (per folder)

For every FAILED frame (pred==none or pred!=gt), it dumps the full picture and
categorizes the failure so you can see the pattern instead of guessing:

  none_but_readable   : pred none, BUT the correct number IS in the OCR candidates
                        -> SELECTION/none-handling problem (the answer was there)
  wrong_but_readable  : pred wrong, correct number IS in candidates
                        -> picked the wrong candidate (selection)
  not_readable        : correct number is NOT in the OCR candidates
                        -> READING problem (det/rec missed it)
  program_confusion   : predicted a value that exists in candidates but isn't gt
                        (e.g. a program number)

Output:
    <result>/failures.json   (detailed, one entry per failed frame)
    stdout summary            (counts per category + a few examples)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from predict_folder import gt_from_name
from temporal_profile_select import classify, best_digit


def _cn(v):
    s = str(v)
    return str(int(s)) if s.isdigit() else s


def _pos(bbox, W, H):
    cx = (bbox[0] + bbox[2]) / 2 / W; cy = (bbox[1] + bbox[3]) / 2 / H
    v = "top" if cy < 0.4 else ("bottom" if cy > 0.6 else "mid")
    h = "left" if cx < 0.4 else ("right" if cx > 0.6 else "center")
    return f"{v}-{h}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="predict_folder_slot --out 결과 디렉토리")
    ap.add_argument("--show", type=int, default=15, help="카테고리별 예시 출력 수")
    args = ap.parse_args()

    res = Path(args.result)
    by_id = {im["image_id"]: im for im in json.loads((res / "full_ocr.json").read_text()).get("images", [])}
    boxes = {}
    pr = res / "profile_report.json"
    if pr.exists():
        for r in json.loads(pr.read_text()):
            if r.get("channel_field_box"):
                boxes[r["folder"]] = r["channel_field_box"]

    # 예측 로드 (per_frame.csv엔 '예측된' 프레임만 있음 → none은 여기 없음)
    preds, folder_of = {}, {}
    with (res / "per_frame.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            preds[r.get("frame", "")] = r.get("channel_number", "")
            folder_of[r.get("frame", "")] = r.get("folder", "")

    # full_ocr(모든 프레임) 기준 순회. 스테이징 uid "{group}__{stem}" → 원본 stem 복원.
    fails, cats = [], Counter()
    n_gt = 0
    for uid, im in by_id.items():
        frame = uid.split("__")[-1]                # 원본 파일명 stem 복원 (".__000"→"000")
        gt = gt_from_name(frame)
        if not gt:
            continue
        n_gt += 1
        gtn = _cn(gt)
        pred = preds.get(frame, "")                # per_frame에 없으면 none
        predn = _cn("".join(ch for ch in str(pred) if ch.isdigit())) if pred else ""
        if pred and predn == gtn:
            continue                                   # 성공
        W = float(im.get("image_width") or 1280) or 1280
        H = float(im.get("image_height") or 720) or 720
        cands = []
        for c in im.get("candidates", []):
            b = c.get("bbox_xyxy"); t = c.get("text", "")
            if not b or len(b) != 4:
                continue
            cl = classify(t); v = _cn(best_digit(t)) if cl in ("channelnum", "othernum") else ""
            if cl in ("channelnum", "othernum"):
                cands.append({"text": t, "value": v, "class": cl,
                              "conf": round(float(c.get("ocr_conf", 0) or 0), 3),
                              "pos": _pos(b, W, H), "bbox": [round(x) for x in b]})
        chan_vals = {c["value"] for c in cands if c["class"] == "channelnum"}
        any_vals = {c["value"] for c in cands}
        gt_readable = gtn in chan_vals
        gt_any = gtn in any_vals
        # 카테고리
        if not pred:
            cat = "none_but_readable" if gt_readable else ("none_gt_as_othernum" if gt_any else "none_not_readable")
        else:
            cat = "wrong_but_readable" if gt_readable else "wrong_not_readable"
        cats[cat] += 1
        fails.append({"frame": frame, "gt": gtn, "pred": pred or "none", "category": cat,
                      "field_box": boxes.get(folder_of.get(frame, ""), None),
                      "gt_readable_as_channel": gt_readable,
                      "gt_conf": sorted([c["conf"] for c in cands if c["value"] == gtn], reverse=True),
                      "channel_candidates": sorted(cands, key=lambda c: -c["conf"])})

    (res / "failures.json").write_text(json.dumps(fails, ensure_ascii=False, indent=2))
    fail_n = len(fails)
    print(f"총 GT {n_gt} / 실패 {fail_n} ({fail_n/max(1,n_gt)*100:.1f}%)  → {res}/failures.json\n")
    print(f"{'category':<22}{'count':>7}{'%실패중':>9}   의미")
    meaning = {
        "none_but_readable": "정답이 후보에 있는데 none → 선택/none처리 문제",
        "wrong_but_readable": "정답이 후보에 있는데 다른값 선택 → 선택 문제",
        "none_not_readable": "정답이 OCR에 아예 없음 → 읽기(det/rec) 문제",
        "wrong_not_readable": "정답 못읽고 다른값 뱉음 → 읽기 문제",
        "none_gt_as_othernum": "정답이 othernum으로 분류됨(YTN24 등) → 분류/선택 문제",
    }
    for cat, n in cats.most_common():
        print(f"{cat:<22}{n:>7}{n/max(1,fail_n)*100:>8.1f}%   {meaning.get(cat,'')}")

    print("\n=== 카테고리별 예시 ===")
    seen = Counter()
    for fr in fails:
        c = fr["category"]
        if seen[c] >= args.show:
            continue
        seen[c] += 1
        top = fr["channel_candidates"][:4]
        cc = "  ".join(f"{d['value'] or '-'}@{d['pos']}({d['conf']})" for d in top)
        print(f"[{c}] {fr['frame']:<18} gt={fr['gt']:<5} pred={fr['pred']:<6} | 후보: {cc}")


if __name__ == "__main__":
    main()
