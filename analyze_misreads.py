#!/usr/bin/env python3
"""Categorize the READ-BUT-WRONG frames (case ①) to shape the rec-training data.

Reads a result dir produced with --gt-from-filename (per_frame.csv where `frame` is
the filename stem and its digits are the GT). Classifies each wrong prediction:

  len_short   pred drops digit(s)      705 -> 70      (crop too tight / merged)
  len_long    pred has extra digit(s)  705 -> 7051    (padding pulls a neighbor)
  sub1        same length, 1 digit off 705 -> 765     (glyph misread, low-res)
  subN        same length, >1 off      705 -> 168     (heavy misread / wrong field)
  other       everything else

Also reports the confusion by GT length and a few examples per bucket, so the
synthetic generator can match the real failure mix.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


def gt_of(stem):
    d = "".join(c for c in str(stem) if c.isdigit())
    return str(int(d)) if d else ""


def norm(s):
    d = "".join(c for c in str(s) if c.isdigit())
    return str(int(d)) if d else ""


def bucket(gt, pred):
    if not pred:
        return "unread"
    if pred == gt:
        return "correct"
    if len(pred) < len(gt):
        return "len_short"
    if len(pred) > len(gt):
        return "len_long"
    diff = sum(a != b for a, b in zip(gt, pred))
    return "sub1" if diff == 1 else "subN"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="predict_folder_slot* --out 결과 디렉토리")
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.result)
    csvs = list(root.glob("per_frame.csv")) + list(root.glob("*/per_frame.csv"))
    if not csvs:
        raise SystemExit(f"per_frame.csv 없음: {root} (--split-output 로 생성됨)")

    rows = []
    for cf in csvs:
        with cf.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                frame = r.get("frame") or r.get("folder", "")
                rows.append((frame, r.get("channel_number", "")))

    cats = Counter()
    by_gtlen = defaultdict(Counter)
    examples = defaultdict(list)
    n_gt = 0
    for frame, pred in rows:
        gt = gt_of(frame)
        if not gt:
            continue
        n_gt += 1
        b = bucket(gt, norm(pred))
        cats[b] += 1
        by_gtlen[len(gt)][b] += 1
        if b not in ("correct",) and len(examples[b]) < args.examples:
            examples[b].append(f"{frame}: gt={gt} pred={norm(pred) or '(none)'}")

    print(f"총 GT 프레임: {n_gt}\n")
    order = ["correct", "unread", "len_short", "len_long", "sub1", "subN", "other"]
    wrong = n_gt - cats["correct"]
    print(f"{'bucket':<12}{'count':>8}{'%전체':>9}{'%오류중':>9}")
    for b in order:
        c = cats.get(b, 0)
        if c == 0:
            continue
        pw = f"{c/wrong*100:.1f}" if wrong and b not in ("correct",) else "-"
        print(f"{b:<12}{c:>8}{c/n_gt*100:>8.1f}%{pw:>9}")

    print("\n=== GT 자릿수별 오류 분포 ===")
    for L in sorted(by_gtlen):
        t = by_gtlen[L]; tot = sum(t.values())
        seg = "  ".join(f"{k}:{t[k]}" for k in order if t.get(k))
        print(f"  {L}자리 (n={tot}):  {seg}")

    print("\n=== 버킷별 예시 ===")
    for b in order:
        if b in ("correct",) or not examples.get(b):
            continue
        print(f"[{b}]")
        for e in examples[b]:
            print(f"   {e}")

    # 합성 레시피 힌트
    print("\n=== 합성 데이터 레시피 힌트 ===")
    if cats.get("len_short") or cats.get("len_long"):
        print("  · 자리수 오류 많음 → crop 경계/패딩 다양화 + 인접 숫자 붙은 케이스 합성")
    if cats.get("sub1"):
        print("  · 1자리 오독 많음 → 저해상도/블러/유사글리프 강조 합성")
    if cats.get("subN"):
        print("  · 다자리 오독 많음 → 심한 겹침/클러터, 또는 일부는 오선택(②)일 수 있음 → 이미지 확인")


if __name__ == "__main__":
    main()
