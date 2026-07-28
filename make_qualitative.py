#!/usr/bin/env python3
"""Draw qualitative results from a finished predict_folder.py run — NO re-inference.

Reads <result>/predictions.json (produced by the pipeline) and draws the selected
channel-number box + predicted value on each frame, saving into <out>, organised
by the original folder. Fast (cv2 only, no GPU).

Ground truth (optional):
  --labels  path to a JSON mapping  { "<folder-relpath>": "<gt channel number>" }
            e.g. {"korea/2017-field-1080p/20161208_skb": "210", ...}
  When given, frames are split into success/ and fail/ and the filename carries
  gt vs pred. Without it, every frame is saved under all/ with the prediction.

Usage:
  python make_qualitative.py --result ./result --out ./result/qualitative
  python make_qualitative.py --result ./result --out ./result/qualitative --labels gt.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def digits(x) -> str:
    return "".join(c for c in str(x or "") if c.isdigit())


def selected_bbox(item):
    """Pull the chosen channel-number box (xyxy) out of a prediction record."""
    sel = (item.get("temporal_selection") or {}).get("best_candidate")
    for src in (sel, item):
        if isinstance(src, dict):
            b = src.get("bbox_xyxy") or src.get("bbox")
            if b and len(b) == 4:
                return [float(v) for v in b]
    return None


def find_image(images_dir: Path, uid: str):
    for ext in IMG_EXTS:
        p = images_dir / f"{uid}{ext}"
        if p.exists():
            return p
    return None


def draw(img_path: Path, bbox, pred: str, gt: str | None, dst: Path):
    im = cv2.imread(str(img_path))
    if im is None:
        return False
    ok = gt is not None and digits(pred) == digits(gt) and gt != ""
    if bbox is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        color = (0, 200, 0) if (gt is None or ok) else (0, 0, 255)  # BGR
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
        label = f"pred={pred or '-'}" + (f"  gt={gt}" if gt is not None else "")
        y = max(0, y1 - 8)
        cv2.putText(im, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), im)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="predict_folder.py --out dir")
    ap.add_argument("--out", required=True, help="where to write qualitative images")
    ap.add_argument("--labels", default=None, help="optional folder->gt JSON")
    ap.add_argument("--numeric-equiv", action="store_true",
                    help="ignore leading zeros when scoring (041 == 41)")
    args = ap.parse_args()
    result, out = Path(args.result).resolve(), Path(args.out).resolve()
    images_dir = result / "images"
    doc = json.loads((result / "predictions.json").read_text())

    gt_map = {}
    if args.labels:
        gt_map = json.loads(Path(args.labels).read_text())

    # uid -> folder relpath.  predict_folder builds uid = "<folder>__<stem>" with
    # "/"->"__"; recover the folder from per_frame.csv (has folder + frame) if present.
    uid_folder = {}
    pf = result / "per_frame.csv"
    if pf.exists():
        import csv
        with pf.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                # reconstruct uid the way predict_folder did
                uid = f"{r['folder']}__{r['frame']}".replace("/", "__").replace(" ", "_")
                uid_folder[uid] = r["folder"]

    def norm(s):
        return str(int(s)) if (args.numeric_equiv and digits(s)) else digits(s)

    n_all = n_ok = n_fail = n_noimg = 0
    per_folder = defaultdict(lambda: [0, 0])   # folder -> [ok, total]
    # group prediction records by folder so we can log one line per folder
    by_folder = defaultdict(list)
    for item in doc["images"]:
        uid = str(item.get("image_id", ""))
        folder = uid_folder.get(uid) or uid.rsplit("__", 1)[0]
        by_folder[folder].append((uid, item))

    for folder in sorted(by_folder):
        f_ok = f_tot = 0
        for uid, item in by_folder[folder]:
            pred = digits(item.get("predicted_channel_number"))
            gt = gt_map.get(folder) if gt_map else None
            img = find_image(images_dir, uid)
            if img is None:
                n_noimg += 1
                continue
            bbox = selected_bbox(item)
            if gt is not None:
                ok = gt != "" and norm(pred) == norm(gt)
                sub = "success" if ok else "fail"
                tag = "ok" if ok else "ng"
                name = f"{uid}__gt{gt}__pred{pred or 'none'}__{tag}.jpg"
                dst = out / sub / folder / name
                per_folder[folder][0] += int(ok); per_folder[folder][1] += 1
                f_ok += int(ok); f_tot += 1
                n_ok += int(ok); n_fail += int(not ok)
            else:
                dst = out / "all" / folder / f"{uid}__pred{pred or 'none'}.jpg"
            if draw(img, bbox, pred, gt, dst):
                n_all += 1
        if gt_map:
            gt = gt_map.get(folder)
            print(f"  [{folder}] gt={gt}  정확도 {round(f_ok/f_tot*100,1) if f_tot else '-'}%  "
                  f"({f_ok}/{f_tot})", flush=True)
        else:
            # no GT: show the majority-vote prediction for the folder
            preds = [digits(it.get("predicted_channel_number")) for _, it in by_folder[folder]]
            preds = [p for p in preds if p]
            best = Counter(preds).most_common(1)[0][0] if preds else "-"
            print(f"  [{folder}] 예측 채널번호(다수결)={best}  ({len(by_folder[folder])}프레임)", flush=True)

    print(f"\n정성이미지 저장: {n_all}장 -> {out}")
    if n_noimg:
        print(f"  (이미지 못 찾음 {n_noimg}장 — 원본 심링크 확인)")
    if gt_map:
        acc = out / "per_folder_accuracy.csv"
        import csv
        with acc.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(["folder", "accuracy(%)", "n_frames", "n_correct"])
            for fld, (ok, tot) in sorted(per_folder.items()):
                w.writerow([fld, round(ok / tot * 100, 2) if tot else "", tot, ok])
        print("\n" + "=" * 60)
        print("=== 폴더(UI)별 채널번호 정확도 ===")
        for fld, (ok, tot) in sorted(per_folder.items()):
            mark = "✓" if tot and ok == tot else ("✗" if tot and ok == 0 else "~")
            print(f"  [{mark}] {fld}: {round(ok/tot*100,1) if tot else '-'}%  ({ok}/{tot})")
        tot_all = n_ok + n_fail
        print("-" * 60)
        print(f"  전체 정확도: {round(n_ok/tot_all*100,2) if tot_all else '-'}%  "
              f"(맞음 {n_ok} / 틀림 {n_fail} / 총 {tot_all}프레임)")
        print(f"  폴더 {len(per_folder)}개  |  success/ · fail/ 로 분리 저장")
        print("=" * 60)
        print(f"  폴더별 정확도 CSV -> {acc}")
    else:
        print("  (정답 없음 → all/ 아래 전부 저장. --labels 주면 success/fail 분리 + 정확도 로그)")


if __name__ == "__main__":
    main()
