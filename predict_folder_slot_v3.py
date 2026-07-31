#!/usr/bin/env python3
"""predict_folder_slot_v3 — v3 드라이버.

full OCR → slot_v3.rolling_analyze(롤링 FIFO + 박스분할 + 강한 선택) → per_frame.csv + 정확도.

A: 숫자+텍스트 박스 분할 / B: 높이·2곳일치 중심 선택 / C: 롤링 큐(직전 N개) — 모두 slot_v3에.
on-device 배포와 동일하게 --window N(직전 N프레임)만 사용. det는 순정, rec는 파인튜닝.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import predict_folder as P
import slot_v3 as V3


def _safe(name, root):
    return name.replace("/", "__").replace(" ", "_") or root.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symlink", action="store_true")
    ap.add_argument("--gt-from-filename", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--window", type=int, default=5, help="롤링 FIFO 크기(직전 N프레임). on-device")
    ap.add_argument("--by-height", action="store_true",
                    help="위치 대신 높이+평행이동축으로 클러스터링(채널박스가 이동하는 UI)")
    ap.add_argument("--band", type=float, default=0.05, help="같은 행/열(또는 위치) 허용 반경")
    ap.add_argument("--rec-model-dir", default="models/full_image_ocr/en_PP-OCRv4_mobile_rec_ft")
    ap.add_argument("--keep-staged", action="store_true")
    ap.add_argument("--no-qualitative", action="store_true")
    ap.add_argument("--samples-per-folder", type=int, default=20)
    args = ap.parse_args()

    cfg = P.load_config()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/opt/conda/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    PY, SRC = cfg["python"], cfg["pipeline_src"]

    flat = out / "images"
    index, meta, seqs = P.collect(root, flat, args.symlink)
    if not index:
        raise SystemExit(f"이미지 없음: {root}")
    print(f"images: {len(index)}  folders: {len(seqs)}", flush=True)

    # === full OCR (det 순정 + rec 파인튜닝) ===
    ocr_cmd = [PY, f"{SRC}/run_paddleocr_export.py", "--images", flat, "--out", out / "full_ocr.json",
               "--use-gpu", "--ocr-version", "PP-OCRv4", "--text-detection-model-name",
               "PP-OCRv4_mobile_det", "--text-recognition-model-name", "en_PP-OCRv4_mobile_rec",
               "--progress-every", 200]
    rd = str(args.rec_model_dir or "").strip()
    if rd.lower() not in ("", "none"):
        rdp = Path(rd)
        if not rdp.is_absolute():
            rdp = Path(__file__).resolve().parent / rdp
        if (rdp / "inference.pdiparams").exists():
            ocr_cmd += ["--text-recognition-model-dir", str(rdp)]
            print(f"[v3] 파인튜닝 rec: {rdp}", flush=True)
    if P.sh(ocr_cmd, env) != 0:
        raise SystemExit("[v3] full OCR 실패")
    by_id = {im["image_id"]: im for im in json.loads((out / "full_ocr.json").read_text()).get("images", [])}

    # === v3 롤링 선택 ===
    rows, report, pred_by_uid = [], [], {}
    for g, uids in sorted(seqs.items()):
        ok = [u for u in sorted(uids) if u in by_id]
        frames = [by_id[u] for u in ok]
        if not frames:
            continue
        primary = V3.rolling_analyze(frames, ok, window=args.window, by_height=args.by_height,
                                     band=args.band, conf_thr=args.min_conf)
        pf = primary["per_frame"] if primary else {}
        report.append({"folder": g, "channel_field_box": primary["box"] if primary else None,
                       "distinct_values": primary["distinct"] if primary else 0,
                       "coverage": round(len(pf) / max(1, len(ok)), 3)})
        for uid, v in sorted(pf.items()):
            rows.append({"folder": g, "frame": meta.get(uid, uid), "channel_number": v})
            pred_by_uid[uid] = v
        print(f"  {g:<40} box={report[-1]['channel_field_box']} cov={report[-1]['coverage']}", flush=True)

    with (out / "per_frame.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "frame", "channel_number"])
        w.writeheader(); w.writerows(rows)
    (out / "profile_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    # === 정확도 ===
    if args.gt_from_filename:
        norm = lambda s: str(int(s)) if str(s).isdigit() else str(s)
        per = {}
        for g, uids in seqs.items():
            for uid in uids:
                if uid not in by_id:
                    continue
                gt = P.gt_from_name(meta.get(uid, uid))
                if not gt:
                    continue
                t = per.setdefault(g, [0, 0, 0]); t[0] += 1
                pr = P._dg(pred_by_uid.get(uid, ""))
                if pr:
                    t[1] += 1
                    if norm(pr) == norm(gt):
                        t[2] += 1
        tot = sum(v[0] for v in per.values()); rdn = sum(v[1] for v in per.values()); co = sum(v[2] for v in per.values())
        pc = lambda a, b: round(a / b * 100, 1) if b else 0
        print("\n=== 폴더별 정확도 (v3) ===")
        print(f"  {'folder':<40}{'e2e%':>7}{'cov%':>7}{'읽었을때%':>10}{'correct/total':>16}")
        for g in sorted(per):
            a, b, c = per[g]
            print(f"  {g:<40}{pc(c,a):>7}{pc(b,a):>7}{pc(c,b):>10}{f'{c}/{a}':>16}")
        print(f"  {'-'*40}\n  {'전체':<40}{pc(co,tot):>7}{pc(rdn,tot):>7}{pc(co,rdn):>10}{f'{co}/{tot}':>16}")

    print(f"\n결과: {out}/per_frame.csv, profile_report.json ({len(rows)}개)", flush=True)
    if not args.keep_staged:
        shutil.rmtree(flat, ignore_errors=True)


if __name__ == "__main__":
    main()
