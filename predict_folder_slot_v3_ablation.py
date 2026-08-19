#!/usr/bin/env python3
"""predict_folder_slot_v3_ablation — slot_v3의 A/B/C 3스텝을 누적 ablation.

slot_v3 docstring의 세 스텝:
    step1 (A) 숫자+텍스트 혼합 박스 분할 + 시간/날짜 배제 + 자릿수(1~5) 제한
    step2 (B) 프레임 내 강한 선택 기준 (순수도 / 2곳 일치 / aspect 게이트)
    step3 (C) 롤링 슬롯 클러스터링 + lifetime 점수 + 위치 그룹 투표

4가지 구성을 같은 OCR 결과 위에서 비교한다 (OCR은 한 번만 돌린다):

    S0  아무것도 안 함   — 후보 텍스트의 첫 숫자열, conf 최고를 선택
    S1  step1만          — 분할/필터 후, conf 최고를 선택
    S2  step1+2          — 분할/필터 후, 프레임내 점수 최고를 선택
    S3  step1+2+3        — 현재 slot_v3 전체 (= predict_folder_slot_v3)

step3가 없으면 채널 '위치'가 없어서 고를 기준이 없으므로, S0/S1/S2는 지시대로
**모델의 숫자 confidence가 가장 높은 후보**를 고른다(S2는 거기에 프레임내 점수를 곱함).

사용:
  # OCR부터 새로 (v3와 동일 인자)
  python predict_folder_slot_v3_ablation.py --root <이미지폴더> --out results/ablation \
      --det-model-dir models/full_image_ocr/det_overlay_frozen_v1 \
      --rec-model-dir models/full_image_ocr/rec_overlay_frozen_v1

  # 이미 끝난 run의 full_ocr.json 재사용 (OCR 생략, 수 초 만에 끝남)
  python predict_folder_slot_v3_ablation.py --from-result results/ft_run --out results/ablation
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import predict_folder as P
import slot_v3 as V3
from predict_folder_slot_v3 import gt_of
from predict_folder_slot_crop_v3 import run_ocr

DIGIT_RUN = re.compile(r"\d+")
STEPS = [("S0", "아무것도 안 함"), ("S1", "step1 (분할·필터)"),
         ("S2", "step1+2 (+프레임내 점수)"), ("S3", "step1+2+3 (현재 v3)")]


# ───────────────────── 스텝별 프레임 예측 ─────────────────────

def pick_s0(im, conf_thr):
    """룰 없음: 후보 텍스트에서 첫 숫자열만 뽑고 conf 최고. 분할·시간배제·자릿수제한 없음."""
    best = None
    for c in im.get("candidates", []):
        cf = float(c.get("ocr_conf", 0.0) or 0.0)
        if cf < conf_thr:
            continue
        m = DIGIT_RUN.search(str(c.get("text", "")))
        if not m:
            continue
        if best is None or cf > best[0]:
            best = (cf, V3._cnorm(m.group()))
    return best[1] if best else ""


def pick_s1(im, conf_thr):
    """step1만: split_candidate(혼합박스 분할 + 시간/날짜 배제 + 1~5자리) 후 conf 최고."""
    best = None
    for c in im.get("candidates", []):
        for sc in V3.split_candidate(c):
            cf = sc["ocr_conf"]
            if cf < conf_thr:
                continue
            if best is None or cf > best[0]:
                best = (cf, V3._cnorm(sc["text"]))
    return best[1] if best else ""


def pick_s2(im, conf_thr):
    """step1+2: 프레임 내에서만 계산 가능한 B의 신호로 점수화.

    score = conf × (0.9 + 0.1×purity) × (0.2 if aspect 벗어남) × (1.6 if 2곳 일치)
    (값 다양성·높이 일관성은 여러 프레임이 필요하므로 step3 소관)
    """
    cands = V3.preprocess_frame(im, conf_thr)
    if not cands:
        return ""
    agreed = V3.within_frame_agreed(cands)
    best = None
    for c in cands:
        b = c["box"]
        w, h = b[2] - b[0], b[3] - b[1]
        aspect = w / max(1e-6, h)
        sc = c["conf"] * (0.9 + 0.1 * c["purity"])
        if not (0.2 <= aspect <= 8.0):
            sc *= 0.2
        if c["value"] in agreed:
            sc *= 1.6
        if best is None or sc > best[0]:
            best = (sc, c["value"])
    return best[1] if best else ""


def pick_s3(frames, ids, args):
    """step1+2+3: 현재 slot_v3 전체 (폴더 단위)."""
    pr = V3.rolling_analyze(frames, ids, window=args.window, by_height=args.by_height,
                            band=args.band, conf_thr=args.min_conf)
    return (pr["per_frame"] if pr else {}), (pr["box"] if pr else None)


# ───────────────────── uid → 폴더 복원 ─────────────────────

def folders_from_report(res):
    p = res / "profile_report.json"
    if not p.exists():
        return []
    rep = json.loads(p.read_text())
    if isinstance(rep, dict):
        rep = rep.get("folders", [])
    return [f["folder"] for f in rep]


def group_uids(by_id, folders):
    """uid를 폴더로 되돌린다. uid = '<folder>__<stem>' (folder의 '/'는 '__'로 치환됨).
    가장 긴 접두사 우선으로 매칭한다."""
    pref = sorted(((g.replace("/", "__").replace(" ", "_") + "__", g) for g in folders),
                  key=lambda t: -len(t[0]))
    seqs, meta, unmatched = defaultdict(list), {}, 0
    for uid in sorted(by_id):
        hit = None
        for p, g in pref:
            if uid.startswith(p):
                hit = (g, uid[len(p):])
                break
        if hit is None:
            unmatched += 1
            hit = ("(unknown)", uid)
        seqs[hit[0]].append(uid)
        meta[uid] = hit[1]
    if unmatched:
        print(f"[주의] 폴더를 못 찾은 uid {unmatched}개 → '(unknown)' 으로 묶음", flush=True)
    return seqs, meta


# ───────────────────── 집계·출력 ─────────────────────

def tally(seqs, meta, preds, by_id):
    """구성별 폴더 정확도 집계. 반환 per[folder] = [total, read, correct]."""
    norm = lambda s: str(int(s)) if str(s).isdigit() else str(s)
    per = {}
    for g, uids in seqs.items():
        for uid in uids:
            if uid not in by_id:
                continue
            gt = gt_of(meta.get(uid, uid))
            if not gt:
                continue
            t = per.setdefault(g, [0, 0, 0])
            t[0] += 1
            pr = P._dg(preds.get(uid, ""))
            if pr:
                t[1] += 1
                if norm(pr) == norm(gt):
                    t[2] += 1
    return per


def print_table(title, per, width=52):
    pc = lambda a, b: round(a / b * 100, 1) if b else 0.0
    print(f"\n=== {title} ===")
    print(f"  {'folder':<{width}}{'e2e%':>7}{'cov%':>7}{'읽었을때%':>10}{'correct/total':>16}")
    for g in sorted(per):
        a, b, c = per[g]
        print(f"  {g:<{width}}{pc(c,a):>7}{pc(b,a):>7}{pc(c,b):>10}{f'{c}/{a}':>16}")
    tot = sum(v[0] for v in per.values())
    rdn = sum(v[1] for v in per.values())
    co = sum(v[2] for v in per.values())
    print(f"  {'-'*width}")
    print(f"  {'전체':<{width}}{pc(co,tot):>7}{pc(rdn,tot):>7}{pc(co,rdn):>10}{f'{co}/{tot}':>16}")
    return tot, rdn, co


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", help="이미지 루트 (OCR부터 새로 실행)")
    src.add_argument("--from-result", help="끝난 run 디렉토리 (full_ocr.json 재사용, OCR 생략)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--symlink", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--by-height", action="store_true")
    ap.add_argument("--band", type=float, default=0.05)
    ap.add_argument("--rec-model-dir", default="models/full_image_ocr/en_PP-OCRv4_mobile_rec_ft")
    ap.add_argument("--det-model-dir", default="")
    ap.add_argument("--keep-staged", action="store_true")
    ap.add_argument("--width", type=int, default=52, help="폴더명 열 너비")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.from_result:
        res = Path(args.from_result).resolve()
        by_id = {im["image_id"]: im
                 for im in json.loads((res / "full_ocr.json").read_text()).get("images", [])}
        umap = res / "uid_map.json"
        if umap.exists():
            m = json.loads(umap.read_text())
            seqs = defaultdict(list)
            meta = {}
            for uid, (g, stem) in m.items():
                if uid in by_id:
                    seqs[g].append(uid); meta[uid] = stem
            for g in seqs:
                seqs[g].sort()
        else:
            seqs, meta = group_uids(by_id, folders_from_report(res))
        print(f"[reuse] {res}/full_ocr.json  프레임 {len(by_id)}  폴더 {len(seqs)}", flush=True)
    else:
        cfg = P.load_config()
        root = Path(args.root).resolve()
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = "/opt/conda/lib:" + env.get("LD_LIBRARY_PATH", "")
        env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        flat = out / "images"
        index, meta, seqs = P.collect(root, flat, args.symlink)
        if not index:
            raise SystemExit(f"이미지 없음: {root}")
        print(f"images: {len(index)}  folders: {len(seqs)}", flush=True)
        (out / "uid_map.json").write_text(json.dumps(
            {u: [index[u], meta[u]] for u in index}, ensure_ascii=False))
        run_ocr(cfg["python"], cfg["pipeline_src"], flat, out / "full_ocr.json",
                args, env, "ablation")
        by_id = {im["image_id"]: im
                 for im in json.loads((out / "full_ocr.json").read_text()).get("images", [])}

    # ── 스텝별 예측 ──
    preds = {k: {} for k, _ in STEPS}
    boxes_s3 = {}
    for g, uids in sorted(seqs.items()):
        ok = [u for u in sorted(uids) if u in by_id]
        if not ok:
            continue
        for uid in ok:
            im = by_id[uid]
            preds["S0"][uid] = pick_s0(im, args.min_conf)
            preds["S1"][uid] = pick_s1(im, args.min_conf)
            preds["S2"][uid] = pick_s2(im, args.min_conf)
        pf, box = pick_s3([by_id[u] for u in ok], ok, args)
        boxes_s3[g] = box
        preds["S3"].update(pf)

    # ── 집계 + 출력 ──
    results, summary = {}, {}
    for key, label in STEPS:
        per = tally(seqs, meta, preds[key], by_id)
        results[key] = per
        summary[key] = print_table(f"폴더별 정확도 ({key}: {label})", per, args.width)

    pc = lambda a, b: round(a / b * 100, 1) if b else 0.0
    print("\n\n=== 스텝별 요약 (전체) ===")
    print(f"  {'구성':<26}{'e2e%':>7}{'cov%':>7}{'읽었을때%':>10}{'correct/total':>16}{'Δe2e':>8}")
    prev = None
    for key, label in STEPS:
        tot, rdn, co = summary[key]
        e2e = pc(co, tot)
        d = "" if prev is None else f"{e2e - prev:+.1f}"
        print(f"  {key+' '+label:<26}{e2e:>7}{pc(rdn,tot):>7}{pc(co,rdn):>10}"
              f"{f'{co}/{tot}':>16}{d:>8}")
        prev = e2e

    # ── 저장 ──
    with (out / "ablation_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["folder", "step", "e2e_pct", "cov_pct", "read_pct", "correct", "total"])
        for key, _ in STEPS:
            for g, (a, b, c) in sorted(results[key].items()):
                w.writerow([g, key, pc(c, a), pc(b, a), pc(c, b), c, a])
        for key, _ in STEPS:
            tot, rdn, co = summary[key]
            w.writerow(["(전체)", key, pc(co, tot), pc(rdn, tot), pc(co, rdn), co, tot])

    with (out / "ablation_per_frame.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["folder", "frame", "gt", "S0", "S1", "S2", "S3"])
        for g, uids in sorted(seqs.items()):
            for uid in sorted(uids):
                if uid not in by_id:
                    continue
                w.writerow([g, meta.get(uid, uid), gt_of(meta.get(uid, uid)) or ""]
                           + [preds[k].get(uid, "") for k, _ in STEPS])

    print(f"\n결과: {out}/ablation_summary.csv, ablation_per_frame.csv", flush=True)
    if args.root and not args.keep_staged:
        shutil.rmtree(out / "images", ignore_errors=True)


if __name__ == "__main__":
    main()
