#!/usr/bin/env python3
"""ablation_folder — predict_folder_slot_v3.py 와 '같은 인자 형식'의 ablation study 드라이버.

--root 이미지폴더만 주면:
  ① full OCR(det+rec)을 딱 한 번 실행 → full_ocr.json  (모든 config가 이 '동일한 det 결과' 재사용)
  ② 각 점수-항 config로 선택만 반복 → 각 항이 성능을 얼마나 올리는지 측정
     · 단독 기여: v3 + 그 항 하나만  (− v3)
     · 필요성   : v4 − 그 항        (v4 하락폭)
전 프레임 선택(v5 같은 프레임 스킵 없음) — ablation study 전용.

사용(예: v3 드라이버와 동일):
  python ablation_folder.py --root <이미지루트> --out <출력> --gt-from-filename --window 24
  # 파인튜닝 det로 평가: --det-model-dir models/full_image_ocr/det_overlay_frozen_v1
"""
from __future__ import annotations
import argparse, csv, json, os, re, shutil
from collections import defaultdict
from pathlib import Path

import predict_folder as P
import slot_v4 as V4
from ablation_general import configs      # 항별 config 재사용


def gt_of(name):
    m = re.match(r"(?i)\s*ch[\s_]*0*(\d+)", str(name))
    return m.group(1) if m else P.gt_from_name(name)


def _resolve(base, dd):
    dd = str(dd or "").strip()
    if dd.lower() in ("", "none"):
        return None
    p = Path(dd)
    if not p.is_absolute():
        p = base / p
    return p if (p / "inference.pdiparams").exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symlink", action="store_true")
    ap.add_argument("--gt-from-filename", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--window", type=int, default=24, help="롤링 통계 윈도우(전 프레임 선택, 스킵 아님)")
    ap.add_argument("--by-height", action="store_true")
    ap.add_argument("--band", type=float, default=0.05)
    ap.add_argument("--rec-model-dir", default="models/full_image_ocr/en_PP-OCRv4_mobile_rec_ft")
    ap.add_argument("--det-model-dir", default="")
    ap.add_argument("--keep-staged", action="store_true")
    ap.add_argument("--reuse-ocr", default="", help="이미 있는 full_ocr.json 경로(주면 OCR 생략)")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    cfg = P.load_config()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/opt/conda/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    PY, SRC = cfg["python"], cfg["pipeline_src"]

    flat = out / "images"
    _, meta, seqs = P.collect(root, flat, args.symlink)
    print(f"images staged, folders: {len(seqs)}", flush=True)

    # ── ① full OCR 한 번 (모든 config 공용, '동일한 det 결과') ──
    ocr_json = Path(args.reuse_ocr) if args.reuse_ocr else (out / "full_ocr.json")
    if not args.reuse_ocr:
        ocr_cmd = [PY, f"{SRC}/run_paddleocr_export.py", "--images", flat, "--out", ocr_json,
                   "--use-gpu", "--ocr-version", "PP-OCRv4",
                   "--text-detection-model-name", "PP-OCRv4_mobile_det",
                   "--text-recognition-model-name", "en_PP-OCRv4_mobile_rec", "--progress-every", 200]
        rec = _resolve(here, args.rec_model_dir)
        if rec:
            ocr_cmd += ["--text-recognition-model-dir", str(rec)]; print(f"[ocr] rec: {rec}", flush=True)
        det = _resolve(here, args.det_model_dir)
        if det:
            ocr_cmd += ["--text-detection-model-dir", str(det)]; print(f"[ocr] det: {det}", flush=True)
        if P.sh(ocr_cmd, env) != 0:
            raise SystemExit("full OCR 실패")
    by_id = {im["image_id"]: im for im in json.loads(Path(ocr_json).read_text()).get("images", [])}

    # ── ② 각 config로 선택 반복 (OCR 재사용) ──
    base = dict(size_weight=0.4, bg_weight=0.3, contrast_weight=0.2, global_weight=0.5, sat_weight=0.0)
    CFG = configs(base)
    nrm = lambda s: str(int(s)) if str(s).isdigit() else str(s)

    def run(weights):
        per, tot = {}, [0, 0]
        for g, uids in sorted(seqs.items()):
            ok = [u for u in sorted(uids) if u in by_id]
            frames = [by_id[u] for u in ok]
            if not frames:
                continue
            r = V4.rolling_analyze(frames, ok, window=args.window,
                                   by_height=args.by_height, band=args.band, conf_thr=args.min_conf, **weights)
            pf = r["per_frame"] if r else {}
            c = t = 0
            for u in ok:
                gt = gt_of(meta.get(u, u)) if args.gt_from_filename else None
                if not gt:
                    continue
                t += 1
                if nrm(re.sub(r"\D", "", str(pf.get(u, "")))) == nrm(gt):
                    c += 1
            per[g] = (c, t); tot[0] += c; tot[1] += t
        return per, tot

    print("=" * 64, flush=True)
    res = {}
    for name, w in CFG:
        per, tot = run(w); res[name] = (per, tot)
        print(f"  {name:<26} {tot[0]/max(1,tot[1])*100:5.1f}%  ({tot[0]}/{tot[1]})", flush=True)
    print("=" * 64)

    acc = lambda t: t[0]/max(1, t[1])*100
    v3a = acc(res["v3  (시각항 OFF, 기준선)"][1]); v4a = acc(res["v4  (전부 ON)"][1])
    print(f"기준선 v3 = {v3a:.1f}%   전부켬 v4 = {v4a:.1f}%   (Δ {v4a-v3a:+.1f}%p)\n")
    print("① 단독 기여 (v3 + 그 항 하나만 − v3):")
    for label, key in [("크기", "v3 + 크기"), ("배경하이라이트", "v3 + 배경하이라이트"),
                       ("대비", "v3 + 대비"), ("전역현저성", "v3 + 전역현저성"), ("채도", "v3 + 채도")]:
        print(f"  {label:<12}: {acc(res[key][1])-v3a:+.1f}%p")
    print("② 필요성 (v4 − 그 항 → v4 하락폭):")
    for label, key in [("크기", "v4 − 크기"), ("배경하이라이트", "v4 − 배경하이라이트"),
                       ("대비", "v4 − 대비"), ("전역현저성", "v4 − 전역현저성")]:
        print(f"  {label:<12}: {v4a-acc(res[key][1]):+.1f}%p")

    # 그룹(UI)별 CSV
    uis = sorted(res["v4  (전부 ON)"][0])
    with (out / "ablation_by_group.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["group", "n"] + [n for n, _ in CFG])
        for g in uis:
            n = res["v4  (전부 ON)"][0][g][1]
            w.writerow([g, n] + [round(res[nm][0][g][0]/max(1, res[nm][0][g][1])*100, 1) for nm, _ in CFG])
    print(f"\n그룹별 표 → {out}/ablation_by_group.csv")
    if not args.keep_staged:
        shutil.rmtree(flat, ignore_errors=True)


if __name__ == "__main__":
    main()
