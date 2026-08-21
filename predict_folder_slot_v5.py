#!/usr/bin/env python3
"""predict_folder_slot_v5 — v3 드라이버와 같은 인자 형식, 단 full_ocr.json 없이 스트리밍.

v3/v4 드라이버: 이미지 → (배치)full OCR → full_ocr.json 저장 → 선택.
v5(온디바이스형): 폴더(시퀀스)마다
   ① 앞 --window 프레임만 full OCR(det+rec) → slot_v3로 채널 ROI 확정
   ② 이후 프레임: det 생략, 그 ROI만 crop→rec-only 로 직접 읽음
제거한 지연요소: full_ocr.json 저장/로드, 매 프레임 det, slot_v4 픽셀 샘플링.
모델은 프로세스당 1번 로드. 출력은 v3와 동일(per_frame.csv + 폴더별 정확도).

사용(예: v3와 동일 형식):
  python predict_folder_slot_v5.py --root <이미지루트> --out <출력> --gt-from-filename --window 5
"""
from __future__ import annotations
import argparse, csv, os, re, sys, tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
try:
    import torch  # noqa
except Exception:
    pass
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import predict_folder as P
import slot_v3 as V3
from ocr_candidate_extractor import create_paddle_ocr, extract_candidates_from_image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def gt_of(name):
    m = re.match(r"(?i)\s*ch[\s_]*0*(\d+)", str(name))
    return m.group(1) if m else P.gt_from_name(name)


def best_digit(t):
    r = [x for x in re.findall(r"\d+", str(t)) if 1 <= len(x) <= 5]
    return max(r, key=len) if r else re.sub(r"\D", "", str(t))[:5]


def load_rec(model_dir):
    from paddleocr import TextRecognition
    if model_dir and Path(model_dir).exists():
        name = "en_PP-OCRv4_mobile_rec"
        y = Path(model_dir) / "inference.yml"
        if y.exists():
            try:
                import yaml
                name = yaml.safe_load(open(y)).get("Global", {}).get("model_name") or name
            except Exception:
                pass
        return TextRecognition(model_name=name, model_dir=str(model_dir))
    return TextRecognition(model_name="en_PP-OCRv4_mobile_rec")


def rec_roi(rec, im, box, tmp, pad=0.2, min_h=120):
    W, H = im.size
    x1, y1, x2, y2 = box
    pw = (x2 - x1) * pad; ph = (y2 - y1) * pad
    cx1, cy1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
    cx2, cy2 = min(W, int(x2 + pw)), min(H, int(y2 + ph))
    if cx2 - cx1 < 3 or cy2 - cy1 < 3:
        return ""
    crop = im.crop((cx1, cy1, cx2, cy2))
    u = max(1.0, min_h / max(1, crop.height))
    if u > 1.0:
        crop = crop.resize((max(1, int(crop.width * u)), max(1, int(crop.height * u))))
    cp = tmp / "c.png"; crop.convert("RGB").save(cp)
    r = rec.predict(str(cp))
    return best_digit(r[0].get("rec_text", "") if r else "")


def resolve_dir(dd):
    dd = str(dd or "").strip()
    if dd.lower() in ("", "none"):
        return None
    p = Path(dd)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return str(p) if (p / "inference.pdiparams").exists() else None


def stream_seq(paths, ocr, rec, tmp, window, min_conf, by_height, band):
    """한 시퀀스 스트리밍. 반환 {stem: 채널값}."""
    frames, ids, out = [], [], {}
    box = None
    for p in paths:
        im = Image.open(p).convert("RGB"); W, H = im.size
        stem = p.stem
        if box is None:                                   # phase A: full OCR
            cands = extract_candidates_from_image(Path(p), ocr=ocr)
            frames.append({"image_id": stem, "image_width": W, "image_height": H,
                           "candidates": [c.to_dict() for c in cands]})
            ids.append(stem); out[stem] = ""
            if len(frames) >= window:
                r = V3.rolling_analyze(frames, ids, window=window, by_height=by_height,
                                       band=band, conf_thr=min_conf)
                box = r["box"] if r else None
                pf = r["per_frame"] if r else {}
                for sid in ids:
                    out[sid] = re.sub(r"\D", "", str(pf.get(sid, "")))
                if box is None:                           # 클러스터 실패 → 폴백 계속 full
                    frames, ids = [], []
        else:                                             # phase B: rec-only
            out[stem] = rec_roi(rec, im, box, tmp)
    return out, box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symlink", action="store_true", help="(호환용, v5는 staging 안 함 → 무시)")
    ap.add_argument("--gt-from-filename", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--window", type=int, default=5,
                    help="phase A(full OCR로 ROI 확정) 프레임 수. 차면 이후 rec-only")
    ap.add_argument("--by-height", action="store_true")
    ap.add_argument("--band", type=float, default=0.05)
    ap.add_argument("--rec-model-dir", default="models/full_image_ocr/en_PP-OCRv4_mobile_rec_ft")
    ap.add_argument("--det-model-dir", default="")
    ap.add_argument("--keep-staged", action="store_true", help="(호환용, 무시)")
    ap.add_argument("--no-qualitative", action="store_true", help="(호환용, 무시)")
    args = ap.parse_args()

    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 폴더(시퀀스) 그룹핑 — staging 없이 원본 경로 그대로 (v3 collect와 동일 그룹 규칙)
    imgs = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    seqs = defaultdict(list)
    for p in imgs:
        g = str(p.parent.relative_to(root)) or "(root)"
        seqs[g].append(p)
    print(f"images: {len(imgs)}  folders: {len(seqs)}  (json 없음, 스트리밍)", flush=True)

    rec_dir = resolve_dir(args.rec_model_dir)
    det_dir = resolve_dir(args.det_model_dir)
    if rec_dir:
        print(f"[v5] 파인튜닝 rec: {rec_dir}", flush=True)
    if det_dir:
        print(f"[v5] 파인튜닝 det: {det_dir}", flush=True)
    ocr = create_paddle_ocr(lang="en", use_gpu=True, ocr_version="PP-OCRv4",
                            text_detection_model_name="PP-OCRv4_mobile_det",
                            text_recognition_model_name="en_PP-OCRv4_mobile_rec",
                            text_recognition_model_dir=rec_dir, text_detection_model_dir=det_dir)
    rec = load_rec(rec_dir)
    tmp = Path(tempfile.mkdtemp(prefix="v5drv_"))

    rows, pred = [], {}
    for g, ps in sorted(seqs.items()):
        ps = sorted(ps)
        o, box = stream_seq(ps, ocr, rec, tmp, args.window, args.min_conf, args.by_height, args.band)
        for p in ps:
            v = o.get(p.stem, "")
            rows.append({"folder": g, "frame": p.stem, "channel_number": v})
            pred[(g, p.stem)] = v
        print(f"  {g:<40} box={box and [round(x,1) for x in box]}", flush=True)

    with (out / "per_frame.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "frame", "channel_number"])
        w.writeheader(); w.writerows(rows)

    if args.gt_from_filename:
        nrm = lambda s: str(int(s)) if str(s).isdigit() else str(s)
        per = {}
        for g, ps in seqs.items():
            for p in sorted(ps):
                gt = gt_of(p.stem)
                if not gt:
                    continue
                t = per.setdefault(g, [0, 0, 0]); t[0] += 1
                pr = re.sub(r"\D", "", str(pred.get((g, p.stem), "")))
                if pr:
                    t[1] += 1
                    if nrm(pr) == nrm(gt):
                        t[2] += 1
        tot = [sum(v[i] for v in per.values()) for i in range(3)]
        pc = lambda a, b: round(a / b * 100, 1) if b else 0
        print("\n=== 폴더별 정확도 (v5) ===")
        print(f"  {'folder':<40}{'e2e%':>7}{'cov%':>7}{'읽었을때%':>10}{'correct/total':>16}")
        for g in sorted(per):
            a, b, c = per[g]
            print(f"  {g:<40}{pc(c,a):>7}{pc(b,a):>7}{pc(c,b):>10}{f'{c}/{a}':>16}")
        print(f"  {'-'*40}\n  {'전체':<40}{pc(tot[2],tot[0]):>7}{pc(tot[1],tot[0]):>7}{pc(tot[2],tot[1]):>10}{f'{tot[2]}/{tot[0]}':>16}")

    print(f"\n결과: {out}/per_frame.csv ({len(rows)}개)", flush=True)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
