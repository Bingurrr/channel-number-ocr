#!/usr/bin/env python3
"""Folder-based channel-number inference (recursive; handles nested folders and
duplicate frame names like 001.jpg across folders).

Every image folder that contains images is treated as one group (= one UI /
capture). Frames in a group are run as a temporal sequence (channel position is
fixed per UI), so accumulation across frames stabilises detection.

    ROOT/.../<any depth>/<folder>/  001.jpg  002.jpg ...

Remote-mount safety (rclone / NAS):
  Frames are COPIED to local disk ONCE (sequentially), then every pipeline step
  reads the local copy. Symlinking to a remote mount instead makes each step
  re-read the same file over the network 3-4x, which can overload an rclone
  mount (especially with several users) and hang it. Use --symlink only for
  data that already sits on fast local disk.

Outputs (under --out):
  per_folder.csv   — one channel number per folder (majority vote over frames)
  per_frame.csv    — per-frame prediction
  qualitative/<folder>/          — up to N sample frames with the channel box drawn
  qualitative/<folder>/_failures — ALL failure frames (no read, or != folder vote)

Model paths come from config.json (package-relative by default).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config():
    c = json.loads((HERE / "config.json").read_text())
    for k in ("pipeline_src", "detector", "numeric_ocr", "selector_dir",
              "recheck_padded", "full_image_ocr"):
        v = c.get(k)
        if v and not str(v).startswith("/"):
            c[k] = str((HERE / v).resolve())
    return c


def collect(root: Path, flat: Path, symlink: bool):
    """Recursively find images. Return (index: uid->group, meta: uid->orig_stem,
    seqs: group->[uid]). Each image is staged into `flat` under a globally-unique
    uid name. Default = COPY once (safe for remote mounts); --symlink to link."""
    flat.mkdir(parents=True, exist_ok=True)
    index, meta, seqs = {}, {}, defaultdict(list)
    used = {}
    imgs = [p for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMG_EXTS]
    n = len(imgs)
    for i, img in enumerate(imgs, 1):
        group = str(img.parent.relative_to(root)) or "(root)"
        uid = f"{group}__{img.stem}".replace("/", "__").replace(" ", "_")
        while uid in used:                      # guarantee uniqueness
            uid += "_x"
        used[uid] = True
        link = flat / f"{uid}{img.suffix.lower()}"
        if link.exists() or link.is_symlink():
            link.unlink()
        if symlink:
            os.symlink(img.resolve(), link)
        else:
            shutil.copy2(img, link)             # read remote ONCE, sequentially
            if i % 500 == 0 or i == n:
                print(f"  staged {i}/{n} to local disk", flush=True)
        index[uid] = group
        meta[uid] = img.stem
        seqs[group].append(uid)
    return index, meta, seqs


def sh(cmd, env):
    print("RUN:", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd], env=env)


def run_pipeline(cfg, flat, out, manifest, device, batch, env):
    PY, SRC = cfg["python"], cfg["pipeline_src"]
    sel = Path(cfg["selector_dir"])
    steps = [
        [PY, f"{SRC}/export_recursive_detector_predictions.py", "--model", cfg["detector"],
         "--images-dir", flat, "--output-dir", out / "detector", "--imgsz", cfg["imgsz"],
         "--device", device, "--batch", batch, "--candidate-conf", 0.05],
        [PY, f"{SRC}/run_paddleocr_export.py", "--images", flat, "--out", out / "full_ocr.json",
         "--use-gpu", "--ocr-version", "PP-OCRv4", "--text-detection-model-name",
         "PP-OCRv4_mobile_det", "--text-recognition-model-name", "en_PP-OCRv4_mobile_rec",
         "--progress-every", 200],
        [PY, f"{SRC}/refine_ocr_candidates.py", "--ocr-json", out / "full_ocr.json",
         "--out", out / "refined_ocr.json"],
        [PY, cfg["recheck_padded"], "--ocr-json", out / "refined_ocr.json",
         "--out", out / "candidates.json", "--yolo-label-dir", out / "detector/labels",
         "--model-dir", cfg["numeric_ocr"], "--model-name", "PP-OCRv5_mobile_rec",
         "--device", "gpu", "--input-shape", "3,48,320", "--min-conf", 0.0,
         "--progress-every", 200],
        [PY, f"{SRC}/temporal_channel_pipeline.py", "--images", flat,
         "--candidate-json", out / "candidates.json", "--sequence-manifest", manifest,
         "--yolo-label-dir", out / "detector/labels", "--out", out / "predictions.json",
         "--mode", "search", "--temporal-history-mode", "roi_prior_recheck",
         "--history-guided-model-dir", cfg["numeric_ocr"], "--history-guided-device", "gpu",
         "--history-guided-input-shape", "3,48,320", "--history-guided-raw-only",
         "--allow-single-digit-channel-recheck", "--suppress-text-hallucinations",
         "--ranker-model", sel / "selector_no_airtel_v1_pairwise_linear.json",
         "--value-group-ranker-model", sel / "value_group_pointwise_logistic.json",
         "--relative-gate-model", sel / "relative_confidence_gate.json",
         "--relative-gate-threshold", 0.7690914017301402,
         "--relative-gate-policy", "positive_first"],
    ]
    for cmd in steps:
        rc = sh(cmd, env)
        if rc != 0:
            raise SystemExit(f"\n[predict_folder] step failed (rc={rc}): {cmd[1]}\n"
                             "이 단계의 위 에러 메시지를 확인하세요 (보통 paddle 버전/GPU 문제).")


def _dg(s):
    return "".join(c for c in str(s or "") if c.isdigit())


def _sel_box(im):
    """channel_number box that was selected (for drawing)."""
    ts = im.get("temporal_selection") or {}
    bc = ts.get("best_candidate") or {}
    if bc.get("bbox_xyxy"):
        return bc["bbox_xyxy"]
    pred = _dg(im.get("predicted_channel_number"))
    for c in im.get("candidates", []):          # fallback: matching candidate
        if _dg(c.get("text")) == pred and pred:
            return c.get("bbox_xyxy")
    return None


def save_qualitative(out, doc, index, meta, folder_vote, per_folder_uids,
                     n_sample, flat):
    """Draw the channel_number box + predicted number. Save up to n_sample
    samples per folder, and ALL failures (empty pred or pred != folder vote)."""
    try:
        import cv2
    except Exception as e:
        print(f"[qualitative] cv2 없음 → 이미지 저장 건너뜀 ({e})", flush=True)
        return
    qdir = out / "qualitative"
    by_uid = {im["image_id"]: im for im in doc["images"]}
    saved = 0
    for folder, uids in sorted(per_folder_uids.items()):
        vote = folder_vote.get(folder, "")
        fdir = qdir / folder.replace("/", "__")
        faildir = fdir / "_failures"
        # classify frames: failure = empty pred OR disagrees with folder vote
        fails, oks = [], []
        for uid in uids:
            im = by_uid.get(uid)
            if im is None:
                continue
            pred = _dg(im.get("predicted_channel_number"))
            (fails if (pred == "" or (vote and pred != vote)) else oks).append(uid)
        # sample n_sample "ok" frames evenly + ALL failures
        if oks:
            step = max(1, len(oks) // n_sample)
            sample = oks[::step][:n_sample]
        else:
            sample = []
        to_draw = [(u, False) for u in sample] + [(u, True) for u in fails]
        if not to_draw:
            continue
        fdir.mkdir(parents=True, exist_ok=True)
        if fails:
            faildir.mkdir(parents=True, exist_ok=True)
        for uid, is_fail in to_draw:
            im = by_uid[uid]
            src = flat / f"{uid}{Path(im.get('image_path','')).suffix or '.jpg'}"
            if not src.exists():
                src = Path(im.get("image_path", ""))
            img = cv2.imread(str(src))
            if img is None:
                continue
            pred = _dg(im.get("predicted_channel_number")) or "(none)"
            box = _sel_box(im)
            color = (0, 0, 255) if is_fail else (0, 200, 0)   # BGR: red / green
            if box:
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"pred:{pred}" + (f" (vote:{vote})" if is_fail else "")
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        color, 2, cv2.LINE_AA)
            dst = (faildir if is_fail else fdir) / f"{meta.get(uid, uid)}.jpg"
            cv2.imwrite(str(dst), img)
            saved += 1
    print(f"\n정성 이미지 저장: {saved}장  →  {qdir}", flush=True)
    print("  (폴더별 샘플 + _failures/ 에 실패 프레임 전부)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=16,
                    help="detector batch (낮을수록 GPU/IO 부담 적음; 기본 16)")
    ap.add_argument("--symlink", action="store_true",
                    help="원격 복사 대신 심링크 (로컬 디스크 데이터에만 사용)")
    ap.add_argument("--samples-per-folder", type=int, default=20,
                    help="폴더당 저장할 정성 샘플 수 (실패 케이스는 이와 별개로 전부 저장)")
    ap.add_argument("--no-qualitative", action="store_true",
                    help="정성 이미지 저장 안 함 (CSV만)")
    ap.add_argument("--keep-staged", action="store_true",
                    help="추론 후 로컬 복사본(images/) 유지 (기본은 삭제해 디스크 절약)")
    args = ap.parse_args()
    cfg = load_config()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/opt/conda/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    flat = out / "images"
    mode = "심링크" if args.symlink else "로컬 복사(원격 마운트 보호)"
    print(f"[stage] {mode}", flush=True)
    index, meta, seqs = collect(root, flat, args.symlink)
    if not index:
        raise SystemExit(f"이미지를 찾지 못했습니다: {root} (하위 폴더 포함 재귀 탐색함). "
                         "경로/확장자를 확인하세요.")
    print(f"images: {len(index)}  folders: {len(seqs)}", flush=True)
    manifest = out / "manifest.json"
    manifest.write_text(json.dumps(
        {"sequence_count": len(seqs),
         "sequences": [{"sequence_id": g.replace('/', '__'), "group_key": g,
                        "images": sorted(v)} for g, v in sorted(seqs.items())]},
        ensure_ascii=False))

    run_pipeline(cfg, flat, out, manifest, args.device, args.batch, env)

    doc = json.loads((out / "predictions.json").read_text())
    per_frame, per_folder = [], defaultdict(list)
    for im in doc["images"]:
        uid = im["image_id"]
        pred = _dg(im.get("predicted_channel_number"))
        folder = index.get(uid, "")
        per_frame.append({"folder": folder, "frame": meta.get(uid, uid), "prediction": pred})
        if pred:
            per_folder[folder].append(pred)

    with (out / "per_frame.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "frame", "prediction"])
        w.writeheader(); w.writerows(per_frame)
    folder_rows, folder_vote = [], {}
    for folder, preds in sorted(per_folder.items()):
        cnt = Counter(preds)
        best, nb = cnt.most_common(1)[0]
        folder_vote[folder] = best
        folder_rows.append({"folder": folder, "channel_number": best,
                            "confidence": round(nb / len(preds), 3), "n_frames": len(preds)})
    with (out / "per_folder.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "channel_number", "confidence", "n_frames"])
        w.writeheader(); w.writerows(folder_rows)

    # per-folder success / failure ratio (success = frame agrees with folder vote)
    frames_by_folder = defaultdict(list)
    for r in per_frame:
        frames_by_folder[r["folder"]].append(_dg(r["prediction"]))
    ratio_rows, tot_ok = [], 0
    tot_n = 0
    for folder in sorted(frames_by_folder):
        vote = folder_vote.get(folder, "")
        preds = frames_by_folder[folder]
        ok = sum(1 for p in preds if vote and p == vote)
        fail = len(preds) - ok
        rate = round(ok / len(preds) * 100, 1) if preds else 0.0
        ratio_rows.append({"folder": folder, "channel_number": vote, "total": len(preds),
                           "success": ok, "fail": fail, "success_rate(%)": rate})
        tot_ok += ok; tot_n += len(preds)
    with (out / "per_folder_success.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "channel_number", "total",
                                          "success", "fail", "success_rate(%)"])
        w.writeheader(); w.writerows(ratio_rows)

    print("\n=== 폴더별 채널번호 (프레임 다수결) ===")
    for r in folder_rows:
        print(f"  {r['folder']}: {r['channel_number']}  (conf {r['confidence']}, {r['n_frames']}프레임)")
    print("\n=== 폴더별 성공/실패 비율 (성공 = 다수결값과 일치) ===")
    for r in ratio_rows:
        print(f"  {r['folder']:<40} ch={r['channel_number']:<5} "
              f"성공 {r['success']}/{r['total']}  실패 {r['fail']}  "
              f"({r['success_rate(%)']}%)")
    print(f"  {'-'*40}")
    print(f"  {'전체':<40} 성공 {tot_ok}/{tot_n}  "
          f"({round(tot_ok/tot_n*100,1) if tot_n else 0}%)")
    print(f"\n결과: {out}/per_folder.csv , per_frame.csv , per_folder_success.csv")

    if not args.no_qualitative:
        save_qualitative(out, doc, index, meta, folder_vote, dict(seqs),
                         args.samples_per_folder, flat)

    if not args.keep_staged:
        shutil.rmtree(flat, ignore_errors=True)   # 로컬 복사본 삭제 (디스크 절약)
        print(f"[cleanup] 로컬 복사본 삭제: {flat}", flush=True)


if __name__ == "__main__":
    main()
