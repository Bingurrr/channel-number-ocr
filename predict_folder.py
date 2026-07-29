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
import re
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


def _decodable(path):
    """True if the image can actually be decoded (skip truncated/corrupt files)."""
    try:
        from PIL import Image as _I
    except Exception:
        return True                              # PIL 없으면 검증 생략
    try:
        with _I.open(path) as im:
            im.load()                            # full decode — catches truncation
        return True
    except Exception:
        return False


def collect(root: Path, flat: Path, symlink: bool):
    """Recursively find images. Return (index: uid->group, meta: uid->orig_stem,
    seqs: group->[uid]). Each image is staged into `flat` under a globally-unique
    uid name. Default = COPY once (safe for remote mounts); --symlink to link.
    Unreadable/corrupt images are skipped so the pipeline doesn't crash on them."""
    flat.mkdir(parents=True, exist_ok=True)
    index, meta, seqs = {}, {}, defaultdict(list)
    used = {}
    bad = []
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
            if not _decodable(link):            # 손상 이미지 스킵
                link.unlink(); del used[uid]; bad.append(str(img)); continue
        else:
            shutil.copyfile(img, link)          # content only (rclone has no xattr);
                                                # read remote ONCE, sequentially
            if not _decodable(link):            # 복사본이 깨졌으면 스킵
                link.unlink(); del used[uid]; bad.append(str(img)); continue
            if i % 500 == 0 or i == n:
                print(f"  staged {i}/{n} to local disk", flush=True)
        index[uid] = group
        meta[uid] = img.stem
        seqs[group].append(uid)
    if bad:
        (flat.parent / "skipped_unreadable.txt").write_text("\n".join(bad))
        print(f"[skip] 손상/읽기불가 이미지 {len(bad)}장 건너뜀 "
              f"(목록: {flat.parent / 'skipped_unreadable.txt'})", flush=True)
    return index, meta, seqs


def sh(cmd, env):
    print("RUN:", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd], env=env)


def run_pipeline(cfg, flat, out, manifest, device, batch, env, accum_frames=None):
    PY, SRC = cfg["python"], cfg["pipeline_src"]
    sel = Path(cfg["selector_dir"])
    # translate "N frames to accumulate" -> EMA span: alpha = 2/(N+1); warm up over N
    accum_args = []
    if accum_frames and accum_frames > 0:
        alpha = round(2.0 / (accum_frames + 1), 4)
        accum_args = ["--smooth-alpha", str(alpha),
                      "--warmup-frames", str(accum_frames),
                      "--min-seen", str(max(2, min(3, accum_frames)))]
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
         "--relative-gate-policy", "positive_first", *accum_args],
    ]
    for cmd in steps:
        rc = sh(cmd, env)
        if rc != 0:
            raise SystemExit(f"\n[predict_folder] step failed (rc={rc}): {cmd[1]}\n"
                             "이 단계의 위 에러 메시지를 확인하세요 (보통 paddle 버전/GPU 문제).")


def _dg(s):
    return "".join(c for c in str(s or "") if c.isdigit())


def gt_from_name(stem: str) -> str:
    """Leading digits of the filename = ground-truth channel number (150.jpg->150)."""
    m = re.match(r"\s*(\d+)", str(stem))
    return m.group(1) if m else ""


def _sel_box(im):
    """channel_number box selected THIS frame (single-frame inference)."""
    ts = im.get("temporal_selection") or {}
    bc = ts.get("best_candidate") or {}
    if bc.get("bbox_xyxy"):
        return bc["bbox_xyxy"]
    pred = _dg(im.get("predicted_channel_number"))
    for c in im.get("candidates", []):          # fallback: matching candidate
        if _dg(c.get("text")) == pred and pred:
            return c.get("bbox_xyxy")
    return None


def _locked_box(im):
    """Accumulated (history-locked) channel_number position, if locked."""
    st = im.get("temporal_slot_state_after") or im.get("temporal_slot_state_before") or {}
    return st.get("locked_slot_bbox")


def save_qualitative(dir_for, doc, index, meta, truth_by_uid, per_folder_uids,
                     n_sample, flat, equiv, viz_history=False, no_label=False):
    """Draw the channel_number box + predicted number. Save up to n_sample
    samples per folder, and ALL failures. `truth_by_uid` = the answer each frame
    is judged against; empty in --no-label mode (then failure = frame with no
    prediction at all). `dir_for(folder)` returns that folder's output dir."""
    try:
        import cv2
    except Exception as e:
        print(f"[qualitative] cv2 없음 → 이미지 저장 건너뜀 ({e})", flush=True)
        return
    norm = (lambda s: str(int(s)) if s else "") if equiv else (lambda s: s)
    by_uid = {im["image_id"]: im for im in doc["images"]}
    saved = 0
    for folder, uids in sorted(per_folder_uids.items()):
        fdir = dir_for(folder)
        faildir = fdir / "_failures"
        # classify frames: failure = empty pred OR pred != that frame's truth
        fails, oks = [], []
        for uid in uids:
            im = by_uid.get(uid)
            if im is None:
                continue
            pred = _dg(im.get("predicted_channel_number"))
            ans = truth_by_uid.get(uid, "")
            (fails if (pred == "" or (ans and norm(pred) != norm(ans))) else oks).append(uid)
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
            ans = truth_by_uid.get(uid, "")
            box = _sel_box(im)                                 # 현재 단독 추론
            color = (0, 0, 255) if is_fail else (0, 200, 0)   # BGR: red / green
            TH = 3                                             # 박스 두께 (굵게)
            if is_fail:
                # 실패 케이스: YOLO가 검출한 모든 박스를 시각화 (원인 파악용)
                for yb in (im.get("yolo_channel_boxes") or []):
                    bb = yb.get("bbox_xyxy")
                    if not bb:
                        continue
                    yx1, yy1, yx2, yy2 = [int(round(v)) for v in bb]
                    ch = yb.get("class_id") in (0, 3)
                    c = (0, 200, 0) if ch else (0, 220, 220)   # 채널=초록, 나머지=노랑
                    cv2.rectangle(img, (yx1, yy1), (yx2, yy2), c, TH if ch else 2)
                    cv2.putText(img, f"{yb.get('class_name','')}:{yb.get('conf',0):.2f}",
                                (yx1, max(0, yy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, c, 1, cv2.LINE_AA)
            if viz_history:
                # 현재 단독 = 초록, 누적 잠금 = 주황 (두 위치 비교)
                lb = _locked_box(im)
                if lb:
                    lx1, ly1, lx2, ly2 = [int(round(v)) for v in lb]
                    cv2.rectangle(img, (lx1, ly1), (lx2, ly2), (0, 165, 255), TH)  # 주황
                    cv2.putText(img, "accum(history)", (lx1, max(0, ly1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
                if box:
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), TH)        # 초록
                    cv2.putText(img, "single", (x1, y2 + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
                cv2.putText(img, "green=single  orange=accum", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            elif box:
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, TH)
            label = f"pred:{pred}"
            if is_fail and ans and not no_label:
                label += f" (ans:{ans})"
            elif is_fail and no_label:
                label += " (unread)"
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        color, 2, cv2.LINE_AA)
            dst = (faildir if is_fail else fdir) / f"{meta.get(uid, uid)}.jpg"
            cv2.imwrite(str(dst), img)
            saved += 1
    print(f"\n정성 이미지 저장: {saved}장", flush=True)
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
    ap.add_argument("--gt-from-filename", action="store_true",
                    help="파일명이 정답 채널번호 (150.jpg->150). 성공/실패를 폴더 다수결이 "
                         "아니라 파일명 정답과 비교. 기본은 각 이미지를 독립(단일프레임)으로 처리.")
    ap.add_argument("--accumulate", action="store_true",
                    help="폴더 프레임을 한 시퀀스로 보고 위치 누적/잠금을 켬. --gt-from-filename과 "
                         "함께 쓰면: 정답은 파일명(프레임별) + 위치는 누적. 연속 프레임 데이터에서만 의미.")
    ap.add_argument("--accum-frames", type=int, default=None,
                    help="누적에 반영할 프레임 수(대략). 클수록 더 많은 과거 프레임을 평균(안정적, "
                         "느린 반응), 작을수록 최근 위주(빠른 반응). 기본 미지정=내부값(약 5프레임 상당).")
    ap.add_argument("--numeric-equiv", action="store_true",
                    help="앞자리 0 무시하고 채점 (041 == 41)")
    ap.add_argument("--viz-history", action="store_true",
                    help="정성 이미지에 [현재 단독 추론=초록] vs [누적 잠금 위치=주황] 두 박스를 "
                         "다른 색으로 표시. 누적이 켜진 기본 모드에서만 의미 있음.")
    ap.add_argument("--keep-staged", action="store_true",
                    help="추론 후 로컬 복사본(images/) 유지 (기본은 삭제해 디스크 절약)")
    ap.add_argument("--split-output", action="store_true",
                    help="--out 아래에 UI별(사진 바로 상위 폴더 이름) 하위폴더를 만들어 "
                         "정성/정량 결과를 각각 저장.")
    ap.add_argument("--no-label", action="store_true",
                    help="정답 라벨이 없는 데이터. accuracy 대신 최종 OCR 결과(폴더별 예측 "
                         "채널번호)만 저장. 파일명도 정답이 아닐 때 사용.")
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
    # ground-truth per frame: filename digits (GT mode) — else filled after voting
    gt_name = {uid: gt_from_name(stem) for uid, stem in meta.items()}
    manifest = out / "manifest.json"
    accumulate = args.accumulate or not args.gt_from_filename   # 기본모드는 항상 누적
    if accumulate:
        # 폴더 프레임을 한 시퀀스로 -> 위치 누적/잠금 발동
        sequences = [{"sequence_id": g.replace('/', '__'), "group_key": g,
                      "images": sorted(v)} for g, v in sorted(seqs.items())]
        if args.gt_from_filename:
            print("[mode] 파일명=정답 + 누적 ON (폴더 시퀀스, 값은 프레임별로 채점)", flush=True)
        else:
            print("[mode] 누적 ON (폴더 시퀀스, 성공=폴더 다수결)", flush=True)
    else:
        # 각 이미지 독립 (파일명마다 채널 다른 낱장) -> 누적 off
        sequences = [{"sequence_id": uid, "group_key": index[uid], "images": [uid]}
                     for uid in sorted(index)]
        print("[mode] 파일명=정답: 각 이미지 단일프레임 처리 (누적 off)", flush=True)
    manifest.write_text(json.dumps(
        {"sequence_count": len(sequences), "sequences": sequences}, ensure_ascii=False))

    if args.accum_frames:
        print(f"[accum] 누적 프레임 수 ≈ {args.accum_frames} "
              f"(alpha={round(2.0/(args.accum_frames+1),4)})", flush=True)
    run_pipeline(cfg, flat, out, manifest, args.device, args.batch, env,
                 accum_frames=args.accum_frames)

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

    # ---- what each frame is judged against ----
    #   --no-label        -> no GT (save predictions only, report read-rate/consistency)
    #   --gt-from-filename -> the frame's own filename digits
    #   default           -> its folder's majority vote
    equiv = args.numeric_equiv or args.gt_from_filename
    norm = (lambda s: str(int(s)) if s else "") if equiv else (lambda s: s)
    if args.no_label:
        truth_by_uid = {}
    elif args.gt_from_filename:
        truth_by_uid = dict(gt_name)
    else:
        truth_by_uid = {uid: folder_vote.get(index.get(uid, ""), "") for uid in index}

    frames_by_folder = defaultdict(list)   # folder -> [(uid, pred)]
    for im in doc["images"]:
        uid = im["image_id"]
        frames_by_folder[index.get(uid, "")].append((uid, _dg(im.get("predicted_channel_number"))))

    # ---- per-folder quantitative summary ----
    summaries, tot_ok, tot_n = {}, 0, 0
    for folder in sorted(frames_by_folder):
        frames = frames_by_folder[folder]
        n = len(frames)
        n_read = sum(1 for _, p in frames if p)
        vote = folder_vote.get(folder, "")
        s = {"folder": folder, "predicted_channel_number": vote, "n_frames": n,
             "n_read": n_read, "read_rate(%)": round(n_read / n * 100, 1) if n else 0.0}
        if args.no_label:
            cons = sum(1 for _, p in frames if p and norm(p) == norm(vote))
            s["accuracy(%)"] = None
            s["consistency(%)"] = round(cons / n_read * 100, 1) if n_read else 0.0
            s["has_label"] = False
        else:
            ok = sum(1 for uid, p in frames if truth_by_uid.get(uid) and norm(p) == norm(truth_by_uid[uid]))
            tot = sum(1 for uid, _ in frames if truth_by_uid.get(uid))
            s["accuracy(%)"] = round(ok / tot * 100, 1) if tot else None
            s["scored"] = tot; s["success"] = ok; s["fail"] = tot - ok
            s["has_label"] = True
            tot_ok += ok; tot_n += tot
        summaries[folder] = s

    def safe(name):
        return (name.replace("/", "__").replace(" ", "_")) or root.name

    # immediate-parent-folder name per group (사진들 바로 상위 폴더 이름), dedup collisions
    gname_of, used_names = {}, {}
    for folder in sorted(summaries):
        base = Path(folder).name if folder not in ("", "(root)") else root.name
        name = safe(base)
        while name in used_names and used_names[name] != folder:
            name += "_x"
        used_names[name] = folder
        gname_of[folder] = name

    # combined outputs
    (out / "summary.json").write_text(json.dumps(
        {"folders": list(summaries.values()),
         "overall_accuracy(%)": (round(tot_ok / tot_n * 100, 1)
                                 if (tot_n and not args.no_label) else None),
         "has_label": not args.no_label}, ensure_ascii=False, indent=2))
    fields = ["folder", "predicted_channel_number", "n_frames", "n_read",
              "read_rate(%)", "accuracy(%)"] + (
                  ["consistency(%)"] if args.no_label else ["scored", "success", "fail"])
    with (out / "per_folder_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(summaries.values())

    # per-UI split output: --out/<UI 이름>/{summary.json, per_frame.csv, qualitative/}
    if args.split_output:
        for folder, s in summaries.items():
            gdir = out / gname_of[folder]
            gdir.mkdir(parents=True, exist_ok=True)
            (gdir / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2))
            with (gdir / "per_frame.csv").open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=["frame", "prediction"])
                w.writeheader()
                for uid, p in frames_by_folder[folder]:
                    w.writerow({"frame": meta.get(uid, uid), "prediction": p})

    # ---- terminal report ----
    print("\n=== 폴더별 결과 ===")
    for folder, s in summaries.items():
        if args.no_label:
            print(f"  {folder:<38} 채널={str(s['predicted_channel_number']):<5} "
                  f"읽음 {s['n_read']}/{s['n_frames']}({s['read_rate(%)']}%)  "
                  f"일관성 {s['consistency(%)']}%")
        else:
            print(f"  {folder:<38} 채널={str(s['predicted_channel_number']):<5} "
                  f"정확도 {s['accuracy(%)']}% (성공 {s['success']}/{s['scored']})")
    if args.no_label:
        print("  ※ 라벨 없음 → 정확도 대신 [예측 채널번호 / 읽음율 / 일관성]만 저장")
    elif tot_n:
        print(f"  {'-'*38}\n  전체 정확도 {round(tot_ok/tot_n*100,1)}% ({tot_ok}/{tot_n})")
    print(f"\n결과: {out}/summary.json , per_folder_summary.csv , per_frame.csv"
          + ("  (+ UI별 하위폴더)" if args.split_output else ""))

    if args.viz_history and args.gt_from_filename and not args.accumulate:
        print("[주의] --viz-history인데 누적이 꺼져 있습니다(--gt-from-filename 단일프레임). "
              "주황(누적) 박스를 보려면 --accumulate 를 추가하세요.", flush=True)
    if not args.no_qualitative:
        if args.split_output:
            dir_for = lambda folder: out / gname_of[folder] / "qualitative"
        else:
            dir_for = lambda folder: out / "qualitative" / safe(folder)
        save_qualitative(dir_for, doc, index, meta, truth_by_uid, dict(seqs),
                         args.samples_per_folder, flat, equiv,
                         viz_history=args.viz_history, no_label=args.no_label)

    if not args.keep_staged:
        shutil.rmtree(flat, ignore_errors=True)   # 로컬 복사본 삭제 (디스크 절약)
        print(f"[cleanup] 로컬 복사본 삭제: {flat}", flush=True)


if __name__ == "__main__":
    main()
