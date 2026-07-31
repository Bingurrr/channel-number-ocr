#!/usr/bin/env python3
"""Channel inference with ENHANCED slot analysis (full OCR only, on-device).

Same as predict_folder_ocr_only.py but the slot step uses slot_analysis.analyze():
  * clustering by POSITION **+ SIZE** (separates crowded boxes near the channel)
  * VALUE DIVERSITY qualifies the channel (zap changes value)
  * DUAL-DISPLAY: a 2nd slot with matching values fills gaps of the primary
  * vertical channel-list penalty (pre/next-channel entries)

The dual slot and the known ROI are the "past experience" that corrects single-frame
errors: if the primary is unread in a frame, the dual (same channel) fills it; and
--force-read re-OCRs the primary ROI crop for anything still missing.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import predict_folder as P
import slot_analysis as SA
from predict_folder_temporal_full import recover_field_values
import step_viz


def _safe(name, root):
    return name.replace("/", "__").replace(" ", "_") or root.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symlink", action="store_true")
    ap.add_argument("--gt-from-filename", action="store_true")
    ap.add_argument("--split-output", action="store_true")
    ap.add_argument("--samples-per-folder", type=int, default=20)
    ap.add_argument("--no-qualitative", action="store_true")
    ap.add_argument("--force-read", action="store_true",
                    help="(선택) 못읽은 프레임의 확정 ROI를 crop+확대해서 full OCR로 재읽기")
    ap.add_argument("--viz-force-read", type=int, default=0, metavar="N",
                    help="force-read 변환 시각화(폴더당 N장): crop→padding→확대→읽기 스텝 저장")
    ap.add_argument("--viz-steps", type=int, default=0, metavar="N")
    ap.add_argument("--keep-staged", action="store_true")
    ap.add_argument("--rec", default=None,
                    help="rec 모델 선택(프리셋): v4 | v6small | v6tiny | stock. "
                         "직접 경로는 --rec-model-dir 사용")
    ap.add_argument("--rec-model-dir", default="models/full_image_ocr/en_PP-OCRv4_mobile_rec_ft",
                    help="파인튜닝된 full-OCR rec 디렉토리(있으면 사용). 순정으로 돌리려면 'none'")
    ap.add_argument("--min-conf", type=float, default=0.3,
                    help="슬롯 클러스터링 전 OCR 신뢰도 임계값(낮추면 none↓, 오답↑ 가능)")
    ap.add_argument("--force-answer", action="store_true",
                    help="못읽은 프레임을 rec-only(det 생략)로 강제 읽기 → none 없애기")
    ap.add_argument("--both-channel", action="store_true",
                    help="채널번호가 2곳(A/B)에 뜨는 UI: 둘 다 저장(ch1/ch2)하고, 한쪽이 "
                         "none이거나 conf 낮으면 다른쪽 값 사용")
    ap.add_argument("--field-priority", action="store_true",
                    help="[v3] 슬롯값 대신 'field box 근처 최고conf 후보'를 우선 사용. "
                         "튜너 검증: field box 읽기 91.5% 정확(슬롯 87.7%보다 높음). "
                         "근처 후보 없는 프레임은 기존 슬롯/force-answer로 커버")
    ap.add_argument("--field-near", type=float, default=0.08,
                    help="field-priority: field box 중심에서 이 정규화 반경 내 후보만 채널로 인정")
    ap.add_argument("--second-channel", action="store_true",
                    help="[v3] 사용자 방법론: primary값과 같은 숫자가 뜨는 '2번째 위치'를 "
                         "프레임 누적 투표로 학습 → primary가 none이거나 conf 낮은 프레임을 "
                         "2번째 위치 읽기로 보완/교정")
    ap.add_argument("--second-min-votes", type=int, default=5,
                    help="second-channel: 2번째영역으로 확정하는 최소 누적 일치 투표수")
    ap.add_argument("--second-conf-gate", type=float, default=0.6,
                    help="second-channel: primary conf가 이 값 미만이고 2번째가 더 높으면 교정")
    ap.add_argument("--color-isolate", action="store_true",
                    help="[v3] force-answer 재읽기 시 여러 프레임에서 폰트색을 학습해 숫자만 "
                         "남기고 배경 제거 후 rec (유사배경/저대비 프레임 개선). "
                         "분리 시각화는 out/color_isolate_viz/ 에 저장")
    ap.add_argument("--font-isolate", action="store_true",
                    help="[v3] slot이 찾은 채널박스에서 성공 detection의 폰트색을 Otsu로 학습 → "
                         "각 프레임 채널박스를 '폰트 vs 배경 최근접'으로 분리(흰-on-흰 구제) → "
                         "재읽기. out/font_isolate_viz/, font_isolate_clean/, summary 저장")
    ap.add_argument("--font-isolate-conf", type=float, default=0.7,
                    help="font-isolate: 폰트색 학습에 쓸 성공 detection 최소 conf")
    ap.add_argument("--font-isolate-margin", type=float, default=1.0,
                    help="font-isolate: 색분리 보수성(>1이면 배경 더 지움, <1이면 덜)")
    ap.add_argument("--font-isolate-viz", type=int, default=30, help="font-isolate: 폴더당 시각화 장수")
    ap.add_argument("--cluster-by-size", action="store_true",
                    help="[v3] 채널박스가 채널마다 다른 위치에 뜨는 UI: 위치 대신 '글자 높이(폰트 "
                         "크기)'로 클러스터링/선택. 위치 이동에 강함(값다양성+채널숫자+높이일관성으로 선택)")
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
    print(f"images: {len(index)}  folders(sequences): {len(seqs)}", flush=True)

    # === 유일한 모델: FULL OCR ===
    ocr_cmd = [PY, f"{SRC}/run_paddleocr_export.py", "--images", flat, "--out", out / "full_ocr.json",
               "--use-gpu", "--ocr-version", "PP-OCRv4", "--text-detection-model-name",
               "PP-OCRv4_mobile_det", "--text-recognition-model-name", "en_PP-OCRv4_mobile_rec",
               "--progress-every", 200]
    # rec 모델 선택: --rec 프리셋이 있으면 그걸로, 없으면 --rec-model-dir.
    REC_PRESETS = {"v3": "models/full_image_ocr/en_PP-OCRv4_mobile_rec_ft",
                   "v4": "models/full_image_ocr/rec_v4",
                   "v6small": "models/full_image_ocr/rec_v6small",
                   "v6tiny": "models/full_image_ocr/rec_v6tiny",
                   "stock": "none"}
    # 파인튜닝된 rec가 있으면 그 가중치로 읽기 (det은 그대로). 'none'이면 순정.
    rec_dir = str(REC_PRESETS.get(args.rec, args.rec) if args.rec else args.rec_model_dir).strip()
    if rec_dir.lower() not in ("", "none"):
        rd = Path(rec_dir)
        if not rd.is_absolute():
            rd = Path(__file__).resolve().parent / rd
        if (rd / "inference.pdiparams").exists():
            ocr_cmd += ["--text-recognition-model-dir", str(rd)]
            print(f"[slot] 파인튜닝 rec 사용: {rd}", flush=True)
        else:
            print(f"[slot] rec-model-dir 없음 → 순정 rec 사용 ({rd})", flush=True)
    rc = P.sh(ocr_cmd, env)
    if rc != 0:
        raise SystemExit(f"[slot] full OCR 실패 (rc={rc})")
    cand = json.loads((out / "full_ocr.json").read_text())
    by_id = {im["image_id"]: im for im in cand.get("images", [])}

    # === 강화된 슬롯 분석 -> 채널 필드(+듀얼) ===
    report, per_folder = [], {}
    for g, uids in sorted(seqs.items()):
        ok_uids = [u for u in sorted(uids) if u in by_id]
        frames = [by_id[u] for u in ok_uids]
        if not frames:
            continue
        primary, duals, allm = SA.analyze(frames, ok_uids, conf_thr=args.min_conf, by_size=args.cluster_by_size)
        if primary:
            p_pf = primary["per_frame"]; p_conf = primary.get("per_frame_conf", {})
            # 듀얼(2번째 채널박스) 값/신뢰도 병합 (여러 듀얼이면 프레임별 최고 conf)
            d_pf, d_conf = {}, {}
            for d in duals:
                dc = d.get("per_frame_conf", {})
                for uid, v in d["per_frame"].items():
                    c = dc.get(uid, 0.5)
                    if uid not in d_conf or c > d_conf[uid]:
                        d_pf[uid] = v; d_conf[uid] = c
            if args.both_channel:
                # 프레임내 중복(같은 숫자 2곳) — 위치 무관 신호라 '움직이는 2번째 박스'에 강함
                wf = SA.within_frame_dupes(frames, ok_uids, conf_thr=args.min_conf)
                # A/B 교차: 둘 다 있으면 conf 높은쪽, 하나면 그것, 둘 다 없으면 프레임내 중복
                final = {}
                for uid in set(p_pf) | set(d_pf) | set(wf):
                    v1, v2 = p_pf.get(uid), d_pf.get(uid)
                    if v1 and v2:
                        final[uid] = v1 if p_conf.get(uid, 0) >= d_conf.get(uid, 0) else v2
                    elif v1 or v2:
                        final[uid] = v1 or v2
                    else:
                        final[uid] = wf.get(uid)               # 이동 박스 등 → 중복으로 복구
                # ch2 = 고정 듀얼 값이 없으면 프레임내 중복값으로 채워 보여줌
                pf2 = dict(d_pf)
                for uid, v in wf.items():
                    pf2.setdefault(uid, v)
                field = {"median_box": primary["box"], "per_frame": {k: v for k, v in final.items() if v},
                         "per_frame_1": dict(p_pf), "per_frame_2": pf2,
                         "box2": duals[0]["box"] if duals else None,
                         "score": primary["score"], "distinct": primary["distinct"],
                         "duals": [d["box"] for d in duals], "both": True}
            else:
                merged = dict(p_pf)
                mconf = dict(p_conf)
                for uid, v in d_pf.items():                     # 기존: primary 빈칸만 채움
                    merged.setdefault(uid, v)
                    mconf.setdefault(uid, d_conf.get(uid, 0.5))
                field = {"median_box": primary["box"], "per_frame": merged,
                         "per_frame_conf": mconf,
                         "score": primary["score"], "distinct": primary["distinct"],
                         "duals": [d["box"] for d in duals], "both": False}
            recover_field_values(frames, ok_uids, field)
        else:
            field = None
        report.append({"folder": g, "channel_field_box": field["median_box"] if field else None,
                       "field_score": field["score"] if field else 0.0,
                       "distinct_values": field["distinct"] if field else 0,
                       "dual_boxes": field.get("duals", []) if field else [],
                       "slots": allm[:6]})
        per_folder[g] = (report[-1], field)

    # === [v3] field box 우선: 슬롯값을 'field box 근처 최고conf 후보'로 오버라이드 ===
    # 튜너 검증: field box 읽기가 91.5% 정확(슬롯 87.7%보다 높음). 근처 후보 없는
    # 프레임은 건드리지 않아(기존 슬롯/force-answer 커버) → v1 이상 보장.
    if args.field_priority:
        overridden = added = 0
        for g, (entry, field) in per_folder.items():
            if not field:
                continue
            ok_uids = [u for u in sorted(seqs.get(g, [])) if u in by_id]
            frames = [by_id[u] for u in ok_uids]
            fbox = field["median_box"]
            fb = SA.read_at_field_box(frames, ok_uids, fbox, conf_thr=args.min_conf, near=args.field_near)
            for uid, v in fb.items():
                prev = field["per_frame"].get(uid)
                if prev is None:
                    added += 1
                elif prev != v:
                    overridden += 1
                field["per_frame"][uid] = v
        print(f"[field-priority/v3] field box 우선 적용: 덮어씀 {overridden}, 신규 {added} 프레임", flush=True)

    # === [v3] 2번째 채널영역 (사용자 방법론): 학습 → 보완/교정 ===
    if args.second_channel:
        t_fill = t_ovr = 0
        for g, (entry, field) in per_folder.items():
            if not field:
                continue
            ok_uids = [u for u in sorted(seqs.get(g, [])) if u in by_id]
            frames = [by_id[u] for u in ok_uids]
            # primary값 = 지금까지 확정된 per_frame (v1 슬롯 + 듀얼 + force)
            primary_of = dict(field["per_frame"])
            region = SA.learn_second_region(frames, ok_uids, primary_of, field["median_box"],
                                            min_votes=args.second_min_votes)
            if not region:
                entry["second_region"] = None
                continue
            pconf = field.get("per_frame_conf", {})
            fill = ovr = 0
            for fi, im in enumerate(frames):
                uid = ok_uids[fi]
                sv, sc = SA.read_at_region(im, region)
                if not sv:
                    continue
                if uid not in field["per_frame"]:
                    field["per_frame"][uid] = sv; fill += 1               # none 보완
                elif pconf.get(uid, 1.0) < args.second_conf_gate and sc > pconf.get(uid, 1.0):
                    field["per_frame"][uid] = sv; ovr += 1                # 저conf 교정
            entry["second_region"] = [round(region[0], 3), round(region[1], 3), region[2]]
            t_fill += fill; t_ovr += ovr
            print(f"[second-channel/v3] {g}: 2번째영역 votes={region[2]}  보완 {fill}  교정 {ovr}", flush=True)
        print(f"[second-channel/v3] 합계: 보완 {t_fill}  교정 {t_ovr} 프레임", flush=True)

    # === [v3] 폰트색 학습 → 채널박스 색분리 → 재읽기 (slot 결과 직접 사용, 파일왕복 X) ===
    if args.font_isolate:
        import preprocess_digits as PD
        import font_color_isolate as FCI
        from PIL import Image as _Img
        rd_fi = Path(rec_dir) if rec_dir.lower() not in ("", "none") else None
        if rd_fi is not None and not rd_fi.is_absolute():
            rd_fi = Path(__file__).resolve().parent / rd_fi
        rec = FCI.load_recognizer(str(rd_fi) if rd_fi and (rd_fi / "inference.pdiparams").exists() else None)
        ftmp = Path(tempfile.mkdtemp(prefix="fontiso_"))
        vizdir = out / "font_isolate_viz"; cleandir = out / "font_isolate_clean"
        for d in (vizdir, cleandir):
            shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True, exist_ok=True)
        fi_rows = []; t_imp = 0
        for g, (entry, field) in per_folder.items():
            if not field:
                continue
            fx1, fy1, fx2, fy2 = field["median_box"]
            ok_uids = [u for u in sorted(seqs.get(g, [])) if u in by_id]
            # 1) 폰트색 학습: 채널박스 안 '성공(고conf) 숫자 detection'의 tight bbox에서
            tight = []
            for uid in ok_uids:
                im = by_id[uid]
                for c in im.get("candidates", []):
                    b = c.get("bbox_xyxy"); t = c.get("text", "")
                    if not b or len(b) != 4 or not re.sub(r"\D", "", str(t)):
                        continue
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    if not (fx1 <= cx <= fx2 and fy1 <= cy <= fy2):
                        continue
                    if float(c.get("ocr_conf", 0) or 0) < args.font_isolate_conf:
                        continue
                    try:
                        tight.append(_Img.open(im["image_path"]).convert("RGB").crop(
                            (int(b[0]), int(b[1]), int(b[2]), int(b[3]))))
                    except Exception:
                        pass
            font, bgcol = PD.learn_font_color_otsu(tight) if tight else (None, None)
            entry["font_color"] = None if font is None else [int(v) for v in font]
            entry["bg_color"] = None if bgcol is None else [int(v) for v in bgcol]
            # 1.5) 박스 이동 대응: 프레임별 실제 채널 위치 추적 + 이동범위(envelope) 계산
            mcx, mcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
            rad = max(fx2 - fx1, fy2 - fy1) * 2.0                # 이동 허용 반경
            per_box, env = {}, [fx1, fy1, fx2, fy2]
            for uid in ok_uids:
                best = None
                for c in by_id[uid].get("candidates", []):
                    b = c.get("bbox_xyxy"); t = c.get("text", "")
                    if not b or len(b) != 4 or not re.sub(r"\D", "", str(t)):
                        continue
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    if ((cx - mcx) ** 2 + (cy - mcy) ** 2) ** 0.5 > rad:
                        continue
                    cf = float(c.get("ocr_conf", 0) or 0)
                    if best is None or cf > best[0]:
                        best = (cf, b)
                if best:
                    per_box[uid] = best[1]
                    env = [min(env[0], best[1][0]), min(env[1], best[1][1]),
                           max(env[2], best[1][2]), max(env[3], best[1][3])]
            epx, epy = (env[2] - env[0]) * 0.15, (env[3] - env[1]) * 0.30
            env = [env[0] - epx, env[1] - epy, env[2] + epx, env[3] + epy]   # 이동범위 + 여백
            # 2) 각 프레임 색분리 → 재읽기 → viz. 검출프레임=실제위치, none=이동범위 전체
            n = 0
            for uid in ok_uids:
                im = by_id[uid]
                box = per_box.get(uid, env)                     # 이동 대응
                try:
                    crop = _Img.open(im["image_path"]).convert("RGB").crop(
                        (int(box[0]), int(box[1]), int(box[2]), int(box[3])))
                except Exception:
                    continue
                clean, mask = PD.isolate_contrast(crop, font, bg=bgcol, margin=args.font_isolate_margin)
                new_d, new_c = FCI.rec_read(rec, clean, ftmp)
                old_v = P._dg(field["per_frame"].get(uid, ""))
                imp = bool(new_d) and (not old_v)
                t_imp += int(imp)
                nm = meta.get(uid, uid)
                fi_rows.append({"folder": g, "frame": nm, "old": old_v or "none",
                                "new": new_d, "new_conf": round(new_c, 3), "was_none_now_read": imp})
                if n < args.font_isolate_viz:
                    clean.save(cleandir / f"{nm}.png")
                    PD.make_viz(crop, mask, clean, font, bgcol,
                                text=f"{old_v or 'none'}->{new_d}({new_c:.2f})").save(vizdir / f"{nm}.jpg")
                    n += 1
            print(f"[font-isolate] {g}: 폰트색={entry['font_color']} 배경색={entry['bg_color']} viz {n}장", flush=True)
        with (out / "font_isolate_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["folder", "frame", "old", "new", "new_conf", "was_none_now_read"])
            w.writeheader(); w.writerows(fi_rows)
        shutil.rmtree(ftmp, ignore_errors=True)
        print(f"[font-isolate] none이었다가 색분리로 읽힌 프레임 {t_imp}개 → {out}/font_isolate_viz/, "
              f"font_isolate_summary.csv", flush=True)

    # === (선택) ROI crop full-OCR 재읽기 ===
    if args.force_read:
        field_lbl = out / "field_labels"; sub = out / "unread_imgs"
        for dd in (field_lbl, sub):
            shutil.rmtree(dd, ignore_errors=True); dd.mkdir(parents=True, exist_ok=True)
        unread = []
        for g, (entry, field) in per_folder.items():
            if not field:
                continue
            bx = field["median_box"]
            for uid in sorted(seqs.get(g, [])):
                if uid in by_id and uid not in field["per_frame"]:
                    im = by_id[uid]; W = float(im.get("image_width") or 1280) or 1280
                    H = float(im.get("image_height") or 720) or 720
                    cx = ((bx[0] + bx[2]) / 2) / W; cy = ((bx[1] + bx[3]) / 2) / H
                    bw = (bx[2] - bx[0]) / W; bh = (bx[3] - bx[1]) / H
                    (field_lbl / f"{uid}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    ip = im.get("image_path")
                    if ip:
                        link = sub / Path(ip).name
                        if not (link.exists() or link.is_symlink()):
                            try:
                                os.symlink(Path(ip).resolve(), link)
                            except Exception:
                                pass
                    unread.append(uid)
        if unread:
            if args.force_answer:
                # rec-only(det 생략) 강제 읽기 → none 제거. 파인튜닝 rec 사용.
                rd = Path(rec_dir) if rec_dir.lower() not in ("", "none") else None
                if rd is not None and not rd.is_absolute():
                    rd = Path(__file__).resolve().parent / rd
                rr = [PY, f"{SRC}/rec_only_read.py", "--images", sub, "--yolo-label-dir", field_lbl,
                      "--out", out / "field_read.json", "--pad", "0.2", "--min-height", "120",
                      "--progress-every", "200"]
                if rd is not None and (rd / "inference.pdiparams").exists():
                    rr += ["--rec-model-dir", str(rd)]
                if args.color_isolate:
                    rr += ["--color-isolate", "--color-isolate-viz", out / "color_isolate_viz"]
                    print(f"[color-isolate] 폰트색 학습→숫자만 남기고 재읽기, 시각화 → "
                          f"{out}/color_isolate_viz/", flush=True)
                rc = P.sh(rr, env)
            else:
                rc = P.sh([PY, f"{SRC}/fullocr_crops.py", "--images", sub, "--yolo-label-dir", field_lbl,
                           "--out", out / "field_read.json", "--pad", "0.2", "--min-height", "48",
                           "--progress-every", "200"], env)
            if rc == 0 and (out / "field_read.json").exists():
                fr = json.loads((out / "field_read.json").read_text())
                readval = {}
                for im in fr.get("images", []):
                    for c in im.get("candidates", []):
                        d = P._dg(c.get("text", ""))
                        if d and 1 <= len(d) <= 5:
                            readval[im["image_id"]] = d; break
                filled = 0
                for g, (entry, field) in per_folder.items():
                    if not field:
                        continue
                    for uid in seqs.get(g, []):
                        if uid in readval and uid not in field["per_frame"]:
                            field["per_frame"][uid] = readval[uid]; filled += 1
                print(f"[force-read/full-OCR] ROI 재읽기 {filled}프레임 (unread {len(unread)}중)", flush=True)
                # crop→pad→확대→읽기 스텝 시각화
                if args.viz_force_read > 0:
                    from PIL import Image as _I
                    cnt = {}
                    for g, (entry, field) in per_folder.items():
                        if not field:
                            continue
                        for uid in sorted(seqs.get(g, [])):
                            if uid in readval and cnt.get(g, 0) < args.viz_force_read:
                                try:
                                    img = _I.open(by_id[uid]["image_path"]).convert("RGB")
                                except Exception:
                                    continue
                                ui = _safe(Path(g).name if g not in ("", "(root)") else root.name, root)
                                step_viz.force_read_steps(
                                    img, field["median_box"],
                                    out / "force_read_viz" / ui / meta.get(uid, uid), readval[uid])
                                cnt[g] = cnt.get(g, 0) + 1
                    print(f"[viz-force-read] crop→pad→확대 저장 → {out}/force_read_viz/", flush=True)
        shutil.rmtree(sub, ignore_errors=True)

    # === 출력 ===
    cols = ["folder", "frame", "channel_number"] + (["channel_number_1", "channel_number_2"] if args.both_channel else [])
    rows, per_folder3 = [], {}
    for g, (entry, field) in per_folder.items():
        frows = []
        if field:
            for uid, val in sorted(field["per_frame"].items()):
                r = {"folder": g, "frame": meta.get(uid, uid), "channel_number": val}
                if args.both_channel:
                    r["channel_number_1"] = field.get("per_frame_1", {}).get(uid, "")
                    r["channel_number_2"] = field.get("per_frame_2", {}).get(uid, "")
                rows.append(r); frows.append(r)
        per_folder3[g] = (entry, frows, field)
    with (out / "per_frame.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    (out / "profile_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    if args.split_output:
        gcols = [c for c in cols if c != "folder"]
        used = {}
        for g, (entry, frows, _f) in per_folder3.items():
            nm = _safe(Path(g).name if g not in ("", "(root)") else root.name, root)
            while nm in used and used[nm] != g:
                nm += "_x"
            used[nm] = g
            gd = out / nm; gd.mkdir(parents=True, exist_ok=True)
            (gd / "profile_report.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2))
            with (gd / "per_frame.csv").open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=gcols); w.writeheader()
                for r in frows:
                    w.writerow({c: r.get(c, "") for c in gcols})

    # === 정성 이미지 (성공 N + 실패 전부, UI별 폴더) ===
    if not args.no_qualitative:
        try:
            from PIL import Image, ImageDraw
        except Exception:
            Image = None
        if Image is not None:
            norm = lambda s: str(int(s)) if s else ""
            saved, usedq = 0, {}
            for g, (entry, frows, field) in per_folder3.items():
                if not field:
                    continue
                box = [int(v) for v in field["median_box"]]; pf = field["per_frame"]
                allu = [u for u in sorted(seqs.get(g, [])) if u in by_id]
                oks, fails = [], []
                for uid in allu:
                    pred = P._dg(pf.get(uid, ""))
                    if args.gt_from_filename:
                        gt = P.gt_from_name(meta.get(uid, uid))
                        isf = (not pred) or (gt and norm(pred) != norm(gt))
                    else:
                        isf = not pred
                    (fails if isf else oks).append(uid)
                step = max(1, len(oks) // max(1, args.samples_per_folder))
                draw = [(u, False) for u in oks[::step][:args.samples_per_folder]] + [(u, True) for u in fails]
                if not draw:
                    continue
                nm = _safe(Path(g).name if g not in ("", "(root)") else root.name, root)
                while nm in usedq and usedq[nm] != g:
                    nm += "_x"
                usedq[nm] = g
                qd = (out / nm / "qualitative") if args.split_output else (out / "qualitative" / _safe(g, root))
                fdir = qd / "_failures"; qd.mkdir(parents=True, exist_ok=True)
                if fails:
                    fdir.mkdir(parents=True, exist_ok=True)
                for uid, isf in draw:
                    try:
                        img = Image.open(by_id[uid].get("image_path")).convert("RGB")
                    except Exception:
                        continue
                    d = ImageDraw.Draw(img); val = pf.get(uid, "") or "(none)"
                    col = (255, 0, 0) if isf else (0, 200, 0)
                    d.rectangle(box, outline=col, width=3)
                    lab = f"ch:{val}"
                    if isf and args.gt_from_filename:
                        lab += f" (gt:{P.gt_from_name(meta.get(uid, uid))})"
                    d.text((box[0], max(0, box[1] - 14)), lab, fill=col)
                    img.save((fdir if isf else qd) / f"{meta.get(uid, uid)}.jpg", quality=88)
                    saved += 1
            print(f"정성 이미지 {saved}장 (성공 샘플 + _failures/ 실패 전부)", flush=True)

    print("\n채널 필드(폴더별):")
    for r in report:
        dm = f"  +듀얼{len(r['dual_boxes'])}" if r["dual_boxes"] else ""
        print(f"  {r['folder']:<40} field={r['channel_field_box']}  score={r['field_score']}{dm}")

    if args.gt_from_filename:
        norm = lambda s: str(int(s)) if s else ""
        pbu = {}
        for g, (entry, frows, field) in per_folder3.items():
            if field:
                for uid, val in field["per_frame"].items():
                    pbu[uid] = P._dg(val)
        per = {}
        for g, uids in seqs.items():
            for uid in uids:
                if uid not in by_id:
                    continue
                gt = P.gt_from_name(meta.get(uid, uid))
                if not gt:
                    continue
                t = per.setdefault(g, [0, 0, 0]); t[0] += 1
                pr = pbu.get(uid, "")
                if pr:
                    t[1] += 1
                    if norm(pr) == norm(gt):
                        t[2] += 1
        tot = sum(v[0] for v in per.values()); rd = sum(v[1] for v in per.values()); co = sum(v[2] for v in per.values())
        pc = lambda a, b: round(a / b * 100, 1) if b else 0
        print("\n=== 폴더별 정확도 ===")
        print(f"  {'folder':<40}{'e2e%':>7}{'cov%':>7}{'읽었을때%':>10}{'correct/total':>16}")
        for g in sorted(per):
            a, b, c = per[g]
            print(f"  {g:<40}{pc(c,a):>7}{pc(b,a):>7}{pc(c,b):>10}{f'{c}/{a}':>16}")
        print(f"  {'-'*40}\n  {'전체':<40}{pc(co,tot):>7}{pc(rd,tot):>7}{pc(co,rd):>10}{f'{co}/{tot}':>16}")

    if args.viz_steps > 0:
        step_viz.render(by_id, seqs, per_folder3, meta, out, args.viz_steps,
                        args.gt_from_filename, lambda s: _safe(s, root), min_conf=args.min_conf)

    print(f"\n결과: {out}/per_frame.csv , profile_report.json  ({len(rows)}개)")
    if not args.keep_staged:
        shutil.rmtree(flat, ignore_errors=True)


if __name__ == "__main__":
    main()
