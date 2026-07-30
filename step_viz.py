#!/usr/bin/env python3
"""Per-UI, per-step algorithm visualization (tells the story of WHY each step).

For each UI folder, picks N success + N fail example frames and writes, for each,
FIVE separate images (one per algorithm step) into:
    step_viz/<UI>/<success|fail>/set{k}/1_input.jpg .. 5_field.jpg

Steps:
  1 input        원본
  2 full OCR     화면 모든 텍스트+박스 (왜: OCR은 다 읽음 → 어느 게 채널?)
  3 slots        같은 위치끼리 묶기 + 그 자리의 프레임별 값들 (왜: 위치 고정)
  4 classify     규칙 분류 (시간=콜론/로고=값고정/텍스트=글자 제외)
  5 field        채널 필드 확정 (값이 바뀌는 순수숫자)
"""
from __future__ import annotations

import statistics as st
from pathlib import Path

from temporal_profile_select import classify, best_digit, TIME, DATE, digits

TYPECOL = {"channelnum": (0, 200, 0), "time": (70, 130, 255), "date": (180, 110, 255),
           "text": (150, 150, 150), "othernum": (255, 150, 0)}
SLOTCOL = [(230, 60, 60), (60, 180, 230), (240, 170, 40), (150, 90, 230),
           (60, 200, 120), (230, 120, 200), (120, 200, 60), (200, 200, 60)]
FONTS = ["/home/irteam/teacher_model/assets/google_fonts/ofl/gothica1/GothicA1-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def _font(sz):
    from PIL import ImageFont
    for p in FONTS:
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def cluster(frames, ids, dist_thr=0.05, conf_thr=0.5):
    """Same position clustering as the profiler; returns slots with box/values/type.

    conf_thr drops low-confidence OCR false positives (e.g. hallucinated digits on
    dark/empty regions with ocr_conf ~0.1) so they don't clutter the visualization.
    """
    cl = []
    for fi, im in enumerate(frames):
        W = float(im.get("image_width") or 1280) or 1280
        H = float(im.get("image_height") or 720) or 720
        for c in im.get("candidates", []):
            b = c.get("bbox_xyxy"); t = c.get("text", "")
            if not b or len(b) != 4 or not digits(t):
                continue
            if float(c.get("ocr_conf", 1.0) or 0.0) < conf_thr:   # 저신뢰 헛읽음 제외
                continue
            cx, cy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
            best, bd = None, dist_thr
            for s in cl:
                d = ((s["cx"] - cx) ** 2 + (s["cy"] - cy) ** 2) ** 0.5
                if d < bd:
                    bd, best = d, s
            if best is None:
                best = {"cx": cx, "cy": cy, "mem": []}
                cl.append(best)
            best["mem"].append({"frame": fi, "uid": ids[fi], "text": t, "box": b,
                                "type": classify(t), "value": best_digit(t)})
            k = len(best["mem"])
            best["cx"] = (best["cx"] * (k - 1) + cx) / k
            best["cy"] = (best["cy"] * (k - 1) + cy) / k
    for s in cl:
        boxes = [m["box"] for m in s["mem"]]
        s["box"] = [st.median([b[i] for b in boxes]) for i in range(4)]
        chan_vals = [m["value"] for m in s["mem"] if m["type"] == "channelnum" and m["value"]]
        s["distinct"] = len(set(chan_vals))
        s["values"] = [m["value"] or m["text"] for m in s["mem"]]
        types = [m["type"] for m in s["mem"]]
        if any(x in ("time", "date") for x in types):
            s["label"] = "time"
        elif sum(x == "text" for x in types) > len(types) * 0.5:
            s["label"] = "text"
        elif s["distinct"] >= 2:
            s["label"] = "channel"
        elif s["distinct"] == 1:
            s["label"] = "constant"          # 고정 로고
        else:
            s["label"] = "other"
    return cl


def _label_box(draw, xy, text, color, font, fill_bg=(0, 0, 0)):
    x, y = xy
    try:
        bb = draw.textbbox((x, y), text, font=font)
        draw.rectangle([bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1], fill=fill_bg)
    except Exception:
        pass
    draw.text((x, y), text, fill=color, font=font)


def force_read_steps(img, box, sdir, read_val="", pad=0.2, min_height=120):
    """Save crop -> padding -> upscale (-> read) steps for one force-read frame."""
    from PIL import ImageDraw
    sdir.mkdir(parents=True, exist_ok=True)
    F = _font(24)
    b = [int(v) for v in box]
    im1 = img.copy(); d = ImageDraw.Draw(im1)
    d.rectangle(b, outline=(0, 220, 0), width=3)
    _label_box(d, (b[0], max(0, b[1] - 26)), f"ROI h={b[3]-b[1]}px", (0, 220, 0), F)
    _label_box(d, (8, 8), "1) known ROI on full frame", (255, 255, 0), F)
    im1.save(sdir / "1_full_ROI.jpg", quality=90)
    img.crop(tuple(b)).save(sdir / "2_crop_tight.jpg")
    pw = int((b[2] - b[0]) * pad); ph = int((b[3] - b[1]) * pad)
    pbox = (max(0, b[0] - pw), max(0, b[1] - ph), b[2] + pw, b[3] + ph)
    padded = img.crop(pbox); padded.save(sdir / "3_crop_padded.jpg")
    u = max(1.0, min_height / max(1, padded.height))
    up = padded.resize((max(1, int(padded.width * u)), max(1, int(padded.height * u))))
    d = ImageDraw.Draw(up)
    _label_box(d, (4, 4), f"4) upscaled x{u:.1f}" + (f" -> read:{read_val}" if read_val else ""),
               (0, 220, 0), _font(18))
    up.save(sdir / "4_upscaled_read.jpg", quality=90)


def render(by_id, seqs, per_folder, meta, out, n_each, gt_mode, safe):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        print("[step_viz] PIL 없음 → 건너뜀"); return
    F = _font(20); Ft = _font(26); Fsmall = _font(16)
    norm = lambda s: str(int(s)) if s else ""
    root_out = out / "step_viz"
    CONF_THR = 0.5          # 2_fullocr / 4_classify 공통 신뢰도 게이트 (cluster 기본값과 동일)

    for g, (entry, frows, field) in per_folder.items():
        if not field:
            continue
        ui = safe(Path(g).name if g not in ("", "(root)") else "root")
        ok_ids = [u for u in sorted(seqs.get(g, [])) if u in by_id]
        frames = [by_id[u] for u in ok_ids]
        slots = cluster(frames, ok_ids)
        # 실제 파이프라인 결정과 일치시킴: analyze()가 고른 primary(+듀얼)만 진짜 채널.
        # 값이 바뀌어도(distinct>=2) 게이트/점수에서 탈락한 슬롯은 'candidate'로 표시.
        try:
            import slot_analysis as SA
            _prim, _duals, _ = SA.analyze(frames, ok_ids)
            chosen_c = [((m["box"][0] + m["box"][2]) / 2, (m["box"][1] + m["box"][3]) / 2)
                        for m in ([_prim] + _duals if _prim else [])]
        except Exception:
            chosen_c = []
        for s in slots:
            s["chosen"] = False
        # 각 선택 중심마다 '가장 가까운 슬롯 하나'만 채널로 → 다중 초록 방지
        for cx, cy in chosen_c:
            near = min(slots, key=lambda s: ((s["box"][0] + s["box"][2]) / 2 - cx) ** 2
                       + ((s["box"][1] + s["box"][3]) / 2 - cy) ** 2, default=None) if slots else None
            if near is not None:
                dc = ((near["box"][0] + near["box"][2]) / 2 - cx) ** 2 + \
                     ((near["box"][1] + near["box"][3]) / 2 - cy) ** 2
                if dc < (0.03 * 1280) ** 2:
                    near["chosen"] = True
        # 채널 필드 슬롯 = median_box 중심에 가장 가까운 슬롯
        fb = field["median_box"]; fcx = (fb[0] + fb[2]) / 2; fcy = (fb[1] + fb[3]) / 2
        fld_slot = min(slots, key=lambda s: (s["box"][0] + s["box"][2]) / 2 - fcx if False else
                       ((s["box"][0] + s["box"][2]) / 2 - fcx) ** 2 + ((s["box"][1] + s["box"][3]) / 2 - fcy) ** 2) \
            if slots else None
        pf = field["per_frame"]
        # 성공/실패 예시 프레임
        succ, fail = [], []
        for uid in ok_ids:
            pred = digits(pf.get(uid, ""))
            gt = meta.get(uid, uid)
            gtv = "".join(c for c in gt if c.isdigit())[:5] if gt_mode else ""
            ok = (pred and gtv and norm(pred) == norm(gtv)) if gt_mode else bool(pred)
            (succ if ok else fail).append(uid)

        for label, exs in [("success", succ[:n_each]), ("fail", fail[:n_each])]:
            for k, uid in enumerate(exs):
                try:
                    base = Image.open(by_id[uid].get("image_path")).convert("RGB")
                except Exception:
                    continue
                W, H = base.size
                sdir = root_out / ui / label / f"set{k}"
                sdir.mkdir(parents=True, exist_ok=True)

                # ---- 1) input ----
                im1 = base.copy(); d = ImageDraw.Draw(im1)
                _label_box(d, (8, 8), "1) input", (255, 255, 0), Ft)
                im1.save(sdir / "1_input.jpg", quality=90)

                # 이 프레임에서 classify 대상이 되는 박스 = conf 게이트 통과한 것들.
                # 2_fullocr 와 4_classify 가 '동일한 박스 집합'을 쓰도록 여기서 한 번만 필터.
                cand = [c for c in by_id[uid].get("candidates", [])
                        if c.get("bbox_xyxy") and len(c["bbox_xyxy"]) == 4
                        and float(c.get("ocr_conf", 1.0) or 0.0) >= CONF_THR]

                # ---- 2) full OCR (all text boxes) ----
                im2 = base.copy(); d = ImageDraw.Draw(im2)
                for c in cand:
                    b = c["bbox_xyxy"]; t = c.get("text", "")
                    d.rectangle([int(v) for v in b], outline=(0, 220, 255), width=3)
                    _label_box(d, (b[0], max(0, b[1] - 20)), str(t)[:14], (0, 220, 255), F)
                _label_box(d, (8, 8), f"2) full OCR: all text (conf>={CONF_THR})", (255, 255, 0), Ft)
                im2.save(sdir / "2_fullocr.jpg", quality=90)

                # ---- 3) slots: same-position grouping + per-frame values ----
                im3 = base.copy(); d = ImageDraw.Draw(im3)
                for si, s in enumerate(slots):
                    col = SLOTCOL[si % len(SLOTCOL)]
                    for m in s["mem"]:
                        if m["uid"] == uid:
                            d.rectangle([int(v) for v in m["box"]], outline=col, width=3)
                    bx = s["box"]
                    vals = ",".join(str(v) for v in s["values"][:6])
                    _label_box(d, (bx[0], max(0, bx[1] - 20)), f"[{vals}]", col, Fsmall)
                _label_box(d, (8, 8), "3) cluster by position (slots) + values across frames",
                           (255, 255, 0), Ft)
                im3.save(sdir / "3_slots.jpg", quality=90)

                # ---- 4) classify: 2_fullocr 의 '모든 박스'에 규칙 타입 색을 칠함 ----
                # 초록=최종 채널(analyze 선택), 노랑=값바뀌나 게이트탈락, 파랑=시간/날짜,
                # 주황=고정로고, 회색=텍스트/기타. → 2_fullocr 와 동일한 박스 집합.
                TCOL = {"time": (70, 130, 255), "date": (70, 130, 255),
                        "channelnum": (240, 210, 40), "othernum": (170, 170, 170),
                        "text": (170, 170, 170)}
                # 이 프레임에서 '채널로 선택된' 박스들 (chosen 슬롯의 현재 프레임 멤버)
                chan_boxes = []
                for s in slots:
                    if s.get("chosen"):
                        m = next((m for m in s["mem"] if m["uid"] == uid), None)
                        if m:
                            chan_boxes.append(m["box"])
                near = lambda a, b: all(abs(a[i] - b[i]) <= 2 for i in range(4))
                im4 = base.copy(); d = ImageDraw.Draw(im4)
                for c in cand:               # 2_fullocr 와 같은 박스들
                    b = c["bbox_xyxy"]; t = c.get("text", "")
                    if any(near(b, cb) for cb in chan_boxes):
                        col, wd, lab = (0, 210, 0), 5, "CHANNEL"
                    else:
                        col, wd, lab = TCOL.get(classify(t), (170, 170, 170)), 2, None
                    d.rectangle([int(v) for v in b], outline=col, width=wd)
                    if lab:                  # 채널만 글자 라벨(나머지는 색만 → 덜 어지럽게)
                        _label_box(d, (b[0], max(0, b[1] - 20)), lab, col, Fsmall)
                _label_box(d, (8, 8), "4) classify (green=selected channel, others excluded)",
                           (255, 255, 0), Ft)
                lg = [("green=CHANNEL(selected)", (0, 210, 0)),
                      ("yellow=changing but rejected", (240, 210, 40)),
                      ("blue=time", (70, 130, 255)), ("orange=logo(fixed)", (255, 150, 0)),
                      ("gray=text", (170, 170, 170))]
                for li, (lt, lc) in enumerate(lg):
                    _label_box(d, (8, 44 + li * 24), lt, lc, F)
                im4.save(sdir / "4_classify.jpg", quality=90)

                # ---- 5) field: channel field confirmed ----
                im5 = base.copy(); d = ImageDraw.Draw(im5)
                val = pf.get(uid, "") or "(none)"
                fbx = [int(v) for v in field["median_box"]]
                okc = digits(pf.get(uid, "")) and gt_mode and \
                    norm(digits(pf.get(uid, ""))) == norm("".join(ch for ch in meta.get(uid, uid) if ch.isdigit())[:5])
                col = (0, 210, 0) if (okc or (not gt_mode and digits(pf.get(uid, "")))) else (255, 60, 60)
                d.rectangle(fbx, outline=col, width=5)
                dv = fld_slot["distinct"] if fld_slot else "?"
                _label_box(d, (fbx[0], max(0, fbx[1] - 24)), f"channel={val}", col, Ft)
                _label_box(d, (8, 8),
                           f"5) channel field (value diversity={dv})", (255, 255, 0), Ft)
                if gt_mode:
                    gtv = "".join(ch for ch in meta.get(uid, uid) if ch.isdigit())[:5]
                    _label_box(d, (8, 40), f"gt={gtv}  pred={digits(val) or '-'}", col, F)
                im5.save(sdir / "5_field.jpg", quality=90)

    print(f"[step_viz] UI별/스텝별 저장 → {root_out}", flush=True)
