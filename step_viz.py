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


def cluster(frames, ids, dist_thr=0.05):
    """Same position clustering as the profiler; returns slots with box/values/type."""
    cl = []
    for fi, im in enumerate(frames):
        W = float(im.get("image_width") or 1280) or 1280
        H = float(im.get("image_height") or 720) or 720
        for c in im.get("candidates", []):
            b = c.get("bbox_xyxy"); t = c.get("text", "")
            if not b or len(b) != 4 or not digits(t):
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


def render(by_id, seqs, per_folder, meta, out, n_each, gt_mode, safe):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        print("[step_viz] PIL 없음 → 건너뜀"); return
    F = _font(20); Ft = _font(26); Fsmall = _font(16)
    norm = lambda s: str(int(s)) if s else ""
    root_out = out / "step_viz"

    for g, (entry, frows, field) in per_folder.items():
        if not field:
            continue
        ui = safe(Path(g).name if g not in ("", "(root)") else "root")
        ok_ids = [u for u in sorted(seqs.get(g, [])) if u in by_id]
        frames = [by_id[u] for u in ok_ids]
        slots = cluster(frames, ok_ids)
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
                _label_box(d, (8, 8), "1) 입력 원본", (255, 255, 0), Ft)
                im1.save(sdir / "1_input.jpg", quality=90)

                # ---- 2) full OCR (모든 텍스트+박스, 크게) ----
                im2 = base.copy(); d = ImageDraw.Draw(im2)
                for c in by_id[uid].get("candidates", []):
                    b = c.get("bbox_xyxy"); t = c.get("text", "")
                    if not b or len(b) != 4:
                        continue
                    d.rectangle([int(v) for v in b], outline=(0, 220, 255), width=3)
                    _label_box(d, (b[0], max(0, b[1] - 20)), str(t)[:14], (0, 220, 255), F)
                _label_box(d, (8, 8), "2) full OCR: 화면 모든 텍스트 (어느게 채널?)", (255, 255, 0), Ft)
                im2.save(sdir / "2_fullocr.jpg", quality=90)

                # ---- 3) slots: 같은 위치끼리 묶기 + 그 자리 프레임별 값 ----
                im3 = base.copy(); d = ImageDraw.Draw(im3)
                for si, s in enumerate(slots):
                    col = SLOTCOL[si % len(SLOTCOL)]
                    # 이 프레임에서 이 슬롯 멤버 박스
                    for m in s["mem"]:
                        if m["uid"] == uid:
                            d.rectangle([int(v) for v in m["box"]], outline=col, width=3)
                    # 슬롯 대표 위치에 프레임별 값 리스트
                    bx = s["box"]
                    vals = ",".join(str(v) for v in s["values"][:6])
                    _label_box(d, (bx[0], max(0, bx[1] - 20)), f"[{vals}]", col, Fsmall)
                _label_box(d, (8, 8), "3) 같은 위치끼리 묶기 (슬롯) + 프레임별 값", (255, 255, 0), Ft)
                im3.save(sdir / "3_slots.jpg", quality=90)

                # ---- 4) classify: 규칙 분류 (제외 이유) ----
                im4 = base.copy(); d = ImageDraw.Draw(im4)
                reason = {"channel": "채널! 값바뀜", "time": "시간(콜론)",
                          "constant": "고정값(로고)", "text": "텍스트(이름)", "other": "기타"}
                for s in slots:
                    lab = s["label"]
                    col = (0, 200, 0) if lab == "channel" else (150, 150, 150)
                    bx = s["box"]
                    d.rectangle([int(v) for v in bx], outline=col,
                                width=4 if lab == "channel" else 2)
                    _label_box(d, (bx[0], max(0, bx[1] - 20)),
                               reason.get(lab, lab), col, F)
                _label_box(d, (8, 8), "4) 규칙 분류: 시간/로고/텍스트 제외", (255, 255, 0), Ft)
                im4.save(sdir / "4_classify.jpg", quality=90)

                # ---- 5) field: 채널 필드 확정 ----
                im5 = base.copy(); d = ImageDraw.Draw(im5)
                val = pf.get(uid, "") or "(none)"
                fbx = [int(v) for v in field["median_box"]]
                okc = digits(pf.get(uid, "")) and gt_mode and \
                    norm(digits(pf.get(uid, ""))) == norm("".join(ch for ch in meta.get(uid, uid) if ch.isdigit())[:5])
                col = (0, 200, 0) if (okc or (not gt_mode and digits(pf.get(uid, "")))) else (255, 60, 60)
                d.rectangle(fbx, outline=col, width=5)
                dv = fld_slot["distinct"] if fld_slot else "?"
                _label_box(d, (fbx[0], max(0, fbx[1] - 24)), f"채널={val}", col, Ft)
                _label_box(d, (8, 8),
                           f"5) 채널 필드 확정 (값 다양성={dv} → zap마다 바뀜)", (255, 255, 0), Ft)
                if gt_mode:
                    gtv = "".join(ch for ch in meta.get(uid, uid) if ch.isdigit())[:5]
                    _label_box(d, (8, 40), f"정답={gtv}  예측={digits(val) or '-'}", col, F)
                im5.save(sdir / "5_field.jpg", quality=90)

    print(f"[step_viz] UI별/스텝별 저장 → {root_out}", flush=True)
