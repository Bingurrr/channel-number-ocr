"""slot_v4 예측 시각화 — 프레임마다 예측 채널박스에 초록 굵은 bbox + 좌상단 Pred/GT/O·X.

기존 full_ocr.json(OCR 결과) + 원본 이미지 폴더로 slot_v4를 UI별로 돌려 시각화 저장.
사용:
  python viz_slot_v4.py --ocr <full_ocr.json> --img-root <UI폴더들 상위> --out <출력>
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

import slot_v4 as V4

FONT = None
for fp in ["/home1/irteam/teacher_model/assets/google_fonts/ofl/nanumgothic/NanumGothic-Bold.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
    if Path(fp).exists():
        FONT = fp; break


def gt_of(uid):
    m = re.match(r"0*(\d+)", uid.rsplit("__", 1)[1])
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", required=True, help="full_ocr.json")
    ap.add_argument("--img-root", required=True, help="UI 폴더들 상위 (원본 이미지)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-ui", type=int, default=0, help="UI당 저장 장수 제한(0=전부)")
    ap.add_argument("--uis", default="", help="쉼표구분 UI만 (예: UI_39,UI_36,UI_02,UI_07)")
    ap.add_argument("--zoom", action="store_true", help="예측 박스 주변만 크롭·확대해서 저장")
    args = ap.parse_args()
    want = set(x.strip() for x in args.uis.split(",") if x.strip())

    d = json.load(open(args.ocr))
    groups = defaultdict(list)
    for im in d["images"]:
        groups[im["image_id"].rsplit("__", 1)[0]].append(im)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(args.img_root)

    tot_ok = tot = 0
    for ui in sorted(groups):
        if want and ui not in want:
            continue
        ims = sorted(groups[ui], key=lambda im: im["image_id"]); ids = [im["image_id"] for im in ims]
        pr = V4.rolling_analyze(ims, ids, window=24)
        pf = pr["per_frame"] if pr else {}
        pb = pr.get("per_frame_box", {}) if pr else {}
        uidir = out / ui; uidir.mkdir(parents=True, exist_ok=True)
        n_ok = n = 0
        for im in ims:
            uid = im["image_id"]; stem = uid.rsplit("__", 1)[1]
            g = gt_of(uid)
            pred = pf.get(uid, ""); predn = "".join(c for c in str(pred) if c.isdigit())
            correct = bool(predn and g and str(int(predn)) == str(int(g)))
            n += 1; n_ok += correct
            if args.per_ui and n > args.per_ui:
                continue
            src = root / ui / f"{stem}.jpg"
            if not src.exists():
                src = root / ui / f"{stem}.png"
            if not src.exists():
                continue
            img = Image.open(src).convert("RGB"); W, H = img.size
            box = pb.get(uid)
            mark = "O" if correct else "X"
            txt = f"Pred: {predn or 'none'}  GT: {g}  Correct: {mark}"
            lab_fill = (0, 255, 0) if correct else (255, 80, 80)

            if args.zoom and box:
                # 전체 이미지 + 예측박스 주변을 확대한 '돋보기 인셋'을 위에 오버레이
                x1, y1, x2, y2 = [int(v) for v in box]
                dr = ImageDraw.Draw(img)
                for w in range(4):                                # 원본 박스(초록)
                    dr.rectangle([x1-w, y1-w, x2+w, y2+w], outline=(0, 255, 0))
                bw, bh = x2 - x1, y2 - y1
                mx = max(int(bw * 0.30), 22); my = max(int(bh * 0.45), 16)  # 여백 최소화 → 확대↑
                cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
                cx2, cy2 = min(W, x2 + mx), min(H, y2 + my)
                crop = img.crop((cx1, cy1, cx2, cy2))
                tw = int(W * 0.45)                                # 인셋 폭 = 화면 45%
                sc = tw / max(1, crop.width)
                ins = crop.resize((tw, max(1, int(crop.height * sc))))
                di = ImageDraw.Draw(ins)
                bx1, by1 = (x1-cx1)*sc, (y1-cy1)*sc; bx2, by2 = (x2-cx1)*sc, (y2-cy1)*sc
                for w in range(4):
                    di.rectangle([bx1-w, by1-w, bx2+w, by2+w], outline=(0, 255, 0))
                for w in range(4):                                # 인셋 테두리(노랑)
                    di.rectangle([w, w, ins.width-1-w, ins.height-1-w], outline=(255, 220, 0))
                # 배치: 아래쪽, 박스 반대편
                bcx = (x1 + x2) / 2 / W
                px = 12 if bcx >= 0.5 else W - ins.width - 12
                py = H - ins.height - 12
                img.paste(ins, (px, py))
                f = ImageFont.truetype(FONT, 28) if FONT else ImageFont.load_default()
                l, t, r, b = f.getbbox(txt)
                dr.rectangle([8, 8, 8+(r-l)+16, 8+(b-t)+14], fill=(0, 0, 0))
                dr.text((16, 12), txt, font=f, fill=lab_fill)
                img.save(uidir / f"{stem}.jpg", quality=92)
            else:
                dr = ImageDraw.Draw(img)
                if box:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    for w in range(4):
                        dr.rectangle([x1-w, y1-w, x2+w, y2+w], outline=(0, 255, 0))
                f = ImageFont.truetype(FONT, 26) if FONT else ImageFont.load_default()
                l, t, r, b = f.getbbox(txt)
                dr.rectangle([8, 8, 8+(r-l)+16, 8+(b-t)+14], fill=(0, 0, 0))
                dr.text((16, 12), txt, font=f, fill=lab_fill)
                img.save(uidir / f"{stem}.jpg", quality=90)
        tot_ok += n_ok; tot += n
        print(f"  {ui}: {n_ok}/{n}", flush=True)
    print(f"\n전체 {tot_ok}/{tot} = {tot_ok/max(1,tot)*100:.1f}%  → {out}")


if __name__ == "__main__":
    main()
