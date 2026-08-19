#!/usr/bin/env python3
"""slot_rank_debug — slot_v3가 '왜 그 슬롯을 채널이라고 골랐는지' 점수를 분해해서 보여준다.

OCR을 다시 돌리지 않는다. 끝난 run의 full_ocr.json만 읽어서 slot_v3의 PASS 1
(슬롯 클러스터링 + 랭킹)을 그대로 재현하고, 후보 슬롯마다 점수 구성요소를 출력한다.

_score_persistent 의 분해:
    score = cov × base × (0.9 + 0.1×purity)
            × (0.4 if h_cv>0.30)
            × (0.2 if aspect 벗어남)
            × (1 + 0.6×min(twoloc,6))
  cov      등장프레임/전체        (lifetime)
  base     distinct>=2 ? 1+0.2×distinct : 0.1   (lifetime, 값 다양성)
  twoloc   같은 값이 한 프레임 2곳에 나온 횟수  (lifetime)
  conflict 2곳일치 값이 있는데 이 슬롯 값은 달랐던 횟수 (**현재 점수에 미반영**)
  h_cv     높이 변동계수          (최근 window개)

사용:
    python slot_rank_debug.py --result results/ft_run --folder UI_39
    python slot_rank_debug.py --result results/ft_run --folder UI_39 --window 5
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import slot_v3 as V3


def uid_prefix(folder):
    return folder.replace("/", "__").replace(" ", "_") + "__"


def build_slots(frames, window, by_height, band, conf_thr, size_lo=0.6, size_hi=1.7):
    """slot_v3.rolling_analyze 의 PASS 1 을 그대로 재현."""
    mem = max(2, window)
    slots, pre_all = [], []
    for i, fr in enumerate(frames):
        cands = V3.preprocess_frame(fr, conf_thr)
        pre_all.append(cands)
        agreed = V3.within_frame_agreed(cands)
        cur = {}
        for c in cands:
            s = V3._assign(slots, c, by_height, band, size_lo, size_hi)
            V3._update(s, c, i, agreed, mem)
            k = id(s)
            if k not in cur or c["conf"] > cur[k][1]:
                cur[k] = (c["value"], c["conf"], c["box"], s)
        for _k, (v, cf, bx, s) in cur.items():
            s.setdefault("pf", {})[i] = (v, cf, bx)
    return slots, pre_all


def breakdown(s, n):
    """_score_persistent 를 항목별로 분해."""
    present = len(s["present"])
    if present == 0 or s["count"] == 0:
        return None
    distinct = len(s["vals"])
    base = (1.0 + 0.2 * distinct) if distinct >= 2 else 0.1
    purity = s["psum"] / s["count"]
    hs = [e[3] for e in s["recent"]]
    h_cv = (st.pstdev(hs) / (sum(hs) / len(hs))) if len(hs) > 1 and sum(hs) > 0 else 0.0
    cov = present / max(1, n)
    aspect = s["asum"] / s["count"]
    m_h = 0.4 if h_cv > 0.30 else 1.0
    m_a = 0.2 if not (0.2 <= aspect <= 8.0) else 1.0
    m_t = (1.0 + 0.6 * min(s["twoloc"], 6)) if s["twoloc"] > 0 else 1.0
    return {"present": present, "cov": cov, "distinct": distinct, "base": base,
            "purity": purity, "h_cv": h_cv, "aspect": aspect,
            "twoloc": s["twoloc"], "conflict": s["conflict"],
            "m_h": m_h, "m_a": m_a, "m_t": m_t,
            "score": cov * base * (0.9 + 0.1 * purity) * m_h * m_a * m_t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="predict_folder_slot_v3 --out 디렉토리")
    ap.add_argument("--folder", required=True, help="폴더 이름 (부분일치, 예: UI_39)")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--band", type=float, default=0.05)
    ap.add_argument("--by-height", action="store_true")
    ap.add_argument("--top", type=int, default=12, help="출력할 상위 슬롯 수")
    ap.add_argument("--min-present", type=int, default=2)
    args = ap.parse_args()

    res = Path(args.result)
    doc = json.loads((res / "full_ocr.json").read_text())
    by_id = {im["image_id"]: im for im in doc.get("images", [])}

    folders = [f["folder"] for f in json.loads((res / "profile_report.json").read_text())] \
        if (res / "profile_report.json").exists() else []
    # profile_report 가 dict 형태(crop_v3)면 folders 키 안에 있음
    if not folders and (res / "profile_report.json").exists():
        rep = json.loads((res / "profile_report.json").read_text())
        folders = [f["folder"] for f in rep.get("folders", [])]
    match = [g for g in folders if args.folder in g] or [args.folder]
    if len(match) > 1:
        print(f"[주의] '{args.folder}' 에 여러 폴더가 매칭: {match} → 첫 번째 사용")
    folder = match[0]

    pre = uid_prefix(folder)
    uids = sorted(u for u in by_id if u.startswith(pre))
    if not uids:
        raise SystemExit(f"'{folder}' 에 해당하는 프레임 없음 (uid 접두사 '{pre}')")
    frames = [by_id[u] for u in uids]
    n = len(frames)
    print(f"폴더: {folder}   프레임 {n}장   window={args.window} band={args.band} "
          f"min-conf={args.min_conf}\n")

    slots, _ = build_slots(frames, args.window, args.by_height, args.band, args.min_conf)
    elig = [s for s in slots if len(s["present"]) >= args.min_present]
    rows = []
    for s in elig:
        b = breakdown(s, n)
        if b:
            b["slot"] = s
            rows.append(b)
    rows.sort(key=lambda r: -r["score"])

    hdr = (f"{'#':>2} {'score':>8} {'cx':>6} {'cy':>6} {'mh':>6} {'present':>8} {'cov':>6} "
           f"{'dist':>5} {'base':>5} {'twoloc':>7} {'conf!':>6} {'h_cv':>6} {'×h':>4} {'×a':>4} {'×2loc':>6}")
    print(hdr); print("-" * len(hdr))
    for i, r in enumerate(rows[:args.top], 1):
        s = r["slot"]
        print(f"{i:>2} {r['score']:>8.3f} {s['cx']:>6.3f} {s['cy']:>6.3f} {s['mh']:>6.3f} "
              f"{r['present']:>8} {r['cov']:>6.3f} {r['distinct']:>5} {r['base']:>5.1f} "
              f"{r['twoloc']:>7} {r['conflict']:>6} {r['h_cv']:>6.3f} "
              f"{r['m_h']:>4.1f} {r['m_a']:>4.1f} {r['m_t']:>6.2f}")

    print("\n=== 상위 슬롯이 실제로 본 값들 (빈도순) ===")
    for i, r in enumerate(rows[:min(args.top, 6)], 1):
        vals = sorted(r["slot"]["vals"].items(), key=lambda kv: -kv[1])[:8]
        bx = [p[2] for p in r["slot"].get("pf", {}).values()]
        med = [round(st.median([b[k] for b in bx]), 1) for k in range(4)] if bx else None
        print(f"  #{i} score={r['score']:.3f} box={med}")
        print(f"      값: {', '.join(f'{v}×{c}' for v, c in vals)}")

    if rows:
        top = rows[0]["slot"]
        group = [top] + [r["slot"] for r in rows[1:] if V3._agree(top, r["slot"])]
        print(f"\n=== 최종 채널 위치 그룹: {len(group)}곳 ===")
        for s in group:
            print(f"  cx={s['cx']:.3f} cy={s['cy']:.3f} mh={s['mh']:.3f} twoloc={s['twoloc']}")

        r0, r1 = rows[0], (rows[1] if len(rows) > 1 else None)
        print("\n=== 1등이 이긴 이유 ===")
        if r1:
            gap = r0["score"] / max(1e-9, r1["score"])
            print(f"  1등 {r0['score']:.3f} vs 2등 {r1['score']:.3f}  ({gap:.1f}배)")
            for k, lab in (("cov", "커버리지"), ("base", "값다양성 base"),
                           ("m_t", "2곳일치 배수"), ("m_h", "높이일관성 배수"),
                           ("m_a", "aspect 배수")):
                a, b = r0[k], r1[k]
                if abs(a - b) > 1e-9:
                    ratio = a / max(1e-9, b)
                    print(f"    {lab:<16} {a:>7.3f} vs {b:>7.3f}   → {ratio:>5.2f}배 기여")
        if r0["twoloc"] == 0:
            print("  ※ twoloc=0 → '2곳 일치' 신호 없이 이김. "
                  "커버리지와 값다양성만으로 결정된 케이스.")
        if r0["conflict"] > 0:
            print(f"  ※ conflict={r0['conflict']} — 현재 점수식에 미반영(계산만 되고 안 쓰임)")


if __name__ == "__main__":
    main()
