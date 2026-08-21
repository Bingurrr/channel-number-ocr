#!/usr/bin/env python3
"""score_term_ablation — slot_v3/v4 선택 점수의 '각 항이 얼마나 기여하는가'를 실측.

두 가지를 낸다.
  ① 레버리지(배수)  : 정답을 맞힌 슬롯에서 각 항이 실제로 몇 배를 곱했나 (분포)
  ② 임팩트(정확도)  : 그 항을 중립(=1.0)으로 껐을 때 E2E 정확도가 얼마나 떨어지나

PASS1(슬롯 클러스터링)은 점수와 무관하므로 폴더당 한 번만 돌리고,
랭킹 + PASS2(교차읽기)만 모드별로 반복한다 → OCR 캐시 위에서 수 초.

  python dump_ocr_cache.py --out ocr_cache.json          # 선행 1회 (GPU)
  python score_term_ablation.py --cache ocr_cache.json --out ablation_out
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import slot_v3 as V3
import slot_v4 as V4

WINDOW = 24
BAND = 0.05
SIZE_LO, SIZE_HI = 0.6, 1.7
MIN_PRESENT = 2
CONF_THR = 0.3
BG_WINDOW = 5


# ───────────────────────── PASS 1 (모드 무관, 폴더당 1회) ─────────────────────────

def pass1(frames, use_visual):
    """slot_v4.rolling_analyze의 PASS1을 그대로 재현. use_visual이면 bg/대비도 샘플."""
    mem = max(2, WINDOW)
    slots, pre_all = [], []
    for i, fr in enumerate(frames):
        img = V4._load(fr.get("image_path")) if use_visual else None
        cands = V3.preprocess_frame(fr, CONF_THR)
        if use_visual:
            for c in cands:
                c["bg"] = V4._sample_bg(img, c["box"])
                c["con"] = V4._sample_contrast(img, c["box"])
        pre_all.append(cands)
        agreed = V3.within_frame_agreed(cands)
        cur = {}
        for c in cands:
            s = V3._assign(slots, c, False, BAND, SIZE_LO, SIZE_HI)
            V3._update(s, c, i, agreed, mem)
            if use_visual and c.get("bg") is not None:
                bg = s.setdefault("bgrecent", [])
                bg.append((i, c["bg"]))
                while bg and bg[0][0] < i - BG_WINDOW + 1:
                    bg.pop(0)
            if use_visual and c.get("con") is not None:
                cr = s.setdefault("conrecent", [])
                cr.append((i, c["con"]))
                while cr and cr[0][0] < i - BG_WINDOW + 1:
                    cr.pop(0)
            k = id(s)
            if k not in cur or c["conf"] > cur[k][1]:
                cur[k] = (c["value"], c["conf"], c["box"], s)
        for _k, (v, cf, bx, s) in cur.items():
            s.setdefault("pf", {})[i] = (v, cf, bx)
    return slots, pre_all


# ───────────────────────── 점수 항 분해 ─────────────────────────

TERMS = ["cov", "base", "purity", "h_cv", "aspect", "twoloc"]


def terms_of(s, i):
    """_score_persistent를 항별로 분해. 곱하면 원래 점수."""
    present = len(s["present"])
    if present == 0 or s["count"] == 0:
        return None
    distinct = len(s["vals"])
    hs = [e[3] for e in s["recent"]]
    h_cv = (st.pstdev(hs) / (sum(hs) / len(hs))) if len(hs) > 1 and sum(hs) > 0 else 0.0
    aspect = s["asum"] / s["count"]
    return {
        "cov": present / max(1, i + 1),
        "base": (1.0 + 0.2 * distinct) if distinct >= 2 else 0.1,
        "purity": 0.9 + 0.1 * (s["psum"] / s["count"]),
        "h_cv": 0.4 if h_cv > 0.30 else 1.0,
        "aspect": 1.0 if 0.2 <= aspect <= 8.0 else 0.2,
        "twoloc": (1.0 + 0.6 * min(s["twoloc"], 6)) if s["twoloc"] > 0 else 1.0,
        "_distinct": distinct, "_h_cv_raw": h_cv, "_aspect_raw": aspect,
        "_twoloc_n": s["twoloc"], "_present": present,
    }


def score_with(s, i, off=()):
    """off에 든 항을 1.0(중립)으로 바꾼 점수."""
    t = terms_of(s, i)
    if t is None:
        return 0.0
    v = 1.0
    for k in TERMS:
        v *= 1.0 if k in off else t[k]
    return v


# ───────────────────────── v4 시각 보너스 ─────────────────────────

def v4_bonus(elig, primary, weights):
    """slot_v4의 tie-break 보너스(①크기 ②bg ③대비)를 weights대로 계산."""
    def rep_digits(s):
        if not s["vals"]:
            return 0
        v = max(s["vals"], key=lambda x: s["vals"][x])
        return len("".join(ch for ch in str(v) if ch.isdigit()))

    def slot_bg(s):
        bg = s.get("bgrecent", [])
        if not bg:
            return None
        return (st.median([e[1][0] for e in bg]), st.median([e[1][1] for e in bg]),
                st.median([e[1][2] for e in bg]))

    def slot_con(s):
        cr = s.get("conrecent", [])
        return st.median([e[1] for e in cr]) if cr else None

    bonus = defaultdict(float)
    top_score = max(primary.values()) if primary else 0.0
    contenders = [s for s in elig if rep_digits(s) >= 2 and primary[id(s)] >= 0.85 * (top_score + 1e-9)]
    if len(contenders) >= 2:
        if weights["size"]:
            hts = {id(s): s["mh"] for s in contenders if s["mh"] > 0}
            if len(hts) >= 2:
                oh = sorted(hts.values(), reverse=True)
                if oh[0] >= 1.3 * (oh[1] + 1e-9):
                    bonus[max(hts, key=hts.get)] += weights["size"]
        if weights["bg"]:
            bgs = {id(s): slot_bg(s) for s in contenders if slot_bg(s) is not None}
            if len(bgs) >= 2:
                bl = list(bgs.values())
                ref = tuple(st.median([b[k] for b in bl]) for k in range(3))
                dd = {sid: V4._cdist(b, ref) for sid, b in bgs.items()}
                od = sorted(dd.values(), reverse=True)
                if od[0] >= 12.0 and od[0] >= 1.5 * (od[1] + 1e-9):
                    bonus[max(dd, key=dd.get)] += weights["bg"]
        if weights["contrast"]:
            cons = {id(s): slot_con(s) for s in contenders if slot_con(s) is not None}
            if len(cons) >= 2:
                oc = sorted(cons.values(), reverse=True)
                if oc[0] >= 25.0 and oc[0] >= 1.4 * (oc[1] + 1e-9):
                    bonus[max(cons, key=cons.get)] += weights["contrast"]

    mhs = [s["mh"] for s in elig if s["mh"] > 0 and rep_digits(s) >= 2]
    gmax, gmin = (max(mhs), min(mhs)) if mhs else (0.0, 0.0)

    def prom(s):
        if gmax <= gmin or rep_digits(s) < 2:
            return 0.0
        return (s["mh"] - gmin) / (gmax - gmin)

    return bonus, prom


# ───────────────────────── 랭킹 + PASS2 (모드별) ─────────────────────────

def run_mode(slots, pre_all, ids, n, off=(), weights=None):
    """off 항을 끈 점수로 랭킹 → 그룹 → PASS2. per_frame 예측 반환."""
    elig = [s for s in slots if len(s["present"]) >= MIN_PRESENT]
    if not elig:
        return {}, None
    primary = {id(s): score_with(s, n - 1, off) for s in elig}
    if weights:
        bonus, prom = v4_bonus(elig, primary, weights)
        gw = weights.get("prom", 0.0)
        rank = {id(s): primary[id(s)] * (1.0 + bonus[id(s)] + gw * prom(s)) for s in elig}
    else:
        rank = primary
    ranked = sorted(elig, key=lambda s: -rank[id(s)])
    top = ranked[0]
    group = [top] + [s for s in ranked[1:] if V3._agree(top, s)]
    locs = [(s["cx"], s["cy"], s["mh"], rank[id(s)]) for s in group]

    per_frame = {}
    for i, cands in enumerate(pre_all):
        reads = []
        for c in cands:
            for li, (lx, ly, lh, lsc) in enumerate(locs):
                if lh > 0 and not (SIZE_LO <= c["h"] / lh <= SIZE_HI):
                    continue
                if ((c["cx"] - lx) ** 2 + (c["cy"] - ly) ** 2) ** 0.5 > BAND:
                    continue
                reads.append((c["value"], c["conf"], lsc))
                break
        if not reads:
            continue
        weight = defaultdict(float)
        for v, cf, lsc in reads:
            weight[v] += (lsc + 0.01) * cf
        per_frame[ids[i]] = max(weight, key=weight.get)
    return per_frame, top


def norm(s):
    s = str(s)
    return str(int(s)) if s.isdigit() and s != "" else s


def acc_of(per_frame, ids, gts):
    tot = cor = read = 0
    for i, g in zip(ids, gts):
        if not g:
            continue
        tot += 1
        p = re.sub(r"\D", "", str(per_frame.get(i, "")))
        if p:
            read += 1
            if norm(p) == norm(g):
                cor += 1
    return tot, read, cor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="ocr_cache.json")
    ap.add_argument("--out", default="ablation_out")
    ap.add_argument("--folders", type=int, default=0)
    args = ap.parse_args()

    cache = json.loads(Path(args.cache).read_text())
    names = sorted(cache)
    if args.folders:
        names = names[:args.folders]

    V4W = {"size": 0.4, "bg": 0.3, "contrast": 0.2, "prom": 0.5}
    modes = [("v3 full", (), None)]
    modes += [(f"v3 -{t}", (t,), None) for t in TERMS]
    modes += [("v3 all-off", tuple(TERMS), None)]
    modes += [(f"v3 only {t}", tuple(x for x in TERMS if x != t), None) for t in TERMS]
    modes += [("v4 full", (), dict(V4W))]
    for k in ("size", "bg", "contrast", "prom"):
        w = dict(V4W); w[k] = 0.0
        modes.append((f"v4 -{k}", (), w))
    modes += [("v4 all-off", (), {"size": 0.0, "bg": 0.0, "contrast": 0.0, "prom": 0.0})]
    for k in ("size", "bg", "contrast", "prom"):
        w = {"size": 0.0, "bg": 0.0, "contrast": 0.0, "prom": 0.0}; w[k] = V4W[k]
        modes.append((f"v4 only {k}", (), w))

    agg = {m[0]: [0, 0, 0] for m in modes}
    per_ui = defaultdict(dict)
    leverage = []       # 정답 슬롯의 항별 값
    allslots = []       # 전체 슬롯(채널+distractor) 항별 값
    win_rows = []       # UI별 1위/2위 분해

    for fi, name in enumerate(names):
        d = cache[name]
        frames, ids, gts = d["frames"], d["ids"], d["gts"]
        n = len(frames)
        slots, pre_all = pass1(frames, use_visual=True)

        for mname, off, w in modes:
            pf, top = run_mode(slots, pre_all, ids, n, off, w)
            t, r, c = acc_of(pf, ids, gts)
            a = agg[mname]; a[0] += t; a[1] += r; a[2] += c
            per_ui[name][mname] = round(c / max(1, t) * 100, 1)

        # ── 레버리지: '정답을 가장 많이 맞히는 슬롯'(=진짜 채널 슬롯)의 항별 값 ──
        elig = [s for s in slots if len(s["present"]) >= MIN_PRESENT]
        gtset = {norm(g) for g in gts if g}
        best_s, best_hit = None, -1
        for s in elig:
            hit = sum(cnt for v, cnt in s["vals"].items() if norm(v) in gtset)
            if hit > best_hit:
                best_hit, best_s = hit, s
        for s_ in elig:                       # distractor 쪽 항 분포 (페널티가 실제로 발동하나)
            tv = terms_of(s_, n - 1)
            if tv:
                tv = dict(tv); tv["ui"] = name
                tv["role"] = "channel" if s_ is best_s else "distractor"
                allslots.append(tv)
        if best_s is not None and best_hit > 0:
            tv = terms_of(best_s, n - 1)
            if tv:
                tv = dict(tv); tv["ui"] = name; tv["role"] = "channel"
                leverage.append(tv)

        # ── 1위 vs 2위 분해 (어느 항이 순위를 갈랐나) ──
        primary = {id(s): score_with(s, n - 1) for s in elig}
        rk = sorted(elig, key=lambda s: -primary[id(s)])
        if len(rk) >= 2:
            t1, t2 = terms_of(rk[0], n - 1), terms_of(rk[1], n - 1)
            if t1 and t2:
                row = {"ui": name, "s1": round(primary[id(rk[0])], 3),
                       "s2": round(primary[id(rk[1])], 3),
                       "top_is_channel": rk[0] is best_s}
                for k in TERMS:
                    row[f"r_{k}"] = round(t1[k] / max(1e-9, t2[k]), 2)
                win_rows.append(row)
        print(f"[{fi+1}/{len(names)}] {name}  v3={per_ui[name]['v3 full']:5.1f}%  "
              f"v4={per_ui[name]['v4 full']:5.1f}%", flush=True)

    outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    base3 = agg["v3 full"][2] / max(1, agg["v3 full"][0]) * 100
    base4 = agg["v4 full"][2] / max(1, agg["v4 full"][0]) * 100
    summary = []
    for mname, _o, _w in modes:
        t, r, c = agg[mname]
        a = c / max(1, t) * 100
        ref = base4 if mname.startswith("v4") else base3
        summary.append({"mode": mname, "acc": round(a, 1),
                        "delta": round(a - ref, 1),
                        "read": round(r / max(1, t) * 100, 1), "frames": t})
    (outd / "summary.json").write_text(json.dumps(
        {"summary": summary, "per_ui": per_ui, "leverage": leverage, "top_vs_2nd": win_rows, "all_slots": allslots},
        ensure_ascii=False, indent=1))

    print("\n" + "=" * 74)
    print(f"{'mode':<16}{'E2E acc':>9}{'Δ':>8}{'read':>8}")
    print("-" * 74)
    for r in summary:
        print(f"{r['mode']:<16}{r['acc']:>8.1f}%{r['delta']:>+8.1f}{r['read']:>7.1f}%")
    print("=" * 74)

    if leverage:
        print("\n[레버리지] 진짜 채널 슬롯에서 각 항이 곱한 배수 (중앙값 / 최소~최대)")
        for k in TERMS:
            vs = sorted(x[k] for x in leverage)
            print(f"  {k:<8} median {st.median(vs):6.2f}   range {vs[0]:.2f} ~ {vs[-1]:.2f}")
        ds = sorted(x["_distinct"] for x in leverage)
        tw = sorted(x["_twoloc_n"] for x in leverage)
        print(f"  (distinct 값개수 median {st.median(ds)}, twoloc 횟수 median {st.median(tw)})")
    dis = [x for x in allslots if x["role"] == "distractor"]
    ch = [x for x in allslots if x["role"] == "channel"]
    if dis:
        print(f"\n[페널티 발동률] 항이 중립(1.0)이 아닌 슬롯 비율  "
              f"(채널 {len(ch)}개 vs distractor {len(dis)}개)")
        for k in TERMS:
            fc = sum(1 for x in ch if abs(x[k] - 1.0) > 1e-9) / max(1, len(ch)) * 100
            fd = sum(1 for x in dis if abs(x[k] - 1.0) > 1e-9) / max(1, len(dis)) * 100
            mc = st.median([x[k] for x in ch]) if ch else 0
            md = st.median([x[k] for x in dis]) if dis else 0
            print(f"  {k:<8} 채널 {fc:5.1f}% (median {mc:6.2f})   distractor {fd:5.1f}% (median {md:6.2f})")
    print(f"\nsaved {outd/'summary.json'}")


if __name__ == "__main__":
    main()
