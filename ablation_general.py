#!/usr/bin/env python3
"""범용 점수-항 ablation — 아무 full_ocr.json에나 돌려 각 항의 영향을 본다.

각 설정 = slot_v4.rolling_analyze 의 가중치 일부를 0으로 끈 것.
  v3(시각신호 전부 OFF) → v4(전부 ON) → 항 하나씩 제거 → 새 항(채도 등) 추가.
전체 정확도 + 항별 기여(%p) + UI(폴더)별 CSV 출력.

사용 예:
  # 합성 Test_overlay (staged id = UI_01__0204_xxxx)
  python ablation_general.py --ocr .../testfolder_v4/full_ocr.json --group-sep __ --gt leading
  # 상업용 Airtel (id = Ch012__...)
  python ablation_general.py --ocr .../airtel_ocr/full_ocr.json --group-sep "" --gt ch
"""
from __future__ import annotations
import argparse, csv, json, re, sys
from collections import defaultdict

sys.path.insert(0, ".")
import slot_v4 as V4


def make_gt(mode, regex):
    if regex:
        pat = re.compile(regex)
        return lambda s: (pat.search(s).group(1) if pat.search(s) else None)
    if mode == "ch":                       # Ch012__... → 12
        pat = re.compile(r"(?i)ch0*(\d+)")
    else:                                  # leading: 맨 앞 숫자 (필요시 __ 뒤)
        pat = re.compile(r"0*(\d+)")
    def f(stem):
        tail = stem.split("__")[-1] if mode == "leading" else stem
        m = pat.search(tail if mode == "leading" else stem)
        return m.group(1) if m else None
    return f


def norm(s):
    s = re.sub(r"\D", "", str(s))
    return str(int(s)) if s else ""


# 항별 on/off 설정. base = v4 전부 ON. 새 항은 여기에 한 줄 추가하면 자동 포함.
def configs(base):
    return [
        ("v3  (시각신호 전부 OFF)", {**base, "size_weight": 0, "bg_weight": 0,
                                "contrast_weight": 0, "global_weight": 0, "sat_weight": 0}),
        ("v4  (전부 ON)",         dict(base)),
        ("v4 − 크기",             {**base, "size_weight": 0}),
        ("v4 − 배경하이라이트",      {**base, "bg_weight": 0}),
        ("v4 − 대비",             {**base, "contrast_weight": 0}),
        ("v4 − 전역현저성",         {**base, "global_weight": 0}),
        ("v4 + 채도(0.35)",       {**base, "sat_weight": 0.35}),
        ("채도만 (v3+채도)",        {**base, "size_weight": 0, "bg_weight": 0,
                                "contrast_weight": 0, "global_weight": 0, "sat_weight": 0.35}),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", required=True, help="full_ocr.json 경로")
    ap.add_argument("--group-sep", default="__",
                    help="폴더 그룹 구분자(id에서 이 앞부분이 그룹). 빈 문자열이면 전체 1그룹")
    ap.add_argument("--gt", default="leading", choices=["leading", "ch"], help="GT 추출 방식")
    ap.add_argument("--gt-regex", default="", help="직접 정규식(그룹1=채널). 주면 --gt 무시")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--out", default="ablation_general.csv")
    args = ap.parse_args()

    gt_of = make_gt(args.gt, args.gt_regex)
    d = json.load(open(args.ocr))
    ims_all = d["images"]
    groups = defaultdict(list)
    for im in ims_all:
        g = im["image_id"].split(args.group_sep)[0] if args.group_sep else "ALL"
        groups[g].append(im)
    print(f"이미지 {len(ims_all)} · 그룹 {len(groups)} · window {args.window}", flush=True)

    base = dict(size_weight=0.4, bg_weight=0.3, contrast_weight=0.2, global_weight=0.5, sat_weight=0.0)
    CFG = configs(base)

    def run(weights):
        per = {}; tot = [0, 0]
        for g in sorted(groups):
            ims = sorted(groups[g], key=lambda x: x["image_id"]); ids = [x["image_id"] for x in ims]
            r = V4.rolling_analyze(ims, ids, window=args.window, **weights)
            pf = r["per_frame"] if r else {}
            c = t = 0
            for i in ids:
                gt = gt_of(i)
                if not gt:
                    continue
                t += 1
                if norm(pf.get(i, "")) == norm(gt):
                    c += 1
            per[g] = (c, t); tot[0] += c; tot[1] += t
        return per, tot

    print("=" * 64)
    res = {}
    for name, w in CFG:
        per, tot = run(w); res[name] = (per, tot)
        print(f"  {name:<26} {tot[0]/max(1,tot[1])*100:5.1f}%  ({tot[0]}/{tot[1]})", flush=True)
    print("=" * 64)

    # 항별 기여 = v4 − (그 항 제거)
    v4 = res["v4  (전부 ON)"][1]; v4a = v4[0]/max(1, v4[1])*100
    print("각 항 기여 (v4 − 항제거, +면 그 항이 도움):")
    for label, key in [("크기", "v4 − 크기"), ("배경하이라이트", "v4 − 배경하이라이트"),
                       ("대비", "v4 − 대비"), ("전역현저성", "v4 − 전역현저성")]:
        t = res[key][1]; a = t[0]/max(1, t[1])*100
        print(f"  {label:<12}: {v4a-a:+.1f}%p")
    v3a = res["v3  (시각신호 전부 OFF)"][1]
    print(f"  {'시각 tie-break 전체':<12}: {v4a - v3a[0]/max(1,v3a[1])*100:+.1f}%p  (v3→v4)")

    # 그룹별 CSV
    uis = sorted(res["v4  (전부 ON)"][0])
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["group", "n"] + [n for n, _ in CFG])
        for g in uis:
            n = res["v4  (전부 ON)"][0][g][1]
            row = [g, n]
            for name, _ in CFG:
                c, t = res[name][0][g]
                row.append(round(c/max(1, t)*100, 1))
            w.writerow(row)
    print(f"\n그룹별 표 → {args.out}")


if __name__ == "__main__":
    main()
