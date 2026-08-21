#!/usr/bin/env python3
"""Detection 파인튜닝 영향 그래프: 전체 90.6% → 81% (대부분 UI 유지·미세향상, 일부 급락).

before = 실측 per-UI 정확도(파인튜닝 전 결과, 92개 provider/device).
after  = 사용자 제시 실측 aggregate(90.6→81)와 관찰된 패턴('일부 특정 UI만 급락')에 맞춘 도식.
         → 실제 det-ft per-UI 파일을 주면 그대로 교체 가능.
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

FP = "/home1/irteam/teacher_model/assets/google_fonts/ofl/nanumgothic/NanumGothic-Bold.ttf"
fm.fontManager.addfont(FP)
KF = fm.FontProperties(fname=FP).get_name()
plt.rcParams["font.family"] = KF
plt.rcParams["axes.unicode_minus"] = False

# ── 실측 before 파싱 ──
rows = []
for ln in open("/home/irteam/파인튜닝 전 결과", encoding="utf-8").read().splitlines():
    m = re.match(r"\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+(\d+)/(\d+)\s*$", ln)
    if m and "folder" not in ln:
        rows.append((m.group(1), float(m.group(2)), int(m.group(5)), int(m.group(6))))
before = np.array([r[1] for r in rows], float)
tot = np.array([r[3] for r in rows], float)
N = len(rows)
w = tot / tot.sum()
before_agg = float((before * w).sum())          # ≈ 90.4 (실측)

# ── after 도식: 대부분 미세향상 + 소수(≈10) 급락, 프레임가중 평균 = 81.0 에 맞춤 ──
TARGET = 81.0
rng = np.random.default_rng(7)
after = np.clip(before + rng.uniform(0.0, 2.2, N), 0, 100)   # 대부분 유지/미세 향상
# 급락 대상 = 원래 정확도 높고(>88) 프레임 중간규모(150~950)인 UI들을 '날카롭게' 급락시켜
#            그 대상집합의 프레임 가중치로 전체=81 을 맞춘다(=일부 UI가 전체를 끌어내림).
cand = [i for i in np.argsort(-tot) if before[i] > 88 and 150 <= tot[i] <= 950]
dropf = {i: float(before[i] * rng.uniform(0.25, 0.42)) for i in cand}   # →25~42% 급락
drop_idx = []
for i in cand:
    drop_idx.append(i)
    tmp = after.copy()
    for j in drop_idx:
        tmp[j] = dropf[j]
    if float((tmp * w).sum()) <= TARGET:
        break
for j in drop_idx:                                          # 급락 확정
    after[j] = dropf[j]
gap = float((after * w).sum()) - TARGET                     # 마지막 UI로 정확히 81 미세조정
if abs(gap) > 0.03 and drop_idx:
    j = drop_idx[-1]
    after[j] = float(np.clip(after[j] - gap / w[j], 0, before[j]))
drop_set = set(drop_idx)
after_agg = float((after * w).sum())
print(f"N={N}  before_agg={before_agg:.1f}  after_agg={after_agg:.1f}  급락 UI={len(drop_idx)}개")

# ── 정렬: before 내림차순 → 익명 UI 인덱스 ──
si = np.argsort(-before)
b, a = before[si], after[si]
is_drop = np.array([si[k] in drop_set for k in range(N)])
x = np.arange(1, N + 1)

# ── 플롯 ──
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8.6), height_ratios=[2.15, 1],
                               gridspec_kw={"hspace": 0.32})
C_B, C_A, C_D, C_UP = "#2E5FA3", "#E1812C", "#D6392E", "#3AA757"

# (1) per-UI before/after
ax1.plot(x, b, "-", color=C_B, lw=1.4, alpha=.55, zorder=1)
ax1.scatter(x, b, s=26, color=C_B, zorder=3, label="파인튜닝 전 (Detection stock)")
ax1.scatter(x[~is_drop], a[~is_drop], s=24, color=C_A, zorder=3,
            label="파인튜닝 후 (Detection fine-tuned)")
ax1.scatter(x[is_drop], a[is_drop], s=95, color=C_D, edgecolor="k", lw=.6, zorder=5,
            marker="v", label="성능 급락 UI")
for k in np.where(is_drop)[0]:                       # 급락 화살표
    ax1.annotate("", xy=(x[k], a[k]), xytext=(x[k], b[k]),
                 arrowprops=dict(arrowstyle="-|>", color=C_D, lw=1.5, alpha=.8), zorder=4)
ax1.axhline(before_agg, ls="--", color=C_B, lw=1.3, alpha=.8)
ax1.axhline(after_agg, ls="--", color=C_A, lw=1.3, alpha=.8)
ax1.text(N + 0.5, before_agg, f" 전체 {before_agg:.1f}%", color=C_B, va="center", fontsize=11, fontweight="bold")
ax1.text(N + 0.5, after_agg, f" 전체 {after_agg:.1f}%", color=C_A, va="center", fontsize=11, fontweight="bold")
ax1.set_title("Detection 파인튜닝 영향 — 전체 90.6% → 81% (대부분 UI 유지·미세 향상, 일부 특정 UI만 급락)",
              fontsize=14.5, fontweight="bold", pad=12)
ax1.set_ylabel("채널번호 정확도 (%)", fontsize=11.5)
ax1.set_xlabel(f"UI 종류 (n={N}, 정확도 내림차순 정렬)", fontsize=11)
ax1.set_xlim(0, N + 7); ax1.set_ylim(-3, 105)
ax1.grid(axis="y", ls=":", alpha=.4)
ax1.legend(loc="lower left", fontsize=10.5, framealpha=.95, ncol=2)

# (2) per-UI Δ (after-before) 정렬 막대
d = a - b
sd = np.argsort(d)
xb = np.arange(N)
cols = [C_D if d[i] < -5 else (C_UP if d[i] > 0.3 else "#B7BCC4") for i in sd]
ax2.bar(xb, d[sd], color=cols, width=.9)
ax2.axhline(0, color="k", lw=.8)
n_drop = int((d < -5).sum()); n_up = int((d > 0.3).sum()); n_keep = N - n_drop - n_up
ax2.set_title(f"UI별 변화량 (후 - 전)   ·   급락 {n_drop}개 (빨강) / 미세향상 {n_up}개 (초록) / 유지 {n_keep}개 (회색)",
              fontsize=12.5, fontweight="bold", pad=8)
ax2.set_ylabel("변화량 (%p)", fontsize=11)
ax2.set_xlabel("UI 종류 (변화량 오름차순 정렬)", fontsize=11)
ax2.set_xlim(-1, N); ax2.grid(axis="y", ls=":", alpha=.4)

fig.text(0.008, 0.006,
         "※ before = 실측 per-UI 정확도(92 provider/device). after per-UI = 실측 aggregate(90.6→81%)와 "
         "관찰 패턴('일부 UI만 붕괴')에 맞춘 도식 — 실제 det-ft per-UI 데이터로 교체 가능.",
         fontsize=8.6, color="#666", ha="left")
fig.savefig("/tmp/claude-500/-home1-irteam/53ded4c1-f246-489d-aea8-0f9713428fd0/scratchpad/det_ft_impact.png",
            dpi=155, bbox_inches="tight", facecolor="white")
print("saved det_ft_impact.png")
