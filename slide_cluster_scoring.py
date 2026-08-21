#!/usr/bin/env python3
"""PPT 한 장: '④ 클러스터 점수화 — 어느 클러스터가 채널인가'.
왼쪽=프레임 속 후보 클러스터, 오른쪽=신호별 점수표, 아래=점수 공식."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib import font_manager as fm

FP = "/home1/irteam/teacher_model/assets/google_fonts/ofl/nanumgothic/NanumGothic-Bold.ttf"
fm.fontManager.addfont(FP)
KF = fm.FontProperties(fname=FP).get_name()
plt.rcParams["font.family"] = KF
plt.rcParams["axes.unicode_minus"] = False

NAVY, GREEN, GREY, RED, BG = "#22304A", "#2E9E5B", "#8A93A3", "#D6392E", "#F4F6F9"
fig = plt.figure(figsize=(16, 9), dpi=120)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")

def rbox(x, y, w, h, fc, ec="none", lw=0, r=0.12, z=1, alpha=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha,
                                mutation_aspect=1))

# ── 제목 ──
rbox(0, 8.15, 16, 0.85, NAVY, r=0)
ax.text(0.45, 8.57, "④ 클러스터 점수화 — 어느 클러스터가 '채널'인가?", color="white",
        fontsize=23, va="center", ha="left")
ax.text(15.55, 8.57, "slot_v3 · 선택(selection)", color="#AEB8CC", fontsize=13, va="center", ha="right")

# ══════════ 왼쪽: 프레임 속 후보 클러스터 ══════════
fx, fy, fw, fh = 0.55, 1.5, 5.6, 5.9
rbox(fx, fy, fw, fh, "#1B2430", r=0.15)                 # TV 화면
ax.text(fx + fw/2, fy + fh + 0.28, "여러 프레임 누적된 후보 클러스터", fontsize=14.5,
        ha="center", color=NAVY)

# 각 클러스터: (x,y,w,h, 라벨, 값예시, 색)
clusters = [
    (fx+0.35, fy+0.55, 1.7, 0.95, "A 채널후보", "271", GREEN),      # 좌하단 큰 숫자
    (fx+3.7, fy+fh-1.15, 1.5, 0.6, "B 시계/라벨", "20:31", GREY),   # 우상단
    (fx+2.2, fy+2.5, 1.2, 0.55, "C 잡숫자", "3:02", GREY),         # 중앙 가끔
]
for cx, cy, cw, ch, lab, val, col in clusters:
    ax.add_patch(Rectangle((cx, cy), cw, ch, fill=False, ec=col, lw=2.6, zorder=4))
    ax.text(cx + cw/2, cy + ch/2, val, color="white", fontsize=17 if col==GREEN else 12,
            ha="center", va="center", zorder=5)
    ax.text(cx, cy - 0.28, lab, color=col, fontsize=12.5, ha="left", va="center", zorder=5)

# ══════════ 오른쪽: 점수표 ══════════
tx, tw = 6.7, 8.9
cols = ["클러스터", "값다양성\n(distinct)", "커버리지", "높이\n일관성", "2곳\n일치", "→ 점수"]
cw = [1.9, 1.5, 1.35, 1.15, 1.0, 2.0]
rows = [
    ("A 채널후보", "8 (자주 바뀜)", "0.95", "O", "O ×2.2", "5.43", True),
    ("B 시계/라벨", "1 (거의 고정)", "0.90", "O", "X", "0.09", False),
    ("C 잡숫자",    "3", "0.30", "X ×0.4", "X", "0.19", False),
]
rh = 1.15; top = 7.35
# 헤더
xacc = tx
ax.add_patch(Rectangle((tx, top), tw, 0.95, fc=NAVY, ec="none", zorder=2))
for c, w in zip(cols, cw):
    ax.text(xacc + w/2, top + 0.48, c, color="white", fontsize=12.5, ha="center", va="center", zorder=3)
    xacc += w
# 데이터 행
for ri, (name, dv, cov, hh, tl, sc, win) in enumerate(rows):
    ry = top - (ri + 1) * rh
    fc = "#E7F5EC" if win else ("white" if ri % 2 else BG)
    ax.add_patch(Rectangle((tx, ry), tw, rh, fc=fc, ec="#DCE1E8", lw=1, zorder=1))
    if win:
        ax.add_patch(Rectangle((tx, ry), tw, rh, fc="none", ec=GREEN, lw=2.6, zorder=3))
    vals = [name, dv, cov, hh, tl, (sc + "  ★" if win else sc)]
    xacc = tx
    for ci, (v, w) in enumerate(zip(vals, cw)):
        col = NAVY; fs = 12.5; fw_ = "normal"
        if ci == 0:
            col = GREEN if win else "#38414F"
        if ci == 5:
            col = GREEN if win else GREY; fs = 18 if win else 14
        if "X" in str(v):
            col = RED
        ax.text(xacc + w/2, ry + rh/2, v, color=col, fontsize=fs, ha="center", va="center", zorder=4)
        xacc += w
    if win:
        ax.text(tx + 0.2, ry + rh - 0.02, "채널 선택", color="white", fontsize=10.5,
                ha="left", va="top", zorder=5,
                bbox=dict(boxstyle="round,pad=0.22", fc=GREEN, ec="none"))

# 화살표: 프레임 A → 표 A행
ax.add_patch(FancyArrowPatch((fx+2.1, fy+1.0), (tx-0.15, top - rh + 0.55),
             arrowstyle="-|>", mutation_scale=22, color=GREEN, lw=2.2, zorder=6,
             connectionstyle="arc3,rad=-0.15"))

# ══════════ 아래: 점수 공식 ══════════
rbox(0.55, 0.35, 15.0, 0.95, BG, ec="#D3D9E2", lw=1.2, r=0.1)
ax.text(0.85, 0.83, "점수 =", color=NAVY, fontsize=15, va="center", ha="left")
formula = ("커버리지 × (1 + 0.2×값다양성) × (0.9+0.1×순도)   "
           "× [높이흔들림 ×0.4]  × [가로세로비 이상 ×0.2]  × [2곳일치 ×(1+0.6·n)]")
ax.text(2.35, 0.83, formula, color="#38414F", fontsize=13.2, va="center", ha="left")
ax.text(0.85, 0.2, "핵심: 채널은 '값이 자주 바뀌고(다양성)·꾸준히 뜨고(커버리지)·글자높이 일정'한 클러스터 → 최고점",
        color=GREEN, fontsize=11.5, va="center", ha="left")

fig.savefig("/tmp/claude-500/-home1-irteam/53ded4c1-f246-489d-aea8-0f9713428fd0/scratchpad/slide_cluster_scoring.png",
            dpi=120, facecolor="white")
print("saved slide_cluster_scoring.png")
