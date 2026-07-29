#!/usr/bin/env python3
"""Temporal-overlap ROI finder (diagnostic / visualization).

Idea: the channel number (and the whole UI banner) is STATIC across consecutive
frames, while the video behind it changes. So over N frames:
  * per-pixel std  -> LOW where the UI/number sits, HIGH where video moves
  * per-pixel mean -> the static UI stays sharp, the video blurs out

This tool, for each folder of consecutive frames, takes N frames and writes:
  mean.jpg          the averaged frame (static UI sharp, video blurred)
  stdmap.jpg        std heatmap (dark = static = candidate UI/number region)
  static.jpg        the mean, but only where std is low (isolated UI overlay)
  roi_overlay.jpg   original frame with proposed static-text ROI boxes (green)

Look at stdmap/static: if the channel number pops out as a clear dark/sharp
region, the temporal signal works on your data and we can wire it into the
pipeline (ROI proposal and/or 5-frame-average clean crop for OCR).

Usage:
    python temporal_overlap_roi.py --root /path/to/frames_folder --out /out
    python temporal_overlap_roi.py --root /path/with/subfolders --out /out   # recursive
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_frames(paths, size):
    gray, color = [], []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB").resize(size)
        except Exception:
            continue
        a = np.asarray(im, dtype=np.float32)
        color.append(a)
        gray.append(a.mean(axis=2))
    return np.stack(gray), np.stack(color)


def edge_map(gray):
    gy, gx = np.gradient(gray)
    e = np.hypot(gx, gy)
    return e / (e.max() + 1e-6)


def propose_boxes(mask, min_w=8, min_h=6, max_h_frac=0.12, H=1):
    """Very light connected-component boxing via row/col runs on a binary mask."""
    boxes = []
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return boxes
    # coarse grid clustering: label by flood over a downsampled mask
    from collections import deque
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    idx = np.argwhere(mask)
    idxset = set(map(tuple, idx))
    for (y, x) in idx:
        if visited[y, x]:
            continue
        q = deque([(y, x)]); visited[y, x] = True
        comp = [(y, x)]
        while q:
            cy, cx = q.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and mask[ny, nx]:
                        visited[ny, nx] = True; q.append((ny, nx)); comp.append((ny, nx))
        cy = [c[0] for c in comp]; cx = [c[1] for c in comp]
        x1, y1, x2, y2 = min(cx), min(cy), max(cx), max(cy)
        bw, bh = x2 - x1, y2 - y1
        if bw >= min_w and bh >= min_h and bh <= max_h_frac * h and len(comp) >= 15:
            boxes.append((x1, y1, x2, y2))
    return boxes


def process(frames, out_dir, n, size):
    if len(frames) < 2:
        return None
    sel = frames[:: max(1, len(frames) // n)][:n] if len(frames) > n else frames
    gray, color = load_frames(sel, size)
    if gray.shape[0] < 2:
        return None
    std = gray.std(axis=0)                       # 낮음=정적(UI), 높음=영상
    mean = color.mean(axis=0)
    std_n = std / (std.max() + 1e-6)
    edges = edge_map(mean.mean(axis=2))
    # 정적(std 낮음) & 텍스트다움(엣지 높음) = 채널번호/UI 텍스트 후보
    static_low = std_n < np.percentile(std_n, 35)
    text_like = edges > np.percentile(edges, 80)
    mask = static_low & text_like

    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mean.astype(np.uint8)).save(out_dir / "mean.jpg", quality=90)
    # std heatmap: 어두울수록 정적
    heat = (std_n * 255).astype(np.uint8)
    Image.fromarray(heat).save(out_dir / "stdmap.jpg")
    # static: std 낮은 곳만 mean, 나머지는 회색
    static_img = mean.copy()
    static_img[~static_low] = 128
    Image.fromarray(static_img.astype(np.uint8)).save(out_dir / "static.jpg", quality=90)
    # roi overlay: 첫 프레임 위에 후보 박스
    base = np.asarray(Image.open(sel[0]).convert("RGB").resize(size)).copy()
    boxes = propose_boxes(mask)
    for (x1, y1, x2, y2) in boxes:
        base[y1:y1 + 2, x1:x2] = [0, 255, 0]; base[y2 - 1:y2 + 1, x1:x2] = [0, 255, 0]
        base[y1:y2, x1:x1 + 2] = [0, 255, 0]; base[y1:y2, x2 - 1:x2 + 1] = [0, 255, 0]
    Image.fromarray(base).save(out_dir / "roi_overlay.jpg", quality=90)
    return len(sel), len(boxes)


def collect_folders(root):
    """Return {folder_name: [frame paths]} — each dir containing images is a sequence."""
    groups = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            groups.setdefault(p.parent, []).append(p)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="연속 프레임 폴더 (또는 하위폴더들)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=5, help="겹칠 프레임 수 (기본 5)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    size = (args.width, args.height)
    groups = collect_folders(root)
    if not groups:
        raise SystemExit(f"이미지 없음: {root}")
    print(f"{len(groups)}개 시퀀스(폴더) 처리, 각 {args.frames}프레임 겹침\n")
    done = 0
    for folder, paths in sorted(groups.items()):
        name = folder.name if folder != root else "(root)"
        od = out / name.replace("/", "__")
        r = process(sorted(paths), od, args.frames, size)
        if r:
            used, nb = r
            print(f"  {name}: {used}프레임 겹침, ROI후보 {nb}개 -> {od}")
            done += 1
    print(f"\n완료: {done}개 폴더. 각 폴더의 stdmap.jpg / static.jpg / roi_overlay.jpg 확인")
    print("  stdmap: 어두운 곳=정적(UI). 채널번호가 뚜렷이 어둡게 나오면 성공")


if __name__ == "__main__":
    main()
