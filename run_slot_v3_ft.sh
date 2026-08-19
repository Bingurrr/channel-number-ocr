#!/usr/bin/env bash
# ==============================================================================
# slot_v3 실행 스크립트 (파인튜닝된 det + rec 사용)
#
# 파인튜닝 모델:
#   det : models/full_image_ocr/det_overlay_frozen_v1   (backbone+neck freeze, head만 학습)
#   rec : models/full_image_ocr/rec_overlay_frozen_v1   (backbone freeze, neck+head 학습, 영어+숫자)
#
# 사용법:
#   ./run_slot_v3_ft.sh <이미지_루트폴더> [출력폴더]
# 예:
#   ./run_slot_v3_ft.sh /path/to/frames results/ft_run
#
# 다른 서버 준비물:
#   - conda 환경(paddlepaddle + paddleocr). PYTHON 환경변수로 인터프리터 지정 가능.
#       예) PYTHON=/opt/conda/envs/channel-ocr/bin/python ./run_slot_v3_ft.sh ...
#   - 순정 det(PP-OCRv4_mobile_det)는 최초 실행 시 PaddleOCR가 자동 다운로드하나,
#     여기서는 --det-model-dir 로 파인튜닝 det를 쓰므로 다운로드 불필요.
# ==============================================================================
set -euo pipefail

ROOT="${1:?사용법: ./run_slot_v3_ft.sh <이미지_루트폴더> [출력폴더]}"
OUT="${2:-results/slot_v3_ft}"
PYTHON="${PYTHON:-python}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

DET_DIR="models/full_image_ocr/det_overlay_frozen_v1"
REC_DIR="models/full_image_ocr/rec_overlay_frozen_v1"

echo "[run] python = $PYTHON"
echo "[run] det    = $DET_DIR"
echo "[run] rec    = $REC_DIR"
echo "[run] root   = $ROOT"
echo "[run] out    = $OUT"

"$PYTHON" predict_folder_slot_v3.py \
  --root "$ROOT" \
  --out "$OUT" \
  --det-model-dir "$DET_DIR" \
  --rec-model-dir "$REC_DIR" \
  --window 24 \
  --min-conf 0.3 \
  "${@:3}"

echo "[run] 완료 → $OUT (per_frame.csv, profile_report.json)"
