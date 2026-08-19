#!/usr/bin/env bash
# ==============================================================================
# conda 없이 순수 python venv 로 slot_v3 실행 환경 만들기
#
# 사용법:
#   ./setup_venv.sh          # GPU (CUDA 12.6)
#   CPU=1 ./setup_venv.sh    # GPU 없는 서버
#   WITH_TRAIN=1 ./setup_venv.sh   # 재학습 의존성까지
#
# 전제: python 3.10~3.12 가 시스템에 있어야 함 (python3 --version 으로 확인).
#       root 없이 동작하도록 opencv 는 headless 로 강제(libGL 불필요).
# 완료 후:
#   source .venv/bin/activate
#   PYTHON=$(which python) ./run_slot_v3_ft.sh <이미지폴더> results/ft_run
# ==============================================================================
set -euo pipefail

PY_BIN="${PY_BIN:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "[venv] python: $($PY_BIN --version)"
$PY_BIN -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel

# 1) PaddlePaddle (GPU 기본 / CPU=1 이면 CPU)
if [ "${CPU:-0}" = "1" ]; then
  python -m pip install "paddlepaddle==3.3.1"
else
  python -m pip install "paddlepaddle-gpu==3.3.1" \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
fi

# 2) PaddleOCR + PaddleX(OCR)
python -m pip install "paddleocr==3.7.0" "paddlex[ocr]==3.7.1"

# 3) opencv 를 headless 로 강제 → 시스템 libGL 없이도 import 됨 (root 불필요)
python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless 2>/dev/null || true
python -m pip install "opencv-contrib-python-headless"

# 4) (선택) 재학습 의존성
if [ "${WITH_TRAIN:-0}" = "1" ]; then
  python -m pip install scikit-image==0.26.0 imgaug==0.4.0 albumentations==2.0.8 \
                        rapidfuzz==3.14.5 lmdb==2.3.0 shapely pyclipper
fi

export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo ""
echo "=== 설치 확인 ==="
python -c "import cv2, paddle, paddleocr; print('cv2', cv2.__version__); print('paddle', paddle.__version__, 'cuda', paddle.is_compiled_with_cuda()); print('paddleocr', paddleocr.__version__)"
echo ""
echo "완료. 사용:"
echo "  source .venv/bin/activate"
echo "  export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True"
echo "  PYTHON=\$(which python) ./run_slot_v3_ft.sh <이미지폴더> results/ft_run"
