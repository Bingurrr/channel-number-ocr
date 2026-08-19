#!/usr/bin/env bash
# ==============================================================================
# channel-ocr conda 환경 생성 스크립트 (slot_v3 실행용)
#
# 사용법:
#   ./setup_env.sh          # GPU (CUDA 12.6) 서버
#   CPU=1 ./setup_env.sh    # GPU 없는 서버 (CPU 전용)
#
# 완료 후:
#   conda activate channel-ocr
#   PYTHON=$(which python) ./run_slot_v3_ft.sh <이미지폴더> results/ft_run
# ==============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-channel-ocr}"
source "$(conda info --base)/etc/profile.d/conda.sh"

# 1) 파이썬 3.12 환경
conda create -y -n "$ENV_NAME" python=3.12
conda activate "$ENV_NAME"

# 2) OpenGL 런타임 (opencv import 시 libGL 필요) — conda로 자족적으로 설치
conda install -y -c conda-forge libgl mesa-libgl-cos7-x86_64 libglib || \
  conda install -y -c conda-forge libgl libglib

# 3) PaddlePaddle 설치 (GPU 기본 / CPU=1 이면 CPU)
if [ "${CPU:-0}" = "1" ]; then
  python -m pip install "paddlepaddle==3.3.1"
else
  python -m pip install "paddlepaddle-gpu==3.3.1" \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
fi

# 4) PaddleOCR + PaddleX(OCR)
python -m pip install "paddleocr==3.7.0" "paddlex[ocr]==3.7.1"

# 5) (선택) 재학습까지 하려면 필요한 학습 의존성. 추론만 하면 생략 가능.
if [ "${WITH_TRAIN:-0}" = "1" ]; then
  python -m pip install scikit-image==0.26.0 imgaug==0.4.0 albumentations==2.0.8 \
                        rapidfuzz==3.14.5 lmdb==2.3.0 shapely pyclipper
fi

# 6) 런타임 라이브러리 경로 (libgomp/libstdc++ 등) — 매 실행 시 필요
conda env config vars set LD_LIBRARY_PATH="$(conda info --base)/envs/${ENV_NAME}/lib:${LD_LIBRARY_PATH:-}"
conda env config vars set PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True"

echo ""
echo "=== 설치 확인 ==="
conda deactivate; conda activate "$ENV_NAME"   # env vars 반영
python -c "import paddle, paddleocr; print('paddle', paddle.__version__, 'cuda', paddle.is_compiled_with_cuda()); print('paddleocr', paddleocr.__version__)"
echo ""
echo "완료. 사용:  conda activate ${ENV_NAME} && PYTHON=\$(which python) ./run_slot_v3_ft.sh <이미지폴더> results/ft_run"
