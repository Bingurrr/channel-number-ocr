# Channel Number OCR

TV channel-number reader: **detector → full-image OCR → numeric OCR (aspect-padded) → selector → temporal**.
Reads the current channel number from EPG/overlay screenshots.

## Models (included, ~26MB total)
| model | path | notes |
|---|---|---|
| Detector (YOLO11n) | `models/detector/best.pt` | 40-UI trained, channel_number class weighted ×4. P 99.8 / R 98.4 / mAP50 99.4 |
| Numeric OCR (PP-OCRv5 mobile) | `models/numeric_ocr/` | digits only, trained on 3:1 aspect-padded crops |
| Full-image OCR (PP-OCRv4) | `models/full_image_ocr/` | frozen |
| Selector | `models/selector/` | pairwise linear + value-group + relative gate |

Detection classes: `channel_number(0)`, `other_number(1)`, `other_text(2)`, `channel_number_area(3)`.

## Install
```bash
pip install -r requirements.txt        # install a CUDA-matched torch/paddle first
# if paddle needs OpenMP:  export LD_LIBRARY_PATH=/path/to/conda/lib:$LD_LIBRARY_PATH
```

## Evaluate a folder (each subfolder = one UI; images inside are frames)
```
ROOT/
  UI_A/  123.jpg  20.jpg  ...    # all frames of the SAME UI/channel
  UI_B/  1.jpg    99.jpg  ...
```
```bash
python predict_folder.py --root /path/to/ROOT --out /path/to/output
```
Outputs:
- `per_folder.csv` — one channel number per subfolder (majority vote over frames)
- `per_frame.csv`  — per-frame prediction

Frames of one subfolder are treated as a temporal sequence (channel position is
fixed per UI), so accumulation across frames stabilises the result.

## Benchmark (our synthetic test set, 40 UIs)
E2E exact accuracy: random 92.8% / narrow 78.9% / advs 90.6% / **all 88.5%**.

## Notes
- The numeric OCR requires 3:1 aspect padding on short numbers (baked into the
  padded recheck); without it 1-digit accuracy collapses (~10% → ~90%).
- `config.json` holds all paths; edit `python`/`pipeline_src` for your environment.
# channel-number-ocr

## Troubleshooting (clone한 새 서버에서)

**`ImportError: libGL.so.1`** — 헤드리스 서버에 그래픽 라이브러리가 없을 때. 둘 중 하나:
```bash
# (a) headless OpenCV 사용 (권장, root 불필요)
pip uninstall -y opencv-python
pip install opencv-python-headless

# (b) 시스템 라이브러리 설치 (root면)
apt-get update && apt-get install -y libgl1 libglib2.0-0
```

**`ModuleNotFoundError: No module named 'ocr_candidate_extractor'`** (또는 다른 src 모듈)
→ 최신 버전을 다시 받으세요: `git pull` (누락 의존 모듈이 src/에 포함됨).

**paddle `libgomp.so.1` 없음**
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # 또는 conda가 설치된 lib 경로
```

**config.json** — 배포 기본은 패키지 상대경로입니다. 다른 위치의 모델을 쓰려면
`detector` / `numeric_ocr` / `selector_dir` / `pipeline_src` 경로를 절대경로로 바꾸세요.
`python`은 사용하는 파이썬(예: `.venv/bin/python`)으로.

**`AttributeError: 'AnalysisConfig' object has no attribute 'set_optimization_level'`**
→ paddle / paddleocr / paddlex 버전 불일치. 모델이 빌드된 버전으로 맞추세요:
```bash
pip install "paddleocr==3.4.1" "paddlex==3.4.3"
# paddle 3.x GPU 휠은 PyPI에 없음 → paddle 인덱스에서 CUDA 맞춰 설치:
nvidia-smi | grep "CUDA Version"
pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/  # CUDA 12.x
# CUDA 11.8이면 cu118, CPU면: pip install paddlepaddle==3.3.1
python check_env.py                      # 버전/모델 확인
```

**`image_count: 0`** → 이미지를 못 찾음. predict_folder/predict_eval은 이제
하위 폴더를 **재귀 탐색**하므로 상위 폴더를 `--root`로 줘도 됩니다. 그래도 0이면
경로/확장자(.jpg 등) 확인.

## 파일명이 프레임 번호일 때 (001.jpg, 002.jpg ...)
이 경우 파일명은 정답 채널번호가 아니라 프레임 순번입니다.
- 예측만 필요 → `predict_folder.py` (폴더별 채널번호 다수결)
- 정답 채점 필요 → 정답이 `_banner_labels.json` 같은 라벨 파일에 있으면 별도 매핑 필요
