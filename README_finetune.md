# slot_v3 — 파인튜닝(det+rec) 실행 가이드

overlay 합성 데이터로 **부분 파인튜닝**(일부 파라미터만 업데이트)한 det/rec 가중치로 slot_v3를 돌립니다.
기존 순정 파이프라인은 그대로 두고 **드롭인 교체**만 합니다.

## 파인튜닝 모델

| 단계 | 폴더 | freeze 범위 | 학습 데이터 | 결과 |
|---|---|---|---|---|
| **det**(글자 영역 찾기) | `models/full_image_ocr/det_overlay_frozen_v1` | backbone+neck 고정, **DBHead(18)만** 학습 | overlay 전체 글자박스 3,600/400 | val hmean 0.82 |
| **rec**(글자 읽기) | `models/full_image_ocr/rec_overlay_frozen_v1` | backbone 고정, **neck+head** 학습 | overlay 영어+숫자 crop 19,845/2,161 | val acc 93.2% |

- 아키텍처는 base(PP-OCRv4 **mobile**)와 100% 동일 — 파라미터 개수/추론모델 크기 불변(det ~4.5MB, rec 7.3MB).
- 부분 freeze로 **처음 보는 UI 일반화 유지**(backbone의 범용 특징 보존) + 소량 데이터 과적합/망각 방지.

## 실행

```bash
# 기본 (파인튜닝 det+rec 자동 사용)
./run_slot_v3_ft.sh <이미지_루트폴더> [출력폴더]

# conda 파이썬 지정
PYTHON=/opt/conda/envs/channel-ocr/bin/python ./run_slot_v3_ft.sh /path/to/frames results/ft_run
```

직접 호출도 가능:

```bash
python predict_folder_slot_v3.py \
  --root /path/to/frames --out results/ft_run \
  --det-model-dir models/full_image_ocr/det_overlay_frozen_v1 \
  --rec-model-dir models/full_image_ocr/rec_overlay_frozen_v1 \
  --window 24 --min-conf 0.3
```

- `--det-model-dir` 미지정 시 순정 `PP-OCRv4_mobile_det`(자동 다운로드) 사용.
- `--rec-model-dir` 미지정 시 기존 `en_PP-OCRv4_mobile_rec_ft` 사용.
- 출력: `<out>/per_frame.csv`, `<out>/profile_report.json`, 정성 이미지.

## 다른 서버 준비물

1. conda 환경: `paddlepaddle`(GPU 또는 CPU) + `paddleocr` + `paddlex[ocr]`.
2. 이 repo clone (파인튜닝 inference 모델이 `models/full_image_ocr/*_overlay_frozen_v1`에 포함됨).
3. `PYTHON` 환경변수로 해당 env의 파이썬을 가리켜 실행.

## 재학습(참고)

- 라벨 생성: `teacher_model/src/prep_overlay_finetune.py`
- det config: `teacher_model/experiments/det_overlay_frozen_v1.yml` (`freeze_backbone/freeze_neck: True`)
- rec config: `teacher_model/experiments/rec_overlay_frozen_v1.yml` (`freeze_backbone: True`)
- 학습: `PaddleOCR/tools/train.py` (freeze 훅 + export_model.py 로 inference 변환)
