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
