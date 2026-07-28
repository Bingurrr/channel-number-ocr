#!/usr/bin/env python3
"""Quick environment check for the inference pipeline."""
import importlib, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
print("python:", sys.version.split()[0])
for m, need in [("paddle","3.3.1"),("paddleocr","3.4.1"),("paddlex","3.4.3"),
                ("ultralytics","8.4.104"),("cv2",None)]:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        tag = "" if (need is None or str(v)==need) else f"  ⚠ expected {need}"
        print(f"  {m}: {v}{tag}")
    except Exception as e:
        print(f"  {m}: FAIL -> {type(e).__name__}: {str(e)[:80]}")
import json
cfg = json.loads((HERE/"config.json").read_text())
print("models:")
for k in ("detector","numeric_ocr","selector_dir"):
    p = cfg[k] if str(cfg[k]).startswith("/") else HERE/cfg[k]
    print(f"  {k}: {'OK' if Path(p).exists() else 'MISSING'}  ({p})")
