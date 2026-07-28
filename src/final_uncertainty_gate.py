"""Apply and evaluate a simple final uncertainty abstention gate.

This is a post-selection gate for diagnostics/deployment experiments. It does
not use GT to decide whether to allow an output; GT is joined only for metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping


def digits(value: Any) -> str:
    return "".join(ch for ch in str(value or "") if "0" <= ch <= "9")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def load_gt(path: Path | None) -> Dict[str, str]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items() if str(value).strip()}
    return {
        str(item.get("image_id") or item.get("id")): str(item.get("channel_number") or item.get("ground_truth_channel_number") or "")
        for item in raw
        if str(item.get("image_id") or item.get("id") or "").strip()
    }


def row_score(row: Mapping[str, str]) -> float:
    for key in ("selected_score", "ranker_score", "selector_score", "score"):
        if row.get(key) not in (None, ""):
            return safe_float(row.get(key))
    return 0.0


def apply_gate(rows: List[Dict[str, str]], *, threshold: float, reject_out_of_pool: bool = False) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        gated = dict(row)
        pred = digits(row.get("raw_prediction") or row.get("prediction") or row.get("final_pred"))
        score = row_score(row)
        reject_reason = ""
        if pred and score < threshold:
            reject_reason = "uncertainty_score_below_threshold"
        if pred and reject_out_of_pool and str(row.get("final_pred_in_candidate_pool", "")) == "0":
            reject_reason = "uncertainty_out_of_pool_prediction"
        gated["uncertainty_gate_score"] = score
        gated["uncertainty_gate_threshold"] = threshold
        gated["uncertainty_gate_reject_reason"] = reject_reason
        gated["uncertainty_gate_rejected"] = int(bool(reject_reason))
        if reject_reason:
            gated["raw_prediction_before_uncertainty_gate"] = row.get("raw_prediction") or row.get("prediction") or row.get("final_pred")
            gated["prediction_before_uncertainty_gate"] = row.get("prediction") or row.get("final_pred")
            gated["raw_prediction"] = ""
            gated["prediction"] = ""
            gated["numeric_prediction"] = ""
        out.append(gated)
    return out


def evaluate(rows: List[Mapping[str, Any]], gt: Mapping[str, str]) -> Dict[str, Any]:
    valid = [row for row in rows if digits(gt.get(str(row.get("image_id", ""))) or row.get("ground_truth") or row.get("gt_text"))]
    correct = wrong = no_output = outputs = 0
    for row in valid:
        truth = digits(gt.get(str(row.get("image_id", ""))) or row.get("ground_truth") or row.get("gt_text"))
        pred = digits(row.get("raw_prediction") or row.get("prediction") or row.get("final_pred"))
        if not pred:
            no_output += 1
            continue
        outputs += 1
        if truth and pred and str(int(truth)) == str(int(pred)):
            correct += 1
        else:
            wrong += 1
    return {
        "valid_gt_count": len(valid),
        "final_exact_correct_count": correct,
        "final_prediction_exact_accuracy": correct / len(valid) if valid else 0.0,
        "coverage": outputs / len(valid) if valid else 0.0,
        "final_wrong_output_count": wrong,
        "final_no_output_count": no_output,
        "output_precision": correct / outputs if outputs else 0.0,
    }


def no_ui_summary(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    outputs = sum(1 for row in rows if digits(row.get("raw_prediction") or row.get("prediction") or row.get("final_pred")))
    return {
        "images": len(rows),
        "outputs": outputs,
        "fp_rate": outputs / len(rows) if rows else 0.0,
        "no_output_count": len(rows) - outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--gt-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--reject-out-of-pool", action="store_true")
    parser.add_argument("--no-ui-csv", type=Path, default=None)
    args = parser.parse_args()

    gt = load_gt(args.gt_json)
    rows = read_csv(args.predictions_csv)
    gated = apply_gate(rows, threshold=args.threshold, reject_out_of_pool=args.reject_out_of_pool)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "gated_predictions.csv", gated)
    summary = {
        "threshold": args.threshold,
        "reject_out_of_pool": bool(args.reject_out_of_pool),
        "input_predictions_csv": str(args.predictions_csv),
        "airtel": evaluate(gated, gt),
    }
    if args.no_ui_csv is not None:
        no_ui_rows = apply_gate(read_csv(args.no_ui_csv), threshold=args.threshold, reject_out_of_pool=False)
        write_csv(args.output_dir / "gated_no_ui_predictions.csv", no_ui_rows)
        summary["no_ui"] = no_ui_summary(no_ui_rows)
    (args.output_dir / "uncertainty_gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Final Uncertainty Gate",
        "",
        f"- threshold: `{args.threshold}`",
        f"- reject_out_of_pool: `{bool(args.reject_out_of_pool)}`",
        "",
        "## Airtel",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in summary["airtel"].items():
        report.append(f"| {key} | {value} |")
    if "no_ui" in summary:
        report.extend(["", "## No-UI", "", "| metric | value |", "|---|---:|"])
        for key, value in summary["no_ui"].items():
            report.append(f"| {key} | {value} |")
    (args.output_dir / "uncertainty_gate_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
