"""Train a tiny candidate ranker from candidate-level feature CSV.

This intentionally avoids heavyweight ML dependencies so it can run in the
current project venv and export a small JSON model suitable for later porting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


EXCLUDE_COLUMNS = {
    "image_id",
    "candidate_id",
    "candidate_index",
    "split_group",
    "split",
    "category",
    "broadcast_name",
    "candidate_text",
    "parent_text",
    "candidate_digits",
    "numeric_candidate",
    "gt_channel_number",
    "numeric_gt",
    "label_value_correct",
    "label_spatial_correct",
    "hard_negative_same_value_wrong_place",
    "source",
    "gt_iou",
    "gt_overlap_min",
    "gt_center_dist",
    "candidate_contains_gt_center",
    "gt_contains_candidate_center",
    "label_spatial_match",
    "image_has_spatial_positive",
    "label_train",
}


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(out) or math.isinf(out):
        return 0.0
    return out


def safe_int(value: Any) -> int:
    return int(round(safe_float(value)))


def is_float_like(value: Any) -> bool:
    text = str(value).strip()
    if text == "":
        return True
    try:
        out = float(text)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(out) or math.isinf(out))


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def infer_feature_names(rows: Sequence[Dict[str, str]]) -> List[str]:
    names: List[str] = []
    for key in rows[0].keys():
        if key in EXCLUDE_COLUMNS:
            continue
        values = [row.get(key, "") for row in rows[:200]]
        if all(is_float_like(value) for value in values):
            names.append(key)
    return names


def build_matrix(rows: Sequence[Dict[str, str]], feature_names: Sequence[str]) -> np.ndarray:
    data = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for col_index, name in enumerate(feature_names):
            data[row_index, col_index] = safe_float(row.get(name, 0.0))
    return data


def standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def image_balanced_weights(rows: Sequence[Dict[str, str]]) -> np.ndarray:
    by_image: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_image[row["image_id"]].append(index)
    weights = np.zeros(len(rows), dtype=np.float64)
    for indexes in by_image.values():
        positives = [idx for idx in indexes if int(rows[idx]["label_train"])]
        negatives = [idx for idx in indexes if not int(rows[idx]["label_train"])]
        if positives and negatives:
            for idx in positives:
                weights[idx] = 0.5 / len(positives)
            for idx in negatives:
                weights[idx] = 0.5 / len(negatives)
        else:
            for idx in indexes:
                weights[idx] = 1.0 / len(indexes)
    weights /= max(1e-8, weights.mean())
    return weights


def train_pointwise_logistic(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    epochs: int,
    lr: float,
    l2: float,
) -> Tuple[np.ndarray, float, List[float]]:
    n_features = x.shape[1]
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0
    history: List[float] = []
    for epoch in range(epochs):
        logits = x @ w + b
        prob = sigmoid(logits)
        error = (prob - y) * weights
        grad_w = (x.T @ error) / len(x) + l2 * w
        grad_b = float(error.mean())
        w -= lr * grad_w
        b -= lr * grad_b
        if epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1:
            eps = 1e-8
            loss = -np.mean(weights * (y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps)))
            loss += 0.5 * l2 * float(w @ w)
            history.append(float(loss))
    return w, b, history


def build_pair_diffs(rows: Sequence[Dict[str, str]], x: np.ndarray, max_negatives_per_positive: int) -> np.ndarray:
    by_image: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_image[row["image_id"]].append(index)
    diffs: List[np.ndarray] = []
    for indexes in by_image.values():
        positives = [idx for idx in indexes if int(rows[idx]["label_train"])]
        negatives = [idx for idx in indexes if not int(rows[idx]["label_train"])]
        if not positives or not negatives:
            continue
        for pos_idx in positives:
            if len(negatives) > max_negatives_per_positive:
                scored = sorted(
                    negatives,
                    key=lambda idx: (
                        -safe_float(rows[idx].get("ocr_conf", 0.0)),
                        abs(safe_int(rows[idx].get("candidate_digit_len", "0")) - 3),
                    ),
                )
                chosen = scored[:max_negatives_per_positive]
            else:
                chosen = negatives
            for neg_idx in chosen:
                diffs.append(x[pos_idx] - x[neg_idx])
    return np.vstack(diffs) if diffs else np.zeros((0, x.shape[1]), dtype=np.float64)


def train_pairwise_linear(
    diffs: np.ndarray,
    *,
    epochs: int,
    lr: float,
    l2: float,
) -> Tuple[np.ndarray, List[float]]:
    w = np.zeros(diffs.shape[1], dtype=np.float64)
    history: List[float] = []
    if len(diffs) == 0:
        return w, history
    for epoch in range(epochs):
        logits = diffs @ w
        # Loss = log(1 + exp(-logits)); grad = -diff * sigmoid(-logits)
        coeff = sigmoid(-logits)
        grad = -(diffs.T @ coeff) / len(diffs) + l2 * w
        w -= lr * grad
        if epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1:
            loss = float(np.mean(np.logaddexp(0.0, -logits)) + 0.5 * l2 * (w @ w))
            history.append(loss)
    return w, history


def select_by_scores(rows: Sequence[Dict[str, str]], scores: np.ndarray) -> List[Dict[str, Any]]:
    by_image: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_image[row["image_id"]].append(index)
    selected: List[Dict[str, Any]] = []
    for image_id, indexes in by_image.items():
        best_idx = max(indexes, key=lambda idx: float(scores[idx]))
        row = rows[best_idx]
        selected.append(
            {
                "image_id": image_id,
                "split": row["split"],
                "selected_candidate_id": row["candidate_id"],
                "selected_text": row["candidate_text"],
                "selected_source": row["source"],
                "selected_score": float(scores[best_idx]),
                "prediction": row["numeric_candidate"],
                "ground_truth": row["numeric_gt"],
                "exact_match": int(row["numeric_candidate"] == row["numeric_gt"]),
                "label_train": int(row["label_train"]),
                "label_value_correct": int(row["label_value_correct"]),
                "label_spatial_correct": int(row["label_spatial_correct"]),
            }
        )
    return selected


def upper_bound(rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    by_image: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    return {
        "images": len(by_image),
        "value_upper_bound": sum(any(int(row["label_value_correct"]) for row in items) for items in by_image.values()),
        "train_label_upper_bound": sum(any(int(row["label_train"]) for row in items) for items in by_image.values()),
        "spatial_upper_bound": sum(any(int(row["label_spatial_correct"]) for row in items) for items in by_image.values()),
    }


def summarize_selection(selected: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(selected)
    correct = sum(int(item["exact_match"]) for item in selected)
    source_counts = Counter(str(item["selected_source"]) for item in selected)
    wrong_source_counts = Counter(str(item["selected_source"]) for item in selected if not int(item["exact_match"]))
    return {
        "images": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "source_counts": dict(source_counts),
        "wrong_source_counts": dict(wrong_source_counts),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def model_size_bytes(model_doc: Dict[str, Any]) -> int:
    return len(json.dumps(model_doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--pairwise-epochs", type=int, default=1800)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--pairwise-lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.0008)
    parser.add_argument("--max-negatives-per-positive", type=int, default=24)
    args = parser.parse_args()

    rows = load_rows(args.features_csv)
    feature_names = infer_feature_names(rows)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]

    x_train_raw = build_matrix(train_rows, feature_names)
    mean, std = standardize_fit(x_train_raw)

    def transform(split_rows: Sequence[Dict[str, str]]) -> np.ndarray:
        return (build_matrix(split_rows, feature_names) - mean) / std

    x_train = transform(train_rows)
    y_train = np.array([int(row["label_train"]) for row in train_rows], dtype=np.float64)
    weights = image_balanced_weights(train_rows)

    point_w, point_b, point_history = train_pointwise_logistic(
        x_train,
        y_train,
        weights,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
    )

    pair_diffs = build_pair_diffs(train_rows, x_train, args.max_negatives_per_positive)
    pair_w, pair_history = train_pairwise_linear(
        pair_diffs,
        epochs=args.pairwise_epochs,
        lr=args.pairwise_lr,
        l2=args.l2,
    )

    model_docs: Dict[str, Dict[str, Any]] = {
        "pointwise_logistic": {
            "model_type": "linear_pointwise_logistic_ranker",
            "feature_names": feature_names,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "weights": point_w.tolist(),
            "bias": point_b,
            "label": "label_train",
        },
        "pairwise_linear": {
            "model_type": "linear_pairwise_ranker",
            "feature_names": feature_names,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "weights": pair_w.tolist(),
            "bias": 0.0,
            "label": "label_train",
        },
    }

    split_rows = {"train": train_rows, "val": val_rows, "test": test_rows}
    metrics: Dict[str, Any] = {
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "row_counts": {name: len(items) for name, items in split_rows.items()},
        "image_upper_bounds": {name: upper_bound(items) for name, items in split_rows.items()},
        "training": {
            "pointwise_loss_history": point_history,
            "pairwise_loss_history": pair_history,
            "pair_count": int(len(pair_diffs)),
        },
        "models": {},
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for model_name, model_doc in model_docs.items():
        w = np.array(model_doc["weights"], dtype=np.float64)
        b = float(model_doc["bias"])
        model_path = args.out_dir / f"{model_name}.json"
        model_path.write_text(json.dumps(model_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metrics["models"][model_name] = {"model_json_bytes": model_size_bytes(model_doc), "splits": {}}
        for split_name, split_items in split_rows.items():
            x_split = transform(split_items)
            scores = x_split @ w + b
            selected = select_by_scores(split_items, scores)
            metrics["models"][model_name]["splits"][split_name] = summarize_selection(selected)
            write_csv(args.out_dir / f"{model_name}_{split_name}_predictions.csv", selected)

    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {metrics_path}")
    for model_name, model_metrics in metrics["models"].items():
        print(model_name, "model_json_bytes", model_metrics["model_json_bytes"])
        for split_name, split_metrics in model_metrics["splits"].items():
            print(
                " ",
                split_name,
                f"{split_metrics['accuracy']:.4f}",
                f"{split_metrics['correct']}/{split_metrics['images']}",
            )


if __name__ == "__main__":
    main()
