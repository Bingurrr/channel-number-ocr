"""Opt-in runtime adapter for value-group ranking and relative gating."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from candidate_ranker import CandidateRanker, candidate_features, normalize_source_alias
from channel_number_fusion import digits, normalized_digits
from train_value_group_relative_gate_no_airtel_v1 import build_gate_rows
from train_value_group_selector_no_airtel_v1 import group_candidate_rows


class ValueGroupRelativeSelector:
    """Rerank eligible candidates by value and apply a relative confidence gate."""

    def __init__(
        self,
        group_model_path: Path,
        gate_model_path: Path,
        gate_threshold: float,
        gate_policy: str = "auto",
        multi_position_rescue: bool = False,
        rescue_min_yolo_conf: float = 0.8,
        rescue_min_ocr_conf: float = 0.9,
        rescue_min_margin: float = 1.0,
    ) -> None:
        if gate_policy not in {"auto", "known_present", "positive_first"}:
            raise ValueError(f"unsupported relative gate policy: {gate_policy}")
        self.group_model_path = Path(group_model_path)
        self.gate_model_path = Path(gate_model_path)
        self.group_ranker = CandidateRanker(self.group_model_path)
        self.gate_ranker = CandidateRanker(self.gate_model_path)
        self.gate_threshold = float(gate_threshold)
        self.gate_policy = gate_policy
        self.multi_position_rescue = bool(multi_position_rescue)
        self.rescue_min_yolo_conf = float(rescue_min_yolo_conf)
        self.rescue_min_ocr_conf = float(rescue_min_ocr_conf)
        self.rescue_min_margin = float(rescue_min_margin)
        self.model_type = "value_group_relative_gate"

    @staticmethod
    def _runtime_key(candidate: Mapping[str, Any], index: int) -> str:
        candidate_id = str(candidate.get("candidate_id") or candidate.get("id") or f"candidate_{index:04d}")
        return f"{index:04d}:{candidate_id}"

    def _candidate_rows(
        self,
        ranked: Sequence[Dict[str, Any]],
        yolo: Sequence[Dict[str, Any]],
        width: int,
        height: int,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for index, item in enumerate(ranked, 1):
            feature_candidate = dict(item)
            # The base ranker has already annotated the runtime candidate. Remove
            # those derived scores so the group model sees the same raw feature
            # distribution that was used to build its training dataset.
            feature_candidate.pop("ranker_score", None)
            feature_candidate.pop("fusion_score", None)
            feature_candidate.pop("selection_model", None)
            if item.get("original_bbox_xyxy"):
                feature_candidate["bbox_xyxy"] = list(item["original_bbox_xyxy"])
            box = [float(value) for value in feature_candidate["bbox_xyxy"]]
            value = digits(item.get("text", ""))
            runtime_key = self._runtime_key(item, index)
            raw_source = item.get("source") or item.get("raw_source") or "ocr"
            row: Dict[str, Any] = {
                "split": "inference",
                "provider": "unknown",
                "image_id": "runtime_image",
                "image_path": "",
                "candidate_id": runtime_key,
                "candidate_text": str(item.get("text", value)),
                "candidate_digits": value,
                "numeric_candidate": value,
                "source": normalize_source_alias(raw_source),
                "raw_source": str(raw_source),
                "bbox": json.dumps(box, separators=(",", ":")),
                "bbox_x1": box[0],
                "bbox_y1": box[1],
                "bbox_x2": box[2],
                "bbox_y2": box[3],
                "gt_channel_number": "",
                "numeric_gt": "",
                "label_exact": 0,
                "label_numeric_equiv": 0,
                "label_train": 0,
            }
            row.update(candidate_features(feature_candidate, list(yolo), width, height))
            rows.append(row)
        return rows

    def select(
        self,
        ranked: Sequence[Dict[str, Any]],
        yolo: Sequence[Dict[str, Any]],
        width: int,
        height: int,
    ) -> Dict[str, Any]:
        if not ranked:
            return {
                "best_candidate": None,
                "ranked_candidates": [],
                "selection_model": self.model_type,
                "relative_gate_policy": self.gate_policy,
                "relative_gate_bypassed": False,
                "reject_reason": "no_runtime_eligible_candidates",
            }

        candidate_rows = self._candidate_rows(ranked, yolo, width, height)
        baseline_scores = [float(item.get("ranker_score", item.get("fusion_score", -999.0))) for item in ranked]
        group_rows = group_candidate_rows(candidate_rows, baseline_scores)
        group_scores = [self.group_ranker.score_features(row) for row in group_rows]
        gate_rows = build_gate_rows(group_rows, group_scores, candidate_rows)
        if not gate_rows:
            return {
                "best_candidate": None,
                "ranked_candidates": [],
                "selection_model": self.model_type,
                "relative_gate_policy": self.gate_policy,
                "relative_gate_bypassed": False,
                "reject_reason": "no_value_groups",
            }
        gate_row = gate_rows[0]
        gate_score = self.gate_ranker.score_features(gate_row)

        group_score_by_value = {
            str(row.get("candidate_digits")): float(score) for row, score in zip(group_rows, group_scores)
        }
        group_rank_by_value = {
            value: rank
            for rank, (value, _) in enumerate(
                sorted(group_score_by_value.items(), key=lambda item: item[1], reverse=True),
                1,
            )
        }
        members_by_value: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in ranked:
            value = digits(item.get("text", ""))
            enriched = dict(item)
            enriched["baseline_fusion_score"] = float(item.get("fusion_score", -999.0))
            enriched["value_group_score"] = group_score_by_value.get(value, -999.0)
            enriched["value_group_rank"] = group_rank_by_value.get(value, 999999)
            enriched["selection_model"] = self.model_type
            members_by_value[value].append(enriched)

        ordered: List[Dict[str, Any]] = []
        for value, _ in sorted(group_score_by_value.items(), key=lambda item: item[1], reverse=True):
            members = sorted(
                members_by_value[value],
                key=lambda item: float(item.get("baseline_fusion_score", -999.0)),
                reverse=True,
            )
            for item in members:
                item["fusion_score"] = round(group_score_by_value[value], 6)
                ordered.append(item)

        selected_value = str(gate_row.get("prediction") or "")
        selected = next((item for item in ordered if digits(item.get("text", "")) == selected_value), None)
        if selected is None:
            return {
                "best_candidate": None,
                "ranked_candidates": ordered[:20],
                "selection_model": self.model_type,
                "relative_gate_policy": self.gate_policy,
                "relative_gate_bypassed": False,
                "reject_reason": "selected_value_not_mapped",
            }

        selected = dict(selected)
        selected["digits"] = selected_value
        selected["normalized_digits"] = normalized_digits(selected_value)
        selected["relative_gate_score"] = round(float(gate_score), 6)
        selected["relative_gate_threshold"] = self.gate_threshold
        selected["value_group_candidate_count"] = int(float(gate_row.get("top1_candidate_support", 0.0)))
        selected["value_group_source_count"] = int(float(gate_row.get("top1_source_support", 0.0)))
        selected["value_group_margin"] = float(gate_row.get("top1_top2_margin", 0.0))
        selected["value_group_softmax_probability"] = float(gate_row.get("top1_softmax_probability", 0.0))
        selected["fusion_score"] = round(float(gate_score), 6)

        score_gate_passed = bool(gate_score >= self.gate_threshold)
        known_present = self.gate_policy == "known_present"
        positive_first = self.gate_policy == "positive_first"
        gate_bypassed = (known_present or positive_first) and not score_gate_passed
        rescue_yolo_conf = max(
            float(gate_row.get("top1_best_cluster_yolo_ch0_conf_max", 0.0)),
            float(gate_row.get("top1_best_cluster_yolo_area3_conf_max", 0.0)),
        )
        rescue_passed = bool(
            self.multi_position_rescue
            and not score_gate_passed
            and float(gate_row.get("top1_multi_position_same_text", 0.0)) > 0.0
            and float(gate_row.get("top1_candidate_support", 0.0)) >= 2.0
            and float(gate_row.get("top1_channel_evidence_component_count", 0.0)) >= 1.0
            and float(gate_row.get("top1_best_cluster_near_yolo_ratio", 0.0)) >= 1.0
            and rescue_yolo_conf >= self.rescue_min_yolo_conf
            and float(gate_row.get("top1_best_cluster_ocr_conf_max", 0.0)) >= self.rescue_min_ocr_conf
            and float(gate_row.get("top1_best_cluster_other_number_overlap_ratio", 0.0)) <= 0.0
            and float(gate_row.get("top1_top2_margin", 0.0)) >= self.rescue_min_margin
        )
        output_allowed = score_gate_passed or known_present or positive_first or rescue_passed

        doc: Dict[str, Any] = {
            "best_candidate": selected if output_allowed else None,
            "ranked_candidates": ordered[:20],
            "selection_model": self.model_type,
            "value_group_count": len(group_rows),
            "relative_gate_score": round(float(gate_score), 6),
            "relative_gate_threshold": self.gate_threshold,
            "relative_gate_policy": self.gate_policy,
            "relative_gate_score_passed": score_gate_passed,
            "relative_gate_bypassed": gate_bypassed,
            "positive_first_output": positive_first,
            "output_confidence_status": "high" if score_gate_passed else "low",
            "relative_gate_multi_position_rescue_enabled": self.multi_position_rescue,
            "relative_gate_multi_position_rescue_passed": rescue_passed,
            "relative_gate_rescue_yolo_conf": rescue_yolo_conf,
            "relative_gate_rescue_min_yolo_conf": self.rescue_min_yolo_conf,
            "relative_gate_rescue_min_ocr_conf": self.rescue_min_ocr_conf,
            "relative_gate_rescue_margin": float(gate_row.get("top1_top2_margin", 0.0)),
            "relative_gate_rescue_min_margin": self.rescue_min_margin,
            "relative_gate_passed": output_allowed,
            "relative_gate_features": {
                name: gate_row.get(name, 0.0) for name in self.gate_ranker.feature_names
            },
        }
        if not output_allowed:
            doc["reject_reason"] = "relative_confidence_gate"
            doc["rejected_best_candidate"] = selected
        return doc


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        common = {"mean": [0.0], "std": [1.0], "weights": [1.0], "bias": 0.0}
        group_path = root / "group.json"
        gate_path = root / "gate.json"
        group_path.write_text(
            json.dumps({"model_type": "test_group", "feature_names": ["group_base_score_max"], **common}),
            encoding="utf-8",
        )
        gate_path.write_text(
            json.dumps({"model_type": "test_gate", "feature_names": ["top1_softmax_probability"], **common}),
            encoding="utf-8",
        )
        selector = ValueGroupRelativeSelector(group_path, gate_path, 0.5)
        candidates = [
            {
                "candidate_id": "a",
                "text": "041",
                "source": "original_ocr",
                "raw_source": "original_ocr",
                "bbox_xyxy": [10.0, 10.0, 40.0, 30.0],
                "ocr_conf": 0.9,
                "ranker_score": 2.0,
                "fusion_score": 2.0,
            },
            {
                "candidate_id": "b",
                "text": "41",
                "source": "refined_numeric_substring",
                "raw_source": "refined_numeric_substring",
                "bbox_xyxy": [10.0, 10.0, 40.0, 30.0],
                "ocr_conf": 0.8,
                "ranker_score": 1.0,
                "fusion_score": 1.0,
            },
        ]
        result = selector.select(candidates, [], 100, 100)
        assert result["best_candidate"] is not None
        assert result["best_candidate"]["digits"] == "041"
        assert result["relative_gate_passed"] is True
        runtime_rows = selector._candidate_rows(candidates, [], 100, 100)
        assert runtime_rows[0]["source"] == "original_ocr"
        assert runtime_rows[0]["raw_source"] == "original_ocr"

        source_missing = dict(candidates[0])
        source_missing.pop("source")
        source_missing.pop("raw_source")
        missing_source_row = selector._candidate_rows([source_missing], [], 100, 100)[0]
        assert missing_source_row["source"] == "original_ocr"
        assert missing_source_row["raw_source"] == "ocr"

        strict_selector = ValueGroupRelativeSelector(group_path, gate_path, 2.0)
        strict_result = strict_selector.select(candidates, [], 100, 100)
        assert strict_result["best_candidate"] is None
        assert strict_result["relative_gate_score_passed"] is False

        known_present_selector = ValueGroupRelativeSelector(
            group_path,
            gate_path,
            2.0,
            gate_policy="known_present",
        )
        known_present_result = known_present_selector.select(candidates, [], 100, 100)
        assert known_present_result["best_candidate"] is not None
        assert known_present_result["relative_gate_score_passed"] is False
        assert known_present_result["relative_gate_bypassed"] is True
        assert known_present_result["relative_gate_passed"] is True

        positive_first_selector = ValueGroupRelativeSelector(
            group_path,
            gate_path,
            2.0,
            gate_policy="positive_first",
        )
        positive_first_result = positive_first_selector.select(candidates, [], 100, 100)
        assert positive_first_result["best_candidate"] is not None
        assert positive_first_result["relative_gate_score_passed"] is False
        assert positive_first_result["relative_gate_bypassed"] is True
        assert positive_first_result["positive_first_output"] is True
        assert positive_first_result["output_confidence_status"] == "low"

        rescue_selector = ValueGroupRelativeSelector(
            group_path,
            gate_path,
            2.0,
            multi_position_rescue=True,
        )
        repeated_candidates = [
            {**candidates[0], "candidate_id": "left", "ocr_conf": 0.95},
            {
                **candidates[0],
                "candidate_id": "right",
                "bbox_xyxy": [60.0, 60.0, 90.0, 80.0],
                "ocr_conf": 0.96,
            },
            {
                **candidates[0],
                "candidate_id": "competitor",
                "text": "99",
                "bbox_xyxy": [45.0, 45.0, 55.0, 55.0],
                "ocr_conf": 0.7,
                "ranker_score": 0.2,
                "fusion_score": 0.2,
            },
        ]
        yolo = [
            {
                "class_id": 0,
                "bbox_xyxy": [9.0, 9.0, 41.0, 31.0],
                "conf": 0.95,
            }
        ]
        rescue_result = rescue_selector.select(repeated_candidates, yolo, 100, 100)
        assert rescue_result["best_candidate"] is not None
        assert rescue_result["relative_gate_score_passed"] is False
        assert rescue_result["relative_gate_multi_position_rescue_passed"] is True


def main() -> None:
    parser = argparse.ArgumentParser(description="Value-group relative-gate runtime self-test.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("value_group_runtime self-test passed")


if __name__ == "__main__":
    main()
