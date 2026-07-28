"""Policy guard for experiments where Airtel is final-test-only.

The generalization project treats Airtel as a complete holdout UI. Any train,
validation, tuning, feature-selection, hard-negative-mining, or temporal-tuning
input that references Airtel must fail early. Final-test inputs may reference
Airtel explicitly.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


FORBIDDEN_ROLES = {
    "train",
    "training",
    "val",
    "validation",
    "tuning",
    "threshold_tuning",
    "feature_selection",
    "hard_negative_mining",
    "temporal_tuning",
}

ALLOWED_AIRTEL_ROLES = {
    "test",
    "final_test",
    "holdout",
    "holdout_test",
    "airtel_final_test",
}


class AirtelPolicyError(RuntimeError):
    """Raised when Airtel appears in a forbidden experiment role."""


def normalize_policy_text(value: str) -> str:
    """Normalize strings before checking for Airtel leakage.

    The project intentionally uses path names such as ``no_airtel`` to document
    policy. Those policy markers are not data leakage, so remove them before
    scanning.
    """

    lowered = value.lower()
    for allowed_marker in ("no_airtel", "no-airtel", "non_airtel", "non-airtel"):
        lowered = lowered.replace(allowed_marker, "")
    return lowered


def contains_airtel_reference(value: object) -> bool:
    return "airtel" in normalize_policy_text(str(value))


def role_allows_airtel(role: str) -> bool:
    return role.strip().lower() in ALLOWED_AIRTEL_ROLES


def role_forbids_airtel(role: str) -> bool:
    lowered = role.strip().lower()
    return lowered in FORBIDDEN_ROLES or not role_allows_airtel(lowered)


def assert_no_airtel_reference(value: object, *, role: str, context: str = "") -> None:
    """Fail if ``value`` references Airtel in a forbidden role."""

    if not role_forbids_airtel(role):
        return
    if contains_airtel_reference(value):
        detail = f" ({context})" if context else ""
        raise AirtelPolicyError(
            f"Airtel is final-test-only; forbidden reference in role={role}{detail}: {value}"
        )


def assert_paths_allowed(paths: Iterable[Path], *, role: str, context: str = "") -> None:
    for path in paths:
        assert_no_airtel_reference(path, role=role, context=context or "path")


def scan_text_file_for_airtel(path: Path, *, max_bytes: int = 1_000_000) -> Optional[str]:
    """Return a short reason if a text-like file appears to reference Airtel."""

    if not path.exists() or not path.is_file():
        return None
    if path.suffix.lower() not in {".txt", ".json", ".jsonl", ".csv", ".md", ".yaml", ".yml"}:
        return None
    try:
        raw = path.read_bytes()[:max_bytes]
        text = raw.decode("utf-8", errors="ignore")
    except OSError:
        return None
    if contains_airtel_reference(text):
        return f"content contains Airtel reference: {path}"
    return None


def assert_input_tree_allowed(path: Path, *, role: str, scan_content: bool = True) -> None:
    """Validate a file or directory for a specific experiment role."""

    assert_no_airtel_reference(path, role=role, context="input path")
    if not role_forbids_airtel(role) or not scan_content:
        return
    if path.is_file():
        reason = scan_text_file_for_airtel(path)
        if reason:
            raise AirtelPolicyError(reason)
        return
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_file():
                assert_no_airtel_reference(child, role=role, context="nested path")
                reason = scan_text_file_for_airtel(child)
                if reason:
                    raise AirtelPolicyError(reason)


def validate_manifest_roles(manifest: dict) -> List[str]:
    """Validate role/path lists in a split manifest and return warning strings."""

    warnings: List[str] = []
    splits = manifest.get("splits", {})
    for split_name, split_doc in splits.items():
        role = str(split_doc.get("role") or split_name)
        for key in ("paths", "images", "labels", "candidate_jsons", "gt_jsons", "sequence_manifests", "yolo_label_dirs"):
            for value in split_doc.get(key, []) or []:
                assert_no_airtel_reference(value, role=role, context=f"{split_name}.{key}")
        if role_allows_airtel(role):
            warnings.append(f"{split_name} allows Airtel because role={role}")
    return warnings


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        train_ok = root / "no_airtel_training_policy" / "train.json"
        train_ok.parent.mkdir(parents=True, exist_ok=True)
        train_ok.write_text('{"provider": "KT"}\n', encoding="utf-8")
        assert_input_tree_allowed(train_ok, role="train")

        train_bad_path = root / "airtel_trial1" / "train.json"
        train_bad_path.parent.mkdir(parents=True, exist_ok=True)
        train_bad_path.write_text('{"provider": "KT"}\n', encoding="utf-8")
        try:
            assert_input_tree_allowed(train_bad_path, role="train")
        except AirtelPolicyError:
            pass
        else:
            raise AssertionError("Airtel path in train role did not fail")

        train_bad_content = root / "kt" / "train.json"
        train_bad_content.parent.mkdir(parents=True, exist_ok=True)
        train_bad_content.write_text('{"source": "Airtel Trial1"}\n', encoding="utf-8")
        try:
            assert_input_tree_allowed(train_bad_content, role="threshold_tuning")
        except AirtelPolicyError:
            pass
        else:
            raise AssertionError("Airtel content in tuning role did not fail")

        assert_input_tree_allowed(train_bad_path, role="final_test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate final-test-only Airtel policy.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--role", default="train")
    parser.add_argument("--paths", nargs="*", type=Path, default=[])
    parser.add_argument("--no-content-scan", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("no_airtel_policy self-test passed")
        return
    for path in args.paths:
        assert_input_tree_allowed(path, role=args.role, scan_content=not args.no_content_scan)
    print("no_airtel_policy validation passed")


if __name__ == "__main__":
    main()
