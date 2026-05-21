from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT2_DIR = PROJECT_ROOT / "deliverables" / "checkpoint-2"
FIGURE_DIR = CHECKPOINT2_DIR / "figures"


REQUIRED_FILES = [
    "README.md",
    "checkpoint2_pr_merge_modeling.ipynb",
    "checkpoint2_milestone_summary.md",
    "checkpoint2_next_steps.md",
    "checkpoint2_model_comparison.csv",
    "checkpoint2_target_distribution.csv",
    "checkpoint2_data_characteristics_summary.csv",
    "checkpoint2_feature_availability_contract.csv",
    "checkpoint2_preprocessing_summary.csv",
    "checkpoint2_algorithm_rationale.csv",
    "checkpoint2_validation_confusion_matrix.csv",
    "checkpoint2_threshold_tuning.csv",
    "checkpoint2_feature_importance.csv",
]

REQUIRED_FIGURES = [
    "checkpoint2_target_distribution.png",
    "checkpoint2_model_comparison_f1.png",
    "checkpoint2_model_comparison_balanced_accuracy.png",
    "checkpoint2_feature_policy_summary.png",
    "checkpoint2_confusion_matrix_validation.png",
    "checkpoint2_threshold_tradeoff.png",
    "checkpoint2_feature_importance.png",
]

CSV_REQUIRED_COLUMNS = {
    "checkpoint2_model_comparison.csv": {
        "model",
        "training_scope",
        "balanced_accuracy",
        "precision_not_merged",
        "recall_not_merged",
        "f1_not_merged",
        "average_precision_not_merged",
    },
    "checkpoint2_target_distribution.csv": {"split", "target_value", "label", "count", "percentage"},
    "checkpoint2_data_characteristics_summary.csv": {
        "topic",
        "evidence",
        "checkpoint2_implication",
        "source_artifact",
    },
    "checkpoint2_feature_availability_contract.csv": {
        "feature_group",
        "checkpoint2_usage",
        "examples",
        "reason",
        "risk_level",
        "presentation_wording",
    },
    "checkpoint2_preprocessing_summary.csv": {
        "pipeline_step",
        "columns_or_scope",
        "method",
        "reason",
        "implementation_note",
    },
    "checkpoint2_algorithm_rationale.csv": {
        "algorithm",
        "course_connection",
        "why_included",
        "main_assumption_or_bias",
        "main_risk",
        "required_preprocessing",
        "checkpoint2_role",
    },
    "checkpoint2_validation_confusion_matrix.csv": {
        "model",
        "actual_not_merged_predicted_not_merged",
        "actual_not_merged_predicted_merged",
        "actual_merged_predicted_not_merged",
        "actual_merged_predicted_merged",
    },
    "checkpoint2_threshold_tuning.csv": {
        "threshold",
        "precision_not_merged",
        "recall_not_merged",
        "f1_not_merged",
    },
    "checkpoint2_feature_importance.csv": {"feature", "importance"},
}


def fail(message: str) -> None:
    raise AssertionError(message)


def assert_exists(path: Path) -> None:
    if not path.exists():
        fail(f"Missing required file: {path.relative_to(PROJECT_ROOT)}")
    if path.is_file() and path.stat().st_size == 0:
        fail(f"Required file is empty: {path.relative_to(PROJECT_ROOT)}")


def assert_required_files() -> None:
    for filename in REQUIRED_FILES:
        assert_exists(CHECKPOINT2_DIR / filename)
    for filename in REQUIRED_FIGURES:
        assert_exists(FIGURE_DIR / filename)


def assert_csvs_non_empty_and_shaped() -> None:
    for filename, required_columns in CSV_REQUIRED_COLUMNS.items():
        path = CHECKPOINT2_DIR / filename
        df = pd.read_csv(path)
        if df.empty:
            fail(f"{path.relative_to(PROJECT_ROOT)} is empty")
        missing = required_columns - set(df.columns)
        if missing:
            fail(f"{path.relative_to(PROJECT_ROOT)} is missing columns: {sorted(missing)}")


def assert_notebook_executed() -> None:
    path = CHECKPOINT2_DIR / "checkpoint2_pr_merge_modeling.ipynb"
    notebook = json.loads(path.read_text())
    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    if not code_cells:
        fail("Checkpoint 2 notebook has no code cells")
    executed = [cell for cell in code_cells if cell.get("execution_count") is not None]
    output_cells = [cell for cell in code_cells if cell.get("outputs")]
    if len(executed) < 3:
        fail("Checkpoint 2 notebook should have at least three executed code cells")
    if len(output_cells) < 3:
        fail("Checkpoint 2 notebook should have visible outputs")
    errors: list[str] = []
    for index, cell in enumerate(code_cells, start=1):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(f"code cell {index}: {output.get('ename')}: {output.get('evalue')}")
    if errors:
        fail("Checkpoint 2 notebook contains error outputs:\n" + "\n".join(errors))


def assert_readme_methodology() -> None:
    text = (CHECKPOINT2_DIR / "README.md").read_text().lower()
    required_phrases = [
        "predicting and explaining github pull request merge outcomes",
        "validation metrics are the checkpoint 2 headline",
        ".venv/bin/python",
        "requirements-local.txt",
        "not causal",
        "not deployment-ready",
        "leakage-safer",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in text]
    if missing:
        fail(f"README.md is missing required methodology phrases: {missing}")

    risky_patterns = [
        r"checkpoint 2 headline[^.\n]*(official|test|holdout)",
        r"official test[^.\n]*(selected|selection|tuning)",
        r"test split[^.\n]*(headline|tuning|selection)",
    ]
    for pattern in risky_patterns:
        match = re.search(pattern, text)
        if match and "not used" not in match.group(0) and "reserved" not in match.group(0):
            fail(f"README.md may misuse official-test language: {pattern}")


def assert_feature_contract_policy() -> None:
    contract = pd.read_csv(CHECKPOINT2_DIR / "checkpoint2_feature_availability_contract.csv")
    expected_groups = {
        "headline_leakage_safer_features",
        "excluded_post_outcome_or_identifier_features",
        "sensitivity_only_features",
        "T2_review_process_features",
        "comments_and_survey",
    }
    observed_groups = set(contract["feature_group"])
    missing = expected_groups - observed_groups
    if missing:
        fail(f"Feature contract missing required groups: {sorted(missing)}")
    sensitivity = contract[contract["feature_group"] == "sensitivity_only_features"]
    if sensitivity.empty or "prior_review_num" not in " ".join(sensitivity["examples"].astype(str)):
        fail("Feature contract must keep prior_review_num in sensitivity_only_features")


def main() -> None:
    assert_required_files()
    assert_csvs_non_empty_and_shaped()
    assert_notebook_executed()
    assert_readme_methodology()
    assert_feature_contract_policy()
    print("Checkpoint 2 output validation passed.")


if __name__ == "__main__":
    main()
