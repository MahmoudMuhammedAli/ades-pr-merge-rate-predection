from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = PROJECT_ROOT / "deliverables" / "final"
SLIDES_DIR = PROJECT_ROOT / "slides"


REQUIRED_FINAL_FILES = [
    "assignment_coverage.csv",
    "cluster_interpretation.csv",
    "cluster_k_selection.csv",
    "cluster_profile.csv",
    "cross_validation_summary.csv",
    "calibration_summary.csv",
    "data_quality_summary.csv",
    "eda_key_findings.csv",
    "error_analysis_key_findings.csv",
    "error_profile_summary.csv",
    "feature_assumption_flags.csv",
    "feature_family_ablation.csv",
    "feature_importance.csv",
    "feature_timing_evidence.csv",
    "feature_timing_summary.csv",
    "final_confusion_matrix.csv",
    "final_metric_confidence_intervals.csv",
    "final_pr_merge_analysis.ipynb",
    "final_report.md",
    "final_report.pdf",
    "final_test_metrics.csv",
    "generalization_stress_tests.csv",
    "leakage_sensitivity.csv",
    "model_comparison.csv",
    "permutation_importance.csv",
    "prediction_contract_comparison.csv",
    "prediction_contract_feature_map.csv",
    "project_creator_concentration.csv",
    "project_cluster_metric_confidence_intervals.csv",
    "review_process_feature_audit.csv",
    "raw_data_fingerprint.csv",
    "sample_size_sensitivity.csv",
    "split_id_integrity.csv",
    "split_overlap_summary.csv",
    "stress_model_comparison.csv",
    "target_distribution_summary.csv",
    "threshold_stability.csv",
    "threshold_tuning_curve_sample.csv",
    "threshold_tuning_top.csv",
    "training_scope_summary.csv",
    "test_prediction_risk_bands.csv",
    "validation_test_gap.csv",
    "comment_dataset_profile.csv",
    "calibration_model_comparison.csv",
    "full_generalization_benchmarks.csv",
    "paired_model_delta_intervals.csv",
    "professor_defense_notes.md",
]

REQUIRED_SLIDE_FILES = [
    "final_presentation.md",
    "final_presentation.pdf",
]

REQUIRED_FIGURES = [
    "cluster_pca.png",
    "calibration_curve.png",
    "eda_safe_feature_patterns.png",
    "error_profile.png",
    "feature_importance.png",
    "final_confusion_matrix.png",
    "model_comparison.png",
    "permutation_importance.png",
    "prediction_contract_comparison.png",
    "safe_feature_correlation.png",
    "target_distribution.png",
    "risk_band_lift.png",
    "threshold_tuning.png",
    "calibration_model_comparison.png",
    "full_generalization_benchmarks.png",
]


def fail(message: str) -> None:
    raise AssertionError(message)


def assert_exists(path: Path) -> None:
    if not path.exists():
        fail(f"Missing required file: {path.relative_to(PROJECT_ROOT)}")
    if path.is_file() and path.stat().st_size == 0:
        fail(f"Required file is empty: {path.relative_to(PROJECT_ROOT)}")


def assert_notebook_has_no_errors(path: Path) -> None:
    notebook = json.loads(path.read_text())
    errors: list[str] = []
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(f"cell {index}: {output.get('ename')}: {output.get('evalue')}")
    if errors:
        fail("Notebook contains error outputs:\n" + "\n".join(errors))


def recompute_metrics_from_confusion(row: pd.Series) -> dict[str, float]:
    tp = row["actual_not_merged_predicted_not_merged"]
    fn = row["actual_not_merged_predicted_merged"]
    fp = row["actual_merged_predicted_not_merged"]
    tn = row["actual_merged_predicted_merged"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    merged_recall = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "precision_not_merged": precision,
        "recall_not_merged": recall,
        "f1_not_merged": f1,
        "balanced_accuracy": (recall + merged_recall) / 2,
    }


def assert_metric_table_consistent(path: Path) -> None:
    df = pd.read_csv(path)
    required = {
        "actual_not_merged_predicted_not_merged",
        "actual_not_merged_predicted_merged",
        "actual_merged_predicted_not_merged",
        "actual_merged_predicted_merged",
        "precision_not_merged",
        "recall_not_merged",
        "f1_not_merged",
        "balanced_accuracy",
    }
    missing = required - set(df.columns)
    if missing:
        fail(f"{path.relative_to(PROJECT_ROOT)} is missing metric columns: {sorted(missing)}")
    for row_number, row in df.iterrows():
        expected = recompute_metrics_from_confusion(row)
        for metric, value in expected.items():
            observed = row[metric]
            if not math.isclose(observed, value, rel_tol=1e-10, abs_tol=1e-10):
                fail(
                    f"{path.relative_to(PROJECT_ROOT)} row {row_number} has inconsistent {metric}: "
                    f"observed={observed}, expected={value}"
                )


def assert_assignment_coverage() -> None:
    coverage = pd.read_csv(FINAL_DIR / "assignment_coverage.csv")
    expected_requirements = {
        "problem description and clarity",
        "exploratory data analysis",
        "empirical data analysis study",
        "algorithm comprehension",
        "data characteristics",
        "main findings and conclusions",
        "notebook/code organization",
        "slides and discussion support",
    }
    observed = set(coverage["requirement"])
    missing = expected_requirements - observed
    if missing:
        fail(f"assignment_coverage.csv does not map assignment criteria: {sorted(missing)}")
    required_reference_columns = {"notebook_section", "report_section", "slide", "artifact"}
    missing_columns = required_reference_columns - set(coverage.columns)
    if missing_columns:
        fail(f"assignment_coverage.csv is missing evidence reference columns: {sorted(missing_columns)}")


def assert_report_claim_language() -> None:
    report = (FINAL_DIR / "final_report.md").read_text().lower()
    required_phrases = [
        "moderate predictive association",
        "not causal",
        "not suitable for automated merge decisions",
        "feature-availability",
        "external-validity",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in report]
    if missing:
        fail(f"final_report.md is missing required claim-framing phrases: {missing}")


def assert_professor_defense_notes() -> None:
    notes = (FINAL_DIR / "professor_defense_notes.md").read_text().lower()
    required_phrases = [
        "forced",
        "leakage",
        "math",
        "generalization",
        "calibration",
        "clustering",
        "not causal",
        "not deployment-ready",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in notes]
    if missing:
        fail(f"professor_defense_notes.md is missing required defense topics: {missing}")


def assert_required_figures() -> None:
    figure_dir = FINAL_DIR / "figures"
    for filename in REQUIRED_FIGURES:
        assert_exists(figure_dir / filename)


def assert_target_distribution_summary() -> None:
    target = pd.read_csv(FINAL_DIR / "target_distribution_summary.csv")
    required = {"split", "label", "count", "percentage"}
    missing = required - set(target.columns)
    if missing:
        fail(f"target_distribution_summary.csv is missing columns: {sorted(missing)}")
    train = target[target["split"] == "train"].set_index("label")
    expected = {
        "Merged": (932_538, 89.16),
        "Not merged": (113_345, 10.84),
    }
    for label, (count, percentage) in expected.items():
        if label not in train.index:
            fail(f"target_distribution_summary.csv is missing train label {label!r}")
        row = train.loc[label]
        if int(row["count"]) != count:
            fail(f"target_distribution_summary.csv has wrong train count for {label}: {row['count']}")
        if not math.isclose(float(row["percentage"]), percentage, abs_tol=0.01):
            fail(
                f"target_distribution_summary.csv has wrong train percentage for {label}: "
                f"{row['percentage']}"
            )


def assert_eda_key_findings() -> None:
    findings = pd.read_csv(FINAL_DIR / "eda_key_findings.csv")
    if len(findings) < 6:
        fail("eda_key_findings.csv should contain at least six professor-facing findings")
    required = {"finding", "evidence", "interpretation"}
    missing = required - set(findings.columns)
    if missing:
        fail(f"eda_key_findings.csv is missing columns: {sorted(missing)}")


def assert_stress_model_comparison() -> None:
    stress = pd.read_csv(FINAL_DIR / "stress_model_comparison.csv")
    required_splits = {
        "random_stratified_sample",
        "temporal_last_25pct_sample",
        "project_group_holdout_sample",
        "creator_group_holdout_sample",
    }
    observed_splits = set(stress["stress_test"])
    missing_splits = required_splits - observed_splits
    if missing_splits:
        fail(f"stress_model_comparison.csv is missing stress tests: {sorted(missing_splits)}")
    if stress["model"].nunique() < 4:
        fail("stress_model_comparison.csv should compare multiple model families")


def assert_split_id_integrity() -> None:
    integrity = pd.read_csv(FINAL_DIR / "split_id_integrity.csv").set_index("check")
    required = {
        "train_duplicate_ids",
        "test_duplicate_ids",
        "train_test_id_overlap",
    }
    missing = required - set(integrity.index)
    if missing:
        fail(f"split_id_integrity.csv is missing checks: {sorted(missing)}")
    for check in required:
        value = int(integrity.loc[check, "value"])
        if value != 0:
            fail(f"split_id_integrity.csv expected {check}=0, observed {value}")


def assert_risk_band_table() -> None:
    risk = pd.read_csv(FINAL_DIR / "test_prediction_risk_bands.csv")
    required = {
        "risk_decile",
        "row_count",
        "actual_not_merged_rate",
        "baseline_not_merged_rate",
        "lift_vs_baseline",
    }
    missing = required - set(risk.columns)
    if missing:
        fail(f"test_prediction_risk_bands.csv is missing columns: {sorted(missing)}")
    if risk["risk_decile"].nunique() != 10:
        fail("test_prediction_risk_bands.csv should contain exactly 10 risk deciles")
    ordered = risk.sort_values("risk_decile")
    if not ordered["risk_decile"].is_monotonic_increasing:
        fail("test_prediction_risk_bands.csv risk_decile is not ordered")
    if (ordered["row_count"] <= 0).any():
        fail("test_prediction_risk_bands.csv contains an empty risk decile")
    top_lift = ordered.loc[ordered["risk_decile"].idxmax(), "lift_vs_baseline"]
    if top_lift <= 1.0:
        fail("highest risk decile should be above the baseline not-merged rate")


def assert_calibration_summary() -> None:
    calibration = pd.read_csv(FINAL_DIR / "calibration_summary.csv")
    required = {
        "calibration_bin",
        "row_count",
        "mean_predicted_not_merged_score",
        "observed_not_merged_rate",
        "absolute_calibration_error",
        "brier_score_not_merged",
    }
    missing = required - set(calibration.columns)
    if missing:
        fail(f"calibration_summary.csv is missing columns: {sorted(missing)}")
    if calibration["calibration_bin"].nunique() != 10:
        fail("calibration_summary.csv should contain exactly 10 calibration bins")
    if (calibration["row_count"] <= 0).any():
        fail("calibration_summary.csv contains an empty calibration bin")
    rate_columns = ["mean_predicted_not_merged_score", "observed_not_merged_rate"]
    for column in rate_columns:
        if ((calibration[column] < 0) | (calibration[column] > 1)).any():
            fail(f"calibration_summary.csv has invalid rates in {column}")


def assert_error_analysis_findings() -> None:
    profile = pd.read_csv(FINAL_DIR / "error_profile_summary.csv")
    required_groups = {
        "correct_not_merged",
        "missed_not_merged",
        "false_not_merged",
        "correct_merged",
    }
    missing_groups = required_groups - set(profile["outcome_group"])
    if missing_groups:
        fail(f"error_profile_summary.csv is missing groups: {sorted(missing_groups)}")
    findings = pd.read_csv(FINAL_DIR / "error_analysis_key_findings.csv")
    required = {"finding", "evidence", "interpretation"}
    missing = required - set(findings.columns)
    if missing:
        fail(f"error_analysis_key_findings.csv is missing columns: {sorted(missing)}")
    if len(findings) < 3:
        fail("error_analysis_key_findings.csv should contain at least three findings")


def assert_prediction_contract_outputs() -> None:
    comparison = pd.read_csv(FINAL_DIR / "prediction_contract_comparison.csv")
    required_columns = {
        "contract",
        "evaluation_split",
        "feature_count",
        "precision_not_merged",
        "recall_not_merged",
        "f1_not_merged",
        "average_precision_not_merged",
        "interpretation",
    }
    missing = required_columns - set(comparison.columns)
    if missing:
        fail(f"prediction_contract_comparison.csv is missing columns: {sorted(missing)}")
    required_contracts = {"T0_creation", "T1_diff", "T2_review_process"}
    required_splits = {"validation", "test"}
    missing_contracts = required_contracts - set(comparison["contract"])
    missing_splits = required_splits - set(comparison["evaluation_split"])
    if missing_contracts:
        fail(f"prediction_contract_comparison.csv is missing contracts: {sorted(missing_contracts)}")
    if missing_splits:
        fail(f"prediction_contract_comparison.csv is missing evaluation splits: {sorted(missing_splits)}")
    for contract in required_contracts:
        observed_splits = set(comparison.loc[comparison["contract"] == contract, "evaluation_split"])
        if required_splits - observed_splits:
            fail(f"prediction_contract_comparison.csv contract {contract} is missing validation or test rows")

    feature_map = pd.read_csv(FINAL_DIR / "prediction_contract_feature_map.csv")
    required_map_columns = {"contract", "feature", "availability", "risk_level", "model_role", "rationale"}
    missing = required_map_columns - set(feature_map.columns)
    if missing:
        fail(f"prediction_contract_feature_map.csv is missing columns: {sorted(missing)}")
    if not required_contracts.issubset(set(feature_map["contract"])):
        fail("prediction_contract_feature_map.csv does not cover all prediction contracts")


def assert_review_process_and_comment_audit() -> None:
    audit = pd.read_csv(FINAL_DIR / "review_process_feature_audit.csv")
    required = {"feature", "stage", "included_in_t2", "excluded_from_early_reason"}
    missing = required - set(audit.columns)
    if missing:
        fail(f"review_process_feature_audit.csv is missing columns: {sorted(missing)}")
    if len(audit) < 10:
        fail("review_process_feature_audit.csv should audit at least ten process fields")
    if not audit["included_in_t2"].astype(bool).any():
        fail("review_process_feature_audit.csv should identify fields included in T2")

    profile = pd.read_csv(FINAL_DIR / "comment_dataset_profile.csv")
    required_profile = {"metric", "value", "interpretation"}
    missing = required_profile - set(profile.columns)
    if missing:
        fail(f"comment_dataset_profile.csv is missing columns: {sorted(missing)}")
    indexed = profile.set_index("metric")
    for metric in ["raw_rows", "unique_repo_pr_keys", "joined_to_prfeatures"]:
        if metric not in indexed.index:
            fail(f"comment_dataset_profile.csv is missing metric {metric}")
    joined_value = str(indexed.loc["joined_to_prfeatures", "value"]).lower()
    if joined_value not in {"false", "0", "no"}:
        fail("comment_dataset_profile.csv should explicitly report joined_to_prfeatures=false")
    fingerprint = pd.read_csv(FINAL_DIR / "raw_data_fingerprint.csv").set_index("dataset")
    if "pr_comments_dataset_publish" not in fingerprint.index:
        fail("raw_data_fingerprint.csv is missing pr_comments_dataset_publish")
    profile_rows = int(float(indexed.loc["raw_rows", "value"]))
    fingerprint_rows = int(fingerprint.loc["pr_comments_dataset_publish", "rows"])
    if profile_rows != fingerprint_rows:
        fail(
            "raw_data_fingerprint.csv should count parsed CSV records for comments, "
            f"not physical lines: profile={profile_rows}, fingerprint={fingerprint_rows}"
        )


def assert_calibration_model_comparison() -> None:
    calibration = pd.read_csv(FINAL_DIR / "calibration_model_comparison.csv")
    required = {
        "calibration_method",
        "brier_score_not_merged",
        "weighted_abs_calibration_error",
        "average_precision_not_merged",
        "roc_auc_merged",
        "f1_not_merged",
    }
    missing = required - set(calibration.columns)
    if missing:
        fail(f"calibration_model_comparison.csv is missing columns: {sorted(missing)}")
    required_methods = {"uncalibrated", "sigmoid", "isotonic"}
    missing_methods = required_methods - set(calibration["calibration_method"])
    if missing_methods:
        fail(f"calibration_model_comparison.csv is missing methods: {sorted(missing_methods)}")
    if ((calibration["brier_score_not_merged"] < 0) | (calibration["brier_score_not_merged"] > 1)).any():
        fail("calibration_model_comparison.csv contains invalid Brier scores")


def assert_full_generalization_benchmarks() -> None:
    benchmarks = pd.read_csv(FINAL_DIR / "full_generalization_benchmarks.csv")
    required = {
        "benchmark",
        "training_rows",
        "validation_rows",
        "precision_not_merged",
        "recall_not_merged",
        "f1_not_merged",
        "average_precision_not_merged",
    }
    missing = required - set(benchmarks.columns)
    if missing:
        fail(f"full_generalization_benchmarks.csv is missing columns: {sorted(missing)}")
    required_benchmarks = {"official_test", "temporal_holdout", "project_holdout", "creator_holdout"}
    missing_benchmarks = required_benchmarks - set(benchmarks["benchmark"])
    if missing_benchmarks:
        fail(f"full_generalization_benchmarks.csv is missing benchmarks: {sorted(missing_benchmarks)}")
    if (benchmarks["validation_rows"] <= 0).any():
        fail("full_generalization_benchmarks.csv contains an empty validation benchmark")


def assert_paired_model_delta_intervals() -> None:
    deltas = pd.read_csv(FINAL_DIR / "paired_model_delta_intervals.csv")
    required = {
        "baseline_model",
        "comparison_model",
        "metric",
        "delta",
        "ci_lower_95",
        "ci_upper_95",
        "bootstrap_resamples",
    }
    missing = required - set(deltas.columns)
    if missing:
        fail(f"paired_model_delta_intervals.csv is missing columns: {sorted(missing)}")
    required_metrics = {"f1_not_merged", "average_precision_not_merged"}
    missing_metrics = required_metrics - set(deltas["metric"])
    if missing_metrics:
        fail(f"paired_model_delta_intervals.csv is missing metrics: {sorted(missing_metrics)}")
    if (deltas["ci_lower_95"] > deltas["ci_upper_95"]).any():
        fail("paired_model_delta_intervals.csv has inverted confidence intervals")


def assert_no_stale_absolute_paths() -> None:
    stale_path = "/Users/tesssb"
    checked_files = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "dataset-summary.md",
        FINAL_DIR / "README.md",
        FINAL_DIR / "final_report.md",
        PROJECT_ROOT / "slides" / "final_presentation.md",
    ]
    offenders = [
        path.relative_to(PROJECT_ROOT)
        for path in checked_files
        if path.exists() and stale_path in path.read_text()
    ]
    if offenders:
        fail(f"Stale absolute paths remain in: {[str(path) for path in offenders]}")


def assert_no_visible_generation_artifacts() -> None:
    checked_files = [
        FINAL_DIR / "final_report.md",
        PROJECT_ROOT / "slides" / "final_presentation.md",
    ]
    for path in checked_files:
        text = path.read_text().lower()
        if re.search(r"(?<![a-z])nan(?![a-z])", text):
            fail(f"{path.relative_to(PROJECT_ROOT)} contains a visible nan value")


def assert_pdf_readable(path: Path, minimum_pages: int) -> None:
    try:
        import fitz
    except ModuleNotFoundError:
        fail("PyMuPDF is required for PDF validation; install pymupdf in the project environment.")
    doc = fitz.open(path)
    if len(doc) < minimum_pages:
        fail(f"{path.relative_to(PROJECT_ROOT)} has {len(doc)} pages, expected at least {minimum_pages}")
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if not text:
            fail(f"{path.relative_to(PROJECT_ROOT)} page {page_number} has no extractable text")


def main() -> int:
    for filename in REQUIRED_FINAL_FILES:
        assert_exists(FINAL_DIR / filename)
    for filename in REQUIRED_SLIDE_FILES:
        assert_exists(SLIDES_DIR / filename)
    assert_required_figures()

    assert_notebook_has_no_errors(FINAL_DIR / "final_pr_merge_analysis.ipynb")
    for filename in [
        "final_test_metrics.csv",
        "model_comparison.csv",
        "leakage_sensitivity.csv",
        "generalization_stress_tests.csv",
    ]:
        assert_metric_table_consistent(FINAL_DIR / filename)
    assert_assignment_coverage()
    assert_target_distribution_summary()
    assert_eda_key_findings()
    assert_stress_model_comparison()
    assert_split_id_integrity()
    assert_risk_band_table()
    assert_calibration_summary()
    assert_error_analysis_findings()
    assert_prediction_contract_outputs()
    assert_review_process_and_comment_audit()
    assert_calibration_model_comparison()
    assert_full_generalization_benchmarks()
    assert_paired_model_delta_intervals()
    assert_report_claim_language()
    assert_professor_defense_notes()
    assert_no_stale_absolute_paths()
    assert_no_visible_generation_artifacts()
    assert_pdf_readable(FINAL_DIR / "final_report.pdf", minimum_pages=6)
    assert_pdf_readable(SLIDES_DIR / "final_presentation.pdf", minimum_pages=12)
    print("Final output validation passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
