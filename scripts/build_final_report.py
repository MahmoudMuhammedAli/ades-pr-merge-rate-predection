from __future__ import annotations

import re
from pathlib import Path
from textwrap import dedent
from xml.sax.saxutils import escape

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = PROJECT_ROOT / "deliverables" / "final"
REPORT_MD = FINAL_DIR / "final_report.md"
REPORT_PDF = FINAL_DIR / "final_report.pdf"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(FINAL_DIR / name)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def table_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if value.is_integer() and abs(value) >= 100:
            return f"{value:,.0f}"
        return f"{value:.3f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    table = df[columns].copy()
    if max_rows is not None:
        table = table.head(max_rows)
    table = table.fillna("")
    string_rows = [[table_cell(value) for value in row] for row in table.to_numpy()]
    widths = [
        max(len(column), *(len(row[index]) for row in string_rows)) if string_rows else len(column)
        for index, column in enumerate(columns)
    ]

    def row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(values)) + " |"

    header = row(columns)
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [row(values) for values in string_rows]
    return "\n".join([header, separator, *body])


def labeled_table(
    df: pd.DataFrame,
    labels: dict[str, str],
    max_rows: int | None = None,
) -> str:
    table = df[list(labels)].rename(columns=labels)
    return markdown_table(table, list(labels.values()), max_rows=max_rows)


THRESHOLD_LABELS = {
    "default_model_threshold": "Default",
    "validation_tuned_not_merged_f1": "Tuned F1",
}


STRESS_LABELS = {
    "random_stratified_sample": "Random sample",
    "temporal_last_25pct_sample": "Temporal holdout",
    "project_group_holdout_sample": "Project holdout",
    "creator_group_holdout_sample": "Creator holdout",
}


POLICY_LABELS = {
    "ultra_conservative_features": "Strict",
    "headline_leakage_safer_features": "Headline",
    "integrator_assumed_features": "Integrator",
    "extended_timing_assumed_features": "Timing",
}


FAMILY_LABELS = {
    "none": "None",
    "contributor_history": "Contributor history",
    "project_context": "Project context",
    "pr_scope": "PR scope",
    "testing_ci_context": "Testing/CI",
    "language_calendar": "Language/calendar",
}


SPLIT_LABELS = {
    "validation": "Validation",
    "test": "Test",
    "internal_validation": "Internal validation",
    "official_test": "Official test",
}


def with_display_labels(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    if "threshold_label" in display.columns:
        display["threshold_label"] = display["threshold_label"].replace(THRESHOLD_LABELS)
    if "stress_test" in display.columns:
        display["stress_test"] = display["stress_test"].replace(STRESS_LABELS)
    if "feature_set" in display.columns:
        display["feature_set"] = display["feature_set"].replace(POLICY_LABELS)
    if "removed_family" in display.columns:
        display["removed_family"] = display["removed_family"].replace(FAMILY_LABELS)
    if "evaluation_split" in display.columns:
        display["evaluation_split"] = display["evaluation_split"].replace(SPLIT_LABELS)
    return display


def build_markdown() -> str:
    final_metrics = read_csv("final_test_metrics.csv")
    model_comparison = read_csv("model_comparison.csv")
    leakage = read_csv("leakage_sensitivity.csv")
    stress = read_csv("generalization_stress_tests.csv")
    stress_models = read_csv("stress_model_comparison.csv")
    threshold_stability = read_csv("threshold_stability.csv")
    coverage = read_csv("assignment_coverage.csv")
    feature_timing = read_csv("feature_timing_evidence.csv")
    split_overlap = read_csv("split_overlap_summary.csv")
    validation_gap = read_csv("validation_test_gap.csv")
    ablation = read_csv("feature_family_ablation.csv")
    permutation = read_csv("permutation_importance.csv")
    cluster = read_csv("cluster_interpretation.csv")
    eda_findings = read_csv("eda_key_findings.csv")
    risk_bands = read_csv("test_prediction_risk_bands.csv")
    calibration = read_csv("calibration_summary.csv")
    error_profile = read_csv("error_profile_summary.csv")
    error_findings = read_csv("error_analysis_key_findings.csv")
    split_id_integrity = read_csv("split_id_integrity.csv")
    prediction_contracts = read_csv("prediction_contract_comparison.csv")
    comment_profile = read_csv("comment_dataset_profile.csv")
    calibration_models = read_csv("calibration_model_comparison.csv")
    full_benchmarks = read_csv("full_generalization_benchmarks.csv")
    paired_deltas = read_csv("paired_model_delta_intervals.csv")

    default_row = final_metrics[final_metrics["threshold_label"] == "default_model_threshold"].iloc[0]
    tuned_row = final_metrics[final_metrics["threshold_label"] == "validation_tuned_not_merged_f1"].iloc[0]
    best_model = model_comparison.iloc[0]["model"]
    weakest_stress = stress.sort_values("f1_not_merged").iloc[0]
    top_cluster = cluster.sort_values("not_merged_rate", ascending=False).iloc[0]
    final_metrics_display = with_display_labels(final_metrics)
    leakage_display = with_display_labels(leakage)
    ablation_display = with_display_labels(ablation)
    stress_display = with_display_labels(stress)
    stress_winners = with_display_labels(stress_models[stress_models["rank_within_stress_f1"] == 1].copy())
    validation_gap_display = with_display_labels(validation_gap)
    contract_display = prediction_contracts.copy()
    contract_display["contract"] = contract_display["contract"].replace(
        {
            "T0_creation": "T0 creation",
            "T1_diff": "T1 submitted diff",
            "T2_review_process": "T2 review process",
        }
    )
    contract_test = prediction_contracts[prediction_contracts["evaluation_split"] == "test"].set_index("contract")
    comment_profile_indexed = comment_profile.set_index("metric")
    best_calibration = calibration_models.sort_values("brier_score_not_merged").iloc[0]
    full_benchmarks_display = full_benchmarks.copy()
    full_benchmarks_display["benchmark"] = full_benchmarks_display["benchmark"].replace(
        {
            "official_test": "Official test",
            "temporal_holdout": "Temporal",
            "project_holdout": "Project holdout",
            "creator_holdout": "Creator holdout",
        }
    )
    paired_f1_delta = paired_deltas[paired_deltas["metric"] == "f1_not_merged"].iloc[0]
    threshold_median = threshold_stability["best_threshold"].median()
    threshold_min = threshold_stability["best_threshold"].min()
    threshold_max = threshold_stability["best_threshold"].max()
    tuned_f1_mean = threshold_stability["tuned_f1_not_merged"].mean()
    top_risk_decile = risk_bands.sort_values("risk_decile").iloc[-1]
    bottom_risk_decile = risk_bands.sort_values("risk_decile").iloc[0]
    calibration_ece = (
        calibration["absolute_calibration_error"] * calibration["row_count"]
    ).sum() / calibration["row_count"].sum()
    calibration_brier = calibration["brier_score_not_merged"].iloc[0]
    id_overlap = split_id_integrity.set_index("check").loc["train_test_id_overlap", "value"]
    error_profile_display = error_profile.copy()
    error_profile_display["outcome_group"] = (
        error_profile_display["outcome_group"].str.replace("_", " ").str.title()
    )
    error_findings_text = "\n".join(
        (
            f"- **{row.finding}.** {row.evidence} "
            f"{row.interpretation}"
        )
        for row in error_findings.itertuples(index=False)
    )

    markdown = dedent(
        f"""
        # Final Report: PR-Level Signals for GitHub Merge Outcomes

        ## Executive Answer

        The project asks whether PR-level features can help explain and predict GitHub PR merge outcomes. The answer is yes, but the early-information answer is only **moderate predictive association**. The selected early model is **{best_model}**. On the untouched official test file, the default threshold reaches not-merged precision {fmt(default_row["precision_not_merged"])}, recall {fmt(default_row["recall_not_merged"])}, and F1 {fmt(default_row["f1_not_merged"])}. A validation-tuned threshold raises not-merged F1 to {fmt(tuned_row["f1_not_merged"])} while lowering recall.

        The upgraded result is the prediction-time contract study: T0 PR-creation features reach test F1 {fmt(contract_test.loc["T0_creation", "f1_not_merged"])}, T1 submitted-diff features reach {fmt(contract_test.loc["T1_diff", "f1_not_merged"])}, and T2 review-process features reach {fmt(contract_test.loc["T2_review_process", "f1_not_merged"])}. This is **not causal** evidence and is **not suitable for automated merge decisions**. The result is useful for empirical understanding because it separates early PR context from information that becomes available only after review has started, under an explicit feature-availability contract and external-validity checks.

        ![Figure 1. Training target distribution](figures/target_distribution.png)

        ## Dataset and Prediction-Time Contracts

        The analysis uses the Zenodo PRFeatures train/test files from "GitHub Pull Request Analysis: Sentiment Data and Developer Survey Responses." The target is `merged_or_not`. The comment dataset and survey file are documented but not used as the main modeling source because the PR-level files already match the research question and comment-derived fields create stronger timing risk.

        The official train/test files are large and class-imbalanced: about 89% merged and 11% not merged. A majority baseline has high accuracy but zero utility for detecting not-merged PRs, so the report emphasizes not-merged precision, recall, F1, balanced accuracy, average precision, and ROC-AUC.

        The headline model uses the `headline_leakage_safer_features` policy. Identifiers and direct post-outcome fields are excluded. Review discussion, sentiment, CI outcomes, success-rate fields, lifetime/close-time fields, and close-time PR evolution fields are held back from the early model. They appear only in a separate T2 review-process contract so the project does not pretend late review information was available at PR creation.

        The separate comment dataset contains {int(float(comment_profile_indexed.loc["raw_rows", "value"])):,} parsed comment rows and {int(float(comment_profile_indexed.loc["unique_repo_pr_keys", "value"])):,} owner/repository/pull-number keys. It is profiled but not joined to PRFeatures because the PRFeatures files expose numeric ids, while the comment file exposes owner/repo/pull number. This avoids forcing a bad linkage.

        {labeled_table(feature_timing.head(10), {"feature": "Feature", "documented_timing": "Timing", "risk_level": "Risk", "model_role": "Role"})}

        Professor-facing safeguards:

        | Risk | What the project does | Why this matters |
        | ---- | --------------------- | ---------------- |
        | Feature timing ambiguity | Uses a headline leakage-safer policy, an ultra-conservative sensitivity model, and a later T2 review-process contract. | A stronger score is not allowed to silently become the early-prediction answer. |
        | Test-set overfitting | Selects model and threshold on internal validation, then reports the untouched official test once. | The final test result is evaluation evidence, not model-selection evidence. |
        | Entity memorization | Excludes direct ids and reports project/creator overlap plus group-holdout diagnostics. | The official split is treated as familiar-ecosystem generalization, not unseen-project proof. |
        | Overstated probabilities | Separates ranking/F1 results from calibration diagnostics. | Calibrated probabilities are discussed as probability quality, not as better classification performance. |
        | Weak clustering | Frames K-means/PCA as profile interpretation only. | The unsupervised task supports characterization, not the main predictive claim. |

        ## Exploratory Data Analysis

        EDA is not used to claim causality. It establishes why imbalance-aware metrics, contributor/project features, and group-aware validation are necessary.

        ![Figure 2. EDA patterns: PR size, contributor context, and entity concentration](figures/eda_safe_feature_patterns.png)

        {labeled_table(eda_findings, {"finding": "Finding", "evidence": "Evidence", "interpretation": "Interpretation"}, max_rows=8)}

        ## Supervised Study

        Models compared on internal validation include a dummy majority classifier, balanced logistic regression, balanced decision tree, balanced random forest, and weighted histogram gradient boosting. The primary selection metric is not-merged F1 because not-merged PRs are the minority class.

        ![Figure 3. Validation model comparison](figures/model_comparison.png)

        {labeled_table(model_comparison, {"model": "Model", "balanced_accuracy": "Bal. acc.", "precision_not_merged": "P0", "recall_not_merged": "R0", "f1_not_merged": "F1-0", "average_precision_not_merged": "AP-0"})}

        Final official test metrics:

        {labeled_table(final_metrics_display, {"threshold_label": "Threshold", "accuracy": "Acc.", "balanced_accuracy": "Bal. acc.", "precision_not_merged": "P0", "recall_not_merged": "R0", "f1_not_merged": "F1-0", "average_precision_not_merged": "AP-0"})}

        The validation-only tuned threshold is stable enough to report separately: across six sampled folds, the median selected threshold is {fmt(threshold_median)}, with range {fmt(threshold_min)} to {fmt(threshold_max)}. The tuned fold mean not-merged F1 is {fmt(tuned_f1_mean)}.

        ![Figure 4. Final tuned-threshold confusion matrix](figures/final_confusion_matrix.png)

        ## Prediction-Time Contract Study

        The central upgrade is that the project no longer treats all predictors as one flat feature set. It asks when each signal becomes available. T0 is the PR-creation model, T1 is the submitted-diff model, and T2 is the review-process model. T2 is allowed to use discussion, sentiment, CI-progress, and reviewer/integrator context, so a stronger T2 score is interpreted as later decision-process signal rather than a deployable early predictor.

        ![Figure 5. Prediction contract comparison](figures/prediction_contract_comparison.png)

        {labeled_table(contract_display, {"contract": "Contract", "evaluation_split": "Split", "feature_count": "Features", "precision_not_merged": "P0", "recall_not_merged": "R0", "f1_not_merged": "F1-0", "average_precision_not_merged": "AP-0"})}

        The contract comparison improves the answer: early PR-level context is already predictive, submitted-diff features add little beyond that, and review-process fields add a much larger signal. This pattern makes the result less forced because it explains why stronger metrics require later information.

        Paired model-delta bootstrap:

        {labeled_table(paired_deltas, {"comparison_model": "Comparison", "metric": "Metric", "delta": "Delta", "ci_lower_95": "CI low", "ci_upper_95": "CI high"})}

        ## Risk Ranking, Calibration, and Error Analysis

        The final model is more useful as a ranking signal than as a hard decision engine. The highest predicted-risk decile has an observed not-merged rate of {fmt(top_risk_decile["actual_not_merged_rate_pct"], 2)}%, compared with the test baseline of {fmt(top_risk_decile["baseline_not_merged_rate_pct"], 2)}% and {fmt(bottom_risk_decile["actual_not_merged_rate_pct"], 2)}% in the lowest-risk decile. This lift supports the claim that the model captures signal even though the tuned F1 remains moderate.

        ![Figure 6. Risk-band lift on the official test split](figures/risk_band_lift.png)

        Calibration is imperfect: the weighted mean absolute calibration error is {fmt(calibration_ece)} and the not-merged Brier score is {fmt(calibration_brier)}. Therefore predicted scores should be treated as relative risk scores, not literal probabilities.

        ![Figure 7. Calibration of not-merged scores](figures/calibration_curve.png)

        Calibrating a held-out validation slice materially improves probability honesty. The best calibration method by Brier score is `{best_calibration["calibration_method"]}` with Brier {fmt(best_calibration["brier_score_not_merged"])} and weighted absolute calibration error {fmt(best_calibration["weighted_abs_calibration_error"])}. This does not replace the F1-based classifier, and the calibrated default-threshold F1 is lower because calibrated minority-class probabilities are conservative under the 11% base rate. The calibration result fixes a narrower weakness: raw Random Forest scores should not be read as literal probabilities.

        ![Figure 8. Calibration method comparison](figures/calibration_model_comparison.png)

        {labeled_table(calibration_models, {"calibration_method": "Method", "brier_score_not_merged": "Brier", "weighted_abs_calibration_error": "Cal. error", "average_precision_not_merged": "AP-0", "f1_not_merged": "F1-0"})}

        Tuned-threshold error profiles show what the model gets wrong:

        {labeled_table(error_profile_display, {"outcome_group": "Outcome group", "row_count": "Rows", "share_of_test_pct": "Test %", "mean_predicted_not_merged_score": "Mean P0 score", "median_open_pr_num": "Median open PRs", "median_stars": "Median stars", "median_churn_addition": "Median added lines"})}

        {error_findings_text}

        ## Generalization and Robustness

        The official test split is useful, but it is not an unseen-project split. Many test projects and creators also appear in the training file. This means the official holdout mostly tests new PRs in familiar ecosystems, not generalization to entirely unseen projects.

        Exact PR-id integrity is clean: train/test PR id overlap is {int(id_overlap)}, and duplicate ids are zero in both official files. The remaining split concern is entity reuse, not duplicate PR leakage.

        {labeled_table(split_overlap, {"entity": "Entity", "train_unique": "Train unique", "test_unique": "Test unique", "test_rows_with_seen_entity_pct": "Seen test rows %"})}

        Additional sampled stress tests use random, temporal, project-group, and creator-group validation. The weakest stress case is {STRESS_LABELS.get(weakest_stress["stress_test"], weakest_stress["stress_test"])} with not-merged F1 {fmt(weakest_stress["f1_not_merged"])}.

        {labeled_table(stress_display, {"stress_test": "Stress test", "balanced_accuracy": "Bal. acc.", "precision_not_merged": "P0", "recall_not_merged": "R0", "f1_not_merged": "F1-0", "roc_auc_merged": "ROC"})}

        A separate model-family stress comparison supports the Random Forest choice, but not as an absolute winner in every stricter split. Random Forest leads the random, temporal, and creator-group diagnostics; histogram gradient boosting narrowly leads the project-group holdout. The submitted model remains Random Forest because the model-selection rule was declared on not-merged F1 before test scoring and because its stress behavior is competitive without changing the feature contract.

        {labeled_table(stress_winners, {"stress_test": "Stress test", "model": "Best F1 model", "balanced_accuracy": "Bal. acc.", "f1_not_merged": "F1-0", "average_precision_not_merged": "AP-0"})}

        The validation-to-test gap is reported directly:

        {labeled_table(validation_gap_display, {"evaluation_split": "Split", "threshold_label": "Threshold", "balanced_accuracy": "Bal. acc.", "f1_not_merged": "F1-0", "average_precision_not_merged": "AP-0"})}

        The larger generalization benchmark repeats the concern with a 500k-row holdout design for temporal, project-group, and creator-group splits. These rows are not intended to exactly match the sampled stress-test values above: the samples, validation rates, and group partitions differ. The consistent conclusion is the important one: the official split is useful, but external-validity is weaker under stricter time/entity holdouts.

        ![Figure 9. Large generalization benchmarks](figures/full_generalization_benchmarks.png)

        {labeled_table(full_benchmarks_display, {"benchmark": "Benchmark", "training_rows": "Train rows", "validation_rows": "Eval rows", "precision_not_merged": "P0", "recall_not_merged": "R0", "f1_not_merged": "F1-0", "average_precision_not_merged": "AP-0"})}

        ## Explanation Evidence

        Random Forest impurity importance is reported, but the stronger explanation checks are permutation importance and feature-family ablation. These do not prove causality; they show which feature groups the model relies on for predictive performance.

        ![Figure 10. Permutation importance](figures/permutation_importance.png)

        {labeled_table(permutation, {"feature": "Feature", "importance_mean": "Mean F1 drop", "importance_std": "Std."}, max_rows=10)}

        Feature-family ablation:

        {labeled_table(ablation_display, {"removed_family": "Removed", "f1_not_merged": "F1-0", "delta_f1_not_merged_vs_full": "Delta F1", "average_precision_not_merged": "AP-0"}, max_rows=8)}

        ## Sensitivity and Unsupervised Profiles

        Feature-policy sensitivity shows that the stricter headline result is lower than the integrator-assumed result, which is exactly why `prior_review_num` should remain outside the headline model. K-means/PCA is used only for unsupervised PR-profile interpretation. The highest not-merged cluster is cluster {int(top_cluster["cluster"])} with a not-merged rate of {fmt(top_cluster["not_merged_rate"] / 100)} and label `{top_cluster["cluster_label"]}`.

        {labeled_table(leakage_display, {"feature_set": "Policy", "evaluation_split": "Split", "precision_not_merged": "P0", "recall_not_merged": "R0", "f1_not_merged": "F1-0", "average_precision_not_merged": "AP-0"})}

        ![Figure 11. K-means/PCA profile view](figures/cluster_pca.png)

        ## Appendix Evidence Map

        {labeled_table(coverage, {"requirement": "Requirement", "weight_or_source": "Weight", "artifact": "Primary evidence"}, max_rows=8)}

        ## Threats to Validity

        The main threat is feature timing: some project/context fields are treated as submission-time snapshots because the source field documentation supports that interpretation, but inherited preprocessing choices cannot be fully audited from the provided CSVs. The official test split has strong project and creator overlap with training. Bootstrap intervals quantify sampling uncertainty but not feature-timing bias, model-selection uncertainty, or external-validity limits. The conclusions are therefore empirical and associational, not causal or operational.

        ## References

        - Zenodo dataset: https://zenodo.org/records/10049493
        - new_pullreq field README: https://www.gitlink.org.cn/Raining/new_pullreq_dataset
        - MSR 2020 dataset paper: https://yuyue.github.io/res/paper/newPR_MSR2020.pdf
        - Zhang et al. PR decisions paper: https://zhangxunhui.github.io/files/TSE_2022_zxh.pdf
        """
    ).strip()
    cleaned_lines = [line[8:] if line.startswith("        ") else line for line in markdown.splitlines()]
    return "\n".join(cleaned_lines) + "\n"


def paragraph_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    return lines


def clean_inline_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = text.replace("`", "")
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    return escape(text)


def build_pdf(markdown: str) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ModuleNotFoundError as exc:
        raise SystemExit("reportlab is required to build final_report.pdf. Install it with pip or uv.") from exc

    styles = getSampleStyleSheet()
    available_width = A4[0] - 3.0 * cm
    doc = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
    )
    story = []
    in_table = False
    table_rows: list[list[str]] = []

    def flush_table() -> None:
        nonlocal table_rows
        if not table_rows:
            return
        clean_rows = [
            [Paragraph(clean_inline_markdown(cell), styles["BodyText"]) for cell in row]
            for row in table_rows
            if not all(set(cell.replace(" ", "")) <= {"-", ":"} for cell in row)
        ]
        if clean_rows:
            col_count = len(clean_rows[0])
            table = Table(clean_rows, repeatRows=1, colWidths=[available_width / col_count] * col_count)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF5")),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B8C2CC")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 8))
        table_rows = []

    for line in paragraph_lines(markdown):
        if line.startswith("|") and line.endswith("|"):
            in_table = True
            table_rows.append([cell.strip() for cell in line.strip("|").split("|")])
            continue
        if in_table:
            flush_table()
            in_table = False
        image_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if image_match:
            caption, image_path = image_match.groups()
            path = (FINAL_DIR / image_path).resolve()
            if path.exists():
                image = Image(str(path))
                scale = min(1.0, available_width / image.drawWidth)
                image.drawWidth *= scale
                image.drawHeight *= scale
                story.append(image)
                if caption:
                    story.append(Paragraph(clean_inline_markdown(caption), styles["Italic"]))
                story.append(Spacer(1, 8))
            continue
        if line.startswith("# "):
            story.append(Paragraph(clean_inline_markdown(line[2:]), styles["Title"]))
            story.append(Spacer(1, 10))
        elif line.startswith("## "):
            story.append(Spacer(1, 8))
            story.append(Paragraph(clean_inline_markdown(line[3:]), styles["Heading2"]))
        elif line.startswith("- "):
            story.append(Paragraph(clean_inline_markdown(line), styles["BodyText"]))
        else:
            story.append(Paragraph(clean_inline_markdown(line), styles["BodyText"]))
            story.append(Spacer(1, 4))
    flush_table()
    doc.build(story)


def main() -> int:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    markdown = build_markdown()
    REPORT_MD.write_text(markdown)
    build_pdf(markdown)
    print(f"Wrote {REPORT_MD.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {REPORT_PDF.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
