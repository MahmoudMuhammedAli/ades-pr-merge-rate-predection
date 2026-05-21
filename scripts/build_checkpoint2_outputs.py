from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = PROJECT_ROOT / "deliverables" / "final"
CHECKPOINT2_DIR = PROJECT_ROOT / "deliverables" / "checkpoint-2"
FIGURE_DIR = CHECKPOINT2_DIR / "figures"


def read_final_csv(filename: str) -> pd.DataFrame:
    path = FINAL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing final artifact: {path.relative_to(PROJECT_ROOT)}")
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, filename: str) -> Path:
    path = CHECKPOINT2_DIR / filename
    df.to_csv(path, index=False)
    return path


def fmt_int(value: float | int) -> str:
    return f"{int(value):,}"


def fmt_pct(value: float | int) -> str:
    return f"{float(value):.2f}%"


def quality_value(quality: pd.DataFrame, split: str, check: str) -> float:
    row = quality[(quality["split"] == split) & (quality["check"] == check)]
    if row.empty:
        raise KeyError(f"Missing quality row for split={split!r}, check={check!r}")
    return float(row.iloc[0]["value"])


def target_value(target: pd.DataFrame, split: str, label: str, column: str) -> float:
    row = target[(target["split"] == split) & (target["label"] == label)]
    if row.empty:
        raise KeyError(f"Missing target row for split={split!r}, label={label!r}")
    return float(row.iloc[0][column])


def build_model_comparison() -> pd.DataFrame:
    comparison = read_final_csv("model_comparison.csv")
    cp2 = comparison[
        comparison["training_scope"].eq("full_source_internal_validation")
        & comparison["feature_policy"].eq("headline_leakage_safer_features")
    ].copy()
    if cp2.empty:
        raise ValueError("No full-source internal validation rows found in model_comparison.csv")
    cp2 = cp2.sort_values(["f1_not_merged", "balanced_accuracy"], ascending=False)
    return cp2


def build_target_distribution() -> pd.DataFrame:
    return read_final_csv("target_distribution_summary.csv")


def build_data_characteristics(model_comparison: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    quality = read_final_csv("data_quality_summary.csv")
    numeric = read_final_csv("numeric_distribution_summary.csv")
    split_overlap = read_final_csv("split_overlap_summary.csv")
    concentration = read_final_csv("project_creator_concentration.csv")

    extreme = numeric[numeric["outlier_flag"].eq("extreme_tail")]
    extreme_examples = ", ".join(extreme.head(5)["feature"].tolist())
    dummy = model_comparison[model_comparison["model"].eq("Dummy majority")].iloc[0]
    project_overlap = split_overlap[split_overlap["entity"].eq("project_id")].iloc[0]
    creator_overlap = split_overlap[split_overlap["entity"].eq("creator_id")].iloc[0]
    project_top = concentration[
        (concentration["split"].eq("train"))
        & (concentration["entity"].eq("project_id"))
        & (concentration["top_n"].eq(500))
    ].iloc[0]

    rows = [
        {
            "topic": "train rows",
            "evidence": fmt_int(quality_value(quality, "train", "rows")),
            "checkpoint2_implication": "Large enough for internal validation and stratified modeling.",
            "source_artifact": "data_quality_summary.csv",
        },
        {
            "topic": "test rows",
            "evidence": fmt_int(quality_value(quality, "test", "rows")),
            "checkpoint2_implication": "Reserved for final evaluation; not used for Checkpoint 2 model or threshold selection.",
            "source_artifact": "data_quality_summary.csv",
        },
        {
            "topic": "number of columns",
            "evidence": (
                f"{fmt_int(quality_value(quality, 'train', 'columns_loaded'))} analysis columns loaded; "
                "project README documents 72 PRFeatures columns in the raw train/test files."
            ),
            "checkpoint2_implication": "The milestone focuses on a curated feature policy rather than all raw fields.",
            "source_artifact": "data_quality_summary.csv; README.md",
        },
        {
            "topic": "target variable",
            "evidence": "merged_or_not, where 1 means merged and 0 means not merged.",
            "checkpoint2_implication": "The minority class of interest is not-merged PRs.",
            "source_artifact": "README.md; target_distribution_summary.csv",
        },
        {
            "topic": "merged/not-merged rate",
            "evidence": (
                f"Train: {fmt_pct(target_value(target, 'train', 'Merged', 'percentage'))} merged / "
                f"{fmt_pct(target_value(target, 'train', 'Not merged', 'percentage'))} not merged; "
                f"test: {fmt_pct(target_value(target, 'test', 'Merged', 'percentage'))} merged / "
                f"{fmt_pct(target_value(target, 'test', 'Not merged', 'percentage'))} not merged."
            ),
            "checkpoint2_implication": "The task is strongly imbalanced, so minority-class metrics matter more than raw accuracy.",
            "source_artifact": "target_distribution_summary.csv",
        },
        {
            "topic": "explicit null/missingness",
            "evidence": (
                f"{fmt_int(quality_value(quality, 'train', 'explicit_null_cells'))} explicit null cells in train and "
                f"{fmt_int(quality_value(quality, 'test', 'explicit_null_cells'))} in test."
            ),
            "checkpoint2_implication": "Explicit nulls are not the main issue; semantic missingness from source preprocessing remains a limitation.",
            "source_artifact": "data_quality_summary.csv",
        },
        {
            "topic": "duplicates",
            "evidence": (
                f"{fmt_int(quality_value(quality, 'train', 'duplicate_loaded_rows'))} duplicate loaded rows in train and "
                f"{fmt_int(quality_value(quality, 'test', 'duplicate_loaded_rows'))} in test."
            ),
            "checkpoint2_implication": "No duplicate-row correction is needed for the loaded modeling frame.",
            "source_artifact": "data_quality_summary.csv",
        },
        {
            "topic": "categorical and high-cardinality handling",
            "evidence": "language is treated as nominal; project/creator/PR identifiers are excluded.",
            "checkpoint2_implication": "One-hot language encoding is kept, while identifiers are excluded to reduce memorization risk.",
            "source_artifact": "prediction_contract_feature_map.csv; feature_timing_summary.csv",
        },
        {
            "topic": "heavy-tailed numeric variables",
            "evidence": f"{len(extreme)} modeled variables flagged extreme_tail; examples: {extreme_examples}.",
            "checkpoint2_implication": "Median imputation and scaling support linear baselines; tree models handle skew more naturally.",
            "source_artifact": "numeric_distribution_summary.csv",
        },
        {
            "topic": "project/creator overlap",
            "evidence": (
                f"{fmt_pct(project_overlap['test_rows_with_seen_entity_pct'])} of test rows have a seen project; "
                f"{fmt_pct(creator_overlap['test_rows_with_seen_entity_pct'])} have a seen creator."
            ),
            "checkpoint2_implication": "Official split is not an unseen-project test; Checkpoint 2 should use internal validation for selection.",
            "source_artifact": "split_overlap_summary.csv",
        },
        {
            "topic": "project concentration",
            "evidence": f"Top 500 train projects cover {fmt_pct(project_top['row_pct'])} of train rows.",
            "checkpoint2_implication": "Repository-level concentration motivates later generalization stress tests.",
            "source_artifact": "project_creator_concentration.csv",
        },
        {
            "topic": "why raw accuracy is insufficient",
            "evidence": (
                f"Dummy majority validation accuracy is {dummy['accuracy']:.3f}, but not-merged F1 is "
                f"{dummy['f1_not_merged']:.3f}."
            ),
            "checkpoint2_implication": "Checkpoint 2 leads with not-merged F1, recall, balanced accuracy, and average precision.",
            "source_artifact": "model_comparison.csv",
        },
    ]
    return pd.DataFrame(rows)


def build_feature_contract() -> pd.DataFrame:
    timing = read_final_csv("feature_timing_summary.csv")
    used = timing[timing["group"].eq("used")].iloc[0]["fields"]
    held = timing[timing["group"].eq("held back")].iloc[0]["fields"]
    excluded = timing[timing["group"].eq("excluded")].iloc[0]["fields"]

    return pd.DataFrame(
        [
            {
                "feature_group": "headline_leakage_safer_features",
                "checkpoint2_usage": "used in headline Checkpoint 2 supervised model",
                "examples": used,
                "reason": "Available under the stated submission-time or historical-context assumptions.",
                "risk_level": "low-to-medium",
                "presentation_wording": "Leakage-safer under stated assumptions, not perfectly safe.",
            },
            {
                "feature_group": "excluded_post_outcome_or_identifier_features",
                "checkpoint2_usage": "excluded",
                "examples": excluded,
                "reason": "Identifiers, closure time, lifetime, and reopen fields can leak outcomes or encourage memorization.",
                "risk_level": "high",
                "presentation_wording": "Excluded because they are target-adjacent, post-outcome, or identifier-like.",
            },
            {
                "feature_group": "sensitivity_only_features",
                "checkpoint2_usage": "not headline; reserved for sensitivity analysis",
                "examples": "prior_review_num; pr_succ_rate; requester_succ_rate; num_commits; src_churn; files_changed; test_churn",
                "reason": "Availability depends on stronger timing or integrator/reviewer assumptions.",
                "risk_level": "medium-to-high",
                "presentation_wording": "Sensitivity-only; prior_review_num is not part of the headline model.",
            },
            {
                "feature_group": "T2_review_process_features",
                "checkpoint2_usage": "separate later-stage analysis only",
                "examples": held,
                "reason": "Review discussion, sentiment, CI progress, and integrator context happen after review starts.",
                "risk_level": "high for early prediction",
                "presentation_wording": "T2 features answer a later-stage question and are not mixed into the early headline model.",
            },
            {
                "feature_group": "comments_and_survey",
                "checkpoint2_usage": "optional context only; not main ML input",
                "examples": "pr_comments_dataset_publish.csv; survey_responses_raw.csv",
                "reason": "Comments are later-stage process evidence; survey responses are too small/scope-limited for the main PRFeatures task.",
                "risk_level": "scope-limited",
                "presentation_wording": "Useful context, but outside the Checkpoint 2 headline supervised model.",
            },
        ]
    )


def build_preprocessing_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pipeline_step": "numeric features",
                "columns_or_scope": "continuous counts, rates, age, size, and churn fields",
                "method": "median imputation followed by StandardScaler",
                "reason": "Median imputation is robust to skew; scaling supports logistic regression and distance-based diagnostics.",
                "implementation_note": "Implemented inside sklearn ColumnTransformer.",
            },
            {
                "pipeline_step": "binary features",
                "columns_or_scope": "first_pr, core_member, test_inclusion, ci_exists, friday_effect",
                "method": "most-frequent imputation",
                "reason": "Binary flags need stable missing-value handling without imposing numeric scaling.",
                "implementation_note": "Implemented inside sklearn ColumnTransformer.",
            },
            {
                "pipeline_step": "categorical language feature",
                "columns_or_scope": "language",
                "method": "most-frequent imputation plus OneHotEncoder(handle_unknown='ignore')",
                "reason": "Language is nominal and future/validation categories should not break inference.",
                "implementation_note": "Encoded in the categorical branch of the ColumnTransformer.",
            },
            {
                "pipeline_step": "model wrapping",
                "columns_or_scope": "all supervised models",
                "method": "sklearn Pipeline",
                "reason": "Keeps preprocessing fit on training folds/splits only.",
                "implementation_note": "Each candidate model is wrapped with the same preprocessing contract.",
            },
            {
                "pipeline_step": "class imbalance",
                "columns_or_scope": "supervised model fitting",
                "method": "class_weight='balanced' or balanced sample weights where supported",
                "reason": "Merged PRs dominate; the minority not-merged class is the evaluation focus.",
                "implementation_note": "Dummy majority remains unweighted to show the raw-accuracy trap.",
            },
            {
                "pipeline_step": "split discipline",
                "columns_or_scope": "model and threshold selection",
                "method": "internal validation for Checkpoint 2 selection; official test reserved for final reporting",
                "reason": "Avoids using the official test split as a tuning target.",
                "implementation_note": "Checkpoint 2 headline metrics come from full-source internal validation artifacts.",
            },
        ]
    )


def build_algorithm_rationale() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "algorithm": "Dummy majority",
                "course_connection": "Baseline and class-imbalance sanity check",
                "why_included": "Shows that high raw accuracy can be meaningless when the minority class is ignored.",
                "main_assumption_or_bias": "Always predicts the majority merged class.",
                "main_risk": "Looks superficially strong on accuracy while detecting no not-merged PRs.",
                "required_preprocessing": "Same feature pipeline for comparability, though features are ignored.",
                "checkpoint2_role": "Lower-bound baseline.",
            },
            {
                "algorithm": "Logistic regression balanced",
                "course_connection": "Linear supervised baseline",
                "why_included": "Tests whether a simple weighted linear separator captures useful signal.",
                "main_assumption_or_bias": "Additive linear decision boundary after preprocessing.",
                "main_risk": "Misses non-linear interactions and can be affected by skewed variables.",
                "required_preprocessing": "Imputation, scaling, one-hot language encoding, balanced class weights.",
                "checkpoint2_role": "Interpretable baseline against stronger models.",
            },
            {
                "algorithm": "Decision tree / CART",
                "course_connection": "Interpretable non-linear classifier",
                "why_included": "Captures threshold-style feature interactions in a single constrained tree.",
                "main_assumption_or_bias": "Recursive axis-aligned splits approximate the decision structure.",
                "main_risk": "Can be unstable and overfit if depth/leaf sizes are not constrained.",
                "required_preprocessing": "Imputation and one-hot language encoding; scaling is harmless through shared pipeline.",
                "checkpoint2_role": "Simple non-linear comparison.",
            },
            {
                "algorithm": "Random forest balanced",
                "course_connection": "Ensemble learning and variance reduction",
                "why_included": "Averages many class-balanced trees and is the current validation F1 leader.",
                "main_assumption_or_bias": "Many decorrelated trees improve robustness over one tree.",
                "main_risk": "Feature importance is predictive, not causal, and can favor continuous/high-cardinality fields.",
                "required_preprocessing": "Shared imputation/encoding pipeline; balanced subsample class weights.",
                "checkpoint2_role": "Current headline validation model.",
            },
            {
                "algorithm": "Histogram Gradient Boosting weighted",
                "course_connection": "Boosted tree ensemble",
                "why_included": "Provides a stronger sequential tree comparator with class-balanced sample weights.",
                "main_assumption_or_bias": "Sequential learners can improve residual errors over additive trees.",
                "main_risk": "May trade precision/recall differently and is not selected solely because it is more complex.",
                "required_preprocessing": "Shared imputation/encoding pipeline and balanced sample weights.",
                "checkpoint2_role": "Stronger supervised comparator.",
            },
            {
                "algorithm": "KMeans-PCA profile analysis",
                "course_connection": "Unsupervised learning and dimensionality reduction",
                "why_included": "Profiles PR groups without using the target during clustering.",
                "main_assumption_or_bias": "KMeans assumes compact scaled clusters; PCA shows only a partial low-dimensional view.",
                "main_risk": "Not a classifier and should not be overinterpreted.",
                "required_preprocessing": "Imputation, scaling, one-hot language encoding before clustering/PCA.",
                "checkpoint2_role": "Optional short profile-discovery section.",
            },
        ]
    )


def build_validation_confusion(model_comparison: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "feature_policy",
        "feature_count",
        "training_rows",
        "validation_rows",
        "training_scope",
        "model",
        "actual_not_merged_predicted_not_merged",
        "actual_not_merged_predicted_merged",
        "actual_merged_predicted_not_merged",
        "actual_merged_predicted_merged",
    ]
    return model_comparison[columns].copy()


def build_threshold_tuning() -> pd.DataFrame:
    curve = read_final_csv("threshold_tuning_curve_sample.csv")
    top = read_final_csv("threshold_tuning_top.csv").head(25)
    curve = curve.copy()
    curve["source_table"] = "curve_sample"
    top = top.copy()
    top["source_table"] = "top_validation_f1"
    return pd.concat([top, curve], ignore_index=True).drop_duplicates(subset=["threshold"])


def build_feature_importance() -> pd.DataFrame:
    importance = read_final_csv("feature_importance.csv")
    return importance[importance["feature_policy"].eq("headline_leakage_safer_features")].copy()


def plot_target_distribution(target: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    pivot = target.pivot(index="split", columns="label", values="percentage")
    pivot[["Merged", "Not merged"]].plot(kind="bar", stacked=True, ax=ax, color=["#4C78A8", "#F58518"])
    ax.set_ylabel("Percentage")
    ax.set_xlabel("")
    ax.set_title("Checkpoint 2 Target Distribution")
    ax.legend(title="")
    ax.set_ylim(0, 100)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", label_type="center", color="white", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "checkpoint2_target_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_model_metric(model_comparison: pd.DataFrame, metric: str, filename: str, title: str) -> None:
    plot_df = model_comparison.sort_values(metric, ascending=True)
    colors = ["#4C78A8" if model != "Dummy majority" else "#A0A0A0" for model in plot_df["model"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(plot_df["model"], plot_df[metric], color=colors)
    ax.set_xlim(0, max(0.75, float(plot_df[metric].max()) + 0.08))
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("")
    ax.set_title(title)
    for index, value in enumerate(plot_df[metric]):
        ax.text(float(value) + 0.01, index, f"{float(value):.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def plot_feature_policy_summary(contract: pd.DataFrame) -> None:
    severity = {
        "low-to-medium": 2,
        "high": 4,
        "medium-to-high": 3,
        "high for early prediction": 4,
        "scope-limited": 1,
    }
    plot_df = contract.copy()
    plot_df["risk_score"] = plot_df["risk_level"].map(severity).fillna(2)
    colors = ["#54A24B", "#E45756", "#F58518", "#B279A2", "#72B7B2"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(plot_df["feature_group"], plot_df["risk_score"], color=colors[: len(plot_df)])
    ax.set_xlim(0, 4.5)
    ax.set_xlabel("Relative timing/leakage risk")
    ax.set_ylabel("")
    ax.set_title("Checkpoint 2 Feature Policy Summary")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["scope", "low-med", "med-high", "high"])
    for index, row in plot_df.iterrows():
        ax.text(row["risk_score"] + 0.06, index, row["checkpoint2_usage"], va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "checkpoint2_feature_policy_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_validation_confusion(confusion: pd.DataFrame) -> None:
    best = confusion.sort_values("model").copy()
    best = confusion[confusion["model"].eq("Random forest balanced")].iloc[0]
    matrix = [
        [
            best["actual_not_merged_predicted_not_merged"],
            best["actual_not_merged_predicted_merged"],
        ],
        [
            best["actual_merged_predicted_not_merged"],
            best["actual_merged_predicted_merged"],
        ],
    ]
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred not merged", "Pred merged"])
    ax.set_yticks([0, 1], labels=["Actual not merged", "Actual merged"])
    ax.set_title("Validation Confusion Matrix: Random Forest")
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, fmt_int(value), ha="center", va="center", color="#111111")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "checkpoint2_confusion_matrix_validation.png", bbox_inches="tight")
    plt.close(fig)


def plot_threshold_tradeoff(thresholds: pd.DataFrame) -> None:
    curve = thresholds[thresholds["source_table"].eq("curve_sample")].sort_values("threshold")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(curve["threshold"], curve["precision_not_merged"], label="Precision not merged", color="#4C78A8")
    ax.plot(curve["threshold"], curve["recall_not_merged"], label="Recall not merged", color="#F58518")
    ax.plot(curve["threshold"], curve["f1_not_merged"], label="F1 not merged", color="#54A24B")
    top = thresholds[thresholds["source_table"].eq("top_validation_f1")].sort_values("f1_not_merged", ascending=False).iloc[0]
    ax.axvline(top["threshold"], color="#E45756", linestyle="--", linewidth=1.5, label="Best validation F1 threshold")
    ax.set_xlabel("Not-merged probability threshold")
    ax.set_ylabel("Validation score")
    ax.set_title("Validation Threshold Tradeoff")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "checkpoint2_threshold_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importance: pd.DataFrame) -> None:
    top = importance.sort_values("importance", ascending=False).head(12).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(top["feature"], top["importance"], color="#4C78A8")
    ax.set_xlabel("Random forest impurity importance")
    ax.set_ylabel("")
    ax.set_title("Preliminary Predictive Feature Importance")
    for index, value in enumerate(top["importance"]):
        ax.text(float(value) + 0.003, index, f"{float(value):.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "checkpoint2_feature_importance.png", bbox_inches="tight")
    plt.close(fig)


def write_readme(best_model: pd.Series) -> None:
    text = f"""# Predicting and Explaining GitHub Pull Request Merge Outcomes

## Checkpoint 2 Scope

This milestone asks whether PR-level features can help explain and predict GitHub pull request merge outcomes. The locked target is `merged_or_not`, with `0` treated as the minority not-merged class of interest. The dataset is strongly imbalanced, so validation not-merged F1, recall, balanced accuracy, and average precision are more informative than raw accuracy.

Checkpoint 2 covers the refined research question, target distribution, data characteristics, leakage-safer feature policy, preprocessing pipeline, supervised baselines, stronger supervised models, internal-validation model comparison, preliminary interpretation, and remaining work for the final submission.

Validation metrics are the Checkpoint 2 headline. The current validation leader is `{best_model["model"]}` with validation not-merged F1 `{best_model["f1_not_merged"]:.3f}` and balanced accuracy `{best_model["balanced_accuracy"]:.3f}`. These values come from full-source internal validation artifacts already generated in `deliverables/final/`; the official test split is reserved for final evaluation and is not used for Checkpoint 2 model or threshold selection.

## Reading Path

1. Open the executed notebook: `checkpoint2_pr_merge_modeling.ipynb`.
2. Inspect `checkpoint2_model_comparison.csv` and `figures/checkpoint2_model_comparison_f1.png`.
3. Read `checkpoint2_feature_availability_contract.csv` for the leakage/timing contract.
4. Review `checkpoint2_target_distribution.csv` and `checkpoint2_data_characteristics_summary.csv`.
5. Optionally inspect the validation confusion matrix, threshold tradeoff, feature importance, and cluster profile support files.

## Execution Notes

Use `.venv/bin/python`, not system `python3`, because the project-local environment contains the required packages. `requirements-local.txt` is the canonical environment file. `requirements-local 2.txt` is a duplicate-looking non-canonical helper file and is not used by this milestone package.

The Checkpoint 2 outputs are regenerated from existing final CSV artifacts, not retrained from raw PRFeatures data. This keeps the milestone lightweight and avoids fabricating results. Raw PRFeatures CSVs are Git LFS files; they are only required if the full final notebook is rerun from source data.

To regenerate this package:

```bash
.venv/bin/python scripts/build_checkpoint2_outputs.py
.venv/bin/python scripts/build_analysis_notebooks.py --only checkpoint2
.venv/bin/python -m jupyter nbconvert --execute --to notebook --inplace deliverables/checkpoint-2/checkpoint2_pr_merge_modeling.ipynb
.venv/bin/python scripts/validate_checkpoint2_outputs.py
```

## Methodological Warning

The claim is bounded: early PR-level features show moderate predictive association with merge outcomes. This is not causal evidence and not deployment-ready automatic decision-making. The feature set is leakage-safer under stated assumptions, not guaranteed safe. T2 review-process features are kept separate from the early headline model because they describe information available only after review starts.
"""
    (CHECKPOINT2_DIR / "README.md").write_text(text)


def write_milestone_summary(best_model: pd.Series) -> None:
    text = f"""# Checkpoint 2 Milestone Summary

## 5-Minute Storyline

The project studies whether GitHub PR-level features can help explain and predict merge outcomes. Since Checkpoint 1, the target is locked to `merged_or_not`, the imbalance problem is explicit, the feature policy is leakage-safer, and the first supervised modeling pipeline is reproducible. The current internal-validation leader is `{best_model["model"]}` by not-merged F1, but the result is moderate and should be presented as predictive association only.

## Suggested 6-Slide Order

1. Objective and dataset: PRFeatures train/test data and the research question.
2. Target imbalance: about 89% merged versus 11% not merged; accuracy is misleading.
3. Feature availability contract: headline leakage-safer features, excluded leakage/identifier fields, sensitivity-only fields, and separate T2 review-process fields.
4. Preprocessing and model setup: `ColumnTransformer`, `Pipeline`, imputation, one-hot language, balanced class weights/sample weights.
5. Validation model comparison: dummy majority, balanced logistic regression, balanced decision tree, balanced random forest, weighted HGB.
6. Preliminary interpretation and next steps: feature importance as predictive association, threshold tradeoff, official test reserved for final reporting.

## Defense Notes

- Not causal: feature importance and model metrics do not prove why a PR is merged.
- Moderate performance: the model finds signal but is not a high-confidence automatic decision system.
- Leakage risk handled explicitly: use the phrase leakage-safer under stated assumptions.
- T2/review-process features stay separate: stronger later-stage signal is not an early prediction model.
- Official test not used for tuning: Checkpoint 2 model/threshold choices come from internal validation.
"""
    (CHECKPOINT2_DIR / "checkpoint2_milestone_summary.md").write_text(text)


def write_next_steps() -> None:
    text = """# Checkpoint 2 Next Steps

- Report official test metrics only in the final submission, after model and threshold choices are locked.
- Add calibration diagnostics so predicted scores are not overread as literal probabilities.
- Run stress tests for temporal, project-group, and creator-group generalization.
- Keep feature timing sensitivity visible, especially for `prior_review_num` and later review-process fields.
- Polish the final report and slides around the bounded claim: moderate predictive association, not causality or deployment-ready automation.
"""
    (CHECKPOINT2_DIR / "checkpoint2_next_steps.md").write_text(text)


def main() -> None:
    CHECKPOINT2_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    model_comparison = build_model_comparison()
    target = build_target_distribution()
    data_characteristics = build_data_characteristics(model_comparison, target)
    feature_contract = build_feature_contract()
    preprocessing = build_preprocessing_summary()
    algorithm_rationale = build_algorithm_rationale()
    validation_confusion = build_validation_confusion(model_comparison)
    threshold_tuning = build_threshold_tuning()
    feature_importance = build_feature_importance()

    write_csv(model_comparison, "checkpoint2_model_comparison.csv")
    write_csv(target, "checkpoint2_target_distribution.csv")
    write_csv(data_characteristics, "checkpoint2_data_characteristics_summary.csv")
    write_csv(feature_contract, "checkpoint2_feature_availability_contract.csv")
    write_csv(preprocessing, "checkpoint2_preprocessing_summary.csv")
    write_csv(algorithm_rationale, "checkpoint2_algorithm_rationale.csv")
    write_csv(validation_confusion, "checkpoint2_validation_confusion_matrix.csv")
    write_csv(threshold_tuning, "checkpoint2_threshold_tuning.csv")
    write_csv(feature_importance, "checkpoint2_feature_importance.csv")

    cluster_profile = read_final_csv("cluster_profile.csv")
    cluster_interpretation = read_final_csv("cluster_interpretation.csv")
    write_csv(cluster_profile, "checkpoint2_cluster_profile.csv")
    write_csv(cluster_interpretation, "checkpoint2_cluster_interpretation.csv")

    plot_target_distribution(target)
    plot_model_metric(
        model_comparison,
        "f1_not_merged",
        "checkpoint2_model_comparison_f1.png",
        "Internal Validation Not-Merged F1",
    )
    plot_model_metric(
        model_comparison,
        "balanced_accuracy",
        "checkpoint2_model_comparison_balanced_accuracy.png",
        "Internal Validation Balanced Accuracy",
    )
    plot_feature_policy_summary(feature_contract)
    plot_validation_confusion(validation_confusion)
    plot_threshold_tradeoff(threshold_tuning)
    plot_feature_importance(feature_importance)

    best_model = model_comparison.iloc[0]
    write_readme(best_model)
    write_milestone_summary(best_model)
    write_next_steps()

    print(f"Wrote Checkpoint 2 package artifacts to {CHECKPOINT2_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
