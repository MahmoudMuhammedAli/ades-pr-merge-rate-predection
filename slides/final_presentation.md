# Can PR-Level Features Explain and Predict GitHub PR Merge Outcomes?

FEUP ADES final presentation source  
Research question: Can PR-level features help explain and predict PR merge outcomes on GitHub?

---

## 1. Problem

- GitHub pull requests are not merged at equal rates.
- The project studies whether PR-level metadata, contributor context, project context, and PR-size features are associated with merge outcomes.
- Framing is predictive and explanatory, not causal.

Target:

- `merged_or_not`
- `1`: merged
- `0`: not merged

---

## 2. Dataset and Scope

Source: Zenodo PRFeatures dataset from "GitHub Pull Request Analysis: Sentiment Data and Developer Survey Responses".

Used:

- `prfeatures_train_data.csv`: main analysis and model-development split
- `prfeatures_test_data.csv`: final holdout evaluation split

Not used as core inputs:

- `pr_comments_dataset_publish.csv`: comment-level data, kept out because timing can create leakage risk
- `survey_responses_raw.csv`: only 22 responses, not suitable as the main dataset

---

## 3. Dataset Audit

- Train split: `1,045,883` rows, `72` columns
- Test split: `260,195` rows, `72` columns
- Train and test contain the same columns in the same order
- `merged_or_not` is present in both files
- Final notebook audit found `0` explicit nulls and `0` duplicate rows in both splits

Implication:

- The data is large enough for modeling, but leakage and imbalance matter more than raw row count.

---

## 4. Target Imbalance

| Split | Merged | Not merged |
|---|---:|---:|
| Train | `932,538` (`89.16%`) | `113,345` (`10.84%`) |
| Test | `232,073` (`89.19%`) | `28,122` (`10.81%`) |

Why this matters:

- A majority-class model reaches about `89%` accuracy while detecting no not-merged PRs.
- Evaluation must include minority-class recall/F1, balanced accuracy, average precision, ROC-AUC, and the confusion matrix.

---

## 5. Leakage-Safe Feature Selection

| Group | Decision | Examples |
|---|---|---|
| Identifiers | Exclude | `id`, `project_id`, `creator_id`, `last_closer_id` |
| Post-outcome fields | Exclude | `last_close_time`, `lifetime_minutes`, `reopen_or_not` |
| Timing-sensitive fields | Hold back | comments, sentiment, CI outcomes, interaction fields |
| Lower-risk PR-level fields | Use | contributor history, project context, PR size |

Principle:

- The model should only use information plausibly available before the final merge decision.
- Retained rate/workload fields are interpreted as historical or PR-submission snapshots. If a field cannot be defended under that timing contract, it should move to the timing-sensitive group.

---

## 6. Modeling Feature Set

Conservative feature families:

- Contributor context: `first_pr`, `core_member`, `prior_review_num`, `prior_interaction`, `followers`, `prev_pullreqs`
- Project context: `team_size`, `language`, `open_issue_num`, `project_age`, `open_pr_num`, `fork_num`, `stars`
- Historical rates: `pr_succ_rate`, `requester_succ_rate`, `perc_external_contribs`
- PR size/scope: `churn_addition`, `churn_deletion`, `src_churn`, `files_changed`, `num_commits`
- Testing/CI availability: `test_inclusion`, `ci_exists`, test-density and churn fields

`language` is treated as categorical, not ordinal.

Prediction-time contract:

- Historical features such as `pr_succ_rate` and `requester_succ_rate` are used as past success context, not as current-PR outcomes.
- Workload fields such as `open_pr_num` are used as project snapshots at or before PR submission.
- Diff fields such as `files_changed`, `src_churn`, and `test_churn` are used as initial submitted PR content, not as closure information.

---

## 7. EDA Finding: PR Size

| Feature | Not merged mean | Merged mean |
|---|---:|---:|
| `files_changed` | `15.44` | `10.91` |
| `churn_addition` | `618.55` | `412.57` |
| `description_length` | `59.92` | `45.95` |
| `num_commits` | `4.70` | `3.98` |

Interpretation:

- Non-merged PRs are larger on average across several size/scope indicators.
- This is association, not proof that large PRs cause rejection.

---

## 8. EDA Finding: Contributor Context

| Feature/value | Merge rate |
|---|---:|
| Repeat contributor PRs (`first_pr = 0`) | `89.45%` |
| First-time contributor PRs (`first_pr = 1`) | `80.68%` |
| Core-member PRs (`core_member = 1`) | `90.79%` |
| Non-core-member PRs (`core_member = 0`) | `82.56%` |

Interpretation:

- Contributor history and project role are strongly associated with merge outcomes.

---

## 9. Supervised Modeling Design

Models compared on an internal validation split from the training file:

- Dummy majority baseline
- Logistic regression with class weights
- Decision tree with class weights
- Random forest with balanced subsampling
- Histogram gradient boosting with balanced sample weights
- XGBoost was included as optional, but skipped locally because `libomp.dylib` was unavailable

Selection criterion:

- Primary: not-merged F1
- Secondary: balanced accuracy and not-merged average precision

---

## 10. Validation Model Comparison

| Model | Accuracy | Balanced accuracy | Not-merged recall | Not-merged F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random forest balanced | `0.817` | `0.704` | `0.560` | `0.399` | `0.784` |
| Hist gradient boosting weighted | `0.775` | `0.711` | `0.629` | `0.377` | `0.783` |
| Decision tree balanced | `0.742` | `0.666` | `0.570` | `0.324` | `0.728` |
| Logistic regression balanced | `0.638` | `0.626` | `0.612` | `0.268` | `0.674` |
| Dummy majority | `0.892` | `0.500` | `0.000` | `0.000` | `0.500` |

Takeaway:

- The dummy baseline has high accuracy but zero not-merged recall/F1.
- Random forest gives the best not-merged F1 on validation.

---

## 11. Repeated Validation and Threshold Tuning

Repeated validation:

- 3-fold, 2-repeat stratified cross-validation on a `120,000` row modeling sample.
- Random forest remains strongest by mean not-merged F1: `0.381 ± 0.005`.
- Histogram gradient boosting is close and has slightly stronger mean balanced accuracy and average precision, but the pre-declared selection rule prioritizes not-merged F1.

Threshold tuning:

- The selected not-merged decision threshold is `0.584`.
- It was chosen on validation predictions only.
- It is applied once to the untouched test split.

---

## 12. Final Test Evaluation

Selected model: **Random forest balanced**

Final holdout results on `prfeatures_test_data.csv`:

| Metric | Default threshold | Tuned threshold |
|---|---:|---:|
| Accuracy | `0.811` | `0.869` |
| Balanced accuracy | `0.673` | `0.649` |
| Not-merged precision | `0.285` | `0.390` |
| Not-merged recall | `0.497` | `0.367` |
| Not-merged F1 | `0.362` | `0.378` |
| ROC-AUC | `0.736` | `0.736` |
| Not-merged average precision | `0.358` | `0.358` |

Interpretation:

- Default threshold finds more not-merged PRs but creates more false alarms.
- Tuned threshold improves not-merged precision and F1, while lowering not-merged recall.
- Both rows must be reported because the tuned threshold is a validation decision, not a new model.

---

## 13. Confidence Intervals

Bootstrap confidence intervals use `500` resamples of final test predictions.

Tuned-threshold 95% intervals:

| Metric | Estimate | 95% CI |
|---|---:|---:|
| Not-merged precision | `0.390` | `0.384` to `0.396` |
| Not-merged recall | `0.367` | `0.362` to `0.373` |
| Not-merged F1 | `0.378` | `0.373` to `0.383` |
| Balanced accuracy | `0.649` | `0.646` to `0.652` |
| ROC-AUC | `0.735` | `0.732` to `0.739` |

Interpretation:

- The model has real predictive signal, but the intervals do not make it operationally reliable.
- The precision/recall tradeoff is the honest story.

---

## 14. Leakage Sensitivity

Strict feature set removed:

- `open_pr_num`
- `pr_succ_rate`
- `requester_succ_rate`
- `test_churn`

| Evaluation | Primary F1 | Strict F1 | Primary balanced accuracy | Strict balanced accuracy |
|---|---:|---:|---:|---:|
| Validation | `0.399` | `0.391` | `0.704` | `0.699` |
| Test | `0.362` | `0.354` | `0.673` | `0.667` |

Takeaway:

- The conclusion survives stricter timing assumptions, but slightly weakens.
- This makes the final claim more credible: PR metadata has signal, but timing-sensitive project snapshots should be defended carefully.

---

## 15. Sample-Size Sensitivity

Selected model trained at three stratified sample sizes:

| Training sample | Validation balanced accuracy | Validation not-merged F1 | Not-merged AP |
|---|---:|---:|---:|
| `100,000` | `0.693` | `0.385` | `0.377` |
| `260,000` | `0.704` | `0.399` | `0.405` |
| `500,000` | `0.715` | `0.413` | `0.427` |

Interpretation:

- More data improves validation performance.
- The submitted model remains sample-based for runtime, but this check shows the direction of improvement is sensible.

---

## 16. Feature Importance

Top Random Forest signals:

1. `prior_review_num`
2. `contrib_perc_commit`
3. `open_pr_num`
4. `pr_succ_rate`
5. `sloc`
6. `fork_num`
7. `stars`
8. `test_lines_per_kloc`
9. `prior_interaction`
10. `prev_pullreqs`

Interpretation:

- The model relies most on contributor history, project/repository context, and project size/activity signals.
- No identifier or clear post-outcome leakage field appears in the final feature set.
- Some high-importance fields are exactly why the strict-feature sensitivity analysis matters.

---

## 17. Unsupervised Analysis

Goal:

- Explore interpretable PR profiles without using the merge target during fitting.

Method:

- Standardize safe features.
- Fit K-means for `k = 2..6`.
- Choose `k = 2` by silhouette score.
- Use PCA only for visualization.
- Profile clusters after fitting using merge-rate composition.
- PCA is visualization only; the first two components explain `25.6%` of transformed variance.

---

## 18. Cluster Findings

| Cluster | Size | Merge rate | Not-merged rate | Profile |
|---|---:|---:|---:|---|
| 0 | `66,400` | `89.56%` | `10.44%` | lower not-merged / larger-change / smaller-project |
| 1 | `3,600` | `81.83%` | `18.17%` | higher not-merged / smaller-change / larger-project |

Discussion:

- Clustering is descriptive, not a separate prediction model.
- The smaller cluster has a noticeably higher not-merged rate, suggesting project/profile context matters.
- Because silhouette chose `k = 2` and one cluster dominates, this is useful interpretation rather than strong evidence of naturally separated groups.

---

## 19. Direct Answer

Research question: Can PR-level features help explain and predict PR merge outcomes on GitHub?

Answer:

- Yes, but the signal is moderate.
- The model beats the majority baseline on imbalance-aware metrics.
- The best explanation is not "we can predict merges perfectly"; it is "contributor history, project context, and PR size/scope contain measurable signal."
- The strict-feature sensitivity check shows the result is not entirely forced by questionable timing fields.
- Findings support association and prediction, not causality.

---

## 20. Threats to Validity

- Leakage risk: some fields may only be known after review or closure.
- Temporal ambiguity: comment, CI, sentiment, and interaction fields may depend on when prediction is made.
- Class imbalance: minority-class performance can be hidden by high accuracy.
- Sampling: models were trained on a stratified sample for runtime, then evaluated on the full test split.
- Generalizability: results apply to this PRFeatures GitHub sample.
- Feature encoding: `language` is categorical and should not be read as an ordered number.

---

## 21. Discussion Prep

Likely questions and short answers:

- Why not use all 72 columns?  
  Because identifiers, post-outcome fields, and timing-sensitive fields risk leakage.

- Why is accuracy not enough?  
  The dummy model reaches `0.892` validation accuracy but detects no not-merged PRs.

- What metric drove selection?  
  Not-merged F1, with balanced accuracy and not-merged average precision as supporting metrics.

- Can we claim large PRs are less likely to merge?  
  We can claim association in this dataset, not causation.

- Are clusters predictive?  
  No. They describe PR profiles and support interpretation.

- Did threshold tuning use the test split?  
  No. The threshold was selected on validation data and then applied once to the test split.

- Does the result disappear under stricter leakage assumptions?  
  No. Test not-merged F1 drops from `0.362` to `0.354`, so the conclusion weakens slightly but survives.
