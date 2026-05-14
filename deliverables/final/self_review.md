# Final Self-Review

## Strengths

- The research question is clear and stable: PR-level features are used to explain and predict `merged_or_not`.
- The dataset choice is strong for the assignment because the PRFeatures train/test splits are large, schema-consistent, and already centered on PR-level behavior.
- The target audit is explicit: both splits are about `89%` merged and `11%` not merged, so the project correctly avoids accuracy-only evaluation.
- Feature selection is leakage-aware. Identifiers and clear post-outcome fields are excluded, and timing-sensitive comment, sentiment, interaction, and CI-progression variables are held back.
- The final notebook now records a feature-level prediction-time rationale for retained safe features, including historical success rates, project workload snapshots, and initial PR-diff measures.
- The final notebook covers the expected ADES tracks: preprocessing, EDA, supervised classification, model comparison, unsupervised clustering, final evaluation, and limitations.
- The supervised comparison includes a dummy baseline, linear model, tree model, random forest, and gradient boosting alternative.
- Final evaluation uses the provided test split only after internal validation model selection.

## Weaknesses

- The final Random Forest improves meaningfully over the dummy baseline, but not-merged precision remains low at `0.285`; many merged PRs are incorrectly flagged as not merged.
- Not-merged recall is moderate at `0.497`, so the model still misses about half of not-merged PRs.
- XGBoost could not run locally because the macOS OpenMP runtime `libomp.dylib` was unavailable. The notebook handles this as an optional skipped baseline and uses scikit-learn histogram gradient boosting instead.
- The final model is trained on a stratified sample for runtime, not the full million-row training split.
- Cluster interpretation is useful but simple: K-means selected `k = 2`, producing one dominant cluster and one much smaller high-star/project-profile cluster.

## Risks

- Any future inclusion of comment, CI result, sentiment, or closure-derived variables could inflate performance if the timing is not justified.
- Retained historical/project snapshot features such as `open_pr_num`, `pr_succ_rate`, `requester_succ_rate`, and `test_churn` remain valid only under the stated prediction-time contract. If the source documentation shows they include post-closure information, they should be moved out of the modeling set.
- The model may still be sensitive to the original PRFeatures preprocessing choices, which are inherited from the dataset source.
- Feature importance from Random Forest is useful for explanation, but it does not prove causal influence.
- The final test metrics should be presented honestly: the model is useful for signal discovery, not reliable enough for automated merge decisions.
- Generalization is limited to the sampled GitHub PRFeatures population.

## Grading-Focused Critique

- **Data understanding:** strong. The final notebook documents dataset source, scope, shape, target distribution, nulls, duplicates, and train/test consistency.
- **Preprocessing:** strong. Feature groups are explicit, `language` is categorical, and model pipelines separate numeric, binary, and categorical handling.
- **EDA:** solid. It connects target imbalance, PR size, contributor context, and correlation patterns directly to the research question.
- **Supervised ML:** strong for course scope. The validation table shows why the majority baseline is inadequate and compares multiple model families using imbalance-aware metrics.
- **Unsupervised ML:** acceptable and clearly bounded. K-means/PCA is presented as profile discovery rather than prediction.
- **Evaluation:** strong. Final test metrics include accuracy, balanced accuracy, minority-class precision/recall/F1, ROC-AUC, average precision, and confusion matrix.
- **Interpretation:** strong if presented carefully. The language avoids causal claims and highlights leakage, imbalance, temporal ambiguity, and generalizability limits.

## Final Results to Report

- Selected model: **Random forest balanced**
- Final test accuracy: `0.811`
- Final test balanced accuracy: `0.673`
- Final test not-merged precision: `0.285`
- Final test not-merged recall: `0.497`
- Final test not-merged F1: `0.362`
- Final test ROC-AUC: `0.736`
- Top feature signals: `prior_review_num`, `contrib_perc_commit`, `open_pr_num`, `pr_succ_rate`, `sloc`, `fork_num`, `stars`, `test_lines_per_kloc`
- K-means selected `k = 2`; the smaller cluster has a higher not-merged rate (`18.17%`) than the main cluster (`10.44%`).
