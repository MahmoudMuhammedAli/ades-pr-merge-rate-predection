# Final Self-Review

## Strengths

- The research question is clear and stable: PR-level features are used to explain and predict `merged_or_not`.
- The dataset choice is strong for the assignment because the PRFeatures train/test splits are large, schema-consistent, and centered on PR-level behavior.
- The target audit is explicit: both splits are about `89%` merged and `11%` not merged, so the project correctly avoids accuracy-only evaluation.
- Feature selection is leakage-aware. Identifiers and clear post-outcome fields are excluded, and timing-sensitive comment, sentiment, interaction, and CI-progression variables are held back.
- The final notebook now includes a strict-feature sensitivity analysis that removes `open_pr_num`, `pr_succ_rate`, `requester_succ_rate`, and `test_churn`.
- Model selection is clearer: validation not-merged F1 is primary, with balanced accuracy and not-merged average precision as secondary checks.
- Repeated 3x2 stratified validation checks that the model ranking is not only a one-split artifact.
- Threshold tuning is validation-only and reported separately from the default-threshold result.
- Bootstrap confidence intervals make the final metrics less brittle and discourage over-claiming small differences.
- The unsupervised analysis now labels clusters, reports PCA explained variance, and frames clustering as profile interpretation rather than prediction.

## Weaknesses

- The final Random Forest has useful signal, but performance is still moderate. Default-threshold not-merged precision is only `0.285`.
- The tuned threshold improves not-merged F1 from `0.362` to `0.378`, but lowers not-merged recall from `0.497` to `0.367`.
- Histogram gradient boosting is close in repeated validation and has stronger average precision, so the final model choice should be explained through the pre-declared F1 rule.
- XGBoost could not run locally because the macOS OpenMP runtime `libomp.dylib` was unavailable. The notebook handles this as optional and uses scikit-learn histogram gradient boosting instead.
- The final model is trained on a stratified sample for runtime, not the full million-row training split.
- K-means selected `k = 2` with one dominant cluster, and the first two PCA components explain only `25.6%` of transformed variance. The clustering section is interpretive, not strong evidence of naturally separated groups.

## Risks

- Retained historical/project snapshot features remain valid only under the stated prediction-time contract.
- If source documentation shows that any retained feature includes post-closure information, that feature should move to the held-back group.
- The model may still be sensitive to original PRFeatures preprocessing choices inherited from the dataset source.
- Feature importance from Random Forest is useful for explanation, but it does not prove causal influence.
- The final test metrics should be presented honestly: this is useful for signal discovery and discussion, not reliable enough for automated merge decisions.

## Grading-Focused Critique

- **Data understanding:** strong. The final notebook documents source, scope, shape, target distribution, nulls, duplicates, and train/test consistency.
- **Preprocessing:** strong. Feature groups are explicit, `language` is categorical, and pipelines separate numeric, binary, and categorical handling.
- **EDA:** solid. It connects target imbalance, PR size, contributor context, and correlation patterns directly to the research question.
- **Supervised ML:** strong for course scope. It compares multiple model families, includes a dummy baseline, uses imbalance-aware metrics, adds repeated validation, and keeps the test split out of selection.
- **Thresholding:** strong. The tuned threshold is selected only on validation predictions and is reported separately from the default threshold.
- **Sensitivity analysis:** strong. Strict-feature and sample-size checks make the conclusions harder to dismiss as forced or cherry-picked.
- **Unsupervised ML:** acceptable and honestly bounded. K-means/PCA is presented as profile discovery rather than prediction.
- **Interpretation:** strong if presented carefully. The language avoids causal claims and highlights leakage, imbalance, temporal ambiguity, and generalizability limits.

## Final Results to Report

- Selected model: **Random forest balanced**
- Default test not-merged precision/recall/F1: `0.285` / `0.497` / `0.362`
- Tuned-threshold test not-merged precision/recall/F1: `0.390` / `0.367` / `0.378`
- Tuned threshold: `0.584`, selected on validation data only
- Tuned-threshold 95% CI for not-merged F1: `0.373` to `0.383`
- Repeated-validation random forest not-merged F1: `0.381 ± 0.005`
- Strict-feature test not-merged F1: `0.354`, compared with `0.362` for the primary default-threshold model
- Sample-size sensitivity improves from `0.385` F1 at `100k` rows to `0.413` F1 at `500k` rows on validation.
- K-means selected `k = 2`; the smaller cluster has a higher not-merged rate (`18.17%`) than the main cluster (`10.44%`).
