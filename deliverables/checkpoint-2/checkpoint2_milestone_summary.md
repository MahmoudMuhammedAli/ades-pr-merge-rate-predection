# Checkpoint 2 Milestone Summary

## 5-Minute Storyline

The project studies whether GitHub PR-level features can help explain and predict merge outcomes. Since Checkpoint 1, the target is locked to `merged_or_not`, the imbalance problem is explicit, the feature policy is leakage-safer, and the first supervised modeling pipeline is reproducible. The current internal-validation leader is `Random forest balanced` by not-merged F1, but the result is moderate and should be presented as predictive association only.

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
