# Predicting and Explaining GitHub Pull Request Merge Outcomes

## Checkpoint 2 Scope

This milestone asks whether PR-level features can help explain and predict GitHub pull request merge outcomes. The locked target is `merged_or_not`, with `0` treated as the minority not-merged class of interest. The dataset is strongly imbalanced, so validation not-merged F1, recall, balanced accuracy, and average precision are more informative than raw accuracy.

Checkpoint 2 covers the refined research question, target distribution, data characteristics, leakage-safer feature policy, preprocessing pipeline, supervised baselines, stronger supervised models, internal-validation model comparison, preliminary interpretation, and remaining work for the final submission.

Validation metrics are the Checkpoint 2 headline. The current validation leader is `Random forest balanced` with validation not-merged F1 `0.372` and balanced accuracy `0.702`. These values come from full-source internal validation artifacts already generated in `deliverables/final/`; the official test split is reserved for final evaluation and is not used for Checkpoint 2 model or threshold selection.

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
