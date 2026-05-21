# Final Self-Review

## Strengths

- The research question is clear and stable: PR-level features are used to explain and predict `merged_or_not`.
- The dataset choice is strong for the assignment because the PRFeatures train/test splits are large, schema-consistent, and centered on PR-level behavior.
- The target audit is explicit: both splits are about `89%` merged and `11%` not merged, so the project correctly avoids accuracy-only evaluation.
- The EDA story is now concrete rather than generic: `eda_key_findings.csv` records class imbalance, first-PR/core-member differences, contributor-history differences, project concentration, CI context, and language-code differences with numeric evidence.
- Feature selection is now stricter. The headline supervised model uses `headline_leakage_safer_features`: 25 PR-level features under a stated prediction-time contract, with identifiers, clear post-outcome fields, timing-sensitive review/process fields, and `prior_review_num` held out of the headline claim.
- The project now compares explicit prediction-time contracts: T0 PR-creation (`0.311` test F1), T1 submitted-diff (`0.314` test F1), and T2 review-process (`0.436` test F1). This directly answers when stronger signal becomes available instead of forcing one flat model story.
- The separate comments dataset is profiled but not force-joined. `comment_dataset_profile.csv` reports `588,097` parsed comment rows and explains that PRFeatures exposes numeric ids while the comments file exposes owner/repo/pull-number keys.
- The `prior_review_num` correction is transparent. It is reported only as an `integrator_assumed_features` sensitivity check because it represents a stronger reviewer/integrator-context availability assumption than the headline early-information model.
- Model selection uses full-source internal validation with `784,412` training rows and `261,471` validation rows, not-merged F1 as the primary criterion, and balanced accuracy / not-merged average precision as secondary checks.
- The final supervised model is retrained on all `1,045,883` training rows before the final holdout test evaluation.
- The notebook includes feature-policy sensitivity analysis comparing 12-feature ultra-conservative, 25-feature headline, 26-feature integrator-assumed, and 31-feature extended timing-assumed policies.
- Repeated 3x2 stratified validation on a `120,000` row sample checks that model ranking is not only a one-split artifact. This is correctly labeled as diagnostic.
- Threshold tuning is validation-only and reported separately from the default-threshold result.
- `threshold_stability.csv` and `stress_model_comparison.csv` add grading-facing evidence that the tuned threshold and selected model are not only one-table claims.
- `calibration_model_comparison.csv` shows that sigmoid/isotonic calibration sharply improves Brier score and weighted calibration error, while the report keeps classification F1 and calibrated probability quality separate.
- `full_generalization_benchmarks.csv` adds larger temporal, project-group, and creator-group benchmarks beyond the original sampled stress tests.
- `paired_model_delta_intervals.csv` adds paired bootstrap evidence that the Random Forest validation lead over histogram gradient boosting is not just a leaderboard artifact.
- Bootstrap confidence intervals make the final metrics less brittle and discourage over-claiming small differences.
- The final notebook includes a professor-facing final answer, compact feature-timing tables, and an assignment-coverage map so grading evidence is visible without hunting through outputs.
- The final report and slide deck now emphasize readable exhibits and permutation importance rather than wide raw metric tables.
- `professor_defense_notes.md` now gives a compact Q&A script for the exact professor concerns: leakage, forced results, math validity, generalization, calibration, and clustering.
- The unsupervised analysis labels clusters, reports PCA explained variance, and frames clustering as profile interpretation rather than prediction.

## Weaknesses

- The stricter headline Random Forest has useful signal, but performance is moderate. Default-threshold not-merged precision is only `0.225`.
- The tuned threshold improves not-merged F1 from `0.314` to `0.326`, but lowers not-merged recall from `0.515` to `0.364`.
- Removing `prior_review_num` reduces headline test F1 from the old roughly `0.373` integrator-assumed result to `0.314`. This should be presented as an honest leakage-hardening tradeoff, not as a failure.
- Histogram gradient boosting is close in sampled repeated validation and sometimes has stronger average precision, so the final model choice should be explained through the pre-declared not-merged F1 rule and the model-family stress comparison.
- The ultra-conservative policy has lower test not-merged F1 (`0.273`), while the extended timing-assumed feature policy gives only a small improvement over the stricter headline result (`0.315` vs `0.314`).
- The T2 review-process score is stronger, but it must be defended as a later-stage explanation model, not as an early PR triage model.
- The calibration models improve probability honesty but default 0.5 classification F1 becomes low because calibrated probabilities are conservative under class imbalance; this needs careful presentation.
- The comments dataset cannot be joined locally to PRFeatures without an external mapping, so the project uses PRFeatures review/process aggregate columns for T2 instead of raw comment aggregation.
- K-means selected `k = 2` with one dominant cluster, and the first two PCA components explain only `28.7%` of transformed variance. The clustering section is interpretive, not strong evidence of naturally separated groups.

## Risks

- Retained historical/project snapshot features remain valid only under the stated prediction-time contract.
- If source documentation shows that any retained headline leakage-safer feature includes post-closure information, that feature should move to the held-back group.
- `prior_review_num` should remain sensitivity evidence unless the presentation can defend the stronger reviewer/integrator-context availability assumption.
- The extended timing-assumed fields (`pr_succ_rate`, `requester_succ_rate`, `num_commits`, `src_churn`, `files_changed`, and `test_churn`) should remain sensitivity evidence, not headline evidence.
- The model may still be sensitive to original PRFeatures preprocessing choices inherited from the dataset source.
- T2 review-process fields may include information very close to the human decision process, so they should be presented as timing contrast evidence, not as deployable predictors.
- Feature importance from Random Forest is useful for explanation, but it does not prove causal influence.
- The final test metrics should be presented honestly: this is useful for signal discovery and discussion, not reliable enough for automated merge decisions.

## Grading-Focused Critique

- **Data understanding:** strong. The final notebook documents source, scope, shape, target distribution, nulls, duplicates, and train/test consistency.
- **Preprocessing:** strong. Feature groups are explicit, `language` is categorical, and pipelines separate numeric, binary, and categorical handling.
- **EDA:** strong. It now connects target imbalance, PR size, contributor context, project/creator concentration, language-code differences, and correlation patterns directly to the research question while keeping EDA plots labeled as diagnostics.
- **Supervised ML:** very strong for course scope. It compares multiple model families, includes a dummy baseline, uses imbalance-aware metrics, selects on full-source validation, retrains on all `1,045,883` train rows, adds prediction-time contracts, sampled repeated-validation/stress diagnostics, larger holdout benchmarks, and keeps the test split out of selection.
- **Thresholding:** strong. The tuned threshold is selected only on validation predictions and is reported separately from the default threshold.
- **Sensitivity analysis:** very strong. Ultra-conservative, headline leakage-safer, integrator-assumed, extended timing-assumed, prediction-time contract, calibration, paired-delta, and sample-size diagnostics make the conclusions harder to dismiss as forced or cherry-picked.
- **Unsupervised ML:** acceptable and honestly bounded. K-means/PCA is presented as profile discovery rather than prediction.
- **Interpretation:** strong if presented carefully. The language avoids causal claims and highlights leakage, imbalance, temporal ambiguity, and generalizability limits.

## Final Results to Report

- Selected model: **Random forest balanced**
- Headline feature policy: `headline_leakage_safer_features`, 25 features
- Model selection scope: full-source internal validation, `784,412` training rows and `261,471` validation rows
- Final training scope: retrained on all `1,045,883` train rows
- Default test not-merged precision/recall/F1: `0.225` / `0.515` / `0.314`
- Tuned-threshold test not-merged precision/recall/F1: `0.296` / `0.364` / `0.326`
- Prediction-time contract test F1: T0 creation `0.311`; T1 submitted diff `0.314`; T2 review process `0.436`
- T2 review-process test precision/recall/F1: `0.315` / `0.708` / `0.436`
- Best calibration Brier score: about `0.086` on the held-out calibration diagnostic
- Paired validation delta vs histogram gradient boosting: `+0.035` not-merged F1, 95% CI `+0.033` to `+0.037`
- Larger holdout F1: temporal `0.246`; project `0.312`; creator `0.312`
- Tuned threshold: `0.578`, selected on validation data only
- Tuned-threshold 95% CI for not-merged F1: `0.321` to `0.331`
- Full-source validation random forest not-merged F1: `0.372`
- Sampled repeated-validation random forest not-merged F1 diagnostic: `0.340 ± 0.006`
- Threshold stability diagnostic: see `threshold_stability.csv` after regeneration for fold-level tuned thresholds and tuned not-merged F1.
- Model-family stress comparison: see `stress_model_comparison.csv` after regeneration for random, temporal, project-group, and creator-group split comparisons.
- Ultra-conservative test not-merged F1: `0.273`
- Integrator-assumed test not-merged F1: `0.373`, showing the old stronger result depends heavily on `prior_review_num`
- Extended timing-assumed test not-merged F1: `0.315`, compared with `0.314` for the headline default-threshold model
- Sample-size sensitivity diagnostic improves from `0.344` F1 at `100k` rows to `0.365` F1 at `500k` rows on validation; full-source validation reaches `0.372`.
- K-means selected `k = 2`; the smaller cluster has a higher not-merged rate (`20.55%`) than the main cluster (`10.43%`).
