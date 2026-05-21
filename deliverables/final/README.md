# Final Package Manifest

This folder contains the final human-facing outputs and generated metrics for the ADES PR merge-outcome analysis.

## Source Data

- Zenodo PRFeatures dataset: <https://zenodo.org/records/10049493>
- Zhang et al. paper: <https://yuyue.github.io/res/paper/PRDecision-TSE2022.pdf>
- Field-timing README: <https://www.gitlink.org.cn/Raining/new_pullreq_dataset>

## Headline Result

- Submitted supervised model: `Random forest balanced`
- Feature policy: `headline_leakage_safer_features`
- Feature count: `25`
- Model selection scope: full-source internal validation with `784,412` training rows and `261,471` validation rows
- Final training scope: retrained on all `1,045,883` train rows
- Final holdout test file: `prfeatures_test_data.csv`
- Default test not-merged precision / recall / F1: `0.225` / `0.515` / `0.314`
- Validation-tuned test not-merged precision / recall / F1: `0.296` / `0.364` / `0.326`
- Prediction-time contract test F1: T0 creation `0.311`, T1 submitted diff `0.314`, T2 review process `0.436`
- Best calibration diagnostic: sigmoid/isotonic reduce Brier score to about `0.086` on a held-out validation slice
- Larger holdout diagnostics: temporal `0.246` F1, project `0.312` F1, creator `0.312` F1
- Ultra-conservative default test not-merged F1: `0.273`
- Integrator-assumed default test not-merged F1: `0.373`
- Extended timing-assumed default test not-merged F1: `0.315`

The headline model excludes `prior_review_num` because using reviewer/integrator-context information requires a stronger availability assumption than the early/submission-time contract. The upgraded final story separates T0 PR-creation, T1 submitted-diff, and T2 review-process contracts. The separate comments dataset is profiled but not force-joined because PRFeatures exposes numeric ids while the comments file exposes owner/repo/pull-number keys. Sampled repeated CV, threshold-stability checks, model-family stress tests, ultra-conservative feature-policy sensitivity, integrator-assumed sensitivity, timing-assumed sensitivity, sample-size sensitivity, calibration diagnostics, paired model deltas, EDA plotting, and clustering are diagnostics only. They are not the submitted early supervised model's final training scope.

## Main Files

- `final_pr_merge_analysis.ipynb`: analysis notebook that produces the final metrics, figures, and interpretation.
- `final_report.md` and `final_report.pdf`: standalone report organized around the assignment criteria.
- `professor_defense_notes.md`: short Q&A rehearsal notes for leakage, forced-result, math, generalization, calibration, and clustering questions.
- `final_test_metrics.csv`: final holdout metrics for the full-train headline model at default and validation-tuned thresholds.
- `prediction_contract_comparison.csv`: T0/T1/T2 validation and test comparison showing how signal changes with feature availability.
- `prediction_contract_feature_map.csv`: feature-by-contract availability map for grading and defense.
- `review_process_feature_audit.csv`: review/process fields and why they are excluded from early contracts.
- `comment_dataset_profile.csv`: parsed comment-dataset profile and explicit no-join rationale.
- `calibration_model_comparison.csv`: uncalibrated, sigmoid, and isotonic calibration comparison on a held-out slice.
- `full_generalization_benchmarks.csv`: official, temporal, project-group, and creator-group large benchmark comparison.
- `paired_model_delta_intervals.csv`: paired bootstrap deltas for Random Forest versus histogram gradient boosting.
- `model_comparison.csv`: full-source validation comparison across candidate supervised models.
- `target_distribution_summary.csv`: train/test target counts used by the corrected target-distribution figure.
- `eda_key_findings.csv`: compact professor-facing EDA findings with numeric evidence and interpretation.
- `project_creator_concentration.csv`: concentration of PR rows in the largest projects and creators.
- `final_metric_confidence_intervals.csv`: bootstrap confidence intervals for final test metrics.
- `project_cluster_metric_confidence_intervals.csv`: project-cluster bootstrap intervals that account for repository-level correlation.
- `test_prediction_risk_bands.csv`: official-test deciles showing whether predicted not-merged risk concentrates actual not-merged PRs.
- `calibration_summary.csv`: binned predicted-vs-observed not-merged rates and Brier score evidence.
- `error_profile_summary.csv` and `error_analysis_key_findings.csv`: tuned-threshold false-positive, false-negative, and correct-case profiles.
- `leakage_sensitivity.csv`: ultra-conservative 12-feature, headline leakage-safer 25-feature, integrator-assumed 26-feature, and extended timing-assumed 31-feature policies compared on validation and test.
- `feature_timing_evidence.csv`, `feature_timing_summary.csv`, and `feature_assumption_flags.csv`: professor-facing feature timing assumptions and source-grounded feature roles.
- `data_quality_summary.csv`, `raw_data_fingerprint.csv`, `split_id_integrity.csv`, and `split_overlap_summary.csv`: row counts, headers, checksums, quality checks, PR-id integrity, and official split overlap diagnostics.
- `validation_test_gap.csv`, `generalization_stress_tests.csv`, and `stress_model_comparison.csv`: internal-validation-to-test drop, sampled random/temporal/project/creator stress tests, and model-family stress comparison.
- `threshold_stability.csv`: repeated sampled threshold-tuning diagnostic so the tuned threshold is not a one-split-only story.
- `permutation_importance.csv` and `feature_family_ablation.csv`: model explanation diagnostics beyond Random Forest impurity importance.
- `assignment_coverage.csv`: direct mapping from assignment requirements to notebook/report/slide/artifact evidence.
- `sample_size_sensitivity.csv`: diagnostic validation results at `100k`, `260k`, and `500k` stratified samples.
- `training_scope_summary.csv`: compact comparison of sample diagnostics, full-source validation, and final full-test scope.
- `feature_importance.csv`: Random Forest feature importances for the headline leakage-safer policy.
- `cluster_profile.csv`: diagnostic cluster profiles and post-fit merge-rate composition.
- `cross_validation_summary.csv` and `cross_validation_fold_scores.csv`: sampled repeated-CV diagnostics.
- `threshold_tuning_top.csv` and `threshold_tuning_curve_sample.csv`: compact validation-only threshold outputs; `threshold_tuning.csv` is kept as a small compatibility copy of the top rows.
- `final_confusion_matrix.csv`: confusion matrix counts for final test predictions.
- `self_review.md`: strengths, weaknesses, risks, and final-report notes.
- `figures/`: exported figures for EDA diagnostics, model comparison, prediction contracts, calibration comparison, larger generalization benchmarks, permutation importance, feature importance, threshold tuning, final confusion matrix, and clustering.

## Inspect

From the project root:

```bash
cd "/Users/mahmoudali/Documents/ADES - first project"
ls deliverables/final
sed -n '1,20p' deliverables/final/final_test_metrics.csv
sed -n '1,20p' deliverables/final/model_comparison.csv
sed -n '1,20p' deliverables/final/eda_key_findings.csv
sed -n '1,20p' deliverables/final/prediction_contract_comparison.csv
sed -n '1,20p' deliverables/final/calibration_model_comparison.csv
sed -n '1,20p' deliverables/final/full_generalization_benchmarks.csv
```

For a quick visual check, open the exported PNGs in `deliverables/final/figures/`.

## Re-run

The notebook is generated from `scripts/build_analysis_notebooks.py`; the final report is generated from `scripts/build_final_report.py`; output validation is in `scripts/validate_final_outputs.py`.

```bash
.venv/bin/python scripts/build_analysis_notebooks.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  deliverables/final/final_pr_merge_analysis.ipynb \
  --ExecutePreprocessor.timeout=-1
.venv/bin/python scripts/build_final_report.py
.venv/bin/python scripts/validate_final_outputs.py
npx -y @marp-team/marp-cli slides/final_presentation.md \
  -o slides/final_presentation.pdf --allow-local-files
```

After re-running, use `final_report.pdf`, `eda_key_findings.csv`, `prediction_contract_comparison.csv`, `calibration_model_comparison.csv`, `full_generalization_benchmarks.csv`, `paired_model_delta_intervals.csv`, `training_scope_summary.csv`, `final_test_metrics.csv`, `test_prediction_risk_bands.csv`, `calibration_summary.csv`, `error_profile_summary.csv`, `leakage_sensitivity.csv`, `generalization_stress_tests.csv`, `stress_model_comparison.csv`, `threshold_stability.csv`, and both confidence-interval CSVs as the source of truth for presentation and review language.
