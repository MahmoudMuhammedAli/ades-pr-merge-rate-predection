# Checkpoint 2 Readiness Audit

## 1. Executive Summary

**Overall judgment:** `partial` for the dedicated Checkpoint 2 deliverable, `strong` for the current full repository package.

The repo now contains a final analysis package that is substantially beyond normal Checkpoint 2 expectations: it has a stable research question, a defended feature-availability contract, a reusable preprocessing pipeline, baseline and stronger supervised models, imbalance-aware evaluation, final test-set results, threshold tuning, calibration diagnostics, stress tests, feature-policy sensitivity, interpretation, a final report, slides, and a validator. If the professor evaluates the full current repo, the project is on a **high-grade trajectory**.

The dedicated Checkpoint 2 notebook at `deliverables/checkpoint-2/checkpoint2_pr_merge_modeling.ipynb` is much weaker as a standalone artifact. It has the right structure, but it is unexecuted, has no outputs, and does not carry the final-package evidence into a milestone-ready narrative. If judged only from that notebook and folder, the current state looks **medium / partial**, not excellent.

Blunt assessment: the intellectual work is now mostly strong, but the milestone packaging is not. The main risk is that a professor sees a thin Checkpoint 2 notebook and misses the stronger final-package work. The second risk is methodological: the final result is moderate, and the project must keep defending feature timing, imbalance-aware evaluation, and official split limitations instead of overselling predictive performance.

## 2. Checkpoint 2 Coverage Audit

### 1. Clearly Defined and Defensible Research Question

**Status:** `strong`

**Evidence found in repo:** `README.md`, `docs/dataset-summary.md`, `deliverables/final/final_pr_merge_analysis.ipynb`, `deliverables/final/final_report.md`, and `slides/final_presentation.md` all use the same core question: "Can PR-level features help explain and predict PR merge outcomes on GitHub?"

**Issues / limitations:** The Checkpoint 2 notebook states the question, but does not independently show the full final reasoning, threat model, or current result interpretation.

**What still needs to be done:** Add a short Checkpoint 2-specific framing note or executed notebook section that states the question, current answer, feature-availability contract, and non-causal boundary.

### 2. Finalized or Near-Final Safe Feature Subset

**Status:** `strong` in final package, `partial` in Checkpoint 2 notebook

**Evidence found in repo:** `scripts/build_analysis_notebooks.py` defines `headline_safe_features`, `ultra_conservative_features`, `integrator_assumed_extension_features`, `timing_assumed_extension_features`, and staged prediction contracts. `deliverables/final/feature_timing_evidence.csv`, `feature_timing_summary.csv`, `feature_assumption_flags.csv`, `prediction_contract_feature_map.csv`, and `review_process_feature_audit.csv` document feature timing and leakage decisions.

**Issues / limitations:** The feature policy is defensible but still assumption-based. Project snapshot fields, contributor history fields, and submitted-diff fields depend on the source documentation being interpreted correctly. `prior_review_num` is correctly moved out of the headline model, but this correction needs to be made obvious in any Checkpoint 2 presentation.

**What still needs to be done:** In the Checkpoint 2 notebook, execute and display the feature tables. Add a concise statement that the headline model has 25 leakage-safer features, with `prior_review_num` sensitivity-only.

### 3. Handling Data Characteristics

**Status:** `strong` in final package

**Evidence found in repo:** `deliverables/final/data_quality_summary.csv` reports rows, loaded columns, explicit nulls, duplicates, not-merged rate, project count, and creator count. `target_distribution_summary.csv` confirms about 89% merged and 11% not merged in train/test. `numeric_distribution_summary.csv` flags heavy tails and skew. `split_overlap_summary.csv` reports project and creator overlap. The final notebook treats `language` as categorical and separates numeric, binary, and categorical preprocessing.

**Issues / limitations:** The current repo handles explicit nulls, but semantic missingness and source preprocessing assumptions cannot be fully audited from the provided CSVs. Heavy-tailed features are described, but the model relies mainly on scaling and tree robustness rather than deeper transformations or winsorization studies.

**What still needs to be done:** Add a Checkpoint 2-facing summary table for imbalance, skew/outliers, categorical handling, binary handling, numeric/rate features, missingness, and semantic completeness concerns.

### 4. Reproducible Preprocessing Pipeline

**Status:** `strong`

**Evidence found in repo:** `scripts/build_analysis_notebooks.py` defines `make_preprocessor()` with `ColumnTransformer`: median imputation and scaling for numeric features, most-frequent imputation for binary features, and one-hot encoding with `handle_unknown="ignore"` for `language`. Models are wrapped in `Pipeline`.

**Issues / limitations:** Reproducibility depends on using `.venv/bin/python`, not system `python3`. Running `python3 scripts/validate_final_outputs.py` fails because `pandas` is missing; `.venv/bin/python scripts/validate_final_outputs.py` passes.

**What still needs to be done:** Make the environment requirement prominent in the Checkpoint 2 handoff and checklist. Do not tell reviewers to use plain `python3` unless dependencies are installed there.

### 5. At Least One Supervised Baseline

**Status:** `strong`

**Evidence found in repo:** `model_comparison.csv` includes `Dummy majority`, `Logistic regression balanced`, and `Decision tree balanced`. The final report explicitly explains why the dummy majority baseline is misleading under imbalance: high accuracy but zero not-merged recall/F1.

**Issues / limitations:** The Checkpoint 2 notebook has baseline code, but no executed outputs, so the baseline evidence is not visible in the milestone notebook.

**What still needs to be done:** Execute or regenerate `deliverables/checkpoint-2/checkpoint2_pr_merge_modeling.ipynb` and save the validation table/plot in the Checkpoint 2 folder.

### 6. At Least One Stronger Supervised Model

**Status:** `strong`

**Evidence found in repo:** The final package compares balanced Random Forest and weighted Histogram Gradient Boosting against simpler baselines. `model_comparison.csv` selects `Random forest balanced` by validation not-merged F1. `stress_model_comparison.csv` checks whether the model choice is stable under random, temporal, project-group, and creator-group diagnostics.

**Issues / limitations:** Random Forest wins the declared validation rule, but HGB is competitive and even leads one stricter project-group diagnostic. The project handles this honestly, but the presentation must avoid "Random Forest is universally best."

**What still needs to be done:** In Checkpoint 2 materials, explain the model choice as "best by pre-declared not-merged F1 on internal validation," not as a universal truth.

### 7. Proper Evaluation Using Imbalance-Aware Metrics

**Status:** `strong`

**Evidence found in repo:** Metrics include not-merged precision, recall, F1, balanced accuracy, ROC-AUC, average precision for not-merged, confusion matrices, risk-band lift, threshold tuning, and confidence intervals. `scripts/validate_final_outputs.py` recomputes key metrics from confusion matrices and passes under `.venv/bin/python`.

**Issues / limitations:** The official test result is moderate: default not-merged F1 is 0.314, validation-tuned not-merged F1 is 0.326. That is academically acceptable if presented as moderate signal, but weak if oversold as high predictive accuracy.

**What still needs to be done:** Keep accuracy secondary. In slides and speaking notes, lead with not-merged F1, recall/precision tradeoff, and average precision.

### 8. Clear Comparison Against Baselines

**Status:** `strong`

**Evidence found in repo:** `deliverables/final/final_report.md` has a clear validation comparison table: Dummy majority has 0 not-merged F1; Random Forest has validation not-merged F1 0.372; HGB has 0.337; Logistic Regression has 0.254. The slide deck repeats this comparison.

**Issues / limitations:** Baseline comparison is excellent in final materials, but not visible in the unexecuted Checkpoint 2 notebook.

**What still needs to be done:** Export a Checkpoint 2-specific `model_comparison` table or figure, or explicitly reference the final artifact if the team uses the final package as the source of truth.

### 9. Sensible Interpretation of Results

**Status:** `strong`

**Evidence found in repo:** `deliverables/final/final_report.md`, `self_review.md`, and `professor_defense_notes.md` consistently frame the result as "moderate predictive association," not causality or deployment readiness. The final report explains the T0/T1/T2 prediction-time contract result and why T2 is not an early predictor.

**Issues / limitations:** The interpretation is strong but fragile: if presenters lead with the stronger T2 F1 of 0.436 without explaining timing, it will look like leakage.

**What still needs to be done:** Rehearse the defense: early model moderate, later review-process model stronger because it uses later information, not because it is a better early model.

### 10. Notebook / Repo Structure for Academic Presentation

**Status:** `partial`

**Evidence found in repo:** The repo has a clear structure: `docs/`, `notebooks/`, `data/`, `deliverables/checkpoint-1/`, `deliverables/checkpoint-2/`, `deliverables/final/`, `slides/`, and `scripts/`. The final notebook is executed and output-rich. The final report, slides, figures, CSVs, and validator are well organized.

**Issues / limitations:** The Checkpoint 2 folder itself is thin: one unexecuted notebook, no visible local CSV/PNG outputs, no Checkpoint 2-specific report or speaking notes. The stronger evidence is mostly in `deliverables/final/`.

**What still needs to be done:** Either strengthen `deliverables/checkpoint-2/` or clearly document that the final package supersedes and backfills Checkpoint 2.

### 11. Meaningful Unsupervised Component

**Status:** `partial`

**Evidence found in repo:** The final notebook includes K-means and PCA profile analysis. `cluster_k_selection.csv`, `cluster_profile.csv`, `cluster_interpretation.csv`, and `figures/cluster_pca.png` exist. The final report honestly frames clustering as profile interpretation only.

**Issues / limitations:** The unsupervised component is acceptable but not a core strength. K-means selects `k=2`, one cluster dominates, and PCA explains only part of the transformed variance. It supports the assignment but should not be oversold.

**What still needs to be done:** If used for Checkpoint 2, present it as "profile discovery" and explicitly state that target labels are used only after clustering for interpretation.

### 12. Clear Readiness for Checkpoint 2 Presentation

**Status:** `partial`

**Evidence found in repo:** The final slide deck is strong and includes problem framing, target imbalance, feature contracts, model comparison, threshold tuning, final confusion matrix, prediction contracts, calibration, stress tests, sensitivity, clustering, threats, and final answer.

**Issues / limitations:** The deck is final-presentation oriented, not Checkpoint 2-specific. A Checkpoint 2 presentation should be shorter and should not look like the team skipped directly to final submission without explaining milestone progress.

**What still needs to be done:** Create a Checkpoint 2 speaking path: research question, safe feature subset, preprocessing pipeline, baseline vs stronger model, validation metrics, preliminary test discipline, early interpretation, and next work.

## 3. Grading-Criteria Audit

### Problem Description, Understanding, and Clarity

**Current strength:** `strong`. The problem is clear, scoped to PR-level merge outcomes, and tied to the Zenodo PRFeatures train/test data.

**Risks:** A professor may ask whether "explain and predict" is too broad. The repo mitigates this by separating predictive association from causality, but the presentation must keep that boundary explicit.

**Recommendations:** Lead with the research question and immediately state the claim boundary: "moderate predictive association, not causal, not automated decision-making."

### Exploratory Data Analysis

**Current strength:** `strong`. EDA covers target imbalance, first-PR/core-member differences, contributor history, project concentration, CI presence, language codes, numeric skew, and class-conditioned summaries.

**Risks:** EDA can look like a sidecar if not tied back to modelling decisions.

**Recommendations:** Make every EDA finding justify one modelling choice: imbalance metrics, feature families, group-holdout diagnostics, and leakage-aware feature policy.

### Empirical Data Analysis Study

**Current strength:** `strong` in final package, `partial` in Checkpoint 2 notebook. The final package includes supervised evaluation, final test metrics, prediction-time contracts, calibration, error analysis, stress tests, sensitivity analysis, paired deltas, and clustering.

**Risks:** It may look too final-package heavy and not milestone-specific. The Checkpoint 2 notebook alone does not prove the empirical study because it is unexecuted.

**Recommendations:** Add or execute a Checkpoint 2 milestone artifact that points to the empirical results and clarifies what was known at Checkpoint 2.

### Comprehension of Selected Algorithms

**Current strength:** `strong`. `algorithm_comprehension.csv`, final notebook, final report, and slides describe why each method was used and its limitation.

**Risks:** Random Forest feature importance could be misread as causal explanation.

**Recommendations:** Keep saying feature importance and permutation importance are predictive diagnostics, not causal evidence.

### Handling of Data Characteristics

**Current strength:** `strong`. Imbalance, missingness, duplicates, skew/outliers, categorical handling, binary handling, leakage, split overlap, and project/creator concentration are all addressed.

**Risks:** Semantic completeness remains limited because the team inherits source preprocessing. Some retained features are valid only under the stated timing contract.

**Recommendations:** Do not claim the feature set is perfectly safe. Say it is leakage-safer and source-documented, with sensitivity checks.

### Main Findings and Conclusions

**Current strength:** `strong`. The final answer is honest: PR-level signal exists, but early predictive strength is moderate, stronger metrics require later review-process information, and external validity is limited.

**Risks:** The moderate model performance can be criticized if the team frames the project as building a high-performing classifier.

**Recommendations:** Frame the contribution as empirical understanding plus disciplined evaluation, not maximizing accuracy.

### Notebook / Code Quality

**Current strength:** `partial` overall. The final notebook and generator script are strong. The Checkpoint 2 notebook is not strong because it is unexecuted.

**Risks:** Reviewers often judge notebooks by visible outputs. An unexecuted notebook is a serious presentation risk.

**Recommendations:** Execute the Checkpoint 2 notebook or replace it with an exported executed milestone notebook.

### Organisation, Implementation, and Comments

**Current strength:** `strong` for final package. Scripts, generated artifacts, figures, report, slides, and validator are coherent.

**Risks:** There are many generated CSVs, which can overwhelm a reviewer. `requirements-local 2.txt` is a duplicate-looking dependency file and may confuse setup.

**Recommendations:** Use `deliverables/final/README.md` as the artifact map. For Checkpoint 2, give reviewers one short path through the evidence.

### Presentation Readiness

**Current strength:** `partial` for Checkpoint 2, `strong` for final. The final slide deck is strong but broader than a Checkpoint 2 presentation.

**Risks:** A professor may see the work as either too unfinished if only the Checkpoint 2 notebook is opened, or too overbuilt/final if the final deck is used without milestone framing.

**Recommendations:** Build a Checkpoint 2 presentation script from the final deck, focusing on modelling readiness and first empirical results.

## 4. Technical Audit

### Data Preparation

**Status:** `strong`

The project uses the PRFeatures train/test split and documents raw file fingerprints, row counts, schema consistency, target distribution, duplicate checks, explicit null checks, PR-id integrity, and project/creator overlap. This is well above a minimal Checkpoint 2 standard.

Weakness: source-level preprocessing is inherited from the dataset. The repo checks explicit nulls but cannot fully audit whether source-imputed zeros or negative placeholders encode semantic missingness.

### Preprocessing

**Status:** `strong`

The preprocessing design is correct for the feature mix: numeric median imputation and scaling, binary most-frequent imputation, categorical one-hot encoding for `language`, and `handle_unknown="ignore"` for future test categories. The preprocessing is inside sklearn `Pipeline`, which reduces train/test leakage risk.

Weakness: heavy-tailed numeric features are scaled but not transformed. Tree models are robust enough for a course project, but logistic regression may be affected.

### Modelling

**Status:** `strong`

The repo includes a dummy baseline, balanced logistic regression, balanced decision tree, balanced random forest, and weighted histogram gradient boosting. It retrains the selected model on all training rows before final test evaluation.

Weakness: no hyperparameter search is shown. This is acceptable for Checkpoint 2, but for "highest possible standard," the project should justify that the goal is empirical comparison and robustness, not leaderboard optimization.

### Evaluation

**Status:** `strong`

Evaluation uses validation, final official test, confusion matrices, not-merged precision/recall/F1, balanced accuracy, average precision, ROC-AUC, threshold tuning, risk-band lift, calibration, bootstrap intervals, validation-test gap, and stress tests.

Weakness: the official test is not unseen-project validation. The repo handles this through split overlap and group-holdout diagnostics, but this must be clearly stated.

### Imbalance Handling

**Status:** `strong`

The target is about 89% merged and 11% not merged. The repo correctly avoids raw accuracy as the main metric, uses class weighting/sample weighting, tracks minority-class metrics, and shows why dummy majority is useless for not-merged detection.

Weakness: tuned threshold improves F1 only modestly and reduces recall. This is not a flaw if presented as a precision/recall tradeoff.

### Feature Selection

**Status:** `strong`

Feature selection is leakage-aware and policy-based rather than purely statistical. The project separates headline leakage-safer features, ultra-conservative features, integrator-assumed sensitivity, timing-assumed sensitivity, and T2 review-process features.

Weakness: the word "safe" should be used carefully. The strongest phrase is "leakage-safer under stated prediction-time assumptions."

### Leakage Control

**Status:** `strong`, with residual risk

Identifiers, direct post-outcome fields, close-time fields, comments/sentiment/review-process fields, CI outcomes, and target-adjacent success-rate fields are excluded from the early headline model or used only in sensitivity/contract comparisons.

Residual risk: some project and contributor snapshot fields may still depend on exact source extraction timing. The repo acknowledges this, which is the right academic posture.

### Reproducibility

**Status:** `partial`

Strong points: the final notebook is generated from `scripts/build_analysis_notebooks.py`; final report generation and output validation scripts exist; final validation passes under `.venv/bin/python`.

Weak points: system `python3` does not have dependencies installed. Raw data and large artifacts depend on local files/Git LFS. The Checkpoint 2 notebook is unexecuted.

### Code Quality

**Status:** `strong`

The implementation uses functions, pipelines, metric helpers, table generators, and validation checks. It is organized for repeatable output generation.

Weakness: `scripts/build_analysis_notebooks.py` is large and monolithic. That is acceptable for a generated course notebook, but not ideal software design.

### Notebook Quality

**Status:** `partial`

Final notebook: strong. It is executed, narrative-rich, and has outputs. Checkpoint 2 notebook: weak as a standalone deliverable because it has no executed code outputs.

## 5. Storytelling / Presentation Audit

### Can the Current Work Support a Strong Checkpoint 2 Presentation?

**Status:** `partial`

Yes, if the team uses the full final-package evidence and trims it into a Checkpoint 2 story. No, if the team relies only on `deliverables/checkpoint-2/checkpoint2_pr_merge_modeling.ipynb` as submitted evidence.

### Is the Narrative Coherent?

**Status:** `strong` in final package

The final story is coherent:

1. The target is imbalanced.
2. Accuracy is not enough.
3. Feature timing is the main validity threat.
4. Early PR-level signals predict moderately.
5. Review-process fields improve metrics but represent later information.
6. Generalization weakens under stricter time/entity splits.
7. The conclusion is empirical association, not causality or automation.

The Checkpoint 2 story is not yet packaged coherently in the Checkpoint 2 folder.

### What Slides Could Currently Claim?

The current slides can defensibly claim:

- The project has a stable research question and target.
- The dataset is large and imbalanced.
- The team uses leakage-aware feature contracts.
- Random Forest is the selected headline model by validation not-merged F1.
- Final official test not-merged F1 is 0.314 default and 0.326 validation-tuned.
- T2 review-process features reach 0.436 test F1 but are later-stage information.
- Calibration improves probability honesty but does not make decisions reliable.
- Official test has clean PR ids but heavy project/creator overlap.
- K-means/PCA is profile analysis, not a classifier.

### Where the Story Is Weak

- The Checkpoint 2 notebook does not show outputs.
- The final package may feel like it skipped past Checkpoint 2 unless the team explicitly says which pieces are preliminary modelling evidence.
- The headline performance is moderate. That is academically fine, but only if the team avoids a "we built a strong predictor" framing.
- The unsupervised component is not central and should not consume much presentation time.

## 6. Highest-Priority Gaps

### 1. Execute or Regenerate the Checkpoint 2 Notebook

**Why it matters:** An unexecuted notebook is the easiest criticism and makes strong underlying work look unfinished.

**What exactly should be done:** Run `deliverables/checkpoint-2/checkpoint2_pr_merge_modeling.ipynb` in the project `.venv`, save outputs, and confirm the model comparison table and plot are visible.

### 2. Add Checkpoint 2-Specific Metric and Figure Outputs

**Why it matters:** The final folder has many outputs, but the Checkpoint 2 folder lacks visible milestone artifacts.

**What exactly should be done:** Save at least a model comparison CSV/PNG, target distribution figure, and feature-policy summary under `deliverables/checkpoint-2/`.

### 3. Write a Short Checkpoint 2 Narrative

**Why it matters:** The professor should not have to infer how final-package work maps to Checkpoint 2.

**What exactly should be done:** Add a short Checkpoint 2 README or notebook section covering question, feature policy, preprocessing, models, validation metrics, interpretation, and next steps.

### 4. Make Feature Timing Assumptions Presentation-Ready

**Why it matters:** Leakage is the biggest academic risk.

**What exactly should be done:** Prepare a compact table with used, excluded, sensitivity-only, and T2 review-process features. Use "leakage-safer" instead of "safe" when discussing retained features.

### 5. Rehearse the Moderate-Result Defense

**Why it matters:** The final F1 values are useful but not impressive if framed as high performance.

**What exactly should be done:** Practice saying: "The result is moderate predictive association; the contribution is disciplined empirical evaluation and feature-timing analysis."

### 6. Clarify Official Test vs External Validity

**Why it matters:** The official test has heavy project and creator overlap, so it is not a clean unseen-project benchmark.

**What exactly should be done:** In presentation notes, explain that PR-id overlap is zero, but 97.85% of test rows come from seen projects and 84.09% from seen creators.

### 7. Keep T2 Review-Process Results Out of the Headline Early Claim

**Why it matters:** T2 is stronger but uses later information. Misstating it would look like leakage.

**What exactly should be done:** Always present T2 as "late-stage review-process signal," not as the early model.

### 8. Document Environment Usage

**Why it matters:** `python3 scripts/validate_final_outputs.py` fails because system Python lacks `pandas`, even though `.venv/bin/python scripts/validate_final_outputs.py` passes.

**What exactly should be done:** In Checkpoint 2 instructions, use `.venv/bin/python` and mention the local requirements file.

### 9. Reduce Artifact Overload for Reviewers

**Why it matters:** The final package contains many CSVs and figures. Without a reading path, it can feel noisy.

**What exactly should be done:** Point graders first to the executed notebook, final report, model comparison, final metrics, prediction contracts, and assignment coverage map.

### 10. Keep the Unsupervised Component Honest and Short

**Why it matters:** K-means/PCA is acceptable but not the project's strongest evidence.

**What exactly should be done:** Present it as profile discovery only. Mention target is used after fitting for interpretation, not during clustering.

## 7. Recommended Action Plan

### Must Do Before Checkpoint 2

- Execute and save `deliverables/checkpoint-2/checkpoint2_pr_merge_modeling.ipynb`.
- Add Checkpoint 2-visible outputs or references to the final outputs.
- Add a short Checkpoint 2 milestone narrative.
- Prepare a one-slide or one-table feature-availability contract.
- Lead evaluation with not-merged precision/recall/F1, balanced accuracy, and average precision.
- Rehearse the leakage and moderate-performance defense.

### Should Do If Time Allows

- Add a compact Checkpoint 2 README inside `deliverables/checkpoint-2/`.
- Export a small Checkpoint 2 presentation subset from the final slide deck.
- Add a short "what changed since Checkpoint 1" section.
- Add a "what remains for final submission" section.
- Include validation command notes using `.venv/bin/python`.

### Nice To Have

- Add a small diagram of T0/T1/T2 prediction-time contracts.
- Add a single table comparing ultra-conservative, headline, integrator-assumed, and T2 feature policies.
- Add a short appendix explaining why comments and survey data are not main modelling inputs.
- Add optional manual verification notes for opening the PDF/slides and checking image rendering.

## 8. Final Verdict

**Are we actually ready?** Not if "ready" means the Checkpoint 2 folder alone can stand up to professor scrutiny. Yes, if the team is allowed to present the current final-package evidence and explain that it supersedes the thin Checkpoint 2 notebook.

**Current grade band:** High-grade trajectory for the full repo/final package. Medium or partial if judged only by `deliverables/checkpoint-2/checkpoint2_pr_merge_modeling.ipynb`.

**Most realistic path to excellent:** Package the strong final-package work into a clean Checkpoint 2 story. Do not add more modelling first. The fastest grade improvement is presentation hygiene: execute the Checkpoint 2 notebook, surface the baseline vs Random Forest comparison, show the feature-availability contract, use imbalance-aware metrics, and clearly defend moderate predictive association with leakage and generalization limits.

**Manual verification recommended:** A human should open the executed Checkpoint 2 notebook, final report PDF, and slides before presentation to confirm that outputs render, figures are readable, and the speaking order matches the milestone story.
