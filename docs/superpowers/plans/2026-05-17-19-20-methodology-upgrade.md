# 19-20 Methodology Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the final ADES package from a single moderate classifier story into a prediction-time, generalization, calibration, and statistical-comparison study that is credible for a 19-20 grade target.

**Architecture:** Keep `scripts/build_analysis_notebooks.py` as the source of truth for all generated notebook cells, CSVs, and figures. Extend `scripts/validate_final_outputs.py` first so the new expected outputs fail before implementation, then update the generator, report builder, and slide source. Re-run the notebook and validators to regenerate all final outputs.

**Tech Stack:** Python, pandas, scikit-learn, matplotlib/seaborn, nbformat, reportlab, Marp.

---

### Task 1: Add Failing Validation Gates

**Files:**
- Modify: `scripts/validate_final_outputs.py`

- [ ] **Step 1: Add required final artifacts**

Add these filenames to `REQUIRED_FINAL_FILES`:

```python
"prediction_contract_comparison.csv",
"prediction_contract_feature_map.csv",
"review_process_feature_audit.csv",
"comment_dataset_profile.csv",
"calibration_model_comparison.csv",
"full_generalization_benchmarks.csv",
"paired_model_delta_intervals.csv",
```

- [ ] **Step 2: Add required figures**

Add these filenames to `REQUIRED_FIGURES`:

```python
"prediction_contract_comparison.png",
"calibration_model_comparison.png",
"full_generalization_benchmarks.png",
```

- [ ] **Step 3: Add table-level assertions**

Add validator functions that assert:

```python
prediction_contract_comparison.csv contains T0, T1, T2 rows for validation and test.
prediction_contract_feature_map.csv contains stage, feature, availability, model_role.
review_process_feature_audit.csv contains review/process fields and timing decisions.
comment_dataset_profile.csv reports raw_rows and joined_to_prfeatures=false.
calibration_model_comparison.csv contains uncalibrated, sigmoid, and isotonic rows.
full_generalization_benchmarks.csv contains official_test, temporal_holdout, project_holdout, creator_holdout rows.
paired_model_delta_intervals.csv contains baseline_model, comparison_model, metric, delta, ci_lower_95, ci_upper_95.
```

- [ ] **Step 4: Run validation and verify it fails**

Run:

```bash
.venv/bin/python scripts/validate_final_outputs.py
```

Expected: failure because the new artifacts do not exist yet.

### Task 2: Add Feature Contracts and Comment Dataset Audit

**Files:**
- Modify: `scripts/build_analysis_notebooks.py`

- [ ] **Step 1: Define contracts**

Add explicit feature lists:

```python
t0_creation_features = [...]
t1_diff_features = [...]
t2_review_process_features = [...]
```

T0 keeps only strongest early-context features. T1 equals the current headline leakage-safer policy. T2 adds PRFeatures review/process fields only, not direct close-time or target-adjacent fields.

- [ ] **Step 2: Generate contract feature map**

Create `prediction_contract_feature_map.csv` with one row per feature and columns:

```python
contract, feature, availability, risk_level, model_role, rationale
```

- [ ] **Step 3: Generate review-process audit**

Create `review_process_feature_audit.csv` listing review/comment/process fields, whether they enter T2, and why they are excluded from T0/T1.

- [ ] **Step 4: Profile comment dataset without forcing a bad join**

Read `pr_comments_dataset_publish.csv` columns only and write `comment_dataset_profile.csv` with raw rows, unique repositories, unique PR keys, label distribution, sentiment/comment columns, and `joined_to_prfeatures=false` because PRFeatures does not expose owner/repo/pull number.

### Task 3: Compare Prediction-Time Contracts

**Files:**
- Modify: `scripts/build_analysis_notebooks.py`

- [ ] **Step 1: Add reusable contract evaluator**

Create a function that fits Random Forest on the internal validation split for each contract, then retrains on all training rows and scores official test.

- [ ] **Step 2: Write outputs**

Write:

```python
prediction_contract_comparison.csv
figures/prediction_contract_comparison.png
```

The table must show validation/test not-merged precision, recall, F1, average precision, feature count, and contract interpretation.

### Task 4: Add Calibration Model Comparison

**Files:**
- Modify: `scripts/build_analysis_notebooks.py`

- [ ] **Step 1: Add calibration helpers**

Use `CalibratedClassifierCV` with `method="sigmoid"` and `method="isotonic"` on a runtime-bounded stratified sample.

- [ ] **Step 2: Compare calibrated scores**

Write `calibration_model_comparison.csv` with method, Brier score, weighted absolute calibration error, average precision, ROC-AUC, and F1 at the validation-tuned threshold.

- [ ] **Step 3: Plot Brier/ECE comparison**

Write `figures/calibration_model_comparison.png`.

### Task 5: Add Full-Scale Generalization Benchmarks

**Files:**
- Modify: `scripts/build_analysis_notebooks.py`

- [ ] **Step 1: Evaluate more than sampled stress tests**

Create `full_generalization_benchmarks.csv` with official test plus larger temporal, project-group, and creator-group holdouts. Use enough rows to be meaningfully larger than the existing stress diagnostic while keeping runtime feasible.

- [ ] **Step 2: Plot F1 and AP by benchmark**

Write `figures/full_generalization_benchmarks.png`.

### Task 6: Add Paired Statistical Deltas

**Files:**
- Modify: `scripts/build_analysis_notebooks.py`

- [ ] **Step 1: Bootstrap model deltas**

Compare headline Random Forest against Hist Gradient Boosting on the same validation rows and bootstrap paired deltas for not-merged F1 and average precision.

- [ ] **Step 2: Write output**

Write `paired_model_delta_intervals.csv`.

### Task 7: Update Report, Slides, README, and Self-Review

**Files:**
- Modify: `scripts/build_final_report.py`
- Modify: `slides/final_presentation.md`
- Modify: `deliverables/final/README.md`
- Modify: `deliverables/final/self_review.md`

- [ ] **Step 1: Reframe final report around contracts**

Add sections for prediction-time contracts, comment dataset audit, calibrated score comparison, full generalization benchmarks, and paired model deltas.

- [ ] **Step 2: Reframe slides**

Make the deck story: “when signal becomes available,” then calibration and generalization.

- [ ] **Step 3: Update package manifest and self-review**

Add the new artifacts and re-grade implications.

### Task 8: Regenerate and Verify

**Files:**
- Generated outputs under `deliverables/final/`
- Generated slides under `slides/`

- [ ] **Step 1: Regenerate notebook**

Run:

```bash
.venv/bin/python scripts/build_analysis_notebooks.py
```

- [ ] **Step 2: Execute final notebook**

Run:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace deliverables/final/final_pr_merge_analysis.ipynb --ExecutePreprocessor.timeout=-1
```

- [ ] **Step 3: Build report and validate outputs**

Run:

```bash
.venv/bin/python scripts/build_final_report.py
.venv/bin/python scripts/validate_final_outputs.py
```

- [ ] **Step 4: Export slides**

Run:

```bash
npx -y @marp-team/marp-cli slides/final_presentation.md -o slides/final_presentation.pdf --allow-local-files
```

- [ ] **Step 5: Render PDF contact sheets**

Use PyMuPDF to render report and slide contact sheets under `tmp/pdfs/rendered/` and inspect them for clipping, blank pages, and broken figures.

