# ADES Project Workspace

Semester workspace for the FEUP course "Data Analysis in Software Engineering" (2025/2026).

## Project Goal

Build a complete data analysis project around the chosen Zenodo dataset:
[GitHub Pull Request Analysis: Sentiment Data and Developer Survey Responses](https://zenodo.org/records/10049493)

Current locked research question:

- "Can PR-level features help explain and predict PR merge outcomes on GitHub?"

The assignment requires:

- data collection and justification
- preprocessing
- exploratory data analysis
- supervised and unsupervised ML tasks
- proper evaluation and comparison
- interpretation of findings
- a commented notebook, datasets, and slides

## Chosen Dataset

The Zenodo record contains four CSV files:

- `prfeatures_train_data.csv`
- `prfeatures_test_data.csv`
- `pr_comments_dataset_publish.csv`
- `survey_responses_raw.csv`

Locked Checkpoint 1 scope:

1. Use `prfeatures_train_data.csv` as the main dataset.
2. Use `prfeatures_test_data.csv` only for schema and label consistency checks.
3. Exclude `pr_comments_dataset_publish.csv` for Checkpoint 1.
4. Do not base the project on `survey_responses_raw.csv`.
5. Use `merged_or_not` as the locked prediction target.

Verified PRFeatures facts:

- `prfeatures_train_data.csv`: `1,045,883` rows and `72` columns
- `prfeatures_test_data.csv`: `260,195` rows and `72` columns
- `merged_or_not` exists in both files
- observed class imbalance is strong in both splits:
  - train: `89.16%` merged, `10.84%` not merged
  - test: `89.19%` merged, `10.81%` not merged

## Repository Layout

- `docs/` project notes, dataset notes, and Codex workflow guidance
- `notebooks/` Colab notebook roadmap and exported notebooks
- `data/raw/` downloaded original files
- `data/interim/` cleaned intermediate outputs
- `data/processed/` modeling-ready tables
- `data/samples/` lighter subsets for fast iteration in Colab
- `deliverables/checkpoint-1/` material for the April 10, 2026 checkpoint
- `deliverables/checkpoint-2/` material for the May 15, 2026 checkpoint
- `deliverables/final/` final notebook, dataset package, and support files
- `slides/` slide decks and presentation assets

## How We Should Use This Workspace

- Treat `scripts/build_analysis_notebooks.py` as the source for the checkpoint and final notebooks.
- Re-run the final notebook, report builder, validator, and slide export before submission.
- Use the generated CSVs in `deliverables/final/` as the source of truth for final metrics and speaking points.
- Keep the headline claim bounded to moderate predictive association, not causality or automated merge decisions.

## Local VS Code Setup

Use the project-local virtual environment so the notebook and VS Code share the same interpreter:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-local.txt nbconvert
```

This workspace includes `.vscode/settings.json`, so VS Code should default to `.venv/bin/python` once the environment exists.

Important: the CSVs in `data/raw/` are tracked with Git LFS. If you only have pointer files, pull the real data before running the notebook, or download the PRFeatures CSVs from the Zenodo record linked above.

## Milestones

- `2026-04-10` Checkpoint 1: problem framing, dataset choice, target audit, leakage-aware feature screening, initial EDA, presentation story
- `2026-05-15` Checkpoint 2: cleaner pipeline, baseline models, evaluation plan, preliminary findings
- `2026-06-08` Final submission: notebook, datasets, and updated slide deck
- `2026-06-11` Final presentation and discussion

## Current Completion Status

This branch contains the end-to-end project package:

- Checkpoint 1 presentation notebook in `deliverables/checkpoint-1/`
- Checkpoint 2 modeling notebook in `deliverables/checkpoint-2/`
- Final analysis notebook, report, metric tables, cluster profiles, figures, and self-review in `deliverables/final/`
- Final presentation source/PDF in `slides/`

The remaining human-facing work is presentation rehearsal: use the final notebook outputs and slide source to explain the feature-availability contract, class imbalance, model comparison, stress tests, clustering limits, and why the findings are predictive/associational rather than causal.
