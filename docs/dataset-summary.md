# Dataset Summary

## Source

- Zenodo record: [GitHub Pull Request Analysis: Sentiment Data and Developer Survey Responses](https://zenodo.org/records/10049493)
- DOI: `10.5281/zenodo.10049493`

Locked project framing:

- research question: "Can PR-level features help explain and predict PR merge outcomes on GitHub?"
- current milestone: Checkpoint 1
- main target: `merged_or_not`

## Files

### `prfeatures_train_data.csv`

- about 354 MB
- verified shape: `1,045,883` rows and `72` columns
- PR-level dataset
- roughly 72 features plus the target-like field `merged_or_not`
- strong candidate for the main exploratory and supervised analysis

Sample columns include:

- `merged_or_not`
- `first_pr`
- `core_member`
- `prior_review_num`
- `team_size`
- `language`
- `open_issue_num`
- `project_age`
- `stars`
- `num_participants`
- `lifetime_minutes`
- `num_comments`
- `num_commits`
- `files_changed`

### `prfeatures_test_data.csv`

- about 88 MB
- verified shape: `260,195` rows and `72` columns
- natural holdout set for final validation if the project uses the PRFeatures split as provided

### `pr_comments_dataset_publish.csv`

- about 219 MB
- comment-level dataset
- includes sentiment and developer-discussion fields
- out of scope for Checkpoint 1

### `survey_responses_raw.csv`

- about 23 KB
- 22 survey responses
- excluded as the project basis
- not part of the Checkpoint 1 workflow

## Locked Checkpoint 1 Scope

- use `prfeatures_train_data.csv` as the main dataset
- use `prfeatures_test_data.csv` only for schema and target consistency checks
- exclude the comments dataset for now
- do not base the project on the 22-row survey file

## Verified Target Facts

Target chosen for the project:

- `merged_or_not`

Verified class distribution:

- train: `932,538` merged (`89.16%`), `113,345` not merged (`10.84%`)
- test: `232,073` merged (`89.19%`), `28,122` not merged (`10.81%`)

Implication:

- the target is strongly imbalanced and later evaluation must account for that

## Recommended Analytical Framing

### Descriptive track

- summary statistics for the PR-level dataset
- feature distributions
- merged vs. not merged comparisons
- correlation analysis on numeric fields
- leakage-aware feature review before any predictive claims

### Supervised track

Primary option:

- classification on `merged_or_not`

Good baseline models:

- logistic regression
- decision tree
- random forest
- gradient boosting or XGBoost if allowed

Evaluation ideas:

- train/validation split or cross-validation on the training file
- final check on `prfeatures_test_data.csv`
- accuracy, precision, recall, F1, ROC-AUC
- confusion matrix

### Unsupervised track

Good options:

- cluster PRs by review/process characteristics
- cluster aggregated PR-comment behavior
- use PCA or UMAP only for visualization, not as the main result by itself

Checkpoint 1 note:

- do not start this track yet; first finish schema auditing, target audit, conservative feature screening, and initial EDA

Baseline algorithms:

- K-means
- hierarchical clustering
- DBSCAN if density-based structure seems plausible

## Risks To Watch

### Scale

The datasets are not tiny. In Colab, start with:

- selected columns
- row samples
- PR-level tables before comment-level tables

### Target leakage

Some variables may encode information that is only fully known after the PR outcome is decided. Any predictive framing must justify which features are available at prediction time.

### Missing data

The dataset description says missing and negative values were already handled in preprocessing, but we still need to audit missingness ourselves and document what we keep or drop.

### Class imbalance

`merged_or_not` may be imbalanced. We should verify this early and choose evaluation metrics accordingly.

## Strong First Analysis Slice

If the team wants a clean first checkpoint, this is the safest path:

1. use `prfeatures_train_data.csv` as the main dataset
2. audit missingness, data types, and target distribution
3. produce initial EDA on a manageable sample
4. define a leakage-aware feature subset
5. document which features are excluded, ambiguous, or conservative-safe
6. extract slide-ready findings for the Checkpoint 1 presentation
