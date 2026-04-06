# Colab Setup Notes

## Practical Approach

This project is a good fit for Colab, but the datasets are large enough that we should work in two modes:

- fast mode with samples
- full mode for final runs

## Suggested Colab Opening Cell

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    silhouette_score,
)
```

If needed later:

```python
!pip install -q imbalanced-learn pyarrow
```

## Recommended Colab Workflow

1. Keep the raw Zenodo files in Google Drive.
2. Save smaller sampled or aggregated outputs for repeated experimentation.
3. Export milestone notebooks back into this repo under `notebooks/`.
4. Keep final figures and tables stable before building slides.

Current scope reminder:

- for Checkpoint 1, work only with `prfeatures_train_data.csv` and `prfeatures_test_data.csv`
- keep the test file limited to schema and target checks
- do not use the comments dataset or survey dataset in the current notebook flow

## Memory-Safe Habits

- read only the columns you need with `usecols=...`
- use `nrows=...` for first-pass exploration
- convert suitable columns to smaller dtypes
- aggregate comment-level data before heavy modeling
- save cleaned subsets to Parquet when possible

## Suggested Early Notebooks

- `01_data_audit.ipynb`
- `02_eda_prfeatures.ipynb`
- `03_supervised_baselines.ipynb`
- `04_unsupervised_analysis.ipynb`
- `05_final_story.ipynb`

## Suggested Research Framing

Safe first framing:

- supervised task: predict whether a PR is merged
- current Checkpoint 1 focus: understand the PRFeatures data before modeling
- use leakage-aware feature screening before any baseline models
- leave clustering and any broader multi-dataset work for later milestones
