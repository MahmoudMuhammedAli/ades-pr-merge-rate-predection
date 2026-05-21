# Codex Workflow For This Project

## What Codex Can Help With

- translate assignment requirements into a concrete notebook plan
- write and debug Colab cells
- explain preprocessing decisions
- suggest charts and summary tables
- compare ML models and metrics
- check for leakage, imbalance, or weak evaluation choices
- help turn notebook results into slide content

## Best Way To Work Together

When asking for help, provide one or more of:

- the notebook cell you are working on
- the error message
- a CSV header or small sample
- a screenshot of an output or chart
- the current objective for that section

## High-Value Prompt Patterns

Use prompts like these during the semester:

- "Build a Colab cell to load this CSV efficiently and print a schema summary."
- "Review this preprocessing plan and point out leakage or weak assumptions."
- "Suggest 5 useful EDA plots for this dataset and write the code."
- "Compare logistic regression, random forest, and histogram gradient boosting for this target."
- "Help me explain these results in report language for the notebook."
- "Turn these findings into 5 presentation-slide bullets."

## Working Rule For Large Files

For the large CSVs, ask Codex to help with:

- column selection
- chunked loading
- sampling
- aggregation
- memory-safe preprocessing

This is better than loading everything at once in early exploration.

## Checkpoint Strategy

### Checkpoint 1 by `2026-04-10`

Aim to have:

- research question
- dataset justification
- schema and missingness audit
- first EDA visuals
- one supervised baseline idea
- one unsupervised baseline idea

### Checkpoint 2 by `2026-05-15`

Aim to have:

- reproducible cleaning pipeline
- evaluated baselines
- clearer feature selection logic
- first serious findings and limitations

### Final by `2026-06-08`

Aim to have:

- cleaned and commented final notebook
- stable plots and tables
- final model comparison
- conclusions and threats to validity
- polished slides
