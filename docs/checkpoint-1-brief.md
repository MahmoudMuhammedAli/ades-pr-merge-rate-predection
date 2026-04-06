# Checkpoint 1 Brief

This file locks the current project framing so future notebook, slide, and documentation work stays aligned.

## Course Context

- Course: Análise de Dados e Engenharia de Software / Data Analysis in Software Engineering (`MESW0009`)
- Institution: FEUP
- Current milestone: Checkpoint 1 presentation

Important dates:

- `2026-04-10` Checkpoint 1 presentation
- `2026-05-15` Checkpoint 2 presentation
- `2026-06-08` Final submission
- `2026-06-11` Final presentation and discussion

## Locked Research Question

- "Can PR-level features help explain and predict PR merge outcomes on GitHub?"

Important wording rule:

- study association and prediction
- do not claim causality

## Locked Checkpoint 1 Scope

- main dataset: `prfeatures_train_data.csv`
- test set role for now: `prfeatures_test_data.csv` only for schema and label consistency checks
- excluded for Checkpoint 1: `pr_comments_dataset_publish.csv`
- excluded as project basis: `survey_responses_raw.csv`
- target variable: `merged_or_not`

## Verified Dataset Facts

- `prfeatures_train_data.csv`: `1,045,883` rows, `72` columns
- `prfeatures_test_data.csv`: `260,195` rows, `72` columns
- target present in both files: `merged_or_not`

Observed target distribution:

- train: `932,538` merged (`89.16%`), `113,345` not merged (`10.84%`)
- test: `232,073` merged (`89.19%`), `28,122` not merged (`10.81%`)

Checkpoint implication:

- the classification target is strongly imbalanced

## Feature-Risk Starting Point

Target:

- `merged_or_not`

Exclude immediately as identifiers or metadata:

- `id`
- `project_id`
- `creator_id`
- `last_closer_id`

Exclude immediately as likely post-outcome or direct leakage:

- `last_close_time`
- `lifetime_minutes`
- `reopen_or_not`

Treat as ambiguous or timing-sensitive for Checkpoint 1:

- `num_comments`
- `has_comments`
- `num_participants`
- `core_comment`
- `contrib_comment`
- `inte_comment`
- `has_exchange`
- `at_tag`
- `num_code_comments`
- `num_code_comments_con`
- `perc_neg_emotion`
- `perc_pos_emotion`
- `comment_conflict`
- `contrib_open`
- `contrib_cons`
- `contrib_extra`
- `contrib_agree`
- `contrib_neur`
- `inte_open`
- `inte_cons`
- `inte_extra`
- `inte_agree`
- `inte_neur`
- `perc_contrib_pos_emo`
- `perc_contrib_neg_emo`
- `perc_inte_pos_emo`
- `perc_inte_neg_emo`
- `social_strength`
- `same_user`
- `ci_build_num`
- `ci_failed_perc`
- `integrator_availability`

Candidate safe features for conservative Checkpoint 1 EDA:

- `first_pr`
- `prior_review_num`
- `core_member`
- `prior_interaction`
- `followers`
- `prev_pullreqs`
- `account_creation_days`
- `contrib_perc_commit`
- `sloc`
- `team_size`
- `language`
- `open_issue_num`
- `project_age`
- `open_pr_num`
- `fork_num`
- `pr_succ_rate`
- `test_lines_per_kloc`
- `stars`
- `test_cases_per_kloc`
- `asserts_per_kloc`
- `perc_external_contribs`
- `requester_succ_rate`
- `churn_addition`
- `churn_deletion`
- `description_length`
- `test_inclusion`
- `ci_exists`
- `test_churn`
- `num_commits`
- `src_churn`
- `files_changed`
- `friday_effect`

## Checkpoint 1 Story

Checkpoint 1 should demonstrate:

- the research question is valid
- the dataset fits the question
- the target is correct
- the team understands schema and class imbalance
- the team has identified unsafe or timing-ambiguous features
- the team has started serious EDA on a conservative safe subset
- the project is ready for modeling work in Checkpoint 2

Planned slide flow:

- Slide 1: problem and motivation
- Slide 2: dataset choice and scope
- Slide 3: target variable and class imbalance
- Slide 4: leakage-aware feature selection
- Slide 5: first EDA findings on safe features
- Slide 6: next steps for Checkpoint 2
