# Professor Defense Notes

Use this file for final rehearsal and Q&A. The goal is to keep the defense consistent with the report, slides, notebook, and generated metrics.

## Core Answer

The project answers the research question with a bounded yes: PR-level features do help explain and predict GitHub PR merge outcomes, but the early/submission-time signal is moderate. The best headline test result is not-merged F1 `0.314` at the default threshold and `0.326` after validation-only threshold tuning. Later review-process fields raise test F1 to `0.436`, but that is a different timing contract, not a better early model. The conclusion is not causal and not deployment-ready.

## If Asked Whether We Forced the Result

No. The strongest-looking feature policy is not used as the headline claim. Adding `prior_review_num` reaches test F1 `0.373`, but it is reported only as integrator-assumed sensitivity because it requires a stronger availability assumption. The report also shows weaker ultra-conservative performance (`0.273` test F1), validation-to-test drop, temporal/project/creator holdouts, calibration limits, and error profiles.

## If Asked About Leakage

The headline model excludes direct identifiers, clear post-outcome fields, review comments, sentiment, CI result fields, target-adjacent success rates, and close-time PR evolution fields. The remaining risk is timing ambiguity in project/contributor snapshot fields. The defense is not that these fields are perfectly safe; it is that the assumptions are explicit, risky alternatives are separated, and sensitivity models show how the answer changes.

## If Asked About the Math

The metric tables are confusion-matrix based and pass `scripts/validate_final_outputs.py`. Model choice and threshold tuning use internal validation only. The final model is retrained on all `1,045,883` training rows and then evaluated on the untouched `260,195` row official test file. The main metric is not-merged F1 because the target is imbalanced: about `89%` merged and `11%` not merged.

## If Asked About Generalization

The official split has no train/test PR-id overlap, but it has heavy entity reuse: `97.85%` of test rows come from projects seen in training and `84.09%` from creators seen in training. Therefore the official test is best described as new PRs in familiar ecosystems, not unseen-project generalization. The stricter holdouts are diagnostics that show the signal weakens under tougher conditions.

## If Asked About Calibration

The Random Forest score is useful for ranking risk, not as a literal probability. Calibration improves probability honesty on a held-out slice, reducing Brier score to about `0.086`, but calibrated default-threshold F1 is lower because the minority-class probability remains conservative under the `11%` base rate. Calibration and classification are intentionally discussed as separate questions.

## If Asked About Clustering

K-means/PCA is not used as proof that PRs naturally split into clean groups. It is included as an unsupervised profile analysis for the assignment requirement. The useful finding is descriptive: the smaller cluster has a higher not-merged rate (`20.55%`) and represents larger-project/smaller-change contexts.

## One-Sentence Closing

The defensible conclusion is that PR-level data contains real but moderate predictive signal, stronger results require later review-process information, and the project is an empirical association study rather than a causal or deployment-ready merge decision system.
