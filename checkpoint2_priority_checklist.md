# Checkpoint 2 Priority Checklist

- [ ] Execute and save `deliverables/checkpoint-2/checkpoint2_pr_merge_modeling.ipynb`. Owner: notebook owner. Urgency: critical.
- [ ] Add Checkpoint 2-visible model comparison outputs, ideally CSV plus PNG. Owner: modelling owner. Urgency: critical.
- [ ] Add a short Checkpoint 2 milestone narrative covering question, feature policy, preprocessing, models, metrics, interpretation, and next steps. Owner: report/presentation owner. Urgency: critical.
- [ ] Create a compact feature-availability table for used, excluded, sensitivity-only, and T2 review-process fields. Owner: methodology owner. Urgency: critical.
- [ ] Rehearse the moderate-result defense: early PR-level signal is useful but not high-performing, causal, or deployment-ready. Owner: whole team. Urgency: critical.
- [ ] Make imbalance-aware metrics the primary evaluation story: not-merged precision, recall, F1, balanced accuracy, and average precision. Owner: modelling owner. Urgency: high.
- [ ] Explain why raw accuracy and majority baseline are insufficient for the 89% / 11% target distribution. Owner: presentation owner. Urgency: high.
- [ ] Clarify that the official test has zero PR-id overlap but heavy project/creator overlap. Owner: methodology owner. Urgency: high.
- [ ] Keep T2 review-process performance separate from the early headline model. Owner: presentation owner. Urgency: high.
- [ ] Document that validation and final package checks should run with `.venv/bin/python`, not plain `python3`. Owner: reproducibility owner. Urgency: high.
- [ ] Add or reference Checkpoint 2-ready figures for target distribution, model comparison, threshold tradeoff, and final confusion matrix. Owner: notebook owner. Urgency: high.
- [ ] Reduce reviewer artifact overload by pointing first to the executed notebook, final report, model metrics, prediction contracts, and assignment coverage map. Owner: report owner. Urgency: medium.
- [ ] Keep the unsupervised component short and honest: K-means/PCA is profile discovery, not a classifier. Owner: modelling owner. Urgency: medium.
- [ ] Add a "what changed since Checkpoint 1" paragraph for presentation continuity. Owner: presentation owner. Urgency: medium.
- [ ] Add a "remaining work for final submission" paragraph so Checkpoint 2 feels like a milestone, not an unexplained final package. Owner: presentation owner. Urgency: medium.
- [ ] Manually open the final report PDF and slide PDF before presentation to confirm figures and text render correctly. Owner: presentation owner. Urgency: medium.
- [ ] Remove or explain duplicate-looking dependency file `requirements-local 2.txt` if it will be visible to graders. Owner: reproducibility owner. Urgency: low.
- [ ] Consider adding a simple T0/T1/T2 diagram if the presentation has time. Owner: presentation owner. Urgency: low.
