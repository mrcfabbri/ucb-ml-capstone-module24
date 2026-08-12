# Module 24 Evaluation Reference

## Metric interpretation

The table compares models on the same untouched January--June 2026 period. **Lower log loss is
better** because it rewards useful probability estimates; **higher ROC-AUC and average precision
are better** because they reward ranking active months above inactive months. The selected model
therefore has stronger historical prediction evidence than the two simple baselines. Anomaly
correctness, commercial usefulness, and financial impact are outside this evaluation.

<!-- METRICS_START -->
The selected activity model is evaluated on an untouched January--June 2026 forward window.
It predicts self-supervised next-month ordering activity. Commercial anomaly correctness is
evaluated separately.

| Model | Log loss | ROC-AUC | Average precision | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Selected activity model (histogram gradient boosting) | 0.2765 | 0.8874 | 0.6523 | 0.5480 | 0.7063 |
| Logistic regression candidate | 0.2967 | 0.8759 | 0.6219 | 0.5560 | 0.7241 |
| Smoothed activity Markov | 0.3767 | 0.6931 | 0.3100 | 0.0000 | 0.5000 |
| Global activity rate | 0.4393 | 0.5000 | 0.1538 | 0.0000 | 0.5000 |

Customer-clustered bootstrap (1,000 resamples, seed 42) gives selected-minus-Markov log-loss
difference -0.1002 (95% CI [-0.1028, -0.0975]) and ROC-AUC difference +0.1943 (95% CI [0.1898,
0.1982]). These intervals quantify sampling uncertainty within the fixed forward window. Business
value is evaluated in the commercial-review stage.

At 250 cases/month, channel S2 versus hurdle queue overlap is 0.0858 Jaccard (237 shared
customer-months). This supports a mixed blinded pilot. Queue usefulness requires reviewer
outcomes.
<!-- METRICS_END -->

## Commercial validation stage

The technical comparison is complete. The next evidence must come from blinded commercial review:
whether cases are novel, actionable, practical to review, and worth operating at the proposed
capacity. See [the validation protocol](docs/business_value_validation.md).
