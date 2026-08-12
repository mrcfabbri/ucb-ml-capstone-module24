# Module 24 Submission Guide

## Project summary

The project helps a commercial team decide which customer situations to investigate when it has
limited monthly review capacity. It compares two ways to create candidates:

- **Channel S2** is a business-friendly deviation score. It highlights customer-months that differ
  sharply from the customer's own history and similar customers.
- **Hurdle/cadence** is a forecast. It estimates whether a customer will order next month and,
  for active months, the likely ordering pattern.

The two queues are deliberately kept separate. Their low overlap means they can contribute
different cases to a pilot. Commercial value requires separate reviewer outcomes.

## Suggested reading path

| Review objective | Artifact | Coverage |
|---|---|---|
| The recommendation | Notebook 99, then `report.md` | What is supported, what is not, and the next decision gate. |
| How S2 works | Notebook 05 | Data checks, peer definitions, fixed-capacity comparisons, and why channel remains the baseline. |
| Whether the forecast is credible | Notebook 06 | Time-safe model selection, untouched forward testing, calibration, and uncertainty. |
| How this becomes human work | Notebook 07 | Monthly capacity, alert episodes, pilot sampling, and blinding. |
| How commercial value will be tested | `business_value_validation.md` | Outcomes, review protocol, and the decision gate. |

## Metric definitions

- **Log loss:** a probability-quality measure; lower is better. It is the primary selection
  metric because the workflow needs usable probabilities, not only a yes/no label.
- **ROC-AUC and average precision:** ranking measures. Higher is better; they show whether cases
  that order next month tend to receive higher forecast probabilities.
- **Calibration/Brier score:** checks whether a stated probability resembles the observed rate.
- **Jaccard overlap:** selected cases in common divided by cases selected by either queue. It
  measures similarity between queues, not accuracy.
- **Fixed monthly capacity:** the maximum number of cases a team can investigate per month. Every
  queue comparison uses the same capacity, so the comparison is operationally fair.

## Evidence boundary

The notebooks demonstrate historical, technical properties: forward forecasting performance,
queue sensitivity to peer definitions, and how much the two queues differ. Genuine anomaly status,
reviewer usefulness, and financial value are evaluated through the blinded commercial review.

## Reproducibility and privacy

The saved notebooks contain aggregate, privacy-checked evidence and can be read without private
data. Re-running them requires three authorized private checkpoints described in the README. Do
not add those inputs, generated row-level outputs, or local environments to a submission. Use the
curated package command in the README when preparing the hand-in.
