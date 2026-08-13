# GelatiCo Business Guide to the Repository

## Why this repository exists

This project helps GelatiCo decide which customer situations deserve attention when the commercial
team cannot investigate every account every month. It does not automatically declare that a
customer is anomalous or prescribe a commercial action. It creates a prioritized, explainable
starting point for human review.

The repository compares two complementary investigation lenses:

1. **S2 deviation** asks whether a customer's recent ordering pattern looks unusual relative to
   that customer's own history and comparable customers.
2. **Hurdle/cadence forecasting** estimates the probability of next-month ordering activity and
   highlights cases where expected behavior may not occur.

The central recommendation is to use both lenses in a limited, blinded review. Their low overlap
means that they tend to surface different cases, which can broaden coverage without pretending that
either method is business truth.

## The business answer in one minute

- The selected forecasting model predicted next-month activity better than the two simple
  historical baselines on the held-back January--June 2026 period.
- At a review capacity of 250 cases per month, the S2 and cadence queues overlapped by only 8.58%.
  They therefore provide different investigation candidates.
- The technical work supports running a mixed human-review pilot.
- It does **not** yet establish which cases are genuinely new, useful, actionable, or financially
  valuable. Only GelatiCo reviewers can supply that evidence.

## A 15-minute reading path

| Time | Open | What it answers |
|---:|---|---|
| 3 minutes | [README](../README.md) | What decision is supported and what remains outside the evidence. |
| 5 minutes | [Notebook 99](../notebooks/99_final_findings.ipynb) | The executive dashboard, findings, recommendations, and next steps. |
| 4 minutes | [Final report](../report.md) | The business problem, principal results, uncertainty, and decision boundary. |
| 3 minutes | [Stakeholder handoff](stakeholder_validation_handoff.md) | How technical candidates were converted into a controlled GelatiCo review package. |

Readers who want the analytical detail can then use:

- [notebook 05](../notebooks/05_peer_strategy_baseline.ipynb) for data quality, S2, and peer
  comparisons;
- [notebook 06](../notebooks/06_hurdle_cadence_model.ipynb) for forecasting, chronological model
  selection, GridSearchCV, and evaluation; and
- [notebook 07](../notebooks/07_alert_episodes_and_pilot.ipynb) for capacity, episode construction,
  and the blinded pilot design.

## Plain-language glossary

| Term | Business meaning |
|---|---|
| **Investigation lens** | A method for finding cases worth looking at; not a final judgment. |
| **S2 deviation** | A measure of how different a customer-month looks from relevant history and peers. |
| **Cadence** | The timing and frequency pattern of customer orders. |
| **Capacity-bounded queue** | A ranked list limited to the number of cases the team can realistically review. |
| **Log loss** | A measure of probability quality; lower is better, especially when avoiding confident mistakes matters. |
| **Jaccard overlap** | The share of selected cases common to two queues; it measures similarity, not correctness. |
| **Forward holdout** | A later historical period kept out of model selection and used as a more realistic test. |
| **Blinded review** | Reviewers assess the business evidence without being told which method selected the case. |

## What GelatiCo can use now

The project is ready to support a controlled review process:

1. agree on the number of cases the commercial team can review;
2. draw cases from both investigation lenses without revealing their source;
3. provide the account context available only inside GelatiCo's authorized environment;
4. record a structured outcome for every case; and
5. compare usefulness, novelty, actionability, review effort, and coverage before deciding whether
   to continue, revise, or stop.

The public repository demonstrates the analytical and governance design. Private customer
references and row-level business context belong only in the authorized review package.

## The decisions that still require GelatiCo

For each reviewed case, the most useful outcome is one of the following:

- **actionable new:** a relevant situation that was not already known and merits follow-up;
- **relevant already known:** a genuine situation already managed through normal processes;
- **real but not actionable:** an explainable change that does not justify intervention;
- **false or unhelpful:** the evidence does not support a useful business case; or
- **insufficient context:** the reviewer cannot decide from the available information.

These outcomes answer questions that model metrics cannot:

- Does the case reveal something the account team did not already know?
- Is the change commercially meaningful?
- Can the team act on it?
- How much time does a useful review require?
- Does the combined queue add coverage beyond standard reporting?

The full measurement and decision protocol is in
[Business Value Validation](business_value_validation.md).

## What should not be concluded

The repository does not prove that:

- every selected case is an anomaly;
- a low model probability means churn or lost revenue;
- low overlap means one method is more accurate;
- a technical confidence score is a probability of commercial importance;
- contacting a selected customer will improve an outcome; or
- the system has demonstrated adoption, financial impact, or return on investment.

Those conclusions require completed stakeholder outcomes and an agreed operating test.

## Privacy and access boundary

The public repository contains executed notebooks and aggregate evidence, but no customer names,
customer identifiers, order-level records, absolute commercial values, or reviewer responses.
Confidential checkpoints are required only for an authorized rerun. GelatiCo-specific enrichment
and identity resolution occur in the private environment and must not be copied into this public
repository.

## Where the work continued

The graded deliverable was deliberately kept focused. Additional neural-network, probabilistic,
LLM-assisted, consensus, and operational-delivery experiments were developed separately and did
not silently replace the submitted model. A public-safe summary appears in
[Beyond the Core Deliverable](post_deliverable_experiments.md).
