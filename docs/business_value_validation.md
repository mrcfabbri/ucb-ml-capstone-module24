# Blinded Commercial Validation Protocol

## Objective

Estimate whether cases are correct, novel, actionable, and practical to review at a fixed monthly
capacity. Technical model scores and overlap do not answer these questions.

This review converts the technical shortlist into evidence for an operating decision. Before its
completion, the project conclusions are limited to technical forecasting and queue results.

## Sample

The June 2026 pilot targets 250 distinct customers:

| Stratum | Target |
|---|---:|
| S2/hurdle agreement core | 40 |
| Channel-only S2 | 60 |
| Hurdle-only | 60 |
| Peer-definition disagreement | 40 |
| Background | 50 |

If a stratum is smaller than its target, the deterministic capacity-fill stratum completes the
sample. The private allocation key records inclusion probabilities. Reviewers receive case IDs and
safe contextual evidence, not detector names, scores, ranks, or sampling strata.

### Reviewer instructions

For each case, independently record the outcome that best describes the customer's situation and
the time needed to reach that decision. Sample membership carries no importance label. The sample
includes agreement cases, disagreement cases, and background cases for a balanced comparison.

## Canonical outcomes

- `actionable_new`
- `relevant_already_known`
- `real_but_not_actionable`
- `false_or_unhelpful`
- `insufficient_context`

Also record review effort in minutes. Free-text notes remain in the approved private review system;
they are not committed to the repository.

## Analysis

- precision at capacity for `actionable_new`;
- relevance rate including `relevant_already_known`;
- false/unhelpful and insufficient-context rates;
- median review effort;
- customer, segment, geography, direction, and model-disagreement coverage;
- inverse-probability-weighted deployed-queue estimates;
- reviewer-clustered uncertainty if reviewers assess multiple cases.

Agreement-core cases measure consensus. Single-model disagreement tails measure incremental value.
They must not be pooled without weighting.

## Decision gate

Commercial leadership must define acceptable novelty, actionability, and effort. Until reviewed
outcomes meet that gate, channel remains the operational peer baseline and the hurdle model remains
a controlled challenger.

The decision gate should answer three concrete questions:

1. Is the share of `actionable_new` cases high enough for the team's capacity?
2. Is median review effort acceptable relative to the value of a useful finding?
3. Do disagreement tails add value beyond the business-friendly channel-S2 baseline?
