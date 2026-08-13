# Beyond the Core Deliverable

## Purpose and scope

The four submitted notebooks tell one focused story: compare two investigation lenses, evaluate
the forecasting model chronologically, and design a capacity-bounded blinded pilot. Beyond that
core storyline, the project also explored whether other models, review mechanisms, and delivery
controls could add useful evidence.

These extensions were additive research. They did not overwrite the submitted notebooks, retune
the selected model against later evidence, or convert technical signals into confirmed business
outcomes. This appendix summarizes what was built, what was learned, and the resulting disposition.

The standalone public repository remains a curated, privacy-safe deliverable rather than a dump of
the private research workspace. Experimental source data, row-level outputs, provider responses,
identity crosswalks, and private run directories are not included here. The figures below are
aggregate results from versioned research artifacts; they document the work but do not make every
extension publicly rerunnable without the authorized environment.

## Experiment map

| Extension | What was built | Principal finding | Disposition |
|---|---|---|---|
| Autoencoder detector | An unsupervised PyTorch neural network that compresses and reconstructs ten standardized customer-month features. | Reconstruction error added a different pattern-level signal, but it lacked a reliable business driver by itself. | Corroboration only; not the primary queue. |
| Expected-order-miss MLP | An 83-input, two-hidden-layer neural-network classifier evaluated in six chronological rebuilds. | It was competitive, but the simpler Stage A benchmark had better overall probability quality. | Retained as a challenger; not promoted. |
| History and feature experiments | Full-history training, long-window features, seasonality, value features, optimizer tests, and classical-model comparisons. | More eligible training rows helped; adding more long-history/value features generally did not. | Keep the bounded full-history challenger isolated pending new future evidence. |
| Probabilistic order challenger | Separate occurrence, count, quantity, and product-mix expectation axes with calibrated tail scores. | It surfaced different cases, but the frozen hurdle model remained the stronger occurrence forecast and the candidate queue was less stable. | Research lens only; no control replacement. |
| Blinded LLM technical review | Privacy-filtered anonymous evidence packets, strict structured labels, append-only votes, and a fixed three-model panel. | The pipeline produced reproducible technical weak labels and useful disagreement diagnostics. | Diagnostic evidence only; not GelatiCo ground truth. |
| Anomaly radar and consensus | Quality gates and independently collapsed signal families feeding conservative corroborated and discovery lanes. | Multiple methods can be combined without counting correlated signals as independent votes. | Candidate prioritization workflow; business quality still requires review. |
| GelatiCo review delivery | Private identifier resolution, business-context enrichment, and matched HTML, Excel, and CSV review materials. | Technical candidates can be converted into a finite, traceable stakeholder workload. | Operational handoff demonstrated; outcome validation remains open. |

## 1. Neural-network experiments

### Autoencoder: a pattern-level corroborator

The autoencoder was an unsupervised detector. It learned to reconstruct a standardized
customer-month profile through a compact three-unit bottleneck. Cases with high reconstruction
error had combinations of behavior that the network found difficult to reproduce from recurring
historical patterns.

This is useful when no single field is extreme but the overall combination is unusual. It also
creates an interpretation problem: reconstruction error does not naturally provide the same clear
driver as an explainable deviation score. For that reason, the autoencoder was treated as
corroborating evidence rather than an authoritative anomaly score. Reviewer-facing explanations
describe the observed pattern, not the prestige of a “deep neural network.”

### MLP: a supervised expected-order-miss challenger

A separate multilayer perceptron predicted whether a customer judged technically due to order
would miss the expected next-month order. The selected architecture was:

```text
83 inputs -> 16 hidden units -> 8 hidden units -> 1 probability
```

It was rebuilt in six chronological folds across 22,546 due-order observations, 3,727 customers,
and 21 evaluation months. The comparison used fold-local preprocessing and calibration gates plus
customer-clustered uncertainty.

The neural network achieved log loss 0.4839, while the simpler Stage A benchmark achieved 0.4765;
lower is better. The MLP-minus-Stage-A difference was +0.0074 with a 95% customer-clustered
interval of [+0.0049, +0.0099]. Stage A therefore remained the stronger overall technical model.
The MLP was retained as research—particularly for limited-history customers—but it did not replace
the benchmark or the submitted hurdle/cadence workflow.

## 2. What the history experiments taught

The project tested whether improvement would come from more history, more features, or more model
complexity. The useful result was narrower than “more data is always better”:

- training the 83-feature MLP on every eligible historical row improved mean rolling-fold log loss
  by about 0.004 and passed its preregistered gate;
- adding longer-window, t-24/t-36 seasonal, expanding-history, or value-history feature families
  generally worsened or failed to improve performance;
- a bounded optimizer search did not beat the original optimizer reliably; and
- logistic regression and histogram gradient boosting were credible comparators but did not pass
  the preregistered five-fold promotion gate against the full-history MLP.

The practical lesson was to prefer disciplined temporal validation and a compact, defensible
feature contract over indiscriminate feature accumulation. Even the passing full-history treatment
remained isolated because the later January--June 2026 period had already been inspected. Promotion
requires genuinely future evidence.

A further prototype redefined an expected miss as no order in either of the following two months.
The target population was constructed and validated, but this sustained-miss idea remained a target
contract and research foundation rather than a promoted model result.

## 3. Probabilistic order-anomaly research

Another challenger modeled several parts of ordering behavior separately: whether an order occurs,
how many orders occur when active, quantity behavior, and product-mix composition. This made it
possible to describe low, high, or compositionally different outcomes relative to an as-of
expectation instead of forcing every signal into one score.

In the matched January--June 2026 occurrence comparison, the probabilistic challenger had log loss
0.3039 versus 0.2788 for the frozen hurdle model. It also produced a materially different queue but
showed lower consecutive-month stability. The experiment therefore contributed another discovery
lens, not evidence to replace the hurdle/cadence model. Quantity and mix conclusions also remained
subject to their own data-semantic and multiplicity gates.

## 4. Blinded LLM technical review

The LLM work tested whether independent language models, shown only a privacy-filtered behavioral
packet, could reproducibly describe a visible change in ordering evidence. Packets excluded customer
identity, detector name, score, rank, sampling stratum, absolute commercial value, and suggested
action. Responses were constrained to a versioned schema with four technical outcomes:

- `visible_change`;
- `no_clear_change`;
- `insufficient_evidence`; and
- `data_quality_issue`.

The executed three-model panel reviewed 250 blinded cases and produced 749 of 750 valid logical
votes. Of the 250 cases, 141 had unanimous 3/3 consensus, 102 had 2/3 consensus, six had no
majority, and one had insufficient votes. A separately frozen 50-case Codex review was used only as
an AI-to-panel robustness check.

This demonstrated a privacy-gated, resumable, append-only technical-review system and exposed where
model judgments disagreed. It did not create business labels. Agreement among models that inspect
similar evidence is construct alignment, not proof that a case is new, actionable, commercially
important, or worth intervention.

## 5. From many signals to a conservative radar

The project also built a capacity-aware anomaly radar and a cross-track consensus experiment.
Instead of rewarding the raw number of agreeing methods, correlated methods were collapsed into
independent signal families before votes were counted. Data-quality, history-depth, and peer-support
gates prevented weak-data cases from being promoted as high-confidence corroboration.

The workflow retained two distinct lanes:

- a corroborated lane requiring support from more than one independent family; and
- a smaller discovery lane for an exceptionally strong single-family signal.

Confidence described the sufficiency and corroboration of the measurement, not the probability
that the customer represented a genuine commercial problem. Explanations used existing evidence
and did not invent a driver for pattern-only autoencoder cases.

This work showed how multiple frozen signals could be combined conservatively, but it still could
not estimate anomaly accuracy or false-positive rate without GelatiCo outcomes.

## 6. Operational business-review package

The final extension moved beyond notebooks. Inside the authorized private environment, shortlisted
cases were joined to the references and business context needed by GelatiCo reviewers. The build
produced matched HTML, Excel, and CSV materials with source digests, selection rules, and output
checksums.

The documented handoff contained 160 unique June 2026 cases across two prioritized cohorts. It
demonstrated that technical signals could become a finite, reviewer-friendly, reproducible work
package while private identifiers and absolute commercial values stayed outside the public
repository.

The 160-case handoff was prioritized rather than a statistically representative evaluation sample.
Until reviewer outcomes are returned, it cannot establish full-queue precision, comparative method
quality, actionability, or ROI. See [Stakeholder Validation Handoff](stakeholder_validation_handoff.md)
for the public-safe operating description.

## What these extensions changed—and did not change

They changed the breadth of the research:

- neural networks were tested rather than assumed to be better;
- model disagreement became inspectable;
- probability, deviation, reconstruction, and consensus signals were kept conceptually separate;
- privacy and authorization were enforced in the LLM workflow;
- multiple signal families could be collapsed without inflating agreement; and
- a traceable stakeholder package was produced.

They did not change the final deliverable's central conclusion. The defensible next step remains a
blinded, capacity-bounded GelatiCo review. Technical sophistication cannot substitute for business
judgment about novelty, usefulness, actionability, operating fit, or financial value.

## Promotion gates for future work

An experimental component should influence the operating workflow only after it has:

1. a frozen target, population, feature, and privacy contract;
2. leakage-safe chronological evidence against an incumbent comparator;
3. uncertainty and capacity-based diagnostics;
4. reproducible artifacts with immutable run provenance;
5. genuinely future or otherwise untouched evaluation evidence; and
6. blinded GelatiCo outcomes demonstrating practical usefulness at an agreed review capacity.

Until those gates are met, the extensions remain research and decision support—not automated
commercial decisioning.
