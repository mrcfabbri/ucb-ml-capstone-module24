# Customer Purchase Behavior Prioritization — Module 24 Final Deliverable

This handout evaluates a capacity-bounded customer investigation queue for a confidential industrial partner. It presents two complementary lenses: a business-friendly S2 deviation score and a hurdle/cadence model for next-month ordering behavior. Both create investigation candidates. Confirmed anomaly classification and automated commercial decisions are outside the project scope.

**The submitted notebooks are pre-executed and contain all aggregate outputs required for grading. Private source data are intentionally excluded and are not required to review the analysis.**

Notebooks 01--04 contain the earlier data-understanding, EDA, feature-engineering, and regression-baseline work published in the separate [Module 20 repository](https://github.com/mrcfabbri/ucb-ml-capstone-module20). This final Module 24 repository continues the sequence at notebook 05 and contains all evidence needed for this submission.

## Submission overview

The hurdle/cadence model forecasts next-month ordering activity more
accurately than the two simple historical baselines. Its investigation queue and the business-friendly
S2 deviation queue overlap very little. That is enough evidence to run a mixed, blinded review;
confirmed anomaly status and financial value require separate commercial evaluation.

Recommended reading sequence:

1. This README for the decision and the scope.
2. [99 — Final findings](notebooks/99_final_findings.ipynb) for the executive dashboard.
3. [report.md](report.md) for a concise, rubric-facing account of the methods and results.
4. Notebooks 05, 06, and 07 for the detailed evidence behind the peer score, forecast, and pilot.

### Reference

- **S2 deviation score:** a business-friendly measure of how unusual a customer-month looks versus
  that customer's history and comparable customers. It prioritizes cases for investigation.
- **Hurdle/cadence model:** a two-step forecast: first, whether the customer will order next
  month; then, what the active-month pattern may look like if it orders.
- **Forward holdout:** January--June 2026, held back from model selection to provide an
  independent historical test.
- **Jaccard overlap:** the share of cases selected by both queues. Low overlap indicates that the
  queues surface different candidates. Commercial usefulness is evaluated separately.

## Summary of findings

- The selected histogram gradient boosting classifier improves next-month activity forecasting over
  both disclosed same-target baselines on the untouched January--June 2026 holdout: log loss is
  **0.2765**, versus **0.3767** for the smoothed Markov baseline and **0.4393** for the global-rate
  baseline.
- At 250 cases per month, channel and company/division S2 queues have **0.8450 Jaccard overlap**;
  the narrower peer definition is technically usable but still requires commercial review before
  adoption.
- Channel S2 and cadence queues have only **0.0858 Jaccard overlap** at the same capacity. They are
  complementary investigation lenses, so the recommended next step is a mixed, blinded 250-case
  review rather than automated action.
- These are technical prioritization results. Anomaly correctness, novelty, actionability, adoption,
  ROI, and financial impact remain outside the evidence until the documented business review is
  completed.

## Scope of conclusions

| Question | Supported conclusion | Outside the current evaluation |
|---|---|---|
| Can the model forecast next-month ordering activity? | The selected model improves the stated forecasting metrics on an untouched future period. | Revenue, churn, and confirmed commercial anomaly classification. |
| Do the two queues add different candidates? | Their low fixed-capacity overlap supports a mixed pilot. | Commercial usefulness, which requires reviewer outcomes. |
| Should the process be deployed automatically? | No. The recommended action is a blinded, capacity-bounded human review. | No ROI, adoption, or financial-impact claim is made. |

## Included evidence

| Artifact | Purpose |
|---|---|
| [report.md](report.md) | Final analytical narrative, metrics, uncertainty, findings, and gates |
| [05 — Peer strategy baseline](notebooks/05_peer_strategy_baseline.ipynb) | Data-quality audit, S2 construction, and peer-definition comparison |
| [06 — Hurdle/cadence model](notebooks/06_hurdle_cadence_model.ipynb) | Chronological model selection, forward metrics, and clustered uncertainty |
| [07 — Alert episodes and pilot](notebooks/07_alert_episodes_and_pilot.ipynb) | Episode construction and blinded 250-case review sample |
| [99 — Final findings](notebooks/99_final_findings.ipynb) | Decision dashboard, findings, recommendations, and next steps |
| [evaluation.md](evaluation.md) | Concise evaluation reference |
| [docs/business_value_validation.md](docs/business_value_validation.md) | Commercial-validation protocol |
| [docs/reader_guide.md](docs/reader_guide.md) | Plain-language guide to the notebook sequence, metrics, and boundaries |
| [docs/stakeholder_validation_handoff.md](docs/stakeholder_validation_handoff.md) | Public-safe operational extension from technical candidates to a private stakeholder-review package |

## Optional operational extension

After the core notebook analysis, the project converted separately frozen analytical signals into a
privacy-controlled stakeholder-review workflow. The extension demonstrates independent-family
agreement controls, data-quality gates, aggregate business prioritization, private identifier
resolution, and reproducible HTML/Excel/CSV delivery. It is documented separately because it did
not change the classifier selection or add stakeholder outcome evidence to the graded analysis.

See [Stakeholder Validation Handoff](docs/stakeholder_validation_handoff.md) for the public-safe
workflow, aggregate package counts, and evidence boundary.

## Data and preparation

The analysis uses a confidential customer-month panel built from the partner's ERP order-line and
customer-master records. One row represents one customer and calendar month. The authorized local
checkpoints contain the panel and prepared scoring inputs; raw records, customer identifiers, and
commercial values are excluded from this handout.

Before scoring, notebook 05 parses month keys, removes invalid keys and duplicate customer-months,
normalizes flags, coerces numeric fields, sorts deterministically, and reports aggregate grain,
coverage, missingness, activity rate, and score completeness. The cadence model uses only
as-of behavioral features: recent activity and order-frequency summaries, inactivity duration,
year-over-year availability, non-monetary rates, tenure, and calendar phase. Monetary values,
current lifecycle fields, open/delivered/invoiced fields, and peer assignments are excluded from
prediction.

## Reproduction and privacy boundary

Executed notebooks are included with aggregate, privacy-checked outputs and can be graded without
private inputs. An authorized local rerun requires three confidential checkpoints that are not
included in this handout; the notebooks neither read raw exports nor publish checkpoint contents.
Place the authorized files in `.private_cache/` at the repository root:

- `03_engineered_customer_month.parquet`
- `05b_scored_customer_month.parquet`
- `06_next_state_customer_month.parquet`

Generated sidecars remain in the ignored local directory `.cache/module24/`. The checkpoint and
generated-output locations can also be overridden with `CAPSTONE_UPSTREAM_CACHE_DIR` and
`CAPSTONE_MODULE24_CACHE_DIR`.

The repository includes its own `pyproject.toml` and `uv.lock`. The Makefile creates a project-local
Jupyter kernel so notebook execution cannot silently fall back to another Python environment:

```bash
make test          # unit tests; private-sidecar checks skip when outputs are unavailable
make public-check  # lint, privacy, saved plots, links, and hashes; no private data
make validate      # adds generated-sidecar tests and refreshes provenance
make full-check    # authorized rerun of all notebooks, validation, and packaging
```

## Provenance and packaging

[provenance_manifest.json](provenance_manifest.json) contains only hashes, sizes, and availability
metadata for the required private checkpoints, plus hashes of every public file and the local
lockfile. It
contains no checkpoint contents, identifiers, or commercial values. Generate it after final
rerun/validation work and immediately before staging the final deliverable commit.

`make package` writes `dist/module24_deliverable.zip`. Its strict allowlist excludes private cache
contents, `.env`, `.csv`, `.parquet`, `__pycache__`, bytecode, `.DS_Store`, and unsupported files.
The package gate decodes all 17 embedded plots, verifies sequential execution counts, scans the
archive for private artifacts, confirms every manifest hash, and compares every ZIP member with its
canonical source file.

## Claim boundary

The evidence covers temporal predictive performance, queue sensitivity, and complementarity.
Anomaly correctness, novelty, actionability, adoption, ROI, and financial impact require the
blinded commercial review documented in
[docs/business_value_validation.md](docs/business_value_validation.md).

## Next steps

1. Complete the blinded commercial review and measure novelty, actionability, and review effort.
2. Compare the two investigation lenses using the retained sampling weights and reviewer outcomes.
3. Consider adoption or automation only if the commercial evidence meets an agreed decision gate.

## Contact

**Marco Fabbri**

[GitHub profile](https://github.com/mrcfabbri) ·
[128336792+mrcfabbri@users.noreply.github.com](mailto:128336792+mrcfabbri@users.noreply.github.com)
