"""Synchronize README/report evidence from the generated evaluation sidecar."""

from __future__ import annotations

import json
import os
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CACHE = Path(
    os.environ.get("CAPSTONE_MODULE24_CACHE_DIR", ROOT / ".cache" / "module24")
).resolve()
EVALUATION = OUTPUT_CACHE / "module24_evaluation.json"
START = "<!-- METRICS_START -->"
END = "<!-- METRICS_END -->"


# Shared with notebook 07 so the report and the pilot chart name each stratum identically.
STRATUM_LABELS = {
    "channel_only": "S2 only",
    "hurdle_only": "cadence only",
    "background": "background sample",
    "agreement_core": "both lenses agree",
    "peer_definition_disagreement": "peer-definition disagreement",
    "deterministic_fill": "deterministic capacity fill",
}


def _metric(records: list[dict], model: str) -> dict:
    return next(row for row in records if row["model"] == model)


def _strata_phrase(pilot_strata: dict[str, int]) -> str:
    """Describe only the strata that actually received cases.

    A stratum can be defined but draw zero cases; naming it anyway would leave the report
    describing a sample the tables do not show.
    """
    populated = [
        f"{STRATUM_LABELS.get(name, name)} ({count})"
        for name, count in pilot_strata.items()
        if count
    ]
    phrase = ", ".join(populated[:-1]) + f", and {populated[-1]}"
    unused = [
        STRATUM_LABELS.get(name, name)
        for name in STRATUM_LABELS
        if not pilot_strata.get(name)
    ]
    if unused:
        phrase += f". The {', '.join(unused)} stratum drew no cases at this capacity"
    return phrase


def _capacity(records: list[dict], capacity: int, comparison: str | None = None) -> dict:
    for row in records:
        if row["capacity_per_month"] != capacity:
            continue
        if comparison is None or row.get("comparison") == comparison:
            return row
    raise KeyError((capacity, comparison))


def render_readme(evidence: dict) -> str:
    workflow = evidence["workflow"]
    activity = workflow["activity_forward"]
    pattern = workflow["active_pattern_forward"]
    forward = evidence["peer_strategy"]["strategy_metrics"]
    channel = next(row for row in forward if row["strategy"] == "channel")
    company = next(row for row in forward if row["strategy"] == "company_division")
    peer_250 = _capacity(
        evidence["peer_strategy"]["capacity_overlap"],
        250,
        "channel_vs_company_division",
    )
    lens_250 = _capacity(
        evidence["operational_layer"]["capacity_overlap"], 250
    )
    operational = evidence["operational_layer"]
    return (
        f"- **Modeling population:** {evidence['data_package']['customer_month_rows']:,} "
        f"customer-months across {evidence['data_package']['customers']:,} customers; "
        f"modeling ends {evidence['data_package']['model_end_month']}.\n"
        f"- **Activity model:** `{workflow['activity_model']}`; forward log loss "
        f"{activity['log_loss']:.4f}, ROC-AUC {activity['roc_auc']:.4f}, average precision "
        f"{activity['average_precision']:.4f}, F1 {activity['f1']:.4f}.\n"
        f"- **Active-pattern stage:** accuracy {pattern['accuracy']:.4f} versus majority "
        f"{pattern['majority_accuracy']:.4f}; macro-F1 {pattern['macro_f1']:.4f}.\n"
        f"- **Peer A/B at 250 cases/month:** channel/company-division Jaccard "
        f"{peer_250['pooled_jaccard']:.4f}; company/division sparse-cell rate "
        f"{company['sparse_peer_rate']:.2%} versus channel {channel['sparse_peer_rate']:.2%}.\n"
        f"- **Lens complementarity at 250 cases/month:** S2/hurdle Jaccard "
        f"{lens_250['channel_vs_hurdle_jaccard']:.4f} "
        f"({lens_250['overlap_rows']:,} shared rows across six months).\n"
        f"- **Operationalization:** {operational['pilot_rows']} blinded pilot cases; "
        f"monthly flags collapsed into alert episodes before review."
    )


def render_report(evidence: dict) -> str:
    workflow = evidence["workflow"]
    activity = workflow["activity_forward"]
    pattern = workflow["active_pattern_forward"]
    peer = evidence["peer_strategy"]
    operational = evidence["operational_layer"]
    peer_table = []
    for capacity in (100, 250, 500):
        company = _capacity(
            peer["capacity_overlap"], capacity, "channel_vs_company_division"
        )
        hybrid = _capacity(peer["capacity_overlap"], capacity, "channel_vs_hybrid")
        peer_table.append(
            f"| {capacity} | {company['pooled_jaccard']:.4f} | "
            f"{hybrid['pooled_jaccard']:.4f} |"
        )
    lens_table = []
    for capacity in (100, 250, 500):
        row = _capacity(operational["capacity_overlap"], capacity)
        lens_table.append(
            f"| {capacity} | {row['channel_vs_hurdle_jaccard']:.4f} | "
            f"{row['overlap_rows']:,} |"
        )
    hurdle_metrics_path = OUTPUT_CACHE / "06_hurdle_metrics.json"
    forward_records = json.loads(hurdle_metrics_path.read_text())["forward_metrics"]
    logistic = _metric(forward_records, "logistic")
    markov = _metric(forward_records, "smoothed_activity_markov")
    global_rate = _metric(forward_records, "global_activity_rate")
    uncertainty = json.loads(hurdle_metrics_path.read_text())["forward_uncertainty"]
    uncertainty_rows = []
    for comparison in uncertainty:
        log_loss_ci = comparison["selected_minus_baseline"]["log_loss"]["ci_95"]
        roc_auc_ci = comparison["selected_minus_baseline"]["roc_auc"]["ci_95"]
        uncertainty_rows.append(
            f"| Selected minus {comparison['baseline']} | "
            f"{comparison['selected_minus_baseline']['log_loss']['point_difference']:.4f} "
            f"[{log_loss_ci[0]:.4f}, {log_loss_ci[1]:.4f}] | "
            f"{comparison['selected_minus_baseline']['roc_auc']['point_difference']:.4f} "
            f"[{roc_auc_ci[0]:.4f}, {roc_auc_ci[1]:.4f}] |"
        )
    return f"""## Generated results

The modeling panel contains **{evidence['data_package']['customer_month_rows']:,} customer-months**
and **{evidence['data_package']['customers']:,} customers**, ending in
**{evidence['data_package']['model_end_month']}**.

### Hurdle/cadence forecast

The selected operational activity model is **`{workflow['activity_model']}`**. The table reports the
two candidate classifiers on the same forward window alongside the two disclosed same-target
baselines; log loss is the primary selection metric because the workflow needs usable probabilities,
not only a yes/no label.

| Model | Log loss | ROC-AUC | Average precision | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Selected activity model (histogram gradient boosting) | {activity['log_loss']:.4f} | {activity['roc_auc']:.4f} | {activity['average_precision']:.4f} | {activity['f1']:.4f} | {activity['balanced_accuracy']:.4f} |
| Logistic regression candidate | {logistic['log_loss']:.4f} | {logistic['roc_auc']:.4f} | {logistic['average_precision']:.4f} | {logistic['f1']:.4f} | {logistic['balanced_accuracy']:.4f} |
| Smoothed activity Markov | {markov['log_loss']:.4f} | {markov['roc_auc']:.4f} | {markov['average_precision']:.4f} | {markov['f1']:.4f} | {markov['balanced_accuracy']:.4f} |
| Global activity rate | {global_rate['log_loss']:.4f} | {global_rate['roc_auc']:.4f} | {global_rate['average_precision']:.4f} | {global_rate['f1']:.4f} | {global_rate['balanced_accuracy']:.4f} |

Conditional active-pattern accuracy is **{pattern['accuracy']:.4f}** versus a
**{pattern['majority_accuracy']:.4f}** majority baseline; macro-F1 is
**{pattern['macro_f1']:.4f}**. These metrics evaluate self-supervised next-month behavior.
Commercial anomaly correctness is evaluated separately.

### Forward uncertainty

Customer-clustered nonparametric bootstrap on the fixed forward holdout ({uncertainty[0]['resamples']:,}
resamples; seed {uncertainty[0]['random_state']}) keeps every sampled customer's six forward
months together. Negative log-loss differences and positive ROC-AUC differences favor the selected
model.

| Comparison | Log-loss difference, 95% CI | ROC-AUC difference, 95% CI |
|---|---:|---:|
{chr(10).join(uncertainty_rows)}

These intervals quantify sampling uncertainty for this historical forward window. Anomaly
correctness, commercial usefulness, and future financial value require commercial-review evidence.

### Peer strategy at fixed monthly capacity

| Cases/month | Channel vs company/division Jaccard | Channel vs hybrid Jaccard |
|---:|---:|---:|
{chr(10).join(peer_table)}

The company/division peer cell is usable and non-sparse for
**{peer['company_division_usable_non_sparse_peer_rate']:.2%}** of forward rows. Its changed cases
must be reviewed before replacing channel.

### S2 versus hurdle queue

| Cases/month | Jaccard | Shared customer-months across six months |
|---:|---:|---:|
{chr(10).join(lens_table)}

At 250 cases/month the queues overlap at only
**{_capacity(operational['capacity_overlap'], 250)['channel_vs_hurdle_jaccard']:.4f}**. This provides
evidence of complementarity and supports sampling both tails. Tail usefulness requires reviewer
outcomes.

### Alert episodes and pilot

{textwrap.fill(
    "The episode layer removes repeated monthly work items before handoff. The blinded pilot "
    f"contains **{operational['pilot_rows']} cases** across "
    f"{_strata_phrase(operational['pilot_strata'])}. Inclusion probabilities are retained in the "
    "private allocation key.",
    width=98,
    break_on_hyphens=False,
)}

The activity forecast answers the operational question of next-month ordering behavior; its
metrics are interpreted only for that self-supervised prediction target.
"""


def render_evaluation(evidence: dict) -> str:
    """Regenerate evaluation.md's metric block from the same sidecar the report uses.

    evaluation.md previously restated these numbers by hand, so a re-run could leave the two
    documents disagreeing. Both are now generated from module24_evaluation.json.
    """
    activity = evidence["workflow"]["activity_forward"]
    lens_250 = _capacity(evidence["operational_layer"]["capacity_overlap"], 250)
    hurdle_metrics = json.loads(
        (OUTPUT_CACHE / "06_hurdle_metrics.json").read_text()
    )
    forward_records = hurdle_metrics["forward_metrics"]
    logistic = _metric(forward_records, "logistic")
    markov = _metric(forward_records, "smoothed_activity_markov")
    global_rate = _metric(forward_records, "global_activity_rate")
    markov_uncertainty = next(
        row
        for row in hurdle_metrics["forward_uncertainty"]
        if row["baseline"] == "smoothed_activity_markov"
    )
    log_loss_difference = markov_uncertainty["selected_minus_baseline"]["log_loss"]
    roc_auc_difference = markov_uncertainty["selected_minus_baseline"]["roc_auc"]
    return f"""The selected activity model is evaluated on an untouched January--June 2026 forward window.
It predicts self-supervised next-month ordering activity. Commercial anomaly correctness is
evaluated separately.

| Model | Log loss | ROC-AUC | Average precision | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Selected activity model (histogram gradient boosting) | {activity['log_loss']:.4f} | {activity['roc_auc']:.4f} | {activity['average_precision']:.4f} | {activity['f1']:.4f} | {activity['balanced_accuracy']:.4f} |
| Logistic regression candidate | {logistic['log_loss']:.4f} | {logistic['roc_auc']:.4f} | {logistic['average_precision']:.4f} | {logistic['f1']:.4f} | {logistic['balanced_accuracy']:.4f} |
| Smoothed activity Markov | {markov['log_loss']:.4f} | {markov['roc_auc']:.4f} | {markov['average_precision']:.4f} | {markov['f1']:.4f} | {markov['balanced_accuracy']:.4f} |
| Global activity rate | {global_rate['log_loss']:.4f} | {global_rate['roc_auc']:.4f} | {global_rate['average_precision']:.4f} | {global_rate['f1']:.4f} | {global_rate['balanced_accuracy']:.4f} |

{textwrap.fill(
    f"Customer-clustered bootstrap ({markov_uncertainty['resamples']:,} resamples, seed "
    f"{markov_uncertainty['random_state']}) gives selected-minus-Markov log-loss difference "
    f"{log_loss_difference['point_difference']:.4f} "
    f"(95% CI [{log_loss_difference['ci_95'][0]:.4f}, {log_loss_difference['ci_95'][1]:.4f}]) "
    f"and ROC-AUC difference {roc_auc_difference['point_difference']:+.4f} "
    f"(95% CI [{roc_auc_difference['ci_95'][0]:.4f}, {roc_auc_difference['ci_95'][1]:.4f}]). "
    "These intervals quantify sampling uncertainty within the fixed forward window. Business "
    "value is evaluated in the commercial-review stage.",
    width=98,
    break_on_hyphens=False,
)}

{textwrap.fill(
    "At 250 cases/month, channel S2 versus hurdle queue overlap is "
    f"{lens_250['channel_vs_hurdle_jaccard']:.4f} Jaccard "
    f"({lens_250['overlap_rows']:,} shared customer-months). This supports a mixed blinded "
    "pilot. Queue usefulness requires reviewer outcomes.",
    width=98,
    break_on_hyphens=False,
)}
"""


def replace_block(path: Path, content: str) -> None:
    """Replace only the marked generated block, preserving the explanatory prose around it.

    The report's narrative is hand-written; numerical evidence is regenerated from the sidecar.
    Keeping those responsibilities separate prevents a stale manual copy of a metric.
    """
    text = path.read_text()
    pattern = re.compile(
        re.escape(START) + r".*?" + re.escape(END),
        flags=re.DOTALL,
    )
    replacement = f"{START}\n{content.rstrip()}\n{END}"
    new_text, count = pattern.subn(replacement, text)
    if count != 1:
        raise ValueError(f"Expected exactly one metrics block in {path}; found {count}")
    path.write_text(new_text)


def main() -> None:
    evidence = json.loads(EVALUATION.read_text())
    replace_block(ROOT / "report.md", render_report(evidence))
    replace_block(ROOT / "evaluation.md", render_evaluation(evidence))
    print("Synchronized report.md and evaluation.md from module24_evaluation.json")


if __name__ == "__main__":
    main()
