from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from module24_workflow import (  # noqa: E402
    CADENCE_FEATURES,
    FORBIDDEN_PREDICTIVE_COLUMNS,
    OUTPUT_CACHE,
    PEER_INPUT_COLUMNS,
    _activity_model_pipelines,
    _customer_clustered_bootstrap,
    _deterministic_top_k,
    _episode_table,
    _expanding_window_folds,
    _fold_boundary_rows,
    _monthly_activity_rate_rows,
    _peer_data_quality,
    _prepare_activity_models,
    _reliability_rows,
    _resolve_s2_score_config,
    _standardized_logistic_coefficients,
    _waterfill_top_k,
    build_cadence_panel,
    run_peer_strategy_evaluation,
)


def _require_generated_artifact(path: Path, *, notebook: str) -> Path:
    """Skip private-rerun assertions when a public clone has no generated sidecar."""

    if not path.exists():
        pytest.skip(
            f"{path.name} is available only after an authorized rerun of {notebook}"
        )
    return path


def _synthetic_input() -> pd.DataFrame:
    rows = []
    months = pd.date_range("2024-01-01", periods=15, freq="MS")
    for customer_number in range(2):
        for month_number, month in enumerate(months):
            order_count = int((month_number + customer_number) % 3 == 0)
            rows.append(
                {
                    "customer_public_id": f"synthetic-{customer_number}",
                    "transaction_month": month,
                    "first_observed_month": months[0],
                    "order_count": order_count,
                    "active_order_days": order_count,
                    "distinct_product_count": order_count * 2,
                    "discount_rate": 0.1 * order_count,
                    "promo_line_share": 0.2 * order_count,
                    "return_line_rate": 0.0,
                    "top_product_revenue_share": 0.5 * order_count,
                }
            )
    return pd.DataFrame(rows)


def test_cadence_panel_is_t_to_t_plus_one(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.parquet"
    _synthetic_input().to_parquet(path, index=False)
    panel = build_cadence_panel(path)

    assert set(CADENCE_FEATURES).isdisjoint(FORBIDDEN_PREDICTIVE_COLUMNS)
    assert (
        panel["target_month"]
        == panel["transaction_month"] + pd.offsets.MonthBegin(1)
    ).all()
    assert panel.groupby("customer_public_id").size().eq(14).all()


def test_report_data_acquisition_figure_is_packaged() -> None:
    figure = ROOT / "figures" / "monthly_activity_rate.png"

    assert figure.is_file()
    assert figure.stat().st_size > 0


def test_provenance_manifest_has_portable_final_package_scope() -> None:
    from generate_provenance_manifest import build_manifest

    manifest = build_manifest()
    package = manifest["package"]

    assert manifest["schema_version"] == 2
    assert package["path"] == "."
    assert package["manifest_scope"] == (
        "all curated public files except this manifest and the ZIP"
    )
    assert manifest["environment"]["uv_project"] == "."
    assert manifest["environment"]["lockfile"]["path"] == "uv.lock"


def test_logistic_interpretation_is_ranked_and_complete() -> None:
    rows = 40
    X = pd.DataFrame(
        {
            feature: np.linspace(0, 1, rows) + feature_number / 100
            for feature_number, feature in enumerate(CADENCE_FEATURES)
        }
    )
    y = pd.Series(([0, 1] * (rows // 2)), name="next_has_order")
    model = _activity_model_pipelines()["logistic"].fit(X, y)

    coefficients = _standardized_logistic_coefficients(model)

    assert len(coefficients) == len(CADENCE_FEATURES)
    assert {row["feature"] for row in coefficients} == set(CADENCE_FEATURES)
    assert all(row["odds_ratio_per_sd"] > 0 for row in coefficients)
    assert [row["absolute_coefficient"] for row in coefficients] == sorted(
        (row["absolute_coefficient"] for row in coefficients), reverse=True
    )


def test_caller_activity_pipelines_are_cloned_and_receive_grid_winner() -> None:
    supplied = _activity_model_pipelines()

    prepared = _prepare_activity_models(
        supplied,
        histogram_params={
            "learning_rate": 0.15,
            "max_leaf_nodes": 7,
            "l2_regularization": 2.0,
        },
    )

    assert prepared["logistic"] is not supplied["logistic"]
    assert prepared["hist_gradient_boosting"] is not supplied["hist_gradient_boosting"]
    assert prepared["hist_gradient_boosting"].get_params()["model__learning_rate"] == 0.15
    assert prepared["hist_gradient_boosting"].get_params()["model__max_leaf_nodes"] == 7


def test_visible_s2_score_configuration_is_validated() -> None:
    config = _resolve_s2_score_config(
        {
            "aggregation": "maximum_absolute_signal",
            "zero_order_bonus": 0.5,
            "sparse_history_multiplier": 0.8,
        }
    )

    assert config["zero_order_bonus"] == 0.5
    assert config["sparse_history_multiplier"] == 0.8

    with pytest.raises(ValueError, match="aggregation"):
        _resolve_s2_score_config({"aggregation": "mean"})


def test_peer_evaluation_accepts_only_complete_prepared_input() -> None:
    incomplete = pd.DataFrame(columns=PEER_INPUT_COLUMNS[:-1])

    with pytest.raises(ValueError, match="missing required columns"):
        run_peer_strategy_evaluation(prepared_frame=incomplete)


def test_notebook_05_passed_its_cleaned_frame_to_scoring() -> None:
    path = _require_generated_artifact(
        OUTPUT_CACHE / "05_peer_strategy_metrics.json",
        notebook="notebook 05",
    )

    evidence = json.loads(path.read_text())
    assert evidence["peer_input_source"] == "caller_prepared_frame"
    assert evidence["peer_input_rows"] > 0


def test_chronological_fold_evidence_has_expanding_train_boundaries() -> None:
    months = pd.date_range("2025-01-01", periods=6, freq="MS")
    panel = pd.DataFrame(
        {
            "target_month": np.repeat(months, 2),
            "data_split": "train",
        }
    )
    train = panel["data_split"].eq("train")

    boundaries = _fold_boundary_rows(
        panel, train, _expanding_window_folds(panel, train, n_folds=2)
    )

    assert len(boundaries) == 2
    assert [row["train_rows"] for row in boundaries] == [4, 8]
    assert [row["validation_rows"] for row in boundaries] == [4, 4]
    assert all(row["train_end"] < row["validation_start"] for row in boundaries)
    assert boundaries[1]["train_start"] == boundaries[0]["train_start"]


def test_reliability_evidence_is_aggregate_only_and_counts_all_rows() -> None:
    rows = _reliability_rows(
        pd.Series([0, 0, 1, 1]),
        np.array([0.01, 0.24, 0.51, 1.0]),
        model="test_raw",
        n_bins=4,
    )

    assert sum(row["rows"] for row in rows) == 4
    assert {row["model"] for row in rows} == {"test_raw"}
    assert all(0 <= row["observed_activity_rate"] <= 1 for row in rows)
    assert all(0 <= row["mean_predicted_probability"] <= 1 for row in rows)
    assert all("customer" not in row for row in rows)


def test_customer_clustered_bootstrap_is_deterministic_and_aggregate_only() -> None:
    customer_ids = pd.Series(["private-a", "private-a", "private-b", "private-b"])
    y_true = pd.Series([0, 1, 0, 1])
    selected = np.array([0.1, 0.9, 0.2, 0.8])
    baseline = np.array([0.6, 0.4, 0.7, 0.3])

    first = _customer_clustered_bootstrap(
        customer_ids, y_true, selected, baseline, baseline="baseline", n_resamples=25
    )
    second = _customer_clustered_bootstrap(
        customer_ids, y_true, selected, baseline, baseline="baseline", n_resamples=25
    )

    assert first == second
    assert first["forward_customers"] == 2
    assert first["selected_minus_baseline"]["log_loss"]["point_difference"] < 0
    assert first["selected_minus_baseline"]["roc_auc"]["point_difference"] > 0
    assert "private-a" not in json.dumps(first)


def test_curated_package_excludes_private_and_editor_residue(tmp_path: Path) -> None:
    from package_submission import is_allowed_directory_file, package_files

    archive_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in package_files():
            archive.write(path, path.relative_to(ROOT))
    names = zipfile.ZipFile(archive_path).namelist()

    assert "README.md" in names
    assert "provenance_manifest.json" in names
    assert ".gitignore" in names
    assert "pyproject.toml" in names
    assert "uv.lock" in names
    assert not any(".cache" in name or "__pycache__" in name for name in names)
    assert not any(name.endswith((".pyc", ".DS_Store")) for name in names)
    assert is_allowed_directory_file(ROOT / "docs" / "public.md")
    assert not is_allowed_directory_file(ROOT / "docs" / "private.csv")
    assert not is_allowed_directory_file(ROOT / "src" / ".env")
    assert not is_allowed_directory_file(ROOT / "figures" / "private.parquet")


def test_privacy_scan_checks_notebook_sources_and_all_public_text() -> None:
    from check_privacy import scan_portable_text

    local_path = "/" + "Users" + "/example/private.parquet"
    assert scan_portable_text(f"checkpoint = '{local_path}'", label="source")
    assert not scan_portable_text("portable relative path: .private_cache", label="source")


def test_peer_eda_evidence_is_aggregate_only() -> None:
    frame = pd.DataFrame(
        {
            "customer_public_id": ["private-a", "private-b", "private-a"],
            "transaction_month": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-02-01"]
            ),
            "zero_order_month_flag": [False, True, False],
            "sparse_history_flag": [False, True, False],
            "channel_s2_score": [1.0, np.nan, 2.0],
            "company_division_s2_score": [1.0, 1.5, 2.0],
            "hybrid_s2_score": [1.0, 1.5, 2.0],
        }
    )
    score_columns = [
        "channel_s2_score",
        "company_division_s2_score",
        "hybrid_s2_score",
    ]

    quality = _peer_data_quality(frame, score_columns=score_columns)
    monthly = _monthly_activity_rate_rows(frame)

    assert quality["customer_month_rows"] == 3
    assert quality["distinct_customers"] == 2
    assert quality["duplicate_customer_month_rows"] == 0
    assert quality["s2_score_null_counts"]["channel"] == 1
    assert [row["customer_month_rows"] for row in monthly] == [2, 1]
    assert all("customer_public_id" not in row for row in monthly)
    assert "private-a" not in json.dumps(
        {"quality": quality, "monthly": monthly}, default=str
    )


def test_exact_capacity_is_selected_per_month() -> None:
    frame = pd.DataFrame(
        {
            "customer_public_id": [f"c{i}" for i in range(10)],
            "transaction_month": [pd.Timestamp("2026-01-01")] * 5
            + [pd.Timestamp("2026-02-01")] * 5,
            "score": np.arange(10, dtype=float),
        }
    )
    flags = _deterministic_top_k(frame, "score", k=2)
    assert int(flags.sum()) == 4
    assert flags.groupby(frame["transaction_month"]).sum().eq(2).all()


def test_waterfill_protects_directions_and_fills_capacity() -> None:
    frame = pd.DataFrame(
        {
            "customer_public_id": [f"c{i}" for i in range(9)],
            "score": np.arange(9, dtype=float),
            "direction": [
                "unexpected_inactivity",
                "unexpected_inactivity",
                "unexpected_activity",
                "unexpected_activity",
                "unexpected_activity",
                "unexpected_active_pattern",
                "unexpected_active_pattern",
                "unexpected_active_pattern",
                "unexpected_active_pattern",
            ],
        }
    )
    flags = _waterfill_top_k(
        frame, score_column="score", direction_column="direction", k=6
    )
    assert int(flags.sum()) == 6
    assert set(frame.loc[flags, "direction"]) == set(frame["direction"])


def test_episode_collapses_consecutive_flags_and_splits_direction() -> None:
    flags = pd.DataFrame(
        {
            "customer_public_id": ["a", "a", "a", "b"],
            "transaction_month": pd.to_datetime(
                ["2026-01-01", "2026-02-01", "2026-03-01", "2026-01-01"]
            ),
            "score": [5.0, 6.0, 7.0, 4.0],
            "signal_direction": ["down", "down", "up", "down"],
            "flag": [True, True, True, True],
        }
    )
    episodes = _episode_table(flags, "test")
    assert len(episodes) == 3
    assert episodes["months_flagged"].max() == 2


def test_generated_pilot_is_blinded_and_exact_capacity() -> None:
    path = _require_generated_artifact(
        OUTPUT_CACHE / "07_pilot_sample_blinded.csv",
        notebook="notebook 07",
    )
    sample = pd.read_csv(path)
    assert len(sample) == 250
    assert "customer_public_id" not in sample
    assert "sampling_stratum" not in sample
    assert "case_id" in sample


def test_generated_evaluation_has_explicit_claim_boundary() -> None:
    path = _require_generated_artifact(
        OUTPUT_CACHE / "module24_evaluation.json",
        notebook="notebook 99",
    )
    evidence = json.loads(path.read_text())
    assert evidence["data_package"]["source"] == "confidential_customer_month_panel"
    assert evidence["workflow"]["main_notebooks"] == 4
    assert "Business precision" in evidence["claim_boundary"]
