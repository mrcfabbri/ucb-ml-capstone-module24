"""Module 24 workflow: peer A/B, hurdle modeling, and alert episodes.

The workflow reads authorized private checkpoints, never raw exports, and writes
only to the configurable local output cache. It deliberately
keeps lifecycle package fields out of historical prediction and keeps raw
company/division codes out of every artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DELIVERABLE_ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = DELIVERABLE_ROOT.parent
UPSTREAM_CACHE = Path(
    os.environ.get("CAPSTONE_UPSTREAM_CACHE_DIR", DELIVERABLE_ROOT / ".private_cache")
).resolve()
OUTPUT_CACHE = Path(
    os.environ.get(
        "CAPSTONE_MODULE24_CACHE_DIR", DELIVERABLE_ROOT / ".cache" / "module24"
    )
).resolve()

ENGINEERED_PATH = UPSTREAM_CACHE / "03_engineered_customer_month.parquet"
S2_PATH = UPSTREAM_CACHE / "05b_scored_customer_month.parquet"
CURRENT_NEXT_STATE_PATH = UPSTREAM_CACHE / "06_next_state_customer_month.parquet"

TRAIN_END = pd.Timestamp("2025-09-01")
CALIB_START = pd.Timestamp("2025-10-01")
CALIB_END = pd.Timestamp("2025-12-01")
FORWARD_START = pd.Timestamp("2026-01-01")
FORWARD_END = pd.Timestamp("2026-06-01")

# These calendar cutoffs are a leakage-control contract.  Model selection may use only the
# training period; calibration is a separate decision window; the forward period is untouched
# until final historical evaluation.

RANDOM_STATE = 42
BOOTSTRAP_RESAMPLES = 1_000
CAPACITIES = (100, 250, 500)

PEER_INPUT_COLUMNS = [
    "customer_public_id",
    "transaction_month",
    "sparse_history_flag",
    "sparse_peer_group_flag",
    "sparse_company_division_peer_group_flag",
    "company_division_peer_public_id",
    "company_division_peer_customer_count",
    "zero_order_month_flag",
    "net_revenue_vs_history_z",
    "order_count_vs_history_z",
    "quantity_vs_history_z",
    "discount_rate_vs_history_z",
    "return_rate_vs_history_z",
    "category_mix_shift_vs_history_z",
    "net_revenue_vs_peer_z",
    "order_value_vs_company_division_peer_z",
    "net_revenue_vs_peers_resid",
    "order_count_vs_peers_resid",
    "net_revenue_yoy_ratio",
    "order_count_yoy_ratio",
    "baseline_anomaly_score",
]

CADENCE_FEATURES = [
    "current_has_order",
    "active_rate_3m",
    "active_rate_6m",
    "active_rate_12m",
    "order_count_mean_3m",
    "order_count_mean_6m",
    "order_count_mean_12m",
    "order_count_sd_12m",
    "months_since_active",
    "inactive_streak",
    "same_month_last_year_active",
    "same_month_last_year_available",
    "current_order_count",
    "current_active_order_days",
    "current_distinct_product_count",
    "current_discount_rate",
    "current_promo_line_share",
    "current_return_line_rate",
    "current_top_product_share",
    "tenure_months_asof",
    "calendar_month_sin",
    "calendar_month_cos",
]

FORBIDDEN_PREDICTIVE_COLUMNS = {
    "originid",
    "open_order_line_count",
    "delivered_order_line_count",
    "invoiced_order_line_count",
    "order_value",
    "order_value_eur",
    "net_revenue",
    "net_revenue_eur",
    "customer_lifecycle_state",
    "company_division_peer_public_id",
}

S2_HISTORY_SIGNALS = [
    "quantity_vs_history_z",
    "discount_rate_vs_history_z",
    "return_rate_vs_history_z",
    "category_mix_shift_vs_history_z",
]


def ensure_output_cache() -> Path:
    OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    return OUTPUT_CACHE


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default, allow_nan=False) + "\n")


def _output_path_metadata(path: Path) -> str:
    """Return a readable private-artifact path for normal and isolated runs."""

    try:
        return str(path.relative_to(ML_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _month_number(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values)
    return dates.dt.year * 12 + dates.dt.month


def _rolling_transform(
    frame: pd.DataFrame, column: str, window: int, statistic: str
) -> pd.Series:
    grouped = frame.groupby("customer_public_id", sort=False)[column]
    if statistic == "mean":
        return grouped.transform(
            lambda values: values.rolling(window, min_periods=1).mean()
        )
    if statistic == "std":
        return grouped.transform(
            lambda values: values.rolling(window, min_periods=2).std()
        )
    raise ValueError(f"Unsupported rolling statistic: {statistic}")


def build_cadence_panel(path: Path = ENGINEERED_PATH) -> pd.DataFrame:
    """Build one as-of feature row at month ``t`` with targets observed at ``t + 1``.

    This is the central timing safeguard for the forecast: every ``current_*`` and rolling feature
    is calculated from the row available at month ``t`` or earlier.  The later ``next_*`` columns
    are targets only, never candidate predictor columns.
    """

    required = [
        "customer_public_id",
        "transaction_month",
        "first_observed_month",
        "order_count",
        "active_order_days",
        "distinct_product_count",
        "discount_rate",
        "promo_line_share",
        "return_line_rate",
        "top_product_revenue_share",
    ]
    frame = pd.read_parquet(path, columns=required)
    frame["transaction_month"] = pd.to_datetime(frame["transaction_month"])
    frame["first_observed_month"] = pd.to_datetime(frame["first_observed_month"])
    frame = frame.sort_values(
        ["customer_public_id", "transaction_month"], kind="mergesort"
    ).reset_index(drop=True)

    frame["current_has_order"] = frame["order_count"].gt(0).astype("int8")
    frame["current_order_count"] = frame["order_count"].astype(float)
    frame["current_active_order_days"] = frame["active_order_days"].fillna(0).astype(float)
    frame["current_distinct_product_count"] = (
        frame["distinct_product_count"].fillna(0).astype(float)
    )
    frame["current_discount_rate"] = frame["discount_rate"].fillna(0).astype(float)
    frame["current_promo_line_share"] = frame["promo_line_share"].fillna(0).astype(float)
    frame["current_return_line_rate"] = frame["return_line_rate"].fillna(0).astype(float)
    frame["current_top_product_share"] = (
        frame["top_product_revenue_share"].fillna(0).astype(float)
    )

    # Rolling transforms are grouped by customer, so one customer's history cannot affect another
    # customer's features.  ``min_periods`` in the helper keeps short histories defined rather
    # than silently dropping those customer-months.
    for window in (3, 6, 12):
        frame[f"active_rate_{window}m"] = _rolling_transform(
            frame, "current_has_order", window, "mean"
        )
        frame[f"order_count_mean_{window}m"] = _rolling_transform(
            frame, "current_order_count", window, "mean"
        )
    frame["order_count_sd_12m"] = _rolling_transform(
        frame, "current_order_count", 12, "std"
    )

    month_number = _month_number(frame["transaction_month"])
    last_active = (
        month_number.where(frame["current_has_order"].eq(1))
        .groupby(frame["customer_public_id"], sort=False)
        .ffill()
    )
    frame["months_since_active"] = (month_number - last_active).fillna(999).clip(0, 999)

    state_change = frame.groupby("customer_public_id", sort=False)[
        "current_has_order"
    ].transform(lambda values: values.ne(values.shift()).cumsum())
    run_position = frame.groupby(
        [frame["customer_public_id"], state_change], sort=False
    ).cumcount() + 1
    frame["inactive_streak"] = run_position.where(
        frame["current_has_order"].eq(0), 0
    ).astype(float)

    lag12 = frame.groupby("customer_public_id", sort=False)["current_has_order"].shift(12)
    frame["same_month_last_year_available"] = lag12.notna().astype("int8")
    frame["same_month_last_year_active"] = lag12.fillna(0).astype("int8")
    frame["tenure_months_asof"] = (
        _month_number(frame["transaction_month"])
        - _month_number(frame["first_observed_month"])
    ).clip(lower=0)
    calendar_month = frame["transaction_month"].dt.month
    frame["calendar_month_sin"] = np.sin(2 * np.pi * calendar_month / 12)
    frame["calendar_month_cos"] = np.cos(2 * np.pi * calendar_month / 12)

    # Shift targets by customer, then keep only genuinely consecutive calendar months below.  This
    # avoids creating a false next-month label across gaps in a customer's observed history.
    grouped = frame.groupby("customer_public_id", sort=False)
    frame["target_month"] = grouped["transaction_month"].shift(-1)
    frame["next_has_order"] = grouped["current_has_order"].shift(-1)
    frame["next_order_count"] = grouped["current_order_count"].shift(-1)
    frame["next_distinct_product_count"] = grouped[
        "current_distinct_product_count"
    ].shift(-1)
    frame["next_discount_rate"] = grouped["current_discount_rate"].shift(-1)
    frame["next_return_line_rate"] = grouped["current_return_line_rate"].shift(-1)

    expected_next_month = frame["transaction_month"] + pd.offsets.MonthBegin(1)
    consecutive = frame["target_month"].eq(expected_next_month)
    frame = frame.loc[consecutive].copy()
    frame["next_has_order"] = frame["next_has_order"].astype("int8")
    frame["data_split"] = np.select(
        [
            frame["target_month"].le(TRAIN_END),
            frame["target_month"].between(CALIB_START, CALIB_END),
            frame["target_month"].between(FORWARD_START, FORWARD_END),
        ],
        ["train", "calibration", "forward"],
        default="unused",
    )

    assert not (set(CADENCE_FEATURES) & FORBIDDEN_PREDICTIVE_COLUMNS)
    assert frame[CADENCE_FEATURES].shape[1] == len(CADENCE_FEATURES)
    return frame


# Fallback used only if the grid search below is skipped or bypassed -- the operational config
# is normally whatever HISTOGRAM_GRID's expanding-window search selects (see run_hurdle_evaluation).
DEFAULT_HISTOGRAM_PARAMS: dict = {
    "learning_rate": 0.06,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
}
HISTOGRAM_GRID: dict[str, list] = {
    "model__learning_rate": [0.05, 0.1, 0.15],
    "model__max_leaf_nodes": [7, 15, 31],
    "model__l2_regularization": [0.5, 1.0, 2.0],
}
HISTOGRAM_GRID_SEARCH_FOLDS = 4
PATTERN_MODEL_PARAMS: dict[str, float | int] = {
    "learning_rate": 0.06,
    "max_iter": 180,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
}
DEFAULT_S2_SCORE_CONFIG: dict[str, float | str] = {
    "aggregation": "maximum_absolute_signal",
    "zero_order_bonus": 0.75,
    "sparse_history_multiplier": 0.75,
}


def _activity_model_pipelines(histogram_params: dict | None = None) -> dict[str, Pipeline]:
    """Return comparable, self-contained pipelines for the two activity-model candidates.

    Imputation and scaling sit inside the pipelines so each chronological fit learns preprocessing
    from its own training partition.  The tree model needs only imputation; the logistic comparator
    also needs scaling because its coefficients are interpreted per standard-deviation change.
    """
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                CADENCE_FEATURES,
            )
        ],
        remainder="drop",
    )
    logistic = Pipeline(
        [
            ("features", preprocessing),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    max_iter=1_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    resolved_histogram_params = {**DEFAULT_HISTOGRAM_PARAMS, **(histogram_params or {})}
    histogram = Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=180,
                    random_state=RANDOM_STATE,
                    **resolved_histogram_params,
                ),
            ),
        ]
    )
    return {"logistic": logistic, "hist_gradient_boosting": histogram}


def _prepare_activity_models(
    activity_models: dict[str, Pipeline] | None,
    *,
    histogram_params: dict,
) -> dict[str, Pipeline]:
    """Clone caller-visible pipelines and apply the train-only grid-search winner."""

    source = activity_models or _activity_model_pipelines()
    required = {"logistic", "hist_gradient_boosting"}
    missing = required - set(source)
    if missing:
        raise ValueError(f"Missing required activity pipelines: {sorted(missing)}")
    models = {name: clone(source[name]) for name in sorted(required)}
    models["hist_gradient_boosting"].set_params(
        **{f"model__{name}": value for name, value in histogram_params.items()}
    )
    return models


def _standardized_logistic_coefficients(model: Pipeline) -> list[dict]:
    """Return the fitted logistic comparator's coefficients in readable rank order.

    The pipeline standardizes every cadence feature, so each coefficient compares a one-standard-
    deviation increase on the fitted training distribution. Interpretation is limited to
    descriptive associations in the logistic comparator; causal inference and explanation of the
    selected tree-based challenger are outside its scope.
    """

    classifier = model.named_steps["model"]
    coefficients = np.asarray(classifier.coef_, dtype=float)
    if coefficients.shape != (1, len(CADENCE_FEATURES)):
        raise ValueError(
            "Expected one binary logistic coefficient per cadence feature; "
            f"found shape {coefficients.shape}"
        )
    rows = [
        {
            "feature": feature,
            "standardized_coefficient": float(coefficient),
            "absolute_coefficient": float(abs(coefficient)),
            "odds_ratio_per_sd": float(np.exp(coefficient)),
            "association_direction": (
                "higher next-month order probability"
                if coefficient > 0
                else "lower next-month order probability"
            ),
        }
        for feature, coefficient in zip(CADENCE_FEATURES, coefficients[0], strict=True)
    ]
    return sorted(rows, key=lambda row: (-row["absolute_coefficient"], row["feature"]))


def _expanding_window_folds(
    panel: pd.DataFrame, mask: pd.Series, n_folds: int = HISTOGRAM_GRID_SEARCH_FOLDS
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Chronological, non-shuffled, expanding-window folds over ``mask``'s calendar months.

    Each fold's validation window is the next block of months; the training window only grows.
    This is what keeps hyperparameter selection leakage-safe on a time series -- a plain (shuffled)
    KFold would let a model tune against months that occur after some of its own training rows,
    which is exactly the leakage the project's forward-holdout design elsewhere exists to avoid.
    Mirrors the fold-construction pattern in
    the chronological model-selection experiment documented in notebook 06.
    """
    eligible = panel.index[mask]
    months = np.sort(panel.loc[eligible, "target_month"].unique())
    if len(months) < n_folds + 1:
        n_folds = max(1, len(months) - 1)
    month_folds = np.array_split(months, n_folds + 1)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_folds):
        validation_months = set(month_folds[i + 1].tolist())
        train_up_to = {
            month for block in month_folds[: i + 1] for month in block.tolist()
        }
        train_positions = np.flatnonzero(
            panel.loc[eligible, "target_month"].isin(train_up_to).to_numpy()
        )
        val_positions = np.flatnonzero(
            panel.loc[eligible, "target_month"].isin(validation_months).to_numpy()
        )
        if len(train_positions) and len(val_positions):
            folds.append((train_positions, val_positions))
    return folds


def _fold_boundary_rows(
    panel: pd.DataFrame,
    mask: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict]:
    """Describe the actual chronological CV folds without exposing customer rows."""

    eligible_months = panel.loc[mask, "target_month"].reset_index(drop=True)
    rows = []
    for fold_number, (train_positions, validation_positions) in enumerate(folds, start=1):
        train_months = eligible_months.iloc[train_positions]
        validation_months = eligible_months.iloc[validation_positions]
        rows.append(
            {
                "fold": fold_number,
                "train_start": train_months.min(),
                "train_end": train_months.max(),
                "validation_start": validation_months.min(),
                "validation_end": validation_months.max(),
                "train_rows": int(len(train_positions)),
                "validation_rows": int(len(validation_positions)),
            }
        )
    return rows


def _reliability_rows(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    *,
    model: str,
    n_bins: int = 8,
) -> list[dict]:
    """Return privacy-safe fixed-width reliability bins for a binary classifier."""

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 0, 1)
    if len(y) != len(p):
        raise ValueError("y_true and probability must have the same length")

    bin_index = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for index in range(n_bins):
        in_bin = bin_index == index
        if not in_bin.any():
            continue
        rows.append(
            {
                "model": model,
                "bin": index + 1,
                "probability_lower": float(index / n_bins),
                "probability_upper": float((index + 1) / n_bins),
                "rows": int(in_bin.sum()),
                "mean_predicted_probability": float(p[in_bin].mean()),
                "observed_activity_rate": float(y[in_bin].mean()),
            }
        )
    return rows


def _grid_search_hist_gradient_boosting(
    panel: pd.DataFrame,
    train_mask: pd.Series,
    *,
    histogram_pipeline: Pipeline | None = None,
    param_grid: dict[str, list] | None = None,
) -> tuple[dict, pd.DataFrame, str | None]:
    """Tune HistGradientBoosting on the training partition only, via expanding-window CV.

    Never touches the calibration or forward splits -- those remain reserved for the isotonic
    calibration gate and the reported forward metrics, exactly as before. Returns the winning
    params, a table of every candidate's mean/std CV log loss, and an edge-of-grid caveat string
    (or None) -- a lightweight version of the boundary-extension check
    Notebook 06 applies this helper only to its declared candidate grid.
    """
    folds = _expanding_window_folds(panel, train_mask)
    X = panel.loc[train_mask, CADENCE_FEATURES]
    y = panel.loc[train_mask, "next_has_order"]
    resolved_grid = param_grid or HISTOGRAM_GRID
    if not resolved_grid or any(not values for values in resolved_grid.values()):
        raise ValueError("The histogram parameter grid must contain non-empty value lists")
    pipe = clone(
        histogram_pipeline
        if histogram_pipeline is not None
        else _activity_model_pipelines()["hist_gradient_boosting"]
    )
    search = GridSearchCV(
        pipe, resolved_grid, scoring="neg_log_loss", cv=folds, n_jobs=-1, refit=False
    )
    search.fit(X, y)
    cv_results = (
        pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score"]]
        .assign(mean_log_loss=lambda d: -d["mean_test_score"])
        .drop(columns="mean_test_score")
        .sort_values("mean_log_loss")
        .reset_index(drop=True)
    )
    best_params = {
        key.removeprefix("model__"): value for key, value in search.best_params_.items()
    }
    boundary_note = None
    edge_hits = [
        name
        for name, value in best_params.items()
        if value
        in (
            resolved_grid[f"model__{name}"][0],
            resolved_grid[f"model__{name}"][-1],
        )
    ]
    if edge_hits:
        boundary_note = (
            f"Best config sits at the grid edge for {edge_hits} -- the true optimum may lie "
            "outside the searched range; treated as a caveat, not re-searched, to keep this "
            "grid's scope contained."
        )
    return best_params, cv_results, boundary_note


def _binary_metrics(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    *,
    model: str,
) -> dict:
    """Calculate the shared forward metrics from probabilities, not hard labels alone.

    The 0.5 threshold is used only for the classification-style metrics.  Log loss, Brier score,
    ROC-AUC, and average precision preserve the probability information used to select the model.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    prediction = p >= 0.5
    return {
        "model": model,
        "rows": int(len(y)),
        "activity_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def _customer_clustered_bootstrap(
    customer_ids: pd.Series,
    y_true: pd.Series | np.ndarray,
    selected_probability: np.ndarray,
    baseline_probability: np.ndarray,
    *,
    baseline: str,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Quantify forward metric differences with customer-clustered resampling.

    The forward holdout remains fixed.  Each replicate samples whole customers with
    replacement and keeps all of each selected customer's forward months together.
    This respects within-customer dependence without saving identifiers or row-level
    bootstrap results.  Positive ROC-AUC differences and negative log-loss
    differences favor the selected model.
    """

    clusters, inverse = np.unique(np.asarray(customer_ids, dtype=str), return_inverse=True)
    y = np.asarray(y_true, dtype=int)
    selected = np.clip(np.asarray(selected_probability, dtype=float), 1e-8, 1 - 1e-8)
    comparator = np.clip(np.asarray(baseline_probability, dtype=float), 1e-8, 1 - 1e-8)
    rng = np.random.default_rng(random_state)
    cluster_rows = [np.flatnonzero(inverse == number) for number in range(len(clusters))]
    log_loss_difference = np.empty(n_resamples, dtype=float)
    roc_auc_difference = np.empty(n_resamples, dtype=float)

    for replicate in range(n_resamples):
        selected_clusters = rng.integers(0, len(clusters), size=len(clusters))
        row_positions = np.concatenate([cluster_rows[number] for number in selected_clusters])
        sample_y = y[row_positions]
        sample_selected = selected[row_positions]
        sample_comparator = comparator[row_positions]
        log_loss_difference[replicate] = (
            log_loss(sample_y, sample_selected) - log_loss(sample_y, sample_comparator)
        )
        if np.unique(sample_y).size == 2:
            roc_auc_difference[replicate] = (
                roc_auc_score(sample_y, sample_selected)
                - roc_auc_score(sample_y, sample_comparator)
            )
        else:
            roc_auc_difference[replicate] = np.nan

    def summary(values: np.ndarray, favorable: str) -> dict:
        usable = values[np.isfinite(values)]
        return {
            "point_difference": float(values[0] * 0 + np.nan),
            "ci_95": [float(np.quantile(usable, 0.025)), float(np.quantile(usable, 0.975))],
            "favorable_replicate_share": float(
                (usable < 0).mean() if favorable == "lower" else (usable > 0).mean()
            ),
            "usable_replicates": int(len(usable)),
        }

    log_loss_summary = summary(log_loss_difference, "lower")
    roc_auc_summary = summary(roc_auc_difference, "higher")
    log_loss_summary["point_difference"] = float(
        log_loss(y, selected) - log_loss(y, comparator)
    )
    roc_auc_summary["point_difference"] = float(
        roc_auc_score(y, selected) - roc_auc_score(y, comparator)
    )
    return {
        "method": "customer_clustered_nonparametric_bootstrap",
        "unit": "customer; all forward months retained within each sampled customer",
        "random_state": random_state,
        "resamples": n_resamples,
        "forward_customers": int(len(clusters)),
        "baseline": baseline,
        "selected_minus_baseline": {
            "log_loss": log_loss_summary,
            "roc_auc": roc_auc_summary,
        },
        "interpretation": (
            "Negative log-loss differences and positive ROC-AUC differences favor the selected "
            "model. Intervals describe sampling uncertainty on this fixed forward holdout, not "
            "commercial value or future business performance."
        ),
    }


def _markov_probabilities(train: pd.DataFrame, apply: pd.DataFrame) -> np.ndarray:
    """Return a smoothed one-step historical baseline conditioned on current activity.

    Laplace smoothing prevents an unseen transition from becoming an exact zero or one probability.
    It is intentionally simple: its role is to make the challenger's improvement interpretable,
    not to compete with the full feature-based model on engineering complexity.
    """
    counts = (
        train.groupby("current_has_order")["next_has_order"]
        .agg(["sum", "count"])
        .reindex([0, 1], fill_value=0)
    )
    probability = (counts["sum"] + 1) / (counts["count"] + 2)
    return apply["current_has_order"].map(probability).fillna(train["next_has_order"].mean()).to_numpy()


def _fit_pattern_labels(panel: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Create fixed, train-derived labels for the conditional active-month pattern task.

    Quantile thresholds are fitted only on active training rows, then reused for calibration and
    forward rows.  The explicit precedence list resolves cases that satisfy more than one label.
    """
    train_active = panel["data_split"].eq("train") & panel["next_has_order"].eq(1)
    discount_p90 = float(panel.loc[train_active, "next_discount_rate"].quantile(0.90))
    breadth_p75 = float(
        panel.loc[train_active, "next_distinct_product_count"].quantile(0.75)
    )
    orders_p75 = float(panel.loc[train_active, "next_order_count"].quantile(0.75))

    labels = np.select(
        [
            panel["next_return_line_rate"].gt(0),
            panel["next_discount_rate"].gt(discount_p90),
            panel["next_distinct_product_count"].ge(breadth_p75),
            panel["next_order_count"].ge(orders_p75),
        ],
        ["return_heavy", "discount_heavy", "broad", "multi_order"],
        default="typical",
    )
    return pd.Series(labels, index=panel.index), {
        "discount_p90": discount_p90,
        "breadth_p75": breadth_p75,
        "orders_p75": orders_p75,
        "precedence": [
            "return_heavy",
            "discount_heavy",
            "broad",
            "multi_order",
            "typical",
        ],
    }


def run_hurdle_evaluation(
    *,
    activity_models: dict[str, Pipeline] | None = None,
    histogram_grid: dict[str, list] | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Fit the two-stage model and evaluate it on the fixed Jan-Jun 2026 window."""

    ensure_output_cache()
    panel = build_cadence_panel()
    train = panel["data_split"].eq("train")
    calibration = panel["data_split"].eq("calibration")
    forward = panel["data_split"].eq("forward")
    if not (train.any() and calibration.any() and forward.any()):
        raise ValueError("The train/calibration/forward split is incomplete")

    # Grid search + cross-validation for the HistGradientBoosting challenger, training-partition
    # only. The calibration and forward splits below are untouched by this search and remain the
    # primary validation for every reported number. Chronological expanding-window CV is used only
    # for hyperparameter selection; the forward holdout provides the final evaluation.
    resolved_histogram_grid = histogram_grid or HISTOGRAM_GRID
    histogram_pipeline = (
        activity_models.get("hist_gradient_boosting")
        if activity_models is not None
        else None
    )
    best_histogram_params, histogram_cv_results, histogram_grid_boundary_note = (
        _grid_search_hist_gradient_boosting(
            panel,
            train,
            histogram_pipeline=histogram_pipeline,
            param_grid=resolved_histogram_grid,
        )
    )
    fold_boundaries = _fold_boundary_rows(
        panel, train, _expanding_window_folds(panel, train)
    )
    grid_search_summary = {
        "grid": resolved_histogram_grid,
        "folds": HISTOGRAM_GRID_SEARCH_FOLDS,
        "fold_boundaries": fold_boundaries,
        "scoring": "neg_log_loss",
        "best_params": best_histogram_params,
        "cv_results": histogram_cv_results.to_dict("records"),
        "boundary_note": histogram_grid_boundary_note,
        "note": (
            "Tuned only on the train split via expanding-window (chronological) CV. The "
            "calibration and forward splits below remain reserved and untouched by this search; "
            "they are the primary validation strategy for every reported metric, not k-fold CV, "
            "because a shuffled k-fold would let the model tune against months that occur after "
            "some of its own training rows."
        ),
    }

    models = _prepare_activity_models(
        activity_models,
        histogram_params=best_histogram_params,
    )
    calibration_metrics = []
    forward_metrics = []
    fitted: dict[str, Pipeline] = {}
    for name, model in models.items():
        X_train = panel.loc[train, CADENCE_FEATURES]
        y_train = panel.loc[train, "next_has_order"]
        model.fit(X_train, y_train)
        fitted[name] = model
        calib_probability = model.predict_proba(
            panel.loc[calibration, CADENCE_FEATURES]
        )[:, 1]
        calibration_metrics.append(
            _binary_metrics(
                panel.loc[calibration, "next_has_order"],
                calib_probability,
                model=name,
            )
        )
    logistic_coefficients = _standardized_logistic_coefficients(fitted["logistic"])

    calib_by_name = {row["model"]: row for row in calibration_metrics}
    selected_name = (
        "hist_gradient_boosting"
        if calib_by_name["hist_gradient_boosting"]["log_loss"]
        <= calib_by_name["logistic"]["log_loss"] - 0.005
        else "logistic"
    )
    selected = fitted[selected_name]
    calibration_fit = calibration & panel["target_month"].lt(CALIB_END)
    calibration_check = calibration & panel["target_month"].eq(CALIB_END)
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(selected), method="isotonic"
    )
    calibrated.fit(
        panel.loc[calibration_fit, CADENCE_FEATURES],
        panel.loc[calibration_fit, "next_has_order"],
    )
    raw_check_probability = selected.predict_proba(
        panel.loc[calibration_check, CADENCE_FEATURES]
    )[:, 1]
    calibrated_check_probability = calibrated.predict_proba(
        panel.loc[calibration_check, CADENCE_FEATURES]
    )[:, 1]
    raw_check = _binary_metrics(
        panel.loc[calibration_check, "next_has_order"],
        raw_check_probability,
        model=f"{selected_name}_raw",
    )
    calibrated_check = _binary_metrics(
        panel.loc[calibration_check, "next_has_order"],
        calibrated_check_probability,
        model=f"{selected_name}_isotonic",
    )
    calibration_reliability = {
        "target_month": CALIB_END.strftime("%Y-%m"),
        "binning": "eight equal-width probability bins; empty bins are omitted",
        "rows": [
            *_reliability_rows(
                panel.loc[calibration_check, "next_has_order"],
                raw_check_probability,
                model=f"{selected_name}_raw",
            ),
            *_reliability_rows(
                panel.loc[calibration_check, "next_has_order"],
                calibrated_check_probability,
                model=f"{selected_name}_isotonic",
            ),
        ],
    }
    use_isotonic = (
        calibrated_check["log_loss"] <= raw_check["log_loss"] - 0.001
        and calibrated_check["brier"] <= raw_check["brier"]
    )
    operational_model = calibrated if use_isotonic else selected
    operational_model_name = (
        f"{selected_name}_isotonic" if use_isotonic else f"{selected_name}_raw"
    )

    for name, model in fitted.items():
        probability = model.predict_proba(panel.loc[forward, CADENCE_FEATURES])[:, 1]
        forward_metrics.append(
            _binary_metrics(
                panel.loc[forward, "next_has_order"], probability, model=name
            )
        )
    selected_probability = operational_model.predict_proba(
        panel.loc[forward, CADENCE_FEATURES]
    )[:, 1]
    forward_metrics.append(
        _binary_metrics(
            panel.loc[forward, "next_has_order"],
            selected_probability,
            model=operational_model_name,
        )
    )
    markov_probability = _markov_probabilities(
        panel.loc[train], panel.loc[forward]
    )
    forward_metrics.append(
        _binary_metrics(
            panel.loc[forward, "next_has_order"],
            markov_probability,
            model="smoothed_activity_markov",
        )
    )
    global_probability = np.repeat(
        panel.loc[train, "next_has_order"].mean(), int(forward.sum())
    )
    forward_metrics.append(
        _binary_metrics(
            panel.loc[forward, "next_has_order"],
            global_probability,
            model="global_activity_rate",
        )
    )
    forward_uncertainty = [
        _customer_clustered_bootstrap(
            panel.loc[forward, "customer_public_id"],
            panel.loc[forward, "next_has_order"],
            selected_probability,
            markov_probability,
            baseline="smoothed_activity_markov",
        ),
        _customer_clustered_bootstrap(
            panel.loc[forward, "customer_public_id"],
            panel.loc[forward, "next_has_order"],
            selected_probability,
            global_probability,
            baseline="global_activity_rate",
        ),
    ]

    pattern_labels, pattern_thresholds = _fit_pattern_labels(panel)
    panel["next_active_pattern"] = pattern_labels
    train_active = train & panel["next_has_order"].eq(1)
    forward_active = forward & panel["next_has_order"].eq(1)
    pattern_model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    random_state=RANDOM_STATE,
                    **PATTERN_MODEL_PARAMS,
                ),
            ),
        ]
    )
    pattern_model.fit(
        panel.loc[train_active, CADENCE_FEATURES],
        panel.loc[train_active, "next_active_pattern"],
    )
    pattern_probability = pattern_model.predict_proba(
        panel.loc[forward_active, CADENCE_FEATURES]
    )
    pattern_prediction = pattern_model.classes_[
        np.argmax(pattern_probability, axis=1)
    ]
    y_pattern = panel.loc[forward_active, "next_active_pattern"].to_numpy()
    pattern_metrics = {
        "rows": int(forward_active.sum()),
        "classes": pattern_model.classes_.tolist(),
        "accuracy": float(accuracy_score(y_pattern, pattern_prediction)),
        "macro_f1": float(
            f1_score(y_pattern, pattern_prediction, average="macro", zero_division=0)
        ),
        "log_loss": float(
            log_loss(y_pattern, pattern_probability, labels=pattern_model.classes_)
        ),
        "majority_accuracy": float(
            pd.Series(y_pattern).value_counts(normalize=True).max()
        ),
    }

    prediction = panel.loc[forward, [
        "customer_public_id",
        "transaction_month",
        "target_month",
        "data_split",
        "next_has_order",
    ]].copy()
    prediction = prediction.rename(columns={"transaction_month": "feature_month"})
    prediction["p_next_order"] = selected_probability
    prediction["predicted_activity"] = prediction["p_next_order"].ge(0.5).astype("int8")
    realized_probability = np.where(
        prediction["next_has_order"].eq(1),
        prediction["p_next_order"],
        1 - prediction["p_next_order"],
    )
    prediction["activity_surprisal"] = -np.log(np.clip(realized_probability, 1e-8, 1))
    prediction["actual_active_pattern"] = "not_active"
    prediction["predicted_active_pattern"] = "not_active"
    prediction["pattern_surprisal"] = 0.0

    active_positions = np.flatnonzero(prediction["next_has_order"].eq(1).to_numpy())
    class_lookup = {name: i for i, name in enumerate(pattern_model.classes_)}
    actual_index = np.array([class_lookup[value] for value in y_pattern])
    actual_probability = pattern_probability[np.arange(len(y_pattern)), actual_index]
    prediction.iloc[
        active_positions,
        prediction.columns.get_loc("actual_active_pattern"),
    ] = y_pattern
    prediction.iloc[
        active_positions,
        prediction.columns.get_loc("predicted_active_pattern"),
    ] = pattern_prediction
    prediction.iloc[
        active_positions,
        prediction.columns.get_loc("pattern_surprisal"),
    ] = -np.log(np.clip(actual_probability, 1e-8, 1))

    unexpected_inactivity = prediction["next_has_order"].eq(0) & prediction[
        "predicted_activity"
    ].eq(1)
    unexpected_activity = prediction["next_has_order"].eq(1) & prediction[
        "predicted_activity"
    ].eq(0)
    unexpected_pattern = (
        prediction["next_has_order"].eq(1)
        & prediction["predicted_activity"].eq(1)
        & prediction["actual_active_pattern"].ne(
            prediction["predicted_active_pattern"]
        )
    )
    prediction["hurdle_direction"] = np.select(
        [unexpected_inactivity, unexpected_activity, unexpected_pattern],
        [
            "unexpected_inactivity",
            "unexpected_activity",
            "unexpected_active_pattern",
        ],
        default="expected",
    )
    prediction["hurdle_surprisal_score"] = np.maximum(
        prediction["activity_surprisal"], prediction["pattern_surprisal"]
    )
    prediction["hurdle_eligible"] = prediction["hurdle_direction"].ne("expected")

    prediction_path = OUTPUT_CACHE / "06_hurdle_forward_predictions.parquet"
    prediction.to_parquet(prediction_path, index=False)
    feature_contract = {
        "feature_month": "t",
        "target_month": "t+1",
        "features": CADENCE_FEATURES,
        "forbidden_predictive_columns": sorted(FORBIDDEN_PREDICTIVE_COLUMNS),
        "activity_selection_rule": {
            "incumbent": "logistic",
            "challenger": "hist_gradient_boosting",
            "calibration_log_loss_margin": 0.005,
            "selected": selected_name,
        },
        "calibration_selection_rule": {
            "fit_months": ["2025-10", "2025-11"],
            "check_month": "2025-12",
            "minimum_log_loss_improvement": 0.001,
            "brier_must_not_worsen": True,
            "selected": operational_model_name,
        },
        "pattern_thresholds_fit_on_train_active_rows": pattern_thresholds,
        "splits_by_target_month": {
            "train_end": TRAIN_END,
            "calibration": [CALIB_START, CALIB_END],
            "forward": [FORWARD_START, FORWARD_END],
        },
    }
    write_json(feature_contract, OUTPUT_CACHE / "06_feature_contract.json")
    metrics = {
        "panel_rows": int(len(panel)),
        "customers": int(panel["customer_public_id"].nunique()),
        "split_rows": panel["data_split"].value_counts().to_dict(),
        "calibration_metrics": calibration_metrics,
        "forward_metrics": forward_metrics,
        "forward_uncertainty": forward_uncertainty,
        "selected_activity_model": selected_name,
        "operational_activity_model": operational_model_name,
        "calibration_check_metrics": [raw_check, calibrated_check],
        "calibration_reliability": calibration_reliability,
        "pattern_metrics": pattern_metrics,
        "pattern_model_design": {
            "selection": "fixed_not_grid_searched",
            "fixed_params": PATTERN_MODEL_PARAMS,
            "rationale": (
                "Stage 2 is exploratory active-month context. Its fixed configuration is "
                "evaluated on the untouched forward window against a majority baseline rather "
                "than presented as a separately optimized classifier."
            ),
        },
        "forward_direction_counts": prediction["hurdle_direction"].value_counts().to_dict(),
        "grid_search": grid_search_summary,
        "activity_model_pipelines": [
            {
                "model": "logistic",
                "fitted_pipeline": repr(fitted["logistic"]),
                "preprocessing": (
                    "Median imputation and StandardScaler are fitted inside the pipeline "
                    "on each training partition."
                ),
            },
            {
                "model": "hist_gradient_boosting",
                "fitted_pipeline": repr(fitted["hist_gradient_boosting"]),
                "preprocessing": (
                    "Median imputation is fitted inside the pipeline on each training "
                    "partition; tree model needs no scaling."
                ),
            },
        ],
        "activity_model_construction": (
            "caller_supplied_notebook_objects"
            if activity_models is not None
            else "workflow_defaults"
        ),
        "logistic_coefficient_interpretation": {
            "meaning": (
                "Standardized logistic-comparator associations fitted on the training split; "
                "descriptive rather than causal and not an explanation of the selected "
                "histogram-gradient-boosting model."
            ),
            "coefficients": logistic_coefficients,
        },
        "output": _output_path_metadata(prediction_path),
    }
    write_json(metrics, OUTPUT_CACHE / "06_hurdle_metrics.json")
    return metrics, prediction


def refresh_forward_uncertainty() -> dict:
    """Add bootstrap evidence to an existing forward sidecar without refitting models.

    This supports a deterministic evidence refresh when the saved model predictions are already
    current. It rebuilds only as-of features to recover the train-only baseline probabilities.
    """

    panel = build_cadence_panel()
    train = panel["data_split"].eq("train")
    forward = panel["data_split"].eq("forward")
    prediction = pd.read_parquet(OUTPUT_CACHE / "06_hurdle_forward_predictions.parquet")
    forward_frame = panel.loc[forward].reset_index(drop=True)
    expected_keys = forward_frame[["customer_public_id", "target_month"]].reset_index(drop=True)
    observed_keys = prediction[["customer_public_id", "target_month"]].reset_index(drop=True)
    if not expected_keys.equals(observed_keys):
        raise ValueError("Saved forward predictions do not match the current private checkpoint")
    markov_probability = _markov_probabilities(panel.loc[train], forward_frame)
    global_probability = np.repeat(panel.loc[train, "next_has_order"].mean(), len(forward_frame))
    uncertainty = [
        _customer_clustered_bootstrap(
            forward_frame["customer_public_id"],
            forward_frame["next_has_order"],
            prediction["p_next_order"].to_numpy(),
            markov_probability,
            baseline="smoothed_activity_markov",
        ),
        _customer_clustered_bootstrap(
            forward_frame["customer_public_id"],
            forward_frame["next_has_order"],
            prediction["p_next_order"].to_numpy(),
            global_probability,
            baseline="global_activity_rate",
        ),
    ]
    metrics_path = OUTPUT_CACHE / "06_hurdle_metrics.json"
    metrics = read_json(metrics_path)
    metrics["forward_uncertainty"] = uncertainty
    write_json(metrics, metrics_path)
    return uncertainty


def _robust_yoy_signal(
    ratio: pd.Series, fit_mask: pd.Series
) -> tuple[pd.Series, dict]:
    raw = pd.to_numeric(ratio, errors="coerce")
    logged = np.log(raw.where(raw.gt(0))).replace([np.inf, -np.inf], np.nan)
    fit = logged.loc[fit_mask].dropna()
    mad = float(fit.abs().median()) if len(fit) else np.nan
    scale = 1.4826 * mad
    method = "MAD"
    if not np.isfinite(scale) or scale <= 0:
        scale = float((fit.quantile(0.75) - fit.quantile(0.25)) / 1.349)
        method = "IQR"
    if not np.isfinite(scale) or scale <= 0:
        scale = float(fit.std(ddof=0))
        method = "standard_deviation"
    transformed = logged / scale if np.isfinite(scale) and scale > 0 else np.nan
    return pd.Series(transformed, index=ratio.index).clip(-10, 10), {
        "scale": scale,
        "method": method,
        "observations": int(len(fit)),
    }


def _peer_group_residual(
    frame: pd.DataFrame, value_column: str, group_column: str
) -> tuple[pd.Series, pd.Series]:
    keys = [frame["transaction_month"], frame[group_column]]
    group = frame.groupby(keys, dropna=False)[value_column]
    median = group.transform("median")
    count = group.transform("count")
    residual = pd.to_numeric(frame[value_column], errors="coerce") - median
    return residual.clip(-10, 10), count


def _resolve_s2_score_config(
    score_config: dict[str, float | str] | None,
) -> dict[str, float | str]:
    """Validate the compact scoring configuration supplied visibly by notebook 05."""

    resolved = {**DEFAULT_S2_SCORE_CONFIG, **(score_config or {})}
    unknown = set(resolved) - set(DEFAULT_S2_SCORE_CONFIG)
    if unknown:
        raise ValueError(f"Unsupported S2 score configuration fields: {sorted(unknown)}")
    if resolved["aggregation"] != "maximum_absolute_signal":
        raise ValueError("S2 aggregation must remain 'maximum_absolute_signal'")
    if float(resolved["zero_order_bonus"]) < 0:
        raise ValueError("zero_order_bonus must be non-negative")
    multiplier = float(resolved["sparse_history_multiplier"])
    if not 0 < multiplier <= 1:
        raise ValueError("sparse_history_multiplier must be in (0, 1]")
    return resolved


def _s2_score(
    frame: pd.DataFrame,
    *,
    peer_level: pd.Series,
    spend_residual: pd.Series,
    order_residual: pd.Series,
    fit_mask: pd.Series,
    score_config: dict[str, float | str] | None = None,
) -> tuple[pd.Series, dict]:
    resolved_config = _resolve_s2_score_config(score_config)
    signals = {
        "spend_effective": spend_residual,
        "order_effective": order_residual,
        "peer_level": peer_level,
    }
    for column in S2_HISTORY_SIGNALS:
        signals[column] = pd.to_numeric(frame[column], errors="coerce").clip(-10, 10)
    yoy_meta = {}
    for column in ("net_revenue_yoy_ratio", "order_count_yoy_ratio"):
        transformed, metadata = _robust_yoy_signal(frame[column], fit_mask)
        signals[f"{column}_robust_z"] = transformed
        yoy_meta[column] = metadata
    signal_frame = pd.DataFrame(signals, index=frame.index)
    score = signal_frame.abs().fillna(0).max(axis=1)
    score = score + frame["zero_order_month_flag"].fillna(False).astype(float) * float(
        resolved_config["zero_order_bonus"]
    )
    score = score.where(
        ~frame["sparse_history_flag"].fillna(True),
        score * float(resolved_config["sparse_history_multiplier"]),
    )
    return score, yoy_meta


def _peer_data_quality(
    frame: pd.DataFrame, *, score_columns: list[str]
) -> dict:
    """Summarize panel completeness with aggregate-only, submission-safe fields."""

    months = pd.to_datetime(frame["transaction_month"])
    return {
        "customer_month_rows": int(len(frame)),
        "distinct_customers": int(frame["customer_public_id"].nunique()),
        "months": int(months.nunique()),
        "month_start": months.min(),
        "month_end": months.max(),
        "duplicate_customer_month_rows": int(
            frame.duplicated(["customer_public_id", "transaction_month"]).sum()
        ),
        "zero_order_month_rate": float(frame["zero_order_month_flag"].mean()),
        "sparse_history_rate": float(frame["sparse_history_flag"].fillna(True).mean()),
        "s2_score_null_counts": {
            column.removesuffix("_s2_score"): int(frame[column].isna().sum())
            for column in score_columns
        },
    }


def _monthly_activity_rate_rows(frame: pd.DataFrame) -> list[dict]:
    """Aggregate monthly activity from the zero-order flag without retaining identities."""

    monthly = (
        frame.groupby("transaction_month", as_index=False)["zero_order_month_flag"]
        .agg(customer_month_rows="size", activity_rate=lambda values: 1 - values.mean())
        .sort_values("transaction_month")
    )
    return monthly.to_dict("records")


def _score_distribution_rows(
    frame: pd.DataFrame, *, score_columns: list[str]
) -> list[dict]:
    """Return compact score summaries rather than private scored customer rows."""

    return [
        {
            "strategy": column.removesuffix("_s2_score"),
            "mean": float(frame[column].mean()),
            "median": float(frame[column].median()),
            "p95": float(frame[column].quantile(0.95)),
            "p99": float(frame[column].quantile(0.99)),
            "max": float(frame[column].max()),
        }
        for column in score_columns
    ]


def _deterministic_top_k(
    frame: pd.DataFrame,
    score_column: str,
    *,
    k: int,
    eligible: pd.Series | None = None,
    month_column: str = "transaction_month",
) -> pd.Series:
    flags = pd.Series(False, index=frame.index)
    reviewable = (
        pd.Series(True, index=frame.index)
        if eligible is None
        else eligible.reindex(frame.index, fill_value=False).astype(bool)
    )
    for _, month_rows in frame.loc[reviewable].groupby(month_column, sort=True):
        ordered = month_rows.sort_values(
            [score_column, "customer_public_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        flags.loc[ordered.head(min(k, len(ordered))).index] = True
    return flags


def _jaccard(left: pd.Series, right: pd.Series) -> float:
    left = left.astype(bool)
    right = right.astype(bool)
    union = int((left | right).sum())
    return float((left & right).sum() / union) if union else math.nan


def run_peer_strategy_evaluation(
    *,
    s2_score_config: dict[str, float | str] | None = None,
    prepared_frame: pd.DataFrame | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Compare channel, company/division, and hybrid S2 strategies at fixed K."""

    ensure_output_cache()
    resolved_s2_score_config = _resolve_s2_score_config(s2_score_config)
    if prepared_frame is None:
        frame = pd.read_parquet(S2_PATH, columns=PEER_INPUT_COLUMNS)
        input_source = "workflow_loaded_checkpoint"
    else:
        missing_columns = sorted(set(PEER_INPUT_COLUMNS) - set(prepared_frame.columns))
        if missing_columns:
            raise ValueError(
                "Prepared peer input is missing required columns: "
                + ", ".join(missing_columns)
            )
        frame = prepared_frame.loc[:, PEER_INPUT_COLUMNS].copy()
        input_source = "caller_prepared_frame"
    frame["transaction_month"] = pd.to_datetime(frame["transaction_month"])
    fit_mask = frame["transaction_month"].le(TRAIN_END) & ~frame[
        "sparse_history_flag"
    ].fillna(True)
    forward = frame["transaction_month"].between(FORWARD_START, FORWARD_END)
    eligible = ~frame["sparse_history_flag"].fillna(True)

    cd_spend_residual, cd_group_count = _peer_group_residual(
        frame, "net_revenue_vs_history_z", "company_division_peer_public_id"
    )
    cd_order_residual, _ = _peer_group_residual(
        frame, "order_count_vs_history_z", "company_division_peer_public_id"
    )
    cd_usable = (
        cd_group_count.ge(5)
        & frame["order_value_vs_company_division_peer_z"].notna()
        & ~frame["sparse_company_division_peer_group_flag"].fillna(True)
    )
    channel_spend = frame["net_revenue_vs_peers_resid"].where(
        ~frame["sparse_peer_group_flag"].fillna(True),
        frame["net_revenue_vs_history_z"],
    )
    channel_order = frame["order_count_vs_peers_resid"].where(
        ~frame["sparse_peer_group_flag"].fillna(True),
        frame["order_count_vs_history_z"],
    )
    cd_spend = cd_spend_residual.where(cd_usable, frame["net_revenue_vs_history_z"])
    cd_order = cd_order_residual.where(cd_usable, frame["order_count_vs_history_z"])

    strategies = {
        "channel": {
            "peer_level": frame["net_revenue_vs_peer_z"],
            "spend": channel_spend,
            "order": channel_order,
            "sparse": frame["sparse_peer_group_flag"].fillna(True),
        },
        "company_division": {
            "peer_level": frame["order_value_vs_company_division_peer_z"],
            "spend": cd_spend,
            "order": cd_order,
            "sparse": ~cd_usable,
        },
        "hybrid": {
            "peer_level": frame["order_value_vs_company_division_peer_z"].where(
                cd_usable, frame["net_revenue_vs_peer_z"]
            ),
            "spend": cd_spend_residual.where(cd_usable, channel_spend),
            "order": cd_order_residual.where(cd_usable, channel_order),
            "sparse": (
                (~cd_usable) & frame["sparse_peer_group_flag"].fillna(True)
            ),
        },
    }

    yoy_metadata = None
    for name, definition in strategies.items():
        score, metadata = _s2_score(
            frame,
            peer_level=definition["peer_level"],
            spend_residual=definition["spend"],
            order_residual=definition["order"],
            fit_mask=fit_mask,
            score_config=resolved_s2_score_config,
        )
        frame[f"{name}_s2_score"] = score
        yoy_metadata = metadata
        for capacity in CAPACITIES:
            frame[f"{name}_top_{capacity}"] = _deterministic_top_k(
                frame,
                f"{name}_s2_score",
                k=capacity,
                eligible=eligible & forward,
            )

    strategy_metrics = []
    for name, definition in strategies.items():
        strategy_metrics.append(
            {
                "strategy": name,
                "forward_rows": int(forward.sum()),
                "sparse_peer_rate": float(
                    definition["sparse"].loc[forward].mean()
                ),
                "spearman_vs_channel": float(
                    frame.loc[forward, f"{name}_s2_score"].corr(
                        frame.loc[forward, "channel_s2_score"], method="spearman"
                    )
                ),
                "pearson_vs_channel": float(
                    frame.loc[forward, f"{name}_s2_score"].corr(
                        frame.loc[forward, "channel_s2_score"], method="pearson"
                    )
                ),
            }
        )

    score_columns = [f"{name}_s2_score" for name in strategies]
    data_quality = _peer_data_quality(frame, score_columns=score_columns)
    monthly_activity_rate = _monthly_activity_rate_rows(frame)
    forward_scores = frame.loc[forward, score_columns]
    score_distribution = _score_distribution_rows(
        forward_scores, score_columns=score_columns
    )
    company_division_max_index = forward_scores["company_division_s2_score"].idxmax()
    top_one_percent_threshold = float(
        forward_scores["company_division_s2_score"].quantile(0.99)
    )
    top_one_percent = forward & frame["company_division_s2_score"].ge(
        top_one_percent_threshold
    )
    company_division_tail_diagnostic = {
        "window": "forward",
        "max_score": float(
            frame.loc[company_division_max_index, "company_division_s2_score"]
        ),
        "max_score_uses_sparse_company_division_peer_cell": bool(
            ~cd_usable.loc[company_division_max_index]
        ),
        "max_score_company_division_peer_customer_count": int(
            frame.loc[
                company_division_max_index, "company_division_peer_customer_count"
            ]
        ),
        "max_score_peer_residual_rows": int(
            cd_group_count.loc[company_division_max_index]
        ),
        "top_one_percent_rows": int(top_one_percent.sum()),
        "top_one_percent_sparse_company_division_peer_cell_rate": float(
            (~cd_usable.loc[top_one_percent]).mean()
        ),
    }

    overlap_rows = []
    for capacity in CAPACITIES:
        channel_flag = frame[f"channel_top_{capacity}"]
        channel_column = f"channel_top_{capacity}"
        for name in ("company_division", "hybrid"):
            candidate = frame[f"{name}_top_{capacity}"]
            candidate_column = f"{name}_top_{capacity}"
            overlap_rows.append(
                {
                    "capacity_per_month": capacity,
                    "comparison": f"channel_vs_{name}",
                    "pooled_jaccard": _jaccard(channel_flag, candidate),
                    "changed_customer_months": int((channel_flag != candidate).sum()),
                    "mean_monthly_jaccard": float(
                        frame.loc[forward]
                        .groupby("transaction_month")
                        .apply(
                            lambda month, channel=channel_column, candidate=candidate_column: _jaccard(
                                month[channel],
                                month[candidate],
                            ),
                            include_groups=False,
                        )
                        .mean()
                    ),
                }
            )

    baseline_comparison = {
        "pearson_recomputed_channel_vs_current_s2": float(
            frame.loc[forward, "channel_s2_score"].corr(
                frame.loc[forward, "baseline_anomaly_score"]
            )
        ),
        "spearman_recomputed_channel_vs_current_s2": float(
            frame.loc[forward, "channel_s2_score"].corr(
                frame.loc[forward, "baseline_anomaly_score"], method="spearman"
            )
        ),
    }

    safe_columns = [
        "customer_public_id",
        "transaction_month",
        "channel_s2_score",
        "company_division_s2_score",
        "hybrid_s2_score",
        *[
            f"{strategy}_top_{capacity}"
            for strategy in strategies
            for capacity in CAPACITIES
        ],
    ]
    output = frame.loc[forward, safe_columns].copy()
    output_path = OUTPUT_CACHE / "05_peer_strategy_forward.parquet"
    output.to_parquet(output_path, index=False)
    metrics = {
        "forward_months": [
            FORWARD_START.strftime("%Y-%m"),
            FORWARD_END.strftime("%Y-%m"),
        ],
        "capacities_per_month": list(CAPACITIES),
        "strategy_metrics": strategy_metrics,
        "capacity_overlap": overlap_rows,
        "baseline_reproduction": baseline_comparison,
        "yoy_calibration": yoy_metadata,
        "company_division_usable_non_sparse_peer_rate": float(
            cd_usable.loc[forward].mean()
        ),
        "data_quality": data_quality,
        "monthly_activity_rate": monthly_activity_rate,
        "score_distribution": score_distribution,
        "company_division_tail_diagnostic": company_division_tail_diagnostic,
        "s2_score_config": resolved_s2_score_config,
        "s2_score_config_source": (
            "caller_supplied_notebook_object"
            if s2_score_config is not None
            else "workflow_defaults"
        ),
        "peer_input_source": input_source,
        "peer_input_rows": int(len(frame)),
        "output": _output_path_metadata(output_path),
    }
    write_json(metrics, OUTPUT_CACHE / "05_peer_strategy_metrics.json")
    return metrics, output


def _waterfill_top_k(
    frame: pd.DataFrame,
    *,
    score_column: str,
    direction_column: str,
    k: int,
) -> pd.Series:
    """Allocate a monthly review capacity across directions, redistributing unused quota.

    A single high-volume direction cannot consume the whole queue before the other operational
    directions are considered.  Within each direction, score descending plus customer ID makes
    selection repeatable when scores tie.
    """

    selected = pd.Series(False, index=frame.index)
    groups = [
        "unexpected_inactivity",
        "unexpected_activity",
        "unexpected_active_pattern",
    ]
    remaining = k
    pools = {
        group: frame.loc[frame[direction_column].eq(group)].sort_values(
            [score_column, "customer_public_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        for group in groups
    }
    active = list(groups)
    while remaining > 0 and active:
        quota = max(1, remaining // len(active))
        next_active = []
        for group in active:
            pool = pools[group].loc[~selected.loc[pools[group].index]]
            take = min(quota, len(pool), remaining)
            if take:
                selected.loc[pool.head(take).index] = True
                remaining -= take
            if len(pool) > take:
                next_active.append(group)
            if remaining == 0:
                break
        if next_active == active and quota == 0:
            break
        active = next_active
    return selected


def _select_hurdle_capacity(prediction: pd.DataFrame, capacity: int) -> pd.Series:
    flags = pd.Series(False, index=prediction.index)
    eligible = prediction["hurdle_eligible"].fillna(False)
    for _, month in prediction.loc[eligible].groupby("target_month", sort=True):
        flags.loc[month.index] = _waterfill_top_k(
            month,
            score_column="hurdle_surprisal_score",
            direction_column="hurdle_direction",
            k=min(capacity, len(month)),
        )
    return flags


def _episode_table(flags: pd.DataFrame, lens: str) -> pd.DataFrame:
    """Collapse consecutive, same-direction monthly flags into customer work episodes.

    Commercial reviewers assess customer situations across time. A new episode begins after a
    monthly gap or when the signal direction changes; the hashed ID avoids exposing the internal
    customer identifier in downstream handoff material.
    """
    work = flags.loc[flags["flag"]].sort_values(
        ["customer_public_id", "transaction_month"], kind="mergesort"
    ).copy()
    month_number = _month_number(work["transaction_month"])
    gap = month_number.groupby(work["customer_public_id"]).diff()
    direction_change = work.groupby("customer_public_id")["signal_direction"].transform(
        lambda values: values.ne(values.shift())
    )
    work["new_episode"] = gap.ne(1) | direction_change
    work["episode_number"] = work.groupby("customer_public_id")[
        "new_episode"
    ].cumsum()
    summary = (
        work.groupby(["customer_public_id", "episode_number"], as_index=False)
        .agg(
            first_month=("transaction_month", "min"),
            last_month=("transaction_month", "max"),
            months_flagged=("transaction_month", "size"),
            max_score=("score", "max"),
            signal_direction=("signal_direction", "last"),
        )
    )
    summary["lens"] = lens
    summary["episode_status"] = np.where(
        summary["months_flagged"].eq(1), "new", "ongoing"
    )
    summary["episode_id"] = [
        hashlib.sha256(
            f"{lens}|{customer}|{number}".encode()
        ).hexdigest()[:16]
        for customer, number in zip(
            summary["customer_public_id"], summary["episode_number"], strict=True
        )
    ]
    return summary


def _deterministic_sample(
    frame: pd.DataFrame, count: int, seed: int
) -> pd.DataFrame:
    """Sample a stratum reproducibly so the pilot design can be audited and refreshed."""
    if len(frame) <= count:
        return frame.copy()
    return frame.sample(n=count, random_state=seed).sort_index()


def run_operational_layer() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Build fixed-capacity queues, customer episodes, and a blinded pilot sample."""

    ensure_output_cache()
    peer = pd.read_parquet(OUTPUT_CACHE / "05_peer_strategy_forward.parquet")
    hurdle = pd.read_parquet(OUTPUT_CACHE / "06_hurdle_forward_predictions.parquet")
    hurdle["target_month"] = pd.to_datetime(hurdle["target_month"])
    peer["transaction_month"] = pd.to_datetime(peer["transaction_month"])
    # Apply the same monthly capacity to every lens before comparing them.  This makes overlap an
    # operational comparison rather than an artifact of one queue simply being larger.
    for capacity in CAPACITIES:
        hurdle[f"hurdle_top_{capacity}"] = _select_hurdle_capacity(
            hurdle, capacity
        )

    merged = peer.merge(
        hurdle,
        left_on=["customer_public_id", "transaction_month"],
        right_on=["customer_public_id", "target_month"],
        how="inner",
        validate="one_to_one",
    )
    capacity_overlap = []
    for capacity in CAPACITIES:
        capacity_overlap.append(
            {
                "capacity_per_month": capacity,
                "channel_vs_hurdle_jaccard": _jaccard(
                    merged[f"channel_top_{capacity}"],
                    merged[f"hurdle_top_{capacity}"],
                ),
                "channel_rows": int(merged[f"channel_top_{capacity}"].sum()),
                "hurdle_rows": int(merged[f"hurdle_top_{capacity}"].sum()),
                "overlap_rows": int(
                    (
                        merged[f"channel_top_{capacity}"]
                        & merged[f"hurdle_top_{capacity}"]
                    ).sum()
                ),
            }
        )

    capacity = 250
    baseline_flags = merged[
        [
            "customer_public_id",
            "transaction_month",
            "channel_s2_score",
            f"channel_top_{capacity}",
        ]
    ].rename(
        columns={
            "channel_s2_score": "score",
            f"channel_top_{capacity}": "flag",
        }
    )
    baseline_flags["signal_direction"] = "baseline_deviation"
    hurdle_flags = merged[
        [
            "customer_public_id",
            "transaction_month",
            "hurdle_surprisal_score",
            f"hurdle_top_{capacity}",
            "hurdle_direction",
        ]
    ].rename(
        columns={
            "hurdle_surprisal_score": "score",
            f"hurdle_top_{capacity}": "flag",
            "hurdle_direction": "signal_direction",
        }
    )
    episodes = pd.concat(
        [
            _episode_table(baseline_flags, "s2_channel"),
            _episode_table(hurdle_flags, "hurdle_cadence"),
        ],
        ignore_index=True,
    )
    episode_path = OUTPUT_CACHE / "07_alert_episodes.parquet"
    episodes.to_parquet(episode_path, index=False)

    latest_month = merged["transaction_month"].max()
    latest = merged.loc[merged["transaction_month"].eq(latest_month)].copy()
    channel = latest[f"channel_top_{capacity}"]
    hurdle_flag = latest[f"hurdle_top_{capacity}"]
    company = latest[f"company_division_top_{capacity}"]
    hybrid = latest[f"hybrid_top_{capacity}"]
    # The pilot deliberately includes consensus, disagreement, and background cases.  That design
    # lets commercial reviewers estimate incremental value instead of validating only the easiest
    # high-score cases.
    strata = {
        "agreement_core": latest.loc[channel & hurdle_flag],
        "channel_only": latest.loc[channel & ~hurdle_flag],
        "hurdle_only": latest.loc[hurdle_flag & ~channel],
        "peer_definition_disagreement": latest.loc[
            (channel != company) | (channel != hybrid)
        ],
        "background": latest.loc[
            ~(channel | hurdle_flag | company | hybrid)
        ],
    }
    target_counts = {
        "agreement_core": 40,
        "channel_only": 60,
        "hurdle_only": 60,
        "peer_definition_disagreement": 40,
        "background": 50,
    }
    sampled_parts = []
    used = set()
    for offset, (stratum, target) in enumerate(target_counts.items()):
        pool = strata[stratum].loc[
            ~strata[stratum]["customer_public_id"].isin(used)
        ]
        sampled = _deterministic_sample(pool, target, RANDOM_STATE + offset)
        sampled = sampled.copy()
        sampled["sampling_stratum"] = stratum
        used.update(sampled["customer_public_id"].tolist())
        sampled_parts.append(sampled)
    allocation = pd.concat(sampled_parts, ignore_index=True)
    pilot_target = sum(target_counts.values())
    if len(allocation) < pilot_target:
        remaining = latest.loc[
            ~latest["customer_public_id"].isin(allocation["customer_public_id"])
        ]
        fill = _deterministic_sample(
            remaining, pilot_target - len(allocation), RANDOM_STATE + 99
        ).copy()
        fill["sampling_stratum"] = "capacity_fill"
        sampled_parts.append(fill)
        allocation = pd.concat(sampled_parts, ignore_index=True)
    # The reviewer-facing sample uses a stable case token; customer identifiers and the allocation
    # key remain in private cache files.
    allocation["case_id"] = [
        hashlib.sha256(
            f"module24|{customer}|{month:%Y-%m}".encode()
        ).hexdigest()[:16]
        for customer, month in zip(
            allocation["customer_public_id"],
            allocation["transaction_month"],
            strict=True,
        )
    ]
    filled_without_extra = sum(
        min(target_counts[name], len(pool)) for name, pool in strata.items()
    )
    inclusion_probabilities = {
        stratum: min(1.0, target_counts[stratum] / max(1, len(pool)))
        for stratum, pool in strata.items()
    }
    inclusion_probabilities["capacity_fill"] = min(
        1.0,
        max(0, pilot_target - filled_without_extra)
        / max(1, len(latest) - filled_without_extra),
    )
    allocation["inclusion_probability"] = allocation["sampling_stratum"].map(
        inclusion_probabilities
    )
    key_columns = [
        "case_id",
        "customer_public_id",
        "transaction_month",
        "sampling_stratum",
        "inclusion_probability",
        f"channel_top_{capacity}",
        f"company_division_top_{capacity}",
        f"hybrid_top_{capacity}",
        f"hurdle_top_{capacity}",
    ]
    allocation_key = allocation[key_columns].copy()
    allocation_key.to_parquet(
        OUTPUT_CACHE / "07_pilot_allocation_private.parquet", index=False
    )
    blinded = allocation[
        [
            "case_id",
            "transaction_month",
            "hurdle_direction",
        ]
    ].copy()
    blinded["review_outcome"] = ""
    blinded["review_effort_minutes"] = np.nan
    blinded["reviewer_notes"] = ""
    blinded_path = OUTPUT_CACHE / "07_pilot_sample_blinded.csv"
    blinded.to_csv(blinded_path, index=False)

    metrics = {
        "capacity_overlap": capacity_overlap,
        "episode_capacity_per_month": capacity,
        "episode_counts": (
            episodes.groupby(["lens", "episode_status"]).size().rename("episodes")
            .reset_index().to_dict(orient="records")
        ),
        "repeat_customer_months_avoided": {
            lens: int(
                group["months_flagged"].sum() - len(group)
            )
            for lens, group in episodes.groupby("lens")
        },
        "latest_month": latest_month,
        "pilot_rows": int(len(blinded)),
        "pilot_strata": allocation["sampling_stratum"].value_counts().to_dict(),
        "canonical_review_outcomes": [
            "actionable_new",
            "relevant_already_known",
            "real_but_not_actionable",
            "false_or_unhelpful",
            "insufficient_context",
        ],
        "blinded_sample": _output_path_metadata(blinded_path),
        "private_allocation": _output_path_metadata(
            OUTPUT_CACHE / "07_pilot_allocation_private.parquet"
        ),
        "episodes": _output_path_metadata(episode_path),
    }
    write_json(metrics, OUTPUT_CACHE / "07_operational_metrics.json")
    return metrics, episodes, blinded


def compile_evaluation() -> dict:
    """Combine every generated metric into one versioned evidence record."""

    peer = read_json(OUTPUT_CACHE / "05_peer_strategy_metrics.json")
    hurdle = read_json(OUTPUT_CACHE / "06_hurdle_metrics.json")
    operational = read_json(OUTPUT_CACHE / "07_operational_metrics.json")
    selected_hurdle = next(
        row
        for row in hurdle["forward_metrics"]
        if row["model"] == hurdle["operational_activity_model"]
    )
    evaluation = {
        "data_package": {
            "source": "confidential_customer_month_panel",
            "customer_month_rows": 564_144,
            "customers": 17_128,
            "model_end_month": "2026-06",
        },
        "workflow": {
            "main_notebooks": 4,
            "operational_lenses": ["S2 channel baseline", "hurdle cadence challenger"],
            "activity_model": hurdle["operational_activity_model"],
            "activity_forward": selected_hurdle,
            "active_pattern_forward": hurdle["pattern_metrics"],
        },
        "peer_strategy": peer,
        "operational_layer": operational,
        "claim_boundary": (
            "Technical and self-supervised evaluation only. Business precision, novelty, "
            "actionability, adoption, and financial value remain unvalidated until the "
            "blinded commercial review is completed."
        ),
    }
    write_json(evaluation, OUTPUT_CACHE / "module24_evaluation.json")
    return evaluation


def metric_table(records: Iterable[dict], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(records)[columns]
