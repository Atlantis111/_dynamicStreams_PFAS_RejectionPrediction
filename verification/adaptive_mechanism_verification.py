"""Mechanism-oriented verification of adaptive PFOS/PFBS prediction.

This experiment is deliberately separate from the primary dynamic comparison.
It tests whether an adaptive gain is attributable to causal label updates and
chronological structure rather than only to the fixed 50/50 split or a time
proxy.  All conditions use the same sequence, initial history, future rows,
batch boundaries, preprocessing, and XGBoost parameters.

For every future batch the protocol is:

    predict -> score pre-update -> reveal labels -> detect -> update

The primary comparison is made on the next unseen batch.  Same-batch
post-update scores are retained only as diagnostics and are never used as the
prospective performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover
    raise ImportError("xgboost is required for this verification experiment.") from exc


ROOT = Path(__file__).resolve().parents[2]
FEATURE_SCHEMA_ROOT = ROOT / "new" / "_4_dynamicFrame"
if str(FEATURE_SCHEMA_ROOT) not in sys.path:
    sys.path.insert(0, str(FEATURE_SCHEMA_ROOT))

from dynamic_feature_schema import (
    DYNAMIC_FEATURE_COLUMNS,
    SEQUENCE_COLUMN as SHARED_SEQUENCE_COLUMN,
    TARGET_COLUMN as SHARED_TARGET_COLUMN,
    TIME_COLUMN as SHARED_TIME_COLUMN,
    validate_dynamic_columns,
)

DEFAULT_DATA = ROOT / "new" / "FPAS_Stream.xlsx"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"
TARGET = SHARED_TARGET_COLUMN
SEQUENCE = SHARED_SEQUENCE_COLUMN
TIME = SHARED_TIME_COLUMN
ALL_FEATURES = list(DYNAMIC_FEATURE_COLUMNS)
CONDITIONS = [
    "Frozen-static",
    "Periodic-only",
    "Error-drift-only",
    "PSI-drift-only",
    "AP-XGBoost",
    "APSI-XGBoost",
]

XGB_PARAMS = {
    "n_estimators": 120,
    "max_depth": 3,
    "min_child_weight": 2,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "objective": "reg:squarederror",
    "n_jobs": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(actual: Iterable[float], predicted: Iterable[float]) -> Dict[str, float]:
    y = np.asarray(list(actual), dtype=float)
    p = np.asarray(list(predicted), dtype=float)
    mse = float(mean_squared_error(y, p))
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y, p)),
        "R2": float("nan") if len(y) < 2 or np.isclose(np.var(y), 0.0) else float(r2_score(y, p)),
    }


class HistoryPreprocessor:
    """Preprocessor fitted once on the initial history only."""

    def __init__(self, features: List[str]) -> None:
        self.features = features
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()

    def fit(self, history: pd.DataFrame) -> None:
        self.scaler.fit(self.imputer.fit_transform(history[self.features]))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(self.imputer.transform(frame[self.features]))

    def manifest(self) -> Dict[str, Any]:
        return {
            "features": self.features,
            "imputer": "median fitted on initial history only",
            "imputer_statistics": dict(zip(self.features, self.imputer.statistics_)),
            "scaler": "standard scaler fitted on initial history only",
            "scaler_mean": dict(zip(self.features, self.scaler.mean_)),
            "scaler_scale": dict(zip(self.features, self.scaler.scale_)),
        }


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, engine="openpyxl")
    validate_dynamic_columns(frame.columns)
    frame = frame.copy()
    frame["source_row"] = np.arange(1, len(frame) + 1)
    frame[SEQUENCE] = frame[SEQUENCE].astype(str).str.strip()
    if frame[SEQUENCE].eq("").any() or frame[SEQUENCE].eq("nan").any():
        raise ValueError("Every dynamic record must have a sequence_id.")
    for column in ALL_FEATURES + [TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[[TARGET, TIME]].isna().any().any():
        raise ValueError("Target and elapsed_time_min may not contain missing values.")
    frame = frame.sort_values([SEQUENCE, TIME, "source_row"], kind="mergesort").reset_index(drop=True)
    frame["chronological_order"] = frame.groupby(SEQUENCE, sort=False).cumcount() + 1
    return frame


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 8) -> float:
    values: List[float] = []
    for index in range(reference.shape[1]):
        baseline, observed = reference[:, index], current[:, index]
        if np.isclose(np.std(baseline), 0.0):
            continue
        edges = np.unique(np.quantile(baseline, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        base_share = np.histogram(baseline, bins=edges)[0] / len(baseline)
        current_share = np.histogram(observed, bins=edges)[0] / len(observed)
        base_share = np.clip(base_share, 1e-6, None)
        current_share = np.clip(current_share, 1e-6, None)
        values.append(float(np.sum((current_share - base_share) * np.log(current_share / base_share))))
    return float(np.mean(values)) if values else 0.0


def relative_error_drift(current_rmse: float, history: Sequence[float], threshold: float) -> Tuple[bool, float | None]:
    if not history:
        return False, None
    baseline = float(np.median(history[-3:]))
    change = (current_rmse - baseline) / max(abs(baseline), 1e-8)
    return bool(change > threshold), float(change)


def fit_model(X: np.ndarray, y: np.ndarray, seed: int, previous: Any = None) -> XGBRegressor:
    model = XGBRegressor(random_state=seed, **XGB_PARAMS)
    if previous is None:
        model.fit(X, y)
    else:
        model.fit(X, y, xgb_model=previous.get_booster())
    return model


def shuffled_future(future: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(future))
    return future.iloc[order].reset_index(drop=True)


def shuffled_labels(values: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(values) < 2:
        return values.copy()
    result = values.copy()
    # Force a non-identity permutation whenever possible.
    for _ in range(10):
        candidate = rng.permutation(values)
        if not np.array_equal(candidate, values):
            return candidate
    return result[::-1]


def evaluate_condition(
    condition: str,
    sequence_id: str,
    initial: pd.DataFrame,
    future: pd.DataFrame,
    features: List[str],
    preprocessor: HistoryPreprocessor,
    frozen_predictions: Dict[int, List[float]],
    seed: int,
    batch_size: int,
    window_size: int,
    retrain_interval: int,
    error_threshold: float,
    psi_threshold: float,
    label_seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    initial_x = preprocessor.transform(initial)
    model = fit_model(initial_x, initial[TARGET].to_numpy(float), seed)
    window_x = initial_x.copy()
    window_y = initial[TARGET].to_numpy(float).copy()
    reference_x = initial_x.copy()
    error_history: List[float] = []
    rows: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    previous_update = "initial_fit"

    for batch_number, start in enumerate(range(0, len(future), batch_size), start=1):
        batch = future.iloc[start : start + batch_size].copy().reset_index(drop=True)
        batch_x = preprocessor.transform(batch)
        actual = batch[TARGET].to_numpy(float)
        prediction = model.predict(batch_x)
        pre = metrics(actual, prediction)
        frozen = np.asarray(frozen_predictions[batch_number], dtype=float)
        frozen_metric = metrics(actual, frozen)

        error_drift, error_value = relative_error_drift(pre["RMSE"], error_history, error_threshold)
        psi_value = psi(reference_x, batch_x)
        psi_drift = psi_value > psi_threshold
        if condition in {"Error-drift-only", "AP-XGBoost", "Wrong-label-AP"}:
            drift, detector, drift_value = error_drift, "relative_prequential_RMSE", error_value
        elif condition in {"PSI-drift-only", "APSI-XGBoost"}:
            drift, detector, drift_value = psi_drift, "PSI_initial_history", psi_value
        else:
            drift, detector, drift_value = False, "not_applicable", None

        # The current labels are revealed only after the pre-update score.
        update_type = "no_update"
        used_y = actual.copy()
        if condition == "Wrong-label-AP":
            used_y = shuffled_labels(actual, label_seed + batch_number)
        window_x = np.vstack([window_x, batch_x])[-window_size:]
        window_y = np.concatenate([window_y, used_y])[-window_size:]

        periodic = batch_number % retrain_interval == 0
        if condition == "Frozen-static":
            update_type = "no_update"
        elif condition == "Periodic-only":
            if periodic:
                model = fit_model(window_x, window_y, seed)
                update_type = "full_retrain_periodic"
        elif condition == "Error-drift-only":
            if error_drift:
                model = fit_model(window_x, window_y, seed)
                update_type = "full_retrain_error_drift"
        elif condition == "PSI-drift-only":
            if psi_drift:
                model = fit_model(window_x, window_y, seed)
                update_type = "full_retrain_psi_drift"
        elif condition in {"AP-XGBoost", "Wrong-label-AP"}:
            if error_drift:
                model = fit_model(window_x, window_y, seed)
                update_type = "full_retrain_error_drift"
            elif periodic:
                model = fit_model(window_x, window_y, seed)
                update_type = "full_retrain_periodic"
            else:
                model = fit_model(batch_x, used_y, seed, previous=model)
                update_type = "warm_start"
        elif condition == "APSI-XGBoost":
            if psi_drift:
                model = fit_model(window_x, window_y, seed)
                update_type = "full_retrain_psi_drift"
            elif periodic:
                model = fit_model(window_x, window_y, seed)
                update_type = "full_retrain_periodic"
            else:
                model = fit_model(batch_x, used_y, seed, previous=model)
                update_type = "warm_start"
        else:
            raise ValueError(f"Unknown condition: {condition}")

        post = metrics(actual, model.predict(batch_x))
        row = {
            "condition": condition,
            "sequence_id": sequence_id,
            "seed": seed,
            "batch": batch_number,
            "n_batch": len(batch),
            "source_rows": ";".join(str(int(value)) for value in batch["source_row"]),
            "elapsed_time_start": float(batch[TIME].min()),
            "elapsed_time_end": float(batch[TIME].max()),
            "pre_MSE": pre["MSE"], "pre_RMSE": pre["RMSE"], "pre_MAE": pre["MAE"], "pre_R2": pre["R2"],
            "frozen_MAE": frozen_metric["MAE"],
            "gain_vs_frozen_MAE": frozen_metric["MAE"] - pre["MAE"],
            "post_MAE_diagnostic": post["MAE"],
            "post_RMSE_diagnostic": post["RMSE"],
            "error_drift": bool(error_drift), "error_drift_value": error_value,
            "psi_drift": bool(psi_drift), "psi_value": psi_value,
            "drift_detected": bool(drift), "drift_detector": detector, "drift_value": drift_value,
            "update_type": update_type, "previous_update_type": previous_update,
            "labels_used_for_update": "permuted" if condition == "Wrong-label-AP" else "true",
            "window_size_after_update": len(window_y),
        }
        rows.append(row)
        events.append({key: row[key] for key in ["condition", "sequence_id", "seed", "batch", "elapsed_time_start", "elapsed_time_end", "error_drift", "error_drift_value", "psi_drift", "psi_value", "drift_detected", "drift_detector", "drift_value", "update_type"]})
        error_history.append(pre["RMSE"])
        previous_update = update_type
    return rows, events


def bootstrap_mean(values: Sequence[float], seed: int, repetitions: int = 2000) -> Tuple[float, float]:
    values = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize(batch: pd.DataFrame, bootstrap_repetitions: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_cols = ["feature_mode", "order_mode", "label_mode", "sequence_id", "condition"]
    summary_rows: List[Dict[str, Any]] = []
    ci_rows: List[Dict[str, Any]] = []
    for keys, group in batch.groupby(group_cols, sort=False):
        values = dict(zip(group_cols, keys))
        row: Dict[str, Any] = {**values, "n_rows": len(group), "n_seeds": group["seed"].nunique()}
        for metric in ["pre_MAE", "pre_RMSE", "pre_R2", "gain_vs_frozen_MAE"]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else float("nan")
        row["updates_mean"] = float(group["update_type"].ne("no_update").mean())
        row["drifts_mean"] = float(group["drift_detected"].mean())
        summary_rows.append(row)
        ci = {**values, "n_rows": len(group), "bootstrap_repetitions": bootstrap_repetitions}
        for metric in ["pre_MAE", "pre_RMSE", "pre_R2", "gain_vs_frozen_MAE"]:
            low, high = bootstrap_mean(group[metric].to_numpy(float), 2024 + len(ci_rows) + len(metric), bootstrap_repetitions)
            ci[f"{metric}_ci95_low"], ci[f"{metric}_ci95_high"] = low, high
        ci_rows.append(ci)

    # The next-batch table excludes batch 1 because no preceding label update
    # can affect it. It is the key prospective learning diagnostic.
    next_batch = batch[batch["batch"] > 1].copy()
    next_batch["prior_update"] = next_batch["previous_update_type"].ne("no_update")
    return pd.DataFrame(summary_rows), pd.DataFrame(ci_rows), next_batch


def paired_contrasts(batch: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    """Calculate paired batch contrasts, keeping sequence/seed/batch aligned."""
    rows: List[Dict[str, Any]] = []
    base_keys = ["feature_mode", "order_mode", "sequence_id", "seed", "batch"]

    def add_contrast(name: str, left: pd.DataFrame, right: pd.DataFrame, keys: List[str]) -> None:
        joined = left.merge(right, on=keys, suffixes=("_left", "_right"), how="inner")
        for metric in ["pre_MAE", "pre_RMSE"]:
            values = (joined[f"{metric}_right"] - joined[f"{metric}_left"]).to_numpy(float)
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                continue
            low, high = bootstrap_mean(finite, 6100 + len(rows), repetitions)
            record = {"contrast": name, "metric": metric, "n_pairs": len(finite), "mean_difference_right_minus_left": float(np.mean(finite)), "ci95_low": low, "ci95_high": high}
            for key in ["feature_mode", "order_mode", "sequence_id"]:
                if key in joined and joined[key].nunique() == 1:
                    record[key] = joined[key].iloc[0]
            rows.append(record)

    normal = batch[batch["label_mode"] == "true_labels"]
    chrono = normal[normal["order_mode"] == "chronological"]
    for sequence_id in chrono["sequence_id"].unique():
        subset = chrono[chrono["sequence_id"] == sequence_id]
        for feature_mode in subset["feature_mode"].unique():
            scoped = subset[subset["feature_mode"] == feature_mode]
            frozen = scoped[scoped["condition"] == "Frozen-static"]
            for condition in ["Periodic-only", "Error-drift-only", "PSI-drift-only", "AP-XGBoost", "APSI-XGBoost"]:
                add_contrast(f"{condition}_vs_Frozen-static", frozen, scoped[scoped["condition"] == condition], base_keys)

    # The shuffled-order and time-free controls use the same seed/batch keys,
    # but order repetitions are averaged first to avoid pseudo-replication.
    for sequence_id in normal["sequence_id"].unique():
        for condition in CONDITIONS:
            for metric in ["pre_MAE", "pre_RMSE"]:
                left = normal[(normal["sequence_id"] == sequence_id) & (normal["condition"] == condition) & (normal["feature_mode"] == "all_dynamic") & (normal["order_mode"] == "chronological")].groupby(["seed", "batch"], as_index=False)[metric].mean()
                right = normal[(normal["sequence_id"] == sequence_id) & (normal["condition"] == condition) & (normal["feature_mode"] == "all_dynamic") & (normal["order_mode"] == "future_shuffled")].groupby(["seed", "batch"], as_index=False)[metric].mean()
                joined = left.merge(right, on=["seed", "batch"], suffixes=("_chrono", "_shuffled"), how="inner")
                values = (joined[f"{metric}_shuffled"] - joined[f"{metric}_chrono"]).to_numpy(float)
                if len(values):
                    low, high = bootstrap_mean(values, 7200 + len(rows), repetitions)
                    rows.append({"contrast": f"future_shuffled_minus_chronological_{condition}", "metric": metric, "sequence_id": sequence_id, "n_pairs": len(values), "mean_difference_right_minus_left": float(np.mean(values)), "ci95_low": low, "ci95_high": high})

        true_ap = normal[(normal["sequence_id"] == sequence_id) & (normal["condition"] == "AP-XGBoost") & (normal["feature_mode"] == "all_dynamic") & (normal["order_mode"] == "chronological")]
        wrong_ap = batch[(batch["sequence_id"] == sequence_id) & (batch["condition"] == "Wrong-label-AP") & (batch["feature_mode"] == "all_dynamic") & (batch["order_mode"] == "chronological")]
        add_contrast("Wrong-label-AP_minus_true_AP", true_ap, wrong_ap, base_keys)

    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    data_path = args.data.resolve()
    data = load_data(data_path)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    all_events: List[Dict[str, Any]] = []
    preprocessing: Dict[str, Any] = {}
    manifest_rows: List[Dict[str, Any]] = []
    sequences = list(data.groupby(SEQUENCE, sort=False))

    for sequence_id, sequence in sequences:
        sequence = sequence.reset_index(drop=True)
        initial_n = min(max(int(np.ceil(len(sequence) * args.initial_fraction)), 8), len(sequence) - 1)
        initial = sequence.iloc[:initial_n].copy().reset_index(drop=True)
        chronological_future = sequence.iloc[initial_n:].copy().reset_index(drop=True)
        for order_mode in ["chronological", "future_shuffled"]:
            order_repetitions = 1 if order_mode == "chronological" else args.order_permutations
            for order_rep in range(order_repetitions):
                for feature_mode in ["all_dynamic", "time_free"]:
                    features = ALL_FEATURES if feature_mode == "all_dynamic" else [
                        value for value in ALL_FEATURES if value != "Measurement time (min)"
                    ]
                    preprocessor = HistoryPreprocessor(features)
                    preprocessor.fit(initial)
                    preprocessing[f"{sequence_id}|{order_mode}|{order_rep}|{feature_mode}"] = preprocessor.manifest()
                    future = chronological_future if order_mode == "chronological" else shuffled_future(chronological_future, args.random_seed + order_rep)
                    for _, row in sequence.iterrows():
                        phase = "initial" if int(row["chronological_order"]) <= initial_n else "future"
                        manifest_rows.append({"sequence_id": sequence_id, "source_row": int(row["source_row"]), "chronological_order": int(row["chronological_order"]), "phase": phase, "order_mode": order_mode, "order_repetition": order_rep, "feature_mode": feature_mode})

                    # Frozen predictions are generated once per seed and reused
                    # by every condition, guaranteeing paired test rows.
                    for seed in args.seeds:
                        initial_x = preprocessor.transform(initial)
                        frozen_model = fit_model(initial_x, initial[TARGET].to_numpy(float), seed)
                        frozen_predictions = {
                            batch_number: frozen_model.predict(preprocessor.transform(future.iloc[start : start + args.batch_size])) .tolist()
                            for batch_number, start in enumerate(range(0, len(future), args.batch_size), start=1)
                        }
                        for label_mode in ["true_labels", "permuted_update_labels"]:
                            selected_conditions = CONDITIONS if label_mode == "true_labels" else ["Wrong-label-AP"]
                            for condition in selected_conditions:
                                effective_condition = condition
                                label_seed = args.random_seed + seed + 1000 * order_rep
                                rows, events = evaluate_condition(
                                    effective_condition,
                                    str(sequence_id),
                                    initial,
                                    future,
                                    features,
                                    preprocessor,
                                    frozen_predictions,
                                    seed,
                                    args.batch_size,
                                    args.window_size,
                                    args.retrain_interval,
                                    args.error_drift_threshold,
                                    args.psi_threshold,
                                    label_seed,
                                )
                                for row in rows:
                                    row.update({"feature_mode": feature_mode, "order_mode": order_mode, "order_repetition": order_rep, "label_mode": label_mode})
                                for event in events:
                                    event.update({"feature_mode": feature_mode, "order_mode": order_mode, "order_repetition": order_rep, "label_mode": label_mode})
                                all_rows.extend(rows)
                                all_events.extend(events)

    batch = pd.DataFrame(all_rows)
    events = pd.DataFrame(all_events)
    summary, ci, next_batch = summarize(batch, args.bootstrap_repetitions)
    contrasts = paired_contrasts(batch, args.bootstrap_repetitions)
    batch.to_csv(output / "verification_batch_results.csv", index=False, encoding="utf-8-sig")
    events.to_csv(output / "drift_detection_events.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "verification_summary.csv", index=False, encoding="utf-8-sig")
    ci.to_csv(output / "verification_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    next_batch.to_csv(output / "next_batch_gain.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(output / "verification_contrasts.csv", index=False, encoding="utf-8-sig")
    batch[batch["order_mode"] == "future_shuffled"].to_csv(output / "order_shuffle_control.csv", index=False, encoding="utf-8-sig")
    batch[batch["label_mode"] == "permuted_update_labels"].to_csv(output / "label_permutation_control.csv", index=False, encoding="utf-8-sig")
    batch[batch["feature_mode"] == "time_free"].to_csv(output / "feature_ablation_time_free.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(manifest_rows).to_csv(output / "verification_stream_manifest.csv", index=False, encoding="utf-8-sig")
    (output / "preprocessing_manifest.json").write_text(json.dumps(preprocessing, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    metadata = {
        "input": str(data_path), "input_sha256": sha256_file(data_path), "n_rows": len(data),
        "sequences": data[SEQUENCE].value_counts().to_dict(), "features": ALL_FEATURES,
        "initial_fraction": args.initial_fraction, "batch_size": args.batch_size,
        "window_size": args.window_size, "retrain_interval": args.retrain_interval,
        "seeds": args.seeds, "conditions": CONDITIONS,
        "protocol": "predict -> pre-update score -> reveal labels -> drift detection -> update",
        "primary_learning_diagnostic": "next-batch performance after a preceding update, not same-batch post-update metrics",
        "paired_contrasts": "batch-aligned MAE/RMSE differences with bootstrap intervals",
        "preprocessing": "median imputation and standard scaling fitted on initial history only",
        "controls": {
            "order_shuffle_repetitions": args.order_permutations,
            "wrong_label_update": "same AP-XGBoost update rule with permuted labels",
            "time_free_features": [value for value in ALL_FEATURES if value != "Measurement time (min)"],
        },
        "bootstrap": {"repetitions": args.bootstrap_repetitions, "unit": "seed/replicate-level rows"},
    }
    (output / "verification_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(summary.round(4).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify whether dynamic adaptive gains arise from causal online learning.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--initial-fraction", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--retrain-interval", type=int, default=2)
    parser.add_argument("--error-drift-threshold", type=float, default=0.2)
    parser.add_argument("--psi-threshold", type=float, default=0.2)
    parser.add_argument("--order-permutations", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=2024)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    if not 0.0 < args.initial_fraction < 1.0 or args.batch_size < 1 or args.window_size < 2:
        parser.error("Invalid initial fraction, batch size, or window size.")
    if args.order_permutations < 1 or args.bootstrap_repetitions < 100:
        parser.error("order-permutations must be >=1 and bootstrap-repetitions must be >=100.")
    run(args)


if __name__ == "__main__":
    main()
