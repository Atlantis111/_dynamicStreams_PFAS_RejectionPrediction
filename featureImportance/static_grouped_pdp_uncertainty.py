"""Generate uncertainty-aware PDPs for the static literature dataset.

The legacy PDPs used one fitted model and omitted where the 151 observations
support each curve.  This script fits the same nested leave-one-reference-out
XGBoost procedure used for the revised static evaluation.  For each outer fold
and each seed, it calculates PDPs from that fold's training records only.
Random seeds are averaged within a fold before uncertainty is estimated.
The displayed 95% percentile intervals bootstrap the eight outer-fold curves;
they describe refit sensitivity across the observed literature studies, not a
causal effect or independent physical replication.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV


ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = ROOT / "_2_staticModels"
if str(STATIC_ROOT) not in sys.path:
    sys.path.insert(0, str(STATIC_ROOT))

from dataset_schema import FEATURE_COLUMNS
from static_grouped_evaluation import _inner_cv, _outer_splits, load_dataset, make_groups, model_pipeline


DEFAULT_DATA = ROOT / "PFAS.xlsx"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs_static_grouped_pdp"


def selected_features() -> List[str]:
    """Resolve the three legacy Figure 11 variables from the current schema."""

    contact_angle = next(column for column in FEATURE_COLUMNS if column.startswith("MB contact angle"))
    return ["rs/rp", "pH", contact_angle]


def common_grids(frame: pd.DataFrame, features: List[str], grid_size: int) -> Dict[str, np.ndarray]:
    return {
        feature: np.linspace(float(frame[feature].min()), float(frame[feature].max()), grid_size)
        for feature in features
    }


def calculate_curves(frame: pd.DataFrame, groups: pd.Series, features: List[str], grids: Dict[str, np.ndarray], seeds: List[int]) -> pd.DataFrame:
    rows = []
    X = frame[FEATURE_COLUMNS]
    y = frame["removal rate (%)"]
    for fold, (train_idx, test_idx) in enumerate(_outer_splits(groups), start=1):
        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx]
        train_groups = groups.iloc[train_idx]
        held_out_group = str(groups.iloc[test_idx].iloc[0])
        for seed in seeds:
            estimator, grid = model_pipeline("xgboost", seed)
            search = GridSearchCV(
                estimator=estimator, param_grid=grid, scoring="neg_mean_squared_error",
                cv=_inner_cv(train_groups), n_jobs=1, refit=True, error_score="raise",
            )
            search.fit(X_train, y_train, groups=train_groups)
            model = search.best_estimator_
            for feature in features:
                for grid_index, value in enumerate(grids[feature]):
                    reference = X_train.copy()
                    reference.loc[:, feature] = value
                    rows.append({
                        "fold": fold, "held_out_reference": held_out_group, "seed": seed,
                        "feature": feature, "grid_index": grid_index, "feature_value": float(value),
                        "pdp_prediction": float(np.mean(model.predict(reference))),
                        "n_training_records": int(len(train_idx)), "n_held_out_records": int(len(test_idx)),
                        "best_params": json.dumps(search.best_params_, sort_keys=True),
                    })
            print(f"Completed outer fold {fold}/{groups.nunique()}, seed {seed}.")
    return pd.DataFrame(rows)


def bootstrap_fold_curves(seed_curves: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260906)
    rows = []
    for (feature, grid_index), group in seed_curves.groupby(["feature", "grid_index"], sort=False):
        values = group["pdp_prediction"].to_numpy(float)
        draws = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
        rows.append({
            "feature": feature, "grid_index": int(grid_index), "feature_value": float(group["feature_value"].iloc[0]),
            "mean_pdp_prediction": float(values.mean()), "sd_across_outer_folds": float(values.std(ddof=1)),
            "ci95_low": float(np.quantile(draws, 0.025)), "ci95_high": float(np.quantile(draws, 0.975)),
            "n_outer_folds": int(group["fold"].nunique()), "n_seeds_per_fold": int(group["n_seed_fits"].iloc[0]),
            "uncertainty_basis": "percentile bootstrap over seed-averaged leave-one-reference-out curves",
        })
    return pd.DataFrame(rows)


def density_table(frame: pd.DataFrame, features: List[str], bins: int) -> pd.DataFrame:
    rows = []
    for feature in features:
        values = frame[feature].dropna().to_numpy(float)
        counts, edges = np.histogram(values, bins=bins)
        for index, count in enumerate(counts):
            rows.append({
                "feature": feature, "bin": index + 1, "bin_low": float(edges[index]), "bin_high": float(edges[index + 1]),
                "count": int(count), "proportion": float(count / len(values)), "n_observed": int(len(values)),
            })
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, density: pd.DataFrame, output: Path) -> None:
    features = selected_features()
    figure = plt.figure(figsize=(14, 6.2))
    outer = figure.add_gridspec(2, len(features), height_ratios=[4.0, 1.0], hspace=0.08, wspace=0.32)
    for index, feature in enumerate(features):
        axis = figure.add_subplot(outer[0, index])
        density_axis = figure.add_subplot(outer[1, index], sharex=axis)
        curve = summary.loc[summary["feature"] == feature].sort_values("feature_value")
        axis.plot(curve["feature_value"], curve["mean_pdp_prediction"], color="#2563A6", linewidth=2.2)
        axis.fill_between(curve["feature_value"], curve["ci95_low"], curve["ci95_high"], color="#2563A6", alpha=0.20, linewidth=0)
        axis.set_title(f"({chr(97 + index)}) {feature}", loc="left", fontweight="bold")
        axis.set_ylabel("Partial dependence\n(predicted rejection, %)")
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(labelbottom=False)
        observed = density.loc[density["feature"] == feature].sort_values("bin")
        widths = observed["bin_high"].to_numpy(float) - observed["bin_low"].to_numpy(float)
        density_axis.bar(observed["bin_low"], observed["count"], width=widths, align="edge", color="#8DA8C7", edgecolor="white", linewidth=0.4)
        density_axis.set_xlabel(feature)
        density_axis.set_ylabel("Count\n(n=151)")
        density_axis.spines[["top", "right"]].set_visible(False)
    figure.text(0.5, 0.985, "Line: mean leave-one-reference-out PDP; band: 95% fold-bootstrap interval", ha="center", va="top", fontsize=10)
    figure.savefig(output / "static_grouped_pdp_with_ci_density.png", dpi=400, bbox_inches="tight")
    figure.savefig(output / "static_grouped_pdp_with_ci_density.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create uncertainty-aware static PDPs using grouped outer validation.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--grid-size", type=int, default=50)
    parser.add_argument("--density-bins", type=int, default=20)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame, _ = load_dataset(args.data.resolve())
    groups = make_groups(frame, "reference")
    features = selected_features()
    grids = common_grids(frame, features, args.grid_size)
    raw_curves = calculate_curves(frame, groups, features, grids, args.seeds)
    seed_curves = (
        raw_curves.groupby(["fold", "held_out_reference", "feature", "grid_index", "feature_value"], as_index=False)
        .agg(pdp_prediction=("pdp_prediction", "mean"), n_seed_fits=("seed", "nunique"), n_training_records=("n_training_records", "first"), n_held_out_records=("n_held_out_records", "first"))
    )
    summary = bootstrap_fold_curves(seed_curves, args.bootstrap_repetitions)
    density = density_table(frame, features, args.density_bins)
    raw_curves.to_csv(output / "pdp_outer_fold_seed_curves.csv", index=False, encoding="utf-8-sig")
    seed_curves.to_csv(output / "pdp_outer_fold_seed_mean_curves.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "pdp_curve_summary_with_ci.csv", index=False, encoding="utf-8-sig")
    density.to_csv(output / "pdp_feature_density.csv", index=False, encoding="utf-8-sig")
    plot(summary, density, output)
    metadata = {
        "data": str(args.data.resolve()), "n_records": int(len(frame)), "n_reference_groups": int(groups.nunique()),
        "outer_validation": "leave-one-reference-out", "inner_tuning": "grouped cross-validation within outer training data",
        "seeds": args.seeds, "features": features, "grid_size": args.grid_size, "density_bins": args.density_bins,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "uncertainty_scope": "fold-bootstrap refit variability, not an independent experimental or causal confidence interval",
        "pdp_background": "outer-training records only; no held-out group used to fit an outer-fold model or calculate its PDP background",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote uncertainty-aware static PDP outputs to", output)


if __name__ == "__main__":
    main()
