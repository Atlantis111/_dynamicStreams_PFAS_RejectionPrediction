"""Run the grouped-validation suite required for reproducibility review.

Static validation is performed separately for four leakage-control grouping
units: publication/reference, membrane, PFAS, and experimental scenario.  The
implementation delegates the actual nested, leakage-free fitting to
``new/_2_staticModels/static_grouped_evaluation.py`` so that all three static
regressors use the same preprocessing and outer test groups.

The dynamic workbook is audited by experimental sequence.  A genuine
leave-one-sequence-out estimate is reported only when at least two independent
sequences exist for the same compound.  The current workbook has one PFOS
sequence and one PFBS sequence, so cross-sequence generalisation is marked
not estimable rather than being replaced by an invalid cross-compound split.

Example (formal run):
    python grouped_validation_suite.py --model all --seeds 42 43 44 45 46

Use ``--group-by`` to run one static scheme, or ``--skip-static``/``--skip-
dynamic-audit`` for focused checks.  Each run writes a timestamp-free,
reproducible output directory containing the per-scheme outputs and suite
manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = ROOT / "new" / "_2_staticModels"
if str(STATIC_DIR) not in sys.path:
    sys.path.insert(0, str(STATIC_DIR))

import static_grouped_evaluation as static_eval  # noqa: E402


DEFAULT_STATIC_DATA = ROOT / "new" / "PFAS.xlsx"
DEFAULT_DYNAMIC_DATA = ROOT / "new" / "FPAS_Stream.xlsx"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs_grouped_suite"
GROUP_SCHEMES = ("reference", "membrane", "pfas", "scenario")
MODEL_NAMES = ("knn", "rf", "xgboost")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_static_suite(
    data_path: Path,
    output_dir: Path,
    schemes: Sequence[str],
    models: List[str],
    seeds: Sequence[int],
    bootstrap_repetitions: int,
) -> pd.DataFrame:
    """Run all requested static schemes and aggregate their summaries."""

    summary_rows: List[pd.DataFrame] = []
    ci_rows: List[pd.DataFrame] = []
    for scheme in schemes:
        scheme_dir = output_dir / f"static_{scheme}"
        print(f"\n=== Static grouped validation: {scheme} ===")
        static_eval.run(
            data_path=data_path,
            output_dir=scheme_dir,
            group_by=scheme,
            model_names=models,
            seeds=seeds,
            bootstrap_repetitions=bootstrap_repetitions,
        )
        summary_file = scheme_dir / "static_summary.csv"
        ci_file = scheme_dir / "static_bootstrap_ci.csv"
        summary = pd.read_csv(summary_file)
        summary.insert(0, "group_by", scheme)
        summary_rows.append(summary)
        if ci_file.exists():
            ci = pd.read_csv(ci_file)
            ci.insert(0, "group_by", scheme)
            ci_rows.append(ci)

    combined_summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    combined_ci = pd.concat(ci_rows, ignore_index=True) if ci_rows else pd.DataFrame()
    combined_summary.to_csv(output_dir / "static_grouped_validation_summary.csv", index=False, encoding="utf-8-sig")
    combined_ci.to_csv(output_dir / "static_grouped_validation_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    return combined_summary


def audit_dynamic_sequences(data_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Write a transparent sequence-level feasibility audit for the dynamic data.

    A leave-one-sequence-out test requires at least two independent sequences
    within the same target population.  PFOS and PFBS are separate compounds,
    so treating one as the other's held-out sequence would confound compound
    extrapolation with sequence extrapolation and is deliberately prohibited.
    """

    frame = pd.read_excel(data_path, engine="openpyxl")
    required = {"sequence_id", "elapsed_time_min", "removal rate (%)"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dynamic workbook is missing required columns: {missing}")

    frame = frame.copy()
    frame["elapsed_time_min"] = pd.to_numeric(frame["elapsed_time_min"], errors="coerce")
    frame["removal rate (%)"] = pd.to_numeric(frame["removal rate (%)"], errors="coerce")
    frame = frame.dropna(subset=["sequence_id", "elapsed_time_min", "removal rate (%)"])
    frame = frame.sort_values(["sequence_id", "elapsed_time_min", "number" if "number" in frame else "sequence_id"]).reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    for sequence_id, sequence in frame.groupby("sequence_id", sort=False):
        compound = "PFOS" if "PFOS" in str(sequence_id).upper() else "PFBS" if "PFBS" in str(sequence_id).upper() else "unknown"
        rows.append({
            "sequence_id": sequence_id,
            "compound": compound,
            "n_records": len(sequence),
            "time_start_min": float(sequence["elapsed_time_min"].min()),
            "time_end_min": float(sequence["elapsed_time_min"].max()),
            "n_unique_times": int(sequence["elapsed_time_min"].nunique()),
            "sequence_validation": "strict chronological within-sequence holdout",
            "leave_one_sequence_out_status": "not_estimable: only one sequence for this compound",
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "dynamic_sequence_group_manifest.csv", index=False, encoding="utf-8-sig")

    compound_counts = manifest.groupby("compound")["sequence_id"].nunique().to_dict()
    estimability_rows = []
    for compound, count in sorted(compound_counts.items()):
        estimability_rows.append({
            "compound": compound,
            "n_independent_sequences": int(count),
            "leave_one_sequence_out_estimable": bool(count >= 2),
            "decision": "run leave-one-sequence-out" if count >= 2 else "retain strict chronological within-sequence evaluation; acquire additional independent runs before claiming sequence generalisation",
        })
    estimability = pd.DataFrame(estimability_rows)
    estimability.to_csv(output_dir / "dynamic_sequence_validation_status.csv", index=False, encoding="utf-8-sig")
    return {
        "n_records": int(len(frame)),
        "n_sequences": int(frame["sequence_id"].nunique()),
        "sequence_counts": {str(key): int(value) for key, value in frame["sequence_id"].value_counts().to_dict().items()},
        "compound_sequence_counts": {str(key): int(value) for key, value in compound_counts.items()},
        "protocol": "sequence-aware strict chronological prediction; no cross-compound leave-one-sequence-out substitution",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grouped static validation and dynamic sequence audit.")
    parser.add_argument("--static-data", type=Path, default=DEFAULT_STATIC_DATA)
    parser.add_argument("--dynamic-data", type=Path, default=DEFAULT_DYNAMIC_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--group-by", nargs="+", choices=list(GROUP_SCHEMES), default=list(GROUP_SCHEMES))
    parser.add_argument("--model", choices=[*MODEL_NAMES, "all"], default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--skip-static", action="store_true")
    parser.add_argument("--skip-dynamic-audit", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_repetitions < 100:
        parser.error("--bootstrap-repetitions must be at least 100")

    static_data = args.static_data.resolve()
    dynamic_data = args.dynamic_data.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    models = list(MODEL_NAMES) if args.model == "all" else [args.model]

    static_summary = pd.DataFrame()
    if not args.skip_static:
        static_summary = run_static_suite(static_data, output, args.group_by, models, args.seeds, args.bootstrap_repetitions)
    dynamic_status: Dict[str, Any] = {}
    if not args.skip_dynamic_audit:
        dynamic_status = audit_dynamic_sequences(dynamic_data, output)

    metadata = {
        "static_data": str(static_data),
        "static_sha256": file_sha256(static_data) if static_data.exists() else None,
        "dynamic_data": str(dynamic_data),
        "dynamic_sha256": file_sha256(dynamic_data) if dynamic_data.exists() else None,
        "group_schemes": list(args.group_by),
        "models": models,
        "seeds": list(args.seeds),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "static_protocol": "leave-one-group-out outer validation; grouped inner CV; train-fold-only median imputation and scaling; group-level bootstrap 95% intervals",
        "dynamic_protocol": "independent sequence audit plus strict chronological within-sequence evaluation; leave-one-sequence-out is reported only when estimable within compound",
        "dynamic_status": dynamic_status,
    }
    (output / "grouped_validation_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if not static_summary.empty:
        print("\n=== Combined static summary ===")
        columns = ["group_by", "model", "n_groups", "n_seeds", "grouped_mean_MAE", "grouped_sd_MAE", "grouped_mean_RMSE", "grouped_sd_RMSE", "grouped_mean_R2", "grouped_sd_R2"]
        print(static_summary[[column for column in columns if column in static_summary]].round(4).to_string(index=False))
    print("\nDynamic sequence validation status:")
    print(json.dumps(dynamic_status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
