#!/usr/bin/env python3
"""Apply the frozen late-death trajectory model to production classifications.

This is a post-classification stage.  It never changes segmentation masks.
It calibrates object features against the current run's frozen d0 fields,
detects persistent multi-site field collapse, refines eligible live objects,
updates per-cell predictions/features and per-field summaries atomically, and
regenerates the affected cell-state QC overlays.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile


METHOD_VERSION = "death_classification_consensus_v2_20260725"
BRANCH_DIRS = {
    "original": "classification_fusion",
    "nucleated_only": "classification_fusion_nucleated_only",
}


def load_local_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL = load_local_module(
    "late_dead_trajectory_model_production",
    SCRIPT_DIR / "_shared" / "late_dead_trajectory_model.py",
)
FUSION = load_local_module(
    "multichannel_classification_production",
    SCRIPT_DIR / "08_fuse_multichannel_classification.py",
)
FIELD_CONFIGURATION = dict(MODEL.APPROVED_FIELD_CONFIGURATION)
OBJECT_CONFIGURATION = dict(MODEL.APPROVED_OBJECT_CONFIGURATION)
LATE_MIN_HOURS = float(MODEL.APPROVED_LATE_MIN_HOURS)


WORKER_REFERENCES: dict[tuple[str, float, str, str], np.ndarray] = {}
WORKER_FIELD_STATES = pd.DataFrame()
WORKER_ARGS: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--calibration-go-no-go",
        type=Path,
        required=True,
        help="GO receipt produced by the frozen no-ground-truth calibration run.",
    )
    parser.add_argument(
        "--segmentation-freeze-receipt",
        type=Path,
        required=True,
        help="Read-only segmentation freeze verification receipt.",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--expected-fields-per-branch", type=int, default=27200)
    parser.add_argument("--expected-wells", type=int, default=80)
    parser.add_argument(
        "--mode",
        choices=("all", "prepare", "well", "finalize"),
        default="all",
        help=(
            "Execution stage. Production Slurm runs use prepare, one well "
            "array task, then finalize. all is a sequential compatibility mode."
        ),
    )
    parser.add_argument(
        "--well-index",
        type=int,
        help="One-based row index in the prepared well manifest for --mode well.",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_approved_calibration_configuration(
    calibration_configuration: dict[str, Any],
) -> None:
    if calibration_configuration.get("production_integration") != "approved":
        raise RuntimeError(
            "Selected calibration configuration is not approved for production"
        )
    if calibration_configuration.get("field_configuration") != FIELD_CONFIGURATION:
        raise RuntimeError(
            "Production field configuration does not match the approved "
            "calibration configuration"
        )
    if calibration_configuration.get("object_configuration") != OBJECT_CONFIGURATION:
        raise RuntimeError(
            "Production object configuration does not match the approved "
            "calibration configuration"
        )
    if not math.isclose(
        float(calibration_configuration.get("late_min_hours", math.nan)),
        LATE_MIN_HOURS,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Production late-min-hours does not match the approved "
            "calibration configuration"
        )


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_frame_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary, path)


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def uncertainty_reason_series(
    frame: pd.DataFrame,
    uncertain: pd.Series,
) -> pd.Series:
    """Return reasons aligned only to the selected uncertain rows."""
    selected = frame.loc[uncertain]
    reasons = np.select(
        [
            bool_series(
                selected["late_death_branch_discordant_uncertain"]
            ),
            bool_series(selected["late_death_track_uncertain"]),
        ],
        [
            "death_classification_branch_discordant",
            "death_classification_track_uncertain",
        ],
        default="death_classification_field_evidence_uncertain",
    )
    return pd.Series(
        reasons,
        index=selected.index,
        dtype="object",
        name="final_reason",
    )


def restore_pre_refinement_state(
    frame: pd.DataFrame,
    state_column: str,
) -> pd.DataFrame:
    """Restore the original classifier values and remove prior annotations."""
    result = frame.copy()
    if "pre_late_death_state" in result:
        baseline = result["pre_late_death_state"].fillna("").astype(str)
        valid = baseline.ne("")
        result.loc[valid, state_column] = baseline.loc[valid]
        if state_column == "final_state" and "state" in result:
            result.loc[valid, "state"] = baseline.loc[valid]
    if "pre_late_death_final_reason" in result and "final_reason" in result:
        baseline = result["pre_late_death_final_reason"].fillna("").astype(str)
        valid = baseline.ne("")
        result.loc[valid, "final_reason"] = baseline.loc[valid]
    if (
        "pre_late_death_classification_confidence" in result
        and "classification_confidence" in result
    ):
        baseline = (
            result["pre_late_death_classification_confidence"]
            .fillna("")
            .astype(str)
        )
        valid = baseline.ne("")
        result.loc[valid, "classification_confidence"] = baseline.loc[valid]
    drop_columns = [
        column
        for column in result.columns
        if column.startswith("late_death_")
    ]
    return result.drop(columns=drop_columns, errors="ignore")


def capture_pre_refinement_state(
    frame: pd.DataFrame,
    state_column: str,
) -> pd.DataFrame:
    result = frame.copy()
    if "pre_late_death_state" not in result:
        result["pre_late_death_state"] = result[state_column].astype(str)
    if (
        "pre_late_death_final_reason" not in result
        and "final_reason" in result
    ):
        result["pre_late_death_final_reason"] = result[
            "final_reason"
        ].astype(str)
    if (
        "pre_late_death_classification_confidence" not in result
        and "classification_confidence" in result
    ):
        result["pre_late_death_classification_confidence"] = result[
            "classification_confidence"
        ].astype(str)
    return result


def prepare_field_states(
    fields: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    result = fields.copy()
    result["treated"] = bool_series(result["treated"])
    result["density_score"] = (
        np.log1p(MODEL.finite_numeric(result["field_countable_count"]))
        + 2.5
        * MODEL.finite_numeric(result["field_mask_fraction_ratio_to_peak"])
    )
    result["density_bin"] = ""
    result["density_percentile"] = np.nan
    result["density_anchor"] = np.nan
    definitions: dict[str, dict[str, Any]] = {}
    for branch, indices in result.groupby("branch").groups.items():
        branch_rows = result.loc[indices]
        reference = branch_rows.loc[
            branch_rows["cohort"].eq("d0_frozen"),
            "density_score",
        ]
        if reference.size != 320:
            raise ValueError(
                f"Expected 320 d0 density fields for {branch}, found {reference.size}"
            )
        q1, q2 = np.quantile(reference.to_numpy(float), [1 / 3, 2 / 3])
        definitions[str(branch)] = {
            "low_middle": float(q1),
            "middle_high": float(q2),
            "continuous_reference_count": int(reference.size),
            "continuous_anchor_percentiles": list(MODEL.DENSITY_ANCHORS),
        }
        density_percentiles = MODEL.empirical_density_percentile(
            reference.to_numpy(float),
            result.loc[indices, "density_score"].to_numpy(float),
        )
        result.loc[indices, "density_percentile"] = density_percentiles
        result.loc[indices, "density_anchor"] = [
            MODEL.nearest_density_anchor(value) for value in density_percentiles
        ]
        result.loc[indices, "density_bin"] = np.select(
            [
                result.loc[indices, "density_score"] <= q1,
                result.loc[indices, "density_score"] <= q2,
            ],
            ["low", "middle"],
            default="high",
        )
    states = MODEL.apply_field_configuration(
        result,
        FIELD_CONFIGURATION,
        LATE_MIN_HOURS,
    )
    states = MODEL.apply_branch_field_consensus(states)
    return states, definitions


def build_live_references(
    inventory: pd.DataFrame,
    field_states: pd.DataFrame,
) -> tuple[dict[tuple[str, float, str, str], np.ndarray], dict[str, int]]:
    field_density = field_states[
        [
            "branch",
            "key",
            "cohort",
            "elapsed_hours",
            "density_percentile",
            "density_anchor",
        ]
    ].drop_duplicates(["branch", "key"])
    references: dict[tuple[str, float, str, str], list[np.ndarray]] = {}
    for row in inventory.itertuples(index=False):
        shard = Path(row.shard_path)
        frame = pd.read_csv(shard, low_memory=False)
        density_rows = field_density.loc[
            field_density["branch"].eq(str(row.branch))
            & field_density["key"].eq(str(row.key)),
        ]
        if len(density_rows) != 1:
            raise ValueError(f"Missing density reference for {row.branch}/{row.key}")
        density_row = density_rows.iloc[0]
        d0 = str(row.cohort) == "d0_frozen"
        eligible = (
            frame["proxy_type"].eq(
                "d0_live_anchor" if d0 else "untreated_live_anchor"
            )
            & bool_series(frame["countable"])
            & ~bool_series(frame["border_touching"])
        )
        live = frame.loc[eligible]
        if live.empty:
            continue
        time_group = MODEL.reference_time_group(
            str(row.cohort),
            float(density_row["elapsed_hours"]),
            not d0,
        )
        density_anchor = float(density_row["density_anchor"])
        for _output, (raw, _direction) in MODEL.RAW_FEATURES.items():
            values = pd.to_numeric(live[raw], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            references.setdefault(
                (str(row.branch), density_anchor, time_group, raw),
                [],
            ).append(values)
    arrays: dict[tuple[str, float, str, str], np.ndarray] = {}
    counts: dict[str, int] = {}
    for key, pieces in references.items():
        values = np.concatenate(pieces) if pieces else np.empty(0, dtype=float)
        if values.size < 20:
            continue
        arrays[key] = np.sort(values)
        counts[
            f"{key[0]}:{key[1]:.2f}:{key[2]}:{key[3]}"
        ] = int(values.size)
    expected = {
        (branch, anchor, "d0", raw)
        for branch in BRANCH_DIRS
        for anchor in MODEL.DENSITY_ANCHORS
        for raw, _direction in MODEL.RAW_FEATURES.values()
    }
    missing = sorted(expected - set(arrays))
    if missing:
        raise ValueError(f"Missing continuous d0 reference groups: {missing}")
    return arrays, counts


def prepared_paths(dataset_root: Path) -> dict[str, Path]:
    root = dataset_root / "prepared_refinement"
    return {
        "root": root,
        "inventory": root / "inventory.csv",
        "field_states": root / "field_states.csv",
        "references": root / "live_references.npz",
        "reference_index": root / "live_references.json",
        "density_definitions": root / "density_definitions.json",
        "reference_counts": root / "reference_counts.json",
        "well_manifest": root / "well_manifest.tsv",
        "receipt": root / "PREPARED.json",
        "success": root / "_SUCCESS",
    }


def well_artifact_paths(
    classification_root: Path,
    well_index: int,
    well: str,
) -> dict[str, Path]:
    root = classification_root / "late_death_refinement" / "well_tasks"
    stem = f"{well_index:03d}_{well}"
    return {
        "status": root / "status" / f"{stem}.csv",
        "success": root / "success" / f"{stem}.json",
        "failure": root / "failure" / f"{stem}.json",
    }


def load_production_inputs(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Path,
]:
    calibration_receipt = json.loads(args.calibration_go_no_go.read_text())
    calibration_configuration_path = (
        args.calibration_go_no_go.parent / "best_configuration.json"
    )
    if not calibration_configuration_path.is_file():
        raise FileNotFoundError(calibration_configuration_path)
    calibration_configuration = json.loads(
        calibration_configuration_path.read_text()
    )
    freeze_receipt = json.loads(args.segmentation_freeze_receipt.read_text())
    if calibration_receipt.get("decision") != "GO":
        raise RuntimeError(
            "Full classification is blocked because calibration decision is "
            f"{calibration_receipt.get('decision', 'MISSING')}"
        )
    if not bool(freeze_receipt.get("verified")):
        raise RuntimeError("Segmentation freeze verification did not pass")
    validate_approved_calibration_configuration(calibration_configuration)
    return (
        calibration_receipt,
        calibration_configuration,
        freeze_receipt,
        calibration_configuration_path,
    )


def load_inventory_and_fields(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory = pd.read_csv(
        args.dataset_root / "feature_cache" / "shard_inventory.csv"
    )
    fields = pd.read_csv(
        args.dataset_root / "feature_cache" / "field_trajectory_metrics.csv"
    )
    if inventory.empty or fields.empty:
        raise ValueError(
            "Late-death dataset inventory and field metrics must be non-empty"
        )
    inventory = inventory.merge(
        fields[["branch", "key", "well"]].drop_duplicates(["branch", "key"]),
        on=["branch", "key"],
        how="left",
        validate="one_to_one",
    )
    if inventory["well"].isna().any():
        raise ValueError("Some inventory rows are missing well metadata")
    counts = inventory.groupby("branch")["key"].nunique().to_dict()
    for branch in BRANCH_DIRS:
        if int(counts.get(branch, 0)) != args.expected_fields_per_branch:
            raise ValueError(
                f"Expected {args.expected_fields_per_branch} fields for {branch}, "
                f"found {counts.get(branch, 0)}"
            )
    return inventory, fields


def write_reference_artifacts(
    paths: dict[str, Path],
    references: dict[tuple[str, float, str, str], np.ndarray],
) -> None:
    payload: dict[str, np.ndarray] = {}
    index: list[dict[str, Any]] = []
    for array_index, key in enumerate(sorted(references), start=1):
        branch, density_anchor, time_group, raw_feature = key
        array_name = f"reference_{array_index:04d}"
        values = np.asarray(references[key], dtype=float)
        payload[array_name] = values
        index.append(
            {
                "array_name": array_name,
                "branch": branch,
                "density_anchor": float(density_anchor),
                "time_group": time_group,
                "raw_feature": raw_feature,
                "count": int(values.size),
            }
        )
    write_npz_atomic(paths["references"], payload)
    write_json_atomic(
        paths["reference_index"],
        {
            "schema_version": 1,
            "references": index,
        },
    )


def load_reference_artifacts(
    paths: dict[str, Path],
) -> dict[tuple[str, float, str, str], np.ndarray]:
    index_payload = json.loads(paths["reference_index"].read_text())
    references: dict[tuple[str, float, str, str], np.ndarray] = {}
    with np.load(paths["references"], allow_pickle=False) as archive:
        for record in index_payload["references"]:
            array_name = str(record["array_name"])
            values = np.asarray(archive[array_name], dtype=float)
            expected_count = int(record["count"])
            if values.size != expected_count:
                raise ValueError(
                    f"Prepared reference count mismatch for {array_name}: "
                    f"expected {expected_count}, found {values.size}"
                )
            key = (
                str(record["branch"]),
                float(record["density_anchor"]),
                str(record["time_group"]),
                str(record["raw_feature"]),
            )
            if key in references:
                raise ValueError(f"Duplicate prepared reference key: {key}")
            references[key] = values
    return references


def prepare_refinement_state(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = prepared_paths(args.dataset_root)
    if paths["success"].is_file() and not args.force:
        _paths, receipt, manifest = validate_prepared_state(args)
        return manifest, receipt
    paths["success"].unlink(missing_ok=True)
    inventory, fields = load_inventory_and_fields(args)
    field_states, density_definitions = prepare_field_states(fields)
    references, reference_counts = build_live_references(
        inventory,
        field_states,
    )
    well_counts = (
        inventory.groupby(["well", "branch"])["key"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("well")
        .reset_index(drop=True)
    )
    missing_branches = [
        branch for branch in BRANCH_DIRS if branch not in well_counts.columns
    ]
    if missing_branches:
        raise ValueError(
            f"Prepared well manifest is missing branches: {missing_branches}"
        )
    if len(well_counts) != args.expected_wells:
        raise ValueError(
            f"Expected {args.expected_wells} wells, found {len(well_counts)}"
        )
    well_counts.insert(0, "array_index", np.arange(1, len(well_counts) + 1))
    shard_counts = inventory.groupby("well").size().rename("total_shards")
    well_counts = well_counts.merge(
        shard_counts,
        on="well",
        how="left",
        validate="one_to_one",
    )
    well_counts = well_counts.rename(
        columns={
            branch: f"{branch}_fields"
            for branch in BRANCH_DIRS
        }
    )
    if (
        well_counts[[f"{branch}_fields" for branch in BRANCH_DIRS]]
        .le(0)
        .any()
        .any()
    ):
        raise ValueError("Every prepared well must contain both branches")

    inventory = inventory.sort_values(["well", "branch", "key"]).reset_index(
        drop=True
    )
    field_states = field_states.sort_values(
        ["well", "branch", "elapsed_hours", "site", "key"]
    ).reset_index(drop=True)
    write_frame_atomic(paths["inventory"], inventory)
    write_frame_atomic(paths["field_states"], field_states)
    write_reference_artifacts(paths, references)
    write_json_atomic(paths["density_definitions"], density_definitions)
    write_json_atomic(paths["reference_counts"], reference_counts)
    paths["well_manifest"].parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = paths["well_manifest"].with_name(
        f".{paths['well_manifest'].name}.tmp.{os.getpid()}"
    )
    well_counts.to_csv(temporary_manifest, sep="\t", index=False)
    os.replace(temporary_manifest, paths["well_manifest"])
    receipt = {
        "schema_version": 1,
        "method_version": METHOD_VERSION,
        "expected_fields_per_branch": args.expected_fields_per_branch,
        "expected_wells": args.expected_wells,
        "well_count": int(len(well_counts)),
        "inventory_rows": int(len(inventory)),
        "field_state_rows": int(len(field_states)),
        "reference_group_count": int(len(references)),
        "artifacts": {
            name: {
                "path": str(paths[name]),
                "sha256": file_sha256(paths[name]),
            }
            for name in (
                "inventory",
                "field_states",
                "references",
                "reference_index",
                "density_definitions",
                "reference_counts",
                "well_manifest",
            )
        },
    }
    write_json_atomic(paths["receipt"], receipt)
    paths["success"].touch()
    return well_counts, receipt


def validate_prepared_state(
    args: argparse.Namespace,
) -> tuple[dict[str, Path], dict[str, Any], pd.DataFrame]:
    paths = prepared_paths(args.dataset_root)
    for name in (
        "inventory",
        "field_states",
        "references",
        "reference_index",
        "density_definitions",
        "reference_counts",
        "well_manifest",
        "receipt",
        "success",
    ):
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])
    receipt = json.loads(paths["receipt"].read_text())
    if receipt.get("method_version") != METHOD_VERSION:
        raise RuntimeError("Prepared refinement method version does not match")
    if int(receipt.get("expected_fields_per_branch", -1)) != int(
        args.expected_fields_per_branch
    ):
        raise RuntimeError("Prepared field-count contract does not match")
    if int(receipt.get("expected_wells", -1)) != int(args.expected_wells):
        raise RuntimeError("Prepared well-count contract does not match")
    for name, artifact in receipt["artifacts"].items():
        path = paths[name]
        if str(path) != str(artifact["path"]):
            raise RuntimeError(f"Prepared artifact path drift: {name}")
        if file_sha256(path) != str(artifact["sha256"]):
            raise RuntimeError(f"Prepared artifact checksum drift: {name}")
    manifest = pd.read_csv(paths["well_manifest"], sep="\t")
    if len(manifest) != args.expected_wells:
        raise ValueError(
            f"Expected {args.expected_wells} prepared wells, found {len(manifest)}"
        )
    if manifest["array_index"].tolist() != list(
        range(1, args.expected_wells + 1)
    ):
        raise ValueError("Prepared well array indices are not contiguous")
    if manifest["well"].astype(str).duplicated().any():
        raise ValueError("Prepared well manifest contains duplicate wells")
    return paths, receipt, manifest


def initialize_worker(
    references: dict[tuple[str, float, str, str], np.ndarray],
    field_states: pd.DataFrame,
    worker_args: dict[str, Any],
) -> None:
    global WORKER_REFERENCES, WORKER_FIELD_STATES, WORKER_ARGS
    WORKER_REFERENCES = references
    WORKER_FIELD_STATES = field_states
    WORKER_ARGS = worker_args


def calibrate_group_features(data: pd.DataFrame) -> pd.DataFrame:
    return MODEL.apply_empirical_feature_calibration(
        data,
        WORKER_REFERENCES,
        error_context="production",
    )


def update_summary(
    summary_path: Path,
    feature_frame: pd.DataFrame,
    field_rows: pd.DataFrame,
) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    if len(summary) != 1:
        raise ValueError(f"Expected one summary row: {summary_path}")
    result = summary.copy()
    row_index = result.index[0]
    states = feature_frame["final_state"].astype(str)
    counts = states.value_counts().to_dict()
    total_masks = len(feature_frame)
    artifact = int(counts.get("artifact", 0))
    total_cells = total_masks - artifact
    live = int(counts.get("live", 0))
    dead = int(counts.get("dead", 0))
    for column in (
        "live_cell_count",
        "dead_cell_count",
        "uncertain_count",
        "transitional_count",
        "artifact_count",
    ):
        baseline = f"pre_late_death_{column}"
        if baseline not in result:
            result[baseline] = result[column]
    result.loc[row_index, "total_masks"] = total_masks
    result.loc[row_index, "total_cell_count"] = total_cells
    result.loc[row_index, "live_cell_count"] = live
    result.loc[row_index, "dead_cell_count"] = dead
    result.loc[row_index, "uncertain_count"] = int(counts.get("uncertain", 0))
    result.loc[row_index, "transitional_count"] = int(
        counts.get("transitional", 0)
    )
    result.loc[row_index, "artifact_count"] = artifact
    result.loc[row_index, "dead_fraction"] = (
        dead / total_cells if total_cells else 0.0
    )
    result.loc[row_index, "live_fraction"] = (
        live / total_cells if total_cells else 0.0
    )
    uncertain_cells = int(counts.get("uncertain", 0))
    result.loc[row_index, "classification_uncertain_count"] = uncertain_cells
    result.loc[row_index, "dead_fraction_lower_bound"] = (
        dead / total_cells if total_cells else 0.0
    )
    result.loc[row_index, "dead_fraction_upper_bound"] = (
        (dead + uncertain_cells) / total_cells if total_cells else 0.0
    )
    if "supplemental_dead_object_count" in result:
        supplemental_value = pd.to_numeric(
            result.loc[row_index, "supplemental_dead_object_count"],
            errors="coerce",
        )
        supplemental = (
            int(supplemental_value)
            if pd.notna(supplemental_value)
            else 0
        )
        result.loc[row_index, "object_aware_dead_count"] = (
            dead + supplemental
        )
    result.loc[row_index, "late_death_refinement_version"] = METHOD_VERSION
    result.loc[row_index, "late_death_field_global"] = bool(
        field_rows["field_global_late_death"].iloc[0]
    )
    result.loc[row_index, "late_death_field_branch_raw_global"] = bool(
        field_rows.get(
            "field_branch_raw_global",
            pd.Series([False]),
        ).iloc[0]
    )
    result.loc[row_index, "late_death_field_branch_discordant"] = bool(
        field_rows.get(
            "field_branch_raw_discordant",
            pd.Series([False]),
        ).iloc[0]
    )
    result.loc[row_index, "late_death_rescue_count"] = int(
        field_rows["late_death_rescue_call"].sum()
    )
    result.loc[row_index, "late_death_uncertain_count"] = int(
        field_rows["late_death_uncertain"].sum()
    )
    return result


def update_field_outputs(field_rows: pd.DataFrame) -> dict[str, Any]:
    first = field_rows.iloc[0]
    key = str(first["key"])
    branch = str(first["branch"])
    image_id = str(first["image_id"])
    out_dir = Path(WORKER_ARGS["classification_root"]) / BRANCH_DIRS[branch]
    feature_path = Path(str(first["source_feature_path"]))
    prediction_path = (
        out_dir / "predictions" / f"{image_id}_per_cell_predictions.csv"
    )
    summary_path = out_dir / "summaries" / f"{image_id}_summary.csv"
    status_path = (
        Path(WORKER_ARGS["classification_root"])
        / "late_death_refinement"
        / "status"
        / branch
        / f"{key}.json"
    )
    if (
        status_path.is_file()
        and not bool(WORKER_ARGS["force"])
    ):
        return json.loads(status_path.read_text())

    features = capture_pre_refinement_state(
        restore_pre_refinement_state(
            pd.read_csv(feature_path),
            "final_state",
        ),
        "final_state",
    )
    predictions = capture_pre_refinement_state(
        restore_pre_refinement_state(
            pd.read_csv(prediction_path),
            "state",
        ),
        "state",
    )
    annotation_columns = [
        "combined_mask_id",
        "field_global_late_death",
        "field_branch_raw_global",
        "field_branch_raw_discordant",
        "field_branch_consensus_late_death",
        "field_state_transition",
        "field_collapse_signal_count",
        "field_site_concordance",
        "density_bin",
        "density_percentile",
        "density_anchor",
        "death_signal_count",
        "healthy_signal_count",
        "death_score",
        "strong_live_evidence",
        "late_dead_object_evidence",
        "branch_partner_matched",
        "branch_partner_distance",
        "branch_partner_current_dead",
        "branch_partner_object_evidence",
        "branch_partner_strong_live",
        "branch_partner_temporal_remnant",
        "branch_partner_death_signal_count",
        "branch_object_evidence_agree",
        "branch_partner_late_death_rescue_call",
        "branch_partner_final_dead_call",
        "branch_late_death_rescue_call_agree",
        "branch_final_dead_call_agree",
        "branch_final_call_discordant",
        "temporal_track_confident",
        "temporal_support_frames",
        "temporal_match_confidence",
        "temporal_carryforward_call",
        "global_late_death_rescue_call",
        "late_death_rescue_call",
        "branch_discordant_uncertain",
        "track_uncertain",
        "field_evidence_uncertain",
        "late_death_uncertain",
        "classification_tier",
        "cell_dead_snr",
        "cell_dead_positive_fraction",
        "cell_dead_core_enrichment",
        *MODEL.MODEL_FEATURES,
    ]
    defaults: dict[str, Any] = {
        "field_branch_raw_global": False,
        "field_branch_raw_discordant": False,
        "field_branch_consensus_late_death": False,
        "field_state_transition": "inactive",
        "density_percentile": 0.5,
        "density_anchor": 0.5,
        "branch_partner_matched": False,
        "branch_partner_distance": float("inf"),
        "branch_partner_current_dead": False,
        "branch_partner_object_evidence": False,
        "branch_partner_strong_live": False,
        "branch_partner_temporal_remnant": False,
        "branch_partner_death_signal_count": 0,
        "branch_object_evidence_agree": False,
        "branch_partner_late_death_rescue_call": False,
        "branch_partner_final_dead_call": False,
        "branch_late_death_rescue_call_agree": False,
        "branch_final_dead_call_agree": False,
        "branch_final_call_discordant": False,
        "temporal_track_confident": False,
        "temporal_support_frames": 0,
        "temporal_match_confidence": 0.0,
        "branch_discordant_uncertain": False,
        "track_uncertain": False,
        "field_evidence_uncertain": False,
        "classification_tier": "confirmed_live",
        "cell_dead_snr": 0.0,
        "cell_dead_positive_fraction": 0.0,
        "cell_dead_core_enrichment": 0.0,
    }
    annotation_source = field_rows.copy()
    for column in annotation_columns:
        if column not in annotation_source:
            annotation_source[column] = defaults.get(column, False)
    annotations = annotation_source[annotation_columns].copy()
    annotation_names = {
        "field_global_late_death": "late_death_field_global",
        "field_branch_raw_global": "late_death_field_branch_raw_global",
        "field_branch_raw_discordant": "late_death_field_branch_discordant",
        "field_branch_consensus_late_death": "late_death_field_branch_consensus",
        "field_state_transition": "late_death_field_state_transition",
        "field_collapse_signal_count": "late_death_field_signal_count",
        "field_site_concordance": "late_death_site_concordance",
        "density_bin": "late_death_density_bin",
        "density_percentile": "late_death_density_percentile",
        "density_anchor": "late_death_density_anchor",
        "death_signal_count": "late_death_signal_count",
        "healthy_signal_count": "late_death_healthy_signal_count",
        "death_score": "late_death_score",
        "strong_live_evidence": "late_death_strong_live_evidence",
        "late_dead_object_evidence": "late_death_object_evidence",
        "temporal_carryforward_call": "late_death_temporal_carryforward_call",
        "global_late_death_rescue_call": "late_death_global_rescue_call",
        "late_death_rescue_call": "late_death_rescue_call",
        "late_death_uncertain": "late_death_uncertain",
        **{
            column: f"late_death_{column}"
            for column in MODEL.MODEL_FEATURES
        },
    }
    for column in annotation_columns:
        if column != "combined_mask_id" and column not in annotation_names:
            annotation_names[column] = (
                column
                if column.startswith("late_death_")
                else f"late_death_{column}"
            )
    annotations = annotations.rename(columns=annotation_names)
    annotations["late_death_refinement_version"] = METHOD_VERSION
    if annotations["combined_mask_id"].duplicated().any():
        raise ValueError(f"Duplicate object ids in refinement field {branch}/{key}")

    features["combined_mask_id"] = pd.to_numeric(
        features["combined_mask_id"],
        errors="raise",
    ).astype(int)
    updated_features = features.merge(
        annotations,
        on="combined_mask_id",
        how="left",
        validate="one_to_one",
    )
    if updated_features["late_death_refinement_version"].isna().any():
        raise ValueError(f"Feature/annotation mismatch for {branch}/{key}")
    rescue = bool_series(updated_features["late_death_rescue_call"])
    uncertain = (
        bool_series(updated_features["late_death_uncertain"])
        & updated_features["final_state"].astype(str).eq("live")
    )
    updated_features.loc[rescue, ["state", "final_state"]] = "dead"
    updated_features.loc[rescue, "final_reason"] = (
        "late_death_trajectory_rescue"
    )
    updated_features.loc[rescue, "classification_confidence"] = "medium"
    updated_features.loc[uncertain, ["state", "final_state"]] = "uncertain"
    feature_uncertain_reasons = uncertainty_reason_series(
        updated_features,
        uncertain,
    )
    updated_features.loc[
        feature_uncertain_reasons.index,
        "final_reason",
    ] = feature_uncertain_reasons
    updated_features.loc[uncertain, "classification_confidence"] = "low"

    prediction_annotations = annotations.rename(
        columns={"combined_mask_id": "mask_id"}
    )
    predictions["mask_id"] = pd.to_numeric(
        predictions["mask_id"],
        errors="raise",
    ).astype(int)
    updated_predictions = predictions.merge(
        prediction_annotations,
        on="mask_id",
        how="left",
        validate="one_to_one",
    )
    if updated_predictions["late_death_refinement_version"].isna().any():
        raise ValueError(f"Prediction/annotation mismatch for {branch}/{key}")
    pred_rescue = bool_series(
        updated_predictions["late_death_rescue_call"]
    )
    pred_uncertain = (
        bool_series(updated_predictions["late_death_uncertain"])
        & updated_predictions["state"].astype(str).eq("live")
    )
    updated_predictions.loc[pred_rescue, "state"] = "dead"
    updated_predictions.loc[pred_rescue, "final_reason"] = (
        "late_death_trajectory_rescue"
    )
    updated_predictions.loc[pred_rescue, "classification_confidence"] = (
        "medium"
    )
    updated_predictions.loc[pred_uncertain, "state"] = "uncertain"
    prediction_uncertain_reasons = uncertainty_reason_series(
        updated_predictions,
        pred_uncertain,
    )
    updated_predictions.loc[
        prediction_uncertain_reasons.index,
        "final_reason",
    ] = prediction_uncertain_reasons
    updated_predictions.loc[pred_uncertain, "classification_confidence"] = "low"
    if int(rescue.sum()) != int(pred_rescue.sum()):
        raise ValueError(f"Feature/prediction rescue mismatch for {branch}/{key}")
    if int(uncertain.sum()) != int(pred_uncertain.sum()):
        raise ValueError(f"Feature/prediction uncertainty mismatch for {branch}/{key}")

    updated_summary = update_summary(
        summary_path,
        updated_features,
        field_rows,
    )
    annotation_path = (
        Path(WORKER_ARGS["classification_root"])
        / "late_death_refinement"
        / "annotations"
        / branch
        / f"{key}.csv"
    )
    write_frame_atomic(annotation_path, annotations)
    write_frame_atomic(feature_path, updated_features)
    write_frame_atomic(prediction_path, updated_predictions)
    write_frame_atomic(summary_path, updated_summary)

    cell_mask = np.squeeze(tifffile.imread(str(first["cell_mask_path"]))).astype(
        np.uint32,
        copy=False,
    )
    rescued_ids = updated_predictions.loc[pred_rescue, "mask_id"].to_numpy(int)
    rescued_mask = np.where(
        np.isin(cell_mask, rescued_ids),
        cell_mask,
        0,
    ).astype(np.uint32)
    rescued_mask_path = (
        out_dir
        / "masks"
        / "late_death_rescued"
        / f"{image_id}_late_death_rescued_masks.tif"
    )
    if rescued_ids.size:
        FUSION.write_label_tiff(rescued_mask_path, rescued_mask, "YX")
    elif rescued_mask_path.exists():
        rescued_mask_path.unlink()

    if int(rescue.sum()) or int(field_rows["late_death_uncertain"].sum()):
        raw_rgb = FUSION.to_rgb(FUSION.read_raw_image(Path(str(first["combined_raw_path"]))))
        FUSION.make_classification_overlay(
            raw_rgb,
            cell_mask,
            updated_predictions.to_dict(orient="records"),
            out_dir / "qc" / "label_overlays" / f"{image_id}_state_overlay.png",
            float(WORKER_ARGS["overlay_alpha"]),
        )
        record = json.loads(Path(str(first["field_record_path"])).read_text())
        nuclei_values = record["profiles"]["Nuclei"]
        nuclei_raw_path = Path(str(nuclei_values.get("raw", "")))
        nuclei_raw = (
            FUSION.scalar_raw(FUSION.read_raw_image(nuclei_raw_path))
            if nuclei_raw_path.is_file()
            else None
        )
        nuclei_mask = np.squeeze(
            tifffile.imread(str(first["nucleus_core_mask_path"]))
        )
        dead_raw_path = Path(str(first["dead_raw_path"]))
        dead_raw = (
            FUSION.scalar_raw(FUSION.read_raw_image(dead_raw_path))
            if dead_raw_path.is_file()
            else None
        )
        cell_state_mask = np.squeeze(
            tifffile.imread(
                out_dir
                / "masks"
                / "cell_state"
                / f"{image_id}_cell_state_masks.tif"
            )
        )
        confirmed_dead_mask = np.squeeze(
            tifffile.imread(
                out_dir
                / "masks"
                / "confirmed_dead"
                / f"{image_id}_confirmed_dead_masks.tif"
            )
        )
        FUSION.make_overlap_state_overlay(
            raw_rgb,
            nuclei_raw,
            nuclei_mask,
            dead_raw,
            cell_state_mask,
            confirmed_dead_mask,
            updated_predictions.to_dict(orient="records"),
            out_dir
            / "qc"
            / "overlap_state_overlays"
            / f"{image_id}_overlap_state_overlay.png",
            float(WORKER_ARGS["overlay_alpha"]),
        )

    status = {
        "branch": branch,
        "key": key,
        "method_version": METHOD_VERSION,
        "field_global_late_death": bool(
            field_rows["field_global_late_death"].iloc[0]
        ),
        "field_branch_raw_global": bool(
            field_rows.get(
                "field_branch_raw_global",
                pd.Series([False]),
            ).iloc[0]
        ),
        "field_branch_raw_discordant": bool(
            field_rows.get(
                "field_branch_raw_discordant",
                pd.Series([False]),
            ).iloc[0]
        ),
        "baseline_dead": int(field_rows["current_dead_call"].sum()),
        "refined_dead": int(field_rows["final_dead_call"].sum()),
        "rescued": int(field_rows["late_death_rescue_call"].sum()),
        "uncertain": int(field_rows["late_death_uncertain"].sum()),
        "branch_discordant_uncertain": int(
            field_rows.get(
                "branch_discordant_uncertain",
                pd.Series(False, index=field_rows.index),
            ).sum()
        ),
        "track_uncertain": int(
            field_rows.get(
                "track_uncertain",
                pd.Series(False, index=field_rows.index),
            ).sum()
        ),
        "matched_objects": int(
            field_rows.get(
                "branch_partner_matched",
                pd.Series(False, index=field_rows.index),
            ).sum()
        ),
        "matched_rescue_call_agree": int(
            field_rows.get(
                "branch_late_death_rescue_call_agree",
                pd.Series(False, index=field_rows.index),
            ).sum()
        ),
        "matched_final_dead_call_agree": int(
            field_rows.get(
                "branch_final_dead_call_agree",
                pd.Series(False, index=field_rows.index),
            ).sum()
        ),
        "objects": int(len(field_rows)),
        "annotation_path": str(annotation_path),
        "rescued_mask_path": str(rescued_mask_path) if rescued_ids.size else "",
    }
    write_json_atomic(status_path, status)
    return status


def refine_group(task: dict[str, Any]) -> list[dict[str, Any]]:
    well = str(task["well"])
    frames: list[pd.DataFrame] = []
    for shard_path in task["shards"]:
        frame = pd.read_csv(shard_path, low_memory=False)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    frames.clear()
    del frames
    gc.collect()
    for column in (
        "countable",
        "border_touching",
        "treated",
        "cyclophosphamide",
    ):
        data[column] = bool_series(data[column])
    data["elapsed_hours"] = MODEL.finite_numeric(data["elapsed_hours"])
    density = WORKER_FIELD_STATES[
        [
            "branch",
            "key",
            "density_bin",
            "density_percentile",
            "density_anchor",
        ]
    ].drop_duplicates(["branch", "key"])
    data = data.merge(
        density,
        on=["branch", "key"],
        how="left",
        validate="many_to_one",
    )
    if data["density_bin"].isna().any():
        raise ValueError(f"Missing density calibration for {well}")
    calibrated = calibrate_group_features(data)
    del data, density
    gc.collect()
    field_subset = WORKER_FIELD_STATES.loc[
        WORKER_FIELD_STATES["well"].eq(well)
    ]
    calls = MODEL.classification_calls(
        calibrated,
        field_subset,
        OBJECT_CONFIGURATION,
        LATE_MIN_HOURS,
        apply_treatment_scope=True,
    )
    del calibrated
    gc.collect()
    statuses = []
    for (_branch, _key), rows in calls.groupby(["branch", "key"], sort=False):
        statuses.append(update_field_outputs(rows.copy()))
    return statuses


def merge_branch_summaries(
    classification_root: Path,
    branch: str,
    expected: int,
) -> Path:
    summaries_dir = classification_root / BRANCH_DIRS[branch] / "summaries"
    paths = sorted(
        path
        for path in summaries_dir.glob("*_summary.csv")
        if path.name != "cell_count_summary.csv"
    )
    if len(paths) != expected:
        raise ValueError(
            f"Expected {expected} per-field summaries for {branch}, found {len(paths)}"
        )
    frames = [pd.read_csv(path) for path in paths]
    if any(len(frame) != 1 for frame in frames):
        raise ValueError(f"Non-singleton per-field summary detected for {branch}")
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.sort_values(
        ["well", "elapsed_hours", "site", "key"]
    ).reset_index(drop=True)
    out_path = summaries_dir / "cell_count_summary.csv"
    write_frame_atomic(out_path, merged)
    return out_path


def write_consensus_summary(
    classification_root: Path,
    summary_paths: dict[str, str],
) -> Path:
    """Publish one authoritative field table with both branch diagnostics."""
    original = pd.read_csv(summary_paths["original"])
    nucleated = pd.read_csv(summary_paths["nucleated_only"])
    partner_columns = [
        "key",
        "total_cell_count",
        "live_cell_count",
        "dead_cell_count",
        "uncertain_count",
        "dead_fraction",
        "dead_fraction_lower_bound",
        "dead_fraction_upper_bound",
        "late_death_field_branch_raw_global",
        "late_death_field_branch_discordant",
        "late_death_rescue_count",
        "late_death_uncertain_count",
    ]
    partner_columns = [
        column for column in partner_columns if column in nucleated.columns
    ]
    partner = nucleated[partner_columns].rename(
        columns={
            column: f"nucleated_only_{column}"
            for column in partner_columns
            if column != "key"
        }
    )
    consensus = original.merge(
        partner,
        on="key",
        how="left",
        validate="one_to_one",
    )
    if consensus.filter(like="nucleated_only_").isna().any().any():
        raise ValueError("Consensus summary lost nucleated-only fields")
    consensus["consensus_source_branch"] = "original"
    consensus["consensus_method_version"] = METHOD_VERSION
    consensus["branch_dead_fraction_abs_diff"] = (
        pd.to_numeric(consensus["dead_fraction"], errors="raise")
        - pd.to_numeric(
            consensus["nucleated_only_dead_fraction"],
            errors="raise",
        )
    ).abs()
    out_path = (
        classification_root
        / "classification_consensus"
        / "summaries"
        / "cell_count_summary.csv"
    )
    write_frame_atomic(out_path, consensus)
    return out_path


def write_production_go_no_go(
    classification_root: Path,
    status_frame: pd.DataFrame,
    consensus_summary_path: Path,
    calibration_receipt: dict[str, Any],
    freeze_receipt: dict[str, Any],
    expected_fields_per_branch: int,
) -> tuple[Path, dict[str, Any]]:
    """Record operational convergence without claiming biological accuracy."""
    consensus = pd.read_csv(consensus_summary_path)
    expected_keys = int(expected_fields_per_branch)
    key_counts = status_frame.groupby("branch")["key"].nunique().to_dict()
    d0 = status_frame["key"].astype(str).str.endswith("_00d00h00m")
    field_states = status_frame.pivot(
        index="key",
        columns="branch",
        values="field_global_late_death",
    )
    field_mismatch = (
        field_states["original"].astype(bool)
        != field_states["nucleated_only"].astype(bool)
    )
    branch_difference = pd.to_numeric(
        consensus["branch_dead_fraction_abs_diff"],
        errors="raise",
    )
    original_rescue_fraction = (
        pd.to_numeric(
            consensus["late_death_rescue_count"],
            errors="raise",
        )
        / pd.to_numeric(
            consensus["total_cell_count"],
            errors="raise",
        ).replace(0, np.nan)
    ).fillna(0.0)
    nucleated_rescue_fraction = (
        pd.to_numeric(
            consensus["nucleated_only_late_death_rescue_count"],
            errors="raise",
        )
        / pd.to_numeric(
            consensus["nucleated_only_total_cell_count"],
            errors="raise",
        ).replace(0, np.nan)
    ).fillna(0.0)
    rescue_difference = (
        original_rescue_fraction - nucleated_rescue_fraction
    ).abs()
    object_count = int(status_frame["objects"].sum())
    uncertain_count = int(status_frame["uncertain"].sum())
    original_status = status_frame.loc[status_frame["branch"].eq("original")]
    matched_pair_count = int(original_status["matched_objects"].sum())
    matched_rescue_agree = int(
        original_status["matched_rescue_call_agree"].sum()
    )
    matched_final_agree = int(
        original_status["matched_final_dead_call_agree"].sum()
    )
    diagnostics = {
        "field_state_mismatch_count": int(field_mismatch.sum()),
        "field_state_mismatch_rate": float(field_mismatch.mean()),
        "field_dead_fraction_abs_diff_median": float(branch_difference.median()),
        "field_dead_fraction_abs_diff_gt_0p10_count": int(
            branch_difference.gt(0.10).sum()
        ),
        "field_dead_fraction_abs_diff_gt_0p10_rate": float(
            branch_difference.gt(0.10).mean()
        ),
        "field_dead_fraction_abs_diff_gt_0p25_count": int(
            branch_difference.gt(0.25).sum()
        ),
        "field_dead_fraction_abs_diff_gt_0p25_rate": float(
            branch_difference.gt(0.25).mean()
        ),
        "field_rescue_fraction_abs_diff_median": float(
            rescue_difference.median()
        ),
        "field_rescue_fraction_abs_diff_gt_0p10_count": int(
            rescue_difference.gt(0.10).sum()
        ),
        "field_rescue_fraction_abs_diff_gt_0p10_rate": float(
            rescue_difference.gt(0.10).mean()
        ),
        "field_rescue_fraction_abs_diff_gt_0p25_count": int(
            rescue_difference.gt(0.25).sum()
        ),
        "field_rescue_fraction_abs_diff_gt_0p25_rate": float(
            rescue_difference.gt(0.25).mean()
        ),
        "uncertain_object_count": uncertain_count,
        "uncertain_object_rate": (
            uncertain_count / object_count if object_count else 0.0
        ),
        "matched_pair_count": matched_pair_count,
        "matched_pair_rescue_call_agreement": (
            matched_rescue_agree / matched_pair_count
            if matched_pair_count
            else 0.0
        ),
        "matched_pair_final_dead_call_agreement": (
            matched_final_agree / matched_pair_count
            if matched_pair_count
            else 0.0
        ),
    }
    gates = {
        "SEGMENTATION_FROZEN": {
            "pass": bool(freeze_receipt.get("verified")),
            "verification_receipt": str(
                freeze_receipt.get("receipt_path", "")
            ),
            "source_run_root": str(freeze_receipt.get("source_run_root", "")),
            "verified_file_count": int(
                freeze_receipt.get("verified_file_count", 0)
            ),
        },
        "CALIBRATION_CONVERGENCE": {
            "pass": calibration_receipt.get("decision") == "GO",
            "calibration_decision": str(
                calibration_receipt.get("decision", "MISSING")
            ),
            "metric_semantics": str(
                calibration_receipt.get("metric_semantics", "")
            ),
        },
        "FULL_COHORT_COMPLETENESS": {
            "pass": (
                int(key_counts.get("original", 0)) == expected_keys
                and int(key_counts.get("nucleated_only", 0)) == expected_keys
                and len(consensus) == expected_keys
            ),
            "expected_fields_per_branch": expected_keys,
            "original_fields": int(key_counts.get("original", 0)),
            "nucleated_only_fields": int(
                key_counts.get("nucleated_only", 0)
            ),
            "consensus_fields": int(len(consensus)),
        },
        "D0_INVARIANCE": {
            "pass": bool(
                int(status_frame.loc[d0, "rescued"].sum()) == 0
                and int(status_frame.loc[d0, "uncertain"].sum()) == 0
            ),
            "d0_field_rows": int(d0.sum()),
            "d0_rescued_objects": int(
                status_frame.loc[d0, "rescued"].sum()
            ),
            "d0_uncertain_objects": int(
                status_frame.loc[d0, "uncertain"].sum()
            ),
        },
        "DUAL_VIEW_DIAGNOSTICS": {
            "pass": bool(diagnostics["field_state_mismatch_rate"] <= 0.01)
            and bool(
                diagnostics["matched_pair_rescue_call_agreement"] >= 0.99
            )
            and bool(
                diagnostics["matched_pair_final_dead_call_agreement"] >= 0.99
            ),
            **diagnostics,
        },
    }
    decision = (
        "GO" if all(bool(gate["pass"]) for gate in gates.values()) else "NO_GO"
    )
    payload = {
        "schema_version": 1,
        "decision": decision,
        "metric_semantics": (
            "operational_proxy_validation_without_manual_biological_ground_truth"
        ),
        "biological_accuracy_claimed": False,
        "method_version": METHOD_VERSION,
        "gates": gates,
    }
    out_path = (
        classification_root
        / "late_death_refinement"
        / "FULL_CLASSIFICATION_GO_NO_GO.json"
    )
    write_json_atomic(out_path, payload)
    return out_path, payload


def run_one_well(
    args: argparse.Namespace,
    well_index: int,
) -> dict[str, Any]:
    paths, _prepared_receipt, manifest = validate_prepared_state(args)
    selected = manifest.loc[manifest["array_index"].eq(well_index)]
    if len(selected) != 1:
        raise ValueError(
            f"Well array index {well_index} matched {len(selected)} manifest rows"
        )
    manifest_row = selected.iloc[0]
    well = str(manifest_row["well"])
    artifacts = well_artifact_paths(args.classification_root, well_index, well)
    if (
        artifacts["success"].is_file()
        and artifacts["status"].is_file()
        and not args.force
    ):
        success = json.loads(artifacts["success"].read_text())
        if file_sha256(artifacts["status"]) != success.get("status_sha256"):
            raise RuntimeError(f"Completed well status checksum drift: {well}")
        print(f"well={well}")
        print(f"well_index={well_index}")
        print("well_status=reused")
        print(f"status_path={artifacts['status']}")
        return success

    artifacts["success"].unlink(missing_ok=True)
    inventory = pd.read_csv(paths["inventory"])
    well_inventory = inventory.loc[
        inventory["well"].astype(str).eq(well)
    ].copy()
    del inventory
    if well_inventory.empty:
        raise ValueError(f"Prepared inventory is empty for well {well}")
    expected_pairs = set(
        zip(
            well_inventory["branch"].astype(str),
            well_inventory["key"].astype(str),
        )
    )
    expected_total = int(
        sum(
            int(manifest_row[f"{branch}_fields"])
            for branch in BRANCH_DIRS
        )
    )
    if len(expected_pairs) != expected_total:
        raise ValueError(
            f"Prepared inventory pair count mismatch for {well}: "
            f"expected {expected_total}, found {len(expected_pairs)}"
        )

    field_states = pd.read_csv(paths["field_states"])
    field_states = field_states.loc[
        field_states["well"].astype(str).eq(well)
    ].copy()
    field_state_pairs = set(
        zip(
            field_states["branch"].astype(str),
            field_states["key"].astype(str),
        )
    )
    if field_state_pairs != expected_pairs:
        missing = sorted(expected_pairs - field_state_pairs)[:10]
        unexpected = sorted(field_state_pairs - expected_pairs)[:10]
        raise ValueError(
            f"Prepared field-state set mismatch for {well}: "
            f"missing={missing} unexpected={unexpected}"
        )
    references = load_reference_artifacts(paths)
    initialize_worker(
        references,
        field_states,
        {
            "classification_root": str(args.classification_root),
            "overlay_alpha": args.overlay_alpha,
            "force": args.force,
        },
    )
    task = {
        "well": well,
        "shards": [str(value) for value in well_inventory["shard_path"]],
    }
    try:
        statuses = refine_group(task)
        status_frame = pd.DataFrame(statuses)
        if status_frame.empty:
            raise ValueError(f"No refinement statuses were produced for {well}")
        if status_frame.duplicated(["branch", "key"]).any():
            raise ValueError(f"Duplicate refinement field statuses for {well}")
        actual_pairs = set(
            zip(
                status_frame["branch"].astype(str),
                status_frame["key"].astype(str),
            )
        )
        if actual_pairs != expected_pairs:
            missing = sorted(expected_pairs - actual_pairs)[:10]
            unexpected = sorted(actual_pairs - expected_pairs)[:10]
            raise ValueError(
                f"Refinement field set mismatch for {well}: "
                f"missing={missing} unexpected={unexpected}"
            )
        status_frame = status_frame.sort_values(["branch", "key"]).reset_index(
            drop=True
        )
        write_frame_atomic(artifacts["status"], status_frame)
        success = {
            "schema_version": 1,
            "method_version": METHOD_VERSION,
            "well": well,
            "well_index": well_index,
            "status": "SUCCESS",
            "field_rows": int(len(status_frame)),
            "branch_field_rows": {
                branch: int(
                    status_frame["branch"].astype(str).eq(branch).sum()
                )
                for branch in BRANCH_DIRS
            },
            "prepared_receipt_sha256": file_sha256(paths["receipt"]),
            "status_path": str(artifacts["status"]),
            "status_sha256": file_sha256(artifacts["status"]),
        }
        write_json_atomic(artifacts["success"], success)
        artifacts["failure"].unlink(missing_ok=True)
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "method_version": METHOD_VERSION,
            "well": well,
            "well_index": well_index,
            "status": "FAILED",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "prepared_receipt_sha256": file_sha256(paths["receipt"]),
        }
        write_json_atomic(artifacts["failure"], failure)
        raise

    print(f"well={well}")
    print(f"well_index={well_index}")
    print("well_status=SUCCESS")
    print(f"field_rows={success['field_rows']}")
    print(f"status_path={artifacts['status']}")
    return success


def finalize_refinement(
    args: argparse.Namespace,
    calibration_receipt: dict[str, Any],
    calibration_configuration_path: Path,
    freeze_receipt: dict[str, Any],
) -> dict[str, Any]:
    paths, _prepared_receipt, manifest = validate_prepared_state(args)
    inventory = pd.read_csv(paths["inventory"])
    status_frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        well_index = int(row.array_index)
        well = str(row.well)
        artifacts = well_artifact_paths(
            args.classification_root,
            well_index,
            well,
        )
        if not artifacts["success"].is_file():
            error = "well success receipt is missing"
            if artifacts["failure"].is_file():
                error = str(
                    json.loads(artifacts["failure"].read_text()).get(
                        "error",
                        error,
                    )
                )
            failures.append(
                {
                    "well": well,
                    "well_index": well_index,
                    "error": error,
                }
            )
            continue
        if not artifacts["status"].is_file():
            failures.append(
                {
                    "well": well,
                    "well_index": well_index,
                    "error": "well status CSV is missing",
                }
            )
            continue
        success = json.loads(artifacts["success"].read_text())
        actual_sha256 = file_sha256(artifacts["status"])
        if actual_sha256 != success.get("status_sha256"):
            failures.append(
                {
                    "well": well,
                    "well_index": well_index,
                    "error": "well status CSV checksum drift",
                }
            )
            continue
        status = pd.read_csv(artifacts["status"])
        if len(status) != int(success.get("field_rows", -1)):
            failures.append(
                {
                    "well": well,
                    "well_index": well_index,
                    "error": "well status row-count drift",
                }
            )
            continue
        status_frames.append(status)

    audit_root = args.classification_root / "late_death_refinement"
    write_frame_atomic(
        audit_root / "failures.csv",
        pd.DataFrame(
            failures,
            columns=("well", "well_index", "error"),
        ),
    )
    if failures:
        raise RuntimeError(
            f"Death-classification consensus failed for {len(failures)} wells"
        )

    status_frame = pd.concat(status_frames, ignore_index=True, sort=False)
    if status_frame.duplicated(["branch", "key"]).any():
        raise ValueError("Duplicate branch/key rows across well status shards")
    expected_pairs = set(
        zip(
            inventory["branch"].astype(str),
            inventory["key"].astype(str),
        )
    )
    actual_pairs = set(
        zip(
            status_frame["branch"].astype(str),
            status_frame["key"].astype(str),
        )
    )
    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)[:10]
        unexpected = sorted(actual_pairs - expected_pairs)[:10]
        raise ValueError(
            "Final refinement field set does not match prepared inventory: "
            f"missing={missing} unexpected={unexpected}"
        )
    status_frame = status_frame.sort_values(["branch", "key"]).reset_index(
        drop=True
    )
    if len(status_frame) != 2 * args.expected_fields_per_branch:
        raise ValueError(
            f"Expected {2 * args.expected_fields_per_branch} refined fields, "
            f"found {len(status_frame)}"
        )
    branch_counts = status_frame.groupby("branch")["key"].nunique().to_dict()
    for branch in BRANCH_DIRS:
        if int(branch_counts.get(branch, 0)) != args.expected_fields_per_branch:
            raise ValueError(
                f"Expected {args.expected_fields_per_branch} finalized fields "
                f"for {branch}, found {branch_counts.get(branch, 0)}"
            )
    write_frame_atomic(audit_root / "refinement_summary.csv", status_frame)
    summary_paths = {
        branch: str(
            merge_branch_summaries(
                args.classification_root,
                branch,
                args.expected_fields_per_branch,
            )
        )
        for branch in BRANCH_DIRS
    }
    consensus_summary_path = write_consensus_summary(
        args.classification_root,
        summary_paths,
    )
    production_receipt_path, production_receipt = write_production_go_no_go(
        args.classification_root,
        status_frame,
        consensus_summary_path,
        calibration_receipt,
        freeze_receipt,
        args.expected_fields_per_branch,
    )
    density_definitions = json.loads(paths["density_definitions"].read_text())
    reference_counts = json.loads(paths["reference_counts"].read_text())
    config_payload = {
        "method_version": METHOD_VERSION,
        "method": (
            "dual_branch_consensus_with_continuous_density_time_calibration_"
            "multiframe_tracking_and_recoverable_field_state"
        ),
        "metric_semantics": "operational_model_without_manual_object_ground_truth",
        "field_configuration": FIELD_CONFIGURATION,
        "object_configuration": OBJECT_CONFIGURATION,
        "late_min_hours": LATE_MIN_HOURS,
        "approved_configuration_source": MODEL.APPROVED_CONFIGURATION_SOURCE,
        "density_definitions": density_definitions,
        "reference_counts": reference_counts,
        "fields_per_branch": args.expected_fields_per_branch,
        "total_fields": len(status_frame),
        "total_rescued": int(status_frame["rescued"].sum()),
        "total_uncertain": int(status_frame["uncertain"].sum()),
        "summary_paths": summary_paths,
        "authoritative_consensus_summary": str(consensus_summary_path),
        "calibration_go_no_go": str(args.calibration_go_no_go),
        "calibration_best_configuration": str(
            calibration_configuration_path
        ),
        "segmentation_freeze_receipt": str(args.segmentation_freeze_receipt),
        "production_go_no_go": str(production_receipt_path),
        "production_decision": production_receipt["decision"],
        "dataset_root": str(args.dataset_root),
        "execution_model": "one_slurm_array_task_per_well_then_finalize",
        "well_count": int(len(manifest)),
        "prepared_refinement_receipt": str(paths["receipt"]),
        "prepared_refinement_receipt_sha256": file_sha256(paths["receipt"]),
    }
    frozen_model_payload = {
        "method_version": METHOD_VERSION,
        "field_configuration": FIELD_CONFIGURATION,
        "object_configuration": OBJECT_CONFIGURATION,
        "late_min_hours": LATE_MIN_HOURS,
        "approved_configuration_source": MODEL.APPROVED_CONFIGURATION_SOURCE,
    }
    config_payload["frozen_model_sha256"] = hashlib.sha256(
        json.dumps(frozen_model_payload, sort_keys=True).encode()
    ).hexdigest()
    config_text = json.dumps(config_payload, sort_keys=True)
    config_payload["configuration_sha256"] = hashlib.sha256(
        config_text.encode()
    ).hexdigest()
    write_json_atomic(audit_root / "production_configuration.json", config_payload)
    print(f"refinement_summary={audit_root / 'refinement_summary.csv'}")
    print(f"production_configuration={audit_root / 'production_configuration.json'}")
    print(f"total_rescued={config_payload['total_rescued']}")
    print(f"total_uncertain={config_payload['total_uncertain']}")
    print(f"production_go_no_go={production_receipt_path}")
    print(f"production_decision={production_receipt['decision']}")
    return config_payload


def main() -> int:
    args = parse_args()
    args.classification_root = args.classification_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    if (
        args.workers <= 0
        or args.expected_fields_per_branch <= 0
        or args.expected_wells <= 0
    ):
        raise ValueError(
            "--workers, --expected-fields-per-branch, and --expected-wells "
            "must be positive"
        )
    if args.mode == "well" and args.well_index is None:
        raise ValueError("--well-index is required for --mode well")
    if args.mode != "well" and args.well_index is not None:
        raise ValueError("--well-index is only valid for --mode well")
    for required in (
        args.classification_root,
        args.dataset_root,
        args.calibration_go_no_go,
        args.segmentation_freeze_receipt,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    (
        calibration_receipt,
        _calibration_configuration,
        freeze_receipt,
        calibration_configuration_path,
    ) = load_production_inputs(args)

    if args.mode in {"all", "prepare"}:
        manifest, receipt = prepare_refinement_state(args)
        print(f"prepared_wells={len(manifest)}")
        print(f"prepared_receipt={prepared_paths(args.dataset_root)['receipt']}")
        print(f"prepared_reference_groups={receipt['reference_group_count']}")
        if args.mode == "prepare":
            return 0

    if args.mode == "well":
        if not 1 <= int(args.well_index) <= args.expected_wells:
            raise ValueError(
                f"--well-index must be between 1 and {args.expected_wells}"
            )
        run_one_well(args, int(args.well_index))
        return 0

    if args.mode == "all":
        for well_index in range(1, args.expected_wells + 1):
            run_one_well(args, well_index)
            if well_index == 1 or well_index % 10 == 0:
                print(
                    f"refined_wells={well_index}/{args.expected_wells}",
                    flush=True,
                )

    finalize_refinement(
        args,
        calibration_receipt,
        calibration_configuration_path,
        freeze_receipt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
