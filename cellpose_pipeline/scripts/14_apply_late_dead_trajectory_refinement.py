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
import hashlib
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile


METHOD_VERSION = "late_dead_trajectory_v1_20260723"
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


WORKER_REFERENCES: dict[tuple[str, str, str], np.ndarray] = {}
WORKER_FIELD_STATES = pd.DataFrame()
WORKER_ARGS: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--expected-fields-per-branch", type=int, default=27200)
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


def bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


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
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    result = fields.copy()
    result["treated"] = bool_series(result["treated"])
    result["density_score"] = (
        np.log1p(MODEL.finite_numeric(result["field_countable_count"]))
        + 2.5
        * MODEL.finite_numeric(result["field_mask_fraction_ratio_to_peak"])
    )
    result["density_bin"] = ""
    definitions: dict[str, dict[str, float]] = {}
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
        }
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
    return states, definitions


def build_d0_references(
    inventory: pd.DataFrame,
    field_states: pd.DataFrame,
) -> tuple[dict[tuple[str, str, str], np.ndarray], dict[str, int]]:
    field_density = field_states[
        ["branch", "key", "density_bin"]
    ].drop_duplicates(["branch", "key"])
    references: dict[tuple[str, str, str], list[np.ndarray]] = {}
    d0_inventory = inventory.loc[inventory["cohort"].eq("d0_frozen")]
    for row in d0_inventory.itertuples(index=False):
        shard = Path(row.shard_path)
        frame = pd.read_csv(shard)
        density_rows = field_density.loc[
            field_density["branch"].eq(str(row.branch))
            & field_density["key"].eq(str(row.key)),
            "density_bin",
        ]
        if len(density_rows) != 1:
            raise ValueError(f"Missing density bin for {row.branch}/{row.key}")
        density_bin = str(density_rows.iloc[0])
        eligible = (
            frame["proxy_type"].eq("d0_live_anchor")
            & bool_series(frame["countable"])
            & ~bool_series(frame["border_touching"])
        )
        live = frame.loc[eligible]
        for _output, (raw, _direction) in MODEL.RAW_FEATURES.items():
            values = pd.to_numeric(live[raw], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            references.setdefault(
                (str(row.branch), density_bin, raw),
                [],
            ).append(values)
    arrays: dict[tuple[str, str, str], np.ndarray] = {}
    counts: dict[str, int] = {}
    for key, pieces in references.items():
        values = np.concatenate(pieces) if pieces else np.empty(0, dtype=float)
        if values.size < 20:
            raise ValueError(f"Insufficient d0 live reference values for {key}: {values.size}")
        arrays[key] = values
        counts[":".join(key)] = int(values.size)
    expected = {
        (branch, density, raw)
        for branch in BRANCH_DIRS
        for density in ("low", "middle", "high")
        for raw, _direction in MODEL.RAW_FEATURES.values()
    }
    missing = sorted(expected - set(arrays))
    if missing:
        raise ValueError(f"Missing d0 reference groups: {missing}")
    return arrays, counts


def initialize_worker(
    references: dict[tuple[str, str, str], np.ndarray],
    field_states: pd.DataFrame,
    worker_args: dict[str, Any],
) -> None:
    global WORKER_REFERENCES, WORKER_FIELD_STATES, WORKER_ARGS
    WORKER_REFERENCES = references
    WORKER_FIELD_STATES = field_states
    WORKER_ARGS = worker_args


def calibrate_group_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    for (branch, density_bin), indices in result.groupby(
        ["branch", "density_bin"]
    ).groups.items():
        for output, (raw, direction) in MODEL.RAW_FEATURES.items():
            reference = WORKER_REFERENCES[(str(branch), str(density_bin), raw)]
            target = pd.to_numeric(
                result.loc[indices, raw],
                errors="coerce",
            ).to_numpy(float)
            result.loc[indices, output] = MODEL.empirical_percentile(
                reference,
                target,
                direction,
            )
    if result[list(MODEL.MODEL_FEATURES)].isna().any().any():
        raise ValueError("Production-calibrated object features contain missing values")
    return result


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
    annotations = field_rows[
        [
            "combined_mask_id",
            "field_global_late_death",
            "field_collapse_signal_count",
            "field_site_concordance",
            "density_bin",
            "death_signal_count",
            "healthy_signal_count",
            "death_score",
            "strong_live_evidence",
            "late_dead_object_evidence",
            "temporal_carryforward_call",
            "global_late_death_rescue_call",
            "late_death_rescue_call",
            "late_death_uncertain",
            *MODEL.MODEL_FEATURES,
        ]
    ].copy()
    annotation_names = {
        "field_global_late_death": "late_death_field_global",
        "field_collapse_signal_count": "late_death_field_signal_count",
        "field_site_concordance": "late_death_site_concordance",
        "density_bin": "late_death_density_bin",
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
    updated_features.loc[rescue, ["state", "final_state"]] = "dead"
    updated_features.loc[rescue, "final_reason"] = (
        "late_death_trajectory_rescue"
    )
    updated_features.loc[rescue, "classification_confidence"] = "medium"

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
    updated_predictions.loc[pred_rescue, "state"] = "dead"
    updated_predictions.loc[pred_rescue, "final_reason"] = (
        "late_death_trajectory_rescue"
    )
    updated_predictions.loc[pred_rescue, "classification_confidence"] = (
        "medium"
    )
    if int(rescue.sum()) != int(pred_rescue.sum()):
        raise ValueError(f"Feature/prediction rescue mismatch for {branch}/{key}")

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
        "baseline_dead": int(field_rows["current_dead_call"].sum()),
        "refined_dead": int(field_rows["final_dead_call"].sum()),
        "rescued": int(field_rows["late_death_rescue_call"].sum()),
        "uncertain": int(field_rows["late_death_uncertain"].sum()),
        "objects": int(len(field_rows)),
        "annotation_path": str(annotation_path),
        "rescued_mask_path": str(rescued_mask_path) if rescued_ids.size else "",
    }
    write_json_atomic(status_path, status)
    return status


def refine_group(task: dict[str, Any]) -> list[dict[str, Any]]:
    branch = str(task["branch"])
    well = str(task["well"])
    frames: list[pd.DataFrame] = []
    for shard_path in task["shards"]:
        frame = pd.read_csv(shard_path)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    for column in (
        "countable",
        "border_touching",
        "treated",
        "cyclophosphamide",
    ):
        data[column] = bool_series(data[column])
    data["elapsed_hours"] = MODEL.finite_numeric(data["elapsed_hours"])
    density = WORKER_FIELD_STATES[
        ["branch", "key", "density_bin"]
    ].drop_duplicates(["branch", "key"])
    data = data.merge(
        density,
        on=["branch", "key"],
        how="left",
        validate="many_to_one",
    )
    if data["density_bin"].isna().any():
        raise ValueError(f"Missing density bins for {branch}/{well}")
    calibrated = calibrate_group_features(data)
    field_subset = WORKER_FIELD_STATES.loc[
        WORKER_FIELD_STATES["branch"].eq(branch)
        & WORKER_FIELD_STATES["well"].eq(well)
    ]
    calls = MODEL.classification_calls(
        calibrated,
        field_subset,
        OBJECT_CONFIGURATION,
        LATE_MIN_HOURS,
        apply_treatment_scope=True,
    )
    statuses = []
    for _key, rows in calls.groupby("key", sort=False):
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


def main() -> int:
    args = parse_args()
    args.classification_root = args.classification_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    if args.workers <= 0 or args.expected_fields_per_branch <= 0:
        raise ValueError("--workers and --expected-fields-per-branch must be positive")
    for required in (args.classification_root, args.dataset_root):
        if not required.exists():
            raise FileNotFoundError(required)
    inventory = pd.read_csv(
        args.dataset_root / "feature_cache" / "shard_inventory.csv"
    )
    fields = pd.read_csv(
        args.dataset_root / "feature_cache" / "field_trajectory_metrics.csv"
    )
    if inventory.empty or fields.empty:
        raise ValueError("Late-death dataset inventory and field metrics must be non-empty")
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

    field_states, density_definitions = prepare_field_states(fields)
    references, reference_counts = build_d0_references(inventory, field_states)
    task_rows: list[dict[str, Any]] = []
    for (branch, well), rows in inventory.groupby(["branch", "well"], sort=True):
        task_rows.append(
            {
                "branch": str(branch),
                "well": str(well),
                "shards": [str(value) for value in rows["shard_path"]],
            }
        )
    worker_args = {
        "classification_root": str(args.classification_root),
        "overlay_alpha": args.overlay_alpha,
        "force": args.force,
    }
    statuses: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_worker,
        initargs=(references, field_states, worker_args),
    ) as executor:
        futures = {executor.submit(refine_group, task): task for task in task_rows}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                statuses.extend(future.result())
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "branch": str(task["branch"]),
                        "well": str(task["well"]),
                        "error": repr(exc),
                    }
                )
            if completed == 1 or completed % 10 == 0 or completed == len(task_rows):
                print(
                    f"refined_groups={completed}/{len(task_rows)} "
                    f"fields={len(statuses)} failures={len(failures)}",
                    flush=True,
                )
    audit_root = args.classification_root / "late_death_refinement"
    write_frame_atomic(
        audit_root / "failures.csv",
        pd.DataFrame(failures, columns=("branch", "well", "error")),
    )
    if failures:
        raise RuntimeError(
            f"Late-death refinement failed for {len(failures)} branch/well groups"
        )
    status_frame = pd.DataFrame(statuses).sort_values(["branch", "key"])
    if len(status_frame) != 2 * args.expected_fields_per_branch:
        raise ValueError(
            f"Expected {2 * args.expected_fields_per_branch} refined fields, "
            f"found {len(status_frame)}"
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
    config_payload = {
        "method_version": METHOD_VERSION,
        "method": "density_aware_field_collapse_with_object_evidence_and_absorbing_time_state",
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
        "dataset_root": str(args.dataset_root),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
