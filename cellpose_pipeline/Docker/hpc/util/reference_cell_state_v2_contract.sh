#!/usr/bin/env bash

# Shared, source-only contract for the V2 historical-reference cell-state
# workflow.  This file intentionally contains no executable top-level work.

REFERENCE_CELL_STATE_V2_SCHEMA="reference_cell_state_hpc_contract_v2"
REFERENCE_CELL_STATE_V2_DEFAULT_EXPERIMENT_ROOT="/share/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/20260619_SUM159_Doxorubicin_Cyclophosphamide"
REFERENCE_CELL_STATE_V2_IDENTITY_LOCK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)/reference_cell_state_v2_sif_identity.tsv"
REFERENCE_CELL_STATE_V2_DEFAULT_CPA_ROOT="/share/lab_crd/taoli/Dependencies/cell-phenotype-annotator/7d1e23077efb85d503ca5333df76acab1ae3640c"
REFERENCE_CELL_STATE_V2_EXPECTED_CPA_COMMIT="7d1e23077efb85d503ca5333df76acab1ae3640c"
REFERENCE_CELL_STATE_V2_EXPECTED_CPA_TREE="7cd003ae4be29a5851ba305d92249cd9f8ff888e"
REFERENCE_CELL_STATE_V2_FORWARD_PREFIXES="PYTHONNOUSERSITE,KMP_DUPLICATE_LIB_OK,MPLCONFIGDIR,CUDA_VISIBLE_DEVICES"

reference_cell_state_v2_abort() {
  printf '%s\n' "$*" >&2
  return 2
}

# Deterministic failure injection used only by the local recovery-contract
# tests.  The formal submitter never forwards either variable to workers, so a
# production Slurm job cannot enable this path through its ambient environment.
reference_cell_state_v2_test_maybe_fail() {
  local boundary="$1"
  [[ -z "${REFERENCE_CELL_STATE_V2_TEST_FAIL_AFTER_BOUNDARY:-}" ]] && return 0
  [[ "${REFERENCE_CELL_STATE_V2_TEST_MODE:-0}" == 1 ]] || {
    reference_cell_state_v2_abort \
      "V2 failure injection is forbidden outside explicit test mode"
    return 2
  }
  [[ "$REFERENCE_CELL_STATE_V2_TEST_FAIL_AFTER_BOUNDARY" != "$boundary" ]] || {
    printf 'reference_cell_state_v2_injected_failure_after=%s\n' "$boundary" >&2
    return 86
  }
}

reference_cell_state_v2_sha256() {
  local path="$1" observed
  observed="$(sha256sum "$path")" || return 1
  printf '%s\n' "${observed%%[[:space:]]*}"
}

reference_cell_state_v2_load_sif_identity() {
  local lock_path="${1:-$REFERENCE_CELL_STATE_V2_IDENTITY_LOCK}"
  local property value header seen=""
  local schema="" status="" image_path="" image_sha256="" docker_tag=""
  local image_bytes="" r_version="" package_count="" dbscan_version="" uwot_version="" glmnet_version=""
  local dplyr_version="" tidyr_version="" purrr_version="" stringr_version="" ggplot2_version=""
  local filesystem_immutability_mode="" runtime_rootfs_read_only=""

  [[ "$lock_path" == /* && -f "$lock_path" && ! -L "$lock_path" ]] || {
    reference_cell_state_v2_abort "V2 SIF identity lock is unavailable: $lock_path"
    return 2
  }
  IFS= read -r header < "$lock_path"
  [[ "$header" == $'property\tvalue' ]] || {
    reference_cell_state_v2_abort "Malformed V2 SIF identity header: $lock_path"
    return 2
  }
  while IFS=$'\t' read -r property value extra; do
    [[ -n "$property" && -n "$value" && -z "${extra:-}" ]] || {
      reference_cell_state_v2_abort "Malformed V2 SIF identity row: $property"
      return 2
    }
    case "$seen" in *"|$property|"*) reference_cell_state_v2_abort "Duplicate V2 SIF identity property: $property"; return 2 ;; esac
    seen="$seen|$property|"
    case "$property" in
      schema_version) schema="$value" ;;
      status) status="$value" ;;
      image_path) image_path="$value" ;;
      image_sha256) image_sha256="$value" ;;
      image_bytes) image_bytes="$value" ;;
      docker_tag) docker_tag="$value" ;;
      r_version) r_version="$value" ;;
      r_locked_package_count) package_count="$value" ;;
      r_dbscan_version) dbscan_version="$value" ;;
      r_uwot_version) uwot_version="$value" ;;
      r_glmnet_version) glmnet_version="$value" ;;
      r_dplyr_version) dplyr_version="$value" ;;
      r_tidyr_version) tidyr_version="$value" ;;
      r_purrr_version) purrr_version="$value" ;;
      r_stringr_version) stringr_version="$value" ;;
      r_ggplot2_version) ggplot2_version="$value" ;;
      filesystem_immutability_mode) filesystem_immutability_mode="$value" ;;
      runtime_rootfs_read_only) runtime_rootfs_read_only="$value" ;;
      activation_rule) ;;
      *) reference_cell_state_v2_abort "Unknown V2 SIF identity property: $property"; return 2 ;;
    esac
  done < <(sed '1d' "$lock_path")

  [[ "$schema" == "reference_cell_state_v2_sif_identity_v1" && \
     "$image_path" == "/share/lab_crd/taoli/Docker/cellpose-cpsam-pipeline_hpc-cellpose-4.2.1.1-models-reference-v2-parity.sif" && \
     "$docker_tag" == "zafiro/cellpose-cpsam-pipeline:hpc-cellpose-4.2.1.1-models-reference-v2-parity" && \
     "$r_version" == "4.2.3" && "$package_count" == "66" && \
     "$dbscan_version" == "1.2.3" && "$uwot_version" == "0.2.4" && \
     "$glmnet_version" == "4.1-10" && \
     "$dplyr_version" == "1.1.4" && "$tidyr_version" == "1.3.1" && \
     "$purrr_version" == "1.2.0" && "$stringr_version" == "1.6.0" && \
     "$ggplot2_version" == "4.0.0" && \
     "$filesystem_immutability_mode" == "shared_filesystem_mode_bits_unavailable" && \
     ( "$runtime_rootfs_read_only" == "verified" || "$runtime_rootfs_read_only" == "PENDING_A30_VERIFICATION" ) ]] || {
    reference_cell_state_v2_abort "V2 SIF identity lock differs from the frozen runtime contract"
    return 2
  }
  [[ "$status" == "VERIFIED" && "$image_sha256" =~ ^[0-9a-f]{64}$ && \
     "$image_bytes" =~ ^[1-9][0-9]*$ && "$runtime_rootfs_read_only" == "verified" ]] || {
    reference_cell_state_v2_abort \
      "V2 SIF is not activated: build/convert/A30-verify it, then record status=VERIFIED and its live SHA-256 in $lock_path"
    return 2
  }
  REFERENCE_CELL_STATE_V2_DEFAULT_SIF="$image_path"
  REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256="$image_sha256"
  REFERENCE_CELL_STATE_V2_DOCKER_TAG="$docker_tag"
  REFERENCE_CELL_STATE_V2_EXPECTED_SIF_BYTES="$image_bytes"
  REFERENCE_CELL_STATE_V2_FILESYSTEM_IMMUTABILITY_MODE="$filesystem_immutability_mode"
  REFERENCE_CELL_STATE_V2_RUNTIME_ROOTFS_READ_ONLY="$runtime_rootfs_read_only"
  export REFERENCE_CELL_STATE_V2_DEFAULT_SIF REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256
  export REFERENCE_CELL_STATE_V2_DOCKER_TAG REFERENCE_CELL_STATE_V2_EXPECTED_SIF_BYTES
  export REFERENCE_CELL_STATE_V2_FILESYSTEM_IMMUTABILITY_MODE REFERENCE_CELL_STATE_V2_RUNTIME_ROOTFS_READ_ONLY
}

reference_cell_state_v2_require_runtime_identity() {
  local sif="$1" cpa_root="$2" observed_sif observed_commit observed_tree observed_status

  [[ -n "${REFERENCE_CELL_STATE_V2_DEFAULT_SIF:-}" && \
     -n "${REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256:-}" ]] || \
    reference_cell_state_v2_load_sif_identity || return 2
  [[ "$sif" == "$REFERENCE_CELL_STATE_V2_DEFAULT_SIF" ]] || {
    reference_cell_state_v2_abort \
      "HPC_CONTAINER_IMAGE must equal the frozen latest V2 SIF path: $REFERENCE_CELL_STATE_V2_DEFAULT_SIF"
    return 2
  }
  [[ "$sif" == /* && -r "$sif" && ! -L "$sif" ]] || {
    reference_cell_state_v2_abort "The frozen latest V2 SIF is unavailable or symlinked: $sif"
    return 2
  }
  observed_sif="$(reference_cell_state_v2_sha256 "$sif")" || return 2
  [[ "$observed_sif" == "$REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256" ]] || {
    reference_cell_state_v2_abort \
      "Live V2 SIF SHA-256 mismatch: expected=$REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256 observed=$observed_sif"
    return 2
  }
  [[ "$(stat -c %s "$sif")" == "$REFERENCE_CELL_STATE_V2_EXPECTED_SIF_BYTES" ]] || {
    reference_cell_state_v2_abort "Live V2 SIF byte size differs from the frozen identity"
    return 2
  }

  [[ "$cpa_root" == "$REFERENCE_CELL_STATE_V2_DEFAULT_CPA_ROOT" ]] || {
    reference_cell_state_v2_abort \
      "CPA_REFERENCE_ROOT must equal the frozen read-only checkout: $REFERENCE_CELL_STATE_V2_DEFAULT_CPA_ROOT"
    return 2
  }
  [[ "$cpa_root" == /* && -d "$cpa_root" && ! -L "$cpa_root" ]] || {
    reference_cell_state_v2_abort "Pinned V2 reference checkout is unavailable or symlinked: $cpa_root"
    return 2
  }
  observed_commit="$(GIT_OPTIONAL_LOCKS=0 git -C "$cpa_root" rev-parse HEAD)" || return 2
  observed_tree="$(GIT_OPTIONAL_LOCKS=0 git -C "$cpa_root" rev-parse 'HEAD^{tree}')" || return 2
  observed_status="$(GIT_OPTIONAL_LOCKS=0 git -C "$cpa_root" status --porcelain --untracked-files=all)" || return 2
  [[ "$observed_commit" == "$REFERENCE_CELL_STATE_V2_EXPECTED_CPA_COMMIT" && \
     "$observed_tree" == "$REFERENCE_CELL_STATE_V2_EXPECTED_CPA_TREE" && \
     -z "$observed_status" ]] || {
    reference_cell_state_v2_abort "Pinned V2 Cell Phenotype Annotator checkout identity mismatch"
    return 2
  }
}

reference_cell_state_v2_verify_container_rootfs_read_only() {
  command -v hpc_apptainer_exec >/dev/null 2>&1 || {
    reference_cell_state_v2_abort "V2 rootfs verification requires the prepared Apptainer runtime"
    return 2
  }
  hpc_apptainer_exec sh -c '
    probe="/.reference-cell-state-v2-rootfs-write-probe-$$"
    if : > "$probe" 2>/dev/null; then
      rm -f -- "$probe"
      echo "V2 SIF root filesystem unexpectedly accepted a write" >&2
      exit 2
    fi
    printf "%s\n" "reference_v2_runtime_rootfs_read_only=PASS"
  '
}

reference_cell_state_v2_require_shadow_root() {
  local path="$1" mode="$2" results_root="$3" resolved_path resolved_results expected_parent name
  [[ "$path" == /* && -d "$path" && ! -L "$path" && \
     "$results_root" == /* && -d "$results_root" && ! -L "$results_root" ]] || {
    reference_cell_state_v2_abort "REFERENCE_SHADOW_ROOT must be an absolute real directory: $path"
    return 2
  }
  resolved_path="$(realpath "$path")" || return 2
  resolved_results="$(realpath "$results_root")" || return 2
  [[ "$path" == "$resolved_path" && "$results_root" == "$resolved_results" ]] || {
    reference_cell_state_v2_abort "V2 root/results paths must use their canonical realpath spelling"
    return 2
  }
  name="$(basename "$resolved_path")"
  case "$mode" in
    formal)
      [[ "$(dirname "$resolved_path")" == "$resolved_results" && \
         "$name" == reference_cell_state_shadow_v2_* && \
         "$name" != reference_cell_state_shadow_v2_test_* ]] || {
        reference_cell_state_v2_abort \
          "Formal V2 root must be a canonical reference_cell_state_shadow_v2_* direct child of RESULTS_ROOT: $path"
        return 2
      }
      ;;
    calibration)
      expected_parent="$resolved_results/Tests_and_Parameters_calibration"
      [[ -d "$expected_parent" && ! -L "$expected_parent" ]] || {
        reference_cell_state_v2_abort "V2 calibration namespace is unavailable"
        return 2
      }
      expected_parent="$(realpath "$expected_parent")" || return 2
      [[ "$(dirname "$resolved_path")" == "$expected_parent" && \
         "$name" == reference_cell_state_shadow_v2_test_* ]] || {
        reference_cell_state_v2_abort \
          "V2 calibration root must be a canonical direct child of Tests_and_Parameters_calibration: $path"
        return 2
      }
      ;;
    *) reference_cell_state_v2_abort "Unknown V2 root mode: $mode"; return 2 ;;
  esac
  case "$name" in
    reference_cell_state_shadow_v2_*|reference_cell_state_shadow_v2_test_*) ;;
    *)
      reference_cell_state_v2_abort \
        "A V1 or ambiguous reference-cell-state root cannot be used by a V2 worker: $path"
      return 2
      ;;
  esac
}

reference_cell_state_v2_require_inside() {
  local path="$1" root="$2" label="$3" resolved_path resolved_root
  [[ "$path" == /* && -e "$path" && ! -L "$path" ]] || {
    reference_cell_state_v2_abort "$label must be absolute, existing, and non-symlinked: $path"
    return 2
  }
  resolved_path="$(realpath "$path")" || return 2
  resolved_root="$(realpath "$root")" || return 2
  case "$resolved_path" in
    "$resolved_root"|"$resolved_root"/*) ;;
    *) reference_cell_state_v2_abort "$label escaped its frozen root: $resolved_path"; return 2 ;;
  esac
}

reference_cell_state_v2_require_new_output() {
  local path="$1" root="$2" label="$3" parent resolved_root resolved_parent
  [[ "$path" == /* && ! -e "$path" && ! -L "$path" ]] || {
    reference_cell_state_v2_abort "$label must be a new absolute non-symlink path: $path"
    return 2
  }
  case "/$path/" in
    *'/../'*|*'/./'*) reference_cell_state_v2_abort "$label is not canonical: $path"; return 2 ;;
  esac
  parent="$(dirname "$path")"
  while [[ ! -e "$parent" ]]; do parent="$(dirname "$parent")"; done
  [[ -d "$parent" && ! -L "$parent" ]] || {
    reference_cell_state_v2_abort "$label has an unsafe ancestor: $parent"
    return 2
  }
  resolved_root="$(realpath "$root")" || return 2
  resolved_parent="$(realpath "$parent")" || return 2
  case "$resolved_parent" in
    "$resolved_root"|"$resolved_root"/*) ;;
    *) reference_cell_state_v2_abort "$label ancestor escaped its frozen root"; return 2 ;;
  esac
}

reference_cell_state_v2_require_recoverable_output() {
  local path="$1" root="$2" label="$3" ancestor resolved_root resolved_ancestor
  [[ "$path" == /* && ! -L "$path" ]] || {
    reference_cell_state_v2_abort "$label must be an absolute non-symlink path: $path"
    return 2
  }
  case "/$path/" in
    *'/../'*|*'/./'*) reference_cell_state_v2_abort "$label is not canonical: $path"; return 2 ;;
  esac
  if [[ -e "$path" && ! -f "$path" ]]; then
    reference_cell_state_v2_abort "$label recovery target is not a regular file: $path"
    return 2
  fi
  ancestor="$(dirname "$path")"
  while [[ ! -e "$ancestor" ]]; do ancestor="$(dirname "$ancestor")"; done
  [[ -d "$ancestor" && ! -L "$ancestor" ]] || {
    reference_cell_state_v2_abort "$label has an unsafe ancestor: $ancestor"
    return 2
  }
  resolved_root="$(realpath "$root")" || return 2
  resolved_ancestor="$(realpath "$ancestor")" || return 2
  case "$resolved_ancestor" in
    "$resolved_root"|"$resolved_root"/*) ;;
    *) reference_cell_state_v2_abort "$label ancestor escaped its frozen root"; return 2 ;;
  esac
}

reference_cell_state_v2_method_parity_status() {
  local shadow_root="$1"
  local audit
  audit="$(reference_cell_state_v2_audit_phase_a "$shadow_root")" || return 2
  awk -F '\t' '$1=="method_parity_status"{print $2; n++} END{if(n!=1) exit 2}' <<< "$audit"
}

reference_cell_state_v2_phase_a_branch() {
  local shadow_root="$1" audit
  audit="$(reference_cell_state_v2_audit_phase_a "$shadow_root")" || return 2
  awk -F '\t' '$1=="one_cluster_fallback"{print $2; n++} END{if(n!=1) exit 2}' <<< "$audit"
}

reference_cell_state_v2_audit_phase_a() {
  local shadow_root="$1"
  local receipt="$shadow_root/workflow_status/reference_cell_state_phase_a_v2/PHASE_A_COMPLETE.tsv"
  local expected_image="${REFERENCE_CELL_STATE_V2_DEFAULT_SIF:-}"
  local expected_image_sha="${REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256:-}"
  python3 -I - "$receipt" "$shadow_root" "$expected_image" "$expected_image_sha" <<'PY'
import hashlib
import csv
import json
import os
import pathlib
import sys

receipt, root = map(pathlib.Path, sys.argv[1:3])
expected_image, expected_image_sha = sys.argv[3:5]
if receipt.is_symlink() or not receipt.is_file():
    raise SystemExit("Phase A receipt is not a regular file")
lines = receipt.read_text(encoding="utf-8").splitlines()
if not lines or lines[0] != "property\tvalue":
    raise SystemExit("Phase A receipt header is malformed")
rows = {}
for line in lines[1:]:
    key, sep, value = line.partition("\t")
    if not sep or not key or key in rows:
        raise SystemExit("Phase A receipt is malformed or duplicates a property")
    rows[key] = value
if rows.get("schema_version") != "reference_cell_state_phase_a_receipt_v2" or rows.get("status") != "COMPLETE":
    raise SystemExit("Phase A receipt is not COMPLETE V2")
if pathlib.Path(rows.get("reference_shadow_root", "")) != root:
    raise SystemExit("Phase A receipt root changed")
if expected_image and pathlib.Path(rows.get("hpc_container_image", "")) != pathlib.Path(expected_image):
    raise SystemExit("Phase A receipt container path changed")
if expected_image_sha and rows.get("hpc_container_sha256") != expected_image_sha:
    raise SystemExit("Phase A receipt SIF SHA changed")

root_resolved = pathlib.Path(os.path.realpath(root))
def verify(path_key, hash_key, *, confined):
    raw = rows.get(path_key)
    digest = rows.get(hash_key)
    if not raw or not digest or len(digest) != 64:
        raise SystemExit(f"Phase A receipt lacks {path_key}/{hash_key}")
    path = pathlib.Path(raw)
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"Phase A artifact missing: {path_key}")
    if confined:
        resolved = pathlib.Path(os.path.realpath(path))
        if resolved != root_resolved and root_resolved not in resolved.parents:
            raise SystemExit(f"Phase A artifact escaped root: {path_key}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit(f"Phase A artifact hash changed: {path_key}")
    return path

verify("project_file", "project_sha256", confined=True)
verify("plate_map", "plate_map_sha256", confined=False)
historical_path = verify(
    "historical_projection_manifest", "historical_projection_manifest_sha256", confined=True
)
verify("runtime_package_receipt", "runtime_package_receipt_sha256", confined=True)
annotation_manifest = verify("annotation_manifest", "annotation_manifest_sha256", confined=True)
morphology_overlay = verify(
    "morphology_overlay_manifest", "morphology_overlay_manifest_sha256", confined=True
)
morphology_workspace = verify(
    "morphology_workspace", "morphology_workspace_sha256", confined=True
)
verify("dependency_lock", "dependency_lock_sha256", confined=False)
if pathlib.Path(rows.get("annotation_dir", "")) != annotation_manifest.parent:
    raise SystemExit("Phase A annotation_dir differs from the frozen annotation manifest")

with morphology_overlay.open(encoding="utf-8") as handle:
    overlay_payload = json.load(handle)
if overlay_payload.get("schema_version") != "reference_morphology_workspace_v2" or overlay_payload.get("status") != "COMPLETE":
    raise SystemExit("Morphology overlay manifest is not COMPLETE V2")
declared_overlay = overlay_payload.get("output_artifact_sha256")
if not isinstance(declared_overlay, dict) or not declared_overlay:
    raise SystemExit("Morphology overlay manifest lacks its artifact hashes")
overlay_root = morphology_overlay.parent
observed_overlay = {}
for candidate in sorted(overlay_root.rglob("*"), key=lambda path: str(path)):
    if candidate == morphology_overlay:
        continue
    if candidate.is_symlink():
        raise SystemExit("Morphology generation contains a symlink")
    if candidate.is_file():
        observed_overlay[candidate.relative_to(overlay_root).as_posix()] = hashlib.sha256(candidate.read_bytes()).hexdigest()
if observed_overlay != declared_overlay:
    raise SystemExit("Morphology overlay artifact inventory/hash changed")

with historical_path.open(encoding="utf-8") as handle:
    historical = json.load(handle)
if historical.get("schema_version") != "reference_cell_state_historical_projection_v2" or historical.get("status") != "COMPLETE":
    raise SystemExit("Historical projection manifest is not COMPLETE V2")
fallback = (historical.get("diagnostic_cluster") or {}).get("one_cluster_fallback")
if not isinstance(fallback, bool):
    raise SystemExit("Historical projection branch is not boolean")
fallback_text = "true" if fallback else "false"
if rows.get("one_cluster_fallback") != fallback_text:
    raise SystemExit("Phase A receipt branch differs from historical manifest")

if fallback:
    expected = {
        "method_parity_status": "historical_core_parity_with_disclosed_no_stable_cluster_sampling_adaptation",
        "review_sampling_contract": "disclosed_seed1_all_unassigned_well_balanced_max500_and_seed2_probability_surrogate_not_pinned_reference_exact_sampling",
        "human_barrier": "seed1_adapted_exact_review_submission_required",
    }
    for key, value in expected.items():
        if rows.get(key) != value:
            raise SystemExit(f"Fallback Phase A contract changed: {key}")
    for path_key, hash_key in (
        ("seed1_selection_manifest", "seed1_selection_manifest_sha256"),
        ("seed1_review_set", "seed1_review_set_sha256"),
        ("seed1_render_manifest", "seed1_render_manifest_sha256"),
        ("seed1_exact_review_html", "seed1_exact_review_html_sha256"),
        ("seed1_crop_status", "seed1_crop_status_sha256"),
    ):
        verify(path_key, hash_key, confined=True)
    selection_manifest = pathlib.Path(rows["seed1_selection_manifest"])
    with selection_manifest.open(encoding="utf-8") as handle:
        selection_payload = json.load(handle)
    if selection_payload.get("schema_version") != "reference_cell_state_seed1_review_v2" or selection_payload.get("status") != "HUMAN_REVIEW_REQUIRED":
        raise SystemExit("Fallback Seed1 selection manifest is invalid")
    selection_outputs = selection_payload.get("output_file_sha256")
    selection_files = {
        "review_set": "seed1_review_set.tsv",
        "label_template": "seed1_review_label_template.tsv",
        "selection_audit": "seed1_selection_audit.tsv",
    }
    if not isinstance(selection_outputs, dict) or set(selection_outputs) != set(selection_files):
        raise SystemExit("Fallback Seed1 selection artifact set is incomplete")
    for role, name in selection_files.items():
        candidate = selection_manifest.parent / name
        if candidate.is_symlink() or not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest() != selection_outputs[role]:
            raise SystemExit(f"Fallback Seed1 selection artifact changed: {role}")

    render_manifest = pathlib.Path(rows["seed1_render_manifest"])
    with render_manifest.open(encoding="utf-8") as handle:
        render_payload = json.load(handle)
    if render_payload.get("schema_version") != "reference_cell_state_exact_review_render_v2" or render_payload.get("status") != "HUMAN_REVIEW_REQUIRED":
        raise SystemExit("Fallback Seed1 render manifest is invalid")
    render_files = {
        "exact_review_html": "exact_review.html",
        "crop_render_status": "crop_render_status.tsv",
        "crop_manifest": "crop_manifest.tsv",
    }
    declared_render = render_payload.get("artifact_file_sha256")
    if not isinstance(declared_render, dict) or set(declared_render) != set(render_files):
        raise SystemExit("Fallback Seed1 render artifact set is incomplete")
    for role, name in render_files.items():
        candidate = render_manifest.parent / name
        if candidate.is_symlink() or not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest() != declared_render[role]:
            raise SystemExit(f"Fallback Seed1 render artifact changed: {role}")
    crop_manifest = render_manifest.parent / "crop_manifest.tsv"
    if hashlib.sha256(crop_manifest.read_bytes()).hexdigest() != render_payload.get("crop_aggregate_sha256"):
        raise SystemExit("Fallback Seed1 crop aggregate hash changed")
    with crop_manifest.open(encoding="utf-8", newline="") as handle:
        crop_rows = list(csv.DictReader(handle, delimiter="\t"))
    required_crop_fields = {"morphology_umap_row_key", "channel", "relative_path", "sha256"}
    if not crop_rows or set(crop_rows[0]) != required_crop_fields:
        raise SystemExit("Fallback Seed1 crop manifest schema is invalid")
    render_root = pathlib.Path(os.path.realpath(render_manifest.parent))
    for crop_row in crop_rows:
        crop = render_manifest.parent / crop_row["relative_path"]
        crop_resolved = pathlib.Path(os.path.realpath(crop))
        if crop.is_symlink() or not crop.is_file() or (crop_resolved != render_root and render_root not in crop_resolved.parents):
            raise SystemExit("Fallback Seed1 crop escaped its render generation")
        if hashlib.sha256(crop.read_bytes()).hexdigest() != crop_row["sha256"]:
            raise SystemExit("Fallback Seed1 crop hash changed")
    if pathlib.Path(rows.get("human_workspace", "")) != pathlib.Path(rows["seed1_exact_review_html"]):
        raise SystemExit("Fallback human workspace differs from frozen exact-review HTML")
    expected_submission = pathlib.Path(rows["seed1_exact_review_html"]).parent / "exact_review_submission.json"
    if pathlib.Path(rows.get("human_submission_expected_path", "")) != expected_submission:
        raise SystemExit("Fallback human submission path differs from exact-review generation")
else:
    expected = {
        "method_parity_status": "historical_core_parity_with_pinned_stable_polygon_sampling",
        "review_sampling_contract": "pinned_reference_stable_polygon_seed1_seed2_sampling",
        "human_barrier": "region_submission_required",
    }
    for key, value in expected.items():
        if rows.get(key) != value:
            raise SystemExit(f"Stable-cluster Phase A contract changed: {key}")
    if pathlib.Path(rows.get("human_workspace", "")) != morphology_workspace:
        raise SystemExit("Stable-cluster human workspace differs from morphology workspace")
    if pathlib.Path(rows.get("human_submission_expected_path", "")) != annotation_manifest.parent / "region_submission.json":
        raise SystemExit("Stable-cluster human submission path differs from annotation generation")
    forbidden = (
        "seed1_selection_manifest", "seed1_review_set", "seed1_render_manifest",
        "seed1_exact_review_html", "seed1_crop_status",
    )
    if any(key in rows for key in forbidden):
        raise SystemExit("Stable-cluster receipt contains fallback-only evidence")

print(f"one_cluster_fallback\t{fallback_text}")
print(f"method_parity_status\t{rows['method_parity_status']}")
PY
}

reference_cell_state_v2_require_posthuman_action_branch() {
  local action="$1" fallback="$2"
  case "$fallback:$action" in
    false:post-region|false:post-seed1|false:post-seed2|false:post-adjudication|true:post-fallback-seed1|true:post-seed2|true:post-adjudication) ;;
    false:post-fallback-seed1)
      reference_cell_state_v2_abort "post-fallback-seed1 is forbidden for the stable-cluster Phase A branch"
      return 2
      ;;
    true:post-region|true:post-seed1)
      reference_cell_state_v2_abort "$action is forbidden for the one-cluster fallback Phase A branch"
      return 2
      ;;
    *) reference_cell_state_v2_abort "Unsupported posthuman action/Phase A branch: action=$action fallback=$fallback"; return 2 ;;
  esac
}

reference_cell_state_v2_ledger_init_or_verify() {
  local ledger="$1"
  local header=$'submitted_utc\tjob_name\tjob_id\tdependency_job_id\tarray_spec'
  if [[ -e "$ledger" || -L "$ledger" ]]; then
    [[ -f "$ledger" && ! -L "$ledger" && "$(sed -n '1p' "$ledger")" == "$header" ]] || return 2
    awk -F '\t' 'NR>1 && NF!=5{exit 2} NR>1 && seen[$2]++{exit 3}' "$ledger"
  else
    printf '%s\n' "$header" > "$ledger"
  fi
}

reference_cell_state_v2_ledger_record() {
  local ledger="$1" job_name="$2" job_id="$3"
  local dependency="${4:-none}" array_spec="${5:-none}" temporary="${ledger}.tmp.$$"
  [[ -z "$(awk -F '\t' -v name="$job_name" 'NR>1 && $2==name{print $2}' "$ledger")" ]] || return 2
  cp -- "$ledger" "$temporary"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$job_name" "$job_id" \
    "${dependency:-none}" "${array_spec:-none}" >> "$temporary"
  mv "$temporary" "$ledger"
}

reference_cell_state_v2_ledger_reuse() {
  local ledger="$1" job_name="$2" dependency="${3:-none}" array_spec="${4:-none}"
  local -a matches=()
  while IFS= read -r line; do [[ -z "$line" ]] || matches+=("$line"); done < <(
    awk -F '\t' -v name="$job_name" 'NR>1 && $2==name{print}' "$ledger"
  )
  [[ "${#matches[@]}" -le 1 ]] || return 2
  [[ "${#matches[@]}" -eq 1 ]] || return 1
  local submitted_utc observed_name job_id observed_dependency observed_array extra
  IFS=$'\t' read -r submitted_utc observed_name job_id observed_dependency observed_array extra <<< "${matches[0]}"
  [[ -z "${extra:-}" && "$observed_name" == "$job_name" && \
     "$observed_dependency" == "$dependency" && "$observed_array" == "$array_spec" && \
     ( "$job_id" =~ ^[0-9]+$ || "$job_id" == DRYRUN_* || "$job_id" == direct_test ) ]] || return 2
  printf '%s\n' "$job_id"
}

reference_cell_state_v2_attempt_ledger_init_or_verify() {
  local ledger="$1"
  local header=$'submitted_utc\tlogical_node\tattempt\tjob_name\tjob_id\tdependency_job_id\tarray_spec\ttrigger_state\tprior_job_id'
  if [[ -e "$ledger" || -L "$ledger" ]]; then
    [[ -f "$ledger" && ! -L "$ledger" && "$(sed -n '1p' "$ledger")" == "$header" ]] || return 2
    awk -F '\t' '
      NR == 1 { next }
      NF != 9 || $2 == "" || $3 !~ /^[1-9][0-9]*$/ ||
        ($5 !~ /^[0-9]+$/ && $5 !~ /^DRYRUN_/ && $5 != "direct_test") { exit 2 }
      {
        expected[$2]++
        if ($3 != expected[$2]) exit 3
        key = $2 SUBSEP $5
        if (seen[key]++) exit 4
        if ($3 == 1 && $9 != "none") exit 5
        if ($3 > 1 && ($9 == "none" || $9 == "")) exit 6
        if ($3 > 1 && $9 != prior_job[$2]) exit 7
        if ($3 == 1 && $8 != "INITIAL") exit 8
        if ($3 > 1 && $8 == "INITIAL") exit 9
        prior_job[$2] = $5
      }
    ' "$ledger"
  else
    printf '%s\n' "$header" > "$ledger"
  fi
}

reference_cell_state_v2_attempt_ledger_record() {
  local ledger="$1" logical_node="$2" job_name="$3" job_id="$4"
  local dependency="${5:-none}" array_spec="${6:-none}"
  local trigger_state="${7:-INITIAL}" prior_job_id="${8:-none}"
  local count attempt temporary="${ledger}.tmp.$$"
  count="$(awk -F '\t' -v node="$logical_node" 'NR>1 && $2==node{n++} END{print n+0}' "$ledger")" || return 2
  attempt=$((count + 1))
  [[ "$job_name" == "$logical_node" && \
     ( "$job_id" =~ ^[0-9]+$ || "$job_id" == DRYRUN_* || "$job_id" == direct_test ) && \
     ( "$dependency" == none || "$dependency" =~ ^[0-9]+$ || "$dependency" == DRYRUN_* || "$dependency" == direct_test ) && \
     "$trigger_state" =~ ^[A-Z_]+$ ]] || return 2
  if [[ "$attempt" -eq 1 ]]; then
    [[ "$prior_job_id" == none && "$trigger_state" == INITIAL ]] || return 2
  else
    [[ "$prior_job_id" =~ ^[0-9]+$ && "$trigger_state" != INITIAL ]] || return 2
  fi
  cp -- "$ledger" "$temporary"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$logical_node" "$attempt" \
    "$job_name" "$job_id" "$dependency" "$array_spec" "$trigger_state" "$prior_job_id" >> "$temporary"
  mv "$temporary" "$ledger"
}

reference_cell_state_v2_attempt_ledger_latest() {
  local ledger="$1" logical_node="$2"
  awk -F '\t' -v node="$logical_node" '
    NR>1 && $2==node { line=$0; n++ }
    END { if(n<1) exit 1; print line }
  ' "$ledger"
}

reference_cell_state_v2_aggregate_sacct_state() {
  local job_id="$1" array_spec="${2:-none}" sacct_output="$3"
  [[ "$job_id" =~ ^[0-9]+$ && -f "$sacct_output" && ! -L "$sacct_output" ]] || return 2
  python3 -I - "$job_id" "$array_spec" "$sacct_output" <<'PY'
import pathlib
import re
import sys

job_id, array_spec, output_name = sys.argv[1:]
output = pathlib.Path(output_name)
active = {
    "PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
    "RESIZING", "REQUEUED", "REQUEUE_FED", "SIGNALING", "STAGE_OUT",
}
terminal_failure = {
    "BOOT_FAIL", "CANCELLED", "DEADLINE", "FAILED", "NODE_FAIL",
    "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "SPECIAL_EXIT", "TIMEOUT",
}
known = active | terminal_failure | {"COMPLETED"}

def normalize_state(value):
    value = value.strip().split(maxsplit=1)[0].rstrip("+") if value.strip() else ""
    if value not in known:
        raise SystemExit(f"Unknown Slurm state in sacct allocation output: {value!r}")
    return value

rows = []
for number, line in enumerate(output.read_text(encoding="utf-8").splitlines(), start=1):
    if not line.strip():
        continue
    fields = line.split("|")
    if len(fields) < 3:
        raise SystemExit(f"Malformed sacct allocation row {number}")
    job_name, raw_name, state = (field.strip() for field in fields[:3])
    rows.append((job_name, raw_name, normalize_state(state)))

if array_spec in ("", "none"):
    matches = [state for job_name, raw_name, state in rows if job_id in (job_name, raw_name)]
    if len(matches) != 1:
        raise SystemExit(f"sacct did not return exactly one allocation row for job {job_id}")
    print(matches[0])
    raise SystemExit(0)

match = re.fullmatch(r"1-([1-9][0-9]*)(?:%[1-9][0-9]*)?", array_spec)
if match is None:
    raise SystemExit(f"Unsupported frozen V2 array specification: {array_spec}")
expected = set(range(1, int(match.group(1)) + 1))
observed = {}

def parse_indices(identifier):
    prefix = f"{job_id}_"
    if not identifier.startswith(prefix):
        return None
    suffix = identifier[len(prefix):]
    if re.fullmatch(r"[1-9][0-9]*", suffix):
        return [int(suffix)]
    bracket = re.fullmatch(r"\[([^]]+)\]", suffix)
    if bracket is None:
        return None
    expression = bracket.group(1)
    expression = expression.split("%", 1)[0]
    indices = []
    for component in expression.split(","):
        if re.fullmatch(r"[1-9][0-9]*", component):
            indices.append(int(component))
            continue
        interval = re.fullmatch(r"([1-9][0-9]*)-([1-9][0-9]*)", component)
        if interval is None:
            raise SystemExit(f"Unsupported compressed sacct array identifier: {identifier}")
        start, stop = map(int, interval.groups())
        if stop < start:
            raise SystemExit(f"Descending compressed sacct array identifier: {identifier}")
        indices.extend(range(start, stop + 1))
    return indices

for job_name, raw_name, state in rows:
    candidates = []
    for identifier in (job_name, raw_name):
        parsed = parse_indices(identifier)
        if parsed is not None:
            candidates.append(parsed)
    if not candidates:
        continue
    indices = candidates[0]
    if any(candidate != indices for candidate in candidates[1:]):
        raise SystemExit("JobID and JobIDRaw disagree on array task identity")
    for index in indices:
        if index not in expected:
            raise SystemExit(f"sacct returned an out-of-contract array index: {index}")
        if index in observed and observed[index] != state:
            raise SystemExit(f"sacct returned conflicting states for array index {index}")
        observed[index] = state

missing = expected.difference(observed)
if missing:
    preview = ",".join(map(str, sorted(missing)[:5]))
    raise SystemExit(f"sacct array coverage is incomplete; missing indices include {preview}")
states = list(observed.values())
active_states = [state for state in states if state in active]
if active_states:
    print("PENDING" if set(active_states) == {"PENDING"} else "RUNNING")
elif any(state in terminal_failure for state in states):
    print("FAILED")
elif all(state == "COMPLETED" for state in states):
    print("COMPLETED")
else:
    raise SystemExit("Could not aggregate complete Slurm array state")
PY
}

reference_cell_state_v2_audit_posthuman_stage() {
  local shadow_root="$1" action="$2"
  local receipt="$shadow_root/workflow_status/reference_cell_state_posthuman_v2/${action}_COMPLETE.tsv"
  python3 -I - "$receipt" "$shadow_root" "$action" <<'PY'
import hashlib
import os
import pathlib
import sys

receipt, root = map(pathlib.Path, sys.argv[1:3])
action = sys.argv[3]
if receipt.is_symlink() or not receipt.is_file():
    raise SystemExit("Post-human COMPLETE receipt is unavailable")
lines = receipt.read_text(encoding="utf-8").splitlines()
if not lines or lines[0] != "property\tvalue":
    raise SystemExit("Post-human receipt header is malformed")
rows = {}
for line in lines[1:]:
    key, sep, value = line.partition("\t")
    if not sep or not key or key in rows:
        raise SystemExit("Post-human receipt is malformed or duplicates a property")
    rows[key] = value
if (
    rows.get("schema_version") != "reference_cell_state_posthuman_stage_receipt_v2"
    or rows.get("status") != "COMPLETE"
    or rows.get("action") != action
):
    raise SystemExit("Post-human receipt action/status changed")
path_keys = sorted(key for key in rows if key.startswith("evidence_") and key.endswith("_path"))
if not path_keys:
    raise SystemExit("Post-human receipt has no evidence")
root_resolved = pathlib.Path(os.path.realpath(root))
for index, path_key in enumerate(path_keys, start=1):
    if path_key != f"evidence_{index:03d}_path":
        raise SystemExit("Post-human evidence indices are not contiguous")
    hash_key = f"evidence_{index:03d}_sha256"
    path = pathlib.Path(rows[path_key])
    resolved = pathlib.Path(os.path.realpath(path))
    if path.is_symlink() or not path.is_file() or (resolved != root_resolved and root_resolved not in resolved.parents):
        raise SystemExit(f"Post-human evidence is unavailable or escaped root: {path_key}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != rows.get(hash_key):
        raise SystemExit(f"Post-human evidence hash changed: {path_key}")
PY
}

reference_cell_state_v2_audit_model_acceptance() {
  local receipt="$1" sidecar="$2"
  [[ -s "$receipt" && -f "$receipt" && ! -L "$receipt" && \
     -s "$sidecar" && -f "$sidecar" && ! -L "$sidecar" ]] || return 2
  local observed recorded
  observed="$(reference_cell_state_v2_sha256 "$receipt")" || return 2
  recorded="$(tr -d '[:space:]' < "$sidecar")"
  [[ "$recorded" == "$observed" ]] || return 2
  python3 -I - "$receipt" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
if (
    payload.get("schema_version") != "reference_cell_state_model_acceptance_v2"
    or payload.get("status") != "ACCEPTED"
    or payload.get("accepted") is not True
):
    raise SystemExit("Model acceptance receipt is not V2 ACCEPTED")
PY
}

reference_cell_state_v2_audit_prediction_shards() {
  local prediction_root="$1" expected_count="$2"
  [[ -d "$prediction_root/shards" && ! -L "$prediction_root/shards" ]] || return 2
  python3 -I - "$prediction_root/shards" "$expected_count" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
receipts = sorted(root.glob("*/*/prediction_receipt.json"))
predictions = sorted(root.glob("*/*/reference_cell_state_predictions.tsv"))
if len(receipts) != expected or len(predictions) != expected:
    raise SystemExit("Completed prediction array does not have exact shard coverage")
for receipt in receipts:
    if receipt.is_symlink():
        raise SystemExit("Prediction receipt is a symlink")
    prediction = receipt.with_name("reference_cell_state_predictions.tsv")
    if prediction.is_symlink() or not prediction.is_file():
        raise SystemExit("Prediction TSV is absent from its atomic generation")
    with receipt.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != "reference_cell_state_shard_prediction_v2" or payload.get("status") != "COMPLETE":
        raise SystemExit("Prediction receipt is not COMPLETE V2")
    if pathlib.Path(payload.get("prediction_tsv", "")) != prediction:
        raise SystemExit("Prediction receipt path changed")
    if hashlib.sha256(prediction.read_bytes()).hexdigest() != payload.get("prediction_tsv_sha256"):
        raise SystemExit("Prediction TSV hash changed")
PY
}

reference_cell_state_v2_audit_final_predictions() {
  local shadow_root="$1"
  local receipt="$shadow_root/predictions/REFERENCE_CELL_STATE_SHADOW_GO_NO_GO.json"
  local predictions="$shadow_root/predictions/reference_cell_state_predictions.tsv"
  [[ -s "$receipt" && ! -L "$receipt" && -s "$predictions" && ! -L "$predictions" ]] || return 2
  python3 -I - "$receipt" "$shadow_root" <<'PY'
import csv, hashlib, itertools, json, pathlib, sys
receipt_path = pathlib.Path(sys.argv[1])
shadow_root = pathlib.Path(sys.argv[2])
if shadow_root.resolve() != shadow_root or receipt_path != shadow_root / "predictions" / "REFERENCE_CELL_STATE_SHADOW_GO_NO_GO.json":
    raise SystemExit("Final prediction audit root/receipt path changed")
with receipt_path.open(encoding="utf-8") as handle:
    payload = json.load(handle)
if (
    payload.get("schema_version") != "reference_cell_state_shard_merge_v2"
    or payload.get("technical_decision") != "GO"
    or payload.get("promotion_decision") != "NO_GO"
    or payload.get("overall_decision") != "SHADOW_ONLY"
):
    raise SystemExit("Final prediction receipt is not a V2 SHADOW_ONLY technical GO")
published = payload.get("published_shadow_axis")
prediction = receipt_path.with_name("reference_cell_state_predictions.tsv")
expected_columns = ["model_id", "cell_id", "reference_cell_state_class_id", "prediction_status"]
if (
    not isinstance(published, dict)
    or pathlib.Path(published.get("path", "")) != prediction
    or published.get("columns") != expected_columns
    or published.get("semantic_axis") != "reference_cell_state"
    or published.get("overwrites_viability_state") is not False
    or published.get("overwrites_trajectory_state") is not False
):
    raise SystemExit("Final prediction receipt path changed")
def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
def regular_canonical(path, label):
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise SystemExit(f"{label} is unavailable, symlinked, or noncanonical")
    return path
regular_canonical(prediction, "Final prediction TSV")
if hash_file(prediction) != published.get("sha256"):
    raise SystemExit("Final prediction TSV hash changed")
acceptance_identity = payload.get("model_acceptance")
if not isinstance(acceptance_identity, dict):
    raise SystemExit("Final prediction receipt lacks model acceptance")
acceptance = regular_canonical(pathlib.Path(acceptance_identity.get("receipt", "")), "Model acceptance")
expected_acceptance = shadow_root / "workflow_status" / "model_acceptance_v2" / "model_acceptance.json"
if acceptance != expected_acceptance or hash_file(acceptance) != acceptance_identity.get("sha256"):
    raise SystemExit("Final prediction model-acceptance identity changed")
sidecar = regular_canonical(acceptance.with_name("model_acceptance.sha256"), "Model acceptance SHA sidecar")
if sidecar.read_text(encoding="utf-8").strip() != acceptance_identity.get("sha256"):
    raise SystemExit("Model acceptance SHA sidecar changed")
with acceptance.open(encoding="utf-8") as handle:
    accepted = json.load(handle)
if (
    accepted.get("schema_version") != "reference_cell_state_model_acceptance_v2"
    or accepted.get("status") != "ACCEPTED"
    or accepted.get("accepted") is not True
    or pathlib.Path(accepted.get("shadow_root", "")) != shadow_root
):
    raise SystemExit("Final prediction model acceptance is not authoritative V2")
accepted_model = accepted.get("model")
if not isinstance(accepted_model, dict) or accepted_model.get("class_ids") != ["live_cell", "dead_cell", "multinucleated_cell"]:
    raise SystemExit("Accepted historical class contract changed")
model_id = accepted_model.get("model_id")
parent_root = pathlib.Path(payload.get("parent_shadow_root", ""))
if not parent_root.is_dir() or parent_root.is_symlink() or parent_root.resolve() != parent_root:
    raise SystemExit("Final prediction parent root changed")
feature_manifest = regular_canonical(pathlib.Path(payload.get("feature_manifest", "")), "Feature manifest")
cells = regular_canonical(pathlib.Path(payload.get("cells", "")), "Frozen cells table")
if (
    feature_manifest != parent_root / "workflow_status" / "feature_inventory" / "original_feature_manifest.tsv"
    or cells != parent_root / "cpa" / "cells.tsv"
    or hash_file(feature_manifest) != payload.get("feature_manifest_sha256")
    or hash_file(cells) != payload.get("cells_sha256")
):
    raise SystemExit("Final prediction frozen parent identity changed")
with feature_manifest.open(encoding="utf-8", newline="") as handle:
    manifest_reader = csv.reader(handle, delimiter="\t")
    try:
        next(manifest_reader)
    except StopIteration:
        raise SystemExit("Feature manifest is empty")
    shard_count = sum(1 for _ in manifest_reader)
if shard_count != payload.get("shard_count") or shard_count != 27200:
    raise SystemExit("Final prediction shard coverage changed")
cell_digest = hashlib.sha256()
first = True
row_count = 0
with prediction.open(encoding="utf-8", newline="") as prediction_handle, cells.open(encoding="utf-8", newline="") as cells_handle:
    prediction_reader = csv.DictReader(prediction_handle, delimiter="\t")
    cells_reader = csv.DictReader(cells_handle, delimiter="\t")
    if prediction_reader.fieldnames != expected_columns or "cell_id" not in (cells_reader.fieldnames or []):
        raise SystemExit("Final prediction/cells column contract changed")
    for row_count, pair in enumerate(itertools.zip_longest(prediction_reader, cells_reader), start=1):
        prediction_row, cell_row = pair
        if prediction_row is None or cell_row is None:
            raise SystemExit("Final prediction/cells universes differ in length")
        prediction_cell = (prediction_row.get("cell_id") or "").strip()
        cell_id = (cell_row.get("cell_id") or "").strip()
        if not cell_id or prediction_cell != cell_id or prediction_row.get("model_id") != model_id:
            raise SystemExit(f"Final prediction global cell/model coverage changed at row {row_count}")
        if not first:
            cell_digest.update(b"\n")
        cell_digest.update(cell_id.encode("utf-8"))
        first = False
if (
    row_count != payload.get("cell_count")
    or row_count != payload.get("ok_count", 0) + payload.get("unavailable_count", 0)
    or cell_digest.hexdigest() != payload.get("cells_cell_id_sha256")
):
    raise SystemExit("Final prediction exact global coverage audit changed")
PY
}

reference_cell_state_v2_audit_comparison() {
  local comparison_root="$1" receipt="$1/comparison_receipt.json"
  [[ -d "$comparison_root" && ! -L "$comparison_root" && -s "$receipt" && ! -L "$receipt" ]] || return 2
  python3 -I - "$receipt" <<'PY'
import json, sys
import hashlib, pathlib
receipt_path = pathlib.Path(sys.argv[1])
with receipt_path.open(encoding="utf-8") as handle:
    payload = json.load(handle)
if (
    payload.get("schema_version") != "current_vs_reference_cell_state_precomparison_v1"
    or payload.get("status") != "PRECOMPARISON_COMPLETE"
):
    raise SystemExit("Comparison receipt is not a complete V2 precomparison")
outputs = payload.get("outputs")
if not isinstance(outputs, dict) or len(outputs) != 9:
    raise SystemExit("Comparison output inventory changed")
for name, identity in outputs.items():
    path = receipt_path.with_name(name)
    if not isinstance(identity, dict) or pathlib.Path(identity.get("path", "")) != path:
        raise SystemExit("Comparison output path changed")
    if path.is_symlink() or not path.is_file():
        raise SystemExit("Comparison output is unavailable")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != identity.get("sha256") or len(content) != identity.get("size_bytes"):
        raise SystemExit("Comparison output identity changed")
inputs = payload.get("inputs")
if not isinstance(inputs, dict):
    raise SystemExit("Comparison input identity is absent")
root_identity = inputs.get("current_root")
current_identity = inputs.get("current")
reference_identity = inputs.get("reference_predictions")
if not all(isinstance(item, dict) for item in (root_identity, current_identity, reference_identity)):
    raise SystemExit("Comparison input identity is malformed")
current_root = pathlib.Path(root_identity.get("path", ""))
current_input = pathlib.Path(current_identity.get("path", ""))
reference_input = pathlib.Path(reference_identity.get("path", ""))
if (
    not current_root.is_dir() or current_root.is_symlink() or current_root.resolve() != current_root
    or not current_input.is_file() or current_input.is_symlink() or current_input.resolve() != current_input
    or current_root not in current_input.parents
    or not reference_input.is_file() or reference_input.is_symlink() or reference_input.resolve() != reference_input
):
    raise SystemExit("Comparison input paths changed or escaped")
if hashlib.sha256(current_input.read_bytes()).hexdigest() != current_identity.get("sha256"):
    raise SystemExit("Comparison current input hash changed")
if hashlib.sha256(reference_input.read_bytes()).hexdigest() != reference_identity.get("sha256"):
    raise SystemExit("Comparison reference input hash changed")
sources = root_identity.get("confined_source_files")
if not isinstance(sources, list) or not sources:
    raise SystemExit("Comparison confined source inventory is absent")
for source in sources:
    if not isinstance(source, dict):
        raise SystemExit("Comparison confined source identity is malformed")
    path = pathlib.Path(source.get("path", ""))
    if path.is_symlink() or not path.is_file() or path.resolve() != path or current_root not in path.parents:
        raise SystemExit("Comparison confined source path changed or escaped")
    if hashlib.sha256(path.read_bytes()).hexdigest() != source.get("sha256"):
        raise SystemExit("Comparison confined source hash changed")
binding_payload = {
    "current_input_path": str(current_input),
    "current_input_sha256": current_identity.get("sha256"),
    "current_root_path": str(current_root),
    "confined_source_files": sources,
}
binding = hashlib.sha256(json.dumps(binding_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
if binding != root_identity.get("identity_sha256") or binding != root_identity.get("confined_input_binding_sha256"):
    raise SystemExit("Comparison portable current-root binding changed")
PY
}

reference_cell_state_v2_reject_ambient_runtime_overrides() {
  local name
  for name in \
    HPC_CONTAINER_BINDS HPC_CONTAINER_RUNTIME_ROOT HPC_CONTAINER_FORWARD_PREFIXES \
    HPC_PROJECT_ROOT HPC_PROJECT_ROOT_SOURCE HPC_CONTAINER_NO_MOUNT \
    CPA_DEPENDENCY_LOCK CELL_PHENOTYPE_ANNOTATOR_ROOT \
    REFERENCE_CELL_STATE_PROJECT_BUILDER REFERENCE_MORPHOLOGY_WORKSPACE_SCRIPT \
    REFERENCE_CELL_STATE_HISTORICAL_PROJECTION_SCRIPT \
    REFERENCE_CELL_STATE_CPA_STAGE_SCRIPT \
    REFERENCE_CELL_STATE_MODEL_ACCEPT_SCRIPT \
    REFERENCE_CELL_STATE_SHARD_PREDICT_SCRIPT \
    REFERENCE_CELL_STATE_FINALIZE_SCRIPT; do
    [[ -z "${!name+x}" ]] || {
      reference_cell_state_v2_abort "Ambient override is forbidden for V2 submission: $name"
      return 2
    }
  done
}

reference_cell_state_v2_require_clean_worker_environment() {
  case "${HPC_CONTAINER_NO_MOUNT+x}:${HPC_CONTAINER_NO_MOUNT:-}" in
    :|x:/share) ;;
    *) reference_cell_state_v2_abort "HPC_CONTAINER_NO_MOUNT must be exactly /share for V2"; return 2 ;;
  esac
  HPC_CONTAINER_NO_MOUNT=/share
  case "${HPC_CONTAINER_FORWARD_PREFIXES+x}:${HPC_CONTAINER_FORWARD_PREFIXES:-}" in
    :) HPC_CONTAINER_FORWARD_PREFIXES="$REFERENCE_CELL_STATE_V2_FORWARD_PREFIXES" ;;
    "x:$REFERENCE_CELL_STATE_V2_FORWARD_PREFIXES") ;;
    *) reference_cell_state_v2_abort "HPC_CONTAINER_FORWARD_PREFIXES differs from the frozen V2 allowlist"; return 2 ;;
  esac
  export HPC_CONTAINER_NO_MOUNT HPC_CONTAINER_FORWARD_PREFIXES
}

reference_cell_state_v2_print_contract() {
  [[ -n "${REFERENCE_CELL_STATE_V2_DEFAULT_SIF:-}" && \
     -n "${REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256:-}" ]] || \
    reference_cell_state_v2_load_sif_identity || return 2
  printf 'reference_cell_state_hpc_contract=%s\n' "$REFERENCE_CELL_STATE_V2_SCHEMA"
  printf 'hpc_container_image=%s\n' "$REFERENCE_CELL_STATE_V2_DEFAULT_SIF"
  printf 'hpc_container_sha256=%s\n' "$REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256"
  printf 'hpc_container_bytes=%s\n' "$REFERENCE_CELL_STATE_V2_EXPECTED_SIF_BYTES"
  printf 'filesystem_immutability_mode=%s\n' "$REFERENCE_CELL_STATE_V2_FILESYSTEM_IMMUTABILITY_MODE"
  printf 'runtime_rootfs_read_only=%s\n' "$REFERENCE_CELL_STATE_V2_RUNTIME_ROOTFS_READ_ONLY"
  printf 'cpa_reference_root=%s\n' "$REFERENCE_CELL_STATE_V2_DEFAULT_CPA_ROOT"
  printf 'cpa_reference_commit=%s\n' "$REFERENCE_CELL_STATE_V2_EXPECTED_CPA_COMMIT"
  printf 'cpa_reference_tree=%s\n' "$REFERENCE_CELL_STATE_V2_EXPECTED_CPA_TREE"
  printf 'hpc_container_gpu=0\n'
  printf 'site_share_mount_disabled=1\n'
}
