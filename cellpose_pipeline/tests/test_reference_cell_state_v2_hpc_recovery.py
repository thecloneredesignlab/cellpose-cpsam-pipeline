from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCKER_HPC = ROOT / "cellpose_pipeline" / "Docker" / "hpc"


CONTRACT_STUB = r'''#!/usr/bin/env bash
REFERENCE_CELL_STATE_V2_FORWARD_PREFIXES=PYTHONNOUSERSITE
REFERENCE_CELL_STATE_V2_DEFAULT_CPA_ROOT="$CPA_REFERENCE_ROOT"
REFERENCE_CELL_STATE_V2_EXPECTED_CPA_COMMIT=fixture-commit
REFERENCE_CELL_STATE_V2_EXPECTED_CPA_TREE=fixture-tree
reference_cell_state_v2_abort(){ printf '%s\n' "$*" >&2; return 2; }
reference_cell_state_v2_load_sif_identity(){
  REFERENCE_CELL_STATE_V2_DEFAULT_SIF="$HPC_CONTAINER_IMAGE"
  REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256="$(sha256sum "$HPC_CONTAINER_IMAGE")"
  REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256="${REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256%%[[:space:]]*}"
  REFERENCE_CELL_STATE_V2_EXPECTED_SIF_BYTES="$(stat -c %s "$HPC_CONTAINER_IMAGE" 2>/dev/null || stat -f %z "$HPC_CONTAINER_IMAGE")"
  REFERENCE_CELL_STATE_V2_FILESYSTEM_IMMUTABILITY_MODE=shared_filesystem_mode_bits_unavailable
  REFERENCE_CELL_STATE_V2_RUNTIME_ROOTFS_READ_ONLY=verified
  export REFERENCE_CELL_STATE_V2_DEFAULT_SIF REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256
  export REFERENCE_CELL_STATE_V2_EXPECTED_SIF_BYTES REFERENCE_CELL_STATE_V2_FILESYSTEM_IMMUTABILITY_MODE
  export REFERENCE_CELL_STATE_V2_RUNTIME_ROOTFS_READ_ONLY
}
reference_cell_state_v2_require_runtime_identity(){ :; }
reference_cell_state_v2_verify_container_rootfs_read_only(){ :; }
reference_cell_state_v2_require_shadow_root(){ :; }
reference_cell_state_v2_require_clean_worker_environment(){
  HPC_CONTAINER_FORWARD_PREFIXES="$REFERENCE_CELL_STATE_V2_FORWARD_PREFIXES"
  export HPC_CONTAINER_FORWARD_PREFIXES
}
reference_cell_state_v2_require_inside(){
  local path="$1" root="$2"
  [[ "$path" == "$root" || "$path" == "$root"/* ]] || return 2
  [[ -e "$path" && ! -L "$path" ]] || return 2
}
reference_cell_state_v2_require_recoverable_output(){
  local path="$1" root="$2"
  [[ "$path" == "$root"/* && ! -L "$path" ]] || return 2
}
reference_cell_state_v2_sha256(){ sha256sum "$1" | awk '{print $1}'; }
reference_cell_state_v2_method_parity_status(){
  printf '%s\n' historical_core_parity_with_disclosed_no_stable_cluster_sampling_adaptation
}
reference_cell_state_v2_audit_phase_a(){
  if [[ "${V2_FIXTURE_FALLBACK:-0}" == 1 ]]; then
    printf 'one_cluster_fallback\ttrue\n'
    printf 'method_parity_status\thistorical_core_parity_with_disclosed_no_stable_cluster_sampling_adaptation\n'
  else
    printf 'one_cluster_fallback\tfalse\n'
    printf 'method_parity_status\thistorical_core_parity_with_pinned_stable_polygon_sampling\n'
  fi
}
reference_cell_state_v2_require_posthuman_action_branch(){
  local action="$1" fallback="$2"
  case "$fallback:$action" in
    false:post-region|false:post-seed1|false:post-seed2|false:post-adjudication|true:post-fallback-seed1|true:post-seed2|true:post-adjudication) return 0 ;;
    *) return 2 ;;
  esac
}
reference_cell_state_v2_test_maybe_fail(){
  local boundary="$1"
  [[ -z "${REFERENCE_CELL_STATE_V2_TEST_FAIL_AFTER_BOUNDARY:-}" ]] && return 0
  [[ "${REFERENCE_CELL_STATE_V2_TEST_MODE:-0}" == 1 ]] || return 2
  [[ "$REFERENCE_CELL_STATE_V2_TEST_FAIL_AFTER_BOUNDARY" != "$boundary" ]] || {
    printf 'reference_cell_state_v2_injected_failure_after=%s\n' "$boundary" >&2
    return 86
  }
}
'''


RUNTIME_STUB = r'''#!/usr/bin/env bash
hpc_container_prepare(){ :; }
hpc_apptainer_exec(){ "$V2_FIXTURE_PYTHON" "$V2_FIXTURE_DRIVER" "$@"; }
'''


IDENTITY_STUB = r'''#!/usr/bin/env bash
broad_phenotype_worker_verify_container_identity(){ :; }
'''


DRIVER = r'''#!/usr/bin/env python3
import json
import os
import pathlib
import sys

args = sys.argv[1:]
root = pathlib.Path(os.environ["REFERENCE_SHADOW_ROOT"])
log_path = pathlib.Path(os.environ["V2_FIXTURE_LOG"])
fallback = os.environ.get("V2_FIXTURE_FALLBACK", "0") == "1"

def value(flag):
    return args[args.index(flag) + 1]

def write(path, text):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SystemExit(f"fixture output drift: {path}")
        return False
    path.write_text(text, encoding="utf-8")
    return True

def publish(component, outputs):
    created = False
    for path, content in outputs:
        created = write(path, content) or created
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{component}:{'create' if created else 'verified_reuse'}\n")

program = args[0]
script = args[1] if program == "Rscript" else args[2]
name = pathlib.Path(script).name

if program == "Rscript" and script == "-":
    publish("runtime_packages", [(args[2], "package\tversion\nR\t4.2.3\n")])
elif program == "Rscript" and name.startswith("28_"):
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("28_preflight:verified_rerunnable\n")
elif program == "python" and name.startswith("26_"):
    project = root / "projection_input" / "representative_umap_v2"
    historical = project / "historical_projection"
    manifest = json.dumps({
        "schema_version": "reference_cell_state_historical_projection_v2",
        "status": "COMPLETE",
        "diagnostic_cluster": {"one_cluster_fallback": fallback},
    }, sort_keys=True) + "\n"
    publish("28_projection", [
        (historical / "historical_projection_manifest.json", manifest),
        (historical / "diagnostic_clusters.tsv", "cell_id\tcluster\nc1\t1\n"),
    ])
    publish("26_project", [
        (project / "project.yml", "project_id: reference_cell_state_development_v2\n"),
        (project / "parent_import_manifest.json", "{}\n"),
    ])
elif program == "python" and name.startswith("20_"):
    stage = value("--stage")
    if stage == "annotation-import":
        out = root / "annotation_import_generation"
        publish("cpa_annotation_import", [(out / "annotation_import_manifest.json", "{}\n")])
    else:
        outputs = []
        if stage == "annotate":
            out = root / "projection_input" / "representative_umap_v2" / "runs" / "fixture"
            outputs.append((out / "annotation_manifest.json", "{}\n"))
        if not outputs:
            out = root / "workflow_status" / "fixture_cpa"
            outputs.append((out / f"{stage}.receipt", f"{stage}\n"))
        publish(f"cpa_{stage}", outputs)
elif program == "python" and name.startswith("27_"):
    out = pathlib.Path(value("--output-dir"))
    publish("27_morphology", [
        (out / "annotation_workspace.html", "<html>fixture</html>\n"),
        (out / "overlay_manifest.json", "{}\n"),
    ])
elif program == "Rscript" and name.startswith("33_"):
    out = pathlib.Path(value("--output-dir"))
    manifest = {"schema_version": "reference_cell_state_seed1_review_v2"}
    if "--annotation-import-dir" in args:
        manifest["annotation_import_dir"] = value("--annotation-import-dir")
    publish("33_seed1", [
        (out / "seed1_review_set.tsv", "cell_id\nc1\n"),
        (out / "seed1_review_label_template.tsv", "cell_id\tlabel\nc1\t\n"),
        (out / "seed1_selection_audit.tsv", "cell_id\nc1\n"),
        (out / "seed1_review_manifest.json", json.dumps(manifest, sort_keys=True) + "\n"),
    ])
elif program == "Rscript" and name.startswith("34_"):
    out = pathlib.Path(value("--output-dir"))
    receipt = pathlib.Path(value("--output-receipt"))
    stage = value("--training-stage")
    publish(f"34_train_{stage}", [
        (out / "training_receipt.json", json.dumps({"stage": stage}) + "\n"),
        (receipt, json.dumps({"stage": stage}) + "\n"),
    ])
elif program == "Rscript" and name.startswith("35_"):
    out = pathlib.Path(value("--output-dir"))
    publish("35_seed2", [
        (out / "seed2_review_set.tsv", "cell_id\nc1\n"),
        (out / "seed2_review_label_template.tsv", "cell_id\tlabel\nc1\t\n"),
        (out / "seed2_selection_audit.tsv", "cell_id\nc1\n"),
        (out / "seed2_review_manifest.json", "{}\n"),
    ])
elif program == "Rscript" and name.startswith("36_"):
    out = pathlib.Path(value("--output-dir"))
    publish("36_merge", [
        (out / "merged_reviewed_labels.tsv", "cell_id\tlabel\nc1\tlive_cell\n"),
        (out / "review_merge_audit.tsv", "cell_id\nc1\n"),
        (out / "review_conflicts.tsv", "cell_id\n"),
        (out / "review_merge_manifest.json", "{}\n"),
    ])
elif program == "python" and name.startswith("37_"):
    out = pathlib.Path(value("--output-dir"))
    publish("37_render", [
        (out / "exact_review.html", "<html>fixture</html>\n"),
        (out / "exact_review_render_manifest.json", "{}\n"),
        (out / "crop_render_status.tsv", "cell_id\tstatus\nc1\tPASS\n"),
    ])
elif program == "Rscript" and name.startswith("38_"):
    out = pathlib.Path(value("--output-dir"))
    publish("38_import", [
        (out / "reviewed_labels.tsv", "cell_id\tlabel\nc1\tlive_cell\n"),
        (out / "review_import_audit.tsv", "cell_id\nc1\n"),
        (out / "review_import_manifest.json", "{}\n"),
    ])
elif program == "Rscript" and name.startswith("30_"):
    receipt = pathlib.Path(value("--output-receipt"))
    publish("30_accept", [(receipt, '{"schema_version":"model_acceptance_v2","status":"GO"}\n')])
elif program == "python" and script == "-":
    trailing = args[3:]
    if len(trailing) == 1 and trailing[0].endswith("historical_projection_manifest.json"):
        payload = json.loads(pathlib.Path(trailing[0]).read_text(encoding="utf-8"))
        print("true" if payload["diagnostic_cluster"]["one_cluster_fallback"] else "false")
    elif len(trailing) == 2 and trailing[0].endswith("seed1_review_manifest.json"):
        payload = json.loads(pathlib.Path(trailing[0]).read_text(encoding="utf-8"))
        print(payload["annotation_import_dir"])
    else:
        print("fixture_inline_check=PASS")
else:
    raise SystemExit(f"unhandled fixture command: {args}")
'''


class WorkerFixture:
    def __init__(self, temporary: str, worker_name: str, *, fallback: bool = False):
        self.base = Path(temporary)
        self.hpc = self.base / "hpc"
        self.project = self.base / "project"
        self.results = self.base / "results"
        self.root = self.results / "reference_cell_state_shadow_v2_fixture"
        self.dataset = self.base / "dataset"
        self.segmentation = self.base / "segmentation"
        self.parent = self.base / "parent"
        self.cpa = self.base / "cpa"
        self.log = self.base / "fixture.log"
        self.driver = self.base / "fixture_driver.py"
        (self.hpc / "util").mkdir(parents=True)
        self.project.mkdir()
        self.results.mkdir()
        self.root.mkdir()
        for directory in (
            self.dataset / "Brightfield",
            self.dataset / "Nuclei",
            self.segmentation / "Combined" / "segmentations",
            self.parent,
            self.cpa / "reference" / "ltee-source" / "code" / "lib",
        ):
            directory.mkdir(parents=True)
        (self.cpa / "reference" / "ltee-source" / "code" / "lib" / "Utils.R").write_text("# fixture\n")
        self.sif = self.base / "fixture.sif"
        self.sif.write_bytes(b"fixture-sif")
        self.identity = self.root / "identity.json"
        self.identity.write_text("{}\n")
        self.driver.write_text(DRIVER, encoding="utf-8")
        self.driver.chmod(0o755)
        (self.hpc / "util" / "reference_cell_state_v2_contract.sh").write_text(CONTRACT_STUB)
        (self.hpc / "util" / "hpc_container_apptainer_runtime.sh").write_text(RUNTIME_STUB)
        (self.hpc / "util" / "broad_phenotype_container_identity.sh").write_text(IDENTITY_STUB)
        self.worker = self.hpc / worker_name
        shutil.copy2(DOCKER_HPC / worker_name, self.worker)
        self.worker.chmod(0o755)
        self._build_project_tree()
        self.env = os.environ.copy()
        self.env.update(
            {
                "REFERENCE_SHADOW_ROOT": str(self.root),
                "RESULTS_ROOT": str(self.results),
                "PARENT_BROAD_SHADOW_ROOT": str(self.parent),
                "DATASET_ROOT": str(self.dataset),
                "SOURCE_SEGMENTATION_ROOT": str(self.segmentation),
                "HPC_CONTAINER_IMAGE": str(self.sif),
                "CPA_REFERENCE_ROOT": str(self.cpa),
                "HPC_CONTAINER_IDENTITY_FILE": str(self.identity),
                "HPC_CONTAINER_IDENTITY_FILE_SHA256": hashlib.sha256(self.identity.read_bytes()).hexdigest(),
                "PROJECT_DIR": str(self.project),
                "V2_FIXTURE_DRIVER": str(self.driver),
                "V2_FIXTURE_PYTHON": sys.executable,
                "V2_FIXTURE_LOG": str(self.log),
                "V2_FIXTURE_FALLBACK": "1" if fallback else "0",
                "REFERENCE_CELL_STATE_V2_TEST_MODE": "1",
                "SLURM_CPUS_PER_TASK": "2",
            }
        )

    def _build_project_tree(self) -> None:
        configs = self.project / "cellpose_pipeline" / "configs"
        scripts = self.project / "cellpose_pipeline" / "scripts"
        resources = scripts / "analysisi" / "resources"
        configs.mkdir(parents=True)
        resources.mkdir(parents=True)
        for name in (
            "cellphenotypeannotator_dependency.lock.tsv",
            "reference_cell_state_features_v2.json",
            "reference_cell_state_classes_v2.tsv",
        ):
            (configs / name).write_text("fixture\n")
        (resources / "SUM159_AC_Experiment1_PlateMap.csv").write_text("fixture\n")
        for number, name in (
            (20, "run_cellphenotypeannotator_stage.py"),
            (26, "prepare_reference_cell_state_project.py"),
            (27, "build_reference_morphology_workspace.py"),
            (28, "build_reference_cell_state_historical_projection.R"),
            (30, "accept_reference_cell_state_model.R"),
            (33, "build_reference_cell_state_seed1_review.R"),
            (34, "train_reference_cell_state_historical.R"),
            (35, "build_reference_cell_state_seed2_review.R"),
            (36, "merge_reference_cell_state_reviews.R"),
            (37, "render_reference_cell_state_exact_review.py"),
            (38, "import_reference_cell_state_exact_review.R"),
        ):
            (scripts / f"{number}_{name}").write_text("# fixture\n")

    def run(self, boundary: str | None = None, **extra: str) -> subprocess.CompletedProcess[str]:
        env = self.env.copy()
        env.update(extra)
        if boundary is None:
            env.pop("REFERENCE_CELL_STATE_V2_TEST_FAIL_AFTER_BOUNDARY", None)
        else:
            env["REFERENCE_CELL_STATE_V2_TEST_FAIL_AFTER_BOUNDARY"] = boundary
        return subprocess.run(
            ["bash", str(self.worker)],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def log_lines(self) -> list[str]:
        return self.log.read_text(encoding="utf-8").splitlines()

    def seed_review_inputs(self, *, include_seed2: bool) -> None:
        annotation = self.root / "annotation_import_generation"
        annotation.mkdir(parents=True)
        (annotation / "annotation_import_manifest.json").write_text("{}\n")
        seeds = (1, 2) if include_seed2 else (1,)
        for seed in seeds:
            selection = self.root / "human_review" / f"seed{seed}" / "selection"
            render = self.root / "human_review" / f"seed{seed}" / "render"
            selection.mkdir(parents=True)
            render.mkdir(parents=True)
            (selection / f"seed{seed}_review_set.tsv").write_text("cell_id\nc1\n")
            manifest = {"schema_version": f"reference_cell_state_seed{seed}_review_v2"}
            if seed == 1:
                manifest["annotation_import_dir"] = str(annotation)
            (selection / f"seed{seed}_review_manifest.json").write_text(json.dumps(manifest) + "\n")
            (render / "exact_review.html").write_text("<html>fixture</html>\n")
            (render / "exact_review_render_manifest.json").write_text("{}\n")
            (render / "crop_render_status.tsv").write_text("cell_id\tstatus\nc1\tPASS\n")
            (render / "exact_review_submission.json").write_text("{}\n")


class ReferenceCellStateV2WorkerRecoveryTests(unittest.TestCase):
    def test_phase_a_recovers_each_component_boundary_on_same_root(self) -> None:
        cases = (
            ("phase_a_after_historical_preflight", False, "28_preflight:verified_rerunnable"),
            ("phase_a_after_project_generation", False, "28_projection:verified_reuse"),
            ("phase_a_after_cpa", False, "cpa_annotate:verified_reuse"),
            ("phase_a_after_morphology", False, "27_morphology:verified_reuse"),
            ("phase_a_after_seed1_selection", True, "33_seed1:verified_reuse"),
            ("phase_a_after_seed1_render", True, "37_render:verified_reuse"),
        )
        for boundary, fallback, expected_reuse in cases:
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as temporary:
                fixture = WorkerFixture(temporary, "run_reference_cell_state_phase_a_v2.sh", fallback=fallback)
                parent_project = fixture.parent / "projection_input" / "representative_umap" / "project.yml"
                parent_projection = fixture.parent / "projection_input" / "projection_input_manifest.json"
                parent_umap = fixture.parent / "projection_input" / "representative_umap" / "runs" / "fixture" / "umap_manifest.json"
                for path in (parent_project, parent_projection, parent_umap):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("fixture\n")
                extra = {
                    "PARENT_PROJECT": str(parent_project),
                    "PARENT_PROJECTION_INPUT_MANIFEST": str(parent_projection),
                    "PARENT_UMAP_MANIFEST": str(parent_umap),
                    "EXPECTED_PARENT_CELL_COUNT": "1",
                }
                failed = fixture.run(boundary, **extra)
                self.assertEqual(failed.returncode, 86, failed.stderr)
                resumed = fixture.run(None, **extra)
                self.assertEqual(resumed.returncode, 0, resumed.stderr)
                self.assertIn(expected_reuse, fixture.log_lines())
                receipt = fixture.root / "workflow_status" / "reference_cell_state_phase_a_v2" / "PHASE_A_COMPLETE.tsv"
                self.assertTrue(receipt.is_file())
                self.assertIn("status\tCOMPLETE", receipt.read_text(encoding="utf-8"))

    def test_posthuman_recovers_import_train_select_merge_boundaries(self) -> None:
        cases = (
            ("post-seed1", "posthuman_after_seed1_import", "38_import:verified_reuse"),
            ("post-seed1", "posthuman_after_initial_train", "34_train_initial:verified_reuse"),
            ("post-seed1", "posthuman_after_seed2_selection", "35_seed2:verified_reuse"),
            ("post-seed2", "posthuman_after_seed2_import", "38_import:verified_reuse"),
            ("post-seed2", "posthuman_after_review_merge", "36_merge:verified_reuse"),
            ("post-seed2", "posthuman_after_final_train", "34_train_final:verified_reuse"),
        )
        for action, boundary, expected_reuse in cases:
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as temporary:
                fixture = WorkerFixture(temporary, "run_reference_cell_state_cpa_stage_v2.sh")
                project_dir = fixture.root / "projection_input" / "representative_umap_v2"
                project_dir.mkdir(parents=True)
                (project_dir / "project.yml").write_text("project_id: fixture\n")
                (project_dir / "parent_import_manifest.json").write_text("{}\n")
                fixture.seed_review_inputs(include_seed2=action == "post-seed2")
                extra = {"V2_ACTION": action}
                failed = fixture.run(boundary, **extra)
                self.assertEqual(failed.returncode, 86, failed.stderr)
                resumed = fixture.run(None, **extra)
                self.assertEqual(resumed.returncode, 0, resumed.stderr)
                self.assertIn(expected_reuse, fixture.log_lines())
                receipt = fixture.root / "workflow_status" / "reference_cell_state_posthuman_v2" / f"{action}_COMPLETE.tsv"
                self.assertTrue(receipt.is_file())
                self.assertIn("status\tCOMPLETE", receipt.read_text(encoding="utf-8"))

    def test_model_acceptance_recovers_receipt_before_sha_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = WorkerFixture(temporary, "run_reference_cell_state_model_acceptance_v2.sh")
            project_dir = fixture.root / "projection_input" / "representative_umap_v2"
            project_dir.mkdir(parents=True)
            project = project_dir / "project.yml"
            parent = project_dir / "parent_import_manifest.json"
            project.write_text("project_id: fixture\n")
            parent.write_text("{}\n")
            model = fixture.root / "models" / "final"
            model.mkdir(parents=True)
            (model / "model.rds").write_text("fixture\n")
            train = fixture.root / "workflow_status" / "training_v2" / "final_train_receipt.json"
            train.parent.mkdir(parents=True)
            train.write_text("{}\n")
            extra = {"CPA_MODEL_DIR": str(model), "CPA_TRAIN_RECEIPT": str(train)}
            failed = fixture.run("model_acceptance_after_receipt", **extra)
            self.assertEqual(failed.returncode, 86, failed.stderr)
            receipt = fixture.root / "workflow_status" / "model_acceptance_v2" / "model_acceptance.json"
            sidecar = fixture.root / "workflow_status" / "model_acceptance_v2" / "model_acceptance.sha256"
            self.assertTrue(receipt.is_file())
            self.assertFalse(sidecar.exists())
            resumed = fixture.run(None, **extra)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertIn("30_accept:verified_reuse", fixture.log_lines())
            self.assertEqual(sidecar.read_text(encoding="utf-8").strip(), hashlib.sha256(receipt.read_bytes()).hexdigest())

    def test_posthuman_rejects_action_branch_mismatches_before_work(self) -> None:
        for fallback, action in ((True, "post-region"), (True, "post-seed1"), (False, "post-fallback-seed1")):
            with self.subTest(fallback=fallback, action=action), tempfile.TemporaryDirectory() as temporary:
                fixture = WorkerFixture(
                    temporary, "run_reference_cell_state_cpa_stage_v2.sh", fallback=fallback
                )
                failed = fixture.run(None, V2_ACTION=action)
                self.assertEqual(failed.returncode, 2, failed.stderr)
                self.assertFalse(fixture.log.exists(), "mismatched action reached a computational component")


class PhaseAReceiptAuditTests(unittest.TestCase):
    contract = DOCKER_HPC / "util" / "reference_cell_state_v2_contract.sh"

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _build_receipt(self, temporary: str, *, fallback: bool) -> tuple[Path, Path, dict[str, Path]]:
        base = Path(temporary)
        root = base / "reference_cell_state_shadow_v2_audit"
        root.mkdir()
        artifacts: dict[str, Path] = {}

        def create(key: str, relative: str, content: str) -> Path:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            artifacts[key] = path
            return path

        project = create("project_file", "projection_input/representative_umap_v2/project.yml", "project_id: fixture\n")
        historical = create(
            "historical_projection_manifest",
            "projection_input/representative_umap_v2/historical_projection/historical_projection_manifest.json",
            json.dumps(
                {
                    "schema_version": "reference_cell_state_historical_projection_v2",
                    "status": "COMPLETE",
                    "diagnostic_cluster": {"one_cluster_fallback": fallback},
                },
                sort_keys=True,
            )
            + "\n",
        )
        runtime = create("runtime_package_receipt", "workflow_status/reference_v2_runtime_packages.tsv", "package\tversion\nR\t4.2.3\n")
        annotation = create(
            "annotation_manifest",
            "projection_input/representative_umap_v2/runs/fixture/annotation_manifest.json",
            "{}\n",
        )
        workspace = create("morphology_workspace", "morphology_reference/annotation_workspace.html", "<html>fixture</html>\n")
        overlay_output = root / "morphology_reference" / "umap_morphology_overlay.png"
        overlay_output.write_bytes(b"fixture-overlay")
        overlay = create(
            "morphology_overlay_manifest",
            "morphology_reference/overlay_manifest.json",
            json.dumps(
                {
                    "schema_version": "reference_morphology_workspace_v2",
                    "status": "COMPLETE",
                    "output_artifact_sha256": {
                        "annotation_workspace.html": self._sha(workspace),
                        "umap_morphology_overlay.png": self._sha(overlay_output),
                    },
                },
                sort_keys=True,
            )
            + "\n",
        )
        plate = base / "plate_map.csv"
        dependency = base / "dependency.lock.tsv"
        plate.write_text("well\nA01\n", encoding="utf-8")
        dependency.write_text("package\tversion\nfixture\t1\n", encoding="utf-8")
        artifacts["plate_map"] = plate
        artifacts["dependency_lock"] = dependency

        rows = {
            "schema_version": "reference_cell_state_phase_a_receipt_v2",
            "status": "COMPLETE",
            "reference_shadow_root": str(root),
            "project_file": str(project),
            "project_sha256": self._sha(project),
            "plate_map": str(plate),
            "plate_map_sha256": self._sha(plate),
            "historical_projection_manifest": str(historical),
            "historical_projection_manifest_sha256": self._sha(historical),
            "runtime_package_receipt": str(runtime),
            "runtime_package_receipt_sha256": self._sha(runtime),
            "annotation_dir": str(annotation.parent),
            "annotation_manifest": str(annotation),
            "annotation_manifest_sha256": self._sha(annotation),
            "morphology_overlay_manifest": str(overlay),
            "morphology_overlay_manifest_sha256": self._sha(overlay),
            "morphology_workspace": str(workspace),
            "morphology_workspace_sha256": self._sha(workspace),
            "dependency_lock": str(dependency),
            "dependency_lock_sha256": self._sha(dependency),
            "hpc_container_image": str(base / "fixture.sif"),
            "hpc_container_sha256": "a" * 64,
            "one_cluster_fallback": "true" if fallback else "false",
        }
        if fallback:
            review_set = create("seed1_review_set", "human_review/seed1/selection/seed1_review_set.tsv", "cell_id\nc1\n")
            label_template = create("seed1_label_template", "human_review/seed1/selection/seed1_review_label_template.tsv", "cell_id\tlabel\nc1\t\n")
            selection_audit = create("seed1_selection_audit", "human_review/seed1/selection/seed1_selection_audit.tsv", "cell_id\nc1\n")
            selection = create(
                "seed1_selection_manifest",
                "human_review/seed1/selection/seed1_review_manifest.json",
                json.dumps(
                    {
                        "schema_version": "reference_cell_state_seed1_review_v2",
                        "status": "HUMAN_REVIEW_REQUIRED",
                        "output_file_sha256": {
                            "review_set": self._sha(review_set),
                            "label_template": self._sha(label_template),
                            "selection_audit": self._sha(selection_audit),
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
            )
            html = create("seed1_exact_review_html", "human_review/seed1/render/exact_review.html", "<html>fixture</html>\n")
            crops = create("seed1_crop_status", "human_review/seed1/render/crop_render_status.tsv", "cell_id\tstatus\nc1\tPASS\n")
            crop_file = root / "human_review" / "seed1" / "render" / "crops" / "c1__brightfield.png"
            crop_file.parent.mkdir(parents=True)
            crop_file.write_bytes(b"fixture-crop")
            crop_manifest = create(
                "seed1_crop_manifest",
                "human_review/seed1/render/crop_manifest.tsv",
                "morphology_umap_row_key\tchannel\trelative_path\tsha256\n"
                f"c1\tbrightfield\tcrops/{crop_file.name}\t{self._sha(crop_file)}\n",
            )
            render_manifest = create(
                "seed1_render_manifest",
                "human_review/seed1/render/exact_review_render_manifest.json",
                json.dumps(
                    {
                        "schema_version": "reference_cell_state_exact_review_render_v2",
                        "status": "HUMAN_REVIEW_REQUIRED",
                        "artifact_file_sha256": {
                            "exact_review_html": self._sha(html),
                            "crop_render_status": self._sha(crops),
                            "crop_manifest": self._sha(crop_manifest),
                        },
                        "crop_aggregate_sha256": self._sha(crop_manifest),
                    },
                    sort_keys=True,
                )
                + "\n",
            )
            for key, path in (
                ("seed1_selection_manifest", selection),
                ("seed1_review_set", review_set),
                ("seed1_render_manifest", render_manifest),
                ("seed1_exact_review_html", html),
                ("seed1_crop_status", crops),
            ):
                rows[key] = str(path)
                rows[f"{key}_sha256"] = self._sha(path)
            rows.update(
                {
                    "method_parity_status": "historical_core_parity_with_disclosed_no_stable_cluster_sampling_adaptation",
                    "review_sampling_contract": "disclosed_seed1_all_unassigned_well_balanced_max500_and_seed2_probability_surrogate_not_pinned_reference_exact_sampling",
                    "human_barrier": "seed1_adapted_exact_review_submission_required",
                    "human_workspace": str(html),
                    "human_submission_expected_path": str(html.parent / "exact_review_submission.json"),
                }
            )
        else:
            rows.update(
                {
                    "method_parity_status": "historical_core_parity_with_pinned_stable_polygon_sampling",
                    "review_sampling_contract": "pinned_reference_stable_polygon_seed1_seed2_sampling",
                    "human_barrier": "region_submission_required",
                    "human_workspace": str(workspace),
                    "human_submission_expected_path": str(annotation.parent / "region_submission.json"),
                }
            )
        receipt = root / "workflow_status" / "reference_cell_state_phase_a_v2" / "PHASE_A_COMPLETE.tsv"
        receipt.parent.mkdir(parents=True)
        receipt.write_text(
            "property\tvalue\n" + "".join(f"{key}\t{value}\n" for key, value in rows.items()),
            encoding="utf-8",
        )
        return root, receipt, artifacts

    def _audit(self, root: Path) -> subprocess.CompletedProcess[str]:
        command = (
            f'source "{self.contract}"; '
            f'REFERENCE_CELL_STATE_V2_DEFAULT_SIF="{root.parent / "fixture.sif"}"; '
            'REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256="' + "a" * 64 + '"; '
            f'reference_cell_state_v2_audit_phase_a "{root}"'
        )
        return subprocess.run(["bash", "-c", command], text=True, capture_output=True, check=False)

    def test_phase_a_receipt_rehashes_every_branch_evidence_and_rejects_tamper(self) -> None:
        keys_by_branch = {
            False: ("historical_projection_manifest", "annotation_manifest", "morphology_overlay_manifest", "morphology_workspace"),
            True: (
                "historical_projection_manifest",
                "annotation_manifest",
                "morphology_overlay_manifest",
                "morphology_workspace",
                "seed1_selection_manifest",
                "seed1_review_set",
                "seed1_render_manifest",
                "seed1_exact_review_html",
                "seed1_crop_status",
            ),
        }
        for fallback, keys in keys_by_branch.items():
            for key in keys:
                with self.subTest(fallback=fallback, artifact=key), tempfile.TemporaryDirectory() as temporary:
                    root, _, artifacts = self._build_receipt(temporary, fallback=fallback)
                    clean = self._audit(root)
                    self.assertEqual(clean.returncode, 0, clean.stderr)
                    with artifacts[key].open("a", encoding="utf-8") as handle:
                        handle.write("tamper\n")
                    rejected = self._audit(root)
                    self.assertNotEqual(rejected.returncode, 0)
                    self.assertIn("hash changed", rejected.stderr)

    def test_phase_a_receipt_rejects_historical_branch_and_parity_mismatch(self) -> None:
        for key, old, new in (
            ("one_cluster_fallback", "false", "true"),
            (
                "method_parity_status",
                "historical_core_parity_with_pinned_stable_polygon_sampling",
                "historical_core_parity_with_disclosed_no_stable_cluster_sampling_adaptation",
            ),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root, receipt, _ = self._build_receipt(temporary, fallback=False)
                text = receipt.read_text(encoding="utf-8")
                self.assertIn(f"{key}\t{old}\n", text)
                receipt.write_text(text.replace(f"{key}\t{old}\n", f"{key}\t{new}\n"), encoding="utf-8")
                rejected = self._audit(root)
                self.assertNotEqual(rejected.returncode, 0)


class FinalPredictionAuditTests(unittest.TestCase):
    contract = DOCKER_HPC / "util" / "reference_cell_state_v2_contract.sh"

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _build_generation(self, temporary: str) -> tuple[Path, dict[str, Path]]:
        base = Path(temporary).resolve()
        root = base / "reference_cell_state_shadow_v2_final_audit"
        parent = base / "broad_phenotype_shadow_fixture"
        predictions_dir = root / "predictions"
        acceptance_dir = root / "workflow_status" / "model_acceptance_v2"
        feature_dir = parent / "workflow_status" / "feature_inventory"
        cells_dir = parent / "cpa"
        for directory in (predictions_dir, acceptance_dir, feature_dir, cells_dir):
            directory.mkdir(parents=True)

        feature_manifest = feature_dir / "original_feature_manifest.tsv"
        feature_manifest.write_text(
            "key\n" + "".join(f"A01_{index}\n" for index in range(27200)),
            encoding="utf-8",
        )
        cell_ids = ["original|A01_1_0d0h0m|1", "original|A01_1_0d0h0m|2"]
        cells = cells_dir / "cells.tsv"
        cells.write_text("cell_id\n" + "\n".join(cell_ids) + "\n", encoding="utf-8")
        predictions = predictions_dir / "reference_cell_state_predictions.tsv"
        predictions.write_text(
            "model_id\tcell_id\treference_cell_state_class_id\tprediction_status\n"
            + "".join(
                f"fixture-model\t{cell_id}\tlive_cell\tok\n" for cell_id in cell_ids
            ),
            encoding="utf-8",
        )
        acceptance = acceptance_dir / "model_acceptance.json"
        acceptance.write_text(
            json.dumps(
                {
                    "schema_version": "reference_cell_state_model_acceptance_v2",
                    "status": "ACCEPTED",
                    "accepted": True,
                    "shadow_root": str(root),
                    "model": {
                        "model_id": "fixture-model",
                        "class_ids": ["live_cell", "dead_cell", "multinucleated_cell"],
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        sidecar = acceptance_dir / "model_acceptance.sha256"
        sidecar.write_text(self._sha(acceptance) + "\n", encoding="utf-8")
        receipt = predictions_dir / "REFERENCE_CELL_STATE_SHADOW_GO_NO_GO.json"
        receipt.write_text(
            json.dumps(
                {
                    "schema_version": "reference_cell_state_shard_merge_v2",
                    "technical_decision": "GO",
                    "promotion_decision": "NO_GO",
                    "overall_decision": "SHADOW_ONLY",
                    "feature_manifest": str(feature_manifest),
                    "feature_manifest_sha256": self._sha(feature_manifest),
                    "model_acceptance": {
                        "receipt": str(acceptance),
                        "sha256": self._sha(acceptance),
                    },
                    "parent_shadow_root": str(parent),
                    "shadow_root": str(root),
                    "cells": str(cells),
                    "cells_sha256": self._sha(cells),
                    "cells_cell_id_sha256": hashlib.sha256(
                        "\n".join(cell_ids).encode("utf-8")
                    ).hexdigest(),
                    "shard_count": 27200,
                    "cell_count": len(cell_ids),
                    "ok_count": len(cell_ids),
                    "unavailable_count": 0,
                    "published_shadow_axis": {
                        "path": str(predictions),
                        "sha256": self._sha(predictions),
                        "columns": [
                            "model_id",
                            "cell_id",
                            "reference_cell_state_class_id",
                            "prediction_status",
                        ],
                        "semantic_axis": "reference_cell_state",
                        "overwrites_viability_state": False,
                        "overwrites_trajectory_state": False,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return root, {
            "receipt": receipt,
            "predictions": predictions,
            "acceptance": acceptance,
            "sidecar": sidecar,
            "cells": cells,
        }

    def _audit(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "-c",
                f'source "{self.contract}"; reference_cell_state_v2_audit_final_predictions "{root}"',
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_final_prediction_audit_requires_receipt_acceptance_and_exact_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, artifacts = self._build_generation(temporary)
            valid = self._audit(root)
            self.assertEqual(valid.returncode, 0, valid.stderr)

            artifacts["receipt"].unlink()
            missing = self._audit(root)
            self.assertNotEqual(missing.returncode, 0)

        for artifact, replacement in (
            ("sidecar", "0" * 64 + "\n"),
            ("predictions", "model_id\tcell_id\nfixture-model\tbroken\n"),
            ("cells", "cell_id\noriginal|wrong|1\n"),
        ):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as temporary:
                root, artifacts = self._build_generation(temporary)
                artifacts[artifact].write_text(replacement, encoding="utf-8")
                rejected = self._audit(root)
                self.assertNotEqual(rejected.returncode, 0)


class CalibrationLauncherRecoveryTests(unittest.TestCase):
    def test_existing_partial_and_complete_calibration_roots_are_resumed(self) -> None:
        launcher_source = DOCKER_HPC / "Parameter_calibration" / "31_run_reference_cell_state_shadow_v2_test.sh"
        for complete in (False, True):
            with self.subTest(complete=complete), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                experiment = base / "experiment"
                results = experiment / "results"
                tests_root = results / "Tests_and_Parameters_calibration"
                parent = tests_root / "broad_phenotype_shadow_test_20260812_073844"
                shadow = tests_root / "reference_cell_state_shadow_v2_test_resume"
                for path in (parent, shadow):
                    path.mkdir(parents=True)
                hpc = base / "hpc"
                calibration = hpc / "Parameter_calibration"
                calibration.mkdir(parents=True)
                launcher = calibration / launcher_source.name
                text = launcher_source.read_text(encoding="utf-8").replace(
                    "/share/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/"
                    "N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/"
                    "20260619_SUM159_Doxorubicin_Cyclophosphamide",
                    str(experiment),
                )
                launcher.write_text(text, encoding="utf-8")
                launcher.chmod(0o755)
                log = base / "submitter.log"
                submitter = hpc / "submit_reference_cell_state_shadow_v2.sh"
                submitter.write_text(
                    "#!/bin/bash\nset -euo pipefail\n"
                    f'printf "%s\\n" "$REFERENCE_SHADOW_ROOT" >> "{log}"\n'
                    'printf "fixture_submitter=PASS\\n"\n',
                    encoding="utf-8",
                )
                submitter.chmod(0o755)
                stub = base / "bin"
                stub.mkdir()
                (stub / "hostname").write_text("#!/bin/bash\nprintf 'hpctpa3pc0009\\n'\n", encoding="utf-8")
                (stub / "hostname").chmod(0o755)
                if complete:
                    receipt = shadow / "workflow_status" / "reference_cell_state_phase_a_v2" / "PHASE_A_COMPLETE.tsv"
                    receipt.parent.mkdir(parents=True)
                    receipt.write_text("property\tvalue\nstatus\tCOMPLETE\n", encoding="utf-8")
                environment = os.environ.copy()
                environment.update(
                    {
                        "PATH": f"{stub}:{environment['PATH']}",
                        "RESULTS_ROOT": str(results),
                        "PARENT_BROAD_SHADOW_ROOT": str(parent),
                        "REFERENCE_SHADOW_ROOT": str(shadow),
                        "V2_STAGE": "phase-a",
                    }
                )
                resumed = subprocess.run(
                    ["bash", str(launcher)], env=environment, text=True, capture_output=True, check=False
                )
                self.assertEqual(resumed.returncode, 0, resumed.stderr)
                self.assertIn("test_root_disposition=resume_existing_through_frozen_archive", resumed.stdout)
                self.assertEqual(log.read_text(encoding="utf-8").strip(), str(shadow))


if __name__ == "__main__":
    unittest.main()
