#!/usr/bin/env Rscript

# Build the historical seed-2 targeted multinucleated review (four buckets,
# 200 total) from the initial historical model. Predictions drive sampling only.

SCHEMA_VERSION <- "reference_cell_state_seed2_multinucleated_review_v2"
Sys.setenv(GIT_OPTIONAL_LOCKS = "0")
script_argument <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_argument) != 1L) stop("Unable to locate adapter source file", call. = FALSE)
script_path <- normalizePath(sub("^--file=", "", script_argument[[1L]]), mustWork = TRUE)
shared_script_path <- normalizePath(
  file.path(dirname(script_path), "_shared", "reference_cell_state_v2.R"),
  mustWork = TRUE
)
source(shared_script_path, local = .GlobalEnv)

parse_args <- function(arguments) {
  allowed <- c(
    "--reference-root", "--dependency-lock", "--project",
    "--annotation-import-dir", "--diagnostic-cluster-manifest", "--initial-model-dir",
    "--seed1-reviewed-labels", "--output-dir"
  )
  out <- list(); i <- 1L
  while (i <= length(arguments)) {
    arg <- arguments[[i]]
    if (!arg %in% allowed || i == length(arguments)) reference_cell_state_v2_abort("Unknown argument or missing value: %s", arg)
    name <- gsub("-", "_", substring(arg, 3L), fixed = TRUE)
    if (!is.null(out[[name]])) reference_cell_state_v2_abort("Repeated argument: %s", arg)
    out[[name]] <- arguments[[i + 1L]]; i <- i + 2L
  }
  required <- c(
    "reference_root", "dependency_lock", "project", "initial_model_dir",
    "seed1_reviewed_labels", "output_dir"
  )
  missing <- required[!vapply(required, function(name) !is.null(out[[name]]) && nzchar(out[[name]]), logical(1))]
  if (length(missing)) reference_cell_state_v2_abort("Missing arguments: %s", paste(missing, collapse = ", "))
  branches <- c(!is.null(out$annotation_import_dir), !is.null(out$diagnostic_cluster_manifest))
  if (sum(branches) != 1L) {
    reference_cell_state_v2_abort(
      "Supply exactly one of --annotation-import-dir or --diagnostic-cluster-manifest"
    )
  }
  out
}

norm <- function(path, label, directory = FALSE, output = FALSE) {
  if (!grepl("^/", path)) reference_cell_state_v2_abort("%s must be absolute", label)
  path <- if (output) reference_cell_state_v2_normalize_output(path, label) else {
    normalizePath(path, winslash = "/", mustWork = TRUE)
  }
  if (!output && directory && !dir.exists(path)) reference_cell_state_v2_abort("%s is not a directory", label)
  if (!output && !directory && (!file.exists(path) || dir.exists(path))) reference_cell_state_v2_abort("%s is not a file", label)
  path
}

inside <- function(path, root) identical(path, root) || startsWith(path, paste0(root, "/"))

source_review <- function(root) {
  lib <- file.path(root, "reference", "ltee-source", "code", "lib")
  files <- c(
    segmentation_qc_review = file.path(lib, "segmentation_qc_review.R"),
    morphology_cell_state_review = file.path(lib, "morphology_cell_state_review.R"),
    morphology_cell_state_multinucleated_review = file.path(lib, "morphology_cell_state_multinucleated_review.R")
  )
  before <- reference_cell_state_v2_run_git(root, "status", "--porcelain", "--untracked-files=all")
  for (path in files) source(path, local = .GlobalEnv)
  required <- c("generate_morphology_cell_state_multinucleated_review_set", "morphology_cell_state_review_label_template")
  missing <- required[!vapply(required, exists, logical(1), mode = "function")]
  if (length(missing)) reference_cell_state_v2_abort("Historical seed2 API is incomplete")
  after <- reference_cell_state_v2_run_git(root, "status", "--porcelain", "--untracked-files=all")
  if (!identical(before, after) || nzchar(after)) {
    reference_cell_state_v2_abort("Loading historical Seed2 sources changed the pinned checkout")
  }
  list(
    loading_policy = "pinned_function_only_review_sources_into_clean_process_global_environment",
    source_file_sha256 = as.list(vapply(files, reference_cell_state_v2_sha256_file, character(1)))
  )
}

verify_annotation_import <- function(annotation_dir, labels_path) {
  manifest_path <- norm(file.path(annotation_dir, "annotation_import_manifest.json"), "annotation import manifest")
  manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
  hashes <- unlist(manifest$artifact_file_sha256, use.names = TRUE)
  if (!identical(as.character(manifest$schema_version), "cell_phenotype_annotator_annotation_import_v1") ||
      !setequal(names(hashes), c("region_submission", "regions", "provisional_labels")) ||
      !identical(reference_cell_state_v2_sha256_file(labels_path), as.character(hashes[["provisional_labels"]]))) {
    reference_cell_state_v2_abort("Seed2 annotation import identity or provisional-label hash changed")
  }
  list(path = manifest_path, sha256 = reference_cell_state_v2_sha256_file(manifest_path))
}

verify_diagnostic_fallback <- function(manifest_path, project_dir, cells) {
  expected_path <- norm(
    file.path(project_dir, "historical_projection", "historical_projection_manifest.json"),
    "canonical historical projection manifest"
  )
  if (!identical(manifest_path, expected_path)) {
    reference_cell_state_v2_abort(
      "Fallback manifest must be the canonical project historical projection manifest"
    )
  }
  manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
  diagnostic <- manifest$diagnostic_cluster
  if (!identical(as.character(manifest$schema_version), "reference_cell_state_historical_projection_v2") ||
      !identical(as.character(manifest$status), "COMPLETE") ||
      !isTRUE(diagnostic$one_cluster_fallback) ||
      !identical(as.character(diagnostic$role), "metadata_only")) {
    reference_cell_state_v2_abort(
      "Seed2 fallback requires an authoritative one_cluster_fallback historical manifest"
    )
  }
  clusters_path <- norm(
    file.path(project_dir, "historical_projection", "diagnostic_clusters.tsv"),
    "diagnostic clusters"
  )
  declared <- as.character(manifest$output_file_sha256[["diagnostic_clusters.tsv"]])
  if (!identical(reference_cell_state_v2_sha256_file(clusters_path), declared)) {
    reference_cell_state_v2_abort("Diagnostic clusters differ from the fallback manifest")
  }
  clusters <- reference_cell_state_v2_read_table(clusters_path, "diagnostic clusters")
  reference_cell_state_v2_require_columns(
    clusters, c("cell_id", "cluster", "cluster_source"), "diagnostic clusters"
  )
  if (anyDuplicated(clusters$cell_id) || !setequal(clusters$cell_id, cells$cell_id) ||
      any(as.character(clusters$cluster) != "1") ||
      any(as.character(clusters$cluster_source) != "optimize_dbscan_v4_one_cluster_fallback") ||
      any(as.character(cells$cluster) != "1")) {
    reference_cell_state_v2_abort(
      "Canonical cells and diagnostic artifact do not prove one-cluster fallback"
    )
  }
  list(
    labels = data.frame(
      cell_id = as.character(cells$cell_id), assignment_state = "unreviewed",
      class_id = "", region_id = "", stringsAsFactors = FALSE
    ),
    manifest = manifest, manifest_path = manifest_path, clusters_path = clusters_path,
    manifest_sha256 = reference_cell_state_v2_sha256_file(manifest_path),
    clusters_sha256 = reference_cell_state_v2_sha256_file(clusters_path)
  )
}

verify_seed1_import <- function(seed1_path, expected_diagnostic_manifest = NULL) {
  manifest_path <- norm(
    file.path(dirname(seed1_path), "review_import_manifest.json"),
    "Seed1 review import manifest"
  )
  manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
  import_implementation <- norm(
    as.character(manifest$implementation), "Seed1 review import implementation"
  )
  import_shared_implementation <- norm(
    as.character(manifest$shared_implementation), "Seed1 review import shared implementation"
  )
  expected_import_implementation <- norm(
    file.path(dirname(script_path), "38_import_reference_cell_state_exact_review.R"),
    "expected Seed1 review import implementation"
  )
  if (!identical(as.character(manifest$schema_version), "reference_cell_state_exact_review_import_v2") ||
      !identical(as.character(manifest$status), "COMPLETE") ||
      !isTRUE(manifest$exact_selection_preserved) || isTRUE(manifest$secondary_sampling_performed) ||
      !identical(import_implementation, expected_import_implementation) ||
      !identical(as.character(manifest$implementation_sha256),
                 reference_cell_state_v2_sha256_file(import_implementation)) ||
      !identical(import_shared_implementation, shared_script_path) ||
      !identical(as.character(manifest$shared_implementation_sha256),
                 reference_cell_state_v2_sha256_file(shared_script_path)) ||
      !identical(as.character(manifest$reviewed_labels_sha256),
                 reference_cell_state_v2_sha256_file(seed1_path))) {
    reference_cell_state_v2_abort(
      "Seed1 reviewed labels differ from their authoritative exact-review import"
    )
  }
  render_path <- norm(as.character(manifest$render_manifest), "Seed1 render manifest")
  if (!identical(reference_cell_state_v2_sha256_file(render_path),
                 as.character(manifest$render_manifest_sha256))) {
    reference_cell_state_v2_abort("Seed1 render manifest changed after authoritative import")
  }
  render <- jsonlite::fromJSON(render_path, simplifyVector = FALSE)
  selection_item <- render$inputs$review_manifest
  selection_path <- norm(as.character(selection_item$path), "Seed1 selection manifest")
  if (!identical(reference_cell_state_v2_sha256_file(selection_path),
                 as.character(selection_item$sha256))) {
    reference_cell_state_v2_abort("Seed1 selection manifest changed after rendering")
  }
  selection <- jsonlite::fromJSON(selection_path, simplifyVector = FALSE)
  if (!identical(as.character(selection$schema_version), "reference_cell_state_seed1_review_v2") ||
      !identical(as.character(selection$status), "HUMAN_REVIEW_REQUIRED")) {
    reference_cell_state_v2_abort("Seed1 selection manifest is not the frozen V2 review generation")
  }
  if (!is.null(expected_diagnostic_manifest)) {
    declared_path <- norm(as.character(selection$diagnostic_cluster_manifest),
                          "Seed1 diagnostic cluster manifest")
    if (!isTRUE(selection$all_unassigned_fallback) ||
        !isTRUE(selection$one_cluster_fallback_proven) ||
        !identical(as.character(selection$selection_input_mode),
                   "authoritative_one_cluster_fallback") ||
        !identical(declared_path, expected_diagnostic_manifest) ||
        !identical(as.character(selection$diagnostic_cluster_manifest_sha256),
                   reference_cell_state_v2_sha256_file(expected_diagnostic_manifest))) {
      reference_cell_state_v2_abort(
        "Seed1 review does not prove ancestry from this one-cluster fallback"
      )
    }
  }
  list(
    manifest = manifest, manifest_path = manifest_path,
    manifest_sha256 = reference_cell_state_v2_sha256_file(manifest_path),
    render_manifest_path = render_path,
    render_manifest_sha256 = reference_cell_state_v2_sha256_file(render_path),
    selection_manifest_path = selection_path,
    selection_manifest_sha256 = reference_cell_state_v2_sha256_file(selection_path),
    selection = selection
  )
}

build_fallback_sampling_surrogate <- function(pseudo, predictions, reviewed_labels) {
  prediction_index <- match(
    as.character(pseudo$morphology_umap_row_key),
    as.character(predictions$morphology_umap_row_key)
  )
  if (anyNA(prediction_index)) {
    reference_cell_state_v2_abort("Fallback initial predictions do not cover canonical cells")
  }
  excluded <- as.character(reviewed_labels$morphology_umap_row_key)
  pseudo$suggested_cell_state_label <- ""
  audit <- list()
  weights <- morphology_cell_state_multinucleated_bucket_weights()
  contexts <- sort(unique(as.character(pseudo$context_key)), method = "radix")
  if (length(contexts) != 2L) {
    reference_cell_state_v2_abort("Fallback seed2 requires exactly two plate-ploidy contexts")
  }
  for (context in contexts) {
    candidates <- which(
      as.character(pseudo$context_key) == context &
        !as.character(pseudo$morphology_umap_row_key) %in% excluded
    )
    predicted_label <- as.character(predictions$predicted_label[prediction_index[candidates]])
    probability <- suppressWarnings(as.numeric(
      predictions$probability_multinucleated_cell[prediction_index[candidates]]
    ))
    if (any(!is.finite(probability))) {
      reference_cell_state_v2_abort("Fallback seed2 has non-finite multinucleated probabilities")
    }
    quota <- morphology_cell_state_allocate_quota(100L, weights)
    positive_required <- unname(quota[["model_positive_precision"]])
    pseudo_miss_required <- unname(
      quota[["pseudo_multi_high_score_miss"]] + quota[["pseudo_multi_diverse_miss"]]
    )
    control_required <- unname(quota[["hard_nonmulti_boundary_control"]])
    positive <- candidates[predicted_label == "multinucleated_cell"]
    nonpositive <- candidates[predicted_label != "multinucleated_cell"]
    if (length(positive) < positive_required ||
        length(nonpositive) < pseudo_miss_required + control_required) {
      reference_cell_state_v2_abort(
        paste0(
          "Fallback seed2 cannot populate the historical four buckets in ", context,
          ": model_positive=%d/%d nonpositive=%d/%d"
        ),
        length(positive), positive_required, length(nonpositive),
        pseudo_miss_required + control_required
      )
    }
    nonpositive_probability <- suppressWarnings(as.numeric(
      predictions$probability_multinucleated_cell[prediction_index[nonpositive]]
    ))
    ordered <- order(
      -nonpositive_probability,
      as.character(pseudo$morphology_umap_row_key[nonpositive]),
      method = "radix"
    )
    surrogate <- nonpositive[ordered[seq_len(pseudo_miss_required)]]
    pseudo$suggested_cell_state_label[surrogate] <- "multinucleated_cell"
    audit[[length(audit) + 1L]] <- data.frame(
      context_key = context,
      available_model_positive = length(positive),
      required_model_positive = positive_required,
      available_model_nonpositive = length(nonpositive),
      surrogate_pseudo_multi_count = length(surrogate),
      required_hard_nonmulti_control = control_required,
      surrogate_rule = "top_initial_uncalibrated_p_multi_among_model_nonpositive_sampling_only",
      stringsAsFactors = FALSE
    )
  }
  list(rows = pseudo, audit = do.call(rbind, audit))
}

verify_initial_model <- function(shadow, project_path, model_dir, seed1_path, lock, api) {
  receipt_path <- norm(
    file.path(shadow, "workflow_status", "training_v2", "initial_train_receipt.json"),
    "initial training receipt"
  )
  receipt <- jsonlite::fromJSON(receipt_path, simplifyVector = FALSE)
  authoritative_receipt <- norm(file.path(model_dir, "training_receipt.json"),
                                "authoritative initial training receipt")
  model_path <- norm(file.path(model_dir, "morphology_cell_state_classifier_uncalibrated_model.rds"), "initial model")
  prediction_path <- norm(file.path(model_dir, "morphology_cell_state_classifier_uncalibrated_full_predictions.tsv"), "initial predictions")
  hashes <- unlist(receipt$output_file_sha256, use.names = TRUE)
  training_implementation <- norm(
    as.character(receipt$implementation), "initial training implementation"
  )
  training_shared_implementation <- norm(
    as.character(receipt$shared_implementation), "initial training shared implementation"
  )
  expected_training_implementation <- norm(
    file.path(dirname(script_path), "34_train_reference_cell_state_historical.R"),
    "expected initial training implementation"
  )
  if (!identical(as.character(receipt$schema_version), "reference_cell_state_historical_train_v2") ||
      !identical(as.character(receipt$status), "COMPLETE") ||
      !identical(as.character(receipt$training_stage), "initial") ||
      !identical(reference_cell_state_v2_sha256_file(authoritative_receipt),
                 reference_cell_state_v2_sha256_file(receipt_path)) ||
      !identical(norm(as.character(receipt$model_dir), "receipt model_dir", directory = TRUE), model_dir) ||
      !identical(norm(as.character(receipt$project), "receipt project"), project_path) ||
      !identical(norm(as.character(receipt$reviewed_labels), "receipt Seed1 labels"), seed1_path) ||
      !identical(as.character(receipt$reviewed_labels_sha256), reference_cell_state_v2_sha256_file(seed1_path)) ||
      !identical(as.character(receipt$dependency$lock_sha256), reference_cell_state_v2_sha256_file(lock)) ||
      !identical(training_implementation, expected_training_implementation) ||
      !identical(as.character(receipt$implementation_sha256),
                 reference_cell_state_v2_sha256_file(training_implementation)) ||
      !identical(training_shared_implementation, shared_script_path) ||
      !identical(as.character(receipt$shared_implementation_sha256),
                 reference_cell_state_v2_sha256_file(shared_script_path)) ||
      !identical(as.character(receipt$model_rds_sha256), reference_cell_state_v2_sha256_file(model_path)) ||
      !identical(as.character(hashes[["model_rds"]]), reference_cell_state_v2_sha256_file(model_path)) ||
      !identical(as.character(hashes[["full_predictions"]]), reference_cell_state_v2_sha256_file(prediction_path))) {
    reference_cell_state_v2_abort("Initial model/prediction generation differs from its training receipt")
  }
  bundle <- readRDS(model_path)
  api$validate_morphology_cell_state_classifier_model_bundle(
    bundle, expected_feature_profile = "promoted_shape_plus_rfs_boundary"
  )
  if (!identical(as.character(bundle$model_sha256), as.character(receipt$model_sha256)) ||
      !identical(as.character(bundle$classifier$classes), reference_cell_state_v2_class_ids())) {
    reference_cell_state_v2_abort("Initial model bundle differs from the frozen historical receipt")
  }
  list(
    receipt = receipt, receipt_path = receipt_path,
    receipt_sha256 = reference_cell_state_v2_sha256_file(receipt_path),
    model_path = model_path, prediction_path = prediction_path, bundle = bundle
  )
}

annotation_rows <- function(cells, labels, plate_contract) {
  cells <- reference_cell_state_v2_validate_canonical_cells(cells, plate_contract)
  reference_cell_state_v2_require_columns(cells, "cluster", "cells historical diagnostic clusters")
  reference_cell_state_v2_require_columns(labels, c("cell_id", "assignment_state", "class_id", "region_id"), "provisional labels")
  if (anyDuplicated(cells$cell_id) || anyDuplicated(labels$cell_id) || !setequal(cells$cell_id, labels$cell_id)) {
    reference_cell_state_v2_abort("Cell/annotation stable-ID universes differ")
  }
  labels <- labels[match(cells$cell_id, labels$cell_id), , drop = FALSE]
  rows <- data.frame(
    morphology_umap_row_key = cells$cell_id,
    stable_cell_id = cells$cell_id,
    segmentation_qc_cell_id = cells$cell_id,
    segmentation_object_id = cells$cell_id,
    context_key = cells$context_key, ltee_cell_line = cells$ltee_cell_line,
    cell_line_family = cells$ltee_cell_line_family,
    well = cells$well, source_id = cells$source_id, suffix = cells$suffix,
    condition = cells$condition, condition_code = cells$condition,
    feature_row_index = as.integer(cells$feature_row_index), mask_label = as.integer(cells$mask_label),
    passage_number = 0L,
    cluster = as.character(cells$cluster),
    suggested_cell_state_label = ifelse(labels$assignment_state == "class_assigned", labels$class_id, ""),
    suggested_region_id = labels$region_id,
    stringsAsFactors = FALSE, check.names = FALSE
  )
  if (any(!grepl("^-?[0-9]+$", rows$cluster))) {
    reference_cell_state_v2_abort("cells historical diagnostic cluster must be an integer for every row")
  }
  rows
}

verify_existing_seed2 <- function(output, expected_manifest, candidate_paths) {
  manifest_path <- norm(file.path(output, "seed2_review_manifest.json"),
                        "existing Seed2 manifest")
  observed <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
  observed_hashes <- unlist(observed$output_file_sha256, use.names = TRUE)
  files <- c(
    review_set = "seed2_review_set.tsv",
    label_template = "seed2_review_label_template.tsv",
    selection_audit = "seed2_selection_audit.tsv",
    fallback_surrogate_audit = "seed2_fallback_surrogate_audit.tsv"
  )
  if (!setequal(names(observed_hashes), names(files))) {
    reference_cell_state_v2_abort("Existing Seed2 output hash set changed")
  }
  for (role in names(files)) {
    path <- norm(file.path(output, files[[role]]), paste("existing Seed2", role))
    if (!identical(reference_cell_state_v2_sha256_file(path),
                   as.character(observed_hashes[[role]]))) {
      reference_cell_state_v2_abort("Existing Seed2 artifact changed: %s", role)
    }
  }
  observed$output_file_sha256 <- NULL
  expected_manifest$output_file_sha256 <- NULL
  if (!identical(
    jsonlite::toJSON(observed, auto_unbox = TRUE, null = "null", digits = NA),
    jsonlite::toJSON(expected_manifest, auto_unbox = TRUE, null = "null", digits = NA)
  )) {
    reference_cell_state_v2_abort("Existing Seed2 generation identity differs")
  }
  compare_tables <- function(left_path, right_path) {
    left <- utils::read.delim(left_path, sep = "\t", header = TRUE, quote = "",
                              comment.char = "", check.names = FALSE,
                              stringsAsFactors = FALSE)
    right <- utils::read.delim(right_path, sep = "\t", header = TRUE, quote = "",
                               comment.char = "", check.names = FALSE,
                               stringsAsFactors = FALSE)
    columns <- setdiff(names(right), "review_selected_at")
    identical(names(left), names(right)) && nrow(left) == nrow(right) &&
      all(vapply(columns, function(column) {
        identical(as.character(left[[column]]), as.character(right[[column]]))
      }, logical(1)))
  }
  for (role in names(files)) {
    if (!compare_tables(file.path(output, files[[role]]), candidate_paths[[role]])) {
      reference_cell_state_v2_abort("Existing Seed2 deterministic artifact differs: %s", role)
    }
  }
  invisible(TRUE)
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  root <- norm(args$reference_root, "--reference-root", directory = TRUE)
  lock <- norm(args$dependency_lock, "--dependency-lock")
  project_path <- norm(args$project, "--project")
  annotation_dir <- if (is.null(args$annotation_import_dir)) NULL else {
    norm(args$annotation_import_dir, "--annotation-import-dir", directory = TRUE)
  }
  diagnostic_manifest <- if (is.null(args$diagnostic_cluster_manifest)) NULL else {
    norm(args$diagnostic_cluster_manifest, "--diagnostic-cluster-manifest")
  }
  model_dir <- norm(args$initial_model_dir, "--initial-model-dir", directory = TRUE)
  seed1_path <- norm(args$seed1_reviewed_labels, "--seed1-reviewed-labels")
  output <- norm(args$output_dir, "--output-dir", output = TRUE)
  project_dir <- dirname(project_path); shadow <- dirname(dirname(project_dir))
  if (!identical(basename(dirname(project_dir)), "projection_input")) {
    reference_cell_state_v2_abort(
      "Project must be under REFERENCE_SHADOW_ROOT/projection_input/<project>/project.yml"
    )
  }
  scoped <- c(
    if (!is.null(annotation_dir)) annotation_dir else diagnostic_manifest,
    model_dir, seed1_path, output
  )
  if (any(!vapply(scoped, inside, logical(1), root = shadow)) || inside(output, root)) {
    reference_cell_state_v2_abort("Seed2 inputs/outputs must remain inside one V2 shadow root")
  }
  dependency <- reference_cell_state_v2_verify_dependency(root, lock)
  review_source_identity <- source_review(root)
  loaded <- reference_cell_state_v2_load_historical_classifier(root)
  api <- loaded$api
  project <- reference_cell_state_v2_read_project(project_path)
  plate_contract <- reference_cell_state_v2_resolve_plate_contract(project_path)
  cells_path <- norm(file.path(project_dir, project$cells_file), "project cells")
  cells <- reference_cell_state_v2_read_table(cells_path, "cells")
  cells <- reference_cell_state_v2_validate_canonical_cells(cells, plate_contract)
  fallback <- NULL
  if (!is.null(annotation_dir)) {
    labels_path <- norm(file.path(annotation_dir, "provisional_labels.tsv"), "provisional labels")
    annotation_identity <- verify_annotation_import(annotation_dir, labels_path)
    labels <- reference_cell_state_v2_read_table(labels_path, "provisional labels")
  } else {
    fallback <- verify_diagnostic_fallback(diagnostic_manifest, project_dir, cells)
    labels_path <- fallback$clusters_path
    annotation_identity <- list(
      path = fallback$manifest_path, sha256 = fallback$manifest_sha256
    )
    labels <- fallback$labels
  }
  pseudo <- annotation_rows(cells, labels, plate_contract)
  seed1 <- reference_cell_state_v2_read_table(seed1_path, "seed1 reviewed labels")
  seed1_identity <- verify_seed1_import(
    seed1_path,
    expected_diagnostic_manifest = if (is.null(fallback)) NULL else diagnostic_manifest
  )
  seed1_import_manifest_path <- seed1_identity$manifest_path
  seed1_import_manifest <- seed1_identity$manifest
  prepared <- api$prepare_morphology_cell_state_manual_labels(seed1)
  if (any(!prepared$labels$manual_review_evidence)) {
    reference_cell_state_v2_abort("Seed1 input contains cells without human review evidence")
  }
  initial <- verify_initial_model(shadow, project_path, model_dir, seed1_path, lock, api)
  model_path <- initial$model_path
  bundle <- initial$bundle
  prediction_path <- initial$prediction_path
  predictions <- reference_cell_state_v2_read_table(prediction_path, "initial full predictions")
  if (!setequal(predictions$morphology_umap_row_key, pseudo$morphology_umap_row_key)) {
    reference_cell_state_v2_abort("Initial predictions do not cover annotation universe exactly")
  }
  fallback_surrogate <- NULL
  if (!is.null(fallback)) {
    fallback_surrogate <- build_fallback_sampling_surrogate(pseudo, predictions, seed1)
    pseudo <- fallback_surrogate$rows
  }
  selected <- generate_morphology_cell_state_multinucleated_review_set(
    pseudo_labels = pseudo, predictions = predictions, reviewed_labels = seed1,
    n_total = 200L, seed = 2L,
    annotation_input_sha256 = reference_cell_state_v2_sha256_file(labels_path)
  )
  review <- selected$review
  if (nrow(review) != 200L || length(unique(review$context_key)) != 2L ||
      anyDuplicated(review$morphology_umap_row_key) ||
      any(review$morphology_umap_row_key %in% seed1$morphology_umap_row_key)) {
    reference_cell_state_v2_abort("Historical seed2 selection violates 200-row/two-context/nonoverlap contract")
  }
  template <- morphology_cell_state_review_label_template(review)
  template$reviewer <- ""; template$reviewed_at <- ""; template$label_confidence <- "high"
  if (any(api$morphology_cell_state_manual_review_evidence(template))) {
    reference_cell_state_v2_abort("Seed2 template unexpectedly contains manual evidence")
  }
  dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
  staging <- tempfile(paste0(".", basename(output), "-staging-"), tmpdir = dirname(output))
  if (!dir.create(staging)) reference_cell_state_v2_abort("Could not create Seed2 staging directory")
  on.exit(unlink(staging, recursive = TRUE, force = TRUE), add = TRUE)
  paths <- c(
    review_set = file.path(staging, "seed2_review_set.tsv"),
    label_template = file.path(staging, "seed2_review_label_template.tsv"),
    selection_audit = file.path(staging, "seed2_selection_audit.tsv"),
    fallback_surrogate_audit = file.path(staging, "seed2_fallback_surrogate_audit.tsv")
  )
  reference_cell_state_v2_write_tsv_atomic(review, paths[["review_set"]])
  reference_cell_state_v2_write_tsv_atomic(template, paths[["label_template"]])
  reference_cell_state_v2_write_tsv_atomic(selected$selection_audit, paths[["selection_audit"]])
  surrogate_audit <- if (is.null(fallback_surrogate)) {
    data.frame(
      context_key = character(), available_model_positive = integer(),
      required_model_positive = integer(), available_model_nonpositive = integer(),
      surrogate_pseudo_multi_count = integer(), required_hard_nonmulti_control = integer(),
      surrogate_rule = character(), stringsAsFactors = FALSE
    )
  } else fallback_surrogate$audit
  reference_cell_state_v2_write_tsv_atomic(
    surrogate_audit, paths[["fallback_surrogate_audit"]]
  )
  manifest <- list(
    schema_version = SCHEMA_VERSION, status = "HUMAN_REVIEW_REQUIRED",
    seed = 2L, n_total = 200L,
    bucket_weights = as.list(morphology_cell_state_multinucleated_bucket_weights()),
    contexts = as.list(sort(unique(review$context_key))),
    source_group_column = "source_id", source_id_semantics = "well",
    suffix_role = "site_plus_elapsed_nested",
    initial_model_path = model_path,
    initial_model_rds_sha256 = reference_cell_state_v2_sha256_file(model_path),
    initial_model_sha256 = bundle$model_sha256,
    initial_train_receipt = initial$receipt_path,
    initial_train_receipt_sha256 = initial$receipt_sha256,
    seed1_reviewed_labels = seed1_path,
    seed1_reviewed_labels_sha256 = reference_cell_state_v2_sha256_file(seed1_path),
    seed1_review_import_manifest = seed1_import_manifest_path,
    seed1_review_import_manifest_sha256 = reference_cell_state_v2_sha256_file(seed1_import_manifest_path),
    seed1_render_manifest = seed1_identity$render_manifest_path,
    seed1_render_manifest_sha256 = seed1_identity$render_manifest_sha256,
    seed1_selection_manifest = seed1_identity$selection_manifest_path,
    seed1_selection_manifest_sha256 = seed1_identity$selection_manifest_sha256,
    historical_parity_scope = if (is.null(fallback)) {
      "pinned_reference_native_seed2_sampling"
    } else "historical_core_parity_with_sampling_adaptation",
    selection_input_mode = if (is.null(fallback)) {
      "accepted_polygon_annotation"
    } else "disclosed_no_stable_cluster_adapter_with_sampling_only_probability_surrogate",
    annotation_import_manifest = if (is.null(fallback)) annotation_identity$path else NULL,
    annotation_import_manifest_sha256 = if (is.null(fallback)) annotation_identity$sha256 else NULL,
    diagnostic_cluster_manifest = if (is.null(fallback)) NULL else fallback$manifest_path,
    diagnostic_cluster_manifest_sha256 = if (is.null(fallback)) NULL else fallback$manifest_sha256,
    diagnostic_clusters_sha256 = if (is.null(fallback)) NULL else fallback$clusters_sha256,
    one_cluster_fallback_proven = !is.null(fallback),
    fallback_sampling_adapter = if (is.null(fallback)) NULL else list(
      name = "disclosed_no_stable_cluster_adapter",
      reason = "historical_four_bucket_helper_requires_pseudo_multi_misses_but_one_cluster_projection_is_all_unassigned",
      rule = "within_context_top_initial_uncalibrated_p_multi_among_model_nonpositive",
      role = "sampling_only_never_training",
      helper_called_unchanged = "generate_morphology_cell_state_multinucleated_review_set",
      reference_native_sampling_claimed = FALSE,
      historical_production_reproduction_claimed = FALSE,
      input_prediction_sha256 = reference_cell_state_v2_sha256_file(prediction_path),
      input_diagnostic_manifest_sha256 = fallback$manifest_sha256,
      input_seed1_reviewed_labels_sha256 = reference_cell_state_v2_sha256_file(seed1_path),
      audit_sha256 = reference_cell_state_v2_sha256_file(paths[["fallback_surrogate_audit"]])
    ),
    pseudo_label_role = "sampling_only_never_training",
    diagnostic_cluster_role = "metadata_only_never_training",
    row_count = nrow(template),
    stable_id_sha256 = reference_cell_state_v2_sha256_text(
      sort(as.character(template$morphology_umap_row_key), method = "radix")
    ),
    output_file_sha256 = as.list(vapply(paths, reference_cell_state_v2_sha256_file, character(1))),
    dependency = dependency, historical_source_identity = loaded$identity,
    historical_review_source_identity = review_source_identity,
    implementation = script_path,
    implementation_sha256 = reference_cell_state_v2_sha256_file(script_path),
    shared_implementation = shared_script_path,
    shared_implementation_sha256 = reference_cell_state_v2_sha256_file(shared_script_path)
  )
  if (dir.exists(output)) {
    verify_existing_seed2(output, manifest, paths)
    writeLines(sprintf("reference_cell_state_seed2_review_verified_reuse=1 rows=200 output=%s", output))
    return(invisible(0L))
  }
  if (file.exists(output) || reference_cell_state_v2_is_symlink(output)) {
    reference_cell_state_v2_abort("Seed2 output exists but is not a regular generation directory")
  }
  reference_cell_state_v2_write_json_atomic(manifest, file.path(staging, "seed2_review_manifest.json"))
  if (file.exists(output) || dir.exists(output) || reference_cell_state_v2_is_symlink(output)) {
    reference_cell_state_v2_abort("Seed2 output appeared during staging")
  }
  if (!file.rename(staging, output)) reference_cell_state_v2_abort("Failed to install Seed2 generation atomically")
  writeLines(sprintf("reference_cell_state_seed2_review_ready=1 rows=200 output=%s", output))
}

status <- tryCatch(main(), error = function(error) { writeLines(conditionMessage(error), stderr()); 1L })
quit(save = "no", status = as.integer(status), runLast = FALSE)
