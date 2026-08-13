#!/usr/bin/env Rscript

# Apply one immutable Cell Phenotype Annotator classifier generation to one
# reference-cell-state feature shard.  The reference checkout is sourced into a
# private environment and is never modified or copied into this repository.

SCHEMA_VERSION <- "reference_cell_state_shard_prediction_v2"
FEATURE_RECEIPT_SCHEMA <- "broad_phenotype_feature_receipt_v1"
MODEL_ACCEPTANCE_SCHEMA <- "reference_cell_state_model_acceptance_v2"
REFERENCE_FEATURES <- c(
  "area_px2", "perimeter_px", "roundness", "aspect_ratio", "extent",
  "solidity", "equivalent_diameter_px", "major_axis_px", "minor_axis_px",
  "bf_boundary_mean", "bf_interior_mean", "bf_interior_minus_boundary_mean"
)
HISTORICAL_FEATURES <- c(
  "Area.\u00b5m.2", "perimeter.\u00b5m", "roundness", "aspect_ratio", "extent",
  "solidity", "equi_diameter", "Major_Axis", "Minor_Axis",
  "candidate_boundary_mean", "candidate_interior_mean",
  "candidate_interior_minus_boundary_mean"
)
REFERENCE_CLASS_IDS <- c("live_cell", "dead_cell", "multinucleated_cell")
Sys.setenv(GIT_OPTIONAL_LOCKS = "0")

abort <- function(format, ...) {
  stop(sprintf(format, ...), call. = FALSE)
}

usage <- function() {
  paste(
    "Usage: Rscript 31_predict_reference_cell_state_shard.R",
    "--reference-root ABS --dependency-lock LOCK.tsv --model-dir MODEL",
    "--model-acceptance-receipt ACCEPTED.json --model-acceptance-sha256 SHA256",
    "--feature-shard ONE.tsv --feature-receipt ONE.json",
    "--output-tsv GENERATION/predictions.tsv --output-receipt GENERATION/receipt.json"
  )
}

parse_args <- function(arguments) {
  if (any(arguments %in% c("-h", "--help"))) {
    writeLines(usage())
    quit(save = "no", status = 0L, runLast = FALSE)
  }
  allowed <- c(
    "--reference-root", "--dependency-lock", "--model-dir",
    "--model-acceptance-receipt", "--model-acceptance-sha256",
    "--feature-shard", "--feature-receipt", "--output-tsv",
    "--output-receipt"
  )
  result <- list()
  index <- 1L
  while (index <= length(arguments)) {
    current <- arguments[[index]]
    if (!current %in% allowed) abort("Unknown argument: %s\n%s", current, usage())
    if (index == length(arguments)) abort("Missing value for %s\n%s", current, usage())
    name <- gsub("-", "_", substring(current, 3L), fixed = TRUE)
    if (!is.null(result[[name]])) abort("Argument supplied more than once: %s", current)
    result[[name]] <- arguments[[index + 1L]]
    index <- index + 2L
  }
  required <- gsub("-", "_", substring(allowed, 3L), fixed = TRUE)
  missing <- required[!vapply(required, function(name) {
    !is.null(result[[name]]) && nzchar(result[[name]])
  }, logical(1))]
  if (length(missing)) abort("Missing required arguments: %s\n%s", paste(missing, collapse = ", "), usage())
  result
}

normalize_input <- function(path, label, directory = FALSE) {
  if (!grepl("^/", path)) abort("%s must be an absolute path: %s", label, path)
  normalized <- normalizePath(path, winslash = "/", mustWork = TRUE)
  if (directory && !dir.exists(normalized)) abort("%s is not a directory: %s", label, normalized)
  if (!directory && (!file.exists(normalized) || dir.exists(normalized))) {
    abort("%s is not a file: %s", label, normalized)
  }
  normalized
}

normalize_output <- function(path, label) {
  if (!grepl("^/", path)) abort("%s must be an absolute path: %s", label, path)
  ancestor <- path
  suffix <- character()
  while (!file.exists(ancestor) && !dir.exists(ancestor)) {
    parent <- dirname(ancestor)
    if (identical(parent, ancestor)) abort("%s has no existing ancestor", label)
    suffix <- c(basename(ancestor), suffix)
    ancestor <- parent
  }
  normalized <- normalizePath(ancestor, winslash = "/", mustWork = TRUE)
  if (length(suffix)) normalized <- do.call(file.path, as.list(c(normalized, suffix)))
  normalized
}

is_within <- function(path, root) {
  identical(path, root) || startsWith(path, paste0(root, "/"))
}

run_git <- function(root, ...) {
  arguments <- c("-C", root, ...)
  status <- suppressWarnings(system2("git", arguments, stdout = TRUE, stderr = TRUE))
  exit_status <- attr(status, "status")
  if (!is.null(exit_status) && exit_status != 0L) {
    abort("Unable to inspect pinned reference checkout with git: %s", paste(status, collapse = "\n"))
  }
  trimws(paste(status, collapse = "\n"))
}

load_reference <- function(root) {
  source_files <- sort(
    list.files(file.path(root, "R"), pattern = "[.]R$", full.names = TRUE),
    method = "radix"
  )
  if (!length(source_files)) abort("Reference checkout contains no R source files: %s", root)
  environment <- new.env(parent = baseenv())
  for (source_file in source_files) sys.source(source_file, envir = environment)
  required <- c(
    "cpa_hash_file", "cpa_canonical_json", "cpa_source_checkout_identity",
    "cpa_classifier_runtime_identity", "cpa_classifier_require_glmnet",
    "cpa_verify_classifier_generation", "cpa_classifier_apply_preprocessor",
    "cpa_classifier_probability_matrix", "cpa_read_delimited_table",
    "cpa_parse_numeric_column", "cpa_validate_id_vector",
    "cpa_write_tsv_atomic", "cpa_write_text_atomic"
  )
  missing <- required[!vapply(required, exists, logical(1), envir = environment, inherits = FALSE)]
  if (length(missing)) abort("Pinned reference API is incomplete: %s", paste(missing, collapse = ", "))
  environment
}

load_historical_reference <- function(root) {
  adapter_script <- normalizePath(
    sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1L]]),
    winslash = "/", mustWork = TRUE
  )
  environment <- new.env(parent = globalenv())
  sys.source(
    file.path(dirname(adapter_script), "_shared", "reference_cell_state_v2.R"),
    envir = environment
  )
  loaded <- environment$reference_cell_state_v2_load_historical_classifier(root)
  list(shared = environment, historical = loaded)
}

read_dependency_lock <- function(path) {
  lock <- utils::read.delim(
    path, header = TRUE, sep = "\t", quote = "", comment.char = "",
    check.names = FALSE, colClasses = "character", stringsAsFactors = FALSE
  )
  required <- c(
    "dependency", "repository", "commit", "tree", "license_path",
    "license_sha256", "description_sha256", "execution_mode"
  )
  if (!identical(names(lock), required)) {
    abort("Dependency lock columns changed: expected=%s observed=%s",
          paste(required, collapse = ","), paste(names(lock), collapse = ","))
  }
  selected <- which(lock$dependency == "cellphenotypeannotator")
  if (length(selected) != 1L) abort("Dependency lock must contain exactly one cellphenotypeannotator row")
  row <- lock[selected, , drop = FALSE]
  if (!identical(row$execution_mode[[1L]], "private_read_only_source_checkout")) {
    abort("Unsupported dependency execution_mode: %s", row$execution_mode[[1L]])
  }
  as.list(row[1L, , drop = FALSE])
}

verify_dependency <- function(reference_root, lock_path, api) {
  lock <- read_dependency_lock(lock_path)
  checkout_status <- run_git(
    reference_root, "status", "--porcelain", "--untracked-files=all"
  )
  if (nzchar(checkout_status)) {
    abort(
      "Pinned reference checkout is not clean; tracked and untracked changes are forbidden:\n%s",
      checkout_status
    )
  }
  observed_commit <- run_git(reference_root, "rev-parse", "HEAD")
  observed_tree <- run_git(reference_root, "rev-parse", "HEAD^{tree}")
  if (!identical(observed_commit, lock$commit)) {
    abort("Reference commit mismatch: expected=%s observed=%s", lock$commit, observed_commit)
  }
  if (!identical(observed_tree, lock$tree)) {
    abort("Reference tree mismatch: expected=%s observed=%s", lock$tree, observed_tree)
  }
  license <- normalizePath(file.path(reference_root, lock$license_path), winslash = "/", mustWork = TRUE)
  description <- normalizePath(file.path(reference_root, "DESCRIPTION"), winslash = "/", mustWork = TRUE)
  observed_license <- api$cpa_hash_file(license)
  observed_description <- api$cpa_hash_file(description)
  if (!identical(observed_license, lock$license_sha256)) abort("Pinned reference LICENSE hash mismatch")
  if (!identical(observed_description, lock$description_sha256)) abort("Pinned reference DESCRIPTION hash mismatch")
  list(
    dependency = lock$dependency,
    repository = lock$repository,
    commit = observed_commit,
    tree = observed_tree,
    license_path = lock$license_path,
    license_sha256 = observed_license,
    description_sha256 = observed_description,
    execution_mode = lock$execution_mode
  )
}

as_character_vector <- function(value, label) {
  result <- as.character(unlist(value, recursive = TRUE, use.names = FALSE))
  if (!length(result) || anyNA(result) || any(!nzchar(result))) abort("%s is empty or invalid", label)
  result
}

validate_model_acceptance <- function(path, expected_sha256, model_dir, bundle,
                                      lock_path, output_tsv, output_receipt, api,
                                      shared_script_path) {
  acceptance <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  if (!identical(as.character(acceptance$schema_version), MODEL_ACCEPTANCE_SCHEMA) ||
      !identical(as.character(acceptance$status), "ACCEPTED") ||
      !isTRUE(acceptance$accepted)) {
    abort("Model-acceptance receipt is not an accepted reference-cell-state generation")
  }
  shadow <- normalize_input(as.character(acceptance$shadow_root),
                            "model acceptance.shadow_root", directory = TRUE)
  if (!is_within(path, shadow) || !is_within(model_dir, shadow) ||
      !is_within(output_tsv, shadow) || !is_within(output_receipt, shadow)) {
    abort("Model, acceptance receipt, and prediction outputs must share the reference shadow root")
  }
  parent_root <- normalize_input(as.character(acceptance$frozen_inputs$parent_shadow_root),
                                 "model acceptance.parent_shadow_root", directory = TRUE)
  parent_import <- normalize_input(as.character(acceptance$frozen_inputs$parent_import_manifest),
                                   "model acceptance.parent_import_manifest")
  acceptance_implementation <- normalize_input(
    as.character(acceptance$implementation), "model acceptance implementation"
  )
  expected_acceptance_implementation <- normalize_input(
    file.path(dirname(shared_script_path), "..", "30_accept_reference_cell_state_model.R"),
    "expected model acceptance implementation"
  )
  if (!is_within(parent_import, shadow) ||
      !identical(api$cpa_hash_file(parent_import),
                 as.character(acceptance$frozen_inputs$parent_import_manifest_sha256))) {
    abort("Model acceptance no longer binds its parent-import manifest")
  }
  manifest_path <- normalizePath(
    file.path(model_dir, "morphology_cell_state_classifier_uncalibrated_training_manifest.tsv"),
    winslash = "/", mustWork = TRUE
  )
  model_path <- normalizePath(
    file.path(model_dir, "morphology_cell_state_classifier_uncalibrated_model.rds"),
    winslash = "/", mustWork = TRUE
  )
  if (!identical(normalizePath(as.character(acceptance$model$dir), winslash = "/", mustWork = TRUE),
                 model_dir) ||
      !identical(as.character(acceptance$model$model_sha256), as.character(bundle$model_sha256)) ||
      !identical(as.character(acceptance$model$model_rds_sha256), api$cpa_hash_file(model_path)) ||
      !identical(as.character(acceptance$model$manifest_sha256), api$cpa_hash_file(manifest_path)) ||
      !identical(as.character(acceptance$model$feature_columns), REFERENCE_FEATURES) ||
      !identical(as.character(acceptance$model$class_ids), REFERENCE_CLASS_IDS) ||
      !identical(as.character(acceptance$dependency$lock_sha256), api$cpa_hash_file(lock_path)) ||
      !identical(acceptance_implementation, expected_acceptance_implementation) ||
      !identical(as.character(acceptance$implementation_sha256),
                 api$cpa_hash_file(expected_acceptance_implementation)) ||
      !identical(normalizePath(as.character(acceptance$shared_implementation),
                               winslash = "/", mustWork = TRUE), shared_script_path) ||
      !identical(as.character(acceptance$shared_implementation_sha256),
                 api$cpa_hash_file(shared_script_path)) ||
      !identical(api$cpa_hash_file(path), expected_sha256)) {
    abort("Model generation differs from the accepted reference-cell-state identity")
  }
  list(
    shadow_root = shadow,
    parent_shadow_root = parent_root,
    parent_import_manifest = parent_import,
    parent_import_manifest_sha256 = api$cpa_hash_file(parent_import),
    model_sha256 = api$cpa_hash_file(model_path),
    model_id = as.character(acceptance$model$model_id),
    model_manifest = manifest_path
  )
}

sha256_cell_ids <- function(cell_ids) {
  digest::digest(enc2utf8(paste(cell_ids, collapse = "\n")), algo = "sha256", serialize = FALSE)
}

validate_feature_generation <- function(feature_shard, feature_receipt_path, api) {
  receipt <- jsonlite::fromJSON(feature_receipt_path, simplifyVector = TRUE)
  required <- c(
    "schema_version", "status", "key", "well", "branch", "feature_schema_version",
    "feature_columns", "feature_tsv", "feature_tsv_sha256", "row_count", "cell_id_sha256"
  )
  missing <- setdiff(required, names(receipt))
  if (length(missing)) abort("Feature receipt is incomplete: %s", paste(missing, collapse = ", "))
  if (!identical(receipt$schema_version, FEATURE_RECEIPT_SCHEMA) || !identical(receipt$status, "COMPLETE")) {
    abort("Feature receipt is not a COMPLETE %s generation", FEATURE_RECEIPT_SCHEMA)
  }
  receipt_feature <- normalizePath(receipt$feature_tsv, winslash = "/", mustWork = TRUE)
  if (!identical(receipt_feature, feature_shard)) abort("Feature receipt points to a different feature shard")
  observed_hash <- api$cpa_hash_file(feature_shard)
  if (!identical(receipt$feature_tsv_sha256, observed_hash)) abort("Feature shard hash does not match its receipt")

  table <- api$cpa_read_delimited_table(feature_shard, "reference cell-state feature shard")
  receipt_columns <- as_character_vector(receipt$feature_columns, "feature receipt.feature_columns")
  if (!identical(names(table), receipt_columns)) abort("Feature shard columns do not exactly match its receipt")
  required_columns <- unique(c(
    "cell_id", "key", "well", "branch", "image_id", "mask_label",
    "feature_schema_version", REFERENCE_FEATURES
  ))
  missing_columns <- setdiff(required_columns, names(table))
  if (length(missing_columns)) abort("Feature shard lacks model/identity columns: %s", paste(missing_columns, collapse = ", "))

  table$cell_id <- api$cpa_validate_id_vector(table$cell_id, "feature shard.cell_id", unique = TRUE)
  if (nrow(table) != as.integer(receipt$row_count)) abort("Feature shard row_count does not match its receipt")
  observed_cell_hash <- sha256_cell_ids(table$cell_id)
  if (!identical(observed_cell_hash, receipt$cell_id_sha256)) abort("Feature shard cell_id hash does not match its receipt")
  identity_expectations <- list(
    key = receipt$key, well = receipt$well, branch = receipt$branch,
    image_id = receipt$key, feature_schema_version = receipt$feature_schema_version
  )
  for (field in names(identity_expectations)) {
    if (any(table[[field]] != identity_expectations[[field]])) {
      abort("Feature shard identity column %s disagrees with its receipt", field)
    }
  }
  labels <- api$cpa_parse_numeric_column(table$mask_label, "feature shard.mask_label", allow_missing = FALSE)
  if (any(labels < 1 | labels != floor(labels)) || (length(labels) > 1L && any(diff(labels) <= 0))) {
    abort("Feature shard mask_label must be positive and strictly increasing")
  }
  expected_cell_ids <- paste(receipt$branch, receipt$key, as.integer(labels), sep = "|")
  if (!identical(table$cell_id, expected_cell_ids)) abort("Feature shard stable cell identity changed")

  historical <- table
  for (index in seq_along(REFERENCE_FEATURES)) {
    historical[[HISTORICAL_FEATURES[[index]]]] <- api$cpa_parse_numeric_column(
      table[[REFERENCE_FEATURES[[index]]]],
      sprintf("feature shard.%s", REFERENCE_FEATURES[[index]]), allow_missing = TRUE
    )
  }
  list(table = table, historical = historical, receipt = receipt, feature_sha256 = observed_hash,
       cell_id_sha256 = observed_cell_hash)
}

validate_existing <- function(output_tsv, output_receipt, expected,
                              current_output_sha256, api) {
  if (!file.exists(output_tsv) && !file.exists(output_receipt)) return(FALSE)
  if (!file.exists(output_tsv) || !file.exists(output_receipt)) {
    abort("Incomplete existing shard prediction generation was preserved")
  }
  receipt <- jsonlite::fromJSON(output_receipt, simplifyVector = FALSE)
  scalar_expected <- c(
    "schema_version", "status", "key", "well", "branch", "model_id",
    "feature_tsv_sha256", "feature_receipt_sha256", "model_sha256",
    "model_manifest_sha256", "model_acceptance_receipt", "model_acceptance_sha256",
    "parent_import_manifest_sha256", "dependency_lock_sha256", "implementation",
    "implementation_sha256", "shared_implementation", "shared_implementation_sha256",
    "row_count", "cell_id_sha256", "ok_count", "unavailable_count",
    "max_probability_sum_error"
  )
  for (field in scalar_expected) {
    if (!identical(as.character(receipt[[field]]), as.character(expected[[field]]))) {
      abort("Existing shard prediction receipt differs for %s", field)
    }
  }
  if (!identical(api$cpa_canonical_json(receipt$class_ids), api$cpa_canonical_json(expected$class_ids)) ||
      !identical(api$cpa_canonical_json(receipt$output_columns), api$cpa_canonical_json(expected$output_columns)) ||
      !identical(api$cpa_canonical_json(receipt$source_identity), api$cpa_canonical_json(expected$source_identity)) ||
      !identical(api$cpa_canonical_json(receipt$runtime_identity), api$cpa_canonical_json(expected$runtime_identity))) {
    abort("Existing shard prediction source, runtime, or class identity differs")
  }
  observed_output_hash <- api$cpa_hash_file(output_tsv)
  if (!identical(as.character(receipt$prediction_tsv_sha256), observed_output_hash)) {
    abort("Existing shard prediction TSV hash does not match its receipt")
  }
  if (!identical(observed_output_hash, current_output_sha256)) {
    abort("Existing shard prediction TSV differs from the freshly computed canonical wide output")
  }
  TRUE
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  reference_root <- normalize_input(args$reference_root, "--reference-root", directory = TRUE)
  lock_path <- normalize_input(args$dependency_lock, "--dependency-lock")
  model_dir <- normalize_input(args$model_dir, "--model-dir", directory = TRUE)
  model_acceptance_receipt <- normalize_input(
    args$model_acceptance_receipt, "--model-acceptance-receipt"
  )
  if (!grepl("^[0-9a-f]{64}$", args$model_acceptance_sha256)) {
    abort("--model-acceptance-sha256 must be one lowercase SHA-256 digest")
  }
  feature_shard <- normalize_input(args$feature_shard, "--feature-shard")
  feature_receipt_path <- normalize_input(args$feature_receipt, "--feature-receipt")
  output_tsv <- normalize_output(args$output_tsv, "--output-tsv")
  output_receipt <- normalize_output(args$output_receipt, "--output-receipt")
  if (identical(output_tsv, output_receipt)) abort("Output TSV and receipt paths must differ")
  generation_dir <- dirname(output_tsv)
  if (!identical(dirname(output_receipt), generation_dir)) {
    abort("Output TSV and receipt must share one immutable shard-generation directory")
  }
  if (is_within(output_tsv, reference_root) || is_within(output_receipt, reference_root)) {
    abort("Outputs must not be written inside the read-only reference checkout")
  }

  api <- load_reference(reference_root)
  observed_acceptance_sha256 <- api$cpa_hash_file(model_acceptance_receipt)
  if (!identical(observed_acceptance_sha256, args$model_acceptance_sha256)) {
    abort("Model-acceptance receipt SHA mismatch")
  }
  dependency <- verify_dependency(reference_root, lock_path, api)
  api$cpa_classifier_require_glmnet()
  historical_loaded <- load_historical_reference(reference_root)
  historical <- historical_loaded$historical$api
  model_path <- normalizePath(
    file.path(model_dir, "morphology_cell_state_classifier_uncalibrated_model.rds"),
    winslash = "/", mustWork = TRUE
  )
  model <- readRDS(model_path)
  historical$validate_morphology_cell_state_classifier_model_bundle(
    model, expected_feature_profile = "promoted_shape_plus_rfs_boundary"
  )
  class_ids <- as_character_vector(model$classifier$classes, "model.classifier.classes")
  if (!identical(class_ids, REFERENCE_CLASS_IDS)) {
    abort("Model must contain exactly live_cell, dead_cell, and multinucleated_cell")
  }
  if (!identical(as.character(model$feature_config$feature_columns), HISTORICAL_FEATURES) ||
      !identical(model$classifier$engine, "glmnet") ||
      !identical(model$classifier$requested_engine, "glmnet")) {
    abort("Model is not the exact historical promoted 12-feature glmnet contract")
  }
  script_path <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1L]]),
                               winslash = "/", mustWork = TRUE)
  shared_script_path <- normalizePath(
    file.path(dirname(script_path), "_shared", "reference_cell_state_v2.R"),
    winslash = "/", mustWork = TRUE
  )
  acceptance_identity <- validate_model_acceptance(
    model_acceptance_receipt, observed_acceptance_sha256, model_dir, model,
    lock_path, output_tsv, output_receipt, api, shared_script_path
  )
  if (!is_within(feature_shard, acceptance_identity$parent_shadow_root) ||
      !is_within(feature_receipt_path, acceptance_identity$parent_shadow_root)) {
    abort("Feature shard and receipt must be inside the accepted read-only parent shadow root")
  }
  source_identity <- api$cpa_source_checkout_identity(reference_root)
  runtime_identity <- api$cpa_classifier_runtime_identity()

  feature <- validate_feature_generation(feature_shard, feature_receipt_path, api)
  model_manifest_path <- acceptance_identity$model_manifest
  common <- list(
    schema_version = SCHEMA_VERSION,
    status = "COMPLETE",
    key = feature$receipt$key,
    well = feature$receipt$well,
    branch = feature$receipt$branch,
    model_id = acceptance_identity$model_id,
    class_ids = as.list(class_ids),
    feature_tsv_sha256 = feature$feature_sha256,
    feature_receipt_sha256 = api$cpa_hash_file(feature_receipt_path),
    model_sha256 = api$cpa_hash_file(model_path),
    model_manifest_sha256 = api$cpa_hash_file(model_manifest_path),
    model_acceptance_receipt = model_acceptance_receipt,
    model_acceptance_sha256 = observed_acceptance_sha256,
    parent_import_manifest_sha256 = acceptance_identity$parent_import_manifest_sha256,
    dependency_lock_sha256 = api$cpa_hash_file(lock_path),
    source_identity = source_identity,
    runtime_identity = runtime_identity,
    implementation = script_path,
    implementation_sha256 = api$cpa_hash_file(script_path),
    shared_implementation = shared_script_path,
    shared_implementation_sha256 = api$cpa_hash_file(shared_script_path)
  )
  historical_values <- as.matrix(data.frame(lapply(
    feature$historical[HISTORICAL_FEATURES], function(value) suppressWarnings(as.numeric(value))
  ), check.names = FALSE))
  available <- rowSums(is.finite(historical_values)) > 0L
  probability <- matrix(NA_real_, nrow = nrow(feature$table), ncol = length(class_ids),
                        dimnames = list(NULL, class_ids))
  if (any(available)) probability[available, ] <- historical$predict_morphology_cell_state_probabilities(
    model$classifier, feature$historical[available, , drop = FALSE]
  )
  if (any(available)) {
    available_probability <- probability[available, , drop = FALSE]
    if (any(!is.finite(available_probability)) ||
        any(available_probability < -1e-12 | available_probability > 1 + 1e-12)) {
      abort("Reference probability output contains non-finite or out-of-range values")
    }
    probability_sum_error <- abs(rowSums(available_probability) - 1)
    max_probability_sum_error <- max(probability_sum_error)
    if (max_probability_sum_error > 1e-6) {
      abort("Reference probability rows do not sum to one within tolerance: %.17g",
            max_probability_sum_error)
    }
  } else {
    max_probability_sum_error <- 0
  }
  if (any(!available) && any(!is.na(probability[!available, , drop = FALSE]))) {
    abort("Unavailable prediction rows unexpectedly contain probabilities")
  }
  predicted <- rep("", nrow(feature$table))
  if (any(available)) {
    predicted_from_probability <- class_ids[max.col(
      probability[available, , drop = FALSE], ties.method = "first"
    )]
    predicted[available] <- predicted_from_probability
    if (!identical(predicted[available], predicted_from_probability)) {
      abort("Predicted class is inconsistent with the first-maximum tie rule")
    }
  }
  output <- data.frame(
    model_id = rep(acceptance_identity$model_id, nrow(feature$table)),
    cell_id = feature$table$cell_id,
    predicted_class_id = predicted,
    prediction_status = ifelse(available, "ok", "unavailable_missing_features"),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  for (index in seq_along(class_ids)) {
    output[[paste0("probability__", class_ids[[index]])]] <- probability[, index]
  }
  output_columns <- c(
    "model_id", "cell_id", "predicted_class_id", "prediction_status",
    paste0("probability__", class_ids)
  )
  if (!identical(names(output), output_columns)) abort("Internal wide prediction schema changed")

  common$ok_count <- sum(available)
  common$unavailable_count <- sum(!available)
  common$max_probability_sum_error <- max_probability_sum_error
  common$row_count <- nrow(output)
  common$cell_id_sha256 <- feature$cell_id_sha256
  common$output_columns <- as.list(output_columns)
  dir.create(dirname(generation_dir), recursive = TRUE, showWarnings = FALSE)
  comparison_path <- tempfile(
    paste0(".", basename(output_tsv), "-current-"), tmpdir = dirname(generation_dir)
  )
  on.exit(unlink(comparison_path), add = TRUE)
  api$cpa_write_tsv_atomic(output, comparison_path)
  current_output_sha256 <- api$cpa_hash_file(comparison_path)
  unlink(comparison_path)
  if (dir.exists(generation_dir) && validate_existing(
    output_tsv, output_receipt, common, current_output_sha256, api
  )) {
    writeLines(sprintf("reference_cell_state_shard_prediction_already_complete=1 key=%s rows=%d output=%s",
                       feature$receipt$key, nrow(feature$table), output_tsv))
    return(invisible(0L))
  }

  if (file.exists(generation_dir) || dir.exists(generation_dir)) {
    abort("Shard prediction generation already exists but could not be verified")
  }
  receipt <- c(common, list(
    feature_tsv = feature_shard,
    feature_receipt = feature_receipt_path,
    feature_schema_version = feature$receipt$feature_schema_version,
    prediction_tsv = output_tsv,
    prediction_tsv_sha256 = current_output_sha256,
    model_dir = model_dir,
    reference_root = reference_root,
    dependency_lock = lock_path,
    dependency = dependency,
    inference_contract = list(
      preprocessor = "historical_apply_morphology_cell_state_preprocessor",
      probability = "historical_predict_morphology_cell_state_probabilities",
      feature_profile = "promoted_shape_plus_rfs_boundary",
      probability_calibration = "uncalibrated_stratified_review_sample",
      tie_break = "max.col(ties.method=first)",
      unavailable_status = "unavailable_missing_features"
    ),
    isolation_contract = list(
      semantic_axis = "reference_cell_state",
      parent_features_read_only = TRUE,
      current_classification_read = FALSE,
      dead_channel_read = FALSE,
      existing_classification_overwritten = FALSE
    )
  ))
  dir.create(dirname(generation_dir), recursive = TRUE, showWarnings = FALSE)
  staging <- tempfile(paste0(".", basename(generation_dir), "-staging-"),
                      tmpdir = dirname(generation_dir))
  if (!dir.create(staging)) abort("Could not create shard prediction staging directory")
  on.exit(unlink(staging, recursive = TRUE, force = TRUE), add = TRUE)
  staged_tsv <- file.path(staging, basename(output_tsv))
  staged_receipt <- file.path(staging, basename(output_receipt))
  api$cpa_write_tsv_atomic(output, staged_tsv, overwrite = FALSE)
  if (!identical(api$cpa_hash_file(staged_tsv), current_output_sha256)) {
    abort("Staged shard prediction hash changed")
  }
  api$cpa_write_text_atomic(
    api$cpa_canonical_json(receipt, pretty = TRUE), staged_receipt,
    overwrite = FALSE
  )
  if (file.exists(generation_dir) || dir.exists(generation_dir)) {
    abort("Shard prediction generation appeared during staging")
  }
  if (!file.rename(staging, generation_dir)) {
    abort("Failed to install shard prediction generation atomically")
  }
  writeLines(sprintf(
    "reference_cell_state_shard_prediction_complete=1 key=%s rows=%d ok=%d unavailable=%d output=%s receipt=%s",
    feature$receipt$key, nrow(output), sum(available), sum(!available), output_tsv, output_receipt
  ))
  invisible(0L)
}

status <- tryCatch(main(), error = function(error) {
  writeLines(conditionMessage(error), con = stderr())
  1L
})
quit(save = "no", status = as.integer(status), runLast = FALSE)
