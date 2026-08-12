#!/usr/bin/env Rscript

# Certify that one immutable CPA classifier generation belongs to the exact
# representative project and input-addressed train stage of this shadow run.

SCHEMA_VERSION <- "reference_cell_state_model_acceptance_v1"
PARENT_IMPORT_SCHEMA_VERSION <- "reference_cell_state_parent_import_v1"
PROJECT_ID <- "reference_cell_state_development"
REFERENCE_FEATURES <- c(
  "area_px2", "perimeter_px", "roundness", "aspect_ratio", "extent",
  "solidity", "equivalent_diameter_px", "major_axis_px", "minor_axis_px"
)
REFERENCE_CLASS_IDS <- c("dead_cell", "live_cell", "multinucleated_cell")
REFERENCE_PRIORITY <- c("multinucleated_cell", "dead_cell", "live_cell")
Sys.setenv(GIT_OPTIONAL_LOCKS = "0")

abort <- function(format, ...) stop(sprintf(format, ...), call. = FALSE)

parse_args <- function(arguments) {
  allowed <- c(
    "--shadow-root", "--project", "--model-dir", "--train-receipt",
    "--feature-config", "--classes-file", "--parent-import-manifest",
    "--reference-root", "--dependency-lock", "--output-receipt"
  )
  result <- list()
  index <- 1L
  while (index <= length(arguments)) {
    current <- arguments[[index]]
    if (!current %in% allowed || index == length(arguments)) {
      abort("Unknown argument or missing value: %s", current)
    }
    name <- gsub("-", "_", substring(current, 3L), fixed = TRUE)
    if (!is.null(result[[name]])) abort("Argument supplied more than once: %s", current)
    result[[name]] <- arguments[[index + 1L]]
    index <- index + 2L
  }
  required <- gsub("-", "_", substring(allowed, 3L), fixed = TRUE)
  missing <- required[!vapply(required, function(name) {
    !is.null(result[[name]]) && nzchar(result[[name]])
  }, logical(1))]
  if (length(missing)) abort("Missing required arguments: %s", paste(missing, collapse = ", "))
  result
}

normalize_input <- function(path, label, directory = FALSE) {
  if (!grepl("^/", path)) abort("%s must be absolute: %s", label, path)
  normalized <- normalizePath(path, winslash = "/", mustWork = TRUE)
  if (directory && !dir.exists(normalized)) abort("%s is not a directory", label)
  if (!directory && (!file.exists(normalized) || dir.exists(normalized))) abort("%s is not a file", label)
  normalized
}

normalize_output <- function(path, label) {
  if (!grepl("^/", path)) abort("%s must be absolute: %s", label, path)
  normalizePath(path, winslash = "/", mustWork = FALSE)
}

is_within <- function(path, root) identical(path, root) || startsWith(path, paste0(root, "/"))

run_git <- function(root, ...) {
  output <- suppressWarnings(system2("git", c("-C", root, ...), stdout = TRUE, stderr = TRUE))
  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) abort("Unable to inspect reference checkout")
  trimws(paste(output, collapse = "\n"))
}

load_reference <- function(root) {
  files <- sort(list.files(file.path(root, "R"), pattern = "[.]R$", full.names = TRUE), method = "radix")
  if (!length(files)) abort("Reference checkout contains no R sources")
  environment <- new.env(parent = baseenv())
  for (file in files) sys.source(file, envir = environment)
  required <- c(
    "validate_project", "cpa_verify_classifier_generation", "cpa_hash_file",
    "cpa_hash_object", "cpa_canonical_json", "cpa_write_text_atomic",
    "cpa_relative_path", "cpa_source_checkout_identity"
  )
  missing <- required[!vapply(required, exists, logical(1), envir = environment, inherits = FALSE)]
  if (length(missing)) abort("Pinned reference API is incomplete: %s", paste(missing, collapse = ", "))
  environment
}

verify_dependency <- function(root, lock_path, api) {
  lock <- utils::read.delim(
    lock_path, header = TRUE, sep = "\t", quote = "", comment.char = "",
    check.names = FALSE, colClasses = "character", stringsAsFactors = FALSE
  )
  selected <- which(lock$dependency == "cellphenotypeannotator")
  if (length(selected) != 1L) abort("Dependency lock must contain exactly one cellphenotypeannotator row")
  row <- lock[selected, , drop = FALSE]
  observed <- list(
    commit = run_git(root, "rev-parse", "HEAD"),
    tree = run_git(root, "rev-parse", "HEAD^{tree}"),
    status = run_git(root, "status", "--porcelain", "--untracked-files=all"),
    license_sha256 = api$cpa_hash_file(file.path(root, row$license_path[[1L]])),
    description_sha256 = api$cpa_hash_file(file.path(root, "DESCRIPTION"))
  )
  if (nzchar(observed$status) ||
      !identical(observed$commit, row$commit[[1L]]) ||
      !identical(observed$tree, row$tree[[1L]]) ||
      !identical(observed$license_sha256, row$license_sha256[[1L]]) ||
      !identical(observed$description_sha256, row$description_sha256[[1L]]) ||
      !identical(row$execution_mode[[1L]], "private_read_only_source_checkout")) {
    abort("Pinned reference checkout differs from its dependency lock")
  }
  list(
    root = root, lock = lock_path, lock_sha256 = api$cpa_hash_file(lock_path),
    commit = observed$commit, tree = observed$tree,
    license_sha256 = observed$license_sha256,
    description_sha256 = observed$description_sha256,
    execution_mode = row$execution_mode[[1L]]
  )
}

command_argument <- function(command, flag) {
  positions <- which(as.character(unlist(command, use.names = FALSE)) == flag)
  if (length(positions) != 1L) abort("Train receipt command must contain exactly one %s", flag)
  values <- as.character(unlist(command, use.names = FALSE))
  if (positions[[1L]] >= length(values)) abort("Train receipt %s has no value", flag)
  values[[positions[[1L]] + 1L]]
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  shadow <- normalize_input(args$shadow_root, "--shadow-root", directory = TRUE)
  project <- normalize_input(args$project, "--project")
  model_dir <- normalize_input(args$model_dir, "--model-dir", directory = TRUE)
  train_receipt_path <- normalize_input(args$train_receipt, "--train-receipt")
  feature_config_path <- normalize_input(args$feature_config, "--feature-config")
  classes_file <- normalize_input(args$classes_file, "--classes-file")
  parent_import_path <- normalize_input(args$parent_import_manifest, "--parent-import-manifest")
  reference_root <- normalize_input(args$reference_root, "--reference-root", directory = TRUE)
  dependency_lock <- normalize_input(args$dependency_lock, "--dependency-lock")
  output <- normalize_output(args$output_receipt, "--output-receipt")
  for (item in c(project, model_dir, train_receipt_path, parent_import_path, output)) {
    if (!is_within(item, shadow)) abort("Project/model/train/parent-import/output must all be inside one shadow root: %s", item)
  }
  expected_parent_import <- normalizePath(
    file.path(dirname(project), "parent_import_manifest.json"), winslash = "/", mustWork = TRUE
  )
  if (!identical(parent_import_path, expected_parent_import)) {
    abort("Parent-import manifest must be the immutable sibling of the representative project")
  }

  api <- load_reference(reference_root)
  dependency <- verify_dependency(reference_root, dependency_lock, api)
  validation <- api$validate_project(project, stage = "predict", strict = TRUE)
  generation <- api$cpa_verify_classifier_generation(model_dir, validation = validation)
  manifest <- generation$manifest

  expected_model_dir <- normalizePath(file.path(
    validation$config$runs_dir, validation$config$project_id, validation$run_id,
    "model", as.character(manifest$model_id)
  ), winslash = "/", mustWork = FALSE)
  if (!identical(model_dir, expected_model_dir)) {
    abort("Model directory is not the classifier generation expected by this representative project/run")
  }

  feature_config <- jsonlite::fromJSON(feature_config_path, simplifyVector = TRUE)
  if (!identical(as.character(feature_config$schema_version),
                 "reference_cell_state_feature_config_v1") ||
      !identical(as.character(feature_config$method_version),
                 "reference_promoted_shape_v1")) {
    abort("Unsupported reference cell-state feature configuration")
  }
  classifier_features <- as.character(feature_config$classifier_feature_columns)
  if (!identical(classifier_features, REFERENCE_FEATURES)) {
    abort("Feature config must contain the exact historical promoted-shape nine-feature allowlist")
  }
  if (!identical(as.character(validation$config$classifier$feature_columns), classifier_features) ||
      !identical(as.character(manifest$feature_columns), classifier_features)) {
    abort("Model/project feature_columns differ from the frozen historical promoted-shape allowlist")
  }
  project_classes <- normalizePath(validation$config$classes_file, winslash = "/", mustWork = TRUE)
  if (!identical(api$cpa_hash_file(project_classes), api$cpa_hash_file(classes_file))) {
    abort("Representative project classes differ from the frozen source classes")
  }
  class_ids <- sort(as.character(validation$classes$class_id[validation$classes$trainable]), method = "radix")
  if (!identical(class_ids, REFERENCE_CLASS_IDS) ||
      !identical(as.character(manifest$class_ids), REFERENCE_CLASS_IDS)) {
    abort("Model/project must contain exactly dead_cell, live_cell, and multinucleated_cell")
  }

  classifier <- validation$config$classifier
  if (!identical(as.character(validation$config$project_id), PROJECT_ID) ||
      !identical(as.character(classifier$engine), "glmnet_multinomial") ||
      !isTRUE(all.equal(as.numeric(classifier$alpha), 1.0)) ||
      !identical(as.character(classifier$lambda_rule), "lambda.1se") ||
      !identical(as.integer(classifier$outer_folds), 5L) ||
      !identical(as.integer(classifier$inner_folds), 5L) ||
      !identical(as.character(classifier$group_column), "well") ||
      !identical(as.numeric(classifier$minimum_confidence), 0.8) ||
      !identical(sort(as.character(classifier$eligible_review_status), method = "radix"),
                 c("confirmed", "corrected"))) {
    abort("Representative project classifier contract differs from the frozen grouped nested-CV lasso contract")
  }

  parent_import <- jsonlite::fromJSON(parent_import_path, simplifyVector = FALSE)
  parent_outputs <- parent_import$output_file_sha256
  output_paths <- list(
    `project.yml` = project,
    `classes.tsv` = normalizePath(validation$config$classes_file, winslash = "/", mustWork = TRUE),
    `cells.tsv` = normalizePath(validation$config$cells_file, winslash = "/", mustWork = TRUE),
    `features.tsv` = normalizePath(validation$config$features_file, winslash = "/", mustWork = TRUE),
    `images.tsv` = normalizePath(validation$config$images_file, winslash = "/", mustWork = TRUE)
  )
  if (!identical(as.character(parent_import$schema_version), PARENT_IMPORT_SCHEMA_VERSION) ||
      !identical(as.character(parent_import$project_id), PROJECT_ID) ||
      !identical(as.character(parent_import$config_inputs$feature_config_sha256),
                 api$cpa_hash_file(feature_config_path)) ||
      !identical(as.character(parent_import$config_inputs$classes_file_sha256),
                 api$cpa_hash_file(classes_file)) ||
      !identical(as.character(unlist(parent_import$reference_contract$feature_columns,
                                     recursive = TRUE, use.names = FALSE)), REFERENCE_FEATURES) ||
      !identical(as.character(unlist(parent_import$reference_contract$class_ids_priority_order,
                                     recursive = TRUE, use.names = FALSE)),
                 REFERENCE_PRIORITY)) {
    abort("Parent-import manifest differs from the frozen reference cell-state contract")
  }
  for (name in names(output_paths)) {
    if (!identical(as.character(parent_outputs[[name]]), api$cpa_hash_file(output_paths[[name]]))) {
      abort("Parent-import output hash mismatch for %s", name)
    }
  }

  train_receipt <- jsonlite::fromJSON(train_receipt_path, simplifyVector = FALSE)
  if (!identical(train_receipt$schema_version, "cellphenotypeannotator_stage_receipt_v1") ||
      !identical(train_receipt$stage, "train") || !identical(train_receipt$status, "complete") ||
      !identical(as.integer(train_receipt$returncode), 0L) ||
      !identical(normalizePath(train_receipt$project, winslash = "/", mustWork = TRUE), project) ||
      !identical(as.character(train_receipt$project_sha256), api$cpa_hash_file(project))) {
    abort("Train-stage receipt does not certify this representative project")
  }
  reviewed_labels <- normalizePath(
    command_argument(train_receipt$command, "--reviewed-labels"), winslash = "/", mustWork = TRUE
  )
  if (!is_within(reviewed_labels, shadow)) abort("Reviewed labels must be inside the same shadow root")
  review_manifest <- normalizePath(
    file.path(dirname(reviewed_labels), "review_import_manifest.json"), winslash = "/", mustWork = TRUE
  )
  if (!identical(as.character(manifest$reviewed_labels_path),
                 api$cpa_relative_path(reviewed_labels, validation$config$project_dir)) ||
      !identical(as.character(manifest$reviewed_labels_sha256), api$cpa_hash_file(reviewed_labels)) ||
      !identical(as.character(manifest$review_import_manifest_sha256), api$cpa_hash_file(review_manifest))) {
    abort("Model manifest is not descended from the train receipt's authoritative reviewed labels")
  }
  input_identity_sha <- as.character(train_receipt$input_identity$sha256)
  if (length(input_identity_sha) != 1L || !grepl("^[0-9a-f]{64}$", input_identity_sha)) {
    abort("Train receipt lacks a valid input-addressed identity SHA")
  }

  classifier_config_sha <- api$cpa_hash_object(validation$config$classifier)
  acceptance <- list(
    schema_version = SCHEMA_VERSION,
    status = "ACCEPTED",
    accepted = TRUE,
    shadow_root = shadow,
    project = list(
      path = project,
      sha256 = api$cpa_hash_file(project),
      project_id = validation$config$project_id,
      run_id = validation$run_id,
      class_config_sha256 = validation$class_config_sha256,
      classifier_config_sha256 = classifier_config_sha,
      feature_columns = as.list(classifier_features),
      class_ids = as.list(class_ids)
    ),
    model = list(
      dir = model_dir,
      manifest = generation$paths[["model_manifest"]],
      manifest_sha256 = api$cpa_hash_file(generation$paths[["model_manifest"]]),
      model_id = manifest$model_id,
      project_id = manifest$project_id,
      run_id = manifest$run_id,
      class_config_sha256 = manifest$class_config_sha256,
      classifier_config_sha256 = manifest$classifier_config_sha256,
      feature_columns = as.list(classifier_features),
      class_ids = as.list(class_ids)
    ),
    train_receipt = list(
      path = train_receipt_path,
      sha256 = api$cpa_hash_file(train_receipt_path),
      input_identity_sha256 = input_identity_sha,
      reviewed_labels_sha256 = api$cpa_hash_file(reviewed_labels),
      review_import_manifest_sha256 = api$cpa_hash_file(review_manifest)
    ),
    frozen_inputs = list(
      parent_import_manifest = parent_import_path,
      parent_import_manifest_sha256 = api$cpa_hash_file(parent_import_path),
      parent_shadow_root = as.character(parent_import$parent$shadow_root),
      feature_config = feature_config_path,
      feature_config_sha256 = api$cpa_hash_file(feature_config_path),
      classes_file = classes_file,
      classes_sha256 = api$cpa_hash_file(classes_file)
    ),
    dependency = dependency,
    semantic_verification = list(
      reference_generation_verified = TRUE,
      expected_model_directory_verified = TRUE,
      parent_import_verified = TRUE,
      train_receipt_input_identity_verified = TRUE,
      model_train_parent_verified = TRUE,
      frozen_hashes_verified = TRUE
    )
  )
  payload <- paste0(api$cpa_canonical_json(acceptance, pretty = TRUE), "\n")
  if (file.exists(output)) {
    if (!identical(readChar(output, file.info(output)$size, useBytes = TRUE), payload)) {
      abort("Existing model acceptance receipt differs and was preserved: %s", output)
    }
  } else {
    dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
    api$cpa_write_text_atomic(payload, output)
  }
  if (!identical(run_git(reference_root, "status", "--porcelain", "--untracked-files=all"), "")) {
    abort("Reference checkout changed during model acceptance")
  }
  writeLines(sprintf("reference_cell_state_model_acceptance_complete=1 receipt=%s sha256=%s", output, api$cpa_hash_file(output)))
  invisible(0L)
}

status <- tryCatch(main(), error = function(error) {
  writeLines(conditionMessage(error), con = stderr())
  1L
})
quit(save = "no", status = as.integer(status), runLast = FALSE)
