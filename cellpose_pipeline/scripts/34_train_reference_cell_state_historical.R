#!/usr/bin/env Rscript

# Train the independent V2 axis by directly invoking the pinned historical
# morphology classifier. Generic Cell Phenotype Annotator training is forbidden.

SCHEMA_VERSION <- "reference_cell_state_historical_train_v2"
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
    "--reference-root", "--dependency-lock", "--feature-config", "--project",
    "--reviewed-labels", "--training-stage", "--output-dir", "--output-receipt"
  )
  result <- list()
  i <- 1L
  while (i <= length(arguments)) {
    current <- arguments[[i]]
    if (!current %in% allowed || i == length(arguments)) {
      reference_cell_state_v2_abort("Unknown argument or missing value: %s", current)
    }
    name <- gsub("-", "_", substring(current, 3L), fixed = TRUE)
    if (!is.null(result[[name]])) reference_cell_state_v2_abort("Repeated argument: %s", current)
    result[[name]] <- arguments[[i + 1L]]
    i <- i + 2L
  }
  required <- gsub("-", "_", substring(allowed, 3L), fixed = TRUE)
  missing <- required[!vapply(required, function(name) {
    !is.null(result[[name]]) && nzchar(result[[name]])
  }, logical(1))]
  if (length(missing)) reference_cell_state_v2_abort("Missing arguments: %s", paste(missing, collapse = ", "))
  if (!result$training_stage %in% c("initial", "final")) {
    reference_cell_state_v2_abort("--training-stage must be initial or final")
  }
  result
}

normalize_input <- function(path, label, directory = FALSE) {
  if (!grepl("^/", path)) reference_cell_state_v2_abort("%s must be absolute", label)
  path <- normalizePath(path, winslash = "/", mustWork = TRUE)
  if (directory && !dir.exists(path)) reference_cell_state_v2_abort("%s is not a directory", label)
  if (!directory && (!file.exists(path) || dir.exists(path))) reference_cell_state_v2_abort("%s is not a file", label)
  path
}

normalize_output <- function(path, label) {
  reference_cell_state_v2_normalize_output(path, label)
}

inside <- function(path, root) identical(path, root) || startsWith(path, paste0(root, "/"))

install_receipt_mirror <- function(authoritative, mirror) {
  if (file.exists(mirror)) {
    if (!identical(reference_cell_state_v2_sha256_file(authoritative),
                   reference_cell_state_v2_sha256_file(mirror))) {
      reference_cell_state_v2_abort("External training receipt differs from authoritative model-generation receipt")
    }
    return(invisible(TRUE))
  }
  dir.create(dirname(mirror), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(paste0(".", basename(mirror), "-"), tmpdir = dirname(mirror))
  on.exit(unlink(temporary), add = TRUE)
  if (!file.copy(authoritative, temporary, overwrite = FALSE) ||
      !identical(reference_cell_state_v2_sha256_file(authoritative),
                 reference_cell_state_v2_sha256_file(temporary)) ||
      !file.rename(temporary, mirror)) {
    reference_cell_state_v2_abort("Could not install verified external training-receipt mirror")
  }
  invisible(TRUE)
}

verify_existing_training <- function(output_dir, receipt_path, args, feature_config_path,
                                     project_path, reviewed_path, lock, api,
                                     implementation_path, shared_implementation_path) {
  authoritative <- file.path(output_dir, "training_receipt.json")
  if (!file.exists(authoritative) || dir.exists(authoritative)) {
    reference_cell_state_v2_abort("Existing model generation lacks authoritative training_receipt.json")
  }
  receipt <- jsonlite::fromJSON(authoritative, simplifyVector = FALSE)
  if (!identical(as.character(receipt$schema_version), SCHEMA_VERSION) ||
      !identical(as.character(receipt$status), "COMPLETE") ||
      !identical(as.character(receipt$training_stage), args$training_stage) ||
      !identical(normalizePath(as.character(receipt$model_dir), mustWork = TRUE), output_dir) ||
      !identical(normalizePath(as.character(receipt$project), mustWork = TRUE), project_path) ||
      !identical(normalizePath(as.character(receipt$reviewed_labels), mustWork = TRUE), reviewed_path) ||
      !identical(as.character(receipt$feature_config_sha256),
                 reference_cell_state_v2_sha256_file(feature_config_path)) ||
      !identical(as.character(receipt$reviewed_labels_sha256),
                 reference_cell_state_v2_sha256_file(reviewed_path)) ||
      !identical(as.character(receipt$dependency$lock_sha256),
                 reference_cell_state_v2_sha256_file(lock)) ||
      !identical(normalizePath(as.character(receipt$implementation), mustWork = TRUE),
                 implementation_path) ||
      !identical(as.character(receipt$implementation_sha256),
                 reference_cell_state_v2_sha256_file(implementation_path)) ||
      !identical(normalizePath(as.character(receipt$shared_implementation), mustWork = TRUE),
                 shared_implementation_path) ||
      !identical(as.character(receipt$shared_implementation_sha256),
                 reference_cell_state_v2_sha256_file(shared_implementation_path))) {
    reference_cell_state_v2_abort("Existing historical training generation identity changed")
  }
  native <- api$morphology_cell_state_classifier_output_paths(output_dir)
  paths <- c(stats::setNames(native$path, native$role), c(
    manual_training_rows = file.path(output_dir, "morphology_cell_state_classifier_uncalibrated_manual_training_rows.tsv"),
    grouped_cv_source_fold_manifest = file.path(output_dir, "morphology_cell_state_classifier_uncalibrated_grouped_cv_source_fold_manifest.tsv"),
    grouped_cv_fit_manifest = file.path(output_dir, "morphology_cell_state_classifier_uncalibrated_grouped_cv_fit_manifest.tsv"),
    loco_status = file.path(output_dir, "morphology_cell_state_classifier_uncalibrated_loco_status.tsv")
  ))
  expected <- unlist(receipt$output_file_sha256, use.names = TRUE)
  if (!setequal(names(paths), names(expected)) || any(!file.exists(paths)) ||
      any(vapply(names(paths), function(role) {
        !identical(reference_cell_state_v2_sha256_file(paths[[role]]),
                   as.character(expected[[role]]))
      }, logical(1)))) {
    reference_cell_state_v2_abort("Existing historical model artifacts differ from authoritative receipt")
  }
  install_receipt_mirror(authoritative, receipt_path)
  receipt
}

validate_feature_config <- function(path, project) {
  config <- jsonlite::fromJSON(path, simplifyVector = TRUE)
  if (!identical(as.character(config$schema_version), "reference_cell_state_feature_config_v2") ||
      !identical(as.character(config$classifier_feature_columns),
                 reference_cell_state_v2_current_features())) {
    reference_cell_state_v2_abort("V2 config must freeze the exact promoted 12-feature classifier allowlist")
  }
  if (!identical(as.character(project$classifier$feature_columns),
                 reference_cell_state_v2_current_features())) {
    reference_cell_state_v2_abort("Project classifier is not the exact V2 promoted 12-feature contract")
  }
  config
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  root <- normalize_input(args$reference_root, "--reference-root", directory = TRUE)
  lock <- normalize_input(args$dependency_lock, "--dependency-lock")
  feature_config_path <- normalize_input(args$feature_config, "--feature-config")
  project_path <- normalize_input(args$project, "--project")
  reviewed_path <- normalize_input(args$reviewed_labels, "--reviewed-labels")
  output_dir <- normalize_output(args$output_dir, "--output-dir")
  receipt_path <- normalize_output(args$output_receipt, "--output-receipt")
  project_dir <- dirname(project_path)
  shadow_root <- dirname(dirname(project_dir))
  if (!inside(reviewed_path, shadow_root) || !inside(output_dir, shadow_root) ||
      !inside(receipt_path, shadow_root) || !inside(project_path, shadow_root)) {
    reference_cell_state_v2_abort("Training inputs/outputs must remain inside one V2 shadow root")
  }
  if (inside(output_dir, root) || inside(receipt_path, root)) {
    reference_cell_state_v2_abort("Pinned reference checkout is read-only")
  }
  if (file.exists(output_dir) && !dir.exists(output_dir)) {
    reference_cell_state_v2_abort("Historical model output path exists but is not a directory")
  }
  if (!dir.exists(output_dir) && file.exists(receipt_path)) {
    reference_cell_state_v2_abort("External training receipt exists without its model generation")
  }
  dependency <- reference_cell_state_v2_verify_dependency(root, lock)
  loaded <- reference_cell_state_v2_load_historical_classifier(root)
  api <- loaded$api
  project <- reference_cell_state_v2_read_project(project_path)
  feature_config <- validate_feature_config(feature_config_path, project)
  if (dir.exists(output_dir)) {
    existing <- verify_existing_training(
      output_dir, receipt_path, args, feature_config_path, project_path,
      reviewed_path, lock, api, script_path, shared_script_path
    )
    writeLines(sprintf(
      "reference_cell_state_historical_train_verified_reuse=1 stage=%s model_id=%s receipt=%s",
      args$training_stage, as.character(existing$model_id), receipt_path
    ))
    return(invisible(0L))
  }
  features_path <- normalize_input(file.path(project_dir, project$features_file), "project features")
  cells_path <- normalize_input(file.path(project_dir, project$cells_file), "project cells")
  features <- reference_cell_state_v2_read_table(features_path, "project features")
  cells <- reference_cell_state_v2_read_table(cells_path, "project cells")
  plate_contract <- reference_cell_state_v2_resolve_plate_contract(project_path)
  mapped <- reference_cell_state_v2_map_features(features, cells, plate_contract)
  historical_config <- api$morphology_cell_state_classifier_feature_config(
    "promoted_shape_plus_rfs_boundary"
  )
  if (!identical(as.character(historical_config$feature_columns),
                 reference_cell_state_v2_historical_features())) {
    reference_cell_state_v2_abort("Pinned historical promoted feature allowlist changed")
  }
  reviews <- reference_cell_state_v2_read_table(reviewed_path, "reviewed labels")
  reference_cell_state_v2_require_columns(
    reviews, c("morphology_umap_row_key", "label"), "reviewed labels"
  )
  if (anyDuplicated(reviews$morphology_umap_row_key)) {
    reference_cell_state_v2_abort("Reviewed labels contain unresolved duplicate stable IDs")
  }
  prepared <- api$prepare_morphology_cell_state_manual_labels(reviews)
  if (!prepared$audit$n_training_eligible[[1L]]) {
    reference_cell_state_v2_abort("No manually reviewed training labels are eligible")
  }
  eligible <- prepared$labels$training_eligible
  if (any(!prepared$labels$manual_review_evidence[eligible]) ||
      any(!prepared$labels$manual_label[eligible] %in% reference_cell_state_v2_class_ids())) {
    reference_cell_state_v2_abort("Pseudo/UMAP/cluster suggestions entered the eligible manual-label set")
  }
  if (!setequal(unique(prepared$labels$manual_label[eligible]),
                reference_cell_state_v2_class_ids())) {
    reference_cell_state_v2_abort("Manual training labels must contain all three fixed ontology classes")
  }
  bf_equivalence <- reference_cell_state_v2_validate_bf_equivalence(api)
  metadata <- list(
    adapter_schema_version = SCHEMA_VERSION,
    training_stage = args$training_stage,
    current_feature_config_sha256 = reference_cell_state_v2_sha256_file(feature_config_path),
    reviewed_labels_sha256 = reference_cell_state_v2_sha256_file(reviewed_path),
    frozen_well_split_sha256 = plate_contract$sha256,
    feature_name_mapping = paste(
      names(reference_cell_state_v2_feature_map()),
      reference_cell_state_v2_feature_map(), sep = "->", collapse = ";"
    ),
    shape_unit_adaptation = paste(
      "pixel_units_constant_scale_adaptation;historical_names_dispatch_reference_allowlist;",
      "no_physical_micrometre_claim"
    ),
    bf_equivalence_max_abs_error = bf_equivalence$max_abs_error,
    context_key_policy = plate_contract$context_policy,
    source_id_policy = plate_contract$source_group_policy,
    suffix_policy = plate_contract$suffix_policy
  )
  workflow <- api$train_morphology_cell_state_classifier_workflow(
    review_labels = reviews,
    feature_rows = mapped,
    seed = 1L,
    folds = 5L,
    alpha = 1,
    lambda_rule = "lambda.1se",
    engine = "glmnet",
    feature_profile = "promoted_shape_plus_rfs_boundary",
    metadata = metadata,
    legacy_mode = TRUE
  )
  if (!identical(workflow$model$classifier$engine, "glmnet") ||
      !identical(workflow$model$model_spec$requested_engine, "glmnet") ||
      !identical(workflow$model$model_spec$source_group_column, "source_id") ||
      !identical(workflow$model$calibration_status, "uncalibrated_stratified_review_sample") ||
      !identical(workflow$model$probability_interpretation,
                 "ranking_and_classification_diagnostics_only_not_population_prevalence")) {
    reference_cell_state_v2_abort("Historical workflow returned a non-frozen classifier contract")
  }
  if (!identical(sort(unique(mapped$cell_line_family)), "SUM-159-NLS")) {
    reference_cell_state_v2_abort("SUM159 V2 must contain one collapsed cell-line family")
  }
  api$validate_morphology_cell_state_classifier_workflow(workflow)
  dir.create(dirname(output_dir), recursive = TRUE, showWarnings = FALSE)
  staging_dir <- tempfile(paste0(".", basename(output_dir), "-staging-"), tmpdir = dirname(output_dir))
  if (!dir.create(staging_dir)) reference_cell_state_v2_abort("Could not create historical training staging directory")
  on.exit(unlink(staging_dir, recursive = TRUE, force = TRUE), add = TRUE)
  api$write_morphology_cell_state_classifier_outputs(
    workflow, staging_dir, overwrite = FALSE
  )
  prefix <- file.path(staging_dir, "morphology_cell_state_classifier_uncalibrated")
  extra_paths <- c(
    manual_training_rows = paste0(prefix, "_manual_training_rows.tsv"),
    grouped_cv_source_fold_manifest = paste0(prefix, "_grouped_cv_source_fold_manifest.tsv"),
    grouped_cv_fit_manifest = paste0(prefix, "_grouped_cv_fit_manifest.tsv"),
    loco_status = paste0(prefix, "_loco_status.tsv")
  )
  reference_cell_state_v2_write_tsv_atomic(workflow$manual_training_rows, extra_paths[["manual_training_rows"]])
  reference_cell_state_v2_write_tsv_atomic(
    workflow$model$grouped_cv_source_fold_manifest,
    extra_paths[["grouped_cv_source_fold_manifest"]]
  )
  reference_cell_state_v2_write_tsv_atomic(
    workflow$model$grouped_cv_fit_manifest, extra_paths[["grouped_cv_fit_manifest"]]
  )
  loco <- data.frame(
    cell_line_family = "SUM-159-NLS", status = "NOT_APPLICABLE",
    reason = "single_cell_line_family_dataset",
    native_reference_status = paste(unique(workflow$model$leave_one_cell_line_family_out_status$status), collapse = ";"),
    native_reference_reason = paste(unique(workflow$model$leave_one_cell_line_family_out_status$reason), collapse = ";"),
    stringsAsFactors = FALSE
  )
  reference_cell_state_v2_write_tsv_atomic(loco, extra_paths[["loco_status"]])

  native_paths <- api$morphology_cell_state_classifier_output_paths(staging_dir)
  all_paths <- c(stats::setNames(native_paths$path, native_paths$role), extra_paths)
  model_stage_path <- all_paths[["model_rds"]]
  model_path <- file.path(output_dir, basename(model_stage_path))
  model_id <- paste0("reference_cell_state_historical_", substr(workflow$model$model_sha256, 1L, 16L))
  receipt <- list(
    schema_version = SCHEMA_VERSION, status = "COMPLETE",
    training_stage = args$training_stage,
    trainer = "pinned_historical_morphology_cell_state_classifier",
    generic_cpa_trainer_used = FALSE,
    shadow_root = shadow_root,
    project = project_path, project_sha256 = reference_cell_state_v2_sha256_file(project_path),
    feature_config = feature_config_path,
    feature_config_sha256 = reference_cell_state_v2_sha256_file(feature_config_path),
    reviewed_labels = reviewed_path,
    reviewed_labels_sha256 = reference_cell_state_v2_sha256_file(reviewed_path),
    parent_import_manifest = plate_contract$parent_import_path,
    parent_import_manifest_sha256 = plate_contract$parent_import_sha256,
    frozen_well_split_manifest = plate_contract$path,
    frozen_well_split_manifest_sha256 = plate_contract$sha256,
    model_dir = output_dir, model_path = model_path,
    model_id = model_id,
    model_sha256 = workflow$model$model_sha256,
    model_rds_sha256 = reference_cell_state_v2_sha256_file(model_stage_path),
    model_spec_id = workflow$model$model_spec$model_spec_id,
    model_spec_sha256 = workflow$model$model_spec_sha256,
    class_ids = as.list(reference_cell_state_v2_class_ids()),
    current_feature_columns = as.list(reference_cell_state_v2_current_features()),
    historical_feature_columns = as.list(reference_cell_state_v2_historical_features()),
    feature_profile = "promoted_shape_plus_rfs_boundary",
    feature_mapping = as.list(reference_cell_state_v2_feature_map()),
    unit_adaptation = "pixel_units_constant_scale_adaptation_not_physical_micrometre_claim",
    unit_adaptation_contract = list(
      status = "constant_unit_schema_adaptation_only",
      current_shape_units = "pixels_and_pixel_squared",
      historical_dispatch_names = "micrometre_named_fields",
      valid_microscopy_calibration = FALSE,
      physical_micron_parity_claimed = FALSE,
      fold_local_standardization_note = paste(
        "constant_positive_scaling_is_removed_for_linear_dispatch_fields_within_each_training_fold;",
        "historical_log1p_dispatch_fields_retain_an_explicit_unknown_scale_caveat"
      )
    ),
    bf_feature_equivalence = bf_equivalence,
    grouping = list(
      context_key = "plate_ploidy_context_SUM-159-NLS-2N_or_4N",
      source_id = "well_independent_group",
      suffix = "site_plus_elapsed_time_nested_not_independent",
      outer_folds = 5L, inner_folds = 5L
    ),
    glmnet = list(family = "multinomial", alpha = 1, lambda_rule = "lambda.1se",
                  type_measure = "deviance", standardize = FALSE),
    loco = list(status = "NOT_APPLICABLE", reason = "single_cell_line_family_dataset"),
    probability = list(
      calibration_status = workflow$model$calibration_status,
      interpretation = workflow$model$probability_interpretation
    ),
    label_contract = list(
      target_column = "label", manual_evidence_required = TRUE,
      confidence_policy = "historical_high_or_medium_blank_as_high_low_excluded",
      pseudo_label_training = FALSE, umap_label_training = FALSE,
      diagnostic_cluster_training = FALSE
    ),
    dependency = dependency,
    historical_source_identity = loaded$identity,
    implementation = script_path,
    implementation_sha256 = reference_cell_state_v2_sha256_file(script_path),
    shared_implementation = shared_script_path,
    shared_implementation_sha256 = reference_cell_state_v2_sha256_file(shared_script_path),
    output_file_sha256 = as.list(vapply(all_paths, reference_cell_state_v2_sha256_file, character(1)))
  )
  authoritative_staging <- file.path(staging_dir, "training_receipt.json")
  reference_cell_state_v2_write_json_atomic(receipt, authoritative_staging)
  if (file.exists(output_dir) || dir.exists(output_dir) || file.exists(receipt_path)) {
    reference_cell_state_v2_abort("Historical training output appeared during staging")
  }
  if (!file.rename(staging_dir, output_dir)) {
    reference_cell_state_v2_abort("Failed to install historical model generation atomically")
  }
  install_receipt_mirror(file.path(output_dir, "training_receipt.json"), receipt_path)
  writeLines(sprintf(
    "reference_cell_state_historical_train_complete=1 stage=%s rows=%d sources=%d model_id=%s receipt=%s",
    args$training_stage, nrow(workflow$manual_training_rows),
    length(unique(workflow$manual_training_rows$source_id)), model_id, receipt_path
  ))
}

status <- tryCatch(main(), error = function(error) {
  writeLines(conditionMessage(error), con = stderr())
  1L
})
quit(save = "no", status = as.integer(status), runLast = FALSE)
