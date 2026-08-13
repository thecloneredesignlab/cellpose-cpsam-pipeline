#!/usr/bin/env Rscript

# Accept only a final V2 model produced by the pinned historical morphology
# classifier. V1 nine-feature and generic CPA model generations are rejected.

SCHEMA_VERSION <- "reference_cell_state_model_acceptance_v2"
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
    "--shadow-root", "--project", "--model-dir", "--train-receipt",
    "--feature-config", "--classes-file", "--parent-import-manifest",
    "--reference-root", "--dependency-lock", "--output-receipt"
  )
  out <- list(); i <- 1L
  while (i <= length(arguments)) {
    arg <- arguments[[i]]
    if (!arg %in% allowed || i == length(arguments)) reference_cell_state_v2_abort("Unknown argument or missing value: %s", arg)
    name <- gsub("-", "_", substring(arg, 3L), fixed = TRUE)
    if (!is.null(out[[name]])) reference_cell_state_v2_abort("Repeated argument: %s", arg)
    out[[name]] <- arguments[[i + 1L]]; i <- i + 2L
  }
  required <- gsub("-", "_", substring(allowed, 3L), fixed = TRUE)
  missing <- required[!vapply(required, function(name) !is.null(out[[name]]) && nzchar(out[[name]]), logical(1))]
  if (length(missing)) reference_cell_state_v2_abort("Missing arguments: %s", paste(missing, collapse = ", "))
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

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  shadow <- norm(args$shadow_root, "--shadow-root", directory = TRUE)
  project <- norm(args$project, "--project")
  model_dir <- norm(args$model_dir, "--model-dir", directory = TRUE)
  train_receipt_path <- norm(args$train_receipt, "--train-receipt")
  feature_config_path <- norm(args$feature_config, "--feature-config")
  classes_path <- norm(args$classes_file, "--classes-file")
  parent_import_path <- norm(args$parent_import_manifest, "--parent-import-manifest")
  root <- norm(args$reference_root, "--reference-root", directory = TRUE)
  lock <- norm(args$dependency_lock, "--dependency-lock")
  output <- norm(args$output_receipt, "--output-receipt", output = TRUE)
  if (any(!vapply(c(project, model_dir, train_receipt_path, parent_import_path, output), inside,
                  logical(1), root = shadow))) {
    reference_cell_state_v2_abort("Project/model/train/parent/output must share one V2 shadow root")
  }
  dependency <- reference_cell_state_v2_verify_dependency(root, lock)
  loaded <- reference_cell_state_v2_load_historical_classifier(root)
  api <- loaded$api
  feature_config <- jsonlite::fromJSON(feature_config_path, simplifyVector = TRUE)
  if (!identical(as.character(feature_config$schema_version), "reference_cell_state_feature_config_v2") ||
      !identical(as.character(feature_config$classifier_feature_columns),
                 reference_cell_state_v2_current_features())) {
    reference_cell_state_v2_abort("Model acceptance requires the V2 promoted 12-feature config; V1 nine-feature models are forbidden")
  }
  classes <- utils::read.delim(classes_path, sep = "\t", header = TRUE, quote = "", comment.char = "",
                               check.names = FALSE, stringsAsFactors = FALSE)
  reference_cell_state_v2_require_columns(classes, c("class_id", "trainable"), "classes file")
  trainable <- tolower(as.character(classes$trainable)) %in% c("true", "t", "1", "yes")
  if (!identical(as.character(classes$class_id[trainable]), reference_cell_state_v2_class_ids())) {
    reference_cell_state_v2_abort("Classes file is not in exact historical live/dead/multinucleated order")
  }
  receipt <- jsonlite::fromJSON(train_receipt_path, simplifyVector = FALSE)
  authoritative_train_receipt <- norm(
    file.path(model_dir, "training_receipt.json"), "authoritative model training receipt"
  )
  if (!identical(reference_cell_state_v2_sha256_file(authoritative_train_receipt),
                 reference_cell_state_v2_sha256_file(train_receipt_path))) {
    reference_cell_state_v2_abort(
      "External train receipt is not the exact mirror of model-generation authority"
    )
  }
  if (!identical(as.character(receipt$schema_version), "reference_cell_state_historical_train_v2") ||
      !identical(as.character(receipt$status), "COMPLETE") ||
      !identical(as.character(receipt$training_stage), "final") ||
      !identical(as.character(receipt$trainer), "pinned_historical_morphology_cell_state_classifier") ||
      !identical(as.character(receipt$unit_adaptation),
                 "pixel_units_constant_scale_adaptation_not_physical_micrometre_claim") ||
      !identical(as.character(receipt$unit_adaptation_contract$valid_microscopy_calibration), "FALSE") ||
      !identical(as.character(receipt$unit_adaptation_contract$physical_micron_parity_claimed), "FALSE") ||
      isTRUE(receipt$generic_cpa_trainer_used)) {
    reference_cell_state_v2_abort("Only a final pinned historical V2 training receipt can be accepted")
  }
  if (!identical(norm(as.character(receipt$model_dir), "train receipt model_dir", directory = TRUE), model_dir) ||
      !identical(norm(as.character(receipt$project), "train receipt project"), project) ||
      !identical(norm(as.character(receipt$feature_config), "train receipt feature config"), feature_config_path) ||
      !identical(norm(as.character(receipt$parent_import_manifest), "train receipt parent import"), parent_import_path)) {
    reference_cell_state_v2_abort("Historical train receipt belongs to another V2 generation")
  }
  if (!identical(as.character(receipt$feature_config_sha256), reference_cell_state_v2_sha256_file(feature_config_path)) ||
      !identical(as.character(receipt$parent_import_manifest_sha256), reference_cell_state_v2_sha256_file(parent_import_path)) ||
      !identical(as.character(receipt$dependency$lock_sha256), dependency$lock_sha256)) {
    reference_cell_state_v2_abort("Historical train receipt frozen-input hashes changed")
  }
  training_implementation <- norm(
    as.character(receipt$implementation), "historical training implementation"
  )
  expected_training_implementation <- norm(
    file.path(dirname(script_path), "34_train_reference_cell_state_historical.R"),
    "expected historical training implementation"
  )
  training_shared_implementation <- norm(
    as.character(receipt$shared_implementation), "historical training shared implementation"
  )
  if (!identical(training_implementation, expected_training_implementation) ||
      !identical(as.character(receipt$implementation_sha256),
                 reference_cell_state_v2_sha256_file(training_implementation)) ||
      !identical(training_shared_implementation, shared_script_path) ||
      !identical(as.character(receipt$shared_implementation_sha256),
                 reference_cell_state_v2_sha256_file(shared_script_path))) {
    reference_cell_state_v2_abort("Historical train implementation identity changed")
  }
  model_path <- norm(as.character(receipt$model_path), "historical model bundle")
  if (!inside(model_path, model_dir) ||
      !identical(as.character(receipt$model_rds_sha256), reference_cell_state_v2_sha256_file(model_path))) {
    reference_cell_state_v2_abort("Historical model RDS differs from train receipt")
  }
  bundle <- readRDS(model_path)
  api$validate_morphology_cell_state_classifier_model_bundle(
    bundle, expected_feature_profile = "promoted_shape_plus_rfs_boundary"
  )
  if (!identical(as.character(bundle$feature_config$feature_columns),
                 reference_cell_state_v2_historical_features()) ||
      !identical(as.character(bundle$classifier$classes), reference_cell_state_v2_class_ids()) ||
      !identical(bundle$classifier$engine, "glmnet") ||
      !identical(bundle$classifier$requested_engine, "glmnet") ||
      !isTRUE(all.equal(as.numeric(bundle$classifier$alpha), 1)) ||
      !identical(bundle$classifier$lambda_rule, "lambda.1se") ||
      !identical(as.integer(bundle$model_spec$grouped_folds), 5L) ||
      !identical(bundle$model_spec$source_group_column, "source_id") ||
      !identical(bundle$model_spec$suffix_role, "nested_technical_metadata_not_independent_group")) {
    reference_cell_state_v2_abort("Historical model bundle violates the frozen 5x5 grouped glmnet contract")
  }
  if (!identical(bundle$calibration_status, "uncalibrated_stratified_review_sample") ||
      !identical(bundle$probability_interpretation,
                 "ranking_and_classification_diagnostics_only_not_population_prevalence")) {
    reference_cell_state_v2_abort("Historical probabilities must remain explicitly uncalibrated")
  }
  reviewed <- norm(as.character(receipt$reviewed_labels), "accepted reviewed labels")
  if (!inside(reviewed, shadow) ||
      !identical(as.character(receipt$reviewed_labels_sha256), reference_cell_state_v2_sha256_file(reviewed))) {
    reference_cell_state_v2_abort("Accepted model reviewed-label ancestry changed")
  }
  loco_path <- file.path(model_dir, "morphology_cell_state_classifier_uncalibrated_loco_status.tsv")
  loco <- utils::read.delim(loco_path, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
  if (nrow(loco) != 1L || !identical(as.character(loco$status[[1L]]), "NOT_APPLICABLE") ||
      !identical(as.character(loco$reason[[1L]]), "single_cell_line_family_dataset")) {
    reference_cell_state_v2_abort("Single-cell-line-family LOCO status must be NOT_APPLICABLE")
  }
  training_manifest <- file.path(model_dir, "morphology_cell_state_classifier_uncalibrated_training_manifest.tsv")
  acceptance <- list(
    schema_version = SCHEMA_VERSION, status = "ACCEPTED", accepted = TRUE,
    shadow_root = shadow,
    project = list(path = project, sha256 = reference_cell_state_v2_sha256_file(project)),
    model = list(
      dir = model_dir, path = model_path, model_id = as.character(receipt$model_id),
      model_sha256 = as.character(bundle$model_sha256),
      model_rds_sha256 = reference_cell_state_v2_sha256_file(model_path),
      model_spec_id = as.character(bundle$model_spec$model_spec_id),
      model_spec_sha256 = as.character(bundle$model_spec_sha256),
      manifest_sha256 = reference_cell_state_v2_sha256_file(training_manifest),
      feature_profile = "promoted_shape_plus_rfs_boundary",
      feature_columns = as.list(reference_cell_state_v2_current_features()),
      historical_feature_columns = as.list(reference_cell_state_v2_historical_features()),
      unit_adaptation = list(
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
      class_ids = as.list(reference_cell_state_v2_class_ids()),
      engine = "historical_glmnet_multinomial",
      calibration_status = bundle$calibration_status,
      probability_interpretation = bundle$probability_interpretation
    ),
    dependency = dependency,
    historical_source_identity = loaded$identity,
    train_receipt = list(path = train_receipt_path, sha256 = reference_cell_state_v2_sha256_file(train_receipt_path)),
    frozen_inputs = list(
      parent_import_manifest = parent_import_path,
      parent_import_manifest_sha256 = reference_cell_state_v2_sha256_file(parent_import_path),
      parent_shadow_root = as.character(jsonlite::fromJSON(parent_import_path, simplifyVector = FALSE)$parent$shadow_root),
      feature_config = feature_config_path,
      feature_config_sha256 = reference_cell_state_v2_sha256_file(feature_config_path),
      classes_file = classes_path, classes_file_sha256 = reference_cell_state_v2_sha256_file(classes_path),
      reviewed_labels = reviewed, reviewed_labels_sha256 = reference_cell_state_v2_sha256_file(reviewed),
      frozen_well_split_manifest = as.character(receipt$frozen_well_split_manifest),
      frozen_well_split_manifest_sha256 = as.character(receipt$frozen_well_split_manifest_sha256)
    ),
    isolation_contract = list(
      semantic_axis = "reference_cell_state", generic_cpa_classifier_used = FALSE,
      current_classification_read = FALSE, diagnostic_cluster_used_for_training = FALSE,
      umap_or_pseudo_label_used_for_training = FALSE, existing_classification_overwritten = FALSE
    ),
    implementation = script_path,
    implementation_sha256 = reference_cell_state_v2_sha256_file(script_path),
    shared_implementation = shared_script_path,
    shared_implementation_sha256 = reference_cell_state_v2_sha256_file(shared_script_path)
  )
  if (file.exists(output) || dir.exists(output) || reference_cell_state_v2_is_symlink(output)) {
    if (!file.exists(output) || dir.exists(output) || reference_cell_state_v2_is_symlink(output)) {
      reference_cell_state_v2_abort("Acceptance authority exists but is not a regular immutable receipt")
    }
    temporary <- tempfile(".acceptance-candidate-", tmpdir = dirname(output))
    on.exit(unlink(temporary), add = TRUE)
    reference_cell_state_v2_write_json_atomic(acceptance, temporary)
    if (!identical(reference_cell_state_v2_sha256_file(output),
                   reference_cell_state_v2_sha256_file(temporary))) {
      reference_cell_state_v2_abort("Existing model acceptance identity differs and was preserved")
    }
    writeLines(sprintf("reference_cell_state_model_acceptance_verified_reuse=1 model_id=%s receipt=%s",
                       acceptance$model$model_id, output))
    return(invisible(0L))
  }
  reference_cell_state_v2_write_json_atomic(acceptance, output)
  writeLines(sprintf("reference_cell_state_model_accepted=1 model_id=%s receipt=%s",
                     acceptance$model$model_id, output))
}

status <- tryCatch(main(), error = function(error) { writeLines(conditionMessage(error), stderr()); 1L })
quit(save = "no", status = as.integer(status), runLast = FALSE)
