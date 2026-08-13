#!/usr/bin/env Rscript

# Authoritatively import the JSON exported by script 37. The exact preselected
# stable-ID universe and crop-render generation are revalidated before writing
# reviewed_labels.tsv for historical training.

SCHEMA_VERSION <- "reference_cell_state_exact_review_import_v2"
SUBMISSION_SCHEMA <- "reference_cell_state_exact_review_submission_v2"
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
  allowed <- c("--reference-root", "--dependency-lock", "--review-set", "--render-dir", "--submission", "--output-dir")
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

reject_symlink_inside <- function(path, root, label) {
  relative <- if (identical(path, root)) "" else substring(path, nchar(root) + 2L)
  parts <- if (nzchar(relative)) strsplit(relative, "/", fixed = TRUE)[[1L]] else character()
  current <- root
  root_link <- Sys.readlink(root)
  if (length(root_link) == 1L && !is.na(root_link) && nzchar(root_link)) {
    reference_cell_state_v2_abort("REFERENCE_SHADOW_ROOT is a symlink: %s", root)
  }
  for (part in parts) {
    if (!nzchar(part) || identical(part, ".")) next
    if (identical(part, "..")) reference_cell_state_v2_abort("%s may not contain ..", label)
    current <- file.path(current, part)
    link <- Sys.readlink(current)
    if (length(link) == 1L && !is.na(link) && nzchar(link)) {
      reference_cell_state_v2_abort("%s contains a symlink path component: %s", label, current)
    }
  }
  invisible(TRUE)
}

norm <- function(path, label, directory = FALSE, output = FALSE) {
  if (!grepl("^/", path)) reference_cell_state_v2_abort("%s must be absolute", label)
  if (isTRUE(output)) {
    lexical <- path
    ancestor <- lexical
    suffix <- character()
    while (!file.exists(ancestor) && !dir.exists(ancestor)) {
      parent <- dirname(ancestor)
      if (identical(parent, ancestor)) reference_cell_state_v2_abort("%s has no existing ancestor", label)
      suffix <- c(basename(ancestor), suffix)
      ancestor <- parent
    }
    path <- do.call(file.path, as.list(c(
      normalizePath(ancestor, winslash = "/", mustWork = TRUE), suffix
    )))
  } else {
    path <- normalizePath(path, winslash = "/", mustWork = TRUE)
  }
  if (!output && directory && !dir.exists(path)) reference_cell_state_v2_abort("%s is not a directory", label)
  if (!output && !directory && (!file.exists(path) || dir.exists(path))) reference_cell_state_v2_abort("%s is not a file", label)
  path
}

inside <- function(path, root) identical(path, root) || startsWith(path, paste0(root, "/"))

require_inside <- function(path, root, label) {
  if (!inside(path, root)) reference_cell_state_v2_abort("%s must remain inside REFERENCE_SHADOW_ROOT", label)
  reject_symlink_inside(path, root, label)
  path
}

valid_timestamp <- function(values) {
  grepl(
    "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\\.[0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})$",
    as.character(values)
  )
}

is_symlink <- function(path) {
  link <- Sys.readlink(path)
  length(link) == 1L && !is.na(link) && nzchar(link)
}

verify_hashed_inputs <- function(render_manifest, shadow) {
  expected_roles <- c("project", "cells", "images", "review_set", "review_manifest")
  inputs <- render_manifest$inputs
  if (!is.list(inputs) || !setequal(names(inputs), expected_roles)) {
    reference_cell_state_v2_abort("Render input hash set is incomplete or unexpected")
  }
  paths <- list()
  for (role in expected_roles) {
    item <- inputs[[role]]
    path <- require_inside(norm(as.character(item$path), paste("render input", role)), shadow, paste("render input", role))
    if (!identical(reference_cell_state_v2_sha256_file(path), as.character(item$sha256))) {
      reference_cell_state_v2_abort("Render input hash mismatch: %s", role)
    }
    paths[[role]] <- path
  }
  paths
}

verify_crop_generation <- function(render_dir, render_manifest, stable_ids) {
  artifact_names <- c(
    exact_review_html = "exact_review.html",
    crop_render_status = "crop_render_status.tsv",
    crop_manifest = "crop_manifest.tsv"
  )
  hashes <- unlist(render_manifest$artifact_file_sha256, use.names = TRUE)
  if (!setequal(names(hashes), names(artifact_names))) {
    reference_cell_state_v2_abort("Render artifact hash set is incomplete or unexpected")
  }
  for (role in names(artifact_names)) {
    path <- norm(file.path(render_dir, artifact_names[[role]]), paste("render artifact", role))
    if (!identical(reference_cell_state_v2_sha256_file(path), as.character(hashes[[role]]))) {
      reference_cell_state_v2_abort("Render artifact hash mismatch: %s", role)
    }
  }
  crop_manifest_path <- file.path(render_dir, "crop_manifest.tsv")
  crops <- reference_cell_state_v2_read_table(crop_manifest_path, "crop manifest")
  expected_columns <- c("morphology_umap_row_key", "channel", "relative_path", "sha256")
  if (!identical(names(crops), expected_columns) || nrow(crops) != 2L * length(stable_ids) ||
      anyDuplicated(paste(crops$morphology_umap_row_key, crops$channel, sep = "|")) ||
      !setequal(crops$morphology_umap_row_key, stable_ids) ||
      !setequal(crops$channel, c("brightfield", "nuclei_support"))) {
    reference_cell_state_v2_abort("Crop manifest does not cover each stable ID/channel exactly once")
  }
  for (i in seq_len(nrow(crops))) {
    relative <- as.character(crops$relative_path[[i]])
    if (!grepl("^crops/[^/]+[.]png$", relative) || grepl("[.][.]", relative, fixed = TRUE)) {
      reference_cell_state_v2_abort("Unsafe crop relative path: %s", relative)
    }
    path <- require_inside(norm(file.path(render_dir, relative), "crop artifact"), render_dir, "crop artifact")
    if (!identical(reference_cell_state_v2_sha256_file(path), as.character(crops$sha256[[i]]))) {
      reference_cell_state_v2_abort("Crop artifact hash mismatch: %s", relative)
    }
  }
  aggregate <- reference_cell_state_v2_sha256_file(crop_manifest_path)
  if (!identical(aggregate, as.character(render_manifest$crop_aggregate_sha256))) {
    reference_cell_state_v2_abort("Crop aggregate SHA mismatch")
  }
  status <- reference_cell_state_v2_read_table(file.path(render_dir, "crop_render_status.tsv"), "crop render status")
  reference_cell_state_v2_require_columns(
    status,
    c("morphology_umap_row_key", "brightfield_status", "nuclei_status", "combined_mask_status"),
    "crop render status"
  )
  if (nrow(status) != length(stable_ids) || anyDuplicated(status$morphology_umap_row_key) ||
      !setequal(status$morphology_umap_row_key, stable_ids) ||
      any(status$brightfield_status != "ok") || any(status$nuclei_status != "ok") ||
      any(status$combined_mask_status != "ok")) {
    reference_cell_state_v2_abort("Crop render status is not complete for the exact stable-ID universe")
  }
  invisible(TRUE)
}

verify_existing_import <- function(output, expected) {
  manifest_path <- norm(file.path(output, "review_import_manifest.json"), "existing review import manifest")
  manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
  scalar <- c(
    "schema_version", "status", "review_id", "render_generation_id",
    "crop_manifest_sha256", "renderer_implementation_sha256", "review_set_sha256",
    "render_manifest_sha256", "submission_sha256", "row_count", "stable_id_sha256",
    "implementation", "implementation_sha256", "shared_implementation",
    "shared_implementation_sha256"
  )
  for (field in scalar) {
    if (!identical(as.character(manifest[[field]]), as.character(expected[[field]]))) {
      reference_cell_state_v2_abort("Existing review import identity differs for %s", field)
    }
  }
  hashes <- unlist(manifest$output_file_sha256, use.names = TRUE)
  files <- c(reviewed_labels = "reviewed_labels.tsv", review_import_audit = "review_import_audit.tsv")
  if (!setequal(names(hashes), names(files))) reference_cell_state_v2_abort("Existing import output hash set changed")
  for (role in names(files)) {
    path <- norm(file.path(output, files[[role]]), paste("existing import", role))
    if (!identical(reference_cell_state_v2_sha256_file(path), as.character(hashes[[role]]))) {
      reference_cell_state_v2_abort("Existing import output hash mismatch: %s", role)
    }
  }
  invisible(TRUE)
}

stable_sha <- function(ids) reference_cell_state_v2_sha256_text(sort(as.character(ids), method = "radix"))

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  root <- norm(args$reference_root, "--reference-root", directory = TRUE)
  lock <- norm(args$dependency_lock, "--dependency-lock")
  render_dir <- norm(args$render_dir, "--render-dir", directory = TRUE)
  dependency <- reference_cell_state_v2_verify_dependency(root, lock)
  loaded <- reference_cell_state_v2_load_historical_classifier(root)
  api <- loaded$api
  render_manifest_path <- norm(file.path(render_dir, "exact_review_render_manifest.json"), "render manifest")
  render_manifest <- jsonlite::fromJSON(render_manifest_path, simplifyVector = FALSE)
  if (!identical(as.character(render_manifest$schema_version), "reference_cell_state_exact_review_render_v2") ||
      !isTRUE(render_manifest$exact_selection_preserved) || isTRUE(render_manifest$secondary_sampling_performed) ||
      !isTRUE(render_manifest$all_crops_available)) {
    reference_cell_state_v2_abort("Render generation is not a complete exact-selection review")
  }
  shadow <- norm(as.character(render_manifest$reference_shadow_root), "render REFERENCE_SHADOW_ROOT", directory = TRUE)
  require_inside(render_dir, shadow, "render directory")
  inputs <- verify_hashed_inputs(render_manifest, shadow)
  review_path <- require_inside(norm(args$review_set, "--review-set"), shadow, "review set")
  submission_path <- require_inside(norm(args$submission, "--submission"), shadow, "submission")
  output <- require_inside(norm(args$output_dir, "--output-dir", output = TRUE), shadow, "import output")
  if (!identical(review_path, inputs$review_set) ||
      !identical(normalizePath(render_manifest$review_set, winslash = "/", mustWork = TRUE), review_path) ||
      !identical(normalizePath(render_manifest$review_manifest, winslash = "/", mustWork = TRUE), inputs$review_manifest)) {
    reference_cell_state_v2_abort("Render/review input paths disagree")
  }
  review <- reference_cell_state_v2_read_table(review_path, "exact review set")
  reference_cell_state_v2_require_columns(review, "morphology_umap_row_key", "exact review set")
  stable_ids <- as.character(review$morphology_umap_row_key)
  if (anyDuplicated(stable_ids) || any(!nzchar(stable_ids))) {
    reference_cell_state_v2_abort("Exact review set has blank or duplicate stable IDs")
  }
  verify_crop_generation(render_dir, render_manifest, stable_ids)
  submission <- jsonlite::fromJSON(submission_path, simplifyVector = FALSE)
  if (!identical(as.character(submission$schema_version), SUBMISSION_SCHEMA) || !is.list(submission$rows)) {
    reference_cell_state_v2_abort("Unsupported exact-review submission")
  }
  identity <- submission$identity
  expected <- render_manifest$identity
  for (field in c(
    "review_id", "project_sha256", "review_set_sha256", "review_manifest_sha256",
    "stable_id_sha256", "row_count", "crop_manifest_sha256",
    "renderer_implementation_sha256", "render_padding", "render_generation_id"
  )) {
    if (!identical(as.character(identity[[field]]), as.character(expected[[field]]))) {
      reference_cell_state_v2_abort("Submission/render identity mismatch: %s", field)
    }
  }
  if (!identical(as.character(expected$crop_manifest_sha256),
                 reference_cell_state_v2_sha256_file(file.path(render_dir, "crop_manifest.tsv"))) ||
      !identical(as.character(expected$renderer_implementation_sha256),
                 as.character(render_manifest$renderer_implementation_sha256)) ||
      !identical(as.character(expected$render_padding),
                 as.character(render_manifest$render_padding))) {
    reference_cell_state_v2_abort("Render-generation identity is not bound to its crops/implementation")
  }
  renderer_implementation <- norm(
    as.character(render_manifest$renderer_implementation), "renderer implementation"
  )
  if (!identical(reference_cell_state_v2_sha256_file(renderer_implementation),
                 as.character(render_manifest$renderer_implementation_sha256))) {
    reference_cell_state_v2_abort("Renderer implementation changed after the crop generation")
  }
  if (!identical(reference_cell_state_v2_sha256_file(review_path), as.character(expected$review_set_sha256))) {
    reference_cell_state_v2_abort("Review set changed after rendering")
  }
  creation <- submission$creation_metadata
  if (!is.list(creation) || !valid_timestamp(as.character(creation$created_at)) ||
      !nzchar(trimws(as.character(creation$client_version)))) {
    reference_cell_state_v2_abort("Submission creation metadata is incomplete or malformed")
  }
  required_row_fields <- c(
    "morphology_umap_row_key", "label", "label_confidence", "reviewer", "reviewed_at", "review_notes"
  )
  if (length(submission$rows) != nrow(review) || any(vapply(submission$rows, function(row) {
    !is.list(row) || !all(required_row_fields %in% names(row)) ||
      any(vapply(row[required_row_fields], length, integer(1)) != 1L)
  }, logical(1)))) {
    reference_cell_state_v2_abort("Submission rows are incomplete")
  }
  rows <- do.call(rbind, lapply(submission$rows, function(row) {
    data.frame(
      morphology_umap_row_key = as.character(row$morphology_umap_row_key),
      label = as.character(row$label), label_confidence = as.character(row$label_confidence),
      reviewer = as.character(row$reviewer), reviewed_at = as.character(row$reviewed_at),
      review_notes = as.character(row$review_notes), stringsAsFactors = FALSE
    )
  }))
  if (!is.data.frame(rows) || nrow(rows) != nrow(review) || anyDuplicated(rows$morphology_umap_row_key) ||
      !setequal(rows$morphology_umap_row_key, review$morphology_umap_row_key) ||
      !identical(stable_sha(rows$morphology_umap_row_key), as.character(expected$stable_id_sha256))) {
    reference_cell_state_v2_abort("Submission does not cover the exact review stable-ID universe")
  }
  evidence_fields <- c(
    "morphology_umap_row_key", "label", "label_confidence", "reviewer",
    "reviewed_at", "review_notes"
  )
  if (any(!rows$label %in% c(reference_cell_state_v2_class_ids(), "uncertain")) ||
      any(!rows$label_confidence %in% c("high", "medium", "low")) ||
      any(!nzchar(rows$reviewer)) || any(nchar(rows$reviewer) > 200L) ||
      any(vapply(evidence_fields, function(field) {
        anyNA(rows[[field]]) || any(grepl("[[:cntrl:]]", as.character(rows[[field]])))
      }, logical(1))) ||
      any(nchar(rows$review_notes) > 4000L) ||
      any(rows$reviewer != trimws(rows$reviewer)) ||
      any(!valid_timestamp(rows$reviewed_at))) {
    reference_cell_state_v2_abort("Submission contains invalid label/confidence or incomplete human evidence")
  }
  reviewed <- review[match(rows$morphology_umap_row_key, review$morphology_umap_row_key), , drop = FALSE]
  for (column in setdiff(names(rows), "morphology_umap_row_key")) reviewed[[column]] <- rows[[column]]
  reviewed$review_changed_from_default <- reviewed$label != reviewed$review_default_label
  reviewed$manual_review_complete <- TRUE
  prepared <- api$prepare_morphology_cell_state_manual_labels(reviewed)
  if (any(!prepared$labels$manual_review_evidence)) reference_cell_state_v2_abort("Imported rows lack manual evidence")
  expected_existing <- list(
    schema_version = SCHEMA_VERSION, status = "COMPLETE",
    review_id = as.character(expected$review_id),
    render_generation_id = as.character(expected$render_generation_id),
    crop_manifest_sha256 = as.character(expected$crop_manifest_sha256),
    renderer_implementation_sha256 = as.character(expected$renderer_implementation_sha256),
    review_set_sha256 = reference_cell_state_v2_sha256_file(review_path),
    render_manifest_sha256 = reference_cell_state_v2_sha256_file(render_manifest_path),
    submission_sha256 = reference_cell_state_v2_sha256_file(submission_path),
    row_count = nrow(reviewed), stable_id_sha256 = stable_sha(reviewed$morphology_umap_row_key),
    implementation = script_path,
    implementation_sha256 = reference_cell_state_v2_sha256_file(script_path),
    shared_implementation = shared_script_path,
    shared_implementation_sha256 = reference_cell_state_v2_sha256_file(shared_script_path)
  )
  if (dir.exists(output)) {
    verify_existing_import(output, expected_existing)
    writeLines(sprintf("reference_cell_state_exact_review_import_verified_reuse=1 rows=%d labels=%s",
                       nrow(reviewed), file.path(output, "reviewed_labels.tsv")))
    return(invisible(0L))
  }
  if (file.exists(output) || is_symlink(output)) {
    reference_cell_state_v2_abort("Import output exists but is not a regular immutable generation directory")
  }
  dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
  staging <- tempfile(paste0(".", basename(output), "-staging-"), tmpdir = dirname(output))
  if (!dir.create(staging)) reference_cell_state_v2_abort("Could not create import staging directory")
  on.exit(unlink(staging, recursive = TRUE, force = TRUE), add = TRUE)
  labels_path <- file.path(staging, "reviewed_labels.tsv")
  audit_path <- file.path(staging, "review_import_audit.tsv")
  reference_cell_state_v2_write_tsv_atomic(reviewed, labels_path)
  audit <- cbind(data.frame(
    row_count = nrow(reviewed), exact_selection_preserved = TRUE,
    all_rows_manually_reviewed = TRUE, all_crops_available = TRUE,
    pseudo_label_training = FALSE, diagnostic_cluster_training = FALSE,
    stringsAsFactors = FALSE
  ), prepared$audit)
  reference_cell_state_v2_write_tsv_atomic(audit, audit_path)
  output_hashes <- c(
    reviewed_labels = reference_cell_state_v2_sha256_file(labels_path),
    review_import_audit = reference_cell_state_v2_sha256_file(audit_path)
  )
  manifest <- c(expected_existing, list(
    review_set = review_path, review_set_sha256 = reference_cell_state_v2_sha256_file(review_path),
    render_manifest = render_manifest_path,
    render_manifest_sha256 = reference_cell_state_v2_sha256_file(render_manifest_path),
    submission = submission_path, submission_sha256 = reference_cell_state_v2_sha256_file(submission_path),
    reviewed_labels = file.path(output, "reviewed_labels.tsv"),
    reviewed_labels_sha256 = unname(output_hashes[["reviewed_labels"]]),
    review_import_audit_sha256 = unname(output_hashes[["review_import_audit"]]),
    output_file_sha256 = as.list(output_hashes),
    exact_selection_preserved = TRUE, secondary_sampling_performed = FALSE,
    manual_review_evidence_required = TRUE,
    dependency = dependency, historical_source_identity = loaded$identity
  ))
  reference_cell_state_v2_write_json_atomic(manifest, file.path(staging, "review_import_manifest.json"))
  if (file.exists(output) || dir.exists(output) || is_symlink(output)) {
    reference_cell_state_v2_abort("Import output appeared during staging")
  }
  if (!file.rename(staging, output)) reference_cell_state_v2_abort("Failed to install import atomically")
  writeLines(sprintf("reference_cell_state_exact_review_import_complete=1 rows=%d labels=%s",
                     nrow(reviewed), file.path(output, "reviewed_labels.tsv")))
}

status <- tryCatch(main(), error = function(error) { writeLines(conditionMessage(error), stderr()); 1L })
quit(save = "no", status = as.integer(status), runLast = FALSE)
