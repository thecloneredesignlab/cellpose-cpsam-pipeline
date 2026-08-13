#!/usr/bin/env Rscript

# Merge seed-1 and seed-2 human reviews by stable ID. Any non-identical overlap
# is a hard human-adjudication barrier; no last-write-wins behavior is allowed.

SCHEMA_VERSION <- "reference_cell_state_review_merge_v2"
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
    "--reference-root", "--dependency-lock", "--seed1-reviewed-labels",
    "--seed2-reviewed-labels", "--output-dir", "--adjudication"
  )
  out <- list(adjudication = NULL); i <- 1L
  while (i <= length(arguments)) {
    arg <- arguments[[i]]
    if (!arg %in% allowed || i == length(arguments)) reference_cell_state_v2_abort("Unknown argument or missing value: %s", arg)
    name <- gsub("-", "_", substring(arg, 3L), fixed = TRUE)
    if (!is.null(out[[name]])) reference_cell_state_v2_abort("Repeated argument: %s", arg)
    out[[name]] <- arguments[[i + 1L]]; i <- i + 2L
  }
  required <- c("reference_root", "dependency_lock", "seed1_reviewed_labels", "seed2_reviewed_labels", "output_dir")
  missing <- required[!vapply(required, function(name) !is.null(out[[name]]) && nzchar(out[[name]]), logical(1))]
  if (length(missing)) reference_cell_state_v2_abort("Missing arguments: %s", paste(missing, collapse = ", "))
  out
}

inside <- function(path, root) identical(path, root) || startsWith(path, paste0(root, "/"))

stable_sha <- function(ids) reference_cell_state_v2_sha256_text(
  sort(as.character(ids), method = "radix")
)

valid_timestamp <- function(values) grepl(
  "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\\.[0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})$",
  as.character(values)
)

verify_import_parent <- function(labels_path, expected_seed, shadow) {
  manifest_path <- norm(file.path(dirname(labels_path), "review_import_manifest.json"),
                        paste0("seed", expected_seed, " import manifest"))
  manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
  reviewed <- reference_cell_state_v2_read_table(labels_path, paste0("seed", expected_seed, " reviewed labels"))
  reference_cell_state_v2_require_columns(
    reviewed,
    c("morphology_umap_row_key", "label", "label_confidence", "reviewer", "reviewed_at", "review_notes"),
    paste0("seed", expected_seed, " reviewed labels")
  )
  stable_ids <- as.character(reviewed$morphology_umap_row_key)
  expected_schema <- if (expected_seed == 1L) {
    "reference_cell_state_seed1_review_v2"
  } else "reference_cell_state_seed2_multinucleated_review_v2"
  paths <- list(
    render = norm(as.character(manifest$render_manifest), "authoritative render manifest"),
    submission = norm(as.character(manifest$submission), "authoritative review submission"),
    review_set = norm(as.character(manifest$review_set), "authoritative review set")
  )
  if (any(!vapply(paths, inside, logical(1), root = shadow))) {
    reference_cell_state_v2_abort("Seed%d human evidence chain leaves the V2 shadow root", expected_seed)
  }
  render <- jsonlite::fromJSON(paths$render, simplifyVector = FALSE)
  submission <- jsonlite::fromJSON(paths$submission, simplifyVector = FALSE)
  selection_item <- render$inputs$review_manifest
  selection_path <- norm(as.character(selection_item$path), "review selection manifest")
  if (!inside(selection_path, shadow)) {
    reference_cell_state_v2_abort("Seed%d selection manifest leaves the V2 shadow root", expected_seed)
  }
  selection <- jsonlite::fromJSON(selection_path, simplifyVector = FALSE)
  inputs <- render$inputs
  expected_input_roles <- c("project", "cells", "images", "review_set", "review_manifest")
  if (!is.list(inputs) || !setequal(names(inputs), expected_input_roles)) {
    reference_cell_state_v2_abort("Seed%d render input hash set changed", expected_seed)
  }
  for (role in expected_input_roles) {
    input_path <- norm(as.character(inputs[[role]]$path), paste("render input", role))
    if (!inside(input_path, shadow) ||
        !identical(reference_cell_state_v2_sha256_file(input_path),
                   as.character(inputs[[role]]$sha256))) {
      reference_cell_state_v2_abort("Seed%d render input changed: %s", expected_seed, role)
    }
  }
  render_dir <- dirname(paths$render)
  artifacts <- c(
    exact_review_html = "exact_review.html",
    crop_render_status = "crop_render_status.tsv",
    crop_manifest = "crop_manifest.tsv"
  )
  artifact_hashes <- unlist(render$artifact_file_sha256, use.names = TRUE)
  if (!setequal(names(artifact_hashes), names(artifacts))) {
    reference_cell_state_v2_abort("Seed%d render artifact hash set changed", expected_seed)
  }
  for (role in names(artifacts)) {
    artifact <- norm(file.path(render_dir, artifacts[[role]]), paste("render artifact", role))
    if (!inside(artifact, shadow) ||
        !identical(reference_cell_state_v2_sha256_file(artifact),
                   as.character(artifact_hashes[[role]]))) {
      reference_cell_state_v2_abort("Seed%d render artifact changed: %s", expected_seed, role)
    }
  }
  crops_path <- file.path(render_dir, "crop_manifest.tsv")
  crops <- reference_cell_state_v2_read_table(crops_path, "crop manifest")
  reference_cell_state_v2_require_columns(
    crops, c("morphology_umap_row_key", "channel", "relative_path", "sha256"),
    "crop manifest"
  )
  if (nrow(crops) != 2L * length(stable_ids) ||
      anyDuplicated(paste(crops$morphology_umap_row_key, crops$channel, sep = "|")) ||
      !setequal(crops$morphology_umap_row_key, stable_ids) ||
      !setequal(crops$channel, c("brightfield", "nuclei_support"))) {
    reference_cell_state_v2_abort("Seed%d crop manifest does not exactly cover its reviewed universe", expected_seed)
  }
  for (i in seq_len(nrow(crops))) {
    relative <- as.character(crops$relative_path[[i]])
    if (!grepl("^crops/[^/]+[.]png$", relative) || grepl("\\.\\.", basename(relative))) {
      reference_cell_state_v2_abort("Seed%d crop path is unsafe", expected_seed)
    }
    crop <- norm(file.path(render_dir, relative), "review crop")
    if (!inside(crop, render_dir) ||
        !identical(reference_cell_state_v2_sha256_file(crop), as.character(crops$sha256[[i]]))) {
      reference_cell_state_v2_abort("Seed%d crop evidence changed: %s", expected_seed, relative)
    }
  }
  render_identity <- render$identity
  submission_identity <- submission$identity
  renderer_implementation <- norm(
    as.character(render$renderer_implementation), "renderer implementation"
  )
  identity_fields <- c(
    "review_id", "project_sha256", "review_set_sha256", "review_manifest_sha256",
    "stable_id_sha256", "row_count", "crop_manifest_sha256",
    "renderer_implementation_sha256", "render_padding", "render_generation_id"
  )
  if (!identical(as.character(render$schema_version), "reference_cell_state_exact_review_render_v2") ||
      !isTRUE(render$exact_selection_preserved) || isTRUE(render$secondary_sampling_performed) ||
      !isTRUE(render$all_crops_available) ||
      !identical(as.character(submission$schema_version),
                 "reference_cell_state_exact_review_submission_v2") ||
      any(vapply(identity_fields, function(field) {
        !identical(as.character(render_identity[[field]]),
                   as.character(submission_identity[[field]]))
      }, logical(1))) ||
      !identical(as.character(render_identity$crop_manifest_sha256),
                 reference_cell_state_v2_sha256_file(crops_path)) ||
      !identical(as.character(render$crop_aggregate_sha256),
                 reference_cell_state_v2_sha256_file(crops_path)) ||
      !identical(reference_cell_state_v2_sha256_file(renderer_implementation),
                 as.character(render$renderer_implementation_sha256))) {
    reference_cell_state_v2_abort("Seed%d submission is not bound to its exact crop/render generation", expected_seed)
  }
  required_submission_fields <- c(
    "morphology_umap_row_key", "label", "label_confidence", "reviewer",
    "reviewed_at", "review_notes"
  )
  if (!is.list(submission$rows) || length(submission$rows) != nrow(reviewed) ||
      any(vapply(submission$rows, function(row) {
        !is.list(row) || !all(required_submission_fields %in% names(row)) ||
          any(vapply(row[required_submission_fields], length, integer(1)) != 1L)
      }, logical(1)))) {
    reference_cell_state_v2_abort("Seed%d submission rows are incomplete", expected_seed)
  }
  creation <- submission$creation_metadata
  if (!is.list(creation) || !valid_timestamp(as.character(creation$created_at)) ||
      !nzchar(trimws(as.character(creation$client_version)))) {
    reference_cell_state_v2_abort("Seed%d submission creation metadata is invalid", expected_seed)
  }
  submission_rows <- do.call(rbind, lapply(submission$rows, function(row) data.frame(
    morphology_umap_row_key = as.character(row$morphology_umap_row_key),
    label = as.character(row$label), label_confidence = as.character(row$label_confidence),
    reviewer = as.character(row$reviewer), reviewed_at = as.character(row$reviewed_at),
    review_notes = as.character(row$review_notes), stringsAsFactors = FALSE
  )))
  if (nrow(submission_rows) != nrow(reviewed) ||
      anyDuplicated(submission_rows$morphology_umap_row_key) ||
      !setequal(submission_rows$morphology_umap_row_key, stable_ids) ||
      !identical(stable_sha(stable_ids), as.character(render_identity$stable_id_sha256))) {
    reference_cell_state_v2_abort("Seed%d submission stable-ID universe changed", expected_seed)
  }
  if (any(!submission_rows$label %in% c(reference_cell_state_v2_class_ids(), "uncertain")) ||
      any(!submission_rows$label_confidence %in% c("high", "medium", "low")) ||
      any(!nzchar(submission_rows$reviewer)) ||
      any(submission_rows$reviewer != trimws(submission_rows$reviewer)) ||
      any(grepl("[\t\r\n]", submission_rows$reviewer)) ||
      any(!valid_timestamp(submission_rows$reviewed_at))) {
    reference_cell_state_v2_abort("Seed%d submission human evidence is malformed", expected_seed)
  }
  aligned <- submission_rows[match(stable_ids, submission_rows$morphology_umap_row_key), , drop = FALSE]
  evidence_fields <- c("label", "label_confidence", "reviewer", "reviewed_at", "review_notes")
  if (any(vapply(evidence_fields, function(field) {
    !identical(as.character(reviewed[[field]]), as.character(aligned[[field]]))
  }, logical(1)))) {
    reference_cell_state_v2_abort("Seed%d imported labels differ from authoritative submission", expected_seed)
  }
  selection_hashes <- unlist(selection$output_file_sha256, use.names = TRUE)
  if (!identical(as.character(selection$schema_version), expected_schema) ||
      !identical(as.character(selection$status), "HUMAN_REVIEW_REQUIRED") ||
      !identical(as.integer(selection$row_count), nrow(reviewed)) ||
      !identical(as.character(selection$stable_id_sha256), stable_sha(stable_ids)) ||
      !identical(as.character(selection_hashes[["review_set"]]),
                 reference_cell_state_v2_sha256_file(paths$review_set))) {
    reference_cell_state_v2_abort("Seed%d selection manifest no longer binds the exact review set", expected_seed)
  }
  output_hashes <- unlist(manifest$output_file_sha256, use.names = TRUE)
  audit_path <- norm(file.path(dirname(labels_path), "review_import_audit.tsv"), "review import audit")
  import_implementation <- norm(as.character(manifest$implementation), "review import implementation")
  import_shared_implementation <- norm(
    as.character(manifest$shared_implementation), "review import shared implementation"
  )
  expected_import_implementation <- norm(
    file.path(dirname(script_path), "38_import_reference_cell_state_exact_review.R"),
    "expected review import implementation"
  )
  if (!identical(as.character(manifest$schema_version), "reference_cell_state_exact_review_import_v2") ||
      !identical(as.character(manifest$status), "COMPLETE") ||
      !identical(as.character(manifest$reviewed_labels_sha256), reference_cell_state_v2_sha256_file(labels_path)) ||
      !identical(as.character(manifest$render_manifest_sha256),
                 reference_cell_state_v2_sha256_file(paths$render)) ||
      !identical(as.character(manifest$submission_sha256),
                 reference_cell_state_v2_sha256_file(paths$submission)) ||
      !identical(as.character(manifest$review_set_sha256),
                 reference_cell_state_v2_sha256_file(paths$review_set)) ||
      !identical(as.character(output_hashes[["reviewed_labels"]]),
                 reference_cell_state_v2_sha256_file(labels_path)) ||
      !identical(as.character(output_hashes[["review_import_audit"]]),
                 reference_cell_state_v2_sha256_file(audit_path)) ||
      !identical(import_implementation, expected_import_implementation) ||
      !identical(as.character(manifest$implementation_sha256),
                 reference_cell_state_v2_sha256_file(import_implementation)) ||
      !identical(import_shared_implementation, shared_script_path) ||
      !identical(as.character(manifest$shared_implementation_sha256),
                 reference_cell_state_v2_sha256_file(shared_script_path)) ||
      !isTRUE(manifest$exact_selection_preserved) || isTRUE(manifest$secondary_sampling_performed)) {
    reference_cell_state_v2_abort("Seed%d reviewed labels differ from their authoritative import", expected_seed)
  }
  list(
    path = manifest_path, sha256 = reference_cell_state_v2_sha256_file(manifest_path),
    render_manifest = paths$render,
    render_manifest_sha256 = reference_cell_state_v2_sha256_file(paths$render),
    submission = paths$submission,
    submission_sha256 = reference_cell_state_v2_sha256_file(paths$submission),
    selection_manifest = selection_path,
    selection_manifest_sha256 = reference_cell_state_v2_sha256_file(selection_path)
  )
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

canonical_value <- function(x) {
  x <- as.character(x); x[is.na(x)] <- ""; trimws(x)
}

conflict_candidate_hash <- function(conflicts, parent) {
  temporary <- tempfile(".review-conflicts-candidate-", tmpdir = parent)
  on.exit(unlink(temporary), add = TRUE)
  utils::write.table(conflicts, temporary, sep = "\t", quote = FALSE,
                     row.names = FALSE, na = "")
  reference_cell_state_v2_sha256_file(temporary)
}

verify_or_create_barrier <- function(barrier, conflicts, identity, create = FALSE) {
  expected_conflicts_sha <- conflict_candidate_hash(conflicts, dirname(barrier))
  if (dir.exists(barrier)) {
    manifest_path <- norm(file.path(barrier, "review_adjudication_barrier_manifest.json"),
                          "review adjudication barrier manifest")
    conflicts_path <- norm(file.path(barrier, "review_conflicts.tsv"),
                           "review adjudication conflicts")
    manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
    for (field in names(identity)) {
      if (!identical(as.character(manifest[[field]]), as.character(identity[[field]]))) {
        reference_cell_state_v2_abort("Existing adjudication barrier identity differs for %s", field)
      }
    }
    if (!identical(as.character(manifest$schema_version),
                   "reference_cell_state_review_adjudication_barrier_v2") ||
        !identical(as.character(manifest$status), "HUMAN_ADJUDICATION_REQUIRED") ||
        !identical(as.integer(manifest$conflict_count), nrow(conflicts)) ||
        !identical(as.character(manifest$conflicts_sha256), expected_conflicts_sha) ||
        !identical(reference_cell_state_v2_sha256_file(conflicts_path), expected_conflicts_sha)) {
      reference_cell_state_v2_abort("Existing adjudication barrier was modified or belongs to another conflict set")
    }
    return(list(path = manifest_path, sha256 = reference_cell_state_v2_sha256_file(manifest_path)))
  }
  if (file.exists(barrier) || reference_cell_state_v2_is_symlink(barrier)) {
    reference_cell_state_v2_abort("Adjudication barrier path is not a regular generation directory")
  }
  if (!isTRUE(create)) {
    reference_cell_state_v2_abort("Human adjudication may only be imported after the immutable conflict barrier is generated")
  }
  staging <- tempfile(paste0(".", basename(barrier), "-staging-"), tmpdir = dirname(barrier))
  if (!dir.create(staging)) reference_cell_state_v2_abort("Could not create adjudication barrier staging directory")
  on.exit(unlink(staging, recursive = TRUE, force = TRUE), add = TRUE)
  conflicts_path <- file.path(staging, "review_conflicts.tsv")
  reference_cell_state_v2_write_tsv_atomic(conflicts, conflicts_path)
  manifest <- c(list(
    schema_version = "reference_cell_state_review_adjudication_barrier_v2",
    status = "HUMAN_ADJUDICATION_REQUIRED", conflict_count = nrow(conflicts),
    conflicts_sha256 = reference_cell_state_v2_sha256_file(conflicts_path)
  ), identity)
  manifest_path <- file.path(staging, "review_adjudication_barrier_manifest.json")
  reference_cell_state_v2_write_json_atomic(manifest, manifest_path)
  if (file.exists(barrier) || dir.exists(barrier) || reference_cell_state_v2_is_symlink(barrier) ||
      !file.rename(staging, barrier)) {
    reference_cell_state_v2_abort("Failed to install immutable adjudication barrier")
  }
  list(
    path = file.path(barrier, "review_adjudication_barrier_manifest.json"),
    sha256 = reference_cell_state_v2_sha256_file(
      file.path(barrier, "review_adjudication_barrier_manifest.json")
    )
  )
}

validate_review <- function(rows, batch, api) {
  reference_cell_state_v2_require_columns(
    rows,
    c("morphology_umap_row_key", "label", "reviewer", "reviewed_at", "label_confidence"),
    paste(batch, "review")
  )
  keys <- canonical_value(rows$morphology_umap_row_key)
  if (any(!nzchar(keys)) || anyDuplicated(keys)) reference_cell_state_v2_abort("%s has blank/duplicate stable IDs", batch)
  prepared <- api$prepare_morphology_cell_state_manual_labels(rows)
  if (any(!prepared$labels$manual_review_evidence)) {
    reference_cell_state_v2_abort("%s contains rows without human review evidence", batch)
  }
  unsupported <- setdiff(unique(prepared$labels$manual_label), c(reference_cell_state_v2_class_ids(), "uncertain"))
  if (length(unsupported)) reference_cell_state_v2_abort("%s contains unsupported manual labels", batch)
  rows$.review_batch_source <- batch
  rows
}

validate_adjudication <- function(rows, unresolved, api) {
  required <- c(
    "morphology_umap_row_key", "label", "reviewer", "reviewed_at",
    "label_confidence", "adjudication_notes"
  )
  reference_cell_state_v2_require_columns(rows, required, "adjudication")
  for (field in required) {
    values <- as.character(rows[[field]])
    if (anyNA(values) || any(grepl("[[:cntrl:]]", values))) {
      reference_cell_state_v2_abort(
        "Adjudication %s contains missing values or control characters", field
      )
    }
    if (any(values != trimws(values))) {
      reference_cell_state_v2_abort("Adjudication %s must be trimmed", field)
    }
  }
  keys <- as.character(rows$morphology_umap_row_key)
  if (any(!nzchar(keys)) || anyDuplicated(keys) || !setequal(keys, unresolved)) {
    reference_cell_state_v2_abort(
      "Adjudication must cover each conflict stable ID exactly once"
    )
  }
  if (any(!rows$label %in% c(reference_cell_state_v2_class_ids(), "uncertain"))) {
    reference_cell_state_v2_abort("Adjudication contains an unsupported exact label")
  }
  if (any(!rows$label_confidence %in% c("high", "medium", "low"))) {
    reference_cell_state_v2_abort("Adjudication confidence must be high, medium, or low")
  }
  if (any(!nzchar(rows$reviewer)) || any(nchar(rows$reviewer, type = "chars") > 200L)) {
    reference_cell_state_v2_abort("Adjudication reviewer is blank or longer than 200 characters")
  }
  if (any(!valid_timestamp(rows$reviewed_at))) {
    reference_cell_state_v2_abort("Adjudication reviewed_at is not an RFC3339 timestamp")
  }
  if (any(!nzchar(rows$adjudication_notes)) ||
      any(nchar(rows$adjudication_notes, type = "chars") > 4000L)) {
    reference_cell_state_v2_abort(
      "Adjudication notes are required and must be at most 4000 characters"
    )
  }
  prepared <- api$prepare_morphology_cell_state_manual_labels(rows)
  if (any(!prepared$labels$manual_review_evidence) ||
      any(prepared$labels$manual_label != as.character(rows$label))) {
    reference_cell_state_v2_abort("Adjudication lacks exact human decision evidence")
  }
  rows
}

verify_existing_merge <- function(output, expected_manifest, candidate_paths) {
  manifest_path <- norm(file.path(output, "review_merge_manifest.json"),
                        "existing review merge manifest")
  observed <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
  hashes <- unlist(observed$output_file_sha256, use.names = TRUE)
  files <- c(
    merged = "merged_reviewed_labels.tsv", audit = "review_merge_audit.tsv",
    conflicts = "review_conflicts.tsv"
  )
  if (!setequal(names(hashes), names(files))) {
    reference_cell_state_v2_abort("Existing review merge output hash set changed")
  }
  for (role in names(files)) {
    path <- norm(file.path(output, files[[role]]), paste("existing merge", role))
    if (!identical(reference_cell_state_v2_sha256_file(path), as.character(hashes[[role]]))) {
      reference_cell_state_v2_abort("Existing review merge artifact changed: %s", role)
    }
    candidate_hash <- reference_cell_state_v2_sha256_file(candidate_paths[[role]])
    if (!identical(candidate_hash, as.character(hashes[[role]]))) {
      reference_cell_state_v2_abort("Existing review merge differs from recomputed human evidence: %s", role)
    }
  }
  observed$output_file_sha256 <- NULL
  expected_manifest$output_file_sha256 <- NULL
  if (!identical(
    jsonlite::toJSON(observed, auto_unbox = TRUE, null = "null", digits = NA),
    jsonlite::toJSON(expected_manifest, auto_unbox = TRUE, null = "null", digits = NA)
  )) {
    reference_cell_state_v2_abort("Existing review merge generation identity differs")
  }
  invisible(TRUE)
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  root <- norm(args$reference_root, "--reference-root", directory = TRUE)
  lock <- norm(args$dependency_lock, "--dependency-lock")
  seed1_path <- norm(args$seed1_reviewed_labels, "--seed1-reviewed-labels")
  seed2_path <- norm(args$seed2_reviewed_labels, "--seed2-reviewed-labels")
  output <- norm(args$output_dir, "--output-dir", output = TRUE)
  barrier <- file.path(dirname(output), paste0(basename(output), "_adjudication_barrier"))
  adjudication_path <- if (is.null(args$adjudication)) NULL else norm(args$adjudication, "--adjudication")
  seed1_marker <- "/human_review/seed1/import/reviewed_labels.tsv"
  seed2_marker <- "/human_review/seed2/import/reviewed_labels.tsv"
  if (!endsWith(seed1_path, seed1_marker) || !endsWith(seed2_path, seed2_marker)) {
    reference_cell_state_v2_abort("Seed review inputs must use canonical V2 authoritative-import paths")
  }
  shadow1 <- substr(seed1_path, 1L, nchar(seed1_path) - nchar(seed1_marker))
  shadow2 <- substr(seed2_path, 1L, nchar(seed2_path) - nchar(seed2_marker))
  if (!identical(shadow1, shadow2) || !inside(output, shadow1) ||
      (!is.null(adjudication_path) && !inside(adjudication_path, shadow1)) ||
      inside(output, root)) {
    reference_cell_state_v2_abort("Seed1/Seed2/adjudication/output must share one isolated V2 shadow root")
  }
  if (file.exists(output) && !dir.exists(output)) {
    reference_cell_state_v2_abort("Review merge output exists but is not a generation directory")
  }
  seed1_import <- verify_import_parent(seed1_path, 1L, shadow1)
  seed2_import <- verify_import_parent(seed2_path, 2L, shadow1)
  dependency <- reference_cell_state_v2_verify_dependency(root, lock)
  loaded <- reference_cell_state_v2_load_historical_classifier(root)
  api <- loaded$api
  seed1 <- validate_review(reference_cell_state_v2_read_table(seed1_path, "seed1 review"), "seed1", api)
  seed2 <- validate_review(reference_cell_state_v2_read_table(seed2_path, "seed2 review"), "seed2", api)
  all_columns <- union(names(seed1), names(seed2))
  for (column in setdiff(all_columns, names(seed1))) seed1[[column]] <- NA
  for (column in setdiff(all_columns, names(seed2))) seed2[[column]] <- NA
  combined <- rbind(seed1[, all_columns, drop = FALSE], seed2[, all_columns, drop = FALSE])
  key <- canonical_value(combined$morphology_umap_row_key)
  duplicated_keys <- sort(unique(key[duplicated(key) | duplicated(key, fromLast = TRUE)]), method = "radix")
  conflicts <- data.frame(
    morphology_umap_row_key = character(), seed1_label = character(), seed2_label = character(),
    seed1_reviewer = character(), seed2_reviewer = character(), conflict_reason = character(),
    resolution = character(), stringsAsFactors = FALSE
  )
  resolved_rows <- list()
  for (stable_id in duplicated_keys) {
    rows <- combined[key == stable_id, , drop = FALSE]
    labels <- unique(canonical_value(rows$label))
    identity_fields <- intersect(
      c("segmentation_object_id", "context_key", "ltee_cell_line", "source_id", "suffix", "mask_label"),
      names(rows)
    )
    identity_conflict <- any(vapply(identity_fields, function(column) {
      length(unique(canonical_value(rows[[column]]))[nzchar(unique(canonical_value(rows[[column]])))]) > 1L
    }, logical(1)))
    label_conflict <- length(labels) > 1L
    if (!identity_conflict && !label_conflict) {
      rows <- rows[order(rows$.review_batch_source, decreasing = TRUE), , drop = FALSE]
      resolved_rows[[stable_id]] <- rows[1L, , drop = FALSE]
      next
    }
    conflicts <- rbind(conflicts, data.frame(
      morphology_umap_row_key = stable_id,
      seed1_label = paste(canonical_value(rows$label[rows$.review_batch_source == "seed1"]), collapse = ";"),
      seed2_label = paste(canonical_value(rows$label[rows$.review_batch_source == "seed2"]), collapse = ";"),
      seed1_reviewer = paste(canonical_value(rows$reviewer[rows$.review_batch_source == "seed1"]), collapse = ";"),
      seed2_reviewer = paste(canonical_value(rows$reviewer[rows$.review_batch_source == "seed2"]), collapse = ";"),
      conflict_reason = paste(c(if (label_conflict) "manual_label_conflict", if (identity_conflict) "identity_conflict"), collapse = ";"),
      resolution = "HUMAN_ADJUDICATION_REQUIRED", stringsAsFactors = FALSE
    ))
  }
  unresolved <- conflicts$morphology_umap_row_key
  adjudication <- NULL
  barrier_identity <- list(
    seed1_reviewed_labels_sha256 = reference_cell_state_v2_sha256_file(seed1_path),
    seed2_reviewed_labels_sha256 = reference_cell_state_v2_sha256_file(seed2_path),
    seed1_review_import_manifest_sha256 = seed1_import$sha256,
    seed2_review_import_manifest_sha256 = seed2_import$sha256,
    implementation = script_path,
    implementation_sha256 = reference_cell_state_v2_sha256_file(script_path),
    shared_implementation = shared_script_path,
    shared_implementation_sha256 = reference_cell_state_v2_sha256_file(shared_script_path)
  )
  barrier_identity_value <- NULL
  if (length(unresolved)) {
    if (is.null(adjudication_path)) {
      dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
      barrier_identity_value <- verify_or_create_barrier(
        barrier, conflicts, barrier_identity, create = TRUE
      )
      reference_cell_state_v2_abort(
        "Review merge requires explicit human adjudication for %d stable ID(s); barrier=%s",
        length(unresolved), barrier
      )
    }
    barrier_identity_value <- verify_or_create_barrier(
      barrier, conflicts, barrier_identity, create = FALSE
    )
    adjudication <- validate_adjudication(
      reference_cell_state_v2_read_table(adjudication_path, "adjudication"),
      unresolved, api
    )
    for (stable_id in unresolved) {
      base <- combined[match(stable_id, key), , drop = FALSE]
      decision <- adjudication[match(stable_id, canonical_value(adjudication$morphology_umap_row_key)), , drop = FALSE]
      for (column in names(decision)) base[[column]] <- decision[[column]][[1L]]
      base$.review_batch_source <- "human_adjudication"
      resolved_rows[[stable_id]] <- base
    }
    conflicts$resolution <- "HUMAN_ADJUDICATED"
  }
  unique_rows <- combined[!key %in% duplicated_keys, , drop = FALSE]
  merged <- if (length(resolved_rows)) rbind(unique_rows, do.call(rbind, resolved_rows)) else unique_rows
  merged <- merged[order(canonical_value(merged$morphology_umap_row_key), method = "radix"), , drop = FALSE]
  rownames(merged) <- NULL
  if (anyDuplicated(merged$morphology_umap_row_key)) reference_cell_state_v2_abort("Merged review retains duplicate stable IDs")
  merged$.review_batch_source <- as.character(merged$.review_batch_source)
  final_prepared <- api$prepare_morphology_cell_state_manual_labels(merged)
  if (any(!final_prepared$labels$manual_review_evidence)) reference_cell_state_v2_abort("Merged labels include non-human rows")
  audit <- data.frame(
    seed1_rows = nrow(seed1), seed2_rows = nrow(seed2), overlapping_stable_ids = length(duplicated_keys),
    identical_overlaps_collapsed = length(duplicated_keys) - nrow(conflicts),
    adjudicated_conflicts = if (is.null(adjudication)) 0L else nrow(conflicts),
    merged_rows = nrow(merged), training_eligible_rows = final_prepared$audit$n_training_eligible,
    pseudo_or_umap_rows_used_as_manual = 0L, stringsAsFactors = FALSE
  )
  dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
  staging <- tempfile(paste0(".", basename(output), "-staging-"), tmpdir = dirname(output))
  if (!dir.create(staging)) reference_cell_state_v2_abort("Could not create review-merge staging directory")
  on.exit(unlink(staging, recursive = TRUE, force = TRUE), add = TRUE)
  paths <- c(
    merged = file.path(staging, "merged_reviewed_labels.tsv"),
    audit = file.path(staging, "review_merge_audit.tsv"),
    conflicts = file.path(staging, "review_conflicts.tsv")
  )
  reference_cell_state_v2_write_tsv_atomic(merged, paths[["merged"]])
  reference_cell_state_v2_write_tsv_atomic(audit, paths[["audit"]])
  reference_cell_state_v2_write_tsv_atomic(conflicts, paths[["conflicts"]])
  manifest <- list(
    schema_version = SCHEMA_VERSION, status = "COMPLETE",
    merge_key = "morphology_umap_row_key", conflict_policy = "human_adjudication_required_no_last_write_wins",
    seed1_reviewed_labels_sha256 = reference_cell_state_v2_sha256_file(seed1_path),
    seed2_reviewed_labels_sha256 = reference_cell_state_v2_sha256_file(seed2_path),
    seed1_review_import_manifest = seed1_import$path,
    seed1_review_import_manifest_sha256 = seed1_import$sha256,
    seed1_selection_manifest_sha256 = seed1_import$selection_manifest_sha256,
    seed1_render_manifest_sha256 = seed1_import$render_manifest_sha256,
    seed1_submission_sha256 = seed1_import$submission_sha256,
    seed2_review_import_manifest = seed2_import$path,
    seed2_review_import_manifest_sha256 = seed2_import$sha256,
    seed2_selection_manifest_sha256 = seed2_import$selection_manifest_sha256,
    seed2_render_manifest_sha256 = seed2_import$render_manifest_sha256,
    seed2_submission_sha256 = seed2_import$submission_sha256,
    complete_human_evidence_chain_revalidated = TRUE,
    adjudication_sha256 = if (is.null(adjudication_path)) NULL else reference_cell_state_v2_sha256_file(adjudication_path),
    adjudication_barrier_manifest = if (is.null(barrier_identity_value)) NULL else barrier_identity_value$path,
    adjudication_barrier_manifest_sha256 = if (is.null(barrier_identity_value)) NULL else barrier_identity_value$sha256,
    merged_row_count = nrow(merged), manual_evidence_required = TRUE,
    pseudo_label_training = FALSE, umap_label_training = FALSE, diagnostic_cluster_training = FALSE,
    output_file_sha256 = as.list(vapply(paths, reference_cell_state_v2_sha256_file, character(1))),
    dependency = dependency, historical_source_identity = loaded$identity,
    implementation = script_path,
    implementation_sha256 = reference_cell_state_v2_sha256_file(script_path),
    shared_implementation = shared_script_path,
    shared_implementation_sha256 = reference_cell_state_v2_sha256_file(shared_script_path)
  )
  if (dir.exists(output)) {
    verify_existing_merge(output, manifest, paths)
    writeLines(sprintf("reference_cell_state_review_merge_verified_reuse=1 rows=%d output=%s",
                       nrow(merged), output))
    return(invisible(0L))
  }
  reference_cell_state_v2_write_json_atomic(manifest, file.path(staging, "review_merge_manifest.json"))
  if (file.exists(output) || dir.exists(output) || reference_cell_state_v2_is_symlink(output)) {
    reference_cell_state_v2_abort("Review merge output appeared during staging")
  }
  if (!file.rename(staging, output)) reference_cell_state_v2_abort("Failed to install review merge atomically")
  writeLines(sprintf("reference_cell_state_review_merge_complete=1 rows=%d output=%s", nrow(merged), output))
}

status <- tryCatch(main(), error = function(error) { writeLines(conditionMessage(error), stderr()); 1L })
quit(save = "no", status = as.integer(status), runLast = FALSE)
