#!/usr/bin/env Rscript

# Build the first human-only reference-cell-state review batch from one
# accepted polygon annotation.  Suggested UMAP/cluster labels are sampling
# strata only: the emitted template is deliberately not a training table until
# a human supplies reviewer/reviewed_at evidence.

SCHEMA_VERSION <- "reference_cell_state_seed1_review_v2"
CLASS_IDS <- c("live_cell", "dead_cell", "multinucleated_cell")
Sys.setenv(GIT_OPTIONAL_LOCKS = "0")

script_argument <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_argument) != 1L) stop("Unable to locate adapter source file", call. = FALSE)
script_path <- normalizePath(sub("^--file=", "", script_argument[[1L]]), mustWork = TRUE)
shared_script_path <- normalizePath(
  file.path(dirname(script_path), "_shared", "reference_cell_state_v2.R"),
  mustWork = TRUE
)
source(shared_script_path, local = .GlobalEnv)

abort <- function(format, ...) stop(sprintf(format, ...), call. = FALSE)

parse_args <- function(arguments) {
  allowed <- c(
    "--reference-root", "--dependency-lock", "--project",
    "--annotation-import-dir", "--diagnostic-cluster-manifest",
    "--expanded-labelability-decision", "--output-dir"
  )
  result <- list()
  i <- 1L
  while (i <= length(arguments)) {
    current <- arguments[[i]]
    if (!current %in% allowed || i == length(arguments)) {
      abort("Unknown argument or missing value: %s", current)
    }
    name <- gsub("-", "_", substring(current, 3L), fixed = TRUE)
    if (!is.null(result[[name]])) abort("Argument supplied more than once: %s", current)
    result[[name]] <- arguments[[i + 1L]]
    i <- i + 2L
  }
  required <- c("reference_root", "dependency_lock", "project", "output_dir")
  missing <- required[!vapply(required, function(name) {
    !is.null(result[[name]]) && nzchar(result[[name]])
  }, logical(1))]
  if (length(missing)) abort("Missing required arguments: %s", paste(missing, collapse = ", "))
  branches <- c(
    !is.null(result$annotation_import_dir),
    !is.null(result$diagnostic_cluster_manifest),
    !is.null(result$expanded_labelability_decision)
  )
  if (sum(branches) != 1L) {
    abort(paste(
      "Supply exactly one of --annotation-import-dir, --diagnostic-cluster-manifest,",
      "or --expanded-labelability-decision"
    ))
  }
  result
}

normalize_input <- function(path, label, directory = FALSE) {
  if (!grepl("^/", path)) abort("%s must be absolute: %s", label, path)
  path <- normalizePath(path, winslash = "/", mustWork = TRUE)
  if (directory && !dir.exists(path)) abort("%s is not a directory", label)
  if (!directory && (!file.exists(path) || dir.exists(path))) abort("%s is not a file", label)
  path
}

normalize_output_dir <- function(path) {
  reference_cell_state_v2_normalize_output(path, "--output-dir")
}

is_within <- function(path, root) identical(path, root) || startsWith(path, paste0(root, "/"))

run_git <- function(root, ...) {
  output <- suppressWarnings(system2("git", c("-C", root, ...), stdout = TRUE, stderr = TRUE))
  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) abort("Unable to inspect pinned reference checkout")
  trimws(paste(output, collapse = "\n"))
}

read_lock <- function(path, root) {
  lock <- utils::read.delim(path, sep = "\t", header = TRUE, quote = "", comment.char = "",
                            check.names = FALSE, stringsAsFactors = FALSE)
  row <- lock[lock$dependency == "cellphenotypeannotator", , drop = FALSE]
  if (nrow(row) != 1L || !identical(row$execution_mode[[1L]], "private_read_only_source_checkout")) {
    abort("Dependency lock lacks one private read-only cellphenotypeannotator row")
  }
  if (nzchar(run_git(root, "status", "--porcelain", "--untracked-files=all")) ||
      !identical(run_git(root, "rev-parse", "HEAD"), row$commit[[1L]]) ||
      !identical(run_git(root, "rev-parse", "HEAD^{tree}"), row$tree[[1L]])) {
    abort("Pinned reference checkout differs from dependency lock")
  }
  row
}

sha256_file <- function(path) digest::digest(file = path, algo = "sha256")

source_reference_review <- function(root) {
  files <- c(
    segmentation_qc_review = file.path(root, "reference", "ltee-source", "code", "lib", "segmentation_qc_review.R"),
    morphology_cell_state_review = file.path(root, "reference", "ltee-source", "code", "lib", "morphology_cell_state_review.R")
  )
  before <- run_git(root, "status", "--porcelain", "--untracked-files=all")
  if (nzchar(before)) abort("Pinned reference checkout became dirty before review source loading")
  for (path in files) source(path, local = .GlobalEnv)
  required <- c(
    "segmentation_qc_targeted_balanced_sample",
    "generate_morphology_cell_state_gold_review_set",
    "morphology_cell_state_review_label_template",
    "morphology_cell_state_review_label_columns"
  )
  missing <- required[!vapply(required, exists, logical(1), mode = "function")]
  if (length(missing)) abort("Historical review API is incomplete: %s", paste(missing, collapse = ", "))
  after <- run_git(root, "status", "--porcelain", "--untracked-files=all")
  if (!identical(before, after)) abort("Loading historical review sources changed the pinned checkout")
  list(
    loading_policy = "pinned_function_only_review_sources_into_clean_process_global_environment",
    source_file_sha256 = as.list(vapply(files, sha256_file, character(1)))
  )
}

read_project <- function(path) {
  payload <- if (requireNamespace("yaml", quietly = TRUE)) {
    yaml::read_yaml(path)
  } else {
    jsonlite::fromJSON(path, simplifyVector = FALSE)
  }
  if (!is.list(payload) || is.null(payload$cells_file)) abort("Project lacks cells_file")
  payload
}

resolve_plate_contract_project <- function(project_path) {
  project_dir <- dirname(project_path)
  parent_import_path <- normalize_input(
    file.path(project_dir, "parent_import_manifest.json"), "project parent import"
  )
  parent_import <- jsonlite::fromJSON(parent_import_path, simplifyVector = FALSE)
  if (!is.null(parent_import$parent)) {
    return(reference_cell_state_v2_resolve_plate_contract(project_path))
  }
  if (!identical(as.character(parent_import$schema_version),
                 "reference_cell_state_expanded_annotation_parent_import_v1") ||
      !identical(as.character(parent_import$status), "COMPLETE")) {
    abort("Project parent import lacks a supported frozen plate-contract lineage")
  }
  base_binding <- parent_import$inputs$base_project
  if (!is.list(base_binding) ||
      !all(c("path", "sha256") %in% names(base_binding))) {
    abort("Expanded project parent import lacks its frozen base-project binding")
  }
  base_project <- normalize_input(as.character(base_binding$path), "expanded base project")
  if (nzchar(Sys.readlink(base_project)) ||
      !identical(sha256_file(base_project), as.character(base_binding$sha256))) {
    abort("Expanded base project differs from its parent-import binding")
  }
  reference_cell_state_v2_resolve_plate_contract(base_project)
}

read_table <- function(path, label) {
  rows <- utils::read.delim(path, sep = "\t", header = TRUE, quote = "", comment.char = "",
                            check.names = FALSE, stringsAsFactors = FALSE)
  if (!is.data.frame(rows) || !nrow(rows)) abort("%s is empty", label)
  rows
}

require_columns <- function(rows, columns, label) {
  missing <- setdiff(columns, names(rows))
  if (length(missing)) abort("%s lacks columns: %s", label, paste(missing, collapse = ", "))
}

stable_key <- function(rows) {
  for (column in c("morphology_umap_row_key", "stable_cell_id", "cell_id")) {
    if (column %in% names(rows)) return(as.character(rows[[column]]))
  }
  abort("Rows lack a stable cell identity column")
}

make_annotation_rows <- function(cells, labels, plate_contract) {
  cells <- reference_cell_state_v2_validate_canonical_cells(cells, plate_contract)
  require_columns(cells, "cluster", "cells.tsv historical diagnostic clusters")
  require_columns(labels, c("cell_id", "assignment_state", "class_id", "region_id"), "provisional labels")
  if (anyDuplicated(cells$cell_id) || anyDuplicated(labels$cell_id) || !setequal(cells$cell_id, labels$cell_id)) {
    abort("cells/provisional-label stable-ID universes are not one-to-one")
  }
  labels <- labels[match(cells$cell_id, labels$cell_id), , drop = FALSE]
  if (any(labels$assignment_state == "class_assigned" & !labels$class_id %in% CLASS_IDS)) {
    abort("Provisional labels contain unsupported class IDs")
  }
  suggested <- ifelse(labels$assignment_state == "class_assigned", labels$class_id, "")
  rows <- data.frame(
    morphology_umap_row_key = as.character(cells$cell_id),
    stable_cell_id = as.character(cells$cell_id),
    segmentation_qc_cell_id = as.character(cells$segmentation_qc_cell_id),
    segmentation_object_id = as.character(cells$segmentation_object_id),
    context_key = as.character(cells$context_key),
    ltee_cell_line = as.character(cells$ltee_cell_line),
    cell_line_family = as.character(cells$ltee_cell_line_family),
    source_id = as.character(cells$source_id),
    suffix = as.character(cells$suffix),
    mask_label = as.integer(cells$mask_label),
    feature_row_index = as.integer(cells$feature_row_index),
    condition = as.character(cells$condition),
    condition_code = as.character(cells$condition),
    passage_number = 0L,
    cluster = as.character(cells$cluster),
    Dim1 = if ("Dim1" %in% names(cells)) as.numeric(cells$Dim1) else NA_real_,
    Dim2 = if ("Dim2" %in% names(cells)) as.numeric(cells$Dim2) else NA_real_,
    suggested_cell_state_label = suggested,
    suggested_region_id = as.character(labels$region_id),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  rows$well <- as.character(cells$well)
  if (any(!grepl("^-?[0-9]+$", rows$cluster))) {
    abort("cells.tsv historical diagnostic cluster must be an integer for every row")
  }
  rows
}

allocate_balanced <- function(capacities, total, max_per_group = 8L) {
  capacity_names <- names(capacities)
  capacities <- stats::setNames(
    pmin(as.integer(capacities), as.integer(max_per_group)), capacity_names
  )
  quotas <- stats::setNames(rep(0L, length(capacities)), names(capacities))
  remaining <- min(as.integer(total), sum(capacities))
  while (remaining > 0L) {
    progressed <- FALSE
    for (name in sort(names(capacities), method = "radix")) {
      if (quotas[[name]] >= capacities[[name]]) next
      quotas[[name]] <- quotas[[name]] + 1L
      remaining <- remaining - 1L
      progressed <- TRUE
      if (!remaining) break
    }
    if (!progressed) abort("Global balanced fallback quota allocation stalled")
  }
  quotas
}

all_unassigned_fallback <- function(rows, seed = 1L, max_total = 500L, max_per_group = 8L) {
  groups <- split(seq_len(nrow(rows)), as.character(rows$source_id))
  contexts <- sort(unique(as.character(rows$context_key)), method = "radix")
  context_capacity <- vapply(contexts, function(context) {
    wells <- unique(as.character(rows$source_id[rows$context_key == context]))
    sum(pmin(vapply(wells, function(well) sum(rows$source_id == well), integer(1)), max_per_group))
  }, integer(1))
  context_quota <- allocate_balanced(context_capacity, max_total, max(context_capacity))
  quotas <- stats::setNames(rep(0L, length(groups)), names(groups))
  for (context in contexts) {
    wells <- sort(unique(as.character(rows$source_id[rows$context_key == context])), method = "radix")
    capacities <- vapply(wells, function(well) length(groups[[well]]), integer(1))
    quotas[wells] <- allocate_balanced(capacities, context_quota[[context]], max_per_group)
  }
  selected <- list()
  audit <- list()
  for (position in seq_along(groups)) {
    group <- names(groups)[[position]]
    candidates <- rows[groups[[position]], , drop = FALSE]
    quota <- quotas[[group]]
    sampled <- if (quota) segmentation_qc_targeted_balanced_sample(
      candidates, n = quota, seed = seed + position
    ) else candidates[FALSE, , drop = FALSE]
    if (nrow(sampled)) {
      sampled$review_sampling_bucket <- "all_unassigned_global_balanced_fallback"
      sampled$review_stratum <- paste0("all_unassigned|", group)
      sampled$review_stratum_quota <- quota
      sampled$umap_derived_label <- ""
      sampled$review_default_label <- "uncertain"
      sampled$suggested_label_source <- "none_human_blind_review_required"
      selected[[length(selected) + 1L]] <- sampled
    }
    audit[[length(audit) + 1L]] <- data.frame(
      context_key = unique(as.character(candidates$context_key)), source_id = group,
      sampling_bucket = "all_unassigned_global_balanced_fallback",
      available_rows = nrow(candidates), requested_rows = quota,
      selected_rows = nrow(sampled), shortage_rows = max(0L, quota - nrow(sampled)),
      stringsAsFactors = FALSE
    )
  }
  review <- do.call(rbind, selected)
  review$review_batch_id <- paste0(
    "cellstate_gold_seed1_all_unassigned_",
    substr(digest::digest(paste(sort(review$morphology_umap_row_key), collapse = "\n"),
                          algo = "sha256", serialize = FALSE), 1L, 12L)
  )
  review$review_seed <- seed
  review$review_selected_at <- format(Sys.time(), "%Y-%m-%dT%H:%M:%OS%z")
  review$review_schema_version <- morphology_cell_state_review_schema_version()
  review$annotation_input_sha256 <- NA_character_
  review <- review[order(review$source_id, review$suffix, review$feature_row_index), , drop = FALSE]
  review$review_selection_index <- seq_len(nrow(review))
  list(review = review, audit = do.call(rbind, audit))
}

stable_polygon_review <- function(rows, seed = 1L, max_total = 500L, max_per_group = 8L) {
  # The historical helper is called unchanged whenever its nominal design fits.
  contexts <- sort(unique(as.character(rows$context_key)), method = "radix")
  nominal <- length(contexts) * (3L * 30L + 10L)
  if (nominal <= max_total) {
    review <- generate_morphology_cell_state_gold_review_set(
      rows, n_per_assigned_class = 30L, n_unassigned_per_context = 10L,
      seed = seed, max_total = max_total
    )
    audit <- aggregate(
      rep(1L, nrow(review)),
      list(context_key = review$context_key, sampling_bucket = review$review_sampling_bucket),
      sum
    )
    names(audit)[[3L]] <- "selected_rows"
    audit$available_rows <- NA_integer_
    audit$requested_rows <- audit$selected_rows
    audit$shortage_rows <- 0L
    return(list(review = review, audit = audit, design = "historical_exact_nominal"))
  }

  abort("Stable-polygon review expected exactly two plate-ploidy contexts within max_total")
}

read_fallback_labels <- function(manifest_path, project_dir, cells) {
  expected_path <- normalizePath(
    file.path(project_dir, "historical_projection", "historical_projection_manifest.json"),
    winslash = "/", mustWork = TRUE
  )
  if (!identical(manifest_path, expected_path)) {
    abort("Fallback manifest must be the canonical project historical projection manifest")
  }
  manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
  diagnostic <- manifest$diagnostic_cluster
  if (!identical(as.character(manifest$schema_version), "reference_cell_state_historical_projection_v2") ||
      !identical(as.character(manifest$status), "COMPLETE") ||
      !isTRUE(diagnostic$one_cluster_fallback) ||
      !identical(as.character(diagnostic$role), "metadata_only")) {
    abort("All-unassigned fallback requires an authoritative one_cluster_fallback historical manifest")
  }
  cluster_path <- normalizePath(file.path(project_dir, "historical_projection", "diagnostic_clusters.tsv"),
                                winslash = "/", mustWork = TRUE)
  declared <- as.character(manifest$output_file_sha256[["diagnostic_clusters.tsv"]])
  if (!identical(sha256_file(cluster_path), declared)) {
    abort("Diagnostic cluster artifact differs from its historical manifest")
  }
  clusters <- read_table(cluster_path, "diagnostic clusters")
  require_columns(clusters, c("cell_id", "cluster", "cluster_source"), "diagnostic clusters")
  if (anyDuplicated(clusters$cell_id) || !setequal(clusters$cell_id, cells$cell_id) ||
      any(as.character(clusters$cluster) != "1") ||
      any(as.character(clusters$cluster_source) != "optimize_dbscan_v4_one_cluster_fallback") ||
      any(as.character(cells$cluster) != "1")) {
    abort("Canonical cells and diagnostic artifact do not prove one-cluster fallback")
  }
  list(
    labels = data.frame(
      cell_id = as.character(cells$cell_id), assignment_state = "unreviewed",
      class_id = "", region_id = "", stringsAsFactors = FALSE
    ),
    manifest = manifest, cluster_path = cluster_path,
    mode = "authoritative_one_cluster_fallback",
    authority_path = manifest_path,
    authority_schema_version = as.character(manifest$schema_version)
  )
}

read_expanded_nogo_labels <- function(decision_path, project_dir, cells) {
  expected_decision <- normalizePath(
    file.path(project_dir, "historical_projection", "labelability_decision.json"),
    winslash = "/", mustWork = TRUE
  )
  if (!identical(decision_path, expected_decision)) {
    abort("Expanded fallback decision must be the canonical project labelability decision")
  }
  historical_path <- normalizePath(
    file.path(project_dir, "historical_projection", "historical_projection_manifest.json"),
    winslash = "/", mustWork = TRUE
  )
  expanded_path <- normalizePath(
    file.path(project_dir, "historical_projection", "expanded_projection_manifest.json"),
    winslash = "/", mustWork = TRUE
  )
  cluster_path <- normalizePath(
    file.path(project_dir, "historical_projection", "diagnostic_clusters.tsv"),
    winslash = "/", mustWork = TRUE
  )
  decision <- jsonlite::fromJSON(decision_path, simplifyVector = FALSE)
  historical <- jsonlite::fromJSON(historical_path, simplifyVector = FALSE)
  expanded <- jsonlite::fromJSON(expanded_path, simplifyVector = FALSE)
  if (!identical(as.character(decision$schema_version),
                 "reference_cell_state_expanded_labelability_decision_v1") ||
      !identical(as.character(decision$status), "COMPLETE") ||
      !identical(as.character(decision$selected_annotation_profile), "expanded39") ||
      !identical(as.character(decision$computational_gate), "FAIL") ||
      !identical(as.character(decision$overall_labelability),
                 "NO_GO_FOR_POLYGON_ANNOTATION_USE_500_CELL_BLIND_REVIEW")) {
    abort("Expanded fallback requires an authoritative computational-labelability NO_GO")
  }
  if (!identical(as.character(historical$schema_version),
                 "reference_cell_state_historical_projection_expanded_v1") ||
      !identical(as.character(historical$status), "COMPLETE") ||
      !identical(as.character(historical$selected_annotation_profile), "expanded39") ||
      !identical(as.character(historical$computational_labelability_gate), "FAIL") ||
      !identical(as.character(historical$expanded_feature_role),
                 "annotation_geometry_and_human_morphology_evidence_only") ||
      !identical(historical$expanded_features_allowed_in_final_classifier, FALSE) ||
      !identical(as.character(historical$labelability_decision_sha256),
                 sha256_file(decision_path)) ||
      !identical(as.character(historical$expanded_projection_manifest_sha256),
                 sha256_file(expanded_path))) {
    abort("Expanded project historical manifest does not bind the NO_GO decision")
  }
  if (!identical(as.character(expanded$schema_version),
                 "reference_cell_state_expanded_projection_comparison_v1") ||
      !identical(as.character(expanded$status), "COMPLETE") ||
      !identical(as.character(expanded$selected_annotation_profile), "expanded39") ||
      !identical(as.character(expanded$classifier_boundary$final_classifier_feature_source),
                 "classifier12") ||
      !identical(expanded$classifier_boundary$expanded39_allowed_in_final_classifier, FALSE)) {
    abort("Expanded projection manifest violates the frozen classifier boundary")
  }
  historical_hashes <- unlist(historical$output_file_sha256, use.names = TRUE)
  required_hashes <- c(
    "diagnostic_clusters.tsv", "expanded_projection_manifest.json",
    "labelability_decision.json"
  )
  if (!all(required_hashes %in% names(historical_hashes)) ||
      !identical(as.character(historical_hashes[["diagnostic_clusters.tsv"]]),
                 sha256_file(cluster_path)) ||
      !identical(as.character(historical_hashes[["expanded_projection_manifest.json"]]),
                 sha256_file(expanded_path)) ||
      !identical(as.character(historical_hashes[["labelability_decision.json"]]),
                 sha256_file(decision_path))) {
    abort("Expanded fallback artifacts differ from the project historical manifest")
  }
  clusters <- read_table(cluster_path, "expanded diagnostic clusters")
  require_columns(clusters, c("cell_id", "cluster"), "expanded diagnostic clusters")
  if (anyDuplicated(clusters$cell_id) || !setequal(clusters$cell_id, cells$cell_id)) {
    abort("Expanded diagnostic clusters do not cover canonical cells exactly")
  }
  clusters <- clusters[match(cells$cell_id, clusters$cell_id), , drop = FALSE]
  if (any(!grepl("^-?[0-9]+$", as.character(clusters$cluster))) ||
      !identical(as.character(clusters$cluster), as.character(cells$cluster))) {
    abort("Canonical cells differ from expanded diagnostic clusters")
  }
  list(
    labels = data.frame(
      cell_id = as.character(cells$cell_id), assignment_state = "unreviewed",
      class_id = "", region_id = "", stringsAsFactors = FALSE
    ),
    manifest = historical, cluster_path = cluster_path,
    mode = "authoritative_expanded_computational_nogo",
    authority_path = decision_path,
    authority_schema_version = as.character(decision$schema_version)
  )
}

read_polygon_labels <- function(annotation_dir, cells) {
  labels_path <- normalize_input(file.path(annotation_dir, "provisional_labels.tsv"), "provisional labels")
  manifest_path <- normalize_input(
    file.path(annotation_dir, "annotation_import_manifest.json"), "annotation manifest"
  )
  manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
  hashes <- unlist(manifest$artifact_file_sha256, use.names = TRUE)
  if (!identical(as.character(manifest$schema_version), "cell_phenotype_annotator_annotation_import_v1") ||
      !setequal(names(hashes), c("region_submission", "regions", "provisional_labels")) ||
      !identical(sha256_file(labels_path), as.character(hashes[["provisional_labels"]]))) {
    abort("Polygon annotation import manifest or provisional-label hash is invalid")
  }
  labels <- read_table(labels_path, "provisional labels")
  require_columns(labels, c("cell_id", "assignment_state", "class_id", "region_id"), "provisional labels")
  if (anyDuplicated(labels$cell_id) || !setequal(labels$cell_id, cells$cell_id)) {
    abort("Polygon provisional labels do not cover canonical cells exactly")
  }
  list(labels = labels, labels_path = labels_path, manifest_path = manifest_path)
}

write_tsv_atomic <- function(rows, path) {
  temporary <- tempfile(paste0(".", basename(path), "-"), tmpdir = dirname(path))
  on.exit(unlink(temporary), add = TRUE)
  utils::write.table(rows, temporary, sep = "\t", quote = FALSE, row.names = FALSE, na = "")
  if (!file.rename(temporary, path)) abort("Failed to install output: %s", path)
}

verify_existing_seed1 <- function(output, expected, selection) {
  manifest_path <- normalize_input(file.path(output, "seed1_review_manifest.json"),
                                   "existing Seed1 manifest")
  manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
  scalar <- c(
    "schema_version", "status", "design", "seed", "max_total", "max_per_group",
    "all_unassigned_fallback", "project_sha256", "selection_input_mode",
    "historical_parity_scope", "annotation_manifest_sha256", "provisional_labels_sha256",
    "diagnostic_cluster_manifest_sha256", "one_cluster_fallback_proven",
    "expanded_labelability_decision_sha256", "computational_nogo_fallback_proven",
    "fallback_authority_schema_version",
    "row_count", "stable_id_sha256", "implementation", "implementation_sha256",
    "shared_implementation", "shared_implementation_sha256"
  )
  for (field in scalar) {
    if (!identical(as.character(manifest[[field]]), as.character(expected[[field]]))) {
      abort("Existing Seed1 generation identity differs for %s", field)
    }
  }
  if (!identical(jsonlite::toJSON(manifest$dependency, auto_unbox = TRUE, null = "null"),
                 jsonlite::toJSON(expected$dependency, auto_unbox = TRUE, null = "null")) ||
      !identical(jsonlite::toJSON(manifest$historical_review_source_identity, auto_unbox = TRUE, null = "null"),
                 jsonlite::toJSON(expected$historical_review_source_identity, auto_unbox = TRUE, null = "null")) ||
      !identical(jsonlite::toJSON(manifest$historical_classifier_source_identity, auto_unbox = TRUE, null = "null"),
                 jsonlite::toJSON(expected$historical_classifier_source_identity, auto_unbox = TRUE, null = "null"))) {
    abort("Existing Seed1 dependency/source identity differs")
  }
  files <- c(
    review_set = "seed1_review_set.tsv", label_template = "seed1_review_label_template.tsv",
    selection_audit = "seed1_selection_audit.tsv"
  )
  hashes <- unlist(manifest$output_file_sha256, use.names = TRUE)
  if (!setequal(names(hashes), names(files))) abort("Existing Seed1 output hash set changed")
  for (role in names(files)) {
    path <- normalize_input(file.path(output, files[[role]]), paste("existing Seed1", role))
    if (!identical(sha256_file(path), as.character(hashes[[role]]))) {
      abort("Existing Seed1 output changed: %s", role)
    }
  }
  existing_review <- read_table(file.path(output, files[["review_set"]]), "existing Seed1 review set")
  existing_template <- read_table(file.path(output, files[["label_template"]]), "existing Seed1 template")
  compare_rows <- function(left, right) {
    ignored <- "review_selected_at"
    columns <- setdiff(names(right), ignored)
    identical(names(left), names(right)) && nrow(left) == nrow(right) &&
      all(vapply(columns, function(column) {
        left_value <- as.character(left[[column]]); left_value[is.na(left_value)] <- ""
        right_value <- as.character(right[[column]]); right_value[is.na(right_value)] <- ""
        identical(left_value, right_value)
      }, logical(1)))
  }
  expected_template <- morphology_cell_state_review_label_template(selection$review)
  expected_template$reviewer <- ""; expected_template$reviewed_at <- ""
  expected_template$label_confidence <- "high"
  existing_audit <- read_table(file.path(output, files[["selection_audit"]]),
                               "existing Seed1 selection audit")
  if (!compare_rows(existing_review, selection$review) ||
      !compare_rows(existing_template, expected_template) ||
      !compare_rows(existing_audit, selection$audit)) {
    abort("Existing Seed1 exact selected universe differs")
  }
  invisible(TRUE)
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  root <- normalize_input(args$reference_root, "--reference-root", directory = TRUE)
  lock <- normalize_input(args$dependency_lock, "--dependency-lock")
  project_path <- normalize_input(args$project, "--project")
  output <- normalize_output_dir(args$output_dir)
  project_dir <- dirname(project_path)
  shadow_root <- dirname(dirname(project_dir))
  annotation_dir <- if (is.null(args$annotation_import_dir)) NULL else {
    normalize_input(args$annotation_import_dir, "--annotation-import-dir", directory = TRUE)
  }
  diagnostic_manifest <- if (is.null(args$diagnostic_cluster_manifest)) NULL else {
    normalize_input(args$diagnostic_cluster_manifest, "--diagnostic-cluster-manifest")
  }
  expanded_decision <- if (is.null(args$expanded_labelability_decision)) NULL else {
    normalize_input(args$expanded_labelability_decision, "--expanded-labelability-decision")
  }
  authority_input <- if (!is.null(annotation_dir)) annotation_dir else {
    if (!is.null(diagnostic_manifest)) diagnostic_manifest else expanded_decision
  }
  scoped <- c(output, authority_input)
  if (any(!vapply(scoped, is_within, logical(1), root = shadow_root))) {
    abort("Seed1 input/output must remain inside the V2 reference shadow root")
  }
  dependency <- reference_cell_state_v2_verify_dependency(root, lock)
  review_source_identity <- source_reference_review(root)
  classifier_source <- reference_cell_state_v2_load_historical_classifier(root)
  project <- read_project(project_path)
  plate_contract <- resolve_plate_contract_project(project_path)
  cells_path <- normalize_input(file.path(project_dir, project$cells_file), "project cells")
  cells <- read_table(cells_path, "cells.tsv")
  fallback <- NULL
  if (!is.null(annotation_dir)) {
    polygon <- read_polygon_labels(annotation_dir, cells)
    labels_path <- polygon$labels_path
    annotation_manifest <- polygon$manifest_path
    labels <- polygon$labels
  } else if (!is.null(diagnostic_manifest)) {
    fallback <- read_fallback_labels(diagnostic_manifest, project_dir, cells)
    labels <- fallback$labels
    labels_path <- fallback$cluster_path
    annotation_manifest <- fallback$authority_path
  } else {
    fallback <- read_expanded_nogo_labels(expanded_decision, project_dir, cells)
    labels <- fallback$labels
    labels_path <- fallback$cluster_path
    annotation_manifest <- fallback$authority_path
  }
  rows <- make_annotation_rows(cells, labels, plate_contract)
  rows$suggested_cell_state_label <- as.character(rows$suggested_cell_state_label)
  all_unassigned <- !any(nzchar(rows$suggested_cell_state_label))
  if (!is.null(fallback) && !all_unassigned) abort("Fallback branch unexpectedly contains assigned labels")
  if (is.null(fallback) && all_unassigned) {
    abort("All-unassigned sampling may only be triggered by an authoritative fallback manifest")
  }
  selection <- if (all_unassigned) {
    result <- all_unassigned_fallback(rows, seed = 1L, max_total = 500L, max_per_group = 8L)
    result$design <- "disclosed_no_stable_cluster_adapter_global_max500_balanced_well_max8"
    result
  } else {
    stable_polygon_review(rows, seed = 1L, max_total = 500L, max_per_group = 8L)
  }
  selection$review$annotation_input_sha256 <- sha256_file(labels_path)
  template <- morphology_cell_state_review_label_template(selection$review)
  # Fail closed: defaults are suggestions, never manual evidence.
  template$reviewer <- ""
  template$reviewed_at <- ""
  template$label_confidence <- "high"
  if (any(classifier_source$api$morphology_cell_state_manual_review_evidence(template))) {
    abort("Seed1 template unexpectedly contains manual-review evidence")
  }
  if (nrow(template) > 500L ||
      (all_unassigned && any(table(template$source_id) > 8L))) {
    abort("Seed1 review violates its global max_total/max_per_group contract")
  }

  manifest <- list(
    schema_version = SCHEMA_VERSION, status = "HUMAN_REVIEW_REQUIRED",
    design = selection$design, seed = 1L, max_total = 500L, max_per_group = 8L,
    class_ids = as.list(CLASS_IDS), all_unassigned_fallback = all_unassigned,
    manual_training_policy = "reviewer_or_reviewed_at_required;defaults_umap_cluster_pseudo_never_train",
    context_key_policy = plate_contract$context_policy,
    context_keys = as.list(sort(unique(rows$context_key), method = "radix")),
    source_group_column = "source_id", source_id_semantics = "well",
    suffix_role = "nested_site_elapsed_time_metadata",
    frozen_well_split_manifest = plate_contract$path,
    frozen_well_split_manifest_sha256 = plate_contract$sha256,
    project = project_path, project_sha256 = sha256_file(project_path),
    selection_input_mode = if (is.null(fallback)) "accepted_polygon_annotation" else fallback$mode,
    historical_parity_scope = if (is.null(fallback)) {
      "pinned_reference_native_seed1_sampling"
    } else "historical_core_parity_with_sampling_adaptation",
    sampling_adaptation = if (is.null(fallback)) NULL else list(
      name = "disclosed_no_stable_cluster_adapter",
      reason = if (identical(fallback$mode, "authoritative_one_cluster_fallback")) {
        "pinned_seed1_sampler_requires_stable_polygon_assignments_but_projection_has_one_cluster_and_no_regions"
      } else {
        "expanded39_projection_failed_preregistered_cluster_size_and_stability_gates_so_polygon_regions_are_not_scientifically_supported"
      },
      rule = "global_max500_balanced_across_context_and_source_id_max8_per_well",
      reference_native_sampling_claimed = FALSE,
      manual_review_required_for_every_selected_row = TRUE
    ),
    annotation_import_dir = annotation_dir,
    annotation_manifest_sha256 = sha256_file(annotation_manifest),
    provisional_labels_sha256 = sha256_file(labels_path),
    diagnostic_cluster_manifest = diagnostic_manifest,
    diagnostic_cluster_manifest_sha256 = if (is.null(diagnostic_manifest)) NULL else sha256_file(diagnostic_manifest),
    expanded_labelability_decision = expanded_decision,
    expanded_labelability_decision_sha256 = if (is.null(expanded_decision)) NULL else sha256_file(expanded_decision),
    one_cluster_fallback_proven = !is.null(fallback) &&
      identical(fallback$mode, "authoritative_one_cluster_fallback"),
    computational_nogo_fallback_proven = !is.null(fallback) &&
      identical(fallback$mode, "authoritative_expanded_computational_nogo"),
    fallback_authority_schema_version = if (is.null(fallback)) NULL else fallback$authority_schema_version,
    dependency = dependency,
    historical_review_source_identity = review_source_identity,
    historical_classifier_source_identity = classifier_source$identity,
    implementation = script_path,
    implementation_sha256 = sha256_file(script_path),
    shared_implementation = shared_script_path,
    shared_implementation_sha256 = sha256_file(shared_script_path),
    row_count = nrow(template), stable_id_sha256 = digest::digest(
      paste(sort(stable_key(template)), collapse = "\n"), algo = "sha256", serialize = FALSE
    ),
    output_file_sha256 = NULL
  )
  if (dir.exists(output)) {
    verify_existing_seed1(output, manifest, selection)
    writeLines(sprintf("reference_cell_state_seed1_review_verified_reuse=1 rows=%d output=%s",
                       nrow(template), output))
    return(invisible(0L))
  }
  if (file.exists(output) || reference_cell_state_v2_is_symlink(output)) {
    abort("Seed1 output exists but is not a regular generation directory")
  }
  dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
  staging <- tempfile(paste0(".", basename(output), "-staging-"), tmpdir = dirname(output))
  if (!dir.create(staging)) abort("Could not create Seed1 staging directory")
  on.exit(unlink(staging, recursive = TRUE, force = TRUE), add = TRUE)
  paths <- c(
    review_set = file.path(staging, "seed1_review_set.tsv"),
    label_template = file.path(staging, "seed1_review_label_template.tsv"),
    selection_audit = file.path(staging, "seed1_selection_audit.tsv"),
    manifest = file.path(staging, "seed1_review_manifest.json")
  )
  write_tsv_atomic(selection$review, paths[["review_set"]])
  write_tsv_atomic(template, paths[["label_template"]])
  write_tsv_atomic(selection$audit, paths[["selection_audit"]])
  manifest$output_file_sha256 <- as.list(vapply(
    paths[names(paths) != "manifest"], sha256_file, character(1)
  ))
  temporary <- tempfile(".seed1-manifest-", tmpdir = staging)
  writeLines(jsonlite::toJSON(manifest, auto_unbox = TRUE, pretty = TRUE, null = "null"), temporary)
  if (!file.rename(temporary, paths[["manifest"]])) abort("Failed to install seed1 manifest")
  if (file.exists(output) || dir.exists(output) || reference_cell_state_v2_is_symlink(output)) {
    abort("Seed1 output appeared during staging")
  }
  if (!file.rename(staging, output)) abort("Failed to install Seed1 generation atomically")
  writeLines(sprintf("reference_cell_state_seed1_review_ready=1 rows=%d output=%s", nrow(template), output))
}

status <- tryCatch(main(), error = function(error) {
  writeLines(conditionMessage(error), con = stderr())
  1L
})
quit(save = "no", status = as.integer(status), runLast = FALSE)
