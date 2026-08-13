# Shared fail-closed helpers for the V2 historical reference-cell-state axis.
# This file is an adapter only.  Statistical fitting and probability inference
# remain the pinned reference implementation.

reference_cell_state_v2_class_ids <- function() {
  c("live_cell", "dead_cell", "multinucleated_cell")
}

reference_cell_state_v2_current_features <- function() {
  c(
    "area_px2", "perimeter_px", "roundness", "aspect_ratio", "extent",
    "solidity", "equivalent_diameter_px", "major_axis_px", "minor_axis_px",
    "bf_boundary_mean", "bf_interior_mean", "bf_interior_minus_boundary_mean"
  )
}

reference_cell_state_v2_historical_features <- function() {
  c(
    "Area.\u00b5m.2", "perimeter.\u00b5m", "roundness", "aspect_ratio", "extent",
    "solidity", "equi_diameter", "Major_Axis", "Minor_Axis",
    "candidate_boundary_mean", "candidate_interior_mean",
    "candidate_interior_minus_boundary_mean"
  )
}

reference_cell_state_v2_feature_map <- function() {
  stats::setNames(
    reference_cell_state_v2_historical_features(),
    reference_cell_state_v2_current_features()
  )
}

reference_cell_state_v2_abort <- function(format, ...) {
  stop(sprintf(format, ...), call. = FALSE)
}

reference_cell_state_v2_is_symlink <- function(path) {
  link <- Sys.readlink(path)
  length(link) == 1L && !is.na(link) && nzchar(link)
}

reference_cell_state_v2_normalize_output <- function(path, label = "output") {
  if (!grepl("^/", path)) reference_cell_state_v2_abort("%s must be absolute", label)
  ancestor <- path
  suffix <- character()
  while (!file.exists(ancestor) && !dir.exists(ancestor)) {
    parent <- dirname(ancestor)
    if (identical(parent, ancestor)) {
      reference_cell_state_v2_abort("%s has no existing ancestor", label)
    }
    suffix <- c(basename(ancestor), suffix)
    ancestor <- parent
  }
  normalized <- normalizePath(ancestor, winslash = "/", mustWork = TRUE)
  if (length(suffix)) normalized <- do.call(file.path, as.list(c(normalized, suffix)))
  normalized
}

reference_cell_state_v2_sha256_file <- function(path) {
  digest::digest(file = path, algo = "sha256")
}

reference_cell_state_v2_sha256_text <- function(text) {
  digest::digest(enc2utf8(paste(text, collapse = "\n")), algo = "sha256", serialize = FALSE)
}

reference_cell_state_v2_run_git <- function(root, ...) {
  output <- suppressWarnings(system2("git", c("-C", root, ...), stdout = TRUE, stderr = TRUE))
  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) {
    reference_cell_state_v2_abort("Unable to inspect pinned reference checkout")
  }
  trimws(paste(output, collapse = "\n"))
}

reference_cell_state_v2_verify_dependency <- function(root, lock_path) {
  lock <- utils::read.delim(
    lock_path, sep = "\t", header = TRUE, quote = "", comment.char = "",
    check.names = FALSE, stringsAsFactors = FALSE
  )
  row <- lock[lock$dependency == "cellphenotypeannotator", , drop = FALSE]
  if (nrow(row) != 1L ||
      !identical(row$execution_mode[[1L]], "private_read_only_source_checkout")) {
    reference_cell_state_v2_abort(
      "Dependency lock must contain one private read-only cellphenotypeannotator row"
    )
  }
  observed <- list(
    commit = reference_cell_state_v2_run_git(root, "rev-parse", "HEAD"),
    tree = reference_cell_state_v2_run_git(root, "rev-parse", "HEAD^{tree}"),
    status = reference_cell_state_v2_run_git(
      root, "status", "--porcelain", "--untracked-files=all"
    )
  )
  if (nzchar(observed$status) || !identical(observed$commit, row$commit[[1L]]) ||
      !identical(observed$tree, row$tree[[1L]])) {
    reference_cell_state_v2_abort("Pinned reference checkout differs from dependency lock")
  }
  license <- normalizePath(file.path(root, row$license_path[[1L]]), mustWork = TRUE)
  description <- normalizePath(file.path(root, "DESCRIPTION"), mustWork = TRUE)
  if (!identical(reference_cell_state_v2_sha256_file(license), row$license_sha256[[1L]]) ||
      !identical(reference_cell_state_v2_sha256_file(description), row$description_sha256[[1L]])) {
    reference_cell_state_v2_abort("Pinned reference checkout file hashes differ from dependency lock")
  }
  list(
    repository = row$repository[[1L]], commit = observed$commit, tree = observed$tree,
    lock_sha256 = reference_cell_state_v2_sha256_file(lock_path),
    execution_mode = row$execution_mode[[1L]]
  )
}

reference_cell_state_v2_ast_function <- function(path, name, environment) {
  expressions <- parse(file = path, keep.source = TRUE)
  matched <- Filter(function(expression) {
    is.call(expression) && length(expression) == 3L &&
      as.character(expression[[1L]]) %in% c("<-", "=") &&
      is.symbol(expression[[2L]]) && identical(as.character(expression[[2L]]), name) &&
      is.call(expression[[3L]]) && identical(as.character(expression[[3L]][[1L]]), "function")
  }, as.list(expressions))
  if (length(matched) != 1L) {
    reference_cell_state_v2_abort(
      "Expected exactly one top-level function %s in pinned source %s", name, path
    )
  }
  eval(matched[[1L]], envir = environment)
  function_value <- get(name, envir = environment, inherits = FALSE)
  list(
    name = name,
    source_file = path,
    source_file_sha256 = reference_cell_state_v2_sha256_file(path),
    function_sha256 = reference_cell_state_v2_sha256_text(deparse(function_value, width.cutoff = 500L))
  )
}

reference_cell_state_v2_load_historical_classifier <- function(root) {
  ltee_root <- file.path(root, "reference", "ltee-source")
  lib <- file.path(ltee_root, "code", "lib")
  files <- list(
    utils = file.path(lib, "Utils.R"),
    registry = file.path(lib, "ltee_image_registry.R"),
    candidate = file.path(lib, "morphology_candidate_features.R"),
    policy = file.path(lib, "morphology_feature_policy.R"),
    classifier = file.path(lib, "morphology_cell_state_classifier.R")
  )
  if (any(!vapply(files, file.exists, logical(1)))) {
    reference_cell_state_v2_abort("Pinned reference snapshot lacks historical classifier sources")
  }
  environment <- new.env(parent = globalenv())
  selected <- list(
    reference_cell_state_v2_ast_function(files$utils, "morphology_umap_row_key", environment),
    reference_cell_state_v2_ast_function(files$registry, "ltee_bind_feature_list", environment)
  )
  # These three files contain function declarations/configuration only and have
  # no package-attaching top-level libraries.  Utils.R is deliberately excluded.
  sys.source(files$candidate, envir = environment)
  sys.source(files$policy, envir = environment)
  sys.source(files$classifier, envir = environment)
  # The pinned classifier's require-helper guard calls exists() through vapply
  # without an explicit envir. In a clean Rscript process that historical code
  # therefore expects these four exact helper bindings in .GlobalEnv. Bind only
  # those functions, never overwrite a conflicting binding, and record their
  # hashes below. All classifier math remains in the private environment.
  compatibility_helpers <- c(
    "morphology_promoted_shape_feature_columns",
    "morphology_promoted_default_sidecar_feature_columns",
    "morphology_umap_row_key", "ltee_bind_feature_list"
  )
  for (name in compatibility_helpers) {
    value <- get(name, envir = environment, inherits = FALSE)
    if (exists(name, envir = .GlobalEnv, inherits = FALSE) &&
        !identical(get(name, envir = .GlobalEnv, inherits = FALSE), value)) {
      reference_cell_state_v2_abort("Conflicting global historical helper binding: %s", name)
    }
    assign(name, value, envir = .GlobalEnv)
  }
  required <- c(
    "morphology_cell_state_classifier_feature_config",
    "train_morphology_cell_state_classifier_workflow",
    "validate_morphology_cell_state_classifier_model_bundle",
    "predict_morphology_cell_state_probabilities",
    "morphology_cell_state_prediction_table",
    "morphology_candidate_normalize_raw_image",
    "morphology_candidate_boundary_interior_summary"
  )
  missing <- required[!vapply(required, exists, logical(1), envir = environment, inherits = FALSE)]
  if (length(missing)) {
    reference_cell_state_v2_abort("Pinned historical classifier API is incomplete: %s", paste(missing, collapse = ", "))
  }
  full_sources <- lapply(files[c("candidate", "policy", "classifier")], function(path) {
    list(path = path, sha256 = reference_cell_state_v2_sha256_file(path))
  })
  list(
    api = environment,
    identity = list(
      loading_policy = "AST_selected_Utils_and_registry_helpers;full_function_only_classifier_sources",
      selected_functions = selected,
      compatibility_global_helpers = as.list(stats::setNames(vapply(
        compatibility_helpers,
        function(name) reference_cell_state_v2_sha256_text(deparse(
          get(name, envir = environment, inherits = FALSE), width.cutoff = 500L
        )), character(1)
      ), compatibility_helpers)),
      full_sources = full_sources
    )
  )
}

reference_cell_state_v2_read_project <- function(path) {
  value <- if (requireNamespace("yaml", quietly = TRUE)) {
    yaml::read_yaml(path)
  } else {
    jsonlite::fromJSON(path, simplifyVector = FALSE)
  }
  if (!is.list(value)) reference_cell_state_v2_abort("Project is not a mapping")
  value
}

reference_cell_state_v2_read_table <- function(path, label) {
  rows <- utils::read.delim(
    path, sep = "\t", header = TRUE, quote = "", comment.char = "",
    check.names = FALSE, stringsAsFactors = FALSE
  )
  if (!is.data.frame(rows) || !nrow(rows)) reference_cell_state_v2_abort("%s is empty", label)
  rows
}

reference_cell_state_v2_require_columns <- function(rows, columns, label) {
  missing <- setdiff(columns, names(rows))
  if (length(missing)) {
    reference_cell_state_v2_abort("%s lacks columns: %s", label, paste(missing, collapse = ", "))
  }
}

reference_cell_state_v2_resolve_plate_contract <- function(project_path) {
  project_dir <- dirname(normalizePath(project_path, winslash = "/", mustWork = TRUE))
  parent_import_path <- normalizePath(
    file.path(project_dir, "parent_import_manifest.json"), winslash = "/", mustWork = TRUE
  )
  parent_import <- jsonlite::fromJSON(parent_import_path, simplifyVector = FALSE)
  projection_manifest_path <- normalizePath(
    as.character(parent_import$parent$projection_input_manifest),
    winslash = "/", mustWork = TRUE
  )
  projection_manifest <- jsonlite::fromJSON(projection_manifest_path, simplifyVector = FALSE)
  split_path <- normalizePath(
    as.character(projection_manifest$well_split_freeze), winslash = "/", mustWork = TRUE
  )
  split <- reference_cell_state_v2_read_table(split_path, "frozen well split manifest")
  expected_columns <- c(
    "well", "plate_row", "plate_column", "doxorubicin_nm", "ploidy",
    "cyclophosphamide", "replicate", "split", "split_strategy", "split_seed",
    "assignment_sha256"
  )
  if (!identical(names(split), expected_columns)) {
    reference_cell_state_v2_abort("Frozen well split schema changed")
  }
  if (nrow(split) != 80L || anyDuplicated(split$well) ||
      !setequal(as.character(split$split), c("development", "heldout")) ||
      sum(split$split == "development") != 64L || sum(split$split == "heldout") != 16L ||
      any(!as.character(split$ploidy) %in% c("2N", "4N"))) {
    reference_cell_state_v2_abort("Frozen well split has duplicate wells or unsupported ploidy")
  }
  expected_hash <- projection_manifest$well_split_freeze_sha256
  parent_hash <- parent_import$parent$input_file_sha256$projection_input_manifest
  if (!identical(reference_cell_state_v2_sha256_file(split_path), as.character(expected_hash)) ||
      !identical(reference_cell_state_v2_sha256_file(projection_manifest_path), as.character(parent_hash))) {
    reference_cell_state_v2_abort("Frozen well split no longer matches parent-import hash")
  }
  project <- reference_cell_state_v2_read_project(project_path)
  project_cells <- reference_cell_state_v2_read_table(
    normalizePath(file.path(project_dir, project$cells_file), mustWork = TRUE), "project cells"
  )
  represented_wells <- sort(unique(as.character(project_cells$well)), method = "radix")
  development_wells <- sort(as.character(split$well[split$split == "development"]), method = "radix")
  if (!setequal(represented_wells, development_wells)) {
    reference_cell_state_v2_abort("V2 project well universe must equal the 64 frozen development wells")
  }
  if (length(unique(split$split_strategy)) != 1L || length(unique(split$split_seed)) != 1L ||
      any(!grepl("^[0-9a-f]{64}$", split$assignment_sha256))) {
    reference_cell_state_v2_abort("Frozen split strategy/seed/assignment identity is invalid")
  }
  split$context_key <- paste0("SUM-159-NLS-", split$ploidy)
  split$ltee_cell_line <- split$context_key
  split$cell_line_family <- "SUM-159-NLS"
  split$condition <- paste0(
    "doxorubicin_nm=", split$doxorubicin_nm,
    "|ploidy=", split$ploidy,
    "|cyclophosphamide=", tolower(as.character(split$cyclophosphamide))
  )
  list(
    rows = split,
    path = split_path,
    sha256 = reference_cell_state_v2_sha256_file(split_path),
    parent_import_path = parent_import_path,
    parent_import_sha256 = reference_cell_state_v2_sha256_file(parent_import_path),
    projection_input_manifest = projection_manifest_path,
    projection_input_manifest_sha256 = reference_cell_state_v2_sha256_file(projection_manifest_path),
    split_strategy = unique(as.character(split$split_strategy)),
    split_seed = unique(as.character(split$split_seed)),
    development_well_count = 64L,
    heldout_well_count = 16L,
    context_policy = "plate_ploidy_2N_4N_to_SUM-159-NLS-context",
    source_group_policy = "well_is_independent_source_id",
    suffix_policy = "site_plus_elapsed_time_nested_technical_metadata"
  )
}

reference_cell_state_v2_validate_canonical_cells <- function(cells, plate_contract) {
  required <- c(
    "cell_id", "well", "site", "elapsed_hours", "mask_label",
    "context_key", "ltee_cell_line", "ltee_cell_line_family", "source_id",
    "suffix", "condition", "feature_row_index", "segmentation_qc_cell_id",
    "segmentation_object_id"
  )
  reference_cell_state_v2_require_columns(cells, required, "V2 canonical cells.tsv")
  if (anyDuplicated(cells$cell_id) || any(!nzchar(as.character(cells$cell_id)))) {
    reference_cell_state_v2_abort("V2 canonical cells.tsv has blank or duplicate cell_id")
  }
  index <- match(as.character(cells$well), as.character(plate_contract$rows$well))
  if (anyNA(index)) reference_cell_state_v2_abort("V2 canonical cells.tsv contains an unknown well")
  plate <- plate_contract$rows[index, , drop = FALSE]
  expected_context <- as.character(plate$context_key)
  expected_condition <- as.character(plate$condition)
  if (any(as.character(cells$source_id) != as.character(cells$well)) ||
      any(as.character(cells$context_key) != expected_context) ||
      any(as.character(cells$ltee_cell_line) != expected_context) ||
      any(as.character(cells$ltee_cell_line_family) != "SUM-159-NLS") ||
      any(as.character(cells$condition) != expected_condition) ||
      any(as.character(cells$segmentation_qc_cell_id) != as.character(cells$cell_id)) ||
      any(as.character(cells$segmentation_object_id) != as.character(cells$cell_id))) {
    reference_cell_state_v2_abort("V2 canonical cells.tsv historical identity disagrees with the frozen plate contract")
  }
  site <- suppressWarnings(as.integer(cells$site))
  elapsed <- suppressWarnings(as.numeric(cells$elapsed_hours))
  mask_label <- suppressWarnings(as.integer(cells$mask_label))
  feature_row_index <- suppressWarnings(as.integer(cells$feature_row_index))
  suffix_match <- regexec("^site=([1-9][0-9]*)[|]elapsed_hours=(.+)$", as.character(cells$suffix))
  suffix_parts <- regmatches(as.character(cells$suffix), suffix_match)
  valid_suffix <- lengths(suffix_parts) == 3L
  suffix_site <- rep(NA_integer_, nrow(cells))
  suffix_elapsed <- rep(NA_real_, nrow(cells))
  suffix_site[valid_suffix] <- suppressWarnings(as.integer(vapply(suffix_parts[valid_suffix], `[[`, character(1), 2L)))
  suffix_elapsed[valid_suffix] <- suppressWarnings(as.numeric(vapply(suffix_parts[valid_suffix], `[[`, character(1), 3L)))
  if (anyNA(site) || anyNA(elapsed) || anyNA(mask_label) || anyNA(feature_row_index) ||
      any(site < 1L) || any(mask_label < 1L) || any(feature_row_index < 1L) ||
      any(!valid_suffix) || any(suffix_site != site) ||
      any(!is.finite(suffix_elapsed)) || any(suffix_elapsed != elapsed)) {
    reference_cell_state_v2_abort("V2 canonical cells.tsv suffix or numeric identity is invalid")
  }
  group <- interaction(as.character(cells$source_id), as.character(cells$suffix),
                       drop = TRUE, lex.order = TRUE)
  expected_index <- integer(nrow(cells))
  for (members in split(seq_len(nrow(cells)), group)) {
    ordered <- members[order(mask_label[members], as.character(cells$cell_id[members]), method = "radix")]
    expected_index[ordered] <- seq_along(ordered)
  }
  if (!identical(feature_row_index, expected_index)) {
    reference_cell_state_v2_abort(
      "V2 canonical cells.tsv feature_row_index must reset within each source_id/suffix after mask_label/cell_id ordering"
    )
  }
  cells
}

reference_cell_state_v2_apply_plate_identity <- function(rows, plate_contract) {
  reference_cell_state_v2_require_columns(rows, c("well"), "rows requiring plate identity")
  index <- match(as.character(rows$well), as.character(plate_contract$rows$well))
  if (anyNA(index)) {
    reference_cell_state_v2_abort(
      "Rows contain well(s) absent from frozen plate contract: %s",
      paste(utils::head(unique(as.character(rows$well[is.na(index)])), 5L), collapse = ",")
    )
  }
  plate <- plate_contract$rows[index, , drop = FALSE]
  rows$context_key <- as.character(plate$context_key)
  rows$ltee_cell_line <- as.character(plate$ltee_cell_line)
  rows$cell_line <- rows$ltee_cell_line
  rows$cell_line_family <- as.character(plate$cell_line_family)
  rows$source_id <- as.character(rows$well)
  rows$condition <- as.character(plate$condition)
  rows$ploidy <- as.character(plate$ploidy)
  rows$doxorubicin_nm <- as.character(plate$doxorubicin_nm)
  rows$cyclophosphamide <- as.character(plate$cyclophosphamide)
  rows$replicate <- as.character(plate$replicate)
  rows
}

reference_cell_state_v2_map_features <- function(features, cells, plate_contract) {
  current <- reference_cell_state_v2_current_features()
  historical <- reference_cell_state_v2_historical_features()
  reference_cell_state_v2_require_columns(features, c("cell_id", current), "V2 feature table")
  if (anyDuplicated(features$cell_id) || any(!nzchar(as.character(features$cell_id)))) {
    reference_cell_state_v2_abort("V2 feature table has blank or duplicate cell_id")
  }
  rows <- features
  names(rows)[match(current, names(rows))] <- historical
  cells <- reference_cell_state_v2_validate_canonical_cells(cells, plate_contract)
  if (!setequal(rows$cell_id, cells$cell_id)) {
    reference_cell_state_v2_abort("Feature/cell stable-ID universes are not one-to-one")
  }
  cells <- cells[match(rows$cell_id, cells$cell_id), , drop = FALSE]
  rows$well <- as.character(cells$well)
  rows$site <- as.character(cells$site)
  rows$elapsed_hours <- as.numeric(cells$elapsed_hours)
  rows$mask_label <- as.integer(cells$mask_label)
  rows$morphology_umap_row_key <- as.character(cells$cell_id)
  rows$stable_cell_id <- as.character(cells$cell_id)
  rows$segmentation_qc_cell_id <- as.character(cells$segmentation_qc_cell_id)
  rows$segmentation_object_id <- as.character(cells$segmentation_object_id)
  rows$context_key <- as.character(cells$context_key)
  rows$ltee_cell_line <- as.character(cells$ltee_cell_line)
  rows$cell_line <- rows$ltee_cell_line
  rows$cell_line_family <- as.character(cells$ltee_cell_line_family)
  rows$source_id <- as.character(cells$source_id)
  rows$suffix <- as.character(cells$suffix)
  rows$condition <- as.character(cells$condition)
  rows$ploidy <- as.character(plate_contract$rows$ploidy[
    match(rows$well, plate_contract$rows$well)
  ])
  rows$feature_row_index <- as.integer(cells$feature_row_index)
  rows$passage_number <- 0L
  rows
}

reference_cell_state_v2_validate_bf_equivalence <- function(api) {
  raw <- matrix(c(
    0, 5, 10, 15, 20,
    2, 7, 12, 17, 22,
    4, 9, 14, 19, 24,
    6, 11, 16, 21, 26,
    8, 13, 18, 23, 28
  ), nrow = 5L, byrow = TRUE)
  mask <- matrix(FALSE, 5L, 5L)
  mask[2:4, 2:4] <- TRUE
  historical_normalized <- api$morphology_candidate_normalize_raw_image(raw)$matrix
  current_normalized <- raw / max(raw)
  historical <- api$morphology_candidate_boundary_interior_summary(
    historical_normalized, mask
  )$values
  up <- rbind(FALSE, mask[-nrow(mask), , drop = FALSE])
  down <- rbind(mask[-1L, , drop = FALSE], FALSE)
  left <- cbind(FALSE, mask[, -ncol(mask), drop = FALSE])
  right <- cbind(mask[, -1L, drop = FALSE], FALSE)
  boundary <- mask & !(up & down & left & right)
  interior <- mask & !boundary
  current <- c(
    candidate_boundary_mean = mean(current_normalized[boundary]),
    candidate_interior_mean = mean(current_normalized[interior]),
    candidate_interior_minus_boundary_mean =
      mean(current_normalized[interior]) - mean(current_normalized[boundary])
  )
  error <- max(abs(as.numeric(historical) - as.numeric(current)))
  if (!is.finite(error) || error > 1e-12) {
    reference_cell_state_v2_abort(
      "Current/reference BF boundary-interior fixture differs: max_abs_error=%.17g", error
    )
  }
  list(
    status = "PASS", tolerance = 1e-12, max_abs_error = error,
    normalization = "already_0_1_or_divide_by_input_max_or_minmax",
    boundary = "four_neighbor_binary_erosion_equivalent",
    fixture_sha256 = reference_cell_state_v2_sha256_text(capture.output(dput(list(raw = raw, mask = mask))))
  )
}

reference_cell_state_v2_write_tsv_atomic <- function(rows, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(paste0(".", basename(path), "-"), tmpdir = dirname(path))
  on.exit(unlink(temporary), add = TRUE)
  utils::write.table(rows, temporary, sep = "\t", quote = FALSE, row.names = FALSE, na = "")
  if (!file.rename(temporary, path)) reference_cell_state_v2_abort("Failed to install %s", path)
}

reference_cell_state_v2_write_json_atomic <- function(value, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(paste0(".", basename(path), "-"), tmpdir = dirname(path))
  on.exit(unlink(temporary), add = TRUE)
  writeLines(jsonlite::toJSON(value, auto_unbox = TRUE, pretty = TRUE, null = "null"), temporary)
  if (!file.rename(temporary, path)) reference_cell_state_v2_abort("Failed to install %s", path)
}
