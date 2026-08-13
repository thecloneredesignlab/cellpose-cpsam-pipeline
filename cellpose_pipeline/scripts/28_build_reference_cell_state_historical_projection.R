#!/usr/bin/env Rscript

# Build the V2 reference-cell-state projection by selectively loading the
# audited historical functions.  The complete historical Utils.R is never
# sourced: it has unrelated top-level package and filesystem side effects.

options(stringsAsFactors = FALSE, warn = 1)

SCHEMA_VERSION <- "reference_cell_state_historical_projection_v2"
GENERATION_IDENTITY_SCHEMA_VERSION <- "reference_cell_state_historical_projection_generation_identity_v2"
GENERATION_IDENTITY_FILENAME <- "historical_projection_generation_identity.json"
CONFIG_SCHEMA_VERSION <- "reference_cell_state_feature_config_v2"
EXPECTED_SOURCE_SHA256 <- "9b913da2bce7e87de3c7e9b6f084502655fd9b6514ccff1a65478b37ea2a9828"
EXPECTED_FUNCTION_SHA256 <- c(
  impute_missing_values_knn = "bd74036e7260d6612aa0e31515b03f7b3295fa01027a02669e5beeb8d06b3466",
  morphology_projection_metadata_columns = "8131192c900a53498fcc86e50b965e8d21edadc4c1cd9f2551317621fb28b68e",
  plot_projection_by_passage = "7714064b8719a0ee626a40deaf4767a5b962bb849d173bbcd93488664d0d3693",
  optimize_dbscan_v4 = "b33a3551bb9e3c67ece03032d4d52c2fa49f34dd3ce9c6e4643b6c57590234c9",
  get_spatially_uniform_representatives = "84f029ec14db0dacf6d7a618c9ccb62792fe735872607d8956665f37ff9d5e72",
  get_all_cell_lines_overlay_representatives = "f46a8a203374527755996beddf1d1c86f8828d4838f7067ac17bc508f848a10e"
)
PROJECTION_FEATURES <- c(
  "area_px2", "perimeter_px", "roundness", "aspect_ratio", "extent",
  "solidity", "equivalent_diameter_px", "major_axis_px", "minor_axis_px"
)
CLASSIFIER_FEATURES <- c(
  PROJECTION_FEATURES,
  "bf_boundary_mean", "bf_interior_mean", "bf_interior_minus_boundary_mean"
)
HISTORICAL_FEATURES <- c(
  "Area.\u00b5m.2", "perimeter.\u00b5m", "roundness", "aspect_ratio", "extent",
  "solidity", "equi_diameter", "Major_Axis", "Minor_Axis"
)
REQUIRED_PACKAGES <- c(
  "jsonlite", "digest", "magrittr", "dplyr", "stringr", "ggplot2", "tidyr", "purrr",
  "uwot", "dbscan", "cluster"
)
OUTPUT_FILES <- c(
  "umap.tsv", "diagnostic_clusters.tsv", "fixed_dbscan_clusters.tsv",
  "dbscan_scan.tsv", "feature_transform_manifest.tsv", "pca_variance.tsv",
  "preprocessed_projection_features.tsv", "historical_representatives.tsv"
)

fail <- function(...) stop(paste0(...), call. = FALSE)

parse_cli <- function(argv) {
  out <- list(check_config = FALSE, overwrite = FALSE, dbscan_cores = 1L)
  index <- 1L
  while (index <= length(argv)) {
    token <- argv[[index]]
    if (token == "--check-config") {
      out$check_config <- TRUE
      index <- index + 1L
    } else if (token == "--overwrite") {
      out$overwrite <- TRUE
      index <- index + 1L
    } else if (token %in% c(
      "--project", "--reference-snapshot-root", "--output-dir",
      "--feature-config", "--dbscan-cores"
    )) {
      if (index == length(argv)) fail("Missing value after ", token)
      key <- gsub("-", "_", sub("^--", "", token))
      out[[key]] <- argv[[index + 1L]]
      index <- index + 2L
    } else {
      fail("Unknown argument: ", token)
    }
  }
  for (key in c("reference_snapshot_root", "feature_config")) {
    if (is.null(out[[key]]) || !nzchar(out[[key]])) fail("--", gsub("_", "-", key), " is required")
  }
  if (!out$check_config) {
    for (key in c("project", "output_dir")) {
      if (is.null(out[[key]]) || !nzchar(out[[key]])) fail("--", gsub("_", "-", key), " is required")
    }
  }
  cores <- suppressWarnings(as.integer(out$dbscan_cores))
  if (!is.finite(cores) || cores < 1L) fail("--dbscan-cores must be a positive integer")
  out$dbscan_cores <- cores
  out
}

sha256_file <- function(path) digest::digest(file = path, algo = "sha256", serialize = FALSE)
sha256_text <- function(value) digest::digest(value, algo = "sha256", serialize = FALSE)

current_script_path <- function() {
  argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(argument) != 1L) fail("Could not resolve this adapter implementation path")
  normalizePath(sub("^--file=", "", argument[[1L]]), mustWork = TRUE)
}

as_character_vector <- function(value, label) {
  result <- unname(unlist(value, use.names = FALSE))
  if (!is.character(result) || anyNA(result)) fail(label, " must be a string array")
  result
}

validate_config <- function(path) {
  if (!file.exists(path)) fail("Feature config does not exist: ", path)
  config <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  if (!identical(config$schema_version, CONFIG_SCHEMA_VERSION)) fail("Unsupported V2 feature config schema")
  if (!identical(as_character_vector(config$projection_feature_columns, "projection_feature_columns"), PROJECTION_FEATURES)) {
    fail("V2 projection feature order differs from the frozen nine-feature contract")
  }
  if (!identical(as_character_vector(config$classifier_feature_columns, "classifier_feature_columns"), CLASSIFIER_FEATURES)) {
    fail("V2 classifier feature order differs from the frozen twelve-feature contract")
  }
  mapping <- config$historical_reference_feature_mapping
  if (!is.list(mapping) || length(mapping) != length(PROJECTION_FEATURES)) fail("Historical feature mapping must have nine rows")
  current <- vapply(mapping, function(row) row$pipeline_feature, character(1))
  historical <- vapply(mapping, function(row) row$historical_feature, character(1))
  if (!identical(current, PROJECTION_FEATURES) || !identical(historical, HISTORICAL_FEATURES)) {
    fail("Historical feature mapping differs from the frozen exact-name mapping")
  }
  identity <- config$reference_identity_contract
  if (is.null(identity) ||
      !identical(identity$plate_map_sha256, "cb8aa2fa4a47cfa83d2f5462420575af0726be7066e47d8a60770543156de2e9") ||
      !identical(identity$cell_line_family, "SUM-159-NLS") ||
      !identical(as_character_vector(identity$contexts, "reference contexts"), c("SUM-159-NLS-2N", "SUM-159-NLS-4N")) ||
      !identical(identity$context_column, "context_key") || !identical(identity$source_id_column, "source_id") ||
      !identical(identity$source_id_semantics, "well") || !identical(identity$suffix_column, "suffix") ||
      !identical(identity$condition_column, "condition")) {
    fail("V2 reference identity config differs from the frozen SUM159 contract")
  }
  hp <- config$historical_projection
  exact <- list(
    reference_source_file = "code/lib/Utils.R",
    reference_source_sha256 = EXPECTED_SOURCE_SHA256,
    size_feature_regex = "Area|Axis|Perimeter|Diameter",
    size_feature_regex_ignore_case = FALSE,
    knn_k = 5L,
    pca_max_components = 10L,
    umap_seed = 42L,
    umap_n_neighbors = 15L,
    umap_n_components = 2L,
    unit_adaptation = "pixel_units_constant_scale_adaptation",
    pixel_calibration_evidence = paste0(
      "incucyte_tiff_generic_72_dpi_only_no_ome_imagej_or_description_",
      "and_repo_declares_no_valid_microscopy_pixel_calibration"
    ),
    historical_name_mapping_role = "case_sensitive_schema_dispatch_only_not_physical_unit_claim",
    unrecoverable_production_artifact_difference = paste0(
      "unknown_px_to_micrometre_scale_cannot_be_recovered_and_is_not_fully_removed_",
      "for_lowercase_perimeter.µm_and_equi_diameter_because_historical_log1p_precedes_zscore"
    )
  )
  for (key in names(exact)) {
    observed <- hp[[key]]
    if (is.numeric(exact[[key]])) observed <- as.integer(observed)
    if (!identical(observed, exact[[key]])) fail("Historical projection config drift: ", key)
  }
  db <- hp$authoritative_dbscan
  expected_db <- list(
    `function` = "optimize_dbscan_v4", eps_start = 0.005, eps_stop = 0.7,
    eps_step = 0.025, min_points_start = 100L, min_points_stop = 200L,
    min_points_step = 5L, max_noise_percent = 25L, max_clusters = 10L,
    silhouette_seed = 1L, silhouette_max_cells = 2000L,
    silhouette_fraction = 0.25, no_valid_candidate = "one_cluster_fallback"
  )
  for (key in names(expected_db)) {
    observed <- unlist(db[[key]], use.names = FALSE)
    expected <- expected_db[[key]]
    if (is.integer(expected)) observed <- as.integer(observed)
    if (is.numeric(expected) && !is.integer(expected)) observed <- as.numeric(observed)
    if (!identical(observed, expected)) fail("Historical DBSCAN optimizer config drift: ", key)
  }
  fixed <- hp$fixed_dbscan_audit
  if (!identical(as.numeric(fixed$eps), 0.5) || !identical(as.integer(fixed$min_points), 5L)) {
    fail("Historical fixed DBSCAN audit config drift")
  }
  representatives <- hp$representative_selection
  expected_representatives <- list(
    outer_function = "get_all_cell_lines_overlay_representatives",
    inner_function = "get_spatially_uniform_representatives",
    seed = 1L, total_n = 300L, balance = 0.2,
    minimum_cluster_representatives = 2L,
    context_column = "context_key",
    cluster_role = "sampling_and_overlay_metadata_only"
  )
  for (key in names(expected_representatives)) {
    observed <- unlist(representatives[[key]], use.names = FALSE)
    expected <- expected_representatives[[key]]
    if (is.integer(expected)) observed <- as.integer(observed)
    if (is.numeric(expected) && !is.integer(expected)) observed <- as.numeric(observed)
    if (!identical(observed, expected)) fail("Historical representative config drift: ", key)
  }
  configured_hashes <- unlist(hp$selective_ast_loader$function_sha256, use.names = TRUE)
  if (!identical(configured_hashes[names(EXPECTED_FUNCTION_SHA256)], EXPECTED_FUNCTION_SHA256)) {
    fail("Selective-loader function hashes differ from the frozen adapter contract")
  }
  config
}

require_dependencies <- function() {
  missing <- REQUIRED_PACKAGES[!vapply(REQUIRED_PACKAGES, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) fail("Missing required R package closure: ", paste(missing, collapse = ", "))
}

function_assignment <- function(expr) {
  is.call(expr) && length(expr) >= 3L && as.character(expr[[1L]]) %in% c("<-", "=") &&
    is.symbol(expr[[2L]]) && is.call(expr[[3L]]) && identical(as.character(expr[[3L]][[1L]]), "function")
}

load_reference_functions <- function(source_file, capture_env) {
  if (!file.exists(source_file)) fail("Pinned reference Utils.R does not exist: ", source_file)
  observed_source_hash <- sha256_file(source_file)
  if (!identical(observed_source_hash, EXPECTED_SOURCE_SHA256)) {
    fail("Pinned reference Utils.R SHA-256 mismatch: expected=", EXPECTED_SOURCE_SHA256, " observed=", observed_source_hash)
  }
  expressions <- as.list(parse(source_file, keep.source = TRUE))
  selected <- list()
  for (expr in expressions) {
    if (!function_assignment(expr)) next
    name <- as.character(expr[[2L]])
    if (name %in% names(EXPECTED_FUNCTION_SHA256)) selected[[name]] <- c(selected[[name]], list(expr))
  }
  counts <- vapply(names(EXPECTED_FUNCTION_SHA256), function(name) length(selected[[name]]), integer(1))
  if (any(counts != 1L)) fail("Pinned reference must define each selected function exactly once: ", paste(names(counts), counts, sep = "=", collapse = ","))
  observed_hashes <- vapply(names(EXPECTED_FUNCTION_SHA256), function(name) {
    sha256_text(paste(deparse(selected[[name]][[1L]], width.cutoff = 500L), collapse = "\n"))
  }, character(1))
  if (!identical(observed_hashes, EXPECTED_FUNCTION_SHA256)) {
    bad <- names(observed_hashes)[observed_hashes != EXPECTED_FUNCTION_SHA256]
    fail("Pinned historical function AST hash mismatch: ", paste(bad, collapse = ","))
  }

  # Bind the real package functions used by the selected AST definitions.
  # This avoids evaluating unrelated top-level Utils.R expressions while
  # preserving the selected historical dplyr/tidyr/purrr and R k-means path.
  env <- new.env(parent = baseenv())
  bindings <- list(
    `%>%` = magrittr::`%>%`, bind_rows = dplyr::bind_rows,
    select = dplyr::select, where = dplyr::where, mutate = dplyr::mutate,
    across = dplyr::across, any_of = dplyr::any_of,
    group_by = dplyr::group_by, summarise = dplyr::summarise,
    n = dplyr::n, left_join = dplyr::left_join, ungroup = dplyr::ungroup,
    nest = tidyr::nest, unnest = tidyr::unnest, map2 = purrr::map2,
    str_extract = stringr::str_extract,
    ggplot = ggplot2::ggplot, aes = ggplot2::aes, geom_point = ggplot2::geom_point,
    scale_color_brewer = ggplot2::scale_color_brewer, guides = ggplot2::guides,
    guide_legend = ggplot2::guide_legend, labs = ggplot2::labs,
    theme_minimal = ggplot2::theme_minimal,
    scale_color_viridis_c = ggplot2::scale_color_viridis_c,
    var = stats::var, median = stats::median, quantile = stats::quantile,
    scale = base::scale, prcomp = stats::prcomp, dist = stats::dist,
    complete.cases = stats::complete.cases, kmeans = stats::kmeans,
    chull = grDevices::chull
  )
  list2env(bindings, envir = env)
  env$mclapply <- function(X, FUN, ..., mc.cores = 1L) {
    value <- parallel::mclapply(X, FUN, ..., mc.cores = mc.cores)
    capture_env$last_mclapply <- value
    value
  }
  for (name in names(EXPECTED_FUNCTION_SHA256)) eval(selected[[name]][[1L]], envir = env)
  list(env = env, source_sha256 = observed_source_hash, function_sha256 = observed_hashes)
}

resolve_project_asset <- function(project_path, value, label) {
  if (!is.character(value) || length(value) != 1L || !nzchar(value)) fail("Project lacks ", label)
  path <- path.expand(value)
  if (!grepl("^/", path)) path <- file.path(dirname(project_path), path)
  normalizePath(path, mustWork = TRUE)
}

write_tsv <- function(value, path) {
  write.table(value, path, sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE, na = "")
}

output_hashes <- function(directory, names) {
  stats::setNames(vapply(names, function(name) sha256_file(file.path(directory, name)), character(1)), names)
}

build_generation_identity <- function(manifest_path, declared_outputs, implementation_sha256) {
  list(
    schema_version = GENERATION_IDENTITY_SCHEMA_VERSION,
    status = "COMPLETE",
    historical_projection_manifest_file = "historical_projection_manifest.json",
    historical_projection_manifest_sha256 = sha256_file(manifest_path),
    implementation_sha256 = implementation_sha256,
    output_file_sha256 = as.list(declared_outputs)
  )
}

validate_reuse <- function(
  output_dir, project_path, cells_path, features_path, config_path, source_file,
  implementation_path
) {
  manifest_path <- file.path(output_dir, "historical_projection_manifest.json")
  identity_path <- file.path(output_dir, GENERATION_IDENTITY_FILENAME)
  if (!file.exists(manifest_path) || !file.exists(identity_path)) {
    fail("Existing projection output is partial: complete manifest/identity missing: ", output_dir)
  }
  expected_files <- c(OUTPUT_FILES, "historical_projection_manifest.json", GENERATION_IDENTITY_FILENAME)
  observed_files <- list.files(output_dir, all.files = TRUE, no.. = TRUE)
  if (!identical(sort(observed_files), sort(expected_files))) {
    fail("Existing projection artifact set conflicts")
  }
  observed_paths <- file.path(output_dir, observed_files)
  if (any(file.info(observed_paths)$isdir) || any(nzchar(Sys.readlink(observed_paths)))) {
    fail("Existing projection artifacts must be plain files")
  }
  manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
  if (!identical(manifest$schema_version, SCHEMA_VERSION) || !identical(manifest$status, "COMPLETE")) fail("Existing projection manifest is not complete V2")
  expected_inputs <- list(
    project = list(path = project_path, sha256 = sha256_file(project_path)),
    cells = list(path = cells_path, sha256 = sha256_file(cells_path)),
    features = list(path = features_path, sha256 = sha256_file(features_path)),
    feature_config = list(path = config_path, sha256 = sha256_file(config_path)),
    reference_utils = list(path = source_file, sha256 = sha256_file(source_file)),
    implementation = list(path = implementation_path, sha256 = sha256_file(implementation_path))
  )
  if (!is.list(manifest$inputs)) fail("Existing projection lacks input identity")
  for (name in names(expected_inputs)) {
    if (!identical(manifest$inputs[[name]], expected_inputs[[name]])) {
      fail("Existing projection input path/hash conflicts: ", name)
    }
  }
  declared <- unlist(manifest$output_file_sha256, use.names = TRUE)
  if (!identical(names(declared), OUTPUT_FILES) || any(!grepl("^[0-9a-f]{64}$", declared))) {
    fail("Existing projection output hash map conflicts")
  }
  for (name in OUTPUT_FILES) {
    path <- file.path(output_dir, name)
    if (!file.exists(path) || !identical(sha256_file(path), declared[[name]])) fail("Existing projection output hash conflict: ", name)
  }
  selection <- manifest$representative_selection
  if (!is.list(selection) ||
      !identical(selection$output_file, "historical_representatives.tsv") ||
      !identical(selection$output_sha256, declared[["historical_representatives.tsv"]])) {
    fail("Existing historical representative manifest conflicts")
  }
  identity <- jsonlite::fromJSON(identity_path, simplifyVector = FALSE)
  expected_identity <- build_generation_identity(
    manifest_path, declared, sha256_file(implementation_path)
  )
  if (!identical(identity, expected_identity)) {
    fail("Existing projection generation identity or manifest conflicts")
  }
  invisible(TRUE)
}

main <- function() {
  args <- parse_cli(commandArgs(trailingOnly = TRUE))
  require_dependencies()
  implementation_path <- current_script_path()
  implementation_sha256 <- sha256_file(implementation_path)
  config_path <- normalizePath(args$feature_config, mustWork = TRUE)
  config <- validate_config(config_path)
  hp <- config$historical_projection
  snapshot_root <- normalizePath(args$reference_snapshot_root, mustWork = TRUE)
  source_file <- normalizePath(file.path(snapshot_root, config$historical_projection$reference_source_file), mustWork = TRUE)
  capture_env <- new.env(parent = emptyenv())
  loaded <- load_reference_functions(source_file, capture_env)
  if (args$check_config) {
    cat("historical_projection_config=PASS\n")
    cat("reference_utils_sha256=", loaded$source_sha256, "\n", sep = "")
    cat("selected_function_count=", length(loaded$function_sha256), "\n", sep = "")
    cat("required_packages=", paste(REQUIRED_PACKAGES, collapse = ","), "\n", sep = "")
    cat("implementation_sha256=", implementation_sha256, "\n", sep = "")
    return(invisible(0L))
  }

  project_path <- normalizePath(args$project, mustWork = TRUE)
  project <- jsonlite::fromJSON(project_path, simplifyVector = FALSE)
  if (!identical(project$schema_version, "cell_phenotype_annotator_project_v1")) fail("Unsupported project schema")
  if (!identical(project$project_id, "reference_cell_state_development_v2")) fail("Adapter accepts only the V2 reference project")
  identity <- config$reference_identity_contract
  projection <- project$projection
  if (!identical(projection$mode, "existing_umap")) fail("V2 project must use existing_umap")
  if (!is.null(projection$feature_columns)) fail("existing_umap project must not duplicate compute-only feature_columns")
  if (!identical(projection$coordinate_file, "historical_projection/umap.tsv")) fail("Project coordinate_file differs from canonical V2 path")
  cells_path <- resolve_project_asset(project_path, project$cells_file, "cells_file")
  features_path <- resolve_project_asset(project_path, project$features_file, "features_file")
  cells <- read.delim(cells_path, check.names = FALSE, stringsAsFactors = FALSE)
  features <- read.delim(features_path, check.names = FALSE, stringsAsFactors = FALSE)
  expected_fields <- c("cell_id", CLASSIFIER_FEATURES)
  if (!identical(names(features), expected_fields)) fail("V2 features.tsv must contain cell_id plus the exact classifier twelve in order")
  forbidden <- grep("(^|_)(dead|rgb|class|label|state|final|trajectory|prediction|probability|confidence|review|annotation)(_|$)", names(features), value = TRUE, ignore.case = TRUE)
  if (length(forbidden)) fail("Forbidden current/dead/RGB feature entered historical adapter: ", paste(forbidden, collapse = ","))
  if (anyNA(features$cell_id) || any(!nzchar(features$cell_id)) || anyDuplicated(features$cell_id)) fail("features.tsv has blank or duplicate cell_id")
  required_cell_fields <- c("cell_id", "context_key", "source_id")
  if (!all(required_cell_fields %in% names(cells))) fail("V2 cells.tsv lacks historical representative identity")
  if (!identical(as.character(cells$cell_id), as.character(features$cell_id))) fail("V2 cells/features order differs")
  if (!identical(sort(unique(as.character(cells$context_key))), c("SUM-159-NLS-2N", "SUM-159-NLS-4N"))) {
    fail("V2 representative selection requires the two frozen contexts")
  }
  for (name in CLASSIFIER_FEATURES) {
    value <- suppressWarnings(as.numeric(features[[name]]))
    if (any(!is.finite(value))) fail("features.tsv has nonfinite classifier feature: ", name)
    features[[name]] <- value
  }

  output_argument <- path.expand(args$output_dir)
  if (file.exists(output_argument) && !dir.exists(output_argument)) {
    fail("Projection output path exists and is not a directory: ", output_argument)
  }
  if (dir.exists(output_argument) && nzchar(Sys.readlink(output_argument))) {
    fail("Projection output directory must not be a symlink: ", output_argument)
  }
  output_dir <- normalizePath(output_argument, mustWork = FALSE)
  if (dir.exists(output_dir)) {
    if (!args$overwrite) fail("Projection output already exists: ", output_dir)
    validate_reuse(
      output_dir, project_path, cells_path, features_path, config_path,
      source_file, implementation_path
    )
    cat("historical_projection=", output_dir, "\n", sep = "")
    cat("generation_status=verified_reuse\n")
    return(invisible(0L))
  }
  parent <- dirname(output_dir)
  dir.create(parent, recursive = TRUE, showWarnings = FALSE)
  staging <- tempfile(pattern = paste0(".", basename(output_dir), ".staging."), tmpdir = parent)
  dir.create(staging)
  on.exit(if (dir.exists(staging)) unlink(staging, recursive = TRUE), add = TRUE)

  historical <- as.data.frame(lapply(PROJECTION_FEATURES, function(name) features[[name]]), check.names = FALSE)
  names(historical) <- HISTORICAL_FEATURES
  historical$source_id <- "A0"
  projection_result <- loaded$env$plot_projection_by_passage(
    list(reference_v2 = historical), method = "umap", seed = 42L,
    n_neighbors = 15L, n_components = 2L, n_pcs = 10L, cluster = FALSE,
    feature_columns = HISTORICAL_FEATURES
  )
  if (!identical(colnames(projection_result$umapinput), HISTORICAL_FEATURES)) fail("Historical helper did not retain the exact projection nine")
  if (!identical(colnames(projection_result$umapinputafterPreprocess), HISTORICAL_FEATURES)) fail("Historical helper dropped a configured feature after transform")
  layout <- as.data.frame(projection_result$umapLayout[, c("Dim1", "Dim2"), drop = FALSE])
  if (nrow(layout) != nrow(features) || any(!is.finite(as.matrix(layout)))) fail("Historical helper returned invalid UMAP coordinates")
  umap <- data.frame(cell_id = features$cell_id, Dim1 = layout$Dim1, Dim2 = layout$Dim2, check.names = FALSE)
  write_tsv(umap, file.path(staging, "umap.tsv"))

  capture_env$last_mclapply <- NULL
  optimized <- loaded$env$optimize_dbscan_v4(
    layout, eps_range = seq(0.005, 0.7, by = 0.025),
    minPts_range = seq(100, 200, by = 5), max_noise_pct = 25,
    maxCluster = 10, n_cores = args$dbscan_cores
  )
  scan_values <- capture_env$last_mclapply
  grid <- expand.grid(eps = seq(0.005, 0.7, by = 0.025), min_points = seq(100, 200, by = 5))
  if (!is.list(scan_values) || length(scan_values) != nrow(grid)) fail("Could not audit the exact optimize_dbscan_v4 grid")
  scan_rows <- lapply(seq_len(nrow(grid)), function(index) {
    value <- scan_values[[index]]
    data.frame(
      grid_index = index, eps = grid$eps[[index]], min_points = grid$min_points[[index]],
      eligible = !is.null(value), n_clusters = if (is.null(value)) NA else value$n_clusters,
      pct_noise = if (is.null(value)) NA else value$pct_noise,
      shannon_norm = if (is.null(value)) NA else value$shannon_norm,
      score = if (is.null(value)) NA else value$score,
      selected = if (is.null(value) || is.na(optimized$eps)) FALSE else isTRUE(all.equal(as.numeric(value$eps), as.numeric(optimized$eps))) && identical(as.integer(value$minPts), as.integer(optimized$minPts))
    )
  })
  scan <- do.call(rbind, scan_rows)
  write_tsv(scan, file.path(staging, "dbscan_scan.tsv"))
  optimized_cluster <- as.integer(optimized$cluster)
  if (length(optimized_cluster) != nrow(features)) fail("Optimized diagnostic cluster length mismatch")
  diagnostic <- data.frame(
    cell_id = features$cell_id, cluster = optimized_cluster,
    cluster_source = if (is.na(optimized$eps)) "optimize_dbscan_v4_one_cluster_fallback" else "optimize_dbscan_v4_selected",
    stringsAsFactors = FALSE
  )
  write_tsv(diagnostic, file.path(staging, "diagnostic_clusters.tsv"))
  fixed_cluster <- dbscan::dbscan(layout, eps = 0.5, minPts = 5)$cluster
  write_tsv(data.frame(cell_id = features$cell_id, cluster = as.integer(fixed_cluster)), file.path(staging, "fixed_dbscan_clusters.tsv"))

  representative_input <- data.frame(
    cell_id = as.character(features$cell_id), Dim1 = layout$Dim1, Dim2 = layout$Dim2,
    cluster = optimized_cluster, context_key = as.character(cells$context_key),
    source_id = as.character(cells$source_id), stringsAsFactors = FALSE,
    check.names = FALSE
  )
  set.seed(1L)
  historical_representatives <- loaded$env$get_all_cell_lines_overlay_representatives(
    representative_input, total_n = 300L, balance = 0.2
  )
  required_representative_fields <- c(
    "cell_id", "context_key", "source_id", "cluster", "Dim1", "Dim2"
  )
  if (!all(required_representative_fields %in% names(historical_representatives))) {
    fail("Historical representative AST returned an incomplete table")
  }
  representative_ids <- as.character(historical_representatives$cell_id)
  if (!length(representative_ids) || length(representative_ids) > 300L ||
      anyNA(representative_ids) || any(!nzchar(representative_ids)) ||
      anyDuplicated(representative_ids) ||
      !all(representative_ids %in% representative_input$cell_id)) {
    fail("Historical representative AST returned an invalid cell universe")
  }
  historical_representatives <- data.frame(
    selection_rank = seq_along(representative_ids),
    cell_id = representative_ids,
    context_key = as.character(historical_representatives$context_key),
    source_id = as.character(historical_representatives$source_id),
    cluster = as.integer(historical_representatives$cluster),
    Dim1 = as.numeric(historical_representatives$Dim1),
    Dim2 = as.numeric(historical_representatives$Dim2),
    stringsAsFactors = FALSE, check.names = FALSE
  )
  write_tsv(
    historical_representatives,
    file.path(staging, "historical_representatives.tsv")
  )

  raw <- as.data.frame(projection_result$umapinput, check.names = FALSE)
  processed <- as.matrix(projection_result$umapinputafterPreprocess)
  transformed <- raw
  size_names <- grep("Area|Axis|Perimeter|Diameter", names(raw), value = TRUE)
  other_names <- setdiff(names(raw), size_names)
  transformed[other_names] <- lapply(transformed[other_names], log1p)
  transform_rows <- lapply(seq_along(PROJECTION_FEATURES), function(index) data.frame(
    pipeline_feature = PROJECTION_FEATURES[[index]], historical_feature = HISTORICAL_FEATURES[[index]],
    unit_adaptation = "pixel_units_constant_scale_adaptation",
    transform = if (HISTORICAL_FEATURES[[index]] %in% size_names) "linear" else "log1p",
    case_sensitive_size_regex_match = HISTORICAL_FEATURES[[index]] %in% size_names,
    unit_scale_equivalence = if (HISTORICAL_FEATURES[[index]] %in% size_names) {
      "positive_constant_scale_removed_by_zscore_fixture_verified"
    } else {
      "not_claimed_after_log1p"
    },
    missing_fraction_raw = mean(is.na(raw[[HISTORICAL_FEATURES[[index]]]])),
    transformed_center = mean(transformed[[HISTORICAL_FEATURES[[index]]]], na.rm = TRUE),
    transformed_scale = stats::sd(transformed[[HISTORICAL_FEATURES[[index]]]], na.rm = TRUE),
    knn_k = 5L
  ))
  write_tsv(do.call(rbind, transform_rows), file.path(staging, "feature_transform_manifest.tsv"))
  pca <- stats::prcomp(processed, center = FALSE, scale. = FALSE)
  processed_table <- data.frame(cell_id = features$cell_id, processed, check.names = FALSE)
  write_tsv(
    processed_table,
    file.path(staging, "preprocessed_projection_features.tsv")
  )
  variance <- pca$sdev^2
  pca_rows <- data.frame(
    component = seq_along(variance), variance = variance,
    proportion = variance / sum(variance), cumulative_proportion = cumsum(variance / sum(variance)),
    used_for_umap = seq_along(variance) <= min(10L, ncol(pca$x))
  )
  write_tsv(pca_rows, file.path(staging, "pca_variance.tsv"))

  files <- OUTPUT_FILES
  fixed_valid <- fixed_cluster[fixed_cluster != 0L]
  manifest <- list(
    schema_version = SCHEMA_VERSION, status = "COMPLETE",
    authority = list(
      coordinate = "umap.tsv from selected historical plot_projection_by_passage AST",
      diagnostic_cluster = "diagnostic_clusters.tsv from optimize_dbscan_v4; metadata only",
      fixed_dbscan = "audit_only_not_authoritative",
      representatives = paste(
        "historical_representatives.tsv from set.seed(1) then selected historical",
        "get_all_cell_lines_overlay_representatives AST; authoritative for rendering"
      )
    ),
    inputs = list(
      project = list(path = project_path, sha256 = sha256_file(project_path)),
      cells = list(path = cells_path, sha256 = sha256_file(cells_path)),
      features = list(path = features_path, sha256 = sha256_file(features_path)),
      feature_config = list(path = config_path, sha256 = sha256_file(config_path)),
      reference_utils = list(path = source_file, sha256 = loaded$source_sha256),
      implementation = list(path = implementation_path, sha256 = implementation_sha256)
    ),
    reference_identity = list(
      plate_map_sha256 = identity$plate_map_sha256,
      cell_line_family = identity$cell_line_family,
      context_column = identity$context_column,
      contexts = identity$contexts,
      source_id_column = identity$source_id_column,
      suffix_column = identity$suffix_column,
      condition_column = identity$condition_column
    ),
    selective_source = list(
      full_source_evaluated = FALSE, selected_function_count = length(loaded$function_sha256),
      function_sha256 = as.list(loaded$function_sha256)
    ),
    features = list(
      projection_feature_columns = as.list(PROJECTION_FEATURES),
      historical_feature_columns = as.list(HISTORICAL_FEATURES),
      classifier_feature_columns_present_but_not_consumed = as.list(setdiff(CLASSIFIER_FEATURES, PROJECTION_FEATURES)),
      physical_unit_claim = "not_asserted",
      pixel_calibration_status = "unavailable_not_recoverable",
      pixel_calibration_evidence = hp$pixel_calibration_evidence,
      historical_name_mapping_role = hp$historical_name_mapping_role,
      unit_adaptation = "pixel_units_constant_scale_adaptation",
      unit_adaptation_scope = paste(
        "Positive constant-scale equivalence is fixture-verified only for case-sensitive",
        "linear size matches Area.µm.2/Major_Axis/Minor_Axis; it is not claimed for",
        "lowercase perimeter.µm/equi_diameter because the historical helper applies log1p."
      ),
      unrecoverable_production_artifact_difference = hp$unrecoverable_production_artifact_difference
    ),
    projection = list(
      row_count = nrow(features), seed = 42L, n_neighbors = 15L,
      knn_k = 5L, pca_components_used = min(10L, ncol(pca$x)),
      umap_components = 2L
    ),
    diagnostic_cluster = list(
      role = "metadata_only", function_name = "optimize_dbscan_v4",
      exact_grid_count = nrow(grid), eps = unname(optimized$eps),
      min_points = unname(optimized$minPts), score = unname(optimized$score),
      n_clusters = unname(optimized$n_clusters), pct_noise = unname(optimized$pct_noise),
      one_cluster_fallback = is.na(optimized$eps)
    ),
    fixed_dbscan_audit = list(
      eps = 0.5, min_points = 5L,
      n_clusters = length(unique(fixed_valid)), pct_noise = mean(fixed_cluster == 0L) * 100
    ),
    representative_selection = list(
      role = "authoritative_rendering_cell_list",
      outer_function_name = "get_all_cell_lines_overlay_representatives",
      inner_function_name = "get_spatially_uniform_representatives",
      seed = 1L, total_n = 300L, balance = 0.2,
      minimum_cluster_representatives = 2L,
      selected_count = nrow(historical_representatives),
      output_file = "historical_representatives.tsv",
      output_sha256 = sha256_file(file.path(staging, "historical_representatives.tsv"))
    ),
    blinding_contract = list(dead_channel = "not_read", combined_rgb = "not_read", current_classification = "not_read"),
    output_file_sha256 = as.list(output_hashes(staging, files))
  )
  manifest_path <- file.path(staging, "historical_projection_manifest.json")
  jsonlite::write_json(manifest, manifest_path, pretty = TRUE, auto_unbox = TRUE, null = "null")
  generation_identity <- build_generation_identity(
    manifest_path, output_hashes(staging, files), implementation_sha256
  )
  jsonlite::write_json(
    generation_identity, file.path(staging, GENERATION_IDENTITY_FILENAME),
    pretty = TRUE, auto_unbox = TRUE, null = "null"
  )
  if (!identical(sha256_file(implementation_path), implementation_sha256)) {
    fail("Historical projection adapter changed during execution")
  }
  if (file.exists(output_dir) || dir.exists(output_dir)) fail("Projection output appeared during staging: ", output_dir)
  if (!file.rename(staging, output_dir)) fail("Failed to install projection output atomically: ", output_dir)
  cat("historical_projection=", output_dir, "\n", sep = "")
  cat("coordinate_authority=", file.path(output_dir, "umap.tsv"), "\n", sep = "")
  cat("diagnostic_cluster_authority=optimized\n")
  cat("generation_status=created\n")
  invisible(0L)
}

tryCatch(main(), error = function(error) {
  message("ERROR: ", conditionMessage(error))
  quit(save = "no", status = 1L, runLast = FALSE)
})
