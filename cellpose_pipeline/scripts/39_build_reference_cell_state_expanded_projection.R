#!/usr/bin/env Rscript

# Compare three reference-cell-state annotation geometries on the exact same
# development-cell universe.  The historical projection and diagnostic DBSCAN
# functions are loaded from the pinned reference snapshot through the audited
# selective loader in script 28.  Expanded features are annotation evidence
# only: the final historical classifier remains the frozen classifier12 model.

options(stringsAsFactors = FALSE, warn = 1)

SCHEMA_VERSION <- "reference_cell_state_expanded_projection_comparison_v1"
IDENTITY_SCHEMA_VERSION <- "reference_cell_state_expanded_projection_generation_identity_v1"
CONFIG_SCHEMA_VERSION <- "reference_cell_state_projection_expanded_config_v1"
PROJECT_SCHEMA_VERSION <- "cell_phenotype_annotator_project_v1"
PROJECT_ID <- "reference_cell_state_development_v2"
EXPECTED_REFERENCE_UTILS_SHA256 <- "9b913da2bce7e87de3c7e9b6f084502655fd9b6514ccff1a65478b37ea2a9828"

SHAPE9 <- c(
  "area_px2", "perimeter_px", "roundness", "aspect_ratio", "extent",
  "solidity", "equivalent_diameter_px", "major_axis_px", "minor_axis_px"
)
CLASSIFIER12 <- c(
  SHAPE9, "bf_boundary_mean", "bf_interior_mean",
  "bf_interior_minus_boundary_mean"
)
EXPANDED39 <- c(
  SHAPE9,
  "bf_object_mean", "bf_object_median", "bf_object_sd", "bf_object_mad",
  "bf_object_p10", "bf_object_p90", "bf_object_iqr",
  "bf_object_mean_minus_global_background",
  "bf_object_mean_over_global_background",
  "bf_object_median_minus_global_background",
  "bf_object_mean_minus_ring_background",
  "bf_object_mean_over_ring_background",
  "bf_boundary_mean", "bf_interior_mean", "bf_interior_minus_boundary_mean",
  "bf_boundary_gradient_mean", "bf_boundary_gradient_p95",
  "bf_glcm_contrast", "bf_glcm_entropy", "bf_glcm_asm", "bf_glcm_idm",
  "bf_glcm_correlation", "bf_tenengrad_mean", "bf_tenengrad_p95",
  "bf_laplacian_variance", "bf_sobel_magnitude_mean", "bf_edge_density",
  "nuclei_count", "nuclei_overlap_area_px2", "nuclei_area_fraction"
)
PROFILE_FEATURES <- list(
  shape9 = SHAPE9,
  classifier12 = CLASSIFIER12,
  expanded39 = EXPANDED39
)
EXCLUDED_FEATURES <- c(
  "bf_object_robust_z_global_background",
  "bf_object_iqr_over_background_iqr"
)
SHAPE_HISTORICAL_NAMES <- c(
  area_px2 = "Area.\u00b5m.2",
  perimeter_px = "perimeter.\u00b5m",
  roundness = "roundness",
  aspect_ratio = "aspect_ratio",
  extent = "extent",
  solidity = "solidity",
  equivalent_diameter_px = "equi_diameter",
  major_axis_px = "Major_Axis",
  minor_axis_px = "Minor_Axis"
)
FEATURE_BLOCKS <- list(
  shape = SHAPE9,
  brightfield = setdiff(EXPANDED39, c(SHAPE9, "nuclei_count", "nuclei_overlap_area_px2", "nuclei_area_fraction")),
  nuclei_support = c("nuclei_count", "nuclei_overlap_area_px2", "nuclei_area_fraction")
)
PROFILE_OUTPUTS <- c(
  "umap.tsv", "diagnostic_clusters.tsv", "fixed_dbscan_clusters.tsv",
  "dbscan_scan.tsv", "feature_transform_manifest.tsv", "feature_statistics.tsv",
  "pca_variance.tsv", "pca_feature_contribution.tsv", "pca_block_contribution.tsv",
  "preprocessed_projection_features.tsv", "cluster_summary.tsv",
  "bootstrap_stability.tsv", "parameter_neighborhood_stability.tsv"
)

fail <- function(...) stop(paste0(...), call. = FALSE)
sha256_file <- function(path) digest::digest(file = path, algo = "sha256", serialize = FALSE)
write_tsv <- function(value, path) {
  write.table(value, path, sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE, na = "")
}

parse_cli <- function(argv) {
  out <- list(overwrite = FALSE, check_config = FALSE, dbscan_cores = 1L)
  index <- 1L
  while (index <= length(argv)) {
    token <- argv[[index]]
    if (token == "--overwrite") {
      out$overwrite <- TRUE
      index <- index + 1L
    } else if (token == "--check-config") {
      out$check_config <- TRUE
      index <- index + 1L
    } else if (token %in% c(
      "--project", "--broad-features", "--reference-snapshot-root",
      "--feature-config", "--historical-adapter-script", "--output-dir",
      "--dbscan-cores"
    )) {
      if (index == length(argv)) fail("Missing value after ", token)
      key <- gsub("-", "_", sub("^--", "", token))
      out[[key]] <- argv[[index + 1L]]
      index <- index + 2L
    } else {
      fail("Unknown argument: ", token)
    }
  }
  for (key in c("reference_snapshot_root", "feature_config", "historical_adapter_script")) {
    if (is.null(out[[key]]) || !nzchar(out[[key]])) fail("--", gsub("_", "-", key), " is required")
  }
  if (!out$check_config) {
    for (key in c("project", "broad_features", "output_dir")) {
      if (is.null(out[[key]]) || !nzchar(out[[key]])) fail("--", gsub("_", "-", key), " is required")
    }
  }
  out$dbscan_cores <- suppressWarnings(as.integer(out$dbscan_cores))
  if (!is.finite(out$dbscan_cores) || out$dbscan_cores < 1L) {
    fail("--dbscan-cores must be a positive integer")
  }
  out
}

current_script_path <- function() {
  argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(argument) != 1L) fail("Could not resolve expanded projection implementation path")
  normalizePath(sub("^--file=", "", argument[[1L]]), mustWork = TRUE)
}

as_character_vector <- function(value, label) {
  result <- unname(unlist(value, use.names = FALSE))
  if (!is.character(result) || anyNA(result)) fail(label, " must be a string array")
  result
}

validate_config <- function(path) {
  config <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  if (!identical(config$schema_version, CONFIG_SCHEMA_VERSION)) fail("Unsupported expanded projection config schema")
  universe <- config$cell_universe
  expected_cell_count <- suppressWarnings(as.integer(universe$expected_cell_count))
  if (!is.finite(expected_cell_count) || expected_cell_count < 1L ||
      !identical(universe$split, "development") ||
      !identical(as_character_vector(universe$expected_contexts, "expected contexts"), c("SUM-159-NLS-2N", "SUM-159-NLS-4N")) ||
      !identical(universe$source_id_semantics, "well") ||
      !identical(universe$heldout_policy, "not_read")) {
    fail("Expanded projection cell-universe contract differs")
  }
  if (!identical(config$selected_annotation_profile, "expanded39")) fail("expanded39 must remain the selected annotation profile")
  configured_profiles <- config$profiles
  if (!is.list(configured_profiles) || !identical(names(configured_profiles), names(PROFILE_FEATURES))) {
    fail("Expanded projection profile names/order differ from the frozen contract")
  }
  for (name in names(PROFILE_FEATURES)) {
    observed <- as_character_vector(configured_profiles[[name]]$feature_columns, paste0(name, " feature_columns"))
    if (!identical(observed, PROFILE_FEATURES[[name]])) fail("Expanded projection feature order differs: ", name)
  }
  excluded <- vapply(config$excluded_feature_columns, function(row) row$feature, character(1))
  if (!identical(excluded, EXCLUDED_FEATURES)) fail("Expanded projection exclusion list differs")
  blocks <- lapply(config$feature_blocks, as_character_vector, label = "feature block")
  if (!identical(blocks, FEATURE_BLOCKS)) fail("Expanded projection feature block assignment differs")
  hp <- config$historical_projection_contract
  expected_hp <- list(
    reference_source_file = "code/lib/Utils.R",
    reference_source_sha256 = EXPECTED_REFERENCE_UTILS_SHA256,
    historical_adapter_script = "28_build_reference_cell_state_historical_projection.R",
    size_feature_regex = "Area|Axis|Perimeter|Diameter",
    size_feature_regex_ignore_case = FALSE,
    non_size_transform = "log1p",
    scaling = "zscore_R_scale",
    imputation = "reference_impute_missing_values_knn",
    knn_k = 5L,
    pca_max_components = 10L,
    umap_seed = 42L,
    umap_n_neighbors = 15L,
    umap_n_components = 2L,
    umap_metric = "euclidean",
    diagnostic_cluster = "reference_optimize_dbscan_v4",
    diagnostic_cluster_role = "navigation_sampling_and_labelability_audit_only_not_class_label"
  )
  for (key in names(expected_hp)) {
    observed <- unlist(hp[[key]], use.names = FALSE)
    expected <- expected_hp[[key]]
    if (is.integer(expected)) observed <- as.integer(observed)
    if (!identical(observed, expected)) fail("Historical expanded projection contract drift: ", key)
  }
  gate <- config$labelability_gate
  expected_gate <- list(
    minimum_non_noise_clusters = 2L,
    minimum_cluster_fraction = 0.01,
    maximum_noise_fraction = 0.25,
    bootstrap_replicates = 20L,
    bootstrap_fraction = 0.8,
    bootstrap_seed = 20260813L,
    minimum_bootstrap_median_adjusted_rand = 0.7,
    parameter_neighborhood_eps_step = 0.025,
    parameter_neighborhood_min_points_step = 5L,
    minimum_parameter_neighborhood_median_adjusted_rand = 0.7,
    minimum_parameter_neighborhood_adjusted_rand = 0.5,
    require_no_one_cluster_neighbor = TRUE,
    morphology_overlay_review = "required_before_GO"
  )
  for (key in names(expected_gate)) {
    observed <- unlist(gate[[key]], use.names = FALSE)
    expected <- expected_gate[[key]]
    if (is.integer(expected)) observed <- as.integer(observed)
    if (is.numeric(expected) && !is.integer(expected)) observed <- as.numeric(observed)
    if (!identical(observed, expected)) fail("Labelability gate drift: ", key)
  }
  classifier <- config$classifier_contract
  if (!identical(as_character_vector(classifier$feature_columns_source, "classifier feature source"), "classifier12") ||
      !identical(unlist(classifier$expanded_features_allowed_in_classifier, use.names = FALSE), FALSE)) {
    fail("Expanded annotation features must remain excluded from the historical classifier")
  }
  config
}

load_projection_library <- function(path) {
  adapter_path <- normalizePath(path, mustWork = TRUE)
  previous <- Sys.getenv("REFERENCE_CELL_STATE_HISTORICAL_PROJECTION_LIBRARY", unset = NA_character_)
  Sys.setenv(REFERENCE_CELL_STATE_HISTORICAL_PROJECTION_LIBRARY = "1")
  on.exit({
    if (is.na(previous)) Sys.unsetenv("REFERENCE_CELL_STATE_HISTORICAL_PROJECTION_LIBRARY") else
      Sys.setenv(REFERENCE_CELL_STATE_HISTORICAL_PROJECTION_LIBRARY = previous)
  }, add = TRUE)
  env <- new.env(parent = globalenv())
  sys.source(adapter_path, envir = env)
  required <- c("require_dependencies", "load_reference_functions", "resolve_project_asset")
  missing <- required[!vapply(required, exists, logical(1), envir = env, inherits = FALSE)]
  if (length(missing)) fail("Historical adapter library API is incomplete: ", paste(missing, collapse = ","))
  list(path = adapter_path, sha256 = sha256_file(adapter_path), env = env)
}

historical_names <- function(features) {
  unname(vapply(features, function(name) {
    if (name %in% names(SHAPE_HISTORICAL_NAMES)) SHAPE_HISTORICAL_NAMES[[name]] else name
  }, character(1)))
}

feature_block <- function(name) {
  matched <- names(FEATURE_BLOCKS)[vapply(FEATURE_BLOCKS, function(values) name %in% values, logical(1))]
  if (length(matched) != 1L) fail("Feature block assignment is not unique: ", name)
  matched[[1L]]
}

choose2 <- function(value) value * (value - 1) / 2

adjusted_rand_index <- function(left, right) {
  if (length(left) != length(right) || !length(left)) fail("ARI inputs must be nonempty and equal length")
  table_value <- table(left, right)
  total <- sum(table_value)
  if (total < 2L) return(1)
  sum_cells <- sum(choose2(table_value))
  sum_rows <- sum(choose2(rowSums(table_value)))
  sum_columns <- sum(choose2(colSums(table_value)))
  total_pairs <- choose2(total)
  expected <- sum_rows * sum_columns / total_pairs
  denominator <- 0.5 * (sum_rows + sum_columns) - expected
  if (abs(denominator) < .Machine$double.eps) {
    return(if (identical(as.integer(factor(left)), as.integer(factor(right)))) 1 else 0)
  }
  (sum_cells - expected) / denominator
}

cluster_metrics <- function(cluster) {
  cluster <- as.integer(cluster)
  assigned <- cluster[cluster != 0L]
  counts <- sort(table(assigned), decreasing = TRUE)
  list(
    n_clusters = length(counts),
    noise_fraction = mean(cluster == 0L),
    minimum_cluster_fraction = if (length(counts)) min(as.numeric(counts)) / length(cluster) else 0,
    largest_cluster_fraction = if (length(counts)) max(as.numeric(counts)) / length(cluster) else 0,
    counts = counts
  )
}

safe_quantile <- function(value, probability) {
  unname(stats::quantile(value, probability, names = FALSE, type = 7))
}

write_profile <- function(
  profile_name, pipeline_features, cells, broad_features, profile_dir,
  loaded, capture_env, dbscan_cores, gate
) {
  historical_features <- historical_names(pipeline_features)
  historical <- as.data.frame(lapply(pipeline_features, function(name) broad_features[[name]]), check.names = FALSE)
  names(historical) <- historical_features
  historical$source_id <- "A0"
  projection_result <- loaded$env$plot_projection_by_passage(
    list(reference_v2_expanded = historical), method = "umap", seed = 42L,
    n_neighbors = 15L, n_components = 2L, n_pcs = 10L, cluster = FALSE,
    feature_columns = historical_features
  )
  if (!identical(colnames(projection_result$umapinput), historical_features)) {
    fail("Historical helper changed the configured feature order for ", profile_name)
  }
  processed <- as.matrix(projection_result$umapinputafterPreprocess)
  if (!identical(colnames(processed), historical_features) || any(!is.finite(processed))) {
    fail("Historical preprocessing returned incomplete/nonfinite values for ", profile_name)
  }
  layout <- as.data.frame(projection_result$umapLayout[, c("Dim1", "Dim2"), drop = FALSE])
  if (nrow(layout) != nrow(cells) || any(!is.finite(as.matrix(layout)))) {
    fail("Historical UMAP returned invalid coordinates for ", profile_name)
  }
  dir.create(profile_dir, recursive = TRUE)
  write_tsv(data.frame(cell_id = cells$cell_id, layout, check.names = FALSE), file.path(profile_dir, "umap.tsv"))

  capture_env$last_mclapply <- NULL
  optimized <- loaded$env$optimize_dbscan_v4(
    layout, eps_range = seq(0.005, 0.7, by = 0.025),
    minPts_range = seq(100, 200, by = 5), max_noise_pct = 25,
    maxCluster = 10, n_cores = dbscan_cores
  )
  scan_values <- capture_env$last_mclapply
  grid <- expand.grid(eps = seq(0.005, 0.7, by = 0.025), min_points = seq(100, 200, by = 5))
  if (!is.list(scan_values) || length(scan_values) != nrow(grid)) {
    fail("Could not audit exact optimize_dbscan_v4 grid for ", profile_name)
  }
  scan_rows <- lapply(seq_len(nrow(grid)), function(index) {
    value <- scan_values[[index]]
    data.frame(
      grid_index = index, eps = grid$eps[[index]], min_points = grid$min_points[[index]],
      eligible = !is.null(value), n_clusters = if (is.null(value)) NA else value$n_clusters,
      pct_noise = if (is.null(value)) NA else value$pct_noise,
      shannon_norm = if (is.null(value)) NA else value$shannon_norm,
      score = if (is.null(value)) NA else value$score,
      selected = if (is.null(value) || is.na(optimized$eps)) FALSE else
        isTRUE(all.equal(as.numeric(value$eps), as.numeric(optimized$eps))) &&
        identical(as.integer(value$minPts), as.integer(optimized$minPts))
    )
  })
  write_tsv(do.call(rbind, scan_rows), file.path(profile_dir, "dbscan_scan.tsv"))
  optimized_cluster <- as.integer(optimized$cluster)
  source <- if (is.na(optimized$eps)) "optimize_dbscan_v4_one_cluster_fallback" else "optimize_dbscan_v4_selected"
  write_tsv(data.frame(
    cell_id = cells$cell_id, cluster = optimized_cluster, cluster_source = source
  ), file.path(profile_dir, "diagnostic_clusters.tsv"))
  fixed_cluster <- dbscan::dbscan(layout, eps = 0.5, minPts = 5)$cluster
  write_tsv(data.frame(cell_id = cells$cell_id, cluster = as.integer(fixed_cluster)), file.path(profile_dir, "fixed_dbscan_clusters.tsv"))

  raw <- as.data.frame(projection_result$umapinput, check.names = FALSE)
  transformed <- raw
  size_names <- grep("Area|Axis|Perimeter|Diameter", names(raw), value = TRUE)
  other_names <- setdiff(names(raw), size_names)
  transformed[other_names] <- lapply(transformed[other_names], log1p)
  transform_rows <- lapply(seq_along(pipeline_features), function(index) data.frame(
    pipeline_feature = pipeline_features[[index]],
    historical_feature = historical_features[[index]],
    feature_block = feature_block(pipeline_features[[index]]),
    transform = if (historical_features[[index]] %in% size_names) "linear" else "log1p",
    case_sensitive_size_regex_match = historical_features[[index]] %in% size_names,
    missing_fraction_raw = mean(is.na(raw[[historical_features[[index]]]])),
    transformed_center = mean(transformed[[historical_features[[index]]]], na.rm = TRUE),
    transformed_scale = stats::sd(transformed[[historical_features[[index]]]], na.rm = TRUE),
    knn_k = 5L
  ))
  write_tsv(do.call(rbind, transform_rows), file.path(profile_dir, "feature_transform_manifest.tsv"))
  statistics <- lapply(seq_along(pipeline_features), function(index) {
    value <- as.numeric(raw[[historical_features[[index]]]])
    data.frame(
      pipeline_feature = pipeline_features[[index]],
      historical_feature = historical_features[[index]],
      feature_block = feature_block(pipeline_features[[index]]),
      row_count = length(value), nonfinite_count = sum(!is.finite(value)),
      minimum = min(value), q01 = safe_quantile(value, 0.01),
      q25 = safe_quantile(value, 0.25), median = stats::median(value),
      mean = mean(value), q75 = safe_quantile(value, 0.75),
      q99 = safe_quantile(value, 0.99), maximum = max(value),
      standard_deviation = stats::sd(value), zero_fraction = mean(value == 0)
    )
  })
  write_tsv(do.call(rbind, statistics), file.path(profile_dir, "feature_statistics.tsv"))
  write_tsv(
    data.frame(cell_id = cells$cell_id, processed, check.names = FALSE),
    file.path(profile_dir, "preprocessed_projection_features.tsv")
  )

  pca <- stats::prcomp(processed, center = FALSE, scale. = FALSE)
  variance <- pca$sdev^2
  used_count <- min(10L, ncol(pca$x))
  pca_rows <- data.frame(
    component = seq_along(variance), variance = variance,
    proportion = variance / sum(variance),
    cumulative_proportion = cumsum(variance / sum(variance)),
    used_for_umap = seq_along(variance) <= used_count
  )
  write_tsv(pca_rows, file.path(profile_dir, "pca_variance.tsv"))
  used_variance <- variance[seq_len(used_count)]
  weighted <- sweep(pca$rotation[, seq_len(used_count), drop = FALSE]^2, 2L, used_variance, `*`)
  contribution <- rowSums(weighted)
  contribution <- contribution / sum(contribution)
  contribution_rows <- data.frame(
    pipeline_feature = pipeline_features,
    historical_feature = historical_features,
    feature_block = vapply(pipeline_features, feature_block, character(1)),
    weighted_used_pc_contribution = as.numeric(contribution)
  )
  write_tsv(contribution_rows, file.path(profile_dir, "pca_feature_contribution.tsv"))
  block_rows <- aggregate(weighted_used_pc_contribution ~ feature_block, contribution_rows, sum)
  block_rows$feature_count <- vapply(block_rows$feature_block, function(block) sum(contribution_rows$feature_block == block), integer(1))
  block_rows <- block_rows[, c("feature_block", "feature_count", "weighted_used_pc_contribution")]
  write_tsv(block_rows, file.path(profile_dir, "pca_block_contribution.tsv"))

  metrics <- cluster_metrics(optimized_cluster)
  cluster_rows <- data.frame(
    cluster = c(0L, as.integer(names(metrics$counts))),
    row_count = c(sum(optimized_cluster == 0L), as.integer(metrics$counts)),
    fraction = c(sum(optimized_cluster == 0L), as.integer(metrics$counts)) / length(optimized_cluster),
    role = c("noise", rep("diagnostic_non_noise", length(metrics$counts)))
  )
  write_tsv(cluster_rows, file.path(profile_dir, "cluster_summary.tsv"))

  set.seed(as.integer(gate$bootstrap_seed))
  seeds <- sample.int(.Machine$integer.max, as.integer(gate$bootstrap_replicates))
  sample_n <- floor(nrow(layout) * as.numeric(gate$bootstrap_fraction))
  bootstrap_rows <- data.frame(
    replicate = seq_along(seeds), seed = seeds, sampled_rows = sample_n,
    adjusted_rand_all = NA_real_, n_clusters = NA_integer_,
    noise_fraction = NA_real_, status = "SKIPPED_ONE_CLUSTER_FALLBACK",
    stringsAsFactors = FALSE
  )
  neighborhood_rows <- data.frame(
    eps = double(), min_points = integer(), adjusted_rand_all = double(),
    n_clusters = integer(), noise_fraction = double(), selected = logical(),
    stringsAsFactors = FALSE
  )
  if (!is.na(optimized$eps)) {
    bootstrap_rows <- do.call(rbind, lapply(seq_along(seeds), function(index) {
      set.seed(seeds[[index]])
      selected <- sort(sample.int(nrow(layout), sample_n, replace = FALSE))
      fitted <- dbscan::dbscan(
        layout[selected, , drop = FALSE], eps = as.numeric(optimized$eps),
        minPts = as.integer(optimized$minPts)
      )$cluster
      fitted_metrics <- cluster_metrics(fitted)
      data.frame(
        replicate = index, seed = seeds[[index]], sampled_rows = sample_n,
        adjusted_rand_all = adjusted_rand_index(optimized_cluster[selected], fitted),
        n_clusters = fitted_metrics$n_clusters,
        noise_fraction = fitted_metrics$noise_fraction,
        status = "COMPLETE"
      )
    }))
    eps_values <- sort(unique(pmax(0.000001, as.numeric(optimized$eps) + c(-1, 0, 1) * as.numeric(gate$parameter_neighborhood_eps_step))))
    min_values <- sort(unique(pmax(2L, as.integer(optimized$minPts) + c(-1L, 0L, 1L) * as.integer(gate$parameter_neighborhood_min_points_step))))
    neighborhood_rows <- do.call(rbind, lapply(eps_values, function(eps) {
      do.call(rbind, lapply(min_values, function(min_points) {
        fitted <- dbscan::dbscan(layout, eps = eps, minPts = min_points)$cluster
        fitted_metrics <- cluster_metrics(fitted)
        data.frame(
          eps = eps, min_points = min_points,
          adjusted_rand_all = adjusted_rand_index(optimized_cluster, fitted),
          n_clusters = fitted_metrics$n_clusters,
          noise_fraction = fitted_metrics$noise_fraction,
          selected = isTRUE(all.equal(eps, as.numeric(optimized$eps))) && min_points == as.integer(optimized$minPts)
        )
      }))
    }))
  }
  write_tsv(bootstrap_rows, file.path(profile_dir, "bootstrap_stability.tsv"))
  write_tsv(neighborhood_rows, file.path(profile_dir, "parameter_neighborhood_stability.tsv"))

  nonselected_neighborhood <- neighborhood_rows[!neighborhood_rows$selected, , drop = FALSE]
  completed_bootstrap <- bootstrap_rows[bootstrap_rows$status == "COMPLETE", , drop = FALSE]
  bootstrap_median <- if (nrow(completed_bootstrap)) stats::median(completed_bootstrap$adjusted_rand_all) else NA_real_
  neighborhood_median <- if (nrow(nonselected_neighborhood)) stats::median(nonselected_neighborhood$adjusted_rand_all) else NA_real_
  neighborhood_minimum <- if (nrow(nonselected_neighborhood)) min(nonselected_neighborhood$adjusted_rand_all) else NA_real_
  neighborhood_no_collapse <- nrow(nonselected_neighborhood) > 0L && all(nonselected_neighborhood$n_clusters >= 2L)
  checks <- c(
    optimized_solution = !is.na(optimized$eps),
    minimum_cluster_count = metrics$n_clusters >= as.integer(gate$minimum_non_noise_clusters),
    minimum_cluster_fraction = metrics$minimum_cluster_fraction >= as.numeric(gate$minimum_cluster_fraction),
    maximum_noise_fraction = metrics$noise_fraction <= as.numeric(gate$maximum_noise_fraction),
    bootstrap_median_adjusted_rand = is.finite(bootstrap_median) && bootstrap_median >= as.numeric(gate$minimum_bootstrap_median_adjusted_rand),
    parameter_neighborhood_median_adjusted_rand = is.finite(neighborhood_median) && neighborhood_median >= as.numeric(gate$minimum_parameter_neighborhood_median_adjusted_rand),
    parameter_neighborhood_minimum_adjusted_rand = is.finite(neighborhood_minimum) && neighborhood_minimum >= as.numeric(gate$minimum_parameter_neighborhood_adjusted_rand),
    parameter_neighborhood_no_one_cluster_collapse = !isTRUE(gate$require_no_one_cluster_neighbor) || neighborhood_no_collapse
  )
  list(
    profile = profile_name,
    pipeline_features = pipeline_features,
    historical_features = historical_features,
    selected_eps = if (is.na(optimized$eps)) NA_real_ else as.numeric(optimized$eps),
    selected_min_points = if (is.na(optimized$minPts)) NA_integer_ else as.integer(optimized$minPts),
    selected_score = if (is.na(optimized$score)) NA_real_ else as.numeric(optimized$score),
    n_clusters = metrics$n_clusters,
    noise_fraction = metrics$noise_fraction,
    minimum_cluster_fraction = metrics$minimum_cluster_fraction,
    largest_cluster_fraction = metrics$largest_cluster_fraction,
    bootstrap_median_adjusted_rand = bootstrap_median,
    parameter_neighborhood_median_adjusted_rand = neighborhood_median,
    parameter_neighborhood_minimum_adjusted_rand = neighborhood_minimum,
    parameter_neighborhood_no_one_cluster_collapse = neighborhood_no_collapse,
    computational_checks = checks,
    computational_gate = if (all(checks)) "PASS" else "FAIL"
  )
}

recursive_hashes <- function(directory, exclude = character()) {
  paths <- list.files(directory, recursive = TRUE, all.files = TRUE, no.. = TRUE, full.names = TRUE)
  paths <- paths[!file.info(paths)$isdir]
  relative <- substring(paths, nchar(directory) + 2L)
  keep <- !(relative %in% exclude)
  paths <- paths[keep]
  relative <- relative[keep]
  if (any(nzchar(Sys.readlink(paths)))) fail("Expanded projection output contains a symlink")
  stats::setNames(vapply(paths, sha256_file, character(1)), relative)
}

validate_reuse <- function(output_dir, expected_inputs, implementation_sha256, adapter_sha256) {
  manifest_path <- file.path(output_dir, "expanded_projection_manifest.json")
  identity_path <- file.path(output_dir, "expanded_projection_generation_identity.json")
  if (!file.exists(manifest_path) || !file.exists(identity_path)) fail("Existing expanded projection is partial")
  manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
  if (!identical(manifest$schema_version, SCHEMA_VERSION) || !identical(manifest$status, "COMPLETE")) {
    fail("Existing expanded projection manifest is not COMPLETE")
  }
  if (!identical(manifest$inputs, expected_inputs)) fail("Existing expanded projection input identity differs")
  declared <- unlist(manifest$output_file_sha256, use.names = TRUE)
  observed <- recursive_hashes(output_dir, c("expanded_projection_manifest.json", "expanded_projection_generation_identity.json"))
  if (!identical(declared, observed)) fail("Existing expanded projection artifact set/hash differs")
  identity <- jsonlite::fromJSON(identity_path, simplifyVector = FALSE)
  expected_identity <- list(
    schema_version = IDENTITY_SCHEMA_VERSION,
    status = "COMPLETE",
    expanded_projection_manifest_sha256 = sha256_file(manifest_path),
    implementation_sha256 = implementation_sha256,
    historical_adapter_sha256 = adapter_sha256,
    output_file_sha256 = as.list(observed)
  )
  if (!identical(identity, expected_identity)) fail("Existing expanded projection generation identity differs")
  invisible(TRUE)
}

main <- function() {
  args <- parse_cli(commandArgs(trailingOnly = TRUE))
  required_packages <- c("jsonlite", "digest", "uwot", "dbscan", "cluster")
  missing <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) fail("Missing expanded projection R packages: ", paste(missing, collapse = ","))
  implementation_path <- current_script_path()
  implementation_sha256 <- sha256_file(implementation_path)
  config_path <- normalizePath(args$feature_config, mustWork = TRUE)
  config <- validate_config(config_path)
  library <- load_projection_library(args$historical_adapter_script)
  library$env$require_dependencies()
  snapshot_root <- normalizePath(args$reference_snapshot_root, mustWork = TRUE)
  reference_utils <- normalizePath(
    file.path(snapshot_root, config$historical_projection_contract$reference_source_file),
    mustWork = TRUE
  )
  capture_env <- new.env(parent = emptyenv())
  loaded <- library$env$load_reference_functions(reference_utils, capture_env)
  if (args$check_config) {
    cat("expanded_projection_config=PASS\n")
    cat("profile_count=", length(PROFILE_FEATURES), "\n", sep = "")
    cat("selected_annotation_profile=expanded39\n")
    cat("historical_adapter_sha256=", library$sha256, "\n", sep = "")
    cat("reference_utils_sha256=", loaded$source_sha256, "\n", sep = "")
    return(invisible(0L))
  }

  project_path <- normalizePath(args$project, mustWork = TRUE)
  broad_features_path <- normalizePath(args$broad_features, mustWork = TRUE)
  for (path in c(project_path, broad_features_path, config_path, library$path, reference_utils, implementation_path)) {
    if (nzchar(Sys.readlink(path))) fail("Expanded projection input must not be a symlink: ", path)
  }
  project <- jsonlite::fromJSON(project_path, simplifyVector = FALSE)
  if (!identical(project$schema_version, PROJECT_SCHEMA_VERSION) || !identical(project$project_id, PROJECT_ID)) {
    fail("Expanded projection requires the frozen reference-cell-state V2 development project")
  }
  cells_path <- library$env$resolve_project_asset(project_path, project$cells_file, "cells_file")
  cells <- read.delim(cells_path, check.names = FALSE, stringsAsFactors = FALSE)
  required_cells <- c("cell_id", "context_key", "source_id", "split")
  if (!all(required_cells %in% names(cells))) fail("V2 cells.tsv lacks expanded projection identity columns")
  expected_cell_count <- as.integer(config$cell_universe$expected_cell_count)
  if (nrow(cells) != expected_cell_count || anyNA(cells$cell_id) || any(!nzchar(cells$cell_id)) || anyDuplicated(cells$cell_id)) {
    fail("Expanded projection cell count/identity differs from the frozen config")
  }
  if (!all(cells$split == "development")) fail("Heldout cells entered expanded projection")
  if (!identical(sort(unique(cells$context_key)), c("SUM-159-NLS-2N", "SUM-159-NLS-4N"))) {
    fail("Expanded projection context universe differs")
  }
  broad <- read.delim(broad_features_path, check.names = FALSE, stringsAsFactors = FALSE)
  required_broad <- c("cell_id", EXPANDED39, EXCLUDED_FEATURES)
  missing_broad <- setdiff(required_broad, names(broad))
  if (length(missing_broad)) fail("Broad representative features lack expanded inputs: ", paste(missing_broad, collapse = ","))
  if (!identical(as.character(broad$cell_id), as.character(cells$cell_id))) {
    fail("Broad feature rows are not in exact lockstep with the frozen V2 cells")
  }
  for (name in EXPANDED39) {
    value <- suppressWarnings(as.numeric(broad[[name]]))
    if (any(!is.finite(value))) fail("Expanded feature is nonfinite: ", name)
    historical_name <- historical_names(name)
    if (!grepl("Area|Axis|Perimeter|Diameter", historical_name) && any(value <= -1)) {
      fail("Expanded non-size feature is outside log1p domain: ", name)
    }
    broad[[name]] <- value
  }

  output_argument <- path.expand(args$output_dir)
  if (file.exists(output_argument) && !dir.exists(output_argument)) fail("Expanded projection output exists and is not a directory")
  if (dir.exists(output_argument) && nzchar(Sys.readlink(output_argument))) fail("Expanded projection output must not be a symlink")
  output_dir <- normalizePath(output_argument, mustWork = FALSE)
  expected_inputs <- list(
    project = list(path = project_path, sha256 = sha256_file(project_path)),
    cells = list(path = cells_path, sha256 = sha256_file(cells_path)),
    broad_features = list(path = broad_features_path, sha256 = sha256_file(broad_features_path)),
    feature_config = list(path = config_path, sha256 = sha256_file(config_path)),
    reference_utils = list(path = reference_utils, sha256 = loaded$source_sha256),
    historical_adapter = list(path = library$path, sha256 = library$sha256),
    implementation = list(path = implementation_path, sha256 = implementation_sha256)
  )
  if (dir.exists(output_dir)) {
    if (!args$overwrite) fail("Expanded projection output already exists: ", output_dir)
    validate_reuse(output_dir, expected_inputs, implementation_sha256, library$sha256)
    cat("expanded_projection=", output_dir, "\n", sep = "")
    cat("generation_status=verified_reuse\n")
    return(invisible(0L))
  }
  parent <- dirname(output_dir)
  dir.create(parent, recursive = TRUE, showWarnings = FALSE)
  staging <- tempfile(pattern = paste0(".", basename(output_dir), ".staging."), tmpdir = parent)
  dir.create(staging)
  on.exit(if (dir.exists(staging)) unlink(staging, recursive = TRUE), add = TRUE)

  summaries <- lapply(names(PROFILE_FEATURES), function(profile_name) {
    write_profile(
      profile_name, PROFILE_FEATURES[[profile_name]], cells, broad,
      file.path(staging, "profiles", profile_name), loaded, capture_env,
      args$dbscan_cores, config$labelability_gate
    )
  })
  names(summaries) <- names(PROFILE_FEATURES)
  summary_table <- do.call(rbind, lapply(summaries, function(value) data.frame(
    profile = value$profile,
    feature_count = length(value$pipeline_features),
    selected_eps = value$selected_eps,
    selected_min_points = value$selected_min_points,
    n_clusters = value$n_clusters,
    noise_fraction = value$noise_fraction,
    minimum_cluster_fraction = value$minimum_cluster_fraction,
    largest_cluster_fraction = value$largest_cluster_fraction,
    bootstrap_median_adjusted_rand = value$bootstrap_median_adjusted_rand,
    parameter_neighborhood_median_adjusted_rand = value$parameter_neighborhood_median_adjusted_rand,
    parameter_neighborhood_minimum_adjusted_rand = value$parameter_neighborhood_minimum_adjusted_rand,
    parameter_neighborhood_no_one_cluster_collapse = value$parameter_neighborhood_no_one_cluster_collapse,
    computational_gate = value$computational_gate
  )))
  write_tsv(summary_table, file.path(staging, "profile_summary.tsv"))

  selected <- summaries[["expanded39"]]
  selected_dir <- file.path(staging, "profiles", "expanded39")
  selected_umap <- read.delim(file.path(selected_dir, "umap.tsv"), check.names = FALSE, stringsAsFactors = FALSE)
  selected_clusters <- read.delim(file.path(selected_dir, "diagnostic_clusters.tsv"), check.names = FALSE, stringsAsFactors = FALSE)
  representative_input <- data.frame(
    cell_id = cells$cell_id, Dim1 = selected_umap$Dim1, Dim2 = selected_umap$Dim2,
    cluster = as.integer(selected_clusters$cluster), context_key = cells$context_key,
    source_id = cells$source_id, stringsAsFactors = FALSE, check.names = FALSE
  )
  set.seed(1L)
  representatives <- loaded$env$get_all_cell_lines_overlay_representatives(
    representative_input, total_n = 300L, balance = 0.2
  )
  representative_ids <- as.character(representatives$cell_id)
  if (!length(representative_ids) || length(representative_ids) > 300L || anyDuplicated(representative_ids) ||
      !all(representative_ids %in% cells$cell_id)) fail("Expanded historical representative selection is invalid")
  representative_table <- data.frame(
    selection_rank = seq_along(representative_ids), cell_id = representative_ids,
    context_key = as.character(representatives$context_key),
    source_id = as.character(representatives$source_id),
    cluster = as.integer(representatives$cluster), Dim1 = as.numeric(representatives$Dim1),
    Dim2 = as.numeric(representatives$Dim2), stringsAsFactors = FALSE, check.names = FALSE
  )
  write_tsv(representative_table, file.path(staging, "historical_representatives.tsv"))
  file.copy(file.path(selected_dir, "umap.tsv"), file.path(staging, "umap.tsv"))
  file.copy(file.path(selected_dir, "diagnostic_clusters.tsv"), file.path(staging, "diagnostic_clusters.tsv"))

  decision <- list(
    schema_version = "reference_cell_state_expanded_labelability_decision_v1",
    status = "COMPLETE",
    selected_annotation_profile = "expanded39",
    computational_gate = selected$computational_gate,
    morphology_overlay_gate = "PENDING_HUMAN_REVIEW",
    overall_labelability = if (identical(selected$computational_gate, "PASS")) {
      "MORPHOLOGY_OVERLAY_REVIEW_REQUIRED"
    } else {
      "NO_GO_FOR_POLYGON_ANNOTATION_USE_500_CELL_BLIND_REVIEW"
    },
    computational_checks = as.list(selected$computational_checks),
    thresholds = config$labelability_gate,
    scientific_boundary = paste(
      "Diagnostic clusters are navigation/sampling metadata, never cell-state labels.",
      "A computational PASS is not a polygon-annotation GO until the morphology overlay",
      "shows coherent within-cluster and distinct between-cluster cellular evidence."
    )
  )
  jsonlite::write_json(decision, file.path(staging, "labelability_decision.json"), pretty = TRUE, auto_unbox = TRUE, null = "null")

  artifacts <- recursive_hashes(staging)
  manifest <- list(
    schema_version = SCHEMA_VERSION,
    status = "COMPLETE",
    role = "three_profile_annotation_geometry_and_labelability_comparison",
    selected_annotation_profile = "expanded39",
    inputs = expected_inputs,
    cell_universe = list(
      row_count = nrow(cells), ordered_cell_id_sha256 = digest::digest(paste(cells$cell_id, collapse = "\n"), algo = "sha256", serialize = FALSE),
      contexts = as.list(sort(unique(cells$context_key))), heldout_read = FALSE
    ),
    profiles = lapply(summaries, function(value) list(
      feature_columns = as.list(value$pipeline_features),
      historical_feature_columns = as.list(value$historical_features),
      n_clusters = value$n_clusters, noise_fraction = value$noise_fraction,
      minimum_cluster_fraction = value$minimum_cluster_fraction,
      bootstrap_median_adjusted_rand = value$bootstrap_median_adjusted_rand,
      parameter_neighborhood_median_adjusted_rand = value$parameter_neighborhood_median_adjusted_rand,
      computational_gate = value$computational_gate
    )),
    excluded_features = config$excluded_feature_columns,
    historical_projection_contract = config$historical_projection_contract,
    labelability_gate = config$labelability_gate,
    classifier_boundary = list(
      final_classifier_feature_source = "classifier12",
      expanded39_allowed_in_final_classifier = FALSE,
      expanded39_role = "annotation_geometry_and_human_morphology_evidence_only"
    ),
    blinding_contract = list(
      dead = "not_read", combined_rgb = "not_read", current_classification = "not_read",
      trajectory = "not_read", heldout = "not_read"
    ),
    output_file_sha256 = as.list(artifacts)
  )
  manifest_path <- file.path(staging, "expanded_projection_manifest.json")
  jsonlite::write_json(manifest, manifest_path, pretty = TRUE, auto_unbox = TRUE, null = "null")
  identity <- list(
    schema_version = IDENTITY_SCHEMA_VERSION,
    status = "COMPLETE",
    expanded_projection_manifest_sha256 = sha256_file(manifest_path),
    implementation_sha256 = implementation_sha256,
    historical_adapter_sha256 = library$sha256,
    output_file_sha256 = as.list(artifacts)
  )
  jsonlite::write_json(
    identity, file.path(staging, "expanded_projection_generation_identity.json"),
    pretty = TRUE, auto_unbox = TRUE, null = "null"
  )
  if (!identical(sha256_file(implementation_path), implementation_sha256) ||
      !identical(sha256_file(library$path), library$sha256)) {
    fail("Expanded projection implementation changed during execution")
  }
  if (file.exists(output_dir) || dir.exists(output_dir)) fail("Expanded projection output appeared during staging")
  if (!file.rename(staging, output_dir)) fail("Failed to install expanded projection atomically")
  cat("expanded_projection=", output_dir, "\n", sep = "")
  cat("selected_annotation_profile=expanded39\n")
  cat("computational_labelability_gate=", selected$computational_gate, "\n", sep = "")
  cat("labelability_status=", decision$overall_labelability, "\n", sep = "")
  cat("generation_status=created\n")
  invisible(0L)
}

tryCatch(main(), error = function(error) {
  message("ERROR: ", conditionMessage(error))
  quit(save = "no", status = 1L, runLast = FALSE)
})
