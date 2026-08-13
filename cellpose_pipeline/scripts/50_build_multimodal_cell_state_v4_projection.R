#!/usr/bin/env Rscript

# Build a dataset-adapted multimodal representation.  This is intentionally not
# a historical-LTEE parity layer: it uses zero-aware Nuclei support and equal
# Shape/Brightfield/Nuclei/Dead block geometry for the SUM159 data available here.

options(stringsAsFactors = FALSE, warn = 1)

SCHEMA_VERSION <- "multimodal_cell_state_v4_projection_v1"
IDENTITY_SCHEMA_VERSION <- "multimodal_cell_state_v4_projection_generation_identity_v1"
CONFIG_SCHEMA_VERSION <- "multimodal_cell_state_v4_config_v1"

SHAPE <- c("area_px2", "roundness", "aspect_ratio", "extent", "solidity")
BRIGHTFIELD <- c(
  "bf_object_median", "bf_object_iqr", "bf_object_p90_minus_p10",
  "bf_object_mean_minus_ring_background", "bf_interior_minus_boundary_mean",
  "bf_boundary_gradient_p95", "bf_laplacian_variance", "bf_edge_density",
  "bf_glcm_contrast", "bf_glcm_entropy", "bf_glcm_correlation"
)
NUCLEI <- c(
  "nucleus_absent", "nuclei_count_excess", "nuclei_area_fraction_if_present",
  "largest_nucleus_fraction", "nucleus_area_cv",
  "small_nuclear_fragment_fraction", "nuclei_spatial_dispersion"
)
NUCLEI_QC <- c("nuclei_cell_to_ring_contrast", "nuclei_signal_positive_fraction")
CONDITIONAL_NUCLEI <- c(
  "nuclei_area_fraction_if_present", "largest_nucleus_fraction",
  "nucleus_area_cv", "small_nuclear_fragment_fraction", "nuclei_spatial_dispersion"
)
DEAD <- c(
  "dead_signal_absent", "dead_cell_to_ring_median_contrast",
  "dead_cell_to_ring_p90_contrast", "dead_object_p90_minus_p10",
  "dead_signal_positive_fraction", "dead_integrated_excess_per_area",
  "dead_largest_positive_component_fraction",
  "dead_positive_component_count_excess", "dead_signal_spatial_dispersion"
)
DEAD_QC <- c("dead_signal_saturation_fraction")
CONDITIONAL_DEAD <- c(
  "dead_largest_positive_component_fraction",
  "dead_positive_component_count_excess", "dead_signal_spatial_dispersion"
)
PRIMARY_PROFILE <- "v4_balanced_four_block"
DEATH_RESOLUTION_PROFILES <- c(
  PRIMARY_PROFILE, "v4_death_resolution_dead35_nuclei25",
  "v4_death_resolution_dead30_bf30"
)
GENERATED_PROFILES <- c(
  "v4_zero_aware_unpruned", "v4_pruned_unbalanced", PRIMARY_PROFILE,
  "v4_death_resolution_dead35_nuclei25",
  "v4_death_resolution_dead30_bf30", "v4_minus_shape",
  "v4_minus_brightfield", "v4_minus_nuclei", "v4_minus_dead",
  "v4_dead_nuclei_only"
)

fail <- function(...) stop(paste0(...), call. = FALSE)
sha256_file <- function(path) digest::digest(file = path, algo = "sha256", serialize = FALSE)
write_tsv <- function(value, path) write.table(
  value, path, sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE, na = ""
)

parse_cli <- function(argv) {
  result <- list(overwrite = FALSE, check_config = FALSE)
  index <- 1L
  while (index <= length(argv)) {
    token <- argv[[index]]
    if (token == "--overwrite") {
      result$overwrite <- TRUE
      index <- index + 1L
    } else if (token == "--check-config") {
      result$check_config <- TRUE
      index <- index + 1L
    } else if (token %in% c("--cells", "--features", "--expanded-projection", "--config", "--output-dir")) {
      if (index == length(argv)) fail("Missing value after ", token)
      result[[gsub("-", "_", sub("^--", "", token))]] <- argv[[index + 1L]]
      index <- index + 2L
    } else {
      fail("Unknown argument: ", token)
    }
  }
  if (is.null(result$config)) fail("--config is required")
  if (!result$check_config) {
    for (name in c("cells", "features", "expanded_projection", "output_dir")) {
      if (is.null(result[[name]]) || !nzchar(result[[name]])) fail("--", gsub("_", "-", name), " is required")
    }
  }
  result
}

current_script_path <- function() {
  argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(argument) != 1L) fail("Could not resolve V4 projection implementation path")
  normalizePath(sub("^--file=", "", argument[[1L]]), mustWork = TRUE)
}

as_vector <- function(value, label) {
  output <- unname(unlist(value, use.names = FALSE))
  if (!length(output) || anyNA(output)) fail(label, " must be a nonempty array")
  output
}

validate_config <- function(path) {
  value <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  if (!identical(value$schema_version, CONFIG_SCHEMA_VERSION) ||
      !identical(value$method_version, "sum159_multimodal_cell_state_v4")) {
    fail("Unsupported multimodal-cell-state V4 config")
  }
  blocks <- value$feature_blocks
  if (!identical(as.character(as_vector(blocks$shape$candidate_columns, "shape features")), SHAPE) ||
      !identical(as.character(as_vector(blocks$brightfield$candidate_columns, "brightfield features")), BRIGHTFIELD) ||
      !identical(as.character(as_vector(blocks$nuclei$candidate_columns, "nuclei features")), NUCLEI) ||
      !identical(as.character(as_vector(blocks$dead$candidate_columns, "Dead features")), DEAD)) {
    fail("V4 block feature order differs")
  }
  equal <- value$block_equalization
  if (!identical(as.character(as_vector(equal$blocks, "equal blocks")), c("shape", "brightfield", "nuclei", "dead")) ||
      abs(as.numeric(equal$target_fraction_each) - 0.25) > 1e-12) {
    fail("V4 block equalization contract differs")
  }
  profiles <- as.character(as_vector(value$projection_profiles, "projection profiles"))
  if (!identical(profiles, c("expanded39_audit", GENERATED_PROFILES))) fail("V4 projection profile order differs")
  grid <- value$projection_grid
  if (!identical(grid$primary_profile, PRIMARY_PROFILE) ||
      any(!as.integer(as_vector(grid$n_neighbors, "n_neighbors")) %in% c(15L, 30L, 50L)) ||
      any(!as.numeric(as_vector(grid$min_dist, "min_dist")) %in% c(0.05, 0.1, 0.3)) ||
      length(unique(as.integer(as_vector(grid$seeds, "seeds")))) != length(as_vector(grid$seeds, "seeds"))) {
    fail("V4 projection grid differs or is invalid")
  }
  if (!identical(value$projection_audit$balanced_dead_island_required, FALSE)) fail("V4 balanced UMAP must not require a dead-cell island")
  weight_profiles <- value$death_resolution_weight_profiles
  if (!identical(sort(names(weight_profiles)), sort(DEATH_RESOLUTION_PROFILES))) fail("V4 death-resolution weight profiles differ")
  for (profile in DEATH_RESOLUTION_PROFILES) {
    weights <- as.numeric(as_vector(weight_profiles[[profile]], paste(profile, "weights")))
    if (length(weights) != 4L || any(!is.finite(weights)) || any(weights <= 0) || abs(sum(weights) - 1) > 1e-12) fail("Invalid V4 block weights: ", profile)
  }
  forbidden <- as.character(as_vector(value$blinding$forbidden_before_model_freeze, "forbidden inputs"))
  allowed <- as.character(as_vector(value$blinding$allowed_before_model_freeze, "allowed inputs"))
  if (!("Dead_raw" %in% allowed) || !all(c("Existing_Dead_segmentation", "current_classification", "trajectory", "heldout_cells") %in% forbidden)) {
    fail("V4 blinding contract is incomplete")
  }
  value
}

transform_kind <- function(name) {
  if (name == "area_px2") return("log1p")
  if (name %in% c("roundness", "extent", "solidity", "nuclei_area_fraction_if_present",
                  "largest_nucleus_fraction", "small_nuclear_fragment_fraction",
                  "nuclei_signal_positive_fraction", "bf_edge_density",
                  "dead_signal_positive_fraction", "dead_largest_positive_component_fraction",
                  "dead_signal_saturation_fraction")) return("logit_clipped")
  if (name %in% c("aspect_ratio", "bf_object_iqr", "bf_object_p90_minus_p10",
                  "bf_boundary_gradient_p95", "bf_laplacian_variance", "bf_glcm_contrast",
                  "bf_glcm_entropy", "nucleus_area_cv", "nuclei_spatial_dispersion",
                  "dead_integrated_excess_per_area", "dead_positive_component_count_excess",
                  "dead_signal_spatial_dispersion")) return("log1p")
  if (name %in% c("bf_object_mean_minus_ring_background", "bf_interior_minus_boundary_mean",
                  "bf_glcm_correlation", "nuclei_cell_to_ring_contrast",
                  "dead_cell_to_ring_median_contrast", "dead_cell_to_ring_p90_contrast",
                  "dead_object_p90_minus_p10")) return("asinh")
  "identity"
}

apply_transform <- function(value, kind) {
  if (kind == "log1p") {
    if (any(value[is.finite(value)] < 0)) fail("log1p feature contains a negative value")
    return(log1p(value))
  }
  if (kind == "logit_clipped") {
    clipped <- pmin(1 - 1e-6, pmax(1e-6, value))
    return(log(clipped / (1 - clipped)))
  }
  if (kind == "asinh") return(asinh(value))
  value
}

fit_feature <- function(name, raw, nuclei_present, nuclei_reliable, dead_present, dead_reliable, winsor) {
  conditional_present <- name %in% CONDITIONAL_NUCLEI
  conditional_dead_present <- name %in% CONDITIONAL_DEAD
  conditional_reliable <- name %in% c("nucleus_absent", "nuclei_count_excess")
  conditional_dead_reliable <- name %in% c("dead_signal_absent", setdiff(DEAD, CONDITIONAL_DEAD))
  fit_rows <- is.finite(raw)
  if (conditional_present) fit_rows <- fit_rows & nuclei_present
  if (conditional_reliable) fit_rows <- fit_rows & nuclei_reliable
  if (conditional_dead_present) fit_rows <- fit_rows & dead_present
  if (conditional_dead_reliable) fit_rows <- fit_rows & dead_reliable
  if (sum(fit_rows) < 2L) return(list(keep = FALSE, reason = "insufficient_finite_fit_rows"))
  kind <- transform_kind(name)
  transformed <- apply_transform(raw, kind)
  fit_values <- transformed[fit_rows & is.finite(transformed)]
  if (length(fit_values) < 2L) return(list(keep = FALSE, reason = "insufficient_finite_transformed_rows"))
  bounds <- as.numeric(stats::quantile(fit_values, probs = winsor, names = FALSE, type = 7))
  clipped <- pmin(bounds[[2L]], pmax(bounds[[1L]], transformed))
  if (name %in% c("nucleus_absent", "dead_signal_absent")) {
    prevalence <- mean(raw[fit_rows])
    scale <- sqrt(prevalence * (1 - prevalence))
    if (!is.finite(scale) || scale <= 0) return(list(keep = FALSE, reason = "constant_binary_indicator"))
    output <- (raw - prevalence) / scale
    valid <- if (name == "nucleus_absent") nuclei_reliable else dead_reliable
    output[!valid | !is.finite(output)] <- 0
    return(list(
      keep = TRUE, value = output, transform = "binary_indicator",
      winsor_low = 0, winsor_high = 1, center = prevalence, scale = scale,
      scale_source = "bernoulli_standard_deviation", fit_rows = sum(fit_rows),
      conditional_present = FALSE, conditional_dead_present = FALSE,
      conditional_reliable = name == "nucleus_absent",
      conditional_dead_reliable = name == "dead_signal_absent"
    ))
  }
  center <- stats::median(clipped[fit_rows], na.rm = TRUE)
  scale <- 1.4826 * stats::median(abs(clipped[fit_rows] - center), na.rm = TRUE)
  scale_source <- "mad_times_1.4826"
  if (!is.finite(scale) || scale <= 0) {
    scale <- diff(stats::quantile(clipped[fit_rows], c(0.25, 0.75), names = FALSE, type = 7)) / 1.349
    scale_source <- "iqr_over_1.349_fallback"
  }
  if (!is.finite(scale) || scale <= 0) return(list(keep = FALSE, reason = "constant_after_transform"))
  output <- (clipped - center) / scale
  if (conditional_present) {
    output[!nuclei_present | !is.finite(output)] <- 0
  } else if (conditional_dead_present) {
    output[!dead_present | !is.finite(output)] <- 0
  } else if (conditional_reliable) {
    output[!nuclei_reliable | !is.finite(output)] <- 0
  } else if (conditional_dead_reliable) {
    output[!dead_reliable | !is.finite(output)] <- 0
  } else output[!is.finite(output)] <- 0
  list(
    keep = TRUE, value = output, transform = kind, winsor_low = bounds[[1L]],
    winsor_high = bounds[[2L]], center = center, scale = scale,
    scale_source = scale_source, fit_rows = sum(fit_rows),
    conditional_present = conditional_present,
    conditional_dead_present = conditional_dead_present,
    conditional_reliable = conditional_reliable,
    conditional_dead_reliable = conditional_dead_reliable
  )
}

mean_pairwise_squared_distance <- function(matrix) {
  matrix <- as.matrix(matrix)
  if (!nrow(matrix) || !ncol(matrix)) return(0)
  2 * sum(apply(matrix, 2L, stats::var))
}

prune_block <- function(processed, candidates, threshold, label) {
  retained <- character()
  removed <- list()
  for (name in intersect(candidates, colnames(processed))) {
    if (!length(retained)) {
      retained <- c(retained, name)
      next
    }
    values <- vapply(retained, function(other) {
      suppressWarnings(abs(stats::cor(processed[, name], processed[, other], method = "spearman", use = "pairwise.complete.obs")))
    }, numeric(1))
    values[!is.finite(values)] <- 0
    maximum <- max(values)
    if (maximum >= threshold) {
      match <- retained[[which.max(values)]]
      removed[[length(removed) + 1L]] <- data.frame(
        feature = name, reason = paste0(label, "_absolute_spearman_at_or_above_threshold"),
        correlated_with = match, absolute_spearman = maximum
      )
    } else retained <- c(retained, name)
  }
  list(retained = retained, removed = if (length(removed)) do.call(rbind, removed) else data.frame(
    feature = character(), reason = character(), correlated_with = character(), absolute_spearman = double()
  ))
}

equalize_blocks <- function(processed, block_map, active_blocks, weights = NULL) {
  output <- processed
  rows <- list()
  if (is.null(weights)) weights <- stats::setNames(rep(1 / length(active_blocks), length(active_blocks)), active_blocks)
  weights <- weights[active_blocks]
  if (anyNA(weights) || any(weights <= 0) || abs(sum(weights) - 1) > 1e-12) fail("Invalid V4 active-block weights")
  for (block in active_blocks) {
    names <- names(block_map)[block_map == block & names(block_map) %in% colnames(processed)]
    if (!length(names)) fail("V4 active block has no retained features: ", block)
    before <- mean_pairwise_squared_distance(processed[, names, drop = FALSE])
    if (!is.finite(before) || before <= 0) fail("V4 active block has zero geometry: ", block)
    target <- unname(weights[[block]])
    multiplier <- sqrt(target / before)
    output[, names] <- output[, names, drop = FALSE] * multiplier
    after <- mean_pairwise_squared_distance(output[, names, drop = FALSE])
    rows[[length(rows) + 1L]] <- data.frame(
      block = block, feature_count = length(names), mpd_before = before,
      multiplier = multiplier, mpd_after = after, target_fraction = target
    )
  }
  table <- do.call(rbind, rows)
  table$observed_fraction <- table$mpd_after / sum(table$mpd_after)
  list(matrix = output, audit = table)
}

neighbor_ids <- function(matrix, k) {
  value <- dbscan::kNN(as.matrix(matrix), k = min(k, nrow(matrix) - 1L), sort = TRUE)$id
  if (is.null(dim(value))) matrix(value, ncol = 1L) else value
}

neighbor_overlap <- function(left, right) {
  if (!identical(dim(left), dim(right))) fail("Neighbor matrices differ")
  mean(vapply(seq_len(nrow(left)), function(index) {
    length(intersect(left[index, ], right[index, ])) / ncol(left)
  }, numeric(1)))
}

neighbor_jaccard <- function(left, right) {
  if (!identical(dim(left), dim(right))) fail("Neighbor matrices differ")
  mean(vapply(seq_len(nrow(left)), function(index) {
    union <- union(left[index, ], right[index, ])
    if (!length(union)) 1 else length(intersect(left[index, ], right[index, ])) / length(union)
  }, numeric(1)))
}

profile_spec <- function(profile, pruned_bf, pruned_dead, config) {
  blocks <- list(shape = SHAPE, brightfield = BRIGHTFIELD, nuclei = NUCLEI, dead = DEAD)
  equal <- FALSE
  prune <- FALSE
  if (profile != "v4_zero_aware_unpruned") prune <- TRUE
  if (prune) blocks$brightfield <- pruned_bf
  if (prune) blocks$dead <- pruned_dead
  if (!(profile %in% c("v4_zero_aware_unpruned", "v4_pruned_unbalanced"))) equal <- TRUE
  if (profile == "v4_minus_shape") blocks$shape <- NULL
  if (profile == "v4_minus_brightfield") blocks$brightfield <- NULL
  if (profile == "v4_minus_nuclei") blocks$nuclei <- NULL
  if (profile == "v4_minus_dead") blocks$dead <- NULL
  if (profile == "v4_dead_nuclei_only") blocks <- blocks[c("nuclei", "dead")]
  weights <- NULL
  if (profile %in% DEATH_RESOLUTION_PROFILES) {
    weights <- unlist(config$death_resolution_weight_profiles[[profile]], use.names = TRUE)
  }
  list(blocks = blocks, equal = equal, weights = weights)
}

recursive_hashes <- function(directory, exclude = character()) {
  paths <- list.files(directory, recursive = TRUE, all.files = TRUE, no.. = TRUE, full.names = TRUE)
  paths <- paths[!file.info(paths)$isdir]
  relative <- substring(paths, nchar(directory) + 2L)
  keep <- !(relative %in% exclude)
  paths <- paths[keep]
  relative <- relative[keep]
  if (any(nzchar(Sys.readlink(paths)))) fail("V4 projection output contains a symlink")
  stats::setNames(vapply(paths, sha256_file, character(1)), relative)
}

verify_existing <- function(output, expected_inputs, implementation_sha) {
  manifest_path <- file.path(output, "projection_manifest.json")
  identity_path <- file.path(output, "projection_generation_identity.json")
  if (!file.exists(manifest_path) || !file.exists(identity_path)) fail("Existing V4 projection is partial")
  manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
  if (!identical(manifest$schema_version, SCHEMA_VERSION) || !identical(manifest$status, "COMPLETE")) fail("Existing V4 projection manifest is incomplete")
  if (!identical(manifest$inputs, expected_inputs)) fail("Existing V4 projection input identity differs")
  observed <- recursive_hashes(output, c("projection_manifest.json", "projection_generation_identity.json"))
  if (!identical(unlist(manifest$output_file_sha256, use.names = TRUE), observed)) fail("Existing V4 projection artifact set/hash differs")
  expected_identity <- list(
    schema_version = IDENTITY_SCHEMA_VERSION, status = "COMPLETE",
    projection_manifest_sha256 = sha256_file(manifest_path),
    implementation_sha256 = implementation_sha, output_file_sha256 = as.list(observed)
  )
  if (!identical(jsonlite::fromJSON(identity_path, simplifyVector = FALSE), expected_identity)) fail("Existing V4 projection generation identity differs")
}

main <- function() {
  args <- parse_cli(commandArgs(trailingOnly = TRUE))
  packages <- c("jsonlite", "digest", "uwot", "dbscan")
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) fail("Missing V4 projection R packages: ", paste(missing, collapse = ","))
  implementation <- current_script_path()
  implementation_sha <- sha256_file(implementation)
  config_path <- normalizePath(args$config, mustWork = TRUE)
  config <- validate_config(config_path)
  if (args$check_config) {
    cat("multimodal_cell_state_v4_config=PASS\n")
    cat("primary_profile=", PRIMARY_PROFILE, "\n", sep = "")
    cat("block_equal_target=0.25\n")
    return(invisible(0L))
  }

  paths <- lapply(c(args$cells, args$features), normalizePath, mustWork = TRUE)
  names(paths) <- c("cells", "features")
  expanded <- normalizePath(args$expanded_projection, mustWork = TRUE)
  if (!file.info(expanded)$isdir) fail("--expanded-projection must be a directory")
  for (path in c(unlist(paths), config_path, implementation, expanded)) {
    if (nzchar(Sys.readlink(path))) fail("V4 projection input must not be a symlink: ", path)
  }
  cells <- read.delim(paths$cells, check.names = FALSE, stringsAsFactors = FALSE)
  features <- read.delim(paths$features, check.names = FALSE, stringsAsFactors = FALSE)
  required_cells <- c("cell_id", "well", "site", "elapsed_hours", "context_key", "source_id", "split")
  required_features <- c(
    "cell_id", SHAPE, BRIGHTFIELD, NUCLEI, NUCLEI_QC, DEAD, DEAD_QC,
    "nuclei_measurement_status", "dead_measurement_status"
  )
  if (!all(required_cells %in% names(cells))) fail("V4 cells lack identity/audit metadata")
  if (!all(required_features %in% names(features))) fail("V4 merged features lack configured inputs")
  expected_n <- as.integer(config$cell_universe$expected_development_cells)
  if (nrow(cells) != expected_n || nrow(features) != expected_n ||
      !identical(as.character(cells$cell_id), as.character(features$cell_id)) ||
      anyDuplicated(cells$cell_id) || any(cells$split != "development")) {
    fail("V4 development row universe/order differs")
  }
  if (!all(features$nuclei_measurement_status %in% c(
    "present_supported", "absent_supported", "possible_nuclei_mask_miss", "low_quality_nuclei_signal"
  ))) fail("V4 Nuclei measurement status is invalid")
  if (!all(features$dead_measurement_status %in% c(
    "dead_signal_present_supported", "dead_signal_absent_supported",
    "dead_signal_saturated", "dead_signal_background_uncertain"
  ))) fail("V4 Dead measurement status is invalid")

  expanded_manifest <- file.path(expanded, "expanded_projection_manifest.json")
  expanded_summary <- file.path(expanded, "profile_summary.tsv")
  expanded_umap <- file.path(expanded, "profiles", "expanded39", "umap.tsv")
  for (path in c(expanded_manifest, expanded_summary, expanded_umap)) if (!file.exists(path)) fail("Expanded39 audit input is incomplete: ", path)
  expanded_value <- jsonlite::fromJSON(expanded_manifest, simplifyVector = FALSE)
  if (!identical(expanded_value$status, "COMPLETE") || !identical(expanded_value$selected_annotation_profile, "expanded39")) fail("Expanded39 audit manifest differs")

  inputs <- list(
    cells = list(path = paths$cells, sha256 = sha256_file(paths$cells)),
    features = list(path = paths$features, sha256 = sha256_file(paths$features)),
    expanded_projection_manifest = list(path = expanded_manifest, sha256 = sha256_file(expanded_manifest)),
    expanded39_umap = list(path = expanded_umap, sha256 = sha256_file(expanded_umap)),
    config = list(path = config_path, sha256 = sha256_file(config_path)),
    implementation = list(path = implementation, sha256 = implementation_sha)
  )
  output_arg <- path.expand(args$output_dir)
  if (file.exists(output_arg) && !dir.exists(output_arg)) fail("V4 projection output exists and is not a directory")
  output <- normalizePath(output_arg, mustWork = FALSE)
  if (dir.exists(output)) {
    if (!args$overwrite) fail("V4 projection output already exists: ", output)
    verify_existing(output, inputs, implementation_sha)
    cat("v4_projection=", output, "\n", sep = "")
    cat("generation_status=verified_reuse\n")
    return(invisible(0L))
  }

  parent <- dirname(output)
  dir.create(parent, recursive = TRUE, showWarnings = FALSE)
  staging <- tempfile(pattern = paste0(".", basename(output), ".staging."), tmpdir = parent)
  dir.create(staging)
  on.exit(if (dir.exists(staging)) unlink(staging, recursive = TRUE), add = TRUE)
  dir.create(file.path(staging, "profiles", "expanded39_audit"), recursive = TRUE)
  file.copy(expanded_umap, file.path(staging, "profiles", "expanded39_audit", "umap.tsv"))
  file.copy(expanded_summary, file.path(staging, "profiles", "expanded39_audit", "profile_summary.tsv"))

  nuclei_present <- features$nuclei_measurement_status == "present_supported"
  nuclei_reliable <- features$nuclei_measurement_status %in% c("present_supported", "absent_supported")
  dead_present <- features$dead_measurement_status == "dead_signal_present_supported"
  dead_reliable <- features$dead_measurement_status %in% c(
    "dead_signal_present_supported", "dead_signal_absent_supported",
    "dead_signal_saturated"
  )
  winsor <- as.numeric(as_vector(config$preprocessing$winsor_quantiles, "winsor quantiles"))
  all_names <- c(SHAPE, BRIGHTFIELD, NUCLEI, NUCLEI_QC, DEAD, DEAD_QC)
  processed <- matrix(NA_real_, nrow = nrow(features), ncol = length(all_names), dimnames = list(NULL, all_names))
  transform_rows <- list()
  retained_base <- character()
  for (name in all_names) {
    raw <- suppressWarnings(as.numeric(features[[name]]))
    fitted <- fit_feature(
      name, raw, nuclei_present, nuclei_reliable, dead_present, dead_reliable,
      winsor
    )
    block <- if (name %in% SHAPE) "shape" else if (name %in% BRIGHTFIELD) "brightfield" else if (name %in% c(NUCLEI, NUCLEI_QC)) "nuclei" else "dead"
    if (isTRUE(fitted$keep)) {
      processed[, name] <- fitted$value
      retained_base <- c(retained_base, name)
      transform_rows[[length(transform_rows) + 1L]] <- data.frame(
        feature = name, block = block,
        retained_after_transform = TRUE, removal_reason = "", transform = fitted$transform,
        winsor_low = fitted$winsor_low, winsor_high = fitted$winsor_high,
        center = fitted$center, scale = fitted$scale, scale_source = fitted$scale_source,
        fit_rows = fitted$fit_rows,
        conditional_present_only = isTRUE(fitted$conditional_present),
        conditional_dead_present_only = isTRUE(fitted$conditional_dead_present),
        conditional_reliable_mask_only = isTRUE(fitted$conditional_reliable),
        conditional_dead_reliable_only = isTRUE(fitted$conditional_dead_reliable)
      )
    } else {
      transform_rows[[length(transform_rows) + 1L]] <- data.frame(
        feature = name, block = block,
        retained_after_transform = FALSE, removal_reason = fitted$reason, transform = transform_kind(name),
        winsor_low = NA, winsor_high = NA, center = NA, scale = NA, scale_source = "",
        fit_rows = 0L, conditional_present_only = name %in% CONDITIONAL_NUCLEI,
        conditional_dead_present_only = name %in% CONDITIONAL_DEAD,
        conditional_reliable_mask_only = name %in% c("nucleus_absent", "nuclei_count_excess"),
        conditional_dead_reliable_only = name %in% DEAD
      )
    }
  }
  transform_manifest <- do.call(rbind, transform_rows)
  write_tsv(transform_manifest, file.path(staging, "feature_transform_manifest.tsv"))
  missing_required <- setdiff(c(SHAPE, NUCLEI), retained_base)
  if (length(missing_required)) fail("Required V4 Shape/Nuclei feature became constant or unavailable: ", paste(missing_required, collapse = ","))
  if (!length(intersect(DEAD, retained_base))) {
    fail("Every V4 Dead feature became constant or unavailable")
  }
  processed <- processed[, retained_base, drop = FALSE]
  bf_pruning <- prune_block(
    processed, BRIGHTFIELD,
    as.numeric(config$feature_blocks$brightfield$correlation_pruning$threshold),
    "brightfield"
  )
  dead_pruning <- prune_block(
    processed, DEAD,
    as.numeric(config$feature_blocks$dead$correlation_pruning$threshold), "dead"
  )
  if (!length(bf_pruning$retained) || !length(dead_pruning$retained)) fail("V4 correlation pruning removed an entire evidence block")

  retention <- data.frame(
    feature = all_names,
    block = ifelse(all_names %in% SHAPE, "shape", ifelse(all_names %in% BRIGHTFIELD, "brightfield", ifelse(all_names %in% c(NUCLEI, NUCLEI_QC), "nuclei", "dead"))),
    transform_retained = all_names %in% retained_base,
    primary_retained = all_names %in% c(SHAPE, bf_pruning$retained, NUCLEI, dead_pruning$retained),
    reason = ifelse(!(all_names %in% retained_base), "transform_constant_or_unavailable",
      ifelse(all_names %in% BRIGHTFIELD & !(all_names %in% bf_pruning$retained), "brightfield_absolute_spearman_pruned",
        ifelse(all_names %in% DEAD & !(all_names %in% dead_pruning$retained), "dead_absolute_spearman_pruned",
          ifelse(all_names %in% c(NUCLEI_QC, DEAD_QC), "qc_only", "retained"))))
  )
  write_tsv(retention, file.path(staging, "feature_retention_manifest.tsv"))
  write_tsv(bf_pruning$removed, file.path(staging, "brightfield_correlation_pruning.tsv"))
  write_tsv(dead_pruning$removed, file.path(staging, "dead_correlation_pruning.tsv"))

  block_map <- c(
    stats::setNames(rep("shape", length(SHAPE)), SHAPE),
    stats::setNames(rep("brightfield", length(BRIGHTFIELD)), BRIGHTFIELD),
    stats::setNames(rep("nuclei", length(c(NUCLEI, NUCLEI_QC))), c(NUCLEI, NUCLEI_QC)),
    stats::setNames(rep("dead", length(c(DEAD, DEAD_QC))), c(DEAD, DEAD_QC))
  )
  grid_neighbors <- as.integer(as_vector(config$projection_grid$n_neighbors, "n_neighbors"))
  grid_min_dist <- as.numeric(as_vector(config$projection_grid$min_dist, "min_dist"))
  grid_seeds <- as.integer(as_vector(config$projection_grid$seeds, "seeds"))
  sample_n <- min(2000L, nrow(features))
  set.seed(20260813L)
  audit_rows <- sort(sample.int(nrow(features), sample_n, replace = FALSE))
  profile_summaries <- list()
  all_grid_metrics <- list()
  all_block_audit <- list()
  primary_runs <- list()

  for (profile in GENERATED_PROFILES) {
    specification <- profile_spec(profile, bf_pruning$retained, dead_pruning$retained, config)
    active_blocks <- names(specification$blocks)
    names <- unname(unlist(specification$blocks, use.names = FALSE))
    names <- names[names %in% colnames(processed)]
    matrix <- processed[, names, drop = FALSE]
    block_audit <- data.frame()
    if (specification$equal) {
      equalized <- equalize_blocks(matrix, block_map, active_blocks, specification$weights)
      matrix <- equalized$matrix
      block_audit <- equalized$audit
    } else {
      block_audit <- do.call(rbind, lapply(active_blocks, function(block) {
        columns <- names[block_map[names] == block]
        data.frame(
          block = block, feature_count = length(columns),
          mpd_before = mean_pairwise_squared_distance(matrix[, columns, drop = FALSE]),
          multiplier = 1, mpd_after = mean_pairwise_squared_distance(matrix[, columns, drop = FALSE]),
          target_fraction = NA_real_
        )
      }))
      block_audit$observed_fraction <- block_audit$mpd_after / sum(block_audit$mpd_after)
    }
    block_audit$profile <- profile
    all_block_audit[[profile]] <- block_audit[, c("profile", setdiff(names(block_audit), "profile"))]
    profile_dir <- file.path(staging, "profiles", profile)
    dir.create(profile_dir, recursive = TRUE)
    write_tsv(data.frame(cell_id = cells$cell_id, matrix, check.names = FALSE), file.path(profile_dir, "processed_features.tsv"))
    pca <- stats::prcomp(matrix, center = FALSE, scale. = FALSE)
    used <- min(as.integer(config$projection_grid$pca_max_components), ncol(pca$x))
    pcs <- pca$x[, seq_len(used), drop = FALSE]
    variance <- pca$sdev^2
    write_tsv(data.frame(
      component = seq_along(variance), variance = variance,
      proportion = variance / sum(variance), cumulative = cumsum(variance / sum(variance)),
      used_for_umap = seq_along(variance) <= used
    ), file.path(profile_dir, "pca_variance.tsv"))
    weighted <- sweep(pca$rotation[, seq_len(used), drop = FALSE]^2, 2L, variance[seq_len(used)], `*`)
    contribution <- rowSums(weighted)
    contribution <- contribution / sum(contribution)
    write_tsv(data.frame(
      feature = rownames(pca$rotation), block = unname(block_map[rownames(pca$rotation)]),
      weighted_used_pc_contribution = contribution
    ), file.path(profile_dir, "pca_feature_contribution.tsv"))

    combinations <- if (profile %in% DEATH_RESOLUTION_PROFILES) expand.grid(
      n_neighbors = grid_neighbors, min_dist = grid_min_dist, seed = grid_seeds,
      KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
    ) else data.frame(n_neighbors = 30L, min_dist = 0.1, seed = grid_seeds[[1L]])
    input_knn <- neighbor_ids(pcs[audit_rows, , drop = FALSE], 15L)
    run_neighbors <- list()
    for (index in seq_len(nrow(combinations))) {
      row <- combinations[index, ]
      set.seed(as.integer(row$seed))
      coordinates <- uwot::umap(
        pcs, n_neighbors = as.integer(row$n_neighbors), min_dist = as.numeric(row$min_dist),
        n_components = 2L, metric = "euclidean", n_threads = 1L,
        init = "spectral", ret_model = FALSE, verbose = FALSE
      )
      run_id <- sprintf("nn%d_md%s_seed%d", as.integer(row$n_neighbors), gsub("\\.", "p", format(as.numeric(row$min_dist), trim = TRUE)), as.integer(row$seed))
      run_dir <- file.path(profile_dir, "grid", run_id)
      dir.create(run_dir, recursive = TRUE)
      write_tsv(data.frame(cell_id = cells$cell_id, Dim1 = coordinates[, 1L], Dim2 = coordinates[, 2L]), file.path(run_dir, "umap.tsv"))
      observed_knn <- neighbor_ids(coordinates[audit_rows, , drop = FALSE], 15L)
      run_neighbors[[run_id]] <- observed_knn
      overlap <- neighbor_overlap(input_knn, observed_knn)
      all_grid_metrics[[paste(profile, run_id, sep = "::")]] <- data.frame(
        profile = profile, run_id = run_id, n_neighbors = as.integer(row$n_neighbors),
        min_dist = as.numeric(row$min_dist), seed = as.integer(row$seed),
        input_umap_neighbor_overlap = overlap, mean_seed_neighbor_jaccard = NA_real_,
        selection_score = NA_real_
      )
      if (profile == PRIMARY_PROFILE) primary_runs[[run_id]] <- list(coordinates = coordinates, knn = observed_knn)
    }
    metrics <- do.call(rbind, all_grid_metrics[grepl(paste0("^", profile, "::"), names(all_grid_metrics))])
    if (profile %in% DEATH_RESOLUTION_PROFILES) {
      parameter_key <- paste(metrics$n_neighbors, metrics$min_dist, sep = "|")
      for (key in unique(parameter_key)) {
        indices <- which(parameter_key == key)
        values <- vapply(indices, function(i) {
          others <- setdiff(indices, i)
          if (!length(others)) return(1)
          mean(vapply(others, function(j) {
            neighbor_jaccard(run_neighbors[[metrics$run_id[[i]]]], run_neighbors[[metrics$run_id[[j]]]])
          }, numeric(1)))
        }, numeric(1))
        metrics$mean_seed_neighbor_jaccard[indices] <- values
      }
      metrics$selection_score <- 0.7 * metrics$input_umap_neighbor_overlap + 0.3 * metrics$mean_seed_neighbor_jaccard
    } else {
      metrics$mean_seed_neighbor_jaccard <- 1
      metrics$selection_score <- metrics$input_umap_neighbor_overlap
    }
    for (index in seq_len(nrow(metrics))) all_grid_metrics[[paste(profile, metrics$run_id[[index]], sep = "::")]] <- metrics[index, ]
    profile_summaries[[profile]] <- data.frame(
      profile = profile, feature_count = ncol(matrix), pca_components = used,
      block_equalized = specification$equal, active_blocks = paste(active_blocks, collapse = ","),
      default_run_id = metrics$run_id[[which.max(metrics$selection_score)]],
      best_selection_score = max(metrics$selection_score)
    )
  }

  grid_metrics <- do.call(rbind, all_grid_metrics)
  rownames(grid_metrics) <- NULL
  write_tsv(grid_metrics, file.path(staging, "umap_grid_metrics.tsv"))
  write_tsv(do.call(rbind, all_block_audit), file.path(staging, "block_distance_contribution.tsv"))
  profile_summary <- do.call(rbind, profile_summaries)
  write_tsv(profile_summary, file.path(staging, "profile_summary.tsv"))
  selected_metrics <- grid_metrics[grid_metrics$profile == PRIMARY_PROFILE, , drop = FALSE]
  selected_index <- which.max(selected_metrics$selection_score)
  selected_run <- selected_metrics$run_id[[selected_index]]
  selected_coordinates <- primary_runs[[selected_run]]$coordinates
  write_tsv(data.frame(cell_id = cells$cell_id, Dim1 = selected_coordinates[, 1L], Dim2 = selected_coordinates[, 2L]), file.path(staging, "umap.tsv"))

  cluster <- dbscan::hdbscan(selected_coordinates, minPts = min(100L, max(5L, floor(nrow(cells) / 100L))))$cluster
  write_tsv(data.frame(
    cell_id = cells$cell_id, diagnostic_cluster = as.integer(cluster),
    cluster_role = "navigation_only_not_cell_state_label"
  ), file.path(staging, "diagnostic_clusters.tsv"))
  cluster_summary <- as.data.frame(table(cluster = as.integer(cluster)), stringsAsFactors = FALSE)
  names(cluster_summary)[[2L]] <- "row_count"
  cluster_summary$fraction <- cluster_summary$row_count / nrow(cells)
  write_tsv(cluster_summary, file.path(staging, "diagnostic_cluster_summary.tsv"))

  selected_knn <- neighbor_ids(selected_coordinates[audit_rows, , drop = FALSE], 15L)
  batch_rows <- list()
  audit_metadata <- data.frame(
    well = cells$well[audit_rows], site = as.character(cells$site[audit_rows]),
    context_key = cells$context_key[audit_rows],
    nuclei_measurement_status = features$nuclei_measurement_status[audit_rows],
    dead_measurement_status = features$dead_measurement_status[audit_rows],
    stringsAsFactors = FALSE
  )
  for (name in names(audit_metadata)) {
    value <- audit_metadata[[name]]
    observed <- mean(vapply(seq_len(nrow(selected_knn)), function(index) mean(value[selected_knn[index, ]] == value[[index]]), numeric(1)))
    baseline <- sum((table(value) / length(value))^2)
    batch_rows[[length(batch_rows) + 1L]] <- data.frame(
      metadata = name, sample_rows = length(value), observed_same_neighbor_fraction = observed,
      global_same_fraction = baseline, enrichment_over_global = observed - baseline,
      interpretation = if (name %in% c("nuclei_measurement_status", "dead_measurement_status")) "measurement_status_audit_not_batch" else "known_design_or_batch_audit_not_optimized"
    )
  }
  write_tsv(do.call(rbind, batch_rows), file.path(staging, "batch_mixing_audit.tsv"))
  write_tsv(data.frame(
    profile = names(profile_summaries),
    active_blocks = vapply(profile_summaries, function(value) value$active_blocks[[1L]], character(1)),
    best_selection_score = vapply(profile_summaries, function(value) value$best_selection_score[[1L]], numeric(1))
  ), file.path(staging, "ablation_summary.tsv"))

  selected_block <- do.call(rbind, all_block_audit)[do.call(rbind, all_block_audit)$profile == PRIMARY_PROFILE, , drop = FALSE]
  tolerance <- as.numeric(config$block_equalization$acceptance_absolute_percentage_points) / 100
  equal_pass <- nrow(selected_block) == 4L && all(abs(selected_block$observed_fraction - 0.25) <= tolerance)
  nuclei_status_counts <- as.data.frame(table(features$nuclei_measurement_status), stringsAsFactors = FALSE)
  names(nuclei_status_counts) <- c("status", "row_count")
  nuclei_status_counts$fraction <- nuclei_status_counts$row_count / nrow(features)
  write_tsv(nuclei_status_counts, file.path(staging, "nuclei_measurement_status_summary.tsv"))
  dead_status_counts <- as.data.frame(table(features$dead_measurement_status), stringsAsFactors = FALSE)
  names(dead_status_counts) <- c("status", "row_count")
  dead_status_counts$fraction <- dead_status_counts$row_count / nrow(features)
  write_tsv(dead_status_counts, file.path(staging, "dead_measurement_status_summary.tsv"))
  decision <- list(
    schema_version = "multimodal_cell_state_v4_calibration_decision_v1",
    status = "COMPLETE",
    representation_gate = if (equal_pass) "GO_TO_INDEPENDENT_ANCHOR_REVIEW" else "NO_GO_BLOCK_EQUALIZATION_FAILED",
    primary_profile = PRIMARY_PROFILE,
    selected_run_id = selected_run,
    selected_n_neighbors = as.integer(selected_metrics$n_neighbors[[selected_index]]),
    selected_min_dist = as.numeric(selected_metrics$min_dist[[selected_index]]),
    selected_seed = as.integer(selected_metrics$seed[[selected_index]]),
    selected_score = as.numeric(selected_metrics$selection_score[[selected_index]]),
    block_equalization_pass = equal_pass,
    balanced_dead_cell_island_required = FALSE,
    death_resolution_selection_status = "PENDING_INDEPENDENT_ANCHOR_REVIEW",
    death_resolution_target = "one_broad_dead_region_with_internal_continuous_state_geometry",
    scientific_boundary = paste(
      "Balanced UMAP and diagnostic clusters organize multimodal morphology; they are not cell-state labels.",
      "Independent three-channel anchor labels select a death-resolution UMAP before broad-region annotation."
    )
  )
  jsonlite::write_json(decision, file.path(staging, "calibration_decision.json"), pretty = TRUE, auto_unbox = TRUE)

  artifacts <- recursive_hashes(staging)
  manifest <- list(
    schema_version = SCHEMA_VERSION, status = "COMPLETE",
    method_version = "sum159_multimodal_cell_state_v4", inputs = inputs,
    row_count = nrow(cells), primary_profile = PRIMARY_PROFILE,
    selected_run_id = selected_run,
    selected_feature_columns = as.list(c(SHAPE, bf_pruning$retained, NUCLEI, dead_pruning$retained)),
    brightfield_pruned_columns = as.list(setdiff(BRIGHTFIELD, bf_pruning$retained)),
    dead_pruned_columns = as.list(setdiff(DEAD, dead_pruning$retained)),
    zero_aware_policy = list(nuclei = config$nuclei_zero_aware, dead = config$dead_zero_aware),
    block_equalization = config$block_equalization,
    projection_grid = config$projection_grid,
    heldout_read = FALSE,
    forbidden_inputs_read = list(),
    balanced_dead_cell_island_required = FALSE,
    death_resolution_profiles = as.list(DEATH_RESOLUTION_PROFILES),
    output_file_sha256 = as.list(artifacts)
  )
  manifest_path <- file.path(staging, "projection_manifest.json")
  jsonlite::write_json(manifest, manifest_path, pretty = TRUE, auto_unbox = TRUE, null = "null")
  identity <- list(
    schema_version = IDENTITY_SCHEMA_VERSION, status = "COMPLETE",
    projection_manifest_sha256 = sha256_file(manifest_path),
    implementation_sha256 = implementation_sha, output_file_sha256 = as.list(artifacts)
  )
  jsonlite::write_json(identity, file.path(staging, "projection_generation_identity.json"), pretty = TRUE, auto_unbox = TRUE)
  if (!identical(sha256_file(implementation), implementation_sha)) fail("V4 projection implementation changed during execution")
  if (file.exists(output) || dir.exists(output)) fail("V4 projection output appeared during staging")
  if (!file.rename(staging, output)) fail("Failed to install V4 projection atomically")
  cat("v4_projection=", output, "\n", sep = "")
  cat("primary_profile=", PRIMARY_PROFILE, "\n", sep = "")
  cat("selected_run_id=", selected_run, "\n", sep = "")
  cat("block_equalization=", if (equal_pass) "PASS" else "FAIL", "\n", sep = "")
  cat("generation_status=created\n")
  invisible(0L)
}

tryCatch(main(), error = function(error) {
  message("ERROR: ", conditionMessage(error))
  quit(save = "no", status = 1L, runLast = FALSE)
})
