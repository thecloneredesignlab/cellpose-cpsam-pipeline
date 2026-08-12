lock_path <- "/opt/hpc-environment/locks/r-packages.lock.tsv"
lock <- read.delim(lock_path, check.names = FALSE, stringsAsFactors = FALSE)

if (!identical(paste(R.version$major, R.version$minor, sep = "."), "4.2.3")) {
  stop(sprintf("Unexpected R version: %s", R.version.string))
}
if (!identical(normalizePath(R.home()), "/opt/R/4.2.3/lib/R")) {
  stop(sprintf("Unexpected R home: %s", R.home()))
}
if (any(grepl("/root|/home/", .libPaths()))) {
  stop(sprintf("User library leaked into .libPaths(): %s", paste(.libPaths(), collapse = ":")))
}

for (index in seq_len(nrow(lock))) {
  package <- lock$package[[index]]
  expected <- lock$version[[index]]
  if (!requireNamespace(package, quietly = TRUE)) {
    stop(sprintf("Locked R package cannot be loaded: %s", package))
  }
  actual <- packageDescription(package)[["Version"]]
  if (!identical(actual, expected)) {
    stop(sprintf(
      "R package version mismatch for %s: expected=%s actual=%s",
      package,
      expected,
      actual
    ))
  }
}

work_dir <- tempfile("r-environment-smoke-")
dir.create(work_dir)
on.exit(unlink(work_dir, recursive = TRUE), add = TRUE)
image <- matrix(seq(0, 1, length.out = 16), nrow = 4L, ncol = 4L)

png_path <- file.path(work_dir, "roundtrip.png")
png::writePNG(image, png_path)
png_result <- png::readPNG(png_path)
stopifnot(identical(dim(png_result), c(4L, 4L)))

jpeg_path <- file.path(work_dir, "roundtrip.jpg")
jpeg::writeJPEG(image, jpeg_path, quality = 1)
jpeg_result <- jpeg::readJPEG(jpeg_path)
stopifnot(identical(dim(jpeg_result), c(4L, 4L)), all(is.finite(jpeg_result)))

tiff_path <- file.path(work_dir, "roundtrip.tiff")
invisible(tiff::writeTIFF(image, tiff_path, bits.per.sample = 8L))
tiff_result <- tiff::readTIFF(tiff_path)
stopifnot(identical(dim(tiff_result), c(4L, 4L)), all(is.finite(tiff_result)))

json_value <- list(runtime = "R-4.2.3", valid = TRUE, values = c(1L, 2L, 3L))
json_roundtrip <- jsonlite::fromJSON(jsonlite::toJSON(json_value, auto_unbox = TRUE))
stopifnot(identical(json_roundtrip$runtime, json_value$runtime))
yaml_roundtrip <- yaml::yaml.load(yaml::as.yaml(json_value))
stopifnot(identical(yaml_roundtrip$runtime, json_value$runtime))
stopifnot(nchar(digest::digest("cellpose", algo = "sha256", serialize = FALSE)) == 64L)

set.seed(9182)
class <- factor(rep(c("a", "b", "c"), each = 20L))
features <- cbind(
  a = as.numeric(class == "a") + rnorm(60L, sd = 0.1),
  b = as.numeric(class == "b") + rnorm(60L, sd = 0.1),
  noise = rnorm(60L)
)
fit <- glmnet::glmnet(
  features,
  class,
  family = "multinomial",
  type.multinomial = "grouped",
  alpha = 0.5,
  lambda = c(0.1, 0.05),
  standardize = FALSE
)
probabilities <- predict(fit, newx = features, s = 0.1, type = "response")[, , 1L]
stopifnot(
  all(is.finite(probabilities)),
  max(abs(rowSums(probabilities) - 1)) < 1e-12
)

umap_input <- matrix(
  sin(seq_len(300L) / 7) + cos(seq_len(300L) / 11),
  nrow = 60L,
  ncol = 5L
)
umap_args <- list(
  X = umap_input,
  n_neighbors = 10L,
  n_components = 2L,
  metric = "euclidean",
  min_dist = 0.1,
  seed = 2468L,
  n_threads = 1L,
  n_sgd_threads = 1L,
  fast_sgd = FALSE,
  batch = FALSE,
  rng_type = "deterministic",
  ret_model = FALSE,
  verbose = FALSE
)
umap_one <- do.call(uwot::umap, umap_args)
umap_two <- do.call(uwot::umap, umap_args)
stopifnot(
  identical(umap_one, umap_two),
  identical(dim(umap_one), c(60L, 2L)),
  all(is.finite(umap_one))
)

shared_libraries <- list.files(
  "/opt/R/4.2.3/lib/R/site-library",
  pattern = "\\.so$",
  recursive = TRUE,
  full.names = TRUE
)
for (shared_library in shared_libraries) {
  output <- system2("ldd", shared_library, stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status")
  if ((!is.null(status) && status != 0L) || any(grepl("not found", output, fixed = TRUE))) {
    stop(sprintf(
      "Shared-library linkage failed for %s:\n%s",
      shared_library,
      paste(output, collapse = "\n")
    ))
  }
}

cat(sprintf("r_version=%s\n", R.version.string))
cat(sprintf("r_package_count=%d\n", nrow(lock)))
cat("glmnet_grouped_multinomial=PASS\n")
cat("uwot_deterministic_repeat=PASS\n")
cat("r_environment_verification=PASS\n")
