args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: install_r_packages.R <r-packages.lock.tsv> <package-dir>")
}

lock_path <- args[[1L]]
package_dir <- args[[2L]]
lock <- read.delim(lock_path, check.names = FALSE, stringsAsFactors = FALSE)
required <- c("install_order", "package", "version", "filename")
if (!all(required %in% names(lock))) {
  stop("R package lock is missing required columns")
}
lock <- lock[order(lock$install_order), , drop = FALSE]
if (!identical(lock$install_order, seq_len(nrow(lock)))) {
  stop("R package install_order must be contiguous and start at one")
}
target_library <- Sys.getenv("R_LIBS_SITE", unset = "")
if (!nzchar(target_library)) {
  stop("R_LIBS_SITE must name the immutable container site library")
}
dir.create(target_library, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(target_library, .libPaths()))

for (index in seq_len(nrow(lock))) {
  row <- lock[index, , drop = FALSE]
  archive <- file.path(package_dir, row$filename)
  if (!file.exists(archive)) {
    stop(sprintf("Missing locked R package archive: %s", archive))
  }
  message(sprintf(
    "Installing binary R package %02d/%02d: %s %s",
    index,
    nrow(lock),
    row$package,
    row$version
  ))
  install.packages(
    archive,
    repos = NULL,
    type = "source",
    lib = target_library,
    quiet = TRUE
  )
  actual <- packageDescription(row$package, lib.loc = target_library)[["Version"]]
  if (!identical(actual, row$version)) {
    stop(sprintf(
      "Installed version mismatch for %s: expected=%s actual=%s",
      row$package,
      row$version,
      actual
    ))
  }
}

message(sprintf("Installed %d locked binary R packages", nrow(lock)))
