args0 <- commandArgs(FALSE)
script <- sub("^--file=", "", grep("^--file=", args0, value = TRUE))
root <- if (length(script) > 0) dirname(dirname(normalizePath(script[1]))) else getwd()

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 2) {
  infile <- args[1]
  outdir <- args[2]
} else {
  infile <- file.path(root, "data", "norw010.rwl")
  outdir <- file.path(root, "results", "dplr")
}

library(dplR)

# keep_series <- NULL
# drop_series <- NULL
drop_series <- c(
        "ff201a",
        "ff201b",
        "ff202a",
        "ff202b",
        "ff203a",
        "ff203b",
        "ff204a",
        "ff204b",
        "ff205a",
        "ff205b",
        "ff207a",
        "ff207b",
        "ff208a",
        "ff208b",
        "ff209a",
        "ff209b",
        "ff210a",
        "ff210b",
        "ff211a",
        "ff211b",
        "ff213a",
        "ff213b",
        "ff216a",
        "ff216b",
        "ff217a",
        "ff217b",
        "ff218a",
        "ff218b",
        "ff220a",
        "ff220b",
        "ff221a",
        "ff221b",
        "ff222a",
        "ff222b",
        "ff224a",
        "ff224b",
        "ff225a",
        "ff225b",
        "ff228a",
        "ff228b",
        "ff229a",
        "ff229b",
        "ff252a",
        "ff252b",
        "ff253a",
        "ff253c",
        "ff254a",
        "ff254b",
        "ff255a",
        "ff255b",
        "ff256a",
        "ff256b",
        "ff257b",
        "ff257c",
        "ff260b",
        "ff260c"
)

special_na <- c(0, 0.001, 0.005, 0.010)
tol <- 1e-12

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

rwl <- read.rwl(infile)

# if (!is.null(keep_series) && length(keep_series) > 0) {
#   missing <- setdiff(keep_series, colnames(rwl))
#   if (length(missing) > 0) stop(paste("missing series:", paste(missing, collapse = ", ")))
#   rwl <- rwl[, keep_series, drop = FALSE]
# }

if (!is.null(drop_series) && length(drop_series) > 0) {
  missing <- setdiff(drop_series, colnames(rwl))
  if (length(missing) > 0) stop(paste("missing drop_series:", paste(missing, collapse = ", ")))
  rwl <- rwl[, setdiff(colnames(rwl), drop_series), drop = FALSE]
}

mat <- as.matrix(rwl)

special_mask <- Reduce(`|`, lapply(special_na, function(v) abs(mat - v) < tol))
special_mask[is.na(special_mask)] <- FALSE

orig_mask <- is.na(mat) | special_mask
orig_mask[is.na(orig_mask)] <- FALSE

special_counts <- sapply(special_na, function(v) sum(abs(mat - v) < tol, na.rm = TRUE))
names(special_counts) <- as.character(special_na)

rwl[special_mask] <- NA

keep_year <- rowSums(!is.na(as.matrix(rwl))) > 0
rwl <- rwl[keep_year, , drop = FALSE]
orig_mask <- orig_mask[keep_year, , drop = FALSE]

rwl <- fill.internal.NA(rwl, fill = "Linear")
rwl <- powt(rwl, method = "cook")

sf <- ssf(
  rwl = rwl,
  method = "AgeDepSpline",
  nyrs = 50,
  difference = FALSE,
  max.iterations = 25,
  return.info = TRUE,
  verbose = FALSE
)

rwi <- as.data.frame(sf$sfRWI_Array[, , sf$k + 1])

m0 <- orig_mask[rownames(rwi), colnames(rwi), drop = FALSE]
rwi[m0] <- NA

n_series_out <- ncol(rwi)
outfile <- file.path(outdir, paste0("tree_rwi_dplr_", n_series_out, ".csv"))

write.csv(
  data.frame(year = as.integer(rownames(rwi)), rwi),
  outfile,
  row.names = FALSE
)

crn <- chron(
  rwi,
  biweight = TRUE,
  prewhiten = FALSE
)

write.csv(
  data.frame(
    year = as.integer(rownames(crn)),
    crn
  ),
  file.path(outdir, paste0("chronology_dplr_", n_series_out, ".csv")),
  row.names = FALSE
)
