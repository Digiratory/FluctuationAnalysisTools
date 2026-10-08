args0 <- commandArgs(FALSE)
script <- sub("^--file=", "", grep("^--file=", args0, value = TRUE))
root <- if (length(script) > 0) dirname(dirname(normalizePath(script[1]))) else getwd()

outdir <- file.path(root, "data")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

urls <- c(
  "https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/europe/norw010-rwl-noaa.txt",
  "https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/europe/norw010.rwl",
  "https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/correlation-stats/norw010.txt"
)

for (u in urls) {
  f <- file.path(outdir, basename(u))
  download.file(u, f, mode = "wb", quiet = FALSE)
}
