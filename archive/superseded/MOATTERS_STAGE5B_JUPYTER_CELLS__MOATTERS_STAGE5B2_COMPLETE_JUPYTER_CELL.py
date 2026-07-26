# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
from pathlib import Path
import subprocess

# ============================================================
# Stage 5B-2 — Run GSVA and Pathifier from one Jupyter Python cell
# ============================================================

rscript = r"C:\Program Files\R\R-4.6.0\bin\Rscript.exe"

r_code = r'''
#!/usr/bin/env Rscript

INPUT_DIR <- "D:/MOATTERS-Output/MOATTERS_STAGE5B_BENCHMARK_INPUTS"
OUTPUT_DIR <- "D:/MOATTERS-Output/MOATTERS_STAGE5B_GSVA_PATHIFIER_SCORES"

dir.create(OUTPUT_DIR, recursive=TRUE, showWarnings=FALSE)
dir.create(file.path(OUTPUT_DIR, "logs"), showWarnings=FALSE)

options(repos = c(CRAN = "https://cloud.r-project.org"))

install_cran_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly=TRUE)) {
    install.packages(pkg)
  }
}

install_bioc_if_missing <- function(pkg) {
  if (!requireNamespace("BiocManager", quietly=TRUE)) {
    install.packages("BiocManager")
  }
  if (!requireNamespace(pkg, quietly=TRUE)) {
    BiocManager::install(pkg, ask=FALSE, update=FALSE)
  }
}

install_cran_if_missing("jsonlite")
install_bioc_if_missing("GSVA")
install_bioc_if_missing("pathifier")

suppressPackageStartupMessages({
  library(GSVA)
  library(pathifier)
  library(jsonlite)
})

expr_file <- file.path(INPUT_DIR, "locked_expression_primary_plus_normal.csv.gz")
gmt_file <- file.path(INPUT_DIR, "locked_30_BRCA_BP_terms.gmt")
sample_file <- file.path(INPUT_DIR, "sample_manifest.csv")

required_files <- c(expr_file, gmt_file, sample_file)
if (!all(file.exists(required_files))) {
  missing <- required_files[!file.exists(required_files)]
  stop(paste("Stage 5B-1 input file(s) missing:", paste(missing, collapse="; ")))
}

cat("Reading locked expression matrix...\n")
expr_df <- read.csv(expr_file, check.names=FALSE, stringsAsFactors=FALSE)
gene_col <- names(expr_df)[1]
genes <- as.character(expr_df[[gene_col]])
expr <- as.matrix(expr_df[, -1, drop=FALSE])
storage.mode(expr) <- "numeric"
rownames(expr) <- genes

cat("Expression matrix:", nrow(expr), "genes x", ncol(expr), "samples\n")

sample_manifest <- read.csv(sample_file, stringsAsFactors=FALSE)
sample_manifest <- sample_manifest[match(colnames(expr), sample_manifest$sample_id), ]
if (any(is.na(sample_manifest$sample_id))) {
  stop("Expression columns and sample manifest are not aligned.")
}

normals <- as.logical(sample_manifest$is_adjacent_normal)

read_locked_gmt <- function(path) {
  lines <- readLines(path, warn=FALSE)
  out <- lapply(lines, function(line) {
    parts <- strsplit(line, "\t", fixed=TRUE)[[1]]
    if (length(parts) < 3) return(character(0))
    unique(parts[-c(1, 2)])
  })
  names(out) <- vapply(lines, function(line) {
    strsplit(line, "\t", fixed=TRUE)[[1]][1]
  }, character(1))
  out
}

gene_sets <- read_locked_gmt(gmt_file)
gene_sets <- lapply(gene_sets, intersect, y=rownames(expr))
gene_sets <- gene_sets[lengths(gene_sets) >= 5]

if (length(gene_sets) != 30) {
  stop(paste("Expected 30 usable locked gene sets, found", length(gene_sets)))
}

cat("Usable locked gene sets:", length(gene_sets), "\n")

cat("Running GSVA...\n")

if (exists("gsvaParam", where=asNamespace("GSVA"), inherits=FALSE)) {
  gsva_par <- GSVA::gsvaParam(
    exprData=expr,
    geneSets=gene_sets,
    kcdf="Gaussian",
    minSize=5,
    maxSize=Inf
  )
  gsva_scores <- GSVA::gsva(gsva_par, verbose=TRUE)
} else {
  gsva_scores <- GSVA::gsva(
    expr,
    gene_sets,
    method="gsva",
    kcdf="Gaussian",
    min.sz=5,
    max.sz=Inf,
    verbose=TRUE
  )
}

write.csv(
  t(gsva_scores),
  file.path(OUTPUT_DIR, "GSVA_scores_sample_by_BP.csv"),
  row.names=TRUE
)

cat("GSVA completed.\n")
cat("Running Pathifier...\n")

pathifier_log <- file.path(OUTPUT_DIR, "logs", "pathifier.log")

pds <- pathifier::quantify_pathways_deregulation(
  data=expr,
  allgenes=rownames(expr),
  syms=gene_sets,
  pathwaynames=names(gene_sets),
  normals=normals,
  attempts=100,
  maximize_stability=TRUE,
  logfile=pathifier_log,
  min_exp=min(expr, na.rm=TRUE),
  min_std=0.4
)

path_scores <- pds$scores

if (ncol(path_scores) == ncol(expr)) {
  path_out <- t(path_scores)
} else if (nrow(path_scores) == ncol(expr)) {
  path_out <- path_scores
} else {
  stop(paste("Unexpected Pathifier score dimensions:", nrow(path_scores), "x", ncol(path_scores)))
}

write.csv(
  path_out,
  file.path(OUTPUT_DIR, "Pathifier_scores_sample_by_BP.csv"),
  row.names=TRUE
)

cat("Pathifier completed.\n")

manifest <- list(
  status="PASS",
  n_gene_sets=length(gene_sets),
  n_genes=nrow(expr),
  n_samples=ncol(expr),
  n_primary_tumors=sum(sample_manifest$is_primary_tumor),
  n_adjacent_normals=sum(sample_manifest$is_adjacent_normal),
  GSVA_version=as.character(packageVersion("GSVA")),
  pathifier_version=as.character(packageVersion("pathifier")),
  GSVA_kcdf="Gaussian",
  Pathifier_attempts=100,
  Pathifier_min_exp=min(expr, na.rm=TRUE),
  Pathifier_min_std=0.4
)

write_json(
  manifest,
  file.path(OUTPUT_DIR, "analysis_manifest.json"),
  pretty=TRUE,
  auto_unbox=TRUE
)

cat("PASS — Stage 5B-2 GSVA / Pathifier scoring completed\n")
cat("Output:", OUTPUT_DIR, "\n")
'''

r_file = Path(r"D:\MOATTERS-Output\MOATTERS_STAGE5B2_RUN_GSVA_PATHIFIER.R")
r_file.parent.mkdir(parents=True, exist_ok=True)
r_file.write_text(r_code, encoding="utf-8")

print("Rscript:", rscript)
print("R file:", r_file)

version_check = subprocess.run(
    [rscript, "--version"],
    capture_output=True,
    text=True,
    timeout=60,
)

print("Rscript return code:", version_check.returncode)
if version_check.stdout:
    print(version_check.stdout)
if version_check.stderr:
    print(version_check.stderr)

if version_check.returncode != 0:
    raise RuntimeError("Rscript exists but could not be executed.")

print("Running GSVA and Pathifier...")

process = subprocess.Popen(
    [rscript, str(r_file)],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

for line in process.stdout:
    print(line, end="")

return_code = process.wait()

if return_code != 0:
    raise RuntimeError(f"Stage 5B-2 failed with exit code {return_code}")

print("PASS — Stage 5B-2 completed successfully")
