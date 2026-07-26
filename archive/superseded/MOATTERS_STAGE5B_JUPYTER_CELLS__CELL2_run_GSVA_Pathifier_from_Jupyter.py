# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
from pathlib import Path
import subprocess

r_code = r"""
#!/usr/bin/env Rscript
# Stage 5B-2 — Generate GSVA and Pathifier scores from locked benchmark inputs.

INPUT_DIR <- "D:/MOATTERS-Output/MOATTERS_STAGE5B_BENCHMARK_INPUTS"
OUTPUT_DIR <- "D:/MOATTERS-Output/MOATTERS_STAGE5B_GSVA_PATHIFIER_SCORES"

dir.create(OUTPUT_DIR, recursive=TRUE, showWarnings=FALSE)
dir.create(file.path(OUTPUT_DIR, "logs"), showWarnings=FALSE)

install_bioc_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly=TRUE)) {
    if (!requireNamespace("BiocManager", quietly=TRUE)) {
      install.packages("BiocManager")
    }
    BiocManager::install(pkg, ask=FALSE, update=FALSE)
  }
}

install_cran_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly=TRUE)) {
    install.packages(pkg)
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

if (!file.exists(expr_file) || !file.exists(gmt_file) || !file.exists(sample_file)) {
  stop("Stage 5B-1 inputs are missing. Run the preparation script first.")
}

expr_df <- read.csv(expr_file, check.names=FALSE)
gene_col <- names(expr_df)[1]
genes <- as.character(expr_df[[gene_col]])
expr <- as.matrix(expr_df[, -1, drop=FALSE])
storage.mode(expr) <- "numeric"
rownames(expr) <- genes

sample_manifest <- read.csv(sample_file, stringsAsFactors=FALSE)
sample_manifest <- sample_manifest[match(colnames(expr), sample_manifest$sample_id), ]
if (any(is.na(sample_manifest$sample_id))) {
  stop("Expression samples and sample manifest are misaligned.")
}
normals <- as.logical(sample_manifest$is_adjacent_normal)

read_locked_gmt <- function(path) {
  lines <- readLines(path, warn=FALSE)
  out <- lapply(lines, function(line) {
    parts <- strsplit(line, "\t", fixed=TRUE)[[1]]
    parts[-c(1,2)]
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

# The Xena expression matrix is continuous normalized expression, therefore
# Gaussian kernel behavior is used for GSVA.
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
    expr, gene_sets, method="gsva", kcdf="Gaussian",
    min.sz=5, max.sz=Inf, verbose=TRUE
  )
}

write.csv(
  t(gsva_scores),
  file.path(OUTPUT_DIR, "GSVA_scores_sample_by_BP.csv"),
  row.names=TRUE
)

# Pathifier requires a logical normal-sample indicator. The minimum expression
# threshold is set to the observed minimum so the benchmark does not impose an
# additional arbitrary floor on the already normalized Xena matrix.
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
  stop("Unexpected Pathifier score dimensions.")
}

write.csv(
  path_out,
  file.path(OUTPUT_DIR, "Pathifier_scores_sample_by_BP.csv"),
  row.names=TRUE
)

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
  pretty=TRUE, auto_unbox=TRUE
)

cat("PASS — Stage 5B-2 GSVA / Pathifier scoring completed\n")
cat("Output:", OUTPUT_DIR, "\n")
"""

r_file = Path(r"D:\MOATTERS-Output\MOATTERS_STAGE5B2_RUN_GSVA_PATHIFIER.R")
r_file.write_text(r_code, encoding="utf-8")

candidates = [
    "Rscript",
    r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe",
    r"C:\Program Files\R\R-4.4.3\bin\Rscript.exe",
    r"C:\Program Files\R\R-4.4.2\bin\Rscript.exe",
    r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe",
    r"C:\Program Files\R\R-4.3.3\bin\Rscript.exe",
]

rscript = None
for candidate in candidates:
    try:
        check = subprocess.run(
            [candidate, "--version"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if check.returncode == 0:
            rscript = candidate
            break
    except Exception:
        pass

if rscript is None:
    raise RuntimeError(
        "找不到 Rscript。請確認已安裝 R，或把實際 Rscript.exe 路徑加入 candidates。"
    )

print("Using:", rscript)
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
    raise RuntimeError(f"R stage failed with exit code {return_code}")

print("PASS — Stage 5B-2 completed")
