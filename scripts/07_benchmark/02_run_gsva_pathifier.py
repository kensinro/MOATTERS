# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
from pathlib import Path
from moatters.config import data_path, output_path, RSCRIPT
import subprocess

# ============================================================
# Stage 5B-2C — Corrected Pathifier rerun from Jupyter
#
# Key fixes:
# 1. Save the full Pathifier object immediately after computation.
# 2. Correctly convert pds$scores, which is a list-column of
#    pathway score vectors, into a pathway x sample matrix.
# 3. Use attempts=20 to reduce rerun time while still performing
#    stability estimation.
# 4. Preserve the existing GSVA output.
# ============================================================

rscript = str(RSCRIPT)

r_code = r'''
#!/usr/bin/env Rscript

INPUT_DIR <- "__MOATTERS_BENCHMARK_INPUT_DIR__"
OUTPUT_DIR <- "__MOATTERS_BENCHMARK_SCORE_DIR__"

dir.create(OUTPUT_DIR, recursive=TRUE, showWarnings=FALSE)
dir.create(file.path(OUTPUT_DIR, "logs"), showWarnings=FALSE)

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("jsonlite", quietly=TRUE)) {
  install.packages("jsonlite")
}
if (!requireNamespace("BiocManager", quietly=TRUE)) {
  install.packages("BiocManager")
}
if (!requireNamespace("GSVA", quietly=TRUE)) {
  BiocManager::install("GSVA", ask=FALSE, update=FALSE)
}
if (!requireNamespace("pathifier", quietly=TRUE)) {
  BiocManager::install("pathifier", ask=FALSE, update=FALSE)
}

suppressPackageStartupMessages({
  library(GSVA)
  library(pathifier)
  library(jsonlite)
})

expr_file <- file.path(
  INPUT_DIR,
  "locked_expression_primary_plus_normal.csv.gz"
)

gmt_file <- file.path(
  INPUT_DIR,
  "locked_30_BRCA_BP_terms.gmt"
)

sample_file <- file.path(
  INPUT_DIR,
  "sample_manifest.csv"
)

required_files <- c(
  expr_file,
  gmt_file,
  sample_file
)

if (!all(file.exists(required_files))) {
  missing <- required_files[!file.exists(required_files)]
  stop(
    paste(
      "Stage 5B-1 input file(s) missing:",
      paste(missing, collapse="; ")
    )
  )
}

cat("Reading locked expression matrix...\n")

expr_df <- read.csv(
  expr_file,
  check.names=FALSE,
  stringsAsFactors=FALSE
)

gene_col <- names(expr_df)[1]
genes <- as.character(expr_df[[gene_col]])

expr <- as.matrix(
  expr_df[, -1, drop=FALSE]
)

storage.mode(expr) <- "numeric"
rownames(expr) <- genes

sample_manifest <- read.csv(
  sample_file,
  stringsAsFactors=FALSE
)

sample_manifest <- sample_manifest[
  match(colnames(expr), sample_manifest$sample_id),
]

if (any(is.na(sample_manifest$sample_id))) {
  stop(
    "Expression columns and sample manifest are not aligned."
  )
}

normals <- as.logical(
  sample_manifest$is_adjacent_normal
)

read_locked_gmt <- function(path) {

  lines <- readLines(
    path,
    warn=FALSE
  )

  out <- lapply(
    lines,
    function(line) {

      parts <- strsplit(
        line,
        "\t",
        fixed=TRUE
      )[[1]]

      if (length(parts) < 3) {
        return(character(0))
      }

      unique(parts[-c(1, 2)])
    }
  )

  names(out) <- vapply(
    lines,
    function(line) {
      strsplit(
        line,
        "\t",
        fixed=TRUE
      )[[1]][1]
    },
    character(1)
  )

  out
}

gene_sets <- read_locked_gmt(
  gmt_file
)

gene_sets <- lapply(
  gene_sets,
  intersect,
  y=rownames(expr)
)

gene_sets <- gene_sets[
  lengths(gene_sets) >= 5
]

if (length(gene_sets) != 30) {
  stop(
    paste(
      "Expected 30 usable locked gene sets, found",
      length(gene_sets)
    )
  )
}

cat(
  "Expression matrix:",
  nrow(expr),
  "genes x",
  ncol(expr),
  "samples\n"
)

cat(
  "Usable locked gene sets:",
  length(gene_sets),
  "\n"
)

gsva_file <- file.path(
  OUTPUT_DIR,
  "GSVA_scores_sample_by_BP.csv"
)

if (!file.exists(gsva_file)) {
  cat("GSVA output is absent; running locked GSVA scoring...\n")
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
    gsva_file,
    row.names=TRUE
  )
  cat("GSVA score matrix written:", gsva_file, "\n")
} else {
  cat("Existing GSVA score matrix retained:", gsva_file, "\n")
}

cat(
  "Running Pathifier with attempts=20...\n"
)

pathifier_log <- file.path(
  OUTPUT_DIR,
  "logs",
  "pathifier_attempts20.log"
)

set.seed(20260725)

pds <- pathifier::quantify_pathways_deregulation(
  data=expr,
  allgenes=rownames(expr),
  syms=gene_sets,
  pathwaynames=names(gene_sets),
  normals=normals,
  attempts=20,
  maximize_stability=TRUE,
  logfile=pathifier_log,
  min_exp=min(expr, na.rm=TRUE),
  min_std=0.4
)

# Save immediately so the expensive computation is never lost again.
raw_rds <- file.path(
  OUTPUT_DIR,
  "Pathifier_full_result_attempts20.rds"
)

saveRDS(
  pds,
  raw_rds
)

cat(
  "Saved full Pathifier object:",
  raw_rds,
  "\n"
)

# Diagnostic structure report.
diag_file <- file.path(
  OUTPUT_DIR,
  "Pathifier_object_structure.txt"
)

sink(diag_file)

cat("class(pds):\n")
print(class(pds))

cat("\nnames(pds):\n")
print(names(pds))

cat("\nlength(pds$scores):\n")
print(length(pds$scores))

cat("\nclass(pds$scores):\n")
print(class(pds$scores))

cat("\nstr(pds$scores, max.level=2):\n")
str(pds$scores, max.level=2)

sink()

# ------------------------------------------------------------
# Correct score extraction
# ------------------------------------------------------------
#
# In pathifier, pds$scores is a list-column.
# Each list element contains one pathway's score vector across samples.
# Convert that list into a pathway x sample numeric matrix.

if (is.null(pds$scores)) {
  stop(
    "Pathifier returned NULL for pds$scores. "
  )
}

if (!is.list(pds$scores)) {
  stop(
    paste(
      "Unexpected pds$scores class:",
      paste(class(pds$scores), collapse=", ")
    )
  )
}

score_rows <- lapply(
  pds$scores,
  function(x) {

    if (is.null(x)) {
      return(NULL)
    }

    values <- as.numeric(x)

    if (length(values) != ncol(expr)) {
      stop(
        paste(
          "A pathway score vector has length",
          length(values),
          "but expected",
          ncol(expr)
        )
      )
    }

    values
  }
)

if (any(vapply(score_rows, is.null, logical(1)))) {
  stop(
    "At least one Pathifier pathway score is NULL."
  )
}

path_scores <- do.call(
  rbind,
  score_rows
)

if (nrow(path_scores) != length(pds$scores)) {
  stop(
    "Pathifier score row count does not match number of processed pathways."
  )
}

if (ncol(path_scores) != ncol(expr)) {
  stop(
    paste(
      "Pathifier score matrix has",
      ncol(path_scores),
      "sample columns; expected",
      ncol(expr)
    )
  )
}

pathway_names <- rownames(pds$scores)

if (is.null(pathway_names)) {
  pathway_names <- names(gene_sets)[pds$sucess]
}

if (length(pathway_names) != nrow(path_scores)) {
  stop(
    "Unable to align pathway names with Pathifier score rows."
  )
}

rownames(path_scores) <- pathway_names
colnames(path_scores) <- colnames(expr)

# Export sample x pathway, matching the GSVA output orientation.
path_out <- t(path_scores)

write.csv(
  path_out,
  file.path(
    OUTPUT_DIR,
    "Pathifier_scores_sample_by_BP.csv"
  ),
  row.names=TRUE
)

cat(
  "Pathifier score matrix written:",
  nrow(path_out),
  "samples x",
  ncol(path_out),
  "pathways\n"
)

manifest <- list(
  status="PASS",
  n_gene_sets_requested=length(gene_sets),
  n_pathways_processed=nrow(path_scores),
  n_genes=nrow(expr),
  n_samples=ncol(expr),
  n_primary_tumors=sum(
    tolower(trimws(as.character(sample_manifest$is_primary_tumor))) %in%
      c("true", "t", "1", "yes", "y"),
    na.rm=TRUE
  ),
  n_adjacent_normals=sum(
    tolower(trimws(as.character(sample_manifest$is_adjacent_normal))) %in%
      c("true", "t", "1", "yes", "y"),
    na.rm=TRUE
  ),
  GSVA_version=as.character(
    packageVersion("GSVA")
  ),
  GSVA_kcdf="Gaussian",
  pathifier_version=as.character(
    packageVersion("pathifier")
  ),
  Pathifier_attempts=20,
  Pathifier_maximize_stability=TRUE,
  Pathifier_min_exp=min(
    expr,
    na.rm=TRUE
  ),
  Pathifier_min_std=0.4,
  GSVA_file_present=file.exists(
    file.path(
      OUTPUT_DIR,
      "GSVA_scores_sample_by_BP.csv"
    )
  ),
  full_Pathifier_object_saved=TRUE,
  full_Pathifier_object_file=basename(raw_rds)
)

write_json(
  manifest,
  file.path(
    OUTPUT_DIR,
    "analysis_manifest.json"
  ),
  pretty=TRUE,
  auto_unbox=TRUE
)

cat(
  "PASS — GSVA/Pathifier scoring completed\n"
)

cat(
  "Output:",
  OUTPUT_DIR,
  "\n"
)
'''

benchmark_input_dir = output_path("MOATTERS_STAGE5B_BENCHMARK_INPUTS")
benchmark_score_dir = output_path("MOATTERS_STAGE5B_GSVA_PATHIFIER_SCORES")
r_code = r_code.replace(
    "__MOATTERS_BENCHMARK_INPUT_DIR__",
    benchmark_input_dir.as_posix(),
).replace(
    "__MOATTERS_BENCHMARK_SCORE_DIR__",
    benchmark_score_dir.as_posix(),
)

r_file = output_path(r"MOATTERS_STAGE5B2C_PATHIFIER_CORRECTED.R")

r_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

r_file.write_text(
    r_code,
    encoding="utf-8"
)

print("Rscript:", rscript)
print("R file:", r_file)
print("GSVA will run only when its locked score file is absent.")
print("Pathifier will run with attempts=20.")
print("The full Pathifier object will be saved before score extraction.")

check = subprocess.run(
    [rscript, "--version"],
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
    timeout=60,
)

if check.returncode != 0:
    raise RuntimeError(
        "Rscript could not be executed."
    )

process = subprocess.Popen(
    [rscript, str(r_file)],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="replace",
    bufsize=1
)

for line in iter(process.stdout.readline, ""):

    if line == "" and process.poll() is not None:
        break

    sys.stdout.write(line); sys.stdout.flush()

return_code = process.wait()

if return_code != 0:
    raise RuntimeError(
        f"Corrected Pathifier run failed with exit code {return_code}"
    )

output_dir = output_path(r"MOATTERS_STAGE5B_GSVA_PATHIFIER_SCORES")

expected = [
    output_dir / "GSVA_scores_sample_by_BP.csv",
    output_dir / "Pathifier_scores_sample_by_BP.csv",
    output_dir / "analysis_manifest.json",
    output_dir / "Pathifier_full_result_attempts20.rds",
    output_dir / "Pathifier_object_structure.txt",
]

print("\nFinal output check:")

for p in expected:
    print(
        f"{p.name}: {'FOUND' if p.exists() else 'MISSING'}"
    )

if all(p.exists() for p in expected):

    print(
        "PASS — Stage 5B-2 is complete. Proceed to benchmark evaluation."
    )

else:

    raise RuntimeError(
        "The R process ended, but one or more required outputs are missing."
    )
