# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from moatters.config import data_path, output_path

import numpy as np
import pandas as pd

# ============================================================
# MOATTERS — METABRIC Stage 1 (v1.1 NA-safe)
# Purpose: lock and harmonize expression/clinical inputs before
#          external reconstruction. No model fitting is performed.
# ============================================================

INPUT_DIR = data_path(r"External\brca_metabric")
OUTPUT_DIR = output_path(r"MOATTERS_METABRIC_STAGE1")

RAW_EXPR_FILE = INPUT_DIR / "data_mrna_illumina_microarray.txt"
ZSCORE_EXPR_FILE = INPUT_DIR / "data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt"
CLINICAL_FILE = INPUT_DIR / "brca_metabric_clinical_data.tsv"

# Primary external-validation matrix is the normalized continuous expression
# matrix. The diploid-reference z-score matrix is retained only as a secondary
# representation audit and is not the primary locked input.
PRIMARY_REPRESENTATION = "normalized_continuous_expression"

NA_VALUES = ["", "NA", "N/A", "NaN", "nan", "null", "NULL", "Not Available", "[Not Available]"]


def log(msg: str, fh=None) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def clean_gene_symbol(x) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.split("|")[0].strip().upper()
    return s or None


def clean_metabric_id(x) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    m = re.search(r"MB-\d+", s)
    return m.group(0) if m else (s or None)


def read_expression(path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t", low_memory=False, na_values=NA_VALUES)
    if df.shape[1] < 3:
        raise ValueError(f"Expression file has too few columns: {path}")

    gene_col = "Hugo_Symbol" if "Hugo_Symbol" in df.columns else df.columns[0]
    genes = df[gene_col].map(clean_gene_symbol)
    sample_cols = [c for c in df.columns if clean_metabric_id(c) is not None and str(c).upper().startswith("MB-")]
    if not sample_cols:
        sample_cols = list(df.columns[2:])

    mat = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    mat.columns = [clean_metabric_id(c) for c in sample_cols]
    mat.index = genes
    mat = mat.loc[mat.index.notna(), mat.columns.notna()]

    n_rows_before = int(mat.shape[0])
    n_duplicate_gene_rows = int(mat.index.duplicated(keep=False).sum())
    n_duplicate_sample_cols = int(pd.Index(mat.columns).duplicated(keep=False).sum())

    # Collapse duplicate samples, then duplicate gene symbols by mean.
    mat = mat.T.groupby(level=0, sort=False).mean().T
    mat = mat.groupby(level=0, sort=False).mean()

    all_missing_gene = mat.isna().all(axis=1)
    all_missing_sample = mat.isna().all(axis=0)
    mat = mat.loc[~all_missing_gene, ~all_missing_sample]

    diag = {
        "path": str(path),
        "sha256": sha256(path),
        "gene_column": str(gene_col),
        "n_rows_before_collapse": n_rows_before,
        "n_duplicate_gene_rows": n_duplicate_gene_rows,
        "n_duplicate_sample_columns": n_duplicate_sample_cols,
        "n_genes_after_collapse": int(mat.shape[0]),
        "n_samples_after_collapse": int(mat.shape[1]),
        "overall_missing_fraction": float(mat.isna().to_numpy().mean()),
        "n_zero_variance_genes": int((mat.var(axis=1, skipna=True) == 0).sum()),
        "median_gene_variance": float(mat.var(axis=1, skipna=True).median()),
        "median_sample_missing_fraction": float(mat.isna().mean(axis=0).median()),
    }
    return mat, diag


def normalize_binary(series: pd.Series, positive: set[str], negative: set[str]) -> pd.Series:
    def conv(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().upper()
        if s in positive:
            return 1.0
        if s in negative:
            return 0.0
        return np.nan
    return series.map(conv)


def stage_group(x):
    if pd.isna(x):
        return np.nan
    try:
        v = float(x)
        if v in (1, 2):
            return "Early"
        if v in (3, 4):
            return "Late"
    except Exception:
        pass
    s = str(x).strip().upper()
    if re.search(r"\b(III|IV)\b", s):
        return "Late"
    if re.search(r"\b(I|II)\b", s):
        return "Early"
    return np.nan


def load_clinical(path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    clin = pd.read_csv(path, sep="\t", low_memory=False, na_values=NA_VALUES)
    required = ["Patient ID", "Sample ID"]
    missing = [c for c in required if c not in clin.columns]
    if missing:
        raise ValueError(f"Missing required clinical columns: {missing}")

    out = pd.DataFrame(index=clin.index)
    out["patient_id"] = clin["Patient ID"].map(clean_metabric_id)
    out["sample_id"] = clin["Sample ID"].map(clean_metabric_id)

    def copy_if(src, dst=None):
        if src in clin.columns:
            out[dst or src] = clin[src]

    for c in [
        "Age at Diagnosis", "Pam50 + Claudin-low subtype", "ER Status", "PR Status",
        "HER2 Status", "Neoplasm Histologic Grade", "Tumor Stage", "Tumor Size",
        "Overall Survival (Months)", "Overall Survival Status",
        "Relapse Free Status (Months)", "Relapse Free Status",
        "Chemotherapy", "Hormone Therapy", "Radio Therapy", "Cohort",
    ]:
        copy_if(c)

    if "ER Status" in out:
        out["ER_binary"] = normalize_binary(out["ER Status"], {"POSITIVE", "POSITIVE", "POS"}, {"NEGATIVE", "NEG"})
    if "PR Status" in out:
        out["PR_binary"] = normalize_binary(out["PR Status"], {"POSITIVE", "POS"}, {"NEGATIVE", "NEG"})
    if "HER2 Status" in out:
        out["HER2_binary"] = normalize_binary(out["HER2 Status"], {"POSITIVE", "POS"}, {"NEGATIVE", "NEG"})
    if "Tumor Stage" in out:
        out["stage_group"] = out["Tumor Stage"].map(stage_group)
        out["stage_late_binary"] = out["stage_group"].map({"Early": 0.0, "Late": 1.0})
    if "Pam50 + Claudin-low subtype" in out:
        p = out["Pam50 + Claudin-low subtype"].astype("string").str.strip()
        out["PAM50_clean"] = p
        p_upper = p.str.upper()
        # Avoid evaluating pd.NA in Python boolean expressions.  Vectorized
        # masks preserve missing values and are stable across pandas versions.
        out["PAM50_luminal_binary"] = pd.Series(
            np.where(p_upper.isna(), np.nan, p_upper.isin(["LUMA", "LUMB"]).astype(float)),
            index=out.index,
            dtype="Float64",
        )
        out["PAM50_basal_binary"] = pd.Series(
            np.where(p_upper.isna(), np.nan, p_upper.eq("BASAL").fillna(False).astype(float)),
            index=out.index,
            dtype="Float64",
        )
    if "Overall Survival Status" in out:
        os_status = out["Overall Survival Status"].astype("string")
        out["OS_event"] = os_status.str.startswith("1:", na=pd.NA).astype("Float64")
    if "Relapse Free Status" in out:
        rfs_status = out["Relapse Free Status"].astype("string")
        out["RFS_event"] = rfs_status.str.startswith("1:", na=pd.NA).astype("Float64")

    out = out.dropna(subset=["sample_id"]).copy()
    duplicate_sample_rows = int(out["sample_id"].duplicated(keep=False).sum())
    out = out.drop_duplicates(subset=["sample_id"], keep="first").set_index("sample_id", drop=False)

    diag = {
        "path": str(path),
        "sha256": sha256(path),
        "n_rows_raw": int(clin.shape[0]),
        "n_columns_raw": int(clin.shape[1]),
        "n_unique_samples": int(out.index.nunique()),
        "n_unique_patients": int(out["patient_id"].nunique()),
        "n_duplicate_sample_rows": duplicate_sample_rows,
    }
    return out, diag


def endpoint_inventory(clin: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        "ER_binary", "PR_binary", "HER2_binary", "stage_late_binary",
        "PAM50_luminal_binary", "PAM50_basal_binary", "OS_event", "RFS_event",
        "Neoplasm Histologic Grade", "Tumor Stage", "PAM50_clean",
    ]
    rows = []
    for c in candidates:
        if c not in clin.columns:
            continue
        s = clin[c]
        counts = s.value_counts(dropna=False).to_dict()
        rows.append({
            "endpoint": c,
            "n_nonmissing": int(s.notna().sum()),
            "n_missing": int(s.isna().sum()),
            "n_unique_nonmissing": int(s.nunique(dropna=True)),
            "value_counts": json.dumps({str(k): int(v) for k, v in counts.items()}, ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def representation_comparison(raw: pd.DataFrame, z: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    genes = raw.index.intersection(z.index)
    samples = raw.columns.intersection(z.columns)
    r = raw.loc[genes, samples]
    zz = z.loc[genes, samples]

    rows = []
    # Gene-wise correlation across samples; sample up to 3000 genes deterministically.
    test_genes = genes[: min(3000, len(genes))]
    for g in test_genes:
        a = r.loc[g]
        b = zz.loc[g]
        ok = a.notna() & b.notna()
        corr = a[ok].corr(b[ok], method="spearman") if ok.sum() >= 10 else np.nan
        rows.append({"gene": g, "n_common": int(ok.sum()), "spearman_raw_vs_diploid_z": corr})
    comp = pd.DataFrame(rows)
    summary = {
        "n_common_genes": int(len(genes)),
        "n_common_samples": int(len(samples)),
        "n_genes_tested": int(len(test_genes)),
        "median_gene_spearman": float(comp["spearman_raw_vs_diploid_z"].median()),
        "q05_gene_spearman": float(comp["spearman_raw_vs_diploid_z"].quantile(0.05)),
        "q95_gene_spearman": float(comp["spearman_raw_vs_diploid_z"].quantile(0.95)),
    }
    return comp, summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table_dir = OUTPUT_DIR / "tables"
    matrix_dir = OUTPUT_DIR / "matrices"
    log_dir = OUTPUT_DIR / "logs"
    for d in [table_dir, matrix_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with (log_dir / "stage1_run.log").open("w", encoding="utf-8") as fh:
        log("Starting METABRIC Stage 1 input lock and harmonization", fh)
        log(f"Input:  {INPUT_DIR}", fh)
        log(f"Output: {OUTPUT_DIR}", fh)

        raw, raw_diag = read_expression(RAW_EXPR_FILE)
        log(f"Normalized expression loaded: genes={raw.shape[0]}, samples={raw.shape[1]}", fh)

        zmat, z_diag = read_expression(ZSCORE_EXPR_FILE)
        log(f"Diploid-reference z-score loaded: genes={zmat.shape[0]}, samples={zmat.shape[1]}", fh)

        clin, clin_diag = load_clinical(CLINICAL_FILE)
        log(f"Clinical loaded: samples={clin.shape[0]}, patients={clin['patient_id'].nunique()}", fh)

        common_primary = raw.columns.intersection(clin.index)
        unmatched_expr = raw.columns.difference(clin.index)
        unmatched_clin = clin.index.difference(raw.columns)
        log(f"Primary expression-clinical overlap: {len(common_primary)}", fh)

        primary_expr = raw.loc[:, common_primary].copy()
        primary_clin = clin.loc[common_primary].copy()

        # Write canonical matrices. Expression remains genes x samples.
        primary_expr.to_csv(matrix_dir / "METABRIC_primary_expression_genes_x_samples.tsv.gz", sep="\t", compression="gzip")
        primary_clin.to_csv(table_dir / "METABRIC_primary_clinical_harmonized.tsv", sep="\t", index=False)

        pd.DataFrame({"sample_id": unmatched_expr}).to_csv(table_dir / "METABRIC_expression_without_clinical.csv", index=False)
        pd.DataFrame({"sample_id": unmatched_clin}).to_csv(table_dir / "METABRIC_clinical_without_expression.csv", index=False)

        endpoint_df = endpoint_inventory(primary_clin)
        endpoint_df.to_csv(table_dir / "METABRIC_endpoint_inventory_locked.csv", index=False, encoding="utf-8-sig")

        gene_qc = pd.DataFrame({
            "gene": primary_expr.index,
            "n_nonmissing": primary_expr.notna().sum(axis=1).to_numpy(),
            "missing_fraction": primary_expr.isna().mean(axis=1).to_numpy(),
            "mean": primary_expr.mean(axis=1, skipna=True).to_numpy(),
            "sd": primary_expr.std(axis=1, skipna=True).to_numpy(),
            "variance": primary_expr.var(axis=1, skipna=True).to_numpy(),
        })
        gene_qc.to_csv(table_dir / "METABRIC_primary_gene_QC.csv.gz", index=False, compression="gzip")

        sample_qc = pd.DataFrame({
            "sample_id": primary_expr.columns,
            "n_nonmissing_genes": primary_expr.notna().sum(axis=0).to_numpy(),
            "missing_fraction": primary_expr.isna().mean(axis=0).to_numpy(),
            "mean_expression": primary_expr.mean(axis=0, skipna=True).to_numpy(),
            "sd_expression": primary_expr.std(axis=0, skipna=True).to_numpy(),
        })
        sample_qc.to_csv(table_dir / "METABRIC_primary_sample_QC.csv", index=False, encoding="utf-8-sig")

        comp, comp_summary = representation_comparison(raw, zmat)
        comp.to_csv(table_dir / "METABRIC_raw_vs_diploid_z_gene_correlations.csv.gz", index=False, compression="gzip")

        manifest = pd.DataFrame([
            {"role": "PRIMARY_EXPRESSION", "path": str(RAW_EXPR_FILE), "sha256": raw_diag["sha256"], "decision": "LOCKED", "rationale": "Normalized continuous expression preserves cohort-wide relative variation and is the primary basis for BP scoring and external reconstruction."},
            {"role": "SECONDARY_REPRESENTATION_AUDIT", "path": str(ZSCORE_EXPR_FILE), "sha256": z_diag["sha256"], "decision": "AUDIT_ONLY", "rationale": "Diploid-reference z-scores are biologically transformed and may alter correlation/network interpretation; retained for sensitivity audit only."},
            {"role": "PRIMARY_CLINICAL", "path": str(CLINICAL_FILE), "sha256": clin_diag["sha256"], "decision": "LOCKED", "rationale": "Consolidated sample-level clinical table contains the required external endpoints and direct MB sample identifiers."},
        ])
        manifest.to_csv(table_dir / "METABRIC_STAGE1_INPUT_LOCK_MANIFEST.csv", index=False, encoding="utf-8-sig")

        summary = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": "METABRIC_STAGE1_INPUT_LOCK",
            "status": "PASS" if len(common_primary) >= 1000 else "REVIEW",
            "primary_representation": PRIMARY_REPRESENTATION,
            "raw_expression": raw_diag,
            "diploid_zscore_expression": z_diag,
            "clinical": clin_diag,
            "primary_overlap": {
                "n_expression_samples": int(raw.shape[1]),
                "n_clinical_samples": int(clin.shape[0]),
                "n_matched_samples": int(len(common_primary)),
                "match_rate_expression": float(len(common_primary) / raw.shape[1]),
                "match_rate_clinical": float(len(common_primary) / clin.shape[0]),
                "n_expression_without_clinical": int(len(unmatched_expr)),
                "n_clinical_without_expression": int(len(unmatched_clin)),
            },
            "locked_primary_matrix": {
                "n_genes": int(primary_expr.shape[0]),
                "n_samples": int(primary_expr.shape[1]),
                "overall_missing_fraction": float(primary_expr.isna().to_numpy().mean()),
                "n_zero_variance_genes": int((primary_expr.var(axis=1, skipna=True) == 0).sum()),
            },
            "representation_comparison": comp_summary,
            "next_step": "Stage 2: load locked TCGA-BRCA task-discriminative BP definitions and reconstruction rules, quantify METABRIC BP coverage, and produce external patient-level states without retraining the TCGA definitions.",
        }
        with (OUTPUT_DIR / "METABRIC_STAGE1_SUMMARY.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        readme = f"""MOATTERS — METABRIC Stage 1\n\nStatus: {summary['status']}\nPrimary expression: {RAW_EXPR_FILE.name}\nSecondary audit representation: {ZSCORE_EXPR_FILE.name}\nClinical: {CLINICAL_FILE.name}\nMatched primary samples: {len(common_primary)}\nLocked matrix: {primary_expr.shape[0]} genes x {primary_expr.shape[1]} samples\n\nDecision:\nThe normalized continuous expression matrix is locked as the primary external-validation input.\nThe diploid-reference z-score matrix is retained only for representation sensitivity analysis.\nNo outcome model or BP state has been fitted in Stage 1.\n\nNext:\nUse TCGA-BRCA-locked BP definitions and patient-state reconstruction rules in METABRIC Stage 2.\n"""
        (OUTPUT_DIR / "README_STAGE1.txt").write_text(readme, encoding="utf-8")

        log("Stage 1 completed", fh)
        log(f"Status: {summary['status']}", fh)
        log(f"Locked samples: {len(common_primary)}", fh)


if __name__ == "__main__":
    main()
