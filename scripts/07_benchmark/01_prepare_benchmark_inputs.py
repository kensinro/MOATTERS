# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Stage 5B-1 — Prepare locked inputs for the GSVA / Pathifier benchmark.

Locked benchmark contract
-------------------------
- 30 task-discriminative BRCA GO-BP terms from the original analysis ZIP.
- The same TCGA-BRCA primary-tumor cohort used in the manuscript.
- TCGA adjacent-normal samples are included only as the Pathifier normal
  reference; evaluation remains restricted to primary tumors.
- Clinical endpoints and the MOATTERS M1–M7 representation are read from the
  original locked ZIP, not reconstructed ad hoc.
- The master GO-BP GMT is used only to recover gene membership for the locked
  30 BP names.

Outputs are written to:
D:\MOATTERS-Output\MOATTERS_STAGE5B_BENCHMARK_INPUTS
"""

from __future__ import annotations
from pathlib import Path
from moatters.config import data_path, output_path
import csv
import gzip
import io
import json
import re
import zipfile

import numpy as np
import pandas as pd

ZIP_SEARCH_ROOTS = [output_path(), output_path()]
BRCA_DATA_ROOT = data_path(r"UCSC_XENA\Breast Cancer (BRCA)")
GO_GMT = data_path(r"GSEA\c5.go.bp.v2026.1.Hs.symbols.gmt")
OUT = output_path(r"MOATTERS_STAGE5B_BENCHMARK_INPUTS")

MEMBERSHIP_SUFFIX = (
    "MOATTERS_BRCA_STATE_ManuscriptDefense_20260531_184850/"
    "tables/02_BRCA_real_BP_module_membership.csv"
)
MOATTERS_SUFFIX = (
    "MOATTERS_BRCA_STATE_ManuscriptDefense_20260531_184850/"
    "tables/10_leave_one_module_patient_scores.csv"
)
CLINICAL_SUFFIX = (
    "MOATTERS_BRCA_STATE_DownstreamValidation_20260531_124416/"
    "tables/99_final_merged_BPstate_clinical_endpoints.csv"
)

def load_current_locked_inputs():
    module_path = (
        output_path("MOATTERS_STAGE_BP_Module_Rewiring_V1_20260531")
        / "BRCA" / "BRCA_module_assignment.csv"
    )
    score_path = (
        output_path("MOATTERS_BRCA_PATIENT_STATE_Profile_V1_20260531")
        / "BRCA_patient_module_scores_z.csv"
    )
    master_path = (
        output_path("MOATTERS_BRCA_PATIENT_STATE_Profile_V1_20260531")
        / "BRCA_patient_strategy_master_table.csv"
    )
    downstream_dirs = sorted(
        output_path().glob("MOATTERS_BRCA_STATE_DownstreamValidation_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not downstream_dirs:
        raise FileNotFoundError("No current downstream-validation output directory was found.")
    clinical_path = downstream_dirs[0] / "tables" / "99_final_merged_BPstate_clinical_endpoints.csv"
    required = [module_path, score_path, master_path, clinical_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing current benchmark inputs: {missing}")

    assignment = pd.read_csv(module_path)
    membership = (
        assignment.loc[assignment["group"].astype(str).eq("stage_III_IV"), ["term", "module_id"]]
        .rename(columns={"term": "BP_term"})
        .drop_duplicates("BP_term")
    )
    module_scores = pd.read_csv(score_path, index_col=0)
    module_scores.index = module_scores.index.astype(str)
    state = module_scores.T.reset_index().rename(columns={"index": "patient_id"})
    rename_map = {
        col: (str(col) if str(col).startswith("M") else f"M{col}")
        for col in state.columns if col != "patient_id"
    }
    state = state.rename(columns=rename_map)
    expected_modules = [f"M{i}" for i in range(1, 8)]
    missing_modules = [m for m in expected_modules if m not in state.columns]
    if missing_modules:
        raise RuntimeError(f"Current module-score table is missing modules: {missing_modules}")

    master = pd.read_csv(master_path)
    absent = sorted({"patient", "stage_group"}.difference(master.columns))
    if absent:
        raise RuntimeError(f"Patient strategy master table is missing: {absent}")
    stage = (
        master[["patient", "stage_group"]]
        .drop_duplicates("patient")
        .rename(columns={"patient": "patient_id", "stage_group": "StageGroup"})
    )
    state = state.merge(stage, on="patient_id", how="left")
    clinical = pd.read_csv(clinical_path)
    source = {"membership": str(module_path), "state": str(score_path), "clinical": str(clinical_path)}
    return membership, state, clinical, source

def read_gmt(path):
    gene_sets = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) >= 3:
                gene_sets[parts[0]] = parts[2:]
    return gene_sets

def looks_like_tcga_sample(x):
    return bool(re.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}", str(x)))

def inspect_expression_candidate(path):
    try:
        if path.suffix.lower() == ".gz":
            df = pd.read_csv(path, sep="\t", nrows=3, compression="gzip")
        else:
            sep = "\t" if path.suffix.lower() in {".txt", ".tsv"} else ","
            df = pd.read_csv(path, sep=sep, nrows=3)
    except Exception:
        return None

    cols = [str(c) for c in df.columns]
    header_tcga = sum(looks_like_tcga_sample(c) for c in cols)
    first_col_tcga = 0
    if len(df.columns):
        first_col_tcga = sum(looks_like_tcga_sample(v) for v in df.iloc[:, 0].astype(str))
    return {
        "path": str(path),
        "n_header_cols": len(cols),
        "header_tcga": header_tcga,
        "first_col_tcga": first_col_tcga,
        "first_column": cols[0] if cols else "",
    }

def discover_expression():
    extensions = {".csv", ".txt", ".tsv", ".gz"}
    inventory = []
    for p in BRCA_DATA_ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in extensions:
            continue
        if p.stat().st_size < 1_000_000:
            continue
        info = inspect_expression_candidate(p)
        if info:
            info["size_bytes"] = p.stat().st_size
            score = (
                5 * info["header_tcga"]
                + 2 * info["n_header_cols"]
                + (1000 if "expression" in str(p).lower() else 0)
                + (1000 if "hiseq" in str(p).lower() else 0)
            )
            info["score"] = score
            inventory.append(info)

    if not inventory:
        raise RuntimeError("No readable expression candidates were found.")

    inv = pd.DataFrame(inventory).sort_values("score", ascending=False)
    inv.to_csv(OUT / "expression_candidate_inventory.csv",
               index=False, encoding="utf-8-sig")

    best = inv.iloc[0]
    if int(best["header_tcga"]) < 100:
        raise RuntimeError(
            "No gene-by-sample expression matrix with TCGA sample columns "
            "was confidently identified. Inspect expression_candidate_inventory.csv."
        )
    return Path(best["path"])

def read_expression(path):
    if path.suffix.lower() == ".gz":
        return pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    sep = "\t" if path.suffix.lower() in {".txt", ".tsv"} else ","
    return pd.read_csv(path, sep=sep, low_memory=False)

def tcga_sample_type(sample):
    s = str(sample)
    return s[13:15] if len(s) >= 15 else ""

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    membership, aido, clinical, source_paths = load_current_locked_inputs()
    zp = "CURRENT_PORTABLE_OUTPUTS"
    membership_member = source_paths["membership"]
    aido_member = source_paths["state"]
    clinical_member = source_paths["clinical"]

    locked_bp = membership["BP_term"].astype(str).drop_duplicates().tolist()
    if len(locked_bp) != 30:
        raise RuntimeError(f"Expected 30 locked BP terms, found {len(locked_bp)}.")

    if not GO_GMT.exists():
        raise FileNotFoundError(f"GO-BP GMT not found: {GO_GMT}")
    all_sets = read_gmt(GO_GMT)
    missing_bp = [bp for bp in locked_bp if bp not in all_sets]
    if missing_bp:
        raise RuntimeError(
            "The master GMT did not contain all locked BP names: "
            + "; ".join(missing_bp)
        )

    locked_sets = {bp: all_sets[bp] for bp in locked_bp}
    unique_genes = sorted(set(g for genes in locked_sets.values() for g in genes))

    locked_gmt = OUT / "locked_30_BRCA_BP_terms.gmt"
    with locked_gmt.open("w", encoding="utf-8", newline="") as fh:
        for bp, genes in locked_sets.items():
            fh.write("\t".join([bp, "locked_from_original_BRCA_reconstruction", *genes]) + "\n")

    pd.DataFrame({
        "BP_term": list(locked_sets),
        "module_id": [
            membership.set_index("BP_term").loc[bp, "module_id"] for bp in locked_sets
        ],
        "n_genes_in_master_GMT": [len(locked_sets[bp]) for bp in locked_sets],
    }).to_csv(OUT / "locked_30_BP_manifest.csv", index=False, encoding="utf-8-sig")

    expression_path = discover_expression()
    raw = read_expression(expression_path)

    gene_col = raw.columns[0]
    raw[gene_col] = raw[gene_col].astype(str)
    sample_cols = [c for c in raw.columns[1:] if looks_like_tcga_sample(c)]
    primary_cols = [c for c in sample_cols if tcga_sample_type(c) == "01"]
    normal_cols = [c for c in sample_cols if tcga_sample_type(c) == "11"]

    if len(primary_cols) < 500:
        raise RuntimeError(f"Only {len(primary_cols)} primary tumors were found.")
    if len(normal_cols) < 10:
        raise RuntimeError(
            f"Only {len(normal_cols)} adjacent normals were found; "
            "Pathifier requires a normal reference."
        )

    subset = raw.loc[raw[gene_col].isin(unique_genes), [gene_col, *primary_cols, *normal_cols]].copy()
    numeric_cols = subset.columns[1:]
    numeric_block = subset.loc[:, numeric_cols].apply(pd.to_numeric, errors="coerce")
    subset = pd.concat([subset[[gene_col]].reset_index(drop=True), numeric_block.reset_index(drop=True)], axis=1)

    # Collapse duplicate symbols by mean.
    subset = subset.groupby(gene_col, as_index=False).mean(numeric_only=True)
    subset.to_csv(
        OUT / "locked_expression_primary_plus_normal.csv.gz",
        index=False, compression="gzip"
    )

    sample_manifest = pd.DataFrame({
        "sample_id": [*primary_cols, *normal_cols],
        "patient_id": [c[:12] for c in [*primary_cols, *normal_cols]],
        "sample_type_code": [
            tcga_sample_type(c) for c in [*primary_cols, *normal_cols]
        ],
        "is_primary_tumor": [True] * len(primary_cols) + [False] * len(normal_cols),
        "is_adjacent_normal": [False] * len(primary_cols) + [True] * len(normal_cols),
    })
    sample_manifest.to_csv(OUT / "sample_manifest.csv", index=False, encoding="utf-8-sig")

    # Locked MOATTERS patient representation.
    keep_aido = ["patient_id", "StageGroup", *[f"M{i}" for i in range(1, 8)]]
    aido[keep_aido].to_csv(
        OUT / "locked_MOATTERS_M1_M7_patient_representation.csv",
        index=False, encoding="utf-8-sig"
    )

    # Compact benchmark endpoint table.
    endpoint_cols = [
        "patient_id", "StageGroup", "PAM50_simplified",
        "ER_status__breast_carcinoma_estrogen_receptor_status__clean",
        "PR_status__breast_carcinoma_progesterone_receptor_status__clean",
        "HER2_status__HER2_Final_Status_nature2012__clean",
    ]
    missing_endpoint_cols = [c for c in endpoint_cols if c not in clinical.columns]
    if missing_endpoint_cols:
        raise RuntimeError(f"Clinical table missing columns: {missing_endpoint_cols}")

    clinical[endpoint_cols].drop_duplicates("patient_id").to_csv(
        OUT / "locked_benchmark_endpoints.csv",
        index=False, encoding="utf-8-sig"
    )

    manifest = {
        "status": "PASS",
        "locked_zip": str(zp),
        "membership_member": membership_member,
        "aido_member": aido_member,
        "clinical_member": clinical_member,
        "expression_path": str(expression_path),
        "GO_GMT": str(GO_GMT),
        "n_locked_BP_terms": len(locked_sets),
        "n_unique_locked_genes": len(unique_genes),
        "n_expression_genes_retained": int(len(subset)),
        "n_primary_tumor_samples": len(primary_cols),
        "n_adjacent_normal_samples": len(normal_cols),
        "interpretation_boundary": (
            "The benchmark compares locked representation/scoring behavior. "
            "It is not a fully nested feature-selection comparison."
        ),
    }
    (OUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("PASS — Stage 5B-1 benchmark inputs prepared")
    print(f"Expression: {expression_path}")
    print(f"Locked BP terms: {len(locked_sets)}")
    print(f"Primary tumors: {len(primary_cols)}")
    print(f"Adjacent normals: {len(normal_cols)}")
    print(f"Output: {OUT}")

if __name__ == "__main__":
    main()
