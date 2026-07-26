# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# -*- coding: utf-8 -*-
r"""
MOATTERS
GSE96058 Stage 2A/2B — cohort-specific transfer lock and GO-BP coverage audit

Purpose
-------
Create a GSE96058-specific transfer manifest using the same locked TCGA-BRCA
artifacts used for METABRIC, then audit GO-BP matched-gene coverage at
K = 5, 10, 15 and 20.

No endpoint is used. No model is fitted. No patient-level reconstruction is
performed in this stage.

Input
-----
D:\MOATTERS-Output\MOATTERS_GSE96058_STAGE1
D:\MOATTERS-Output\MOATTERS_METABRIC_STAGE2B_LOCK_COVERAGE_V1_1
D:\MOATTERS-Data\GSEA\c5.go.bp.v2026.1.Hs.symbols.gmt

Output
------
D:\MOATTERS-Output\MOATTERS_GSE96058_STAGE2AB_TRANSFER_COVERAGE
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from moatters.config import data_path, output_path

import numpy as np
import pandas as pd


STAGE1 = output_path(r"MOATTERS_GSE96058_STAGE1")
METABRIC_LOCK = output_path(r"MOATTERS_METABRIC_STAGE2B_LOCK_COVERAGE_V1_1")
LOCKED_SRC = METABRIC_LOCK / "locked_artifacts"
GMT = data_path(r"GSEA\c5.go.bp.v2026.1.Hs.symbols.gmt")
OUT = output_path(r"MOATTERS_GSE96058_STAGE2AB_TRANSFER_COVERAGE")

EXPR_CANDIDATES = [
    STAGE1 / "matrices" / "GSE96058_primary_expression_genes_x_samples.tsv.gz",
    STAGE1 / "matrices" / "GSE96058_primary_expression_genes_x_samples.tsv",
]

ARTIFACT_FILES = [
    "BRCA_DStage_BP_results.csv",
    "BRCA_selected_BP_for_network.csv",
    "BRCA_BP_correlation_network_edges.csv",
    "BRCA_BP_correlation_network_node_metrics.csv",
    "BRCA_module_assignment.csv",
    "BRCA_profile_module_composition.csv",
    "BRCA_module_late_alignment_direction.csv",
    "BRCA_early_late_module_centroids.csv",
    "BRCA_patient_module_scores_z.csv",
    "BRCA_patient_strategy_master_table.csv",
]

K_VALUES = [5, 10, 15, 20]
MODULE_GROUP = "stage_III_IV"


def log(msg, fh):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def sha256_file(path, chunk_size=4 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def clean_gene(x):
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def normalize_term(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    s = re.sub(r"^GOBP[_:\- ]*", "", s)
    s = re.sub(r"^GO[_:\- ]*BIOLOGICAL[_ ]PROCESS[_:\- ]*", "", s)
    s = re.sub(r"[^A-Z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def load_gmt(path):
    rows = []
    lookup = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:
                continue
            term = parts[0].strip()
            genes = sorted({clean_gene(g) for g in parts[2:] if clean_gene(g)})
            norm = normalize_term(term)
            row = {
                "gmt_line": line_no,
                "term": term,
                "term_norm": norm,
                "n_genes_gmt": len(genes),
                "genes": genes,
            }
            rows.append(row)
            lookup[norm] = row
    return pd.DataFrame(rows), lookup


def load_expression_gene_index(path):
    df = pd.read_csv(path, sep="\t", usecols=[0], compression="infer")
    return {clean_gene(x) for x in df.iloc[:, 0] if clean_gene(x)}


def normalize_module_id(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    try:
        v = float(s)
        if v.is_integer():
            return str(int(v))
    except Exception:
        pass
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ["locked_artifacts", "tables", "logs", "audit"]:
        (OUT / sub).mkdir(exist_ok=True)

    with (OUT / "logs" / "stage2ab.log").open("w", encoding="utf-8") as fh:
        log("Starting GSE96058 Stage 2A/2B transfer and coverage audit", fh)

        expr_path = next((p for p in EXPR_CANDIDATES if p.exists()), None)
        if expr_path is None:
            raise FileNotFoundError("GSE96058 Stage 1 expression matrix not found.")
        if not GMT.exists():
            raise FileNotFoundError(GMT)

        missing = [name for name in ARTIFACT_FILES if not (LOCKED_SRC / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing locked TCGA-BRCA artifacts: {missing}")

        manifest_rows = []
        for name in ARTIFACT_FILES:
            src = LOCKED_SRC / name
            dst = OUT / "locked_artifacts" / name
            shutil.copy2(src, dst)
            manifest_rows.append({
                "filename": name,
                "source_path": str(src),
                "locked_copy": str(dst),
                "sha256": sha256_file(src),
                "size_bytes": src.stat().st_size,
            })
        manifest = pd.DataFrame(manifest_rows)
        manifest.to_csv(
            OUT / "tables" / "GSE96058_TCGA_BRCA_TRANSFER_MANIFEST.csv",
            index=False, encoding="utf-8-sig"
        )
        log(f"Locked artifacts copied: {len(manifest)}", fh)

        genes = load_expression_gene_index(expr_path)
        gmt_df, gmt_lookup = load_gmt(GMT)
        log(f"GSE96058 unique genes: {len(genes)}", fh)
        log(f"GO-BP GMT terms: {len(gmt_df)}", fh)

        selected = pd.read_csv(
            OUT / "locked_artifacts" / "BRCA_selected_BP_for_network.csv",
            low_memory=False
        )
        assignment = pd.read_csv(
            OUT / "locked_artifacts" / "BRCA_module_assignment.csv",
            low_memory=False
        )
        composition = pd.read_csv(
            OUT / "locked_artifacts" / "BRCA_profile_module_composition.csv",
            low_memory=False
        )

        if "term" not in selected.columns:
            raise RuntimeError("Selected-BP artifact has no 'term' column.")
        if not {"group", "module_id", "term"}.issubset(assignment.columns):
            raise RuntimeError("Module-assignment schema is incomplete.")

        profile = assignment[
            assignment["group"].astype(str) == MODULE_GROUP
        ].copy()
        profile["module_id_norm"] = profile["module_id"].map(normalize_module_id)

        if selected["term"].nunique() != 30:
            raise RuntimeError(
                f"Expected 30 selected BP terms, found {selected['term'].nunique()}."
            )
        if profile["term"].nunique() != 30:
            raise RuntimeError(
                f"Expected 30 BP terms in {MODULE_GROUP}, found {profile['term'].nunique()}."
            )
        if profile["module_id_norm"].nunique() != 7:
            raise RuntimeError(
                f"Expected 7 modules, found {profile['module_id_norm'].nunique()}."
            )

        coverage_rows = []
        for term in selected["term"].dropna().astype(str).drop_duplicates():
            norm = normalize_term(term)
            gmt_row = gmt_lookup.get(norm)
            if gmt_row is None:
                coverage_rows.append({
                    "term": term,
                    "gmt_match": False,
                    "n_genes_gmt": np.nan,
                    "n_genes_matched_GSE96058": np.nan,
                    "coverage_fraction": np.nan,
                    **{f"eligible_K{k}": False for k in K_VALUES},
                })
                continue

            term_genes = set(gmt_row["genes"])
            n_total = len(term_genes)
            n_match = len(term_genes & genes)
            coverage_rows.append({
                "term": term,
                "gmt_match": True,
                "n_genes_gmt": n_total,
                "n_genes_matched_GSE96058": n_match,
                "coverage_fraction": n_match / n_total if n_total else np.nan,
                **{f"eligible_K{k}": n_match >= k for k in K_VALUES},
            })

        coverage = pd.DataFrame(coverage_rows)
        coverage.to_csv(
            OUT / "tables" / "GSE96058_selected_30_BP_coverage.csv",
            index=False, encoding="utf-8-sig"
        )

        if int(coverage["gmt_match"].sum()) != 30:
            raise RuntimeError("Not all 30 selected BP terms matched the locked GMT.")

        module_rows = []
        for module_id, sub in profile.groupby("module_id_norm"):
            terms = sub["term"].dropna().astype(str).tolist()
            term_cov = coverage.set_index("term").loc[terms]
            row = {
                "module_id": module_id,
                "n_BP_locked": len(terms),
                "locked_terms": " | ".join(terms),
            }
            for k in K_VALUES:
                eligible_terms = term_cov.index[term_cov[f"eligible_K{k}"]].tolist()
                excluded_terms = term_cov.index[~term_cov[f"eligible_K{k}"]].tolist()
                row[f"n_BP_retained_K{k}"] = len(eligible_terms)
                row[f"retained_fraction_K{k}"] = len(eligible_terms) / len(terms)
                row[f"eligible_terms_K{k}"] = " | ".join(eligible_terms)
                row[f"excluded_terms_K{k}"] = " | ".join(excluded_terms)
                row[f"module_observable_K{k}"] = len(eligible_terms) > 0
            module_rows.append(row)

        module_cov = pd.DataFrame(module_rows).sort_values("module_id")
        module_cov.to_csv(
            OUT / "tables" / "GSE96058_module_BP_retention_by_K.csv",
            index=False, encoding="utf-8-sig"
        )

        k_rows = []
        for k in K_VALUES:
            n_bp = int(coverage[f"eligible_K{k}"].sum())
            n_modules = int(module_cov[f"module_observable_K{k}"].sum())
            missing_modules = module_cov.loc[
                ~module_cov[f"module_observable_K{k}"], "module_id"
            ].astype(str).tolist()
            k_rows.append({
                "K_min_matched_genes": k,
                "n_selected_BP_locked": 30,
                "n_BP_eligible": n_bp,
                "BP_eligible_fraction": n_bp / 30.0,
                "n_locked_modules": 7,
                "n_modules_observable": n_modules,
                "missing_modules": " | ".join(missing_modules),
                "reconstruction_space": (
                    "full_locked_7_module"
                    if n_modules == 7
                    else "reduced_locked_module_space"
                ),
            })

        k_summary = pd.DataFrame(k_rows)
        k_summary.to_csv(
            OUT / "tables" / "GSE96058_K_sensitivity_eligibility_summary.csv",
            index=False, encoding="utf-8-sig"
        )

        contract = {
            "cohort": "GSE96058",
            "primary_platform": "GPL11154",
            "primary_locked_samples": 3069,
            "expression_matrix": str(expr_path),
            "expression_sha256": sha256_file(expr_path),
            "go_bp_gmt": str(GMT),
            "go_bp_gmt_sha256": sha256_file(GMT),
            "tcga_brca_artifact_source": str(LOCKED_SRC),
            "module_group": MODULE_GROUP,
            "locked_BP_terms": 30,
            "locked_modules": 7,
            "primary_K": 10,
            "sensitivity_K": K_VALUES,
            "endpoint_used": False,
            "next_step": (
                "Stage 2C patient-level reconstruction using the same historical "
                "scoring, centroid and risk-direction contract as METABRIC."
            ),
        }
        with (OUT / "GSE96058_STAGE2AB_LOCKED_CONTRACT.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(contract, f, indent=2, ensure_ascii=False)

        status = "PASS" if (
            int(coverage["eligible_K10"].sum()) == 30
            and int(module_cov["module_observable_K10"].sum()) == 7
        ) else "HOLD"

        summary = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": "GSE96058_STAGE2AB_TRANSFER_AND_COVERAGE",
            "status": status,
            "n_unique_expression_genes": len(genes),
            "n_selected_BP_matched_to_GMT": int(coverage["gmt_match"].sum()),
            "k_sensitivity": k_rows,
            "primary_contract": (
                "K>=10; locked 30-BP, 7-module TCGA-BRCA representation; "
                "no endpoint refitting."
            ),
            "next_step": (
                "Stage 2C reconstruction."
                if status == "PASS"
                else "Resolve K>=10 BP/module observability before reconstruction."
            ),
        }
        with (OUT / "GSE96058_STAGE2AB_SUMMARY.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        (OUT / "README_STAGE2AB.txt").write_text(
            "MOATTERS — GSE96058 Stage 2A/2B\n\n"
            f"Status: {status}\n"
            f"Expression genes: {len(genes)}\n"
            "Locked BP terms: 30\n"
            "Locked modules: 7\n"
            "Primary K: 10\n"
            "No clinical endpoint was used.\n",
            encoding="utf-8",
        )

        log(f"Stage 2A/2B completed: {status}", fh)
        for row in k_rows:
            log(
                f"K>={row['K_min_matched_genes']}: "
                f"BP={row['n_BP_eligible']}/30; "
                f"modules={row['n_modules_observable']}/7; "
                f"missing={row['missing_modules'] or 'None'}",
                fh,
            )


if __name__ == "__main__":
    main()
