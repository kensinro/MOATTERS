# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# -*- coding: utf-8 -*-
r"""
MOATTERS
TCGA-LUAD Cross-Cancer Stage 0 — local data inventory and readiness audit

Input
-----
D:\MOATTERS-Data\UCSC_XENA\Lung Adenocarcinoma (LUAD)

Output
------
D:\MOATTERS-Output\MOATTERS_LUAD_STAGE0

Purpose
-------
Inventory the local TCGA-LUAD UCSC Xena files before any reconstruction.

This stage:
1. Records file names, sizes, timestamps and SHA-256.
2. Inspects headers and small previews only.
3. Identifies likely gene-expression, phenotype, survival and clinical files.
4. Detects candidate patient/sample identifiers.
5. Searches for stage, grade, smoking, survival and other endpoint fields.
6. Produces a provisional input ranking and readiness report.

No expression matrix is loaded in full.
No BP selection, network construction, module fitting or endpoint analysis is run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from moatters.config import data_path, output_path

import pandas as pd


INPUT_DIR = data_path(r"UCSC_XENA/Lung Adenocarcinoma (LUAD)")
OUT_DIR = output_path(r"MOATTERS_LUAD_STAGE0")

PREVIEW_ROWS = 8
HASH_CHUNK = 4 * 1024 * 1024

ENDPOINT_TERMS = [
    "stage", "pathologic stage", "clinical stage",
    "grade", "tumor grade", "histologic grade",
    "os", "overall survival", "dss", "disease specific survival",
    "pfi", "progression free", "dfi", "disease free",
    "event", "death", "vital status",
    "age", "gender", "sex",
    "t", "n", "m", "node", "metastasis",
    "histology", "subtype",
    "smoking", "tobacco", "pack year", "pack_year",
    "egfr", "kras", "alk", "stk11", "tp53",
]

ID_TERMS = [
    "sample", "patient", "barcode", "_sample", "_patient", "sampleid",
]


def log(msg: str, fh) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        first = f.readline()
    counts = {
        "\t": first.count("\t"),
        ",": first.count(","),
        ";": first.count(";"),
    }
    return max(counts, key=counts.get)


def inspect_table(path: Path) -> dict:
    result = {
        "readable": False,
        "delimiter": "",
        "n_columns": None,
        "columns": [],
        "first_column": "",
        "preview_rows": [],
        "candidate_id_columns": [],
        "endpoint_keyword_columns": [],
        "orientation_guess": "",
        "error": "",
    }
    try:
        sep = detect_delimiter(path)
        df = pd.read_csv(
            path,
            sep=sep,
            nrows=PREVIEW_ROWS,
            low_memory=False,
            encoding_errors="replace",
        )
        cols = [str(c) for c in df.columns]
        candidate_ids = [
            c for c in cols
            if any(term in c.lower() for term in ID_TERMS)
        ]
        endpoint_cols = [
            c for c in cols
            if any(term in c.lower() for term in ENDPOINT_TERMS)
        ]

        ncols = len(cols)
        if ncols > 500:
            orientation = "likely_features_by_samples"
        elif ncols < 200:
            orientation = "likely_metadata_or_clinical"
        else:
            orientation = "undetermined"

        result.update({
            "readable": True,
            "delimiter": "\\t" if sep == "\t" else sep,
            "n_columns": ncols,
            "columns": cols,
            "first_column": cols[0] if cols else "",
            "preview_rows": df.astype("string").fillna("").to_dict(orient="records"),
            "candidate_id_columns": candidate_ids,
            "endpoint_keyword_columns": endpoint_cols,
            "orientation_guess": orientation,
        })
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def score_candidate(path: Path, inspection: dict) -> dict:
    name = path.name.lower()
    size_mb = path.stat().st_size / (1024 ** 2)

    scores = {
        "expression_score": 0,
        "phenotype_score": 0,
        "survival_score": 0,
        "clinical_score": 0,
        "mutation_score": 0,
        "copy_number_score": 0,
    }

    if name in {"ge.tsv", "gene_expression.tsv"} or "gene_expression" in name:
        scores["expression_score"] += 100
    if name.startswith("ge.") or "expression" in name:
        scores["expression_score"] += 60
    if "phenotype" in name:
        scores["phenotype_score"] += 100
        scores["clinical_score"] += 70
    if "clinicalmatrix" in name or "clinical" in name:
        scores["clinical_score"] += 100
    if "survival" in name or "stage_groups" in name:
        scores["survival_score"] += 100
        scores["clinical_score"] += 30
    if name.startswith("mu") or "mutation" in name:
        scores["mutation_score"] += 100
    if name.startswith("cn") or "copy" in name:
        scores["copy_number_score"] += 100

    ncols = inspection.get("n_columns") or 0
    endpoint_hits = len(inspection.get("endpoint_keyword_columns", []))

    if ncols > 500:
        scores["expression_score"] += 15
    if endpoint_hits:
        scores["phenotype_score"] += min(endpoint_hits * 2, 40)
        scores["clinical_score"] += min(endpoint_hits * 2, 40)
        survival_hits = sum(
            any(term in c.lower() for term in [
                "os", "survival", "death", "dss", "pfi", "dfi"
            ])
            for c in inspection.get("endpoint_keyword_columns", [])
        )
        scores["survival_score"] += min(survival_hits * 5, 30)

    return {
        "filename": path.name,
        "full_path": str(path),
        "size_mb": size_mb,
        **scores,
    }


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ["tables", "previews", "logs"]:
        (OUT_DIR / sub).mkdir(exist_ok=True)

    with (OUT_DIR / "logs" / "luad_stage0.log").open(
        "w", encoding="utf-8"
    ) as fh:
        log("Starting TCGA-LUAD Stage 0 inventory", fh)
        log(f"Input:  {INPUT_DIR}", fh)
        log(f"Output: {OUT_DIR}", fh)

        files = sorted([p for p in INPUT_DIR.iterdir() if p.is_file()])
        log(f"Files found: {len(files)}", fh)

        inventory_rows = []
        candidate_rows = []
        endpoint_rows = []

        for i, path in enumerate(files, start=1):
            log(f"Inspecting {i}/{len(files)}: {path.name}", fh)
            inspection = inspect_table(path)
            stat = path.stat()

            inventory_rows.append({
                "filename": path.name,
                "full_path": str(path),
                "extension": path.suffix.lower(),
                "size_bytes": stat.st_size,
                "size_mb": stat.st_size / (1024 ** 2),
                "modified_time": datetime.fromtimestamp(
                    stat.st_mtime
                ).isoformat(timespec="seconds"),
                "sha256": sha256_file(path),
                "readable_preview": inspection["readable"],
                "delimiter": inspection["delimiter"],
                "n_columns": inspection["n_columns"],
                "first_column": inspection["first_column"],
                "candidate_id_columns": " | ".join(
                    inspection["candidate_id_columns"]
                ),
                "endpoint_keyword_columns": " | ".join(
                    inspection["endpoint_keyword_columns"]
                ),
                "orientation_guess": inspection["orientation_guess"],
                "preview_error": inspection["error"],
            })
            candidate_rows.append(score_candidate(path, inspection))

            if inspection["readable"]:
                preview_path = (
                    OUT_DIR / "previews" /
                    f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', path.name)}.preview.json"
                )
                preview_path.write_text(
                    json.dumps(inspection, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                for col in inspection["endpoint_keyword_columns"]:
                    endpoint_rows.append({
                        "filename": path.name,
                        "column": col,
                        "column_lower": col.lower(),
                    })

        inventory = pd.DataFrame(inventory_rows)
        inventory.to_csv(
            OUT_DIR / "tables" / "LUAD_file_inventory.csv",
            index=False,
            encoding="utf-8-sig",
        )

        candidates = pd.DataFrame(candidate_rows)
        candidates.to_csv(
            OUT_DIR / "tables" / "LUAD_input_candidate_ranking.csv",
            index=False,
            encoding="utf-8-sig",
        )

        pd.DataFrame(endpoint_rows).drop_duplicates().to_csv(
            OUT_DIR / "tables" / "LUAD_endpoint_column_inventory.csv",
            index=False,
            encoding="utf-8-sig",
        )

        def top_candidate(score_col: str) -> dict:
            row = candidates.sort_values(
                [score_col, "size_mb"],
                ascending=[False, False],
            ).iloc[0]
            return row.to_dict()

        summary = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": "TCGA_LUAD_STAGE0_INVENTORY",
            "status": "PASS",
            "input_directory": str(INPUT_DIR),
            "output_directory": str(OUT_DIR),
            "n_files": len(files),
            "primary_expression_candidate": top_candidate("expression_score"),
            "primary_phenotype_candidate": top_candidate("phenotype_score"),
            "primary_survival_candidate": top_candidate("survival_score"),
            "primary_clinical_candidate": top_candidate("clinical_score"),
            "stage0_boundaries": [
                "No full expression matrix was loaded.",
                "No sample exclusion was performed.",
                "No stage grouping was imposed.",
                "No BP selection, network or module reconstruction was performed.",
                "Candidate rankings are provisional until Stage 1 schema inspection.",
            ],
            "next_step": (
                "Stage 1 harmonization: lock the expression matrix, patient-level "
                "sample map, stage contrast, smoking context, survival endpoints "
                "and sample overlap."
            ),
        }
        with (OUT_DIR / "LUAD_STAGE0_SUMMARY.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                summary,
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        (OUT_DIR / "README_LUAD_STAGE0.txt").write_text(
            "MOATTERS — TCGA-LUAD Stage 0\n\n"
            "Status: PASS\n"
            f"Files inventoried: {len(files)}\n"
            f"Primary expression candidate: "
            f"{summary['primary_expression_candidate']['filename']}\n"
            f"Primary phenotype candidate: "
            f"{summary['primary_phenotype_candidate']['filename']}\n"
            f"Primary clinical candidate: "
            f"{summary['primary_clinical_candidate']['filename']}\n\n"
            "This stage performs inventory only. No reconstruction or endpoint "
            "analysis is performed.\n",
            encoding="utf-8",
        )

        log("LUAD Stage 0 completed: PASS", fh)
        log(
            f"Expression candidate: "
            f"{summary['primary_expression_candidate']['filename']}",
            fh,
        )
        log(
            f"Clinical candidate: "
            f"{summary['primary_clinical_candidate']['filename']}",
            fh,
        )


if __name__ == "__main__":
    main()
