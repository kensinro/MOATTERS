# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# -*- coding: utf-8 -*-
r"""
MOATTERS
GSE96058 Stage 0 — data inventory and input-readiness audit

Input
-----
D:\MOATTERS-Data\External\GSE96058

Output
------
D:\MOATTERS-Output\MOATTERS_GSE96058_STAGE0

Purpose
-------
Inventory the local GSE96058/SCAN-B files without loading the two very large
expression matrices into memory. The script:

1. Records file size, timestamp and SHA-256.
2. Inspects CSV headers and a few rows only.
3. Estimates row/column orientation from headers.
4. Parses GEO series-matrix metadata and phenotype fields.
5. Identifies likely sample IDs, receptor/subtype/stage/survival fields.
6. Produces candidate rankings for:
   - primary gene-expression matrix
   - transcript-expression matrix
   - phenotype/clinical metadata
   - gene annotation support
7. Does not transform, normalize, or model any data.

The two large expression files are intentionally not read in full at Stage 0.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from moatters.config import data_path, output_path

import pandas as pd


INPUT_DIR = data_path(r"External\GSE96058")
OUT_DIR = output_path(r"MOATTERS_GSE96058_STAGE0")

TEXT_EXTENSIONS = {".csv", ".txt", ".tsv", ".gtf"}
HASH_CHUNK = 4 * 1024 * 1024
PREVIEW_ROWS = 8
MAX_GEO_METADATA_LINES = 10000

ENDPOINT_TERMS = [
    "er", "estrogen", "pr", "progesterone", "her2", "pam50", "subtype",
    "basal", "luminal", "stage", "grade", "survival", "overall survival",
    "relapse", "recurrence", "rfs", "distant", "event", "death", "age",
    "node", "tumor size", "treatment", "follow-up",
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


def detect_delimiter_from_line(line: str) -> str:
    candidates = [",", "\t", ";"]
    counts = {sep: line.count(sep) for sep in candidates}
    return max(counts, key=counts.get)


def inspect_csv_like(path: Path) -> dict:
    result = {
        "readable": False,
        "delimiter": "",
        "header_columns": [],
        "n_header_columns": 0,
        "preview_rows": [],
        "first_column_name": "",
        "first_values": [],
        "orientation_guess": "",
        "error": "",
    }
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            first_line = f.readline()
            if not first_line:
                return result
            sep = detect_delimiter_from_line(first_line)
            f.seek(0)
            reader = csv.reader(f, delimiter=sep)
            rows = []
            for i, row in enumerate(reader):
                rows.append(row)
                if i >= PREVIEW_ROWS:
                    break

        if not rows:
            return result

        header = rows[0]
        preview = rows[1:]
        first_values = [r[0] if r else "" for r in preview]

        sample_like = sum(
            bool(re.search(r"(GSM\d+|SCAN[-_ ]?B|SAMPLE|PATIENT)", str(x), re.I))
            for x in header[1: min(len(header), 100)]
        )
        gene_like = sum(
            bool(re.fullmatch(r"[A-Za-z0-9_.\-]+", str(x))) and len(str(x)) <= 30
            for x in first_values
        )

        if len(header) > 500 and sample_like > 0:
            orientation = "features_by_samples"
        elif len(header) > 500:
            orientation = "likely_features_by_samples"
        elif len(header) < 100 and len(preview) > 0:
            orientation = "undetermined_or_metadata"
        else:
            orientation = "undetermined"

        result.update({
            "readable": True,
            "delimiter": "\\t" if sep == "\t" else sep,
            "header_columns": header,
            "n_header_columns": len(header),
            "preview_rows": preview,
            "first_column_name": header[0] if header else "",
            "first_values": first_values,
            "orientation_guess": orientation,
        })
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def inspect_geo_series_matrix(path: Path) -> tuple[pd.DataFrame, dict]:
    metadata_rows = []
    summary = {
        "sample_count_from_header": None,
        "sample_geo_accessions": [],
        "phenotype_field_count": 0,
        "endpoint_keyword_hits": [],
        "matrix_table_detected": False,
        "error": "",
    }

    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                if line_no > MAX_GEO_METADATA_LINES:
                    break

                s = line.rstrip("\r\n")
                if s.startswith("!series_matrix_table_begin"):
                    summary["matrix_table_detected"] = True
                    break

                if not s.startswith("!Sample_"):
                    continue

                parts = s.split("\t")
                field = parts[0]
                values = [x.strip().strip('"') for x in parts[1:]]

                if field == "!Sample_geo_accession":
                    summary["sample_geo_accessions"] = values
                    summary["sample_count_from_header"] = len(values)

                if field == "!Sample_characteristics_ch1":
                    summary["phenotype_field_count"] += 1

                joined = " | ".join(values[:50]).lower()
                hits = sorted({term for term in ENDPOINT_TERMS if term in joined})
                if hits:
                    summary["endpoint_keyword_hits"].append({
                        "line_no": line_no,
                        "field": field,
                        "hits": hits,
                        "example_values": values[:10],
                    })

                metadata_rows.append({
                    "line_no": line_no,
                    "field": field,
                    "n_values": len(values),
                    "first_values": json.dumps(values[:10], ensure_ascii=False),
                    "endpoint_keyword_hits": " | ".join(hits),
                })
    except Exception as exc:
        summary["error"] = repr(exc)

    return pd.DataFrame(metadata_rows), summary


def score_file_candidate(path: Path, inspection: dict) -> dict:
    name = path.name.lower()
    size_gb = path.stat().st_size / (1024 ** 3)

    gene_score = 0
    transcript_score = 0
    clinical_score = 0
    annotation_score = 0

    if "gene_expression" in name:
        gene_score += 100
    if "transcript_expression" in name:
        transcript_score += 100
    if "series_matrix" in name:
        clinical_score += 75
    if path.suffix.lower() == ".gtf":
        annotation_score += 100
    if "transformed" in name:
        gene_score += 20
    if inspection.get("orientation_guess") in {
        "features_by_samples", "likely_features_by_samples"
    }:
        gene_score += 10
        transcript_score += 10
    if size_gb > 0.5:
        gene_score += 5
        transcript_score += 5

    return {
        "filename": path.name,
        "full_path": str(path),
        "size_gb": size_gb,
        "gene_expression_score": gene_score,
        "transcript_expression_score": transcript_score,
        "clinical_metadata_score": clinical_score,
        "annotation_score": annotation_score,
    }


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tables").mkdir(exist_ok=True)
    (OUT_DIR / "logs").mkdir(exist_ok=True)
    (OUT_DIR / "geo_metadata").mkdir(exist_ok=True)

    with (OUT_DIR / "logs" / "stage0.log").open("w", encoding="utf-8") as fh:
        log("Starting GSE96058 Stage 0 inventory", fh)
        log(f"Input:  {INPUT_DIR}", fh)
        log(f"Output: {OUT_DIR}", fh)

        files = sorted([p for p in INPUT_DIR.iterdir() if p.is_file()])
        log(f"Files found: {len(files)}", fh)

        inventory_rows = []
        candidate_rows = []
        endpoint_rows = []
        geo_summaries = []

        for i, path in enumerate(files, start=1):
            log(f"Inspecting {i}/{len(files)}: {path.name}", fh)
            stat = path.stat()
            inspection = inspect_csv_like(path) if path.suffix.lower() in TEXT_EXTENSIONS else {}

            row = {
                "filename": path.name,
                "full_path": str(path),
                "extension": path.suffix.lower(),
                "size_bytes": stat.st_size,
                "size_mb": stat.st_size / (1024 ** 2),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "sha256": sha256_file(path),
                "readable_preview": inspection.get("readable", False),
                "delimiter": inspection.get("delimiter", ""),
                "n_header_columns": inspection.get("n_header_columns", None),
                "first_column_name": inspection.get("first_column_name", ""),
                "orientation_guess": inspection.get("orientation_guess", ""),
                "first_values": json.dumps(inspection.get("first_values", []), ensure_ascii=False),
                "preview_error": inspection.get("error", ""),
            }
            inventory_rows.append(row)
            candidate_rows.append(score_file_candidate(path, inspection))

            if "series_matrix" in path.name.lower():
                geo_df, geo_summary = inspect_geo_series_matrix(path)
                geo_df.to_csv(
                    OUT_DIR / "geo_metadata" / f"{path.stem}_sample_metadata_inventory.csv",
                    index=False, encoding="utf-8-sig"
                )
                geo_summary["filename"] = path.name
                geo_summaries.append(geo_summary)

                for item in geo_summary.get("endpoint_keyword_hits", []):
                    endpoint_rows.append({
                        "filename": path.name,
                        "line_no": item["line_no"],
                        "field": item["field"],
                        "endpoint_keyword_hits": " | ".join(item["hits"]),
                        "example_values": json.dumps(item["example_values"], ensure_ascii=False),
                    })

        inventory = pd.DataFrame(inventory_rows)
        inventory.to_csv(
            OUT_DIR / "tables" / "GSE96058_file_inventory.csv",
            index=False, encoding="utf-8-sig"
        )

        candidates = pd.DataFrame(candidate_rows)
        candidates.to_csv(
            OUT_DIR / "tables" / "GSE96058_input_candidate_ranking.csv",
            index=False, encoding="utf-8-sig"
        )

        pd.DataFrame(endpoint_rows).to_csv(
            OUT_DIR / "tables" / "GSE96058_endpoint_keyword_inventory.csv",
            index=False, encoding="utf-8-sig"
        )

        with (OUT_DIR / "GSE96058_GEO_SERIES_SUMMARIES.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(geo_summaries, f, indent=2, ensure_ascii=False)

        gene_candidate = candidates.sort_values(
            ["gene_expression_score", "size_gb"], ascending=False
        ).iloc[0]
        clinical_candidate = candidates.sort_values(
            ["clinical_metadata_score", "size_gb"], ascending=False
        ).iloc[0]
        annotation_candidate = candidates.sort_values(
            ["annotation_score", "size_gb"], ascending=False
        ).iloc[0]

        summary = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": "GSE96058_STAGE0_INVENTORY",
            "status": "PASS",
            "input_directory": str(INPUT_DIR),
            "output_directory": str(OUT_DIR),
            "n_files": len(files),
            "primary_gene_expression_candidate": gene_candidate.to_dict(),
            "primary_clinical_candidate": clinical_candidate.to_dict(),
            "primary_annotation_candidate": annotation_candidate.to_dict(),
            "geo_series_summaries": geo_summaries,
            "stage0_boundaries": [
                "Large expression matrices were not loaded in full.",
                "No normalization or transformation was performed.",
                "No sample or endpoint was excluded.",
                "Candidate rankings are provisional until Stage 1 schema inspection.",
            ],
            "next_step": (
                "Stage 1: inspect the selected gene-expression schema in chunks, "
                "extract GEO phenotype fields, harmonize sample IDs/endpoints, and "
                "write canonical locked inputs."
            ),
        }
        with (OUT_DIR / "GSE96058_STAGE0_SUMMARY.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        (OUT_DIR / "README_STAGE0.txt").write_text(
            "MOATTERS — GSE96058 Stage 0\n\n"
            "Status: PASS\n"
            f"Files inventoried: {len(files)}\n"
            f"Primary gene-expression candidate: {gene_candidate['filename']}\n"
            f"Primary clinical candidate: {clinical_candidate['filename']}\n"
            f"Primary annotation candidate: {annotation_candidate['filename']}\n\n"
            "The large expression files were inspected by header/preview only and "
            "were not loaded into memory.\n",
            encoding="utf-8",
        )

        log("Stage 0 completed: PASS", fh)
        log(
            f"Primary gene-expression candidate: {gene_candidate['filename']}",
            fh
        )
        log(
            f"Primary clinical candidate: {clinical_candidate['filename']}",
            fh
        )


if __name__ == "__main__":
    main()
