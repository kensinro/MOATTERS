# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# -*- coding: utf-8 -*-
r"""
MOATTERS
GSE96058 Stage 1 — canonical input lock and phenotype harmonization

Primary cohort
--------------
GPL11154 series-matrix samples (expected n=3,069).

Secondary/non-primary set
-------------------------
GPL18573 samples (expected n=340) are inventoried and retained separately. They
are not mixed into the primary external-validation cohort.

Input
-----
D:\MOATTERS-Data\External\GSE96058

Output
------
D:\MOATTERS-Output\MOATTERS_GSE96058_STAGE1

Actions
-------
1. Parse both GEO series-matrix files into sample-level metadata.
2. Read only the header of the 1.8-GB gene-expression CSV to establish sample IDs.
3. Match expression columns to GEO samples through GSM accession, title, and
   SCAN-B external ID.
4. Stream the gene-expression file in chunks and write a canonical
   genes x primary-samples matrix for GPL11154 only.
5. Harmonize ER, PR/PgR, HER2, PAM50, OS and available clinicopathologic fields.
6. Record the 340 GPL18573 samples separately.
7. Produce endpoint counts, matching diagnostics, QC and an input-lock manifest.

No endpoint is used to select genes or transform the expression values.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from moatters.config import data_path, output_path

import numpy as np
import pandas as pd


INPUT_DIR = data_path(r"External\GSE96058")
OUT_DIR = output_path(r"MOATTERS_GSE96058_STAGE1")

EXPR_FILE = INPUT_DIR / "GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv"
SERIES_PRIMARY = INPUT_DIR / "GSE96058-GPL11154_series_matrix.txt"
SERIES_SECONDARY = INPUT_DIR / "GSE96058-GPL18573_series_matrix.txt"

CHUNK_ROWS = 1000


def log(msg: str, fh) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def sha256_file(path: Path, chunk_size=4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def strip_geo_value(x: str) -> str:
    return str(x).strip().strip('"')


def normalize_id(x: str) -> str:
    if pd.isna(x):
        return ""
    s = strip_geo_value(x).strip()
    return re.sub(r"\s+", "", s).upper()


def parse_series_matrix(path: Path, platform: str) -> pd.DataFrame:
    sample_fields = {}
    characteristics = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.rstrip("\r\n")
            if s.startswith("!series_matrix_table_begin"):
                break
            if not s.startswith("!Sample_"):
                continue

            parts = s.split("\t")
            field = parts[0]
            values = [strip_geo_value(x) for x in parts[1:]]

            if field == "!Sample_characteristics_ch1":
                characteristics.append(values)
            else:
                sample_fields.setdefault(field, []).append(values)

    gsm = sample_fields.get("!Sample_geo_accession", [[]])[0]
    if not gsm:
        raise RuntimeError(f"No !Sample_geo_accession found in {path}")

    n = len(gsm)
    out = pd.DataFrame({
        "geo_accession": gsm,
        "platform": platform,
    })

    simple_map = {
        "!Sample_title": "sample_title",
        "!Sample_source_name_ch1": "source_name",
        "!Sample_description": "sample_description",
    }
    for source, target in simple_map.items():
        vals = sample_fields.get(source, [[]])[0]
        if len(vals) == n:
            out[target] = vals

    # Each characteristics line has one key:value field across samples.
    duplicate_counter = {}
    for vals in characteristics:
        if len(vals) != n:
            continue
        keys = []
        parsed_vals = []
        for v in vals:
            if ":" in v:
                key, value = v.split(":", 1)
                keys.append(key.strip().lower())
                parsed_vals.append(value.strip())
            else:
                keys.append("")
                parsed_vals.append(v.strip())

        nonempty_keys = [k for k in keys if k]
        if not nonempty_keys:
            continue
        canonical_key = pd.Series(nonempty_keys).mode().iloc[0]
        canonical_key = re.sub(r"[^a-z0-9]+", "_", canonical_key).strip("_")
        duplicate_counter[canonical_key] = duplicate_counter.get(canonical_key, 0) + 1
        if duplicate_counter[canonical_key] > 1:
            canonical_key = f"{canonical_key}_{duplicate_counter[canonical_key]}"
        out[canonical_key] = parsed_vals

    return out


def clean_binary(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.upper()
    out = pd.Series(pd.NA, index=s.index, dtype="Float64")
    out[s.isin(["1", "POS", "POSITIVE", "TRUE", "YES"])] = 1.0
    out[s.isin(["0", "NEG", "NEGATIVE", "FALSE", "NO"])] = 0.0
    return out


def choose_first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((x for x in names if x in df.columns), None)


def harmonize_clinical(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Stable matching aliases.
    out["geo_accession_norm"] = out["geo_accession"].map(normalize_id)
    if "sample_title" in out:
        out["sample_title_norm"] = out["sample_title"].map(normalize_id)

    external_col = choose_first_existing(out, [
        "scan_b_external_id", "scan_b_external_id_2", "scan_b_id",
    ])
    if external_col:
        out["scan_b_external_id_clean"] = out[external_col].astype("string").str.strip()
        out["scan_b_external_id_norm"] = out[external_col].map(normalize_id)
    else:
        out["scan_b_external_id_clean"] = pd.NA
        out["scan_b_external_id_norm"] = ""

    # Primary pathology receptor fields.
    er_col = choose_first_existing(out, ["er_status"])
    pr_col = choose_first_existing(out, ["pgr_status", "pr_status"])
    her2_col = choose_first_existing(out, ["her2_status"])

    out["ER_binary"] = clean_binary(out[er_col]) if er_col else pd.Series(pd.NA, index=out.index, dtype="Float64")
    out["PR_binary"] = clean_binary(out[pr_col]) if pr_col else pd.Series(pd.NA, index=out.index, dtype="Float64")
    out["HER2_binary"] = clean_binary(out[her2_col]) if her2_col else pd.Series(pd.NA, index=out.index, dtype="Float64")

    # MGC and SGC predictions are retained as separate secondary endpoints.
    for source, target in [
        ("er_prediction_mgc", "ER_MGC_binary"),
        ("pgr_prediction_mgc", "PR_MGC_binary"),
        ("her2_prediction_mgc", "HER2_MGC_binary"),
        ("er_prediction_sgc", "ER_SGC_binary"),
        ("pgr_prediction_sgc", "PR_SGC_binary"),
        ("her2_prediction_sgc", "HER2_SGC_binary"),
    ]:
        if source in out:
            out[target] = clean_binary(out[source])

    pam_col = choose_first_existing(out, ["pam50_subtype"])
    if pam_col:
        p = out[pam_col].astype("string").str.strip()
        pu = p.str.upper()
        out["PAM50_clean"] = p
        out["PAM50_luminal_binary"] = pd.Series(
            np.where(pu.isin(["LUMA", "LUMB"]), 1.0,
                     np.where(pu.notna(), 0.0, np.nan)),
            index=out.index, dtype="Float64"
        )
        out["PAM50_basal_binary"] = pd.Series(
            np.where(pu.eq("BASAL").fillna(False), 1.0,
                     np.where(pu.notna(), 0.0, np.nan)),
            index=out.index, dtype="Float64"
        )

    # Survival in days.
    os_time_col = choose_first_existing(out, ["overall_survival_days"])
    os_event_col = choose_first_existing(out, ["overall_survival_event"])
    if os_time_col:
        out["OS_time_days"] = pd.to_numeric(out[os_time_col], errors="coerce")
        out["OS_time_months"] = out["OS_time_days"] / 30.4375
    if os_event_col:
        out["OS_event"] = clean_binary(out[os_event_col])

    # Other useful clinicopathologic fields.
    for source, target in [
        ("age_at_diagnosis", "age_at_diagnosis"),
        ("tumor_size", "tumor_size_mm"),
    ]:
        if source in out:
            out[target] = pd.to_numeric(out[source], errors="coerce")

    if "lymph_node_status" in out:
        s = out["lymph_node_status"].astype("string").str.upper()
        out["node_positive_binary"] = pd.Series(
            np.where(s.str.contains("POSITIVE", na=False), 1.0,
                     np.where(s.str.contains("NEGATIVE", na=False), 0.0, np.nan)),
            index=out.index, dtype="Float64"
        )

    return out


def inspect_expression_header(path: Path) -> tuple[str, list[str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    if len(header) < 2:
        raise RuntimeError("Expression header has fewer than two columns.")
    return header[0], header[1:]


def build_alias_map(clinical: pd.DataFrame) -> dict[str, list[int]]:
    aliases = {}
    for idx, row in clinical.iterrows():
        values = [
            row.get("geo_accession", ""),
            row.get("sample_title", ""),
            row.get("scan_b_external_id_clean", ""),
        ]
        # Also include major dot-delimited components because expression columns
        # may use shortened SCAN-B identifiers.
        ext = str(row.get("scan_b_external_id_clean", "") or "")
        values.extend([x for x in ext.split(".") if x])

        for value in values:
            key = normalize_id(value)
            if not key:
                continue
            aliases.setdefault(key, []).append(idx)
    return aliases


def match_expression_columns(expr_columns: list[str], clinical: pd.DataFrame) -> pd.DataFrame:
    alias_map = build_alias_map(clinical)
    rows = []

    for col in expr_columns:
        key = normalize_id(col)
        candidate_keys = [key]

        # Header forms may have prefixes/suffixes or use one SCAN-B component.
        candidate_keys.extend([
            normalize_id(x) for x in re.split(r"[|;,\s]+", str(col)) if x
        ])
        candidate_keys.extend([
            normalize_id(x) for x in str(col).split(".") if x
        ])

        hits = []
        matched_key = ""
        for ck in candidate_keys:
            if ck in alias_map:
                hits = alias_map[ck]
                matched_key = ck
                break

        if len(hits) == 1:
            idx = hits[0]
            row = clinical.loc[idx]
            rows.append({
                "expression_column": col,
                "expression_column_norm": key,
                "match_status": "unique_match",
                "matched_alias": matched_key,
                "clinical_row_index": int(idx),
                "geo_accession": row["geo_accession"],
                "platform": row["platform"],
                "sample_title": row.get("sample_title", ""),
                "scan_b_external_id": row.get("scan_b_external_id_clean", ""),
            })
        elif len(hits) > 1:
            rows.append({
                "expression_column": col,
                "expression_column_norm": key,
                "match_status": "ambiguous_match",
                "matched_alias": matched_key,
                "clinical_row_index": pd.NA,
                "geo_accession": "",
                "platform": "",
                "sample_title": "",
                "scan_b_external_id": "",
            })
        else:
            rows.append({
                "expression_column": col,
                "expression_column_norm": key,
                "match_status": "unmatched",
                "matched_alias": "",
                "clinical_row_index": pd.NA,
                "geo_accession": "",
                "platform": "",
                "sample_title": "",
                "scan_b_external_id": "",
            })

    return pd.DataFrame(rows)


def stream_primary_expression(
    source: Path,
    first_col: str,
    selected_columns: list[str],
    output_path: Path,
    qc_path: Path,
    fh,
) -> dict:
    # The source CSV has a blank first header cell. pandas renames that column
    # internally (typically to "Unnamed: 0"), so selecting it by the literal
    # empty string fails. Resolve the actual pandas column name from a header-only
    # read, while preserving the expression sample columns by name.
    header_df = pd.read_csv(source, nrows=0)
    actual_columns = list(header_df.columns)
    if not actual_columns:
        raise RuntimeError("Expression file has no readable columns.")

    actual_first_col = actual_columns[0]
    missing_selected = [c for c in selected_columns if c not in actual_columns]
    if missing_selected:
        raise RuntimeError(
            f"{len(missing_selected)} selected expression columns were not found. "
            f"Examples: {missing_selected[:10]}"
        )

    usecols = [actual_first_col] + selected_columns
    first_write = True
    qc_rows = []
    n_rows = 0
    n_duplicate_symbols = 0
    seen = set()

    # Output uncompressed temporary file, then gzip once. This is considerably
    # faster than repeatedly appending compressed chunks.
    temp_path = output_path.with_suffix("")
    log(
        f"Resolved blank first CSV header to pandas column '{actual_first_col}'",
        fh,
    )

    for chunk_no, chunk in enumerate(
        pd.read_csv(source, usecols=usecols, chunksize=CHUNK_ROWS, low_memory=False),
        start=1,
    ):
        gene_col = actual_first_col
        chunk = chunk.rename(columns={gene_col: "gene_symbol"})
        chunk["gene_symbol"] = (
            chunk["gene_symbol"].astype("string").str.strip().str.upper()
        )
        chunk = chunk[chunk["gene_symbol"].notna() & chunk["gene_symbol"].ne("")]

        values = chunk.drop(columns=["gene_symbol"]).apply(pd.to_numeric, errors="coerce")
        chunk = pd.concat([chunk[["gene_symbol"]], values], axis=1)

        # Gene-level file should already be unique; record and aggregate if not.
        dup = int(chunk["gene_symbol"].duplicated(keep=False).sum())
        n_duplicate_symbols += dup
        if dup:
            chunk = chunk.groupby("gene_symbol", as_index=False).mean(numeric_only=True)

        for _, row in chunk.iterrows():
            gene = row["gene_symbol"]
            arr = pd.to_numeric(row.iloc[1:], errors="coerce")
            qc_rows.append({
                "gene_symbol": gene,
                "missing_fraction": float(arr.isna().mean()),
                "mean_expression": float(arr.mean()) if arr.notna().any() else np.nan,
                "sd_expression": float(arr.std(ddof=1)) if arr.notna().sum() > 1 else np.nan,
            })
            seen.add(gene)

        chunk.to_csv(
            temp_path,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )
        first_write = False
        n_rows += len(chunk)
        log(f"Expression chunk {chunk_no}: cumulative genes={n_rows}", fh)

    with temp_path.open("rb") as src, gzip.open(output_path, "wb", compresslevel=5) as dst:
        while True:
            b = src.read(4 * 1024 * 1024)
            if not b:
                break
            dst.write(b)
    temp_path.unlink(missing_ok=True)

    pd.DataFrame(qc_rows).drop_duplicates("gene_symbol").to_csv(
        qc_path, index=False, encoding="utf-8-sig"
    )

    return {
        "n_gene_rows_written": n_rows,
        "n_unique_gene_symbols": len(seen),
        "duplicate_symbol_rows_detected": n_duplicate_symbols,
        "n_samples": len(selected_columns),
    }


def endpoint_inventory(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    endpoints = [
        "ER_binary", "PR_binary", "HER2_binary",
        "ER_MGC_binary", "PR_MGC_binary", "HER2_MGC_binary",
        "ER_SGC_binary", "PR_SGC_binary", "HER2_SGC_binary",
        "PAM50_luminal_binary", "PAM50_basal_binary",
        "node_positive_binary", "OS_event",
    ]
    for col in endpoints:
        if col not in df:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "endpoint": col,
            "n_total": len(df),
            "n_usable": int(x.isin([0, 1]).sum()),
            "n_positive": int((x == 1).sum()),
            "n_negative": int((x == 0).sum()),
            "n_missing_or_other": int((~x.isin([0, 1])).sum()),
        })

    if "PAM50_clean" in df:
        counts = df["PAM50_clean"].value_counts(dropna=False)
        for level, n in counts.items():
            rows.append({
                "endpoint": f"PAM50_level::{level}",
                "n_total": len(df),
                "n_usable": int(n),
                "n_positive": int(n),
                "n_negative": pd.NA,
                "n_missing_or_other": pd.NA,
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ["tables", "matrices", "logs", "audit"]:
        (OUT_DIR / sub).mkdir(exist_ok=True)

    with (OUT_DIR / "logs" / "stage1.log").open("w", encoding="utf-8") as fh:
        log("Starting GSE96058 Stage 1 input lock", fh)

        for p in [EXPR_FILE, SERIES_PRIMARY, SERIES_SECONDARY]:
            if not p.exists():
                raise FileNotFoundError(p)

        primary_meta = parse_series_matrix(SERIES_PRIMARY, "GPL11154")
        secondary_meta = parse_series_matrix(SERIES_SECONDARY, "GPL18573")
        all_meta = pd.concat([primary_meta, secondary_meta], ignore_index=True)
        all_meta = harmonize_clinical(all_meta)

        log(
            f"GEO metadata parsed: GPL11154={len(primary_meta)}, "
            f"GPL18573={len(secondary_meta)}, total={len(all_meta)}",
            fh,
        )

        first_col, expr_columns = inspect_expression_header(EXPR_FILE)
        log(
            f"Expression header: first column='{first_col}', "
            f"sample columns={len(expr_columns)}",
            fh,
        )

        matches = match_expression_columns(expr_columns, all_meta)
        matches.to_csv(
            OUT_DIR / "audit" / "GSE96058_expression_to_GEO_sample_matching.csv",
            index=False, encoding="utf-8-sig"
        )

        status_counts = matches["match_status"].value_counts().to_dict()
        log(f"Expression-column matching: {status_counts}", fh)

        primary_matches = matches[
            (matches["match_status"] == "unique_match")
            & (matches["platform"] == "GPL11154")
        ].copy()
        secondary_matches = matches[
            (matches["match_status"] == "unique_match")
            & (matches["platform"] == "GPL18573")
        ].copy()

        # One expression column per GEO sample. Keep first and record duplicates.
        primary_matches["duplicate_geo_accession"] = primary_matches[
            "geo_accession"
        ].duplicated(keep=False)
        primary_locked = primary_matches.drop_duplicates(
            "geo_accession", keep="first"
        ).copy()

        if len(primary_locked) < 3000:
            raise RuntimeError(
                f"Primary GPL11154 matching unexpectedly low: {len(primary_locked)}"
            )

        primary_clinical = all_meta[
            all_meta["geo_accession"].isin(primary_locked["geo_accession"])
        ].copy()
        expr_name_map = primary_locked.set_index("geo_accession")[
            "expression_column"
        ].to_dict()
        primary_clinical["expression_column"] = primary_clinical[
            "geo_accession"
        ].map(expr_name_map)
        primary_clinical["sample_id"] = primary_clinical["geo_accession"]

        # Preserve expression-column order.
        order = {
            col: i for i, col in enumerate(primary_locked["expression_column"])
        }
        primary_clinical["_order"] = primary_clinical["expression_column"].map(order)
        primary_clinical = primary_clinical.sort_values("_order").drop(columns="_order")

        primary_clinical.to_csv(
            OUT_DIR / "tables" / "GSE96058_primary_clinical_harmonized.tsv",
            sep="\t", index=False
        )
        all_meta[all_meta["platform"] == "GPL18573"].to_csv(
            OUT_DIR / "tables" / "GSE96058_secondary_GPL18573_clinical.tsv",
            sep="\t", index=False
        )

        endpoint_inventory(primary_clinical).to_csv(
            OUT_DIR / "tables" / "GSE96058_endpoint_inventory_locked.csv",
            index=False, encoding="utf-8-sig"
        )

        expr_output = (
            OUT_DIR / "matrices" /
            "GSE96058_primary_expression_genes_x_samples.tsv.gz"
        )
        expr_qc = OUT_DIR / "tables" / "GSE96058_primary_gene_QC.csv"
        stream_info = stream_primary_expression(
            EXPR_FILE,
            first_col,
            primary_locked["expression_column"].tolist(),
            expr_output,
            expr_qc,
            fh,
        )

        # Write sample-map in canonical order.
        primary_locked.to_csv(
            OUT_DIR / "tables" / "GSE96058_primary_sample_lock.csv",
            index=False, encoding="utf-8-sig"
        )
        secondary_matches.to_csv(
            OUT_DIR / "tables" / "GSE96058_secondary_sample_matches.csv",
            index=False, encoding="utf-8-sig"
        )

        manifest = pd.DataFrame([
            {
                "role": "primary_gene_expression_source",
                "path": str(EXPR_FILE),
                "sha256": sha256_file(EXPR_FILE),
                "selection": "GPL11154-matched columns only",
            },
            {
                "role": "primary_GEO_metadata",
                "path": str(SERIES_PRIMARY),
                "sha256": sha256_file(SERIES_PRIMARY),
                "selection": "all uniquely expression-matched GPL11154 samples",
            },
            {
                "role": "secondary_GEO_metadata",
                "path": str(SERIES_SECONDARY),
                "sha256": sha256_file(SERIES_SECONDARY),
                "selection": "retained separately; not mixed into primary cohort",
            },
        ])
        manifest.to_csv(
            OUT_DIR / "tables" / "GSE96058_STAGE1_INPUT_LOCK_MANIFEST.csv",
            index=False, encoding="utf-8-sig"
        )

        summary = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": "GSE96058_STAGE1_INPUT_LOCK",
            "status": "PASS",
            "primary_platform": "GPL11154",
            "primary_metadata_samples": int(len(primary_meta)),
            "secondary_metadata_samples": int(len(secondary_meta)),
            "expression_sample_columns": int(len(expr_columns)),
            "match_status_counts": status_counts,
            "primary_unique_locked_samples": int(len(primary_locked)),
            "secondary_unique_matched_samples": int(
                secondary_matches["geo_accession"].nunique()
            ),
            "primary_expression": stream_info,
            "endpoint_inventory": endpoint_inventory(
                primary_clinical
            ).to_dict(orient="records"),
            "important_boundaries": [
                "GPL11154 is the primary external-validation cohort.",
                "GPL18573 samples are retained separately and not pooled.",
                "The supplied gene-level transformed matrix is used as provided.",
                "No endpoint was used to select genes or samples beyond availability.",
                "Pathology receptor fields are primary; MGC/SGC predictions are secondary.",
            ],
            "next_step": (
                "Stage 2A/2B cohort-specific transfer manifest and GO-BP coverage "
                "audit using the same locked TCGA-BRCA artifacts as METABRIC."
            ),
        }
        with (OUT_DIR / "GSE96058_STAGE1_SUMMARY.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        (OUT_DIR / "README_STAGE1.txt").write_text(
            "MOATTERS — GSE96058 Stage 1\n\n"
            "Status: PASS\n"
            "Primary platform: GPL11154\n"
            f"Primary locked samples: {len(primary_locked)}\n"
            f"Secondary GPL18573 samples matched: "
            f"{secondary_matches['geo_accession'].nunique()}\n"
            f"Expression genes written: {stream_info['n_unique_gene_symbols']}\n\n"
            "Pathology ER/PR/HER2 fields are retained as primary endpoints. "
            "MGC and SGC predictions are stored as secondary endpoint variants.\n",
            encoding="utf-8",
        )

        log("Stage 1 completed: PASS", fh)
        log(f"Primary locked samples: {len(primary_locked)}", fh)
        log(f"Expression genes: {stream_info['n_unique_gene_symbols']}", fh)


if __name__ == "__main__":
    main()
