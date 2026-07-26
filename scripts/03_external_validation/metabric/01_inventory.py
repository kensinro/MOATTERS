# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from moatters.config import data_path, output_path
from typing import Any

import pandas as pd

# ============================================================
# MOATTERS
# METABRIC Stage 0: input inventory and compatibility audit
# ============================================================

INPUT_DIR = data_path(r"External\brca_metabric")
OUTPUT_DIR = output_path(r"MOATTERS_METABRIC_STAGE0")

MAX_PREVIEW_ROWS = 5
MAX_SCAN_ROWS = 200
MAX_HASH_BYTES = 128 * 1024 * 1024  # full SHA256 for files <=128 MB

TEXT_EXTENSIONS = {
    ".txt", ".tsv", ".csv", ".tab", ".data", ".gct", ".maf", ".clin",
}
COMPRESSED_EXTENSIONS = {".gz", ".zip", ".bz2", ".xz"}

EXPRESSION_HINTS = (
    "expression", "mrna", "rna", "rsem", "gene", "zscore", "z-score",
    "data_expression", "illumina", "microarray", "log2",
)
CLINICAL_HINTS = (
    "clinical", "patient", "sample", "phenotype", "survival", "metadata",
    "data_clinical", "brca", "pam50", "er_status", "pr_status", "her2",
)


def ensure_output_dirs() -> dict[str, Path]:
    dirs = {
        "root": OUTPUT_DIR,
        "tables": OUTPUT_DIR / "tables",
        "previews": OUTPUT_DIR / "previews",
        "logs": OUTPUT_DIR / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def safe_stat(path: Path) -> dict[str, Any]:
    try:
        st = path.stat()
        return {
            "size_bytes": int(st.st_size),
            "modified_time": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        }
    except Exception as exc:
        return {"size_bytes": None, "modified_time": None, "stat_error": str(exc)}


def sha256_file(path: Path) -> tuple[str | None, str]:
    try:
        size = path.stat().st_size
        if size > MAX_HASH_BYTES:
            return None, f"skipped_size_gt_{MAX_HASH_BYTES}"
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest(), "complete"
    except Exception as exc:
        return None, f"error:{exc}"


def sniff_encoding(path: Path) -> str:
    candidates = ["utf-8-sig", "utf-8", "utf-16", "utf-16-le", "latin-1"]
    raw = path.read_bytes()[:65536]
    for enc in candidates:
        try:
            raw.decode(enc)
            return enc
        except Exception:
            continue
    return "latin-1"


def sniff_delimiter(path: Path, encoding: str) -> str | None:
    try:
        with path.open("r", encoding=encoding, errors="replace", newline="") as fh:
            sample = fh.read(32768)
        if not sample.strip():
            return None
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,;|")
            return dialect.delimiter
        except Exception:
            counts = {d: sample.count(d) for d in ["\t", ",", ";", "|"]}
            delim, count = max(counts.items(), key=lambda kv: kv[1])
            return delim if count > 0 else None
    except Exception:
        return None


def read_preview(path: Path, encoding: str, delimiter: str | None) -> pd.DataFrame:
    kwargs = {
        "encoding": encoding,
        "low_memory": False,
        "nrows": MAX_SCAN_ROWS,
        "comment": None,
    }
    if delimiter:
        kwargs["sep"] = delimiter
    else:
        kwargs["sep"] = None
        kwargs["engine"] = "python"

    # cBioPortal files often begin with comment/metadata lines prefixed by #.
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        kwargs["comment"] = "#"
        return pd.read_csv(path, **kwargs)


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def classify_file(path: Path, columns: list[str], shape: tuple[int, int] | None) -> dict[str, Any]:
    filename = normalize_name(path.name)
    colnorm = [normalize_name(c) for c in columns]
    text = " ".join([filename] + colnorm)

    expression_score = sum(h.replace("-", "_") in text for h in EXPRESSION_HINTS)
    clinical_score = sum(h.replace("-", "_") in text for h in CLINICAL_HINTS)

    sample_id_like = [
        c for c in colnorm
        if c in {"sample_id", "sample", "patient_id", "patient", "case_id", "study_id"}
        or "sample_id" in c or "patient_id" in c
    ]
    gene_id_like = [
        c for c in colnorm
        if c in {"hugo_symbol", "gene_symbol", "gene", "entrez_gene_id", "ensembl_gene_id"}
        or "gene_symbol" in c or "hugo" in c or "entrez" in c or "ensembl" in c
    ]

    likely_type = "unknown"
    rationale = []
    if gene_id_like and shape and shape[1] >= 20:
        expression_score += 3
        rationale.append("gene identifier column plus many columns")
    if sample_id_like:
        clinical_score += 2
        rationale.append("sample/patient identifier column")
    if any(x in text for x in ["os_months", "os_status", "dfs_months", "rfs", "pam50", "er_status"]):
        clinical_score += 3
        rationale.append("clinical endpoint columns")

    if expression_score >= clinical_score + 2:
        likely_type = "expression_or_gene_matrix"
    elif clinical_score >= expression_score + 2:
        likely_type = "clinical_or_sample_metadata"
    elif expression_score > 0 and clinical_score > 0:
        likely_type = "mixed_or_ambiguous"

    return {
        "likely_type": likely_type,
        "expression_score": expression_score,
        "clinical_score": clinical_score,
        "sample_id_columns": ";".join(sample_id_like),
        "gene_id_columns": ";".join(gene_id_like),
        "classification_rationale": "; ".join(rationale),
    }


def detect_endpoints(columns: list[str]) -> list[str]:
    patterns = {
        "ER": [r"(^|_)er(_|$)", r"er_status", r"estrogen"],
        "PR": [r"(^|_)pr(_|$)", r"pr_status", r"progesterone"],
        "HER2": [r"her2", r"erbb2"],
        "PAM50": [r"pam50", r"intrinsic_subtype", r"claudin"],
        "STAGE": [r"stage", r"tumor_stage", r"pathologic_stage"],
        "GRADE": [r"grade", r"histologic_grade"],
        "OS": [r"overall_survival", r"os_month", r"os_status", r"death"],
        "DFS/RFS": [r"disease_free", r"relapse", r"recurrence", r"rfs", r"dfs"],
        "AGE": [r"age"],
        "TREATMENT": [r"treatment", r"therapy", r"chemo", r"hormone", r"radiation"],
    }
    normalized = [normalize_name(c) for c in columns]
    found = []
    for endpoint, pats in patterns.items():
        if any(any(re.search(p, c) for p in pats) for c in normalized):
            found.append(endpoint)
    return found


def inspect_file(path: Path, dirs: dict[str, Path]) -> dict[str, Any]:
    rel = path.relative_to(INPUT_DIR)
    info: dict[str, Any] = {
        "relative_path": str(rel),
        "filename": path.name,
        "extension": path.suffix.lower(),
        "parent_folder": str(rel.parent),
        **safe_stat(path),
    }
    digest, hash_status = sha256_file(path)
    info["sha256"] = digest
    info["hash_status"] = hash_status

    suffix = path.suffix.lower()
    if suffix in COMPRESSED_EXTENSIONS:
        info.update({
            "inspection_status": "compressed_not_opened",
            "likely_type": "compressed_archive",
        })
        return info

    if suffix not in TEXT_EXTENSIONS and suffix != "":
        info.update({
            "inspection_status": "non_tabular_or_unsupported_extension",
            "likely_type": "other",
        })
        return info

    try:
        encoding = sniff_encoding(path)
        delimiter = sniff_delimiter(path, encoding)
        df = read_preview(path, encoding, delimiter)
        cols = [str(c) for c in df.columns]
        classification = classify_file(path, cols, df.shape)
        endpoints = detect_endpoints(cols)

        preview_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(rel)) + ".preview.csv"
        df.head(MAX_PREVIEW_ROWS).to_csv(dirs["previews"] / preview_name, index=False)

        info.update({
            "inspection_status": "read_success",
            "encoding": encoding,
            "delimiter": repr(delimiter),
            "preview_rows_read": int(df.shape[0]),
            "n_columns": int(df.shape[1]),
            "columns": json.dumps(cols, ensure_ascii=False),
            "detected_endpoints": ";".join(endpoints),
            **classification,
        })
    except Exception as exc:
        info.update({
            "inspection_status": "read_failed",
            "read_error": f"{type(exc).__name__}: {exc}",
            "likely_type": "unknown",
        })
    return info


def candidate_rank(df: pd.DataFrame, target_type: str) -> pd.DataFrame:
    if df.empty:
        return df
    ranked = df.copy()
    if target_type == "expression":
        ranked["candidate_score"] = pd.to_numeric(ranked.get("expression_score", 0), errors="coerce").fillna(0)
    else:
        ranked["candidate_score"] = pd.to_numeric(ranked.get("clinical_score", 0), errors="coerce").fillna(0)
    ranked = ranked.sort_values(
        ["candidate_score", "size_bytes"], ascending=[False, False], na_position="last"
    )
    return ranked


def main() -> int:
    dirs = ensure_output_dirs()
    log_path = dirs["logs"] / "stage0_run.log"

    def log(msg: str) -> None:
        stamp = datetime.now().isoformat(timespec="seconds")
        line = f"[{stamp}] {msg}"
        print(line)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log("Starting METABRIC Stage 0 inventory")
    log(f"Input:  {INPUT_DIR}")
    log(f"Output: {OUTPUT_DIR}")

    if not INPUT_DIR.exists():
        msg = f"Input directory does not exist: {INPUT_DIR}"
        log("ERROR: " + msg)
        (OUTPUT_DIR / "STAGE0_FAILED.txt").write_text(msg, encoding="utf-8")
        return 2

    files = sorted([p for p in INPUT_DIR.rglob("*") if p.is_file()])
    log(f"Files found: {len(files)}")
    if not files:
        msg = "No files found in the METABRIC input directory."
        log("ERROR: " + msg)
        (OUTPUT_DIR / "STAGE0_FAILED.txt").write_text(msg, encoding="utf-8")
        return 3

    records = []
    for idx, path in enumerate(files, 1):
        log(f"Inspecting {idx}/{len(files)}: {path.relative_to(INPUT_DIR)}")
        records.append(inspect_file(path, dirs))

    inventory = pd.DataFrame(records)
    inventory.to_csv(dirs["tables"] / "METABRIC_file_inventory.csv", index=False, encoding="utf-8-sig")

    expr = candidate_rank(inventory, "expression")
    clin = candidate_rank(inventory, "clinical")
    expr.head(20).to_csv(dirs["tables"] / "METABRIC_expression_candidates.csv", index=False, encoding="utf-8-sig")
    clin.head(20).to_csv(dirs["tables"] / "METABRIC_clinical_candidates.csv", index=False, encoding="utf-8-sig")

    endpoint_rows = inventory.loc[
        inventory.get("detected_endpoints", pd.Series(index=inventory.index, dtype=str)).fillna("").ne("")
    ].copy()
    endpoint_rows.to_csv(dirs["tables"] / "METABRIC_endpoint_inventory.csv", index=False, encoding="utf-8-sig")

    successful = int((inventory["inspection_status"] == "read_success").sum())
    failed = int((inventory["inspection_status"] == "read_failed").sum())
    top_expr = expr.iloc[0]["relative_path"] if not expr.empty else None
    top_clin = clin.iloc[0]["relative_path"] if not clin.empty else None

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_directory": str(INPUT_DIR),
        "output_directory": str(OUTPUT_DIR),
        "n_files": len(files),
        "n_tabular_read_success": successful,
        "n_tabular_read_failed": failed,
        "top_expression_candidate": top_expr,
        "top_clinical_candidate": top_clin,
        "stage0_status": "PASS" if successful > 0 else "FAIL",
        "next_step": (
            "Review candidate tables, then lock expression/clinical files and sample-ID mapping "
            "for Stage 1 reconstruction."
        ),
    }
    with (OUTPUT_DIR / "METABRIC_STAGE0_SUMMARY.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    readme = f"MOATTERS — METABRIC Stage 0\n\n"
    readme += f"Input directory: {INPUT_DIR}\nOutput directory: {OUTPUT_DIR}\n\n"
    readme += f"Files found: {len(files)}\nTabular files read successfully: {successful}\nRead failures: {failed}\n\n"
    readme += f"Top expression candidate: {top_expr}\nTop clinical candidate: {top_clin}\n\n"
    readme += "Main outputs:\n"
    readme += "- tables/METABRIC_file_inventory.csv\n"
    readme += "- tables/METABRIC_expression_candidates.csv\n"
    readme += "- tables/METABRIC_clinical_candidates.csv\n"
    readme += "- tables/METABRIC_endpoint_inventory.csv\n"
    readme += "- previews/*.preview.csv\n"
    readme += "- METABRIC_STAGE0_SUMMARY.json\n"
    readme += "\nDo not start the external reconstruction until the selected expression and clinical files, data orientation, identifiers, endpoint coding, and transform scale are explicitly locked.\n"
    (OUTPUT_DIR / "README_STAGE0.txt").write_text(readme, encoding="utf-8")

    log("Stage 0 completed")
    log(f"Top expression candidate: {top_expr}")
    log(f"Top clinical candidate: {top_clin}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        err = traceback.format_exc()
        (OUTPUT_DIR / "UNHANDLED_ERROR.txt").write_text(err, encoding="utf-8")
        print(err, file=sys.stderr)
        raise
