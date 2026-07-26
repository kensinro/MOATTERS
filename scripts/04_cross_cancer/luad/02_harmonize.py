# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# -*- coding: utf-8 -*-
r"""
MOATTERS
TCGA-LUAD Cross-Cancer Stage 1 — expression/clinical harmonization and stage lock

Input
-----
D:\MOATTERS-Data\UCSC_XENA\Lung Adenocarcinoma (LUAD)

Output
------
D:\MOATTERS-Output\MOATTERS_LUAD_STAGE1

Purpose
-------
1. Lock GE.tsv as the primary transcriptomic matrix.
2. Retain primary-tumour samples (TCGA sample type code 01).
3. Harmonize sample-level expression with patient-level clinical metadata.
4. Define the primary reconstruction contrast:
      Early = Stage I/II
      Late  = Stage III/IV
5. Derive OS time/event.
6. Retain T/N/M, age, sex, smoking history and pack-years as context.
7. Write canonical genes x samples expression and patient-level clinical tables.

No GO-BP scoring or endpoint-driven feature selection is performed here.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from moatters.config import data_path, output_path

import numpy as np
import pandas as pd


INPUT_DIR = data_path(r"UCSC_XENA/Lung Adenocarcinoma (LUAD)")
OUT_DIR = output_path(r"MOATTERS_LUAD_STAGE1")

GE = INPUT_DIR / "GE.tsv"
CLINICAL = INPUT_DIR / "TCGA.LUAD.sampleMap_LUAD_clinicalMatrix"
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


def patient_barcode(sample: str) -> str:
    s = str(sample).strip().upper()
    return "-".join(s.split("-")[:3]) if s.startswith("TCGA-") else s


def sample_type_code(sample: str) -> str:
    parts = str(sample).strip().upper().split("-")
    if len(parts) < 4:
        return ""
    return parts[3][:2]


def stage_group(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    # Covers Stage IA/IB/IIA/IIB and IIIA/IIIB/IV.
    if re.search(r"\bSTAGE\s+(I|II)(?:[ABC])?\b", s):
        return "EARLY"
    if re.search(r"\bSTAGE\s+(III|IV)(?:[ABC])?\b", s):
        return "LATE"
    return ""


def numeric_or_nan(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def harmonize_clinical(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["sample_id"] = out["sampleID"].astype(str).str.strip()
    out["patient_id"] = out["_PATIENT"].astype(str).str.strip().str.upper()
    out["sample_type_code"] = out["sample_id"].map(sample_type_code)
    out["is_primary_tumor"] = out["sample_type_code"].eq("01")

    out["stage_group"] = out["pathologic_stage"].map(stage_group)
    out["stage_late_binary"] = out["stage_group"].map(
        {"EARLY": 0.0, "LATE": 1.0}
    ).astype("Float64")

    out["age_at_diagnosis"] = numeric_or_nan(
        out["age_at_initial_pathologic_diagnosis"]
    )
    out["sex"] = out["gender"].astype("string").str.upper().str.strip()

    vital = out["vital_status"].astype("string").str.upper().str.strip()
    out["vital_status_clean"] = vital
    out["OS_event"] = pd.Series(pd.NA, index=out.index, dtype="Float64")
    dead_mask = vital.str.contains(r"DECEASED|DEAD", na=False, regex=True)
    living_mask = vital.str.contains(r"LIVING|ALIVE", na=False, regex=True)
    out.loc[dead_mask, "OS_event"] = 1.0
    out.loc[living_mask, "OS_event"] = 0.0

    d_death = numeric_or_nan(out["days_to_death"])
    d_follow = numeric_or_nan(out["days_to_last_followup"])
    out["OS_time_days"] = pd.Series(np.nan, index=out.index, dtype="float64")
    event1 = out["OS_event"].eq(1).fillna(False)
    event0 = out["OS_event"].eq(0).fillna(False)
    out.loc[event1, "OS_time_days"] = d_death.loc[event1]
    out.loc[event0, "OS_time_days"] = d_follow.loc[event0]

    missing_time = out["OS_time_days"].isna()
    out.loc[missing_time, "OS_time_days"] = pd.concat(
        [d_death, d_follow], axis=1
    ).max(axis=1, skipna=True).loc[missing_time]
    out["OS_time_months"] = out["OS_time_days"] / 30.4375

    out["pack_years_smoked"] = numeric_or_nan(
        out["number_pack_years_smoked"]
    )
    out["tobacco_smoking_history_clean"] = (
        out["tobacco_smoking_history"]
        .astype("string").str.strip()
    )
    out["tobacco_smoking_indicator_clean"] = (
        out["tobacco_smoking_history_indicator"]
        .astype("string").str.strip()
    )
    out["stopped_smoking_year_numeric"] = numeric_or_nan(
        out["stopped_smoking_year"]
    )
    out["smoking_onset_year_numeric"] = numeric_or_nan(
        out["year_of_tobacco_smoking_onset"]
    )

    for source, target in [
        ("pathologic_T", "pathologic_T_clean"),
        ("pathologic_N", "pathologic_N_clean"),
        ("pathologic_M", "pathologic_M_clean"),
        ("pathologic_stage", "pathologic_stage_clean"),
        ("histological_type", "histological_type_clean"),
        ("Expression_Subtype", "expression_subtype_clean"),
        ("EGFR", "EGFR_clean"),
        ("KRAS", "KRAS_clean"),
        ("ALK_translocation", "ALK_translocation_clean"),
        ("STK11", "STK11_clean"),
    ]:
        if source in out.columns:
            out[target] = out[source].astype("string").str.strip()

    return out


def read_header(path: Path) -> tuple[str, list[str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
    return header[0], header[1:]


def stream_expression(
    source: Path,
    selected_samples: list[str],
    output_path: Path,
    gene_qc_path: Path,
    fh,
) -> dict:
    header_df = pd.read_csv(source, sep="\t", nrows=0)
    actual_cols = list(header_df.columns)
    first_col = actual_cols[0]

    missing = [c for c in selected_samples if c not in actual_cols]
    if missing:
        raise RuntimeError(
            f"{len(missing)} selected samples absent from GE.tsv. "
            f"Examples: {missing[:10]}"
        )

    usecols = [first_col] + selected_samples
    temp = output_path.with_suffix("")
    first_write = True
    qc_parts = []
    n_rows = 0

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            source,
            sep="\t",
            usecols=usecols,
            chunksize=CHUNK_ROWS,
            low_memory=False,
        ),
        start=1,
    ):
        chunk = chunk.rename(columns={first_col: "gene_symbol"})
        chunk["gene_symbol"] = (
            chunk["gene_symbol"].astype("string").str.strip().str.upper()
        )
        chunk = chunk[
            chunk["gene_symbol"].notna()
            & chunk["gene_symbol"].ne("")
        ]
        values = chunk.drop(columns=["gene_symbol"]).apply(
            pd.to_numeric, errors="coerce"
        )
        chunk = pd.concat([chunk[["gene_symbol"]], values], axis=1)

        if chunk["gene_symbol"].duplicated(keep=False).any():
            chunk = chunk.groupby(
                "gene_symbol", as_index=False
            ).mean(numeric_only=True)

        arr = chunk.drop(columns=["gene_symbol"])
        qc_parts.append(pd.DataFrame({
            "gene_symbol": chunk["gene_symbol"].to_numpy(),
            "missing_fraction": arr.isna().mean(axis=1).to_numpy(),
            "mean_expression": arr.mean(axis=1).to_numpy(),
            "sd_expression": arr.std(axis=1, ddof=1).to_numpy(),
        }))

        chunk.to_csv(
            temp,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )
        first_write = False
        n_rows += len(chunk)
        log(f"Expression chunk {chunk_no}: cumulative genes={n_rows}", fh)

    with temp.open("rb") as src, gzip.open(
        output_path, "wb", compresslevel=5
    ) as dst:
        while True:
            b = src.read(4 * 1024 * 1024)
            if not b:
                break
            dst.write(b)
    temp.unlink(missing_ok=True)

    qcdf = pd.concat(qc_parts, ignore_index=True)
    qcdf = qcdf.drop_duplicates("gene_symbol", keep="first")
    qcdf.to_csv(gene_qc_path, index=False, encoding="utf-8-sig")

    return {
        "n_gene_rows_written": int(n_rows),
        "n_unique_gene_symbols": int(qcdf["gene_symbol"].nunique()),
        "n_samples": int(len(selected_samples)),
        "n_zero_variance_genes": int(
            pd.to_numeric(qcdf["sd_expression"], errors="coerce")
            .fillna(0).eq(0).sum()
        ),
        "n_genes_with_any_missing": int(
            pd.to_numeric(qcdf["missing_fraction"], errors="coerce")
            .gt(0).sum()
        ),
    }


def main():
    for p in [GE, CLINICAL]:
        if not p.exists():
            raise FileNotFoundError(p)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ["tables", "matrices", "logs", "audit"]:
        (OUT_DIR / sub).mkdir(exist_ok=True)

    with (OUT_DIR / "logs" / "luad_stage1.log").open(
        "w", encoding="utf-8"
    ) as fh:
        log("Starting TCGA-LUAD Stage 1 harmonization", fh)

        first_col, expression_samples = read_header(GE)
        log(
            f"GE.tsv header: first column='{first_col}', "
            f"sample columns={len(expression_samples)}",
            fh,
        )

        clin = pd.read_csv(CLINICAL, sep="\t", low_memory=False)
        clin = harmonize_clinical(clin)
        log(f"Clinical rows: {len(clin)}", fh)

        expr_map = pd.DataFrame({"expression_sample": expression_samples})
        expr_map["patient_id"] = expr_map["expression_sample"].map(
            patient_barcode
        )
        expr_map["sample_type_code"] = expr_map[
            "expression_sample"
        ].map(sample_type_code)
        expr_map["is_primary_tumor"] = expr_map[
            "sample_type_code"
        ].eq("01")

        primary_expr = expr_map[expr_map["is_primary_tumor"]].copy()
        log(
            f"Primary-tumour expression samples: {len(primary_expr)}",
            fh,
        )

        clin_primary = clin[clin["is_primary_tumor"]].copy()
        direct = primary_expr.merge(
            clin_primary,
            left_on="expression_sample",
            right_on="sample_id",
            how="left",
            suffixes=("", "_clinical"),
        )
        direct["matched_clinical"] = direct["_PATIENT"].notna()
        log(
            f"Direct expression-clinical matches: "
            f"{int(direct['matched_clinical'].sum())}/{len(direct)}",
            fh,
        )

        direct["stage_usable"] = direct["stage_group"].isin(
            ["EARLY", "LATE"]
        )
        direct["os_usable"] = (
            pd.to_numeric(direct["OS_time_days"], errors="coerce").gt(0)
            & pd.to_numeric(direct["OS_event"], errors="coerce").isin([0, 1])
        )
        direct = direct.sort_values(
            ["patient_id", "stage_usable", "os_usable", "expression_sample"],
            ascending=[True, False, False, True],
        )
        locked = direct.drop_duplicates(
            "patient_id", keep="first"
        ).copy()
        locked = locked[locked["matched_clinical"]].copy()

        selected_samples = locked["expression_sample"].tolist()
        log(
            f"Locked patient-level primary-tumour samples: {len(selected_samples)}",
            fh,
        )

        clinical_cols = [
            "expression_sample", "patient_id", "sample_type_code",
            "pathologic_stage_clean", "stage_group", "stage_late_binary",
            "pathologic_T_clean", "pathologic_N_clean", "pathologic_M_clean",
            "age_at_diagnosis", "sex", "vital_status_clean",
            "OS_time_days", "OS_time_months", "OS_event",
            "pack_years_smoked",
            "tobacco_smoking_history_clean",
            "tobacco_smoking_indicator_clean",
            "stopped_smoking_year_numeric",
            "smoking_onset_year_numeric",
            "histological_type_clean",
            "expression_subtype_clean",
            "EGFR_clean", "KRAS_clean",
            "ALK_translocation_clean", "STK11_clean",
        ]
        clinical_cols = [c for c in clinical_cols if c in locked.columns]
        clinical_locked = locked[clinical_cols].copy()
        clinical_locked = clinical_locked.rename(
            columns={"expression_sample": "sample_id"}
        )
        clinical_locked.to_csv(
            OUT_DIR / "tables" / "LUAD_primary_clinical_harmonized.tsv",
            sep="\t",
            index=False,
        )

        locked[[
            "expression_sample", "patient_id", "sample_type_code",
            "matched_clinical", "stage_usable", "os_usable",
        ]].to_csv(
            OUT_DIR / "tables" / "LUAD_primary_sample_lock.csv",
            index=False,
            encoding="utf-8-sig",
        )

        endpoint_rows = []
        for endpoint, col in [
            ("stage_late", "stage_late_binary"),
            ("OS_event", "OS_event"),
        ]:
            x = pd.to_numeric(clinical_locked[col], errors="coerce")
            endpoint_rows.append({
                "endpoint": endpoint,
                "n_total": len(clinical_locked),
                "n_usable": int(x.isin([0, 1]).sum()),
                "n_positive": int((x == 1).sum()),
                "n_negative": int((x == 0).sum()),
                "n_missing_or_other": int((~x.isin([0, 1])).sum()),
            })

        smoking = pd.to_numeric(
            clinical_locked.get("pack_years_smoked"), errors="coerce"
        )
        endpoint_rows.append({
            "endpoint": "pack_years_smoked",
            "n_total": len(clinical_locked),
            "n_usable": int(smoking.notna().sum()),
            "n_positive": np.nan,
            "n_negative": np.nan,
            "n_missing_or_other": int(smoking.isna().sum()),
        })
        pd.DataFrame(endpoint_rows).to_csv(
            OUT_DIR / "tables" / "LUAD_endpoint_inventory_locked.csv",
            index=False,
            encoding="utf-8-sig",
        )

        expr_output = (
            OUT_DIR / "matrices" /
            "LUAD_primary_expression_genes_x_samples.tsv.gz"
        )
        expr_qc = OUT_DIR / "tables" / "LUAD_primary_gene_QC.csv"
        expr_info = stream_expression(
            GE,
            selected_samples,
            expr_output,
            expr_qc,
            fh,
        )

        stage_counts = clinical_locked["stage_group"].value_counts(
            dropna=False
        ).to_dict()
        sample_type_counts = expr_map["sample_type_code"].value_counts(
            dropna=False
        ).to_dict()
        os_event_count = int(
            pd.to_numeric(
                clinical_locked["OS_event"], errors="coerce"
            ).eq(1).sum()
        )

        manifest = pd.DataFrame([
            {
                "role": "primary_gene_expression_source",
                "path": str(GE),
                "sha256": sha256_file(GE),
                "selection": "one primary-tumour expression sample per patient",
            },
            {
                "role": "primary_clinical_source",
                "path": str(CLINICAL),
                "sha256": sha256_file(CLINICAL),
                "selection": "matched primary-tumour clinical metadata",
            },
        ])
        manifest.to_csv(
            OUT_DIR / "tables" / "LUAD_STAGE1_INPUT_LOCK_MANIFEST.csv",
            index=False,
            encoding="utf-8-sig",
        )

        status = (
            "PASS"
            if stage_counts.get("EARLY", 0) >= 50
            and stage_counts.get("LATE", 0) >= 50
            and os_event_count >= 20
            else "HOLD"
        )

        summary = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": "TCGA_LUAD_STAGE1_HARMONIZATION",
            "status": status,
            "expression_source_sample_columns": len(expression_samples),
            "expression_sample_type_counts": sample_type_counts,
            "primary_tumour_expression_samples": int(len(primary_expr)),
            "direct_expression_clinical_matches": int(
                direct["matched_clinical"].sum()
            ),
            "locked_patient_level_samples": int(len(clinical_locked)),
            "stage_group_counts": {
                str(k): int(v) for k, v in stage_counts.items()
            },
            "OS_event_count": os_event_count,
            "pack_years_usable": int(smoking.notna().sum()),
            "primary_expression": expr_info,
            "primary_contrast": "Stage I/II versus Stage III/IV",
            "endpoint_inventory": endpoint_rows,
            "important_boundaries": [
                "Only primary-tumour sample code 01 was retained.",
                "One expression sample per patient was locked.",
                "Early = Stage I/II; Late = Stage III/IV.",
                "Smoking variables are contextual covariates, not the primary task.",
                "No GO-BP or endpoint-driven feature selection was performed.",
                "TCGA-LUAD will receive its own BP selection, network and modules.",
            ],
            "next_step": (
                "Stage 2A: GO-BP observation-readiness and LUAD stage "
                "discriminability screening."
            ),
        }
        with (OUT_DIR / "LUAD_STAGE1_SUMMARY.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                summary,
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        (OUT_DIR / "README_LUAD_STAGE1.txt").write_text(
            "MOATTERS — TCGA-LUAD Stage 1\n\n"
            f"Status: {status}\n"
            f"Locked patient-level samples: {len(clinical_locked)}\n"
            f"Early-stage: {stage_counts.get('EARLY', 0)}\n"
            f"Late-stage: {stage_counts.get('LATE', 0)}\n"
            f"OS events: {os_event_count}\n"
            f"Pack-years usable: {int(smoking.notna().sum())}\n"
            f"Expression genes: {expr_info['n_unique_gene_symbols']}\n\n"
            "The LUAD-specific reconstruction has not yet been fitted.\n",
            encoding="utf-8",
        )

        log(f"LUAD Stage 1 completed: {status}", fh)
        log(f"Stage counts: {stage_counts}", fh)
        log(f"OS events: {os_event_count}", fh)
        log(
            f"Expression genes: {expr_info['n_unique_gene_symbols']}",
            fh,
        )


if __name__ == "__main__":
    main()
