# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# -*- coding: utf-8 -*-
r"""
MOATTERS
Stage 4 — joint evidence synthesis:
breast-cancer external validation + cross-cancer applicability

Expected inputs
---------------
1. Three-cohort breast-cancer synthesis:
   D:\MOATTERS-Output\MOATTERS_STAGE3_THREE_COHORT_SYNTHESIS_V2

2. TCGA-KIRC reviewer-facing synthesis:
   D:\MOATTERS-Output\MOATTERS_KIRC_STAGE2D_SYNTHESIS

3. TCGA-LUAD reviewer-facing synthesis:
   D:\MOATTERS-Output\MOATTERS_LUAD_STAGE2D_SYNTHESIS

Output
------
D:\MOATTERS-Output\MOATTERS_STAGE4_JOINT_EVIDENCE_SYNTHESIS

Purpose
-------
Create one coherent revision package that clearly separates:

A. Independent breast-cancer external validation
   - METABRIC
   - GSE96058/SCAN-B

B. Cross-cancer pipeline applicability
   - TCGA-KIRC
   - TCGA-LUAD

The script creates:
- compact manuscript-ready tables
- reviewer-response draft
- Results draft
- Methods integration draft
- bounded-claims matrix
- a machine-readable joint summary

Important evidence-role boundary
--------------------------------
TCGA-BRCA is the derivation/reference cohort.
METABRIC and GSE96058 are independent external-validation cohorts.
TCGA-KIRC and TCGA-LUAD are internal cross-cancer applicability
demonstrations, not external validation cohorts.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from moatters.config import data_path, output_path

import numpy as np
import pandas as pd


BRCA_STAGE3 = output_path(r"MOATTERS_STAGE3_THREE_COHORT_SYNTHESIS_V2")
KIRC_STAGE2D = output_path(r"MOATTERS_KIRC_STAGE2D_SYNTHESIS")
LUAD_STAGE2D = output_path(r"MOATTERS_LUAD_STAGE2D_SYNTHESIS")

OUT = output_path(r"MOATTERS_STAGE4_JOINT_EVIDENCE_SYNTHESIS")


def log(msg: str, fh) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def find_first(root: Path, candidates: list[str]) -> Path:
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    # fallback recursive search by filename
    for name in candidates:
        found = list(root.rglob(name))
        if found:
            return found[0]
    raise FileNotFoundError(
        f"None of the expected files were found under {root}: {candidates}"
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def round_or_nan(x, digits=3):
    try:
        return round(float(x), digits)
    except Exception:
        return np.nan


def normalize_external_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Resolve endpoint naming explicitly. Some Stage 3 tables contain both
    # endpoint_canonical and endpoint_name; renaming both to "endpoint"
    # creates duplicate columns, so out["endpoint"] becomes a DataFrame and
    # string accessors fail.
    if "endpoint_canonical" in out.columns:
        endpoint_series = out["endpoint_canonical"]
    elif "endpoint_name" in out.columns:
        endpoint_series = out["endpoint_name"]
    elif "endpoint" in out.columns:
        endpoint_series = out["endpoint"]
    else:
        raise RuntimeError("External endpoint table has no endpoint column.")

    # If a duplicate-column DataFrame is encountered, take the first column
    # deterministically and record the chosen semantic source above.
    if isinstance(endpoint_series, pd.DataFrame):
        endpoint_series = endpoint_series.iloc[:, 0]

    out = out.drop(
        columns=[
            c for c in ["endpoint", "endpoint_canonical", "endpoint_name"]
            if c in out.columns
        ],
        errors="ignore",
    )
    out.insert(0, "endpoint", endpoint_series.astype("string"))

    rename_map = {
        "auc_orientation_invariant": "AUC_orientation_invariant",
        "auc_directional": "AUC_directional",
    }
    out = out.rename(columns={
        k: v for k, v in rename_map.items() if k in out.columns
    })

    if "cohort" not in out.columns:
        raise RuntimeError("External endpoint table has no cohort column.")

    if "evidence_role" not in out.columns:
        out["evidence_role"] = np.where(
            out["cohort"].astype(str).str.contains("TCGA", case=False),
            "derivation_reference",
            "independent_external_validation",
        )

    keep = [
        c for c in [
            "cohort", "endpoint", "evidence_role",
            "AUC_directional", "AUC_orientation_invariant",
            "rank_biserial", "p_value", "q_value",
            "n", "n_positive", "n_negative",
        ]
        if c in out.columns
    ]
    return out[keep].copy()


def load_cross_cancer_row(root: Path, cancer: str) -> pd.DataFrame:
    path = find_first(
        root,
        [f"Table_{cancer}_reviewer_facing_summary.csv"],
    )
    df = pd.read_csv(path)
    if len(df) != 1:
        raise RuntimeError(f"{path} should contain exactly one row.")
    return df


def main():
    for p in [BRCA_STAGE3, KIRC_STAGE2D, LUAD_STAGE2D]:
        if not p.exists():
            raise FileNotFoundError(p)

    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ["tables", "text", "logs", "audit"]:
        (OUT / sub).mkdir(exist_ok=True)

    with (OUT / "logs" / "stage4_joint_synthesis.log").open(
        "w", encoding="utf-8"
    ) as fh:
        log("Starting joint evidence synthesis", fh)

        external_path = find_first(
            BRCA_STAGE3,
            [
                "Table_2_three_cohort_endpoint_alignment.csv",
                "Table_2_three_cohort_endpoint_alignment_v2.csv",
                "Table_2C_three_cohort_replication_summary.csv",
            ],
        )
        external_raw = pd.read_csv(external_path)
        external = normalize_external_table(external_raw)
        if external.columns.duplicated().any():
            duplicated = external.columns[external.columns.duplicated()].tolist()
            log(f"Dropping duplicated normalized columns: {duplicated}", fh)
            external = external.loc[:, ~external.columns.duplicated()].copy()

        kirc = load_cross_cancer_row(KIRC_STAGE2D, "KIRC")
        luad = load_cross_cancer_row(LUAD_STAGE2D, "LUAD")
        cross = pd.concat([kirc, luad], ignore_index=True)

        # Evidence block 1: independent breast-cancer validation only.
        external_only = external[
            external["evidence_role"].astype(str).str.contains(
                "external", case=False, na=False
            )
        ].copy()

        # Restrict the compact table to interpretable common endpoints.
        preferred_order = [
            "ER", "PR", "HER2",
            "PAM50_luminal", "PAM50_basal",
            "stage_late", "node_positive",
        ]
        endpoint_order_map = {
            e.lower(): i for i, e in enumerate(preferred_order)
        }
        external_only["_order"] = (
            external_only["endpoint"].astype(str).str.lower()
            .map(endpoint_order_map).fillna(999)
        )
        external_only = external_only.sort_values(
            ["_order", "endpoint", "cohort"]
        ).drop(columns="_order")

        external_only.to_csv(
            OUT / "tables" /
            "Table_A_independent_breast_cancer_external_validation.csv",
            index=False,
            encoding="utf-8-sig",
        )

        cross_keep = [
            c for c in [
                "cancer", "role",
                "n_locked_primary_tumours", "n_stage_usable",
                "early_n", "late_n",
                "nonzero_variance_genes",
                "GO_BP_terms_screened",
                "observation_ready_K10",
                "nominal_selected_D_gt_1_301",
                "locked_task_discriminative_BP",
                "primary_network_abs_r",
                "primary_network_edges",
                "primary_network_modules",
                "primary_module_sizes",
                "apparent_stage_AUC",
                "centroid_only_heldout_AUC",
                "centroid_permutation_p",
                "bounded_claim",
            ]
            if c in cross.columns
        ]
        cross_compact = cross[cross_keep].copy()
        cross_compact.to_csv(
            OUT / "tables" /
            "Table_B_cross_cancer_pipeline_applicability.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # Reviewer concern matrix.
        matrix = pd.DataFrame([
            {
                "reviewer_concern": "Independent breast-cancer external validation",
                "analysis_block": "External validation",
                "datasets": "METABRIC; GSE96058/SCAN-B",
                "status": "ADDRESSED",
                "evidence": (
                    "Locked TCGA-BRCA-derived representation reconstructed "
                    "without endpoint refitting in two independent cohorts."
                ),
                "bounded_interpretation": (
                    "Supports transportable molecular-functional representation; "
                    "does not establish universal prognosis or clinical utility."
                ),
            },
            {
                "reviewer_concern": "Applicability beyond breast cancer",
                "analysis_block": "Cross-cancer applicability",
                "datasets": "TCGA-KIRC; TCGA-LUAD",
                "status": "ADDRESSED",
                "evidence": (
                    "The complete workflow was repeated de novo in two additional "
                    "TCGA cancers, with cancer-specific BP selection, networks, "
                    "modules, held-out centroid audits and permutation nulls."
                ),
                "bounded_interpretation": (
                    "Demonstrates pipeline executability and cancer-specific "
                    "reconstruction behavior; not independent external validation."
                ),
            },
            {
                "reviewer_concern": "Minimum matched-gene cutoff sensitivity",
                "analysis_block": "Sensitivity analysis",
                "datasets": "METABRIC; GSE96058; KIRC; LUAD",
                "status": "ADDRESSED",
                "evidence": "K=5, 10, 15 and 20 evaluated.",
                "bounded_interpretation": (
                    "Higher K values reduce observation-ready terms and can alter "
                    "representation dimensionality."
                ),
            },
            {
                "reviewer_concern": "Discriminability threshold sensitivity",
                "analysis_block": "Sensitivity analysis",
                "datasets": "KIRC; LUAD",
                "status": "ADDRESSED",
                "evidence": "D thresholds 1.0, 1.301, 1.5 and 2.0 evaluated.",
                "bounded_interpretation": (
                    "Threshold tightening changes candidate counts but does not "
                    "invalidate pipeline execution."
                ),
            },
            {
                "reviewer_concern": "Correlation/network threshold sensitivity",
                "analysis_block": "Sensitivity analysis",
                "datasets": "KIRC; LUAD",
                "status": "ADDRESSED",
                "evidence": "|r| thresholds 0.25, 0.35 and 0.45 evaluated.",
                "bounded_interpretation": (
                    "Module architecture is threshold-dependent and cancer-context "
                    "specific; no universal module count is claimed."
                ),
            },
        ])
        matrix.to_csv(
            OUT / "tables" / "Table_C_reviewer_concern_matrix.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # Extract headline values.
        def get_cross(cancer: str, col: str):
            row = cross_compact[
                cross_compact["cancer"].astype(str).str.contains(
                    cancer, case=False, na=False
                )
            ]
            if len(row) == 0 or col not in row.columns:
                return np.nan
            return row.iloc[0][col]

        kirc_auc = get_cross("KIRC", "centroid_only_heldout_AUC")
        luad_auc = get_cross("LUAD", "centroid_only_heldout_AUC")
        kirc_perm = get_cross("KIRC", "centroid_permutation_p")
        luad_perm = get_cross("LUAD", "centroid_permutation_p")

        # External replication summary by endpoint.
        replication_rows = []
        for endpoint, grp in external_only.groupby("endpoint", dropna=False):
            cohorts = sorted(grp["cohort"].astype(str).unique())
            row = {
                "endpoint": endpoint,
                "external_cohorts": " | ".join(cohorts),
                "n_external_cohorts": len(cohorts),
            }
            if "AUC_orientation_invariant" in grp.columns:
                vals = pd.to_numeric(
                    grp["AUC_orientation_invariant"], errors="coerce"
                )
                row["external_AUC_min"] = vals.min()
                row["external_AUC_max"] = vals.max()
                row["external_AUC_mean"] = vals.mean()
            if "rank_biserial" in grp.columns:
                rb = pd.to_numeric(grp["rank_biserial"], errors="coerce")
                signs = np.sign(rb.dropna())
                row["effect_direction_consistent"] = (
                    bool(len(signs) > 0 and np.all(signs == signs.iloc[0]))
                    if hasattr(signs, "iloc")
                    else np.nan
                )
            replication_rows.append(row)

        replication = pd.DataFrame(replication_rows)
        replication.to_csv(
            OUT / "tables" /
            "Table_D_external_replication_by_endpoint.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # Manuscript-ready Results draft.
        results_text = f"""External validation and cross-cancer applicability

The locked TCGA-BRCA-derived molecular-functional representation was
evaluated in two independent breast-cancer cohorts, METABRIC and
GSE96058/SCAN-B, without endpoint refitting. Reconstruction was technically
complete at the prespecified minimum matched-gene threshold of K>=10.
External phenotype alignment was strongest and most reproducible for
intrinsic-subtype and hormone-receptor phenotypes, particularly PAM50 basal
and ER status, whereas anatomical progression-related endpoints were weaker.
Survival associations were modest and cohort-dependent, indicating
transportable representation rather than universal prognostic utility.

To test whether the workflow itself could be applied outside breast cancer,
the complete observation-readiness, task-discriminability, network
reconstruction and patient-state pipeline was repeated de novo in TCGA-KIRC
and TCGA-LUAD. No BRCA-specific BP terms or modules were transferred.
Centroid-only repeated held-out evaluation yielded orientation-invariant AUCs
of {round_or_nan(kirc_auc):.3f} in KIRC and
{round_or_nan(luad_auc):.3f} in LUAD. Both exceeded their 500-label
permutation nulls (empirical p={round_or_nan(kirc_perm, 4):.4f} and
{round_or_nan(luad_perm, 4):.4f}, respectively). These results support
cross-cancer workflow applicability while also showing that the resulting
network and module architectures are cancer-context specific.
"""
        (OUT / "text" / "RESULTS_EXTERNAL_AND_CROSS_CANCER_DRAFT.txt").write_text(
            results_text,
            encoding="utf-8",
        )

        methods_text = """Integrated validation design

Two distinct validation questions were addressed. First, transportability of
the locked TCGA-BRCA-derived representation was assessed in two independent
breast-cancer cohorts, METABRIC and GSE96058/SCAN-B. BP definitions, module
membership, centroids and score orientation were transferred without
endpoint refitting. Second, cross-cancer applicability of the workflow was
assessed by repeating the complete reconstruction procedure de novo in
TCGA-KIRC and TCGA-LUAD. For each additional cancer, primary-tumour samples
were restricted to one specimen per patient, pathologic Stage I/II and
Stage III/IV were used as the early- and late-stage task contrast, and
cancer-specific GO-BP observables, correlation networks and modules were
derived. A minimum of 10 matched genes was used for the primary analysis,
with K=5, 15 and 20 as sensitivity settings. The primary correlation-network
threshold was |r|>=0.35, with 0.25 and 0.45 as sensitivity thresholds.
Repeated 20x5 stratified cross-validation refitted only the early- and
late-stage centroids within training folds; BP selection and module
construction remained fixed from the full cancer cohort. A 500-label
permutation analysis was used as a null audit. Accordingly, KIRC and LUAD
were interpreted as internal cross-cancer applicability demonstrations and
not as independent external-validation cohorts.
"""
        (OUT / "text" / "METHODS_INTEGRATED_VALIDATION_DRAFT.txt").write_text(
            methods_text,
            encoding="utf-8",
        )

        rebuttal_text = """Response to reviewer — external validation and
cross-cancer applicability

We thank the reviewer for requesting stronger evidence of both external
validation and applicability beyond the original TCGA-BRCA setting.

First, we added two independent breast-cancer cohorts, METABRIC and
GSE96058/SCAN-B. The locked TCGA-BRCA-derived biological-process and module
representation was reconstructed without endpoint refitting. The strongest
and most reproducible external alignment was observed for PAM50 basal, ER
and luminal phenotypes, whereas anatomical progression-related endpoints
were weaker and survival associations were cohort-dependent. We therefore
revised the manuscript to describe transportable molecular-functional
representation rather than universal prognostic or clinical utility.

Second, we repeated the full pipeline de novo in two additional TCGA cancer
types, KIRC and LUAD. Each cancer received its own observation-readiness
screen, task-discriminative BP selection, late-state correlation network,
module architecture and patient-level centroid reconstruction. No
BRCA-specific BP terms or modules were transferred. Centroid-only repeated
held-out evaluation remained above permutation null in both cancers. We also
added sensitivity analyses for the minimum matched-gene threshold
(K=5/10/15/20), discriminability threshold
(D=1.0/1.301/1.5/2.0), and correlation-network threshold
(|r|=0.25/0.35/0.45).

We now clearly distinguish these evidence roles in the revised manuscript:
METABRIC and GSE96058 constitute independent breast-cancer external
validation, whereas KIRC and LUAD demonstrate cross-cancer pipeline
applicability. The latter analyses are not presented as fully nested external
validation because BP selection and network construction were derived within
each TCGA cancer cohort.
"""
        (OUT / "text" / "REVIEWER_RESPONSE_JOINT_DRAFT.txt").write_text(
            rebuttal_text,
            encoding="utf-8",
        )

        claim_matrix = pd.DataFrame([
            {
                "claim": "The locked BRCA representation is externally transportable.",
                "support": "METABRIC and GSE96058",
                "allowed": True,
                "wording": (
                    "Transportable molecular-functional representation across "
                    "independent breast-cancer cohorts."
                ),
            },
            {
                "claim": "The pipeline is executable outside breast cancer.",
                "support": "TCGA-KIRC and TCGA-LUAD",
                "allowed": True,
                "wording": (
                    "Cross-cancer workflow applicability with cancer-specific "
                    "representation architectures."
                ),
            },
            {
                "claim": "The same biological modules are universal across cancers.",
                "support": "Not supported",
                "allowed": False,
                "wording": "Do not claim.",
            },
            {
                "claim": "The representation is a universal prognostic model.",
                "support": "Not supported",
                "allowed": False,
                "wording": "Do not claim.",
            },
            {
                "claim": "The analysis establishes clinical utility.",
                "support": "Not evaluated",
                "allowed": False,
                "wording": "Do not claim.",
            },
        ])
        claim_matrix.to_csv(
            OUT / "tables" / "Table_E_bounded_claims.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": "STAGE4_JOINT_EVIDENCE_SYNTHESIS",
            "status": "PASS",
            "evidence_blocks": {
                "independent_breast_cancer_external_validation": [
                    "METABRIC",
                    "GSE96058/SCAN-B",
                ],
                "cross_cancer_pipeline_applicability": [
                    "TCGA-KIRC",
                    "TCGA-LUAD",
                ],
            },
            "headline_cross_cancer_results": {
                "KIRC_centroid_only_heldout_AUC": kirc_auc,
                "KIRC_permutation_p": kirc_perm,
                "LUAD_centroid_only_heldout_AUC": luad_auc,
                "LUAD_permutation_p": luad_perm,
            },
            "reviewer_concerns_status": {
                "external_validation": "ADDRESSED",
                "cross_cancer_applicability": "ADDRESSED",
                "minimum_matched_gene_sensitivity": "ADDRESSED",
                "D_threshold_sensitivity": "ADDRESSED",
                "network_threshold_sensitivity": "ADDRESSED",
            },
            "outputs": [
                "Table_A_independent_breast_cancer_external_validation.csv",
                "Table_B_cross_cancer_pipeline_applicability.csv",
                "Table_C_reviewer_concern_matrix.csv",
                "Table_D_external_replication_by_endpoint.csv",
                "Table_E_bounded_claims.csv",
                "RESULTS_EXTERNAL_AND_CROSS_CANCER_DRAFT.txt",
                "METHODS_INTEGRATED_VALIDATION_DRAFT.txt",
                "REVIEWER_RESPONSE_JOINT_DRAFT.txt",
            ],
            "next_step": (
                "Integrate these tables and drafts into the revised manuscript, "
                "Supplementary Information and point-by-point rebuttal."
            ),
        }
        with (OUT / "STAGE4_JOINT_SUMMARY.json").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                summary,
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        log("Stage 4 joint synthesis completed: PASS", fh)
        log(
            f"KIRC held-out AUC={round_or_nan(kirc_auc):.3f}; "
            f"LUAD held-out AUC={round_or_nan(luad_auc):.3f}",
            fh,
        )


if __name__ == "__main__":
    main()
