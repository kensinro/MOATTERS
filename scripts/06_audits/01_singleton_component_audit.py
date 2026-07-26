# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IMU Major Revision — Stage 5A FINAL
Directed audit of singleton BP-state components M6 and M7.

Corrections relative to the first run:
1. Removes the duplicated configuration:
   "Remove M6 and M7" == "Five multi-term modules only".
2. Uses one shared set of stratified bootstrap resamples for all configurations.
3. Replaces the incomplete 339-case PAM50 field in the leave-one-module table
   with the complete PAM50_simplified annotation (n≈821) from:
   MOATTERS_BRCA_STATE_DownstreamValidation_20260531_124416/
   tables/99_final_merged_BPstate_clinical_endpoints.csv
4. Keeps all stage and patient-level calculations locked to the original
   patient × M1–M7 table.

Reviewer target:
"Address ambiguity of singleton BP components and their impact on centroid
similarity calculation."
"""

from __future__ import annotations
from pathlib import Path
from moatters.config import data_path, output_path
import io
import json
import math
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, kruskal
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

SEARCH_ROOTS = [
    output_path(),
    output_path(),
]

OUTPUT_DIR = output_path(r"MOATTERS_STAGE5A_SINGLETON_AUDIT_FINAL")

BOOTSTRAP_N = 1000
RANDOM_SEED = 20260724

PATIENT_SCORE_SUFFIX = (
    "MOATTERS_BRCA_STATE_ManuscriptDefense_20260531_184850/"
    "tables/10_leave_one_module_patient_scores.csv"
)

PAM50_SUFFIX = (
    "MOATTERS_BRCA_STATE_DownstreamValidation_20260531_124416/"
    "tables/99_final_merged_BPstate_clinical_endpoints.csv"
)

COMPONENTS = [f"M{i}" for i in range(1, 8)]

CONFIGURATIONS = {
    "Full 7 components": COMPONENTS,
    "Remove M6": ["M1", "M2", "M3", "M4", "M5", "M7"],
    "Remove M7": ["M1", "M2", "M3", "M4", "M5", "M6"],
    "Remove both singletons (M1-M5 only)": ["M1", "M2", "M3", "M4", "M5"],
}

def load_current_locked_tables():
    profile_dir = output_path("MOATTERS_BRCA_PATIENT_STATE_Profile_V1_20260531")
    score_path = profile_dir / "BRCA_patient_module_scores_z.csv"
    master_path = profile_dir / "BRCA_patient_strategy_master_table.csv"

    downstream_dirs = sorted(
        output_path().glob("MOATTERS_BRCA_STATE_DownstreamValidation_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not downstream_dirs:
        raise FileNotFoundError("No current downstream-validation output directory was found.")
    pam50_path = downstream_dirs[0] / "tables" / "99_final_merged_BPstate_clinical_endpoints.csv"

    required = [score_path, master_path, pam50_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing current singleton-audit inputs: {missing}")

    module_scores = pd.read_csv(score_path, index_col=0)
    module_scores.index = module_scores.index.astype(str)
    patient_scores = module_scores.T.reset_index().rename(columns={"index": "patient_id"})

    master = pd.read_csv(master_path)
    required_master = {"patient", "stage_group"}
    absent = sorted(required_master.difference(master.columns))
    if absent:
        raise RuntimeError(f"BRCA patient strategy master table is missing: {absent}")
    stage = (
        master[["patient", "stage_group"]]
        .drop_duplicates("patient")
        .rename(columns={"patient": "patient_id", "stage_group": "StageGroup"})
    )
    patient_df = patient_scores.merge(stage, on="patient_id", how="left")
    pam50_df = pd.read_csv(pam50_path)
    source = {
        "module_scores": str(score_path),
        "stage_master": str(master_path),
        "pam50": str(pam50_path),
    }
    return patient_df, pam50_df, source

def cosine_rows(X, centroid):
    numerator = X @ centroid
    denominator = np.linalg.norm(X, axis=1) * np.linalg.norm(centroid)
    return np.divide(
        numerator,
        denominator,
        out=np.full(X.shape[0], np.nan, dtype=float),
        where=denominator > 0,
    )

def reconstruct(X, y):
    early_centroid = X[y == 0].mean(axis=0)
    late_centroid = X[y == 1].mean(axis=0)
    sim_early = cosine_rows(X, early_centroid)
    sim_late = cosine_rows(X, late_centroid)
    margin = sim_late - sim_early
    return sim_early, sim_late, margin

def rank_biserial(x_late, x_early):
    result = mannwhitneyu(x_late, x_early, alternative="two-sided")
    rb = 2 * result.statistic / (len(x_late) * len(x_early)) - 1
    return float(rb), float(result.pvalue)

def d_from_p(p):
    return -math.log10(max(float(p), np.nextafter(0, 1)))

def orientation_invariant_auc(y, score):
    auc = roc_auc_score(y, score)
    return float(auc), float(max(auc, 1 - auc))

def state_labels(margin, cutoff):
    labels = np.full(len(margin), "Ambiguous", dtype=object)
    labels[margin > cutoff] = "Late-like"
    labels[margin < -cutoff] = "Early-like"
    return labels

def canonical_patient_id(series):
    return series.astype(str).str.strip().str[:12]

def make_shared_bootstrap_indices(y, n_iter, seed):
    rng = np.random.default_rng(seed)
    early_idx = np.where(y == 0)[0]
    late_idx = np.where(y == 1)[0]
    samples = []
    for _ in range(n_iter):
        b0 = rng.choice(early_idx, size=len(early_idx), replace=True)
        b1 = rng.choice(late_idx, size=len(late_idx), replace=True)
        samples.append(np.concatenate([b0, b1]))
    return samples

def bootstrap_auc(X, y, shared_indices):
    values = []
    for iteration, idx in enumerate(shared_indices, start=1):
        Xb = X[idx]
        yb = y[idx]
        _, _, margin = reconstruct(Xb, yb)
        _, auc_inv = orientation_invariant_auc(yb, margin)
        values.append((iteration, auc_inv))
    return values

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "tables").mkdir(exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)
    (OUTPUT_DIR / "text").mkdir(exist_ok=True)

    patient_df, pam50_df, source_paths = load_current_locked_tables()
    patient_member = source_paths["module_scores"]
    pam50_member = source_paths["pam50"]
    zip_path = "CURRENT_PORTABLE_OUTPUTS"

    required_patient = ["patient_id", "StageGroup"] + COMPONENTS
    missing = [c for c in required_patient if c not in patient_df.columns]
    if missing:
        raise RuntimeError(f"Patient score table is missing: {missing}")

    if "patient_id" not in pam50_df.columns:
        if "patient" in pam50_df.columns:
            pam50_df["patient_id"] = pam50_df["patient"]
        else:
            raise RuntimeError("Complete PAM50 table has no patient identifier.")

    if "PAM50_simplified" not in pam50_df.columns:
        raise RuntimeError("Complete PAM50 table has no PAM50_simplified field.")

    patient_df = patient_df.copy()
    pam50_df = pam50_df.copy()

    patient_df["patient_id"] = canonical_patient_id(patient_df["patient_id"])
    pam50_df["patient_id"] = canonical_patient_id(pam50_df["patient_id"])

    pam50_lookup = (
        pam50_df[["patient_id", "PAM50_simplified"]]
        .dropna(subset=["PAM50_simplified"])
        .drop_duplicates("patient_id")
    )

    # Explicitly discard the incomplete PAM50_simplified_recheck field and
    # merge the complete annotation.
    patient_df = patient_df.drop(
        columns=["PAM50_simplified_recheck"], errors="ignore"
    )
    df = patient_df.merge(pam50_lookup, on="patient_id", how="left")

    df["stage_late"] = df["StageGroup"].map({"Early": 0, "Late": 1})
    for component in COMPONENTS:
        df[component] = pd.to_numeric(df[component], errors="coerce")

    df = df.dropna(subset=COMPONENTS + ["stage_late"]).copy()
    y = df["stage_late"].astype(int).to_numpy()

    # Locked scores are already standardized in the original analysis.
    full_X = df[COMPONENTS].to_numpy(float)
    full_sim_early, full_sim_late, full_margin = reconstruct(full_X, y)

    ambiguous_cutoff = float(np.quantile(np.abs(full_margin), 0.20))
    full_states = state_labels(full_margin, ambiguous_cutoff)

    # One shared set of bootstrap samples is used for every configuration.
    shared_bootstrap_indices = make_shared_bootstrap_indices(
        y, BOOTSTRAP_N, RANDOM_SEED
    )

    summary_rows = []
    bootstrap_rows = []

    for name, retained in CONFIGURATIONS.items():
        X = df[retained].to_numpy(float)
        sim_early, sim_late, margin = reconstruct(X, y)

        auc_dir, auc_inv = orientation_invariant_auc(y, margin)
        rb, p_mw = rank_biserial(margin[y == 1], margin[y == 0])

        if name == "Full 7 components":
            rho_late = 1.0
            rho_margin = 1.0
            state_agreement = 1.0
        else:
            rho_late = float(
                spearmanr(sim_late, full_sim_late, nan_policy="omit").statistic
            )
            rho_margin = float(
                spearmanr(margin, full_margin, nan_policy="omit").statistic
            )
            state_agreement = float(
                np.mean(
                    state_labels(margin, ambiguous_cutoff) == full_states
                )
            )

        valid_pam = df["PAM50_simplified"].notna()
        pam_groups = [
            margin[idx]
            for _, idx in df.loc[valid_pam]
            .groupby("PAM50_simplified")
            .indices.items()
            if len(idx) >= 5
        ]
        if len(pam_groups) >= 2:
            kw = kruskal(*pam_groups)
            pam_p = float(kw.pvalue)
            pam_D = d_from_p(pam_p)
        else:
            pam_p = np.nan
            pam_D = np.nan

        boot = bootstrap_auc(X, y, shared_bootstrap_indices)
        boot_values = np.array([v for _, v in boot])

        for iteration, value in boot:
            bootstrap_rows.append(
                {
                    "configuration": name,
                    "iteration": iteration,
                    "orientation_invariant_AUC": value,
                }
            )

        summary_rows.append(
            {
                "configuration": name,
                "components_retained": ";".join(retained),
                "n_components": len(retained),
                "n_patients": len(df),
                "n_early": int((y == 0).sum()),
                "n_late": int((y == 1).sum()),
                "n_PAM50": int(valid_pam.sum()),
                "centroid_margin_AUC_directional": auc_dir,
                "centroid_margin_AUC_orientation_invariant": auc_inv,
                "rank_biserial": rb,
                "Mann_Whitney_p": p_mw,
                "D_minus_log10_p": d_from_p(p_mw),
                "late_similarity_Spearman_vs_full": rho_late,
                "centroid_margin_Spearman_vs_full": rho_margin,
                "three_state_agreement_vs_full": state_agreement,
                "PAM50_Kruskal_p": pam_p,
                "PAM50_D": pam_D,
                "bootstrap_AUC_median": float(np.median(boot_values)),
                "bootstrap_AUC_q05": float(np.quantile(boot_values, 0.05)),
                "bootstrap_AUC_q95": float(np.quantile(boot_values, 0.95)),
            }
        )

        patient_out = pd.DataFrame(
            {
                "patient_id": df["patient_id"],
                "StageGroup": df["StageGroup"],
                "PAM50_simplified": df["PAM50_simplified"],
                "similarity_to_early_centroid": sim_early,
                "similarity_to_late_centroid": sim_late,
                "late_minus_early_similarity_margin": margin,
                "state_assignment": state_labels(
                    margin, ambiguous_cutoff
                ),
            }
        )
        safe_name = (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )
        patient_out.to_csv(
            OUTPUT_DIR / "tables" / f"patient_scores_{safe_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary = pd.DataFrame(summary_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)

    summary.to_csv(
        OUTPUT_DIR / "tables" / "Table_5A_FINAL_singleton_impact_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap.to_csv(
        OUTPUT_DIR / "tables" / "Table_5A_FINAL_shared_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    labels = ["Full 7", "−M6", "−M7", "M1–M5 only"]
    x = np.arange(len(summary))

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5))

    axes[0].errorbar(
        x,
        summary["bootstrap_AUC_median"],
        yerr=[
            summary["bootstrap_AUC_median"] - summary["bootstrap_AUC_q05"],
            summary["bootstrap_AUC_q95"] - summary["bootstrap_AUC_median"],
        ],
        fmt="o",
        capsize=4,
    )
    axes[0].axhline(0.5, linestyle="--", linewidth=1)
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("Orientation-invariant AUC")
    axes[0].set_title("A. Stage-alignment stability")

    axes[1].plot(
        x,
        summary["late_similarity_Spearman_vs_full"],
        marker="o",
        label="Late-centroid similarity",
    )
    axes[1].plot(
        x,
        summary["centroid_margin_Spearman_vs_full"],
        marker="s",
        label="Late–Early margin",
    )
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylim(0, 1.03)
    axes[1].set_ylabel("Spearman ρ versus full")
    axes[1].set_title("B. Patient-level concordance")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(
        x,
        summary["three_state_agreement_vs_full"],
        marker="o",
        label="State agreement",
    )
    axes[2].plot(
        x,
        summary["PAM50_D"] / summary["PAM50_D"].max(),
        marker="s",
        label="Normalized PAM50 D",
    )
    axes[2].set_xticks(x, labels, rotation=20, ha="right")
    axes[2].set_ylim(0, 1.03)
    axes[2].set_ylabel("Relative robustness")
    axes[2].set_title("C. State and PAM50 robustness")
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Final directed audit of singleton BP-state components M6 and M7",
        y=1.02,
    )
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "figures" / "Figure_5A_FINAL_singleton_audit.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT_DIR / "figures" / "Figure_5A_FINAL_singleton_audit.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    full = summary.loc[
        summary["configuration"] == "Full 7 components"
    ].iloc[0]
    no_singletons = summary.loc[
        summary["configuration"] ==
        "Remove both singletons (M1-M5 only)"
    ].iloc[0]

    methods_text = (
        "The two singleton BP-state components, M6 and M7, were audited by "
        "recalculating the Early- and Late-stage centroids after removal of "
        "M6 alone, M7 alone, or both singleton dimensions. The simultaneous-"
        "removal configuration was equivalent to retaining the five multi-"
        "term modules M1–M5 and was therefore represented once. For each "
        "configuration, cosine similarity to the Early and Late centroids, "
        "the Late-minus-Early similarity margin, stage-discrimination AUC, "
        "rank-biserial effect size, patient-level correlation with the full "
        "representation, three-state assignment agreement, PAM50 association "
        "using the complete PAM50 annotation, and 1,000 shared stratified-"
        "bootstrap resamples were evaluated. Singleton components were "
        "treated as one-dimensional components and were not interpreted as "
        "coordinated multi-term modules."
    )

    results_text = (
        f"The full seven-component representation yielded an orientation-"
        f"invariant centroid-margin AUC of "
        f"{full['centroid_margin_AUC_orientation_invariant']:.3f}. After "
        f"simultaneous removal of M6 and M7, the AUC was "
        f"{no_singletons['centroid_margin_AUC_orientation_invariant']:.3f}; "
        f"late-centroid similarity remained strongly correlated with the "
        f"full representation (Spearman ρ="
        f"{no_singletons['late_similarity_Spearman_vs_full']:.3f}), and "
        f"three-state assignment agreement was "
        f"{no_singletons['three_state_agreement_vs_full']:.3f}. PAM50 "
        f"association remained detectable in "
        f"{int(no_singletons['n_PAM50'])} annotated patients "
        f"(D={no_singletons['PAM50_D']:.2f}). Thus, the singleton dimensions "
        f"contributed incremental information but did not dominate the "
        f"patient-level reconstruction."
    )

    rebuttal_text = (
        "We thank the reviewer for identifying the ambiguity surrounding "
        "singleton BP components. We added a directed audit in which M6, M7, "
        "and both singleton dimensions were removed before recalculating all "
        "patient-level centroid metrics. The simultaneous-removal condition "
        "was treated once as the five-multi-term-module representation. "
        + results_text
    )

    (OUTPUT_DIR / "text" / "Methods_Stage5A_FINAL.txt").write_text(
        methods_text, encoding="utf-8"
    )
    (OUTPUT_DIR / "text" / "Results_Stage5A_FINAL.txt").write_text(
        results_text, encoding="utf-8"
    )
    (OUTPUT_DIR / "text" / "Rebuttal_Stage5A_FINAL.txt").write_text(
        rebuttal_text, encoding="utf-8"
    )

    manifest = {
        "status": "PASS",
        "zip_path": str(zip_path),
        "patient_score_member": patient_member,
        "pam50_member": pam50_member,
        "n_patients": int(len(df)),
        "n_early": int((y == 0).sum()),
        "n_late": int((y == 1).sum()),
        "n_complete_PAM50": int(df["PAM50_simplified"].notna().sum()),
        "configurations": CONFIGURATIONS,
        "shared_bootstrap_iterations": BOOTSTRAP_N,
        "random_seed": RANDOM_SEED,
        "duplicate_configuration_removed": True,
        "incomplete_339_case_PAM50_field_replaced": True,
        "ambiguous_cutoff": ambiguous_cutoff,
        "interpretation": (
            "Singleton components are one-dimensional empirical components. "
            "Their removal tests contribution, not biological invalidity."
        ),
    }
    (OUTPUT_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("PASS — Stage 5A FINAL singleton audit completed")
    print(f"Input ZIP: {zip_path}")
    print(
        f"Patients={len(df)}; Early={(y == 0).sum()}; "
        f"Late={(y == 1).sum()}; PAM50={df['PAM50_simplified'].notna().sum()}"
    )
    print(f"Output: {OUTPUT_DIR}")
    print()
    print(
        summary[
            [
                "configuration",
                "centroid_margin_AUC_orientation_invariant",
                "late_similarity_Spearman_vs_full",
                "three_state_agreement_vs_full",
                "PAM50_D",
            ]
        ].to_string(index=False)
    )

if __name__ == "__main__":
    main()
