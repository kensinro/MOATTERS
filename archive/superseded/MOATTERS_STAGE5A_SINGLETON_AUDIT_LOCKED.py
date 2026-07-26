# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IMU Major Revision — Stage 5A
Directed audit of singleton BP-state components M6 and M7.

Reviewer target:
"Address ambiguity of singleton BP components and their impact on centroid
similarity calculation."

Locked source file:
MOATTERS_BRCA_STATE_ManuscriptDefense_20260531_184850/
tables/10_leave_one_module_patient_scores.csv

The script can read this CSV either from an extracted folder or directly from
MOATTERS_BRCA_STATE.zip.
"""

from __future__ import annotations
from pathlib import Path
import io
import json
import math
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, kruskal
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# --------------------------- USER PATHS ---------------------------------
SEARCH_ROOTS = [
    Path(r"D:\MOATTERS-Output"),
    Path(r"D:\MOATTERS-Output"),
]
OUTPUT_DIR = Path(
    r"D:\MOATTERS-Output\MOATTERS_STAGE5A_SINGLETON_AUDIT"
)
BOOTSTRAP_N = 1000
RANDOM_SEED = 20260724
# -----------------------------------------------------------------------

TARGET_SUFFIX = (
    "MOATTERS_BRCA_STATE_ManuscriptDefense_20260531_184850/"
    "tables/10_leave_one_module_patient_scores.csv"
)

COMPONENTS = [f"M{i}" for i in range(1, 8)]
CONFIGURATIONS = {
    "Full 7 components": COMPONENTS,
    "Remove M6": ["M1", "M2", "M3", "M4", "M5", "M7"],
    "Remove M7": ["M1", "M2", "M3", "M4", "M5", "M6"],
    "Remove M6 and M7": ["M1", "M2", "M3", "M4", "M5"],
    "Five multi-term modules only": ["M1", "M2", "M3", "M4", "M5"],
}

def locate_locked_input():
    # 1. Extracted directory.
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("10_leave_one_module_patient_scores.csv"):
            if "MOATTERS_BRCA_STATE_ManuscriptDefense" in str(p):
                return "csv", p, None

    # 2. ZIP file, including names such as MOATTERS_BRCA_STATE(1).zip.
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for zp in root.glob("MOATTERS_BRCA_STATE*.zip"):
            try:
                with zipfile.ZipFile(zp) as zf:
                    members = [
                        n for n in zf.namelist()
                        if n.replace("\\", "/").endswith(TARGET_SUFFIX)
                    ]
                    if members:
                        return "zip", zp, members[0]
            except zipfile.BadZipFile:
                continue

    raise FileNotFoundError(
        "Locked input was not found. Place MOATTERS_BRCA_STATE.zip in "
        r"D:\MOATTERS-Output or extract it there."
    )

def read_locked_input():
    mode, path, member = locate_locked_input()
    if mode == "csv":
        df = pd.read_csv(path)
        source = str(path)
    else:
        with zipfile.ZipFile(path) as zf:
            df = pd.read_csv(io.BytesIO(zf.read(member)))
        source = f"{path}::{member}"
    return df, source

def cosine_similarity_rows(X, centroid):
    numerator = X @ centroid
    denominator = np.linalg.norm(X, axis=1) * np.linalg.norm(centroid)
    return np.divide(
        numerator,
        denominator,
        out=np.full(X.shape[0], np.nan, dtype=float),
        where=denominator > 0,
    )

def reconstruct_centroid_metrics(X, y):
    early_centroid = X[y == 0].mean(axis=0)
    late_centroid = X[y == 1].mean(axis=0)

    sim_early = cosine_similarity_rows(X, early_centroid)
    sim_late = cosine_similarity_rows(X, late_centroid)
    margin = sim_late - sim_early

    # Three-state assignment makes the audit more faithful than a forced
    # binary label. The ambiguous band is fixed from the full representation
    # and applied unchanged to reduced representations.
    return sim_early, sim_late, margin

def orientation_invariant_auc(y, score):
    auc = roc_auc_score(y, score)
    return auc, max(auc, 1.0 - auc)

def rank_biserial(x_late, x_early):
    u = mannwhitneyu(x_late, x_early, alternative="two-sided")
    rb = 2.0 * u.statistic / (len(x_late) * len(x_early)) - 1.0
    return float(rb), float(u.pvalue)

def D_from_p(p):
    return -math.log10(max(float(p), np.nextafter(0, 1)))

def state_labels(margin, ambiguous_cutoff):
    out = np.full(len(margin), "Ambiguous", dtype=object)
    out[margin > ambiguous_cutoff] = "Late-like"
    out[margin < -ambiguous_cutoff] = "Early-like"
    return out

def stratified_bootstrap_auc(X, y, rng, n_iter):
    early_idx = np.where(y == 0)[0]
    late_idx = np.where(y == 1)[0]
    values = []
    for iteration in range(1, n_iter + 1):
        b_early = rng.choice(early_idx, size=len(early_idx), replace=True)
        b_late = rng.choice(late_idx, size=len(late_idx), replace=True)
        idx = np.concatenate([b_early, b_late])
        Xb = X[idx]
        yb = y[idx]
        _, _, margin = reconstruct_centroid_metrics(Xb, yb)
        _, auc_inv = orientation_invariant_auc(yb, margin)
        values.append((iteration, auc_inv))
    return values

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "tables").mkdir(exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)
    (OUTPUT_DIR / "text").mkdir(exist_ok=True)

    df, source = read_locked_input()

    required = ["patient_id", "StageGroup", "PAM50_simplified_recheck"] + COMPONENTS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Locked table is missing columns: {missing}")

    df = df.copy()
    df["stage_late"] = df["StageGroup"].map({"Early": 0, "Late": 1})
    for component in COMPONENTS:
        df[component] = pd.to_numeric(df[component], errors="coerce")

    df = df.dropna(subset=COMPONENTS + ["stage_late"]).copy()
    y = df["stage_late"].astype(int).to_numpy()

    # The locked M1–M7 columns are already standardized (mean≈0, SD≈1).
    # Re-standardization is intentionally avoided.
    full_X = df[COMPONENTS].to_numpy(float)
    full_sim_early, full_sim_late, full_margin = reconstruct_centroid_metrics(
        full_X, y
    )

    # Define the ambiguous band once from the full representation:
    # the central 20% of |margin| values are marked ambiguous. This cutoff is
    # not used for AUC or continuous-effect statistics.
    ambiguous_cutoff = float(np.quantile(np.abs(full_margin), 0.20))
    full_states = state_labels(full_margin, ambiguous_cutoff)

    rng = np.random.default_rng(RANDOM_SEED)
    summary_rows = []
    bootstrap_rows = []
    patient_tables = {}

    for name, retained in CONFIGURATIONS.items():
        X = df[retained].to_numpy(float)
        sim_early, sim_late, margin = reconstruct_centroid_metrics(X, y)

        auc_directional, auc_invariant = orientation_invariant_auc(y, margin)
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
            states = state_labels(margin, ambiguous_cutoff)
            state_agreement = float(np.mean(states == full_states))

        pam50 = df["PAM50_simplified_recheck"]
        valid_pam = pam50.notna()
        pam_groups = []
        for _, idx in df.loc[valid_pam].groupby(
            "PAM50_simplified_recheck"
        ).groups.items():
            values = margin[df.index.get_indexer(idx)]
            if len(values) >= 5:
                pam_groups.append(values)
        if len(pam_groups) >= 2:
            kw = kruskal(*pam_groups)
            pam50_p = float(kw.pvalue)
            pam50_D = D_from_p(pam50_p)
        else:
            pam50_p = np.nan
            pam50_D = np.nan

        boot = stratified_bootstrap_auc(
            X, y, rng, BOOTSTRAP_N
        )
        for iteration, value in boot:
            bootstrap_rows.append(
                {
                    "configuration": name,
                    "iteration": iteration,
                    "orientation_invariant_AUC": value,
                }
            )
        boot_values = np.array([v for _, v in boot])

        states = state_labels(margin, ambiguous_cutoff)
        summary_rows.append(
            {
                "configuration": name,
                "components_retained": ";".join(retained),
                "n_components": len(retained),
                "n_patients": len(df),
                "n_early": int((y == 0).sum()),
                "n_late": int((y == 1).sum()),
                "centroid_margin_AUC_directional": auc_directional,
                "centroid_margin_AUC_orientation_invariant": auc_invariant,
                "rank_biserial": rb,
                "Mann_Whitney_p": p_mw,
                "D_minus_log10_p": D_from_p(p_mw),
                "late_similarity_Spearman_vs_full": rho_late,
                "centroid_margin_Spearman_vs_full": rho_margin,
                "three_state_agreement_vs_full": state_agreement,
                "PAM50_n": int(valid_pam.sum()),
                "PAM50_Kruskal_p": pam50_p,
                "PAM50_D": pam50_D,
                "bootstrap_AUC_median": float(np.median(boot_values)),
                "bootstrap_AUC_q05": float(np.quantile(boot_values, 0.05)),
                "bootstrap_AUC_q95": float(np.quantile(boot_values, 0.95)),
            }
        )

        patient_tables[name] = pd.DataFrame(
            {
                "patient_id": df["patient_id"].astype(str),
                "StageGroup": df["StageGroup"],
                "PAM50_simplified_recheck": df[
                    "PAM50_simplified_recheck"
                ],
                "similarity_to_early_centroid": sim_early,
                "similarity_to_late_centroid": sim_late,
                "late_minus_early_similarity_margin": margin,
                "state_assignment": states,
            }
        )

    summary = pd.DataFrame(summary_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)

    summary.to_csv(
        OUTPUT_DIR / "tables" / "Table_5A_singleton_impact_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap.to_csv(
        OUTPUT_DIR / "tables" / "Table_5A_singleton_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    for name, table in patient_tables.items():
        safe = (
            name.lower()
            .replace(" ", "_")
            .replace("+", "plus")
            .replace("/", "_")
        )
        table.to_csv(
            OUTPUT_DIR / "tables" / f"patient_scores_{safe}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # Reviewer-facing 3-panel figure.
    labels = ["Full 7", "−M6", "−M7", "−M6/−M7", "5 multi-term"]
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
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
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
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].set_ylim(0, 1.03)
    axes[1].set_ylabel("Spearman ρ versus full")
    axes[1].set_title("B. Patient-level concordance")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(
        x,
        summary["three_state_agreement_vs_full"],
        marker="o",
    )
    axes[2].set_xticks(x, labels, rotation=25, ha="right")
    axes[2].set_ylim(0, 1.03)
    axes[2].set_ylabel("Three-state agreement")
    axes[2].set_title("C. State-assignment robustness")

    fig.suptitle(
        "Directed audit of BRCA singleton BP-state components M6 and M7",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "figures" / "Figure_5A_singleton_component_audit.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT_DIR / "figures" / "Figure_5A_singleton_component_audit.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    full = summary.iloc[0]
    no_singletons = summary.loc[
        summary["configuration"] == "Remove M6 and M7"
    ].iloc[0]

    methods = (
        "The two singleton BP-state components, M6 and M7, were audited by "
        "recalculating Early- and Late-stage centroids after removal of M6 "
        "alone, M7 alone, or both singleton dimensions. A five-multi-term-"
        "module representation was evaluated identically. For each "
        "configuration, cosine similarity to the Early and Late centroids, "
        "the Late-minus-Early similarity margin, orientation-invariant AUC, "
        "rank-biserial effect size, PAM50 association, patient-level "
        "correlation with the full representation, three-state assignment "
        "agreement, and 1,000-iteration stratified-bootstrap stability were "
        "quantified. Singleton components were retained as one-dimensional "
        "components and were not interpreted as coordinated multi-term "
        "modules."
    )
    results = (
        f"The full seven-component representation yielded an "
        f"orientation-invariant centroid-margin AUC of "
        f"{full['centroid_margin_AUC_orientation_invariant']:.3f}. "
        f"After simultaneous removal of M6 and M7, the corresponding AUC was "
        f"{no_singletons['centroid_margin_AUC_orientation_invariant']:.3f}; "
        f"late-centroid similarity remained correlated with the full "
        f"representation (Spearman ρ="
        f"{no_singletons['late_similarity_Spearman_vs_full']:.3f}), and "
        f"three-state assignment agreement was "
        f"{no_singletons['three_state_agreement_vs_full']:.3f}. "
        "These analyses directly quantify the effect of the singleton "
        "dimensions without assigning them multi-term mechanistic meaning."
    )
    rebuttal = (
        "We thank the reviewer for identifying the ambiguity surrounding "
        "singleton BP components. We added a directed audit in which M6, M7, "
        "and both singleton dimensions were removed before recalculating the "
        "patient-level centroid metrics. " + results
    )

    (OUTPUT_DIR / "text" / "Methods_Stage5A.txt").write_text(
        methods, encoding="utf-8"
    )
    (OUTPUT_DIR / "text" / "Results_Stage5A.txt").write_text(
        results, encoding="utf-8"
    )
    (OUTPUT_DIR / "text" / "Rebuttal_Stage5A.txt").write_text(
        rebuttal, encoding="utf-8"
    )

    manifest = {
        "status": "PASS",
        "locked_input": source,
        "n_patients": int(len(df)),
        "n_early": int((y == 0).sum()),
        "n_late": int((y == 1).sum()),
        "n_PAM50": int(df["PAM50_simplified_recheck"].notna().sum()),
        "components": COMPONENTS,
        "singleton_components": ["M6", "M7"],
        "ambiguous_cutoff_definition": (
            "20th percentile of absolute full-representation centroid margin; "
            "used only for three-state agreement, not for AUC/effect tests"
        ),
        "bootstrap_iterations": BOOTSTRAP_N,
        "random_seed": RANDOM_SEED,
    }
    (OUTPUT_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("PASS — Stage 5A singleton-component audit completed")
    print(f"Locked input: {source}")
    print(f"Patients: {len(df)}; Early={(y == 0).sum()}; Late={(y == 1).sum()}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    print(summary[
        [
            "configuration",
            "centroid_margin_AUC_orientation_invariant",
            "late_similarity_Spearman_vs_full",
            "three_state_agreement_vs_full",
            "PAM50_D",
        ]
    ].to_string(index=False))

if __name__ == "__main__":
    main()
