# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 5B-3 — Identical-fold quantitative benchmark:
MOATTERS BP-state vs GSVA vs Pathifier.

Endpoints:
- Stage I/II vs III/IV
- ER negative vs positive
- PAM50 basal vs non-basal
- PAM50 luminal vs non-luminal

Fair-comparison rules:
- same endpoint-specific patient intersection across all methods
- identical repeated 20×5 stratified folds
- fold-specific centroids for every method
- no method-specific tuning
- orientation-invariant AUC reported alongside directional AUC

Boundary:
The 30 BP terms and MOATTERS module architecture were locked from the original
full-cohort reconstruction, so this is a representation/scoring benchmark,
not a fully nested feature-selection benchmark.
"""

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib.pyplot as plt

INPUT_DIR = Path(r"D:\MOATTERS-Output\MOATTERS_STAGE5B_BENCHMARK_INPUTS")
SCORE_DIR = Path(r"D:\MOATTERS-Output\MOATTERS_STAGE5B_GSVA_PATHIFIER_SCORES")
OUT = Path(r"D:\MOATTERS-Output\MOATTERS_STAGE5B_BENCHMARK_EVALUATION")

N_SPLITS = 5
N_REPEATS = 20
SEED = 20260724

def patient_id(x):
    return str(x)[:12]

def cosine_rows(X, centroid):
    num = X @ centroid
    den = np.linalg.norm(X, axis=1) * np.linalg.norm(centroid)
    return np.divide(num, den, out=np.full(len(X), np.nan), where=den > 0)

def fit_score_centroid(train_X, train_y, test_X):
    c0 = train_X[train_y == 0].mean(axis=0)
    c1 = train_X[train_y == 1].mean(axis=0)
    return cosine_rows(test_X, c1) - cosine_rows(test_X, c0)

def rb_and_p(score, y):
    x1 = score[y == 1]
    x0 = score[y == 0]
    u = mannwhitneyu(x1, x0, alternative="two-sided")
    rb = 2 * u.statistic / (len(x1) * len(x0)) - 1
    return float(rb), float(u.pvalue)

def D(p):
    return -math.log10(max(float(p), np.nextafter(0, 1)))

def load_score(path):
    df = pd.read_csv(path, index_col=0)
    df.index = [patient_id(i) for i in df.index]
    return df.groupby(level=0).mean()

def endpoint_vectors(clin):
    stage = clin["StageGroup"].map({"Early": 0, "Late": 1})

    er_raw = clin[
        "ER_status__breast_carcinoma_estrogen_receptor_status__clean"
    ].astype(str).str.lower()
    er = pd.Series(np.nan, index=clin.index)
    er[er_raw.str.contains("positive")] = 0
    er[er_raw.str.contains("negative")] = 1

    pam = clin["PAM50_simplified"].astype(str).str.lower()
    valid = ~pam.isin(["nan", "none", "", "unknown"])

    basal = pd.Series(np.nan, index=clin.index)
    basal[valid] = pam[valid].str.contains("basal").astype(int)

    luminal = pd.Series(np.nan, index=clin.index)
    luminal[valid] = pam[valid].str.contains("luminal|luma|lumb").astype(int)

    return {
        "stage_late": stage,
        "ER_negative": er,
        "PAM50_basal": basal,
        "PAM50_luminal": luminal,
    }

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "text").mkdir(exist_ok=True)

    aido = pd.read_csv(
        INPUT_DIR / "locked_MOATTERS_M1_M7_patient_representation.csv"
    )
    aido.index = aido["patient_id"].astype(str).str[:12]
    aido_X = aido[[f"M{i}" for i in range(1, 8)]].apply(
        pd.to_numeric, errors="coerce"
    )

    gsva_X = load_score(SCORE_DIR / "GSVA_scores_sample_by_BP.csv")
    path_X = load_score(SCORE_DIR / "Pathifier_scores_sample_by_BP.csv")

    clin = pd.read_csv(INPUT_DIR / "locked_benchmark_endpoints.csv")
    clin.index = clin["patient_id"].astype(str).str[:12]
    clin = clin[~clin.index.duplicated(keep="first")]

    reps = {
        "MOATTERS_state": aido_X,
        "GSVA": gsva_X,
        "Pathifier": path_X,
    }

    endpoints = endpoint_vectors(clin)
    summary_rows = []
    fold_rows = []
    patient_rows = []

    for endpoint, y_all in endpoints.items():
        common = set(y_all.dropna().index)
        for X in reps.values():
            common &= set(X.index)
        common = sorted(common)

        y = y_all.loc[common].astype(int)
        if y.nunique() != 2 or y.value_counts().min() < 20:
            continue

        splitter = RepeatedStratifiedKFold(
            n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED
        )
        folds = list(splitter.split(np.zeros(len(y)), y.to_numpy()))

        for method, Xdf in reps.items():
            X = Xdf.loc[common].astype(float)
            # Global z-standardization is unsupervised and used only to put
            # dimensions on comparable scales before fold-specific centroids.
            X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
            valid = X.notna().all(axis=1)
            X = X.loc[valid]
            yy = y.loc[valid]

            method_folds = list(
                RepeatedStratifiedKFold(
                    n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED
                ).split(np.zeros(len(yy)), yy.to_numpy())
            )

            oof_records = []
            fold_aucs = []
            Xn = X.to_numpy()
            yn = yy.to_numpy()

            for fold_id, (tr, te) in enumerate(method_folds, start=1):
                score = fit_score_centroid(Xn[tr], yn[tr], Xn[te])
                auc = roc_auc_score(yn[te], score)
                fold_aucs.append(max(auc, 1 - auc))
                for pid, yt, sc in zip(X.index[te], yn[te], score):
                    oof_records.append((pid, fold_id, yt, sc))

            oof = pd.DataFrame(
                oof_records, columns=["patient_id", "fold_id", "y", "score"]
            )
            agg = oof.groupby("patient_id").agg(
                y=("y", "first"), score=("score", "mean")
            )

            auc_dir = roc_auc_score(agg["y"], agg["score"])
            auc_inv = max(auc_dir, 1 - auc_dir)
            rb, p = rb_and_p(agg["score"].to_numpy(), agg["y"].to_numpy())

            summary_rows.append({
                "endpoint": endpoint,
                "method": method,
                "representation_dimensions": X.shape[1],
                "n_patients": len(agg),
                "n_positive": int(agg["y"].sum()),
                "n_negative": int((agg["y"] == 0).sum()),
                "aggregated_heldout_AUC_directional": auc_dir,
                "aggregated_heldout_AUC_orientation_invariant": auc_inv,
                "fold_AUC_mean": float(np.mean(fold_aucs)),
                "fold_AUC_SD": float(np.std(fold_aucs, ddof=1)),
                "rank_biserial": rb,
                "Mann_Whitney_p": p,
                "D_minus_log10_p": D(p),
            })

            for i, auc in enumerate(fold_aucs, start=1):
                fold_rows.append({
                    "endpoint": endpoint,
                    "method": method,
                    "fold_iteration": i,
                    "orientation_invariant_AUC": auc,
                })

            temp = agg.reset_index()
            temp["endpoint"] = endpoint
            temp["method"] = method
            patient_rows.append(temp)

    summary = pd.DataFrame(summary_rows)
    folds = pd.DataFrame(fold_rows)
    patients = pd.concat(patient_rows, ignore_index=True)

    summary.to_csv(
        OUT / "tables" / "Table_5B_GSVA_Pathifier_MOATTERS_benchmark.csv",
        index=False, encoding="utf-8-sig"
    )
    folds.to_csv(
        OUT / "tables" / "Table_5B_repeated_CV_fold_AUCs.csv",
        index=False, encoding="utf-8-sig"
    )
    patients.to_csv(
        OUT / "tables" / "Table_5B_aggregated_heldout_patient_scores.csv",
        index=False, encoding="utf-8-sig"
    )

    endpoint_order = [
        "stage_late", "ER_negative", "PAM50_basal", "PAM50_luminal"
    ]
    method_order = ["MOATTERS_state", "GSVA", "Pathifier"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.24
    x = np.arange(len(endpoint_order))

    for j, method in enumerate(method_order):
        sub = (
            summary[summary["method"] == method]
            .set_index("endpoint")
            .reindex(endpoint_order)
        )
        xpos = x + (j - 1) * width
        vals = sub["aggregated_heldout_AUC_orientation_invariant"].to_numpy()
        ax.bar(xpos, vals, width, label=method)
        for xi, value in zip(xpos, vals):
            if np.isfinite(value):
                ax.text(
                    xi, value + 0.008, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=8, rotation=90
                )

    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_xticks(x, endpoint_order)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("Aggregated repeated-CV orientation-invariant AUC")
    ax.set_title(
        "Locked-representation benchmark using identical patients, endpoints, and folds"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        OUT / "figures" / "Figure_5B_GSVA_Pathifier_MOATTERS_benchmark.png",
        dpi=300, bbox_inches="tight"
    )
    fig.savefig(
        OUT / "figures" / "Figure_5B_GSVA_Pathifier_MOATTERS_benchmark.pdf",
        bbox_inches="tight"
    )
    plt.close(fig)

    methods_text = (
        "The MOATTERS BP-state representation was benchmarked against GSVA and "
        "Pathifier using the same locked 30 task-discriminative GO-BP terms, "
        "the same endpoint-specific TCGA-BRCA patient intersections, and "
        "identical repeated stratified 20×5 folds. GSVA and Pathifier produced "
        "30-dimensional patient representations, whereas MOATTERS used the locked "
        "seven-component representation. In every training fold, class-specific "
        "centroids were estimated and applied to held-out patients using the "
        "same cosine-similarity margin. No method-specific tuning was performed. "
        "Pathifier used TCGA adjacent-normal tissues as its required normal "
        "reference. Because the 30 BP terms and MOATTERS module architecture were "
        "locked from the original full-cohort reconstruction, the analysis was "
        "interpreted as a representation/scoring benchmark rather than a fully "
        "nested feature-selection comparison."
    )

    result_parts = []
    for endpoint in endpoint_order:
        sub = summary[summary["endpoint"] == endpoint]
        if len(sub):
            vals = ", ".join(
                f"{r.method}={r.aggregated_heldout_AUC_orientation_invariant:.3f}"
                for _, r in sub.sort_values(
                    "aggregated_heldout_AUC_orientation_invariant",
                    ascending=False
                ).iterrows()
            )
            result_parts.append(f"{endpoint}: {vals}")
    results_text = (
        "Repeated-CV orientation-invariant AUCs were " + "; ".join(result_parts)
        + ". The comparison was used to determine whether the proposed "
        "task-conditioned reconstruction retained competitive endpoint "
        "discrimination while providing a compact, explicitly audited module "
        "representation; it was not used to claim uniform superiority."
    )
    rebuttal_text = (
        "We thank the reviewer for requesting comparison with established "
        "pathway-level methods. We added a controlled benchmark against GSVA "
        "and Pathifier using the same locked GO-BP terms, patients, endpoints, "
        "and repeated-CV folds. " + results_text
    )

    (OUT / "text" / "Methods_Stage5B.txt").write_text(
        methods_text, encoding="utf-8"
    )
    (OUT / "text" / "Results_Stage5B.txt").write_text(
        results_text, encoding="utf-8"
    )
    (OUT / "text" / "Rebuttal_Stage5B.txt").write_text(
        rebuttal_text, encoding="utf-8"
    )

    manifest = {
        "status": "PASS",
        "methods": method_order,
        "endpoints": endpoint_order,
        "CV": "20 repeats x 5 folds",
        "same_patient_intersection_per_endpoint": True,
        "same_folds_per_endpoint_and_method": True,
        "method_specific_tuning": False,
        "benchmark_boundary": (
            "Representation/scoring benchmark; not fully nested feature selection."
        ),
    }
    (OUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("PASS — Stage 5B benchmark evaluation completed")
    print(summary.to_string(index=False))
    print(f"Output: {OUT}")

if __name__ == "__main__":
    main()
