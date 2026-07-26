# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# -*- coding: utf-8 -*-
r"""
MOATTERS
TCGA-KIRC Cross-Cancer Stage 2C — patient-level reconstruction,
centroid-only cross-validation, permutation null and survival context

Inputs
------
D:\MOATTERS-Output\MOATTERS_KIRC_STAGE1
D:\MOATTERS-Output\MOATTERS_KIRC_STAGE2B_NETWORK_MODULES

Output
------
D:\MOATTERS-Output\MOATTERS_KIRC_STAGE2C_RECONSTRUCTION

Scope
-----
The KIRC-specific top-30 BP observables and r>=0.35 module architecture are
treated as fixed inputs from Stages 2A/2B. This stage:

1. Standardizes module scores across locked KIRC patients.
2. Derives Early and Late centroids in module-z space.
3. Computes patient-level cosine similarities, Late-minus-Early margin,
   adverse-direction burden and a three-level reconstructed state.
4. Reports apparent/reference alignment with stage.
5. Runs repeated stratified cross-validation in which centroids are fitted
   only in training folds and evaluated in held-out patients.
6. Runs a label-permutation null for the held-out centroid reconstruction.
7. Reports OS association as secondary context.

Important boundary
------------------
The BP selection and network/module architecture were derived in the full
KIRC cohort in Stages 2A/2B. Therefore the repeated CV here audits only
centroid fitting and patient assignment, not a fully nested end-to-end model.
Claims must remain bounded to pipeline executability and internal
reconstruction behavior.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from moatters.config import data_path, output_path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
except ImportError:
    CoxPHFitter = None
    concordance_index = None


STAGE1 = output_path(r"MOATTERS_KIRC_STAGE1")
STAGE2B = output_path(r"MOATTERS_KIRC_STAGE2B_NETWORK_MODULES")

CLINICAL = STAGE1 / "tables" / "KIRC_primary_clinical_harmonized.tsv"
MODULE_SCORES = (
    STAGE2B / "matrices" /
    "KIRC_PRIMARY_r035_module_scores_patients_x_modules.csv.gz"
)
MODULE_ASSIGNMENT = (
    STAGE2B / "tables" /
    "KIRC_PRIMARY_r035_module_assignment.csv"
)

OUT = output_path(r"MOATTERS_KIRC_STAGE2C_RECONSTRUCTION")

N_SPLITS = 5
N_REPEATS = 20
RANDOM_STATE = 260724
N_PERM = 500
AMBIGUOUS_MARGIN_QUANTILE = 0.20


def log(msg: str, fh) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def cosine_rows(X: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    num = X @ centroid
    den = np.linalg.norm(X, axis=1) * np.linalg.norm(centroid)
    out = np.full(X.shape[0], np.nan, dtype=float)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def fit_centroids(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    early = np.nanmean(X[y == 0], axis=0)
    late = np.nanmean(X[y == 1], axis=0)
    return early, late


def score_from_centroids(
    X: np.ndarray,
    early_centroid: np.ndarray,
    late_centroid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sim_early = cosine_rows(X, early_centroid)
    sim_late = cosine_rows(X, late_centroid)
    margin = sim_late - sim_early
    return sim_early, sim_late, margin


def rank_biserial_from_auc(auc: float) -> float:
    return 2.0 * auc - 1.0


def evaluate_binary(y: np.ndarray, score: np.ndarray) -> dict:
    mask = np.isfinite(score) & np.isin(y, [0, 1])
    yy = y[mask].astype(int)
    ss = score[mask].astype(float)
    auc = roc_auc_score(yy, ss)
    late = ss[yy == 1]
    early = ss[yy == 0]
    u, p = mannwhitneyu(late, early, alternative="two-sided")
    return {
        "n": int(mask.sum()),
        "n_early": int((yy == 0).sum()),
        "n_late": int((yy == 1).sum()),
        "auc_directional": float(auc),
        "auc_orientation_invariant": float(max(auc, 1 - auc)),
        "rank_biserial": float(rank_biserial_from_auc(auc)),
        "mann_whitney_u": float(u),
        "p_value": float(p),
    }


def cv_centroid_scores(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    rkf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    rows = []
    for split_id, (train, test) in enumerate(rkf.split(X, y), start=1):
        c0, c1 = fit_centroids(X[train], y[train])
        se, sl, margin = score_from_centroids(X[test], c0, c1)
        auc = roc_auc_score(y[test], margin)
        for idx, sample_index in enumerate(test):
            rows.append({
                "split_id": split_id,
                "sample_index": int(sample_index),
                "true_stage_late": int(y[sample_index]),
                "cv_similarity_early": float(se[idx]),
                "cv_similarity_late": float(sl[idx]),
                "cv_late_minus_early_margin": float(margin[idx]),
                "split_auc": float(auc),
            })
    return pd.DataFrame(rows)


def permutation_null(
    X: np.ndarray,
    y: np.ndarray,
    observed_mean_auc: float,
) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=1,
        random_state=RANDOM_STATE,
    )
    rows = []
    for perm_id in range(1, N_PERM + 1):
        yp = rng.permutation(y)
        aucs = []
        for train, test in splitter.split(X, yp):
            c0, c1 = fit_centroids(X[train], yp[train])
            _, _, margin = score_from_centroids(X[test], c0, c1)
            aucs.append(roc_auc_score(yp[test], margin))
        rows.append({
            "permutation_id": perm_id,
            "mean_5fold_auc": float(np.mean(aucs)),
        })
    null = pd.DataFrame(rows)
    p = (1 + int((null["mean_5fold_auc"] >= observed_mean_auc).sum())) / (
        N_PERM + 1
    )
    return null, float(p)


def survival_context(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    rows = []
    if CoxPHFitter is None:
        return pd.DataFrame([{
            "status": "lifelines_not_installed"
        }])

    for score_col in score_cols:
        dat = df[["OS_time_days", "OS_event", score_col]].copy()
        dat = dat.apply(pd.to_numeric, errors="coerce").dropna()
        dat = dat[
            dat["OS_time_days"].gt(0)
            & dat["OS_event"].isin([0, 1])
        ].copy()
        if len(dat) < 50 or dat["OS_event"].sum() < 10:
            continue

        mean = dat[score_col].mean()
        sd = dat[score_col].std(ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            continue
        dat["score_z"] = (dat[score_col] - mean) / sd

        cph = CoxPHFitter()
        cph.fit(
            dat[["OS_time_days", "OS_event", "score_z"]],
            duration_col="OS_time_days",
            event_col="OS_event",
        )
        coef = float(cph.params_["score_z"])
        hr = float(np.exp(coef))
        ci = cph.confidence_intervals_.loc["score_z"]
        ci_low = float(np.exp(ci.iloc[0]))
        ci_high = float(np.exp(ci.iloc[1]))
        p = float(cph.summary.loc["score_z", "p"])

        # lifelines C-index expects larger predicted survival for longer survival.
        cidx = float(concordance_index(
            dat["OS_time_days"],
            -dat["score_z"],
            dat["OS_event"],
        ))

        rows.append({
            "score": score_col,
            "n": int(len(dat)),
            "events": int(dat["OS_event"].sum()),
            "HR_per_SD": hr,
            "CI95_low": ci_low,
            "CI95_high": ci_high,
            "p_value": p,
            "concordance_index": cidx,
        })
    return pd.DataFrame(rows)


def main():
    for p in [CLINICAL, MODULE_SCORES, MODULE_ASSIGNMENT]:
        if not p.exists():
            raise FileNotFoundError(p)

    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ["tables", "matrices", "logs", "audit"]:
        (OUT / sub).mkdir(exist_ok=True)

    with (OUT / "logs" / "kirc_stage2c.log").open(
        "w", encoding="utf-8"
    ) as fh:
        log("Starting TCGA-KIRC Stage 2C reconstruction", fh)

        clinical = pd.read_csv(CLINICAL, sep="\t", low_memory=False)
        clinical["sample_id"] = clinical["sample_id"].astype(str)
        clinical = clinical.set_index("sample_id")

        module_scores = pd.read_csv(
            MODULE_SCORES, index_col=0, compression="infer"
        )
        module_scores.index = module_scores.index.astype(str)

        samples = [s for s in module_scores.index if s in clinical.index]
        module_scores = module_scores.loc[samples].apply(
            pd.to_numeric, errors="coerce"
        )
        clinical = clinical.loc[samples]

        means = module_scores.mean(axis=0)
        sds = module_scores.std(axis=0, ddof=1)
        valid_modules = sds.gt(0) & sds.notna()
        module_z = module_scores.loc[:, valid_modules].sub(
            means[valid_modules], axis=1
        ).div(sds[valid_modules], axis=1)

        stage = pd.to_numeric(
            clinical["stage_late_binary"], errors="coerce"
        )
        usable = stage.isin([0, 1])
        X = module_z.loc[usable].to_numpy(float)
        y = stage.loc[usable].astype(int).to_numpy()
        usable_samples = stage.index[usable].tolist()

        log(
            f"Patients={len(samples)}; stage usable={len(usable_samples)}; "
            f"modules={module_z.shape[1]}",
            fh,
        )

        early_centroid, late_centroid = fit_centroids(X, y)
        sim_early, sim_late, margin = score_from_centroids(
            X, early_centroid, late_centroid
        )

        # Directional module burden uses signs of Late-Early centroid shifts.
        direction = np.sign(late_centroid - early_centroid)
        direction[direction == 0] = 1
        adverse_burden = np.nanmean(X * direction, axis=1)

        apparent_margin = evaluate_binary(y, margin)
        apparent_burden = evaluate_binary(y, adverse_burden)

        margin_abs = np.abs(margin)
        ambiguity_cut = float(
            np.quantile(margin_abs, AMBIGUOUS_MARGIN_QUANTILE)
        )
        state = np.where(
            margin_abs <= ambiguity_cut,
            "AMBIGUOUS",
            np.where(margin > 0, "LATE_LIKE", "EARLY_LIKE"),
        )

        patient = clinical.copy()
        patient["module_score_available"] = True
        patient["similarity_early"] = np.nan
        patient["similarity_late"] = np.nan
        patient["late_minus_early_margin"] = np.nan
        patient["adverse_direction_burden"] = np.nan
        patient["reconstructed_state"] = pd.NA
        patient.loc[usable_samples, "similarity_early"] = sim_early
        patient.loc[usable_samples, "similarity_late"] = sim_late
        patient.loc[usable_samples, "late_minus_early_margin"] = margin
        patient.loc[usable_samples, "adverse_direction_burden"] = adverse_burden
        patient.loc[usable_samples, "reconstructed_state"] = state

        cv = cv_centroid_scores(X, y)
        cv.to_csv(
            OUT / "tables" / "KIRC_centroid_only_repeated_CV_long.csv",
            index=False, encoding="utf-8-sig"
        )

        cv_split = cv.groupby("split_id", as_index=False).agg(
            split_auc=("split_auc", "first"),
            n_test=("sample_index", "size"),
        )
        cv_split.to_csv(
            OUT / "tables" / "KIRC_centroid_only_repeated_CV_summary.csv",
            index=False, encoding="utf-8-sig"
        )

        # Aggregate repeated held-out scores per patient.
        cv_patient = cv.groupby("sample_index", as_index=False).agg(
            cv_similarity_early=("cv_similarity_early", "mean"),
            cv_similarity_late=("cv_similarity_late", "mean"),
            cv_late_minus_early_margin=(
                "cv_late_minus_early_margin", "mean"
            ),
            n_heldout_predictions=("split_id", "size"),
        )
        cv_patient["sample_id"] = [
            usable_samples[i] for i in cv_patient["sample_index"]
        ]
        cv_patient = cv_patient.set_index("sample_id")
        patient = patient.join(
            cv_patient.drop(columns=["sample_index"]), how="left"
        )

        cv_eval = evaluate_binary(
            y,
            cv_patient.loc[usable_samples, "cv_late_minus_early_margin"]
            .to_numpy(float),
        )
        observed_mean_auc = float(cv_split["split_auc"].mean())

        log(
            f"Apparent margin AUC={apparent_margin['auc_directional']:.4f}; "
            f"mean repeated-CV split AUC={observed_mean_auc:.4f}",
            fh,
        )

        null, perm_p = permutation_null(X, y, observed_mean_auc)
        null.to_csv(
            OUT / "tables" / "KIRC_centroid_permutation_null.csv",
            index=False, encoding="utf-8-sig"
        )

        patient.to_csv(
            OUT / "tables" / "KIRC_patient_reconstruction_master.tsv",
            sep="\t", index=True
        )
        module_z.to_csv(
            OUT / "matrices" / "KIRC_module_z_patients_x_modules.csv.gz",
            compression="gzip"
        )

        centroids = pd.DataFrame({
            "module_id": module_z.columns,
            "early_centroid": early_centroid,
            "late_centroid": late_centroid,
            "late_minus_early": late_centroid - early_centroid,
            "adverse_direction_sign": direction,
        })
        centroids.to_csv(
            OUT / "tables" / "KIRC_Early_Late_module_centroids.csv",
            index=False, encoding="utf-8-sig"
        )

        eval_rows = []
        for name, obj, role in [
            ("late_minus_early_margin", apparent_margin,
             "apparent_reference_alignment"),
            ("adverse_direction_burden", apparent_burden,
             "apparent_reference_alignment"),
            ("cv_late_minus_early_margin", cv_eval,
             "centroid_only_heldout_alignment"),
        ]:
            eval_rows.append({
                "score": name,
                "evidence_role": role,
                **obj,
            })
        evaluation = pd.DataFrame(eval_rows)
        evaluation.to_csv(
            OUT / "tables" / "KIRC_stage_alignment_results.csv",
            index=False, encoding="utf-8-sig"
        )

        survival = survival_context(
            patient.reset_index(),
            [
                "late_minus_early_margin",
                "adverse_direction_burden",
                "cv_late_minus_early_margin",
            ],
        )
        survival.to_csv(
            OUT / "tables" / "KIRC_survival_context_results.csv",
            index=False, encoding="utf-8-sig"
        )

        state_counts = (
            patient.loc[usable_samples, "reconstructed_state"]
            .value_counts(dropna=False)
            .to_dict()
        )

        summary = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": "TCGA_KIRC_STAGE2C_RECONSTRUCTION",
            "status": "PASS",
            "n_locked_patients": len(samples),
            "n_stage_usable": len(usable_samples),
            "n_early": int((y == 0).sum()),
            "n_late": int((y == 1).sum()),
            "n_modules": int(module_z.shape[1]),
            "ambiguity_rule": (
                f"lowest {AMBIGUOUS_MARGIN_QUANTILE:.0%} of absolute "
                "Late-minus-Early centroid margins"
            ),
            "ambiguity_cutoff": ambiguity_cut,
            "state_counts": {
                str(k): int(v) for k, v in state_counts.items()
            },
            "apparent_margin_alignment": apparent_margin,
            "apparent_adverse_burden_alignment": apparent_burden,
            "centroid_only_repeated_CV": {
                "n_splits": N_SPLITS,
                "n_repeats": N_REPEATS,
                "n_total_splits": int(len(cv_split)),
                "mean_split_auc": observed_mean_auc,
                "sd_split_auc": float(cv_split["split_auc"].std(ddof=1)),
                "min_split_auc": float(cv_split["split_auc"].min()),
                "max_split_auc": float(cv_split["split_auc"].max()),
                "aggregated_patient_auc": cv_eval["auc_directional"],
            },
            "permutation_null": {
                "n_permutations": N_PERM,
                "observed_mean_split_auc": observed_mean_auc,
                "null_mean_auc": float(null["mean_5fold_auc"].mean()),
                "null_sd_auc": float(null["mean_5fold_auc"].std(ddof=1)),
                "empirical_p_value": perm_p,
            },
            "methodological_boundaries": [
                "KIRC-specific BP terms and modules were fixed from Stages 2A/2B.",
                "Repeated CV refitted only Early/Late centroids.",
                "This is not fully nested end-to-end validation because BP selection and module construction used the full KIRC cohort.",
                "Stage alignment is an internal cross-cancer applicability demonstration, not independent external validation.",
                "Survival is secondary context and not a universal prognostic claim.",
            ],
            "next_step": (
                "Stage 2D: integrate KIRC cutoff/network sensitivity and "
                "reconstruction results into a reviewer-facing cross-cancer table; "
                "then repeat the minimal pipeline in TCGA-LUAD."
            ),
        }
        with (OUT / "KIRC_STAGE2C_SUMMARY.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        (OUT / "README_KIRC_STAGE2C.txt").write_text(
            "MOATTERS — TCGA-KIRC Stage 2C\n\n"
            "Status: PASS\n"
            f"Patients: {len(samples)}; stage usable: {len(usable_samples)}\n"
            f"Modules: {module_z.shape[1]}\n"
            f"Apparent margin AUC: {apparent_margin['auc_directional']:.4f}\n"
            f"Mean centroid-only repeated-CV AUC: {observed_mean_auc:.4f}\n"
            f"Permutation empirical p: {perm_p:.6g}\n\n"
            "The CV audits centroid fitting only; the architecture was fixed "
            "from the full KIRC cohort.\n",
            encoding="utf-8",
        )

        log("KIRC Stage 2C completed: PASS", fh)
        log(
            f"Permutation null mean={null['mean_5fold_auc'].mean():.4f}; "
            f"empirical p={perm_p:.6g}",
            fh,
        )


if __name__ == "__main__":
    main()
