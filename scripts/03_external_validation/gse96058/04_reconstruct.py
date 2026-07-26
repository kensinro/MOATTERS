# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# -*- coding: utf-8 -*-
r"""
MOATTERS
GSE96058 Stage 2C — locked patient-level external reconstruction

Primary analysis
----------------
- External cohort: GSE96058/SCAN-B (GPL11154), n=3,069
- Locked TCGA-BRCA module group: stage_III_IV
- Locked selected BP set: 30 terms
- Primary minimum matched genes: K >= 10
- Locked TCGA early/late centroids
- Locked TCGA late-alignment direction
- Original patient-state thresholds:
    BURDEN_HIGH = 0.50
    LATE_SIM_HIGH = 0.35

Sensitivity analyses
--------------------
K = 5, 10, 15, 20.

Important boundary
------------------
No GSE96058 endpoint is used to select BP terms, form modules, estimate risk
direction, or estimate early/late centroids. Gene and module values are
standardized within GSE96058 using the historical implementation, after which
the locked TCGA-BRCA centroids and direction are applied.

Output
------
D:\MOATTERS-Output\MOATTERS_GSE96058_STAGE2C_RECONSTRUCTION
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from moatters.config import data_path, output_path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


STAGE1 = output_path(r"MOATTERS_GSE96058_STAGE1")
STAGE2B = output_path(r"MOATTERS_GSE96058_STAGE2AB_TRANSFER_COVERAGE")
OUT = output_path(r"MOATTERS_GSE96058_STAGE2C_RECONSTRUCTION")
GMT = data_path(r"GSEA\c5.go.bp.v2026.1.Hs.symbols.gmt")

MODULE_GROUP = "stage_III_IV"
K_VALUES = [5, 10, 15, 20]
PRIMARY_K = 10
BURDEN_HIGH = 0.50
LATE_SIM_HIGH = 0.35

EXPRESSION_CANDIDATES = [
    STAGE1 / "matrices" / "GSE96058_primary_expression_genes_x_samples.tsv.gz",
    STAGE1 / "matrices" / "GSE96058_primary_expression_genes_x_samples.tsv",
]
CLINICAL_CANDIDATES = [
    STAGE1 / "tables" / "GSE96058_primary_clinical_harmonized.tsv",
]

LOCKED = STAGE2B / "locked_artifacts"
ASSIGNMENT_PATH = LOCKED / "BRCA_module_assignment.csv"
COMPOSITION_PATH = LOCKED / "BRCA_profile_module_composition.csv"
DIRECTION_PATH = LOCKED / "BRCA_module_late_alignment_direction.csv"
CENTROID_PATH = LOCKED / "BRCA_early_late_module_centroids.csv"

# Optional historical labels used only to name the dominant route.
LABEL_FILENAME = "BRCA_module_TME_condition_labels.csv"
LABEL_SEARCH_ROOTS = [
    output_path(),
]


def log(msg: str, fh) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def clean_gene_symbol(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    return s.split("|")[0].strip().upper()


def normalize_module_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def locate_first(candidates: list[Path], description: str) -> Path:
    p = next((x for x in candidates if x.exists()), None)
    if p is None:
        raise FileNotFoundError(f"{description} not found. Tried: {candidates}")
    return p


def locate_optional_label_file() -> Path | None:
    hits = []
    seen = set()
    for root in LABEL_SEARCH_ROOTS:
        if not root.exists():
            continue
        key = str(root.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        hits.extend(root.rglob(LABEL_FILENAME))
    return sorted(hits, key=lambda p: (len(str(p)), str(p)))[0] if hits else None


def load_gmt(path: Path) -> dict[str, list[str]]:
    gene_sets = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:
                continue
            term = parts[0].strip()
            genes = sorted({clean_gene_symbol(g) for g in parts[2:] if clean_gene_symbol(g)})
            if genes:
                gene_sets[term] = genes
    return gene_sets


def zscore_rows(mat: pd.DataFrame) -> pd.DataFrame:
    means = mat.mean(axis=1, skipna=True)
    sds = mat.std(axis=1, skipna=True).replace(0, np.nan)
    return mat.sub(means, axis=0).div(sds, axis=0)


def zscore_scores_by_row(score_df: pd.DataFrame) -> pd.DataFrame:
    means = score_df.mean(axis=1, skipna=True)
    sds = score_df.std(axis=1, skipna=True).replace(0, np.nan)
    return score_df.sub(means, axis=0).div(sds, axis=0)


def vector_distance(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))


def vector_corr(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    r, _ = spearmanr(a[ok], b[ok])
    return float(r) if np.isfinite(r) else np.nan


def infer_strategy_from_label(label_info: dict) -> str:
    if not label_info:
        return "unknown_strategy"

    label = str(label_info.get("module_condition_label", "unknown")).lower()
    proxy = str(label_info.get("top_proxy", "")).lower()

    if any(x in label or x in proxy for x in ["immune", "inflammatory", "tnfa", "ifn"]):
        return "immune_inflammatory_route"
    if any(x in label or x in proxy for x in ["stromal", "hypoxia", "emt", "angiogenesis"]):
        return "stromal_EMT_hypoxia_route"
    if any(x in label or x in proxy for x in ["metabolic", "glycolysis", "oxidative", "fatty"]):
        return "metabolic_route"
    if any(x in label or x in proxy for x in ["proliferation", "e2f", "g2m", "myc"]):
        return "proliferation_route"
    if any(x in label or x in proxy for x in ["stress", "damage", "p53", "apoptosis", "repair"]):
        return "stress_damage_route"
    return "mixed_or_unclear_route"


def classify_patient_state(
    adverse_burden: float,
    sim_late: float,
    dis_late: float,
    dis_early: float,
    dominant_strategy: str,
) -> str:
    closer_to_late = (
        np.isfinite(dis_late) and np.isfinite(dis_early) and dis_late < dis_early
    )
    if adverse_burden >= BURDEN_HIGH and closer_to_late:
        return f"late_adverse_like__{dominant_strategy}"
    if adverse_burden >= BURDEN_HIGH:
        return f"high_burden__{dominant_strategy}"
    if np.isfinite(sim_late) and sim_late >= LATE_SIM_HIGH:
        return f"late_similarity__{dominant_strategy}"
    if adverse_burden <= -BURDEN_HIGH:
        return "favorable_or_opposite_to_late"
    return f"intermediate__{dominant_strategy}"


def load_labels(path: Path | None) -> dict[tuple[str, str], dict]:
    if path is None:
        return {}
    df = pd.read_csv(path, low_memory=False)
    out = {}
    for _, row in df.iterrows():
        key = (str(row.get("group", "")), normalize_module_id(row.get("module_id")))
        out[key] = {
            "module_condition_label": row.get("module_condition_label", "unknown"),
            "top_proxy": row.get("top_proxy", ""),
            "top_proxy_r": row.get("top_proxy_r", np.nan),
            "top_category": row.get("top_category", ""),
        }
    return out


def compute_bp_scores(
    zge: pd.DataFrame,
    gene_sets: dict[str, list[str]],
    terms: list[str],
    k_min: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ge_genes = set(zge.index.astype(str))
    scores = {}
    rows = []

    for term in terms:
        defined = gene_sets.get(term, [])
        matched = [g for g in defined if g in ge_genes]
        eligible = len(matched) >= k_min

        if eligible:
            scores[term] = zge.loc[matched].mean(axis=0, skipna=True)

        rows.append({
            "term": term,
            "n_defined": len(defined),
            "n_matched": len(matched),
            "matched_fraction": len(matched) / len(defined) if defined else 0.0,
            "K_min": k_min,
            "eligible": eligible,
            "readiness_class": (
                "observation_ready" if eligible
                else "low_resolution" if matched
                else "near_unobservable"
            ),
        })

    return pd.DataFrame(scores).T, pd.DataFrame(rows)


def compute_module_scores(
    bp_scores: pd.DataFrame,
    module_map: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = {}
    rows = []

    for module_id, locked_terms in module_map.items():
        present = [t for t in locked_terms if t in bp_scores.index]
        if present:
            scores[module_id] = bp_scores.loc[present].mean(axis=0, skipna=True)
        rows.append({
            "module_id": module_id,
            "n_BP_locked": len(locked_terms),
            "n_BP_scored": len(present),
            "retained_fraction": len(present) / len(locked_terms) if locked_terms else np.nan,
            "scored_terms": " | ".join(present),
            "excluded_terms": " | ".join([t for t in locked_terms if t not in present]),
        })

    return pd.DataFrame(scores).T, pd.DataFrame(rows)


def reconstruct_patients(
    module_scores_z: pd.DataFrame,
    early_centroid: pd.Series,
    late_centroid: pd.Series,
    direction: dict[str, int],
    label_map: dict,
) -> pd.DataFrame:
    modules = [
        m for m in early_centroid.index
        if m in module_scores_z.index and m in late_centroid.index and m in direction
    ]
    early = early_centroid.loc[modules]
    late = late_centroid.loc[modules]
    z = module_scores_z.loc[modules]

    rows = []
    for patient in z.columns:
        v = z[patient]
        dis_early = vector_distance(v.values, early.values)
        dis_late = vector_distance(v.values, late.values)
        sim_early = vector_corr(v.values, early.values)
        sim_late = vector_corr(v.values, late.values)

        risk_aligned = pd.Series(
            {m: direction[m] * v.loc[m] for m in modules},
            dtype=float,
        )
        adverse_burden = float(risk_aligned.mean())
        adverse_max = float(risk_aligned.max())
        adverse_top_module = normalize_module_id(risk_aligned.idxmax())

        dominant_active_module = normalize_module_id(v.idxmax())
        dominant_suppressed_module = normalize_module_id(v.idxmin())

        label_info = label_map.get((MODULE_GROUP, adverse_top_module), {})
        dominant_strategy = infer_strategy_from_label(label_info)
        state_class = classify_patient_state(
            adverse_burden, sim_late, dis_late, dis_early, dominant_strategy
        )

        rows.append({
            "patient": patient,
            "n_modules_used": len(modules),
            "adverse_burden": adverse_burden,
            "adverse_max_module_score": adverse_max,
            "adverse_top_module": adverse_top_module,
            "dominant_active_module": dominant_active_module,
            "dominant_suppressed_module": dominant_suppressed_module,
            "dominant_strategy": dominant_strategy,
            "Sim_early_centroid": sim_early,
            "Sim_late_centroid": sim_late,
            "Dis_early_centroid": dis_early,
            "Dis_late_centroid": dis_late,
            "late_vs_early_similarity_delta": (
                sim_late - sim_early
                if np.isfinite(sim_late) and np.isfinite(sim_early)
                else np.nan
            ),
            "late_vs_early_distance_delta": (
                dis_late - dis_early
                if np.isfinite(dis_late) and np.isfinite(dis_early)
                else np.nan
            ),
            "closer_to_late_centroid": (
                bool(dis_late < dis_early)
                if np.isfinite(dis_late) and np.isfinite(dis_early)
                else pd.NA
            ),
            "patient_state_class": state_class,
        })

    return pd.DataFrame(rows)


def safe_spearman(a: pd.Series, b: pd.Series) -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3:
        return np.nan
    r, _ = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
    return float(r) if np.isfinite(r) else np.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ["tables", "matrices", "audit", "logs"]:
        (OUT / sub).mkdir(exist_ok=True)

    with (OUT / "logs" / "stage2c_reconstruction.log").open(
        "w", encoding="utf-8"
    ) as fh:
        log("Starting locked GSE96058 patient-level reconstruction", fh)

        expr_path = locate_first(EXPRESSION_CANDIDATES, "Stage 1 expression matrix")
        clinical_path = next((p for p in CLINICAL_CANDIDATES if p.exists()), None)

        required = [
            GMT, ASSIGNMENT_PATH, COMPOSITION_PATH, DIRECTION_PATH, CENTROID_PATH
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing required inputs: {missing}")

        label_path = locate_optional_label_file()
        log(f"Expression: {expr_path}", fh)
        log(f"Optional TME labels: {label_path if label_path else 'NOT FOUND'}", fh)

        expression = pd.read_csv(
            expr_path, sep="\t", index_col=0, compression="infer", low_memory=False
        )
        expression.index = [clean_gene_symbol(x) for x in expression.index]
        expression = expression.loc[expression.index != ""]
        expression = expression.groupby(expression.index).mean()
        expression = expression.apply(pd.to_numeric, errors="coerce")
        log(f"GSE96058 expression loaded: {expression.shape[0]} genes x {expression.shape[1]} samples", fh)

        gene_sets = load_gmt(GMT)
        assignment = pd.read_csv(ASSIGNMENT_PATH, low_memory=False)
        assignment["module_id_norm"] = assignment["module_id"].map(normalize_module_id)
        profile = assignment[
            assignment["group"].astype(str) == MODULE_GROUP
        ].copy()

        if profile["term"].nunique() != 30:
            raise RuntimeError(
                f"Expected 30 locked BP terms; found {profile['term'].nunique()}."
            )

        module_map = {
            module_id: sub["term"].dropna().astype(str).tolist()
            for module_id, sub in profile.groupby("module_id_norm")
        }

        centroids = pd.read_csv(CENTROID_PATH, low_memory=False)
        centroids["module_id_norm"] = centroids["module_id"].map(normalize_module_id)
        centroids = centroids.set_index("module_id_norm")
        early_centroid = pd.to_numeric(centroids["early_centroid"], errors="coerce")
        late_centroid = pd.to_numeric(centroids["late_centroid"], errors="coerce")

        direction_df = pd.read_csv(DIRECTION_PATH, low_memory=False)
        direction_df["module_id_norm"] = direction_df["module_id"].map(normalize_module_id)
        direction = {
            row["module_id_norm"]: int(row["risk_direction_sign"])
            for _, row in direction_df.iterrows()
        }

        label_map = load_labels(label_path)

        module_order = list(early_centroid.index)
        if set(module_map) != set(module_order):
            raise RuntimeError(
                "Locked module IDs in assignment and TCGA centroids do not match: "
                f"assignment={sorted(module_map)}, centroids={sorted(module_order)}"
            )

        # Historical implementation: row-wise gene z-score within cohort.
        log("Computing row-wise GSE96058 gene z-scores", fh)
        zge = zscore_rows(expression)

        result_by_k = {}
        bp_readiness_by_k = {}
        module_coverage_by_k = {}
        module_z_by_k = {}

        for k in K_VALUES:
            log(f"Reconstruction at K>={k}", fh)
            bp_scores, bp_readiness = compute_bp_scores(
                zge, gene_sets, profile["term"].drop_duplicates().tolist(), k
            )
            module_raw, module_coverage = compute_module_scores(bp_scores, module_map)

            available_modules = [m for m in module_order if m in module_raw.index]
            missing_modules = [m for m in module_order if m not in module_raw.index]

            if not available_modules:
                raise RuntimeError(f"K={k} produced no reconstructable modules.")

            module_raw = module_raw.loc[available_modules]
            module_z = zscore_scores_by_row(module_raw)

            # K=10 is the locked primary seven-module reconstruction.
            # Stricter K values may remove every BP from a small module (for example M6).
            # Those sensitivity runs therefore use the remaining locked modules and are
            # explicitly labelled as reduced-space reconstructions; no module is refit.
            metrics = reconstruct_patients(
                module_z,
                early_centroid.loc[available_modules],
                late_centroid.loc[available_modules],
                {m: direction[m] for m in available_modules},
                label_map,
            )
            metrics["n_locked_modules_available"] = len(available_modules)
            metrics["missing_locked_modules"] = " | ".join(missing_modules)
            metrics["reconstruction_space"] = (
                "full_locked_7_module"
                if len(available_modules) == len(module_order)
                else "reduced_locked_module_space"
            )
            metrics["K_min_matched_genes"] = k

            result_by_k[k] = metrics
            bp_readiness_by_k[k] = bp_readiness
            module_coverage_by_k[k] = module_coverage
            module_z_by_k[k] = module_z

            bp_readiness.to_csv(
                OUT / "audit" / f"GSE96058_BP_readiness_K{k}.csv",
                index=False, encoding="utf-8-sig"
            )
            module_coverage.to_csv(
                OUT / "audit" / f"GSE96058_module_BP_retention_K{k}.csv",
                index=False, encoding="utf-8-sig"
            )
            module_raw.to_csv(
                OUT / "matrices" / f"GSE96058_module_scores_raw_K{k}.csv"
            )
            module_z.to_csv(
                OUT / "matrices" / f"GSE96058_module_scores_z_K{k}.csv"
            )
            metrics.to_csv(
                OUT / "tables" / f"GSE96058_patient_reconstruction_K{k}.csv",
                index=False, encoding="utf-8-sig"
            )

            log(
                f"K>={k}: BP eligible={int(bp_readiness['eligible'].sum())}/30; "
                f"modules={module_z.shape[0]}; patients={module_z.shape[1]}",
                fh,
            )

        # Primary locked output.
        primary = result_by_k[PRIMARY_K].copy()
        if clinical_path is not None:
            clinical = pd.read_csv(clinical_path, sep="\t", low_memory=False)
            patient_col = next(
                (c for c in ["sample_id", "patient_id", "Sample ID", "PATIENT_ID"]
                 if c in clinical.columns),
                None,
            )
            if patient_col is not None:
                clinical = clinical.rename(columns={patient_col: "patient"})
                primary = primary.merge(clinical, on="patient", how="left")
                log(f"Clinical metadata merged using '{patient_col}'", fh)
            else:
                log("Clinical table found, but no recognized patient column; not merged", fh)

        primary.to_csv(
            OUT / "GSE96058_patient_reconstruction_PRIMARY_K10.csv",
            index=False, encoding="utf-8-sig"
        )

        # Cross-K patient stability relative to primary K=10.
        base = result_by_k[PRIMARY_K].set_index("patient")
        stability_rows = []
        class_tables = []

        for k in K_VALUES:
            cur = result_by_k[k].set_index("patient")
            common = base.index.intersection(cur.index)
            agreement = (
                base.loc[common, "patient_state_class"]
                == cur.loc[common, "patient_state_class"]
            )
            stability_rows.append({
                "comparison": f"K10_vs_K{k}",
                "K_reference": 10,
                "K_comparator": k,
                "n_common_patients": len(common),
                "state_class_agreement": float(agreement.mean()),
                "adverse_burden_spearman": safe_spearman(
                    base.loc[common, "adverse_burden"],
                    cur.loc[common, "adverse_burden"],
                ),
                "late_similarity_delta_spearman": safe_spearman(
                    base.loc[common, "late_vs_early_similarity_delta"],
                    cur.loc[common, "late_vs_early_similarity_delta"],
                ),
                "late_distance_delta_spearman": safe_spearman(
                    base.loc[common, "late_vs_early_distance_delta"],
                    cur.loc[common, "late_vs_early_distance_delta"],
                ),
                "closer_to_late_agreement": float(
                    (
                        base.loc[common, "closer_to_late_centroid"].astype("string")
                        == cur.loc[common, "closer_to_late_centroid"].astype("string")
                    ).mean()
                ),
            })

            class_counts = (
                cur["patient_state_class"].value_counts(dropna=False)
                .rename_axis("patient_state_class")
                .reset_index(name="n_patients")
            )
            class_counts["K_min_matched_genes"] = k
            class_tables.append(class_counts)

        stability = pd.DataFrame(stability_rows)
        stability.to_csv(
            OUT / "tables" / "GSE96058_cross_K_patient_stability.csv",
            index=False, encoding="utf-8-sig"
        )
        pd.concat(class_tables, ignore_index=True).to_csv(
            OUT / "tables" / "GSE96058_state_class_counts_by_K.csv",
            index=False, encoding="utf-8-sig"
        )

        # Per-module cross-K stability relative to K10.
        module_rows = []
        base_mod = module_z_by_k[PRIMARY_K]
        for k in K_VALUES:
            cur_mod = module_z_by_k[k]
            for module_id in module_order:
                if module_id in cur_mod.index:
                    module_rows.append({
                        "comparison": f"K10_vs_K{k}",
                        "module_id": module_id,
                        "module_available": True,
                        "spearman_r": safe_spearman(
                            base_mod.loc[module_id],
                            cur_mod.loc[module_id],
                        ),
                        "mean_abs_difference": float(
                            (base_mod.loc[module_id] - cur_mod.loc[module_id]).abs().mean()
                        ),
                    })
                else:
                    module_rows.append({
                        "comparison": f"K10_vs_K{k}",
                        "module_id": module_id,
                        "module_available": False,
                        "spearman_r": np.nan,
                        "mean_abs_difference": np.nan,
                    })
        pd.DataFrame(module_rows).to_csv(
            OUT / "tables" / "GSE96058_cross_K_module_score_stability.csv",
            index=False, encoding="utf-8-sig"
        )

        # Locked contract and provenance.
        contract = {
            "external_cohort": "GSE96058",
            "primary_platform": "GPL11154",
            "n_samples": int(expression.shape[1]),
            "expression_path": str(expr_path),
            "expression_sha256": sha256_file(expr_path),
            "go_bp_gmt": str(GMT),
            "go_bp_gmt_sha256": sha256_file(GMT),
            "module_group": MODULE_GROUP,
            "n_locked_BP": int(profile["term"].nunique()),
            "n_locked_modules": int(len(module_map)),
            "primary_K_min_matched_genes": PRIMARY_K,
            "sensitivity_K_values": K_VALUES,
            "gene_standardization": "row-wise z-score across GSE96058 patients",
            "BP_score": "mean standardized expression across matched genes",
            "module_score": "mean BP score across eligible locked BP terms in module",
            "module_standardization": "row-wise z-score across GSE96058 patients",
            "centroids": "locked TCGA-BRCA early/late module-z centroids",
            "risk_direction": "locked TCGA-BRCA late-minus-early sign",
            "metabric_endpoint_used_for_reconstruction": False,
            "BURDEN_HIGH": BURDEN_HIGH,
            "LATE_SIM_HIGH": LATE_SIM_HIGH,
            "strategy_label_path": str(label_path) if label_path else None,
            "important_interpretive_boundary": (
                "This is cohort-standardized external reconstruction under locked "
                "TCGA-BRCA BP/module definitions, centroid references, direction, and "
                "classification thresholds. It is not endpoint-refitted reconstruction."
            ),
        }
        with (OUT / "GSE96058_STAGE2C_LOCKED_CONTRACT.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(contract, f, indent=2, ensure_ascii=False)

        k_summary = []
        for k in K_VALUES:
            readiness = bp_readiness_by_k[k]
            retention = module_coverage_by_k[k]
            metrics = result_by_k[k]
            k_summary.append({
                "K_min_matched_genes": k,
                "n_BP_eligible": int(readiness["eligible"].sum()),
                "n_BP_locked": 30,
                "n_modules_reconstructed": int(module_z_by_k[k].shape[0]),
                "missing_locked_modules": " | ".join(
                    [m for m in module_order if m not in module_z_by_k[k].index]
                ),
                "reconstruction_space": (
                    "full_locked_7_module"
                    if module_z_by_k[k].shape[0] == len(module_order)
                    else "reduced_locked_module_space"
                ),
                "minimum_module_BP_retained": int(retention["n_BP_scored"].min()),
                "mean_module_retained_fraction": float(retention["retained_fraction"].mean()),
                "n_patients": int(metrics.shape[0]),
                "mean_adverse_burden": float(metrics["adverse_burden"].mean()),
                "fraction_closer_to_late": float(
                    metrics["closer_to_late_centroid"].astype("boolean").mean()
                ),
            })

        summary = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": "GSE96058_STAGE2C_LOCKED_EXTERNAL_RECONSTRUCTION",
            "status": "PASS",
            "primary_K": PRIMARY_K,
            "n_patients_primary": int(result_by_k[PRIMARY_K].shape[0]),
            "n_modules_primary": int(module_z_by_k[PRIMARY_K].shape[0]),
            "n_locked_modules": 7,
            "n_locked_BP": 30,
            "sensitivity_interpretation": (
                "K15/K20 may be reduced-space reconstructions when all BP terms "
                "from a small locked module fail the stricter matched-gene threshold."
            ),
            "k_summary": k_summary,
            "cross_K_stability": stability.to_dict(orient="records"),
            "primary_state_class_counts": (
                result_by_k[PRIMARY_K]["patient_state_class"]
                .value_counts()
                .to_dict()
            ),
            "next_step": (
                "Stage 2D: endpoint-blinded external validation against GSE96058 stage, "
                "ER, PR, HER2, PAM50, OS and RFS, with effect sizes, uncertainty, "
                "permutation/null checks, and bounded claims."
            ),
        }
        with (OUT / "GSE96058_STAGE2C_SUMMARY.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        (OUT / "README_STAGE2C.txt").write_text(
            "MOATTERS — GSE96058 Stage 2C\n\n"
            "Status: PASS\n"
            f"Patients reconstructed: {result_by_k[PRIMARY_K].shape[0]}\n"
            "Locked BP terms: 30\n"
            "Locked modules: 7\n"
            "Primary K: 10\n"
            "Sensitivity K: 5, 10, 15, 20\n\n"
            "No GSE96058 endpoint was used to define or fit the reconstruction.\n",
            encoding="utf-8",
        )

        log("Stage 2C completed: PASS", fh)
        for row in k_summary:
            log(
                f"K>={row['K_min_matched_genes']}: "
                f"BP={row['n_BP_eligible']}/30; "
                f"modules={row['n_modules_reconstructed']}/7; "
                f"missing={row['missing_locked_modules'] or 'None'}; "
                f"min module BP={row['minimum_module_BP_retained']}; "
                f"patients={row['n_patients']}",
                fh,
            )


if __name__ == "__main__":
    main()
