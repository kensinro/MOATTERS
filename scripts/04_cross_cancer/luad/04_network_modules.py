# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# -*- coding: utf-8 -*-
r"""
MOATTERS
TCGA-LUAD Cross-Cancer Stage 2B — top-30 BP lock, late-state network,
correlation-threshold sensitivity and LUAD-specific modules

Inputs
------
D:\MOATTERS-Output\MOATTERS_LUAD_STAGE2A_BP_SCREEN
D:\MOATTERS-Output\MOATTERS_LUAD_STAGE1

Output
------
D:\MOATTERS-Output\MOATTERS_LUAD_STAGE2B_NETWORK_MODULES

Primary analysis contract
-------------------------
1. Select the 30 most stage-discriminative LUAD GO-BP observables among
   K>=10 observation-ready terms, ranked by D=-log10(p).
2. Use only Late-stage LUAD patients to estimate the BP-BP Pearson network.
3. Primary network threshold: |r| >= 0.35.
4. Sensitivity thresholds: |r| >= 0.25, 0.35, 0.45.
5. Derive LUAD-specific communities by greedy modularity optimization.
6. Isolated nodes remain singleton modules.
7. No BRCA or KIRC BP terms, edges, centroids, modules or patient labels
   are reused.

This stage derives the LUAD representation architecture only. Patient-level
centroid reconstruction and validation are deferred to Stage 2C.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from moatters.config import data_path, output_path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

try:
    import networkx as nx
except ImportError as exc:
    raise ImportError(
        "networkx is required for Stage 2B. Install with: pip install networkx"
    ) from exc


STAGE1 = output_path(r"MOATTERS_LUAD_STAGE1")
STAGE2A = output_path(r"MOATTERS_LUAD_STAGE2A_BP_SCREEN")
CLINICAL = STAGE1 / "tables" / "LUAD_primary_clinical_harmonized.tsv"
ALL_RESULTS = (
    STAGE2A / "tables" / "LUAD_GO_BP_stage_discriminability_all.csv"
)
BP_SCORES = (
    STAGE2A / "matrices" /
    "LUAD_K10_observation_ready_BP_scores_terms_x_patients.csv.gz"
)
OUT = output_path(r"MOATTERS_LUAD_STAGE2B_NETWORK_MODULES")

TOP_N = 30
PRIMARY_R = 0.35
R_THRESHOLDS = [0.25, 0.35, 0.45]


def log(msg: str, fh) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def build_edge_table(score_patient_by_term: pd.DataFrame) -> pd.DataFrame:
    terms = list(score_patient_by_term.columns)
    rows = []
    for i, a in enumerate(terms):
        xa = pd.to_numeric(score_patient_by_term[a], errors="coerce")
        for b in terms[i + 1:]:
            xb = pd.to_numeric(score_patient_by_term[b], errors="coerce")
            mask = xa.notna() & xb.notna()
            n = int(mask.sum())
            if n < 4:
                r, p = np.nan, np.nan
            else:
                r, p = pearsonr(xa.loc[mask], xb.loc[mask])
            rows.append({
                "source": a,
                "target": b,
                "n_late_patients": n,
                "pearson_r": float(r) if np.isfinite(r) else np.nan,
                "abs_r": abs(float(r)) if np.isfinite(r) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
            })
    return pd.DataFrame(rows)


def graph_and_modules(
    terms: list[str],
    edges: pd.DataFrame,
    threshold: float,
) -> tuple[nx.Graph, list[set[str]], dict[str, str]]:
    g = nx.Graph()
    g.add_nodes_from(terms)

    kept = edges[
        pd.to_numeric(edges["abs_r"], errors="coerce").ge(threshold)
    ]
    for row in kept.itertuples(index=False):
        g.add_edge(
            row.source,
            row.target,
            weight=float(row.abs_r),
            signed_r=float(row.pearson_r),
        )

    nonisolated = [n for n in g.nodes if g.degree(n) > 0]
    isolated = sorted([n for n in g.nodes if g.degree(n) == 0])

    communities = []
    if nonisolated:
        sub = g.subgraph(nonisolated).copy()
        communities = [
            set(c)
            for c in nx.algorithms.community.greedy_modularity_communities(
                sub, weight="weight"
            )
        ]

    communities.extend([{n} for n in isolated])
    communities = sorted(
        communities,
        key=lambda c: (-len(c), sorted(c)[0]),
    )

    module_map = {}
    for idx, community in enumerate(communities, start=1):
        module_id = f"M{idx}"
        for term in sorted(community):
            module_map[term] = module_id

    return g, communities, module_map


def modularity_safe(g: nx.Graph, communities: list[set[str]]) -> float:
    if g.number_of_edges() == 0 or len(communities) <= 1:
        return np.nan
    try:
        return float(
            nx.algorithms.community.modularity(
                g, communities, weight="weight"
            )
        )
    except Exception:
        return np.nan


def main():
    for p in [CLINICAL, ALL_RESULTS, BP_SCORES]:
        if not p.exists():
            raise FileNotFoundError(p)

    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ["tables", "matrices", "logs", "audit"]:
        (OUT / sub).mkdir(exist_ok=True)

    with (OUT / "logs" / "luad_stage2b.log").open(
        "w", encoding="utf-8"
    ) as fh:
        log("Starting TCGA-LUAD Stage 2B network/module derivation", fh)

        results = pd.read_csv(ALL_RESULTS)
        eligible = results[
            results["observation_ready_K10"].astype(bool)
            & pd.to_numeric(results["D"], errors="coerce").notna()
        ].copy()
        eligible = eligible.sort_values(
            ["D", "n_genes_matched", "term"],
            ascending=[False, False, True],
        )

        if len(eligible) < TOP_N:
            raise RuntimeError(
                f"Only {len(eligible)} eligible BP terms; need {TOP_N}."
            )

        locked = eligible.head(TOP_N).copy()
        locked["LUAD_rank_by_D"] = np.arange(1, TOP_N + 1)
        locked.to_csv(
            OUT / "tables" / "LUAD_top30_BP_lock.csv",
            index=False,
            encoding="utf-8-sig",
        )
        terms = locked["term"].tolist()

        scores = pd.read_csv(
            BP_SCORES,
            index_col=0,
            compression="infer",
        )
        missing_terms = [t for t in terms if t not in scores.index]
        if missing_terms:
            raise RuntimeError(
                f"Top-30 BP terms missing from score matrix: {missing_terms}"
            )

        clinical = pd.read_csv(CLINICAL, sep="\t", low_memory=False)
        clinical["sample_id"] = clinical["sample_id"].astype(str)
        clinical = clinical.set_index("sample_id")
        stage = pd.to_numeric(
            clinical["stage_late_binary"], errors="coerce"
        )
        late_samples = [
            s for s in scores.columns
            if s in stage.index and stage.loc[s] == 1
        ]
        if len(late_samples) < 50:
            raise RuntimeError(
                f"Unexpectedly few Late-stage patients: {len(late_samples)}"
            )

        late_matrix = scores.loc[terms, late_samples].T
        late_matrix.index.name = "sample_id"
        late_matrix.to_csv(
            OUT / "matrices" / "LUAD_top30_BP_scores_late_patients.csv.gz",
            compression="gzip",
        )

        edges = build_edge_table(late_matrix)
        edges.to_csv(
            OUT / "tables" / "LUAD_top30_late_all_pairwise_correlations.csv",
            index=False,
            encoding="utf-8-sig",
        )

        sensitivity_rows = []
        assignment_rows = []
        primary_communities = None
        primary_map = None

        for threshold in R_THRESHOLDS:
            g, communities, module_map = graph_and_modules(
                terms, edges, threshold
            )
            degrees = dict(g.degree())
            components = list(nx.connected_components(g))

            sensitivity_rows.append({
                "abs_r_threshold": threshold,
                "n_nodes": g.number_of_nodes(),
                "n_edges": g.number_of_edges(),
                "network_density": float(nx.density(g)),
                "n_connected_components": len(components),
                "largest_component_size": max(map(len, components)),
                "n_isolated_nodes": int(
                    sum(1 for n in g.nodes if g.degree(n) == 0)
                ),
                "n_modules": len(communities),
                "largest_module_size": max(map(len, communities)),
                "smallest_module_size": min(map(len, communities)),
                "modularity": modularity_safe(g, communities),
            })

            for term in terms:
                assignment_rows.append({
                    "abs_r_threshold": threshold,
                    "term": term,
                    "module_id": module_map[term],
                    "module_size": len(
                        next(c for c in communities if term in c)
                    ),
                    "degree": degrees[term],
                    "is_isolate": degrees[term] == 0,
                })

            kept = edges[
                pd.to_numeric(edges["abs_r"], errors="coerce").ge(threshold)
            ].copy()
            kept.to_csv(
                OUT / "tables" /
                f"LUAD_top30_edges_abs_r_ge_{threshold:.2f}.csv",
                index=False,
                encoding="utf-8-sig",
            )

            if np.isclose(threshold, PRIMARY_R):
                primary_communities = communities
                primary_map = module_map

        sensitivity = pd.DataFrame(sensitivity_rows)
        sensitivity.to_csv(
            OUT / "tables" /
            "LUAD_network_threshold_sensitivity_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        assignments = pd.DataFrame(assignment_rows)
        assignments.to_csv(
            OUT / "tables" /
            "LUAD_module_assignments_all_r_thresholds.csv",
            index=False,
            encoding="utf-8-sig",
        )

        primary_assignment = assignments[
            np.isclose(assignments["abs_r_threshold"], PRIMARY_R)
        ].copy()
        primary_assignment = primary_assignment.merge(
            locked[[
                "term", "LUAD_rank_by_D", "D",
                "late_minus_early",
                "cohens_d_late_minus_early",
                "n_genes_matched",
            ]],
            on="term",
            how="left",
        )
        primary_assignment = primary_assignment.sort_values(
            ["module_id", "LUAD_rank_by_D"]
        )
        primary_assignment.to_csv(
            OUT / "tables" /
            "LUAD_PRIMARY_r035_module_assignment.csv",
            index=False,
            encoding="utf-8-sig",
        )

        all_top30 = scores.loc[terms].T
        module_scores = {}
        for module_id in sorted(
            set(primary_map.values()),
            key=lambda x: int(x[1:])
        ):
            members = [
                term for term, mid in primary_map.items()
                if mid == module_id
            ]
            module_scores[module_id] = all_top30[members].mean(
                axis=1, skipna=True
            )

        module_scores = pd.DataFrame(module_scores)
        module_scores.index.name = "sample_id"
        module_scores.to_csv(
            OUT / "matrices" /
            "LUAD_PRIMARY_r035_module_scores_patients_x_modules.csv.gz",
            compression="gzip",
        )

        primary_summary = sensitivity[
            np.isclose(sensitivity["abs_r_threshold"], PRIMARY_R)
        ].iloc[0].to_dict()

        status = "PASS"
        if primary_summary["n_edges"] == 0:
            status = "HOLD_EMPTY_PRIMARY_NETWORK"
        elif primary_summary["n_modules"] < 2:
            status = "HOLD_SINGLE_PRIMARY_MODULE"

        summary = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": "TCGA_LUAD_STAGE2B_NETWORK_MODULES",
            "status": status,
            "top_N_BP": TOP_N,
            "selection_rule": (
                "Top 30 K>=10 LUAD GO-BP terms ranked by D=-log10(p) "
                "for Stage I/II versus Stage III/IV"
            ),
            "n_late_patients_for_network": len(late_samples),
            "primary_abs_r_threshold": PRIMARY_R,
            "primary_network": primary_summary,
            "threshold_sensitivity": sensitivity_rows,
            "primary_module_sizes": {
                f"M{i}": len(c)
                for i, c in enumerate(primary_communities, start=1)
            },
            "methodological_boundaries": [
                "The top-30 BP lock is LUAD-specific.",
                "The BP network was estimated only in Late-stage LUAD patients.",
                "The primary edge threshold was |r|>=0.35.",
                "Threshold sensitivity was assessed at |r|>=0.25, 0.35 and 0.45.",
                "BRCA and KIRC BP terms, modules and centroids were not transferred.",
                "Patient reconstruction has not yet been evaluated.",
            ],
            "next_step": (
                "Stage 2C: derive Early/Late centroids in LUAD module space, "
                "reconstruct patient-level molecular-functional states, and "
                "audit internal stability/null behavior."
            ),
        }
        with (OUT / "LUAD_STAGE2B_SUMMARY.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        (OUT / "README_LUAD_STAGE2B.txt").write_text(
            "MOATTERS — TCGA-LUAD Stage 2B\n\n"
            f"Status: {status}\n"
            f"Top BP terms locked: {TOP_N}\n"
            f"Late-stage patients used for network: {len(late_samples)}\n"
            f"Primary threshold: |r| >= {PRIMARY_R}\n"
            f"Primary edges: {int(primary_summary['n_edges'])}\n"
            f"Primary modules: {int(primary_summary['n_modules'])}\n"
            "The architecture is LUAD-specific; no BRCA/KIRC module transfer occurred.\n",
            encoding="utf-8",
        )

        log(f"LUAD Stage 2B completed: {status}", fh)
        log(
            f"Primary r>={PRIMARY_R}: edges={int(primary_summary['n_edges'])}, "
            f"modules={int(primary_summary['n_modules'])}, "
            f"isolates={int(primary_summary['n_isolated_nodes'])}",
            fh,
        )


if __name__ == "__main__":
    main()
