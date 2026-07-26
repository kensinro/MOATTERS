# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
# ============================================================
# MOATTERS Mission Attractor Pilot V3
# Stage I/II vs Stage III/IV high-D BP trend + BP network
# Fixes:
#   1. Phenotype.tsv UTF-16 / UTF-16-LE encoding problem
#   2. If one stage file fails, continue to next candidate file
#   3. PPI sequence files are skipped
#   4. BRCA / BLCA / CESC / COAD etc. can continue running
# ============================================================

import os
import re
import math
import json
import warnings
from pathlib import Path
from moatters.config import data_path, output_path

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu, spearmanr
import networkx as nx
import matplotlib.pyplot as plt


# ============================================================
# 0. USER CONFIG
# ============================================================

ROOT = data_path()

UCSC_XENA_DIR = ROOT / "UCSC_XENA"
GSEA_DIR = ROOT / "GSEA"
PPI_DIR = ROOT / "Kaggle-PPI"

GMT_PATH = GSEA_DIR / "c5.go.bp.v2026.1.Hs.symbols.gmt"
# GMT_PATH = GSEA_DIR / "h.all.v2026.1.Hs.symbols.gmt"
# GMT_PATH = GSEA_DIR / "c2.cp.reactome.v2026.1.Hs.symbols.gmt"

OUT_DIR = output_path("MOATTERS_STAGE_BP_Attractor_Pilot_V3_20260530")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANCER_FOLDERS = [
    "Bladder Cancer (BLCA)",
    "Breast Cancer (BRCA)",
    "Cervical Cancer (CESC)",
    "Colon Cancer (COAD)",
    "Lung Adenocarcinoma (LUAD)",
    "Lung Squamous Cell Carcinoma (LUSC)",
    "Kidney Clear Cell Carcinoma (KIRC)",
    "Liver Cancer (LIHC)",
]

MIN_MATCHED_GENES = 10
MIN_PATIENTS_TOTAL = 40
MIN_PATIENTS_PER_GROUP = 10

D_THRESHOLD = 1.301
TOP_N_BP_FOR_NETWORK = 30

SCORE_METHOD = "mean_z_gene"

CORR_METHOD = "spearman"
CORR_ABS_THRESHOLD = 0.35
CORR_P_THRESHOLD = 0.05

MAKE_FIGURES = True


# ============================================================
# 1. BASIC HELPERS
# ============================================================

def tcga_patient_id(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", s, flags=re.I)
    if m:
        return m.group(1).upper()
    return s.upper()


def clean_gene_symbol(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.split("|")[0].strip()
    return s.upper()


def safe_neglog10(p):
    if p is None or pd.isna(p):
        return np.nan
    p = float(p)
    if p <= 0:
        return 300.0
    return -math.log10(p)


def detect_encoding_by_bom(path):
    """
    Quick BOM detection.
    """
    with open(path, "rb") as f:
        raw = f.read(4)

    if raw.startswith(b"\xff\xfe"):
        return "utf-16"
    if raw.startswith(b"\xfe\xff"):
        return "utf-16"
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"

    return None


def read_table_safely(path, nrows=None, verbose=False):
    """
    Robust TSV/CSV reader with encoding fallback.

    This fixes the UCSC Xena Phenotype.tsv issue:
      many phenotype files are UTF-16 little-endian.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    bom_enc = detect_encoding_by_bom(path)

    encodings = []
    if bom_enc:
        encodings.append(bom_enc)

    encodings += [
        "utf-8-sig",
        "utf-8",
        "utf-16",
        "utf-16-le",
        "latin1",
    ]

    # remove duplicates
    encodings = list(dict.fromkeys(encodings))

    errors = []

    for enc in encodings:
        # tab, C engine
        try:
            df = pd.read_csv(
                path,
                sep="\t",
                encoding=enc,
                low_memory=False,
                nrows=nrows
            )
            if df.shape[1] > 1:
                if verbose:
                    print(f"[read_table_safely] OK sep=tab enc={enc}: {path.name}, shape={df.shape}")
                return df
        except Exception as e:
            errors.append(("tab", enc, repr(e)[:150]))

        # comma, C engine
        try:
            df = pd.read_csv(
                path,
                sep=",",
                encoding=enc,
                low_memory=False,
                nrows=nrows
            )
            if df.shape[1] > 1:
                if verbose:
                    print(f"[read_table_safely] OK sep=comma enc={enc}: {path.name}, shape={df.shape}")
                return df
        except Exception as e:
            errors.append(("comma", enc, repr(e)[:150]))

        # automatic separator, python engine
        # NOTE: no low_memory with python engine
        try:
            df = pd.read_csv(
                path,
                sep=None,
                engine="python",
                encoding=enc,
                nrows=nrows
            )
            if verbose:
                print(f"[read_table_safely] OK sep=auto enc={enc}: {path.name}, shape={df.shape}")
            return df
        except Exception as e:
            errors.append(("auto", enc, repr(e)[:150]))

    msg = f"Could not read table: {path}\nTried encodings/errors:\n"
    for item in errors[:12]:
        msg += f"  sep={item[0]} enc={item[1]} err={item[2]}\n"
    raise RuntimeError(msg)


def looks_like_tcga_colnames(cols, min_frac=0.2):
    cols = list(map(str, cols))
    if len(cols) == 0:
        return False
    hits = sum(
        1 for c in cols
        if re.search(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", c, flags=re.I)
    )
    return hits / max(len(cols), 1) >= min_frac


def looks_like_tcga_values(series, min_frac=0.2):
    vals = series.dropna().astype(str).head(300).tolist()
    if len(vals) == 0:
        return False
    hits = sum(
        1 for v in vals
        if re.search(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", v, flags=re.I)
    )
    return hits / max(len(vals), 1) >= min_frac


def extract_cancer_code(folder_name):
    m = re.search(r"\(([A-Z0-9]+)\)", folder_name)
    return m.group(1) if m else folder_name.replace(" ", "_")


# ============================================================
# 2. GMT LOADING
# ============================================================

def load_gmt(gmt_path):
    gene_sets = {}

    with open(gmt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            term = parts[0].strip()
            genes = [
                clean_gene_symbol(g)
                for g in parts[2:]
                if clean_gene_symbol(g)
            ]
            genes = sorted(set(genes))

            if genes:
                gene_sets[term] = genes

    return gene_sets


# ============================================================
# 3. GE LOADING
# ============================================================

def infer_gene_column(df):
    possible = [
        "gene", "genes", "symbol", "gene_symbol",
        "hugo_symbol", "Hugo_Symbol", "Gene", "GENE",
        "sample"
    ]

    cols_lower = {str(c).lower(): c for c in df.columns}
    for p in possible:
        if p.lower() in cols_lower:
            return cols_lower[p.lower()]

    return df.columns[0]


def load_ge_matrix(ge_path):
    df = read_table_safely(ge_path)

    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    # Case A: genes rows, TCGA sample columns
    if looks_like_tcga_colnames(df.columns):
        gene_col = infer_gene_column(df)
        genes = df[gene_col].map(clean_gene_symbol)

        mat = df.drop(columns=[gene_col])
        mat.columns = [tcga_patient_id(c) for c in mat.columns]
        mat.index = genes
        mat = mat[mat.index.notna()]
        mat = mat.apply(pd.to_numeric, errors="coerce")

        mat = mat.T.groupby(level=0).mean().T
        mat = mat.groupby(mat.index).mean()

        return mat

    # Case B: samples rows, genes columns
    first_col = df.columns[0]
    if looks_like_tcga_values(df[first_col]):
        samples = df[first_col].map(tcga_patient_id)

        mat = df.drop(columns=[first_col])
        mat.index = samples
        mat.columns = [clean_gene_symbol(c) for c in mat.columns]
        mat = mat.loc[mat.index.notna(), :]
        mat = mat.loc[:, pd.notna(mat.columns)]
        mat = mat.apply(pd.to_numeric, errors="coerce")

        mat = mat.groupby(mat.index).mean()
        mat = mat.T
        mat = mat.groupby(mat.index).mean()

        return mat

    # Case C: assume first column gene
    gene_col = infer_gene_column(df)
    genes = df[gene_col].map(clean_gene_symbol)

    mat = df.drop(columns=[gene_col])
    mat.columns = [tcga_patient_id(c) for c in mat.columns]
    mat.index = genes
    mat = mat[mat.index.notna()]
    mat = mat.apply(pd.to_numeric, errors="coerce")

    mat = mat.T.groupby(level=0).mean().T
    mat = mat.groupby(mat.index).mean()

    return mat


# ============================================================
# 4. STAGE LOADING
# ============================================================

def roman_stage_to_group(x):
    if pd.isna(x):
        return None

    s0 = str(x).strip()
    if not s0:
        return None

    s = s0.upper()

    bad = [
        "", "NAN", "NA", "N/A", "NONE", "UNKNOWN",
        "NOT REPORTED", "NOT AVAILABLE", "[NOT AVAILABLE]",
        "[NOT EVALUATED]", "NOT EVALUATED", "TX", "TIS",
        "STAGE X", "X"
    ]
    if s in bad:
        return None

    sl = s.lower().strip()
    if sl in ["early", "stage_i_ii", "i_ii", "stage i/ii", "stage i-ii", "i/ii"]:
        return "Early"
    if sl in ["late", "stage_iii_iv", "iii_iv", "stage iii/iv", "stage iii-iv", "iii/iv"]:
        return "Late"

    s = s.replace("PATHOLOGIC", " ")
    s = s.replace("CLINICAL", " ")
    s = s.replace("STAGE", " ")
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # Roman, order matters
    if re.search(r"\bIV\b|IVA|IVB|IVC", s):
        return "Late"
    if re.search(r"\bIII\b|IIIA|IIIB|IIIC", s):
        return "Late"
    if re.search(r"\bII\b|IIA|IIB|IIC", s):
        return "Early"
    if re.search(r"\bI\b|IA|IB|IC", s):
        return "Early"

    # Numeric
    if re.search(r"\b4\b", s):
        return "Late"
    if re.search(r"\b3\b", s):
        return "Late"
    if re.search(r"\b2\b", s):
        return "Early"
    if re.search(r"\b1\b", s):
        return "Early"

    return None


def find_stage_files(cancer_dir):
    cancer_dir = Path(cancer_dir)

    candidates = []
    candidates += list(cancer_dir.glob("*stage_groups_from_survival*.tsv"))
    candidates += list(cancer_dir.glob("Phenotype.tsv"))
    candidates += list(cancer_dir.glob("*clinicalMatrix*"))
    candidates += list(cancer_dir.glob("*clinical*.tsv"))
    candidates += list(cancer_dir.glob("*phenotype*.tsv"))

    seen = set()
    out = []
    for p in candidates:
        if p.exists() and p.is_file() and str(p) not in seen:
            out.append(p)
            seen.add(str(p))

    return out


def infer_patient_col(df):
    possible = [
        "patient", "Patient", "PATIENT", "_PATIENT",
        "sample", "Sample", "SAMPLE",
        "sampleID", "SampleID",
        "barcode", "submitter_id", "case_id",
        "bcr_patient_barcode", "patient_id"
    ]

    cols_lower = {str(c).lower(): c for c in df.columns}
    for p in possible:
        if p.lower() in cols_lower:
            return cols_lower[p.lower()]

    for c in df.columns:
        try:
            if looks_like_tcga_values(df[c], min_frac=0.1):
                return c
        except Exception:
            pass

    return df.columns[0]


def infer_stage_cols(df):
    priority_patterns = [
        "stage_group",
        "stage_raw",
        "pathologic_stage",
        "ajcc_pathologic_stage",
        "clinical_stage",
        "tumor_stage",
        "pathologic_tumor_stage",
        "clinical_tumor_stage",
        "Converted_Stage_nature2012",
        "AJCC_Stage_nature2012",
        "stage"
    ]

    cols = list(df.columns)
    cols_lower = {str(c).lower(): c for c in cols}

    ordered = []

    for p in priority_patterns:
        if p.lower() in cols_lower and cols_lower[p.lower()] not in ordered:
            ordered.append(cols_lower[p.lower()])

    for c in cols:
        cl = str(c).lower()
        if "stage" in cl and c not in ordered:
            ordered.append(c)

    return ordered


def parse_stage_file(stage_file):
    try:
        df = read_table_safely(stage_file)
    except Exception as e:
        return pd.DataFrame(columns=["patient", "stage_group"]), {
            "file": str(stage_file),
            "status": "read_failed",
            "error": repr(e)[:500]
        }

    if df.empty or df.shape[1] < 2:
        return pd.DataFrame(columns=["patient", "stage_group"]), {
            "file": str(stage_file),
            "status": "empty_or_too_few_columns"
        }

    patient_col = infer_patient_col(df)
    stage_cols = infer_stage_cols(df)

    if not stage_cols:
        return pd.DataFrame(columns=["patient", "stage_group"]), {
            "file": str(stage_file),
            "status": "no_stage_column",
            "patient_col": str(patient_col),
            "columns_preview": list(map(str, df.columns[:30]))
        }

    best_out = pd.DataFrame(columns=["patient", "stage_group"])
    best_diag = None

    for stage_col in stage_cols:
        out = pd.DataFrame()
        out["patient"] = df[patient_col].map(tcga_patient_id)
        out["stage_group"] = df[stage_col].map(roman_stage_to_group)
        out = out.dropna(subset=["patient", "stage_group"])
        out = out.drop_duplicates(subset=["patient"])

        counts = out["stage_group"].value_counts().to_dict() if not out.empty else {}

        diag = {
            "file": str(stage_file),
            "status": "parsed",
            "patient_col": str(patient_col),
            "stage_col": str(stage_col),
            "n": int(out.shape[0]),
            "counts": counts
        }

        if set(counts.keys()) >= {"Early", "Late"}:
            return out, diag

        if out.shape[0] > best_out.shape[0]:
            best_out = out
            best_diag = diag

    if best_diag is None:
        best_diag = {
            "file": str(stage_file),
            "status": "no_parseable_stage_values",
            "patient_col": str(patient_col),
            "stage_cols": list(map(str, stage_cols))
        }

    return best_out, best_diag


def load_stage_groups(cancer_dir):
    cancer_dir = Path(cancer_dir)
    files = find_stage_files(cancer_dir)

    if not files:
        warnings.warn(f"No stage/clinical files found in {cancer_dir}")
        return pd.DataFrame(columns=["patient", "stage_group"])

    best_df = pd.DataFrame(columns=["patient", "stage_group"])
    best_diag = None

    for f in files:
        out, diag = parse_stage_file(f)

        counts = out["stage_group"].value_counts().to_dict() if not out.empty else {}

        print(f"[Stage candidate] {cancer_dir.name} | {Path(f).name}")
        print(f"  diag = {diag}")

        if out.shape[0] > best_df.shape[0]:
            best_df = out
            best_diag = diag

        if set(counts.keys()) >= {"Early", "Late"}:
            print(f"[Stage] {cancer_dir.name}: using {Path(f).name}, n={out.shape[0]}")
            print(out["stage_group"].value_counts(dropna=False).to_string())
            return out

    if best_df.empty:
        warnings.warn(f"No usable stage groups parsed in {cancer_dir}")
        return pd.DataFrame(columns=["patient", "stage_group"])

    warnings.warn(
        f"{cancer_dir.name}: no file produced both Early and Late. "
        f"Best file={best_diag}"
    )
    print(f"[Stage fallback] {cancer_dir.name}: using best single-group parsed file.")
    print(best_df["stage_group"].value_counts(dropna=False).to_string())

    return best_df


# ============================================================
# 5. MOATTERS-h AND BP SCORE
# ============================================================

def filter_gene_sets_by_readiness(gene_sets, ge_genes, min_matched=10):
    ge_genes = set(map(str.upper, ge_genes))

    rows = []
    ready = {}

    for term, genes in gene_sets.items():
        genes_clean = [
            clean_gene_symbol(g)
            for g in genes
            if clean_gene_symbol(g)
        ]

        genes_clean = sorted(set(genes_clean))
        matched = sorted(set(genes_clean).intersection(ge_genes))

        n_defined = len(genes_clean)
        n_matched = len(matched)
        matched_fraction = n_matched / n_defined if n_defined > 0 else 0.0

        if n_matched >= min_matched:
            cls = "observation_ready"
            ready[term] = matched
        elif n_matched > 0:
            cls = "low_resolution"
        else:
            cls = "near_unobservable"

        rows.append({
            "term": term,
            "n_defined": n_defined,
            "n_matched": n_matched,
            "matched_fraction": matched_fraction,
            "readiness_class": cls
        })

    readiness_df = pd.DataFrame(rows)
    return ready, readiness_df


def zscore_rows(mat):
    mean = mat.mean(axis=1, skipna=True)
    std = mat.std(axis=1, skipna=True).replace(0, np.nan)
    z = mat.sub(mean, axis=0).div(std, axis=0)
    return z


def compute_bp_scores(ge, ready_gene_sets, method="mean_z_gene"):
    if method != "mean_z_gene":
        raise ValueError(f"Unknown score method: {method}")

    zge = zscore_rows(ge)

    scores = {}
    for term, genes in ready_gene_sets.items():
        genes2 = [g for g in genes if g in zge.index]
        if len(genes2) == 0:
            continue

        scores[term] = zge.loc[genes2].mean(axis=0, skipna=True)

    return pd.DataFrame(scores).T


# ============================================================
# 6. STAGE D TEST
# ============================================================

def stage_test_for_bp(bp_scores, stage_df):
    common = sorted(set(bp_scores.columns).intersection(set(stage_df["patient"])))

    if len(common) < MIN_PATIENTS_TOTAL:
        return pd.DataFrame(), common

    st = stage_df.set_index("patient").loc[common, "stage_group"]

    early_patients = st[st == "Early"].index.tolist()
    late_patients = st[st == "Late"].index.tolist()

    if len(early_patients) < MIN_PATIENTS_PER_GROUP or len(late_patients) < MIN_PATIENTS_PER_GROUP:
        return pd.DataFrame(), common

    rows = []

    for term in bp_scores.index:
        x_early = bp_scores.loc[term, early_patients].dropna().astype(float)
        x_late = bp_scores.loc[term, late_patients].dropna().astype(float)

        if len(x_early) < MIN_PATIENTS_PER_GROUP or len(x_late) < MIN_PATIENTS_PER_GROUP:
            continue

        try:
            stat, p = mannwhitneyu(x_late, x_early, alternative="two-sided")
        except Exception:
            p = np.nan

        mu_early = float(np.nanmean(x_early))
        mu_late = float(np.nanmean(x_late))
        delta = mu_late - mu_early

        rows.append({
            "term": term,
            "n_early": int(len(x_early)),
            "n_late": int(len(x_late)),
            "mu_early": mu_early,
            "mu_late": mu_late,
            "delta_late_minus_early": delta,
            "p_stage": p,
            "D_stage": safe_neglog10(p),
            "stage_trend": "late_high" if delta > 0 else "late_low"
        })

    res = pd.DataFrame(rows)

    if not res.empty:
        res = res.sort_values(
            ["D_stage", "delta_late_minus_early"],
            ascending=[False, False]
        )

    return res, common


# ============================================================
# 7. BP-BP NETWORK
# ============================================================

def compute_bp_corr_network(
    bp_scores_subset,
    patients,
    group_name,
    corr_abs_threshold=0.35,
    p_threshold=0.05
):
    patients = [p for p in patients if p in bp_scores_subset.columns]

    if len(patients) < 10:
        G = nx.Graph()
        return pd.DataFrame(), pd.DataFrame(), G

    X = bp_scores_subset.loc[:, patients].T
    terms = list(X.columns)

    edges = []
    G = nx.Graph()
    G.add_nodes_from(terms)

    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            a = X[terms[i]].astype(float)
            b = X[terms[j]].astype(float)
            ok = a.notna() & b.notna()

            if ok.sum() < 10:
                continue

            if CORR_METHOD == "spearman":
                r, p = spearmanr(a[ok], b[ok])
            else:
                r = np.corrcoef(a[ok], b[ok])[0, 1]
                p = np.nan

            if pd.isna(r):
                continue

            if abs(r) >= corr_abs_threshold and (pd.isna(p) or p <= p_threshold):
                edges.append({
                    "group": group_name,
                    "bp1": terms[i],
                    "bp2": terms[j],
                    "corr": float(r),
                    "p_corr": p,
                    "abs_corr": float(abs(r))
                })
                G.add_edge(
                    terms[i],
                    terms[j],
                    weight=float(abs(r)),
                    corr=float(r)
                )

    edges_df = pd.DataFrame(edges)

    if len(G.nodes) == 0:
        return edges_df, pd.DataFrame(), G

    deg = dict(G.degree())
    weighted_deg = dict(G.degree(weight="weight"))

    if len(G.nodes) > 1:
        btw = nx.betweenness_centrality(G, weight="weight")
    else:
        btw = {n: 0 for n in G.nodes}

    metrics = []
    for n in G.nodes:
        metrics.append({
            "group": group_name,
            "term": n,
            "degree": deg.get(n, 0),
            "weighted_degree": weighted_deg.get(n, 0.0),
            "betweenness": btw.get(n, 0.0)
        })

    metrics_df = pd.DataFrame(metrics).sort_values(
        ["degree", "weighted_degree", "betweenness"],
        ascending=False
    )

    return edges_df, metrics_df, G


def graph_summary(G, group_name):
    n = G.number_of_nodes()
    m = G.number_of_edges()

    density = nx.density(G) if n > 1 else 0
    components = nx.number_connected_components(G) if n > 0 else 0

    largest_cc = 0
    if n > 0:
        largest_cc = max((len(c) for c in nx.connected_components(G)), default=0)

    avg_degree = np.mean([d for _, d in G.degree()]) if n > 0 else 0

    return {
        "group": group_name,
        "n_nodes": int(n),
        "n_edges": int(m),
        "density": float(density),
        "n_components": int(components),
        "largest_component_size": int(largest_cc),
        "avg_degree": float(avg_degree)
    }


# ============================================================
# 8. GENE OVERLAP
# ============================================================

def compute_gene_overlap(selected_terms, ready_gene_sets):
    rows = []

    for i in range(len(selected_terms)):
        for j in range(i + 1, len(selected_terms)):
            t1 = selected_terms[i]
            t2 = selected_terms[j]

            g1 = set(ready_gene_sets.get(t1, []))
            g2 = set(ready_gene_sets.get(t2, []))

            inter = len(g1 & g2)
            union = len(g1 | g2)

            jac = inter / union if union > 0 else 0
            overlap_coef = inter / min(len(g1), len(g2)) if min(len(g1), len(g2)) > 0 else 0

            rows.append({
                "bp1": t1,
                "bp2": t2,
                "n_gene_bp1": len(g1),
                "n_gene_bp2": len(g2),
                "n_shared_genes": inter,
                "jaccard": jac,
                "overlap_coefficient": overlap_coef
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["jaccard", "n_shared_genes"],
        ascending=False
    )


# ============================================================
# 9. OPTIONAL PPI
# ============================================================

def try_load_ppi_edges(ppi_dir):
    ppi_dir = Path(ppi_dir)

    if not ppi_dir.exists():
        print("[PPI] PPI directory not found. Skipping.")
        return pd.DataFrame(columns=["gene1", "gene2"])

    files = (
        list(ppi_dir.glob("*.csv")) +
        list(ppi_dir.glob("*.tsv")) +
        list(ppi_dir.glob("*.txt"))
    )

    all_edges = []

    edge_col_pairs = [
        ("protein1", "protein2"),
        ("protein_a", "protein_b"),
        ("gene1", "gene2"),
        ("gene_a", "gene_b"),
        ("source", "target"),
        ("node1", "node2"),
        ("interactor_a", "interactor_b"),
        ("interactor1", "interactor2"),
        ("protein_1", "protein_2"),
        ("preferred_name_a", "preferred_name_b"),
        ("preferred_name_1", "preferred_name_2"),
    ]

    skip_keywords = [
        "sequence",
        "sequences",
        "positive_protein_sequences",
        "negative_protein_sequences",
        "readme"
    ]

    for f in files:
        fname = f.name.lower()

        if any(k in fname for k in skip_keywords):
            print(f"[PPI] Skipping likely sequence/readme file: {f.name}")
            continue

        try:
            df = read_table_safely(f)
        except Exception as e:
            print(f"[PPI] Could not read {f.name}: {e}")
            continue

        cols_lower = {str(c).lower().strip(): c for c in df.columns}

        found = None
        for a, b in edge_col_pairs:
            if a in cols_lower and b in cols_lower:
                found = (cols_lower[a], cols_lower[b])
                break

        if found is None:
            print(f"[PPI] {f.name}: no clear PPI edge columns detected. Skipping.")
            print(f"      columns preview: {list(map(str, df.columns[:10]))}")
            continue

        c1, c2 = found

        tmp = pd.DataFrame({
            "gene1": df[c1].map(clean_gene_symbol),
            "gene2": df[c2].map(clean_gene_symbol),
            "source_file": f.name
        }).dropna()

        tmp = tmp[tmp["gene1"] != tmp["gene2"]]

        if tmp.empty:
            print(f"[PPI] {f.name}: no usable edges after cleaning.")
            continue

        all_edges.append(tmp)
        print(f"[PPI] Loaded PPI edge-list: {f.name}, edges={tmp.shape[0]}")

    if not all_edges:
        print("[PPI] No usable PPI edge-list detected. PPI support will be skipped.")
        return pd.DataFrame(columns=["gene1", "gene2"])

    edges = pd.concat(all_edges, ignore_index=True)

    edges["a"] = edges[["gene1", "gene2"]].min(axis=1)
    edges["b"] = edges[["gene1", "gene2"]].max(axis=1)

    edges = edges.drop_duplicates(subset=["a", "b"])
    edges = edges[["a", "b"]].rename(columns={"a": "gene1", "b": "gene2"})

    return edges


def compute_ppi_support(selected_terms, ready_gene_sets, ppi_edges):
    if ppi_edges is None or ppi_edges.empty:
        return pd.DataFrame()

    edge_set = set()

    for _, r in ppi_edges.iterrows():
        a = clean_gene_symbol(r["gene1"])
        b = clean_gene_symbol(r["gene2"])
        if a and b and a != b:
            edge_set.add(tuple(sorted((a, b))))

    rows = []

    for i in range(len(selected_terms)):
        for j in range(i + 1, len(selected_terms)):
            t1 = selected_terms[i]
            t2 = selected_terms[j]

            g1 = set(ready_gene_sets.get(t1, []))
            g2 = set(ready_gene_sets.get(t2, []))

            count = 0
            for a in g1:
                for b in g2:
                    if a == b:
                        continue
                    if tuple(sorted((a, b))) in edge_set:
                        count += 1

            rows.append({
                "bp1": t1,
                "bp2": t2,
                "ppi_edges_between_bp_genes": count
            })

    return pd.DataFrame(rows).sort_values(
        "ppi_edges_between_bp_genes",
        ascending=False
    )


# ============================================================
# 10. FIGURES
# ============================================================

def plot_stage_scatter(stage_res, cancer_code, out_path):
    if stage_res.empty:
        return

    plt.figure(figsize=(8, 6))

    plt.scatter(
        stage_res["delta_late_minus_early"],
        stage_res["D_stage"],
        s=12,
        alpha=0.7
    )

    plt.axhline(D_THRESHOLD, linestyle="--", linewidth=1)
    plt.axvline(0, linestyle="--", linewidth=1)

    plt.xlabel("BP activity change: Stage III/IV minus Stage I/II")
    plt.ylabel("D_stage = -log10(p)")
    plt.title(f"{cancer_code}: Stage-associated BP shifts")

    top = stage_res.head(10)
    for _, r in top.iterrows():
        plt.text(
            r["delta_late_minus_early"],
            r["D_stage"],
            str(r["term"])[:40],
            fontsize=7
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_network(G, title, out_path):
    if G.number_of_nodes() == 0:
        return

    plt.figure(figsize=(10, 8))

    pos = nx.spring_layout(G, seed=1, k=0.8)

    degrees = dict(G.degree())
    node_sizes = [80 + 30 * degrees.get(n, 0) for n in G.nodes()]

    weights = [G[u][v].get("weight", 0.5) for u, v in G.edges()]
    widths = [0.5 + 2.0 * w for w in weights]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.85)
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.45)
    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: n[:25] for n in G.nodes()},
        font_size=7
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_heatmap_selected_bp(bp_scores, selected_terms, stage_df, patients, cancer_code, out_path):
    patients = [p for p in patients if p in bp_scores.columns]
    st = stage_df.set_index("patient").loc[patients, "stage_group"]

    early = st[st == "Early"].index.tolist()
    late = st[st == "Late"].index.tolist()
    ordered_patients = early + late

    if len(ordered_patients) == 0 or len(selected_terms) == 0:
        return

    data = bp_scores.loc[selected_terms, ordered_patients].copy()

    row_mean = data.mean(axis=1)
    row_std = data.std(axis=1).replace(0, np.nan)
    z = data.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)

    plt.figure(figsize=(12, max(6, len(selected_terms) * 0.25)))
    plt.imshow(z.values, aspect="auto", interpolation="nearest")
    plt.colorbar(label="BP score z")

    plt.yticks(range(len(selected_terms)), [t[:50] for t in selected_terms], fontsize=6)
    plt.xticks([])

    if len(early) > 0:
        plt.axvline(len(early) - 0.5, linestyle="--", linewidth=1)

    plt.title(f"{cancer_code}: selected BP profile, Stage I/II | Stage III/IV")
    plt.xlabel("Patients ordered by stage")
    plt.ylabel("Selected BP")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================
# 11. PER-CANCER ANALYSIS
# ============================================================

def run_one_cancer(cancer_folder, gene_sets, ppi_edges=None):
    cancer_dir = UCSC_XENA_DIR / cancer_folder
    cancer_code = extract_cancer_code(cancer_folder)

    print("\n" + "=" * 100)
    print(f"[Cancer] {cancer_folder} / {cancer_code}")

    ge_path = cancer_dir / "GE.tsv"

    if not ge_path.exists():
        print(f"[Skip] GE.tsv not found: {ge_path}")
        return None

    cancer_out = OUT_DIR / cancer_code
    cancer_out.mkdir(parents=True, exist_ok=True)

    print("[Load] GE matrix...")
    ge = load_ge_matrix(ge_path)

    ge = ge.loc[[g for g in ge.index if g not in ["?", "", None]], :]

    print(f"[GE] genes x patients = {ge.shape}")
    print(f"[GE] first genes: {list(ge.index[:5])}")
    print(f"[GE] first patients: {list(ge.columns[:5])}")

    stage_df = load_stage_groups(cancer_dir)

    if stage_df.empty:
        print("[Skip] No usable stage groups.")
        return None

    common_patients = sorted(set(ge.columns).intersection(stage_df["patient"]))

    print(f"[Match] GE-stage common patients = {len(common_patients)}")

    if len(common_patients) < MIN_PATIENTS_TOTAL:
        print("[Skip] Too few common patients.")
        return None

    tmp_stage = stage_df[stage_df["patient"].isin(common_patients)].copy()

    print("[Stage after GE matching]")
    print(tmp_stage["stage_group"].value_counts(dropna=False).to_string())

    if set(tmp_stage["stage_group"].dropna().unique()) < {"Early", "Late"}:
        print("[Skip] Stage groups do not contain both Early and Late after GE matching.")
        return None

    ge = ge.loc[:, common_patients]
    stage_df = tmp_stage.copy()

    print("[MOATTERS-h] Filtering gene sets by matched genes...")
    ready_gene_sets, readiness_df = filter_gene_sets_by_readiness(
        gene_sets,
        ge.index,
        min_matched=MIN_MATCHED_GENES
    )

    readiness_df.to_csv(
        cancer_out / f"{cancer_code}_BP_readiness.csv",
        index=False
    )

    print(readiness_df["readiness_class"].value_counts().to_string())

    if len(ready_gene_sets) == 0:
        print("[Skip] No observation-ready gene sets.")
        return None

    print("[Score] Computing BP scores...")
    bp_scores = compute_bp_scores(
        ge,
        ready_gene_sets,
        method=SCORE_METHOD
    )

    print(f"[BP score] terms x patients = {bp_scores.shape}")

    print("[D_stage] Testing Stage I/II vs Stage III/IV...")
    stage_res, common = stage_test_for_bp(bp_scores, stage_df)

    if stage_res.empty:
        print("[Skip] Stage test returned empty.")
        return None

    stage_res.to_csv(
        cancer_out / f"{cancer_code}_DStage_BP_results.csv",
        index=False
    )

    highD = stage_res[stage_res["D_stage"] >= D_THRESHOLD].copy()

    highD.to_csv(
        cancer_out / f"{cancer_code}_DStage_highD_BP.csv",
        index=False
    )

    print(f"[D_stage] tested terms = {stage_res.shape[0]}")
    print(f"[D_stage] high-D terms D>={D_THRESHOLD} = {highD.shape[0]}")
    print("[Top BP]")
    print(
        stage_res.head(10)[
            ["term", "D_stage", "p_stage", "delta_late_minus_early", "stage_trend"]
        ].to_string(index=False)
    )

    if MAKE_FIGURES:
        plot_stage_scatter(
            stage_res,
            cancer_code,
            cancer_out / f"{cancer_code}_stage_shift_scatter.png"
        )

    selected = stage_res.head(TOP_N_BP_FOR_NETWORK)["term"].tolist()
    selected = [t for t in selected if t in bp_scores.index]

    selected_df = stage_res[stage_res["term"].isin(selected)].copy()
    selected_df.to_csv(
        cancer_out / f"{cancer_code}_selected_BP_for_network.csv",
        index=False
    )

    st = stage_df.set_index("patient").loc[common_patients, "stage_group"]

    early_patients = st[st == "Early"].index.tolist()
    late_patients = st[st == "Late"].index.tolist()

    bp_sel = bp_scores.loc[selected, common_patients]

    all_edges, all_metrics, G_all = compute_bp_corr_network(
        bp_sel,
        common_patients,
        "all_patients",
        CORR_ABS_THRESHOLD,
        CORR_P_THRESHOLD
    )

    early_edges, early_metrics, G_early = compute_bp_corr_network(
        bp_sel,
        early_patients,
        "stage_I_II",
        CORR_ABS_THRESHOLD,
        CORR_P_THRESHOLD
    )

    late_edges, late_metrics, G_late = compute_bp_corr_network(
        bp_sel,
        late_patients,
        "stage_III_IV",
        CORR_ABS_THRESHOLD,
        CORR_P_THRESHOLD
    )

    edges_df = pd.concat(
        [all_edges, early_edges, late_edges],
        ignore_index=True
    )

    metrics_df = pd.concat(
        [all_metrics, early_metrics, late_metrics],
        ignore_index=True
    )

    edges_df.to_csv(
        cancer_out / f"{cancer_code}_BP_correlation_network_edges.csv",
        index=False
    )

    metrics_df.to_csv(
        cancer_out / f"{cancer_code}_BP_correlation_network_node_metrics.csv",
        index=False
    )

    net_summary = pd.DataFrame([
        graph_summary(G_all, "all_patients"),
        graph_summary(G_early, "stage_I_II"),
        graph_summary(G_late, "stage_III_IV"),
    ])

    net_summary.to_csv(
        cancer_out / f"{cancer_code}_BP_network_summary.csv",
        index=False
    )

    print("[Network summary]")
    print(net_summary.to_string(index=False))

    if MAKE_FIGURES:
        plot_network(
            G_all,
            f"{cancer_code}: BP network, all patients",
            cancer_out / f"{cancer_code}_network_all.png"
        )
        plot_network(
            G_early,
            f"{cancer_code}: BP network, Stage I/II",
            cancer_out / f"{cancer_code}_network_stage_I_II.png"
        )
        plot_network(
            G_late,
            f"{cancer_code}: BP network, Stage III/IV",
            cancer_out / f"{cancer_code}_network_stage_III_IV.png"
        )
        plot_heatmap_selected_bp(
            bp_scores,
            selected,
            stage_df,
            common_patients,
            cancer_code,
            cancer_out / f"{cancer_code}_selected_BP_stage_heatmap.png"
        )

    overlap_df = compute_gene_overlap(selected, ready_gene_sets)

    overlap_df.to_csv(
        cancer_out / f"{cancer_code}_selected_BP_gene_overlap.csv",
        index=False
    )

    if ppi_edges is not None and not ppi_edges.empty:
        ppi_support_df = compute_ppi_support(
            selected,
            ready_gene_sets,
            ppi_edges
        )
        ppi_support_df.to_csv(
            cancer_out / f"{cancer_code}_selected_BP_PPI_support.csv",
            index=False
        )
    else:
        ppi_support_df = pd.DataFrame()

    summary = {
        "cancer_folder": cancer_folder,
        "cancer_code": cancer_code,
        "n_ge_genes": int(ge.shape[0]),
        "n_ge_patients": int(ge.shape[1]),
        "n_stage_patients": int(stage_df.shape[0]),
        "n_common_patients": int(len(common_patients)),
        "n_early": int(len(early_patients)),
        "n_late": int(len(late_patients)),
        "n_gene_sets_total": int(len(gene_sets)),
        "n_observation_ready_gene_sets": int(len(ready_gene_sets)),
        "n_tested_bp": int(stage_res.shape[0]),
        "n_highD_bp": int(highD.shape[0]),
        "max_D_stage": float(stage_res["D_stage"].max()),
        "top_BP": str(stage_res.iloc[0]["term"]),
        "top_BP_D_stage": float(stage_res.iloc[0]["D_stage"]),
        "top_BP_delta_late_minus_early": float(stage_res.iloc[0]["delta_late_minus_early"]),
        "network_edges_all": int(G_all.number_of_edges()),
        "network_edges_early": int(G_early.number_of_edges()),
        "network_edges_late": int(G_late.number_of_edges()),
        "network_density_all": float(nx.density(G_all)) if G_all.number_of_nodes() > 1 else 0.0,
        "network_density_early": float(nx.density(G_early)) if G_early.number_of_nodes() > 1 else 0.0,
        "network_density_late": float(nx.density(G_late)) if G_late.number_of_nodes() > 1 else 0.0,
    }

    with open(
        cancer_out / f"{cancer_code}_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    return summary


# ============================================================
# 12. MAIN
# ============================================================

def main():
    print("=" * 100)
    print("MOATTERS Stage BP Attractor Pilot V3")
    print("=" * 100)

    if not GMT_PATH.exists():
        raise FileNotFoundError(f"GMT not found: {GMT_PATH}")

    print(f"[GMT] Loading: {GMT_PATH}")
    gene_sets = load_gmt(GMT_PATH)
    print(f"[GMT] n gene sets = {len(gene_sets)}")

    print("[PPI] Trying to load true PPI edge-list...")
    ppi_edges = try_load_ppi_edges(PPI_DIR)
    print(f"[PPI] usable edges = {ppi_edges.shape[0]}")

    all_summaries = []

    for cancer_folder in CANCER_FOLDERS:
        try:
            summary = run_one_cancer(
                cancer_folder,
                gene_sets,
                ppi_edges=ppi_edges
            )

            if summary is not None:
                all_summaries.append(summary)

        except Exception as e:
            print(f"[ERROR] {cancer_folder}: {repr(e)}")
            import traceback
            traceback.print_exc()

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(
            OUT_DIR / "MOATTERS_stage_BP_attractor_pilot_summary.csv",
            index=False
        )

        print("\n" + "=" * 100)
        print("[DONE] Overall summary:")
        print(summary_df.to_string(index=False))
        print(f"\nOutput folder:\n{OUT_DIR}")

    else:
        print("[DONE] No successful cancer analyses.")


if __name__ == "__main__":
    main()
