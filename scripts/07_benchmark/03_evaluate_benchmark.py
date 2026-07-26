# Public release: MOATTERS reproducibility code
# MOATTERS = Modularity, Observability, Auditability, and Traceability
# for Task-conditioned Evidence Reconstruction into Patient States.
# Generated from the locked analysis scripts; scientific logic was preserved.
from pathlib import Path
from moatters.config import data_path, output_path
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# Stage 5B-3 — Final controlled benchmark
# MOATTERS BP-state vs GSVA vs Pathifier
#
# Run this entire code in ONE Jupyter Python cell.
#
# Fairness controls:
# - same endpoint-specific patient set across all three methods
# - complete-case intersection before fold generation
# - identical repeated 20 x 5 stratified folds
# - training-fold-only standardization
# - training-fold-only class centroids
# - no method-specific tuning
#
# Interpretation boundary:
# This is a locked representation/scoring benchmark, not a fully
# nested feature-selection benchmark.
# ============================================================

INPUT_DIR = output_path(r"MOATTERS_STAGE5B_BENCHMARK_INPUTS")

SCORE_DIR = output_path(r"MOATTERS_STAGE5B_GSVA_PATHIFIER_SCORES")

OUT = output_path(r"MOATTERS_STAGE5B_BENCHMARK_EVALUATION")

N_SPLITS = 5
N_REPEATS = 20
SEED = 20260725

METHOD_ORDER = [
    "MOATTERS_state",
    "GSVA",
    "Pathifier",
]

ENDPOINT_ORDER = [
    "stage_late",
    "ER_negative",
    "PAM50_basal",
    "PAM50_luminal",
]

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)
(OUT / "text").mkdir(exist_ok=True)
(OUT / "audit").mkdir(exist_ok=True)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def patient_id(value):
    return str(value).strip()[:12]


def safe_d(p):
    p = max(float(p), np.nextafter(0, 1))
    return -math.log10(p)


def cosine_similarity_rows(X, centroid):
    numerator = X @ centroid
    denominator = (
        np.linalg.norm(X, axis=1)
        * np.linalg.norm(centroid)
    )

    return np.divide(
        numerator,
        denominator,
        out=np.full(X.shape[0], np.nan, dtype=float),
        where=denominator > 0,
    )


def centroid_margin(
    X_train,
    y_train,
    X_test,
):
    centroid_0 = X_train[y_train == 0].mean(axis=0)
    centroid_1 = X_train[y_train == 1].mean(axis=0)

    similarity_0 = cosine_similarity_rows(
        X_test,
        centroid_0,
    )

    similarity_1 = cosine_similarity_rows(
        X_test,
        centroid_1,
    )

    return similarity_1 - similarity_0


def rank_biserial_and_p(
    scores,
    labels,
):
    group_1 = scores[labels == 1]
    group_0 = scores[labels == 0]

    test = mannwhitneyu(
        group_1,
        group_0,
        alternative="two-sided",
    )

    rank_biserial = (
        2 * test.statistic
        / (len(group_1) * len(group_0))
        - 1
    )

    return (
        float(rank_biserial),
        float(test.pvalue),
    )


def load_score_matrix(path):
    df = pd.read_csv(
        path,
        index_col=0,
    )

    df.index = [
        patient_id(x)
        for x in df.index
    ]

    df = df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # One primary-tumor sample per patient is expected.
    # The mean is a defensive fallback if duplicate patient IDs occur.
    df = df.groupby(
        level=0,
        sort=False,
    ).mean()

    return df


def normalize_text(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
    )


def build_endpoints(clinical):
    # Stage
    stage_text = normalize_text(
        clinical["StageGroup"]
    )

    stage = pd.Series(
        np.nan,
        index=clinical.index,
        dtype=float,
    )

    stage[
        stage_text.str.contains(
            "early|stage i|stage ii",
            regex=True,
            na=False,
        )
    ] = 0

    stage[
        stage_text.str.contains(
            "late|stage iii|stage iv",
            regex=True,
            na=False,
        )
    ] = 1

    # ER: positive=0, negative=1
    er_col = (
        "ER_status__breast_carcinoma_"
        "estrogen_receptor_status__clean"
    )

    er_text = normalize_text(
        clinical[er_col]
    )

    er = pd.Series(
        np.nan,
        index=clinical.index,
        dtype=float,
    )

    er[
        er_text.str.contains(
            "positive",
            na=False,
        )
    ] = 0

    er[
        er_text.str.contains(
            "negative",
            na=False,
        )
    ] = 1

    # PAM50
    pam_text = normalize_text(
        clinical["PAM50_simplified"]
    )

    invalid_pam = pam_text.isin(
        [
            "nan",
            "none",
            "",
            "unknown",
            "not available",
            "na",
        ]
    )

    basal = pd.Series(
        np.nan,
        index=clinical.index,
        dtype=float,
    )

    basal[~invalid_pam] = (
        pam_text[~invalid_pam]
        .str.contains(
            "basal",
            na=False,
        )
        .astype(int)
    )

    luminal = pd.Series(
        np.nan,
        index=clinical.index,
        dtype=float,
    )

    luminal[~invalid_pam] = (
        pam_text[~invalid_pam]
        .str.contains(
            "luminal|luma|lumb",
            regex=True,
            na=False,
        )
        .astype(int)
    )

    return {
        "stage_late": stage,
        "ER_negative": er,
        "PAM50_basal": basal,
        "PAM50_luminal": luminal,
    }


def holm_adjust(p_values):
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)

    running_max = 0.0
    m = len(p_values)

    for rank, idx in enumerate(order):
        value = (
            (m - rank)
            * p_values[idx]
        )

        running_max = max(
            running_max,
            value,
        )

        adjusted[idx] = min(
            running_max,
            1.0,
        )

    return adjusted


# ------------------------------------------------------------
# Load and audit inputs
# ------------------------------------------------------------

required_files = [
    INPUT_DIR / "locked_MOATTERS_M1_M7_patient_representation.csv",
    INPUT_DIR / "locked_benchmark_endpoints.csv",
    SCORE_DIR / "GSVA_scores_sample_by_BP.csv",
    SCORE_DIR / "Pathifier_scores_sample_by_BP.csv",
    SCORE_DIR / "analysis_manifest.json",
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    raise FileNotFoundError(
        "Required input file(s) missing:\n"
        + "\n".join(missing_files)
    )

state_raw = pd.read_csv(
    INPUT_DIR
    / "locked_MOATTERS_M1_M7_patient_representation.csv"
)

required_aido_columns = [
    "patient_id",
    *[f"M{i}" for i in range(1, 8)],
]

missing_aido_columns = [
    col
    for col in required_aido_columns
    if col not in state_raw.columns
]

if missing_aido_columns:
    raise ValueError(
        "MOATTERS representation is missing columns: "
        + ", ".join(missing_aido_columns)
    )

state_raw.index = (
    state_raw["patient_id"]
    .astype(str)
    .map(patient_id)
)

state_X = (
    state_raw[
        [f"M{i}" for i in range(1, 8)]
    ]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
)

state_X = state_X.groupby(
    level=0,
    sort=False,
).mean()

gsva_X = load_score_matrix(
    SCORE_DIR
    / "GSVA_scores_sample_by_BP.csv"
)

pathifier_X = load_score_matrix(
    SCORE_DIR
    / "Pathifier_scores_sample_by_BP.csv"
)

clinical = pd.read_csv(
    INPUT_DIR
    / "locked_benchmark_endpoints.csv"
)

clinical.index = (
    clinical["patient_id"]
    .astype(str)
    .map(patient_id)
)

clinical = clinical[
    ~clinical.index.duplicated(
        keep="first"
    )
]

with open(
    SCORE_DIR / "analysis_manifest.json",
    "r",
    encoding="utf-8",
) as handle:
    score_manifest = json.load(handle)

representations = {
    "MOATTERS_state": state_X,
    "GSVA": gsva_X,
    "Pathifier": pathifier_X,
}

endpoint_vectors = build_endpoints(
    clinical
)

input_audit = {
    "MOATTERS_shape": list(state_X.shape),
    "GSVA_shape": list(gsva_X.shape),
    "Pathifier_shape": list(pathifier_X.shape),
    "clinical_shape": list(clinical.shape),
    "GSVA_Pathifier_column_identity": (
        list(gsva_X.columns)
        == list(pathifier_X.columns)
    ),
    "GSVA_Pathifier_patient_overlap": int(
        len(
            set(gsva_X.index)
            & set(pathifier_X.index)
        )
    ),
    "score_manifest": score_manifest,
}

(
    OUT
    / "audit"
    / "input_audit.json"
).write_text(
    json.dumps(
        input_audit,
        indent=2,
    ),
    encoding="utf-8",
)

print("Input audit:")
print("MOATTERS:", state_X.shape)
print("GSVA:", gsva_X.shape)
print("Pathifier:", pathifier_X.shape)
print("Clinical:", clinical.shape)


# ------------------------------------------------------------
# Repeated-CV benchmark
# ------------------------------------------------------------

summary_rows = []
fold_rows = []
patient_rows = []
denominator_rows = []

for endpoint_name in ENDPOINT_ORDER:

    y_all = endpoint_vectors[
        endpoint_name
    ]

    # Start with endpoint-eligible patients.
    common_patients = set(
        y_all.dropna().index
    )

    # Require presence in all three methods.
    for representation in representations.values():
        common_patients &= set(
            representation.index
        )

    common_patients = sorted(
        common_patients
    )

    if not common_patients:
        raise RuntimeError(
            f"No common patients for {endpoint_name}."
        )

    # Require complete features in all methods before making folds,
    # so every method uses exactly the same patient set.
    complete_mask = pd.Series(
        True,
        index=common_patients,
    )

    for representation in representations.values():

        current = (
            representation
            .loc[common_patients]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .notna()
            .all(axis=1)
        )

        complete_mask &= current

    final_patients = complete_mask[
        complete_mask
    ].index.tolist()

    y = (
        y_all.loc[final_patients]
        .astype(int)
    )

    class_counts = (
        y.value_counts()
        .sort_index()
    )

    if (
        y.nunique() != 2
        or class_counts.min() < N_SPLITS
    ):
        raise RuntimeError(
            f"Endpoint {endpoint_name} does not have "
            "enough observations in both classes."
        )

    denominator_rows.append(
        {
            "endpoint": endpoint_name,
            "n_common_complete_patients": int(
                len(final_patients)
            ),
            "n_class_0": int(
                (y == 0).sum()
            ),
            "n_class_1": int(
                (y == 1).sum()
            ),
        }
    )

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=SEED,
    )

    fold_indices = list(
        splitter.split(
            np.zeros(
                len(final_patients)
            ),
            y.to_numpy(),
        )
    )

    # Save one explicit fold assignment table per endpoint.
    fold_assignment_rows = []

    for fold_number, (train_idx, test_idx) in enumerate(
        fold_indices,
        start=1,
    ):
        repeat_number = (
            (fold_number - 1)
            // N_SPLITS
            + 1
        )

        split_number = (
            (fold_number - 1)
            % N_SPLITS
            + 1
        )

        for idx in test_idx:
            fold_assignment_rows.append(
                {
                    "endpoint": endpoint_name,
                    "patient_id": final_patients[idx],
                    "repeat": repeat_number,
                    "fold": split_number,
                    "fold_iteration": fold_number,
                    "y": int(y.iloc[idx]),
                }
            )

    pd.DataFrame(
        fold_assignment_rows
    ).to_csv(
        OUT
        / "audit"
        / f"fold_assignments_{endpoint_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for method_name in METHOD_ORDER:

        X_df = (
            representations[method_name]
            .loc[final_patients]
            .astype(float)
        )

        X = X_df.to_numpy()
        labels = y.to_numpy()

        out_of_fold_rows = []
        method_fold_aucs = []

        for fold_number, (
            train_idx,
            test_idx,
        ) in enumerate(
            fold_indices,
            start=1,
        ):

            repeat_number = (
                (fold_number - 1)
                // N_SPLITS
                + 1
            )

            split_number = (
                (fold_number - 1)
                % N_SPLITS
                + 1
            )

            scaler = StandardScaler(
                with_mean=True,
                with_std=True,
            )

            X_train = scaler.fit_transform(
                X[train_idx]
            )

            X_test = scaler.transform(
                X[test_idx]
            )

            # Defensive replacement of zero-variance features.
            X_train = np.nan_to_num(
                X_train,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            X_test = np.nan_to_num(
                X_test,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            heldout_scores = centroid_margin(
                X_train=X_train,
                y_train=labels[train_idx],
                X_test=X_test,
            )

            directional_auc = roc_auc_score(
                labels[test_idx],
                heldout_scores,
            )

            invariant_auc = max(
                directional_auc,
                1 - directional_auc,
            )

            method_fold_aucs.append(
                invariant_auc
            )

            fold_rows.append(
                {
                    "endpoint": endpoint_name,
                    "method": method_name,
                    "repeat": repeat_number,
                    "fold": split_number,
                    "fold_iteration": fold_number,
                    "n_train": int(
                        len(train_idx)
                    ),
                    "n_test": int(
                        len(test_idx)
                    ),
                    "directional_AUC": float(
                        directional_auc
                    ),
                    "orientation_invariant_AUC": float(
                        invariant_auc
                    ),
                }
            )

            for local_idx, score in zip(
                test_idx,
                heldout_scores,
            ):
                out_of_fold_rows.append(
                    {
                        "patient_id": final_patients[
                            local_idx
                        ],
                        "endpoint": endpoint_name,
                        "method": method_name,
                        "repeat": repeat_number,
                        "fold": split_number,
                        "fold_iteration": fold_number,
                        "y": int(
                            labels[local_idx]
                        ),
                        "heldout_score": float(
                            score
                        ),
                    }
                )

        out_of_fold = pd.DataFrame(
            out_of_fold_rows
        )

        aggregated = (
            out_of_fold
            .groupby(
                "patient_id",
                as_index=False,
            )
            .agg(
                y=("y", "first"),
                heldout_score=(
                    "heldout_score",
                    "mean",
                ),
                n_heldout_predictions=(
                    "heldout_score",
                    "size",
                ),
            )
        )

        aggregate_directional_auc = roc_auc_score(
            aggregated["y"],
            aggregated["heldout_score"],
        )

        aggregate_invariant_auc = max(
            aggregate_directional_auc,
            1 - aggregate_directional_auc,
        )

        rank_biserial, p_value = (
            rank_biserial_and_p(
                aggregated[
                    "heldout_score"
                ].to_numpy(),
                aggregated["y"].to_numpy(),
            )
        )

        summary_rows.append(
            {
                "endpoint": endpoint_name,
                "method": method_name,
                "representation_dimensions": int(
                    X.shape[1]
                ),
                "n_patients": int(
                    len(aggregated)
                ),
                "n_class_0": int(
                    (aggregated["y"] == 0).sum()
                ),
                "n_class_1": int(
                    (aggregated["y"] == 1).sum()
                ),
                "aggregated_heldout_AUC_directional": float(
                    aggregate_directional_auc
                ),
                "aggregated_heldout_AUC_orientation_invariant": float(
                    aggregate_invariant_auc
                ),
                "fold_AUC_mean": float(
                    np.mean(
                        method_fold_aucs
                    )
                ),
                "fold_AUC_SD": float(
                    np.std(
                        method_fold_aucs,
                        ddof=1,
                    )
                ),
                "fold_AUC_median": float(
                    np.median(
                        method_fold_aucs
                    )
                ),
                "rank_biserial": float(
                    rank_biserial
                ),
                "Mann_Whitney_p": float(
                    p_value
                ),
                "D_minus_log10_p": float(
                    safe_d(
                        p_value
                    )
                ),
            }
        )

        aggregated[
            "endpoint"
        ] = endpoint_name

        aggregated[
            "method"
        ] = method_name

        patient_rows.append(
            aggregated
        )


summary = pd.DataFrame(
    summary_rows
)

fold_results = pd.DataFrame(
    fold_rows
)

patient_results = pd.concat(
    patient_rows,
    ignore_index=True,
)

denominators = pd.DataFrame(
    denominator_rows
)


# ------------------------------------------------------------
# Paired fold-level method comparisons
# ------------------------------------------------------------

pairwise_rows = []

method_pairs = [
    ("MOATTERS_state", "GSVA"),
    ("MOATTERS_state", "Pathifier"),
    ("GSVA", "Pathifier"),
]

for endpoint_name in ENDPOINT_ORDER:

    endpoint_fold = fold_results[
        fold_results["endpoint"]
        == endpoint_name
    ]

    endpoint_pair_indices = []

    for method_a, method_b in method_pairs:

        a = (
            endpoint_fold[
                endpoint_fold["method"]
                == method_a
            ]
            .sort_values(
                "fold_iteration"
            )
        )

        b = (
            endpoint_fold[
                endpoint_fold["method"]
                == method_b
            ]
            .sort_values(
                "fold_iteration"
            )
        )

        if not np.array_equal(
            a["fold_iteration"].to_numpy(),
            b["fold_iteration"].to_numpy(),
        ):
            raise RuntimeError(
                f"Fold mismatch for {endpoint_name}: "
                f"{method_a} vs {method_b}"
            )

        differences = (
            a[
                "orientation_invariant_AUC"
            ].to_numpy()
            - b[
                "orientation_invariant_AUC"
            ].to_numpy()
        )

        try:
            test = wilcoxon(
                differences,
                alternative="two-sided",
                zero_method="wilcox",
            )

            p_value = float(
                test.pvalue
            )

            statistic = float(
                test.statistic
            )

        except ValueError:
            p_value = 1.0
            statistic = 0.0

        pairwise_rows.append(
            {
                "endpoint": endpoint_name,
                "method_A": method_a,
                "method_B": method_b,
                "mean_paired_AUC_difference_A_minus_B": float(
                    differences.mean()
                ),
                "median_paired_AUC_difference_A_minus_B": float(
                    np.median(
                        differences
                    )
                ),
                "Wilcoxon_statistic": statistic,
                "Wilcoxon_p": p_value,
                "n_paired_folds": int(
                    len(differences)
                ),
            }
        )

        endpoint_pair_indices.append(
            len(pairwise_rows) - 1
        )

    endpoint_p_values = [
        pairwise_rows[idx][
            "Wilcoxon_p"
        ]
        for idx in endpoint_pair_indices
    ]

    adjusted = holm_adjust(
        endpoint_p_values
    )

    for idx, adjusted_p in zip(
        endpoint_pair_indices,
        adjusted,
    ):
        pairwise_rows[idx][
            "Wilcoxon_p_Holm_within_endpoint"
        ] = float(adjusted_p)

pairwise = pd.DataFrame(
    pairwise_rows
)


# ------------------------------------------------------------
# Save tables
# ------------------------------------------------------------

summary.to_csv(
    OUT
    / "tables"
    / "Table_5B_MOATTERS_GSVA_Pathifier_benchmark.csv",
    index=False,
    encoding="utf-8-sig",
)

fold_results.to_csv(
    OUT
    / "tables"
    / "Table_5B_repeated_CV_fold_AUCs.csv",
    index=False,
    encoding="utf-8-sig",
)

patient_results.to_csv(
    OUT
    / "tables"
    / "Table_5B_aggregated_heldout_patient_scores.csv",
    index=False,
    encoding="utf-8-sig",
)

pairwise.to_csv(
    OUT
    / "tables"
    / "Table_5B_paired_fold_method_comparisons.csv",
    index=False,
    encoding="utf-8-sig",
)

denominators.to_csv(
    OUT
    / "tables"
    / "Table_5B_endpoint_denominators.csv",
    index=False,
    encoding="utf-8-sig",
)


# ------------------------------------------------------------
# Publication-ready figure
# ------------------------------------------------------------

plot_table = (
    summary
    .pivot(
        index="endpoint",
        columns="method",
        values="aggregated_heldout_AUC_orientation_invariant",
    )
    .reindex(
        ENDPOINT_ORDER
    )
)

fig, ax = plt.subplots(
    figsize=(11.5, 6.2)
)

x = np.arange(
    len(ENDPOINT_ORDER)
)

width = 0.24

for method_index, method_name in enumerate(
    METHOD_ORDER
):

    values = (
        plot_table[
            method_name
        ].to_numpy()
    )

    positions = (
        x
        + (method_index - 1)
        * width
    )

    bars = ax.bar(
        positions,
        values,
        width=width,
        label=method_name,
    )

    for bar, value in zip(
        bars,
        values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1,
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        "Stage\nlate vs early",
        "ER\nnegative vs positive",
        "PAM50\nbasal vs non-basal",
        "PAM50\nluminal vs non-luminal",
    ]
)

ax.set_ylim(
    0.48,
    min(
        1.0,
        max(
            0.75,
            float(
                np.nanmax(
                    plot_table.to_numpy()
                )
            )
            + 0.10,
        ),
    ),
)

ax.set_ylabel(
    "Aggregated repeated-CV\norientation-invariant AUC"
)

ax.set_title(
    "Locked-representation benchmark using identical patients and folds"
)

ax.legend(
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        1.12,
    ),
)

fig.tight_layout()

fig.savefig(
    OUT
    / "figures"
    / "Figure_5B_MOATTERS_GSVA_Pathifier_benchmark.png",
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    OUT
    / "figures"
    / "Figure_5B_MOATTERS_GSVA_Pathifier_benchmark.pdf",
    bbox_inches="tight",
)

plt.show()
plt.close(fig)


# ------------------------------------------------------------
# Draft Methods, Results, Rebuttal
# ------------------------------------------------------------

methods_text = (
    "The locked MOATTERS BP-state representation was benchmarked against "
    "GSVA and Pathifier using the same 30 task-discriminative GO-BP "
    "gene sets. Pathifier used 114 TCGA-BRCA adjacent-normal samples "
    "as its normal reference and was run with 20 stability attempts; "
    "evaluation was restricted to primary tumors. Four prespecified "
    "binary endpoints were examined: early versus late pathological "
    "stage, ER-positive versus ER-negative status, PAM50 basal versus "
    "non-basal status, and PAM50 luminal versus non-luminal status. "
    "For each endpoint, a complete-case patient intersection was fixed "
    "before fold generation so that MOATTERS, GSVA, and Pathifier were "
    "evaluated on identical patients. Repeated stratified five-fold "
    "cross-validation was performed for 20 repeats using identical "
    "fold assignments across methods. Within each training fold, each "
    "representation was standardized using training data only, "
    "class-specific centroids were estimated, and held-out patients "
    "were scored by the difference between cosine similarity to the "
    "positive-class and negative-class centroids. No method-specific "
    "tuning was performed. Because the 30 GO-BP terms and MOATTERS module "
    "architecture were locked from the original TCGA-BRCA analysis, "
    "this experiment was interpreted as a representation/scoring "
    "benchmark rather than a fully nested feature-selection comparison."
)

result_sentences = []

for endpoint_name in ENDPOINT_ORDER:

    subset = (
        summary[
            summary["endpoint"]
            == endpoint_name
        ]
        .set_index(
            "method"
        )
        .reindex(
            METHOD_ORDER
        )
    )

    values = ", ".join(
        [
            (
                f"{method_name} "
                f"{subset.loc[method_name, 'aggregated_heldout_AUC_orientation_invariant']:.3f}"
            )
            for method_name in METHOD_ORDER
        ]
    )

    n_value = int(
        subset.iloc[0][
            "n_patients"
        ]
    )

    result_sentences.append(
        f"{endpoint_name} (n={n_value}): {values}"
    )

results_text = (
    "In the controlled repeated-cross-validation benchmark, "
    "orientation-invariant held-out AUCs were "
    + "; ".join(
        result_sentences
    )
    + ". Pairwise fold-level differences and Holm-adjusted Wilcoxon "
    "tests are reported in the supplementary benchmark table. The "
    "comparison was used to evaluate whether the proposed compact "
    "task-conditioned reconstruction retained endpoint discrimination "
    "relative to established pathway-scoring alternatives. It was not "
    "used to claim uniform superiority, and results were interpreted "
    "together with MOATTERS's additional outputs for module structure, "
    "patient-state geometry, threshold sensitivity, and auditability."
)

rebuttal_text = (
    "We thank the reviewer for requesting quantitative benchmarking "
    "against established pathway-level methods. We added a controlled "
    "comparison with GSVA and Pathifier using the same locked 30 GO-BP "
    "gene sets, identical endpoint-specific patient intersections, and "
    "identical repeated 20×5 cross-validation folds. Pathifier used "
    "TCGA adjacent-normal samples as its reference. All representations "
    "were evaluated with the same fold-specific centroid-margin scoring "
    "procedure and no method-specific tuning. "
    + results_text
)

(
    OUT
    / "text"
    / "Methods_Stage5B.txt"
).write_text(
    methods_text,
    encoding="utf-8",
)

(
    OUT
    / "text"
    / "Results_Stage5B.txt"
).write_text(
    results_text,
    encoding="utf-8",
)

(
    OUT
    / "text"
    / "Rebuttal_Stage5B.txt"
).write_text(
    rebuttal_text,
    encoding="utf-8",
)


# ------------------------------------------------------------
# Final manifest
# ------------------------------------------------------------

manifest = {
    "status": "PASS",
    "methods": METHOD_ORDER,
    "endpoints": ENDPOINT_ORDER,
    "cross_validation": {
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "seed": SEED,
        "identical_folds_across_methods": True,
        "complete_case_intersection_before_folds": True,
        "training_fold_only_standardization": True,
        "training_fold_only_centroids": True,
    },
    "representations": {
        "MOATTERS_dimensions": int(
            state_X.shape[1]
        ),
        "GSVA_dimensions": int(
            gsva_X.shape[1]
        ),
        "Pathifier_dimensions": int(
            pathifier_X.shape[1]
        ),
    },
    "Pathifier": {
        "attempts": int(
            score_manifest.get(
                "Pathifier_attempts",
                20,
            )
        ),
        "maximize_stability": bool(
            score_manifest.get(
                "Pathifier_maximize_stability",
                True,
            )
        ),
        "normal_reference_samples": int(
            score_manifest.get(
                "n_adjacent_normals",
                114,
            )
        ),
    },
    "method_specific_tuning": False,
    "benchmark_boundary": (
        "Locked representation/scoring benchmark; "
        "not fully nested feature selection."
    ),
    "outputs": {
        "summary_table": (
            "tables/"
            "Table_5B_MOATTERS_GSVA_Pathifier_benchmark.csv"
        ),
        "fold_table": (
            "tables/"
            "Table_5B_repeated_CV_fold_AUCs.csv"
        ),
        "paired_comparison_table": (
            "tables/"
            "Table_5B_paired_fold_method_comparisons.csv"
        ),
        "patient_score_table": (
            "tables/"
            "Table_5B_aggregated_heldout_patient_scores.csv"
        ),
        "figure_png": (
            "figures/"
            "Figure_5B_MOATTERS_GSVA_Pathifier_benchmark.png"
        ),
        "figure_pdf": (
            "figures/"
            "Figure_5B_MOATTERS_GSVA_Pathifier_benchmark.pdf"
        ),
    },
}

(
    OUT
    / "analysis_manifest.json"
).write_text(
    json.dumps(
        manifest,
        indent=2,
    ),
    encoding="utf-8",
)


# ------------------------------------------------------------
# Console summary
# ------------------------------------------------------------

print()
print("=" * 80)
print("PASS — Stage 5B-3 benchmark evaluation completed")
print("=" * 80)
print()

display_columns = [
    "endpoint",
    "method",
    "n_patients",
    "aggregated_heldout_AUC_orientation_invariant",
    "fold_AUC_mean",
    "fold_AUC_SD",
    "rank_biserial",
    "Mann_Whitney_p",
]

print(
    summary[
        display_columns
    ].to_string(
        index=False
    )
)

print()
print("Output folder:")
print(OUT)
