# MOATTERS reproducibility code

[![Version](https://img.shields.io/badge/version-1.0.0-blue)](https://github.com/kensinro/MOATTERS)
[![Reproducibility](https://img.shields.io/badge/REP1.0-rerun--validated-success)](https://github.com/kensinro/MOATTERS)

**MOATTERS** denotes **Modularity, Observability, Auditability, and Traceability for Task-conditioned Evidence Reconstruction into Patient States**.

This repository contains the version-locked and rerun-validated analysis code supporting the associated biomedical-informatics study. The implementation reconstructs patient-level molecular-functional states from task-discriminative biological-process observables while preserving modularity, traceability, explicit audit boundaries, and reproducible execution.

Repository:

`https://github.com/kensinro/MOATTERS`

Current public version:

`v1.0.0`

---

## Scientific scope

The repository preserves the following evidence blocks:

1. TCGA-BRCA derivation and standardized reference audit;
2. independent breast-cancer reconstruction and external validation in METABRIC;
3. independent breast-cancer reconstruction and external validation in GSE96058/SCAN-B;
4. de novo cross-cancer workflow applicability in TCGA-KIRC;
5. de novo cross-cancer workflow applicability in TCGA-LUAD;
6. three-cohort and joint evidence synthesis;
7. singleton-component audit;
8. matched GSVA and Pathifier benchmarking.

METABRIC and GSE96058/SCAN-B are independent breast-cancer external-validation cohorts.

TCGA-KIRC and TCGA-LUAD are cross-cancer applicability demonstrations. They do not constitute external validation of the BRCA-derived modules and should not be described as such.

---

## Release status

Version `1.0.0` is the first public, rerun-validated release of the MOATTERS reproducibility workflow.

The principal workflow was rerun from configured cohort inputs through:

- TCGA-BRCA derivation;
- TCGA-BRCA standardized endpoint audit;
- METABRIC external reconstruction and validation;
- GSE96058/SCAN-B external reconstruction and validation;
- TCGA-KIRC cross-cancer analysis;
- TCGA-LUAD cross-cancer analysis;
- three-cohort evidence synthesis;
- joint evidence synthesis;
- singleton-component audit;
- GSVA and Pathifier benchmarking.

The rerun confirmed the principal cohort denominators and workflow outputs:

| Cohort | Locked analysis denominator |
|---|---:|
| TCGA-BRCA | 1,073 matched patients |
| METABRIC | 1,980 patients |
| GSE96058/SCAN-B | 3,069 patients |
| TCGA-KIRC | 533 primary-tumour patients |
| TCGA-LUAD | 515 primary-tumour patients |

The code release incorporates the portability, compatibility, and dependency repairs identified during REP1.0. These repairs are documented in:

`docs/REP1_0_RERUN_FIXES.md`

---

## Principal rerun results

### TCGA-BRCA derivation

The locked TCGA-BRCA derivation included:

- 1,073 patients matched between expression and stage data;
- 803 early-stage and 270 late-stage patients;
- 7,538 GO biological-process gene sets evaluated;
- 5,294 observation-ready biological-process terms;
- 30 selected biological-process terms used for network reconstruction;
- seven patient-state components in the locked BRCA representation.

### METABRIC external validation

The METABRIC workflow locked:

- 20,385 expression genes;
- 1,980 expression-clinical matched patients;
- 30/30 locked biological-process terms retained at `K >= 10`;
- 7/7 locked modules retained at `K >= 10`;
- six binary endpoint analyses;
- OS and RFS survival analyses.

### GSE96058/SCAN-B external validation

The GSE96058/SCAN-B workflow locked:

- 30,863 expression genes;
- 3,069 primary analysis patients;
- 30/30 locked biological-process terms retained at `K >= 10`;
- 7/7 locked modules retained at `K >= 10`;
- pathology-, molecular-classifier-, subtype-, and node-related endpoint analyses;
- OS survival analysis.

### TCGA-KIRC applicability

The KIRC workflow produced:

- 533 primary-tumour patients;
- 531 patients with usable stage labels;
- two derived modules at the primary network threshold;
- apparent margin AUC of approximately `0.714`;
- centroid-only held-out AUC of approximately `0.713`;
- empirical permutation p-value of approximately `0.002`.

### TCGA-LUAD applicability

The LUAD workflow produced:

- 515 primary-tumour patients;
- 507 patients with usable stage labels;
- three derived modules at the primary network threshold;
- apparent margin AUC of approximately `0.665`;
- centroid-only held-out AUC of approximately `0.659`;
- empirical permutation p-value of approximately `0.002`.

---

## Singleton-component audit

The directed singleton-component audit evaluated whether the two singleton components contributed only redundant structure or provided incremental patient-state resolution.

For the complete seven-component representation:

- centroid-margin orientation-invariant AUC: `0.648923`;
- late-similarity Spearman correlation versus the full representation: `1.000000`;
- three-state assignment agreement: `1.000000`;
- PAM50 discrimination D: `1.323727`.

Removing both singleton components reduced:

- centroid-margin AUC to `0.631032`;
- late-similarity Spearman correlation to `0.907849`;
- three-state agreement to `0.796831`;
- PAM50 D to `0.390643`.

These results support a bounded interpretation: the singleton components provide incremental state resolution and assignment stability, but they do not dominate the reconstruction.

---

## GSVA and Pathifier benchmark

The matched benchmark compared the compact MOATTERS patient-state representation with GSVA and Pathifier using the same locked biological-process set and endpoint definitions.

Aggregated repeated-cross-validation orientation-invariant AUC values were:

| Endpoint | MOATTERS state | GSVA | Pathifier |
|---|---:|---:|---:|
| Stage late vs early | 0.641041 | 0.638799 | 0.536451 |
| ER negative | 0.802190 | 0.811826 | 0.924033 |
| PAM50 basal | 0.922344 | 0.917990 | 0.964507 |
| PAM50 luminal | 0.815004 | 0.828777 | 0.947979 |

The benchmark does not support universal superiority of MOATTERS over existing pathway-scoring methods.

The supported interpretation is that MOATTERS provides a compact, interpretable seven-component patient-state representation with stage-discriminative performance comparable to GSVA, whereas Pathifier provides stronger discrimination for ER- and PAM50-related endpoints.

---

## Portable configuration

All active scripts resolve paths through:

`moatters/config.py`

Paths can be configured by either:

1. copying `config/moatters_config.example.json` to `config/moatters_config.json`; or
2. setting the following environment variables:

   - `MOATTERS_DATA_ROOT`
   - `MOATTERS_OUTPUT_ROOT`
   - `MOATTERS_RSCRIPT`
   - `MOATTERS_CONFIG`

`MOATTERS_CONFIG` is optional.

Active workflow scripts do not require manual editing of hard-coded drive paths.

Relative paths in the JSON configuration are resolved against the repository root rather than the shell working directory.

---

## Software requirements

### Python

The validated Python range for this release is:

`Python >=3.10,<3.14`

The REP1.0 rerun was conducted using Python `3.12.4`.

Principal Python dependencies include:

- NumPy;
- pandas;
- SciPy;
- scikit-learn;
- statsmodels;
- NetworkX;
- lifelines;
- Matplotlib.

### R and Bioconductor

The GSVA/Pathifier benchmark additionally requires:

- R;
- Bioconductor;
- GSVA;
- pathifier.

The REP1.0 benchmark was completed using R `4.6.0` with Bioconductor `3.23`.

R is required only for the benchmark block. The remaining workflow is Python-based.

---

## Installation

Create and activate a virtual environment.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies and local package:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

Run the structural preflight:

```bash
python tools/preflight.py --scope structure
```

Available preflight scopes are:

- `structure`
- `core`
- `benchmark`
- `full`

---

## Required data

Public cohort matrices are not redistributed in this repository.

Users must obtain the source data from their original public repositories and configure the local paths described in:

`docs/DATA_AND_INPUTS.md`

The expected default layout is:

```text
data/
├── GSEA/
│   ├── c5.go.bp.v2026.1.Hs.symbols.gmt
│   └── h.all.v2026.1.Hs.symbols.gmt
├── UCSC_XENA/
│   ├── Breast Cancer (BRCA)/
│   ├── Kidney Clear Cell Carcinoma (KIRC)/
│   └── Lung Adenocarcinoma (LUAD)/
└── External/
    ├── brca_metabric/
    └── GSE96058/
```

The exact required filenames and cohort-specific assumptions are documented in the data guide.

---

## Running the workflow

Inspect all commands without executing them:

```bash
python run_pipeline.py all --dry-run
```

Run the full ordered workflow:

```bash
python run_pipeline.py all
```

Run one evidence block:

```bash
python run_pipeline.py derivation
python run_pipeline.py metabric
python run_pipeline.py gse96058
python run_pipeline.py kirc
python run_pipeline.py luad
python run_pipeline.py synthesis
python run_pipeline.py singleton
python run_pipeline.py benchmark
```

See:

- `RUN_ORDER.md`
- `docs/DATA_AND_INPUTS.md`
- `docs/RELEASE_CHECKLIST.md`
- `docs/REP1_0_RERUN_FIXES.md`

---

## Locked primary settings

The primary locked settings are:

- minimum matched genes: `K >= 10`;
- nominal task discriminability: `D = -log10(p) >= 1.301`;
- primary network threshold: `|r| >= 0.35`;
- repeated cross-validation: `20 x 5`, where specified;
- label permutations: `500`, where specified;
- bootstrap resampling: according to the stage-specific locked configuration.

Sensitivity analyses include:

- `K >= 5`
- `K >= 10`
- `K >= 15`
- `K >= 20`

These thresholds support observation readiness, task discrimination, and structural sensitivity audits. They must not be interpreted as multiple-testing-adjusted biological discoveries unless an adjusted analysis is explicitly reported.

---

## Locked outputs and reproducibility record

A compact companion archive preserves the principal rerun evidence without redistributing large public source matrices or all regenerable intermediates.

The companion reproducibility package includes:

- `MOATTERS_REP1_LOCKED_OUTPUTS_v1.0.0.zip`
- `MOATTERS_REP1_REPRODUCIBILITY_REPORT.md`
- `MOATTERS_REP1_OUTPUT_SHA256_MANIFEST.tsv`
- `MOATTERS_REP1_RELEASE_CONTENTS.md`

The locked-output archive contains:

- final CSV and TSV result tables;
- JSON manifests and summaries;
- selected figures;
- execution logs;
- successful-run evidence;
- explicitly classified failed or superseded attempt records;
- a per-file SHA-256 integrity manifest.

Large source matrices, intermediate expression matrices, and large regenerable serialized objects are excluded by design.

The Zenodo DOI will be added here after the first public release has been archived.

---

## Reproducibility and integrity

The REP1.0 rerun proceeded from configured cohort inputs through the principal workflow rather than relying only on static code inspection.

The release includes:

- tiered preflight checks;
- explicit run ordering;
- portable path resolution;
- environment and dependency checks;
- execution logs;
- locked result summaries;
- SHA-256 file manifests;
- a discrepancy and repair record.

The compact locked-output companion preserves selected failed or superseded attempt logs as part of the audit trail. These records are explicitly classified and must not be interpreted as final scientific outputs.

For the downstream-validation block, the locked successful output is:

`MOATTERS_BRCA_STATE_DownstreamValidation_20260726_112549`

The earlier timestamped attempt:

`MOATTERS_BRCA_STATE_DownstreamValidation_20260726_112437`

is retained only as a superseded failed attempt.

---

## Interpretation boundaries

The workflow reconstructs evidence-constrained patient-level molecular-functional state representations.

It does not establish:

- a new biological subtype;
- a true latent disease state;
- a causal mechanism;
- universal prognostic validity;
- clinical decision utility;
- superiority over all pathway-scoring methods.

Survival analyses are secondary and cohort-dependent.

External validation supports transportability of the reconstruction framework within the tested breast-cancer cohorts. Cross-cancer analyses support workflow applicability in the tested tumour types, not identity of the underlying BRCA-derived biology.

---

## Provenance

The directory:

`archive/superseded/`

retains selected earlier working variants for historical provenance.

These files are excluded from the recommended execution path. They should not be treated as active workflow components.

The public release should use the active scripts referenced in:

- `run_pipeline.py`
- `RUN_ORDER.md`

---

## Citation

Citation metadata are provided in:

`CITATION.cff`

The repository should be cited together with the archived Zenodo release after the DOI becomes available.

Repository:

`https://github.com/kensinro/MOATTERS`

---

## License

License terms are provided in the repository-level:

`LICENSE`

Third-party datasets and software remain subject to their original licenses and terms of use.
