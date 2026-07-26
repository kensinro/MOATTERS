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

```text
docs/REP1_0_RERUN_FIXES.md
