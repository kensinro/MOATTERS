# MOATTERS reproducibility code

**MOATTERS** denotes **Modularity, Observability, Auditability, and Traceability for Task-conditioned Evidence Reconstruction into Patient States**. This repository contains the locked analysis scripts supporting the associated manuscript.

## Scientific scope

The repository preserves eight evidence blocks: TCGA-BRCA derivation/reference, independent reconstruction in METABRIC and GSE96058/SCAN-B, de novo workflow applicability in TCGA-KIRC and TCGA-LUAD, evidence synthesis, singleton-component audit, and a matched GSVA/Pathifier benchmark. KIRC and LUAD are cross-cancer applicability demonstrations, not external validation of BRCA modules.

## Portable configuration

All active scripts now resolve paths through `moatters/config.py`. Configure paths by either:

1. copying `config/moatters_config.example.json` to `config/moatters_config.json`; or
2. setting `MOATTERS_DATA_ROOT`, `MOATTERS_OUTPUT_ROOT`, `MOATTERS_RSCRIPT`, and optionally `MOATTERS_CONFIG`.

No active script requires editing a hard-coded drive path.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python tools/preflight.py --scope full
```

The benchmark additionally requires R/Bioconductor with GSVA and Pathifier. Use `--scope structure`, `core`, `benchmark`, or `full` to distinguish repository/runtime readiness from data and R-benchmark readiness.

## Running

Inspect commands without execution:

```bash
python run_pipeline.py all --dry-run
```

Run one evidence block:

```bash
python run_pipeline.py metabric
python run_pipeline.py gse96058
python run_pipeline.py kirc
python run_pipeline.py luad
python run_pipeline.py singleton
python run_pipeline.py benchmark
```

See `RUN_ORDER.md` and `docs/DATA_AND_INPUTS.md`. Individual scripts remain executable after `pip install -e .`.

## Locked primary settings

- minimum matched genes: `K >= 10`;
- nominal task-discriminability: `D = -log10(p) >= 1.301`;
- primary network threshold: `|r| >= 0.35`;
- repeated cross-validation: 20 x 5 where specified;
- label permutations: 500 where specified.

These thresholds are audited through the associated sensitivity analyses and must not be interpreted as multiple-testing-adjusted biological discoveries.

## Provenance

`archive/superseded/` retains earlier working variants for historical provenance only. It is excluded from the recommended execution path and should normally be omitted from a compact Zenodo software release unless provenance retention is desired.

## Release status

This is a release candidate. The portable configuration and tiered preflight have been audited in REP1.0, but numerical reproducibility remains to be demonstrated by rerunning the required stages against the locked inputs. Before public deposition, complete the license choice, repository URL, Zenodo DOI, and final code-to-result hash lock described in `docs/RELEASE_CHECKLIST.md`.


## REP1 rerun repair level

This archive is release candidate `1.0.0rc4`. It incorporates the clean-rerun repairs listed in `docs/REP1_0_RERUN_FIXES.md`.
