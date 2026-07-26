# Recommended execution order

Use `python run_pipeline.py <stage>` or execute the same scripts individually.

1. `derivation` — TCGA-BRCA derivation and standardized reference audit.
2. `metabric` and `gse96058` — independent locked reconstruction; the two cohorts may run independently.
3. `kirc` and `luad` — cancer-specific de novo reconstruction; the two cancers may run independently.
4. `synthesis` — three-cohort and joint-evidence summaries after upstream outputs exist.
5. `singleton` — targeted simultaneous singleton-removal audit.
6. `benchmark` — benchmark input preparation, R-based GSVA/Pathifier scoring, then matched evaluation.

`--dry-run` prints the exact ordered command list. The runner stops on the first failed script unless `--continue-on-error` is supplied.
