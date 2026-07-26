# Data and input resources

The repository does not redistribute public cohort matrices or annotation resources. Obtain them from their original repositories and record source versions and checksums before a reproducibility run.

## Cohorts

- TCGA-BRCA, TCGA-KIRC, and TCGA-LUAD transcriptomic and clinical data: UCSC Xena and associated TCGA resources.
- METABRIC: cBioPortal and associated public repositories.
- GSE96058/SCAN-B: NCBI Gene Expression Omnibus, accession GSE96058.

## Knowledge resource

- Gene Ontology Biological Process GMT used in the locked analysis: `c5.go.bp.v2026.1.Hs.symbols.gmt`.

## Expected local layout

Active scripts do **not** require manual path editing. They resolve paths through `moatters/config.py`.

Configure the roots in either `config/moatters_config.json` or environment variables:

```json
{
  "data_root": "D:/MOATTERS-Data",
  "output_root": "D:/MOATTERS-Output",
  "rscript": "C:/Program Files/R/R-4.6.0/bin/Rscript.exe"
}
```

The default expected structure under `data_root` is:

```text
GSEA/
  c5.go.bp.v2026.1.Hs.symbols.gmt
UCSC_XENA/
  Breast Cancer (BRCA)/
  Kidney Clear Cell Carcinoma (KIRC)/
  Lung Adenocarcinoma (LUAD)/
External/
  brca_metabric/
  GSE96058/
```

Run the machine-readable preflight after configuration:

```bash
python tools/preflight.py --scope full
```

Useful narrower checks:

```bash
python tools/preflight.py --scope structure   # Python runtime and writable output root
python tools/preflight.py --scope core        # Python runtime plus all analysis inputs
python tools/preflight.py --scope benchmark   # Python plus R/GSVA/Pathifier
```

Relative paths in the bundled JSON configuration are resolved from the repository root, so execution remains stable even when `run_pipeline.py` is called from another working directory. Environment-variable paths retain normal shell behaviour and resolve from the caller's current directory when relative.

Exact source filenames inside each cohort directory may differ across repository snapshots. Preserve the original filenames used for the locked run and record checksums in the execution ledger.
