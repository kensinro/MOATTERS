# Reproducibility and interpretation notes

1. The nominal `p < 0.05` / `D >= 1.301` screen is a reconstruction-oriented feature screen, not an FDR-adjusted discovery claim for individual BP terms.
2. External breast-cancer analyses reconstruct the locked TCGA-BRCA representation without endpoint refitting.
3. KIRC and LUAD analyses are de novo workflow-applicability demonstrations. BP selection and module construction were fixed from the full cancer cohort before centroid-only repeated cross-validation; these analyses are not fully nested end-to-end validation.
4. Orientation-invariant AUC is used to summarize discrimination magnitude where score orientation is opposite to the positive endpoint label. Biological direction must be interpreted separately.
5. Survival analyses are secondary contextual analyses and do not establish universal prognostic or clinical utility.
6. The GSVA/Pathifier comparison evaluates information retention under matched folds; it does not establish universal predictive superiority.
7. The singleton audit tests sensitivity to removing M6 and M7. Singleton components are valid one-term dimensions but are not described as coordinated multi-term modules.
