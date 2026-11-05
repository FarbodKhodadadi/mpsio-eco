# Additional experiments

- `toy_mpsio.py`: reconstructed synthetic-data experiments described in the
  manuscript (not the original source file; see the root README).
- `train_feature_ranking_encoder.py`: auxiliary encoder training from the
  supplied feature-ranking notebook. It uses a legacy normalized/squared SI
  formulation and a record-level split, so it is intentionally separate from
  the primary MPSIO-ECO benchmark.
- `exploratory_open_set_distance.py`: optional analysis based on the uploaded
  notebook's latent-centroid open-set idea. It is not part of the final paper's
  reported experiments and calibrates its threshold on validation data.

The SVM/KNN/MLP/SMOTE/decision-boundary cells in the supplied notebook were
not copied into the reproducibility package because they were exploratory,
duplicated one another, and were not part of the manuscript's reported method.
