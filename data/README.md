# Data policy

Do not commit raw whole-slide images, patient-level clinical files, GDC tokens, or derived embeddings to GitHub.

Commit only small, non-identifying metadata manifests when their source permits redistribution. Keep raw data under `data/raw/`, which is ignored by Git.

For this project, start by creating a GDC metadata manifest with `scripts/query_gdc_metadata.py`. Download slides only after reviewing the number, size, and access requirements in the GDC portal.
