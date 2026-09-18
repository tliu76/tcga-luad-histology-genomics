#!/usr/bin/env python3
"""Create a manifest of open-access TCGA-LUAD diagnostic (FFPE) H&E slides.

One slide per patient: the smallest diagnostic slide, to keep download time down.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests
import yaml

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"


def build_filters(config: dict, project: str | None = None) -> dict:
    """Restrict the query to open-access TCGA diagnostic slide images."""
    project = project or config["project"]
    return {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
            {"op": "in", "content": {"field": "data_type", "value": [config["data_type"]]}},
            {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ],
    }


def query_gdc(config: dict, project: str) -> pd.DataFrame:
    response = requests.post(
        GDC_FILES_ENDPOINT,
        json={
            "filters": build_filters(config, project),
            "fields": "file_id,file_name,file_size,cases.submitter_id",
            "size": "5000",
            "format": "JSON",
        },
        timeout=120,
    )
    response.raise_for_status()
    hits = response.json()["data"]["hits"]
    df = pd.DataFrame(
        [
            {
                "file_id": h["file_id"],
                "file_name": h["file_name"],
                "size_gb": h["file_size"] / 1e9,
                "patient": h["cases"][0]["submitter_id"],
            }
            for h in hits
        ]
    )
    # one slide per patient; take the smallest file to save download time
    return df.sort_values("size_gb").drop_duplicates("patient", keep="first").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/luad_egfr.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/gdc_slides.csv"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    frames = []
    for project in ["TCGA-LUAD", "TCGA-LUSC"]:
        df = query_gdc(config, project).assign(project=project)
        print(f"{project}: {len(df)} patients, median {df.size_gb.median():.2f} GB, total {df.size_gb.sum():.0f} GB")
        frames.append(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
