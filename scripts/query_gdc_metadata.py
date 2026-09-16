#!/usr/bin/env python3
"""Create a public metadata manifest for TCGA-LUAD diagnostic slide images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests
import yaml

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"


def build_filters(config: dict) -> dict:
    """Restrict the query to public TCGA-LUAD primary-tumor slide images."""
    return {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [config["project"]]}},
            {"op": "in", "content": {"field": "data_type", "value": [config["data_type"]]}},
            {"op": "in", "content": {"field": "cases.samples.sample_type", "value": [config["sample_type"]]}},
        ],
    }


def query_gdc(config: dict) -> pd.DataFrame:
    fields = [
        "file_id",
        "file_name",
        "file_size",
        "data_format",
        "cases.case_id",
        "cases.submitter_id",
        "cases.samples.sample_type",
    ]
    params = {
        "filters": json.dumps(build_filters(config)),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": "10000",
    }
    response = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=60)
    response.raise_for_status()
    hits = response.json()["data"]["hits"]

    rows = []
    for hit in hits:
        case = hit.get("cases", [{}])[0]
        rows.append(
            {
                "file_id": hit["file_id"],
                "file_name": hit["file_name"],
                "file_size_gb": round(hit.get("file_size", 0) / 1_000_000_000, 3),
                "data_format": hit.get("data_format"),
                "case_id": case.get("case_id"),
                "submitter_id": case.get("submitter_id"),
            }
        )
    return pd.DataFrame(rows).sort_values(["submitter_id", "file_name"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/tcga_luad_slide_manifest.tsv"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    manifest = query_gdc(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(manifest)} slide records to {args.output}")
    print(f"Total listed size: {manifest['file_size_gb'].sum():.1f} GB")


if __name__ == "__main__":
    main()
