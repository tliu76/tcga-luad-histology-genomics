#!/usr/bin/env python3
"""Download open-access TCGA diagnostic slides from the GDC AWS Open Data mirror.

GDC open files are mirrored at s3://tcga-2-open/<file_id>/<file_name>, which is
faster and more reliable than api.gdc.cancer.gov/data for large .svs files.
Downloads resume-safely (.part -> rename) and go smallest-first so downstream
feature extraction can start immediately.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

MIRROR = "https://tcga-2-open.s3.amazonaws.com"


def download(row, out_dir: Path, retries: int = 4) -> str:
    dest = out_dir / row.file_name
    if dest.exists():
        return "skip"
    part = dest.with_suffix(".part")
    for attempt in range(retries):
        try:
            with requests.get(f"{MIRROR}/{row.file_id}/{row.file_name}", stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(part, "wb") as fh:
                    for chunk in r.iter_content(1 << 22):
                        fh.write(chunk)
            part.rename(dest)
            return "ok"
        except Exception as exc:  # network hiccups: back off and retry
            print(f"retry {attempt + 1} {row.patient}: {exc}", flush=True)
            time.sleep(5 * (attempt + 1))
    return "failed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/gdc_slides.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/raw/slides"))
    ap.add_argument("--max-lusc", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(args.manifest)
    labelled = {tag: set(pd.read_csv(f"data/processed/labels_{tag}.csv").patient) for tag in ["luad", "lusc"]}
    luad = m[(m.project == "TCGA-LUAD") & m.patient.isin(labelled["luad"])].sort_values("size_gb")
    lusc = m[(m.project == "TCGA-LUSC") & m.patient.isin(labelled["lusc"])].sort_values("size_gb").head(args.max_lusc)
    # interleave so LUSC controls arrive alongside LUAD, but LUAD dominates the queue
    queue = pd.concat([luad, lusc]).sort_values("size_gb")
    print(f"queue: {len(luad)} LUAD + {len(lusc)} LUSC slides, {queue.size_gb.sum():.0f} GB", flush=True)

    t0, done_gb = time.time(), 0.0
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(download, row, args.out): row for row in queue.itertuples()}
        for i, fut in enumerate(as_completed(futs), 1):
            row = futs[fut]
            status = fut.result()
            done_gb += row.size_gb if status == "ok" else 0
            rate = done_gb * 1000 / max(time.time() - t0, 1)
            print(f"[{i}/{len(queue)}] {status} {row.project} {row.patient} {row.size_gb:.2f} GB  ({rate:.0f} MB/s)", flush=True)


if __name__ == "__main__":
    main()
