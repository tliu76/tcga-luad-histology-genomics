from pathlib import Path

import yaml

from scripts.query_gdc_metadata import build_filters

def test_filters_target_the_requested_tcga_project() -> None:
    config_path = Path("configs/luad_egfr.yaml")
    config = yaml.safe_load(config_path.read_text())
    filters = build_filters(config)
    values = [item["content"]["value"] for item in filters["content"]]
    assert ["TCGA-LUAD"] in values
    assert ["Slide Image"] in values
