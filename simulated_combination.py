#!/usr/bin/env python3
"""Enumerate all simulated genotype combinations from one sample genotype matrix.

Inputs:
1. Genotype matrix CSV:
   - rows are samples
   - first column must be ``sample``
   - remaining columns are site ids such as ``Chr1:12345``
   - values must be genotype codes in {0, 1, 2}
2. Mutable-site CSV:
   - first column ``Chromosome``
   - second column ``Position``
3. Variant-effect CSV:
   - first column ``Chromosome``
   - second column ``Position``
   - last column ``variant_effect``

For each mutable site, the script enumerates the two genotype values different
from the original genotype. If the original genotype is 0, the mutated values
are 1 and 2. Therefore, N mutable sites produce 2^N simulated samples.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import pickle
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Sequence

from utils import get_logger


LOGGER = None
VALID_GENOTYPES = {0, 1, 2}
SITE_ID_PATTERN = re.compile(r"^Chr\d+:\d+$")


@dataclass(frozen=True)
class MutableSite:
    chrom: str
    pos: int

    @property
    def site_id(self) -> str:
        return f"{self.chrom}:{self.pos}"


def parse_scalar_config_value(raw_value: str) -> object:
    value = raw_value.strip()
    if value == "":
        return ""
    lower = value.lower()
    if lower in {"null", "none"}:
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return ast.literal_eval(value)
    except Exception:
        return value.strip("'\"")


def load_yaml_config(config_file: str) -> dict[str, object]:
    path = Path(config_file)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None

    if yaml is not None:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML config must be a key-value mapping: {path}")
        return {str(key): value for key, value in data.items()}

    config: dict[str, object] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(
                    f"Unsupported YAML format at {path}:{line_no}. "
                    "Without PyYAML installed, only simple 'key: value' lines are supported."
                )
            key, value = line.split(":", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"Empty config key at {path}:{line_no}")
            config[key] = parse_scalar_config_value(value)
    return config


def require_config_value(config: dict[str, object], key: str) -> object:
    if key not in config or config[key] in {None, ""}:
        raise ValueError(f"Missing required config field: {key}")
    return config[key]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = [{str(k): str(v).strip() for k, v in row.items() if k is not None} for row in reader]
    return rows


def write_csv(rows: list[dict[str, object]], path: Path, fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_parquet(rows: list[dict[str, object]], path: Path) -> bool:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        LOGGER.warning("pandas is not installed, skip parquet output: %s", path)
        return False
    try:
        pd.DataFrame(rows).to_parquet(path, index=False)
        LOGGER.info("Wrote parquet: %s rows=%d", path, len(rows))
        return True
    except Exception as exc:
        LOGGER.warning("Failed to write parquet %s: %s", path, exc)
        return False


def load_model(model_file: str):
    path = Path(model_file)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".joblib", ".jl"}:
        try:
            import joblib  # type: ignore
        except Exception as exc:
            raise ImportError("joblib is required to load .joblib model files") from exc
        model_obj = joblib.load(path)
    else:
        with path.open("rb") as handle:
            model_obj = pickle.load(handle)

    if hasattr(model_obj, "predict"):
        return model_obj
    if isinstance(model_obj, dict):
        if "model" in model_obj and hasattr(model_obj["model"], "predict"):
            return model_obj["model"]
        if "predictor" in model_obj and hasattr(model_obj["predictor"], "predict"):
            return model_obj["predictor"]
    raise ValueError(f"Unsupported model object in file: {path}")


def get_model_feature_order(model, site_columns: Sequence[str]) -> list[str]:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return list(site_columns)
    return [str(name) for name in feature_names]


def predict_rows(model, weighted_rows: list[dict[str, object]], site_columns: Sequence[str]) -> list[float]:
    feature_order = get_model_feature_order(model, site_columns)
    missing = [feature for feature in feature_order if feature not in site_columns]
    if missing:
        raise ValueError(
            "Model expects features missing from simulated matrix: "
            + ",".join(missing[:20])
            + ("..." if len(missing) > 20 else "")
        )

    matrix = [[float(row[feature]) for feature in feature_order] for row in weighted_rows]
    try:
        predictions = model.predict(matrix)
    except Exception as exc:
        raise RuntimeError(f"Model prediction failed: {exc}") from exc
    return [float(value) for value in predictions]


def rank_predictions(
    weighted_rows: list[dict[str, object]],
    predictions: list[float],
    rank_sample: str,
    descending: bool = True,
) -> list[dict[str, object]]:
    prediction_rows = [
        {
            "sample": str(row["sample"]),
            "predicted_phenotype": float(pred),
        }
        for row, pred in zip(weighted_rows, predictions)
    ]
    prediction_rows.sort(key=lambda row: row["predicted_phenotype"], reverse=descending)
    for rank, row in enumerate(prediction_rows, start=1):
        row["rank"] = rank
        row["is_target"] = row["sample"] == rank_sample

    matched = [row for row in prediction_rows if row["sample"] == rank_sample]
    if not matched:
        raise ValueError(f"Rank sample '{rank_sample}' not found in simulated samples")
    return prediction_rows


def _scale_value(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if math.isclose(src_min, src_max):
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def write_rank_scatter_svg(
    ranking_rows: list[dict[str, object]],
    target_sample: str,
    output_path: Path,
) -> None:
    width = 1200
    height = 700
    left = 90
    right = 40
    top = 40
    bottom = 80
    plot_width = width - left - right
    plot_height = height - top - bottom

    ranks = [int(row["rank"]) for row in ranking_rows]
    values = [float(row["predicted_phenotype"]) for row in ranking_rows]
    min_rank = min(ranks)
    max_rank = max(ranks)
    min_value = min(values)
    max_value = max(values)

    points: list[str] = []
    target_label = ""
    for row in ranking_rows:
        x = _scale_value(float(row["rank"]), float(min_rank), float(max_rank), left, left + plot_width)
        y = _scale_value(float(row["predicted_phenotype"]), min_value, max_value, top + plot_height, top)
        is_target = bool(row["is_target"])
        color = "#d62728" if is_target else "#bdbdbd"
        radius = 5 if is_target else 3
        opacity = "1.0" if is_target else "0.75"
        points.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}" fill-opacity="{opacity}" />')
        if is_target:
            label_y = max(20.0, y - 14.0)
            target_label = (
                f'<text x="{x + 8:.2f}" y="{label_y:.2f}" font-size="14" fill="#d62728">'
                f'{row["sample"]} (rank={row["rank"]}, pred={row["predicted_phenotype"]:.6f})'
                "</text>"
                f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x + 6:.2f}" y2="{label_y - 5:.2f}" stroke="#d62728" stroke-width="1.5" />'
            )

    axis_lines = [
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333" stroke-width="1.5" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333" stroke-width="1.5" />',
    ]

    x_ticks = []
    tick_count = min(10, len(ranking_rows))
    for i in range(tick_count):
        tick_rank = 1 if tick_count == 1 else round(min_rank + i * (max_rank - min_rank) / (tick_count - 1))
        x = _scale_value(float(tick_rank), float(min_rank), float(max_rank), left, left + plot_width)
        x_ticks.append(f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 6}" stroke="#333" />')
        x_ticks.append(f'<text x="{x:.2f}" y="{top + plot_height + 24}" font-size="12" text-anchor="middle" fill="#333">{tick_rank}</text>')

    y_ticks = []
    for i in range(5):
        tick_value = min_value if i == 0 else min_value + i * (max_value - min_value) / 4.0
        y = _scale_value(tick_value, min_value, max_value, top + plot_height, top)
        y_ticks.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#333" />')
        y_ticks.append(f'<text x="{left - 10}" y="{y + 4:.2f}" font-size="12" text-anchor="end" fill="#333">{tick_value:.4f}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width / 2:.0f}" y="24" text-anchor="middle" font-size="20" fill="#222">Predicted Phenotype Ranking</text>
<text x="{width / 2:.0f}" y="{height - 18}" text-anchor="middle" font-size="15" fill="#333">rank</text>
<text x="24" y="{height / 2:.0f}" text-anchor="middle" font-size="15" fill="#333" transform="rotate(-90 24 {height / 2:.0f})">predicted phenotype</text>
{''.join(axis_lines)}
{''.join(x_ticks)}
{''.join(y_ticks)}
{''.join(points)}
{target_label}
<circle cx="{width - 170}" cy="44" r="4" fill="#bdbdbd" />
<text x="{width - 160}" y="48" font-size="12" fill="#444">background samples</text>
<circle cx="{width - 170}" cy="66" r="5" fill="#d62728" />
<text x="{width - 160}" y="70" font-size="12" fill="#444">target sample: {target_sample}</text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")
    LOGGER.info("Wrote ranking scatter SVG: %s", output_path)


def load_single_sample_genotype(genotype_file: str, sample_name: str | None) -> tuple[str, list[str], dict[str, int]]:
    path = Path(genotype_file)
    if not path.exists():
        raise FileNotFoundError(f"Genotype file not found: {path}")

    rows = load_csv_rows(path)
    if not rows:
        raise ValueError(f"Genotype file is empty: {path}")
    if "sample" not in rows[0]:
        raise ValueError("Genotype file must contain a 'sample' column as the first column")

    if sample_name is None:
        if len(rows) != 1:
            raise ValueError(
                "Genotype file contains multiple samples. Please provide --sample-name to choose one sample."
            )
        target_row = rows[0]
    else:
        matched = [row for row in rows if row.get("sample", "") == sample_name]
        if not matched:
            raise ValueError(f"Sample '{sample_name}' not found in genotype file: {path}")
        if len(matched) > 1:
            raise ValueError(f"Sample '{sample_name}' appears multiple times in genotype file: {path}")
        target_row = matched[0]

    selected_sample = str(target_row["sample"])
    site_columns = [column for column in target_row.keys() if column != "sample"]
    if not site_columns:
        raise ValueError("Genotype file does not contain any site columns")
    invalid_site_columns = [column for column in site_columns if SITE_ID_PATTERN.fullmatch(str(column)) is None]
    if invalid_site_columns:
        raise ValueError(
            "Genotype feature columns must use the format 'ChrN:Pos'. Invalid columns: "
            + ",".join(invalid_site_columns[:20])
            + ("..." if len(invalid_site_columns) > 20 else "")
        )

    genotype_map: dict[str, int] = {}
    for site_id in site_columns:
        raw_value = str(target_row.get(site_id, "")).strip()
        try:
            genotype = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid genotype value for site {site_id}: {raw_value}") from exc
        if genotype not in VALID_GENOTYPES:
            raise ValueError(f"Genotype value for site {site_id} must be one of {sorted(VALID_GENOTYPES)}")
        genotype_map[site_id] = genotype

    LOGGER.info(
        "Loaded genotype sample: sample=%s total_sites=%d source=%s",
        selected_sample,
        len(site_columns),
        path,
    )
    return selected_sample, site_columns, genotype_map


def load_mutable_sites(mutable_sites_file: str) -> list[MutableSite]:
    path = Path(mutable_sites_file)
    if not path.exists():
        raise FileNotFoundError(f"Mutable-site file not found: {path}")

    rows = load_csv_rows(path)
    if not rows:
        raise ValueError(f"Mutable-site file is empty: {path}")
    header = list(rows[0].keys())
    if len(header) < 2 or header[0] != "Chromosome" or header[1] != "Position":
        raise ValueError("Mutable-site file first two columns must be: Chromosome, Position")

    sites: list[MutableSite] = []
    seen: set[str] = set()
    for row in rows:
        site = MutableSite(chrom=str(row["Chromosome"]).strip(), pos=int(str(row["Position"]).strip()))
        if site.site_id in seen:
            raise ValueError(f"Duplicate mutable site detected: {site.site_id}")
        seen.add(site.site_id)
        sites.append(site)
    LOGGER.info("Loaded mutable sites: count=%d source=%s", len(sites), path)
    return sites


def validate_mutable_sites_in_genotype(site_columns: Sequence[str], mutable_sites: Sequence[MutableSite]) -> None:
    site_set = set(site_columns)
    missing = [site.site_id for site in mutable_sites if site.site_id not in site_set]
    if missing:
        raise ValueError(
            "Some mutable sites are not present in the genotype matrix columns: "
            + ",".join(missing)
        )


def load_variant_effects(variant_effect_file: str, site_columns: Sequence[str]) -> dict[str, float]:
    path = Path(variant_effect_file)
    if not path.exists():
        raise FileNotFoundError(f"Variant-effect file not found: {path}")

    rows = load_csv_rows(path)
    if not rows:
        raise ValueError(f"Variant-effect file is empty: {path}")

    header = list(rows[0].keys())
    if len(header) < 3:
        raise ValueError(
            "Variant-effect file must contain at least three columns: Chromosome, Position, ..., variant_effect"
        )
    if header[0] != "Chromosome" or header[1] != "Position" or header[-1] != "variant_effect":
        raise ValueError(
            "Variant-effect file format must be: first column Chromosome, second column Position, last column variant_effect"
        )

    effect_map: dict[str, float] = {}
    for row in rows:
        site_id = f"{str(row['Chromosome']).strip()}:{int(str(row['Position']).strip())}"
        effect_map[site_id] = float(str(row["variant_effect"]).strip())

    missing_effects = [site_id for site_id in site_columns if site_id not in effect_map]
    if missing_effects:
        raise ValueError(
            "Variant-effect file is missing some genotype sites: "
            + ",".join(missing_effects[:20])
            + ("..." if len(missing_effects) > 20 else "")
        )
    LOGGER.info("Loaded variant effects: count=%d source=%s", len(effect_map), path)
    return effect_map


def alternative_genotypes(original: int) -> list[int]:
    return [value for value in sorted(VALID_GENOTYPES) if value != original]


def build_sample_rows(
    sample_id: str,
    site_columns: Sequence[str],
    genotype_map: dict[str, int],
    effect_map: dict[str, float],
) -> tuple[dict[str, object], dict[str, object]]:
    genotype_row: dict[str, object] = {"sample": sample_id}
    weighted_row: dict[str, object] = {"sample": sample_id}
    for site_id in site_columns:
        genotype_value = int(genotype_map[site_id])
        genotype_row[site_id] = genotype_value
        weighted_row[site_id] = float(genotype_value) * float(effect_map[site_id])
    return genotype_row, weighted_row


def build_simulated_rows(
    original_sample: str,
    site_columns: Sequence[str],
    genotype_map: dict[str, int],
    mutable_sites: Sequence[MutableSite],
    effect_map: dict[str, float],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    mutable_site_ids = [site.site_id for site in mutable_sites]
    mutable_value_choices = [alternative_genotypes(genotype_map[site_id]) for site_id in mutable_site_ids]
    total_samples = 2 ** len(mutable_site_ids)
    LOGGER.info(
        "Enumerating simulated combinations: mutable_sites=%d total_simulated_samples=%d",
        len(mutable_site_ids),
        total_samples,
    )

    genotype_rows: list[dict[str, object]] = []
    weighted_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    for combo_index, combo_values in enumerate(product(*mutable_value_choices), start=1):
        sample_id = f"{original_sample}_sim_{combo_index:06d}"
        simulated_genotypes = dict(genotype_map)
        mutation_desc_parts: list[str] = []
        for site_id, new_value in zip(mutable_site_ids, combo_values):
            old_value = simulated_genotypes[site_id]
            simulated_genotypes[site_id] = int(new_value)
            mutation_desc_parts.append(f"{site_id}:{old_value}>{new_value}")

        genotype_row, weighted_row = build_sample_rows(
            sample_id=sample_id,
            site_columns=site_columns,
            genotype_map=simulated_genotypes,
            effect_map=effect_map,
        )
        genotype_rows.append(genotype_row)
        weighted_rows.append(weighted_row)
        metadata_rows.append(
            {
                "sample": sample_id,
                "source_sample": original_sample,
                "mutable_site_count": len(mutable_site_ids),
                "mutation_path": ";".join(mutation_desc_parts),
            }
        )

    return genotype_rows, weighted_rows, metadata_rows


def save_outputs(
    outdir: Path,
    site_columns: Sequence[str],
    genotype_rows: list[dict[str, object]],
    weighted_rows: list[dict[str, object]],
    metadata_rows: list[dict[str, object]],
    original_sample: str,
    mutable_sites: Sequence[MutableSite],
) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)

    genotype_csv = outdir / "simulated_genotype_012.csv"
    genotype_parquet = outdir / "simulated_genotype_012.parquet"
    weighted_csv = outdir / "simulated_variant_effect_matrix.csv"
    weighted_parquet = outdir / "simulated_variant_effect_matrix.parquet"
    metadata_csv = outdir / "simulated_metadata.csv"
    metadata_json = outdir / "simulated_summary.json"

    genotype_fields = ["sample"] + list(site_columns)
    write_csv(genotype_rows, genotype_csv, genotype_fields)
    genotype_parquet_ok = maybe_write_parquet(genotype_rows, genotype_parquet)
    LOGGER.info("Wrote simulated genotype matrix: %s rows=%d", genotype_csv, len(genotype_rows))

    write_csv(weighted_rows, weighted_csv, genotype_fields)
    weighted_parquet_ok = maybe_write_parquet(weighted_rows, weighted_parquet)
    LOGGER.info("Wrote simulated weighted matrix: %s rows=%d", weighted_csv, len(weighted_rows))

    write_csv(
        metadata_rows,
        metadata_csv,
        ["sample", "source_sample", "mutable_site_count", "mutation_path"],
    )
    LOGGER.info("Wrote simulation metadata: %s rows=%d", metadata_csv, len(metadata_rows))

    summary = {
        "source_sample": original_sample,
        "total_sites": len(site_columns),
        "mutable_sites": len(mutable_sites),
        "simulated_samples": len(genotype_rows),
        "summary_json": str(metadata_json),
        "files": {
            "simulated_genotype_csv": str(genotype_csv),
            "simulated_genotype_parquet": str(genotype_parquet) if genotype_parquet_ok else None,
            "simulated_variant_effect_matrix_csv": str(weighted_csv),
            "simulated_variant_effect_matrix_parquet": str(weighted_parquet) if weighted_parquet_ok else None,
            "simulated_metadata_csv": str(metadata_csv),
        },
    }
    metadata_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Wrote simulation summary: %s", metadata_json)
    return summary


def write_summary_json(summary: dict[str, object]) -> None:
    summary_path = Path(str(summary["summary_json"]))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Updated simulation summary: %s", summary_path)


def save_prediction_outputs(
    outdir: Path,
    ranking_rows: list[dict[str, object]],
    rank_sample: str,
) -> dict[str, object]:
    prediction_csv = outdir / "simulated_prediction_ranking.csv"
    prediction_parquet = outdir / "simulated_prediction_ranking.parquet"
    scatter_svg = outdir / "simulated_prediction_ranking.svg"

    fields = ["sample", "predicted_phenotype", "rank", "is_target"]
    write_csv(ranking_rows, prediction_csv, fields)
    prediction_parquet_ok = maybe_write_parquet(ranking_rows, prediction_parquet)
    write_rank_scatter_svg(ranking_rows, rank_sample, scatter_svg)

    target_row = next(row for row in ranking_rows if bool(row["is_target"]))
    LOGGER.info(
        "Saved prediction ranking: sample=%s rank=%s predicted_phenotype=%.6f",
        rank_sample,
        target_row["rank"],
        float(target_row["predicted_phenotype"]),
    )
    return {
        "prediction_ranking_csv": str(prediction_csv),
        "prediction_ranking_parquet": str(prediction_parquet) if prediction_parquet_ok else None,
        "prediction_ranking_svg": str(scatter_svg),
        "target_sample": rank_sample,
        "target_rank": int(target_row["rank"]),
        "target_predicted_phenotype": float(target_row["predicted_phenotype"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate all genotype combinations by mutating selected sites of one sample."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config file path.",
    )
    return parser.parse_args()


def main() -> None:
    global LOGGER
    args = parse_args()
    config = load_yaml_config(args.config)

    genotype_file = str(require_config_value(config, "genotype_file"))
    mutable_sites_file = str(require_config_value(config, "mutable_sites_file"))
    variant_effect_file = str(require_config_value(config, "variant_effect_file"))
    outdir = Path(str(config.get("outdir", "simulated_combination_outputs")))
    sample_name = config.get("sample_name")
    model_file = config.get("model_file")
    rank_order = str(config.get("rank_order", "desc")).lower()
    log_level = str(config.get("log_level", "INFO")).upper()

    if rank_order not in {"desc", "asc"}:
        raise ValueError("rank_order must be one of: desc, asc")

    LOGGER = get_logger(outdir / "simulated_combination.log", level=log_level)
    LOGGER.info("Loaded config: %s", args.config)

    original_sample, site_columns, genotype_map = load_single_sample_genotype(
        genotype_file=genotype_file,
        sample_name=None if sample_name in {None, ""} else str(sample_name),
    )
    mutable_sites = load_mutable_sites(mutable_sites_file)
    validate_mutable_sites_in_genotype(site_columns, mutable_sites)
    effect_map = load_variant_effects(variant_effect_file, site_columns)

    genotype_rows, weighted_rows, metadata_rows = build_simulated_rows(
        original_sample=original_sample,
        site_columns=site_columns,
        genotype_map=genotype_map,
        mutable_sites=mutable_sites,
        effect_map=effect_map,
    )
    summary = save_outputs(
        outdir=outdir,
        site_columns=site_columns,
        genotype_rows=genotype_rows,
        weighted_rows=weighted_rows,
        metadata_rows=metadata_rows,
        original_sample=original_sample,
        mutable_sites=mutable_sites,
    )

    if model_file not in {None, ""}:
        original_genotype_row, original_weighted_row = build_sample_rows(
            sample_id=original_sample,
            site_columns=site_columns,
            genotype_map=genotype_map,
            effect_map=effect_map,
        )
        prediction_weighted_rows = [original_weighted_row] + weighted_rows
        model = load_model(str(model_file))
        predictions = predict_rows(model, prediction_weighted_rows, site_columns)
        ranking_rows = rank_predictions(
            weighted_rows=prediction_weighted_rows,
            predictions=predictions,
            rank_sample=original_sample,
            descending=(rank_order == "desc"),
        )
        summary["prediction"] = save_prediction_outputs(
            outdir=outdir,
            ranking_rows=ranking_rows,
            rank_sample=original_sample,
        )
        summary["prediction"]["rank_target"] = "original_sample"
        summary["prediction"]["original_sample_included"] = True

    write_summary_json(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
