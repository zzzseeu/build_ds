#!/usr/bin/env python3
"""Simulate genotype-feature combinations for one sample across location groups.

Workflow:
1. Read genotype and phenotype matrices with pandas.
2. Merge phenotype columns on the left and genotype columns on the right by ``sample``.
3. Keep only rows for the requested ``sample_id``.
4. Read ``feature_importance.csv`` and select the top-N non-location genotype features.
5. For each location group of the selected sample, enumerate all genotype combinations
   for mutable features using values in {0, 1, 2}.
6. Convert genotype features to weighted variant-effect features via ``var_effect``.
7. Predict phenotype values with the provided model, rank all combinations, and
   generate scatter plots highlighting the original sample combination.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import pickle
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from utils import get_logger


LOGGER = None
VALID_GENOTYPES = (0, 1, 2)
SITE_ID_PATTERN = re.compile(r"^Chr\d+:\d+$")


def _format_log_message(message: str, *args: object) -> str:
    if not args:
        return message
    try:
        return message.format(*args)
    except Exception:
        return " ".join([message, *[str(arg) for arg in args]])


def log_info(message: str, *args: object) -> None:
    if LOGGER is not None:
        LOGGER.info(_format_log_message(message, *args))


def log_warning(message: str, *args: object) -> None:
    if LOGGER is not None:
        LOGGER.warning(_format_log_message(message, *args))


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


def maybe_write_parquet(df: pd.DataFrame, path: Path) -> bool:
    try:
        df.to_parquet(path, index=False)
        log_info("Wrote parquet: {} rows={}", path, len(df))
        return True
    except Exception as exc:
        log_warning("Failed to write parquet {}: {}", path, exc)
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


def get_model_feature_order(model, feature_columns: Sequence[str]) -> list[str]:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return list(feature_columns)
    return [str(name) for name in feature_names]


def predict_frame(model, feature_df: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    feature_order = get_model_feature_order(model, feature_columns)
    missing = [feature for feature in feature_order if feature not in feature_df.columns]
    if missing:
        raise ValueError(
            "Model expects features missing from prediction matrix: "
            + ",".join(missing[:20])
            + ("..." if len(missing) > 20 else "")
        )
    matrix = feature_df.loc[:, feature_order]
    try:
        predictions = model.predict(matrix)
    except Exception as exc:
        raise RuntimeError(f"Model prediction failed: {exc}") from exc
    return np.asarray(predictions, dtype=float)


def validate_genotype_columns(columns: Sequence[str]) -> None:
    invalid_columns = [column for column in columns if SITE_ID_PATTERN.fullmatch(str(column)) is None]
    if invalid_columns:
        raise ValueError(
            "Genotype feature columns must use the format 'ChrN:Pos'. Invalid columns: "
            + ",".join(invalid_columns[:20])
            + ("..." if len(invalid_columns) > 20 else "")
        )


def load_input_matrices(genotype_file: str, phenotype_file: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    genotype_path = Path(genotype_file)
    phenotype_path = Path(phenotype_file)
    if not genotype_path.exists():
        raise FileNotFoundError(f"Genotype file not found: {genotype_path}")
    if not phenotype_path.exists():
        raise FileNotFoundError(f"Phenotype file not found: {phenotype_path}")

    genotype_df = pd.read_csv(genotype_path)
    phenotype_df = pd.read_csv(phenotype_path)
    if genotype_df.empty:
        raise ValueError(f"Genotype file is empty: {genotype_path}")
    if phenotype_df.empty:
        raise ValueError(f"Phenotype file is empty: {phenotype_path}")

    genotype_columns = list(genotype_df.columns)
    phenotype_columns = list(phenotype_df.columns)
    if genotype_columns[0] != "sample":
        raise ValueError("genotype_file first column must be 'sample'")
    if len(phenotype_columns) < 3 or phenotype_columns[0] != "sample" or phenotype_columns[1] != "value":
        raise ValueError("phenotype_file first two columns must be 'sample' and 'value'")

    location_columns = [column for column in phenotype_columns[2:] if str(column).startswith("loc")]
    non_location_columns = [column for column in phenotype_columns[2:] if column not in location_columns]
    if non_location_columns:
        raise ValueError(
            "All phenotype feature columns after 'value' must start with 'loc'. Invalid columns: "
            + ",".join(non_location_columns[:20])
            + ("..." if len(non_location_columns) > 20 else "")
        )

    genotype_feature_columns = genotype_columns[1:]
    if not genotype_feature_columns:
        raise ValueError("genotype_file must contain genotype feature columns after 'sample'")
    validate_genotype_columns(genotype_feature_columns)

    genotype_df = genotype_df.copy()
    phenotype_df = phenotype_df.copy()
    genotype_df["sample"] = genotype_df["sample"].astype(str)
    phenotype_df["sample"] = phenotype_df["sample"].astype(str)

    for column in genotype_feature_columns:
        genotype_df[column] = pd.to_numeric(genotype_df[column], errors="raise").astype(int)
        invalid_mask = ~genotype_df[column].isin(VALID_GENOTYPES)
        if bool(invalid_mask.any()):
            raise ValueError(f"Genotype column contains values outside {VALID_GENOTYPES}: {column}")

    phenotype_df["value"] = pd.to_numeric(phenotype_df["value"], errors="raise").astype(float)
    for column in location_columns:
        phenotype_df[column] = pd.to_numeric(phenotype_df[column], errors="raise").astype(int)

    log_info(
        "Loaded input matrices: genotype_rows={} phenotype_rows={} genotype_features={} location_features={}",
        len(genotype_df),
        len(phenotype_df),
        len(genotype_feature_columns),
        len(location_columns),
    )
    return genotype_df, phenotype_df, genotype_feature_columns, location_columns


def build_merged_matrix(phenotype_df: pd.DataFrame, genotype_df: pd.DataFrame) -> pd.DataFrame:
    merged = phenotype_df.merge(genotype_df, on="sample", how="inner", sort=False)
    phenotype_columns = list(phenotype_df.columns)
    genotype_columns = [column for column in genotype_df.columns if column != "sample"]
    merged = merged.loc[:, phenotype_columns + genotype_columns]
    if merged.empty:
        raise ValueError("Merged phenotype/genotype matrix is empty. Please check shared sample IDs.")
    log_info("Built merged matrix: rows={} columns={}", len(merged), len(merged.columns))
    return merged


def filter_sample_rows(merged_df: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    filtered = merged_df[merged_df["sample"].astype(str) == str(sample_id)].copy()
    if filtered.empty:
        raise ValueError(f"sample_id '{sample_id}' not found in merged phenotype/genotype matrix")
    filtered = filtered.reset_index(drop=True)
    log_info("Filtered merged matrix by sample: sample_id={} rows={}", sample_id, len(filtered))
    return filtered


def load_mutable_features(
    feature_importance_file: str,
    genotype_feature_columns: Sequence[str],
    top_n: int,
) -> list[str]:
    path = Path(feature_importance_file)
    if not path.exists():
        raise FileNotFoundError(f"Feature-importance file not found: {path}")

    importance_df = pd.read_csv(path)
    required_columns = ["feature", "importance_gain", "importance_split"]
    missing_columns = [column for column in required_columns if column not in importance_df.columns]
    if missing_columns:
        raise ValueError("feature_importance_file missing columns: " + ",".join(missing_columns))

    importance_df = importance_df.copy()
    importance_df["feature"] = importance_df["feature"].astype(str)
    importance_df["importance_gain"] = pd.to_numeric(importance_df["importance_gain"], errors="raise").astype(float)
    importance_df["importance_split"] = pd.to_numeric(importance_df["importance_split"], errors="raise").astype(float)

    genotype_feature_set = set(str(column) for column in genotype_feature_columns)
    filtered = importance_df[
        importance_df["feature"].isin(genotype_feature_set)
        & ~importance_df["feature"].str.startswith("loc")
    ].copy()
    filtered = filtered.sort_values(["importance_gain", "importance_split"], ascending=[False, False])
    mutable_features = filtered["feature"].head(int(top_n)).tolist()
    if not mutable_features:
        raise ValueError("No mutable genotype features found in feature_importance_file")

    missing_from_genotype = [feature for feature in mutable_features if feature not in genotype_feature_set]
    if missing_from_genotype:
        raise ValueError("Mutable features missing from genotype matrix: " + ",".join(missing_from_genotype))

    log_info("Selected mutable features: top_n={} selected={}", top_n, len(mutable_features))
    return mutable_features


def load_variant_effects(variant_effect_file: str, genotype_feature_columns: Sequence[str]) -> dict[str, float]:
    path = Path(variant_effect_file)
    if not path.exists():
        raise FileNotFoundError(f"Variant-effect file not found: {path}")

    effect_df = pd.read_csv(path)
    required_columns = ["Chromosome", "Position", "site_id", "var_effect"]
    missing_columns = [column for column in required_columns if column not in effect_df.columns]
    if missing_columns:
        raise ValueError("variant_effect_file missing columns: " + ",".join(missing_columns))

    effect_df = effect_df.copy()
    effect_df["site_id"] = effect_df["site_id"].astype(str)
    effect_df["var_effect"] = pd.to_numeric(effect_df["var_effect"], errors="raise").astype(float)
    effect_map = dict(zip(effect_df["site_id"], effect_df["var_effect"]))

    missing_effects = [feature for feature in genotype_feature_columns if feature not in effect_map]
    if missing_effects:
        raise ValueError(
            "variant_effect_file is missing genotype feature effects: "
            + ",".join(missing_effects[:20])
            + ("..." if len(missing_effects) > 20 else "")
        )
    log_info("Loaded variant effects: count={} source={}", len(effect_map), path)
    return effect_map


def group_filtered_rows_by_location(filtered_df: pd.DataFrame, location_columns: Sequence[str]) -> list[dict[str, object]]:
    if not location_columns:
        raise ValueError("No location columns found in phenotype_file")

    grouped: list[dict[str, object]] = []
    groupby_obj = filtered_df.groupby(list(location_columns), dropna=False, sort=False)
    for index, (group_key, group_df) in enumerate(groupby_obj, start=1):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        location_features = {
            column: int(value)
            for column, value in zip(location_columns, group_key)
        }
        active_locations = [column for column, value in location_features.items() if int(value) == 1]
        location_label = "__".join(active_locations) if active_locations else f"location_group_{index:03d}"
        grouped.append(
            {
                "group_id": f"group_{index:03d}",
                "location_label": location_label,
                "location_features": location_features,
                "row_count": int(len(group_df)),
                "actual_value_mean": float(group_df["value"].mean()),
                "group_df": group_df.reset_index(drop=True),
            }
        )
    log_info("Grouped filtered rows by location: groups={}", len(grouped))
    return grouped


def build_combination_frame(
    base_row: pd.Series,
    sample_id: str,
    genotype_feature_columns: Sequence[str],
    mutable_features: Sequence[str],
    effect_map: dict[str, float],
) -> pd.DataFrame:
    mutable_features = list(mutable_features)
    mutable_feature_set = set(mutable_features)
    mutable_feature_index = {feature: idx for idx, feature in enumerate(mutable_features)}
    genotype_feature_columns = list(genotype_feature_columns)
    total_combinations = len(VALID_GENOTYPES) ** len(mutable_features)
    if total_combinations <= 0:
        raise ValueError("No combinations generated for mutable features")

    weighted_matrix = np.empty((total_combinations, len(genotype_feature_columns)), dtype=np.float32)
    target_index = 0
    mutable_count = len(mutable_features)

    for column_index, column in enumerate(genotype_feature_columns):
        effect = np.float32(effect_map[column])
        if column in mutable_feature_set:
            mutable_index = mutable_feature_index[column]
            repeat_each = 3 ** (mutable_count - mutable_index - 1)
            tile_count = 3 ** mutable_index
            genotype_pattern = np.tile(np.repeat(np.asarray(VALID_GENOTYPES, dtype=np.int8), repeat_each), tile_count)
            weighted_matrix[:, column_index] = genotype_pattern.astype(np.float32) * effect
            target_index += int(base_row[column]) * (3 ** (mutable_count - mutable_index - 1))
        else:
            weighted_matrix[:, column_index] = np.float32(base_row[column]) * effect

    if target_index < 0 or target_index >= total_combinations:
        raise ValueError("Original genotype combination index is out of bounds")

    target_mask = np.zeros(total_combinations, dtype=bool)
    target_mask[target_index] = True

    combo_ids = [f"sim_{idx:06d}" for idx in range(total_combinations)]
    samples = [f"{sample_id}_sim_{idx:06d}" for idx in range(total_combinations)]
    combo_ids[target_index] = "original"
    samples[target_index] = str(sample_id)

    feature_df = pd.DataFrame({
        "sample": samples,
        "combination_id": combo_ids,
        "is_target": target_mask.astype(bool),
        "source_sample": [str(sample_id)] * total_combinations,
    })
    genotype_feature_df = pd.DataFrame(weighted_matrix, columns=genotype_feature_columns, copy=False)
    feature_df = pd.concat([feature_df, genotype_feature_df], axis=1)
    return feature_df


def append_location_features(
    simulated_feature_df: pd.DataFrame,
    location_features: dict[str, int],
) -> pd.DataFrame:
    prediction_df = simulated_feature_df.copy()
    for column, value in location_features.items():
        prediction_df[column] = int(value)
    return prediction_df


def rank_predictions(
    prediction_df: pd.DataFrame,
    predictions: np.ndarray,
    descending: bool,
) -> pd.DataFrame:
    ranking_df = prediction_df.loc[:, ["sample", "combination_id", "is_target", "source_sample"]].copy()
    ranking_df["predicted_phenotype"] = np.asarray(predictions, dtype=float)
    ranking_df = ranking_df.sort_values("predicted_phenotype", ascending=not descending).reset_index(drop=True)
    ranking_df["rank"] = np.arange(1, len(ranking_df) + 1, dtype=int)
    return ranking_df.loc[:, ["sample", "combination_id", "predicted_phenotype", "rank", "is_target", "source_sample"]]


def _scale_value(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if math.isclose(src_min, src_max):
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def write_rank_scatter_svg(
    ranking_df: pd.DataFrame,
    target_sample: str,
    output_path: Path,
) -> None:
    rows = ranking_df.to_dict(orient="records")
    width = 1200
    height = 700
    left = 90
    right = 40
    top = 40
    bottom = 80
    plot_width = width - left - right
    plot_height = height - top - bottom

    ranks = [int(row["rank"]) for row in rows]
    values = [float(row["predicted_phenotype"]) for row in rows]
    min_rank = min(ranks)
    max_rank = max(ranks)
    min_value = min(values)
    max_value = max(values)

    points: list[str] = []
    target_label = ""
    for row in rows:
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
                f'{target_sample} (rank={row["rank"]}, pred={row["predicted_phenotype"]:.6f})'
                "</text>"
                f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x + 6:.2f}" y2="{label_y - 5:.2f}" stroke="#d62728" stroke-width="1.5" />'
            )

    axis_lines = [
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333" stroke-width="1.5" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333" stroke-width="1.5" />',
    ]

    x_ticks = []
    tick_count = min(10, len(rows))
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
<text x="{width - 160}" y="48" font-size="12" fill="#444">background combinations</text>
<circle cx="{width - 170}" cy="66" r="5" fill="#d62728" />
<text x="{width - 160}" y="70" font-size="12" fill="#444">target sample: {target_sample}</text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")
    log_info("Wrote ranking scatter SVG: {}", output_path)


def save_group_outputs(
    outdir: Path,
    group_id: str,
    merged_group_df: pd.DataFrame,
    prediction_feature_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    target_sample: str,
) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)

    merged_csv = outdir / f"merged_pheno_geno_{group_id}.csv"
    merged_parquet = outdir / f"merged_pheno_geno_{group_id}.parquet"
    prediction_feature_csv = outdir / f"prediction_features_{group_id}.csv"
    prediction_feature_parquet = outdir / f"prediction_features_{group_id}.parquet"
    ranking_csv = outdir / f"simulated_prediction_ranking_{group_id}.csv"
    ranking_parquet = outdir / f"simulated_prediction_ranking_{group_id}.parquet"
    scatter_svg = outdir / f"simulated_prediction_ranking_{group_id}.svg"

    merged_group_df.to_csv(merged_csv, index=False)
    merged_parquet_ok = maybe_write_parquet(merged_group_df, merged_parquet)
    prediction_feature_df.to_csv(prediction_feature_csv, index=False)
    prediction_feature_parquet_ok = maybe_write_parquet(prediction_feature_df, prediction_feature_parquet)
    ranking_df.to_csv(ranking_csv, index=False)
    ranking_parquet_ok = maybe_write_parquet(ranking_df, ranking_parquet)
    write_rank_scatter_svg(ranking_df, target_sample, scatter_svg)

    target_row = ranking_df[ranking_df["is_target"]].iloc[0]
    return {
        "group_id": group_id,
        "merged_pheno_geno_csv": str(merged_csv),
        "merged_pheno_geno_parquet": str(merged_parquet) if merged_parquet_ok else None,
        "prediction_feature_csv": str(prediction_feature_csv),
        "prediction_feature_parquet": str(prediction_feature_parquet) if prediction_feature_parquet_ok else None,
        "prediction_ranking_csv": str(ranking_csv),
        "prediction_ranking_parquet": str(ranking_parquet) if ranking_parquet_ok else None,
        "prediction_ranking_svg": str(scatter_svg),
        "target_rank": int(target_row["rank"]),
        "target_predicted_phenotype": float(target_row["predicted_phenotype"]),
    }


def write_summary_json(summary: dict[str, object]) -> None:
    summary_path = Path(str(summary["summary_json"]))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log_info("Updated simulation summary: {}", summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate mutable genotype-feature combinations for one sample across phenotype location groups."
    )
    parser.add_argument("--config", required=True, help="YAML config file path.")
    return parser.parse_args()


def main() -> None:
    global LOGGER
    args = parse_args()
    config = load_yaml_config(args.config)

    genotype_file = str(require_config_value(config, "genotype_file"))
    phenotype_file = str(require_config_value(config, "phenotype_file"))
    feature_importance_file = str(require_config_value(config, "feature_importance_file"))
    variant_effect_file = str(require_config_value(config, "variant_effect_file"))
    model_file = str(require_config_value(config, "model_file"))
    sample_id = str(require_config_value(config, "sample_id"))
    outdir = Path(str(config.get("outdir", "simulated_combination_outputs")))
    top_n = int(config.get("top_n", 10))
    rank_order = str(config.get("rank_order", "desc")).lower()
    log_level = str(config.get("log_level", "INFO")).upper()

    if rank_order not in {"desc", "asc"}:
        raise ValueError("rank_order must be one of: desc, asc")
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer")

    LOGGER = get_logger(outdir / "simulated_combination.log", level=log_level)
    log_info("Loaded config: {}", args.config)

    genotype_df, phenotype_df, genotype_feature_columns, location_columns = load_input_matrices(
        genotype_file=genotype_file,
        phenotype_file=phenotype_file,
    )
    merged_pheno_geno = build_merged_matrix(phenotype_df, genotype_df)
    merged_sample_df = filter_sample_rows(merged_pheno_geno, sample_id)
    mutable_features = load_mutable_features(
        feature_importance_file=feature_importance_file,
        genotype_feature_columns=genotype_feature_columns,
        top_n=top_n,
    )
    effect_map = load_variant_effects(
        variant_effect_file=variant_effect_file,
        genotype_feature_columns=genotype_feature_columns,
    )
    location_groups = group_filtered_rows_by_location(merged_sample_df, location_columns)
    model = load_model(model_file)

    outdir.mkdir(parents=True, exist_ok=True)
    merged_sample_csv = outdir / "merged_pheno_geno_filtered.csv"
    merged_sample_parquet = outdir / "merged_pheno_geno_filtered.parquet"
    simulated_feature_csv = outdir / "simulated_weighted_features.csv"
    simulated_feature_parquet = outdir / "simulated_weighted_features.parquet"
    merged_sample_df.to_csv(merged_sample_csv, index=False)
    merged_sample_parquet_ok = maybe_write_parquet(merged_sample_df, merged_sample_parquet)

    group_summaries: list[dict[str, object]] = []
    feature_columns_for_prediction = list(location_columns) + list(genotype_feature_columns)
    descending = rank_order == "desc"
    total_combinations = len(VALID_GENOTYPES) ** len(mutable_features)
    base_row = merged_sample_df.iloc[0]
    simulated_feature_df = build_combination_frame(
        base_row=base_row,
        sample_id=sample_id,
        genotype_feature_columns=genotype_feature_columns,
        mutable_features=mutable_features,
        effect_map=effect_map,
    )
    simulated_feature_df.to_csv(simulated_feature_csv, index=False)
    simulated_feature_parquet_ok = maybe_write_parquet(simulated_feature_df, simulated_feature_parquet)

    for group in location_groups:
        group_id = str(group["group_id"])
        group_df = pd.DataFrame(group["group_df"])
        prediction_df = append_location_features(
            simulated_feature_df=simulated_feature_df,
            location_features=dict(group["location_features"]),
        )
        prediction_df = prediction_df.loc[:, ["sample", "combination_id", "is_target", "source_sample"] + feature_columns_for_prediction]
        predictions = predict_frame(model, prediction_df, feature_columns_for_prediction)
        ranking_df = rank_predictions(prediction_df, predictions, descending=descending)

        group_outdir = outdir / "prediction_by_location" / group_id
        group_summary = save_group_outputs(
            outdir=group_outdir,
            group_id=group_id,
            merged_group_df=group_df,
            prediction_feature_df=prediction_df,
            ranking_df=ranking_df,
            target_sample=sample_id,
        )
        group_summary["location_label"] = str(group["location_label"])
        group_summary["row_count"] = int(group["row_count"])
        group_summary["actual_value_mean"] = float(group["actual_value_mean"])
        group_summary["location_features"] = dict(group["location_features"])
        group_summaries.append(group_summary)

    summary = {
        "summary_json": str(outdir / "simulated_summary.json"),
        "sample_id": sample_id,
        "merged_rows": int(len(merged_sample_df)),
        "genotype_feature_count": len(genotype_feature_columns),
        "location_feature_count": len(location_columns),
        "mutable_feature_count": len(mutable_features),
        "mutable_features": mutable_features,
        "combinations_per_group": int(total_combinations),
        "files": {
            "merged_pheno_geno_filtered_csv": str(merged_sample_csv),
            "merged_pheno_geno_filtered_parquet": str(merged_sample_parquet) if merged_sample_parquet_ok else None,
            "simulated_feature_csv": str(simulated_feature_csv),
            "simulated_feature_parquet": str(simulated_feature_parquet) if simulated_feature_parquet_ok else None,
        },
        "prediction": {
            "mode": "grouped_by_location",
            "rank_order": rank_order,
            "groups": group_summaries,
        },
    }
    write_summary_json(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
