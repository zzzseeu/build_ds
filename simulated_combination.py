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
   - either ``site_id,var_effect``
   - or ``Chromosome,Position,var_effect``

For each mutable site, the script enumerates the two genotype values different
from the original genotype. If the original genotype is 0, the mutated values
are 1 and 2. Therefore, N mutable sites produce 2^N simulated samples.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Sequence

from utils import get_logger


LOGGER = None
VALID_GENOTYPES = {0, 1, 2}


@dataclass(frozen=True)
class MutableSite:
    chrom: str
    pos: int

    @property
    def site_id(self) -> str:
        return f"{self.chrom}:{self.pos}"


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
    required = {"Chromosome", "Position"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Mutable-site file must contain columns: {sorted(required)}")

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

    effect_map: dict[str, float] = {}
    columns = set(rows[0].keys())
    if {"site_id", "var_effect"}.issubset(columns):
        for row in rows:
            effect_map[str(row["site_id"]).strip()] = float(str(row["var_effect"]).strip())
    elif {"Chromosome", "Position", "var_effect"}.issubset(columns):
        for row in rows:
            site_id = f"{str(row['Chromosome']).strip()}:{int(str(row['Position']).strip())}"
            effect_map[site_id] = float(str(row["var_effect"]).strip())
    else:
        raise ValueError(
            "Variant-effect file must contain either columns [site_id, var_effect] "
            "or [Chromosome, Position, var_effect]"
        )

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

        genotype_row: dict[str, object] = {"sample": sample_id}
        weighted_row: dict[str, object] = {"sample": sample_id}
        for site_id in site_columns:
            genotype_value = int(simulated_genotypes[site_id])
            genotype_row[site_id] = genotype_value
            weighted_row[site_id] = float(genotype_value) * float(effect_map[site_id])
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate all genotype combinations by mutating selected sites of one sample."
    )
    parser.add_argument(
        "--genotype-file",
        required=True,
        help="Input genotype matrix CSV. First column must be sample, remaining columns are site ids.",
    )
    parser.add_argument(
        "--mutable-sites-file",
        required=True,
        help="CSV containing mutable sites. Required columns: Chromosome, Position.",
    )
    parser.add_argument(
        "--variant-effect-file",
        required=True,
        help="Variant-effect CSV with either [site_id,var_effect] or [Chromosome,Position,var_effect].",
    )
    parser.add_argument(
        "--sample-name",
        default=None,
        help="Sample name to simulate. Required only when genotype file contains multiple samples.",
    )
    parser.add_argument(
        "--outdir",
        default="simulated_combination_outputs",
        help="Output directory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> None:
    global LOGGER
    args = parse_args()
    outdir = Path(args.outdir)
    LOGGER = get_logger(outdir / "simulated_combination.log", level=args.log_level)

    original_sample, site_columns, genotype_map = load_single_sample_genotype(
        genotype_file=args.genotype_file,
        sample_name=args.sample_name,
    )
    mutable_sites = load_mutable_sites(args.mutable_sites_file)
    validate_mutable_sites_in_genotype(site_columns, mutable_sites)
    effect_map = load_variant_effects(args.variant_effect_file, site_columns)

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
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
