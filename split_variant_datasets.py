"""Split genotype and gene-feature datasets into train_val and test subsets.

Inputs are the `genotype_012` matrix, the `gene_feature_matrix`, and the site
table produced by `gwas_qtl_variant_extractor.py`. Splitting is sample-based.
Optional gene filtering keeps only SNP features mapped to the requested genes
and only gene-feature columns derived from those genes.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from logger import init_logger


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format for {path}; expected .csv or .parquet")


def _save_dual(df: pd.DataFrame, path_without_suffix: Path) -> tuple[Path, Path]:
    csv_path = path_without_suffix.with_suffix(".csv")
    parquet_path = path_without_suffix.with_suffix(".parquet")
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    return csv_path, parquet_path


def _read_name_df(path: Path | None) -> list[str]:
    if path is None:
        return []
    df = pd.read_csv(path)
    if df.empty or df.shape[1] == 0:
        return []
    values = df.iloc[:, 0].astype(str).map(str.strip)
    return [value for value in values.tolist() if value]


def _extract_gene_from_feature_col(col: str) -> str | None:
    if col == "sample":
        return None
    if "-PC" in col:
        return col.split("-PC", 1)[0]
    if "-embed-" in col:
        return col.split("-embed-", 1)[0]
    return None


def _filter_genotype_columns(
    genotype_df: pd.DataFrame,
    site_df: pd.DataFrame,
    gene_list: list[str],
) -> pd.DataFrame:
    if not gene_list:
        return genotype_df

    gene_set = set(gene_list)
    site_subset = site_df[site_df["Gene"].astype(str).isin(gene_set)].copy()
    site_subset["feature_col"] = (
        site_subset["Chromosome"].astype(str) + ":" + site_subset["Position"].astype(str)
    )
    keep_cols = ["sample"] + [col for col in site_subset["feature_col"].drop_duplicates().tolist() if col in genotype_df.columns]
    return genotype_df.loc[:, keep_cols].copy()


def _filter_gene_feature_columns(
    gene_feature_df: pd.DataFrame,
    gene_list: list[str],
) -> pd.DataFrame:
    if not gene_list:
        return gene_feature_df

    gene_set = set(gene_list)
    keep_cols = ["sample"] + [
        col
        for col in gene_feature_df.columns
        if col != "sample" and _extract_gene_from_feature_col(str(col)) in gene_set
    ]
    return gene_feature_df.loc[:, keep_cols].copy()


def _align_by_sample(
    genotype_df: pd.DataFrame,
    gene_feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    geno_samples = genotype_df["sample"].astype(str)
    feature_samples = gene_feature_df["sample"].astype(str)
    shared_samples = [sample for sample in geno_samples.tolist() if sample in set(feature_samples.tolist())]
    if not shared_samples:
        raise ValueError("No shared samples found between genotype_012 and gene_feature_matrix")

    genotype_aligned = genotype_df.set_index("sample").loc[shared_samples].reset_index()
    gene_feature_aligned = gene_feature_df.set_index("sample").loc[shared_samples].reset_index()
    return genotype_aligned, gene_feature_aligned


def _build_split_sample_lists(
    sample_list: list[str],
    test_ratio: float,
    isolated_sample_list: list[str],
    random_state: int,
) -> tuple[list[str], list[str]]:
    sample_set = set(sample_list)
    isolated = [sample for sample in isolated_sample_list if sample in sample_set]
    if len(isolated) != len(set(isolated)):
        isolated = list(dict.fromkeys(isolated))

    target_test_size = max(len(isolated), int(math.ceil(len(sample_list) * test_ratio)))
    available = [sample for sample in sample_list if sample not in set(isolated)]
    rng = np.random.default_rng(random_state)
    extra_needed = max(0, target_test_size - len(isolated))
    if extra_needed > len(available):
        extra_needed = len(available)
    extra_test = rng.choice(np.array(available, dtype=object), size=extra_needed, replace=False).tolist() if extra_needed > 0 else []
    test_samples = isolated + extra_test
    test_set = set(test_samples)
    train_val_samples = [sample for sample in sample_list if sample not in test_set]
    return train_val_samples, test_samples


def _subset_by_samples(df: pd.DataFrame, sample_list: list[str]) -> pd.DataFrame:
    return df.set_index("sample").loc[sample_list].reset_index()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split genotype_012 and gene_feature_matrix into train_val_ds and test_ds"
    )
    parser.add_argument("--genotype_012_path", required=True, help="Path to genotype_012 .csv or .parquet")
    parser.add_argument("--gene_feature_matrix_path", required=True, help="Path to gene_feature_matrix .csv or .parquet")
    parser.add_argument("--site_df_path", required=True, help="Path to site file produced by gwas_qtl_variant_extractor.py")
    parser.add_argument("--outdir", required=True, help="Output directory for split datasets")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Fraction of samples kept in test set")
    parser.add_argument(
        "--isolated_sample_df",
        default=None,
        help="Optional CSV file; the first column contains sample names that must be kept in test set",
    )
    parser.add_argument(
        "--gene_df",
        default=None,
        help="Optional CSV file; the first column contains gene names used to filter feature columns",
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for split reproducibility")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / f"split_variant_datasets_{datetime.now().strftime('%Y-%m-%d')}.log"
    logger = init_logger("SplitVariantDatasets", log_file=log_path)

    genotype_path = Path(args.genotype_012_path)
    gene_feature_path = Path(args.gene_feature_matrix_path)
    site_df_path = Path(args.site_df_path)

    gene_list = _read_name_df(Path(args.gene_df)) if args.gene_df else []
    isolated_sample_list = _read_name_df(Path(args.isolated_sample_df)) if args.isolated_sample_df else []
    if not 0 < float(args.test_ratio) < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    logger.info("main: loading input tables")
    genotype_df = _read_table(genotype_path)
    gene_feature_df = _read_table(gene_feature_path)
    site_df = _read_table(site_df_path)

    logger.info(
        "main: loaded genotype shape=%s gene_feature shape=%s site_df shape=%s",
        genotype_df.shape,
        gene_feature_df.shape,
        site_df.shape,
    )

    genotype_df = _filter_genotype_columns(genotype_df, site_df, gene_list)
    gene_feature_df = _filter_gene_feature_columns(gene_feature_df, gene_list)
    genotype_df, gene_feature_df = _align_by_sample(genotype_df, gene_feature_df)

    logger.info(
        "main: filtered genotype shape=%s gene_feature shape=%s shared_samples=%d",
        genotype_df.shape,
        gene_feature_df.shape,
        len(genotype_df),
    )

    sample_list = genotype_df["sample"].astype(str).tolist()
    train_val_samples, test_samples = _build_split_sample_lists(
        sample_list=sample_list,
        test_ratio=float(args.test_ratio),
        isolated_sample_list=isolated_sample_list,
        random_state=int(args.random_state),
    )

    train_val_genotype = _subset_by_samples(genotype_df, train_val_samples)
    test_genotype = _subset_by_samples(genotype_df, test_samples)
    train_val_gene_feature = _subset_by_samples(gene_feature_df, train_val_samples)
    test_gene_feature = _subset_by_samples(gene_feature_df, test_samples)

    geno_train_csv, geno_train_parquet = _save_dual(train_val_genotype, outdir / "train_val_ds_genotype_012")
    geno_test_csv, geno_test_parquet = _save_dual(test_genotype, outdir / "test_ds_genotype_012")
    feat_train_csv, feat_train_parquet = _save_dual(train_val_gene_feature, outdir / "train_val_ds_gene_feature_matrix")
    feat_test_csv, feat_test_parquet = _save_dual(test_gene_feature, outdir / "test_ds_gene_feature_matrix")

    meta = {
        "n_samples": len(sample_list),
        "test_ratio": float(args.test_ratio),
        "random_state": int(args.random_state),
        "isolated_sample_list": isolated_sample_list,
        "gene_list": gene_list,
        "shapes": {
            "train_val_genotype_012": list(train_val_genotype.shape),
            "test_genotype_012": list(test_genotype.shape),
            "train_val_gene_feature_matrix": list(train_val_gene_feature.shape),
            "test_gene_feature_matrix": list(test_gene_feature.shape),
        },
        "samples": {
            "train_val": train_val_samples,
            "test": test_samples,
        },
        "outputs": {
            "train_val_genotype_012_csv": str(geno_train_csv),
            "train_val_genotype_012_parquet": str(geno_train_parquet),
            "test_genotype_012_csv": str(geno_test_csv),
            "test_genotype_012_parquet": str(geno_test_parquet),
            "train_val_gene_feature_matrix_csv": str(feat_train_csv),
            "train_val_gene_feature_matrix_parquet": str(feat_train_parquet),
            "test_gene_feature_matrix_csv": str(feat_test_csv),
            "test_gene_feature_matrix_parquet": str(feat_test_parquet),
        },
    }
    meta_path = outdir / "split_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "main: saved train_val/test splits train_val_samples=%d test_samples=%d meta=%s",
        len(train_val_samples),
        len(test_samples),
        meta_path,
    )
    print(json.dumps({"train_val_samples": len(train_val_samples), "test_samples": len(test_samples)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
