#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pysam

from site_dataset_builder import SiteDatasetBuilder, standard_chrom


def build_all_vcf_site_df(vcf_path: str) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    with pysam.VariantFile(vcf_path) as vcf:
        for rec in vcf:
            chrom = standard_chrom(str(rec.chrom))
            if chrom is None:
                continue
            ref = str(rec.ref).upper()
            alts = rec.alts or ()
            for alt in alts:
                if alt is None:
                    continue
                rows.append(
                    {
                        "Chromosome": chrom,
                        "Position": int(rec.pos),
                        "REF": ref,
                        "ALT": str(alt).upper(),
                    }
                )
    if not rows:
        raise ValueError(f"No valid variant rows found in VCF: {vcf_path}")
    return pd.DataFrame(rows, columns=["Chromosome", "Position", "REF", "ALT"])


def build_minimal_builder(
    *,
    vcf_path: str,
    fasta_path: str,
    outdir: str,
    embedder_type: str,
    model_name_or_path: str,
    device: str,
    pooling: str,
    local_files_only: bool,
    k: int,
) -> SiteDatasetBuilder:
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    dummy_csv = outdir_path / "_dummy_empty.csv"
    if not dummy_csv.exists():
        pd.DataFrame({"placeholder": []}).to_csv(dummy_csv, index=False)

    return SiteDatasetBuilder(
        {
            "gwas_csv_path": str(dummy_csv),
            "qtl_csv_path": str(dummy_csv),
            "gff3_path": str(dummy_csv),
            "fasta_path": fasta_path,
            "vcf_path": vcf_path,
            "type": "union",
            "outdir": outdir,
            "k": int(k),
            "embedder_type": embedder_type,
            "model_name_or_path": model_name_or_path,
            "device": device,
            "pooling": pooling,
            "local_files_only": bool(local_files_only),
        }
    )


def compute_all_vcf_variant_effects(
    *,
    vcf_path: str,
    fasta_path: str,
    outdir: str,
    embedder_type: str,
    model_name_or_path: str,
    device: str = "cpu",
    pooling: str = "mean",
    local_files_only: bool = True,
    k: int = 100,
) -> pd.DataFrame:
    builder = build_minimal_builder(
        vcf_path=vcf_path,
        fasta_path=fasta_path,
        outdir=outdir,
        embedder_type=embedder_type,
        model_name_or_path=model_name_or_path,
        device=device,
        pooling=pooling,
        local_files_only=local_files_only,
        k=k,
    )
    site_df = build_all_vcf_site_df(vcf_path)
    variant_effect_df = builder._compute_variant_effects(site_df)

    outdir_path = Path(outdir)
    out_csv = outdir_path / "all_vcf_site_variant_effect_by_site.csv"
    out_parquet = outdir_path / "all_vcf_site_variant_effect_by_site.parquet"
    variant_effect_df.to_csv(out_csv, index=False)
    variant_effect_df.to_parquet(out_parquet, index=False)
    return variant_effect_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute variant-effect scores for all sites in a VCF file.")
    parser.add_argument("--vcf-path", required=True, help="Input VCF/VCF.GZ file")
    parser.add_argument("--fasta-path", required=True, help="Reference FASTA file")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--embedder-type", required=True, help="Embedder type passed to SiteDatasetBuilder")
    parser.add_argument("--model-name-or-path", required=True, help="Embedding model name or local path")
    parser.add_argument("--device", default="cpu", help="Embedding device, e.g. cpu or cuda")
    parser.add_argument("--pooling", default="mean", help="Embedding pooling strategy")
    parser.add_argument("--local-files-only", action="store_true", help="Load model from local files only")
    parser.add_argument("--k", type=int, default=100, help="Flanking sequence length on each side of the variant")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variant_effect_df = compute_all_vcf_variant_effects(
        vcf_path=args.vcf_path,
        fasta_path=args.fasta_path,
        outdir=args.outdir,
        embedder_type=args.embedder_type,
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        pooling=args.pooling,
        local_files_only=bool(args.local_files_only),
        k=int(args.k),
    )
    print(
        f"Computed variant effects for {len(variant_effect_df)} rows. "
        f"Outputs saved under: {Path(args.outdir).resolve()}"
    )


if __name__ == "__main__":
    main()
