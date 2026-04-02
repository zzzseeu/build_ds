#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pysam

from site_dataset_builder import SiteDatasetBuilder, standard_chrom
from utils import get_logger


THREAD_LOCAL = threading.local()


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


def split_dataframe(df: pd.DataFrame, num_chunks: int) -> list[pd.DataFrame]:
    if df.empty:
        return []
    num_chunks = max(1, min(int(num_chunks), len(df)))
    chunk_size = math.ceil(len(df) / num_chunks)
    return [df.iloc[start:start + chunk_size].reset_index(drop=True) for start in range(0, len(df), chunk_size)]


def get_thread_builder(
    *,
    worker_id: int,
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
    builder = getattr(THREAD_LOCAL, "builder", None)
    builder_key = getattr(THREAD_LOCAL, "builder_key", None)
    current_key = (worker_id, vcf_path, fasta_path, outdir, embedder_type, model_name_or_path, device, pooling, local_files_only, k)
    if builder is not None and builder_key == current_key:
        return builder

    worker_outdir = str(Path(outdir) / f"thread_{worker_id:02d}")
    builder = build_minimal_builder(
        vcf_path=vcf_path,
        fasta_path=fasta_path,
        outdir=worker_outdir,
        embedder_type=embedder_type,
        model_name_or_path=model_name_or_path,
        device=device,
        pooling=pooling,
        local_files_only=local_files_only,
        k=k,
    )
    THREAD_LOCAL.builder = builder
    THREAD_LOCAL.builder_key = current_key
    return builder


def compute_variant_effects_chunk(
    *,
    worker_id: int,
    chunk_df: pd.DataFrame,
    vcf_path: str,
    fasta_path: str,
    outdir: str,
    embedder_type: str,
    model_name_or_path: str,
    device: str,
    pooling: str,
    local_files_only: bool,
    k: int,
    logger,
) -> pd.DataFrame:
    builder = get_thread_builder(
        worker_id=worker_id,
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
    logger.info("Thread-%02d started: rows=%d", worker_id, len(chunk_df))
    result_df = builder._compute_variant_effects(chunk_df)
    logger.info("Thread-%02d completed: rows_in=%d rows_out=%d", worker_id, len(chunk_df), len(result_df))
    return result_df


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
    num_threads: int = 1,
) -> pd.DataFrame:
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    logger = get_logger(outdir_path / "compute_all_vcf_variant_effects.log")
    site_df = build_all_vcf_site_df(vcf_path)
    logger.info("Loaded all VCF sites: rows=%d", len(site_df))

    if int(num_threads) <= 1:
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
        logger.info("Running in single-thread mode")
        variant_effect_df = builder._compute_variant_effects(site_df)
    else:
        chunk_dfs = split_dataframe(site_df, int(num_threads))
        logger.info("Running in multi-thread mode: threads=%d chunks=%d", int(num_threads), len(chunk_dfs))
        completed_rows = 0
        result_frames: list[pd.DataFrame | None] = [None] * len(chunk_dfs)
        with ThreadPoolExecutor(max_workers=int(num_threads), thread_name_prefix="var_effect") as executor:
            future_to_idx = {
                executor.submit(
                    compute_variant_effects_chunk,
                    worker_id=idx + 1,
                    chunk_df=chunk_df,
                    vcf_path=vcf_path,
                    fasta_path=fasta_path,
                    outdir=outdir,
                    embedder_type=embedder_type,
                    model_name_or_path=model_name_or_path,
                    device=device,
                    pooling=pooling,
                    local_files_only=local_files_only,
                    k=k,
                    logger=logger,
                ): idx
                for idx, chunk_df in enumerate(chunk_dfs)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                chunk_result = future.result()
                result_frames[idx] = chunk_result
                completed_rows += len(chunk_dfs[idx])
                logger.info(
                    "Main progress: completed_chunks=%d/%d completed_rows=%d/%d",
                    sum(frame is not None for frame in result_frames),
                    len(chunk_dfs),
                    completed_rows,
                    len(site_df),
                )
        variant_effect_df = pd.concat([frame for frame in result_frames if frame is not None], ignore_index=True)

    out_csv = outdir_path / "all_vcf_site_variant_effect_by_site.csv"
    out_parquet = outdir_path / "all_vcf_site_variant_effect_by_site.parquet"
    variant_effect_df.to_csv(out_csv, index=False)
    variant_effect_df.to_parquet(out_parquet, index=False)
    logger.info("Saved outputs: csv=%s parquet=%s rows=%d", out_csv, out_parquet, len(variant_effect_df))
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
    parser.add_argument("--num-threads", type=int, default=1, help="Number of worker threads for variant-effect computation")
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
        num_threads=int(args.num_threads),
    )
    print(
        f"Computed variant effects for {len(variant_effect_df)} rows. "
        f"Outputs saved under: {Path(args.outdir).resolve()}"
    )


if __name__ == "__main__":
    main()
