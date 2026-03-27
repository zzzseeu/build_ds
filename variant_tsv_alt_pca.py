"""Build REF/ALT sequence windows for TSV-selected VCF sites and run PCA on ALT embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from embedding import UnifiedEmbedder
from utils import (
    build_fasta_chrom_map,
    classify_variant_type,
    fetch_window_with_padding,
    initLogger,
    standard_chrom,
    to_numpy_1d_embedding,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


TSV_COLUMN_ALIASES = {
    "chrom": "Chromosome",
    "chr": "Chromosome",
    "chromosome": "Chromosome",
    "pos": "Position",
    "position": "Position",
    "ref": "REF",
    "alt": "ALT",
}


def progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def infer_site_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in TSV_COLUMN_ALIASES and TSV_COLUMN_ALIASES[key] not in df.columns:
            rename_map[col] = TSV_COLUMN_ALIASES[key]
    out = df.rename(columns=rename_map).copy()
    required = {"Chromosome", "Position"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"TSV missing required columns: {missing}")
    out["Chromosome"] = out["Chromosome"].map(standard_chrom)
    out["Position"] = pd.to_numeric(out["Position"], errors="coerce").astype("Int64")
    out = out[out["Chromosome"].notna() & out["Position"].notna()].copy()
    out["Position"] = out["Position"].astype(int)
    if "REF" in out.columns:
        out["REF"] = out["REF"].astype(str).str.upper()
    if "ALT" in out.columns:
        out["ALT"] = out["ALT"].astype(str).str.upper()
    return out


def make_site_key(chrom: str, pos: int, ref: str | None = None, alt: str | None = None) -> tuple[str, int, str | None, str | None]:
    return (str(chrom), int(pos), ref if ref else None, alt if alt else None)


def load_tsv_sites(tsv_path: Path) -> tuple[pd.DataFrame, set[tuple[str, int]], set[tuple[str, int, str | None, str | None]]]:
    site_df = infer_site_columns(pd.read_csv(tsv_path, sep="\t"))
    positional_keys = {
        (str(row.Chromosome), int(row.Position))
        for row in site_df.itertuples(index=False)
    }
    allele_keys = {
        make_site_key(
            row.Chromosome,
            row.Position,
            getattr(row, "REF", None),
            getattr(row, "ALT", None),
        )
        for row in site_df.itertuples(index=False)
    }
    return site_df, positional_keys, allele_keys


def collect_variants_from_vcf(
    vcf_path: Path,
    site_df: pd.DataFrame,
    positional_keys: set[tuple[str, int]],
    allele_keys: set[tuple[str, int, str | None, str | None]],
) -> pd.DataFrame:
    try:
        import pysam  # type: ignore
    except Exception as exc:
        raise ImportError("pysam is required for VCF/FASTA processing") from exc

    rows: list[dict[str, object]] = []
    has_allele_filter = "REF" in site_df.columns and "ALT" in site_df.columns

    with pysam.VariantFile(str(vcf_path)) as vcf:
        for rec in progress(vcf, desc="Reading VCF", unit="record"):
            chrom = standard_chrom(rec.chrom)
            if chrom is None:
                continue
            pos_key = (chrom, int(rec.pos))
            if pos_key not in positional_keys:
                continue
            ref = str(rec.ref).upper()
            for alt in rec.alts or ():
                alt = str(alt).upper()
                if has_allele_filter and make_site_key(chrom, rec.pos, ref, alt) not in allele_keys:
                    continue
                rows.append(
                    {
                        "Chromosome": chrom,
                        "Position": int(rec.pos),
                        "REF": ref,
                        "ALT": alt,
                        "mutation_type": classify_variant_type(ref, alt),
                    }
                )

    variant_df = pd.DataFrame(rows)
    if variant_df.empty:
        return variant_df
    return variant_df.drop_duplicates(subset=["Chromosome", "Position", "REF", "ALT"]).reset_index(drop=True)


class SequenceEmbedderCache:
    """Embed sequences with on-disk caching."""

    def __init__(
        self,
        outdir: Path,
        embedder_type: str,
        model_name_or_path: str,
        device: str,
        pooling: str,
        local_files_only: bool,
        embedder_kwargs: dict,
    ):
        self.cache_dir = outdir / "embedding_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = UnifiedEmbedder(
            embedder_type=embedder_type,
            model_name_or_path=model_name_or_path,
            device=device,
            pooling=pooling,
            local_files_only=local_files_only,
            **embedder_kwargs,
        )

    def embed_sequence(self, seq: str) -> np.ndarray:
        seq_hash = hashlib.sha1(seq.encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"{seq_hash}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        vec = to_numpy_1d_embedding(self.embedder(seq))
        np.save(cache_path, vec)
        return vec


def add_sequence_windows(variant_df: pd.DataFrame, fasta_path: Path, k: int) -> pd.DataFrame:
    try:
        import pysam  # type: ignore
    except Exception as exc:
        raise ImportError("pysam is required for VCF/FASTA processing") from exc

    rows: list[dict[str, object]] = []
    with pysam.FastaFile(str(fasta_path)) as fasta:
        chrom_map = build_fasta_chrom_map(fasta)
        for row in progress(
            variant_df.itertuples(index=False),
            total=len(variant_df),
            desc="Building REF/ALT windows",
            unit="site",
        ):
            chrom = str(row.Chromosome)
            fasta_chrom = chrom_map.get(chrom)
            if fasta_chrom is None:
                continue
            pos = int(row.Position)
            ref = str(row.REF).upper()
            alt = str(row.ALT).upper()
            upstream = fetch_window_with_padding(fasta, fasta_chrom, pos - k, pos - 1)
            downstream = fetch_window_with_padding(
                fasta,
                fasta_chrom,
                pos + len(ref),
                pos + len(ref) + k - 1,
            )
            rows.append(
                {
                    "Chromosome": chrom,
                    "Position": pos,
                    "REF": ref,
                    "ALT": alt,
                    "site_id": f"{chrom}:{pos}:{ref}>{alt}",
                    "mutation_type": classify_variant_type(ref, alt),
                    "upstream_seq": upstream,
                    "downstream_seq": downstream,
                    "ref_seq": upstream + ref + downstream,
                    "alt_seq": upstream + alt + downstream,
                }
            )
    return pd.DataFrame(rows)


def run_pca(embedding_matrix: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.decomposition import PCA  # type: ignore

        model = PCA(n_components=n_components)
        coords = model.fit_transform(embedding_matrix)
        explained = model.explained_variance_ratio_
        return coords, explained
    except Exception:
        centered = embedding_matrix - embedding_matrix.mean(axis=0, keepdims=True)
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        coords = u[:, :n_components] * s[:n_components]
        var = (s ** 2) / max(1, centered.shape[0] - 1)
        explained = var[:n_components] / var.sum()
        return coords, explained


def maybe_plot_pca(pca_df: pd.DataFrame, explained: np.ndarray, out_path: Path, logger) -> Path | None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        logger.warning("matplotlib not available, skip PCA plot")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    for mutation_type, sub_df in pca_df.groupby("mutation_type", dropna=False):
        ax.scatter(sub_df["PC1"], sub_df["PC2"], s=28, alpha=0.8, label=str(mutation_type))
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.2f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.2f}%)")
    ax.set_title("ALT embedding PCA by mutation type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract REF/ALT windows from VCF+FASTA for TSV sites and run PCA on ALT embeddings")
    parser.add_argument("--tsv-path", required=True, help="Input site TSV path; requires Chromosome/Position or aliases")
    parser.add_argument("--vcf-path", required=True, help="Input VCF path")
    parser.add_argument("--fasta-path", required=True, help="Reference FASTA path")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--k", type=int, default=100, help="Upstream/downstream window length")
    parser.add_argument("--embedder-type", required=True, help="Embedder type, e.g. rice8k")
    parser.add_argument("--model-name-or-path", required=True, help="Local model path or model name")
    parser.add_argument("--device", default="cpu", help="Embedding device, e.g. cpu or cuda")
    parser.add_argument("--pooling", default="mean", choices=["mean", "last"], help="Embedding pooling strategy")
    parser.add_argument("--local-files-only", action="store_true", help="Only load model files from local path/cache")
    parser.add_argument("--pca-components", type=int, default=2, help="Number of PCA dimensions to keep")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = initLogger(outdir / "variant_tsv_alt_pca.log")

    site_df, positional_keys, allele_keys = load_tsv_sites(Path(args.tsv_path))
    logger.info(f"Loaded TSV sites: {len(site_df)}")

    variant_df = collect_variants_from_vcf(Path(args.vcf_path), site_df, positional_keys, allele_keys)
    if variant_df.empty:
        raise ValueError("No matching VCF variants found for TSV sites")
    logger.info(f"Matched VCF variants: {len(variant_df)}")

    seq_df = add_sequence_windows(variant_df, Path(args.fasta_path), args.k)
    if seq_df.empty:
        raise ValueError("No sequence windows built; please check FASTA chromosome names")
    seq_df.to_csv(outdir / "variant_sequences.tsv", sep="\t", index=False)

    embedder = SequenceEmbedderCache(
        outdir=outdir,
        embedder_type=args.embedder_type,
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        pooling=args.pooling,
        local_files_only=bool(args.local_files_only),
        embedder_kwargs={},
    )

    ref_embeddings: list[np.ndarray] = []
    alt_embeddings: list[np.ndarray] = []
    for row in progress(
        seq_df.itertuples(index=False),
        total=len(seq_df),
        desc="Embedding sequences",
        unit="site",
    ):
        ref_embeddings.append(embedder.embed_sequence(str(row.ref_seq)))
        alt_embeddings.append(embedder.embed_sequence(str(row.alt_seq)))

    ref_matrix = np.vstack(ref_embeddings)
    alt_matrix = np.vstack(alt_embeddings)
    diff_matrix = alt_matrix - ref_matrix

    np.save(outdir / "ref_embeddings.npy", ref_matrix)
    np.save(outdir / "alt_embeddings.npy", alt_matrix)
    np.save(outdir / "alt_minus_ref_embeddings.npy", diff_matrix)

    coords, explained = run_pca(alt_matrix, n_components=max(2, int(args.pca_components)))
    coord_cols = {f"PC{i + 1}": coords[:, i] for i in range(coords.shape[1])}
    pca_df = seq_df[["site_id", "Chromosome", "Position", "REF", "ALT", "mutation_type"]].copy()
    for key, value in coord_cols.items():
        pca_df[key] = value
    pca_df.to_csv(outdir / "alt_embedding_pca.tsv", sep="\t", index=False)

    plot_path = maybe_plot_pca(pca_df, explained, outdir / "alt_embedding_pca.png", logger)
    summary = {
        "n_input_sites": int(len(site_df)),
        "n_matched_variants": int(len(seq_df)),
        "k": int(args.k),
        "embedder_type": str(args.embedder_type),
        "pca_components": int(coords.shape[1]),
        "explained_variance_ratio": [float(x) for x in explained[: coords.shape[1]]],
        "mutation_type_counts": {
            str(k): int(v) for k, v in pca_df["mutation_type"].value_counts().sort_index().items()
        },
        "outputs": {
            "variant_sequences_tsv": str(outdir / "variant_sequences.tsv"),
            "alt_embedding_pca_tsv": str(outdir / "alt_embedding_pca.tsv"),
            "ref_embeddings_npy": str(outdir / "ref_embeddings.npy"),
            "alt_embeddings_npy": str(outdir / "alt_embeddings.npy"),
            "alt_minus_ref_embeddings_npy": str(outdir / "alt_minus_ref_embeddings.npy"),
            "plot_png": str(plot_path) if plot_path is not None else None,
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
