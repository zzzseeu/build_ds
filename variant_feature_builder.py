"""Build downstream datasets from a genotype matrix exported by GWAS/QTL extraction.

This module expects a ``geno_df`` CSV produced by
``GWASQTLGenotypeExtractor``. The metadata columns must include genomic
location, gene interval, trait labels, and ``REF/ALT`` alleles; the
remaining columns must be standardized sample genotype columns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from logging import init_logger
from utils import standard_chrom, standard_sample_name


@dataclass
class VariantRecord:
    """Container for one unique VCF variant."""

    chromosome: str
    position: int
    ref: str
    alt: str


class GWASQTLGenotypeExtractor:
    """Build a sample genotype matrix from GWAS/QTL selected variant sites.

    The input site table is expected to have columns:
    ``Chromosome, Position, Gene, gene_start, gene_end, gwas_trait, qtl_trait``.
    Genotypes are encoded as:
    - ``0``: reference / no variant
    - ``1``: heterozygous variant
    - ``2``: homozygous variant
    """

    SITE_COLUMNS = ["Chromosome", "Position", "Gene", "gene_start", "gene_end", "gwas_trait", "qtl_trait"]

    def __init__(
        self,
        vcf_path: str,
        site_df_path: str | None = None,
        site_df: pd.DataFrame | None = None,
        outprefix: str | None = None,
    ) -> None:
        if site_df_path is None and site_df is None:
            raise ValueError("site_df_path or site_df must be provided")
        self.site_df_path = Path(site_df_path) if site_df_path else None
        self.site_df = site_df.copy() if site_df is not None else None
        self.vcf_path = Path(vcf_path)
        self.outprefix = Path(outprefix) if outprefix else None

        site_parent = self.site_df_path.parent if self.site_df_path else Path.cwd()
        log_dir = self.outprefix.parent if self.outprefix else site_parent
        log_dir.mkdir(parents=True, exist_ok=True)
        site_stem = self.site_df_path.stem if self.site_df_path else "site_df"
        log_stem = self.outprefix.name if self.outprefix else site_stem
        log_path = log_dir / f"{log_stem}_genotype_{Path.cwd().name}.log"
        self.logger = init_logger("GWASQTLGenotypeExtractor", log_file=log_path)

    @staticmethod
    def _encode_gt(gt: Tuple[int | None, ...] | None) -> int:
        """Encode VCF GT tuple into 0/1/2."""
        if gt is None or any(x is None for x in gt):
            return 0
        alt_count = sum(1 for x in gt if x and x > 0)
        if alt_count <= 0:
            return 0
        if alt_count == 1:
            return 1
        return 2

    @staticmethod
    def _standardize_sample_names(raw_samples: List[str]) -> Dict[str, str]:
        """Standardize VCF sample names and keep them unique."""
        sample_map: Dict[str, str] = {}
        used: set[str] = set()
        for idx, sample in enumerate(raw_samples):
            std = standard_sample_name(sample) or f"sample_{idx + 1}"
            if std in used:
                std = f"{std}_{idx + 1}"
            used.add(std)
            sample_map[sample] = std
        return sample_map

    def _load_site_df(self) -> pd.DataFrame:
        """Load and normalize site metadata dataframe."""
        if self.site_df is not None:
            df = self.site_df.copy()
        else:
            assert self.site_df_path is not None
            df = pd.read_csv(self.site_df_path)
        missing = [c for c in self.SITE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"site_df missing columns: {missing}")

        df = df.copy()
        df["Chromosome"] = df["Chromosome"].astype(str).map(standard_chrom)
        df = df[df["Chromosome"].notna()]
        df["Position"] = df["Position"].astype(int)
        df["Gene"] = df["Gene"].astype(str)
        df["gene_start"] = pd.to_numeric(df["gene_start"], errors="coerce").astype(int)
        df["gene_end"] = pd.to_numeric(df["gene_end"], errors="coerce").astype(int)
        df["gwas_trait"] = df["gwas_trait"].fillna("").astype(str)
        df["qtl_trait"] = df["qtl_trait"].fillna("").astype(str)
        df = df.drop_duplicates().reset_index(drop=True)
        self.logger.info(
            "Loaded site_df: shape=%s unique_sites=%d",
            df.shape,
            df[["Chromosome", "Position"]].drop_duplicates().shape[0],
        )
        return df

    def run(self) -> pd.DataFrame:
        """Extract all-sample genotypes from VCF for the given site table."""
        try:
            import pysam  # type: ignore
        except Exception as exc:
            raise ImportError("pysam is required to read VCF") from exc

        site_df = self._load_site_df()
        target_sites = {
            (str(chrom), int(pos))
            for chrom, pos in site_df[["Chromosome", "Position"]].drop_duplicates().itertuples(index=False, name=None)
        }

        vcf = pysam.VariantFile(str(self.vcf_path))
        raw_samples = list(vcf.header.samples)
        sample_map = self._standardize_sample_names(raw_samples)
        samples = [sample_map[sample] for sample in raw_samples]
        self.logger.info("VCF loaded: samples=%d target_sites=%d", len(samples), len(target_sites))

        genotype_map: Dict[Tuple[str, int], Dict[str, str | int]] = {}
        seen = 0
        matched = 0
        for rec in vcf:
            seen += 1
            chrom = standard_chrom(str(rec.chrom))
            if chrom is None:
                continue
            pos = int(rec.pos)
            key = (chrom, pos)
            if key not in target_sites:
                continue

            row_map: Dict[str, str | int] = {
                "REF": str(rec.ref),
                "ALT": str(rec.alts[0]) if rec.alts else "",
            }
            row_map.update({
                sample_map[sample]: self._encode_gt(rec.samples[sample].get("GT"))
                for sample in raw_samples
            })
            genotype_map[key] = row_map
            matched += 1
            if matched % 1000 == 0 or matched == len(target_sites):
                self.logger.info(
                    "VCF genotype extraction progress: matched=%d/%d seen=%d",
                    matched,
                    len(target_sites),
                    seen,
                )

        rows: List[Dict[str, str | int]] = []
        for row in site_df.itertuples(index=False):
            key = (str(row.Chromosome), int(row.Position))
            default_row: Dict[str, str | int] = {"REF": "", "ALT": ""}
            default_row.update({sample: 0 for sample in samples})
            sample_genotypes = genotype_map.get(key, default_row)
            out_row: Dict[str, str | int] = {
                "Chromosome": str(row.Chromosome),
                "Position": int(row.Position),
                "Gene": str(row.Gene),
                "gene_start": int(row.gene_start),
                "gene_end": int(row.gene_end),
                "gwas_trait": str(row.gwas_trait),
                "qtl_trait": str(row.qtl_trait),
                "REF": str(sample_genotypes.get("REF", "")),
                "ALT": str(sample_genotypes.get("ALT", "")),
            }
            out_row.update({sample: int(sample_genotypes.get(sample, 0)) for sample in samples})
            rows.append(out_row)

        out_df = pd.DataFrame(rows, columns=self.SITE_COLUMNS + ["REF", "ALT"] + samples)
        self.logger.info("Built genotype matrix: shape=%s", out_df.shape)

        if self.outprefix is not None:
            out_csv = self.outprefix.with_suffix(".csv")
            out_df.to_csv(out_csv, index=False)
            self.logger.info("Genotype matrix saved: %s", out_csv)

        return out_df


class VariantFeatureBuilder:
    """Build 012 and gene-sequence datasets.

    Parameters
    ----------
    geno_df_path : str
        Path to the genotype matrix CSV from ``GWASQTLGenotypeExtractor``.
    fasta_path : str
        Reference FASTA path.
    outdir : str
        Output directory for dataframes, dictionaries, embeddings, and PCA models.
    embedder : object | None
        Callable embedder. It should accept a single DNA sequence string and
        return a 1D vector or a 2D array with batch dimension 1.
    use_pca : bool
        Whether to fit PCA per site / per gene embedding block.
    pca_var_threshold : float
        Target cumulative explained variance threshold when selecting PCA
        component count.
    """

    META_COLUMNS = ["Chromosome", "Position", "Gene", "gene_start", "gene_end", "gwas_trait", "qtl_trait", "REF", "ALT"]

    def __init__(
        self,
        geno_df_path: str,
        fasta_path: str,
        outdir: str,
        embedder=None,
        embedder_type: str | None = None,
        model_name_or_path: str | None = None,
        device: str = "cpu",
        pooling: str = "mean",
        local_files_only: bool = True,
        embedder_kwargs: dict | None = None,
        use_pca: bool = True,
        pca_var_threshold: float = 0.95,
    ) -> None:
        self.geno_df_path = Path(geno_df_path)
        self.fasta_path = Path(fasta_path)
        self.outdir = Path(outdir)
        self.embedder = embedder
        self.embedder_type = embedder_type
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.pooling = pooling
        self.local_files_only = bool(local_files_only)
        self.embedder_kwargs = dict(embedder_kwargs or {})
        self.use_pca = bool(use_pca)
        self.pca_var_threshold = float(pca_var_threshold)

        self.outdir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.outdir / "embedding_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dict_dir = self.outdir / "dicts"
        self.dict_dir.mkdir(parents=True, exist_ok=True)
        self.pca_dir = self.outdir / "pca_models"
        self.pca_dir.mkdir(parents=True, exist_ok=True)

        log_path = self.outdir / "variant_feature_builder.log"
        self.logger = init_logger("VariantFeatureBuilder", log_file=log_path)

        self.geno_df: pd.DataFrame | None = None
        self.sample_columns: List[str] = []
        self.variant_map: Dict[Tuple[str, int], VariantRecord] = {}
        self.embed_log_every = 500
        self._embed_total = 0
        self._embed_cache_hits = 0
        self._embed_cache_misses = 0

        if self.embedder is None and self.embedder_type is not None:
            from embedding import UnifiedEmbedder

            self.embedder = UnifiedEmbedder(
                embedder_type=self.embedder_type,
                model_name_or_path=self.model_name_or_path,
                device=self.device,
                pooling=self.pooling,
                local_files_only=self.local_files_only,
                **self.embedder_kwargs,
            )

    def _load_geno_df(self) -> pd.DataFrame:
        """Load and normalize the input ``geno_df`` CSV."""
        df = pd.read_csv(self.geno_df_path)
        missing = [col for col in self.META_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"geno_df missing columns: {missing}")

        df = df.copy()
        df["Chromosome"] = df["Chromosome"].astype(str).map(standard_chrom)
        df = df[df["Chromosome"].notna()]
        df["Position"] = df["Position"].astype(int)
        df["Gene"] = df["Gene"].astype(str)
        df["gene_start"] = pd.to_numeric(df["gene_start"], errors="coerce").astype(int)
        df["gene_end"] = pd.to_numeric(df["gene_end"], errors="coerce").astype(int)
        df["gwas_trait"] = df["gwas_trait"].fillna("").astype(str)
        df["qtl_trait"] = df["qtl_trait"].fillna("").astype(str)
        df["REF"] = df["REF"].fillna("").astype(str)
        df["ALT"] = df["ALT"].fillna("").astype(str)

        sample_cols = [col for col in df.columns if col not in self.META_COLUMNS]
        renamed = {}
        seen: set[str] = set()
        for idx, col in enumerate(sample_cols):
            std = standard_sample_name(col) or f"sample_{idx + 1}"
            while std in seen:
                std = f"{std}_{idx + 1}"
            renamed[col] = std
            seen.add(std)
        df = df.rename(columns=renamed)
        self.sample_columns = [renamed[col] for col in sample_cols]

        for col in self.sample_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).clip(0, 2)

        df = df.drop_duplicates().reset_index(drop=True)
        self.logger.info(
            "Loaded geno_df: shape=%s samples=%d unique_sites=%d genes=%d",
            df.shape,
            len(self.sample_columns),
            df[["Chromosome", "Position"]].drop_duplicates().shape[0],
            df["Gene"].nunique(),
        )
        return df

    def _load_variant_map(self) -> Dict[Tuple[str, int], VariantRecord]:
        """Load reference and alternative alleles from geno_df metadata."""
        assert self.geno_df is not None
        variant_map: Dict[Tuple[str, int], VariantRecord] = {}
        uniq_df = (
            self.geno_df[["Chromosome", "Position", "REF", "ALT"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        for row in uniq_df.itertuples(index=False):
            chrom = str(row.Chromosome)
            pos = int(row.Position)
            key = (chrom, pos)
            variant_map[key] = VariantRecord(
                chromosome=chrom,
                position=pos,
                ref=str(row.REF),
                alt=str(row.ALT),
            )

        self.logger.info("Loaded variant_map from geno_df: sites=%d", len(variant_map))
        return variant_map

    def _feature_name(self, chrom: str, pos: int) -> str:
        return f"{chrom}:{int(pos)}"

    @staticmethod
    def _chrom_sort_key(chrom: str) -> tuple[int, str]:
        text = str(chrom)
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            return (int(digits), text)
        return (10**9, text)

    def _ensure_embedder(self) -> None:
        if self.embedder is None:
            raise ValueError("embedder must be provided for sequence-based datasets")

    @staticmethod
    def _to_numpy_1d(x) -> np.ndarray:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().float().numpy()
        except Exception:
            pass
        arr = np.asarray(x)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D embedding vector, got shape={arr.shape}")
        return arr.astype(np.float32)

    def _embed_sequence(self, seq: str) -> np.ndarray:
        """Embed one DNA sequence with ``.npy`` caching.

        Identical sequences share the same SHA1-based cache file, so repeated
        sequences are not embedded more than once.
        """
        self._ensure_embedder()
        self._embed_total += 1
        seq_hash = hashlib.sha1(seq.encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"{seq_hash}.npy"
        if cache_path.exists():
            self._embed_cache_hits += 1
            vec = np.load(cache_path)
        else:
            self._embed_cache_misses += 1
            vec = self._to_numpy_1d(self.embedder(seq))
            np.save(cache_path, vec)

        if self._embed_total % self.embed_log_every == 0:
            self.logger.info(
                "Embedding progress: total=%d cache_hits=%d cache_misses=%d",
                self._embed_total,
                self._embed_cache_hits,
                self._embed_cache_misses,
            )

        return vec

    @staticmethod
    def _save_json(data: dict, path: Path) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _save_npz_dict(data: Dict[str, np.ndarray], path: Path) -> None:
        np.savez_compressed(path, **data)

    def _fit_pca_block(self, block: np.ndarray, block_name: str, subdir: str) -> tuple[np.ndarray, List[str]]:
        """Optionally fit PCA for one site/gene embedding block."""
        if not self.use_pca:
            cols = [f"{block_name}_embed_{i}" for i in range(block.shape[1])]
            return block.astype(np.float32), cols

        from sklearn.decomposition import PCA
        import joblib

        if block.shape[0] <= 1:
            reduced = block[:, :1].astype(np.float32)
            cols = [f"{block_name}_PC1"]
            return reduced, cols

        max_comp = min(block.shape[0], block.shape[1])
        pca_full = PCA(n_components=max_comp, svd_solver="full")
        pca_full.fit(block)
        cum = np.cumsum(pca_full.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cum, self.pca_var_threshold) + 1)
        n_comp = max(1, min(n_comp, max_comp))

        pca = PCA(n_components=n_comp, svd_solver="full")
        reduced = pca.fit_transform(block).astype(np.float32)
        model_dir = self.pca_dir / subdir
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pca, model_dir / f"{block_name}.joblib")
        cols = [f"{block_name}_PC{i + 1}" for i in range(reduced.shape[1])]
        return reduced, cols

    def build_geno012_df(self) -> pd.DataFrame:
        """Build a sample-by-site 012 matrix from ``geno_df``."""
        assert self.geno_df is not None
        site_df = (
            self.geno_df.assign(
                _chrom_order=self.geno_df["Chromosome"].map(self._chrom_sort_key)
            )
            .sort_values(["_chrom_order", "Position"])
            .drop_duplicates(subset=["Chromosome", "Position"])
            .drop(columns=["_chrom_order"])
            .reset_index(drop=True)
        )
        feature_names = [
            self._feature_name(chrom, pos)
            for chrom, pos in site_df[["Chromosome", "Position"]].itertuples(index=False, name=None)
        ]
        data = site_df[self.sample_columns].to_numpy(dtype=np.int8).T
        out_df = pd.DataFrame(data, columns=feature_names)
        out_df.insert(0, "sample", self.sample_columns)
        out_df.to_csv(self.outdir / "geno012_df.csv", index=False)
        self.logger.info("Built geno012_df: shape=%s", out_df.shape)
        return out_df

    def _get_fasta(self):
        try:
            import pysam  # type: ignore
        except Exception as exc:
            raise ImportError("pysam is required to read FASTA") from exc
        return pysam.FastaFile(str(self.fasta_path))

    def _build_fasta_chrom_map(self, fasta) -> Dict[str, str]:
        """Map standardized chromosome names to original FASTA references."""
        out: Dict[str, str] = {}
        for ref in fasta.references:
            std = standard_chrom(ref)
            if std is not None and std not in out:
                out[std] = ref
        return out

    @staticmethod
    def _resolve_fasta_chrom(chrom: str, chrom_map: Dict[str, str]) -> str:
        if chrom not in chrom_map:
            raise KeyError(f"Chromosome {chrom} not found in FASTA references")
        return chrom_map[chrom]

    def _apply_variants_to_region(
        self,
        reference_seq: str,
        region_start: int,
        variant_rows: Iterable[dict[str, str | int]],
        sample: str,
    ) -> str:
        """Apply sample-specific variants to one reference region."""
        seq = reference_seq
        offset = 0
        rows = sorted(list(variant_rows), key=lambda x: int(x["Position"]))
        for row in rows:
            if int(row[sample]) <= 0:
                continue
            key = (str(row["Chromosome"]), int(row["Position"]))
            if key not in self.variant_map:
                continue
            variant = self.variant_map[key]
            rel_start = variant.position - region_start + offset
            rel_end = rel_start + len(variant.ref)
            if rel_start < 0 or rel_end > len(seq):
                continue
            ref_in_seq = seq[rel_start:rel_end]
            if variant.ref and ref_in_seq and ref_in_seq.upper() != variant.ref.upper():
                self.logger.warning(
                    "Reference mismatch for %s:%d expected=%s observed=%s",
                    variant.chromosome,
                    variant.position,
                    variant.ref,
                    ref_in_seq,
                )
            seq = f"{seq[:rel_start]}{variant.alt}{seq[rel_end:]}"
            offset += len(variant.alt) - len(variant.ref)
        return seq

    def _build_ref_gene_seq_dict(self) -> dict[str, str]:
        """Build reference gene sequences from FASTA using gene_start/gene_end."""
        assert self.geno_df is not None
        fa = self._get_fasta()
        fasta_chrom_map = self._build_fasta_chrom_map(fa)
        ref_gene_seq_dict: dict[str, str] = {}

        for gene, gene_df in self.geno_df.groupby("Gene", sort=True):
            gene_df = (
                gene_df.assign(_chrom_order=gene_df["Chromosome"].map(self._chrom_sort_key))
                .sort_values(["_chrom_order", "Position"])
                .drop(columns=["_chrom_order"])
                .reset_index(drop=True)
            )
            chrom = str(gene_df["Chromosome"].iloc[0])
            region_start = int(gene_df["gene_start"].iloc[0])
            region_end = int(gene_df["gene_end"].iloc[0])
            fasta_chrom = self._resolve_fasta_chrom(chrom, fasta_chrom_map)
            ref_gene_seq_dict[str(gene)] = fa.fetch(fasta_chrom, region_start - 1, region_end)

        self._save_json(ref_gene_seq_dict, self.dict_dir / "ref_gene_seq_dict.json")
        self.logger.info("Built ref_gene_seq_dict: genes=%d", len(ref_gene_seq_dict))
        return ref_gene_seq_dict

    def _build_alt_gene_seq_dict(self, ref_gene_seq_dict: dict[str, str]) -> dict[str, dict[str, str]]:
        """Build sample-specific mutated gene sequences from ref sequences."""
        assert self.geno_df is not None
        alt_gene_seq_dict: dict[str, dict[str, str]] = {sample: {} for sample in self.sample_columns}

        for gene, gene_df in self.geno_df.groupby("Gene", sort=True):
            gene_df = (
                gene_df.assign(_chrom_order=gene_df["Chromosome"].map(self._chrom_sort_key))
                .sort_values(["_chrom_order", "Position"])
                .drop(columns=["_chrom_order"])
                .reset_index(drop=True)
            )
            if str(gene) not in ref_gene_seq_dict:
                continue
            region_start = int(gene_df["gene_start"].iloc[0])
            variant_rows = [row._asdict() for row in gene_df.itertuples(index=False)]
            ref_seq = ref_gene_seq_dict[str(gene)]

            for sample in self.sample_columns:
                alt_gene_seq_dict[sample][str(gene)] = self._apply_variants_to_region(
                    reference_seq=ref_seq,
                    region_start=region_start,
                    variant_rows=variant_rows,
                    sample=sample,
                )

        self._save_json(alt_gene_seq_dict, self.dict_dir / "alt_gene_seq_dict.json")
        self.logger.info("Built alt_gene_seq_dict: samples=%d genes=%d", len(alt_gene_seq_dict), len(ref_gene_seq_dict))
        return alt_gene_seq_dict

    def build_gene_seq_df(self) -> pd.DataFrame:
        """Build sample-by-gene mutated sequence embedding features."""
        ref_gene_seq_dict = self._build_ref_gene_seq_dict()
        alt_gene_seq_dict = self._build_alt_gene_seq_dict(ref_gene_seq_dict)
        genes = sorted(ref_gene_seq_dict)
        self.logger.info("Embedding gene sequences: genes=%d samples=%d", len(genes), len(self.sample_columns))
        embed_dict: Dict[str, Dict[str, List[float]]] = {sample: {} for sample in self.sample_columns}
        pca_dict: Dict[str, Dict[str, float]] = {sample: {} for sample in self.sample_columns}
        blocks = []
        columns = ["sample"]

        for gene in genes:
            seqs = [alt_gene_seq_dict[sample].get(gene, "") for sample in self.sample_columns]
            embed_block = np.vstack([self._embed_sequence(seq) if seq else np.zeros(1, dtype=np.float32) for seq in seqs])
            # Ensure consistent dimensionality if empty sequences appear.
            if embed_block.ndim == 1:
                embed_block = embed_block.reshape(-1, 1)
            for sample, seq, vec in zip(self.sample_columns, seqs, embed_block):
                embed_dict[sample][gene] = vec.astype(np.float32).tolist()
            reduced, cols = self._fit_pca_block(embed_block, gene, subdir="gene_seq")
            renamed_cols = [col.replace("_", "-", 1) for col in cols]
            for sample_idx, sample in enumerate(self.sample_columns):
                for col_idx, col in enumerate(renamed_cols):
                    pca_dict[sample][col] = float(reduced[sample_idx, col_idx])
            blocks.append(reduced)
            columns.extend(renamed_cols)

        self._save_json(embed_dict, self.dict_dir / "alt_gene_seq_embedding_dict.json")
        self._save_json(pca_dict, self.dict_dir / "alt_gene_seq_embedding_pca_dict.json")
        self.logger.info(
            "Finished gene-sequence embedding: total=%d cache_hits=%d cache_misses=%d",
            self._embed_total,
            self._embed_cache_hits,
            self._embed_cache_misses,
        )
        matrix = np.concatenate(blocks, axis=1) if blocks else np.empty((len(self.sample_columns), 0), dtype=np.float32)
        out_df = pd.DataFrame(matrix, columns=columns[1:])
        out_df.insert(0, "sample", self.sample_columns)
        out_df.to_csv(self.outdir / "gene_seq_df.csv", index=False)
        self.logger.info("Built gene_seq_df: shape=%s", out_df.shape)
        return out_df

    def run(self) -> Dict[str, pd.DataFrame]:
        """Build all configured datasets and return them as dataframes."""
        self.geno_df = self._load_geno_df()
        self.variant_map = self._load_variant_map()
        outputs = {
            "geno012_df": self.build_geno012_df(),
            "gene_seq_df": self.build_gene_seq_df(),
        }
        self.logger.info("VariantFeatureBuilder finished")
        return outputs


def _parse_embedder_kwargs(raw: str | None) -> dict:
    if raw is None:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("--embedder_kwargs must be a valid JSON object string") from exc
    if not isinstance(value, dict):
        raise ValueError("--embedder_kwargs must decode to a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build genotype matrices and downstream variant feature datasets")
    subparsers = p.add_subparsers(dest="command", required=True)

    p_genotype = subparsers.add_parser(
        "genotype",
        help="Extract sample genotype matrix (geno_df) from a selected site table and VCF",
    )
    p_genotype.add_argument("--vcf_path", required=True)
    p_genotype.add_argument("--site_df_path", required=True)
    p_genotype.add_argument("--outprefix", default=None)

    p_build = subparsers.add_parser(
        "build",
        help="Build geno012_df and gene_seq_df from a geno_df CSV",
    )
    p_build.add_argument("--geno_df_path", required=True)
    p_build.add_argument("--fasta_path", required=True)
    p_build.add_argument("--outdir", required=True)
    p_build.add_argument("--embedder_type", required=True, choices=["generator", "evo2", "nt", "agront", "rice8k"])
    p_build.add_argument("--model_name_or_path", required=True)
    p_build.add_argument("--device", default="cpu")
    p_build.add_argument("--pooling", default="mean")
    p_build.add_argument("--local_files_only", action="store_true", default=True)
    p_build.add_argument("--no-local_files_only", dest="local_files_only", action="store_false")
    p_build.add_argument("--embedder_kwargs", default=None, help='JSON object string, e.g. \'{"torch_dtype":"bfloat16"}\'')
    p_build.add_argument("--use_pca", action="store_true", default=True)
    p_build.add_argument("--no-use_pca", dest="use_pca", action="store_false")
    p_build.add_argument("--pca_var_threshold", type=float, default=0.95)
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "genotype":
        out_df = GWASQTLGenotypeExtractor(
            vcf_path=args.vcf_path,
            site_df_path=args.site_df_path,
            outprefix=args.outprefix,
        ).run()
        print(out_df.shape)
        return

    if args.command == "build":
        builder = VariantFeatureBuilder(
            geno_df_path=args.geno_df_path,
            fasta_path=args.fasta_path,
            outdir=args.outdir,
            embedder_type=args.embedder_type,
            model_name_or_path=args.model_name_or_path,
            device=args.device,
            pooling=args.pooling,
            local_files_only=args.local_files_only,
            embedder_kwargs=_parse_embedder_kwargs(args.embedder_kwargs),
            use_pca=args.use_pca,
            pca_var_threshold=args.pca_var_threshold,
        )
        outputs = builder.run()
        print({name: df.shape for name, df in outputs.items()})
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
