"""Build downstream datasets from a genotype matrix exported by GWAS/QTL extraction.

This module expects a ``geno_df`` CSV produced by
``GWASQTLGenotypeExtractor``. The first five columns must be:
``Chromosome, Position, Gene, gwas_trait, qtl_trait`` and the remaining
columns must be standardized sample genotype columns.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from embedding import UnifiedEmbedder
from logging import init_logger
from utils import standard_chrom, standard_sample_name


@dataclass
class VariantRecord:
    """Container for one unique VCF variant."""

    chromosome: str
    position: int
    ref: str
    alt: str


class VariantFeatureBuilder:
    """Build 012, SNP extended-sequence, and gene-sequence datasets.

    Parameters
    ----------
    geno_df_path : str
        Path to the genotype matrix CSV from ``GWASQTLGenotypeExtractor``.
    fasta_path : str
        Reference FASTA path.
    vcf_path : str
        VCF path used to recover reference and alternative alleles.
    outdir : str
        Output directory for dataframes, dictionaries, embeddings, and PCA models.
    embedder : object | None
        Callable embedder. It should accept a single DNA sequence string and
        return a 1D vector or a 2D array with batch dimension 1.
    flank_k : int
        Upstream/downstream extension length for SNP extended-sequence features.
    use_pca : bool
        Whether to fit PCA per site / per gene embedding block.
    pca_var_threshold : float
        Target cumulative explained variance threshold when selecting PCA
        component count.
    """

    META_COLUMNS = ["Chromosome", "Position", "Gene", "gwas_trait", "qtl_trait"]

    def __init__(
        self,
        geno_df_path: str,
        fasta_path: str,
        vcf_path: str,
        outdir: str,
        embedder=None,
        embedder_type: str | None = None,
        model_name_or_path: str | None = None,
        device: str = "cpu",
        pooling: str = "mean",
        local_files_only: bool = True,
        embedder_kwargs: dict | None = None,
        flank_k: int = 50,
        use_pca: bool = True,
        pca_var_threshold: float = 0.95,
    ) -> None:
        self.geno_df_path = Path(geno_df_path)
        self.fasta_path = Path(fasta_path)
        self.vcf_path = Path(vcf_path)
        self.outdir = Path(outdir)
        self.embedder = embedder
        self.embedder_type = embedder_type
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.pooling = pooling
        self.local_files_only = bool(local_files_only)
        self.embedder_kwargs = dict(embedder_kwargs or {})
        self.flank_k = int(flank_k)
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

        if self.embedder is None and self.embedder_type is not None:
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
        df["gwas_trait"] = df["gwas_trait"].fillna("").astype(str)
        df["qtl_trait"] = df["qtl_trait"].fillna("").astype(str)

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
        """Load reference and alternative alleles from VCF for all unique sites."""
        try:
            import pysam  # type: ignore
        except Exception as exc:
            raise ImportError("pysam is required to read VCF") from exc

        assert self.geno_df is not None
        target_sites = {
            (str(chrom), int(pos))
            for chrom, pos in self.geno_df[["Chromosome", "Position"]].drop_duplicates().itertuples(index=False, name=None)
        }

        variant_map: Dict[Tuple[str, int], VariantRecord] = {}
        vcf = pysam.VariantFile(str(self.vcf_path))
        for rec in vcf:
            chrom = standard_chrom(str(rec.chrom))
            if chrom is None:
                continue
            pos = int(rec.pos)
            key = (chrom, pos)
            if key not in target_sites:
                continue
            alt = str(rec.alts[0]) if rec.alts else ""
            variant_map[key] = VariantRecord(
                chromosome=chrom,
                position=pos,
                ref=str(rec.ref),
                alt=alt,
            )

        self.logger.info("Loaded variant_map: matched=%d target_sites=%d", len(variant_map), len(target_sites))
        return variant_map

    def _feature_name(self, chrom: str, pos: int) -> str:
        return f"{chrom}:{int(pos)}"

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
        """Embed one DNA sequence with on-disk caching."""
        self._ensure_embedder()
        seq_hash = hashlib.sha1(seq.encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"{seq_hash}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        vec = self._to_numpy_1d(self.embedder(seq))
        np.save(cache_path, vec)
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
            self.geno_df.sort_values(["Chromosome", "Position"])
            .drop_duplicates(subset=["Chromosome", "Position"])
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

    def _build_extseq_dict(self) -> dict[str, dict[str, str]]:
        """Build per-site reference and alternative extended sequences."""
        assert self.geno_df is not None
        fa = self._get_fasta()
        fasta_chrom_map = self._build_fasta_chrom_map(fa)
        extseq_dict: dict[str, dict[str, str]] = {}
        uniq_sites = (
            self.geno_df[["Chromosome", "Position"]]
            .drop_duplicates()
            .sort_values(["Chromosome", "Position"])
            .itertuples(index=False, name=None)
        )

        for chrom, pos in uniq_sites:
            key = (str(chrom), int(pos))
            if key not in self.variant_map:
                continue
            variant = self.variant_map[key]
            fasta_chrom = self._resolve_fasta_chrom(variant.chromosome, fasta_chrom_map)
            left_start = max(1, variant.position - self.flank_k)
            left_seq = fa.fetch(fasta_chrom, left_start - 1, variant.position - 1)
            right_start = variant.position + len(variant.ref)
            right_end = right_start + self.flank_k - 1
            right_seq = fa.fetch(fasta_chrom, right_start - 1, right_end)
            site_key = self._feature_name(variant.chromosome, variant.position)
            extseq_dict[site_key] = {
                "ref_seq": f"{left_seq}{variant.ref}{right_seq}",
                "alt_seq": f"{left_seq}{variant.alt}{right_seq}",
            }

        self._save_json(extseq_dict, self.dict_dir / "site_extseq_dict.json")
        self.logger.info("Built site_extseq_dict: n=%d", len(extseq_dict))
        return extseq_dict

    def build_snp_extseq_df(self) -> pd.DataFrame:
        """Build sample-by-site extended-sequence embedding features."""
        assert self.geno_df is not None
        extseq_dict = self._build_extseq_dict()
        embed_dict: Dict[str, np.ndarray] = {}
        for site_key, seq_dict in extseq_dict.items():
            embed_dict[f"{site_key}|ref"] = self._embed_sequence(seq_dict["ref_seq"])
            embed_dict[f"{site_key}|alt"] = self._embed_sequence(seq_dict["alt_seq"])
        self._save_npz_dict(embed_dict, self.dict_dir / "site_extseq_embedding.npz")

        site_df = (
            self.geno_df.sort_values(["Chromosome", "Position"])
            .drop_duplicates(subset=["Chromosome", "Position"])
            .reset_index(drop=True)
        )
        blocks = []
        columns = ["sample"]
        for chrom, pos in site_df[["Chromosome", "Position"]].itertuples(index=False, name=None):
            site_key = self._feature_name(chrom, pos)
            if site_key not in extseq_dict:
                continue
            block = np.vstack(
                [
                    embed_dict[f"{site_key}|alt"] * float(gt)
                    for gt in site_df.loc[
                        (site_df["Chromosome"] == chrom) & (site_df["Position"] == pos),
                        self.sample_columns,
                    ].iloc[0].tolist()
                ]
            ).astype(np.float32)
            reduced, cols = self._fit_pca_block(block, site_key, subdir="snp_extseq")
            blocks.append(reduced)
            columns.extend(cols)

        matrix = np.concatenate(blocks, axis=1) if blocks else np.empty((len(self.sample_columns), 0), dtype=np.float32)
        out_df = pd.DataFrame(matrix, columns=columns[1:])
        out_df.insert(0, "sample", self.sample_columns)
        out_df.to_csv(self.outdir / "snp_extseq_df.csv", index=False)
        self.logger.info("Built snp_extseq_df: shape=%s", out_df.shape)
        return out_df

    def _apply_variants_to_region(
        self,
        reference_seq: str,
        region_start: int,
        variant_rows: Iterable[pd.Series],
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
            seq = f"{seq[:rel_start]}{variant.alt}{seq[rel_end:]}"
            offset += len(variant.alt) - len(variant.ref)
        return seq

    def _build_gene_seq_dict(self) -> dict[str, dict[str, str]]:
        """Build per-sample per-gene mutated sequences.

        Since only ``geno_df + VCF + FASTA`` are provided, each gene region is
        defined by the span covered by its variants on the reference genome.
        """
        assert self.geno_df is not None
        fa = self._get_fasta()
        fasta_chrom_map = self._build_fasta_chrom_map(fa)
        gene_seq_dict: dict[str, dict[str, str]] = {sample: {} for sample in self.sample_columns}

        for gene, gene_df in self.geno_df.groupby("Gene", sort=True):
            gene_df = gene_df.sort_values(["Chromosome", "Position"]).reset_index(drop=True)
            chrom = str(gene_df["Chromosome"].iloc[0])
            starts = []
            ends = []
            for row in gene_df.itertuples(index=False):
                key = (str(row.Chromosome), int(row.Position))
                if key not in self.variant_map:
                    continue
                starts.append(int(row.Position))
                ends.append(int(row.Position) + len(self.variant_map[key].ref) - 1)
            if not starts:
                continue
            region_start = min(starts)
            region_end = max(ends)
            fasta_chrom = self._resolve_fasta_chrom(chrom, fasta_chrom_map)
            ref_seq = fa.fetch(fasta_chrom, region_start - 1, region_end)
            variant_rows = [row._asdict() for row in gene_df.itertuples(index=False)]

            for sample in self.sample_columns:
                gene_seq_dict[sample][str(gene)] = self._apply_variants_to_region(
                    reference_seq=ref_seq,
                    region_start=region_start,
                    variant_rows=variant_rows,
                    sample=sample,
                )

        self._save_json(gene_seq_dict, self.dict_dir / "gene_seq_dict.json")
        self.logger.info("Built gene_seq_dict: samples=%d", len(gene_seq_dict))
        return gene_seq_dict

    def build_gene_seq_df(self) -> pd.DataFrame:
        """Build sample-by-gene mutated sequence embedding features."""
        gene_seq_dict = self._build_gene_seq_dict()
        genes = sorted({gene for sample_data in gene_seq_dict.values() for gene in sample_data})
        embed_dict: Dict[str, np.ndarray] = {}
        blocks = []
        columns = ["sample"]

        for gene in genes:
            seqs = [gene_seq_dict[sample].get(gene, "") for sample in self.sample_columns]
            embed_block = np.vstack([self._embed_sequence(seq) if seq else np.zeros(1, dtype=np.float32) for seq in seqs])
            # Ensure consistent dimensionality if empty sequences appear.
            if embed_block.ndim == 1:
                embed_block = embed_block.reshape(-1, 1)
            for sample, seq, vec in zip(self.sample_columns, seqs, embed_block):
                embed_dict[f"{sample}|{gene}"] = vec.astype(np.float32)
            reduced, cols = self._fit_pca_block(embed_block, gene, subdir="gene_seq")
            blocks.append(reduced)
            columns.extend(cols)

        self._save_npz_dict(embed_dict, self.dict_dir / "gene_seq_embedding.npz")
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
            "snp_extseq_df": self.build_snp_extseq_df(),
            "gene_seq_df": self.build_gene_seq_df(),
        }
        self.logger.info("VariantFeatureBuilder finished")
        return outputs
