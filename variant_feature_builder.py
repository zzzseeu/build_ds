"""Build genotype and gene-sequence feature datasets from GWAS/QTL variant sites.

All genomic coordinates are interpreted as 1-based.
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


@dataclass(frozen=True)
class VariantRecord:
    chromosome: str
    position: int
    gene: str
    gene_start: int
    gene_end: int
    ref: str
    alt: str
    qtl_trait: str
    gwas_trait: str


class GWASQTLGenotypeExtractor:
    """Extract sample genotype matrix from the site CSV and VCF."""

    SITE_COLUMNS = ["Chromosome", "Position", "Gene", "gene_start", "gene_end", "qtl_trait", "gwas_trait"]
    OUTPUT_COLUMNS = ["Chromosome", "Position", "Gene", "gene_start", "gene_end", "REF", "ALT", "qtl_trait", "gwas_trait"]

    def __init__(
        self,
        vcf_path: str,
        site_df_path: str | None = None,
        site_df: pd.DataFrame | None = None,
        outprefix: str | None = None,
    ) -> None:
        if site_df_path is None and site_df is None:
            raise ValueError("site_df_path or site_df must be provided")

        self.vcf_path = Path(vcf_path)
        self.site_df_path = Path(site_df_path) if site_df_path else None
        self.site_df = site_df.copy() if site_df is not None else None
        self.outprefix = Path(outprefix) if outprefix else None

        site_parent = self.site_df_path.parent if self.site_df_path else Path.cwd()
        log_dir = self.outprefix.parent if self.outprefix else site_parent
        log_dir.mkdir(parents=True, exist_ok=True)
        log_stem = self.outprefix.name if self.outprefix else (self.site_df_path.stem if self.site_df_path else "site_df")
        self.logger = init_logger(
            "GWASQTLGenotypeExtractor",
            log_file=log_dir / f"{log_stem}_genotype_{Path.cwd().name}.log",
        )

    @staticmethod
    def _encode_gt(gt: Tuple[int | None, ...] | None) -> int:
        if gt is None or any(x is None for x in gt):
            return 0
        alt_count = sum(1 for x in gt if x and x > 0)
        if alt_count <= 0:
            return 0
        if alt_count == 1:
            return 1
        return 2

    @staticmethod
    def _chrom_sort_key(chrom: str) -> tuple[int, str]:
        text = str(chrom)
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            return (int(digits), text)
        return (10**9, text)

    @staticmethod
    def _standardize_sample_names(raw_samples: List[str]) -> Dict[str, str]:
        sample_map: Dict[str, str] = {}
        used: set[str] = set()
        for idx, sample in enumerate(raw_samples):
            std = standard_sample_name(sample) or f"sample_{idx + 1}"
            if std in used:
                std = f"{std}_{idx + 1}"
            sample_map[sample] = std
            used.add(std)
        return sample_map

    def _load_site_df(self) -> pd.DataFrame:
        df = self.site_df.copy() if self.site_df is not None else pd.read_csv(self.site_df_path)
        missing = [c for c in self.SITE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"site_df missing columns: {missing}")

        df = df.copy()
        df["Chromosome"] = df["Chromosome"].astype(str).map(standard_chrom)
        df = df[df["Chromosome"].notna()]
        df["Position"] = pd.to_numeric(df["Position"], errors="coerce").astype("Int64")
        df["gene_start"] = pd.to_numeric(df["gene_start"], errors="coerce").astype("Int64")
        df["gene_end"] = pd.to_numeric(df["gene_end"], errors="coerce").astype("Int64")
        df = df[df["Position"].notna() & df["gene_start"].notna() & df["gene_end"].notna()]
        df["Position"] = df["Position"].astype(int)
        df["gene_start"] = df["gene_start"].astype(int)
        df["gene_end"] = df["gene_end"].astype(int)
        swapped = df["gene_start"] > df["gene_end"]
        if swapped.any():
            df.loc[swapped, ["gene_start", "gene_end"]] = df.loc[swapped, ["gene_end", "gene_start"]].to_numpy()
        df = df[(df["Position"] >= 1) & (df["gene_start"] >= 1) & (df["gene_end"] >= 1)]
        df = df[(df["Position"] >= df["gene_start"]) & (df["Position"] <= df["gene_end"])]
        df["Gene"] = df["Gene"].astype(str)
        df["qtl_trait"] = df["qtl_trait"].fillna("").astype(str)
        df["gwas_trait"] = df["gwas_trait"].fillna("").astype(str)
        df = (
            df.assign(_chrom_order=df["Chromosome"].map(self._chrom_sort_key))
            .sort_values(["_chrom_order", "Position", "Gene"])
            .drop(columns=["_chrom_order"])
            .drop_duplicates()
            .reset_index(drop=True)
        )
        self.logger.info("Loaded site_df: shape=%s", df.shape)
        return df

    def run(self) -> pd.DataFrame:
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
        sample_columns = [sample_map[s] for s in raw_samples]

        genotype_map: Dict[Tuple[str, int], Dict[str, str | int]] = {}
        for rec in vcf:
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

        rows: list[dict[str, str | int]] = []
        for row in site_df.itertuples(index=False):
            key = (str(row.Chromosome), int(row.Position))
            site_gt = genotype_map.get(key, {"REF": "", "ALT": "", **{sample: 0 for sample in sample_columns}})
            out_row: dict[str, str | int] = {
                "Chromosome": str(row.Chromosome),
                "Position": int(row.Position),
                "Gene": str(row.Gene),
                "gene_start": int(row.gene_start),
                "gene_end": int(row.gene_end),
                "REF": str(site_gt.get("REF", "")),
                "ALT": str(site_gt.get("ALT", "")),
                "qtl_trait": str(row.qtl_trait),
                "gwas_trait": str(row.gwas_trait),
            }
            out_row.update({sample: int(site_gt.get(sample, 0)) for sample in sample_columns})
            rows.append(out_row)

        geno_df = pd.DataFrame(rows, columns=self.OUTPUT_COLUMNS + sample_columns)
        self.logger.info("Built geno_df: shape=%s", geno_df.shape)
        if self.outprefix is not None:
            out_csv = self.outprefix.with_suffix(".csv")
            geno_df.to_csv(out_csv, index=False)
            self.logger.info("Saved geno_df: %s", out_csv)
        return geno_df


class VariantFeatureBuilder:
    """Build genotype-012, gene-sequence, and gene-feature datasets."""

    META_COLUMNS = ["Chromosome", "Position", "Gene", "gene_start", "gene_end", "REF", "ALT", "qtl_trait", "gwas_trait"]

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
        gene_feature_format: str = "csv",
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
        self.gene_feature_format = str(gene_feature_format).lower()

        if self.gene_feature_format not in {"csv", "parquet"}:
            raise ValueError("gene_feature_format must be one of: csv, parquet")

        self.outdir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.outdir / "embedding_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dict_dir = self.outdir / "dicts"
        self.dict_dir.mkdir(parents=True, exist_ok=True)
        self.pca_dir = self.outdir / "pca_models"
        self.pca_dir.mkdir(parents=True, exist_ok=True)

        self.logger = init_logger("VariantFeatureBuilder", log_file=self.outdir / "variant_feature_builder.log")
        self.geno_df: pd.DataFrame | None = None
        self.sample_columns: List[str] = []
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

    @staticmethod
    def _chrom_sort_key(chrom: str) -> tuple[int, str]:
        text = str(chrom)
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            return (int(digits), text)
        return (10**9, text)

    @staticmethod
    def _save_json(data: dict, path: Path) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _save_matrix(df: pd.DataFrame, path_without_suffix: Path, fmt: str) -> Path:
        if fmt == "csv":
            path = path_without_suffix.with_suffix(".csv")
            df.to_csv(path, index=False)
            return path
        path = path_without_suffix.with_suffix(".parquet")
        df.to_parquet(path, index=False)
        return path

    def _ensure_embedder(self) -> None:
        if self.embedder is None:
            raise ValueError("embedder or embedder_type must be provided")

    def _load_geno_df(self) -> pd.DataFrame:
        df = pd.read_csv(self.geno_df_path)
        missing = [c for c in self.META_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"geno_df missing columns: {missing}")

        df = df.copy()
        df["Chromosome"] = df["Chromosome"].astype(str).map(standard_chrom)
        df = df[df["Chromosome"].notna()]
        df["Position"] = pd.to_numeric(df["Position"], errors="coerce").astype("Int64")
        df["gene_start"] = pd.to_numeric(df["gene_start"], errors="coerce").astype("Int64")
        df["gene_end"] = pd.to_numeric(df["gene_end"], errors="coerce").astype("Int64")
        df = df[df["Position"].notna() & df["gene_start"].notna() & df["gene_end"].notna()]
        df["Position"] = df["Position"].astype(int)
        df["gene_start"] = df["gene_start"].astype(int)
        df["gene_end"] = df["gene_end"].astype(int)
        swapped = df["gene_start"] > df["gene_end"]
        if swapped.any():
            df.loc[swapped, ["gene_start", "gene_end"]] = df.loc[swapped, ["gene_end", "gene_start"]].to_numpy()
        df = df[(df["Position"] >= 1) & (df["gene_start"] >= 1) & (df["gene_end"] >= 1)]
        df = df[(df["Position"] >= df["gene_start"]) & (df["Position"] <= df["gene_end"])]
        df["Gene"] = df["Gene"].astype(str)
        df["REF"] = df["REF"].fillna("").astype(str)
        df["ALT"] = df["ALT"].fillna("").astype(str)
        df["qtl_trait"] = df["qtl_trait"].fillna("").astype(str)
        df["gwas_trait"] = df["gwas_trait"].fillna("").astype(str)

        sample_cols = [c for c in df.columns if c not in self.META_COLUMNS]
        renamed: dict[str, str] = {}
        used: set[str] = set()
        for idx, col in enumerate(sample_cols):
            std = standard_sample_name(col) or f"sample_{idx + 1}"
            while std in used:
                std = f"{std}_{idx + 1}"
            renamed[col] = std
            used.add(std)
        df = df.rename(columns=renamed)
        self.sample_columns = [renamed[col] for col in sample_cols]
        for col in self.sample_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).clip(0, 2)

        df = (
            df.assign(_chrom_order=df["Chromosome"].map(self._chrom_sort_key))
            .sort_values(["_chrom_order", "Position", "Gene"])
            .drop(columns=["_chrom_order"])
            .drop_duplicates()
            .reset_index(drop=True)
        )
        self.logger.info("Loaded geno_df: shape=%s samples=%d", df.shape, len(self.sample_columns))
        return df

    def _get_fasta(self):
        try:
            import pysam  # type: ignore
        except Exception as exc:
            raise ImportError("pysam is required to read FASTA") from exc
        return pysam.FastaFile(str(self.fasta_path))

    def _build_fasta_chrom_map(self, fasta) -> Dict[str, str]:
        chrom_map: Dict[str, str] = {}
        for ref in fasta.references:
            std = standard_chrom(ref)
            if std is not None and std not in chrom_map:
                chrom_map[std] = ref
        return chrom_map

    @staticmethod
    def _resolve_fasta_chrom(chrom: str, chrom_map: Dict[str, str]) -> str:
        if chrom not in chrom_map:
            raise KeyError(f"Chromosome {chrom} not found in FASTA")
        return chrom_map[chrom]

    def _validate_ref_against_fasta(self) -> None:
        assert self.geno_df is not None
        fasta = self._get_fasta()
        chrom_map = self._build_fasta_chrom_map(fasta)
        mismatched = 0
        checked = 0
        unique_sites = self.geno_df[["Chromosome", "Position", "REF"]].drop_duplicates()
        for row in unique_sites.itertuples(index=False):
            ref = str(row.REF)
            if not ref:
                continue
            fasta_chrom = self._resolve_fasta_chrom(str(row.Chromosome), chrom_map)
            observed = fasta.fetch(fasta_chrom, int(row.Position) - 1, int(row.Position) - 1 + len(ref))
            checked += 1
            if observed.upper() != ref.upper():
                mismatched += 1
                self.logger.warning(
                    "REF mismatch at %s:%d expected=%s observed=%s",
                    str(row.Chromosome),
                    int(row.Position),
                    ref,
                    observed,
                )
        self.logger.info("REF validation finished: checked=%d mismatched=%d", checked, mismatched)

    def build_genotype_matrix(self) -> pd.DataFrame:
        assert self.geno_df is not None
        site_df = self.geno_df.drop_duplicates(subset=["Chromosome", "Position"]).reset_index(drop=True)
        feature_names = [f"{chrom}:{pos}" for chrom, pos in site_df[["Chromosome", "Position"]].itertuples(index=False, name=None)]
        matrix = site_df[self.sample_columns].to_numpy(dtype=np.int8).T
        out_df = pd.DataFrame(matrix, columns=feature_names)
        out_df.insert(0, "sample", self.sample_columns)
        out_path = self.outdir / "genotype_012.csv"
        out_df.to_csv(out_path, index=False)
        self.logger.info("Saved genotype_012 matrix: %s shape=%s", out_path, out_df.shape)
        return out_df

    def _build_ref_gene_seq_dict(self) -> dict[str, str]:
        assert self.geno_df is not None
        fasta = self._get_fasta()
        chrom_map = self._build_fasta_chrom_map(fasta)
        ref_gene_seq_dict: dict[str, str] = {}

        gene_meta = (
            self.geno_df[["Gene", "Chromosome", "gene_start", "gene_end"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        for row in gene_meta.itertuples(index=False):
            gene = str(row.Gene)
            chrom = str(row.Chromosome)
            start = int(row.gene_start)
            end = int(row.gene_end)
            fasta_chrom = self._resolve_fasta_chrom(chrom, chrom_map)
            ref_gene_seq_dict[gene] = fasta.fetch(fasta_chrom, start - 1, end)

        self._save_json(ref_gene_seq_dict, self.dict_dir / "ref_gene_seq_dict.json")
        self.logger.info("Built ref_gene_seq_dict: genes=%d", len(ref_gene_seq_dict))
        return ref_gene_seq_dict

    def _add_gene_seq_column(self, ref_gene_seq_dict: dict[str, str]) -> pd.DataFrame:
        assert self.geno_df is not None
        df = self.geno_df.copy()
        df["gene_seq"] = df["Gene"].map(ref_gene_seq_dict)
        out_path = self.outdir / "geno_df_with_gene_seq.csv"
        df.to_csv(out_path, index=False)
        self.logger.info("Saved geno_df_with_gene_seq: %s shape=%s", out_path, df.shape)
        self.geno_df = df
        return df

    def _apply_variants_to_gene(self, reference_seq: str, gene_start: int, variant_rows: Iterable[dict[str, str | int]], sample: str) -> str:
        seq = reference_seq
        offset = 0
        rows = sorted(list(variant_rows), key=lambda x: int(x["Position"]))
        for row in rows:
            if int(row[sample]) <= 0:
                continue
            pos = int(row["Position"])
            ref = str(row["REF"])
            alt = str(row["ALT"])
            rel_start = pos - gene_start + offset
            rel_end = rel_start + len(ref)
            if rel_start < 0 or rel_end > len(seq):
                continue
            observed = seq[rel_start:rel_end]
            if ref and observed.upper() != ref.upper():
                self.logger.warning(
                    "Gene sequence replacement mismatch for %s at %s:%d expected=%s observed=%s",
                    row["Gene"],
                    row["Chromosome"],
                    pos,
                    ref,
                    observed,
                )
            seq = f"{seq[:rel_start]}{alt}{seq[rel_end:]}"
            offset += len(alt) - len(ref)
        return seq

    def build_gene_sequence_matrix(self) -> pd.DataFrame:
        assert self.geno_df is not None
        ref_gene_seq_dict = self._build_ref_gene_seq_dict()
        self._add_gene_seq_column(ref_gene_seq_dict)
        gene_seq_dict: dict[str, dict[str, str]] = {sample: {} for sample in self.sample_columns}

        for gene, gene_df in self.geno_df.groupby("Gene", sort=True):
            gene_df = gene_df.sort_values(["Position"]).reset_index(drop=True)
            reference_seq = ref_gene_seq_dict[str(gene)]
            gene_start = int(gene_df["gene_start"].iloc[0])
            variant_rows = [row._asdict() for row in gene_df.itertuples(index=False)]
            for sample in self.sample_columns:
                gene_seq_dict[sample][str(gene)] = self._apply_variants_to_gene(
                    reference_seq=reference_seq,
                    gene_start=gene_start,
                    variant_rows=variant_rows,
                    sample=sample,
                )

        gene_names = sorted(ref_gene_seq_dict)
        matrix_rows = [{"sample": sample, **{gene: gene_seq_dict[sample].get(gene, "") for gene in gene_names}} for sample in self.sample_columns]
        out_df = pd.DataFrame(matrix_rows, columns=["sample"] + gene_names)
        out_path = self.outdir / "gene_sequence_matrix.csv"
        out_df.to_csv(out_path, index=False)
        self._save_json(gene_seq_dict, self.dict_dir / "gene_sequence_dict.json")
        self.logger.info("Saved gene_sequence_matrix: %s shape=%s", out_path, out_df.shape)
        return out_df

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

    def _fit_pca_block(self, block: np.ndarray, gene: str) -> tuple[np.ndarray, list[str]]:
        if not self.use_pca:
            return block.astype(np.float32), [f"{gene}-embed-{i}" for i in range(block.shape[1])]

        from sklearn.decomposition import PCA
        import joblib

        if block.shape[0] <= 1:
            return block[:, :1].astype(np.float32), [f"{gene}-PC1"]

        max_comp = min(block.shape[0], block.shape[1])
        pca_full = PCA(n_components=max_comp, svd_solver="full")
        pca_full.fit(block)
        cum = np.cumsum(pca_full.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cum, self.pca_var_threshold) + 1)
        n_comp = max(1, min(n_comp, max_comp))
        pca = PCA(n_components=n_comp, svd_solver="full")
        reduced = pca.fit_transform(block).astype(np.float32)
        joblib.dump(pca, self.pca_dir / f"{gene}.joblib")
        return reduced, [f"{gene}-PC{i + 1}" for i in range(reduced.shape[1])]

    def build_gene_feature_matrix(self, gene_seq_df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_embedder()
        seq_values = gene_seq_df.iloc[:, 1:].to_numpy().ravel()
        unique_sequences = sorted({str(seq) for seq in seq_values if str(seq) != ""})
        sequence_embedding_dict = {seq: self._embed_sequence(seq).tolist() for seq in unique_sequences}
        self._save_json(sequence_embedding_dict, self.dict_dir / "unique_gene_sequence_embedding_dict.json")

        blocks: list[np.ndarray] = []
        columns = ["sample"]
        gene_embedding_dict: dict[str, dict[str, list[float] | float]] = {sample: {} for sample in gene_seq_df["sample"].tolist()}

        for gene in gene_seq_df.columns[1:]:
            seqs = gene_seq_df[gene].astype(str).tolist()
            embed_block = np.vstack([np.asarray(sequence_embedding_dict[seq], dtype=np.float32) for seq in seqs])
            for sample, vec in zip(gene_seq_df["sample"].tolist(), embed_block):
                gene_embedding_dict[sample][gene] = vec.astype(np.float32).tolist()
            reduced, cols = self._fit_pca_block(embed_block, gene)
            for sample_idx, sample in enumerate(gene_seq_df["sample"].tolist()):
                for col_idx, col in enumerate(cols):
                    gene_embedding_dict[sample][col] = float(reduced[sample_idx, col_idx])
            blocks.append(reduced)
            columns.extend(cols)

        matrix = np.concatenate(blocks, axis=1) if blocks else np.empty((len(gene_seq_df), 0), dtype=np.float32)
        out_df = pd.DataFrame(matrix, columns=columns[1:])
        out_df.insert(0, "sample", gene_seq_df["sample"].tolist())
        out_path = self._save_matrix(out_df, self.outdir / "gene_feature_matrix", self.gene_feature_format)
        self._save_json(gene_embedding_dict, self.dict_dir / "gene_feature_embedding_dict.json")
        self.logger.info("Saved gene_feature_matrix: %s shape=%s", out_path, out_df.shape)
        return out_df

    def run(self) -> Dict[str, pd.DataFrame]:
        self.geno_df = self._load_geno_df()
        self._validate_ref_against_fasta()
        genotype_df = self.build_genotype_matrix()
        gene_seq_df = self.build_gene_sequence_matrix()
        gene_feature_df = self.build_gene_feature_matrix(gene_seq_df)
        self.logger.info("VariantFeatureBuilder finished")
        return {
            "genotype_012": genotype_df,
            "gene_sequence_matrix": gene_seq_df,
            "gene_feature_matrix": gene_feature_df,
        }


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
    parser = argparse.ArgumentParser(description="Build genotype and gene-sequence feature datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_genotype = subparsers.add_parser("genotype", help="Build geno_df from site CSV and VCF")
    p_genotype.add_argument("--vcf_path", required=True)
    p_genotype.add_argument("--site_df_path", required=True)
    p_genotype.add_argument("--outprefix", required=True)

    p_build = subparsers.add_parser("build", help="Build genotype_012, gene_sequence_matrix, and gene_feature_matrix from geno_df")
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
    p_build.add_argument("--gene_feature_format", choices=["csv", "parquet"], default="csv")
    return parser


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
        outputs = VariantFeatureBuilder(
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
            gene_feature_format=args.gene_feature_format,
        ).run()
        print({name: df.shape for name, df in outputs.items()})
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
