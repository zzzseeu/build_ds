"""Build geno_df, genotype 012, gene-sequence, and gene-feature datasets.

This pipeline consumes the site CSV exported by ``gwas_qtl_variant_extractor.py``.
All genomic coordinates are interpreted as 1-based.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from logging import init_logger
from utils import standard_chrom, standard_sample_name

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None


class VariantFeatureBuilder:
    """End-to-end pipeline from site CSV + VCF + FASTA to downstream matrices."""

    SITE_COLUMNS = ["Chromosome", "Position", "Gene", "gene_start", "gene_end", "qtl_trait", "gwas_trait"]
    GENO_META_COLUMNS = ["Chromosome", "Position", "Gene", "gene_start", "gene_end", "REF", "ALT", "qtl_trait", "gwas_trait"]

    def __init__(
        self,
        site_df_path: str | None = None,
        site_df: pd.DataFrame | None = None,
        vcf_path: str | None = None,
        fasta_path: str | None = None,
        outdir: str | None = None,
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
        if site_df_path is None and site_df is None:
            raise ValueError("site_df_path or site_df must be provided")
        if vcf_path is None:
            raise ValueError("vcf_path is required")
        if fasta_path is None:
            raise ValueError("fasta_path is required")
        if outdir is None:
            raise ValueError("outdir is required")

        self.site_df_path = Path(site_df_path) if site_df_path else None
        self.site_df = site_df.copy() if site_df is not None else None
        self.vcf_path = Path(vcf_path)
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

        self.site_df_loaded: pd.DataFrame | None = None
        self.geno_df: pd.DataFrame | None = None
        self.sample_columns: list[str] = []
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
    def _standardize_sample_names(raw_samples: List[str]) -> Dict[str, str]:
        sample_map: Dict[str, str] = {}
        used: set[str] = set()
        for idx, sample in enumerate(raw_samples):
            std = standard_sample_name(sample) or f"sample_{idx + 1}"
            if std in used:
                std = f"{std}_{idx + 1}"
            used.add(std)
            sample_map[sample] = std
        return sample_map

    @staticmethod
    def _save_json(data: dict, path: Path) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _save_matrix(df: pd.DataFrame, path_without_suffix: Path, fmt: str) -> Path:
        if fmt == "parquet":
            path = path_without_suffix.with_suffix(".parquet")
            df.to_parquet(path, index=False)
            return path
        path = path_without_suffix.with_suffix(".csv")
        df.to_csv(path, index=False)
        return path

    @staticmethod
    def _save_matrix_dual(df: pd.DataFrame, path_without_suffix: Path) -> tuple[Path, Path]:
        csv_path = path_without_suffix.with_suffix(".csv")
        parquet_path = path_without_suffix.with_suffix(".parquet")
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
        return csv_path, parquet_path

    @staticmethod
    def _merge_trait_values(values: pd.Series) -> str:
        uniq = sorted({str(v).strip() for v in values if str(v).strip()})
        return ";".join(uniq)

    def _ensure_embedder(self) -> None:
        if self.embedder is None:
            raise ValueError("embedder or embedder_type must be provided")

    @staticmethod
    def _progress(iterable, **kwargs):
        if tqdm is None:
            return iterable
        return tqdm(iterable, **kwargs)

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
        self.site_df_loaded = df
        return df

    def _build_geno_df(self) -> pd.DataFrame:
        try:
            import pysam  # type: ignore
        except Exception as exc:
            raise ImportError("pysam is required to read VCF/FASTA") from exc

        site_df = self.site_df_loaded if self.site_df_loaded is not None else self._load_site_df()
        target_sites = {
            (str(chrom), int(pos))
            for chrom, pos in site_df[["Chromosome", "Position"]].drop_duplicates().itertuples(index=False, name=None)
        }

        vcf = pysam.VariantFile(str(self.vcf_path))
        raw_samples = list(vcf.header.samples)
        sample_map = self._standardize_sample_names(raw_samples)
        self.sample_columns = [sample_map[s] for s in raw_samples]

        genotype_map: dict[tuple[str, int], dict[str, str | int]] = {}
        for rec in self._progress(vcf, desc="Scanning VCF records", unit="record"):
            chrom = standard_chrom(str(rec.chrom))
            if chrom is None:
                continue
            pos = int(rec.pos)
            key = (chrom, pos)
            if key not in target_sites:
                continue

            row_map: dict[str, str | int] = {
                "REF": str(rec.ref),
                "ALT": str(rec.alts[0]) if rec.alts else "",
            }
            row_map.update({
                sample_map[sample]: self._encode_gt(rec.samples[sample].get("GT"))
                for sample in raw_samples
            })
            genotype_map[key] = row_map

        rows: list[dict[str, str | int]] = []
        for row in self._progress(
            site_df.itertuples(index=False),
            total=len(site_df),
            desc="Building geno_df rows",
            unit="row",
        ):
            key = (str(row.Chromosome), int(row.Position))
            site_gt = genotype_map.get(key, {"REF": "", "ALT": "", **{sample: 0 for sample in self.sample_columns}})
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
            out_row.update({sample: int(site_gt.get(sample, 0)) for sample in self.sample_columns})
            rows.append(out_row)

        geno_df = pd.DataFrame(rows, columns=self.GENO_META_COLUMNS + self.sample_columns)
        geno_df = self._deduplicate_geno_df(geno_df)
        if self.sample_columns:
            geno_df[self.sample_columns] = geno_df[self.sample_columns].astype(np.int8)
        self.geno_df = geno_df
        geno_path = self.outdir / "geno_df.csv"
        geno_df.to_csv(geno_path, index=False)
        self.logger.info("Saved geno_df: %s shape=%s", geno_path, geno_df.shape)
        return geno_df

    def _deduplicate_geno_df(self, geno_df: pd.DataFrame) -> pd.DataFrame:
        subset = ["Gene", "Chromosome", "Position"]
        dup_mask = geno_df.duplicated(subset=subset, keep=False)
        if not dup_mask.any():
            return geno_df.reset_index(drop=True)

        dup_count = int(dup_mask.sum())
        self.logger.warning(
            "Found duplicated gene-site rows in geno_df: rows=%d unique_sites=%d; deduplicating by %s",
            dup_count,
            int(geno_df.loc[dup_mask, subset].drop_duplicates().shape[0]),
            "+".join(subset),
        )

        dup_rows = (
            geno_df.loc[dup_mask, ["Gene", "Chromosome", "Position", "REF", "ALT"]]
            .drop_duplicates()
        )
        for row in self._progress(
            dup_rows.itertuples(index=False),
            total=len(dup_rows),
            desc="Checking duplicated geno_df sites",
            unit="site",
        ):
            group = geno_df[
                (geno_df["Gene"] == row.Gene)
                & (geno_df["Chromosome"] == row.Chromosome)
                & (geno_df["Position"] == row.Position)
            ]
            if group["REF"].nunique(dropna=False) > 1 or group["ALT"].nunique(dropna=False) > 1:
                self.logger.warning(
                    "Inconsistent duplicated site for %s at %s:%d REF=%s ALT=%s",
                    row.Gene,
                    row.Chromosome,
                    int(row.Position),
                    sorted(group["REF"].astype(str).unique().tolist()),
                    sorted(group["ALT"].astype(str).unique().tolist()),
                )

        agg: dict[str, str] = {
            "gene_start": "first",
            "gene_end": "first",
            "REF": "first",
            "ALT": "first",
            "qtl_trait": self._merge_trait_values,
            "gwas_trait": self._merge_trait_values,
        }
        agg.update({sample: "max" for sample in self.sample_columns})

        out = (
            geno_df.groupby(subset, as_index=False, sort=False)
            .agg(agg)
            .loc[:, self.GENO_META_COLUMNS + self.sample_columns]
            .reset_index(drop=True)
        )
        self.logger.info("Deduplicated geno_df by %s: %d -> %d rows", "+".join(subset), len(geno_df), len(out))
        return out

    def _get_fasta(self):
        try:
            import pysam  # type: ignore
        except Exception as exc:
            raise ImportError("pysam is required to read FASTA") from exc
        return pysam.FastaFile(str(self.fasta_path))

    def _build_fasta_chrom_map(self, fasta) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for ref in fasta.references:
            std = standard_chrom(ref)
            if std is not None and std not in out:
                out[std] = ref
        return out

    @staticmethod
    def _resolve_fasta_chrom(chrom: str, chrom_map: Dict[str, str]) -> str:
        if chrom not in chrom_map:
            raise KeyError(f"Chromosome {chrom} not found in FASTA")
        return chrom_map[chrom]

    def build_genotype_012(self) -> pd.DataFrame:
        assert self.geno_df is not None
        site_df = self.geno_df.drop_duplicates(subset=["Chromosome", "Position"]).reset_index(drop=True)
        feature_names = [f"{chrom}:{pos}" for chrom, pos in site_df[["Chromosome", "Position"]].itertuples(index=False, name=None)]
        matrix = site_df[self.sample_columns].to_numpy(dtype=np.int8).T
        out_df = pd.DataFrame(matrix, columns=feature_names)
        out_df.insert(0, "sample", self.sample_columns)
        csv_path, parquet_path = self._save_matrix_dual(out_df, self.outdir / "genotype_012")
        self.logger.info(
            "Saved genotype_012: csv=%s parquet=%s shape=%s",
            csv_path,
            parquet_path,
            out_df.shape,
        )
        del matrix
        del site_df
        gc.collect()
        return out_df

    def _validate_variant_ref_in_gene_seq(
        self,
        reference_seq: str,
        gene_start: int,
        row: dict[str, str | int],
    ) -> tuple[int, int] | None:
        pos = int(row["Position"])
        ref = str(row["REF"])
        rel_start = pos - gene_start
        rel_end = rel_start + len(ref)
        if rel_start < 0 or rel_end > len(reference_seq):
            self.logger.warning(
                "Variant out of gene bounds for %s at %s:%d gene_start=%d gene_len=%d ref=%s",
                row["Gene"],
                row["Chromosome"],
                pos,
                gene_start,
                len(reference_seq),
                ref,
            )
            return None

        observed = reference_seq[rel_start:rel_end]
        if ref and observed.upper() != ref.upper():
            self.logger.warning(
                "REF mismatch between VCF and gene reference for %s at %s:%d expected=%s observed=%s",
                row["Gene"],
                row["Chromosome"],
                pos,
                ref,
                observed,
            )
            return None
        return rel_start, rel_end

    def _apply_variants_to_gene(
        self,
        reference_seq: str,
        gene_start: int,
        variant_rows: Iterable[dict[str, str | int]],
        sample: str,
    ) -> str:
        parts: list[str] = []
        cursor = 0
        rows = sorted(list(variant_rows), key=lambda x: int(x["Position"]))
        for row in rows:
            if int(row[sample]) <= 0:
                continue
            alt = str(row["ALT"])
            validated_span = self._validate_variant_ref_in_gene_seq(
                reference_seq=reference_seq,
                gene_start=gene_start,
                row=row,
            )
            if validated_span is None:
                continue
            pos = int(row["Position"])
            rel_start, rel_end = validated_span
            if rel_start < cursor:
                self.logger.warning(
                    "Overlapping variants for %s at %s:%d; skipping current variant",
                    row["Gene"],
                    row["Chromosome"],
                    pos,
                )
                continue
            parts.append(reference_seq[cursor:rel_start])
            parts.append(alt)
            cursor = rel_end
        parts.append(reference_seq[cursor:])
        return "".join(parts)

    def _build_sample_gene_sequences(
        self,
        gene: str,
        gene_df: pd.DataFrame,
        fasta,
        chrom_map: dict[str, str],
    ) -> list[str]:
        gene_df = gene_df.sort_values(["Position"]).reset_index(drop=True)
        gene_start = int(gene_df["gene_start"].iloc[0])
        fasta_chrom = self._resolve_fasta_chrom(str(gene_df["Chromosome"].iloc[0]), chrom_map)
        reference_seq = fasta.fetch(fasta_chrom, gene_start - 1, int(gene_df["gene_end"].iloc[0]))
        variant_rows = [row._asdict() for row in gene_df.itertuples(index=False)]
        active_samples = sum(bool(int(gene_df[sample].fillna(0).gt(0).any())) for sample in self.sample_columns)
        self.logger.info(
            "Applying gene variants: gene=%s variant_rows=%d active_samples=%d",
            gene,
            len(variant_rows),
            active_samples,
        )
        return [
            self._apply_variants_to_gene(
                reference_seq=reference_seq,
                gene_start=gene_start,
                variant_rows=variant_rows,
                sample=sample,
            )
            for sample in self.sample_columns
        ]

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

    def _load_cached_embedding(self, seq: str) -> np.ndarray:
        seq_hash = hashlib.sha1(seq.encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"{seq_hash}.npy"
        if not cache_path.exists():
            return self._embed_sequence(seq)
        return np.load(cache_path)

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
        self.logger.info(
            "Fitted PCA for gene=%s samples=%d input_dim=%d output_dim=%d var_threshold=%.3f explained=%.4f",
            gene,
            block.shape[0],
            block.shape[1],
            reduced.shape[1],
            self.pca_var_threshold,
            float(cum[n_comp - 1]),
        )
        return reduced, [f"{gene}-PC{i + 1}" for i in range(reduced.shape[1])]

    def build_gene_feature_matrix(self) -> pd.DataFrame:
        self._ensure_embedder()
        assert self.geno_df is not None
        self._embed_total = 0
        self._embed_cache_hits = 0
        self._embed_cache_misses = 0
        sample_list = list(self.sample_columns)
        fasta = self._get_fasta()
        chrom_map = self._build_fasta_chrom_map(fasta)
        grouped = self.geno_df.groupby("Gene", sort=True)
        gene_count = int(self.geno_df["Gene"].nunique())
        self.logger.info(
            "Building gene feature matrix: samples=%d genes=%d embedder_type=%s model=%s use_pca=%s",
            len(sample_list),
            gene_count,
            self.embedder_type,
            self.model_name_or_path,
            self.use_pca,
        )
        block_meta: list[tuple[Path, list[str]]] = []
        total_feature_cols = 0
        with tempfile.TemporaryDirectory(dir=self.outdir, prefix="gene_feature_blocks_") as tempdir_name:
            tempdir = Path(tempdir_name)
            for gene, gene_df in self._progress(
                grouped,
                desc="Projecting gene embeddings",
                total=gene_count,
                unit="gene",
            ):
                seqs = self._build_sample_gene_sequences(str(gene), gene_df, fasta, chrom_map)
                unique_gene_sequences = sorted({seq for seq in seqs if seq})
                gene_embed_cache = {
                    seq: self._load_cached_embedding(seq).astype(np.float32)
                    for seq in unique_gene_sequences
                }
                if not gene_embed_cache:
                    self.logger.warning("Skipping gene with empty sequences: %s", gene)
                    continue
                embed_block = np.vstack([gene_embed_cache[seq] for seq in seqs]).astype(np.float32, copy=False)
                reduced, cols = self._fit_pca_block(embed_block, gene)
                block_path = tempdir / f"{gene}.npy"
                np.save(block_path, reduced.astype(np.float32, copy=False))
                block_meta.append((block_path, cols))
                total_feature_cols += len(cols)
                del seqs
                del unique_gene_sequences
                del gene_embed_cache
                del embed_block
                del reduced
                gc.collect()

            self.logger.info(
                "Embedding completed: total=%d cache_hits=%d cache_misses=%d cache_hit_rate=%.2f%% output_feature_cols=%d",
                self._embed_total,
                self._embed_cache_hits,
                self._embed_cache_misses,
                100.0 * self._embed_cache_hits / self._embed_total if self._embed_total else 0.0,
                total_feature_cols,
            )
            memmap_path = tempdir / "gene_feature_matrix.dat"
            matrix = np.memmap(
                memmap_path,
                dtype=np.float32,
                mode="w+",
                shape=(len(sample_list), total_feature_cols),
            )
            columns: list[str] = []
            offset = 0
            for block_path, cols in self._progress(
                block_meta,
                total=len(block_meta),
                desc="Assembling feature matrix",
                unit="block",
            ):
                block = np.load(block_path)
                width = block.shape[1]
                matrix[:, offset:offset + width] = block
                columns.extend(cols)
                offset += width
                del block
            matrix.flush()
            out_df = pd.DataFrame(np.asarray(matrix), columns=columns)
        out_df.insert(0, "sample", sample_list)
        csv_path, parquet_path = self._save_matrix_dual(out_df, self.outdir / "gene_feature_matrix")
        self.logger.info(
            "Saved gene_feature_matrix: csv=%s parquet=%s shape=%s",
            csv_path,
            parquet_path,
            out_df.shape,
        )
        return out_df

    def run(self) -> Dict[str, pd.DataFrame]:
        self._load_site_df()
        self._build_geno_df()
        self.site_df_loaded = None
        self.site_df = None
        gc.collect()
        genotype_012 = self.build_genotype_012()
        gc.collect()
        gene_feature_matrix = self.build_gene_feature_matrix()
        gc.collect()
        self.geno_df = None
        gc.collect()
        gc.collect()
        self.logger.info("VariantFeatureBuilder finished")
        return {
            "genotype_012": genotype_012,
            "gene_feature_matrix": gene_feature_matrix,
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
    parser = argparse.ArgumentParser(description="Build geno_df, genotype_012, gene_sequence_matrix, and gene_feature_matrix")
    parser.add_argument("--site_df_path", required=True)
    parser.add_argument("--vcf_path", required=True)
    parser.add_argument("--fasta_path", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--embedder_type", required=True, choices=["generator", "evo2", "nt", "agront", "rice8k"])
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--no-local_files_only", dest="local_files_only", action="store_false")
    parser.add_argument("--embedder_kwargs", default=None, help='JSON object string, e.g. \'{"torch_dtype":"bfloat16"}\'')
    parser.add_argument("--use_pca", action="store_true", default=True)
    parser.add_argument("--no-use_pca", dest="use_pca", action="store_false")
    parser.add_argument("--pca_var_threshold", type=float, default=0.95)
    parser.add_argument("--gene_feature_format", choices=["csv", "parquet"], default="csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = VariantFeatureBuilder(
        site_df_path=args.site_df_path,
        vcf_path=args.vcf_path,
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
    print({name: df.shape for name, df in outputs.items() if isinstance(df, pd.DataFrame)})


if __name__ == "__main__":
    main()
