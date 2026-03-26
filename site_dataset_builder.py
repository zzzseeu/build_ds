"""Build site, genotype, variant-effect, and gene-feature datasets from GWAS/QTL/VCF/FASTA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    extract_gff3_feature_interval_trees,
    extract_gff3_feature_interval_trees_gffutils,
    initLogger,
    query_feature_interval_trees,
    standard_chrom,
    standard_sample_name,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


class SiteDatasetBuilder:
    """Single-entry pipeline for site filtering and dataset construction."""

    def __init__(self, config: dict):
        self.config = dict(config)
        self.gwas_csv_path = Path(self.config["gwas_csv_path"])
        self.qtl_csv_path = Path(self.config["qtl_csv_path"])
        self.gff3_path = Path(self.config["gff3_path"])
        self.fasta_path = Path(self.config["fasta_path"])
        self.vcf_path = Path(self.config["vcf_path"])
        self.type = str(self.config["type"]).strip().lower()
        self.outdir = Path(self.config.get("outdir", "site_dataset_builder_outputs"))
        self.ext_len = int(self.config.get("ext_len", 500))
        self.k = int(self.config.get("k", 100))
        self.gene_csv_path = self.config.get("gene_csv_path")
        self.gff3_feature = self.config.get("gff3_feature", "gene")
        self.use_gffutils = bool(self.config.get("use_gffutils", False))
        self.pvalue_threshold = float(self.config.get("pvalue_threshold", 1e6))
        self.LOD_threshold = float(self.config.get("LOD_threshold", 0.0))
        self.PVE_threshold = float(self.config.get("PVE_threshold", 0.0))
        self.embedder_type = self.config["embedder_type"]
        self.model_name_or_path = self.config["model_name_or_path"]
        self.device = self.config.get("device", "cpu")
        self.pooling = self.config.get("pooling", "mean")
        self.local_files_only = bool(self.config.get("local_files_only", True))
        self.embedder_kwargs = dict(self.config.get("embedder_kwargs", {}))
        self.split_test_ratio = self.config.get("split_test_ratio")
        self.isolated_sample_csv = self.config.get("isolated_sample_csv")
        self.split_random_state = int(self.config.get("split_random_state", 42))

        if self.type not in {"union", "intersect"}:
            raise ValueError("type must be one of: union, intersect")
        if self.ext_len < 0:
            raise ValueError("ext_len must be >= 0")
        if self.k < 0:
            raise ValueError("k must be >= 0")
        if self.split_test_ratio is not None and not 0 < float(self.split_test_ratio) < 1:
            raise ValueError("split_test_ratio must be between 0 and 1")

        self.outdir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.outdir / "embedding_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = initLogger(self.outdir / "site_dataset_builder.log")
        self.sample_map: dict[str, str] = {}
        self.sample_list: list[str] = []

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
    def _progress(iterable, **kwargs):
        if tqdm is None:
            return iterable
        return tqdm(iterable, **kwargs)

    @staticmethod
    def _encode_gt(gt: tuple[int | None, ...] | None) -> int:
        if gt is None or any(x is None for x in gt):
            return 0
        alt_count = sum(1 for x in gt if x and x > 0)
        if alt_count <= 0:
            return 0
        if alt_count == 1:
            return 1
        return 2

    @staticmethod
    def _standardize_sample_names(raw_samples: list[str]) -> dict[str, str]:
        sample_map: dict[str, str] = {}
        used: set[str] = set()
        for idx, sample in enumerate(raw_samples):
            std = standard_sample_name(sample) or f"sample_{idx + 1}"
            if std in used:
                std = f"{std}_{idx + 1}"
            used.add(std)
            sample_map[str(sample)] = std
        return sample_map

    def _set_sample_metadata(self, raw_samples: list[str]) -> None:
        self.sample_map = self._standardize_sample_names([str(sample) for sample in raw_samples])
        self.sample_list = [self.sample_map[str(sample)] for sample in raw_samples]

    def _save_sample_mapping(self) -> Path | None:
        if not self.sample_map:
            return None
        out_path = self.outdir / "sample_name_mapping.csv"
        mapping_df = pd.DataFrame(
            {
                "raw_sample": list(self.sample_map.keys()),
                "sample": list(self.sample_map.values()),
            }
        )
        mapping_df.to_csv(out_path, index=False)
        self.logger.info(f"Saved sample mapping: {out_path} shape={mapping_df.shape}")
        return out_path

    def _read_isolated_samples(self) -> list[str]:
        if not self.isolated_sample_csv:
            return []
        isolated_df = pd.read_csv(self.isolated_sample_csv)
        if isolated_df.empty or isolated_df.shape[1] == 0:
            return []
        isolated = [
            str(x).strip()
            for x in isolated_df.iloc[:, 0].astype(str).tolist()
            if str(x).strip()
        ]
        sample_set = set(self.sample_list)
        isolated = [sample for sample in isolated if sample in sample_set]
        if len(isolated) != len(set(isolated)):
            isolated = list(dict.fromkeys(isolated))
        return isolated

    def _build_split_sample_lists(self) -> tuple[list[str], list[str]]:
        if self.split_test_ratio is None:
            raise ValueError("split_test_ratio is not configured")
        isolated = self._read_isolated_samples()
        target_test_size = max(len(isolated), int(math.ceil(len(self.sample_list) * float(self.split_test_ratio))))
        available = [sample for sample in self.sample_list if sample not in set(isolated)]
        rng = np.random.default_rng(self.split_random_state)
        extra_needed = max(0, target_test_size - len(isolated))
        if extra_needed > len(available):
            extra_needed = len(available)
        extra_test = (
            rng.choice(np.array(available, dtype=object), size=extra_needed, replace=False).tolist()
            if extra_needed > 0
            else []
        )
        test_samples = isolated + extra_test
        test_set = set(test_samples)
        train_val_samples = [sample for sample in self.sample_list if sample not in test_set]
        return train_val_samples, test_samples

    @staticmethod
    def _subset_by_samples(df: pd.DataFrame, sample_list: list[str]) -> pd.DataFrame:
        return df.set_index("sample").loc[sample_list].reset_index()

    @staticmethod
    def _save_dual(df: pd.DataFrame, path_without_suffix: Path) -> tuple[Path, Path]:
        csv_path = path_without_suffix.with_suffix(".csv")
        parquet_path = path_without_suffix.with_suffix(".parquet")
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
        return csv_path, parquet_path

    @staticmethod
    def _merge_values(values: pd.Series) -> str:
        uniq = sorted({str(v).strip() for v in values if str(v).strip()})
        return ",".join(uniq)

    @staticmethod
    def _chrom_sort_key(chrom: str) -> tuple[int, str]:
        digits = "".join(ch for ch in str(chrom) if ch.isdigit())
        if digits:
            return (int(digits), str(chrom))
        return (10**9, str(chrom))

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
        seq_hash = hashlib.sha1(seq.encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"{seq_hash}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        vec = self._to_numpy_1d(self.embedder(seq))
        np.save(cache_path, vec)
        return vec

    @staticmethod
    def _build_fasta_chrom_map(fasta) -> dict[str, str]:
        chrom_map: dict[str, str] = {}
        for ref in fasta.references:
            chrom = standard_chrom(ref)
            if chrom is not None and chrom not in chrom_map:
                chrom_map[chrom] = ref
        return chrom_map

    @staticmethod
    def _fetch_window_with_padding(fasta, fasta_chrom: str, start_1based: int, end_1based: int) -> str:
        if end_1based < start_1based:
            return ""
        ref_len = fasta.get_reference_length(fasta_chrom)
        left_pad = max(0, 1 - start_1based)
        right_pad = max(0, end_1based - ref_len)
        fetch_start = max(1, start_1based)
        fetch_end = min(ref_len, end_1based)
        seq = ""
        if fetch_start <= fetch_end:
            seq = fasta.fetch(fasta_chrom, fetch_start - 1, fetch_end).upper()
        return ("N" * left_pad) + seq + ("N" * right_pad)

    def _load_gene_filter(self) -> set[str] | None:
        if not self.gene_csv_path:
            return None
        gene_df = pd.read_csv(self.gene_csv_path)
        if gene_df.shape[1] < 1:
            raise ValueError("gene_csv_path must contain at least one column with gene_id")
        gene_set = {
            str(x).strip()
            for x in gene_df.iloc[:, 0].astype(str)
            if str(x).strip()
        }
        return gene_set or None

    def _read_gwas(self) -> pd.DataFrame:
        df = pd.read_csv(self.gwas_csv_path).copy()
        required = {"Chromosome", "Position", "Trait"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"GWAS CSV missing columns: {missing}")
        df["Chromosome"] = df["Chromosome"].astype(str).map(standard_chrom)
        df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
        df["Trait"] = df["Trait"].astype(str)
        df = df[df["Chromosome"].notna() & df["Position"].notna()].copy()
        df["Position"] = df["Position"].astype(int)
        if "pvalue" in df.columns:
            df["pvalue"] = pd.to_numeric(df["pvalue"], errors="coerce")
            df = df[df["pvalue"].notna() & (df["pvalue"] < self.pvalue_threshold)].copy()
        self.logger.info(f"GWAS loaded: shape={df.shape}")
        return df

    def _read_qtl(self) -> pd.DataFrame:
        df = pd.read_csv(self.qtl_csv_path).copy()
        required = {"Chromosome", "start_pos", "end_pos", "Trait"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"QTL CSV missing columns: {missing}")
        df["Chromosome"] = df["Chromosome"].astype(str).map(standard_chrom)
        df["start_pos"] = pd.to_numeric(df["start_pos"], errors="coerce")
        df["end_pos"] = pd.to_numeric(df["end_pos"], errors="coerce")
        df["Trait"] = df["Trait"].astype(str)
        if "LOD" in df.columns:
            df["LOD"] = pd.to_numeric(df["LOD"], errors="coerce")
            df = df[df["LOD"].notna() & (df["LOD"] > self.LOD_threshold)].copy()
        if "PVE" in df.columns:
            df["PVE"] = pd.to_numeric(df["PVE"], errors="coerce")
            df = df[df["PVE"].notna() & (df["PVE"] > self.PVE_threshold)].copy()
        df = df[
            df["Chromosome"].notna() & df["start_pos"].notna() & df["end_pos"].notna()
        ].copy()
        df["start_pos"] = df["start_pos"].astype(int)
        df["end_pos"] = df["end_pos"].astype(int)
        swapped = df["start_pos"] > df["end_pos"]
        if swapped.any():
            df.loc[swapped, ["start_pos", "end_pos"]] = df.loc[
                swapped, ["end_pos", "start_pos"]
            ].to_numpy()
        self.logger.info(f"QTL loaded: shape={df.shape}")
        return df

    def _build_gwas_site_map(self, gwas_df: pd.DataFrame) -> dict[tuple[str, int], dict[str, bool]]:
        site_map: dict[tuple[str, int], dict[str, bool]] = {}
        for row in self._progress(
            gwas_df.itertuples(index=False),
            total=len(gwas_df),
            desc="Building GWAS site map",
            unit="site",
        ):
            key = (str(row.Chromosome), int(row.Position))
            trait_map = site_map.setdefault(key, {})
            trait_map[str(row.Trait)] = True
        return site_map

    def _build_qtl_trees(self, qtl_df: pd.DataFrame) -> dict[str, object]:
        interval_df = qtl_df.rename(columns={"start_pos": "Start", "end_pos": "End"}).copy()
        interval_df["Feature"] = interval_df["Trait"].astype(str)
        interval_df["ID"] = interval_df["QTL"].astype(str) if "QTL" in interval_df.columns else ""
        interval_df["Name"] = interval_df["Trait"].astype(str)
        interval_df["Parent"] = ""
        from utils import build_feature_interval_trees

        return build_feature_interval_trees(interval_df, ext_len=0)

    def _select_sites_from_vcf(
        self,
        gwas_site_map: dict[tuple[str, int], dict[str, bool]],
        qtl_trees: dict[str, object],
    ) -> pd.DataFrame:
        try:
            import pysam  # type: ignore
        except Exception as exc:
            raise ImportError("pysam is required for VCF scanning") from exc

        rows: list[dict[str, str | int]] = []
        with pysam.VariantFile(str(self.vcf_path)) as vcf:
            for rec in self._progress(vcf, desc="Selecting sites from VCF", unit="site"):
                chrom = standard_chrom(str(rec.chrom))
                if chrom is None:
                    continue
                pos = int(rec.pos)
                gwas_traits = sorted(gwas_site_map.get((chrom, pos), {}).keys())
                qtl_hits = query_feature_interval_trees(qtl_trees, chrom, pos)
                qtl_traits = sorted({str(hit.get("feature", "")).strip() for hit in qtl_hits if str(hit.get("feature", "")).strip()})

                if self.type == "intersect":
                    shared_traits = sorted(set(gwas_traits) & set(qtl_traits))
                    keep = bool(shared_traits)
                    gwas_traits = shared_traits
                    qtl_traits = shared_traits
                else:
                    keep = bool(gwas_traits or qtl_traits)

                if not keep:
                    continue

                rows.append(
                    {
                        "Chromosome": chrom,
                        "Position": pos,
                        "qtl_trait": ",".join(qtl_traits),
                        "gwas_trait": ",".join(gwas_traits),
                        "REF": str(rec.ref).upper(),
                        "ALT": str(rec.alts[0]).upper() if rec.alts else "",
                    }
                )

        out_df = pd.DataFrame(
            rows,
            columns=["Chromosome", "Position", "qtl_trait", "gwas_trait", "REF", "ALT"],
        )
        if out_df.empty:
            return out_df
        out_df = (
            out_df.groupby(["Chromosome", "Position", "REF", "ALT"], as_index=False)
            .agg({"qtl_trait": self._merge_values, "gwas_trait": self._merge_values})
            .assign(_chrom_order=lambda x: x["Chromosome"].map(self._chrom_sort_key))
            .sort_values(["_chrom_order", "Position", "REF", "ALT"])
            .drop(columns="_chrom_order")
            .reset_index(drop=True)
        )
        return out_df

    def _save_initial_sites(self, site_df: pd.DataFrame) -> Path:
        out_path = self.outdir / "sites_union_or_intersect.csv"
        site_df.loc[:, ["Chromosome", "Position", "qtl_trait", "gwas_trait"]].to_csv(out_path, index=False)
        self.logger.info(f"Saved initial sites: {out_path} shape={site_df.shape}")
        return out_path

    def _build_gene_trees(self) -> dict[str, object]:
        if self.use_gffutils:
            trees = extract_gff3_feature_interval_trees_gffutils(
                gff3_path=self.gff3_path,
                feature=self.gff3_feature,
                ext_len=self.ext_len,
            )
        else:
            trees = extract_gff3_feature_interval_trees(
                gff3_path=self.gff3_path,
                feature=self.gff3_feature,
                ext_len=self.ext_len,
            )
        self.logger.info(f"Built gene trees: chromosomes={len(trees)} ext_len={self.ext_len}")
        return trees

    def _map_sites_to_genes(self, site_df: pd.DataFrame, gene_trees: dict[str, object]) -> pd.DataFrame:
        rows: list[dict[str, str | int]] = []
        for row in self._progress(
            site_df.itertuples(index=False),
            total=len(site_df),
            desc="Mapping sites to genes",
            unit="site",
        ):
            hits = query_feature_interval_trees(gene_trees, str(row.Chromosome), int(row.Position))
            for hit in hits:
                gene_id = (
                    str(hit.get("id", "")).strip()
                    or str(hit.get("name", "")).strip()
                    or str(hit.get("parent", "")).strip()
                )
                if not gene_id:
                    continue
                rows.append(
                    {
                        "Chromosome": str(row.Chromosome),
                        "Position": int(row.Position),
                        "qtl_trait": str(row.qtl_trait),
                        "gwas_trait": str(row.gwas_trait),
                        "gene_id": gene_id,
                        "gene_start": int(hit.get("start", 0)),
                        "gene_end": int(hit.get("end", 0)),
                        "REF": str(row.REF),
                        "ALT": str(row.ALT),
                    }
                )
        out_df = pd.DataFrame(
            rows,
            columns=[
                "Chromosome",
                "Position",
                "qtl_trait",
                "gwas_trait",
                "gene_id",
                "gene_start",
                "gene_end",
                "REF",
                "ALT",
            ],
        )
        if out_df.empty:
            return out_df
        out_df = (
            out_df.groupby(["Chromosome", "Position", "gene_id", "gene_start", "gene_end", "REF", "ALT"], as_index=False)
            .agg({"qtl_trait": self._merge_values, "gwas_trait": self._merge_values})
            .assign(_chrom_order=lambda x: x["Chromosome"].map(self._chrom_sort_key))
            .sort_values(["_chrom_order", "Position", "gene_id"])
            .drop(columns="_chrom_order")
            .reset_index(drop=True)
        )
        return out_df

    def _filter_gene_subset(self, site_gene_df: pd.DataFrame) -> pd.DataFrame:
        gene_set = self._load_gene_filter()
        if gene_set is None or site_gene_df.empty:
            return site_gene_df
        out_df = site_gene_df[site_gene_df["gene_id"].astype(str).isin(gene_set)].copy()
        self.logger.info(f"Applied gene subset filter: genes={len(gene_set)} rows={len(out_df)}")
        return out_df.reset_index(drop=True)

    def _save_gene_sites(self, site_gene_df: pd.DataFrame) -> Path:
        out_path = self.outdir / "sites_in_gene.csv"
        site_gene_df.loc[:, ["Chromosome", "Position", "qtl_trait", "gwas_trait", "gene_id"]].to_csv(out_path, index=False)
        self.logger.info(f"Saved gene-mapped sites: {out_path} shape={site_gene_df.shape}")
        return out_path

    def _extract_genotypes(self, final_site_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            import pysam  # type: ignore
        except Exception as exc:
            raise ImportError("pysam is required for genotype extraction") from exc

        site_df = final_site_df.drop_duplicates(subset=["Chromosome", "Position"]).reset_index(drop=True)
        target_sites = {(str(r.Chromosome), int(r.Position)) for r in site_df.itertuples(index=False)}
        genotype_map: dict[tuple[str, int], dict[str, int]] = {}

        with pysam.VariantFile(str(self.vcf_path)) as vcf:
            raw_samples = list(vcf.header.samples)
            self._set_sample_metadata([str(sample) for sample in raw_samples])
            for rec in self._progress(vcf, desc="Extracting genotypes", unit="site"):
                chrom = standard_chrom(str(rec.chrom))
                if chrom is None:
                    continue
                pos = int(rec.pos)
                key = (chrom, pos)
                if key not in target_sites:
                    continue
                genotype_map[key] = {
                    self.sample_map[str(sample)]: self._encode_gt(rec.samples[sample].get("GT"))
                    for sample in raw_samples
                }

        site_ids = [f"{chrom}:{pos}" for chrom, pos in site_df[["Chromosome", "Position"]].itertuples(index=False, name=None)]
        site_by_sample = pd.DataFrame(
            [
                {
                    "Chromosome": str(row.Chromosome),
                    "Position": int(row.Position),
                    "site_id": f"{row.Chromosome}:{row.Position}",
                    **genotype_map.get((str(row.Chromosome), int(row.Position)), {}),
                }
                for row in site_df.itertuples(index=False)
            ]
        )
        for sample in self.sample_list:
            if sample not in site_by_sample.columns:
                site_by_sample[sample] = 0
        site_by_sample = site_by_sample.loc[:, ["Chromosome", "Position", "site_id"] + self.sample_list]
        sample_by_site = pd.DataFrame(
            np.asarray(site_by_sample[self.sample_list].to_numpy(dtype=np.int8)).T,
            index=self.sample_list,
            columns=site_ids,
            dtype=np.int8,
        )
        sample_by_site.index.name = "sample"

        site_by_sample.to_csv(self.outdir / "genotype_012_site_by_sample.csv", index=False)
        sample_df = sample_by_site.reset_index()
        sample_df.to_csv(self.outdir / "genotype_012.csv", index=False)
        sample_df.to_parquet(self.outdir / "genotype_012.parquet", index=False)
        self._save_sample_mapping()
        self.logger.info(f"Saved genotype matrices: sites={len(site_df)} samples={len(self.sample_list)}")
        return site_by_sample, sample_by_site

    def _compute_variant_effects(self, site_df: pd.DataFrame) -> pd.DataFrame:
        try:
            import pysam  # type: ignore
        except Exception as exc:
            raise ImportError("pysam is required for FASTA processing") from exc

        rows: list[dict[str, str | int | float]] = []
        with pysam.FastaFile(str(self.fasta_path)) as fasta:
            chrom_map = self._build_fasta_chrom_map(fasta)
            for row in self._progress(
                site_df.itertuples(index=False),
                total=len(site_df),
                desc="Calculating variant effects",
                unit="site",
            ):
                chrom = str(row.Chromosome)
                fasta_chrom = chrom_map.get(chrom)
                if fasta_chrom is None:
                    continue
                ref = str(row.REF).upper()
                alt = str(row.ALT).upper()
                pos = int(row.Position)
                upstream = self._fetch_window_with_padding(fasta, fasta_chrom, pos - self.k, pos - 1)
                downstream = self._fetch_window_with_padding(
                    fasta,
                    fasta_chrom,
                    pos + len(ref),
                    pos + len(ref) + self.k - 1,
                )
                seq_ref = upstream + ref + downstream
                seq_alt = upstream + alt + downstream
                emb_ref = self._embed_sequence(seq_ref)
                emb_alt = self._embed_sequence(seq_alt)
                rows.append(
                    {
                        "Chromosome": chrom,
                        "Position": pos,
                        "site_id": f"{chrom}:{pos}",
                        "var_effect": float(np.linalg.norm(emb_ref - emb_alt)),
                    }
                )
        return pd.DataFrame(rows, columns=["Chromosome", "Position", "site_id", "var_effect"])

    def _save_variant_effect_matrix(
        self,
        sample_by_site: pd.DataFrame,
        variant_effect_df: pd.DataFrame,
    ) -> tuple[Path, Path, pd.DataFrame]:
        effect_map = dict(zip(variant_effect_df["site_id"], variant_effect_df["var_effect"]))
        matrix = sample_by_site.copy()
        for col in matrix.columns:
            matrix[col] = matrix[col].astype(np.float32) * float(effect_map.get(col, 0.0))
        out_df = matrix.reset_index()
        csv_path = self.outdir / "variant_effect_matrix.csv"
        parquet_path = self.outdir / "variant_effect_matrix.parquet"
        out_df.to_csv(csv_path, index=False)
        out_df.to_parquet(parquet_path, index=False)
        self.logger.info(f"Saved variant effect matrix: {csv_path} shape={out_df.shape}")
        return csv_path, parquet_path, out_df

    def _apply_sample_variants_to_gene(
        self,
        reference_seq: str,
        gene_start: int,
        variant_rows: list[dict[str, str | int]],
        sample: str,
    ) -> str:
        parts: list[str] = []
        cursor = 0
        rows = sorted(variant_rows, key=lambda x: int(x["Position"]))
        for row in rows:
            if int(row[sample]) <= 0:
                continue
            pos = int(row["Position"])
            ref = str(row["REF"])
            alt = str(row["ALT"])
            rel_start = pos - gene_start
            rel_end = rel_start + len(ref)
            if rel_start < 0 or rel_end > len(reference_seq):
                continue
            if reference_seq[rel_start:rel_end].upper() != ref.upper():
                continue
            if rel_start < cursor:
                continue
            parts.append(reference_seq[cursor:rel_start])
            parts.append(alt)
            cursor = rel_end
        parts.append(reference_seq[cursor:])
        return "".join(parts)

    def _build_gene_feature_matrix(
        self,
        final_site_df: pd.DataFrame,
        site_by_sample: pd.DataFrame,
    ) -> tuple[Path, Path, pd.DataFrame]:
        try:
            import pysam  # type: ignore
        except Exception as exc:
            raise ImportError("pysam is required for FASTA processing") from exc

        sample_columns = [c for c in site_by_sample.columns if c not in {"Chromosome", "Position", "site_id"}]
        geno_lookup = site_by_sample.set_index("site_id")[sample_columns]
        gene_df = final_site_df.copy()
        gene_df["site_id"] = gene_df["Chromosome"].astype(str) + ":" + gene_df["Position"].astype(str)
        gene_feature_rows: dict[str, list[float]] = {sample: [] for sample in sample_columns}
        gene_columns: list[str] = []

        with pysam.FastaFile(str(self.fasta_path)) as fasta:
            chrom_map = self._build_fasta_chrom_map(fasta)
            gene_groups = list(gene_df.groupby("gene_id", sort=True))
            for gene_id, sub_df in self._progress(
                gene_groups,
                total=len(gene_groups),
                desc="Building gene sequence features",
                unit="gene",
            ):
                sub_df = sub_df.sort_values(["Position"]).drop_duplicates(["Chromosome", "Position", "gene_id"]).reset_index(drop=True)
                chrom = str(sub_df["Chromosome"].iloc[0])
                fasta_chrom = chrom_map.get(chrom)
                if fasta_chrom is None:
                    continue
                gene_start = int(sub_df["gene_start"].iloc[0])
                gene_end = int(sub_df["gene_end"].iloc[0])
                reference_seq = fasta.fetch(fasta_chrom, gene_start - 1, gene_end).upper()
                ref_emb = self._embed_sequence(reference_seq)

                variant_rows: list[dict[str, str | int]] = []
                for row in sub_df.itertuples(index=False):
                    site_id = f"{row.Chromosome}:{row.Position}"
                    geno_values = geno_lookup.loc[site_id].to_dict() if site_id in geno_lookup.index else {}
                    variant_rows.append(
                        {
                            "Chromosome": str(row.Chromosome),
                            "Position": int(row.Position),
                            "REF": str(row.REF),
                            "ALT": str(row.ALT),
                            **{sample: int(geno_values.get(sample, 0)) for sample in sample_columns},
                        }
                    )

                gene_columns.append(str(gene_id))
                for sample in sample_columns:
                    mutated_seq = self._apply_sample_variants_to_gene(reference_seq, gene_start, variant_rows, sample)
                    if mutated_seq == reference_seq:
                        gene_feature_rows[sample].append(0.0)
                        continue
                    alt_emb = self._embed_sequence(mutated_seq)
                    gene_feature_rows[sample].append(float(np.linalg.norm(ref_emb - alt_emb)))

        out_df = pd.DataFrame(gene_feature_rows, index=gene_columns).T.reset_index().rename(columns={"index": "sample"})
        csv_path = self.outdir / "gene_sequence_feature_matrix.csv"
        parquet_path = self.outdir / "gene_sequence_feature_matrix.parquet"
        out_df.to_csv(csv_path, index=False)
        out_df.to_parquet(parquet_path, index=False)
        self.logger.info(f"Saved gene feature matrix: {csv_path} shape={out_df.shape}")
        return csv_path, parquet_path, out_df

    def _split_sample_datasets(
        self,
        genotype_df: pd.DataFrame,
        variant_effect_df: pd.DataFrame,
        gene_feature_df: pd.DataFrame,
    ) -> dict[str, str]:
        if self.split_test_ratio is None:
            return {}

        split_outdir = self.outdir / "splits"
        split_outdir.mkdir(parents=True, exist_ok=True)
        train_val_samples, test_samples = self._build_split_sample_lists()

        train_val_genotype = self._subset_by_samples(genotype_df, train_val_samples)
        test_genotype = self._subset_by_samples(genotype_df, test_samples)
        train_val_variant_effect = self._subset_by_samples(variant_effect_df, train_val_samples)
        test_variant_effect = self._subset_by_samples(variant_effect_df, test_samples)
        train_val_gene_feature = self._subset_by_samples(gene_feature_df, train_val_samples)
        test_gene_feature = self._subset_by_samples(gene_feature_df, test_samples)

        geno_train_csv, geno_train_parquet = self._save_dual(train_val_genotype, split_outdir / "train_val_genotype_012")
        geno_test_csv, geno_test_parquet = self._save_dual(test_genotype, split_outdir / "test_genotype_012")
        ve_train_csv, ve_train_parquet = self._save_dual(train_val_variant_effect, split_outdir / "train_val_variant_effect_matrix")
        ve_test_csv, ve_test_parquet = self._save_dual(test_variant_effect, split_outdir / "test_variant_effect_matrix")
        gf_train_csv, gf_train_parquet = self._save_dual(train_val_gene_feature, split_outdir / "train_val_gene_sequence_feature_matrix")
        gf_test_csv, gf_test_parquet = self._save_dual(test_gene_feature, split_outdir / "test_gene_sequence_feature_matrix")

        meta = {
            "split_test_ratio": float(self.split_test_ratio),
            "split_random_state": self.split_random_state,
            "isolated_samples": self._read_isolated_samples(),
            "samples": {
                "train_val": train_val_samples,
                "test": test_samples,
            },
        }
        meta_path = split_outdir / "split_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info(
            f"Saved split datasets: train_val_samples={len(train_val_samples)} test_samples={len(test_samples)} outdir={split_outdir}"
        )
        return {
            "split_meta_json": str(meta_path),
            "train_val_genotype_csv": str(geno_train_csv),
            "train_val_genotype_parquet": str(geno_train_parquet),
            "test_genotype_csv": str(geno_test_csv),
            "test_genotype_parquet": str(geno_test_parquet),
            "train_val_variant_effect_csv": str(ve_train_csv),
            "train_val_variant_effect_parquet": str(ve_train_parquet),
            "test_variant_effect_csv": str(ve_test_csv),
            "test_variant_effect_parquet": str(ve_test_parquet),
            "train_val_gene_feature_csv": str(gf_train_csv),
            "train_val_gene_feature_parquet": str(gf_train_parquet),
            "test_gene_feature_csv": str(gf_test_csv),
            "test_gene_feature_parquet": str(gf_test_parquet),
        }

    def run(self) -> dict[str, str]:
        self.logger.info("Pipeline started")
        gwas_df = self._read_gwas()
        qtl_df = self._read_qtl()
        gwas_site_map = self._build_gwas_site_map(gwas_df)
        qtl_trees = self._build_qtl_trees(qtl_df)

        site_df = self._select_sites_from_vcf(gwas_site_map, qtl_trees)
        if site_df.empty:
            raise ValueError("No sites remained after GWAS/QTL filtering")
        initial_site_path = self._save_initial_sites(site_df)

        gene_trees = self._build_gene_trees()
        site_gene_df = self._map_sites_to_genes(site_df, gene_trees)
        site_gene_df = self._filter_gene_subset(site_gene_df)
        if site_gene_df.empty:
            raise ValueError("No sites remained after gene interval filtering")
        gene_site_path = self._save_gene_sites(site_gene_df)

        site_by_sample, sample_by_site = self._extract_genotypes(site_gene_df)
        genotype_df = sample_by_site.reset_index()
        unique_site_df = site_gene_df.drop_duplicates(subset=["Chromosome", "Position"]).reset_index(drop=True)
        variant_effect_df = self._compute_variant_effects(unique_site_df)
        variant_effect_csv, variant_effect_parquet, variant_effect_matrix_df = self._save_variant_effect_matrix(sample_by_site, variant_effect_df)
        gene_feature_csv, gene_feature_parquet, gene_feature_df = self._build_gene_feature_matrix(site_gene_df, site_by_sample)

        outputs = {
            "initial_site_csv": str(initial_site_path),
            "gene_site_csv": str(gene_site_path),
            "genotype_site_by_sample_csv": str(self.outdir / "genotype_012_site_by_sample.csv"),
            "genotype_sample_by_site_csv": str(self.outdir / "genotype_012.csv"),
            "genotype_sample_by_site_parquet": str(self.outdir / "genotype_012.parquet"),
            "variant_effect_csv": str(variant_effect_csv),
            "variant_effect_parquet": str(variant_effect_parquet),
            "gene_feature_csv": str(gene_feature_csv),
            "gene_feature_parquet": str(gene_feature_parquet),
        }
        outputs.update(
            self._split_sample_datasets(
                genotype_df=genotype_df,
                variant_effect_df=variant_effect_matrix_df,
                gene_feature_df=gene_feature_df,
            )
        )
        self.logger.info(f"Pipeline finished: {json.dumps(outputs, ensure_ascii=False)}")
        return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build filtered site, genotype, variant-effect, and gene-feature datasets")
    parser.add_argument("--config", required=True, help="YAML config file path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import yaml
    except Exception as exc:
        raise ImportError("PyYAML is required to load the config file") from exc

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    outputs = SiteDatasetBuilder(config).run()
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
