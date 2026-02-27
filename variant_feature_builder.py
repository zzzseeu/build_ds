from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Type

import numpy as np
import pandas as pd


def _build_embedder_registry() -> Dict[str, Type]:
    """Return the mapping from embedder name to embedder implementation class.

    Returns
    -------
    Dict[str, Type]
        Registry used by ``load_pretrained``. Keys are lowercase embedder names.
    """
    from embedder import (
        AgrontEmbedder,
        Evo2Embedder,
        GENERatorEmbedder,
        NucleotideTransformerEmbedder,
        Rice8kEmbedder,
    )

    return {
        "generator": GENERatorEmbedder,
        "evo2": Evo2Embedder,
        "nt": NucleotideTransformerEmbedder,
        "agront": AgrontEmbedder,
        "rice8k": Rice8kEmbedder,
    }


def load_pretrained(embedder_type: str, **kwargs):
    """Instantiate a concrete embedder by name.

    Parameters
    ----------
    embedder_type : str
        Embedder key in the registry, e.g. ``rice8k``.
    **kwargs
        Passed through to the concrete embedder constructor.

    Returns
    -------
    object
        Concrete embedder instance exposing ``embed(sequence)``.

    Raises
    ------
    ValueError
        If ``embedder_type`` is not registered.
    """
    registry = _build_embedder_registry()
    embedder_type = embedder_type.lower()
    if embedder_type not in registry:
        raise ValueError(
            f"Unknown embedder type: {embedder_type}. Available: {list(registry.keys())}"
        )
    return registry[embedder_type](**kwargs)


class UnifiedEmbedder:
    """Lightweight callable wrapper over registry embedders.

    This wrapper normalizes usage to ``embedder(sequence)``.
    """

    def __init__(self, embedder_type: str, **kwargs):
        """Create a unified embedder instance.

        Parameters
        ----------
        embedder_type : str
            Embedder type key accepted by :func:`load_pretrained`.
        **kwargs
            Forwarded to the concrete embedder constructor.
        """
        self.embedder = load_pretrained(embedder_type, **kwargs)
        self.embedder_type = embedder_type

    def __call__(self, sequence: str | List[str]):
        """Compute embedding for one sequence or a batch of sequences."""
        return self.embedder.embed(sequence)


class VariantFeatureBuilder:
    """
    End-to-end builder for variant-derived machine-learning feature datasets.

    The class consumes:
    - site dataframe from selector step,
    - VCF with sample genotypes,
    - reference FASTA.

    It then generates multiple dataset views:
    - ``genotype_012`` (sample x site dosage),
    - gene-mutated sequence table + per-gene embedding PCA features,
    - concatenated sequence embedding features,
    - per-site extended-context embedding PCA features,
    - per-site ``distance(ref,alt) * dosage`` features,
    - train/val/test sample splits.

    Coordinate convention is 1-based throughout public inputs/outputs.
    """

    def __init__(
        self,
        variant_df_path: str,
        vcf_path: str,
        fasta_path: str,
        outdir: str,
        embedder: Optional[Callable[[Sequence[str] | str], np.ndarray]] = None,
        isolated_sample: Optional[List[str]] = None,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        random_seed: int = 42,
        flank_k: int = 20,
        pca_var_threshold: float = 0.95,
        save_file: bool = True,
        file_format: str = "csv",
        log_level: int = logging.INFO,
        log_file: str = "variant_feature_builder.log",
    ) -> None:
        """Initialize feature builder.

        Parameters
        ----------
        variant_df_path : str
            CSV path produced by VariantSelector-like step. Requires
            ``Chromosome, Position, Gene_id, Gene_position``.
        vcf_path : str
            Input VCF containing variant and genotype information.
        fasta_path : str
            Reference genome FASTA path.
        outdir : str
            Directory for generated outputs and embedding cache.
        embedder : callable, optional
            Embedder callable returning vector(s). Required for embedding-based features.
        isolated_sample : list[str], optional
            Samples forced into test split.
        test_ratio : float
            Target fraction for test split.
        val_ratio : float
            Target fraction of remaining samples for validation split.
        random_seed : int
            Random seed used by split sampling.
        flank_k : int
            Upstream/downstream context length for site context sequence.
        pca_var_threshold : float
            Cumulative explained variance threshold for PCA component selection.
        save_file : bool
            Whether to persist generated dataframes.
        file_format : str
            Output table format: ``csv`` or ``parquet``.
        log_level : int
            Python logging level, e.g. ``logging.INFO``.
        log_file : str
            Log file name written under ``outdir``.
        """
        self.variant_df_path = Path(variant_df_path)
        self.vcf_path = Path(vcf_path)
        self.fasta_path = Path(fasta_path)
        self.outdir = Path(outdir)
        self.embedder = embedder

        self.isolated_sample = set(isolated_sample or [])
        self.test_ratio = float(test_ratio)
        self.val_ratio = float(val_ratio)
        self.random_seed = int(random_seed)
        self.flank_k = int(flank_k)
        self.pca_var_threshold = float(pca_var_threshold)
        self.save_file = bool(save_file)
        self.file_format = file_format.lower().strip()
        if self.file_format not in {"csv", "parquet"}:
            raise ValueError("file_format must be 'csv' or 'parquet'")

        self.outdir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.outdir / "embedding_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_file

        self.logger = logging.getLogger("VariantFeatureBuilder")
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        if not any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            for h in self.logger.handlers
        ):
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)
        file_path = self.outdir / self.log_file
        if not any(
            isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == file_path
            for h in self.logger.handlers
        ):
            fh = logging.FileHandler(file_path)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        self.variant_df: Optional[pd.DataFrame] = None
        self.samples: List[str] = []
        self.site_order: List[Tuple[str, int]] = []
        self.record_order: List[Tuple[str, int, str]] = []
        self.site_meta: Dict[Tuple[str, int], Dict[str, object]] = {}
        self.genotype012_df: Optional[pd.DataFrame] = None
        self.logger.info("Initialized VariantFeatureBuilder")
        self.logger.info(
            "config: test_ratio=%.3f val_ratio=%.3f flank_k=%d pca_var_threshold=%.3f save_file=%s format=%s",
            self.test_ratio,
            self.val_ratio,
            self.flank_k,
            self.pca_var_threshold,
            self.save_file,
            self.file_format,
        )

    # --------------------------
    # Main pipeline
    # --------------------------
    def run(self) -> Dict[str, pd.DataFrame]:
        """Run full pipeline and optionally save all generated datasets.

        Returns
        -------
        Dict[str, pd.DataFrame]
            All output datasets keyed by dataset name.
        """
        self.logger.info("Pipeline started")
        self.variant_df = self._load_variant_df()
        self._load_vcf_genotypes()

        ds012 = self.build_genotype_012_matrix()
        gene_seq_df, gene_pc_df = self.build_gene_sequence_pca_dataset()
        concat_embed_df = self.build_concat_altseq_embedding_dataset()
        extseq_raw_df, extseq_pc_df = self.build_extseq_embedding_pca_dataset()
        distance_gt_df = self.build_distance_times_genotype_dataset()

        splits = self.build_splits(ds012["sample"].tolist())

        outputs: Dict[str, pd.DataFrame] = {
            "genotype_012": ds012,
            "gene_sequence": gene_seq_df,
            "gene_pca": gene_pc_df,
            "concat_embedding": concat_embed_df,
            "extseq_raw": extseq_raw_df,
            "extseq_pca": extseq_pc_df,
            "distance_x_gt": distance_gt_df,
            "split_train": pd.DataFrame({"sample": splits["train"]}),
            "split_val": pd.DataFrame({"sample": splits["val"]}),
            "split_test": pd.DataFrame({"sample": splits["test"]}),
        }

        if self.save_file:
            for name, df in outputs.items():
                self._save_df(df, name)
            self._save_meta(splits)
        for name, df in outputs.items():
            self._log_df_info(f"output[{name}]", df)
        self.logger.info("Pipeline finished")

        return outputs

    # --------------------------
    # Input loading
    # --------------------------
    def _load_variant_df(self) -> pd.DataFrame:
        """Load and validate input site dataframe."""
        df = pd.read_csv(self.variant_df_path)
        required = ["Chromosome", "Position", "Gene_id", "Gene_position"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Variant dataframe missing columns: {missing}")

        df = df[required + [c for c in df.columns if c not in required]].copy()
        df["Chromosome"] = df["Chromosome"].astype(str)
        df["Position"] = df["Position"].astype(int)
        df["Gene_id"] = df["Gene_id"].astype(str)
        df["Gene_position"] = df["Gene_position"].astype(int)
        self.logger.info(
            "Loaded variant_df: shape=%s unique_sites=%d unique_genes=%d",
            df.shape,
            df[["Chromosome", "Position"]].drop_duplicates().shape[0],
            df["Gene_id"].nunique(),
        )
        self._log_df_info("variant_df", df)
        return df

    def _load_vcf_genotypes(self) -> None:
        """Load VCF and extract site metadata + sample 0/1/2 dosages.

        Notes
        -----
        - Coordinates are interpreted as 1-based.
        - Only first ALT allele is used.
        - Genotypes not in {0,1} are treated as reference dosage 0.
        """
        try:
            import pysam  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("pysam is required. Please install pysam.") from exc

        assert self.variant_df is not None

        target_sites: Set[Tuple[str, int]] = set(
            zip(self.variant_df["Chromosome"], self.variant_df["Position"])
        )

        vcf = pysam.VariantFile(str(self.vcf_path))
        self.samples = list(vcf.header.samples)

        site_meta: Dict[Tuple[str, int], Dict[str, object]] = {}
        for rec in vcf:
            key = (str(rec.chrom), int(rec.pos))
            if key not in target_sites:
                continue

            ref = str(rec.ref)
            alt = str(rec.alts[0]) if rec.alts else None
            if not alt:
                continue

            gts: List[int] = []
            for sample in self.samples:
                gt = rec.samples[sample].get("GT")
                if gt is None or any(a is None for a in gt):
                    dosage = 0
                elif any(a not in (0, 1) for a in gt):
                    dosage = 0
                else:
                    dosage = int(sum(gt))
                    if dosage > 2:
                        dosage = 2
                gts.append(dosage)

            site_meta[key] = {
                "Chromosome": key[0],
                "Position": key[1],
                "Ref": ref,
                "Alt": alt,
                "dosage": np.array(gts, dtype=np.float32),
            }

        self.site_order = sorted(site_meta.keys(), key=lambda x: (x[0], x[1]))
        self.site_meta = site_meta

        if not self.site_order:
            raise ValueError("No matched VCF sites found for input variant dataframe.")

        self.record_order = self._build_record_order()
        self.logger.info(
            "Loaded VCF genotypes: samples=%d matched_sites=%d matched_records=%d",
            len(self.samples),
            len(self.site_order),
            len(self.record_order),
        )
        self._log_dict_info("site_meta", self.site_meta)

    # --------------------------
    # Step 1: 012
    # --------------------------
    def build_genotype_012_matrix(self) -> pd.DataFrame:
        """Build sample x site dosage matrix with 0/1/2 encoding."""
        cols = [self._record_feature_name(r) for r in self.record_order]
        mat = np.vstack([self.site_meta[(c, p)]["dosage"] for c, p, _ in self.record_order]).T
        df = pd.DataFrame(mat, columns=cols)
        df.insert(0, "sample", self.samples)
        self.genotype012_df = df
        self._log_df_info("genotype_012", df)
        return df

    # --------------------------
    # Step 2: gene sequence + embedding PCA
    # --------------------------
    def build_gene_sequence_pca_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build gene sequence table and per-gene PCA embedding feature table.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            ``(gene_sequence_df, gene_pca_df)``.
        """
        self._check_embedder()
        gene_map = self._build_gene_site_map()
        self._log_dict_info("gene_map", gene_map)

        ref_gene_seq = self._fetch_reference_gene_sequences(gene_map)
        self._log_dict_info("ref_gene_seq", ref_gene_seq)
        n_samples = len(self.samples)
        self.logger.info(
            "Building gene sequence/PCA dataset: genes=%d samples=%d",
            len(gene_map),
            n_samples,
        )

        # sample x gene sequence table
        gene_seq_data: Dict[str, List[str]] = {}
        for gi, (gene_id, ginfo) in enumerate(gene_map.items(), start=1):
            ref_seq = ref_gene_seq[gene_id]
            ref_arr = np.frombuffer(ref_seq.encode("ascii"), dtype=np.uint8)
            seq_mat = np.tile(ref_arr, (n_samples, 1))

            sites = ginfo["sites"]
            site_pos = np.array([int(ginfo["site_to_gene_pos"][s]) - 1 for s in sites], dtype=np.int64)
            for sj, site in enumerate(sites):
                idx = int(site_pos[sj])
                if idx < 0 or idx >= seq_mat.shape[1]:
                    continue
                mask = self.site_meta[site]["dosage"] > 0
                if not np.any(mask):
                    continue
                alt_char = str(self.site_meta[site]["Alt"])[0]
                seq_mat[mask, idx] = ord(alt_char)

            gene_seq_data[gene_id] = [row.tobytes().decode("ascii") for row in seq_mat]
            if gi % 10 == 0 or gi == len(gene_map):
                self.logger.info("Gene mutation sequence progress: %d/%d", gi, len(gene_map))

        gene_seq_df = pd.DataFrame({"sample": self.samples, **gene_seq_data})
        self._log_df_info("gene_sequence", gene_seq_df)

        # Embed per gene and PCA per gene
        blocks = []
        columns = ["sample"]
        for gi, gene_id in enumerate(gene_map.keys(), start=1):
            seqs = gene_seq_df[gene_id].astype(str).tolist()
            emb = self._embed_sequences(seqs)  # n x d
            comp = self._pca_reduce(emb, self.pca_var_threshold)
            blocks.append(comp)
            for k in range(comp.shape[1]):
                columns.append(f"{gene_id}_PC{k+1}")
            if gi % 10 == 0 or gi == len(gene_map):
                self.logger.info("Gene embedding/PCA progress: %d/%d", gi, len(gene_map))

        out = pd.DataFrame(np.concatenate(blocks, axis=1), columns=columns[1:])
        out.insert(0, "sample", self.samples)
        self._log_df_info("gene_pca", out)
        return gene_seq_df, out

    # --------------------------
    # Step 3: concat seq embedding
    # --------------------------
    def build_concat_altseq_embedding_dataset(self) -> pd.DataFrame:
        """Build sample-level concatenated sequence embedding feature table."""
        self._check_embedder()

        sample_seqs: List[str] = []
        for si, _sample in enumerate(self.samples):
            bases: List[str] = []
            for site in self.site_order:
                meta = self.site_meta[site]
                dosage = int(meta["dosage"][si])
                bases.append(str(meta["Alt"]) if dosage > 0 else str(meta["Ref"]))
            sample_seqs.append("".join(bases))

        emb = self._embed_sequences(sample_seqs)
        cols = [f"concat_embed_{i}" for i in range(emb.shape[1])]
        df = pd.DataFrame(emb, columns=cols)
        df.insert(0, "sample", self.samples)
        self._log_df_info("concat_embedding", df)
        return df

    # --------------------------
    # Step 4: extended seq embedding x 012 + PCA
    # --------------------------
    def build_extseq_embedding_pca_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build extended-context sequence table and per-site PCA feature table.

        For each site, sequence embedding is multiplied by dosage before PCA.
        """
        self._check_embedder()
        contexts = self._build_site_context_sequences(self.flank_k)
        self._log_dict_info("site_contexts", contexts)
        self.logger.info(
            "Building extended-sequence datasets: sites=%d samples=%d flank_k=%d",
            len(self.site_order),
            len(self.samples),
            self.flank_k,
        )

        # Raw sequence table (sample x site)
        site_cols = [self._record_feature_name(r) for r in self.record_order]
        raw_rows: List[Dict[str, object]] = []
        for si, sample in enumerate(self.samples):
            row = {"sample": sample}
            for rec, col in zip(self.record_order, site_cols):
                site = (rec[0], rec[1])
                dosage = int(self.site_meta[site]["dosage"][si])
                row[col] = contexts[site]["alt"] if dosage > 0 else contexts[site]["ref"]
            raw_rows.append(row)
        raw_df = pd.DataFrame(raw_rows)
        self._log_df_info("extseq_raw", raw_df)

        # Per record: embed sample sequences, multiply by dosage, PCA.
        # If multiple records share one site, reuse site PCA result.
        blocks = []
        out_cols = ["sample"]
        site_comp_cache: Dict[Tuple[str, int], np.ndarray] = {}
        for si, rec in enumerate(self.record_order, start=1):
            site = (rec[0], rec[1])
            base_name = self._record_feature_name(rec)
            seqs = raw_df[base_name].astype(str).tolist()
            if site not in site_comp_cache:
                emb = self._embed_sequences(seqs)
                dosage = self.site_meta[site]["dosage"].reshape(-1, 1)
                weighted = emb * dosage
                site_comp_cache[site] = self._pca_reduce(weighted, self.pca_var_threshold)
            comp = site_comp_cache[site]
            blocks.append(comp)
            for k in range(comp.shape[1]):
                out_cols.append(f"{base_name}_PC{k+1}")
            if si % 500 == 0 or si == len(self.record_order):
                self.logger.info("Extended-sequence PCA progress: %d/%d", si, len(self.record_order))

        out = pd.DataFrame(np.concatenate(blocks, axis=1), columns=out_cols[1:])
        out.insert(0, "sample", self.samples)
        self._log_df_info("extseq_pca", out)
        return raw_df, out

    # --------------------------
    # Step 5: distance(ref,alt) x 012
    # --------------------------
    def build_distance_times_genotype_dataset(self) -> pd.DataFrame:
        """Build per-site ``distance(ref, alt) * dosage`` feature table."""
        self._check_embedder()
        contexts = self._build_site_context_sequences(self.flank_k)
        self.logger.info("Building distance*dosage dataset: sites=%d samples=%d", len(self.site_order), len(self.samples))

        site_dist: Dict[Tuple[str, int], float] = {}
        for si, site in enumerate(self.site_order, start=1):
            ref_vec = self._embed_sequences([contexts[site]["ref"]])[0]
            alt_vec = self._embed_sequences([contexts[site]["alt"]])[0]
            site_dist[site] = float(np.linalg.norm(ref_vec - alt_vec, ord=2))
            if si % 500 == 0 or si == len(self.site_order):
                self.logger.info("Distance precompute progress: %d/%d", si, len(self.site_order))
        self._log_dict_info("site_dist", site_dist)

        rows = []
        for si, sample in enumerate(self.samples):
            row: Dict[str, object] = {"sample": sample}
            for rec in self.record_order:
                site = (rec[0], rec[1])
                dist = site_dist[site]
                dosage = float(self.site_meta[site]["dosage"][si])
                base_name = self._record_feature_name(rec)
                row[base_name] = dist * dosage
            rows.append(row)

        out = pd.DataFrame(rows)
        self._log_df_info("distance_x_gt", out)
        return out

    # --------------------------
    # Step 6: split
    # --------------------------
    def build_splits(self, samples: List[str]) -> Dict[str, List[str]]:
        """Split samples into train/val/test with isolated-sample constraint.

        Parameters
        ----------
        samples : List[str]
            All sample IDs.

        Returns
        -------
        Dict[str, List[str]]
            Keys: ``train``, ``val``, ``test``.
        """
        n = len(samples)
        if n == 0:
            return {"train": [], "val": [], "test": []}

        rng = np.random.default_rng(self.random_seed)

        isolated = [s for s in samples if s in self.isolated_sample]
        remaining = [s for s in samples if s not in self.isolated_sample]

        test_n = max(1, int(round(n * self.test_ratio)))
        if len(isolated) > test_n:
            test_n = len(isolated)

        remain_pick = max(0, test_n - len(isolated))
        if remain_pick > len(remaining):
            remain_pick = len(remaining)

        if remain_pick > 0:
            extra_test = list(rng.choice(np.array(remaining), size=remain_pick, replace=False))
        else:
            extra_test = []

        test = isolated + extra_test
        remain2 = [s for s in samples if s not in set(test)]

        val_n = int(round(len(remain2) * self.val_ratio))
        if val_n > 0:
            val = list(rng.choice(np.array(remain2), size=val_n, replace=False))
        else:
            val = []
        train = [s for s in remain2 if s not in set(val)]

        result = {"train": train, "val": val, "test": test}
        self.logger.info(
            "Built splits: train=%d val=%d test=%d isolated_in_test=%d",
            len(result["train"]),
            len(result["val"]),
            len(result["test"]),
            len([s for s in result["test"] if s in self.isolated_sample]),
        )
        return result

    # --------------------------
    # Helpers
    # --------------------------
    def _build_gene_site_map(self) -> Dict[str, Dict[str, object]]:
        """Construct per-gene metadata and site mapping for sequence mutation."""
        assert self.variant_df is not None
        variant_sites = set(self.site_order)
        df = self.variant_df[
            self.variant_df.apply(lambda r: (r["Chromosome"], int(r["Position"])) in variant_sites, axis=1)
        ].copy()

        gene_map: Dict[str, Dict[str, object]] = {}
        for gene_id, g in df.groupby("Gene_id"):
            sites = []
            site_to_gene_pos: Dict[Tuple[str, int], int] = {}
            starts = []
            ends = []

            for _, row in g.iterrows():
                site = (str(row["Chromosome"]), int(row["Position"]))
                gene_pos = int(row["Gene_position"])
                sites.append(site)
                site_to_gene_pos[site] = gene_pos
                starts.append(int(row["Position"]) - gene_pos + 1)
                ends.append(int(row["Position"]) - gene_pos + 1 + gene_pos - 1)

            gene_start = int(np.median(starts)) if starts else 1

            if "Gene_start" in g.columns and "Gene_end" in g.columns:
                gene_start = int(g["Gene_start"].iloc[0])
                gene_end = int(g["Gene_end"].iloc[0])
            else:
                max_gene_pos = max(site_to_gene_pos.values()) if site_to_gene_pos else 1
                gene_end = gene_start + max_gene_pos - 1

            gene_map[str(gene_id)] = {
                "chrom": str(g["Chromosome"].iloc[0]),
                "gene_start": gene_start,
                "gene_end": gene_end,
                "sites": sorted(list(set(sites)), key=lambda x: (x[0], x[1])),
                "site_to_gene_pos": site_to_gene_pos,
            }
        self.logger.info("Built gene_site_map: genes=%d", len(gene_map))
        return gene_map

    def _build_record_order(self) -> List[Tuple[str, int, str]]:
        """Build record list preserving one row per (site, gene) mapping."""
        assert self.variant_df is not None
        valid_sites = set(self.site_order)
        g = self.variant_df[
            self.variant_df.apply(lambda r: (str(r["Chromosome"]), int(r["Position"])) in valid_sites, axis=1)
        ][["Chromosome", "Position", "Gene_id"]].drop_duplicates()
        g["Chromosome"] = g["Chromosome"].astype(str)
        g["Position"] = g["Position"].astype(int)
        g["Gene_id"] = g["Gene_id"].astype(str)
        g = g.sort_values(["Chromosome", "Position", "Gene_id"])
        out = [(row["Chromosome"], int(row["Position"]), row["Gene_id"]) for _, row in g.iterrows()]
        self.logger.info("Built record_order: n_records=%d", len(out))
        return out

    def _record_feature_name(self, record: Tuple[str, int, str]) -> str:
        """Format record feature name as ``Chr:Position:Ref>Alt-gene_id``."""
        chrom, pos, gene_id = record
        site = (chrom, pos)
        ref = str(self.site_meta[site]["Ref"])
        alt = str(self.site_meta[site]["Alt"])
        return f"{chrom}:{pos}:{ref}>{alt}-{gene_id}"

    def _fetch_reference_gene_sequences(self, gene_map: Dict[str, Dict[str, object]]) -> Dict[str, str]:
        """Fetch reference gene sequences from FASTA using inferred gene intervals."""
        try:
            import pysam  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("pysam is required. Please install pysam.") from exc

        fa = pysam.FastaFile(str(self.fasta_path))
        out: Dict[str, str] = {}
        for gene_id, info in gene_map.items():
            chrom = str(info["chrom"])
            start = int(info["gene_start"])
            end = int(info["gene_end"])
            seq = fa.fetch(chrom, start - 1, end)
            out[gene_id] = seq
        self.logger.info("Fetched reference gene sequences: genes=%d", len(out))
        return out

    def _build_site_context_sequences(self, k: int) -> Dict[Tuple[str, int], Dict[str, str]]:
        """Build site-centered reference/alternate context sequences.

        Parameters
        ----------
        k : int
            Flanking size on both sides. Final length is ``2*k+1`` for SNVs.
        """
        try:
            import pysam  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("pysam is required. Please install pysam.") from exc

        fa = pysam.FastaFile(str(self.fasta_path))

        contexts: Dict[Tuple[str, int], Dict[str, str]] = {}
        for site in self.site_order:
            chrom, pos = site
            ref = str(self.site_meta[site]["Ref"])
            alt = str(self.site_meta[site]["Alt"])

            left_start = max(1, pos - k)
            left = fa.fetch(chrom, left_start - 1, pos - 1)
            right = fa.fetch(chrom, pos, pos + k)

            ref_seq = f"{left}{ref}{right}"
            alt_seq = f"{left}{alt}{right}"
            contexts[site] = {"ref": ref_seq, "alt": alt_seq}
        self.logger.info("Built site context sequences: sites=%d context_len=%d", len(contexts), 2 * k + 1)
        return contexts

    def _check_embedder(self) -> None:
        """Validate embedder availability for embedding-based datasets."""
        if self.embedder is None:
            raise ValueError("embedder is required for embedding-based datasets")

    def _embed_sequences(self, seqs: List[str]) -> np.ndarray:
        """Embed a sequence list with on-disk cache and batch fallback.

        Parameters
        ----------
        seqs : List[str]
            Input DNA sequences.

        Returns
        -------
        np.ndarray
            Matrix of shape ``(n_sequences, embedding_dim)``.
        """
        # Cache on disk by sequence hash to avoid repeated embedding.
        pending: List[str] = []
        vectors: Dict[str, np.ndarray] = {}

        for s in seqs:
            key = self._seq_hash(s)
            path = self.cache_dir / f"{key}.npy"
            if path.exists():
                vectors[s] = np.load(path)
            else:
                pending.append(s)

        n_total = len(seqs)
        n_hit = n_total - len(pending)
        if pending:
            # Try batch call first, fallback to one-by-one.
            try:
                out = self.embedder(pending)  # type: ignore[misc]
                arr = self._to_2d_numpy(out)
                if arr.shape[0] != len(pending):
                    raise ValueError("Embedding batch size mismatch")
                for s, v in zip(pending, arr):
                    key = self._seq_hash(s)
                    path = self.cache_dir / f"{key}.npy"
                    np.save(path, v)
                    vectors[s] = v
            except Exception:
                for s in pending:
                    out = self.embedder(s)  # type: ignore[misc]
                    v = self._to_1d_numpy(out)
                    key = self._seq_hash(s)
                    path = self.cache_dir / f"{key}.npy"
                    np.save(path, v)
                    vectors[s] = v

        mat = np.vstack([vectors[s] for s in seqs]).astype(np.float32)
        self.logger.debug(
            "Embedding sequences: total=%d cache_hit=%d cache_miss=%d dim=%d",
            n_total,
            n_hit,
            len(pending),
            mat.shape[1],
        )
        return mat

    @staticmethod
    def _pca_reduce(mat: np.ndarray, var_threshold: float) -> np.ndarray:
        """Reduce feature matrix by PCA with explained-variance threshold."""
        from sklearn.decomposition import PCA

        if mat.shape[0] <= 1:
            return mat[:, :1]

        max_comp = min(mat.shape[0], mat.shape[1])
        pca_full = PCA(n_components=max_comp, svd_solver="full")
        pca_full.fit(mat)
        cum = np.cumsum(pca_full.explained_variance_ratio_)
        k = int(np.searchsorted(cum, var_threshold) + 1)
        k = max(1, min(k, max_comp))

        pca = PCA(n_components=k, svd_solver="full")
        return pca.fit_transform(mat).astype(np.float32)

    @staticmethod
    def _seq_hash(seq: str) -> str:
        """Return deterministic SHA1 key for a sequence."""
        return hashlib.sha1(seq.encode("utf-8")).hexdigest()

    @staticmethod
    def _to_1d_numpy(x) -> np.ndarray:
        """Convert tensor/array-like embedding to 1D numpy vector."""
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
            raise ValueError(f"Expected 1D embedding, got shape={arr.shape}")
        return arr

    @staticmethod
    def _to_2d_numpy(x) -> np.ndarray:
        """Convert tensor/array-like embedding to 2D numpy matrix."""
        try:
            import torch

            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().float().numpy()
        except Exception:
            pass

        arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D embedding, got shape={arr.shape}")
        return arr

    def _save_df(self, df: pd.DataFrame, name: str) -> Path:
        """Save one dataframe in configured output format and return path."""
        if self.file_format == "csv":
            p = self.outdir / f"{name}.csv"
            df.to_csv(p, index=False)
        else:
            p = self.outdir / f"{name}.parquet"
            df.to_parquet(p, index=False)
        return p

    def _save_meta(self, splits: Dict[str, List[str]]) -> None:
        """Write run metadata and split assignments to ``meta.json``."""
        meta = {
            "n_samples": len(self.samples),
            "n_sites": len(self.site_order),
            "test_ratio": self.test_ratio,
            "val_ratio": self.val_ratio,
            "isolated_sample": sorted(list(self.isolated_sample)),
            "splits": splits,
        }
        (self.outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    def _log_df_info(self, name: str, df: pd.DataFrame) -> None:
        """Log dataframe dimensional info and key metadata."""
        self.logger.info("%s: shape=%s", name, df.shape)
        if df.shape[1] > 0:
            self.logger.info("%s: n_columns=%d first_columns=%s", name, df.shape[1], list(df.columns[:5]))

    def _log_dict_info(self, name: str, d: Dict) -> None:
        """Log dict size and small key preview for scale tracking."""
        self.logger.info("%s: n_keys=%d", name, len(d))
        if d:
            keys = list(d.keys())[:3]
            self.logger.info("%s: sample_keys=%s", name, keys)


# --------------------------
# CLI
# --------------------------
def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for end-to-end feature generation."""
    p = argparse.ArgumentParser(description="Build variant-derived datasets with optional embedding + PCA")
    p.add_argument("--variant_df_path", required=True, help="Path to VariantSelector output dataframe (csv)")
    p.add_argument("--vcf_path", required=True, help="Path to VCF file")
    p.add_argument("--fasta_path", required=True, help="Path to FASTA file")
    p.add_argument("--outdir", required=True, help="Output directory")

    p.add_argument("--isolated_sample", nargs="*", default=None, help="Samples forced into test split")
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--flank_k", type=int, default=20)
    p.add_argument("--pca_var_threshold", type=float, default=0.95)
    p.add_argument("--file_format", choices=["csv", "parquet"], default="csv")

    p.add_argument("--save_file", action="store_true", default=True)
    p.add_argument("--no-save_file", dest="save_file", action="store_false")

    p.add_argument("--model_name_or_path", default=None, help="When set, build UnifiedEmbedder automatically")
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--use_flash_attention", action="store_true", default=False)
    p.add_argument("--pooling", default="mean")

    return p


def build_embedder_from_args(args):
    """Build ``UnifiedEmbedder`` from CLI arguments.

    Returns ``None`` when model path is not provided.
    """
    if args.model_name_or_path is None:
        return None

    import torch

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    embedder = UnifiedEmbedder(
        "rice8k",
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        torch_dtype=dtype_map[args.torch_dtype],
        use_flash_attention=args.use_flash_attention,
        pooling=args.pooling,
    )
    return embedder


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    embedder = build_embedder_from_args(args)

    builder = VariantFeatureBuilder(
        variant_df_path=args.variant_df_path,
        vcf_path=args.vcf_path,
        fasta_path=args.fasta_path,
        outdir=args.outdir,
        embedder=embedder,
        isolated_sample=args.isolated_sample,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed,
        flank_k=args.flank_k,
        pca_var_threshold=args.pca_var_threshold,
        save_file=args.save_file,
        file_format=args.file_format,
    )

    outputs = builder.run()
    for k, v in outputs.items():
        print(f"{k}: shape={v.shape}")


if __name__ == "__main__":
    main()
