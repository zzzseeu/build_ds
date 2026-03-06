from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from embedding import UnifiedEmbedder


class VariantFeatureBuilder:
    """Build variant-derived datasets from variant positions, VCF and FASTA.

    Inputs are 1-based coordinates.
    """

    def __init__(
        self,
        variant_df_path: str,
        vcf_path: str,
        fasta_path: str,
        outdir: str,
        embedder_type: str,
        model_name_or_path: str,
        pooling: str,
        local_files_only: bool,
        embedder_kwargs: Dict,
        sample_list_path: str | None,
        test_ratio: float,
        val_ratio: float,
        random_seed: int,
        flank_k: int,
        pca_var_threshold: float,
        use_pca: bool,
    ) -> None:
        self.variant_df_path = Path(variant_df_path)
        self.vcf_path = Path(vcf_path)
        self.fasta_path = Path(fasta_path)
        self.outdir = Path(outdir)

        self.embedder_type = embedder_type
        self.model_name_or_path = model_name_or_path
        self.pooling = pooling
        self.local_files_only = bool(local_files_only)
        self.embedder_kwargs = dict(embedder_kwargs or {})
        self.sample_list_path = Path(sample_list_path) if sample_list_path else None

        self.test_ratio = float(test_ratio)
        self.val_ratio = float(val_ratio)
        self.random_seed = int(random_seed)
        self.flank_k = int(flank_k)
        self.pca_var_threshold = float(pca_var_threshold)
        self.use_pca = bool(use_pca)

        self.outdir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.outdir / "embedding_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("VariantFeatureBuilder")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        if not self.logger.handlers:
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            self.logger.addHandler(sh)
            fh = logging.FileHandler(self.outdir / "variant_feature_builder.log")
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

        self.embedder = UnifiedEmbedder(
            embedder_type=self.embedder_type,
            model_name_or_path=self.model_name_or_path,
            device="cuda",
            pooling=self.pooling,
            local_files_only=self.local_files_only,
            **self.embedder_kwargs,
        )

        self.variant_df: pd.DataFrame | None = None
        self.genotype_df: pd.DataFrame | None = None
        self.samples: List[str] = []
        self.isolated_sample_list: List[str] = []

    # --------------------------
    # Pipeline
    # --------------------------
    def run(self) -> Dict[str, pd.DataFrame]:
        self.logger.info("Pipeline started")

        self.variant_df = self._load_variant_df()
        self.genotype_df = self._build_genotype_df()
        self.isolated_sample_list = self._load_isolated_sample_list()
        geno012_df = self._build_geno012_df(self.genotype_df)

        gene_seq_df = self._build_gene_seq_df(self.genotype_df)
        gene_seq_embedding_df = self._embed_sequence_matrix(
            gene_seq_df,
            name_prefix="gene",
            weight_df=None,
        )

        snp_extseq_df = self._build_snp_extseq_df(self.genotype_df)
        snp_weight_df = self._sample_site_genotype_matrix(self.genotype_df)
        snp_extseq_embedding_df = self._embed_sequence_matrix(
            snp_extseq_df,
            name_prefix="snp_ext",
            weight_df=snp_weight_df,
        )

        snp_extseq_distance_df = self._build_snp_extseq_distance_df(self.genotype_df)

        outputs = {
            "geno012_df": geno012_df,
            "genotype_df": self.genotype_df,
            "gene_seq_df": gene_seq_df,
            "gene_seq_embedding_df": gene_seq_embedding_df,
            "snp_extseq_df": snp_extseq_df,
            "snp_extseq_embedding_df": snp_extseq_embedding_df,
            "snp_extseq_distance_df": snp_extseq_distance_df,
        }

        for name, df in outputs.items():
            self._save_both(df, name)
            if "sample" not in df.columns:
                self.logger.info(
                    "Skip split for %s: no 'sample' column (shape=%s)",
                    name,
                    df.shape,
                )
                continue
            split = self._split_df(df)
            self._save_both(split["train"], f"{name}_train")
            self._save_both(split["val"], f"{name}_val")
            self._save_both(split["test"], f"{name}_test")

        self.logger.info("Pipeline finished")
        return outputs

    # --------------------------
    # Step 1
    # --------------------------
    def _load_variant_df(self) -> pd.DataFrame:
        df = pd.read_csv(self.variant_df_path)
        required = ["Chromosome", "Position", "Gene_id", "Gene_position"]
        miss = [c for c in required if c not in df.columns]
        if miss:
            raise ValueError(f"variant_df missing columns: {miss}")

        df = df.copy()
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
        return df

    def _load_isolated_sample_list(self) -> List[str]:
        """Load optional isolated sample list from CSV containing column 'sample'."""
        if self.sample_list_path is None:
            self.logger.info("sample_list_path not provided, use normal split.")
            return []

        sdf = pd.read_csv(self.sample_list_path)
        if "sample" not in sdf.columns:
            raise ValueError("sample_list_path file must contain column: sample")
        samples = sorted(set(sdf["sample"].dropna().astype(str).tolist()))
        self.logger.info("Loaded isolated_sample_list: n=%d", len(samples))
        return samples

    # --------------------------
    # Step 2
    # --------------------------
    def _build_genotype_df(self) -> pd.DataFrame:
        try:
            import pysam  # type: ignore
        except Exception as exc:
            raise ImportError("pysam is required") from exc

        assert self.variant_df is not None
        target_sites = set(zip(self.variant_df["Chromosome"], self.variant_df["Position"]))

        vcf = pysam.VariantFile(str(self.vcf_path))
        self.samples = list(vcf.header.samples)
        self.logger.info("VCF samples=%d", len(self.samples))

        site_rows: Dict[Tuple[str, int], Dict] = {}
        for rec in vcf:
            key = (str(rec.chrom), int(rec.pos))
            if key not in target_sites:
                continue
            ref = str(rec.ref)
            alt = str(rec.alts[0]) if rec.alts else "N"
            row = {
                "Chromosome": key[0],
                "Position": key[1],
                "Ref": ref,
                "Alt": alt,
            }
            for s in self.samples:
                gt = rec.samples[s].get("GT")
                if gt is None or any(a is None for a in gt):
                    g = 0
                elif any(a not in (0, 1) for a in gt):
                    g = 0
                else:
                    g = int(sum(gt))
                    if g > 2:
                        g = 2
                row[s] = g
            site_rows[key] = row

        merged = self.variant_df[["Chromosome", "Position", "Gene_id"]].copy()
        merged = merged.drop_duplicates()
        merged = merged.rename(columns={"Gene_id": "gene_name"})

        gdf = pd.DataFrame(
            [
                {
                    "Chromosome": c,
                    "Position": p,
                    "Ref": site_rows[(c, p)]["Ref"],
                    "Alt": site_rows[(c, p)]["Alt"],
                    **{s: site_rows[(c, p)][s] for s in self.samples},
                }
                for c, p in merged[["Chromosome", "Position"]].itertuples(index=False, name=None)
                if (c, p) in site_rows
            ]
        )
        merged = merged.merge(gdf, on=["Chromosome", "Position"], how="inner")

        out_cols = ["Chromosome", "Position", "Ref", "Alt", "gene_name"] + self.samples
        genotype_df = merged[out_cols].copy()
        self.logger.info("Built genotype_df: shape=%s", genotype_df.shape)
        return genotype_df

    def _build_geno012_df(self, genotype_df: pd.DataFrame) -> pd.DataFrame:
        """Build sample x SNP matrix encoded in 0/1/2 from filtered genotype_df."""
        snp_df = genotype_df[["Chromosome", "Position", "Ref", "Alt"] + self.samples].drop_duplicates(
            subset=["Chromosome", "Position", "Ref", "Alt"]
        )
        snp_names = [
            f"{c}:{int(p)}:{r}>{a}"
            for c, p, r, a in snp_df[["Chromosome", "Position", "Ref", "Alt"]].itertuples(index=False, name=None)
        ]
        mat = snp_df[self.samples].to_numpy(dtype=np.float32).T
        geno012_df = pd.DataFrame(mat, columns=snp_names)
        geno012_df.insert(0, "sample", self.samples)
        self.logger.info("Built geno012_df: shape=%s", geno012_df.shape)
        return geno012_df

    # --------------------------
    # Step 3
    # --------------------------
    def _build_gene_seq_df(self, genotype_df: pd.DataFrame) -> pd.DataFrame:
        try:
            import pysam  # type: ignore
        except Exception as exc:
            raise ImportError("pysam is required") from exc

        assert self.variant_df is not None
        fa = pysam.FastaFile(str(self.fasta_path))

        gene_rows: Dict[str, Dict[str, str]] = {}
        groups = genotype_df.groupby("gene_name", sort=True)
        self.logger.info("Building gene sequences: genes=%d", len(groups))

        for gi, (gene, g) in enumerate(groups, start=1):
            chrom = str(g["Chromosome"].iloc[0])
            rep = g.iloc[0]
            gene_feat = self._site_feature_name(rep, include_gene=True)

            gpos = self.variant_df[self.variant_df["Gene_id"] == gene][["Position", "Gene_position"]].drop_duplicates()
            starts = (gpos["Position"].astype(int) - gpos["Gene_position"].astype(int) + 1).values
            gene_start = int(np.median(starts))
            max_gene_pos = int(gpos["Gene_position"].max())
            gene_end = gene_start + max_gene_pos - 1

            ref_seq = fa.fetch(chrom, gene_start - 1, gene_end)
            row = {}
            for s in self.samples:
                seq = list(ref_seq)
                for _, r in g.iterrows():
                    gp = int(
                        self.variant_df.loc[
                            (self.variant_df["Gene_id"] == gene)
                            & (self.variant_df["Chromosome"] == r["Chromosome"])
                            & (self.variant_df["Position"] == int(r["Position"])),
                            "Gene_position",
                        ].iloc[0]
                    )
                    if int(r[s]) > 0:
                        idx = gp - 1
                        if 0 <= idx < len(seq):
                            seq[idx] = str(r["Alt"])
                row[s] = "".join(seq)
            gene_rows[gene_feat] = row
            if gi % 20 == 0 or gi == len(groups):
                self.logger.info("Gene sequence progress: %d/%d", gi, len(groups))

        gene_seq_by_gene = pd.DataFrame.from_dict(gene_rows, orient="index")
        gene_seq_by_gene.index.name = "feature"
        gene_seq_df = gene_seq_by_gene.T
        gene_seq_df = gene_seq_df.reset_index(names="sample")
        self.logger.info("Built gene_seq_df: shape=%s", gene_seq_df.shape)
        return gene_seq_df

    # --------------------------
    # Step 4
    # --------------------------
    def _build_snp_extseq_df(self, genotype_df: pd.DataFrame) -> pd.DataFrame:
        contexts = self._build_site_contexts(genotype_df)
        row_data: Dict[str, Dict[str, str]] = {}

        for _, r in genotype_df.iterrows():
            feat = self._site_feature_name(r, include_gene=False)
            ref_seq, alt_seq = contexts[(str(r["Chromosome"]), int(r["Position"]), str(r["Ref"]), str(r["Alt"]))]
            row = {}
            for s in self.samples:
                row[s] = alt_seq if int(r[s]) > 0 else ref_seq
            row_data[feat] = row

        snp_by_site = pd.DataFrame.from_dict(row_data, orient="index")
        snp_by_site.index.name = "site"
        snp_extseq_df = snp_by_site.T
        snp_extseq_df = snp_extseq_df.reset_index(names="sample")
        self.logger.info("Built snp_extseq_df: shape=%s", snp_extseq_df.shape)
        return snp_extseq_df

    def _build_snp_extseq_distance_df(self, genotype_df: pd.DataFrame) -> pd.DataFrame:
        contexts = self._build_site_contexts(genotype_df)
        dist_map: Dict[str, float] = {}

        for _, r in genotype_df.iterrows():
            feat = self._site_feature_name(r, include_gene=False)
            key = (str(r["Chromosome"]), int(r["Position"]), str(r["Ref"]), str(r["Alt"]))
            ref_seq, alt_seq = contexts[key]
            ref_emb = self._embed_sequences([ref_seq])[0]
            alt_emb = self._embed_sequences([alt_seq])[0]
            dist_map[feat] = float(np.linalg.norm(ref_emb - alt_emb))

        rows = []
        for s in self.samples:
            row = {"sample": s}
            for _, r in genotype_df.iterrows():
                feat = self._site_feature_name(r, include_gene=False)
                row[feat] = dist_map[feat] * float(r[s])
            rows.append(row)

        out = pd.DataFrame(rows)
        self.logger.info("Built snp_extseq_distance_df: shape=%s", out.shape)
        return out

    # --------------------------
    # Embedding + PCA
    # --------------------------
    def _embed_sequence_matrix(
        self,
        seq_df: pd.DataFrame,
        name_prefix: str,
        weight_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        self.logger.info("Embedding matrix: %s shape=%s use_pca=%s", name_prefix, seq_df.shape, self.use_pca)
        features = [c for c in seq_df.columns if c != "sample"]
        blocks = []
        out_cols = ["sample"]

        for i, feat in enumerate(features, start=1):
            seqs = seq_df[feat].astype(str).tolist()
            emb = self._embed_sequences(seqs)

            if weight_df is not None:
                emb = emb * weight_df[feat].astype(float).values.reshape(-1, 1)

            if self.use_pca:
                comp = self._pca_reduce(emb, self.pca_var_threshold)
                blocks.append(comp)
                out_cols.extend([f"{feat}_PC{k+1}" for k in range(comp.shape[1])])
            else:
                blocks.append(emb)
                out_cols.extend([f"{feat}_embed_{k}" for k in range(emb.shape[1])])

            if i % 200 == 0 or i == len(features):
                self.logger.info("Embedding progress(%s): %d/%d", name_prefix, i, len(features))

        out = pd.DataFrame(np.concatenate(blocks, axis=1), columns=out_cols[1:])
        out.insert(0, "sample", seq_df["sample"].tolist())
        self.logger.info("Built %s_embedding_df: shape=%s", name_prefix, out.shape)
        return out

    # --------------------------
    # Helpers
    # --------------------------
    def _sample_site_genotype_matrix(self, genotype_df: pd.DataFrame) -> pd.DataFrame:
        feat_rows = {}
        for _, r in genotype_df.iterrows():
            feat = self._site_feature_name(r, include_gene=False)
            feat_rows[feat] = {s: float(r[s]) for s in self.samples}
        mat = pd.DataFrame.from_dict(feat_rows, orient="index").T
        mat.index.name = "sample"
        mat = mat.reset_index()
        return mat

    def _build_site_contexts(self, genotype_df: pd.DataFrame) -> Dict[Tuple[str, int, str, str], Tuple[str, str]]:
        try:
            import pysam  # type: ignore
        except Exception as exc:
            raise ImportError("pysam is required") from exc

        fa = pysam.FastaFile(str(self.fasta_path))
        out: Dict[Tuple[str, int, str, str], Tuple[str, str]] = {}
        uniq = genotype_df[["Chromosome", "Position", "Ref", "Alt"]].drop_duplicates()

        for _, r in uniq.iterrows():
            chrom = str(r["Chromosome"])
            pos = int(r["Position"])
            ref = str(r["Ref"])
            alt = str(r["Alt"])
            left = fa.fetch(chrom, max(1, pos - self.flank_k) - 1, pos - 1)
            right = fa.fetch(chrom, pos, pos + self.flank_k)
            out[(chrom, pos, ref, alt)] = (f"{left}{ref}{right}", f"{left}{alt}{right}")
        self.logger.info("Built site contexts: n=%d", len(out))
        return out

    def _site_feature_name(self, row: pd.Series, include_gene: bool = False) -> str:
        base = f"{row['Chromosome']}:{int(row['Position'])}:{row['Ref']}>{row['Alt']}"
        if include_gene:
            return f"{base}-{row['gene_name']}"
        return base

    def _embed_sequences(self, seqs: List[str]) -> np.ndarray:
        vectors: Dict[str, np.ndarray] = {}
        unique_seqs = list(dict.fromkeys(seqs))

        for i, s in enumerate(unique_seqs, start=1):
            key = self._seq_hash(s)
            p = self.cache_dir / f"{key}.npy"
            if p.exists():
                v = np.load(p)
            else:
                # Embed one sequence each call to avoid batch embedding.
                v = self._to_1d_numpy(self.embedder(s))
                np.save(p, v)
            vectors[s] = v

            if i % 500 == 0 or i == len(unique_seqs):
                self.logger.info("Sequence embedding progress: %d/%d", i, len(unique_seqs))

        return np.vstack([vectors[s] for s in seqs]).astype(np.float32)

    @staticmethod
    def _pca_reduce(mat: np.ndarray, threshold: float) -> np.ndarray:
        from sklearn.decomposition import PCA

        if mat.shape[0] <= 1:
            return mat[:, :1]

        max_comp = min(mat.shape[0], mat.shape[1])
        pca_full = PCA(n_components=max_comp, svd_solver="full")
        pca_full.fit(mat)
        cum = np.cumsum(pca_full.explained_variance_ratio_)
        k = int(np.searchsorted(cum, threshold) + 1)
        k = max(1, min(k, max_comp))
        pca = PCA(n_components=k, svd_solver="full")
        return pca.fit_transform(mat).astype(np.float32)

    def _split_df(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        rng = np.random.default_rng(self.random_seed)
        if "sample" not in df.columns:
            raise ValueError("Dataframe must contain 'sample' column for splitting")

        n = len(df)
        all_idx = np.arange(n)
        sample_values = df["sample"].astype(str).values
        isolated_set = set(self.isolated_sample_list)
        isolated_idx = np.array([i for i, s in enumerate(sample_values) if s in isolated_set], dtype=int)

        test_n = max(1, int(round(n * self.test_ratio)))
        if len(isolated_idx) > test_n:
            test_n = len(isolated_idx)

        remaining_idx = np.array([i for i in all_idx if i not in set(isolated_idx)], dtype=int)
        rng.shuffle(remaining_idx)
        extra_n = max(0, test_n - len(isolated_idx))
        extra_test_idx = remaining_idx[:extra_n]
        test_idx = np.concatenate([isolated_idx, extra_test_idx]) if len(isolated_idx) > 0 else extra_test_idx

        rem_idx = np.array([i for i in all_idx if i not in set(test_idx)], dtype=int)
        rng.shuffle(rem_idx)
        val_n = int(round(len(rem_idx) * self.val_ratio))
        val_idx = rem_idx[:val_n]
        train_idx = rem_idx[val_n:]

        split = {
            "train": df.iloc[train_idx].reset_index(drop=True),
            "val": df.iloc[val_idx].reset_index(drop=True),
            "test": df.iloc[test_idx].reset_index(drop=True),
        }
        self.logger.info(
            "Split result: train=%d val=%d test=%d isolated_in_test=%d",
            len(split["train"]),
            len(split["val"]),
            len(split["test"]),
            split["test"]["sample"].astype(str).isin(isolated_set).sum(),
        )
        return split

    def _save_both(self, df: pd.DataFrame, name: str) -> None:
        df.to_csv(self.outdir / f"{name}.csv", index=False)
        df.to_parquet(self.outdir / f"{name}.parquet", index=False)

    @staticmethod
    def _seq_hash(seq: str) -> str:
        return hashlib.sha1(seq.encode("utf-8")).hexdigest()

    @staticmethod
    def _to_1d_numpy(x) -> np.ndarray:
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
            raise ValueError(f"Expected 2D embeddings, got shape={arr.shape}")
        return arr


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build feature datasets from variants")
    p.add_argument("--variant_df_path", required=True)
    p.add_argument("--vcf_path", required=True)
    p.add_argument("--fasta_path", required=True)
    p.add_argument("--outdir", required=True)

    p.add_argument("--embedder_type", required=True)
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--pooling", default="mean")
    p.add_argument("--local_files_only", action="store_true", default=True)
    p.add_argument("--allow_remote", dest="local_files_only", action="store_false")
    p.add_argument("--embedder_kwargs", default="{}")
    p.add_argument("--sample_list_path", default=None)

    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--flank_k", type=int, default=20)
    p.add_argument("--pca_var_threshold", type=float, default=0.95)
    p.add_argument("--use_pca", action="store_true", default=True)
    p.add_argument("--no-use_pca", dest="use_pca", action="store_false")
    return p


def main() -> None:
    args = build_parser().parse_args()
    ekw = json.loads(args.embedder_kwargs) if args.embedder_kwargs else {}
    if not isinstance(ekw, dict):
        raise ValueError("embedder_kwargs must be JSON object")

    builder = VariantFeatureBuilder(
        variant_df_path=args.variant_df_path,
        vcf_path=args.vcf_path,
        fasta_path=args.fasta_path,
        outdir=args.outdir,
        embedder_type=args.embedder_type,
        model_name_or_path=args.model_name_or_path,
        pooling=args.pooling,
        local_files_only=args.local_files_only,
        embedder_kwargs=ekw,
        sample_list_path=args.sample_list_path,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed,
        flank_k=args.flank_k,
        pca_var_threshold=args.pca_var_threshold,
        use_pca=args.use_pca,
    )
    outputs = builder.run()
    for k, v in outputs.items():
        print(f"{k}: {v.shape}")


if __name__ == "__main__":
    main()
