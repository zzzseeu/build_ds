"""Extract GWAS/QTL variants from VCF.

All genomic coordinates are treated as 1-based.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from logging import init_logger
from utils import (
    extract_gff3_feature_interval_trees,
    extract_gff3_feature_interval_trees_gffutils,
    query_feature_interval_trees,
    standard_chrom,
)


@dataclass
class Interval:
    """Simple inclusive interval for 1-based coordinates."""

    start: int
    end: int
    trait: str

    def contains(self, pos: int) -> bool:
        return self.start <= pos <= self.end


class IntervalTree:
    """Lightweight interval tree via sorted intervals + binary prunable scan."""

    def __init__(self) -> None:
        self.intervals: List[Interval] = []
        self._built = False

    def add(self, start: int, end: int, trait: str) -> None:
        if start > end:
            start, end = end, start
        self.intervals.append(Interval(start=start, end=end, trait=trait))
        self._built = False

    def build(self) -> None:
        self.intervals.sort(key=lambda x: (x.start, x.end))
        self._built = True

    def query_traits(self, pos: int) -> Set[str]:
        if not self._built:
            self.build()
        traits: Set[str] = set()
        for iv in self.intervals:
            if iv.start > pos:
                break
            if iv.contains(pos):
                traits.add(iv.trait)
        return traits


class GWASQTLVariantExtractor:
    """Extract GWAS/QTL union or intersection sites from VCF."""

    def __init__(
        self,
        gwas_csv_path: str,
        qtl_csv_path: str,
        type: str,
        vcf_path: str,
        gff3_path: str,
        outprefix: str,
        trait: str | None = None,
        pvalue_threshold: float = 1e6,
        lod_threshold: float = 2.5,
        pve_threshold: float = 10.0,
        gff3_feature: str = "gene",
        ext_len: int = 500,
        use_gffutils: bool = True,
        gene_list: str | None = None,
    ) -> None:
        type = type.lower().strip()
        if type not in {"intersect", "union"}:
            raise ValueError("type must be one of: intersect, union")

        self.gwas_csv_path = Path(gwas_csv_path)
        self.qtl_csv_path = Path(qtl_csv_path)
        self.type = type
        self.vcf_path = Path(vcf_path)
        self.gff3_path = Path(gff3_path)
        self.outprefix = Path(outprefix)
        self.trait = trait
        self.trait_set = self._parse_trait_filter(trait)
        self.gene_set = self._parse_gene_filter(gene_list)
        self.pvalue_threshold = float(pvalue_threshold)
        self.lod_threshold = float(lod_threshold)
        self.pve_threshold = float(pve_threshold)
        self.gff3_feature = gff3_feature
        self.ext_len = int(ext_len)
        self.use_gffutils = bool(use_gffutils)

        self.outprefix.parent.mkdir(parents=True, exist_ok=True)
        log_path = self.outprefix.parent / f"{self.outprefix.name}_{datetime.now().strftime('%Y-%m-%d')}.log"
        self.logger = init_logger("GWASQTLVariantExtractor", log_file=log_path)

    @staticmethod
    def _parse_trait_filter(trait: str | None) -> Set[str] | None:
        if trait is None:
            return None
        items = [x.strip() for x in str(trait).split(",")]
        items = [x for x in items if x]
        if not items:
            return None
        return set(items)

    @staticmethod
    def _parse_feature_filter(feature: str) -> list[str]:
        items = [x.strip() for x in str(feature).split(",")]
        return [x for x in items if x]

    @staticmethod
    def _parse_gene_filter(gene_list: str | None) -> Set[str] | None:
        if gene_list is None:
            return None
        items = [x.strip() for x in str(gene_list).split(",")]
        items = [x for x in items if x]
        if not items:
            return None
        return set(items)

    def _read_gwas(self) -> pd.DataFrame:
        required = ["Chromosome", "Position", "Trait", "pvalue"]
        df = pd.read_csv(self.gwas_csv_path)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"GWAS CSV missing columns: {missing}")

        df = df.copy()
        df["Chromosome"] = df["Chromosome"].astype(str).map(standard_chrom)
        df = df[df["Chromosome"].notna()]
        df["Position"] = df["Position"].astype(int)
        df["Trait"] = df["Trait"].astype(str)
        df["pvalue"] = pd.to_numeric(df["pvalue"], errors="coerce")
        df = df[df["pvalue"] < self.pvalue_threshold]
        if self.trait_set is not None:
            df = df[df["Trait"].isin(self.trait_set)]

        self.logger.info("GWAS loaded: shape=%s", df.shape)
        return df

    def _read_qtl(self) -> pd.DataFrame:
        required = ["QTL", "Chromosome", "LOD", "PVE", "start_pos", "end_pos", "Trait"]
        df = pd.read_csv(self.qtl_csv_path)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"QTL CSV missing columns: {missing}")

        df = df.copy()
        df["Chromosome"] = df["Chromosome"].astype(str).map(standard_chrom)
        df = df[df["Chromosome"].notna()]
        df["start_pos"] = pd.to_numeric(df["start_pos"], errors="coerce").astype("Int64")
        df["end_pos"] = pd.to_numeric(df["end_pos"], errors="coerce").astype("Int64")
        df["Trait"] = df["Trait"].astype(str)
        df["LOD"] = pd.to_numeric(df["LOD"], errors="coerce")
        df["PVE"] = pd.to_numeric(df["PVE"], errors="coerce")

        df = df[(df["LOD"] > self.lod_threshold) & (df["PVE"] > self.pve_threshold)]
        df = df[df["start_pos"].notna() & df["end_pos"].notna()]
        if self.trait_set is not None:
            df = df[df["Trait"].isin(self.trait_set)]

        self.logger.info("QTL loaded: shape=%s", df.shape)
        return df

    def _build_qtl_trees(self, qtl_df: pd.DataFrame) -> Dict[str, IntervalTree]:
        trees: Dict[str, IntervalTree] = {}
        for row in qtl_df.itertuples(index=False):
            chrom = str(row.Chromosome)
            start = int(row.start_pos)
            end = int(row.end_pos)
            trait = str(row.Trait)
            if chrom not in trees:
                trees[chrom] = IntervalTree()
            trees[chrom].add(start, end, trait)

        for tree in trees.values():
            tree.build()

        self.logger.info("QTL interval trees built: chromosomes=%d", len(trees))
        return trees

    @staticmethod
    def _build_gwas_site_map(gwas_df: pd.DataFrame) -> Dict[Tuple[str, int], Set[str]]:
        site_map: Dict[Tuple[str, int], Set[str]] = {}
        for row in gwas_df.itertuples(index=False):
            key = (str(row.Chromosome), int(row.Position))
            site_map.setdefault(key, set()).add(str(row.Trait))
        return site_map

    def _extract_from_vcf(
        self,
        gwas_site_map: Dict[Tuple[str, int], Set[str]],
        qtl_trees: Dict[str, IntervalTree],
    ) -> pd.DataFrame:
        try:
            import pysam  # type: ignore
        except Exception as exc:
            raise ImportError("pysam is required to read VCF") from exc

        rows: List[Tuple[str, int, str]] = []
        seen = 0
        kept = 0

        vcf = pysam.VariantFile(str(self.vcf_path))
        for rec in vcf:
            seen += 1
            chrom = standard_chrom(str(rec.chrom))
            if chrom is None:
                continue
            pos = int(rec.pos)  # 1-based
            key = (chrom, pos)

            g_traits = gwas_site_map.get(key, set())
            q_traits = qtl_trees.get(chrom, IntervalTree()).query_traits(pos) if chrom in qtl_trees else set()

            if self.type == "intersect":
                traits = g_traits & q_traits
            else:
                traits = g_traits | q_traits

            if traits:
                kept += 1
                for t in sorted(traits):
                    rows.append((chrom, pos, t))

            if seen % 100000 == 0:
                self.logger.info("VCF scan progress: seen=%d kept_sites=%d", seen, kept)

        out_df = pd.DataFrame(rows, columns=["Chromosome", "Position", "Trait"])
        out_df = out_df.drop_duplicates().sort_values(["Chromosome", "Position", "Trait"]).reset_index(drop=True)

        self.logger.info("VCF extraction done: seen=%d output_rows=%d", seen, len(out_df))
        return out_df

    def _build_feature_trees(self) -> dict[str, object] | None:
        features = self._parse_feature_filter(self.gff3_feature)
        if not features:
            raise ValueError("gff3_feature must contain at least one feature name")

        if self.use_gffutils:
            trees = extract_gff3_feature_interval_trees_gffutils(
                gff3_path=self.gff3_path,
                feature=features,
                ext_len=self.ext_len,
            )
        else:
            trees = extract_gff3_feature_interval_trees(
                gff3_path=self.gff3_path,
                feature=features,
                ext_len=self.ext_len,
            )

        self.logger.info(
            "Feature interval trees built: chromosomes=%d features=%s ext_len=%d backend=%s",
            len(trees),
            features,
            self.ext_len,
            "gffutils" if self.use_gffutils else "text",
        )
        return trees

    def _filter_by_gene_list(self, site_df: pd.DataFrame) -> pd.DataFrame:
        if self.gene_set is None:
            return site_df
        out_df = site_df[site_df["Gene"].astype(str).isin(self.gene_set)].copy()
        self.logger.info(
            "Gene-list filtered output built: rows=%d genes=%d",
            len(out_df),
            out_df["Gene"].nunique() if not out_df.empty else 0,
        )
        return out_df

    @staticmethod
    def _feature_label(hit: dict[str, str | int]) -> str:
        name = str(hit.get("name", "")).strip()
        feature_id = str(hit.get("id", "")).strip()
        parent = str(hit.get("parent", "")).strip()
        feature = str(hit.get("feature", "")).strip()
        return name or feature_id or parent or feature

    def _filter_sites_by_feature_trees(
        self,
        site_df: pd.DataFrame,
        feature_trees: dict[str, object] | None,
    ) -> pd.DataFrame:
        if feature_trees is None:
            return site_df

        rows: list[tuple[str, int, str]] = []
        for row in site_df.itertuples(index=False):
            hits = query_feature_interval_trees(
                interval_trees=feature_trees,
                chromosome=str(row.Chromosome),
                position=int(row.Position),
            )
            if not hits:
                continue
            for hit in hits:
                rows.append((str(row.Chromosome), int(row.Position), self._feature_label(hit)))

        out_df = pd.DataFrame(rows, columns=["Chromosome", "Position", "Gene"])
        out_df = out_df.drop_duplicates().sort_values(["Chromosome", "Position", "Gene"]).reset_index(drop=True)
        self.logger.info("Feature-filtered output built: rows=%d", len(out_df))
        return out_df

    def run(self) -> pd.DataFrame:
        self.logger.info("Run started: type=%s trait_filter=%s", self.type, sorted(self.trait_set) if self.trait_set else None)

        gwas_df = self._read_gwas()
        qtl_df = self._read_qtl()
        qtl_trees = self._build_qtl_trees(qtl_df)
        gwas_site_map = self._build_gwas_site_map(gwas_df)
        feature_trees = self._build_feature_trees()

        out_df = self._extract_from_vcf(gwas_site_map, qtl_trees)
        out_df = self._filter_sites_by_feature_trees(out_df, feature_trees)
        out_df = self._filter_by_gene_list(out_df)
        date_tag = datetime.now().strftime("%Y-%m-%d")
        out_csv = self.outprefix.parent / f"{self.outprefix.name}_{date_tag}.csv"
        out_df.to_csv(out_csv, index=False)

        self.logger.info("Output saved: %s", out_csv)
        self.logger.info("Run finished")
        return out_df


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract GWAS/QTL union or intersection variants from VCF")
    p.add_argument("--gwas_csv_path", required=True)
    p.add_argument("--qtl_csv_path", required=True)
    p.add_argument("--type", required=True, choices=["intersect", "union"])
    p.add_argument("--vcf_path", required=True)
    p.add_argument("--gff3_path", required=True, help="Annotation path in GFF3 or GTF format")
    p.add_argument("--outprefix", required=True)
    p.add_argument("--trait", default=None, help="Single trait or comma-separated traits, e.g. Yield or Yield,Height")
    p.add_argument("--pvalue_threshold", type=float, default=1e6)
    p.add_argument("--LOD_threshold", type=float, default=2.5, dest="lod_threshold")
    p.add_argument("--PVE_threshold", type=float, default=10.0, dest="pve_threshold")
    p.add_argument("--gff3_feature", default="gene", help="Single feature or comma-separated features, e.g. gene or gene,exon")
    p.add_argument("--ext_len", type=int, default=500)
    p.add_argument("--gene_list", default=None, help="Comma-separated gene names or IDs to keep in final output")
    p.add_argument("--use_gffutils", action="store_true", default=True)
    p.add_argument("--no-use_gffutils", dest="use_gffutils", action="store_false")
    return p


def main() -> None:
    args = build_parser().parse_args()
    extractor = GWASQTLVariantExtractor(
        gwas_csv_path=args.gwas_csv_path,
        qtl_csv_path=args.qtl_csv_path,
        type=args.type,
        vcf_path=args.vcf_path,
        gff3_path=args.gff3_path,
        outprefix=args.outprefix,
        trait=args.trait,
        pvalue_threshold=args.pvalue_threshold,
        lod_threshold=args.lod_threshold,
        pve_threshold=args.pve_threshold,
        gff3_feature=args.gff3_feature,
        ext_len=args.ext_len,
        use_gffutils=args.use_gffutils,
        gene_list=args.gene_list,
    )
    out_df = extractor.run()
    print(out_df.shape)


if __name__ == "__main__":
    main()
