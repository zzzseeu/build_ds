from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

try:
    from intervaltree import Interval, IntervalTree  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "intervaltree is required. Please install with: pip install intervaltree"
    ) from exc


@dataclass(frozen=True)
class IntervalRecord:
    start: int  # 1-based inclusive
    end: int  # 1-based inclusive
    record_id: str


class GWASQTLVariantExtractor:
    """
    Extract VCF sites by GWAS/QTL intersection or union, then optionally filter by gene intervals.

    All coordinates are treated as 1-based.
    """

    @staticmethod
    def standard_chrom(chrom: str) -> str:
        """
        Normalize chromosome name to 'ChrN'-style.
        Examples: '1' -> 'Chr1', 'chr1' -> 'Chr1', 'CHR01' -> 'Chr1', 'x' -> 'ChrX'.
        """
        s = str(chrom).strip()
        if not s:
            return s

        s = re.sub(r"^chr", "", s, flags=re.IGNORECASE).strip()
        if not s:
            return "Chr"

        if re.fullmatch(r"\d+", s):
            s = str(int(s))
        else:
            s = s.upper()

        return f"Chr{s}"

    def __init__(
        self,
        gwas_path: str,
        qtl_path: str,
        vcf_path: str,
        type: str = "union",
        gene_interval_path: Optional[str] = None,
        ext_length: int = 0,
        outdir: str = ".",
        save_file: bool = True,
        out_prefix: str = "gwas_qtl_sites",
    ) -> None:
        self.gwas_path = Path(gwas_path)
        self.qtl_path = Path(qtl_path)
        self.vcf_path = Path(vcf_path)
        self.type = type.lower().strip()
        self.gene_interval_path = Path(gene_interval_path) if gene_interval_path else None
        self.ext_length = int(ext_length)
        self.outdir = Path(outdir)
        self.save_file = bool(save_file)
        self.out_prefix = out_prefix

        if self.type not in {"intersection", "union"}:
            raise ValueError("type must be one of: 'intersection', 'union'")
        if self.ext_length < 0:
            raise ValueError("ext_length must be >= 0")

    def run(self) -> pd.DataFrame:
        gwas_df = self._read_gwas()
        qtl_df = self._read_qtl()

        qtl_tree = self._build_interval_tree(
            df=qtl_df,
            chr_col="Chromosome",
            start_col="Start",
            end_col="End",
            id_col="QTL_name",
            ext_length=0,
        )

        selected_sites = self._select_sites_from_vcf(gwas_df=gwas_df, qtl_tree=qtl_tree)

        if self.gene_interval_path is not None:
            gene_df = self._read_gene_intervals()
            gene_tree = self._build_interval_tree(
                df=gene_df,
                chr_col="Chromosome",
                start_col="Start",
                end_col="End",
                id_col="Gene_id",
                ext_length=self.ext_length,
            )
            result_df = self._filter_by_gene_intervals(selected_sites, gene_tree)
        else:
            result_df = pd.DataFrame(
                [
                    {
                        "Chromosome": chrom,
                        "Position": pos,
                        "Gene_id": pd.NA,
                        "Gene_position": pd.NA,
                    }
                    for chrom, pos in sorted(selected_sites)
                ],
                columns=["Chromosome", "Position", "Gene_id", "Gene_position"],
            )

        if self.save_file:
            self.outdir.mkdir(parents=True, exist_ok=True)
            out_file = self.outdir / f"{self.out_prefix}_{self.type}.csv"
            result_df.to_csv(out_file, index=False)

        return result_df

    def _read_gwas(self) -> pd.DataFrame:
        df = pd.read_csv(self.gwas_path)
        required = ["Chromosome", "Position", "Trait"]
        self._validate_columns(df, required, "GWAS")
        df = df[required].copy()
        df["Chromosome"] = df["Chromosome"].map(self.standard_chrom)
        df["Position"] = df["Position"].astype(int)
        return df

    def _read_qtl(self) -> pd.DataFrame:
        df = pd.read_csv(self.qtl_path)
        required = ["Chromosome", "Start", "End", "Trait", "QTL_name"]
        self._validate_columns(df, required, "QTL")
        df = df[required].copy()
        df["Chromosome"] = df["Chromosome"].map(self.standard_chrom)
        df["Start"] = df["Start"].astype(int)
        df["End"] = df["End"].astype(int)
        return df

    def _read_gene_intervals(self) -> pd.DataFrame:
        assert self.gene_interval_path is not None
        df = pd.read_csv(self.gene_interval_path)

        # Support files with optional extra columns.
        required = ["Chromosome", "Start", "End", "Gene_id"]
        self._validate_columns(df, required, "Gene interval")

        df = df[required].copy()
        df["Chromosome"] = df["Chromosome"].map(self.standard_chrom)
        df["Start"] = df["Start"].astype(int)
        df["End"] = df["End"].astype(int)
        return df

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required: List[str], table_name: str) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{table_name} file missing columns: {missing}")

    def _build_interval_tree(
        self,
        df: pd.DataFrame,
        chr_col: str,
        start_col: str,
        end_col: str,
        id_col: str,
        ext_length: int,
    ) -> Dict[str, IntervalTree]:
        tree_dict: Dict[str, IntervalTree] = {}

        for _, row in df.iterrows():
            chrom = str(row[chr_col])
            start = max(1, int(row[start_col]) - ext_length)
            end = int(row[end_col]) + ext_length
            record_id = str(row[id_col])

            tree = tree_dict.setdefault(chrom, IntervalTree())
            # intervaltree uses [begin, end), convert from 1-based inclusive to half-open
            tree.add(Interval(start, end + 1, IntervalRecord(start=start, end=end, record_id=record_id)))

        return tree_dict

    def _select_sites_from_vcf(
        self,
        gwas_df: pd.DataFrame,
        qtl_tree: Dict[str, IntervalTree],
    ) -> Set[Tuple[str, int]]:
        gwas_sites: Set[Tuple[str, int]] = set(
            zip(gwas_df["Chromosome"].astype(str), gwas_df["Position"].astype(int))
        )

        selected: Set[Tuple[str, int]] = set()

        for chrom, pos in self._iter_vcf_sites(self.vcf_path):
            in_gwas = (chrom, pos) in gwas_sites
            in_qtl = bool(qtl_tree.get(chrom, IntervalTree()).overlap(pos, pos + 1))

            if self.type == "intersection":
                if in_gwas and in_qtl:
                    selected.add((chrom, pos))
            else:  # union
                if in_gwas or in_qtl:
                    selected.add((chrom, pos))

        return selected

    def _filter_by_gene_intervals(
        self,
        sites: Set[Tuple[str, int]],
        gene_tree: Dict[str, IntervalTree],
    ) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []

        for chrom, pos in sorted(sites):
            hits = gene_tree.get(chrom, IntervalTree()).overlap(pos, pos + 1)
            for hit in sorted(hits, key=lambda h: (h.data.start, h.data.end, h.data.record_id)):
                rec: IntervalRecord = hit.data
                rows.append(
                    {
                        "Chromosome": chrom,
                        "Position": pos,
                        "Gene_id": rec.record_id,
                        "Gene_position": pos - rec.start + 1,
                    }
                )

        return pd.DataFrame(rows, columns=["Chromosome", "Position", "Gene_id", "Gene_position"])

    @staticmethod
    def _iter_vcf_sites(vcf_path: Path) -> Iterable[Tuple[str, int]]:
        # Try pysam first (supports .vcf/.vcf.gz), fallback to text parsing.
        try:
            import pysam  # type: ignore

            vcf = pysam.VariantFile(str(vcf_path))
            for rec in vcf:
                yield GWASQTLVariantExtractor.standard_chrom(str(rec.chrom)), int(rec.pos)
            return
        except Exception:
            pass

        open_func = open
        if str(vcf_path).endswith(".gz"):
            import gzip

            open_func = gzip.open  # type: ignore

        with open_func(vcf_path, "rt") as f:  # type: ignore
            for line in f:
                if not line or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 2:
                    continue
                chrom = GWASQTLVariantExtractor.standard_chrom(fields[0])
                pos = int(fields[1])
                yield chrom, pos


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract VCF variants by GWAS/QTL union or intersection, "
            "with optional gene-interval filtering."
        )
    )
    parser.add_argument("--gwas_path", required=True, help="GWAS CSV path (Chromosome,Position,Trait)")
    parser.add_argument("--qtl_path", required=True, help="QTL CSV path (Chromosome,Start,End,Trait,QTL_name)")
    parser.add_argument("--vcf_path", required=True, help="VCF path (.vcf or .vcf.gz)")
    parser.add_argument(
        "--type",
        default="union",
        choices=["intersection", "union"],
        help="How to combine GWAS and QTL sites (default: union)",
    )
    parser.add_argument(
        "--gene_interval_path",
        default=None,
        help="Optional gene interval CSV path (Chromosome,Start,End,Gene_id)",
    )
    parser.add_argument(
        "--ext_length",
        type=int,
        default=0,
        help="Extend gene intervals upstream/downstream by this length (default: 0)",
    )
    parser.add_argument("--outdir", default=".", help="Output directory when save_file is enabled")
    parser.add_argument("--out_prefix", default="gwas_qtl_sites", help="Output file prefix")
    parser.add_argument(
        "--save_file",
        action="store_true",
        default=True,
        help="Save output CSV (default: True)",
    )
    parser.add_argument(
        "--no-save_file",
        dest="save_file",
        action="store_false",
        help="Do not save output file; only print result summary",
    )
    return parser


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    extractor = GWASQTLVariantExtractor(
        gwas_path=args.gwas_path,
        qtl_path=args.qtl_path,
        vcf_path=args.vcf_path,
        type=args.type,
        gene_interval_path=args.gene_interval_path,
        ext_length=args.ext_length,
        outdir=args.outdir,
        save_file=args.save_file,
        out_prefix=args.out_prefix,
    )
    out = extractor.run()
    print(out.head())
    print(f"rows={len(out)}")


if __name__ == "__main__":
    main()
