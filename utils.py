"""Utility helpers for genomic file parsing."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable
import re
from urllib.parse import unquote

import numpy as np
import pandas as pd

try:
    from loguru import logger as _loguru_logger
except Exception:  # pragma: no cover - optional dependency
    _loguru_logger = None


GFF3_COLUMNS = [
    "seqid",
    "source",
    "type",
    "start",
    "end",
    "score",
    "strand",
    "phase",
    "attributes",
]


_LOGURU_CONFIGURED = False


def initLogger(log_file: str | Path | None = None, level: str = "INFO"):
    """Initialize and return a shared project logger backed by loguru."""
    global _LOGURU_CONFIGURED
    if _loguru_logger is None:
        raise ImportError("loguru is required for initLogger")

    if not _LOGURU_CONFIGURED:
        _loguru_logger.remove()
        _loguru_logger.add(
            sys.stdout,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        )
        _LOGURU_CONFIGURED = True

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _loguru_logger.add(
            str(log_file),
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        )
    return _loguru_logger


def getLogger():
    """Return the shared project logger, configuring stdout if needed."""
    return initLogger()


def standard_chrom(chrom: str) -> str | None:
    """Normalize chromosome names to ``ChrN`` and keep numeric chromosomes only.

    Examples
    --------
    ``1`` -> ``Chr1``
    ``chr1`` -> ``Chr1``
    ``chr01`` -> ``Chr1``
    ``chr_01`` -> ``Chr1``
    """
    c = str(chrom).strip()
    if not c:
        return None
    match = re.search(r"(\d+)", c)
    if match is None:
        return None
    return f"Chr{int(match.group(1))}"


def standard_sample_name(sample: str) -> str | None:
    """Normalize sample names to ``sample_N`` using the first numeric token.

    Examples
    --------
    ``1`` -> ``sample_1``
    ``sample1`` -> ``sample_1``
    ``Sample_01`` -> ``sample_1``
    """
    s = str(sample).strip()
    if not s:
        return None
    match = re.search(r"(\d+)", s)
    if match is None:
        return None
    return f"sample_{int(match.group(1))}"


def to_numpy_1d_embedding(x) -> np.ndarray:
    """Convert an embedding-like object into a 1D float32 numpy array."""
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


def build_fasta_chrom_map(fasta) -> dict[str, str]:
    """Map normalized chromosome names to FASTA reference names."""
    chrom_map: dict[str, str] = {}
    for ref in fasta.references:
        chrom = standard_chrom(ref)
        if chrom is not None and chrom not in chrom_map:
            chrom_map[chrom] = ref
    return chrom_map


def fetch_window_with_padding(
    fasta,
    fasta_chrom: str,
    start_1based: int,
    end_1based: int,
) -> str:
    """Fetch a 1-based closed interval from FASTA and pad out-of-bound bases with N."""
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


def classify_variant_type(ref: str, alt: str) -> str:
    """Classify variant type from REF/ALT allele strings."""
    ref = str(ref).upper()
    alt = str(alt).upper()
    if len(ref) == 1 and len(alt) == 1:
        return "SNV"
    if len(ref) == len(alt):
        return "MNV"
    if len(ref) < len(alt):
        return "Insertion"
    if len(ref) > len(alt):
        return "Deletion"
    return "Complex"


def parse_gff3_attributes(attr_text: str) -> dict[str, str]:
    """Parse the GFF3 or GTF attribute column into a dictionary.

    Parameters
    ----------
    attr_text : str
        Raw attribute text from the 9th annotation column.

    Returns
    -------
    dict[str, str]
        Parsed key-value pairs. Invalid fragments are ignored.
    """
    attr_dict: dict[str, str] = {}
    if not isinstance(attr_text, str) or not attr_text.strip():
        return attr_dict

    for item in attr_text.strip().split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif " " in item:
            # Support common GTF fragments such as gene_id "X".
            key, value = item.split(" ", 1)
        else:
            continue

        key = key.strip()
        value = unquote(value.strip().strip('"'))
        if not key:
            continue

        if key in attr_dict and attr_dict[key]:
            existing = set(x for x in attr_dict[key].split(",") if x)
            existing.update(x for x in value.split(",") if x)
            attr_dict[key] = ",".join(sorted(existing))
        else:
            attr_dict[key] = value
    return attr_dict


def _normalize_feature_set(feature: str | Iterable[str]) -> set[str]:
    """Normalize target feature input into a deduplicated set."""
    if isinstance(feature, str):
        return {feature.strip()}
    return {str(x).strip() for x in feature if str(x).strip()}


def _annotation_db_path(annotation_path: Path) -> Path:
    """Build the default gffutils SQLite path for an annotation file."""
    return annotation_path.with_suffix(annotation_path.suffix + ".gffutils.db")


def _build_feature_labels(attrs: dict[str, str]) -> dict[str, str]:
    """Build unified labels for both GFF3 and GTF attributes."""
    feature_id = attrs.get("ID") or attrs.get("gene_id") or attrs.get("transcript_id") or ""
    feature_name = attrs.get("Name") or attrs.get("gene_name") or attrs.get("gene_id") or feature_id
    parent = attrs.get("Parent") or attrs.get("transcript_id") or attrs.get("gene_id") or ""
    return {
        "ID": feature_id,
        "Name": feature_name,
        "Parent": parent,
    }


def _parse_gff3_record_line(line: str) -> dict[str, str] | None:
    """Parse one text GFF3 record line into a column dictionary."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    parts = stripped.split("\t")
    if len(parts) != 9:
        return None
    return dict(zip(GFF3_COLUMNS, parts))


def extract_gff3_feature_intervals(
    gff3_path: str | Path,
    feature: str | Iterable[str],
) -> pd.DataFrame:
    """Extract intervals for one or more target features from a GFF3 or GTF file.

    All coordinates are kept in the original annotation-file convention, which
    is 1-based and closed on both ends.

    Parameters
    ----------
    gff3_path : str | Path
        Path to the input GFF3 or GTF file.
    feature : str | Iterable[str]
        Target feature name or a collection of feature names, for example
        ``gene`` or ``[\"gene\", \"exon\"]``.

    Returns
    -------
    pd.DataFrame
        Interval table with columns:
        ``Chromosome, Start, End, Feature, Strand, ID, Name, Parent, Attributes``.
    """
    gff3_path = Path(gff3_path)
    feature_set = _normalize_feature_set(feature)
    if not feature_set:
        raise ValueError("feature must contain at least one non-empty feature name")

    rows: list[dict[str, str | int]] = []
    with gff3_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("##FASTA"):
                break

            record = _parse_gff3_record_line(line)
            if record is None:
                continue
            record_type = str(record["type"])
            if record_type not in feature_set:
                continue

            attrs = parse_gff3_attributes(str(record["attributes"]))
            labels = _build_feature_labels(attrs)
            try:
                start = int(record["start"])
                end = int(record["end"])
            except (TypeError, ValueError):
                continue

            rows.append(
                {
                    "Chromosome": str(record["seqid"]),
                    "Start": start,
                    "End": end,
                    "Feature": record_type,
                    "Strand": str(record["strand"]),
                    "ID": labels["ID"],
                    "Name": labels["Name"],
                    "Parent": labels["Parent"],
                    "Attributes": str(record["attributes"]),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "Chromosome",
            "Start",
            "End",
            "Feature",
            "Strand",
            "ID",
            "Name",
            "Parent",
            "Attributes",
        ],
    )

def extract_gff3_feature_intervals_gffutils(
    gff3_path: str | Path,
    feature: str | Iterable[str],
    db_path: str | Path | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Extract target feature intervals from a GFF3 or GTF file with ``gffutils``.

    This function builds or reuses a local ``gffutils`` database, then queries
    one or more feature types such as ``gene`` or ``exon``. Coordinates remain
    1-based, matching the GFF3 convention.

    Parameters
    ----------
    gff3_path : str | Path
        Path to the input GFF3 or GTF file.
    feature : str | Iterable[str]
        Target feature name or feature collection.
    db_path : str | Path | None
        Optional path to the generated ``gffutils`` SQLite database. When
        omitted, a sidecar database named ``<annotation>.gffutils.db`` is used.
    force_rebuild : bool
        Whether to rebuild the database even if it already exists.

    Returns
    -------
    pd.DataFrame
        Interval table with columns:
        ``Chromosome, Start, End, Feature, Strand, ID, Name, Parent, Attributes``.
    """
    try:
        import gffutils  # type: ignore
    except Exception as exc:
        raise ImportError("gffutils is required for extract_gff3_feature_intervals_gffutils") from exc

    gff3_path = Path(gff3_path)
    if isinstance(feature, str):
        feature_list = [feature]
    else:
        feature_list = [str(x) for x in feature]

    if db_path is None:
        db_path = _annotation_db_path(gff3_path)
    db_path = Path(db_path)

    if force_rebuild and db_path.exists():
        db_path.unlink()

    if not db_path.exists():
        create_kwargs = {
            "data": str(gff3_path),
            "dbfn": str(db_path),
            "force": True,
            "keep_order": True,
            "merge_strategy": "merge",
            "sort_attribute_values": True,
        }
        if gff3_path.suffix.lower() == ".gtf":
            create_kwargs["disable_infer_genes"] = True
            create_kwargs["disable_infer_transcripts"] = True

        gffutils.create_db(**create_kwargs)

    db = gffutils.FeatureDB(str(db_path), keep_order=True)

    rows: list[dict[str, str | int]] = []
    for feature_name in feature_list:
        for record in db.features_of_type(feature_name):
            attrs = {k: ",".join(v) for k, v in record.attributes.items()}
            labels = _build_feature_labels(attrs)
            rows.append(
                {
                    "Chromosome": str(record.seqid),
                    "Start": int(record.start),
                    "End": int(record.end),
                    "Feature": str(record.featuretype),
                    "Strand": str(record.strand),
                    "ID": labels["ID"],
                    "Name": labels["Name"],
                    "Parent": labels["Parent"],
                    "Attributes": str(record.attributes),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "Chromosome",
            "Start",
            "End",
            "Feature",
            "Strand",
            "ID",
            "Name",
            "Parent",
            "Attributes",
        ],
    )


def build_feature_interval_trees(
    interval_df: pd.DataFrame,
    ext_len: int = 500,
) -> dict[str, object]:
    """Build per-chromosome interval trees from a feature interval dataframe.

    The returned structure is ``{chromosome: IntervalTree(...)}``. Interval
    payload stores a compact feature descriptor with the feature name, start,
    end, and common identifiers.

    Parameters
    ----------
    interval_df : pd.DataFrame
        Dataframe produced by one of the GFF3 interval extraction helpers.
    ext_len : int
        Extension length added to upstream and downstream of each interval.

    Returns
    -------
    dict[str, object]
        Mapping from chromosome name to ``intervaltree.IntervalTree``.
    """
    try:
        from intervaltree import Interval, IntervalTree  # type: ignore
    except Exception as exc:
        raise ImportError("intervaltree is required for build_feature_interval_trees") from exc

    required = {"Chromosome", "Start", "End", "Feature"}
    missing = sorted(required - set(interval_df.columns))
    if missing:
        raise ValueError(f"interval_df missing columns: {missing}")

    trees: dict[str, object] = {}
    for row in interval_df.itertuples(index=False):
        chrom = standard_chrom(str(row.Chromosome))
        if chrom is None:
            continue
        start = int(row.Start)
        end = int(row.End)
        feature_name = str(row.Feature)
        if end < start:
            start, end = end, start

        start = max(1, start - ext_len)
        end = end + ext_len
        payload = {
            "feature": feature_name,
            "start": start,
            "end": end,
            "id": getattr(row, "ID", ""),
            "name": getattr(row, "Name", ""),
            "parent": getattr(row, "Parent", ""),
        }

        if chrom not in trees:
            trees[chrom] = IntervalTree()

        # intervaltree uses half-open intervals, so convert 1-based closed
        # [start, end] into [start, end + 1) while preserving original bounds
        # in the payload.
        trees[chrom].add(Interval(start, end + 1, payload))

    return trees


def extract_gff3_feature_interval_trees(
    gff3_path: str | Path,
    feature: str | Iterable[str],
    ext_len: int = 500,
) -> dict[str, object]:
    """Parse GFF3 or GTF text and return per-chromosome feature interval trees."""
    interval_df = extract_gff3_feature_intervals(gff3_path=gff3_path, feature=feature)
    return build_feature_interval_trees(interval_df, ext_len=ext_len)


def extract_gff3_feature_interval_trees_gffutils(
    gff3_path: str | Path,
    feature: str | Iterable[str],
    db_path: str | Path | None = None,
    force_rebuild: bool = False,
    ext_len: int = 500,
) -> dict[str, object]:
    """Parse GFF3 or GTF with gffutils and return per-chromosome feature interval trees."""
    interval_df = extract_gff3_feature_intervals_gffutils(
        gff3_path=gff3_path,
        feature=feature,
        db_path=db_path,
        force_rebuild=force_rebuild,
    )
    return build_feature_interval_trees(interval_df, ext_len=ext_len)


def query_feature_interval_trees(
    interval_trees: dict[str, object],
    chromosome: str,
    position: int,
) -> list[dict[str, str | int]]:
    """Query per-chromosome interval trees with 1-based coordinates."""
    chrom = standard_chrom(chromosome)
    if chrom is None or chrom not in interval_trees:
        return []
    tree = interval_trees[chrom]
    hits = []
    for iv in sorted(tree.at(int(position)), key=lambda x: (x.begin, x.end, str(x.data))):
        if isinstance(iv.data, dict):
            hits.append(iv.data)
    return hits
