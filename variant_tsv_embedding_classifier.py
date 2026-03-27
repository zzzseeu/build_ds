"""Build variant embeddings and train a VARIANT_TYPE classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from embedding import UnifiedEmbedder
from utils import (
    CachedSequenceEmbedder,
    build_fasta_chrom_map,
    classify_variant_type,
    fetch_window_with_padding,
    initLogger,
    standard_chrom,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


TSV_COLUMN_ALIASES = {
    "chrom": "Chromosome",
    "chr": "Chromosome",
    "chromosome": "Chromosome",
    "pos": "Position",
    "position": "Position",
    "ref": "REF",
    "ref_allele": "REF",
    "alt": "ALT",
    "alt_allele": "ALT",
    "variant_type": "VARIANT_TYPE",
}


def progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def require_sklearn():
    try:
        from sklearn.calibration import CalibratedClassifierCV  # noqa: F401
        from sklearn.ensemble import RandomForestClassifier  # noqa: F401
        from sklearn.linear_model import LogisticRegression  # noqa: F401
        from sklearn.metrics import (  # noqa: F401
            accuracy_score,
            average_precision_score,
            balanced_accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            matthews_corrcoef,
            precision_recall_curve,
            precision_score,
            recall_score,
            roc_auc_score,
            roc_curve,
        )
        from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit  # noqa: F401
        from sklearn.pipeline import Pipeline  # noqa: F401
        from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize  # noqa: F401
    except Exception as exc:
        raise ImportError("scikit-learn is required for classification mode") from exc


def infer_site_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in TSV_COLUMN_ALIASES and TSV_COLUMN_ALIASES[key] not in df.columns:
            rename_map[col] = TSV_COLUMN_ALIASES[key]
    out = df.rename(columns=rename_map).copy()
    required = {"Chromosome", "Position", "VARIANT_TYPE"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"TSV missing required columns: {missing}")
    out["Chromosome"] = out["Chromosome"].map(standard_chrom)
    out["Position"] = pd.to_numeric(out["Position"], errors="coerce").astype("Int64")
    out["VARIANT_TYPE"] = out["VARIANT_TYPE"].astype(str).str.strip()
    out = out[out["Chromosome"].notna() & out["Position"].notna() & out["VARIANT_TYPE"].ne("")].copy()
    out["Position"] = out["Position"].astype(int)
    if "REF" in out.columns:
        out["REF"] = out["REF"].astype(str).str.upper()
    if "ALT" in out.columns:
        out["ALT"] = out["ALT"].astype(str).str.upper()
    return out


def make_site_key(
    chrom: str,
    pos: int,
    ref: str | None = None,
    alt: str | None = None,
) -> tuple[str, int, str | None, str | None]:
    return (str(chrom), int(pos), ref if ref else None, alt if alt else None)


def load_tsv_sites(
    tsv_path: Path,
) -> tuple[
    pd.DataFrame,
    set[tuple[str, int]],
    set[tuple[str, int, str | None, str | None]],
    dict[tuple[str, int, str | None, str | None], str],
]:
    site_df = infer_site_columns(pd.read_csv(tsv_path, sep="\t"))
    positional_keys = {
        (str(row.Chromosome), int(row.Position))
        for row in site_df.itertuples(index=False)
    }
    allele_keys = {
        make_site_key(
            row.Chromosome,
            row.Position,
            getattr(row, "REF", None),
            getattr(row, "ALT", None),
        )
        for row in site_df.itertuples(index=False)
    }
    variant_type_map: dict[tuple[str, int, str | None, str | None], str] = {}
    for row in site_df.itertuples(index=False):
        key = make_site_key(
            row.Chromosome,
            row.Position,
            getattr(row, "REF", None),
            getattr(row, "ALT", None),
        )
        value = str(getattr(row, "VARIANT_TYPE", "")).strip()
        if value:
            variant_type_map[key] = value
    return site_df, positional_keys, allele_keys, variant_type_map


def collect_variants_from_vcf(
    vcf_path: Path,
    site_df: pd.DataFrame,
    positional_keys: set[tuple[str, int]],
    allele_keys: set[tuple[str, int, str | None, str | None]],
    variant_type_map: dict[tuple[str, int, str | None, str | None], str],
) -> pd.DataFrame:
    try:
        import pysam  # type: ignore
    except Exception as exc:
        raise ImportError("pysam is required for VCF/FASTA processing") from exc

    rows: list[dict[str, object]] = []
    has_allele_filter = "REF" in site_df.columns and "ALT" in site_df.columns

    with pysam.VariantFile(str(vcf_path)) as vcf:
        for rec in progress(vcf, desc="Reading VCF", unit="record"):
            chrom = standard_chrom(rec.chrom)
            if chrom is None:
                continue
            pos_key = (chrom, int(rec.pos))
            if pos_key not in positional_keys:
                continue
            ref = str(rec.ref).upper()
            for alt in rec.alts or ():
                alt = str(alt).upper()
                allele_key = make_site_key(chrom, rec.pos, ref, alt)
                if has_allele_filter and allele_key not in allele_keys:
                    continue
                rows.append(
                    {
                        "Chromosome": chrom,
                        "Position": int(rec.pos),
                        "REF": ref,
                        "ALT": alt,
                        "VARIANT_TYPE": variant_type_map.get(allele_key) or classify_variant_type(ref, alt),
                    }
                )

    variant_df = pd.DataFrame(rows)
    if variant_df.empty:
        return variant_df
    return variant_df.drop_duplicates(subset=["Chromosome", "Position", "REF", "ALT"]).reset_index(drop=True)


def add_sequence_windows(variant_df: pd.DataFrame, fasta_path: Path, k: int) -> pd.DataFrame:
    try:
        import pysam  # type: ignore
    except Exception as exc:
        raise ImportError("pysam is required for VCF/FASTA processing") from exc

    rows: list[dict[str, object]] = []
    with pysam.FastaFile(str(fasta_path)) as fasta:
        chrom_map = build_fasta_chrom_map(fasta)
        for row in progress(
            variant_df.itertuples(index=False),
            total=len(variant_df),
            desc="Building REF/ALT windows",
            unit="site",
        ):
            chrom = str(row.Chromosome)
            fasta_chrom = chrom_map.get(chrom)
            if fasta_chrom is None:
                continue
            pos = int(row.Position)
            ref = str(row.REF).upper()
            alt = str(row.ALT).upper()
            upstream = fetch_window_with_padding(fasta, fasta_chrom, pos - k, pos - 1)
            downstream = fetch_window_with_padding(
                fasta,
                fasta_chrom,
                pos + len(ref),
                pos + len(ref) + k - 1,
            )
            rows.append(
                {
                    "Chromosome": chrom,
                    "Position": pos,
                    "REF": ref,
                    "ALT": alt,
                    "site_id": f"{chrom}:{pos}:{ref}>{alt}",
                    "VARIANT_TYPE": str(getattr(row, "VARIANT_TYPE", "")).strip() or classify_variant_type(ref, alt),
                    "upstream_seq": upstream,
                    "downstream_seq": downstream,
                    "ref_seq": upstream + ref + downstream,
                    "alt_seq": upstream + alt + downstream,
                }
            )
    return pd.DataFrame(rows)


def embed_sequence_table(seq_df: pd.DataFrame, embedder: CachedSequenceEmbedder) -> tuple[np.ndarray, np.ndarray]:
    ref_embeddings: list[np.ndarray] = []
    alt_embeddings: list[np.ndarray] = []
    for row in progress(
        seq_df.itertuples(index=False),
        total=len(seq_df),
        desc="Embedding sequences",
        unit="site",
    ):
        ref_embeddings.append(embedder.embed_sequence(str(row.ref_seq)))
        alt_embeddings.append(embedder.embed_sequence(str(row.alt_seq)))
    return np.vstack(ref_embeddings), np.vstack(alt_embeddings)


def build_feature_matrix(
    seq_df: pd.DataFrame,
    ref_matrix: np.ndarray,
    alt_matrix: np.ndarray,
    feature_type: str,
) -> tuple[np.ndarray, dict[str, object]]:
    len_ref = seq_df["REF"].astype(str).str.len().to_numpy(dtype=np.float32).reshape(-1, 1)
    len_alt = seq_df["ALT"].astype(str).str.len().to_numpy(dtype=np.float32).reshape(-1, 1)
    len_delta = len_alt - len_ref
    diff_matrix = alt_matrix - ref_matrix

    if feature_type == "alt":
        X = alt_matrix
    elif feature_type == "alt_minus_ref":
        X = diff_matrix
    elif feature_type == "alt_ref_concat":
        X = np.concatenate([alt_matrix, ref_matrix], axis=1)
    elif feature_type == "all":
        X = np.concatenate([alt_matrix, ref_matrix, diff_matrix, len_ref, len_alt, len_delta], axis=1)
    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    meta = {
        "feature_type": feature_type,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }
    return X.astype(np.float32), meta


def filter_classes(
    seq_df: pd.DataFrame,
    min_class_size: int,
    selected_classes: list[str] | None,
) -> pd.DataFrame:
    out = seq_df.copy()
    if selected_classes:
        out = out[out["VARIANT_TYPE"].isin(selected_classes)].copy()
    counts = out["VARIANT_TYPE"].value_counts()
    keep = counts[counts >= int(min_class_size)].index.tolist()
    out = out[out["VARIANT_TYPE"].isin(keep)].copy()
    if out.empty:
        raise ValueError("No samples left after class filtering")
    return out.reset_index(drop=True)


def prepare_labels(
    seq_df: pd.DataFrame,
    task_type: str,
    positive_class: str | None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    require_sklearn()
    from sklearn.preprocessing import LabelEncoder

    labels = seq_df["VARIANT_TYPE"].astype(str).to_numpy()
    unique_labels = sorted(set(labels.tolist()))
    inferred_task_type = task_type
    if task_type == "auto":
        inferred_task_type = "binary" if len(unique_labels) == 2 else "multiclass"

    if inferred_task_type == "binary":
        if len(unique_labels) < 2:
            raise ValueError("Binary classification requires at least 2 classes")
        if positive_class is None:
            positive_class = unique_labels[1] if len(unique_labels) == 2 else unique_labels[0]
        y = (labels == positive_class).astype(np.int64)
        label_df = seq_df[["site_id", "VARIANT_TYPE"]].copy()
        label_df["label_id"] = y
        meta = {
            "task_type": "binary",
            "classes": ["negative", "positive"],
            "positive_class": positive_class,
            "negative_classes": sorted([label for label in unique_labels if label != positive_class]),
            "label_counts": {
                "negative": int((y == 0).sum()),
                "positive": int((y == 1).sum()),
            },
        }
        return label_df, y, meta

    if len(unique_labels) < 3:
        raise ValueError("Multiclass classification requires at least 3 classes after filtering")
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    label_df = seq_df[["site_id", "VARIANT_TYPE"]].copy()
    label_df["label_id"] = y
    meta = {
        "task_type": "multiclass",
        "classes": [str(x) for x in encoder.classes_.tolist()],
        "label_counts": {
            str(label): int(count)
            for label, count in seq_df["VARIANT_TYPE"].value_counts().sort_index().items()
        },
    }
    return label_df, y.astype(np.int64), meta


def build_model(
    model_type: str,
    task_type: str,
    random_state: int,
    class_weight: str | None,
    standardize_features: bool,
):
    require_sklearn()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps: list[tuple[str, object]] = []
    if standardize_features and model_type == "logistic_regression":
        steps.append(("scaler", StandardScaler()))

    if model_type == "logistic_regression":
        estimator = LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            class_weight=class_weight,
            multi_class="auto",
        )
    elif model_type == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    steps.append(("model", estimator))
    _ = task_type
    return Pipeline(steps)


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    require_sklearn()
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def evaluate_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    require_sklearn()
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize

    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "ovr_roc_auc_macro": float(roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")),
        "ovr_roc_auc_weighted": float(roc_auc_score(y_true_bin, y_proba, average="weighted", multi_class="ovr")),
        "ovr_pr_auc_macro": float(average_precision_score(y_true_bin, y_proba, average="macro")),
    }


def build_prediction_table(
    seq_df: pd.DataFrame,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    task_type: str,
    class_names: list[str],
    split_name: str,
) -> pd.DataFrame:
    out = seq_df.iloc[indices][["site_id", "Chromosome", "Position", "REF", "ALT", "VARIANT_TYPE"]].copy()
    out["split"] = split_name
    out["true_label_id"] = y_true
    out["pred_label_id"] = y_pred
    if task_type == "binary":
        out["pred_label"] = np.where(y_pred == 1, class_names[1], class_names[0])
        out["prob_positive"] = y_proba[:, 1]
    else:
        out["pred_label"] = [class_names[idx] for idx in y_pred.tolist()]
        for idx, class_name in enumerate(class_names):
            out[f"prob_{class_name}"] = y_proba[:, idx]
    return out.reset_index(drop=True)


def save_confusion_matrix_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_path: Path,
) -> None:
    require_sklearn()
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.index.name = "true_label"
    cm_df.to_csv(out_path, sep="\t")


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_path: Path,
) -> None:
    require_sklearn()
    from sklearn.metrics import classification_report

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    out_path.write_text(report, encoding="utf-8")


def maybe_plot_binary_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    outdir: Path,
    logger,
) -> dict[str, str | None]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        logger.warning("matplotlib not available, skip ROC/PR plots")
        return {"roc_curve_png": None, "pr_curve_png": None}

    require_sklearn()
    from sklearn.metrics import precision_recall_curve, roc_curve

    roc_path = outdir / "roc_curve.png"
    pr_path = outdir / "pr_curve.png"

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(pr_path, dpi=200)
    plt.close(fig)
    return {"roc_curve_png": str(roc_path), "pr_curve_png": str(pr_path)}


def run_holdout_training(
    X: np.ndarray,
    y: np.ndarray,
    seq_df: pd.DataFrame,
    task_type: str,
    class_names: list[str],
    args,
) -> tuple[dict[str, object], pd.DataFrame, np.ndarray, np.ndarray]:
    require_sklearn()
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(args.test_size), random_state=int(args.random_state))
    train_idx, test_idx = next(splitter.split(X, y))

    model = build_model(
        model_type=args.model_type,
        task_type=task_type,
        random_state=int(args.random_state),
        class_weight=None if args.class_weight == "none" else args.class_weight,
        standardize_features=bool(args.standardize_features),
    )
    model.fit(X[train_idx], y[train_idx])
    y_pred = model.predict(X[test_idx])
    y_proba = model.predict_proba(X[test_idx])

    if task_type == "binary":
        metrics = evaluate_binary(y[test_idx], y_pred, y_proba[:, 1])
    else:
        metrics = evaluate_multiclass(y[test_idx], y_pred, y_proba, class_names)

    metrics["n_train"] = int(len(train_idx))
    metrics["n_test"] = int(len(test_idx))
    pred_df = build_prediction_table(seq_df, test_idx, y[test_idx], y_pred, y_proba, task_type, class_names, "test")
    return metrics, pred_df, train_idx, test_idx


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    seq_df: pd.DataFrame,
    task_type: str,
    class_names: list[str],
    args,
) -> tuple[dict[str, object], pd.DataFrame]:
    require_sklearn()
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=int(args.cv_folds), shuffle=True, random_state=int(args.random_state))
    fold_metrics: list[dict[str, object]] = []
    pred_dfs: list[pd.DataFrame] = []

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        model = build_model(
            model_type=args.model_type,
            task_type=task_type,
            random_state=int(args.random_state) + fold_id,
            class_weight=None if args.class_weight == "none" else args.class_weight,
            standardize_features=bool(args.standardize_features),
        )
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        y_proba = model.predict_proba(X[test_idx])

        if task_type == "binary":
            fold_metric = evaluate_binary(y[test_idx], y_pred, y_proba[:, 1])
        else:
            fold_metric = evaluate_multiclass(y[test_idx], y_pred, y_proba, class_names)
        fold_metric["fold"] = int(fold_id)
        fold_metric["n_train"] = int(len(train_idx))
        fold_metric["n_test"] = int(len(test_idx))
        fold_metrics.append(fold_metric)

        pred_df = build_prediction_table(seq_df, test_idx, y[test_idx], y_pred, y_proba, task_type, class_names, f"fold_{fold_id}")
        pred_dfs.append(pred_df)

    fold_df = pd.DataFrame(fold_metrics)
    summary_metrics = {
        key: float(fold_df[key].mean())
        for key in fold_df.columns
        if key not in {"fold", "n_train", "n_test"}
    }
    summary_metrics.update(
        {
            f"{key}_std": float(fold_df[key].std(ddof=0))
            for key in fold_df.columns
            if key not in {"fold", "n_train", "n_test"}
        }
    )
    summary_metrics["n_folds"] = int(args.cv_folds)
    summary_metrics["n_samples"] = int(len(y))
    return {
        "summary": summary_metrics,
        "fold_metrics": fold_metrics,
    }, pd.concat(pred_dfs, ignore_index=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract REF/ALT windows from VCF+FASTA for TSV sites and train a VARIANT_TYPE classifier")
    parser.add_argument("--config", default=None, help="YAML config file path")
    parser.add_argument("--tsv-path", default=None, help="Input site TSV path; requires Chromosome/Position/VARIANT_TYPE or aliases")
    parser.add_argument("--vcf-path", default=None, help="Input VCF path")
    parser.add_argument("--fasta-path", default=None, help="Reference FASTA path")
    parser.add_argument("--outdir", default=None, help="Output directory")
    parser.add_argument("--k", type=int, default=100, help="Upstream/downstream window length")
    parser.add_argument("--embedder-type", default=None, help="Embedder type, e.g. rice8k")
    parser.add_argument("--model-name-or-path", default=None, help="Local model path or model name")
    parser.add_argument("--device", default="cpu", help="Embedding device, e.g. cpu or cuda")
    parser.add_argument("--pooling", default="mean", choices=["mean", "last"], help="Embedding pooling strategy")
    parser.add_argument("--local-files-only", action="store_true", help="Only load model files from local path/cache")
    parser.add_argument("--task-type", default="auto", choices=["auto", "binary", "multiclass"], help="Classification task type")
    parser.add_argument("--feature-type", default="alt", choices=["alt", "alt_minus_ref", "alt_ref_concat", "all"], help="Feature source built from embeddings")
    parser.add_argument("--model-type", default="logistic_regression", choices=["logistic_regression", "random_forest"], help="Classifier type")
    parser.add_argument("--eval-mode", default="holdout", choices=["holdout", "cv", "auto"], help="Evaluation mode")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test split ratio")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--min-class-size", type=int, default=10, help="Minimum samples required to keep a class")
    parser.add_argument("--selected-classes", default=None, help="Comma-separated class list to keep")
    parser.add_argument("--positive-class", default=None, help="Positive class for binary classification")
    parser.add_argument("--class-weight", default="balanced", choices=["balanced", "none"], help="Class weighting strategy")
    parser.add_argument("--standardize-features", action="store_true", help="Standardize features before linear models")
    return parser


def _load_yaml_config(config_path: str | None) -> dict[str, object]:
    if not config_path:
        return {}
    try:
        import yaml
    except Exception as exc:
        raise ImportError("PyYAML is required to load the config file") from exc

    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_args_with_config(parsed_args) -> argparse.Namespace:
    config = _load_yaml_config(parsed_args.config)
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML mapping/object at top level")

    parser = build_parser()
    defaults = {
        action.dest: action.default
        for action in parser._actions
        if action.dest != "help"
    }
    merged = dict(config)
    for key, default_value in defaults.items():
        cli_value = getattr(parsed_args, key, default_value)
        if cli_value != default_value:
            merged[key] = cli_value
        elif key not in merged:
            merged[key] = default_value

    required = ["tsv_path", "vcf_path", "fasta_path", "outdir", "embedder_type", "model_name_or_path"]
    missing = [key for key in required if not merged.get(key)]
    if missing:
        raise ValueError(f"Missing required parameters after merging config and CLI: {missing}")

    merged["config"] = parsed_args.config
    return argparse.Namespace(**merged)


def main() -> None:
    args = _resolve_args_with_config(build_parser().parse_args())
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = initLogger(outdir / "variant_tsv_alt_pca.log")

    selected_classes = None
    if args.selected_classes:
        selected_classes = [x.strip() for x in str(args.selected_classes).split(",") if x.strip()]

    site_df, positional_keys, allele_keys, variant_type_map = load_tsv_sites(Path(args.tsv_path))
    logger.info(f"Loaded TSV sites: {len(site_df)}")

    variant_df = collect_variants_from_vcf(
        Path(args.vcf_path),
        site_df,
        positional_keys,
        allele_keys,
        variant_type_map,
    )
    if variant_df.empty:
        raise ValueError("No matching VCF variants found for TSV sites")
    logger.info(f"Matched VCF variants: {len(variant_df)}")

    seq_df = add_sequence_windows(variant_df, Path(args.fasta_path), args.k)
    if seq_df.empty:
        raise ValueError("No sequence windows built; please check FASTA chromosome names")
    seq_df = filter_classes(seq_df, min_class_size=int(args.min_class_size), selected_classes=selected_classes)
    seq_df.to_csv(outdir / "variant_sequences.tsv", sep="\t", index=False)

    label_distribution_df = seq_df["VARIANT_TYPE"].value_counts().sort_index().rename_axis("VARIANT_TYPE").reset_index(name="count")
    label_distribution_df.to_csv(outdir / "label_distribution.tsv", sep="\t", index=False)

    base_embedder = UnifiedEmbedder(
        embedder_type=args.embedder_type,
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        pooling=args.pooling,
        local_files_only=bool(args.local_files_only),
    )
    embedder = CachedSequenceEmbedder(outdir / "embedding_cache", base_embedder)

    ref_matrix, alt_matrix = embed_sequence_table(seq_df, embedder)
    diff_matrix = alt_matrix - ref_matrix
    np.save(outdir / "ref_embeddings.npy", ref_matrix)
    np.save(outdir / "alt_embeddings.npy", alt_matrix)
    np.save(outdir / "alt_minus_ref_embeddings.npy", diff_matrix)

    X, feature_meta = build_feature_matrix(seq_df, ref_matrix, alt_matrix, feature_type=args.feature_type)
    np.save(outdir / "features.npy", X)
    (outdir / "feature_meta.json").write_text(json.dumps(feature_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    label_df, y, label_meta = prepare_labels(seq_df, task_type=args.task_type, positive_class=args.positive_class)
    label_df.to_csv(outdir / "labels.tsv", sep="\t", index=False)

    class_names = label_meta["classes"]
    effective_eval_mode = args.eval_mode
    if effective_eval_mode == "auto":
        effective_eval_mode = "cv" if len(y) < 1000 else "holdout"

    if effective_eval_mode == "holdout":
        metrics, pred_df, train_idx, test_idx = run_holdout_training(X, y, seq_df, label_meta["task_type"], class_names, args)
        pred_df.to_csv(outdir / "predictions.tsv", sep="\t", index=False)
        np.save(outdir / "train_indices.npy", train_idx)
        np.save(outdir / "test_indices.npy", test_idx)

        save_confusion_matrix_table(
            y_true=pred_df["true_label_id"].to_numpy(dtype=np.int64),
            y_pred=pred_df["pred_label_id"].to_numpy(dtype=np.int64),
            class_names=class_names,
            out_path=outdir / "confusion_matrix.tsv",
        )
        save_classification_report(
            y_true=pred_df["true_label_id"].to_numpy(dtype=np.int64),
            y_pred=pred_df["pred_label_id"].to_numpy(dtype=np.int64),
            class_names=class_names,
            out_path=outdir / "classification_report.txt",
        )
        plot_outputs = {}
        if label_meta["task_type"] == "binary":
            plot_outputs = maybe_plot_binary_curves(
                y_true=pred_df["true_label_id"].to_numpy(dtype=np.int64),
                y_score=pred_df["prob_positive"].to_numpy(dtype=np.float32),
                outdir=outdir,
                logger=logger,
            )
    else:
        cv_metrics, pred_df = run_cross_validation(X, y, seq_df, label_meta["task_type"], class_names, args)
        pred_df.to_csv(outdir / "predictions.tsv", sep="\t", index=False)
        pd.DataFrame(cv_metrics["fold_metrics"]).to_csv(outdir / "fold_metrics.tsv", sep="\t", index=False)
        save_confusion_matrix_table(
            y_true=pred_df["true_label_id"].to_numpy(dtype=np.int64),
            y_pred=pred_df["pred_label_id"].to_numpy(dtype=np.int64),
            class_names=class_names,
            out_path=outdir / "confusion_matrix.tsv",
        )
        save_classification_report(
            y_true=pred_df["true_label_id"].to_numpy(dtype=np.int64),
            y_pred=pred_df["pred_label_id"].to_numpy(dtype=np.int64),
            class_names=class_names,
            out_path=outdir / "classification_report.txt",
        )
        metrics = cv_metrics["summary"]
        plot_outputs = {}

    summary = {
        "n_input_sites": int(len(site_df)),
        "n_matched_variants": int(len(variant_df)),
        "n_training_sites": int(len(seq_df)),
        "k": int(args.k),
        "embedder_type": str(args.embedder_type),
        "feature_type": str(args.feature_type),
        "model_type": str(args.model_type),
        "task_type": label_meta["task_type"],
        "eval_mode": effective_eval_mode,
        "label_meta": label_meta,
        "metrics": metrics,
        "outputs": {
            "variant_sequences_tsv": str(outdir / "variant_sequences.tsv"),
            "label_distribution_tsv": str(outdir / "label_distribution.tsv"),
            "labels_tsv": str(outdir / "labels.tsv"),
            "features_npy": str(outdir / "features.npy"),
            "feature_meta_json": str(outdir / "feature_meta.json"),
            "predictions_tsv": str(outdir / "predictions.tsv"),
            "metrics_json": str(outdir / "metrics.json"),
            "confusion_matrix_tsv": str(outdir / "confusion_matrix.tsv"),
            "classification_report_txt": str(outdir / "classification_report.txt"),
            "ref_embeddings_npy": str(outdir / "ref_embeddings.npy"),
            "alt_embeddings_npy": str(outdir / "alt_embeddings.npy"),
            "alt_minus_ref_embeddings_npy": str(outdir / "alt_minus_ref_embeddings.npy"),
            **plot_outputs,
        },
    }
    (outdir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
