"""Unified interface for loading pretrained DNA sequence embedding models.

Supports loading models from local directories or HuggingFace Hub.
"""

from __future__ import annotations

from typing import Dict, Type

from embedder import (
    AgrontEmbedder,
    Evo2Embedder,
    GENERatorEmbedder,
    NucleotideTransformerEmbedder,
    Rice8kEmbedder,
)

EMBEDDER_REGISTRY: Dict[str, Type] = {
    "generator": GENERatorEmbedder,
    "evo2": Evo2Embedder,
    "nt": NucleotideTransformerEmbedder,
    "agront": AgrontEmbedder,
    "rice8k": Rice8kEmbedder,
}


def load_pretrained(embedder_type: str, **kwargs):
    """Unified entrypoint for loading different pretrained embedder types.

    Parameters
    ----------
    embedder_type : str
        Type of embedder to load: "generator", "evo2", "nt", or "agront"
    **kwargs
        Additional keyword arguments passed to the embedder constructor.
        Can include:
        - model_dir: Local directory containing model files
        - model_name_or_path: HuggingFace model name or local path
        - device: Device to use ("cpu" or "cuda")
        - choose: Aggregation method ("last" or "mean")

    Returns
    -------
    Embedder instance of the specified type

    Raises
    ------
    ValueError
        If embedder_type is not recognized

    Examples
    --------
    >>> from embedding import load_pretrained
    >>> embedder = load_pretrained("evo2", model_dir="/path/to/models", device="cuda")
    >>> embedding = embedder.embed("ATCGATCG")
    """
    embedder_type = embedder_type.lower()
    if embedder_type not in EMBEDDER_REGISTRY:
        raise ValueError(
            f"Unknown embedder type: {embedder_type}. "
            f"Available: {list(EMBEDDER_REGISTRY.keys())}"
        )
    return EMBEDDER_REGISTRY[embedder_type](**kwargs)


class UnifiedEmbedder:
    """Callable wrapper that always exposes __call__ -> embed().

    This wrapper provides a consistent interface for all embedders,
    making them callable via the __call__ method.

    Parameters
    ----------
    embedder_type : str
        Type of embedder to use
    **kwargs
        Additional keyword arguments passed to load_pretrained

    Examples
    --------
    >>> from embedding import UnifiedEmbedder
    >>> embedder = UnifiedEmbedder("generator", model_name_or_path="path/to/model")
    >>> embedding = embedder("ATCGATCG")
    """

    def __init__(self, embedder_type: str, **kwargs):
        self.embedder = load_pretrained(embedder_type, **kwargs)
        self.embedder_type = embedder_type

    def __call__(self, sequence: str):
        """Generate embedding for a DNA sequence.

        Parameters
        ----------
        sequence : str
            DNA sequence string

        Returns
        -------
        numpy.ndarray
            Embedding vector
        """
        return self.embedder.embed(sequence)
