"""Load Hugging Face-style pretrained models for embeddings with a unified interface."""

from __future__ import annotations

import os

os.environ["HF_HOME"] = "/share/org/YZWL/yzbsl_zhangchao/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/share/org/YZWL/yzbsl_zhangchao/.cache/huggingface/hub"

import numpy as np
import torch

try:
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoTokenizer,
    )
except Exception as exc:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoModelForMaskedLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    _TRANSFORMERS_ERROR = exc
else:
    _TRANSFORMERS_ERROR = None

try:
    from evo2 import Evo2
except Exception as exc:  # pragma: no cover - optional dependency
    Evo2 = None  # type: ignore[assignment]
    _EVO2_ERROR = exc
else:
    _EVO2_ERROR = None


def _require_transformers() -> None:
    if _TRANSFORMERS_ERROR is not None or AutoTokenizer is None:
        raise ImportError(
            "transformers is required for this embedder. "
            "Install with: pip install transformers"
        ) from _TRANSFORMERS_ERROR


def _require_evo2() -> None:
    if _EVO2_ERROR is not None or Evo2 is None:
        raise ImportError(
            "evo2 is required for this embedder. "
            "Install the evo2 package and its dependencies."
        ) from _EVO2_ERROR


def _resolve_model_path(model_name_or_path: str | None) -> str:
    if model_name_or_path:
        return model_name_or_path
    raise ValueError("model_name_or_path is required.")


def _raise_on_unused_kwargs(kwargs: dict, cls_name: str) -> None:
    if kwargs:
        raise ValueError(f"{cls_name} got unsupported kwargs: {sorted(kwargs.keys())}")


class GENERatorEmbedder:
    """Use GENERator (causal LM) to embed DNA sequences."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        pooling: str = "mean",
        local_files_only: bool = True,
        **kwargs,
    ):
        """
        pooling:
          "last" -> last valid token embedding
          "mean" -> mean embedding across valid tokens
        """
        _require_transformers()
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling must be 'last' or 'mean'")
        self.pooling = pooling

        model_path = _resolve_model_path(model_name_or_path)
        self.device = device
        _raise_on_unused_kwargs(kwargs, "GENERatorEmbedder")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()
        self.max_length = self.model.config.max_position_embeddings

    def embed(self, sequence: str) -> np.ndarray:
        # Preprocess sequence: truncate to multiple of 6 and add BOS token
        # This is specific to GENERator's 6-mer tokenizer
        if isinstance(sequence, str):
            processed_sequence = (
                self.tokenizer.bos_token + sequence[: len(sequence) // 6 * 6]
            )
        else:
            sequence = [
                self.tokenizer.bos_token + seq[: len(seq) // 6 * 6] for seq in sequence
            ]
            processed_sequence = sequence

        self.tokenizer.padding_side = "right"
        inputs = self.tokenizer(
            processed_sequence,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states[-1]  # (B, L, C)
        attention_mask = inputs["attention_mask"]  # (B, L)

        if self.pooling == "last":
            # Get last valid token (EOS) embedding
            last_token_indices = attention_mask.sum(dim=1) - 1
            emb = hidden_states[
                torch.arange(hidden_states.size(0)), last_token_indices, :
            ]
        else:
            # Mean pooling over all tokens
            expanded_mask = (
                attention_mask.unsqueeze(-1)
                .expand(hidden_states.size())
                .to(torch.float32)
            )
            sum_embeddings = torch.sum(hidden_states * expanded_mask, dim=1)
            emb = sum_embeddings / expanded_mask.sum(dim=1)

        return emb.squeeze(0).cpu().float().numpy()


class Evo2Embedder:
    """Use Evo2 to embed DNA sequences."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        pooling: str = "mean",
        local_files_only: bool = True,
        **kwargs,
    ):
        """
        pooling:
          "last" -> last token embedding
          "mean" -> mean embedding across tokens
        """
        _require_evo2()
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling must be 'last' or 'mean'")
        self.pooling = pooling
        self.device = device

        _ = local_files_only
        self.layer_name = kwargs.get("layer_name", None)
        if self.layer_name is None:
            self.layer_name = "blocks.28.mlp.l3"
        kwargs.pop("layer_name", None)
        model_version = kwargs.get("model_version", None)
        kwargs.pop("model_version", None)
        model_path = _resolve_model_path(model_name_or_path)
        if model_version is None:
            model_version = os.path.basename(model_path)
        _raise_on_unused_kwargs(kwargs, "Evo2Embedder")
        if not os.path.exists(model_path):
            raise ValueError(
                f"{model_version} model path does not exist. "
                "Evo2 requires a local model path."
            )

        model_file = os.path.join(model_path, f"{model_version}.pt")
        if not os.path.exists(model_file):
            raise ValueError(f"Model file {model_version}.pt does not exist.")

        self.model = Evo2(model_version, local_path=model_file)

    def embed(self, sequence: str) -> np.ndarray:
        token_ids = (
            torch.tensor(self.model.tokenizer.tokenize(sequence), dtype=torch.int)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            _, outputs = self.model(
                token_ids, return_embeddings=True, layer_names=[self.layer_name]
            )

        hidden = outputs[self.layer_name]  # (B, L, C)
        if self.pooling == "last":
            emb = hidden[0, -1, :]
        else:
            emb = hidden[0].mean(dim=0)

        return emb.cpu().float().numpy()


class NucleotideTransformerEmbedder:
    """Use Nucleotide Transformer (Masked LM) to embed DNA sequences."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        pooling: str = "mean",
        local_files_only: bool = True,
        **kwargs,
    ):
        _require_transformers()
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling must be 'last' or 'mean'")
        self.pooling = pooling

        model_path = _resolve_model_path(model_name_or_path)
        self.device = device
        _raise_on_unused_kwargs(kwargs, "NucleotideTransformerEmbedder")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        )
        self.max_length = self.tokenizer.model_max_length
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()

    def embed(self, sequence: str) -> np.ndarray:
        if isinstance(sequence, str):
            sequence = [sequence]

        tokens_ids = self.tokenizer.batch_encode_plus(
            sequence,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
        )["input_ids"]
        attention_mask = tokens_ids != self.tokenizer.pad_token_id

        with torch.no_grad():
            torch_outs = self.model(
                tokens_ids,
                attention_mask=attention_mask,
                encoder_attention_mask=attention_mask,
                output_hidden_states=True,
            ).to(self.device)

        embeddings = torch_outs["hidden_states"][-1].detach().numpy()
        attention_mask = torch.unsqueeze(attention_mask, dim=-1)

        if self.pooling == "last":
            idx = attention_mask.sum(dim=1) - 1
            emb = embeddings[0, idx, :]
        else:
            emb = torch.sum(attention_mask * embeddings, axis=-2) / torch.sum(
                attention_mask, axis=1
            )

        return emb.squeeze(0).cpu().float().numpy()


class AgrontEmbedder:
    """Use Agro Nucleotide Transformer (Masked LM) to embed DNA sequences."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        pooling: str = "mean",
        local_files_only: bool = True,
        **kwargs,
    ):
        _require_transformers()
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling must be 'last' or 'mean'")
        self.pooling = pooling

        model_path = _resolve_model_path(model_name_or_path)
        self.device = device
        _raise_on_unused_kwargs(kwargs, "AgrontEmbedder")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=local_files_only
        )
        self.max_length = self.tokenizer.model_max_length
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_path, local_files_only=local_files_only
        ).to(self.device)
        self.model.eval()

    def embed(self, sequence: str) -> np.ndarray:
        if isinstance(sequence, str):
            sequence = [sequence]

        tokens_ids = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
        )["input_ids"].to(self.device)
        # torch_batch_tokens = torch.tensor(tokens_ids)
        torch_batch_tokens = tokens_ids.clone()
        attention_mask = torch_batch_tokens != self.tokenizer.pad_token_id

        with torch.no_grad():
            torch_outs = self.model(
                torch_batch_tokens,
                attention_mask=attention_mask,
                encoder_attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Get the final layer embeddings (keep as tensor on device)
        embeddings = torch_outs.hidden_states[-1].detach()  # (B, L, C) on device

        if self.pooling == "last":
            # Get last valid token embedding
            idx = (attention_mask.sum(dim=1) - 1).to(self.device)
            emb = embeddings[0, idx, :]
        else:
            # Mean pooling (keep operations in torch)
            emb = torch.sum(
                attention_mask.unsqueeze(-1) * embeddings, dim=1
            ) / attention_mask.sum(dim=1, keepdim=True)

        # Convert to numpy only at the end
        return emb.squeeze(0).cpu().float().numpy()


class Rice8kEmbedder:
    """Use Rice 1B model (8k context) to embed DNA sequences."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        pooling: str = "mean",
        local_files_only: bool = True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        model_name_or_path : str
            HuggingFace model name or local path.
        device : str
            Device to use. Default: "cuda"
        pooling : str
            Aggregation method: "last" or "mean". Default: "mean"
        local_files_only : bool
            Whether to force local model files for HF loading.
        **kwargs
            Model-specific options:
            - use_flash_attention (default True)
            - torch_dtype (default None)
        """
        _require_transformers()
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling must be 'last' or 'mean'")
        self.pooling = pooling
        self.device = device
        use_flash_attention = kwargs.get("use_flash_attention", None)
        if use_flash_attention is None:
            use_flash_attention = True
        dtype = kwargs.get("dtype", torch.float16)
        kwargs.pop("use_flash_attention", None)
        kwargs.pop("dtype", None)
        _raise_on_unused_kwargs(kwargs, "Rice8kEmbedder")
        self.dtype = dtype
        model_path = _resolve_model_path(model_name_or_path)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        )

        # Prepare model loading arguments
        model_kwargs = {
            "output_hidden_states": True,
            "trust_remote_code": True,
        }

        if self.dtype is not None:
            model_kwargs["dtype"] = self.dtype

        if use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                # Fallback if flash_attention_2 is not available
                pass

        # Load model
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            **model_kwargs,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def embed(self, sequence: str) -> np.ndarray:
        """Generate embedding for a DNA sequence.

        Parameters
        ----------
        sequence : str
            DNA sequence string

        Returns
        -------
        np.ndarray
            Embedding vector from the model's hidden states
        """
        # Tokenize input
        inputs = self.tokenizer(sequence, return_tensors="pt")

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract hidden states
        hidden_states = outputs.hidden_states  # Tuple of hidden states
        last_hidden = hidden_states[-1].clone()  # Last layer
        # last_hidden = hidden_states[-1].clone()
        attention_mask = inputs["attention_mask"]

        if self.pooling == "last":
            # Get last valid token embedding
            idx = attention_mask.sum(dim=1) - 1
            emb = last_hidden[0, idx, :]
        else:
            # Mean pooling
            emb = torch.sum(
                attention_mask.unsqueeze(-1) * last_hidden, dim=1
            ) / attention_mask.sum(dim=1, keepdim=True)

        return emb.squeeze(0).cpu().float().numpy()
