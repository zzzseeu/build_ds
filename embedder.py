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


class GENERatorEmbedder:
    """Use GENERator (causal LM) to embed DNA sequences."""

    def __init__(
        self,
        model_dir: str | None = None,
        model_version: str = "GENERator-eukaryote-1.2b-base",
        model_name_or_path: str | None = None,
        device: str = "cpu",
        pooling: str = "last",
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

        if model_name_or_path:
            model_path = model_name_or_path
            verify_local = False
        elif model_dir:
            model_path = os.path.join(model_dir, model_version)
            verify_local = True
        else:
            raise ValueError("Provide model_name_or_path or model_dir + model_version.")
        self.device = device

        if verify_local and not os.path.exists(model_path):
            raise ValueError(
                "Model or tokenizer path does not exist. "
                "If using a remote repo, pass model_name_or_path."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
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
        model_dir: str | None = None,
        model_version: str = "evo2_7b",
        model_name_or_path: str | None = None,
        device: str = "cpu",
        pooling: str = "last",
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

        if model_name_or_path:
            model_path = model_name_or_path
        elif model_dir:
            model_path = os.path.join(model_dir, model_version)
        else:
            raise ValueError("Provide model_name_or_path or model_dir + model_version.")
        if not os.path.exists(model_path):
            raise ValueError(
                f"{model_version} model path does not exist. "
                "Evo2 requires a local model path."
            )

        model_file = os.path.join(model_path, f"{model_version}.pt")
        if not os.path.exists(model_file):
            raise ValueError(f"Model file {model_version}.pt does not exist.")

        self.model = Evo2(model_version, local_path=model_file)

    def embed(self, sequence: str, layer_name: str = "blocks.28.mlp.l3") -> np.ndarray:
        token_ids = (
            torch.tensor(self.model.tokenizer.tokenize(sequence), dtype=torch.int)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            _, outputs = self.model(
                token_ids, return_embeddings=True, layer_names=[layer_name]
            )

        hidden = outputs[layer_name]  # (B, L, C)
        if self.pooling == "last":
            emb = hidden[0, -1, :]
        else:
            emb = hidden[0].mean(dim=0)

        return emb.cpu().float().numpy()


class NucleotideTransformerEmbedder:
    """Use Nucleotide Transformer (Masked LM) to embed DNA sequences."""

    def __init__(
        self,
        model_dir: str | None = None,
        model_version: str = "nucleotide-transformer-2.5b-multi-species",
        model_name_or_path: str | None = None,
        device: str = "cpu",
        pooling: str = "mean",
    ):
        _require_transformers()
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling must be 'last' or 'mean'")
        self.pooling = pooling

        if model_name_or_path:
            model_path = model_name_or_path
            verify_local = False
        elif model_dir:
            model_path = os.path.join(model_dir, model_version)
            verify_local = True
        else:
            raise ValueError("Provide model_name_or_path or model_dir + model_version.")
        self.device = device

        if verify_local and not os.path.exists(model_path):
            raise ValueError(
                "Model or tokenizer path does not exist. "
                "If using a remote repo, pass model_name_or_path."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = self.tokenizer.model_max_length
        self.model = AutoModelForMaskedLM.from_pretrained(model_path).to(self.device)
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
        model_dir: str | None = None,
        model_version: str = "agro-nucleotide-transformer-1b",
        model_name_or_path: str | None = None,
        device: str = "cpu",
        pooling: str = "last",
    ):
        _require_transformers()
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling must be 'last' or 'mean'")
        self.pooling = pooling

        if model_name_or_path:
            model_path = model_name_or_path
            verify_local = False
        elif model_dir:
            model_path = os.path.join(model_dir, model_version)
            verify_local = True
        else:
            raise ValueError("Provide model_name_or_path or model_dir + model_version.")
        self.device = device

        if verify_local and not os.path.exists(model_path):
            raise ValueError(
                "Model or tokenizer path does not exist. "
                "If using a remote repo, pass model_name_or_path."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        self.max_length = self.tokenizer.model_max_length
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_path, local_files_only=True
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
        torch_batch_tokens = torch.tensor(tokens_ids)
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
        model_dir: str | None = None,
        model_version: str = "rice_1B_stage2_8k_hf",
        model_name_or_path: str | None = None,
        device: str = "cuda",
        pooling: str = "last",
        use_flash_attention: bool = True,
        torch_dtype=None,
    ):
        """
        Parameters
        ----------
        model_dir : str | None
            Local directory containing model files
        model_version : str
            Model version/directory name. Default: "rice_1B_stage2_8k_hf"
        model_name_or_path : str | None
            HuggingFace model name or path (takes precedence over model_dir)
        device : str
            Device to use. Default: "cuda"
        pooling : str
            Aggregation method: "last" or "mean". Default: "last"
        use_flash_attention : bool
            Whether to use flash_attention_2. Default: True
        torch_dtype : torch.dtype | None
            Data type for the model (e.g., torch.bfloat16). Default: None (auto)
        """
        _require_transformers()
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling must be 'last' or 'mean'")
        self.pooling = pooling
        self.device = device
        self.torch_dtype = torch_dtype

        if model_name_or_path:
            model_path = model_name_or_path
            verify_local = False
        elif model_dir:
            model_path = os.path.join(model_dir, model_version)
            verify_local = True
        else:
            raise ValueError("Provide model_name_or_path or model_dir + model_version.")

        if verify_local and not os.path.exists(model_path):
            raise ValueError(
                "Model or tokenizer path does not exist. "
                "If using a remote repo, pass model_name_or_path."
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Prepare model loading arguments
        model_kwargs = {
            "output_hidden_states": True,
            "trust_remote_code": True,
        }

        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        if use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                # Fallback if flash_attention_2 is not available
                pass

        # Load model
        self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
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
        last_hidden = torch.tensor(hidden_states[-1])  # Last layer
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
