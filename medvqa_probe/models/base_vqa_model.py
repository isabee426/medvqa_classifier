"""Abstract interface + concrete implementation for a VQA backbone model.

The model loads a HuggingFace vision-language model, registers forward hooks
on specified transformer layers, and returns pooled internal states segmented
by token type (vision / question / answer).

Supports both BLIP-2 (query-token prefix) and BLIP v1 (separate vision
encoder + cross-attention text decoder) architectures.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

from medvqa_probe.utils.config import ExtractionConfig
from medvqa_probe.utils.logging import setup_logging

logger = setup_logging(name=__name__)


@dataclass
class ModelOutput:
    """Return value from a single forward pass."""
    logits: torch.Tensor | None = None
    activations: dict[int, torch.Tensor] = field(default_factory=dict)
    attentions: dict[int, torch.Tensor] = field(default_factory=dict)  # [batch, heads, seq, seq]
    # Token-type masks so the caller can pool per segment.
    vision_mask: torch.Tensor | None = None  # (seq_len,) bool
    question_mask: torch.Tensor | None = None
    answer_mask: torch.Tensor | None = None


class BaseVQAModel(abc.ABC):
    """Interface every VQA backbone must implement."""

    @abc.abstractmethod
    def load(self, cfg: ExtractionConfig) -> None: ...

    @abc.abstractmethod
    def forward(
        self,
        image: Image.Image,
        question: str,
        answer: str,
    ) -> ModelOutput: ...


# ---------------------------------------------------------------------------
# Concrete implementation for HuggingFace VLMs (BLIP v1 / BLIP-2)
# ---------------------------------------------------------------------------

class HFVLMModel(BaseVQAModel):
    """Wraps a HuggingFace VLM for feature extraction.

    Auto-detects BLIP v1 vs BLIP-2 architecture and hooks accordingly:
    - BLIP v1: hooks vision encoder layers + text decoder layers separately.
    - BLIP-2: hooks language model layers (visual query tokens are prefixed).
    """

    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.cfg: ExtractionConfig | None = None
        self._hooks: list[Any] = []
        self._vision_activations: dict[int, torch.Tensor] = {}
        self._text_activations: dict[int, torch.Tensor] = {}
        self._vision_attentions: dict[int, torch.Tensor] = {}
        self._text_attentions: dict[int, torch.Tensor] = {}
        # Cross-attention from text decoder → vision encoder (BLIP v1 only).
        self._cross_attentions: dict[int, torch.Tensor] = {}
        self._is_blip_v1: bool = False

    # ---- lifecycle -------------------------------------------------------

    def load(self, cfg: ExtractionConfig) -> None:
        from transformers import AutoProcessor, AutoModelForImageTextToText as AutoModelForVision2Seq

        self.cfg = cfg
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(cfg.dtype, torch.float16)
        device = cfg.device

        logger.info("Loading model %s (dtype=%s, device=%s)",
                     cfg.model_name_or_path, cfg.dtype, device)

        self.processor = AutoProcessor.from_pretrained(cfg.model_name_or_path)
        self.model = AutoModelForVision2Seq.from_pretrained(
            cfg.model_name_or_path,
            torch_dtype=dtype,
            output_attentions=getattr(cfg, "use_attention", False),
        ).to(device).eval()

        # Detect architecture.
        self._is_blip_v1 = self._detect_blip_v1()
        arch = "BLIP v1" if self._is_blip_v1 else "BLIP-2 / other"
        logger.info("Detected architecture: %s", arch)

        self._register_hooks(cfg.layers)
        logger.info("Model loaded. Hooks registered on layers %s.", cfg.layers)

    def _detect_blip_v1(self) -> bool:
        """Return True if the model is BLIP v1 (BlipForConditionalGeneration)."""
        return (
            hasattr(self.model, "vision_model")
            and hasattr(self.model, "text_decoder")
            and not hasattr(self.model, "language_model")
        )

    # ---- hook registration -----------------------------------------------

    def _register_hooks(self, layer_indices: list[int]) -> None:
        if self._is_blip_v1:
            self._register_blip_v1_hooks(layer_indices)
        else:
            self._register_blip2_hooks(layer_indices)

    def _register_blip_v1_hooks(self, layer_indices: list[int]) -> None:
        """BLIP v1: hook vision encoder layers + text decoder layers.

        Also hooks the crossattention sub-module in each BERT text layer so we
        capture text→vision attention maps (shape: heads × text_len × vis_len).
        """
        vision_layers = self.model.vision_model.encoder.layers
        text_layers = self.model.text_decoder.bert.encoder.layer

        for idx in layer_indices:
            # Vision encoder hook.
            if idx < len(vision_layers):
                handle = vision_layers[idx].register_forward_hook(
                    self._make_hook(idx, self._vision_activations, self._vision_attentions)
                )
                self._hooks.append(handle)
            else:
                logger.warning("Vision encoder has %d layers, skipping layer %d.",
                               len(vision_layers), idx)
            # Text decoder full-layer hook (hidden states).
            if idx < len(text_layers):
                handle = text_layers[idx].register_forward_hook(
                    self._make_hook(idx, self._text_activations, self._text_attentions)
                )
                self._hooks.append(handle)
                # Cross-attention hook: text tokens attending to vision tokens.
                layer = text_layers[idx]
                if hasattr(layer, "crossattention"):
                    handle = layer.crossattention.register_forward_hook(
                        self._make_hook(idx, {}, self._cross_attentions)
                    )
                    self._hooks.append(handle)
                else:
                    logger.warning("Text layer %d has no crossattention sub-module.", idx)
            else:
                logger.warning("Text decoder has %d layers, skipping layer %d.",
                               len(text_layers), idx)

    def _register_blip2_hooks(self, layer_indices: list[int]) -> None:
        """BLIP-2: hook the language model's transformer layers."""
        lm = self._get_language_model()
        layers = self._get_layers(lm)
        for idx in layer_indices:
            if idx >= len(layers):
                logger.warning("Requested layer %d but model has %d layers; skipping.",
                               idx, len(layers))
                continue
            handle = layers[idx].register_forward_hook(
                self._make_hook(idx, self._text_activations, self._text_attentions)
            )
            self._hooks.append(handle)

    @staticmethod
    def _make_hook(
        layer_idx: int,
        hidden_storage: dict[int, torch.Tensor],
        attn_storage: dict[int, torch.Tensor],
    ):
        def hook_fn(_module: Any, _input: Any, output: Any) -> None:
            if isinstance(output, tuple):
                hidden_storage[layer_idx] = output[0].detach()
                # Attention weights are typically the second element when output_attentions=True.
                if len(output) > 1 and output[1] is not None and isinstance(output[1], torch.Tensor):
                    attn_storage[layer_idx] = output[1].detach()
            else:
                hidden_storage[layer_idx] = output.detach()
        return hook_fn

    def _get_language_model(self) -> Any:
        model = self.model
        if hasattr(model, "language_model"):
            return model.language_model
        if hasattr(model, "model"):
            return model.model
        return model

    def _get_layers(self, lm: Any) -> Any:
        if hasattr(lm, "model") and hasattr(lm.model, "decoder") and hasattr(lm.model.decoder, "layers"):
            return lm.model.decoder.layers
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        if hasattr(lm, "transformer") and hasattr(lm.transformer, "h"):
            return lm.transformer.h
        for _name, child in lm.named_modules():
            if "layers" in _name and hasattr(child, "__len__"):
                return child
        raise RuntimeError("Cannot locate transformer layers in the language model.")

    # ---- forward ---------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        image: Image.Image,
        question: str,
        answer: str,
    ) -> ModelOutput:
        assert self.cfg is not None and self.model is not None
        self._vision_activations.clear()
        self._text_activations.clear()
        self._vision_attentions.clear()
        self._text_attentions.clear()
        self._cross_attentions.clear()
        device = self.cfg.device

        prompt = f"Question: {question} Answer: {answer}"
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        forward_kwargs: dict[str, Any] = dict(inputs)
        if getattr(self.cfg, "use_attention", False):
            forward_kwargs["output_attentions"] = True
        outputs = self.model(**forward_kwargs)
        logits = outputs.logits if hasattr(outputs, "logits") else None

        if self._is_blip_v1:
            return self._build_blip_v1_output(inputs, question, answer, logits)
        else:
            return self._build_blip2_output(inputs, question, answer, logits)

    # ---- BLIP v1 output --------------------------------------------------

    def _build_blip_v1_output(
        self, inputs: Any, question: str, answer: str, logits: Any,
    ) -> ModelOutput:
        """For BLIP v1: vision activations come from the vision encoder,
        text activations come from the text decoder. We merge them into
        a unified representation using sentinel masks."""

        # Text decoder sequence length.
        text_seq_len = inputs["input_ids"].shape[1]
        # Vision encoder sequence length (number of patch tokens).
        vis_seq_len = 0
        if self._vision_activations:
            first = next(iter(self._vision_activations.values()))
            vis_seq_len = first.shape[1]

        total_len = vis_seq_len + text_seq_len
        device = inputs["input_ids"].device

        # Build masks over the virtual concatenated sequence [vision ; text].
        vision_mask = torch.zeros(total_len, dtype=torch.bool, device=device)
        question_mask = torch.zeros(total_len, dtype=torch.bool, device=device)
        answer_mask = torch.zeros(total_len, dtype=torch.bool, device=device)

        vision_mask[:vis_seq_len] = True

        q_ids = self.processor.tokenizer(question, add_special_tokens=False)["input_ids"]
        a_ids = self.processor.tokenizer(answer, add_special_tokens=False)["input_ids"]

        text_offset = vis_seq_len
        q_end = min(text_offset + len(q_ids) + 3, total_len)
        a_start = q_end
        a_end = min(a_start + len(a_ids) + 3, total_len)

        question_mask[text_offset:q_end] = True
        answer_mask[a_start:a_end] = True
        if a_end < total_len:
            answer_mask[a_end:] = True

        # Merge [vision_encoder ; text_decoder] hidden states per layer.
        merged: dict[int, torch.Tensor] = {}
        for layer_idx in self.cfg.layers:
            v_act = self._vision_activations.get(layer_idx)
            t_act = self._text_activations.get(layer_idx)
            parts = []
            if v_act is not None:
                parts.append(v_act.squeeze(0))
            if t_act is not None:
                parts.append(t_act.squeeze(0))
            if parts:
                merged[layer_idx] = torch.cat(parts, dim=0).unsqueeze(0)

        # Merge attention maps: use cross-attention (text→vision) if available,
        # falling back to text self-attention otherwise.
        merged_attn: dict[int, torch.Tensor] = {}
        for layer_idx in self.cfg.layers:
            cross_attn = self._cross_attentions.get(layer_idx)
            if cross_attn is not None:
                merged_attn[layer_idx] = cross_attn
            else:
                t_attn = self._text_attentions.get(layer_idx)
                if t_attn is not None:
                    merged_attn[layer_idx] = t_attn

        return ModelOutput(
            logits=logits,
            activations=merged,
            attentions=merged_attn,
            vision_mask=vision_mask,
            question_mask=question_mask,
            answer_mask=answer_mask,
        )

    # ---- BLIP-2 output ---------------------------------------------------

    def _build_blip2_output(
        self, inputs: Any, question: str, answer: str, logits: Any,
    ) -> ModelOutput:
        """For BLIP-2: visual query tokens are prefixed in the text sequence."""
        seq_len = inputs["input_ids"].shape[1]
        device = inputs["input_ids"].device

        num_vision = getattr(self.model.config, "num_query_tokens", 32)
        num_vision = min(num_vision, seq_len)

        q_ids = self.processor.tokenizer(question, add_special_tokens=False)["input_ids"]
        a_ids = self.processor.tokenizer(answer, add_special_tokens=False)["input_ids"]

        text_start = num_vision
        q_end = min(text_start + len(q_ids) + 5, seq_len)
        a_start = q_end
        a_end = min(a_start + len(a_ids) + 5, seq_len)

        vision_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        question_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        answer_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

        vision_mask[:num_vision] = True
        question_mask[text_start:q_end] = True
        answer_mask[a_start:a_end] = True
        if a_end < seq_len:
            answer_mask[a_end:] = True

        return ModelOutput(
            logits=logits,
            activations=dict(self._text_activations),
            attentions=dict(self._text_attentions),
            vision_mask=vision_mask,
            question_mask=question_mask,
            answer_mask=answer_mask,
        )


# ---------------------------------------------------------------------------
# Hidden state pooling
# ---------------------------------------------------------------------------

def pool_activations(
    activations: dict[int, torch.Tensor],
    vision_mask: torch.Tensor,
    question_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    layers: list[int],
    segments: list[str],
    pooling: str = "mean",
) -> np.ndarray:
    """Pool per-layer, per-segment activations and concatenate into a 1-D vector."""
    parts: list[np.ndarray] = []
    segment_masks = {
        "vision": vision_mask,
        "question": question_mask,
        "answer": answer_mask,
    }

    for layer_idx in layers:
        if layer_idx not in activations:
            continue
        hidden = activations[layer_idx].squeeze(0)  # (seq_len, hidden_dim)
        for seg_name in segments:
            mask = segment_masks[seg_name]
            selected = hidden[mask]
            if selected.numel() == 0:
                pooled = torch.zeros(hidden.shape[-1], device=hidden.device)
            elif pooling == "mean":
                pooled = selected.float().mean(dim=0)
            elif pooling == "max":
                pooled = selected.float().max(dim=0).values
            elif pooling == "cls":
                pooled = selected[0].float()
            else:
                pooled = selected.float().mean(dim=0)
            parts.append(pooled.cpu().numpy())

    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)


# ---------------------------------------------------------------------------
# Attention pooling (VADE-inspired)
# ---------------------------------------------------------------------------

def pool_attention_maps(
    attentions: dict[int, torch.Tensor],
    vision_mask: torch.Tensor,
    question_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    layers: list[int],
    reduction: str = "mean",
) -> np.ndarray:
    """Pool cross-modal attention scores per segment pair.

    For each layer, extracts three grounding signals:
    - question->vision: how strongly question tokens attend to visual tokens
    - answer->vision:   how strongly answer tokens attend to visual tokens
    - answer->question: how strongly answer tokens attend to question tokens

    With reduction="entropy", also adds attention entropy (uncertainty signal).

    Attention tensor shape: (batch, num_heads, seq_len, seq_len)
    Returns a 1-D numpy array of shape (n_layers * n_signals,).
    """
    parts: list[float] = []

    # Derive the vision-sequence length from vision_mask (it occupies a contiguous
    # prefix of the virtual concatenated sequence).
    total_len = vision_mask.shape[0]
    vis_seq_len = int(vision_mask.sum().item())
    text_seq_len = total_len - vis_seq_len

    # Text-local masks (relative to the text decoder's own token sequence).
    text_question_mask = question_mask[vis_seq_len:]  # length = text_seq_len
    text_answer_mask = answer_mask[vis_seq_len:]      # length = text_seq_len

    for layer_idx in layers:
        if layer_idx not in attentions:
            # Pad with zeros if attention not available for this layer.
            n_signals = 4 if reduction == "entropy" else 3
            parts.extend([0.0] * n_signals)
            continue

        attn = attentions[layer_idx].squeeze(0).float()  # (heads, rows, cols)
        attn_mean = attn.mean(dim=0)  # (rows, cols)

        rows, cols = attn_mean.shape

        # Choose the right row/column masks depending on the attention tensor shape.
        # Case A: text self-attention  → rows=cols=text_seq_len
        # Case B: text cross-attention → rows=text_seq_len, cols=vis_seq_len
        # Case C: combined / other     → rows=cols=total_len (use original masks)
        if rows == text_seq_len:
            row_q_mask = text_question_mask
            row_a_mask = text_answer_mask
            if cols == vis_seq_len:
                # Cross-attention: all columns are vision tokens.
                col_v_mask = torch.ones(cols, dtype=torch.bool, device=vision_mask.device)
                col_q_mask = torch.zeros(cols, dtype=torch.bool, device=vision_mask.device)
            else:
                # Self-attention: columns are also text tokens.
                col_v_mask = torch.zeros(cols, dtype=torch.bool, device=vision_mask.device)
                col_q_mask = text_question_mask[:cols] if cols <= text_seq_len else text_question_mask
        elif rows == total_len:
            row_q_mask = question_mask
            row_a_mask = answer_mask
            col_v_mask = vision_mask[:cols] if cols <= total_len else vision_mask
            col_q_mask = question_mask[:cols] if cols <= total_len else question_mask
        else:
            # Unknown shape — skip this layer.
            n_signals = 4 if reduction == "entropy" else 3
            parts.extend([0.0] * n_signals)
            continue

        # Extract cross-segment attention windows.
        # attn_mean[i, j] = how much token i attends to token j.
        q_rows = attn_mean[row_q_mask]  # (n_q, cols)
        a_rows = attn_mean[row_a_mask]  # (n_a, cols)

        if q_rows.numel() > 0 and col_v_mask.any():
            q_to_v = q_rows[:, col_v_mask].mean().item()
        else:
            q_to_v = 0.0

        if a_rows.numel() > 0 and col_v_mask.any():
            a_to_v = a_rows[:, col_v_mask].mean().item()
        else:
            a_to_v = 0.0

        if a_rows.numel() > 0 and col_q_mask.any():
            a_to_q = a_rows[:, col_q_mask].mean().item()
        else:
            a_to_q = 0.0

        parts.extend([q_to_v, a_to_v, a_to_q])

        if reduction == "entropy":
            # Attention entropy over visual tokens: low entropy = confident grounding.
            if q_rows.numel() > 0 and col_v_mask.any():
                q_vis_attn = q_rows[:, col_v_mask]  # (n_q, n_v)
                # Normalize to probability distribution over visual tokens.
                q_vis_attn = q_vis_attn / (q_vis_attn.sum(dim=-1, keepdim=True) + 1e-10)
                entropy = -(q_vis_attn * torch.log(q_vis_attn + 1e-10)).sum(dim=-1).mean().item()
            else:
                entropy = 0.0
            parts.append(entropy)

    return np.array(parts, dtype=np.float32)
