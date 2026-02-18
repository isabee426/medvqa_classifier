"""Configuration loading and schema definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


# ---------------------------------------------------------------------------
# Dataclass‑based config schema
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    dataset_name: str = "vqa-rad"
    hf_dataset_id: str = "flaviagiammarino/vqa-rad"
    local_data_dir: str | None = None
    use_local: bool = False
    splits: list[str] = field(default_factory=lambda: ["train", "test"])
    max_examples_per_split: int | None = None  # None = use all
    label_type: Literal["corruption", "hallucination"] = "corruption"


@dataclass
class CorruptionConfig:
    enabled: bool = True
    seed: int = 42
    # Which corruption strategies to use and their relative probabilities.
    strategies: dict[str, float] = field(default_factory=lambda: {
        "swap_answer": 0.4,
        "swap_image": 0.3,
        "empty_answer": 0.3,
    })
    # Number of corrupted examples per clean example.
    negatives_per_positive: int = 1


@dataclass
class ExtractionConfig:
    model_name_or_path: str = "Salesforce/blip2-opt-2.7b"
    layers: list[int] = field(default_factory=lambda: [4, 8, 12])
    pooling: Literal["mean", "max", "cls"] = "mean"
    segments: list[str] = field(default_factory=lambda: ["vision", "question", "answer"])
    batch_size: int = 1
    device: str = "cuda"
    dtype: str = "float16"  # "float16" | "bfloat16" | "float32"
    output_dir: str = "outputs/features"
    save_every_n: int = 50  # flush to disk every N examples
    use_attention: bool = False  # extract cross-modal attention maps (VADE-inspired)
    attention_reduction: str = "entropy"  # "mean", "max", "entropy"


@dataclass
class ClassifierConfig:
    input_dim: int | None = None  # inferred from features if None
    hidden_dim: int = 512
    num_layers: int = 2  # 1 or 2 hidden layers
    dropout: float = 0.1
    activation: Literal["relu", "gelu"] = "gelu"
    num_classes: int = 1  # 1 → binary (BCE), >1 → multi‑class (CE)


@dataclass
class TrainingConfig:
    label_type: Literal["corruption", "hallucination"] = "corruption"
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    scheduler: Literal["cosine", "step", "none"] = "cosine"
    step_size: int = 10
    step_gamma: float = 0.1
    warmup_steps: int = 0
    loss: Literal["bce", "focal", "ce"] = "bce"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    class_weights: list[float] | None = None
    checkpoint_dir: str = "outputs/checkpoints"
    checkpoint_metric: str = "val_auc"
    patience: int = 0  # early stopping; 0 = disabled
    noise_std: float = 0.0  # Gaussian noise on features during training; 0 = disabled
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda"
    pretrained_checkpoint: str | None = None  # Stage-1 .pt path for warm-start
    freeze_first_n_layers: int = 0  # freeze first N MLP layers during fine-tuning
    features_dirs: list[str] = field(default_factory=list)  # multi-dir loading


@dataclass
class EvalConfig:
    checkpoint_path: str = ""
    features_dir: str = "outputs/features"
    features_dirs: list[str] = field(default_factory=list)  # multi-dir eval (Stage-2)
    splits: list[str] = field(default_factory=lambda: ["test"])
    label_type: Literal["corruption", "hallucination"] = "corruption"
    output_dir: str = "outputs/eval"
    device: str = "cuda"
    batch_size: int = 128
    dump_scores: bool = True


@dataclass
class FullConfig:
    """Top‑level config that composes all sub‑configs."""
    data: DataConfig = field(default_factory=DataConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (in‑place, returns base)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge_dicts(base[k], v)
        else:
            base[k] = v
    return base


def _dataclass_from_dict(cls: type, d: dict) -> Any:
    """Instantiate a dataclass from a (possibly nested) dict, ignoring extra keys."""
    import dataclasses
    fieldnames = {f.name for f in dataclasses.fields(cls)}
    filtered = {}
    for k, v in d.items():
        if k not in fieldnames:
            continue
        ft = cls.__dataclass_fields__[k].type  # type: ignore[attr-defined]
        # If the field is itself a dataclass, recurse.
        origin = getattr(ft, "__origin__", None)
        if isinstance(v, dict) and dataclasses.is_dataclass(ft):
            filtered[k] = _dataclass_from_dict(ft, v)
        else:
            filtered[k] = v
    return cls(**filtered)


def load_config(path: str | Path, overrides: dict | None = None) -> FullConfig:
    """Load a YAML or JSON config file and return a :class:`FullConfig`."""
    path = Path(path)
    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            raw: dict = yaml.safe_load(f) or {}
        else:
            raw = json.load(f)

    if overrides:
        _merge_dicts(raw, overrides)

    cfg = FullConfig()
    if "data" in raw:
        cfg.data = _dataclass_from_dict(DataConfig, raw["data"])
    if "corruption" in raw:
        cfg.corruption = _dataclass_from_dict(CorruptionConfig, raw["corruption"])
    if "extraction" in raw:
        cfg.extraction = _dataclass_from_dict(ExtractionConfig, raw["extraction"])
    if "classifier" in raw:
        cfg.classifier = _dataclass_from_dict(ClassifierConfig, raw["classifier"])
    if "training" in raw:
        cfg.training = _dataclass_from_dict(TrainingConfig, raw["training"])
    if "eval" in raw:
        cfg.eval = _dataclass_from_dict(EvalConfig, raw["eval"])
    return cfg
