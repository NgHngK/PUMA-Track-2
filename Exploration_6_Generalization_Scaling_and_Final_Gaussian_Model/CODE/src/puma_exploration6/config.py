from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import re


@dataclass(frozen=True)
class AugmentationConfig:
    geometric: bool = False
    stain_mode: str = "none"  # none | rgb_od | fixed_hed
    stain_strength: float = 0.0
    coordinate_jitter_sigma: float = 0.0
    coordinate_jitter_clip: float = 0.0

    def validate(self) -> None:
        if self.stain_mode not in {"none","rgb_od","fixed_hed"}:
            raise ValueError("unsupported stain_mode")
        if self.stain_mode == "none" and self.stain_strength != 0:
            raise ValueError("stain_strength requires a non-none stain_mode")
        if self.stain_mode != "none" and self.stain_strength <= 0:
            raise ValueError("non-none stain_mode requires positive stain_strength")
        if not 0.0 <= self.stain_strength <= 0.5:
            raise ValueError("stain_strength must be in [0, 0.5]")
        if self.coordinate_jitter_sigma < 0 or self.coordinate_jitter_clip < 0:
            raise ValueError("coordinate jitter values must be non-negative")
        if self.coordinate_jitter_sigma > 0 and self.coordinate_jitter_clip <= 0:
            raise ValueError("positive jitter sigma requires positive jitter clip")
        if self.coordinate_jitter_sigma == 0 and self.coordinate_jitter_clip != 0:
            raise ValueError("coordinate_jitter_clip must be zero when jitter sigma is zero")


@dataclass(frozen=True)
class SamplerConfig:
    mode: str = "inverse"  # natural | tempered | inverse
    alpha: float = 1.0
    num_samples_multiplier: float = 1.0

    def validate(self) -> None:
        if self.mode not in {"natural", "tempered", "inverse"}:
            raise ValueError(f"unsupported sampler mode: {self.mode}")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0,1]")
        if self.mode == "inverse" and self.alpha != 1.0:
            raise ValueError("inverse sampler requires alpha=1")
        if self.mode == "natural" and self.alpha != 0.0:
            raise ValueError("natural sampler requires alpha=0")
        if self.num_samples_multiplier <= 0:
            raise ValueError("num_samples_multiplier must be >0")


@dataclass(frozen=True)
class LossConfig:
    kind: str = "ce"  # ce | logit_adjusted | balanced_softmax
    tau: float = 0.0
    label_smoothing: float = 0.0

    def validate(self) -> None:
        if self.kind not in {"ce", "logit_adjusted", "balanced_softmax"}:
            raise ValueError(f"unsupported loss: {self.kind}")
        if not 0 <= self.tau <= 2:
            raise ValueError("tau must be in [0,2]")
        if not 0 <= self.label_smoothing <= 0.2:
            raise ValueError("label_smoothing must be in [0,0.2]")
        if self.kind == "ce" and self.tau != 0:
            raise ValueError("CE requires tau=0")
        if self.kind == "balanced_softmax" and self.tau != 1:
            raise ValueError("Balanced Softmax requires tau=1")


@dataclass(frozen=True)
class ModelConfig:
    representation: str = "cls"  # cls | target_patch | gaussian | neighborhood | global_local | scalar_global_local
    gaussian_sigma: float = 1.5
    neighborhood_radius: int = 1
    interaction_rank: int = 8
    representation_dropout: float = 0.0
    interaction_dropout: float = 0.0
    tier_a_dropout: float = 0.0
    use_tier_a: bool = True
    scalar_mix_init: float = 0.5

    def validate(self) -> None:
        valid = {"cls", "target_patch", "gaussian", "neighborhood", "global_local", "scalar_global_local"}
        if self.representation not in valid:
            raise ValueError(f"unsupported representation: {self.representation}")
        if self.gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be positive")
        if self.neighborhood_radius < 0 or self.neighborhood_radius > 7:
            raise ValueError("neighborhood_radius outside [0,7]")
        if self.interaction_rank < 1 or self.interaction_rank > 64:
            raise ValueError("interaction_rank outside [1,64]")
        for name, value in [
            ("representation_dropout", self.representation_dropout),
            ("interaction_dropout", self.interaction_dropout),
            ("tier_a_dropout", self.tier_a_dropout),
        ]:
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0,1)")
        if not 0 <= self.scalar_mix_init <= 1:
            raise ValueError("scalar_mix_init must be in [0,1]")


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 64  # physical microbatch
    accumulation_steps: int = 1  # effective batch ~= batch_size * accumulation_steps
    max_epochs: int = 20
    patience: int = 4
    min_delta: float = 1e-4
    scheduler: str = "constant"  # constant | cosine | warmup_cosine
    warmup_fraction: float = 0.05
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    amp: bool = False
    decay_bias_and_vectors: bool = True
    selection_mode: str = "early_stop"  # early_stop | fixed_final

    def validate(self) -> None:
        if self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("invalid lr/weight_decay")
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        if self.batch_size < 1 or self.accumulation_steps < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("batch_size/accumulation_steps/max_epochs/patience must be positive")
        if self.scheduler not in {"constant", "cosine", "warmup_cosine"}:
            raise ValueError("unsupported scheduler")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0,1)")
        if not 0 < self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in (0,1]")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be >0")
        if self.selection_mode not in {"early_stop","fixed_final"}:
            raise ValueError("unsupported selection_mode")


@dataclass(frozen=True)
class EncoderConfig:
    adaptation: str = "frozen"  # frozen | lora
    allow_adaptation: bool = False
    adapter_lr: float = 1e-5
    lora_rank: int = 4
    lora_alpha: float = 8.0
    last_blocks: int = 2

    def validate(self) -> None:
        if self.adaptation not in {"frozen","lora"}:
            raise ValueError("unsupported encoder adaptation")
        if self.adaptation != "frozen" and not self.allow_adaptation:
            raise ValueError("encoder adaptation gate is closed; set allow_adaptation only after Exploration-6 gate approval")
        if self.adapter_lr <= 0 or self.lora_rank < 1 or self.lora_alpha <= 0 or not 1 <= self.last_blocks <= 24:
            raise ValueError("invalid encoder adaptation settings")


@dataclass(frozen=True)
class DecoupledConfig:
    enabled: bool = False
    retrain_epochs: int = 5
    retrain_lr: float = 1e-3
    reset_classifier: bool = True
    sampler: SamplerConfig = field(default_factory=lambda: SamplerConfig(mode="inverse", alpha=1.0))
    loss: LossConfig = field(default_factory=LossConfig)

    def validate(self) -> None:
        if self.retrain_epochs < 1 or self.retrain_lr <= 0:
            raise ValueError("invalid decoupled retraining settings")
        self.sampler.validate()
        self.loss.validate()
        if self.sampler.mode != "natural" and self.loss.kind in {"balanced_softmax", "logit_adjusted"} and self.loss.tau != 0:
            raise ValueError("do not stack decoupled replacement/tempered sampling with prior-adjusted loss without an explicit derived protocol")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    manifest: str
    output_dir: str
    uni2_weights: str | None = None
    cached_features: str | None = None
    cached_tokens: str | None = None
    tier_a: str | None = None
    fov: int = 96
    seed: int = 17
    device: str = "cuda"
    num_workers: int = 0
    selection_metric: str = "roi_macro_f1"
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoupled: DecoupledConfig = field(default_factory=DecoupledConfig)

    def validate(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id cannot be empty")
        if self.experiment_id in {".",".."} or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*",self.experiment_id) is None:
            raise ValueError("experiment_id must be a safe filename token using only letters, digits, dot, underscore or hyphen")
        if self.fov < 16 or self.fov % 2:
            raise ValueError("fov must be an even integer >=16")
        if self.seed < 0 or self.num_workers < 0:
            raise ValueError("invalid seed/workers")
        if self.selection_metric not in {"roi_macro_f1", "macro_f1"}:
            raise ValueError("unsupported selection metric")
        source_count = sum(x is not None for x in (self.uni2_weights, self.cached_features, self.cached_tokens))
        if source_count != 1:
            raise ValueError("exactly one of uni2_weights, cached_features, cached_tokens must be supplied")
        if self.cached_features is not None and self.model.representation != "cls":
            raise ValueError("cached_features supports CLS only; use cached_tokens for local/global-local representations")
        if self.model.use_tier_a and self.tier_a is None:
            raise ValueError("Tier-A enabled but tier_a path is missing")
        if self.sampler.mode != "natural" and self.loss.kind in {"balanced_softmax", "logit_adjusted"} and self.loss.tau != 0:
            raise ValueError("do not stack replacement/tempered sampling with prior-adjusted loss without an explicit derived protocol")
        self.augmentation.validate(); self.sampler.validate(); self.loss.validate(); self.model.validate(); self.optimizer.validate(); self.encoder.validate(); self.decoupled.validate()
        if self.encoder.adaptation != "frozen" and self.uni2_weights is None:
            raise ValueError("encoder adaptation requires raw-image uni2_weights mode")
        if self.uni2_weights is None and (
            self.augmentation.geometric or self.augmentation.stain_strength > 0 or self.augmentation.coordinate_jitter_sigma > 0
        ):
            raise ValueError("stochastic image augmentation requires raw-image UNI2 mode")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ExperimentConfig":
        if not isinstance(d, dict):
            raise TypeError("experiment config must be a JSON object")
        allowed = {
            "experiment_id","manifest","output_dir","uni2_weights","cached_features","cached_tokens","tier_a",
            "fov","seed","device","num_workers","selection_metric","augmentation","sampler","loss","model",
            "optimizer","encoder","decoupled",
        }
        unknown = set(d) - allowed
        if unknown:
            raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
        cfg = ExperimentConfig(
            experiment_id=d["experiment_id"], manifest=d["manifest"], output_dir=d["output_dir"],
            uni2_weights=d.get("uni2_weights"), cached_features=d.get("cached_features"), cached_tokens=d.get("cached_tokens"),
            tier_a=d.get("tier_a"), fov=int(d.get("fov",96)), seed=int(d.get("seed",17)), device=d.get("device","cuda"),
            num_workers=int(d.get("num_workers",0)), selection_metric=d.get("selection_metric","roi_macro_f1"),
            augmentation=AugmentationConfig(**d.get("augmentation",{})), sampler=SamplerConfig(**d.get("sampler",{})),
            loss=LossConfig(**d.get("loss",{})), model=ModelConfig(**d.get("model",{})),
            optimizer=OptimizerConfig(**d.get("optimizer",{})),
            encoder=EncoderConfig(**d.get("encoder",{})),
            decoupled=DecoupledConfig(
                **{k:v for k,v in d.get("decoupled",{}).items() if k not in {"sampler","loss"}},
                sampler=SamplerConfig(**d.get("decoupled",{}).get("sampler", {"mode":"inverse","alpha":1.0})),
                loss=LossConfig(**d.get("decoupled",{}).get("loss", {})),
            ),
        )
        cfg.validate(); return cfg

    @staticmethod
    def load(path: str | Path) -> "ExperimentConfig":
        return ExperimentConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
