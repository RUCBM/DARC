from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Optional

from ..trainer.config import PPOConfig


@dataclass
class SpiceCorpusConfig:
    general_parquet: str = os.getenv("SPICE_GENERAL_PARQUET", "")
    math_parquet: str = os.getenv("SPICE_MATH_PARQUET", "")
    text_key: str = "text"
    mix_math_ratio: float = 0.5
    max_doc_per_parquet: int = 8000
    seed: int = 1


@dataclass
class SpiceLoopConfig:
    total_steps: int = 256
    batch_size_b: int = 128
    challenger_attempts_n: int = 8
    group_size_g: int = 8
    invalid_penalty_rho: float = -1.0
    min_valid_tasks_per_step: int = 1
    max_challenger_resample_rounds: int = 1
    seed: int = 1


@dataclass
class SpiceRewardConfig:
    challenger_target_variance: float = 0.25
    challenger_sigma: float = 0.1


@dataclass
class SpicePromptConfig:
    max_doc_chars: int = 20000
    max_question_chars: int = 2000
    challenger_truncation: str = "left"
    reasoner_truncation: str = "right"
    challenger_answer_type: str = "integer"  # {"integer","float"}


@dataclass
class SpiceConfig:
    corpus: SpiceCorpusConfig = field(default_factory=SpiceCorpusConfig)
    loop: SpiceLoopConfig = field(default_factory=SpiceLoopConfig)
    reward: SpiceRewardConfig = field(default_factory=SpiceRewardConfig)
    prompts: SpicePromptConfig = field(default_factory=SpicePromptConfig)


@dataclass
class SpiceRunConfig:
    """Top-level config for `python -m verl.spice.main`."""

    ppo: PPOConfig = field(default_factory=PPOConfig)
    spice: SpiceConfig = field(default_factory=SpiceConfig)
    # Optional: explicitly set a different tokenizer path for prompt building
    tokenizer_path: Optional[str] = None

    def deep_post_init(self) -> None:
        self.ppo.deep_post_init()
