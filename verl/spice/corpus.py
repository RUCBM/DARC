from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple

from datasets import load_dataset


@dataclass(frozen=True)
class SampledDoc:
    text: str
    source: str  # "general" | "math"


class ParquetTextCorpus:
    def __init__(self, parquet_path: str, *, text_key: str = "text", max_rows: int = 10000):
        if not parquet_path:
            raise ValueError("parquet_path is empty")
        self.parquet_path = parquet_path
        self.text_key = text_key
        self.max_rows = int(max_rows) if max_rows is not None else 0

        if os.path.isdir(parquet_path):
            self.ds = load_dataset("parquet", data_dir=parquet_path, split="train")
        elif os.path.isfile(parquet_path):
            self.ds = load_dataset("parquet", data_files=parquet_path, split="train")
        else:
            raise FileNotFoundError(f"Parquet path not found: {parquet_path}")

        if self.text_key not in self.ds.column_names:
            raise KeyError(f"Column `{self.text_key}` not in parquet columns: {self.ds.column_names}")

        if self.max_rows > 0 and len(self.ds) > self.max_rows:
            self.ds = self.ds.select(range(self.max_rows))

    def __len__(self) -> int:
        return len(self.ds)

    def sample_text(self, rng: random.Random) -> str:
        idx = rng.randrange(len(self.ds))
        item = self.ds[int(idx)]
        return str(item[self.text_key])


class MixedCorpusSampler:
    def __init__(
        self,
        *,
        general: Optional[ParquetTextCorpus],
        math: Optional[ParquetTextCorpus],
        mix_math_ratio: float = 0.5,
        seed: int = 1,
        max_doc_per_parquet: int = 10000,

    ):
        if general is None and math is None:
            raise ValueError("At least one corpus must be provided.")
        if not (0.0 <= mix_math_ratio <= 1.0):
            raise ValueError(f"mix_math_ratio must be in [0,1], got {mix_math_ratio}")

        self.general = general
        self.math = math
        self.mix_math_ratio = mix_math_ratio
        self.rng = random.Random(seed)
        self.max_doc_per_parquet = max_doc_per_parquet

    def sample(self) -> SampledDoc:
        if self.general is None:
            return SampledDoc(text=self.math.sample_text(self.rng), source="math")
        if self.math is None:
            return SampledDoc(text=self.general.sample_text(self.rng), source="general")

        use_math = self.rng.random() < self.mix_math_ratio
        if use_math:
            return SampledDoc(text=self.math.sample_text(self.rng), source="math")
        return SampledDoc(text=self.general.sample_text(self.rng), source="general")

    def sample_batch(self, batch_size: int) -> Tuple[list[str], list[str]]:
        texts, sources = [], []
        for _ in range(batch_size):
            s = self.sample()
            texts.append(s.text)
            sources.append(s.source)
        return texts, sources
