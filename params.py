from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Iterable


@dataclass(frozen=True)
class CHOParams:
    K: int
    tau: float
    alphas: Tuple[float, ...]   # length K
    cs: Tuple[float, ...]       # typically length 3

    def validate(self) -> None:
        if self.K <= 0:
            raise ValueError("K must be positive")
        if len(self.alphas) != self.K:
            raise ValueError(f"alphas must have length K={self.K}")
        if any(self.alphas[i] >= self.alphas[i+1] for i in range(self.K - 1)):
            raise ValueError("alphas must be strictly increasing")
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if len(self.cs) == 0:
            raise ValueError("cs must be nonempty")
        if any(c <= 0 or c >= 1 for c in self.cs):
            raise ValueError("cs must be in (0,1) for this verifier")
