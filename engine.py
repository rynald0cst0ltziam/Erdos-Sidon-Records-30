from __future__ import annotations

import json
from typing import Any, Dict, Optional

from params import CHOParams
import verify


def load_params_json(path: str) -> CHOParams:
    with open(path, "r") as f:
        d = json.load(f)

    p = CHOParams(
        K=int(d["K"]),
        tau=float(d["tau"]),
        alphas=tuple(float(x) for x in d["alphas"]),
        cs=tuple(float(x) for x in d["cs"]),
    )
    p.validate()
    return p


def compute_b_bound(
    p: CHOParams,
    *,
    quiet: bool = True,
    cutoff: Optional[float] = None,
    probe_first_idx: Optional[int] = None,
    return_certs: bool = False,
) -> Dict[str, Any]:
    """
    Compute b_inf bound for params p.

    cutoff:
      If provided, verify.py may early-abort once it proves b_inf > cutoff.

    probe_first_idx:
      If provided, verify.py will evaluate that intersection first (fast early-abort).
      Use the baseline worst_idx for best speed.
    """
    p.validate()
    return verify.run_theorem_3_3_with_params(
        p.K,
        p.tau,
        p.alphas,
        p.cs,
        quiet=quiet,
        cutoff=cutoff,
        probe_first_idx=probe_first_idx,
        return_certs=return_certs,
    )
