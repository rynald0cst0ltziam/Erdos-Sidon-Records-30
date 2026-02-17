
from decimal import Decimal, ROUND_HALF_UP
from typing import Tuple, Dict, Any

def round_val(x: float, places: int) -> float:
    """
    Rounds a float to a fixed number of decimal places using Decimal for precision.
    This is used to produce 'clean' constants for publication.
    """
    d = Decimal(str(x))
    rounded = d.quantize(Decimal('1e-{}'.format(places)), rounding=ROUND_HALF_UP)
    return float(rounded)

def round_params(p: Any, decimals: Dict[str, int] = None) -> Tuple[float, Tuple[float, ...], Tuple[float, ...]]:
    """
    Produces rounded versions of tau, alphas, and cs.
    Important: This only creates a candidate for certification. 
    The certification layer (certify.py) then rigorously proves the bound for these rounded values.
    """
    if decimals is None:
        from config import DEFAULT_ROUNDING
        decimals = DEFAULT_ROUNDING
    
    tau_r = round_val(p.tau, decimals.get("tau", 6))
    alphas_r = tuple(round_val(a, decimals.get("alphas", 6)) for a in p.alphas)
    cs_r = tuple(round_val(c, decimals.get("cs", 6)) for c in p.cs)
    
    # Enforce strict increasing for alphas after rounding
    alphas_r = sorted(list(alphas_r))
    for i in range(1, len(alphas_r)):
        if alphas_r[i] <= alphas_r[i-1]:
            alphas_r[i] = alphas_r[i-1] + 1e-6
            
    return tau_r, tuple(alphas_r), tuple(cs_r)
