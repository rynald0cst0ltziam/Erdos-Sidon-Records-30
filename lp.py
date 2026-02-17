
import numpy as np
import scipy.optimize
from typing import Tuple, Optional, Any, Dict

def solve_lp_with_certificate(
    c: np.ndarray,
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    bounds: Any,
    method: str = "highs"
) -> Dict[str, Any]:
    """
    Wrapper for linprog that extracts dual marginals.
    """
    res = scipy.optimize.linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=method
    )
    
    cert = {
        "status": res.status,
        "success": res.success,
        "fun": float(res.fun) if res.success else None,
        "x": res.x.tolist() if res.success else None,
        "dual_ub": (res.ineqlin.marginals.tolist() if res.ineqlin.marginals is not None else None) if hasattr(res, 'ineqlin') and res.ineqlin is not None else None,
        "dual_eq": (res.eqlin.marginals.tolist() if res.eqlin.marginals is not None else None) if hasattr(res, 'eqlin') and res.eqlin is not None else None,
        "dual_lower": (res.lower.marginals.tolist() if res.lower.marginals is not None else None) if hasattr(res, 'lower') and res.lower is not None else None,
        "dual_upper": (res.upper.marginals.tolist() if res.upper.marginals is not None else None) if hasattr(res, 'upper') and res.upper is not None else None,
        # Matrix info for reconstruction
        "A_ub": A_ub.tolist() if hasattr(A_ub, 'tolist') else list(A_ub),
        "b_ub": b_ub.tolist() if hasattr(b_ub, 'tolist') else list(b_ub),
        "A_eq": A_eq.tolist() if hasattr(A_eq, 'tolist') else list(A_eq),
        "b_eq": b_eq.tolist() if hasattr(b_eq, 'tolist') else list(b_eq),
        "c": c.tolist() if hasattr(c, 'tolist') else list(c),
        "bounds": [(float(lo) if lo is not None else None, float(hi) if hi is not None else None) for (lo, hi) in bounds]
    }
    return cert
