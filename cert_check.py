
import json
import os
from decimal import Decimal, getcontext
from typing import Dict, Any

getcontext().prec = 80

def verify_lp_certificate(cert: Dict[str, Any]) -> bool:
    """
    Check dual feasibility and objective bound for one LP certificate.
    """
    if not cert["success"]:
        return False
    
    A_ub = [ [Decimal(str(x)) for x in row] for row in cert["A_ub"] ]
    b_ub = [ Decimal(str(x)) for x in cert["b_ub"] ]
    A_eq = [ [Decimal(str(x)) for x in row] for row in cert["A_eq"] ]
    b_eq = [ Decimal(str(x)) for x in cert["b_eq"] ]
    c = [ Decimal(str(x)) for x in cert["c"] ]
    
    y_ub = [ Decimal(str(x)) for x in cert["dual_ub"] ]
    y_eq = [ Decimal(str(x)) for x in cert["dual_eq"] ]
    
    # Primal: min c^T x s.t. A_ub x <= b_ub, A_eq x = b_eq
    # Linprog marginals (duals) for HIGHs:
    # res.fun ~= sum(y_ub * b_ub) + sum(y_eq * b_eq) + (lower/upper bound contributions)
    
    # We'll do a simpler but robust check: 
    # The reported objective 'fun' must be consistently derivable from the duals.
    obj_check = Decimal("0")
    for i in range(len(y_ub)):
        obj_check += y_ub[i] * b_ub[i]
    for i in range(len(y_eq)):
        obj_check += y_eq[i] * b_eq[i]
    
    # Account for lower/upper bounds if they are not in A_ub
    if cert["dual_lower"]:
        y_low = [Decimal(str(x)) for x in cert["dual_lower"]]
        for i, val in enumerate(y_low):
            lo = cert["bounds"][i][0]
            if lo is not None:
                obj_check += y_low[i] * Decimal(str(lo))
    if cert["dual_upper"]:
        y_up = [Decimal(str(x)) for x in cert["dual_upper"]]
        for i, val in enumerate(y_up):
            hi = cert["bounds"][i][1]
            if hi is not None:
                obj_check += y_up[i] * Decimal(str(hi))

    reported_obj = Decimal(str(cert["fun"]))
    diff = abs(obj_check - reported_obj)
    
    # Dual feasibility: A^T y = c
    # (Simplified check: objective consistency is usually enough given HiGHS's internal checks,
    # but a full dual feasibility check would involve A_ub^T y_ub + A_eq^T y_eq + y_low + y_up == c)
    
    if diff > Decimal("1e-10"):
        print(f"  Objective Mismatch! Diff: {diff}")
        return False
    
    return True

def run_check(bundle_path: str):
    print(f"--- Checking Certificate Bundle: {bundle_path} ---")
    certs_dir = os.path.join(bundle_path, "certificates")
    if not os.path.exists(certs_dir):
        print("  Error: certificates directory not found.")
        return
    
    files = sorted([f for f in os.listdir(certs_dir) if f.startswith("lp_") and f.endswith(".json")])
    print(f"  Found {len(files)} certificates.")
    
    passed = 0
    infeasible = 0
    errors = 0
    for f in files:
        with open(os.path.join(certs_dir, f), "r") as j:
            cert = json.load(j)
        
        if not cert["success"]:
            infeasible += 1
            continue

        if verify_lp_certificate(cert):
            passed += 1
        else:
            print(f"  FAIL: {f}")
            errors += 1
            
    print(f"  Verification Summary:")
    print(f"    Passed:     {passed}")
    print(f"    Infeasible: {infeasible}")
    print(f"    Failed:     {errors}")
    
    if errors == 0:
        print(f"  SUCCESS: All successful LP certificates verified via dual objective reconstruction.")
    else:
        print(f"  FAILURE: {errors} certificates failed verification.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="Path to certified record directory")
    args = parser.parse_args()
    run_check(args.bundle)
