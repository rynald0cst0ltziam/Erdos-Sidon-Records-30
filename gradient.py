
import numpy as np
import json
from typing import Tuple, List, Dict, Any
from params import CHOParams
from verify import f, s_to_p

def df_da(a: float, b: float, C: float) -> float:
    if a >= C: return 2.0 * (a - C)
    return 0.0

def df_db(a: float, b: float, C: float) -> float:
    if b <= C: return 2.0 * (b - C)
    return 0.0

def get_b1_gradient(tau, alpha, w) -> Dict[str, Any]:
    K = len(alpha) - 2
    # O1 = tau + 1/tau - 2*tau * sum_{j=0}^K (w_{j+1} - w_j) f(alpha_j, alpha_{j+1}, 1)
    
    # dO1/dtau
    term_sum = 0.0
    for j in range(K + 1):
        term_sum += (w[j+1] - w[j]) * f(alpha[j], alpha[j+1], 1.0)
    
    d_tau = 1.0 - 1.0/(tau**2) - 2.0 * term_sum
    
    # dO1/dalpha_k for k=1..K
    d_alphas = [0.0] * (K + 1) # index 1 to K will be used
    for k in range(1, K + 1):
        # alpha_k appears in window [k-1, k] as b and [k, k+1] as a
        term1 = (w[k] - w[k-1]) * df_db(alpha[k-1], alpha[k], 1.0)
        term2 = (w[k+1] - w[k]) * df_da(alpha[k], alpha[k+1], 1.0)
        d_alphas[k] = -2.0 * tau * (term1 + term2)
        
    return {"tau": d_tau, "alphas": d_alphas[1:]}

def get_b2_gradient(tau, alpha, c, s, w) -> Dict[str, Any]:
    K = len(alpha) - 2
    p, _ = s_to_p(s)
    
    # O2 = c*tau + 1/(c*tau) + sum factor_j * (w_{p[j][1]} - w[p[j-1][1]])
    # factor_j = -2*tau/(c^2) * f(alpha_eta_j_1 - alpha_zeta_j, alpha_eta_j - alpha_zeta_j_1, c)
    
    zeta = [None]; eta = [None]; j_zeta = 0; j_eta = 0
    for letter, _ in p:
        if letter == "r": j_eta += 1
        if letter == "q": j_zeta += 1
        zeta.append(j_zeta); eta.append(j_eta)

    # dO2/dtau
    d_tau = c - 1.0/(c * tau**2)
    for j in range(1, len(p)):
        val_f = f(alpha[eta[j]-1] - alpha[zeta[j]], alpha[eta[j]] - alpha[max(zeta[j]-1, 0)], c)
        d_tau += (w[p[j][1]] - w[p[j-1][1]]) * (-2.0 / (c**2) * val_f)
    
    # dO2/dalpha_k
    d_alphas = [0.0] * (K + 1)
    factor_const = -2.0 * tau / (c**2)
    for j in range(1, len(p)):
        w_diff = w[p[j][1]] - w[p[j-1][1]]
        a_idx = eta[j]-1; b_idx = zeta[j] # a term: alpha[a_idx] - alpha[b_idx]
        c_idx = eta[j]; d_idx = max(zeta[j]-1, 0) # b term: alpha[c_idx] - alpha[d_idx]
        
        arg_a = alpha[a_idx] - alpha[b_idx]
        arg_b = alpha[c_idx] - alpha[d_idx]
        
        d_val_da = df_da(arg_a, arg_b, c)
        d_val_db = df_db(arg_a, arg_b, c)
        
        # d/d_alpha_k (alpha_a - alpha_b) = delta(k,a) - delta(k,b)
        if 1 <= a_idx <= K: d_alphas[a_idx] += w_diff * factor_const * d_val_da
        if 1 <= b_idx <= K: d_alphas[b_idx] -= w_diff * factor_const * d_val_da
        if 1 <= c_idx <= K: d_alphas[c_idx] += w_diff * factor_const * d_val_db
        if 1 <= d_idx <= K: d_alphas[d_idx] -= w_diff * factor_const * d_val_db
            
    return {"tau": d_tau, "alphas": d_alphas[1:]}

def compute_full_gradient(p: CHOParams, cert: Dict[str, Any]) -> Dict[str, Any]:
    # Extract duals for z constraints (the objectives)
    # Our objectives were [b1, b2_c1, b2_c2, b2_c3]
    # They were the last 4 constraints in A_ub?
    # No, in multioptimize:
    # 1. Boundary constraints (len(boundary))
    # 2. Objectives (len(objectives))
    
    duals = cert["dual_ub"]
    n_boundary = len(cert["b_ub"]) - (1 + len(p.cs)) # objectives are at the end
    obj_duals = duals[n_boundary:]
    
    w = cert["x"][:-1]
    
    # K indices
    alpha_full = (0.0,) + p.alphas + (float('inf'),)
    
    # Get gradients for each objective
    grads = []
    # 1. b1
    grads.append(get_b1_gradient(p.tau, alpha_full, w))
    # 2. b2_c's
    # We need the 's' for the bottleneck cell.
    # We didn't save 's' in the certificate. 
    # But we have worst_idx. We can re-derive 's'.
    from verify import _final_intersections_and_boundaries
    intersections, base_cells = _final_intersections_and_boundaries(p.cs, p.K)
    indices = intersections[cert.get("worst_idx", 0)]
    s_bottleneck = tuple(base_cells[i][idx_i] for i, idx_i in enumerate(indices))
    
    for i, c in enumerate(p.cs):
        grads.append(get_b2_gradient(p.tau, alpha_full, c, s_bottleneck[i], w))
        
    # Combine using dual weights
    final_grad = {"tau": 0.0, "alphas": [0.0] * p.K}
    # Important: obj_duals are marginals for min(-fun) which is max(z).
    # Since we have z - sum M_ij w_j <= C_i, the dual y_i is >= 0 and sum y_i = 1.
    for y_i, g in zip(obj_duals, grads):
        final_grad["tau"] += y_i * g["tau"]
        for k in range(p.K):
            final_grad["alphas"][k] += y_i * g["alphas"][k]
            
    return final_grad

def refine_step(p_path="best_found.json", learning_rate=1e-5):
    from engine import load_params_json
    import verify
    import json
    
    with open(p_path) as f:
        d = json.load(f)
        best_b_in = d["b_inf"]

    p = load_params_json(p_path)
    print(f"Refining record: b_inf = {best_b_in}")
    
    # 1. Get bottleneck certificate
    out = verify.run_theorem_3_3_with_params(p.K, p.tau, p.alphas, p.cs, quiet=True, return_certs=True)
    worst_idx = out["worst_idx"]
    cert = out["certificates"][worst_idx]
    cert["worst_idx"] = worst_idx # pass it along
    
    # 2. Compute Gradient
    grads = []
    # 1. b1
    alpha_full = (0.0,) + p.alphas + (float('inf'),)
    w = cert["x"][:-1]
    g1 = get_b1_gradient(p.tau, alpha_full, w)
    print(f"  b1 grad: tau={g1['tau']:.6f}")
    
    from verify import _final_intersections_and_boundaries
    intersections, base_cells = _final_intersections_and_boundaries(p.cs, p.K)
    indices = intersections[worst_idx]
    s_bottleneck = tuple(base_cells[i][idx_i] for i, idx_i in enumerate(indices))
    
    for i, c in enumerate(p.cs):
        gi = get_b2_gradient(p.tau, alpha_full, c, s_bottleneck[i], w)
        print(f"  b2_{c} grad: tau={gi['tau']:.6f}")

    grad = compute_full_gradient(p, cert)
    print(f"Total Grad: tau={grad['tau']:.6f}")
    
    # 3. Update (Descent because we want to MINIMIZE b_inf)
    new_tau = p.tau - learning_rate * grad["tau"]
    new_alphas = [a - learning_rate * ga for a, ga in zip(p.alphas, grad["alphas"])]
    
    # 4. Validate and Save
    new_p = CHOParams(p.K, new_tau, tuple(new_alphas), p.cs)
    new_out = verify.run_theorem_3_3_with_params(new_p.K, new_p.tau, new_p.alphas, new_p.cs, quiet=True)
    
    print(f"New bound: {new_out['b_inf']:.10f} (Change: {new_out['b_inf'] - best_b_in:.10f})")
    
    if new_out["b_inf"] < best_b_in:
        # Success!
        from search import save_best
        save_best(new_p, new_out, "refined_found.json")
        print("Improved record saved to refined_found.json")
        return True
    else:
        print("Gradient step did not improve bound (LR too high or local min).")
        return False

if __name__ == "__main__":
    refine_step(learning_rate=1e-6)
