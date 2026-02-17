#!/usr/bin/env python3
from __future__ import annotations

import time
import itertools as it
from functools import cache
from typing import Iterable, Tuple, Dict, Any, Optional, Sequence, List

import numpy as np
import scipy.optimize
import multiprocessing
from concurrent.futures import ProcessPoolExecutor


# -----------------------------
# Core helper + lemma bounds
# -----------------------------


_worker_data = {}

def init_worker(indices, cells):
    global _worker_data
    _worker_data['indices'] = indices
    _worker_data['cells'] = cells

def bound_worker(args):
    idx, K, cs, b1_c, tau, alpha, cutoff, return_certs = args
    indices = _worker_data['indices'][idx]
    base_cells = _worker_data['cells']
    
    cell_intersection = tuple(base_cells[i][idx_i] for i, idx_i in enumerate(indices))
    boundary = tuple(
        bound_ineq
        for s, c in zip(cell_intersection, cs)
        for bound_ineq in s_to_boundary(s, c)
    )
    res = process_intersection((idx, cell_intersection, boundary, K, cs, b1_c, tau, alpha, cutoff, return_certs))
    return res

def f(a: float, b: float, C: float) -> float:
    if a >= C:
        return (a - C) ** 2
    if b <= C:
        return (b - C) ** 2
    return 0.0


def b1(tau: float, alpha: Sequence[float], w: Sequence[float]) -> float:
    return (
        tau + 1.0 / tau
        - 2.0 * tau * sum(
            (w[j + 1] - w[j]) * f(alpha[j], alpha[j + 1], 1.0)
            for j in range(len(alpha) - 1)
        )
    )


def b2(tau: float, alpha: Sequence[float], c: float, w: Sequence[float]) -> float:
    p = []
    for wj in w:
        if 0.0 <= wj <= 1.0:
            p.append((wj, "r"))
        if 0.0 <= wj + c <= 1.0:
            p.append((wj + c, "q"))
    p.sort()

    zeta = [None]
    eta = [None]
    j_zeta = 0
    j_eta = 0
    for _, letter in p:
        if letter == "r":
            j_eta += 1
        if letter == "q":
            j_zeta += 1
        zeta.append(j_zeta)
        eta.append(j_eta)

    return (
        c * tau + 1.0 / (c * tau)
        - 2.0 * tau / (c ** 2)
        * sum(
            (p[j][0] - p[j - 1][0])
            * f(
                alpha[eta[j] - 1] - alpha[zeta[j]],
                alpha[eta[j]] - alpha[max(zeta[j] - 1, 0)],
                c,
            )
            for j in range(1, len(p))
        )
    )


# -----------------------------
# LP / cell machinery
# -----------------------------

def is_feasible(K: int, bound_ineq: Sequence[Tuple[int, int, float]]) -> bool:
    """
    Check feasibility of difference constraints w_y - w_x <= bound
    using Bellman-Ford algorithm. Faster than calling a general LP solver.
    """
    n = K + 2
    # dist[i] is the potential of node i
    dist = [0.0] * n
    
    # Standard difference constraints w_y - w_x <= b
    # We also have w_0 = 0, w_{K+1} = 1
    # w_0 <= 0 and w_0 >= 0 is implied by dist[0]=0 and not having negative cycles
    # w_{K+1} <= 1 and w_{K+1} >= 1
    
    edges = list(bound_ineq)
    edges.append((0, K + 1, 1.0))   # w_{K+1} - w_0 <= 1
    edges.append((K + 1, 0, -1.0))  # w_0 - w_{K+1} <= -1  => w_{K+1} >= 1
    
    # Add bounds 0 <= w_i <= 1
    for i in range(1, K + 1):
        edges.append((0, i, 1.0))   # w_i - w_0 <= 1
        edges.append((i, 0, 0.0))   # w_0 - w_i <= 0 => w_i >= 0

    # Bellman-Ford: run up to n iterations
    for i in range(n):
        changed = False
        for x, y, bound in edges:
            if dist[y] > dist[x] + bound + 1e-14:
                dist[y] = dist[x] + bound
                changed = True
        if not changed:
            return True
    return False


@cache
def s_to_p(s: Tuple[int, ...]):
    p = [("r", j) for j in range(len(s))]
    end_q = None
    for j, q_pos in enumerate(s):
        if q_pos >= len(s) - 1:
            end_q = j
            break
        p.insert(p.index(("r", q_pos + 1)), ("q", j))
    assert end_q is not None
    return p, end_q


@cache
def s_to_boundary(s: Tuple[int, ...], c: float):
    p, end_q = s_to_p(s)
    bound_ineq = []
    for j in range(len(p) - 1):
        x = p[j + 1][1]
        y = p[j][1]
        bound = 0.0
        if p[j][0] == "q":
            bound -= c
        if p[j + 1][0] == "q":
            bound += c
        bound_ineq.append((x, y, bound))
    bound_ineq.append((end_q, len(s) - 1, c))
    return tuple(bound_ineq)


def _feasible_cells(c: float, K: int):
    for s in it.product(*(range(j, K + 2) for j in range(K + 2))):
        if not all(s[j] <= s[j + 1] for j in range(K + 1)):
            continue
        bound_ineq = s_to_boundary(tuple(s), float(c))
        if is_feasible(K, bound_ineq):
            yield tuple(s)


@cache
def feasible_cells(c: float, K: int):
    return list(_feasible_cells(float(c), int(K)))


@cache
def b1_coeffs(tau: float, alpha: Tuple[float, ...]):
    K = len(alpha) - 2
    out = [0.0] * (K + 1) + [tau + 1.0 / tau]
    for j in range(K + 1):
        delta = -2.0 * tau * f(alpha[j], alpha[j + 1], 1.0)
        out[j + 1] += delta
        out[j] -= delta
    return tuple(out)


@cache
def s_to_b2_coeffs(s: Tuple[int, ...], tau: float, alpha: Tuple[float, ...], c: float):
    K = len(alpha) - 2
    p, _end_q = s_to_p(s)

    zeta = [None]
    eta = [None]
    j_zeta = 0
    j_eta = 0
    for letter, _ in p:
        if letter == "r":
            j_eta += 1
        if letter == "q":
            j_zeta += 1
        zeta.append(j_zeta)
        eta.append(j_eta)

    out = [0.0] * (K + 1) + [c * tau + 1.0 / (c * tau)]
    for j in range(1, len(p)):
        factor = (
            -2.0 * tau / (c ** 2)
            * f(
                alpha[eta[j] - 1] - alpha[zeta[j]],
                alpha[eta[j]] - alpha[max(zeta[j] - 1, 0)],
                c,
            )
        )
        out[p[j][1]] += factor
        out[p[j - 1][1]] -= factor
        if p[j][0] == "q":
            out[-1] += factor * c
        if p[j - 1][0] == "q":
            out[-1] -= factor * c

    return tuple(out)


from lp import solve_lp_with_certificate

def multioptimize(K: int, boundary, objectives, return_cert: bool = False):
    """
    Maximize min(objectives) subject to boundary constraints.
    Variables: w0, w1, ..., wK+1, z
    """
    n_vars = K + 2 + 1
    A_ub = []
    b_ub = []

    for (x, y, bound) in boundary:
        row = [0.0] * n_vars
        row[x] = -1.0
        row[y] = 1.0
        A_ub.append(row)
        b_ub.append(float(bound))

    for obj in objectives:
        row = [-float(coeff) for coeff in obj] + [1.0]
        A_ub.append(row)
        b_ub.append(0.0)

    A_eq = []
    row0 = [0.0] * n_vars
    row0[0] = 1.0
    A_eq.append(row0)
    rowK1 = [0.0] * n_vars
    rowK1[K+1] = 1.0
    A_eq.append(rowK1)
    b_eq = [0.0, 1.0]

    c_lp = [0.0] * (K + 2) + [-1.0]
    lp_bounds = [(0.0, 1.0)] * (K + 2) + [(None, None)]

    A_ub_np = np.array(A_ub, dtype=float)
    b_ub_np = np.array(b_ub, dtype=float)
    A_eq_np = np.array(A_eq, dtype=float)
    b_eq_np = np.array(b_eq, dtype=float)

    if return_cert:
        cert = solve_lp_with_certificate(c_lp, A_ub_np, b_ub_np, A_eq_np, b_eq_np, lp_bounds)
        if cert["success"]:
            return -cert["fun"], np.array(cert["x"][:-1]), 1, cert
        else:
            return 0.0, None, 1, cert
    else:
        res = scipy.optimize.linprog(
            c=c_lp, A_ub=A_ub_np, b_ub=b_ub_np, A_eq=A_eq_np, b_eq=b_eq_np, bounds=lp_bounds, method="highs"
        )
        if res.status == 0:
            return -float(res.fun), res.x[:-1], 1, None
        else:
            return 0.0, None, 1, None


@cache
def cell_intersection_nonempty(K: int, bound_ineqs):
    flat = []
    for bi in bound_ineqs:
        flat.extend(bi)
    return is_feasible(K, tuple(flat))


@cache
def is_valid_cell_pair(c1: float, c2: float, K: int, a: int, b: int):
    to_test = (
        tuple(s_to_boundary(feasible_cells(c1, K)[a], c1)),
        tuple(s_to_boundary(feasible_cells(c2, K)[b], c2)),
    )
    return cell_intersection_nonempty(K, to_test)


@cache
def _final_intersections_and_boundaries(cs: Tuple[float, ...], K: int):
    print(f"  [init] Calculating feasible cells for K={K}...")
    base_cells = []
    for i, c in enumerate(cs):
        cells = feasible_cells(c, K)
        base_cells.append(cells)
        print(f"  [init] c{i+1}={c} has {len(cells)} nonempty cells")

    # We store indices into base_cells instead of the cell tuples themselves to save massive memory
    cell_indices: List[Tuple[int, ...]] = [()]

    print(f"  [init] Crossing cells...")
    for i in range(len(base_cells)):
        new_cell_indices = []
        n_current = len(cell_indices)
        for cur_idx, cur_a in enumerate(cell_indices):
            for a, _ in enumerate(base_cells[i]):
                if not all(is_valid_cell_pair(cs[i], cs[j], K, a, b) for (j, b) in enumerate(cur_a)):
                    continue
                new_cell_indices.append(cur_a + (a,))
            
            if n_current > 50 and cur_idx % (n_current // 10 + 1) == 0:
                print(f"  [init]   Crossing layer {i+1}/{len(cs)}: {cur_idx}/{n_current} processed...")

        cell_indices = new_cell_indices
        print(f"  [init]   Layer {i+1} complete: {len(cell_indices)} candidates")

    final_cell_indices: List[Tuple[int, ...]] = []

    print(f"  [init] Final feasibility check on {len(cell_indices)} intersections...")
    n_total = len(cell_indices)
    for idx, indices in enumerate(cell_indices):
        if n_total > 50 and idx % (n_total // 10 + 1) == 0:
            print(f"  [init]   Progress: {idx}/{n_total} ...")
        
        # Reconstruct intersection for check
        cell_intersection = tuple(base_cells[i][idx_i] for i, idx_i in enumerate(indices))
        to_test = tuple(sorted(tuple(s_to_boundary(s, c)) for s, c in zip(cell_intersection, cs)))
        if cell_intersection_nonempty(K, to_test):
            final_cell_indices.append(indices)

    print(f"  [init] Finalization: {len(final_cell_indices)} verified intersections")
    return tuple(final_cell_indices), tuple(base_cells)


def process_intersection(args):
    idx, cell_intersection, boundary, K, cs, b1_c, tau, alpha, cutoff, return_cert = args
    
    objectives = [b1_c]
    for i, c in enumerate(cs):
        s = cell_intersection[i]
        objectives.append(s_to_b2_coeffs(s, tau, alpha, c))

    bound, x, _cases, cert = multioptimize(K, boundary, tuple(objectives), return_cert=return_cert)
    return bound, x, idx, cert


def combined_bound(
    tau: float,
    alpha: Tuple[float, ...],
    cs: Tuple[float, ...],
    printing: bool = False,
    cutoff: Optional[float] = None,
    probe_first_idx: Optional[int] = None,
    return_certs: bool = False,
):
    K = len(alpha) - 2
    # Memory-optimized: final_cell_indices is (tuple of indices), base_cells is (tuple of list of cell tuples)
    final_cell_indices, base_cells = _final_intersections_and_boundaries(tuple(cs), int(K))
    n_inter = len(final_cell_indices)

    if printing:
        print(f"There are {n_inter} nonempty cell intersections")

    worst_bound = 0.0
    at = None
    worst_idx = None
    b1_c = b1_coeffs(tau, alpha)
    all_certs = {} if return_certs else None

    # Helper to reconstruct boundary on the fly
    def get_cell_and_boundary(indices):
        cell_intersection = tuple(base_cells[i][idx_i] for i, idx_i in enumerate(indices))
        boundary = tuple(
            bound_ineq
            for s, c in zip(cell_intersection, cs)
            for bound_ineq in s_to_boundary(s, c)
        )
        return cell_intersection, boundary

    # 1. Probes
    probe_indices = []
    if probe_first_idx is not None:
        if isinstance(probe_first_idx, int):
            probe_indices = [probe_first_idx]
        elif isinstance(probe_first_idx, (list, tuple)):
            probe_indices = list(probe_first_idx)

    for pidx in probe_indices:
        if 0 <= pidx < n_inter:
            cell_int, boundary = get_cell_and_boundary(final_cell_indices[pidx])
            res = process_intersection((pidx, cell_int, boundary, K, cs, b1_c, tau, alpha, cutoff, return_certs))
            bound, x, _, cert = res
            if all_certs is not None: all_certs[pidx] = cert
            if cutoff is not None and bound > cutoff:
                return bound, x, 1, n_inter, pidx, all_certs
            if bound > worst_bound:
                worst_bound = bound
                at = x
                worst_idx = pidx

    # 2. Main loop in parallel
    print(f"  [eval] Calculating bounds for {n_inter} intersections using multi-core...")
    
    # We use a smaller worker function to avoid passing huge objects
    worker_args = []
    for idx in range(n_inter):
        if idx in probe_indices: continue
        worker_args.append(idx)

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    
    # Chunking is essential for 722k tasks
    chunk_size = max(1, n_inter // (multiprocessing.cpu_count() * 50))
    
    # 2. Main loop in parallel ONLY IF we are in the main process to avoid nesting
    is_main = (multiprocessing.current_process().name == 'MainProcess')
    
    if is_main:
        print(f"  [eval] Calculating bounds for {n_inter} intersections using multi-core...")
        with ProcessPoolExecutor(
            max_workers=multiprocessing.cpu_count(),
            initializer=init_worker,
            initargs=(final_cell_indices, base_cells)
        ) as executor:
            iterable = [(idx, K, cs, b1_c, tau, alpha, cutoff, return_certs) for idx in worker_args]
            results = executor.map(bound_worker, iterable, chunksize=chunk_size)
            
            for bound, x, idx, cert in results:
                if all_certs is not None: all_certs[idx] = cert
                if cutoff is not None and bound > cutoff:
                    print(f"  [eval] Early abort: bound {bound:.10f} > cutoff {cutoff:.10f}")
                    executor.shutdown(wait=False, cancel_futures=True)
                    return bound, x, 1, n_inter, idx, all_certs
                if bound > worst_bound:
                    worst_bound = bound
                    at = x
                    worst_idx = idx
    else:
        # We are already in a worker process, do it sequentially
        for idx in worker_args:
            indices = final_cell_indices[idx]
            cell_int, boundary = get_cell_and_boundary(indices)
            res = process_intersection((idx, cell_int, boundary, K, cs, b1_c, tau, alpha, cutoff, return_certs))
            bound, x, _, cert = res
            if all_certs is not None: all_certs[idx] = cert
            if cutoff is not None and bound > cutoff:
                return bound, x, 1, n_inter, idx, all_certs
            if bound > worst_bound:
                worst_bound = bound
                at = x
                worst_idx = idx

    return worst_bound, at, n_inter, n_inter, worst_idx, all_certs


# -----------------------------
# Public API
# -----------------------------

def run_bound(
    tau: float,
    alphas: Iterable[float],
    cs: Iterable[float],
    *,
    quiet: bool = True,
    cutoff: Optional[float] = None,
    probe_first_idx: Optional[int] = None,
    return_certs: bool = False,
) -> Dict[str, Any]:
    alphas = tuple(float(x) for x in alphas)
    alpha = tuple([0.0] + list(alphas) + [float("inf")])
    cs = tuple(float(c) for c in cs)

    b_inf, at, total_cases, n_inter, worst_idx, certs = combined_bound(
        float(tau),
        alpha,
        cs,
        printing=not quiet,
        cutoff=cutoff,
        probe_first_idx=probe_first_idx,
        return_certs=return_certs
    )
    w = tuple(at) if at is not None else None

    out: Dict[str, Any] = {
        "b_inf": float(b_inf),
        "worst_w": tuple(float(x) for x in w) if w is not None else None,
        "total_cases": int(total_cases),
        "n_intersections": int(n_inter),
        "worst_idx": worst_idx,
        "early_abort": cutoff is not None and b_inf > cutoff,
        "certificates": certs
    }

    if w is not None:
        out["lemma31"] = float(b1(float(tau), alpha, w))
        out["lemma32"] = [float(b2(float(tau), alpha, c, w)) for c in cs]
    else:
        out["lemma31"] = None
        out["lemma32"] = None

    return out


def run_theorem_3_3_with_params(
    K: int,
    tau: float,
    alphas: Iterable[float],
    cs: Iterable[float],
    *,
    quiet: bool = True,
    cutoff: Optional[float] = None,
    probe_first_idx: Optional[int] = None,
    return_certs: bool = False,
):
    alphas = tuple(float(x) for x in alphas)
    if len(alphas) != int(K):
        raise ValueError(f"K={K} but len(alphas)={len(alphas)}")
    return run_bound(
        tau=float(tau),
        alphas=alphas,
        cs=tuple(cs),
        quiet=quiet,
        cutoff=cutoff,
        probe_first_idx=probe_first_idx,
        return_certs=return_certs
    )


# -----------------------------
# Script entry point
# -----------------------------

def _main() -> None:
    start_time = time.time()

    # Theorem 2.1
    tau = 1.07950
    alphas = (0.72720, 1.31609)
    cs = (0.86838,)
    out = run_bound(tau, alphas, cs, quiet=True)
    print(f"Theorem 2.1 gives b_inf <= {out['b_inf']}\n")

    # Theorem 3.3
    print("Theorem 3.3 data:")
    tau = 1.12733
    alphas = (0.70749, 0.78822, 0.87175, 1.12464, 1.18020, 1.24610)
    cs = (0.66461, 0.67780, 0.71884)
    out = run_bound(tau, alphas, cs, quiet=False)

    print(f"\nTheorem 3.3 gives b_inf <= {out['b_inf']}")
    print(f"Worst case w is {out['worst_w']}")
    print(f"Worst intersection idx is {out['worst_idx']}")
    print(f"Lemma 3.1 at this w: {out['lemma31']}")
    for j, val in enumerate(out["lemma32"], start=1):
        print(f"Lemma 3.2 at this w with c=c{j}: {val}")

    print(f"Time taken: {time.time() - start_time} seconds")


if __name__ == "__main__":
    _main()
