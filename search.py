from __future__ import annotations

import json
import os
import time
import random
from typing import List, Dict, Any, Tuple

from params import CHOParams
from engine import compute_b_bound, load_params_json

from config import RECORD, RECORD_C, TOPK_PATH, BEST_PATH, LOG_PATH, AUTO_CERTIFY_THRESHOLD
from records import RecordStatus, create_discovery_record, append_to_ledger, get_record_id
from rounding import round_params
rng = random.Random(0)

def save_best(p: CHOParams, out: Dict[str, Any], path: str) -> None:
    # Always save with full certification-ready metadata
    data = {
        "K": p.K,
        "tau": p.tau,
        "alphas": list(p.alphas),
        "cs": list(p.cs),
        "b_inf": out.get("b_inf"),
        "c_record": out.get("b_inf", 0.0) / 2.0 if out.get("b_inf") else None,
        "gap_to_ref_record": RECORD_C - (out.get("b_inf", 0.0) / 2.0),
        "worst_w": out.get("worst_w"),
        "worst_idx": out.get("worst_idx"),
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lemma31": out.get("lemma31"),
        "lemma32": out.get("lemma32")
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_best_or_default(default_path: str = "params_cho.json", override_path: str = None) -> Tuple[CHOParams, float, Any]:
    # Returns (params, best_b, worst_idx)
    path = override_path if (override_path and os.path.exists(override_path)) else (BEST_PATH if os.path.exists(BEST_PATH) else default_path)
    print(f"Loading parameters from: {path}")
    with open(path, "r") as f:
        d = json.load(f)
    
    p = CHOParams(
        K=int(d["K"]),
        tau=float(d["tau"]),
        alphas=tuple(float(x) for x in d["alphas"]),
        cs=tuple(float(x) for x in d["cs"]),
    )
    p.validate()
    
    best_b = d.get("b_inf", 2.0) # Default to a high value if not present
    worst_idx = d.get("worst_idx", None)
    
    return p, best_b, worst_idx


def enforce_alpha_strict(alphas: List[float], eps: float = 1e-6) -> Tuple[float, ...]:
    alphas = sorted(alphas)
    for i in range(1, len(alphas)):
        if alphas[i] <= alphas[i - 1] + eps:
            alphas[i] = alphas[i - 1] + eps
    return tuple(alphas)


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def append_log(rec: Dict[str, Any]) -> None:
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()


def update_topk(topk: List[Dict[str, Any]], cand_rec: Dict[str, Any], k: int = 20) -> List[Dict[str, Any]]:
    topk.append(cand_rec)
    topk.sort(key=lambda r: r["b_inf"])
    topk = topk[:k]
    with open(TOPK_PATH, "w") as f:
        json.dump(topk, f, indent=2)
    return topk


def propose_batch(
    best: CHOParams,
    step_tau: float,
    step_a: float,
    step_c: float,
    n_rand: int,
    allow_cs_moves: bool,
    jitter_scale: float = 1.0,
) -> List[CHOParams]:
    """
    Batch = coordinate +/- steps + random jitters around best.
    If allow_cs_moves=False, cs is frozen (critical for speed).
    Includes micro-jitters (very small sigma) to shave tiny improvements.
    """
    cands: List[CHOParams] = []

    # Coordinate: tau +/- step
    for sgn in (-1.0, 1.0):
        cands.append(CHOParams(best.K, best.tau + sgn * step_tau, best.alphas, best.cs))

    # Coordinate: each alpha_i +/- step
    for i in range(best.K):
        for sgn in (-1.0, 1.0):
            al = list(best.alphas)
            al[i] += sgn * step_a
            al = list(enforce_alpha_strict(al))
            cands.append(CHOParams(best.K, best.tau, tuple(al), best.cs))

    # Coordinate: cs +/- step (rare, expensive)
    if allow_cs_moves and step_c > 0:
        for j in range(len(best.cs)):
            for sgn in (-1.0, 1.0):
                cs = list(best.cs)
                cs[j] = clamp(cs[j] + sgn * step_c, 0.05, 0.95)
                cands.append(CHOParams(best.K, best.tau, best.alphas, tuple(cs)))

    # Random jitters (moderate)
    for _ in range(n_rand):
        tau = best.tau + rng.gauss(0, step_tau * 2.0 * jitter_scale)

        if allow_cs_moves and step_c > 0:
            cs = [clamp(c + rng.gauss(0, step_c * 2.0 * jitter_scale), 0.05, 0.95) for c in best.cs]
        else:
            cs = list(best.cs)

        al = [a + rng.gauss(0, step_a * 2.0 * jitter_scale) for a in best.alphas]
        al = list(enforce_alpha_strict(al))

        cands.append(CHOParams(best.K, float(tau), tuple(al), tuple(cs)))

    # Extra micro-jitters (high chance to find tiny improvements)
    # cs is frozen here; cs moves handled elsewhere (rare rounds)
    micro_scale = max(0.1, 0.2 * jitter_scale)
    for _ in range(8):
        tau = best.tau + rng.gauss(0, step_tau * micro_scale)
        cs = list(best.cs)
        al = [a + rng.gauss(0, step_a * micro_scale) for a in best.alphas]
        al = list(enforce_alpha_strict(al))
        cands.append(CHOParams(best.K, float(tau), tuple(al), tuple(cs)))

    # validate + de-duplicate
    uniq: Dict[Tuple[float, Tuple[float, ...], Tuple[float, ...]], CHOParams] = {}
    for p in cands:
        try:
            p.validate()
        except Exception:
            continue
        key = (round(p.tau, 12), tuple(round(x, 12) for x in p.alphas), tuple(round(x, 12) for x in p.cs))
        uniq[key] = p
    return list(uniq.values())


import concurrent.futures

def evaluate_candidate(p: CHOParams, best_b: float, baseline_worst_idx: Any) -> Tuple[CHOParams, Dict[str, Any]]:
    # We call verify.run_theorem_3_3_with_params via compute_b_bound
    # Since verify is now single-threaded, we can run many in parallel.
    out = compute_b_bound(p, quiet=True, cutoff=best_b, probe_first_idx=baseline_worst_idx)
    return p, out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Input parameter file")
    args = parser.parse_args()

    default_in = "params_cho.json"
    best, best_b_from_file, baseline_worst_idx = load_best_or_default(default_in, override_path=args.input)

    topk: List[Dict[str, Any]] = []
    if os.path.exists(TOPK_PATH):
        try:
            topk = json.load(open(TOPK_PATH, "r"))
        except Exception:
            topk = []

    # FRESH EVALUATION for stable baseline
    print("Initializing engine and verifying current state...")
    t0 = time.time()
    out0 = compute_b_bound(best, quiet=True, cutoff=None, probe_first_idx=baseline_worst_idx)
    best_b = out0["b_inf"]
    baseline_worst_idx = out0.get("worst_idx", None)

    print(f"START best_b={best_b:.16f} (C={best_b/2:.16f})")
    print(f"  REFERENCE CHO RECORD C={RECORD_C:.16f}")
    print(f"  CURRENT GAP TO RECORD: {RECORD_C - (best_b/2):.10f}")
    
    # Multi-temperature search parameters
    REFINE_STEPS = dict(tau=2e-4, a=2e-4, c=2e-4)
    EXPLORE_STEPS = dict(tau=4e-3, a=4e-3, c=4e-3)
    BOLD_EXPLORE_STEPS = dict(tau=1e-2, a=1e-2, c=1e-2)
    GRAND_EXPLORE_STEPS = dict(tau=5e-2, a=5e-2, c=5e-2) # For "jumping" to new basins

    refine_scale = 1.0
    evals = 0
    early = 0
    full = 0
    last_beat = time.time()
    last_improvement_round = 0

    round_idx = 0

    # Using ProcessPoolExecutor for parallel candidate evaluation
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        while True:
            round_idx += 1

            jitter_scale = 1.0
            stagnation = round_idx - last_improvement_round

            if round_idx % 100 == 0:
                mode = "GRAND_EXPLORE"
                step_tau, step_a, step_c = GRAND_EXPLORE_STEPS["tau"], GRAND_EXPLORE_STEPS["a"], GRAND_EXPLORE_STEPS["c"]
                n_rand = 30
                allow_cs_moves = True
                jitter_scale = 1.8
            elif round_idx % 30 == 0:
                mode = "EXPLORE"
                step_tau, step_a, step_c = EXPLORE_STEPS["tau"], EXPLORE_STEPS["a"], EXPLORE_STEPS["c"]
                n_rand = 20
                allow_cs_moves = True
                jitter_scale = 1.2
            else:
                mode = "refine"
                step_tau = REFINE_STEPS["tau"] * refine_scale
                step_a = REFINE_STEPS["a"] * refine_scale
                step_c = REFINE_STEPS["c"] * refine_scale
                n_rand = 12
                allow_cs_moves = (round_idx % 10 == 0)

            # Stagnation-aware overrides
            if stagnation >= 150:
                mode = "GRAND_EXPLORE"
                step_tau, step_a, step_c = GRAND_EXPLORE_STEPS["tau"], GRAND_EXPLORE_STEPS["a"], GRAND_EXPLORE_STEPS["c"]
                n_rand = 40
                allow_cs_moves = True
                jitter_scale = 2.5
            elif stagnation >= 80 and mode != "GRAND_EXPLORE":
                mode = "BOLD_EXPLORE"
                step_tau, step_a, step_c = BOLD_EXPLORE_STEPS["tau"], BOLD_EXPLORE_STEPS["a"], BOLD_EXPLORE_STEPS["c"]
                n_rand = 28
                allow_cs_moves = True
                jitter_scale = 1.8
            elif stagnation >= 40 and mode == "refine":
                mode = "refine"
                jitter_scale = 1.4
                allow_cs_moves = True

            cands = propose_batch(best, step_tau, step_a, step_c, n_rand=n_rand, allow_cs_moves=allow_cs_moves, jitter_scale=jitter_scale)

            improved_this_round = False
            
            # Submit batch
            futures = [executor.submit(evaluate_candidate, p, best_b, baseline_worst_idx) for p in cands]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    p, out = future.result()
                except Exception as e:
                    print(f"Worker error: {e}")
                    continue
                    
                evals += 1
                if out.get("early_abort", False):
                    early += 1
                else:
                    full += 1

                rec = {
                    "ts": time.time(),
                    "round": round_idx,
                    "mode": mode,
                    "allow_cs_moves": allow_cs_moves,
                    "b_inf": out["b_inf"],
                    "c_record": out["b_inf"] / 2.0,
                    "early_abort": out.get("early_abort", False),
                    "tau": p.tau,
                    "alphas": list(p.alphas),
                    "cs": list(p.cs),
                    "steps": {"tau": step_tau, "a": step_a, "c": step_c},
                }
                append_log(rec)
                topk = update_topk(topk, rec, k=20)

                if out["b_inf"] < best_b:
                    improved_this_round = True
                    prev_b = best_b
                    best_b = out["b_inf"]
                    best = p
                    last_improvement_round = round_idx
                    
                    # 1. Discovery Record
                    rec_entry = create_discovery_record(best, out)
                    rid = rec_entry["record_id"]
                    
                    # 2. Update local best_found.json for engine baseline
                    save_best(best, out, BEST_PATH)
                    baseline_worst_idx = out.get("worst_idx", baseline_worst_idx)

                    elapsed = time.time() - t0
                    improvement_C = (prev_b - best_b) / 2.0
                    total_lead_CHO = RECORD_C - (best_b / 2.0)

                    print(f"\n[{round_idx}] NEW DISCOVERY record {rid} ({mode})")
                    print(f"  b_inf       = {best_b:.16f}")
                    print(f"  C record    = {best_b/2:.16f}")
                    print(f"  vs Prev Best: -{improvement_C:.12f}")
                    print(f"  vs CHO Ref:   +{total_lead_CHO:.12f}")
                    
                    # 3. Check for Candidate Status
                    current_gap = RECORD_C - (best_b/2.0)
                    if current_gap > AUTO_CERTIFY_THRESHOLD:
                        print(f"  >>> THRESHOLD CROSSED ({AUTO_CERTIFY_THRESHOLD}). Marking as CANDIDATE_CLEAN.")
                        tau_r, alphas_r, cs_r = round_params(best)
                        cand_entry = rec_entry.copy()
                        cand_entry["status"] = RecordStatus.CANDIDATE_CLEAN
                        cand_entry["tau"] = tau_r
                        cand_entry["alphas"] = list(alphas_r)
                        cand_entry["cs"] = list(cs_r)
                        cand_entry["record_id"] = f"CAND_{rid}"
                        append_to_ledger(cand_entry)
                        print(f"  >>> Candidate clean record saved: {cand_entry['record_id']}")

                    print("-" * 30)

                    if mode == "refine":
                        refine_scale = max(0.20, refine_scale * 0.90)

            # Heartbeat every ~60s
            now = time.time()
            if now - last_beat >= 60:
                elapsed = now - t0
                rate = evals / elapsed * 3600.0
                print(
                    f"[hb] {elapsed/3600:.2f}h evals={evals} rate={rate:.1f}/h "
                    f"early={early} full={full} best_C={best_b/2:.10f} "
                    f"mode={mode} "
                    f"refine_scale={refine_scale:.3f}"
                )
                last_beat = now

            if not improved_this_round and (mode == "refine"):
                refine_scale = min(2.0, refine_scale * 1.02)


if __name__ == "__main__":
    main()
