
import json
import os
import hashlib
import datetime
from typing import Dict, Any, List
from config import LEDGER_PATH, TOPK_PATH
from envinfo import get_env_info

class RecordStatus:
    DISCOVERY_ONLY = "DISCOVERY_ONLY"
    CANDIDATE_CLEAN = "CANDIDATE_CLEAN"
    CERTIFYING = "CERTIFYING"
    CERTIFIED = "CERTIFIED"
    REJECTED = "REJECTED"

def get_record_id(params: Dict[str, Any]) -> str:
    # Stable ID based on params
    s = json.dumps(params, sort_keys=True)
    h = hashlib.sha256(s.encode()).hexdigest()[:12]
    ts = int(os.environ.get("FORCE_TS", 0)) or int(datetime.datetime.now().timestamp())
    return f"{ts}_{h}"

def append_to_ledger(entry: Dict[str, Any]):
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    entry["env"] = get_env_info()
    with open(LEDGER_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

def update_top_k(k: int = 10):
    if not os.path.exists(LEDGER_PATH):
        return
    
    records = []
    with open(LEDGER_PATH, "r") as f:
        for line in f:
            records.append(json.loads(line))
    
    # Sort by b_inf (ascending is better for these bounds)
    records.sort(key=lambda r: r.get("b_inf", 2.0))
    
    # Filter unique per params to avoid cluttering top-k with minor updates
    unique_records = {}
    for r in records:
        key = f"{r['K']}_{r['tau']}_{tuple(r['alphas'])}_{tuple(r['cs'])}"
        if key not in unique_records or r['b_inf'] < unique_records[key]['b_inf']:
            unique_records[key] = r
            
    top_k_list = sorted(unique_records.values(), key=lambda r: r['b_inf'])[:k]
    
    with open(TOPK_PATH, "w") as f:
        json.dump(top_k_list, f, indent=2)

def create_discovery_record(p, out) -> Dict[str, Any]:
    entry = {
        "record_id": get_record_id({"tau": p.tau, "alphas": p.alphas, "cs": p.cs}),
        "K": p.K,
        "tau": p.tau,
        "alphas": list(p.alphas),
        "cs": list(p.cs),
        "b_inf": out["b_inf"],
        "c_record": out["b_inf"] / 2.0,
        "status": RecordStatus.DISCOVERY_ONLY,
        "worst_idx": out.get("worst_idx"),
        "witness_w": out.get("worst_w")
    }
    append_to_ledger(entry)
    update_top_k()
    return entry
