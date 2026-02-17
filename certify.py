
import json
import os
import time
import shutil
import platform
import sys
from typing import Dict, Any
from config import CERTIFIED_DIR, RECORDS_DIR, LEDGER_PATH
from rounding import round_params
from engine import compute_b_bound
from params import CHOParams
from records import RecordStatus, append_to_ledger, get_record_id

def certify_candidate(p_orig: CHOParams, record_id: str):
    print(f"--- Certification of Record {record_id} ---")
    
    # 1. Rounding
    tau_r, alphas_r, cs_r = round_params(p_orig)
    p_rounded = CHOParams(p_orig.K, tau_r, alphas_r, cs_r)
    
    # 2. Preparation
    cert_path = os.path.join(CERTIFIED_DIR, f"record_{record_id}")
    os.makedirs(cert_path, exist_ok=True)
    
    # 3. Full rigorous verification
    print("Running exhaustive verification with dual certificate capture...")
    import verify
    out = verify.run_theorem_3_3_with_params(
        p_rounded.K, p_rounded.tau, p_rounded.alphas, p_rounded.cs,
        quiet=False, cutoff=None, return_certs=True
    )
    
    # 4. Save certificates
    certs = out.pop("certificates")
    certs_dir = os.path.join(cert_path, "certificates")
    os.makedirs(certs_dir, exist_ok=True)
    
    print(f"Exporting {len(certs)} LP certificates...")
    for idx, cert_data in certs.items():
        with open(os.path.join(certs_dir, f"lp_{idx}.json"), "w") as f:
            json.dump(cert_data, f)
            
    # 5. Save certified record metadata
    entry = {
        "record_id": record_id,
        "K": p_rounded.K,
        "tau": p_rounded.tau,
        "alphas": list(p_rounded.alphas),
        "cs": list(p_rounded.cs),
        "b_inf": out["b_inf"],
        "c_record": out["b_inf"] / 2.0,
        "status": RecordStatus.CERTIFIED,
        "worst_idx": out.get("worst_idx"),
        "witness_w": out.get("worst_w"),
        "lemma31": out.get("lemma31"),
        "lemma32": out.get("lemma32"),
        "total_cases": out.get("total_cases"),
        "artifact_path": cert_path
    }
    
    with open(os.path.join(cert_path, "certified_record.json"), "w") as f:
        json.dump(entry, f, indent=2)
    
    # Update ledger
    append_to_ledger(entry)
    print(f"CERTIFICATION COMPLETE. Record {record_id} is now CERTIFIED.")
    print(f"Certified bound C = {entry['c_record']:.10f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to best_found.json or similar")
    args = parser.parse_args()
    
    from engine import load_params_json
    p = load_params_json(args.input)
    rid = get_record_id({"tau": p.tau, "alphas": p.alphas, "cs": p.cs})
    certify_candidate(p, rid)
