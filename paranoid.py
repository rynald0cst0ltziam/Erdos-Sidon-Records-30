
import subprocess
import json
import sys
import os
import time

def run_verify_process(params_path):
    cmd = [
        sys.executable, "-c",
        f"import json; from engine import load_params_json, compute_b_bound; p = load_params_json('{params_path}'); print(json.dumps(compute_b_bound(p)))"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return None
    try:
        return json.loads(res.stdout.splitlines()[-1])
    except:
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to best_found.json")
    args = parser.parse_args()
    
    print(f"--- Paranoid Stress Test ---")
    results = []
    for i in range(3):
        print(f"  Run {i+1}/3...")
        start = time.time()
        out = run_verify_process(args.input)
        dt = time.time() - start
        if out:
            results.append({"b_inf": out["b_inf"], "time": dt})
            print(f"    b_inf: {out['b_inf']:.16f} ({dt:.2f}s)")
        else:
            print("    FAILED.")
            
    if len(results) >= 2:
        vals = [r["b_inf"] for r in results]
        spread = max(vals) - min(vals)
        print(f"\n  Final Spread: {spread:g}")
        if spread < 1e-12:
            print("  PARANOID CONSISTENCY: CONFIRMED")
        else:
            print("  PARANOID CONSISTENCY: FAILED (High spread)")
    else:
        print("  Insufficient valid runs.")

if __name__ == "__main__":
    main()
