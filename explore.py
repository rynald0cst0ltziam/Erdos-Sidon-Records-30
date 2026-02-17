from __future__ import annotations
import argparse
import time

from engine import load_params_json, compute_b_bound


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    p = load_params_json(args.params)

    t0 = time.time()
    out = compute_b_bound(p, quiet=not args.verbose)
    dt = time.time() - t0

    b = out["b_inf"]
    print(f"b_inf = {b:.16f}")
    print(f"c = b/2 = {b/2:.16f}")
    print(f"time = {dt:.3f}s")
    print("worst_w =", out["worst_w"])
    print("lemma31 =", out["lemma31"])
    print("lemma32 =", out["lemma32"])


if __name__ == "__main__":
    main()
