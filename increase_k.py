
import json
import numpy as np
from params import CHOParams

def interpolate_alphas(old_alphas: tuple, new_k: int) -> tuple:
    old_k = len(old_alphas)
    # Old x-grid: 1 to old_k
    # New x-grid: 1 to new_k
    x_old = np.linspace(0, 1, old_k + 2)
    y_old = np.array([0.0] + list(old_alphas) + [0.0]) # Just a dummy for interpolation
    # Actually, let's just use linear interp per index
    # We'll just spread them out.
    new_alphas = np.interp(
        np.linspace(1, old_k, new_k),
        np.arange(1, old_k + 1),
        old_alphas
    )
    return tuple(new_alphas)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="best_found.json")
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()
    
    with open(args.input) as f:
        data = json.load(f)
    
    old_alphas = tuple(data["alphas"])
    new_alphas = interpolate_alphas(old_alphas, args.k)
    
    new_data = {
        "K": args.k,
        "tau": data["tau"],
        "alphas": list(new_alphas),
        "cs": data["cs"]
    }
    
    out_path = f"params_K{args.k}.json"
    with open(out_path, "w") as f:
        json.dump(new_data, f, indent=2)
    print(f"Generated K={args.k} parameters in {out_path}")

if __name__ == "__main__":
    main()
